"""The people of the constraint method: who is in view, and the numbers they are given.

docs/constraint_method.md describes the method. A people source finds
the people of one observation, each with a floor position, a heading where it can be told and a body point in the
image. The `vlm` source is here: one model call on the camera image. The `oracle` source reads ground truth and
lives in the run harness. `number_people` then merges, cuts and numbers what either of them found.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from sgi.config import PROMPTS_DIR, Strategy, Thresholds
from sgi.geometry import bearing, wrap_angle
from sgi.imaging import encode_png
from sgi.inputs import ObservationData
from sgi.prompts import build_people_request
from sgi.schemas import FACINGS, PersonEntry, people_from_wire, people_schema
from sgi.stages.goal import _lift_point            # the path every point of a model takes to the world
from sgi.types import Call, DetectedPerson, GroundedTarget
from sgi.vlm.base import VLMClient, to_call

_EPS = 1e-9                                        # float noise where a distance is held against a radius: no threshold
# How far each of the first eight facings turns the bearing from a person to the camera,
# anticlockwise seen from above, in degrees. The ninth, unclear, gives no heading.
_TURN_DEG = dict(zip(FACINGS, (0, 45, 90, 135, 180, -135, -90, -45)))


@dataclass(frozen=True)
class FoundPerson:
    """A person as a people source found them: a DetectedPerson that has no number yet."""
    xy: tuple[float, float]                        # the centre of the body on the floor, world frame
    yaw: float | None                              # where they face, world frame; None where it cannot be told
    pixel: tuple[float, float]                     # the body point in the image
    label: str | None                              # a few words that pick the person out; None from ground truth


@dataclass(frozen=True)
class PeopleFound:
    """What a people source returns for one observation.

    A call that brought no valid answer leaves `found` empty, and its status in `calls` says so. That is not
    nobody in view: the goal step then ends with that status and no pose."""
    found: list[FoundPerson]                       # in the order found
    calls: list[Call]                              # the model calls made, none by a source that asks no model
    call_time_s: float                             # the stage time of those calls: counted under `goal`
    geometry_time_s: float                         # the time taken to place the people: counted under `geometry`


class PeopleSource(Protocol):
    """Whatever finds the people of an observation: the `vlm` source below, the harness's oracle, or a detector."""

    def __call__(self, observation: ObservationData) -> PeopleFound: ...


def _apart(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _centre(body_xy: np.ndarray, camera_xy: np.ndarray, th: Thresholds) -> np.ndarray:
    """The centre of a body from the ground under its body point: `person_radius_m` farther along the ground line
    from the camera through that point, which lies on the side of the body the camera sees."""
    away = body_xy - camera_xy
    length = float(np.hypot(*away))
    return body_xy + th.person_radius_m * away / length if length > 0 else body_xy


def _heading(centre: np.ndarray, faced: np.ndarray | None, facing: str, camera_xy: np.ndarray,
             th: Thresholds) -> float | None:
    """Where a person faces, in the world frame. The bearing from their centre to the ground under the lifted faced
    point, when there is one at least `person_radius_m` from the centre. Otherwise the bearing from the centre to
    the camera, turned by the facing's row of the table. None for a facing that is unclear."""
    if faced is not None and _apart(centre, faced[:2]) >= th.person_radius_m - _EPS:
        return bearing(centre, faced[:2])
    if facing not in _TURN_DEG:
        return None
    return wrap_angle(bearing(centre, camera_xy) + math.radians(_TURN_DEG[facing]))


def _place(entry: PersonEntry, observation: ObservationData, th: Thresholds) -> FoundPerson | None:
    """One entry of the answer as a person on the floor; None when its body point does not lift, or lifts within
    `floor_tolerance_m` of the ground, where no body is."""
    pixel, body = _lift_point(observation, entry.body_point, th)
    if body is None or abs(body[2]) <= th.floor_tolerance_m:
        return None
    camera_xy = observation.camera.T_world_camera[:2, 3]
    centre = _centre(body[:2], camera_xy, th)
    faced = None if entry.faced_point is None else _lift_point(observation, entry.faced_point, th)[1]
    yaw = _heading(centre, faced, entry.facing, camera_xy, th)
    return FoundPerson(xy=(float(centre[0]), float(centre[1])), yaw=yaw, pixel=pixel, label=entry.appearance)


class VlmPeople:
    """The `vlm` people source: one model call, stage `people`, on the camera image as it is.

    Nothing of an episode goes into the request, so the episodes of a scene share one cached answer. The answer
    is asked to list at most `th.max_people`; the bound itself is the cut of `number_people`."""

    def __init__(self, client: VLMClient, strategy: Strategy, th: Thresholds, max_image_side: int | None = None,
                 trial: int = 0, prompts_dir: Path = PROMPTS_DIR):
        self.client = client
        self.strategy = strategy
        self.th = th
        self.max_image_side = max_image_side
        self.trial = trial
        self.prompts_dir = prompts_dir

    def __call__(self, observation: ObservationData) -> PeopleFound:
        entered = time.perf_counter()
        image = encode_png(observation.rgb, self.max_image_side)
        pf, size = self.client.entry.point_format, (image.width, image.height)
        system, parts = build_people_request(self.strategy, image, pf, self.th.max_people,
                                             prompts_dir=self.prompts_dir)
        asked = time.perf_counter()
        resp = self.client.generate(parts, people_schema(self.strategy, pf, size), system=system, trial=self.trial)
        answered = time.perf_counter()
        calls = [to_call("people", resp)]

        def call_time(end: float) -> float:
            """The call's time up to `end`, with the response's own latency in place of the time `generate` took."""
            return (end - entered) - (answered - asked) + resp.latency_total_s

        if resp.status != "ok":
            return PeopleFound(found=[], calls=calls, call_time_s=call_time(answered), geometry_time_s=0.0)
        answer = people_from_wire(resp.parsed, pf, size)
        read = time.perf_counter()
        placed = [_place(entry, observation, self.th) for entry in answer.people]
        found = [person for person in placed if person is not None]
        return PeopleFound(found=found, calls=calls, call_time_s=call_time(read),
                           geometry_time_s=time.perf_counter() - read)


def number_people(found: Sequence[FoundPerson], target: GroundedTarget, observation: ObservationData, th: Thresholds,
                  max_people: int) -> tuple[list[DetectedPerson], bool]:
    """The people the constraint method works with, in the order of their numbers, and whether any was cut.

    The three steps, the same for every source and in this order. Merge: in the order
    found, a person whose centre is within `person_radius_m` of one already kept is dropped. Cut: the `max_people`
    nearest the ground under the target point are kept, the one found first of two as near; the flag says that
    someone was dropped here. Number: from 1, from the left of the image by the column of `pixel`; in one column,
    the nearer the camera's ground position first.
    """
    kept: list[FoundPerson] = []
    for person in found:
        if all(_apart(person.xy, other.xy) > th.person_radius_m + _EPS for other in kept):
            kept.append(person)
    nearest = sorted(kept, key=lambda person: _apart(person.xy, target.world[:2]))[:max_people]
    camera_xy = observation.camera.T_world_camera[:2, 3]
    ordered = sorted(nearest, key=lambda person: (person.pixel[0], _apart(person.xy, camera_xy)))
    people = [DetectedPerson(number=number, xy=person.xy, yaw=person.yaw, pixel=person.pixel, label=person.label)
              for number, person in enumerate(ordered, 1)]
    return people, len(kept) > max_people
