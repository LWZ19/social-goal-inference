"""The goal step of the constraint method: the people in view get numbers, a model states constraints over them,
and the solver finds the place.

Design: docs/superpowers/specs/2026-10-09-constraint-goal-inference-design.md, sections 2, 5, 6 and 8. The
inferrer stands where the baseline's does, behind the same interface, so the two are run on the same episodes. It
is built from a people source: the `vlm` source of `sgi.stages.people`, the harness's oracle, or a detector.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np

from sgi.config import PROMPTS_DIR, Strategy, Thresholds
from sgi.geometry import in_image, project
from sgi.imaging import draw_people_numbers, draw_target_marker, encode_png
from sgi.inputs import ObservationData
from sgi.prompts import build_constraints_request, people_lines
from sgi.schemas import constraints_from_wire, constraints_schema
from sgi.solver import solve
from sgi.stages.goal import standard_goal
from sgi.stages.people import PeopleSource, number_people
from sgi.types import Constraint, ContextCheck, DetectedPerson, Flags, GoalResult, GroundedTarget, Subtask
from sgi.vlm.base import VLMClient, to_call

_SOURCE = "constraints"                            # what a result of this method gives as its source
_NO_CALL = "none"                                  # the constraints mode in which no model is asked: the solver alone


def numbered_image(observation: ObservationData, target: GroundedTarget, people: Sequence[DetectedPerson]) -> np.ndarray:
    """The image the constraints step is shown (the design, section 5): the camera image with the target ringed
    as in the baseline, and each person's number in a disc on their body point."""
    marked = draw_target_marker(observation.rgb, *target.pixel)
    return draw_people_numbers(marked, people, observation.camera.T_world_camera[:2, 3])


class ConstraintInferrer:
    """The constraint method of the design's section 2, a `GoalInferrer`.

    `people_source` finds the people of the observation. They are merged, cut to `th.max_people` and numbered
    (`number_people`), and the numbers are drawn on them. One model call, stage `constraints`, then states what
    they require of the place, and the solver finds the place on the floor that depth shows, with the standard
    goal to tell equal places apart. With `constraints_mode` "none" no model is asked, `client` is not used, and
    the solver runs on what it applies by itself.

    A result is `ok` with `source` "constraints" and carries the people, what the solver made of each constraint,
    the runners-up and the solver's note. An answer with no usable constraint is no failure: the solver runs all
    the same, and `no_usable_constraint` says so. A people call or a constraints call that brought no valid
    answer ends the result with the status of that call and no pose, and with what was known by then.
    `goal_pixel` is the place as the picture shows it, None where it falls outside the picture. The calls of the
    people source come before the constraints call. In `timings_s` the calls count under `goal`, and placing the
    people and the solver under `geometry`; the people source's own two times are added to them.
    """

    def __init__(self, people_source: PeopleSource, client: VLMClient | None, strategy: Strategy, th: Thresholds,
                 max_image_side: int | None = None, trial: int = 0, constraints_mode: str = "vlm",
                 prompts_dir: Path = PROMPTS_DIR):
        self.people_source = people_source
        self.client = client
        self.strategy = strategy
        self.th = th
        self.max_image_side = max_image_side
        self.trial = trial
        self.constraints_mode = constraints_mode
        self.prompts_dir = prompts_dir

    def infer(self, observation: ObservationData, request: str, subtask: Subtask, target: GroundedTarget,
              context_check: ContextCheck | None) -> GoalResult:
        th = self.th
        sourced = self.people_source(observation)
        calls = list(sourced.calls)
        timings = {"goal": sourced.call_time_s, "geometry": sourced.geometry_time_s}
        shared = {"source": _SOURCE, "target": target, "calls": calls, "timings_s": timings}
        unanswered = next((call.status for call in calls if call.status != "ok"), None)
        if unanswered is not None:                 # the people are not known, which is not nobody in view
            return GoalResult(status=unanswered, **shared)

        started = time.perf_counter()
        people, people_cut = number_people(sourced.found, target, observation, th, th.max_people)
        timings["geometry"] += time.perf_counter() - started

        constraints: list[Constraint] = []
        if self.constraints_mode != _NO_CALL:
            entered = time.perf_counter()
            image = encode_png(numbered_image(observation, target, people), self.max_image_side)
            said = None if context_check is None else context_check.relevant_people
            system, parts = build_constraints_request(self.strategy, request, subtask, image,
                                                      people_lines(people, target, observation, th), said,
                                                      th.standoff_m, th.max_constraints, prompts_dir=self.prompts_dir)
            asked = time.perf_counter()
            resp = self.client.generate(parts, constraints_schema(self.strategy), system=system, trial=self.trial)
            answered = time.perf_counter()
            calls.append(to_call("constraints", resp))
            if resp.status == "ok":
                constraints = constraints_from_wire(resp.parsed).constraints
            # This call's time, with the response's own latency in place of the time `generate` took.
            timings["goal"] += (time.perf_counter() - entered) - (answered - asked) + resp.latency_total_s
            if resp.status != "ok":
                return GoalResult(status=resp.status, people=people, flags=Flags(people_cut=people_cut), **shared)

        started = time.perf_counter()
        solved = solve(observation, target, people, constraints, standard_goal(observation, target, th), th)
        shown = project(observation.camera, (*solved.pose_world[:2], 0.0))
        in_picture = shown is not None and in_image(observation.camera, *shown[:2])
        timings["geometry"] += time.perf_counter() - started
        return GoalResult(status="ok", pose_world=solved.pose_world, pose_robot=solved.pose_robot,
                          goal_pixel=shown[:2] if in_picture else None,
                          flags=solved.flags.model_copy(update={"people_cut": people_cut}), people=people,
                          constraints=solved.uses, candidates=solved.candidates, solver=solved.note, **shared)
