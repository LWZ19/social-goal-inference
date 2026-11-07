"""The pipeline as a local HTTP service, for a simulator that runs under another Python (docs/habitat.md).

    python -m sgi.serve [--model ENTRY] [--stage-model STAGE=ENTRY ...] [--strategy STAGE=NAME ...]
                        [--goal constraints|direct_marking] [--host HOST] [--port PORT]

One request is one live observation and one request of a person; the answer is the goal pose of the plan's
first navigation subtask, in the world frame of the observation and in the robot frame. Nothing is scored and
no ground truth is read: the service is given what a robot has, and nothing else.

    GET  /health   {"ok": true, "goal": ..., "models": {stage: entry}}
    POST /goal     the body below; the answer below
    POST /resolve  a stored answer of the constraint method solved again on tracked people (below)
    POST /plan     the plan call alone and its target's world point, ahead of /goal on the same body

The body is JSON. `rgb_png` is a PNG, `depth_npy` an .npy file of an (H, W) float32 array of camera Z in metres,
each in base64. `camera` holds fx, fy, cx, cy, width, height and T_world_camera, a 4 x 4 list that maps optical
camera coordinates (x right, y down, z forward) into the world frame, in which z is up and the floor is z = 0.
`robot` holds x, y and yaw in that frame. `context` may be null.

The answer holds `status` (ok, or why there is no goal), `plan_status`, the `subtasks` of the plan as
[type, description] pairs, `subtask_index` of the one the goal is for, `source` (standard, direct_marking or
constraints), `pose_world` and `pose_robot` as [x, y, theta] or null, `target_world` as [x, y, z] or null, the
`calls` made as [stage, status] pairs and `timings_s`. Under `inference` is what the stages worked out, for a
viewer to show: the `plan` (its reasoning and subtasks, each with the target it names and its point), and, once
the goal stage ran, the `target` (its 0..1000 point, its pixel and its world point), the `context_check` (the
people it names, whether they matter, and why), the `goal_mark` and `goal_pixel` of the baseline, the `people`
found and the `constraints` stated with what the solver made of each, the runner-up `candidates`, the `solver`'s
note and the `flags` that are set; and each call's `stage`, `status`, `model`, `latency_s`, `cached` and `error`.
The models' raw text is left out. A body that cannot be read is answered with status 400 and {"error": ...}.

An answer of the constraint method that placed a goal also holds a `session` id (else null). A /resolve body is an
observation as above, without `request`, plus `session`, `t` (seconds, the caller's clock), `tracks` (a list of
{"id", "xy": [x, y], "yaw": radians or null}, world frame, as the people stand now) and, on the first /resolve of
a session, `tracks_at_answer` (the same, as they stood when the answer was asked). The stored answer's constraints
are solved again on the tracked people and the new observation, with no model call (`sgi.resolve` has the rules).
The answer holds `status`, `pose_world`, `pose_robot`, `in_turn` (whether a `come_after` placed it), `joined`,
`dropped` (the numbers taken out of a `come_after`), `match_rate` and `people` (how many were solved with), and
`timings_s`. The tracks are what a tracker gives the robot; /goal never reads them. The floor is that of the view
the answer was asked on, with the people who stood in it then painted out (`sgi.resolve.paint_out`), unless the
first /resolve gives `floor_map`, the robot's navigation map ({"origin": [x, y], "resolution": m, "png": base64,
non-zero where the robot may stand, rows along y}), which is then kept and used round the robot's current pose.
The new observation gives the robot's pose. `in_turn` holds when the place is within 0.3 m of the place in turn.
The first /resolve may give `variant`, options of re-solving v2 (research arm CT2: `readmit`, `line_span_m`,
`vacate_after_s`, `path_memory`, and CT3's `co_motion_s`; see `sgi.resolve.Resolver`); the answer then also says whether the pose is the
vacated place (`vacated`) and the direction the solver was given (`solve_direction`). The research options of the
role belief (`sgi.roles`) are `role_belief` (CT4), `static_win_s` and `role_decision` (CT5), and `role_direction`
(CT4P: "path" lines up along the remembered line at the back of the roster; the answer says which in
`direction_source`).
With `"settle": true` (research arm CTs), a /resolve made while a tracked person near the waiting people has moved
more than 0.1 m in the last second returns the last solve again with status `held` (`Resolver.settled`).
For research ceilings only, the first /resolve may give `oracle`: {"come_after": [track ids]} replaces the answer's
people and constraints with the tracks and one `come_after` of those ids; {"target_xy": [x, y]} moves the target
point (and the standard goal with it, where one can be placed). The answer then lists them under `oracle`.
With the role belief, the variant option `oracle_roster` (a ceiling) takes the roster at every /resolve from its
`oracle_ahead`, the track ids of those waiting ahead of the robot, in place of the belief's.

The goal method is `--goal`: constraints by default, the method that measured best. Under direct_marking the goal
stage is put on cot unless --strategy names its strategy: a service has no example scenes to show.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Sequence

import numpy as np
from PIL import Image

from sgi.config import Config, ConfigError, load_config, with_model_options
from sgi.geometry import Camera, RobotPose, in_image, lift, point_to_pixel, project
from sgi.inputs import ObservationData
from sgi.pipeline import Pipeline, build_pipeline
from sgi.resolve import Resolver, Track, paint_out
from sgi.roles import UNSET, RoleBelief, decide
from sgi.stages.people import FoundPerson, PeopleFound
from sgi.solver import FloorMap, solve, turn_roles
from sgi.stages.goal import standard_goal
from sgi.types import Constraint, DetectedPerson
from sgi.vlm.base import QuotaExceededError
from sgi.vlm.env import MissingKeyError

log = logging.getLogger("sgi.serve")

_CONSTRAINTS, _BASELINE = "constraints", "direct_marking"
_WITH_EXAMPLES = ("few_shot", "few_shot_cot")      # goal strategies that show example scenes


class RequestError(ValueError):
    """A body the service cannot read: answered with status 400."""


def make_config(model: str | None, stage_models: Sequence[str] = (), strategies: Sequence[str] = (),
                goal: str = _CONSTRAINTS, max_rpm: int | None = None) -> Config:
    """The default configuration under the model options, with `run.goal` set to `goal`. Under direct_marking
    a goal strategy that shows examples is replaced by cot, unless `strategies` names the goal's strategy.
    `max_rpm` lowers every entry's request limit to at most that many a minute: for services that share a key."""
    cfg = with_model_options(load_config(overrides={"run": {"goal": goal}}), model, stage_models, strategies)
    if max_rpm is not None:
        cfg = cfg.model_copy(update={"models": {
            name: entry.model_copy(update={"max_requests_per_minute": min(entry.max_requests_per_minute or max_rpm,
                                                                           max_rpm)})
            for name, entry in cfg.models.items()}})
    named = any(value.partition("=")[0].strip() == "goal" for value in strategies)
    if goal == _BASELINE and not named and cfg.stages["goal"].strategy in _WITH_EXAMPLES:
        cfg = with_model_options(cfg, None, (), ["goal=cot"])
    return cfg


class Sessions:
    """Stored answers of the constraint method, by session id, for /resolve: the newest `keep` of them."""

    def __init__(self, keep: int = 64):
        self._items: dict[str, dict] = {}
        self._keep = keep
        self._lock = threading.Lock()

    def add(self, item: dict) -> str:
        session = uuid.uuid4().hex
        with self._lock:
            self._items[session] = item
            while len(self._items) > self._keep:
                self._items.pop(next(iter(self._items)))
        return session

    def get(self, session: str) -> dict | None:
        with self._lock:
            return self._items.get(session)


def _field(body: dict, key: str) -> Any:
    if key not in body:
        raise RequestError(f"the body has no '{key}'")
    return body[key]


def observation_from(body: dict) -> ObservationData:
    """The observation a /goal body describes. A missing field or one of the wrong shape is a RequestError."""
    try:
        rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(_field(body, "rgb_png")))).convert("RGB"))
        depth = np.load(io.BytesIO(base64.b64decode(_field(body, "depth_npy"))), allow_pickle=False)
        cam, robot = _field(body, "camera"), _field(body, "robot")
        T = np.asarray(cam["T_world_camera"], dtype=np.float64)
        camera = Camera(fx=float(cam["fx"]), fy=float(cam["fy"]), cx=float(cam["cx"]), cy=float(cam["cy"]),
                        width=int(cam["width"]), height=int(cam["height"]), T_world_camera=T)
        pose = RobotPose(x=float(robot["x"]), y=float(robot["y"]), yaw=float(robot["yaw"]))
    except RequestError:
        raise
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise RequestError(f"the observation cannot be read: {error}") from error
    depth = depth.astype(np.float32)
    if T.shape != (4, 4):
        raise RequestError(f"T_world_camera is {T.shape}, not 4 x 4")
    if depth.shape != rgb.shape[:2] or depth.shape != (camera.height, camera.width):
        raise RequestError(f"rgb {rgb.shape[:2]}, depth {depth.shape} and camera {(camera.height, camera.width)} "
                           "differ in size")
    return ObservationData(rgb=rgb, depth=depth, camera=camera, robot=pose)


def _triple(values) -> list[float] | None:
    return None if values is None else [float(v) for v in values]


_CALL_FIELDS = {"stage", "status", "model", "latency_s", "cached", "error"}


def _inference(plan, calls, result=None) -> dict:
    """What the stages worked out, as JSON: the `inference` of a /goal answer."""
    shown = {"plan": None if plan is None else plan.model_dump(mode="json", exclude={"image_sha256"}),
             "calls": [call.model_dump(mode="json", include=_CALL_FIELDS) for call in calls]}
    if result is not None:
        shown.update(result.model_dump(mode="json", exclude={"calls", "status", "source", "pose_world", "pose_robot",
                                                             "timings_s", "flags", "location"}))
        shown["flags"] = [name for name, value in result.flags.model_dump().items() if value]
    return shown


class GivenPeople:
    """A people source for people the caller already has, from a detector of its own: each person's world centre
    and heading, shown at their point 1.1 m up as the camera sees it. Those not in the picture are left out (the
    model is shown the picture). No model call."""

    def __init__(self, people: Sequence[dict]):
        try:
            self.people = [((float(p["xy"][0]), float(p["xy"][1])), None if p.get("yaw") is None else float(p["yaw"]))
                           for p in people]
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise RequestError(f"'people' cannot be read: {error}") from error

    def __call__(self, observation: ObservationData) -> PeopleFound:
        started = time.perf_counter()
        camera, found = observation.camera, []
        for xy, yaw in self.people:
            shown = project(camera, (xy[0], xy[1], _SHOWN_AT_M))
            if shown is not None and in_image(camera, *shown[:2]):
                found.append(FoundPerson(xy=xy, yaw=yaw, pixel=(float(shown[0]), float(shown[1])), label=None))
        return PeopleFound(found=found, calls=[], call_time_s=0.0, geometry_time_s=time.perf_counter() - started)


_SHOWN_AT_M = 1.1                                   # a person's point shown in the picture: 1.1 m above the floor


def infer(pipeline: Pipeline, body: dict, sessions: Sessions | None = None) -> dict:
    """Plan on the observation and place the goal of the plan's first navigation subtask: the /goal answer. With
    `sessions`, an answer of the constraint method that placed a goal is stored there for /resolve. With `people`
    in the body (the caller's own detections, [{"xy": [x, y], "yaw": radians or null}]), the constraint method uses
    them instead of a people call."""
    request = _field(body, "request")
    if not isinstance(request, str) or not request.strip():
        raise RequestError("'request' has to be a non-empty string")
    context = body.get("context")
    observation = observation_from(body)
    started = time.perf_counter()
    outcome = pipeline.plan(request, context, observation=observation)
    subtasks = [] if outcome.plan is None else outcome.plan.subtasks
    answer = {"status": outcome.status, "plan_status": outcome.status,
              "subtasks": [[subtask.type, subtask.description] for subtask in subtasks], "subtask_index": None,
              "source": None, "pose_world": None, "pose_robot": None, "target_world": None, "session": None,
              "calls": [] if outcome.call is None else [[outcome.call.stage, outcome.call.status]],
              "timings_s": {"plan": outcome.time_s},
              "inference": _inference(outcome.plan, [] if outcome.call is None else [outcome.call])}
    index = next((i for i, subtask in enumerate(subtasks) if subtask.type == "navigate"), None)
    if outcome.plan is None or index is None:
        if outcome.plan is not None:
            answer["status"] = "no_target"
        return answer
    inferrer = None
    if body.get("people") is not None and pipeline.cfg.run.goal == _CONSTRAINTS:
        inferrer = pipeline.constraint_inferrer(people_source=GivenPeople(body["people"]))
    result = pipeline.infer_goal(observation, request, context, outcome.plan, index, inferrer=inferrer)
    answer.update({"status": result.status, "subtask_index": index, "source": result.source,
                   "pose_world": _triple(result.pose_world), "pose_robot": _triple(result.pose_robot),
                   "target_world": None if result.target is None else _triple(result.target.world),
                   "calls": answer["calls"] + [[call.stage, call.status] for call in result.calls],
                   "timings_s": {**answer["timings_s"], **result.timings_s,
                                 "total": time.perf_counter() - started},
                   "inference": _inference(outcome.plan, [outcome.call, *result.calls], result)})
    constraint_answer = (result.source == _CONSTRAINTS and result.people is not None
                         and result.constraints is not None)
    # `session_always` (research arm G): also keep an answer without people, for a /resolve whose oracle gives them
    if (sessions is not None and result.status == "ok" and result.target is not None
            and (constraint_answer or body.get("session_always"))):
        standard = standard_goal(observation, result.target, pipeline.cfg.thresholds)
        if standard.pose_world is not None:
            answer["session"] = sessions.add({"target": result.target, "people": result.people or [],
                                              "standard": standard,
                                              "constraints": [use.constraint for use in result.constraints or []],
                                              "observation": observation, "resolver": None})
    return answer


def plan_target(pipeline: Pipeline, body: dict) -> dict:
    """The plan call alone, and the world point of the target of its first navigation subtask: the /plan answer,
    for a caller that moves towards the target while the rest of the answer is worked out. The plan call is the one
    /goal makes first, so a /goal on the same body afterwards finds it in the cache."""
    request = _field(body, "request")
    if not isinstance(request, str) or not request.strip():
        raise RequestError("'request' has to be a non-empty string")
    observation = observation_from(body)
    outcome = pipeline.plan(request, body.get("context"), observation=observation)
    subtasks = [] if outcome.plan is None else outcome.plan.subtasks
    index = next((i for i, subtask in enumerate(subtasks) if subtask.type == "navigate"), None)
    target_world = None
    if index is not None and subtasks[index].target_point is not None:
        point, camera = subtasks[index].target_point, observation.camera
        u, v = point_to_pixel(point.x, point.y, camera.width, camera.height)
        world = lift(observation.depth, camera, u, v, window=pipeline.cfg.thresholds.lift_window)
        target_world = None if world is None else [float(w) for w in world]
    return {"status": outcome.status, "target_world": target_world, "timings_s": {"plan": outcome.time_s}}


def _tracks(body: dict, key: str) -> list[Track]:
    try:
        return [Track(id=int(t["id"]), xy=(float(t["xy"][0]), float(t["xy"][1])),
                      yaw=None if t.get("yaw") is None else float(t["yaw"])) for t in body.get(key) or []]
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise RequestError(f"'{key}' cannot be read: {error}") from error


_IN_TURN_M = 0.3                                   # a place this near the place in turn is it
_VARIANT_OPTIONS = {"readmit", "line_span_m", "vacate_after_s", "vacate_rejoined", "path_memory",  # CT2
                    "co_motion_s", "co_motion_follow", "co_motion_rejoin",                            # and CT3
                    "group_follow", "anchor_rederive",                       # ideas round 2: CT3F, the bolt-on
                    "role_belief",                                           # and CT4, one estimate of who waits
                    "static_win_s", "role_decision",                         # and CT5: its window, a cost decision
                    "role_direction", "role_joined",                 # and CT4P, CT4PJ: direction, joined rule
                    "oracle_roster"}                                 # and a ceiling: the roster from ground truth
_ROLE_PATH = {"path_memory": True, "line_span_m": 0.5}     # the remembered line, as re-solving v2 (CT2) builds it


def _floor_map(body: dict) -> FloorMap | None:
    """The robot's navigation map from a /resolve body: {"origin": [x, y] of the centre of the first pixel,
    "resolution": metres, "png": base64 PNG, non-zero where the robot may stand, rows along y}."""
    given = body.get("floor_map")
    if given is None:
        return None
    try:
        mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(given["png"]))).convert("L")) > 0
        return FloorMap(origin_xy=(float(given["origin"][0]), float(given["origin"][1])),
                        resolution=float(given["resolution"]), navigable=mask)
    except (KeyError, TypeError, ValueError, OSError, IndexError) as error:
        raise RequestError(f"'floor_map' cannot be read: {error}") from error


def resolve(pipeline: Pipeline, sessions: Sessions, body: dict) -> dict:
    """Solve a stored answer of the constraint method again on tracked people and a new observation: the /resolve
    answer. No model is called."""
    session = _field(body, "session")
    item = sessions.get(session) if isinstance(session, str) else None
    if item is None:
        raise RequestError(f"no session {session!r}")
    observation = observation_from(body)
    tracks = _tracks(body, "tracks")
    try:
        t = float(_field(body, "t"))
    except (TypeError, ValueError) as error:
        raise RequestError(f"'t' cannot be read: {error}") from error
    started = time.perf_counter()
    if item["resolver"] is None:
        at_answer = _tracks(body, "tracks_at_answer")
        people, constraints = item["people"], item["constraints"]
        oracle = body.get("oracle") or {}
        item["oracle"] = []
        if oracle.get("chain_m") is not None:         # research arm G: who waits, from geometry alone
            item["chain"] = _chain(at_answer, item["target"].world[:2], float(oracle["chain_m"]))
            oracle = {**oracle, "come_after": item["chain"]}
            item["oracle"].append("chain")
        if oracle.get("come_after") is not None:      # a ceiling for research: who waits, from ground truth
            people = [DetectedPerson(number=t.id + 1, xy=t.xy, yaw=t.yaw, pixel=(0.0, 0.0)) for t in at_answer]
            known = {t.id for t in at_answer}
            constraints = [Constraint(because="oracle: those waiting", kind="come_after",
                                      people=[int(i) + 1 for i in oracle["come_after"] if int(i) in known])]
            if "chain" not in item["oracle"]:
                item["oracle"].append("come_after")
        if oracle.get("target_xy") is not None:       # a ceiling for research: the target point, from ground truth
            x, y = (float(v) for v in oracle["target_xy"])
            item["target"] = item["target"].model_copy(update={"world": (x, y, item["target"].world[2])})
            standard = standard_goal(item.get("observation") or observation, item["target"], pipeline.cfg.thresholds)
            if standard.pose_world is not None:
                item["standard"] = standard
            item["oracle"].append("target")
        variant = body.get("variant") or {}
        unknown = set(variant) - _VARIANT_OPTIONS
        if unknown:
            raise RequestError(f"'variant' has unknown options: {sorted(unknown)}")
        variant = dict(variant)
        role_belief = bool(variant.pop("role_belief", False))
        static_win_s = float(variant.pop("static_win_s", 0.0))
        role_decision = bool(variant.pop("role_decision", False))
        role_direction = variant.pop("role_direction", None)
        role_joined = bool(variant.pop("role_joined", False))
        oracle_roster = bool(variant.pop("oracle_roster", False))
        if (static_win_s or role_decision or role_direction or role_joined or oracle_roster) and not role_belief:
            raise RequestError("'static_win_s', 'role_decision', 'role_direction', 'role_joined' and 'oracle_roster' "
                               "need 'role_belief'")
        if role_direction not in (None, "belief", "path"):
            raise RequestError(f"'role_direction' is 'belief' or 'path', not {role_direction!r}")
        if role_joined and role_direction != "path":
            raise RequestError("'role_joined' needs 'role_direction' \"path\"")
        item["resolver"] = Resolver(people, constraints, item["target"].world[:2], at_answer, **variant)
        item["belief"] = None
        if role_belief:                             # research arm CT4: who waits from one estimate (sgi.roles)
            history = body.get("history") or []
            start = history[0].get("robot") if history and history[0].get("robot") else (observation.robot.x,
                                                                                         observation.robot.y)
            item["belief"] = RoleBelief(people, constraints, item["target"].world[:2], at_answer, start,
                                        {"STATIC_WIN_S": static_win_s} if static_win_s else None)
            item["belief_t"] = None
            item["role_decision"], item["choice"] = role_decision, UNSET     # research arm CT5
            # research arm CT4P: a second resolver, for the remembered line's geometry only
            item["path"] = (Resolver(people, constraints, item["target"].world[:2], at_answer, **_ROLE_PATH)
                            if role_direction == "path" else None)
            item["role_joined"] = role_joined           # research arm CT4PJ: the path's join test and "behind"
            item["oracle_roster"] = oracle_roster       # a ceiling: every /resolve gives the roster (`oracle_ahead`)
            if oracle_roster:
                item["oracle"].append("roster")
        item["answer_come_after"] = _come_after_tracks(people, constraints, at_answer)
        item["floor_map"] = _floor_map(body)
        # without a map: the floor of the view the answer was asked on, the people who stood in it painted out
        item["floor_view"] = paint_out(item.get("observation") or observation, [t.xy for t in at_answer])
        if body.get("history") and item["belief"] is not None:
            for frame in sorted(body["history"], key=lambda f: float(f["t"])):
                if float(frame["t"]) < t:
                    _belief_step(item, float(frame["t"]), _tracks(frame, "tracks"))
        elif body.get("history"):                   # research: the tracks of the wait, from the photo on
            _replay_wait(item["resolver"], body["history"], item["target"].world[:2], t,
                         (observation.robot.x, observation.robot.y), pipeline.cfg.thresholds)
    resolver = item["resolver"]
    if item.get("oracle_roster"):
        if body.get("oracle_ahead") is None:
            raise RequestError("'oracle_roster' needs 'oracle_ahead', the track ids ahead, at every /resolve")
        item["oracle_ahead"] = [int(i) for i in body["oracle_ahead"]]
    if item.get("belief") is not None:
        return _resolve_with_belief(pipeline, item, observation, tracks, t, started)
    people, constraints = resolver.inputs(tracks, (observation.robot.x, observation.robot.y), t)
    if body.get("settle"):                          # research arm CTs: solve only on settled tracks
        still = resolver.settled(tracks, t)
        if not still and item.get("last") is not None:
            return {**item["last"], "status": "held", "timings_s": {"resolve": time.perf_counter() - started}}
    th = pipeline.cfg.thresholds
    if resolver.vacated_pose is not None:            # v2: take the place of the last who left
        x, y, yaw = resolver.vacated_pose
        resolver.record((x, y), True)
        item["last"] = {**(item.get("last") or {}), "status": "ok", "pose_world": [x, y, yaw], "pose_robot": None,
                        "in_turn": True, "vacated": True, "joined": resolver.joined,
                        "dropped": sorted(resolver.dropped), "people": len(people)}
        return {**item["last"], "timings_s": {"resolve": time.perf_counter() - started}}
    exempt = resolver.exempt()                      # CT3: whoever has not moved as a joiner would
    if item["floor_map"] is not None:
        solved = solve(observation, item["target"], people, constraints, item["standard"], th,
                       line_direction=resolver.solve_direction, floor_map=item["floor_map"], exempt=exempt)
    else:
        solved = solve(item["floor_view"], item["target"], people, constraints, item["standard"], th,
                       line_direction=resolver.solve_direction, exempt=exempt)
    turn_m = solved.note.keys.turn_m                # at the place in turn, not only with a come_after used
    in_turn = turn_m is not None and turn_m <= _IN_TURN_M
    resolver.record(solved.pose_world[:2], in_turn, solved.note.joined_waiting)
    item["last"] = {"status": "ok", "pose_world": _triple(solved.pose_world), "pose_robot": _triple(solved.pose_robot),
                    "in_turn": in_turn, "joined": resolver.joined, "dropped": sorted(resolver.dropped),
                    "line_direction": resolver.line_direction, "solve_direction": resolver.solve_direction,
                    "vacated": False,
                    "floor": "map" if item["floor_map"] is not None else "first view", "oracle": item["oracle"],
                    "match_rate": resolver.match_rate, "people": len(people),
                    # for research logs, by track id: who the solver took as last and as joining, who was exempt,
                    # and who was taken as not following the line (CT3), with when
                    "last": None if solved.turn_last is None else resolver.track_of(solved.turn_last),
                    "turn_joined": [resolver.track_of(n) for n in solved.note.joined_waiting],
                    "exempt": sorted(resolver.track_of(n) for n in exempt),
                    "roster": sorted(i for i in (resolver.track_of(n) for n in resolver.waiting()) if i is not None),
                    "non_followers": {str(resolver.track_of(n)): t for n, t in resolver.non_follower_t.items()},
                    "answer_come_after": item.get("answer_come_after"), "chain": item.get("chain"),
                    "co_motion_events": [_event_by_track(resolver, e) for e in resolver.events]}
    return {**item["last"], "timings_s": {"resolve": time.perf_counter() - started}}


def _belief_step(item: dict, t: float, tracks: Sequence[Track]) -> None:
    last = item.get("belief_t")
    item["belief"].step(t, tracks, 0.5 if last is None else max(t - last, 0.0))
    item["belief_t"] = t


def _resolve_with_belief(pipeline: Pipeline, item: dict, observation: ObservationData, tracks: Sequence[Track],
                         t: float, started: float) -> dict:
    """Research arm CT4: the roles from the role belief (`sgi.roles`), the place from the solver. The resolver only
    numbers the tracks and carries the answer's other constraints; the come_after is the belief's roster, everyone
    else is exempt from the place-in-turn rule, and the line runs along the belief's direction. With
    `role_direction` "path" (research arm CT4P) it runs along the remembered line at the back of the roster
    instead, as `path_memory` points the solver, and along the belief's direction when there is no such line. With
    `role_joined` (research arm CT4PJ) the remembered line also keeps the Resolver's join test, and once the robot
    has joined, the people farther back than it are left out of the solve, as the Resolver leaves them out."""
    resolver, belief = item["resolver"], item["belief"]
    _belief_step(item, t, tracks)
    robot_xy = (observation.robot.x, observation.robot.y)
    people, constraints = resolver.inputs(tracks, robot_xy, t)
    marginals = belief.marginals()
    roster = sorted(i for i, v in marginals.items() if v >= 0.5 and any(tr.id == i for tr in tracks))
    if item.get("oracle_roster"):                   # a ceiling: who waits ahead of the robot, from ground truth
        roster = sorted(i for i in set(item["oracle_ahead"]) if any(tr.id == i for tr in tracks))
    joined, behind, ahead = item.get("role_joined") and item["path"], set(), roster
    if joined:
        joined.update_joined(robot_xy, t)
        if joined.joined:
            behind = {tr.id for tr in tracks if joined.behind(tr.xy, robot_xy)}
            gone = {resolver._number_of[i] for i in behind if i in resolver._number_of}
            people, ahead = [p for p in people if p.number not in gone], [i for i in roster if i not in behind]
    chosen, decision = ahead, {}
    if item.get("role_decision"):                   # research arm CT5: whom to come after, by expected cost
        choice, costs = decide(belief, {tr.id for tr in tracks}, item["choice"])
        item["choice"] = choice
        chosen = [] if choice is None else sorted(
            {i for i in roster if belief.dist_now.get(i, np.inf) < belief.dist_now[choice]} | {choice})
        decision = {"choice": choice,
                    "costs": {("first" if k is None else str(k)): round(v, 4) for k, v in costs.items()}}
    waiting = sorted(resolver._number_of[i] for i in chosen if i in resolver._number_of)
    constraints = [c for c in constraints if c.kind != "come_after"]
    if waiting:
        constraints.append(Constraint(because="the role belief: those waiting", kind="come_after", people=waiting))
    exempt = {p.number for p in people} - set(waiting)
    direction, source = (float(belief.dir[0]), float(belief.dir[1])), "belief"
    if item.get("path") is not None:
        tangent = item["path"].back_tangent([tr.xy for tr in tracks if tr.id in ahead])
        if tangent is not None:
            direction, source = (float(tangent[0]), float(tangent[1])), "path"
    th = pipeline.cfg.thresholds
    floor = observation if item["floor_map"] is not None else item["floor_view"]
    extra = {"floor_map": item["floor_map"]} if item["floor_map"] is not None else {}
    solved = solve(floor, item["target"], people, constraints, item["standard"], th, line_direction=direction,
                   exempt=exempt, **extra)
    turn_m = solved.note.keys.turn_m
    in_turn = turn_m is not None and turn_m <= _IN_TURN_M
    if joined:                                     # as the Resolver records its solve, for the next join test
        joined.record(solved.pose_world[:2], in_turn)
    item["last"] = {"status": "ok", "pose_world": _triple(solved.pose_world), "pose_robot": _triple(solved.pose_robot),
                    "in_turn": in_turn, "joined": bool(joined and joined.joined), "dropped": [],
                    "line_direction": [float(v) for v in belief.dir], "solve_direction": list(direction),
                    "direction_source": source, "vacated": False,
                    "floor": "map" if item["floor_map"] is not None else "first view", "oracle": item["oracle"],
                    "match_rate": resolver.match_rate, "people": len(people),
                    "last": None if solved.turn_last is None else resolver.track_of(solved.turn_last),
                    "turn_joined": [resolver.track_of(n) for n in solved.note.joined_waiting],
                    "exempt": sorted(resolver.track_of(n) for n in exempt), "roster": roster,
                    "marginals": {str(i): round(v, 3) for i, v in marginals.items()},
                    "non_followers": {}, "answer_come_after": item.get("answer_come_after"), "chain": item.get("chain"),
                    "co_motion_events": [], **decision, **({"behind": sorted(behind)} if joined else {})}
    return {**item["last"], "timings_s": {"resolve": time.perf_counter() - started}}


def _replay_wait(resolver: Resolver, history: Sequence[dict], target_xy, t_now: float, robot_now,
                 th) -> None:
    """Run the resolver's bookkeeping over the tracks of the wait between the photo and the first /resolve (a list
    of {"t", "tracks", "robot": [x, y]}, oldest first), each tick as a re-solve would, with the place-in-turn rule
    standing in for the solve: who left, who joined, who did not follow. A slow answer's latency is evidence too."""
    for frame in sorted(history, key=lambda f: float(f["t"])):
        t = float(frame["t"])
        if t >= t_now:
            continue
        robot = tuple(float(v) for v in (frame.get("robot") or robot_now)[:2])
        people, constraints = resolver.inputs(_tracks(frame, "tracks"), robot, t)
        if resolver.vacated_pose is not None:
            resolver.record(tuple(resolver.vacated_pose[:2]), True)
            continue
        _, joined, place = turn_roles(people, constraints, target_xy, robot, th, resolver.solve_direction,
                                      resolver.exempt())
        if place is None:
            resolver.record(robot, False, [])
        else:
            resolver.record(place, True, joined)


def _event_by_track(resolver: Resolver, event: tuple) -> list:
    """A co-motion event of the resolver with its people as track ids: (t, \"advance\", q, p, ...) or
    (t, \"drop\", p, ...)."""
    people = (2, 3) if event[1] == "advance" else (2,)
    return [resolver.track_of(x) if i in people else x for i, x in enumerate(event)]


def _chain(tracks: Sequence[Track], target_xy, reach_m: float) -> list[int]:
    """Research arm G: the tracked person nearest the target point, and everyone within `reach_m` of anyone taken,
    again and again; track ids in the order taken."""
    left = list(tracks)
    if not left:
        return []
    first = min(left, key=lambda tr: math.dist(tr.xy, target_xy))
    chain, left = [first], [tr for tr in left if tr is not first]
    while True:
        near = [tr for tr in left if any(math.dist(tr.xy, c.xy) < reach_m for c in chain)]
        if not near:
            return [tr.id for tr in chain]
        nearest = min(near, key=lambda tr: min(math.dist(tr.xy, c.xy) for c in chain))
        chain.append(nearest)
        left = [tr for tr in left if tr is not nearest]


def _come_after_tracks(people: Sequence[DetectedPerson], constraints: Sequence[Constraint],
                       at_answer: Sequence[Track]) -> list[list]:
    """For research logs: each person a come_after of the answer names, with the nearest track at the answer and
    how far it was: [number, track id or None, metres or None]."""
    named = [n for c in constraints if c.kind == "come_after" for n in c.people]
    where = {p.number: p.xy for p in people}
    rows = []
    for n in named:
        if n not in where or not at_answer:
            rows.append([n, None, None])
            continue
        near = min(at_answer, key=lambda tr: math.dist(tr.xy, where[n]))
        rows.append([n, near.id, round(math.dist(near.xy, where[n]), 3)])
    return rows


def make_handler(pipeline: Pipeline) -> type[BaseHTTPRequestHandler]:
    cfg = pipeline.cfg
    sessions = Sessions()
    health = {"ok": True, "goal": cfg.run.goal, "models": {stage: cfg.stages[stage].model for stage in pipeline.clients}}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, health)
            else:
                self._send(404, {"error": f"no such path: {self.path}"})

        def do_POST(self):
            if self.path not in ("/goal", "/resolve", "/plan"):
                self._send(404, {"error": f"no such path: {self.path}"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if not isinstance(body, dict):
                    raise RequestError("the body is not a JSON object")
                if self.path == "/resolve":
                    self._send(200, resolve(pipeline, sessions, body))
                    return
                if self.path == "/plan":
                    self._send(200, plan_target(pipeline, body))
                    return
                answer = infer(pipeline, body, sessions)
            except (RequestError, json.JSONDecodeError) as error:
                self._send(400, {"error": str(error)})
                return
            except QuotaExceededError as error:
                self._send(503, {"error": f"the provider's allowance is used up: {error}"})
                return
            log.info("%s: %s, pose_world %s", body.get("request"), answer["status"], answer["pose_world"])
            self._send(200, answer)

        def log_message(self, format, *args):      # the answers are logged above, one line each
            pass

    return Handler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sgi.serve",
                                     description="Serve the pipeline over HTTP for a live simulator.")
    parser.add_argument("--model", help="entry of the model table in configs/default.yaml, used for every stage")
    parser.add_argument("--stage-model", action="append", default=[], metavar="STAGE=ENTRY",
                        help="entry to use for one stage, applied after --model; may be given several times")
    parser.add_argument("--strategy", action="append", default=[], metavar="STAGE=NAME",
                        help="prompting strategy for one stage; may be given several times")
    parser.add_argument("--goal", choices=(_CONSTRAINTS, _BASELINE), default=_CONSTRAINTS,
                        help="the method that places the goal where people matter (default: constraints)")
    parser.add_argument("--max-rpm", type=int, help="at most this many requests a minute to any provider, for a "
                        "service that shares its key with others")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        pipeline = build_pipeline(make_config(args.model, args.stage_model, args.strategy, args.goal, args.max_rpm))
    except (MissingKeyError, ConfigError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    server = HTTPServer((args.host, args.port), make_handler(pipeline))
    models = ", ".join(f"{stage}={pipeline.cfg.stages[stage].model}" for stage in pipeline.clients)
    print(f"sgi.serve: listening on http://{args.host}:{args.port} (goal {args.goal}; {models})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
