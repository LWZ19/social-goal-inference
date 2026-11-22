"""The goal pose: the standard goal, from geometry alone, and the baseline in which a model marks the goal."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from sgi.config import PROMPTS_DIR, Strategy, Thresholds
from sgi.geometry import (RobotPose, bearing, floor_in_view, lift, lift_all, make_poses, point_to_pixel,
                          surface_normal_xy)
from sgi.imaging import draw_target_marker, encode_png
from sgi.inputs import ObservationData
from sgi.prompts import Example, build_goal_request
from sgi.schemas import goal_from_wire, goal_schema
from sgi.types import ContextCheck, Flags, GoalMark, GoalResult, GroundedTarget, Point, Pose, Subtask
from sgi.vlm.base import VLMClient, to_call

_OBSTACLE_STRIDE = 4                               # every fourth depth pixel each way is tested as an obstacle
_RIM_POINTS = 8                                    # the floor is asked at a stop and at this many points round it
_LOOK_AT_MODE = "vlm_look_at"                      # the heading mode that faces the model's look_at_point


class GoalInferrer(Protocol):
    """A method that places the goal where the people in view matter."""

    def infer(self, observation: ObservationData, request: str, subtask: Subtask,
              target: GroundedTarget, context_check: ContextCheck | None) -> GoalResult: ...


def _poses(robot: RobotPose, goal_xy, faced_xy, th: Thresholds) -> tuple[Pose, Pose, bool]:
    """(pose_world, pose_robot, degenerate) for a goal position that faces `faced_xy`.

    Closer together than `heading_degenerate_m` the two points give no bearing: the robot keeps its heading
    and `degenerate` is True.
    """
    degenerate = math.hypot(faced_xy[0] - goal_xy[0], faced_xy[1] - goal_xy[1]) < th.heading_degenerate_m
    theta_world = robot.yaw if degenerate else bearing(goal_xy, faced_xy)
    pose_world, pose_robot = make_poses(robot, goal_xy, theta_world)
    return pose_world, pose_robot, degenerate


def _is_clear(xy: np.ndarray, obstacles_xy: np.ndarray, clearance_m: float) -> bool:
    """Whether every obstacle point is at least `clearance_m` from `xy` on the ground plane."""
    if len(obstacles_xy) == 0:
        return True
    return bool(np.hypot(obstacles_xy[:, 0] - xy[0], obstacles_xy[:, 1] - xy[1]).min() >= clearance_m)


def _floor_shown(observation: ObservationData, xy: np.ndarray, unit: np.ndarray, th: Thresholds) -> bool:
    """Whether the image shows the floor under a robot that stands at `xy`: at that point and at `_RIM_POINTS`
    points `robot_radius_m` round it, the first of them along `unit`. The robot is twice that radius wide,
    and no part of it should reach into space the picture does not show."""
    turns = math.atan2(unit[1], unit[0]) + 2 * math.pi * np.arange(_RIM_POINTS) / _RIM_POINTS
    rim = xy + th.robot_radius_m * np.stack([np.cos(turns), np.sin(turns)], axis=1)
    return all(floor_in_view(observation.depth, observation.camera, spot, th.floor_tolerance_m,
                             th.visibility_depth_tolerance_m, th.lift_window) for spot in (xy, *rim))


def _first_clear(anchor: np.ndarray, unit: np.ndarray, obstacles_xy: np.ndarray, th: Thresholds,
                 reach: float = math.inf, in_view: ObservationData | None = None) -> np.ndarray | None:
    """The first clear step from the anchor along `unit`: from `standoff_m` out by `standard_goal_step_m` at a
    time, as far as `standard_goal_max_slide_m` and never beyond `reach`. With `in_view`, a step also has to
    stand, as wide as the robot is, on floor that observation shows. None when no step passes."""
    last_step = math.floor(th.standard_goal_max_slide_m / th.standard_goal_step_m + 1e-9)
    for k in range(last_step + 1):
        d = th.standoff_m + k * th.standard_goal_step_m
        if d > reach + 1e-9:
            return None
        candidate = anchor + d * unit
        if _is_clear(candidate, obstacles_xy, th.standard_goal_clearance_m) and (
                in_view is None or _floor_shown(in_view, candidate, unit, th)):
            return candidate
    return None


def standard_goal(observation: ObservationData, target: GroundedTarget, th: Thresholds) -> GoalResult:
    """Stop `standoff_m` in front of the surface the target point lies on, facing the target.

    In front is along the horizontal normal of that surface, on the side the camera sees: a stop on the robot's
    own line of approach is beside a target that is seen at an angle. `standard_goal_frontal` is set when the
    stop is taken there. Where the depth image shows no near-vertical flat surface round the point, or no place
    before it that is clear and on floor the image shows, the stop is `standoff_m` short of the target on the
    line from it to the robot.

    On either line the stop slides away from the target, by at most `standard_goal_max_slide_m`, to the first
    step with nothing standing between the floor tolerance and the camera's height within
    `standard_goal_clearance_m`. Before the surface a step is taken only where the image shows its floor, at
    the step and a robot radius round it: what the camera does not see is not clear, and there the stop can
    lie behind or inside something, or beside the picture. Floor just under the picture's lower edge counts
    as shown where the picture ends on floor above it (`floor_in_view`). On the line to the robot, which the
    camera looks along, the stop never slides past the robot.
    When no step of that line is clear either, the pose at `standoff_m` on it is returned with
    `standard_goal_blocked` set. The camera's height stands for the robot's, as in scene check 11: a ceiling or
    a sign overhead is in nobody's way. A robot already nearer the target than `standoff_m` stays where it is
    and turns to it.
    """
    started = time.perf_counter()
    robot = observation.robot
    anchor = np.array(target.world[:2], dtype=np.float64)
    origin = np.array([robot.x, robot.y], dtype=np.float64)
    distance = float(np.hypot(*(origin - anchor)))
    blocked = frontal = False
    if distance < th.standoff_m:
        goal = origin                              # already inside the standoff: stay and turn to the anchor
    else:
        points = lift_all(observation.depth, observation.camera, stride=_OBSTACLE_STRIDE)
        z = points[:, 2]
        obstacles_xy = points[(z > th.floor_tolerance_m) & (z < observation.camera.T_world_camera[2, 3]), :2]
        normal = surface_normal_xy(observation.depth, observation.camera, target.world, th.surface_radius_m,
                                   th.surface_min_points, th.surface_max_tilt, th.surface_max_rms_m,
                                   th.surface_min_spread_m, th.surface_max_rms_ratio)
        goal = None if normal is None else _first_clear(anchor, normal, obstacles_xy, th, in_view=observation)
        frontal = goal is not None
        if goal is None:                           # no face to stand before, or no clear place in view before it
            unit = (origin - anchor) / distance
            goal = _first_clear(anchor, unit, obstacles_xy, th, reach=distance)   # past the robot is behind it
            if goal is None:
                goal, blocked = anchor + th.standoff_m * unit, True
    pose_world, pose_robot, degenerate = _poses(robot, goal, anchor, th)
    flags = Flags(standard_goal_blocked=blocked, standard_goal_frontal=frontal, heading_degenerate=degenerate)
    return GoalResult(status="ok", source="standard", pose_world=pose_world, pose_robot=pose_robot, target=target,
                      flags=flags, timings_s={"goal": 0.0, "geometry": time.perf_counter() - started})


def _lift_point(observation: ObservationData, point: Point,
                th: Thresholds) -> tuple[tuple[float, float], np.ndarray | None]:
    """A normalised point as a pixel of the observation, and the world point there (None where depth gives none).

    The pixel is taken at the observation's own size: the image the point was marked on may have been sent smaller.
    """
    camera = observation.camera
    u, v = point_to_pixel(point.x, point.y, camera.width, camera.height)
    return (u, v), lift(observation.depth, camera, u, v, th.lift_window)


class DirectMarkingInferrer:
    """The baseline: a model marks the goal on the image and depth lifts the mark to the ground."""

    def __init__(self, client: VLMClient, strategy: Strategy, heading_mode: str, examples: Sequence[Example],
                 th: Thresholds, max_image_side: int | None = None, trial: int = 0,
                 prompts_dir: Path = PROMPTS_DIR):
        self.client = client
        self.strategy = strategy
        self.heading_mode = heading_mode
        self.examples = examples
        self.th = th
        self.max_image_side = max_image_side
        self.trial = trial
        self.prompts_dir = prompts_dir

    def infer(self, observation: ObservationData, request: str, subtask: Subtask, target: GroundedTarget,
              context_check: ContextCheck | None) -> GoalResult:
        entered = time.perf_counter()
        image = encode_png(draw_target_marker(observation.rgb, *target.pixel), self.max_image_side)
        pf, size = self.client.entry.point_format, (image.width, image.height)
        people = None if context_check is None else context_check.relevant_people
        system, parts = build_goal_request(self.strategy, request, subtask, image, people, self.examples, pf,
                                           self.heading_mode, prompts_dir=self.prompts_dir)
        schema = goal_schema(self.strategy, pf, size, self.heading_mode)
        asked = time.perf_counter()
        resp = self.client.generate(parts, schema, system=system, trial=self.trial)
        answered = time.perf_counter()
        shared = {"source": "direct_marking", "target": target, "calls": [to_call("goal", resp)]}

        def goal_time(end: float) -> float:
            """This stage's time up to `end`, with the response's own latency in place of the time `generate` took."""
            return (end - entered) - (answered - asked) + resp.latency_total_s

        if resp.status != "ok":
            return GoalResult(status=resp.status, timings_s={"goal": goal_time(answered), "geometry": 0.0}, **shared)
        mark = goal_from_wire(resp.parsed, pf, size)
        read = time.perf_counter()
        placed = self._place(observation, target, mark)
        timings = {"goal": goal_time(read), "geometry": time.perf_counter() - read}
        return GoalResult(goal_mark=mark, timings_s=timings, **placed, **shared)

    def _place(self, observation: ObservationData, target: GroundedTarget, mark: GoalMark) -> dict[str, Any]:
        """What the mark and the depth decide: the status, the goal pixel and, when the mark lifts, poses and flags."""
        th = self.th
        pixel, p = _lift_point(observation, mark.goal_point, th)
        if p is None:
            return {"status": "lift_failed", "goal_pixel": pixel}
        faced_xy, fallback = target.world[:2], False
        if self.heading_mode == _LOOK_AT_MODE:
            looked = None if mark.look_at_point is None else _lift_point(observation, mark.look_at_point, th)[1]
            if looked is None:
                fallback = True                    # nothing to look at: the target stays the faced point
            else:
                faced_xy = looked[:2]
        pose_world, pose_robot, degenerate = _poses(observation.robot, p[:2], faced_xy, th)
        flags = Flags(marked_point_off_floor=bool(abs(p[2]) > th.floor_tolerance_m), heading_fallback=fallback,
                      heading_degenerate=degenerate)
        return {"status": "ok", "goal_pixel": pixel, "pose_world": pose_world, "pose_robot": pose_robot,
                "flags": flags}
