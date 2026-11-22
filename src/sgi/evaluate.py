"""The scorer and the supplementary measures.

Harness code and generic geometry over ground truth: nothing here reads an episode's situation or calls a
model, and every number comes from the thresholds."""
import math
from typing import Sequence

from pydantic import BaseModel

from sgi import freespace
from sgi.config import Thresholds
from sgi.geometry import bearing, to_world_xy, wrap_angle
from sgi.inputs import ObservationData
from sgi.polygons import distance_to_polygon, point_in_polygon
from sgi.scene import EpisodeGT, GroundTruth, PlanStep, Region, Target
from sgi.types import GoalResult, Plan

_Point = tuple[float, float]
_Entry = tuple[str, str | None]     # a subtask's type with what its target matched: a target id, "ungrounded",
                                    # "unmatched", or None for an act step
_UNGROUNDED = "ungrounded"          # a navigate subtask with no target point in the scene


class Score(BaseModel):
    """One scored goal: success and its four components, then the supplementary measures."""
    success: bool
    target_correct: bool
    matched_target_id: str | None = None
    reachable: bool = False
    in_region: bool = False
    heading_ok: bool = False
    distance_to_region_m: float | None = None
    min_body_gap_m: float | None = None
    personal_space_intrusion: bool | None = None
    forbidden: dict[str, bool] = {}                # only kinds that have a region in the episode


def _indicates(p: _Point, target: Target, truth: GroundTruth, th: Thresholds) -> bool:
    """Whether the ground point p is on the target: within the tolerance of its footprint or, without a
    footprint, within its match radius of the anchor. A person target must also be the person nearest to p."""
    if target.footprint is not None:
        on_target = distance_to_polygon(p, target.footprint) <= th.match_tolerance_m
    else:
        radius = th.default_match_radius_m if target.match_radius is None else target.match_radius
        on_target = math.hypot(p[0] - target.anchor[0], p[1] - target.anchor[1]) <= radius
    if not on_target or target.kind != "person":
        return on_target
    nearest = min(truth.people, key=lambda person: math.hypot(p[0] - person.x, p[1] - person.y), default=None)
    return nearest is not None and nearest.id == target.person_id


def match_target(target_world, truth: GroundTruth, ep: EpisodeGT, th: Thresholds) -> str | None:
    """The first of the episode's targets that the ground projection of the lifted target point falls on."""
    p = (float(target_world[0]), float(target_world[1]))
    for target_id in ep.target_ids:
        if _indicates(p, truth.targets[target_id], truth, th):
            return target_id
    return None


def _heading_within(region: Region, goal: _Point, theta: float) -> bool:
    """Whether the world heading theta, taken at the goal, is within the region's tolerance of its allowed heading."""
    heading = region.heading
    allowed = heading.yaw if heading.yaw is not None else bearing(goal, heading.face_point)
    return abs(wrap_angle(theta - allowed)) <= math.radians(region.tolerance_deg)


def score(result: GoalResult, observation: ObservationData, truth: GroundTruth, ep: EpisodeGT,
          th: Thresholds) -> Score:
    """Score a goal against an episode's ground truth.

    The pose scored is pose_robot, taken to the world frame with the observation's robot pose. A result
    that is not ok, or has no pose, fails every component and carries no measure."""
    if result.status != "ok" or result.pose_robot is None:
        return Score(success=False, target_correct=False)
    robot = observation.robot
    goal = to_world_xy(robot, result.pose_robot[0], result.pose_robot[1])
    theta = result.pose_robot[2] + robot.yaw

    matched = None if result.target is None else match_target(result.target.world, truth, ep, th)
    candidates = [] if matched is None else [
        region for region in ep.acceptable_regions or [] if region.target_id in (None, matched)]
    holding = [region for region in candidates if point_in_polygon(goal, region.polygon)]
    in_region = bool(holding)
    heading_ok = any(_heading_within(region, goal, theta) for region in holding)
    reachable = freespace.reachable(freespace.robot_grid(truth, robot, th), (robot.x, robot.y), goal)

    body_gap = min((math.hypot(goal[0] - person.x, goal[1] - person.y) - th.person_radius_m - th.robot_radius_m
                    for person in truth.people), default=None)
    forbidden: dict[str, bool] = {}
    for region in ep.forbidden_regions or []:
        forbidden[region.kind] = forbidden.get(region.kind, False) or point_in_polygon(goal, region.polygon)

    return Score(
        success=matched is not None and reachable and in_region and heading_ok,
        target_correct=matched is not None,
        matched_target_id=matched,
        reachable=reachable,
        in_region=in_region,
        heading_ok=heading_ok,
        distance_to_region_m=min((distance_to_polygon(goal, region.polygon) for region in candidates), default=None),
        min_body_gap_m=body_gap,
        personal_space_intrusion=None if body_gap is None else body_gap < th.personal_gap_m,
        forbidden=forbidden,
    )


def reduce_plan(plan: Plan, matched: Sequence[str | None]) -> list[tuple[str, str | None]]:
    """The plan as one (type, matched) entry per subtask, with adjacent identical entries merged."""
    if len(matched) != len(plan.subtasks):
        raise ValueError(f"matched has {len(matched)} entries for a plan of {len(plan.subtasks)} subtasks")
    reduced: list[_Entry] = []
    for subtask, match in zip(plan.subtasks, matched):
        entry = (subtask.type, match)
        if not reduced or reduced[-1] != entry:
            reduced.append(entry)
    return reduced


def _same_step(a: PlanStep, b: PlanStep) -> bool:
    """Whether two reference steps ask for the same thing: one type and one set of target ids. Next to each
    other they are one step, because reduce_plan leaves one entry for a plan that answers both the same way."""
    return a.type == b.type and set(a.target_ids) == set(b.target_ids)


def _merge_reference(reference: Sequence[PlanStep]) -> list[PlanStep]:
    """Adjacent steps of one type and one set of target ids as a single step, optional only when all of them are."""
    merged: list[PlanStep] = []
    for step in reference:
        last = merged[-1] if merged else None
        if last is not None and _same_step(last, step):
            merged[-1] = last.model_copy(update={"optional": last.optional and step.optional})
        else:
            merged.append(step)
    return merged


def _fits(entry: _Entry, step: PlanStep) -> bool:
    """Whether a predicted entry answers a reference step. A navigate step accepts any of its target ids, or an
    ungrounded target when it lists none (its target is not in the scene)."""
    kind, matched = entry
    if kind != step.type:
        return False
    if kind != "navigate":
        return True
    return matched in step.target_ids if step.target_ids else matched == _UNGROUNDED


def _fits_in_order(predicted: Sequence[_Entry], steps: Sequence[PlanStep], kept: PlanStep | None = None) -> bool:
    """Whether the entries fit the steps one for one and in order, with some of the optional steps left out.

    Both choices are tried at an optional step, because an entry that fits it may belong to a later step.
    kept is the last step that was not left out. A step equal to it is its neighbour once the optional steps
    between them are gone, so it may share that step's entry instead of taking one of its own."""
    if not steps:
        return not predicted
    step, later = steps[0], steps[1:]
    if predicted and _fits(predicted[0], step) and _fits_in_order(predicted[1:], later, step):
        return True
    shares = kept is not None and _same_step(kept, step)
    return (step.optional or shares) and _fits_in_order(predicted, later, kept)


def _navigate_recall(predicted: Sequence[_Entry], steps: Sequence[PlanStep]) -> float:
    """The share of the non-optional navigate steps answered by a predicted entry of their own, in any order.

    The optional steps are left out first and the rest merged again, so navigate steps that are equal and
    have only optional steps between them count once."""
    required = [step for step in _merge_reference([step for step in steps if not step.optional])
                if step.type == "navigate"]
    if not required:
        return 1.0
    unused = list(predicted)
    found = 0
    for step in required:
        hit = next((i for i, entry in enumerate(unused) if _fits(entry, step)), None)
        if hit is not None:
            del unused[hit]
            found += 1
    return found / len(required)


def decomposition_accuracy(predicted: Sequence[tuple[str, str | None]],
                           reference: Sequence[PlanStep]) -> tuple[bool, float]:
    """(exact sequence match, navigate-step recall) of a reduced plan against a reference plan."""
    steps = _merge_reference(reference)
    return _fits_in_order(predicted, steps), _navigate_recall(predicted, steps)


def tour_frame_correct(chosen: int, shown: Sequence[int], truth_frames: Sequence[int]) -> bool:
    """Whether the chosen tour frame is one of truth_frames, the frames in which a target is visible.

    When none of those was shown, every shown frame nearest in index to one of them is correct instead."""
    if not truth_frames:
        raise ValueError("no reference tour frames to measure against")
    if any(index in truth_frames for index in shown):
        return chosen in truth_frames
    distance = {index: min(abs(index - frame) for frame in truth_frames) for index in shown}
    return chosen in distance and distance[chosen] == min(distance.values())
