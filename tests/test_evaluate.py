import math
from dataclasses import replace

import pytest

import sgi.evaluate as ev
from sgi.config import Thresholds
from sgi.geometry import wrap_angle
from sgi.scene import PlanStep, load_scene
from sgi.types import GoalResult, GroundedTarget, Plan, Point, Subtask
from tests.helpers import floor_observation, scene_dict, write_scene

TH = Thresholds()     # match_tolerance_m 0.3, default_match_radius_m 0.75, personal_gap_m 0.3, radii 0.25 and 0.30
UP = math.pi / 2      # world +Y: the robot's yaw
def box(x0, y0, x1, y1): return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

@pytest.fixture(scope="module")
def hall(tmp_path_factory):       # returns (observation, truth, e0); tests derive variants with model_copy and replace
    s = scene_dict()
    gt, ep = s["ground_truth"], s["episodes"][0]["ground_truth"]
    gt["people"] = [{"id": i, "x": x, "y": y, "yaw": UP} for i, x, y in [("p0", 8.0, 1.0), ("p1", 8.5, 1.0), ("p2", 3.9, 5.0)]]
    gt["targets"] += [{"id": "t1", "kind": "object", "label": "crate", "anchor": [5.0, 6.0, 0.4]},
                      {"id": "t2", "kind": "object", "label": "cone", "anchor": [6.0, 1.0, 0.3], "match_radius": 0.4},
                      {"id": "t3", "kind": "person", "label": "person in red", "anchor": [8.0, 1.0, 1.3], "person_id": "p0"}]
    gt["obstacles"].append({"polygon": box(2.4, 4.9, 2.5, 5.1)})             # a pillar on the left edge of t0's region
    ep["target_ids"] = ["t0", "t1", "t2", "t3"]
    ep["acceptable_regions"] = [
        {"target_id": "t0", "polygon": box(2.4, 4.4, 3.6, 5.4), "heading": {"face_point": [3.0, 6.0]}},
        {"target_id": "t1", "polygon": box(4.4, 4.4, 5.6, 5.4), "heading": {"yaw": math.radians(-160.0)}, "tolerance_deg": 20.0}]
    ep["forbidden_regions"] = [{"kind": kind, "target_id": tid, "polygon": box(*corners)} for kind, tid, corners in [
        ("ahead_in_line", "t0", (2.4, 5.2, 3.6, 5.8)), ("ahead_in_line", "t0", (2.4, 5.8, 3.6, 6.4)),
        ("group_space", "t1", (6.0, 3.0, 7.0, 4.0))]]
    inputs, truth = load_scene(write_scene(tmp_path_factory.mktemp("scene"), s, floor_observation()))
    return inputs.observation, truth, truth.episodes["e0"]

def test_match_target(hall):
    _, truth, ep = hall
    cases = [((1.0, 6.5), "t0"), ((1.0, 5.75), "t0"), ((1.0, 5.65), None),   # in the footprint, 0.25 m and 0.35 m off it
             ((5.7, 6.0), "t1"), ((5.8, 6.0), None),       # default radius 0.75; the 0.3 m tolerance is not added to it
             ((6.35, 1.0), "t2"), ((6.45, 1.0), None),     # its own match_radius of 0.4
             ((7.9, 1.0), "t3"), ((8.4, 1.0), None)]       # 0.4 m from the anchor, but p1 is the nearest person
    for (x, y), expected in cases:
        assert ev.match_target((x, y, 1.0), truth, ep, TH) == expected, (x, y)
    wide = Thresholds(match_tolerance_m=0.4, default_match_radius_m=0.9)     # both limits come from th
    assert [ev.match_target((x, y, 1.0), truth, ep, wide) for x, y in [(1.0, 5.65), (5.8, 6.0)]] == ["t0", "t1"]

DESK, CRATE, NOWHERE = (1.0, 6.0, 1.0), (5.0, 6.0, 0.4), (3.0, 9.0, 1.0)     # lifted target points: on t0, on t1, on neither

def result(x, y, theta, world=DESK, **fields):
    """An ok GoalResult for the world goal (x, y, theta); ROBOT is at (1, 2) facing world +Y. pose_world is deliberately wrong."""
    target = GroundedTarget(point=Point(x=500, y=500), pixel=(320.0, 240.0), world=world)
    pose = {"pose_robot": (y - 2.0, 1.0 - x, wrap_angle(theta - UP)), "pose_world": (9.0, 9.0, 0.0)}
    return GoalResult(**{"status": "ok", "target": target, **pose, **fields})

@pytest.mark.parametrize("x, y, theta_deg, world, expected, distance", [
    # expected is (matched_target_id, reachable, in_region, heading_ok, success); distance is distance_to_region_m
    (3.0, 5.0, 90.0, DESK, ("t0", True, True, True, True), 0.0),            # in t0's region, facing its face_point
    (3.0, 5.0, 130.0, DESK, ("t0", True, True, True, True), 0.0),           # 40 degrees off; tolerance_deg defaults to 45
    (3.0, 5.0, 40.0, DESK, ("t0", True, True, False, False), 0.0),          # 50 degrees off
    (3.0, 4.0, 90.0, DESK, ("t0", True, False, False, False), 0.4),         # short of the region, same heading
    (2.6, 5.0, 90.0, DESK, ("t0", False, True, True, False), 0.0),          # 0.1 m from the pillar
    (3.0, 5.0, 90.0, NOWHERE, (None, True, False, False, False), None),     # no region counts without a match
    (5.0, 5.0, -170.0, CRATE, ("t1", True, True, True, True), 0.0),         # t1's region: yaw -160 within 20 degrees
    (5.0, 5.0, 170.0, CRATE, ("t1", True, True, False, False), 0.0),        # 30 degrees off, through the +-pi seam
    (5.0, 5.0, -170.0, DESK, ("t0", True, False, False, False), 1.4),       # the same goal, target point on t0
], ids=["facing", "off_40", "off_50", "outside", "obstacle", "wrong_target", "seam_10", "seam_30", "other_target"])
def test_success_components(hall, x, y, theta_deg, world, expected, distance):
    s = ev.score(result(x, y, math.radians(theta_deg), world=world), *hall, TH)
    assert (s.matched_target_id, s.reachable, s.in_region, s.heading_ok, s.success) == expected
    assert s.target_correct is (expected[0] is not None)
    assert s.distance_to_region_m == (None if distance is None else pytest.approx(distance))

def test_failed_result_scores_every_component_false(hall):
    for bad in (result(3.0, 5.0, UP, status="parse_failed"), result(3.0, 5.0, UP, pose_robot=None), GoalResult(status="no_target")):
        assert ev.score(bad, *hall, TH) == ev.Score(success=False, target_correct=False)
    assert not ev.score(result(3.0, 5.0, UP, target=None), *hall, TH).target_correct       # ok status, no target

def test_body_gap_and_personal_space_flag(hall):                 # a flag only: it never changes success
    near = ev.score(result(3.1, 5.0, UP), *hall, TH)             # 0.8 m from p2, centre to centre: 0.8 - 0.25 - 0.30
    assert near.min_body_gap_m == pytest.approx(0.25) and (near.personal_space_intrusion, near.success) == (True, True)
    assert ev.score(result(3.1, 5.0, UP), *hall, Thresholds(personal_gap_m=0.2)).personal_space_intrusion is False
    clear = ev.score(result(3.0, 5.0, UP), *hall, TH)            # 0.9 m
    assert clear.min_body_gap_m == pytest.approx(0.35) and clear.personal_space_intrusion is False

def test_forbidden_kinds(hall):
    hit = ev.score(result(3.0, 5.3, UP), *hall, TH)              # in t0's region and in the first ahead_in_line piece only
    assert hit.forbidden == {"ahead_in_line": True, "group_space": False} and hit.success     # reported, never part of success
    miss = ev.score(result(3.0, 5.0, UP, world=NOWHERE), *hall, TH)          # measured whatever target matched, or none
    assert miss.forbidden == {"ahead_in_line": False, "group_space": False}  # a kind with no region (audience_view): no key

def test_region_without_target_id_and_absent_ground_truth(hall):
    obs, truth, ep = hall
    region = ep.acceptable_regions[0].model_copy(update={"target_id": None})
    solo = ep.model_copy(update={"target_ids": ["t0"], "acceptable_regions": [region]})     # the usual one-target episode
    on, off = (ev.score(result(3.0, 5.0, UP, world=w), obs, truth, solo, TH) for w in (DESK, CRATE))
    assert on.success and (off.target_correct, off.in_region, off.distance_to_region_m) == (False, False, None)   # t1 is not listed
    bare = solo.model_copy(update={"acceptable_regions": None, "forbidden_regions": None})  # e.g. a held-out situation
    s = ev.score(result(3.0, 5.0, UP), obs, replace(truth, people=()), bare, TH)            # every other field at its default
    assert s == ev.Score(success=False, target_correct=True, matched_target_id="t0", reachable=True)

NAV, ACT, BACK, LOST = ("navigate", "t0"), ("act", None), ("navigate", "ungrounded"), ("navigate", "unmatched")
GO, DO, HOME = PlanStep(type="navigate", target_ids=["t0", "t9"]), PlanStep(type="act"), PlanStep(type="navigate")
MAYBE, DETOUR = PlanStep(type="act", optional=True), PlanStep(type="navigate", optional=True)
REFERENCE = [GO, DO, HOME, MAYBE]

def test_reduce_plan_merges_adjacent_identical_entries():
    steps = [NAV, ACT, ACT, BACK, BACK, LOST, ACT]               # an extra act; BACK and LOST differ, so both stay
    plan = Plan(subtasks=[Subtask(type=kind, description="d") for kind, _ in steps])
    assert ev.reduce_plan(plan, [m for _, m in steps]) == [NAV, ACT, BACK, LOST, ACT]
    with pytest.raises(ValueError): ev.reduce_plan(plan, ["t0"])             # one entry per subtask is required

@pytest.mark.parametrize("predicted, reference, expected", [     # expected is (exact match, navigate recall)
    ([("navigate", "t9"), ACT, BACK, ACT], REFERENCE, (True, 1.0)),          # target_ids is any-of
    ([NAV, ACT, BACK], REFERENCE, (True, 1.0)),                  # the optional step is absent
    ([NAV, ACT], REFERENCE, (False, 0.5)),                       # a navigate step is missing
    ([NAV, ACT, LOST], REFERENCE, (False, 0.5)),                 # only "ungrounded" fits a step without ids
    ([BACK, ACT, NAV], REFERENCE, (False, 1.0)),                 # recall ignores order
    ([NAV, ACT, BACK], [GO, DO, DETOUR, MAYBE, HOME], (True, 1.0)),          # BACK is HOME, not the optional DETOUR
    ([NAV, ACT], [GO, DO, DO], (True, 1.0)),                     # adjacent identical reference steps merge
    ([NAV, ACT], [GO, DO, GO], (False, 0.5)),                    # a predicted entry answers one reference step
    ([ACT], [DO], (True, 1.0)),                                  # no navigate step to recall
    ([NAV, ACT, BACK], [GO, DO, DETOUR, DO, HOME], (True, 1.0)),             # without DETOUR the two acts are one step
    ([NAV], [GO, MAYBE, GO], (True, 1.0)),                       # so are the two GO steps, which the recall counts once
    ([NAV, ("navigate", "t9")], [GO, MAYBE, GO], (True, 1.0)),   # two different entries still answer one GO step each
    ([NAV], [GO, MAYBE, GO, HOME], (False, 0.5)),                # HOME has other targets: still a step, and missing
])
def test_decomposition_accuracy(predicted, reference, expected):
    assert ev.decomposition_accuracy(predicted, reference) == expected

def test_tour_frame_correct():
    shown = [0, 4, 8, 12]
    cases = [(4, [4, 5], True), (8, [4, 5], False),              # a reference frame is on show: only it is correct
             (8, [6, 7], True), (4, [6, 7], False),              # none is: 8 is 1 from frame 7, 4 is 2 from frame 6
             (4, [6], True), (8, [6], True), (12, [6], False)]   # 4 and 8 tie at 2 from frame 6
    for chosen, truth_frames, expected in cases:
        assert ev.tour_frame_correct(chosen, shown, truth_frames) is expected, (chosen, truth_frames)
    with pytest.raises(ValueError): ev.tour_frame_correct(4, shown, [])      # no reference frames: nothing to measure
