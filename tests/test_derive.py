import dataclasses
import math
import numpy as np
import pytest

import sgi.synth.generate as gen
from sgi.config import load_config
from sgi.evaluate import score
from sgi.geometry import RobotPose, make_poses, wrap_angle
from sgi.gt import derive
from sgi.gt import regions as rules
from sgi.polygons import distance_to_boundary, grid_points_in, oriented_rectangle
from sgi.scene import Person, Region, Target, load_scene, save_scene
from sgi.synth.render import Box, Figure
from sgi.types import GoalResult
from tests.helpers import ROBOT, rendered_observation, scene_dict, write_scene

CFG = load_config()
TH = CFG.thresholds
DESK = Box(cx=1.0, cy=6.5, yaw=-math.pi / 2, sx=1.0, sy=1.0, sz=1.5)   # scene_dict's t0: front face on y = 6.0
SIDE = RobotPose(-4.0, 4.0, 0.0)                                        # looks along +x, across LINE
LINE = [("p0", 1.0, 5.3, "g0"), ("p1", 1.0, 4.6, "g0"), ("p2", 1.0, 3.9, "g0"), ("q0", 3.0, 2.3, "g1"),
        ("q1", 3.9, 1.5, "g1"), ("b0", 3.4, 4.9, None), ("b1", -1.0, 0.0, None)]   # SIDE sees b0 exactly behind p1
FRONT = [[0.4, 4.4], [1.6, 4.4], [1.6, 5.4], [0.4, 5.4]]                # the no_one region of t0

def _region(polygon, **extra):                                          # a region dict that faces t0's anchor
    return {"polygon": polygon, "heading": {"face_point": [1.0, 6.0]}, **extra}

def _scene(tmp_path, people=(), robot=ROBOT, **episode_gt):             # people: (id, x, y, group), rendered and in truth
    obs = rendered_observation([DESK], [Figure(x, y, math.pi / 2) for _, x, y, _ in people], robot)
    s = scene_dict()
    s["observation"].update(robot_pose_world=dataclasses.asdict(robot), T_world_camera=obs.camera.T_world_camera.tolist())
    s["ground_truth"]["people"] = [{"id": i, "x": x, "y": y, "yaw": math.pi / 2, "group": g} for i, x, y, g in people]
    s["episodes"][0]["ground_truth"].update(episode_gt)
    (folder := tmp_path / f"scene{len(list(tmp_path.iterdir()))}").mkdir()
    return load_scene(write_scene(folder, s, obs))

def test_oracle_point_target_and_path(tmp_path):
    inputs, truth = _scene(tmp_path)
    obs, t0 = inputs.observation, truth.targets["t0"]
    assert derive.oracle_point(obs, (1.0, 0.0, 1.0)) is None            # behind the camera
    assert derive.oracle_point(obs, (8.0, 6.0, 1.0)) is None            # projects to u = 1386, outside the image
    target = derive.oracle_target(obs, t0, TH)
    assert (target.point.x, target.point.y) == (500, 147) and target.pixel == pytest.approx((320.0, 70.56))   # the rounded point
    assert target.world == pytest.approx((1.0, 6.0, 1.0), abs=0.02)     # on the desk front
    inside = derive.oracle_target(obs, t0.model_copy(update={"anchor": (1.0, 6.5, 1.0)}), TH)
    assert inside.world == pytest.approx((1.0, 6.0, 1.02), abs=0.02)    # lifted from depth: the desk front, not the anchor
    assert derive.oracle_target(obs, t0.model_copy(update={"anchor": (10.0, 40.0, 3.0)}), TH) is None   # sky: no depth
    assert tuple(derive.oracle_path(obs, (1.0, 4.9), TH)) == pytest.approx((1.0, 4.9, 0.0), abs=0.02)

def test_representative_point_is_the_deepest_lattice_point_that_passes(tmp_path):
    inputs, truth = _scene(tmp_path)
    square = Region(**_region([(0.6, 4.0), (1.4, 4.0), (1.4, 4.8), (0.6, 4.8)]))
    p = derive.representative_point(square, inputs.observation, truth, TH)
    assert isinstance(p, tuple) and p == pytest.approx((1.0, 4.4))      # 0.40 m from every edge; (1.0, 4.45), 0.35 m, is a grid step nearer the anchor, so as near, and farther from the robot
    strip = Region(**_region([(0.575, 4.825), (0.775, 5.025), (1.625, 4.175), (1.425, 3.975)]))
    assert derive.representative_point(strip, inputs.observation, truth, TH) == pytest.approx((0.85, 4.8))  # 13 tie at 0.141 m; beside them, 0.106 m: of the four within a grid step of the nearest the anchor, the nearest the robot
    for height, polygon in ((0.3, FRONT), (0.1, [[0.6, 5.0], [1.4, 5.0], [1.4, 5.15], [0.6, 5.15]])):   # under a platform the map
        raised = rendered_observation([DESK, Box(cx=1.0, cy=4.6, yaw=0.0, sx=1.6, sy=2.0, sz=height)])   # lacks, points come back off
        assert derive.representative_point(Region(**_region(polygon)), raised, truth, TH) is None        # the floor, or outside the strip
    inputs, truth = _scene(tmp_path, [("p0", 1.0, 4.9, None)])          # stands on the deepest points of FRONT
    q = derive.representative_point(Region(**_region(FRONT)), inputs.observation, truth, TH)
    assert q is not None and math.dist(q, (1.0, 4.9)) > 0.45            # shallower: in a free cell, off the 0.55 m disc

STEP = TH.grid_resolution_m
HALF_RING_CENTROID_M = 4 * (1.2 ** 3 - 0.5 ** 3) / (3 * math.pi * (1.2 ** 2 - 0.5 ** 2))   # of a half ring from 0.5 to 1.2 m: 0.572 m out on its axis

def _equals(region, obs, truth):                                        # the lattice points that pass and are no more than a grid step shallower
    grid, mask = derive._reach(obs, truth, TH)                          # than the deepest of them, each with its depth. Which points pass is the
    depth = {(float(x), float(y)): distance_to_boundary((x, y), region.polygon)   # module's own test: the choice among them is what is tested
             for x, y in grid_points_in(region.polygon, STEP) if derive._passes((x, y), region, obs, grid, mask, TH)}
    return {p: d for p, d in depth.items() if d >= max(depth.values()) - STEP - 1e-9}

def _least(points, key):                                                # those of the points with the least key, in the grid's order
    least = min(key(p) for p in points)
    return sorted(p for p in points if key(p) < least + 1e-9)

def _near(points, to):                                                  # those of the points no more than a grid step farther from `to` than the nearest
    least = min(math.dist(p, to) for p in points)
    return [p for p in points if math.dist(p, to) <= least + STEP + 1e-9]

def _behind(p, last, yaw):                                              # p as seen from `last`, who faces along yaw: metres behind, metres to the side
    dx, dy = p[0] - last[0], p[1] - last[1]
    return -(dx * math.cos(yaw) + dy * math.sin(yaw)), abs(dx * math.sin(yaw) - dy * math.cos(yaw))

def _generated(tmp_path, recipe, n_people, seed):                       # one placeholder layout, under the generator's random rigid transform
    layout = gen._layout(np.random.default_rng(seed), (recipe, n_people, None, 0, False))
    scene, rgb, depth = gen.build_scene(layout, layout.name)
    save_scene(tmp_path / layout.name, scene, rgb, depth)
    inputs, truth = load_scene(tmp_path / layout.name)
    return inputs, truth, derive.prepare(inputs, truth, CFG)["e0"].gt.acceptable_regions[0]

def test_in_a_half_ring_the_point_is_straight_behind_the_one_it_is_drawn_round(tmp_path):
    inputs, truth = _scene(tmp_path)                                    # bare floor before the desk: every lattice point in view passes
    obs, last = inputs.observation, (1.13, 4.62)
    for deg in (20, 65, 100, 160, -70, -135):                           # the way the one at `last` faces: right round, the last along the grid's diagonal
        yaw = math.radians(deg)
        ahead = truth.targets["t0"].model_copy(update={"anchor": (last[0] + 3 * math.cos(yaw), last[1] + 3 * math.sin(yaw), 1.0)})
        (ring,), _ = rules.region_waiting_line(ahead, [Person(id="p0", x=last[0], y=last[1], yaw=yaw, group="g0")])
        p, equals = derive.representative_point(ring, obs, truth, TH), _equals(ring, obs, truth)
        on_axis = _least(equals, lambda q: _behind(q, last, yaw)[1])    # a heading that is a direction: of the equals, the least to the side of the axis
        centroid = (last[0] - HALF_RING_CENTROID_M * math.cos(yaw), last[1] - HALF_RING_CENTROID_M * math.sin(yaw))
        assert (len(on_axis) > 1) == (deg == -135) and [p] == _least(on_axis, lambda q: math.dist(q, centroid)), deg   # and of two as far to the side, the nearer the centroid
        back, side = _behind(p, last, yaw)                              # the equals lie 0.80 to 0.90 m out there, and a lattice point is within STEP / sqrt(2) of any place
        assert 0.8 - STEP < back < 0.9 + STEP and side < STEP / math.sqrt(2), deg
    # Over 300 half rings at random the point is 0.007 m to the side at the median and 0.024 m at most. Nearest the centroid, it was 0.14 m at most.
    ahead = truth.targets["t0"].model_copy(update={"anchor": (0.975, 7.6, 1.0)})    # along the grid, the axis midway between two of its columns
    (ring,), _ = rules.region_waiting_line(ahead, [Person(id="p0", x=0.975, y=4.6, yaw=math.pi / 2, group="g0")])
    on_axis = _least(_equals(ring, obs, truth), lambda q: abs(q[0] - 0.975))
    assert _least(on_axis, lambda q: math.dist(q, (0.975, 4.6 - HALF_RING_CENTROID_M))) == [pytest.approx((0.95, 3.8)), pytest.approx((1.0, 3.8))]
    assert derive.representative_point(ring, obs, truth, TH) == pytest.approx((1.0, 3.8))   # as far to the side and from the centroid: the one on ROBOT's side, second in the grid's order
    inputs, truth, region = _generated(tmp_path, "waiting_line", 2, seed=11)
    last = max(((q.x, q.y) for q in truth.people), key=lambda q: math.dist(q, truth.targets["t0"].anchor[:2]))
    back, side = _behind(region.representative_point, last, region.heading.yaw)
    assert 0.8 - STEP < back < 0.9 + STEP and side < STEP               # a generated scene, its people in the picture; the grid's order gave 0.61 m to the side

MIRRORED = RobotPose(-2.0, 5.0, 0.0)                                    # ROBOT, mirrored in the diagonal through (0.5, 4.5): (x, y) goes to (y - 4, x + 4)

def _turn(p, who):                                                      # degrees from the way `who` faces to where p lies from them
    return math.degrees(wrap_angle(math.atan2(p[1] - who.y, p[0] - who.x) - who.yaw))

def test_in_a_band_round_people_the_point_is_by_the_one_to_face_and_on_the_side_of_the_robot(tmp_path):
    group = [("p0", 0.5, 4.5, "g0"), ("p1", 1.6, 4.8, "g0"), ("p2", 0.8, 5.6, "g0")]    # p1 and p2 mirror one another in that diagonal
    points = []
    for robot in (ROBOT, MIRRORED):                                     # the same people, come upon from either side
        inputs, truth = _scene(tmp_path, group, robot)
        obs, who = inputs.observation, truth.person("p0")
        asked = Target(id="t1", kind="person", label="person in red", anchor=(who.x, who.y, 1.3), person_id="p0")
        (band,), _ = rules.region_conversation_group(asked, truth.group("g0"))
        assert band.heading.face_point == (0.5, 4.5)                    # a heading that faces a point: the equals within a grid step of the nearest to it are equally near
        by_them = _near(_equals(band, obs, truth), (who.x, who.y))
        assert len(by_them) > 20 and max(math.dist(q, (who.x, who.y)) for q in by_them) < 0.8 + 2 * STEP   # an arc of them round p0, 0.80 to 0.87 m off
        points.append(derive.representative_point(band, obs, truth, TH))
        assert [points[-1]] == _least(by_them, lambda q: math.dist(q, (robot.x, robot.y)))           # of these, the nearest the robot
    assert points == [pytest.approx((0.55, 3.65)), pytest.approx((-0.35, 4.65))]   # either side of the diagonal, each on its robot's: the grid's order has no say.
    # Not mirror images to the grid step: the desk hides from ROBOT the deepest floor of the band, which MIRRORED sees, so the equals of the two differ.
    assert [(q[1] - 4.5) - (q[0] - 0.5) > 0 for q in points] == [False, True]      # below the diagonal with ROBOT, above it with MIRRORED
    inputs, truth, region = _generated(tmp_path, "conversation_group", 3, seed=7)
    obs, who, p = inputs.observation, truth.person(truth.targets["t0"].person_id), region.representative_point
    by_them = _near(_equals(region, obs, truth), (who.x, who.y))
    assert [p] == _least(by_them, lambda q: math.dist(q, (obs.robot.x, obs.robot.y)))
    turn, robot_turn = _turn(p, who), _turn((obs.robot.x, obs.robot.y), who)
    assert 45 < abs(turn) < 135 and turn * robot_turn > 0               # beside them (+76 degrees) on the side the robot is on (+46), 0.84 m off;
                                                                        # nearest alone gave a point at their back (+172), the grid's order one 1.75 m off, across the group

def test_in_a_rectangle_the_point_stays_or_moves_towards_what_is_faced_and_keeps_its_depth_to_a_grid_step(tmp_path):
    inputs, truth = _scene(tmp_path)
    square, turned = [(0.6, 4.0), (1.4, 4.0), (1.4, 4.8), (0.6, 4.8)], oriented_rectangle((1.0, 6.0), -1.9, 0.6, 1.6, 1.2)
    for polygon in (FRONT, square, turned):                             # each faces t0's anchor, (1.0, 6.0)
        region = Region(**_region(polygon))
        p, equals = derive.representative_point(region, inputs.observation, truth, TH), _equals(region, inputs.observation, truth)
        deepest = min(equals, key=lambda q: (-round(equals[q], 9), q))  # by the grid's order among the deepest: what the point was
        assert p in equals                                              # no more than a grid step shallower, and never deeper
        assert p == deepest or math.dist(p, (1.0, 6.0)) < math.dist(deepest, (1.0, 6.0))
        assert [p] == _least(_near(equals, (1.0, 6.0)), lambda q: math.dist(q, (ROBOT.x, ROBOT.y)))   # of the equals no more than a grid step farther from the anchor than the nearest, the nearest the robot

def test_a_region_with_one_point_that_passes_gives_it_and_a_region_with_none_gives_none(tmp_path):
    inputs, truth = _scene(tmp_path)
    cell = [(0.96, 4.46), (1.04, 4.46), (1.04, 4.54), (0.96, 4.54)]     # holds one lattice point
    hidden = [(0.6, 7.4), (1.4, 7.4), (1.4, 8.2), (0.6, 8.2)]           # floor the desk hides from ROBOT
    for heading in ({"face_point": (1.0, 6.0)}, {"yaw": 0.3}):
        one = derive.representative_point(Region(polygon=cell, heading=heading), inputs.observation, truth, TH)
        assert one == pytest.approx((1.0, 4.5))
        assert derive.representative_point(Region(polygon=hidden, heading=heading), inputs.observation, truth, TH) is None

def test_prepare_fills_regions_and_the_standard_goal_passes_from_the_front(tmp_path):
    inputs, truth = _scene(tmp_path)
    et = derive.prepare(inputs, truth, CFG)["e0"]
    (region,) = et.gt.acceptable_regions
    assert region.target_id == "t0" and region.representative_point == pytest.approx((1.0, 4.9))   # the middle of five that tie at 0.50 m: (1.0, 4.95) is a grid step nearer the anchor, so as near, and farther from the robot
    assert et.gt.forbidden_regions == [] and truth.episodes["e0"].acceptable_regions is None        # truth unchanged
    assert et.standard_goal_ok == {"t0": True} and derive.reference_people_matter(et, None) is False
    assert et.problems == () and et.relevant_people == () and et.derived_factors.group_size == 0
    staffed = derive.prepare(*_scene(tmp_path, [("s0", 1.0, 6.55, None)]), CFG)["e0"]      # someone in no group, 0.55 m behind the desk's front face
    assert staffed.gt.acceptable_regions == et.gt.acceptable_regions and staffed.relevant_people == ("s0",)   # near, and shapes nothing: one polygon, one point

def test_prepare_waiting_line_seen_from_the_side(tmp_path):
    inputs, truth = _scene(tmp_path, LINE, SIDE, situation="waiting_line", group_at_target="g0",
                           factors={"density": "low", "group_size": 5})
    et = derive.prepare(inputs, truth, CFG)["e0"]
    assert et.gt.acceptable_regions[0].representative_point == pytest.approx((1.0, 3.1))   # 0.80 m straight behind p2: a grid step nearer the centroid than the deepest
    assert [(r.kind, r.target_id) for r in et.gt.forbidden_regions] == [("ahead_in_line", "t0")] * 3
    assert et.standard_goal_ok == {"t0": False} and derive.reference_people_matter(et, "t0") is True
    assert et.derived_factors.model_dump(exclude={"view_angle_deg"}) == dict(density=None, group_size=3, bystanders=2, other_groups=1)
    assert et.factors.model_dump(exclude={"view_angle_deg"}) == dict(density="low", group_size=5, bystanders=2, other_groups=1)   # entered values kept
    assert et.relevant_people == ("p0", "p1", "p2", "q0", "q1", "b0")   # q0: acceptable region only; b0: that and the corridor; q1: group
    assert et.hidden_relevant_people == ("b0",) and et.problems == ()   # b1 is 3.18 m from the nearest region

def test_the_strip_of_a_region_nearest_a_person_is_in_the_region_and_cannot_be_reached(tmp_path):
    inputs, truth = _scene(tmp_path, LINE, SIDE, situation="waiting_line", group_at_target="g0")
    obs, et = inputs.observation, derive.prepare(inputs, truth, CFG)["e0"]
    target = derive.oracle_target(obs, truth.targets["t0"], TH)
    def stop(metres, turn_deg):                         # that far from p2, the last in line, turned off straight behind them
        a = -math.pi / 2 + math.radians(turn_deg)
        _, pose = make_poses(obs.robot, (1.0 + metres * math.cos(a), 3.9 + metres * math.sin(a)), math.pi / 2)
        sc = score(GoalResult(status="ok", pose_robot=pose, target=target), obs, truth, et.gt, TH)
        return sc.target_correct, sc.in_region, sc.heading_ok, sc.reachable, sc.success
    turns = (-80, -40, 0, 40, 80)
    assert TH.person_radius_m + TH.robot_radius_m == pytest.approx(0.55)    # how near a person's centre the scorer lets the robot's come
    assert {stop(0.49, t) for t in turns} == {(True, False, False, False, False)}   # nearer than the region begins, at 0.5 m
    assert {stop(0.51, t) for t in turns} == {(True, True, True, False, False)}     # in it, facing as the line does, and nearer than the robot can come
    assert {stop(0.60, t) for t in turns} == {(True, True, True, True, True)}       # from 0.59 m on p2 no longer stands in the way

def test_prepare_reports_problems(tmp_path):
    hidden = _region([[0.6, 7.4], [1.4, 7.4], [1.4, 8.2], [0.6, 8.2]])  # floor the desk hides from ROBOT
    et = derive.prepare(*_scene(tmp_path, acceptable_regions=[hidden]), CFG)["e0"]
    assert et.problems == ("e0: region 0: no representative point passes the oracle path",)
    blocked = _region(FRONT, representative_point=[1.1, 4.9])           # 0.4 m from p0: inside its 0.55 m disc
    et = derive.prepare(*_scene(tmp_path, [("p0", 1.5, 4.9, None)], acceptable_regions=[blocked]), CFG)["e0"]
    assert et.problems == ("e0: region 0: supplied representative point fails the oracle path",)
    assert et.gt.acceptable_regions[0].representative_point == (1.1, 4.9)                # kept, never replaced
    et = derive.prepare(*_scene(tmp_path, LINE, SIDE, situation="waiting_line"), CFG)["e0"]     # no group_at_target
    assert et.problems == ("e0: no region (waiting_line needs group_at_target)",)
    assert et.gt.acceptable_regions == [] and et.standard_goal_ok == {"t0": False}

def test_prepare_reports_relevant_and_hidden_people(tmp_path):
    people = [("p0", 1.6, 4.6, None), ("p1", 1.8, 5.4, None),           # on one ray from the camera at (1.0, 2.2)
              ("back", 1.0, 8.5, None), ("far", 1.0, 9.5, None)]        # 1.5 m and 2.5 m behind t0's footprint
    et = derive.prepare(*_scene(tmp_path, people), CFG)["e0"]
    assert et.relevant_people == ("p0", "p1", "back")                   # back is 2.5 m from the anchor
    assert et.hidden_relevant_people == ("p1", "back")                  # p1 is behind p0, back is behind the desk

def test_view_angle_30_degrees_off_the_front_and_a_situation_without_a_rule(tmp_path):
    off = RobotPose(-1.0, 6.0 - 2.0 * math.sqrt(3.0), 1.2)              # 4 m from the anchor, turned 9 degrees past it
    et = derive.prepare(*_scene(tmp_path, robot=off, situation="picnic", factors={"view_angle_deg": 5.0}), CFG)["e0"]   # a made-up label: every one the config names has a rule
    assert et.derived_factors.view_angle_deg == pytest.approx(30.0, abs=1e-3)
    assert et.factors.view_angle_deg == pytest.approx(30.0, abs=1e-3)   # the computed value replaces the entered 5.0
    assert et.problems == ("e0: no region (no rule for picnic)",) and et.gt.acceptable_regions == []
    et = derive.prepare(*_scene(tmp_path, robot=off, situation="audience"), CFG)["e0"]      # a rule, and nobody named as the group at the target
    assert et.problems == ("e0: no region (audience needs group_at_target)",) and et.gt.acceptable_regions == []
    assert [derive._why_no_region(s, None, "t0") for s in ("group_photo", "lift_wait", "no_one")] == [
        "group_photo needs group_at_target", "lift_wait needs group_at_target", "no_one rule gives no region for t0"]
    assert derive._why_no_region("group_photo", "g0", "t0") == "group_photo rule gives no region for t0"    # the group is named, and its picture leaves no place

def test_people_matter_when_the_supplied_region_excludes_the_standard_goal(tmp_path):
    side = _region([[1.6, 5.0], [2.2, 5.0], [2.2, 5.6], [1.6, 5.6]])    # the standard goal (1.0, 5.0) is outside it
    people = [("p0", 4.0, 6.0, None), ("p1", -2.0, 6.0, None)]          # the only people, 3 m from the anchor
    et = derive.prepare(*_scene(tmp_path, people, situation="audience", acceptable_regions=[side]), CFG)["e0"]
    assert et.problems == () and et.standard_goal_ok == {"t0": False}
    two = dataclasses.replace(et, standard_goal_ok={"t0": False, "t1": True})            # as if a second target were fine
    assert [derive.reference_people_matter(two, m) for m in ("t0", "t1", None)] == [True, False, True]   # None: the first id

KIOSK = Box(cx=1.7, cy=6.5, yaw=-math.pi / 2, sx=1.0, sy=0.4, sz=1.5)  # t1, against the desk's side: its front is on y = 6.0 too
RIGHT = RobotPose(3.0, 2.5, 2.0)                                        # sees both fronts, from the kiosk's side

def _two_targets(tmp_path, target_ids):                                 # the desk t0 and the kiosk t1, both accepted by e0
    obs = rendered_observation([DESK, KIOSK], (), RIGHT)
    s, footprint = scene_dict(), [[1.5, 6.0], [1.9, 6.0], [1.9, 7.0], [1.5, 7.0]]
    s["observation"].update(robot_pose_world=dataclasses.asdict(RIGHT), T_world_camera=obs.camera.T_world_camera.tolist())
    s["ground_truth"]["targets"].append({"id": "t1", "kind": "place", "label": "ticket kiosk", "anchor": [1.7, 6.0, 1.0],
                                         "footprint": footprint, "front_yaw": -math.pi / 2})   # anchor 0.2 m from t0's footprint
    s["ground_truth"]["obstacles"].append({"polygon": footprint})
    s["episodes"][0]["ground_truth"]["target_ids"] = target_ids
    (folder := tmp_path / f"scene{len(list(tmp_path.iterdir()))}").mkdir()
    return load_scene(write_scene(folder, s, obs))

def test_each_target_is_scored_against_its_own_regions_in_either_order(tmp_path):
    # t1's anchor is within match_tolerance_m of t0's footprint, so the scorer given both ids matches whichever comes first.
    # t1's standard goal (2.05, 5.06) is inside t1's region and 0.45 m outside t0's, which ends at x = 1.6.
    for target_ids in (["t0", "t1"], ["t1", "t0"]):
        et = derive.prepare(*_two_targets(tmp_path, target_ids), CFG)["e0"]
        assert et.problems == () and et.standard_goal_ok == {"t0": True, "t1": True}
        assert derive.reference_people_matter(et, "t1") is False        # nobody is in the scene
        assert et.gt.target_ids == target_ids and [r.target_id for r in et.gt.acceptable_regions] == target_ids   # gt kept whole

def test_an_episode_without_a_target_is_a_problem_and_not_a_crash(tmp_path):
    inputs, truth = _scene(tmp_path, LINE, group_at_target="g0")
    def prepared(**update):                                             # emptied here, not in the file: the loader may come to refuse it
        bare = truth.episodes["e0"].model_copy(update={"target_ids": [], **update})
        return derive.prepare(inputs, dataclasses.replace(truth, episodes={"e0": bare}), CFG)["e0"]
    et = prepared()
    assert et.problems == ("e0: no target_ids",) and et.gt.acceptable_regions == [] and et.standard_goal_ok == {}
    assert et.derived_factors.model_dump() == dict(density=None, group_size=0, bystanders=2, other_groups=2, view_angle_deg=None)
    assert et.relevant_people == () and derive.reference_people_matter(et, None) is True   # g0 is at no target; no goal can pass
    hidden = Region(**_region([[0.6, 7.4], [1.4, 7.4], [1.4, 8.2], [0.6, 8.2]]))            # a supplied region does not excuse it
    et = prepared(acceptable_regions=[hidden])
    assert et.problems == ("e0: no target_ids", "e0: region 0: no representative point passes the oracle path")
    assert derive.reference_people_matter(et, None) is True

def _with_roles(truth, **roles):                                        # the same truth, the named people given those roles
    return dataclasses.replace(truth, people=tuple(p.model_copy(update={"role": roles[p.id]}) if p.id in roles else p
                                                   for p in truth.people))

def test_a_person_with_a_role_is_not_a_bystander(tmp_path):
    inputs, truth = _scene(tmp_path, LINE, SIDE, situation="waiting_line", group_at_target="g0")
    plain = derive.prepare(inputs, truth, CFG)["e0"].derived_factors
    assert plain.bystanders == 2
    extra = truth.people[-1].model_copy(update={"id": "s0", "role": "staff"})
    factors = derive.prepare(inputs, dataclasses.replace(truth, people=(*truth.people, extra)), CFG)["e0"].derived_factors
    assert factors == plain                                             # bystanders, group_size and other_groups all unchanged
    clerks = derive.prepare(inputs, _with_roles(truth, b0="staff", b1="clerk"), CFG)["e0"].derived_factors
    assert clerks.bystanders == 0 and (clerks.group_size, clerks.other_groups) == (plain.group_size, plain.other_groups)

def test_a_person_with_a_role_is_not_relevant_unless_a_target_refers_to_them(tmp_path):
    inputs, truth = _scene(tmp_path, [("p0", 1.0, 4.9, None), ("s0", 1.0, 7.5, None)])      # s0: 0.5 m behind the desk
    assert derive.prepare(inputs, truth, CFG)["e0"].relevant_people == ("p0", "s0")
    staffed = _with_roles(truth, s0="staff")
    assert derive.prepare(inputs, staffed, CFG)["e0"].relevant_people == ("p0",)
    refers = staffed.targets["t0"].model_copy(update={"id": "t1", "kind": "person", "person_id": "s0", "footprint": None,
                                                      "anchor": (1.0, 7.5, 1.0)})
    both = dataclasses.replace(staffed, targets={**staffed.targets, "t1": refers})
    ep = both.episodes["e0"]
    assert [p.id for p in derive.relevant_people(both, ep, TH)] == ["p0"]
    assert [p.id for p in derive.relevant_people(both, ep.model_copy(update={"target_ids": ["t1"]}), TH)] == ["s0"]
    assert [p.id for p in derive.relevant_people(both, ep.model_copy(update={"target_ids": ["t0", "t1"]}), TH)] == ["p0", "s0"]

COUNTER = Box(cx=1.0, cy=6.2, yaw=-math.pi / 2, sx=0.4, sy=1.0, sz=1.5)     # the desk's front, 0.4 m deep: staff stand behind

def test_staff_behind_a_counter_move_no_region_and_no_representative_point(tmp_path):
    def build(staff):
        obs = rendered_observation([COUNTER], [Figure(1.0, 6.55, math.pi / 2)] if staff else [], ROBOT)
        s = scene_dict()
        s["observation"].update(T_world_camera=obs.camera.T_world_camera.tolist())
        s["ground_truth"]["obstacles"] = [{"polygon": [[0.5, 6.0], [1.5, 6.0], [1.5, 6.4], [0.5, 6.4]]}]   # the counter only
        s["ground_truth"]["people"] = [{"id": "s0", "x": 1.0, "y": 6.55, "yaw": math.pi / 2, "role": "staff"}] if staff else []
        (folder := tmp_path / f"counter{int(staff)}").mkdir()
        return derive.prepare(*load_scene(write_scene(folder, s, obs)), CFG)["e0"]
    alone, staffed = build(False), build(True)
    (a,), (b,) = alone.gt.acceptable_regions, staffed.gt.acceptable_regions
    assert a.polygon == b.polygon and a.representative_point is not None
    assert a.representative_point == b.representative_point
    distance = math.dist(b.representative_point, (1.0, 6.55))           # measured: 1.653 m from the staff member
    assert distance > TH.person_radius_m + TH.robot_radius_m            # 0.55 m: the robot's disc clear of the person's
    assert staffed.problems == () and staffed.relevant_people == ()
