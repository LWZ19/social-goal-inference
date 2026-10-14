import math
from dataclasses import replace
import numpy as np
import pytest
from sgi.check_scene import check_scene
from sgi.config import Thresholds, load_config
from sgi.geometry import (Camera, RobotPose, bearing, camera_on_robot, floor_in_view, in_image, lift, lift_all, project,
                          surface_normal_xy, wrap_angle)
from sgi.gt.derive import oracle_target, prepare, reference_people_matter
from sgi.inputs import ObservationData
from sgi.polygons import point_in_polygon
from sgi.scene import load_scene, save_scene
from sgi.stages.goal import standard_goal
from sgi.synth.generate import build_scene
from sgi.synth.recipes import CAMERA_LENS, CAMERA_MOUNT, recipe_no_one
from sgi.synth.render import Box, Figure, Room, render
from sgi.types import GroundedTarget, Point
from tests.helpers import INTRINSICS, MOUNT, ROBOT, floor_observation, rendered_observation
from tests.test_check_scene import build as desk_scene

DESK = Box(cx=1.0, cy=6.5, yaw=-math.pi / 2, sx=1.0, sy=1.0, sz=1.0)    # front face on y = 6.0, 4 m ahead of ROBOT
ROW = [Figure(1.35, y, -math.pi / 2) for y in (5.2, 4.7, 4.2)]          # a line 0.35 m beside the approach ray
PILLAR = dict(boxes=[Box(cx=1.0, cy=5.32, yaw=-math.pi / 2, sx=0.4, sy=0.6, sz=1.3)],   # its face is 0.32 m ahead of a
              robot=RobotPose(1.0, 4.8, math.pi / 2))                   # robot 1.2 m out: the clear steps are behind the robot

def low_box(cy=5.2, sx=0.24):                # 0.3 m high, on the approach ray, front face at cy - sx / 2
    return Box(cx=1.0, cy=cy, yaw=-math.pi / 2, sx=sx, sy=0.3, sz=0.3)

def target(x, y, z):                         # standard_goal reads only `world`
    return GroundedTarget(point=Point(x=500, y=450), pixel=(320.0, 216.0), world=(x, y, z))

@pytest.mark.parametrize("world, pose_world, pose_robot", [
    ((1.0, 6.0, 0.0), (1.0, 5.0, math.pi / 2), (3.0, 0.0, 0.0)),
    ((2.0, 6.0, 0.9), (1.757464, 5.029857, 1.325818), (3.029857, -0.757464, -0.244979))])
def test_bare_floor_stops_one_standoff_short_of_the_anchor_facing_it(world, pose_world, pose_robot):
    r = standard_goal(floor_observation(), target(*world), Thresholds())
    assert r.status == "ok" and r.source == "standard" and r.calls == [] and r.goal_pixel is None
    assert r.pose_world == pytest.approx(pose_world, abs=1e-5) and r.pose_robot == pytest.approx(pose_robot, abs=1e-5)
    assert r.target.world == world and not any(r.flags.model_dump().values())
    assert set(r.timings_s) == {"goal", "geometry"} and r.timings_s["goal"] == 0.0 and r.timings_s["geometry"] > 0.0

@pytest.mark.parametrize("box, th, slid", [
    (low_box(), Thresholds(), 1.35),
    (low_box(cy=5.0, sx=0.14), Thresholds(), 1.5),                      # 1.0 m before the desk: only the last step is clear
    (low_box(), Thresholds(standard_goal_clearance_m=0.3), 1.25),
    (low_box(), Thresholds(standard_goal_step_m=0.1), 1.4),
    (low_box(), Thresholds(floor_tolerance_m=0.35), 1.0)])              # the box now counts as floor
def test_low_box_before_the_target_slides_the_goal_to_the_first_clear_step(box, th, slid):
    obs = rendered_observation(boxes=[DESK, box])
    r = standard_goal(obs, target(1.0, 6.0, 0.8), th)
    assert r.pose_world == pytest.approx((1.0, 6.0 - slid, math.pi / 2), abs=1e-6) and not r.flags.standard_goal_blocked
    pts = lift_all(obs.depth, obs.camera, stride=4)
    high = pts[pts[:, 2] > th.floor_tolerance_m]
    assert np.hypot(high[:, 0] - r.pose_world[0], high[:, 1] - r.pose_world[1]).min() >= th.standard_goal_clearance_m

@pytest.mark.parametrize("scene, th", [
    (dict(boxes=[DESK], figures=ROW), Thresholds()),                    # an unbounded slide would stop behind the row
    (dict(boxes=[DESK, low_box()]), Thresholds(standard_goal_max_slide_m=0.3)),
    (dict(boxes=[DESK, low_box()]), Thresholds(standoff_m=0.6)), (PILLAR, Thresholds())])
def test_blocked_band_gives_the_standoff_pose_with_the_flag(scene, th):
    r = standard_goal(rendered_observation(**scene), target(1.0, 6.0, 0.8), th)
    assert r.status == "ok" and r.flags.standard_goal_blocked
    assert r.pose_world == pytest.approx((1.0, 6.0 - th.standoff_m, math.pi / 2), abs=1e-6)

@pytest.mark.parametrize("height, ahead, boxed, slid", [
    (2.5, 6.5, False, 1.0),                  # a low ceiling: the generator's camera sees it from 4.2 m on
    (3.0, 8.0, False, 1.0),                  # an ordinary one: from 6.3 m on
    (3.0, 8.0, True, 1.35)])                 # and under it the low box still slides the goal to the first clear step
def test_a_ceiling_in_view_above_the_stop_is_no_obstacle(height, ahead, boxed, slid):
    # The camera of the helpers, 1.2 m up and pitched 20 degrees down, never sees a ceiling; the generator's does.
    front = ROBOT.y + ahead                  # the desk's front face, straight ahead of the robot
    boxes = [replace(DESK, cy=front + 0.5)] + [low_box(cy=front - 0.8)] * boxed
    camera = Camera(**CAMERA_LENS, T_world_camera=camera_on_robot(ROBOT, **CAMERA_MOUNT))
    rgb, depth = render(camera, boxes, room=Room.rectangle(1.0, 7.0, 0.0, 10.0, 14.0, height=height))   # walls 5 m to each side
    obs = ObservationData(rgb=rgb, depth=depth, camera=camera, robot=ROBOT)
    pts = lift_all(obs.depth, obs.camera, stride=4)
    overhead = pts[pts[:, 2] > height - 0.01]
    assert np.hypot(overhead[:, 0] - 1.0, overhead[:, 1] - (front - 1.0)).min() < 0.1     # ceiling right above the stop
    r = standard_goal(obs, target(1.0, front, 0.8), Thresholds())
    assert not r.flags.standard_goal_blocked
    assert r.pose_world == pytest.approx((1.0, front - slid, math.pi / 2), abs=1e-6)

def test_robot_inside_the_standoff_keeps_its_position_and_faces_the_anchor():
    robot = RobotPose(1.0, 5.4, 0.3)                                    # 0.6 m from the anchor
    r = standard_goal(floor_observation(robot), target(1.0, 6.0, 0.0), Thresholds())
    assert r.pose_world == pytest.approx((1.0, 5.4, math.pi / 2), abs=1e-9) and not r.flags.standard_goal_blocked
    assert r.pose_robot == pytest.approx((0.0, 0.0, math.pi / 2 - 0.3), abs=1e-9)

CFG = load_config()
TH = CFG.thresholds
LINE_OF_SIGHT = dict(surface_min_points=10 ** 9)          # no surface has that many points: the rule without the normal
OLD = TH.model_copy(update=LINE_OF_SIGHT)
ASIDE = RobotPose(1.0 + 5.0 * math.sin(math.radians(58.0)), 6.0 - 5.0 * math.cos(math.radians(58.0)), math.radians(148.0))
#       5 m from the middle of the desk's front face, 58 degrees off its normal, looking at it

def fit(th):                                              # the arguments of surface_normal_xy that a Thresholds gives
    return (th.surface_radius_m, th.surface_min_points, th.surface_max_tilt, th.surface_max_rms_m, th.surface_min_spread_m,
            th.surface_max_rms_ratio)

def standing(obs, th=TH):                                 # the ground positions of what the stop keeps clear of, as the rule takes them
    pts = lift_all(obs.depth, obs.camera, stride=4)
    return pts[(pts[:, 2] > th.floor_tolerance_m) & (pts[:, 2] < obs.camera.T_world_camera[2, 3]), :2]

def gap(obs, xy):                                         # from a ground position to the nearest of them
    return float(np.hypot(*(standing(obs) - np.asarray(xy)).T).min())

def shown(obs, xy):                                       # whether the picture shows the floor there
    return floor_in_view(obs.depth, obs.camera, xy, TH.floor_tolerance_m, TH.visibility_depth_tolerance_m, TH.lift_window)

def stands(obs, xy):                                      # and under the whole robot there: also at eight points a robot radius round it
    turns = np.arange(8) * math.pi / 4
    rim = np.asarray(xy) + TH.robot_radius_m * np.stack([np.cos(turns), np.sin(turns)], axis=1)
    return all(shown(obs, spot) for spot in (xy, *rim))

def test_a_face_seen_from_the_side_gets_the_stop_before_it_not_beside_it():
    obs = rendered_observation(boxes=[DESK], robot=ASIDE)
    r, old = standard_goal(obs, target(1.0, 6.0, 0.8), Thresholds()), standard_goal(obs, target(1.0, 6.0, 0.8), Thresholds(**LINE_OF_SIGHT))
    assert r.status == "ok" and r.source == "standard" and r.pose_world == pytest.approx((1.0, 5.0, math.pi / 2), abs=0.02)
    assert {name for name, on in r.flags.model_dump().items() if on} == {"standard_goal_frontal"}
    assert old.pose_world == pytest.approx((1.0 + math.sin(math.radians(58.0)), 6.0 - math.cos(math.radians(58.0)), math.radians(148.0)), abs=1e-6)
    assert not any(old.flags.model_dump().values())       # 1 m towards the robot: 0.85 m to the side of the desk's middle

@pytest.mark.parametrize("off", [LINE_OF_SIGHT, dict(surface_radius_m=0.01), dict(surface_max_tilt=0.0), dict(surface_max_rms_m=0.0),
                                 dict(surface_min_spread_m=0.2), dict(surface_max_rms_ratio=0.0)])
def test_each_number_of_the_fit_is_read_from_the_thresholds(off):          # each of these settings leaves no surface to stand before
    obs, grounded = rendered_observation(boxes=[DESK], robot=ASIDE), target(1.0, 6.0, 0.8)
    assert surface_normal_xy(obs.depth, obs.camera, grounded.world, *fit(Thresholds())) == pytest.approx((0.0, -1.0), abs=0.02)
    assert surface_normal_xy(obs.depth, obs.camera, grounded.world, *fit(Thresholds(**off))) is None
    r, old = standard_goal(obs, grounded, Thresholds(**off)), standard_goal(obs, grounded, Thresholds(**LINE_OF_SIGHT))
    assert (r.pose_world, r.pose_robot, r.flags) == (old.pose_world, old.pose_robot, old.flags) and not r.flags.standard_goal_frontal

def test_robot_inside_the_standoff_of_a_face_keeps_its_position_too():
    obs, grounded = rendered_observation(boxes=[DESK], robot=ASIDE), target(1.0, 6.0, 0.8)     # the face is found from there
    r = standard_goal(obs, grounded, Thresholds(standoff_m=5.5))          # and the robot, 5 m from the point, is inside this standoff
    assert r.pose_world == pytest.approx((ASIDE.x, ASIDE.y, ASIDE.yaw), abs=1e-9) and not any(r.flags.model_dump().values())
    assert r.pose_robot == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)

def generated(tmp_path, view_angle_deg, kind="place", seed=0, extra=()):
    """A nobody-at-target layout of the generator, written as a scene and loaded: its observation, the reference
    target and the derived truth of its first episode. The layout is in the recipe's own frame: the target's
    ground point at the origin, a place's front along +x. `extra` are boxes drawn into the picture with it."""
    layout = recipe_no_one(np.random.default_rng(seed), view_angle_deg, kind)
    scene, rgb, depth = build_scene(replace(layout, boxes=(*layout.boxes, *extra)), "s")
    save_scene(tmp_path / "s", scene, rgb, depth)
    inputs, truth = load_scene(tmp_path / "s")
    return inputs.observation, oracle_target(inputs.observation, truth.targets["t0"], TH), prepare(inputs, truth, CFG)["e0"]

def low(x_near, x_far):                                   # a box 0.3 m high and 0.3 m wide on the line out of the place's front
    return Box(cx=(x_near + x_far) / 2, cy=0.0, yaw=0.0, sx=x_far - x_near, sy=0.3, sz=0.3)

@pytest.mark.parametrize("seed", range(5))                # a ticket counter from either side, a help desk, a print kiosk, a coffee stall
def test_a_place_seen_58_degrees_off_its_front_gets_the_stop_inside_its_region(tmp_path, seed):
    obs, grounded, et = generated(tmp_path, 58.0, seed=seed)
    (region,) = et.gt.acceptable_regions                  # 0.6 to 1.6 m out from the front, 0.6 m to each side
    new, old = standard_goal(obs, grounded, TH), standard_goal(obs, grounded, OLD)
    assert {name for name, on in new.flags.model_dump().items() if on} == {"standard_goal_frontal"}
    assert new.pose_world[:2] == pytest.approx((1.0, 0.0), abs=0.05) and abs(wrap_angle(new.pose_world[2] - math.pi)) < 0.05
    assert point_in_polygon(new.pose_world[:2], region.polygon) and not point_in_polygon(old.pose_world[:2], region.polygon)
    side = math.copysign(1.0, obs.robot.y)                # the old stop: 1 m towards the robot, 0.85 m to the side of the front's middle
    assert old.pose_world[:2] == pytest.approx((math.cos(math.radians(58.0)), side * math.sin(math.radians(58.0))), abs=0.05)
    assert not any(old.flags.model_dump().values())
    assert et.standard_goal_ok == {"t0": True} and reference_people_matter(et, None) is False

@pytest.mark.parametrize("seed", range(5))
def test_a_place_seen_head_on_keeps_its_stop_within_a_few_centimetres(tmp_path, seed):
    obs, grounded, et = generated(tmp_path, 0.0, seed=seed)
    new, old = standard_goal(obs, grounded, TH), standard_goal(obs, grounded, OLD)
    assert new.flags.standard_goal_frontal and not new.flags.standard_goal_blocked and not old.flags.standard_goal_frontal
    assert math.dist(new.pose_world[:2], old.pose_world[:2]) < 0.03 and abs(wrap_angle(new.pose_world[2] - old.pose_world[2])) < 0.03
    assert old.pose_world[:2] == pytest.approx((1.0, 0.0), abs=0.02) and et.standard_goal_ok == {"t0": True}

@pytest.mark.parametrize("seed", [1, 2])                  # from either side
def test_something_low_before_the_face_slides_the_stop_out_along_the_normal(tmp_path, seed):
    obs, grounded, _ = generated(tmp_path, 58.0, seed=seed, extra=[low(0.48, 0.78)])
    r = standard_goal(obs, grounded, TH)                  # its far face is 0.78 m out: the first step 0.4 m clear of it is the one at 1.2 m
    assert r.flags.standard_goal_frontal and not r.flags.standard_goal_blocked
    assert r.pose_world[:2] == pytest.approx((1.2, 0.0), abs=0.03) and abs(wrap_angle(r.pose_world[2] - math.pi)) < 0.05
    pts = lift_all(obs.depth, obs.camera, stride=4)
    high = pts[(pts[:, 2] > TH.floor_tolerance_m) & (pts[:, 2] < 1.5)]
    assert np.hypot(high[:, 0] - r.pose_world[0], high[:, 1] - r.pose_world[1]).min() >= TH.standard_goal_clearance_m

@pytest.mark.parametrize("seed", [1, 2])
def test_with_every_step_along_the_normal_blocked_the_line_of_sight_rule_answers(tmp_path, seed):
    obs, grounded, _ = generated(tmp_path, 58.0, seed=seed, extra=[low(0.45, 1.95)])     # it lies under every step from 1.0 to 1.5 m
    assert surface_normal_xy(obs.depth, obs.camera, grounded.world, *fit(TH)) is not None
    r, old = standard_goal(obs, grounded, TH), standard_goal(obs, grounded, OLD)
    assert (r.pose_world, r.pose_robot, r.flags) == (old.pose_world, old.pose_robot, old.flags)
    assert not any(r.flags.model_dump().values())         # beside the place the line of sight is clear: not frontal, not blocked
    assert math.hypot(*r.pose_world[:2]) == pytest.approx(1.0, abs=0.02) and abs(r.pose_world[1]) > 0.8

@pytest.mark.parametrize("view_angle_deg, seed", [(0.0, 0), (30.0, 2), (60.0, 3), (30.0, 1)])
def test_an_object_on_a_table_keeps_the_stop_of_the_line_of_sight(tmp_path, view_angle_deg, seed):
    # The fit takes in the parcel and the table top round it, and its normal rises by about the tilt limit: over it there
    # is no face, under it the face is the parcel's own side. The table is turned to the robot, so either way the stop is
    # that of the line of sight.
    obs, grounded, et = generated(tmp_path, view_angle_deg, "object", seed)
    normal = surface_normal_xy(obs.depth, obs.camera, grounded.world, *fit(TH))
    r, old = standard_goal(obs, grounded, TH), standard_goal(obs, grounded, OLD)
    assert not (normal is None and r.flags.standard_goal_frontal) and et.standard_goal_ok == {"t0": True}
    if r.flags.standard_goal_frontal:                     # that side's normal is the line of sight
        assert math.dist(r.pose_world[:2], old.pose_world[:2]) < 0.01 and not r.flags.standard_goal_blocked
    else:                                                 # the result of the rule without the normal, to the last digit
        assert (r.pose_world, r.pose_robot, r.flags) == (old.pose_world, old.pose_robot, old.flags)

CABINET = Box(cx=1.0, cy=5.0, yaw=-math.pi / 2, sx=1.0, sy=1.0, sz=2.0)   # 1 m square, taller than the camera, 0.5 to 1.5 m before the desk
STEPS = [(1.0, 6.0 - d) for d in 1.0 + 0.05 * np.arange(11)]             # the steps of the slide before the desk's face

def test_a_place_before_the_face_whose_floor_is_hidden_is_not_taken():
    obs, grounded = rendered_observation(boxes=[DESK, CABINET], robot=ASIDE), target(1.0, 6.0, 0.8)
    assert surface_normal_xy(obs.depth, obs.camera, grounded.world, *fit(TH)) == pytest.approx((0.0, -1.0), abs=0.02)
    # The place 1 m before the face is the cabinet's middle. Nothing the camera sees is near it: the two sides it sees are
    # 0.5 m away, and its top is above the camera. But the floor there is not seen, and what is not seen is not clear.
    assert gap(obs, STEPS[0]) >= TH.standard_goal_clearance_m and not shown(obs, STEPS[0])
    assert not any(shown(obs, step) and gap(obs, step) >= TH.standard_goal_clearance_m for step in STEPS)   # nor a step further out
    r, old = standard_goal(obs, grounded, TH), standard_goal(obs, grounded, OLD)
    assert (r.pose_world, r.pose_robot, r.flags) == (old.pose_world, old.pose_robot, old.flags)
    assert not any(r.flags.model_dump().values())         # the line of sight passes beside the cabinet, and a step of it is clear
    x, y = r.pose_world[:2]
    assert max(abs(x - CABINET.cx), abs(y - CABINET.cy)) > 0.5 + TH.standard_goal_clearance_m - 0.01 and shown(obs, (x, y))

def test_something_lower_than_the_floor_tolerance_at_the_place_is_floor_to_stop_on():
    mat = Box(cx=1.0, cy=5.0, yaw=-math.pi / 2, sx=1.2, sy=1.2, sz=0.06)  # under every step before the face
    obs, grounded = rendered_observation(boxes=[DESK, mat], robot=ASIDE), target(1.0, 6.0, 0.8)
    u, v, z = project(obs.camera, (*STEPS[0], 0.0))       # the ray to the ground under the mat meets its top 0.06 m up, 0.22 m nearer:
    assert 0.15 < z - float(obs.depth[math.floor(v), math.floor(u)]) < TH.visibility_depth_tolerance_m    # within both tolerances
    r = standard_goal(obs, grounded, TH)
    assert {name for name, on in r.flags.model_dump().items() if on} == {"standard_goal_frontal"}
    assert r.pose_world[:2] == pytest.approx(STEPS[0], abs=0.02)

def test_a_place_before_the_face_that_is_outside_the_picture_is_not_taken():
    panned = RobotPose(ASIDE.x, ASIDE.y, ASIDE.yaw - math.radians(25.0))  # the target is 25 degrees left of where the camera looks
    obs, grounded = rendered_observation(boxes=[DESK], robot=panned), target(1.0, 6.0, 0.8)
    assert surface_normal_xy(obs.depth, obs.camera, grounded.world, *fit(TH)) == pytest.approx((0.0, -1.0), abs=0.02)
    for step in STEPS:                                    # and every place before the face further left than the picture reaches
        u, v, _ = project(obs.camera, (*step, 0.0))
        assert u < 0.0 and not in_image(obs.camera, u, v) and gap(obs, step) >= TH.standard_goal_clearance_m
    r, old = standard_goal(obs, grounded, TH), standard_goal(obs, grounded, OLD)
    assert (r.pose_world, r.pose_robot, r.flags) == (old.pose_world, old.pose_robot, old.flags)
    assert not any(r.flags.model_dump().values()) and shown(obs, r.pose_world[:2])
    assert r.pose_world[:2] == pytest.approx((1.0 + math.sin(math.radians(58.0)), 6.0 - math.cos(math.radians(58.0))), abs=1e-6)

def test_the_stop_slides_out_along_the_normal_to_the_first_step_whose_floor_is_seen():
    eye = camera_on_robot(ASIDE, **MOUNT)[:2, 3]
    hidden = np.array((1.0, 5.2))                         # 0.8 m before the face; a post stands 1.2 m from there towards the camera
    cx, cy = hidden + 1.2 * (eye - hidden) / np.hypot(*(eye - hidden))
    post = Box(cx=cx, cy=cy, yaw=ASIDE.yaw, sx=0.3, sy=0.3, sz=1.0)
    obs, grounded = rendered_observation(boxes=[DESK, post], robot=ASIDE), target(1.0, 6.0, 0.8)
    seen = [shown(obs, step) for step in STEPS]           # the post stands near no step. It hides the floor of the first steps,
    whole = [stands(obs, step) for step in STEPS]         # and of the next ones floor that a side of the robot would stand on
    first, room = seen.index(True), whole.index(True)
    assert not any(seen[:first]) and all(seen[first:]) and not any(whole[:room]) and all(whole[room:])
    assert 1 <= first and first + 3 <= room <= 9
    assert all(gap(obs, step) >= TH.standard_goal_clearance_m for step in STEPS)
    r = standard_goal(obs, grounded, TH)
    assert {name for name, on in r.flags.model_dump().items() if on} == {"standard_goal_frontal"}
    assert r.pose_world[:2] == pytest.approx(STEPS[room], abs=0.03) and stands(obs, r.pose_world[:2])
    bare = standard_goal(rendered_observation(boxes=[DESK], robot=ASIDE), grounded, TH)
    assert bare.pose_world[:2] == pytest.approx(STEPS[0], abs=0.02)       # without the post: the first step

def test_a_robot_nearer_the_face_than_the_stop_gets_no_stop_beside_or_behind_itself():
    near = RobotPose(1.0 + 1.2 * math.sin(math.radians(58.0)), 6.0 - 1.2 * math.cos(math.radians(58.0)), math.radians(148.0))
    obs, grounded = rendered_observation(boxes=[DESK], robot=near), target(1.0, 6.0, 0.8)      # 1.2 m from the point, looking at it
    assert surface_normal_xy(obs.depth, obs.camera, grounded.world, *fit(TH)) == pytest.approx((0.0, -1.0), abs=0.02)
    assert not any(shown(obs, step) for step in STEPS)    # 1 m and more before the face is beside the camera, not before it
    r, old = standard_goal(obs, grounded, TH), standard_goal(obs, grounded, OLD)
    assert (r.pose_world, r.pose_robot, r.flags) == (old.pose_world, old.pose_robot, old.flags)
    assert math.dist(r.pose_world[:2], (near.x, near.y)) == pytest.approx(0.2, abs=1e-6) and not shown(obs, r.pose_world[:2])
    assert not any(r.flags.model_dump().values())         # on its own line, at its feet: no floor is asked for there, and it is clear

@pytest.mark.parametrize("mount, fx, under", [
    (MOUNT, 300.0, False),                                # a lens 94 degrees wide: the place before the face is in the picture
    (CAMERA_MOUNT, 250.0, True)])                         # the generator's mount, 104 degrees wide: it lies under the picture
def test_the_slide_along_the_normal_is_not_cut_at_the_robots_distance(mount, fx, under):
    # 1.2 m from the point, between the standoff and the end of the slide, and 70 degrees off the face's front, turned 25
    # degrees towards the place before the face. A lens 60 degrees wide shows the point or that place, never both.
    a = math.radians(70.0)
    robot = RobotPose(1.0 + 1.2 * math.sin(a), 6.0 - 1.2 * math.cos(a), math.pi / 2 + a + math.radians(25.0))
    camera = Camera(**{**INTRINSICS, "fx": fx, "fy": fx}, T_world_camera=camera_on_robot(robot, **mount))
    crate = Box(cx=1.0, cy=5.35, yaw=-math.pi / 2, sx=0.4, sy=0.3, sz=0.3)       # low, from 0.45 to 0.85 m before the face
    rgb, depth = render(camera, [DESK, crate])
    obs, grounded = ObservationData(rgb=rgb, depth=depth, camera=camera, robot=robot), target(1.0, 6.0, 0.8)
    assert all(shown(obs, step) for step in STEPS) and (project(camera, (*STEPS[5], 0.0))[1] >= camera.height) == under
    r = standard_goal(obs, grounded, TH)                  # the first step 0.4 m clear of the crate is 1.25 m out: not on the
    assert {name for name, on in r.flags.model_dump().items() if on} == {"standard_goal_frontal"}   # robot's line, so not past it
    assert r.pose_world[:2] == pytest.approx(STEPS[5], abs=0.06)
    assert math.dist(r.pose_world[:2], (1.0, 6.0)) == pytest.approx(1.25, abs=1e-6)    # the robot is 1.2 m from the point

def desk_seen(view_deg, distance, *more):
    """The desk and what stands with it, seen by the generator's camera from `distance` off the middle of the desk's
    front face and `view_deg` off that face's front."""
    a = math.radians(view_deg)
    robot = RobotPose(1.0 + distance * math.sin(a), 6.0 - distance * math.cos(a), math.pi / 2 + a)
    camera = Camera(**CAMERA_LENS, T_world_camera=camera_on_robot(robot, **CAMERA_MOUNT))
    rgb, depth = render(camera, [DESK, *more])
    return ObservationData(rgb=rgb, depth=depth, camera=camera, robot=robot)

def test_a_place_before_the_face_that_lies_just_under_the_picture_is_taken_where_the_columns_end_on_floor():
    # 1.5 m up and tilted 10 degrees down, the generator's camera shows no floor nearer than 2.3 m ahead. From 3 m and 45
    # degrees off the face's front, the place 1 m before the face is 2.2 m from the camera: under the picture's lower edge.
    obs, grounded = desk_seen(45.0, 3.0), target(1.0, 6.0, 0.8)
    for step in STEPS:
        u, v, _ = project(obs.camera, (*step, 0.0))
        assert v >= obs.camera.height and 0 <= u < obs.camera.width and gap(obs, step) >= TH.standard_goal_clearance_m
    r, old = standard_goal(obs, grounded, TH), standard_goal(obs, grounded, OLD)
    assert {name for name, on in r.flags.model_dump().items() if on} == {"standard_goal_frontal"}
    assert r.pose_world == pytest.approx((1.0, 5.0, math.pi / 2), abs=0.02)
    side = math.sin(math.radians(45.0))                   # the stop of the line of sight is 0.7 m to the side of the desk's middle
    assert old.pose_world[:2] == pytest.approx((1.0 + side, 6.0 - side), abs=1e-6) and not any(old.flags.model_dump().values())
    # With the cabinet on that place its columns end on the cabinet, not on floor. Nothing that is seen is near its middle.
    hidden = desk_seen(45.0, 3.0, CABINET)
    assert gap(hidden, STEPS[0]) >= TH.standard_goal_clearance_m and not any(shown(hidden, step) for step in STEPS)
    r, old = standard_goal(hidden, grounded, TH), standard_goal(hidden, grounded, OLD)
    assert (r.pose_world, r.pose_robot, r.flags) == (old.pose_world, old.pose_robot, old.flags)
    assert not r.flags.standard_goal_frontal

def test_a_stop_whose_centre_is_on_seen_floor_and_whose_rim_is_not_is_not_taken():
    panned = RobotPose(ASIDE.x, ASIDE.y, ASIDE.yaw - math.radians(18.0))  # the place before the face is at the picture's left edge
    left = np.array([math.cos(panned.yaw + math.pi / 2), math.sin(panned.yaw + math.pi / 2)])
    cx, cy = np.array(STEPS[0]) + 0.5 * left              # a pillar 0.5 m square, its near side 0.25 m to the left of that place
    pillar = Box(cx=float(cx), cy=float(cy), yaw=panned.yaw, sx=0.5, sy=0.5, sz=2.0)
    obs, grounded = rendered_observation(boxes=[DESK, pillar], robot=panned), target(1.0, 6.0, 0.8)
    assert np.array_equal(obs.depth, rendered_observation(boxes=[DESK], robot=panned).depth)      # not one pixel shows it
    u, _, _ = project(obs.camera, (*STEPS[0], 0.0))
    assert 0 <= u < 30 and shown(obs, STEPS[0]) and gap(obs, STEPS[0]) >= TH.standard_goal_clearance_m
    # The floor at the stop's centre is seen and nothing that is seen is near it. But the robot is 0.6 m wide: its left side
    # would stand where the picture does not reach, and there it would be 0.05 m inside the pillar.
    assert not shown(obs, np.array(STEPS[0]) + TH.robot_radius_m * left) and not any(stands(obs, step) for step in STEPS)
    r, old = standard_goal(obs, grounded, TH), standard_goal(obs, grounded, OLD)
    assert (r.pose_world, r.pose_robot, r.flags) == (old.pose_world, old.pose_robot, old.flags)
    assert not any(r.flags.model_dump().values())
    slim = standard_goal(obs, grounded, TH.model_copy(update={"robot_radius_m": 0.05}))    # a robot 0.1 m wide is all in the picture
    assert {name for name, on in slim.flags.model_dump().items() if on} == {"standard_goal_frontal"}
    assert slim.pose_world[:2] == pytest.approx(STEPS[0], abs=0.02)

def test_the_floor_is_asked_at_eight_points_round_the_stop_and_one_of_them_outside_the_picture_refuses_it():
    a = math.radians(45.0)                                # 5 m from the face and 45 degrees off its front, panned 17 degrees
    robot = RobotPose(1.0 + 5.0 * math.sin(a), 6.0 - 5.0 * math.cos(a), math.pi / 2 + a - math.radians(17.0))
    obs, grounded = rendered_observation(boxes=[DESK], robot=robot), target(1.0, 6.0, 0.8)
    quarters = np.arange(4) * math.pi / 2                 # along the normal and across it, and the four directions between
    cross, between = (np.array(STEPS[0]) + TH.robot_radius_m * np.stack([np.cos(turns), np.sin(turns)], axis=1)
                      for turns in (quarters, quarters + math.pi / 4))
    assert shown(obs, STEPS[0]) and all(shown(obs, spot) for spot in cross)
    assert [shown(obs, spot) for spot in between] == [True, True, False, True]         # that one is 3 pixels left of the picture
    r, old = standard_goal(obs, grounded, TH), standard_goal(obs, grounded, OLD)
    assert (r.pose_world, r.pose_robot, r.flags) == (old.pose_world, old.pose_robot, old.flags)
    assert not any(r.flags.model_dump().values())

def test_one_stray_near_pixel_at_the_stop_does_not_refuse_it_unless_the_lift_window_is_that_pixel():
    obs, grounded = rendered_observation(boxes=[DESK], robot=ASIDE), target(1.0, 6.0, 0.8)
    u, v, _ = project(obs.camera, (*STEPS[0], 0.0))
    depth = obs.depth.copy()
    depth[math.floor(v), math.floor(u)] *= 0.5            # the pixel of the stop reads half the floor's depth: 0.6 m up, half way there
    stray = replace(obs, depth=depth)
    r = standard_goal(stray, grounded, TH)                # the depth at a pixel is the median of the lift window, as for a lifted point
    assert r.flags.standard_goal_frontal and r.pose_world[:2] == pytest.approx(STEPS[0], abs=0.02)
    alone = standard_goal(stray, grounded, TH.model_copy(update={"lift_window": 1}))      # read by itself, that pixel is no floor:
    assert alone.flags.standard_goal_frontal and alone.pose_world[:2] == pytest.approx(STEPS[1], abs=0.02)    # the next step

def person_seen(view_deg, distance=4.5):
    """A person at (1, 6) facing -y, seen from `view_deg` off their front by the generator's camera, and the target a
    point on their chest gives."""
    a = math.radians(view_deg)
    robot = RobotPose(1.0 + distance * math.sin(a), 6.0 - distance * math.cos(a), math.pi / 2 + a)
    camera = Camera(**CAMERA_LENS, T_world_camera=camera_on_robot(robot, **CAMERA_MOUNT))
    rgb, depth = render(camera, [], [Figure(1.0, 6.0, -math.pi / 2)])
    obs = ObservationData(rgb=rgb, depth=depth, camera=camera, robot=robot)
    u, v, _ = project(camera, (1.0, 6.0, 1.1))
    return obs, GroundedTarget(point=Point(x=500, y=450), pixel=(u, v), world=tuple(lift(depth, camera, u, v)))

@pytest.mark.parametrize("view_deg, side", [(0.0, 0.0), (30.0, 0.0), (60.0, 0.0), (120.0, 180.0), (150.0, 180.0), (180.0, 180.0)])
def test_a_person_is_faced_on_the_broad_side_that_is_seen(view_deg, side):          # their front, or from behind their back
    obs, grounded = person_seen(view_deg)
    normal = surface_normal_xy(obs.depth, obs.camera, grounded.world, *fit(TH))
    anchor, robot = grounded.world[:2], (obs.robot.x, obs.robot.y)
    off = lambda yaw: abs(math.degrees(wrap_angle(math.atan2(normal[1], normal[0]) - yaw)))
    front = -math.pi / 2 + math.radians(side)             # the direction that side of the body faces
    sight = bearing(anchor, robot)
    assert off(front) <= abs(math.degrees(wrap_angle(sight - front))) + 1.0 and off(front) < 30.0
    assert off(sight) <= 40.0                             # between the line of sight and that side's own direction
    r = standard_goal(obs, grounded, TH)
    assert {name for name, on in r.flags.model_dump().items() if on} == {"standard_goal_frontal"}
    assert r.pose_world[:2] == pytest.approx(np.array(anchor) + TH.standoff_m * normal, abs=1e-6)

@pytest.mark.parametrize("view_deg", [81.0, 84.0, 87.0, 90.0, 93.0, 96.0, 99.0])
def test_a_person_in_profile_is_no_face_and_gets_the_stop_of_the_line_of_sight(view_deg):
    obs, grounded = person_seen(view_deg)                 # about as deep as wide from there: a ratio of 0.6 to 0.9, the limit is 0.5
    assert surface_normal_xy(obs.depth, obs.camera, grounded.world, *fit(TH)) is None
    loose = TH.model_copy(update={"surface_max_rms_ratio": 1.0})
    assert surface_normal_xy(obs.depth, obs.camera, grounded.world, *fit(loose)) is not None     # that limit alone refuses them
    r, old = standard_goal(obs, grounded, TH), standard_goal(obs, grounded, OLD)
    assert (r.pose_world, r.pose_robot, r.flags) == (old.pose_world, old.pose_robot, old.flags)
    assert not any(r.flags.model_dump().values())

def test_without_that_limit_the_normal_of_a_person_in_profile_swings_with_the_view():
    loose = TH.model_copy(update={"surface_max_rms_ratio": 1.0})
    yaws = []
    for view_deg in (84.0, 96.0):
        obs, grounded = person_seen(view_deg)
        normal = surface_normal_xy(obs.depth, obs.camera, grounded.world, *fit(loose))
        yaws.append(math.atan2(normal[1], normal[0]))
    assert abs(math.degrees(wrap_angle(yaws[1] - yaws[0]))) > 60.0        # 12 degrees of view apart

def test_scene_check_9_and_the_reference_context_check_follow_the_rule(tmp_path):
    folder = desk_scene(tmp_path, None)                   # nobody about, and the desk is seen 42 degrees off its front
    line_of_sight = CFG.model_copy(update={"thresholds": OLD})
    warned = lambda cfg: {r.number for r in check_scene(folder, cfg, write_png=False) if r.level == "warning"}
    assert warned(CFG) == set() and warned(line_of_sight) == {9}          # before the desk's face; beside the desk, as it was
    inputs, truth = load_scene(folder)
    matter = [reference_people_matter(prepare(inputs, truth, cfg)["e0"], None) for cfg in (CFG, line_of_sight)]
    assert matter == [False, True]                        # the reference answer of the context check is the rule's own outcome
