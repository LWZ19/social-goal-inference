import json
import math
import numpy as np
import pytest
from PIL import Image
from sgi.check_scene import check_scene, has_errors
from sgi.config import load_config
from sgi.geometry import RobotPose, camera_on_robot
from sgi.synth.render import Box, Figure
from tests.helpers import rendered_observation, scene_dict, write_scene

CFG = load_config()
# Every situation the default config names has a reference rule. A made-up label for each list stands for one that has none.
RULELESS = load_config(overrides={"dev_situations": [*CFG.dev_situations, "fire_drill"],
                                  "held_out_situations": [*CFG.held_out_situations, "parade"]})
VIEW = RobotPose(-3.0, 1.5, 0.7)      # from the side: the desk, the line and the floor behind the line are in view
DESK = Box(1.0, 6.5, -math.pi / 2, 1.0, 1.0, 1.5, label="INFO")     # on scene_dict's footprint, front face at y = 6

def build(tmp_path, bystander=(-0.5, 5.5)):       # a two-person line at the desk and a bystander; None: nobody at all
    who = [] if bystander is None else [
        dict(id="p0", x=1.0, y=5.3, yaw=math.pi / 2, group="g0"), dict(id="p1", x=1.0, y=4.6, yaw=math.pi / 2, group="g0"),
        dict(id="b0", x=bystander[0], y=bystander[1], yaw=0.0, group=None)]
    obs = rendered_observation([DESK], [Figure(p["x"], p["y"], p["yaw"]) for p in who], robot=VIEW)
    scene = scene_dict()
    scene["observation"].update(T_world_camera=obs.camera.T_world_camera.tolist(),
                                robot_pose_world=dict(x=VIEW.x, y=VIEW.y, yaw=VIEW.yaw))
    scene["ground_truth"]["people"] = who
    if who:                           # with nobody, scene_dict's own no_one episode stays
        scene["episodes"][0]["ground_truth"].update(situation="waiting_line", group_at_target="g0", factors=dict(group_size=2))
    return write_scene(tmp_path, scene, obs)

@pytest.fixture
def scene(tmp_path):                  # the folder of a correct scene, for the test to break
    return build(tmp_path)

def edit(folder, change):             # change(scene.json as a dict, the ground truth of episode e0)
    data = json.loads((folder / "scene.json").read_text())
    change(data, data["episodes"][0]["ground_truth"])
    (folder / "scene.json").write_text(json.dumps(data))

def run(folder, cfg=CFG):
    results = check_scene(folder, cfg, write_png=False)
    by = {r.number: r for r in results}
    errors = {n for n, r in by.items() if r.level == "error"}
    assert [r.number for r in results] == sorted(by) and has_errors(results) == bool(errors)
    return by, errors, {n for n, r in by.items() if r.level == "warning"}

def set_T(f):
    return lambda d, gt: d["observation"].update(T_world_camera=f(np.array(d["observation"]["T_world_camera"])).tolist())

@pytest.mark.parametrize("change, last, word", [
    (lambda d, gt: d["ground_truth"]["people"][0].update(yaw=90.0), 1, "yaw"),      # a yaw in degrees
    (lambda d, gt: gt.update(situation="picnic"), 1, "s0: episode e0: unknown situation 'picnic'"),
    (set_T(np.transpose), 2, ""),
    (set_T(np.linalg.inv), 2, ""),                                                  # here camera +Y ends up pointing up
    (set_T(lambda T: T @ np.diag([1.0, -1.0, -1.0, 1.0])), 2, ""),                  # USD camera axes
    (set_T(lambda T: T @ np.diag([-1.0, 1.0, 1.0, 1.0])), 2, ""),                   # camera X mirrored: determinant -1, floor depth unchanged
    (set_T(lambda T: T * [1.05, 1.05, 1.05, 1.0]), 2, ""),                          # rotation scaled: not orthonormal
    (set_T(lambda T: camera_on_robot(VIEW, 0.2, 1.2, math.radians(65.0))), 2, ""),  # optical axis 25 degrees off vertical
    (set_T(lambda T: T + np.outer([0, 0, 2.3, 0], [0, 0, 0, 1])), 2, ""),           # camera at 3.5 m; the limit is 3.0
    (set_T(lambda T: T - np.outer([0, 0, 1.1, 0], [0, 0, 0, 1])), 2, ""),           # camera at 0.1 m; the limit is 0.2
])
def test_structure_or_pose_error_ends_the_check(scene, change, last, word):
    edit(scene, change)
    by, errors, _ = run(scene)
    assert sorted(by) == list(range(1, last + 1)) and errors == {last} and word in by[last].message

REGION = dict(polygon=[[0.6, 2.7], [1.4, 2.7], [1.4, 4.4], [0.6, 4.4]], heading=dict(face_point=[1.0, 4.6]))
UNDER_B0 = dict(polygon=[[-0.8, 5.2], [-0.2, 5.2], [-0.2, 5.8], [-0.8, 5.8]])
AT_CAMERA = dict(polygon=[[-3.5, 1.0], [-2.2, 1.0], [-2.2, 2.3], [-3.5, 2.3]])    # ends before the first floor pixel, 1.27 m ahead

@pytest.mark.parametrize("change, errors, warnings", [
    (lambda d, gt: None, set(), set()),                                                           # the correct scene
    (set_T(lambda T: camera_on_robot(VIEW, 0.2, 1.2, -0.6)), {3, 4, 5, 6}, {8}),                  # pitched up: no floor, no landmark
    (lambda d, gt: d["ground_truth"].update(bounds=AT_CAMERA), {3, 6}, {10}),                     # no free floor in view
    (lambda d, gt: d["ground_truth"]["people"][2].update(x=4.0, y=4.0), {4}, {8, 11}),            # b0 listed on empty floor, 1.86 m from the region
    (lambda d, gt: d["ground_truth"]["targets"][0].update(anchor=[1.0, 6.6, 1.0]), {5}, set()),   # 0.6 m inside the desk
    (lambda d, gt: gt.update(acceptable_regions=[dict(REGION, representative_point=[1.0, 3.2])]), set(), set()),
    (lambda d, gt: gt.update(acceptable_regions=[dict(REGION, representative_point=[1.0, 4.2])]), {6}, set()),  # 0.4 m from p1
    (lambda d, gt: gt.update(situation="fire_drill"), {6}, set()),                                # no rule, no regions
    (lambda d, gt: gt.update(situation="parade"), set(), {6}),                                    # the same, held out
    (lambda d, gt: gt.update(situation="audience", group_at_target=None, factors=None), {6}, set()),    # a rule that needs the group, and no group
    (lambda d, gt: gt.update(situation="lift_wait", group_at_target=None, factors=None), set(), {6}),   # the same, held out
    (lambda d, gt: d["observation"]["robot_pose_world"].update(x=1.0, y=6.5), {6, 7}, set()),     # in the desk, cut off from the region
    (lambda d, gt: d["observation"]["robot_pose_world"].update(x=1.0, y=4.5), {6, 7}, set()),     # 0.1 m from p1's centre, likewise
    (lambda d, gt: d["ground_truth"]["obstacles"].append(UNDER_B0), set(), {10}),                 # b0 stands on an obstacle
    (lambda d, gt: gt.update(factors=dict(group_size=5)), set(), {12}),                           # the line has 2
])
def test_scene_gives_exactly_these_errors_and_warnings(scene, change, errors, warnings):
    edit(scene, change)
    by, got_errors, got_warnings = run(scene, RULELESS)
    assert len(by) == 12 and (got_errors, got_warnings) == (errors, warnings) and not (scene / "check.png").exists()
    if 3 in errors:                   # in this table check 3 fails only for want of floor pixels
        assert by[3].value is None and by[3].message.startswith("too little visible floor")
    else:
        assert by[2].value == pytest.approx(1.2) and by[3].value < 0.001

def test_range_depth_fails_floor_depth(scene):
    v, u = np.mgrid[0:480, 0:640] + 0.5
    ray = np.sqrt(1 + ((u - 320.0) / 554.3) ** 2 + ((v - 240.0) / 554.3) ** 2)
    np.save(scene / "depth.npy", (np.load(scene / "depth.npy") * ray).astype(np.float32))
    by, errors, _ = run(scene)
    assert 3 in errors and by[3].value == pytest.approx(0.074, abs=0.01)

@pytest.mark.parametrize("bystander, warnings, matches", [
    ((1.6, 5.1), {8}, 3.0),           # directly behind p1 as the camera sees it: occluded
    ((-4.5, 0.2), set(), 3.0),        # behind the camera: skipped
    (None, set(), 1.0),               # nobody; the desk is seen 42 degrees off its front, and the standard goal stands before its face
])
def test_bystander_hidden_or_behind_the_camera_and_empty_hall(tmp_path, bystander, warnings, matches):
    by, errors, got = run(build(tmp_path, bystander))
    assert not errors and got == warnings and by[4].value == matches

@pytest.mark.parametrize("flip, warnings", [(False, set()), (True, {11})])
def test_flipped_occupancy_image(scene, flip, warnings):
    image = np.full((400, 400), 255, np.uint8)        # 20 m square at 0.05 m per pixel; the top row is y = 12
    image[100:120, 190:210] = 0                       # the desk: x 0.5 to 1.5, y 6 to 7
    Image.fromarray(image[::-1] if flip else image).save(scene / "occupancy.png")
    occupancy = dict(image="occupancy.png", resolution=0.05, origin=[-9.0, -8.0])
    edit(scene, lambda d, gt: d["ground_truth"].update(bounds=None, obstacles=[], occupancy=occupancy))
    by, errors, got = run(scene)
    assert not errors and got == warnings and (by[11].value < 0.5) == flip

from sgi.check_scene import main

def test_main_prints_every_check_and_returns_the_exit_code(scene, capsys):
    edit(scene, lambda d, gt: gt.update(people_matter=False))      # a hand label the derived value contradicts
    assert main([str(scene)]) == 0 and (scene / "check.png").stat().st_size > 0
    out = capsys.readouterr().out
    assert sum(line.startswith("[") for line in out.splitlines()) == 12 and "[ 3] OK      floor depth: " in out
    assert ("\ne0: people_matter=True group_size=2 bystanders=1 other_groups=0 view_angle_deg=41.6"
            " (hand label people_matter=False disagrees)\n") in out
    edit(scene, lambda d, gt: gt.update(situation="picnic"))
    assert main([str(scene)]) == 1
    assert "[ 1] ERROR   structure: s0: episode e0: unknown situation 'picnic'\nchecks 2 to 12 skipped\n" in capsys.readouterr().out

# The surface fit switched off, as LINE_OF_SIGHT and OLD do in tests/test_standard_goal.py: no surface has that many
# points, so the standard goal stops on the robot's line of sight, and for a desk seen from the side that stop fails.
SIGHT_CFG = CFG.model_copy(update={"thresholds": CFG.thresholds.model_copy(update=dict(surface_min_points=10 ** 9))})

def _staffed(folder, role, at=(1.0, 7.5), cfg=CFG):                   # a person behind the desk, with or without a role
    edit(folder, lambda d, gt: d["ground_truth"]["people"].append(
        dict(id="s0", x=at[0], y=at[1], yaw=-math.pi / 2, group=None, role=role)))
    return run(folder, cfg)

def test_warning_9_fires_on_a_staffed_scene_with_nobody_else(tmp_path):
    # Check 9 speaks only where the standard goal fails, and whether it fails here is not this test's subject: so the
    # desk, seen 42 degrees off its front, is checked under the line-of-sight rule, whose stop beside the desk fails.
    staff, plain = build(tmp_path / "staff", bystander=None), build(tmp_path / "plain", bystander=None)
    _, errors, warnings = _staffed(staff, "staff", cfg=SIGHT_CFG)
    assert (errors, warnings) == (set(), {9})             # a person with a role is not relevant: nobody is, so 9 speaks
    _, errors, warnings = _staffed(plain, None, cfg=SIGHT_CFG)
    assert (errors, warnings) == (set(), {8})             # without a role the same person is relevant: 9 is silent, and
    #                                                       8 says that the desk hides them from the camera
    assert run(staff)[1:] == (set(), set())               # default config: the stop is before the desk's face and
    #                                                       passes, so the staffed scene has no warning 9 to keep

def test_staff_on_an_obstacle_cell_still_raise_warning_10(tmp_path):
    by, errors, warnings = _staffed(build(tmp_path, bystander=None), "staff", at=(1.0, 6.5))
    assert 10 in warnings and not errors
