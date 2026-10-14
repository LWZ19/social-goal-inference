import dataclasses
import math
import numpy as np
import pytest
from PIL import Image
from tests.helpers import camera_for, floor_observation, scene_dict, write_scene
from sgi.inputs import EpisodeInputs
from sgi.scene import SceneError, SceneFile, load_scene, read_scene_file, save_scene

OBS = floor_observation()
FOOTPRINT = [(0.5, 6.0), (1.5, 6.0), (1.5, 7.0), (0.5, 7.0)]
SQUARE = [[0.4, 4.4], [1.6, 4.4], [1.6, 5.4], [0.4, 5.4]]
P0 = {"id": "p0", "x": 1.0, "y": 5.0, "yaw": -math.pi / 2, "group": "g0"}
OCC = {"image": "occupancy.png", "resolution": 0.05, "origin": [-9.0, -8.0, 0]}

def ob(s): return s["observation"]
def gt(s): return s["ground_truth"]
def t0(s): return s["ground_truth"]["targets"][0]
def ep(s): return s["episodes"][0]["ground_truth"]
def region(**fields): return {"polygon": SQUARE, "heading": {"face_point": [1.0, 6.0]}, **fields}
def nav(*ids): return {"type": "navigate", "target_ids": list(ids)}

def two_targets(s):      # adds person p0 (group g0) and a second place target t1; episode e0 accepts both targets
    gt(s)["people"] = [dict(P0)]
    gt(s)["targets"].append(dict(t0(s), id="t1", label="help desk"))
    ep(s)["target_ids"] = ["t0", "t1"]
    return s

def spare_target(s):     # adds a second place target t1 that episode e0 does not name: its target_ids stay ["t0"]
    gt(s)["targets"].append(dict(t0(s), id="t1", label="help desk"))
    return s

def load(tmp_path, edit=lambda s: None):
    s = scene_dict()
    edit(s)
    return load_scene(write_scene(tmp_path, s, OBS))

def test_save_then_load_round_trip(tmp_path):
    T = camera_for().T_world_camera
    scene = SceneFile.model_validate(scene_dict(tour=[{"rgb": "tour/000.png", "T_world_camera": T.tolist()}]))
    save_scene(tmp_path / "s0", scene, OBS.rgb, OBS.depth, tour=[OBS.rgb[::2, ::2]])
    assert read_scene_file(tmp_path / "s0") == scene and np.load(tmp_path / "s0" / "depth.npy").dtype == np.float32
    inputs, truth = load_scene(tmp_path / "s0")
    obs, cam, frame = inputs.observation, inputs.observation.camera, inputs.tour[0]
    assert obs.rgb.dtype == np.uint8 and np.array_equal(obs.rgb, OBS.rgb)
    assert obs.depth.dtype == np.float32 and np.array_equal(obs.depth, OBS.depth)
    assert (cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height) == (554.3, 554.3, 320.0, 240.0, 640, 480)
    assert np.array_equal(cam.T_world_camera, T) and np.array_equal(frame.T_world_camera, T)
    assert (obs.robot.x, obs.robot.y, obs.robot.yaw) == pytest.approx((1.0, 2.0, math.pi / 2))
    assert inputs.scene_id == truth.scene_id == "s0" and inputs.episodes == (EpisodeInputs("e0", "take me to the info desk", None),)
    assert frame.index == 0 and frame.rgb.shape == (240, 320, 3)      # a tour image need not match width and height
    t, e = truth.targets["t0"], truth.episodes["e0"]
    assert (t.kind, t.label, t.anchor, t.footprint) == ("place", "info desk", (1.0, 6.0, 1.0), FOOTPRINT)
    assert t.front_yaw == pytest.approx(-math.pi / 2) and t.match_radius is None          # nothing is filled in
    assert truth.bounds == [(-9.0, -8.0), (11.0, -8.0), (11.0, 12.0), (-9.0, 12.0)]
    assert truth.obstacles == (FOOTPRINT,) and truth.people == () and truth.occupancy_image is None
    assert (e.request_type, e.target_ids, e.situation, e.acceptable_regions) == ("grounded", ["t0"], "no_one", None)
    with pytest.raises(SceneError):                                   # a tour image without a tour entry
        save_scene(tmp_path / "x", SceneFile.model_validate(scene_dict()), OBS.rgb, OBS.depth, tour=[OBS.rgb])

def test_inputs_expose_no_ground_truth(tmp_path):
    inputs, _ = load(tmp_path)
    names = lambda o: [f.name for f in dataclasses.fields(o)]
    assert names(inputs) == ["scene_id", "observation", "tour", "episodes"]
    assert names(inputs.observation) == ["rgb", "depth", "camera", "robot"]
    assert names(inputs.episodes[0]) == ["episode_id", "request", "context"]

def test_rgba_image_loads_as_three_channels(tmp_path):
    folder = write_scene(tmp_path, scene_dict(), OBS)
    rgba = np.random.default_rng(0).integers(0, 256, (480, 640, 4), dtype=np.uint8)
    Image.fromarray(rgba).save(folder / "rgb.png")
    rgb = load_scene(folder)[0].observation.rgb
    assert rgb.dtype == np.uint8 and np.array_equal(rgb, rgba[..., :3])      # alpha dropped, not blended

def test_yaws_are_wrapped_on_load(tmp_path):
    def edit(s):
        gt(s)["people"] = [dict(P0, yaw=-math.pi)]
        ob(s)["robot_pose_world"]["yaw"] = math.pi + 0.0005                  # inside the 0.001 tolerance
        t0(s)["front_yaw"] = -math.pi - 0.0005
        ep(s)["acceptable_regions"] = [region(heading={"yaw": -math.pi})]
    inputs, truth = load(tmp_path, edit)
    assert truth.people[0].yaw == pytest.approx(math.pi) and inputs.observation.robot.yaw == pytest.approx(-math.pi + 0.0005)
    assert truth.targets["t0"].front_yaw == pytest.approx(math.pi - 0.0005)
    assert truth.episodes["e0"].acceptable_regions[0].heading.yaw == pytest.approx(math.pi)

def test_colour_occupancy_is_converted_to_grey(tmp_path):
    s = scene_dict()
    del gt(s)["bounds"], gt(s)["obstacles"]                   # occupancy alone is a valid free-space form
    gt(s)["occupancy"] = dict(OCC)
    folder = write_scene(tmp_path, s, OBS)
    colour = np.full((30, 40, 3), 255, np.uint8)
    colour[0, 0], colour[1, 0] = (255, 0, 0), (0, 0, 0)
    Image.fromarray(colour).save(folder / "occupancy.png")
    truth = load_scene(folder)[1]
    grey = truth.occupancy_image
    assert grey.shape == (30, 40) and grey.dtype == np.uint8 and (grey[0, 0], grey[1, 0], grey[2, 0]) == (76, 0, 255)
    assert truth.occupancy.origin == [-9.0, -8.0] and truth.bounds is None and truth.obstacles == ()

def test_group_for_plain_id_and_map(tmp_path):
    def plain(s):
        gt(s)["people"] = [dict(P0), dict(P0, id="p1", y=4.3), dict(P0, id="p2", x=4.0, group=None)]
        ep(s)["group_at_target"] = "g0"
    truth = load(tmp_path / "plain", plain)[1]
    assert truth.group_for("e0", "t0") == "g0" and truth.person("p2").group is None
    assert [p.id for p in truth.group("g0")] == ["p0", "p1"] and truth.group("g9") == []
    truth = load(tmp_path / "map", lambda s: ep(two_targets(s)).update(
        group_at_target={"t0": "g0", "t1": None}, acceptable_regions=[region(target_id="t0"), region(target_id="t1")],
        reference_plan=[nav(), nav("t1", "t0"), {"type": "act", "optional": True}]))[1]
    assert [truth.group_for("e0", t) for t in ("t0", "t1", "t9")] == ["g0", None, None]     # t9 is not in the map

def test_unknown_situation_label_is_not_rejected(tmp_path):
    truth = load(tmp_path, lambda s: ep(s).update(situation="tea_party"))[1]
    assert truth.episodes["e0"].situation == "tea_party"

EP, T0, AR = "episodes.0.ground_truth.", "ground_truth.targets.0.", "episodes.0.ground_truth.acceptable_regions.0."

def toured(s, *frames):      # a one-frame tour whose image is rgb.png, the file write_scene writes; target t0 gets these tour_frames
    s["tour"] = [{"rgb": "rgb.png", "T_world_camera": ob(s)["T_world_camera"]}]
    t0(s)["tour_frames"] = list(frames)

BAD = {   # rule: (edit applied to scene_dict(), dotted path the message must name)
    "unknown_field": (lambda s: t0(s).update(colour="red"), T0 + "colour"),
    "missing_required": (lambda s: t0(s).pop("label"), T0 + "label"),
    "matrix_not_4x4": (lambda s: ob(s)["T_world_camera"].pop(), "observation.T_world_camera"),
    "heading_empty": (lambda s: ep(s).update(acceptable_regions=[region(heading={})]), AR + "heading"),
    "origin_rotated": (lambda s: gt(s).update(occupancy=dict(OCC, origin=[0.0, 0.0, 1.57])), "ground_truth.occupancy.origin"),
    "rgb_shape": (lambda s: ob(s).update(width=320), "observation.rgb"),
    "depth_shape_swapped": (lambda s: ob(s).update(width=480, height=640), "observation.depth"),
    "missing_file": (lambda s: ob(s).update(depth="nope.npy"), "observation.depth"),
    "duplicate_person": (lambda s: gt(s).update(people=[dict(P0), dict(P0)]), "ground_truth.people.1.id"),
    "duplicate_target": (lambda s: gt(s)["targets"].append(dict(t0(s))), "ground_truth.targets.1.id"),
    "duplicate_episode": (lambda s: s["episodes"].append(s["episodes"][0]), "episodes.1.episode_id"),
    "unknown_target_ids": (lambda s: ep(s).update(target_ids=["t9"]), EP + "target_ids"),
    "unknown_person_id": (lambda s: t0(s).update(kind="person", person_id="p9"), T0 + "person_id"),
    "unknown_region_target": (lambda s: ep(s).update(acceptable_regions=[region(target_id="t9")]), AR + "target_id"),
    # A real target, but not one of this episode's: the region would never be a candidate and no goal could succeed.
    "region_of_another_target": (lambda s: ep(spare_target(s)).update(acceptable_regions=[region(target_id="t1")]),
                                 AR + "target_id: 't1' is not one of this episode's target_ids ['t0']"),
    "forbidden_of_another_target": (lambda s: ep(spare_target(s)).update(
        forbidden_regions=[{"kind": "group_space", "polygon": SQUARE, "target_id": "t1"}]),
        EP + "forbidden_regions.0.target_id: 't1' is not one of this episode's target_ids ['t0']"),
    "unknown_group": (lambda s: ep(s).update(group_at_target="g9"), EP + "group_at_target"),
    "unknown_group_in_map": (lambda s: ep(s).update(group_at_target={"t0": "g9"}), EP + "group_at_target"),
    "tour_frame_past_end": (lambda s: toured(s, 1), T0 + "tour_frames"),      # the only index of a one-frame tour is 0: 1 is 1-based numbering
    "tour_frame_negative": (lambda s: toured(s, -1), T0 + "tour_frames"),
    "yaw_in_degrees": (lambda s: gt(s).update(people=[dict(P0, yaw=180.0)]), "ground_truth.people.0.yaw"),
    "yaw_past_tolerance": (lambda s: ob(s)["robot_pose_world"].update(yaw=math.pi + 0.002), "observation.robot_pose_world.yaw"),
    "front_yaw_in_degrees": (lambda s: t0(s).update(front_yaw=-90.0), T0 + "front_yaw"),
    "heading_yaw_in_degrees": (lambda s: ep(s).update(acceptable_regions=[region(heading={"yaw": 90.0})]), AR + "heading.yaw"),
    "place_without_extent": (lambda s: t0(s).pop("footprint"), T0 + "footprint"),
    "person_without_person_id": (lambda s: t0(s).update(kind="person"), T0 + "person_id"),
    "no_free_space": (lambda s: gt(s).pop("bounds"), "ground_truth.bounds"),
    "bare_group_two_targets": (lambda s: ep(two_targets(s)).update(group_at_target="g0"), EP + "group_at_target"),
    "region_without_target_id": (lambda s: ep(two_targets(s)).update(acceptable_regions=[region()]), AR + "target_id"),
    "forbidden_without_target_id": (lambda s: ep(two_targets(s)).update(
        forbidden_regions=[{"kind": "group_space", "polygon": SQUARE}]), EP + "forbidden_regions.0.target_id"),
    "plan_lists_other_ids": (lambda s: ep(two_targets(s)).update(reference_plan=[nav("t1")]), EP + "reference_plan"),
}

@pytest.mark.parametrize("rule", BAD)
def test_load_rejects(tmp_path, rule):
    edit, path = BAD[rule]
    with pytest.raises(SceneError) as err:
        load(tmp_path, edit)
    assert str(err.value).startswith("s0: ") and path in str(err.value)

def test_tour_frames_are_held_to_the_tour_only_when_the_scene_has_one(tmp_path):
    inputs, truth = load(tmp_path / "toured", lambda s: toured(s, 0))
    assert len(inputs.tour) == 1 and truth.targets["t0"].tour_frames == [0]
    truth = load(tmp_path / "bare", lambda s: t0(s).update(tour_frames=[2, 3]))[1]      # no tour: kept as written, not checked
    assert truth.targets["t0"].tour_frames == [2, 3]

def test_a_person_may_carry_a_role(tmp_path):
    staff = dict(P0, id="s0", group=None, role="staff")
    truth = load(tmp_path / "with", lambda s: gt(s).update(people=[staff, dict(P0, id="p1")]))[1]
    assert truth.person("s0").role == "staff" and truth.person("p1").role is None      # absent loads as None
    assert load(tmp_path / "null", lambda s: gt(s).update(people=[dict(P0, role=None)]))[1].person("p0").role is None
    other = load(tmp_path / "other", lambda s: gt(s).update(people=[dict(P0, group=None, role="cashier")]))[1]
    assert other.person("p0").role == "cashier"                         # any non-empty string: the harness keys on "has a role"

def test_a_person_with_a_role_and_a_group_is_refused(tmp_path):
    with pytest.raises(SceneError) as err:
        load(tmp_path, lambda s: gt(s).update(people=[dict(P0, role="staff")]))          # P0 has group g0
    message = str(err.value)
    assert message.startswith("s0: ") and "ground_truth.people.0" in message and "'p0'" in message
    assert "role" in message and "group" in message

def test_an_empty_role_is_refused(tmp_path):
    with pytest.raises(SceneError) as err:
        load(tmp_path, lambda s: gt(s).update(people=[dict(P0, group=None, role="")]))
    assert "ground_truth.people.0.role" in str(err.value)
