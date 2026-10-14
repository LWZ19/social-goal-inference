"""The scene description of the realistic set, and the stub that draws it with the numpy ray caster."""
import dataclasses
import json
import math

import numpy as np
import pytest

from sgi.geometry import Camera, RobotPose, camera_on_robot, in_image, project
from sgi.synth.realistic.spec import (MATERIALS, PAINT, SIGN_STYLES, Frame, ModelRef, Opening, Part, PersonRef,
                                      RoomSpec, SceneSpec, Sign, SignArt, from_json, to_json)
from sgi.synth.realistic.stub import StubRenderer, material_colour
from sgi.synth.render import Box, Cylinder, Figure, Room, Sphere, render

LENS = dict(fx=554.3, fy=554.3, cx=320.0, cy=240.0, width=640, height=480)
MOUNT = dict(forward_m=0.2, height_m=1.5, pitch_down_rad=math.radians(10.0))
CAMERA = Camera(**LENS, T_world_camera=camera_on_robot(RobotPose(0.0, 0.0, 0.0), **MOUNT))
EYE = CAMERA.T_world_camera[:3, 3]
NEAR_M = 0.02                                      # a point and the ray through the centre of its pixel part by less

# A room 10 m by 8 m seen from one end, and five things in a row 4.8 m ahead of the camera, none before another.
# Nothing about the room is as the placeholder's Room has it by default, so that a field left out shows.
ROOM = RoomSpec(polygon=((-2.0, -4.0), (8.0, -4.0), (8.0, 4.0), (-2.0, 4.0)), height=2.8, floor="floor_tile",
                ceiling="ceiling_tile", wall=(214, 222, 228), dado=(150, 170, 180), dado_height=1.2, lights=2.0)
CRATE = Part(Box(5.0, -2.0, 0.0, 0.5, 0.6, 1.2, color=(20, 40, 230)), "wood", group=1)
POST = Part(Cylinder(5.0, -1.0, 0.25, 1.2), "steel", group=2)
BALL = Part(Sphere(5.0, 0.0, 0.6, 0.3, color=(0, 0, 255)), PAINT)
PLANT = ModelRef("plant_a", x=5.0, y=1.0, z=0.0, yaw=math.pi / 2, scale=(1.0, 1.0, 1.0), size=(0.4, 0.8, 1.3), group=3)
PERSON = PersonRef("p0", "avatar_a", "waiting_a", x=5.0, y=2.0, yaw=math.pi, height=1.84, build=1.1, group=4)
SCENE = SceneSpec(ROOM, parts=(CRATE, POST, BALL), models=(PLANT,), people=(PERSON,))

CRATE_C, POST_C, BALL_C, PLANT_C = (5.0, -2.0, 0.6), (5.0, -1.0, 0.6), (5.0, 0.0, 0.6), (5.0, 1.0, 0.65)
CHEST = (5.0, 2.0, 1.2)                            # on the person's upright axis
FLOOR, WALL = (3.0, 0.0, 0.0), (8.0, -3.0, 2.0)    # with nothing between them and the camera
PERSON_RADIUS = 0.22 * PERSON.build                # of the footprint a figure of that build stays inside


def depth_at(depth: np.ndarray, point) -> float:
    """The depth image at the pixel that a world point projects into."""
    u, v, _ = project(CAMERA, point)
    assert in_image(CAMERA, u, v)
    return float(depth[math.floor(v), math.floor(u)])


def colour_at(picture: np.ndarray, point) -> np.ndarray:
    u, v, _ = project(CAMERA, point)
    return picture[math.floor(v), math.floor(u)]


def camera_z(point) -> float:
    return project(CAMERA, point)[2]


def towards_the_eye(point, share: float) -> np.ndarray:
    """The point on the line of sight to `point`, at this share of the way from the camera."""
    return EYE + share * (np.asarray(point) - EYE)


def where_sight_meets_x(point, x: float) -> np.ndarray:
    """The point at which the line of sight to `point` crosses the upright plane at this x."""
    return towards_the_eye(point, (x - EYE[0]) / (point[0] - EYE[0]))


def image_box(x: tuple[float, float], y: tuple[float, float], top: float) -> tuple[float, float, float, float]:
    """The columns and the rows, a pixel to spare, that hold the upright block on the ground rectangle x by y."""
    u, v, _ = np.array([project(CAMERA, (i, j, k)) for i in x for j in y for k in (0.0, top)]).T
    return u.min() - 1, u.max() + 1, v.min() - 1, v.max() + 1


def hue(colour) -> np.ndarray:
    """A colour without its brightness, which the light changes."""
    colour = np.asarray(colour, dtype=np.float64)
    return colour / colour.max()


def full_description() -> SceneSpec:
    """A description with every kind of thing in it and no field left at its default."""
    art = SignArt("logo", text="Mora", ink=(250, 240, 220), ground=(30, 60, 90), width_m=1.6, height_m=0.35, seed=7)
    board = Part(Box(1.0, 2.0, 0.5, 0.07, 1.6, 0.35, z0=1.9, color=(30, 60, 90), label="TICKETS", label_face="back",
                     label_color=(255, 255, 255), label_height=0.6, label_width=0.7, panel=(240, 240, 240),
                     panel_rect=(0.1, 0.2, 0.9, 0.8), band=(10, 10, 10), band_range=(0.0, 0.2), top=(1, 2, 3)),
                 "painted_metal", group=2, bevel=0.002,
                 signs=(Sign("back", art, rect=(0.05, 0.1, 0.95, 0.9), emission=1.5), Sign("front", SignArt("plate"))),
                 tint=(52, 56, 62))
    pane = Part(Box(1.2, 2.0, 0.5, 0.01, 1.6, 0.8, z0=1.1), "glass", glass=True)
    post = Part(Cylinder(0.3, 0.4, 0.05, 2.25, z0=0.1, color=(78, 82, 90), top=(1, 1, 1), top_radius=0.03), "steel")
    ball = Part(Sphere(2.0, 2.5, 0.9, 0.12, color=(200, 30, 30)), PAINT, group=1)
    room = dataclasses.replace(ROOM, floor="floor_terrazzo", dado=None, openings=(
        Opening("window", x=8.0, y=1.0, yaw=math.pi, width=1.8, z0=0.9, height=1.2),
        Opening("door", x=3.0, y=-4.0, yaw=math.pi / 2, width=0.9, z0=0.0, height=2.1)))
    people = (PersonRef("p0", "avatar_a", "waiting_a", 2.0, 1.0, -1.2, 1.74, build=1.08, prop="phone", group=4),
              PersonRef("s0", "avatar_b", "serving_a", 0.5, 0.5, 0.0, 1.62))
    return SceneSpec(room, parts=(board, pane, post, ball), people=people, lighting="evening", hdri="street_a", seed=11,
                     models=(ModelRef("bench_a", 1.0, -2.0, 0.0, 0.3, (1.0, 1.05, 0.95), (1.6, 0.5, 0.45), group=3),))


def test_the_fields_and_defaults_are_those_the_tasks_share():
    names = {cls.__name__: [field.name for field in dataclasses.fields(cls)]
             for cls in (SignArt, Sign, Part, ModelRef, PersonRef, Opening, RoomSpec, SceneSpec, Frame)}
    assert names == {
        "SignArt": ["style", "text", "ink", "ground", "width_m", "height_m", "seed"],
        "Sign": ["face", "art", "rect", "emission"],
        "Part": ["solid", "material", "group", "bevel", "signs", "glass", "tint"],
        "ModelRef": ["asset", "x", "y", "z", "yaw", "scale", "size", "group"],
        "PersonRef": ["person_id", "avatar", "pose", "x", "y", "yaw", "height", "build", "prop", "group"],
        "Opening": ["kind", "x", "y", "yaw", "width", "z0", "height"],
        "RoomSpec": ["polygon", "height", "floor", "ceiling", "wall", "dado", "dado_height", "lights", "openings"],
        "SceneSpec": ["room", "parts", "models", "people", "lighting", "hdri", "seed"],
        "Frame": ["camera", "beauty", "only"]}
    art = SignArt("label")
    assert art == SignArt("label", "", (245, 245, 245), (40, 40, 40), 1.0, 0.3, 0)
    assert Sign("front", art) == Sign("front", art, (0.0, 0.0, 1.0, 1.0), 0.0)
    assert Part(POST.solid, "steel") == Part(POST.solid, "steel", 0, 0.004, (), False, None)
    assert ModelRef("m", 5.0, 1.0, 0.0, 0.5, (1.0, 1.0, 1.0), (0.4, 0.8, 1.3)).group == 0
    assert PersonRef("p0", "a", "w", 1.0, 2.0, 0.5, 1.7) == PersonRef("p0", "a", "w", 1.0, 2.0, 0.5, 1.7, 1.0, None, 0)
    assert ROOM.openings == () and SceneSpec(ROOM) == SceneSpec(ROOM, (), (), (), "ceiling", None, 0)
    assert (Frame(CAMERA).beauty, Frame(CAMERA).only) == (True, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        SCENE.seed = 1                             # plain data that nothing changes once it is made


def test_the_names_the_tasks_share_are_listed():
    assert SIGN_STYLES == ("label", "logo", "pictogram", "plate", "menu", "map", "exit", "notice", "screen", "paper")
    assert MATERIALS == (
        "floor_tile", "floor_terrazzo", "floor_laminate", "ceiling_tile", "wall_paint", "wood", "wood_dark",
        "laminate_white", "steel", "painted_metal", "plastic_dark", "plastic_light", "fabric", "stone", "cork", "paper",
        "cardboard", "ceramic", "glass", "screen")
    assert PAINT == "paint" and PAINT not in MATERIALS


def test_a_scene_description_survives_json():
    for spec in (full_description(), SCENE, SceneSpec(ROOM)):
        data = to_json(spec)
        text = json.dumps(data)
        assert json.loads(text) == data            # nothing but what JSON holds: a stored one equals one made anew
        assert from_json(json.loads(text)) == spec
    data = to_json(full_description())
    assert [part["solid"]["shape"] for part in data["parts"]] == ["box", "box", "cylinder", "sphere"]
    assert data["room"]["dado"] is None and data["room"]["polygon"][1] == [8.0, -4.0]
    assert data["parts"][0]["signs"][0]["art"]["text"] == "Mora" and data["parts"][1]["glass"] is True
    assert [part["tint"] for part in data["parts"]] == [[52, 56, 62], None, None, None]
    assert [person["prop"] for person in data["people"]] == ["phone", None]
    assert (data["lighting"], data["hdri"], data["seed"]) == ("evening", "street_a", 11)


def test_a_description_stored_before_a_part_had_a_tint_still_loads():
    """A part's tint was added after scenes had been stored: their descriptions hold no such key, and read as
    parts that take their material as the library has it."""
    spec = full_description()
    assert spec.parts[0].tint is not None
    stored = json.loads(json.dumps(to_json(spec)))
    for part in stored["parts"]:
        del part["tint"]
    untinted = dataclasses.replace(spec, parts=tuple(dataclasses.replace(part, tint=None) for part in spec.parts))
    assert from_json(stored) == untinted and from_json(stored) != spec
    assert all("tint" in part for part in to_json(from_json(stored))["parts"])     # and are written with it from now on


def test_numbers_drawn_with_numpy_are_written_as_plain_numbers():
    rng = np.random.default_rng(0)
    crate = Box(float(rng.uniform()), np.float64(1.5), np.float32(0.25), 1.0, 1.0, 1.0)
    spec = SceneSpec(ROOM, parts=(Part(crate, "wood", group=np.int64(3)),), seed=np.int64(5))
    data = json.loads(json.dumps(to_json(spec)))
    assert (data["parts"][0]["group"], data["seed"], data["parts"][0]["solid"]["yaw"]) == (3, 5, 0.25)
    assert from_json(data) == spec


def test_a_solid_of_an_unknown_shape_is_refused():
    data = to_json(SCENE)
    data["parts"][0]["solid"]["shape"] = "cone"
    with pytest.raises(ValueError, match="cone"):
        from_json(data)


def test_the_stub_draws_parts_models_and_people_where_they_stand():
    assert (StubRenderer.name, StubRenderer.version) == ("stub", None)
    ((rgb, depth),) = StubRenderer().render(SCENE, [Frame(CAMERA)])
    assert (rgb.shape, rgb.dtype, depth.shape, depth.dtype) == ((480, 640, 3), np.uint8, (480, 640), np.float32)
    assert (depth > 0).all()                       # inside the room every ray meets something
    first_hits = {                                 # the centre of each thing, and where the line of sight meets it
        "the crate's near face": (CRATE_C, where_sight_meets_x(CRATE_C, 4.75)),
        "the post's side": (POST_C, towards_the_eye(POST_C, 1.0 - 0.25 / math.hypot(*(np.array(POST_C) - EYE)[:2]))),
        "the ball": (BALL_C, towards_the_eye(BALL_C, 1.0 - 0.3 / np.linalg.norm(np.array(BALL_C) - EYE))),
        # the model's box is turned a quarter: its 0.8 m lie along x, so its near face is 0.4 m before its centre
        "the model's near face": (PLANT_C, where_sight_meets_x(PLANT_C, 4.6)),
        "the top of the model": ((5.0, 1.0, 1.25), where_sight_meets_x((5.0, 1.0, 1.25), 4.6)),
        "the floor": (FLOOR, FLOOR), "the far wall": (WALL, WALL)}
    for name, (centre, hit) in first_hits.items():
        assert depth_at(depth, centre) == pytest.approx(camera_z(hit), abs=NEAR_M), name
    ceiling = (7.0, 0.5, ROOM.height)              # seen at so shallow an angle that a pixel covers centimetres of it
    assert depth_at(depth, ceiling) == pytest.approx(camera_z(ceiling), abs=3 * NEAR_M)
    # The body stands round its axis up to its own height, which is above the top of a figure of the usual 1.7 m.
    for on_the_body in (CHEST, (5.0, 2.0, PERSON.height - 0.06)):
        assert camera_z(on_the_body) - PERSON_RADIUS < depth_at(depth, on_the_body) < camera_z(on_the_body)
    for above in ((5.0, 2.0, PERSON.height + 0.1), (5.0, 1.0, 1.4)):    # over the head, over the model: the wall behind
        assert depth_at(depth, above) > camera_z(above) + 1.0


def test_the_stubs_depth_is_the_ray_casters_for_the_same_primitives():
    ((_, depth),) = StubRenderer().render(SCENE, [Frame(CAMERA, beauty=False)])
    room = Room(ROOM.polygon, height=2.8)
    boxes = [CRATE.solid, POST.solid, BALL.solid, Box(5.0, 1.0, math.pi / 2, 0.4, 0.8, 1.3)]
    figures = [Figure(5.0, 2.0, math.pi, height=1.84, radius=PERSON_RADIUS)]     # a build widens the figure
    assert np.array_equal(depth, render(CAMERA, boxes, figures, room=room, antialias=False)[1])
    narrow = [Figure(5.0, 2.0, math.pi, height=1.84)]
    assert not np.array_equal(depth, render(CAMERA, boxes, narrow, room=room, antialias=False)[1])


def test_a_part_takes_the_colour_of_its_material_and_paint_keeps_the_solids_own():
    ((rgb, _),) = StubRenderer().render(SCENE, [Frame(CAMERA)])
    assert hue(colour_at(rgb, CRATE_C)) == pytest.approx(hue(material_colour("wood")), abs=0.03)   # not its own blue
    assert hue(colour_at(rgb, POST_C)) == pytest.approx(hue(material_colour("steel")), abs=0.03)
    assert hue(colour_at(rgb, BALL_C)) == pytest.approx(hue((0, 0, 255)), abs=0.03)


def test_a_tinted_part_is_drawn_in_its_tint_and_paint_takes_none():
    green, pale = (30, 200, 60), (214, 208, 196)
    tinted = (dataclasses.replace(CRATE, tint=green), dataclasses.replace(POST, tint=pale),
              dataclasses.replace(BALL, tint=green))
    ((rgb, depth),) = StubRenderer().render(dataclasses.replace(SCENE, parts=tinted), [Frame(CAMERA)])
    ((plain, same),) = StubRenderer().render(SCENE, [Frame(CAMERA)])
    assert hue(colour_at(rgb, CRATE_C)) == pytest.approx(hue(green), abs=0.03)        # not its wood, nor its own blue
    assert hue(colour_at(rgb, POST_C)) == pytest.approx(hue(pale), abs=0.03)
    assert colour_at(rgb, POST_C).astype(int).sum() > colour_at(plain, POST_C).astype(int).sum() + 60    # steel, paler
    assert hue(colour_at(rgb, BALL_C)) == pytest.approx(hue((0, 0, 255)), abs=0.03)   # paint is the solid's own colour
    assert np.array_equal(depth, same)                                     # a tint moves nothing


def test_the_room_has_the_walls_the_floor_the_ceiling_and_the_lights_of_the_description():
    scene = dataclasses.replace(SCENE, room=dataclasses.replace(ROOM, floor="floor_laminate", ceiling="wood"))
    plain = dataclasses.replace(scene, room=dataclasses.replace(scene.room, dado=None, lights=0.0))
    ((rgb, _),) = StubRenderer().render(scene, [Frame(CAMERA)])
    ((rgb_plain, _),) = StubRenderer().render(plain, [Frame(CAMERA)])
    low_on_the_wall = (8.0, -0.5, 1.05)            # under this room's dado rail, and over a room's usual one at 0.95 m
    fitting, beside_it = (7.0, 1.0, ROOM.height), (7.0, 0.2, ROOM.height)      # fittings 2 m apart: one is centred here
    wood, laminate = hue(material_colour("wood")), hue(material_colour("floor_laminate"))
    assert hue(colour_at(rgb, WALL)) == pytest.approx(hue(ROOM.wall), abs=0.03)
    assert hue(colour_at(rgb, low_on_the_wall)) == pytest.approx(hue(ROOM.dado), abs=0.03)
    assert hue(colour_at(rgb_plain, low_on_the_wall)) == pytest.approx(hue(ROOM.wall), abs=0.03)
    assert hue(colour_at(rgb, FLOOR)) == pytest.approx(laminate, abs=0.05)             # a tile's tone strays a little
    assert hue(colour_at(rgb, beside_it)) == pytest.approx(wood, abs=0.03) and colour_at(rgb, beside_it).max() < 240
    assert tuple(colour_at(rgb, fitting)) == (255, 254, 246)                           # a lit panel
    assert hue(colour_at(rgb_plain, fitting)) == pytest.approx(wood, abs=0.03)         # and none


def test_the_stub_has_a_colour_for_every_shared_material_and_refuses_any_other():
    colours = [material_colour(name) for name in MATERIALS]
    assert all(len(colour) == 3 and all(0 <= level <= 255 for level in colour) for colour in colours)
    with pytest.raises(ValueError, match="marzipan"):
        StubRenderer().render(dataclasses.replace(SCENE, parts=(dataclasses.replace(CRATE, material="marzipan"),)),
                              [Frame(CAMERA, beauty=False)])
    with pytest.raises(ValueError, match="paint"):                     # a floor has no colour of its own to keep
        StubRenderer().render(dataclasses.replace(SCENE, room=dataclasses.replace(ROOM, floor=PAINT)),
                              [Frame(CAMERA, beauty=False)])


def test_only_draws_the_named_groups_and_nothing_else():
    (_, full), (_, alone), (_, none) = StubRenderer().render(SCENE, [
        Frame(CAMERA, beauty=False), Frame(CAMERA, beauty=False, only=(1, 4)), Frame(CAMERA, beauty=False, only=())])
    assert alone.dtype == np.float32 and alone.shape == full.shape
    for drawn in (CRATE_C, CHEST):                                     # groups 1 and 4, as the whole scene shows them
        assert depth_at(alone, drawn) == depth_at(full, drawn) > 0
    for left_out in (POST_C, BALL_C, PLANT_C, FLOOR, WALL):            # two other groups, no group, and the room
        assert depth_at(alone, left_out) == 0.0
    rows, cols = np.nonzero(alone)
    assert rows.size > 5000 and np.array_equal(alone[rows, cols], full[rows, cols])
    reach = PERSON_RADIUS
    boxes = (image_box((4.75, 5.25), (-2.3, -1.7), 1.2),
             image_box((5.0 - reach, 5.0 + reach), (2.0 - reach, 2.0 + reach), PERSON.height))
    inside = [(cols >= left) & (cols + 1 <= right) & (rows >= top) & (rows + 1 <= bottom)
              for left, right, top, bottom in boxes]
    assert inside[0].any() and inside[1].any() and (inside[0] | inside[1]).all()      # depth is 0 off the two of them
    assert not none.any()


def test_a_depth_only_frame_has_no_picture():
    (rgb, depth), (nothing, same), (cut_out, _) = StubRenderer().render(SCENE, [
        Frame(CAMERA), Frame(CAMERA, beauty=False), Frame(CAMERA, only=(1,))])
    assert nothing is None and np.array_equal(same, depth) and same.dtype == np.float32
    assert rgb.dtype == np.uint8 and rgb.shape == (480, 640, 3)
    assert cut_out.dtype == np.uint8 and cut_out.shape == (480, 640, 3)   # some groups alone still have a picture


def test_each_frame_is_drawn_through_its_own_camera_in_order():
    behind = Camera(fx=138.575, fy=138.575, cx=80.0, cy=60.0, width=160, height=120,
                    T_world_camera=camera_on_robot(RobotPose(0.0, 0.0, math.pi), **MOUNT))
    (_, back), (_, front) = StubRenderer().render(SCENE, [Frame(behind, beauty=False), Frame(CAMERA, beauty=False)])
    assert (back.shape, front.shape) == ((120, 160), (480, 640))
    # the wall at x = -2 is 1.8 m from the turned camera, measured level; its axis is pitched 10 degrees down
    assert float(back[59:61, 79:81].mean()) == pytest.approx(1.8 / math.cos(MOUNT["pitch_down_rad"]), abs=NEAR_M)
    assert StubRenderer().render(SCENE, []) == []
