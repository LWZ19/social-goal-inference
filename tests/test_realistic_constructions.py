"""How each kind of item is built for the realistic set (realistic-scenes spec 5.3): what a construction draws,
what it blocks, where its staff stand, what names it, and how all of that follows the placeholder's own item.

An item is made here as a recipe makes it, its builder called with its options and the result stood in a frame
of its own, and items are also taken from layouts as the generator draws them. The rules are tried with a small
library made here, with an empty one, and with the library the manifest describes, whatever that holds.
"""
import dataclasses
import inspect
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import sgi.synth.generate as gen
from sgi.geometry import Camera, RobotPose, camera_on_robot, wrap_angle
from sgi.polygons import distance_to_boundary, distance_to_polygon, point_in_polygon
from sgi.synth import recipes
from sgi.synth.realistic import constructions
from sgi.synth.realistic.constructions import SIGN_MODES, STAFFED_KINDS, Construction, Lettered, Surface, construct
from sgi.synth.realistic.library import Library, ModelAsset, load_library
from sgi.synth.realistic.spec import (MATERIALS, PAINT, SIGN_STYLES, Frame, ModelRef, Opening, Part, RoomSpec,
                                      SceneSpec, Sign, SignArt, from_json, to_json)
from sgi.synth.realistic.stub import StubRenderer
from sgi.synth.render import _FACES, Box, Sphere

EYE_M = recipes.CAMERA_MOUNT["height_m"]            # 1.5: what stands lower is in the robot's way, and on its map
FLOOR_M = 0.15                                      # no higher than this is floor to the scene check
CELL_M = 0.05                                       # a cell of the map
OVERHEAD_M = 1.85                                   # above every head: the sign over a place begins higher
BODY_M = 0.2                                        # the disc a standing body needs
REACH_M = 0.45                                      # the furthest a pose gets from the body's axis (spec 5.4)
STRIP_M = 0.36                                      # the least depth of the floor behind a counter, and the most of
#                                                     the counter
NAMING = ("label", "logo", "plate", "pictogram", "exit")       # the sign styles that name what they are on
LABEL_FILLS = (0.86, 0.62)                          # of a label's rectangle, the most its lettering takes: as
#                                                     recipes._place_sign letters a board, and as signs.draw does
PICTOGRAMS = ("information", "ticket", "printer", "cup", "exit")
TYPE_M = 0.04                                       # the tallest type of natural signage, on whatever carries it: a
#                                                     plate and a logo as much as a menu, a map or a notice
NAMES_TRIED = 64                                    # the seeds that a sign which names its item is measured with

PLACES = {"stall": recipes._stall, "ticket_counter": recipes._ticket_counter, "desk": recipes._desk,
          "kiosk": recipes._kiosk}
FURNISHINGS = {"plant": recipes._plant, "bench": recipes._bench, "bin": recipes._bin, "pillar": recipes._pillar,
               "poster": recipes._poster, "door": recipes._door, "window": recipes._window,
               "board": recipes._notice_board, "cooler": recipes._water_cooler, "lockers": recipes._lockers,
               "extinguisher": recipes._extinguisher, "shop": recipes._shop}
BUILDERS = {**PLACES, "back_room": recipes._back_room, "parcel_table": recipes._parcel_table, **FURNISHINGS}
KINDS = tuple(BUILDERS)                             # the eighteen
UNSTAFFED = tuple(kind for kind in KINDS if kind not in STAFFED_KINDS)
SERVED = tuple(kind for kind in STAFFED_KINDS if kind in PLACES)       # the staffed kinds a scene can be about: they
#                                                     are built anew behind their front. The fourth staffed kind, the
#                                                     second stall, is a furnishing and stands as its twin does
MODELLED = ("plant", "bench", "cooler", "lockers")
OPTIONS = {**{recipes._PLACE_KINDS[build]: {"label": label, "text": text, "color": color}
              for label, text, color, build, *_ in recipes._PLACES},
           "back_room": {"depth": 2.4, "text": "COFFEE", "color": (126, 58, 40)}, "plant": {"scale": 1.07},
           "window": {"width": 1.9}, "bench": {"length": 1.6},
           "poster": dict(zip(("text", "ink", "width"), recipes._POSTERS[0])),
           "shop": dict(zip(("text", "color", "goods"), recipes._SHOPS[0]))}
FRAME = (3.2, -1.7, 2.3)                            # where an item made here stands: off the origin and off the axes
# Rows of the set, one of each kind of scene, and the seeds with which their deep place is a desk, a stall, a
# kiosk and a booth: between them every kind of item, and every staffed kind with a back room behind it.
ROWS = (("no_one", 30, "deep_place", 0, False), ("no_one", 0, "object", 0, False),
        ("waiting_line", 2, None, 0, False), ("conversation_group", 3, None, 0, True))
SEEDS = (0, 1, 3, 4)


def rng(seed=0):
    return np.random.default_rng(seed)


def a_library(*models):
    return Library(models={model.id: model for model in models}, avatars={}, poses={}, materials={}, hdris=(),
                   fonts={}, prepared=Path("not_prepared"))


def a_model(name, kind, size, brand=False):
    return ModelAsset(name, kind, size, frozenset(), brand)


# A few models, some that go into their place and some that do not: the cooler and the chair are too large
# by far, and the lockers that fit best show a brand.
SMALL = a_library(a_model("plant_a", "plant", (0.62, 0.6, 1.3)), a_model("plant_b", "plant", (0.7, 0.72, 1.5)),
                  a_model("bench_a", "bench", (0.46, 1.5, 0.84)), a_model("cooler_wide", "cooler", (0.5, 0.5, 1.3)),
                  a_model("lockers_a", "lockers", (0.45, 1.1, 1.8)),
                  a_model("lockers_branded", "lockers", (0.41, 1.2, 1.8), brand=True),
                  a_model("monitor_a", "monitor", (0.15, 0.55, 0.45)),
                  a_model("chair_a", "office_chair", (0.62, 0.62, 1.05)),
                  a_model("machine_a", "espresso_machine", (0.3, 0.34, 0.36)),
                  a_model("cup_a", "cup", (0.085, 0.085, 0.1)))
EMPTY = a_library()


@pytest.fixture(params=("small", "empty", "manifest"))
def library(request):
    if request.param == "manifest":
        try:
            return load_library()
        except ValueError as error:                 # it is being written beside this: no rule here is about it
            pytest.skip(f"the manifest does not load yet: {error}")
    return {"small": SMALL, "empty": EMPTY}[request.param]


@pytest.fixture(params=SIGN_MODES)
def signs(request):
    return request.param


def to_world(frame, x, y):
    fx, fy, yaw = frame
    c, s = math.cos(yaw), math.sin(yaw)
    return fx + c * x - s * y, fy + s * x + c * y


def an_item(kind, frame=FRAME, **options):
    """An item of the kind as a recipe makes it, stood in the frame, and the primitives that draw it there."""
    options = {**OPTIONS.get(kind, {}), **options}
    build = BUILDERS[kind]
    parts = build(**{name: options[name] for name in inspect.signature(build).parameters})
    if kind in PLACES:                              # the 0.8 m people are served at, across the place
        x0, y0, x1, y1 = -0.8, -0.8, 0.0, 0.8
    elif kind == "back_room":                       # the rest of a deep place, behind that
        x0, y0, x1, y1 = -options["depth"], -0.8, -0.8, 0.8
    else:                                           # what its primitives fill
        x0, y0, x1, y1 = recipes._extent(parts)
    footprint = [to_world(frame, x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    x, y, yaw = frame
    item = recipes.Item(kind, x, y, wrap_angle(yaw), footprint, options, "boxes", 0, len(parts))
    return item, recipes._placed(parts, x, y, yaw)


def built(kind, signs, library, seed=0, group=0, **options):
    item, primitives = an_item(kind, **options)
    return item, primitives, construct(item, primitives, signs, library, rng(seed), group)


def variants(kind):                                 # the options a kind is tried with: a poster, with each of its words
    if kind != "poster":
        return [{}]
    return [dict(zip(("text", "ink", "width"), poster)) for poster in recipes._POSTERS]


@lru_cache(maxsize=None)
def layouts():                                      # as the generator draws them: extras added, then moved at random
    return [gen._layout(rng(seed), row) for row in ROWS for seed in SEEDS]


def primitives_of(layout, item):
    return getattr(layout, item.source)[item.start:item.stop]


def in_frame(item, point):                          # a ground point of the world, in the item's own frame
    c, s = math.cos(item.yaw), math.sin(item.yaw)
    dx, dy = point[0] - item.x, point[1] - item.y
    return c * dx + s * dy, c * dy - s * dx


def sides(item, points):                            # (x0, y0, x1, y1) of the points, in the item's frame
    xs, ys = zip(*(in_frame(item, point) for point in points))
    return min(xs), min(ys), max(xs), max(ys)


def model_box(model):                               # the box a model fills, as the stub draws it
    return Box(model.x, model.y, model.yaw, *model.size, z0=model.z)


def solids(construction, glass=True):               # all that a construction stands in the room
    return [*(part.solid for part in construction.parts if glass or not part.glass),
            *(model_box(model) for model in construction.models)]


def heights(solid):                                 # its underside and its top
    if isinstance(solid, Sphere):
        return solid.z - solid.radius, solid.z + solid.radius
    return solid.z0, solid.z0 + (solid.sz if isinstance(solid, Box) else solid.height)


def corners(box):
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    return [(box.cx + c * i * box.sx / 2 - s * j * box.sy / 2, box.cy + s * i * box.sx / 2 + c * j * box.sy / 2)
            for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def reach(solid):                                   # of a round solid, from its axis
    return solid.radius if isinstance(solid, Sphere) else max(solid.radius, solid.top_radius or 0.0)


def ground(solid, n=5):                             # points over the ground it stands on, its outline among them
    if isinstance(solid, Box):
        c, s = math.cos(solid.yaw), math.sin(solid.yaw)
        steps = np.linspace(-0.5, 0.5, n)
        return [(solid.cx + c * i * solid.sx - s * j * solid.sy, solid.cy + s * i * solid.sx + c * j * solid.sy)
                for i in steps for j in steps]
    turns = np.linspace(0.0, 2 * math.pi, 48, endpoint=False)
    return [(solid.x, solid.y), *((solid.x + reach(solid) * math.cos(t), solid.y + reach(solid) * math.sin(t))
                                  for t in turns)]


def gap(point, solid):                              # from a ground point to the ground the solid stands on
    if isinstance(solid, Box):
        return distance_to_polygon(point, corners(solid))
    return max(0.0, math.hypot(point[0] - solid.x, point[1] - solid.y) - reach(solid))


def face_rectangle(box, face, rect):
    """A rectangle on a face of a box, given as Box.panel_rect gives one (left, bottom, right, top as its reader
    sees the face): its centre, the face's normal, the way it reads, its width and its height, in the world.
    The arithmetic of sgi.synth.generate._reads."""
    normal, sign, a_axis, a_sign, b_axis, b_sign = _FACES[face]
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    axes = np.array([(c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0)])
    size = np.array([box.sx, box.sy, box.sz])
    left, bottom, right, top = rect
    middle = (np.array([box.cx, box.cy, box.z0 + box.sz / 2]) + sign * size[normal] / 2 * axes[normal]
              + a_sign * ((left + right) / 2 - 0.5) * size[a_axis] * axes[a_axis]
              + b_sign * ((bottom + top) / 2 - 0.5) * size[b_axis] * axes[b_axis])
    width, height = (right - left) * size[a_axis], (top - bottom) * size[b_axis]
    return middle, sign * axes[normal], a_sign * axes[a_axis], width, height


def placeholder_lettering(box):
    """Where the placeholder letters a box: about the middle of its panel, or of its label face without one,
    over label_width and label_height of that."""
    left, bottom, right, top = box.panel_rect if box.panel is not None else (0.0, 0.0, 1.0, 1.0)
    across, up = box.label_width * (right - left) / 2, box.label_height * (top - bottom) / 2
    x, y = (left + right) / 2, (bottom + top) / 2
    return face_rectangle(box, box.label_face, (x - across, y - up, x + across, y + up))


def signs_of(construction):
    return [(part, sign) for part in construction.parts for sign in part.signs]


def sign_art():
    """The module that draws sign art, once it draws: a test that measures lettering is skipped until then."""
    art_module = pytest.importorskip("sgi.synth.realistic.signs")
    if not art_module.text_heights(SignArt("menu", "coffee", width_m=0.6, height_m=0.9)):
        pytest.skip("sign art is not drawn yet: signs.text_heights gives no lettering for a menu")
    return art_module


@lru_cache(maxsize=None)
def tallest_type(art):
    """How tall the tallest capitals of a sign stand, in metres, as the module that draws its art sets them at
    the scale a scene is rendered with; 0.0 for art without words. For a test that has asked for sign_art."""
    import sgi.synth.realistic.signs as art_module
    return max(art_module.text_heights(art), default=0.0) * art.height_m


def is_at(lettered, rectangle, off=1e-9):           # a Lettered against a face_rectangle
    middle, normal, across, width, height = rectangle
    return (np.allclose(lettered.centre, middle, atol=off) and np.allclose(lettered.normal, normal, atol=1e-9)
            and np.allclose(lettered.across, across, atol=1e-9) and math.isclose(lettered.width, width, abs_tol=1e-9)
            and math.isclose(lettered.height, height, abs_tol=1e-9))


def assert_on_footprint(item, construction):
    for solid in solids(construction):
        if heights(solid)[0] >= EYE_M:              # what begins above the camera may overhang
            continue
        assert all(distance_to_polygon(point, item.footprint) <= 0.01 + 1e-9 for point in ground(solid)), solid


def assert_covered(construction, obstacles):
    for solid in solids(construction):
        low, high = heights(solid)
        if high <= FLOOR_M or low >= EYE_M:
            continue
        for point in ground(solid):
            assert min(distance_to_polygon(point, obstacle) for obstacle in obstacles) <= CELL_M + 1e-9, solid


def assert_staffed(item, front, obstacles, standing):
    """The rules on a staffed place's one staff spot: `obstacles` and `standing` are those of all the place,
    its back room with it if it has one."""
    (spot,) = front.staff_spots
    x, y, yaw = spot
    assert yaw == pytest.approx(item.yaw)           # they face out of the front
    # On the front item's own ground, the 0.8 m behind the front, and so never in a room behind it.
    assert point_in_polygon((x, y), item.footprint) and distance_to_boundary((x, y), item.footprint) >= CELL_M
    assert sides(item, item.footprint) == pytest.approx((-0.8, -0.8, 0.0, 0.8))
    # On a free cell whatever the grid: 5 cm outside every obstacle, when a cell's centre is 3.6 cm from its corner.
    assert all(distance_to_polygon((x, y), obstacle) >= CELL_M - 1e-9 for obstacle in obstacles)
    # Room for a body: nothing that stands lower than the camera within 0.2 m.
    for solid in standing:
        if heights(solid)[0] < EYE_M:
            assert gap((x, y), solid) >= BODY_M - 1e-9, solid
    # The strip: floor that is free from the back of the counter on, through the spot, at least 0.36 m deep
    # and wide enough each way for the body.
    lx, ly = in_frame(item, (x, y))

    def free(at_x, at_y):
        point = to_world((item.x, item.y, item.yaw), at_x, at_y)
        return all(distance_to_polygon(point, obstacle) > 0.0 for obstacle in obstacles)

    for at_y in (ly - BODY_M, ly, ly + BODY_M):
        front_x = back_x = lx
        while front_x + 0.005 <= 0.0 and free(front_x + 0.005, at_y):
            front_x += 0.005
        while back_x - 0.005 >= -0.8 and free(back_x - 0.005, at_y):
            back_x -= 0.005
        assert front_x - back_x >= STRIP_M - 0.01, (at_y, back_x, front_x)
    # The counter is the obstacle under the front face, and is no deeper than 0.36 m.
    (counter,) = [obstacle for obstacle in front.obstacles
                  if point_in_polygon(to_world((item.x, item.y, item.yaw), -0.01, 0.0), obstacle)]
    x0, _, x1, _ = sides(item, counter)
    assert x1 == pytest.approx(0.0, abs=1e-9) and x1 - x0 <= STRIP_M + 1e-9
    # They show over the counter: nothing that is not glass stands before their head and shoulders.
    for solid in solids(front, glass=False):
        low, high = heights(solid)
        if low < 1.6 and high > 1.35:
            before = [in_frame(item, point) for point in ground(solid)]
            assert not any(px > lx and abs(py - ly) < BODY_M for px, py in before), solid


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_has_a_construction(kind, signs, library):
    item, primitives, made = built(kind, signs, library, group=7)
    assert isinstance(made, Construction)
    assert made.parts or made.models or made.openings
    assert all(isinstance(part, Part) for part in made.parts)
    assert all(isinstance(model, ModelRef) for model in made.models)
    assert all(isinstance(lettered, Lettered) for lettered in made.lettering)
    assert all(isinstance(surface, Surface) for surface in made.surfaces)
    assert all(isinstance(opening, Opening) for opening in made.openings)
    assert all(isinstance(field, tuple) for field in (made.parts, made.models, made.obstacles, made.staff_spots,
                                                        made.lettering, made.surfaces, made.openings))
    # Every part and every model draws the object group it was built for.
    assert {thing.group for thing in (*made.parts, *made.models)} <= {7}
    # The materials are the shared ones, which the stub and the library know, or a coat of paint.
    assert {part.material for part in made.parts} <= {*MATERIALS, PAINT}
    # Edges are rounded by 4 mm at the most, and by less than half of what a thin part measures.
    for part in made.parts:
        solid = part.solid
        least = (min(solid.sx, solid.sy, solid.sz) if isinstance(solid, Box) else
                 2 * solid.radius if isinstance(solid, Sphere) else min(2 * solid.radius, solid.height))
        assert 0.0 <= part.bevel <= 0.004 and part.bevel <= 0.25 * least, part
    # What it blocks: polygons of plain numbers.
    assert made.obstacles
    for obstacle in made.obstacles:
        assert len(obstacle) >= 3 and all(type(c) is float for corner in obstacle for c in corner)
    # A model is one the library may show, at the size its scale gives it.
    for model in made.models:
        asset = library.models[model.asset]
        assert not asset.brand
        assert model.size == pytest.approx(tuple(s * own for s, own in zip(model.scale, asset.size)))
        assert all(0.85 - 1e-9 <= s <= 1.15 + 1e-9 for s in model.scale)
        assert model.z >= 0.0


@pytest.mark.parametrize("kind", KINDS)
def test_a_construction_is_a_scene_description_that_json_holds(kind, signs, library):
    item, primitives, made = built(kind, signs, library, group=3)
    floor = tuple(to_world(FRAME, x, y) for x, y in ((-6.0, -6.0), (6.0, -6.0), (6.0, 6.0), (-6.0, 6.0)))
    room = RoomSpec(floor, 3.0, "floor_tile", "ceiling_tile", (233, 227, 213), (200, 190, 170), 0.95, 2.4,
                    made.openings)
    spec = SceneSpec(room, made.parts, made.models)
    stored = json.loads(json.dumps(to_json(spec)))
    assert stored == to_json(spec)
    assert from_json(stored) == spec


@pytest.mark.parametrize("kind", KINDS)
def test_the_stub_draws_every_construction(kind):
    item, primitives, made = built(kind, "natural", SMALL, group=1)
    floor = tuple(to_world(FRAME, x, y) for x, y in ((-6.0, -6.0), (6.0, -6.0), (6.0, 6.0), (-6.0, 6.0)))
    room = RoomSpec(floor, 3.0, "floor_tile", "ceiling_tile", (233, 227, 213), None, 0.95, 0.0, made.openings)
    stand = RobotPose(*to_world(FRAME, 4.5, 0.0), wrap_angle(FRAME[2] + math.pi))
    camera = Camera(69.3, 69.3, 40.0, 30.0, 80, 60, camera_on_robot(stand, **recipes.CAMERA_MOUNT))
    ((_, depth),) = StubRenderer().render(SceneSpec(room, made.parts, made.models), [Frame(camera, False, (1,))])
    assert (depth > 0).any() == bool(made.parts or made.models)


def test_a_material_is_chosen_from_the_colour_and_what_is_known_by_its_colour_keeps_it(signs):
    """Wood for the browns, metal for the greys, stone, cork and paper where the kind says so; and paint for
    what its colour says what it is: the blue parcel stays blue, the extinguisher red, the leaves green."""
    def materials(kind, **options):
        return {(part.solid.color, part.material) for part in built(kind, signs, EMPTY, **options)[2].parts}

    table = materials("parcel_table")
    assert ((40, 80, 200), PAINT) in table           # the parcel the scene is about
    assert ((172, 128, 86), "wood") in table and ((176, 140, 96), "cardboard") in table
    assert {material for _, material in materials("plant") if material != PAINT} == {"wood_dark"}       # its stem
    assert ((204, 36, 32), PAINT) in materials("extinguisher")
    assert {material for _, material in materials("pillar")} == {"stone"}
    assert ((150, 156, 162), "steel") in materials("bin")
    assert {"cork", "paper", "wood_dark"} <= {material for _, material in materials("board")}
    assert ((156, 112, 74), "wood") in materials("stall") and ((156, 112, 74), "wood") in materials("bench")
    assert {"glass", "painted_metal", "steel"} <= {material for _, material in materials("ticket_counter")}
    # A solid keeps its colour whatever it is made of, and a part of paint is of that colour.
    assert all(len(colour) == 3 for colour, _ in table)


FAR = 40                                            # of 255, on the channel that differs most: from here on a part's
#                                                     colour is no longer its material's own
TINTED = {"painted_metal": (92, 104, 120), "plastic_dark": (48, 50, 54), "plastic_light": (214, 214, 210),
          "laminate_white": (236, 236, 232), "paper": (244, 242, 232), "ceramic": (240, 238, 232),
          "fabric": (88, 104, 132), "stone": (78, 78, 79), "wall_paint": (233, 227, 213)}
#                                                     the materials a tint colours outright, and what each comes out in
#                                                     without one


def off(colour, own):
    return max(abs(int(a) - int(b)) for a, b in zip(colour, own))


@pytest.mark.parametrize("kind", KINDS)
def test_a_part_keeps_its_colour_as_a_tint_where_its_material_would_come_out_far_from_it(kind, signs, library):
    """A material that takes any colour comes out in its own where a part gives it no tint. A part whose colour
    is far from that keeps it: the tint is the part's colour. Paint, glass and a material with a colour of its
    own (wood, steel, cork) take none, and nothing is tinted that would look the same without."""
    for chosen in variants(kind):
        for part in built(kind, signs, library, **chosen)[2].parts:
            own = TINTED.get(part.material)
            if own is None or part.glass:
                assert part.tint is None, part
            elif off(part.solid.color, own) >= FAR:
                assert part.tint == tuple(part.solid.color), part
                assert all(type(level) is int and 0 <= level <= 255 for level in part.tint), part
            else:
                assert part.tint is None, part


def test_a_pillar_is_pale_a_plinth_dark_and_a_chair_in_its_cloths_colour(signs):
    """Where the placeholder's colour is what makes a thing what it is, the restyled part has it: a pillar is
    pale stone and not the dark stone of a counter's top, a booth's frame and plinth are darker than its walls,
    and the cloth of a desk chair is the twin's and not the library's blue."""
    def tints(kind, material):
        return {(part.solid.color, part.tint) for part in built(kind, signs, EMPTY)[2].parts
                if part.material == material}

    assert tints("pillar", "stone") == {((170, 164, 152), (170, 164, 152)), ((214, 208, 196), (214, 208, 196))}
    assert ((66, 58, 54), None) in tints("stall", "stone")               # the counter's top: dark stone as it is
    booth = tints("ticket_counter", "painted_metal")
    assert ((88, 96, 112), None) in booth                                # its walls: the paint as the library has it
    assert ((60, 62, 70), (60, 62, 70)) in booth and ((52, 56, 66), (52, 56, 66)) in booth    # frame and plinth
    assert tints("desk", "fabric") == {((190, 96, 60), (190, 96, 60))}
    assert ((60, 60, 64), (60, 60, 64)) in tints("bench", "painted_metal")


def test_the_colours_a_tint_is_measured_against_are_the_manifests():
    """What a material comes out in without a tint is said twice, in the manifest for Blender and beside the
    constructions for the rule above: the two agree, and each of those materials is one that a tint colours
    outright, made in code or called neutral."""
    from sgi.synth.realistic import constructions
    from sgi.synth.realistic.library import manifest_entries

    entries = {entry["id"]: entry for entry in manifest_entries()["materials"]}
    assert constructions._OWN == TINTED and constructions._FAR == FAR
    for name, own in TINTED.items():
        entry = entries[name]
        assert entry["kind"] == "plain" or entry["prepare"].get("neutral") is True, name
        assert tuple(entry["prepare"]["colour"]) == own, name
    # And no other material that a part is made of takes a colour outright: its tint would only darken it.
    for name in MATERIALS:
        if name not in TINTED and not name.startswith("floor_") and name not in ("glass", "screen"):
            assert entries[name]["kind"] == "pbr" and not entries[name]["prepare"].get("neutral"), name


def on_the_counter(item, made, low, high=OVERHEAD_M):
    """The parts and model boxes of a construction that stand at or over `low` and end under `high`, each with
    the sides of its ground in the item's frame."""
    found = []
    for solid in solids(made):
        bottom, top = heights(solid)
        if bottom >= low - 1e-9 and top < high:
            found.append((solid, sides(item, ground(solid, n=3))))
    return found


@pytest.mark.parametrize("held", ("empty", "manifest"))
def test_the_stalls_machine_is_built_of_parts_on_the_counter_beside_the_staff(held, signs):
    """No model of the library goes into a counter 0.3 m deep, so the machine is built: a steel body with a dark
    front, two group heads over a drip tray, a small screen, and cups on its top. All of it stays in the box
    the placeholder gives a machine, moved onto the shallower counter: at one end, beside the staff."""
    library = EMPTY if held == "empty" else load_library()
    item, primitives, made = built("stall", signs, library)
    assert not [model for model in made.models if library.models[model.asset].kind == "espresso_machine"]
    (spot,) = made.staff_spots
    lx, ly = in_frame(item, spot[:2])
    (counter,) = [sides(item, obstacle) for obstacle in made.obstacles if sides(item, obstacle)[2] > -1e-9]
    assert counter == pytest.approx((-0.3, -0.8, 0.0, 0.8))
    machine = [(solid, where) for solid, where in on_the_counter(item, made, 1.1) if where[1] > 0.42]
    assert len(machine) >= 12
    for solid, (x0, y0, x1, y1) in machine:
        assert counter[0] + 0.01 - 1e-9 <= x0 and x1 <= counter[2] - 0.01 + 1e-9, solid   # on the counter's rectangle
        assert 0.43 - 1e-9 <= y0 and y1 <= 0.79 + 1e-9, solid            # 0.36 m of its width, at its end
        assert heights(solid)[1] <= 1.1 + 0.36 + 1e-9, solid             # and no higher than the placeholder's
        assert y0 - ly >= 2 * BODY_M, solid                              # beside the staff, not before them
    parts = {part.solid: part for part in made.parts}
    of = {}
    for solid, where in machine:
        of.setdefault(parts[solid].material, []).append((solid, where))
    assert {"steel", "plastic_dark", "screen", "ceramic"} <= set(of)
    steel = [solid for solid, _ in of["steel"] if isinstance(solid, Box)]
    body = max(steel, key=lambda box: box.sx * box.sy * box.sz)
    assert body.sz >= 0.15 and max(heights(box)[1] for box in steel) >= 1.1 + 0.2     # brushed steel, 0.2 m and more
    cups = [solid for solid, _ in of["ceramic"]]
    assert len(cups) >= 4
    for cup in cups:                                                     # cups stand on its top
        assert any(heights(box)[1] == pytest.approx(heights(cup)[0]) and gap((cup.x, cup.y), box) == 0.0
                   for box in steel), cup
    heads = [solid for solid, _ in of["steel"] if not isinstance(solid, Box) and solid.radius >= 0.02]
    assert len(heads) == 2 and all(heights(head)[0] >= 1.1 + 0.1 for head in heads)   # two group heads, hung high
    tray = min(steel, key=lambda box: heights(box)[0])
    assert heights(tray)[0] == pytest.approx(1.1) and heights(tray)[1] <= 1.1 + 0.06
    for head in heads:                                                   # each under the body's head, over the tray
        assert gap((head.x, head.y), tray) == 0.0
        assert any(heights(box)[0] == pytest.approx(heights(head)[1]) and gap((head.x, head.y), box) == 0.0
                   for box in steel), head
    # The machine is the same whatever the signage.
    other = built("stall", SIGN_MODES[1 - SIGN_MODES.index(signs)], library)[2]
    assert {solid for solid, _ in machine} <= {part.solid for part in other.parts}


SHOP_GOODS = {text: goods for text, _, goods in recipes._SHOPS}
GOODS_BOX = (0.55, -1.0, 1.05, 1.0)                 # the shop's counter, in its frame: its goods stand on it
GOODS_TOP_M = 1.5                                   # and end under the camera's height


def shop_goods(signs, library, text, **more):
    colour = next(color for name, color, _ in recipes._SHOPS if name == text)
    item, primitives, made = built("shop", signs, library, text=text, color=colour, goods=SHOP_GOODS[text], **more)
    return item, primitives, made, on_the_counter(item, made, 1.06)


@pytest.mark.parametrize("text", SHOP_GOODS)
def test_the_second_stalls_goods_are_what_it_sells_and_stay_on_its_counter(text, signs, library):
    """Fruit heaped in shallow crates, flowers in buckets, sweets in jars, in the colours the placeholder gives
    its goods; none of it a ball on a stand, all of it on the counter and under the camera's height."""
    item, primitives, made, goods = shop_goods(signs, library, text)
    parts = {part.solid: part for part in made.parts}
    assert len(goods) >= 15
    for solid, (x0, y0, x1, y1) in goods:
        assert GOODS_BOX[0] - 1e-9 <= x0 and x1 <= GOODS_BOX[2] + 1e-9, solid
        assert GOODS_BOX[1] - 1e-9 <= y0 and y1 <= GOODS_BOX[3] + 1e-9, solid
        assert heights(solid)[1] <= GOODS_TOP_M, solid
    # The placeholder's goods are gone: no ball 0.12 m in radius, and no stand under one.
    was = [solid for solid in primitives if not isinstance(solid, Box)]
    assert len(was) == 10 and not any(solid in parts for solid in was)
    assert not any(isinstance(solid, Sphere) and solid.radius >= 0.1 for solid, _ in goods)
    # Every colour of the goods is there, as paint or as a tint.
    shown = {tuple(parts[solid].tint or solid.color) for solid, _ in goods}
    assert set(SHOP_GOODS[text]) <= shown
    balls = [solid for solid, _ in goods if isinstance(solid, Sphere)]
    boxes = [solid for solid, _ in goods if isinstance(solid, Box)]
    round_ones = [solid for solid, _ in goods if not isinstance(solid, (Box, Sphere))]
    if text == "FRUIT":                             # pieces of fruit, a handful to a crate, in crates of wood
        assert len(balls) >= 30 and all(0.025 <= ball.radius <= 0.05 for ball in balls)
        assert len(boxes) >= 5 and {parts[box].material for box in boxes} <= {"wood", "wood_dark", "cardboard"}
        for ball in balls:                          # each lies in a crate: over one of them, and not beside it
            assert any(gap((ball.x, ball.y), box) == 0.0 for box in boxes), ball
        assert max(heights(box)[1] for box in boxes) <= 1.06 + 0.12       # shallow
    if text == "FLOWERS":                           # bunches in buckets
        assert len(round_ones) >= 5 and len(balls) >= 25
        for ball in balls:                          # every bloom and every leaf is over a bucket's ground or near it
            assert min(math.hypot(ball.x - pot.x, ball.y - pot.y) for pot in round_ones) <= 0.14, ball
    if text == "SWEETS":                            # jars with lids: upright and round, none of them of glass
        assert len(round_ones) >= 10 and not boxes and not any(parts[solid].glass for solid, _ in goods)
        jars = [solid for solid in round_ones if solid.height >= 0.1]
        assert len(jars) >= 5 and all(any(heights(lid)[0] == pytest.approx(heights(jar)[1])
                                          and math.hypot(lid.x - jar.x, lid.y - jar.y) < 1e-6
                                          for lid in round_ones) for jar in jars)


def test_the_goods_are_the_same_whatever_the_signage_and_other_words_keep_the_placeholders(library):
    for text in SHOP_GOODS:
        shown = [[solid for solid, _ in shop_goods(mode, library, text)[3]] for mode in SIGN_MODES]
        assert shown[0] == shown[1], text
    # A shop of which nothing says what it sells keeps the balls on their stands, restyled.
    item, primitives = an_item("shop", text="SUNDRIES")
    made = construct(item, primitives, "natural", EMPTY, rng())
    assert all(solid in {part.solid for part in made.parts} for solid in primitives if not isinstance(solid, Box))


def test_a_construction_makes_no_generator_of_its_own(monkeypatch):
    """What varies within a construction, the fruit in a crate as much as the name on a logo, follows from the
    one number it takes: a dressing has one generator for all its constructions, and counts on that."""
    streams = {(kind, text): rng(3) for kind in KINDS for text in (SHOP_GOODS if kind == "shop" else (None,))}

    def refuse(*seed):
        raise AssertionError(f"a construction made a generator of its own, seeded with {seed}")

    monkeypatch.setattr(np.random, "default_rng", refuse)
    for (kind, text), stream in streams.items():
        item, primitives = an_item(kind, **({} if text is None else {"text": text, "goods": SHOP_GOODS[text]}))
        for mode in SIGN_MODES:
            assert construct(item, primitives, mode, SMALL, stream).obstacles


def test_the_staffed_kinds_are_the_three_service_points_and_the_second_stall():
    """With the second stall among them, the place a request names is not the only one in a scene with someone
    behind it."""
    assert STAFFED_KINDS == ("stall", "desk", "ticket_counter", "shop") and SIGN_MODES == ("natural", "labels")
    assert SERVED == ("stall", "desk", "ticket_counter")


SHOP_BOARD = (0.0, -1.0, 0.08, 1.0)                 # the shop's back board on the wall, in its frame
SHOP_STRIP_M = 0.47                                 # from the board to the counter (GOODS_BOX): where its vendor stands
GOODS_Y = (-0.72, -0.36, 0.0, 0.36, 0.72)           # where the placeholder's five goods stand along the counter


@pytest.mark.parametrize("text", SHOP_GOODS)
def test_the_second_stall_has_its_staff_spot_on_the_strip_between_its_board_and_its_counter(text, signs, library):
    """One spot, facing into the room, in the middle of the strip's depth and beside the middle of the goods,
    between two of them: on free floor whatever the grid, with room for a body, and with head and shoulders
    showing over the goods from the front."""
    for seed in range(6):
        item, primitives, made, goods = shop_goods(signs, library, text, seed=seed)
        (spot,) = made.staff_spots
        x, y, yaw = spot
        assert yaw == pytest.approx(item.yaw)       # +x of a furnishing's frame: into the room, to the customers
        lx, ly = in_frame(item, (x, y))
        assert lx == pytest.approx((SHOP_BOARD[2] + GOODS_BOX[0]) / 2) and ly == pytest.approx(-0.18)
        assert all(abs(ly - at) >= 0.18 - 1e-9 for at in GOODS_Y)      # between two of the goods, not behind one
        assert point_in_polygon((x, y), item.footprint)
        assert all(distance_to_polygon((x, y), obstacle) >= BODY_M for obstacle in made.obstacles)
        for solid in solids(made):                  # room for a body: nothing lower than the camera within 0.2 m
            if heights(solid)[0] < EYE_M:
                assert gap((x, y), solid) >= BODY_M - 1e-9, solid
        for solid in solids(made):                  # nothing stands before their head and shoulders
            low, high = heights(solid)
            if low < 1.6 and high > 1.45:
                assert not any(px > lx and abs(py - ly) < BODY_M for px, py in
                               (in_frame(item, point) for point in ground(solid))), solid
    # A staff spot is ground truth: it is the same whatever the item draws and whatever the signage. Of the two
    # gaps beside the middle it is the one at -0.18 wherever the stall stands: the side away from the price board,
    # which hangs over the other half of the counter and would have a shoulder before its near edge.
    assert len({built("shop", mode, library, seed=seed, text=text, goods=SHOP_GOODS[text])[2].staff_spots
                for mode in SIGN_MODES for seed in range(6)}) == 1
    for k in range(12):
        item, _, made = built("shop", signs, library, frame=(0.7 * k, -1.3 * k, 0.4 * k), text=text,
                              goods=SHOP_GOODS[text])
        assert in_frame(item, made.staff_spots[0][:2])[1] == pytest.approx(-0.18), k
    item, _, made = built("shop", "natural", library, text=text, goods=SHOP_GOODS[text])
    ((board, prices),) = [(part, sign) for part, sign in signs_of(made) if sign.art.style == "menu"]
    middle, _, _, width, _ = face_rectangle(board.solid, prices.face, prices.rect)
    near_edge = in_frame(item, middle[:2])[1] - width / 2
    assert near_edge == pytest.approx(0.45) and near_edge - in_frame(item, made.staff_spots[0][:2])[1] >= 0.6


@pytest.mark.parametrize("text", SHOP_GOODS)
def test_the_second_stall_blocks_its_board_and_its_counter_and_leaves_the_strip_free(text, signs, library):
    """Where its twin has the one rectangle of all its ground: ground truth says what is drawn. The posts of
    its sign stand on the counter's ground, and are no obstacle of their own. Nothing of it is moved for the
    vendor: every box the placeholder draws is still there, in its place and at its size."""
    item, primitives, made, goods = shop_goods(signs, library, text)
    assert sides(item, item.footprint) == pytest.approx((0.0, -1.0, 1.05, 1.0))
    board, counter = made.obstacles
    assert sides(item, board) == pytest.approx(SHOP_BOARD) and sides(item, counter) == pytest.approx(GOODS_BOX)
    assert GOODS_BOX[0] - SHOP_BOARD[2] == pytest.approx(SHOP_STRIP_M)
    for obstacle in made.obstacles:                 # each a rectangle of four corners, on the item's own ground
        assert len(obstacle) == 4 and all(distance_to_polygon(corner, item.footprint) <= 1e-9 for corner in obstacle)
    assert_covered(made, made.obstacles)
    # The strip is floor: nothing that the robot's map would hold stands on it, from end to end of the stall.
    for at_x in np.arange(SHOP_BOARD[2] + 0.02, GOODS_BOX[0] - 0.01, 0.05):
        for at_y in np.arange(-0.975, 1.0, 0.05):
            point = to_world(FRAME, float(at_x), float(at_y))
            assert all(distance_to_polygon(point, obstacle) > 0.0 for obstacle in made.obstacles)
            assert all(gap(point, solid) > 0.0 for solid in solids(made) if heights(solid)[0] < EYE_M), (at_x, at_y)
    # What the placeholder builds it of, its goods apart: each box where it was, as large as it was.
    drawn = [part.solid for part in made.parts if isinstance(part.solid, Box)]
    for box in (solid for solid in primitives if isinstance(solid, Box)):
        low, high = heights(box)
        whole = [solid for solid in drawn if np.allclose(corners(solid), corners(box), atol=1e-9)
                 and heights(solid)[0] >= low - 1e-9 and heights(solid)[1] <= high + 1e-9]   # a banded box is in pieces
        assert whole and min(heights(solid)[0] for solid in whole) == pytest.approx(low), box
        assert max(heights(solid)[1] for solid in whole) == pytest.approx(high), box


def test_an_unknown_kind_and_an_unknown_signage_are_errors():
    item, primitives = an_item("bench")
    with pytest.raises(ValueError, match="unknown signage 'none'.*natural, labels"):
        construct(item, primitives, "none", EMPTY, rng())
    with pytest.raises(ValueError, match="unknown kind 'fountain'.*stall"):
        construct(dataclasses.replace(item, kind="fountain"), primitives, "natural", EMPTY, rng())


@pytest.mark.parametrize("kind", KINDS)
def test_parts_and_models_stay_on_the_item_footprint_below_the_camera_height(kind, signs, library):
    item, primitives, made = built(kind, signs, library)
    assert_on_footprint(item, made)


@pytest.mark.parametrize("kind", KINDS)
def test_obstacles_cover_everything_solid_between_15_cm_and_the_camera_height(kind, signs, library):
    item, primitives, made = built(kind, signs, library)
    assert_covered(made, made.obstacles)
    # And they lie on the item's own ground.
    for obstacle in made.obstacles:
        assert all(distance_to_polygon(corner, item.footprint) <= 1e-9 for corner in obstacle)


@pytest.mark.parametrize("kind", [kind for kind in KINDS if kind not in (*STAFFED_KINDS, "parcel_table")])
def test_an_item_that_is_an_obstacle_of_its_twin_blocks_exactly_its_footprint(kind, signs, library):
    """A furnishing other than the second stall, a kiosk and a back room: one obstacle, corner for corner the
    item's rectangle."""
    item, primitives, made = built(kind, signs, library)
    assert [[tuple(corner) for corner in obstacle] for obstacle in made.obstacles] == \
        [[tuple(corner) for corner in item.footprint]]


def test_a_parcel_table_blocks_the_table_and_the_crate_and_leaves_the_floor_between_them_free(signs, library):
    item, primitives, made = built("parcel_table", signs, library)
    table, crate = made.obstacles
    assert np.allclose(table, recipes._rectangle(primitives[0]))        # as the twin's obstacles are made
    assert np.allclose(crate, recipes._rectangle(primitives[2]))
    assert sides(item, table) == pytest.approx((-0.45, -0.65, 0.45, 0.65))
    assert sides(item, crate) == pytest.approx((0.75, 0.2, 1.05, 0.5))
    between = to_world(FRAME, 0.6, 0.35)             # the floor from the table to the crate
    assert point_in_polygon(between, item.footprint)
    assert all(distance_to_polygon(between, obstacle) >= 0.1 for obstacle in made.obstacles)
    assert all(gap(between, solid) >= 0.1 for solid in solids(made))


@pytest.mark.parametrize("kind", PLACES)
def test_a_place_keeps_its_front_face_under_the_anchor(kind, signs, library):
    """The anchor is at (0, 0, 1.0) on the front, in the place's frame: a part's face lies on x = 0 there and
    0.3 m to each side, from the floor to the top of the counter at 1.1 m."""
    item, primitives, made = built(kind, signs, library)

    def faces(part, y, z):
        box = part.solid
        if part.glass or not isinstance(box, Box) or abs(wrap_angle(box.yaw - item.yaw)) > 1e-9:
            return False
        x, across = in_frame(item, (box.cx, box.cy))
        return (abs(x + box.sx / 2) <= 1e-6 and abs(y - across) <= box.sy / 2 + 1e-9
                and box.z0 - 1e-9 <= z <= box.z0 + box.sz + 1e-9)

    for y in (-0.3, -0.15, 0.0, 0.15, 0.3):
        for z in (0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.05, 1.1):
            assert any(faces(part, y, z) for part in made.parts), (y, z)
    # Nothing stands before it.
    assert all(in_frame(item, point)[0] <= 0.01 + 1e-9 for solid in solids(made) for point in ground(solid))


@pytest.mark.parametrize("deep", (False, True), ids=("alone", "deep"))
@pytest.mark.parametrize("kind", SERVED)
def test_a_staffed_place_leaves_a_free_strip_with_its_staff_spot_on_it(kind, deep, signs, library):
    item, primitives, made = built(kind, signs, library)
    obstacles, standing = list(made.obstacles), solids(made)
    if deep:
        room, walls = an_item("back_room", text=item.options["text"], color=item.options["color"])
        behind = construct(room, walls, signs, library, rng(1))
        assert not behind.staff_spots
        assert not point_in_polygon(made.staff_spots[0][:2], room.footprint)
        obstacles, standing = obstacles + list(behind.obstacles), standing + solids(behind)
    assert_staffed(item, made, obstacles, standing)


@pytest.mark.parametrize("kind", SERVED)
def test_a_staffed_place_blocks_what_it_draws_and_no_more(kind, signs, library):
    """The obstacles are rectangles square to the place, apart from one another, and each has something of
    the place standing on it from the floor up: none is there for nothing."""
    item, primitives, made = built(kind, signs, library)
    expected = {"stall": 2, "ticket_counter": 4, "desk": 2 + (2 if signs == "labels" else 0)}[kind]
    assert len(made.obstacles) == expected           # the desk: its strip, its chair, and under the sign two rear posts
    low = [point for solid in solids(made) if heights(solid)[0] < FLOOR_M for point in ground(solid)]
    for obstacle in made.obstacles:
        x0, y0, x1, y1 = sides(item, obstacle)
        own = [in_frame(item, corner) for corner in obstacle]
        assert len(own) == 4 and all(any(math.dist(corner, at) < 1e-9 for corner in own)
                                     for at in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
        assert any(distance_to_polygon(point, obstacle) == 0.0 for point in low), obstacle
    for one in made.obstacles:                       # apart: no corner of one in another
        assert not any(point_in_polygon(corner, other) and distance_to_boundary(corner, other) > 1e-9
                       for other in made.obstacles if other is not one for corner in one)
    # The floor behind the counter is not blocked: of the place's 1.28 square metres, at least 0.4 are free.
    cells = [to_world(FRAME, x, y) for x in np.arange(-0.775, 0.0, 0.05) for y in np.arange(-0.775, 0.8, 0.05)]
    free = [cell for cell in cells if all(distance_to_polygon(cell, obstacle) > 0 for obstacle in made.obstacles)]
    assert len(free) * 0.05 * 0.05 >= 0.4


def test_the_ticket_counter_is_glazed_before_its_staff_and_at_both_sides_above_the_counter(signs, library):
    item, primitives, made = built("ticket_counter", signs, library)
    panes = [part.solid for part in made.parts if part.glass]
    assert panes and all(part.material == "glass" for part in made.parts if part.glass)
    assert all(heights(pane)[0] >= 1.1 - 1e-9 for pane in panes)          # solid up to the counter's top
    (spot,) = made.staff_spots
    lx, ly = in_frame(item, spot[:2])
    where = [sides(item, corners(pane)) for pane in panes]
    # Before them: a pane across their place, at head height.
    assert any(x0 > lx and y0 <= ly - BODY_M and y1 >= ly + BODY_M and heights(pane)[1] >= 1.8
               for pane, (x0, y0, x1, y1) in zip(panes, where))
    # At each side: a pane along the strip they stand on, from the counter's top to above their head.
    for side in (1, -1):
        assert any(side * (y0 + y1) / 2 > 0.7 and x0 <= lx - BODY_M and x1 >= lx + BODY_M
                   and heights(pane)[0] <= 1.1 + 1e-9 and heights(pane)[1] >= 1.8
                   for pane, (x0, y0, x1, y1) in zip(panes, where)), side
    # A gap at counter height, before the staff: no pane there.
    hatch = to_world(FRAME, max(x0 for x0, *_ in where if x0 > lx) + 0.005, ly)
    assert not any(gap(hatch, pane) == 0.0 and heights(pane)[0] < 1.25 for pane in panes)
    # No glass anywhere else: of the other kinds none has a pane.
    for kind in (*UNSTAFFED, "shop"):
        assert not any(part.glass for part in built(kind, signs, library)[2].parts), kind


@pytest.mark.parametrize("kind", UNSTAFFED)
def test_the_kiosk_and_the_furnishings_have_no_staff_spot(kind, signs, library):
    assert kind == "kiosk" or kind not in PLACES
    assert built(kind, signs, library)[2].staff_spots == ()


@pytest.mark.parametrize("kind", KINDS)
def test_labels_mode_keeps_the_placeholder_lettering_rectangles(kind, library):
    """One Lettered for each box the placeholder letters, in their order, with the centre, the normal and the
    size of its lettering; and on each a sign with the placeholder's words in its colours, at its size."""
    for chosen in variants(kind):
        item, primitives, made = built(kind, "labels", library, **chosen)
        lettered = [box for box in primitives if isinstance(box, Box) and box.label]
        assert len(made.lettering) == len(lettered)
        labels = [(part, sign) for part, sign in signs_of(made) if sign.art.style == "label"]
        assert len(labels) == len(lettered)
        assert not any(sign.art.style in NAMING for _, sign in signs_of(made) if sign.art.style != "label")
        for box, found, (part, sign) in zip(lettered, made.lettering, labels):
            assert is_at(found, placeholder_lettering(box)), box
            assert sign.art.text == box.label
            assert sign.art.ground == (box.panel if box.panel is not None else box.color)
            assert sign.art.ink == box.label_color
            # The sign lies where the lettering is, no further off the placeholder's face than a plate is thick,
            # and signs.draw fills 0.86 by 0.62 of it with letters: the placeholder's size.
            middle, normal, across, width, height = face_rectangle(part.solid, sign.face, sign.rect)
            assert np.allclose(middle, found.centre, atol=0.005) and np.allclose(normal, found.normal)
            assert np.allclose(across, found.across)
            assert LABEL_FILLS[0] * width == pytest.approx(found.width)
            assert LABEL_FILLS[1] * height == pytest.approx(found.height)
    assert bool(lettered) == (kind in (*PLACES, "back_room", "poster", "shop"))


@pytest.mark.parametrize("kind", KINDS)
def test_natural_mode_has_no_sign_wider_than_half_a_metre_on_a_target(kind, library):
    """Construction.lettering lists what names the item: a logo, a plate, a pictogram. None is a label, none is
    wider than 0.5 m, and each is a sign that the construction draws just there."""
    for chosen in variants(kind):
        item, primitives, made = built(kind, "natural", library, **chosen)
        assert not any(sign.art.style == "label" for _, sign in signs_of(made))
        assert not any(isinstance(part.solid, Box) and part.solid.label for part in made.parts)
        assert all(lettered.width <= 0.5 + 1e-9 for lettered in made.lettering)
        naming = [face_rectangle(part.solid, sign.face, sign.rect) for part, sign in signs_of(made)
                  if sign.art.style in NAMING]
        assert len(naming) == len(made.lettering)
        for lettered in made.lettering:
            assert any(is_at(lettered, rectangle, off=0.005) for rectangle in naming), lettered
    if kind in PLACES:                              # a place still shows what it is
        assert made.lettering


@pytest.mark.parametrize("kind", KINDS)
def test_natural_mode_sets_no_lettering_taller_than_4_cm_on_any_sign(kind, library):
    """Nothing in natural mode spells a target out in large letters (spec 5.3). The half metre of the test above
    bounds how wide a sign that names its item is, and leaves its letters as tall as its rectangle lets them be:
    a plate half a metre square reads 10 cm tall. So the letters are measured, as the module that draws them
    sets them: on whatever a construction shows, a plate, a logo, a screen, a menu, no capital stands taller
    than 4 cm. A sign that names its item keeps to that with any seed: a logo's seed, which comes from where the
    place stands, picks its made-up name, and a short name is set larger than a long one."""
    sign_art()
    for chosen in variants(kind):
        for seed in range(3):
            for _, sign in signs_of(built(kind, "natural", library, seed=seed, **chosen)[2]):
                tried = [sign.art]
                if sign.art.style in NAMING:
                    tried += [dataclasses.replace(sign.art, seed=other) for other in range(NAMES_TRIED)]
                for art in tried:
                    assert tallest_type(art) <= TYPE_M + 1e-9, (art, tallest_type(art))
                if sign.art.style in ("logo", "plate"):      # these two say something in words: it is their
                    assert tallest_type(sign.art) > 0.0, sign.art         # letters that were measured, not nothing


@pytest.mark.parametrize("kind", KINDS)
def test_a_sign_has_art_of_the_size_of_the_rectangle_it_fills(kind, signs, library):
    item, primitives, made = built(kind, signs, library)
    for part, sign in signs_of(made):
        assert isinstance(sign, Sign) and isinstance(sign.art, SignArt) and isinstance(part.solid, Box)
        assert sign.face in ("front", "back", "left", "right") and sign.art.style in SIGN_STYLES
        left, bottom, right, top = sign.rect
        assert -1e-9 <= left < right <= 1 + 1e-9 and -1e-9 <= bottom < top <= 1 + 1e-9
        _, _, _, width, height = face_rectangle(part.solid, sign.face, sign.rect)
        assert sign.art.width_m == pytest.approx(width) and sign.art.height_m == pytest.approx(height)
        for colour in (sign.art.ink, sign.art.ground):
            assert len(colour) == 3 and all(type(c) is int and 0 <= c <= 255 for c in colour)
        if sign.art.style == "pictogram":
            assert sign.art.text in PICTOGRAMS
        # What glows: a screen, an exit sign, and the strip of the light fitting in a booth, which is a blank sheet.
        lamp = kind == "ticket_counter" and sign.art.style == "paper" and sign.art.ink == sign.art.ground
        assert sign.emission >= 0.0 and (sign.emission == 0.0 or sign.art.style in ("screen", "exit") or lamp)


NATURAL = {   # kind: the styles of sign it shows in natural mode (spec 5.3), and those of them that name it
    "desk": ({"plate", "screen", "pictogram"}, {"plate", "pictogram"}),
    "stall": ({"logo", "menu", "screen"}, {"logo"}),
    "ticket_counter": ({"plate", "menu", "paper"}, {"plate"}), "kiosk": ({"screen", "pictogram"}, {"pictogram"}),
    "shop": ({"logo", "menu"}, {"logo"}), "back_room": ({"plate"}, {"plate"}), "board": ({"notice", "paper"}, set()),
    "parcel_table": ({"paper"}, set())}


@pytest.mark.parametrize("kind", NATURAL)
def test_natural_mode_shows_what_a_real_place_has(kind):
    shown, naming = NATURAL[kind]
    item, primitives, made = built(kind, "natural", EMPTY)
    styles = {sign.art.style for _, sign in signs_of(made)}
    assert styles == shown
    assert {style for style in styles if style in NAMING} == naming
    if kind == "desk":                              # no canopy on posts: what is over it hangs from the ceiling
        assert not any(heights(solid)[0] < OVERHEAD_M < heights(solid)[1] for solid in solids(made))
        posted = solids(built(kind, "labels", EMPTY)[2])
        assert any(heights(solid)[0] < OVERHEAD_M < heights(solid)[1] for solid in posted)
    if kind in ("stall", "ticket_counter", "kiosk"):         # the header stays, with the small sign on its three sides
        marks = [(part, sign) for part, sign in signs_of(made) if sign.art.style in naming]
        assert len(marks) == 3 and all(part.solid.z0 >= OVERHEAD_M for part, _ in marks)
        assert {sign.face for _, sign in marks} == {"front", "left", "right"}
        assert len({sign.art for _, sign in marks}) == 1     # one name, one mark, on all three


HUNG_Z_M = 1.9                                    # what hangs over a place, and a light fitting, begin no lower
CEILING_M = recipes._ROOM_HEIGHT_M                  # where it hangs from


def test_the_help_desk_shows_its_pictogram_from_the_front_and_from_both_sides_in_natural_mode(library):
    """Without its canopy a desk was a white box with a plate a few pixels wide, and from along it, which is how
    a waiting line is seen, nothing said what it was. An information point has a sign overhead: here a cube in
    the desk's colour hung from the ceiling over it, above every head, with the pictogram on its four sides and
    not a word; and the pictogram again on the front, 0.2 m high. Each is listed as lettering, none wider than
    half a metre. Nothing of it stands on the floor, so the desk blocks what it did."""
    item, primitives, made = built("desk", "natural", library)
    colour = tuple(item.options["color"])
    symbols = [(part, sign) for part, sign in signs_of(made) if sign.art.style == "pictogram"]
    assert all(sign.art.text == "information" and sign.art.ground == colour and sign.art.ink == (255, 255, 255)
               and sign.emission == 0.0 for _, sign in symbols)
    hung = [(part, sign) for part, sign in symbols if heights(part.solid)[0] >= HUNG_Z_M]
    assert {sign.face for _, sign in hung} == {"front", "back", "left", "right"} and len(hung) == 4
    (cube,) = {part.solid for part, _ in hung}
    # As high as the placeholder builds a place and no higher: the robot's view of the twin holds it.
    assert cube.sx == cube.sy == pytest.approx(0.35) and heights(cube) == pytest.approx((1.9, recipes._PLACE_TOP_M))
    assert all(0.3 <= sign.art.width_m == sign.art.height_m <= 0.35 for _, sign in hung)
    assert sides(item, corners(cube))[2] <= 0.0 and all(      # over the desk, behind its front, on its ground
        distance_to_polygon(corner, item.footprint) == 0.0 for corner in corners(cube))
    # It hangs: rods from its top to the ceiling, and nothing under it down to the desk.
    rods = [solid for solid in solids(made) if not isinstance(solid, (Box, Sphere))
            and heights(solid)[0] == pytest.approx(heights(cube)[1])]
    assert len(rods) == 2 and all(heights(rod)[1] == pytest.approx(CEILING_M) and gap((rod.x, rod.y), cube) == 0.0
                                  for rod in rods)
    assert constructions._CEILING_M == CEILING_M
    assert not any(heights(solid)[0] < HUNG_Z_M < heights(solid)[1] for solid in solids(made))
    # The pictogram on the front: about 0.24 m square, so that its symbol is 0.2 m, under the top of the front.
    ((front, on_front),) = [(part, sign) for part, sign in symbols if heights(part.solid)[0] < HUNG_Z_M]
    assert on_front.face == "front" and on_front.art.width_m == on_front.art.height_m == pytest.approx(0.24)
    assert sides(item, corners(front.solid))[2] == pytest.approx(0.0)
    # All five are lettering of the place, with the name card's plate: each no wider than half a metre.
    rectangles = [face_rectangle(part.solid, sign.face, sign.rect) for part, sign in symbols]
    assert all(any(is_at(lettered, rectangle, off=0.005) for lettered in made.lettering) for rectangle in rectangles)
    assert len(made.lettering) == 6 and all(lettered.width <= 0.5 for lettered in made.lettering)
    # The control sets keep the placeholder's canopy with its words, and gain nothing.
    labels = built("desk", "labels", library)[2]
    assert not any(sign.art.style == "pictogram" for _, sign in signs_of(labels))
    assert len(labels.obstacles) == len(made.obstacles) + 2 and labels.staff_spots == made.staff_spots


def test_the_chair_behind_a_desk_is_a_swivel_chair_on_the_square_the_placeholders_fills(signs):
    """Where no model goes in, it is built of parts: five spokes on castors, a column, a seat and a low back in
    the twin's cloth, and no slab of a head rest showing over the ledge. Its obstacle is the placeholder's chair's
    square, moved to where the chair stands."""
    item, primitives, made = built("desk", signs, EMPTY)
    was = [solid for solid in recipes._office_chair(-0.6, -0.22, (190, 96, 60))]
    x0, y0, x1, y1 = recipes._extent(was)
    (square,) = [obstacle for obstacle in made.obstacles
                 if sides(item, obstacle)[2] - sides(item, obstacle)[0] == pytest.approx(x1 - x0)
                 and sides(item, obstacle)[3] - sides(item, obstacle)[1] == pytest.approx(y1 - y0)]
    assert sides(item, square)[1::2] == pytest.approx((y0, y1))           # at the side of the place it was at
    chair = [solid for solid in solids(made) if heights(solid)[1] < 1.1 and all(
        distance_to_polygon(point, square) <= 1e-9 for point in ground(solid))]
    spokes = [solid for solid in chair if isinstance(solid, Box) and heights(solid)[1] < 0.1]
    castors = [solid for solid in chair if isinstance(solid, Sphere)]
    assert len(spokes) == len(castors) == 5 and len({round(spoke.yaw, 3) for spoke in spokes}) == 5
    cloth = [part.solid for part in made.parts if part.material == "fabric"]
    assert len(cloth) == 2 and all(solid in chair and solid.color == (190, 96, 60) for solid in cloth)
    assert max(heights(solid)[1] for solid in chair) <= 1.1               # under the ledge's top: 1.14 m
    assert not any(solid in [part.solid for part in made.parts] for solid in primitives
                   if not isinstance(solid, Box) and heights(solid)[1] < 0.5)       # the placeholder's foot and column


def test_the_ticket_booth_has_a_light_fitting_inside_under_its_sign(signs, library):
    """So that whoever stands in it is seen from the side and not a dark figure behind glass: a fitting hung from
    the roof of the sign, above every head, between the screen and the staff, with a strip that glows towards
    them more strongly than a screen does. The same in either signage."""
    item, primitives, made = built("ticket_counter", signs, library)
    ((part, sign),) = [(part, sign) for part, sign in signs_of(made) if sign.art.style == "paper"]
    low, high = heights(part.solid)
    assert low >= HUNG_Z_M and high == pytest.approx(recipes._PLACE_TOP_M - 0.04)     # up to the sign's roof
    assert sign.emission >= 5.0 and sign.art.ink == sign.art.ground and min(sign.art.ground) >= 225
    middle, normal, _, width, height = face_rectangle(part.solid, sign.face, sign.rect)
    assert middle[2] >= HUNG_Z_M and width >= 0.8 and 0.03 <= height <= 0.1
    (spot,) = made.staff_spots
    towards = np.array([spot[0] - middle[0], spot[1] - middle[1]])
    assert normal[:2] @ towards > 0.1               # it faces the staff, who stand behind it
    lx = in_frame(item, spot[:2])[0]
    panes = [sides(item, corners(one.solid)) for one in made.parts if one.glass]
    screen = max(x0 for x0, _, _, _ in panes if x0 > lx)
    assert lx + 0.1 < sides(item, corners(part.solid))[0] and sides(item, corners(part.solid))[2] < screen
    other = built("ticket_counter", SIGN_MODES[1 - SIGN_MODES.index(signs)], library)[2]
    assert part in other.parts


def test_a_poster_becomes_a_map_an_exit_sign_or_notices_in_natural_mode():
    wanted = {"MAP": {"map"}, "EXIT": {"exit"}, "NOTICES": {"notice"}, "EVENTS": {"notice"}}
    for text, ink, width in recipes._POSTERS:
        item, primitives, made = built("poster", "natural", EMPTY, text=text, ink=ink, width=width)
        found = [sign for _, sign in signs_of(made)]
        assert {sign.art.style for sign in found} == wanted[text]
        assert len(found) >= (3 if wanted[text] == {"notice"} else 1)
        assert len(made.lettering) == (1 if text == "EXIT" else 0)       # an exit sign names its place; a map does not
        if text == "EXIT":                          # a standard small sign, lit, where the head of the poster was
            (sign,) = found
            assert sign.art.width_m <= 0.5 and sign.emission > 0
            assert all(heights(solid)[0] >= 1.9 for solid in solids(made))
        if wanted[text] == {"notice"}:              # sheets of paper, none larger than A3, each its own
            assert all(sign.art.width_m <= 0.3 and sign.art.height_m <= 0.42 for sign in found)
            assert len({sign.art.seed for sign in found}) == len(found)


def test_the_front_and_the_back_room_of_a_place_show_one_name_in_natural_mode():
    """The two items of a deep place are built apart, and what names the place has to agree between them."""
    marks = []
    for kind in ("stall", "back_room"):
        item, primitives, made = built(kind, "natural", EMPTY, text="COFFEE", color=(126, 58, 40))
        marks.append({(sign.art.text, sign.art.seed) for _, sign in signs_of(made) if sign.art.style in NAMING})
    assert len(marks[0]) == len(marks[1]) == 1 and marks[0] == marks[1]
    # Another place has another name.
    elsewhere = construct(*an_item("stall", frame=(-4.0, 2.5, 0.3)), "natural", EMPTY, rng())
    assert {(sign.art.text, sign.art.seed) for _, sign in signs_of(elsewhere) if sign.art.style in NAMING} != marks[0]


@pytest.mark.parametrize("kind", KINDS)
def test_sign_art_is_drawable_and_what_does_not_name_a_target_is_set_in_small_type(kind):
    """A menu, a price list, a map or a notice may be wider than half a metre and is no lettering of the target:
    its type is no taller than 4 cm, in either mode. Measured by the module that draws it, once that draws."""
    art_module = sign_art()
    for mode in SIGN_MODES:
        for chosen in variants(kind):
            for seed in range(3):
                for _, sign in signs_of(built(kind, mode, EMPTY, seed=seed, **chosen)[2]):
                    art = sign.art
                    picture = art_module.draw(art, px_per_m=64)
                    assert picture.shape[:2] == (max(8, round(art.height_m * 64)), max(8, round(art.width_m * 64)))
                    if art.style in ("menu", "map", "notice"):
                        assert 0.0 < tallest_type(art) <= TYPE_M + 1e-9, (art, tallest_type(art))


@pytest.mark.parametrize("kind", (*MODELLED, "desk", "stall"))
def test_a_kind_without_a_model_is_restyled(kind, signs):
    """With an empty library, parts only, on the ground the placeholder's own primitives fill."""
    item, primitives, made = built(kind, signs, EMPTY)
    assert made.models == () and made.parts
    if kind in MODELLED:
        assert sides(item, [point for solid in solids(made) for point in ground(solid)]) == \
            pytest.approx(sides(item, item.footprint), abs=0.005)        # a plate on a face is 3 mm proud of it
        assert max(heights(solid)[1] for solid in solids(made)) == pytest.approx(max(heights(p)[1] for p in primitives))
        assert len(made.parts) >= len(primitives)   # every primitive is there, and what was painted on some
    # A model too large for its place by more than 15 % is left out the same way: the cooler and the chair.
    item, primitives, made = built(kind, signs, SMALL)
    assert not {model.asset for model in made.models} & {"cooler_wide", "chair_a", "lockers_branded"}
    if kind == "cooler":
        assert made.models == () and made.parts


@pytest.mark.parametrize("kind", ("plant", "bench", "lockers"))
def test_a_furnishing_with_a_model_that_fits_is_the_model_alone(kind, signs):
    item, primitives, made = built(kind, signs, SMALL)
    (model,) = made.models
    assert made.parts == () and SMALL.models[model.asset].kind == kind
    assert model.yaw == pytest.approx(item.yaw) and model.z == 0.0       # on the floor, its front into the room
    x0, y0, x1, y1 = sides(item, corners(model_box(model)))
    fx0, fy0, fx1, fy1 = sides(item, item.footprint)
    assert fx0 - 1e-9 <= x0 and x1 <= fx1 + 1e-9 and fy0 - 1e-9 <= y0 and y1 <= fy1 + 1e-9
    assert (y0 + y1) / 2 == pytest.approx((fy0 + fy1) / 2)               # in the middle of its place along the wall
    if kind != "plant":                             # its back to the wall side of its rectangle
        assert x0 == pytest.approx(fx0)


def test_a_model_smaller_than_its_place_stands_at_the_wall_and_a_plant_in_the_middle(signs):
    """A model is enlarged by 15 % at the most, so a small one does not fill its rectangle: a bench keeps its
    back on the wall's side of it, a plant stands in the middle."""
    slim = a_library(a_model("bench_slim", "bench", (0.3, 1.2, 0.7)), a_model("plant_slim", "plant", (0.4, 0.4, 1.0)))
    for kind in ("bench", "plant"):
        item, primitives, made = built(kind, signs, slim)
        (model,) = made.models
        assert model.scale == pytest.approx((1.15, 1.15, 1.15))
        x0, y0, x1, y1 = sides(item, corners(model_box(model)))
        fx0, fy0, fx1, fy1 = sides(item, item.footprint)
        assert fx1 - x1 > 0.05 and y0 - fy0 > 0.05   # it leaves room
        assert (y0 + y1) / 2 == pytest.approx((fy0 + fy1) / 2)
        if kind == "bench":
            assert x0 == pytest.approx(fx0)
        else:
            assert (x0 + x1) / 2 == pytest.approx((fx0 + fx1) / 2)


def test_the_things_on_a_desk_and_a_stall_are_models_where_they_fit(signs):
    item, primitives, made = built("desk", signs, SMALL)
    assert [model.asset for model in made.models] == ["monitor_a"]       # the chair's model is too deep for the strip
    (monitor,) = made.models
    assert wrap_angle(monitor.yaw - item.yaw - math.pi) == pytest.approx(0.0, abs=1e-9)   # its screen to the staff
    assert 0.7 <= monitor.z <= 0.8                   # on the desk top
    # That one is taller than the ledge is high over the desk's top, and stands clear of the ledge. A screen
    # low enough stands under the ledge, where there is more room for its foot; one too deep for the first
    # place and too tall for the second is left out.
    assert sides(item, corners(model_box(monitor)))[2] <= -0.2 and monitor.z + monitor.size[2] > 1.14
    item, primitives, made = built("desk", signs, a_library(a_model("monitor_low", "monitor", (0.2125, 0.5, 0.3393))))
    (low,) = made.models
    assert sides(item, corners(model_box(low)))[2] > -0.2 and low.z + low.size[2] <= 1.1
    assert low.size[0] > 0.2 and all(0.85 <= s <= 1.15 for s in low.scale)
    assert not built("desk", signs, a_library(a_model("monitor_mid", "monitor", (0.2125, 0.5, 0.42))))[2].models
    item, primitives, made = built("stall", signs, SMALL)
    assert sorted(model.asset for model in made.models) == ["cup_a", "cup_a", "cup_a", "machine_a"]
    assert all(model.z == pytest.approx(1.1) for model in made.models)   # on the counter
    assert len({(round(model.x, 6), round(model.y, 6)) for model in made.models}) == 4


@pytest.mark.parametrize("kind", KINDS)
def test_a_model_shares_no_space_with_a_part_or_with_another_model(kind, signs):
    """A model is known by the box it fills alone: where that box is, nothing else of the construction is,
    whatever the model's shape. Touching is not sharing: a screen stands on its desk."""
    item, primitives, made = built(kind, signs, SMALL)
    boxes = [model_box(model) for model in made.models]
    for index, box in enumerate(boxes):
        inside = ground(Box(box.cx, box.cy, box.yaw, box.sx - 0.01, box.sy - 0.01, box.sz), n=9)
        low, high = heights(box)
        for other in (*(part.solid for part in made.parts), *boxes[:index], *boxes[index + 1:]):
            bottom, top = heights(other)
            if min(high, top) - max(low, bottom) > 1e-3:
                assert not any(gap(point, other) == 0.0 for point in inside), (box, other)
                assert not any(gap(point, box) < 0.004 for point in ground(other, n=9)), (box, other)


def test_a_model_is_used_only_between_85_and_115_percent_of_life_size():
    item, primitives = an_item("plant", scale=1.0)
    x0, y0, x1, y1 = sides(item, item.footprint)
    depth, width, height = x1 - x0, y1 - y0, max(heights(solid)[1] for solid in primitives)

    def placed(*size):
        made = construct(item, primitives, "natural", a_library(a_model("only", "plant", size)), rng())
        assert len(made.models) <= 1 and bool(made.models) != bool(made.parts)     # the model, or the restyle
        for model in made.models:
            assert all(0.85 - 1e-9 <= s <= 1.15 + 1e-9 for s in model.scale)
            assert model.size == pytest.approx(tuple(s * own for s, own in zip(model.scale, size)))
            mx0, my0, mx1, my1 = sides(item, corners(model_box(model)))
            assert x0 - 1e-9 <= mx0 and mx1 <= x1 + 1e-9 and y0 - 1e-9 <= my0 and my1 <= y1 + 1e-9
            assert model.size[2] <= 1.15 * height + 1e-9
        return made.models[0] if made.models else None

    assert placed(depth, width, height).scale == pytest.approx((1.0, 1.0, 1.0))
    # One that has to shrink by more than 15 % along any side of its ground does not go in.
    assert placed(depth / 0.84, width, height) is None
    assert placed(depth, width / 0.84, height) is None
    squeezed = placed(depth / 0.86, width / 0.9, height)
    assert squeezed.scale == pytest.approx((0.86, 0.9, 1.0)) and squeezed.size == pytest.approx((depth, width, height))
    # A small one is brought towards the placeholder's size by 15 % at most.
    assert placed(depth / 1.5, width / 1.1, height / 1.3).scale == pytest.approx((1.15, 1.1, 1.15))
    # Up, it ends no more than 15 % over the placeholder's height, and never under 85 % of its own.
    tall = placed(depth, width, height * 1.3)
    assert tall.scale[2] == pytest.approx(0.85) and tall.size[2] == pytest.approx(0.85 * 1.3 * height)
    assert placed(depth, width, height * 1.36) is None
    # A model that shows a brand is never used, however well it fits; nor one of another kind.
    for other in (a_model("branded", "plant", (depth, width, height), brand=True),
                  a_model("bench", "bench", (depth, width, height))):
        made = construct(item, primitives, "natural", a_library(other), rng())
        assert made.models == () and made.parts


def test_a_window_becomes_an_opening_and_draws_no_pane_into_the_room(signs, library):
    item, primitives, made = built("window", signs, library, width=1.83)
    (opening,) = made.openings
    assert opening == Opening("window", pytest.approx(item.x), pytest.approx(item.y), pytest.approx(item.yaw),
                              pytest.approx(1.83), pytest.approx(1.05), pytest.approx(1.2))
    # Nothing of it reaches into the room: a sill no deeper than 5 cm at the most, and no pane.
    assert not any(part.glass for part in made.parts) and made.models == ()
    assert all(in_frame(item, point)[0] <= 0.05 + 1e-9 for solid in solids(made) for point in ground(solid))
    assert made.lettering == () and made.surfaces == () and made.staff_spots == ()
    # No other kind but the door cuts into a wall.
    for kind in KINDS:
        assert bool(built(kind, signs, library)[2].openings) == (kind in ("window", "door")), kind


def test_a_door_becomes_an_opening_and_draws_nothing_into_the_room(signs, library):
    item, primitives, made = built("door", signs, library)
    (opening,) = made.openings
    assert opening == Opening("door", pytest.approx(item.x), pytest.approx(item.y), pytest.approx(item.yaw),
                              pytest.approx(1.0), pytest.approx(0.0), pytest.approx(2.1))
    assert all(in_frame(item, point)[0] <= 0.05 + 1e-9 for solid in solids(made) for point in ground(solid))


def test_an_opening_is_where_the_placeholder_has_it_on_the_wall():
    """Its place is read from the placeholder's box, not taken for the item's origin: a box off the origin moves it."""
    item, primitives = an_item("window", width=1.6)
    c, s = math.cos(item.yaw), math.sin(item.yaw)
    moved = [recipes._moved(solid, 1.0, 0.0, -s * 0.4, c * 0.4, 0.0) for solid in primitives]   # 0.4 m along the wall
    (opening,) = construct(item, moved, "natural", EMPTY, rng()).openings
    assert (opening.x, opening.y) == pytest.approx(to_world(FRAME, 0.0, 0.4))


@pytest.mark.parametrize("kind", KINDS)
def test_a_surface_is_a_free_top_out_of_the_staffs_reach(kind, signs, library):
    """A level rectangle on the item's own ground, below the camera, with something of the construction right
    under it, nothing on it or within 0.3 m over it, and no point of it within reach of a staff member."""
    item, primitives, made = built(kind, signs, library)
    for surface in made.surfaces:
        assert surface.yaw == pytest.approx(item.yaw) and FLOOR_M < surface.z < EYE_M
        assert surface.length >= 0.1 and surface.depth >= 0.08
        c, s = math.cos(surface.yaw), math.sin(surface.yaw)
        over = [(surface.x + c * i * surface.depth - s * j * surface.length,
                 surface.y + s * i * surface.depth + c * j * surface.length)
                for i in (-0.5, 0.0, 0.5) for j in (-0.5, 0.0, 0.5)]
        for point in over:
            assert distance_to_polygon(point, item.footprint) == 0.0
            assert any(gap(point, solid) == 0.0 and abs(heights(solid)[1] - surface.z) <= 1e-6
                       for solid in solids(made)), (surface, point)
            assert not any(gap(point, solid) < 0.01 and heights(solid)[0] < surface.z + 0.3
                           and heights(solid)[1] > surface.z + 1e-6 for solid in solids(made)), (surface, point)
            assert all(math.dist(point, spot[:2]) >= REACH_M for spot in made.staff_spots), (surface, point)
    if kind in (*PLACES, "parcel_table"):           # a counter, a desk, a table: each has room for something
        assert made.surfaces
    if kind == "bench":                             # its seat, where it is built of parts: a model's seat is not known
        assert bool(made.surfaces) == (not made.models)
    if kind == "parcel_table":                      # away from the parcel, which the target's match radius is round
        anchor = to_world(FRAME, 0.0, 0.0)
        assert all(math.dist(anchor, (surface.x, surface.y)) - math.hypot(surface.length, surface.depth) / 2 > 0.3
                   for surface in made.surfaces)


@pytest.mark.parametrize("kind", KINDS)
def test_the_same_draw_gives_the_same_construction_and_every_item_takes_one_number(kind):
    """So that a stream of draws gives every variant the same constructions: what an item takes from it does
    not depend on the signage, on the library or on its kind."""
    item, primitives = an_item(kind)
    first, again = (construct(item, primitives, "natural", SMALL, rng(5)) for _ in range(2))
    assert first == again
    after = []
    for mode in SIGN_MODES:
        for held in (SMALL, EMPTY):
            draws = rng(5)
            construct(item, primitives, mode, held, draws)
            after.append(int(draws.integers(1 << 30)))
    once = rng(5)
    once.integers(1 << 30)
    assert set(after) == {int(once.integers(1 << 30))}


def test_the_signage_changes_what_is_written_and_not_what_stands_there():
    """The variants of a scene show the same things. Between the two kinds of signage the models, the staff
    spots, the free tops and the openings are the same, and so is every part that begins lower than a head:
    but for a poster, which natural signage may replace, and a desk, which loses the posts and the lettered
    panel of its sign and gains a name card."""
    def low(made):
        return sorted(repr(solid) for solid in solids(made) if heights(solid)[0] < OVERHEAD_M)

    for kind in KINDS:
        item, primitives = an_item(kind)
        natural, labels = (construct(item, primitives, mode, SMALL, rng(2)) for mode in SIGN_MODES)
        assert natural.models == labels.models and natural.staff_spots == labels.staff_spots, kind
        assert natural.surfaces == labels.surfaces and natural.openings == labels.openings, kind
        if kind != "poster":
            assert (low(natural) == low(labels)) == (kind != "desk"), kind


def test_every_item_of_the_layouts_the_generator_draws_is_constructed(signs):
    """Layouts as the generator draws them, each under its own rigid transform: every item of every one is
    built, stays on its ground, and blocks what the twin blocks; a staffed place keeps its staff spot free
    with its own back room behind it."""
    seen, deep = set(), set()
    assert set(ROWS) <= {*gen._EVAL, *gen._EXAMPLES}
    for layout in layouts():
        made = [construct(item, primitives_of(layout, item), signs, SMALL, rng(index), group=index + 1)
                for index, item in enumerate(layout.items)]
        twin = [[tuple(corner) for corner in obstacle.polygon] for obstacle in layout.scene_gt.obstacles]
        for item, construction in zip(layout.items, made):
            seen.add(item.kind)
            assert_on_footprint(item, construction)
            if item.source == "props":              # a furnishing: its obstacle is the twin's
                assert [tuple(corner) for corner in item.footprint] in twin
                assert_covered(construction, construction.obstacles)
                listed = [[tuple(corner) for corner in obstacle] for obstacle in construction.obstacles]
                if item.kind == "shop":             # but the second stall's: its board and its counter, on its ground
                    assert len(listed) == 2 and len(construction.staff_spots) == 1
                    assert all(distance_to_polygon(corner, item.footprint) <= 1e-9
                               for obstacle in listed for corner in obstacle)
                    (spot,) = construction.staff_spots
                    assert all(distance_to_polygon(spot[:2], obstacle) >= BODY_M for obstacle in listed)
                else:
                    assert listed == [[tuple(corner) for corner in item.footprint]]
        about = [(item, construction) for item, construction in zip(layout.items, made) if item.source == "boxes"]
        if not about:                               # a scene about a person
            continue
        obstacles = [obstacle for _, construction in about for obstacle in construction.obstacles]
        standing = [solid for _, construction in about for solid in solids(construction)]
        for _, construction in about:
            assert_covered(construction, obstacles)
        front, construction = about[0]
        if front.kind in SERVED:                    # with its own back room behind it, if it is deep
            assert_staffed(front, construction, obstacles, standing)
        elif front.kind == "parcel_table" or len(about) == 1:    # the twin's obstacles under it, polygon for polygon
            assert np.allclose(obstacles, layout.footprints)
        else:                                       # an unstaffed deep place: its two items' rectangles, which
            assert [kind for kind in (item.kind for item, _ in about)] == ["kiosk", "back_room"]   # divide the twin's
            assert np.allclose(obstacles, [item.footprint for item, _ in about])
        deep.update(front.kind for _ in about[1:])
    assert seen == set(KINDS) and deep == set(PLACES)
