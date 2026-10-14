"""Dressing a layout (realistic-scenes spec 5.4, 5.6 and 6): what dress adds to a layout of the placeholder
generator and what it leaves as the twin has it.

The rules are tried on the layouts the generator keeps for seed 0, on layouts drawn as it draws them with other
seeds, and on layouts changed by hand; with a small library written here as a manifest, and with the library
that the repository's manifest describes, whatever that holds. Nothing here needs Blender: where a scene has
to be seen, the stub renders it.
"""
import copy
import dataclasses
import json
import math
from functools import lru_cache

import numpy as np
import pytest

import sgi.synth.generate as gen
from sgi.check_scene import check_scene
from sgi.config import load_config
from sgi.freespace import static_grid
from sgi.geometry import Camera, camera_on_robot, wrap_angle
from sgi.polygons import distance_to_polygon, point_in_polygon
from sgi.scene import (GroundTruth, Intrinsics, ObservationModel, Person, PolygonModel, RobotPoseModel, SceneFile,
                       SceneGT, Source, Target, save_scene)
from sgi.synth import recipes
from sgi.synth.realistic.constructions import SIGN_MODES, STAFFED_KINDS, Lettered, construct
from sgi.synth.realistic.dress import VARIANTS, Dressed, Variant, dress
from sgi.synth.realistic.library import load_library, manifest_entries
from sgi.synth.realistic.spec import Frame, ModelRef, PersonRef, SceneSpec, from_json, to_json
from sgi.synth.realistic.stub import StubRenderer
from sgi.synth.render import Box, Sphere

CFG = load_config()
SEED = (0, 1, 7, 0)                                 # as the generator seeds a dressing: seed, split, row, attempt
REACH_M = 0.45                                      # the furthest a pose may get from the body's axis (spec 5.4)
BUILD = (0.92, 1.15)                                # how much wider than the avatar a visitor is made
WIDTH_SCALE = 1.07                                  # of an avatar whose manifest entry gives no max_width_scale
STAFF_POSE = "listen_upright"
STAFF_SHORTER_M = (0.02, 0.05)                      # how far under their natural height the staff stand
VISITORS = {"casual", "business"}                   # an avatar so dressed may be a visitor
NEAR_RED = {"red", "pinkish"}                       # worn by nobody but the person a request asks for in red
NEXT_TO_M = 1.5                                     # two people this near stand next to each other
FLOORS = ("floor_tile", "floor_terrazzo", "floor_laminate")
SMALL_KINDS = ("cup", "book", "box")                # and no bottle: the library's reads as something else on a desk
CLEAR_M = 0.3                                       # a free top has nothing within this far over it
PLATE_M = 0.003                                     # how far a plate stands out of the face it is on
BARE = Variant("bare", "natural", True, False)      # the default without its small things
NOBODY = "Nobody is waiting at the target or standing close to it."     # the placeholder, of a target with nobody at it
NOBODY_BUT_STAFF = ("Nobody is waiting at the target or standing close to it on the visitors' side; a member of "
                    "staff stands behind it.")      # the same where someone on duty stands behind the target
# Layouts as the generator draws them, not all of which it would keep: places with a room behind them, a line
# of six, groups with a second group and bystanders, under other seeds than the set's.
ROWS = (("no_one", 30, "deep_place", 0, False), ("no_one", 0, "place", 1, False), ("no_one", 0, "object", 0, False),
        ("no_one", 0, "person", 0, False), ("waiting_line", 6, None, 0, False), ("waiting_line", 3, None, 0, False),
        ("conversation_group", 4, None, 1, True), ("conversation_group", 2, None, 2, False))
SEEDS = (1, 2, 3, 4)
# One scene of the set for each thing a scene check has to see: a desk, a booth with a bystander, a stall, a
# deep place, the parcel table, someone alone, a kiosk, lines at a stall and at a desk, a group with a second.
CHECKED = ("eval_00_no_one_place", "eval_01_no_one_place", "eval_02_no_one_place", "eval_03_no_one_deep_place",
           "eval_04_no_one_object", "eval_05_no_one_person", "eval_06_waiting_line_1", "eval_08_waiting_line_4",
           "eval_09_waiting_line_6", "eval_11_conversation_group_3")
ABOUT = {"eval_00_no_one_place": ["desk"], "eval_01_no_one_place": ["ticket_counter"],
         "eval_02_no_one_place": ["stall"], "eval_03_no_one_deep_place": ["ticket_counter", "back_room"],
         "eval_04_no_one_object": ["parcel_table"], "eval_05_no_one_person": [], "eval_06_waiting_line_1": ["kiosk"],
         "eval_08_waiting_line_4": ["stall"], "eval_09_waiting_line_6": ["desk"], "eval_11_conversation_group_3": []}


def rng(seed=0):
    return np.random.default_rng(seed)


# --- The small library, written as a manifest so that it is read as the real one is. ---

def avatar(name, sex, height, *tags, widest=None):
    entry = {"id": name, "sex": sex, "natural_height": height, "tags": [sex, *tags]}
    return entry if widest is None else {**entry, "max_width_scale": widest}


def pose(name, reach, *tags, sex=None, sides=None):
    """`sides`: how far the body gets from its axis ahead, behind and to the side, where the manifest says so."""
    entry = {"id": name, "reach": reach, "tags": list(tags)}
    entry = entry if sides is None else {**entry, "measured": dict(zip(("front", "back", "side"), sides))}
    return entry if sex is None else {**entry, "sex": sex}


def model(name, kind, size, brand=False):
    return {"id": name, "kind": kind, "size": list(size), "tags": [], "brand": brand}


AVATARS = [   # ten visitors in plain dress, two in red, one in pink, and staff, four of them in uniform: one of
    #           those for a second stall, so that its vendor is found whoever the visitors are (the manifest has
    #           none in a uniform there, and nine in everyday dress)
    avatar("ada", "female", 1.70, "casual"), avatar("bea", "female", 1.74, "casual", widest=1.20),
    avatar("cyd", "female", 1.66, "business"), avatar("dot", "female", 1.72, "casual"),
    avatar("eve", "female", 1.76, "casual", "staff_stall"), avatar("abe", "male", 1.82, "casual", widest=1.05),
    avatar("bob", "male", 1.86, "casual"), avatar("dan", "male", 1.78, "casual", widest=1.10),
    avatar("cal", "male", 1.80, "business", "staff_desk", "staff_ticket_counter"),
    avatar("eli", "male", 1.84, "business"), avatar("rosa", "female", 1.71, "casual", "red"),
    avatar("rob", "male", 1.83, "casual", "red", widest=1.09), avatar("pia", "female", 1.73, "casual", "pinkish"),
    avatar("chef", "female", 1.72, "uniform", "staff_stall"), avatar("grocer", "male", 1.79, "uniform", "staff_shop"),
    avatar("guard", "male", 1.85, "uniform", "staff_desk", "staff_ticket_counter"),
    avatar("pilot", "female", 1.77, "uniform", "staff_desk", "staff_ticket_counter")]
POSES = [   # "wide" reaches too far on anyone; "watch" and "talk_b" on someone tall for their avatar and broad.
    #         The poses to wait in are measured each way, "watch" with a hand held far out; the others are not.
    pose("stand_a", 0.30, "waiting", sides=(0.20, 0.22, 0.30)),
    pose("stand_b", 0.33, "waiting", sides=(0.22, 0.23, 0.33)),
    pose("stand_c", 0.35, "waiting", sides=(0.26, 0.24, 0.35)), pose("lean", 0.32, "waiting", sides=(0.21, 0.24, 0.32)),
    pose("watch", 0.40, "waiting", sides=(0.40, 0.25, 0.34)), pose("think", 0.36, "waiting", "talking"),
    pose("phone_look", 0.39, "phone", "waiting", sides=(0.35, 0.24, 0.34)),
    pose("phone_ear", 0.35, "phone", "waiting", sides=(0.27, 0.23, 0.34)),
    pose("cup_sip", 0.36, "cup", "talking"), pose("cup_low", 0.34, "cup", "talking"), pose("talk_a", 0.37, "talking"),
    pose("talk_b", 0.40, "talking"), pose("talk_c", 0.33, "talking"), pose("nod", 0.35, "talking"),
    pose(STAFF_POSE, 0.34, "serving", "talking"), pose("serve", 0.35, "serving"),
    pose("wide", 0.60, "waiting", "talking"), pose("curtsy", 0.31, "waiting", sex="female", sides=(0.20, 0.20, 0.31))]
MODELS = [   # one cup shows a brand, and one box is taller than a free top has room for
    model("cup_small", "cup", (0.09, 0.09, 0.11)), model("cup_branded", "cup", (0.08, 0.08, 0.1), brand=True),
    model("bottle_small", "bottle", (0.08, 0.08, 0.24)), model("book_flat", "book", (0.16, 0.22, 0.04)),
    model("box_small", "box", (0.3, 0.24, 0.2)), model("box_tall", "box", (0.2, 0.2, 0.5)),
    model("plant_a", "plant", (0.62, 0.6, 1.3))]


def write_manifest(root, **files):
    (root / "manifest").mkdir(parents=True)
    for name, entries in files.items():
        (root / "manifest" / f"{name}.json").write_text(json.dumps(entries), encoding="utf-8")
    return load_library(root)


@pytest.fixture(scope="module")
def libraries(tmp_path_factory):
    small = write_manifest(tmp_path_factory.mktemp("small"), avatars=AVATARS, poses=POSES, models=MODELS,
                           hdris=[{"id": "town"}, {"id": "sky"}])
    return {"small": small, "manifest": load_library(), "empty": load_library(tmp_path_factory.mktemp("empty"))}


def widest(library):                                # each avatar's max_width_scale, as its manifest gives it
    entries = manifest_entries(library.prepared.parent.parent)["avatars"]
    return {entry["id"]: entry.get("max_width_scale", WIDTH_SCALE) for entry in entries}


def sides_of(library):                              # each measured pose: ahead of the axis, behind it, to the side
    entries = manifest_entries(library.prepared.parent.parent)["poses"]
    return {entry["id"]: tuple(entry["measured"][key] for key in ("front", "back", "side"))
            for entry in entries if "measured" in entry}


def towards(sides, off):
    """How far a body with these sides gets from its axis the way `off` radians from its facing: the radius
    of the ellipse through its side and its front, or its back."""
    along = sides[0] if math.cos(off) >= 0 else sides[1]
    return 1.0 / math.hypot(math.cos(off) / along, math.sin(off) / sides[2])


# --- Layouts. ---

@pytest.fixture(scope="module")
def accepted(tmp_path_factory):
    """The layouts the generator keeps for seed 0, by the placeholder's scene id."""
    return {scene.scene_id: layout
            for _, _, layout, scene in gen.accepted_layouts(tmp_path_factory.mktemp("accepted"), CFG, seed=0)}


@lru_cache(maxsize=None)
def drawn():
    return {f"{'_'.join(str(field) for field in row)}/{seed}": gen._layout(rng(seed), row)
            for row in ROWS for seed in SEEDS}


def to_world(frame, x, y):
    fx, fy, yaw = frame
    c, s = math.cos(yaw), math.sin(yaw)
    return fx + c * x - s * y, fy + s * x + c * y


def with_a_second_place(layout, kind, frame):
    """The layout with one more place of the kind, standing in the frame: its parts after the layout's own, its
    item after the layout's items of `boxes`, its target t1 after the layout's, and its ground the next of the
    layout's footprints and of its obstacles."""
    label, text, color, build, *_ = next(row for row in recipes._PLACES if recipes._PLACE_KINDS[row[3]] == kind)
    x, y, yaw = frame
    parts = recipes._placed(build(text, color), x, y, yaw)
    ground = [to_world(frame, px, py) for px, py in ((-0.8, -0.8), (0.0, -0.8), (0.0, 0.8), (-0.8, 0.8))]
    start = len(layout.boxes)
    item = recipes.Item(kind, x, y, wrap_angle(yaw), ground, {"label": label, "text": text, "color": color}, "boxes",
                        start, start + len(parts))
    target = Target(id="t1", kind="place", label=label, anchor=(x, y, 1.0), footprint=ground, front_yaw=wrap_angle(yaw))
    truth, n = layout.scene_gt, len(layout.footprints)
    about = [one for one in layout.items if one.source == "boxes"]
    obstacles = [*truth.obstacles[:n], PolygonModel(polygon=ground), *truth.obstacles[n:]]
    return dataclasses.replace(
        layout, boxes=(*layout.boxes, *parts), footprints=(*layout.footprints, ground),
        items=(*about, item, *layout.items[len(about):]),
        scene_gt=truth.model_copy(update={"targets": [*truth.targets, target], "obstacles": obstacles}))


@pytest.fixture(scope="module")
def layouts(accepted):
    """Every layout the rules are tried on, by name: the set's, the drawn ones, and a stall with a line that
    has a desk beside it, which no recipe draws: two targets, two staffed places."""
    line = accepted["eval_08_waiting_line_4"]
    two = with_a_second_place(line, "desk", (line.robot.x, line.robot.y, 0.7))
    return {**accepted, **drawn(), "two_places": two}


class Dressings:
    """dress, remembering what it gave: the tests ask for the same dressings over and over.

    Each layout is dressed with a seed of its own, as the generator gives each scene one: with one seed for all
    of them every scene would take the same draws, and show the same floor and the same first person."""

    def __init__(self, layouts, libraries):
        self.layouts, self.libraries, self.made = layouts, libraries, {}
        self.rows = {name: number for number, name in enumerate(layouts)}

    def seed(self, name, attempt=0):
        return (0, 1, self.rows[name], attempt)

    def __call__(self, name, variant="real", library="small", attempt=0) -> Dressed:
        variant = VARIANTS[variant] if isinstance(variant, str) else variant
        key = (name, variant, library, attempt)
        if key not in self.made:
            self.made[key] = dress(self.layouts[name], variant, self.libraries[library], self.seed(name, attempt))
        return self.made[key]


@pytest.fixture(scope="module")
def dressed(layouts, libraries):
    return Dressings(layouts, libraries)


@pytest.fixture(params=("small", "manifest"))
def held(request):                                  # the name of a library that can dress every layout
    return request.param


# --- What the tests read off a layout and a dressing. ---

def primitives(layout, item):
    return getattr(layout, item.source)[item.start:item.stop]


def constructions(layout, variant, library, seed):
    """What the items are built as, by the rule dress is to follow: generator 0 of the seed, one item after
    another. The group is left out: it changes no measure."""
    draws = np.random.default_rng([*seed, 0])
    return [construct(item, primitives(layout, item), variant.signs, library, draws) for item in layout.items]


def corners_of(polygon):
    return [tuple(corner) for corner in polygon]


def middle(polygon):
    xs, ys = zip(*polygon)
    return sum(xs) / len(xs), sum(ys) / len(ys)


def own_items(layout, target):
    """The indices of the items that draw a target: those of `boxes` that stand on a place's footprint, or
    under an object's anchor; none for a person."""
    if target.kind == "person":
        return []
    return [index for index, item in enumerate(layout.items) if item.source == "boxes" and (
        distance_to_polygon(middle(item.footprint), target.footprint) <= 1e-6 if target.footprint is not None
        else distance_to_polygon(target.anchor[:2], item.footprint) <= 1e-6)]


def staffed_places(layout):
    """Each place target whose front is of a staffed kind, with the indices of its items."""
    found = []
    for target in layout.scene_gt.targets:
        own = own_items(layout, target)
        if own and target.footprint is not None and layout.items[own[0]].kind in STAFFED_KINDS:
            found.append((target, own))
    return found


def second_stalls(layout):
    """The indices of the furnishings of a staffed kind: the second stall, which no target is."""
    return [index for index, item in enumerate(layout.items) if item.source == "props" and item.kind in STAFFED_KINDS]


def staff_tag(kind):                                # each staffed kind has a tag of its own, the second stall too
    return f"staff_{kind}"


def doing(layout):
    """What each person of a layout is doing, by id: "line" for the group at a place or an object, "circle" for
    the group about a person who is asked for, "pair" for another group, "asked" for someone asked for who
    stands alone, and "alone"."""
    truth = layout.scene_gt
    kinds = {target.id: target.kind for target in truth.targets}
    at = {}
    for episode in layout.episodes:
        group = episode.ground_truth.group_at_target
        for target_id in episode.ground_truth.target_ids:
            named = group.get(target_id) if isinstance(group, dict) else group
            if named is not None:
                at.setdefault(named, kinds[target_id])
    asked = {target.person_id for target in truth.targets if target.kind == "person"}
    found = {}
    for person in truth.people:
        if person.group in at:
            found[person.id] = "circle" if at[person.group] == "person" else "line"
        elif person.group is not None:
            found[person.id] = "pair"
        else:
            found[person.id] = "asked" if person.id in asked else "alone"
    return found


def placed_reach(ref, library):                     # how far a person's pose gets from their axis, as they stand
    return library.poses[ref.pose].reach * ref.height / library.avatars[ref.avatar].natural_height * ref.build


def visitors(made):                                 # the description's people who are in the layout
    return [ref for ref, person in zip(made.spec.people, made.scene_gt.people) if person.role is None]


def staff(made):
    return [ref for ref, person in zip(made.spec.people, made.scene_gt.people) if person.role is not None]


def box_of(thing):                                  # the box a model fills
    return Box(thing.x, thing.y, thing.yaw, *thing.size, z0=thing.z)


def box_corners(box):
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    return [(box.cx + c * i * box.sx / 2 - s * j * box.sy / 2, box.cy + s * i * box.sx / 2 + c * j * box.sy / 2)
            for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def ground_points(solid):                           # the outline of what a solid stands on
    if isinstance(solid, Box):
        return box_corners(solid)
    reach = solid.radius if isinstance(solid, Sphere) else max(solid.radius, solid.top_radius or 0.0)
    return [(solid.x + reach * math.cos(turn), solid.y + reach * math.sin(turn))
            for turn in np.linspace(0.0, 2 * math.pi, 24, endpoint=False)]


def top_of(solid):
    if isinstance(solid, Sphere):
        return solid.z + solid.radius
    return solid.z0 + (solid.sz if isinstance(solid, Box) else solid.height)


def in_block(point, block, slack):
    x, y, yaw, half_x, half_y, _ = block
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = point[0] - x, point[1] - y
    return abs(c * dx + s * dy) <= half_x + slack and abs(c * dy - s * dx) <= half_y + slack


def on_surface(thing, surface):
    """Whether a model stands on a free top: at its height, with every corner of its box on the rectangle."""
    c, s = math.cos(surface.yaw), math.sin(surface.yaw)
    for x, y in box_corners(box_of(thing)):
        dx, dy = x - surface.x, y - surface.y
        if abs(c * dx + s * dy) > surface.depth / 2 + 1e-9 or abs(c * dy - s * dx) > surface.length / 2 + 1e-9:
            return False
    return thing.z == surface.z


def apart(a, b):
    """Whether two boxes share no ground: some side of one has the other wholly beyond it."""
    first, second = box_corners(a), box_corners(b)
    for corners in (first, second):
        for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
            normal = (y1 - y0, x0 - x1)
            mine = [normal[0] * x + normal[1] * y for x, y in first]
            theirs = [normal[0] * x + normal[1] * y for x, y in second]
            if max(mine) <= min(theirs) or max(theirs) <= min(mine):
                return True
    return False


def is_plain(value):
    """Whether a value of a description is plain data: dataclasses and tuples of Python's own numbers, words,
    None, True and False, at any depth. No list, no dictionary, no number that numpy made."""
    if dataclasses.is_dataclass(value):
        return all(is_plain(getattr(value, field.name)) for field in dataclasses.fields(value))
    if isinstance(value, tuple):
        return all(is_plain(item) for item in value)
    return value is None or type(value) in (int, float, str, bool)


def truth_of(made):                                 # the ground truth as the harness holds a loaded scene's
    gt = made.scene_gt
    return GroundTruth(scene_id="dressed", people=tuple(gt.people),
                       targets={target.id: target for target in gt.targets}, bounds=gt.bounds.polygon,
                       obstacles=tuple(obstacle.polygon for obstacle in gt.obstacles),
                       occupancy=None, occupancy_image=None, episodes={})


def camera_of(layout):
    return Camera(**recipes.CAMERA_LENS, T_world_camera=camera_on_robot(layout.robot, **recipes.CAMERA_MOUNT))


def write_scene(folder, layout, made):
    """The dressed layout as a scene folder: the layout's robot, camera and episodes, the dressing's ground
    truth, the depth the stub renders of its description, and a grey picture."""
    camera = camera_of(layout)
    ((_, depth),) = StubRenderer().render(made.spec, [Frame(camera, beauty=False)])
    observation = ObservationModel(
        rgb="rgb.png", depth="depth.npy", width=camera.width, height=camera.height,
        intrinsics=Intrinsics(fx=camera.fx, fy=camera.fy, cx=camera.cx, cy=camera.cy),
        T_world_camera=camera.T_world_camera.tolist(),
        robot_pose_world=RobotPoseModel(x=layout.robot.x, y=layout.robot.y, yaw=layout.robot.yaw))
    scene = SceneFile(format_version="0.1", scene_id=folder.name, source=Source(simulator="sgi-synth-stub/test"),
                      observation=observation, episodes=list(layout.episodes), ground_truth=made.scene_gt)
    save_scene(folder, scene, np.full((camera.height, camera.width, 3), 128, dtype=np.uint8), depth)


def findings(folder):                               # the checks that are not ok, by number
    return {result.number: (result.level, result.message)
            for result in check_scene(folder, CFG, write_png=False) if result.level != "ok"}


# --- The variants. ---

def test_the_variants_are_three_presets_of_signage_staff_and_small_things():
    assert VARIANTS == {"real": Variant("real", "natural", True, True),
                        "plain": Variant("plain", "labels", False, False),
                        "staffed": Variant("staffed", "labels", True, False)}
    assert list(VARIANTS) == ["real", "plain", "staffed"]
    assert [field.name for field in dataclasses.fields(Variant)] == ["name", "signs", "staff", "small_things"]
    assert [field.name for field in dataclasses.fields(Dressed)] == ["spec", "scene_gt", "target_groups",
                                                                      "target_blocks", "lettering", "episodes"]
    assert {variant.signs for variant in VARIANTS.values()} <= set(SIGN_MODES)
    with pytest.raises(dataclasses.FrozenInstanceError):
        VARIANTS["real"].staff = False


def test_a_signage_that_no_construction_knows_is_refused(layouts, libraries):
    with pytest.raises(ValueError, match="unknown signage 'neon'"):
        dress(layouts["eval_02_no_one_place"], Variant("odd", "neon", True, True), libraries["small"], SEED)


# --- The twin: what dressing leaves alone. ---

@pytest.mark.parametrize("variant", VARIANTS)
def test_the_people_targets_bounds_and_robot_are_the_layouts(layouts, libraries, held, variant):
    """Staff apart. The layout itself is not changed either: its robot, its episodes with their ids, requests,
    contexts and request types, its items and their options are as they were."""
    for name, layout in layouts.items():
        before = copy.deepcopy(layout)
        made = dress(layout, VARIANTS[variant], libraries[held], SEED)
        assert layout == before, name
        assert isinstance(made, Dressed) and isinstance(made.spec, SceneSpec) and isinstance(made.scene_gt, SceneGT)
        truth, twin = made.scene_gt, layout.scene_gt
        assert [person for person in truth.people if person.role is None] == twin.people, name
        assert truth.people[:len(twin.people)] == twin.people, name      # in the twin's order, the staff after them
        assert truth.targets == twin.targets and truth.bounds == twin.bounds and truth.occupancy == twin.occupancy
        # The description stands everyone where the ground truth has them, and people[i] draws people[i].
        assert [(ref.person_id, ref.x, ref.y, ref.yaw) for ref in made.spec.people] == [
            (person.id, person.x, person.y, person.yaw) for person in truth.people], name
        # A person who is asked for is still where the anchor is.
        for target in truth.targets:
            if target.kind == "person":
                (ref,) = [ref for ref in made.spec.people if ref.person_id == target.person_id]
                assert (ref.x, ref.y) == tuple(target.anchor[:2]), name


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_episodes_are_the_twins_but_for_what_staff_make_untrue_of_the_people(layouts, dressed, held, variant):
    """An episode with nobody at its target says so of the people. Where a member of staff stands behind the
    target that would be false, and it says instead that nobody stands on the visitors' side and that a member
    of staff stands behind it. Nothing else of any episode differs from the layout's; and nothing at all in a
    variant without staff, at a target without staff (a kiosk, the parcel table, a person), in another
    situation, or for the vendor of a second stall, which is no target."""
    changed, kept = 0, 0
    for name, layout in layouts.items():
        made = dressed(name, variant, held)
        served = {target.id for target, _ in staffed_places(layout)} if VARIANTS[variant].staff else set()
        assert isinstance(made.episodes, tuple) and len(made.episodes) == len(layout.episodes), name
        for ours, theirs in zip(made.episodes, layout.episodes):
            mine, twins = ours.ground_truth, theirs.ground_truth
            if not (twins.situation == "no_one" and served & set(twins.target_ids)):
                assert ours == theirs, name
                kept += 1
                continue
            changed += 1
            assert twins.rationale.people == NOBODY and mine.rationale.people == NOBODY_BUT_STAFF, name
            # But for that text, the episode is the twin's: its id, request, context, request type and the rest.
            as_twin = mine.model_copy(update={"rationale": mine.rationale.model_copy(update={"people": NOBODY})})
            assert ours.model_copy(update={"ground_truth": as_twin}) == theirs, name
    assert kept >= 40 and (changed >= 12 if VARIANTS[variant].staff else changed == 0)
    assert second_stalls(layouts["eval_05_no_one_person"])          # a vendor in the scene, and no text changed
    assert dressed("eval_05_no_one_person", variant, held).episodes == layouts["eval_05_no_one_person"].episodes


def test_a_dressing_is_written_with_the_models_of_the_scene_format(layouts, dressed):
    """So that it loads: the ground truth is a SceneGT whose people, targets and polygons are the scene
    format's own models, and it goes through JSON and back unchanged."""
    for name in layouts:
        truth = dressed(name).scene_gt
        assert all(isinstance(person, Person) for person in truth.people)
        assert all(isinstance(obstacle, PolygonModel) for obstacle in truth.obstacles)
        assert SceneGT.model_validate(json.loads(json.dumps(truth.model_dump(mode="json")))) == truth, name


# --- Staff. ---

@pytest.mark.parametrize("variant", ("real", "staffed"))
def test_staff_stand_at_each_staffed_place_with_the_role_and_no_group(layouts, libraries, dressed, held, variant):
    """One for each staff spot of the constructions, in their order, as s0, s1, ...: on the spot, facing out of
    the place's front, cast from the avatars tagged for that kind of place, upright and a little under their
    natural height, in no object group."""
    library, seen = libraries[held], set()
    for name, layout in layouts.items():
        made = dressed(name, variant, held)
        built = constructions(layout, VARIANTS[variant], library, dressed.seed(name))
        spots = [(item, spot) for item, one in zip(layout.items, built) for spot in one.staff_spots]
        assert len(spots) == sum(item.kind in STAFFED_KINDS for item in layout.items), name
        on_duty = [person for person in made.scene_gt.people if person.role is not None]
        assert [person.id for person in on_duty] == [f"s{number}" for number in range(len(spots))], name
        for person, ref, (item, (x, y, yaw)) in zip(on_duty, staff(made), spots):
            assert person == Person(id=person.id, x=x, y=y, yaw=yaw, group=None, role="staff"), name
            assert yaw == item.yaw                  # the place's front
            on = [target for target in layout.scene_gt.targets if target.footprint is not None
                  and distance_to_polygon((x, y), target.footprint) == 0.0]
            if item.source == "boxes":              # behind the front of the target they serve at
                (target,) = on
                assert target.front_yaw == pytest.approx(yaw)
            else:                                   # the second stall's vendor: on its ground, which is no target's
                assert item.kind == "shop" and not on and point_in_polygon((x, y), item.footprint), name
            chosen = library.avatars[ref.avatar]
            assert staff_tag(item.kind) in chosen.tags and not chosen.tags & NEAR_RED, name
            assert (ref.pose, ref.build, ref.prop, ref.group) == (STAFF_POSE, 1.0, None, 0), name
            assert STAFF_SHORTER_M[0] <= chosen.natural_height - ref.height <= STAFF_SHORTER_M[1], name
            seen.add(item.kind)
    assert seen == set(STAFFED_KINDS)
    # A stall with a desk beside it and a second stall on the wall: the targets' staff first, then the vendor.
    two = dressed("two_places", variant, held)
    assert [person.id for person in two.scene_gt.people if person.role] == ["s0", "s1", "s2"]
    assert [item.kind for item in layouts["two_places"].items if item.kind in STAFFED_KINDS] == ["stall", "desk",
                                                                                                 "shop"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_vendor_stands_at_the_second_stall_where_the_variant_has_staff(layouts, libraries, dressed, held, variant):
    """So that the place a request names is not the only staffed place of its scene. With staff: one person
    with the role and no group on the second stall's spot, facing into the room, with the next free staff id
    after the staff of the targets, cast from the avatars tagged for a second stall, who are tagged for no
    other place, and wearing nobody else's avatar. Without staff: nobody. The target, the visitors and the
    episodes are as they were."""
    library, vendors, both = libraries[held], 0, 0
    for name, layout in layouts.items():
        made = dressed(name, variant, held)
        built = constructions(layout, VARIANTS[variant], library, dressed.seed(name))
        shops = second_stalls(layout)
        assert len(shops) <= 1 and all(layout.items[index].kind == "shop" for index in shops), name
        assert all(len(built[index].staff_spots) == 1 for index in shops), name
        on_duty = [(person, ref) for person, ref in zip(made.scene_gt.people, made.spec.people) if person.role]
        at_shop = [(person, ref) for person, ref in on_duty
                   if any(point_in_polygon((person.x, person.y), layout.items[index].footprint) for index in shops)]
        if not VARIANTS[variant].staff:
            assert not on_duty, name
            continue
        assert len(at_shop) == len(shops), name
        for index, (person, ref) in zip(shops, at_shop):
            x, y, yaw = built[index].staff_spots[0]
            others = len(on_duty) - len(at_shop)    # the staff of the targets, who come first
            assert person == Person(id=f"s{others}", x=x, y=y, yaw=yaw, group=None, role="staff"), name
            assert (person, ref) == on_duty[-1] and ref.person_id == person.id, name
            worn = library.avatars[ref.avatar].tags
            assert {tag for tag in worn if tag.startswith("staff_")} == {"staff_shop"}, name
            assert ref.pose == STAFF_POSE and [one.avatar for one in made.spec.people].count(ref.avatar) == 1, name
            vendors += 1
            both += any(item.kind == "stall" for item in layout.items)
        assert made.scene_gt.targets == layout.scene_gt.targets, name
        assert [person for person in made.scene_gt.people if person.role is None] == layout.scene_gt.people, name
    if VARIANTS[variant].staff:
        assert vendors >= 20 and both >= 3          # every scene of the set has a second stall; some a stall too


def test_plain_has_no_staff_and_the_same_cast_and_poses_as_real(layouts, dressed, held):
    """The three variants are dressed from the same draws: the people of the layout are the same person for
    person, the staff of the two variants that have them are the same, and so are the models of the
    constructions, the floor and the seed. Leaving the staff or the small things out changes nothing else."""
    for name, layout in layouts.items():
        real, plain, staffed, bare = (dressed(name, variant, held) for variant in ("real", "plain", "staffed", BARE))
        n = len(layout.scene_gt.people)
        assert plain.scene_gt.people == layout.scene_gt.people, name
        assert all(person.role is None for person in plain.scene_gt.people)
        assert len(plain.spec.people) == n
        assert plain.spec.people == real.spec.people[:n] == staffed.spec.people[:n], name
        assert staffed.spec.people == real.spec.people == bare.spec.people, name
        assert staffed.scene_gt.people == real.scene_gt.people, name
        assert plain.spec.parts == staffed.spec.parts and real.spec.parts == bare.spec.parts, name
        assert plain.spec.models == staffed.spec.models == bare.spec.models, name
        assert real.spec.models[:len(bare.spec.models)] == bare.spec.models, name
        assert plain.spec.room == staffed.spec.room == real.spec.room, name
        assert plain.spec.seed == staffed.spec.seed == real.spec.seed and plain.spec.hdri == real.spec.hdri
        assert plain.scene_gt.obstacles == staffed.scene_gt.obstacles, name    # the same constructions, staffed or not
        assert real.target_blocks.keys() == plain.target_blocks.keys() == staffed.target_groups.keys()


@pytest.mark.parametrize("variant", ("real", "staffed"))
def test_staff_stand_on_free_cells_of_the_static_grid(layouts, dressed, held, variant):
    """On the map the scene check reads, and with the twin's obstacles they would not: the place's whole
    footprint is one obstacle there."""
    for name, layout in layouts.items():
        made = dressed(name, variant, held)
        on_duty = [person for person in made.scene_gt.people if person.role is not None]
        if not on_duty:
            continue
        grid = static_grid(truth_of(made), CFG.thresholds.grid_resolution_m)
        assert all(grid.is_free(person.x, person.y) for person in on_duty), name
        as_twin = dataclasses.replace(truth_of(made), obstacles=tuple(
            obstacle.polygon for obstacle in layout.scene_gt.obstacles))
        blocked = static_grid(as_twin, CFG.thresholds.grid_resolution_m)
        assert not any(blocked.is_free(person.x, person.y) for person in on_duty), name
        # Everyone else is on a free cell as before.
        assert all(grid.is_free(person.x, person.y) for person in layout.scene_gt.people), name


# --- Obstacles. ---

@pytest.mark.parametrize("variant", VARIANTS)
def test_obstacles_are_the_twins_except_under_a_staffed_place_and_under_the_second_stall(layouts, libraries, dressed,
                                                                                         held, variant):
    """The layout's own list, polygon for polygon and in its order. Only the one rectangle under a staffed
    target place gives way, where it stands in the list, to what the constructions of the place's items list,
    and the one rectangle of a second stall to what its construction lists; and that in every variant, staffed
    or not, since all are built alike."""
    library, staffed_seen, deep_seen, stalls_seen = libraries[held], 0, 0, 0
    for name, layout in layouts.items():
        made = dressed(name, variant, held)
        built = constructions(layout, VARIANTS[variant], library, dressed.seed(name))
        twin, got = layout.scene_gt.obstacles, made.scene_gt.obstacles
        instead = {}                                # the place in the twin's list: what stands there now
        for target, own in staffed_places(layout):
            (at,) = [index for index, obstacle in enumerate(twin)
                     if corners_of(obstacle.polygon) == corners_of(target.footprint)]
            assert at < len(layout.footprints), name                # one of the footprints, which come first
            listed = [corners_of(polygon) for index in own for polygon in built[index].obstacles]
            assert len(listed) > 1
            # What replaces it lies on the footprint it replaces, which stays the target's.
            assert all(distance_to_polygon(corner, target.footprint) <= 1e-9
                       for polygon in listed for corner in polygon), name
            instead[at] = [PolygonModel(polygon=polygon) for polygon in listed]
            staffed_seen, deep_seen = staffed_seen + 1, deep_seen + (len(own) > 1)
        for index in second_stalls(layout):         # its item's rectangle, among those of the furnishings
            item = layout.items[index]
            (at,) = [place for place, obstacle in enumerate(twin)
                     if corners_of(obstacle.polygon) == corners_of(item.footprint)]
            assert at >= len(layout.footprints) and at not in instead, name
            listed = [corners_of(polygon) for polygon in built[index].obstacles]
            assert len(listed) == 2                 # its back board and its counter: the strip between them is floor
            assert all(distance_to_polygon(corner, item.footprint) <= 1e-9
                       for polygon in listed for corner in polygon), name
            instead[at] = [PolygonModel(polygon=polygon) for polygon in listed]
            stalls_seen += 1
        expected = [obstacle for index, kept in enumerate(twin) for obstacle in instead.get(index, [kept])]
        assert got == expected, name
        assert [target.footprint for target in made.scene_gt.targets] == [
            target.footprint for target in layout.scene_gt.targets]
        if not instead:
            assert got == twin, name
        # Where an item is one obstacle of its twin, its construction says the same; and so does the parcel
        # table for its two. Asserted, not trusted: the twin's polygons are the ones kept.
        for item, one in zip(layout.items, built):
            if item.source == "props" and item.kind not in STAFFED_KINDS:
                assert [corners_of(polygon) for polygon in one.obstacles] == [corners_of(item.footprint)], name
                assert PolygonModel(polygon=item.footprint) in got, name
            elif item.source == "props":            # the second stall's one rectangle is gone
                assert PolygonModel(polygon=item.footprint) not in got, name
            elif item.kind == "parcel_table":
                assert np.allclose(one.obstacles, layout.footprints) and len(layout.footprints) == 2, name
                assert got[:2] == twin[:2], name
    assert staffed_seen >= 12 and deep_seen >= 2 and stalls_seen >= 20


def test_the_twin_of_a_place_without_staff_keeps_its_one_obstacle(layouts, dressed):
    """A kiosk, alone or with a room behind it: the constructions divide its ground between them, and the
    ground truth keeps the twin's one rectangle. (The furnishings' obstacles follow it in the list; among them
    only a second stall's differs from the twin's.)"""
    kiosks = [name for name, layout in layouts.items() if any(item.kind == "kiosk" for item in layout.items)]
    assert kiosks and any(len(own_items(layouts[name], layouts[name].scene_gt.targets[0])) == 2 for name in kiosks)
    for name in kiosks:
        twin, under = layouts[name].scene_gt.obstacles, len(layouts[name].footprints)
        assert under == 1 and twin[0] == PolygonModel(polygon=layouts[name].scene_gt.targets[0].footprint)
        for variant in VARIANTS:
            got = dressed(name, variant).scene_gt.obstacles
            assert got[:under] == twin[:under], name
            assert (got == twin) == (not second_stalls(layouts[name])), name


# --- The cast. ---

def test_nobody_is_cast_twice_and_heights_are_the_figures(layouts, libraries, dressed, held):
    """One avatar for each person, none twice in a scene, the staff counted; a visitor is dressed as one, and
    never in a uniform; everyone is as tall as the placeholder's figure of them."""
    library = libraries[held]
    for name, layout in layouts.items():
        made = dressed(name, "real", held)
        worn = [ref.avatar for ref in made.spec.people]
        assert len(set(worn)) == len(worn), name
        assert all(isinstance(ref, PersonRef) and ref.avatar in library.avatars and ref.pose in library.poses
                   for ref in made.spec.people)
        assert len(visitors(made)) == len(layout.figures)
        for ref, figure in zip(visitors(made), layout.figures):
            assert ref.height == figure.height, name
            assert library.avatars[ref.avatar].tags & VISITORS and "uniform" not in library.avatars[ref.avatar].tags


def test_only_the_person_in_red_wears_red(layouts, libraries, dressed, held):
    """The person a request asks for in red is cast from the avatars tagged red, and nobody else in any scene,
    staff included, wears red or pink: not in a scene with such a person, and not in one without."""
    library, with_red, without = libraries[held], 0, 0
    for name, layout in layouts.items():
        in_red = {target.person_id for target in layout.scene_gt.targets
                  if target.kind == "person" and "red" in target.label.split()}
        with_red, without = with_red + bool(in_red), without + (not in_red)
        for variant in VARIANTS:
            for ref in dressed(name, variant, held).spec.people:
                tags = library.avatars[ref.avatar].tags
                assert ("red" in tags) if ref.person_id in in_red else not tags & NEAR_RED, (name, ref)
    assert with_red >= 10 and without >= 10


def test_the_colour_is_read_from_the_label_of_the_target(layouts, libraries):
    """Not from a constant, and not from the placeholder's shirt: another label that says red is cast in red,
    and one that says no colour is cast as anyone else is."""
    library, layout = libraries["small"], layouts["eval_05_no_one_person"]

    def relabelled(label):
        (target,) = layout.scene_gt.targets
        truth = layout.scene_gt.model_copy(update={"targets": [target.model_copy(update={"label": label})]})
        ref, vendor = dress(dataclasses.replace(layout, scene_gt=truth), VARIANTS["real"], library, SEED).spec.people
        assert (ref.person_id, vendor.person_id) == ("p0", "s0")    # the person asked for, and the second stall's
        return library.avatars[ref.avatar].tags

    assert "red" in relabelled("person in red") and "red" in relabelled("the woman in the red coat")
    assert not relabelled("person with a suitcase") & NEAR_RED
    assert not relabelled("the bored person") & NEAR_RED          # a word that holds "red" does not say it


def test_no_pose_reaches_further_than_45_cm(layouts, libraries, dressed, held):
    """As the person stands: the pose's reach at the size they are placed at, widened by their build. The
    small library has a pose that never keeps within that, and two that do only on someone slight."""
    library, widest_seen, used = libraries[held], 0.0, set()
    for name in layouts:
        for attempt in (0, 1):
            for ref in dressed(name, "real", held, attempt).spec.people:
                assert placed_reach(ref, library) <= REACH_M + 1e-12, (name, ref)
                widest_seen = max(widest_seen, placed_reach(ref, library))
                used.add(ref.pose)
    assert widest_seen > 0.4                        # the limit is one that the poses come near
    if held == "small":
        assert "wide" not in used and {"watch", "talk_b"} <= used


def test_a_build_is_drawn_between_92_and_115_percent_and_limited_to_what_the_avatar_shows_at(layouts, libraries,
                                                                                            dressed, held):
    """A person's width is their height over the avatar's, times their build; the manifest says how wide each
    avatar may get, and one that does not say takes 1.07."""
    library = libraries[held]
    most, limited, free = widest(library), 0, 0
    for name in layouts:
        for attempt in (0, 1):
            for ref in visitors(dressed(name, "real", held, attempt)):
                natural = library.avatars[ref.avatar].natural_height
                limit = min(BUILD[1], most.get(ref.avatar, WIDTH_SCALE) * natural / ref.height)
                assert type(ref.build) is float and ref.build <= limit + 1e-12, (name, ref)
                assert ref.build >= BUILD[0] or ref.build == pytest.approx(limit), (name, ref)
                limited, free = limited + (ref.build == pytest.approx(limit)), free + (ref.build < limit - 1e-9)
    assert limited >= 10 and free >= 10
    if held == "small":                             # an avatar without the key, and avatars with it
        assert "ada" not in {entry["id"] for entry in AVATARS if "max_width_scale" in entry}
        assert most["abe"] == 1.05 and most["ada"] == WIDTH_SCALE


def test_people_are_posed_by_what_they_are_doing(layouts, libraries, dressed, held):
    """In a line at a place: waiting, perhaps with a phone. In the group about a person who is asked for:
    talking, perhaps with a cup, and nobody with a phone: whoever is on the phone is not in the conversation,
    and four people talking would read as three and a bystander. In another group: any of those. Alone:
    waiting, a phone or a cup, and no gesture made to nobody. Asked for and alone: waiting."""
    library, seen = libraries[held], set()
    suits = {"line": {"waiting", "phone"}, "circle": {"talking", "cup"}, "asked": {"waiting"},
             "pair": {"waiting", "talking", "phone", "cup"}, "alone": {"waiting", "phone", "cup"}}
    for name, layout in layouts.items():
        does = doing(layout)
        for attempt in (0, 1):
            made = dressed(name, "real", held, attempt)
            for ref in visitors(made):
                tags = library.poses[ref.pose].tags
                assert tags & suits[does[ref.person_id]], (name, ref, does[ref.person_id])
                assert ref.prop == ("phone" if "phone" in tags else "cup" if "cup" in tags else None), (name, ref)
                assert library.poses[ref.pose].sex in (None, library.avatars[ref.avatar].sex), (name, ref)
                seen.add((does[ref.person_id], ref.prop))
            talking = [ref for ref in visitors(made) if does[ref.person_id] == "circle"]
            assert not any(ref.prop == "phone" or "phone" in library.poses[ref.pose].tags for ref in talking), name
            assert all("talking" in library.poses[ref.pose].tags for ref in talking), name
            # No two of a conversation in one pose: a circle of four is wider than people next to each other stand,
            # and the rule that nobody takes a pose someone in the scene has, while others are free, keeps them apart.
            poses = [ref.pose for ref in talking]
            assert len(set(poses)) == len(poses), (name, poses)
    assert {kind for kind, _ in seen} == set(suits)
    assert {("line", "phone"), ("line", None), ("circle", "cup"), ("circle", None), ("pair", None)} <= seen
    assert ("circle", "phone") not in seen
    # A pose tagged for talking and for a phone both is still not one for a conversation.
    if held == "small":
        group = "eval_12_conversation_group_4"
        both = dataclasses.replace(library, poses={name: dataclasses.replace(
            one, tags=one.tags | {"talking"}) if "phone" in one.tags else one for name, one in library.poses.items()})
        for attempt in range(6):
            made = dress(layouts[group], VARIANTS["real"], both, (0, 0, 12, attempt))
            does = doing(layouts[group])
            assert not any(ref.prop == "phone" for ref in visitors(made) if does[ref.person_id] == "circle"), attempt


def test_a_conversation_is_not_made_of_listeners_alone(layouts, libraries, dressed, held):
    """In the group about a person who is asked for, someone stands in a pose that is not for listening only.
    The library names a pose in which one only listens, arms down and no gesture, `listen_...`; with both of a
    pair in such poses nobody is talking, and the pair does not read as a conversation."""
    def listens(ref):
        return ref.pose.startswith("listen_")

    library, circles = libraries[held], 0
    for name, layout in layouts.items():
        does = doing(layout)
        for attempt in (0, 1):
            talking = [ref for ref in visitors(dressed(name, "real", held, attempt)) if does[ref.person_id] == "circle"]
            circles += bool(talking)
            assert not talking or not all(listens(ref) for ref in talking), (name, attempt, talking)
    assert circles >= 20
    # Where the rule decides: a library with two poses to listen in and one gesture, and a conversation of two.
    # Whenever the first of them listens, the second is refused the other listening pose, which nothing else
    # would keep from them: it is another pose, and no take of the first.
    small, pair = libraries["small"], "eval_10_conversation_group_2"
    few = dataclasses.replace(small, poses={
        **{name: one for name, one in small.poses.items() if "talking" not in one.tags},
        STAFF_POSE: small.poses[STAFF_POSE], "listen_nod": dataclasses.replace(small.poses["nod"], id="listen_nod"),
        "talk_a": small.poses["talk_a"]})
    does, first_listened = doing(layouts[pair]), 0
    for attempt in range(24):
        made = dress(layouts[pair], VARIANTS["plain"], few, (0, 0, 10, attempt))
        talking = [ref for ref in visitors(made) if does[ref.person_id] == "circle"]
        assert len(talking) == 2 and {ref.pose for ref in talking} <= {STAFF_POSE, "listen_nod", "talk_a"}
        assert not all(listens(ref) for ref in talking), (attempt, talking)
        first_listened += listens(talking[0])
    assert first_listened >= 8
    # With nothing but listening poses to talk in, they are taken all the same.
    only = dataclasses.replace(few, poses={name: one for name, one in few.poses.items() if name != "talk_a"})
    talking = [ref for ref in visitors(dress(layouts[pair], VARIANTS["plain"], only, (0, 0, 10, 0)))
               if does[ref.person_id] == "circle"]
    assert len(talking) == 2 and all(listens(ref) for ref in talking)


def test_people_next_to_each_other_do_not_share_a_pose_while_others_are_free(layouts, libraries, dressed, held):
    """Nor two takes of one pose, which the libraries name alike but for a last letter: two people side by side
    with a hand at the chin look as alike as two in the very same pose."""
    def take_of(name):
        return name[:-2] if name[-2:-1] == "_" and name[-1].islower() else name

    library = libraries[held]
    assert take_of("thoughtful_b") == take_of("thoughtful_a") == "thoughtful" and take_of("phone_ear") == "phone_ear"
    for name in layouts:
        for attempt in (0, 1):
            people = visitors(dressed(name, "real", held, attempt))
            for index, one in enumerate(people):
                for other in people[index + 1:]:
                    if math.hypot(one.x - other.x, one.y - other.y) <= NEXT_TO_M:
                        assert take_of(one.pose) != take_of(other.pose), (name, one, other)
    # With two poses to wait in, a line of six takes them in turn, whether they are two takes of one pose or
    # two poses; with one, everyone takes it.
    line = "waiting_line_6_None_0_False/1"
    waiting = [item for item in library.poses.items() if "waiting" in item[1].tags]
    alike = [item for item in waiting if take_of(item[0]) == take_of(waiting[0][0])][:2]
    unlike = [waiting[0], next(item for item in waiting if take_of(item[0]) != take_of(waiting[0][0]))]
    for few in (alike, unlike, waiting[:1]):
        poses = [ref.pose for ref in dress(layouts[line], VARIANTS["plain"], dataclasses.replace(
            library, poses=dict(few)), SEED).spec.people]
        assert len(set(poses)) == len(few) and len(poses) == 6
        assert all(a != b for a, b in zip(poses, poses[1:])) == (len(few) == 2)
    # With three, nobody has the pose of either of the two before them, who both stand next to them. The three
    # are poses that anyone may take and that hold least out ahead, so that each has room behind anyone.
    ahead, three = sides_of(library), {}
    for name, one in sorted(waiting, key=lambda item: ahead.get(item[0], (0.0,))[0]):
        if one.sex is None and one.reach <= 0.37:
            three.setdefault(take_of(name), (name, one))
    poses = [ref.pose for ref in dress(layouts[line], VARIANTS["plain"], dataclasses.replace(
        library, poses=dict(list(three.values())[:3])), SEED).spec.people]
    assert len(set(poses)) == 3 and all(len({*poses[k:k + 3]}) == 3 for k in range(4)), poses


def test_nobody_reaches_into_the_person_next_to_them(layouts, libraries, dressed, held):
    """As far as the manifest measured the poses, ahead of the body, behind it and to its side: a hand held out
    is not in the back of whoever stands before them in a line. A pose with a hand far out is still taken where
    there is room for it, and where the layout leaves no pose room, one is taken all the same."""
    library = libraries[held]
    sides, pairs, far_out = sides_of(library), 0, set()

    def width(ref):
        return ref.height / library.avatars[ref.avatar].natural_height * ref.build

    def touching(people):
        found = []
        for index, one in enumerate(people):
            for other in people[index + 1:]:
                gap = math.hypot(other.x - one.x, other.y - one.y)
                if gap > NEXT_TO_M or one.pose not in sides or other.pose not in sides:
                    continue
                way = math.atan2(other.y - one.y, other.x - one.x)
                reach = (towards(sides[one.pose], way - one.yaw) * width(one)
                         + towards(sides[other.pose], way + math.pi - other.yaw) * width(other))
                found.append(reach > gap + 1e-9)
        return found

    for name in layouts:
        for attempt in (0, 1):
            people = visitors(dressed(name, "real", held, attempt))
            assert not any(touching(people)), (name, attempt)
            pairs += len(touching(people))
            far_out |= {ref.pose for ref in people if ref.pose in sides and sides[ref.pose][0] >= 0.34}
    assert pairs >= 100 and len(far_out) >= 2
    # With nothing to wait in but a pose that holds a hand far out, the people in a line take it and touch.
    reaching = max((name for name in sides if "waiting" in library.poses[name].tags), key=lambda name: sides[name][0])
    only = dataclasses.replace(library, poses={reaching: library.poses[reaching]})
    line = visitors(dress(layouts["waiting_line_6_None_0_False/1"], VARIANTS["plain"], only, SEED))
    assert {ref.pose for ref in line} == {reaching} and any(touching(line))


def test_casting_says_what_the_library_lacks(layouts, libraries):
    """Too few avatars for the visitors, none in red for the person asked for, none tagged for a kind of place,
    no pose to stand in: each a ValueError in plain words, and none of them where the variant does not ask."""
    small, real, plain = libraries["small"], VARIANTS["real"], VARIANTS["plain"]
    line, group, stall = "eval_08_waiting_line_4", "eval_11_conversation_group_3", "eval_02_no_one_place"

    def having(keep):                               # the small library with some of its avatars
        return dataclasses.replace(small, avatars={name: one for name, one in small.avatars.items() if keep(one)})

    with pytest.raises(ValueError, match="the scene has 4 visitors and the library has 2 avatars for them"):
        dress(layouts[line], plain, having(lambda one: one.id in ("ada", "bob", "rosa", "pia", "chef")), SEED)
    with pytest.raises(ValueError, match="the scene has 4 visitors and the library has 0 avatars for them"):
        dress(layouts[line], plain, libraries["empty"], SEED)
    # The group has one person in red and four others: without the red avatars, the others are served.
    with pytest.raises(ValueError, match="no avatar tagged red is left for the target"):
        dress(layouts[group], plain, having(lambda one: "red" not in one.tags), SEED)
    with pytest.raises(ValueError, match="no avatar tagged staff_stall"):
        dress(layouts[stall], real, having(lambda one: "staff_stall" not in one.tags), SEED)
    assert dress(layouts[stall], plain, having(lambda one: "staff_stall" not in one.tags), SEED).spec.people == ()
    # The visitors may wear every avatar that is tagged for the place: the staff then have none left. (The
    # uniform of a second stall is kept: the scene has one, whose vendor is then always found.)
    crowded = having(lambda one: "uniform" not in one.tags or "staff_shop" in one.tags)
    with pytest.raises(ValueError, match="no avatar tagged staff_desk is left"):
        for attempt in range(40):                   # until the one avatar tagged for a desk is in the line
            dress(layouts["eval_09_waiting_line_6"], real, crowded, (0, 0, 9, attempt))
    with pytest.raises(ValueError, match="no pose tagged waiting or phone"):
        dress(layouts[line], plain, dataclasses.replace(small, poses={
            name: one for name, one in small.poses.items() if "waiting" not in one.tags}), SEED)
    with pytest.raises(ValueError, match=f"no pose {STAFF_POSE}"):
        dress(layouts[stall], real, dataclasses.replace(small, poses={
            name: one for name, one in small.poses.items() if name != STAFF_POSE}), SEED)
    # A scene with nobody in it, and no staff asked for, needs no avatar at all.
    bare = dress(layouts[stall], plain, libraries["empty"], SEED)
    assert bare.spec.people == () and bare.spec.models == () and bare.spec.hdri is None and bare.spec.parts


# --- Small things. ---

def test_small_things_change_no_obstacle_and_stand_on_surfaces(layouts, libraries, dressed, held):
    """What the default adds to the constructions' models: a cup, a book or a box of the library, at
    most two on a free top and two of a kind on one place, each wholly on its top and standing on it, turned a
    little, clear of the other, low enough for the room the top has, in the group of what it stands on. The map
    is the same with or without."""
    library, counts = libraries[held], []
    for name, layout in layouts.items():
        real, bare = dressed(name, "real", held), dressed(name, BARE, held)
        assert real.scene_gt == bare.scene_gt, name
        things = real.spec.models[len(bare.spec.models):]
        built = constructions(layout, VARIANTS["real"], library, dressed.seed(name))
        tops = [(surface, index) for index, one in enumerate(built) for surface in one.surfaces]
        on = {}
        for thing in things:
            asset = library.models[thing.asset]
            assert isinstance(thing, ModelRef) and asset.kind in SMALL_KINDS and not asset.brand, (name, thing)
            assert len(set(thing.scale)) == 1 and 0.85 - 1e-9 <= thing.scale[0] <= 1.15 + 1e-9, (name, thing)
            assert thing.size == pytest.approx(tuple(thing.scale[0] * side for side in asset.size))
            assert thing.size[2] <= CLEAR_M + 1e-9, (name, thing)
            (where,) = [number for number, (surface, _) in enumerate(tops) if on_surface(thing, surface)]
            surface, index = tops[where]
            assert abs(wrap_angle(thing.yaw - surface.yaw)) <= 0.5, (name, thing)
            # In the group of the construction it stands on: that of the target the item draws, or none.
            groups = {part.group for part in real.spec.parts
                      if part in [dataclasses.replace(own, group=part.group) for own in built[index].parts]}
            assert groups == {thing.group}, (name, thing)
            # Over ground that is blocked already.
            assert all(min(distance_to_polygon(corner, obstacle.polygon) for obstacle in real.scene_gt.obstacles)
                       <= 1e-9 for corner in box_corners(box_of(thing))), (name, thing)
            on.setdefault(where, []).append(thing)
        of_a_kind = {}                              # (the construction, the kind): how many stand on all its tops
        for where, together in on.items():
            assert len(together) <= 2, name
            assert len(together) < 2 or apart(box_of(together[0]), box_of(together[1])), (name, together)
            for thing in together:
                key = (tops[where][1], library.models[thing.asset].kind)
                of_a_kind[key] = of_a_kind.get(key, 0) + 1
        # No more than two of one kind on one place: a desk with four cups and nothing else is a shelf of cups.
        assert max(of_a_kind.values(), default=0) <= 2, (name, of_a_kind)
        counts.append(len(things))
        for variant in ("plain", "staffed"):
            assert len(dressed(name, variant, held).spec.models) == len(bare.spec.models), name
    assert sum(count > 0 for count in counts) >= 10 and max(counts) >= 2
    used = {ref.asset for name in layouts for ref in dressed(name, "real", held).spec.models}
    assert not any(library.models[asset].brand for asset in used)
    if held == "small":
        assert {"cup_small", "book_flat"} <= used and not used & {"cup_branded", "box_tall", "bottle_small"}


def test_a_library_without_small_things_dresses_a_scene_without_them(layouts, libraries):
    name = "eval_02_no_one_place"
    bare_shelves = dataclasses.replace(libraries["small"], models={})
    made = dress(layouts[name], VARIANTS["real"], bare_shelves, SEED)
    assert made.spec.models == () and len(made.spec.people) == 2       # behind the stall, and at the second stall


# --- The room. ---

@pytest.mark.parametrize("variant", VARIANTS)
def test_the_room_is_the_layouts_with_a_floor_and_a_view_of_the_librarys(layouts, libraries, dressed, held, variant):
    library, floors = libraries[held], set()
    for name, layout in layouts.items():
        made = dressed(name, variant, held)
        room, twin = made.spec.room, layout.room
        assert room.polygon == tuple((x, y) for x, y in twin.polygon) == tuple(
            corners_of(layout.scene_gt.bounds.polygon)), name
        assert (room.height, room.wall, room.dado, room.dado_height, room.lights) == (
            twin.height, twin.wall, twin.dado, twin.dado_height, twin.lights), name
        assert room.ceiling == "ceiling_tile" and room.floor in FLOORS
        built = constructions(layout, VARIANTS[variant], library, dressed.seed(name))
        assert room.openings == tuple(opening for one in built for opening in one.openings), name
        assert len(room.openings) == sum(item.kind in ("window", "door") for item in layout.items), name
        assert made.spec.lighting == "ceiling" and made.spec.hdri == library.hdris[0]
        assert type(made.spec.seed) is int and 0 <= made.spec.seed < 2 ** 31
        floors.add(room.floor)
    assert floors == set(FLOORS)
    # By the seed and not by the layout: one layout dressed again and again sees every floor, and two layouts
    # dressed with one seed the same floor.
    again = [dress(layouts["eval_02_no_one_place"], VARIANTS[variant], library, (0, 0, 2, attempt)).spec
             for attempt in range(12)]
    assert {spec.room.floor for spec in again} == set(FLOORS) and len({spec.seed for spec in again}) == 12
    assert len({dress(layouts[name], VARIANTS[variant], library, SEED).spec.room.floor
                for name in ("eval_02_no_one_place", "eval_04_no_one_object", "eval_10_conversation_group_2")}) == 1


# --- Groups, blocks and lettering. ---

@pytest.mark.parametrize("variant", VARIANTS)
def test_every_target_has_its_groups_blocks_and_lettering(layouts, libraries, dressed, held, variant):
    """A group number of its own for each target, 1, 2, ..., carried by what draws it and by nothing else:
    a place's constructions with the small things on them, the person, the construction that holds an object.
    The blocks hold what the groups draw, and the lettering is that of the target's constructions."""
    library, kinds = libraries[held], set()
    for name, layout in layouts.items():
        made = dressed(name, variant, held)
        bare = dressed(name, dataclasses.replace(VARIANTS[variant], name="bare", small_things=False), held)
        built = constructions(layout, VARIANTS[variant], library, dressed.seed(name))
        targets = layout.scene_gt.targets
        assert list(made.target_groups) == list(made.target_blocks) == list(made.lettering) == [t.id for t in targets]
        assert list(made.target_groups.values()) == [(number,) for number in range(1, len(targets) + 1)], name
        things = [*made.spec.parts, *made.spec.models, *made.spec.people]
        assert {thing.group for thing in things} <= {0, *range(1, len(targets) + 1)}, name
        assert all(ref.group == 0 for ref in staff(made))
        for target in targets:
            (group,), blocks = made.target_groups[target.id], made.target_blocks[target.id]
            own = own_items(layout, target)
            parts = [part for part in made.spec.parts if part.group == group]
            models = [one for one in made.spec.models if one.group == group]
            people = [ref for ref in made.spec.people if ref.group == group]
            assert all(type(side) is float for block in blocks for side in block) and isinstance(blocks, tuple)
            assert all(isinstance(block, tuple) and len(block) == 6 for block in blocks)
            assert all(isinstance(lettered, Lettered) for lettered in made.lettering[target.id])
            assert made.lettering[target.id] == tuple(lettered for index in own for lettered in built[index].lettering)
            kinds.add(target.kind)
            if target.kind == "person":
                (ref,), reach = people, placed_reach(people[0], library)
                assert ref.person_id == target.person_id and not parts and not models
                assert blocks == ((ref.x, ref.y, 0.0, reach, reach, ref.height),), name
                assert made.lettering[target.id] == ()
                continue
            assert not people and own, name
            # What the group draws is what the target's constructions build, and the small things on them.
            assert parts == [dataclasses.replace(part, group=group) for index in own for part in built[index].parts]
            fitted = [dataclasses.replace(one, group=group) for index in own for one in built[index].models]
            assert models[:len(fitted)] == fitted, name
            assert [one for one in bare.spec.models if one.group == group] == fitted, name
            small = models[len(fitted):]
            assert all(any(on_surface(thing, surface) for index in own for surface in built[index].surfaces)
                       for thing in small), name
            drawn_solids = [*(part.solid for part in parts), *(box_of(one) for one in models)]
            if target.footprint is None:            # an object: a block round its match radius, as high as it is
                (block,), radius = blocks, target.match_radius
                assert block[:5] == (target.anchor[0], target.anchor[1], 0.0, radius, radius)
                on_it = [solid for solid in drawn_solids if all(
                    math.dist(point, target.anchor[:2]) <= radius + 1e-6 for point in ground_points(solid))]
                assert on_it and block[5] == max(top_of(solid) for solid in on_it)
                assert target.anchor[2] < block[5] < target.anchor[2] + 0.2     # the parcel's top, not the room's
                continue
            # A place: one block for each of its items, on the item's own ground, up to the top of what is drawn.
            # What reaches the ceiling is what a sign over the place hangs by: drawn with it, and no part of it.
            ceiling = layout.room.height
            hangers = [solid for solid in drawn_solids if top_of(solid) >= ceiling - 1e-6]
            assert all(top_of(solid) == ceiling and not isinstance(solid, Box) for solid in hangers), name
            drawn_solids = [solid for solid in drawn_solids if solid not in hangers]
            assert len(blocks) == len(own)
            for block, index in zip(blocks, own):
                item = layout.items[index]
                assert block[2] == item.yaw
                stands_on = sorted(to_world(block[:3], i * block[3], j * block[4]) for i in (-1, 1) for j in (-1, 1))
                assert np.allclose(stands_on, sorted(corners_of(item.footprint)), rtol=0.0, atol=1e-9), name
                mine = [*(part.solid for part in built[index].parts), *(box_of(one) for one in built[index].models),
                        *(box_of(thing) for thing in small
                          if any(on_surface(thing, surface) for surface in built[index].surfaces))]
                assert block[5] == max(top_of(solid) for solid in mine if top_of(solid) < ceiling - 1e-6), name
            for solid in drawn_solids:              # but for a plate, which stands 3 mm out of its face
                assert all(any(in_block(point, block, PLATE_M + 1e-9) and top_of(solid) <= block[5]
                               for block in blocks) for point in ground_points(solid)), (name, solid)
            assert made.lettering[target.id], name   # a place says what it is, in either signage
    assert kinds == {"place", "object", "person"}
    two = dressed("two_places", variant, held)
    assert two.target_groups == {"t0": (1,), "t1": (2,)} and len(two.target_blocks["t1"]) == 1


@pytest.mark.parametrize("variant", VARIANTS)
def test_no_target_is_held_by_a_block_higher_than_the_placeholder_builds_a_place(layouts, dressed, held, variant):
    """The placeholder builds no target higher than 2.25 m and stands the robot far enough off for that to be in
    its view; the blocks say which tour frames show a target. A sign hung over a desk ends at that height, and
    the rods it hangs by, which reach the ceiling, are in the target's group and not in its block."""
    hung = 0
    for name, layout in layouts.items():
        made = dressed(name, variant, held)
        for target in layout.scene_gt.targets:
            for block in made.target_blocks[target.id]:
                assert block[5] <= recipes._PLACE_TOP_M + 1e-9, (name, target.id, block)
        groups = {group for owned in made.target_groups.values() for group in owned}
        hung += any(part.group in groups and top_of(part.solid) >= layout.room.height - 1e-6
                    for part in made.spec.parts)
    assert hung >= 4 if variant == "real" else hung == 0        # a desk in natural signage has such a sign

@pytest.mark.parametrize("variant", VARIANTS)
def test_the_description_is_plain_data_that_json_holds(layouts, dressed, held, variant):
    for name in layouts:
        spec = dressed(name, variant, held).spec
        assert is_plain(spec), name
        stored = json.loads(json.dumps(to_json(spec)))
        assert stored == to_json(spec) and from_json(stored) == spec, name
        assert isinstance(spec.parts, tuple) and isinstance(spec.models, tuple) and isinstance(spec.people, tuple)


def test_the_same_seed_dresses_the_same_way_and_another_seed_another(layouts, libraries, held):
    library = libraries[held]
    for name in ("eval_08_waiting_line_4", "eval_12_conversation_group_4", "eval_03_no_one_deep_place", "two_places"):
        layout = layouts[name]
        for variant in VARIANTS.values():
            first, again = dress(layout, variant, library, SEED), dress(layout, variant, library, list(SEED))
            assert first == again, name
            assert json.dumps(to_json(first.spec)) == json.dumps(to_json(again.spec))
        other = dress(layout, VARIANTS["real"], library, (*SEED[:3], SEED[3] + 1))
        first = dress(layout, VARIANTS["real"], library, SEED)
        assert other.spec != first.spec and other.spec.seed != first.spec.seed, name
        assert other.spec.people != first.spec.people, name
        # Another dressing of the same layout: the twin's part of the ground truth does not move.
        assert other.scene_gt.targets == first.scene_gt.targets and other.scene_gt.bounds == first.scene_gt.bounds
        assert [(p.x, p.y, p.yaw) for p in other.scene_gt.people] == [(p.x, p.y, p.yaw) for p in first.scene_gt.people]
        assert other.target_groups == first.target_groups
    # A longer or a shorter seed is another seed.
    line = layouts["eval_08_waiting_line_4"]
    assert dress(line, VARIANTS["real"], library, (0,)).spec != dress(line, VARIANTS["real"], library, (0, 0)).spec


def test_each_part_of_a_dressing_has_random_numbers_of_its_own(layouts, libraries, monkeypatch):
    """Five generators, seeded with the seed and 0 to 4: the constructions, the cast, the staff, the small
    things and the room's look, in that order."""
    import sgi.synth.realistic.dress as module
    asked = []
    real_rng = np.random.default_rng

    def recording(seed=None):
        asked.append(seed)
        return real_rng(seed)

    small, line = libraries["small"], layouts["eval_08_waiting_line_4"]
    with monkeypatch.context() as patch:
        patch.setattr(module.np.random, "default_rng", recording)
        crowded = dress(line, VARIANTS["real"], small, SEED)
    assert [list(seed) for seed in asked] == [[*SEED, number] for number in range(5)]
    # Each part takes its numbers from its own generator alone. The staff member is as much under their height
    # with nobody waiting as with four in line, which is the staff's second number whoever was cast before.
    nobody = dataclasses.replace(line, figures=(), scene_gt=line.scene_gt.model_copy(update={"people": []}))
    on_duty = list(zip(staff(crowded), staff(dress(nobody, VARIANTS["real"], small, SEED)), strict=True))
    assert len(on_duty) == 2                        # behind the stall, and at the second stall
    for busy, idle in on_duty[:1]:                  # the first is drawn before anyone on duty wears an avatar
        assert small.avatars[busy.avatar].natural_height - busy.height == pytest.approx(
            small.avatars[idle.avatar].natural_height - idle.height, abs=1e-12)
    # The small things stand the same with the staff left out, and the people the same with the small things.
    unstaffed = dress(line, Variant("unstaffed", "natural", False, True), small, SEED)
    assert unstaffed.spec.models == crowded.spec.models and len(crowded.spec.models) > len(
        dress(line, BARE, small, SEED).spec.models)
    assert unstaffed.spec.people == crowded.spec.people[:4] and unstaffed.spec.parts == crowded.spec.parts


def test_a_layout_without_a_room_is_refused(layouts, libraries):
    bare = dataclasses.replace(layouts["eval_02_no_one_place"], room=None)
    with pytest.raises(ValueError, match="room"):
        dress(bare, VARIANTS["plain"], libraries["small"], SEED)


# --- The scene check, through the stub. ---

@pytest.mark.parametrize("name", CHECKED)
def test_a_dressed_scene_passes_the_scene_check_with_the_stubs_depth(name, layouts, dressed, held, tmp_path):
    """Written as a scene folder with the layout's robot, camera and episodes, each variant loads and passes
    with no error, and with no warning that its placeholder twin does not raise."""
    layout = layouts[name]
    assert [item.kind for item in layout.items if item.source == "boxes"] == ABOUT[name]
    twin_scene, twin_rgb, twin_depth = gen.build_scene(layout, name)
    save_scene(tmp_path / name, twin_scene, twin_rgb, twin_depth)
    twin = findings(tmp_path / name)
    assert set(twin) <= {gen._EXPECTED_WARNING}
    for variant in VARIANTS if held == "small" else ("real",):
        folder = tmp_path / f"{variant}_{name}"
        write_scene(folder, layout, dressed(name, variant, held))
        found = findings(folder)
        assert set(found) <= set(twin), (variant, found)


def test_the_scenes_checked_have_staff_a_deep_place_a_second_group_and_someone_alone(layouts, dressed):
    """What the scene check above has to see, so that it is not passed by scenes that hold none of it."""
    seen = set()
    for name in CHECKED:
        layout, made = layouts[name], dressed(name)
        seen |= {item.kind for item in layout.items if item.source == "boxes"}
        seen |= {f"staff at a {item.kind}" for item in layout.items if item.kind in STAFFED_KINDS and staff(made)}
        seen |= set(doing(layout).values())
    assert {"desk", "stall", "ticket_counter", "back_room", "parcel_table", "kiosk", "staff at a desk",
            "staff at a stall", "staff at a ticket_counter", "line", "circle", "pair", "alone", "asked"} <= seen
