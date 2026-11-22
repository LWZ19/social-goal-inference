"""The items of a placeholder layout and the stream of accepted layouts.

An item says what a run of a layout's primitives is and where it stands, so that a second generator can build
the same thing another way. accepted_layouts gives that generator the layouts generate() keeps, without tours.
"""
import dataclasses
import inspect
import math
from functools import lru_cache

import numpy as np
import pytest

import sgi.synth.generate as gen
from sgi.config import load_config
from sgi.geometry import wrap_angle
from sgi.polygons import distance_to_boundary, distance_to_polygon, point_in_polygon
from sgi.scene import read_scene_file
from sgi.synth import recipes
from sgi.synth.render import Box, Sphere
from tests.test_run import placeholder  # noqa: F401
from tests.test_synth import EXTRAS, ROWS

CFG = load_config()
ALL = list(dict.fromkeys([*ROWS, *EXTRAS]))        # every row of test_synth's two lists, once
SEEDS = range(5)                                    # with these the rows draw every kind of place: seed 4, the booth
PLACES = {"stall": recipes._stall, "ticket_counter": recipes._ticket_counter, "desk": recipes._desk,
          "kiosk": recipes._kiosk}
FURNISHINGS = {"plant": recipes._plant, "bench": recipes._bench, "bin": recipes._bin, "pillar": recipes._pillar,
               "poster": recipes._poster, "door": recipes._door, "window": recipes._window,
               "board": recipes._notice_board, "cooler": recipes._water_cooler, "lockers": recipes._lockers,
               "extinguisher": recipes._extinguisher, "shop": recipes._shop}
BUILDERS = {**PLACES, "back_room": recipes._back_room, "parcel_table": recipes._parcel_table, **FURNISHINGS}
OPTIONS = {**dict.fromkeys(PLACES, {"label", "text", "color"}), "back_room": {"depth", "text", "color"},
           "plant": {"scale"}, "window": {"width"}, "poster": {"text", "ink", "width"},
           "shop": {"text", "color", "goods"}, "bench": {"length"}}       # every other kind: none


def rng(seed=0):
    return np.random.default_rng(seed)


def name(row):
    return "_".join(str(field) for field in row)


@lru_cache(maxsize=None)
def layouts(row):                                   # as the generator draws them: extras added, then moved at random
    return [gen._layout(rng(seed), row) for seed in SEEDS]


def corners(polygon):                               # a polygon's corners, whatever holds them
    return [tuple(corner) for corner in polygon]


def of(lay, source):
    return [item for item in lay.items if item.source == source]


def same(a, b, tol=1e-9):                           # two primitives alike: in place up to rounding, in the rest exactly
    if type(a) is not type(b):
        return False
    for field in dataclasses.fields(a):
        one, other = getattr(a, field.name), getattr(b, field.name)
        if field.name in ("cx", "cy", "x", "y"):
            alike = abs(one - other) <= tol
        elif field.name == "yaw":
            alike = abs(wrap_angle(one - other)) <= tol
        else:
            alike = one == other
        if not alike:
            return False
    return True


def assert_tiled(lay):                              # the items' ranges tile `boxes` and `props`, in order, none empty
    assert {item.source for item in lay.items} <= {"boxes", "props"}
    for source in ("boxes", "props"):
        runs = [(item.start, item.stop) for item in of(lay, source)]
        assert all(start < stop for start, stop in runs)
        assert [start for start, _ in runs] + [len(getattr(lay, source))] == [0] + [stop for _, stop in runs], source


def assert_rebuilt(lay):                            # each builder, called with its options and placed, gives the run
    for item in lay.items:
        build = BUILDERS[item.kind]
        own = {parameter: item.options[parameter] for parameter in inspect.signature(build).parameters}
        rebuilt = recipes._placed(build(**own), item.x, item.y, item.yaw)
        drawn = getattr(lay, item.source)[item.start:item.stop]
        assert len(rebuilt) == len(drawn) > 0, item
        assert all(same(a, b) for a, b in zip(rebuilt, drawn)), item


def in_frame(item, point):                          # a ground point of the world, in the item's own frame
    c, s = math.cos(item.yaw), math.sin(item.yaw)
    dx, dy = point[0] - item.x, point[1] - item.y
    return c * dx + s * dy, c * dy - s * dx


def sides(item, polygon):                           # (x0, y0, x1, y1) of the polygon's corners, in the item's frame
    xs, ys = zip(*(in_frame(item, corner) for corner in polygon))
    return min(xs), min(ys), max(xs), max(ys)


def filled(item, parts):                            # the same of the ground that the primitives fill between them
    xs, ys = [], []
    for part in parts:
        if isinstance(part, Box):
            c, s = math.cos(part.yaw), math.sin(part.yaw)
            ground = [in_frame(item, (part.cx + c * i * part.sx / 2 - s * j * part.sy / 2,
                                      part.cy + s * i * part.sx / 2 + c * j * part.sy / 2))
                      for i in (-1, 1) for j in (-1, 1)]
        else:                                       # round: the disc about its axis
            reach = part.radius if isinstance(part, Sphere) else max(part.radius, part.top_radius or 0.0)
            x, y = in_frame(item, (part.x, part.y))
            ground = [(x - reach, y - reach), (x + reach, y + reach)]
        xs += [x for x, _ in ground]
        ys += [y for _, y in ground]
    return min(xs), min(ys), max(xs), max(ys)


def near(box):                                      # the four sides of a rectangle, up to rounding
    return pytest.approx(box, abs=1e-9)


def assert_stood_on(lay):                           # an item's footprint is its own ground: square to its frame, and
    for item in lay.items:                          # every one of its primitives stands on it
        x0, y0, x1, y1 = sides(item, item.footprint)
        own = [in_frame(item, corner) for corner in item.footprint]
        assert len(own) == 4 and x0 < x1 and y0 < y1, item
        assert all(any(math.dist(corner, at) < 1e-9 for corner in own)
                   for at in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))), item
        fx0, fy0, fx1, fy1 = filled(item, getattr(lay, item.source)[item.start:item.stop])
        assert x0 - 1e-9 <= fx0 and y0 - 1e-9 <= fy0 and fx1 <= x1 + 1e-9 and fy1 <= y1 + 1e-9, item
        if item.kind in PLACES:                     # the 0.8 m people are served at, across the width of the place
            assert (x0, y0, x1, y1) == near((-0.8, -0.8, 0.0, 0.8)), item
        elif item.kind == "back_room":              # the rest of the place, behind that
            assert (x0, y0, x1, y1) == near((-item.options["depth"], -0.8, -0.8, 0.8)), item
        else:                                       # a furnishing, a table with its crate: no more than they fill
            assert (x0, y0, x1, y1) == near((fx0, fy0, fx1, fy1)), item


def assert_accounted_for(lay):                      # what the footprints are to the obstacles and the target footprint
    (target,), obstacles = lay.scene_gt.targets, [corners(o.polygon) for o in lay.scene_gt.obstacles]
    about, n = of(lay, "boxes"), len(lay.footprints)
    assert obstacles[:n] == [corners(footprint) for footprint in lay.footprints]
    # A furnishing has an obstacle to itself, which is its footprint: they come in the order of the items.
    assert [corners(item.footprint) for item in of(lay, "props")] == obstacles[n:]
    # No obstacle is left out: each lies on the footprints of the items.
    assert all(any(distance_to_polygon(corner, item.footprint) <= 1e-9 for item in lay.items)
               for obstacle in obstacles for corner in obstacle)
    if not about:                                   # a person is asked for: nothing stands in the room but furnishings
        assert n == 0 and target.footprint is None
    elif about[0].kind == "parcel_table":           # one item on two obstacles: its footprint holds both and is neither
        (item,), (table, crate) = about, lay.footprints
        assert target.footprint is None and corners(item.footprint) not in obstacles
        assert sides(item, table) == near((-0.45, -0.65, 0.45, 0.65)) == filled(item, lay.boxes[:1])       # the top
        assert sides(item, crate) == near((0.75, 0.2, 1.05, 0.5)) == filled(item, lay.boxes[2:3])
        assert sides(item, item.footprint) == near((-0.45, -0.65, 1.05, 0.65))
    else:                                           # a place: the one obstacle under it is the target's footprint
        front, *behind = about
        assert n == 1 and obstacles[0] == corners(target.footprint)
        if not behind:                              # one item, which has all of it
            assert corners(front.footprint) == obstacles[0]
            return
        (back,) = behind                            # two items, which have it between them and meet on one line
        both = [*corners(front.footprint), *corners(back.footprint)]
        assert set(obstacles[0]) < set(both) and len(set(both)) == 6
        assert sides(front, target.footprint) == near((-back.options["depth"], -0.8, 0.0, 0.8))
        assert corners(front.footprint) not in obstacles and corners(back.footprint) not in obstacles


@pytest.mark.parametrize("row", ALL, ids=name)
def test_items_cover_every_primitive_exactly_once(row):
    for lay in layouts(row):
        assert_tiled(lay)
        assert of(lay, "props") and bool(of(lay, "boxes")) == bool(lay.boxes)


@pytest.mark.parametrize("row", ALL, ids=name)
def test_an_item_frame_and_options_rebuild_its_primitives(row):
    for lay in layouts(row):
        assert_rebuilt(lay)


@pytest.mark.parametrize("row", ALL, ids=name)
def test_every_primitive_of_an_item_stands_on_its_footprint(row):
    for lay in layouts(row):
        assert_stood_on(lay)


# The plan calls this test "an item footprint is among the obstacles or is the target footprint". So it is for a
# furnishing and for a place that is one item. A footprint is also the ground its item's primitives stand on (the
# test above), and the two items of a deep place and a parcel table with its crate cannot be both: of a deep place
# the footprints divide the obstacle between them, and a parcel table's holds its two.
@pytest.mark.parametrize("row", ALL, ids=name)
def test_the_item_footprints_account_for_every_obstacle_and_for_the_target_footprint(row):
    for lay in layouts(row):
        assert_accounted_for(lay)


def test_a_deep_place_divides_the_target_footprint_between_its_front_and_its_back_room():
    for seed in SEEDS:                              # in the canonical frame: the front on x = 0, the place behind it
        lay, shallow = (recipes.recipe_no_one(rng(seed), 30.0, kind) for kind in ("deep_place", "place"))
        (front, back), (target,), (alone,) = of(lay, "boxes"), lay.scene_gt.targets, of(shallow, "boxes")
        assert corners(target.footprint) == [(-2.4, -0.8), (0.0, -0.8), (0.0, 0.8), (-2.4, 0.8)]
        assert corners(front.footprint) == [(-0.8, -0.8), (0.0, -0.8), (0.0, 0.8), (-0.8, 0.8)]
        assert corners(back.footprint) == [(-2.4, -0.8), (-0.8, -0.8), (-0.8, 0.8), (-2.4, 0.8)]
        assert [corners(o.polygon) for o in lay.scene_gt.obstacles[:1]] == [corners(target.footprint)]
        # The front has the ground of the same place with nothing behind it, which there is the target's footprint.
        assert alone.kind == front.kind and corners(alone.footprint) == corners(front.footprint)
        assert corners(alone.footprint) == corners(shallow.scene_gt.targets[0].footprint)
        # The floor behind the counter is the front's: the room behind does not reach over it.
        assert distance_to_polygon((-0.6, 0.0), front.footprint) == 0.0
        assert distance_to_polygon((-0.6, 0.0), back.footprint) == pytest.approx(0.2)


def test_a_parcel_table_footprint_holds_the_table_and_the_crate_and_the_free_floor_between_them():
    for lay in layouts(("no_one", 0, "object", 0, False)):
        (item,), (table, crate) = of(lay, "boxes"), lay.footprints
        assert [corners(o.polygon) for o in lay.scene_gt.obstacles[:2]] == [corners(table), corners(crate)]
        assert sides(item, item.footprint) == near((-0.45, -0.65, 1.05, 0.65))     # the table, and on to the crate
        c, s = math.cos(item.yaw), math.sin(item.yaw)
        between = (item.x + 0.6 * c - 0.35 * s, item.y + 0.6 * s + 0.35 * c)       # in its frame, (0.6, 0.35)
        # On the footprint, 0.15 m from the table and from the crate, and on no obstacle: the footprint is none.
        assert distance_to_polygon(between, item.footprint) == 0.0
        assert distance_to_polygon(between, table) == pytest.approx(0.15)
        assert distance_to_polygon(between, crate) == pytest.approx(0.15)
        assert min(distance_to_polygon(between, o.polygon) for o in lay.scene_gt.obstacles) == pytest.approx(0.15)


@pytest.mark.parametrize("row", ALL, ids=name)
def test_a_place_stands_under_its_anchor_facing_out_and_a_furnishing_on_a_wall_facing_in(row):
    for lay in layouts(row):
        (target,), floor, robot = lay.scene_gt.targets, lay.scene_gt.bounds.polygon, lay.robot
        for item in lay.items:
            assert -math.pi < item.yaw <= math.pi, item         # wrapped, as every yaw of a layout is
            ahead = (item.x + 0.1 * math.cos(item.yaw), item.y + 0.1 * math.sin(item.yaw))
            if item.source == "props":                          # on the wall, and squarely off it into the room
                assert distance_to_boundary((item.x, item.y), floor) < 1e-9 and point_in_polygon(ahead, floor), item
                assert distance_to_boundary(ahead, floor) == pytest.approx(0.1), item
                continue
            assert (item.x, item.y) == pytest.approx(target.anchor[:2]), item
            if item.kind == "parcel_table":                     # it has no front: it is turned to the robot
                assert abs(wrap_angle(item.yaw - math.atan2(robot.y - item.y, robot.x - item.x))) < 1e-9
            else:
                assert abs(wrap_angle(item.yaw - target.front_yaw)) < 1e-9, item
                assert not point_in_polygon(ahead, target.footprint), item


def test_every_kind_of_item_turns_up_with_the_options_its_builder_was_called_with():
    assert set(FURNISHINGS) == {*recipes._FURNISHINGS, "shop"}
    places = [(label, text, color, build) for label, text, color, build, *_ in recipes._PLACES]
    seen = set()
    for row in ALL:
        for lay in layouts(row):
            (target,), about = lay.scene_gt.targets, of(lay, "boxes")
            for item in lay.items:
                assert item.kind in BUILDERS and set(item.options) == OPTIONS.get(item.kind, set()), item
                assert item.source == ("props" if item.kind in FURNISHINGS else "boxes"), item
                seen.add(item.kind)
                options = item.options
                if item.kind in PLACES:                         # a row of the table of places, under the target's label
                    assert (options["label"], options["text"], options["color"], PLACES[item.kind]) in places
                    assert options["label"] == target.label and target.kind == "place"
                elif item.kind == "shop":
                    assert (options["text"], options["color"], options["goods"]) in recipes._SHOPS
                elif item.kind == "poster":
                    assert (options["text"], options["ink"], options["width"]) in recipes._POSTERS
                elif item.kind == "plant":
                    assert 0.95 <= options["scale"] <= 1.2
                elif item.kind == "window":
                    assert 1.6 <= options["width"] <= 2.2
                elif item.kind == "bench":
                    assert options["length"] == 1.6
            kinds = [item.kind for item in about]
            if row[2] == "deep_place":                          # the front and the back room are two items
                front, back = about
                assert front.kind in PLACES and back.kind == "back_room" and back.stop - back.start == 2
                assert back.options == {"depth": 2.4, "text": front.options["text"], "color": front.options["color"]}
                assert (back.x, back.y, back.yaw) == (front.x, front.y, front.yaw)
            elif row[2] == "object":
                assert kinds == ["parcel_table"] and (about[0].start, about[0].stop) == (0, len(lay.boxes))
            elif target.kind == "place":
                assert len(kinds) == 1 and kinds[0] in PLACES
            else:
                assert kinds == [] and lay.boxes == ()          # a person: nothing but the furnishings
    assert seen == set(BUILDERS)


def test_items_move_with_the_layout():
    lay = recipes.recipe_no_one(rng(2), 30.0, "deep_place")
    front, back = of(lay, "boxes")                              # in the canonical frame a place fronts +x at the origin
    assert (front.x, front.y, front.yaw) == (back.x, back.y, back.yaw) == (0.0, 0.0, 0.0)
    # The furnishings face in from the four walls, and the one that faces along -y does so at -90 degrees, not at 270.
    assert {round(math.degrees(item.yaw)) for item in of(lay, "props")} == {0, 90, 180, -90}
    assert all(-math.pi < item.yaw <= math.pi for item in lay.items)
    out = recipes.transform_layout(lay, math.pi / 2, 3.0, -2.0)             # (x, y) -> (3 - y, x - 2)
    assert len(out.items) == len(lay.items) > 6
    for before, after in zip(lay.items, out.items):
        assert dataclasses.replace(after, x=0.0, y=0.0, yaw=0.0, footprint=[]) == dataclasses.replace(
            before, x=0.0, y=0.0, yaw=0.0, footprint=[])       # what it is, its options and its range stay
        assert (after.x, after.y) == pytest.approx((3.0 - before.y, before.x - 2.0))
        assert -math.pi < after.yaw <= math.pi and math.cos(after.yaw - before.yaw - math.pi / 2) == pytest.approx(1.0)
        assert np.allclose(after.footprint, [(3.0 - y, x - 2.0) for x, y in before.footprint])
    assert {round(math.degrees(item.yaw)) for item in of(out, "props")} >= {-90, 180}     # 270 and 180 degrees, wrapped
    for moved in (lay, out):
        assert_tiled(moved)
        assert_rebuilt(moved)
        assert_stood_on(moved)
        assert_accounted_for(moved)


def test_adding_people_keeps_the_place_items_and_refurnishes():
    lay = recipes.recipe_no_one(rng(1), 30.0, "deep_place")
    newcomer = recipes._person(rng(5), 0, 2.0, -9.0, 0.0, (60, 90, 200))    # far to one side: the room grows round them
    more = recipes._with_more_people(lay, rng(7), [newcomer])
    assert more.boxes == lay.boxes and of(more, "boxes") == of(lay, "boxes") and len(of(lay, "boxes")) == 2
    floor = more.scene_gt.bounds.polygon
    assert min(y for _, y in floor) < min(y for _, y in lay.scene_gt.bounds.polygon) - 1.0
    assert more.props != lay.props and of(more, "props") != of(lay, "props")
    assert all(distance_to_boundary((item.x, item.y), floor) < 1e-9 for item in of(more, "props"))   # on the new walls
    assert_tiled(more)
    assert_rebuilt(more)
    assert_stood_on(more)
    assert_accounted_for(more)


def test_a_layout_built_by_hand_has_no_items():
    lay = recipes.recipe_no_one(rng(), 0.0, "person")
    bare = recipes.Layout(lay.name, lay.boxes, lay.figures, lay.robot, lay.scene_gt, lay.episodes)
    assert bare.items == () and isinstance(lay.items, tuple) and isinstance(lay.items[0], recipes.Item)
    with pytest.raises(dataclasses.FrozenInstanceError):
        lay.items[0].kind = "bench"


def test_an_item_and_the_stream_have_the_names_a_second_generator_reads():
    assert [field.name for field in dataclasses.fields(recipes.Item)] == [
        "kind", "x", "y", "yaw", "footprint", "options", "source", "start", "stop"]
    assert [field.name for field in dataclasses.fields(recipes.Layout)][-1] == "items"
    parameters = inspect.signature(gen.accepted_layouts).parameters
    assert list(parameters) == ["work_dir", "cfg", "seed"] and parameters["seed"].default == 0
    assert inspect.isgeneratorfunction(gen.accepted_layouts)


# What each row drew for seed 0 before layouts had items, taken from the recipes as they were then: how many
# primitives in `boxes` and in `props`, and the random number that came next. One draw more or fewer moves the last.
DRAWN = {("no_one", 30, "place", 1, False): (30, 105, 4383484563380973246),
         ("no_one", 30, "deep_place", 0, False): (32, 93, 1752477103795589356),
         ("no_one", 0, "object", 0, False): (11, 72, 2863436885560441321),
         ("no_one", 0, "person", 0, False): (0, 60, 1848109042443936927),
         ("waiting_line", 6, None, 0, False): (14, 100, 3897534530025584117),
         ("conversation_group", 3, None, 0, True): (0, 115, 206858495680298190),
         ("conversation_group", 2, None, 2, False): (0, 107, 3765323618380763626),
         ("conversation_group", 4, None, 1, True): (0, 121, 3616415170571332773)}


@pytest.mark.parametrize("row", ALL, ids=name)
def test_naming_the_items_draws_no_random_number(row):
    draw = rng(0)
    lay = gen._layout(draw, row)
    assert (len(lay.boxes), len(lay.props), int(draw.integers(2 ** 62))) == DRAWN[row]


def files_under(folder):
    return sorted(str(path.relative_to(folder)) for path in folder.rglob("*"))


def test_accepted_layouts_are_the_ones_generate_writes(placeholder, tmp_path):
    seen = list(gen.accepted_layouts(tmp_path, CFG, seed=0))
    assert [(split, index) for split, index, _, _ in seen] == [
        (split, index) for split, rows in gen._SPLITS for index in range(len(rows))]
    # generate() writes a second pass after these, numbered on from them in each split: a screen, pictures being taken
    # and a lift, with people and without, which this stream leaves out. So the stream is the folders of the first
    # pass, and no more.
    written = sorted(str(folder.relative_to(placeholder)) for folder in placeholder.glob("*/*"))
    rows = {split: len(rows) for split, rows in gen._SPLITS}
    first_pass = [name for name in written if int(name.split("/")[1].split("_")[1]) < rows[name.split("/")[0]]]
    assert len(written) - len(first_pass) == sum(len(rows) for _, rows in gen._MORE_SPLITS) == 9
    assert [f"{split}/{scene.scene_id}" for split, _, _, scene in seen] == first_pass
    for split, _, lay, scene in seen:
        written = read_scene_file(placeholder / split / scene.scene_id)
        truth, pose = written.ground_truth, written.observation.robot_pose_world
        assert written.scene_id == scene.scene_id and scene.ground_truth == lay.scene_gt
        assert truth.people == lay.scene_gt.people and truth.bounds == lay.scene_gt.bounds
        assert truth.obstacles == lay.scene_gt.obstacles
        assert [target.model_copy(update={"tour_frames": None}) for target in truth.targets] == lay.scene_gt.targets
        assert (pose.x, pose.y, pose.yaw) == (lay.robot.x, lay.robot.y, lay.robot.yaw)
        assert written.observation == scene.observation         # the camera too
        # The scene is the one the scene check saw: no tour yet, and no regions.
        assert scene.tour == [] and len(written.tour) == 8
        assert [(e.episode_id, e.request, e.context) for e in scene.episodes] == [
            (e.episode_id, e.request, e.context) for e in written.episodes]
        for mine, theirs in zip(scene.episodes, written.episodes):
            assert mine.ground_truth.acceptable_regions is None and mine.ground_truth.forbidden_regions is None
            assert theirs.ground_truth.acceptable_regions and mine.ground_truth.factors == theirs.ground_truth.factors
        assert all(target.tour_frames is None for target in scene.ground_truth.targets)
        assert_tiled(lay)
        assert_rebuilt(lay)
        assert_stood_on(lay)
        assert_accounted_for(lay)
    assert files_under(tmp_path) == []


def test_accepted_layouts_leave_nothing_behind(tmp_path, monkeypatch):
    mine = tmp_path / "eval" / "eval_00_no_one_place" / "scene.json"         # where generate() writes its first scene
    mine.parent.mkdir(parents=True)
    mine.write_text("mine")
    before = files_under(tmp_path)
    stream = gen.accepted_layouts(tmp_path, CFG, seed=0)
    for _ in range(2):                              # while it runs, a layout's folder is gone before it is handed out
        next(stream)
        assert [path for path in tmp_path.rglob("*") if path.is_file()] == [mine]
    stream.close()                                  # stopped early
    assert files_under(tmp_path) == before and mine.read_text() == "mine"
    monkeypatch.setattr(gen, "_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(gen, "_rejection", lambda scene_dir, cfg: "check_3")
    with pytest.raises(RuntimeError, match="eval_00_no_one_place: no layout accepted in 2 attempts; .*check_3=2$"):
        next(gen.accepted_layouts(tmp_path, CFG, seed=0))
    assert files_under(tmp_path) == before and mine.read_text() == "mine"
    monkeypatch.undo()
    stream = gen.accepted_layouts(tmp_path / "new" / "work", CFG, seed=0)    # a folder that is not there is made
    next(stream)
    stream.close()
    assert files_under(tmp_path / "new") == ["work"]


def test_a_rejected_layout_is_passed_over_as_generate_passes_over_it(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "_SPLITS", (("eval", gen._EVAL[6:8]),))        # two short waiting lines,
    monkeypatch.setattr(gen, "_MORE_SPLITS", ())                            # and no second pass after them
    real, turned_down = gen._rejection, []

    def rejection(scene_dir, cfg):                  # the first layout of a run is turned down, whatever it is
        if not turned_down:
            turned_down.append(scene_dir.name)
            return "check_3"
        return real(scene_dir, cfg)

    monkeypatch.setattr(gen, "_rejection", rejection)
    assert gen.generate(tmp_path / "out", CFG, seed=3)["check_3"] == 1
    turned_down.clear()
    seen = list(gen.accepted_layouts(tmp_path / "work", CFG, seed=3))
    assert turned_down == ["eval_00_waiting_line_1"]
    out = tmp_path / "out" / "eval"
    assert [scene.scene_id for *_, scene in seen] == sorted(folder.name for folder in out.iterdir())
    for _, _, lay, scene in seen:
        written = read_scene_file(out / scene.scene_id)
        assert written.observation == scene.observation and written.ground_truth.people == lay.scene_gt.people
        assert written.ground_truth.obstacles == lay.scene_gt.obstacles
    monkeypatch.setattr(gen, "_rejection", real)    # and with nothing turned down, the first layout is another one
    stream = gen.accepted_layouts(tmp_path / "work", CFG, seed=3)
    assert next(stream)[3].observation != seen[0][3].observation
    stream.close()
    assert files_under(tmp_path / "work") == []
