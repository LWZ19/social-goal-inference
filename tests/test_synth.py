import dataclasses
import math
from types import SimpleNamespace
import numpy as np
import pytest
from sgi.geometry import Camera, RobotPose, camera_on_robot, in_image, lift_all, project, wrap_angle
from sgi.inputs import ObservationData
from sgi.polygons import distance_to_boundary, distance_to_polygon, point_in_polygon
from sgi.synth import recipes
from sgi.synth.render import Box, Sphere, appearance, render

RED = (210, 40, 40)
PLACES = {"print kiosk", "help desk", "ticket counter", "coffee stall"}

def rng(seed=0):
    return np.random.default_rng(seed)

def ground(part):                                               # the ground rectangle of a box, or points round a round part
    if isinstance(part, Box):
        return recipes._rectangle(part)
    r = part.radius if isinstance(part, Sphere) else max(part.radius, part.top_radius or 0.0)
    return [(part.x + r * math.cos(k * math.pi / 8), part.y + r * math.sin(k * math.pi / 8)) for k in range(16)]

def top_of(part):
    return part.z0 + part.sz if isinstance(part, Box) else part.z + part.radius if isinstance(part, Sphere) else part.z0 + part.height

def within(part, polygons, slack=1e-6):                         # whether one of the polygons holds the whole part
    return any(all(distance_to_polygon(corner, polygon) <= slack for corner in ground(part)) for polygon in polygons)

def test_no_one_layouts_in_the_canonical_frame():
    sides = set()
    for seed in range(20):
        lay = recipes.recipe_no_one(rng(seed), 30.0)
        (t,), r = lay.scene_gt.targets, lay.robot
        assert (lay.name, t.id, t.kind, t.anchor, t.front_yaw) == ("no_one_place", "t0", "place", (0.0, 0.0, 1.0), 0.0)
        assert t.label in PLACES and [e.context for e in lay.episodes] == [None, None]
        assert 4.5 <= math.hypot(r.x, r.y) <= 5.5 and abs(math.degrees(math.atan2(r.y, r.x))) == pytest.approx(30.0)
        assert abs(wrap_angle(r.yaw - math.atan2(-r.y, -r.x))) <= math.radians(5.0) + 1e-9
        xs, ys = zip(*lay.scene_gt.bounds.polygon)              # 3 m round the robot and the box
        assert (min(xs), max(xs)) == pytest.approx((-3.8, r.x + 3.0)) and max(ys) - min(ys) == pytest.approx(abs(r.y) + 6.8)
        sides.add(r.y > 0)
    assert sides == {True, False}
    deep, obj, per = (recipes.recipe_no_one(rng(), 20.0, kind) for kind in ("deep_place", "object", "person"))
    assert min(x for x, _ in deep.scene_gt.targets[0].footprint) == pytest.approx(-2.4) and per.figures[0].color == RED
    (t,), crate, r = obj.scene_gt.targets, obj.boxes[2], obj.robot      # the crate: 0.9 m towards the robot, 0.35 m to the side
    assert (t.kind, t.label, t.anchor, t.footprint, t.match_radius, t.front_yaw) == ("object", "blue parcel", (0.0, 0.0, 0.825), None, 0.3, None)
    assert (crate.cx * r.x + crate.cy * r.y) / math.hypot(r.x, r.y) == pytest.approx(0.9)
    assert math.hypot(crate.cx, crate.cy) == pytest.approx(math.hypot(0.9, 0.35)) and len(obj.footprints) == 2     # the table and the crate
    table, parcel, others = obj.boxes[0], obj.boxes[1], obj.boxes[7:9]  # between them: the crate and the table's four legs
    assert (table.sx, table.sy, parcel.z0, parcel.color) == (0.9, 1.3, 0.7, (40, 80, 200)) and table.z0 + table.sz == pytest.approx(0.7)
    assert len(others) == 2 and all(within(part, [recipes._rectangle(table)]) and part.z0 == 0.7 for part in others)
    assert all(0.3 + 0.03 < min(math.hypot(x, y) for x, y in ground(part)) for part in others)   # none inside the parcel's match radius
    tapes = obj.boxes[9:]                                       # last, the tape over the parcel and over the other package
    assert len(tapes) == 2 and all(within(tape, [recipes._rectangle(table)]) and tape.z0 == 0.7 for tape in tapes)
    for tape, taped in zip(tapes, (parcel, others[0])):         # a strip 0.045 m wide along its top, 3 mm proud of the box it is on
        assert (tape.cx, tape.cy, tape.sx) == (taped.cx, taped.cy, 0.045) and 0 < tape.sy - taped.sy < 0.01 and 0 < tape.sz - taped.sz < 0.01
    assert max(math.hypot(x, y) for x, y in ground(parcel)) < 0.3 and parcel.panel is not None   # all of it inside the radius; a label on it

@pytest.mark.parametrize("label, text, color, build, grounded, high_level", recipes._PLACES, ids=[row[0] for row in recipes._PLACES])
def test_a_place_is_built_of_parts_that_carry_its_name_and_stay_on_its_footprint(label, text, color, build, grounded, high_level):
    parts = build(text, color)
    front = [(-0.8, -0.8), (0.0, -0.8), (0.0, 0.8), (-0.8, 0.8)]
    assert len(parts) >= 10 and all(within(part, [front]) and top_of(part) <= 2.25 + 1e-9 for part in parts)
    signs = [p for p in parts if isinstance(p, Box) and p.label == text and p.z0 == 1.9]     # above every head: the tallest is 1.84 m
    assert sorted(p.label_face for p in signs) == ["front", "left", "right"] and {p.color for p in signs} == {color}
    across, *along = sorted(signs, key=lambda p: p.label_face)  # the side boards end flush with the one across the front,
    for board in along:                                         # which then hides none of their letters from the side
        assert abs(board.cy) + board.sy / 2 == pytest.approx(across.sy / 2) and board.cx + board.sx / 2 == pytest.approx(across.cx - across.sx / 2)
    faced = [p for p in parts if isinstance(p, Box) and p.cx + p.sx / 2 == pytest.approx(0.0) and abs(p.cy) < p.sy / 2 and p.z0 <= 1.0 <= p.z0 + p.sz]
    assert len(faced) == 1                                      # the anchor (0, 0, 1) lies on the front of one part
    deep = [*parts, *recipes._back_room(2.4, text, color)]      # a deep place has a room behind, named on both sides
    assert len(deep) == len(parts) + 2 and sorted(p.label_face for p in deep[-2:]) == ["left", "right"]
    assert min(x for part in deep for x, _ in ground(part)) == pytest.approx(-2.4) and {p.label for p in deep[-2:]} == {text}

ROWS = [("no_one", 30, "place", 1, False), ("no_one", 30, "deep_place", 0, False), ("no_one", 0, "object", 0, False),
        ("no_one", 0, "person", 0, False), ("waiting_line", 6, None, 0, False), ("conversation_group", 3, None, 0, True)]

@pytest.mark.parametrize("row", ROWS, ids=lambda row: f"{row[0]}_{row[1]}_{row[2]}")
def test_every_layout_stands_in_a_furnished_room_and_every_solid_on_an_obstacle(row):
    for seed in range(3):
        lay = gen._layout(rng(seed), row)
        gt, robot = lay.scene_gt, lay.robot
        floor, obstacles, n = gt.bounds.polygon, [o.polygon for o in gt.obstacles], len(lay.footprints)
        assert list(lay.room.polygon) == floor and lay.room.height == 3.0 and obstacles[:n] == list(lay.footprints)
        furnishings = obstacles[n:]
        assert len(furnishings) >= 5 and len(lay.props) > len(furnishings)      # several things, each of several parts
        assert all(within(part, lay.footprints) for part in lay.boxes) and all(within(part, furnishings) for part in lay.props)
        action = [(p.x, p.y) for p in gt.people] + [corner for footprint in lay.footprints for corner in footprint]
        for polygon in furnishings:                             # against a wall, and far from everyone and from the target
            assert all(point_in_polygon(c, floor) for c in polygon) and min(distance_to_boundary(c, floor) for c in polygon) < 0.1
            assert max(distance_to_boundary(c, floor) for c in polygon) <= 1.1
            assert min(distance_to_polygon(point, polygon) for point in action) >= 2.6 - 1e-9
            assert distance_to_polygon((robot.x, robot.y), polygon) >= 1.9
        looked_at = [(p.x, p.y) for p in gt.people] + [t.anchor[:2] for t in gt.targets]
        sights = [point for x, y in looked_at for point in recipes._sight_line(robot, x, y)[::4]]
        assert min(distance_to_polygon(point, polygon) for point in sights for polygon in furnishings) >= 1.9
        signs = {p.label for p in lay.props if isinstance(p, Box) and p.z0 == 1.9}      # the second stall's sign, if it has room
        assert len(signs) <= 1 and not signs & {text for _, text, *_ in recipes._PLACES}

def test_the_people_of_a_layout_differ_and_only_the_target_person_wears_red():
    for row in (("waiting_line", 6, None, 0, False), ("waiting_line", 4, None, 2, False), ("conversation_group", 3, None, 1, True)):
        for seed in range(4):
            lay = gen._layout(rng(seed), row)
            shirts = [f.color for f in lay.figures]
            assert len(set(shirts)) == len(shirts) == 6 and shirts.count(RED) == (row[0] == "conversation_group")
            assert all(RED == f.color for f, p in zip(lay.figures, lay.scene_gt.people) if p.id == lay.scene_gt.targets[0].person_id)
            for i, f in enumerate(lay.figures):                 # person i has the build, skin, hair and trousers of index i
                assert (f.height, f.skin, f.hair, f.trousers) == tuple(appearance(i)[k] for k in ("height", "skin", "hair", "trousers"))
            looks = [(f.height, f.skin, f.hair, f.trousers) for f in lay.figures]
            assert len(set(looks)) == 6 and len({f.height for f in lay.figures}) == 6 and max(f.height for f in lay.figures) < 1.9

def square(x, y, half=0.2):
    return [(x - half, y - half), (x + half, y - half), (x + half, y + half), (x - half, y + half)]

def test_the_view_tells_what_the_picture_holds_misses_and_hides():
    view = recipes._View(RobotPose(0.0, 0.0, 0.0))             # the camera 0.2 m ahead of the robot, looking along +x
    ahead, beside, across, behind = square(5.0, 0.0), square(5.0, 4.0), square(5.0, 2.7), square(-3.0, 0.0)
    assert view.holds(ahead, 0.0, 2.0) and not view.misses(ahead, 0.0, 2.0)
    assert view.misses(beside, 0.0, 2.0) and not view.holds(beside, 0.0, 2.0) and view.misses(behind, 0.0, 2.0)
    assert not view.holds(across, 0.0, 2.0) and not view.misses(across, 0.0, 2.0)       # the image's side cuts it
    assert not view.holds(square(2.0, 0.0), 0.0, 2.2)          # 1.6 m before the camera the image ends 1.88 m up
    assert view.span(behind) is None and view.span(ahead) == pytest.approx((-math.atan2(0.2, 4.6), math.atan2(0.2, 4.6), 4.6, math.hypot(5.0, 0.2)), abs=1e-2)
    near, far, aside = view.span(square(3.0, 0.0)), view.span(square(8.0, 0.1)), view.span(square(8.0, 3.0))
    assert view.hides(near, 1.8, far, 1.0) and not view.hides(far, 1.8, near, 1.0) and not view.hides(near, 1.8, aside, 1.0)
    assert not view.hides(near, 1.2, far, 1.0) and view.hides(near, 1.2, far, 0.0)      # the camera looks over it, but not at the floor behind
    assert view.faces(ahead, math.pi) and view.faces(ahead, math.radians(180 - 60)) and not view.faces(ahead, math.radians(180 - 65))
    draw = rng(1)
    for _ in range(300):                                        # holds and misses against the projection itself
        robot = RobotPose(draw.uniform(-3, 3), draw.uniform(-3, 3), draw.uniform(-math.pi, math.pi))
        polygon, (low, high) = square(draw.uniform(-8, 8), draw.uniform(-8, 8), draw.uniform(0.1, 1.0)), sorted(draw.uniform(0.0, 2.6, 2))
        seen = [project(mounted(robot), (x, y, z)) for x, y in polygon for z in (low, high)]
        inside = [uvz is not None and in_image(mounted(robot), uvz[0], uvz[1]) for uvz in seen]
        view = recipes._View(robot)
        assert not (view.holds(polygon, low, high) and not all(inside)) and not (view.misses(polygon, low, high) and any(inside))

def lettered(lay):                                              # the posters and the second stall's sign, among the furnishings
    return [p for p in lay.props if isinstance(p, Box) and p.label]

def face_points(box, inset=0.06):                               # a grid over the lettered front face of a box, just before it
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    return [(box.cx + c * (box.sx / 2 + 0.01) - s * a * box.sy, box.cy + s * (box.sx / 2 + 0.01) + c * a * box.sy, box.z0 + b * box.sz)
            for a in np.linspace(inset - 0.5, 0.5 - inset, 7) for b in np.linspace(inset, 1 - inset, 4)]

def shown(obs, point):                                          # whether the picture shows the point itself: None when it is not in it
    uvz = project(obs.camera, point)
    if uvz is None or not in_image(obs.camera, uvz[0], uvz[1]):
        return None
    return abs(float(obs.depth[int(uvz[1]), int(uvz[0])]) - uvz[2]) < 0.05

def rect(camera, polygon, low, high):                           # (u0, u1, v0, v1) round what stands on a polygon; None when not all ahead
    seen = [project(camera, (x, y, z)) for x, y in polygon for z in (low, high)]
    if any(uvz is None for uvz in seen):
        return None
    u, v, _ = zip(*seen)
    return min(u), max(u), min(v), max(v)

def apart(a, b):                                                # whether two such frames do not overlap; one that is None overlaps nothing
    return a is None or b is None or a[0] > b[1] or a[1] < b[0] or a[2] > b[3] or a[3] < b[2]

LONG = {"BBBB", "CC"}                                           # a bench and a pillar, by their parts: boxes, cylinders, spheres
SMALL = {"CCSB", "CCSSSSSS", "BCC", "BBCCB"}                    # a bin, a plant, a water cooler, a fire extinguisher

@pytest.mark.parametrize("row", ROWS + [("waiting_line", 2, None, 0, False), ("waiting_line", 4, None, 2, False), ("conversation_group", 4, None, 0, False),
                                        ("conversation_group", 2, None, 0, False), ("no_one", 60, "place", 0, False)], ids=lambda row: f"{row[0]}_{row[1]}_{row[2]}")
def test_the_picture_shows_lettering_whole_or_not_at_all_and_nothing_behind_what_the_scene_is_about(row):
    kinds = set()
    for seed in range(4):
        lay = gen._layout(rng(seed), row)
        obs, gt = observed(lay), lay.scene_gt
        for sign in lettered(lay):                              # every point of its face is in the picture, or none is
            seen = [shown(obs, point) for point in face_points(sign)]
            assert all(seen) or seen.count(None) == len(seen), (seed, sign.label, seen)
            if all(seen):                                       # and it is read from no more than 62 degrees off square
                to_camera = np.subtract(obs.camera.T_world_camera[:2, 3], (sign.cx, sign.cy))
                assert to_camera @ (math.cos(sign.yaw), math.sin(sign.yaw)) >= math.cos(math.radians(62.5)) * np.hypot(*to_camera)
        n = len(lay.footprints)
        top = max((top_of(part) for part in lay.boxes), default=0.0)
        asked = [recipes._body(p) for p in gt.people if p.id == gt.targets[0].person_id]
        about = [rect(obs.camera, polygon, 0.0, top) for polygon in lay.footprints] + [rect(obs.camera, body, 0.0, 1.85) for body in asked]
        bodies = [rect(obs.camera, recipes._body(p), 0.0, 1.85) for p in gt.people]
        for polygon in (o.polygon for o in gt.obstacles[n:]):
            parts = [part for part in lay.props if within(part, [polygon])]
            kind, (low, high) = "".join(type(part).__name__[0] for part in parts), recipes._heights(parts)
            mine = rect(obs.camera, polygon, low, high)
            assert all(apart(mine, other) for other in about), (seed, kind)      # none overlaps the place, the table or the person asked for
            corners = [project(obs.camera, (x, y, z)) for x, y in polygon for z in (low, high)]
            inside = [uvz is not None and in_image(obs.camera, uvz[0], uvz[1]) for uvz in corners]
            assert kind in LONG or all(inside) or not any(inside), (seed, kind)     # but for the long ones, each is whole in the picture or not in it
            assert kind not in SMALL or all(apart(mine, other) for other in bodies), (seed, kind)     # and nobody stands before a small one
            kinds.add(kind)
    assert kinds & LONG and len(kinds & SMALL) >= 3              # the layouts have such things

def test_a_picture_with_room_for_it_shows_a_poster_or_the_second_stall_whole():
    rows = [row for row in gen._EVAL if row[0] == "no_one" or row[:2] in (("conversation_group", 3), ("conversation_group", 4))]
    with_lettering = []                                         # behind a line seen from the side, people and the place hide the far wall
    for row in rows:
        for seed in range(4):
            lay = gen._layout(rng(seed), row)
            frames = [rect(mounted(lay.robot), recipes._rectangle(sign), sign.z0, sign.z0 + sign.sz) for sign in lettered(lay)]
            with_lettering.append(any(f is not None and f[0] >= 0 and f[1] <= 640 and f[2] >= 0 and f[3] <= 480 for f in frames))
    assert len(with_lettering) == 32 and sum(with_lettering) >= 30      # the rule is not kept by hanging nothing where it shows

@pytest.mark.parametrize("n, x, h", [(1, 0.7, 4.4), (3, 1.3, 5.04), (4, 1.6, 5.678908), (6, 1.884807, 5.942523)])
def test_waiting_line_robot_looks_at_the_line_from_the_side(n, x, h):
    lay = recipes.recipe_waiting_line(rng(n), n)
    people, r = lay.scene_gt.people, lay.robot
    assert (lay.name, len(lay.figures), [(p.id, p.group) for p in people]) == (f"waiting_line_{n}", n, [(f"p{i}", "g0") for i in range(n)])
    assert np.allclose([(p.x, p.y) for p in people], [(0.8 + 0.6 * i, 0.0) for i in range(n)], atol=0.0501)
    assert all(abs(wrap_angle(p.yaw - math.pi)) <= 0.1501 for p in people)
    assert (r.x, abs(r.y)) == pytest.approx((x, h)) and r.yaw == pytest.approx(-math.copysign(math.pi / 2, r.y))
    near, behind = math.copysign(0.8, r.y), 0.8 + 0.6 * (n - 1) + 1.4       # the place's side towards the robot; past the region's far edge
    whole = [(-0.8, near, 0.0), (-0.8, near, 2.25), (0.0, near, 2.25), (behind, 0.0, 0.0)]
    assert all(in_image(mounted(r), *project(mounted(r), point)[:2]) for point in whole)     # the picture cuts off neither

def test_a_long_line_moves_the_robot_back_until_the_whole_place_is_in_the_picture():
    for n, least, most in ((4, 5.9, 5.9), (5, 5.6, 5.7), (6, 6.2, 6.3)):       # up to four, 5.9 m from the target is enough
        r = recipes.recipe_waiting_line(rng(n), n).robot
        assert least - 1e-6 <= math.hypot(r.x, r.y) <= most + 1e-6, n

@pytest.mark.parametrize("n, radius, bearing_deg", [(2, 0.6, 90.0), (3, 0.6928203, 60.0), (4, 0.8485281, 72.5)])
def test_conversation_group_stands_round_a_centre_with_the_robot_off_every_line_of_two(n, radius, bearing_deg):
    lay = recipes.recipe_conversation_group(rng(n), n)
    (t,), people, r = lay.scene_gt.targets, lay.scene_gt.people, lay.robot
    turn = [math.pi + 2 * math.pi * i / n for i in range(n)]
    assert np.allclose([(p.x, p.y) for p in people], [(radius * (1 + math.cos(a)), radius * math.sin(a)) for a in turn], atol=0.0501)
    assert all(abs(wrap_angle(p.yaw - a - math.pi)) <= 0.1501 and p.group == "g0" for p, a in zip(people, turn))
    assert (lay.name, t.kind, t.person_id, t.anchor) == (f"conversation_group_{n}", "person", "p0", (people[0].x, people[0].y, 1.1))
    assert [f.color == RED for f in lay.figures] == [True] + [False] * (n - 1)
    assert abs(math.degrees(math.atan2(r.y, r.x - radius))) == pytest.approx(bearing_deg, abs=1e-4)
    assert 4.2 <= math.hypot(r.x - radius, r.y) <= 4.8 and r.yaw == pytest.approx(math.atan2(-r.y, radius - r.x), abs=1e-6)

def bearings_apart(robot, p, q):                                # degrees between the directions in which the robot sees two ground points
    return abs(math.degrees(wrap_angle(math.atan2(p[1] - robot.y, p[0] - robot.x) - math.atan2(q[1] - robot.y, q[0] - robot.x))))

def shares_shown(lay):                                          # the picture the generator takes of the layout, and the share of each figure that it shows:
    obs = observed(lay)                                         # 1.0 where nobody and nothing stands before any part of them
    return obs, [gen._share_shown(obs.camera, obs.depth, [figure]) for figure in lay.figures]

def attending(lay, name, kind, label, anchor_z, footprint, requests, situation):      # what the layouts of people watching and of people waiting have in common
    (t,), people, r = lay.scene_gt.targets, lay.scene_gt.people, lay.robot
    assert (lay.name, t.id, t.kind, t.label, t.anchor, t.front_yaw, t.person_id) == (name, "t0", "place", label, (0.0, 0.0, anchor_z), 0.0, None)
    assert sorted(t.footprint) == pytest.approx(sorted(footprint)) and list(lay.footprints) == [t.footprint]
    assert len(lay.boxes) >= 10 and all(within(part, [t.footprint]) for part in lay.boxes)      # built of several parts, each on the target's footprint
    assert [(i.kind, i.source, i.start, i.stop, i.footprint) for i in lay.items[:1]] == [(kind, "boxes", 0, len(lay.boxes), t.footprint)]
    assert [(p.id, p.group) for p in people] == [(f"p{i}", "g0") for i in range(len(people))] and RED not in [f.color for f in lay.figures]
    assert all(abs(wrap_angle(p.yaw - math.atan2(-p.y, -p.x))) <= 0.2 for p in people)         # each faces the target, more or less
    assert [(e.request, e.context, e.ground_truth.situation, e.ground_truth.group_at_target) for e in lay.episodes] == [
        (request, None, situation, "g0") for request in requests]
    view = recipes._View(r)                                     # the picture holds everyone from the floor to above their head
    assert all(view.holds(recipes._body(p), 0.0, 1.85) for p in people)
    cx, cy = np.mean([(p.x, p.y) for p in people], axis=0)      # where the robot stands, as seen from the middle of them: degrees off straight behind, and metres
    return t, people, r, math.degrees(math.atan2(abs(r.y - cy), r.x - cx)), math.hypot(r.x - cx, r.y - cy)

@pytest.mark.parametrize("n, bearing_deg, rows", [(3, 50.0, 1), (4, 52.0, 2), (5, 55.0, 2)])
def test_an_audience_stands_in_rows_before_a_screen_and_the_robot_looks_from_behind_and_to_one_side(n, bearing_deg, rows):
    sides, first_rows = set(), []
    for seed in range(8):
        lay = recipes.recipe_audience(rng(seed), n)
        t, people, r, bearing, distance = attending(lay, f"audience_{n}", "screen", "big screen", 1.5, [(-0.5, -1.0), (0.0, -1.0), (0.0, 1.0), (-0.5, 1.0)],
                                                    ("go to the big screen", "note down the results for me"), "audience")
        assert all(2.0 - 1e-9 <= p.x <= 3.2 + 1e-9 for p in people)             # the rows are 2 to 3.2 m from the screen
        nearest = [min(math.hypot(p.x - q.x, p.y - q.y) for q in people if q is not p) for p in people]
        assert 0.75 <= min(nearest) and max(nearest) <= 1.05                    # a compact group: everyone has a neighbour 0.9 m off, more or less
        depths = sorted(p.x for p in people)
        assert sum(b - a > 0.5 for a, b in zip(depths, depths[1:])) == rows - 1 # one row of three; two rows of four and of five, 0.9 m apart
        assert bearing == pytest.approx(bearing_deg, abs=1.0) and 3.9 <= distance <= 4.5
        assert recipes._View(r).holds([(0.0, -1.0), (0.0, 1.0)], 0.0, 2.1)      # the picture holds the screen's whole face
        assert all(e.ground_truth.rationale is not None and "arm's length or a little more" in e.ground_truth.rationale.placement for e in lay.episodes)
        (region,), (view,) = rules.build_regions("audience", t, people, "g0")   # one region, round the back of the group from one side to the other
        assert view.kind == "audience_view" and min(x for x, _ in region.polygon) >= min(p.x for p in people) - 1e-6     # nowhere nearer the screen than the nearest of them
        sides.add(r.y > 0), first_rows.append(depths[0])
        if seed < 3:                                 # in the picture nobody hides any part of anyone, and the middle of the screen is seen past them all
            obs, shares = shares_shown(lay)
            assert min(shares) >= 0.9 and anchored(obs, t), (seed, shares)
    assert sides == {True, False}
    assert max(first_rows) - min(first_rows) >= (0.1 if rows == 2 else 0.4)     # the first row's distance is drawn: over 0.2 m for two rows, 1.1 m for one

@pytest.mark.parametrize("n, bearing_deg, level, apart", [(2, 45.0, 0.6, 9.0), (4, 30.0, 0.2, 15.0)])
def test_people_wait_at_a_lift_gathered_before_its_doors_and_the_robot_looks_from_behind_and_to_one_side(n, bearing_deg, level, apart):
    sides = set()
    ordinals = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6"}
    for seed in range(8):
        lay = recipes.recipe_lift_wait(rng(seed), n)
        t, people, r, bearing, distance = attending(lay, f"lift_wait_{n}", "lift", "lift", 1.2, [(-1.8, -1.3), (0.0, -1.3), (0.0, 1.3), (-1.8, 1.3)],
                                                    ("go to the lift", "take this up to the fifth floor"), "lift_wait")
        assert 1.2 <= min(p.x for p in people) <= 1.8                           # the nearest of them, from the doors
        cx, cy = np.mean([(p.x, p.y) for p in people], axis=0)
        assert max(math.hypot(p.x - cx, p.y - cy) for p in people) <= 1.3       # a compact cluster
        first, second = sorted(people, key=lambda p: p.x)[:2]                   # the two nearest the doors: in the picture they stand side by side before them,
        assert second.x - first.x <= level                                      # little apart in distance from the doors
        assert bearings_apart(r, (first.x, first.y), (second.x, second.y)) >= apart     # and clearly apart in bearing: a body is 6 degrees wide there
        assert bearing == pytest.approx(bearing_deg, abs=1.0) and 3.9 <= distance <= 4.5
        assert all(e.ground_truth.rationale is None for e in lay.episodes)      # a held-out situation: no rationale, so never a worked example
        doorway = [(0.0, y, z) for y in (-0.63, 0.63) for z in (0.0, 2.4)]      # the frame round the doors, up to the number over them
        assert all(in_image(mounted(r), *project(mounted(r), point)[:2]) for point in doorway)
        (plate,) = [b for b in lay.boxes if isinstance(b, Box) and b.sy == 0.13]      # the call plate: beside the doorway on the robot's side
        side = math.copysign(1.0, r.y)
        assert (plate.cy, plate.z0) == (side * 0.86, 1.0) and lay.items[0].options == {"label": "lift", "plate": side}
        assert max(top_of(part) for part in lay.boxes) == 3.0                   # the shaft reaches the ceiling
        (lit,) = [b.label for b in lay.boxes if isinstance(b, Box) and b.label]       # the number lit over the doors is the floor the lift is at,
        asked = next(digit for word, digit in ordinals.items() if word in lay.episodes[1].request)
        assert lit == "3" and asked == "5" and lit != asked                     # and the errand is to another floor
        (region,), (ahead,) = rules.build_regions("lift_wait", t, people, "g0")
        assert ahead.kind == "ahead_in_line" and min(x for x, _ in region.polygon) >= min(p.x for p in people) - 1e-6     # nowhere nearer the doors than the nearest of them
        sides.add(r.y > 0)
        if seed < 3:                                 # in the picture nobody stands behind another, and the middle of the doors is seen between them
            obs, shares = shares_shown(lay)
            assert min(shares) >= 0.9 and anchored(obs, t), (seed, shares)
    assert sides == {True, False}
    assert recipes._LIFT_PLACES[4][2][0] > 0.2 and recipes._LIFT_PLACES[4][3][0] > 1.0      # the four are not the two with two more: three in an arc, one behind
    assert recipes._LIFT_BEARING_DEG[4] < recipes._LIFT_BEARING_DEG[2] - 10      # and seen from nearer straight behind

@pytest.mark.parametrize("n, who", [(3, "posing"), (2, "taker"), (2, "posing"), (3, "taker")])     # the first two are the rows of the set
def test_a_picture_is_taken_of_a_row_by_someone_behind_a_camera_on_a_tripod(n, who):
    sides, turned = set(), []
    for seed in range(8):
        lay = recipes.recipe_group_photo(rng(seed), n, who)
        (t,), people, r = lay.scene_gt.targets, lay.scene_gt.people, lay.robot
        posing, taker, side = people[:n], people[n], math.copysign(1.0, r.y)
        assert lay.name == f"group_photo_{n}_{who}" and [(p.id, p.group) for p in people] == [(f"p{i}", "g0") for i in range(n + 1)]
        assert np.allclose([(p.x, p.y) for p in posing], [(0.0, 0.6 * (i - (n - 1) / 2)) for i in range(n)], atol=0.0501)     # a row across the origin
        assert all(abs(wrap_angle(p.yaw)) <= 0.1501 for p in posing)            # each facing the camera, more or less
        assert 2.45 <= taker.x <= 3.55 and abs(taker.y) <= 0.05                 # the picture-taker 2.5 to 3.5 m from the row, whoever the target is
        aim = math.atan2(-taker.y, -taker.x)                                    # from where they stand to the middle of the row
        assert abs(wrap_angle(taker.yaw - aim)) <= 0.1501 + 0.02                # they face the row as loosely as anyone faces anything,
        turned.append(abs(wrap_angle(taker.yaw - aim)))
        asked = taker if who == "taker" else posing[n - 1 if side > 0 else 0]   # the one at the row's end on the robot's side
        assert (t.kind, t.label, t.person_id, t.anchor) == ("person", "person in red", asked.id, (asked.x, asked.y, 1.1))
        assert [f.color == RED for f in lay.figures] == [p is asked for p in people] and len({f.color for f in lay.figures}) == n + 1
        assert [(f.x, f.y, f.yaw) for f in lay.figures] == [(p.x, p.y, p.yaw) for p in people]
        (tripod,) = lay.items[:1]                                               # and the camera on its tripod does not turn with them: 0.5 m before them on that line, looking along it
        assert (tripod.kind, tripod.source, tripod.start, tripod.stop) == ("tripod_camera", "boxes", 0, len(lay.boxes)) and len(lay.boxes) == 40
        assert (tripod.x, tripod.y, tripod.yaw) == pytest.approx((taker.x + 0.5 * math.cos(aim), taker.y + 0.5 * math.sin(aim), aim))
        assert list(lay.footprints) == [tripod.footprint] and all(within(part, [tripod.footprint]) for part in lay.boxes)
        assert distance_to_polygon((taker.x, taker.y), tripod.footprint) > 0.22 + 0.1       # clear of the one behind it, whose body reaches 0.22 m
        lens = max((b for b in lay.boxes if isinstance(b, Box)), key=lambda b: math.hypot(b.cx - taker.x, b.cy - taker.y))
        assert lens.panel is not None and 1.3 < lens.z0 < 1.5 and max(top_of(part) for part in lay.boxes) < 1.6      # the lens before the body, below the eye of who stands behind
        assert lens.yaw == pytest.approx(aim)
        assert [(e.request, e.context, e.ground_truth.situation, e.ground_truth.group_at_target, e.ground_truth.rationale) for e in lay.episodes] == [
            ("go to the person in red", None, "group_photo", "g0", None), ("give this to Sam", "Sam is the person wearing red.", "group_photo", "g0", None)]
        if who == "taker":               # the robot behind the row and to one side: 20 degrees off the line from the row to the camera, at the picture-taker's place
            assert -3.0 <= r.x <= -2.6 and math.degrees(abs(math.atan2(r.y, taker.x - r.x))) == pytest.approx(20.0, abs=0.6)
        else:                            # the robot behind the picture-taker and to one side: 27 degrees off that line, at the middle of the row
            assert 1.95 <= r.x - taker.x <= 2.45 and math.degrees(abs(math.atan2(r.y, r.x))) == pytest.approx(27.0, abs=1e-6)
        view = recipes._View(r)                                                 # the picture holds everyone and the tripod
        assert all(view.holds(recipes._body(p), 0.0, 1.85) for p in people) and view.holds(tripod.footprint, 0.0, 1.6)
        (region,), (picture,) = rules.build_regions("group_photo", t, people, "g0")   # the rule finds the one picture-taker, however they have turned: one region,
        assert picture.kind == "photo_frame" and region.heading.face_point == (asked.x, asked.y)      # round them, to be stood in facing the person asked for
        assert max(math.hypot(x - taker.x, y - taker.y) for x, y in region.polygon) == pytest.approx(1.2) and all(x >= taker.x - 0.05 for x, _ in region.polygon)
        sides.add(side)
        if seed < 3:                     # in the picture nobody posing is hidden at all; the camera stands before part of the one who takes the picture
            obs, shares = shares_shown(lay)
            assert min(shares[:n]) >= 0.9 and shares[n] >= 0.5 and anchored(obs, t), (seed, shares)
    assert sides == {1.0, -1.0} and max(turned) > 0.05          # in some draw the picture-taker stands turned from the row: nothing sets their yaw to the aim

COUNTERS = {"help desk", "ticket counter", "coffee stall"}    # the places with a counter or a desk

@pytest.mark.parametrize("n, before", [(4, 2), (6, 3)])
def test_a_bent_line_turns_round_a_pillar_after_two_or_three_people_and_the_robot_sees_both_legs_from_outside_the_bend(n, before):
    sides, places = set(), set()
    step, corner = 0.6 / math.sqrt(2.0), 0.8 + 0.6 * (before - 1)      # the first after the bend stands diagonally on from the last before it
    for seed in range(8):
        lay = recipes.recipe_waiting_line_bent(rng(seed), n)
        (t,), people, r = lay.scene_gt.targets, lay.scene_gt.people, lay.robot
        side = math.copysign(1.0, people[-1].y)
        assert lay.name == f"waiting_line_bent_{n}" and [(p.id, p.group) for p in people] == [(f"p{i}", "g0") for i in range(n)]
        assert (t.id, t.kind, t.anchor, t.front_yaw) == ("t0", "place", (0.0, 0.0, 1.0), 0.0) and t.label in COUNTERS
        grounded, high_level = next(row[4:] for row in recipes._PLACES if row[0] == t.label)
        assert [(e.request, e.context, e.ground_truth.situation, e.ground_truth.group_at_target) for e in lay.episodes] == [
            (grounded, None, "waiting_line", "g0"), (high_level, None, "waiting_line", "g0")]
        assert all("join at the back" in e.ground_truth.rationale.expectations for e in lay.episodes)
        spots = [(0.8 + 0.6 * i, 0.0) for i in range(before)] + [(corner + step, side * (step + 0.6 * j)) for j in range(n - before)]
        assert np.allclose([(p.x, p.y) for p in people], spots, atol=0.0501)     # the first leg out from the place, the second across it
        assert all(0.45 <= math.dist((p.x, p.y), (q.x, q.y)) <= 0.75 for p, q in zip(people, people[1:]))     # spaced as a straight line, round the bend too
        first = math.atan2(people[before - 1].y - people[0].y, people[before - 1].x - people[0].x)
        second = math.atan2(people[-1].y - people[before].y, people[-1].x - people[before].x)
        assert abs(math.degrees(wrap_angle(second - first))) == pytest.approx(90.0, abs=12.0)      # the line turns by about 90 degrees
        away = [math.hypot(p.x, p.y) for p in people]
        assert all(b > a for a, b in zip(away, away[1:]))                      # each further from the place than the one before: the line's order is the rule's
        ahead = [(-1.0, 0.0)] + [(p.x, p.y) for p in people[:-1]]
        assert all(abs(wrap_angle(p.yaw - math.atan2(q[1] - p.y, q[0] - p.x))) <= 0.4 for p, q in zip(people, ahead))    # each faces the one before, the first the place
        (pillar,) = [i for i in lay.items if i.kind == "pillar" and i.source == "boxes"]     # the one the line turns round: no furnishing
        assert (pillar.source, pillar.stop) == ("boxes", len(lay.boxes)) and list(lay.footprints) == [t.footprint, pillar.footprint]
        assert max(top_of(part) for part in lay.boxes[pillar.start:]) == 3.0 and all(within(part, [pillar.footprint]) for part in lay.boxes[pillar.start:])
        px, py = np.mean(pillar.footprint, axis=0)                              # inside the corner, the line round it
        assert px < people[before].x and side * py > side * people[before].y and min(math.hypot(px - p.x, py - p.y) for p in people) >= 0.6
        assert max(math.hypot(px - q.x, py - q.y) for q in people[before - 1:before + 1]) <= 0.8
        (region,), corridor = rules.build_regions("waiting_line", t, people, "g0")      # the region round the last of the second leg, facing along it
        assert region.heading.yaw == pytest.approx(-side * math.pi / 2, abs=0.2) and len(corridor) == n
        assert max(math.hypot(x - people[-1].x, y - people[-1].y) for x, y in region.polygon) == pytest.approx(1.2)
        assert not point_in_polygon((corner + 0.9, 0.0), region.polygon)       # and not where the first leg would end, straight on
        view = recipes._View(r)                     # the picture holds the place, sign and all, everyone from head to foot, and the floor behind the last
        assert view.holds(t.footprint, 0.0, 2.25) and all(view.holds(recipes._body(p), 0.0, 1.85) for p in people)
        assert in_image(mounted(r), *project(mounted(r), (spots[-1][0], spots[-1][1] + side * 1.4, 0.0))[:2])
        bend = (corner + step / 2, side * step / 2)                             # the robot on the bend's outer bisector, 4.3 to 4.7 m out
        assert abs(wrap_angle(math.atan2(r.y - bend[1], r.x - bend[0]) + side * math.pi / 4)) < 1e-9
        assert 4.3 - 1e-9 <= math.hypot(r.x - bend[0], r.y - bend[1]) <= 4.7 + 1e-9
        sides.add(side), places.add(t.label)
        if seed < 3:                                 # in the picture everyone is seen, past the others, and the anchor on the counter
            obs, shares = shares_shown(lay)
            assert min(shares) >= 0.6 and anchored(obs, t), (seed, shares)
    assert sides == {1.0, -1.0} and len(places) >= 2

@pytest.mark.parametrize("machine, label, text, requests", [("cash", "cash machine", "CASH", ("go to the cash machine", "I need some cash")),
                                                            ("ticket", "ticket machine", "TICKETS", ("go to the ticket machine", "buy me a train ticket"))])
def test_someone_uses_a_machine_and_the_robot_looks_from_behind_them_and_to_one_side(machine, label, text, requests):
    sides = set()
    for seed in range(8):
        lay = recipes.recipe_machine_use(rng(seed), machine)
        (t,), (user,), r = lay.scene_gt.targets, lay.scene_gt.people, lay.robot
        deep, wide = recipes._MACHINE_BUILDERS[machine][1]
        assert (lay.name, t.id, t.kind, t.label, t.anchor, t.front_yaw, t.person_id) == (f"machine_use_{machine}", "t0", "place", label, (0.0, 0.0, 0.8), 0.0, None)
        assert sorted(t.footprint) == pytest.approx(sorted([(-deep, -wide / 2), (0.0, -wide / 2), (0.0, wide / 2), (-deep, wide / 2)]))
        assert list(lay.footprints) == [t.footprint] and len(lay.boxes) >= 7 and all(within(part, [t.footprint]) for part in lay.boxes)
        assert [(i.kind, i.source, i.start, i.stop) for i in lay.items[:1]] == [(f"{machine}_machine", "boxes", 0, len(lay.boxes))]
        top = max(top_of(part) for part in lay.boxes)
        (head,) = [b for b in lay.boxes if isinstance(b, Box) and b.label]     # lettered on the head on top of it, and nowhere else
        assert (head.label, head.label_face, head.z0 + head.sz) == (text, "front", top) and top <= 2.25
        faced = [b for b in lay.boxes if isinstance(b, Box) and b.cx + b.sx / 2 == pytest.approx(0.0) and abs(b.cy) < b.sy / 2 and b.z0 <= 0.8 <= b.z0 + b.sz]
        assert len(faced) == 1                       # the anchor lies on the front of one part
        assert (user.id, user.group) == ("p0", "g0") and abs(user.x - 0.5) <= 0.0501 and abs(user.y) <= 0.0501    # 0.5 m before its face,
        assert abs(wrap_angle(user.yaw - math.pi)) <= 0.1501                    # facing it
        assert [(e.request, e.context, e.ground_truth.situation, e.ground_truth.group_at_target, e.ground_truth.rationale) for e in lay.episodes] == [
            (q, None, "machine_use", "g0", None) for q in requests]             # held out: no rationale, so never a worked example
        side = math.copysign(1.0, r.y)               # behind the user and to one side: 50 degrees off straight behind, 4.0 to 4.4 m from them
        assert math.degrees(math.atan2(abs(r.y), r.x - 0.5)) == pytest.approx(50.0) and 4.0 - 1e-9 <= math.hypot(r.x - 0.5, r.y) <= 4.4 + 1e-9
        view = recipes._View(r)                      # the picture holds the machine, head and all, and the user from head to foot
        assert view.holds(t.footprint, 0.0, top) and view.holds(recipes._body(user), 0.0, 1.85)
        (region,), _ = rules.build_regions("machine_use", t, [user], "g0")
        assert not point_in_polygon((r.x, r.y), region.polygon) and in_image(mounted(r), *project(mounted(r), (2.0, 0.0, 0.0))[:2])    # 1.5 m behind the user: in it
        sides.add(side)
        if seed < 3:                                 # in the picture nothing hides the user, and the anchor is seen past them
            obs, shares = shares_shown(lay)
            assert shares == [1.0] and anchored(obs, t), (seed, shares)
    assert sides == {1.0, -1.0}

def test_recipes_refuse_what_they_cannot_frame():
    for call in (lambda: recipes.recipe_no_one(rng(), 0.0, "door"), lambda: recipes.recipe_waiting_line(rng(), 0),
                 lambda: recipes.recipe_waiting_line(rng(), 7), lambda: recipes.recipe_conversation_group(rng(), 5),
                 lambda: recipes.recipe_audience(rng(), 2), lambda: recipes.recipe_audience(rng(), 6),
                 lambda: recipes.recipe_lift_wait(rng(), 3), lambda: recipes.recipe_group_photo(rng(), 4, "posing"),
                 lambda: recipes.recipe_group_photo(rng(), 2, "bystander"), lambda: recipes.recipe_waiting_line_bent(rng(), 5),
                 lambda: recipes.recipe_waiting_line_bent(rng(), 2), lambda: recipes.recipe_machine_use(rng(), "coffee")):
        with pytest.raises(ValueError):
            call()

@pytest.mark.parametrize("kind, label, anchor_z, requests", [("lift", "lift", 1.2, ["go to the lift", "take this up to the fifth floor"]),
                                                             ("screen", "big screen", 1.5, ["go to the big screen", "note down the results for me"])])
def test_the_lift_and_the_screen_also_stand_with_nobody_at_them(kind, label, anchor_z, requests):
    sides = set()
    for seed in range(6):
        lay = recipes.recipe_no_one(rng(seed), 30.0, kind)
        (t,), r = lay.scene_gt.targets, lay.robot
        assert (lay.name, t.id, t.kind, t.label, t.anchor, t.front_yaw) == (f"no_one_{kind}", "t0", "place", label, (0.0, 0.0, anchor_z), 0.0)
        assert lay.scene_gt.people == [] and list(lay.footprints) == [t.footprint] and all(within(part, [t.footprint]) for part in lay.boxes)
        assert [(i.kind, i.start, i.stop, i.footprint) for i in lay.items[:1]] == [(kind, 0, len(lay.boxes), t.footprint)]
        assert [(e.request, e.context, e.ground_truth.situation, e.ground_truth.group_at_target) for e in lay.episodes] == [(q, None, "no_one", None) for q in requests]
        assert all("Nobody is waiting" in e.ground_truth.rationale.people for e in lay.episodes)
        assert 4.5 <= math.hypot(r.x, r.y) <= 5.5 and abs(math.degrees(math.atan2(r.y, r.x))) == pytest.approx(30.0)      # seen as a place is, from 30 degrees off its front
        if kind == "lift":               # the call plate on the robot's side of the doors, as where people wait
            (plate,) = [b for b in lay.boxes if isinstance(b, Box) and b.sy == 0.13]
            assert plate.cy == math.copysign(0.86, r.y) and lay.items[0].options["plate"] == math.copysign(1.0, r.y)
        (region,), forbidden = rules.build_regions("no_one", t, [], None)       # the rectangle before a place: 0.6 to 1.6 m out of its front, 1.2 m wide
        assert forbidden == [] and sorted(region.polygon) == pytest.approx(sorted([(0.6, -0.6), (1.6, -0.6), (1.6, 0.6), (0.6, 0.6)]))
        sides.add(r.y > 0)
    assert sides == {True, False}

def test_transform_layout_moves_everything_rigidly_and_wraps_yaws():
    lay = recipes.recipe_waiting_line(rng(2), 2)
    out = recipes.transform_layout(lay, math.pi / 2, 3.0, -2.0)             # (x, y) -> (3 - y, x - 2)
    assert (out.robot.x, out.robot.y) == pytest.approx((3.0 - lay.robot.y, lay.robot.x - 2.0))
    assert math.cos(out.robot.yaw) == pytest.approx(math.copysign(1.0, lay.robot.y))      # it faced the line, across y
    (t,), box = out.scene_gt.targets, out.boxes[0]              # the counter of the place: 0.6 m deep, behind its front
    assert t.anchor == pytest.approx((3.0, -2.0, 1.0)) and t.front_yaw == pytest.approx(math.pi / 2)
    assert np.allclose(sorted(t.footprint), [(2.2, -2.8), (2.2, -2.0), (3.8, -2.8), (3.8, -2.0)])
    assert np.allclose(out.scene_gt.obstacles[0].polygon, t.footprint) and (box.cx, box.cy, box.yaw) == pytest.approx((3.0, -2.3, math.pi / 2))
    assert np.allclose(out.footprints[0], t.footprint) and len(out.footprints) == 1
    assert np.allclose(out.scene_gt.bounds.polygon, [(3.0 - y, x - 2.0) for x, y in lay.scene_gt.bounds.polygon])
    assert np.allclose(out.room.polygon, out.scene_gt.bounds.polygon) and out.room.wall == lay.room.wall
    assert np.allclose([o.polygon for o in out.scene_gt.obstacles], [[(3.0 - y, x - 2.0) for x, y in o.polygon] for o in lay.scene_gt.obstacles])
    for before, after in ((lay.boxes, out.boxes), (lay.props, out.props)):      # every part, round ones too
        assert len(before) == len(after) > 10 and [type(a) for a in after] == [type(b) for b in before]
        for b, a in zip(before, after):
            x, y = (b.cx, b.cy) if isinstance(b, Box) else (b.x, b.y)
            assert ((a.cx, a.cy) if isinstance(a, Box) else (a.x, a.y)) == pytest.approx((3.0 - y, x - 2.0))
            assert not isinstance(b, Box) or (-math.pi < a.yaw <= math.pi and math.cos(a.yaw - b.yaw - math.pi / 2) == pytest.approx(1.0))
    for a, b, f in zip(lay.scene_gt.people, out.scene_gt.people, out.figures):
        assert (b.x, b.y, b.id, b.group) == (pytest.approx(3.0 - a.y), pytest.approx(a.x - 2.0), a.id, a.group)
        assert -math.pi < b.yaw <= math.pi and abs(wrap_angle(b.yaw + math.pi / 2)) <= 0.1501      # pi + pi/2 wraps
        assert (f.x, f.y, f.yaw) == pytest.approx((b.x, b.y, b.yaw))
    assert (out.name, out.episodes) == (lay.name, lay.episodes)

import sgi.synth.generate as gen
from sgi.check_scene import CheckResult, check_scene, has_errors
from sgi.config import load_config
from sgi.geometry import make_poses
from sgi.gt import regions as rules
from sgi.gt.derive import is_visible, oracle_target, prepare, reference_people_matter
from sgi.scene import Person, load_scene, read_scene_file, save_scene
from sgi.stages.goal import standard_goal
from tests.helpers import INTRINSICS
from tests.test_run import placeholder  # noqa: F401

CFG = load_config()

def mounted(robot):                                             # the generator's camera: 1.5 m up, pitched 10 degrees down
    return Camera(**INTRINSICS, T_world_camera=camera_on_robot(robot, 0.2, 1.5, math.radians(10.0)))

def observed(lay):                                              # the picture the generator takes of a layout
    _, rgb, depth = gen.build_scene(lay, "s")
    return ObservationData(rgb=rgb, depth=depth, camera=mounted(lay.robot), robot=lay.robot)

def test_rejected_layouts_are_removed_retried_and_counted(tmp_path, monkeypatch):
    (tmp_path / "eval" / "stale").mkdir(parents=True)
    checks, goals = [], []

    def fake_check(scene_dir, cfg, write_png=True):             # an error, warning 9 twice, then warning 8 for ever
        checks.append(write_png)
        number, level = {1: (3, "error"), 2: (9, "warning"), 3: (9, "warning")}.get(len(checks), (8, "warning"))
        return [CheckResult(number=number, name="x", level=level)]

    def fake_goal(observation, target, th):                     # the first standard goal lands on the target itself
        goals.append(target)
        world, robot = make_poses(observation.robot, target.world[:2], 0.0)
        real = standard_goal(observation, target, th)
        return real if len(goals) > 1 else real.model_copy(update={"pose_world": world, "pose_robot": robot})

    monkeypatch.setattr(gen, "check_scene", fake_check)
    monkeypatch.setattr(gen, "standard_goal", fake_goal)
    with pytest.raises(RuntimeError, match="check_3=1, check_8=20, standard_goal_unreachable=1"):
        gen.generate(tmp_path, CFG, seed=5)
    assert len(checks) == 23 and not any(checks)                # 3 attempts at scene 0, 20 at scene 1; never a check.png
    assert [p.name for p in (tmp_path / "eval").iterdir()] == ["eval_00_no_one_place"]
    kept = read_scene_file(tmp_path / "eval" / "eval_00_no_one_place")
    assert kept.episodes[0].ground_truth.acceptable_regions[0].representative_point is not None

EXTRAS = [row for row in gen._EVAL if row[3] or row[4]]       # the rows with bystanders or a second group

def shade(obs, p_world, reach=3):                               # "red" or "blue" where most of the pixels round a point are, else None
    u, v, _ = project(obs.camera, p_world)
    r, g, b = obs.rgb[int(v) - reach:int(v) + reach + 1, int(u) - reach:int(u) + reach + 1].reshape(-1, 3).astype(float).T
    return "red" if np.mean((r > 1.8 * g) & (r > 1.8 * b)) > 0.5 else "blue" if np.mean((b > 1.8 * r) & (b > 1.4 * g)) > 0.5 else None

def hidden(people, observation):
    return [p.id for p in people if not is_visible(p, observation, CFG.thresholds)]

@pytest.mark.parametrize("row", EXTRAS, ids=lambda row: f"{row[0]}_{row[1]}")
def test_the_image_shows_every_bystander_and_second_group(row):
    recipe, amount, _, bystanders, second_group = row
    first = 0 if recipe == "no_one" else amount                 # the people of the situation come first
    for seed in range(6):
        lay = gen._layout(rng(seed), row)
        people = lay.scene_gt.people
        assert [(p.id, p.group) for p in people[first:]] == [(f"p{first + i}", "g1") for i in range(2 * second_group)] + [
            (f"p{first + 2 * second_group + i}", None) for i in range(bystanders)]
        assert hidden(people, observed(lay)) == [], seed

def test_clear_view_is_an_open_line_of_sight_inside_the_view_cone():
    lay = recipes.recipe_waiting_line(rng(3), 3)                # people near x = 0.8, 1.4 and 2.0; the robot, 5.04 m to
    side = math.copysign(1.0, lay.robot.y)                      # one side of x = 1.3, looks squarely at their line

    def clear(*spots):                                          # (x, how far beyond the line from the robot)
        return recipes._in_clear_view(lay, [Person(id=f"q{i}", x=x, y=-side * beyond, yaw=0.0) for i, (x, beyond) in enumerate(spots)])

    assert clear((3.6, 3.0))                                    # past the line's far end
    assert not clear((1.4, 3.0))                                # behind the second in line
    assert not clear((-0.35, 3.0))                              # the line of sight passes 0.1 m from a corner of the place
    assert not clear((5.2, 1.0))                                # 33 degrees off the robot's heading, with nothing in the way
    assert not clear((1.4, -2.0))                               # in front of the second in line, who would be hidden
    assert clear((3.2, 3.0), (4.3, 3.0))                        # two newcomers side by side
    assert not clear((3.4, 3.0), (3.678, 4.064))                # and one 1.1 m behind the other
    cross = lay.scene_gt.people[0].x - 0.31                     # a line of sight that passes 0.31 m before the first in line
    served = (cross + (cross - lay.robot.x) * 3.0 / abs(lay.robot.y), 3.0)      # and so clear of the place too: nobody hidden,
    assert not clear(served)                                    # but seen between the place and the first in line, as if served
    strangers = [p.model_copy(update={"group": None}) for p in lay.scene_gt.people]
    lay = dataclasses.replace(lay, scene_gt=lay.scene_gt.model_copy(update={"people": strangers}))
    assert clear(served)                                        # with nobody waiting at the place there is no such spot

def test_extras_are_placed_afresh_when_one_of_them_finds_no_spot(monkeypatch):
    row, calls = ("waiting_line", 3, None, 1, True), []
    add_group, add_bystanders = recipes._add_second_group, recipes._add_bystanders

    def group(layout, rng_):
        calls.append("group")
        return add_group(layout, rng_)

    def bystanders(layout, rng_, n):                            # the first two second groups leave the bystander no spot
        calls.append("bystanders")
        if calls.count("bystanders") <= 2:
            raise recipes._NoFreeSpot("no free spot")
        return add_bystanders(layout, rng_, n)

    monkeypatch.setattr(recipes, "_add_second_group", group)
    monkeypatch.setattr(recipes, "_add_bystanders", bystanders)
    lay = gen._layout(rng(), row)
    assert len(calls) >= 6 and calls == ["group", "bystanders"] * (len(calls) // 2)      # the group is placed again each time
    assert [(p.id, p.group) for p in lay.scene_gt.people] == [(f"p{i}", g) for i, g in enumerate(["g0"] * 3 + ["g1"] * 2 + [None])]
    assert len(lay.figures) == 6
    monkeypatch.setattr(recipes, "_SPOT_MAX_DRAWS", 0)          # now nobody finds a spot, however often they try
    with pytest.raises(RuntimeError, match="waiting_line_3: no free spots"):
        gen._layout(rng(), row)

def test_main_writes_with_the_seed_asked_for_and_prints_the_rejections(monkeypatch, capsys, tmp_path):
    calls = []
    monkeypatch.setattr(gen, "generate", lambda out, cfg, seed: calls.append((out, cfg, seed)) or {"check_3": 2, "check_8": 0})
    assert gen.main(["--out", str(tmp_path), "--seed", "7"]) == 0 and calls == [(tmp_path, CFG, 7)]
    assert gen.main(["--out", str(tmp_path)]) == 0 and calls[1] == (tmp_path, CFG, 0)      # default config, seed 0
    assert capsys.readouterr().out.splitlines()[0] == f"wrote 30 scenes to {tmp_path}; rejected layouts: check_3=2"   # 21 of the first pass, 9 of the second

@pytest.mark.parametrize("row", gen._EVAL_MORE[2:4], ids=lambda row: f"{row[1]}_{row[2]}")
def test_at_a_picture_being_taken_the_plain_stop_fails_clearly_however_the_layout_is_drawn(row, tmp_path):
    assert [r[:3] for r in gen._EVAL_MORE[2:4]] == [("group_photo", 3, "posing"), ("group_photo", 2, "taker")]
    misses = []
    for seed in range(8):                # the accept step does not look at this, and the set shows one draw of each row
        lay = gen._layout(rng(seed), row)
        scene, rgb, depth = gen.build_scene(lay, f"s{seed}")
        save_scene(tmp_path / f"s{seed}", scene, rgb, depth)
        inputs, truth = load_scene(tmp_path / f"s{seed}")
        et = prepare(inputs, truth, CFG)["e0"]
        assert et.problems == () and et.standard_goal_ok == {"t0": False} and reference_people_matter(et, None), seed
        goal = standard_goal(inputs.observation, oracle_target(inputs.observation, truth.targets["t0"], CFG.thresholds), CFG.thresholds)
        misses.append(min(distance_to_polygon(goal.pose_world[:2], r.polygon) for r in et.gt.acceptable_regions))
    assert min(misses) >= 0.8            # the stop is before the person asked for, and the region is behind and beside the picture-taker: a metre off, or two

def test_the_second_pass_is_written_after_the_first_numbered_on_from_it_and_drawn_with_a_generator_of_its_own(monkeypatch, tmp_path):
    written = []
    monkeypatch.setattr(gen, "_write_scene", lambda split_dir, index, row, draw, cfg, counts: written.append((split_dir, index, row, draw)))
    assert gen.generate(tmp_path, CFG, seed=3) == dict.fromkeys(gen._REASONS, 0)
    first = [(tmp_path / split, index, row) for split, rows in gen._SPLITS for index, row in enumerate(rows)]
    more = [(tmp_path / "eval", 14 + k, row) for k, row in enumerate(gen._EVAL_MORE)] + [(tmp_path / "examples", 7, row) for row in gen._EXAMPLES_MORE]
    assert [w[:3] for w in written] == first + more and (len(first), len(more)) == (21, 9)     # every row of the first pass, then every row of the second
    one, other = written[0][3], written[-1][3]
    assert all(w[3] is one for w in written[:21]) and all(w[3] is other for w in written[21:]) and one is not other
    assert one.bit_generator.state == rng(3).bit_generator.state                # nothing was drawn here, so each is as it was made:
    assert other.bit_generator.state == np.random.default_rng([3, 1]).bit_generator.state     # the first from the seed alone, as before there was a second pass
    assert [row[:3] for row in gen._EVAL_MORE] == [("audience", 3, None), ("audience", 5, None), ("group_photo", 3, "posing"), ("group_photo", 2, "taker"),
                                                   ("lift_wait", 2, None), ("lift_wait", 4, None),
                                                   ("no_one", 30, "lift"), ("no_one", 0, "screen")]     # last, the lift and the screen with nobody at them
    assert gen._EXAMPLES_MORE == (("audience", 4, None, 0, False),) and all(row[3:] == (0, False) for row in gen._EVAL_MORE)   # no bystander, no second group
    assert not {row[0] for row in gen._EXAMPLES_MORE} & set(CFG.held_out_situations)     # no example of a held-out situation
    assert {row[0] for row in gen._EVAL_MORE} == {"audience", "no_one", *CFG.held_out_situations} - {"machine_use"}    # written on request only
    assert not {*gen._EVAL, *gen._EXAMPLES} & {*gen._EVAL_MORE, *gen._EXAMPLES_MORE}     # no row of the first pass, which is all that accepted_layouts reads

ON_REQUEST = gen._ON_REQUEST["bend_and_machine"]

def test_the_rows_on_request_are_drawn_with_a_generator_of_their_own_and_never_by_default(monkeypatch, tmp_path):
    assert [row[:3] for row in ON_REQUEST] == [("waiting_line_bent", 4, None), ("waiting_line_bent", 6, None), ("machine_use", 1, "cash"), ("machine_use", 1, "ticket")]
    assert all(row[3:] == (0, False) for row in ON_REQUEST) and list(gen._ON_REQUEST) == ["bend_and_machine"]     # no bystander, no second group
    assert not set(ON_REQUEST) & {*gen._EVAL, *gen._EXAMPLES, *gen._EVAL_MORE, *gen._EXAMPLES_MORE}     # generate() writes none of them
    assert "machine_use" in CFG.held_out_situations and "machine_use" not in {row[0] for rows in (gen._EVAL, gen._EXAMPLES, gen._EVAL_MORE, gen._EXAMPLES_MORE) for row in rows}
    written = []
    monkeypatch.setattr(gen, "_write_scene", lambda split_dir, index, row, draw, cfg, counts: written.append((split_dir, index, row, draw)))
    assert gen.generate_rows(tmp_path / "seed9", CFG, ON_REQUEST, seed=9) == dict.fromkeys(gen._REASONS, 0)
    assert [w[:3] for w in written] == [(tmp_path / "seed9", k, row) for k, row in enumerate(ON_REQUEST)]     # straight into the folder, numbered from 0
    draw = written[0][3]
    assert all(w[3] is draw for w in written) and draw.bit_generator.state == np.random.default_rng([9, 2]).bit_generator.state   # neither pass's generator
    calls = []
    monkeypatch.setattr(gen, "generate_rows", lambda out, cfg, rows, seed: calls.append((out, cfg, rows, seed)) or {"check_3": 1})
    monkeypatch.setattr(gen, "generate", lambda *args: pytest.fail("the two splits are not written with --rows"))
    assert gen.main(["--out", str(tmp_path), "--rows", "bend_and_machine", "--seed", "600"]) == 0 and calls == [(tmp_path, CFG, ON_REQUEST, 600)]
    with pytest.raises(SystemExit):
        gen.main(["--out", str(tmp_path), "--rows", "parade"])

def test_the_rows_on_request_pass_the_scene_check_with_the_regions_of_their_rules(tmp_path):
    out = tmp_path / "seed5"
    (out / "seed5_00_stale").mkdir(parents=True)                # a scene of an earlier run under the same prefix goes; the rest stays
    (out / "seed5_00_stale" / "scene.json").write_text("{}")
    (out / "notes.txt").write_text("kept")
    (out / "seed5_keep").mkdir()
    counts = gen.generate_rows(out, CFG, ON_REQUEST, seed=5)
    assert set(counts) == set(gen._REASONS) and sum(counts.values()) <= 2
    names = ["seed5_00_waiting_line_bent_4", "seed5_01_waiting_line_bent_6", "seed5_02_machine_use_cash", "seed5_03_machine_use_ticket"]
    assert sorted(p.name for p in out.iterdir()) == sorted(names + ["notes.txt", "seed5_keep"])
    for name in names:
        results = check_scene(out / name, CFG, write_png=False)
        assert len(results) == 12 and not has_errors(results) and all(r.level == "ok" or r.number == 9 for r in results), name
        inputs, truth = load_scene(out / name)
        assert inputs.scene_id == name and len(inputs.tour) == 8 and hidden(truth.people, inputs.observation) == [], name
        situation = "machine_use" if "machine" in name else "waiting_line"
        ets = prepare(inputs, truth, CFG)
        for episode_id, et in ets.items():
            ep = truth.episodes[episode_id]
            assert (ep.situation, truth.group_for(episode_id, "t0"), et.problems) == (situation, "g0", ()), name
            assert et.standard_goal_ok == {"t0": False} and reference_people_matter(et, None), name      # the plain stop fails: the people matter
            regions, forbidden = rules.build_regions(situation, truth.targets["t0"], truth.people, "g0")
            assert [r.polygon for r in ep.acceptable_regions] == [r.polygon for r in regions] and all(r.representative_point for r in ep.acceptable_regions), name
            kinds = ["over_shoulder", "beside_user"] if situation == "machine_use" else ["ahead_in_line"] * len(truth.people)
            assert [f.kind for f in ep.forbidden_regions] == kinds, name
        assert (truth.episodes["e0"].rationale is None) == (situation == "machine_use"), name

@pytest.fixture(scope="module")
def data(tmp_path_factory, placeholder):                        # a full run of its own, beside the one the run tests share
    a, b, returned = tmp_path_factory.mktemp("a"), placeholder, []
    with pytest.MonkeyPatch.context() as patch:                 # main prints the counts that generate returns: keep them
        patch.setattr(gen, "generate", lambda *args, real=gen.generate: returned.append(real(*args)) or returned[0])
        assert gen.main(["--out", str(a)]) == 0                 # default config, seed 0
    counts = returned[0]
    rows = []
    for folder in sorted(a.glob("*/*")):
        inputs, truth = load_scene(folder)
        rows.append((folder, inputs, truth, prepare(inputs, truth, CFG)["e0"]))
    return a, b, counts, rows

def test_generate_is_deterministic_and_every_scene_passes_the_scene_check(data):
    a, b, counts, rows = data
    names = [str(folder.relative_to(a)) for folder, *_ in rows]
    assert names == sorted(str(p.relative_to(b)) for p in b.glob("*/*"))
    warned = set()
    for n in names:
        assert (a / n / "scene.json").read_text() == (b / n / "scene.json").read_text(), n
        assert np.array_equal(np.load(a / n / "depth.npy"), np.load(b / n / "depth.npy")), n
        results = check_scene(a / n, CFG, write_png=False)
        assert len(results) == 12 and not has_errors(results) and all(r.level == "ok" or r.number == 9 for r in results), n
        warned |= {n for r in results if r.level == "warning"}
    assert warned == set()                                      # not even the two seen 60 degrees off the front: the stop is before it
    assert set(counts) == {f"check_{k}" for k in range(1, 13)} | {"standard_goal_unreachable"}
    assert all(type(v) is int and v >= 0 for v in counts.values())

def test_scenes_cover_the_spec(data):
    rows, poses = data[3], set()
    first = ("no_one_place no_one_place no_one_place no_one_deep_place no_one_object no_one_person waiting_line_1 waiting_line_2 "
             "waiting_line_4 waiting_line_6 conversation_group_2 conversation_group_3 conversation_group_4 waiting_line_3 "
             "audience_3 audience_5 group_photo_3_posing group_photo_2_taker lift_wait_2 lift_wait_4 "
             "no_one_lift no_one_screen").split()                 # the last eight: the second pass
    second = ("no_one_place no_one_place waiting_line_2 waiting_line_3 conversation_group_2 conversation_group_3 "
              "no_one_person audience_4").split()
    assert [folder.name for folder, *_ in rows] == (
        [f"eval_{i:02d}_{name}" for i, name in enumerate(first)] + [f"examples_{i:02d}_{name}" for i, name in enumerate(second)])
    # From the target to the robot. In the second pass the robot looks past the people at what they are at, from 4 m or
    # more off their middle: a screen is 5.3 to 7 m away, a lift 5 to 6.3 m. Whoever the robot came for at a picture is
    # at the far end of its view: one posing past the picture-taker, the picture-taker past the row, 5 to 7 m off.
    reach = {"eval_09_waiting_line_6": (4.0, 6.3), "audience": (5.3, 7.0), "lift_wait": (5.0, 6.3), "group_photo": (5.0, 7.0)}
    #                                                behind six people the robot stands back for the whole place
    for folder, inputs, truth, et in rows:
        obs, target, e0, e1 = inputs.observation, truth.targets["t0"], truth.episodes["e0"], truth.episodes["e1"]
        assert inputs.scene_id == folder.name and obs.rgb.shape == (480, 640, 3)
        near, far = next((span for name, span in reach.items() if name in folder.name), (4.0, 6.0))
        assert near <= math.hypot(obs.robot.x - target.anchor[0], obs.robot.y - target.anchor[1]) <= far, folder.name
        assert np.allclose(obs.camera.T_world_camera, mounted(obs.robot).T_world_camera) and obs.camera.fx == INTRINSICS["fx"]
        poses.add((obs.robot.x, obs.robot.y, obs.robot.yaw))
        assert (e0.request_type, e1.request_type, e1.situation) == ("grounded", "high_level", e0.situation) and e0.situation in folder.name
        assert [(s.type, s.target_ids, s.optional) for s in e0.reference_plan + e1.reference_plan] == [
            ("navigate", ["t0"], False)] * 2 + [("act", [], False), ("navigate", [], True), ("act", [], True)]
        for ep in (e0, e1):                                     # the filled regions are in scene.json
            assert ep.acceptable_regions and all(r.representative_point and r.target_id == "t0" for r in ep.acceptable_regions)
        if target.kind == "person":
            assert target.label == "person in red" and "red" in inputs.episodes[1].context
            assert shade(obs, target.anchor) == "red", folder.name            # the picture shows the colour the context names
        if target.kind == "object":
            assert "blue box on the table" in inputs.episodes[1].context and shade(obs, target.anchor, 8) == "blue"   # tape crosses its middle
            assert (target.footprint, target.match_radius) == (None, 0.3)
    assert len(poses) == 30 and (0.0, 0.0, 0.0) not in poses    # every scene has its own pose, never the identity
    places = [truth.targets["t0"] for _, _, truth, _ in rows if truth.targets["t0"].kind == "place"]
    fronts = [t.front_yaw for t in places]                      # a place's anchor and front are its scene's random transform
    assert all(max(abs(t.anchor[0]), abs(t.anchor[1])) <= 5.0 for t in places) and max(fronts) - min(fronts) > math.pi / 2
    every = [(truth, et) for folder, _, truth, et in rows if folder.parent.name == "eval"]
    ev, more = every[:14], every[14:]                           # the first pass, and the eight scenes of the second
    assert [et.derived_factors.group_size for _, et in ev] == [0, 0, 0, 0, 0, 0, 1, 2, 4, 6, 2, 3, 4, 3]
    assert [truth.episodes["e0"].factors.density for truth, _ in ev] == ["low"] * 8 + ["medium", "high", "medium", "medium", "high", "medium"]
    assert [sum(p.group == "g1" for p in truth.people) for truth, _ in ev] == [0] * 11 + [2, 2, 0]
    loose = [sum(p.group is None and p.id != truth.targets["t0"].person_id for p in truth.people) for truth, _ in ev]
    assert loose == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0]   # bystanders
    lines = [(truth, et) for truth, et in ev if truth.episodes["e0"].situation == "waiting_line"]
    # Nobody but the line stands in a waiting-line picture: seen from the side, anyone past its far end reads as part of it.
    assert len(lines) == 5 and all(len(truth.people) == et.derived_factors.group_size for truth, et in lines)
    assert [truth.targets["t0"].kind for truth, _ in ev] == ["place"] * 4 + ["object", "person"] + ["place"] * 4 + ["person"] * 3 + ["place"]
    assert len({truth.targets["t0"].label for truth, _ in ev} & PLACES) >= 2
    angles = [et.derived_factors.view_angle_deg for _, et in ev[:4]]     # asked for 0, 30, 60 and 30 degrees
    assert angles[0] <= 3.01 and 26.99 <= angles[1] <= 30.0 and 56.99 <= angles[2] <= 60.0 and 26.99 <= angles[3] <= 30.0
    assert [et.standard_goal_ok["t0"] for _, et in ev[:6]] == [True] * 6       # at 60 degrees too: the stop is before the front
    # The second pass: everyone in the picture is in the group at the target, and the picture-taker counts among its members.
    # Then the lift and the screen once more with nobody at them, where the plain stop before them is right.
    assert [truth.episodes["e0"].situation for truth, _ in more] == ["audience"] * 2 + ["group_photo"] * 2 + ["lift_wait"] * 2 + ["no_one"] * 2
    assert [et.derived_factors.group_size for _, et in more] == [len(truth.people) for truth, _ in more] == [3, 5, 4, 3, 2, 4, 0, 0]
    assert [truth.episodes["e0"].factors.density for truth, _ in more] == ["medium"] * 4 + ["low", "medium", "low", "low"]
    assert [truth.targets["t0"].kind for truth, _ in more] == ["place"] * 2 + ["person"] * 2 + ["place"] * 4
    assert [truth.targets["t0"].label for truth, _ in more] == ["big screen"] * 2 + ["person in red"] * 2 + ["lift"] * 3 + ["big screen"]
    assert all(et.derived_factors.bystanders == et.derived_factors.other_groups == 0 for _, et in more)
    assert [et.standard_goal_ok["t0"] for _, et in more] == [False] * 6 + [True] * 2          # the people matter wherever there are any
    lift, screen = (et for _, et in more[6:])
    assert 26.99 <= lift.derived_factors.view_angle_deg <= 30.0 and screen.derived_factors.view_angle_deg <= 3.01     # asked for 30 and 0 degrees off the front
    assert [(len(et.gt.acceptable_regions), et.gt.forbidden_regions) for et in (lift, screen)] == [(1, [])] * 2       # the rectangle before a place with nobody at it
    examples = [(truth, et) for folder, _, truth, et in rows if folder.parent.name == "examples"]
    assert [reference_people_matter(et, None) for _, et in examples] == [False, False, True, True, True, True, True, True]
    assert all(ep.rationale is not None for truth, _ in examples for ep in truth.episodes.values())
    assert [truth.episodes["e0"].situation for truth, _ in examples][7:] == ["audience"]      # the one example of the second pass: of a development situation
    _, inputs, truth, _ = rows[4]                               # the object: the crate makes the standard goal slide
    grounded = oracle_target(inputs.observation, truth.targets["t0"], CFG.thresholds)
    goal = standard_goal(inputs.observation, grounded, CFG.thresholds)
    slid = math.hypot(goal.pose_world[0] - grounded.world[0], goal.pose_world[1] - grounded.world[1])
    assert 1.1 <= slid <= 1.4 and not goal.flags.standard_goal_blocked

def test_the_last_example_is_a_person_alone_seen_in_profile_where_the_plain_answer_fails_for_no_social_reason(data):
    a, _, _, rows = data
    assert gen._EXAMPLES[-1] == ("no_one", 90, "person", 0, False) and len(gen._EXAMPLES) == 7
    folder, inputs, truth, et = rows[-2]                        # the last example of the first pass; the second pass adds one after it
    assert folder == a / "examples" / "examples_06_no_one_person" and inputs.scene_id == "examples_06_no_one_person"
    target = truth.targets["t0"]
    (stands,) = [p for p in truth.people if p.id == target.person_id]
    assert target.kind == "person" and len(truth.people) == 1 and stands.group is None
    assert 80.0 <= et.derived_factors.view_angle_deg <= 100.0          # seen in profile (the person's yaw is jittered)
    results = check_scene(folder, CFG, write_png=False)
    assert not has_errors(results) and len(results) == 12
    assert et.standard_goal_ok["t0"] is False and reference_people_matter(et, None)      # nothing to stand before
    assert truth.episodes["e0"].situation == "no_one"
    # The earlier rows are the same draws as before: the new row is the last of the one random generator.
    assert [f.name for f in sorted((a / "examples").iterdir())][:6] == [
        f"examples_{i:02d}_{name}" for i, name in enumerate(
            "no_one_place no_one_place waiting_line_2 waiting_line_3 conversation_group_2 conversation_group_3".split())]

def test_in_every_scene_of_the_second_pass_that_has_people_they_matter_and_the_regions_are_those_of_its_rule(data):
    rows = {folder.name: (inputs, truth, et) for folder, inputs, truth, et in data[3]}
    kinds = {"audience": "audience_view", "group_photo": "photo_frame", "lift_wait": "ahead_in_line"}
    names = ("eval_14_audience_3 eval_15_audience_5 eval_16_group_photo_3_posing eval_17_group_photo_2_taker eval_18_lift_wait_2 "
             "eval_19_lift_wait_4 examples_07_audience_4").split()        # seven of its nine: the lift and the screen with nobody at them are in test_scenes_cover_the_spec
    pieces = []
    for name in names:
        inputs, truth, et = rows[name]
        target, e0, e1 = truth.targets["t0"], truth.episodes["e0"], truth.episodes["e1"]
        assert et.problems == () and et.standard_goal_ok == {"t0": False} and reference_people_matter(et, None), name   # the plain stop fails: among the people, or in the picture
        assert truth.group_for("e0", "t0") == "g0" and {p.group for p in truth.people} == {"g0"} and et.hidden_relevant_people == (), name
        assert (e0.rationale is None, e1.rationale is None) == (e0.situation in CFG.held_out_situations,) * 2, name   # a held-out situation has none
        regions, forbidden = rules.build_regions(e0.situation, target, truth.people, "g0")      # scene.json has what the rule gives, with a point in each region
        for episode in (e0, e1):
            assert [r.polygon for r in episode.acceptable_regions] == [r.polygon for r in regions], name
            assert [r.heading for r in episode.acceptable_regions] == [r.heading for r in regions], name
            assert [(f.kind, f.polygon) for f in episode.forbidden_regions] == [(kinds[e0.situation], forbidden[0].polygon)], name
        faced = (truth.person(target.person_id).x, truth.person(target.person_id).y) if target.kind == "person" else target.anchor[:2]
        assert {r.heading.face_point for r in regions} == {faced}, name            # facing what is watched or waited at, or the person asked for
        pieces.append(len(regions))
        if target.kind == "place":       # watched or waited at: the point taken as the good standing place is no nearer the thing than the nearest of the people
            out = lambda x, y: (x - target.anchor[0]) * math.cos(target.front_yaw) + (y - target.anchor[1]) * math.sin(target.front_yaw)
            point = e0.acceptable_regions[0].representative_point
            assert out(*point) >= min(out(p.x, p.y) for p in truth.people) - 1e-6, name       # for the example this keeps what it shows and its rationale together
            assert min(math.dist(point, (p.x, p.y)) for p in truth.people) >= 0.5 - 1e-6, name
    assert pieces == [1] * 7             # one region in every one of them: round the back of those attending, or round the picture-taker
    assert "behind or beside" in rows["examples_07_audience_4"][1].episodes["e0"].rationale.placement

def test_a_digest_of_the_first_passs_scene_files_is_what_it_was_before_there_was_a_second_pass(data):
    # The 21 scene.json files of the first pass, as the generator wrote them at 140f475, before the recipes of the second
    # pass were added to the module they share. A change of this digest means an old scene has changed: its people, its
    # robot, its regions or its tour. The pictures are compared file by file when the set is regenerated for a check.
    import hashlib
    a, digest = data[0], hashlib.sha256()
    names = [f"{split}_{index:02d}" for split, rows in gen._SPLITS for index in range(len(rows))]
    files = sorted(f for f in a.glob("*/*/scene.json") if f.parent.name[:len(f.parent.parent.name) + 3] in names)
    for f in files:
        digest.update(f.parent.name.encode() + b"\n" + f.read_bytes() + b"\n")
    assert len(files) == 21 and digest.hexdigest() == "4cbad923f96227d5ba9caf940884a63cb20b118284c023a40bf96bca6f9bd108"

def test_a_digest_of_the_second_passs_scene_files_is_what_it_was_before_the_rows_on_request(data):
    # The 9 scene.json files of the second pass, as the generator wrote them at d50a7b3, before the bent lines and the
    # machines were added to the module they share and the machine rule to sgi.gt. With the digest of the first pass
    # above, this pins every scene of the set: writing the rows on request changes none of them.
    import hashlib
    a, digest = data[0], hashlib.sha256()
    names = [f"{split}_{index:02d}" for split, rows in gen._MORE_SPLITS
             for index in range(len(dict(gen._SPLITS)[split]), len(dict(gen._SPLITS)[split]) + len(rows))]
    files = sorted(f for f in a.glob("*/*/scene.json") if f.parent.name[:len(f.parent.parent.name) + 3] in names)
    for f in files:
        digest.update(f.parent.name.encode() + b"\n" + f.read_bytes() + b"\n")
    assert len(files) == 9 and digest.hexdigest() == "373056b886ed4a5a25a6608495939b31cfa755ac54d49b319a0058b9e4baa44e"

def test_the_image_shows_every_person_of_every_scene(data):    # no factor counts someone the image does not show
    for folder, inputs, truth, et in data[3]:
        obs = inputs.observation
        assert hidden(truth.people, obs) == [], folder.name
        for person in map(truth.person, et.relevant_people):    # from the floor at their feet to above the tallest head
            assert all(in_image(obs.camera, *project(obs.camera, (person.x, person.y, z))[:2]) for z in (0.0, 1.85)), folder.name
        floor = np.abs(lift_all(obs.depth, obs.camera)[:, 2]) < 0.02
        assert (obs.depth > 0).all() and 0.3 < floor.mean() < 0.55, folder.name     # a closed room, and not mostly floor

def test_every_scene_has_a_deterministic_tour_that_shows_its_target(data):
    a, b, _, rows = data
    shown = []
    for folder, inputs, truth, _ in rows:
        names = sorted(p.name for p in (folder / "tour").iterdir())
        assert names == [f"{k:03d}.png" for k in range(8)], folder.name
        twin = b / folder.relative_to(a) / "tour"
        assert all((folder / "tour" / n).read_bytes() == (twin / n).read_bytes() for n in names), folder.name
        assert [f.index for f in inputs.tour] == list(range(8)) and np.array_equal(inputs.tour[3].rgb, inputs.observation.rgb)
        assert np.array_equal(inputs.tour[3].T_world_camera, inputs.observation.camera.T_world_camera)
        frames = truth.targets["t0"].tour_frames
        assert 3 in frames and frames == sorted(set(frames)) and set(frames) <= set(range(8)), folder.name
        assert len(frames) >= 3, folder.name                    # the walk shows the target more than once or twice
        shown.append(len(frames))
        spots = [tuple(f.T_world_camera[:2, 3]) for f in inputs.tour]
        assert len({(round(x, 2), round(y, 2)) for x, y in spots}) == 8, folder.name      # eight different standpoints
        for f, xy in zip(inputs.tour, spots):                   # in the room, clear of all in it, on the generator's mount
            T, where = f.T_world_camera, f"{folder.name} frame {f.index}"
            assert point_in_polygon(xy, truth.bounds) and distance_to_boundary(xy, truth.bounds) >= 0.8 - 1e-6, where
            assert all(distance_to_polygon(xy, obstacle) >= 0.8 - 1e-6 for obstacle in truth.obstacles), where
            assert all(math.hypot(xy[0] - q.x, xy[1] - q.y) >= 1.05 - 1e-6 for q in truth.people), where
            assert T[2, 3] == pytest.approx(1.5) and T[2, 2] == pytest.approx(-math.sin(math.radians(10.0))) and abs(T[2, 0]) < 1e-9, where
    assert min(shown) < 8 and len(shown) == 30                  # and somewhere a frame does not show it

def tour_of(lay):                                               # each tour frame of a layout: what it shows, and whether it lists the target
    scene, _, _ = gen.build_scene(lay, "s")
    taken, real = [], gen._render
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(gen, "_render", lambda layout, camera: taken.append((camera, (out := real(layout, camera))[1])) or out)
        toured, _ = gen._build_tour(lay, scene, CFG.thresholds)
    listed = toured.ground_truth.targets[0].tour_frames
    return [(SimpleNamespace(camera=camera, depth=depth), k in listed) for k, (camera, depth) in enumerate(taken)]

def anchored(obs, target):                                      # scene check 5 on a frame: the anchor in view, on the surface the depth shows
    uvz = project(obs.camera, target.anchor)
    return uvz is not None and in_image(obs.camera, uvz[0], uvz[1]) and abs(float(obs.depth[int(uvz[1]), int(uvz[0])]) - uvz[2]) <= 0.3

def name_reads(obs, board):                                     # the lettered side of a sign board is whole in the frame, within 55 degrees of square
    turn = {"front": 0.0, "left": math.pi / 2, "right": -math.pi / 2}[board.label_face]
    face = dataclasses.replace(board, yaw=board.yaw + turn, **({"sx": board.sy, "sy": board.sx} if turn else {}))
    to_camera = np.subtract(obs.camera.T_world_camera[:2, 3], (face.cx, face.cy))
    square = to_camera @ (math.cos(face.yaw), math.sin(face.yaw)) >= math.cos(math.radians(55.0)) * np.hypot(*to_camera)
    return bool(square and all(shown(obs, point) for point in face_points(face)))

def test_a_tour_frame_lists_a_place_whose_name_it_shows_though_the_line_or_the_place_itself_hides_the_anchor():
    down_the_line = from_the_side = 0
    for row, seed in ((("waiting_line", 3, None, 0, False), 0), (("waiting_line", 3, None, 0, False), 1),
                      (("waiting_line", 4, None, 0, False), 2), (("no_one", 60, "place", 0, False), 0)):
        lay = gen._layout(rng(seed), row)
        (target,) = lay.scene_gt.targets
        names = [p for p in lay.boxes if isinstance(p, Box) and p.label and p.z0 == 1.9]      # the boards of its sign: front, left, right
        assert len(names) == 3
        for k, (obs, listed) in enumerate(tour_of(lay)):
            read = [board.label_face for board in names if name_reads(obs, board)]
            assert listed or not (read or anchored(obs, target)), (row, seed, k)
            if read and not anchored(obs, target):
                down_the_line += read == ["front"]              # looking along the line, whose people stand before the counter
                from_the_side += read != ["front"]              # looking along the counter, whose front is then edge-on
    assert down_the_line >= 4 and from_the_side >= 3

def test_a_tour_frame_lists_a_person_when_it_shows_half_of_them_and_not_when_it_shows_less():
    by_half, too_little = 0, 0
    for n, seed in ((3, 0), (3, 1), (4, 0)):
        lay = gen._layout(rng(seed), ("conversation_group", n, None, 0, False))
        (target,), body = lay.scene_gt.targets, lay.figures[0]  # the person in red, hidden by the others from some of the walk
        for k, (obs, listed) in enumerate(tour_of(lay)):
            ends = [project(obs.camera, (body.x + i * body.radius, body.y + j * body.radius, z)) for i in (-1, 1) for j in (-1, 1) for z in (0.0, body.height)]
            if not all(uvz is not None and in_image(obs.camera, uvz[0], uvz[1]) for uvz in ends):
                continue                                        # the image cuts them off: not a case for this count
            alone, nothing = render(obs.camera, figures=[body], antialias=False)[1], render(obs.camera, antialias=False)[1]
            theirs = alone != nothing                           # the pixels they would fill with nobody else there
            share = np.mean(obs.depth[theirs] == alone[theirs])
            if abs(share - 0.5) > 0.02:
                assert listed == (anchored(obs, target) or share > 0.5), (n, seed, k, share)
            by_half += bool(not anchored(obs, target) and share > 0.52)
            too_little += bool(not anchored(obs, target) and share < 0.48)
    assert by_half >= 3 and too_little >= 2

def test_leaving_out_what_is_out_of_view_changes_no_picture():
    for row in (("waiting_line", 4, None, 2, False), ("no_one", 30, "deep_place", 0, False)):
        lay = gen._layout(rng(1), row)
        scene, rgb, depth = gen.build_scene(lay, "s")
        full = render(mounted(lay.robot), lay.boxes, lay.figures, room=lay.room, props=lay.props)
        assert np.array_equal(rgb, full[0]) and np.array_equal(depth, full[1])
        kept = sum(gen._in_view(mounted(lay.robot), thing) for thing in (*lay.boxes, *lay.props))
        assert 10 < kept < len(lay.boxes) + len(lay.props) - 5 and all(gen._in_view(mounted(lay.robot), f) for f in lay.figures)
        behind = mounted(dataclasses.replace(lay.robot, yaw=wrap_angle(lay.robot.yaw + math.pi)))     # looking away from the scene
        assert not any(gen._in_view(behind, thing) for thing in (*lay.boxes, *lay.figures))
        rgb, depth = gen._render(lay, behind)
        full = render(behind, lay.boxes, lay.figures, room=lay.room, props=lay.props)
        assert np.array_equal(rgb, full[0]) and np.array_equal(depth, full[1])


def test_the_page_states_the_plain_stop_as_the_code_places_it():
    from pathlib import Path
    page = (Path(__file__).resolve().parents[1] / "docs" / "scene_format.md").read_text(encoding="utf-8")
    assert "It fails where there is no such surface." not in page
    assert "within half a metre of that stop, there is a place clear of everything standing" in page
    assert "no plane, nothing clear there, or floor out of view" in page
    assert "up to 0.5 m past what is in the way" in page
    assert "seen well off it (from about 45 degrees)" in page
