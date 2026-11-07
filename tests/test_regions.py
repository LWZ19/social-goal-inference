import math

import pytest

from sgi.evaluate import _heading_within
from sgi.geometry import bearing
from sgi.gt import regions as rules
from sgi.polygons import convex_hull, point_in_polygon
from sgi.scene import Person, Target
from tests.helpers import scene_dict

def _desk(**changes): return Target(**scene_dict()["ground_truth"]["targets"][0]).model_copy(update=changes)   # t0: anchor (1.0, 6.0, 1.0), front_yaw -pi/2
def _person_target(p): return Target(id="t1", kind="person", label="person in red", anchor=(p.x, p.y, 1.3), person_id=p.id)
def _corners(polygon): return sorted((round(x, 6), round(y, 6)) for x, y in polygon)     # sorted: the vertex order does not matter
def _in(regions, x, y): return any(point_in_polygon((x, y), r.polygon) for r in regions)
def _out(origin, deg, r): return (origin[0] + r * math.cos(math.radians(deg)), origin[1] + r * math.sin(math.radians(deg)))   # r from origin on the bearing deg
EDGES = [False, True, True, False]       # a point just short of a region's near edge, two just inside its edges, one just past its far edge

def test_no_one_rectangle_in_front_of_a_place():
    (r,), forbidden = rules.region_no_one(_desk(), [])
    assert (forbidden, r.target_id, r.heading.face_point, r.heading.yaw, r.tolerance_deg) == ([], "t0", (1.0, 6.0), None, 45.0)
    assert _corners(r.polygon) == [(0.4, 4.4), (0.4, 5.4), (1.6, 4.4), (1.6, 5.4)]
    assert [_in([r], 1.0, y) for y in (5.5, 5.0, 4.3)] == [False, True, False]      # 0.5, 1.0 and 1.7 m in front
    assert _in([r], 1.5, 5.0) and not _in([r], 1.7, 5.0)                            # 0.5 and 0.7 m to the side

def test_no_one_ring_when_the_target_has_no_front():
    (r,), forbidden = rules.region_no_one(_desk(front_yaw=None), [])
    assert forbidden == [] and r.heading.face_point == (1.0, 6.0)
    for deg in range(17, 360, 45):
        ux, uy = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        for d in (0.3, 0.5, 0.7, 1.0, 1.5, 1.7, 2.0):                               # the ring spans 0.6 to 1.6 m
            assert _in([r], 1.0 + d * ux, 6.0 + d * uy) is (0.6 < d < 1.6), (deg, d)

def test_no_one_person_target_runs_from_0_5_to_1_2_m_along_the_person_yaw():
    people = [Person(id="p9", x=0.0, y=0.0, yaw=0.3), Person(id="p0", x=3.0, y=2.0, yaw=math.pi)]
    (r,), forbidden = rules.region_no_one(_person_target(people[1]), people)
    assert (forbidden, _corners(r.polygon), r.heading.face_point) == ([], [(1.8, 1.4), (1.8, 2.6), (2.5, 1.4), (2.5, 2.6)], (3.0, 2.0))
    assert [_in([r], 3.0 - d, 2.0) for d in (0.45, 0.55, 1.15, 1.25)] == EDGES      # in front of the person: the paper's 0.5 to 1.2 m
    assert [_in([r], 2.2, 2.0 + side) for side in (-0.65, -0.55, 0.55, 0.65)] == EDGES      # 1.2 m wide, as a place's is
    (turned,), _ = rules.region_no_one(_person_target(people[1]).model_copy(update={"front_yaw": 0.0}), people)
    assert _corners(turned.polygon) == [(3.5, 1.4), (3.5, 2.6), (4.2, 1.4), (4.2, 2.6)]     # a front_yaw of 0.0 is a front, not a missing one
    with pytest.raises(ValueError):                  # the person is not in the list: an error, not a fall-back to the ring
        rules.region_no_one(_person_target(people[1]), people[:1])

def _line(*ys): return [Person(id=f"q{i}", x=1.0, y=y, yaw=math.pi / 2, group="g0") for i, y in enumerate(ys)]   # queueing towards the desk

def test_waiting_line_three_people_then_a_fourth_off_the_axis():
    desk = _desk(anchor=(1.6, 6.0, 1.0))             # 0.6 m off the line's axis: the direction must not come from the anchor
    (r,), forbidden = rules.region_waiting_line(desk, _line(4.5, 3.8, 5.2))          # not given in line order
    last = (1.0, 3.8)                                # the line runs along -y, so "straight behind" is the bearing -90
    assert (r.target_id, r.heading.face_point, r.tolerance_deg) == ("t0", None, 45.0)
    assert r.heading.yaw == pytest.approx(math.pi / 2)                               # the way the line faces, not "at the last person"
    for deg in (-90, -45, -135, -1, -179):           # straight behind, diagonally behind on either side, and almost level
        assert [_in([r], *_out(last, deg, d)) for d in (0.45, 0.55, 1.15, 1.25)] == EDGES, deg      # the ring of 0.5 to 1.2 m
    assert [_in([r], x, y) for x in (0.15, 1.85) for y in (3.79, 3.81)] == [True, False] * 2      # 0.85 m to either side: 1 cm behind, 1 cm ahead
    assert not _in([r], 1.0, 4.65) and not _in([r], 1.6, 4.4)                        # 0.85 m ahead of the last person, straight and diagonally
    straight, diagonal = _out(last, -90, 0.85), _out(last, -30, 0.85)                # the second is 60 degrees off the line's axis
    assert _heading_within(r, straight, math.pi / 2) and _heading_within(r, diagonal, math.pi / 2)
    assert _heading_within(r, diagonal, math.radians(134.0)) and not _heading_within(r, diagonal, math.radians(136.0))
    assert _heading_within(r, straight, bearing(straight, last)) and not _heading_within(r, diagonal, bearing(diagonal, last))   # at the last person: 60 degrees off
    assert [(f.kind, f.target_id) for f in forbidden] == [("ahead_in_line", "t0")] * 3
    assert [_corners(f.polygon) for f in forbidden] == [                             # last person first, anchor last
        [(0.4, 3.8), (0.4, 4.5), (1.6, 3.8), (1.6, 4.5)], [(0.4, 4.5), (0.4, 5.2), (1.6, 4.5), (1.6, 5.2)],
        [(0.52, 5.56), (1.12, 6.36), (1.48, 4.84), (2.08, 5.64)]]
    tail = Person(id="q9", x=1.42, y=3.24, yaw=0.0, group="g0")                      # joins behind (1.0, 3.8), off the axis
    (r,), forbidden = rules.region_waiting_line(desk, [tail] + _line(4.5, 3.8, 5.2))   # the line now runs along (0.6, -0.8): second-to-last member to last
    assert len(forbidden) == 4 and r.heading.yaw == pytest.approx(math.atan2(0.8, -0.6))
    assert [_in([r], 1.42 + d * 0.6, 3.24 - d * 0.8) for d in (0.45, 0.55, 1.15, 1.25, -0.85)] == EDGES + [False]    # behind the tail, and ahead of it
    level = [(1.42 + side * 0.8, 3.24 + side * 0.6) for side in (0.85, -0.85)]        # beside the tail, on the line across the line's direction
    assert [_in([r], x + k * 0.6, y - k * 0.8) for x, y in level for k in (0.01, -0.01)] == [True, False] * 2      # 1 cm behind it, 1 cm ahead

def test_waiting_line_one_person_uses_the_anchor_to_member_direction():
    member = Person(id="q0", x=1.2, y=1.6, yaw=0.0, group="g0")
    (r,), forbidden = rules.region_waiting_line(_desk(anchor=(0.0, 0.0, 1.0)), [member])
    assert (r.heading.face_point, r.heading.yaw) == (None, pytest.approx(math.atan2(-1.6, -1.2)))       # towards the anchor's ground point
    assert [_in([r], 1.2 + d * 0.6, 1.6 + d * 0.8) for d in (0.45, 0.55, 1.15, 1.25, -0.85)] == EDGES + [False]     # behind the member: away from the anchor
    assert [_in([r], 0.52 + k * 0.6, 2.11 + k * 0.8) for k in (0.01, -0.01)] == [True, False]     # 0.85 m beside the member: 1 cm behind, 1 cm ahead
    assert [_corners(f.polygon) for f in forbidden] == [[(-0.48, 0.36), (0.48, -0.36), (0.72, 1.96), (1.68, 1.24)]]
    head, behind = _line(5.2, 4.5)                   # the target person heads the line and stands on the anchor
    assert len(rules.region_waiting_line(_person_target(head), [head, behind])[1]) == 1      # no zero-length corridor piece

def _group():                            # centroid (2.0, 2.0); p0 is 0.8 m from it, p1 and p2 0.72 m; all face it
    places = [(2.0, 1.2, math.pi / 2), (1.4, 2.4, -0.588), (2.6, 2.4, -2.554)]
    return [Person(id=f"p{i}", x=x, y=y, yaw=yaw, group="g1") for i, (x, y, yaw) in enumerate(places)]

def test_conversation_group_three_members():
    regions, (space,) = rules.region_conversation_group(_person_target(_group()[0]), _group())
    assert len(regions) == 1 and (regions[0].target_id, regions[0].heading.face_point) == ("t1", (2.0, 1.2))   # one band right round the group, facing p0
    centre = (2.0, 2.0)                  # the radii below are from it; the group's circle has radius 0.8 m
    cases = {270: (1.25, 1.35, 1.95, 2.05),          # behind p0, who stands on the circle and faces away: 0.45, 0.55, 1.15 and 1.25 m from them
             146.31: (1.17, 1.27, 1.87, 1.97),       # past p1, 0.72 m out: the same four distances from them
             210: (0.75, 0.85, 1.33, 1.43),          # between p0 and p1, on p0's side: the circle, then 1.2 m from p0 at 1.38 m
             90: (0.75, 0.85, 1.39, 1.49)}           # opposite p0, between p1 and p2: the circle, then 1.2 m from both at 1.44 m
    for deg, radii in cases.items():
        assert [_in(regions, *_out(centre, deg, r)) for r in radii] == EDGES, deg
    radii = [math.hypot(x - 2.0, y - 2.0) for x, y in space.polygon]     # 0.8 m is the largest member distance; the mean is 0.75 m
    assert (space.kind, space.target_id, round(min(radii), 2), round(max(radii), 2)) == ("group_space", "t1", 0.8, 0.8)

def test_conversation_group_two_members_on_every_side_of_the_pair():
    members = [Person(id="p0", x=-0.6, y=0.0, yaw=math.radians(20.0), group="g1"), Person(id="p1", x=0.6, y=0.0, yaw=math.pi, group="g1")]
    regions, (space,) = rules.region_conversation_group(_person_target(members[0]), members)
    assert len(regions) == 1 and regions[0].heading.face_point == (-0.6, 0.0)        # the target person, wherever the robot stands
    for side in (1, -1):                 # beside the pair, either side: the circle at 0.6 m, then 1.2 m from both at 1.04 m
        assert [_in(regions, 0.0, side * y) for y in (0.55, 0.65, 0.99, 1.09)] == EDGES, side
    for end in (1, -1):                  # past either member, behind the target person too: 0.45, 0.55, 1.15 and 1.25 m from them
        assert [_in(regions, end * (0.6 + d), 0.0) for d in (0.45, 0.55, 1.15, 1.25)] == EDGES, end
    assert _in(regions, *_out((0.0, 0.0), -1.0, 1.45))       # between the last ray of the band and the first: it closes on itself
    assert max(math.hypot(x, y) for x, y in space.polygon) == pytest.approx(0.6, abs=0.005)

def test_only_the_group_at_the_target_and_the_target_person_shape_a_region():
    def stranger(x, y): return [Person(id="s0", x=x, y=y, yaw=0.0)]      # in no group: a bystander, or someone at work behind a counter
    desk, line, members = _desk(), _line(5.2, 4.5), _group()
    talker = _person_target(members[0])
    assert rules.region_no_one(desk, stranger(1.0, 6.55)) == rules.region_no_one(desk, [])      # 0.55 m behind the desk's front face
    assert rules.build_regions("waiting_line", desk, line + stranger(1.3, 4.2), "g0") == rules.region_waiting_line(desk, line)     # 0.42 m from the last in line
    assert rules.build_regions("conversation_group", talker, members + stranger(2.0, 0.6), "g1") == rules.region_conversation_group(talker, members)   # 0.6 m behind p0

def test_build_regions_dispatch_and_missing_inputs():
    members, desk = _group(), _desk()
    talker = _person_target(members[0])                                              # p0 of group g1
    people = members + _line(5.2, 4.5) + [Person(id="b0", x=1.0, y=3.6, yaw=0.0)]    # b0: a bystander behind the line
    none = [("picnic", desk, people, "g0"), ("waiting_line", desk, people, None), ("conversation_group", talker, people, None), ("waiting_line", desk, people, "g9"),
            ("conversation_group", desk, people, "g1"), ("conversation_group", talker, people[:1], "g1"), ("waiting_line", talker, people[:1], "g1")]
    assert [rules.build_regions(*args) for args in none] == [None] * 7   # no rule; no group, twice; empty group; target not a member; one member; a line of the target alone
    assert rules.build_regions("no_one", talker, people, None) == rules.region_no_one(talker, people)
    assert rules.build_regions("waiting_line", desk, people, "g0") == rules.region_waiting_line(desk, _line(5.2, 4.5))
    assert rules.build_regions("conversation_group", talker, people, "g1") == rules.region_conversation_group(talker, members)

# --- What is no rule, and what is a fault ---

def test_a_rules_own_condition_that_is_not_met_is_no_rule_and_any_other_error_passes_through(monkeypatch):
    desk, members = _desk(), _group()
    talker, head = _person_target(members[0]), _line(5.2, 4.5)[0]
    no_rule = [(rules.region_waiting_line, desk, []),                                # nobody in the line
               (rules.region_waiting_line, desk, _line(4.5, 4.5)),                   # its last member stands on the one before: no direction
               (rules.region_waiting_line, _person_target(head), [head]),            # one member, on the anchor: no direction either
               (rules.region_conversation_group, talker, members[:1]),               # one member
               (rules.region_conversation_group, _person_target(head), members)]     # the target is not a member
    for rule, target, people in no_rule:
        with pytest.raises(rules.NoRule):
            rule(target, people)
    assert issubclass(rules.NoRule, ValueError)
    def faulty(target, people): raise ValueError("a fault, not a condition of the rule")
    for situation in rules._GROUP_RULES:                                             # build_regions answers NoRule alone with None
        with monkeypatch.context() as patch:
            patch.setitem(rules._GROUP_RULES, situation, faulty)
            with pytest.raises(ValueError, match="a fault") as raised:
                rules.build_regions(situation, desk, _line(5.2, 4.5), "g0")
            assert raised.type is ValueError
    lost = [members[0], members[1].model_copy(update={"x": math.nan}), members[2]]   # a member whose position is no number: an error, as
    for off in (math.nan, math.inf):                                                 # it was before these rules, and not "no region"
        lost[1] = members[1].model_copy(update={"x": off})
        with pytest.raises(ValueError) as raised:
            rules.build_regions("conversation_group", talker, lost, "g1")
        assert raised.type is ValueError and "NaN" in str(raised.value)
        # The later rules say so themselves, of every position they read and, at a picture, of the way each one faces.
        later = [("audience", SCREEN, [ROW[0].model_copy(update={"y": off}), *ROW[1:]]), ("lift_wait", LIFT, [PAIR[0], PAIR[1].model_copy(update={"x": off})]),
                 ("audience", SCREEN.model_copy(update={"anchor": (off, 0.0, 1.4)}), ROW),
                 ("audience", SCREEN.model_copy(update={"footprint": [(-0.5, -1.0), (0.0, off), (0.0, 1.0), (-0.5, 1.0)]}), ROW),
                 ("group_photo", _whose(TAKER), [POSING[0].model_copy(update={"x": off}), *POSING[1:], TAKER]),
                 ("group_photo", _whose(TAKER), [*POSING, TAKER.model_copy(update={"yaw": off})])]
        for situation, target, people in later:
            with pytest.raises(ValueError, match="is not a finite number") as raised:
                rules.build_regions(situation, target, people, "g0")
            assert raised.type is ValueError, (situation, off)
        # A front that is no direction is the rule's own condition, as a line with no direction is: nothing is nearer
        # along it or further, so there is no rule and no region.
        with pytest.raises(rules.NoRule, match="front_yaw is no direction"):
            rules.region_attending(SCREEN.model_copy(update={"front_yaw": off}), ROW, "audience_view")
        assert [rules.build_regions(name, thing.model_copy(update={"front_yaw": off}), people, "g0")
                for name, thing, people in (("audience", SCREEN, ROW), ("lift_wait", LIFT, PAIR))] == [None, None]

# --- People attending to the target (audience, lift_wait) and a picture being taken (group_photo) ---

def test_the_page_states_the_three_later_rules_with_the_numbers_of_the_code():
    from pathlib import Path
    page = (Path(__file__).resolve().parents[1] / "docs" / "scene_format.md").read_text(encoding="utf-8")
    rule = {name: next(line for line in page.splitlines() if line.startswith(f"- **`{name}`**")) for name in ("audience", "lift_wait", "group_photo")}
    reach = f"{rules._JOIN_NEAR_M:g} to {rules._JOIN_FAR_M:g} m"                    # 0.5 to 1.2 m, from the nearest of the people stood by
    assert reach in rule["audience"] and f"at least {rules._VIEW_CLEAR_M:g} m outside" in rule["audience"] and "convex hull" in rule["audience"]
    assert "no nearer the target than the member nearest it" in rule["audience"] and "`front_yaw`" in rule["audience"]
    assert "`audience_view`" in rule["audience"] and "`ahead_in_line`" in rule["lift_wait"] and "`photo_frame`" in rule["group_photo"]
    assert "The rule is that of `audience`" in rule["lift_wait"]
    assert rule["group_photo"].count(f"within {rules._TAKER_SEES_DEG:g} degrees") == 2          # the picture-taker of the others, and they of the picture-taker
    assert reach in rule["group_photo"] and f"{rules._FRAME_MARGIN_M:g} m to each side" in rule["group_photo"] and f"{rules._JOIN_FAR_M:g} m behind" in rule["group_photo"]
    assert "no rule yet" not in page and "this project's own first definitions" in page and "compact group" in page
    assert page.count("`photo_frame`") == 3                                         # with the rule, and in both lists of the kinds

SCREEN = Target(id="t0", kind="place", label="big screen", anchor=(0.0, 0.0, 1.4), front_yaw=0.0,
                footprint=[(-0.5, -1.0), (0.0, -1.0), (0.0, 1.0), (-0.5, 1.0)])     # 2 m wide, its face on x = 0, looking along +x
LIFT = Target(id="t0", kind="place", label="lift", anchor=(0.0, 0.0, 1.2), front_yaw=0.0,
              footprint=[(-1.8, -1.3), (0.0, -1.3), (0.0, 1.3), (-1.8, 1.3)])       # a shaft 2.6 m wide, its doors on x = 0
def _watching(*places): return [Person(id=f"w{i}", x=x, y=y, yaw=math.pi, group="g0") for i, (x, y) in enumerate(places)]   # facing the target
ROW = _watching((2.5, -1.0), (2.5, 0.0), (2.5, 1.0))     # three in a row 2.5 m from the screen, a metre apart
PAIR = _watching((1.5, 0.35), (1.95, -0.45))             # two waiting, not in single file: the first is 1.5 m from the doors
def _area(polygon): return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1]))) / 2

def test_audience_is_joined_behind_the_row_and_beside_its_ends_and_never_between_it_and_the_screen_or_ahead_of_it():
    (r,), (view,) = rules.region_attending(SCREEN, ROW, "audience_view")             # one band, from beside one end round the back to the other
    assert (r.target_id, r.heading.face_point, r.heading.yaw, r.tolerance_deg) == ("t0", (0.0, 0.0), None, 45.0)      # facing the screen
    for y in (-1.0, -0.5, 0.0, 0.5, 1.0):            # behind the back row, anyone in it and the gaps: from 0.3 m clear of the row's line,
        near = max(0.3, math.sqrt(0.25 - min(abs(y - m.y) for m in ROW) ** 2))       # or 0.5 m from the nearest of them, to 1.2 m from them
        far = math.sqrt(1.44 - min(abs(y - m.y) for m in ROW) ** 2)
        assert [_in([r], 2.5 + d, y) for d in (near - 0.1, near + 0.1, far - 0.1, far + 0.1)] == EDGES, y
    assert _in([r], 3.2, 0.0)                                                        # 0.7 m straight behind the middle watcher
    assert not _in([r], 2.7, 0.5)                                                    # behind a gap and only 0.2 m behind the row's line: a body would stand in it
    assert not _in([r], 2.5, 0.5) and not _in([r], 1.5, 0.0)                         # in a gap of the row, and between the middle watcher and the screen
    for side in (1, -1):                             # the row's two ends mirror one another
        assert [_in([r], 2.6, side * (1.0 + d)) for d in (0.4, 0.6, 1.1, 1.3)] == EDGES, side      # beside the row's end: 0.5 to 1.2 m from the one who stands there
        assert [_in([r], 2.5 + back, side * 1.7) for back in (-0.1, -0.01, 0.01, 0.1)] == [False, False, True, True], side   # level with them the region ends, on the line itself
        assert not _in([r], 1.5, side * 1.4)                                         # ahead of the row and well to the side: 0.4 m clear of the space between, and nearer the screen than anyone
    assert (view.kind, view.target_id) == ("audience_view", "t0") and _corners(view.polygon) == [(-0.5, -1.0), (-0.5, 1.0), (2.5, -1.0), (2.5, 1.0)]
    assert [point_in_polygon(p, view.polygon) for p in ((1.5, 0.0), (2.4, 0.9), (-0.4, 0.0), (2.6, 0.0), (3.9, 0.0), (1.5, 1.1))] == [True] * 3 + [False] * 3    # the people's and the screen's hull, and nothing behind them

def test_audience_of_two_rows_is_joined_behind_and_beside_and_not_between_the_rows():
    rows = _watching((2.05, -1.0), (2.05, 0.0), (2.05, 1.0), (3.15, -0.75), (3.15, 0.25))
    (r,), (view,) = rules.region_attending(SCREEN, rows, "audience_view")
    assert not _in([r], 2.6, 0.0) and point_in_polygon((2.6, 0.0), view.polygon)     # between the two rows, 0.55 m from one of the front row: inside the hull
    assert [_in([r], 3.15 + d, 0.25) for d in (0.4, 0.6, 1.1, 1.3)] == EDGES         # behind one of the back row
    assert _in([r], 2.2, 1.8) and not _in([r], 1.9, 1.8)                             # beside the front row's end: behind the line level with it, and ahead of that

def test_lift_wait_is_the_audience_rule_with_another_name_for_what_is_forbidden():
    (waited,), (ahead,) = rules.region_attending(LIFT, PAIR, "ahead_in_line")
    (watched,), (view,) = rules.region_attending(LIFT, PAIR, "audience_view")
    assert (ahead.kind, view.kind, ahead.polygon, waited) == ("ahead_in_line", "audience_view", view.polygon, watched)
    assert waited.heading.face_point == (0.0, 0.0)                                   # facing the doors
    assert not _in([waited], 1.1, 1.3)               # diagonally ahead of the first person waiting: 1.03 m from them and 0.59 m clear of the space before the doors, and nearer the doors than they are
    assert _in([waited], 1.6, 1.05)                  # beside them, 0.1 m behind the line level with them
    assert [_in([waited], 1.5 + back, 1.15) for back in (-0.1, 0.1)] == [False, True]      # that line cuts the band beside an edge of the space: to a sample, not to the centimetre
    assert not _in([waited], 0.8, 0.0) and point_in_polygon((0.8, 0.0), ahead.polygon)     # in the doorway, between them and the doors
    assert [_in([waited], 1.95 + d, -0.45) for d in (0.4, 0.6, 1.1, 1.3)] == EDGES   # behind the second
    assert sorted(_corners(ahead.polygon)) == sorted([(-1.8, -1.3), (-1.8, 1.3), (0.0, -1.3), (0.0, 1.3), (1.5, 0.35), (1.95, -0.45)])   # the shaft's whole footprint counts as the thing
    turned = LIFT.model_copy(update={"front_yaw": math.pi / 2})                      # a front along +y: nearer is then measured along y, and the one at y = -0.45 is the nearest
    (r,), _ = rules.region_attending(turned, PAIR, "ahead_in_line")
    assert [_in([r], 2.75, y) for y in (-0.55, -0.35)] == [False, True]              # 0.8 m beside them, clear of the space: 0.1 m on the far side of the line level with them, and on the near side
    # That line leaves the space at the nearest member's own corner of it, between two of the normals the walk takes
    # every 2 degrees round the corner. A normal along the line itself is added there, so a metre beside that member
    # the region ends on the line to the centimetre; without it, it would end 2.5 cm short.
    assert [_in([r], 2.95, -0.45 + back) for back in (-0.01, 0.01)] == [False, True]

def test_a_thing_without_a_footprint_has_room_of_its_own_and_nearer_is_plain_distance_where_it_has_no_front():
    notice = Target(id="t0", kind="object", label="notice", anchor=(0.0, 0.0, 1.4))  # no footprint, no front: a disc of 0.5 m round its anchor stands for it
    one = _watching((2.0, 0.0))
    (r,), (view,) = rules.region_attending(notice, one, "audience_view")
    assert max(math.hypot(x, y) for x, y in view.polygon if x < 0.1) == pytest.approx(0.5) and (2.0, 0.0) in view.polygon      # the disc, drawn out to the watcher
    assert [point_in_polygon(p, view.polygon) for p in ((1.0, 0.0), (0.0, 0.4), (1.2, 0.1), (0.0, 0.6), (1.2, 0.35), (2.1, 0.0))] == [True] * 3 + [False] * 3
    assert [_in([r], 2.0 + d, 0.0) for d in (0.4, 0.6, 1.1, 1.3)] == EDGES           # behind the watcher
    assert _in([r], 2.0, 0.8) and _in([r], 2.0, -0.8)                                # beside them: 2.15 m from the notice, further than they are
    assert not _in([r], 1.6, 0.9)                    # diagonally before them: 0.98 m from them and 0.77 m clear of the space, and 1.84 m from the notice, nearer than their 2.0 m
    assert not _in([r], 1.2, 0.35)                   # 0.15 m outside the space between: less than the clearance
    apart = _watching((1.5, 0.0), (2.4, 0.3))        # two at different distances from the notice, 1.5 and 2.42 m: the nearest of them is the measure
    (r,), _ = rules.region_attending(notice, apart, "audience_view")
    assert _in([r], 1.75, -0.75)                     # beside the nearer one, 0.79 m from them: 1.90 m from the notice, further than they are and nearer than the other
    assert not _in([r], 1.05, -0.9)                  # diagonally before them, 1.0 m from them and well clear of the space: 1.38 m from the notice, nearer than they are
    speaker = Target(id="t0", kind="person", label="person in red", anchor=(0.0, 0.0, 1.1), person_id="x0")      # someone spoken to: not one of those listening
    crowd = _watching((1.5, -0.6), (1.5, 0.6), (-0.4, 1.6))
    regions, (space,) = rules.region_attending(speaker, crowd, "audience_view")
    assert regions and not any(_in(regions, *_out((0.0, 0.0), deg, 0.75)) for deg in range(0, 360, 15))      # nowhere within 0.5 + 0.3 m of the one spoken to, on any side
    assert all(math.hypot(x, y) >= 1.6 - 0.03 for r in regions for x, y in r.polygon)       # and nowhere nearer them than the nearest listener, 1.62 m off: to a sample
    # Those two hold by the nearest listener's distance alone. That a person, too, has half a metre of their own shows in
    # the space between: it holds the place 0.4 m behind the speaker, away from all who listen, and reaches 0.5 m from them there.
    assert [point_in_polygon(p, space.polygon) for p in ((0.0, -0.4), (-0.4, -0.2), (0.0, -0.6))] == [True, True, False]
    assert max(math.hypot(x, y) for x, y in space.polygon if x < 0.1 and y < -0.1) == pytest.approx(0.5)
    couple = _watching((1.0, 0.0), (0.0, 2.0))       # two listening, a quarter turn apart round the speaker
    regions, _ = rules.region_attending(speaker, couple, "audience_view")
    assert not _in(regions, -0.42, 1.1)              # 0.99 m from the far listener and 1.18 m from the speaker. Round the speaker's disc the space between comes within 0.18 m
    assert _in(regions, -0.6, 1.4)                   # of it; drawn from the speaker's own spot it would be 0.42 m off, and the place in the region. Further out, it is
    fronted = notice.model_copy(update={"front_yaw": 0.0})
    (r,), _ = rules.region_attending(fronted, one, "audience_view")                  # with a front, nearer is measured along it: the line across it through the watcher
    assert [_in([r], 2.0 + back, 0.8) for back in (-0.1, -0.01, 0.01, 0.1)] == [False, False, True, True]

def test_a_space_between_without_an_inside_is_a_segment_and_nothing_is_forbidden():
    assert rules._space_between([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (1.0, 2.0), (0.0, 2.0), (1.0, 1.0)]) == ([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)], True)
    turn = math.radians(20.0)            # a straight row of five, turned so that float noise puts two of its middle three off the line:
    row = [(1.0 + 0.9 * k * math.cos(turn), 2.0 + 0.9 * k * math.sin(turn)) for k in range(5)]
    assert (len(convex_hull(row)), len(convex_hull(row + [(1.0, 5.0)]))) == (4, 5)   # convex_hull keeps them as corners, alone and beside a real corner, and they are none
    assert rules._space_between(row) == ([row[0], row[4]], False)
    with_a_corner = rules._space_between(row + [(1.0, 5.0)])
    assert with_a_corner == ([row[0], row[4], (1.0, 5.0)], True)                     # the row keeps its two ends and no more
    assert rules._space_between([(1.0, 1.0)] * 3) == ([(1.0, 1.0)], False)           # all one point
    sliver = [(0.0, 0.0), (1.0, 5e-10), (2.0, 0.0)]                                  # an area of 5e-10 m2 over a perimeter of 4 m: no inside
    assert rules._space_between(sliver) == ([(0.0, 0.0), (2.0, 0.0)], False)
    line = SCREEN.model_copy(update={"footprint": [(-0.5, 0.0), (0.0, 0.0)]})        # a thing whose outline is a line, and one watcher on that line
    (r,), forbidden = rules.region_attending(line, _watching((2.0, 0.0)), "audience_view")
    assert forbidden == []
    assert [_in([r], 2.0 + d, 0.0) for d in (0.4, 0.6, 1.1, 1.3)] == EDGES           # behind the watcher
    assert _in([r], 2.1, 0.8) and not _in([r], 1.5, 0.2) and not _in([r], 1.5, 0.6)  # beside them; 0.2 m off the line of sight; and clear of it, but nearer the thing

def test_a_band_that_runs_across_the_start_of_the_walk_is_one_region_and_the_walk_has_no_normal_twice():
    # The walk round the space between starts at the hull's first vertex. In the layouts above the thing is at smaller x
    # than the people, so that is a corner of the thing, where no normal has a stretch. Mirrored, it is the watcher at
    # the row's end: the walk starts and ends in the band, straight behind the row.
    mirror = SCREEN.model_copy(update={"front_yaw": math.pi, "footprint": [(0.5, 1.0), (0.0, 1.0), (0.0, -1.0), (0.5, -1.0)]})    # its face on x = 0, looking along -x
    row = [m.model_copy(update={"x": -m.x, "yaw": 0.0}) for m in ROW]
    space, solid = rules._space_between([*((m.x, m.y) for m in row), *mirror.footprint])
    walk = rules._walk(space, rules._VIEW_CLEAR_M)
    assert solid and space[0] == (-2.5, -1.0) and [walk[k][1] for k in (0, -1)] == [pytest.approx((-1.0, 0.0))] * 2
    (r,), (view,) = rules.region_attending(mirror, row, "audience_view")             # one band, not two that end where the walk starts
    (plain,), _ = rules.region_attending(SCREEN, ROW, "audience_view")
    assert _area(r.polygon) == pytest.approx(_area(plain.polygon), abs=1e-6) and _corners(view.polygon) == [(-2.5, -1.0), (-2.5, 1.0), (0.5, -1.0), (0.5, 1.0)]
    for y in (-1.0, 0.0, 1.0):                       # behind each of the row, the one at the walk's start among them
        assert [_in([r], -2.5 - d, y) for d in (0.4, 0.6, 1.1, 1.3)] == EDGES, y
    twin = ROW[1].model_copy(update={"id": "w9"})    # a second watcher on one's spot changes nothing
    assert rules.region_attending(SCREEN, ROW + [twin], "audience_view") == rules.region_attending(SCREEN, ROW, "audience_view")
    # A normal across the target's front is added to the walk only where the walk has none: round a thing without a
    # footprint, whose disc turns 7.5 degrees at each vertex, a front at a multiple of 7.5 degrees has both already.
    notice = Target(id="t0", kind="object", label="notice", anchor=(0.0, 0.0, 1.4))
    counts = {}
    for front_deg in (0.0, 7.5, 90.0, 180.0, 33.0):
        front = math.radians(front_deg)
        space, _ = rules._space_between([(2.0 * math.cos(front), 2.0 * math.sin(front)), *rules.disc((0.0, 0.0), rules._JOIN_NEAR_M)])
        walk = rules._walk(space, rules._VIEW_CLEAR_M, (front + math.pi / 2, front - math.pi / 2))
        normals = [normal for _, normal in walk]
        steps = [math.dist(p, q) + math.dist(n, m) for (p, n), (q, m) in zip(walk, walk[1:] + walk[:1])]
        assert min(steps) > 1e-6, front_deg                                          # no two neighbours are one normal at one point
        for across in (front + math.pi / 2, front - math.pi / 2):                    # and each direction across the front is there
            assert min(math.dist(n, (math.cos(across), math.sin(across))) for n in normals) < 1e-9, front_deg
        counts[front_deg] = len(walk) - len(rules._walk(space, rules._VIEW_CLEAR_M))
    assert counts == {0.0: 0, 7.5: 0, 90.0: 0, 180.0: 0, 33.0: 2}                    # two more normals only where the walk had not got them

def test_attending_has_no_rule_without_members_or_for_a_target_among_them():
    one_of_them = _person_target(ROW[0]).model_copy(update={"id": "t0"})
    for members, target in (([], SCREEN), (ROW, one_of_them)):
        with pytest.raises(rules.NoRule):
            rules.region_attending(target, members, "audience_view")
    for situation in ("audience", "lift_wait"):
        none = [(situation, SCREEN, ROW, None), (situation, SCREEN, ROW, "g9"), (situation, one_of_them, ROW, "g0")]
        assert [rules.build_regions(*args) for args in none] == [None] * 3, situation    # no group; nobody in it; the target is a member
    stranger = [Person(id="s0", x=2.5, y=1.9, yaw=0.0)]                              # in no group, 0.9 m beside the row's end: shapes nothing
    assert rules.build_regions("audience", SCREEN, ROW + stranger, "g0") == rules.region_attending(SCREEN, ROW, "audience_view")
    assert rules.build_regions("lift_wait", LIFT, PAIR + stranger, "g0") == rules.region_attending(LIFT, PAIR, "ahead_in_line")

TAKER = Person(id="c0", x=3.0, y=0.0, yaw=math.pi, group="g0")                      # looks along -x at the row, 3 m away
POSING = [Person(id=f"s{i}", x=0.0, y=y, yaw=0.0, group="g0") for i, y in enumerate((-0.6, 0.0, 0.6))]     # a row across the picture, facing the camera
def _whose(p): return _person_target(p).model_copy(update={"id": "t0"})

def test_group_photo_is_joined_beside_or_behind_the_picture_taker_whoever_the_robot_came_for():
    everyone = POSING + [TAKER]
    (r,), (picture,) = rules.region_group_photo(_whose(TAKER), everyone)
    assert (r.target_id, r.heading.face_point, r.heading.yaw, picture.kind, picture.target_id) == ("t0", (3.0, 0.0), None, "photo_frame", "t0")
    assert [_in([r], 3.0 + d, 0.0) for d in (0.4, 0.6, 1.1, 1.3)] == EDGES           # behind the picture-taker
    for side in (1, -1):
        assert [_in([r], 3.1, side * d) for d in (0.38, 0.6, 1.09, 1.3)] == EDGES, side              # beside them, 0.1 m behind the line level with them
        assert [_in([r], 3.0 + back, side * 0.85) for back in (-0.1, -0.01, 0.01, 0.1)] == [False, False, True, True], side   # the region ends on that line itself
        assert not _in([r], 2.5, side * 0.7)                                         # half a metre ahead of them and 0.7 m aside: 0.86 m from them, on the side of those posing
        assert not _in([r], 0.0, side * 1.4)                                         # beside the row's end
    assert not _in([r], 2.2, 0.0)                                                    # before the camera
    assert _area(r.polygon) == pytest.approx(math.pi * (1.44 - 0.25) / 2, abs=0.01)  # half the ring: 1.87 m2
    for target in POSING:                            # for one of those posing the place is the same, and the robot faces them from it
        (same,), (strip,) = rules.region_group_photo(_whose(target), everyone)
        assert same.polygon == r.polygon and same.heading.face_point == (target.x, target.y) and strip.polygon == picture.polygon
    turned = [*POSING, TAKER.model_copy(update={"yaw": math.pi + math.radians(8.0)})]    # the picture-taker's body turned 8 degrees: nothing reads it
    assert rules.region_group_photo(_whose(TAKER), turned) == rules.region_group_photo(_whose(TAKER), everyone)
    wide = [Person(id="a0", x=0.0, y=-1.2, yaw=0.0, group="g0"), *POSING, TAKER]     # one more posing, on one side: ahead is towards their middle, (0, -0.3)
    (off,), _ = rules.region_group_photo(_whose(TAKER), wide)
    w = (-3.0 / math.hypot(3.0, 0.3), -0.3 / math.hypot(3.0, 0.3))
    assert [_in([off], 3.0 - 0.85 * w[1] + k * w[0], 0.85 * w[0] + k * w[1]) for k in (0.1, -0.1)] == [False, True]    # 0.85 m beside them across that way: ahead, and behind

def test_the_strip_of_a_picture_runs_from_the_camera_through_those_posing_and_on_behind_them():
    _, (picture,) = rules.region_group_photo(_whose(POSING[0]), POSING + [TAKER])
    inside = [(1.5, 0.0), (2.2, 0.0), (0.0, -1.0), (-0.9, -0.6), (-1.5, 0.0), (3.4, 0.0), (1.5, 0.6)]      # before the camera; before those posing; 0.4 m beside the row's end; behind them; behind the camera by less than the margin
    outside = [(0.0, -1.4), (3.8, 0.0), (-2.0, 0.0), (1.5, 1.0), (3.0, 0.7)]         # beside the row's end, 0.28 m clear; behind the picture-taker; far behind the row; beside the strip; beside the picture-taker
    assert [point_in_polygon(p, picture.polygon) for p in inside + outside] == [True] * len(inside) + [False] * len(outside)
    reach = [max(x * ux + y * uy for x, y in picture.polygon) for ux, uy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    assert reach == pytest.approx([3.5, 1.7, 0.6 + 1.2 * 0.6 / math.hypot(3.0, 0.6) + 0.5, 0.6 + 1.2 * 0.6 / math.hypot(3.0, 0.6) + 0.5], abs=2e-3)     # 0.5 m round the camera, round each posing and round the point 1.2 m behind each: discs of 48 vertices

def test_the_picture_taker_is_the_one_who_faces_all_the_others_and_is_faced_by_all_of_them():
    def refused(target, members):
        with pytest.raises(rules.NoRule):
            rules.region_group_photo(target, members)
        return rules.build_regions("group_photo", target, members, "g0") is None
    everyone = POSING + [TAKER]
    assert refused(_whose(TAKER), [TAKER])                                           # one member
    assert refused(_whose(Person(id="x0", x=5.0, y=0.0, yaw=0.0)), everyone)         # the target is not a member
    assert refused(_whose(TAKER), [POSING[1], TAKER])                                # a picture of one person: each of the two faces the other
    assert refused(_whose(POSING[0]), POSING)                                        # a row alone: nobody faces the others
    assert refused(_whose(POSING[0]), POSING + [TAKER.model_copy(update={"yaw": math.pi / 2})])      # the camera turned away from the row
    assert refused(_whose(POSING[0]), [p.model_copy(update={"yaw": math.pi}) for p in POSING] + [TAKER])   # the row turned its back on the camera
    def aimed(off_deg):                  # the picture-taker turned so that the one at the row's end is that many degrees off the way they face
        end = math.degrees(math.atan2(0.6, 3.0))     # where that one is seen from the picture-taker's place, off the middle of the row
        return [*POSING, TAKER.model_copy(update={"yaw": math.pi + math.radians(off_deg - end)})]
    (r,), _ = rules.region_group_photo(_whose(TAKER), everyone)
    assert rules.region_group_photo(_whose(TAKER), aimed(59.0))[0][0].polygon == r.polygon      # 59 degrees off: still seen, and the place is where it was
    assert refused(_whose(TAKER), aimed(61.0))                                       # 61 degrees off: nobody faces all the others
    def faced(off_deg):                  # the other half: the one at the row's end turned so that the camera is that many degrees off the way they face
        return [POSING[0].model_copy(update={"yaw": math.atan2(0.6, 3.0) + math.radians(off_deg)}), *POSING[1:], TAKER]
    for side in (1, -1):
        assert rules.region_group_photo(_whose(TAKER), faced(side * 59.0))[0][0].polygon == r.polygon      # 59 degrees off: the one at the camera is still faced by all
        assert refused(_whose(TAKER), faced(side * 61.0))                            # 61 degrees off: they are not, and nobody is the picture-taker
    # Two people on one spot are not seen by one another. Of those posing that is not asked, so with two of them on one
    # spot the one at the camera is still the one picture-taker.
    (doubled,), _ = rules.region_group_photo(_whose(TAKER), [*POSING, POSING[0].model_copy(update={"id": "s9"}), TAKER])
    assert [_in([doubled], 3.0 + d, 0.0) for d in (0.4, 0.6, 1.1, 1.3)] == EDGES and doubled.heading.face_point == (3.0, 0.0)
    # A second person on the picture-taker's own spot leaves nobody who sees all the others. The row is turned to a
    # camera on its other side for this, where the picture-taker faces along +x: the way from a spot to itself, which
    # atan2 gives as 0, is then the way they face, and only the rule's own word for it keeps the two from seeing one another.
    left = [p.model_copy(update={"yaw": math.pi}) for p in POSING]
    camera = TAKER.model_copy(update={"x": -3.0, "yaw": 0.0})
    assert len(rules.region_group_photo(_whose(camera), left + [camera])[0]) == 1
    beside = camera.model_copy(update={"id": "c1", "yaw": math.radians(50.0)})       # turned so that the row's end is 61 degrees off: no picture-taker themselves
    assert refused(_whose(camera), left + [camera, beside])
    near = [Person(id=f"s{i}", x=0.0, y=y, yaw=math.atan2(-y, 1.5), group="g0") for i, y in enumerate((-1.0, 0.0, 1.0))]     # a wide row 1.5 m off, its ends turned to the camera:
    close = TAKER.model_copy(update={"x": 1.5})      # each end has the whole row within 60 degrees of the way they face, and the middle one does not have them so
    (r,), (picture,) = rules.region_group_photo(_whose(near[0]), near + [close])
    assert picture.polygon[0] != r.polygon[0] and [_in([r], 1.5 + d, 0.0) for d in (0.4, 0.6, 1.1, 1.3)] == EDGES    # exactly one picture-taker, the one at the camera
    assert rules.build_regions("group_photo", _whose(TAKER), everyone, None) is None and rules.build_regions("group_photo", _whose(TAKER), everyone, "g9") is None
    stranger = [Person(id="b0", x=3.0, y=0.9, yaw=math.pi)]                          # in no group, 0.9 m beside the picture-taker and facing as they do
    assert rules.build_regions("group_photo", _whose(TAKER), everyone + stranger, "g0") == rules.region_group_photo(_whose(TAKER), everyone)

# --- Someone using a machine (machine_use) ---

MACHINE = Target(id="t0", kind="place", label="cash machine", anchor=(0.0, 0.0, 0.8), front_yaw=0.0,
                 footprint=[(-0.7, -0.4), (0.0, -0.4), (0.0, 0.4), (-0.7, 0.4)])    # 0.8 m wide, its face on x = 0, looking along +x
USER = Person(id="u0", x=0.5, y=0.0, yaw=math.pi, group="g0")                        # 0.5 m before its face, facing it

def test_machine_use_is_joined_one_to_two_metres_behind_the_user_and_within_60_degrees_of_straight_behind():
    (r,), (shoulder, beside) = rules.region_machine_use(MACHINE, [USER])
    assert (r.target_id, r.heading.face_point, r.heading.yaw, r.tolerance_deg) == ("t0", (0.0, 0.0), None, 45.0)    # facing the machine
    assert [_in([r], 0.5 + d, 0.0) for d in (0.95, 1.05, 1.95, 2.05)] == EDGES       # straight behind: 1.0 to 2.0 m from the user
    for side in (1, -1):
        assert [_in([r], *_out((0.5, 0.0), side * deg, 1.5)) for deg in (0.0, 30.0, 58.0, 62.0, 90.0, 120.0)] == [True] * 3 + [False] * 3, side   # within 60 degrees of behind
        assert [_in([r], *_out((0.5, 0.0), side * 59.0, d)) for d in (0.95, 1.05, 1.95, 2.05)] == EDGES, side      # the band holds near its sides too
        assert not _in([r], 0.5, side * 1.5)                                         # level beside the user: not polite
    assert not _in([r], -0.5, 0.0) and not _in([r], 0.2, 1.2)                        # ahead of the user, on the machine's side
    assert _area(r.polygon) == pytest.approx(math.pi * (4.0 - 1.0) / 3, abs=0.01)    # a third of the ring 1.0 to 2.0 m: 3.14 m2
    assert _heading_within(r, (2.0, 0.0), math.pi) and _heading_within(r, (1.5, 1.0), math.radians(180 + 33.7 - 44.0))
    assert not _heading_within(r, (2.0, 0.0), math.pi / 2)                           # turned away from the machine
    assert (shoulder.kind, shoulder.target_id, beside.kind, beside.target_id) == ("over_shoulder", "t0", "beside_user", "t0")
    for d in (0.9, 1.1):                             # within 1.0 m of the user, on every side
        for deg in (0, 90, 180, 270, 45):
            assert point_in_polygon(_out((0.5, 0.0), deg, d), shoulder.polygon) is (d < 1.0), (d, deg)
    # The strip before the machine beside the user: from its face out to 0.5 m past the user, and 0.5 m past each side of it.
    assert _corners(beside.polygon) == [(0.0, -0.9), (0.0, 0.9), (1.0, -0.9), (1.0, 0.9)]
    assert [point_in_polygon(p, beside.polygon) for p in ((0.5, 0.75), (0.3, -0.85), (0.9, 0.0), (1.1, 0.0), (0.5, 0.95), (-0.1, 0.0))] == [True] * 3 + [False] * 3

def test_machine_use_reads_the_way_the_user_faces_and_takes_the_near_limit_as_a_parameter():
    turned = USER.model_copy(update={"yaw": -math.pi + 0.2})                         # turned 11.5 degrees: straight behind is the bearing 0.2
    (r,), _ = rules.region_machine_use(MACHINE, [turned])
    assert [_in([r], *_out((0.5, 0.0), deg, 1.5)) for deg in (-55.0, 65.0, 11.5 - 58.0, 11.5 + 58.0)] == [False, True, True, True]
    assert rules._MACHINE_NEAR_M == 1.0 and rules._MACHINE_FAR_M == 2.0 and rules._MACHINE_BEHIND_DEG == 60.0
    assert rules._MACHINE_NEAR_SENSITIVITY_M == 0.8                                  # the variant reported beside the rule, fixed with it
    (near,), (shoulder, beside) = rules.region_machine_use(MACHINE, [USER], near=rules._MACHINE_NEAR_SENSITIVITY_M)
    assert [_in([near], 0.5 + d, 0.0) for d in (0.75, 0.85, 1.95, 2.05)] == EDGES    # 0.8 to 2.0 m: the near limit alone moves,
    assert point_in_polygon((1.25, 0.0), shoulder.polygon) and not point_in_polygon((1.35, 0.0), shoulder.polygon)    # and the disc over the shoulder with it
    assert beside == rules.region_machine_use(MACHINE, [USER])[1][1]                 # the strip before the machine stays
    assert rules.region_machine_use(MACHINE, [USER], near=1.0) == rules.region_machine_use(MACHINE, [USER])

def test_machine_use_needs_one_user_and_a_machine_with_a_front():
    def refused(target, members):
        with pytest.raises(rules.NoRule):
            rules.region_machine_use(target, members)
        return rules.build_regions("machine_use", target, members, "g0") is None
    second = USER.model_copy(update={"id": "u1", "y": 0.6})
    assert refused(MACHINE, []) and refused(MACHINE, [USER, second])                 # nobody, or two at the machine: not one user
    assert refused(MACHINE.model_copy(update={"front_yaw": None}), [USER])           # a machine without a front
    assert refused(MACHINE.model_copy(update={"front_yaw": math.nan}), [USER])
    assert refused(_person_target(USER).model_copy(update={"id": "t0"}), [USER])     # the target is the user: no machine
    for off in (math.nan, math.inf):                 # a position or a facing that is no number is a fault of the scene
        for bad in ({"x": off}, {"yaw": off}):
            with pytest.raises(ValueError, match="is not a finite number"):
                rules.build_regions("machine_use", MACHINE, [USER.model_copy(update=bad)], "g0")
    stranger = [Person(id="s0", x=1.5, y=0.5, yaw=0.0)]                              # in no group, a metre behind the user: shapes nothing
    assert rules.build_regions("machine_use", MACHINE, [USER, *stranger], "g0") == rules.region_machine_use(MACHINE, [USER])
    assert rules.build_regions("machine_use", MACHINE, [USER], None) is None         # no group at the target
    away = MACHINE.model_copy(update={"anchor": (3.0, 2.0, 0.8), "front_yaw": math.pi / 2,
                                      "footprint": [(2.6, 1.3), (3.4, 1.3), (3.4, 2.0), (2.6, 2.0)]})      # moved and turned: its face on y = 2, looking along +y
    user = Person(id="u0", x=3.0, y=2.5, yaw=-math.pi / 2, group="g0")
    (r,), (_, beside) = rules.region_machine_use(away, [user])
    assert _in([r], 3.0, 4.0) and not _in([r], 4.5, 2.5) and r.heading.face_point == (3.0, 2.0)
    assert _corners(beside.polygon) == [(2.1, 2.0), (2.1, 3.0), (3.9, 2.0), (3.9, 3.0)]

def test_the_page_states_the_machine_rule_with_the_numbers_of_the_code():
    from pathlib import Path
    page = (Path(__file__).resolve().parents[1] / "docs" / "scene_format.md").read_text(encoding="utf-8")
    (rule,) = [line for line in page.splitlines() if line.startswith("- **`machine_use`**")]
    assert f"{rules._MACHINE_NEAR_M:g} to {rules._MACHINE_FAR_M:g} m" in rule and f"within {rules._MACHINE_BEHIND_DEG:g} degrees" in rule
    assert "`over_shoulder`" in rule and "`beside_user`" in rule and f"{rules._MACHINE_NEAR_SENSITIVITY_M:g} m" in rule
