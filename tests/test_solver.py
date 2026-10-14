import math
from dataclasses import replace

import numpy as np
import pytest

from sgi import solver
from sgi.config import Thresholds, load_config
from sgi.geometry import lift, project, to_robot_xy
from sgi.gt.derive import is_visible, oracle_target
from sgi.scene import load_scene
from sgi.stages.goal import standard_goal
from sgi.synth.render import Box, Cylinder, Figure
from sgi.types import GoalResult, GroundedTarget, Point
from tests.solver_layouts import (HERE, TH, Layout, ask, at_points, facing, layout, looking, moved, moved_xy, mounted,
                                  mounted_camera, numbered, open_floor, person, target_at)
from tests.test_run import placeholder, scene_dirs  # noqa: F401

SHARED, VIEW, AFTER, WITHIN, VISIBLE, FACE = ("keep_out_of_shared_space", "keep_out_of_view", "come_after",
                                              "within_distance", "visible_to", "face")
SIN35, COS35, TAN35 = (f(math.radians(35.0)) for f in (math.sin, math.cos, math.tan))       # the view half-angle
ROOMY = Thresholds(max_constraints=40)             # reads a whole column of the table in one answer
UNKNOWN = "none of its numbers is in the list of people"
TOO_FEW = "fewer than two people are listed"
NO_DIRECTION = "no listed person gives a direction"
ONLY_TARGET = "nobody is listed but the target person"
NO_RANGE = "no range of two distances, neither of them negative"
NEITHER = "neither a person nor the target"
NO_HEADING = "no listed person has a heading"


def near(xy, abs=1e-9):
    return pytest.approx(xy, abs=abs)


# ------------------------------------------------------------------------------------------- what the answer is read as
def trio():
    """Person 1 with a heading, person 2 without, and person 3, who is the target."""
    return layout(HERE, [(4.0, 1.0, math.pi), (4.0, -1.0), (6.0, 0.0, math.pi)], needed=3)


def read(*constraints, th=ROOMY):
    return [(use.status, use.reason) for use in trio().solve(*constraints, th=th).uses]


USED, NOT = ("used", None), "unusable"


def test_the_target_person_is_the_one_who_stands_where_the_target_point_was_marked():
    here = open_floor()                            # its camera is over the robot: the point is moved straight away from it
    at = lambda *spots: solver.target_person(here, target_at(4.8, 0.0, 1.1), numbered(spots), TH)
    assert at((5.0, 0.0)) == 1 and at((3.0, 2.0), (5.0, 0.0)) == 2        # 0.05 m from the point moved 0.25 m: (5.05, 0)
    assert at((5.3, 0.0)) == 1 and at((5.31, 0.0)) is None and at() is None          # within a body's radius of it, no farther
    assert at((5.05, 0.2), (5.05, -0.1), (5.2, 0.0)) == 2                 # the nearest of several
    assert at((5.05, 0.1), (5.05, -0.1)) == 1                             # and of two as near, the lower number
    lay = layout(HERE, [(4.0, 1.0), (6.0, 0.0), (6.0, 0.6)], needed=2)    # the target point is marked on person 2
    assert solver.target_person(lay.observation, lay.target, lay.people, TH) == 2
    assert solver.target_person(lay.observation, target_at(6.0, 3.0), lay.people, TH) is None


def test_the_target_point_is_the_centre_of_the_target_person_or_the_ground_under_the_marked_point():
    spots = [(4.0, 0.0), (4.5, 0.0), (3.0, 0.0)]                          # 1.0, 0.5 and 2.0 m from a person at (5, 0)
    person_is_target = at_points(spots, [(5.0, 0.0)], [], target=(4.8, 0.0))
    assert person_is_target.target_xy == near((5.0, 0.0))                 # not (4.8, 0), the side of the body that is seen
    assert person_is_target.keys["preference_m"] == near([0.0, 0.5, 1.0])            # the approach distance is measured to it
    thing = at_points(spots, [(5.4, 0.0)], [], target=(4.8, 0.0))         # that person stands 0.35 m beyond the moved point
    assert thing.target_xy == near((4.8, 0.0)) and thing.keys["preference_m"] == near([0.2, 0.7, 0.8])


def test_a_shared_space_reads_people_and_needs_two_of_them():
    assert read(ask(SHARED, 1, 2), ask(SHARED, 1, 2, 2, 9), ask(SHARED, 3, 1, target=True, between=(0.0, 0.1))) == [USED] * 3
    assert read(ask(SHARED, 1), ask(SHARED), ask(SHARED, 1, 1), ask(SHARED, 1, 9)) == [(NOT, TOO_FEW)] * 4
    assert read(ask(SHARED, 8, 9)) == [(NOT, UNKNOWN)]


def test_a_view_reads_people_and_the_target_and_needs_one_who_gives_a_direction():
    assert read(ask(VIEW, 1), ask(VIEW, 1, 2), ask(VIEW, 3), ask(VIEW, 2, target=True), ask(VIEW, 2, 3, target=True)) == [USED] * 5
    # Straight ahead a person without a heading gives no direction, and towards the target the target person gives none.
    assert read(ask(VIEW, 2), ask(VIEW, 3, target=True), ask(VIEW), ask(VIEW, target=True)) == [(NOT, NO_DIRECTION)] * 4
    assert read(ask(VIEW, 7, target=True)) == [(NOT, UNKNOWN)]


def test_coming_after_reads_people_and_needs_one_who_is_not_the_target():
    assert read(ask(AFTER, 1, 2), ask(AFTER, 3, 1), ask(AFTER, 2, target=True, between=(5.0, 6.0))) == [USED] * 3
    assert read(ask(AFTER, 3), ask(AFTER), ask(AFTER, 3, 3)) == [(NOT, ONLY_TARGET)] * 3
    assert read(ask(AFTER, 9)) == [(NOT, UNKNOWN)]


def test_a_distance_reads_people_or_else_the_target_and_a_range_that_does_not_start_too_far():
    inf, nan = math.inf, math.nan
    assert read(ask(WITHIN, 1, between=(0.5, 1.5)), ask(WITHIN, target=True, between=(0.5, 1.5)),
                ask(WITHIN, 1, 2, target=True, between=(0.0, 0.0)), ask(WITHIN, 1, between=(2.99, 9.0)),
                ask(WITHIN, 1, between=(4.0, 1.0))) == [USED] * 5         # at least above at most: they are exchanged
    assert read(ask(WITHIN, 1), ask(WITHIN, 1, between=(-0.1, 1.0)), ask(WITHIN, 1, between=(0.5, inf)),
                ask(WITHIN, target=True, between=(nan, 1.0)), ask(WITHIN)) == [(NOT, NO_RANGE)] * 5
    assert read(ask(WITHIN, 1, between=(3.0, 4.0)), ask(WITHIN, target=True, between=(5.0, 7.0)),
                ask(WITHIN, 1, between=(80.0, 120.0)), ask(WITHIN, 1, between=(4.0, 3.0)),
                ask(WITHIN, between=(3.0, 3.0))) == [(NOT, "range too far")] * 5
    assert read(ask(WITHIN, 1, between=(2.0, 2.5)), th=Thresholds(max_keep_m=2.0)) == [(NOT, "range too far")]
    assert read(ask(WITHIN, between=(0.5, 1.5))) == [(NOT, NEITHER)]
    assert read(ask(WITHIN, 9, target=True, between=(2.0, 2.5))) == [(NOT, UNKNOWN)]           # not read as about the target


def test_being_visible_reads_people_and_needs_one_with_a_heading():
    assert read(ask(VISIBLE, 1), ask(VISIBLE, 2, 1), ask(VISIBLE, 3, target=True)) == [USED] * 3
    assert read(ask(VISIBLE, 2), ask(VISIBLE), ask(VISIBLE, target=True)) == [(NOT, NO_HEADING)] * 3
    assert read(ask(VISIBLE, 9, 8)) == [(NOT, UNKNOWN)]


def test_the_first_usable_face_stands_and_the_constraints_past_the_bound_are_cut():
    later = ("superseded", "an earlier face stands")
    assert read(ask(FACE), ask(FACE, 9), ask(FACE, target=True), ask(FACE, 1), ask(FACE), ask(FACE, 9)) == [
        (NOT, NEITHER), (NOT, UNKNOWN), USED, later, later, (NOT, UNKNOWN)]
    assert read(ask(FACE, 2), ask(FACE, 1, 2)) == [USED, later]           # a person without a heading can be faced
    answer = [ask(SHARED, 1, 2)] * 11 + [ask(SHARED, 1), ask(SHARED, 1, 2), ask(FACE)]
    assert read(*answer, th=TH) == [USED] * 11 + [(NOT, TOO_FEW)] + [("cut", "beyond the first 12")] * 2
    assert read(ask(SHARED, 1, 2), ask(VISIBLE, 1), th=Thresholds(max_constraints=1)) == [USED, ("cut", "beyond the first 1")]
    assert read() == []


# ------------------------------------------------------------------------------------------- each field, at hand-picked spots
def test_own_space_is_a_standoff_round_every_person_and_the_first_two_keys_are_its_largest_and_its_sum():
    spots = [(0.75, 0.0), (0.5, 0.0), (0.6, 0.0), (0.6, 0.8), (3.0, 0.0)]
    worked = at_points(spots, [(0.0, 0.0), (1.5, 0.0)], [], target=(9.0, 9.0))
    assert worked.own_space[1] == near([0.25, 0.5, 0.4, 0.0, 0.0]) and worked.own_space[2] == near([0.25, 0.0, 0.1, 0.0, 0.0])
    assert worked.keys["excluded_m"] == near([0.25, 0.5, 0.4, 0.0, 0.0])
    assert worked.keys["excluded_total_m"] == near([0.5, 0.5, 0.5, 0.0, 0.0])
    wider = at_points(spots, [(0.0, 0.0), (1.5, 0.0)], [], target=(9.0, 9.0), th=Thresholds(standoff_m=1.2))
    assert wider.own_space[1] == near([0.45, 0.7, 0.6, 0.2, 0.0])


def shared(spots, people, *listed):
    """The violation of a shared space of the listed people, at each spot."""
    return at_points(spots, people, [ask(SHARED, *listed)], target=(9.0, 9.0)).violations[0]


def test_the_space_between_two_people_is_the_rhombus_of_what_each_sees_of_the_other():
    # Two people 2 m apart: the rhombus is 2 tan 35 = 1.40 m wide at the middle, its corners at (0, +-0.70).
    two = [(-1.0, 0.0), (1.0, 0.0)]
    spots = [(0.0, 0.0), (0.0, 1.0), (0.0, 1.1), (-1.5, 0.0), (-1.2, 0.0), (-0.5, 0.6), (0.0, -0.9)]
    inside = 0.3 + SIN35                           # the middle is sin 35 from each side, and a robot's radius is kept
    off_side = 0.3 - (0.6 * COS35 - 0.5 * SIN35)   # (-0.5, 0.6) lies 0.205 m outside the side that leaves (-1, 0)
    expected = [inside, 0.3 - (1.0 - TAN35), 0.0, 0.0, 0.1, off_side, 0.3 - (0.9 - TAN35)]
    assert expected == near([0.8736, 0.0002, 0.0, 0.0, 0.1, 0.0953, 0.1002], abs=1e-4)
    assert shared(spots, two, 1, 2) == near(expected) and shared(spots, two, 2, 1, 1) == near(expected)
    # People on one line have a hull of one side: the rhombus between its two ends.
    assert shared(spots, [(-1.0, 0.0), (0.2, 0.0), (1.0, 0.0)], 1, 2, 3) == near(expected)
    # A third who stands 5 cm off the line between two others leaves their rhombus where it was.
    assert shared(spots, [(-1.0, 0.0), (1.0, 0.0), (0.0, 0.05)], 1, 2, 3)[[1, 2, 6]] == near(expected[1:3] + expected[6:])
    wide = at_points(spots, two, [ask(SHARED, 1, 2)], target=(9.0, 9.0), th=Thresholds(view_half_angle_deg=45.0))
    assert wide.violations[0][:3] == near([0.3 + math.sin(math.radians(45.0)), 0.3, 0.2])    # its corner is now at (0, 1)


def test_the_space_between_several_people_is_their_hull_with_a_rhombus_on_every_side():
    three = [(-1.0, 0.0), (1.0, 0.0), (0.0, 2.0)]
    spots = [(0.0, 1.0), (0.0, -0.5), (0.0, -1.2), (0.0, -0.9), (3.0, 1.0)]
    in_hull = 0.3 + 1.0 / math.sqrt(5.0)           # (0, 1) is 0.447 m from each of the two long sides
    in_rhombus = 0.3 + (SIN35 - 0.5 * COS35)       # (0, -0.5) is outside the hull, 0.164 m inside the rhombus of its base
    assert shared(spots, three, 1, 2, 3) == near([in_hull, in_rhombus, 0.0, 0.3 - (0.9 - TAN35), 0.0])
    assert (in_hull, in_rhombus) == near((0.7472, 0.4640), abs=1e-4)
    assert shared(spots, three + [(0.0, 0.7)], 1, 2, 3, 4) == near(shared(spots, three, 1, 2, 3))    # one inside the hull adds nothing


def view(spots, people, *listed, target=None):
    """The violation of a view of the listed people, at each spot: towards the target's point when one is given."""
    constraint = ask(VIEW, *listed, target=target is not None)
    return at_points(spots, people, [constraint], target=(9.0, 9.0) if target is None else target).violations[0]


def test_a_view_is_a_sector_before_each_person_and_ends_at_what_is_watched():
    watcher = [(0.0, 0.0, 0.0)]                    # at the origin, looking along +x
    spots = [(2.0, 0.0), (0.0, 1.0), (1.0, 0.9), (-0.2, 0.0), (11.0, 0.0), (1.0, -0.9)]
    beside = 0.3 - (0.9 * COS35 - SIN35)           # (1, 0.9) is 42 degrees off the heading, 0.164 m outside the sector
    ahead = [0.3 + 2.0 * SIN35, 0.0, beside, 0.1, 0.3 + 11.0 * SIN35, beside]
    assert ahead == near([1.4472, 0.0, 0.1363, 0.1, 6.6093, 0.1363], abs=1e-4)
    assert view(spots, watcher, 1) == near(ahead)                         # straight ahead of them it has no far end
    # Towards the target the sector reaches as far as the target's point: here 3 m along +x, whatever the heading.
    spots = [(2.0, 0.0), (3.2, 0.0), (4.0, 0.0), (0.0, 1.0), (2.9, 0.0)]
    to_target = [0.3 + 1.0, 0.1, 0.0, 0.0, 0.3 + 0.1]
    assert view(spots, watcher, 1, target=(3.0, 0.0)) == near(to_target)
    assert view(spots, [(0.0, 0.0, 2.0)], 1, target=(3.0, 0.0)) == near(to_target)
    assert view(spots, [(0.0, 0.0)], 1, target=(3.0, 0.0)) == near(to_target)        # nor does it take a heading
    # Of several people the nearest zone counts, and one without a heading gives none.
    spots = [(2.0, 0.0), (2.0, 3.0), (1.0, 2.0)]
    pair = [(0.0, 0.0, 0.0), (0.0, 3.0, 0.0), (1.0, 1.0)]
    assert view(spots, pair, 1, 2, 3) == near([0.3 + 2.0 * SIN35, 0.3 + 2.0 * SIN35, 0.3 - (COS35 - SIN35)])
    wide = at_points(spots[:1], watcher, [ask(VIEW, 1)], target=(9.0, 9.0), th=Thresholds(view_half_angle_deg=45.0))
    assert wide.violations[0] == near([0.3 + math.sqrt(2.0)])


def test_people_who_look_at_one_another_keep_the_space_between_them_and_no_more():
    spots = [(1.5, 0.0), (4.0, 0.0), (-1.0, 0.0), (3.4, 0.0)]
    facing = [(0.0, 0.0, 0.0), (3.0, 0.0, math.pi)]                       # 3 m apart, each looking at the other
    between = 0.3 + 1.5 * SIN35                    # the middle is 0.86 m from the sides of either sector
    assert view(spots, facing, 1, 2) == near([between, 0.0, 0.0, 0.0])    # each zone ends at the other person
    one_way = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0)]                          # the second looks away: no end to the first's zone
    assert view(spots, one_way, 1, 2) == near([between, 0.3 + 4.0 * SIN35, 0.0, 0.3 + 3.4 * SIN35])
    assert view(spots, facing, 1) == near([between, 0.3 + 4.0 * SIN35, 0.0, 0.3 + 3.4 * SIN35])     # the other is not listed
    aside = [(0.0, 0.0, 0.0), (3.0, 2.5, math.pi)]                        # faced back from outside the first's zone
    assert view(spots[:2], aside, 1, 2)[1] == near(0.3 + 4.0 * SIN35)


def test_a_distance_is_violated_by_how_far_a_place_is_outside_its_range():
    spots = [(0.4, 0.0), (1.5, 0.0), (2.5, 0.0), (1.0, 0.0), (2.0, 0.0)]
    people = [(0.0, 0.0), (3.0, 0.0)]
    from_one = at_points(spots, people, [ask(WITHIN, 1, between=(1.0, 2.0))], target=(9.0, 0.0))
    assert from_one.violations[0] == near([0.6, 0.0, 0.5, 0.0, 0.0])
    exchanged = at_points(spots, people, [ask(WITHIN, 1, between=(2.0, 1.0))], target=(9.0, 0.0))
    assert exchanged.violations[0] == near([0.6, 0.0, 0.5, 0.0, 0.0])
    from_both = at_points(spots, people, [ask(WITHIN, 1, 2, target=True, between=(1.0, 2.0))], target=(9.0, 0.0))
    assert from_both.violations[0] == near([0.6 + 0.6, 0.0, 0.5 + 0.5, 0.0, 0.0])    # each listed person; the target is not read
    # A preference is added to the approach distance, a standoff from the target's point, here at (9, 0) ...
    assert from_one.keys["preference_m"] == near([0.6 + 7.6, 6.5, 0.5 + 5.5, 7.0, 6.0])
    # ... unless a range is about the target, or about the person who is the target: that is then the only distance.
    about_thing = at_points([(4.6, 0.0), (3.5, 0.0), (2.5, 0.0)], people, [ask(WITHIN, target=True, between=(1.0, 2.0))], target=(5.0, 0.0))
    assert about_thing.violations[0] == near([0.6, 0.0, 0.5]) and about_thing.keys["preference_m"] == near([0.6, 0.0, 0.5])
    about_person = at_points(spots, people, [ask(WITHIN, 1, between=(1.0, 2.0))], target=(0.2, 0.0), robot=(5.0, 0.0))
    assert about_person.target_xy == near((0.0, 0.0)) and about_person.keys["preference_m"] == near([0.6, 0.0, 0.5, 0.0, 0.0])


def test_being_visible_is_violated_by_the_angle_behind_the_shoulder_and_by_a_body_in_the_way():
    spots = [(1.0, 0.0), (0.5, math.sqrt(0.75)), (0.0, 1.0), (-1.0, 0.0), (-3.0, 0.0), (0.0, -2.0)]
    alone = at_points(spots, [(0.0, 0.0, 0.0)], [ask(VISIBLE, 1)], target=(9.0, 9.0))
    behind = math.radians(120.0)                   # straight behind is 120 degrees past what is seen without turning round
    assert alone.violations[0] == near([0.0, 0.0, math.radians(30.0), behind, behind, math.radians(30.0)])
    assert alone.violations[0][3] == near(2.0944, abs=1e-4)               # a standoff times the angle: closing in changes nothing
    narrow = at_points(spots, [(0.0, 0.0, 0.0)], [ask(VISIBLE, 1)], target=(9.0, 9.0), th=Thresholds(visible_half_angle_deg=30.0))
    assert narrow.violations[0][:3] == near([0.0, math.radians(30.0), math.radians(60.0)])
    # Every other detected person whose body is in the line of sight adds what of their radius reaches into it.
    spots = [(2.0, 0.0), (2.0, 1.0), (0.5, 0.0), (2.0, 0.2)]
    between = at_points(spots, [(0.0, 0.0, 0.0), (1.0, 0.1), (1.0, -0.2)], [ask(VISIBLE, 1)], target=(9.0, 9.0))
    assert 0.8 / math.sqrt(5.0) > 0.25             # (1, 0.1) is 0.358 m from the line of sight to (2, 1): not in it
    assert between.violations[0] == near([0.15 + 0.05, 0.0, 0.0, 0.25])   # and on the line to (2, 0.2), which passes the other by 0.30 m
    four = [(0.0, 0.0, 0.0), (4.0, 0.0, math.pi), (1.0, 0.1), (3.0, -0.1)]
    both = at_points(spots[:1], four, [ask(VISIBLE, 1, 2)], target=(9.0, 9.0))
    assert both.violations[0] == near([0.15 + 0.15])                      # for each listed person, their own line of sight


def test_what_a_kind_does_not_read_changes_nothing():
    spots = [(0.3, 0.2), (2.0, 0.5), (1.4, -1.0)]
    people = [(0.0, 0.0, 0.0), (3.0, 0.0, math.pi), (1.0, 2.0)]
    for kind, listed in ((SHARED, (1, 2)), (VIEW, (1, 2)), (VISIBLE, (1, 2)), (FACE, (1,))):
        plain = at_points(spots, people, [ask(kind, *listed)], target=(9.0, 0.0))
        dressed = at_points(spots, people, [ask(kind, *listed, 7, listed[0], target=True, between=(0.0, 0.1))], target=(9.0, 0.0))
        if kind != VIEW:                           # a view reads the target
            assert all(np.array_equal(plain.keys[name], dressed.keys[name]) for name in plain.keys), kind
        plain_range = at_points(spots, people, [ask(kind, *listed, between=(0.0, 0.1))], target=(9.0, 0.0))
        assert all(np.array_equal(plain.keys[name], plain_range.keys[name]) for name in plain.keys), kind


# --------------------------------------------------------------------------------------------------------------- the order
def test_the_order_compares_key_by_key_with_a_cell_of_slack_at_the_place_in_turn_and_the_preferences():
    def first(among=None, **columns):
        names = ("excluded_m", "excluded_total_m", "turn_m", "face_m", "preference_m", "standard_m", "robot_m")
        keys = {name: np.array(columns.get(name, [0.0] * 4), dtype=np.float64) for name in names
                if name not in ("turn_m", "face_m") or name in columns}       # a place has one of the two, or neither
        return solver._first(keys, np.ones(4, dtype=bool) if among is None else np.array(among), 0.05)
    assert first() == 0                                                   # equal in every key: the first in the grid's order
    assert first(among=[False, False, True, True]) == 2 and first(among=[False] * 4) is None
    # 1. The largest violation of any one exclusion comes before their sum.
    assert first(excluded_m=[0.2, 0.1, 0.1, 0.3], excluded_total_m=[0.2, 0.9, 0.8, 0.3]) == 2
    assert first(excluded_m=[0.101, 0.1, 0.2, 0.3], robot_m=[0.0, 9.0, 9.0, 9.0]) == 1          # with no slack at all
    # 2. Then their sum, before everything else.
    assert first(excluded_total_m=[0.3, 0.2, 0.2, 0.3], turn_m=[0.0, 5.0, 5.0, 0.0], preference_m=[0.0, 3.0, 2.0, 0.0]) == 2
    # 3. The place in turn: the cells within one cell's width of the least stay in, and no others.
    assert first(turn_m=[0.30, 0.34, 0.36, 0.35], preference_m=[2.0, 1.0, 0.0, 1.5]) == 1
    assert first(turn_m=[0.30, 0.351, 0.36, 0.35], preference_m=[2.0, 1.0, 0.0, 1.5]) == 3
    # 4. How far behind the seen face, in the place of 3: after the exclusions, before the preferences, with no slack.
    assert first(excluded_total_m=[0.1, 0.0, 0.0, 0.1], face_m=[0.0, 0.3, 0.2, 0.0], preference_m=[0.0, 0.0, 9.0, 0.0]) == 2
    assert first(face_m=[0.001, 0.0, 0.0, 0.002], preference_m=[0.0, 3.0, 2.0, 0.0]) == 2
    # 5. The preferences with a cell of slack, as 3 has it, before the standard goal.
    assert first(preference_m=[0.50, 0.54, 0.56, 0.55], standard_m=[2.0, 1.0, 0.0, 1.5]) == 1
    assert first(turn_m=[0.0] * 4, preference_m=[0.50, 0.54, 0.56, 0.55], standard_m=[2.0, 1.0, 0.0, 1.5]) == 1
    assert first(face_m=[0.0] * 4, preference_m=[0.50, 0.54, 0.56, 0.55], standard_m=[2.0, 1.0, 0.0, 1.5]) == 1
    # 6. The standard goal, to the millimetre; 7. the robot; then the order of the grid.
    assert first(standard_m=[1.001, 1.0, 1.0, 1.002], robot_m=[0.0, 3.0, 2.0, 0.0]) == 2
    assert first(standard_m=[1.0, 1.0, 1.0, 1.0], robot_m=[4.0, 3.0, 3.0, 5.0]) == 1


def test_with_nobody_about_the_place_is_the_standard_goal():
    solved = layout(HERE, [], thing=(5.0, 0.0)).solve()                   # the standard goal stands on a cell's centre
    assert solved.pose_world[:2] == near((4.0, 0.0)) and solved.pose_robot[:2] == near((4.0, 0.0))
    assert solved.note.keys.model_dump() == dict(excluded_m=0.0, excluded_total_m=0.0, turn_m=None, face_m=None,
                                                 preference_m=0.0, standard_m=0.0)       # an open floor shows no face
    assert solved.note.target_distance_m == near(1.0) and solved.uses == [] and solved.note.not_ahead_of == []
    for robot, thing in ((HERE, (5.02, 0.31)), (moved(HERE, 1.1, (2.0, -3.0)), (4.0, 1.0)), (HERE, (0.0, -7.3))):
        lay = layout(robot, [], thing=thing)
        place, standard = lay.place(), lay.standard.pose_world
        assert math.hypot(place[0] - standard[0], place[1] - standard[1]) <= 0.05 * math.sqrt(0.5) + 1e-9      # to a cell
    askew = layout(HERE, [], thing=(5.0, 0.0), standard=(5.4, 0.9))       # a standard goal before a face seen at an angle
    keys = askew.solve().note.keys                                        # that cell is 0.985 m from the thing: within the slack
    assert askew.place() == near((5.4, 0.9)) and (keys.preference_m, keys.standard_m) == (0.015, 0.0)


def test_a_person_on_the_standard_goal_moves_the_place_round_them_key_by_key():
    # The target is a thing at (5, 0) and someone stands on the standard goal, (4, 0). Own space leaves the cells a
    # metre or more from them. The approach distance is least, 0, at (5, +-1); the cells within 0.05 m of that stay
    # in, a ring 0.95 to 1.05 m round the thing. Of those the nearest the standard goal are (4.55, +-0.85), 1.012 m
    # from it and 0.962 m from the thing. They are as far from the robot, and the lower row is on the robot's right.
    solved = layout(HERE, [(4.0, 0.0)], thing=(5.0, 0.0)).solve()
    assert solved.pose_world[:2] == near((4.55, -0.85))
    assert solved.note.keys.model_dump() == dict(excluded_m=0.0, excluded_total_m=0.0, turn_m=None, face_m=None,
                                                 preference_m=0.038, standard_m=1.012)
    assert solved.note.target_distance_m == near(math.hypot(0.45, 0.85))


def test_cells_equal_in_every_key_are_told_apart_by_the_robot_then_the_row_then_the_column():
    # A thing 3 m ahead with the standard goal on it: the four cells 0.95 m from it along the grid's axes are equal up
    # to the standard goal, and the robot is nearest (2.05, 0).
    assert layout(HERE, [], thing=(3.0, 0.0), standard=(3.0, 0.0)).place() == near((2.05, 0.0))
    # The thing and the standard goal where the robot stands: the four are as far from the robot too. The lowest row
    # is the one farthest to the robot's right ...
    assert layout(HERE, [], thing=(0.0, 0.0), standard=(0.0, 0.0)).place() == near((0.0, -0.95))
    # ... and with no floor there, two are left in one row: the lowest column is the one farthest behind the robot.
    no_right = layout(HERE, [], thing=(0.0, 0.0), standard=(0.0, 0.0), floor=lambda ahead, left: left > -0.5)
    assert no_right.place() == near((-0.95, 0.0))
    turned = moved(HERE, math.pi / 2, (2.0, 1.0))                         # the rows and columns are the robot's, not the world's
    assert layout(turned, [], thing=(2.0, 1.0), standard=(2.0, 1.0)).place() == near((2.95, 1.0))


def test_the_solver_gives_the_same_answer_twice_and_changes_nothing_it_is_given():
    lay = layout(HERE, [(4.0, 1.0, math.pi), (4.0, -1.0, math.pi / 2), (6.0, 0.0, math.pi)], needed=3)
    answer = [ask(SHARED, 1, 2, 3), ask(VISIBLE, 3), ask(WITHIN, 3, between=(1.5, 0.5)), ask(FACE, 3)]
    depth, people, kept = lay.observation.depth.copy(), [p.model_copy() for p in lay.people], [c.model_copy(deep=True) for c in answer]
    first, second = lay.solve(*answer), lay.solve(*answer)
    assert first == second and first.pose_world == second.pose_world
    assert np.array_equal(lay.observation.depth, depth) and lay.people == people and answer == kept
    assert [use.constraint for use in first.uses] == kept                # each constraint as it was written, in its order
    with pytest.raises(ValueError, match="standard goal"):
        solver.solve(lay.observation, lay.target, lay.people, answer, GoalResult(status="lift_failed"), TH)


# ----------------------------------------------------------------------------------------------------------- the frame
THING = Box(cx=6.3, cy=0.4, yaw=math.pi, sx=0.6, sy=1.6, sz=1.1)         # its face 6.0 m ahead of the robot, towards it
STANDING = [(4.6, -0.9, math.radians(60.0)), (4.9, 0.9, math.radians(-100.0)), (3.4, 1.6, math.radians(-30.0))]


def rendered(robot=HERE, turn=0.0, shift=(0.0, 0.0), needed=None):
    """The same picture of a thing and three people wherever the world frame lies, and what the solver is given
    with it: the people where they stand in that frame, the target point lifted from the picture, and the
    standard goal of the first spec for it."""
    figures = [Figure(x, y, yaw) for x, y, yaw in STANDING]
    observation = replace(mounted(boxes=[THING], figures=figures), camera=mounted_camera(robot), robot=robot)
    people = [person(n, *moved_xy((x, y), turn, shift), yaw + turn) for n, (x, y, yaw) in enumerate(STANDING, 1)]
    marked = (6.0, 0.4, 0.8) if needed is None else (STANDING[needed - 1][0] - 0.2, STANDING[needed - 1][1], 1.1)
    u, v, _ = project(mounted_camera(), marked)                           # the pixel is the same in every frame
    world = lift(observation.depth, observation.camera, u, v, TH.lift_window)
    target = GroundedTarget(point=Point(x=500, y=500), pixel=(u, v), world=tuple(float(c) for c in world))
    return observation, target, people, standard_goal(observation, target, TH)


@pytest.mark.parametrize("needed, answer", [
    (None, []),
    (None, [ask(VIEW, 1, 2, target=True)]),        # the thing has a face in view, and two are stated as watching it
    (None, [ask(VIEW, 1, 2, target=True), ask(WITHIN, target=True, between=(1.0, 2.0)), ask(FACE, target=True)]),
    (1, [ask(SHARED, 1, 2), ask(VISIBLE, 1), ask(FACE, 1)]),
    (2, [ask(VIEW, 3), ask(WITHIN, 2, between=(1.2, 1.6))])])
def test_the_same_picture_gives_the_same_place_wherever_the_world_frame_lies(needed, answer):
    observation, target, people, standard = rendered(needed=needed)
    here = solver.solve(observation, target, people, answer, standard, TH)
    assert math.hypot(*here.pose_world[:2]) > 2.5                         # it went somewhere
    for degrees, shift in ((0.0, (0.013, 0.031)), (17.0, (0.41, -0.27)), (90.0, (0.02, 0.04)), (-140.0, (30.1, -12.3))):
        turn = math.radians(degrees)
        observation, target, people, standard = rendered(moved(HERE, turn, shift), turn, shift, needed)
        there = solver.solve(observation, target, people, answer, standard, TH)
        expected = moved_xy(here.pose_world[:2], turn, shift)
        cell = TH.grid_resolution_m * math.sqrt(2.0)                      # the standard goal is fitted in the world frame
        assert math.hypot(there.pose_world[0] - expected[0], there.pose_world[1] - expected[1]) <= cell + 1e-9
        assert math.hypot(there.pose_robot[0] - here.pose_robot[0], there.pose_robot[1] - here.pose_robot[1]) <= cell + 1e-9
        assert there.note.keys.model_dump() == pytest.approx(here.note.keys.model_dump(), abs=cell)
        assert [use.status for use in there.uses] == [use.status for use in here.uses]


# -------------------------------------------------------------------------------------- a shared space, laid out on a floor
def apart(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def needing_the_first(spots, robot):
    """People on an open floor, the robot sent to the first of them and looking at them."""
    return layout(looking(robot, spots[0]), spots, needed=1)


@pytest.mark.parametrize("gap, robot, recorded", [
    (1.2, (5.1, 1.3), (0.03, 0.83)), (2.0, (5.5, 1.3), (-0.38, 0.83)), (3.0, (6.0, 1.3), (-0.88, 0.83)),
    (3.0, (0.3, 5.5), (-0.97, 0.88)), (4.0, (6.5, 1.3), (-1.38, 0.83))])
def test_beside_two_who_attend_to_one_another_the_robot_stands_well_off_the_line_between_them(gap, robot, recorded):
    pair = [(-gap / 2, 0.0, 0.0), (gap / 2, 0.0, math.pi)]               # each looks at the other
    solved = needing_the_first(pair, robot).solve(ask(SHARED, 1, 2), ask(VISIBLE, 1), ask(FACE, 1))
    place, keys = solved.pose_world[:2], solved.note.keys
    off_line = math.degrees(math.atan2(abs(place[1]), place[0] + gap / 2))
    assert off_line >= 45.0 and (keys.excluded_m, keys.excluded_total_m) == (0.0, 0.0)       # 53 to 59 degrees, as recorded
    assert 1.0 - 5e-4 <= apart(place, pair[0]) <= 1.1 and apart(place, pair[1]) >= 1.0 - 5e-4
    assert apart(place, recorded) <= 0.08          # where the reference solver stood, its grid laid in the world frame


@pytest.mark.parametrize("robot", [(6.0, -1.3), (0.3, -5.5)])
def test_a_third_member_beside_the_line_between_two_leaves_the_place_where_it_was(robot):
    pair = [(-1.5, 0.0, 0.0), (1.5, 0.0, math.pi)]
    alone = needing_the_first(pair, robot).place(ask(SHARED, 1, 2), ask(VISIBLE, 1), ask(FACE, 1))
    for third in ((0.0, 0.05), (0.0, 0.3), (0.0, 0.8)):                   # on the far side of the line from the robot
        three = pair + [(*third, facing(third, (0.0, -0.5)))]
        place = needing_the_first(three, robot).place(ask(SHARED, 1, 2, 3), ask(VISIBLE, 1), ask(FACE, 1))
        assert apart(place, alone) <= TH.grid_resolution_m * math.sqrt(2.0) + 1e-9           # the same to a cell


@pytest.mark.parametrize("robot, recorded", [((-5.3, 0.5), (0.98, 1.03)), ((0.5, -5.0), (0.98, -1.07))])
def test_a_group_given_both_as_a_shared_space_and_as_a_view_keeps_the_robot_beside_the_one_it_needs(robot, recorded):
    four = [(0.8 * math.cos(turn), 0.8 * math.sin(turn), turn + math.pi) for turn in (0.0, math.pi / 2, math.pi, -math.pi / 2)]
    lay = needing_the_first(four, robot)           # four in a circle, each looking at its middle
    both = lay.solve(ask(SHARED, 1, 2, 3, 4), ask(VIEW, 1, 2, 3, 4), ask(VISIBLE, 1), ask(FACE, 1))
    place = both.pose_world[:2]
    assert apart(place, four[0]) <= 1.1 and both.note.keys.excluded_m == 0.0 and apart(place, recorded) <= 0.08
    assert place == lay.place(ask(SHARED, 1, 2, 3, 4), ask(VISIBLE, 1), ask(FACE, 1))          # the view adds nothing here:
    assert place == lay.place(ask(VIEW, 1, 2, 3, 4), ask(VISIBLE, 1), ask(FACE, 1))            # each zone ends at who faces back


# ---------------------------------------------------------------------------------------------------- the place in turn
THING_XY = (0.0, 0.0)                              # what the people of this section wait at
FILE3 = [(0.0, 1.2), (0.0, 2.2), (0.0, 3.2)]       # a straight file that leads away from it
FILE4 = FILE3 + [(0.0, 4.2)]
BENT = FILE3 + [(0.9, 3.6), (1.9, 3.7), (2.9, 3.7)]
CLUSTER = [(-0.8, -1.6), (0.4, -2.0), (1.1, -1.5), (-0.2, -2.9)]         # four who stand loosely before it
PAIR_AT_END = [(0.0, 1.2), (0.0, 2.2), (-0.3, 3.2), (0.3, 3.2)]           # the last two side by side
FROM = (3.5, 7.0)                                  # where the robot comes from, unless a test says otherwise


def turn(spots, *answer, robot=FROM, target=THING_XY, th=TH):
    """The place in turn the solver works out for an answer, with the last of the waiting people and who joined."""
    answer = [given if not isinstance(given, tuple) else ask(AFTER, *given) for given in answer]
    return at_points([(0.0, 0.0)], spots, answer, target=target, robot=robot, th=th).turn


def waiting(spots, robot=FROM, look_at=(0.0, 2.5), thing=THING_XY, needed=None):
    return layout(looking(robot, look_at), spots, thing=None if needed else thing, needed=needed)


def behind(place, last, before):
    """How far a place is behind the last of a file, along the line from the one before, and how far to the side."""
    along = np.subtract(last, before) / math.hypot(last[0] - before[0], last[1] - before[1])
    offset = np.subtract(place, last)
    return float(offset @ along), float(abs(offset[0] * along[1] - offset[1] * along[0]))


def test_the_place_in_turn_is_a_standoff_beyond_the_last_on_the_line_from_the_one_before():
    straight = turn(FILE3, (1, 2, 3))
    assert (straight.last, straight.joined) == (3, []) and straight.place == near((0.0, 4.2))
    assert straight.last_from_target == near(3.2) and turn(FILE3, (3, 1, 2, 2)).place == near((0.0, 4.2))    # in any order
    bent = turn(BENT, (1, 2, 3, 4, 5, 6))          # the last is 1.0 m past the one before, along +x
    assert bent.last == 6 and bent.place == near((3.9, 3.7))
    cluster = turn(CLUSTER, (1, 2, 3, 4))          # the farthest is 4, and 2 is their nearest neighbour, 1.08 m ahead of them
    assert cluster.last == 4 and cluster.place == near((-0.2 - 0.6 / math.sqrt(1.17), -2.9 - 0.9 / math.sqrt(1.17)))
    assert cluster.place == near((-0.755, -3.732), abs=1e-3)
    assert turn(FILE3, (1, 2, 3), th=Thresholds(standoff_m=1.5)).place == near((0.0, 4.7))
    # Nobody before the last: the line starts from the target's point.
    assert turn(FILE3, (3,)).place == near((0.0, 4.2)) and turn([(3.0, 4.0)], (1,)).place == near((3.6, 4.8))
    # And when the last stands on that point too, it runs from them towards the robot, here along (0.6, 0.8).
    on_it = turn([(0.03, 0.02)], (1,), robot=(3.03, 4.02))
    assert on_it.last == 1 and on_it.place == near((0.63, 0.82))
    assert turn([(0.05, 0.0)], (1,), robot=(3.03, 4.02)).place == near((1.05, 0.0))            # 0.05 m off it: from the point


def test_who_is_last_and_who_is_before_them_is_read_from_distances():
    # The last two are equally far from the thing: the nearer the robot is the last, and the other stands beside
    # them, not before. The line comes from the thing, through whichever is taken.
    along = (0.3 / math.hypot(0.3, 3.2), 3.2 / math.hypot(0.3, 3.2))
    right, left = turn(PAIR_AT_END, (1, 2, 3, 4), robot=(3.5, 7.0)), turn(PAIR_AT_END, (1, 2, 3, 4), robot=(-3.5, 7.0))
    assert right.last == 4 and right.place == near((0.3 + along[0], 3.2 + along[1]))
    assert left.last == 3 and left.place == near((-0.3 - along[0], 3.2 + along[1]))
    assert turn(PAIR_AT_END, (1, 2, 3, 4), robot=(0.0, 9.0)).last == 3    # as near the robot too, to the millimetre: the lower number
    # Within a body's radius of the farthest counts as equally far ...
    close = FILE3[:2] + [(0.2, 2.4)]               # 2.408 m from the thing, the second 2.2 m: 0.21 m nearer
    assert turn(close, (1, 2, 3)).last == 3 and turn(close, (1, 2, 3), robot=(-3.5, 0.5)).last == 2
    assert turn(close, (1, 2, 3)).place == near((0.2 + 0.2 / math.hypot(0.2, 2.4), 2.4 + 2.4 / math.hypot(0.2, 2.4)))
    assert turn(close, (1, 2, 3), robot=(-3.5, 0.5)).place == near((0.0, 3.2))       # either way the line comes from the thing
    # ... and farther than that the nearest neighbour is before the last, wherever the robot is.
    farther = FILE3[:2] + [(0.2, 2.5)]             # 2.508 m from the thing: 0.31 m farther than the second
    for robot in ((3.5, 7.0), (-3.5, 7.0)):
        found = turn(farther, (1, 2, 3), robot=robot)
        assert found.last == 3 and found.place == near((0.2 + 0.2 / math.hypot(0.2, 0.3), 2.5 + 0.3 / math.hypot(0.2, 0.3)))
    # The one before is the last one's nearest neighbour and nobody else: here that neighbour stands beside them.
    row = [(0.0, 1.0), (-1.0, 3.0), (0.0, 3.1)]    # person 1 is well ahead, but 2 is the nearest to 3 and as far as they
    assert turn(row, (1, 2, 3)).last == 3 and turn(row, (1, 2, 3)).place == near((0.0, 4.1))
    slim = Thresholds(person_radius_m=0.1)         # with a narrower body the two of `close` are no longer equally far
    assert turn(close, (1, 2, 3), robot=(-3.5, 0.5), th=slim).last == 3


def test_no_heading_is_read_for_the_place_in_turn():
    plain = waiting(FILE4).solve(ask(AFTER, 1, 2, 3, 4))
    assert apart(plain.pose_world, (0.0, 5.2)) <= 0.1 and plain.note.keys.turn_m <= 0.1
    to_thing = [facing(spot, THING_XY) for spot in FILE4]
    to_robot = [facing(spot, FROM) for spot in FILE4]
    round_about = [heading + math.pi for heading in to_thing]
    for headings in (to_thing,                                           # each looks at the thing
                     to_thing[:2] + to_robot[2:],                        # the last two look at the robot
                     to_robot,                                           # everyone looks at it
                     to_thing[:2] + [round_about[2], None],              # the third has turned round to the fourth
                     [round_about[0], None, None, None]):                # the first has turned round
        people = [(x, y, heading) for (x, y), heading in zip(FILE4, headings)]
        assert turn(people, (1, 2, 3, 4)).place == near((0.0, 5.2))
        assert waiting(people).solve(ask(AFTER, 1, 2, 3, 4)).pose_world[:2] == plain.pose_world[:2]
    # Two side by side at the end of a file, one turned to the other: the place is behind them, not beside.
    for robot, nearer, other in ((FROM, 4, 3), ((-3.5, 7.0), 3, 4)):
        people = [(x, y, facing((x, y), THING_XY)) for x, y in PAIR_AT_END]
        people[nearer - 1] = (*PAIR_AT_END[nearer - 1], facing(PAIR_AT_END[nearer - 1], PAIR_AT_END[other - 1]))
        place = waiting(people, robot).place(ask(AFTER, 1, 2, 3, 4))
        back, aside = behind(place, PAIR_AT_END[nearer - 1], PAIR_AT_END[1])
        assert place == waiting(PAIR_AT_END, robot).place(ask(AFTER, 1, 2, 3, 4)) and 0.8 <= back <= 1.2 and aside <= 0.4


@pytest.mark.parametrize("spots, robot, look_at, in_turn, recorded", [
    (FILE3, (3.5, 7.0), (0.0, 2.5), (0.0, 4.2), (0.08, 4.23)),
    (FILE3, (-4.0, 2.0), (0.0, 2.5), (0.0, 4.2), (-0.07, 4.23)),
    (FILE3, (0.6, 7.6), (0.0, 2.5), (0.0, 4.2), (0.08, 4.23)),
    (BENT, (5.5, 7.0), (1.0, 3.0), (3.9, 3.7), (3.93, 3.62)),
    (BENT, (-3.5, 7.0), (1.0, 3.0), (3.9, 3.7), (3.93, 3.62)),
    (CLUSTER, (2.5, -6.5), (0.0, -1.8), (-0.755, -3.732), (-0.77, -3.72)),
    (CLUSTER, (-4.5, -3.0), (0.0, -1.8), (-0.755, -3.732), (-0.77, -3.72))])
def test_the_robot_stands_on_the_place_in_turn_of_a_straight_file_a_bent_one_and_a_loose_cluster(spots, robot, look_at, in_turn,
                                                                                               recorded):
    lay = waiting(spots, robot, look_at)
    solved = lay.solve(ask(AFTER, *range(1, len(spots) + 1)))
    place, keys = solved.pose_world[:2], solved.note.keys
    assert apart(place, in_turn) <= 0.1 and keys.turn_m == pytest.approx(apart(place, in_turn), abs=1e-3)
    assert apart(place, recorded) <= 0.15          # where the reference solver stood, its grid laid in the world frame:
    #                                                the place may fall a cell to either side of the line
    assert (keys.excluded_m, keys.excluded_total_m) == (0.0, 0.0) and solved.note.joined_waiting == []
    assert min(apart(place, spot) for spot in spots) >= 1.0 - 5e-4                   # a standoff from every one of them,
    assert all(apart(spot, THING_XY) <= apart(place, THING_XY) for spot in spots)    # and nearer the thing than none


def test_coming_after_is_violated_by_being_nearer_the_target_than_the_last_and_several_are_one():
    spots = [(0.5, 1.0), (3.2, 0.0), (0.0, 4.2), (2.0, 2.0), (0.0, 3.3)]
    one = at_points(spots, FILE3, [ask(AFTER, 1, 2, 3)], target=THING_XY, robot=FROM)
    nearer = [3.2 - math.hypot(0.5, 1.0), 0.0, 0.0, 3.2 - math.sqrt(8.0), 0.0]       # the last is 3.2 m from the thing
    assert one.violations[0] == near(nearer) and nearer == near([2.082, 0.0, 0.0, 0.372, 0.0], abs=1e-3)
    assert one.keys["turn_m"] == near([round(apart(spot, (0.0, 4.2)), 3) for spot in spots])     # to the place in turn
    assert one.keys["excluded_m"][[0, 3, 4]] == near([2.082, 0.372, 0.9])            # the last one's own space at (0, 3.3)
    several = at_points(spots, FILE3, [ask(AFTER, 2, 1), ask(FACE, target=True), ask(AFTER, 3, 9)], target=THING_XY, robot=FROM)
    assert several.violations[0] == near(nearer) and several.violations[2] == near(nearer) and several.violations[1] is None
    assert several.turn.place == near((0.0, 4.2)) and all(np.array_equal(one.keys[k], several.keys[k]) for k in one.keys)
    # The person who is the target does not wait for their own turn: the last is measured from their centre.
    at_person = at_points(spots, FILE3, [ask(AFTER, 1, 2, 3)], target=(0.1, 1.38), robot=FROM)
    assert at_person.target_xy == near((0.0, 1.2)) and at_person.turn.last_from_target == near(2.0)
    assert at_person.turn.place == near((0.0, 4.2)) and at_person.violations[0][[0, 3]] == near([2.0 - math.hypot(0.5, 0.2), 0.0])
    assert "turn_m" not in at_points(spots, FILE3, [], target=THING_XY).keys and at_points(spots, FILE3, [], target=THING_XY).turn is None


def test_the_place_in_turn_comes_before_what_the_answer_prefers():
    lay = waiting(FILE3)
    near_thing = [ask(WITHIN, target=True, between=(0.5, 1.5)), ask(FACE, target=True)]
    after = lay.solve(ask(AFTER, 1, 2, 3), *near_thing)                   # asked to wait its turn and to be at the thing
    assert apart(after.pose_world, (0.0, 4.2)) <= 0.1 and after.note.keys.preference_m == pytest.approx(4.23 - 1.5, abs=0.02)
    assert apart(after.pose_world, lay.place(ask(AFTER, 1, 2, 3))) <= 0.08
    alone = lay.solve(*near_thing)                                        # without the turn the same answer goes to the thing,
    assert apart(alone.pose_world, THING_XY) <= 1.5 + 1e-9 and alone.note.keys.turn_m is None       # beside the first person
    assert apart(alone.pose_world, FILE3[0]) >= 1.0 - 5e-4


@pytest.mark.parametrize("spacing", [0.6, 1.0, 1.5, 1.6, 1.9])
def test_a_waiting_person_the_answer_leaves_out_is_taken_as_waiting(spacing):
    spots = [(0.0, 1.0 + spacing * k) for k in range(4)]
    last = spots[3]
    lay = waiting(spots, robot=(3.5, last[1] + 2.5), look_at=spots[1])
    for answer in ([ask(AFTER, 1, 2, 3)],
                   [ask(AFTER, 1, 2, 3), ask(VIEW, 1, 2, 3, 4, target=True)]):       # listed under a view, they are still taken
        found = turn(spots, *answer, robot=(3.5, last[1] + 2.5))
        assert (found.last, found.joined) == (4, [4]) and found.place == near((0.0, last[1] + 1.0))
        solved = lay.solve(*answer)
        back, aside = behind(solved.pose_world[:2], last, spots[2])
        assert solved.note.joined_waiting == [4] and 0.95 <= back <= 1.1 and aside <= 0.1
    assert lay.solve(ask(AFTER, 1, 2, 3, 4)).pose_world == lay.solve(ask(AFTER, 1, 2, 3)).pose_world


def test_someone_has_taken_the_place_in_turn_when_their_own_space_covers_it_and_they_stand_behind_the_last():
    def taken(spot, *answer, spots=FILE3):
        return turn(spots + [spot], (1, 2, 3), *answer)
    on_it = taken((0.1, 4.25))                     # the place in turn was (0, 4.2)
    along = (0.1 / math.hypot(0.1, 1.05), 1.05 / math.hypot(0.1, 1.05))
    assert (on_it.last, on_it.joined) == (4, [4]) and on_it.place == near((0.1 + along[0], 4.25 + along[1]))
    assert on_it.last_from_target == near(math.hypot(0.1, 4.25))          # the robot now comes after them
    assert taken((0.5, 4.7)).joined == [4]         # 0.71 m from it and 18 degrees off the line: behind the last
    assert taken((0.9, 4.0)).joined == []          # 0.92 m from it, but 48 degrees off the line: beside the last
    assert taken((0.0, 5.21)).joined == [] and taken((0.0, 5.19)).joined == [4]      # a standoff from it or more: not on it
    assert taken((0.57, 4.0)).joined == [] and taken((0.55, 4.0)).joined == [4]      # 35.5 and 34.5 degrees off the line
    assert taken((0.0, 2.7)).joined == []          # within reach of it, and ahead of the last
    wider = turn(FILE3 + [(0.9, 4.0)], (1, 2, 3), th=Thresholds(view_half_angle_deg=50.0))
    assert wider.joined == [4]
    # The nearest to it of those who have taken it joins, the lower number of two as near; then the place is worked out again.
    two = turn(FILE3 + [(0.3, 4.6), (-0.3, 4.6)], (1, 2, 3))
    assert two.joined == [4] and two.last == 4 and turn(FILE3 + [(0.3, 4.6), (-0.2, 4.6)], (1, 2, 3)).joined == [5]
    # The person who is the target has not taken it, wherever they stand.
    assert turn(FILE3 + [(0.1, 4.25)], (1, 2, 3), target=(0.2, 4.4)).joined == []
    solved = waiting(FILE3 + [(0.1, 4.25)]).solve(ask(AFTER, 1, 2, 3))
    assert solved.note.joined_waiting == [4] and apart(solved.pose_world, on_it.place) <= 0.12       # the reference stood at (0.28, 5.28)
    assert apart(solved.pose_world, (0.28, 5.28)) <= 0.11 and apart(solved.pose_world, (0.1, 4.25)) >= 1.0 - 5e-4


def three_talking(centre, radius=0.65, turned=0.5):
    spots = [(centre[0] + radius * math.cos(turned + 2 * math.pi * k / 3), centre[1] + radius * math.sin(turned + 2 * math.pi * k / 3))
             for k in range(3)]
    return [(x, y, facing((x, y), centre)) for x, y in spots]


@pytest.mark.parametrize("centre", [(0.4, 6.5), (0.6, 6.3), (-0.5, 6.6)])
def test_people_the_answer_lists_as_attending_to_one_another_are_not_taken_as_waiting(centre):
    file = [(x, y, facing((x, y), THING_XY)) for x, y in FILE4]
    spots = file + three_talking(centre)           # behind the end of the file, the nearest 0.7 to 0.9 m from the place in turn
    assert 0.7 <= min(apart(spot, (0.0, 5.2)) for spot in spots[4:]) <= 0.92
    lay = waiting(spots, robot=(3.5, 9.5), look_at=(0.0, 4.0))
    everyone = ask(AFTER, 1, 2, 3, 4)
    listed = lay.solve(everyone, ask(SHARED, 5, 6, 7))
    back, aside = behind(listed.pose_world[:2], FILE4[3], FILE4[2])
    assert listed.note.joined_waiting == [] and 0.9 <= back <= 1.1 and aside <= 0.45          # a metre behind the last
    assert 1.0 - 5e-4 <= apart(listed.pose_world, FILE4[3]) <= 1.1
    assert turn(spots, everyone, ask(SHARED, 5, 6, 7), robot=(3.5, 9.5)).place == near((0.0, 5.2))
    # Listed nowhere, or in a shared space that cannot be used, or under a view, the nearest of them is taken.
    for answer in ([everyone], [everyone, ask(SHARED, 5, 9), ask(SHARED, 6), ask(SHARED, 7, 7)], [everyone, ask(VIEW, 5, 6, 7)]):
        assert turn(spots, *answer, robot=(3.5, 9.5)).joined[:1] in ([5], [6], [7]), answer
    assert turn(spots, everyone, ask(SHARED, 6, 7), ask(SHARED, 7, 5), robot=(3.5, 9.5)).joined == []     # each in a usable one


def test_two_listed_side_by_side_and_an_unlisted_third_beside_them_leave_the_place_in_turn_free():
    spots = [(1.0, 2.82), (0.0, 3.0), (-0.84, 2.70)]                      # the third is 1.55 m from the place in turn
    found = turn(spots, (1, 2), robot=(0.0, 7.0))
    assert (found.last, found.joined) == (2, []) and found.place == near((0.0, 4.0))
    solved = waiting(spots, robot=(0.0, 7.0), look_at=(0.0, 3.0)).solve(ask(AFTER, 1, 2))
    assert solved.note.joined_waiting == [] and apart(solved.pose_world, (0.0, 4.0)) <= 0.1


def test_three_who_wait_in_an_arc_keep_their_place_in_turn_when_one_of_them_moves_three_centimetres():
    places = []
    for left in ((-1.2, -1.50), (-1.2, -1.53)):    # 1.921 and then 1.944 m from the thing: the others 1.900 and 1.937 m
        spots = [left, (0.0, -1.90), (1.2, -1.52)]
        found = turn(spots, (1, 2, 3), robot=(0.5, -6.5))
        assert found.last == 2 and found.place == near((0.0, -2.9))       # all equally far: the nearest the robot, from the thing
        places.append(waiting(spots, robot=(0.5, -6.5), look_at=(0.0, -1.5)).place(ask(AFTER, 1, 2, 3)))
    assert apart(places[0], places[1]) <= 0.05 + 1e-9 and apart(places[0], (0.0, -2.9)) <= 0.1


# ------------------------------------------------------------------------------- where nothing is free, and what is recorded
def test_where_every_cell_breaks_an_exclusion_the_one_broken_least_is_taken_and_recorded():
    # The robot stands between two people 1.5 m apart, and the only floor is a strip 0.4 m wide across the line between
    # them, and its own disc. Every cell is inside somebody's own space. The cells that break their worst exclusion
    # least are the two on the rim of the disc, 0.3 m ahead of the robot and behind it: 0.808 m from either person.
    between = layout(HERE, [(0.0, 0.75), (0.0, -0.75)], thing=(3.0, 0.0),
                     floor=lambda ahead, left: (np.abs(ahead) <= 0.2 + 1e-9) & (np.abs(left) <= 0.75))
    solved = between.solve()
    short = round(1.0 - math.hypot(0.3, 0.75), 3)
    assert short == 0.192 and solved.pose_world[:2] == near((0.3, 0.0))   # of the two, the one nearer the thing
    assert (solved.note.keys.excluded_m, solved.note.keys.excluded_total_m) == (0.192, 0.384)
    assert solved.note.own_space_broken_m == {1: 0.192, 2: 0.192} and solved.flags.exclusion_broken
    cells, worked = between.worked()
    assert worked.keys["excluded_m"].min() == 0.192 and len(cells) < 200  # no cell breaks its worst exclusion by less
    # The largest violation comes before the sum: amid three people 0.8 m away each, 0.2 m inside every one's own space,
    # is better than 0.3 m inside one. The least sum alone would press the robot against one of them.
    side = 0.8 * math.sqrt(3.0)
    pressed = at_points([(-0.7, 0.0), (side / 2, 0.4)], [(0.0, 0.0), (side, 0.0), (side / 2, 1.2)], [], target=(9.0, 0.0))
    assert pressed.keys["excluded_m"] == near([0.3, 0.2]) and pressed.keys["excluded_total_m"] == near([0.3, 0.6])
    assert pressed.picks[0] == 1


def test_an_exclusion_of_the_answer_that_is_broken_is_recorded_with_its_violation():
    # Someone 2 m behind the robot looks along its heading, and the only floor is a lane 0.4 m wide from 0.3 m behind
    # the robot to 3 m ahead of it: all of it in their view. The view is broken least at the lane's near corners, 0.811 m
    # inside the sector (1.7 sin 35 - 0.2 cos 35) with a robot's radius to keep; the lower row is on the robot's right.
    strip = lambda ahead, left: (np.abs(left) <= 0.2 + 1e-9) & (ahead >= -0.3 - 1e-9) & (ahead <= 3.0)       # noqa: E731
    lane = layout(HERE, [(-2.0, 0.0, 0.0)], thing=(3.0, 0.0), floor=strip)
    solved = lane.solve(ask(VIEW, 1), ask(FACE, target=True))
    broken = round(0.3 + 1.7 * SIN35 - 0.2 * COS35, 3)
    assert broken == 1.111 and solved.pose_world[:2] == near((-0.3, -0.2))
    assert [(use.status, use.broken_m) for use in solved.uses] == [("used", 1.111), ("used", 0.0)]
    assert solved.note.keys.excluded_m == 1.111 and solved.note.own_space_broken_m == {} and solved.flags.exclusion_broken
    assert solved.note.not_ahead_of == []          # straight ahead of them is no view of the target
    # Towards the thing their view ends at it, 5 m on, where the lane ends too: its far corners are 4 mm past that end.
    # With the standard goal on the thing there is no side the robot was sent to, and the zone is all that is kept.
    on_it = layout(HERE, [(-2.0, 0.0, 0.0)], thing=(3.0, 0.0), standard=(3.0, 0.0), floor=strip)
    held = on_it.solve(ask(VIEW, 1, target=True))
    assert held.pose_world[:2] == near((3.0, -0.2)) and held.uses[0].broken_m == round(0.3 - (math.hypot(5.0, 0.2) - 5.0), 3) == 0.296
    assert held.flags.exclusion_broken and held.note.not_ahead_of == []
    # With the standard goal before the thing they watch it from the side the robot was sent to, and the whole lane is
    # ahead of them, by 1.7 m at its near end. No cell is level with them, so the robot is not kept back with them at
    # all: the zone alone is kept, as above, and the record says why.
    back = lane.solve(ask(VIEW, 1, target=True))
    assert back.pose_world[:2] == near((3.0, -0.2)) and back.uses[0].broken_m == 0.296 and back.note.not_ahead_of == []
    assert (back.note.keys.excluded_m, back.note.keys.excluded_total_m) == (0.296, 0.296) and back.flags.exclusion_broken
    assert back.note.not_ahead_reason == "no cell is level with or behind person 1"
    assert held.note.not_ahead_reason == "the target has no seen face and the standard goal stands on the target point"
    free = layout(HERE, [(-2.0, 0.0, 0.0)], thing=(3.0, 0.0)).solve(ask(VIEW, 1))            # on an open floor it holds
    assert free.uses[0].broken_m == 0.0 and free.note.keys.excluded_m == 0.0 and not free.flags.exclusion_broken


def test_a_preference_that_cannot_be_met_is_recorded_and_breaks_no_exclusion():
    four = [(0.8 * math.cos(turn), 0.8 * math.sin(turn), turn + math.pi) for turn in (0.0, math.pi / 2, math.pi, -math.pi / 2)]
    lay = needing_the_first(four, (-5.3, 0.5))     # the robot comes from behind the one across from the one it needs
    close = lay.solve(ask(SHARED, 1, 2, 3, 4), ask(WITHIN, 1, between=(0.0, 0.6)), ask(VISIBLE, 1), ask(FACE, 1))
    away = apart(close.pose_world, four[0])
    assert close.note.keys.excluded_m == 0.0 and not close.flags.exclusion_broken and 1.0 - 5e-4 <= away <= 1.05
    assert [use.status for use in close.uses] == ["used"] * 4 and close.uses[0].broken_m == 0.0
    assert close.uses[1].broken_m == pytest.approx(away - 0.6, abs=1e-3)  # own space keeps it a standoff from them
    assert close.uses[2].broken_m > 0.5 and close.uses[3].broken_m == 0.0                    # and behind their shoulder
    assert close.note.keys.preference_m == pytest.approx(close.uses[1].broken_m + close.uses[2].broken_m, abs=2e-3)
    assert close.note.own_space_broken_m == {}


# -------------------------------------------------------------------------------------------------------- the runners-up
def test_the_runners_up_are_chosen_in_the_same_way_a_robots_width_from_every_place_taken():
    solved = layout(HERE, [], thing=(5.0, 0.0)).solve()                   # the place is the standard goal, (4, 0)
    spots = [candidate.pose_world[:2] for candidate in solved.candidates]
    # The cells within the slack of a standoff from the thing and 0.6 m or more from (4, 0): the nearest that place are
    # (4.15, +-0.6), 0.618 m from it and 1.04 m from the thing, the robot's right first; and so on round the thing.
    assert spots[:2] == [near((4.15, -0.6)), near((4.15, 0.6))] and len(spots) == TH.candidate_count == 4
    assert solved.candidates[0].keys.model_dump() == dict(excluded_m=0.0, excluded_total_m=0.0, turn_m=None, face_m=None,
                                                          preference_m=0.04, standard_m=0.618)
    taken = [solved.pose_world[:2], *spots]
    assert all(apart(a, b) >= 0.6 - 1e-9 for i, a in enumerate(taken) for b in taken[:i])
    assert [c.keys.standard_m for c in solved.candidates] == sorted(c.keys.standard_m for c in solved.candidates)
    assert all(c.keys.preference_m <= 0.05 for c in solved.candidates)   # each a standoff from the thing, to the slack
    assert all(c.pose_world[2] == near(facing(c.pose_world, (5.0, 0.0))) for c in solved.candidates)   # and facing it
    fewer = layout(HERE, [], thing=(5.0, 0.0)).solve(th=Thresholds(candidate_count=2))
    assert [c.pose_world for c in fewer.candidates] == [c.pose_world for c in solved.candidates[:2]]
    assert layout(HERE, [], thing=(5.0, 0.0)).solve(th=Thresholds(candidate_count=0)).candidates == []
    wide = layout(HERE, [], thing=(5.0, 0.0)).solve(th=Thresholds(robot_radius_m=0.5))
    assert apart(wide.candidates[0].pose_world, wide.pose_world) >= 1.0 - 1e-9
    # With no floor but its own disc the robot has one other place: the far end of the disc, 0.6 m away.
    confined = layout(HERE, [], thing=(5.0, 0.0), floor=lambda ahead, left: ahead > 99.0).solve()
    assert confined.pose_world[:2] == near((0.3, 0.0)) and [c.pose_world[:2] for c in confined.candidates] == [near((-0.3, 0.0))]
    turn_kept = waiting(FILE3).solve(ask(AFTER, 1, 2, 3))                 # a runner-up carries the same keys as the place
    assert all(c.keys.turn_m is not None and c.keys.turn_m >= turn_kept.note.keys.turn_m for c in turn_kept.candidates)


# ------------------------------------------------------------------------------------------------------------ the heading
def test_the_robot_faces_what_the_answer_names_or_the_last_who_waits_or_the_target():
    def heading(lay, *answer):
        solved = lay.solve(*answer)
        return solved.pose_world[:2], solved.pose_world[2]
    # A person stands before the person the robot needs: the place is beside them, and each of the three is a different way.
    lay = layout(HERE, [(6.0, 0.0, math.pi), (5.0, 0.0)], needed=1)
    place, theta = heading(lay)
    assert abs(place[1]) > 0.5 and theta == near(facing(place, (6.0, 0.0)))          # the target person's centre, not the mark
    assert heading(lay, ask(FACE, 2)) == (place, near(facing(place, (5.0, 0.0))))    # the listed person
    assert heading(lay, ask(FACE, 2, 1)) == (place, near(facing(place, (5.5, 0.0))))             # the middle of several
    assert heading(lay, ask(FACE, target=True), ask(FACE, 2)) == (place, near(theta))            # the first usable one stands
    assert heading(lay, ask(FACE), ask(FACE, 9), ask(FACE, 2), ask(FACE, 1)) == (place, near(facing(place, (5.0, 0.0))))
    thing = layout(HERE, [(4.0, 0.0)], thing=(5.0, 0.0))
    assert heading(thing) == (near((4.55, -0.85)), near(math.atan2(0.85, 0.45)))     # the target's point
    assert heading(thing, ask(FACE, 1))[1] == near(math.atan2(0.85, -0.55))
    # Where people wait, the robot faces the last of them; the answer can still name something else.
    file = waiting(FILE3)
    place, theta = heading(file, ask(AFTER, 1, 2, 3))
    assert theta == near(facing(place, FILE3[2])) and abs(theta + math.pi / 2) < 0.1
    assert heading(file, ask(AFTER, 1, 2, 3), ask(FACE, target=True))[1] == near(facing(place, THING_XY))
    assert heading(file, ask(AFTER, 1, 2, 3), ask(FACE, 1, 2))[1] == near(facing(place, (0.0, 1.7)))
    joined = waiting(FILE3 + [(0.1, 4.25)])        # someone has taken the place in turn: they are the last now
    place, theta = heading(joined, ask(AFTER, 1, 2, 3))
    assert theta == near(facing(place, (0.1, 4.25)))
    solved = file.solve(ask(AFTER, 1, 2, 3))
    assert solved.heading == solved.pose_world[2] and solved.candidates[0].pose_world[2] == near(
        facing(solved.candidates[0].pose_world, FILE3[2]))                # a runner-up faces the same thing from where it is


def test_the_robot_keeps_its_heading_when_what_it_faces_is_where_it_stands():
    robot = looking((0.0, -2.0), (5.0, 0.0))       # two people 2.2 m apart, and the standard goal midway between them
    lay = layout(robot, [(4.0, 1.1), (4.0, -1.1)], thing=(5.0, 0.0), standard=(4.0, 0.0))
    amid = lay.solve(ask(FACE, 1, 2))              # the robot stands on the cell that holds their middle
    assert apart(amid.pose_world, (4.0, 0.0)) < TH.heading_degenerate_m and amid.flags.heading_degenerate
    assert amid.pose_world[2] == near(robot.yaw) and amid.pose_robot[2] == 0.0 and amid.note.keys.excluded_m == 0.0
    turned = lay.solve(ask(FACE, 1))
    assert turned.pose_world[:2] == amid.pose_world[:2] and not turned.flags.heading_degenerate
    assert turned.pose_world[2] == near(facing(turned.pose_world, (4.0, 1.1)))
    assert turned.pose_robot == near((*to_robot_xy(robot, *turned.pose_world[:2]), turned.pose_world[2] - robot.yaw))
    exact = lay.solve(ask(FACE, 1, 2), th=Thresholds(heading_degenerate_m=0.01))
    assert not exact.flags.heading_degenerate      # 0.02 m from their middle is far enough for a bearing then


# -------------------------------------------------------------------------------------------------------------- the flags
def flags(solved):
    return {name for name, value in solved.flags.model_dump().items() if value}


def test_the_flags_say_what_was_made_of_the_answer():
    lay = trio()
    assert flags(lay.solve()) == {"no_usable_constraint"}                 # an answer with no constraint is not an error
    assert flags(lay.solve(ask(SHARED, 1), ask(FACE))) == {"no_usable_constraint", "constraint_unusable"}
    assert flags(lay.solve(ask(FACE, target=True))) == set()
    assert flags(lay.solve(ask(SHARED, 1, 2), ask(VISIBLE, 2))) == {"constraint_unusable"}
    assert flags(lay.solve(ask(FACE, 1), ask(FACE, 2))) == {"constraint_unusable"}               # a superseded one counts
    one = Thresholds(max_constraints=1)
    assert flags(lay.solve(ask(SHARED, 1, 2), ask(SHARED, 1), th=one)) == {"constraints_cut"}
    assert flags(lay.solve(ask(SHARED, 1), ask(SHARED, 1, 2), th=one)) == {"no_usable_constraint", "constraint_unusable",
                                                                           "constraints_cut"}


def test_a_place_far_from_the_target_is_flagged_unless_the_robot_waits_its_turn():
    # The floor ends a metre ahead of the robot and the thing is 6 m ahead: the robot stops 5 m short of it.
    cut_off = layout(HERE, [], thing=(6.0, 0.0), floor=lambda ahead, left: ahead <= 1.0 + 1e-9)
    stopped = cut_off.solve()
    assert stopped.pose_world[:2] == near((1.0, 0.0)) and stopped.note.target_distance_m == near(5.0)
    assert "far_from_target" in flags(stopped) and "exclusion_broken" not in flags(stopped)
    assert "far_from_target" not in flags(cut_off.solve(th=Thresholds(max_keep_m=5.0)))          # beyond it, not at it
    assert "far_from_target" in flags(cut_off.solve(th=Thresholds(max_keep_m=4.99)))
    assert "far_from_target" not in flags(layout(HERE, [], thing=(6.0, 0.0)).solve())
    # Behind four who wait the robot is 5.2 m from the thing, where its turn is: not flagged.
    assert waiting(FILE4).solve(ask(AFTER, 1, 2, 3, 4)).note.target_distance_m > 5.0
    assert flags(waiting(FILE4).solve(ask(AFTER, 1, 2, 3, 4))) == set()
    # A come_after that cannot be used does not count.
    assert flags(cut_off.solve(ask(AFTER, 9))) == {"far_from_target", "no_usable_constraint", "constraint_unusable"}
    # Nor does a view of the target with which nobody is stood back with: someone watches the thing from beyond it, and
    # the floor ends 3.5 m short of it.
    lane = layout(HERE, [(7.0, 0.0, math.pi)], thing=(5.0, 0.0),
                  floor=lambda ahead, left: (np.abs(left) <= 0.2 + 1e-9) & (ahead >= -0.3 - 1e-9) & (ahead <= 1.5))
    short = lane.solve(ask(VIEW, 1, target=True))
    assert short.note.not_ahead_of == [] and short.note.target_distance_m == near(3.5) and flags(short) == {"far_from_target"}
    # A range that can be used starts short of that distance, and by itself keeps the robot inside it.
    asked = layout(HERE, [], thing=(6.0, 0.0)).solve(ask(WITHIN, target=True, between=(2.9, 3.5)))
    assert asked.note.target_distance_m == near(2.85) and flags(asked) == set()               # the range's near end, less the slack


# ----------------------------------------------------------------------------------------------------------- the seen side
WATCHED = Box(cx=6.3, cy=0.4, yaw=math.pi, sx=0.6, sy=1.6, sz=1.9)       # it stands free: its face at x = 6.0, from y = -0.4
MARKED, FACE_AT = (6.0, 0.4, 1.6), (6.0, 0.4)                             # to 1.2, looks along -x, and the mark is on it
BEHIND_IT = Box(cx=6.75, cy=0.0, yaw=0.0, sx=0.3, sy=16.0, sz=2.5)       # a wall straight behind it, where a layout has one
SIDE, OTHER = looking((0.8, -3.6), (5.2, 0.4)), looking((1.0, 4.4), (5.2, 0.4))      # the robot comes from either side, or from HERE
CELL = TH.grid_resolution_m * math.sqrt(2.0)       # two places are the same to a cell
HAIR = 1e-6                                        # a distance just inside a bound, or just outside it, is this far from it
BROKEN_MORE = "every cell level with or behind person {} breaks the other exclusions more than some cell ahead of them"


def watching(spots, at=FACE_AT):
    """People at `spots`, each of them looking at `at`."""
    return [(x, y, facing((x, y), at)) for x, y in spots]


ROW = watching([(3.3, -0.7), (3.25, 0.4), (3.3, 1.5)])                    # the hindmost of them 2.75 m before the face
ROW_OPEN = watching([(3.3, -1.1), (3.25, 1.0), (3.3, 2.1)])               # for HERE, behind them: the mark shows between two
TWO_ROWS = watching([(3.6, -0.6), (3.6, 0.5), (3.6, 1.6), (2.5, -0.1), (2.5, 1.0)])           # the back row 3.50 m before it
TWO_ROWS_OPEN = watching([(3.6, -1.0), (3.6, 1.0), (3.6, 2.1), (2.5, -1.6), (2.5, 1.6)])
ROUND_IT = watching([(4.3, 2.4), (5.0, 2.6), (5.8, 2.4), (4.6, -2.5), (5.5, -2.4)], (5.0, 0.0))      # three on one side, two across


def pictured(robot, boxes, spots, mark):
    """What the mounted camera of a robot shows of things and of people, each (x, y, heading), and what the solver
    is given with it: the people where they stand, the target point lifted from the picture where `mark` is, and
    the standard goal of the first spec for it."""
    observation = mounted(robot, boxes=boxes, figures=[Figure(*spot) for spot in spots])
    u, v, _ = project(observation.camera, mark)
    world = lift(observation.depth, observation.camera, u, v, TH.lift_window)
    assert world is not None and math.dist(world, mark) < 0.15            # the mark is in view, on what it was put on
    target = GroundedTarget(point=Point(x=500, y=500), pixel=(u, v), world=tuple(float(c) for c in world))
    return Layout(observation, numbered(spots), target, standard_goal(observation, target, TH))


def ahead_of(place, spots, at):
    """How far ahead of the hindmost of the people at `spots` a place is, along the direction from the middle of
    them to the ground point of `at`; negative behind that person."""
    at, xy = np.array(at[:2]), np.array([spot[:2] for spot in spots])
    along = at - xy.mean(axis=0)
    along = along / math.hypot(*along)
    return float(((at - xy) @ along).max() - (at - np.array(place[:2])) @ along)


def nearest(place, spots):
    return min(apart(place, spot) for spot in spots)


def same(one, other):
    """Whether two answers were solved to the same place, with the same keys, note, flags and runners-up. Why nobody
    is stood back with is left out of the note: it is said only where the answer states a view of the target."""
    notes = [solved.note.model_copy(update={"not_ahead_reason": None}) for solved in (one, other)]
    return (one.pose_world, notes[0], one.flags, one.candidates) == (other.pose_world, notes[1], other.flags, other.candidates)


def test_the_seen_face_is_the_front_of_the_flat_upright_surface_the_target_point_lies_on():
    for robot in (HERE, SIDE, OTHER):              # the same face from three places, each time in the robot's own frame
        lay = pictured(robot, [WATCHED], [], MARKED)
        assert solver._seen_face(lay.observation, lay.target, TH) == near((-math.cos(robot.yaw), math.sin(robot.yaw)), abs=5e-3)
    assert solver._seen_face(open_floor(), target_at(5.0, 0.0), TH) is None                   # a mark on the floor has none,
    low = pictured(SIDE, [Box(cx=6.0, cy=0.4, yaw=0.0, sx=1.0, sy=1.0, sz=0.75)], [], (6.0, 0.4, 0.75))
    assert solver._seen_face(low.observation, low.target, TH) is None                          # nor has one on a level top


def test_how_far_behind_the_seen_face_a_cell_is_orders_the_cells_that_break_the_exclusions_least():
    # A face at x = 3 that looks along -x, and someone a metre before it. Of three spots, one before the face is 0.2 m
    # from them, one is 0.9 m behind the face, and one before it is a metre from them and 1.41 m from the mark.
    spots, someone = [(2.2, 0.0), (3.9, 0.0), (2.0, 1.0)], [(2.0, 0.0)]
    worked = at_points(spots, someone, [], target=(3.0, 0.0), standard=(2.0, 0.0), front=(-1.0, 0.0))
    assert list(worked.keys) == ["excluded_m", "excluded_total_m", "face_m", "preference_m", "standard_m", "robot_m"]
    assert worked.keys["face_m"] == near([0.0, 0.9, 0.0]) and worked.keys["preference_m"] == near([0.2, 0.1, 0.414])
    assert worked.picks[0] == 2                    # before the face, whatever the preferences say: without it the one behind
    assert at_points(spots, someone, [], target=(3.0, 0.0), standard=(2.0, 0.0)).picks[0] == 1
    assert "face_m" not in at_points(spots, someone, [], target=(3.0, 0.0), standard=(2.0, 0.0)).keys
    # No cell is removed. Where the only cell before the face is inside someone's own space the place is behind it,
    # and the key says by how much.
    only = at_points(spots[:2], someone, [], target=(3.0, 0.0), standard=(2.0, 0.0), front=(-1.0, 0.0))
    assert only.picks[0] == 1 and only.keys["face_m"][1] == 0.9 and only.keys["excluded_m"] == near([0.8, 0.0])
    # A place in turn takes the key's place, and a person who is the target has no seen face.
    after = at_points(spots, someone, [ask(AFTER, 1)], target=(3.0, 0.0), standard=(2.0, 0.0), front=(-1.0, 0.0))
    assert list(after.keys)[2:4] == ["turn_m", "preference_m"]
    person_is_target = at_points(spots, [(3.2, 0.0, 0.0)], [], target=(3.0, 0.0), standard=(2.0, 0.0), front=(-1.0, 0.0))
    assert person_is_target.target_xy == near((3.2, 0.0)) and "face_m" not in person_is_target.keys
    # Nor is the key read where the answer says the robot came for a person: it names someone to face, and says the
    # place is to be visible to that same person. The place is then chosen by their front, and a face fitted by depth
    # does not rank above that. The one a metre before the face looks along +x here, and of the two places outside
    # their own space the one they see without turning round is the one behind the face.
    looks_on = [(2.0, 0.0, 0.0)]

    def worked_for(*answer, people=looks_on, th=TH):
        return at_points(spots, people, list(answer), target=(3.0, 0.0), standard=(2.0, 0.0), front=(-1.0, 0.0), th=th)

    def keyed(*answer, **given):
        """Whether the seen face is read, which is whether the place is the one before it."""
        found = worked_for(*answer, **given)
        assert found.picks[0] == (2 if "face_m" in found.keys else 1), answer
        return "face_m" in found.keys
    for_them = worked_for(ask(VISIBLE, 1), ask(FACE, 1))
    assert "face_m" not in for_them.keys and for_them.picks[0] == 1
    assert list(for_them.keys) == ["excluded_m", "excluded_total_m", "preference_m", "standard_m", "robot_m"]
    # Being visible to someone is not by itself coming for them: alone, or with the robot told to face the target, the
    # key is read and the place is the one before the face, which they do not see (0.524 m of violation).
    alone = worked_for(ask(VISIBLE, 1))
    assert "face_m" in alone.keys and alone.picks[0] == 2 and alone.violations[0] == near([0.0, 0.0, 0.524], abs=1e-3)
    assert keyed() and keyed(ask(VISIBLE, 1)) and keyed(ask(VISIBLE, 1), ask(FACE, target=True)) and keyed(ask(FACE, 1))
    assert keyed(ask(WITHIN, 1, between=(0.5, 1.5)), ask(FACE, 1))           # a range about them says nothing of a front
    # It takes both, in any order and wherever they stand among the constraints that are used.
    assert not keyed(ask(FACE, 1), ask(VISIBLE, 1))
    assert not keyed(ask(WITHIN, 1, between=(0.5, 1.5)), ask(VISIBLE, 1), ask(FACE, 1))
    assert not keyed(ask(FACE, 1), ask(VISIBLE, 1), ask(WITHIN, 1, between=(0.5, 1.5)))
    # Neither counts unless it is used: a `visible_to` or a `face` that is unusable, cut or superseded says nothing.
    assert keyed(ask(VISIBLE, 9), ask(FACE, 1)) and keyed(ask(VISIBLE), ask(FACE, 1)) and keyed(ask(VISIBLE, 1), ask(FACE, 9))
    assert keyed(ask(VISIBLE, 1), ask(FACE, 1), people=someone)              # without a heading they have no front to be seen from
    one = Thresholds(max_constraints=1)
    assert keyed(ask(FACE, 1), ask(VISIBLE, 1), th=one) and keyed(ask(VISIBLE, 1), ask(FACE, 1), th=one)
    assert not keyed(ask(FACE, 1), ask(VISIBLE, 1), th=Thresholds(max_constraints=2))
    assert keyed(ask(FACE, target=True), ask(FACE, 1), ask(VISIBLE, 1))      # the face that stands is the target's
    # The two are about the same person: the one faced is among those the place is to be visible to, with others or
    # not. A second person stands off to the side and looks along +x as well.
    two = [(2.0, 0.0, 0.0), (1.0, 2.0, 0.0)]
    assert not keyed(ask(VISIBLE, 2, 1), ask(FACE, 1), people=two) and not keyed(ask(VISIBLE, 1), ask(VISIBLE, 2), ask(FACE, 2), people=two)
    assert not keyed(ask(VISIBLE, 1), ask(FACE, 1, 2), people=two)           # faced with another: still one the robot came for
    assert keyed(ask(VISIBLE, 2), ask(FACE, 1), people=two) and keyed(ask(VISIBLE, 1, 2), ask(FACE, target=True), people=two)
    # Of the person faced only a `visible_to` counts as being seen by them: a shared space or a view that lists them says
    # nothing of a front, and a `face` of several is read in all of its people, not the first alone.
    assert not keyed(ask(VISIBLE, 2), ask(FACE, 1, 2), people=two)
    assert keyed(ask(VIEW, 1), ask(FACE, 1))
    # Told by the key itself: with the shared space the place is the one behind the face all the same, for the shared
    # space excludes the two before it (0.8 and 0.447 m), but the key is read.
    shared = worked_for(ask(SHARED, 1, 2), ask(FACE, 1), people=two)
    assert "face_m" in shared.keys and shared.keys["face_m"] == near([0.0, 0.9, 0.0])
    # A faced person without a heading is not among those who see, whoever is listed with them.
    assert keyed(ask(VISIBLE, 1, 2), ask(FACE, 1), people=[(2.0, 0.0), (1.0, 2.0, 0.0)])
    # The people who watch the target are joined all the same: the seen face still tells the side the robot was sent to.
    both = at_points(spots, looks_on, [ask(VISIBLE, 1), ask(FACE, 1), ask(VIEW, 1, target=True)], target=(3.0, 0.0), front=(-1.0, 0.0))
    assert "face_m" not in both.keys and both.not_ahead_of == [1]            # the standard goal is on the target: no side but the face's


def test_a_view_of_the_target_is_violated_as_well_by_how_far_a_place_is_ahead_of_those_who_watch():
    # Someone at the origin watches the target at (3, 0), and the standard goal stands on their side of it. Ahead of
    # them is everything farther along +x than they are, the floor beyond the target with it: by 2.0 m at (2, 0).
    spots = [(2.0, 0.0), (3.2, 0.0), (4.0, 0.0), (0.0, 1.0), (2.9, 0.0), (-1.0, 2.0)]
    zone = [0.3 + 1.0, 0.1, 0.0, 0.0, 0.3 + 0.1, 0.0]                     # their view alone, as the test of the sector has it
    one = at_points(spots, [(0.0, 0.0)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0), standard=(2.0, 0.0))
    assert one.not_ahead_of == [1] and one.violations[0] == near([2.0, 3.2, 4.0, 0.0, 2.9, 0.0])     # the larger of the two
    assert one.keys["excluded_m"][:3] == near([2.0, 3.2, 4.0]) and one.keys["excluded_total_m"][:3] == near([3.3, 3.3, 4.0])
    twice = at_points(spots, [(0.0, 0.0)], [ask(VIEW, 1, target=True)] * 2, target=(3.0, 0.0), standard=(2.0, 0.0))
    assert twice.violations[1] == near([2.0, 3.2, 4.0, 0.0, 2.9, 0.0]) and twice.not_ahead_of == [1]
    assert twice.keys["excluded_total_m"][:3] == near([1.3 + 1.3 + 2.0, 0.1 + 0.1 + 3.2, 4.0])      # ahead counts once
    assert one.not_ahead_reason is None and twice.not_ahead_reason is None
    # Several views of the target are one, about all their people. Of two people listed in a constraint each, the
    # hindmost is the second: along the direction from their middle, (-0.5, 0.25), they stand 4.026 m before the target.
    apart_listed = at_points(spots, [(0.0, 0.0), (-1.0, 0.5)], [ask(VIEW, 1, target=True), ask(VIEW, 2, target=True)],
                             target=(3.0, 0.0), standard=(2.0, 0.0))
    along = np.array([3.5, -0.25]) / math.hypot(3.5, 0.25)
    ahead = [max(0.0, float(np.array([4.0, -0.5]) @ along) - float(np.subtract((3.0, 0.0), spot) @ along)) for spot in spots]
    assert apart_listed.not_ahead_of == [1, 2] and ahead == near([3.0280, 4.2250, 5.0229, 0.9618, 3.9257, 0.0], abs=1e-4)
    assert apart_listed.violations[0] == near(ahead) and apart_listed.violations[1] == near(ahead)
    # With the standard goal on the target there is no side the robot was sent to: the zone alone. It is on the target
    # when it is within `heading_degenerate_m` of it, here 0.03 m, and it is not at 0.1 m.
    no_side = at_points(spots, [(0.0, 0.0)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0))
    assert no_side.not_ahead_of == [] and no_side.violations[0] == near(zone)
    nearly = at_points(spots, [(0.0, 0.0)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0), standard=(2.97, 0.0))
    assert nearly.not_ahead_of == [] and nearly.violations[0] == near(zone)
    no_face_no_side = "the target has no seen face and the standard goal stands on the target point"
    assert nearly.not_ahead_reason == no_side.not_ahead_reason == no_face_no_side
    assert at_points(spots, [(0.0, 0.0)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0), standard=(2.9, 0.0)).not_ahead_of == [1]
    # The bound is the threshold itself, whatever it is set to: a hair nearer the target than that there is no side, and
    # a hair farther there is one.
    for th in (TH, Thresholds(heading_degenerate_m=0.2)):
        inside, outside = (at_points(spots, [(0.0, 0.0)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0), standard=(3.0 - gap, 0.0), th=th)
                           for gap in (th.heading_degenerate_m - HAIR, th.heading_degenerate_m + HAIR))
        assert inside.not_ahead_of == [] and inside.not_ahead_reason == no_face_no_side, th.heading_degenerate_m
        assert outside.not_ahead_of == [1] and outside.not_ahead_reason is None, th.heading_degenerate_m
    # At exactly the threshold there is a side: it is nearer than `heading_degenerate_m` that there is none. 0.25 m is
    # exact in floating point.
    exact = Thresholds(heading_degenerate_m=0.25)
    at_it = at_points(spots, [(0.0, 0.0)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0), standard=(2.75, 0.0), th=exact)
    assert at_it.not_ahead_of == [1] and at_it.not_ahead_reason is None
    # A seen face tells the side whatever the standard goal: before a face that looks at them they are joined, and
    # behind one that looks away from them they are not.
    faced = at_points(spots, [(0.0, 0.0)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0), front=(-1.0, 0.0))
    assert faced.not_ahead_of == [1] and faced.violations[0] == near([2.0, 3.2, 4.0, 0.0, 2.9, 0.0])
    away = at_points(spots, [(0.0, 0.0)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0), standard=(2.0, 0.0), front=(1.0, 0.0))
    assert away.not_ahead_of == [] and away.violations[0] == near(zone)
    assert away.not_ahead_reason == "none of them stands on the side the robot was sent to"
    # Of three listed, the one beyond the target is not joined. The direction runs from the middle of the other two,
    # (0.5, 0.5), to the target, and the hindmost along it is the one at the origin, 2.942 m before the target.
    three = at_points(spots, [(0.0, 0.0), (1.0, 1.0), (5.0, 0.5)], [ask(VIEW, 1, 2, 3, target=True)], target=(3.0, 0.0),
                      standard=(2.0, 0.0))
    along = np.array([2.5, -0.5]) / math.hypot(2.5, 0.5)
    ahead = [max(0.0, 3.0 * along[0] - float(np.subtract((3.0, 0.0), spot) @ along)) for spot in spots]
    assert three.not_ahead_of == [1, 2] and three.violations[0] == near(ahead)
    assert ahead == near([1.9612, 3.1379, 3.9223, 0.0, 2.8437, 0.0], abs=1e-4)
    # The hindmost is taken among those joined, not among everyone listed. Before a face at x = 3 that looks along -x
    # stands one person, 2.062 m before the target along their own line to it; another, listed with them, stands behind
    # the plane of the face and twice as far off. A place at (2, 0) is ahead of the first by 1.819 m, and of nobody else.
    behind_face = at_points(spots, [(2.5, 2.0), (3.5, 4.0)], [ask(VIEW, 1, 2, target=True)], target=(3.0, 0.0), front=(-1.0, 0.0))
    assert behind_face.not_ahead_of == [1] and behind_face.violations[0][0] == near(math.hypot(0.5, 2.0) - 0.5 / math.hypot(0.5, 2.0))
    assert behind_face.violations[0][0] == near(1.819, abs=1e-3) and behind_face.violations[0][5] == near(0.0)
    # Nothing is done where the direction they watch in cannot be told: two people either side of the target, their
    # middle 0.01 m from it.
    amid = at_points(spots, [(2.99, 2.0), (2.99, -2.0)], [ask(VIEW, 1, 2, target=True)], target=(3.0, 0.0), front=(-1.0, 0.0))
    assert amid.not_ahead_of == [] and amid.not_ahead_reason == "the middle of them is on the target point"
    off_it = at_points(spots, [(2.9, 2.0), (2.9, -2.0)], [ask(VIEW, 1, 2, target=True)], target=(3.0, 0.0), front=(-1.0, 0.0))
    assert off_it.not_ahead_of == [1, 2]           # 0.1 m from it is far enough
    for th in (TH, Thresholds(heading_degenerate_m=0.2)):                 # here too the bound is the threshold itself
        inside, outside = (at_points(spots, [(3.0 - gap, 2.0), (3.0 - gap, -2.0)], [ask(VIEW, 1, 2, target=True)], target=(3.0, 0.0),
                                     front=(-1.0, 0.0), th=th)
                           for gap in (th.heading_degenerate_m - HAIR, th.heading_degenerate_m + HAIR))
        assert inside.not_ahead_of == [] and inside.not_ahead_reason == "the middle of them is on the target point", th.heading_degenerate_m
        assert outside.not_ahead_of == [1, 2] and outside.not_ahead_reason is None, th.heading_degenerate_m
    at_it = at_points(spots, [(2.75, 2.0), (2.75, -2.0)], [ask(VIEW, 1, 2, target=True)], target=(3.0, 0.0), front=(-1.0, 0.0), th=exact)
    assert at_it.not_ahead_of == [1, 2] and at_it.not_ahead_reason is None      # and so is there a direction, at exactly it
    # Neither a place in turn nor a person who is the target is stood back from.
    waited = at_points(spots, [(0.0, 0.0)], [ask(VIEW, 1, target=True), ask(AFTER, 1)], target=(3.0, 0.0), standard=(2.0, 0.0))
    assert waited.not_ahead_of == [] and waited.violations[0] == near(zone) and waited.not_ahead_reason == "a place in turn is used"
    needed = at_points(spots, [(0.0, 0.0), (3.2, 0.0)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0), standard=(2.0, 0.0))
    assert needed.target_xy == near((3.2, 0.0)) and needed.not_ahead_of == [] and needed.not_ahead_reason == "the target is a person"
    # No reason is recorded where no view of the target is used: nothing was there to apply.
    for answer in ([ask(VIEW, 1), ask(AFTER, 1)], [ask(VIEW, 9, target=True)], []):
        assert at_points(spots, [(0.0, 0.0, 0.0)], answer, target=(3.0, 0.0), standard=(2.0, 0.0)).not_ahead_reason is None


def test_where_no_place_is_level_with_or_behind_those_who_watch_the_robot_is_not_kept_back_with_them_at_all():
    # Someone at the origin watches the target at (3, 0), as in the test above, and every place there is lies ahead of
    # them. The rule cannot be met, so it is not applied: no exclusion is added, the view is violated by its zone alone,
    # and the record says why. Applied all the same, it would only trade how far ahead a place is against the other
    # exclusions, and send the robot to the farthest floor there is.
    spots = [(2.0, 0.0), (3.2, 0.0), (4.0, 0.0), (2.9, 0.0), (0.5, 1.0)]
    zone = [0.3 + 1.0, 0.1, 0.0, 0.3 + 0.1, 0.0]
    none_level = at_points(spots, [(0.0, 0.0)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0), standard=(2.0, 0.0))
    assert none_level.not_ahead_of == [] and none_level.violations[0] == near(zone)
    assert none_level.not_ahead_reason == "no cell is level with or behind person 1"
    assert none_level.keys["excluded_m"] == near(zone) and none_level.keys["excluded_total_m"] == near(zone)
    plain = at_points(spots, [(0.0, 0.0)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0))    # as where there is no side
    assert all(np.array_equal(none_level.keys[name], plain.keys[name]) for name in ("excluded_m", "excluded_total_m", "preference_m"))
    # Level is told to the millimetre, as every value that is compared: 0.4 mm ahead of them is level, 2 mm is not.
    for by, joined in ((0.0004, [1]), (0.002, [])):
        worked = at_points(spots + [(by, 1.0)], [(0.0, 0.0)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0), standard=(2.0, 0.0))
        assert worked.not_ahead_of == joined, by
    # Of several to be joined the hindmost decides, and the record names them. With a place level with the first of two
    # and none level with the second, who stands 1.3 m farther back, neither is stood back with: the rule is not
    # applied to those it could be met for. That place is 1.26 m from the first and breaks nothing.
    more, pair = spots + [(0.0, 2.2)], [(0.4, 1.0), (-1.0, -0.5)]
    two = at_points(more, pair, [ask(VIEW, 1, 2, target=True)], target=(3.0, 0.0), standard=(2.0, 0.0))
    assert two.not_ahead_of == [] and two.not_ahead_reason == "no cell is level with or behind person 2"
    first = at_points(more, pair, [ask(VIEW, 1, target=True)], target=(3.0, 0.0), standard=(2.0, 0.0))
    assert first.not_ahead_of == [1] and first.not_ahead_reason is None and first.picks[0] == 5
    assert (first.keys["excluded_m"][5], first.own_space[1][5], first.own_space[2][5]) == (0.0, 0.0, 0.0)


def test_the_robot_is_not_kept_back_with_those_who_watch_where_that_breaks_another_exclusion_more():
    # Someone at (0.4, 1) watches the target at (3, 0). One place is level with them, 0.4 m from them: 0.6 m inside their
    # own space. A place beyond the target breaks nothing. Kept back with them, the robot stood on the first, inside
    # their own space on account of this rule. The rule is applied only where, among the places that break the other
    # exclusions least, one is level with the hindmost or behind: here none is, so no exclusion is added, the view is
    # violated by its zone alone, and the record says why.
    spots, pair = [(2.0, 0.0), (3.2, 0.0), (4.0, 0.0), (2.9, 0.0), (0.5, 1.0), (0.0, 1.0)], [(0.4, 1.0), (-1.0, -0.5)]
    inside = at_points(spots, pair, [ask(VIEW, 1, target=True)], target=(3.0, 0.0), standard=(2.0, 0.0))
    assert inside.own_space[1] == near([0.0, 0.0, 0.0, 0.0, 0.9, 0.6]) and inside.not_ahead_of == []
    assert inside.not_ahead_reason == BROKEN_MORE.format(1) and inside.picks[0] == 2
    assert (inside.keys["excluded_m"][2], inside.own_space[1][2], inside.own_space[2][2]) == (0.0, 0.0, 0.0)
    plain = at_points(spots, pair, [ask(VIEW, 1, target=True)], target=(3.0, 0.0))            # as where there is no side
    assert all(np.array_equal(inside.keys[name], plain.keys[name]) for name in ("excluded_m", "excluded_total_m", "preference_m"))
    assert np.array_equal(inside.violations[0], plain.violations[0]) and plain.not_ahead_reason != inside.not_ahead_reason
    # The same with the one who watches at the origin and one place level with them, 0.3 m beside them.
    beside = at_points(spots[:5] + [(0.0, 0.3)], [(0.0, 0.0)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0), standard=(2.0, 0.0))
    assert beside.not_ahead_of == [] and beside.not_ahead_reason == BROKEN_MORE.format(1) and beside.picks[0] == 2
    assert beside.own_space[1][5] == near(0.7) and beside.own_space[1][2] == 0.0
    # More is told to the millimetre, as every value that is compared: 3 cm inside their own space is more than none,
    # and 0.4 mm inside it is none.
    for gap, joined in ((0.97, []), (0.9996, [1])):
        edge = at_points([(0.0, gap), (4.0, 0.0)], [(0.0, 0.0)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0), standard=(2.0, 0.0))
        assert edge.not_ahead_of == joined and edge.picks[0] == (0 if joined else 1), gap
    # Where every place breaks another exclusion the rule is still applied, when one of those that break them least is
    # level with the hindmost. Of two places, the one level with them is 0.4 m inside their own space, and the other is
    # 0.5 m inside it and 0.587 m inside their view: the first is the place with or without the rule, and they are joined.
    least = at_points([(0.0, 0.6), (0.5, 0.0)], [(0.0, 0.0)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0), standard=(2.0, 0.0))
    assert least.not_ahead_of == [1] and least.not_ahead_reason is None and least.picks[0] == 0
    assert least.keys["excluded_m"] == near([0.4, 0.587]) and least.own_space[1] == near([0.4, 0.5])
    # Least is told as the order tells it: by the largest violation, then by the sum. Two places are 0.4 m inside the
    # own space of the one who watches. The one level with them is 0.4 m inside a second person's as well, and the one
    # 0.36 m ahead of them is 0.113 m inside their view: 0.8 m in sum against 0.513 m. Kept back, the robot stood inside
    # the second person's own space, which the place it has without the rule is not. So nobody is joined.
    places = [(0.0, 0.6), (0.36, -0.48)]
    summed = at_points(places, [(0.0, 0.0), (0.0, 1.2)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0), standard=(2.0, 0.0))
    assert summed.keys["excluded_m"] == near([0.4, 0.4]) and summed.keys["excluded_total_m"] == near([0.8, 0.513])
    assert summed.not_ahead_of == [] and summed.not_ahead_reason == BROKEN_MORE.format(1) and summed.picks[0] == 1
    assert summed.own_space[2] == near([0.4, 0.0])
    alone = at_points(places, [(0.0, 0.0)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0), standard=(2.0, 0.0))
    assert alone.not_ahead_of == [1] and alone.picks[0] == 0                # without the second person the two are equal, and
    assert alone.keys["excluded_total_m"] == near([0.4, 0.513 + 0.36])       # the one ahead is then ahead by 0.36 m more
    # The largest violation comes first: a place level with them 0.3 m inside their own space, and one ahead of them
    # 0.2 m inside theirs and a second person's. The second has the larger sum and is the place without the rule.
    ahead = (0.8 * math.cos(math.radians(-60.0)), 0.8 * math.sin(math.radians(-60.0)))
    largest = at_points([(0.0, 0.7), ahead], [(0.0, 0.0), (ahead[0], ahead[1] - 0.8)], [ask(VIEW, 1, target=True)], target=(3.0, 0.0),
                        standard=(2.0, 0.0))
    assert largest.keys["excluded_m"] == near([0.3, 0.2]) and largest.keys["excluded_total_m"] == near([0.3, 0.4])
    assert largest.not_ahead_of == [] and largest.not_ahead_reason == BROKEN_MORE.format(1) and largest.picks[0] == 1
    # Another exclusion is any of them, one the answer states as well as own space. The place level with the one who
    # watches lies midway between two who attend to one another, a metre from each and 0.874 m inside their space.
    talking = [(0.0, 0.0), (-0.5, 2.0), (-0.5, 4.0)]
    amid = at_points([(-0.5, 3.0), (4.0, 0.0)], talking, [ask(VIEW, 1, target=True), ask(SHARED, 2, 3)], target=(3.0, 0.0), standard=(2.0, 0.0))
    assert amid.violations[1] == near([0.874, 0.0], abs=1e-3) and all((space == 0.0).all() for space in amid.own_space.values())
    assert amid.not_ahead_of == [] and amid.not_ahead_reason == BROKEN_MORE.format(1) and amid.picks[0] == 1
    unlisted = at_points([(-0.5, 3.0), (4.0, 0.0)], talking, [ask(VIEW, 1, target=True)], target=(3.0, 0.0), standard=(2.0, 0.0))
    assert unlisted.not_ahead_of == [1] and unlisted.picks[0] == 0          # with their space not stated, the place is that one
    # The two comparisons that tell least are separate. Each case has one watcher at the origin, the target at (3, 0), and
    # two places, the second of them 0.36 m ahead of the watcher and inside their view; nobody is joined in the first two.
    # (a) By the largest violation to the millimetre, whatever the sum: 0.401 m against 0.400 m, sums 0.401 and 0.513.
    bare = dict(target=(3.0, 0.0), standard=(2.0, 0.0))
    mm = at_points([(0.0, 0.599), (0.36, -0.48)], [(0.0, 0.0)], [ask(VIEW, 1, target=True)], **bare)
    assert mm.keys["excluded_m"] == near([0.401, 0.4]) and mm.keys["excluded_total_m"] == near([0.401, 0.513])
    assert mm.not_ahead_of == [] and mm.not_ahead_reason == BROKEN_MORE.format(1)
    # (b) By the sum to the millimetre among those at the least largest: both are 0.4 m, sums 0.518 m and 0.513 m.
    tie = at_points([(0.0, 0.6), (0.36, -0.48)], [(0.0, 0.0), (0.0, 1.482)], [ask(VIEW, 1, target=True)], **bare)
    assert tie.keys["excluded_m"] == near([0.4, 0.4]) and tie.keys["excluded_total_m"] == near([0.518, 0.513])
    assert tie.not_ahead_of == [] and tie.not_ahead_reason == BROKEN_MORE.format(1)
    # (c) The sum is compared among those at the least largest, not over all: the level place has the larger sum (0.6 m
    # against 0.513 m, which is 0.873 m with the 0.36 m by which it is ahead) and the smaller largest (0.3 m against
    # 0.4 m). It alone is of the least, so person 1 is joined.
    by_largest = at_points([(0.0, 0.7), (0.36, -0.48)], [(0.0, 0.0), (0.0, 1.4)], [ask(VIEW, 1, target=True)], **bare)
    assert by_largest.keys["excluded_m"] == near([0.3, 0.4]) and by_largest.keys["excluded_total_m"] == near([0.6, 0.873])
    assert by_largest.not_ahead_of == [1] and by_largest.not_ahead_reason is None and by_largest.picks[0] == 0
    # The record names the hindmost, not the first of those joined: person 2 stands 1.12 m behind person 1 along the way
    # they watch, and the place level with them is 0.7 m inside person 1's own space.
    hindmost = at_points([(-1.0, -0.2), (4.0, 0.0)], pair, [ask(VIEW, 1, 2, target=True)], **bare)
    assert hindmost.not_ahead_of == [] and hindmost.not_ahead_reason == BROKEN_MORE.format(2) and hindmost.picks[0] == 1
    # The zone of a view of the target is one of the other exclusions: a place level with the hindmost whose only violation
    # is 0.07 m inside it (it is in nobody's own space) is not applied against a free place at (6, 0).
    zone = at_points([(1.979, 3.1246), (6.0, 0.0)], [(0.0, -1.0), (2.8, 4.0)], [ask(VIEW, 1, 2, target=True)], front=(-1.0, 0.0), **bare)
    assert all((space == 0.0).all() for space in zone.own_space.values()) and zone.keys["excluded_m"] == near([0.07, 0.0], abs=1e-3)
    assert zone.not_ahead_of == [] and zone.not_ahead_reason == BROKEN_MORE.format(2) and zone.picks[0] == 1


@pytest.mark.parametrize("robot, row", [(SIDE, ROW), (OTHER, ROW), (HERE, ROW_OPEN)])
def test_the_robot_joins_a_row_of_people_who_watch_a_thing_and_does_not_go_behind_the_thing(robot, row):
    # Their views take the front of the ring a standoff round the mark. The thing stands free, so there is floor behind
    # it, and the nearest free cell of that ring was there: 0.97 m behind the face from SIDE, out of everyone's view.
    lay = pictured(robot, [WATCHED], row, MARKED)
    solved = in_order(lay, ask(VIEW, 1, 2, 3, target=True))
    place, keys = solved.pose_world[:2], solved.note.keys
    assert solved.note.not_ahead_of == [1, 2, 3] and ahead_of(place, row, lay.target.world) <= 5e-4
    assert 6.0 - place[0] >= 2.75 - 2e-3           # level with the hindmost of them or behind, and so before the face
    assert 1.0 - 5e-4 <= nearest(place, row) <= 1.1 and (keys.excluded_m, keys.face_m, solved.uses[0].broken_m) == (0.0, 0.0, 0.0)
    assert flags(solved) == set()                  # 3.5 m from the thing beside the row, and that is not far from it
    assert solved.note.target_distance_m > (TH.max_keep_m if row is ROW else 2.9)
    # Against a wall the thing has no floor behind it. The robot then stood at its flank, 2.0 m from the mark and
    # ahead of everyone; it now stands where it stands without the wall.
    walled = in_order(pictured(robot, [WATCHED, BEHIND_IT], row, MARKED), ask(VIEW, 1, 2, 3, target=True))
    assert apart(walled.pose_world, place) <= CELL + 1e-9 and walled.note.not_ahead_of == [1, 2, 3]


@pytest.mark.parametrize("robot, rows", [(SIDE, TWO_ROWS), (OTHER, TWO_ROWS), (HERE, TWO_ROWS_OPEN)])
def test_behind_two_rows_of_people_who_watch_a_thing_the_robot_stands_level_with_the_back_row(robot, rows):
    lay = pictured(robot, [WATCHED], rows, MARKED)
    solved = in_order(lay, ask(VIEW, 1, 2, 3, 4, 5, target=True))
    place = solved.pose_world[:2]
    assert solved.note.not_ahead_of == [1, 2, 3, 4, 5] and ahead_of(place, rows, lay.target.world) <= 5e-4
    assert 6.0 - place[0] >= 3.45                  # the back row stands 3.50 m before the face
    assert 1.0 - 5e-4 <= nearest(place, rows) <= 1.25 and solved.uses[0].broken_m == 0.0 and flags(solved) == set()
    walled = in_order(pictured(robot, [WATCHED, BEHIND_IT], rows, MARKED), ask(VIEW, 1, 2, 3, 4, 5, target=True))
    assert apart(walled.pose_world, place) <= CELL + 1e-9


def test_only_the_people_the_answer_lists_as_watching_the_target_are_stood_back_with():
    lay = pictured(SIDE, [WATCHED], TWO_ROWS, MARKED)
    solved = in_order(lay, ask(VIEW, 1, 2, 3, target=True))               # the front row alone
    place = solved.pose_world[:2]
    assert solved.note.not_ahead_of == [1, 2, 3] and -0.1 <= ahead_of(place, TWO_ROWS[:3], lay.target.world) <= 5e-4
    assert 1.0 - 5e-4 <= nearest(place, TWO_ROWS) <= 1.1 and place[0] > 3.0          # beside that row, ahead of the back one
    # Stated as a range about the target and facing it as well, the place is the same: the range is a preference.
    row = pictured(SIDE, [WATCHED], ROW, MARKED)
    plain = row.solve(ask(VIEW, 1, 2, 3, target=True))
    asked = in_order(row, ask(VIEW, 1, 2, 3, target=True), ask(WITHIN, target=True, between=(0.5, 1.5)), ask(FACE, target=True))
    assert apart(asked.pose_world, plain.pose_world) <= CELL + 1e-9 and flags(asked) == set()
    assert asked.uses[0].broken_m == 0.0 and asked.uses[1].broken_m == pytest.approx(asked.note.target_distance_m - 1.5, abs=2e-3)
    assert asked.uses[1].broken_m > 1.9            # 3.5 m from the mark, where at most 1.5 m was asked for


def test_of_people_who_watch_a_thing_from_two_sides_those_on_the_side_the_robot_was_sent_to_are_joined():
    # A thin thing that stands free, its face at x = 5.35 towards the robot: two people watch that face and two the
    # other one, from beyond it. The robot stood 1.81 m from the mark, ahead of the near two.
    thin = Box(cx=5.5, cy=0.0, yaw=math.pi, sx=0.3, sy=2.4, sz=1.9)
    spots = watching([(3.2, 0.9), (3.3, -0.8)], (5.35, 0.0)) + watching([(7.6, 2.4), (7.5, -2.3)], (5.65, 0.0))
    lay = pictured(HERE, [thin], spots, (5.35, 0.0, 1.6))
    everyone = in_order(lay, ask(VIEW, 1, 2, 3, 4, target=True))
    place = everyone.pose_world[:2]
    assert everyone.note.not_ahead_of == [1, 2] and ahead_of(place, spots[:2], lay.target.world) <= 5e-4
    assert 1.0 - 5e-4 <= nearest(place, spots) <= 1.1 and everyone.note.keys.excluded_m == 0.0 and flags(everyone) == set()
    # Those beyond it, listed alone, are not joined. Their zones are kept, which end at the thing, and no more.
    beyond = in_order(lay, ask(VIEW, 3, 4, target=True))
    assert beyond.note.not_ahead_of == [] and beyond.uses[0].broken_m == 0.0
    assert beyond.pose_world == lay.solve().pose_world and apart(beyond.pose_world, lay.standard.pose_world) <= CELL
    # A thing without a face, on an open floor: the side is the one the standard goal stands on.
    lay = layout(looking((5.0, 7.0), (5.0, 0.0)), ROUND_IT, thing=(5.0, 0.0))
    solved = in_order(lay, ask(VIEW, 1, 2, 3, 4, 5, target=True))
    place = solved.pose_world[:2]
    assert solved.note.not_ahead_of == [1, 2, 3] and solved.note.keys.face_m is None and flags(solved) == set()
    assert ahead_of(place, ROUND_IT[:3], (5.0, 0.0)) <= 5e-4 and 1.0 - 5e-4 <= nearest(place, ROUND_IT) <= 1.1


def test_someone_who_watches_a_thing_from_its_other_side_is_not_joined():
    # Someone at work behind a low thing looks over it at the robot, and the answer lists them under a view of the
    # target. The place stays the standard goal, a standoff before the face.
    low = Box(cx=5.3, cy=0.0, yaw=math.pi, sx=0.6, sy=2.4, sz=1.1)
    lay = pictured(HERE, [low], [(6.2, 0.0, math.pi)], (5.0, 0.0, 0.8))
    solved = in_order(lay, ask(VIEW, 1, target=True))
    assert solved.note.not_ahead_of == [] and solved.note.keys.face_m == 0.0 and flags(solved) == set()
    assert solved.pose_world == lay.solve().pose_world and apart(solved.pose_world, (4.0, 0.0)) <= CELL
    # A long thing seen from the side, the mark near one end, someone on the stop before the mark and someone at work
    # behind the other end. The standard goal is then on the robot's own line, and told from it that person would stand
    # on the side the robot was sent to: 4.2 m from the mark, level with them. The seen face says they do not.
    long = Box(cx=5.3, cy=0.0, yaw=math.pi, sx=0.6, sy=5.0, sz=1.1)
    spots = [(6.3, -1.9, math.pi), (4.0, 1.6, 0.0)]
    lay = pictured(looking((2.0, -4.2), (5.0, 0.4)), [long], spots, (5.0, 1.6, 0.8))
    at, standard = np.array(lay.target.world[:2]), np.array(lay.standard.pose_world[:2])
    assert not lay.standard.flags.standard_goal_frontal and float((np.array(spots[0][:2]) - at) @ (standard - at)) > 0
    solved = in_order(lay, ask(VIEW, 1, target=True))
    assert solved.note.not_ahead_of == [] and solved.note.keys.face_m == 0.0 and flags(solved) == set()
    assert solved.note.target_distance_m <= 1.05 and nearest(solved.pose_world, spots) >= 1.0 - 5e-4


def test_a_watched_thing_without_a_face_is_joined_on_the_side_the_standard_goal_stands_on():
    low, top = Box(cx=6.0, cy=0.4, yaw=0.0, sx=1.0, sy=1.0, sz=0.75), (6.0, 0.4, 0.75)       # the mark is on its level top
    for robot, row in ((SIDE, ROW), (HERE, ROW_OPEN)):                    # from SIDE the robot stood beyond the thing
        lay = pictured(robot, [low], row, top)
        solved = in_order(lay, ask(VIEW, 1, 2, 3, target=True))
        place = solved.pose_world[:2]
        assert solved.note.keys.face_m is None and solved.note.not_ahead_of == [1, 2, 3] and flags(solved) == set()
        assert ahead_of(place, row, lay.target.world) <= 5e-4 and 1.0 - 5e-4 <= nearest(place, row) <= 1.1
    across = pictured(SIDE, [low], [(7.4, 0.4, math.pi)], top)            # someone across it, listed: not joined
    solved = in_order(across, ask(VIEW, 1, target=True))
    assert solved.note.not_ahead_of == [] and solved.pose_world == across.solve().pose_world
    assert solved.note.target_distance_m <= 1.1


@pytest.mark.parametrize("answer", [[ask(SHARED, 1, 2, 3)], []])
def test_people_before_a_thing_that_stands_free_leave_the_robot_on_the_side_of_its_face_it_saw(answer):
    # Three people stand 0.7 m round a point 1.2 m before the face and attend to one another. Their own spaces, with or
    # without the space between them, take the front of the ring a standoff round the mark, and no view is stated.
    # The nearest free cell of that ring was behind the thing, 0.94 m past its face.
    talk = watching([(4.8, 1.1), (4.19, 0.05), (5.41, 0.05)], (4.8, 0.4))
    lay = pictured(looking((2.0, -4.6), (5.4, 0.2)), [WATCHED], talk, MARKED)
    solved = in_order(lay, *answer)
    place, keys = solved.pose_world[:2], solved.note.keys
    assert (keys.face_m, keys.excluded_m) == (0.0, 0.0) and 6.0 - place[0] >= 0.2 and solved.note.not_ahead_of == []
    assert 1.0 - 5e-4 <= nearest(place, talk) <= 1.1 and solved.note.target_distance_m <= 1.3
    assert "exclusion_broken" not in flags(solved) and "far_from_target" not in flags(solved)


@pytest.mark.parametrize("robot", [SIDE, HERE])
def test_with_nobody_near_a_thing_that_stands_free_the_place_is_the_standard_goal_before_its_face(robot):
    lay = pictured(robot, [WATCHED], [], MARKED)
    solved = in_order(lay)
    assert lay.standard.flags.standard_goal_frontal and apart(lay.standard.pose_world, (5.0, 0.4)) <= 0.01
    assert apart(solved.pose_world, lay.standard.pose_world) <= CELL and solved.note.not_ahead_of == []
    assert solved.note.keys.model_dump(exclude={"preference_m", "standard_m"}) == dict(excluded_m=0.0, excluded_total_m=0.0,
                                                                                    turn_m=None, face_m=0.0)
    passed = pictured(robot, [WATCHED], [(3.5, -2.6, 0.5)], MARKED).solve()                   # a passer-by, listed nowhere
    assert apart(passed.pose_world, solved.pose_world) <= 1e-6 and passed.note.keys == solved.note.keys


def test_where_a_place_in_turn_is_used_neither_rule_of_the_seen_side_is_applied():
    # People who wait watch the thing they wait at, and an answer may say so. The place in turn already says where
    # the robot stands among them.
    lay = waiting(FILE4)
    for after, view in ((ask(AFTER, 1, 2, 3, 4), ask(VIEW, 1, target=True)), (ask(AFTER, 1, 2, 3, 4), ask(VIEW, 1, 2, 3, 4, target=True)),
                        (ask(AFTER, 1, 2, 3), ask(VIEW, 1, 2, 3, 4, target=True))):
        plain, also = lay.solve(after), lay.solve(after, view)
        assert same(also, plain) and also.uses[1].broken_m == 0.0 and apart(also.pose_world, (0.0, 5.2)) <= 0.1
        assert also.note.keys.face_m is None and also.note.not_ahead_of == [] and flags(also) == set()
        assert (plain.note.not_ahead_reason, also.note.not_ahead_reason) == (None, "a place in turn is used")
    # A file of three straight out from the face of a thing that stands free and is seen at an angle, the mark on that
    # face 0.8 m beside the file.
    out = np.array([math.cos(math.radians(205.0)), math.sin(math.radians(205.0))])               # where the face looks
    middle = np.array([6.3, 1.0]) + 0.2 * out                                                    # of the face
    file = [(*(middle + gap * out), math.radians(25.0)) for gap in (1.0, 2.0, 3.0)]
    mark = (*(middle - 0.8 * np.array([-out[1], out[0]])), 1.6)
    lay = pictured(HERE, [Box(cx=6.3, cy=1.0, yaw=math.radians(205.0), sx=0.4, sy=2.0, sz=1.9)], file, mark)
    assert solver._seen_face(lay.observation, lay.target, TH) is not None                        # the mark lies on a face
    plain = in_order(lay, ask(AFTER, 1, 2, 3))
    assert apart(plain.pose_world, middle + 4.0 * out) <= 0.1 and plain.note.keys.face_m is None
    for view in (ask(VIEW, 1, target=True), ask(VIEW, 1, 2, 3, target=True)):
        assert same(lay.solve(ask(AFTER, 1, 2, 3), view), plain)
    # A thing with a face on every side, the mark on the one beside the one the people wait at. The place in turn is
    # 0.45 m behind the plane of the marked face, and the robot stands on it.
    file = [(5.0, 0.1, math.pi / 2), (5.0, -0.9, math.pi / 2), (5.0, -1.9, math.pi / 2)]
    lay = pictured(HERE, [Box(cx=5.0, cy=1.5, yaw=-math.pi / 2, sx=0.8, sy=0.8, sz=1.9)], file, (4.6, 1.5, 1.2))
    assert solver._seen_face(lay.observation, lay.target, TH) == near((-1.0, 0.0), abs=5e-3)
    solved = in_order(lay, ask(AFTER, 1, 2, 3), ask(VIEW, 1, target=True))
    assert apart(solved.pose_world, (5.0, -2.9)) <= 0.1 and solved.pose_world[0] - 4.6 >= 0.4
    assert solved.note.keys.face_m is None and solved.note.not_ahead_of == [] and same(solved, lay.solve(ask(AFTER, 1, 2, 3)))


def test_people_who_wait_and_are_stated_only_as_watching_the_target_are_not_cut_in_on():
    # The answer's fault: no come_after. The robot went to the thing, 1.03 m from it, past all four. It now stands
    # beside the last of them, no nearer the thing than they are.
    solved = in_order(waiting(FILE4), ask(VIEW, 1, 2, 3, 4, target=True))
    place = solved.pose_world[:2]
    assert solved.note.not_ahead_of == [1, 2, 3, 4] and solved.note.keys.turn_m is None and flags(solved) == set()
    assert apart(place, THING_XY) >= apart(FILE4[3], THING_XY) and 1.0 - 5e-4 <= apart(place, FILE4[3]) <= 1.1


def test_a_person_who_is_the_target_has_no_seen_face_and_nobody_is_stood_back_from():
    # Four in a circle, the robot needs one of them: the others, listed as watching the one who is the target, change
    # nothing.
    four = [(0.8 * math.cos(turn), 0.8 * math.sin(turn), turn + math.pi) for turn in (0.0, math.pi / 2, math.pi, -math.pi / 2)]
    lay = needing_the_first(four, (-5.3, 0.5))
    careful = [ask(SHARED, 1, 2, 3, 4), ask(VISIBLE, 1), ask(FACE, 1)]
    plain, also = lay.solve(*careful), lay.solve(*careful, ask(VIEW, 2, 3, 4, target=True))
    assert same(also, plain) and also.uses[3].broken_m == 0.0 and apart(plain.pose_world, (0.98, 1.03)) <= 0.08
    assert plain.note.keys.face_m is None and also.note.not_ahead_of == []
    assert (plain.note.not_ahead_reason, also.note.not_ahead_reason) == (None, "the target is a person")
    # Someone seen from behind: the robot comes round to their side, past the plane of their back.
    lay = layout(HERE, [(5.0, 0.0, 0.0)], needed=1)
    alone, round_them = lay.solve(), lay.solve(ask(VISIBLE, 1), ask(FACE, 1))
    assert alone.pose_world[:2] == near((3.95, 0.0)) and round_them.pose_world[:2] == near((5.5, -0.9))
    assert alone.note.keys.face_m is None and round_them.note.keys.face_m is None
    others = layout(HERE, [(5.0, 0.0, 0.0)] + watching([(7.6, 0.9), (7.6, -0.9), (8.3, 0.0)], (5.0, 0.0)), needed=1)
    watched = others.solve(ask(VIEW, 2, 3, 4, target=True))               # three who watch them from the other side
    assert watched.pose_world[:2] == near((3.95, 0.0)) and watched.note.not_ahead_of == [] and watched.note.keys.face_m is None
    assert watched.note.not_ahead_reason == "the target is a person"
    # The same person on a picture, where the fit does find a face: on their back, looking at the robot. No key is made
    # of it, with or without an answer, and the robot still comes round to their side, past the plane of their back.
    seen = pictured(HERE, [], [(5.0, 0.0, 0.0)], (4.8, 0.0, 1.1))
    assert solver._seen_face(seen.observation, seen.target, TH) == near((-1.0, 0.0), abs=5e-3)
    assert solver.target_person(seen.observation, seen.target, seen.people, TH) == 1
    alone, faced, round_them = in_order(seen), in_order(seen, ask(FACE, 1)), in_order(seen, ask(VISIBLE, 1), ask(FACE, 1))
    assert alone.note.keys.face_m is None and faced.note.keys.face_m is None and round_them.note.keys.face_m is None
    assert all(candidate.keys.face_m is None for candidate in alone.candidates + round_them.candidates)
    assert apart(alone.pose_world, (3.95, 0.0)) <= CELL and round_them.pose_world[0] >= 5.4 and flags(round_them) == set()


def test_where_the_answer_says_the_robot_came_for_a_person_the_seen_face_is_not_read():
    # Someone seen from behind, whom the robot needs, and the list of people has them 0.67 m from where they stand: too
    # far to be known as the target. The target is then a thing to the solver, and the fit finds a face on their back
    # that looks at the robot. With no answer the robot stops before that face, behind them.
    seen = pictured(HERE, [], [(5.0, 0.0, 0.0)], (4.8, 0.0, 1.1))
    lay = Layout(seen.observation, [person(1, 5.6, 0.3, 0.0)], seen.target, seen.standard)
    assert solver._seen_face(lay.observation, lay.target, TH) == near((-1.0, 0.0), abs=5e-3)
    assert solver.target_person(lay.observation, lay.target, lay.people, TH) is None
    alone = in_order(lay)
    assert alone.note.keys.face_m == 0.0 and alone.pose_world[0] <= 4.0
    # An answer that names them to face, and says the place is to be visible to them, says the robot came for them: the
    # place is chosen by their front. The face is then not read, and the robot comes round to where they look, past
    # the plane of their back; ranked above the answer, the face held it behind them.
    for answer in ([ask(VISIBLE, 1), ask(FACE, 1)], [ask(FACE, 1), ask(VISIBLE, 1)]):
        by_them = in_order(lay, *answer)
        assert by_them.note.keys.face_m is None and all(candidate.keys.face_m is None for candidate in by_them.candidates)
        assert by_them.pose_world[0] >= 5.4 and by_them.uses[answer[0].kind == FACE].broken_m <= 0.2 and flags(by_them) == set()
    # A `visible_to` the solver cannot use says nothing, and the face is read as before.
    unusable = in_order(lay, ask(VISIBLE, 7), ask(FACE, 1))
    assert unusable.note.keys.face_m == 0.0 and unusable.pose_world[:2] == alone.pose_world[:2]
    # Nor is the face left unread for a `visible_to` without a `face` of the same person. The answer then does not say
    # the robot came for them, and the robot is held at the plane of the face, beside them: it does not come round.
    for answer in ([ask(VISIBLE, 1)], [ask(VISIBLE, 1), ask(FACE, target=True)]):
        held = in_order(lay, *answer)
        assert held.note.keys.face_m == 0.0 and held.pose_world[0] <= 4.85 and held.uses[0].broken_m >= 1.0


def test_being_visible_to_someone_the_robot_did_not_come_for_does_not_send_it_behind_the_thing():
    # With any usable `visible_to` in the answer the seen face was not read, and the robot went behind a real thing
    # again, to be seen by somebody it was not sent to. Each of these four has a `visible_to` and no `face` of the same
    # person, and the place is before the face in each.
    def before_the_face(lay, *answer, face_x):
        solved = in_order(lay, *answer)
        assert solved.note.keys.face_m == 0.0 and face_x - solved.pose_world[0] >= 0.2 and solved.note.keys.excluded_m == 0.0
        assert all(candidate.keys.face_m is not None for candidate in solved.candidates) and flags(solved) == set()
        return solved
    # Three who talk before a thing that stands free, stated as a shared space, with the place to be visible to one of
    # them and the robot to face the target: it stood 0.94 m behind the face.
    talk = watching([(4.8, 1.1), (4.19, 0.05), (5.41, 0.05)], (4.8, 0.4))
    lay = pictured(looking((2.0, -4.6), (5.4, 0.2)), [WATCHED], talk, MARKED)
    for answer in ([ask(SHARED, 1, 2, 3), ask(VISIBLE, 1), ask(FACE, target=True)], [ask(SHARED, 1, 2, 3), ask(VISIBLE, 1)]):
        solved = before_the_face(lay, *answer, face_x=6.0)
        assert 1.0 - 5e-4 <= nearest(solved.pose_world, talk) <= 1.1 and solved.note.target_distance_m <= 1.5
    # The row with a passer-by far behind it, seen from the side, all four stated as watching the target and the place
    # to be visible to one of the row: it stood 0.98 m behind the face. It now stands where the view alone leaves it.
    lay = pictured(SIDE, [WATCHED], FAR_BACK, MARKED)
    view = ask(VIEW, 1, 2, 3, 4, target=True)
    solved = before_the_face(lay, view, ask(VISIBLE, 2), face_x=6.0)
    assert apart(solved.pose_world, in_order(lay, view).pose_world) <= CELL and solved.uses[1].broken_m == 0.0
    # A low thing with someone behind it who looks away from the robot, and the place to be visible to them: it stood
    # 0.85 m behind the face. Before the face they do not see it, and the constraint is recorded as broken.
    low = Box(cx=5.3, cy=0.0, yaw=math.pi, sx=0.6, sy=2.4, sz=1.1)
    lay = pictured(HERE, [low], [(6.2, 0.0, 0.0)], (5.0, 0.0, 0.8))
    for answer in ([ask(VISIBLE, 1)], [ask(VISIBLE, 1), ask(FACE, target=True)]):
        solved = before_the_face(lay, *answer, face_x=5.0)
        assert solved.note.target_distance_m <= 1.05 and solved.uses[0].broken_m >= 1.0
    # One bystander beside the thing who looks past it, and the place to be visible to them: it stood 0.40 m behind
    # the face.
    lay = pictured(HERE, [WATCHED], [(5.2, 2.2, 0.0)], MARKED)
    solved = before_the_face(lay, ask(VISIBLE, 1), face_x=6.0)
    assert solved.note.target_distance_m <= 1.05 and 1.0 - 5e-4 <= apart(solved.pose_world, (5.2, 2.2)) <= 1.1


# ------------------------------------------------------------------------------- the limits that are stated, and stay
def in_order(lay, *answer, th=TH):
    """Solves, and asserts the solver's own properties: the place is a selectable cell; it breaks no exclusion
    where some cell breaks none; and no selectable cell comes before it in the order, taken here key by key from
    the values of every cell."""
    solved = lay.solve(*answer, th=th)
    cells, worked = lay.worked(*answer, th=th)
    floor = solver.floor(lay.observation, lay.people, th)
    assert floor.selectable[floor.index(*solved.pose_world[:2])] and len(cells) == floor.selectable.sum()
    left = np.ones(len(cells), dtype=bool)
    for name in ("excluded_m", "excluded_total_m", "turn_m", "face_m", "preference_m", "standard_m", "robot_m"):
        if name in worked.keys:
            slack = th.grid_resolution_m if name in ("turn_m", "preference_m") else 0.0
            left &= worked.keys[name] <= worked.keys[name][left].min() + slack + 1e-9
    assert solved.pose_robot[:2] == near(tuple(cells[np.argmax(left)]))
    if (worked.keys["excluded_m"] == 0.0).any():
        assert solved.note.keys.excluded_m == 0.0 and not solved.flags.exclusion_broken
    return solved


def circle(n, radius=None, side=None, turned=0.35):
    """`n` people in a circle who look at its middle, with `side` between neighbours or of the given radius."""
    radius = side / (2 * math.sin(math.pi / n)) if radius is None else radius
    return [(radius * math.cos(turn), radius * math.sin(turn), turn + math.pi) for turn in (turned + 2 * math.pi * k / n for k in range(n))]


def off_facing(place, spot):
    """How far round from where someone looks a place is, in degrees."""
    return math.degrees(abs(math.remainder(facing(spot, place) - spot[2], 2 * math.pi)))


@pytest.mark.parametrize("n, radius, least_off", [(6, 1.2, 110.0), (8, 1.3, 120.0), (10, 1.6, 125.0)])
def test_in_a_circle_of_six_or_more_the_robot_stands_behind_the_shoulder_of_the_one_it_needs(n, radius, least_off):
    people = circle(n, radius)                     # 1.2, 1.0 and 1.0 m between neighbours
    needed = people[0]
    for robot in ((needed[0] * 5.0, needed[1] * 5.0 + 1.5), (-needed[0] * 5.0, -needed[1] * 5.0 + 1.0)):
        solved = in_order(needing_the_first(people, robot), ask(SHARED, *range(1, n + 1)), ask(VISIBLE, 1), ask(FACE, 1))
        place = solved.pose_world[:2]
        assert flags(solved) == set() and min(apart(place, spot) for spot in people) >= 1.0 - 5e-4
        assert 1.0 - 5e-4 <= apart(place, needed) <= 1.2 and solved.uses[0].broken_m == 0.0
        assert off_facing(place, needed) >= least_off and solved.uses[1].broken_m >= 0.9      # own space wins over being seen


def test_six_in_a_circle_two_metres_apart_leave_the_robot_behind_the_shoulder_as_well():
    people = circle(6, side=2.0)                   # the rhombus of each side keeps it there, whatever the spacing
    for turn in (0.35 + 0.3, 0.35 + 1.9, 0.35 - 1.9, 0.35 + 3.0):
        robot = (6.5 * math.cos(turn), 6.5 * math.sin(turn))
        solved = in_order(needing_the_first(people, robot), ask(SHARED, 1, 2, 3, 4, 5, 6), ask(VISIBLE, 1), ask(FACE, 1))
        assert flags(solved) == set() and 105.0 <= off_facing(solved.pose_world, people[0]) <= 120.0     # 112 to 113 recorded
        assert 1.0 - 5e-4 <= apart(solved.pose_world, people[0]) <= 1.2


@pytest.mark.parametrize("pitch, inside", [(1.45, True), (1.4, False), (1.2, False)])
def test_among_people_closer_together_than_two_own_spaces_the_robot_stays_outside_and_says_so(pitch, inside):
    crowd = [((i - 2) * pitch, (j - 2) * pitch, 0.3 * i - 0.2 * j) for i in range(5) for j in range(5)]
    crowd[12] = (0.0, 0.0, -math.pi / 2)           # the one needed, in the middle of 25, looks towards the robot
    robot = (0.4, -(2 * pitch + 3.5))
    lay = layout(looking(robot, (0.0, 0.0)), crowd, needed=13)
    solved = in_order(lay, ask(VISIBLE, 13), ask(FACE, 13))
    away = apart(solved.pose_world, (0.0, 0.0))
    assert solved.note.target_distance_m == near(away) and not solved.flags.exclusion_broken
    if inside:                                     # 1.45 m apart leaves cells a metre from four people at once
        assert away <= 1.1 and flags(solved) == set()
    else:                                          # 1.41 m or less leaves none: any cell outside beats every cell inside
        assert 3.2 <= away <= 3.7 and flags(solved) == {"far_from_target"} and solved.note.own_space_broken_m == {}


@pytest.mark.parametrize("robot", [(3.5, 6.5), (-4.0, 5.0)])
def test_a_file_along_a_front_with_the_mark_at_its_middle_is_ended_at_its_head(robot):
    file = [(-1.3, 0.8), (-0.5, 0.9), (0.3, 0.9)]  # served at the left end: the head is the farthest from the mark
    solved = in_order(waiting(file, robot, look_at=(0.3, 1.0)), ask(AFTER, 1, 2, 3))
    found = turn(file, (1, 2, 3), robot=robot)
    assert found.last == 1 and found.place == near((-1.3 - 0.8 / math.hypot(0.8, 0.1), 0.8 - 0.1 / math.hypot(0.8, 0.1)))
    assert apart(solved.pose_world, found.place) <= 0.1 and flags(solved) == set() and solved.note.joined_waiting == []
    assert apart(solved.pose_world, file[2]) == pytest.approx(2.6, abs=0.1)          # from the person who is really last


def test_people_one_behind_another_behind_the_end_of_a_file_are_all_taken_as_waiting():
    passing = [(-0.4, 5.8), (-0.2, 6.9), (0.4, 7.7)]
    solved = in_order(waiting(FILE4 + passing, robot=(3.5, 9.5), look_at=(0.0, 4.0)), ask(AFTER, 1, 2, 3, 4))
    assert solved.note.joined_waiting == [5, 6, 7] and flags(solved) == set()
    assert apart(solved.pose_world, FILE4[3]) == pytest.approx(4.3, abs=0.2)         # behind the last who was listed
    assert apart(solved.pose_world, (1.08, 8.48)) <= 0.08                            # where the reference solver stood
    two = in_order(waiting(FILE4 + [(0.5, 5.9), (0.9, 6.8)], robot=(3.5, 9.5), look_at=(0.0, 4.0)), ask(AFTER, 1, 2, 3, 4))
    assert two.note.joined_waiting == [5, 6] and flags(two) == set()
    # A waiting person more than a standoff past the place in turn is not taken: the robot stands in the gap before them.
    spaced = [(0.0, 1.0 + 2.2 * k) for k in range(4)]
    gap = in_order(waiting(spaced, robot=(3.5, 10.1), look_at=spaced[1]), ask(AFTER, 1, 2, 3))
    assert gap.note.joined_waiting == [] and flags(gap) == set() and apart(gap.pose_world, (0.0, 6.4)) <= 0.1


@pytest.mark.parametrize("robot, taken, back, aside", [((3.5, 8.0), 4, 0.24, 0.97), ((5.5, 3.0), 4, 0.24, 0.97),
                                                       ((-3.5, 8.0), 3, -0.8, 1.0), ((-5.5, 3.0), 3, -0.8, 1.0)])
def test_a_file_that_turns_a_corner_with_one_person_past_it_is_ended_beside_its_end(robot, taken, back, aside):
    file = FILE3 + [(0.8, 3.2)]                    # the last two are 3.2 and 3.3 m from the mark: equally far
    solved = in_order(waiting(file, robot, look_at=FILE3[2]), ask(AFTER, 1, 2, 3, 4))
    found = turn(file, (1, 2, 3, 4), robot=robot)
    assert found.last == taken and flags(solved) == set() and apart(solved.pose_world, found.place) <= 0.1
    assert behind(found.place, file[3], file[2]) == near((back, aside), abs=0.01)    # not a metre straight behind the last
    assert 1.0 - 5e-4 <= apart(solved.pose_world, file[3]) <= 1.5


FAR_BACK = ROW + watching([(0.6, 2.4)])            # the row, and a passer-by 2.7 m behind it who looks the same way


def test_an_unrelated_person_listed_as_watching_from_far_back_keeps_the_robot_back_with_them():
    # The hindmost of the listed people decides, with no bound. A row before a thing and a passer-by 2.7 m behind the
    # row, all four listed, seen from behind them all: the robot stands level with the passer-by, 1.2 m beside them and
    # 5.6 m from the thing. Nothing is flagged: far_from_target is not set where people are joined, and only
    # not_ahead_of and target_distance_m show it.
    lay = pictured(looking((-3.5, 1.0), (5.2, 0.4)), [WATCHED], FAR_BACK, MARKED)
    solved = in_order(lay, ask(VIEW, 1, 2, 3, 4, target=True))
    assert solved.note.not_ahead_of == [1, 2, 3, 4] and solved.note.not_ahead_reason is None and flags(solved) == set()
    assert solved.note.target_distance_m == pytest.approx(5.65, abs=0.1) and solved.uses[0].broken_m == 0.0       # beyond max_keep_m
    assert ahead_of(solved.pose_world, FAR_BACK, lay.target.world) <= 5e-4 and 1.0 - 5e-4 <= apart(solved.pose_world, FAR_BACK[3]) <= 1.3
    assert ahead_of(solved.pose_world, ROW, lay.target.world) < -2.0      # well behind the row that does watch


def test_where_no_floor_level_with_the_hindmost_listed_person_is_in_view_nobody_is_stood_back_with():
    # The same four from the side: the picture shows no floor level with the passer-by. The rule cannot be met, and it
    # is not applied at all, to the row no more than to them. The robot does not back off to the farthest floor in view,
    # 6.7 m from the thing with the exclusion broken, as it did when the least-broken cell was taken. It stands where
    # the four views of the thing leave it: 2.0 m from the mark at the flank, ahead of the row, inside nobody's own
    # space. No flag is set, and the record says why nobody was joined.
    lay = pictured(SIDE, [WATCHED], FAR_BACK, MARKED)
    solved = in_order(lay, ask(VIEW, 1, 2, 3, 4, target=True))
    assert solved.note.not_ahead_of == [] and solved.note.not_ahead_reason == "no cell is level with or behind person 4"
    assert flags(solved) == set() and solved.note.own_space_broken_m == {} and nearest(solved.pose_world, FAR_BACK) >= 1.0 - 5e-4
    assert (solved.note.keys.excluded_m, solved.note.keys.face_m, solved.uses[0].broken_m) == (0.0, 0.0, 0.0)
    assert solved.note.target_distance_m == pytest.approx(2.0, abs=0.1) and ahead_of(solved.pose_world, ROW, lay.target.world) >= 1.0
    # Listed without the passer-by, the row is joined from the same place.
    row = in_order(lay, ask(VIEW, 1, 2, 3, target=True))
    assert row.note.not_ahead_of == [1, 2, 3] and ahead_of(row.pose_world, ROW, lay.target.world) <= 5e-4 and flags(row) == set()
    # far_from_target is judged as without the rule. A band of floor a metre either side of the robot, a thing 6 m ahead
    # and someone 2.5 m behind the robot who watches it: no floor is level with them. The robot stops at the band's edge,
    # out of their view, 5.6 m from the thing, and that is flagged as it is with no constraint.
    band = lambda ahead, left: (ahead <= 1.0 + 1e-9) & (ahead >= -1.0 - 1e-9)                    # noqa: E731
    cut_off = layout(HERE, watching([(-2.5, 1.0)], (6.0, 0.0)), thing=(6.0, 0.0), floor=band)
    stopped = in_order(cut_off, ask(VIEW, 1, target=True))
    assert stopped.note.not_ahead_of == [] and stopped.note.not_ahead_reason == "no cell is level with or behind person 1"
    assert flags(stopped) == {"far_from_target"} and stopped.pose_world[0] == near(1.0) and stopped.uses[0].broken_m == 0.0
    assert flags(cut_off.solve()) == {"far_from_target", "no_usable_constraint"}


def test_where_the_only_floor_level_with_those_who_watch_is_inside_their_own_space_nobody_is_stood_back_with():
    # A mark on a bare wall and one person 4 m along the wall who looks at it, seen from the room. Floor level with
    # them is in view, but only a sliver of it, all within a standoff of them, and plenty of floor breaks nothing. The
    # rule could be met only inside their own space, so it is not applied, and the record says why. Applied, it was
    # traded against their own space as any two exclusions are: the robot stood 0.86 m from them, 0.14 m inside their
    # own space, 0.09 m ahead of them and 4.04 m from the mark, with exclusion_broken set. It now stands a metre before
    # the mark, out of their view, 4.4 m from them, with nothing broken and no flag: where it stops unasked, to 0.4 m.
    reader = watching([(6.0, -3.6)], (6.6, 0.4))
    lay = pictured(looking((1.0, 1.0), (6.0, -0.5)), [BEHIND_IT], reader, (6.6, 0.4, 1.5))
    floor = solver.floor(lay.observation, lay.people, TH)
    level = [xy for xy in floor.centres()[floor.selectable] if ahead_of(xy, reader, lay.target.world) <= 5e-4]
    assert level and all(apart(xy, reader[0]) < 1.0 - 5e-4 for xy in level)
    solved = in_order(lay, ask(VIEW, 1, target=True))
    assert solved.note.not_ahead_of == [] and solved.note.not_ahead_reason == BROKEN_MORE.format(1) and flags(solved) == set()
    assert solved.note.own_space_broken_m == {} and (solved.note.keys.excluded_m, solved.uses[0].broken_m) == (0.0, 0.0)
    assert apart(solved.pose_world, reader[0]) >= 4.0 and solved.note.target_distance_m <= 1.05
    assert ahead_of(solved.pose_world, reader, lay.target.world) >= 4.0   # ahead of them, as every place outside their own space is
    unasked = in_order(lay)
    assert unasked.note.target_distance_m <= 1.05 and flags(unasked) == {"no_usable_constraint"}
    assert apart(solved.pose_world, unasked.pose_world) <= 0.4 and unasked.note.not_ahead_reason is None


def test_one_person_far_from_a_thing_who_is_listed_as_watching_it_keeps_the_robot_back_with_them():
    # The rule reads every view of the target as people to join. It does not look at how many they are, how far they
    # stand or whether the answer also asks to be near the target. One onlooker 4 m before a thing at a wall, the robot
    # behind them: unasked it stops a metre before the face; with the view stated it stands 1.1 m beside the onlooker,
    # 4.2 m from the thing, with nothing flagged. A range about the target changes nothing: it is a preference.
    lay = pictured(looking((-1.5, -1.2), (6.0, 0.4)), [WATCHED, BEHIND_IT], watching([(2.0, 0.9)]), MARKED)
    assert in_order(lay).note.target_distance_m <= 1.05
    for answer in ([ask(VIEW, 1, target=True)], [ask(VIEW, 1, target=True), ask(WITHIN, target=True, between=(0.5, 1.5))]):
        solved = in_order(lay, *answer)
        assert solved.note.not_ahead_of == [1] and flags(solved) == set() and solved.note.keys.excluded_m == 0.0
        assert solved.note.target_distance_m == pytest.approx(4.2, abs=0.1) and 1.0 - 5e-4 <= apart(solved.pose_world, (2.0, 0.9)) <= 1.15
    assert solved.uses[1].broken_m == pytest.approx(solved.note.target_distance_m - 1.5, abs=2e-3)


def test_a_person_found_on_the_watched_thing_is_taken_for_the_target_and_neither_rule_is_applied():
    # A row watches a thing that stands free, and the list of people has a fourth "person" on the thing itself: a false
    # find, 0.25 m behind the mark as a body would be. They are the target person then. A person has no seen face and
    # nobody is stood back from, so the fault the two rules were added for is back: the robot goes round behind the
    # thing. The record shows it by a null face_m and by the reason.
    for robot, row in ((SIDE, ROW), (HERE, ROW_OPEN)):
        seen = pictured(robot, [WATCHED], row, MARKED)
        camera, mark = seen.observation.camera.T_world_camera[:2, 3], np.array(seen.target.world[:2])
        on_it = mark + TH.person_radius_m * (mark - camera) / math.hypot(*(mark - camera))
        lay = Layout(seen.observation, seen.people + [person(4, float(on_it[0]), float(on_it[1]), math.pi)], seen.target, seen.standard)
        assert solver.target_person(lay.observation, lay.target, lay.people, TH) == 4
        solved = in_order(lay, ask(VIEW, 1, 2, 3, target=True))
        assert solved.note.not_ahead_of == [] and solved.note.not_ahead_reason == "the target is a person"
        assert solved.note.keys.face_m is None and flags(solved) == set() and solved.note.target_distance_m <= 1.2
        assert solved.pose_world[0] - 6.0 >= 0.8                              # 0.8 m or more behind the face, out of view
        assert in_order(seen, ask(VIEW, 1, 2, 3, target=True)).note.not_ahead_of == [1, 2, 3]    # without the false find


def test_with_the_mark_on_a_narrow_side_of_a_thing_who_is_joined_turns_on_centimetres():
    # A robot at the flank of a watched thing gets its target point on the narrow side face, which looks along -y from
    # y = -0.4. The people before the true front are on the side the robot was sent to only where their centre lies
    # before the plane of that side face. With the nearest of a row 0.02 m before that plane, that one person is joined
    # and the robot stands behind the row, 4.7 m from the mark and 2.4 m behind the plane of the face it saw. With them
    # 0.02 m behind it, nobody is joined and the robot stays a metre before the side face, beside the thing.
    flank, mark = looking((6.3, -5.0), (6.3, -0.4)), (6.3, -0.4, 1.6)
    for first, joined in ((-0.42, [1]), (-0.38, [])):
        row = watching([(3.3, first), (3.25, first + 1.1), (3.3, first + 2.2)])
        lay = pictured(flank, [WATCHED], row, mark)
        solved = in_order(lay, ask(VIEW, 1, 2, 3, target=True))
        assert solved.note.not_ahead_of == joined and flags(solved) == set() and solved.note.keys.excluded_m == 0.0, first
        if joined:
            assert solved.note.target_distance_m == pytest.approx(4.66, abs=0.1) and solved.note.keys.face_m == pytest.approx(2.4, abs=0.1)
            assert solved.pose_world[0] <= 3.3 and nearest(solved.pose_world, row) <= 1.1
        else:
            assert solved.note.not_ahead_reason == "none of them stands on the side the robot was sent to"
            assert solved.note.target_distance_m <= 1.05 and solved.note.keys.face_m == 0.0 and solved.pose_world[1] <= -1.3


def test_a_person_the_robot_needs_whom_the_list_of_people_has_elsewhere_is_a_thing_to_the_rule_about_those_who_watch():
    # Four in a circle, and the robot needs the first. Listed where they stand they are the target person, and the three
    # others, stated as watching the target, change nothing. Listed 0.67 m off, as a people step can put them, they are
    # not known to be the target: the target is a thing, the three are people who watch it, and the robot is kept level
    # with the hindmost of them, across the circle from the one it needs, 1.9 m from the point marked on them. The
    # answer's face and visible_to of that person switch the seen face off (above); they do not switch this off.
    four = [(0.8 * math.cos(turn), 0.8 * math.sin(turn), turn + math.pi) for turn in (0.0, math.pi / 2, math.pi, -math.pi / 2)]
    known = needing_the_first(four, (-5.3, 0.5))
    careful, view = [ask(SHARED, 1, 2, 3, 4), ask(VISIBLE, 1), ask(FACE, 1)], ask(VIEW, 2, 3, 4, target=True)
    assert same(known.solve(*careful, view), known.solve(*careful)) and known.solve(*careful, view).note.not_ahead_of == []
    elsewhere = Layout(known.observation, [person(1, 1.4, 0.3, math.pi)] + known.people[1:], known.target, known.standard)
    assert solver.target_person(elsewhere.observation, elsewhere.target, elsewhere.people, TH) is None
    plain, also = in_order(elsewhere, *careful), in_order(elsewhere, *careful, view)
    assert plain.note.not_ahead_of == [] and plain.note.target_distance_m <= 1.4
    assert also.note.not_ahead_of == [2, 3, 4] and flags(also) == set() and also.note.keys.excluded_m == 0.0
    assert also.note.target_distance_m == pytest.approx(1.9, abs=0.1) and apart(also.pose_world, plain.pose_world) >= 2.0


def test_a_round_post_is_taken_for_a_face_on_the_side_the_camera_sees():
    # The fit of the seen face takes the near side of a round post of 0.2 m radius for a flat upright surface. Its
    # front then points at the camera, to within 3 degrees, wherever the robot stands: it is not a front of the post.
    for robot in (HERE, SIDE, OTHER):
        to_post = np.array([6.0 - robot.x, 0.4 - robot.y]) / math.hypot(6.0 - robot.x, 0.4 - robot.y)
        lay = pictured(robot, [Cylinder(x=6.0, y=0.4, radius=0.2, height=1.9)], [], (*(np.array([6.0, 0.4]) - 0.2 * to_post), 1.5))
        front = solver._seen_face(lay.observation, lay.target, TH)
        assert front is not None
        c, s = math.cos(robot.yaw), math.sin(robot.yaw)                       # the front is given in the robot's frame
        towards = float(np.array([c * front[0] - s * front[1], s * front[0] + c * front[1]]) @ -to_post)
        assert math.degrees(math.acos(min(1.0, towards))) <= 3.0 and in_order(lay).note.keys.face_m == 0.0


def test_people_who_watch_a_thing_and_are_stated_as_looking_straight_ahead_are_not_joined():
    # A view with `target: false` says nothing of the target. The row's zones are kept and the robot stands 2.1 m from
    # the thing, 1.5 m ahead of the row, as it did before the seen side was read.
    lay = pictured(SIDE, [WATCHED], ROW, MARKED)
    solved = in_order(lay, ask(VIEW, 1, 2, 3))
    assert solved.note.not_ahead_of == [] and solved.note.keys.face_m == 0.0 and flags(solved) == set()
    assert ahead_of(solved.pose_world, ROW, lay.target.world) >= 1.0 and solved.note.target_distance_m == pytest.approx(2.1, abs=0.1)


def test_people_round_a_thing_without_a_face_are_told_apart_by_the_robots_own_line_of_approach():
    # Three people watch a thing from one side and two from the other, and the robot comes along the gap between the two
    # sides. The side it was sent to is the standard goal's, on its own line: one person of each side stands on it, the
    # hindmost of the two is 0.6 m before the thing, and the robot stops 1.77 m from it, nearer than any of the five.
    lay = layout(HERE, ROUND_IT, thing=(5.0, 0.0))
    solved = in_order(lay, ask(VIEW, 1, 2, 3, 4, 5, target=True))
    assert solved.note.not_ahead_of == [1, 4] and solved.note.keys.face_m is None and flags(solved) == set()
    assert apart(solved.pose_world, (3.25, 0.25)) <= CELL and solved.note.target_distance_m == pytest.approx(1.77, abs=0.05)
    assert solved.note.target_distance_m < min(apart(spot, (5.0, 0.0)) for spot in ROUND_IT)


def test_told_to_face_and_to_be_visible_to_someone_behind_a_thing_who_looks_away_the_robot_goes_behind_it():
    # The seen face is not read where the answer names a person to face and says the place is to be visible to them:
    # the solver takes it that the robot came for that person, whatever the target is. Someone stands behind a low
    # thing, which is the target, and looks away from the robot. Told to face them and to be visible to them, the robot
    # goes round the thing: 0.85 m behind its face and 1.72 m from the mark, beside them, where it is less far round
    # from where they look than before the face (0.75 m of violation against 1.63 m) and still not seen. Nothing is
    # flagged; the record shows it by a null face_m.
    low = Box(cx=5.3, cy=0.0, yaw=math.pi, sx=0.6, sy=2.4, sz=1.1)
    lay = pictured(HERE, [low], [(6.2, 0.0, 0.0)], (5.0, 0.0, 0.8))
    assert solver.target_person(lay.observation, lay.target, lay.people, TH) is None
    assert solver._seen_face(lay.observation, lay.target, TH) == near((-1.0, 0.0), abs=5e-3)
    for answer in ([ask(VISIBLE, 1), ask(FACE, 1)], [ask(FACE, 1), ask(VISIBLE, 1)]):
        solved = in_order(lay, *answer)
        assert solved.note.keys.face_m is None and solved.pose_world[0] - 5.0 >= 0.8 and flags(solved) == set()
        assert solved.note.target_distance_m == pytest.approx(1.72, abs=0.1) and solved.note.keys.excluded_m == 0.0
        assert 0.5 <= solved.uses[answer[0].kind == FACE].broken_m <= 1.0    # still behind their shoulder, by less
    # Where they look at the robot over the thing, the same answer leaves it at the standard goal before the face: being
    # seen by them needs no going round. That holds with nobody else before the thing (see the talkers below).
    towards = pictured(HERE, [low], [(6.2, 0.0, math.pi)], (5.0, 0.0, 0.8))
    solved = in_order(towards, ask(VISIBLE, 1), ask(FACE, 1))
    assert solved.note.keys.face_m is None and apart(solved.pose_world, (4.0, 0.0)) <= CELL and solved.uses[0].broken_m == 0.0
    # The exception acts wherever the named person stands. Three talk before a thing that stands free, and the answer
    # states their shared space, the place to be visible to the first of them and the robot to face that same person:
    # the robot goes behind the thing, 0.94 m behind its face and 1.08 m from the mark, where it hides the robot from
    # them. Nothing is flagged and face_m is null; faced at the target instead, the place is before the face.
    talk = watching([(4.8, 1.1), (4.19, 0.05), (5.41, 0.05)], (4.8, 0.4))
    lay = pictured(looking((2.0, -4.6), (5.4, 0.2)), [WATCHED], talk, MARKED)
    behind = in_order(lay, ask(SHARED, 1, 2, 3), ask(VISIBLE, 1), ask(FACE, 1))
    assert behind.note.keys.face_m is None and flags(behind) == set() and behind.pose_world[0] - 6.0 == pytest.approx(0.94, abs=0.02)
    assert behind.note.target_distance_m == pytest.approx(1.08, abs=0.03)
    before = in_order(lay, ask(SHARED, 1, 2, 3), ask(VISIBLE, 1), ask(FACE, target=True))
    assert before.note.keys.face_m == 0.0 and 6.0 - before.pose_world[0] >= 0.2


def test_which_end_of_a_row_that_is_the_same_on_both_sides_is_settled_by_the_standard_goal_to_the_millimetre():
    # At hand-picked places, not on a picture. Three in a row watch a face, and one place lies level with them beside
    # either end of the row: the two are the same in every exclusion, before the face, and the same in preference. The
    # standard goal tells them apart, and only where it does not, the robot. With the standard goal on the line through
    # the middle of the row the robot stands at the end it comes from; with it 2 mm to one side, at that side's end,
    # also when it comes from the other.
    row = watching([(1.0, -1.1), (1.0, 0.0), (1.0, 1.1)], (3.0, 0.0))
    ends, from_right, from_left = [(1.0, -2.15), (1.0, 2.15)], (-1.0, -4.0), (-1.0, 4.0)

    def end(standard, robot):
        worked = at_points(ends, row, [ask(VIEW, 1, 2, 3, target=True)], target=(3.0, 0.0), robot=robot, standard=standard,
                           front=(-1.0, 0.0))
        assert worked.not_ahead_of == [1, 2, 3] and worked.keys["excluded_m"] == near([0.0, 0.0])
        assert all(np.ptp(worked.keys[name]) == 0.0 for name in ("excluded_total_m", "face_m", "preference_m"))
        return worked.picks[0]
    assert end((2.0, 0.0), from_right) == 0 and end((2.0, 0.0), from_left) == 1
    assert end((2.0, 0.002), from_right) == 1 and end((2.0, 0.002), from_left) == 1
    assert end((2.0, -0.002), from_right) == 0 and end((2.0, -0.002), from_left) == 0


# ---------------------------------------------------------------------------------------------------- placeholder scenes
def test_on_every_placeholder_scene_the_place_is_first_in_the_order_and_breaks_no_exclusion(placeholder):
    th = load_config().thresholds
    solved_episodes = 0
    for folder in scene_dirs(placeholder / "eval"):
        inputs, truth = load_scene(folder)
        observation = inputs.observation
        seen = [p for p in truth.people if is_visible(p, observation, th)]
        seen.sort(key=lambda p: project(observation.camera, (p.x, p.y, th.person_mark_height_m))[0])     # from the left
        number = {p.id: n for n, p in enumerate(seen, 1)}
        people = [person(number[p.id], p.x, p.y, p.yaw) for p in seen]
        for episode in inputs.episodes:
            gt = truth.episodes[episode.episode_id]
            wanted = truth.targets[gt.target_ids[0]]
            target = oracle_target(observation, wanted, th)
            group = truth.group_for(episode.episode_id, wanted.id)
            members = [number[p.id] for p in truth.people if group is not None and p.group == group and p.id in number]
            if wanted.person_id in number and members:                    # people round a person attend to one another
                answer = [ask(SHARED, *members), ask(VISIBLE, number[wanted.person_id]), ask(FACE, number[wanted.person_id])]
            else:                                                         # and people at a thing wait their turn
                answer = [ask(AFTER, *members)] if members else []
            lay = Layout(observation, people, target, standard_goal(observation, target, th))
            solved = in_order(lay, *answer, th=th)
            assert solved.note.keys.excluded_m == 0.0 and not solved.flags.exclusion_broken and not solved.flags.far_from_target
            assert all(use.status == "used" for use in solved.uses), (inputs.scene_id, episode.episode_id)
            solved_episodes += 1
    assert solved_episodes == 44                   # two episodes a scene, of every scene as the set is generated now: the two
    #                                                rules of thumb above are constraints to solve with in each of them, and no
    #                                                more than that where people watch a thing or a picture is being taken
