"""Re-solving a stored constraint answer on tracked people (sgi.resolve)."""
import math

import pytest

from sgi.resolve import Resolver, Track, match_one_to_one
from sgi.types import Constraint, DetectedPerson

TARGET = (0.0, 0.0)                                # the counter; the line runs back along +x


def person(number, x, y, yaw=math.pi):
    return DetectedPerson(number=number, xy=(x, y), yaw=yaw, pixel=(0.0, 0.0))


def after(*numbers):
    return Constraint(because="they wait", kind="come_after", people=list(numbers))


def shared(*numbers):
    return Constraint(because="they talk", kind="keep_out_of_shared_space", people=list(numbers))


def tracks(*xy, yaw=math.pi):
    return [Track(id=i, xy=p, yaw=yaw) for i, p in enumerate(xy)]


LINE = [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]       # three waiting, 1 m apart, the first nearest the counter


def resolver(people=None, constraints=None, at_answer=None, options=None):
    people = people or [person(i + 1, *xy) for i, xy in enumerate(LINE)]
    constraints = constraints if constraints is not None else [after(1, 2, 3)]
    return Resolver(people, constraints, TARGET, at_answer or tracks(*LINE), **(options or {}))


# ---- the one-to-one match

def test_match_pairs_each_row_with_its_nearest_free_column():
    cost = [[0.1, 0.2], [0.15, 0.9]]               # greedy would give row 0 column 0, and row 1 a cost of 0.9
    assert match_one_to_one(cost, gate=0.75) == {0: 1, 1: 0}


def test_match_leaves_out_pairs_beyond_the_gate():
    assert match_one_to_one([[0.2, 5.0], [4.0, 3.0]], gate=0.75) == {0: 0}


def test_match_takes_more_columns_than_rows():
    assert match_one_to_one([[3.0, 0.1, 2.0]], gate=0.75) == {0: 1}


# ---- the first match: numbers follow tracks

def test_numbers_follow_their_tracks_as_people_move():
    r = resolver(at_answer=tracks((1.1, 0.1), (2.0, -0.1), (2.9, 0.0)))
    people, _ = r.inputs(tracks((0.4, 0.0), (1.4, 0.0), (2.4, 0.0)), robot_xy=(5.0, 0.0), t=3.0)
    assert {p.number: p.xy for p in people} == {1: (0.4, 0.0), 2: (1.4, 0.0), 3: (2.4, 0.0)}
    assert r.match_rate == 1.0


def test_a_named_person_with_no_track_is_stripped_from_the_constraints():
    named = [person(1, 1.0, 0.0), person(2, 2.0, 0.0), person(3, 6.0, 3.0)]   # 3 was mislocated by the model
    r = resolver(people=named, constraints=[after(1, 2, 3)], at_answer=tracks(*LINE))
    people, constraints = r.inputs(tracks(*LINE), robot_xy=(5.0, 0.0), t=3.0)
    assert constraints[0].people == [1, 2]
    assert r.match_rate == pytest.approx(2 / 3)
    assert 3 not in {p.number for p in people if p.xy == (6.0, 3.0)}


def test_a_track_the_model_did_not_number_gets_a_new_unlisted_number():
    r = resolver()
    people, constraints = r.inputs(tracks(*LINE, (4.0, 0.0)), robot_xy=(6.0, 0.0), t=3.0)
    new = [p for p in people if p.xy == (4.0, 0.0)]
    assert len(new) == 1 and new[0].number not in (1, 2, 3)
    assert new[0].number not in constraints[0].people


def test_the_new_number_of_a_track_stays_the_same():
    r = resolver()
    first, _ = r.inputs(tracks(*LINE, (4.0, 0.0)), robot_xy=(6.0, 0.0), t=3.0)
    second, _ = r.inputs(tracks(*LINE, (4.2, 0.0)), robot_xy=(6.0, 0.0), t=3.5)
    assert [p.number for p in first if p.xy[0] > 3.5] == [p.number for p in second if p.xy[0] > 3.5]


def test_the_match_uses_where_people_stood_when_the_answer_was_asked():
    # by the first re-solve everyone has stepped up by a slot: matched on the old positions, nobody swaps numbers
    r = resolver(at_answer=tracks(*LINE))
    people, _ = r.inputs(tracks((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)), robot_xy=(5.0, 0.0), t=3.0)
    assert {p.number: p.xy for p in people} == {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0)}


# ---- leaving the line

def test_a_named_person_who_walks_away_from_the_target_is_dropped():
    r = resolver()
    r.inputs(tracks((0.3, 0.0), (1.5, 0.0), (2.5, 0.0)), robot_xy=(5.0, 0.0), t=3.0)    # 1 is at the counter
    _, constraints = r.inputs(tracks((0.3, 1.2), (1.5, 0.0), (2.5, 0.0)), robot_xy=(5.0, 0.0), t=4.0)
    assert constraints[0].people == [2, 3]
    assert r.dropped == {1}


def test_a_dropped_person_stays_dropped_when_they_come_back():
    r = resolver()
    r.inputs(tracks((0.3, 3.0), (1.5, 0.0), (2.5, 0.0)), robot_xy=(5.0, 0.0), t=3.0)
    _, constraints = r.inputs(tracks((3.5, 0.0), (1.0, 0.0), (2.0, 0.0)), robot_xy=(5.0, 0.0), t=20.0)
    assert constraints[0].people == [2, 3]


def test_stepping_up_is_never_leaving():
    r = resolver()
    for t, step in enumerate([0.0, 0.3, 0.6, 1.0]):
        _, constraints = r.inputs(tracks(*[(x - step, y) for x, y in LINE]), robot_xy=(5.0, 0.0), t=3.0 + t)
    assert constraints[0].people == [1, 2, 3] and r.dropped == set()


def test_swaying_in_place_is_not_leaving():
    r = resolver()
    r.inputs(tracks(*LINE), robot_xy=(5.0, 0.0), t=3.0)
    _, constraints = r.inputs(tracks((1.4, 0.3), (2.3, -0.2), (3.4, 0.2)), robot_xy=(5.0, 0.0), t=4.0)
    assert constraints[0].people == [1, 2, 3]


def test_only_people_named_in_come_after_can_leave_the_line():
    r = resolver(constraints=[after(1), shared(2, 3)])
    r.inputs(tracks(*LINE), robot_xy=(5.0, 0.0), t=3.0)
    _, constraints = r.inputs(tracks((1.0, 0.0), (4.0, 0.5), (5.0, 0.5)), robot_xy=(6.0, 0.0), t=4.0)
    assert r.dropped == set() and constraints[0].people == [1]


def test_the_stored_constraints_are_not_changed():
    stored = [after(1, 2, 3)]
    r = resolver(constraints=stored)
    r.inputs(tracks((0.3, 3.0), (1.5, 0.0), (2.5, 0.0)), robot_xy=(5.0, 0.0), t=3.0)
    assert stored[0].people == [1, 2, 3]


# ---- the robot joins the line

def test_once_joined_people_behind_the_robot_are_left_out():
    r = resolver()
    r.record(place_xy=(4.0, 0.0), in_turn=True)
    r.inputs(tracks(*LINE), robot_xy=(4.1, 0.0), t=10.0)
    r.inputs(tracks(*LINE), robot_xy=(4.0, 0.1), t=11.0)                       # 1 s near a place in turn
    people, _ = r.inputs(tracks(*LINE, (5.0, 0.0)), robot_xy=(4.0, 0.0), t=11.5)
    assert r.joined
    assert all(p.xy != (5.0, 0.0) for p in people)


def test_not_joined_before_the_dwell():
    r = resolver()
    r.record(place_xy=(4.0, 0.0), in_turn=True)
    r.inputs(tracks(*LINE), robot_xy=(4.1, 0.0), t=10.0)
    people, _ = r.inputs(tracks(*LINE, (5.0, 0.0)), robot_xy=(4.0, 0.0), t=10.5)
    assert not r.joined
    assert any(p.xy == (5.0, 0.0) for p in people)


def test_not_joined_at_a_place_that_was_not_a_turn():
    r = resolver()
    r.record(place_xy=(4.0, 0.0), in_turn=False)
    for t in (10.0, 11.0, 12.0):
        r.inputs(tracks(*LINE), robot_xy=(4.0, 0.0), t=t)
    assert not r.joined


def test_leaving_the_place_before_the_dwell_starts_the_clock_again():
    r = resolver()
    r.record(place_xy=(4.0, 0.0), in_turn=True)
    r.inputs(tracks(*LINE), robot_xy=(4.0, 0.0), t=10.0)
    r.inputs(tracks(*LINE), robot_xy=(5.0, 0.0), t=10.6)                       # walked off
    r.inputs(tracks(*LINE), robot_xy=(4.0, 0.0), t=11.0)
    assert not r.joined


# ---- the line's direction, kept from when the answer was asked

def test_the_line_runs_from_its_front_to_its_back_as_the_answer_saw_it():
    assert resolver().line_direction == pytest.approx((1.0, 0.0))


def test_no_line_direction_from_one_person_or_a_short_span():
    assert resolver(constraints=[after(1)]).line_direction is None
    close = [person(1, 1.0, 0.0), person(2, 1.3, 0.2)]
    assert resolver(people=close, constraints=[after(1, 2)], at_answer=tracks((1.0, 0.0), (1.3, 0.2))).line_direction is None


def _two_at_the_front(target_world, line_direction=None):
    """Two people left at the front of a line along -y, the robot behind them; returns the solved place."""
    from sgi.config import load_config
    from sgi.solver import solve
    from sgi.types import GoalResult, GroundedTarget, Point
    from tests.helpers import floor_observation
    th = load_config().thresholds
    up = math.pi / 2
    people = [person(1, 1.0, 8.0, up), person(2, 1.0, 7.3, up)]
    target = GroundedTarget(point=Point(x=500, y=300), pixel=(320.0, 150.0), world=target_world)
    standard = GoalResult(status="ok", pose_world=(1.0, 5.0, up))
    solved = solve(floor_observation(), target, people, [after(1, 2)], standard, th, line_direction=line_direction)
    return solved.pose_world[:2]


def test_where_nobody_is_before_the_last_the_line_runs_the_way_it_was_seen():
    # the target point 5 m to the side: the two differ by 0.12 m in distance to it, so neither is before the other
    assert _two_at_the_front((-4.0, 8.5, 0.0), line_direction=(0.0, -1.0)) == pytest.approx((1.0, 6.3), abs=0.1)


def test_without_a_direction_the_line_still_starts_at_the_target_point():
    x, _ = _two_at_the_front((-4.0, 8.5, 0.0))
    assert x > 1.6                                         # off to the side, as before this change


def test_a_direction_that_agrees_with_the_target_point_changes_nothing():
    assert _two_at_the_front((1.0, 9.0, 0.0), line_direction=(0.0, -1.0)) == pytest.approx(_two_at_the_front((1.0, 9.0, 0.0)))


def test_with_a_direction_the_last_is_the_one_farthest_back_along_it():
    # three in a line along -y; the target point 5 m to the side and a little back, so that by distance to it the
    # middle one would be taken as the last
    from sgi.config import load_config
    from sgi.solver import solve
    from sgi.types import GoalResult, GroundedTarget, Point
    from tests.helpers import floor_observation
    th, up = load_config().thresholds, math.pi / 2
    people = [person(1, 1.0, 8.0, up), person(2, 0.6, 7.2, up), person(3, 1.0, 6.4, up)]
    target = GroundedTarget(point=Point(x=500, y=300), pixel=(320.0, 150.0), world=(-5.0, 7.0, 0.0))
    standard = GoalResult(status="ok", pose_world=(1.0, 5.0, up))
    solved = solve(floor_observation(), target, people, [after(1, 2, 3)], standard, th, line_direction=(0.0, -1.0))
    assert solved.pose_world[:2] == pytest.approx((1.0, 5.4), abs=0.1)


# ---- the first view's floor, with the people painted out

def _with_a_person_at(obs, xy, half_width=0.25, height=1.7):
    """The observation with a flat-fronted body standing at `xy`, facing the camera, drawn into the depth (from 5 cm
    up: lower, the solver reads a point as floor)."""
    import numpy as np
    from sgi.inputs import ObservationData
    cam = obs.camera
    H, W = obs.depth.shape
    U, V = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([(U - cam.cx) / cam.fx, (V - cam.cy) / cam.fy, np.ones_like(U)], -1) @ cam.T_world_camera[:3, :3].T
    origin = cam.T_world_camera[:3, 3]
    front_y = xy[1] - 0.13
    t = (front_y - origin[1]) / rays[..., 1]
    hit = origin + t[..., None] * rays
    body = (t > 0) & (t < obs.depth) & (np.abs(hit[..., 0] - xy[0]) <= half_width) & (hit[..., 2] > 0.05) & (hit[..., 2] < height)
    depth = obs.depth.copy()
    depth[body] = t[body]
    return ObservationData(obs.rgb, depth, cam, obs.robot), body


def test_painting_a_person_out_gives_back_the_floor_behind_them():
    import numpy as np
    from sgi.resolve import paint_out
    from tests.helpers import floor_observation
    floor = floor_observation()
    seen, body = _with_a_person_at(floor, (1.0, 5.0))
    assert body.sum() > 500
    painted = paint_out(seen, [(1.0, 5.0)], radius_m=0.4)
    assert np.allclose(painted.depth[body], floor.depth[body], rtol=1e-4)
    assert np.array_equal(painted.depth[~body], seen.depth[~body])


def test_painting_out_someone_elsewhere_changes_nothing():
    import numpy as np
    from sgi.resolve import paint_out
    from tests.helpers import floor_observation
    seen, _ = _with_a_person_at(floor_observation(), (1.0, 5.0))
    assert np.array_equal(paint_out(seen, [(3.0, 9.0)], radius_m=0.4).depth, seen.depth)


# ---- who the solver takes as joining the end stays in the line

def test_someone_taken_as_joining_twice_running_waits_from_then_on():
    r = resolver()
    now = tracks(*LINE, (4.0, 0.0))                        # track 3, unlisted: number 4
    r.inputs(now, robot_xy=(6.0, 0.0), t=3.0)
    r.record((5.0, 0.0), True, absorbed=[4])
    _, constraints = r.inputs(now, robot_xy=(6.0, 0.0), t=3.5)
    assert constraints[0].people == [1, 2, 3]              # once is not enough
    r.record((5.0, 0.0), True, absorbed=[4])
    _, constraints = r.inputs(now, robot_xy=(6.0, 0.0), t=4.0)
    assert constraints[0].people == [1, 2, 3, 4]


def test_the_line_is_kept_when_everyone_the_model_named_has_left():
    r = resolver()
    now = tracks(*LINE, (4.0, 0.0))
    for t in (3.0, 3.5):
        r.inputs(now, robot_xy=(6.0, 0.0), t=t)
        r.record((5.0, 0.0), True, absorbed=[4])
    gone = tracks((0.0, 5.0), (0.0, 6.0), (0.0, 7.0), (1.0, 0.0))      # 1-3 walked off; 4 stepped up to the front
    _, constraints = r.inputs(gone, robot_xy=(6.0, 0.0), t=20.0)
    assert constraints[0].people == [4]


def test_a_waiting_person_who_steps_off_the_line_sideways_is_dropped():
    # 2 steps 1.2 m to the side: 0.33 m farther from the target point, but off the line the answer saw
    r = resolver()
    _, constraints = r.inputs(tracks((1.0, 0.0), (2.0, 1.2), (3.0, 0.0)), robot_xy=(5.0, 0.0), t=3.0)
    assert constraints[0].people == [1, 3]


# ---- a floor from the robot's navigation map, for a re-solve whose picture no longer shows the line

def _facing_away(floor_map=None):
    """The two at the front of a line along -y, behind a robot that faces away from them."""
    from sgi.config import load_config
    from sgi.geometry import RobotPose
    from sgi.solver import solve
    from sgi.types import GoalResult, GroundedTarget, Point
    from tests.helpers import floor_observation
    th = load_config().thresholds
    up = math.pi / 2
    people = [person(1, 1.0, 8.0, up), person(2, 1.0, 7.3, up)]
    target = GroundedTarget(point=Point(x=500, y=300), pixel=(320.0, 150.0), world=(1.0, 9.0, 0.0))
    standard = GoalResult(status="ok", pose_world=(1.0, 8.5, up))
    view = floor_observation(RobotPose(1.0, 2.0, -up))
    return solve(view, target, people, [after(1, 2)], standard, th, floor_map=floor_map)


def _open_map(blocked_near=None, radius=0.3):
    import numpy as np
    from sgi.solver import FloorMap
    navigable = np.ones((400, 300), dtype=bool)            # x from -5 to 10, y from -5 to 15
    if blocked_near is not None:
        ys, xs = np.mgrid[0:400, 0:300]
        navigable &= np.hypot(-5.0 + 0.05 * xs - blocked_near[0], -5.0 + 0.05 * ys - blocked_near[1]) > radius
    return FloorMap(origin_xy=(-5.0, -5.0), resolution=0.05, navigable=navigable)


def test_with_a_floor_map_the_place_in_turn_is_found_out_of_view():
    solved = _facing_away(_open_map())
    assert solved.pose_world[:2] == pytest.approx((1.0, 6.3), abs=0.1)
    assert solved.note.keys.turn_m == pytest.approx(0.0, abs=0.06)


def test_without_one_the_out_of_view_place_cannot_be_chosen():
    assert _facing_away().note.keys.turn_m > 1.0


def test_the_map_keeps_the_robot_off_what_it_cannot_stand_on():
    solved = _facing_away(_open_map(blocked_near=(1.0, 6.3), radius=0.4))
    assert math.hypot(solved.pose_world[0] - 1.0, solved.pose_world[1] - 6.3) > 0.35


# ---- someone who has left is taken out of every constraint that names them

def test_who_left_the_line_leaves_its_other_constraints_too():
    r = resolver(constraints=[after(1, 2, 3), shared(1, 2, 3)])
    r.inputs(tracks((0.3, 0.0), (1.5, 0.0), (2.5, 0.0)), robot_xy=(5.0, 0.0), t=3.0)
    _, constraints = r.inputs(tracks((0.3, 2.0), (1.5, 0.0), (2.5, 0.0)), robot_xy=(5.0, 0.0), t=4.0)
    assert constraints[0].people == [2, 3] and constraints[1].people == [2, 3]


def test_someone_named_who_does_not_wait_and_walks_off_leaves_their_constraints():
    r = resolver(constraints=[after(1), shared(2, 3)])
    _, constraints = r.inputs(tracks((1.0, 0.0), (2.0, 2.0), (3.0, 0.0)), robot_xy=(5.0, 0.0), t=3.0)
    assert constraints[1].people == [3]
    _, constraints = r.inputs(tracks((1.0, 0.0), (2.0, 0.5), (3.0, 0.0)), robot_xy=(5.0, 0.0), t=4.0)
    assert constraints[1].people == [3]                    # for good


# ---- settled: re-solve only when the people near the waiting set have stopped moving (CTs)

def test_people_standing_still_are_settled():
    r = resolver()
    assert r.settled(tracks(*LINE), t=3.0)                 # no history yet: nothing seen moving
    assert r.settled(tracks(*LINE), t=3.5)
    assert r.settled(tracks(*LINE), t=4.0)


def test_someone_near_the_waiting_people_stepping_up_is_not_settled():
    r = resolver()
    r.settled(tracks(*LINE), t=3.0)
    assert not r.settled(tracks((0.8, 0.0), (2.0, 0.0), (3.0, 0.0)), t=3.5)   # 1 moved 0.2 m in 0.5 s
    assert not r.settled(tracks((0.8, 0.0), (2.0, 0.0), (3.0, 0.0)), t=4.0)   # 0.2 m within the last 1 s
    assert r.settled(tracks((0.8, 0.0), (2.0, 0.0), (3.0, 0.0)), t=4.6)       # still for over 1 s


def test_someone_moving_far_from_the_waiting_people_does_not_count():
    r = resolver()
    r.settled(tracks(*LINE, (9.0, 9.0)), t=3.0)
    assert r.settled(tracks(*LINE, (9.0, 8.0)), t=3.5)


# ---- CT2 (moving/CT2_FROZEN.md): re-admit, a shorter span, the vacated place, path memory

V2 = dict(readmit=True, line_span_m=0.5, vacate_after_s=5.0, path_memory=True)


def test_v2_re_admits_someone_who_left_and_joined_again_behind():
    r = resolver(**{"people": [person(1, 1.0, 0.0), person(2, 2.0, 0.0)], "constraints": [after(1, 2)],
                    "at_answer": tracks((1.0, 0.0), (2.0, 0.0))}, options=V2)
    r.inputs(tracks((0.3, 3.0), (1.0, 0.0)), robot_xy=(5.0, 0.0), t=10.0)          # 1 served and gone
    assert r.dropped == {1}
    for t in (30.0, 30.5):                                                           # 1 is back, behind 2
        r.record((3.0, 0.0), True, absorbed=[1])
    _, constraints = r.inputs(tracks((2.0, 0.0), (1.0, 0.0)), robot_xy=(5.0, 0.0), t=31.0)
    assert sorted(constraints[0].people) == [1, 2] and r.dropped == set()


def test_v1_still_never_re_admits():
    r = resolver(people=[person(1, 1.0, 0.0), person(2, 2.0, 0.0)], constraints=[after(1, 2)],
                 at_answer=tracks((1.0, 0.0), (2.0, 0.0)))
    r.inputs(tracks((0.3, 3.0), (1.0, 0.0)), robot_xy=(5.0, 0.0), t=10.0)
    for t in (30.0, 30.5):
        r.record((3.0, 0.0), True, absorbed=[1])
    _, constraints = r.inputs(tracks((2.0, 0.0), (1.0, 0.0)), robot_xy=(5.0, 0.0), t=31.0)
    assert constraints[0].people == [2]


def test_v2_gives_a_two_person_line_a_direction():
    two = dict(people=[person(1, 1.0, 0.0), person(2, 1.7, 0.0)], constraints=[after(1, 2)],
               at_answer=tracks((1.0, 0.0), (1.7, 0.0)))
    assert resolver(**two).line_direction is None
    assert resolver(**two, options=V2).line_direction == pytest.approx((1.0, 0.0))


def _joined_behind(options):
    r = resolver(people=[person(1, 1.0, 0.0), person(2, 1.7, 0.0)], constraints=[after(1, 2)],
                 at_answer=tracks((1.0, 0.0), (1.7, 0.0)), options=options)
    r.record((2.7, 0.0), True)
    for t in (5.0, 5.5, 6.0, 6.5):                                                  # still; the robot settles behind
        r.inputs(tracks((1.0, 0.0), (1.7, 0.0)), robot_xy=(2.7, 0.0), t=t)
    assert r.joined
    return r


def test_v2_takes_the_place_of_the_last_who_left_once_they_have_been_gone_5_s():
    r = _joined_behind(V2)
    r.inputs(tracks((0.0, 3.0), (1.0, 0.0)), robot_xy=(2.7, 0.0), t=20.0)          # 1 served; 2 stepped up, still
    for t in (20.5, 21.0, 21.5):
        r.inputs(tracks((0.0, 3.5), (1.0, 0.0)), robot_xy=(2.0, 0.0), t=t)
    r.inputs(tracks((0.0, 4.0), (1.0, 2.5)), robot_xy=(2.0, 0.0), t=30.0)          # 2 served too, walks off
    assert r.vacated_pose is None                                                   # not yet 5 s
    r.inputs(tracks((0.0, 4.0), (1.0, 3.5)), robot_xy=(2.0, 0.0), t=35.5)
    assert r.vacated_pose is not None and r.vacated_pose[:2] == pytest.approx((1.0, 0.0), abs=0.01)


def test_v2_does_not_vacate_before_the_robot_has_joined():
    r = resolver(people=[person(1, 1.0, 0.0)], constraints=[after(1)], at_answer=tracks((1.0, 0.0)), options=V2)
    r.inputs(tracks((1.0, 0.0)), robot_xy=(6.0, 0.0), t=3.0)
    r.inputs(tracks((0.0, 4.0)), robot_xy=(6.0, 0.0), t=10.0)
    r.inputs(tracks((0.0, 4.0)), robot_xy=(6.0, 0.0), t=20.0)
    assert r.vacated_pose is None


BENT = [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (3.0, 2.0)]                  # turns 90 deg at the third


def test_v2_path_memory_keeps_people_stepping_round_the_corner():
    people = [person(i + 1, *xy) for i, xy in enumerate(BENT)]
    r = resolver(people=people, constraints=[after(1, 2, 3, 4, 5)], at_answer=tracks(*BENT), options=V2)
    stepped = [(0.3, 3.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (3.0, 1.0)]          # 1 served; the rest step up
    _, constraints = r.inputs(tracks(*stepped), robot_xy=(6.0, 2.0), t=10.0)
    assert sorted(constraints[0].people) == [2, 3, 4, 5]
    straight = resolver(people=people, constraints=[after(1, 2, 3, 4, 5)], at_answer=tracks(*BENT))
    _, constraints = straight.inputs(tracks(*stepped), robot_xy=(6.0, 2.0), t=10.0)
    assert 4 not in constraints[0].people                   # the straight axis drops the one at the corner


def test_v2_path_memory_points_the_solver_along_the_back_of_the_line():
    people = [person(i + 1, *xy) for i, xy in enumerate(BENT)]
    r = resolver(people=people, constraints=[after(1, 2, 3, 4, 5)], at_answer=tracks(*BENT), options=V2)
    r.inputs(tracks(*BENT), robot_xy=(6.0, 2.0), t=3.0)
    assert r.solve_direction == pytest.approx((0.0, 1.0), abs=0.02)


# ---- co-motion (research arm CT3): named people must move with the line; nobody joins it without moving

CO = dict(V2, co_motion_s=4.0)
ASIDE = (2.9, 0.7)                     # 0.92 m from the second in line and off to its side: a bystander


def run(r, where, t0, t1, robot_xy=(6.0, 0.0), dt=0.5):
    """Feed the resolver the tracks `where(t)` every `dt` seconds from t0 to t1; returns the last constraints."""
    constraints, t = None, t0
    while t <= t1 + 1e-9:
        _, constraints = r.inputs(tracks(*where(t)), robot_xy=robot_xy, t=t)
        t += dt
    return constraints


def served_at_10(third):
    """1 is served at t = 10 and walks off; 2 steps up a metre over the next second; the third is `third(t)`."""
    def where(t):
        first = (1.0, 0.0) if t < 10.0 else (0.3, 3.0)
        second = (2.0, 0.0) if t < 10.0 else (max(1.0, 2.0 - (t - 10.0)), 0.0)
        return [first, second, third(t)]
    return where


def test_co_motion_drops_a_named_person_who_does_not_follow_the_person_ahead():
    people = [person(1, 1.0, 0.0), person(2, 2.0, 0.0), person(3, *ASIDE)]          # the model put 3 in the line
    r = resolver(people=people, constraints=[after(1, 2, 3)], at_answer=tracks((1.0, 0.0), (2.0, 0.0), ASIDE),
                 options=CO)
    constraints = run(r, served_at_10(lambda t: ASIDE), 0.0, 16.0)
    assert constraints[0].people == [2]
    assert 3 in r.dropped


def test_without_co_motion_v2_keeps_the_bystander_it_was_given():
    people = [person(1, 1.0, 0.0), person(2, 2.0, 0.0), person(3, *ASIDE)]
    r = resolver(people=people, constraints=[after(1, 2, 3)], at_answer=tracks((1.0, 0.0), (2.0, 0.0), ASIDE),
                 options=V2)
    constraints = run(r, served_at_10(lambda t: ASIDE), 0.0, 16.0)
    assert constraints[0].people == [2, 3]


def test_co_motion_keeps_a_named_person_who_steps_up_behind_the_person_ahead():
    third = lambda t: (3.0, 0.0) if t < 10.5 else (max(2.0, 3.0 - (t - 10.5)), 0.0)
    r = resolver(people=[person(1, 1.0, 0.0), person(2, 2.0, 0.0), person(3, 3.0, 0.0)], constraints=[after(1, 2, 3)],
                 at_answer=tracks(*LINE), options=CO)
    constraints = run(r, served_at_10(third), 0.0, 20.0)
    assert constraints[0].people == [2, 3]


def test_co_motion_ignores_which_way_people_face():
    third = lambda t: (3.0, 0.0) if t < 10.5 else (max(2.0, 3.0 - (t - 10.5)), 0.0)
    r = resolver(people=[person(1, 1.0, 0.0), person(2, 2.0, 0.0), person(3, 3.0, 0.0, yaw=math.pi / 2)],
                 constraints=[after(1, 2, 3)], at_answer=tracks(*LINE, yaw=math.pi / 2), options=CO)
    t, constraints = 0.0, None
    while t <= 20.0:
        _, constraints = r.inputs([Track(id=i, xy=p, yaw=math.pi / 2) for i, p in enumerate(served_at_10(third)(t))],
                                  robot_xy=(6.0, 0.0), t=t)
        t += 0.5
    assert constraints[0].people == [2, 3]


def test_co_motion_does_not_let_someone_who_never_moved_join():
    people = [person(1, 1.0, 0.0), person(2, 2.0, 0.0), person(3, *ASIDE)]          # 3 is numbered, not waiting
    still = lambda t: [(1.0, 0.0), (2.0, 0.0), ASIDE]
    for options, joins in ((V2, True), (CO, False)):
        r = resolver(people=people, constraints=[after(1, 2)], at_answer=tracks(*still(0.0)), options=options)
        run(r, still, 0.0, 12.0)
        for _ in range(2):
            r.record((3.0, 0.0), True, absorbed=[3])
        _, constraints = r.inputs(tracks(*still(12.5)), robot_xy=(6.0, 0.0), t=12.5)
        assert (3 in constraints[0].people) is joins


def test_co_motion_still_re_admits_someone_who_walked_back_to_the_end():
    def where(t):
        if t < 10.0:
            first = (1.0, 0.0)
        elif t < 20.0:
            first = (0.3, 3.0)
        else:                                          # walks back over 4 s and stands behind 2
            k = min(1.0, (t - 20.0) / 4.0)
            first = (0.3 + k * 1.7, 3.0 - k * 3.0)
        second = (2.0, 0.0) if t < 10.0 else (max(1.0, 2.0 - (t - 10.0)), 0.0)
        return [first, second]
    r = resolver(people=[person(1, 1.0, 0.0), person(2, 2.0, 0.0)], constraints=[after(1, 2)],
                 at_answer=tracks((1.0, 0.0), (2.0, 0.0)), options=CO)
    run(r, where, 0.0, 26.0)
    assert r.dropped == {1}
    for _ in range(2):
        r.record((3.0, 0.0), True, absorbed=[1])
    _, constraints = r.inputs(tracks(*where(26.5)), robot_xy=(6.0, 0.0), t=26.5)
    assert sorted(constraints[0].people) == [1, 2]


def test_co_motion_lets_an_unnamed_person_who_steps_up_with_the_line_join():
    third = lambda t: (3.0, 0.0) if t < 10.5 else (max(2.0, 3.0 - (t - 10.5)), 0.0)   # the model missed 3
    r = resolver(people=[person(1, 1.0, 0.0), person(2, 2.0, 0.0), person(3, 3.0, 0.0)], constraints=[after(1, 2)],
                 at_answer=tracks(*LINE), options=CO)
    run(r, served_at_10(third), 0.0, 14.0)
    for _ in range(2):
        r.record((3.0, 0.0), True, absorbed=[3])
    _, constraints = r.inputs(tracks(*served_at_10(third)(14.5)), robot_xy=(6.0, 0.0), t=14.5)
    assert sorted(constraints[0].people) == [2, 3]


def test_co_motion_exempts_from_the_solver_whoever_it_would_not_let_join():
    people = [person(1, 1.0, 0.0), person(2, 2.0, 0.0), person(3, *ASIDE)]
    still = lambda t: [(1.0, 0.0), (2.0, 0.0), ASIDE]
    r = resolver(people=people, constraints=[after(1, 2)], at_answer=tracks(*still(0.0)), options=CO)
    run(r, still, 0.0, 5.0)
    assert r.exempt() == {3}
    v2 = resolver(people=people, constraints=[after(1, 2)], at_answer=tracks(*still(0.0)), options=V2)
    run(v2, still, 0.0, 5.0)
    assert v2.exempt() == set()


def test_co_motion_without_the_follow_check_keeps_a_named_bystander():
    people = [person(1, 1.0, 0.0), person(2, 2.0, 0.0), person(3, *ASIDE)]
    r = resolver(people=people, constraints=[after(1, 2, 3)], at_answer=tracks((1.0, 0.0), (2.0, 0.0), ASIDE),
                 options=dict(CO, co_motion_follow=False))
    constraints = run(r, served_at_10(lambda t: ASIDE), 0.0, 16.0)
    assert constraints[0].people == [2, 3] and r.non_followers == set()


def test_the_track_a_number_stands_for():
    r = resolver(at_answer=tracks((3.0, 0.0), (1.0, 0.0), (2.0, 0.0)))
    assert [r.track_of(n) for n in (1, 2, 3, 99)] == [1, 2, 0, None]


def test_non_followers_are_kept_with_when_they_were_dropped():
    people = [person(1, 1.0, 0.0), person(2, 2.0, 0.0), person(3, *ASIDE)]
    r = resolver(people=people, constraints=[after(1, 2, 3)], at_answer=tracks((1.0, 0.0), (2.0, 0.0), ASIDE),
                 options=CO)
    run(r, served_at_10(lambda t: ASIDE), 0.0, 16.0)
    assert list(r.non_follower_t) == [3] and 14.0 <= r.non_follower_t[3] <= 16.0


def test_co_motion_rejoin_cancels_a_follow_check_made_before_someone_was_taken_back():
    def where(t):                                       # 1 steps up at 10; 2 walks off at 10-12, back behind at 13-14
        first = (1.0, 0.0) if t < 10.0 else (0.3, 0.0)
        if t < 10.0:
            second = (2.0, 0.0)
        elif t < 12.0:
            second = (2.0 + (t - 10.0) * 1.5, 1.0)
        else:
            second = (2.2, 0.0)
        return [first, second]

    def drive(options):
        r = resolver(people=[person(1, 1.0, 0.0), person(2, 2.0, 0.0)], constraints=[after(1, 2)],
                     at_answer=tracks((1.0, 0.0), (2.0, 0.0)), options=options)
        run(r, where, 0.0, 13.0)
        assert 2 in r.dropped                            # walked off: left the line
        for _ in range(2):
            r.record((3.0, 0.0), True, absorbed=[2])     # and is taken back, behind 1
        run(r, where, 13.5, 17.0)
        return r
    assert 2 in drive(CO).non_followers                  # CT3: the check made at the step-up still fires
    kept = drive(dict(CO, co_motion_rejoin=True))        # CT3b: it is dropped with the re-admission
    assert 2 not in kept.non_followers and 2 not in kept.dropped


# ---- ideas round 2: the best bolt-on of the same evidence (CT3-fix-gated, moving/ideas2_1b/CT4_FROZEN.md)

GROUP = dict(CO, group_follow=True)


def test_group_follow_checks_everyone_behind_an_advance_not_only_the_next():
    # 2 steps up a metre; 3 follows 0.3 m (a follow, too short to be an advance); 4, named, stays put further back
    def where(t):
        first = (1.0, 0.0) if t < 10.0 else (0.3, 3.0)
        second = (2.0, 0.0) if t < 10.0 else (max(1.0, 2.0 - (t - 10.0)), 0.0)
        third = (3.0, 0.0) if t < 10.0 else (max(2.7, 3.0 - 0.3 * (t - 10.0)), 0.0)
        return [first, second, third, (4.0, 0.0)]
    got = {}
    for name, options in (("ct3", CO), ("group", GROUP)):
        people = [person(i + 1, *xy) for i, xy in enumerate(LINE + [(4.0, 0.0)])]
        r = resolver(people=people, constraints=[after(1, 2, 3, 4)], at_answer=tracks(*LINE, (4.0, 0.0)), options=options)
        run(r, where, 0.0, 18.0)
        got[name] = r
    assert 4 not in got["ct3"].dropped and 3 not in got["ct3"].dropped
    assert 4 in got["group"].dropped and 4 in got["group"].non_followers and 3 not in got["group"].dropped


ANCHOR = dict(GROUP, anchor_rederive=True)


def test_anchor_rederive_finds_a_new_line_once_everyone_named_has_gone_but_not_someone_who_never_moved():
    # 1 and 2 are named; both leave at 5 s. Two newcomers walk in from far off and stand at (1, 0) and (1.8, 0) by
    # 12 s; someone at (2.6, 0) stood still all along and was not named.
    def where(t):
        gone = t >= 5.0
        a = (1.0, 0.0) if not gone else (1.0, 6.0)
        b = (2.0, 0.0) if not gone else (2.0, 6.0)
        k = min(max(t - 6.0, 0.0) / 6.0, 1.0)
        c = (1.0 + 5.0 * (1 - k), -4.0 * (1 - k))
        d = (1.8 + 5.0 * (1 - k), -4.0 * (1 - k))
        return [a, b, (2.6, 0.0), c, d]
    got = {}
    for name, options in (("group", GROUP), ("anchor", ANCHOR)):
        people = [person(1, 1.0, 0.0), person(2, 2.0, 0.0), person(3, 2.6, 0.0)]
        r = resolver(people=people, constraints=[after(1, 2)],
                     at_answer=tracks((1.0, 0.0), (2.0, 0.0), (2.6, 0.0), (6.0, -4.0), (6.8, -4.0)), options=options)
        got[name] = run(r, where, 0.0, 16.0)[0].people
    newcomers = {4, 5}                                       # the numbers the resolver gave the two new tracks
    assert got["group"] == []
    assert set(got["anchor"]) == newcomers
