"""The role estimator (experiment arm CT4)."""
import math

import numpy as np

from sgi.resolve import Track
from sgi.roles import RoleBelief, decide
from sgi.types import Constraint, DetectedPerson

TARGET = (0.0, 0.0)                                # the counter; the line runs back along +x


def person(number, x, y):
    return DetectedPerson(number=number, xy=(x, y), yaw=math.pi, pixel=(0.0, 0.0))


def tracks(*xy):
    return [Track(id=i, xy=p, yaw=math.pi) for i, p in enumerate(xy)]


def run(belief, where, t0, t1, dt=0.5):
    t = t0
    while t <= t1 + 1e-9:
        belief.step(t, tracks(*where(t)), dt)
        t += dt
    return belief.marginals()


def test_the_answer_is_the_prior_at_the_photo():
    at = [(1.0, 0.0), (2.0, 0.0), (5.0, 3.0)]
    b = RoleBelief([person(1, *at[0]), person(2, *at[1]), person(3, *at[2])],
                   [Constraint(because="they wait", kind="come_after", people=[1, 2])], TARGET, tracks(*at), (6.0, 0.0))
    m = run(b, lambda t: at, 0.0, 0.0)
    assert m[0] >= 0.5 and m[1] >= 0.5 and m[2] < 0.5
    assert b.named == {0, 1}
    assert b.dir[0] > 0.99                         # front to back, along +x


def test_a_named_person_who_does_not_follow_a_step_up_stops_waiting():
    # 0 is served at 10 s and walks off; 1 steps up a metre; 2 follows a metre; 3 (named) stays put behind
    def where(t):
        k = min(max(t - 10.0, 0.0), 1.0)
        first = (1.0, 0.0) if t < 10.0 else (1.0 + 2.0 * k, 3.0 * k)
        return [first, (2.0 - k, 0.0), (3.0 - k, 0.0), (4.0, 0.0)]
    at = where(0.0)
    b = RoleBelief([person(i + 1, *xy) for i, xy in enumerate(at)],
                   [Constraint(because="they wait", kind="come_after", people=[1, 2, 3, 4])], TARGET, tracks(*at),
                   (6.0, 2.0))
    before = run(b, where, 0.0, 9.5)
    assert before[3] >= 0.5
    after = run(b, where, 10.0, 22.0)
    assert after[3] < 0.5 and after[2] >= 0.5 and after[1] >= 0.5


def test_a_window_on_the_static_reading_damps_a_brief_odd_stance():
    # three in line; for one tick the last stands well off the line, then is back
    def where(t):
        return [(1.0, 0.0), (2.0, 0.0), (3.0, 0.9) if abs(t - 10.0) < 1e-9 else (3.0, 0.0)]
    at = where(0.0)
    got = {}
    for name, params in (("instant", None), ("window", {"STATIC_WIN_S": 5.0})):
        b = RoleBelief([person(i + 1, *xy) for i, xy in enumerate(at)],
                       [Constraint(because="they wait", kind="come_after", people=[1, 2, 3])], TARGET, tracks(*at),
                       (6.0, 0.0), params)
        got[name] = run(b, where, 0.0, 10.0)
    assert got["window"][2] > got["instant"][2] + 0.05


def two_people(p_second):
    """A belief over two people on the line, 1 m and 2 m from the target: the first waits for certain, the second
    with probability `p_second`."""
    at = [(1.0, 0.0), (2.0, 0.0)]
    b = RoleBelief([person(1, *at[0]), person(2, *at[1])], [], TARGET, tracks(*at), (6.0, 0.0))
    b.step(0.0, tracks(*at), 0.5)
    b.b = np.array([0.0, 1.0 - p_second, 0.0, p_second])         # sets {}, {0}, {1}, {0, 1}
    b.static = np.ones(4)
    return b


def test_decide_comes_after_a_possible_last_unless_they_wait_with_less_than_a_fifth():
    assert decide(two_people(0.3), {0, 1})[0] == 1                # 0.25 x 0.7 behind beats 1 x 0.3 ahead
    assert decide(two_people(0.15), {0, 1})[0] == 0               # 1 x 0.15 ahead beats 0.25 x 0.85 behind
    choice, costs = decide(two_people(0.3), {0, 1})
    assert set(costs) == {None, 0, 1} and abs(costs[0] - 0.3) < 1e-9 and abs(costs[1] - 0.175) < 1e-9


def test_decide_keeps_its_choice_unless_switching_saves_more_than_the_move():
    assert decide(two_people(0.25), {0, 1}, current=0)[0] == 0    # saves 0.0625, the 1 m move costs 0.125
    assert decide(two_people(0.6), {0, 1}, current=0)[0] == 1     # saves 0.5


def test_decide_chooses_afresh_when_its_person_is_gone_and_goes_first_when_nobody_waits():
    assert decide(two_people(0.3), {0}, current=1)[0] == 0
    b = two_people(0.0)
    b.b = np.array([1.0, 0.0, 0.0, 0.0])
    assert decide(b, {0, 1})[0] is None
