"""Verifier's own replacement of sgi.solver._turn (I06 repair probe, FROZEN.md section 2). Written independently of
the compute agent's code/rules.py (not imported). Reads only body positions and yaws, the waiting list, the apart set
and the target's number.

    VR0    the package's _turn (installed as the original object), with a shadow call of my copy (rule R0) that must
           agree on last, joined, last_from_target and place.
    VRF90  a candidate that passes the package's test joins only if f.u > 0.0 - 1e-9 (unknown yaw: joins).
    VRF60  the same with 0.5.
    VRN    no candidate joins.

LOG collects, per call, every candidate that passed the package's own test, with the then-last, its gap, its facing
cosine towards the then-last, and whether it was admitted.
"""
from __future__ import annotations

import contextlib
import math

import numpy as np

import sgi.solver as S

PACKAGE_TURN = S._turn
THRESH = {"R0": None, "RF90": 0.0, "RF60": 0.5, "RN": None}
LOG: list = []
STATS = {"shadow_calls": 0, "shadow_mismatch": 0}


def cos_to_last(body, last_body):
    if body.yaw is None:
        return None
    d = last_body.xy - body.xy
    n = math.hypot(float(d[0]), float(d[1]))
    if n < 1e-12:
        return None
    return (math.cos(body.yaw) * float(d[0]) + math.sin(body.yaw) * float(d[1])) / n


def my_turn(waiting, apart, bodies, target_number, target_xy, robot_xy, th, rule, log=None):
    lim = THRESH[rule]
    in_line = list(waiting)
    joined = []
    last, before = S._line_end(in_line, bodies, target_xy, robot_xy, th)
    cos_view = math.cos(math.radians(th.view_half_angle_deg))
    calls = []
    while True:
        place, along = S._behind(bodies[last], None if before is None else bodies[before], target_xy, robot_xy, th)
        ok = []
        seen = []
        for num, b in bodies.items():
            if num in in_line or num == target_number or num in apart:
                continue
            back = b.xy - bodies[last].xy
            gap = float(np.hypot(*(b.xy - place)))
            behind = float(back @ along) >= cos_view * float(np.hypot(*back)) - S._EPS
            if not (gap < th.standoff_m - S._EPS and behind):
                continue
            fu = cos_to_last(b, bodies[last])
            if rule == "RN":
                admit = False
            elif lim is None or fu is None:
                admit = True
            else:
                admit = fu > lim - 1e-9
            seen.append((int(num), int(last), round(gap, 6), None if fu is None else round(fu, 6), admit))
            if admit:
                ok.append((round(gap, S._MM), num))
        calls.append(seen)
        if not ok:
            if log is not None:
                log.append({"rule": rule, "iters": calls, "joined": [int(j) for j in joined]})
            return S._Turn(last=last, last_from_target=float(np.hypot(*(bodies[last].xy - target_xy))),
                           place=place, joined=joined)
        nxt = min(ok)[1]
        before, last = last, nxt
        in_line.append(nxt)
        joined.append(nxt)


def _r0_with_shadow(waiting, apart, bodies, target_number, target_xy, robot_xy, th):
    got = PACKAGE_TURN(waiting, apart, bodies, target_number, target_xy, robot_xy, th)
    mine = my_turn(waiting, apart, bodies, target_number, target_xy, robot_xy, th, "R0", LOG)
    STATS["shadow_calls"] += 1
    same = (got.last == mine.last and list(got.joined) == list(mine.joined)
            and got.last_from_target == mine.last_from_target
            and np.array_equal(np.asarray(got.place), np.asarray(mine.place)))
    if not same:
        STATS["shadow_mismatch"] += 1
        raise AssertionError(f"shadow mismatch: {got} vs {mine}")
    return got


@contextlib.contextmanager
def rule_on(rule):
    assert S._turn is PACKAGE_TURN
    if rule == "R0":
        fn = _r0_with_shadow
    else:
        def fn(waiting, apart, bodies, target_number, target_xy, robot_xy, th):
            return my_turn(waiting, apart, bodies, target_number, target_xy, robot_xy, th, rule, LOG)
    S._turn = fn
    try:
        yield
    finally:
        S._turn = PACKAGE_TURN


def take_log():
    out = list(LOG)
    LOG.clear()
    return out
