"""I06 repair probe: the joining rules (FROZEN.md section 2). Research code, installed at run time in this process
only; the package is never edited.

turn_copy is a line-for-line copy of src/sgi/solver.py `_turn` (sha256 10ece70f...c5d800, lines 514-541), using the
solver module's own _line_end, _behind, _Turn, _EPS and _MM, with one added condition on the candidate test: the
facing gate. It reads only the positions and yaws of detected bodies, the waiting list, the `apart` set and the
target's number.

  R0    no gate (the package's test).            RF90  f . u > 0.0 - 1e-9.
  RF60  f . u > 0.5 - 1e-9.                      RN    nobody joins.

`trace`, where a list is given, receives one entry per call: every candidate that passed the package's own test
(space covers the place in turn, behind the last), its gap, its facing cosine towards the current last (None for
unknown yaw or no direction) and whether the gate let it in. It is a record only and changes nothing.
"""
from __future__ import annotations

import contextlib
import functools
import math

import numpy as np

from sgi import solver as S
from sgi.solver import _EPS, _MM, _Turn, _behind, _line_end

ORIG = S._turn                                     # the package's function object, captured before any patch
RULES = ("R0", "RF90", "RF60", "RN")
COS_MIN = {"R0": None, "RF90": 0.0, "RF60": 0.5, "RN": None}


class CopyCheckError(AssertionError):
    """The research copy with no gate disagrees with the package's _turn: the work stops."""


def facing_cos(body, last_body):
    """f . u for a body towards the current last member (u the unit vector from the body to the last, f the body's
    facing unit vector); None where the yaw is unknown or the two stand on one point. A record, not the gate."""
    if body.yaw is None:
        return None
    to_last = last_body.xy - body.xy
    dist = float(np.hypot(*to_last))
    if dist < 1e-12:
        return None
    f = np.array([math.cos(body.yaw), math.sin(body.yaw)])
    return float(f @ to_last) / dist


def gate(body, last_body, cos_min):
    if cos_min is None or body.yaw is None:      # no gate (R0's test), or unknown yaw: joins as in R0
        return True
    to_last = last_body.xy - body.xy              # from the person to the current last member
    dist = float(np.hypot(*to_last))
    if dist < 1e-12:                              # (choice) no direction defined: as R0
        return True
    f = np.array([math.cos(body.yaw), math.sin(body.yaw)])
    return float(f @ to_last) / dist > cos_min - 1e-9


def turn_copy(waiting, apart, bodies, target_number, target_xy, robot_xy, th, *, rule, trace=None):
    """src/sgi/solver.py _turn, copied line for line, with the facing gate added to the candidate test."""
    assert rule in RULES, rule
    entry = {"rule": rule, "iterations": []} if trace is not None else None
    waiting, joined = list(waiting), []
    last, before = _line_end(waiting, bodies, target_xy, robot_xy, th)
    while True:
        place, along = _behind(bodies[last], None if before is None else bodies[before], target_xy, robot_xy, th)
        taken = []
        seen = []
        for number, body in bodies.items():
            if number in waiting or number == target_number or number in apart:
                continue
            gap, back = float(np.hypot(*(body.xy - place))), body.xy - bodies[last].xy
            behind = float(back @ along) >= math.cos(math.radians(th.view_half_angle_deg)) * float(np.hypot(*back)) - _EPS
            if gap < th.standoff_m - _EPS and behind and rule != "RN" and gate(body, bodies[last], COS_MIN[rule]):
                taken.append((round(gap, _MM), number))
            if entry is not None and gap < th.standoff_m - _EPS and behind:
                fu = facing_cos(body, bodies[last])
                seen.append({"number": int(number), "gap": round(gap, 6), "last": int(last),
                             "yaw_known": body.yaw is not None, "fu": None if fu is None else round(fu, 6),
                             "admitted": any(n == number for _, n in taken)})
        if entry is not None:
            entry["iterations"].append(seen)
        if not taken:
            from_target = float(np.hypot(*(bodies[last].xy - target_xy)))
            if entry is not None:
                entry["joined"] = [int(n) for n in joined]
                trace.append(entry)
            return _Turn(last=last, last_from_target=from_target, place=place, joined=joined)
        last, before = min(taken)[1], last
        waiting.append(last)
        joined.append(last)


def checked_r0(trace=None):
    """The function installed in an R0 pass: it returns the package's own result, and checks on every call that
    turn_copy(rule="R0") agrees with it (last, joined, last_from_target equal; place equal element for element)."""
    def run(waiting, apart, bodies, target_number, target_xy, robot_xy, th):
        got = ORIG(waiting, apart, bodies, target_number, target_xy, robot_xy, th)
        copy = turn_copy(waiting, apart, bodies, target_number, target_xy, robot_xy, th, rule="R0", trace=trace)
        same = (got.last == copy.last and list(got.joined) == list(copy.joined)
                and got.last_from_target == copy.last_from_target
                and len(got.place) == len(copy.place) and all(a == b for a, b in zip(got.place, copy.place)))
        if not same:
            raise CopyCheckError(f"copy check failed: package {got} copy {copy}")
        CHECKS["calls"] += 1
        return got
    return run


CHECKS = {"calls": 0}


@contextlib.contextmanager
def installed(rule, *, check=True, trace=None):
    """Sets sgi.solver._turn for `rule` and restores the package's function on exit. R0 with check=False installs
    the original function object; R0 with check=True installs checked_r0 (the copy check); the repair rules install
    functools.partial(turn_copy, rule=..., trace=...)."""
    assert S._turn is ORIG, "a rule is already installed"
    if rule == "R0":
        fn = checked_r0(trace) if check else ORIG
    else:
        fn = functools.partial(turn_copy, rule=rule, trace=trace)
    S._turn = fn
    try:
        yield fn
    finally:
        S._turn = ORIG
