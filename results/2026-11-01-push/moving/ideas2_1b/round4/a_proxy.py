"""Round 4, step A: can a goal be judged offline without the solver? Zero calls; read-only on the pilot runs.

Proxy place = the chosen last's tracked position + standoff_m (1.0) along the direction the solver was given
(sgi.solver._behind with a line). It is judged by the sim's rules against the true last by distance along the line
(j_decompose.goal_check without heading, since the proxy has no yaw). This is compared with goal_check on the
solver's logged pose_world, re-solve by re-solve, for the arms that log the direction they used (CT4, W0, CT5).
"""
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "round3"))
from i_followups import BAND_M, CLEAR_M, first_service_t, line_of, load, pos  # noqa: E402
from j_decompose import goal_check  # noqa: E402

STANDOFF_M = 1.0   # sgi.config.Thresholds.standoff_m


def place_check(xy, r, line):
    tr, p = r["truth"], pos(r)
    la = tr.get("last_along")
    if la is None or la not in p:
        return "nobody ahead"
    g_al, g_side = line.project(*xy)
    if g_side > 0.6:
        return "side"
    if g_al - line.project(*p[la])[0] < 0:
        return "ahead of the last"
    gap = math.dist(xy, p[la])
    if not BAND_M[0] <= gap <= BAND_M[1]:
        return "gap far" if gap > BAND_M[1] else "gap near"
    if min(math.dist(xy, q) for q in p.values()) < CLEAR_M:
        return "nearest"
    return "ok"


def proxy_place(r):
    p, last, d = pos(r), r.get("last"), r.get("solve_direction") or r.get("line_direction")
    if last is None or last not in p or not d:
        return None
    n = math.hypot(*d)
    return (p[last][0] + STANDOFF_M * d[0] / n, p[last][1] + STANDOFF_M * d[1] / n)


def main():
    sel, eps = load("runs_pilot3/dsv41")
    for arm in ("CT4", "W0", "CT5"):
        k = Counter()
        for lay in sel:
            e = eps.get((lay["name"], arm))
            t0 = first_service_t(e) if e else None
            if t0 is None:
                continue
            line = line_of(lay["name"])
            for r in e["resolves"]:
                if r.get("status") != "ok" or not r.get("pose_world") or r["sent_t"] < t0 or not r["truth"]["ahead"]:
                    continue
                xy = proxy_place(r)
                if xy is None:
                    continue
                real = goal_check(r, line)
                if real == "nobody ahead":
                    continue
                real_ok = real in ("ok", "heading")          # the proxy has no yaw
                prox_ok = place_check(xy, r, line) == "ok"
                k[(real_ok, prox_ok)] += 1
        n = sum(k.values())
        agree = (k[(True, True)] + k[(False, False)]) / max(n, 1)
        print(f"{arm}: n {n}, agreement {agree:.3f}; solver ok & proxy ok {k[(True, True)]}, solver ok & proxy not "
              f"{k[(True, False)]}, solver not & proxy ok {k[(False, True)]}, both not {k[(False, False)]}; "
              f"share ok: solver {(k[(True, True)] + k[(True, False)]) / max(n, 1):.3f}, proxy "
              f"{(k[(True, True)] + k[(False, True)]) / max(n, 1):.3f}")


if __name__ == "__main__":
    main()
