"""Post hoc PH-C (FOLLOWUPS.md): is the goal itself a strictly in-line place, and given that, is the robot in line?
Zero calls; read-only on the runs.

    .venv/bin/python round3/j_decompose.py runs_pilot3/dsv41
    (from best-version-and-running-guess/; RUN_ROOT relative to results/data/moving/)
"""
import math
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from i_followups import ARMS, BAND_M, CLEAR_M, HEADING_DEG, first_service_t, in_line, line_of, load, pos  # noqa: E402


def up_at(line, x, y):
    """Heading up the line at the segment nearest (x, y), as the sim's Line.project returns it."""
    best, n = None, len(line.length)
    for k in range(n):
        d = np.array([x, y]) - line.p[k]
        s = float(d @ line.unit[k])
        lo = -math.inf if k == 0 else 0.0
        hi = math.inf if k == n - 1 else float(line.length[k])
        s = min(max(s, lo), hi)
        side = float(np.hypot(*(np.array([x, y]) - (line.p[k] + s * line.unit[k]))))
        if best is None or side < best[0] - 1e-9:
            best = (side, math.atan2(-line.unit[k][1], -line.unit[k][0]))
    return best[1]


def off(a, b):
    return abs(math.degrees((a - b + math.pi) % (2 * math.pi) - math.pi))


def goal_check(r, line):
    """First failing sim condition for the re-solve's pose as the robot's place, or 'ok'."""
    tr, p = r["truth"], pos(r)
    gx, gy, gyaw = r["pose_world"]
    la = tr.get("last_along")
    if la is None or la not in p:
        return "nobody ahead"
    g_al, g_side = line.project(gx, gy)
    l_al, _ = line.project(*p[la])
    if g_side > 0.6:
        return "side"
    if g_al - l_al < 0:
        return "ahead of the last"
    gap = math.dist((gx, gy), p[la])
    if not BAND_M[0] <= gap <= BAND_M[1]:
        return "gap far" if gap > BAND_M[1] else "gap near"
    if off(gyaw, up_at(line, gx, gy)) > HEADING_DEG:
        return "heading"
    if min(math.dist((gx, gy), q) for q in p.values()) < CLEAR_M:
        return "nearest"
    return "ok"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "runs_pilot3/dsv41"
    sel, eps = load(root)
    print(f"# PH-C (post hoc): the goal or the getting there, {root}\n")
    print("Samples from the first service on, with someone ahead (sim) and a re-solve with a pose.\n")
    print("| Arm | samples | strict in line | goal in line | robot in line given goal in line | robot in line given "
          "goal not in line |")
    print("|---|---|---|---|---|---|")
    fails = {}
    for arm in ARMS:
        k, f = Counter(), Counter()
        for lay in sel:
            e = eps.get((lay["name"], arm))
            t0 = first_service_t(e) if e else None
            if t0 is None:
                continue
            line = line_of(lay["name"])
            rs = [r for r in e["resolves"] if r.get("status") == "ok" and r.get("pose_world")]
            for s in (x for x in e["samples"] if x["t"] >= t0 - 1e-9 and x["ahead"] > 0):
                r = max((r for r in rs if r["sent_t"] <= s["t"] + 1e-9), key=lambda r: r["sent_t"], default=None)
                if r is None:
                    continue
                g = goal_check(r, line)
                if g == "nobody ahead":
                    continue
                ok, ins = g == "ok", in_line(s, "strict")
                k["n"] += 1
                k["in"] += ins
                k["g"] += ok
                k["in_g"] += ins and ok
                k["in_ng"] += ins and not ok
                f[g] += 1
        n, ng = max(k["n"], 1), max(k["g"], 1)
        print(f"| {arm} | {k['n']} | {k['in'] / n:.3f} | {k['g'] / n:.3f} | {k['in_g'] / ng:.3f} | "
              f"{k['in_ng'] / max(k['n'] - k['g'], 1):.3f} |")
        fails[arm] = f
    print("\nWhy the goal is not in line (first failing condition, share of samples)\n")
    for arm in ARMS:
        f = fails[arm]
        n = max(sum(f.values()), 1)
        print(f"- {arm}: " + ", ".join(f"{a} {b / n:.3f}" for a, b in f.most_common() if a != "ok"))


if __name__ == "__main__":
    main()
