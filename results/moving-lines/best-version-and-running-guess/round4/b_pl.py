"""Round 4, step B: PL (ROUND4.md, cc30d5e0) against the threshold rule and CT5's cost rule, offline on the 24 pilot
layouts (robot idle) with the proxy goal, and on the round 3 design cells × 4 conditions (classes only). Zero calls.

    .venv/bin/python round4/b_pl.py [pilot|design|both]   (from best-version-and-running-guess/)
"""
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round3"))
sys.path.insert(0, str(HERE))
from r3lib import CONDITIONS, CT4, CT5, UNSET, classify, decide, design_cells, ticks_of, turn_last  # noqa: E402
from g_pilot_predict import WIN, pilot_cells  # noqa: E402
from i_followups import CLEAR_M, BAND_M, line_of  # noqa: E402
from replay_lib import dump_tracks, dump_truth, robot_at  # noqa: E402
from sgi.resolve import Resolver  # noqa: E402

C_MOVE = 0.125
STANDOFF_M = 1.0


def p_last(est, present):
    """P(last = j) over present people (by dist_now), and P(nobody present waits) under key None."""
    post = est.b * getattr(est, "static", 1.0)          # as CT4.marginals and CT5.posterior
    post = post / post.sum()
    ids = [i for i in est.ids if i in present and i in est.dist_now]
    if not ids:
        return {None: 1.0}
    cols = np.array([est.k[i] for i in ids])
    d = np.array([est.dist_now[i] for i in ids])
    W = est.bits[:, cols]
    masked = np.where(W, d[None, :], -np.inf)
    arg = masked.argmax(axis=1)
    none = ~W.any(axis=1)
    out = {None: float(post[none].sum())}
    for k, i in enumerate(ids):
        out[i] = float(post[(~none) & (arg == k)].sum())
    return out


def decide_pl(est, present, current=UNSET, c_move=C_MOVE):
    s = p_last(est, present)
    best = max(s, key=s.get)
    if current is not UNSET and current in s:
        dpos = {None: 0.0, **{i: est.dist_now[i] for i in s if i is not None}}
        if s[best] - s[current] <= c_move * abs(dpos[best] - dpos[current]):
            return current, s
    return best, s


class Path_:
    """The answer's remembered line (CT3F's, CT4P's): a Resolver with path_memory, for geometry only."""

    def __init__(self, cell):
        at = dump_tracks(cell.dump, cell.asked_t)
        self.r = Resolver(cell.people, cell.constraints, cell.target, at, path_memory=True, line_span_m=0.5)

    def tangent(self, xy):
        if self.r._path is None or xy is None:
            return None
        return self.r._on_path(xy)[2]


def run(cell, stress, est_cls, params, decision, direction):
    est = est_cls(cell, params)
    path = Path_(cell) if direction == "path" else None
    rows, cur = [], UNSET
    for t, tracks, _ in ticks_of(cell, stress, 0.5):
        est.step(t, tracks, 0.5)
        if t + 1e-9 < cell.delivered_t:
            continue
        m = est.marginals()
        present = {tr.id for tr in tracks}
        pos = {tr.id: tr.xy for tr in tracks}
        roster = {i for i, v in m.items() if v >= 0.5 and i in present}
        robot = robot_at(cell.episode, t)
        if not isinstance(est, CT5):                     # CT4 keeps no dist_now: distance to the target point now
            est.dist_now = {i: float(np.hypot(*(np.asarray(xy) - est.tgt))) for i, xy in pos.items()}
        if decision == "threshold":
            come = roster
            back = max(come, key=lambda i: est.dist_now.get(i, -1)) if come else None
        else:
            cur, _ = (decide if decision == "cost" else decide_pl)(est, present, cur)
            come = set() if cur is None else {i for i in roster if est.dist_now.get(i, np.inf) < est.dist_now[cur]} | {cur}
            back = cur
        d = est.dir
        if path is not None and back is not None and back in pos:
            tg = path.tangent(pos[back])
            if tg is not None:
                d = np.array(tg, dtype=np.float64)
        saved = est.dir
        est.dir = d
        last = turn_last(come, tracks, est, robot)
        est.dir = saved
        q, order = dump_truth(cell.dump, t)
        rows.append(dict(t=t, last=last, roster=roster, q=q, order=order, dir=d, pos=pos))
    return rows


def proxy_ok(row, line):
    """The place STANDOFF_M behind the chosen last along the direction, by the sim's rules (no heading)."""
    q, pos, last = row["q"], row["pos"], row["last"]
    members = [i for i in q if i in pos]
    if not members or last is None or last not in pos:
        return False
    la = max(members, key=lambda i: line.project(*pos[i])[0])
    d = row["dir"] / max(float(np.hypot(*row["dir"])), 1e-9)
    xy = (pos[last][0] + STANDOFF_M * d[0], pos[last][1] + STANDOFF_M * d[1])
    al, side = line.project(*xy)
    if side > 0.6 or al < line.project(*pos[la])[0]:
        return False
    if not BAND_M[0] <= math.dist(xy, pos[la]) <= BAND_M[1]:
        return False
    return min(math.dist(xy, p) for p in pos.values()) >= CLEAR_M


VARIANTS = {
    "CT4 (threshold, chord)": (CT4, None, "threshold", "chord"),
    "CT4P (threshold, path)": (CT4, None, "threshold", "path"),
    "W0 (threshold, chord)": (CT5, WIN, "threshold", "chord"),
    "W0 threshold, path": (CT5, WIN, "threshold", "path"),
    "CT5 (cost, chord)": (CT5, WIN, "cost", "chord"),
    "CT5 cost, path": (CT5, WIN, "cost", "path"),
    "PL-chord (W0 belief)": (CT5, WIN, "pl", "chord"),
    "PL-path (W0 belief)": (CT5, WIN, "pl", "path"),
    "PL-path, CT4 belief (check)": (CT4, None, "pl", "path"),
}


def summarise(rows_by_cell, lines=None):
    k, sw, mins, ok, n = Counter(), 0, 0.0, 0, 0
    per_cell_sw = []
    for key, rows in rows_by_cell.items():
        k.update(classify(r) for r in rows)
        s = sum(1 for a, b in zip(rows, rows[1:]) if a["last"] != b["last"])
        m = max((rows[-1]["t"] - rows[0]["t"]) / 60, 1e-9) if rows else 1e-9
        sw += s
        mins += m
        per_cell_sw.append(s / m)
        if lines is not None:
            for r in rows:
                if r["q"]:
                    n += 1
                    ok += proxy_ok(r, lines[key])
    scored = sum(v for kk, v in k.items() if kk is not None)
    out = {x: k[x] / max(scored, 1) for x in ("right", "cut", "far", "empty")}
    out["sw"] = sw / max(mins, 1e-9)
    out["proxy"] = ok / max(n, 1) if lines is not None else None
    return out


def pilot():
    cs, missing = pilot_cells("runs_pilot3/dsv41")
    lines = {lay["name"]: line_of(lay["name"]) for lay, _ in cs}
    print(f"## Pilot layouts, robot idle: {len(cs)} layouts ({len(missing)} left out)\n")
    print("| Variant | right | cut | far | empty | last switches / min | goal in line (proxy) |")
    print("|---|---|---|---|---|---|---|")
    res = {}
    for name, (cls, params, dec, dr) in VARIANTS.items():
        rows = {lay["name"]: run(c, None, cls, params, dec, dr) for lay, c in cs}
        s = summarise(rows, lines)
        res[name] = s
        print(f"| {name} | {s['right']:.3f} | {s['cut']:.3f} | {s['far']:.3f} | {s['empty']:.3f} | {s['sw']:.1f} | "
              f"{s['proxy']:.3f} |")
    return res


def design():
    cs = design_cells()
    print(f"\n## Design cells × conditions: {len(cs)} cells\n")
    print("| Variant | condition | right | cut | far | empty | last switches / min |")
    print("|---|---|---|---|---|---|---|")
    for name in ("W0 (threshold, chord)", "CT5 (cost, chord)", "PL-chord (W0 belief)", "CT4 (threshold, chord)"):
        cls, params, dec, dr = VARIANTS[name]
        for cond, stress in CONDITIONS.items():
            rows = {c.key: run(c, stress, cls, params, dec, dr) for c in cs}
            s = summarise(rows)
            print(f"| {name} | {cond} | {s['right']:.3f} | {s['cut']:.3f} | {s['far']:.3f} | {s['empty']:.3f} | "
                  f"{s['sw']:.1f} |")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    print("# Round 4, step B: PL offline (ROUND4.md cc30d5e0)\n")
    if which in ("pilot", "both"):
        pilot()
    if which in ("design", "both"):
        design()


if __name__ == "__main__":
    main()
