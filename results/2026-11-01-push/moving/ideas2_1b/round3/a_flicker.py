"""Round 3, step A (design data only, zero calls): does the offline replay show the sweep's CT4 failure?

For each design cell (bystander and original scenes, seeds 10-44, DeepSeek and Sonnet answers), frozen CT4-min from
the photo, decisions from delivery, every 0.5 s. Reported per condition (clean; S2m = lag matched to the sim's,
U(0,4) s plus 20% wait for the next service):
- calibration: marginal bin -> share in the queue (dump truth);
- roster flicker: share of consecutive ticks whose roster changed, with the static snapshot (as frozen) and without
  it (the dynamic belief alone);
- right last, as replay_lib.score counts it.

    .venv/bin/python round3/a_flicker.py      (from ideas2_1b/)
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import stress as ST  # noqa: E402
from ct4 import CT4  # noqa: E402
from replay_lib import (BYSTANDER_SCENES, ORIGINAL_SCENES, TH, Track, cells, dump_tracks,  # noqa: E402
                        dump_truth, robot_at)
from sgi import solver as S  # noqa: E402


def s2m(cell, ticks):
    """S2 with the sim's lag range: U(0, 4) s, 20% wait for the next service (stress.s2 otherwise)."""
    rng = ST._rng(cell, "S2m")
    sv = ST._services(cell)
    lag, joined_at = {}, {}

    def epoch(t):
        return sum(1 for s in sv if s <= t)

    def lag_of(pid, t):
        k = (pid, epoch(t))
        if k not in lag:
            if rng.random() < 0.2:
                nxt = next((s for s in sv if s > t), t + 15.0)
                lag[k] = (nxt - t) + rng.uniform(0, 2)
            else:
                lag[k] = rng.uniform(0, 4)
        return lag[k]

    out = []
    for t, trs, lg in ticks:
        q, _ = dump_truth(cell.dump, t)
        for tr in trs:
            if tr.id in q:
                joined_at.setdefault(tr.id, t)
            else:
                joined_at.pop(tr.id, None)
        new, cache = [], {}
        for tr in trs:
            if tr.id in q:
                t_then = max(t - lag_of(tr.id, t), joined_at[tr.id])
                if t_then not in cache:
                    cache[t_then] = {x.id: x for x in dump_tracks(cell.dump, t_then)}
                old = cache[t_then].get(tr.id, tr)
                new.append(Track(id=tr.id, xy=old.xy, yaw=old.yaw))
            else:
                new.append(tr)
        out.append((t, new, lg))
    return out


def last_of(roster, tracks, est, robot):
    if not roster:
        return None
    bodies = {tr.id + 1: S._Body(tr.id + 1, np.array(tr.xy, dtype=np.float64), tr.yaw) for tr in tracks}
    waiting = sorted(i + 1 for i in roster if i + 1 in bodies)
    if not waiting:
        return None
    turn = S._turn(waiting, set(bodies) - set(waiting), bodies, None, est.tgt, np.array(robot, dtype=np.float64),
                   TH, est.dir)
    return turn.last - 1


def run(cell, stress=None, every_s=0.5):
    e = cell.episode
    end = e["samples"][-1]["t"]
    t0 = cell.asked_t
    n = int(math.floor((end - t0) / every_s)) + 1
    ticks = [(t0 + k * every_s, dump_tracks(cell.dump, t0 + k * every_s), None) for k in range(n)]
    cell._stale = {}
    if stress is not None:
        ticks = stress(cell, ticks)
    est = CT4(cell)
    rows = []
    for t, tracks, _ in ticks:
        est.step(t, tracks, every_s)
        if t + 1e-9 < cell.delivered_t:
            continue
        m = est.marginals()
        dyn = est.b / est.b.sum()
        md = {i: float(dyn[est.bits[:, est.k[i]]].sum()) for i in est.ids}
        q, order = dump_truth(cell.dump, t)
        true_last = max(q, key=lambda i: order[i]) if q else None
        roster = {i for i, v in m.items() if v >= 0.5}
        last = last_of(roster, tracks, est, robot_at(e, t))
        rows.append(dict(t=t, m=m, md=md, q=q, roster=roster, roster_d={i for i, v in md.items() if v >= 0.5},
                         right=(true_last is not None and last == true_last), scored=bool(q)))
    return rows


def main():
    cs = [c for c in cells(roots=("runs/dsv41", "runs/sonnet55_low"), scenes=BYSTANDER_SCENES + ORIGINAL_SCENES,
                           arm="CT3", seeds=range(10, 45)) if c.dump is not None]
    print(f"# Round 3, step A: frozen CT4-min offline on {len(cs)} design cells\n")
    for name, stress in (("clean", None), ("S2m", s2m)):
        bins = defaultdict(lambda: [0, 0])
        flips = flips_d = pairs = right = scored = 0
        for c in cs:
            rows = run(c, stress)
            for r in rows:
                for i, v in r["m"].items():
                    b = min(int(v * 10), 9)
                    bins[b][0] += 1
                    bins[b][1] += i in r["q"]
                if r["scored"]:
                    scored += 1
                    right += r["right"]
            for a, b in zip(rows, rows[1:]):
                pairs += 1
                flips += a["roster"] != b["roster"]
                flips_d += a["roster_d"] != b["roster_d"]
        print(f"## {name}\n")
        print(f"- right last (ticks with a queue): {right / scored:.3f} ({scored} ticks)")
        print(f"- roster changed between consecutive ticks: with the static snapshot {flips / pairs:.3f}; "
              f"dynamic belief alone {flips_d / pairs:.3f}")
        print("- calibration (marginal bin: share in the queue, n):")
        for b in range(10):
            k, hit = bins[b]
            if k:
                print(f"  - {b / 10:.1f}-{(b + 1) / 10:.1f}: {hit / k:.3f} ({k})")
        print()


if __name__ == "__main__":
    main()
