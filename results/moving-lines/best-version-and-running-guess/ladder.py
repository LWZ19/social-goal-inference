"""Phase 1b ladder on clean tracks (the people dumps): CT2, CT3, and CT3 plus each cheap fix, on role-tracking
measures. Zero calls. Post hoc design data (seeds 10-44).

    .venv/bin/python ladder.py [--stress S1|S2|S3|S4 ...]
"""
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_lib import BYSTANDER_SCENES, CT2, CT3, cells, dump_tracks, replay, score  # noqa: E402
from sgi.resolve import Resolver  # noqa: E402
from variants import CT3Anchored, CT3Fix, CT3FixGated, CT3Group  # noqa: E402
from stress import STRESS  # noqa: E402

RUNGS = [
    ("CT2", Resolver, CT2, "delivery"),
    ("CT3", Resolver, CT3, "delivery"),
    ("+group", CT3Group, CT3, "delivery"),
    ("+photo", Resolver, CT3, "photo"),
    ("+anchor", CT3Anchored, CT3, "delivery"),
    ("+group+photo", CT3Group, CT3, "photo"),
    ("CT3-fix", CT3Fix, CT3, "photo"),
    ("CT3-fix gated", CT3FixGated, CT3, "photo"),
]


def named_tracks(cell):
    at = dump_tracks(cell.dump, cell.asked_t)
    where = {p.number: p.xy for p in cell.people}
    out = set()
    for c in cell.constraints:
        if c.kind != "come_after":
            continue
        for n in c.people:
            if n in where and at:
                tr = min(at, key=lambda tr: math.dist(tr.xy, where[n]))
                if math.dist(tr.xy, where[n]) <= 0.5:
                    out.add(tr.id)
    return out


def run(rungs=RUNGS, extra=None, stress=None):
    cs = cells()
    agg = defaultdict(lambda: defaultdict(float))
    drops = defaultdict(list)
    for c in cs:
        group = (c.model, "bystander" if c.scenario in BYSTANDER_SCENES else "original")
        nb = named_tracks(c) & c.bystanders
        for name, cls, opts, start in rungs + (extra or []):
            ticks = replay(c, opts, source="dump", start=start, resolver_cls=cls, stress=stress)
            tot, fd, dt = score(c, ticks)
            a = agg[(group, name)]
            for k, v in tot.items():
                a[k] += v
            a["false_drops"] += len(fd)
            a["cells"] += 1
            a["captured_cells"] += tot["bys_last"] >= 2          # >= 1 s at 0.5 s ticks
            for b in nb:
                drops[(group, name)].append(None if b not in dt else dt[b] - c.delivered_t)
    return agg, drops


def report(agg, drops, names):
    for group in sorted({g for g, _ in agg}):
        print(f"\n## {group[0]}, {group[1]} scenes ({int(agg[(group, names[0])]['cells'])} cells); seconds over all "
              f"cells from delivery (0.5 s ticks)")
        print("| Rung | right last | empty while a line stands | bystander as last (capture) | cells captured | "
              "bystander in roster | member recall | false drops | named bystanders dropped, median s after delivery |")
        print("|---|---|---|---|---|---|---|---|---|")
        for name in names:
            a = agg[(group, name)]
            n = a["ticks"]
            d = drops[(group, name)]
            got = sorted(x for x in d if x is not None)
            med = f"{got[len(got) // 2]:.1f}" if got else "-"
            print(f"| {name} | {a['right'] / n:.2f} | {a['empty'] / n:.2f} | {a['bys_last'] * 0.5:.0f} s | "
                  f"{int(a['captured_cells'])} | {a['bys_roster'] * 0.5:.0f} s | {a['rec_num'] / max(a['rec_den'], 1):.2f} | "
                  f"{int(a['false_drops'])} | {len(got)} of {len(d)}, {med} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stress", default=None)
    a = ap.parse_args()
    agg, drops = run(stress=STRESS[a.stress] if a.stress else None)
    print(f"# stress: {a.stress or 'none (clean tracks)'}")
    report(agg, drops, [r[0] for r in RUNGS])
