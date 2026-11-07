"""Round 3, step D: who flips in and out of CT4's roster, and why. Design data, fit seeds 10-34, clean and S2m.

For each roster change between consecutive ticks, per person who crossed 0.5: were they a queue member (dump
truth), walking or still, and did the static snapshot's factor for them change more than the dynamic belief's
odds did (log-odds change from static vs from dynamics)? Also a sanity check that CT5 set to behave as CT4 gives
the same rosters as the frozen CT4.

    .venv/bin/python round3/d_flips.py      (from ideas2_1b/)
"""
import math
from collections import Counter

import numpy as np

from r3lib import CONDITIONS, design_cells, dump_truth, ticks_of  # first: it sets the import path
from ct4 import CT4  # noqa: E402
from ct5 import CT5  # noqa: E402


def lo(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def main():
    cs = [c for c in design_cells() if c.seed < 35]
    k = Counter()
    same = diff = 0
    for c in cs:
        for cond in ("clean", "S2m"):
            a, b = CT4(c), CT5(c, dict(WALKERS_STATIC=True, LR_EXIT=1.0))
            prev = None
            for t, tracks, _ in ticks_of(c, CONDITIONS[cond]):
                a.step(t, tracks, 0.5)
                b.step(t, tracks, 0.5)
                if t + 1e-9 < c.delivered_t:
                    continue
                ma, mb = a.marginals(), b.marginals()
                ra = {i for i, v in ma.items() if v >= 0.5}
                same += ra == {i for i, v in mb.items() if v >= 0.5}
                diff += ra != {i for i, v in mb.items() if v >= 0.5}
                dyn = a.b / a.b.sum()
                md = {i: float(dyn[a.bits[:, a.k[i]]].sum()) for i in a.ids}
                q, _ = dump_truth(c.dump, t)
                pos = {tr.id for tr in tracks}
                cur = dict(m=ma, md=md, r=ra)
                if prev is not None:
                    for i in ra ^ prev["r"]:
                        if i not in pos:
                            continue
                        d_total = lo(ma[i]) - lo(prev["m"][i])
                        d_dyn = lo(md[i]) - lo(prev["md"][i])
                        d_static = d_total - d_dyn
                        walking = a._speed(i, t) > a.p["WALK_MPS"]
                        k[("member" if i in q else "non-member", "walking" if walking else "still",
                           "static-driven" if abs(d_static) > abs(d_dyn) else "dynamics-driven")] += 1
                prev = cur
    n = sum(k.values())
    print(f"# Who flips (CT4-min frozen; fit seeds; clean + S2m): {n} crossings of 0.5\n")
    for key, v in k.most_common():
        print(f"- {', '.join(key)}: {v} ({v / n:.3f})")
    print(f"\nSanity: CT5 set as CT4 gives the same roster on {same}/{same + diff} ticks")


if __name__ == "__main__":
    main()
