"""Per-cell paired comparison of CT4-min (frozen defaults) against CT3-fix gated on right last share: wins / losses /
ties (|d| < 0.01) by condition and model, bystander and original cells. Design data; zero calls."""
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ct4 import replay_ct4  # noqa: E402
from replay_lib import BYSTANDER_SCENES, CT3, cells, replay, score  # noqa: E402
from stress import STRESS  # noqa: E402
from variants import CT3FixGated  # noqa: E402

res = defaultdict(lambda: [0, 0, 0])
diffs = defaultdict(list)
for cond in (None, "S2", "S3", "S23"):
    st = STRESS[cond] if cond else None
    for c in cells():
        c._stale = {}
        a, _, _ = score(c, replay(c, CT3, start="photo", resolver_cls=CT3FixGated, stress=st))
        c._stale = {}
        b, _, _ = score(c, replay_ct4(c, stress=st))
        d = b["right"] / b["ticks"] - a["right"] / a["ticks"]
        key = (cond or "clean", c.model, "bys" if c.scenario in BYSTANDER_SCENES else "orig")
        res[key][0 if d > 0.01 else 1 if d < -0.01 else 2] += 1
        diffs[key].append(d)
for k in sorted(res):
    w, l, t = res[k]
    m = sum(diffs[k]) / len(diffs[k])
    print(f"{k[0]:6} {k[1]:13} {k[2]:4} CT4-min vs CT3-fix gated, right last: {w}/{l}/{t}, mean {m:+.3f}, "
          f"untied win rate {w / max(w + l, 1):.2f}")
