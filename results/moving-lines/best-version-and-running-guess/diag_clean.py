"""Where the clean-track leftover of +group+photo and CT3-fix gated sits (by scenario), and every false drop. Zero calls."""
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_lib import BYSTANDER_SCENES, CT3, cells, dump_truth, replay, score  # noqa: E402
from variants import CT3FixGated, CT3Group  # noqa: E402

cap = defaultdict(float)
for c in cells(scenes=BYSTANDER_SCENES):
    for name, cls in (("+group+photo", CT3Group), ("CT3-fix gated", CT3FixGated)):
        ticks = replay(c, CT3, start="photo", resolver_cls=cls)
        tot, fd, dt = score(c, ticks)
        cap[(c.model, name, c.scenario)] += tot["bys_last"] * 0.5
        for i in fd:
            q, order = dump_truth(c.dump, dt[i])
            print(f"false drop: {c.model} {c.scenario} s{c.seed} {name}: track {i} at {dt[i]:.1f} s "
                  f"(delivered {c.delivered_t:.1f}); queue order then {order}")
for k in sorted(cap):
    print(k, f"{cap[k]:.1f} s")
