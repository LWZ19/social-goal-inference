"""Which bystander each constraint answer names (lone or pair), CT2 cells, both models (post hoc, zero calls). Run from moving/."""
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bystanders_score import BYS, load, named_bystanders, scenario_info
eps = [e for e in load(["runs/dsv41", "runs/sonnet55_low"], set(BYS), set(range(10, 45))) if e["arm"] == "CT2"]
c = Counter()
for e in eps:
    m = Path(e["_path"]).parts[-4]
    a = next((x for x in e.get("answers", []) if x.get("status") == "ok"), None)
    if not a or a.get("source") != "constraints":
        c[(m, e["scenario"], "non-constraint answer")] += 1
        continue
    info = scenario_info(e["scenario"])
    nb = named_bystanders(e) or []
    kind = "none" if not nb else "+".join(sorted({"lone" if i in info["lone"] else "pair" for i in nb}))
    c[(m, e["scenario"], kind)] += 1
for k in sorted(c):
    print(k, c[k])
