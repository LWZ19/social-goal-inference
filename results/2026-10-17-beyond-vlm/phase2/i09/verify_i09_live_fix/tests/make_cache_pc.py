"""From the count-only rows of the NEW driver on cache_full, make cache_pc = cache_full minus every constraints key
(as verify_i09_live_build/d_run.sh did), and write keys.json (constraints keys removed; context keys by record)."""
import collections
import glob
import json
import shutil
from pathlib import Path

V = Path(__file__).resolve().parent
full, pc = V / "cache_full", V / "cache_pc"
rows = []
for m in ("glm53", "dsv41"):
    path = sorted(glob.glob(str(V / "cnt_full" / f"count_only_{m}_*.jsonl")))[-1]
    rows += [json.loads(line) for line in open(path)]
outcomes = collections.Counter((r["stage"], r["outcome"]) for r in rows)
con = sorted({r["key"] for r in rows if r["stage"] == "constraints"})
ctx = {f"{r['arm']}|{r['scene']}|{r['episode']}|{m}": r["key"] for m in ("glm53", "dsv41") for r in rows
       if r["stage"] == "context" and False}
if pc.exists():
    shutil.rmtree(pc)
shutil.copytree(full, pc)
for k in con:
    (pc / f"{k}.json").unlink()
# context keys per record, in run order, per model (the rows carry no model: split by the count-only file)
by_model = {}
for m in ("glm53", "dsv41"):
    path = sorted(glob.glob(str(V / "cnt_full" / f"count_only_{m}_*.jsonl")))[-1]
    by_model[m] = [{"arm": r["arm"], "scene": r["scene"], "episode": r["episode"], "stage": r["stage"], "key": r["key"]}
                   for r in map(json.loads, open(path))]
doc = {"outcomes_on_full": {f"{s}:{o}": n for (s, o), n in outcomes.items()}, "constraints_removed": con,
       "n_removed": len(con), "files_full": len(list(full.glob("*.json"))), "files_pc": len(list(pc.glob("*.json"))),
       "rows_by_model": by_model}
(V / "keys.json").write_text(json.dumps(doc, indent=1))
print({k: v for k, v in doc.items() if k not in ("constraints_removed", "rows_by_model")})
