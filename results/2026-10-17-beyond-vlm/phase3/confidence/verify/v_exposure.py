"""Verifier, post hoc: how much of P1's ranking the design's disclosed exposure (A2's 37 D-flip records, known before
the freeze) already implied. The flip indicator uses the outcome, so it is a diagnostic, never a signal."""
import collections, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v_tables as V
rows, i04, i14 = V.load()
prim = [r for r in rows if r["cell"] in V.PRIMARY]
for r in prim:
    r["sig"] = V.signals(r, i04[r["key"]])
    r["sig"]["flip"] = float(r["D_success"] != r["success"])
    # P1 with the flip records pushed to the top (what the exposure guaranteed at most)
out = {}
for c in list(V.PRIMARY) + ["pooled"]:
    rs = prim if c == "pooled" else [r for r in prim if r["cell"] == c]
    fl = [r for r in rs if r["sig"]["flip"]]
    out[c] = dict(flip_auc=round(V.brute_auroc([r["sig"]["flip"] for r in rs], [not r["success"] for r in rs]), 4),
                  P1_auc=round(V.brute_auroc([r["sig"]["P1"] for r in rs], [not r["success"] for r in rs]), 4),
                  n_flip=len(fl), flip_P1_min=min(r["sig"]["P1"] for r in fl) if fl else None,
                  P1_rank_of_flip_failures_pct=None)
print(json.dumps(out, indent=1))
