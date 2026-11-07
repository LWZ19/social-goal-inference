"""Verifier, post hoc robustness probes (changes no verdict): ties, rounding, +inf readings, right-target subset,
single-layout and single-trial influence, between-cell confounding, bootstrap stability of the criterion."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v_tables as V  # noqa: E402  (own code)

INF = float("inf")
rows, i04, i14 = V.load()
for r in rows:
    r["sig"] = V.signals(r, i04[r["key"]])
P = V.PRIMARY
by = collections.defaultdict(list)
for r in rows:
    by[r["cell"]].append(r)
prim = [r for c in P for r in by[c]]


def auc(rs, s, lab=lambda r: not r["success"]):
    return V.brute_auroc([r["sig"][s] for r in rs], [lab(r) for r in rs])


def bounds(rs, s):
    pos = np.array([r["sig"][s] for r in rs if not r["success"]])
    neg = np.array([r["sig"][s] for r in rs if r["success"]])
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    n = len(pos) * len(neg)
    return gt / n, (gt + eq) / n, eq / n


def verdict(sname, subset=lambda r: True):
    cells = [auc([r for r in by[c] if subset(r)], sname) for c in P]
    pooled = auc([r for r in prim if subset(r)], sname)
    s5a = auc([r for r in prim if subset(r)], "S5a")
    ok = sum(c is not None and c >= 0.70 for c in cells)
    v = "kill" if pooled < 0.60 else "success" if ok >= 2 and pooled > s5a else "neither"
    return dict(cells=[None if c is None else round(c, 4) for c in cells], pooled=round(pooled, 4),
                s5a=round(s5a, 4), cells_ok=ok, verdict=v)


out = {}
# 1. ties
out["tie_bounds_P1"] = {c: [round(x, 4) for x in bounds(by[c], "P1")] for c in P}
out["tie_bounds_P1"]["pooled"] = [round(x, 4) for x in bounds(prim, "P1")]
out["P1_zero_by_label"] = {c: dict(fail=sum(r["sig"]["P1"] == 0 and not r["success"] for r in by[c]),
                                   succ=sum(r["sig"]["P1"] == 0 and r["success"] for r in by[c])) for c in P}
small = [r["sig"]["P1"] for r in prim]
out["P1_distribution_primary"] = {"=0": sum(x == 0 for x in small), "(0,1mm)": sum(0 < x < 1e-3 for x in small),
                                  "[1mm,1cm)": sum(1e-3 <= x < 1e-2 for x in small),
                                  "[1cm,10cm)": sum(1e-2 <= x < 0.1 for x in small),
                                  "[10cm,1m)": sum(0.1 <= x < 1 for x in small), ">=1m": sum(x >= 1 for x in small)}
# 2. alternative P1 readings
for s in ("P1", "P1b", "P1_noround", "P1_mm", "P1_cm", "P1_step0"):
    out[f"verdict_{s}"] = verdict(s)
# 3. right-target subset (descriptive; conditions on a scorer field, so not a deployable criterion)
out["verdict_P1_right_target_only"] = verdict("P1", lambda r: r["target_correct"])
# 4. single-layout influence on the criterion
lays = sorted({r["layout"] for r in prim})
lolo = {}
flips = []
for L in lays:
    keep = lambda r, L=L: r["layout"] != L
    v = verdict("P1", keep)
    lolo[L] = v
    if v["verdict"] != "neither":
        flips.append((L, v))
out["lolo_cells_range"] = {c: [min(v["cells"][i] for v in lolo.values() if v["cells"][i] is not None),
                               max(v["cells"][i] for v in lolo.values() if v["cells"][i] is not None)]
                           for i, c in enumerate(P)}
out["lolo_pooled_range"] = [min(v["pooled"] for v in lolo.values()), max(v["pooled"] for v in lolo.values())]
out["lolo_verdict_flips"] = flips
# 5. single trial
out["by_trial"] = {t: verdict("P1", lambda r, t=t: r["trial"] == t) for t in (0, 1, 2)}
# 6. between-cell confounding: model identity alone (cell failure rate as the signal)
rate = {c: sum(not r["success"] for r in by[c]) / len(by[c]) for c in P}
for r in prim:
    r["sig"]["cellrate"] = rate[r["cell"]]
out["pooled_auc_cell_identity"] = round(auc(prim, "cellrate"), 4)
# 7. bootstrap: share of draws in which each cell >= 0.70 and in which the success condition holds
rng = np.random.default_rng(20261018)
D = rng.integers(0, 13, size=(2000, 13))
K = np.stack([np.bincount(d, minlength=13) for d in D]).astype(float)
L = {l: i for i, l in enumerate(lays)}
draws = {}
for c in P:
    Cm, F, S = V.conc_matrix([r["sig"]["P1"] for r in by[c]], [not r["success"] for r in by[c]],
                             [L[r["layout"]] for r in by[c]], 13)
    draws[c] = V.auroc_draws(Cm, F, S, K)
for s in ("P1", "S5a"):
    Cm, F, S = V.conc_matrix([r["sig"][s] for r in prim], [not r["success"] for r in prim],
                             [L[r["layout"]] for r in prim], 13)
    draws["pooled_" + s] = V.auroc_draws(Cm, F, S, K)
ge = {c: float(np.nanmean(draws[c] >= 0.70)) for c in P}
okc = sum(np.nan_to_num(draws[c], nan=0) >= 0.70 for c in P)
succ = (okc >= 2) & (draws["pooled_P1"] > draws["pooled_S5a"]) & (draws["pooled_P1"] >= 0.60)
kill = draws["pooled_P1"] < 0.60
out["boot_share_cell_ge_070"] = {c: round(v, 3) for c, v in ge.items()}
out["boot_share_success"] = round(float(succ.mean()), 3)
out["boot_share_kill"] = round(float(kill.mean()), 3)
# 8. mechanism: P1 restricted to records where D's verdict equals C's (D_success == success): does P1 still rank?
for c in P:
    same = [r for r in by[c] if r.get("D_success") == r["success"]]
    out.setdefault("P1_where_D_verdict_unchanged", {})[c] = dict(n=len(same), n_fail=sum(not r["success"] for r in same),
                                                                 auc=None if auc(same, "P1") is None else round(auc(same, "P1"), 4))
same = [r for r in prim if r.get("D_success") == r["success"]]
out["P1_where_D_verdict_unchanged"]["pooled"] = dict(n=len(same), n_fail=sum(not r["success"] for r in same),
                                                     auc=round(auc(same, "P1"), 4))
out["D_flips"] = {c: dict(fail_to_succ=sum((not r["success"]) and r["D_success"] for r in by[c]),
                          succ_to_fail=sum(r["success"] and not r["D_success"] for r in by[c])) for c in P}
# 9. S4 +inf counts overall
out["S4_inf_total"] = dict(missing=sum(r["S4_missing_place"] for r in rows),
                           target=sum((not r["S4_missing_place"]) and r["S4_target_spread"] > 1.0 for r in rows))
out["S4_inf_primary"] = dict(missing=sum(r["S4_missing_place"] for r in prim),
                             target=sum((not r["S4_missing_place"]) and r["S4_target_spread"] > 1.0 for r in prim))
print(json.dumps(out, indent=1, default=str))
(HERE / "vbreak.json").write_text(json.dumps(out, indent=1, default=str))
