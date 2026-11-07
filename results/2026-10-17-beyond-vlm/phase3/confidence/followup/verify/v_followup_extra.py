"""Verifier, extra POST HOC checks on claims in the compute agent's reading (changes no verdict). Uses only the
verifier's own v_followup.py functions.
  1. exact draw counts behind the rounded shares (94.4% / 94.3% / 4.3%);
  2. per-cell paired S2 - S5a on (b) (does the edge over the flags 'come from GLM'?);
  3. LOLO logistic ablation trained on the 404 (does 'adding P1 make the score worse', or is it S3?);
  4. selective accuracy at 50% kept on (b) (is S2 'the best zero-call abstain rule' beyond the 80% point?).
"""
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v_followup as F  # the verifier's own module

rows = F.load()
n = len(rows)
layouts = sorted({r["layout"] for r in rows})
lix = {L: i for i, L in enumerate(layouts)}
fail = np.array([not r["success"] for r in rows])
cell = np.array([r["cell"] for r in rows])
lay_idx = np.array([lix[r["layout"]] for r in rows])
inB = np.array([r["D_success"] == r["success"] for r in rows])
X = F.features(rows)
y = fail.astype(float)
cl = F.draw_matrix(13)
counts = np.zeros((F.DRAWS, 13), int)
for d in range(F.DRAWS):
    for a in cl[d]:
        counts[d, a] += 1
sig = {k: np.array([r[k] for r in rows], float) for k in ("P1", "S2", "S3", "S4", "S5a")}

print("1. exact draw counts (b), pooled")
b = {s: F.boot_aucs(sig[s][inB], fail[inB], lay_idx[inB], counts) for s in sig}
s6r, _ = F.lolo(rows, X, y, inB, layouts)
b["S6z_refit"] = F.boot_aucs(s6r[inB], fail[inB], lay_idx[inB], counts)
for a_, b_ in (("S2", "P1"), ("S2", "S5a"), ("S6z_refit", "S2")):
    d = b[a_] - b[b_]
    print(f"  {a_}-{b_}: positive in {(d > 0).sum()} of {len(d)} draws = {(d > 0).mean() * 100:.2f}%  "
          f"(ties {(d == 0).sum()}); formatted .3f -> {(d > 0).mean():.3f}, .1f% -> {(d > 0).mean() * 100:.1f}")
print(f"  S2 pooled (b): min over draws {np.nanmin(b['S2']):.4f}; P1 > 0.5 in {(b['P1'] > 0.5).sum()} of 2000")

print("\n2. per-cell S2 - S5a on (b), paired over the same draws (dropped draws excluded)")
for c in ["old_qwen", "old_glm", "old_deepseek"]:
    m = inB & (cell == c)
    pa = F.auroc(sig["S2"][m], fail[m]) - F.auroc(sig["S5a"][m], fail[m])
    da = F.boot_aucs(sig["S2"][m], fail[m], lay_idx[m], counts) - F.boot_aucs(sig["S5a"][m], fail[m], lay_idx[m], counts)
    ok = ~np.isnan(da)
    print(f"  {c:13s} point {pa:+.3f}  [{np.percentile(da[ok], 2.5):+.3f}, {np.percentile(da[ok], 97.5):+.3f}]  "
          f"positive {(da[ok] > 0).mean():.3f} over {ok.sum()} draws")
# stratified S2 vs S5a
print(f"  stratified (b): S2 {F.stratified(sig['S2'][inB], fail[inB], cell[inB]):.3f}, "
      f"S5a {F.stratified(sig['S5a'][inB], fail[inB], cell[inB]):.3f}")

print("\n3. LOLO logistic trained on the 404 (lambda 1, same features/standardising), AUROC on (b) pooled {strat}")
sets = {"P1,S2,S3 (S6z refit)": [0, 1, 2, 3], "S2 only": [2], "S2,S3": [2, 3], "P1,S2": [0, 1, 2], "P1 only": [0, 1],
        "P1,S3": [0, 1, 3]}
for name, cols in sets.items():
    sc, info = F.lolo(rows, X[:, cols], y, inB, layouts)
    bb = F.boot_aucs(sc[inB], fail[inB], lay_idx[inB], counts)
    d = bb - b["S2"]
    print(f"  {name:22s} pooled {F.auroc(sc[inB], fail[inB]):.3f} {{{F.stratified(sc[inB], fail[inB], cell[inB]):.3f}}}"
          f"  minus raw S2: {F.auroc(sc[inB], fail[inB]) - F.auroc(sig['S2'][inB], fail[inB]):+.3f} "
          f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]  all folds converged {all(i['conv'] for i in info)}")
print(f"  raw S2 pooled {F.auroc(sig['S2'][inB], fail[inB]):.3f}")

print("\n4. selective accuracy at 50% kept on (b), pooled: acc | caught (of 66)")
succ = ~fail
for s in ("P1", "S2", "S3", "S4", "S5a"):
    acc, cau, base = F.selective(sig[s][inB], succ[inB], Fraction(1, 2))
    print(f"  {s:10s} {acc:.3f} | {cau:.1f}  (base {base:.3f})")
sf, _ = F.lolo(rows, X, y, np.ones(n, bool), layouts)
for name, sc in (("S6z_frozen", sf), ("S6z_refit", s6r)):
    acc, cau, base = F.selective(sc[inB], succ[inB], Fraction(1, 2))
    print(f"  {name:10s} {acc:.3f} | {cau:.1f}")

print("\n3b. the LOLO handicap: S2-only LOLO on (a), paired (P1,S2) - (S2 only) on (b), fold weights of S2 only")
sc_a, _ = F.lolo(rows, X[:, [2]], y, np.ones(n, bool), layouts)
print(f"  (a) S2-only LOLO fit on 441: pooled {F.auroc(sc_a, fail):.3f} {{{F.stratified(sc_a, fail, cell):.3f}}} "
      f"vs raw S2 {F.auroc(sig['S2'], fail):.3f}; S6z frozen 0.752")
s2o, _ = F.lolo(rows, X[:, [2]], y, inB, layouts)
p12, _ = F.lolo(rows, X[:, [0, 1, 2]], y, inB, layouts)
d = F.boot_aucs(p12[inB], fail[inB], lay_idx[inB], counts) - F.boot_aucs(s2o[inB], fail[inB], lay_idx[inB], counts)
d2 = b["S6z_refit"] - F.boot_aucs(s2o[inB], fail[inB], lay_idx[inB], counts)
print(f"  (b) LOLO (P1,S2) - LOLO (S2 only): {F.auroc(p12[inB], fail[inB]) - F.auroc(s2o[inB], fail[inB]):+.3f} "
      f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] positive {(d > 0).mean():.3f}")
print(f"  (b) LOLO S6z refit (P1,S2,S3) - LOLO (S2 only): "
      f"{F.auroc(s6r[inB], fail[inB]) - F.auroc(s2o[inB], fail[inB]):+.3f} "
      f"[{np.percentile(d2, 2.5):+.3f}, {np.percentile(d2, 97.5):+.3f}] positive {(d2 > 0).mean():.3f}")
lay = np.array([r["layout"] for r in rows])
print("  S2-only folds trained on the 404 (held-out layout: b, w_S2, held-out (b) failure rate, n):")
for L in layouts:
    tr = (lay != L) & inB
    m = F.fit(X[tr][:, [2]], y[tr])
    te = (lay == L) & inB
    print(f"    {L}: b {m['b']:+.3f} w {m['w'][0]:+.3f}  held-out fail {fail[te].mean():.3f} n {te.sum()}")
