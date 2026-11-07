"""Phase 3 confidence, POST HOC follow-up (README section 8, question 6): do P1, S2, S3, S4 and S6z keep any ranking
once the records whose verdict A2's setting D changes are taken out?

The subsets depend on the outcome (a record is a "flip" when D's verdict differs from C's recorded verdict), so every
number here is a diagnostic. Nothing here changes a verdict of FROZEN.md.

Inputs: the 3 primary row files of pass 1 (rows_old_{qwen,glm,deepseek}.jsonl, 441 records) and summary.json (for
reproduction asserts only). The frozen measure code is imported unchanged from ../code/tables.py (auroc, weighted_auc,
selective, features, fit_logistic, predict, lolo, Boot, verify_frozen). No harness, no recorded run, no scene, no
call, no network. Writes exposure.txt next to this file and nothing else.

    (from the cs6244_project-scenes worktree root)
    PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python -B \
        results/still-scenes/confidence-signal/followup/exposure.py

Subsets of the 441 primary records:
  (a) all;  (b) D_success == success (404);  (c) D_success != success (37).
Signals: P1, S2, S3, S4 (two extra runs), S6z frozen (pass 2's leave-one-layout-out scores over all 441, restricted to
the subset), S6z refit (the same frozen procedure, fitted only on the 404 non-flip records: for each layout L, fit on
the non-flip records of the other 12 layouts and score every record of L, so (c) gets scores from a model that never
saw a flip). Reference rows: S5a (any flag) and the model identity score (each cell's failure rate within the subset;
outcome-derived, a low bar).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
CONF = HERE.parent
_spec = importlib.util.spec_from_file_location("frozen_tables", CONF / "code" / "tables.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

PRIMARY = T.PRIMARY
INF = float("inf")
SIGS = ("P1", "S2", "S3", "S4", "S6z_frozen", "S6z_refit")
REFS = ("S5a", "model_id")
LABEL = {"P1": "P1 re-solve shift (judged)", "S2": "S2 people 1-F1", "S3": "S3 named unsupported",
         "S4": "S4 trial disagreement (2 runs)", "S6z_frozen": "S6z frozen (fit on 441)",
         "S6z_refit": "S6z refit (fit on 404)", "S5a": "S5a any flag (ref.)",
         "model_id": "model identity (ref.)"}
SUBSETS = {"a": ("all 441", lambda r: True), "b": ("D verdict = C (404)", lambda r: not r["flip"]),
           "c": ("D verdict != C (37)", lambda r: r["flip"])}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def f3(x):
    return "  —  " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def lolo_trained_on(rows, train_mask):
    """The frozen S6z procedure (features f1-f4, fold standardising, L2 logistic, lambda 1), leaving one layout out,
    with training restricted to train_mask; every record of the held-out layout is scored."""
    X = T.features(rows, False)
    y = np.array([not r["success"] for r in rows], dtype=np.float64)
    lays = np.array([r["layout"] for r in rows])
    oof = np.full(len(rows), np.nan)
    folds = []
    for L in sorted(set(lays.tolist())):
        te = lays == L
        tr = (~te) & train_mask
        m = T.fit_logistic(X[tr], y[tr])
        oof[te] = T.predict(m, X[te])
        folds.append(dict(layout=L, n_train=int(tr.sum()), iters=m["iters"], converged=m["converged"]))
    full = T.fit_logistic(X[train_mask], y[train_mask])
    assert not np.isnan(oof).any()
    return oof, folds, full


def main():
    t0 = time.monotonic()
    fz = T.verify_frozen()
    rows = []
    for c in PRIMARY:
        rows += [json.loads(line) for line in open(CONF / f"rows_{c}.jsonl")]
    assert len(rows) == 441 and [sum(r["cell"] == c for r in rows) for c in PRIMARY] == [143, 141, 157]
    for r in rows:
        assert isinstance(r["D_success"], bool) and isinstance(r["success"], bool), r["key"]
        r["flip"] = r["D_success"] != r["success"]
    summ = json.load(open(CONF / "summary.json"))

    # ------------------------------------------------------------------ S6z frozen (exactly pass 2) and refit
    oof, _, _ = T.lolo(rows, False)
    for r, p in zip(rows, oof.tolist()):
        r["S6z_frozen"] = p
    all_mask = np.ones(len(rows), dtype=bool)
    chk, _, _ = lolo_trained_on(rows, all_mask)
    assert np.max(np.abs(chk - oof)) == 0.0                     # own loop == frozen lolo with no restriction
    keep = np.array([not r["flip"] for r in rows])
    oof_r, folds_r, full_r = lolo_trained_on(rows, keep)
    for r, p in zip(rows, oof_r.tolist()):
        r["S6z_refit"] = p
    _, _, full_f = lolo_trained_on(rows, all_mask)

    # model identity: each cell's failure rate within the subset (computed per subset below)
    def model_id(rr):
        rate = {c: np.mean([not r["success"] for r in rr if r["cell"] == c]) if any(r["cell"] == c for r in rr)
                else 0.0 for c in PRIMARY}
        return np.array([rate[r["cell"]] for r in rr])

    def values(rr, sig):
        if sig == "model_id":
            return model_id(rr)
        return T.sig_values(rr, sig)

    # ------------------------------------------------------------------ bootstrap: the frozen primary matrix
    rng = np.random.default_rng(T.SEED)
    lay_p = sorted({r["layout"] for r in rows})
    assert len(lay_p) == 13 and lay_p == summ["bootstrap"]["layouts_primary"]
    dp = rng.integers(0, 13, size=(T.DRAWS, 13))                 # first matrix drawn, as in pass 2
    boot = T.Boot(lay_p, dp)

    # ------------------------------------------------------------------ AUROC tables
    res = {}
    draws = {}
    for sk, (_, pick) in SUBSETS.items():
        for sig in SIGS + REFS:
            d = {}
            num = den = 0.0
            for grp in list(PRIMARY) + ["pooled"]:
                rr = [r for r in rows if pick(r) and (grp == "pooled" or r["cell"] == grp)]
                yf = np.array([not r["success"] for r in rr])
                s = values(rr, sig)
                a = T.auroc(s, yf)
                e = dict(n=len(rr), n_fail=int(yf.sum()), auc=a)
                if grp != "pooled" and a is not None:
                    n1, n0 = int(yf.sum()), len(rr) - int(yf.sum())
                    num += a * n1 * n0
                    den += n1 * n0
                if grp == "pooled":
                    e["stratified"] = num / den if den else None
                if sk in ("a", "b") and sig != "model_id":
                    aucs, ok = boot.auc(rr, s, yf)
                    e["ci"] = T.ci(aucs[ok]) if ok.any() else None
                    e["dropped"] = int((~ok).sum())
                    draws[(sk, sig, grp)] = aucs
                d[grp] = e
            res[(sk, sig)] = d

    # ------------------------------------------------------------------ reproduction asserts
    for sig, key in (("P1", "P1"), ("S2", "S2"), ("S3", "S3"), ("S4", "S4"), ("S6z_frozen", "S6z"), ("S5a", "S5a")):
        ref = summ["T2"][key]
        for c in PRIMARY:
            assert abs(res[("a", sig)][c]["auc"] - ref["cells"][c]["auc"]) < 1e-12, (sig, c)
            assert np.allclose(res[("a", sig)][c]["ci"], ref["cells"][c]["ci"], atol=1e-12, rtol=0), (sig, c)
        assert abs(res[("a", sig)]["pooled"]["auc"] - ref["pooled"]["auc"]) < 1e-12, sig
        assert np.allclose(res[("a", sig)]["pooled"]["ci"], ref["pooled"]["ci"], atol=1e-12, rtol=0), sig
        assert abs(res[("a", sig)]["pooled"]["stratified"] - ref["pooled"]["stratified"]) < 1e-12, sig
    vexp = {"old_qwen": 0.543, "old_glm": 0.594, "old_deepseek": 0.532, "pooled": 0.557}   # README section 5
    for g, v in vexp.items():
        assert round(res[("b", "P1")][g]["auc"], 3) == v, (g, res[("b", "P1")][g]["auc"])
    nflip = {c: sum(r["flip"] for r in rows if r["cell"] == c) for c in PRIMARY}
    assert nflip == {"old_qwen": 2, "old_glm": 16, "old_deepseek": 19}, nflip
    breaks = [r for r in rows if r["flip"] and r["success"]]
    assert len(breaks) == 1 and breaks[0]["cell"] == "old_deepseek"
    wt_in_flip = sum(not r["target_correct"] for r in rows if r["flip"])

    # ------------------------------------------------------------------ selective accuracy at 80% kept, subset (b)
    sel = {}
    for sig in SIGS + ("S5a",):
        for grp in list(PRIMARY) + ["pooled"]:
            rr = [r for r in rows if not r["flip"] and (grp == "pooled" or r["cell"] == grp)]
            s = values(rr, sig)
            ys = np.array([r["success"] for r in rr])
            w80, a80 = T.selective(s, ys.tolist(), Fraction(4, 5))
            fail = ~ys
            wt = np.array([not r["target_correct"] for r in rr])
            sel[(sig, grp)] = dict(n=len(rr), base=float(ys.mean()), acc80=a80, caught=float(((1 - w80) * fail).sum()),
                                   failures=int(fail.sum()), caught_wt=float(((1 - w80) * fail * wt).sum()),
                                   wt=int((fail & wt).sum()))
    # reproduction of T5 on all records for S6z and P1 (sanity)
    for sig, key in (("S6z_frozen", "S6z"), ("P1", "P1"), ("S2", "S2")):
        s = values(rows, sig)
        ys = [r["success"] for r in rows]
        _, a = T.selective(s, ys, Fraction(4, 5))
        assert abs(a - summ["T5"][key]["pooled primary"]["acc80"]) < 1e-12, sig

    # ------------------------------------------------------------------ paired differences on (b)
    pairs = [("S2", "P1"), ("S2", "S5a"), ("S3", "P1"), ("S6z_refit", "S2"), ("S6z_frozen", "P1"), ("S4", "S2")]
    pdiff = {}
    for x, yv in pairs:
        dx = draws[("b", x, "pooled")] - draws[("b", yv, "pooled")]
        ok = ~np.isnan(dx)
        pdiff[(x, yv)] = dict(point=res[("b", x)]["pooled"]["auc"] - res[("b", yv)]["pooled"]["auc"],
                              ci=T.ci(dx[ok]), share_pos=float((dx[ok] > 0).mean()), n=int(ok.sum()))
    above_half = {}
    for sig in SIGS + ("S5a",):
        for grp in list(PRIMARY) + ["pooled"]:
            dd = draws[("b", sig, grp)]
            ok = ~np.isnan(dd)
            above_half[(sig, grp)] = float((dd[ok] > 0.5).mean())

    # ------------------------------------------------------------------ write
    L = []
    L.append("Phase 3 confidence, POST HOC exposure follow-up (README section 8, question 6). Changes no verdict.")
    L.append(f"Generated {time.strftime('%F %T %z')} by followup/exposure.py; frozen measure code imported unchanged "
             f"from code/tables.py (sha256 {sha256(CONF / 'code' / 'tables.py')[:16]}...).")
    L.append("Hashes re-verified: " + ", ".join(f"{k} {v[:12]}..." for k, v in fz.items()))
    L.append("Rows: " + ", ".join(f"rows_{c}.jsonl {sha256(CONF / f'rows_{c}.jsonl')[:12]}..." for c in PRIMARY))
    L.append("Failure is the positive class; AUROC by midranks (ties half, +inf ties and ranks top). Intervals: the "
             "frozen primary layout-cluster bootstrap (2,000 paired draws, seed 20261018, first matrix), 95% "
             "percentile; dN = draws dropped (no failure or no success in the resample).")
    L.append("The subsets are chosen by D's verdict against C's recorded verdict, which uses the outcome: diagnostics only.")
    L.append("Reproduction asserts passed: every (a) AUROC, interval and stratified value of P1, S2, S3, S4, S6z and "
             "S5a equals summary.json T2 to 1e-12; P1 on (b) equals the verifier's 0.543 / 0.594 / 0.532 / 0.557; "
             "flips 2 / 16 / 19 with 1 break (DeepSeek); T5 acc@80% of P1, S2 and S6z on (a) reproduced; the refit "
             "loop with no restriction equals pass 2's S6z exactly.")
    L.append("")
    L.append("Counts (n, failures):")
    for sk, (nm, pick) in SUBSETS.items():
        parts = []
        for grp in list(PRIMARY) + ["pooled"]:
            rr = [r for r in rows if pick(r) and (grp == "pooled" or r["cell"] == grp)]
            parts.append(f"{grp} {len(rr)} ({sum(not r['success'] for r in rr)})")
        L.append(f"  ({sk}) {nm:22s} " + ", ".join(parts))
    L.append("  (c) holds 36 fixes (C failed, D succeeded) and 1 break (C succeeded, D failed; old DeepSeek): its "
             "AUROC is undefined for Qwen and GLM (no success) and rests on a single success for DeepSeek and pooled.")
    L.append(f"  Wrong-target failures among the 37 flips: {wt_in_flip}. So all 34 wrong-target failures of the 441 "
             "are in (b), and (b)'s 66 failures are 34 wrong-target and 32 right-target." if wt_in_flip == 0 else
             f"  Wrong-target failures among the 37 flips: {wt_in_flip}.")
    L.append("")

    def row_line(sk, sig, with_ci):
        d = res[(sk, sig)]
        parts = []
        for grp in list(PRIMARY) + ["pooled"]:
            e = d[grp]
            if with_ci and e.get("ci"):
                parts.append(f"{f3(e['auc'])} [{f3(e['ci'][0])},{f3(e['ci'][1])}] d{e['dropped']}")
            else:
                parts.append(f3(e["auc"]))
        w = 28 if with_ci else 14
        return f"  {LABEL[sig]:32s}" + "".join(f"{p:>{w}s}" for p in parts) + f"{f3(d['pooled']['stratified']):>9s}"

    for sk, title, with_ci in (("a", "Table A. AUROC for failure, (a) all 441 primary records", True),
                               ("b", "Table B. AUROC for failure, (b) the 404 records where D's verdict equals C's", True),
                               ("c", "Table C. AUROC for failure, (c) the 37 records where D's verdict differs "
                                     "(36 failures, 1 success)", False)):
        L.append(title)
        w = 28 if with_ci else 14
        L.append(f"  {'signal':32s}" + "".join(f"{g:>{w}s}" for g in list(PRIMARY) + ["pooled"]) + f"{'strat.':>9s}")
        for sig in SIGS + REFS:
            if sk == "c" and sig == "model_id":
                continue
            L.append(row_line(sk, sig, with_ci))
        L.append("")
    L.append("  model identity = each cell's failure rate within the subset used as the score (outcome-derived; pooled "
             "only; per cell it is constant, so 0.500).")
    L.append("")

    L.append("Table D. Change from (a) to (b), pooled one ranking and stratified")
    for sig in SIGS + ("S5a",):
        a, b = res[("a", sig)]["pooled"], res[("b", sig)]["pooled"]
        L.append(f"  {LABEL[sig]:32s} pooled {f3(a['auc'])} -> {f3(b['auc'])} ({b['auc'] - a['auc']:+.3f});  "
                 f"stratified {f3(a['stratified'])} -> {f3(b['stratified'])} ({b['stratified'] - a['stratified']:+.3f})")
    L.append("")

    L.append("Table E. Share of the (b) bootstrap draws with AUROC above 0.5")
    L.append(f"  {'signal':32s}" + "".join(f"{g:>14s}" for g in list(PRIMARY) + ["pooled"]))
    for sig in SIGS + ("S5a",):
        L.append(f"  {LABEL[sig]:32s}" + "".join(f"{above_half[(sig, g)]:>14.3f}" for g in list(PRIMARY) + ["pooled"]))
    L.append("")

    L.append("Table F. Paired differences in pooled AUROC on (b), same draws (descriptive)")
    for (x, yv), d in pdiff.items():
        L.append(f"  {LABEL[x]:28s} - {LABEL[yv]:28s} {d['point']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]  "
                 f"share > 0: {d['share_pos']:.3f} over {d['n']} draws")
    L.append("")

    L.append("Table G. Selective accuracy at 80% kept on (b) (frozen rule: ties at the cutoff kept in part), with "
             "failures caught at 20% abstention (random = 0.2 x failures)")
    L.append(f"  {'signal':32s}" + "".join(f"{g:>26s}" for g in list(PRIMARY) + ["pooled"]))
    base = [sel[("P1", g)] for g in list(PRIMARY) + ["pooled"]]
    L.append(f"  {'base rate (keep all) / random':32s}" + "".join(
        f"{e['base']:.3f} | {0.2 * e['failures']:.1f}/{e['failures']}".rjust(26) for e in base))
    for sig in SIGS + ("S5a",):
        L.append(f"  {LABEL[sig]:32s}" + "".join(
            f"{sel[(sig, g)]['acc80']:.3f} ({sel[(sig, g)]['acc80'] - sel[(sig, g)]['base']:+.3f}) | "
            f"{sel[(sig, g)]['caught']:.1f}".rjust(26) for g in list(PRIMARY) + ["pooled"]))
    e = sel[("S2", "pooled")]
    L.append(f"  (each entry: success among kept (difference from base) | failures caught; (b) has {e['failures']} "
             f"failures pooled, {e['wt']} of them wrong-target)")
    L.append("")

    L.append("Table H. What the signals see on (b): right-target failures against (b)'s successes, and wrong-target "
             "failures against (b)'s successes (point values; per cell and pooled one ranking)")
    L.append(f"  {'signal':32s}" + "".join(f"{g:>14s}" for g in list(PRIMARY) + ["pooled"]) + "   |"
             + "".join(f"{g:>14s}" for g in list(PRIMARY) + ["pooled"]))
    hdr_n = []
    for kind in ("rt", "wt"):
        for grp in list(PRIMARY) + ["pooled"]:
            rr = [r for r in rows if not r["flip"] and (grp == "pooled" or r["cell"] == grp)]
            npos = sum((not r["success"]) and (r["target_correct"] == (kind == "rt")) for r in rr)
            hdr_n.append(f"{kind} {npos}")
    L.append(f"  {'(positives)':32s}" + "".join(f"{h:>14s}" for h in hdr_n[:4]) + "   |"
             + "".join(f"{h:>14s}" for h in hdr_n[4:]))
    for sig in SIGS + ("S5a",):
        parts = []
        for kind in ("rt", "wt"):
            for grp in list(PRIMARY) + ["pooled"]:
                rr = [r for r in rows if not r["flip"] and (grp == "pooled" or r["cell"] == grp)
                      and (r["success"] or r["target_correct"] == (kind == "rt"))]
                yf = np.array([not r["success"] for r in rr])
                parts.append(f3(T.auroc(values(rr, sig), yf)))
        L.append(f"  {LABEL[sig]:32s}" + "".join(f"{x:>14s}" for x in parts[:4]) + "   |"
                 + "".join(f"{x:>14s}" for x in parts[4:]))
    L.append("")

    L.append("Table I. S6z coefficients, full-data fit on standardised features (lambda 1; intercept unpenalised)")
    fn = ["ln(1+min(P1,10))", "1{P1=inf}", "S2", "S3"]
    for nm, m in (("fit on 441 (frozen)", full_f), ("fit on 404 (refit)", full_r)):
        L.append(f"  {nm:22s} b {m['b']:+.3f}  " + "  ".join(f"{f} {w:+.3f}" for f, w in zip(fn, m["w"].tolist()))
                 + f"  (iters {m['iters']}, converged {m['converged']})")
    L.append(f"  refit folds: {len(folds_r)}, all converged {all(f['converged'] for f in folds_r)}, max iters "
             f"{max(f['iters'] for f in folds_r)}, training sizes {min(f['n_train'] for f in folds_r)}-"
             f"{max(f['n_train'] for f in folds_r)}")
    L.append("")

    # plain reading, computed
    b2 = res[("b", "S2")]
    b1 = res[("b", "P1")]
    L.append("Reading (computed values; inference marked):")
    L.append(f"  S2 on (b): cells {f3(b2['old_qwen']['auc'])} / {f3(b2['old_glm']['auc'])} / "
             f"{f3(b2['old_deepseek']['auc'])}, pooled {f3(b2['pooled']['auc'])} [{f3(b2['pooled']['ci'][0])}, "
             f"{f3(b2['pooled']['ci'][1])}], stratified {f3(b2['pooled']['stratified'])}; against P1's "
             f"{f3(b1['old_qwen']['auc'])} / {f3(b1['old_glm']['auc'])} / {f3(b1['old_deepseek']['auc'])}, pooled "
             f"{f3(b1['pooled']['auc'])} [{f3(b1['pooled']['ci'][0])}, {f3(b1['pooled']['ci'][1])}].")
    d = pdiff[("S2", "P1")]
    L.append(f"  S2 - P1 on (b): {d['point']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}], positive in "
             f"{d['share_pos']:.1%} of draws.")
    L.append(f"  Model identity on (b) pooled: {f3(res[('b', 'model_id')]['pooled']['auc'])} (the bar any pooled "
             "value must clear before it says more than which model answered).")
    (HERE / "exposure.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"done in {time.monotonic() - t0:.1f} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
