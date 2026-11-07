"""Phase 3 confidence, pass 2 (FROZEN.md section 9, step 4): R8 first, then S6 / S6z and every measure of section 5,
the verdict of section 7 and the prediction reading of section 8. numpy only; reads pass 1's rows_<cell>.jsonl and
checks.txt (refuses to run unless R1-R7 all passed). No call, no harness, no scene.

    PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
        .venv/bin/python -B code/tables.py          -> tables.txt, summary.json, pass2.log
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1]
FROZEN_SHA = "2a3a890a3b23ff466e8b72c63e6dbef9143d7738f68badc34a4fad6a17ccb4b3"
INF = float("inf")
SEED = 20261018
DRAWS = 2000
LAMBDA = 1.0
TOL_R8 = 0.0005

PRIMARY = ("old_qwen", "old_glm", "old_deepseek")
WATCH = ("watch_qwen", "watch_glm", "watch_deepseek")
S23 = ("new_glm", "new_deepseek", "new_haiku", "new_sonnet", "new_gemini")
CELLS = PRIMARY + WATCH + S23
FLAGS = ("exclusion_broken", "far_from_target", "constraint_unusable")

# signal -> (label, deployment cost, defined at watching)
SIGNALS = {
    "P1": ("P1 re-solve sensitivity (judged)", "detector + 1 solver run (0 extra calls)", False),
    "P1b": ("P1b, + target-person change", "detector + 1 solver run (0 extra calls)", False),
    "S2": ("S2 people disagreement 1-F1", "detector only (0 extra calls)", True),
    "S3": ("S3 named-person unsupported", "detector only (0 extra calls)", True),
    "S4": ("S4 trial disagreement", "TWO EXTRA FULL PIPELINE RUNS (extra calls)", True),
    "S4_place": ("S4-place (no target rule)", "TWO EXTRA FULL PIPELINE RUNS (extra calls)", True),
    "S5a": ("S5a any flag (baseline)", "free", True),
    "flag_exclusion_broken": ("flag exclusion_broken", "free", True),
    "flag_far_from_target": ("flag far_from_target", "free", True),
    "flag_constraint_unusable": ("flag constraint_unusable", "free", True),
    "S5b": ("S5b 1 - I04 V1 concentration", "free", True),
    "S6": ("S6 LOLO logistic (P1,S2,S3,S4)", "all of the above, incl. TWO EXTRA RUNS", False),
    "S6z": ("S6z LOLO logistic (P1,S2,S3)", "detector + 1 solver run (0 extra calls)", False),
}
JUDGED_SINGLE = ("P1", "S2", "S3", "S4", "S5a", "S5b")          # FROZEN.md section 8, part 2
I14_SIGNALS = ("S2", "S3", "S4", "S4_place", "S5a", "flag_exclusion_broken", "flag_far_from_target",
               "flag_constraint_unusable", "S5b")

# FROZEN.md section 6, R8 (phase1d Tables 1.5 and 1.7)
R8_CELLS = ("old_qwen", "old_glm", "old_deepseek", "watch_qwen", "watch_glm", "watch_deepseek")
R8_EXPECT = {
    "S5b": (0.559, 0.303, 0.540, 0.312, 0.579, 0.627),
    "flag_exclusion_broken": (0.500, 0.511, 0.531, 0.500, 0.500, 0.500),
    "flag_far_from_target": (0.500, 0.511, 0.500, 0.500, 0.538, 0.500),
    "flag_constraint_unusable": (0.478, 0.659, 0.496, 0.500, 0.555, 0.435),
}
R8_POOLED = {"S5b": 0.470, "flag_exclusion_broken": 0.515, "flag_far_from_target": 0.508,
             "flag_constraint_unusable": 0.563}
R8_S23 = {"new_glm": 0.318, "new_deepseek": 0.607, "new_haiku": 0.683, "new_sonnet": 0.898, "new_gemini": 0.532}

LOG_LINES = []


def now():
    return time.strftime("%F %T %z")


def log(msg):
    line = f"{now()} {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)
    with open(OUT / "pass2.log", "a") as f:
        f.write(line + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen():
    listed = (OUT / "FROZEN.sha256").read_text().split()
    assert listed[1] == "FROZEN.md" and listed[0] == FROZEN_SHA, listed
    got = sha256(OUT / "FROZEN.md")
    assert got == FROZEN_SHA, f"FROZEN.md hash mismatch: {got}"
    out = {"FROZEN.md": got}
    for listing in sorted(OUT.glob("DEVIATIONS*.sha256")):        # DEVIATIONS.md, DEVIATIONS_2.md (DEVIATIONS_2.md)
        h, name = listing.read_text().split()[:2]
        got_d = sha256(OUT / name)
        assert got_d == h, f"{name} hash mismatch: {got_d} != {h}"
        out[name] = got_d
    return out


def r8_close(got, expected):
    """|got - expected| <= 0.0005 in exact arithmetic: got as the exact float, expected as the published decimal
    (DEVIATIONS_2.md)."""
    return abs(Fraction(got) - Fraction(str(expected))) <= Fraction(5, 10000)


# ------------------------------------------------------------------------------------------------ measures
def auroc(s, yfail):
    """AUROC for failure: (R1 - n1(n1+1)/2) / (n1 n0), R1 the failures' midranks (ties half; +inf ties, ranks top)."""
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(yfail, dtype=bool)
    assert not np.isnan(s).any()
    n = len(s)
    n1 = int(y.sum())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r1 = ranks[y].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def weighted_auc(s, yfail, W):
    """AUROC for failure of each row of weights W (D, n): pairs weighted by w_i w_j, ties half. Equals the midrank
    AUROC of the sample with each record repeated w times. Returns (auc (D,), ok (D,))."""
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(yfail, dtype=bool)
    u, vidx = np.unique(s, return_inverse=True)
    onehot = np.zeros((len(s), len(u)))
    onehot[np.arange(len(s)), vidx] = 1.0
    F = (W * y) @ onehot
    S = (W * ~y) @ onehot
    W1, W0 = F.sum(axis=1), S.sum(axis=1)
    below = np.cumsum(S, axis=1) - S
    conc = (F * (below + 0.5 * S)).sum(axis=1)
    ok = (W1 > 0) & (W0 > 0)
    auc = np.full(W.shape[0], np.nan)
    auc[ok] = conc[ok] / (W1[ok] * W0[ok])
    return auc, ok


def selective(s, ysucc, q: Fraction):
    """Weights at coverage q (ties at the cutoff kept in part, expected value under random tie-breaking) and the
    selective accuracy sum(w y) / K, K = q n."""
    s = np.asarray(s, dtype=np.float64)
    n = len(s)
    K = q * n
    u, vidx = np.unique(s, return_inverse=True)
    counts = np.bincount(vidx, minlength=len(u))
    before = np.cumsum(counts) - counts
    gw = []
    for c, g in zip(before.tolist(), counts.tolist()):
        if c + g <= K:
            gw.append(Fraction(1))
        elif c >= K:
            gw.append(Fraction(0))
        else:
            gw.append((K - c) / g)
    w = [gw[v] for v in vidx.tolist()]
    acc = sum((wi for wi, yi in zip(w, ysucc) if yi), Fraction(0)) / K
    assert sum(w, Fraction(0)) == K
    return np.array([float(x) for x in w]), float(acc)


# ------------------------------------------------------------------------------------------------ S6
def features(rows, with_s4):
    def lcap(v):
        return math.log(1.0 + min(v, 10.0))
    X = []
    for r in rows:
        f = [lcap(r["P1"]), float(r["P1"] == INF), r["S2"], r["S3"]]
        if with_s4:
            f += [lcap(r["S4"]), float(r["S4"] == INF)]
        X.append(f)
    return np.array(X, dtype=np.float64)


def fit_logistic(X, y):
    """L2 logistic regression (lambda 1, intercept free) on fold-standardised features, Newton-Raphson from zero."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    active = sd >= 1e-12
    Z = (X[:, active] - mu[active]) / sd[active]
    A = np.column_stack([np.ones(len(X)), Z])
    pen = np.r_[0.0, np.full(Z.shape[1], LAMBDA)]
    beta = np.zeros(A.shape[1])

    def grad_p(b):
        p = 1.0 / (1.0 + np.exp(-(A @ b)))
        return A.T @ (p - y) + pen * b, p
    it = 0
    g, p = grad_p(beta)
    while np.max(np.abs(g)) >= 1e-9 and it < 100:
        Hm = A.T @ (A * (p * (1.0 - p))[:, None]) + np.diag(pen)
        beta = beta - np.linalg.solve(Hm, g)
        it += 1
        g, p = grad_p(beta)
    converged = bool(np.max(np.abs(g)) < 1e-9)
    w = np.zeros(X.shape[1])
    w[active] = beta[1:]
    return dict(b=float(beta[0]), w=w, mu=mu, sd=sd, active=active, iters=it, converged=converged,
                max_grad=float(np.max(np.abs(g))))


def predict(m, X):
    act = m["active"]
    z = m["b"] + ((X[:, act] - m["mu"][act]) / m["sd"][act]) @ m["w"][act]
    return 1.0 / (1.0 + np.exp(-z))


def lolo(rows, with_s4):
    X = features(rows, with_s4)
    y = np.array([not r["success"] for r in rows], dtype=np.float64)
    lays = np.array([r["layout"] for r in rows])
    oof = np.full(len(rows), np.nan)
    folds = []
    for L in sorted(set(lays.tolist())):
        te = lays == L
        m = fit_logistic(X[~te], y[~te])
        oof[te] = predict(m, X[te])
        folds.append(dict(layout=L, n_train=int((~te).sum()), n_test=int(te.sum()), b=m["b"], w=m["w"].tolist(),
                          active=m["active"].tolist(), iters=m["iters"], converged=m["converged"],
                          max_grad=m["max_grad"]))
    full = fit_logistic(X, y)
    assert not np.isnan(oof).any()
    return oof, folds, dict(b=full["b"], w=full["w"].tolist(), mu=full["mu"].tolist(), sd=full["sd"].tolist(),
                            active=full["active"].tolist(), iters=full["iters"], converged=full["converged"],
                            max_grad=full["max_grad"])


# ------------------------------------------------------------------------------------------------ helpers
def f3(x):
    return "  —  " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def sig_values(rows, sig):
    out = []
    for r in rows:
        if sig.startswith("flag_"):
            out.append(float(r["flags"][sig[5:]]))
        else:
            v = r[sig]
            assert v is not None, (sig, r["key"])
            out.append(float(v))
    return np.array(out, dtype=np.float64)


def ci(arr):
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return [float(lo), float(hi)]


class Boot:
    """One layout-cluster draw matrix shared by every signal and cell of its group (paired draws)."""

    def __init__(self, layouts, draws):
        self.layouts = layouts
        self.index = {L: i for i, L in enumerate(layouts)}
        self.M = np.stack([np.bincount(d, minlength=len(layouts)) for d in draws]).astype(np.float64)

    def weights(self, rows):
        idx = np.array([self.index[r["layout"]] for r in rows])
        return self.M[:, idx]

    def auc(self, rows, s, yfail):
        aucs, ok = weighted_auc(s, yfail, self.weights(rows))
        return aucs, ok


def main():
    t0 = time.monotonic()
    fz = verify_frozen()
    log(f"pass 2 start; hashes verified {fz}")
    head = (OUT / "checks.txt").read_text().splitlines()
    assert "ALL PASS" in head[0], "R1-R7 did not all pass: nothing is scored"
    rows_all = []
    for c in CELLS:
        rows_all += [json.loads(line) for line in open(OUT / f"rows_{c}.jsonl")]
    by = {c: [r for r in rows_all if r["cell"] == c] for c in CELLS}
    T = []                                                       # tables.txt lines
    summary = {"frozen_sha256": fz, "generated": now(), "R8": {}, "T1": {}, "T2": {}, "T3": {}, "T4": {}, "T5": {},
               "T6": {}, "T7": {}, "T8": {}, "T9": {}}

    # ---------------------------------------------------------------- R8 first
    r8_lines, r8_ok = [], True
    for sig, exp in R8_EXPECT.items():
        got = []
        for c, e in zip(R8_CELLS, exp):
            a = auroc(sig_values(by[c], sig), [not r["success"] for r in by[c]])
            good = r8_close(a, e)
            r8_ok &= good
            got.append(a)
            summary["R8"][f"{sig}/{c}"] = dict(got=a, expected=e, ok=good)
        r8_lines.append(f"  {sig:26s} " + "  ".join(f"{c} {g:.4f} ({e:.3f}{'' if r8_close(g, e) else ' FAIL'})"
                                                   for c, g, e in zip(R8_CELLS, got, exp)))
    pool549 = [r for c in R8_CELLS for r in by[c]]
    for sig, e in R8_POOLED.items():
        a = auroc(sig_values(pool549, sig), [not r["success"] for r in pool549])
        good = r8_close(a, e)
        r8_ok &= good
        summary["R8"][f"{sig}/pooled549"] = dict(got=a, expected=e, ok=good)
        r8_lines.append(f"  {sig:26s} pooled 549 (one ranking) {a:.4f} ({e:.3f}{'' if good else ' FAIL'})")
    for c, e in R8_S23.items():
        a = auroc(sig_values(by[c], "S5b"), [not r["success"] for r in by[c]])
        good = r8_close(a, e)
        r8_ok &= good
        summary["R8"][f"S5b/{c}"] = dict(got=a, expected=e, ok=good)
        r8_lines.append(f"  S5b (c_V1, Table 1.7)      {c} {a:.4f} ({e:.3f}{'' if good else ' FAIL'})")
    log(f"R8 AUROC code check: {'PASS' if r8_ok else 'FAIL'}")
    for line in r8_lines:
        log(line)
    summary["R8"]["pass"] = bool(r8_ok)
    if not r8_ok:
        (OUT / "tables.txt").write_text("R8 FAILED: the AUROC code does not reproduce phase 1d; nothing is scored.\n"
                                        + "\n".join(r8_lines) + "\n")
        log("stopped: R8 failed (FROZEN.md section 6 stop rule)")
        return 2

    # weighted AUROC at weight 1 equals the midrank AUROC (self-check)
    for c in CELLS:
        for sig in ("S2", "S3", "S4", "S5b"):
            s = sig_values(by[c], sig)
            yf = np.array([not r["success"] for r in by[c]])
            a1 = auroc(s, yf)
            a2, _ = weighted_auc(s, yf, np.ones((1, len(s))))
            assert abs(a1 - a2[0]) < 1e-12, (c, sig, a1, a2)

    # ---------------------------------------------------------------- S6 / S6z (out-of-fold)
    s6info = {}
    for grp, cells in (("primary", PRIMARY), ("s23", S23)):
        rows = [r for c in cells for r in by[c]]
        for name, with_s4 in (("S6", True), ("S6z", False)):
            oof, folds, full = lolo(rows, with_s4)
            for r, p in zip(rows, oof.tolist()):
                r[name] = p
            s6info[f"{grp}/{name}"] = dict(folds=folds, full=full)
    for c in WATCH:
        for r in by[c]:
            r["S6"] = r["S6z"] = None

    # ---------------------------------------------------------------- bootstrap matrices (order fixed)
    rng = np.random.default_rng(SEED)
    lay_p = sorted({r["layout"] for c in PRIMARY for r in by[c]})
    lay_s = sorted({r["layout"] for c in S23 for r in by[c]})
    lay_w = sorted({r["layout"] for c in WATCH for r in by[c]})
    assert (len(lay_p), len(lay_s), len(lay_w)) == (13, 11, 6), (len(lay_p), len(lay_s), len(lay_w))
    dp = rng.integers(0, 13, size=(DRAWS, 13))
    ds = rng.integers(0, 11, size=(DRAWS, 11))
    dw = rng.integers(0, 6, size=(DRAWS, 6))
    boots = {"primary": Boot(lay_p, dp), "s23": Boot(lay_s, ds), "watch": Boot(lay_w, dw)}
    summary["bootstrap"] = dict(seed=SEED, draws=DRAWS, layouts_primary=lay_p, layouts_s23=lay_s, layouts_watch=lay_w)

    def signals_for(cells):
        return [s for s in SIGNALS if SIGNALS[s][2] or not set(cells) & set(WATCH)]

    # ---------------------------------------------------------------- AUROC block
    def auc_block(cells, pooled_name, boot, subset=lambda r: True, positive=lambda r: not r["success"],
                  negative=lambda r: r["success"], do_boot=True):
        out = {}
        for sig in signals_for(cells):
            res = {"cells": {}}
            num = den = 0.0
            cell_aucs = []
            for c in cells:
                rr = [r for r in by[c] if subset(r) and (positive(r) or negative(r))]
                yf = np.array([positive(r) for r in rr])
                s = sig_values(rr, sig)
                a = auroc(s, yf)
                d = {"n": len(rr), "n_pos": int(yf.sum()), "auc": a}
                if a is not None:
                    n1, n0 = int(yf.sum()), len(rr) - int(yf.sum())
                    num += a * n1 * n0
                    den += n1 * n0
                    cell_aucs.append(a)
                if do_boot and len(rr):
                    aucs, ok = boot.auc(rr, s, yf)
                    d["ci"] = ci(aucs[ok]) if ok.any() else None
                    d["dropped"] = int((~ok).sum())
                    d["_draws"] = aucs
                res["cells"][c] = d
            rr = [r for c in cells for r in by[c] if subset(r) and (positive(r) or negative(r))]
            yf = np.array([positive(r) for r in rr])
            s = sig_values(rr, sig)
            p = {"n": len(rr), "n_pos": int(yf.sum()), "auc": auroc(s, yf),
                 "stratified": (num / den) if den else None,
                 "mean_cells": float(np.mean(cell_aucs)) if cell_aucs else None}
            if do_boot:
                aucs, ok = boot.auc(rr, s, yf)
                p["ci"] = ci(aucs[ok]) if ok.any() else None
                p["dropped"] = int((~ok).sum())
                p["_draws"] = aucs
            res[pooled_name] = p
            out[sig] = res
        return out

    def strip_draws(block):
        return {s: {k: ({c: {kk: vv for kk, vv in d.items() if kk != "_draws"} for c, d in v.items()}
                        if k == "cells" else {kk: vv for kk, vv in v.items() if kk != "_draws"})
                    for k, v in res.items()} for s, res in block.items()}

    def fmt_block(block, cells, pooled_name, title, boot_shown=True):
        L = [title]
        hdr = f"  {'signal':34s}" + "".join(f"{c:>26s}" for c in cells) + f"{pooled_name:>28s}{'strat.':>8s}{'mean':>7s}  cost"
        L.append(hdr)
        for sig, res in block.items():
            parts = []
            for c in cells:
                d = res["cells"][c]
                if boot_shown and d.get("ci"):
                    parts.append(f"{f3(d['auc'])} [{f3(d['ci'][0])},{f3(d['ci'][1])}] d{d['dropped']}")
                else:
                    parts.append(f"{f3(d['auc'])}")
            p = res[pooled_name]
            pp = (f"{f3(p['auc'])} [{f3(p['ci'][0])},{f3(p['ci'][1])}] d{p['dropped']}" if boot_shown and p.get("ci")
                  else f3(p["auc"]))
            L.append(f"  {SIGNALS[sig][0]:34s}" + "".join(f"{x:>26s}" for x in parts) + f"{pp:>28s}"
                     f"{f3(p['stratified']):>8s}{f3(p['mean_cells']):>7s}  {SIGNALS[sig][1]}")
        ns = "  n (positives): " + ", ".join(
            f"{c} {block[next(iter(block))]['cells'][c]['n']} ({block[next(iter(block))]['cells'][c]['n_pos']})"
            for c in cells) + f", {pooled_name} {block[next(iter(block))][pooled_name]['n']} " \
            f"({block[next(iter(block))][pooled_name]['n_pos']})"
        L.append(ns)
        return L

    def sel_block(cells, pooled_name):
        out = {}
        for sig in signals_for(cells):
            res = {}
            for name, rr in [(c, by[c]) for c in cells] + [(pooled_name, [r for c in cells for r in by[c]])]:
                s = sig_values(rr, sig)
                ys = np.array([r["success"] for r in rr])
                base = float(ys.mean())
                w80, a80 = selective(s, ys.tolist(), Fraction(4, 5))
                _, a50 = selective(s, ys.tolist(), Fraction(1, 2))
                fail = ~ys
                wt = np.array([not r["target_correct"] for r in rr])
                caught = float(((1 - w80) * fail).sum())
                nf = int(fail.sum())
                res[name] = dict(n=len(rr), base=base, acc80=a80, d80=a80 - base, acc50=a50, d50=a50 - base,
                                 failures=nf, caught=caught, caught_share=(caught / nf) if nf else None,
                                 random=0.2 * nf, caught_wrong_target=float(((1 - w80) * fail * wt).sum()),
                                 caught_right_target=float(((1 - w80) * fail * ~wt).sum()),
                                 wrong_target_failures=int((fail & wt).sum()),
                                 right_target_failures=int((fail & ~wt).sum()))
            out[sig] = res
        return out

    def fmt_sel(block, names, title):
        L = [title, "  per cell: base rate | acc@80% (diff) | acc@50% (diff) | caught at 20% abstention of failures "
                    "(share; random 0.2 x failures) [wrong-target / right-target caught of wt / rt failures]"]
        for sig, res in block.items():
            L.append(f"  {SIGNALS[sig][0]}  [{SIGNALS[sig][1]}]")
            for nm in names:
                d = res[nm]
                L.append(f"    {nm:15s} n {d['n']:4d}  base {d['base']:.3f} | {d['acc80']:.3f} ({d['d80']:+.3f}) | "
                         f"{d['acc50']:.3f} ({d['d50']:+.3f}) | caught {d['caught']:.1f} of {d['failures']} "
                         f"({f3(d['caught_share'])}; random {d['random']:.1f}) [wt {d['caught_wrong_target']:.1f}/"
                         f"{d['wrong_target_failures']}, rt {d['caught_right_target']:.1f}/{d['right_target_failures']}]")
        return L

    # ---------------------------------------------------------------- T1
    T.append("Phase 3 confidence from perception disagreement: tables (pass 2, FROZEN.md sections 5-9)")
    T.append(f"FROZEN.md sha256 {fz['FROZEN.md']}" + (f"; {', '.join(f'{k} {v}' for k, v in fz.items() if k != 'FROZEN.md')}"
                                                    if len(fz) > 1 else ""))
    T.append("Failure is the positive class; higher signal = less confident; AUROC ties count half; +inf ranks top.")
    T.append("Intervals: layout-cluster bootstrap, 2,000 paired draws, 95% percentile; dN = draws dropped (no failure "
             "or no success in the resample).")
    T.append("")
    T.append("T0. Checks")
    for line in head:
        T.append("  " + line)
    T.append(f"  R8 AUROC code check: {'PASS' if r8_ok else 'FAIL'} (tolerance +-{TOL_R8} inclusive, compared exactly "
             "(DEVIATIONS_2.md); AUROC for failure of 1 - c equals phase 1d's AUROC for success of c; the first attempt "
             "stopped on watch_qwen 0.3125 vs 0.312, a float artefact at the boundary, see pass2.log)")
    T.extend("  " + l for l in r8_lines)
    T.append("")
    T.append("T1. Records per cell")
    T.append(f"  {'cell':16s} {'n':>5s} {'succ':>5s} {'fail':>5s} {'wrongT':>7s} {'base':>6s}")
    groups_t1 = [(c, by[c]) for c in CELLS] + [("pooled primary", [r for c in PRIMARY for r in by[c]]),
                                              ("pooled watching", [r for c in WATCH for r in by[c]]),
                                              ("pooled s23", [r for c in S23 for r in by[c]]),
                                              ("pooled I14 549", [r for c in PRIMARY + WATCH for r in by[c]])]
    for nm, rr in groups_t1:
        n = len(rr)
        s = sum(r["success"] for r in rr)
        wt = sum(not r["target_correct"] for r in rr)
        assert all(not r["success"] for r in rr if not r["target_correct"])
        T.append(f"  {nm:16s} {n:5d} {s:5d} {n - s:5d} {wt:7d} {s / n:6.3f}")
        summary["T1"][nm] = dict(n=n, successes=s, failures=n - s, wrong_target=wt, base=s / n)
    T.append("")
    T.append("Deployment cost (FROZEN.md section 4): P1, P1b = detector (I09 A1, ~56 ms on the 4060) + one extra solver "
             "run (~0.25 s CPU); S2, S3 = detector only; S4 = TWO EXTRA FULL PIPELINE RUNS per goal step (extra model "
             "calls); S5 = free; S6 = all of the above; S6z = detector + one solver run.")
    T.append("")

    # ---------------------------------------------------------------- T2 primary
    t2 = auc_block(PRIMARY, "pooled", boots["primary"])
    T.extend(fmt_block(t2, PRIMARY, "pooled", "T2. AUROC for failure, primary cells (old wording, realistic); pooled = "
                                             "one ranking over the 441 records (judged); strat. = stratified pooled; "
                                             "mean = mean of the 3 cells"))
    d_p1 = t2["P1"]["pooled"]["_draws"] - t2["S5a"]["pooled"]["_draws"]
    okd = ~np.isnan(d_p1)
    diff = dict(point=t2["P1"]["pooled"]["auc"] - t2["S5a"]["pooled"]["auc"], ci=ci(d_p1[okd]),
                draws=int(okd.sum()))
    T.append(f"  Paired difference P1 - S5a, pooled primary AUROC: {diff['point']:+.3f} [{diff['ci'][0]:+.3f}, "
             f"{diff['ci'][1]:+.3f}] over {diff['draws']} draws (descriptive)")
    summary["T2"] = strip_draws(t2)
    summary["T2"]["paired_P1_minus_S5a"] = diff
    T.append("")
    # 549 pooled, I14 signals (point values; compare phase 1d)
    T.append("  Pooled over all 549 I14 primary records (old + watching; one ranking; point values, descriptive):")
    rr549 = [r for c in PRIMARY + WATCH for r in by[c]]
    yf549 = np.array([not r["success"] for r in rr549])
    summary["T2"]["pooled549"] = {}
    for sig in I14_SIGNALS:
        a = auroc(sig_values(rr549, sig), yf549)
        summary["T2"]["pooled549"][sig] = a
        T.append(f"    {SIGNALS[sig][0]:34s} {f3(a)}  [{SIGNALS[sig][1]}]")
    T.append("")

    # ---------------------------------------------------------------- T3 right target
    t3 = auc_block(PRIMARY, "pooled", boots["primary"], subset=lambda r: r["target_correct"])
    T.extend(fmt_block(t3, PRIMARY, "pooled", "T3. AUROC for failure, right-target records only (primary)"))
    summary["T3"] = strip_draws(t3)
    T.append("")

    # ---------------------------------------------------------------- T4 wrong target
    t4 = auc_block(PRIMARY, "pooled", boots["primary"], positive=lambda r: not r["target_correct"],
                   negative=lambda r: r["success"], do_boot=False)
    T.extend(fmt_block(t4, PRIMARY, "pooled", "T4. Wrong-target AUROC (positives = wrong-target records, all failures; "
                                             "negatives = successes; right-target failures left out; point values)",
                       boot_shown=False))
    summary["T4"] = strip_draws(t4)
    T.append("")

    # ---------------------------------------------------------------- T5 selective
    t5 = sel_block(PRIMARY, "pooled primary")
    T.extend(fmt_sel(t5, list(PRIMARY) + ["pooled primary"], "T5. Selective accuracy (C's success rate among the 80% "
                                                             "and 50% most confident) and failures caught at 20% "
                                                             "abstention, primary"))
    summary["T5"] = t5
    T.append("")

    # ---------------------------------------------------------------- T6 secondary
    T.append("T6. Secondary cells (reported, not judged)")
    T.append("")
    t6 = {}
    for grp, cells, pname, boot in (("watch", WATCH, "pooled watching", boots["watch"]),
                                    ("s23", S23, "pooled s23", boots["s23"])):
        a = auc_block(cells, pname, boot)
        b = auc_block(cells, pname, boot, subset=lambda r: r["target_correct"])
        cw = auc_block(cells, pname, boot, positive=lambda r: not r["target_correct"],
                       negative=lambda r: r["success"], do_boot=False)
        d = sel_block(cells, pname)
        T.extend(fmt_block(a, cells, pname, f"T6.{grp} a. AUROC for failure (as T2)"))
        T.append("")
        T.extend(fmt_block(b, cells, pname, f"T6.{grp} b. Right-target records only (as T3)"))
        T.append("")
        if any(any(not r["target_correct"] for r in by[c]) for c in cells):
            T.extend(fmt_block(cw, cells, pname, f"T6.{grp} c. Wrong-target AUROC (as T4; point values)",
                               boot_shown=False))
            T.append("")
        T.extend(fmt_sel(d, list(cells) + [pname], f"T6.{grp} d. Selective accuracy and failures caught (as T5)"))
        T.append("")
        t6[grp] = dict(auc=strip_draws(a), right_target=strip_draws(b), wrong_target=strip_draws(cw), selective=d)
    summary["T6"] = t6

    # ---------------------------------------------------------------- T7 coefficients
    T.append("T7. S6 / S6z coefficients (standardised features; lambda 1; intercept unpenalised)")
    fnames = ["f1 ln(1+min(P1,10))", "f2 1{P1=inf}", "f3 S2", "f4 S3", "f5 ln(1+min(S4,10))", "f6 1{S4=inf}"]
    for key, info in s6info.items():
        k = len(info["full"]["w"])
        folds = info["folds"]
        W = np.array([f["w"] for f in folds])
        B = np.array([f["b"] for f in folds])
        nconv = sum(f["converged"] for f in folds)
        T.append(f"  {key}: full-data fit b {info['full']['b']:+.4f} (iters {info['full']['iters']}, converged "
                 f"{info['full']['converged']}); folds {len(folds)}, converged {nconv}, max iters "
                 f"{max(f['iters'] for f in folds)}; fold b mean {B.mean():+.4f} range [{B.min():+.4f}, {B.max():+.4f}]")
        for j in range(k):
            T.append(f"    {fnames[j]:22s} full {info['full']['w'][j]:+.4f} (active {info['full']['active'][j]}, "
                     f"mean {info['full']['mu'][j]:.4f}, sd {info['full']['sd'][j]:.4f}); folds mean {W[:, j].mean():+.4f} "
                     f"range [{W[:, j].min():+.4f}, {W[:, j].max():+.4f}], active in "
                     f"{sum(f['active'][j] for f in folds)} of {len(folds)}")
        summary["T7"][key] = dict(full=info["full"], folds=folds, n_converged=nconv)
    T.append("")

    # ---------------------------------------------------------------- T8 descriptive counts
    T.append("T8. Descriptive counts")
    T.append(f"  {'cell':15s} {'P1=0':>5s} {'P1>0':>5s} {'P1inf':>6s} {'P1=0&dHead>1deg':>16s} {'P1b inf':>8s} "
             f"{'S4inf miss':>11s} {'S4inf tgt':>10s} {'S3=0':>5s} {'names none':>11s}")
    for c in CELLS:
        rr = by[c]
        real = c not in WATCH
        d = dict(n=len(rr),
                 P1_zero=sum(r["P1"] == 0 for r in rr) if real else None,
                 P1_pos=sum(r["P1"] is not None and 0 < r["P1"] < INF for r in rr) if real else None,
                 P1_inf=sum(r["P1"] == INF for r in rr) if real else None,
                 P1_zero_heading=sum(r["P1"] == 0 and r["D_heading_diff_deg"] is not None
                                     and r["D_heading_diff_deg"] > 1.0 for r in rr) if real else None,
                 P1b_inf=sum(r["P1b"] == INF for r in rr) if real else None,
                 S4_inf_missing=sum(r["S4_reason"] == "missing_place" for r in rr),
                 S4_inf_target=sum(r["S4_reason"] == "target_spread" for r in rr),
                 S3_zero=sum(r["S3"] == 0 for r in rr), names_none=sum(len(r["named"]) == 0 for r in rr))
        summary["T8"][c] = d
        g = lambda v: "—" if v is None else str(v)  # noqa: E731
        T.append(f"  {c:15s} {g(d['P1_zero']):>5s} {g(d['P1_pos']):>5s} {g(d['P1_inf']):>6s} "
                 f"{g(d['P1_zero_heading']):>16s} {g(d['P1b_inf']):>8s} {d['S4_inf_missing']:>11d} "
                 f"{d['S4_inf_target']:>10d} {d['S3_zero']:>5d} {d['names_none']:>11d}")
    T.append("  (S4 values are shared by the goal-step records of one trial group; counts are records, not groups.)")
    T.append("")

    # ---------------------------------------------------------------- T9 criteria and prediction
    p1 = t2["P1"]
    cell_p1 = {c: p1["cells"][c]["auc"] for c in PRIMARY}
    n_cells70 = sum(a >= 0.70 for a in cell_p1.values())
    pooled_p1 = p1["pooled"]["auc"]
    pooled_s5a = t2["S5a"]["pooled"]["auc"]
    succ = n_cells70 >= 2 and pooled_p1 > pooled_s5a
    kill = pooled_p1 < 0.60
    verdict = "kill" if kill else ("success" if succ else "neither")
    T.append("T9. Pre-registered criteria for P1 (441 primary records; point estimates) and the prediction")
    T.append(f"  P1 AUROC per primary cell: " + ", ".join(f"{c} {a:.3f}" for c, a in cell_p1.items())
             + f" -> {n_cells70} of 3 cells >= 0.70 (success needs >= 2)")
    T.append(f"  P1 pooled AUROC {pooled_p1:.3f} [{p1['pooled']['ci'][0]:.3f}, {p1['pooled']['ci'][1]:.3f}] against "
             f"S5a (flags) pooled {pooled_s5a:.3f} [{t2['S5a']['pooled']['ci'][0]:.3f}, "
             f"{t2['S5a']['pooled']['ci'][1]:.3f}]: strictly above {pooled_p1 > pooled_s5a}")
    T.append(f"  Success condition: {succ}; kill condition (pooled P1 < 0.60): {kill}")
    T.append(f"  VERDICT: {verdict.upper()}")
    T.append("")
    # prediction
    part1 = cell_p1["old_glm"] >= 0.70 and cell_p1["old_deepseek"] >= 0.70 and cell_p1["old_qwen"] < 0.60
    pooled_single = {s: t2[s]["pooled"]["auc"] for s in JUDGED_SINGLE}
    others = [v for s, v in pooled_single.items() if s != "S4"]
    part2 = pooled_single["S4"] > max(others)
    wt_pooled = {s: t4[s]["pooled"]["auc"] for s in ("P1", "S2", "S3")}
    part3 = all(v < 0.60 for v in wt_pooled.values())
    T.append("  Prediction (the , verbatim): \"P1 helps for GLM and DeepSeek, whose people answers carry phantoms "
             "that the detector does not confirm, and barely for Qwen, whose people answers are nearly perfect; S4 "
             "(trial disagreement) is the strongest single signal but costs extra calls; wrong-target failures stay "
             "invisible to P1-S3.\"")
    T.append(f"  Part 1 (P1 >= 0.70 in old GLM and old DeepSeek, < 0.60 in old Qwen): old GLM {cell_p1['old_glm']:.3f}, "
             f"old DeepSeek {cell_p1['old_deepseek']:.3f}, old Qwen {cell_p1['old_qwen']:.3f} (9 failures) -> "
             f"{'holds' if part1 else 'does not hold'}")
    T.append("  Part 2 (S4 strictly highest pooled primary AUROC among P1, S2, S3, S4, S5a, S5b): "
             + ", ".join(f"{s} {v:.3f}" for s, v in pooled_single.items()) + f" -> {'holds' if part2 else 'does not hold'}")
    T.append("  Part 3 (pooled primary wrong-target AUROC < 0.60 for each of P1, S2, S3): "
             + ", ".join(f"{s} {v:.3f}" for s, v in wt_pooled.items()) + f" -> {'holds' if part3 else 'does not hold'}")
    summary["T9"] = dict(P1_cells=cell_p1, n_cells_ge_070=n_cells70, P1_pooled=pooled_p1, S5a_pooled=pooled_s5a,
                         P1_pooled_ci=p1["pooled"]["ci"], S5a_pooled_ci=t2["S5a"]["pooled"]["ci"],
                         success_condition=succ, kill_condition=kill, verdict=verdict,
                         prediction=dict(part1=part1, part2=part2, part3=part3, pooled_single=pooled_single,
                                         wrong_target_pooled=wt_pooled))
    (OUT / "tables.txt").write_text("\n".join(T) + "\n")
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log(f"VERDICT {verdict}; P1 cells {cell_p1}, pooled {pooled_p1:.4f} vs S5a {pooled_s5a:.4f}; prediction parts "
        f"{part1}/{part2}/{part3}")
    log(f"wrote tables.txt, summary.json in {time.monotonic() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
