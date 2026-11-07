"""Independent verifier, pass 2: checks R1-R8, every measure of FROZEN.md section 5, the S6/S6z fits, the criteria
and the prediction reading, from verify/vrows_*.jsonl (own pass 1) plus I04 / I14 rows as data. Own code only: the
AUROC is computed by explicit pairwise comparison aggregated into layout x layout concordance matrices (the
bootstrap AUROC of a draw with layout multiplicities k is k'Ck / ((k.F)(k.S))), not by midranks.

    v_tables.py   -> verify/vtables.json and stdout
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONF = HERE.parent
RD = CONF.parents[1]
assert hashlib.sha256((CONF / "FROZEN.md").read_bytes()).hexdigest() == \
    "2a3a890a3b23ff466e8b72c63e6dbef9143d7738f68badc34a4fad6a17ccb4b3"
INF = float("inf")
PRIMARY = ("old_qwen", "old_glm", "old_deepseek")
WATCH = ("watch_qwen", "watch_glm", "watch_deepseek")
S23 = ("new_glm", "new_deepseek", "new_haiku", "new_sonnet", "new_gemini")
CELLS = PRIMARY + WATCH + S23
EXPECTED = {"old_qwen": (143, 134, 6), "old_glm": (141, 96, 14), "old_deepseek": (157, 109, 14),
            "watch_qwen": (36, 32, 0), "watch_glm": (36, 23, 2), "watch_deepseek": (36, 23, 0),
            "new_glm": (141, 86, 22), "new_deepseek": (151, 106, 15), "new_haiku": (146, 134, 1),
            "new_sonnet": (143, 134, 7), "new_gemini": (135, 109, 26)}
FLAGS = ("exclusion_broken", "far_from_target", "constraint_unusable")


def load():
    rows = []
    for g in ("oldwatch", "s23"):
        rows += [json.loads(line) for line in open(HERE / f"vrows_{g}.jsonl")]
    i04 = {}
    for p in sorted((RD / "phase1d" / "i04").glob("rows_*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            i04[r["key"]] = r
    i14 = {}
    for name in ("rows_oldwatch.jsonl", "rows_new.jsonl"):
        for line in open(RD / "phase1" / "i14_v0" / name):
            r = json.loads(line)
            if r["stage"] == "s21":
                continue
            i14[f"{r['run']}/{r['scene']}/{r['ep']}/{r['trial']}"] = r
    return rows, i04, i14


def signals(r, i04row):
    s = {}
    if "P1_raw" in r:
        s["P1"] = round(r["P1_raw"], 6)
        s["P1_noround"] = r["P1_raw"]
        s["P1_mm"] = round(r["P1_raw"], 3)
        s["P1_cm"] = round(r["P1_raw"], 2)
        s["P1_step0"] = round(r["P1_from_step0"], 6)
        s["P1b"] = INF if r["tnum_C"] != r["tnum_D"] else s["P1"]
    elif r["label"] in ("old", "new"):
        s["P1"] = s["P1b"] = INF            # D gives no place (never expected)
    nv, nd, m = r["S2_parts"]
    s["S2"] = float(Fraction(0) if nv + nd == 0 else 1 - Fraction(2 * m, nv + nd))
    u, n = r["S3_parts"]
    s["S3"] = float(Fraction(0) if n == 0 else Fraction(u, n))
    if r["S4_missing_place"]:
        s["S4"] = s["S4_place"] = INF
    else:
        sp = round(r["S4_pose_spread"], 6)
        s["S4_place"] = sp
        s["S4"] = INF if r["S4_target_spread"] > 1.0 else sp
    s["S5a"] = float(any(r["flags"].values()))
    s["S5b"] = 1.0 - i04row["V1"]["conc"]
    for f in FLAGS:
        s["flag_" + f] = float(r["flags"][f])
    return s


# ------------------------------------------------------------------------------------------------ AUROC machinery
def conc_matrix(sig, fail, lay, nL):
    """C[a, b] = sum over failures in layout a and successes in layout b of 1{s_f > s_s} + 0.5 * 1{s_f == s_s}."""
    sig = np.asarray(sig, dtype=float)
    fail = np.asarray(fail, dtype=bool)
    lay = np.asarray(lay)
    sf, ss = sig[fail], sig[~fail]
    cmp = (sf[:, None] > ss[None, :]).astype(float) + 0.5 * (sf[:, None] == ss[None, :])
    A = np.zeros((len(sf), nL))
    A[np.arange(len(sf)), lay[fail]] = 1
    B = np.zeros((len(ss), nL))
    B[np.arange(len(ss)), lay[~fail]] = 1
    C = A.T @ cmp @ B
    F = np.bincount(lay[fail], minlength=nL).astype(float)
    S = np.bincount(lay[~fail], minlength=nL).astype(float)
    return C, F, S


def auroc_point(C, F, S):
    d = F.sum() * S.sum()
    return None if d == 0 else C.sum() / d


def auroc_draws(C, F, S, K):
    """K: (draws, nL) multiplicities. Returns array with nan for dropped draws."""
    num = np.einsum("da,ab,db->d", K, C, K)
    den = (K @ F) * (K @ S)
    out = np.full(len(K), np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def brute_auroc(sig, fail):
    pos = [s for s, f in zip(sig, fail) if f]
    neg = [s for s, f in zip(sig, fail) if not f]
    if not pos or not neg:
        return None
    c = Fraction(0)
    for a in pos:
        for b in neg:
            c += 1 if a > b else Fraction(1, 2) if a == b else 0
    return float(c / (len(pos) * len(neg)))


def interval(arr):
    a = arr[~np.isnan(arr)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))], int(np.isnan(arr).sum())


# ------------------------------------------------------------------------------------------------ selective
def selective(sig, succ, q):
    n = len(sig)
    K = Fraction(q).limit_denominator(10) * n
    order = sorted(range(n), key=lambda i: sig[i])
    w = [Fraction(0)] * n
    i, c = 0, 0
    while i < n:
        j = i
        while j < n and sig[order[j]] == sig[order[i]]:
            j += 1
        g = j - i
        if c + g <= K:
            wt = Fraction(1)
        elif c >= K:
            wt = Fraction(0)
        else:
            wt = (K - c) / g
        for t in range(i, j):
            w[order[t]] = wt
        c += g
        i = j
    acc = sum(w[i] * succ[i] for i in range(n)) / K
    return float(acc), w


# ------------------------------------------------------------------------------------------------ S6
def fit_logistic(X, y, lam=1.0):
    n, p = X.shape
    Z = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(p + 1)
    pen = np.r_[0.0, np.full(p, lam)]
    for it in range(1, 101):
        z = Z @ beta
        pr = 1 / (1 + np.exp(-z))
        g = Z.T @ (pr - y) + pen * beta
        if np.max(np.abs(g)) < 1e-9:
            return beta, it - 1, True
        Hm = Z.T @ (Z * (pr * (1 - pr))[:, None]) + np.diag(pen)
        beta = beta - np.linalg.solve(Hm, g)
    z = Z @ beta
    pr = 1 / (1 + np.exp(-z))
    g = Z.T @ (pr - y) + pen * beta
    return beta, 100, bool(np.max(np.abs(g)) < 1e-9)


def feats(s, zero_call):
    def lg(x):
        return math.log(11.0) if x == INF else math.log(1 + min(x, 10.0))
    f = [lg(s["P1"]), float(s["P1"] == INF), s["S2"], s["S3"]]
    if not zero_call:
        f += [lg(s["S4"]), float(s["S4"] == INF)]
    return f


def s6(recs, layouts, zero_call):
    X = np.array([feats(r["sig"], zero_call) for r in recs])
    y = np.array([0.0 if r["success"] else 1.0 for r in recs])
    lay = np.array([r["layout"] for r in recs])
    prob = np.full(len(recs), np.nan)
    info = []
    for L in layouts:
        tr, te = lay != L, lay == L
        if te.sum() == 0:
            continue
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0)
        keep = sd >= 1e-12
        Xtr = (X[tr][:, keep] - mu[keep]) / sd[keep]
        Xte = (X[te][:, keep] - mu[keep]) / sd[keep]
        beta, its, conv = fit_logistic(Xtr, y[tr])
        prob[te] = 1 / (1 + np.exp(-(beta[0] + Xte @ beta[1:])))
        info.append(dict(layout=L, its=its, conv=conv, kept=keep.tolist()))
    mu, sd = X.mean(axis=0), X.std(axis=0)
    keep = sd >= 1e-12
    beta, its, conv = fit_logistic((X[:, keep] - mu[keep]) / sd[keep], y)
    return prob, info, dict(beta=beta.tolist(), its=its, conv=conv, kept=keep.tolist())


# ------------------------------------------------------------------------------------------------ main
def main():
    rows, i04, i14 = load()
    out = {"checks": {}}
    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)
    real = [r for r in rows if r["label"] in ("old", "new")]
    # ---------------- R1
    keys = [r["key"] for r in rows]
    r1 = (len(set(keys)) == len(keys) == 1265 and set(keys) == set(i14)
          and all(len(by_cell[c]) == EXPECTED[c][0] for c in CELLS))
    out["checks"]["R1"] = dict(ok=r1, n=len(keys), per_cell={c: len(by_cell[c]) for c in CELLS})
    # ---------------- R2
    bad2 = [r["key"] for r in rows if not (r["success"] == r["rescore_success"] == i14[r["key"]]["rec_success"]
                                           == i04[r["key"]]["top1"])]
    succ = {c: sum(r["success"] for r in by_cell[c]) for c in CELLS}
    out["checks"]["R2"] = dict(ok=not bad2 and all(succ[c] == EXPECTED[c][1] for c in CELLS), bad=bad2, succ=succ)
    # ---------------- R3
    bad3 = [r["key"] for r in rows if r["target_correct"] != i14[r["key"]]["target_ok"]]
    wt = {c: sum(not r["target_correct"] for r in by_cell[c]) for c in CELLS}
    wt_fail = all(not r["success"] for r in rows if not r["target_correct"])
    out["checks"]["R3"] = dict(ok=not bad3 and all(wt[c] == EXPECTED[c][2] for c in CELLS), bad=bad3, wt=wt,
                               wrong_target_all_failures=wt_fail)
    # ---------------- R4
    bad4 = [r["key"] for r in real if not (r.get("step0_moved_m") is not None and r["step0_moved_m"] <= 1e-6)]
    out["checks"]["R4"] = dict(ok=not bad4 and len(real) == 1157, n=len(real), bad=bad4,
                               max_moved=max(r.get("step0_moved_m") or 0 for r in real))
    # ---------------- R5
    bad5 = collections.defaultdict(list)
    gaps = []
    for r in real:
        a = r["a2"]
        if a is None or not a["solver_step"]:
            bad5["join"].append(r["key"])
            continue
        if "P1_raw" not in r:
            bad5["noplace"].append(r["key"])
            continue
        if r["D_success"] != a["D"]:
            bad5["success"].append(r["key"])
        gap = abs(round(r["P1_raw"], 6) - a["moved"])
        gaps.append(gap)
        if gap > 1e-4:
            bad5["moved"].append((r["key"], r["P1_raw"], a["moved"]))
        if r["tnum_C"] != a["tnum_C"] or r["tnum_D"] != a["tnum_D"]:
            bad5["tnum"].append(r["key"])
    out["checks"]["R5"] = dict(ok=not any(bad5.values()), bad={k: v[:10] for k, v in bad5.items()},
                               max_gap=max(gaps), D_succ=sum(bool(r.get("D_success")) for r in real),
                               A2_D_succ=sum(bool(r["a2"]["D"]) for r in real if r["a2"]),
                               D_build_same=sum(r["D_build_same"] for r in real),
                               match_same=sum(r["match_same"] for r in rows),
                               D_errors=[r["key"] for r in real if "D_error" in r])
    # ---------------- R6
    bad6 = [r["key"] for r in rows if any(bool(i04[r["key"]]["flags"][f]) != r["flags"][f] for f in FLAGS)]
    lay_bad = [r["key"] for r in rows if i04[r["key"]]["layout"] != r["layout"]]
    cell_bad = [r["key"] for r in rows if i04[r["key"]]["cell"] != r["cell"]]
    out["checks"]["R6"] = dict(ok=not bad6 and not cell_bad, flags_bad=bad6, layout_bad=lay_bad, cell_bad=cell_bad)

    for r in rows:
        r["sig"] = signals(r, i04[r["key"]])

    # layouts
    lay_primary = sorted({r["layout"] for c in PRIMARY for r in by_cell[c]})
    lay_s23 = sorted({r["layout"] for c in S23 for r in by_cell[c]})
    lay_watch = sorted({r["layout"] for c in WATCH for r in by_cell[c]})
    assert len(lay_primary) == 13 and len(lay_s23) == 11 and len(lay_watch) == 6, (lay_primary, lay_s23, lay_watch)
    rng = np.random.default_rng(20261018)
    D_primary = rng.integers(0, 13, size=(2000, 13))
    D_s23 = rng.integers(0, 11, size=(2000, 11))
    D_watch = rng.integers(0, 6, size=(2000, 6))

    def counts(D, nL):
        return np.stack([np.bincount(d, minlength=nL) for d in D]).astype(float)
    K = {"primary": counts(D_primary, 13), "s23": counts(D_s23, 11), "watch": counts(D_watch, 6)}
    LAY = {"primary": lay_primary, "s23": lay_s23, "watch": lay_watch}
    GROUP_CELLS = {"primary": PRIMARY, "s23": S23, "watch": WATCH}

    # S6 / S6z
    s6info = {}
    for grp in ("primary", "s23"):
        recs = [r for c in GROUP_CELLS[grp] for r in by_cell[c]]
        for name, zc in (("S6", False), ("S6z", True)):
            prob, info, full = s6(recs, LAY[grp], zc)
            for r, p in zip(recs, prob):
                r["sig"][name] = float(p)
            s6info[f"{grp}/{name}"] = dict(folds_converged=all(i["conv"] for i in info),
                                           max_its=max(i["its"] for i in info), full=full,
                                           const_features=[i["kept"] for i in info][0])
    out["s6"] = s6info

    # ---------------- R8
    r8 = {}
    exp15 = {"S5b": [0.559, 0.303, 0.540, 0.312, 0.579, 0.627],
             "flag_exclusion_broken": [0.500, 0.511, 0.531, 0.500, 0.500, 0.500],
             "flag_far_from_target": [0.500, 0.511, 0.500, 0.500, 0.538, 0.500],
             "flag_constraint_unusable": [0.478, 0.659, 0.496, 0.500, 0.555, 0.435]}
    pooled549 = {"S5b": 0.470, "flag_exclusion_broken": 0.515, "flag_far_from_target": 0.508,
                 "flag_constraint_unusable": 0.563}
    exp17 = {"new_glm": 0.318, "new_deepseek": 0.607, "new_haiku": 0.683, "new_sonnet": 0.898, "new_gemini": 0.532}
    r8ok = True
    for sname, vals in exp15.items():
        for c, v in zip(PRIMARY + WATCH, vals):
            got = brute_auroc([r["sig"][sname] for r in by_cell[c]], [not r["success"] for r in by_cell[c]])
            okv = Fraction(abs(Fraction(got) - Fraction(str(v)))) <= Fraction(5, 10000)
            r8[f"{sname}/{c}"] = (got, v, okv)
            r8ok &= okv
        pool = [r for c in PRIMARY + WATCH for r in by_cell[c]]
        got = brute_auroc([r["sig"][sname] for r in pool], [not r["success"] for r in pool])
        okv = abs(Fraction(got) - Fraction(str(pooled549[sname]))) <= Fraction(5, 10000)
        r8[f"{sname}/pooled549"] = (got, pooled549[sname], okv)
        r8ok &= okv
    for c, v in exp17.items():
        got = brute_auroc([r["sig"]["S5b"] for r in by_cell[c]], [not r["success"] for r in by_cell[c]])
        okv = abs(Fraction(got) - Fraction(str(v))) <= Fraction(5, 10000)
        r8[f"S5b/{c}"] = (got, v, okv)
        r8ok &= okv
    out["checks"]["R8"] = dict(ok=r8ok, values=r8)

    # ---------------- AUROC tables
    SIGS = ["P1", "P1b", "S2", "S3", "S4", "S4_place", "S5a", "S5b", "flag_exclusion_broken",
            "flag_far_from_target", "flag_constraint_unusable", "S6", "S6z", "P1_noround", "P1_mm", "P1_cm",
            "P1_step0"]

    def table(grp, subset=lambda r: True, label=lambda r: not r["success"], positives=None):
        cells = GROUP_CELLS[grp]
        L = {l: i for i, l in enumerate(LAY[grp])}
        res = {}
        for s in SIGS:
            recs_all = [r for c in cells for r in by_cell[c] if s in r["sig"] and subset(r)]
            if not recs_all:
                continue
            ent = {}
            mats = {}
            for c in cells:
                rc = [r for r in recs_all if r["cell"] == c]
                if not rc:
                    continue
                Cm, F, S = conc_matrix([r["sig"][s] for r in rc], [label(r) for r in rc],
                                       [L[r["layout"]] for r in rc], len(L))
                mats[c] = (Cm, F, S)
                pt = auroc_point(Cm, F, S)
                dr = auroc_draws(Cm, F, S, K[grp])
                iv, dropped = interval(dr) if pt is not None else (None, None)
                ent[c] = dict(auc=pt, ci=iv, dropped=dropped, n=len(rc), n1=int(F.sum()), n0=int(S.sum()))
            Cm, F, S = conc_matrix([r["sig"][s] for r in recs_all], [label(r) for r in recs_all],
                                   [L[r["layout"]] for r in recs_all], len(L))
            pt = auroc_point(Cm, F, S)
            dr = auroc_draws(Cm, F, S, K[grp])
            iv, dropped = interval(dr)
            strat_num = sum(m[0].sum() for m in mats.values())
            strat_den = sum(m[1].sum() * m[2].sum() for m in mats.values())
            cell_aucs = [e["auc"] for e in ent.values() if e["auc"] is not None]
            ent["pooled"] = dict(auc=pt, ci=iv, dropped=dropped, n=len(recs_all), n1=int(F.sum()), n0=int(S.sum()),
                                 stratified=strat_num / strat_den if strat_den else None,
                                 mean_cells=float(np.mean(cell_aucs)) if cell_aucs else None,
                                 _draws=dr)
            res[s] = ent
        return res

    T2 = {g: table(g) for g in ("primary", "s23", "watch")}
    T3 = {g: table(g, subset=lambda r: r["target_correct"]) for g in ("primary", "s23", "watch")}
    T4 = {g: table(g, subset=lambda r: (not r["target_correct"]) or r["success"]) for g in ("primary", "s23")}
    # paired difference P1 - S5a pooled primary
    d = T2["primary"]["P1"]["pooled"]["_draws"] - T2["primary"]["S5a"]["pooled"]["_draws"]
    out["paired_P1_minus_S5a"] = dict(point=T2["primary"]["P1"]["pooled"]["auc"] - T2["primary"]["S5a"]["pooled"]["auc"],
                                      ci=interval(d)[0])
    # brute force cross-check of pooled point values
    bf = {}
    pool = [r for c in PRIMARY for r in by_cell[c]]
    for s in ("P1", "S2", "S3", "S4", "S5a", "S5b", "S6", "S6z"):
        bf[s] = brute_auroc([r["sig"][s] for r in pool], [not r["success"] for r in pool])
    out["brute_pooled_primary"] = bf
    # 549 pool
    pool549 = [r for c in PRIMARY + WATCH for r in by_cell[c]]
    out["pooled549"] = {s: brute_auroc([r["sig"][s] for r in pool549], [not r["success"] for r in pool549])
                        for s in ("S2", "S3", "S4", "S5a", "S5b")}

    def strip(t):
        return {g: {s: {c: {k: v for k, v in e.items() if not k.startswith("_")} for c, e in ent.items()}
                    for s, ent in tt.items()} for g, tt in t.items()}
    out["T2"], out["T3"], out["T4"] = strip(T2), strip(T3), strip(T4)

    # ---------------- selective accuracy
    T5 = {}
    for grp in ("primary", "s23", "watch"):
        for s in ["P1", "P1b", "S2", "S3", "S4", "S4_place", "S5a", "S5b", "S6", "S6z"]:
            for c in list(GROUP_CELLS[grp]) + (["pooled"] if grp == "primary" else []):
                rc = [r for r in (by_cell[c] if c != "pooled" else [x for cc in PRIMARY for x in by_cell[cc]])
                      if s in r["sig"]]
                if not rc:
                    continue
                sig = [r["sig"][s] for r in rc]
                y = [1 if r["success"] else 0 for r in rc]
                a80, w80 = selective(sig, y, 0.8)
                a50, _ = selective(sig, y, 0.5)
                caught = sum((1 - w) * (1 - yy) for w, yy in zip(w80, y))
                c_wt = sum((1 - w) for w, r in zip(w80, rc) if not r["target_correct"])
                c_rt = sum((1 - w) * (1 - yy) for w, yy, r in zip(w80, y, rc) if r["target_correct"])
                nf = len(y) - sum(y)
                T5[f"{grp}/{s}/{c}"] = dict(base=sum(y) / len(y), acc80=a80, acc50=a50, caught=float(caught),
                                            caught_share=float(caught) / nf if nf else None, random=0.2 * nf,
                                            caught_wt=float(c_wt), caught_rt=float(c_rt))
    out["T5"] = T5

    # ---------------- descriptive
    T8 = {}
    for c in PRIMARY + S23:
        rc = by_cell[c]
        T8[c] = dict(P1_zero=sum(r["sig"]["P1"] == 0 for r in rc), P1_pos=sum(0 < r["sig"]["P1"] < INF for r in rc),
                     P1_inf=sum(r["sig"]["P1"] == INF for r in rc),
                     P1_zero_heading_gt1=sum(r["sig"]["P1"] == 0 and r["D_heading_diff_deg"] > 1 for r in rc),
                     P1b_inf=sum(r["sig"]["P1b"] == INF for r in rc))
    for c in CELLS:
        rc = by_cell[c]
        T8.setdefault(c, {}).update(
            S4_inf_missing=sum(r["S4_missing_place"] for r in rc),
            S4_inf_target=sum((not r["S4_missing_place"]) and r["S4_target_spread"] > 1.0 for r in rc),
            S3_zero=sum(r["sig"]["S3"] == 0 for r in rc), names_none=sum(r["S3_parts"][1] == 0 for r in rc),
            trial_group_sizes=dict(collections.Counter(len(r["S4_trials"]) for r in rc)))
    out["T8"] = T8

    # ---------------- criteria
    p1 = T2["primary"]["P1"]
    cells_ok = sum(p1[c]["auc"] >= 0.70 for c in PRIMARY)
    pooled = p1["pooled"]["auc"]
    s5a = T2["primary"]["S5a"]["pooled"]["auc"]
    success = cells_ok >= 2 and pooled > s5a
    kill = pooled < 0.60
    verdict = "kill" if kill else "success" if success else "neither"
    out["verdict"] = dict(cells=[p1[c]["auc"] for c in PRIMARY], cells_ok=cells_ok, pooled=pooled, s5a=s5a,
                          verdict=verdict)
    singles = {s: T2["primary"][s]["pooled"]["auc"] for s in ("P1", "S2", "S3", "S4", "S5a", "S5b")}
    part1 = p1["old_glm"]["auc"] >= 0.70 and p1["old_deepseek"]["auc"] >= 0.70 and p1["old_qwen"]["auc"] < 0.60
    part2 = all(singles["S4"] > v for k, v in singles.items() if k != "S4")
    part3 = all(T4["primary"][s]["pooled"]["auc"] < 0.60 for s in ("P1", "S2", "S3"))
    out["prediction"] = dict(part1=part1, part2=part2, part3=part3, singles=singles,
                             wt={s: T4["primary"][s]["pooled"]["auc"] for s in ("P1", "S2", "S3")})
    (HERE / "vtables.json").write_text(json.dumps(out, indent=1, default=str))

    # ---------------- print summary
    for k, v in out["checks"].items():
        print(k, "PASS" if v["ok"] else "FAIL", {kk: vv for kk, vv in v.items() if kk not in ("values",)} if k != "R8" else "")
    print("R8 values:", {k: (round(a, 6), b, ok) for k, (a, b, ok) in r8.items()})
    for grp in ("primary", "s23", "watch"):
        print(f"== T2 {grp}")
        for s, ent in T2[grp].items():
            print(f"  {s:26s}", "  ".join(
                f"{c}: {e['auc']:.4f}" + (f" [{e['ci'][0]:.3f},{e['ci'][1]:.3f}] d{e['dropped']}" if e['ci'] else "")
                if e['auc'] is not None else f"{c}: -" for c, e in ent.items()),
                f"strat {ent['pooled']['stratified']:.4f} mean {ent['pooled']['mean_cells']:.4f}")
    for name, T in (("T3 right target", T3), ("T4 wrong target", T4)):
        for grp, tt in T.items():
            print(f"== {name} {grp}")
            for s, ent in tt.items():
                print(f"  {s:26s}", "  ".join(
                    (f"{c}: {e['auc']:.4f}" + (f" [{e['ci'][0]:.3f},{e['ci'][1]:.3f}] d{e['dropped']}" if e['ci'] else "")
                     + f" n{e['n']}/{e['n1']}") if e['auc'] is not None else f"{c}: - n{e['n']}/{e['n1']}"
                    for c, e in ent.items()))
    print("paired", out["paired_P1_minus_S5a"])
    print("brute", bf)
    print("pooled549", out["pooled549"])
    print("== T5")
    for k, v in T5.items():
        if k.startswith("primary/") and k.endswith("/pooled"):
            print(f"  {k:30s} base {v['base']:.4f} acc80 {v['acc80']:.6f} acc50 {v['acc50']:.6f} caught {v['caught']:.2f} "
                  f"({v['caught_share']:.3f}; random {v['random']:.1f}) wt {v['caught_wt']:.2f} rt {v['caught_rt']:.2f}")
    print("== T8", json.dumps(T8))
    print("== S6", json.dumps(s6info))
    print("== verdict", out["verdict"], out["prediction"])


if __name__ == "__main__":
    main()
