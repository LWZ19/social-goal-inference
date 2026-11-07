"""Independent verifier, POST HOC exposure follow-up (README section 8, question 6). Changes no verdict.

Recomputes, from the three primary row files only, AUROC for failure of P1, S2, S3, S4, S6z (frozen 441 LOLO fit and a
refit trained on the 404 records where D's verdict equals C's), S5a and model identity on
  (a) all 441 primary records, (b) the 404 with D verdict == C verdict, (c) the 37 flips,
with the frozen primary layout-cluster bootstrap, paired differences, shares above 0.5, selective accuracy at 80%,
right/wrong-target splits on (b) and the S6z coefficients.

Own code throughout: numpy only; imports nothing from code/, verify/ or followup/exposure.py. AUROC is brute-force
pairwise concordance (ties half), gathered into layout-by-layout matrices for the bootstrap. Reads only
rows_old_{qwen,glm,deepseek}.jsonl and (for the (a) reproduction check) summary.json. No model or network call.
"""
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONF = HERE.parent.parent  # phase3/confidence/
CELLS = ["old_qwen", "old_glm", "old_deepseek"]
SIGS = ["P1", "S2", "S3", "S4", "S6z_frozen", "S6z_refit", "S5a"]
SEED, DRAWS = 20261018, 2000


# ---------------------------------------------------------------- data
def load():
    rows = []
    for c in CELLS:
        with open(CONF / f"rows_{c}.jsonl") as fh:
            for line in fh:
                r = json.loads(line)
                assert r["cell"] == c
                rows.append(r)
    return rows


def sanity(rows):
    """Light consistency checks on the row fields used (the signals themselves were verified on 1,265 records)."""
    out = {}
    bad = 0
    for r in rows:
        f = Fraction(r["S2_frac"])
        nv, nd, m = r["n_v"], r["n_d"], r["n_match"]
        exp = Fraction(0) if nv + nd == 0 else 1 - Fraction(2 * m, nv + nd)
        bad += (f != exp) or (float(f) != r["S2"])
        bad += float(Fraction(r["S3_frac"])) != r["S3"]
        bad += int(any(r["flags"].values())) != r["S5a"]
        bad += r["D_success"] != r["a2_D"]
        bad += r["D_exception"] is not None
        bad += not math.isfinite(r["P1"])
    out["field_mismatches"] = bad
    grp = {}
    for r in rows:
        grp.setdefault((r["run"], r["scene"], r["episode"]), set()).add(r["S4"])
    out["S4_groups"] = len(grp)
    out["S4_groups_nonconstant"] = sum(len(v) > 1 for v in grp.values())
    return out


# ---------------------------------------------------------------- AUROC (brute force)
def conc_matrix(s_pos, s_neg):
    """pairwise concordance: 1 if failure scores higher, 0.5 tie (inf ties inf), 0 otherwise."""
    a = np.asarray(s_pos, float)[:, None]
    b = np.asarray(s_neg, float)[None, :]
    return (a > b).astype(float) + 0.5 * (a == b)


def auroc(scores, fail):
    scores = np.asarray(scores, float)
    fail = np.asarray(fail, bool)
    p, n = scores[fail], scores[~fail]
    if len(p) == 0 or len(n) == 0:
        return None
    return float(conc_matrix(p, n).sum() / (len(p) * len(n)))


def stratified(scores, fail, cells):
    num = den = 0.0
    scores, fail, cells = np.asarray(scores, float), np.asarray(fail, bool), np.asarray(cells)
    for c in CELLS:
        m = cells == c
        p, n = scores[m & fail], scores[m & ~fail]
        if len(p) and len(n):
            num += conc_matrix(p, n).sum()
            den += len(p) * len(n)
    return num / den if den else None


# ---------------------------------------------------------------- bootstrap
def draw_matrix(n_layouts):
    rng = np.random.default_rng(SEED)
    return rng.integers(0, n_layouts, size=(DRAWS, n_layouts))  # first matrix = primary


def boot_aucs(scores, fail, lay_idx, counts):
    """weighted AUROC per draw: records repeated by layout multiplicity. Returns array with nan where undefined."""
    scores, fail, lay_idx = np.asarray(scores, float), np.asarray(fail, bool), np.asarray(lay_idx)
    L = counts.shape[1]
    P = np.zeros(L)
    N = np.zeros(L)
    C = np.zeros((L, L))
    for a in range(L):
        pa = scores[(lay_idx == a) & fail]
        P[a] = len(pa)
        N[a] = ((lay_idx == a) & ~fail).sum()
        if len(pa) == 0:
            continue
        for b in range(L):
            nb = scores[(lay_idx == b) & ~fail]
            if len(nb):
                C[a, b] = conc_matrix(pa, nb).sum()
    w = counts.astype(float)
    np_ = w @ P
    nn_ = w @ N
    num = np.einsum("da,ab,db->d", w, C, w)
    out = np.full(len(w), np.nan)
    ok = (np_ > 0) & (nn_ > 0)
    out[ok] = num[ok] / (np_[ok] * nn_[ok])
    return out


def pct(x):
    x = x[~np.isnan(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


# ---------------------------------------------------------------- selective accuracy (frozen rule, exact fractions)
def selective(scores, succ, q=Fraction(4, 5)):
    scores = list(map(float, scores))
    succ = list(map(int, succ))
    n = len(scores)
    K = q * n
    order = sorted(range(n), key=lambda i: scores[i])
    w = [Fraction(0)] * n
    i = 0
    below = 0
    while i < n:
        j = i
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        g = j - i
        wt = min(max((K - below) / g, Fraction(0)), Fraction(1))
        for k in range(i, j):
            w[order[k]] = wt
        below += g
        i = j
    assert sum(w) == K
    acc = sum(wi * yi for wi, yi in zip(w, succ)) / K
    caught = sum((1 - wi) * (1 - yi) for wi, yi in zip(w, succ))
    return float(acc), float(caught), float(Fraction(sum(succ), n))


# ---------------------------------------------------------------- S6z: own penalised logistic (Newton), LOLO
def features(rows):
    X = []
    for r in rows:
        p1 = r["P1"]
        f1 = math.log(11.0) if math.isinf(p1) else math.log1p(min(p1, 10.0))
        X.append([f1, 1.0 if math.isinf(p1) else 0.0, r["S2"], r["S3"]])
    return np.array(X, float)


def fit(X, y, lam=1.0):
    mu = X.mean(0)
    sd = X.std(0)  # population SD
    act = sd >= 1e-12
    Z = np.zeros_like(X)
    Z[:, act] = (X[:, act] - mu[act]) / sd[act]
    Za = np.hstack([np.ones((len(X), 1)), Z[:, act]])
    beta = np.zeros(Za.shape[1])
    pen = np.full(Za.shape[1], lam)
    pen[0] = 0.0
    it = 0
    conv = False
    for it in range(1, 101):
        z = Za @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        g = Za.T @ (p - y) + pen * beta
        if np.max(np.abs(g)) < 1e-9:
            conv = True
            it -= 1
            break
        H = (Za * (p * (1 - p))[:, None]).T @ Za + np.diag(pen)
        beta = beta - np.linalg.solve(H, g)
    else:
        z = Za @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        g = Za.T @ (p - y) + pen * beta
        conv = bool(np.max(np.abs(g)) < 1e-9)
    w = np.zeros(X.shape[1])
    w[act] = beta[1:]
    return dict(b=float(beta[0]), w=w, mu=mu, sd=sd, act=act, iters=it, conv=conv)


def predict(m, X):
    Z = np.zeros_like(X)
    Z[:, m["act"]] = (X[:, m["act"]] - m["mu"][m["act"]]) / m["sd"][m["act"]]
    return 1.0 / (1.0 + np.exp(-(m["b"] + Z @ m["w"])))


def lolo(rows, X, y, train_mask, layouts):
    lay = np.array([r["layout"] for r in rows])
    out = np.full(len(rows), np.nan)
    info = []
    for L in layouts:
        tr = (lay != L) & train_mask
        te = lay == L
        m = fit(X[tr], y[tr])
        out[te] = predict(m, X[te])
        info.append(dict(layout=L, n_train=int(tr.sum()), n_test=int(te.sum()), iters=m["iters"], conv=m["conv"]))
    assert not np.isnan(out).any()
    return out, info


# ---------------------------------------------------------------- main
def main():
    rows = load()
    n = len(rows)
    assert n == 441
    san = sanity(rows)
    layouts = sorted({r["layout"] for r in rows})
    assert len(layouts) == 13
    lix = {L: i for i, L in enumerate(layouts)}
    fail = np.array([not r["success"] for r in rows])
    succ = ~fail
    cell = np.array([r["cell"] for r in rows])
    lay_idx = np.array([lix[r["layout"]] for r in rows])
    tc = np.array([r["target_correct"] for r in rows])
    flip = np.array([r["D_success"] != r["success"] for r in rows])
    inB = ~flip
    X = features(rows)
    y = fail.astype(float)

    s6z_frozen, info_f = lolo(rows, X, y, np.ones(n, bool), layouts)
    s6z_refit, info_r = lolo(rows, X, y, inB, layouts)
    sig = {
        "P1": np.array([r["P1"] for r in rows], float),
        "S2": np.array([r["S2"] for r in rows], float),
        "S3": np.array([r["S3"] for r in rows], float),
        "S4": np.array([r["S4"] for r in rows], float),
        "S6z_frozen": s6z_frozen,
        "S6z_refit": s6z_refit,
        "S5a": np.array([r["S5a"] for r in rows], float),
    }
    full_f = fit(X, y)
    full_r = fit(X[inB], y[inB])

    counts_lay = draw_matrix(13)
    counts = np.zeros((DRAWS, 13), int)
    for d in range(DRAWS):
        for a in counts_lay[d]:
            counts[d, a] += 1

    subsets = {"a": np.ones(n, bool), "b": inB, "c": flip}
    res = {"sanity": san, "layouts": layouts, "counts": {}, "auc": {}, "boot": {}, "share_gt_half": {},
           "share_ge_half": {}, "paired": {}, "selective80": {}, "targetsplit_b": {}, "model_identity": {},
           "coef": {}, "folds": {"frozen": info_f, "refit": info_r}}

    # counts and composition
    for k, m in subsets.items():
        res["counts"][k] = {c: [int((m & (cell == c)).sum()), int((m & (cell == c) & fail).sum())] for c in CELLS}
        res["counts"][k]["pooled"] = [int(m.sum()), int((m & fail).sum())]
    res["counts"]["flip_fixes"] = int((flip & fail).sum())
    res["counts"]["flip_breaks"] = int((flip & succ).sum())
    res["counts"]["flip_breaks_cells"] = [c for c in CELLS for _ in range(int((flip & succ & (cell == c)).sum()))]
    res["counts"]["flip_wrong_target"] = int((flip & ~tc).sum())
    res["counts"]["b_wrong_target_failures"] = int((inB & fail & ~tc).sum())
    res["counts"]["b_right_target_failures"] = int((inB & fail & tc).sum())
    res["counts"]["layouts_in_b"] = len({rows[i]["layout"] for i in np.where(inB)[0]})

    boot = {}
    for k, m in subsets.items():
        res["auc"][k] = {}
        for s in SIGS:
            v = sig[s]
            e = {c: auroc(v[m & (cell == c)], fail[m & (cell == c)]) for c in CELLS}
            e["pooled"] = auroc(v[m], fail[m])
            e["stratified"] = stratified(v[m], fail[m], cell[m])
            res["auc"][k][s] = e
        # model identity: each cell's failure rate within the subset
        mi = np.zeros(n)
        for c in CELLS:
            mc = m & (cell == c)
            mi[mc] = fail[mc].mean() if mc.sum() else 0.0
        res["model_identity"][k] = auroc(mi[m], fail[m])
        if k == "c":
            continue
        res["boot"][k] = {}
        res["share_gt_half"][k] = {}
        res["share_ge_half"][k] = {}
        for s in SIGS:
            v = sig[s]
            res["boot"][k][s] = {}
            res["share_gt_half"][k][s] = {}
            res["share_ge_half"][k][s] = {}
            for c in CELLS + ["pooled"]:
                mc = m if c == "pooled" else m & (cell == c)
                ba = boot_aucs(v[mc], fail[mc], lay_idx[mc], counts)
                boot[(k, s, c)] = ba
                ok = ~np.isnan(ba)
                res["boot"][k][s][c] = dict(ci=pct(ba), dropped=int((~ok).sum()))
                res["share_gt_half"][k][s][c] = float((ba[ok] > 0.5).mean())
                res["share_ge_half"][k][s][c] = float((ba[ok] >= 0.5).mean())

    # paired differences on (b), pooled
    pairs = [("S2", "P1"), ("S2", "S5a"), ("S3", "P1"), ("S6z_refit", "S2"), ("S6z_frozen", "P1"), ("S4", "S2"),
             ("P1", "S5a")]
    for k in ("a", "b"):
        res["paired"][k] = {}
        for a_, b_ in pairs:
            d = boot[(k, a_, "pooled")] - boot[(k, b_, "pooled")]
            ok = ~np.isnan(d)
            pt = res["auc"][k][a_]["pooled"] - res["auc"][k][b_]["pooled"]
            res["paired"][k][f"{a_}-{b_}"] = dict(point=pt, ci=pct(d), share_gt0=float((d[ok] > 0).mean()),
                                                 n=int(ok.sum()))

    # selective accuracy at 80% kept
    for k in ("a", "b"):
        m = subsets[k]
        res["selective80"][k] = {}
        for s in SIGS:
            e = {}
            for c in CELLS + ["pooled"]:
                mc = m if c == "pooled" else m & (cell == c)
                acc, caught, base = selective(sig[s][mc], succ[mc])
                e[c] = dict(acc=acc, caught=caught, base=base, failures=int((mc & fail).sum()),
                            random=0.2 * int((mc & fail).sum()))
            res["selective80"][k][s] = e

    # right-target and wrong-target failures against (b)'s successes
    for s in SIGS:
        e = {}
        for kind, tmask in (("rt", tc), ("wt", ~tc)):
            for c in CELLS + ["pooled"]:
                base_m = inB if c == "pooled" else inB & (cell == c)
                pos = base_m & fail & tmask
                neg = base_m & succ
                sel = pos | neg
                e[f"{kind}/{c}"] = auroc(sig[s][sel], fail[sel])
                e[f"{kind}/{c}/n_pos"] = int(pos.sum())
        res["targetsplit_b"][s] = e

    for name, m in (("frozen_441", full_f), ("refit_404", full_r)):
        res["coef"][name] = dict(b=m["b"], w=m["w"].tolist(), iters=m["iters"], conv=m["conv"])

    # (a) reproduction against summary.json T2
    summ = json.load(open(CONF / "summary.json"))
    T2 = summ["T2"]
    keymap = {"P1": "P1", "S2": "S2", "S3": "S3", "S4": "S4", "S6z_frozen": "S6z", "S5a": "S5a"}
    worst = 0.0
    nchk = 0
    for s, t in keymap.items():
        for c in CELLS:
            ref = T2[t]["cells"][c]
            worst = max(worst, abs(ref["auc"] - res["auc"]["a"][s][c]))
            for u, v in zip(ref["ci"], res["boot"]["a"][s][c]["ci"]):
                worst = max(worst, abs(u - v))
            assert ref["dropped"] == res["boot"]["a"][s][c]["dropped"]
            nchk += 3
        ref = T2[t]["pooled"]
        worst = max(worst, abs(ref["auc"] - res["auc"]["a"][s]["pooled"]),
                    abs(ref["stratified"] - res["auc"]["a"][s]["stratified"]))
        for u, v in zip(ref["ci"], res["boot"]["a"][s]["pooled"]["ci"]):
            worst = max(worst, abs(u - v))
        nchk += 4
    res["repro_a_T2"] = dict(values=nchk, max_abs_diff=worst)
    # T5 acc80 / caught on (a)
    T5 = summ["T5"]
    w5 = 0.0
    for s, t in keymap.items():
        for c in CELLS + ["pooled"]:
            ref = T5[t][c if c != "pooled" else "pooled primary"]
            got = res["selective80"]["a"][s][c]
            w5 = max(w5, abs(ref["acc80"] - got["acc"]), abs(ref["caught"] - got["caught"]))
    res["repro_a_T5"] = dict(max_abs_diff=w5)
    # T7 full-fit coefficients
    T7 = summ["T7"]["primary/S6z"]["full"]
    res["repro_T7"] = dict(max_abs_diff=max(abs(T7["b"] - full_f["b"]),
                                            *[abs(u - v) for u, v in zip(T7["w"], full_f["w"])]))
    # model identity on (a) published 0.658
    res["model_identity_a_vs_published"] = res["model_identity"]["a"]

    json.dump(res, open(HERE / "v_followup.json", "w"), indent=1, default=float)
    report(res)


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def report(res):
    p = print
    p("Independent verifier, POST HOC exposure follow-up (changes no verdict). Own code; numpy only.")
    p(f"sanity: {res['sanity']}")
    p(f"(a) reproduction vs summary.json T2: {res['repro_a_T2']}; T5 acc80/caught: {res['repro_a_T5']}; "
      f"T7 S6z full fit: {res['repro_T7']}")
    p(f"counts: {json.dumps(res['counts'])}")
    for k, title in (("a", "(a) all 441"), ("b", "(b) D verdict == C (404)"), ("c", "(c) 37 flips")):
        p(f"\nAUROC {title}   [cells qwen / glm / deepseek / pooled {{stratified}}]")
        for s in SIGS:
            e = res["auc"][k][s]
            parts = []
            for c in CELLS + ["pooled"]:
                t = f3(e[c])
                if k in res["boot"]:
                    b = res["boot"][k][s][c]
                    t += f" [{b['ci'][0]:.3f}, {b['ci'][1]:.3f}] d{b['dropped']}"
                parts.append(t)
            p(f"  {s:11s} " + " / ".join(parts) + f"  {{{f3(e['stratified'])}}}")
        p(f"  model identity pooled: {f3(res['model_identity'][k])}")
    p("\nShare of draws with AUROC > 0.5 (and >= 0.5), (b)")
    for s in SIGS:
        g = res["share_gt_half"]["b"][s]
        h = res["share_ge_half"]["b"][s]
        p(f"  {s:11s} " + "  ".join(f"{c}:{g[c]:.4f}({h[c]:.4f})" for c in CELLS + ["pooled"]))
    for k in ("a", "b"):
        p(f"\nPaired pooled differences ({k})")
        for name, e in res["paired"][k].items():
            p(f"  {name:22s} {e['point']:+.4f} [{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}] share>0 {e['share_gt0']:.4f} "
              f"(n {e['n']})")
    for k in ("a", "b"):
        p(f"\nSelective accuracy at 80% kept ({k}): acc | caught (base, random)")
        for s in SIGS:
            e = res["selective80"][k][s]
            p(f"  {s:11s} " + "  ".join(
                f"{c}: {e[c]['acc']:.4f} | {e[c]['caught']:.2f}" for c in CELLS + ["pooled"]))
        e = res["selective80"][k]["P1"]
        p("  base/random " + "  ".join(f"{c}: {e[c]['base']:.4f} | {e[c]['random']:.1f}/{e[c]['failures']}"
                                      for c in CELLS + ["pooled"]))
    p("\nTarget split on (b): rt failures vs (b) successes | wt failures vs (b) successes")
    for s in SIGS:
        e = res["targetsplit_b"][s]
        p(f"  {s:11s} rt " + " / ".join(f3(e[f'rt/{c}']) for c in CELLS + ["pooled"]) +
          "  | wt " + " / ".join(f3(e[f'wt/{c}']) for c in CELLS + ["pooled"]))
    e = res["targetsplit_b"]["P1"]
    p("  n_pos rt " + "/".join(str(e[f'rt/{c}/n_pos']) for c in CELLS + ["pooled"]) +
      "  wt " + "/".join(str(e[f'wt/{c}/n_pos']) for c in CELLS + ["pooled"]))
    p("\nS6z full-data coefficients (standardised; b, f1 ln(1+min(P1,10)), f2 1{P1=inf}, f3 S2, f4 S3)")
    for name, e in res["coef"].items():
        p(f"  {name}: b {e['b']:+.4f} w " + " ".join(f"{x:+.4f}" for x in e["w"]) + f" iters {e['iters']} conv {e['conv']}")
    for name, info in res["folds"].items():
        p(f"  LOLO {name}: n_train {min(i['n_train'] for i in info)}-{max(i['n_train'] for i in info)}, "
          f"max iters {max(i['iters'] for i in info)}, all converged {all(i['conv'] for i in info)}")


if __name__ == "__main__":
    sys.exit(main())
