"""Scores CONFIRM_CT4PJ.md: H1, then H2 only if H1 passes, plus the reported measures. Zero calls; read-only.

    .venv/bin/python round4/o_confirm_score.py RUN_ROOT SELECTION_JSON LABEL
    (from best-version-and-running-guess/; RUN_ROOT relative to results/data/moving/, e.g. runs_confirm4/dsv41;
    SELECTION_JSON relative to results/moving-lines/, e.g. test-layouts/confirm_selection.json.)
A dry run on the pilot (runs_pilot3/dsv41 test-layouts/pilot_selection.json) tests the code; it is not a result.
"""
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round3"))
from i_followups import MOVING, first_service_t  # noqa: E402
from score_sweep import episode_scores  # noqa: E402
from summarise_fresh import measures, sign_p  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

ARMS = ("CT3F", "CT4", "CT4P", "CT4PJ")
DELTA = 0.05
TIE = 0.01


def t95(df):
    """One-sided 0.95 quantile of Student's t (Abramowitz and Stegun 26.7.5; error under 1e-4 for df >= 10)."""
    z = 1.6448536269514722
    v = float(df)
    return (z + (z ** 3 + z) / (4 * v) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * v ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * v ** 3))


def load(root, sel_path):
    sel = json.loads((paths.MOVING / sel_path).read_text())["layouts"]
    eps = {}
    for lay in sel:
        for arm in ARMS:
            hits = sorted((MOVING / root).glob(f"**/{lay['name']}/seed{lay['seed']}/{arm}/episode.json"))
            if hits:
                eps[(lay["name"], arm)] = json.loads(hits[0].read_text())
    return sel, eps


def strict(eps, name, arm):
    e = eps.get((name, arm))
    return None if e is None else episode_scores(e)["after_service_strict"]


def diffs(sel, eps, a, b):
    out = []
    for lay in sel:
        x, y = strict(eps, lay["name"], a), strict(eps, lay["name"], b)
        if x is not None and y is not None:
            out.append((lay, x - y))
    return out


def dropped(sel, eps, a, b):
    """Counts only (no names): layouts that drop out of the a-b comparison, and why."""
    k = {}
    for lay in sel:
        for arm in (a, b):
            if (lay["name"], arm) not in eps:
                k[f"no {arm} episode"] = k.get(f"no {arm} episode", 0) + 1
                break
            if strict(eps, lay["name"], arm) is None:
                k[f"no strict value for {arm}"] = k.get(f"no strict value for {arm}", 0) + 1
                break
    return k


def sign(ds):
    w = sum(1 for _, d in ds if d >= TIE)
    l = sum(1 for _, d in ds if d <= -TIE)
    return w, l, len(ds) - w - l, sign_p(w, l)


def lower_bound(ds):
    v = [d for _, d in ds]
    n = len(v)
    if n < 2:
        return float("nan"), float("nan"), n
    m, sd = statistics.mean(v), statistics.stdev(v)
    return m, m - t95(n - 1) * sd / math.sqrt(n), n


def stays(e):
    out, t_join = [], None
    for ev in e.get("joins", []):
        if ev["event"] == "joined":
            t_join = ev["t"]
        elif ev["event"] == "left" and t_join is not None:
            out.append(ev["t"] - t_join)
            t_join = None
    return out


def joined_share(e):
    t0 = first_service_t(e)
    if t0 is None:
        return None
    s = [x for x in e["samples"] if x["t"] >= t0 - 1e-9]
    return sum(x["joined"] for x in s) / len(s) if s else None


def main():
    root, sel_path, label = sys.argv[1], sys.argv[2], sys.argv[3]
    sel, eps = load(root, sel_path)
    print(f"# CONFIRM_CT4PJ.md scored: {label} ({root}, {sel_path})\n")
    print(f"- layouts {len(sel)}; episodes: " + ", ".join(f"{a} {sum(1 for k in eps if k[1] == a)}" for a in ARMS))
    print("- strict share in line from the first service, mean over layouts with a value: " +
          ", ".join(f"{a} {statistics.mean(v):.3f} (n {len(v)})" for a in ARMS
                    for v in [[x for x in (strict(eps, l['name'], a) for l in sel) if x is not None]] if v))

    print("\n## H1: CT4PJ > CT4P (sign test, ties |d| < 0.01, two-sided p <= 0.05 with wins > losses)\n")
    d1 = diffs(sel, eps, "CT4PJ", "CT4P")
    w, l, t, p = sign(d1)
    h1 = w > l and p <= 0.05
    print(f"- {w}/{l}/{t} (wins / losses / ties), p {p:.4f}, mean d {statistics.mean([d for _, d in d1]):+.3f}, "
          f"n {len(d1)} -> {'PASS' if h1 else 'FAIL'}")
    print(f"- dropped from H1: {dropped(sel, eps, 'CT4PJ', 'CT4P') or 'none'}")

    print(f"\n## H2: CT4PJ not worse than CT3F by more than {DELTA} (one-sided 95% paired t lower bound of mean d)\n")
    d2 = diffs(sel, eps, "CT4PJ", "CT3F")
    m, lb, n = lower_bound(d2)
    h2 = lb > -DELTA
    status = ("PASS" if h2 else "FAIL") if h1 else ("would " + ("pass" if h2 else "fail") +
                                                    "; NOT claimed, H1 failed")
    print(f"- mean d {m:+.3f}, lower bound {lb:+.3f}, n {n} -> {status}")
    print(f"- dropped from H2: {dropped(sel, eps, 'CT4PJ', 'CT3F') or 'none'}")
    print("- layouts where both arms are 0 enter with d = 0: "
          f"{sum(1 for lay, d in d2 if strict(eps, lay['name'], 'CT4PJ') == 0 and strict(eps, lay['name'], 'CT3F') == 0)}")

    print("\n## Reported, not claimed\n")
    w, l, t, p = sign(d2)
    print(f"- CT4PJ vs CT3F sign test: CT4PJ better {w}, worse {l}, tied {t}; p {p:.3f}")
    rng = np.random.default_rng(2000)
    v = np.array([d for _, d in d2])
    if len(v):
        boots = v[rng.integers(0, len(v), size=(10000, len(v)))].mean(axis=1)
        print(f"- H2 sensitivity, paired bootstrap of mean d (10,000, seed 2000): 5th percentile "
              f"{np.percentile(boots, 5):+.3f}")
    print("- by stratum (H1 sign test; H2 mean d and lower bound):")
    for s in sorted({x["stratum"] for x in sel}):
        a = [(lay, d) for lay, d in d1 if lay["stratum"] == s]
        b = [(lay, d) for lay, d in d2 if lay["stratum"] == s]
        w, l, t, p = sign(a)
        m, lb, n = lower_bound(b)
        print(f"  - {s}: H1 {w}/{l}/{t} p {p:.3f}; H2 mean d {m:+.3f}, lower bound {lb:+.3f} (n {n})")
    print("- per arm:")
    for a in ARMS:
        es = [eps[(x["name"], a)] for x in sel if (x["name"], a) in eps]
        if not es:
            continue
        rl = [s for s in (episode_scores(e)["right_last"] for e in es) if s is not None]
        cut = [measures(e)["cut"] for e in es]
        js = [s for s in (joined_share(e) for e in es) if s is not None]
        st = [x for e in es for x in stays(e)]
        print(f"  - {a}: right last {statistics.mean(rl):.3f}; cut-in samples {statistics.mean(cut):.2f}; joined share "
              f"after the first service {statistics.mean(js):.3f}; median time joined before leaving "
              f"{(statistics.median(st) if st else float('nan')):.1f} s (n {len(st)})")
    es = [eps[(x["name"], "CT4PJ")] for x in sel if (x["name"], "CT4PJ") in eps]
    ever = sum(1 for e in es if any(r.get("joined") for r in e["resolves"] if r.get("status") == "ok"))
    rj = [bool(r.get("joined")) for e in es for r in e["resolves"] if r.get("status") == "ok"]
    print(f"- CT4PJ's own join test: fired in {ever} of {len(es)} episodes; re-solves joined "
          f"{sum(rj) / max(len(rj), 1):.3f}")
    print("\nFixed-sequence verdict: " + ("H1 and H2 pass" if h1 and h2 else "H1 passes, H2 fails" if h1 else
                                          "H1 fails (H2 not claimed)"))


if __name__ == "__main__":
    main()
