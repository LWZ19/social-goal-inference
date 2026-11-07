"""Item 5 (I5_FROZEN.md): CTd (people tracked from the robot's own camera) against CT2 (the sim's true positions) on
the fresh headline cells: non-inferiority. Zero calls; read-only.

    .venv/bin/python results/moving-lines/own-camera/i5_score.py
"""
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import mcompare as C  # noqa: E402
from summarise_fresh import tracking  # noqa: E402

MARGIN = -0.05
# Both arms are re-run, cache-only, at the current code into runs_i5/<model> (I5_FROZEN.md); the earlier episodes in
# runs/<model> are read only for the repeat check. --test reads the earlier runs instead (old data, 30 cells).
POOLED = {"Sonnet 5.5": ("sonnet55_low", range(10, 20)), "DeepSeek V4.1": ("dsv41", range(10, 15)),
          "Haiku 5.5": ("claude_haiku55", range(10, 15)), "Gemini 3.8": ("gemini_fast", range(10, 15))}
REPORTED = {"Qwen 3.8": ("soclaas_qwen38_low", range(10, 15)), "GLM-5.3": ("soclaas_glm53", range(10, 15))}


def t95(df):
    z = 1.6448536269514722
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def lower_bound(ps, key):
    d = [x[key] - y[key] for _, x, y in ps]
    if len(d) < 3:
        return float("nan"), float("nan")
    m = statistics.mean(d)
    return m, m - t95(len(d) - 1) * statistics.stdev(d) / math.sqrt(len(d))


def main():
    test = "--test" in sys.argv
    base = "runs" if test else "runs_i5"
    print(f"# Item 5: CTd against CT2 (non-inferiority, margin -0.05 on strict share){' TEST on old data' if test else ''}\n")
    pooled, rows, eps_all, repeat = [], [], [], []
    for name, (d, seeds) in {**POOLED, **REPORTED}.items():
        eps = C.load(f"{base}/{d}", C.EVAL5, seeds, ("CTd", "CT2"))
        if not test:
            old = C.load(f"runs/{d}", C.EVAL5, seeds, ("CTd", "CT2"))
            for arm in ("CT2", "CTd"):
                both = {(*k[:2], arm + "_new"): e for k, e in eps.items() if k[2] == arm}
                both.update({(*k[:2], arm + "_old"): e for k, e in old.items() if k[2] == arm})
                ps = C.pairs(both, arm + "_new", arm + "_old")
                if ps:
                    dd = [x["share_strict"] - y["share_strict"] for _, x, y in ps]
                    repeat.append(f"- {name} {arm}, re-run vs earlier run, {len(ps)} cells: mean d {statistics.mean(dd):+.3f}, "
                                  f"mean |d| {statistics.mean(abs(v) for v in dd):.3f}")
        n = sum(1 for (_, _, a), e in eps.items() if a == "CTd" and e is not None)
        if n == 0:
            print(f"- {name}: no CTd episodes yet")
            continue
        for arm in ("CTd", "CT2"):
            k, s, f = C.means(eps, arm)
            print(f"- {name} {arm}: {k} episodes, strict {s:.3f}, primary {f:.3f}, vlm_error episodes "
                  f"{sum(1 for (_, _, a), e in eps.items() if a == arm and C.errors(e))}")
        ps = C.pairs(eps, "CTd", "CT2")
        rows.append((name, ps))
        if name in POOLED:
            pooled += ps
        eps_all += [e for (_, _, a), e in eps.items() if a == "CTd" and e is not None]
    rows.append(("pooled (Sonnet, DeepSeek, Haiku, Gemini)", pooled))
    print("\n| | pairs | strict mean d | strict 95% lower bound | primary mean d | primary 95% lower bound | strict W/L/T (sign p) |")
    print("|---|---|---|---|---|---|---|")
    for name, ps in rows:
        ms, lbs = lower_bound(ps, "share_strict")
        mf, lbf = lower_bound(ps, "share_frozen")
        r = C.row(name, ps)
        print(f"| {name} | {len(ps)} | {ms:+.3f} | {lbs:+.3f} | {mf:+.3f} | {lbf:+.3f} | "
              f"{'/'.join(map(str, r['strict']))} ({r['p_strict']:.3f}) |")
    ms, lbs = lower_bound(pooled, "share_strict")
    mf, lbf = lower_bound(pooled, "share_frozen")
    print(f"\nBAR H5 (pooled, strict: one-sided 95% lower bound of mean(CTd - CT2) > {MARGIN}): {lbs > MARGIN}"
          f"{'' if len(pooled) == 125 else f'  [pooled pairs {len(pooled)}, expected 125: check mechanics]'}")
    print(f"Same under the primary scoring (reported): {lbf > MARGIN}")
    if repeat:
        print("\n## Repeat check: the same arm, same cached answer, earlier run vs this run (timing only)\n")
        print("\n".join(repeat))
    tracking(eps_all)


if __name__ == "__main__":
    main()
