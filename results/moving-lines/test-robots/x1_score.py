"""Extra X1 (X1_FROZEN.md): CT3FT (CT3F with the true counter point) against CT3F on four models' cached first
answers, both re-run together in runs_x1/<model>; pooled over models. Zero calls; read-only.

    .venv/bin/python results/moving-lines/test-robots/x1_score.py [--test]
    (--test reads the earlier DeepSeek pair: CT3FT in runs_ceiling4/dsv41, CT3F in runs_confirm4/dsv41)

Measure: strict share from the first service at band top 1.2 m (round7/c_gaps.py's episode(), as ROUND7_D.md).
"""
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
sys.path.insert(0, str(paths.MOVING / "best-version-and-running-guess" / "round7"))
import json  # noqa: E402

from c_gaps import load  # noqa: E402
from summarise_fresh import sign_p  # noqa: E402

MODELS = {"DeepSeek V4.1": "dsv41", "Sonnet 5.5": "sonnet55_low", "Haiku 5.5": "claude_haiku55", "Gemini 3.8": "gemini_fast"}
TIE, BAR = 0.01, 0.05


def t95(df):
    z = 1.6448536269514722
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def summary(d):
    if len(d) < 3:
        return len(d), float("nan"), float("nan"), 0, 0, 1.0
    m = statistics.mean(d)
    w, lo = sum(x >= TIE for x in d), sum(x <= -TIE for x in d)
    return len(d), m, m - t95(len(d) - 1) * statistics.stdev(d) / math.sqrt(len(d)), w, lo, sign_p(w, lo)


def main():
    test = "--test" in sys.argv
    sel = json.loads((paths.MOVING / "test-layouts" / "confirm_selection.json").read_text())["layouts"]
    names = [l["name"] for l in sel]
    per_layout = {n: [] for n in names}
    print(f"# X1: CT3FT against CT3F, strict from the first service (band top 1.2 m){' TEST' if test else ''}\n")
    print("| model | layouts | mean d | 95% lower bound | front part | W/L/T | sign p |\n|---|---|---|---|---|---|---|")
    for name, d in MODELS.items():
        if test:
            if d != "dsv41":
                continue
            sc = {**load(sel, "runs_ceiling4/dsv41", ("CT3FT",)), **load(sel, "runs_confirm4/dsv41", ("CT3F",))}
        else:
            sc = load(sel, f"runs_x1/{d}", ("CT3FT", "CT3F"))
        rows = [(n, sc.get((n, "CT3FT")), sc.get((n, "CT3F"))) for n in names]
        rows = [(n, a[1.2], b[1.2]) for n, a, b in rows if a and b]
        if not rows:
            print(f"| {name} | not run | | | | | |")
            continue
        dd = [a["all"] - b["all"] for _, a, b in rows]
        fr = statistics.mean(a["front"] - b["front"] for _, a, b in rows)
        k, m, lb, w, lo, p = summary(dd)
        print(f"| {name} | {k} | {m:+.3f} | {lb:+.3f} | {fr:+.3f} | {w}/{lo}/{k - w - lo} | {p:.4f} |")
        for (n, _, _), v in zip(rows, dd):
            per_layout[n].append(v)
    pooled = [statistics.mean(v) for v in per_layout.values() if v]
    k, m, lb, w, lo, p = summary(pooled)
    complete = sum(1 for v in per_layout.values() if len(v) == (1 if test else len(MODELS)))
    print(f"\nPooled: per-layout mean over the models, {k} layouts ({complete} with every model): mean {m:+.3f}, "
          f"lower bound {lb:+.3f}, W/L/T {w}/{lo}/{k - w - lo}, sign p {p:.4f}")
    print(f"BAR X1-consistent (wins > losses and sign p <= 0.05): {w > lo and p <= 0.05}")
    print(f"BAR X1-size (pooled mean >= {BAR}, ROUND7_D's bar): {m >= BAR}; with the lower bound above 0: {lb > 0}")


if __name__ == "__main__":
    main()
