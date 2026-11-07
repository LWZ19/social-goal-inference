"""Extra X3 (X3_FROZEN.md): CTe (detector people for the first answer, plus detector tracking) against CTd (the model's
people call for the first answer, plus detector tracking), run together into runs_x3/<model>. Zero calls; read-only.

    .venv/bin/python fill/x3_cte/x3_score.py [--test]
    (--test: the earlier Sonnet cells, seeds 10-12, in runs/sonnet55_low)
"""
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mcompare as C  # noqa: E402
from summarise_fresh import sign_p  # noqa: E402

MARGIN = -0.05
MODELS = {"DeepSeek V4.1": ("runs_x3/dsv41", range(10, 15)), "Qwen 3.8": ("runs_x3/soclaas_qwen38_low", range(10, 15))}


def t95(df):
    z = 1.6448536269514722
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def first_answer_s(e):
    """Seconds from the request to the first answer's arrival in the sim (its latency as the sim applied it)."""
    a = (e or {}).get("answers") or []
    if not a or a[0].get("status") != "ok":
        return None
    return a[0].get("latency_s")


def main():
    test = "--test" in sys.argv
    groups = {"Sonnet (seeds 10-12)": ("runs/sonnet55_low", range(10, 13))} if test else MODELS
    print(f"# X3: CTe against CTd{' TEST' if test else ''}\n")
    pooled, verdicts = [], []
    for name, (root, seeds) in groups.items():
        eps = C.load(root, C.EVAL5, seeds, ("CTe", "CTd"))
        ps = C.pairs(eps, "CTe", "CTd")
        if not ps:
            print(f"- {name}: not run")
            continue
        d = [x["share_strict"] - y["share_strict"] for _, x, y in ps]
        m = statistics.mean(d)
        lb = m - t95(len(d) - 1) * statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 2 else float("nan")
        lat = [(first_answer_s(eps[(*k, "CTe")]), first_answer_s(eps[(*k, "CTd")])) for k, _, _ in ps]
        lat = [(a, b) for a, b in lat if a is not None and b is not None]
        faster, slower = sum(a < b for a, b in lat), sum(a > b for a, b in lat)
        r = C.row(name, ps)
        print(f"- {name}: {len(ps)} cells; strict CTe {statistics.mean(x['share_strict'] for _, x, _ in ps):.3f}, CTd "
              f"{statistics.mean(y['share_strict'] for _, _, y in ps):.3f}; mean d {m:+.3f}, lower bound {lb:+.3f}; "
              f"W/L/T {'/'.join(map(str, r['strict']))} (p {r['p_strict']:.3f})")
        if lat:
            print(f"  first answer: median CTe {statistics.median(a for a, _ in lat):.1f} s, CTd "
                  f"{statistics.median(b for _, b in lat):.1f} s; CTe earlier in {faster}, later in {slower} "
                  f"(sign p {sign_p(faster, slower):.4f})")
        verdicts.append(f"- {name}: BAR X3-speed (CTe's first answer earlier, sign p <= 0.05): "
                        f"{faster > slower and sign_p(faster, slower) <= 0.05}; per-model X3-quality (lower bound > "
                        f"{MARGIN}): {lb > MARGIN}")
        pooled += d
    if len(pooled) > 2:
        m = statistics.mean(pooled)
        lb = m - t95(len(pooled) - 1) * statistics.stdev(pooled) / math.sqrt(len(pooled))
        verdicts.append(f"- POOLED over {len(pooled)} cells: BAR X3-quality (CTe non-inferior to CTd, strict lower "
                        f"bound > {MARGIN}): {lb > MARGIN} (mean {m:+.3f}, bound {lb:+.3f})")
    print("\n" + "\n".join(verdicts))


if __name__ == "__main__":
    main()
