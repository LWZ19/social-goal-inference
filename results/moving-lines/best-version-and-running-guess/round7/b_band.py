"""Descriptive: strict share in line from the first service at band tops 1.2 m (as scored)
and 1.5 m (the sim's JOIN_GAP_M top), from the logged samples. Zero calls; read-only.

    .venv/bin/python round7/b_band.py SELECTION_JSON RUN_ROOT ARM [ARM ...] \
        [--pairs A-B,C-D]
    (SELECTION_JSON relative to results/moving-lines/, RUN_ROOT relative to results/data/moving/.)
"""
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
sys.path.insert(0, str(HERE))
from a_loss import MOVING, window  # noqa: E402
from summarise_fresh import in_line, sign_p  # noqa: E402

TOPS = (1.2, 1.5)
TIE = 0.01


def strict_at(x, top):
    """The primary and strict scorings' test on one logged sample, with the band's top raised to `top`.
    At 1.2 it is the sim's own logged flag: gap_m is rounded to 1 mm, so a logged 1.2 can be a failed 1.2004.
    Above 1.2 a sample also passes when every other condition holds and 1.2 <= gap_m <= top."""
    if in_line(x, "strict") or top <= 1.2 or x["ahead"] < 1:
        return in_line(x, "strict")
    return bool(x.get("behind_m", -1) >= 0 and 1.2 <= x.get("gap_m", 0) <= top
                and x.get("heading_off_deg", 180) <= 45.0 and x["nearest_m"] >= 0.55 and x["side"] <= 0.6)


def episode(e):
    _, win = window(e)
    if not win:
        return None
    out = {top: sum(strict_at(x, top) for x in win) / len(win) for top in TOPS}
    ref = sum(in_line(x, "strict") for x in win) / len(win)
    assert abs(out[1.2] - ref) < 1e-9, (out[1.2], ref)
    far = [x for x in win if x["ahead"] >= 1 and x.get("gap_m", 0) > 1.2]
    out["far"], out["far_pass"] = len(far), sum(strict_at(x, 1.5) for x in far)
    return out


def t95(df):
    """One-sided 0.95 t quantile (Cornish-Fisher from the normal; fine for df > 20)."""
    z = 1.6448536269514722
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def compare(sc, a, b, top):
    ds = [sc[(lay, a)][top] - sc[(lay, b)][top] for lay in {k[0] for k in sc}
          if sc.get((lay, a)) is not None and sc.get((lay, b)) is not None]
    w = sum(d >= TIE for d in ds)
    lo = sum(d <= -TIE for d in ds)
    t = len(ds) - w - lo
    m = statistics.mean(ds) if ds else float("nan")
    lb = m - t95(len(ds) - 1) * statistics.stdev(ds) / math.sqrt(len(ds)) if len(ds) > 2 else float("nan")
    return len(ds), w, lo, t, sign_p(w, lo), m, lb


def main():
    args = sys.argv[1:]
    pairs = []
    if "--pairs" in args:
        i = args.index("--pairs")
        pairs = [tuple(p.split("-")) for p in args[i + 1].split(",")]
        args = args[:i] + args[i + 2:]
    sel_path, root, arms = args[0], args[1], args[2:]
    sel = json.loads((paths.MOVING / sel_path).read_text())["layouts"]
    sc = {}
    for lay in sel:
        for arm in arms:
            hits = sorted((MOVING / root).glob(f"**/{lay['name']}/seed{lay['seed']}/{arm}/episode.json"))
            sc[(lay["name"], arm)] = episode(json.loads(hits[0].read_text())) if hits else None
    print(f"# Round 7 part B, DESCRIPTIVE: band top 1.2 m (frozen) against 1.5 m ({sel_path}, {root})\n")
    print("| arm | episodes | strict at 1.2 | strict at 1.5 | 'too far' seconds at 1.2 that pass at 1.5 |")
    print("|---|---|---|---|---|")
    for arm in arms:
        v = [sc[(lay["name"], arm)] for lay in sel if sc.get((lay["name"], arm)) is not None]
        if not v:
            print(f"| {arm} | 0 | - | - | - |")
            continue
        far, fp = sum(x["far"] for x in v), sum(x["far_pass"] for x in v)
        print(f"| {arm} | {len(v)} | {statistics.mean(x[1.2] for x in v):.3f} | {statistics.mean(x[1.5] for x in v):.3f}"
              f" | {fp}/{far} |")
    if pairs:
        print("\n| pair (a − b) | top | n | wins/losses/ties | sign p | mean d | 95% lower bound |")
        print("|---|---|---|---|---|---|---|")
        for a, b in pairs:
            for top in TOPS:
                n, w, lo, t, p, m, lb = compare(sc, a, b, top)
                print(f"| {a} − {b} | {top} | {n} | {w}/{lo}/{t} | {p:.3f} | {m:+.3f} | {lb:+.3f} |")


if __name__ == "__main__":
    main()
