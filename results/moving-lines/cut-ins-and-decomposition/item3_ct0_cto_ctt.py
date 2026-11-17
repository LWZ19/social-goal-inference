"""Audit item 3 (zero calls): CT0, CTo and CTt on seeds 0-2, scored under SCORING.md (frozen, primary) and
SCORING_STRICT.md, with summarise_fresh.py's functions (in_line, measures, sign_p) imported unchanged.

    python item3_ct0_cto_ctt.py > item3_out.txt

Cells: the five seeds-0-2 scenarios the arms ran on, seeds 0, 1, 2, per model folder under results/data/moving/runs/.
summarise_fresh.load() keeps only seeds >= 10, so episodes are loaded here (same glob, same JSON), seeds 0-2.
Ties: |d| < 0.01 (the audit's rule). Every episode has 91 samples, so shares move in steps of 1/91 = 0.011 and the
0.01 rule gives the same W/L/T as summarise_fresh's exact ties (|d| < 1e-9); the script asserts this.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

M = paths.MOVING_RUNS                    # run folders: M / "runs" / <model> / ...
sys.path.insert(0, str(paths.DESIGN))
from summarise_fresh import measures, sign_p  # noqa: E402  (frozen and strict scorings, unchanged)

SCENES = ["cafe", "room_coffee_cart", "cafe_line2_moving", "cafe_line6_moving", "sweet_spot_line2_moving"]
SEEDS = [0, 1, 2]
ARMS = ["R1", "C1", "CT", "CT0", "CTo", "CTt"]          # R1 and C1/CT for context and pairing
NEW = ["CT0", "CTo", "CTt"]
PAIRS = [("CT0", "C1", "people positions from tracking (one re-solve, then frozen)"),
         ("CT", "CT0", "re-solving as people move"),
         ("CTt", "CT", "oracle ceiling: true counter point as target"),
         ("CTo", "CT", "oracle ceiling: true members as come_after"),
         ("CT", "C1", "context: CT's whole gain over C1")]
TIE = 0.01


def load(model):
    eps = {}
    for sc in SCENES:
        for sd in SEEDS:
            for arm in ARMS:
                f = M / "runs" / model / sc / f"seed{sd}" / arm / "episode.json"
                if f.exists():
                    d = json.loads(f.read_text())
                    assert (d["scenario"], d["seed"], d["arm"]) == (sc, sd, arm), f
                    eps[(sc, sd, arm)] = d
    return eps


def wlt(rows, tie):
    w = sum(a - b >= tie for a, b in rows) if tie > 0 else sum(a > b + 1e-9 for a, b in rows)
    l = sum(b - a >= tie for a, b in rows) if tie > 0 else sum(a < b - 1e-9 for a, b in rows)
    return w, l, len(rows) - w - l


def main():
    models = sorted(p.name for p in (M / "runs").iterdir() if p.is_dir()
                    and any((p / sc / f"seed{sd}" / a / "episode.json").exists()
                            for sc in SCENES for sd in SEEDS for a in NEW))
    print("# Audit item 3: CT0, CTo, CTt on seeds 0-2 (zero calls; logs only)\n")
    print(f"Model folders with any of {NEW} on seeds 0-2: {models}")
    print(f"Cells: {len(SCENES)} scenarios x seeds {SEEDS} = {len(SCENES) * len(SEEDS)} per arm. "
          f"Ties |d| < {TIE} (identical to exact ties here: shares step by 1/91). "
          "Sign p: summarise_fresh.sign_p (exact, two-sided, over untied pairs).")
    for model in models:
        eps = load(model)
        label = sorted({e["label"] for e in eps.values()})
        print(f"\n## {model} (label {label})\n")
        missing = [(sc, sd, a) for sc in SCENES for sd in SEEDS for a in ARMS if (sc, sd, a) not in eps]
        by_arm = defaultdict(list)
        for (sc, sd, a), e in eps.items():
            by_arm[a].append((sc, sd))
        miss = {a: [f"{sc} s{sd}" for sc, sd, b in missing if b == a] for a in ARMS}
        txt = [f"{a}: {'all 15 (arm not run)' if len(v) == 15 else ', '.join(v)}" for a, v in miss.items() if v]
        print("Missing episodes: " + ("; ".join(txt) if txt else "none"))
        m = {k: measures(e) for k, e in eps.items()}

        print("\n| Arm | Cells | Mean share (frozen, primary) | Mean share (strict) | End in line (strict) | Cut-in samples |")
        print("|---|---|---|---|---|---|")
        for a in ARMS:
            xs = [m[(sc, sd, a)] for sc in SCENES for sd in SEEDS if (sc, sd, a) in m]
            if xs:
                print(f"| {a} | {len(xs)} | {statistics.mean(x['share_frozen'] for x in xs):.3f} | "
                      f"{statistics.mean(x['share_strict'] for x in xs):.3f} | {sum(x['end_strict'] for x in xs)} | "
                      f"{sum(x['cut'] for x in xs)} |")

        print("\nMean share by scenario, frozen / strict (3 seeds each):\n")
        arms_here = [a for a in ARMS if by_arm.get(a)]
        print("| Scenario | " + " | ".join(arms_here) + " |")
        print("|---|" + "---|" * len(arms_here))
        for sc in SCENES:
            cells = []
            for a in arms_here:
                xs = [m[(sc, sd, a)] for sd in SEEDS if (sc, sd, a) in m]
                cells.append("-" if not xs else f"{statistics.mean(x['share_frozen'] for x in xs):.2f} / "
                             f"{statistics.mean(x['share_strict'] for x in xs):.2f}")
            print(f"| {sc} | " + " | ".join(cells) + " |")

        print("\nPaired by (scenario, seed): A vs B, wins = A higher share.\n")
        print("| Comparison | What it measures | Pairs | Frozen W/L/T | Frozen mean d | Frozen sign p "
              "| Strict W/L/T | Strict mean d | Strict sign p |")
        print("|---|---|---|---|---|---|---|---|---|")
        for a, b, what in PAIRS:
            keys = [(sc, sd) for sc in SCENES for sd in SEEDS if (sc, sd, a) in m and (sc, sd, b) in m]
            if not keys:
                print(f"| {a} vs {b} | {what} | 0 (arm not run) | - | - | - | - | - | - |")
                continue
            out = []
            for key in ("share_frozen", "share_strict"):
                rows = [(m[(sc, sd, a)][key], m[(sc, sd, b)][key]) for sc, sd in keys]
                w, l, t = wlt(rows, TIE)
                assert (w, l, t) == wlt(rows, 0), (model, a, b, key)
                d = statistics.mean(x - y for x, y in rows)
                out += [f"{w}/{l}/{t}", f"{d:+.3f}", f"{sign_p(w, l):.3f}"]
            print(f"| {a} vs {b} | {what} | {len(keys)} | " + " | ".join(out) + " |")


if __name__ == "__main__":
    main()
