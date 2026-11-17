"""CT4_FROZEN.md section 4's "no-harm on fresh original-scene seeds 20-29" (reported, not a claim): DeepSeek,
cafe_line2_moving, room_coffee_cart_slots, sweet_spot_line2_moving and cafe_line6_moving, seeds 20-29, arms CT2, CT3,
CT3F and CT4. The note names no measure for it, so this reports the sweep's own per-episode measures (the frozen
`test-layouts/score_sweep.episode_scores`, imported unchanged): strict share in line, right last, cut-in samples, and share
after the first service; each arm paired with CT3 by scene and seed (exact sign test, ties |d| < 0.01). Zero calls.

    python3 noharm_report.py results/data/moving/runs_noharm/dsv41   (from the repo root)
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
sys.path.insert(0, str(paths.MOVING / "test-layouts"))
sys.path.insert(0, str(HERE))
from score_sweep import episode_scores  # noqa: E402
from summarise_fresh import sign_p  # noqa: E402

ARMS = ("CT2", "CT3", "CT3F", "CT4")
KEYS = (("share_strict", "strict share in line"), ("right_last", "right last"), ("after_service_strict",
        "strict share after the first service"), ("cut", "cut-in samples (lower is better)"))


def main(root: Path) -> int:
    scores = {}
    for f in sorted(root.glob("*/seed*/*/episode.json")):
        scene, seed, arm = f.parts[-4], f.parts[-3], f.parts[-2]
        if arm in ARMS:
            scores[(scene, seed, arm)] = episode_scores(json.loads(f.read_text()))
    cells = sorted({(s, d) for s, d, _ in scores})
    print(f"# No-harm, {root}: {len(cells)} scene-seed cells\n")
    print("| Arm | " + " | ".join(label for _, label in KEYS) + " |")
    print("|---|" + "---|" * len(KEYS))
    for arm in ARMS:
        row = []
        for key, _ in KEYS:
            v = [scores[(s, d, arm)][key] for s, d in cells if scores.get((s, d, arm), {}).get(key) is not None]
            row.append(f"{statistics.mean(v):.3f} (n {len(v)})" if v else "-")
        print(f"| {arm} | " + " | ".join(row) + " |")
    print("\nPaired against CT3 (wins/losses/ties of the arm; for cut-ins a win is fewer):")
    for arm in ("CT2", "CT3F", "CT4"):
        for key, label in KEYS:
            w = l = t = 0
            for s, d in cells:
                a, b = scores.get((s, d, arm), {}).get(key), scores.get((s, d, "CT3"), {}).get(key)
                if a is None or b is None:
                    continue
                diff = (b - a) if key == "cut" else (a - b)
                tie = abs(diff) < (0.5 if key == "cut" else 0.01)
                w, l, t = w + (diff > 0 and not tie), l + (diff < 0 and not tie), t + tie
            print(f"- {arm} vs CT3, {label}: {w}/{l}/{t}, sign p {sign_p(w, l):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
