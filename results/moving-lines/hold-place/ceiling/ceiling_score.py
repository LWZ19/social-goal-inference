"""The ideal-place robot (OW) in the updated simulator, next to CT3F and CT3FL from runs_hold, on the hold-place
measure (strict share, whole 92 s run). Zero calls.
    PYTHONPATH=src python results/moving-lines/hold-place/ceiling/ceiling_score.py"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
sys.path.insert(0, str(paths.DESIGN))
from summarise_fresh import measures  # noqa: E402

HERE = Path(__file__).resolve().parent
TUNED = {"cafe_2010", "cafe_2036", "cafe_2042", "cafe_2438", "sweet_2010", "sweet_2144", "sweet_2163", "sweet_2364"}
MODELS = {"dsv41": "DeepSeek V4.1", "sonnet55_low": "Claude Sonnet 5.5"}


def episode(root, scen, seed, arm):
    f = paths.MOVING_RUNS / root / scen / f"seed{seed}" / arm / "episode.json"
    return measures(json.loads(f.read_text())) if f.exists() else None


for m, name in MODELS.items():
    jobs = [line.split() for line in (HERE / f"jobs_{m}.txt").read_text().splitlines() if line.strip()]
    for subset, keep in (("64 untuned layouts", lambda s: s.split("/")[1] not in TUNED), ("all 72 layouts", lambda s: True)):
        rows = [(episode(f"runs_hold_ceiling/{m}", s, seed, "OW"), episode(f"runs_hold/{m}", s, seed, "CT3F"),
                 episode(f"runs_hold/{m}", s, seed, "CT3FL")) for s, seed in jobs if keep(s)]
        done = [r for r in rows if all(r)]
        print(f"## {name}, {subset}: {len(done)} of {len(rows)} layouts with all three arms (updated simulator)")
        if not done:
            continue
        for key, label in (("share_strict", "strict"), ("share_frozen", "primary")):
            ow, a, b = (statistics.mean(r[i][key] for r in done) for i in range(3))
            print(f"- {label} share: ideal place (OW) {ow:.3f}, best version (CT3F) {a:.3f}, "
                  f"with hold place (CT3FL) {b:.3f}")
    print()
