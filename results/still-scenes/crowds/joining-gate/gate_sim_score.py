"""The joining gate in moving lines: the hold-place version (CT3FL)
replayed with the gate, against CT3FL replayed without it from the same stored answers (a replay does not reproduce
the live runs exactly), on the same layouts and seeds, on the hold-place measure (strict share, whole 92 s run). The
recorded live runs are shown for reference. Zero calls.
    PYTHONPATH=src python results/still-scenes/crowds/joining-gate/gate_sim_score.py"""
import json
import statistics
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
sys.path.insert(0, str(paths.DESIGN))
from summarise_fresh import measures  # noqa: E402

JOBS = paths.RESULTS / "moving-lines/hold-place/ceiling"
TUNED = {"cafe_2010", "cafe_2036", "cafe_2042", "cafe_2438", "sweet_2010", "sweet_2144", "sweet_2163", "sweet_2364"}
MODELS = {"dsv41": "DeepSeek V4.1", "sonnet55_low": "Claude Sonnet 5.5"}


ROOTS = (("runs_hold", "recorded live"), ("runs_hold_replay", "replay"), ("runs_hold_gate", "replay with the gate"))


def episode(root, scen, seed):
    f = paths.MOVING_RUNS / root / scen / f"seed{seed}" / "CT3FL" / "episode.json"
    return measures(json.loads(f.read_text())) if f.exists() else None


def sign_p(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n)


for m, name in MODELS.items():
    jobs = [line.split() for line in (JOBS / f"jobs_{m}.txt").read_text().splitlines() if line.strip()]
    for subset, keep in (("64 untuned layouts", lambda s: s.split("/")[1] not in TUNED), ("all 72 layouts", lambda s: True)):
        rows = [tuple(episode(f"{root}/{m}", s, seed) for root, _ in ROOTS) for s, seed in jobs if keep(s)]
        done = [r for r in rows if all(r)]
        print(f"## {name}, {subset}: {len(done)} of {len(rows)} layouts with all three runs")
        if not done:
            continue
        for key, label in (("share_strict", "strict"), ("share_frozen", "primary")):
            means = [statistics.mean(r[i][key] for r in done) for i in range(3)]
            print(f"- {label} share: " + ", ".join(f"{ROOTS[i][1]} {means[i]:.3f}" for i in range(3)))
            for a, b in ((2, 1), (1, 0)):
                d = [round(r[a][key] - r[b][key], 3) for r in done]
                w, l = sum(x > 0 for x in d), sum(x < 0 for x in d)
                print(f"  {ROOTS[a][1]} minus {ROOTS[b][1]}: {100 * (means[a] - means[b]):+.1f} points; layouts better "
                      f"{w}, worse {l}, level {len(d) - w - l}, sign p {sign_p(w, l):.4f}")
    print()
