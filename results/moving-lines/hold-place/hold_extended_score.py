"""CT3FL (hold place) against CT3F on the 72 confirm4 layouts, four models, both arms in the updated simulator;
and CT3F in the old simulator (runs_x1) against the updated one. Paired by layout. Zero calls.
    PYTHONPATH=src python results/moving-lines/hold-place/hold_extended_score.py"""
import json, math, statistics, sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
sys.path.insert(0, str(paths.DESIGN))
from summarise_fresh import measures, sign_p  # noqa: E402

TUNED = {"cafe_2010", "cafe_2036", "cafe_2042", "cafe_2438", "sweet_2010", "sweet_2144", "sweet_2163", "sweet_2364"}
JOBS = [l.split() for l in (paths.MOVING / "test-layouts/jobs_confirm.txt").read_text().splitlines() if l.strip()]
MODELS = {"sonnet55_low": "Claude Sonnet 5.5", "dsv41": "DeepSeek V4.1", "claude_haiku55": "Claude Haiku 5.5",
          "gemini_fast": "Gemini 3.8 Flash"}


def episode(root, scen, seed, arm):
    f = paths.MOVING_RUNS / root / scen / f"seed{seed}" / arm / "episode.json"
    return measures(json.loads(f.read_text())) if f.exists() else None


def compare(rows, key, label_a, label_b):
    d = [a[key] - b[key] for a, b in rows]
    w, l = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
    ma, mb = statistics.mean(a[key] for a, _ in rows), statistics.mean(b[key] for _, b in rows)
    return f"{label_a} {ma:.3f} vs {label_b} {mb:.3f} (mean {statistics.mean(d):+.3f}; {w}/{l}/{len(d)-w-l}, sign p {sign_p(w, l):.4f})"


for m, name in MODELS.items():
    for subset, keep in (("64 unseen layouts", lambda s: s.split("/")[1] not in TUNED), ("8 tuning layouts", lambda s: s.split("/")[1] in TUNED)):
        new = [(episode(f"runs_hold/{m}", s, seed, "CT3FL"), episode(f"runs_hold/{m}", s, seed, "CT3F")) for s, seed in JOBS if keep(s)]
        new = [r for r in new if r[0] and r[1]]
        if not new:
            continue
        print(f"## {name}, {subset}: {len(new)} paired layouts (updated simulator)")
        print("- strict share: " + compare(new, "share_strict", "CT3FL", "CT3F"))
        print("- primary share: " + compare(new, "share_frozen", "CT3FL", "CT3F"))
        print(f"- cut-in seconds: CT3FL {sum(a['cut'] for a, _ in new)} vs CT3F {sum(b['cut'] for _, b in new)}")
    sim = [(episode(f"runs_hold/{m}", s, seed, "CT3F"), episode(f"runs_x1/{m}", s, seed, "CT3F")) for s, seed in JOBS]
    sim = [r for r in sim if r[0] and r[1]]
    if sim:
        print(f"## {name}: CT3F, updated simulator against the old one, {len(sim)} layouts")
        print("- strict share: " + compare(sim, "share_strict", "updated", "old"))
    print()
