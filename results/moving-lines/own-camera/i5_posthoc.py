"""Added after the planned test (i5_score_out.txt): the shape of the per-cell differences CTd - CT2 (strict and
primary): median, quartiles, and how many cells lose or gain more than 0.05. Zero calls; read-only."""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import mcompare as C  # noqa: E402

MODELS = {"Sonnet 5.5": ("sonnet55_low", range(10, 20)), "DeepSeek V4.1": ("dsv41", range(10, 15)),
          "Haiku 5.5": ("claude_haiku55", range(10, 15)), "Gemini 3.8": ("gemini_fast", range(10, 15))}
allp = []
for name, (d, seeds) in MODELS.items():
    allp += C.pairs(C.load(f"runs_i5/{d}", C.EVAL5, seeds, ("CTd", "CT2")), "CTd", "CT2")
for key in ("share_strict", "share_frozen"):
    dd = [x[key] - y[key] for _, x, y in allp]
    q = statistics.quantiles(dd, n=4)
    print(f"{key}: {len(dd)} cells; median {statistics.median(dd):+.3f}, quartiles {q[0]:+.3f} / {q[2]:+.3f}; "
          f"losses > 0.05: {sum(v < -0.05 for v in dd)}, gains > 0.05: {sum(v > 0.05 for v in dd)}; "
          f"losses <= 0.05: {sum(-0.05 <= v < 0 for v in dd)}, gains <= 0.05: {sum(0 < v <= 0.05 for v in dd)}, zero {sum(v == 0 for v in dd)}")
by = {}
for (sc, s), x, y in allp:
    by.setdefault(sc, []).append(x["share_strict"] - y["share_strict"])
for sc, v in sorted(by.items()):
    print(f"  {sc}: {len(v)} cells, strict mean {statistics.mean(v):+.3f}, losses {sum(a < 0 for a in v)}, gains {sum(a > 0 for a in v)}")
