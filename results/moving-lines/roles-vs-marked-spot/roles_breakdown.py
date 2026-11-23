"""A further breakdown: the secondary RTw vs R1. RTw comes from runs_i4/<model> and R1 from runs/<model>, so the two
arms ran under different machine loads (roles_score.py runs RTw and CT2 together to avoid this).
Zero calls; read-only."""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import mcompare as C  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from roles_score import POOLED, REPORTED  # noqa: E402

rows, pooled = [], []
for name, (root, seeds) in {**POOLED, **REPORTED}.items():
    new = C.load(root, C.EVAL5, seeds, ("RTw",))
    old = C.load(root.replace("runs_i4/", "runs/"), C.EVAL5, seeds, ("R1",))
    if not new:
        continue
    eps = {**new, **old}
    rr = C.pairs(eps, "RTw", "R1")
    rows.append(C.row(f"{name}: RTw (runs_i4) vs R1 (runs)", rr))
    if name in POOLED:
        pooled += rr
rows.append(C.row("pooled (Sonnet, DeepSeek, Haiku, Gemini)", pooled))
print("# A further breakdown, post hoc: RTw vs R1 (R1 from the earlier runs; different machine loads)\n\n" + C.table(rows))
