"""Where things are in this repository, so that no script depends on how deep its own folder sits.

A script finds this module from anywhere under results/ with:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
    import paths
"""
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1]
REPO = RESULTS.parent
LIB = RESULTS / "_lib"
STILL = RESULTS / "still-scenes"
MOVING = RESULTS / "moving-lines"
RUNS = REPO / "runs"                       # still-scene run folders (logs/still-runs.tar.gz, unpacked at the repo root)
MOVING_RUNS = RESULTS / "data" / "moving"  # simulator run folders runs, runs_i4, runs_i10, ... (logs/moving-runs.tar.gz)
DESIGN = MOVING / "design-and-scoring"     # run_moving.py, summarise_fresh.py and the moving-line scoring notes
