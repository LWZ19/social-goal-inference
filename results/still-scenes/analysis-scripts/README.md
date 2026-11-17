# Shared analysis scripts

**Question.** Small judge scripts that read recorded runs and print success, latency and cascade bounds.

**Result.** No stored outputs; other folders port these scripts and cite their results.

**Files.**
- `bounds.py, cross.py, mixedC.py`: per-arm success bounds and cross-method tables
- `cp.py, cp2.py, lat.py`: critical-path and latency per run
- `decomp.py, near.py, clus.py`: failure decomposition, near misses, clustering
- `cascade_peek.py`: cascade peek (changes into `runs/`)
- `harness/`: the offline harness; documented in its own folder

**Reproduce.** From the repo root, for example `python results/still-scenes/analysis-scripts/lat.py <run-name-pattern>`; `cp.py` and `cp2.py` take run folder names under `runs/`. Module names are kept because other scripts import from this folder.

**Data.** `results/logs/still-runs.tar.gz` (unpacks to `runs/` at the repo root)
