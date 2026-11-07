# Depth-snap fix: first check

**Question.** If a marked point is moved to the nearest selectable floor cell, how many failed R and E records become successes?

**Result.** Over 3024 records the rescored recorded verdict equals the stored one in all cases. With the depth-only snap, Qwen goes from 179 to 190 (R) and 196 to 207 (E) of 252; the people-aware solver (C) reaches 228. Most of the failures that remain are at lines.

**Files.**
- `snap.py`: applies the snap and rescores; writes `snap_records.jsonl`
- `summarize.py / snap_tables.txt`: the tables
- `n1n2.py / n1n2.txt, n1_s23.py`: near-miss counts by model (`n1_s23.py` writes `n1_s23.txt`, which is not stored)
- `explore.py`: first look
- `verify/`: independent recomputation

**Reproduce.** From the repo root: `SGI_CACHE_ONLY=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src python results/still-scenes/depth-fix/first-check/snap.py`. then `summarize.py`. The scripts import the harness in `results/still-scenes/analysis-scripts/harness/` and read the development scenes from `data/dev_scenes/` (release dev-scenes-v1). `summarize.py` and `n1_s23.py` read only the stored `snap_records.jsonl`.

**Data.** `results/logs/still-runs.tar.gz` (unpacks to `runs/` at the repo root); development scenes in `data/dev_scenes/` (release dev-scenes-v1).
