# Repairs: fix one input of the people call at a time

**Question.** If one input to the solver were correct (positions, facing, missed people, spurious people), how many failed goal steps would be fixed?

**Result.** The replay reproduces all recorded places (0 moved). For Qwen (new wording, 252 records) C is 228; correct positions, facing and added people each leave it at 228, dropping spurious people gains 2, and the oracle target (R6) reaches 239. Lines are where gains show: for GLM (old wording) they go from 42 to 61 of 90 with reference constraints (R5).

**Files.**
- `repairs.py`: runs every repair arm and writes `rows_<wording>_<model>.jsonl`
- `tables.py / tables.txt`: per model and situation tables
- `mechanism.py / mechanism.txt`: what the failing records have in common
- `verify.py / verify.txt`: check against the first-pass counts
- `probe.py, smoke.py`: small probes, no stored output
- `verify/`: independent re-implementation and its tables

**Reproduce.** From the repo root: `SGI_CACHE_ONLY=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src python results/still-scenes/replay-harness/repairs/repairs.py`. then `tables.py`, `mechanism.py`, `verify.py`. The scripts import the harness in `results/still-scenes/analysis-scripts/harness/` and read the development scenes from `data/dev_scenes/` (release dev-scenes-v1). `repairs.py`, `verify.py` and `verify/vrun.py` each take more than 10 minutes; the stored `rows_*.jsonl` and `verify/vrows_*.jsonl` come from full runs.

**Data.** `results/logs/still-runs.tar.gz` (unpacks to `runs/` at the repo root); development scenes in `data/dev_scenes/` (release dev-scenes-v1).
