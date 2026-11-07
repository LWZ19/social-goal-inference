# Baselines: solver alone and people-blind standard goal

**Question.** How much does the constraint method gain over placing the standard goal with no people information?

**Result.** For Qwen (252 realistic records) R scores 179, E 196, C 228, solver alone 143, standard goal 139, come-after-only 228. For GLM: C 170, solver alone 132, standard goal 124.

**Files.**
- `i01_baselines.py`: computes the rows, writes `records.jsonl` and `re_records.jsonl`
- `tables.py / tables.txt`: the tables
- `check_empty.py / check_empty.txt`: an empty constraint list reaches the solver as no constraints
- `verify/`: independent recomputation and comparison

**Reproduce.** From the repo root: `SGI_CACHE_ONLY=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src python results/still-scenes/replay-harness/baselines/i01_baselines.py`. then `tables.py`. The scripts import the harness in `results/still-scenes/analysis-scripts/harness/` and read the development scenes from `data/dev_scenes/` (release dev-scenes-v1). `i01_baselines.py` and `verify/v1_compute.py` each take more than 10 minutes.

**Data.** `results/logs/still-runs.tar.gz` (unpacks to `runs/` at the repo root); development scenes in `data/dev_scenes/` (release dev-scenes-v1).
