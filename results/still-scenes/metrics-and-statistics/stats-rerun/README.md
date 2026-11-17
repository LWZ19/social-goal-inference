# Layout-clustered tests, re-computed (I46)

**Question.** Do the C-over-E and C-over-R gains hold under a test whose unit is the layout?

**Result.** At the new wording, C beats E and R for all six models: two-sided exact sign-flip p between 0.0039 and 0.0156 over 12 comparisons (8 or 9 of 14 layouts favour C, none favour E); C is below R in 2 of 84 model x layout cells.

**Files.**
- `stats.py`, `stats_output.txt` : the tests (also imported by `../statistics/i46_stats.py`).
- `list_runs.py`, `list_runs_output.txt` : the runs read.
- `explore.py`, `explore_output.txt` : a first look at the cells.
- `verify/` : an independent re-computation (`v_stats.py`, `v_rescore.py`, `v_runs.py`) with outputs.

**Reproduce.** From the repo root: `PYTHONPATH=src python results/still-scenes/metrics-and-statistics/stats-rerun/stats.py`. Uses the step-0 replay harness in `results/still-scenes/analysis-scripts/harness/` and the development scenes in `data/dev_scenes/` (release dev-scenes-v1).

The stored `stats_output.txt` and `verify/v_stats.txt` are subsets of what the scripts print now (the power section is missing); every stored line is reproduced.

**Data.** `logs/still-runs.tar.gz`.
