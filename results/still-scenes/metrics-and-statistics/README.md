# Metrics and statistics

**Question.** How should the still-scene runs be scored and compared: which place-set metrics, and which statistics at the level of layouts?

**Result.** The v0 metric note fixes top-1 success plus hit@k and P(success) over the solver's five places. The statistics study shows records within a layout are correlated (ICC of success by layout 0.08 to 0.75), so comparisons use an exact sign-flip test with the layout as the unit. At the new wording C beats E and R in all 12 model comparisons, largest two-sided p = 0.0156.

**Files.**
- `metric-v0/` : the frozen v0 metric note, the scorer (`i14.py`), its per-record rows and tables.
- `metric-v1/` : the draft v1 metric list, its changelog, fact-check and confidence update (notes only).
- `statistics/` : intraclass correlation, t intervals, power and minimum detectable difference by layout, and the reporting guide.
- `stats-rerun/` : the layout-clustered sign-flip tests recomputed from the recorded runs.

**Reproduce.** See each subfolder. Everything reads recorded runs through the replay harness and calls no model.

**Data.** `logs/still-runs.tar.gz` (the `runs/` folder at the repo root) and `logs/research-data.tar.gz` (rows and tables beside the notes). Uses the step-0 replay harness in `results/still-scenes/analysis-scripts/harness/` and the development scenes in `data/dev_scenes/` (release dev-scenes-v1).
