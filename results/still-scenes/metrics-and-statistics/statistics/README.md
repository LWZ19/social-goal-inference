# Layout statistics for the realistic sets (I46)

**Question.** How many independent observations do 252 records per model carry, and what can a comparison of methods detect?

**Result.** Success is clustered by layout: ICC(1) runs from 0.08 to 0.75 over 27 model x method cells (design effect 1 + 17 ICC), so 252 records carry the information of about 18 to 107 independent ones. Comparisons should be made over the 14 layouts, with an exact sign-flip test (`HOW_TO_REPORT.md`).

**Files.**
- `i46_stats.py` : computes the five tables and `i46_stats_output.txt`.
- `icc_table.txt`, `tci_table.txt`, `power_table.txt`, `noise_table.txt`, `mdd_table.txt` : ICC, t intervals, power, batch noise, minimum detectable difference.
- `HOW_TO_REPORT.md` : the reporting rule; `HOW_TO_REPORT.before-fix.md` is the earlier text.
- `verify_i46.py`, `verify_output.txt` : an independent check with its own formulas.
- `verify/` : a second verification (`v_core.py`, `v_power.py`, `v_extra.py`) with outputs; `list*.py` list runs.

**Reproduce.** From the repo root: `PYTHONPATH=src python results/still-scenes/metrics-and-statistics/statistics/i46_stats.py`. Uses the step-0 replay harness in `results/still-scenes/analysis-scripts/harness/` and the development scenes in `data/dev_scenes/` (release dev-scenes-v1). `verify/v_core.py` writes a `data.pkl` cache that `v_power.py` and `v_extra.py` then read; the cache is not stored, so run them in that order from a scratch copy. All three reproduce their stored outputs exactly. `i46_stats.py` reproduces every stored number; the stored output files omit a few explanatory lines that the script prints.

**Data.** `logs/still-runs.tar.gz`.
