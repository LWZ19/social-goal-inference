# Soft density over places (I04)

**Question.** Does a soft density over the floor, fitted with leave-one-layout-out, give a better distribution of acceptable places than the solver's single answer?

**Result.** Held out over 549 primary goal-step records, the solver's top place scores a mean success of 0.760 against 0.670 (V1) and 0.681 (V2) for the soft densities and 0.749 for the Gaussian control (V3). Lift over the floor: 8.90, 7.85, 7.98 and 8.78. The soft density is not better than the solver (`tables.txt`).

**Files.**
- `FROZEN.md`, `DEVIATIONS.md` : the frozen protocol and the (empty) list of deviations.
- `i04_cache.py` : per-record cache; `cache/` holds its metadata and checks (the `.npz` files are not stored).
- `i04_check.py`, `cache/check.txt` : cache checks.
- `i04_fit.py`, `fits.json`, `folds.json` : the fold fits.
- `i04_heldout.py`, `tables.txt`, `summary.json`, `rows_*.jsonl` : held-out scoring.
- `prefreeze/`, `posthoc/`, `verify/` : format check, in-sample grid, independent verification.

**Reproduce.** From the repo root with `PYTHONPATH=src`: `i04_cache.py SHARD` (twice), `i04_check.py`, `i04_fit.py`, `i04_heldout.py`. Uses the step-0 replay harness in `results/still-scenes/analysis-scripts/harness/` and the development scenes in `data/dev_scenes/` (release dev-scenes-v1). `i04_check.py` was re-run and its stored output is identical. `i04_cache.py` needs more than 10 minutes per shard; its first rows match the stored ones apart from timing. The later steps need the cache `.npz` files and `FITS.sha256`, which are not stored.

**Data.** `logs/still-runs.tar.gz`, `logs/bulk.tar.gz` (cache).
