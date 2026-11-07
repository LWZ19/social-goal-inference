# Lifting fixes (pre-registered Stage 0)

**Question.** Would a zero-call repair of the pixel lift (snap to the person or object, depth-relative joins, front-surface rules V1-V4) fix the wrong targets that come from lifting?

**Result.** Verdict "neither": 20 of 46 GLM and DeepSeek wrong-target records have a lift class (43.5%), the oracle fixes 20 with 0 breaks of 454, and V1 reaches 9 of the oracle's 20 (share 0.45, bar 0.70). A front-surface snap fixes 7 but breaks 24.

**Files.**
- `FROZEN.md`, `DEVIATIONS.md`: the plan and its notes.
- `i11lib.py`, `run_eval.py`, `s0_checks.py`, `report.py`, `repro_lifting.py`, `verify_indep.py`: library, runs, checks and report.
- `rows_*.jsonl`, `m4_*.jsonl`, `meta_*.json`, `summary.json`, `rows.jsonl`, `tables.txt`, `out_primary.txt`, `out_extras.txt`, `s0_fit.txt`, `verify_out.txt`: outputs.
- `repro_lifting/`: reproduction of the lifting rerun outputs.
- `verify_i11_stage0/`: an independent re-implementation (`vlib.py`, `vcompute.py`, ...) with its outputs.

**Reproduce.** `report.py` rebuilds `tables.txt` and `out_primary.txt` from the stored rows and the per-record `records_*.tsv` files, which are missing from this repository. `run_eval.py`, `s0_checks.py`, `repro_lifting.py`, `verify_indep.py` and the `verify_i11_stage0` scripts read runs and scenes through the step-0 harness (`results/still-scenes/analysis-scripts/harness/`) and `data/dev_scenes/`. The code pins hashes of `FROZEN.md`, `lifting-rerun/lifting.py` and `frontsnap.py`, of the harness and of three `src/sgi` files; the copies in this repository were edited after the hashes were taken (the harness was re-pointed), so the pin checks stop the scripts. With the checks skipped in a scratch copy they run.

**Data.** `results/logs/still-runs.tar.gz` (the `runs/` folder at the repo root) and the development scenes in `data/dev_scenes/` (release dev-scenes-v1). Data files beside the notes: `results/logs/research-data.tar.gz`.