# Depth-blob people detector: frozen fit, evaluation and solver replays

**Question.** Can a numpy-only detector that finds people as blobs in the depth image replace or repair the model's people answer, without any tuning on the evaluation scenes?

**Result.** Stage A (frozen setting, 68 evaluation scenes): recall 36/42 (real), 35/42 (staffed), 34/34 (plain), 58/58 (placeholder); the misses are staff whose blob merges with tall structure. The stage verdict was "neither". Stage B replays the solver with the blobs: GLM on lines goes from 42 to 44 (add), 55 (prune) and 57 (relocate), while Qwen moves by at most 2 records.

**Files.**
- `FROZEN.md`, `DEVIATIONS.md`, `DEVIATIONS_POST_EVAL.md`, `ADDENDUM_FIX.md`: the plan fixed before evaluation and the later notes.
- `blobs.py`, `people_scorer.py`: the detector and the shared people scorer.
- `gen_fresh.py`: renders the fresh fit seeds (`../fresh_seeds/`).
- `fit_selection.json`, `fit_selection_fix.json`: the fitted setting and the one general fix.
- `stageA/`: fit grid, fix, evaluation (`tables.txt`, `summary.json`, `rows.jsonl`, `chosen.json`).
- `stageB/`: blob cache, solver replays (`replay.py`), tables (`tables.txt`, `summary.json`), post hoc checks (`posthoc.txt`).
- `verify_i03_stageA/`, `verify_i03_stageB/`: independent re-implementations and comparisons.

**Reproduce.** From the repository root: `PYTHONPATH=src python results/still-scenes/perception/depth-blob-detector/stageB/tables.py` rebuilds `stageB/tables.txt` and `summary.json` from the stored rows (identical); `verify_i03_stageB/v_tables.py`, `v_extra.py`, `v_extra2.py`, `v_recordcmp.py`, `v_roles.py`, `stageA/check_matching.py` and `verify_i03_stageA/v_fresh.py` also reproduce their stored outputs. The solver replays reproduce theirs byte for byte: `stageB/replay.py step0|arms ph|old|new [model]` (all 24 rows and step0 files), `stageB/posthoc.py`, and `verify_i03_stageB/v_replay.py ph|old|new` (all 12 row files). `verify_i03_stageB/v_blobs.py` and `verify_i03_stageA/v_vlm.py` reproduce theirs apart from the scene-folder paths. The stored files name scene folders as recorded, and the readers map them with the harness's `relocated()`. Blocked here: (a) steps that need the rendered fit seeds (`stageA/fit_grid.py`, `fit_fix.py`, `selfcheck.py`, `fix_analysis.py`, `verify_i03_stageA/v_fit.py`, `v_scorer_xcheck.py`): the fresh seeds keep only their json files, no images or depth maps; (b) steps that check hash files (`*.sha256`) which are not in this folder (`stageA/evaluate.py`, `stageB/blobcache.py`). They use the step-0 harness (`results/still-scenes/analysis-scripts/harness/`) and the scenes in `data/dev_scenes/`.

**Data.** `results/logs/still-runs.tar.gz` (the `runs/` folder at the repo root) and the development scenes in `data/dev_scenes/` (release dev-scenes-v1). Data files beside the notes (`rows_*.jsonl`, `*.json`): `results/logs/research-data.tar.gz`.