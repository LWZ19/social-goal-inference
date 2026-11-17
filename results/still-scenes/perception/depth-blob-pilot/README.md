# Depth-blob pilot (rerun)

**Question.** Do the pilot's reported numbers for a simple depth-blob people detector (recall, false blobs, facing) hold up on development scenes?

**Result.** Counting only people inside the image, recall is 34/34 (plain), 36-42 of 42 (real) and 34-39 of 42 (staffed) depending on the split rule the pilot did not state; placeholder scenes give 42/42. Facing from the blob axis (modulo 180 degrees) lands within 45 degrees for about 0.68-0.70 of people.

**Files.**
- `blobpilot.py`: the detector and tables; `blobpilot_tables.txt`, `blobpilot_diag.txt` are its outputs.
- `diag.py`: recall by split rule and by subset (`blobpilot_diag.txt`).
- `list_scenes.py`, `probe_one.py`: scene listing and a single-scene probe (`list_scenes.txt`, `probe_one.txt`).
- `verify/`: an independent re-implementation and checks (`v_blobs.py`, `v_match.py`, `v_extra.py`, `v_list.py`, `v_probe.py`) with outputs and `per_scene/` json.

**Reproduce.** `PYTHONPATH=src python results/still-scenes/perception/depth-blob-pilot/blobpilot.py > out.txt` (and `diag.py`, `list_scenes.py`, `probe_one.py`, `verify/*.py` the same way). They read runs and scenes through the step-0 harness (`results/still-scenes/analysis-scripts/harness/`). Re-run here, `blobpilot.py`, `diag.py`, `probe_one.py` (up to load times) and `verify/v_*.py` reproduce the stored text; `list_scenes.py` differs only in the scene-folder path it prints; `verify/v_blobs.py` rewrites `per_scene/*.json` with only the `folder` path changed.

**Data.** `results/logs/still-runs.tar.gz` (the `runs/` folder at the repo root) and the development scenes in `data/dev_scenes/` (release dev-scenes-v1).