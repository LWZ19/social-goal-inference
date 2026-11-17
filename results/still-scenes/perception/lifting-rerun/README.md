# Lifting rerun: wrong targets, lifting versus choice

**Question.** For records whose chosen target is wrong, is the clicked pixel lifted to the wrong place (occluder, background), or did the model choose the wrong thing?

**Result.** 57 wrong-target records (Qwen 7, GLM 28, DeepSeek 22; 53 have a point). An oracle snap to the nearest body or target pixel within 15 px fixes 26 of 53: person 15/15, parcel 6/15, place 5/23. The rest are choice errors. Lift reproduction error is 0.0 m.

**Files.**
- `lifting.py` -> `lifting_out.txt` (tables) and `lifting_records.tsv` (per record; not in this repository).
- `frontsnap.py` -> `frontsnap_out.txt`: a zero-call front-surface snap test.
- `same_points.py`, `probe_criteria.py`, `crops.py`: follow-up checks (`same_points_out.txt`, `probe_criteria_out.txt`, `crops_out.txt`).
- `verify/`: independent re-counts.

**Reproduce.** `PYTHONPATH=src python results/still-scenes/perception/lifting-rerun/lifting.py > lifting_out.txt`. It reads runs and scenes through the step-0 harness (`results/still-scenes/analysis-scripts/harness/`) and `data/dev_scenes/`. Re-run here, `lifting.py`, `frontsnap.py`, `same_points.py`, `probe_criteria.py` and `verify/v_counts.py`, `v_n2.py`, `v_n4.py` reproduce the stored outputs exactly; `verify/v_lift.py realistic` differs in a few numbers printed with one fewer trailing zero. `lifting_records.tsv` is not in this repository and is rewritten by `lifting.py`.

**Data.** `results/logs/still-runs.tar.gz` (the `runs/` folder at the repo root) and the development scenes in `data/dev_scenes/` (release dev-scenes-v1). `lifting_records.tsv` (a data file beside the notes) is missing from the archives.