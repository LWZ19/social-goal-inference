# Perception: lifting a clicked pixel, and a depth-blob people detector

**Question.** When the pipeline picks the wrong target or misses people, is the cause the model's choice or the perception step (lifting a pixel to 3D, finding people), and can a cheap geometric repair help?

**Result.** Of 57 wrong-target records (Qwen, GLM, DeepSeek, realistic scenes) the pixel lift explains about half (oracle snap fixes 26 of 53 that have a point, 22 of those become full successes). A depth-blob detector finds 36/42 (real), 35/42 (staffed) and 34/34 (plain) visible people on the evaluation scenes, but the Stage A verdict was "neither"; replaying the solver with its blobs helps GLM mostly on lines.

**Files.**
- `depth-blob-pilot/`: first, rough blob detector on development scenes.
- `depth-blob-detector/`: the frozen detector, its fit (Stage A) and solver replays (Stage B), with independent re-checks.
- `fresh_seeds/`: manifest of the extra scene seeds (100-109) used to fit the detector.
- `lifting-rerun/`: first recount of wrong targets (lifting versus choice) from recorded runs.
- `lifting/`: the pre-registered follow-up on lifting fixes (oracle, V1-V4, front snap).
- `perception-rerun/`: recount of how many missed people the people call never found.

**Reproduce.** See each subfolder. Scripts run from the repository root with `PYTHONPATH=src`. Most read recorded runs through the step-0 harness (`results/still-scenes/analysis-scripts/harness/`) and the scenes in `data/dev_scenes/`; both are in this repository.

**Data.** `results/logs/still-runs.tar.gz` (the `runs/` folder at the repo root) and the development scenes in `data/dev_scenes/` (release dev-scenes-v1). Data files beside the notes: `results/logs/research-data.tar.gz`.