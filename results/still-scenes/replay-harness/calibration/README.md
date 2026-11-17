# Calibration of the replay on placeholder scenes

**Question.** Does the replay with repaired people match what a real oracle-people run achieves on the same scenes?

**Result.** On 420 paired records the target is identical in 420 of 420, and the rescore equals the recorded verdict (420 of 420 and 420 of 420). The solver now reproduces 212 of 213 recorded goal steps (one moved 5.9 m). Two implementations of the repairs agree on 213 of 213 goal steps.

**Files.**
- `calibrate.py / calibrate2.py`: first implementation and the version using the shared repair code; write `rows.jsonl`, `rows2.jsonl`
- `tables.py / tables.txt`: real gain versus replayed gain per model
- `disagree.py / disagree.txt, people_share.py / people_share.txt, check_cache.py / check_cache.txt`: side checks
- `explore.py, explore2.py`: first look at the data
- `verify/`: an independent re-computation and its tables

**Reproduce.** From the repo root: `SGI_CACHE_ONLY=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src python results/still-scenes/replay-harness/calibration/calibrate2.py`. then `tables.py`. The scripts import the harness in `results/still-scenes/analysis-scripts/harness/` and read the development scenes from `data/dev_scenes/` (release dev-scenes-v1). `calibrate.py` and `calibrate2.py` each take more than 10 minutes.

**Data.** `results/logs/still-runs.tar.gz` (unpacks to `runs/` at the repo root); development scenes in `data/dev_scenes/` (release dev-scenes-v1).
