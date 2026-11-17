# Speed

**Question.** How long do the calls take, and how much time would running steps in parallel save?

**Result.** See the two subfolders: latency per stage and critical-path saving (`latency/`), and how far from frontal people stand (`profile/`).

**Files.**
- `latency/`: latency replay
- `profile/`: camera-angle profile of people in the scenes

**Reproduce.** See each subfolder.

**Data.** `results/logs/still-runs.tar.gz` (unpacks to `runs/` at the repo root); development scenes in `data/dev_scenes/` (release dev-scenes-v1) for `profile/`.
