# Replay harness: recorded runs replayed offline

**Question.** How much of each model's failures can be traced to target finding, people finding or placement, using only recorded model answers and no new calls?

**Result.** Replaying the unchanged solver on the recorded answers reproduces every recorded place (0 records moved). Removing the constraints drops the lines to 0 of 90 for every model; repairing one input at a time (true positions, true facing, added people) changes little for Qwen (228 of 252 stays 228), while the oracle target gives 239.

**Files.**
- `sets-and-definitions/`: the sets, definitions and the three headline tables (solver alone, people-blind standard goal, target-found versus placement)
- `repairs/`: one-input-at-a-time repairs of the people call, with an independent check in `repairs/verify/`
- `calibration/`: the placeholder-scene calibration of the replay against a run with oracle people, with checks in `calibration/verify/`
- `split/`: each failure split into wrong target, outside region, unreachable and heading
- `baselines/`: solver-alone and standard-goal baselines per model and situation, with checks in `baselines/verify/`

**Reproduce.** Each subfolder has its own README with the command. The scripts import the harness in `results/still-scenes/analysis-scripts/harness/` and read the development scenes from `data/dev_scenes/` (release dev-scenes-v1).

**Data.** `results/logs/still-runs.tar.gz` (unpacks to `runs/` at the repo root) for the run folders.
