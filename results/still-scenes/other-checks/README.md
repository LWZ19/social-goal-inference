# Other checks

**Question.** Do some smaller numbers in the still-scene report (solver flags, straight-line approaches, model speed) hold up when recomputed?

**Result.** Recomputed in `flags/` (M1 to M3) and `i44/` (Qwen 3.6 against Qwen 3.8 output speed: median 123 against 69 tokens per second, about 1.8 times faster).

**Files.**
- `flags/` : solver flags on successes and failures, standard-goal misses, straight-line approaches.
- `i44/` : speed and head-to-head comparison of two Qwen versions.

**Reproduce.** See each subfolder.

**Data.** `logs/still-runs.tar.gz`. Uses the step-0 replay harness in `results/still-scenes/analysis-scripts/harness/` and the development scenes in `data/dev_scenes/` (release dev-scenes-v1).
