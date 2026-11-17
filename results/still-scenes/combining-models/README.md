# Combining models and trials

**Question.** Does asking several trials or several models and combining their answers (fusion), or escalating from a cheap to a strong model (cascade), beat a single model on the realistic scenes?

**Result.** Fusion: any-of-3 trials lifts GLM from 186/252 (per-trial mean 62.0) to 78 of 84 episodes and DeepSeek to 79; pooling the nine samples of Qwen, GLM and DeepSeek reaches 84 of 84 with at least one success but only 22 with all nine. See `fusion/` and `cascade/`.

**Files.**
- `fusion/`: trial and model fusion counts.
- `cascade/`: the cheap-then-strong escalation curves.

**Reproduce.** See each subfolder. Both read recorded runs through the step-0 harness (`results/still-scenes/analysis-scripts/harness/`) and the scenes in `data/dev_scenes/`.

**Data.** `results/logs/still-runs.tar.gz` and `data/dev_scenes/` (release dev-scenes-v1).