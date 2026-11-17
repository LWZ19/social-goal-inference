# Thinking levels

**Question.** How much of a model's output is hidden thinking, and what does a higher thinking level buy in success and time?

**Result.** Hidden thinking is about 56 to 84% of Qwen's output tokens by stage at low effort. Pooled right-target goal steps: 49 of 52 with no thinking versus 51 of 52 at low. The stage-12 runs behind the 65 versus 74 of 84 comparison are not on the run list, so that row cannot be rebuilt.

**Files.**
- `thinking.py / thinking_tables.txt`: the tables T1 to T4 and per-stage call times
- `verify/`: independent recomputation

**Reproduce.** From the repo root: `SGI_CACHE_ONLY=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src python results/still-scenes/thinking-levels/thinking.py`. The scripts import the harness in `results/still-scenes/analysis-scripts/harness/` and read the development scenes from `data/dev_scenes/` (release dev-scenes-v1).

**Data.** `results/logs/still-runs.tar.gz` (unpacks to `runs/` at the repo root)
