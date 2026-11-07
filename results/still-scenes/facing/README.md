# Facing: how well the people call gets each person's facing

**Question.** How accurately does each model's people call report which way each person faces, against ground truth?

**Result.** Share of people within 45 degrees (of those with a known yaw): Gemini 0.906, Sonnet 0.809, Qwen 0.765, DeepSeek 0.616, Haiku 0.616, GLM 0.547. Median error is 23.0 degrees for Gemini and 38.5 for GLM.

**Files.**
- `a0_facing.py`: computes the facing metric per model, role, angle and situation
- `a0_tables.txt, a0_pairs.jsonl`: its tables and the matched pairs
- `explore.py, explore2.py`: first look
- `verify/`: an independent re-computation, with extra and typical-case checks

**Reproduce.** From the repo root: `SGI_CACHE_ONLY=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src python results/still-scenes/facing/a0_facing.py`. The scripts import the harness in `results/still-scenes/analysis-scripts/harness/` and read the development scenes from `data/dev_scenes/` (release dev-scenes-v1). Run `verify/v_a0.py` first: it writes `verify/v_pairs.pkl`, which the other `verify/v_*.py` scripts read.

**Data.** `results/logs/still-runs.tar.gz` (unpacks to `runs/` at the repo root); development scenes in `data/dev_scenes/` (release dev-scenes-v1).
