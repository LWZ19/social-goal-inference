# Qwen 3.6 against Qwen 3.8 (I44)

**Question.** Is Qwen 3.6 (35B) faster, and does it score better, than Qwen 3.8 on the same scenes?

**Result.** On fresh calls Qwen 3.6 outputs a median 123 tokens per second against 69 for Qwen 3.8 (about 1.8 times), with a median call latency 0.6 times as long. The head-to-head win and loss counts per pair of runs and rule (trial, last, first, episode majority, and so on) are in `search_q2.txt`.

**Files.**
- `i44.py`, `i44_tables.txt` : speed tables.
- `search_q2.py`, `search_q2.txt` : head-to-head counts by several rules.
- `explore.py`, `explore2.py` : first looks.
- `verify/` : independent checks (`v0` to `v7`) with outputs.

**Reproduce.** From the repo root: `PYTHONPATH=src python results/still-scenes/other-checks/i44/i44.py`. Uses the step-0 replay harness in `results/still-scenes/analysis-scripts/harness/` and the development scenes in `data/dev_scenes/` (release dev-scenes-v1).

Every script was re-run; the stored tables are reproduced, apart from printed scene paths.

**Data.** `logs/still-runs.tar.gz`.
