# Model-escalation cascade

**Question.** If a cheap model answers first and a strong model is asked only when a signal says the cheap answer is doubtful, how much of the strong model's accuracy is kept, and at what latency?

**Result.** With GLM as the first model (186/252) and Qwen as the strong one (234/252), asking Qwen for the 20% of GLM's goal-step records with the most trial disagreement (28 of 141) lifts successes from 186 to 198, against 193.0 expected from random escalation and 214 for an oracle gate; it costs about 20 s per goal-step record instead of 9.8 s. Reusing the cheap model's plan caps at 207 (212 when the solver is replayed). `explore_out.txt` and `explore2_out.txt` hold the first scans of the run list and the raw solver answers.

**Files.**
- `cascade.py`: the escalation curves, a port of `../../analysis-scripts/cascade_peek.py`.
- `explore.py`, `explore2.py`: scans (`explore_out.txt`, `explore2_out.txt`).
- `verify/`: independent re-checks (`verify.py`, `robust.py`, `robust_new.py`, `replay_s19.py`, `new_extra.py`, `cache_check.py`, `list_runs.py`) and some outputs.

**Reproduce.** `PYTHONPATH=src python results/still-scenes/combining-models/cascade/cascade.py`. It reads runs through the step-0 harness (`results/still-scenes/analysis-scripts/harness/`) and `data/dev_scenes/`. Re-run here, `cascade.py`, `explore*.py` and the `verify/` scripts reproduce the stored outputs, and write the files this repository lacks (`cascade_out.txt`, `verify/replay_s19_out.txt`, `verify/robust_out.txt`, `verify/verify_out.txt`) identical to the original outputs.

**Data.** `results/logs/still-runs.tar.gz` and `data/dev_scenes/` (release dev-scenes-v1).