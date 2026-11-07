# Solver flags and straight-line approaches (M1 to M3)

**Question.** Do the solver flags separate successes from failures (M1), how many standard-goal misses are fallbacks (M2), and how often does a straight path from the start to the final pose cross a forbidden region (M3)?

**Result.** For M3, pooled over Qwen, GLM and DeepSeek C records, 3.3 percent (25 of 752) cross a forbidden polygon with the strict definition and 9.2 percent (69 of 752) when the robot body is swept along the path. See `m3_output.txt` and the other `*_output.txt` files for M1 and M2.

**Files.**
- `flags_misc.py`, `flags_misc_full_stdout.txt`, `m1_output.txt`, `m2_output.txt`, `m1b_output.txt`, `m2b_output.txt`, `m3_output.txt` : the three computations.
- `m3_variants.py`, `m3_variants_output.txt` : variants of M3.
- `explore*.py`, `explore*_output.txt` : first looks.
- `verify/` : independent re-computations with outputs.

**Reproduce.** From the repo root: `PYTHONPATH=src python results/still-scenes/other-checks/flags/flags_misc.py`. Uses the step-0 replay harness in `results/still-scenes/analysis-scripts/harness/` and the development scenes in `data/dev_scenes/` (release dev-scenes-v1).

Every script was re-run and reproduces its stored output exactly.

**Data.** `logs/still-runs.tar.gz`.
