# Re-solving only on settled tracks

**Question.** Does skipping a re-solve while people near the line are still stepping up (CTs), and replacing the
goal when someone stands on it, keep the robot in line better than CT?

**Result.** A small probe only: Claude Sonnet 5.5 on seeds 10-12, 15 cells, CTs against CT 5/7/3 wins/losses/ties,
+0.011 share in line (strict), sign p 0.77. Its redraw-spread prediction was never tested, and the forward ratchet of
CT2 later addressed the same failure.

**Files.**
- `CTs_FROZEN.md`: the method, runs and prediction, frozen before any result on these seeds.
- Output: the "CTs vs CT" row of `../updating-vs-once/results_fresh_sonnet55_low.md`.
- Scorer: `../design-and-scoring/summarise_fresh.py`.

**Reproduce.** From the repository root:

    PYTHONPATH=src .venv/bin/python results/moving-lines/design-and-scoring/summarise_fresh.py results/data/moving/runs/sonnet55_low

**Data.** `logs/moving-runs.tar.gz` (`results/data/moving/runs/`).
