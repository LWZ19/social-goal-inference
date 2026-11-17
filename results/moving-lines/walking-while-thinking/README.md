# Moving while the model thinks

**Question.** If the robot starts towards the plan's target point as soon as the first, fast part of the answer
arrives (CT2m, and Rm for the marked-point arm), does it get in line sooner and spend more time in line?

**Result.** No gain on any of three models, strict scoring, CT2m against CT2: Claude Sonnet 5.5 12/27/11
wins/losses/ties (+0.002), DeepSeek V4.1 Flash 7/10/8 (-0.012), x-test-3 3/5/7 (-0.010). Rm against R1 is level
too. The interim stop helps only when the plan's target point predicts the place, which for a line is its back, not
the target; why is in `../questions/README.md` (Q1).

**Files.**
- `CT2m_FROZEN.md`: the plan and predictions, frozen before the method was built and before any fresh-seed result.
- Outputs: the "CT2m vs CT2" and "Rm vs R1" rows of `../updating-vs-once/results_fresh_*.md`.
- Scorer: `../design-and-scoring/summarise_fresh.py`. Follow-up checks: `../questions/ct2m_*.py`.

**Reproduce.** From the repository root:

    PYTHONPATH=src .venv/bin/python results/moving-lines/design-and-scoring/summarise_fresh.py results/data/moving/runs/sonnet55_low

(and `.../runs/dsv41`, `.../runs/x3`); see `../updating-vs-once/README.md` on later episodes in those folders.

**Data.** `logs/moving-runs.tar.gz` (`results/data/moving/runs/`).
