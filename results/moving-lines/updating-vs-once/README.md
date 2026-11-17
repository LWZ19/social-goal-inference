# Updating the spot as people move, against deciding once

**Question.** Once the model has named who to stand behind, does re-solving the spot on tracked people (CT, and its
second version CT2) keep the robot in line better than acting on one answer (R1, C1)?

**Result.** On fresh seeds, strict scoring, CT2 beats asking once with the marked point (R1): Claude Sonnet 5.5
43/2/5 wins/losses/ties, +0.27 share in line (0.31 against 0.03); DeepSeek V4.1 Flash 19/3/3, +0.23; x-test-3 9/0/6,
+0.08. CT2 also beats the first version CT (Sonnet 32/8/10, +0.08; DeepSeek 18/3/4, +0.08), the frozen primary test
of `CT2_FROZEN.md`. On the development seeds 0-2 (DeepSeek), CT and R1 were level (0.18 against 0.20).

**Files.**
- `CT2_FROZEN.md`: the plan for CT2 (its five parts, arms, ablations), frozen before the fresh seeds.
- `results_fresh_sonnet55_low.md`, `results_fresh_dsv41.md`, `results_fresh_x3.md`: the fresh-seed tables
  (`summarise_fresh.py`).
- `results_fresh_sonnet.md`: an earlier Sonnet table, written when 355 of the episodes existed.
- `results_dsv41.md`: the development seeds 0-2 for DeepSeek (`summarise.py`).
- Scorers: `../design-and-scoring/summarise_fresh.py` and `../design-and-scoring/summarise.py`; the scorings are in
  `../design-and-scoring/`. What carries CT2's gain is in `../questions/`.

**Reproduce.** From the repository root:

    PYTHONPATH=src .venv/bin/python results/moving-lines/design-and-scoring/summarise_fresh.py results/data/moving/runs/sonnet55_low
    PYTHONPATH=src .venv/bin/python results/moving-lines/design-and-scoring/summarise.py results/data/moving/runs/dsv41

The tables were written while runs were still being added. The run folders now also hold later episodes (for
example the bystander scenes and DeepSeek seeds 15-44), so a re-run counts more cells; on the episodes that existed
when each table was written it gives the same table.

**Data.** `logs/moving-runs.tar.gz` (`results/data/moving/runs/`).
