# Where the time out of line goes

**Question.** In the development episodes (seeds 0-2), what causes each second the robot is not in line, and where
would a fix have to live: the model's answer, its latency, the re-solve rules, the arm, or the scoring?

**Result.** Of 31,554 failing seconds in 396 episodes, late start (latency 0.28, approach 0.11) and the model's
answer (0.39 together, mostly a spot beside the line or no usable place in turn) dominate. For CT, its largest own
loss is the line running out (0.17 of CT time), then the replace rule (0.06). The analysis also found that R1's score
rested on the scorer (46% of its in-line seconds standing first, 38% beside the line), which led to the strict scoring.

**Files.**
- `FAILURES.md`: the taxonomy, tables, upper bounds, counterfactual checks, cases and surprises.
- `failures.py`: the classifier and table maker (offline solver replays, no model calls).
- `IDEAS_interface.md`, `IDEAS_robotics.md`, `IDEAS_social_eval.md`: ideas drawn from these tables (post hoc leads).

**Reproduce.** From the repository root:

    PYTHONPATH=src .venv/bin/python results/moving-lines/failure-analysis/failures.py             # section 1 and the rest
    PYTHONPATH=src .venv/bin/python results/moving-lines/failure-analysis/failures.py --matrix    # section 2
    PYTHONPATH=src .venv/bin/python results/moving-lines/failure-analysis/failures.py --episode dsv41/cafe/seed0/CT

The tables in `FAILURES.md` were laid out by hand from the script's output at 00:40 on 2 Nov (396 episodes). The run
folders now hold more seed 0-2 episodes (794), so a re-run gives larger counts; on those 396 episodes the class
table's numbers are the same.

**Data.** `logs/moving-runs.tar.gz` (`results/data/moving/runs/`).
