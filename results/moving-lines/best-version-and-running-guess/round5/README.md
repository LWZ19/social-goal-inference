# Round 5: a belief built on how a waiting line moves

**Question.** Can the line's own movement (people waiting only step a short way forward; people who leave walk away
from the counter) make the robot's belief about who is waiting more accurate, offline and with zero calls?

**Result.** The planned version (QB v1) failed its first check: 20% of the walking bouts of people who stay in line
broke its step-up bound, against a bar of 5% (`b_falsefire_out.txt`), so it stopped there. The later variants are
exploratory: walking away from the counter is a strong sign of leaving (it fires on 77-82% of ex-member bouts and
1.1-5.5% of member bouts), but no variant met the bars. The last one (v4) had the best proxy goal on the pilot layouts
(0.283 against 0.232) but lower recall (0.774 against 0.825), and on the design cells its gain in right share was
only +0.006.

**Files.**
- `ROUND5.md`: the plan and its bars, written before the belief was run.
- `qb.py`: the QB belief (variants v1 to v4).
- `a_bouts.py` -> `a_bouts_out.txt`: walking bouts by who walks (description made before the plan's bounds).
- `b_falsefire.py` -> `b_falsefire_out.txt`: check 1, false fires on members.
- `c_eval.py` -> `c_eval_pilot_out.txt`, `c_eval_design_out.txt`: check 2 on the pilot layouts and the design cells.
- `d_recall.py` -> `d_recall_out.txt` and `d2_fires.py` -> `d2_fires_out.txt`: why recall drops.
- `RESULTS_1.md`: the results and how they are read.

**Reproduce.** From the repo root (`D` is `results/moving-lines/best-version-and-running-guess/round5`):

    PYTHONPATH=src .venv/bin/python $D/b_falsefire.py
    PYTHONPATH=src .venv/bin/python $D/c_eval.py pilot
    PYTHONPATH=src .venv/bin/python $D/c_eval.py design

`a_bouts.py`, `d_recall.py` and `d2_fires.py` also take no arguments.

**Data.** `logs/moving-runs.tar.gz` (`runs/dsv41`, `runs/sonnet55_low` for the design cells, `runs_pilot3/dsv41`) and
`logs/research-data.tar.gz` (`human_dump/`, `test-layouts/pilot_dump/`, `test-layouts/pilot_selection.json`). The
scripts use the round 3 and round 4 code next to this folder. `c_eval.py pilot` also reads the generated layout
files in `sim/scenarios/pilot3/`, which are not in the repo: `test-layouts/select_pilot.py` writes them from the layout
candidates in `logs/bulk.tar.gz`.
