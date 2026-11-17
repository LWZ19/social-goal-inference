# Round 6: why the role belief trails CT3F

**Question.** On the fresh confirm layouts, why does CT4PJ (the role belief with its join rules) keep the robot in
line less than CT3F: does its goal flicker, or does it choose the wrong person?

**Result.** CT4PJ switches the person it follows about twice as often as CT3F (6.7 and 7.5 against 2.6 and 3.0
switches a minute while joined, DeepSeek and Sonnet), and these switches come before it leaves the line (lift 2.3
and 2.2). No hold-or-switch rule on its evidence reaches CT3F's accuracy on the confirm logs (best right last 0.57
and 0.56, against 0.66 and 0.71); most of the gap is at the front, where CT3F knows nobody is ahead 2.4 and 2.6
times as often. Exploratory; nothing here is a claim.

**Files.**
- `ROUND6.md`, `ROUND6_B.md`, `ROUND6_C.md`: the questions and how each answer is read, written before each part.
- `a_jumps.py` -> `a_jumps_pilot_out.txt`, `a_jumps_confirm_out.txt`: part A, goal jumps and leaves.
- `b_pred.py` -> `b_pred_pilot_out.txt`, `b_pred_confirm_out.txt`: part B, committing in the wrong place, and a
  frozen predecessor.
- `c_tradeoff.py` -> `c_tradeoff_pilot_out.txt`, `c_tradeoff_confirm_out.txt`: part C, stability against accuracy.
- `d_none.py` -> `d_none_pilot_out.txt`, `d_none_confirm_out.txt`: a scoring check on part B (nobody ahead), added
  after part C.
- `RESULTS_1.md` (parts A and B) and `RESULTS_2.md` (the correction to part B, and part C).

**Reproduce.** From the repo root (selection files relative to `results/moving-lines/`, run folders relative to
`results/data/moving/`):

    PYTHONPATH=src .venv/bin/python results/moving-lines/best-version-and-running-guess/round6/a_jumps.py \
        test-layouts/confirm_selection.json runs_confirm4/dsv41 runs_confirm4/sonnet55_low
    PYTHONPATH=src .venv/bin/python results/moving-lines/best-version-and-running-guess/round6/a_jumps.py \
        test-layouts/pilot_selection.json runs_pilot3/dsv41

`b_pred.py`, `c_tradeoff.py` and `d_none.py` take the same arguments.

**Data.** `logs/moving-runs.tar.gz` (`runs_pilot3/dsv41`, `runs_confirm4/dsv41`, `runs_confirm4/sonnet55_low`) and
`logs/research-data.tar.gz` (`test-layouts/pilot_selection.json`, `test-layouts/confirm_selection.json`).
