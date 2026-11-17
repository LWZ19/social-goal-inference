# Why a better guess of the last person did not keep the robot in line (post hoc)

**Question.** On the spent test layouts, CT4 named the right last person more often than CT3F, so why was its robot
in line less of the time?

**Result.** Exploratory, on the DeepSeek sweep runs. CT4 swaps CT3F's empty guesses for people who are not waiting
(right last 0.42 against 0.38; empty 0.04 against 0.21; served or never-queued people 0.20 against 0.07 of ticks).
Its roster flickers (it changes between 0.172 of consecutive re-solves, against 0.051) and the goal follows it (22
moves a minute against 11), so a right guess reaches the robot less often: in line 0.23 of the time when the last is
right, against 0.43. The same flicker shows on Sonnet (0.138 against 0.053; 23 against 9 moves a minute).

**Files.**
- `POSTHOC_1.md`: the questions, written before running, and the results.
- `posthoc1.py`: Q1-Q4 (where wrong guesses go, CT4's belief, whether a right guess reaches the robot); output
  `posthoc1_out.txt`.
- `posthoc1_q5.py`: Q5 (goal and roster jitter) on DeepSeek; output `posthoc1_q5_out.txt`.
- `posthoc1_q5_any.py`: Q5 for any model's sweep run; output `posthoc1_q5_sonnet_out.txt` (Sonnet).
- `REPLICATE_SONNET.md`: the prediction for Sonnet, written before running, and its result.

**Reproduce.** From the repository root:

```bash
PYTHONPATH=src python results/moving-lines/best-version-and-running-guess/posthoc/posthoc1.py runs_sweep/dsv41
PYTHONPATH=src python results/moving-lines/best-version-and-running-guess/posthoc/posthoc1_q5.py
PYTHONPATH=src python results/moving-lines/best-version-and-running-guess/posthoc/posthoc1_q5_any.py sonnet55_low
```

**Data.** `results/logs/moving-runs.tar.gz` (`runs_sweep/dsv41`, `runs_sweep/sonnet55_low`); the layout list
`../../test-layouts/selection.json` is in git.
