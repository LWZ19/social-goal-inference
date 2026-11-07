# Detector people for the first answer (moving lines)

**Question.** If the detector's people replace the model's people call for the first answer (`CTe`), does the robot
get its first answer sooner than with the people call (`CTd`, the same detector tracking afterwards), without losing
time in line? DeepSeek and Qwen, 25 cells each.

**Result.** No: both bars fail. On DeepSeek, `CTe`'s first answer came later (median 47 s against 19 s, later in 21
of 25) and its strict share was lower (0.102 against 0.287, 2/16/7, p 0.001); pooled over 50 cells the mean is
−0.096, lower bound −0.138. Qwen is latency-bound in both arms. Post hoc, measured only after each arm's own answer
arrived, DeepSeek's `CTe` is still worse (5/10/5, −0.144). The speed race was not fair: `CTd` replayed a stored
latency, while `CTe` called the model live.

**Files.**
- `X3_FROZEN.md`: the plan. `RUN_PROVENANCE.txt`: how the runs were started.
- `x3_score.py`: the scorer, output `x3_score_out.txt`; with `--test` (the earlier Sonnet cells),
  `x3_score_test_out.txt`.
- `x3_posthoc.py`: post hoc, where the delay comes from and quality after delivery, output `x3_posthoc_out.txt`.
- `first-check/`: the first 15-episode check on Sonnet (E1c).

**Reproduce.** From the repository root: `PYTHONPATH=src python results/moving-lines/people-finder/x3_score.py` (add
`--test` for the test run); `x3_posthoc.py` the same way. All reproduce their stored outputs exactly.

**Data.** `results/logs/moving-runs.tar.gz` (`runs_x3/` and `runs/sonnet55_low` in `results/data/moving/`).
