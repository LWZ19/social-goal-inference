# Test robot told the serving spot

**Question.** How much time in line does the best version (`CT3F`) gain when it is told the true serving spot
(`CT3FT`) instead of using the model's answer, on four models' stored answers?

**Result.** Strict share from the first service rises by +0.079 pooled over 72 layouts (lower bound +0.052; 46/14/12,
p < 0.0001), so both bars hold. Per model: DeepSeek +0.042, Sonnet +0.086, Haiku +0.049, Gemini +0.128 (40/12/19,
p 0.0001). Most of the gain is at the front of the line, where the robot must pick the serving spot itself. So
observing the serving spot comes first among the future-work leads.

**Files.**
- `X1_FROZEN.md`: the plan. `RUN_PROVENANCE.txt`: how the runs were started (stage A only; stage B was not run).
- `x1_score.py`: the scorer, using `../best-version-and-running-guess/round7/c_gaps.py`; output `x1_score_out.txt`;
  with `--test` (the earlier DeepSeek pair), `x1_score_test_out.txt`.

**Reproduce.** From the repository root: `PYTHONPATH=src python results/moving-lines/test-robots/x1_score.py` (add
`--test` for the test run). Both reproduce their stored outputs exactly.

**Data.** `results/logs/moving-runs.tar.gz` (`runs_x1/`, `runs_ceiling4/`, `runs_confirm4/` in
`results/data/moving/`); `results/logs/research-data.tar.gz` (`../test-layouts/confirm_selection.json`).
