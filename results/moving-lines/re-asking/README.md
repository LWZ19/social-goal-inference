# Re-solving against re-asking

**Question.** Does asking once and re-solving (`CT2`) keep the robot in line at least as well as asking the model
again, after every answer (`Rq0`) or every 10 s (`Cq`), and with how many calls?

**Result.** Yes, by a wide margin. On Sonnet's 25 cells (the pooled bar) `CT2` − `Rq0` is +0.305 (lower bound +0.225;
22/2/1) and `CT2` − `Cq` +0.228 (+0.156; 21/4/0), with 4.0 calls per run against 29.0 and 22.4. DeepSeek, left out of
the bar by the gateway-speed check, agrees: +0.174 and +0.220, lower bounds +0.067 and +0.129. Qwen is latency-bound.

**Files.**
- `I10_FROZEN.md`: the plan, with amendments 1-3. `RESULTS_I10.md`: the write-up.
- `i10_score.py`: the scorer, output `i10_score_out.txt`; with `--test` (old seeds 0-2, `CT`), `i10_score_test_out.txt`.
- `gateway_check_series.py`: the gateway-speed check (amendment 3), outputs `gateway_check_dsv41_run1.txt` and
  `gateway_check_dsv41_run2.txt`. It reads the model-call cache and its file times, which are not in this repository.

**Reproduce.** From the repository root: `PYTHONPATH=src python results/moving-lines/re-asking/i10_score.py` (add
`--test` for the test run). The test output reproduces exactly. In the main output only the Qwen rows differ: its
lane was finished after the output was written, so it now has 25 pairs instead of 9 and 8. Every other row and the
pooled bars are unchanged.

**Data.** `results/logs/moving-runs.tar.gz` (`runs_i10/` and `runs/` in `results/data/moving/`).
