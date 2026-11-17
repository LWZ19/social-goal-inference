# The school's models, and older models, on moving lines

**Question.** Does asking once and re-solving (`CT2`) beat deciding once from a marked spot (`R1`) on the school
gateway's Qwen 3.8 and GLM-5.3, and on seven older Claude and Gemini models (seeds 10-14, 25 cells each)?

**Result.** At 92 s Qwen and GLM are latency-bound (20 and 25 of 25 constraint runs with no answer in time), so bar
H9 fails for both and the method itself is not measured. Given 240 s, Qwen's `CT2` beats `R1` 19/1/5 (p < 0.0001);
GLM is still latency-bound (24 of 25; 0/3/22). On the older models `CT2` beats `R1` on 5 of 7 (for example Sonnet 4.6
14/3/8, Gemini 3.1 Flash Lite 22/2/1); not on Haiku 4.5 (5/2/18) or Gemini 3 Flash (13/11/1).

**Files.**
- `I9_FROZEN.md`: the plan, with amendments 1-3 and 2b. `RESULTS_I9.md`: the write-up.
- `i9_score.py`: the scorer. Its outputs:
  - `i9_score_out.txt`: the bar, written while the older models were still running;
  - `i9_score_conservative_out.txt`: the same with `--retried retried_candidates_glm.txt`, at the same time;
  - `i9_score_e3b_out.txt`: the same scorer once every older model had run;
  - `i9_score_long_out.txt`: `--long`, the 240 s secondary;
  - `i9_score_test_out.txt`: an earlier version of the scorer, run before the Qwen and GLM runs.
- `map_retries.py`: which first answers met an HTTP 429, output `retried_soclaas_glm53.txt` (the same list as
  `retried.txt`). It needs the service and run logs, which are not in this repository.
- `retried_candidates_glm.txt`: every arm that may have retried a call. `timeouts.txt`: arms still timed out (none).
- `run_i9.sh`: the simulator runs (live model calls; needs Habitat).

**Reproduce.** From the repository root: `PYTHONPATH=src python results/moving-lines/school-models/i9_score.py` now
prints `i9_score_e3b_out.txt` exactly; `--long` reproduces `i9_score_long_out.txt` exactly. With
`--retried results/moving-lines/school-models/retried_candidates_glm.txt` it prints the conservative table with the
older models complete. In `i9_score_out.txt` and `i9_score_conservative_out.txt` only the older models' rows differ
from a run today; the Qwen and GLM rows and the verdicts are unchanged.

**Data.** `results/logs/moving-runs.tar.gz` (`runs/` and `runs_i9long/` in `results/data/moving/`).
