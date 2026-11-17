# Run-to-run noise

**Question.** How much do the totals of pointing (R), pointing with worked examples (E) and the constraint method (C)
move between independent repeats of the same still-scene runs, against the gaps between the methods?

**Result.** For DeepSeek V4.1 the method gap is far larger than the noise: over three batches C scores 185, 210 and
219 of 252 (SD 17.6) against R 120, 125, 126 (SD 3.2) and E 126, 138, 137 (SD 6.7), so C−R is +65 to +93 in every
batch. For GLM-5.3, with every retry
finished, C scores 170, 167 and 157 (SD 6.8) against R 132, 136, 133 (SD 2.1) and E 123, 144, 151 (SD 14.6), so C−R
is +24 to +38 in every batch. In both models the method gap holds in every repeat. (An earlier stored output,
`x4_score_out.txt`, was made while many GLM repeat answers were still being retried, and showed lower C totals.)

**Files.**
- `X4_FROZEN.md`: the plan (reported, no bar).
- `configs/`: the configs of the two repeat batches (noise1, noise2), each with its own answer cache;
  `configs.diff`: their differences from the first batch's configs.
- `x4_score.py`: the scorer. `x4_score_final_out.txt`: the result with every GLM retry finished;
  `x4_score_out.txt`: an earlier output made while GLM repeats were still being retried. `x4_score_test_out.txt`: the check on the first batch
  alone, before the repeats ran (it reproduces GLM 132/123/170 and DeepSeek 120/126/185).

**Reproduce.** From the repository root: `PYTHONPATH=src python results/still-scenes/run-to-run-noise/x4_score.py`.
It prints `x4_score_final_out.txt`.

**Data.** `results/logs/still-runs.tar.gz` (the runs `runs/*_s23_*` and `runs/*_s28_noise*`).
