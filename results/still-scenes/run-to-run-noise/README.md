# Run-to-run noise

**Question.** How much do the totals of pointing (R), pointing with worked examples (E) and the constraint method (C)
move between independent repeats of the same still-scene runs, against the gaps between the methods?

**Result.** For DeepSeek V4.1 the method gap is far larger than the noise: over three batches C scores 185, 210 and
219 of 252 (SD 17.6) against R 120, 125, 126 (SD 3.2) and E 126, 138, 137 (SD 6.7), so C−R is +65 to +93 in every
batch. For GLM-5.3 the repeats move a lot, above all for C (170 in the first batch, 105 and 97 in the repeats as
stored), and many of its repeat records failed on the model gateway and were still being retried when the stored
output was made, so its rows are not final.

**Files.**
- `X4_FROZEN.md`: the plan (reported, no bar).
- `configs/`: the configs of the two repeat batches (noise1, noise2), each with its own answer cache;
  `configs.diff`: their differences from the first batch's configs.
- `x4_score.py`: the scorer. `x4_score_out.txt`: the result. `x4_score_test_out.txt`: the check on the first batch
  alone, before the repeats ran (it reproduces GLM 132/123/170 and DeepSeek 120/126/185).

**Reproduce.** From the repository root: `PYTHONPATH=src python results/still-scenes/run-to-run-noise/x4_score.py`.
The DeepSeek rows and the first-batch rows come out as stored. The GLM repeat rows come out from the runs as archived,
which hold more finished GLM records than the stored output (noise1 C 111 of 234, noise2 C 104 of 228).

**Data.** `results/logs/still-runs.tar.gz` (the runs `runs/*_s23_*` and `runs/*_s28_noise*`).
