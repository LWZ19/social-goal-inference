# Roles against the marked spot

**Question.** When people move, does re-solving the model's roles ("come after people 4, 5 and 6", arm `CT2`) keep
the robot in line better than following the model's marked spot with the person nearest it (`RTw`)?

**Result.** Yes. Pooled over 125 cells (Sonnet, DeepSeek, Haiku, Gemini), `CT2` beats `RTw` on strict share 87/31/7,
mean +0.197, p < 0.0001, so bar H4 holds (primary scoring 87/36/2, +0.225). Gemini alone does not (10/15/0). `RTw`
fails because the person its spot was marked beside is served and walks off: this happened in 117 of 140 episodes.
Post hoc, `RTw` is only a little better than asking once (`R1`): 46/5/74, +0.051.

**Files.**
- `I4_FROZEN.md`: the plan, written before the runs, with amendment 1 (both arms re-run together).
- `i4_score.py`: the scorer, output `i4_score_out.txt`; with `--test` (old seeds 0-2, `CT`), `i4_score_test_out.txt`.
- `i4_posthoc_r1.py`: post hoc `RTw` against `R1` (taken from the earlier runs), output `i4_posthoc_r1_out.txt`.
- `run_i4.sh`: the simulator runs, replayed from stored answers (needs Habitat).

**Reproduce.** From the repository root:
`PYTHONPATH=src python results/moving-lines/roles-vs-marked-spot/i4_score.py` (add `--test` for the test run);
`i4_posthoc_r1.py` the same way. Both reproduce their stored outputs exactly.

**Data.** `results/logs/moving-runs.tar.gz` (`runs_i4/` and `runs/` in `results/data/moving/`).
