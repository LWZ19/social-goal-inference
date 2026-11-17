# The best version on new layouts

**Question.** On 72 fresh layouts, does the best moving version (`CT3F`) keep the robot in line better than `CT2`,
measured from the first service after the answer arrives (DeepSeek)?

**Result.** Not shown: `CT3F` − `CT2` is 9/2/12, p 0.065, n 23, mean +0.137 (lower bound +0.064), so bar H11 fails.
The slow gateway is the cause: the live first answer was missing in 32 of 72 runs and failed in 1, so only 23 layouts
had a service after delivery. Reported: on whole-episode strict share `CT3F` beats `CT2` 12/2/57, p 0.013.

**Files.**
- `I11_FROZEN.md`: the plan, with amendments. `RUN_PROVENANCE.txt`: how the runs were started.
- `select_i11.py`: picks the 72 layouts from the candidates in `../test-layouts/i11_candidates/`; writes
  `sim/scenarios/i11/` and `../test-layouts/i11_selection.json`. `select_i11.diff`: its only differences from
  `../test-layouts/select_confirm.py`, as frozen.
- `sweep_report_out.txt`: the candidate report (`../test-layouts/sweep_report.py i11_candidates/`).
- `i11_score.py`: the scorer, output `i11_score_out.txt`; on the four models' confirmation layouts,
  `i11_score_test_out.txt`.

**Reproduce.** From the repository root:
- `PYTHONPATH=src python results/moving-lines/best-version-new-layouts/i11_score.py test-layouts/i11_selection.json runs_i11/dsv41 runs_i11/dsv41_live`
- test: `... i11_score.py test-layouts/confirm_selection.json runs_confirm4/dsv41 runs_confirm4/sonnet55_low runs_confirm4/claude_haiku55 runs_confirm4/gemini_fast`
- Both reproduce their outputs; only the selection path in the first line changed (it was `sweep/`).
- `python results/moving-lines/best-version-new-layouts/select_i11.py` rebuilds the same 72 scenario files (sha256
  5f8d31be) and the same `i11_selection.json`; it overwrites them. `sweep_report.py` on a copy of
  `i11_candidates/` reproduces `sweep_report_out.txt` and `report.json` exactly.

**Data.** `results/logs/moving-runs.tar.gz` (`runs_i11/`, `runs_confirm4/`); `results/logs/research-data.tar.gz`
(the selection files and `i11_candidates/report.json`); `results/logs/bulk.tar.gz` (the candidate layouts).
