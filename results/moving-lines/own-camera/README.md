# Tracking people with the robot's own camera

**Question.** If the robot tracks people with its own camera (a people detector and a tracker, arm `CTd`) instead of
using the simulator's true positions (`CT2`), does it lose time in line?

**Result.** Not on average. Pooled over 125 cells (Sonnet, DeepSeek, Haiku, Gemini), strict mean(`CTd` − `CT2`) is
+0.004 with a one-sided 95% lower bound of −0.023, above the −0.05 margin, so bar H5 holds. But `CTd` is a little
worse in most cells (69 worse, 33 better), and the mean hides a split by scene: +0.189 in sweet_spot_line2_moving,
−0.114 in cafe_slots. Under the primary scoring the mean is −0.034 (bound −0.062).

**Files.**
- `I5_FROZEN.md`: the plan. `RESULTS_I5.md`: the write-up, with tracking quality per model.
- `i5_score.py`: the scorer, output `i5_score_out.txt`; with `--test` (the earlier runs), `i5_score_test_out.txt`.
- `i5_posthoc.py`: post hoc, the spread of the per-cell differences, output `i5_posthoc_out.txt`.
- `i5_sweet_check.py`, `i5_sweet_check2.py`: post hoc checks of why sweet_spot favours `CTd` (`*_out.txt`).
- `run_i5.sh`: the simulator runs (needs Habitat, a GPU and the detector weights).
- `detector-on-sim-frames/`: the detector and tracker `CTd` uses, and their first checks.

**Reproduce.** From the repository root: `PYTHONPATH=src python results/moving-lines/own-camera/i5_score.py` (add
`--test` for the test run); the post hoc scripts the same way, with no arguments. All reproduce their stored outputs
exactly.

**Data.** `results/logs/moving-runs.tar.gz` (`runs_i5/` and `runs/` in `results/data/moving/`).
