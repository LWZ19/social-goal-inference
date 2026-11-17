# New situations: a bent line and someone at a machine

**Question.** Does the constraint method (C) beat pointing (R) and pointing with worked examples (E) on two
situations it was not developed on: a waiting line that turns a corner, and one person using a cash or ticket machine
(the robot should wait behind them)?

**Result.** Yes, pooled over the six models, on all 12 layouts of each test: on the bent line C beats R by net +284 and
E by +281, and on machine use by +207 and +205 (layouts 12/0/0, p 0.0005 each). Every model is ahead on its own except
GLM-5.3 at the machine, where it is level (C−R −4, C−E +2). With the looser 0.8 m machine rule (reported, no bar) the
pooled margins are +226 and +204, p 0.001.

**Files.**
- `design/`: the design note, the method configs and the 24 scene files (its own README).
- `I13_FROZEN.md`: the plan and the bars, fixed before any model call (amendment 1, on failed calls, before scoring).
- `e4_geometry.py`: the zero-call validity check (geometry baselines with a perfect click).
  `e4_geometry_out.txt` (the 24 test scenes: people matter, the bend is tested, both success areas can be reached),
  `e4_geometry_devtest_out.txt` (straight development lines), `e4_geometry_selftest_out.txt` (`--selftest`).
- `e4_score.py`: the scorer. `e4_score_six_out.txt`: all six models. `e4_score_five_out.txt`: without GLM, scored
  first. `e4_score_test_out.txt`: a check on the development runs (s21/s23, real variant).
- `e4_rescore_0p8.py`: re-scores copies of the runs with the machine rule's near limit at 0.8 m;
  `e4_rescore_0p8_six_log.txt`. `rescore_0p8/`: the five-model copies. Their scores: `e4_score_five_0p8_out.txt`,
  `e4_score_six_0p8_out.txt`.
- `run_e4.sh`: how the runs were started.

**Reproduce.** From the repository root:

```bash
PYTHONPATH=src python results/still-scenes/new-situations/e4_score.py runs/*_s26_e4_*         # e4_score_six_out.txt
PYTHONPATH=src python results/still-scenes/new-situations/e4_score.py results/still-scenes/new-situations/rescore_0p8/runs/*
PYTHONPATH=src python results/still-scenes/new-situations/e4_geometry.py --selftest
PYTHONPATH=src python results/still-scenes/new-situations/e4_geometry.py \
    data/dev_scenes/placeholder_first14/eval_{06,07,08,09,13}_waiting_line_*            # e4_geometry_devtest_out.txt
PYTHONPATH=src python results/still-scenes/new-situations/e4_rescore_0p8.py OUT runs/*_s26_e4_*  # then e4_score.py OUT/runs/*
```

The first four need only the repository and the development scenes in `data/dev_scenes/` (see `results/README.md`).
`e4_rescore_0p8.py`, and `e4_geometry.py` on `design/scenes`, also need the test scenes' pictures and depth images,
which are not in the repository. With them the rescore gives
`e4_rescore_0p8_six_log.txt`, `e4_score_six_0p8_out.txt` and the records in `rescore_0p8/` again, and the geometry
check prints `e4_geometry_out.txt` plus three machine-use summary lines (G3 24/24) that the stored copy lacks.

**Data.** `results/logs/still-runs.tar.gz` (the runs in `runs/`); `results/logs/research-data.tar.gz` (the scene files
and `rescore_0p8/`); the development scenes in `data/dev_scenes/` for the straight-line check.
