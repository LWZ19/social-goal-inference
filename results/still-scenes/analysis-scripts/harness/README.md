# The offline harness of the still-scene analyses

**Question.** How do the still-scene analyses read the recorded runs and the development scenes the same way, without
ever touching held-out data or calling a model?

**Result.** One shared harness reads the 153 recorded development runs (10,896 records) and their 68 distinct
development scenes, plus 29 worked-example scenes. Everything else is refused:
- held-out runs and scenes;
- `.cache` and `.env`;
- any run or scene not on the written lists.

It scores a pose with the package's scorer and re-solves a recorded answer with the package's solver. No model is
called, no network is used, and nothing is written.

With the code in this repository, its full check passes (`full_check_out.txt`):
- all 7,479 recorded successes score as successes again, with 0 mismatches;
- 2,253 of the 2,260 solver places are given again. The other 7 move by exactly the distances listed in
  `research_moved.txt`, because of a solver rule added after those runs.

**Files.**
- `research_harness.py`: the harness. Its docstring lists the functions the analyses use (`runs`, `records`,
  `scene`, `score_pose`, `resolve`, `rescore` and others). As a script it checks every listed run.
- `replay_solver.py`: re-solves recorded answers with the current solver. The harness uses its `solve_again`.
- `research_runs.txt`: the recorded runs that may be read, one `RUN_DIR=SCENES_DIR` per line.
- `research_scenes.txt`: the development scenes that may be read, with the worked-example scenes.
- `research_moved.txt`: the solver places that the code at the time of the analyses no longer gave again, each with
  the reason.
- `replay_runs.txt`, `replay_accepted.txt`: the run list and the accepted losses for `replay_solver.py --gate`.
- `full_check_out.txt`: the output of the full check below.
- `tests/test_research_harness.py`: the guard and list tests. Tests that need the runs or the scenes skip when they
  are absent.

**Reproduce.**
- Quick listing:
  `PYTHONPATH=src python results/still-scenes/analysis-scripts/harness/research_harness.py --no-rescore --no-resolve`
- Full check (re-score and re-solve every record; about 45 minutes): the same command without the two flags.
- Tests: `PYTHONPATH=src python -m pytest results/still-scenes/analysis-scripts/harness/tests` (80 tests).

Run from the repository root. The analyses import the harness with
`sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')`, or through `paths.HARNESS` from
`results/_lib/paths.py`.

Many analyses also check the sha256 of their frozen plan, code and data before they run. Those pinned hashes refer to
the original files. The copies in this repository had local paths removed, so those checks fail here and such scripts
stop at the check; their stored outputs are the record.

**Data.**
- The recorded runs in `runs/` come from `results/logs/still-runs.tar.gz`.
- The development scenes go in `data/dev_scenes/`, from the `dev-scenes-v1` release. In that release,
  `placeholder_first14/` holds links to `data/placeholder/eval/`.
- The runs' `config.yaml` files name the scene sets by their old folder. The harness maps that folder to
  `data/dev_scenes/` (`relocated()`).
