# Design and scoring of the moving-line tests

**Question.** How is a moving-line test run, and when does a second count as "in line"?

**Result.** Not an experiment of its own: the plan, the frozen scorings and the shared scorers that every moving-line
folder uses. SCORING.md is the primary scoring (frozen before any model episode); SCORING_STRICT.md adds a 0.6 m
side check and a 0.6 m check on the serving spot (frozen before the fresh seeds 10-19); DEVIATIONS.md lists the 11
changes made after the freeze.

**Files.**
- `DESIGN.md`: the plan: question, hypotheses, arms (R1, Rq, C1, Cq, CT, ...), scenes, tracking and re-solve rules.
- `SCORING.md`, `SCORING_STRICT.md`: the two scorings.
- `DEVIATIONS.md`: every change after the freeze, with its reason.
- `run_moving.py`: runs the arms of one model in the simulator (needs the simulator environment and model access).
- `summarise.py`: scores run folders on the development seeds 0-2 (output: `../updating-vs-once/results_dsv41.md`).
- `summarise_fresh.py`: scores the fresh seeds 10 and up under the three scorings, paired by scene and seed
  (outputs: `../updating-vs-once/results_fresh_*.md`). Most moving-line scripts import its `measures` and `sign_p`.
- `noharm_report.py`: CT4_FROZEN.md's no-harm check on the original scenes (output in the run folder,
  `results/data/moving/runs_noharm/dsv41/noharm_report_out.txt`).
- `tracker_service.py`: people tracks from the robot's own camera, as a service for the CTd arm.
- `queue*.sh`, `run_rat.sh`, `dump_*.sh` and the `*.out` files: the launch scripts and console logs of the runs.

**Reproduce.** From the repository root:

    PYTHONPATH=src .venv/bin/python results/moving-lines/design-and-scoring/summarise.py results/data/moving/runs/dsv41
    PYTHONPATH=src .venv/bin/python results/moving-lines/design-and-scoring/summarise_fresh.py results/data/moving/runs/sonnet55_low
    PYTHONPATH=src .venv/bin/python results/moving-lines/design-and-scoring/noharm_report.py results/data/moving/runs_noharm/dsv41

New runs: `cd results/moving-lines/design-and-scoring && ../../../.venv/bin/python run_moving.py --model ENTRY --out
runs_NAME/MODEL ...` (a relative `--out` is created under `results/data/moving/`).

**Data.** `logs/moving-runs.tar.gz` (the run folders, unpacked to `results/data/moving/`); `logs/bulk.tar.gz` (most
launch scripts and console logs).
