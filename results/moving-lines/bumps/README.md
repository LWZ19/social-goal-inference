# How close the robot comes to people

**Question.** How often does each moving version bring the robot within 0.55 m of someone (the clearance limit) or
within 0.3 m (the bodies would touch)?

**Result.** The updating versions come close least: within 0.55 m 2.1-2.8% of the time (`CT`, `CT2`, `CT3F`), against
7.1% for deciding once from a marked spot (`R1`), 8.0% for following the marked spot (`RTw`), 10.2% for re-asking
after every answer (`Rq0`) and 14.5% for deciding once with the constraint method (`C1`). Within 0.3 m: 0.44-0.67%
for the updating versions, against 3.00% for `R1` and 4.57% for `Rq0`.

**Files.**
- `bumps.py`: reads every fresh-seed episode (seed 10 and up) in every run folder, once per model, scene, seed and
  version, and counts the seconds near people; output `bumps_out.txt`.

**Reproduce.** From the repository root: `PYTHONPATH=src python results/moving-lines/bumps/bumps.py`. It reproduces
the stored output exactly.

**Data.** `results/logs/moving-runs.tar.gz` (all run folders in `results/data/moving/`).
