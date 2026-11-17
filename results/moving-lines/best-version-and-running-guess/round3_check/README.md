# Round 3 checks: the ported code, the free place in line, and two failed bars

**Question.** Does the simulator's code for the round 3 and round 4 arms do what the research code does, is the place
behind the last person ever blocked, and why did two bars fail (CT4P's line direction, CT4PJ's J1)?

**Result.**
- The CT5 port is exact offline (17,719 of 17,719 choices agree in each of 4 track conditions) and in the closed loop
  (same choice 3,531 of 3,532; W0 same last 3,525 of 3,534; bar 0.95).
- The place in line is almost never blocked from the first service on: the band behind the last person is free in
  0.992-0.999 of closed-loop samples for every arm, and the fallback rule credited no sample.
- CT3F's own path rule fails CT4P's direction bar too (median error 52.1° against CT4P's 50.0° on the high-error
  layouts).
- CT4PJ's J1 failure (56 of 1,146 joined re-solves) is not a port error: 50 are on one layout where the arm never
  joined; the join test rebuilt from the logs agrees with the logged one on 3,441 of 3,534 re-solves (0.974).

**Files.**
- `check_port_ct5.py`: offline check of the CT5 port against `../round3/ct5.py`; output `check_port_ct5_out.txt`.
- `closed_loop_port_out.txt`: closed-loop check of CT5 and W0, made by `../round3/check_port_ct5.py`.
- `check_band.py`: is the band behind the last person free (part A robot idle, part B closed loop, part C room in
  the band); outputs `a_design_out.txt`, `a_sweep_ours_out.txt`, `a_sweep_other_out.txt`, `b_sweep_ours_out.txt`,
  `b_sweep_other_out.txt`, `c_design_out.txt`, `c_sweep_ours_out.txt`; `bc_done.txt` marks the end of parts B and C.
- `dircheck_ct3f.py`: CT3F's line direction by CT4P's check; output `dircheck_ct3f_out.txt`.
- `ct4pj_join_check.py`: CT4PJ's J1 violations and the join test rebuilt from the logs; output
  `ct4pj_join_check_out.txt`.
- The plans and readings are in `../round3/ROUND3.md` (section 6, the blocked place) and
  `../../test-layouts/RUN_NOTE.md`.

**Reproduce.** From the repository root:

```bash
P=results/moving-lines/best-version-and-running-guess
PYTHONPATH=src python $P/round3_check/check_port_ct5.py
PYTHONPATH=src python $P/round3/check_port_ct5.py runs_pilot3/dsv41      # closed_loop_port_out.txt
PYTHONPATH=src python $P/round3_check/dircheck_ct3f.py
PYTHONPATH=src python $P/round3_check/ct4pj_join_check.py
# check_band.py builds simulator episodes: the simulator's Python, PYTHONPATH=sim, and the generated layout files
PYTHONPATH=sim ~/miniforge3/envs/sgi-habitat/bin/python $P/round3_check/check_band.py B results/data/moving/runs_sweep/dsv41 sim/scenarios/sweep [names...]
```

`dircheck_ct3f.py` and `ct4pj_join_check.py` read the layout files in `sim/scenarios/pilot3/`, and `check_band.py`
those in `sim/scenarios/sweep/`. They are not in git: `../../test-layouts/select_pilot.py` and `select_sweep.py`
write them from the candidates.

**Data.** `results/logs/moving-runs.tar.gz` (`runs_pilot3`, `runs_sweep`) and `results/logs/research-data.tar.gz`
(the people dumps in `../human_dump/` and `../../test-layouts/human_dump/`).
