# The best rule-based version against a running guess of who waits

**Question.** Which way of keeping track of who is waiting keeps the robot in line best: CT3F, the rule-based best
version (CT3 with the check on everyone behind a step, and the line re-found from the target), or CT4 and its
successors, one running probability over who waits, seeded by the model's answer?

**Result.** Offline, the running guess looked better (clean tracks: right last person 0.40/0.44 against 0.37/0.34
for DeepSeek/Sonnet, empty roster 0.04 against 0.23), and the rule fixes collapsed under stress (bystanders taken as
last 641/435 s with all stresses combined, against CT3's 234/287). In the simulator it did not hold: on the 72 fresh
sweep layouts CT4 tied CT3F on the right last person (14/14/1, sign p 1.0), and on 72 further fresh layouts CT3F
stayed ahead of the best variant CT4PJ (strict share from the first service 0.208 against 0.159, DeepSeek). The
ceiling check puts the ideal place at 0.640 and the true people ahead plus the true counter point at 0.330: much of
the remaining loss is at the front of the line, not in who waits.

**Files.**
- `INTERIM.md`: the offline method and findings (replay fidelity, the rule ladder, stresses, CT4-min).
- `CT4_FROZEN.md`: the frozen code, test and bars for the fresh sweep; `DEVIATION_1.md`: its port check.
- Offline code: `replay_lib.py` (replays on the people dumps and the runs' tracks), `variants.py`, `ct4.py`,
  `stress.py` (perturbations S1-S4), `validate_replay.py` (fidelity).
- Offline outputs: `ladder.py` (`ladder_*_out.txt`), `ladder_gated.py`, `ladder_ct4.py` (`frozen_*_out.txt` go with
  CT4_FROZEN.md; `ladder_ct4_*_out.txt` and their `_v2` are earlier development outputs, from before `ct4.py` was
  frozen), `ct4_variant_run.py` (also development), `ct4_ablate.py`, `diag_clean.py`, `percell.py`.
- Port checks: `check_port.py`, `check_port_solverfed.py` (with outputs).
- People dumps: `dump_humans.py` and `dump_all.sh` make `human_dump/` (simulator Python, robot idle).
- Later rounds, each with its own README: `posthoc/` (why a better last-person guess did not help), `round3/` (a
  steadier guess, and the line-direction control), `round3_check/` (port and band checks), `round4/` (keeping the
  place once in line; the fresh confirmation), `round5/` (a guess built on how a line moves), `round6/` (why the guess
  trails CT3F), `round7/` (the ceiling, and where time in line is lost).
- The sweep's scorer and outputs are in `../test-layouts/`.

**Reproduce.** From the repository root (`B=results/moving-lines/best-version-and-running-guess`):

    PYTHONPATH=src .venv/bin/python $B/validate_replay.py
    PYTHONPATH=src .venv/bin/python $B/ladder.py --stress S1234
    PYTHONPATH=src .venv/bin/python $B/ladder_ct4.py S2
    PYTHONPATH=src .venv/bin/python $B/check_port.py runs_ideas2_check/dsv41 runs_ideas2_check/sonnet55_low

Run folders are named relative to `results/data/moving/`. Some later scripts read the generated layouts in
`sim/scenarios/{pilot3,confirm4,sweep}/`; make them first with the `select_*.py` scripts in `../test-layouts/`.

**Data.** `logs/moving-runs.tar.gz` (`runs/dsv41`, `runs/sonnet55_low`, `runs_ideas2_check`, `runs_sweep`,
`runs_pilot3`, `runs_confirm4`, `runs_ceiling4`), `logs/research-data.tar.gz` (`human_dump/`, the round 3 fits and
the test-layout dumps), `logs/bulk.tar.gz` (the layout candidates).
