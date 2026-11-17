# Round 3: a steadier running guess (W0, CT5) and the line-direction control (CT4P)

**Question.** Does a steadier belief about who waits, the static evidence read over a 5 s window (W0) and a choice
of whom to follow by expected cost (CT5), give the robot more time in line than CT4, and does giving CT4 CT3F's
line direction (CT4P) close its gap to CT3F?

**Result.** Offline on 129 design cells the window halved roster changes (0.148 to 0.076), but on the 24 pilot
layouts (DeepSeek) neither arm gained time in line: strict share from the first service CT4 0.068, W0 0.072, CT5
0.036, against CT3F 0.188 (prediction P4 not held). The follow-ups trace the loss to where the robot goes (the role
arms follow walkers and people off the line) rather than to roster flicker, and the direction control did not help
(CT4P 0.060; Q1 and Q3 not held, CT3F ahead on 14 layouts, behind on 3).

**Files.**
- Plan and offline results: `ROUND3.md` (cost model, CT5, offline measures, what worked and what did not).
- Offline steps on the design cells, each with its `*_out.txt`: `a_flicker.py` (the replay shows the sweep's
  flicker), `b_calibrate.py` (calibration by counting; writes `fitted.json`, or `fitted_walkers.json` with
  `walkers`), `c_eval.py` (variants V0-V4; `window` for W0-W4, `sensw4` for cost sensitivity), `d_flips.py` (who
  flips in and out of the roster), `e_acf.py` (the 5 s decorrelation time), `f_exit_diag.py` (exit evidence; prints
  only, quoted in `ROUND3.md`).
- Shared code: `r3lib.py` (design cells, stress conditions, replay loop), `ct5.py` (the CT5 belief and cost decision).
- Pilot: `PILOT_PREDICTIONS.md` (P1-P5, written before the pilot ran), `g_pilot_predict.py` (the offline
  predictions), `h_pilot_score.py` (scorer), `check_port_ct5.py` (closed-loop port check; output in
  `../round3_check/closed_loop_port_out.txt`).
- Follow-ups: `FOLLOWUPS.md`, `i_followups.py` (X1, X2, PH-A, PH-B), `j_decompose.py` (PH-C: the goal or getting
  there), `k_geometry/` (post hoc geometry checks, one script and output each: direction error by tick and by
  layout, gaps, sides, transit, extra roster members, layouts where every arm scores near zero).
- CT4P: `CT4P.md` (plan, predictions Q1-Q4), `l_ct4p_score.py` (scorer), `CT4P_CHECK.md` (port check, and a scorer
  that printed past its own stop line), `CT4P_DEVIATION.md` (how the port differs from the plan's wording),
  `CT4P_READING.md` (the result).

**Reproduce.** From the repository root:

```bash
PYTHONPATH=src python results/moving-lines/best-version-and-running-guess/round3/h_pilot_score.py runs_pilot3/dsv41
PYTHONPATH=src python results/moving-lines/best-version-and-running-guess/round3/l_ct4p_score.py runs_pilot3/dsv41
PYTHONPATH=src python results/moving-lines/best-version-and-running-guess/round3/c_eval.py window
```

Run folders are named relative to `results/data/moving/`. Scripts that read the line's shape (`i_followups.line_of`:
`i_followups.py`, `j_decompose.py`, `l_ct4p_score.py`, `k_geometry/`) need `sim/scenarios/pilot3/`, which
`../../test-layouts/select_pilot.py` writes from the pilot candidates. `b_calibrate.py` rewrites `fitted.json`.

**Data.** `results/logs/moving-runs.tar.gz` (`runs/dsv41`, `runs/sonnet55_low`, `runs_pilot3`) and
`results/logs/research-data.tar.gz` (the people dumps in `../human_dump/` and `../../test-layouts/pilot_dump/`,
`../../test-layouts/pilot_selection.json`, `fitted*.json`); `results/logs/bulk.tar.gz` for the pilot candidates.
