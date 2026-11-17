# Round 4: choosing a place, and keeping it once in line (CT4PJ)

**Question.** With the line direction already equal, does a rule that goes to the most likely end of the line (PL),
or CT3F's rule of ignoring people behind the robot once it has joined (CT4PJ), close the role belief's gap to CT3F,
and does CT4PJ's gain hold on 72 fresh layouts?

**Result.** PL failed three of its four offline bars (right share 0.297 against W0's 0.342) and was not run in the
simulator. On the 24 pilot layouts CT4PJ lifted strict share in line from the first service from 0.060 (CT4P) to
0.182 (14/4/6, p 0.031), level with CT3F's 0.188. On the fresh layouts the gain halved and failed for the primary
model (DeepSeek 27/16/23, p 0.126, mean d +0.052; Sonnet, reported only, 33/17/21, p 0.033), and CT3F stayed ahead
(0.208 against 0.159; 0.246 against 0.174), so nothing from the test is claimed.

**Files.**
- Plans: `ROUND4.md` (PL and its four bars), `CT4PJ.md` (the CT4PJ control, predictions J1-J5), `CONFIRM_CT4PJ.md`
  (the fresh-layout test on candidates 2000-2449: H1, then H2 parity within 0.05).
- Write-ups: `RESULTS_B.md` (PL on the pilot layouts; CT4 against CT4P offline), `RESULTS_B2.md` (PL on the design
  cells), `RESULTS_C.md` (staying in line once joined), `RESULTS_D.md` (where the belief's false members come from),
  `CT4PJ_CHECK.md` (port check; why J1 failed on its premise), `CONFIRM_RESULT.md` (verdict and follow-ups).
- Scripts, each with its `*_out.txt`: `b_pl.py` (PL; `pilot`, `design`), `c_ct4p_offline.py` (CT4 against CT4P
  offline), `e_ct4p_port.py` and `g_ct4pj_port.py` (port checks), `m_ct4pj_score.py` (J1-J5 scorer; stops after a
  failed J1), `n_ct4pj_release.py` (J2-J5, released after J1's premise failed), `q_confirm_explore.py` (exploratory
  E1-E4 on the fresh runs; also writes `q_confirm_layouts.csv`).
- `a_proxy.py`: the offline goal proxy (prints only; agreement with the solver 0.87-0.93, quoted in `ROUND4.md`).
- `o_confirm_score.py` (H1, H2 scorer) and `p_port_checks.py` (port checks on the fresh runs); their outputs are
  `../../test-layouts/confirm_score_*_out.txt` and `../../test-layouts/confirm_port_*_out.txt`.
- `d_joined/` (`behind.py`, `joined.py`: time joined, and lasts behind the robot while joined; for `RESULTS_C.md`).
- `f_belief/` (`exmember.py`, `lasts.py`, `marg.py`, `named.py`: who the belief wrongly counts as waiting; for
  `RESULTS_D.md`).

**Reproduce.** From the repository root:

```bash
PYTHONPATH=src python results/moving-lines/best-version-and-running-guess/round4/n_ct4pj_release.py runs_pilot3/dsv41
PYTHONPATH=src python results/moving-lines/best-version-and-running-guess/round4/o_confirm_score.py \
    runs_confirm4/dsv41 test-layouts/confirm_selection.json "DeepSeek V4.1 Flash (primary)"
PYTHONPATH=src python results/moving-lines/best-version-and-running-guess/round4/p_port_checks.py \
    runs_confirm4/dsv41 test-layouts/confirm_selection.json test-layouts/confirm_dump
```

Run folders are named relative to `results/data/moving/`; selection files and dump folders relative to
`results/moving-lines/`. The pilot scripts that read the line's shape need `sim/scenarios/pilot3/`, which
`../../test-layouts/select_pilot.py` writes.

**Data.** `results/logs/moving-runs.tar.gz` (`runs_pilot3`, `runs_confirm4`, and `runs/dsv41`, `runs/sonnet55_low`
for the design cells) and `results/logs/research-data.tar.gz` (the people dumps in `../human_dump/`,
`../../test-layouts/pilot_dump/`, `../../test-layouts/confirm_dump/`, and the selection files).
