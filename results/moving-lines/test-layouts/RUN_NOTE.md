
The test is pre-registered in `ideas2_1b/CT4_FROZEN.md` (sha256 2df880ff), with `ideas2_1b/DEVIATION_1.md`
(32aab43f). The layouts are `sweep/selection.json`; the scenario files' sha256 is 047f49a5.

## DeepSeek (free gateway)

- `queue_sweep_dsv41.sh`: 72 layouts × arms CT2, CT3, Gl, CT3F, CT4 and R1, one seed per layout, 4 workers.
  - CT2 makes the first answer (about 4 calls a layout). CT3, Gl, CT3F and CT4 replay it from the cache.
  - R1 makes its own marked-point answer (1-2 calls).
  - Output: `runs_sweep/dsv41/`.
- **Third start, stopped after about 3 min, before any episode finished.**
  - On a vacated tick (CT3F takes the place of the last person served), serve repeats the previous answer's
    fields, `last` included, and the log did not carry the `vacated` flag that marks it. Such a tick would have
    been scored with a stale `last`.
  - `vacated` is now logged. The offline score counts a vacated tick as not right; the closed-loop scorer does the
    same.
  - All five fields (`history_t`, `truth`, `roster`, `vacated`, `last`) were checked present on a cached CT3F
    design cell (`runs_debug/fields_check`: 64 re-solves, 2 vacated). Then restarted.

## Sonnet: committed in advance

Sonnet runs on the 40 "ours" layouts (`sweep/jobs_ours.txt`), arms CT2, CT3, Gl, CT3F and CT4, **whatever the
DeepSeek result for C1 is.** It runs after DeepSeek for GPU headroom, and so that a fault in the closed loop is
found before money is spent.

Only a **mechanical fault** in the DeepSeek runs may stop or delay it, defined now:
- re-solve errors;
- a missing `history_t` or `truth`;
- arms that do not replay CT2's first answer from the cache.

Scores are not a reason. A fault found is fixed, stated in a deviation note, and the run then goes ahead.

## Scoring

## Scorer frozen (before any sweep output exists)

## DeepSeek sweep: finished (3 Nov, 00:29-03:33)

- 432 of 432 episodes exited 0.
- **Stale services (mechanical, no effect on the results).** The fourth start's eight services could not bind
  (`Address already in use`): the third start's services (pids 477769-477778, started 00:26:50) were never stopped,
  and every episode of the fourth start was served by them.
  - Their code is the code a fresh start would have loaded. `serve.py` and `resolve.py` were last written at
    00:26:38, before they started; `roles.py` and `solver.py` earlier. The one later change, `experiment.py` at
    00:27:24, is the sim side, which each episode starts afresh.
  - No state crosses episodes: each answer gets its own uuid4 session.
  - Their logs had been written to files the restart deleted; they were copied from the processes' open file handles
    to `runs_sweep/dsv41/serve_actual/` before the services were stopped by pid. The only errors there are broken
    pipes (and one empty body) where the sim closed a connection at an episode's end.
- Mechanical checks (`sweep/check_mech.py`, output `runs_sweep/dsv41/check_mech_out.txt`): no faults. 38,119
  re-solves, all ok and all with `truth`; `history_t` in every photo-time arm; every replaying arm's first answer is
  CT2's, from the cache. Five layouts have a first answer with no target (a model outcome, the same in every arm),
  and seven more an answer with no session (no one to come after), so no re-solves.
- **Scored once** with the frozen scorer: `sweep/score_dsv41_out.txt` (read-only, sha256 3791113d; its header carries
  the scorer's 7cea0c9e, `summarise_fresh.py` ee0d3b9f and `selection.json` 9d7d34da).
  - **C1 fails**: CT4 vs CT3F on the 36 lag-flagged layouts, 14/14/1, p 1.000 (7 layouts without a value).
- **A wording error, not a scoring change.** This note (third start) and the scorer's docstring say only CT3F
  vacates. In fact every arm with the V2 options vacates (`vacate_after_s`): the output shows vacated ticks with
  someone ahead for CT2 (26), CT3 (45), Gl (49) and CT3F (82); CT4 has none. The scorer treats a vacated tick the same
  way for every arm, so no number changes; C1's sensitivity (vacated ticks out of the denominator) only touches CT3F
  there, as stated.

## No-harm run (DeepSeek, original scenes, seeds 20-29): finished 3 Nov ~04:50

- `queue_noharm_dsv41.sh`: 160 of 160 episodes; mechanical checks clean (`runs_noharm/dsv41/check_mech_out.txt`).
- CT4_FROZEN section 4 lists it as reported and names no measure. `noharm_report.py` reports the sweep's
  per-episode measures (the frozen `score_sweep.episode_scores`, unchanged), each arm paired with CT3; output in
  `runs_noharm/dsv41/noharm_report_out.txt`.
  - Strict share in line: CT2 0.341, CT3 0.337, CT3F 0.400 (25/7/8 vs CT3, p 0.002), CT4 0.259 (18/21/1, p 0.75).
  - After the first service: CT3F 0.501 vs 0.421 (p 0.001); CT4 0.310.
  - CT3F has fewer cut-ins (4.6 against 7.5, 12/3).
  - CT3F shows no harm and a gain. CT4 gives up time in line here too, consistent with the post-hoc flicker.

- ROUND3.md section 6's blocked-place rule. From the first service on, the band is free in more than 98.5% of
  robot-idle frames (design scenes and spent "ours") and in 99.2-99.9% of closed-loop samples. The fallback
  credited no sample. So the blocked place is a before-first-service matter, where the solver's 1.0 m own-space
  exclusion bites.
- A flexible standoff (any free spot in the band) would be a shared solver change. It is noted for later, with its
  own test, and is not part of round 3.

## Round 3 pilot (design data, candidates 1000-1199; `sweep/RANGES.md`)

- Stage 1, CT2 only (DeepSeek, about 96 free calls): 24 of 24, no mechanical faults (3 Nov 04:46-04:57).
- CT5 port in src (`sgi.roles.decide`, `STATIC_WIN_S`; serve options `static_win_s` and `role_decision`), applied
  after Sonnet's servers had started. Offline exactness against `ct5.py` (W4) on 129 design cells x 4 conditions:
  70,876 of 70,876 choices agree, worst marginal difference 0.0 (`round3_check/check_port_ct5_out.txt`).

## Sonnet sweep: finished (reported, not a claim)

- Mechanical checks (`check_mech.py`): every first answer ok, every re-solve ok, no faults.
- Scored once with `score_sweep.py` v2 (7cea0c9e); output `sweep/score_sonnet55_low_out.txt`, sha256 048ec526,
  read-only. The scorer lists the 32 layouts Sonnet did not run as "no episode"; expected.
- Right last, lag-flagged "ours": CT4 vs CT3F 5/15/0, sign p 0.041 (CT3F ahead). Clean 5/5; excursion 3/6/1.
  This agrees in direction with DeepSeek's C1 failure and is stronger.
- Against CT3 on right last: CT3F 20/12/8 (p 0.22), CT4 24/16 (p 0.27); Gl and CT2 far behind.
- As on DeepSeek, CT4 has higher recall (0.92-0.93 in every cell) and fewer empty ticks; CT3F has lower capture.
- Share in line after the first service (strict): CT3F 0.223, CT3 0.170, CT4 0.121. CT4 again gives up time in
  line, consistent with the post-hoc flicker (POSTHOC_1).
- Full test suite with the CT5 port: 3418 passed, 47 skipped.

## Round 3 pilot, stage 2 (started 06:43)

- Sim arms W0 and CT5 added (`experiment_oracle.py`: photo-time arms with variant `role_belief`, `static_win_s` 5,
  `role_decision` for CT5; `experiment.py` logs `choice` and `costs`). Applied after the Sonnet run had ended.
- Closed-loop reproduction check (`ideas2_1b/round3/check_port_ct5.py` 34cc1529): CT5 same choice 3531/3532,
  same last 3532/3532; W0 same last 3525/3534 (0.997). Both pass the 0.95 bar. Output
  `round3_check/closed_loop_port_out.txt`, sha256 076cf4bd, read-only.
- Scored once (`h_pilot_score.py` 8bc3e123); output `ideas2_1b/round3/h_pilot_score_out.txt`, sha256 5f8b7d52,
  read-only.
  - P1 goal jumps: CT5 < CT4 NOT held (12/11/1); W0 < CT4 held (17/3/4, p 0.003).
  - P2 roster changes below 0.6 of CT4: held for both (W0 0.59, CT5 0.58).
  - P3 cut-ins: CT5 < CT4 held (8/3/13; 6.4 against 9.1); W0 not below CT4 held.
  - **P4 strict in line from the first service: NOT held for either.** CT5 is worse than CT4 (2/11/11, p 0.022;
    0.036 against 0.068); W0 ties (5/7/12).
  - P5 right last: CT5 < CT4 held (14/5/5).
  - Against CT3F (0.188 strict from the first service): CT5 1/15/8 and W0 1/15/8, both p 0.001. No confirmatory
    test: by the readings fixed in advance, P4 failing means looking at where the robot goes.

  | Arm | right last | a bystander (not in the queue) | a waiter nearer the front | none | behind the robot |
  |---|---|---|---|---|---|
  | CT3F | 0.446 | 0.227 | 0.237 | 0.050 | 0.029 |
  | CT4 | 0.368 | 0.343 | 0.229 | 0.014 | 0.046 |
  | W0 | 0.324 | 0.372 | 0.242 | 0.015 | 0.046 |
  | CT5 | 0.299 | 0.487 | 0.172 | 0 | 0.042 |

## Round 3 control arm CT4P (design data, zero calls; started 07:45)

- The control (`ideas2_1b/round3/CT4P.md` 76f162e6; scorer `l_ct4p_score.py` e8a8f23d, both read-only, written
  before any CT4P run): CT4 with only the direction changed.
- One deviation from the note's wording, for exactness: the note says to build the role arm's Resolver with
  `path_memory` True and `line_span_m` 0.5. That Resolver also decides who has dropped out, which removes people
  from the answer's other constraints (keep out of shared space, distance, visibility, facing). So the line is
  remembered in a second Resolver, built with those options and used for its geometry only. The role arm's own
  Resolver is CT4's.
- Tests first:
  - the path tangent at a bend;
  - nothing but the direction changes on a straight line (whole answer equal, bar `direction_source`);
  - the fallback without a line;
  - the option needs a role belief and a known source.

  101 passed (serve, roles, resolve), plus the generality lint and config tests.
- Sim: arm `CT4P` (photo-time; variant `role_belief`, `role_direction` "path"); episodes now log
  `solve_direction` and `direction_source` for every arm run from here on.
- Order: check_mech, the scorer's direction check (median angle under 15° on high-error layouts), then
  `l_ct4p_score.py` once, saved, hashed, read-only.
- CT4P run (07:44-07:57): 24 of 24 episodes. Mechanical checks clean: all replay CT2's cached answer with the same
  pose and have `history_t`. Direction source: path on 3070 of 3531 re-solves, belief on 461.
- Scored once (`l_ct4p_score.py` e8a8f23d). Output `ideas2_1b/round3/l_ct4p_score_out.txt`, sha256 855b679c,
  read-only.
- **The direction check FAILED** (median 50.0° on the high-error layouts, bar under 15°). The scorer printed Q1-Q4
  after its FAIL line instead of withholding them.
- Investigation (zero calls):
  - the failure is the bar's premise (`ideas2_1b/round3/CT4P_CHECK.md` fa96598e, read-only). Of the 12 high-error
    layouts:
    - 3 are reversed by the answer's own order of the waiting people (path and chord both 150-180°);
    - 2 have no path;
    - 2 have a back tangent that swings between segments;
    - 4 are fixed by the path (for example sweet_1011, 19° against the chord's 63°).

    No direction read from the answer can fix the first two groups. The bar should have been "the path's angle
    below the chord's, on layouts with a path that is not reversed";
  - CT3F's own direction quality is being put on record: CT3F rerun on the 12 high-error layouts into
    `runs_pilot3/dsv41_dircheck` (zero calls), now that `solve_direction` is logged.
- So Q1-Q4 stand as scored:
  - Q1 NOT held: 3/1/1 on the 5 informative high-error layouts, 4 needed;
  - Q2 held: -0.027 on the low-error layouts;
  - Q3 NOT held: 0.42 against the bar 0.55;
  - Q4 held: CT3F ahead 14/3/7, p 0.013;
  - means: CT3F 0.188, CT4 0.068, CT4P 0.060.

- CT3F's own direction, put on record (`round3_check/dircheck_ct3f.py` 16dc9f49, out fb98380a, read-only; CT3F
  rerun on the 12 high-error layouts, `runs_pilot3/dsv41_dircheck`, zero calls, mechanical checks clean):
  - by the same check, CT3F's median is 52.1° against CT4P's 50.0°, so CT3F's own rule fails the bar too;
  - CT3F is exact (0°) on cafe_1031, sweet_1130, cafe_1000, cafe_1002 and cafe_1001, where CT4P is 0-52° off;
  - CT3F is reversed on sweet_1011 (135°, CT4P 19°);
  - both are reversed on the same 3 layouts.

  The point sets differ: CT3F uses its active waiting set, CT4P the belief's roster, whose extra non-waiters pull
  the back point onto the wrong stretch of the path.

## Round 4 control arm CT4PJ (design data, zero calls; started 08:03)

- Port:
  - the Resolver's join test and its "farther back than the robot" test are now public methods
    (`update_joined`, `behind`). This is a refactor with the same behaviour; 58 resolver tests pass;
  - the serve option `role_joined` needs `role_direction` "path". The geometry Resolver records each solve
    (`in_turn` = turn_m ≤ 0.3, `pose_world`) and runs the join test. Once joined, those behind are left out of
    the solver's people, the come_after and the direction points;
  - the logged `roster` stays the belief's own; `joined` and `behind` are logged.
- Tests first:
  - joining, then the back person stepping back past the robot: left out, last 8;
  - the same without the rule: still counted;
  - option errors.

  108 passed (serve, roles, resolve, lint).
- One difference from CT3F, by the spec: CT3F's resolver also runs its join test over the replayed wait, while
  CT4PJ's starts at the first re-solve. The robot does not move before the answer is delivered, so it is not
  expected to matter.
- CT4PJ run (08:03-08:16): 24 of 24 episodes. Mechanical checks clean: all replay CT2's cached answer with the
  same pose and have `history_t`; `behind` is logged on every re-solve. The arm joined in 15 episodes (1577
  re-solves), and someone was left out as behind on 1341.
- Scored once (`m_ct4pj_score.py` 8b82dcaa). Output `ideas2_1b/round4/m_ct4pj_score_out.txt`, sha256 cfa6a5ce,
  read-only.
- **J1 NOT held** (0.049 against a bar under 0.03), so the scorer withheld J2-J5 as designed.
- Investigation (`round3_check/ct4pj_join_check.py` e71af8b4, out 62c316b4, read-only, zero calls):
  - 50 of the 56 violations are on cafe_1031, where the arm never joined. While the sim counts the robot as in
    the queue, the robot is mostly 1.8-4 m from the goal its previous re-solve chose, or that goal was not a place
    in turn. The join test, which needs the robot at a goal in turn for 1 s, cannot fire;
  - the other 6 are on cafe_1081 (no path, so "behind" is by distance to the target, not along the sim's line).
    CT3F has the same kind there and on sweet_1011 (10 in all);
  - the join test rebuilt offline agrees with the logged `joined` on 3441 of 3534 re-solves, with the robot
    interpolated between 1-s samples. The first join matches in 22 of 24 episodes; the 2 others sit at the
    0.35 m radius (0.31-0.32 m and 0.353 m), within interpolation error.

- Candidates drawn after the freeze (08:25): `make_sweep.py --start 2000 --per-base 450`, 900 of 900 made. By
  stratum and base (cafe/sweet): ours 45/46, chain 89/100, mixed 188/161, easy 128/143. Every cell covers its quota,
  so the primary quotas apply and no fallback is needed.
- Selection: `select_confirm.py` 0e65fc0d (from `select_pilot.py`; only the docstring, OUT, CAND, QUOTA and the
  output names changed: `select_pilot_to_confirm.diff` e20e67d2). 72 layouts:
  - ours 24+24, chain 6+6, mixed 3+3, easy 3+3;
  - flags (lag, restless) balanced 12 each in ours and 3 each in chain; 2/2/1/1 in mixed and in easy;
  - all from 2000-2449, 72 distinct seeds.

  Files: `sweep/confirm_selection.json` a675d06d, `sweep/jobs_confirm.txt` 883b37f3, scenario files sha256
  1832d191 (`sim/scenarios/confirm4`), all read-only.
- Dumps: 72 of 72, no errors. DeepSeek CT2 (08:27-08:5x): 72 of 72 episodes, mechanical checks clean. 5 first
  answers are `no_target`, with every call ok (plan and locate), so the model's own outcome and not a fault. By
  §6 those layouts drop out of the comparisons and are counted by the scorer.
- DeepSeek replay arms (CT3F, CT4P, CT4PJ, CT4; 288 episodes) started after CT2.
- DeepSeek replay arms (08:57-11:0x): 288 of 288 episodes, all exit 0. 72 of 72 in each of CT2, CT3F, CT4P, CT4PJ, CT4.
  - check_mech (`confirm_mech_dsv41_out.txt`, da6772ce): faults none; every re-solve ok; the same 5 no_target first
    answers in every arm.
  - Port checks (`round4/p_port_checks.py` 54c11c0e; `confirm_port_dsv41_out.txt`, 21fda87b): CT4P direction
    6807/6808 (0.9999), no fallback with a path; CT4PJ behind 4827/4857 (0.9938; 25 boundary, 5 differ by more),
    no "behind" last. Both PASS. The port check leaves out 6 episodes (5 no_target plus 1 answer without people).
- Sonnet replay arms (CT3F, CT4P, CT4PJ, CT4; 288 episodes, cache only, no calls) started after CT2.
- Sonnet replay resumed (13:27-14:08): 48 of 48, no errors; 72 of 72 in every arm.
  - check_mech (`confirm_mech_sonnet55_low_out.txt`, 264ba49e): faults none, every first answer ok.
  - Cache check (`confirm_cache_sonnet55_low_out.txt`, 2ce28e4d): faults none.
  - Port checks (`confirm_port_sonnet55_low_out.txt`, eb45e31d): CT4P 9914/9914; CT4PJ behind 5380/5427 (0.9913),
    no behind last. Both PASS.
- Scored once per model with `o_confirm_score.py` (f536382f):
  - DeepSeek (`confirm_score_dsv41_out.txt`, bf2f51eb): H1 FAILS, 27/16/23, p 0.126, mean d +0.052 (n 66). H2 not
    claimed: mean d -0.049, lower bound -0.080. CT3F ahead of CT4PJ, 34 vs 17, p 0.024. Means: CT3F 0.208,
    CT4PJ 0.159, CT4 0.119, CT4P 0.107.
  - Sonnet (`confirm_score_sonnet55_low_out.txt`, 74d6b929; reported, not claimed): H1 passes, 33/17/21, p 0.033;
    H2 fails, mean d -0.072, lower bound -0.105. CT3F ahead 44 vs 18, p 0.001.
  - By §8: on the primary model, the pilot's commitment effect did not replicate.

- OW: no solver. The goal is the ideal place from the sim's own line: 0.85 m behind the last person ahead, on the
  axis, facing up the line, or slot 0 when nobody is ahead.
- CT4OT: CT4PJ with the roster replaced at every re-solve by the true members ahead (`oracle_roster`, new in
  sgi.serve, default off, tested), and the target moved to the counter's true point.
- CT4O: CT4OT with the model's target point kept.
- Smoke test on one pilot layout (cafe_1090), strict from the first service: CT4O 0.00, CT4OT 0.29, OW 0.91. On
  that layout the model's target is beside the counter and every method scores 0. CT4OT picks the right last 0.89
  of the time but is in line only 0.29.
- Full run started, 2 workers (memory). Out `runs_ceiling4/dsv41`, log `queue_ceiling4_dsv41.out`.
- Ceiling run finished (22:02 to about 23:3x, 4 workers): 72 of 72 in each of OW, CT4OT, CT4O, CT4PJT; 218 episodes
  this pass (the rest from before the stop), all ok. Scores unread.
  - check_mech (`ceiling_mech_dsv41_out.txt`, 0f7a6a8e): faults none; every re-solve ok; the same 5 no_target first
    answers in every arm (no re-solves there, OW included).
  - Cache and oracle check (`check_ceiling.py` f2d68f7a, `ceiling_cache_dsv41_out.txt` 83d3855a): every arm replays
    CT2's cached first answer from runs_confirm4/dsv41 with the same pose and status; history_t on the photo-time
    arms; CT4O and CT4OT roster = the sim's people ahead (tracked) on every ok re-solve (9683 and 9687); OW roster
    and last = truth on all 9700. The target oracle changes CT4OT's first goal by > 0.05 m in 24 of 66 layouts.
    Faults none.
- Smoke on pilot cafe_1090 (design data, with CT3F beside it; `queue_smoke_ct3ft.out`): both 153 of 153 re-solves
  ok, first answer cached and the same, history_t and co-motion fields logged; the true counter moves CT3FT's first
  goal 3.55 m (the model's target there is beside the counter).
- CT3FT finished 01:19: 72 of 72, exit 0. Scores unread.
  - check_mech (`ct3ft_mech_dsv41_out.txt`, 6f558a06): faults none; every re-solve ok (9537); the same 5 no_target
    first answers.
  - Cache and oracle check (`check_ct3ft.py` 2237eae4, `ct3ft_cache_dsv41_out.txt` 1c429de8): CT2's cached first
    answer with the same pose and status in all 72; history_t and co-motion fields present; the true counter moves
    the first goal against CT3F's by > 0.05 m in 28 of 66 layouts. Faults none.
