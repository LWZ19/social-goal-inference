# Questions on the moving-people results (analysis, 2 Nov)
# Open questions on CT2: what carries its gain, and why moving early did not help

**Question.** Post hoc, on the fresh-seed runs: why did moving while thinking (CT2m) not get the robot in line
sooner, which of CT2's five parts carries its gain over CT, and is that gain an artefact of the scoring?

**Result.** The forward ratchet (taking any goal more than 0.08 m up the line) carries most of CT2's gain: without it
CT2 keeps +0.007 of +0.084 over CT on Sonnet and +0.015 of +0.078 on DeepSeek, while switching off re-admission and
the vacated place keeps it all (+0.083, +0.085). The ratchet's gain is a band-edge effect (at a 1.5 m band top CT2
beats CT by only +0.008 and +0.033), but CT2's lead over R1 holds at every band top with the side check (Sonnet
+0.28 to +0.31). Moving early helps only when the plan's target point predicts the place, which for a line is its back.

**Files.**
- Frozen post hoc plans: `CT2_NORA_POSTHOC.md`, `CT2_NORV_POSTHOC.md`, `CT2_NONE_RAT_POSTHOC.md`.
- Q1 (moving early): `ct2m_diag.py`, `ct2m_by_scene.py`, `ct2m_geom.py`, `ct2m_ahead.py`, `ct2_settle.py`, `ct2m_x3.py`.
- Q2 (what carries the gain): `ct2_attrib.py`, `ct2_attrib2.py`, `ct2_noreadmit_cells.py`; scorers of the frozen
  ablations `nora_score.py`, `norv_score.py`, `norv_classes.py`, `none_rat_score.py`.
- Q3 (an interim from a geometry guess): `interim_ceiling.py`.
- Scoring checks: `ratchet_band.py`, `headline_band.py`.
- Each script's output is `<script>_out.txt`. The findings in detail follow below.

**Reproduce.** From the repository root, for any script here (no arguments):

    PYTHONPATH=src .venv/bin/python results/moving-lines/questions/nora_score.py

The run folders now hold later episodes too (DeepSeek's bystander scenes and seeds 15-44), so some counts grow on a
re-run (for example "CT2-nora present in 25/40 cells" against 25/25); on the episodes that existed when each output
was written, the outputs are the same.

**Data.** `logs/moving-runs.tar.gz` (`results/data/moving/runs/`).

## Q1: why CT2m did not bring first-in-line earlier
Scripts: `ct2m_diag.py`, `ct2m_by_scene.py`, `ct2m_geom.py`, `ct2m_ahead.py`, `ct2_settle.py`.

- **The effect depends on the scene.** Sonnet, CT2m against CT2, first in line earlier / same / later:

  | Scene | Earlier / same / later |
  |---|---|
  | cafe_slots | 6/3/0 |
  | room_coffee_cart_slots | 1/7/1 |
  | cafe_line2_moving | 4/1/4 |
  | cafe_line6_moving | 0/2/8 |
  | sweet_spot_line2_moving | 0/0/8 |

  The timing-only floor (CT2 repeat against CT2) has first in line the same in 10 of 13 cells. So the "later" cells are systematic.
- **(a) Partly right, in a sharper form.** The interim never moved the robot farther from the place it finally took (0/49 Sonnet, 0/25 DeepSeek). By the full answer it was a median 0.8 m closer. But the ray points at the plan's target point, and the place is the line's back:
  - **cafe_line6:** the interim stops beside the line, nearer the counter than the place in 10/10 cells (2.5 m against 4.6 m from the target). It saves 0.1 m and leaves the robot needing an 88° turn, against 37° from the start.
  - **sweet_spot:** the straight ray takes the robot around the wrong side of an obstacle. After the answer it drives back past its start (path 4.0 m against 3.3 m, path/distance 1.67) and is 3.3 s later.
  - **cafe_slots:** the place is on the ray (5.0 → 1.8 m), so CT2m is earlier there.
- **(b) Right but harmless.** The person stop fired in 4/49 Sonnet cells and 2/25 DeepSeek cells. There were 0 cut-in samples before the full answer in every cell, so the 2.5 m stop alone was enough.
- **(c) Wrong.** The first re-solve's pose is the same within 0.3 m in 49/49 and 25/25 cells: oracle tracks and the floor map don't depend on where the robot stands. Nor does the re-solver need time to settle: the first re-solve after the answer is already in turn in about 90% of cells.
- **Reading.** Moving while thinking helps only as far as the plan's target point predicts the place. For a line, the place is the line's back, which the target point doesn't give. A straight ray also ignores the navigation route. The net effect on share is about zero (+0.002), made of many small losses (later arrival) and a few larger wins.

## Q2: what carries CT2's gain over CT v1
Scripts: `ct2_attrib.py`, `ct2_attrib2.py`, `ct2_noreadmit_cells.py`. Frozen note: `CT2_NORA_POSTHOC.md`.

- **Path memory carries nothing on these scenes.** All 5 evaluation lines are straight, where path memory equals the straight axis.
- **Failure classes, CT → CT2:**
  - "Line ran out" falls most: Sonnet −215 samples, DeepSeek −173.
  - Of that fall, Sonnet −168 and DeepSeek −131 are in the cells where readmit fired (Sonnet 25/49, DeepSeek 15/25).
- **DeepSeek:** the whole share gain sits in the readmit cells: +0.155 (14/1/0), against −0.037 in the other 10.
- **Sonnet:** the gain is the same with and without readmit: +0.084 (16/7/2) against +0.085 (16/1/7).
  - The no-readmit gain is mostly cafe_line6: +0.220 (4/0/0), where the replace-rule class falls by 70 samples, the forward ratchet's target.
  - But the seeds 10–12 ablation CT2-rat lost no share, because the classes substitute: without the ratchet, replace rule goes 2 → 81 while "line ran out" goes 163 → 64.
- **0.5 m span:** two-person lines get a line direction in CT2 (0 → 80–100% of re-solves). In sweet_spot (no readmit) "line ran out" falls by 128 samples, but in-line time doesn't rise (−6): the failure mode changes, not the outcome.
- **Caveat.** Readmit-fired cells are those where someone left and came back, which is also where the vacated place can act. So this is association, not attribution. CT2-nora (frozen above) is the test.

### Q1 addendum: x-test-3 (seeds 10–12; `ct2m_x3.py`, `ct2m_x3_out.txt`)
- **Timing.** The interim goal arrives at 11–17 s and the full answer at 36–88 s. The robot reaches the interim's stop point long before the answer, then waits.
- **The distance gained does not grow with latency.** At the answer it is 0.1–1.0 m (start → place against interim → place), about the same as for Sonnet. The 2.5 m stop short of the target caps the gain by distance, not by time.
- **Why the frozen prediction fails.** It expected "the largest gain for the slowest model". The interim's benefit is a fixed distance (about 1–1.5 s of travel), whatever the model's latency.
- **sweet_spot again.** CT2 got in line in 2 cells where CT2m never did.

## Q3: CT2g (a speculative interim from a geometry-only queue guess): my critique, 04:45
Script: `interim_ceiling.py`.

- **As specified, a patch.**
  - The guess is a queue rule: the nearest person to the target, a chain within 1 m, stand 1 m behind.
  - The test scenes are ones where geometry is already right 15/15, so a pass is guaranteed and can't measure what a wrong guess costs.
  - Recommended not to run before the report.
- **The idea under it is a method: speculative execution.** Its ceiling, if the guess were right from the plan stage, is the full answer time minus the plan stage time:

  | Model | Plan-to-answer time (median) | Share ceiling |
  |---|---|---|
  | Sonnet | 9.5 s | about +0.10 |
  | DeepSeek | 14.4 s | about +0.16 |
  | x-test-3 | 35.5 s | about +0.39 |

- **A method test needs** a situation-agnostic prior (C5's learned goal distribution, or a fast-model cascade) and distractor scenes. Proposed as future work in the report.

## CT2-nora result (judged against CT2_NORA_POSTHOC.md; `nora_score.py`, `nora_score_out.txt`)
- **P1 fails on both models.** In the readmit cells, "line ran out" did not come back: Sonnet 184 (bar ≥ 301; CT2 217), DeepSeek 137 (bar ≥ 155; CT2 89).
- **P2 fails (DeepSeek).** Share against CT2 is −0.028, with 7 wins and 7 losses.
- **P3 passes (Sonnet):** +0.024. This is trivial, since nothing dropped.
- **P4 passes:** the no-readmit floor is Sonnet −0.016 (mean |d| 0.023), DeepSeek −0.002 (0.005).
- **Reading.** Readmit is not what carries CT2's gain. The log attribution ("the gain sits in the cells where readmit fired") was association; the ablation corrects it.
  - Hypothesis: readmit and the vacated place are redundant fixes for the same failure.
  - Test: CT2-norv, frozen as `CT2_NORV_POSTHOC.md`.
- **Missing cell, now included.** DeepSeek sweet_spot s14 CT2-nora finished at 04:46:56, after the first pass. It is a no-readmit cell; the rescored output includes it.

## CT2-norv result (judged against CT2_NORV_POSTHOC.md; `norv_score.py`, `norv_classes.py`)
- **Q1 passes on both models.** In the readmit cells, "line ran out" comes back almost to CT's level:
  - Sonnet: CT 385, CT2 217, nora 184, norv 365.
  - DeepSeek: CT 220, CT2 89, nora 137, norv 194.
  - So readmit and the vacated place are redundant fixes for that failure class.
- **Q2 and Q3 fail on both models.** Share doesn't drop.
  - Readmit cells, norv against CT2: Sonnet +0.020 (11/10), DeepSeek −0.021 (8/6).
  - Over all cells, norv keeps all of CT2's gain over CT: Sonnet +0.083, DeepSeek +0.085.
- **Where the time goes.** With both fixes off, the time moves from "other re-solve" (all cells, Sonnet 343 → 97) back to "line ran out" (491 → 824). Time in line barely changes (1836 → 1821). The fixes turn one failure into another; they don't buy time in line.
- **Scene pattern (post hoc, small n).** In the two-person lines norv falls back to CT's level:
  - Sonnet line2: CT 0.068, CT2 0.126, norv 0.065. Sonnet sweet_spot: 0.100 / 0.137 / 0.104.
  - DeepSeek line2: 0.000 / 0.055 / 0.009.
  - In line6, slots and room, norv matches or beats CT2.
- **Reading.** Most of CT2's gain over CT does not come from handling a line that runs out. The replace-rule class (CT 358 → 6–14 in every CT2 variant that keeps the ratchet) and the scene pattern point at the ratchet for the longer lines. On 10–12, though, CT2-rat substituted again (replace rule 2 → 81 with share held).
- **Next (frozen, `CT2_NONE_RAT_POSTHOC.md`).** A sanity arm with all five parts off, then the ratchet ablation on all cells.

## CT2-none result (N1 of CT2_NONE_RAT_POSTHOC.md; `none_rat_score_out.txt`)
- **N1 passes.** The CT2 code path with all five parts off reproduces CT:
  - Sonnet: −0.009, mean |d| 0.061, 14/24/11.
  - DeepSeek: −0.011, mean |d| 0.063, 7/6/12.
- **A full-set timing floor.** The same method on the same answers differs per cell by mean |d| ≈ 0.06, and its mean shift stays within 0.011.

## CT2-rat result on all cells (R1 and R2 of CT2_NONE_RAT_POSTHOC.md; `none_rat_score_out.txt`)
- **R1 passes.** Without the ratchet the replace-rule class comes back above CT's level: Sonnet 441 (CT 358), DeepSeek 167 (CT 119).
- **R2 fails on both models; my redundancy prediction was wrong.** CT2-rat keeps almost none of CT2's gain over CT:
  - Sonnet: +0.007 kept, of +0.084. Against CT2: −0.077, 12/30/7.
  - DeepSeek: +0.015 kept, of +0.078. Against CT2: −0.063, 4/13/8.
- **By the frozen reading rule** (N1 holds, R2 fails): **the forward ratchet carries most of CT2's gain.**
- **The 15-cell CT2-rat run had misled.** It kept +0.040 on seeds 10–12 against +0.007 on all 49 cells. 15-cell ablations are too weak to settle this.
- **By scene (post hoc, small n), mean share strict:**

  | Scene | CT | CT2 | Ratchet off | Readmit + vacate off | Carried by |
  |---|---|---|---|---|---|
  | Sonnet, 6-person line | 0.376 | 0.587 | 0.362 | 0.642 | ratchet |
  | Sonnet, cafe_slots | 0.202 | 0.268 | 0.216 | — | ratchet |
  | Sonnet, room | 0.379 | 0.425 | 0.381 | — | ratchet |
  | Sonnet, 2-person line (line2) | 0.068 | 0.126 | 0.131 | 0.065 | readmit and vacate |
  | DeepSeek, line2 | 0.000 | 0.055 | — | 0.009 | (readmit and vacate) |

  In line2 the ratchet isn't needed; readmit and the vacated place carry it. DeepSeek follows the same pattern, with smaller n.
- **What the ratchet is.**
  - CT's replace rule needs a new goal to move 0.3 m or 25° before the robot takes it.
  - As a line moves up in small steps, the robot stays behind and falls out of the gap band.
  - CT2 takes any goal more than 0.08 m up the line, and any goal someone now stands on.
- **Reading for the report.**
  - The research claim, re-solving on the model's roles against asking once (CT/CT2 against R1), is untouched.
  - CT2's improvement over CT v1 is mostly this control fix. In the two-person lines it is the line-ran-out fixes.
  - All parts off reproduces CT (N1), so nothing outside the named parts is involved.

## Is it scorer-shaped? (`ratchet_band.py`, `headline_band.py`; post hoc, zero calls)
- **The ratchet: yes, it is a band-edge effect.**
  - It keeps the robot 0.1–0.3 m closer behind the person ahead. Gaps of 1.2–1.5 m while joined: CT 30%, CT2 9%, CT2-rat 33%.
  - With the band top at 1.5 m: CT2 against CT is +0.008 (Sonnet) and +0.033 (DeepSeek), and CT2 against CT2-rat is −0.015 and −0.009.
- **The headline against R1 with the side check:** robust at band tops 1.2, 1.5 and 2.0 m (Sonnet +0.28 to +0.31, DeepSeek +0.23).
- **Without the side check, Sonnet's gain over R1 shrinks as the band widens:** +0.19 at 1.2 m, +0.06 at 1.5 m, 0.00 at 2.0 m. Sonnet's marked point stands level with the line but beside it. DeepSeek holds +0.17 even at 2.0 m.
- **Reading.** Against R the claim rests on the on-axis criterion, frozen before the fresh seeds. CT2 over CT rests on the 1.2 m band top. Both are added to SUMMARY.md's caveats.
