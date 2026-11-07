# Phase 1: results so far (2026-10-17)

Five zero-call analyses from plan steps 1 and 2: I01's repairs (Steps 0–1), I01's calibration (Step 2), A0 (VLM facing accuracy, I09's step A0), I14 version 0 and the I46 statistics.

How they were made:
- Every number was computed through the step-0 harness (`docs/model_notes/tools/research_harness.py`) from whitelisted development runs.
- An independent verifier then recomputed each analysis with its own code.
- Where the two differ, this page uses the verifier's final value. Where the verifier could not settle a point, both values are given (section 7).
- No model or network call was made. Nothing was tuned. No held-out data was read. Nothing in `src/` or `tests/` changed.

Paths below are relative to this folder (`results/still-scenes/overview/phase-notes/phase1-files/`). Each analysis has its own folder, with its scripts, its tables (`.txt`) and a `verify/` subfolder holding the verifier's code and outputs.

## 0. Verdicts at a glance

| Analysis | Pre-registered outcome | One-line reading (inference) |
|---|---|---|
| I01 Step 0 | Met: 1,301 of 1,301 solver places reproduce | The replay is exact. |
| I01 Step 1 (repairs) | Neither success nor kill | Perception helps GLM and DeepSeek at lines, but by dropping mislocated detections, not by adding missed people. |
| I01 Step 2 (calibration) | Success met (pooled), kill not met | The repair harness reproduces the real oracle-people gain on the placeholder panel, with caveats. |
| A0 facing accuracy | Delivered (no pass/fail of its own) | It sets I09's facing kill bar at 0.806 (Gemini 0.906 minus 10 points). |
| I14 version 0 | S1 and S3 met, S2 and S4 not met; K1 and K2 do not fire | The set story survives its kill tests, but this probe does not support it. |
| I46 statistics | Tasks 1–5 met; the 40-layout kill line does not fire | A sealed set needs at least 20 fresh layouts. |

Shared cautions:
- Realistic sets: arms real, staffed and plain, 252 records per model and method (nobody near 108, lines 90, groups 54). Old wording: Qwen and GLM C from s19, DeepSeek from s20. New wording: Qwen s21; GLM, DeepSeek, Haiku, Sonnet and Gemini s23.
- The C people answers of s21 Qwen and s23 GLM and DeepSeek are cache copies of s19/s20. Their new-wording rows are not independent evidence on perception.
- Within a stage, R, E and C share one target draw. Each cell is a single 3-trial batch. Both episodes of a scene and trial share one people answer.

---

## 1. I01 Steps 0–1: counterfactual repairs (`i01_repairs/`)

### What was done

- **Solver step**: a record with source `constraints` and a recorded pose, people list and constraints (`Record.solver_step`). Only these are re-solved; every other record keeps its recorded verdict.
- **Step 0**: re-solve each solver step with its recorded inputs (`H.resolve`). Reproduced means the place moves at most 1e-6 m.
- **Matching**: the package's `sgi.run._matched_people` (greedy, nearest first, one to one, within 0.5 m, against visible true people).
- **Repairs** (each re-solved with the unchanged solver, then scored by the package scorer):
  - R0: constraints removed.
  - R1: matched detections moved to the true position. R2: true facing for matched detections.
  - R3: unmatched visible true people added as bodies no constraint mentions.
  - R4: unmatched detections dropped, and their numbers removed from every constraint.
  - P: R1 + R2 + R3 + R4.
  - R5: reference constraints, copied from `tests/test_reference_constraints.py` and written against the model's matched numbers. It reads the situation label and the true group, so it is circular.
  - R6: the oracle target on solver steps. R6x adds the standard goal from the oracle target on ok records that are not solver steps.
  - P5: true people with reference constraints. ALL: P5 with the oracle target.
- **Line failure** (RERUN R01): a waiting-line solver step with the right target that fails because the place is outside the region or unreachable.
- **Flip**: a recorded failure that succeeds after the repair. **Break**: a recorded success that fails after it.

### Main tables

Table 1.1. Old wording, successes of 252 (flips / breaks against recorded C). Solver steps: Qwen 143, GLM 141, DeepSeek 157.

| Arm | Qwen | GLM | DeepSeek |
|---|---|---|---|
| C (recorded) | 234 | 186 | 182 |
| R0 | 147 (+0/−87) | 145 (+1/−42) | 128 (+14/−68) |
| R1 | 234 (+0/−0) | 186 (+0/−0) | 186 (+4/−0) |
| R2 | 234 | 187 (+1/−0) | 184 (+4/−2) |
| R3 | 234 | 188 (+2/−0) | 184 (+2/−0) |
| R4 | 236 (+2/−0) | 200 (+15/−1) | 188 (+6/−0) |
| P | 236 (+2/−0) | 202 (+16/−0) | 193 (+15/−4) |
| R5 (circular) | 235 (+2/−1) | 206 (+20/−0) | 206 (+24/−0) |
| R6 | 240 (+6/−0) | 196 (+10/−0) | 197 (+15/−0) |
| P5 | 237 | 216 | 213 |
| ALL | 243 | 231 | 230 |

Table 1.2. Line failures flipped, by repair (old wording).

| Model (n line failures) | R1 | R2 | R3 | R4 | P | P at 1.0 m | R5 | R6 | P5 | ALL |
|---|---|---|---|---|---|---|---|---|---|---|
| GLM (29) | 0 | 0 | 2 | 15 | 15 | 17 | 19 | 2 | 29 | 29 |
| DeepSeek (15) | 2 | 0 | 2 | 4 | 8 | 8 | 9 | 1 | 15 | 15 |
| Qwen (2) | 0 | 0 | 0 | 2 | 2 | 2 | 2 | 0 | 2 | 2 |

New wording, line failures flipped: GLM (n 32) R3 2, R4 19, P 20, R5 24, R6 5, P5 32. DeepSeek (n 17) R3 3, R4 4, P 8, R5 12, P5 17. Haiku (n 11) P 1, R5 11. Qwen (n 2) P 2. Sonnet (n 2) P 0, R5 2. Gemini has no line failures; its line misses are wrong targets.

Successes lost over all solver steps:
- P: GLM 0, DeepSeek 4, Qwen 0. The 4 DeepSeek losses are 3 groups at `eval_11` and 1 line at `eval_09`. In all 4 the people were all found, so only true positions and facing changed. 2 also break under R2 alone. R5 succeeds on all 4.
- Over line solver steps only, DeepSeek P loses 1.
- R3 loses 0 for every model. R4 loses 1 (GLM, a group at `eval_12`).

Table 1.3. Ceilings, net percentage points of 252, against the brief's guesses (I01 Payoff).

| Ceiling | Qwen | GLM | DeepSeek |
|---|---|---|---|
| Perception, best of R1–R4 and P (lower bound) | +0.8 (R4 = P) | +6.3 (P) | +4.4 (P) |
| Guess | +1 | +10 to +11.5 | +6 |
| Oracle target R6, goal steps only | +2.4 | +4.0 | +6.0 |
| Oracle target, whole-record wrong-target fixes | +2.4 (6 records) | +6.3 (16) | +6.7 (17) |
| Oracle target R6x, whole record | +5.6 | +9.5 | +14.3 |
| Guess | +2.8 | +9.5 | +8.7 |
| Reference constraints R5 (circular) | +0.4 | +7.9 | +9.5 |
| ALL | +3.6 | +17.9 | +19.0 |

- The guesses for the oracle target equal the old wrong-target shares among ok records (7, 24 and 22 of 252). So their order (Qwen / GLM / DeepSeek) is inferred, and the like-for-like comparison is the whole-record wrong-target row. That row is below every guess. Whole-record R6x is at or above the guesses, because it also fixes right-target placement.
- On goal steps, R6 fixes 6 of 6, 7 of 14 and 10 of 14 wrong-target records. The oracle target never failed to lift and always scored as the right target.

Table 1.4. New wording, successes of 252 (extra rows).

| Model | C | P | R5 | R6 | Perception ceiling (pp) | R6 / R6x (pp) |
|---|---|---|---|---|---|---|
| Qwen | 228 | 230 (+2/−0) | 230 | 239 | +0.8 | +4.4 / +8.7 |
| GLM | 170 | 191 (+21/−0) | 195 | 184 | +8.3 | +5.6 / +15.5 |
| DeepSeek | 185 | 195 (+14/−4) | 209 | 197 (+13/−1) | +4.0 | +4.8 / +13.5 |
| Haiku | 240 | 241 (+1/−0) | 251 | 241 (+3/−2) | +0.4 | +0.4 / +0.4 |
| Sonnet | 242 | 242 | 244 | 249 | 0.0 | +2.8 / +2.8 |
| Gemini | 214 | 214 | 214 | 240 (+26/−0) | 0.0 | +10.3 / +11.9 |

### Mechanism (verifier's corrected text)

- GLM: all 15 P flips are R4 flips. In each, `come_after` lists an unmatched detection 3.7–4.6 m from every visible true person. 18 of the 29 failures list such a detection (20 list an unmatched detection at any distance).
- GLM's 14 failures that P leaves: 8 state no `come_after`, 1 states an empty one, 3 list only unmatched numbers (2 at 0.90 m, 1 at 0.51 m plus about 4 m), and 2 keep a matched number. R5 fixes 4 of the 14.
- DeepSeek's 8 P flips: 4 through R4 (detections 3.7–4.8 m off), 2 through R3 (`eval_13`), 2 only together. In all 7 that stay, `come_after` lists only some members. In the recorded people, 5 of the 7 have no missed or spurious person and 2 miss one member. R5 fixes 5 of the 7.
- RERUN found such far detections within 25 px of an unfound member. So they are probably real members whose depth fell through, not phantoms (plausible, not proven). Matching counts each as one spurious detection plus one missed person. That is why R4 does the work and R3 does not.
- Effective n is small: GLM's 29 failures come from 19 distinct people answers, and its 15 P flips from 9 answers in 3 layouts (`eval_08`, `09`, `13`). DeepSeek's 8 P flips come from 4 answers. Each pair of R3 flips comes from one answer.

### Sensitivity and checks

- Leaving dropped numbers in the constraints (R4s, Ps) gives the same verdict as R4 and P on every solver step of all 9 cells.
- With matching widened to 1.0 m, P flips 17 of 29 and 8 of 15; R3 flips 0 and 2.
- The verifier replayed all 9 cells, every record, with its own code: 0 disagreements over 2,268 records × 14 arms, plus P5 and ALL. Both codes share the harness, the package functions and the definitional choices, so this confirms the implementation, not those choices.
- Extra (A0-style): share of matched detections within 45 deg of the true facing, over solver steps: Qwen 455 of 595, GLM 283 of 510, DeepSeek 350 of 570, Haiku 358 of 582, Sonnet 483 of 597, Gemini 551 of 608. Section 3 gives the deduplicated measure.

---

## 2. I01 Step 2: calibration on the placeholder panel (`i01_calibration/`)

### What was done

- Runs: s8 for glm53, ornith15, qwen36_35, qwen38 and qwen3vl, 84 records each (420 pooled).
- `s8_conorc` uses true people with VLM constraints. `s8_convlm` uses VLM people with VLM constraints. Both use goal = constraints. All 420 configs were checked.
- The two arms share the plan and context answers and the target in 420 of 420 pairs (cache hits). Both have goal steps in the same 213 records. The 207 records without a goal step have identical verdicts. The constraints calls are fresh draws in both arms.
- Real gain = conorc successes − convlm successes. Replayed gain = P-repaired convlm − convlm (P as in section 1, with the shared `solver_row` code). Share = replayed / real.
- Agreement = share of records where the replayed verdict equals the conorc verdict. It is given on the 213 goal steps (the strict denominator) and on all 420.
- A match is ambiguous when its detection is within 0.5 m of 2 or more visible true people, or its true person is within 0.5 m of 2 or more detections.
- Bootstrap: 2,000 resamples of (scene, trial) clusters, since a people answer is shared by both episodes of a scene and trial.

### Main table

Table 2.1. Calibration per model (n 84 each; goal steps 42–44 each, 213 pooled).

| Model | convlm → conorc | Real gain [95% CI] | P replayed | Share | P agreement, goal steps | All records | Do-nothing agreement, goal steps | Ambiguous |
|---|---|---|---|---|---|---|---|---|
| glm53 | 54 → 59 | +5 [−2, 13] / [−3, 13] | +8 | 1.60 | 40/43 (93.0%) | 96.4% | 74.4% | 4/132 (3.0%) |
| ornith15 | 61 → 63 | +2 [−2, 7] / [−2, 8] | 0 | 0.00 | 38/42 (90.5%) | 95.2% | 90.5% | 0/161 |
| qwen36_35 | 69 → 71 | +2 [−5, 8] / [−4, 8] | +2 | 1.00 | 38/44 (86.4%) | 92.9% | 81.8% | 0/169 |
| qwen38 | 65 → 67 | +2 [−2, 6] / [−2, 6] | 0 | 0.00 | 38/42 (90.5%) | 95.2% | 90.5% | 0/164 |
| qwen3vl | 54 → 65 | +11 [2, 21] / [1, 21] | +12 | 1.09 | 39/42 (92.9%) | 96.4% | 64.3% | 0/104 |
| Pooled | 303 → 325 | +22 [7, 37] | +22 | 1.00 [0.64, 2.10] / [0.66, 2.0] | 193/213 (90.6%) | 400/420 (95.2%) | 171/213 (80.3%) | 4/730 (0.5%) |

Other results:
- Single repairs, pooled replayed gain: R1 −6, R2 +1, R3 +2, R4 +17, P +22 (Ps +22, P at 1.0 m +23). R4 gives +12 of +12 for qwen3vl but only +4 of +8 for GLM.
- Paired flips on goal steps: real wins 32, real losses 10. P wins 24, every one a real win. P reproduces 24 of 32 real wins and 0 of 10 real losses (GLM 8/8 and qwen3vl 13/13 wins; ornith15 1/3, qwen36_35 2/5, qwen38 0/3). So the exact +22 = +22 is partly cancellation.
- Of the 20 goal-step disagreements, 16 have perfect matching. The difference there is conorc's fresh constraints draw against the true people.
- Noise ceiling: conorc agrees with itself across trials in 157 of 176 pairs (89.2%). P's 90.6% sits at it.
- Checks: re-solving reproduces 212 of 213 places in each arm (the one moved place keeps its verdict). P's people list equals conorc's in 213 of 213 goal steps. Two independent implementations agree record by record.

---

## 3. A0: VLM facing accuracy (I09's step A0) (`a0_facing/`)

### What was done

- **What the pipeline uses**: each person gets one of 8 image-relative facing bins, or `unclear`, plus an optional faced point. The world yaw is the bearing to the lifted faced point when it lands at least 0.25 m away. Otherwise it is the camera bearing turned by the bin. Otherwise it is unknown.
- **Unit**: one people answer per (arm, scene, trial), deduplicated. Every key has exactly 1 distinct raw answer. Qwen and GLM come from s19, DeepSeek from s20 (old wording); Haiku, Sonnet and Gemini from s23.
- The cache copies (s21 Qwen, s23 GLM and DeepSeek) are byte-identical to s19/s20 in every shared key and are not counted.
- **Matching**: as in section 1. Facing is scored only on matched people.
- **Error**: |wrap(reported world yaw − true yaw)| in degrees. "Within 45" counts errors of at most 45 deg among known yaws.
- **View class**: the person's angle to the camera. Frontal < 45, oblique 45–75, profile 75–105, back > 105 deg.
- **Intervals**: percentile bootstrap over layouts (2,000 draws), resampling whole layouts.

### Main tables

Table 3.1. Facing accuracy of the yaw the pipeline uses, all matched people.

| Model | n pairs (people, layouts) | Recall | Within 45 [95% CI] | Median error | Unknown | Label alone, within 45 |
|---|---|---|---|---|---|---|
| Gemini 3.8 low | 310 (36, 8) | 0.994 | 0.906 [0.81, 0.98] | 23.0 | 0.000 | 0.906 |
| Sonnet 5.5 low | 300 (36, 8) | 0.962 | 0.809 [0.58, 0.97] | 28.0 | 0.003 | 0.866 |
| Qwen 3.8 | 298 (35, 8) | 0.955 | 0.765 [0.67, 0.86] | 24.6 | 0.000 | 0.809 |
| DeepSeek V4.1 Flash | 295 (41, 12) | 0.891 | 0.616 [0.51, 0.75] | 29.7 | 0.003 | 0.728 |
| Haiku 5.5 | 308 (36, 9) | 0.981 | 0.616 [0.47, 0.78] | 29.4 | 0.052 | 0.747 |
| GLM-5.3 | 267 (40, 11) | 0.842 | 0.547 [0.40, 0.73] | 38.5 | 0.000 | 0.779 |

- On the 98 instances every model found (34 people), the order is the same: Gemini 0.929, Sonnet 0.834, Qwen 0.766, Haiku 0.638, DeepSeek 0.621, GLM 0.575.
- Counting unknown as a miss changes only DeepSeek (0.614), Haiku (0.584) and Sonnet (0.807).

Table 3.2. Within 45 by view class (n pairs per model in brackets).

| Model | Frontal (26–37) | Oblique (57–67) | Profile (134–171) | Back (43–45) |
|---|---|---|---|---|
| Gemini | 0.926 | 0.851 | 0.906 | 0.978 |
| Sonnet | 0.920 | 0.754 | 0.835 | 0.721 |
| Qwen | 0.963 | 0.789 | 0.716 | 0.800 |
| DeepSeek | 0.892 | 0.586 | 0.532 | 0.721 |
| Haiku | 1.000 | 0.655 | 0.488 | 0.833 |
| GLM | 0.643 | 0.548 | 0.463 | 0.744 |

- Distinct people per class: about 3–6 frontal, 8–10 oblique, 19 profile, 5–6 back.
- By role, line members: Gemini 0.979, Sonnet 0.903, Qwen 0.683, DeepSeek 0.515, Haiku 0.510, GLM 0.460 (n 113–144, 16 people). Profile is not the same set as line members: 18 of 144 line-member pairs fall outside 75–105 deg.
- Label confusion: every model folds away_right and away_left into away_from_camera. Qwen flips left and right (13 + 22 pairs). DeepSeek reports 41 true "right" people as towards_right.

### I09's bars

- I09's facing kill: within 45 under 70%, or more than 10 points below the best VLM. The best VLM is Gemini at 0.906, so the detector's bar is 0.806. With the school panel only (best Qwen 0.765), the absolute 0.70 binds. Gemini's label-alone score is also 0.906, so the bar is the same either way.
- Comparison caveat (verifier): Gemini's 0.906 comes from the people Gemini matched, in the 8 layouts where a people call ran. A1's detector will see all 42 images. So A1 must be scored on the same (arm, scene, person) instances (`a0_pairs.jsonl`) or on common support. Gemini's lower CI bound (0.81) sits at the bar.
- I09's success bars applied to the VLMs, for reference only: only Gemini meets all three (non-profile median 22.4 deg, non-profile within 45 0.906, profile front/back correct 0.994). GLM, DeepSeek and Haiku miss all three. Read as a people source, three VLMs fall below the absolute 70% themselves.

### Post-hoc finding: the faced point against the label

- The faced point sets the yaw for 65–91% of people. On those pairs it does worse than the model's own 8-bin label for 5 of 6 models: Qwen 0.796 vs 0.862 (n 196), GLM 0.539 vs 0.794 (243), DeepSeek 0.618 vs 0.749 (251), Haiku 0.693 vs 0.884 (199), Sonnet 0.852 vs 0.936 (203). Gemini ties (0.926, n 270).
- The faced point turns the yaw toward the camera by a median of +12 to +25 deg for Qwen, Haiku, Sonnet and Gemini. The label alone is unbiased (−1.6 to +1.6).
- On the placeholder panel (s8 convlm) the label is better for glm53 (0.805 vs 0.571) and qwen38 (0.844 vs 0.781), and worse for qwen36_35 (0.663 vs 0.753) and qwen3vl (0.661 vs 0.750).
- This is a diagnostic of the existing pipeline on development sets. Changing the heading rule would need pre-registering and testing on other sets.

---

## 4. I14 version 0: scoreboard for sets, with chance baselines (`i14_v0/`)

### What was done

- Records: 1,409 goal steps (the only records with runners-up). Primary cells: old-wording realistic (Qwen s19 n 143, GLM s19 n 141, DeepSeek s20 n 157) and watching (dev seeds 0–2, audience goal steps, n 36 each). The new wording is reported beside them.
- Places: place 1 is the solver's top place; places 2–5 are the recorded runners-up. **hit@k**: any of places 1..k succeeds under the package scorer. hit@1 is official top-1. **Gain** = any-of-5 − top-1.
- Floor: the solver's selectable cells for that record, built with the record's own VLM people (no oracle people). **Disc**: floor cells within 3 m of the recorded target point.
- Baselines (1,000 draws per record; cells face the target): R5 = 5 random disc cells; F5 = farthest-point 5; AR = top place + 4 random disc cells; AF = top place + 4 farthest-point disc cells.
- **Recovery** = (baseline any-of-5 − top-1) / (solver any-of-5 − top-1).
- **Lift_disc** = top-1 / disc share of the region; **Lift_room** = the same over all floor cells; **chance ratio** = disc share / room share (ratios of sums).

### Main tables

Table 4.1. Runner-up headroom and baselines, primary cells (expected any-of-5 hits for baselines).

| Cell (n) | Top-1 | Any-of-5 | Gain | R5 | AR (recovery) | AF (recovery) |
|---|---|---|---|---|---|---|
| Qwen realistic (143) | 134 | 134 | 0 | 62.9 | – | – |
| GLM realistic (141) | 96 | 99 | +3 | 61.3 (exact 61.5) | 104.0 (267%) | 107.0 (367%) |
| DeepSeek realistic (157) | 109 | 123 | +14 | 63.1 | 122.4 (96%) | 113.0 (29%) |
| Qwen watching (36) | 32 | 35 | +3 | 6.6 | 40% (exact) | 33% |
| GLM watching (36) | 23 | 29 | +6 | 6.5 | 40% | 83% |
| DeepSeek watching (36) | 23 | 33 | +10 | 7.2 (exact 7.3) | 26.3 (33%) | 24.0 (10%) |

- hit@k for DeepSeek realistic: 109, 120, 121, 123, 123. Watching: 23, 29, 30, 33, 33.
- New wording, gain and AR recovery: DeepSeek +12 (95%), GLM +3 (about 240%), Haiku +7 (61%); Qwen, Sonnet and Gemini gain 0.
- Layout bootstrap of AR recovery: DeepSeek realistic 96% [65–66, 145–146], watching 33% [4, 59–60].
- Post hoc, drawing over the whole room instead of the disc: AR recovers 57% (DeepSeek realistic) and 69% (DeepSeek watching).

Table 4.2. Lift and chance ratio, primary cells.

| Cell | Lift_disc | Lift_room | Chance ratio |
|---|---|---|---|
| Qwen realistic | 6.90 | 13.26 | 1.92 |
| GLM realistic | 4.92 | 10.12 | 2.06 |
| DeepSeek realistic | 5.63 | 10.74 | 1.91 |
| Qwen watching | 19.91 | 5.46 | 0.27 |
| GLM watching | 14.04 | 4.25 | 0.30 |
| DeepSeek watching | 12.53 | 3.94 | 0.31 |

Other measures:
- **Disc coverage.** The 3 m disc leaves out about 90% of the watching region and 10–17% of the realistic one. The top place lies outside it in 470 of 790 line records and 100 of 108 watching records.
- **Skeptic's guess.** Random-5 in the disc lands in the region (heading aside) in 77–80% of right-target group records, not 85–95%.
- **P(success) of the set** (uniform over 5 places): solver 0.42–0.59 on the old-wording realistic cells (0.38–0.62 new wording); random-5 0.12–0.15.
- **Forbidden mass.** The solver's 5 places put at most 0.019 of their mass in a forbidden kind. Random disc places land in audience view 0.24 of the time.
- **Jitter (5 cm, 8 directions).** The top-1 verdict flips in 1, 2 and 6 records (old Qwen, GLM, DeepSeek) and at most 7 in any cell.
- **Heading tolerance 30 / 45 / 60 deg.** DeepSeek realistic top-1 is 101 / 109 / 111. Watching DeepSeek is 23 / 23 / 30. Most cells are flat.
- **Checks.** The scorer replica agrees with the package on 428,091 cells and places (0 disagreements). The verifier scored every selectable cell of all 1,409 records with the package scorer and reproduced every value.

---

## 5. I46 statistics (`i46_stats/`)

### What was done

- **Unit**: the layout (scene id without the arm), 14 per realistic set: 6 nobody near, 5 lines, 3 groups, 18 records each.
- **ICC(1)** of success by layout, one-way ANOVA. Design effect = 1 + 17 ICC. Effective n = 252 / design effect.
- **Intervals**: t(13) on the 14 layout mean differences, paired on (arm, scene, episode, trial).
- **Test**: two-sided exact sign-flip by layout (statistic: the sum).
- **Power**: draw n fresh layouts with replacement from the 14 observed layout differences and apply the exact test at 0.05. This is post-hoc and optimistic: it reuses the layouts that showed the effect.
- **Batch SD**: Qwen's 9-draw pool for R and E (s16b, s22 trials 3–5, s21; s22 trials 0–2 dropped as copies of s16b).

### Main tables

Table 5.1. ICC of success by layout, new wording (R / E / C); design effect in brackets.

| Model | R | E | C |
|---|---|---|---|
| Qwen | 0.264 (5.49) | 0.119 (3.03) | 0.122 (3.08) |
| GLM | 0.350 (6.95) | 0.413 (8.02) | 0.261 (5.44) |
| DeepSeek | 0.390 (7.63) | 0.321 (6.46) | 0.125 (3.12) |
| Haiku | 0.747 (13.70) | 0.678 (12.52) | 0.265 (5.50) |
| Sonnet | 0.718 (13.21) | 0.544 (10.26) | 0.244 (5.15) |
| Gemini | 0.401 (7.81) | 0.389 (7.61) | 0.258 (5.38) |

- Old wording: Qwen 0.290 / 0.114 / 0.080, GLM 0.398 / 0.486 / 0.317, DeepSeek 0.393 / 0.331 / 0.168.
- Over all 27 cells: ICC 0.080–0.747, design effect 2.35–13.70, effective n 18–107 of 252. C is the least clustered method by median (not for every model: new Qwen C 0.122 vs E 0.119).

Table 5.2. New wording, layout t-intervals, records of 252 (n 252 pairs, 14 layouts).

| Model | C − E | C − R | E − R |
|---|---|---|---|
| Qwen | +32 (+8.4 to +55.6) | +49 (+11.9 to +86.1) | +17 (−2.8 to +36.8) |
| GLM | +47 (+17.3 to +76.7) | +38 (+8.2 to +67.8) | −9 (−26.5 to +8.5) |
| DeepSeek | +59 (+20.4 to +97.6) | +65 (+22.4 to +107.6) | +6 (−10.7 to +22.7) |
| Haiku | +92 (+38.1 to +145.9) | +89 (+32.4 to +145.6) | −3 (−16.1 to +10.1) |
| Sonnet | +64 (+14.4 to +113.6) | +64 (+8.0 to +120.0) | 0 (−17.9 to +17.9) |
| Gemini | +41 (+11.0 to +71.0) | +50 (+16.5 to +83.5) | +9 (−8.5 to +26.5) |

- C − E and C − R repeat RERUN's "New rows" (12 of 12 match). C − E exact p is 0.0039–0.0156, each at its floor. No E − R is significant (p 0.125–1.0).

Table 5.3. Power of a sealed set of n fresh layouts, two-sided (exact values where a closed form exists).

| Cells | n = 8 | n = 14 | n = 20 | n = 30 |
|---|---|---|---|---|
| C − E: Qwen, GLM, DeepSeek, Gemini; C − R: Gemini | 0.259 | 0.911 | 0.996 | 1.000 |
| C − E and C − R: Haiku | 0.411 | 0.972 | 1.000 | 1.000 |
| C − E: Sonnet; C − R: Qwen, GLM, Sonnet | 0.145 | 0.788 | 0.979 | 1.000 |
| C − R: DeepSeek (Monte Carlo) | 0.375 | 0.916 | 0.994 | 1.000 |

- In 11 of 12 cells every non-zero layout difference favours C, so power is P(Bin(n, k/14) ≥ 6) with k non-zero layouts (k = 8, 9 and 7 for the three exact rows). The first check's 10,000-draw values agree within Monte Carlo error.
- Smallest n with power ≥ 0.80: 14, or 20 for the four k = 7 cells.
- Why 8 layouts is weak: p ≤ 0.05 needs at least 6 non-tied layouts, and the nobody-near layouts are usually tied.

Table 5.4. Batch noise and minimum detectable differences (80% power, two-sided 0.05; records of 252).

| Quantity | Value |
|---|---|
| Qwen batch totals, R / E | 175 / 188 / 179; 214 / 204 / 196 |
| SD of batch totals, pooled over R and E | 7.93 (df 4, exact 95% CI 4.8 to 22.8) |
| Within-batch estimators for Qwen | 6.20 (trial totals), 5.98 (item binomial) |
| Within-batch noise, other models | 3.9 to 6.7 |
| MDD, two independent 3-trial batches | 19.8 / 23.8 / 27.7 / 31.4 at SD 5 / 6 / 7 / 7.93 |
| MDD, paired within one stage | 11 to 21 (Qwen pool E − R 19.6; 36 with the across-batch SD of 13.0) |
| MDD over 14 layouts (t test) | 18 to 28 for E − R-like contrasts; 33 to 79 for C − E and C − R |

- Per-situation power, for I46's kill line (corrected text): 0 of 24 lines and groups cells need more than 40 layouts. The smallest n is 14 for GLM C − R lines and for the groups cells of Qwen C − R, DeepSeek C − E, DeepSeek C − R, Sonnet C − E and Sonnet C − R. It is 6 for the other 18 cells. This design resamples 3–5 values and is degenerate by construction.
- A change confined to lines (floor p 0.0625) or groups (0.25) cannot reach p ≤ 0.05 on these sets.
- `results/still-scenes/metrics-and-statistics/statistics/HOW_TO_REPORT.md` holds the one-page reporting note. The verifier asks for two fixes, not yet applied: (a) R and E are only nearly identical at nobody-near (identical for Qwen, Haiku, Sonnet and Gemini; DeepSeek differs in 3 of 108 records at s23 and 7 of 108 at s20); (b) "at least 6 layouts per situation" is the test's floor, not a powered size (power at 6 is about 0.09 when one group layout is tied).

---

## 6. Pre-registered criteria and outcomes

### I01 first probe (Steps 0–2)

| Criterion (REPORT I01) | Value | Outcome |
|---|---|---|
| 100% of places reproduced | 1,301 of 1,301 | Met |
| P flips ≥ 15 of 29 (GLM) | 15 (from 9 people answers) | Met, exactly at the bar |
| P flips ≥ 8 of 15 (DeepSeek) | 8 (from 4 people answers) | Met, exactly at the bar |
| R3 alone flips ≥ 10 of 29 | 2 (DeepSeek 2 of 15) | Not met |
| At most 2 successes lost per model | P loses DeepSeek 4 over all solver steps, 1 over line steps; GLM 0, Qwen 0; R3 0 | Not met under the all-situation reading; met under a line-only reading (unsettled, see below) |
| Calibration recovers ≥ half the real gain | Pooled share 1.00 | Met (pooled) |
| ≥ 80% per-record agreement | 90.6% of 213 goal steps; 95.2% of 420 | Met (doing nothing already gives 80.3% and 90.0%) |
| ≤ 10% ambiguous matches | 0.5% (4 of 730) | Met |
| Kill: P ≤ 8 of 29 and ≤ 4 of 15 | 15 and 8 | Does not fire |
| Kill: R3 ≤ 5 while breaking ≥ 3 | R3 flips 2, breaks 0 | Does not fire |
| Kill: calibration share < 1/3 or agreement < 65% | 1.00, 90.6% | Does not fire (pooled) |
| Kill: > 20% ambiguous matches | 0.5% | Does not fire |
| **Overall** | Success needs all of the above | **Neither success nor kill** |

Per model, the calibration verdict is not uniform. glm53 (real +5) and qwen3vl (+11) meet all three success parts. qwen36_35 (+2) meets them by the letter. ornith15 and qwen38 (+2 each) meet the kill's share clause by the letter (share 0.00), on real gains whose CIs include 0 and that come from the constraints redraw, which the repair leaves alone by design.

### A0

| Item | Outcome |
|---|---|
| Facing accuracy per model and situation: within 45, median, unknown, 45-deg confusion, view split | Delivered (six models, 267–310 pairs each) |
| One answer per (arm, scene, trial); cache copies not counted | Met |
| I09 kill and success bars | Not applicable to the VLMs: they reviewer the detector in A1. A0 supplies the bar of 0.806. |

### I14 version 0

| Criterion | Value | Outcome |
|---|---|---|
| S1 exact reproduction (134/96/109, 32/23/23; any-of-5 134/99/123, 35/29/33) | All 6 cells exact; I46 p values and power replay identical | Met |
| S2 anchored random-4 recovers ≤ 50% of +14 and +10 | 96% realistic, 33% watching (post hoc whole room 57%, 69%) | Not met |
| S3 solver beats anchored farthest-4 | DeepSeek 123 vs 113.0, 33 vs 24.0; GLM 99 vs 107 (old), 89 vs 98 (new) | Met on the frozen DeepSeek reading; not met for GLM (unsettled, see below) |
| S4 Lift_disc ≥ 3 and chance ratio ≥ 2 in every primary cell | Lift 4.9–19.9 in 6 of 6; chance ratio ≥ 2 in 1 of 6 | Not met (under every reading) |
| K1 anchored baselines recover ≥ 80% on both DeepSeek cells | AR 96% / 33%; AF 29% / 10% | Does not fire |
| K2 random-5 ≥ solver any-of-5 for GLM or DeepSeek | 61.5 vs 99; 63.1 vs 123 | Does not fire |
| **Overall** | | **Survives the kill; success not met** |

### I46

| Criterion | Value | Outcome |
|---|---|---|
| Tasks 1–5 (ICC, t-intervals, power curve, batch SD 7.9, reporting note) | All delivered and checked | Met |
| I14's item: reproduce p = 0.031, 0.0078, 0.00098 and the power table | Confirmed in RERUN R05/R34; power replay identical | Met |
| Success: protocol file plus fresh seeds at 100% under full oracle | Not part of this task | Not applicable |
| Success / kill: a blind author commits ≥ 2 situations by about 2026-10-31 | Deadline not reached; an project action | Not applicable |
| Kill: more than 40 layouts needed per situation | 0 of 24 cells | Does not fire |

---

## 7. Where the first check and the verifier differ
Not settled; both are given:
1. **I01 "at most 2 successes lost".** REPORT does not say whether this covers all situations or only lines. Over all solver steps, DeepSeek P loses 4 (not met). Over line steps it loses 1 (met). The overall Step-1 verdict is the same either way, because R3 fails.
2. **I14 S3.** The frozen note limits S3 to the two DeepSeek cells (met). REPORT's wording is unqualified, and on it GLM fails (99 vs 107 old, 89 vs 98 new). The overall verdict does not change.
3. **I01 calibration, pooled or per model.** REPORT does not say. The first check chose pooled as primary after the fact (met). Per model, 2 of 5 models meet the kill's share clause by the letter.
4. **Bootstrap intervals depend on the seed.** Calibration pooled share [0.64, 2.10] (first check) vs [0.66, 2.0] (verifier); qwen3vl real gain lower bound 2 vs 1 (so qwen3vl is only barely distinguishable from 0). I14's recovery intervals move by a few points (old GLM's upper bound by more: 733% vs 669%). No conclusion changes.

Settled by the verifier (its value is used above):
- I01 mechanism: 18 of 29 GLM failures (not 20) list a detection 3.7–4.6 m off; the 14 that stay break down 8 / 1 / 3 / 2 (not 8 / 1 / 4 / 1); in the recorded people, 2 of DeepSeek's 7 stays miss a member.
- I01 oracle target: R6 acts only on goal steps (6 / 6, 7 / 14, 10 / 14 wrong-target fixes, not "of 7, 28, 22"). The headline "ceilings below the guesses" holds for perception, and for the oracle target only in the whole-record wrong-target sense.
- I01 effective n (9 and 4 people answers behind the at-the-bar flips) was added by the verifier.
- Calibration: GLM gains half through R4 (+4 of +8), not "mostly"; the "locate answers identical" check is vacuous (s8 has no locate call); P's people list was checked directly against conorc's (213 of 213).
- A0: the sentence "true right reported as towards_right in 31 to 53 profile line members per model" does not reproduce and is dropped. The closest figures are 11–50 (pipeline yaw's bin, profile line members) and 2–19 (label). Also, 9 s20 DeepSeek answers have no s23 copy.
- I14: random-5 P(success) is 0.12–0.15 (not 0.13–0.15); watching Qwen AR recovery is 40% exact (42% was a draw estimate); R5@1 ranges 17.9–21.4.
- I46: R10's prose misstated its own table (corrected in section 5); exact power values replace Monte Carlo ones; the batch-SD CI upper bound is 22.8 exact, not 23.7.

Not independently checked: I14's AR recovery at 30 and 60 deg and its whole-room in-region proxy (65%, 104%); I46's cluster-bootstrap ICC intervals, item ICC and one-sided power, and whether Qwen's R/E prompt text is the same across s16b, s22 and s21 (it bears on whether 7.93 is pure run noise); I01's claims about added people near kept detections and the 40-person cap.

---

## 8. What it means for the plan (inference)

**I01.**
- For GLM and DeepSeek, perception is a real lever at lines under the recorded constraints. But the lever is mislocated detections that `come_after` names, not missed people. Adding missed people (R3) fixes 2 of 29.
- This supports finding 9's "relocate people, don't add them" reading. It favours I11's snap-and-lift and an I03 audit that moves or drops far-off detections. It weakens the case for depth blobs as a way to add missing people.
- A pixel-space match or a depth-consistency filter that catches these detections is the natural next arm. It is not pre-registered.
- The constraints matter as much as the people. R5 fixes 19 of 29 and 9 of 15, and true people with reference constraints fix every line failure. GLM's 14 leftovers are mostly missing or partial `come_after`. That raises the cheaper routes to "who do I come after" (I19, I24, I32, I05).
- The ceilings are smaller than the brief guessed (perception +6.3 / +4.4 / +0.8 pp, lower bounds). Programme A's payoff for these models is a few points, not ten.
- For the top tier, perception repair is worth at most +0.8 pp. Gemini's loss is the target (R6 +10.3 pp), which is I11's ground. Haiku's line failures are constraints (R5 11 of 11, P 1 of 11).
- The calibration makes the repair harness usable for comparing people sources at the pooled level and for weak perceivers (GLM, qwen3vl). It cannot reproduce the constraints redraw, so per-record agreement tops out near 90%.
- (inference) The optional pipeline oracle-people arm (a big run) may not be needed to read Step 1, since the calibration already gives an offline answer.

**A0 and I09.**
- A1's detector must reach 0.806 within 45 deg on the same matched people, or 0.70 if only the school panel counts.
- True facing alone (R2) flips 0 of 29 and 0 of 15 line failures and breaks 2 DeepSeek successes. So facing errors seem to cost C little on these sets. A detector's main value would be recall and position, plus speed.

**I14.**
- The recorded runners-up add little over random places near the top. They do not license P-4 or O-1, and they do not by themselves justify I04, I24, I36 or I15.
- Group rings are easy to hit by chance (about 78%), in line with finding 8's "groups barely test people-awareness".

**I46.**
- Size the sealed set at 20 or more fresh layouts, with a per-situation floor of 6 and 14 wherever a layout may tie.
- Per-situation claims stay descriptive on the current sets.
- Live A/B checks should be paired within one stage. Independent batches resolve only about 24–32 records (9–13 points).
- Report everything per layout. 252 records carry the information of 18–107 independent ones.

---

## 9. Proposed REPORT edits (not applied)

Each item gives the exact old text, found by text search in `REPORT.md` (each occurs exactly once), and the proposed new text. The tag [phase 1] is new and has its own definition in item 1.

### 1. "How to read": define the [phase 1] tag

Old:
New:

### 2. Opening status line (optional)

Old:
```text
Apart from the step-0 offline harness and its re-run (`RERUN.md`, `results/still-scenes/replay-harness/sets-and-definitions/README.md`), nothing has been built, and no model has been run.
```
New:
```text
Apart from the step-0 offline harness, its re-run (`RERUN.md`, `results/still-scenes/replay-harness/sets-and-definitions/README.md`) and the phase-1 zero-call analyses (`results/still-scenes/overview/phase-notes/phase1.md`: I01 Steps 0–2, A0, I14 version 0, the I46 statistics), nothing has been built, and no model has been run.
```

### 3. Section 1.2, step 2 (I01)

Old:
```text
Deliverables 1–3 (R0, the target and placement split, I34 Stage 0) done 17 Oct (`results/still-scenes/replay-harness/sets-and-definitions/README.md`); R1–R6 and facing accuracy (A0) remain.
```
New:
```text
Deliverables 1–3 (R0, the target and placement split, I34 Stage 0) done 17 Oct (`results/still-scenes/replay-harness/sets-and-definitions/README.md`). Steps 0–2 (R1–R6 and the s8 calibration) and facing accuracy (A0) done 17 Oct (`results/still-scenes/overview/phase-notes/phase1.md`): Step 1 is neither success nor kill, the calibration passes, and A0 sets I09's facing bar at 0.806. Only the optional pipeline oracle-people arm (a big run) is left.
```

### 4. Section 1.2, step 3 (I14 and I46)

Old:
```text
3. I14 scoreboard, plus the statistics half of I46.
```
New:

### 5. Section 1.2, the day-6 partial check

Old:
New:

### 6. Section 6, step 1 row (I01)

Old:
```text
Deliverables 1–3 done 17 Oct (`results/still-scenes/replay-harness/sets-and-definitions/README.md`); R1–R6 and A0 remain.
```
New:
```text
Deliverables 1–3 done 17 Oct (`results/still-scenes/replay-harness/sets-and-definitions/README.md`). Steps 0–2 and A0 done 17 Oct (`results/still-scenes/overview/phase-notes/phase1.md`): 1,301 of 1,301 places reproduce; Step 1 is neither success nor kill; the s8 calibration passes pooled. The optional pipeline oracle-people arm is not run (a big run, listed below).
```

### 7. Section 6, step 2 row (I46 and I14)

Old:
```text
| 2 | Days 5–7 | I46 statistics and I14 version 0 | First result: I46 about 1 d, I14 about 2 d; about 2.5 d together | 0 | Freeze the metric note first |
```
New:
| 2 | Days 5–7 | I46 statistics and I14 version 0. **Done 17 Oct** (`results/still-scenes/overview/phase-notes/phase1.md`): the metric note was frozen before computing (v0); I14's kill tests do not fire, but S2 and S4 fail; I46's tasks are done and its 40-layout kill line does not fire. | First result: I46 about 1 d, I14 about 2 d; about 2.5 d together | 0 | Freeze the metric note first (done; agreeing the list is still the project's and the mentor's) |

### 8. Section 6, day-6 row

Old:
New:

### 9. I01, Payoff: the measured ceilings

Old:
```text
  - Oracle target: at most +2.8, +9.5 and +8.7 pp.
```
New:
```text
  - Oracle target: at most +2.8, +9.5 and +8.7 pp.
  - [phase 1] Measured on 17 Oct (old wording, net of 252): perception repairs +6.3 (GLM), +4.4 (DeepSeek), +0.8 (Qwen), lower bounds; the oracle target's whole-record wrong-target fixes +2.4 (Qwen), +6.3 (GLM), +6.7 (DeepSeek). Both are below these guesses. The whole-record oracle-target arm, which also fixes right-target placement, gives +5.6 / +9.5 / +14.3 (details under the first probe).
```

### 10. I01, first probe: the result

Old:
```text
  - More than 20% of matches are ambiguous.
```
New:
```text
  - More than 20% of matches are ambiguous.
- **Phase-1 result (17 Oct; `results/still-scenes/overview/phase-notes/phase1.md` sections 1–2) [phase 1].**
  - Step 0: 1,301 of 1,301 recorded solver places reproduce (s19 and s20 included).
  - Step 1, old wording, line failures flipped (GLM n 29 / DeepSeek n 15): P 15 / 8, exactly on the bars; R3 alone 2 / 2; R4 alone (dropping unmatched detections) 15 / 4; R5 (circular) 19 / 9; true people with reference constraints 29 / 15. Over all solver steps, P loses 0 (GLM), 4 (DeepSeek) and 0 (Qwen) successes, and R3 loses 0; over line steps only, DeepSeek P loses 1.
  - Step 1 verdict: neither success nor kill. R3 misses its bar badly. DeepSeek's 4 losses miss "at most 2" if the clause covers all situations (it is met if it covers lines only). The two P bars rest on few independent answers: GLM's 15 flips come from 9 people answers in 3 layouts, DeepSeek's 8 from 4.
  - Mechanism: all 15 GLM P flips come through R4. Each has a `come_after` that lists a detection 3.7–4.6 m from every true person, probably a member whose depth fell through (plausible, not proven). So perception helps here by removing mislocated detections that `come_after` names, not by adding missed people.
  - Step 2 (s8, five placeholder models, 420 records): P recovers +22 of the real +22 oracle-people gain (share 1.00, 95% CI about 0.65–2.1), agrees with `s8_conorc` on 90.6% of 213 goal steps (95.2% of 420), with 0.5% ambiguous matches. Success met and kill not met, pooled. Caveats: doing nothing already agrees on 80.3% of goal steps; the net match is partly cancellation (24 of 32 real wins and 0 of 10 real losses reproduced); the criteria do not say pooled or per model, and by the letter ornith15 and qwen38 (real gain +2 each, CI includes 0) meet the kill's share clause.
  - New wording (extra rows; Qwen, GLM and DeepSeek people answers are cache copies): perception repairs give at most +0.4 pp for Haiku, Sonnet and Gemini. Gemini's loss is the target (oracle target +10.3 pp). R5 fixes all 11 of Haiku's line failures, and P fixes 1.
  - (inference) For GLM and DeepSeek, effort should go to removing or relocating mislocated people and to stating `come_after` correctly, more than to adding missed people. A pixel-space match or depth-consistency filter for these detections is the natural next arm (not pre-registered).
```

### 11. I01, dependencies: where `reference_constraints` lives (optional)

Old:
```text
Copying it into a harness tool is lighter.
```
New:
```text
Copying it into a harness tool is lighter. Phase 1 copied its rules into a research script (`results/still-scenes/replay-harness/repairs/repairs.py`, arm R5); nothing in `src/` or `tests/` changed.
```

### 12. I09, Why: facing accuracy is now measured

Old:
```text
VLM facing accuracy has never been measured.
```
New:
```text
VLM facing accuracy is now measured (A0, 17 Oct, [phase 1]): 0.55–0.91 of matched people within 45 deg by model, lowest on people seen in profile.
```

### 13. I09, first probe: the A0 result

Old:
```text
  - A0, the VLM facing error, runs inside I01 at no cost.
```
New:
```text
  - A0, the VLM facing error, runs inside I01 at no cost. Done 17 Oct (`results/still-scenes/overview/phase-notes/phase1.md` section 3) [phase 1]. Within 45 deg, on the yaw the pipeline uses: Gemini 0.906 (n 310 matched pairs, 36 people, 8 layouts; 95% CI 0.81–0.98), Sonnet 0.809, Qwen 0.765, DeepSeek 0.616, Haiku 0.616, GLM 0.547. Weaker models lose most on people seen in profile, mainly line members (GLM 0.463, Haiku 0.488).
  - So the facing kill bar below is 0.806 (10 points below Gemini), or the absolute 0.70 if only the school panel counts. Score A1 on the same matched (arm, scene, person) instances (`results/still-scenes/facing/a0_pairs.jsonl`), since Gemini's 0.906 comes only from the scenes where it ran a people call.
  - Post hoc: the faced point replaces the model's own 8-bin label for 65–91% of people, and for 5 of 6 models this costs 4–23 points (GLM 0.547 against 0.779 with the label alone). A heading-from-label change would need pre-registering and testing on other sets; the placeholder panel splits 2 to 2 on it.
```

### 14. I14, first probe: the version-0 result

Old:
```text
- Kill (the set story, not the code): the anchored baselines recover at least 80% on both DeepSeek cells, or the unconditional random-5 is at least the solver's any-of-5 for GLM or DeepSeek.
```
New:
  - K1 and K2 do not fire. Random-5 in the 3 m disc expects 61.5 (GLM) and 63.1 (DeepSeek) hits, against the solver's any-of-5 of 99 and 123. No anchored baseline recovers 80% on both DeepSeek cells.
  - S2 fails. The top place plus 4 random disc cells recovers 96% of DeepSeek's realistic +14 (layout bootstrap about 65–146%) and 33% of the watching +10 (about 4–60%). Post hoc, drawing over the whole room: 57% and 69%.
  - The skeptic's 85–95% comes in at 77–80%: random-5 in the disc lands in the region (heading aside) in about 78% of right-target group records.
  - The 3 m disc is a poor chance region away from groups. It leaves out about 90% of the watching region and 10–17% of the realistic one, and the top place lies outside it in 470 of 790 line records.

### 15. I46, What: fresh seeds sized

Old:
```text
  - Fresh seeds (3 and up) of the existing situations, sized by the power analysis.
```
New:
```text
  - Fresh seeds (3 and up) of the existing situations, sized by the power analysis (17 Oct: at least 20 fresh layouts; see the statistics below).
```

### 16. I46, first probe: the statistics half

Old:
```text
  - The final run is about 6,000 calls.
```
New:
  - The kill line "more than 40 layouts needed per situation" does not fire: 0 of 24 lines and groups cells need more than 40 (6, or 14 where a layout is tied). That design resamples 3–5 values and is degenerate, and 6 per situation is the test's floor, not a powered size.
  - the Qwen batch SD of 7.9 reproduces as 7.93 (pooled SD of three batch totals, df 4, exact 95% CI 4.8–22.8). A live A/B detects about 24–32 records between two independent 3-trial batches, 11–21 when paired within one stage, and 18–79 when generalising over 14 layouts (80% power, two-sided 0.05).
  - The one-page reporting note is `results/still-scenes/metrics-and-statistics/statistics/HOW_TO_REPORT.md` (two fixes pending: R and E are only nearly identical at nobody-near, since DeepSeek differs in 3 of 108; and 6 layouts per situation is a floor).

### 17. Section 8, batch noise (optional)

Old:
```text
At SD 7–8, the same arithmetic gives about 28–32 records (11–13 points) (inference).
```
New:
```text
At SD 7–8, the same arithmetic gives about 28–32 records (11–13 points) (inference). [phase 1] I46 reproduces Qwen's 7.9 as 7.93 (df 4, exact 95% CI 4.8–22.8), which gives 31.4 records. Paired within one stage, the figure is 11–21 records; over 14 layouts it is 18–79 (`results/still-scenes/metrics-and-statistics/statistics/mdd_table.txt`).
```

### 18. Finding 1 in section 1.1 (optional)

Old:
```text
an oracle snap-and-lift fixes 15 of 15.
```
New:
```text
an oracle snap-and-lift fixes 15 of 15.
   - [phase 1] Under the recorded constraints, the full perception repair fixes 15 of GLM's 29 and 8 of DeepSeek's 15 line failures. All 15 of GLM's come from dropping detections 3.7–4.6 m from every true person (probably mislifted members); adding the missed people fixes 2 of 29. The perception ceilings are +6.3 (GLM), +4.4 (DeepSeek) and +0.8 pp (Qwen), lower bounds below the guessed +10–11.5, +6 and +1. Reference constraints alone give +7.9 and +9.5 pp (circular). (inference) Mislocated people and unstated `come_after` matter more than missed people.
```

### 19. Revision note (optional)

Old:
```text
I11, I22, I44, I47 and C5 get small wording fixes.
```
New:
```text
I11, I22, I44, I47 and C5 get small wording fixes.
- **Phase 1, first results (17 Oct).** I01 Steps 0–2, A0, I14 version 0 and the I46 statistics are done, zero-call, each recomputed by an independent verifier (`results/still-scenes/overview/phase-notes/phase1.md`). "How to read" defines [phase 1]. I01, I09, I14, I46, sections 1.2 and 6 (and optionally finding 1 and section 8) carry the results. I01 Step 1 is neither success nor kill, and its calibration passes. I14's set story survives its kill tests but fails S2 and S4. A0 sets I09's facing bar at 0.806. I46 sizes the sealed set at 20 or more fresh layouts.
```
