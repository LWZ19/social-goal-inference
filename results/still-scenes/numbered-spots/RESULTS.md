# Marks: results

## M1: coverage and the perfect-pick ceiling (zero calls; frozen `M1_FROZEN.md`, sha256 `04c23e25…`)

- **Proposer P0.** I11's frozen G1 depth components (objects and structure), one mark per component at its most interior pixel, lifted with the component's own depths. There is no truth and no model.
- **Candidates per image.** Median 8, 90th percentile 25, maximum 29, over 68 development scenes.
- **Coverage (H1): holds.** Every target instance has a candidate the scorer counts as the right target, in every case within the first 10 candidates:
  - realistic: 84 of 84 (6 objects, 24 people, 54 places);
  - placeholder: 52 of 52.
- **Ceiling gate: fails.** A perfect pick among the first 20 candidates, replayed with each record's recorded people and constraints, gains only +13 records pooled. The bar was +36, half of the oracle-target replay R6's +71.

Table M1.1. Perfect pick among the first 20 candidates, by model, over 18 new-wording C runs.

| Model | Recorded C | Perfect pick | Gain | R6 gain |
|---|---|---|---|---|
| Qwen 3.8 low | 228 | 226 | −2 | +11 |
| GLM-5.3 | 170 | 188 | +18 | +14 |
| DeepSeek V4.1 | 185 | 198 | +13 | +12 |
| Haiku 5.5 | 240 | 223 | **−17** | +1 |
| Sonnet 5.5 low | 242 | 231 | **−11** | +7 |
| Gemini 3.8 low | 214 | 226 | +12 | +26 |

### Surprise, and why (M1b, post hoc, `m1b_diagnose.py`, `m1b_out.txt`)
Picking the right object is not enough: the place on the object the model clicks carries information that the downstream geometry uses.

- **What the pick loses.** It turns 109 successes into failures:
  - every one is a place target, picked as an object-class component;
  - 96 are in nobody-near scenes and 13 at lines;
  - 93 go through the standard goal and 16 through the solver.
- **The pick point versus the click.** The pick's interior pixel lies a median 0.49 m from the model's click (10th to 90th percentile 0.30–0.56 m). The standard goal places the robot in front of the surface at the target point, so moving the point across a counter or kiosk moves the stop out of the region.
- **What the pick wins.** It turns 122 failures into successes: 81 places, 25 people and 16 objects; 71 through the solver and 51 through the standard goal.
- **Keeping both kinds of information.** In the variant M1b-snap, the right component comes from the oracle and the location from the model's own click, moved to the nearest pixel of that component. It gains **+98 pooled**, against +13 for one mark per component and +71 for R6:

  | Model | Recorded | One mark per component | Right component, click inside it |
  |---|---|---|---|
  | Qwen | 228 | 226 | 246 |
  | GLM | 170 | 188 | 187 |
  | DeepSeek | 185 | 198 | 201 |
  | Haiku | 240 | 223 | 242 |
  | Sonnet | 242 | 231 | 249 |
  | Gemini | 214 | 226 | 252 |

  Only records with a target are counted. The right component is chosen by oracle, so this is a ceiling, not an estimate.

**What it means (inference).** The click does two jobs:
- it chooses the target, where models make choice errors and near-misses;
- it says where on the target the robot should go, which the strong models already do well.

Marks should take over only the first job. The design to test is therefore "number plus point": the model gives a mark number for the target and a point on it, and the point is moved into the chosen component. That keeps the location and fixes the identity. Whether models pick the right number when shown marks is the live question (M2).

### Where the +98 comes from (M1c, post hoc, `m1c_split.py`, `m1c_fixed.log`; the questioner's Q8–Q10)
**A bug, fixed.** The first M1c run cached candidates by Python object id. The harness reuses ids after it evicts a scene, so 90 records were scored against another scene's candidates. The key is now (scenes folder, scene id).
- M1's and M1b's numbers are unchanged by the fix: identical output.
- M1c's first run is kept in `buggy_idcache/`.

**The snap ceiling's 98 gains, with 0 losses, by where the click was:**

| Click | Gains | By model |
|---|---|---|
| A near miss (within 15 px of the right component) | 43 | Qwen 17, DeepSeek 11, GLM 8, Sonnet 7 |
| On the wrong object (more than 15 px away) | 54 | Gemini 38, GLM 9, DeepSeek 5, Qwen 1, Haiku 1 |
| Inside | 1 | Haiku |

- **Wrong objects.** Only a changed choice gets these, and that is where marks earn their place. Gemini's 34 wrong-target records are all of this kind; I11's oracle snap only reached 15 px.
- **Where the location comes from (Q9).** For the wrong-object gains it comes from the component, not from the model:
  - the component's interior pixel succeeds in 44 of 54;
  - a random pixel of it succeeds 0.80 of the time on average;
  - the snap's 54 of 54 partly reflects the near edge.

  For near misses, random pixels succeed 0.90. So the realistic ceiling for a mark choice is about +44 (wrong objects, at the interior pixel), plus the near misses, which a snap would get anyway.
- **"Click, then confirm among the K components nearest the click" (Q10).** On the 111 wrong-target records, the nearest components contain the right one at:
  - K=1: 0.16; K=2: 0.59; K=4–6: 0.71; K=8: 0.76.

  A two-step design would miss about 29% of the wrong-object cases. So marks should cover the whole image, not just the area near the click.
- **Mark count (Q7, Q10).** Ordered by size, a realistic target is in the first 3 for every place, but people need up to 8. Coverage is 0.79 at N=3, 0.93 at N=4–6, and 1.00 at N=8.

**Sonnet at eval_12 (Q14, the image viewed).** "The person in red" stands behind a man whose arm overlaps her. Sonnet clicks on the visible part of her shirt beside his sleeve, which lands on the man's arm. It is a near miss caused by occlusion, not an ambiguous request. Touching people like these may also merge into one depth component, so person marks should come from the detector's people, which are stored for every development scene (`results/still-scenes/people-finder/early-detector/phase1c/a1/eval_people.json`), not from components (Q13).

**What this means for M2 (inference).**
- **The free-model run mainly measures harm.** That covers marks breaking a right choice, losing location information, binding errors and number bias. It can say little about benefit, except for GLM.

## M2: harm check on the free models (frozen `M2_FROZEN.md`, sha256 `c709374a…`; sent to the run, not run yet)
- **Arms.** Paired plan calls, unmarked (A0) and marked (A2, "number plus point"), at fresh trials 3–5.
- **Size.** 84 realistic episodes × 3 trials, for Qwen low, GLM and DeepSeek (x-test-3 after a smoke check): 504 calls per model.
- **Judged (harm).** Right→wrong at most 2% of A0-right per model, and a plan-exact drop of at most 2%.
- **Reported.** Gain, replayed success, the number-only and point-only rules, binding conflicts, and number salience.
- **Precheck.** The M2 mark set (detector people plus components, at most 8) covers 84 of 84 target instances.

### M2 results so far: DeepSeek and x-test-3 (GLM and Qwen still running)
- **Frozen reading, rule "number plus point", 252 pairs each:**

  | | Target right, A0 → A2 | Gain / harm | Harm share | Plan exact, A0 → A2 |
  |---|---|---|---|---|
  | DeepSeek | 232 → 241 | 19 / 10 | 4.3% | 176 → 179 |
  | x-test-3 | 238 → 243 | 12 / 7 | 2.9% | 204 → 214 |

  - **The judged harm bar (2%) fails for both,** so under the frozen rule there is no Gemini arm.
  - **The bar was miscalibrated** (post hoc, `m2c_noise.py`). Two fresh unmarked draws (A0 against A0) already flip right→wrong / wrong→right 16/12 (DeepSeek) and 20/8 (x-test-3) per 252. The marks' harm (9–10 and 7) is smaller than that redraw noise, and their gain sits inside it. So the honest reading is "no detectable effect on target choice either way". The bar should have been judged against a same-arm redraw, not against 0.
- **Replayed success, DeepSeek:** A0 195 → A2 168 (post hoc, `m2b_diagnose.py`).
  - In 28 pairs with both clicks on the right target, A2's click moved a median 98 px on place targets, and the stop failed.
  - Only 1 of 28 unmarked clicks lay under a label. x-test-3 has 0 such pairs.
  - Those pairs were selected on A2 losing, so M2P re-tests them with fresh draws.
- **What the flips are:** semantic, not binding.
  - Harms mostly move from a desk or counter to the person staffing it, or to a sign.
  - Gains move the other way: the occluded person in red, "back of the line" → counter, notices board → ticket counter.
  - Binding conflicts: 14 and 8.
  - A type filter (type-then-mark) would have blocked about 9 harms but also about 6 gains, so it was dropped.
- **Number salience:** mild. Centre-nearest picks are 49 and 60, against 39 and 40 by chance. The K=7 excess is correct picks of centred targets.

### M2P: prompt or drawn marks? (frozen `M2P_FROZEN.md`, sha256 `ab0b1453…`; sent to the run)
- A fresh 2×2 for DeepSeek on the 20 episodes behind the 28 pairs, trials 6–8: 240 calls.
- Arms: the pipeline or the marks prompt, crossed with an unmarked or a marked image.
- Decision rule as frozen.

### M2 final: all four free models (frozen bars, with the post hoc noise floor beside them)

| Model | Target right A0 → A2 | Gain / harm | Two unmarked draws flip (rw / wr) | Sign p | Replayed success A0 → A2 | Plan exact A0 → A2 |
|---|---|---|---|---|---|---|
| GLM-5.3 | 215 → 237 | 30 / 8 | 26 / 26 | 0.0005 | 155 → 177 | 200 → 228 |
| DeepSeek V4.1 (1017) | 232 → 241 | 19 / 10 | 16 / 12 | 0.09 | 195 → 168 | 176 → 179 |
| Qwen 3.8 low | 243 → 247 | 9 / 5 | 8 / 8 | 0.42 | 233 → 234 | 233 → 234 |
| x-test-3 (Qwen 3.8 Next) | 238 → 243 | 12 / 7 | 20 / 8 | 0.36 | — (no recorded C run) | 204 → 214 |

- **As frozen:** the harm bar (≤ 2%) fails for every model, so there is no Gemini arm under M2's rule.
- **Calibration:** as found post hoc, the bar should have been judged against a same-arm redraw. Against that floor, marks change target choice only for GLM, where 30 against 8 is well outside its 26/26 redraw flips. For the other three models they are inside noise.
- **Plan decomposition** is never harmed, and for GLM it improves (+28).
- **The sign test treats records as independent.** Records cluster by layout, so p is optimistic.

**Reading (inference).**
- Marks help the model that points worst, GLM: +22 on target choice and +22 on replayed success.
- They do nothing measurable for the strong pointers.
- They harm DeepSeek's location (−27 replayed), through label attraction (M2P below).

This is a finding about the interface, not a deployable win for every model.

### M2P: the drawn labels move DeepSeek's click (frozen `M2P_FROZEN.md`; fresh 2×2, 60 cells)

| Arm | Prompt | Image | Right | Replayed | Click moved from P0U (median px) |
|---|---|---|---|---|---|
| P0U | pipeline | unmarked | 57 | 49 | — |
| P2U | marks | unmarked | 58 | 53 | 41.9 |
| P0M | pipeline | marked | 56 | 40 | 38.7 |
| P2M | marks | marked | 58 | 38 | 68.4 |

- **Judged cause:** the drawn marks. The replay loss is 11 for P2M and 9 for P0M with the pipeline's own prompt; P2U shows none (−4).
- **Mechanism** (post hoc): labels attract the click. The median distance from the click to the nearest label centre is 39 px in the unmarked arms and 27–28 px in the marked ones. Clicks within 12 px of a label are 3–6 of 60 unmarked, against 8 and 14 marked, even with "do not point at a label". Labels sit at each object's interior pixel, which M1b showed is a poor reference for place targets.
- **Invented marks:** with the marks prompt but no labels visible (P2U), DeepSeek still gave a mark number in 16 of 60 answers.

### M2d: the two-call design (post hoc, zero calls, from M2's answers)
The number comes from the marked call and the location from the unmarked call's click.

| Model | Target right A0 → combined | Replayed A0 → combined |
|---|---|---|
| GLM | 215 → 239 | 155 → 167 (A2 alone: 177) |
| DeepSeek | 231 → 237 | 194 → 167 |
| Qwen | 243 → 247 | 233 → 228 |

It does not beat one marked call, and it costs a second call. Dropped.

## Where marks stand (inference)
- **A model-dependent tool.** Numbered marks fix choice errors for a weak pointer (GLM) and are neutral for strong ones. They can pull a model's click towards the label (DeepSeek).
- **Next probe, if pursued: label placement.** Labels away from the object's surface, at an edge or outside with a leader line, could remove the attraction. A small M3 on DeepSeek's 20 episodes would test it, with a P0M-style arm under each placement.

### GLM's effect, checked three ways (post hoc, `m2e_glm.py`; the questioner's notes)
1. **At the layout level it holds.** Per-layout net (gain − harm) is positive in 9 of the 11 non-zero layouts and +22 in total. The exact two-sided sign-flip test gives p = 0.011 (record level 0.0005).
2. **It is mostly identity, not snapping.** Of GLM's 30 gains, 24 had A0's click more than 15 px from every right candidate, i.e. on another object. Only 6 were near misses a zero-call snap would also fix.
3. **Plan exact has two sources.** It improved in 40 pairs and worsened in 12. Of the 40, 28 are the same pairs as target gains. Another 13 drop a trailing act subtask: a navigate-then-act plan becomes navigate only, matching the reference. That structural part may come from the marks prompt's wording ("set both to null for every act subtask") rather than from the marks, so it is not claimed as a marks effect.

- **Target right:** A0 214 → A2 221, with gain 15 and harm 8. Gemini's own A0–A0 redraw flips are 7/15, so **marks do nothing beyond noise for Gemini.**
- **Replayed success:** A0 208 → A2 199. Gemini's A0–A0 replay flips are 7/19, against 23/14 for A0–A2: a small location cost. In 9 pairs the click moved a median 111 px on place targets. Gemini's typical click moves only 8 px, so label attraction is weak here.
- **Plan exact:** 204 → 213.
- **What Gemini's errors are.**
  - Harms: "coffee counter" → "the person at the end of the line" (4), print station → "the person" (2), ticket counter → events poster (2).
  - Gains: mostly the reverse, plus the occluded person in red (7 of 15).

  So Gemini's line errors are an **interpretation** choice between the counter and the last person in line, not a pointing or identity error. Marks flip that choice both ways at about the redraw rate.
- **Two-call design:** target right 221, replayed 198. No help.
- **What this refutes (inference).** It refutes M1c's expectation that Gemini's 34 wrong-object errors were the class marks can fix. "Wrong object" by pixel distance included interpretation errors. Marks fix identity errors (GLM). They don't fix a model's different reading of what the target should be. That reading matters, because the plan prompt defines the target as "what the following act subtasks are performed at", and "the person at the end of the line" breaks that definition. That is a prompt-interpretation question, outside marks.

### M4: marks on D1's sim frames (frozen `M4_FROZEN.md`, sha256 `68ad54ec…`; 180 free calls)

| Model | Target within 1 m of the counter, unmarked → marked (of 45) | Gain / harm | Unmarked trial-to-trial flips | Median distance |
|---|---|---|---|---|
| DeepSeek | 35 → 32 | 7 / 10 | 12 | 0.42 → 0.43 m |
| GLM | 28 → 26 | 8 / 10 | 18 | 0.37 → 0.53 m |

- **Judged:** no for both (marked − unmarked < +5), so no CT episodes with marked first answers.
- **Against the prediction:** it held for DeepSeek. The GLM exception did not appear on sim frames.
- **Possible reasons (inference):**
  - the sim marks are depth components only, with no person marks;
  - in the 3 cafe frames the counter carries no mark;
  - GLM's static-scene gains were identity errors among realistic objects, which these coffee scenes hardly contain.
- **What this closes:** the marks × CT bridge.
