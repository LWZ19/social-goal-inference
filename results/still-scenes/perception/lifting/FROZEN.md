# I11 Stage 0 (phase 1b): frozen protocol, with I33's Stage B trigger

The I11 Stage 0 lead wrote this note on 17 Oct 2026, before any snap variant of this batch scored an evaluation record. Its sha256 is in `FROZEN.sha256`, and the file is never edited. Any change goes into a dated addendum with its own hash, written before the data it affects is scored (section 10).

- Scope: the C wrong targets of the realistic old-wording runs come first; the extras come after.
- Error classes: those of `results/still-scenes/perception/lifting-rerun/lifting.py`.
- Fixes replayed: the oracle snap-and-lift, and a pre-registered family of at most 4 depth-component snaps with one primary.
- The REPORT's criteria are quoted verbatim, with their denominators defined here.
- The I33 Stage B trigger is pre-registered here.
- `match_target` stays unchanged, and is reported on separately.

**Exposure (disclosed).** This note is not blind. Before writing it I read:
- REPORT.md: entries I03, I11, I09 and I33, and in passing the entries printed around them.
- The sibling I03 note (`results/still-scenes/perception/depth-blob-detector/FROZEN.md`), for its format only.
- RERUN.md: rows R36–R40, R55, R56 and the I11 summary line.
- `results/still-scenes/overview/phase-notes/phase1.md`, sections 0–1.
- `results/still-scenes/replay-harness/sets-and-definitions/README.md`, line 229. It says by scene where the new-wording wrong targets lie (for example Sonnet 7 of 7 at eval_12, Gemini 34 of 34 at lines), but nothing about their pixels.
- The code and outputs of `results/still-scenes/perception/lifting-rerun/`. `lifting_out.txt` lists each of the 15 old-wording person wrong targets with its snap distance, its 5×5 window depth range, its body pixels in the window, the surface the lift landed on, and its class. It also lists the place near-misses. `frontsnap_out.txt` tallies the front-surface snap per model and class.

From the worked-example scenes, read through `H.example`, I took only:
- the camera: 640×480, fx = fy = 554.3 px, height 1.5 m, pitched 10° down;
- the depth: float32, with no invalid pixels;
- the ceiling: z = 3.0 m;
- object sizes: target footprints and obstacle polygons are at most 2.26 m across, and rooms are 12.2–15.5 m across (diagonals).

Consequences:
- **V3** exists because the occluder-capture rows show target pixels inside the lift window in 5 of 9 cases. It is data-informed.
- **The I33 result on the 15 old-wording person records is not blind.** The rows I read suggest that trigger T1 catches about 11 of 15 (section 12). The new-wording person wrong targets are the first blind test, and they are reported beside (section 8).
- **Nothing in sections 4–5 was chosen from evaluation records.** Each parameter comes from sensor geometry, a package constant, results/still-scenes/overview/rerun-files/lifting's pre-fixed radius, or fit-data ground truth. Each one's source is given where it is defined.

## 1. Data
**Primary runs.** Old wording, realistic arms, method con, 252 records each. Each is selected with `H.runs(stage, arm, model, method='con')`, asserting exactly one match:
- s19 `qwen38_low`, arms real, staffed and plain: `20261011-160334`, `-160344`, `-160527`.
- s19 `glm53`: `20261011-160500`, `-160352`, `-160452`.
- s20 `dsv41`: `20261013-191337`, `-191345`, `-191402`.

**Expected counts** (results/still-scenes/overview/rerun-files/lifting, RERUN R39). These are recomputed; any difference is reported with its reason.

| | Qwen | GLM | DeepSeek | All three | GLM + DeepSeek |
|---|---|---|---|---|---|
| Records | 252 | 252 | 252 | 756 | 504 |
| With a point | 252 | 248 | 252 | 752 | 500 |
| Wrong target | 7 | 28 | 22 | 57 | 50 |
| Wrong target with a point | 7 | 24 | 22 | 53 | **46** |
| Right target with a point | 245 | 224 | 230 | 699 | **454** |

**Extras.** Each is reported per model and run, and never pooled with the primary.
- **X-new** (new wording, realistic arms, con): s21 `qwen38_low`, and s23 `glm53`, `dsv41`, `haiku55`, `sonnet55_low` and `gemini38_low`, 18 runs. Their target points are fresh plan draws that no earlier pixel-level analysis has looked at. This is the blind replication.
- **X-ph** (placeholder, con): s19 `qwen38_low` and `glm53` on arms ph and dev_seed0–2, and s20 `dsv41` on the same arms, 12 runs.
- **X-ph2** (earlier stages): s10 con (5 runs) and s8 convlm and conorc (10 runs). A run whose records lack a target pixel, or whose records fail the harness reproduction checks, is listed and skipped. Draws repeated across runs (same model, scene, episode, trial and pixel) are counted.
- **Never:** E or R runs. Within a stage, C and E hold the same target points (756 of 756 pairs, RERUN), so E adds no targets.

**Fit data.** The 28 worked-example scenes of `H.examples()`, used only for the S0 checks (section 10). Every parameter is fixed in this note, so nothing is fitted after the hash.

**Fresh seeds** (`results/still-scenes/perception/fresh_seeds/`, seeds 100–109, from the I03 lead):
- If present when measure M4 runs, they are an extra domain for M4 only, read directly.
- They are used for nothing else, since they hold no model answers.
- Seeds 3–99 are never generated or opened.

**Thresholds.** Each record uses its run's recorded config (`H.recorded_config`). The code asserts these values for every record: `floor_tolerance_m` 0.15, `lift_window` 5, `seen_depth_ratio` 0.03, `seen_depth_margin_m` 0.02, `match_tolerance_m` 0.3 and `default_match_radius_m` 0.75.

| File | sha256 |
|---|---|
| `src/sgi/evaluate.py` | `198f11ae6adf6e8a7eda4bb2f8c78f982baaf38f15cbe6f8ff6276eb66086ca1` |
| `src/sgi/geometry.py` | `99af62055b6998faafbf6748b7eece13adb427227c71821b5190fdcdbd4b4db2` |
| `src/sgi/stages/goal.py` | `edce50f09990b80e5413c41caf19b543747f33a912e6e14c93aed5db1f7ba600` |
| `docs/model_notes/tools/research_harness.py` | `391483778ec1c5a232909292862926724a6580fc5f9c307b1f3f9c789cee455c` |
| `docs/model_notes/tools/replay_solver.py` (modified in the folder; `H.resolve` runs it) | `eed1dcc1bb0287208e98a4b150236486b7c23d12613859b1dc92890b7d3af243` |
| `results/still-scenes/perception/lifting-rerun/lifting.py` | `10ec43fe1871ab89f749dfa3991498c03c856de9b125eeddcf9561ee9c80fe63` |
| `results/still-scenes/perception/lifting-rerun/frontsnap.py` | `a45aae92673562fab43fd2f4dee9cecfac5c3b0b214fe641254a24d358671307` |

The run script checks these hashes first and stops if any file has changed.

**Rules.** Reads go only through the harness. No model or network call is made. Nothing is written outside `results/still-scenes/perception/lifting/`. At most 2 worker processes run at once, with `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2` and `MKL_NUM_THREADS=2`.

## 2. Shared definitions
**Record sets.**
- **With a point:** `rec.target` is not None.
- **Right target / wrong target:** `sgi.evaluate.match_target(world, truth, gt, th)` returns an id or None. On the primary runs, the code asserts that each record's recorded world point gives its recorded `target_correct`. In the extras, a disagreement is counted and reported (section 9), and the verdict `match_target` gives now is used.
- **The point:** the recorded continuous pixel `rec.target['pixel']` = (u, v).
- **The pipeline lift:** `H.target_at(rec, pixel=(u, v))`. It must reproduce the recorded world point to within 1e-9 m. This is asserted on the primary runs. In the extras, a record that fails it is counted and skipped.

**True target.** `gt.target_ids[0]`. Every primary episode has one target. In an extra episode with several, the true target is the one whose body mask is nearest to the point, and the oracle uses the union of all their body masks.

**Masks.** Imported unchanged from `results/still-scenes/perception/lifting-rerun/lifting.py` (`scene_data`, `target_masks`, `nearest_pixel`, `window`, `lift_at_depth`, `surface`):
- **Per-pixel lift:** each pixel's own depth, lifted at its centre (j + 0.5, i + 0.5). It equals `sgi.geometry.lift_all`.
- **Body mask:** the target's approximate visible surface from ground truth, with `R_PERSON` = 0.30 m.
  - Person: within 0.30 m horizontally of that person, with 0.05 < z < 2.0 m, and that person is the nearest.
  - Place: within 0.05 m of the footprint, with z > 0.05 m.
  - Object: within the match radius of the anchor, with z > 0.05 m.
- **Match mask:** the pixels whose per-pixel lift `match_target` accepts.

**Error classes.** These apply to wrong targets with a point. They are lifting.py's rules with the same precedence, `SNAP_R` = 15 px and `FRONT_MARGIN` = 0.25 m. `Zpt` is the median of the 5×5 window's valid depths. `Zt` is the median depth of the body-mask pixels within 15 px of the point, or the nearest body pixel's depth when none are that close. Tested in this order:
1. **Choice error:** no body-mask pixel within 15 px.
2. **Occluder capture:** Zpt < Zt − 0.25 m.
3. **Background leak:** Zpt > Zt + 0.25 m.
4. **Floor slip:** the recorded lift has z < 0.05 m.
5. **Near-depth miss:** anything else.

**Lift classes** are 2–4. A near-depth miss counts as neither a lift class nor a choice error. It is reported apart, and the 46 still counts it in the denominator (conservative). The precedence puts a background leak that lands on the floor under "background leak", not "floor slip". A descriptive cross-tab of lifting.py's `surface()` (person, obstacle, footprint, wall, wall base, floor) is reported beside the classes.

**Target kind.** Person, object (the parcel) and place, as in lifting.py's `class` column.

## 3. The oracle snap-and-lift
This is lifting.py's fix, unchanged, but now applied to every record with a point, not only to wrong targets.
- **Trigger:** some body-mask pixel of the true target lies within 15 px of (u, v), measured from (u, v) to pixel centres.
- **Move:**
  - Snap to the nearest such pixel q (ties go to row-major order).
  - Take Zm, the median depth of the body-mask pixels in the 5×5 window centred on q, clipped at the image border.
  - Lift q's centre at Zm.
  - The new target is `GroundedTarget(point=pixel_to_point(q centre), pixel=q centre, world=lift)`.
- **Otherwise** the recorded target is kept.
- **Prior.** It fixed 20 of 46 GLM + DeepSeek wrong targets (16 full successes), and 26 of 53 over all three models. It has never been applied to right-target records before.

## 4. Depth components (no ground truth, no ML)
Each is computed once per scene image, from the depth and camera only.
- **Floor pixels:** valid pixels whose per-pixel lift has z < `floor_tolerance_m` = 0.15 m. This is the package's own floor band: the solver and the oracle path treat anything below it as floor.
- **Join (primary, G1):** two 4-neighbours, both valid and neither floor, are joined when the 3D distance between their per-pixel lifts is at most g = 0.10 m.
  - Source: sensor geometry. The depth is exact (rendered float32; the brief has no noise model), so g only has to span neighbouring pixels on one surface.
  - A pixel spans Z / 554.3 m: 7.2 mm at 4 m, 21.6 mm at 12 m (`floor_range_m`).
  - With g = 0.10 m, one surface stays joined up to an incidence from its normal of acos(Z / (554.3 × 0.10)): 85.9° at 4 m, 81.7° at 8 m, 77.5° at 12 m. Two bodies more than 0.10 m apart in 3D separate.
  - Known cost, from the same geometry: horizontal tops 0.4–0.75 m below the camera split into row strips beyond about 4.7–6.4 m. M4 counts this.
- **Components:** connected sets of joined pixels, computed in numpy (label propagation to a fixed point; the result does not depend on the method).
- **Fragments:** components smaller than 25 px, the area of the 5×5 lift window, since such a piece cannot fill one lift window. A fragment is never a snap destination and is ignored by the straddle test.
- **Structure:** a component of at least 25 px is structure when its robust horizontal extent exceeds L_S = 4.6 m.
  - Extent: the principal axes of its pixels' lifted (x, y), and, over the two axes, the larger 98th-minus-2nd-percentile range.
  - L_S is twice the largest target or obstacle diagonal in the 28 worked-example scenes (2.26 m), plus g: 4.52 + 0.10 = 4.62 m, rounded to 4.6 m. Two of the largest objects touching still count as objects, while the walls and ceiling (one envelope in rooms 12–15 m across) do not.
- **Object components:** every other component of at least 25 px.
- **Surface of the pipeline lift (s):** the label of the pixel, in the 5×5 window (clipped) at (⌊v⌋, ⌊u⌋), whose valid depth is nearest to the window's median valid depth. Ties go to the pixel centre nearest (u, v), then row-major. Its class is floor, structure, object or fragment.
- **Component lift onto component c:**
  - Take q, the pixel of c whose centre is nearest (u, v), within R = 15 px (ties row-major).
  - Take Zc, the median depth of c's pixels in the 5×5 window centred on q.
  - Lift q's centre at Zc, and build the new target exactly as the oracle does.
  - R is results/still-scenes/overview/rerun-files/lifting's `SNAP_R`, fixed there before its counts. Using it for the oracle and every variant keeps the "share of the oracle gain" like for like.

## 5. The variant family (all reported; none chosen on results)
**V1 (primary): structure-gated snap.**
- **Trigger:** s is floor or structure.
- **Move:** component lift onto the object component that has the pixel nearest (u, v) within 15 px. With none, nothing moves.
- **Reasoning:** a pointed target is an object. When the lift lands on the room's envelope (floor, walls, ceiling), the point has fallen off an object, so it goes back to the nearest one. Points whose lift already lies on an object or a fragment are never touched. This avoids the front-surface snap's failure, where a right point on a person moved to a nearer surface (36 of its 37 breaks were person targets). V1 cannot fix occluder captures, since there the lift lies on an object.

**V2: always snap (the REPORT's literal "snap to the nearest depth component"; the oracle's procedure with object components in place of the true body mask).**
- **Trigger:** always.
- **Move:** component lift onto the object component that has the pixel nearest (u, v) within 15 px (distance about 0 when the point is on one). With none, nothing moves.
- **What it tests:** whether the 5×5 window's mixed median helps or hurts, compared with a median taken within one component.

**V3: V1 plus an amodal back-step.**
- **Trigger:** V1's, OR all of these: s is an object component c_n; the 5×5 window also holds pixels of another object component c_f; and c_f's window-median depth is greater than c_n's.
- **Move:**
  - Under V1's trigger: as V1.
  - Otherwise: component lift onto c_f. If several components qualify, take the one with the pixel nearest (u, v), then the lower label.
- **Hypothesis:** the model marks the amodal centre of a partly hidden object. Data-informed (Exposure). It may fix the occluder captures whose window straddles the edge, at a break cost on right targets that lie in front of another object.

**V4: V1 with a depth-relative join (G2).**
- **Definition:** V1, with neighbours joined when |Z_a − Z_b| ≤ 0.03 · min(Z_a, Z_b) + 0.02 m. These are the recorded config's `seen_depth_ratio` and `seen_depth_margin_m`, the package's own "same surface" slack.
- **Unchanged:** floor, fragments, L_S and R.
- **What it tests:** how sensitive V1 is to the component definition.

**Reference row (not a variant).**
- `results/still-scenes/perception/lifting-rerun/frontsnap.py`, re-run unchanged.
- It must reproduce 8 fixes and 37 breaks of 752. Otherwise the code is wrong and nothing is reported until that is resolved.

## 6. Replay and measures
**Moved.** A fix moves a record when its target pixel or world point differs from the recorded one. An unmoved record keeps its recorded verdict.

**Full-success replay** of a moved record. This is lifting.py's path, which assumes the pipeline takes the same branch:
- **Solver step:** `H.resolve(rec, target=new)`.
- **Source `standard`:** `sgi.stages.goal.standard_goal(observation, new, th)`, scored with `H.score_pose(rec, pose, target=new)`. No pose counts as a failure.
- **Otherwise:** `H.score_pose(rec, rec.pose_world, target=new)`.

**Measures**, for the oracle, V1–V4 and the reference row:
- **M1 (fixes):** wrong target → right target, on the 46 (GLM + DeepSeek) and on the 53, by error class and target kind.
- **M2 (target breaks):** right target → wrong target.
  - Over the 454 GLM + DeepSeek right-target records with a point. This is the criterion denominator. It defines the REPORT's "about 400 correct records".
  - Over the 699 right-target records of all three models, which equals breaks over all 752 records with a point (comparable with the front snap's 37 of 752).
  - By target kind.
  - Moves count separately for right and wrong records.
- **M3 (full successes):**
  - The fixed records that the replay makes successes.
  - Net successes over all records with a point: gains minus losses, where a loss is a recorded success that fails after the move.
  - GLM + DeepSeek and all three models.
- **M5 (share of the oracle gain):** gain = M1 fixes on the 46 minus M2 breaks on the 454.
  - Share = gain(variant) / gain(oracle), undefined when the oracle's gain is ≤ 0.
  - Secondary: the same over all three models, and with net successes (M3) in place of target gain.
- **M4 (component recovery):**
  - **Unit:** each distinct (scene folder, scene, true target) of the primary runs' scenes (realistic). The same is done for X-ph scenes, for the fit scenes (S0) and for fresh seeds if present.
  - **Mask:** B = body mask ∩ {z ≥ 0.15 m}, since components exclude the floor band by construction.
  - **Best component:** c* = the component (any class) with the largest overlap with B.
  - **Scores:** IoU = |c* ∩ B| / |c* ∪ B|; coverage = |c* ∩ B| / |B|; purity = |c* ∩ B| / |c*|.
  - **Flags:**
    - merged: purity < 0.5;
    - merged with another body: c* holds ≥ 25 px of another person's body mask or another target's body mask;
    - split: coverage < 0.5;
    - absorbed into structure: c* is structure.
  - **Reported per kind:** n, median IoU, share with IoU ≥ 0.5 and ≥ 0.7, and the flag counts, for G1 and for G2.
  - **Also:** the same for the true targets of the 53 wrong-target records, split by error class.
- **Clustering**, for every count: the distinct layouts (eval_XX without the arm prefix) and distinct (model, layout) pairs behind it. The three arms are renders of the same 14 layouts, so the effective n is small (13 of the 15 person wrong targets are at eval_12).

## 7. I11 Stage 0 criteria
**Verbatim from REPORT.md (I11, First probe):**
> Success, all of:
> - Lift classes are at least 40% of the GLM and DeepSeek C wrong targets (at least about 18 of 46).
> - Oracle snap-and-lift fixes at least 15 of 46 and breaks at most 2 of about 400 correct records.
> - Replay turns at least 8 of those into full successes.
> - Depth components reach at least 70% of the oracle gain.
>
> Kill: lift classes under 20%, fewer than 6 fixes, or more than 5 breaks.

**How each is judged.**
- **S1 (lift classes):** at least 40% of the 46, so at least 19 (18.4 rounded up; 18 of 46 is 39.1% and does not meet it). Also reported over 50, counting the 4 GLM records with no point as non-lift (20 of 50 would be exactly 40%).
- **S2 (oracle):** oracle M1 ≥ 15 of 46, and oracle M2 ≤ 2 of the 454.
- **S3 (replay):** at least 8 of the oracle's fixed records among the 46 become full successes (M3).
- **S4 (depth components):** V1's M5 share ≥ 0.70. It is judged on V1 only. If another variant reaches 0.70 while V1 does not, that is reported as a post-hoc observation, not a pass.
- **Kill (the idea):** lift classes < 20% of 46 (at most 9), oracle fixes < 6, or oracle breaks > 5 of the 454.
- **Verdict:** success when S1–S4 all hold and no kill fires. Kill when any kill clause fires. Neither otherwise.

- A V1 kill ends the zero-call snap route. It does not kill I11's Stage 0 verdict, which rests on the oracle.
- When V1 meets S4 but trips its kill, the kill wins.
- V2–V4 are reported against the same bars, without verdicts.

**Also reported for the REPORT's Stage 1 gate:** the share of choice errors among the 46. Rerun/lifting found 26 of 46. Stage 1 itself is not run.

## 8. I33 Stage B trigger (pre-registered)
**T1 (primary).** A grounded target is flagged as suspicious, using no ground truth, when either holds:
- s (section 4) is floor or structure; or
- the valid pixels of the 5×5 lift window carry at least 2 distinct labels among floor (one label), structure components and object components. Fragments are ignored.

In words: the lift landed on the room's envelope, or the lift window straddles a depth edge. It reads only the depth the pipeline's own lift reads, plus the component labels.

**T2 (secondary, reported only).** T1, OR s is an object component and another object component has a pixel within 15 px of (u, v). This measures what a wider reach costs.

**Population.** The 15 person wrong targets of the primary runs: Qwen 5, GLM 2, DeepSeek 8.

**False-trigger denominators** (primary runs):
- **D1** (decisive): right-target records whose true target is a person, about 201 (front-snap table: 165 + 36).
- **D2** (reported): all right-target records with a point, 699.

**Criteria, verbatim from REPORT.md (I33):**
> Stage B: the trigger catches at least 12 of 15 person wrong-targets, with at most 10% false triggers.
>
> Kill: [...] The trigger catches fewer than 10 of 15.

**How they are judged.**
- Success: T1 catches ≥ 12 of 15, and flags ≤ 10% of D1.
- Kill: T1 catches < 10 of 15.
- Neither otherwise. The D2 rate is reported, not judged.

**Also reported:**
- T1 and T2 on all 53 wrong targets, by class;
- the share of flagged records that are wrong targets;
- the flag rate over all 752 records (the share of goal steps that would carry I33's verification call);
- per model, in each extra: on X-new person wrong targets (the blind test) against its own D1 and D2, and on X-ph.

## 9. `match_target`
This batch does not change how `match_target` scores a target. Every verdict uses `sgi.evaluate.match_target` as imported at commit `f1a488a`, and its hash is checked (section 1).

As in lifting.py, the vectorised masks are checked against `match_target` on sampled pixels (400 random plus up to 200 on the mask, per wrong-target record). Any mismatch is reported in its own section, and so is any record whose recorded `target_correct` disagrees with `match_target` now.

## 10. Checks before scoring (S0), and the one general fix
1. **Unit checks on synthetic depth:**
   - a box 0.5 m in front of a wall gives two components, with the wall as structure once it is wide enough;
   - a floor plane gives only floor pixels;
   - a slanted plane at 80° incidence at 4 m stays one component under G1;
   - the component lift equals `lifting.lift_at_depth` for a one-component window.
2. **Fit checks on the 28 worked examples**, written to `s0_fit.txt`. No evaluation record or scene is loaded before this step.
   - M4 on their true targets.
   - The share of pixels within 0.3 m of the room's bounding polygon, or with z > 2.85 m, that are labelled structure.
   - The share of ground-truth person, target and obstacle pixels that are labelled object.
   - The component and fragment counts per image.
   - The trigger rates of T1 and V1 at the oracle target of each fit episode (`sgi.gt.derive.oracle_target`): how often a perfect point would be flagged or moved.
3. **One general fix.**
   - It is allowed only if a section 4 definition fails visibly on fit data. For example: walls are not labelled structure, or ground-truth people are absorbed into structure in more than 10% of fit cases.
   - It must be designed from fit-data failures only, may name no situation, scene or label, and is written as `ADDENDUM_1.md` with its own sha256 before step 4.
   - Under such a fix, both the original and the amended V1 are reported on evaluation data, and the amended one is primary.
   - Implementation bugs (code that does not do what this note says) are fixed without an addendum, and logged.
4. **Evaluation:** one pass over the primary runs, then the extras. Outputs go to `out_primary.txt`, `out_extras.txt` and a per-record `records.tsv` (one row per record × fix).
5. **Verification:** an independent verifier recomputes, with its own code, the class counts, oracle M1/M2/M3, V1 M1/M2/M5, and T1 catches and false triggers on the primary runs. Disagreements are reported, and the verifier's value is used when the two cannot be reconciled.

## 11. Report
For each fix (oracle, V1–V4, front-snap reference) and each model, in the primary runs:
- the moves;
- M1 by class and kind;
- M2 over 454 and 699, by kind;
- M3 and M5;
- the S1–S4 and kill lines, each with n.

Then M4, the I33 table, and the extras per model and run. Every number carries its n.

Anything not defined in this note is labelled post hoc, including any reading of patterns in the records, extra cuts and alternative thresholds.

## 12. Predictions (inference, written before scoring)
- **V1:** it can fix only lift-class records whose lift lands on the envelope. Of the 20 GLM + DeepSeek lift-class records, 11 are background leaks and 9 are occluder captures (results/still-scenes/overview/rerun-files/lifting), so V1 is likely to reach at most about 11 of 46, and a share of at most about 0.55. **S4 is therefore likely to fail.** V1 was still chosen as primary because it is the rule with the fewest expected breaks, not to pass.
- **V3:** it may add up to about 5 occluder captures, with breaks of unknown size.
- **T1:** from the window ranges and landing surfaces in `lifting_out.txt`, it is expected to catch about 11 of 15 (all 6 background leaks and the 5 straddling occluder captures). That is between the kill line (10) and the success line (12). The 4 occluder captures whose window lies wholly on the occluder cannot be flagged by any rule that reads only the lift window.
