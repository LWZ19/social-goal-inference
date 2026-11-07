# I09 phase 1c (plan step 3b): frozen protocol

The I09 lead wrote this on 17 Oct 2026, after the API probe below (21:21 +08:00) and before any detector or pose model processed an evaluation image. Its sha256 is in `FROZEN.sha256`. This file is never edited. Changes go into dated addenda with their own hashes, written before the data they affect is processed. Paths are relative to `results/still-scenes/people-finder/early-detector/phase1c/` unless they start with `phase1/`, `phase1b/` or `src/`.

- **Split.** The REPORT says "freeze on the example layouts, report once on 42 eval images". Here fitting uses I03's split: the worked examples plus 260 fresh-seed scenes. E-ph is added as a secondary domain.
- **Facing.** The REPORT says "shoulder-and-hip plane, sign from face keypoints". Here there are 4 declared variants, and fit data picks one.
- **A2 arms.** The REPORT says "constraints re-addressed to the detected people". Here that becomes D (primary), D-pix and D-keep.
- **Facing populations.** Section 10 names the population of each facing clause.
- **Pose model.** ViTPose only. RTMW would need installs in the read-only sidecar.
- **Confidence.** The REPORT's per-person confidence is the detector score. It is carried, and the solver does not use it.

**Exposure (disclosed).**
- So I know these facts about the evaluation scenes:
  - every realistic line member stands 74.5–107 deg from facing the camera;
  - the blobs miss 13 counter staff and give 6 false bodies in one layout;
  - Gemini's A0 facing is 0.906;
  - I01's come_after-listed detections lie 3.7–4.6 m from every true person.
- I viewed two fit images (`real_examples_02_waiting_line_2` and placeholder `examples_02_waiting_line_2`) and the probe output before fixing section 4's details.
- No evaluation image has been opened, viewed or run.

- **What ran.** The project venv wrote the path of ONE fit image (`real_examples_02_waiting_line_2`) through the harness. In the sidecar, RT-DETRv2-r50vd, D-FINE-large and ViTPose-base-simple were each loaded once from their snapshots and run on it.
- **Detectors.** `RTDetrImageProcessor` (640×640) for both. `post_process_object_detection(threshold, target_sizes=[(H, W)])` returns absolute xyxy `boxes`, `labels` and `scores`. Person is label 0.
- **ViTPose.** `VitPoseImageProcessor` (256×192) takes COCO [x, y, w, h] boxes. Its post-processing returns absolute pixel `keypoints` (17×2) and `scores`, which can exceed 1 (1.02 seen). `id2label` gives 0 Nose, 1/2 L/R_Eye, 3/4 L/R_Ear, 5/6 L/R_Shoulder, 11/12 L/R_Hip.
- **Left and right are the person's own.**
  - The obliquely camera-facing staff member has L_Shoulder image-right of R_Shoulder (u 531 against 514).
  - The two profile line members facing image-right have near-coincident shoulders (Δu 2.5–4.2 px) and the nose image-right of both ears.
- **Partial boxes.** Partial (head or upper-body) boxes of people already boxed scored 0.11–0.36.
- **Peak memory** was 457 MiB.
- No threshold was set from the probe.

## 1. Data (identical to I03, `results/still-scenes/perception/depth-blob-detector/FROZEN.md` section 1)
**Fit data.** Everything is chosen on these scenes only, in `results/still-scenes/perception/depth-blob-detector/stageA/fitdata.py` `fit_refs()` order, loaded with its `load()`:
- **F-real**: the 21 realistic worked examples, 39 visible people;
- **F-ph**: the 8 placeholder worked examples plus the 260 scenes of fresh seeds 100–109 (already generated, never regenerated), 268 scenes and 585 visible people.

**Evaluation data,** scored once: the 68 scenes of `H.distinct_scenes()`. The set and domain come from the scenes folder (`results/still-scenes/perception/depth-blob-detector/stageA/evaluate.py` `set_of`).
- **E-real** (PRIMARY): 42 images, 14 layouts × real/staffed/plain, 118 visible people.
- **E-ph** (secondary): 26 scenes. These are `data/placeholder/eval` `eval_00`–`13`, plus `eval_14`, `15`, `20` and `21` of `keep/dev_new/seed{0,1,2}/scenes`.

**Visibility.** Fit scenes use `configs/default.yaml`. Evaluation scenes use the run's recorded config, asserted equal across a scene's runs (`thresholds_for`).

## 2. Sidecar and project side
**The sidecar** (`sidecar/infer.py`; `-I`, HF offline, never imports sgi):
- **Input.** It reads image paths only from manifests that a project-venv script wrote through the harness (`H.examples`/`H.check_example`, `H.distinct_scenes`/`H.guard_scene`) and `fitdata.py`. It checks each image's sha256.
- **Path guard.** It asserts that no path contains `keep/heldout`, `s18`, `s20_held`, `eval_16`–`eval_19`, `group_photo` or `lift_wait`.
- **Seed folders.** A path component `seed<k>` must be seed100–109 directly under `results/still-scenes/perception/fresh_seeds`, or seed0–2 directly under `keep/dev_new`.
  - I read the rule as protecting seeds 3–99.
- **Writes** (JSONL, one line per image):
  - every person box with score ≥ 0.05 (xyxy, score);
  - for every box with score ≥ 0.10, ViTPose's 17 (u, v, confidence).
  - A meta JSON adds the GPU, driver, torch/CUDA/transformers versions, timestamps and peak memory.
- **Run settings.** fp32, eval mode, `torch.manual_seed(0)`, cudnn deterministic, benchmark off, TF32 off. Detector batch 1; pose takes one batch per image.
- **GPU check.** `nvidia-smi` runs before each job. If another process holds more than 500 MiB, the job waits (polled, logged).

**The project side** (torch-free venv): the lift, filters, facing, scoring and replays.

**Order guard.** `manifest_eval.json` is written only after `stageA/FIT.sha256` verifies. The evaluation pass runs only the detector in `chosen.json`.

## 3. Pipeline (fixed; only section 5's grid is fitted)
**Conventions.** Pixel (c, r) covers [c, c+1) × [r, r+1), and its centre is (c+0.5, r+0.5). Depth is camera Z. Lifting is `sgi.geometry.lift`/`lift_all` arithmetic. A keypoint is *confident* at a score ≥ 0.30.

For each person box with score ≥ τ:
1. **Torso region.**
   - When all four of L/R shoulder and L/R hip are confident, the region is the pixels whose centres lie in their convex hull, provided at least 20 have valid depth.
   - Otherwise it is the box's central region: u ∈ [x0+0.3w, x0+0.7w], v ∈ [y0+0.2h, y0+0.5h].
2. **Depth band.** Z_med is the median valid depth of the region. Pixels with |Z − Z_med| ≤ 0.30 m are kept. The box is rejected when fewer than 10 are kept.
3. **Position.** The coordinate-wise median of the kept points' world x and y, pushed d_push farther along the ground line from the camera's xy (as `sgi.stages.people._centre` does).
4. **Plausibility.** z_top is the height of pixel (box centre u, y0) back-projected at Z_med. The box is rejected unless h_lo ≤ z_top ≤ h_hi. The lower bound is skipped when y0 < 2 px.
5. **Flat test** (when on).
   - Q is the lifted valid pixels in the box, even rows and columns.
   - A plane is fitted (centroid and least singular vector) to step 2's kept points.
   - The box is rejected when at least 0.80 of Q lies within 0.04 m of that plane. With |Q| < 30, it is not flat.
6. **Duplicates.** In descending score, a box is dropped when its ground point lies within d_dup of one already kept.
7. **Output** in descending score: (x, y, yaw or None, score, box, torso pixel).

No situation, scene or label is read.

## 4. Facing (world yaw, as truth: radians, anticlockwise from +x)
**Used keypoints.** A keypoint is used when it is confident, lifts with `sgi.geometry.lift` (5×5), and has depth within ±0.40 m of Z_med.

**Lateral vector.** ℓ = P(L) − P(R), horizontal, for each shoulder or hip pair whose two keypoints are used. Facing is ℓ turned by −90 deg: yaw = atan2(−ℓx, ℓy).
- **F1.** Shoulders only. Yaw is None when the pair is unusable or |ℓ| < 0.05 m.
- **F2.** The mean of the usable shoulder and hip vectors, same None rule.
- **F3.** F2, then: if nose, L_Eye and R_Eye are all confident and cos(yaw − bearing(person → camera xy)) < 0, add 180 deg. This never creates a yaw.
- **F4.** F3, plus a profile rule.
  - **Edge-on:** both shoulders confident and |u_LS − u_RS| · Z_med / fx < 0.12 m (about 69–111 deg from the camera for a 0.33 m span).
  - When edge-on with a confident nose:
    - u_ref is the mean u of the confident ears, or the shoulder midpoint;
    - r̂ is the unit vector from the camera's xy to the person;
    - yaw = atan2(r̂) − 90 deg if u_nose > u_ref, or + 90 deg if u_nose < u_ref.
  - F3's yaw is kept when |u_nose − u_ref| < 1 px, or when the rule does not apply.

**Unit check** (before selection).
- **Synthetic.** L at +y and R at −y gives yaw 0. With the camera at the origin looking along +y, a person at (0, 3) with the nose image-right gives yaw 0 under F4.
- **Fit data, chosen setting.** Among matched people whose true angle to the camera is < 45 or > 135 deg, with both shoulders used and not edge-on, F1's share with error < 90 deg must exceed 0.5 in each fit domain.
- A failure is a sign bug. Its code fix is logged in `DEVIATIONS.md`.

**Selection.**
- The metric is within 45 deg, unknown counted as wrong, over the chosen setting's matched fit people, per domain.
- Maximise the lower domain. Ties go to the higher pooled value, then the earlier variant.

## 5. Grid and selection (fit data only)
**Grid** (192 settings): detector {RT-DETRv2-r50vd, D-FINE-large} × τ {0.25, 0.40, 0.55, 0.70} × [h_lo, h_hi] {[1.00, 2.30], [1.30, 2.10] m} × flat {off, on} × d_dup {0.30, 0.50 m} × d_push {0.00, 0.10, 0.20 m}.

**Metrics.** Per fit domain D, with the scorer and `exempt_xy=exempt_points(truth)` as I03:
- recall_D (pooled, 0.5 m);
- fb_D, false people per scene, exempt ones not counted;
- med_D, the median xy error.

**Rule.**
1. Feasible means fb_D ≤ 0.10 in both domains.
2. Maximise min_D recall_D.
3. Break ties in this order:
   - lower max_D fb_D;
   - lower max_D med_D;
   - the faster detector: lower median detector-only latency over the 289 fit images, after 5 warm-up images;
   - the least restrictive setting: lower τ, then the wider band, flat off, smaller d_dup, smaller d_push.
4. **If nothing is feasible** (as I03): the lowest max fb with min recall ≥ 0.95; else the highest min recall with max fb ≤ 0.30; else the lowest max fb. The fallback is stated.

**Freeze.** `stageA/fit_grid.txt` (192 rows), `stageA/fit_rows.jsonl` and `stageA/chosen.json` (setting, facing variant, unit check) are hashed into `stageA/FIT.sha256` before any evaluation manifest exists.

- It is considered only if the choice misses a section 10 success bar measurable on fit data (recall overall and on line members, false, xy, the three facing bars) in either fit domain.
- It is one revision, of detection or of facing, designed from fit failures only. It names no situation or scene.
- It is written as a hashed `ADDENDUM_FIX.md` before evaluation and re-fitted with the same grid and rule.
- It is adopted only if the rule over the union of rows (for facing, of variants) picks it. The kill bars are judged after it.

## 6. Scorer and A1 report
**Scorer.** `results/still-scenes/perception/depth-blob-detector/people_scorer.py`, unchanged: `score_people(people, scene, th, exempt_xy=exempt_points(scene.truth), facing="full")`, `pool` and `wilson`.

**Report** per set (real, staffed, plain, placeholder), per domain and pooled, each with its n:
- recall at 0.5 m (k/n, Wilson 95%) and by role (line, group, staff, other);
- false people per image, plus raw, exempt and near-hidden counts;
- merged neighbour pairs, and the median xy error;
- misses by class.

**Facing:**
- within 45 deg and the median error, by A0 view class (frontal < 45, oblique 45–75, profile 75–105 with both ends in, back > 105), from the true yaw against the bearing from the true person to the camera's xy;
- non-profile = frontal + oblique + back;
- profile front/back correct = error < 90 deg;
- coverage = matched people with a known yaw.

Each facing figure is given with unknown counted as wrong (180 deg in medians) and among known, and the axis version is given too. E-real recall and facing also get layout cluster-bootstrap intervals (2000 draws, seed 0).

## 7. Like for like (as `results/still-scenes/overview/phase-notes/phase1b.md` Table 1.2)
On E-real (arm, scene, trial) keys with a people list (`results/still-scenes/perception/depth-blob-detector/stageA/vlm_rows.jsonl`):
- each of the six models' recall, false per answer, merged pairs and full-yaw within 45;
- beside them, the detector and the blobs (`results/still-scenes/perception/depth-blob-detector/stageA/rows.jsonl`), each re-weighted to that model's keys, with an axis comparison for the blobs.

Also given: E-ph for Qwen, GLM and DeepSeek, and a sensitivity row in which keys without a list count as empty lists.

## 8. A0-matched facing (`results/still-scenes/facing/a0_pairs.jsonl`; arm, scene, pid)
For each model's pairs, the detector's yaw for the same true person is scored (scorer matching on that scene).
- **Coverage:** pairs whose person the detector matched with a known yaw.
- **Primary measure:** within 45 over detector-matched pairs, unknown counted as wrong. Unmatched pairs are recall misses: counted, and left out.
- **Sensitivities:** among known; and all pairs, with unmatched counted as wrong.
- **Weighting:** pair-weighted and on distinct (arm, scene, pid) instances.
- **Context:** the model's own value on the same subset, and a layout bootstrap (2000 draws, seed 0).
- **Kill population:** Gemini's pairs (310 pairs, 104 instances, 36 people, 8 layouts). The bars are 0.806 and 0.70.

## 9. Runtime (4060)
- **Sidecar.** The deployed path with the frozen detector: an image decoded in memory → detector pre-processing, forward and post-processing at τ* → ViTPose on boxes ≥ τ* → keypoints on the CPU (synchronised).
  - Batch 1, fp32. 5 warm-up passes on F-real images, then each of the 42 E-real images timed once. Median and p90.
  - Keypoints must agree with the raw pass within 1 px (checked and reported).
- **Project side.** Steps 1–7 plus facing per image on the CPU (2 threads), with inputs in memory. 5 warm-up images, then median and p90 over the 42.
- **Criterion.** median(sidecar) + median(project) ≤ 1.0 s. The p90 sum and peak GPU memory are reported (`max_memory_allocated`, polled `nvidia-smi`).

## 10. A1 criteria (verbatim REPORT I09; judged on E-real; E-ph is reported against the same bars)
**Success** needs all of:
- recall at least 0.95 overall and on line members;
- at most 0.10 false people per image;
- median position error at most 0.15 m;
- facing for non-profile people: median at most 25 deg, at least 85% within 45 deg;
- profile front/back correct at least 90%;
- warm runtime at most 1.0 s on the 4060.

The success facing clauses use all matched E-real people by view class, with unknown counted as wrong.

**Kill** fires on any of:
- recall below 0.90;
- more than 0.30 false people per image;
- facing within 45 deg under 70%, or more than 10 points below the best VLM.

The kill's facing clauses use section 8's primary measure on Gemini's pairs: a value below 0.806 fires. The school-panel reading (bar 0.70) is reported beside it. E-ph has no A0 pairs, so this clause is not judged there.

**Overall I09:** success needs both A1 and A2 success. A kill fires on any A1 or A2 kill clause.

## 11. A2: solver replays (no change to `src`; harness only, as `results/still-scenes/replay-harness/repairs/repairs.py`)
**Records.**
- Primary: the old wording, s19 `qwen38_low` and `glm53`, and s20 `dsv41`. All con, realistic arms, 252 each.
- Extras: the new wording, s21 Qwen and s23 GLM, DeepSeek, Haiku, Sonnet and Gemini. The Qwen, GLM and DeepSeek people answers there are cache copies of s19/s20.
- Further extras: the whitelisted placeholder C runs, s19/s20 `ph` and `dev_seed0`–`2`, 156 each.

**Solving.**
- Each solver step is solved again with `H.resolve(rec, people=, constraints=)`, with the recorded target unchanged, as repairs.py's P arm does. The solver re-derives the target person from the new list by position (`sgi.solver.target_person`).
- Records that are not solver steps, including every record without a people list, keep their recorded verdict.
- A solver step with an empty list gets every detector person added.

**Arms.**
- **D (primary).**
  - VLM detections are matched to the scene's frozen detector people with `sgi.run._matched_people` semantics: greedy nearest first, one to one, within 0.5 m in xy; ties go to the lower number, then the earlier detector person.
  - A matched detection takes the detector person's xy and yaw (its own recorded yaw when the detector's is None) and keeps its number, label and pixel.
  - Unmatched detections are dropped, and their numbers are stripped from every constraint (`repairs.strip`).
  - Unmatched detector people are added in detector order as M+1, M+2, …, where M is the largest number of any listed person or any constraint (the I03 number-collision lesson). Each gets the detector yaw, pixel = projection of (x, y, `person_mark_height_m`) (or the box centre), and label None.
  - No merge or cut is applied; pairs closer than 0.25 m are counted.
- **D-pix** (secondary, data-informed). As D, but the candidates are the detector people whose box contains the detection's recorded pixel (edges included). They are taken greedily one to one by the pixel distance to the box centre; ties go to the lower number, then detector order.
- **D-keep** (secondary). As D, but unmatched VLM detections stay where they were recorded, and nothing is stripped.

**Baselines** (same harness):
- the recorded C, and its step-0 replay (≤ 1e-6 m);
- I01's P, R3 and R4 from the unchanged `repairs.solver_row`, asserted equal, record by record, to `phase1/i01_repairs/rows_*`; for the placeholder runs, computed fresh.

**Criteria** (verbatim): "In A2, GLM and DeepSeek within 3 records of ground-truth people and at least the recorded C; Qwen loses at most 2." Kill: "A2 below recorded C for both GLM and DeepSeek." They are judged on D, old wording.
- Ground-truth people is I01 P over all 252: GLM 202, DeepSeek 193.
- **Success:** GLM ≥ 199 (and ≥ 186), DeepSeek ≥ 190 (and ≥ 182), Qwen ≥ 232 (234 − 2, net).
- **Kill:** GLM < 186 and DeepSeek < 182.

**Report** per arm and model:
- successes of 252 (flips and breaks against C);
- lines of 90, groups of 54 and nobody near of 108;
- the pre-registered line failures flipped (GLM 29, DeepSeek 15, Qwen 2; RERUN R01);
- the 25 fixable line failures recovered (2 + 15 + 8);
- overlap with P's flips;
- clustering: the people answers and layouts behind the flips and breaks;
- dropped VLM detections matched to a visible true person, by role;
- added detector people, matched or false;
- an assertion that no added number appears in any constraint.

Secondary arms and extras are reported against the same bars, labelled as such.

**Prediction** (inference): D lands between I01's R4 and P for GLM (200–202) and DeepSeek (188–193), and Qwen stays at 234–236.

## 12. Order and honesty
1. This note and `FROZEN.sha256`.
2. The pipeline code and its synthetic checks.
3. `manifest_fit.json`, then the sidecar fit pass: 2 detectors, 289 images.
4. The fit grid, the unit check, facing selection, then `FIT.sha256`.
5. The fix addendum, if one applies.
6. `manifest_eval.json`, the sidecar evaluation pass (68 images, chosen detector) and the timing pass.
7. Sections 6–8 and 10, scored once; then A2.

Every step logs timestamps, and hashes are verified on use. Every number has its n. Undeclared analyses are labelled post hoc. Choices made after seeing results on these runs are labelled data-informed. Deviations go to a hashed `DEVIATIONS.md` before the data they touch is processed. No provider or network call is made. No held-out scene, run or seed 3–99 is read. `src/`, `tests/`, `docs/` and `configs/` are unchanged.
