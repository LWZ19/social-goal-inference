# Phase 1, batch 2: results (2026-10-17)

Three zero-call analyses from plan step 4 (REPORT section 1.2; section 6, step 3): I03 Stages A–B (the depth-blob audit), I11 Stage 0 (where wrong targets come from, and snap-and-lift) and I33's Stage B trigger, which I11's protocol note also covers.

How they were made:
- Each analysis has a protocol note written and hashed before any evaluation scene or record was scored: `results/still-scenes/perception/depth-blob-detector/FROZEN.md` (sha256 `8106f74a…b979a`) and `results/still-scenes/perception/lifting/FROZEN.md` (sha256 `11f7b327…9165`). Changes and clarifications after the freeze are logged in their own files, most of them hashed (section 6 says which were not).
- Every number was computed through the step-0 harness (`docs/model_notes/tools/research_harness.py`) from whitelisted development runs and scenes. The only other input is fresh seeds 100–109, generated in this batch and used as fit data only.
- An independent verifier then recomputed each analysis with its own code. Where the two differ, this page uses the verifier's final value. Where the verifier could not settle a point, both are given (section 6).
- No model or network call was made. No held-out scene, run or situation was read. Seeds 3–99 were not generated or opened; they are reserved for the I46 sealed set. Nothing in `src/`, `tests/`, `docs/` or `configs/` changed.

Paths below are relative to this folder (`phase1b/` in the research folder).
- `i03/`: the I03 note, the fix addendum (`ADDENDUM_FIX.md`), `DEVIATIONS.md`, `DEVIATIONS_POST_EVAL.md`, the people scorer (`people_scorer.py`), the detector (`blobs.py`), the seed wrapper (`gen_fresh.py`) and the fit tables (`fit_selection.json`, `fit_selection_fix.json`).
  - `results/still-scenes/perception/depth-blob-detector/stageA/`: `tables.txt`, the blob rows (`rows.jsonl`), the VLM rows (`vlm_rows.jsonl`) and `chosen.json`.
  - `results/still-scenes/perception/depth-blob-detector/stageB/`: `tables.txt`, `CLARIFICATIONS.md`, per-record rows (`rows_*.jsonl`) and `posthoc.txt`.
  - Verifiers: `results/still-scenes/perception/depth-blob-detector/verify_i03_stageA/`, `results/still-scenes/perception/depth-blob-detector/verify_i03_stageB/`.
- `i11/`: the note, `DEVIATIONS.md`, `i11lib.py`, `run_eval.py`, `tables.txt`, `out_primary.txt`, `out_extras.txt` and per-record rows (`rows.jsonl`, 3,716 records). Verifier: `results/still-scenes/perception/lifting/verify_i11_stage0/`.
- `fresh_seeds/`: seeds 100–109 and `manifest.json`.

## 0. Verdicts at a glance

| Analysis | Pre-registered outcome | One-line reading (inference) |
|---|---|---|
| I03 Stage A (blob detector) | Neither success nor kill. Placeholder meets every success bar. Realistic misses recall (0.890) and false bodies (0.143 per scene) and crosses no kill bar | Blobs find every non-staff person. They miss counter staff who stand against tall structure. |
| I03 Stage B, add-only (primary) | Neither: 3 success bars missed; kill 1 misses by one record | Adding blobs does exactly what I01's oracle add (R3) did, and no more. |
| I03 Stage B, secondary arms | All bars met, no kill (secondary results, not the verdict) | The gain comes from relocating or dropping mislocated detections. Pruning also drops real staff. |
| I11 Stage 0 | Neither: S1–S3 met, S4 not met (0.45 against 0.70); no kill | Lift errors are real, and an oracle mask snap fixes them without breaking anything. Zero-call depth components get under half of that. |
| I33 Stage B trigger | Neither: 11 of 15 (bar 12; kill below 10), 6.0% false triggers | A cheap trigger catches background leaks. It misses occluder captures whose window holds no target pixel. |
| Fresh seeds 100–109 | Delivered: 260 scenes, 0 held-out episodes | Fit data for I03, and an extra domain for I11's component check. |

Shared cautions:
- **Realistic evaluation (E-real):** 42 images = 14 layouts × 3 arms (real, staffed, plain). The real and staffed arms share staff positions. **Placeholder evaluation (E-ph):** 26 scenes (`eval_00`–`13`, plus `eval_14`, `15`, `20`, `21` from dev seeds 0–2).
- **Records.** Primary records are the old wording, all C: Qwen and GLM from s19, DeepSeek from s20, 252 records each (nobody near 108, lines 90, groups 54). The new wording (s21 Qwen; s23 GLM, DeepSeek, Haiku, Sonnet and Gemini) and the placeholder runs are extras. At the new wording, the Qwen, GLM and DeepSeek people answers are cache copies of s19/s20.
- **Clustering.** Both episodes of a scene and trial share one people answer, so flips cluster. For example, B-add's 4 flips come from 2 answers.
- **Not blind.** I03's B-relocate arm, and I11's V3 and T1, were registered by authors who had seen earlier results on these runs. The notes disclose this. The blob pilot re-run had also already run on all 68 evaluation scenes, and its summaries shaped the I03 grid ranges.

---

## 1. I03 Stage A: the depth-blob detector (`i03/`, `results/still-scenes/perception/depth-blob-detector/stageA/`)

### What was done

- **Data split (frozen).**
  - Fit data: F-real = the 21 realistic worked examples (39 visible people). F-ph = the 8 placeholder worked examples plus 260 fresh-seed scenes (585 visible people).
  - Evaluation: the 68 whitelisted development scenes (`H.distinct_scenes()`), E-real and E-ph, each scored once.
- **Fresh seeds.** `results/still-scenes/perception/depth-blob-detector/gen_fresh.py` calls the package's `_write_scene` with the same random generators as `generate()`, and skips group_photo and lift_wait rows before any random draw.
  - Seeds 100–109 give 26 scenes each (18 eval, 8 examples): 260 scenes (no_one 110, waiting_line 70, conversation_group 50, audience 30) and 520 episodes.
  - 0 rejected layouts and 0 held-out episodes. Mean 4.13 s per scene (1.9–9.8 s).
  - Because of the skip, `eval_20`, `eval_21` and `examples_07` differ from what `generate()` would write.
- **Detector** (numpy; the pipeline is fixed, only the thresholds are fitted):
  - lift every valid pixel and keep the height band [z_lo, z_hi];
  - build a 5 cm height map (world-anchored) and its 8-connected components;
  - split them at local maxima: seeds are window maxima within split_r, and each cell goes to its nearest seed;
  - keep body-sized blobs: top height in [top_lo, z_hi − 0.05] and footprint in [foot_lo, foot_hi], where the footprint is the extent along the first principal axis plus 0.05 m;
  - position = the mean of the cell centres; facing axis = the minor principal axis.
- **Grid:** 216 settings. z_lo {0.10, 0.40}, z_hi {1.90, 2.05, 2.20}, top_lo {1.10, 1.30, 1.50}, split_r {0.15, 0.25, 0.35}, foot_lo {0.15, 0.30}, foot_hi {0.70, 1.00}.
- **Selection.**
  - Feasible means at most 0.10 false bodies per scene in each fit domain. Among feasible settings, take the highest lower-of-the-two-domain recall.
  - Ties go to the lower worst-domain merged share, then the lower worst-domain false-body rate, then the least restrictive setting.
  - The table and the choice were hashed (`results/still-scenes/perception/depth-blob-detector/fit_selection.json`) before any evaluation scene was loaded.
- **One general fix.** It is allowed only if the choice misses a fit bar, must be designed from fit failures only and written as a hashed addendum, and is adopted only if the same rule picks it.
- **Scorer** (`results/still-scenes/perception/depth-blob-detector/people_scorer.py`; I09 A1 is meant to reuse it):
  - visible = `sgi.gt.derive.is_visible`; matching = `sgi.run._matched_people` (greedy, nearest first, within 0.5 m); recall = matched / visible;
  - false = an unmatched detection not within 0.5 m of the anchor of a target that is not a person;
  - a neighbour pair = two visible people < 0.8 m apart; it is merged when they are not both matched and one detection lies within 0.5 m of both;
  - facing within 45 deg, as full yaw or as an axis (modulo 180);
  - staff behind counters = missed staff whose lower body is hidden (at least 2 of the samples at z 0.3, 0.5 and 0.7 m are occluded by more than 0.3 m).
- **VLM lists.** One answer per (arm, scene, trial) = the union of that key's records' lists, scored with the same scorer and exemption.

### Fit and the one fix (fit data only)

- **The choice.**
  - 8 of 216 settings were feasible, all with F-real recall 33 of 39.
  - Chosen: z 0.40–1.90 m, top_lo 1.50 m (top_hi 1.85 m), split_r 0.35 m, footprint 0.30–0.70 m.
  - Fit metrics: F-real 33/39 = 0.846, 0 false, 0/9 merged. F-ph 579/585 = 0.990, 4/268 false, 0/143 merged.
  - Facing variant: torso band 1.0–1.6 m (F-real 24/33, F-ph 578/579), against 18/33 and 523/579 with all points.
- **The fix.** The choice missed the F-real recall bar. All 6 F-real misses were staff whose blob joined tall structure at the clip.
  - The fix, written and hashed before it ran, drops clip-reaching cells.
  - Re-fit: 0 of 216 settings were feasible. The lowest worst-domain false-body rate was 1.72 per scene, and the 34 settings with recall 1 in both domains had 6.1–10.1.
  - The rule on the union of the 432 rows kept the original detector.

### Main tables

Table 1.1. The blob detector on the evaluation scenes (frozen setting, scored once).

| Domain (scenes) | Visible | Recall [Wilson 95%] | False bodies (per scene) | Merged pairs | Median xy error (n) | Axis facing within 45 |
|---|---|---|---|---|---|---|
| **E-real (42)** | 118 | **105/118 = 0.890 [0.821, 0.934]** | **6 (0.143)** | 0/33 | 0.077 m (105) | 72/105 = 0.686 |
| real (14) | 42 | 36/42 = 0.857 | 2 | 0/11 | 0.077 m (36) | 25/36 |
| staffed (14) | 42 | 35/42 = 0.833 | 2 | 0/11 | 0.077 m (35) | 24/35 |
| plain (14) | 34 | 34/34 = 1.000 | 2 | 0/11 | 0.079 m (34) | 23/34 |
| **E-ph (26)** | 58 | **58/58 = 1.000 [0.938, 1.000]** | **0** | 0/11 | 0.067 m (58) | 57/58 = 0.983 |
| Pooled (68) | 176 | 163/176 = 0.926 [0.878, 0.956] | 6 (0.088) | 0/44 | 0.072 m (163) | 129/163 = 0.791 |

- Axis facing is taken modulo 180, so chance is 0.50. No axis was unknown.
- **Recall by role**, pooled over 68 scenes: line members 64/64, group members 60/60, others 36/36, staff 3/16. All staff stand in the real and staffed arms.
- **Misses.** 13, all staff: 8 behind counters and 5 other staff.
  - All 13 are "rejected: top above the clip". Each person's blob reaches 1.90 m, with footprints of 0.52–1.68 m.
  - Verifier: they come from 7 layouts. 6 layouts (`eval_01`, `02`, `03`, `07`, `08`, `13`) are missed in both the real and staffed arms. `eval_09`'s staff member is missed only in staffed.
- **False bodies.** Post hoc: all 6 are the same two blobs in the three arms of `eval_08_waiting_line_4` (top 1.79 m, footprints 0.50 and 0.61 m, 3.4 m from any person). 0 blobs were exempt, and 0 false blobs lie near a hidden person.
- **Merges.** 0 merged detections in any scene.

Table 1.2. VLM people lists and blobs on E-real, same scorer. The unit is answers with a people list: 126 keys per model, 40–54 of them without a list, all in no_one scenes.

| Model | Recall | False per answer | Merged pairs | Full yaw within 45 (axis) | Blobs on the same keys: recall / false per key |
|---|---|---|---|---|---|
| Gemini 3.8 low | 310/312 = 0.994 | 0.028 | 0/99 | 281/310 = 0.906 (282) | 0.933 / 0.250 |
| Haiku 5.5 | 308/314 = 0.981 | 0.081 | 0/99 | 180/292 = 0.616 (193); 16 unknown | 0.927 / 0.243 |
| Sonnet 5.5 low | 300/312 = 0.962 | 0.097 | 0/99 | 242/299 = 0.809 (257) | 0.933 / 0.250 |
| Qwen 3.8 | 298/312 = 0.955 | 0.167 | 0/99 | 228/298 = 0.765 (264) | 0.933 / 0.250 |
| DeepSeek V4.1 Flash | 295/331 = 0.891 | 0.337 | 1/99 | 181/294 = 0.616 (192) | 0.906 / 0.209 |
| GLM-5.3 | 267/317 = 0.842 | 0.342 | 2/99 | 146/267 = 0.547 (162) | 0.924 / 0.237 |

- **E-ph** (78 keys per model, 36 without a list): Qwen 167/168 with 0 false; GLM 136/168 = 0.810 with 0.524 false per answer; DeepSeek 150/168 = 0.893 with 0.381. Blobs on the same keys: 168/168, 0 false.
- **Sensitivity.** Counting keys without a list as empty lists, E-real recall drops to GLM 0.754, DeepSeek 0.833, Qwen 0.842, Sonnet 0.847, Haiku 0.870 and Gemini 0.876.
- **Which keys.** Keys without a list are all no_one scenes, where most counter staff stand. That is why the blob figures are re-weighted to each model's keys.
- **Facing is not like for like.** Blobs give an axis only, with no front or back. Compared axis to axis, the blobs (192–201 of 291–300, about 0.66) sit with Haiku (0.661) and DeepSeek (0.653), above GLM (0.607) and below Qwen (0.886), Sonnet (0.860) and Gemini (0.910).

### Checks

- **Self-check.** Given the true people as detections, the scorer gives recall 1, 0 false, 0 merges and facing 1 in 289 of 289 fit scenes.
- **Matching.** It reproduces the recorded `people_found_share` × visible and `people_unmatched`: the compute agent checked 96 of 96 records (8 runs, 6 models), the verifier 1,113 of 1,113.
- **Verifier's own detector and scorer.**
  - All 432 rows of `fit_selection.json`, all 432 of `fit_selection_fix.json` and all 68 evaluation rows match.
  - The frozen scorer and the verifier's agree on 900 of 900 random detection sets.
  - All 990 VLM keys match.
- **Post hoc (verifier).** Shifting the grid by half a cell changes no count. Only the median xy error moves, by 0.002 m.
- **Order.** FROZEN.md 18:42 → seeds → `fit_selection.json` 19:12 → `ADDENDUM_FIX.md` 19:14 → `chosen.json` 19:16:21 and its hash 19:16:50 → first evaluation scene loaded 19:16:57. Every recorded hash verifies.
- **Incident P1.** The first evaluation pass wrote `rows.jsonl`, then stopped in the VLM part on a set-name lookup (`keep/placeholder_first14`). The second pass read the blob rows back (same sha256) and scored only the VLM lists. No threshold, detector or scorer changed between the passes.

---

## 2. I03 Stage B: solver replays with the blobs (`results/still-scenes/perception/depth-blob-detector/stageB/`)

### What was done

- **Records.** Primary: the old wording (s19 `qwen38_low` and `glm53`, s20 `dsv41`; con; realistic arms; 252 each). Extras: the new wording (6 models), and the placeholder C runs (`ph` + `dev_seed0`–`2`, 156 records per model).
- **Solving.**
  - Each solver step is re-solved with `H.resolve(rec, people=, constraints=)` and the recorded target. Every other record keeps its verdict.
  - A record without a people list never gets blobs. The methods never read situation or scene labels.
- **Blobs.** The frozen Stage A kept blobs, with no fix: 169 in 68 scenes.
- **Terms** (xy distances):
  - supported = a detection with a blob within 0.5 m;
  - unexplained = a blob with no listed detection within 0.5 m;
  - target blob = a blob within 0.5 m of the record's grounded target point. It is never added.
- **Arms:**
  - **B-add** (primary): add every unexplained blob that is not a target blob as a person. Numbers run from the highest listed number + 1; yaw None; pixel = the projection at 1.1 m.
  - **B-prune**: B-add, plus dropping unsupported detections and stripping their numbers from the constraints. A detection is exempt when the depth cannot see its spot (fewer than 3 of 9 axis samples observable).
  - **B-relocate** (data-informed; registered after I01): pair each unsupported detection one to one with an unexplained blob whose projection at 1.0 m lies within 40 px of the detection's pixel. Move it to its blob, keeping its number, yaw, pixel and label. Then B-add the rest.
  - **B-relocate-prune**: B-relocate, then prune what is still unsupported.
- **Fixable offline** = a recorded failure that I01's P arm flips. A **line failure** is as in RERUN R01: 46 pooled (GLM 29, DeepSeek 15, Qwen 2), of which 25 are fixable (2 + 15 + 8).
- `CLARIFICATIONS.md` was written and hashed before step 0.

### Main tables

Table 2.1. Old wording, successes of 252 (flips / breaks against recorded C), and line successes of 90.

| Arm | Qwen | GLM | DeepSeek | Qwen lines | GLM lines | DeepSeek lines |
|---|---|---|---|---|---|---|
| C (recorded) | 234 | 186 | 182 | 86 | 42 | 68 |
| **B-add (primary)** | 234 (+0/−0) | 187 (+2/−1) | 184 (+2/−0) | 86 | **44** | **70** |
| B-prune | 236 (+2/−0) | 198 (+13/−1) | 189 (+7/−0) | 88 | 55 | 74 |
| B-relocate | 236 (+2/−0) | 200 (+15/−1) | 185 (+3/−0) | 88 | 57 | 71 |
| B-relocate-prune | 236 (+2/−0) | 200 (+15/−1) | 189 (+7/−0) | 88 | 57 | 74 |
| I01 R3 (oracle add) | 234 (+0/−0) | 188 (+2/−0) | 184 (+2/−0) | 86 | 44 | 70 |
| I01 R4 (drop unmatched) | 236 (+2/−0) | 200 (+15/−1) | 188 (+6/−0) | 88 | 57 | 72 |
| I01 P | 236 (+2/−0) | 202 (+16/−0) | 193 (+15/−4) | 88 | 57 | 75 |

- The I01 rows are re-run in the same harness and match `results/still-scenes/overview/phase-notes/phase1.md`.
- The single GLM break under all four arms is one record (`real_eval_10_conversation_group_2` e1 t1); see the post hoc findings below. DeepSeek's groups gain 1 under the prune arms.

Table 2.2. Pre-registered line failures flipped, and fixable line failures recovered (old wording).

| Arm | GLM (29) | DeepSeek (15) | Qwen (2) | Fixable recovered, of 25 (Qwen / GLM / DeepSeek) |
|---|---|---|---|---|
| B-add | 2 | 2 | 0 | **2** (0/2, 0/15, 2/8) |
| B-prune | 13 | 6 | 2 | 18 (2/2, 13/15, 3/8) |
| B-relocate | 15 | 3 | 2 | 16 (2/2, 11/15, 3/8) |
| B-relocate-prune | 15 | 6 | 2 | 18 (2/2, 13/15, 3/8) |
| I01 R3 / R4 / P | 2 / 15 / 15 | 2 / 4 / 8 | 0 / 2 / 2 | 2 / 21 / 25 |

- **Overlap with I01.**
  - B-add's 4 flips are I01 R3's flips, record for record.
  - All 13 of B-prune's GLM flips are also P and R4 flips.
  - B-relocate's 15 GLM flips include 4 that P does not flip (`staffed_eval_06` ×2, `plain_eval_07` ×2).
  - Each prune arm flips 7 DeepSeek records: 4 that P flips and 3 that it does not.
- **Over all old-wording failures** (fixable 2 / 16 / 15 of 18 / 66 / 70 failures; Qwen / GLM / DeepSeek):
  - B-add recovers 0 / 0 / 2;
  - B-prune and B-relocate-prune recover 2 / 13 / 4;
  - B-relocate recovers 2 / 11 / 3.
- **Effective n.**
  - B-add's 4 flips come from 2 people answers in 2 layouts (GLM `staffed_eval_06` trial 1; DeepSeek `real_eval_13` trial 2).
  - B-prune's 13 GLM flips come from 8 answers in 2 layouts (`eval_09`, `eval_13`), and B-relocate-prune's 15 from 9 answers in 3 layouts.
  - The prune arms' 7 DeepSeek flips come from 6 answers in 4 layouts, and Qwen's 2 flips from 1 answer.

### Mechanism

- **What B-add adds** (old wording, solver steps): Qwen 49 blobs (13 real missed people, 36 false), GLM 109 (79 / 30), DeepSeek 91 (59 / 32). Every false addition is one of Stage A's two false blobs in `eval_08_waiting_line_4` (verifier: all of them, not "almost all").
- **Why adding barely helps.** On GLM's 29 line failures, B-add added 44 blobs, 40 of them real missed people, in 24 of the 29. An added body joined the line (the solver's `joined_waiting`) in only 2 of them, and those 2 are the 2 flips.
- **Why the line failures B-add leaves still fail.** The classes were fixed before solving; the first that applies is used.
  - GLM, 27 left:
    - 9 have no `come_after` used (8 state none, 1 an empty one); none of them is P-fixable;
    - 18 name a waiting number that no blob supports (23 such numbers); these hold all 15 P-fixable failures.
  - DeepSeek, 13 left:
    - 5 have an unsupported waiting number (4 of them P-fixable);
    - 7 have a visible member who is listed but neither named in `come_after` nor joined (2 P-fixable);
    - 1 has a complete line and still fails.
  - Qwen, 2 left: both have an unsupported waiting number (both P-fixable).
  - Under B-relocate-prune, GLM has 14 left: 11 with no `come_after` used, which is 2 more than B-add's 9 because stripping can empty a `come_after` (verifier's value), and 3 with a member left out of the line. DeepSeek has 9 left, all with a member left out of the line.
- **What prune and relocate act on.**
  - B-prune dropped 44 / 85 / 85 detections (Qwen / GLM / DeepSeek, summed over solver steps). Of these, 40 / 34 / 44 are matched to a real visible staff member the blobs miss, and 4 / 51 / 41 are matched to no one. Only 2–6 unsupported detections per model were exempt.
  - B-relocate made 2 / 23 / 23 pairs. None of the moved detections had been matched, and the median move was 4.17 / 4.16 / 3.545 m. These distances match I01's come_after-listed detections, which lie 3.7–4.6 m from every true person.
- **Out of reach.** Over all old-wording failures, people repairs cannot touch the 9 / 21 / 22 records that are not solver steps, or the 6 / 14 / 14 with a wrong target.

### Extras

Table 2.3. New wording, successes of 252.

| Model | C | B-add | B-prune | B-relocate | B-relocate-prune |
|---|---|---|---|---|---|
| Qwen | 228 | 228 | 230 (+2/−0) | 230 | 230 |
| GLM | 170 | 172 (+2/−0; lines 39 to 41) | 188 (+18/−0) | 186 (+16/−0) | 192 (+22/−0; lines 61) |
| DeepSeek | 185 | 188 (+3/−0; lines 68 to 71) | 194 (+10/−1) | 189 (+4/−0) | 194 (+10/−1) |
| Haiku | 240 | 240 | 239 (+0/−1) | 240 | 239 (+0/−1) |
| Sonnet | 242 | 242 | 242 | 242 | 242 |
| Gemini | 214 | 214 | 214 | 214 | 214 |

- **Line failures flipped by B-add:** GLM 2 of 32, DeepSeek 3 of 17, Haiku 0 of 11, Qwen 0 of 2, Sonnet 0 of 2. Gemini has no line failures.
- **Cache copies.** The Qwen, GLM and DeepSeek people answers are cache copies (72/72, 72/72 and 77/77 shared keys identical).
- For Haiku, Sonnet and Gemini, B-add adds only the 36 false `eval_08` blobs (Sonnet also 10 real people) and changes no verdict.

Table 2.4. Placeholder C runs (`ph` + dev seeds 0–2), successes of 156.

| Model | C | B-add | B-prune | B-relocate | B-relocate-prune |
|---|---|---|---|---|---|
| Qwen | 147 | 147 | 147 | 147 | 147 |
| GLM | 113 | 116 (+3/−0) | 124 (+12/−1) | 115 (+2/−0) | 124 (+12/−1) |
| DeepSeek | 116 | 116 | 122 (+6/−0) | 118 (+2/−0) | 124 (+8/−0) |

- Fixable recovered by B-add: GLM 1/13, DeepSeek 0/13, Qwen 0/1. By B-prune: 12/13, 6/13, 0/1.
- Blobs are near perfect on these scenes. Every B-add addition was a real missed person, and every pruned detection was matched to no one. So these rows mostly show the solver's ceiling for each arm.

### Post hoc findings (`posthoc.py`, written after the tables were read; it changes no arm)

- **(a) Number collision.**
  - When a model lists nobody but its constraints still name numbers, the added people take the numbers 1, 2, …, and those constraints bind to arbitrary blobs. This happened in 4 old-wording GLM steps, 5 new-wording GLM steps and 1 new-wording DeepSeek step.
  - It explains the only B-add break: old GLM `real_eval_10_conversation_group_2` e1 t1, where the heading was lost. It also explains one of GLM's two new-wording B-add flips (`staffed_eval_09` e1 t1).
  - The verifier renumbered added people above every constraint number: the break and that flip both go away. I01's R3 numbers people the same way.
- **(b) Pruning removes staff that the blobs miss.** This is the cause of the 2 new-wording breaks (DeepSeek t2 and Haiku t0, `real_eval_07_waiting_line_2` e0). The pruned detection is staff member s0, named in `come_after`.

### Checks

- **Step 0.** 1,301 of 1,301 realistic places and verdicts reproduce, and 248 of 248 placeholder ones.
- **Blob cache.** It equals Stage A in 68 of 68 scenes. The verifier's own detector gives the same 169 blobs in the same order (maximum difference 0.0).
- **Record by record.** The verifier compared 2,736 records × 4 arms: 0 verdict disagreements, and 0 disagreements in the added, pruned, exempt, moved, matched or joined counts.
- **Relocate sensitivity.** The verifier's B-relocate variant (unpaired instead of recomputed) gives the same 20 flips and 1 break.

---

## 3. I11 Stage 0: where wrong targets come from, and snap-and-lift (`i11/`)

### What was done

- **Primary.** The 9 old-wording realistic C runs: 756 records, 752 with a point. 57 wrong targets, 53 with a point, 46 of them GLM + DeepSeek. Right-target records with a point: 454 (GLM + DeepSeek) and 699 (all three).
- **Extras, never pooled with the primary:**
  - X-new: new wording, 18 runs, 1,512 records, 111 wrong;
  - X-ph: placeholder `ph` and dev-seed runs, 12 runs, 468 records, 17 wrong;
  - X-ph2: the s10 and s8 C runs, 15 runs, 980 records, 63 wrong; it holds 476 duplicate draws, so its counts are not independent.
- **Error classes.** `results/still-scenes/perception/lifting-rerun/lifting.py`'s rules, tested in this order:
  - choice error: no body-mask pixel within 15 px;
  - occluder capture: the 5×5 median depth lies more than 0.25 m in front of the target;
  - background leak: it lies more than 0.25 m behind;
  - floor slip: the lift lands below z = 0.05 m.
  - Lift classes are the last three.
- **Oracle snap-and-lift.** Snap to the nearest true-target body-mask pixel within 15 px, take the median of the body-mask depths in that pixel's 5×5 window, and lift its centre. It was applied to every record with a point, so its breaks can be counted.
- **Depth components** (no ML):
  - 4-neighbour components of the depth image, with the floor band (z < 0.15 m) removed;
  - neighbours are joined when their 3D gap is at most 0.10 m (G1);
  - pieces under 25 px are fragments, and a component whose robust horizontal extent exceeds 4.6 m is structure.
- **Variants.**
  - V1 (primary) snaps only when the lift lands on floor or structure: it moves to the nearest object component within 15 px and takes the component's median depth.
  - V2 always snaps.
  - V3 is V1 plus a step back to the object behind when the window straddles two objects (data-informed).
  - V4 is V1 with a depth-relative join (G2).
  - The front-surface snap is re-run as a reference.
- **Scoring.**
  - Fix = wrong → right; break = right → wrong, under the unchanged `sgi.evaluate.match_target`.
  - Full success is replayed via `H.resolve` or `standard_goal` + `score_pose`.
  - Share of the oracle gain = (variant fixes − breaks) / (oracle fixes − breaks), as counts on 46 and 454.
- **Fit checks** used the 29 worked examples only. No addendum was needed.

### Main tables

Table 3.1. Error classes of the wrong targets with a point (primary).

| Set | n | Choice error | Occluder capture | Background leak | Floor slip | Lift classes |
|---|---|---|---|---|---|---|
| Qwen | 7 | 1 | 5 | 1 | 0 | 6 |
| GLM | 24 | 18 | 1 | 5 | 0 | 6 |
| DeepSeek | 22 | 8 | 8 | 6 | 0 | 14 |
| **GLM + DeepSeek** | **46** | **26 (56.5%)** | 9 | 11 | 0 | **20 (43.5%)** |
| All three | 53 | 27 | 14 | 12 | 0 | 26 |

- **By target kind** (all 53): person 15 (occluder 9, background 6); parcel 15 (choice 9, occluder 5, background 1); place 23 (choice 18, background 5).
- The lift classes among the 46 come from 6 layouts (8 model × layout pairs). Over 50, counting the 4 GLM no-point records as non-lift, they are 20/50 = 40.0%.

Table 3.2. Fixes and breaks (primary).

| Fix | Fixes of 46 | Breaks of 454 | Fixes of 53 | Breaks of 699 | Full successes among the 46's fixes | Net successes, GLM + DeepSeek | Share of oracle gain |
|---|---|---|---|---|---|---|---|
| Oracle snap-and-lift | 20 | 0 | 26 | 0 | 16 / 20 | +16 | 1.00 |
| **V1 (primary)** | **9** | **0** | 9 | 0 | 7 / 9 | +8 | **0.45** |
| V2 (always snap) | 9 | 1 | 9 | 2 | 7 / 9 | +6 | 0.40 |
| V3 (+ step back; data-informed) | 10 | 0 | 14 | 0 | 7 / 10 | +8 | 0.50 |
| V4 (depth-relative join) | 9 | 0 | 9 | 0 | 7 / 9 | +8 | 0.45 |
| Front-surface snap (reference) | 7 | 24 | 8 | 37 | 6 / 7 | −16 | −0.85 |

- **The oracle** fixes 9 of 9 occluder captures and 11 of 11 background leaks among the 46, and 0 of 26 choice errors. Over the 53 it fixes person 15, parcel 6 and place 5. It moves 453 of the 454 right-target GLM + DeepSeek records without breaking one.
- **V1** fixes background leaks only (person 5, place 4). By design it cannot move an occluder capture, whose lift lies on an object. Over all three models, V3 adds 5 occluder captures, 4 of them Qwen's.
- **Clustering.** V1's fixes come from 4 layouts (5 model × layout pairs), the oracle's from 6 (8). The secondary shares are 0.35 (V1 over all three models) and 0.50 (V1 by net successes).
- **The front snap's breaks.** 36 of its 37 breaks are person targets.

Table 3.3. Component recovery of target masks (M4, G1 join; IoU against the best-overlapping component).

| Domain (targets) | Person, IoU ≥ 0.7 | Parcel merged with its table | Place, median IoU (IoU ≥ 0.7) | Place merged with another body | Split |
|---|---|---|---|---|---|
| Primary, realistic (42) | 12/12 | 3/3 (median IoU 0.31) | 0.87 (18/27) | 14/27 | 0 |
| Fresh seeds 100–109 (260) | 70/70 | 10/10 | 0.90 (103/180) | 0 | 6 places |
| X-ph (26) | 4/4 | 1/1 | 0.64 (0.99 with G2) | n/a | n/a |

- On the true targets of the 53 wrong records, the median IoU is 1.00 for occluder captures (n 14) and background leaks (n 12), and 0.65 for choice errors (n 27).
- No target was absorbed into structure.

Table 3.4. New wording (X-new; the blind check), per model.

| Model | Wrong targets | Oracle fixes (breaks of right) | V1 fixes (breaks) | V3 fixes |
|---|---|---|---|---|
| Qwen | 12 | 12 (0 of 240) | 1 (0) | 5 |
| DeepSeek | 24 | 19 (0 of 228) | 5 (0) | 6 |
| GLM | 33 | 17 (0 of 219) | 12 (0) | 12 (1 break) |
| Haiku | 1 | 1 (0 of 251) | 0 (0) | 1 |
| Sonnet | 7 | 7 (0 of 245) | 0 (0) | 7 |
| Gemini | 34 | 0 (0 of 218) | 0 (0) | 0 |

- **Gemini.** All 34 of Gemini's wrong targets are choice errors on place targets, in 5 layouts.
- **X-ph:**
  - oracle fixes GLM 5/10 and DeepSeek 2/7, with 0 breaks;
  - V1 fixes GLM 2/10 and DeepSeek 0/7, with 0 breaks.
- **X-ph2:**
  - the oracle fixes 23/63 and V1 12/63, both with 0 breaks;
  - V3 breaks 3, 4 and 6 records on qwen38, qwen36_35 and qwen3vl.
- **V1 breaks nothing anywhere.** It broke no right-target record in any set: 0 breaks over all 3,716 records.

### Checks

- **Reproduction.**
  - `lifting_out.txt`, `lifting_records.tsv` and `frontsnap_out.txt` reproduce byte for byte.
  - The lift reproduces with 0.0 m error over 752 records, and recorded `target_correct` disagrees with `match_target` 0 times in 752.
  - The vectorised mask disagrees with `match_target` on 0 of 31,800 sampled pixels.
- **Fit checks** (29 worked examples):
  - walls and ceiling are 99.6% structure, person pixels 99.9% object and target pixels 100% object;
  - only obstacles that touch a wall are labelled structure (66 of the 74 that do; none of the 41 free-standing ones);
  - at the 58 fit oracle targets, T1 flags 0 and V1 moves 0.
- **Verifier.** It re-implemented the method from the note alone, with its own union-find components, masks, windows and replay. Result: 0 field differences over all 3,716 records (primary 756, X-new 1,512, X-ph 468, X-ph2 980). The new world point of every fix agrees to 1e-9.
- **Pins.** All 7 pinned code files match their hashes, and `match_target` is unchanged.

---

## 4. I33 Stage B: the target-snapping trigger (`i11/`)

### What was done

- **T1** (primary; frozen in `results/still-scenes/perception/lifting/FROZEN.md` section 8) fires when the lift lands on floor or structure, or when the valid pixels of the 5×5 lift window carry at least 2 labels among floor, structure and object components (fragments ignored).
- **T2** (reported only) is T1, or any other object component within 15 px.
- **Catches** are counted on the 15 person wrong targets of the old wording (all three models).
- **False triggers** are counted over two denominators:
  - D1 (decisive) = the 201 right-target records whose true target is a person;
  - D2 = all 699 right-target records with a point.

### Main table

Table 4.1. Trigger catches and false triggers.

| Set | Person wrong targets caught | D1 false triggers | D2 false triggers |
|---|---|---|---|
| **Old wording, T1** | **11/15** (background leaks 6/6, occluder captures 5/9) | **12/201 = 6.0%** | 45/699 = 6.4% |
| Old wording, T2 | 15/15 | 103/201 = 51.2% | 188/699 = 26.9% |
| New wording, T1: Qwen | 4/8 | 6/64 = 9.4% | |
| DeepSeek | 5/10 | 2/62 = 3.2% | |
| GLM | 3/5 | 2/67 = 3.0% | |
| Haiku | 1/1 | 0/71 | |
| Sonnet | 7/7 | 8/65 = 12.3% | |
| Gemini | 0/0 | 14/72 = 19.4% | |
| New wording, pooled (post hoc) | 20/31 = 64.5% (occluder 13/23, background 4/4, choice 3/4) | 32/401 = 8.0% | 62/1,401 = 4.4% |

- **Which occluder captures.** The 5 caught are exactly those whose window holds target pixels (2–9 px). The 4 missed hold none.
- **All wrong targets.** On all 53, T1 flags 40 (choice 20/27, occluder 8/14, background 12/12).
- **Load.** T1 flags 85 of the 752 records with a point (11.3%), and 40 of those 85 are wrong targets (47.1%). T2 flags 236 (31.4%), 48 of them wrong.
- **Clustering.** The 15 come from 3 layouts (13 at `eval_12`; 5 model × layout pairs). The new-wording 31 come from 3 layouts (7 pairs). On X-ph, T1 catches 1 of 2 person wrong targets.
- **Not blind.** The note's author had read the per-record diagnostics and predicted 11 of 15. The new-wording rows are the blind check.

---

## 5. Pre-registered criteria and outcomes

### I03 Stage A (REPORT I03; FROZEN.md section 5: success needs every bar in both domains; a kill fires on any bar in either)

| Criterion | E-real | E-ph | Outcome |
|---|---|---|---|
| Recall ≥ 0.95 at 0.5 m | 0.890 | 1.000 | Not met (E-real) |
| False bodies ≤ 0.10 per scene | 0.143 | 0.000 | Not met (E-real) |
| Merged neighbours ≤ 10% | 0/33 | 0/11 | Met |
| Kill: recall < 0.85 after the one fix | 0.890 | 1.000 | Does not fire, though the E-real Wilson lower bound is 0.821 and F-real on fit data was 0.846 |
| Kill: false bodies > 0.3 per scene | 0.143 | 0.000 | Does not fire |
| Kill: merges > 20% | 0% | 0% | Does not fire |
| The one fix applied only if a fit bar is missed, and adopted only if it wins | | | Met (applied; not adopted) |
| Thresholds chosen on fit data and frozen before evaluation | | | Met (hashes and timeline verified) |
| **Overall** | | | **Neither success nor kill** |

### I03 Stage B (B-add is primary; FROZEN.md section 8)

| Criterion | Value | Outcome |
|---|---|---|
| Step 0 reproduces the recorded C | 1,301 / 1,301; 248 / 248 | Met |
| GLM lines ≥ 47 of 90 | 44 | Not met |
| DeepSeek lines ≥ 71 of 90 | 70 | Not met |
| No model loses more than 1 net on non-line records | Qwen 0, GLM −1, DeepSeek 0 | Met |
| Qwen total ≥ 234 | 234 | Met |
| ≥ 13 of 25 fixable line failures recovered | 2 | Not met |
| Kill 1: GLM gain ≤ 2 and DeepSeek gain ≤ 1 | +2 and +2 | Does not fire, by one record (DeepSeek's +2 is one people answer) |
| Kill 2: breaks ≥ flips over 756 | 1 vs 4 | Does not fire |
| Kill 3: > 60% unfixable and the live check gains < 2 of 10 | 21/46 = 46% unfixable; no live check run | Cannot fire |
| **Overall (B-add)** | | **Neither success nor kill** |
| Prediction in the note (inference): B-add trips kill 1 | | Did not hold, by one record |

### I11 Stage 0

| Criterion (REPORT I11) | Value | Outcome |
|---|---|---|
| S1: lift classes ≥ 40% of the 46 (≥ 19) | 20/46 = 43.5% (20/50 = 40.0%) | Met |
| S2: oracle fixes ≥ 15 of 46 and breaks ≤ 2 of "about 400" (454) | 20; 0 | Met |
| S3: ≥ 8 of those become full successes | 16 | Met |
| S4: depth components reach ≥ 70% of the oracle gain (V1) | 0.45 | Not met |
| Kill: lift classes < 20%, fixes < 6, or breaks > 5 | 43.5%, 20, 0 | Does not fire |
| **Overall** | | **Neither success nor kill** |
| The front-snap reference reproduces 8 fixes and 37 breaks | 8 / 37 | Met |
| Stage 1 gate input: choice errors ≥ 30% of the 46 | 26/46 = 56.5% | Holds (Stage 1 also needs family F) |
| Prediction in the note: V1 at most about 11 of 46, share at most about 0.55 | 9 of 46, 0.45 | Held |

### I33 Stage B

| Criterion (REPORT I33) | Value | Outcome |
|---|---|---|
| The trigger catches ≥ 12 of 15 person wrong targets with ≤ 10% false triggers | 11/15; 12/201 = 6.0% | Not met (the catch bar misses by one) |
| Kill: the trigger catches fewer than 10 of 15 | 11 | Does not fire |
| **Overall** | | **Neither success nor kill** |
| Prediction in the note (data-informed): about 11 of 15 | 11 | Held |

---

Every criterion verdict agrees. The verifiers reproduced every reported number with their own code: all Stage A rows; 2,736 records × 4 arms in Stage B; all 3,716 I11 records, with 0 field differences. What differs is wording and description.

Settled by the verifier (its value is used above):
1. **I03 Stage A, the 13 misses.** `DEVIATIONS_POST_EVAL.md` P2 says they are 7 staff positions, "each repeated across the real and staffed arms". In fact 6 positions repeat, and `eval_09`'s staff member is missed only in staffed (6 × 2 + 1 = 13).
2. **I03 Stage A, the timing wording in `DEVIATIONS.md`.** It says it was written after the fresh seeds and the self-check. Its mtime (19:01:44) is before seeds 105–109 finished and before the full self-check. It was still hashed (19:11:15) before the fit and before any evaluation load, so the protocol order holds.
3. **I03 Stage B, B8.** Under B-relocate-prune, GLM's no-`come_after` class is 11, which is 2 more than B-add's 9, not 3 more. The "3 more" holds for B-prune's 12.
4. **I03 Stage B, B7.** All false B-add additions come from `eval_08_waiting_line_4`: 36 / 30 / 32 at the old wording and 36 / 32 / 32 / 36 / 36 / 36 at the new. The compute text said "almost all".
5. **I03 Stage B, B9.** DeepSeek's median relocation move is 3.545 m; it was reported as 3.54.
6. **I11, how the share is written.** R3's definition and `out_primary.txt` write the share with rates, "(fixes/46 − breaks/454) / the oracle's". The values use counts, as FROZEN M5 defines them: (fixes − breaks) / (oracle fixes − oracle breaks). Read literally, the rate formula would give V2 0.445 and the front snap +0.23.
7. **I11 D5**, "an independent re-check is still owed", is now done. The verifier's note-only re-implementation agrees on every record.
8. **Matching check scope (I03 Stage A).** The compute agent checked 96 records; the verifier checked 1,113, all of which agree.
9. **The "partly" labels.** The compute outputs give the overall-verdict rows the outcome label "partly". Under the frozen rules each verdict is "neither", and both sides agree.

Not settled (both readings given; neither changes a number used here):
- **Stage B class labels.** The compute agent leaves the solver's target person out of the "waiting" numbers, which `CLARIFICATIONS.md` point 18 does not say. This gives a different class (3 against the verifier's 4) on 24 new-wording waiting-line failures that are not line failures (Gemini 23, DeepSeek 1). No reported table uses them, and all old-wording classes agree.
- **I11 "merged with another body".** It can be read per body or for the union of other bodies. Both readings give 14 of 27 places.
- **I03 grid origin.** The note does not say where the 5 cm grid starts. Both implementations anchor it at world (0, 0), and a half-cell shift changes no count.

Incidental, with no effect on any result:
- Importing the unchanged `lifting.py` and `frontsnap.py` wrote bytecode caches into `rerun/lifting/__pycache__`, outside `phase1b/`. The original `results/still-scenes/overview/rerun-files/lifting` outputs are untouched.
- `lifting.py`'s `main()` reads the E runs s16b and s20 through the harness, for the reference reproduction only. No I11 measure uses them.
- I11's first fresh-seed M4 pass read 195 scenes while seed 108 was still being written. Its output was deleted unread, and M4 was re-run on all 260 scenes (D8). It stayed inside seeds 100–109.

---

## 7. What it means for the plan (inference)

**I03, the detector.**
- On these clean scenes the frozen blob detector finds everyone except counter staff: every line member, group member and other person, with no merges and about 7 cm xy error. Its two failures are staff who stand against tall structure (13 of 16 missed) and one layout's furniture (6 false bodies).
- Like for like, it beats GLM's and DeepSeek's people lists on recall and false people, and trails Qwen, Haiku, Sonnet and Gemini. On placeholder scenes it is perfect.
  - So a blob list is a credible recall meter, and a cross-check for weak perceivers.
  - It is not a replacement for the top tier's lists, mainly because of the staff, and because it gives no front or back.
- "Neither" rather than "kill" rests on thin, clustered evidence. The realistic recall interval spans the 0.85 kill line, and the fit-domain recall sat below it.
- Post hoc observation, not tested: the pilot re-run got 0.82–0.84 axis facing on the realistic sets with all blob points. Fit data chose the torso band instead (24/33 against 18/33), which gives 0.686 here. This is not a reason to switch without a new pre-registration.

**I03, the replays.**
- **Add-only is closed as a C fix on these sets.** B-add flips exactly what the oracle add (R3) flips, because the solver uses an unlisted body only when it joins the line.
  - L3, the I03 online check, is gated on "Stage B passes". The primary arm does not license it.
- **The gain lies in relocating or pruning** the come_after-listed detections that stand 3.5–4.2 m from where they should be.
  - B-relocate passes every bar and drops no one. But its 40 px radius was registered after I01 on these same runs, and DeepSeek lands exactly on its bar.
  - B-prune and B-relocate-prune pass with margin. But they drop 34–44 detections of real staff per model and break 2 records at the new wording.
- **Two design points for any later version**, both post hoc and not pre-registered:
  - number added people above every number a constraint names;
  - do not prune a detection whose spot the detector rejected as tall structure (this would protect the staff).
- **The constraints remain.** Of the line failures the best arm leaves, GLM's mostly state no `come_after` (11 of 14) and DeepSeek's leave a listed member out of the line (9 of 9). This backs I01's pointer to the cheaper routes to "who do I come after" (I19, I24, I32, I05).

**I33 and I09.**
- **I33's Stage A kill.** If I33's Stage A reused this detector unchanged, its recall kill (below 0.90) would fire on the realistic images by 0.01 (105/118), all from staff.
  - Blob facing is an axis (0.686 realistic, 0.983 placeholder), so I33's verification call would still have to supply front and back.
- **I09 A1.** The scorer (`results/still-scenes/perception/depth-blob-detector/people_scorer.py`) and both comparison sets (`results/still-scenes/perception/depth-blob-detector/stageA/rows.jsonl`, `results/still-scenes/perception/depth-blob-detector/stageA/vlm_rows.jsonl`) are ready.
  - Blobs already find every line member, so a detector beats them mainly on staff (3 of 16 for blobs) and on facing sign.
  - A1 could freeze on the same fit data (worked examples and seeds 100–109), so that all three people sources share one split.

**I46.** Seeds 100–109 are now fit data: I03 fitted its thresholds on them, and I11 measured its components on them. They must stay out of the sealed set. Seeds 3–99 remain unopened and reserved.

**I11.**
- **Zero-call depth components get only background leaks** (V1 9 of 46, 0.45 of the gain).
  - Occluder captures need the object behind, which V3 finds sometimes (data-informed).
  - A parcel always merges with its table, so components cannot isolate a parcel.
- **Choice errors are the larger share**: 56.5% of the 46, and all 34 of Gemini's new-wording wrong targets. For the top tier, the target lever is choice, not lift.
  - The first half of Stage 1's gate (marks, L4) now holds. The second half waits on family F (I06).
- **The I33 trigger is weak as a gate.** It catches 11 of 15 on data its author had seen and 20 of 31 blind, and it fires on up to 19% of Gemini's right person targets.
  - A verification call gated on it would run on about 1 in 9 records with a target point (85 of 752), about half of them wrong targets.
  - T2 catches all 15 but fires on half of the right person targets.

---

## 8. Proposed REPORT edits (not applied)

Each item gives the exact old text from `REPORT.md` and the proposed new text. Each old text occurs exactly once (checked with Python's `str.count`). The tag [phase 1] keeps its meaning; item 1 widens its definition to this page. Items marked optional are not required by the batch.

### 1. "How to read": the [phase 1] tag covers this batch

Old:
```text
- **[phase 1]** marks a phase-1 result (17 Oct): computed through the step-0 harness from whitelisted development runs and recomputed by an independent verifier with its own code (`results/still-scenes/overview/phase-notes/phase1.md`, with its section).
```
New:
```text
- **[phase 1]** marks a phase-1 result (17 Oct): computed through the step-0 harness from whitelisted development runs and recomputed by an independent verifier with its own code (`results/still-scenes/overview/phase-notes/phase1.md`, or `results/still-scenes/overview/phase-notes/phase1b.md` for the second batch, with its section). The second batch (I03 Stages A–B, I11 Stage 0, I33's Stage B trigger) also froze and hashed a protocol note before any evaluation scoring.
```

### 2. Opening status line (optional)

Old:
```text
(`results/still-scenes/overview/phase-notes/phase1.md`: I01 Steps 0–2, A0, I14 version 0, the I46 statistics), nothing has been built, and no model has been run. The re-run already answers a few zero-call probes (finding 9), and phase 1 answers I01 Steps 0–2, I09's step A0, I14 version 0 and the statistics half of I46.
```
New:
```text
(`results/still-scenes/overview/phase-notes/phase1.md`: I01 Steps 0–2, A0, I14 version 0, the I46 statistics; `results/still-scenes/overview/phase-notes/phase1b.md`: I03 Stages A–B, I11 Stage 0, I33's Stage B trigger), nothing has been built, and no model has been run. The re-run already answers a few zero-call probes (finding 9), and phase 1 answers I01 Steps 0–2, I09's step A0, I14 version 0, the statistics half of I46, I03 Stages A–B, I11 Stage 0 and I33's Stage B trigger.
```

### 3. Section 1.2, plan step 4 (done)

Old:
```text
4. I03: depth-blob audit. I11 Stage 0: wrong-target decomposition with snap-and-lift.
```
New:

### 4. Section 6, step-3 row (done)

Old:
```text
| 3 | Days 8–11 | I03 Stages A–B; I11 Stage 0 (shared mask and blob code; gives I33 Stage B) | First result: about 2 d each; about 3.5 d together | 0 | — |
```
New:

Old:
```text
- Stage A: detector quality. Thresholds are re-derived from scratch against ground-truth positions only, on whitelisted dev scenes plus fresh seeds 3–12. No threshold is taken from the pilot. Ignore blobs within 0.5 m of the target point.
```
New:

Old:
```text
- the fresh-seed generation for I03 Stage A (seeds 3–12, seconds per scene), started on day 7 while I01 finishes;
```
New:
```text
- the fresh-seed generation for I03 Stage A (seeds 100–109, moved from 3–12 on 17 Oct because seeds 3–99 are reserved for the I46 sealed set; done 17 Oct: 260 scenes at a mean of 4.1 s each, no held-out situation written);
```

### 7. I03, first probe: the Stage A and B results

Old:
```text
  - More than 60% of failures unfixable offline, and the optional 10–20-call check gains fewer than 2 of 10.
```
New:

### 8. I11, first probe: the Stage 0 result

Old:
```text
- [re-run] Oracle snap-and-lift fixes 20 of 46 GLM + DeepSeek wrong targets (16 full successes), meeting three of the four criteria; depth components were not tested (old wording).
```
New:
```text
- [re-run] Oracle snap-and-lift fixes 20 of 46 GLM + DeepSeek wrong targets (16 full successes), meeting three of the four criteria; depth components were not tested (old wording).
- **Stage 0 result (17 Oct; protocol frozen and hashed before scoring, `results/still-scenes/perception/lifting/FROZEN.md`; `results/still-scenes/overview/phase-notes/phase1b.md` section 3) [phase 1].**
  - `results/still-scenes/overview/rerun-files/lifting` reproduces line for line. Lift classes are 20 of 46 (43.5%; exactly 40.0% if the 4 no-point records count as non-lift): 9 occluder captures, 11 background leaks, 0 floor slips. Choice errors are 26 of 46 (56.5%).
  - Oracle snap-and-lift, applied to every record with a point: fixes 20 of 46 (every occluder capture and background leak, no choice error), 16 of them full successes, and breaks 0 of 454 right-target records (0 of 699 over all three models; the front-surface snap broke 37). S1–S3 met.
  - Depth components, no ML (V1: 4-neighbour components joined across 3D gaps of at most 0.10 m, snapping only when the lift lands on floor or structure): 9 of 46 fixes, all background leaks, and 0 breaks, so 0.45 of the oracle's gain against the 0.70 bar. S4 not met. No variant reaches 0.70 (always snap 0.40; a step back to the object behind 0.50, data-informed; a depth-relative join 0.45). No kill clause fires. Verdict: neither.
  - Components recover people (12 of 12 person targets at IoU ≥ 0.7), but the parcel always merges with its table (3 of 3; 10 of 10 on fresh seeds), and 14 of 27 place targets merge with people standing against them.
  - V1 broke no right-target record in any set (3,716 records, new wording and placeholder runs included). At the new wording it fixes 12 of GLM's 33 wrong targets. Gemini's 34 are all choice errors on place targets, and even the oracle fixes none of them.
  - Choice errors at 56.5% meet the first half of Stage 1's gate; family F is still needed. The evidence is clustered: the oracle's 20 fixes come from 6 layouts, V1's 9 from 4.
```

### 9. I33, first probe: the Stage B trigger result

Old:
```text
    - Stage B: the trigger catches at least 12 of 15 person wrong-targets, with at most 10% false triggers.
```
New:
```text
    - Stage B: the trigger catches at least 12 of 15 person wrong-targets, with at most 10% false triggers.
      - [phase 1] Measured on 17 Oct (`results/still-scenes/overview/phase-notes/phase1b.md` section 4; trigger frozen in `results/still-scenes/perception/lifting/FROZEN.md`). The trigger fires when the lift lands on floor or structure, or the 5×5 lift window spans two depth components. On the old wording it catches 11 of 15 person wrong targets (all 6 background leaks; 5 of 9 occluder captures, exactly those whose window holds target pixels) and fires on 12 of 201 right-target person records (6.0%). Verdict: neither (the catch bar misses by one; the kill line of 10 is not crossed). The note's author had read the per-record diagnostics and predicted 11, so the blind check is the new wording: 20 of 31 pooled (post hoc pooling), with false triggers of 0–19.4% by model (Gemini 19.4%, Sonnet 12.3%). The 15 come from 3 layouts, 13 at `eval_12`. The trigger flags 11.3% of records with a point, 47% of them wrong targets.
```

### 10. I33, kill: the recall line against I03's detector (optional)

Old:
```text
    - Recall below 0.90.
```
New:
```text
    - Recall below 0.90.
      - [phase 1] (inference) I03 Stage A's frozen detector finds 0.890 of visible people on the realistic images (105 of 118; every miss is counter staff) and 1.000 on placeholder scenes. If Stage A here reused it unchanged, this kill line would fire on the realistic set by 0.01 (`results/still-scenes/overview/phase-notes/phase1b.md` section 1).
```

### 11. I09, first probe: I03's people scorer is ready for A1

Old:
```text
  - A1 (after the OK, about 6 h): freeze on the example layouts, report once on 42 eval images.
```
New:
```text
  - A1 (after the OK, about 6 h): freeze on the example layouts, report once on 42 eval images.
    - [phase 1] I03's people scorer is ready for A1 (17 Oct): `results/still-scenes/perception/depth-blob-detector/people_scorer.py`, `score_people(people, scene, th=None, *, exempt_xy=(), facing='full')`. It calls the package's own matching (`sgi.run._matched_people`) and visibility test, and returns recall, false people, merged neighbour pairs, xy error, full-yaw and axis facing, and misses by role. The rows to compare against, on the same scenes: the depth blobs (`results/still-scenes/perception/depth-blob-detector/stageA/rows.jsonl`; realistic recall 0.890, 0.143 false per image, every miss counter staff, axis facing 0.686) and the six models' recorded people lists (`results/still-scenes/perception/depth-blob-detector/stageA/vlm_rows.jsonl`; realistic recall 0.842–0.994, 0.028–0.342 false per answer; `results/still-scenes/overview/phase-notes/phase1b.md` section 1). (inference) A1 could freeze on I03's fit data (the worked examples and fresh seeds 100–109), so that all three people sources share one split.
```

Old:
New:

### 13. Section 6, L3's gate (optional)

Old:
New:

### 14. I46, What: which seeds the sealed set uses (optional)

Old:
```text
  - Fresh seeds (3 and up) of the existing situations, sized by the power analysis (17 Oct: at least 20 fresh layouts; see the statistics below).
```
New:

### 15. Finding 9: relocating with numbers kept is now tested (optional)

Old:
```text
and moving a detection with its `come_after` number kept was not tested (`results/still-scenes/overview/phase-notes/phase1.md` section 1).
```
New:
```text
and moving a detection with its `come_after` number kept was not tested (`results/still-scenes/overview/phase-notes/phase1.md` section 1). [phase 1] I03 Stage B then tested it with real depth blobs: moving each unsupported detection to an unexplained blob within 40 px, with its number and constraints kept, fixes 15 of GLM's 29 and 3 of DeepSeek's 15 line failures; adding the blobs as new people fixes 2 and 2 (`results/still-scenes/overview/phase-notes/phase1b.md` section 2).
```

### 16. Finding 9: Gemini's wrong targets are now classified (optional)

Old:
```text
(inference) Gemini's gap is wrong targets at lines; whether I11's snap-and-lift can recover them is untested, since the re-run did not classify them.
```
New:
```text
(inference) Gemini's gap is wrong targets at lines. [phase 1] I11 Stage 0 classified them at the new wording: all 34 of Gemini's wrong targets are choice errors on place targets, and even the oracle snap-and-lift fixes none of them. So they are a choice problem (I11 Stage 1's marks), not a lift problem (`results/still-scenes/overview/phase-notes/phase1b.md` section 3).
```

### 17. Finding 9: the blob pilot and the zero-call snap (optional)

Old:
```text
but the one zero-call rule tried fixes 8 and breaks 37 of 752 C records.
```
New:
```text
but the one zero-call rule tried fixes 8 and breaks 37 of 752 C records. [phase 1] With thresholds re-derived on fit data, the frozen blob detector keeps the pattern on the development scenes (every non-staff person found; 13 of 16 realistic staff missed). A designed zero-call depth-component snap fixes 9 of the 46 GLM + DeepSeek wrong targets and breaks none of 3,716 records, which is 0.45 of the oracle's gain (`results/still-scenes/overview/phase-notes/phase1b.md` sections 1 and 3).
```

### 18. Revision note

Old:
```text
Short [phase 1] notes also go into Programme A's thesis, I03, I04, I11 and the I36 parked row.
```
New:
