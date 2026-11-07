# Phase 1, step 3b: I09 A1–A2 results (2026-10-17)

Plan step 5 (REPORT section 1.2; section 6, step 3b): I09 A1, a local person detector plus a pose model as the people source, scored against ground truth on the development scenes; and I09 A2, solver replays in which the VLM's recorded people are replaced by the detector's. The detector, the depth blobs (I03, `phase1b/`) and the six models' people lists are compared on the same scenes with the same scorer.

How it was made:
- One protocol note covers both steps: `i09/FROZEN.md` (sha256 `f315e96c…df15`). The I09 lead wrote and hashed it at 21:27, after a one-image API probe on a fit image (21:21) and before any detector or pose model processed an evaluation image. Changes and clarifications after the freeze are in hashed files: `i09/DEVIATIONS.md` (21:33), `i09/a1/ADDENDUM_FIX.md` (the one general fix, 21:42), `i09/DEVIATIONS_2.md` (21:49) and `i09/a2/CLARIFICATIONS.md` (21:57, before any replay).
- **Two environments.** GPU inference ran in the separate torch environment, used read-only as a sidecar. It never imported the package, read image paths only from manifests that the project venv wrote through the step-0 harness (and the fresh-seed loader), checked each image's sha256, and refused held-out names. Its `pip freeze` was unchanged afterwards (both verifiers checked). Lifting, filtering, facing, scoring and replays ran in the torch-free project venv through the harness.
- **Inputs.** Fit data is I03's: the worked examples plus fresh seeds 100–109 (read, not regenerated). Evaluation is the 68 whitelisted development scenes and the whitelisted recorded runs, each scored once.
- **Verifiers.** An independent verifier recomputed each step with its own code. A1's verifier ran its own sidecar (raw detector and pose outputs bit-identical on 97 images), its own pipeline, scorer and timing. A2's verifier wrote its own replay. Where a compute agent and its verifier differ, this page uses the verifier's final value; where the verifier could not settle a point, both are given (section 5).

Paths below are relative to this folder (`phase1c/` in the research folder).
- `i09/`: `FROZEN.md` and its hash, `DEVIATIONS.md`, `DEVIATIONS_2.md` (each hashed), and `probe/` (the pre-freeze probe).
- `i09/a1/`: the sidecar (`sidecar/infer.py`), `pipeline.py`, `common.py`, `selfcheck.py`/`.txt`, the manifests (`manifest_fit.json`, `manifest_eval.json`), the raw outputs (`raw_fit/`, `raw_eval/`, `timing/`), the fit (`fit_grid.txt`, `fit_rows.jsonl`, `chosen.json`, `FIT.sha256`), the fix (`ADDENDUM_FIX.md`, `fit_fix.txt`, `chosen_fix.json`, `FIT_FIX.sha256`), the evaluation (`tables.txt`, `rows.jsonl`, `people_rows.jsonl`, `summary.json`), the frozen people for A2 (`eval_people.json`), `env.txt`, `ORDER.txt` and `EVAL.sha256`.
- `i09/a2/`: `CLARIFICATIONS.md`, `replay_d.py`, `selfcheck.txt`, `tables.txt`, per-record rows (`base_*.jsonl`, `rows_*.jsonl`), `summary.json`, the logs, `ORDER.txt` and `A2.sha256`.
- Verifiers: `i09/verify_i09_a1/`, `i09/verify_i09_a2/`.

## 0. Verdicts at a glance

| Analysis | Pre-registered outcome | One-line reading (inference) |
|---|---|---|
| I09 A1 (detector plus pose, realistic images) | **Success.** Every bar met, no kill clause fired. Only after the one general fix, to facing; the choice to fix facing rather than detection is data-informed. Without the fix the facing kill would have fired (post hoc) | On realistic images the detector finds everyone except 2 counter staff, places people within about 5 cm, and gets facing right where the VLMs fail (profile and back views). |
| I09 A1, placeholder scenes (reported, not judged) | Recall 0.845, below the 0.90 kill line; every other bar met | The COCO detector does not transfer to placeholder avatars seen side-on or from behind. The blobs find all of them. |
| I09 A2, arm D (primary) | **Success.** Qwen 236, GLM 202, DeepSeek 199 of 252 (bars 232, 199, 190); no kill | With the detector's people, GLM reaches ground-truth people record for record, by I01's drop mechanism. |
| I09 A2, secondary arms and extras | D-pix success; D-keep neither; new wording success (relative bars); placeholder runs neither (Qwen −8) | Dropping unsupported detections carries the gain, and it costs real people wherever the detector misses them. |
| **I09 overall** | **Success** (A1 and A2 success; no A1 or A2 kill clause fired) | A firm offline result on realistic images, resting on one data-informed fix and 14 layouts. |

Shared cautions:
- **Evaluation domains.** E-real (the verdict domain): 42 images = 14 layouts × 3 arms (real, staffed, plain), 118 visible people. The real and staffed arms share staff positions. E-ph (secondary): 26 placeholder scenes (`eval_00`–`13`, plus `eval_14`, `15`, `20`, `21` from dev seeds 0–2), 58 visible people.
- **Records.** A2's primary records are the old wording, all C: Qwen and GLM from s19, DeepSeek from s20, 252 records each (nobody near 108, lines 90, groups 54). The new wording (s21 Qwen; s23 GLM, DeepSeek, Haiku, Sonnet, Gemini) and the placeholder runs are extras. At the new wording the Qwen, GLM and DeepSeek people answers are cache copies of s19/s20 (72/72, 72/72, 77/77 identical).
- **Clustering.** Both episodes of a scene and trial share one people answer, so flips cluster: GLM's 16 D flips come from 10 answers in 4 layouts.
- **Not blind.** The note's author knew several facts about the evaluation scenes: every realistic line member stands side-on to the camera; the blobs miss 13 counter staff; Gemini's A0 facing is 0.906; and I01's come_after-listed detections lie 3.7–4.6 m from every true person. The note labels as data-informed F4's profile rule, D-pix, the A2 failure populations and possibly the grid's ranges. A1's compute agent adds the choice of fixing facing rather than detection.
- **The fit data did not support the setting cleanly.** None of the 192 settings met the false-people bar in both fit domains, so the stated fallback chose the setting.

---

## 1. I09 A1: detector plus pose as the people source (`i09/a1/`)

### What was done

- **Data split (frozen; identical to I03).**
  - Fit data: F-real = the 21 realistic worked examples (39 visible people). F-ph = the 8 placeholder worked examples plus the 260 scenes of seeds 100–109 (268 scenes, 585 visible people). 289 images in all.
  - Evaluation: the 68 scenes of `H.distinct_scenes()`, scored once. Visibility uses each run's recorded thresholds.
- **Sidecar.** RT-DETRv2-r50vd and D-FINE-large (both in the fit pass; only the chosen one in the evaluation pass), and ViTPose-base-simple.
  - It writes every person box with score ≥ 0.05 and ViTPose's 17 keypoints for every box with score ≥ 0.10.
  - The probe (`probe/`) fixed the API facts only. No threshold came from it.
- **Project side** (fixed; only the grid below is fitted). For each person box with score ≥ τ:
  1. torso region: the pixels inside the hull of the four shoulder and hip keypoints when all are confident (score ≥ 0.30) and at least 20 have valid depth; otherwise the box's central region (u 30–70%, v 20–50%);
  2. depth band: keep pixels within ±0.30 m of the region's median depth; reject the box below 10 points;
  3. position: the median x and y of the kept points, pushed d_push farther along the line from the camera (as `sgi.stages.people._centre` does);
  4. plausibility: the box top back-projected at the median depth must lie in the height band (lower bound skipped when the box touches the image top);
  5. flat test (when on): reject the box when at least 80% of its depth points lie within 0.04 m of one plane;
  6. duplicates: in descending score, drop a box within d_dup of one already kept.
- **Facing** (world yaw). A keypoint is used when it is confident and its depth lies within ±0.40 m of the torso median. Facing is the left-to-right shoulder (or hip) vector turned by −90 deg.
  - F1: shoulders only. F2: the mean of the shoulder and hip vectors.
  - F3: F2, then turned 180 deg when the nose and both eyes are confident and the yaw points away from the camera (a one-sided face rule).
  - F4: F3 plus a profile rule. When the shoulders are edge-on (projected width under 0.12 m) and the nose is confident, yaw is set ±90 deg from the camera-to-person ray, by whether the nose lies image-right or image-left of the ears (or the shoulder midpoint), with a 1 px dead zone. This rule is data-informed (the author knew the line members stand side-on).
  - F5 (added by the one general fix): F4 without F3's face rule.
- **Grid** (192 settings): detector {RT-DETRv2, D-FINE} × τ {0.25, 0.40, 0.55, 0.70} × height band {[1.00, 2.30], [1.30, 2.10] m} × flat test {off, on} × d_dup {0.30, 0.50 m} × d_push {0, 0.10, 0.20 m}.
- **Selection.**
  - Feasible means at most 0.10 false people per scene in each fit domain. Among feasible settings, take the highest lower-of-the-two-domain recall.
  - Ties go to the lower worst-domain false rate, the lower worst-domain median xy error, the faster detector (median detector-only latency over the 289 fit images), then the least restrictive setting.
  - If nothing is feasible (as I03): the lowest false rate with recall ≥ 0.95; else the highest recall with false ≤ 0.30; else the lowest false rate.
  - Facing variant: the highest lower-domain share within 45 deg (unknown counted as wrong) over the chosen setting's matched fit people.
  - `fit_grid.txt`, `fit_rows.jsonl` and `chosen.json` were hashed into `FIT.sha256` before any evaluation manifest existed.
- **One general fix**, as I03: allowed only if the choice misses a success bar measurable on fit data; one revision, of detection or of facing, designed from fit failures only and written as a hashed addendum; adopted only if the same rule picks it over the union of rows.
- **Scorer.** `phase1b/i03/people_scorer.py`, unchanged. Recall = matched / visible (`sgi.run._matched_people`, within 0.5 m). False = an unmatched detection not within 0.5 m of a non-person target anchor.
- **Facing report.** By A0 view class, from the angle between the true yaw and the direction from the true person to the camera: frontal < 45, oblique 45–75, profile 75–105 (both ends in), back > 105 deg. Non-profile = frontal + oblique + back. Front/back correct = error < 90 deg. Unknown counts as wrong (180 deg in medians).
- **A0-matched facing.** On each model's A0 pairs (`phase1/a0_facing/a0_pairs.jsonl`; arm, scene, person), the detector's yaw for the same true person. Primary measure: within 45 deg over detector-matched pairs, unknown counted as wrong; unmatched pairs are recall misses, counted and left out. The kill population is Gemini's 310 pairs (104 instances, 36 people, 8 layouts).
- **Runtime.** The deployed path: a decoded image → the detector at τ → ViTPose on the kept boxes → keypoints on the CPU, synchronised; batch 1, fp32; 5 warm-ups on F-real images, then each of the 42 E-real images timed once. The project side runs on the CPU with 2 threads. Criterion: median sidecar plus median project side ≤ 1.0 s.

### Fit and the one fix (fit data only)

- **The setting.**
  - 0 of 192 settings were feasible. In every setting either F-real's false rate exceeded 0.10 or F-ph's recall was low.
  - Fallback 4b (highest lower-domain recall with false ≤ 0.30) left 12 settings tied at 513/585 = 0.877. 6 remain at the lowest false rate (0.143), their median errors are equal, and all are RT-DETRv2, so the latency tie-break was not reached. The least restrictive rule (wide band, then flat off) chose setting #5.
  - **Chosen:** RT-DETRv2, τ 0.25, band [1.00, 2.30] m, flat test off, d_dup 0.50 m, d_push 0.20 m.
  - Fit metrics: F-real 36/39 = 0.923, 3/21 false (0.143), median xy 0.060 m, line members 15/15. F-ph 513/585 = 0.877, 0/268 false, 0.055 m, line members 208/215.
  - Detector-only latency medians: RT-DETRv2 34.8 ms, D-FINE 29.6 ms. The best D-FINE setting had F-real 38/39 but F-ph only 238/585 (verifier), so the placeholder recall decided the detector.
- **Facing, as frozen.** Within 45 deg, lower of the two fit domains: F1 0.487, F2 0.513, F3 0.244, F4 0.735 (F-real 33/36, F-ph 377/513). F4 was chosen.
- **Unit check.** F1's error is under 90 deg for F-real 5/5 and F-ph 123/123 frontal or back people (both shoulders used, not edge-on). Synthetic sign checks all pass.
- **The fix trigger.** The choice missed bars measurable on fit data:

| Fit domain | Missed bar | Value |
|---|---|---|
| F-real | recall ≥ 0.95 | 36/39 |
| F-real | false ≤ 0.10 per image | 3/21 |
| F-real | non-profile median ≤ 25 deg | 27.4 deg |
| F-real | non-profile within 45 ≥ 0.85 | 12/15 |
| F-ph | recall ≥ 0.95 | 513/585 |
| F-ph | non-profile median ≤ 25 deg | 130.8 deg |
| F-ph | non-profile within 45 ≥ 0.85 | 131/263 |

- **Fit failures.**
  - Facing: F3's face rule fires on every matched back-view person (F-real 3 of 3, F-ph 140 of 140), because ViTPose reports a confident nose and both eyes on people seen from behind. Each time it turns a correct shoulder yaw toward the camera. It fired on no frontal or oblique person; its 74 F-ph profile firings were overridden by F4's profile rule. Across 549 matched fit people it corrected no one and broke 143.
  - Detection: 71 of F-ph's 72 misses have a box over the person's chest scoring 0.05–0.25, below every τ of the grid. F-real's 3 misses are staff whose lower body is hidden, in 2 layouts, boxed at 0.84–0.86 but placed 0.61–0.63 m off.
- **The fix: F5 = F4 without the face rule.** Re-fit over F1–F5: F5 gives F-real 36/36 and F-ph 505/513 (lower domain 0.984), so the rule picked it and it was adopted. The 192 detection rows were recomputed and are unchanged. With F5, F-real still misses the recall and false bars, and F-ph the recall bar.

### Main tables

Table 1.1. The detector on the evaluation scenes (frozen setting, scored once).

| Domain (scenes) | Visible | Recall [Wilson 95%] | False people (per image) | Merged pairs | Median xy error (n) | Staff recall |
|---|---|---|---|---|---|---|
| **E-real (42)** | 118 | **116/118 = 0.983 [0.940, 0.995]** | **3 (0.071)** | 0/33 | 0.053 m (116) | 14/16 |
| real (14) | 42 | 42/42 = 1.000 | 1 | 0/11 | 0.053 m (42) | 8/8 |
| staffed (14) | 42 | 40/42 = 0.952 | 2 | 0/11 | 0.051 m (40) | 6/8 |
| plain (14) | 34 | 34/34 = 1.000 | 0 | 0/11 | 0.051 m (34) | — |
| **E-ph (26)** | 58 | **49/58 = 0.845 [0.731, 0.916]** | **0** | 0/11 | 0.059 m (49) | — |
| Pooled (68) | 176 | 165/176 = 0.938 [0.892, 0.965] | 3 (0.044) | 0/44 | 0.053 m (165) | 14/16 |

- **Layout cluster bootstrap** (E-real, 14 layouts, 2000 draws): recall [0.954, 1.000] (verifier 0.953 with its own RNG), line recall [1.000, 1.000], false per image [0.000, 0.190], which spans the 0.10 bar.
- **Recall by role**, E-real: line members 48/48, group members 27/27, others 27/27, staff 14/16. E-ph: line 16/16, group 24/33, other 9/9.
- **Misses and false people on E-real.** Both misses are staff with the lower body hidden (`staffed_eval_02_no_one_place` s0 and `staffed_eval_08_waiting_line_4` s0). Each was boxed (scores 0.909 and 0.777) but placed 0.62 and 0.646 m off, so each also counts as a false person. The third false person is in `real_eval_08_waiting_line_4` (score 0.608), 0.646 m from the nearest visible person. 0 exempt, 0 near a hidden person, 0 merges.
- **E-ph misses** (verifier): all 9 are group members of `eval_15_audience_5` across seeds 0–2, at 92–126 deg from facing the camera. Their best box over the chest scores 0.07–0.23, below τ. On the same scenes the blobs found 58 of 58.

Table 1.2. Facing by view class, all matched people, variant F5 (unknown counted as wrong; none was unknown).

| View class | E-real n | Within 45 (median deg) | Front/back < 90 | E-ph n | E-ph within 45 | E-real with F4 (post hoc): within 45 / front/back |
|---|---|---|---|---|---|---|
| frontal | 18 | 18/18 (1.1) | 18/18 | 4 | 4/4 | 18/18 / 18/18 |
| oblique | 23 | 23/23 (22.0) | 23/23 | 5 | 5/5 | 23/23 / 23/23 |
| profile | 57 | 52/57 (4.4) | **57/57** | 22 | 22/22 | 49/57 / 54/57 |
| back | 18 | 18/18 (4.9) | 18/18 | 18 | 18/18 | **0/18** (median 175 deg) / 0/18 |
| **non-profile** | 59 | **59/59 (6.4)** | 59/59 | 27 | 27/27 (11.5) | 41/59 (22.0) / 41/59 |
| all matched | 116 | 111/116 = 0.957 (6.3) | 116/116 | 49 | 49/49 | 90/116 / 95/116 |

- Coverage (matched people with a known yaw): 116/116 on E-real, 49/49 on E-ph.
- Layout bootstrap of the E-real non-profile median: 1.3 to 23.5 deg.
- The 5 profile people outside 45 deg are line members of layout `eval_09`, 46.8–73.4 deg off, with front and back still right.
- The F4 column is post hoc (not pre-registered). It shows that the face rule would have reversed every back-view person on the evaluation scenes too.

Table 1.3. A0-matched facing: the detector's yaw for the same true person on each model's A0 pairs.

| Model's pairs | Pairs (detector-matched; unmatched) | Detector within 45 [layout bootstrap] | Model's own, same pairs | Detector, all pairs (unmatched wrong) | Detector with F4 (post hoc) |
|---|---|---|---|---|---|
| **Gemini 3.8 low** (kill population) | 310 (307; 3) | **292/307 = 0.951 [0.865, 1.000]** | 278/307 = 0.906 | 292/310 = 0.942 | 238/307 = 0.775 |
| Sonnet 5.5 low | 300 (300; 0) | 285/300 = 0.950 [0.865, 1.000] | 242/300 = 0.807 | 0.950 | 0.777 |
| Qwen 3.8 | 298 (295; 3) | 280/295 = 0.949 [0.863, 1.000] | 225/295 = 0.763 | 0.940 | 0.769 |
| DeepSeek V4.1 Flash | 295 (291; 4) | 277/291 = 0.952 [0.871, 1.000] | 180/291 = 0.619 | 0.939 | 0.770 |
| Haiku 5.5 | 308 (308; 0) | 293/308 = 0.951 [0.863, 1.000] | 180/308 = 0.584 | 0.951 | 0.776 |
| GLM-5.3 | 267 (265; 2) | 254/265 = 0.958 [0.878, 1.000] | 145/265 = 0.547 | 0.951 | 0.770 |

- **Gemini's pairs by view class**, detector (Gemini's own): frontal 27/27 (25), oblique 64/64 (54), profile 156/171 (155), back 45/45 (44). On distinct instances the detector gets 98 of 103.
- Gemini's 3 unmatched pairs are one staff member (`staffed_eval_08` s0) in 3 trials.
- The verifier's bootstrap lower bound on Gemini's pairs is 0.869 (agent 0.865); the bootstrap has 8 layouts, so the intervals are rough.
- The F4 column is post hoc: before the fix, the detector would have sat below Gemini, at 0.775, under the 0.806 kill bar.

**Runtime** (the deployed path, warm, on the 4060):
- Sidecar (decoded image to keypoints on the CPU): median 54.9 ms, p90 103.0 ms. File read and decode (median 5.4 ms) are not counted.
- Project side (lift, filters, F5; 2 threads): median 0.7 ms, p90 2.5 ms. It is fast partly because the chosen setting has the flat test off.
- **Criterion: median sum 55.6 ms (≤ 1.0 s).** p90 sum 105.5 ms. Peak GPU memory 626.7 MiB allocated (931 MiB in use on `nvidia-smi`).
- The timing pass's keypoints agree with the raw pass within 1 px in 42 of 42 images (largest gap 6.1e-5 px, from a different pose batch size). The deployed path gives the same people as the evaluation path in 42 of 42.

### Checks

- **Self-check.** Given the true visible people as detections, the scorer gives recall 1, 0 false and all facing within 45 deg in 289 of 289 fit scenes.
- **Raw outputs.** The verifier's own sidecar (same snapshots and settings) reproduces `raw_eval/` and `raw_fit/` bit for bit: RT-DETRv2 on 97 images (68 evaluation, 29 worked examples), D-FINE on 29, with box, score and keypoint differences of 0.
- **Fit and evaluation.** The verifier's separate pipeline reproduces all 192 fit rows, the fallback choice, the facing selection, the fix's adoption, and every one of the 168 evaluation people (xy difference 0, yaw difference 9e-16 rad).
- **Order** (file times, +08:00): FROZEN.md 21:27:07 → `DEVIATIONS.md` 21:33:39 → fit manifest 21:33:55 → fit pass 21:35:08–45 → `chosen.json` and `FIT.sha256` 21:39:53 → `ADDENDUM_FIX.md` 21:42:43 → `FIT_FIX.sha256` 21:44:00 → `DEVIATIONS_2.md` 21:49:18 → evaluation manifest 21:49:19 → the one evaluation pass 21:49:32–40 (first image 21:49:35; RT-DETRv2 only) → timing 21:49:50 → scoring 21:50:12–25. Every hash file verifies.
- **Code after the fit freeze.** `pipeline.py` was saved at 21:42:55, after `FIT.sha256`, and code was hashed only at the end (`EVAL.sha256`). The verifier's separate implementation reproduces every fit row and evaluation person, so this changes no number.

---

## 2. I09 A2: solver replays with the detector's people (`i09/a2/`)

### What was done

- **Records.** Primary: the old wording (s19 `qwen38_low` and `glm53`, s20 `dsv41`; con; realistic arms; 252 each). Extras: the new wording (6 models) and the placeholder C runs (`ph` + `dev_seed0`–`2`, 156 per model).
- **Detector people.** `a1/eval_people.json`, unchanged and hash-checked on every load: the frozen A1 people of each scene, in descending score. No detector was run in A2.
- **Solving.** Each solver step is solved again with `H.resolve(rec, people=, constraints=)` and the recorded target; the solver re-derives the target person by position. Records that are not solver steps (including every record without a people list) keep their recorded verdict. A solver step with an empty list gets every detector person added.
- **Arms:**
  - **D** (primary): match each VLM detection to a detector person within 0.5 m (greedy nearest first, one to one, as `sgi.run._matched_people`). A matched detection takes the detector's xy and yaw and keeps its number, label and pixel. Unmatched VLM detections are dropped, and their numbers are stripped from every constraint (`repairs.strip`). Unmatched detector people are added, numbered above every listed and constraint number (the I03 number-collision lesson).
  - **D-pix** (secondary, data-informed): as D, but a detection's candidates are the detector boxes that contain its recorded pixel, taken by pixel distance to the box centre.
  - **D-keep** (secondary): as D, but unmatched VLM detections stay where recorded and nothing is stripped.
- **Baselines.** The recorded C and its step-0 replay; I01's P (true people under the recorded constraints), R3 (add the missed people) and R4 (drop unmatched detections), from the unchanged `repairs.solver_row`.
- **Criteria** (verbatim REPORT I09, judged on D, old wording): GLM ≥ 199 (I01 P 202 − 3, and ≥ C 186), DeepSeek ≥ 190 (P 193 − 3, and ≥ C 182), Qwen ≥ 232 (C 234 − 2). Kill: GLM < 186 and DeepSeek < 182.
- **Fixable offline** = a recorded failure that I01's P flips. A **line failure** is as in RERUN R01: 46 pooled (GLM 29, DeepSeek 15, Qwen 2), of which 25 are fixable (2 + 15 + 8).
- `CLARIFICATIONS.md` was written and hashed (21:57:01) before any replay. Its author first took a structural look at the records only (scene mapping, step counts, empty lists, constraint numbers).

### Main tables

Table 2.1. Old wording, successes of 252 (flips / breaks against recorded C), and line successes of 90.

| Arm | Qwen | GLM | DeepSeek | Qwen lines | GLM lines | DeepSeek lines |
|---|---|---|---|---|---|---|
| C (recorded) | 234 | 186 | 182 | 86 | 42 | 68 |
| **D (primary)** | **236 (+2/−0)** | **202 (+16/−0)** | **199 (+18/−1)** | 88 | **57** | **78** |
| D-pix (data-informed) | 236 (+2/−0) | 203 (+17/−0) | 199 (+18/−1) | 88 | 58 | 78 |
| D-keep | 234 (+0/−0) | 189 (+3/−0) | 191 (+10/−1) | 86 | 44 | 72 |
| I01 R3 (oracle add) | 234 (+0/−0) | 188 (+2/−0) | 184 (+2/−0) | 86 | 44 | 70 |
| I01 R4 (drop unmatched) | 236 (+2/−0) | 200 (+15/−1) | 188 (+6/−0) | 88 | 57 | 72 |
| I01 P (true people) | 236 (+2/−0) | 202 (+16/−0) | 193 (+15/−4) | 88 | 57 | 75 |

- **Groups of 54** (C → D): Qwen 48 → 48, GLM 51 → 52, DeepSeek 34 → 41 (+7/−0; P gives 38). **Nobody near of 108**: 100 / 93 / 80, unchanged under every arm.
- The I01 rows are re-run in the same harness and match `phase1/i01_repairs/rows_old_*` record by record.

Table 2.2. Pre-registered line failures flipped, and fixable line failures recovered (old wording).

| Arm | GLM (29) | DeepSeek (15) | Qwen (2) | Fixable line failures recovered, of 25 (Qwen / GLM / DeepSeek) | All fixable failures recovered (Qwen / GLM / DeepSeek) |
|---|---|---|---|---|---|
| **D** | 15 | 11 | 2 | **25** (2/2, 15/15, 8/8) | 2/2, 16/16, 14/15 |
| D-pix | 16 | 11 | 2 | 25 | 2/2, 16/16, 14/15 |
| D-keep | 2 | 5 | 0 | 2 (0/2, 0/15, 2/8) | 0/2, 1/16, 6/15 |
| I01 R3 / R4 / P | 2 / 15 / 15 | 2 / 4 / 8 | 0 / 2 / 2 | 2 / 21 / 25 | P: 2/2, 16/16, 15/15 |

- **Overlap with I01** (record level):
  - Qwen: D's 2 flips are P's 2 and R4's 2.
  - GLM: D's 16 flips are exactly P's 16; 15 are shared with R4 (plus 1 D only) and none with R3's 2.
  - DeepSeek: 14 of P's 15 flips are shared (4 D only, 1 P only), as are all 6 of R4's and both of R3's. D's one break (`staffed_eval_09_waiting_line_6` e0 t2: line complete, heading wrong) is one of P's 4 breaks; D avoids P's other 3 (`eval_11_conversation_group_3`).
- **Effective n.** Qwen's 2 flips come from 1 people answer in 1 layout (`eval_13`). GLM's 16 come from 10 answers in 4 layouts (net by layout: `eval_13` +7, `eval_09` +6, `eval_08` +2, `eval_10` +1; identical to P, layout by layout). DeepSeek's 18 flips come from 14 answers in 7 layouts, and its break from 1 answer.

### Mechanism

- **What D drops** (old wording, summed over solver steps; distinct people answers in brackets):
  - Qwen 28 [14]: 6 [3] are real staff, 22 [11] match no one; 22 are named in a constraint, 2 in `come_after`.
  - GLM 55 [31]: 4 [2] staff, 51 [29] no one; 48 named, 33 in `come_after`.
  - DeepSeek 58 [31]: 7 [5] staff, 51 [26] no one; 39 named, 21 in `come_after`.
  - In all, 17 of the 141 dropped detections were real people, all counter staff (the detector's 2 staff misses). D dropped no line member, group member or other person.
- **What D adds** (detector people the list lacked):
  - Qwen 38: 28 real people the list missed (group 18, line 4, staff 4, other 2), 10 false.
  - GLM 98: 91 missed (line 56, group 18, other 14, staff 3), 7 false.
  - DeepSeek 78: 69 missed (other 27, line 20, group 18, staff 4), 1 the list had found, 8 false.
  - GLM's 6 empty-list steps received every detector person and changed no verdict.
- **Housekeeping.** No added number collides with a constraint number (asserted in every arm and step). Relocation changed no detection's identity. No pair is closer than 0.25 m in any arm. In D and D-pix the final list is exactly the detector's people in all 1,549 solver steps. No detector yaw was unknown, so the VLM-yaw fallback was never used.
- **What the VLM still contributes.** Only the numbers its constraints name. The constraints were written against the VLM's own people; a pipeline that showed the model the detector's numbered people would get different constraints, and that is not tested here.
- **The line failures D leaves** (classes fixed before solving; the first that applies is used):
  - GLM 14: 12 use no `come_after`, 2 leave a visible member out of the line.
  - DeepSeek 4: all leave a visible member out of the line.
  - Qwen 0. None of the 18 is P-fixable.

### Extras

Table 2.3. New wording, successes of 252 (secondary; the Qwen, GLM and DeepSeek people answers are cache copies of R2's).

| Model | C | D | D-pix | D-keep | I01 P | Lines, C → D |
|---|---|---|---|---|---|---|
| Qwen | 228 | 230 (+2/−0) | 230 | 228 | 230 | 85 → 87 |
| GLM | 170 | 190 (+20/−0) | 192 (+22/−0) | 172 | 191 | 39 → 58 |
| DeepSeek | 185 | 198 (+16/−3) | 198 | 189 (+8/−4) | 195 | 68 → 78 |
| Haiku | 240 | 240 | 240 | 240 | 241 | 79 → 79 |
| Sonnet | 242 | 242 | 242 | 242 | 242 | 87 → 87 |
| Gemini | 214 | 214 | 214 | 214 | 214 | 56 → 56 |

- DeepSeek's 3 D breaks all sit in `eval_11_conversation_group_3` and come from positions and yaws only.
- For Haiku, Sonnet and Gemini, D drops 0–8 detections and adds 6–20 people, and changes no verdict.

Table 2.4. Placeholder C runs (`ph` + dev seeds 0–2), successes of 156 (secondary).

| Model | C | D (= D-pix) | D-keep | I01 P |
|---|---|---|---|---|
| Qwen | 147 | **139 (+3/−11)** | 147 | 148 |
| GLM | 113 | 117 (+11/−7) | 114 | 125 |
| DeepSeek | 116 | 123 (+11/−4) | 121 | 129 |

- Qwen's 11 breaks all sit in `eval_15_audience_5`. There D dropped 54 Qwen detections, all real audience members the detector missed and all named in a constraint. GLM dropped 102 (46 real group members) and DeepSeek 79 (48).
- This is the arm's failure mode: it drops every VLM detection the detector did not find, so the detector's misses become the method's misses.

### Post hoc findings (Tables 7b–7c, written after Tables 1–7 were read; they change no arm or verdict)

- **Where D's flips come from** (old wording; first class that applies):
  - Qwen 2 of 2 and GLM 15 of 16 come from dropping a detection a constraint named (I01's R4 mechanism). GLM's other flip changed only positions and yaws (largest move 0.07 m).
  - DeepSeek: 9 from a named drop, 2 from other drops or adds, and 7 from positions and yaws only (largest moves 0.07–0.28 m). Its one break is also positions and yaws only (0.16 m).
- **D against P.** They differ in 0 GLM records and 8 DeepSeek records. In all 8, neither arm drops or adds anyone; only positions and yaws differ, and D wins 7. So DeepSeek's surplus over P (199 against 193) is solver sensitivity at about the 0.1 m and facing scale, not better perception, and some of it may be luck. The row fields do not separate the yaw effect from the position effect.
- **Sensitivity.** Without its 7 position-and-yaw flips, DeepSeek's D is 192 (bar 190); without its 1, GLM's is 201 (bar 199). Without its largest layout (`eval_13`, +7), GLM's net gain over C falls to +9.

### Checks

- **Step 0.** 1,301 of 1,301 realistic places reproduce within 1e-6 m with the recorded verdict (old 441, new 860), and 248 of 248 placeholder places.
- **Baselines.** I01's P, R3 and R4 match `phase1/i01_repairs/rows_old_*` and `rows_new_*` with 0 of 2,268 records disagreeing, and `phase1b/i03/stageB/rows_ph_*` with 0 of 468. The pre-registered line-failure sets (2 / 29 / 15) and the 25 fixable line failures are asserted equal. Synthetic matching and numbering checks: 16 of 16 pass.
- **Verifier.** Its own replay agrees record by record with `rows_old_*` on D, D-pix, D-keep, P, R3, R4, the recorded C, the line-failure flag and the solver-step flag (0 disagreements), and reproduces every table above.
- **Order.** `CLARIFICATIONS.md` 21:57:01 → `replay_d.py` last saved 21:59:08 → base pass 21:59:27 → old-wording checks 22:07:39 → old arms 22:12:04–22:14:36 → new base done about 22:15 → new and placeholder arms done 22:19:12 → tables 22:20:51. Every log line re-verifies the FROZEN, `eval_people.json` and CLARIFICATIONS hashes. At most 2 worker processes ran at once.

---

## 3. Three people sources on the same scenes

Table 3.1. E-real, like for like (as `phase1b/README.md` Table 1.2). The unit is one answer per (arm, scene, trial) with a people list; the detector and the blobs are re-weighted to each model's keys. Facing is full yaw within 45 deg for the models and the detector (among known, with the number unknown in brackets) and the axis (modulo 180 deg) for the blobs, which give no front or back.

| Model (keys with a list; visible) | VLM: recall / false per answer / facing | Detector, same keys: recall / false per key / facing | Blobs, same keys: recall / false per key / axis facing |
|---|---|---|---|
| Gemini 3.8 low (72; 312) | 310/312 = 0.994 / 0.028 / 281/310 = 0.906 | 309/312 = 0.990 / 0.083 / 294/309 = 0.951 | 291/312 = 0.933 / 0.250 / 192/291 |
| Haiku 5.5 (74; 314) | 308/314 = 0.981 / 0.081 / 180/292 = 0.616 (16) | 311/314 = 0.990 / 0.081 / 296/311 = 0.952 | 291/314 = 0.927 / 0.243 / 192/291 |
| Sonnet 5.5 low (72; 312) | 300/312 = 0.962 / 0.097 / 242/299 = 0.809 (1) | 309/312 = 0.990 / 0.083 / 294/309 = 0.951 | 291/312 = 0.933 / 0.250 / 192/291 |
| Qwen 3.8 (72; 312) | 298/312 = 0.955 / 0.167 / 228/298 = 0.765 | 309/312 = 0.990 / 0.083 / 294/309 = 0.951 | 291/312 = 0.933 / 0.250 / 192/291 |
| DeepSeek V4.1 Flash (86; 331) | 295/331 = 0.891 / 0.337 / 181/294 = 0.616 (1) | 326/331 = 0.985 / 0.093 / 311/326 = 0.954 | 300/331 = 0.906 / 0.209 / 201/300 |
| GLM-5.3 (76; 317) | 267/317 = 0.842 / 0.342 / 146/267 = 0.547 | 314/317 = 0.991 / 0.079 / 299/314 = 0.952 | 293/317 = 0.924 / 0.237 / 194/293 |

- **Merged neighbour pairs** (of 99): GLM 2, DeepSeek 1, the other models 0; the detector and the blobs 0.
- **Which keys.** Keys without a list are all no_one scenes, where most counter staff stand. Counting them as empty lists, VLM recall falls to 0.754 (GLM) – 0.876 (Gemini), while the detector over all 126 keys gives 348/354 = 0.983 with 0.071 false per key, and the blobs 315/354 = 0.890 with 0.143.
- **The detector's facing** is front and back, not an axis: its axis share equals its full-yaw share here (every error above 45 deg is also above 45 deg modulo 180).

Table 3.2. Whole-domain summary (per image for the detector and the blobs).

| Source | E-real recall | False per image | Median xy error | Counter staff | Facing on E-real | E-ph recall (false) | Warm time |
|---|---|---|---|---|---|---|---|
| Detector + ViTPose (I09 A1) | 116/118 = 0.983 | 0.071 | 0.053 m | 14/16 | full yaw: 111/116 within 45; A0 pairs 0.949–0.958 | 49/58 = 0.845 (0) | 55.6 ms (verifier 61.9 ms), 4060 |
| Depth blobs (I03 Stage A) | 105/118 = 0.890 | 0.143 | 0.077 m | 3/16 | axis only: 72/105 = 0.686 | 58/58 = 1.000 (0) | not measured |
| VLM people lists (6 models) | 0.842–0.994 per answer with a list | 0.028–0.342 per answer | not compared | not compared | full yaw: 0.547–0.906 (A0) | Qwen 167/168, DeepSeek 150/168, GLM 136/168 | Qwen's people call 16.0 s (brief section 3.6) |

- On E-ph keys (78 per model, 36 without a list): the detector finds 141/168 = 0.839 with 0 false and all 141 matched people within 45 deg; Qwen 0.994 (0 false), DeepSeek 0.893 (0.381 per answer), GLM 0.810 (0.524); the blobs 168/168 with 0 false (axis 165/168).
- (inference) On realistic images the detector matches Gemini's recall with about three times Gemini's false-people rate, beats every other list on both, and beats every list on facing. On placeholder images it is the weakest of the three sources on recall.

---

## 4. Pre-registered criteria and outcomes

### I09 A1 (FROZEN.md section 10: judged on E-real; E-ph reported against the same bars)

| Criterion | E-real | E-ph (reported) | Outcome |
|---|---|---|---|
| Recall ≥ 0.95 overall | 116/118 = 0.983 | 49/58 = 0.845 | Met (E-real) |
| Recall ≥ 0.95 on line members | 48/48 | 16/16 | Met |
| False people ≤ 0.10 per image | 3/42 = 0.071 (bootstrap 0.000–0.190) | 0/26 | Met |
| Median position error ≤ 0.15 m | 0.053 m (n 116) | 0.059 m | Met |
| Non-profile facing median ≤ 25 deg | 6.4 deg (n 59) | 11.5 deg | Met |
| Non-profile ≥ 85% within 45 deg | 59/59 | 27/27 | Met |
| Profile front/back correct ≥ 90% | 57/57 | 22/22 | Met |
| Warm runtime ≤ 1.0 s on the 4060 | 55.6 ms (verifier 61.9 ms) | (E-real only) | Met |
| Kill: recall < 0.90 | 0.983 | **0.845** | Does not fire on E-real; crossed on E-ph, which is not judged |
| Kill: > 0.30 false people per image | 0.071 | 0.000 | Does not fire |
| Kill: facing within 45 below 0.806 on Gemini's A0 pairs (or below 0.70, the school-panel reading) | 292/307 = 0.951 | (no A0 pairs) | Does not fire |
| The one fix applied only if a fit bar is missed, and adopted only if it wins | | | Met (applied and adopted; its choice of target is data-informed) |
| Settings chosen on fit data and frozen before evaluation | | | Met (hashes and timeline verified) |
| **A1 overall** | | | **Success** |
| Post hoc: the pre-fix choice F4 | Gemini's pairs 0.775; non-profile 41/59 | | Would have fired the facing kill (not pre-registered; changes no verdict) |

### I09 A2 (FROZEN.md section 11: judged on D, old wording)

| Criterion | Value | Outcome |
|---|---|---|
| Step 0 reproduces the recorded C; I01 P, R3, R4 reproduce record by record | 1,301 / 1,301 and 248 / 248; 0 of 2,268 and 0 of 468 disagree | Met |
| GLM within 3 of ground-truth people (≥ 199) and ≥ recorded C (186) | 202 | Met |
| DeepSeek within 3 of ground-truth people (≥ 190) and ≥ recorded C (182) | 199 (192 without its 7 position-and-yaw flips, post hoc) | Met |
| Qwen loses at most 2 (≥ 232) | 236 | Met |
| Kill: below recorded C for both GLM and DeepSeek | 202 and 199 | Does not fire |
| **A2 overall (D)** | | **Success** |
| Prediction in the note (inference): GLM 200–202, DeepSeek 188–193, Qwen 234–236 | 202, 199, 236 | Held for GLM and Qwen; DeepSeek landed above the range |
| Secondary, D-pix, same bars | 236 / 203 / 199 | Met (secondary; data-informed arm) |
| Secondary, D-keep, same bars | 234 / 189 / 191 | Neither (GLM below 199) |
| Secondary, new wording, D, bars relative to that wording (Qwen C − 2; others max(P − 3, C)) | 230 (≥ 226) / 190 (≥ 188) / 198 (≥ 192) | Met (secondary; the bars' reading is an interpretation, section 5) |
| Secondary, placeholder runs, D, same relative bars | 139 (< 145) / 117 (< 122) / 123 (< 126) | Neither (no kill: GLM and DeepSeek above C) |

### I09 overall (FROZEN.md section 10)

| Criterion | Value | Outcome |
|---|---|---|
| Success needs A1 and A2 success | A1 success (verified); A2 success (verified) | Met |
| Kill on any A1 or A2 kill clause | None fired (E-ph's recall line is crossed but not judged) | Does not fire |
| **Overall** | | **Success** |

---

Every criterion verdict agrees. The verifiers reproduced every reported number with their own code: the raw detector and pose outputs bit for bit, all 192 fit rows, every evaluation person, and A2's records under every arm. What differs is wording, description and two measurements.

Settled by the verifier (its value is used above):
2. **A2, the roles of DeepSeek's added people.** The compute agent lists "staff 5" among the 69 missed people, which makes the roles sum to 70. The fifth staff member is the one detector person the list had already found: among the 69 missed, staff is 4.
3. **A1, the placeholder misses.** The compute agent describes them as placeholder group members in profile or back view, boxed below τ. The verifier pins them down: all 9 are group members of `eval_15_audience_5` across seeds 0–2, at 92–126 deg from facing the camera, with best box scores over the chest of 0.07–0.23.
4. **A1, the two staff misses.** The verifier gives their box scores (0.909 and 0.777) and offsets (0.62 and 0.646 m), where the compute text says 0.62–0.65 m.
5. **A1, `pipeline.py` saved after the fit freeze.** No code was hashed at the fit freeze. The verifier's separate implementation reproduces all 192 fit rows and every evaluation person exactly, so the edit has no effect.
6. **A1, the addendum's time.** Its header says "about 21:50"; it was hashed at 21:42:43, before the re-fit and the evaluation manifest. Cosmetic, and disclosed in `DEVIATIONS_2.md`.
7. **I09 overall.** A2's compute agent judged the overall verdict from A1's compute output alone. A1's verifier independently confirms A1's success, and A2's verifier checked A1's verifier files, so the overall success stands on verified results.

Not settled (both readings given; neither changes a verdict):
- **Bootstrap lower bounds** depend on the random-number implementation: E-real recall 0.954 (agent) against 0.953 (verifier); A0 on Gemini's pairs 0.865 against 0.869.
- **A2, distinct-answer counts for the dropped detections** (the bracketed figures in section 2). The two episodes of a scene and trial share one people answer but have separate constraints. The compute agent counts by the first record: GLM's `come_after` [18] and DeepSeek's [10]; named in a constraint [11] / [26] / [22]. Counting a detection if any episode names it gives `come_after` [20] and [13], and named [12] / [29] / [22]. The totals over solver steps agree exactly.
- **A2, the secondary bars.** "Same bars" for the new wording and the placeholder runs was applied relative to each run set's own C and P (Qwen C − 2; GLM and DeepSeek max(P − 3, C)). Under the primary numbers instead, new-wording Qwen 230 < 232 and GLM 190 < 199. These results are secondary and not judged.
- **A2, descriptive line classes.** 24 records differ between the two implementations, all new-wording wrong-target records that are not line failures (Gemini 23, DeepSeek 1); the verifier did not leave the target number out of the waiting set. No reported table uses them, and every line-failure class agrees.
- **A2's order clarification** (point 6, baselines checked before any arm) held per wording, not globally: the new-wording base pass finished at about 22:15, after the old-wording arms had started at 22:12:04. Baselines do not feed the arms, and the new-wording arms started only after the new base had been checked.

Incidental, with no effect on any result:
- The only nondeterminism seen is up to 6.1e-5 px in keypoints between the timing pass and the raw pass, from a different pose batch size.
- A1's compute agent read `phase1/a0_facing/a0_pairs.jsonl` (the view classes of the evaluation pairs) before choosing what to fix. Both sides report this, and it is why that choice is labelled data-informed.

---

## 6. What it means for the plan (inference)

**I09, the detector.**
- On these clean realistic images, a COCO detector plus ViTPose, lifted with exact depth, is the best people source measured so far. It finds 116 of 118 people, places them within about 5 cm, and gives facing within 45 deg for 0.95 of people on every model's A0 pairs, where the models score 0.55–0.91. Like for like, it matches Gemini's recall with about three times Gemini's false-people rate, and beats every other list on recall and false people.
- **Placeholder avatars.** The detector does not transfer to them (0.845; side-on and back views score 0.07–0.23), while the blobs find all of them. Untested idea, prompted by these results: a union of the two sources, since each misses what the other finds (the blobs miss counter staff, the detector misses placeholder avatars). It would need its own pre-registration.
- **Counter staff fail both depth methods, differently.** The blobs reject them (their blob merges with tall structure); the detector boxes them but places them 0.62–0.65 m off, probably because the torso median catches the counter in front. This is a post hoc observation, not a tested cause.

**I09, the replays.**
- D reaches ground-truth people for GLM (202, the same 16 flips as I01's P) and for Qwen (236), and passes P for DeepSeek through position and yaw differences only (post hoc). The mechanism is I01's R4: dropping detections that no detector supports, with their numbers. Keeping them (D-keep) gains GLM only 3, as adding alone did in I01 (R3) and I03 (B-add). So a people source is worth what it lets the pipeline drop.
- **The constraints remain.** Of the line failures D leaves, 12 of GLM's 14 use no `come_after`, and DeepSeek's 4 leave a listed member out of the line. As in `phase1b/README.md`, this points to the cheaper routes to "who do I come after" (I19, I24, I32, I05).
- **Facing.** The detector's facing lead over the VLMs is large, but on these sets facing errors cost C little (I01's R2 flips no line failure). DeepSeek's 7 position-and-yaw flips show the solver responds at about the 0.1 m and facing scale, but their sign may be luck. The facing lead would matter more on I06's crowd variants and at groups.

**I03's pruning question (section 7).**
- A2's D is a prune against the detector. With a reference of about 0.98 recall, the drop removed 17 real people over solver steps on the realistic images (all counter staff, the detector's 2 misses) and changed no verdict. I03's blob prune removed 34–44 real staff detections per model and broke 2 records at the new wording. On the placeholder runs, where the detector misses 9 of 58, D removed real audience members and broke 11 Qwen records.

**I33.**
- I33 proposes people from depth blobs and asks the VLM to verify "person?" and a 4-way facing. On realistic images the detector beats the blobs on recall (0.983 against 0.890) and supplies front-and-back facing better than any model, so a verification call would add little on facing there. A "person?" check would also not catch the detector's false people, since 2 of its 3 are real staff placed 0.6 m off.
- On placeholder scenes the blobs stay the better proposer. Swapping the detector into I33's Stage A would need a new pre-registration. I33's Stage B trigger is unaffected.

**Speed (Programme C, I21).**
- The detector and pose model take a median 55–62 ms warm on an idle 4060 (p90 about 0.10–0.12 s), against 16.0 s for Qwen's people call (brief section 3.6). As a people source, they could take the people call off the critical path entirely, where concurrency (I21) takes it off only partly (61.2 → 49.3 s per Qwen goal step).
- The timing was taken on an idle GPU; a GPU shared with other work was not measured.
- Section 3.2's open question, "no measurement of real-photo detectors on avatar renders", now has an answer for one detector: 0.983 on the realistic renders, 0.845 on the placeholder ones.

**Rulings this raises.**

**I46 and the plan.** Seeds 100–109 are now fit data for I03 and I09 A1, and seeds 3–99 remain unopened. With step 3b done, every phase-1 step in section 1.2 has a result; I01's optional pipeline oracle-people arm remains a big run.

---

## 7. Proposed REPORT edits (not applied)

Each item gives the exact old text from `REPORT.md` and the proposed new text. Each old text occurs exactly once (checked with Python's `str.count`), and applying items 1–11 in order works (checked on a copy in memory; each old text still occurs exactly once when its turn comes). The tag [phase 1] keeps its meaning; item 1 widens its definition to this page. Items marked optional are not required by the step.

### 1. "How to read": the [phase 1] tag covers step 3b

Old:
```text
- **[phase 1]** marks a phase-1 result (17 Oct): computed through the step-0 harness from whitelisted development runs and recomputed by an independent verifier with its own code (`phase1/README.md`, or `phase1b/README.md` for the second batch, with its section). The second batch (I03 Stages A–B, I11 Stage 0, I33's Stage B trigger) also froze and hashed a protocol note before any evaluation scoring. Its only input outside the whitelist is fresh seeds 100–109, generated in that batch (I03's fit data, and an extra domain for I11's component check). Parts registered after earlier results on the same runs are marked data-informed.
```
New:
```text
- **[phase 1]** marks a phase-1 result (17 Oct): computed through the step-0 harness from whitelisted development runs and recomputed by an independent verifier with its own code (`phase1/README.md`, `phase1b/README.md` for the second batch, or `phase1c/README.md` for step 3b, I09 A1–A2, with its section). The second batch (I03 Stages A–B, I11 Stage 0, I33's Stage B trigger) and step 3b also froze and hashed a protocol note before any evaluation scoring. Their only input outside the whitelist is fresh seeds 100–109, generated in the second batch (fit data for I03 and I09 A1, and an extra domain for I11's component check). Step 3b also ran a local detector and pose model on the GPU, on whitelisted scenes and those seeds only; no provider model was called. Parts registered after earlier results on the same runs are marked data-informed.
```

### 2. Opening status line

Old:
```text
(`phase1/README.md`: I01 Steps 0–2, A0, I14 version 0, the I46 statistics; `phase1b/README.md`: I03 Stages A–B, I11 Stage 0, I33's Stage B trigger), nothing has been built, and no model has been run. The re-run already answers a few zero-call probes (finding 9), and phase 1 answers I01 Steps 0–2, I09's step A0, I14 version 0, the statistics half of I46, I03 Stages A–B, I11 Stage 0 and I33's Stage B trigger.
```
New:
```text
(`phase1/README.md`: I01 Steps 0–2, A0, I14 version 0, the I46 statistics; `phase1b/README.md`: I03 Stages A–B, I11 Stage 0, I33's Stage B trigger; `phase1c/README.md`: I09 A1–A2), nothing has been built into the pipeline, and no VLM has been run; the only models run are I09's local detector and pose model, offline on the GPU. The re-run already answers a few zero-call probes (finding 9), and phase 1 answers I01 Steps 0–2, I09's steps A0 and A1–A2, I14 version 0, the statistics half of I46, I03 Stages A–B, I11 Stage 0 and I33's Stage B trigger.
```

### 3. Section 1.2, plan step 5 (done)

Old:
New:

### 4. Section 6, step-3b row (done)

Old:
New:

### 5. I09, first probe: the A1 and A2 results

Old:
```text
  - 0 calls offline. Live: about 30 (up to about 120), then a full run of about 1,400.
```
New:
```text
  - 0 calls offline. Live: about 30 (up to about 120), then a full run of about 1,400.
  - **Phase-1 result, A1–A2 (17 Oct; protocol frozen and hashed before any evaluation image, `phase1c/i09/FROZEN.md`; `phase1c/README.md` sections 1–4) [phase 1].**
    - Setup. A sidecar in the torch environment (used read-only; ViTPose only, no RTMW) runs the detector and the pose model; the torch-free project side lifts with depth. Fit data as I03: 21 realistic and 8 placeholder worked examples plus 260 fresh-seed scenes (289 images). A 192-setting grid covered the two detectors, score threshold, height band, a flat-surface test, a duplicate radius and a depth push. No setting met the false-people bar in both fit domains, so the stated fallback chose RT-DETRv2 at score 0.25 (band 1.00–2.30 m, flat test off).
    - Facing came from 4 declared variants. Fit data chose shoulders plus hips, with a face rule for the sign and a nose-against-ears rule for edge-on (profile) people. That choice failed on fit data: the face rule turned every back-view person around (143 of 143), because ViTPose reports confident face keypoints on people seen from behind. The one allowed general fix, written and hashed before evaluation, drops the face rule (fit: 36 of 36 and 505 of 513 within 45 deg). Choosing a facing fix over a detection fix is data-informed: its author had counted the view classes of the A0 evaluation pairs.
    - A1, scored once on the 42 realistic images: recall 116 of 118 = 0.983 (Wilson 95% 0.940–0.995; line members 48 of 48; counter staff 14 of 16, against the blobs' 3 of 16), 0.071 false people per image (3 in 42; layout bootstrap 0.000–0.190), 0 of 33 neighbour pairs merged, median position error 0.053 m. Both misses are counter staff with the lower body hidden, boxed but placed 0.62–0.65 m off.
    - Facing, all 116 matched people, none unknown: non-profile 59 of 59 within 45 deg (median 6.4 deg); profile front/back right for 57 of 57 (52 of 57 within 45). On the A0 pairs: 292 of 307 detector-matched Gemini pairs = 0.951, against Gemini's own 0.906 on the same pairs and the 0.806 kill bar; 0.949–0.958 on every other model's pairs, where the models score 0.547–0.807.
    - Warm runtime on the 4060: median 55.6 ms (the verifier re-measured 61.9 ms), p90 about 0.11 s, peak GPU memory 627 MiB.
    - A1 verdict: success; no kill clause fired. Post hoc: without the fix, the facing kill would have fired (0.775 on Gemini's pairs) and the non-profile bar would have failed (41 of 59). Placeholder scenes (reported, not judged): recall 49 of 58 = 0.845, below the 0.90 kill line. All 9 misses are audience members of one layout seen side-on or from behind and boxed below the threshold; the blobs found 58 of 58 there.
    - A2 (old wording, 252 records per model; step 0 reproduces all 1,301 realistic places, and I01's P, R3 and R4 reproduce record for record). The primary arm D moves each VLM detection onto the detector person within 0.5 m (keeping its number), drops the unmatched detections and strips their numbers from the constraints, and adds the unmatched detector people numbered above every constraint number. Qwen 236 (+2/−0; bar 232), GLM 202 (+16/−0; bar 199; ground-truth people 202), DeepSeek 199 (+18/−1; bar 190; ground-truth people 193). Lines 88, 57 and 78 of 90 (recorded 86, 42, 68); all 25 fixable line failures recovered. Verdict: success; no kill.
    - GLM's 16 flips are exactly I01 P's, from 10 people answers in 4 layouts. DeepSeek's surplus over P is post hoc traced to position and yaw differences only (D and P differ in 8 records, D wins 7); without those flips DeepSeek still scores 192. Secondary: matching by pixel also passes (236, 203, 199); keeping the unsupported detections instead of dropping them does not (GLM 189); the new wording passes relative bars (230, 190, 198). On the placeholder runs D loses 8 net for Qwen (139 against 147), because it drops real audience members the detector missed.
    - Overall I09: success (A1 and A2 both success; no kill clause fired). Not blind: the note's author knew I01's mechanism, D is in effect I01's P with the detector's people in place of the true people, and D matched the note's prediction for GLM and Qwen.
    - (inference) On these realistic images the detector is the best people source measured: recall level with Gemini's list (0.990 against 0.994 on the same answers) with about three times its false-people rate (0.083 against 0.028), and better facing than every model. Its weak spots are counter staff and placeholder avatars. A2's gain is I01's R4 mechanism, and D still needed the VLM's list to bind the constraint numbers, so the −16 s people-call saving is not yet shown: the next step is a live check in which the constraints call sees the detector's numbered people. Calling it a method would also need a confirmation on unseen layouts, with nothing re-tuned.
```

### 6. Section 7: the heading question

Old:
```text
A change would need pre-registering and testing on other sets; the placeholder panel splits 2 to 2.
```
New:
```text
A change would need pre-registering and testing on other sets; the placeholder panel splits 2 to 2. [phase 1] (inference) If I09's detector supplies the yaw, this question matters less: on the same A0 pairs its yaw is within 45 deg for 0.949–0.958 of every model's pairs, where the models score 0.547–0.906 (`phase1c/README.md` section 1).
```

Old:
```text
(inference) Before either arm is called a method, it would need a pre-registered test on fresh layouts (`phase1b/README.md` section 7).
```
New:

### 8. Section 7: torch and the detector in the pipeline

Old:
```text
Diffusers: recommend no.
```
New:

### 9. Revision note

Old:
New:

### 10. Section 3.2: detectors on avatar renders (optional)

Old:
```text
**Open:** no measurement of real-photo detectors on avatar renders. Ideas: I03, I09, I33, I10.
```
New:
```text
**Open:** no measurement of real-photo detectors on avatar renders. Ideas: I03, I09, I33, I10. [phase 1] I09 A1 made one: a COCO-trained RT-DETRv2 finds 116 of 118 visible people on the 42 realistic development images, but 49 of 58 on the placeholder scenes, where avatars seen side-on or from behind score below its threshold (`phase1c/README.md` section 1).
```

### 11. I33, kill: the recall line against I09's detector (optional)

Old:
```text
If Stage A here reused it unchanged, this kill line would fire on the realistic set by 0.01 (`phase1b/README.md` section 1).
```
New:
```text
If Stage A here reused it unchanged, this kill line would fire on the realistic set by 0.01 (`phase1b/README.md` section 1). [phase 1] (inference) I09 A1's frozen detector finds 0.983 on the realistic images, with front-and-back facing, but 0.845 on the placeholder scenes. As Stage A's proposer it would clear this line on the realistic set and cross it on the placeholder one; swapping it in would need its own pre-registration (`phase1c/README.md` sections 1 and 6).
```
