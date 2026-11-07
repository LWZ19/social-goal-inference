# Phase 2: I09 unseen-layout confirmation and live-check build (2026-10-18)

- **(A) Unseen-layout confirmation.** A1's frozen detector, with nothing re-tuned, scored once on realistic layouts it has never seen (seeds 200–212). This step is complete.

How it was made:
- **Live-check protocol.** `live/FROZEN.md` (sha256 `7e1d3212…8b84`), written and hashed by the builder at 01:16:27, before `driver.py` existed (created 01:23:29). Before the note, the builder had extracted commit 202c165 read-only, compared its prompts with the recorded runs, and counted the baselines from recorded data; the note discloses all three.
- **Environments.**
  - Detector and pose: the torch sidecar, used read-only. Its `pip freeze` (160 lines) was unchanged before and after both GPU passes. It read image paths only from the manifest and refused any path outside `phase2/seeds/realistic/seed200`–`219`.
  - Everything else: the torch-free project venv. Recorded runs and development scenes were read through the step-0 harness only.
- **No held-out data.** No held-out scene, run or situation was read or instantiated. Seeds 3–99 were not generated or opened. Nothing was written into `data/`, `src/`, `tests/`, `docs/`, `configs/` or `REPORT.md`.
- **Images.** Nobody viewed a test image. The live builder viewed two numbered images of development scenes from its dry run.

Paths below are relative to this folder (`phase2/i09/` in the research folder).
- `confirm/`:
  - protocol: `FROZEN.md`, `DEVIATIONS_1.md` (each hashed);
  - scripts: `pathguard.py`, `confirm_lib.py`, `c2_repro.py`, `manifest.py`, `c3_truth.py`, `sidecar/infer_confirm.py`, `score.py`, `posthoc_false.py`;
  - inputs and raw outputs: `manifest.json` (`MANIFEST.sha256`), `raw/` (`RAW.sha256`), `pip_freeze_before.txt`, `pip_freeze_after.txt`, `sidecar_pass.log`;
  - results: `tables.txt`, `rows.jsonl`, `people_rows.jsonl`, `summary.json` (`SCORE.sha256`), `score.log`;
  - checks and logs: `c2_repro.txt`, `c3_truth.txt`, `posthoc_false.txt` (`POSTHOC.sha256`), `ORDER.txt`.
- `../seeds/realistic/seed200`–`seed212`: the test scenes (773 MB).
- `live/`: `FROZEN.md` (hashed), `driver.py` (`BUILD.sha256`), `dryrun.py` and `dryrun/`, `REQUEST.md`, and `src_202c165/` (the package at 202c165).
- Verifiers: `verify_i09_confirm/`, `verify_i09_live_build/`.

## 0. At a glance

| Step | Outcome | One-line reading (inference) |
|---|---|---|
| (A) Unseen-layout confirmation, 182 layouts (seeds 200–212) | **Neither** (pre-registered). 6 of 7 success bars met; false people 44/182 = 0.242 per image against the 0.10 bar; no kill clause fired | Recall, placement and facing carry over to new layouts; false people do not, and most of them sit around counter staff (post hoc). |
| I09 as a method | Phase 1's A1–A2 success on the development scenes stands; the unseen-layout confirmation does not confirm it | Before the detector is called a method, it would need a new pre-registered fix and fresh layouts. |

Shared cautions:
- **Scope of the test set.** 13 of the 20 planned seeds, because the 3 h rendering clock ran out. It uses the same 14 rows and 3 situations, the `real` variant only, the same renderer and asset library, and exact rendered depth. It does not test placeholder avatars, the held-out situations, real photographs or the pipeline.
- **Post hoc.** The breakdown of the false people (section 1) was written after scoring. It is labelled as such and changes no verdict.
- **The live check uses the development scenes.** It runs on the 42 E-real scenes A1 was evaluated on, not on seeds 200–212.

---

## 1. (A) Unseen-layout confirmation (`confirm/`)

### How the test set was made

- **What was rendered.** For each seed k: `python -m sgi.synth.realistic.generate --variant real --seed k --out phase2/seeds/realistic/seed<k> --only <14 ids>`.
  - The ids are the 14 first-pass eval rows `real_eval_00`–`13`: no_one 6 rows, waiting_line 5, conversation_group 3.
  - The generator replays the first pass only. It never draws the second pass (audience, group photo, lift).
  - The `staffed` and `plain` variants were not rendered. A layout is one (seed, row) image.
- **The stop rule.** Seeds in ascending order, one generate command per seed. Rendering stops after seed 219, or once the summed command wall time reaches 3 h; waits for resources do not count, and a command already running may finish. Seeds 200–209 (140 layouts) were required; with fewer, the outcome would have been "incomplete".
- **What happened.**
  - 13 complete seeds, 200–212 (182 layouts). Seeds 213–219 were never started.
  - The clock stood at 10,802.9 s (3 h 00 min 03 s) when seed 212 finished, 2.9 s past the mark.
  - Time per scene from the generator log (n 182): median 48.3 s, range 31.4–96.4 s. Time per full-seed command (n 13): median 796 s.
  - One dressing retry (seed 205, `real_eval_10_conversation_group_2`, attempt 1 kept). No generator stop, no technical failure, no re-run, no wait for resources.
- **The manifest** (`confirm/manifest.json`, sha256 `cf2937ab…`, written at 04:41:03 from the files, not from a status file):
  - seeds 200–212 are complete, each with exactly the 14 ids, and there is no `examples/` folder;
  - every render job says `real`, the folder's seed, Blender 4.5.14, and thresholds equal to `configs/default.yaml` key by key;
  - 2,574 names under `seeds/realistic` were checked; none contains group_photo, lift_wait, heldout, held_out or eval_16–19;
  - situations per scene: no_one 78, waiting_line 65, conversation_group 39;

### The method and the scorer (frozen; nothing re-tuned)

- **Setting.** A1's frozen choice with its one fix: RT-DETRv2-r50vd at τ 0.25, height band 1.00–2.30 m, flat test off, duplicate radius 0.50 m, depth push 0.20 m. Facing is F5: shoulders and hips, plus the nose-against-ears profile rule, without the face rule.
- **Code by import, nothing copied.** A1's `pipeline.py` and `common.py`, the model functions of A1's `sidecar/infer.py`, I03's `people_scorer.py`, `chosen.json`, `chosen_fix.json`, the fix addendum, A1's `FROZEN.md` and `configs/default.yaml`. Each is pinned by sha256 and asserted by every script.
- **Sidecar driver** (`confirm/sidecar/infer_confirm.py`).
  - It calls A1's `infer.run_pass` unchanged. It replaces only the path guard (A1's refuses these seed folders) and the GPU wait (A1's waits for an idle GPU).
  - Boxes with score ≥ 0.05 are written; ViTPose runs on boxes ≥ 0.10, one batch per image. fp32 and deterministic settings; no warm-up, as in A1's evaluation pass.
  - One pass, 04:45:34–04:46:32, 182 images.
- **Scorer.** `phase1b/i03/people_scorer.py`, unchanged, with facing "full", the anchors of targets without a person as exempt points, and the default thresholds. Definitions as in A1 (`phase1c/README.md` section 1):
  - Recall = matched / visible. Matching is `sgi.run._matched_people` (greedy, nearest first, one to one, within 0.5 m); visibility is `is_visible`.
  - A false person is an unmatched detection not within 0.5 m of the anchor of a target with no person. Exempt detections are counted apart. A detection near a person who fails `is_visible` counts as false and is flagged.
  - View classes (angle between the true yaw and the bearing to the camera): frontal < 45, oblique 45 to < 75, profile 75–105, back > 105 deg.
  - Within 45 means error ≤ 45 deg; front/back correct means error < 90 deg. An unknown yaw counts as wrong, and as 180 deg in medians.
- **Intervals.**
  - Wilson 95% for proportions.
  - A layout-cluster bootstrap: 182 single-image layouts, 2000 draws, `random.Random(0)`, percentile intervals.
  - A seed-cluster bootstrap (13 seeds) as a sensitivity.
- **Development comparators.** From A1's recorded rows (FROZEN section 7): E-real, 42 images = 14 layouts × 3 arms; and the `real` arm alone, 14 images, which is the like-for-like comparator.

### Main tables

Table 1.1. The criteria on the pooled test set (182 images), judged once on point estimates, beside the development numbers.

| Bar | Test [Wilson 95%] | Layout bootstrap | Seed bootstrap | Dev E-real (42) | Dev `real` arm (14) | Outcome |
|---|---|---|---|---|---|---|
| S1 recall ≥ 0.95 | 510/520 = 0.981 [0.965, 0.990] | [0.968, 0.992] | [0.971, 0.990] | 116/118 = 0.983 | 42/42 | Met |
| S2 line members ≥ 0.95 | 207/208 = 0.995 [0.973, 0.999] | [0.984, 1.000] | [0.986, 1.000] | 48/48 | 16/16 | Met |
| S3 false people ≤ 0.10 per image | **44/182 = 0.242** | [0.176, 0.313] | [0.198, 0.280] | 3/42 = 0.071 (bootstrap 0.000–0.190) | 1/14 = 0.071 | **Not met** |
| S4 median position error ≤ 0.15 m | 0.061 m (n 510) | [0.058, 0.065] | [0.059, 0.064] | 0.053 m (n 116) | 0.053 m (n 42) | Met |
| S5 non-profile facing median ≤ 25 deg | 10.3 deg (n 287) | [8.5, 14.2] | [8.8, 13.0] | 6.4 deg (n 59) | 6.6 deg (n 23) | Met |
| S6 non-profile within 45 ≥ 0.85 | 278/287 = 0.969 [0.941, 0.983] | [0.946, 0.987] | [0.941, 0.990] | 59/59 | 23/23 | Met |
| S7 profile front/back correct ≥ 0.90 | 223/223 = 1.000 [0.983, 1.000] | [1.000, 1.000] | [1.000, 1.000] | 57/57 | 19/19 | Met |
| K1 kill: recall < 0.90 | 0.981 | [0.968, 0.992] | [0.971, 0.990] | 0.983 | 1.000 | Does not fire |
| K2 kill: > 0.30 false people per image | 0.242 | [0.176, 0.313] | [0.198, 0.280] | 0.071 | 0.071 | Does not fire |
| K3 kill: all matched within 45 < 0.70 | 494/510 = 0.969 [0.950, 0.981] | [0.953, 0.982] | [0.954, 0.982] | 111/116 = 0.957 | 41/42 | Does not fire |

- **Verdict: neither.** S1, S2 and S4–S7 are met, S3 is not, and no kill clause fires.
- **Intervals (descriptive, not judged).** For every bar except S3, the whole layout-bootstrap interval lies on the passing side. For S3, both bootstrap intervals lie wholly above 0.10. For K2, the layout bootstrap's upper end (0.313) crosses the 0.30 kill line; the seed bootstrap's (0.280) does not.
- **Counts.** 520 visible people: line 208, group 117, staff 78 (70 with the lower body hidden), other 117. 194 hidden people. 560 detections kept.
- **Other measures.** Yaw coverage 505/510 (the 5 unknown are all oblique). Merged neighbour pairs: 1 of 143 (development 0 of 33). Exempt detections: 6. False people within 0.5 m of a hidden person: 9.

Table 1.2. Recall by role class (k/n [Wilson 95%]).

| Role class | Test | Dev E-real | Dev `real` arm |
|---|---|---|---|
| line | 207/208 = 0.995 [0.973, 0.999] | 48/48 | 16/16 |
| group | 117/117 = 1.000 [0.968, 1.000] | 27/27 | 9/9 |
| staff | **69/78 = 0.885 [0.795, 0.938]** | 14/16 | 8/8 |
| other | 117/117 = 1.000 [0.968, 1.000] | 27/27 | 9/9 |

Table 1.3. By row, pooled over the 13 seeds (13 images per row).

| Row | Visible | Recall | False people (per image) | Exempt | Misses |
|---|---|---|---|---|---|
| `eval_00_no_one_place` | 12 | 12/12 | 0 (0.000) | 0 | 0 |
| `eval_01_no_one_place` | 25 | 25/25 | 4 (0.308) | 0 | 0 |
| `eval_02_no_one_place` | 8 | 7/8 | 1 (0.077) | 2 | 1 |
| `eval_03_no_one_deep_place` | 9 | 9/9 | 0 (0.000) | 0 | 0 |
| `eval_04_no_one_object` | 0 | — | 0 (0.000) | 0 | 0 |
| `eval_05_no_one_person` | 13 | 13/13 | 5 (0.385) | 0 | 0 |
| `eval_06_waiting_line_1` | 18 | 18/18 | 5 (0.385) | 0 | 0 |
| `eval_07_waiting_line_2` | 33 | 29/33 | **7 (0.538)** | 2 | 4 |
| `eval_08_waiting_line_4` | 54 | 53/54 | 6 (0.462) | 1 | 1 |
| `eval_09_waiting_line_6` | 89 | 88/89 | 5 (0.385) | 0 | 1 |
| `eval_10_conversation_group_2` | 52 | 52/52 | 1 (0.077) | 0 | 0 |
| `eval_11_conversation_group_3` | 70 | 70/70 | 3 (0.231) | 0 | 0 |
| `eval_12_conversation_group_4` | 91 | 91/91 | 0 (0.000) | 0 | 0 |
| `eval_13_waiting_line_3` | 46 | 43/46 | **7 (0.538)** | 1 | 3 |

Table 1.4. Facing by view class, all matched people, variant F5 (unknown counts as wrong, and as 180 deg in medians).

| View class | n | Known yaw | Within 45 | Median deg | Front/back < 90 | Dev E-real: n, within 45 (median), front/back |
|---|---|---|---|---|---|---|
| frontal | 95 | 95 | 95/95 | 3.2 | 95/95 | 18, 18/18 (1.1), 18/18 |
| oblique | 115 | 110 | 107/115 = 0.930 | 19.2 | 110/115 | 23, 23/23 (22.0), 23/23 |
| profile | 223 | 223 | 216/223 = 0.969 | 5.8 | **223/223** | 57, 52/57 (4.4), 57/57 |
| back | 77 | 77 | 76/77 = 0.987 | 8.3 | 77/77 | 18, 18/18 (4.9), 18/18 |
| **non-profile** | 287 | 282 | **278/287 = 0.969** | **10.3** | 282/287 | 59, 59/59 (6.4), 59/59 |
| all matched | 510 | 505 | 494/510 = 0.969 | 7.5 | 505/510 | 116, 111/116 (6.3), 116/116 |

**Seed batches** (descriptive, as frozen). Seeds 200–209 (140 images): recall 392/402 = 0.975, false 35/140 = 0.250. Seeds 210–212 (42 images; the clock left only these of the second batch): recall 118/118, false 9/42 = 0.214.

**Per seed** (post hoc). False people per 14 images: 4, 4, 2, 4, 4, 3, 2, 3, 4, 5 (seeds 200–209) and 4, 1, 4 (seeds 210–212). So 12 of the 13 seeds alone exceed 0.10 per image; only seed 211 (1/14) does not.

**Misses** (10 of 520):
- 9 are counter staff with the lower body hidden, all oblique (54–70 deg): `eval_07` ×4 (seeds 200, 204, 206, 209), `eval_13` ×3 (201, 202, 208), `eval_09` (200) and `eval_02` (201).
- Each was boxed: 8 have a best raw box of 0.670–0.893 over the point at 1.1 m; one (seed 201, `eval_02`) has no box over that point. Each was placed 0.63–0.65 m off (nearest kept detection, a post hoc diagnostic), so each also leaves a false person.
- The tenth is line member p2 of seed 205, `eval_08`, in profile (best raw box 0.936). Its neighbour's detection, 0.47 m away, took the match: the single merged pair.
- The best raw box is the highest-scoring box with score ≥ 0.05 that contains the projection of (x, y, 1.1 m) (`DEVIATIONS_1.md` C7).

Table 1.5. POST HOC: where the 44 false people come from (`confirm/posthoc_false.py`, written after scoring, from truth and rows only; no image was viewed; it changes no verdict).

| Class | Definition | n | Scores (min / median / max) | Rows |
|---|---|---|---|---|
| A, displaced counter staff | within 0.8 m of a visible staff member with the lower body hidden | 16: 9 pair with a staff miss; 7 are a second detection beside a staff member who is matched | 0.41 / 0.77 / 0.89 | `eval_09` 5, `eval_13` 5, `eval_07` 4, `eval_08` 1, `eval_02` 1 |
| B, hidden-by-rule staff | within 0.5 m (0.42–0.50 m) of a counter staff member who fails `is_visible` but whose point at 1.1 m projects into the picture | 9 | 0.55 / 0.84 / 0.88 | `eval_08` 4, `eval_07` 2, `eval_13` 2, `eval_06` 1 |
| C, near hidden-by-rule staff | 0.52–0.61 m from such staff, 1.21–1.34 m from the nearest visible person | 4 | 0.86 / 0.87 / 0.88 | `eval_06` 4 |
| D, far and small | 3.2–4.5 m from every true person; boxes 9–57 px tall | 15 | 0.25 / 0.29 / 0.53 | `eval_05` 5, `eval_01` 4, `eval_11` 3, `eval_07`, `eval_08`, `eval_10` 1 each |

- **Sensitivity** (post hoc). Without class A: 28/182 = 0.154 per image. Without A and B: 19/182 = 0.104. Without A, B and C: 15/182 = 0.082.
- **Class A.** These are A1's known counter-staff failure, now at scale: in development it gave 2 misses and 2 false people in 42 images. The 7 second detections beside matched staff lie 0.63–0.66 m from them, so the 0.50 m duplicate radius did not remove them.
- **Classes B and C.** The frozen scorer counts a detection of a person whom the package's visibility rule calls hidden as false. A1's development scenes had none (0 near a hidden person).

**Runtime** (not judged).
- Detector plus pose (`t_det_s + t_pose_s`): median 215.1 ms, p90 401.8 ms (n 182). This pass had no warm-up and posed every box ≥ 0.10 on a shared machine, so it is not the deployed path.
- By the same measure, A1's development raw pass gave 71.4 ms and the verifier's own pass on these images 127.0 ms. A1's deployed warm path was 55.6 ms (verifier 61.9 ms).
- Project side (`P.run_image`, 2 BLAS threads): median 3.4 ms, p90 8.0 ms.

### Checks

- **C1, code identity.** Every script verified all 9 pinned code hashes and the section 7 data hashes; the verifier confirmed them.
- **C2, development reproduction.** Run through the confirmation's code, A1's recorded raw pass gives 119 people on the 42 E-real scenes, equal to `a1/eval_people.json` within 1e-9. It also reproduces FROZEN section 7's E-real and `real`-arm numbers exactly (`confirm/c2_repro.txt`).
- **C3, truth self-check.** Given as detections, the true visible people score recall 1, 0 false and all within 45 deg on 182 of 182 test scenes.
- **C4.** The manifest assertions passed.
- **The raw pass reproduced.** The verifier's own sidecar pass reproduces all 182 images bit for bit: 2,791 boxes and 994 pose sets, with a maximum difference of 0 in boxes, scores and keypoints.
- **People and scoring reproduced.** The verifier's own lift-and-facing code reproduces the 560 detector people (largest difference 4.4e-16). Its own scorer reproduces all 520 person rows and every scene row with 0 differences, every bootstrap interval within 1e-12, and the post hoc breakdown.
- **No installs.** `pip freeze` of the sidecar (sha256 `20f344bf…`, 160 lines) was identical before and after both GPU passes.
- **Order** (file times, +08:00): `FROZEN.md` 01:17:11 → `FROZEN.sha256` 01:17:19 → render trial 01:22:06 → first seed folder 01:23:29 → render stop 04:31:15 → `DEVIATIONS_1.md` hashed 04:36:38 → C2 04:39:35 → manifest 04:41:03 → C3 04:42:27 → `score.py` saved 04:45:24 → the one detector pass 04:45:34–04:46:32 → scoring once 04:46:43–04:47:33 → post hoc breakdown 04:49:15 → verifier 04:53–05:03. There is no `raw_failed` folder and no sign of a re-run.
- **No held-out data.** The verifier searched `/`, the home folder, `/tmp` and `/dev/shm`: the only seed200–219 folders are seed200–212 under `phase2/seeds/realistic`. Nothing in `data/` changed after the freeze, and no seed 3–99 was touched. Development scenes were read only through the harness (C2).

### Criteria and outcome

| Criterion (FROZEN.md section 5) | Value | Outcome |
|---|---|---|
| Test set: each of seeds 200–209 complete | 13 complete seeds (200–212), 182 layouts | Met |
| S1 recall ≥ 0.95 | 510/520 = 0.981 | Met |
| S2 recall on line members ≥ 0.95 | 207/208 = 0.995 | Met |
| S3 false people ≤ 0.10 per image | 44/182 = 0.242 (both bootstrap intervals above 0.10) | **Not met** |
| S4 median position error ≤ 0.15 m | 0.061 m (n 510) | Met |
| S5 non-profile facing median ≤ 25 deg | 10.3 deg (n 287) | Met |
| S6 non-profile within 45 ≥ 85% | 278/287 = 0.969 | Met |
| S7 profile front/back correct ≥ 90% | 223/223 | Met |
| K1 kill: recall < 0.90 | 0.981 | Does not fire |
| K2 kill: > 0.30 false people per image | 0.242 | Does not fire |
| K3 kill: all matched within 45 < 0.70 | 494/510 = 0.969 | Does not fire |
| C2 and C3 | both pass | Met |
| **Overall** | | **Neither** |
| Prediction (FROZEN.md section 12; inference) | recall 0.95–0.99 (0.981); line ≥ 0.98 (0.995); false 0.03–0.15 (**0.242**); position 0.05–0.07 m (0.061); non-profile median < 15 deg (10.3); profile front/back ≥ 0.95 (1.000) | Held, except false people, which landed above the range |

Every criterion verdict agrees, and the verifier found no discrepancy in any reported number. Three departures from the note's text remain; none changes the verdict.
1. **The render trial, not disclosed.** FROZEN 1.4 says one generate command per seed, and seed 200 had two. The trial's scene was kept, and its 131.4 s were counted on the clock, as 1.4 defines the clock. Without the trial, the sum after seed 212 would have been 10,671.5 s, and seed 213 would also have been rendered. S3 would still fail: a 14th seed with no false person at all would give 44/196 = 0.224.
2. **The verifier's own detector pass.** FROZEN section 9 step 7 says the optional verifier runs no detector or pose model on a test image. The verifier's dispatch asked for its own sidecar pass, so it ran one, once, after scoring had finished, only to compare raw outputs. It was bit-identical and is logged in `verify_i09_confirm/v_order.txt`.

Not settled (not judged): the detector-plus-pose median, 215.1 ms in the compute pass against 127.0 ms in the verifier's, on the same machine under different load.

---

## 2. (B) The live-check build (`live/`)

### The frozen protocol (`live/FROZEN.md`)

- **Question.** Does the C pipeline keep A2's gain when the constraints call itself sees the detector's numbered people, with no VLM people call at all? A2 replayed constraints written against the VLM's own people. Here the model writes its constraints against the detector's people, numbered on the image and listed exactly as the package does for any people source.
- **Arm D-live.** The recorded C configuration at trial t0, with one change: the people source is A1's frozen detector people.
  - GLM-5.3 (gateway id x-test-1): `s19_{real,staffed,plain}_glm53_con` (11 Oct).
  - DeepSeek V4.1 Flash (x-test-2): `s20_{real,staffed,plain}_dsv41_con` (13 Oct).
  - Qwen is left out: it sits at its ceiling, and leaving it out keeps the run within the cap.
  - 84 records per model (3 arms × 14 scenes × 2 episodes: no_one 36, waiting_line 30, conversation_group 18).
  - The scenes are the 42 E-real development scenes, so this is not an unseen-layout test.
- **People source, DetectorPeople.**
  - It has OraclePeople's interface. For each scene it returns the people of `phase1c/i09/a1/eval_people.json` (sha256 `ffa59545…` checked on load), in descending score, with label None and the pixel built as OraclePeople builds it.
  - Scenes are matched by the harness fingerprint. A scene with no entry is an error, never an empty list.
  - The package then numbers, draws and lists the people as it does for every people source.
- **Validity.** D-live is "C with one change" only if every plan, locate and context answer is the recorded t0 answer. Hence two rules:
  - a count-only pass must find 0 upstream misses before going live;
  - after the run, the driver compares each record's plan and context answers with the recorded ones.

Table 2.1. Baselines at t0, 84 records per model (waiting_line successes of 30 in brackets), counted before the note.

| Model | Recorded C | A2 D replay | I01 P (true people) | Recorded C per trial (t0 / t1 / t2) |
|---|---|---|---|---|
| GLM | 64 (14) | 69 (19) | 69 (19) | 64 / 65 / 57 |
| DeepSeek | 65 (25) | 71 (28) | 70 (27) | 65 / 60 / 57 |

- **Pre-registered reading** (successes of 84):
  - consistent with A2 if GLM ≥ max(64, 69 − 3) = **66** and DeepSeek ≥ max(65, 71 − 3) = **68**;
  - contradicts A2 if GLM ≤ 63 and DeepSeek ≤ 64;
  - mixed otherwise.
  - It applies only to a complete run (no cap stop) whose upstream answers all match. With one trial it resolves only large effects. Since t0 is DeepSeek's best recorded C trial and GLM's second best, "at least recorded C t0" is a demanding bar.
- **Other measures** (reported, not part of the reading):
  - line successes, and flips and breaks against C, D and P, with the layouts they come from;
  - constraint usability;
  - the share of named numbers that exist in the list shown;
  - goal-step latency: D-live's constraints call against recorded C's people call plus constraints call, with A1's 55.6 ms detector time added as a constant;
  - descriptive extras.
- **Calls.** A call is one request sent to the back end; validation and transient retries count. Expected: 102 new constraints calls (GLM 48 = 17 + 16 + 15; DeepSeek 54 = 18 + 20 + 16), the recorded t0 goal steps. None of the 102 recorded t0 constraints calls needed a retry. Hard stop: the 121st call in total, or the 61st of one model, is refused.
- **Go-live rule.** Go live only if count-only finds at most 120 misses in total (upper bound), at most 60 per model, and 0 plan, locate and context misses. Then GLM first, then DeepSeek, into the same output folder.

### What the driver does (`live/driver.py`, sha256 `e6e73181…f9470`, in `BUILD.sha256`)

- **Episodes.** It runs the package's own `sgi.run.run_episode` from `src_202c165/src`. One hook is replaced: `sgi.run._given_inferrer`, which for `run.people: oracle` returns a constraint inferrer over OraclePeople. The driver returns the same inferrer over DetectorPeople instead, so the trial still enters the constraints call's cache key.
  - `run.people` stays `vlm` in `config.yaml`, because the config has no detector value. A `people_source` block and a per-record `config.people_source` mark the detector.
  - The people stage's client raises `UnexpectedCall` if it is ever asked.
- **Calls.** Plan, locate and context requests are rebuilt as recorded, so the cache answers them; only the constraints call of each goal step is new.
  - A ledger (`<out>/ledger.jsonl`) counts every request before it is sent and enforces the caps.
  - `--count-only` builds no back end and reads no key. A miss is counted and answered locally as `vlm_error`; nothing is sent and nothing is written to the cache.
- **Guards and keys.**
  - It refuses to start if `FROZEN.md` no longer matches its hash, or if the prompts differ from a run's `prompts/` folder.
- **Outputs.**
  - One run folder per arm and model (`<time>_s24_i09_<arm>_<model>_con/`) in the package's format.
  - A check file per invocation, with the cache counts by stage and a record-by-record comparison of the upstream answers.
  - The harness reads these names as arm `ph`, so the runs are selected by name.

### Dry-run checks (zero calls; `live/dryrun/`, `verify_i09_live_build/`)

| Check (FROZEN.md section 8) | Builder | Verifier |
|---|---|---|
| (a) Mock model end to end, 202c165 code, recorded entries | 10 episodes, 3 arms (no_one 2, waiting_line 4, conversation_group 4) | 16 episodes on 8 scenes, including `staffed_eval_04` with 0 detector people |
| The people stage makes no call | plan, context, constraints 10 each; people 0 | 16 each; people 0 |
| The constraints request carries the detector's numbered people | 10 of 10 numbered images byte-equal to an independent rebuild. Two images viewed: in `real_eval_08` the detector's one false person in that scene sits at the counter as a numbered disc | 16 of 16 |
| Nothing sent leaks a scene id, target id, situation or forbidden-region kind (`tests/test_leakage.py` rule) | 30 requests, 11 secret strings, none leaked | 48 requests, 14 secret strings, none leaked |
| Run folders readable by the package, the harness and rescore | 10 of 10 | — |
| (b) Count-only on an empty cache | 84 misses per model (all at plan), 0 sent, 0 cache writes | Same |
| (b2) Count-only on a cache filled by a fake gateway | exactly the 6 removed constraints keys miss; plan and context hit | Replicated |
| (c) Hard stop with fake clients | c1–c6 pass: caps across two ledger instances, retries counted, 121st and 61st refused, folder marked incomplete, exit 3 through `main()` | Sequential runs at 120/60 pass with retries counted. **A concurrent run of the same model passes the cap: 114 sends against 60** |
| DetectorPeople | — | Equals the package's OraclePeople output for the detector positions on 42 of 42 scenes (119 people); errors on an unknown scene or another scene's observation |
| Which models can be called | — | Only x-test-1 and x-test-2 at the school gateway |

Builder total: 17 of 17 checks pass. Verifier: everything it re-ran passes except its new lock test.

### The verifier's review (verdict: mixed)

- **Defect, to settle before going live: the lock.**
  - In `main()` the lock is taken inside the `try`, and `finally` releases it unconditionally. So an invocation refused with "ledger.lock exists" deletes the running invocation's lock.
  - The ledger also counts `ledger.jsonl` only when it starts, so two concurrent invocations each enforce the cap against a stale count.
  - Shown with a fake gateway and zero model calls (`verify_i09_live_build/e_trace.txt`): A (GLM) runs; B (GLM) is refused and removes A's lock; C (GLM) then runs beside A. A sent 60 and C 54, which is 114 GLM sends against a cap of 60.
  - Two different models running at once cannot pass 120 in total, since each is held to 60; it takes two concurrent runs of one model.
  - Suggested fix: take the lock before the `try` (or release it only if this invocation took it), and re-read or lock the ledger at each reservation. Then re-hash `BUILD.sha256`, update `REQUEST.md` and re-run test c. `FROZEN.md` would not change.
- **Exit codes are documented wrongly.**
  - Every refusal (the lock, a FROZEN mismatch, differing prompts, an unknown scene), a missing key, `UnexpectedCall` and `SceneError` exit 1, not 2. Only argument errors and quota or back-end errors exit 2.
  - `REQUEST.md` says "if a model exits 1, run the same command once more". After a lock refusal, that advice leads straight to the concurrent run above.
- **Suggestions.**
  - The cap counts per `--out` folder, so every command must use `$LIVE/runs`.
  - DeepSeek has 6 calls of headroom (54 of 60). The recorded t0 rate was 8 transient retries in about 276 DeepSeek calls, so a cap stop is unlikely but possible.
  - `REQUEST.md` has no `sha256sum -c BUILD.sha256` step.
  - The driver writes `__pycache__` into `src_202c165/` (harmless).
- **Agreed.**
  - `FROZEN.md` was hashed before the driver existed.
  - The prompts, the expected calls, the baselines, the reading thresholds, the trial spread and the recorded latencies all reproduce.
  - The sidecar is untouched, and no held-out scene, run or seed was read.

- **The steps.**
  1. Check `FROZEN.sha256`.
  3. Apply the go-live rule.
  4. Go live with GLM, then DeepSeek, into the same `--out`.
  5. Send back the console output and exit codes, the count-only and check files, the six run folders, and the ledger totals.
- **Expected cost and time.** 102 calls on the free school gateway (GLM and DeepSeek only; no Claude, no Gemini), about 6 and 8 min (guess), one CPU process, no GPU.
- **Status at 05:10 on 18 Oct.** `driver.py` is still `e6e73181…` and `REQUEST.md` `105edfb3…`, both unchanged since the verifier's review. There is no `live/runs/` folder, so neither count-only nor the live run has happened. **No live result.**
- **Analysis.** The analysis of FROZEN sections 5–6 runs offline after the live run; its script is not built yet.

---

## 3. What it means (inference)

**I09's detector on unseen layouts.**
- **What carries over.** Recall, placement and facing hold on new layouts of the same 14 rows:
  - recall 0.981 against 0.983 in development, and line members 207 of 208;
  - position error 0.061 against 0.053 m;
  - every facing bar met, with its whole interval on the passing side.
- **The facing fix holds where it mattered.** Back views are 76 of 77 within 45 deg, and profile front/back is right for 223 of 223. The non-profile median is higher than in development (10.3 against 6.4 deg), still well under the 25 deg bar.
- **False people do not carry over.** At 0.242 per image the rate is about 3.4 times the development rate and above the predicted range. The development estimate rested on 3 false people in 42 images, and its interval (0.000–0.190) already spanned the bar. Post hoc, 12 of 13 seeds alone exceed 0.10 per image, so this is not one bad seed. The confirmation did what it was for: the 14 development layouts under-sampled this failure.
- **Where the false people come from (post hoc).** About two thirds relate to counter staff, in two different ways:
  - **(a) A1's known displacement.** The staff member is boxed but placed 0.63–0.66 m off: 16 false people and 9 misses here, against 2 and 2 in development. Why (perhaps the torso median catching the counter in front) remains an untested guess, as in phase 1c.
  - **(b) Disagreement with the visibility rule.** 13 detections sit 0.42–0.61 m from counter staff whom the package's visibility rule calls hidden, though their marked point is in the picture. Whether such a detection should count as false is a question about the scorer and its visibility rule, not a detector fault; the frozen scorer counts it.

  The rest, 15 small low-score boxes far from everyone, would alone give 0.082 per image.
- **What it would mean in use.** As a people source on scenes like these, the detector would supply near-complete recall and good facing, plus about one extra person every four images, most often near the counter.

**A second confirmation, if wanted.**
- **New pre-registration.** Any change the breakdown suggests is data-informed: counter-staff placement, the duplicate rule, a higher threshold, or how detections of hidden-by-rule people are scored. It would need its own pre-registration, a fit on data other than seeds 200–212, and fresh unseen layouts.
- **A scorer change.** Counting detections of hidden-by-rule people differently would change the measure, not the method. It would have to be applied to the development numbers as well.

**The live check.**
- **What it can test.** It runs on the development scenes, where the detector had 3 false people in 42 images, so it cannot show what the unseen-layout rate would do. On unseen layouts the constraints call would see an extra numbered person about once every four images, often at the counter; the dry-run image of `real_eval_08` shows one such disc. Whether that hurts the constraints is open.
- **What a result would show.** A "consistent" result would show that A2's gain survives when the model writes its constraints against the detector's people. It would also measure the saving from dropping the people call (recorded medians 2.1 s for GLM and 3.7 s for DeepSeek, against about 0.06 s for the detector). It would test neither Qwen nor unseen layouts, and one trial resolves only large effects.

3. **Lead.** Should the confirmation's departures be accepted and recorded in a hashed `DEVIATIONS_2.md`? They are `DEVIATIONS_1.md` R1–R4, the render trial that was not disclosed, and the verifier's own detector pass.

---

## 5. Proposed REPORT edits (not applied)

Each item gives the exact old text from `REPORT.md` and the proposed new text. Each old text occurs exactly once (checked with Python's `str.count`), and applying items 1–6 in order works (checked on a copy in memory; each old text still occurs exactly once when its turn comes). The tag [phase 2] is defined in item 1; another phase-2 page may propose the same definition, and one of them should be applied. Items marked optional are not required by this step.

### 1. "How to read": define [phase 2]

Old:
```text
Parts registered after earlier results on the same runs are marked data-informed.
```
New:

### 2. I09: the confirmation result and the live check's status

Old:
```text
Calling it a method would also need a confirmation on unseen layouts, with nothing re-tuned.
```
New:

### 3. Revision note

Old:
```text
add step-3b notes to the opening lines, Programmes A and C, section 6's cut order, I46's seeds, section 7's runs and section 8's render threat.
```
New:

### 4. I46: the seeds used (optional)

Old:
New:

Old:
New:

### 6. Section 7, the runs: the live check's status (optional)

Old:
New:

---

## 6. After this page was written (the research lead, 18 Oct)

- **The lock defect is fixed** (section 4, question 1: fixed rather than run under a rule). `FROZEN.md` is unchanged.
  - Round 1 (driver `84bab2e4`): the lock is taken before the `try`; the ledger is re-counted under `fcntl.flock` at
    every reservation; live mode refuses plan, locate and context sends (`UpstreamMiss`); refusals exit 2. Two
    independent verifiers checked it with zero calls and a fake gateway (`verify_i09_live_fix/tests/`,
    `verify_i09_live_fix/review/`): ready with notes. The caps held exactly under concurrency (3 GLM + 1 DeepSeek
    processes: 60 + 60), and the first build's ledger let 320–360 through under the same race.
  - Round 2 (driver `41940c3c`): the lock is an exclusive `flock` held for the run's life, so no stale lock is left;
    a ledger losing rows stops the run; every stop writes the check file; an unknown `--only` scene is refused;
    SIGTERM and SIGHUP clean up as Ctrl-C does. A re-test passed R1–R7 (`verify_i09_live_fix/retest/`).
  - Earlier builds are kept in `live/before_fix_e6e73181/`, `before_fix2_84bab2e4/` and `before_fix3_41940c3c/`.
  - `REQUEST.md` was rewritten: every command in full with absolute paths, `run_in_background` for the driver commands,
    at most one re-run on exit 1, no DeepSeek after a GLM failure, what a lock refusal means, which exits write a
    check file, and that `people_asked` counts detector answers rather than model calls.
- **The offline analysis** is built and hashed before any live run (`live_analysis/`, `analyse.py` `ab66ec30…` after
  one pre-run amendment recorded in `LEAD_DECISIONS.md`). It re-derives the baselines, applies FROZEN's reading only
  when every gating validity check passes, and was tested on 25 synthetic and dry-run inputs (25 of 25 pass). Its
  independent verifier rated it ready with notes. The result will be computed with `--count-vlm-errors-as-failures`
  (FROZEN 5.1).
