# I09 unseen-layout confirmation (phase 2): frozen protocol

The I09 confirmation lead wrote this on 18 Oct 2026, between 01:14 and 01:30 +08:00, before any scene of seeds 200–219 existed (section 1.6 gives the check). Its sha256 is in `FROZEN.sha256`. This file is never edited. Changes and clarifications go into numbered `DEVIATIONS_<n>.md` files, each with its own sha256, written before the data they affect is processed.

Path roots:
- Bare paths are relative to this folder, `results/still-scenes/people-finder/early-detector/phase2/confirm/`.
- Paths that start with `phase1b/`, `phase1c/` or `phase2/` are relative to the research folder `results/2026-10-17-beyond-vlm/`.

- it rests on 14 layouts;
- it needed one data-informed facing fix (`results/still-scenes/people-finder/early-detector/phase1c/a1/ADDENDUM_FIX.md`);
- the false-people interval spans its bar (layout bootstrap 0.000–0.190 against 0.10).

The phase 1c README concluded that calling the detector a method needs a confirmation on layouts it has never seen, with nothing re-tuned. This is that confirmation.

**Rulings** (lead researcher, 18 Oct). This note implements them; where it fills a gap, section 11 lists the choice.
1. Purpose as above.
3. Method: A1's frozen pipeline exactly, with the one adopted fix. Nothing is re-tuned and no new fix is allowed.
4. Scorer: `results/still-scenes/perception/depth-blob-detector/people_scorer.py`, unchanged, with facing "full" and `exempt_xy=exempt_points(truth)`, under the default thresholds. Visibility follows the package's rule.
5. Criteria: A1's success bars verbatim, on the pooled test set; kill as stated there (section 5). Runtime is not judged again.
6. Order: this note, the renders, the manifest, one detector pass, scoring once.

**What a success would show, and what it would not.** It would show that A1's realistic-image result holds on 140–280 layouts the method has never seen, with nothing re-tuned. Those layouts come from the same 14 rows (3 situations: no_one, waiting_line, conversation_group), the same renderer and asset library, the `real` variant only, and exact rendered depth. It would not test:
- other situations, the audience rows or the held-out situations;
- the placeholder avatars, where A1's recall was 0.845;
- real photographs or sensor depth;
- the pipeline with the detector in it (the I09 live check, a separate step).

## Exposure (disclosed)

**What I read before writing:**
- REPORT I09 and `results/still-scenes/overview/phase-notes/phase1c.md`, in full;
- `results/still-scenes/people-finder/early-detector/phase1c/FROZEN.md`, `DEVIATIONS.md`, `DEVIATIONS_2.md`, `a1/ADDENDUM_FIX.md`, `a1/chosen.json` and `a1/chosen_fix.json`;
- the code of `a1/pipeline.py`, `a1/common.py`, `a1/sidecar/infer.py`, the first 200 lines of `a1/evaluate.py`, and `results/still-scenes/perception/depth-blob-detector/people_scorer.py`;
- `docs/realistic_scenes.md`;
- the docstring, `generate()` and CLI of `src/sgi/synth/realistic/generate.py`, the row tables and `accepted_layouts` of `src/sgi/synth/generate.py`, and the render job's fields;
- the guards of `docs/model_notes/tools/research_harness.py`, `results/still-scenes/perception/depth-blob-detector/stageA/fitdata.py`, and the thresholds in `configs/default.yaml`.

**What I computed.** I computed section 7's development comparators from A1's recorded `a1/people_rows.jsonl` and `a1/rows.jsonl`.

**What I therefore know.** I know A1's development results in full, including its weak spots:
- 2 counter staff with the lower body hidden (both in the `staffed` arm), each boxed but placed 0.62–0.65 m off, so each counts as a miss and a false person;
- placeholder avatars seen side-on or from behind;
- 5 profile line members of `eval_09` at 47–73 deg, with front and back still right.

I also know that in seed 0 the vendor at the second stall stands outside the observation picture in all 21 scenes (`docs/realistic_scenes.md`). In other seeds the vendor may stand inside it.

**What this note chooses, and what it does not.** It chooses no part of the method: the method is A1's frozen choice. It chooses the test-set mechanics, the report, and definitions the rulings leave open (section 11).

**1.1 Seeds.**
- They are not part of I46's sealed seeds (3–99) and are never used for fitting.
- A1's fit data was seed 0's worked examples plus placeholder seeds 100–109. Its development scenes were seed 0's eval rows plus development seeds 0–2. No layout of seeds 200–219 has been drawn before.
- The rows are the development rows' designs; only the layouts are new. This tests new layouts, not new situations.

| Row | Scene id | Row | Scene id |
|---|---|---|---|
| 00 | `real_eval_00_no_one_place` | 07 | `real_eval_07_waiting_line_2` |
| 01 | `real_eval_01_no_one_place` | 08 | `real_eval_08_waiting_line_4` |
| 02 | `real_eval_02_no_one_place` | 09 | `real_eval_09_waiting_line_6` |
| 03 | `real_eval_03_no_one_deep_place` | 10 | `real_eval_10_conversation_group_2` |
| 04 | `real_eval_04_no_one_object` | 11 | `real_eval_11_conversation_group_3` |
| 05 | `real_eval_05_no_one_person` | 12 | `real_eval_12_conversation_group_4` |
| 06 | `real_eval_06_waiting_line_1` | 13 | `real_eval_13_waiting_line_3` |

How the generator behaves:
- It replays `accepted_layouts(seed)`, which draws the first pass only: 14 eval rows and 7 example rows, all of no_one, waiting_line or conversation_group.
- It writes each layout tried into a temporary folder and removes it again.
- It renders only the named rows.
- It never draws the second pass (audience, group_photo, lift_wait, the lift and the screen).

Nothing is written into `data/`.

**1.3 Layouts.** A layout is a (seed, row) pair. There is one image per layout, the `real` variant. The `staffed` and `plain` variants are not rendered.

**1.4 Order and stop.**
- **Rendering wall time** is the summed wall-clock time of the generate commands, each from its start to its end. Waits for GPU or memory before a command starts are logged and not counted.

**1.5 Complete seeds and the test set.**
- A seed is **complete** when all 14 scene folders exist under `seed<k>/eval/`, each holding `scene.json`, `rgb.png`, `depth.npy`, the tour frames and `render_job.json`, and loading with `sgi.scene.load_scene`. Its render job must say variant `real`, seed k and renderer `blender`.
- The **test set** is every complete seed at the stop.
- **Required:** each of seeds 200–209 is complete (140 layouts). Otherwise the outcome is "incomplete", and section 5's criteria are not judged.

**1.6 Nothing existed before this note.** At 01:12:10 +08:00:
- a search of `~` and `/tmp` found no folder named `seed200`–`seed219`;
- `results/still-scenes/scene-seeds/phase2/realistic/` did not exist;
- no `sgi.synth` or Blender process was running.

`results/still-scenes/scene-seeds/phase2/` held only another step's placeholder seeds 110–113.

## 2. The manifest (project venv)

**It asserts:**
- every scene folder is guarded with `H.guard_scene` and lies under `phase2/seeds/realistic/seed<k>/eval/`, with 200 ≤ k ≤ 219;
- each complete seed holds exactly the 14 ids of section 1.2, and nothing else, under `eval/`;
- `examples/` is absent or empty;
- no file or folder name anywhere under `results/still-scenes/scene-seeds/phase2/realistic/` contains `group_photo`, `lift_wait`, `heldout`, `held_out` or `eval_16`–`eval_19` (case ignored);
- `render_job.json` gives variant `real`, the folder's seed and renderer `blender`;
- its thresholds equal `configs/default.yaml`'s, every key;
- `scene.json`'s scene id equals the folder name.

**Each entry holds:** seed, row number, row id, scene id, folder, the rgb path, and the sha256 of `rgb.png`, `depth.npy`, `scene.json` and `render_job.json`. It also holds the render job's attempt and renderer version.

**Order:** seed ascending, then row ascending. The manifest also lists every seed left out, with the reason.

**Setting.** From `results/still-scenes/people-finder/early-detector/phase1c/a1/chosen.json` and `chosen_fix.json`, unchanged:
- RT-DETRv2-r50vd with score threshold τ 0.25;
- height band [1.00, 2.30] m, flat test off;
- duplicate radius 0.50 m, depth push 0.20 m;
- facing variant F5 (F4 without the face rule: shoulders and hips, plus the nose-against-ears profile rule).

**Code by import.** Nothing is copied. Every script asserts these sha256 values at start and refuses to run on a mismatch.

| File | sha256 | Used for |
|---|---|---|
| `results/still-scenes/people-finder/early-detector/phase1c/a1/pipeline.py` | `4826be88d6939f3f5e4d14375a2fc9698abd94973ca46243d6697e263e5d063a` | `run_image`, which calls `candidates` and `people_for`: lift, filters, duplicates, F5 facing |
| `results/still-scenes/people-finder/early-detector/phase1c/a1/common.py` | `cdc33d742f6db5d09c3ad9a4557c1210ef10f8909484cee101ce983f9f8b2f53` | `score_scene`, `view_angle_deg`, `view_class`, `facing_summary`, `boot_ci`, `p90` |
| `results/still-scenes/perception/depth-blob-detector/people_scorer.py` | `e12776eaddac1de0af75b58f8a272cd0a13332596da626bcc81b085f5900d685` | `score_people`, `pool`, `wilson`, `exempt_points`, `truth_info`, `default_thresholds`, `FreshScene` |
| `results/still-scenes/people-finder/early-detector/phase1c/a1/sidecar/infer.py` | `bee1322fd692bf0e85e820871413d486fda04d5770f72b5d3d75e18266709484` | `setup`, `env_info`, `load_detector`, `load_pose`, `detect`, `pose`, and `BOX_MIN`, `POSE_MIN` |
| `results/still-scenes/people-finder/early-detector/phase1c/a1/chosen.json` | `19b1e16340d810e75f8f772eeeabeaf4b788e09614e74d78d3afc206531119ff` | the setting |
| `results/still-scenes/people-finder/early-detector/phase1c/a1/chosen_fix.json` | `fe8177165773f7208d417227c358adae7d04326bd29cb6c75d73ed87b3f83f52` | facing F5 (`fix_adopted`; its setting is asserted equal to chosen.json's) |
| `results/still-scenes/people-finder/early-detector/phase1c/a1/ADDENDUM_FIX.md` | `3a7d185526e3d9801bd2adcdaf529bfbe744c1a636dc8e03d0f53de45d2e208c` | the fix's record |
| `results/still-scenes/people-finder/early-detector/phase1c/FROZEN.md` | `f315e96c1145ad9f2c86128d16711bb5305690d99f5d8513c0177290c474df15` | A1's protocol |
| `configs/default.yaml` | `c651f4c484fff15d33290e13f28bc53c9221c8edd79c0b7a24f24a128098b4fc` | thresholds |

`pipeline.py` and `common.py` carry the hashes A1 recorded in `a1/EVAL.sha256`. `infer.py` (last saved 21:34:58 on 17 Oct, before A1's fit pass) and `people_scorer.py` (18:58:56 on 17 Oct) are pinned here at their current contents.

- It imports `infer.py` by path and calls its model functions unchanged:
  - RT-DETRv2 from the snapshot `models--PekingU--rtdetr_v2_r50vd/snapshots/2824940…c817`, keeping every person box with score ≥ 0.05;
  - ViTPose-base-simple (`a93ac0c…4c45`) on every box with score ≥ 0.10, as one batch per image;
  - fp32, eval mode, `torch.manual_seed(0)`, cudnn deterministic, benchmark off, TF32 off;
  - no warm-up, as in A1's evaluation pass.
- It writes one line per image in the format of A1's `raw_eval/rtdetr_v2_r50vd.jsonl`, so `pipeline.candidates` reads it unchanged. It also writes a meta JSON.
- It replaces only two functions of `infer.py`. Neither changes what the models compute:
  - `guard()`, which would refuse a seed200–219 folder. The new guard asserts that every rgb path, as written and resolved, lies under `phase2/seeds/realistic/seed<k>/eval/` with 200 ≤ k ≤ 219, holds exactly one `seed<k>` component, and contains none of `keep/heldout`, `s18`, `s20_held`, `eval_16`–`eval_19`, `group_photo` or `lift_wait` (case ignored). It also checks each image's sha256 against the manifest.
  - `wait_for_gpu()`, which waits for an idle GPU. Its replacement follows the shared-machine rule in section 10.
- D-FINE is not loaded.

**Project side** (project venv, 2 BLAS threads). For each image: `P.run_image(raw, depth, cam, setting, "F5")`. Only F5 is scored and reported. No situation, label or row is read before the people are produced; roles and rows are used for tabulation only.

**Loading.** Test scenes are loaded the way I03 loaded its fresh seeds: `H.guard_scene(folder)`, then `PS.FreshScene(load_scene(folder))`. The sha256 of `rgb.png`, `depth.npy` and `scene.json` is checked against the manifest.

**Thresholds.** `th = PS.default_thresholds()`, which is `configs/default.yaml`. `PS.check_thresholds(th)` is asserted: person radius 0.25 m, 3 visibility samples, 0.3 m visibility depth tolerance, 0.05 m grid. As in A1's fit scenes.

**Scoring.** `C.score_scene(people, sc, th, "F5")`, which calls `PS.score_people(points, sc, th, exempt_xy=PS.exempt_points(sc.truth), facing="full")`, unchanged. Results are pooled with `PS.pool`.

**Definitions:**
- **Visible:** passes `sgi.gt.derive.is_visible` under th.
- **Matching:** `sgi.run._matched_people`: greedy, nearest first, one to one, within 0.5 m.
- **Recall:** matched / visible.
- **False person:** an unmatched detection that is not within 0.5 m of the anchor of a target with no person. Exempt detections are counted apart. **False people per image** = pooled false / images.
- **Position error:** the xy distance of each matched pair. The median is taken over all matched people.
- **View class:** from the angle between the true yaw and the bearing from the true person to the camera's xy (`C.view_angle_deg`, `C.view_class`): frontal < 45, oblique 45 to under 75, profile 75–105 (both ends in), back > 105 deg. **Non-profile** = frontal + oblique + back.
- **Facing error:** |wrap(yaw_det − yaw_true)| in degrees, for matched people. **Within 45** means error ≤ 45 deg. **Front/back correct** means error < 90 deg. An unknown yaw counts as wrong, and as 180 deg in medians. Coverage (matched people with a known yaw) is reported.
- **Role class** (`PS.role_classes`): staff (has a role), line (group at a target of a waiting_line episode), group (group at a target of a conversation_group episode; the scorer's audience and group_photo cases cannot occur in these rows), and other.
- **Line members:** role class "line".

Success needs all of:

| # | Bar (verbatim REPORT I09) | Population |
|---|---|---|
| S1 | recall at least 0.95 overall | all visible people |
| S2 | recall at least 0.95 on line members | visible line members |
| S3 | at most 0.10 false people per image | all test images |
| S4 | median position error at most 0.15 m | matched people |
| S5 | facing for non-profile people: median at most 25 deg | matched non-profile people, unknown = 180 deg |
| S6 | facing for non-profile people: at least 85% within 45 deg | matched non-profile people, unknown wrong |
| S7 | profile front/back correct at least 90% | matched profile people, unknown wrong |

A kill fires on any of:

| # | Clause | Population |
|---|---|---|
| K1 | recall below 0.90 | all visible people |
| K2 | more than 0.30 false people per image | all test images |

**How it is judged:**
- Bars are judged on the point estimates, as in A1. Intervals are reported and not judged.
- "At least" means ≥, "at most" ≤, "below" <, "more than" >.

**Outcomes:**
- **Success:** S1–S7 all met and no kill clause.
- **Kill:** any of K1–K3. REPORT I09's consequence applies: "keep it only as a recall cross-check".
- **Neither:** any other outcome.
- **Incomplete:** section 1.5's requirement is not met.

**Not carried over from A1, by the rulings:**
- the runtime bar (not judged again);
- the kill clause "more than 10 points below the best VLM" and its A0-pair population, because no VLM ran on these scenes;
- A2, which is not part of this step.

## 6. The report

**Main table** (pooled test set). For each bar:
- the value with its n;
- a Wilson 95% interval for proportions (`PS.wilson`);
- a layout-cluster bootstrap 95% interval. Clusters are (seed, row) layouts, each one image with all its people. There are 2000 draws with `random.Random(0)`, percentile intervals, through `C.boot_ci`, and the statistic is computed on each draw as on the full set;
- section 7's two development comparators beside it.

The verdict line follows. A descriptive line also says, for each bar, whether its whole bootstrap interval lies on the passing side.

**Also reported:**
- recall by role (line, group, staff, other) and by row (pooled over seeds), each as k/n with Wilson 95%;
- false people per image by row;
- facing by view class: n, within 45, median, front/back;
- coverage and merged neighbour pairs;
- counts of images, layouts, visible people and staff;
- misses by class (staff behind counters, other staff, non-staff) and by role, each with its best raw box, meaning the highest-scoring box with score ≥ 0.05 that contains the projection of (x, y, 1.1 m), and with its score;
- false people with their score, distance to the nearest visible person, and whether they are near a hidden person;
- the two seed batches, 200–209 and 210–219, side by side (descriptive);
- a seed-cluster bootstrap (whole seeds resampled, 2000 draws, seed 0), as a sensitivity.

**Runtime** (not judged): the median and p90 of `t_det_s + t_pose_s` per image from the detector pass, and the project side's median per image. The GPU is shared, and this is not A1's deployed timing path; both facts are stated.

Any analysis not listed here is labelled post hoc.

## 7. Development comparators (from A1's recorded rows; not recomputed)

Sources, with their sha256 as in `a1/EVAL.sha256`:
- `results/still-scenes/people-finder/early-detector/phase1c/a1/people_rows.jsonl`: `96384dab0cb313715aa27c04fe1f8f4274b5b6ef3a8f83d6e1aa6cdf248f9f86`
- `results/still-scenes/people-finder/early-detector/phase1c/a1/rows.jsonl`: `d0d889051e97720f93df32a6dfbeab1126438385cf438322fa49daeb2331d51f`
- `results/still-scenes/people-finder/early-detector/phase1c/a1/summary.json`: `f2abcf10dc74623f43ebfd8f4f332a8b43320161d28b73064ad63afa703df60c`

The `real`-arm column is the like-for-like comparator, since the test set renders the `real` variant only.

| Measure | E-real (42 images, 14 layouts × 3 arms) | `real` arm only (14 images) |
|---|---|---|
| Recall | 116/118 = 0.983 | 42/42 |
| Line members | 48/48 | 16/16 |
| Group / staff / other | 27/27 / 14/16 / 27/27 | 9/9 / 8/8 / 9/9 |
| False people (per image) | 3 (0.071; layout bootstrap 0.000–0.190) | 1 (0.071) |
| Median position error | 0.053 m (n 116) | 0.053 m (n 42) |
| Non-profile within 45 (median) | 59/59 (6.4 deg) | 23/23 (6.6 deg) |
| Profile front/back | 57/57 | 19/19 |
| All matched within 45 | 111/116 | 41/42 |

The development bootstrap resampled 14 layouts with their 3 arms each. The test bootstrap resamples single-image layouts.

## 8. Checks before the detector pass

- **C1. Code identity.** The hashes of section 3, asserted by every script.
- **C2. Development reproduction** (no test data):
  - The confirmation's project side and scoring run on A1's recorded `a1/raw_eval/rtdetr_v2_r50vd.jsonl` (sha256 `8a03b82ca89bb82bdd0b10a8e56944b8562f3a580967087646e408dd4862a97d`, as in `a1/EVAL.sha256`) for the 42 E-real scenes, loaded through the harness (`H.distinct_scenes`, `H.scene`, A1's recorded thresholds through `C.thresholds_for`).
  - They must reproduce `a1/eval_people.json`'s people for those scenes: same order, xy and yaw within 1e-9.
  - They must also reproduce section 7's E-real counts and medians exactly. The bootstrap interval is not part of this check.
- **C3. Truth self-check** on the test scenes (reads truth only). The true visible people, given as detections with their true yaws, must score recall 1, 0 false and every person within 45 deg on every test scene.
- **C4.** The manifest assertions of section 2.

A failed check stops the step. Its fix is logged in a hashed `DEVIATIONS_<n>.md` before work goes on, and no fix may change the method, the scorer, the thresholds or the criteria.

## 9. Order and failure rules

1. This note and `FROZEN.sha256`.
3. C1 and C2, at any time before step 5.
4. `manifest.json` and `MANIFEST.sha256`, then C3.
5. One detector pass: the sidecar over the manifest, once, giving `raw/rtdetr_v2_r50vd.jsonl` and `raw/meta.json`, hashed into `RAW.sha256`. `pip freeze` of the torch environment is recorded before and after and must not change.
6. Scoring once: `score.py` gives `tables.txt`, `rows.jsonl`, `people_rows.jsonl` and `summary.json`, hashed into `SCORE.sha256`.
7. Optional: an independent verifier recomputes step 6 from the same raw pass with its own code. It runs no detector or pose model on a test image.
8. `README.md`.

`ORDER.txt` logs each step's timestamps. Every script verifies the hashes of everything it reads.

**Failure rules:**
- **The detector pass.** If it dies before it has written every image, for any technical reason, nothing is scored. The partial output is moved to `raw_failed_<n>/` and kept, and the pass is run again from the start, unchanged. The overlapping images are then compared and the comparison reported.
- **The scoring script.** If it fails before it writes any metric, the bug is fixed (logged with its diff in a hashed `DEVIATIONS_<n>.md`) and the script is run again. Once a metric has been written, there is no re-run, except to fix a bug a verifier finds. In that case both values are reported.
- **Viewing test images.** Nobody views a test image until scoring is finished. Any later look, for diagnosis, is post hoc.

## 10. Resources and safety

**Shared machine:**
- Before the GPU job, `nvidia-smi` must show at least 3,000 MiB free.
- Before any heavy job, `free -m` must show at least 3,000 MiB available.
- Otherwise the job waits, polling every 5 min, and the wait is logged.

**No calls and no installs:**
- No model-provider or network call and no download; Hugging Face runs offline.
- Nothing is installed in either environment.

**Excluded data:**
- No held-out situation, scene or run is read or instantiated, and no seed 3–99 is generated or opened.
- Recorded runs and whitelisted scenes (C2 only) are read through the step-0 harness alone.

## 11. Choices this note makes where the rulings leave room

3. **Code reuse by import, not copy.** The sidecar driver replaces only the path guard and the GPU wait, because A1's guard refuses these seed folders and A1's GPU wait expects an idle GPU.
5. **Verdict on point estimates,** as in A1. Bootstrap robustness is reported, not judged.
7. **C2,** the end-to-end reproduction of A1's development people before the test pass, and **C3,** the truth self-check, are added.
8. **Only F5** is scored and reported. The other variants that `pipeline.py` computes internally are not tabulated.

## 12. Prediction (inference, written before any test scene existed)

Recall 0.95–0.99 and line members at least 0.98. False people 0.03–0.15 per image, the bar most at risk: fit F-real gave 0.143, and the development bootstrap spans 0.000–0.190. Median position error 0.05–0.07 m. The facing bars met with room: non-profile median under 15 deg and profile front/back at least 0.95.

The main risk is counter staff with the lower body hidden. In these seeds this includes vendors at the second stall, who may stand inside the picture. Each such misplacement costs a miss and a false person at once. A kill is unlikely (guess).
