# I09 second unseen-layout confirmation (A1 with fix 2, C07): frozen protocol

Path roots:
- Bare paths are relative to `phase3/i09_fix/`, the folder that holds this note.
- Paths that start with `phase1b/`, `phase1c/`, `phase2/` or `phase3/` are relative to the research folder `results/2026-10-17-beyond-vlm/`.
- `.superpowers/` is git-ignored. The test scenes are never committed.

## 0. Purpose, and what a result would and would not show

**Purpose.** I09 A1's frozen detector failed its first unseen-layout confirmation on false people alone: 44/182 = 0.242 per image against the 0.10 bar, with the other six success bars met (`phase2/i09/README.md` section 1; outcome "neither"). The fix step then chose fix 2 by a pre-registered rule on training seeds 110–115. Fix 2 is candidate C07: head re-lift, duplicate radius 0.30 m, A1's flat test, and dropping boxes too small for that test (`FIX_PROTOCOL.md`, `chosen_fix2.json`). It held on fit-validate (seeds 116–119) and on A1's development scenes.

The fix step itself flagged two limits:
- fit-validate shares camera–staff geometry templates with fit-train (`DIAGNOSIS.md` section 7);
- a fresh pre-registered confirmation on layouts nobody has seen would be needed before I09 is called a method.

This is that confirmation. A1's frozen detector with fix 2 is scored once on new layouts, with nothing re-tuned after this note.

**What a success would show.** That A1 with fix 2 meets A1's success bars on at least 140 layouts that no one has rendered or seen, with nothing re-tuned. It would not show:
- **New stall geometries.** The layouts come from the same 14 first-pass eval rows (3 situations: no_one, waiting_line, conversation_group), the same generator, renderer and asset library, and the `real` variant only. The generator very likely reuses its camera–staff geometry templates (`DIAGNOSIS.md` section 7), so this tests new layouts, not new stall geometries.
- **Other situations.** The audience rows, the held-out situations and the second-pass rows are not tested.
- **Placeholder avatars.** On those, fix 2 changes nothing: F-ph recall stays 513/585 (`fix_confirm.txt`).
- **Real photographs or sensor depth.** The head re-lift reads 5×5 depth at head keypoints, and depth here is exact rendered depth.
- **The pipeline with the detector in it.** That is the I09 live check, a separate step.

**Not blind.** The fix was designed after the first confirmation's published aggregate failure, including its published post hoc classes (`phase2/i09/README.md` Table 1.5). It was tuned on seeds 110–115 and checked on 116–119. The test seeds are new layouts, and nobody has seen them.

## Exposure (disclosed)

**What I read before writing:**
- Of the first confirmation (`phase2/i09/`): `confirm/FROZEN.md`, `DEVIATIONS_1.md`, `DEVIATIONS_2.md`, `ORDER.txt`, the code of `confirm/sidecar/infer_confirm.py`, and `README.md` sections 0–1, which hold the published aggregates only.
- Of the fix step (this folder):
  - in full: `FIX_PROTOCOL.md`, `chosen_fix2.json`, `fix_confirm.txt`, `fix/fixlib.py`, `fitlib.py` and `pathguard_fit.py`;
  - partly: `DIAGNOSIS.md` (its headings and section 7), the call sites in `fix/confirm_chosen.py`, the diff between `sidecar/infer_fit.py` and the first confirmation's driver, `raw_fit/meta.json`, and the field names and row counts of `confirm_rows.jsonl`.
- The render log, stdout and manifest of training seeds 110–119 (`phase3/seeds/realistic_train/`).
- A text search of the research folder, `src/`, `configs/`, `docs/` and `tests/` for mentions of seeds 213 and above. It found only mentions that seeds 213–219 were never rendered (REPORT, `phase2/i09/`, `phase3/metrics_v1/FACTCHECK.md`) and one mention of them as an option for another step (`phase3/confidence/README.md`, item 1).

**What I computed.** The sha256 of every file this note pins (section 3), a `sha256sum -c` of `FIX.sha256`, `CHOSEN2.sha256`, `RAW_FIT.sha256` and `MANIFEST_FIT.sha256` (all OK), and section 1.7's existence check.

**What I therefore know.** I know the fix step's results in full:
- fit-train is the selection data: C07 238/238 recall, 4/84 false;
- fit-validate: 164/164 recall, 2/56 false;
- development: 118/118 recall, 0 false;
- F-real is for information only: 39/39 recall, 1 false.

All 4 of C07's remaining fit-train false people are glass-booth staff. I also know the first confirmation's published aggregates, including Table 1.5's class counts:
- A, 16 displaced counter staff;
- B, 9 hidden-by-rule staff;
- C, 4 near hidden-by-rule staff;
- D, 15 far and small boxes.

**What I have not read.** No per-image or per-person row of the first confirmation: no `rows.jsonl`, `people_rows.jsonl`, `raw/` or `posthoc_false.txt`. I opened no file under `phase2/seeds/realistic/` and nothing of seeds 200–219.

**What this note chooses, and what it does not.** It chooses no part of the method: the method is the fix step's recorded choice. It does not choose the test seeds. It fixes the mechanics, the report, and the definitions the brief leaves open (section 11).

Timing:
- The addendum is dated and hashed before any test scene is rendered. Its file time must precede the start of the first generate command on any listed seed (check C5).

**1.2 Requirements for the seed list.** Each must hold, and C5 asserts each by code where it can:
- **Size.** At least 10 distinct seeds, so at least 140 layouts at 14 rows per seed. No seed is listed twice.
- **Allowed ranges.** Every seed is an integer outside these ranges:
  - 0–2: development, seen;
  - 3–99: I46's sealed set, never generated or opened;
  - 200–212: spent as test data by the first confirmation.

  The allowed seeds are therefore 213–219 and seeds from 220 up.
- **Never rendered or seen.** No layout of a listed seed has been rendered, drawn into a scene folder, replayed with `accepted_layouts`, opened or looked at by anyone, agent or person. At the addendum's time, a search of `~`, `/tmp` and `/dev/shm` finds no folder named `seed<k>` (with optional leading zeros) for any listed k. The addendum records the command, its time and its output.

**1.4 The addendum's format.** `CONFIRM2_SEEDS.md` holds:
- its date and time (+08:00);
- one line that starts `SEED_LIST:` and is followed by the seeds as comma-separated integers, in render order. This is the only line scripts parse;
- the sha256 of this note (`CONFIRM2_FROZEN.md`);
- section 1.2's existence search: command, time and output;
- a statement that every requirement of section 1.2 holds;
- which seeds, if any, are also given to another step.

The required core is the first ten seeds of `SEED_LIST` (section 1.8). Nothing else in the addendum changes anything in this note.

```
cd .-scenes && PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  TMPDIR=<research>/phase3/i09_fix/confirm2/render/tmp nice -n 10 \
  .venv/bin/python -m sgi.synth.realistic.generate --variant real --seed k \
  --out <research>/phase3/i09_fix/confirm2/seeds/seed<k> --only <the 14 ids>
```

The 14 ids are the first-pass eval rows of `src/sgi/synth/generate.py` (`_EVAL`), the same rows as the first confirmation and as seeds 110–119:

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
- It renders only the named rows.
- It never draws the second pass: audience, group_photo, lift_wait, the lift and the screen.

Only `eval_00`..`eval_13` are rendered. Nothing is written into `data/`. A layout is a (seed, row) pair with one image, the `real` variant; the `staffed` and `plain` variants are not rendered.

Render rules (they turn the first confirmation's departures into rules):
- **No trial command.** No generate command is run on a test seed except section 1.5's full-seed command. There is no one-scene trial and no partial `--only`.
- **Every command is logged.** Each generate command, re-runs included, gets one line in `confirm2/render/render_log.jsonl`:
  - seed, the exact command, start and end (ISO, +08:00), seconds and exit code;
  - the `nvidia-smi` and `free -m` readings before it, and any wait.

  Its stdout and stderr go to `confirm2/render/seed<k>.stdout.txt`. Any command missing from the log is a departure and must be reported in a deviation file.
- **Where it writes.** Only under `confirm2/seeds/` and `confirm2/render/`. Its own record of the scenes, if it keeps one, is not section 2's manifest.

**1.6 Order, clock and stop.**
- Seeds are rendered in `SEED_LIST` order, one generate command at a time.
- **Rendering wall time** is the summed wall-clock time of the generate commands, each from its start to its end. Waits for GPU or memory before a command starts are logged and not counted.
- **What decides the stop.** Time and render outcomes only. Nobody opens a detector output or views a test image before the stop.

**1.7 Nothing existed before this note.** At 19:17:59–19:18:03 +08:00:
- a search of `~`, `/tmp` and `/dev/shm` found no folder whose name is `seed` followed by any number from 213 up;
- `confirm2/` and any `CONFIRM2_*` file did not exist;
- no `sgi.synth` or Blender process was running.

**1.8 Complete seeds and the test set.**
- **Complete seed.** A seed is complete when all 14 scene folders exist under `confirm2/seeds/seed<k>/eval/`, each holding `scene.json`, `rgb.png`, `depth.npy`, the tour frames and `render_job.json`, and loading with `sgi.scene.load_scene`. Its render job must say variant `real`, seed k and renderer `blender`.
- **Test set.** Every complete listed seed at the stop.
- **Required.** Each of the first ten seeds of `SEED_LIST` is complete (140 layouts). Otherwise the outcome is "incomplete", and section 5's criteria are not judged.

## 2. The manifest (project venv)

`confirm2/manifest.py` writes `confirm2/manifest.json` and `confirm2/MANIFEST.sha256` once. It runs after `CONFIRM2_FROZEN.sha256` and `CONFIRM2_SEEDS.sha256` verify and after the render log records the stop. It decides completeness from the files, not from any status file.

**It asserts:**
- every scene folder passes `confirm2/pathguard2.py` (section 3) and `H.guard_scene`, and lies under `confirm2/seeds/seed<k>/eval/` with k in `SEED_LIST`;
- `confirm2/seeds/` holds no folder for a seed not in `SEED_LIST`;
- each complete seed holds exactly section 1.5's 14 ids under `eval/`, and nothing else;
- `examples/` is absent or empty;
- no file or folder name under `confirm2/seeds/` contains `group_photo`, `lift_wait`, `heldout`, `held_out` or `eval_16`–`eval_19` (case ignored);
- `render_job.json` gives variant `real`, the folder's seed and renderer `blender`;
- its thresholds equal `configs/default.yaml`'s, every key;
- `scene.json`'s scene id equals the folder name.

**Each entry holds:** seed, row number, row id, scene id, folder, rgb path, the sha256 of `rgb.png`, `depth.npy`, `scene.json` and `render_job.json`, and the render job's attempt and renderer version. Under `images` it also holds the five fields that `infer.run_pass` reads, as `manifest_fit.json` does: `scene_id` = `seed<k>/<scene id>`, `domain` = `"test"`, `set` = `"real"`, `rgb`, and the `sha256` of `rgb.png`. Raw lines are joined to entries by their rgb path.

## 3. The method: A1's frozen detector with fix 2 (C07), by import

**Base setting.** A1's frozen choice, from `phase1c/i09/a1/chosen.json` and `chosen_fix.json`, unchanged:
- RT-DETRv2-r50vd with score threshold τ 0.25;
- height band `wide`, 1.00–2.30 m on z_top, with A1's top-touch exception;
- depth push 0.20 m;
- A1's step-2 reject;
- facing variant F5 (shoulders and hips, plus the nose-against-ears profile rule, without the face rule).

**Fix 2 = C07 = HR + d030 + FLAT + TINY**, as specified in `FIX_PROTOCOL.md` section 3 and implemented in `fix/fixlib.py`. It has 4 parameters:
- **HR, head re-lift.**
  - On every candidate box not rejected at step 2, take the head keypoints (COCO 0–4) with confidence ≥ 0.30 whose 5×5 depth is valid. z_h is the nearest of their depths.
  - HR fires iff z_h − Z_med > 0.30 m. It then:
    - sets body_xy to the median world xy of the head keypoints with depth ≤ z_h + 0.30 m;
    - recomputes z_top with A1's formula at z_h;
    - recomputes the yaws with A1's `P.facing` at Z_ref = z_h.
  - The score, the box, the flat flag and `n_Q` are unchanged.
- **d030.** A1's world duplicate rule with radius 0.30 m instead of 0.50 m.
- **FLAT.** A1's flat test switched on: drop a box when at least 0.80 of the valid depth points on its even-row, even-column grid lie within 0.04 m of the torso band's plane, with `n_Q` ≥ 30. It is applied after the height band and before the push.
- **TINY.** In the same step, drop a box with `n_Q` < 30, one too small for the flat test.

**Per image.** The pipeline is `people = FX.run_all(raw, depth, cam, setting, "F5", ids=["C00", "C07"])`, with `setting, _ = F.frozen_setting()`, `depth = sc.observation.depth` and `cam = sc.observation.camera`, as `fix/confirm_chosen.py` calls it.
- `people["C07"]` is the method's output and the only one judged.
- `people["C00"]` is A1 frozen, reported beside it as a descriptive reference. `run_all` asserts that it equals `P.run_image` on every image.
- No other candidate is computed on test data.
- No situation label, role, group id, row or scene name is read before the people are produced. Truth and rows are used for scoring and tabulation only.

**Nothing re-tuned.** No constant, block, threshold, band, radius, facing variant or scorer setting may differ from the files pinned below. No other fix may be added after this note.

**Code by import.** Nothing is copied. Every script of the confirmation asserts these sha256 values at start and refuses to run on any mismatch. Importing `fixlib` imports `fitlib`, which asserts A1's and I03's pins in turn.

| File | sha256 | Used for |
|---|---|---|
| `fix/fixlib.py` | `e4219ec60d9035b14c27007a802151e3d87edfb5aea51dfe56b2d562bb545225` | `run_all`, `m1_variant`, `people_for`, `CANDIDATES`, `secondary`, `hidden_staff_points`, `m3_audit`, `posthoc_class`, `score_image`, `summary`, `wilson`, `poisson_exact` |
| `fitlib.py` | `08c7b04d25f9d23e657677efb789f01ef46812d326476d3e37009f4454d26897` | imported by fixlib: A1/I03 pins, `frozen_setting`; `load_fit_scene` in C2 only |
| `pathguard_fit.py` | `ee51dcbd2a4a0af49f428e8e2b166e600ec492bd755770382acb098dc7d0e664` | imported by fitlib; fit scenes in C2 only |
| `chosen_fix2.json` | `829409d7703dd7fea5a4e67c75755406eb124671cad56c9aa61a27e9888a8c61` | the choice and its checks; C2 targets |
| `chosen_fix2_precheck.json` | `047d9aa6301cbe0ae675e8d2b39412d11149a872da51eb4b9b74c8c6f4f1f31e` | the choice as hashed before its checks |
| `FIX_PROTOCOL.md` | `59654e7bf67d90827ba2fa1e74f23fbe96ce64497d5a2633a5cea9c63c4fa600` | the blocks' specification |
| `confirm_rows.jsonl` | `abb9751bdd0b0117353728713ad48dfba00557562f2be329f4b4c34e6eb2ebd9` | C2 targets (fit-validate rows) |
| `confirm_summary.json` | `f8366ba9db2d2230ebe752dbae36f9759b0ccbe0fe5fe11dc3f4061db96e7b18` | C2 targets |
| `manifest_fit.json` | `e597eb338303d54638fab3167df295ccce8230deb445059d6c1f18aaec1ee903` | C2 input |
| `raw_fit/rtdetr_v2_r50vd.jsonl` | `bff284afbe93ec1441fab86a4b5ea9f1b1d76944849d44421aa1ca64941517c3` | C2 input |
| `sidecar/infer_fit.py` | `7d5e30d973448fe8b39802c8248d8a1162909d8667659f4c709f7523a35cda95` | template of the new sidecar driver (C6) |
| `phase1c/i09/a1/pipeline.py` | `4826be88d6939f3f5e4d14375a2fc9698abd94973ca46243d6697e263e5d063a` | `candidates`, `run_image`, `facing`, `push`, constants |
| `phase1c/i09/a1/common.py` | `cdc33d742f6db5d09c3ad9a4557c1210ef10f8909484cee101ce983f9f8b2f53` | `score_scene`, `view_angle_deg`, `view_class`, `facing_summary`, `boot_ci`, `p90`, `thresholds_for` |
| `phase1b/i03/people_scorer.py` | `e12776eaddac1de0af75b58f8a272cd0a13332596da626bcc81b085f5900d685` | `score_people`, `pool`, `wilson`, `exempt_points`, `truth_info`, `default_thresholds`, `check_thresholds`, `role_classes`, `FreshScene` |
| `phase1c/i09/a1/sidecar/infer.py` | `bee1322fd692bf0e85e820871413d486fda04d5770f72b5d3d75e18266709484` | `run_pass`, the model functions, `BOX_MIN`, `POSE_MIN`, `SNAP` |
| `phase1c/i09/a1/chosen.json` | `19b1e16340d810e75f8f772eeeabeaf4b788e09614e74d78d3afc206531119ff` | the base setting |
| `phase1c/i09/a1/chosen_fix.json` | `fe8177165773f7208d417227c358adae7d04326bd29cb6c75d73ed87b3f83f52` | facing F5 |
| `phase1c/i09/a1/ADDENDUM_FIX.md` | `3a7d185526e3d9801bd2adcdaf529bfbe744c1a636dc8e03d0f53de45d2e208c` | the facing fix's record |
| `phase1c/i09/FROZEN.md` | `f315e96c1145ad9f2c86128d16711bb5305690d99f5d8513c0177290c474df15` | A1's protocol and bars |
| `phase1c/i09/a1/raw_eval/rtdetr_v2_r50vd.jsonl` | `8a03b82ca89bb82bdd0b10a8e56944b8562f3a580967087646e408dd4862a97d` | C2 input (development) |
| `phase1c/i09/a1/eval_people.json` | `ffa595453c94b444c8dd030d053d37ac103441a7e74f07e3f172836b3d0c24b3` | C2 target (C00 people) |
| `phase1c/i09/a1/manifest_eval.json` | `6c321c3800d0b76632da0711a2385dce4915a28e9cd1f563ff3f8a5782be0b22` | C2 input |
| `configs/default.yaml` | `c651f4c484fff15d33290e13f28bc53c9221c8edd79c0b7a24f24a128098b4fc` | thresholds |

Further assertions:
- `chosen_fix2.json` gives `chosen` = `"C07"`, `chosen_blocks` = `["HR", "d030", "FLAT", "TINY"]`, `code_sha256["fixlib.py"]` equal to the pin above, `constants.GAP_M` = 0.3 and `constants.D030_M` = 0.3, and `checks.outcome` = `"confirmed"`.
- `FX.CANDIDATES["C07"]` is `("HR", "d030", "FLAT", "TINY")`, and `FX.GAP_M`, `FX.D030_M` and `FX.STAFF_EXEMPT_M` are 0.30, 0.30 and 0.6.
- `src/` and `configs/` are identical to commit f1a488a: `git diff --quiet f1a488a -- src configs` succeeds, and there are no untracked files under them. This pins `sgi.geometry`, `sgi.scene`, `sgi.gt.derive`, `sgi.run._matched_people` and the generator.

**The path guard.** `confirm2/pathguard2.py` uses the standard library only, so both environments load it, and follows `pathguard_fit.py`. It accepts a path only if all of these hold for the path as written and as resolved:
- it lies under `<research>/phase3/i09_fix/confirm2/seeds/seed<k>/eval/<scene>/`;
- it holds exactly one `seed<k>` component, with k in `SEED_LIST` (read from the verified addendum);
- it contains none of `keep/heldout`, `s18`, `s20_held`, `eval_16`–`eval_19`, `group_photo`, `lift_wait`, `heldout`, `held_out`, `phase2/seeds` or `realistic_train` (case ignored).

- It is `sidecar/infer_fit.py` with only these changes:
  - the constants: paths, manifest name, expected image count;
  - the guard module: `pathguard2`;
  - the hash checks of this note, the addendum and `MANIFEST.sha256`.
- It imports A1's `infer.py` by path, with bytecode writing off, and calls `infer.run_pass("eval", confirm2/manifest.json, confirm2/raw/, ("rtdetr_v2_r50vd",))` unchanged:
  - RT-DETRv2 from snapshot `282494075698cab9faa1096ae26856890030c817`, keeping every box with score ≥ 0.05;
  - ViTPose-base-simple from snapshot `a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45`, on every box with score ≥ 0.10, as one batch per image;
  - fp32, eval mode, seed 0, cudnn deterministic, benchmark off, TF32 off;
  - no warm-up.
- It replaces only two functions:
  - `infer.guard`: `pathguard2` plus manifest membership and the sha256 of each image;
  - `infer.wait_for_gpu`: section 10's rule.
- D-FINE is not loaded. Nothing is installed.

**Project side.** The project venv `.venv/bin/python`, with `PYTHONPATH` empty, torch-free, bytecode writing off, 2 BLAS threads and `nice -n 10`.

## 4. The scorer and definitions

**Loading.** Test scenes are loaded as I03 loaded its fresh seeds: `pathguard2.check(folder)`, then `H.guard_scene(folder)`, then `PS.FreshScene(load_scene(folder))`. The sha256 of `rgb.png`, `depth.npy` and `scene.json` is checked against the manifest on every load.

**Thresholds.** `th = PS.default_thresholds()`, which is `configs/default.yaml`, and `PS.check_thresholds(th)` is asserted: person radius 0.25 m, 3 visibility samples, 0.3 m visibility depth tolerance, 0.05 m grid.

**Primary scorer, the only one judged.** I03's scorer, unchanged: `C.score_scene(people, sc, th, "F5")`, which calls `PS.score_people(points, sc, th, exempt_xy=PS.exempt_points(sc.truth), facing="full")`, pooled with `PS.pool`. Visibility follows the package's rule. In the code this runs through `FX.score_image`, which calls `C.score_scene` unchanged.

**Secondary scorer variant, reported and never judged.** As in `FIX_PROTOCOL.md` section 8 and `FX.secondary`:
- A detection the primary scorer counts as false is counted as exempt instead when it lies within 0.6 m (world xy) of a true person who meets three conditions: they have a role, they fail `is_visible`, and their marked point (x, y, 1.1 m) projects into the picture (`FX.hidden_staff_points`).
- Recall, matching and everything else are unchanged.
- It is reported for C07 and C00 as false and exempt counts, false per image, and the number of detections it reclassifies.

**Definitions** (unchanged from the first confirmation):
- **Visible:** passes `sgi.gt.derive.is_visible` under th.
- **Matching:** `sgi.run._matched_people`: greedy, nearest first, one to one, within 0.5 m.
- **Recall:** matched / visible.
- **False person:** an unmatched detection that is not within 0.5 m of the anchor of a target with no person. Exempt detections are counted apart. **False people per image** = pooled false / images.
- **Position error:** the xy distance of each matched pair. The median is taken over all matched people.
- **View class:** from the angle between the true yaw and the bearing from the true person to the camera's xy (`C.view_angle_deg`, `C.view_class`): frontal < 45, oblique 45 to under 75, profile 75–105 (both ends in), back > 105 deg. **Non-profile** = frontal + oblique + back.
- **Facing error:** |wrap(yaw_det − yaw_true)| in degrees, for matched people.
  - **Within 45** means error ≤ 45 deg. **Front/back correct** means error < 90 deg.
  - An unknown yaw counts as wrong, and as 180 deg in medians.
  - Coverage (matched people with a known yaw) is reported.
- **Role class** (`PS.role_classes`): staff (has a role); line (group at a target of a waiting_line episode); group (group at a target of a conversation_group episode); other.
- **Line members:** role class "line".

## 5. Criteria: A1's bars verbatim, judged once on the pooled test set, for C07 only

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
| K3 | facing within 45 deg below 0.70 | all matched people, unknown wrong |

**How it is judged** (the same readings as the first confirmation):
- Bars are judged on the point estimates, as in A1. Intervals are reported and not judged.
- "At least" means ≥, "at most" ≤, "below" <, "more than" >.
- Only C07, scored by the primary scorer, is judged. C00 and the secondary variant are descriptive.

**Outcomes:**
- **Success:** S1–S7 all met and no kill clause. Fix 2 is confirmed on unseen layouts, within section 0's scope.
- **Kill:** any of K1–K3. REPORT I09's consequence applies: "keep it only as a recall cross-check".
- **Neither:** any other outcome. Fix 2 is not confirmed. No further fix may be scored on these seeds; they are spent.
- **Incomplete:** section 1.8's requirement is not met.

**Not carried over from A1:**
- the runtime bar (not judged again);
- the kill clause "more than 10 points below the best VLM" and its A0-pair population, because no VLM runs on these scenes;
- A2, which is not part of this step.

## 6. The report

**Main table** (pooled test set, C07). For each bar:
- the value with its n;
- a Wilson 95% interval for proportions (`PS.wilson`);
- an exact Poisson (Garwood) 95% interval for false people per image (`FX.poisson_exact`), with the share of images with at least one false person and its Wilson interval;
- a layout-cluster bootstrap 95% interval. Clusters are `seed<k>/<scene id>` layouts, each one image with all its people. There are 2000 draws with `random.Random(0)`, percentile intervals, through `C.boot_ci`, and the statistic is computed on each draw as on the full set. A draw where the statistic is undefined is skipped, as `C.boot_ci` does;
- a seed-cluster bootstrap (whole seeds resampled, 2000 draws, seed 0), as a sensitivity;
- section 7's comparators beside it;
- C00 on the same test images, as a descriptive reference column;
- the secondary variant's S3 and K2 values (not judged).

The verdict line follows. A descriptive line also says, for each bar, whether its whole layout-bootstrap interval lies on the passing side: the lower end ≥ the bar for "at least" bars, the upper end ≤ the bar for "at most" bars.

**Also reported, for C07 and C00:**
- recall by role (line, group, staff, other) and by row, pooled over seeds, each as k/n with Wilson 95%;
- false people per image by row, and per seed;
- facing by view class: n, known yaw, within 45, median, front/back;
- coverage, and merged neighbour pairs;
- counts of images, layouts, visible people (and staff with the lower body hidden), hidden people and detections kept;
- misses by class (staff behind counters, other staff, non-staff) and by role. Each miss is listed with its best raw box: the highest-scoring box with score ≥ 0.05 whose closed rectangle contains the projection of (x, y, 1.1 m), or None;
- false people, each with its detector score, box height in pixels, distance to the nearest visible person, whether it is within 0.5 m of a hidden person, and its class A–D under `FX.posthoc_class`. These classes are defined before any test data, so they are pre-registered and descriptive, not post hoc. Exempt detections are listed apart;
- the secondary variant: false, exempt, reclassified, false per image;
- the M3 audit (`FX.m3_audit`): C00 detections within 0.6 m of hidden-by-rule staff whose mark is in the picture, and how many C07 no longer outputs;
- block activity for C07: the number of kept boxes HR fired on, and how many of them match a true person. Also the number of boxes dropped by FLAT and by TINY, and how many of those C00 had matched. All counts are taken from the candidate lists, without changing the method;
- a paired comparison of C07 and C00 per image: the images where C07 has fewer, the same or more false people, and the people matched by one and not the other;
- the required core (the first ten listed seeds) beside the rest of the test set. If the list holds both 213–219 and seeds from 220 up, those two subsets are shown too. All descriptive.

**Runtime** (not judged): the median and p90 of `t_det_s + t_pose_s` per image from the detector pass, and the project side's median and p90 per image for `FX.run_all` with C00 and C07. The GPU is shared, and this is not A1's deployed timing path; both facts are stated.

Any analysis not listed here is labelled post hoc.

## 7. Comparators (recorded; not recomputed, except where C2 says)

Sources:
- `chosen_fix2.json` (sha256 above), in particular `fit_train`, `a1_fit_train` and `checks`;
- `fix_confirm.txt`;
- the published Table 1.1 of `phase2/i09/README.md` (sha256 `6045e5889ac03fd6b09da04d8d14e0c6ab8abbe2fda246767c41dd672367c638`) for A1 on the first confirmation.

| Measure | C07, development E-real (42) | C07, fit-validate (56) | C07, fit-train (84; selection data) | A1 (C00), fit-validate (56) | A1, first confirmation (182; published) |
|---|---|---|---|---|---|
| Recall | 118/118 | 164/164 | 238/238 | 157/164 = 0.957 | 510/520 = 0.981 |
| Line members | 48/48 | 64/64 | 96/96 | 62/64 | 207/208 |
| Staff | 16/16 | 28/28 | 34/34 | 23/28 | 69/78 |
| False people (per image) | 0 (0.000) | 2 (0.036; exact Poisson 0.004–0.129) | 4 (0.048) | 12 (0.214) | 44 (0.242) |
| Secondary variant: false (per image) | 0 (0.000) | 1 (0.018) | 0 (0.000) | 11 (0.196) | not computed |
| Median position error | 0.053 m | 0.061 m | 0.060 m | 0.060 m | 0.061 m |
| Non-profile within 45 (median) | 61/61 (6.6 deg) | 93/95 (12.8 deg) | 132/136 (13.1 deg) | 89/90 (10.7 deg) | 278/287 (10.3 deg) |
| Profile front/back | 57/57 | 69/69 | 102/102 | 67/67 | 223/223 |

K3's population (all matched people, within 45) was not recorded by the fix step. C2 computes it for the development and fit-validate sets, and the report prints it beside the test value, labelled "computed at C2".

## 8. Checks before the detector pass

- **C1. Code identity.** Section 3's hashes, the `chosen_fix2.json` assertions and the `src/`/`configs/` pin, asserted by every script.
- **C2. Reproduction of the fix step** (no test data). The confirmation's own code, through `FX.run_all(..., ids=["C00", "C07"])` and `FX.score_image(..., with_classes=True)`, must reproduce two recorded results:
  - **Fit-validate.** The 56 fit-validate entries of `manifest_fit.json` (seeds 116–119), loaded with `F.load_fit_scene`, on `raw_fit/rtdetr_v2_r50vd.jsonl` (verified by `RAW_FIT.sha256`):
    - every field of every fit-validate line of `confirm_rows.jsonl`, for both C00 and C07: counts exactly, floats within 1e-9;
    - the pooled summaries of `chosen_fix2.json` `checks.fit_validate` (`chosen` and `a1`): counts exactly, medians within 1e-12.
  - **Development.** A1's 42 E-real scenes, read through the harness (`H.distinct_scenes`, `H.scene`, `C.thresholds_for(runs)`), on A1's recorded `raw_eval`:
    - the pooled summaries of `checks.dev_e_real` for both candidates, exactly;
    - C00's people equal to `a1/eval_people.json` within 1e-9.
- **C3. Truth self-check** on the test scenes, reading truth only. The true visible people, given as detections with their true yaws, must score recall 1, 0 false and every person within 45 deg on every test scene.
- **C4.** The manifest assertions of section 2.
- **C6. The driver.** Before the pass, `diff sidecar/infer_fit.py confirm2/sidecar/infer_confirm2.py` is saved to `confirm2/sidecar/driver_diff.txt`. It must show only section 3's permitted changes. Then `infer_confirm2.py --preflight` (no model, no image read) passes and is logged.

A failed check stops the step. Its fix is logged in a hashed `confirm2/DEVIATIONS_<n>.md` before work goes on, and no fix may change the method, the scorer, the thresholds, the criteria or the seed list.

## 9. Order of work and failure rules

0. This note, then `CONFIRM2_FROZEN.sha256`; both made read-only.
3. C1 and C2, at any time before step 5. C2 reads no test data and may run before step 1.
4. `confirm2/manifest.json` and `MANIFEST.sha256` (C4), then C3.
5. C6, then one detector pass: the sidecar over the manifest, once, giving `confirm2/raw/rtdetr_v2_r50vd.jsonl` and `raw/meta.json`, hashed into `confirm2/RAW.sha256`. The torch environment's `pip freeze` is recorded before and after and must not change.
6. Scoring once: `confirm2/score.py` gives `tables.txt`, `rows.jsonl`, `people_rows.jsonl` and `summary.json`, hashed into `confirm2/SCORE.sha256`.
8. `confirm2/README.md`.

`confirm2/ORDER.txt` logs each step's timestamps as it happens. Every script verifies the hashes of everything it reads.

**Cost and time** (information). Zero model-provider calls and zero cost. One GPU job at a time. Renders take about 67–133 min for ten seeds at the rates of section 1.6, capped at 3 h. The detector pass takes about 1 min and scoring about 1 min.

**Failure rules:**
- **The detector pass.** If it dies before it has written every image, for any technical reason, nothing is scored. The partial output is moved to `confirm2/raw_failed_<n>/` and kept, and the pass is run again from the start, unchanged. The overlapping images are compared and the comparison reported.
- **The scoring script.** If it fails before it writes any metric, the bug is fixed (logged with its diff in a hashed `confirm2/DEVIATIONS_<n>.md`) and the script is run again. Once a metric has been written there is no re-run, except to fix a bug a verifier finds; then both values are reported.
- **Viewing test images.** Nobody views a test image until scoring is finished. Any later look, for diagnosis, is post hoc.

## 10. Resources and safety

**Shared machine:**
- Before each GPU job (each generate command and the detector pass), `nvidia-smi` must show at least 3,000 MiB free.
- Before any heavy job, `free -m` must show at least 3,000 MiB available.
- Otherwise the job waits, polling every 5 min, and the wait is logged.

**No calls and no installs:**
- No model-provider or network call and no download; Hugging Face runs offline.
- Nothing is installed in either environment. The torch sidecar is run with `-I` and is read-only.

**Excluded data:**
- No held-out situation, scene or run is read or instantiated: no `keep/heldout`, `s18`, `s20_held`, `eval_16`–`eval_19`, `group_photo` or `lift_wait`.
- No seed 3–99 is generated or opened.
- Nothing of seeds 200–219 other than the listed seeds is touched. Nothing of `phase2/seeds/realistic/` is opened.
- No per-image or per-person row of the first confirmation is read.
- Recorded runs and development scenes (C2 only) are read through the step-0 harness alone.

## 11. Choices this note makes where the brief leaves room

1. **Where the test scenes go.** `confirm2/seeds/`, inside this folder: apart from the training seeds' folder, and outside every earlier phase folder. Neither path guard can then accept the other's scenes.
3. **The clock.** 3 h of summed generate-command time, as in the first confirmation, with waits not counted. A command running at the mark may finish.
5. **C00 is a descriptive reference** on the test images, so that the fix's effect can be read on the same layouts. Only C07 is judged.
7. **Classes A–D are pre-registered** as descriptive, using `FX.posthoc_class`.
8. **Exact Poisson intervals** are added for false people per image, as the fix step used them, beside the bootstraps of the first confirmation.
9. **C2 reproduces the fix step's checks** (fit-validate and development), and C6 bounds the new driver by a diff against the fix step's driver.

## 12. Prediction (inference, written before any test seed was named)

**Bars I expect to be met with room** (S1, S2, S4–S7):
- recall 0.97–1.00, with line members at least 0.98 and staff at least 0.95;
- median position error 0.055–0.065 m;
- non-profile facing median 9–15 deg and within 45 at least 0.95;
- profile front/back at least 0.97.

**False people: 0.03–0.11 per image, with a central guess near 0.06.** S3 is again the bar most at risk. I guess it is met with probability about two in three. The reasoning:
- **Glass-booth staff are untouched by design.** In the first confirmation, classes B and C (detections near hidden-by-rule staff) were 13 detections in 182 images, 0.071 per image, and fix 2 does not target them. All of C07's remaining false people on fit-train were of this kind, and one of its two on fit-validate (the one the secondary variant reclassifies).
- **The other classes were cleared on the fit data.** Under C07, no false person of class A (displaced staff) or class D (far, small boxes) remains on either fit set, where A1 had A 9 + 5 and D 4 + 5 (`chosen_fix2.json` and `fix_confirm.txt`). They may recur at a low rate on new stall dressings.

**Secondary variant:** 0.01–0.05 per image.

**Kill:** very unlikely.
