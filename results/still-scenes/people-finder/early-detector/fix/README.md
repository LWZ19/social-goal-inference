# Phase 3: I09 detector fix and the frozen second confirmation (2026-10-18)

- **Render.** Training seeds 110–119 were rendered as realistic layouts of the same 14 rows.
- **Diagnosis.** A1's frozen detector ran on them, and the false people and misses on seeds 110–115 were traced to their causes.
- **Fix.** A fix was chosen by a rule frozen before any candidate was scored, then checked on the held-back seeds 116–119 and on A1's development scenes.

How it was made:

  Where an agent and the verifier differ, section 4 says so, and this page uses the verifier's value where it settles the point.
- **Data and roles.**
  - Fit-train: seeds 110–115, rows `real_eval_00`–`13` (84 images). The diagnosis, the what-ifs and the selection rule used these.
  - Fit-validate: seeds 116–119, the same rows (56 images), held back. The detector ran on them, but nobody scored or viewed them before the choice was hashed (the verifier confirmed this from file access times and every tool call).
  - A1's 42 development E-real scenes, read through the step-0 harness: the second check of the chosen fix. It was not blind (section 4).
  - A1's fit data, F-real (21 worked examples) and F-ph (268 placeholder scenes): information only.
- **Protocols.**
  - `FIX_PROTOCOL.md` (sha256 `59654e7b…a600`). It was born and hashed at 19:06:13 (+08:00), before `fix/` existed (19:08:03) and before any candidate was scored. It fixes 11 candidates, the selection rule, the checks and a secondary scorer.
  - The choice, written by code and hashed at 19:10:11 (`chosen_fix2_precheck.json`, sha256 `047d9aa6…`, in `CHOSEN2.sha256`), before the checks ran (19:10:24–19:10:46).
  - `CONFIRM2_FROZEN.md` (sha256 `69aa2ccb…3b23`), saved at 19:23:48 and hashed at 19:24:00; both files are read-only. At 19:17:59–19:18:03 a search found no folder named seed213 or higher and no `confirm2/`.
- **Not blind.** The fix was designed after the first confirmation's published aggregate failure, including its post hoc classes (`results/still-scenes/people-finder/early-detector/phase2/README.md` Table 1.5). The fix protocol was written knowing the diagnosis and its what-ifs, which had already scored similar rules on fit-train, on A1's fit data and on the development scenes.
- **Environments.**
  - Detector and pose: the torch sidecar, used read-only (`-I`, `HF_HUB_OFFLINE=1`). A1's `infer.run_pass` was imported by path, with only the path guard and the GPU wait replaced, and only the pinned snapshots were loaded (RT-DETRv2 `2824940…`, ViTPose `a93ac0c…`). The sidecar's `pip freeze` (160 lines, sha256 `20f344bf…`, the same as at the first confirmation) was unchanged.
  - Everything else: the torch-free project venv, with A1's and I03's code imported and pinned by sha256, never copied.
- **No held-out data.** No held-out situation, scene or run was read or instantiated, and no seed 3–99 was generated or opened. The one breach of the test-data rules is the protocol author's text search, which opened files of the spent seeds 200–212 but printed file names only (section 4).
- **At writing** (19:57 on 18 Oct), every hash file listed below verified with `sha256sum -c`, and `confirm2/` did not exist.

Paths below are relative to this folder (`results/still-scenes/people-finder/early-detector/fix/` in the research folder).
- `results/still-scenes/scene-seeds/phase3/realistic_train/`: the renders `seed110`–`seed119` (589 MB, 1,260 PNG files), `manifest.json` (sha256 `ff4fefa7…`), `render_log.jsonl` and `seed<k>.stdout.txt`.
- Diagnosis:
  - note: `DIAGNOSIS.md`;
  - scripts: `pathguard_fit.py`, `fitlib.py`, `manifest_fit.py`, `sidecar/infer_fit.py`, `score_fit.py`, `diagnose.py`, `whatif.py`, `figs.py`;
  - data: `manifest_fit.json`, `raw_fit/`, `fit_train_*.jsonl`, `fit_train_summary.json`, `diag_*.jsonl`, `sidecar_pass.log`, `pip_freeze_before.txt`, `pip_freeze_after.txt`;
  - tables and figures: `diagnose_tables.txt`, `whatif.txt`, `figs/` (14 images);
  - hashes: `MANIFEST_FIT.sha256`, `RAW_FIT.sha256`, `DIAG.sha256`.
- Fix:
  - protocol: `FIX_PROTOCOL.md` (`FIX_PROTOCOL.sha256`);
  - code: `fix/fixlib.py`, `fix/select_candidates.py`, `fix/confirm_chosen.py`;
  - the choice: `chosen_fix2.json`, `chosen_fix2_precheck.json` (`CHOSEN2.sha256`);
  - results: `fix_select.txt`, `fix_confirm.txt`, `fix_tables.txt`, `select_summary.json`, `select_rows.jsonl`, `confirm_summary.json`, `confirm_rows.jsonl`, `select_run.log`, `confirm_run.log`;
  - post hoc: `fix_posthoc/`; hashes: `FIX.sha256`.
- Second confirmation: `CONFIRM2_FROZEN.md` and `CONFIRM2_FROZEN.sha256` (read-only). `confirm2/` does not exist yet.
- Verifier: `verify/` (`vlib.py`, `vscore.py`, `v_compare.txt`, `v_summary.json`, `v_rows.jsonl`, `sidecar_verify.py` and `.log`, `raw_seed116/`, `vposthoc.py`, `v_posthoc.txt`, `vmed.py`, `v_medians.txt`, `vc2.py`, `v_c2.txt`, `audit_calls.py` and `.txt`, `calls_*.jsonl`, `VERIFY.sha256`).

## 0. At a glance

This step has no pass/fail verdict of its own. It reports a diagnosis, a fix with its pre-registered checks, and a frozen protocol.

| Step | Outcome | One-line reading (inference) |
|---|---|---|
| Diagnosis on fit-train (A1 frozen, 84 images) | Recall 226/238 = 0.950; false people 17/84 = **0.202** per image. Four mechanisms explain all 17 false people and all 12 misses | The training renders reproduce the confirmation's failure (0.242), so it could be studied without test data. |
| Fix 2 (C07), chosen by the frozen rule | Fit-train 238/238 and 4/84 = 0.048. Fit-validate (held back): recall 164/164 [0.977, 1.000], false **2/56 = 0.036** [0.004, 0.129], against A1's 12/56 = 0.214. Development: 118/118 and 0 false (A1 116/118 and 3). Outcome **confirmed** | The fix clears the displaced-staff, line-merge and far-box failures on these seeds. What is left sits beside staff whom the visibility rule calls hidden. |
| Verifier | Every number reproduced (0 differences); the detector re-run is bit-identical. Two integrity findings | The development check was not blind, so "confirmed" rests on fit-validate. A text search opened spent test-seed files, printing names only. |
| I09 as a method | Unchanged: not yet a method | It needs the second confirmation. |

Shared cautions:
- **Training seeds with shared templates.** Fit-validate is new seeds of the same 14 rows, and the generator repeats its camera–staff geometry across the seeds of a row (section 1). So fit-validate confirms less than its 56 images suggest. The second confirmation would also test new layouts of the same rows, not new stall geometries.
- **Not blind.** The fix was designed after the first confirmation's published failure, and its development check was known to pass before the protocol was written (section 4).
- **Exact rendered depth only.** The `real` variant, the same renderer and asset library. The head re-lift reads 5×5 depth at head keypoints and is untested on sensor depth.
- **Post hoc** analyses are labelled as such; none changes an outcome.

---

## 1. Rendering and diagnosis (fit-train)

### Rendering (`results/still-scenes/scene-seeds/phase3/realistic_train/`)

- **Command.** For each seed k in 110–119: `python -m sgi.synth.realistic.generate --variant real --seed k --out phase3/seeds/realistic_train/seed<k> --only <14 ids>`.
  - The ids are `real_eval_00`–`13` (no_one 6 rows, waiting_line 5, conversation_group 3), the same rows as the first confirmation.
  - The generator replays the first pass only.
- **What happened.** 10 commands from 17:22:38 to 18:30:33, all exit 0, 4,075.6 s in total (1.13 h), with no trial command. Beforehand the GPU was idle and the folder did not exist.
- **The manifest** (`manifest.json`, sha256 `ff4fefa7…23b2`, written at 18:30:51 from the files on disk):
  - 10 seeds with 14 ids each, variant `real`, no `examples/` folder;
  - no name contains group_photo, lift_wait, heldout, held_out or eval_16–19.

### A1's frozen detector on fit-train

- **Detector pass.** One sidecar pass over all 140 images: 2,234 boxes (score ≥ 0.05) and 790 pose sets in 17 s.
- **Scoring, fit-train only.**
  - Method: A1's frozen setting (RT-DETRv2 at τ 0.25, band 1.00–2.30 m, flat test off, duplicate radius 0.50 m, push 0.20 m, facing F5). Every image was asserted equal to `P.run_image`.
  - Scorer: I03's, unchanged (facing "full", the anchors of targets without a person as exempt points, default thresholds).

Table 1.1. A1 frozen on fit-train, beside its earlier sets.

| | Fit-train, seeds 110–115 (84) | A1 development E-real (42) | First confirmation (182, published) |
|---|---|---|---|
| Recall | 226/238 = 0.950 | 116/118 = 0.983 | 510/520 = 0.981 |
| Line members | 92/96 | 48/48 | 207/208 |
| Counter staff | 26/34 | 14/16 | 69/78 |
| False people per image | **17/84 = 0.202** | 3/42 = 0.071 | **44/182 = 0.242** |
| Exempt detections | 5 | 0 | 6 |
| Median position error | 0.059 m | 0.053 m | 0.061 m |
| Non-profile within 45 deg | 125/128 | 59/59 | 278/287 |
| Profile front/back | 98/98 | 57/57 | 223/223 |

- **By the first confirmation's post hoc classes** (its rules reused unchanged): A 9 (8 paired with a staff miss, 1 second detection), B 4, C 0, D 4.
- **Misses:** 8 counter staff and 4 line members.

### Four mechanisms (`DIAGNOSIS.md`, `diagnose_tables.txt`)

Table 1.2. What explains the 17 false people and 12 misses on fit-train.

| Mechanism | False | Exempt | Misses | First confirmation's class (published aggregate) |
|---|---|---|---|---|
| M1. Occluded torso: the depth median lands on an object in front of a counter staff member | 9 (8 paired with a staff miss, 1 second detection) | 0 | 8 staff | A: 16 |
| M2. The 0.50 m duplicate radius drops a line neighbour | 0 | 0 | 4 line members | the 1 line miss |
| M3. Staff behind glass: the rendered depth sees the glass, the RGB sees the person, and the visibility rule calls the person hidden | 4 | 5 | 0 | B: 9; C: 4 |
| M4. Small far boxes on pictures of people | 4 | 0 | 0 | D: 15 |

**M1, occluded torso** (`figs/a_110_06.png`, `a_110_02.png`, `a_114_09.png`).
- **The cause.** A1 takes the plain median depth over the shoulder–hip hull. ViTPose places the hips behind the counter at confidence 0.41–0.84, so the hull is used, and 64–98% of it lies on an espresso machine, a till or the counter end.
- **The effect.** The median lands 0.76–0.77 m in front of the staff member. The displacement is 0.82–0.85 m along the camera ray, and the 0.20 m push leaves 0.62–0.65 m: the confirmation's "0.63–0.66 m off".
- **The signal.** The head is above the occluder, at the right depth. The nearest head keypoint lies 0.60–0.64 m behind the torso median for all 9 class-A boxes, against −0.20 to +0.15 m for all 226 matched boxes.
- **The second detection.** It is 0.76 m from the matched box in the world, more than the 0.50 m radius, yet lies wholly inside it in the image (noses 1.7 px apart).
- Whether a staff member is displaced turns on the dressing and the avatar's width, not on the camera geometry alone.

**M2, line merges** (`figs/l_110_09.png`).
- Four line members in profile (boxes scoring 0.93) stand 0.50–0.53 m apart. Their lifted positions are 0.45–0.48 m apart, so the second box is dropped as a duplicate, though the boxes do not overlap.
- Image overlap separates the cases on fit-train: every same-person pair had containment ≥ 0.83, and every pair of different people ≤ 0.34. World distance does not: the same person reached 0.76 m, and different people came within 0.45 m.

**M3, staff behind glass** (`figs/b_115_02.png`, `b_110_09.png`).
- All 9 fit-train people who fail `is_visible` with their 1.1 m mark in the picture are counter staff in glass booths.
- Every visibility sample hits the glass (5–7 per person) or the booth counter, 0.38–0.87 m in front, because the rendered depth treats glass as opaque. The RGB shows the person.
- The detector boxes them (scores 0.79–0.88) and the lift lands on the glass, 0.21–0.50 m from the labelled point. Whether the detection counts as false (4) or exempt (5) depends on whether that glass point happens to lie within 0.5 m of a target's anchor.
- This is a scorer question. (inference) A real depth camera often returns no depth, or the depth behind, on clear glass.

**M4, small far boxes** (`figs/d_*.png`).
- 4 boxes, 9–34 px tall, scoring 0.26–0.29, 3.3–4.1 m from everyone: two person pictograms on posters and two figures seen through windows.
- Depth inside each box equals the depth around it (flat share 1.00 where it can be computed), and each box floats 1.1–1.9 m above the floor with nothing nearer below.
- Here they give 0.048 per image. At the confirmation, class D was 15 boxes (0.082 per image), scoring up to 0.53.

**Facing** is not a problem: no what-if moved it.

### The camera–staff geometry repeats across seeds

- In the waiting-line rows, the staff member's projected position (u, v, Z) is identical across seeds. The seed changes the room, the avatars and the stall dressing: an espresso counter, a counter with a till, or a glass booth.
- The same geometry then gives M1 with a machine, M3 with glass, or a clean match. The 9 M1 cases come from 5 geometries, and the 9 M3 cases from 5.
- (inference) Fit-validate and the test seeds very likely share these templates.

### Indicative what-ifs (`whatif.txt`; not a choice)

Table 1.3. Selected what-ifs, each scored with I03's scorer unchanged (recall; false people).

| What-if | Fit-train (84) | A1 F-real (21) | A1 F-ph (268) | A1 development (42) |
|---|---|---|---|---|
| A1 as frozen | 226/238; 17 | 36/39; 3 | 513/585; 0 | 116/118; 3 |
| 1a, re-lift occluded torsos from the head | 234/238; 8 | 39/39; 1 | unchanged | 118/118; 0 |
| 2a, image-space duplicates only | 230/238; 16 | 36/39; **5** | unchanged | 115/118; 3 |
| 2b, duplicate radius 0.30 m | 230/238; 17 | 36/39; **5** | unchanged | 116/118; 3 |
| 4c, score threshold 0.35 | 226/238; 13 | unchanged | **483/585** | **115/118**; 3 |
| 1a + 2a + A1's flat test | 238/238; 5 | 39/39; 1 | unchanged | 118/118; 0 |

- Each duplicate rule alone brings back a ghost box behind a booth's frosted side glass in F-real, unless the flat test is also on.
- These what-ifs scored the development scenes before the fix protocol was written (section 4).

---

## 2. The fix (fix 2 = C07)

### The protocol (`FIX_PROTOCOL.md`)

- **What a candidate may change.** Every candidate is a post-processing or lift change on A1's raw detector and pose outputs, which are reused, never recomputed (no GPU job). No block reads a situation label, role, group id, scene name or truth.

Table 2.1. Building blocks.

| Block | Rule | Parameters | Aimed at |
|---|---|---|---|
| HR, head re-lift | Head keypoints (nose, eyes, ears) with confidence ≥ 0.30 and valid 5×5 depth; z_h is the nearest. If z_h − Z_med > 0.30 m: position = median world xy of the head keypoints within 0.30 m of z_h; z_top and the yaws recomputed at z_h | 1 | M1 |
| BR, band on the head | The same trigger; A1's ±0.30 m band re-centred on z_h inside A1's own torso region (falls back to HR) | 1 | M1 |
| d030 | A1's world duplicate rule with radius 0.30 m instead of 0.50 m | 1 | M2 |
| IMG | A duplicate iff containment ≥ 0.50 and world distance ≤ 1.00 m | 2 | M2 |
| FLAT | A1's flat test switched on: ≥ 0.80 of the box's depth points within 0.04 m of the torso plane, with at least 30 points | 1 | M4 |
| TINY | With FLAT, also drop a box with fewer than 30 depth points (too small to test) | 1 | M4 |

- **Candidates.** C00 (A1 frozen), C01 HR, C02 BR, C03 HR+d030, C04 HR+IMG, C05 HR+d030+FLAT, C06 HR+IMG+FLAT, C07 HR+d030+FLAT+TINY, C08 HR+IMG+FLAT+TINY, C09 BR+d030+FLAT+TINY, C10 BR+IMG+FLAT+TINY.
- **Considered and left out, with reasons:**
  - a floating-box test (it would also delete two glass-booth staff detections);
  - τ 0.35 (placeholder recall);
  - a 0.5 m minimum height (it cuts head-and-shoulders staff);
  - a no-silhouette test (it deletes staff behind glass);
  - a larger radius, a keypoint-aware duplicate test, and a small-and-weak box rule.
- **Selection rule.**
  1. Eligible: fit-train recall ≥ 0.97, line-member recall ≥ 0.97 and non-profile within 45 deg ≥ 0.90.
  2. Among the eligible, the fewest false people on fit-train.
  3. Ties: fewer parameters, then the lower candidate number. If none is eligible, no fix is chosen.
- **Checks of the chosen fix.**
  - Fit-validate holds iff recall and line recall are ≥ 0.97, non-profile within 45 is ≥ 0.90, and false people per image are lower than A1's on the same 56 images.
  - Development: recall ≥ 0.95 and false people ≤ 3.
  - Both pass: "confirmed".
- **Secondary scorer** (never used to choose or check). A detection the primary scorer counts as false is counted as exempt instead when it lies within 0.6 m of a counter staff member who fails `is_visible` but whose mark (x, y, 1.1 m) projects into the picture.

### Selection on fit-train

Table 2.2. Every candidate on fit-train (84 images), I03's scorer unchanged.

| Id | Blocks | Params | Recall (line, staff) | False (per image) | Exempt | Non-profile within 45 | Profile f/b | Secondary false | Eligible |
|---|---|---|---|---|---|---|---|---|---|
| C00 | A1 frozen | 0 | 226/238 (92/96, 26/34) | 17 (0.202) | 5 | 125/128 | 98/98 | 13 | no |
| C01 | HR | 1 | 234/238 (92/96, 34/34) | 8 (0.095) | 5 | 132/136 | 98/98 | 4 | no (line) |
| C02 | BR | 1 | 234/238 (92/96, 34/34) | 8 (0.095) | 5 | 132/136 | 98/98 | 4 | no (line) |
| C03 | HR+d030 | 2 | 238/238 | 8 (0.095) | 6 | 132/136 | 102/102 | 4 | yes |
| C04 | HR+IMG | 3 | 238/238 | 8 (0.095) | 5 | 132/136 | 102/102 | 4 | yes |
| C05 | HR+d030+FLAT | 3 | 238/238 | 5 (0.060) | 4 | 132/136 | 102/102 | 1 | yes |
| C06 | HR+IMG+FLAT | 4 | 238/238 | 5 (0.060) | 4 | 132/136 | 102/102 | 1 | yes |
| **C07** | **HR+d030+FLAT+TINY** | **4** | **238/238** | **4 (0.048)** | 4 | 132/136 | 102/102 | **0** | **yes: chosen** |
| C08 | HR+IMG+FLAT+TINY | 5 | 238/238 | 4 (0.048) | 4 | 132/136 | 102/102 | 0 | yes |
| C09 | BR+d030+FLAT+TINY | 4 | 238/238 | 4 (0.048) | 4 | 132/136 | 102/102 | 0 | yes |
| C10 | BR+IMG+FLAT+TINY | 5 | 238/238 | 4 (0.048) | 4 | 132/136 | 102/102 | 0 | yes |

- **The rule, applied by code.** C03–C10 are eligible. C07–C10 tie at 4/84. C07 and C09 have 4 parameters against 5 for the other two, and C07 has the lower number, so C07 is chosen. The choice was written and hashed before any candidate was scored on fit-validate or the development scenes.
- **What C07 leaves on fit-train.** All 4 remaining false people are glass-booth staff (class B). Median position error 0.060 m (A1 0.059 m); non-profile median 13.1 deg (A1 12.6 deg).
- **HR fired** on 8 kept boxes, none of which A1 had matched. C00 reproduces A1 exactly.
- **M3 audit.** FLAT (A1's own test) makes C05–C10 drop one glass-booth staff detection (seed 115 `eval_02`, flat share 0.98, scored exempt). The primary count does not change.

### Checks of C07

Table 2.3. Fit-validate (seeds 116–119, 56 images, held back): C07 against A1 frozen on the same images, with 95% intervals (Wilson; exact Poisson for false people per image).

| Measure | C07 | A1 frozen (C00) |
|---|---|---|
| Recall | 164/164 = 1.000 [0.977, 1.000] | 157/164 = 0.957 [0.915, 0.979] |
| Line members | 64/64 [0.943, 1.000] | 62/64 = 0.969 [0.893, 0.991] |
| Counter staff | 28/28 [0.879, 1.000] | 23/28 = 0.821 [0.644, 0.921] |
| False people per image | **2/56 = 0.036 [0.004, 0.129]** | 12/56 = 0.214 [0.111, 0.374] |
| Images with a false person | 2/56 = 0.036 [0.010, 0.121] | 11/56 = 0.196 [0.113, 0.318] |
| Exempt detections | 1 | 2 |
| Median position error | 0.061 m | 0.060 m |
| Non-profile within 45 deg (median) | 93/95 = 0.979 [0.926, 0.994] (12.8 deg) | 89/90 = 0.989 [0.940, 0.998] (10.7 deg) |
| Profile front/back | 69/69 | 67/67 |
| All matched within 45 deg (K3's population; verifier) | 157/164 = 0.957 | 151/157 = 0.962 |
| False people by class | B 1, C 1 | A 5, B 1, C 1, D 5 |
| Secondary scorer: false (per image) | 1 (0.018) | 11 (0.196) |

- **Check 1 holds.** Recall, line recall and non-profile facing are above their bars, and false people per image are below A1's.
- **Check 2, A1's 42 development E-real scenes.**
  - C07: 118/118 (staff 16/16) and 0 false; non-profile 61/61 within 45 deg (median 6.6 deg); all matched people within 45 deg 113/118 = 0.958 (verifier).
  - A1: 116/118 and 3 false, reproduced exactly.
  - The check passes.
- **Outcome: confirmed.**
- **Where C07's two fit-validate false people sit** (verifier). Each lies beside a counter staff member (s0) whom the visibility rule calls hidden: one at 0.41 m (class B; the secondary scorer reclassifies it) and one at 0.61 m (class C, just outside the secondary scorer's 0.6 m).
- **Block activity.** HR fired on 5 kept boxes on fit-validate and 3 on the development scenes, none of which A1 had matched. FLAT dropped one detection beside a hidden-by-rule staff member on fit-validate (seed 118 `eval_09`, flat share 0.84), which A1's run had scored exempt.

Table 2.4. A1's fit data, for information only (never a check).

| Set | C07 | A1 frozen |
|---|---|---|
| F-real (21 worked examples) | 39/39, 1 false; non-profile within 45 deg 17/18 | 36/39, 3 false; 15/15 |
| F-ph (268 placeholder scenes) | 513/585, 0 false (HR never fires) | identical |

**Post hoc: facing of re-lifted people** (`fix_posthoc/facing_relift.txt`; written after the checks; changes nothing; the verifier confirmed it).
- HR recomputes the yaw at the head depth. Over the 19 matched people C07 re-lifted (fit-train 8, fit-validate 5, development 3, F-real 3), the recomputed yaw is within 45 deg for 16 and unknown for 3.
- A1's original yaw of the same boxes was within 45 deg for 17 and 90 deg off for 2. So the recomputation is roughly neutral: it trades two wrong yaws for three unknown ones. F-real's 17/18 is one of those unknowns.

---

## 3. The frozen second confirmation (`CONFIRM2_FROZEN.md`)

- **Purpose.** Score A1 with fix 2 once on layouts nobody has rendered or seen, with nothing re-tuned, as the fix step itself recommended.
- **Rendering.**
  - The 14 rows `real_eval_00`–`13`, variant `real`, one fixed generate command per seed in list order, with `TMPDIR` inside `confirm2/render/` and the scenes in `confirm2/seeds/`.
  - Clock: no new command starts after 3 h of summed command time; waits do not count.
- **Method.**
  - `FX.run_all(..., ids=["C00", "C07"])` on one raw pass. Only C07 is judged; C00 (A1 alone) is shown beside it on the same images.
  - 23 files are pinned by sha256, and `src/` and `configs/` are pinned to f1a488a through `git diff`.
  - The new sidecar driver may differ from `sidecar/infer_fit.py` only in its paths, its guard module and its hash checks.
- **Criteria.** A1's S1–S7 and K1–K3 verbatim (the verifier diffed them against the first protocol), judged once on pooled point estimates. Outcomes: success, kill, neither or incomplete. Not carried over: the runtime bar, the "10 points below the best VLM" clause and A2.
- **Intervals** (reported, not judged): Wilson; exact Poisson for false people per image; a layout-cluster bootstrap (2000 draws, `random.Random(0)`); a seed-cluster bootstrap as a sensitivity.
- **Checks before the detector pass.**
  - C1, code identity.
  - C2: the new code reproduces the fix step's fit-validate rows and summaries and its development results exactly. The verifier found this feasible.
  - C3, a truth self-check on the test scenes.
  - C4, the manifest.
  - C5: the seed note is valid and was hashed before the first render.
  - C6, the driver diff.
- **Prediction** (the author's, written before any seed was named; inference):
  - recall 0.97–1.00, with the position and facing bars met with room;
  - false people 0.03–0.11 per image, central guess 0.06; S3 met with probability about two in three, the main risk being staff behind glass;
  - secondary scorer 0.01–0.05 per image; a kill very unlikely.
- **Cost and time.**
  - Zero calls and zero cost; one GPU job at a time.
  - Rendering ten seeds takes about 68 min at the training seeds' rate (4,075.6 s for 10 seeds) and about 133 min at the first confirmation's median (796 s per seed); the reason for that difference is not recorded. The 3 h cap covers about 13–26 seeds.
  - The detector pass and scoring take about 1 min each.

Table 3.1. The comparators the note records (section 7 of the note; the verifier's numbers match every one).

| Measure | C07, development E-real (42) | C07, fit-validate (56) | C07, fit-train (84; selection data) | A1 (C00), fit-validate (56) | A1, first confirmation (182; published) |
|---|---|---|---|---|---|
| Recall | 118/118 | 164/164 | 238/238 | 157/164 = 0.957 | 510/520 = 0.981 |
| Line members | 48/48 | 64/64 | 96/96 | 62/64 | 207/208 |
| Staff | 16/16 | 28/28 | 34/34 | 23/28 | 69/78 |
| False people (per image) | 0 (0.000) | 2 (0.036; exact Poisson 0.004–0.129) | 4 (0.048) | 12 (0.214) | 44 (0.242) |
| Secondary scorer: false (per image) | 0 (0.000) | 1 (0.018) | 0 (0.000) | 11 (0.196) | not computed |
| Median position error | 0.053 m | 0.061 m | 0.060 m | 0.060 m | 0.061 m |
| Non-profile within 45 (median) | 61/61 (6.6 deg) | 93/95 (12.8 deg) | 132/136 (13.1 deg) | 89/90 (10.7 deg) | 278/287 (10.3 deg) |
| Profile front/back | 57/57 | 69/69 | 102/102 | 67/67 | 223/223 |

---

**The numbers agree.** The verifier re-scored everything with its own code (`verify/vlib.py`, `verify/vscore.py`). It implemented the blocks from the protocol's text and imported only A1's pipeline, I03's scorer, the harness and sgi, never `fitlib.py` or `fix/`.
- **0 differences.** All 11 candidates on fit-train; C00 and C07 on fit-validate, the development scenes, F-real and F-ph; every interval; the secondary scorer, the M3 audit and the post hoc classes; and 1,036 per-image rows, down to each kept detection's position.
- **The same choice and outcome.** On its numbers the rule picks C07, and the outcome is "confirmed". The fix step's post hoc claims also hold.
- **The raw pass reproduces.** Its re-run of A1's detector and pose pass on seed 116 is bit-identical (234 boxes, 87 pose sets, 1,479 keypoints), with `pip freeze` unchanged.
- **The order holds.** The protocol was hashed before `fix/` existed, selection ran once, and the choice was hashed before the checks. Fit-validate was touched only by manifest hashing and the detector pass before the checks.

2. **A text search opened spent test-seed files** (verifier). At 19:18:09 the protocol author ran a recursive `grep -rIl` over `.superpowers`, `src`, `configs`, `docs` and `tests`, piped through `grep -v /phase2/seeds/`.
   - It opened 364 files under `results/still-scenes/scene-seeds/phase2/realistic` (seed200–212), plus `results/still-scenes/people-finder/early-detector/phase2/confirm/posthoc_false.txt`, `raw/meta.json`, `summary.json`, `tables.txt` and `results/still-scenes/learned-models/soft-density/i04_heldout.py`.
   - It still breaks the rule that no file under `results/still-scenes/scene-seeds/phase2/realistic/` may be opened by any process of this run. It also makes the exposure section of `CONFIRM2_FROZEN.md` wrong: the note says it opened no such file and lists `posthoc_false.txt` as not read. Access times cannot settle it (relatime).
3. **Minor points in `CONFIRM2_FROZEN.md`** (verifier, plus one writer's check).
   - The exposure statement (finding 2).
   - The harness's sha256 is recorded (`39148377…`) but not asserted. Asserting it now would change check C1, which the note's rules do not allow through a deviation file. Recording its hash at run time, as information, would be allowed.
   - Section 12 says one of C07's two fit-validate false people is of the glass-staff kind. In fact both lie beside hidden-by-rule staff (0.41 m, class B; 0.61 m, class C). This page uses the verifier's reading.
   - Writer's check: section 1.6 gives the training seeds' rate as 403–412 s per seed; the render log gives 395.5–420.6 s. It is information only, not a rule.
4. **Undisclosed reads and scratch files** (verifier; no rows and no test content).
5. **Render departures** (verifier): one resource check at the start instead of one before each command, and no `TMPDIR`.
7. **The protocol author's disclosed departure.** It added seeds 0–2 (development, seen) to the excluded ranges, which the brief had not listed.

**Not settled.** Whether IMG would be more robust than d030. C08 has the same fit-train numbers as C07, and d030 kept a second box 0.46 m beside a glass-booth staff member (seed 114 `eval_02`, containment 1.00, scored exempt). By the protocol, C08 was never scored on fit-validate.

---

## 5. What it means (inference)

**The diagnosis.**
- **The failure is the method's, not one seed batch's.** The training renders reproduce the first confirmation's failure at a similar rate (0.202 against 0.242), with a similar mix of causes to its published classes. So that failure belongs to the method on this generator, not to seeds 200–212.
- **Three causes are on the detector side.** M1 (depth median on the occluder), M2 (the world-only duplicate radius) and M4 (pictures of people) each have a simple geometric fix.
- **The fourth is a scorer mismatch.** M3 is the scorer's exact-depth visibility rule disagreeing with the picture, because the renderer makes glass opaque.

**The fix.**
- **On the fit data it does what it was built for.** On fit-validate, false people fall from 0.214 to 0.036 per image while recall rises from 0.957 to 1.000. HR never fired on a box A1 had matched, in any set, and placeholder scenes are unchanged.
- **56 images cannot show the bar is met.** The upper end of the false-people interval (0.129) still crosses 0.10, and fit-validate shares the fit-train geometry templates.
- **What remains is staff whom the visibility rule calls hidden.** That covers all 4 false people on fit-train and both on fit-validate.
- **So S3 will likely turn on the glass booths.** Suppose the test seeds behave like the fit data: classes A and D are cleared, and classes B and C stay at the first confirmation's 13 in 182 (0.071 per image). Then S3 would be met with about 0.03 to spare. A draw with more glass booths could push it over. This is the protocol author's reasoning too (`CONFIRM2_FROZEN.md` section 12).

**The scorer question is now concrete.** It is about the renderer's opaque glass, not about the detector.

**What a success would and would not show.**
- **It would show:** new layouts of the same 14 rows from the same generator.
- **It would not show:**
  - new stall geometries (the templates repeat);
  - placeholder avatars (fix 2 changes nothing there; placeholder recall stays 0.877, its misses mostly group members seen in profile or from behind);
  - real sensor depth;
  - the pipeline with the detector in it.
- The draft metric list's decision D14 (`results/still-scenes/metrics-and-statistics/metric-v1/METRIC_LIST_v1_DRAFT.md`, not agreed) would word such a claim as "new seeds of the same rows".

**For the live check.**
- **It measures A1, not fix 2.** The live check's frozen people are A1's (`results/still-scenes/people-finder/early-detector/phase1c/a1/eval_people.json`).
- **The difference there is small.** On those 42 development scenes, fix 2 differs from A1 only in counter-staff people: A1's 2 staff misses and 3 false people. Fix 2's effect on a live result would likely be small.

**Status of I09.** By its own pre-registered standard, the detector is still not a method. Fix 2 is a candidate with a confirmed check on training seeds and a frozen test. The next informative step is the second confirmation.

---

1. **The test seeds.** Name at least 10 seeds that have never been rendered: for example 213–219 plus at least three from 220 up, or ten or more from 220 up.
   - The confidence page names 213–219 as an option for its own confirmation. Say whether any seed is shared, and in which order; the seed note must record it.
   - More than 10 seeds may be listed. The 3 h clock then decides how many are used (about 13–26).
   - After scoring, the seeds are spent as I09 test data.

3. **Whether to run the second confirmation, and when.**
   - Cost: zero calls and zero cost.
   - Time: about 68–133 min of rendering for ten seeds (training-seed rate about 408 s per seed; the first confirmation's median 796 s), capped at 3 h, then about 1 min each for the detector pass and the scoring.
4. **A1's flat test and staff behind glass.** The flat test, which fix 2 switches on, drops two detections beside staff whom the visibility rule calls hidden (both scored exempt; the fit-train one is a glass-booth staff member whom the picture shows). Is that acceptable, given that the robot can see such people?

5. **The text-search exposure.** Should it be accepted and recorded in a hashed `confirm2/DEVIATIONS_1.md`, written before the seed note or any render? That file would also correct the frozen note's exposure statement and its section 12 wording, and it could record the harness hash as information (an assertion would change check C1).
6. **"Confirmed".** Should it be read as resting on fit-validate alone, since the development check was known to pass?
7. **The render step's departures** (one resource check, no `TMPDIR`): accept them?
8. **d030 or IMG.**
   - The rule picked C07 (d030) on the parameter tie-break. C08 (IMG) has identical fit-train numbers and is arguably more robust, since it merges second detections by image overlap even without head keypoints.
   - Switching now would be a choice made after fit-validate, on which C08 was never scored, and the frozen note fixes C07.
   - Recommendation (inference): keep C07 for this confirmation, and carry IMG as a candidate for any later revision.
9. **Recomputed facing on re-lifted boxes** (post hoc, roughly neutral): keep it, or revert to A1's yaw in a later revision?
10. **New stall geometries.** Should a later test add generator geometries beyond the 14 rows' templates? That is outside the frozen confirmation.

---

## 7. Proposed REPORT edits (not applied)

Each item gives the exact old text from the current `REPORT.md` and the proposed new text.
- **Checked.** Each old text occurs exactly once (Python's `str.count`, at writing).
- **They apply in order.** On a copy in memory, applying items 1–11 in order works: each old text still occurs exactly once when its turn comes, and each new text occurs once afterwards. Items 1–6, 8 and 11 alone, in order, also apply cleanly.
- **Optional items:** 7, 9 and 10.
- **The tag.** New text carries [phase 3]. "How to read" already defines it, right after the [phase 2] definition, for the confidence page. Item 1 replaces that one-bullet definition with one that names this page too, in the same place.
- **Before applying,** keep a copy of the report as `REPORT.before-phase3-i09-fix.md` (item 11 names it).

**Merging with another phase-3 page.**
- Keep one [phase 3] bullet in "How to read" and add each page to its list.
- List every phase-3 page in one parenthesis in the opening line (item 2).
- Put the revision-note bullet at the very end (item 11). If another bullet has been appended since this page was written, anchor on that bullet's final sentence instead.

### 1. "How to read": the [phase 3] definition, now naming this page

The [phase 3] bullet already sits right after the [phase 2] bullet (the confidence page's definition, applied on 18 Oct). This item replaces that bullet with one that names both pages, keeping everything it says about the confidence page. Do not add a second [phase 3] bullet.

Old:
New:

### 2. Opening status line: the list of pages

Old:
```text
and phase 3's zero-call confidence probe (`results/still-scenes/confidence-signal/README.md`) [phase 3], nothing has been built
```
New:
```text
and phase 3's zero-call confidence probe (`results/still-scenes/confidence-signal/README.md`) and zero-call I09 detector fix (`results/still-scenes/people-finder/early-detector/fix/README.md`) [phase 3], nothing has been built
```

### 3. Opening status line: what phase 3 answers

Old:
```text
and the verdict is fragile. Every other probe below is a plan
```
New:

### 4. Section 1.2, item 5 (I09)

Old:
```text
so by its own pre-registered standard the detector is not yet a method, and no kill fired (`results/still-scenes/people-finder/early-detector/phase2/README.md` section 1).
```
New:

Old:
```text
and whether to replay it first on records not used to design it.
```
New:

### 6. I09: the fix and the frozen second confirmation

Old:
```text
which would raise the disagreement on successes too (untested).
```
New:

### 7. Section 6, step-3b row (optional)

Old:
```text
The unseen-layout confirmation (seeds 200–212) is neither: 0.242 false people per image against 0.10 (`results/still-scenes/people-finder/early-detector/phase2/README.md`) |
```
New:

Old:
```text
a scorer change that would apply to the development numbers too.
```
New:

### 9. I46: the seeds used (optional)

Old:
```text
they stay out of this sealed set and could serve later C5 work as training data only (`results/still-scenes/learned-models/flow-model-pilot/README.md` section 7).
```
New:

### 10. Section 8, synthetic renders (optional)

Old:
```text
finding 0.983 of visible people with the realistic avatars and 0.845 with the placeholder ones (`results/still-scenes/overview/phase-notes/phase1c.md` section 1).
```
New:
```text
finding 0.983 of visible people with the realistic avatars and 0.845 with the placeholder ones (`results/still-scenes/overview/phase-notes/phase1c.md` section 1). [phase 3] I09's fix found three more render effects (`results/still-scenes/people-finder/early-detector/fix/README.md` sections 1 and 5): the rendered depth treats glass as opaque, so the package's visibility rule calls staff in glass booths hidden while the picture shows them; the fix re-lifts occluded staff from 5×5 depth at head keypoints, which a real sensor would make noisy at head edges; and the generator reuses its camera–staff geometry across the seeds of a row, so new seeds of the same rows are weaker evidence than their count.
```

### 11. Revision note

Append at the very end of the revision note. If another bullet has been appended since this page was written, anchor on that bullet's final sentence instead.

Old:
```text
The report before these changes is kept as `REPORT.before-phase3-confidence.md`.
```
New:

---

## 8. Lead addendum (18 Oct, 20:15; written after sections 0–7)

- **The deviation file.** `confirm2/DEVIATIONS_1.md` (sha256 `464e8c8a…16ee`, read-only) was written at 20:05, before any seed note or render; at writing, `confirm2/` did not exist and no seed213+ folder existed anywhere. It does three things:
  - accepts the text-search exposure (section 4, finding 2) and corrects the frozen note's exposure statement;
  - corrects section 12's premise: both of C07's fit-validate false people lie beside hidden-by-rule staff;
  - makes the scoring step print the harness hash as information, not as a check.

  The frozen note itself is unchanged, and its hash verifies.
- **REPORT.** The edits of section 7 were applied with small changes:
  - item 4 marks the development result as not blind;
  - item 6 and item 11 name the deviation file;
  - item 10 separates the two render effects from the untested sensor-depth dependence.

  The report before them is `REPORT.before-phase3-i09-fix.md`.
