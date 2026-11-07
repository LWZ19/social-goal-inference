# I09 fix 2: protocol (written before any candidate is scored)

Fix agent, 2026-10-18, written from 19:05 (+08:00). This note fixes the candidate list, the selection rule and the
checks before any candidate is scored. Its sha256 is in `FIX_PROTOCOL.sha256`. Nothing in `fix/` exists when this
note is hashed; the code is written after it and its sha256 goes into `chosen_fix2.json`.

Inputs known when this was written: `DIAGNOSIS.md`, `diagnose_tables.txt` and `whatif.txt` (fit-train, A1's fit
data and A1's development scenes), and the published aggregates of `phase2/i09/README.md`. Fit-validate (seeds
116-119) has not been scored or viewed by anyone. Paths are relative to `phase3/i09_fix/`.

## 1. Data and roles

| Data | Images | Role |
|---|---|---|
| Fit-train: seeds 110-115, rows `real_eval_00`-`13` | 84 | Every candidate is scored; the selection rule is applied here only |
| Fit-validate: seeds 116-119, same rows | 56 | Check of the chosen fix (and A1 frozen as the reference) only, after the choice is written and hashed |
| A1 development E-real (through the step-0 harness, as `whatif.py`) | 42 | Check of the chosen fix (and A1 frozen) only |
| A1 fit data F-real (21) and F-ph (268), through `fitdata.load` | 289 | Reported for the chosen fix and A1 frozen only, for information; never a check, never used to choose |

- Raw detector and pose outputs are reused, never recomputed: `raw_fit/rtdetr_v2_r50vd.jsonl` (sha256 in
  `RAW_FIT.sha256`, covers seeds 110-119) and A1's `raw_eval/` and `raw_fit/` (sha256 pinned in `fitlib.py` and
  checked against A1's `raw_fit/meta.json`). No detector or pose pass, no GPU job, no new model.
- Fit scenes are read only through `fitlib.load_fit_scene` (fit path guard, then `H.guard_scene`); A1's development
  scenes only through the harness (`H.distinct_scenes`, `H.scene`).
- Nothing from seeds 200-219, `phase2/seeds/realistic/`, the confirmation's rows, people rows, raw outputs or
  per-row post hoc detail is read. Seeds 3-99 are not touched.

## 2. The common base (unchanged in every candidate)

- A1's code by import, pinned by sha256 (`fitlib.PINNED`): `pipeline.py` (`P`), `common.py` (`C`), I03's
  `people_scorer.py` (`PS`). A1's frozen setting from `chosen.json`/`chosen_fix.json`: RT-DETRv2-r50vd, tau 0.25,
  height band `wide` (1.00-2.30 m on z_top), d_push 0.20 m, facing variant F5.
- Every candidate starts from `P.candidates(raw, depth, cam, 0.25)`: A1's torso region (shoulder-hip hull, else the
  box's central rectangle), Z_med, the +-0.30 m band, body_xy, z_top, the flat flag and `n_Q`, the F1-F5 yaws. The
  steps after it are A1's `people_for` (score >= tau; step-2 reject; height band; flat test when on; push; duplicate
  test in descending score order), changed only where a building block below says so.
- Scorer: `C.score_scene` -> `PS.score_people(..., exempt_xy=PS.exempt_points(truth), facing="full")`, default
  thresholds for fit scenes, `C.thresholds_for(runs)` for development scenes. Unchanged.
- Candidate C00 must reproduce `P.run_image` exactly on every image (asserted), and A1's recorded numbers on fit-train
  (226/238, 17 false) and on the development scenes (116/118, 3 false).

## 3. Building blocks (fully specified)

No block reads a situation label, role, group id, scene name or truth. Each uses only the box, its score, the
keypoints, the depth image and the camera.

**HR, head re-lift of an occluded torso (M1).** Applies to every candidate box that A1 did not reject at step 2.
- Head keypoints: nose, left eye, right eye, left ear, right ear (COCO 0-4) with confidence >= 0.30 (A1's `CONF`)
  whose 5x5 depth (`sgi.geometry._depth_at`, window 5 = A1's `KP_WINDOW`) is valid. If there are none, no change.
- z_h = the smallest of those depths (the nearest head keypoint; never the head median, DIAGNOSIS section 8).
- Fires iff z_h - Z_med > GAP, with GAP = 0.30 m.
- When it fires: H' = the head keypoints with depth <= z_h + GAP. body_xy = (median world x, median world y) of
  `sgi.geometry.lift(depth, cam, u, v, 5)` over H'. Z_ref = z_h. z_top is recomputed with A1's formula (box top
  centre lifted at Z_ref instead of Z_med). The yaws are recomputed with A1's `P.facing(kp, depth, cam, Z_ref,
  body_xy, cam_xy)` (so the keypoint depth test is against Z_ref). torso_pixel = the median (u, v) of H'.
- Unchanged when it fires: the score, the box, the flat flag and `n_Q` (computed by A1 on the original region).
- Then the usual height band, push and duplicate test.
- Parameters: 1 (GAP, used for the trigger and for the H' filter).

**BR, band re-anchored on the head (M1, the "depth mode above the occluder").** Same trigger as HR (same head
keypoints, z_h, GAP 0.30 m).
- When it fires: S = the pixels of A1's own torso region (the same hull or rectangle A1 used) whose depth lies within
  0.30 m (A1's `BAND_M`) of z_h. If |S| >= 10 (A1's `MIN_KEPT`): Z' = the median depth of S; K = the region pixels
  within 0.30 m of Z'; if |K| >= 10, body_xy = (median world x, median world y) of K, torso_pixel = median (u, v) of
  K, Z_ref = Z'. Otherwise (|S| < 10 or |K| < 10) the HR result is used.
- z_top and the yaws are recomputed at Z_ref as in HR; the rest as in HR.
- Parameters: 1 (GAP; the band and the minimum count are A1's constants).

**d030, a smaller world duplicate radius (M2).** A1's world rule with d_dup 0.30 m instead of 0.50 m (A1's other grid
value). Parameters: 1.

**IMG, image-space duplicates (M2, keypoint-free).** A box that passed the earlier steps is a duplicate of an
already kept box iff containment >= 0.50 and the world distance between their pushed positions is <= 1.00 m.
Containment = intersection area / area of the smaller box (pixel boxes as the detector gives them). The 0.50 m
world-only rule is not used. Parameters: 2 (containment 0.50, world cap 1.00 m).

**FLAT, A1's flat test on (M4).** Drop a box when A1's flat flag is set: at least 0.80 of the valid depth points on
the box's even-row, even-column grid lie within 0.04 m of the plane through the torso band's kept points, with at
least 30 such points (`n_Q >= 30`). Applied where A1 applies it (after the height band, before the push).
Parameters: 1 (A1's switch).

**TINY, too small to verify (M4; only together with FLAT).** In the flat step, also drop a box with `n_Q < 30`, i.e.
one for which A1's flat test cannot be computed. A standing person of 1.7 m covers such a small box only far beyond
any distance that matters to the robot (inference). Parameters: 1 (the rule; it reuses A1's `FLAT_MIN_Q`).

Parameter count of a candidate = the sum over its blocks (A1's constants reused unchanged count 0).

## 4. The candidates (11, including A1 frozen)

The order is fixed here and is the last tie-break: A1 first, then single blocks, then combinations, HR before BR,
d030 before IMG, fewer blocks first.

| Id | Blocks | Parameters | Addresses |
|---|---|---|---|
| C00 | A1 frozen | 0 | (reference) |
| C01 | HR | 1 | M1 |
| C02 | BR | 1 | M1 |
| C03 | HR + d030 | 2 | M1, M2 |
| C04 | HR + IMG | 3 | M1, M2 |
| C05 | HR + d030 + FLAT | 3 | M1, M2, M4 |
| C06 | HR + IMG + FLAT | 4 | M1, M2, M4 |
| C07 | HR + d030 + FLAT + TINY | 4 | M1, M2, M4 |
| C08 | HR + IMG + FLAT + TINY | 5 | M1, M2, M4 |
| C09 | BR + d030 + FLAT + TINY | 4 | M1, M2, M4 |
| C10 | BR + IMG + FLAT + TINY | 5 | M1, M2, M4 |

Considered and not included:
- **Floating-box test (DIAGNOSIS 4b).** On fit-train its 0.15 m "nearer below" threshold would also delete two
  glass-booth staff detections (below-strip contrast -0.09 and -0.12 m, `diagnose_tables.txt`), i.e. it would lower
  the count partly by deleting people the robot can see. Moving the threshold to spare them would be set by looking
  at those two values. FLAT + TINY covers the four M4 boxes without that interaction.
- **Global score threshold 0.35 (4c).** Known costs: A1's placeholder recall 0.877 -> 0.826 and one development staff
  member (`whatif.txt`); the confirmation's class-D boxes scored up to 0.53, so it would not have caught them all.
- **Minimum metric height 0.5 m (4d).** Cuts 2 matched head-and-shoulders staff on fit-train.
- **A "no depth silhouette" test (3b).** Deletes staff behind glass whom the RGB shows (flagged in the diagnosis).
- **A larger world radius, or the image test added on top of 0.50 m (1e).** Does not fix M2; 1e alone dropped a
  correctly placed development staff member. With HR, the M1 second detection is re-lifted onto the matched one, so
  the world rule already merges it (expected).
- **A keypoint-aware duplicate test (nose distance).** On fit-train it separates the same pairs as IMG (same person:
  noses <= 2 px, containment >= 0.83; different people: noses >= 15 px, containment <= 0.34) with more parameters.
- **A small-and-weak box rule (score and metric height).** No expected gain on fit-train over FLAT + TINY; an
  untested risk for partly hidden people with low scores.

Expectation from the diagnosis's what-ifs (not binding; HR here also recomputes z_top and facing, which the what-if
did not): C03-C04 about 8 false, C05-C06 about 5, C07-C10 about 4 (the four glass detections that count as false),
recall about 238/238 for every candidate with an M2 block.

## 5. Metrics, for every candidate on fit-train

Recall (overall and by role class: line, staff, group, other), false people and false per image, exempt, near
hidden, merged pairs / neighbour pairs, median xy error, non-profile facing within 45 deg (`C.facing_summary` over
matched people whose true view is not profile: `w45 / n`, an unknown yaw counted as not within), profile front/back,
the number of kept boxes the M1 block fired on (and how many of them A1 had matched), and the secondary scorer
variant (section 8). Also, as an M3 audit: the number of A1-kept detections lying within 0.6 m of a counter staff
member who fails `is_visible` while the 1.1 m mark is in the picture that the candidate no longer outputs (truth is
used only for this count, after the candidate's people are fixed).

## 6. Selection rule (fixed in advance)

1. Eligible: fit-train recall >= 0.97, line-member recall >= 0.97, non-profile facing within 45 deg >= 0.90 (point
   estimates; A1 frozen is eligible if it meets them).
2. Among eligible candidates, choose the one with the fewest false people per image on fit-train (equivalently the
   fewest false people, all on the same 84 images).
3. Ties: fewer parameters (section 3); then the lower candidate number (section 4).
4. If no candidate is eligible, no fix is chosen and that is the result.

The rule is applied by code, which writes the choice to `chosen_fix2.json` (with the sha256 of every file in `fix/`)
and a copy `chosen_fix2_precheck.json` with its sha256 in `CHOSEN2.sha256`, before fit-validate or the development
scenes are scored by any candidate.

## 7. Checks of the chosen fix (after the choice is written)

**Fit-validate (seeds 116-119, 56 images).** The chosen fix and A1 frozen are scored. Reported with 95% intervals:
Wilson (`PS.wilson`) for recall, line-member recall, staff recall, non-profile within 45 deg, and the share of
images with at least one false person; an exact Poisson (Garwood) interval for false people per image. The fix
**holds** on fit-validate iff
- recall >= 0.97, line-member recall >= 0.97 and non-profile within 45 deg >= 0.90 (point estimates), and
- its false people per image are lower than A1 frozen's on the same 56 images.

**A1's 42 development E-real scenes.** The fix must not make them worse: recall >= 0.95 and false people <= 3 (A1's
3 of 42).

Outcome labels: **confirmed** (both checks pass), **fails fit-validate**, **fails development**, or **fails both**.
A failed check is reported as it is. No other candidate is then chosen, and no candidate other than the chosen fix
and A1 frozen is scored on fit-validate or the development scenes.

For information only (not checks): the same numbers on A1's F-real and F-ph fit data, and the fit-validate rows
broken down by the diagnosis's post hoc classes (A-D) computed with the same rules as `diagnose.py`.

Identical to I03's scorer, except that a detection the scorer counts as false is counted as exempt instead when it
lies within 0.6 m (world xy) of a counter staff member (a true person with a role) who fails `is_visible` but whose
marked point (x, y, 1.1 m) projects into the picture (`sgi.geometry.project`, `in_image`). Recall, matching and
everything else are unchanged. Reported for every candidate on fit-train, and for the chosen fix and A1 frozen on
fit-validate and the development scenes, as false and exempt counts and false per image, with the number of
detections it reclassifies.

## 9. Order of work and outputs

1. This note, then `FIX_PROTOCOL.sha256`.
2. Code in `fix/`: `fixlib.py` (the blocks, the candidates, the secondary scorer, intervals), `select.py` (fit-train,
   the rule, `chosen_fix2.json`), `confirm.py` (fit-validate, development and information sets for the chosen fix
   and A1 frozen). Bytecode writing off; A1 and I03 code imported, never copied.
3. `select.py` once: every candidate on fit-train, the rule applied, the choice written and hashed.
4. `confirm.py` once: checks for the chosen fix; `chosen_fix2.json` is then completed with the checks (the choice
   block unchanged, verifiable against `chosen_fix2_precheck.json`).
5. Tables in `fix_tables.txt`; hashes of every output in `FIX.sha256`.
