# I03 phase 1b: frozen protocol

The I03 lead wrote this on 17 Oct 2026, before any I03 code scored an evaluation scene. Its sha256 is in `FROZEN.sha256`. This file is never edited. Any change goes into a dated addendum with its own hash, written before the data it affects is scored.

- The fresh seeds are 100–109, not 3–12. Seeds 3–99 stay reserved for I46.
- Thresholds are fitted on fit data only. The whitelisted development scenes become evaluation data (the REPORT fitted on "dev scenes plus fresh seeds").
- A B-relocate arm is added.
- The scorer is a shared module.

Paths are relative to `phase1b/`.

**Exposure (disclosed).** The whitelist re-run of the blob pilot (RERUN R36–R38 and R56, `results/still-scenes/perception/depth-blob-pilot/`) has already run a blob detector on all 68 evaluation scenes, using the pilot's thresholds.
- I read its summaries in REPORT.md and RERUN.md, never its per-scene files. I03 code will not read `results/still-scenes/perception/depth-blob-pilot/per_scene`.
- Those summaries shaped the grid's generic ranges. They report that the misses are staff behind counters or blobs merged with tall structure at 2.0–2.1 m, and that false bodies run 0.6–2.6 per realistic image.
- B-relocate and its 40 px radius are informed by I01 and RERUN results on these same development runs.
- So are the Stage B failure population and its fixable shares (section 8).
- Nothing else in this note was chosen from evaluation data.

## 1. Data
**Fit data.** Thresholds are chosen on these only.
- **F-real** (21 scenes): the realistic worked examples, `H.examples()` under `keep/realistic_final/{real,staffed,plain}/examples`, loaded with `H.example()` under the default config. They hold 42 visible true people.
- **F-ph** (268 scenes): the 8 placeholder worked examples (`data/placeholder/examples`) and the 260 fresh-seed scenes described below.

**Evaluation data.** Scored once, with the frozen settings. These are the 68 scenes of `H.distinct_scenes()`, split into two domains:
- **E-real** (42 images): `real_`, `staffed_` and `plain_eval_00`–`13`, that is 14 layouts × 3 arms.
- **E-ph** (26 scenes): `data/placeholder/eval` `eval_00`–`13`, plus `eval_14`, `eval_15`, `eval_20` and `eval_21` of `keep/dev_new/seed{0,1,2}/scenes`.

These are the scenes of I09 A1. Fit and evaluation layouts come from different generator rows or seeds.

**Thresholds.** Visibility on evaluation scenes uses the run's recorded config. On fit scenes it uses `configs/default.yaml`. The scorer asserts that `person_radius_m` is 0.25, `visibility_min_samples` is 3 and `visibility_depth_tolerance_m` is 0.3 in every case.

**Fresh seeds** (`results/still-scenes/perception/depth-blob-detector/gen_fresh.py`, a single process):
- `cfg = load_config(DEFAULT_CONFIG)`.
- For each seed k from 100 to 109, the output goes to `fresh_seeds/seed<k>/`.
- First pass: set `rng = np.random.default_rng(k)`. For every row of `_SPLITS`, call `_write_scene(out/split, index, row, rng, cfg, counts)` if `row[0] in cfg.dev_situations`.
- Second pass: set `more = np.random.default_rng([k, _MORE_STREAM])` and do the same over `_MORE_SPLITS`, with indices starting at `len(dict(_SPLITS)[split])`.
- Rows of `group_photo` and `lift_wait` are skipped before any random draw, so no held-out scene is ever rendered.
- The first pass, `eval_14` and `eval_15` match what `generate(k)` would write. `eval_20`, `eval_21` and `examples_07` differ from `generate(k)`.
- Each seed gives 26 scenes: `eval_00`–`15`, `eval_20`, `eval_21` and `examples_00`–`07` (no_one 11, waiting_line 7, conversation_group 5, audience 3).

## 2. Stage A detector (numpy only)
**Fixed steps.**
1. Lift every valid depth pixel with `sgi.geometry.lift_all` (the floor is z = 0).
2. Keep the points with z_lo < z < z_hi.
3. Build a height map on the package's 0.05 m grid: each cell holds its highest kept point.
4. Find 8-connected components.
5. Split each component at its local maxima. A seed is a cell equal to the component's maximum within a square window of half-width split_r. Touching seed cells form one seed. Every cell joins its nearest seed cell.

**Blob features.**
- Top: the highest cell.
- Footprint: the extent of the cell centres along their first principal axis, plus 0.05 m.
- Position: the mean of the cell centres.

**Body filter.** A blob is kept when top_lo ≤ top ≤ z_hi − 0.05 and foot_lo ≤ footprint ≤ foot_hi. A blob that reaches the clip height is taller than the clip, so it is not a body.

| Threshold | Values |
|---|---|
| z_lo | 0.10, 0.40 m |
| z_hi | 1.90, 2.05, 2.20 m |
| top_lo | 1.10, 1.30, 1.50 m |
| split_r | 0.15, 0.25, 0.35 m |
| foot_lo | 0.15, 0.30 m |
| foot_hi | 0.70, 1.00 m |

**Facing axis** (reported only). The minor principal axis of the xy of the blob's kept points, taken either from all of them or from those at 1.0–1.6 m. The variant with the higher fit-data axis accuracy wins (the minimum over the two domains). On a tie, all points.

## 3. Selection on fit data
**Metrics.** For each fit domain D (F-real, F-ph):
- recall_D: matched people over visible people, pooled within D, at 0.5 m.
- fb_D: false bodies per scene. Exempt blobs (section 4) do not count.
- mp_D: the share of neighbour pairs merged, or 0 when D has no pairs.

**Rule.**
1. A setting is feasible when fb_D ≤ 0.10 in both domains.
2. Among feasible settings, take the highest min_D recall_D.
3. Break ties in this order: lower max_D mp_D, then lower max_D fb_D, then the least restrictive setting. Least restrictive means larger split_r, then lower z_lo, higher z_hi, lower top_lo, lower foot_lo and higher foot_hi.
4. If no setting is feasible, take the lowest max_D fb_D among settings with min recall ≥ 0.95. Failing that, take the highest min recall among settings with max_D fb_D ≤ 0.30. Failing that, take the lowest max_D fb_D.

**Freeze.** The 216-row table and the choice go to `results/still-scenes/perception/depth-blob-detector/fit_selection.json`, with its hash in `FIT.sha256`. Both are written before Stage A code loads any evaluation scene.

**One general fix.** The REPORT's kill line allows "one general fix".
- It applies only if the chosen setting misses a Stage A success bar on fit data.
- It is at most one revision of the detector, designed from fit-data failures only. It may name no situation, scene or label.
- It is written as a hashed addendum and re-fitted with the same grid and rule.
- It replaces the original only if it wins under the same ordering.
- It is never made after seeing evaluation data. The kill bars are judged after it.

## 4. Scorer (`results/still-scenes/perception/depth-blob-detector/people_scorer.py`; I09 A1 reuses it unchanged)
`score_people(people, scene, th=None, *, exempt_xy=(), facing="full") -> dict`

**Inputs.**
- `people`: a list of (x, y, yaw or None), in world metres and radians.
- `scene`: any object with `.observation` and `.truth`, such as `H.scene()`, `H.example()`, or a wrapped `sgi.scene.load_scene`.
- `th`: `Thresholds`; the default is `configs/default.yaml`.

The function returns counts, so callers pool by summing them, and medians over the pooled lists.

**Definitions.**
- **Visible:** a person in `truth.people` who passes `sgi.gt.derive.is_visible`.
- **Matching:** the scorer calls `sgi.run._matched_people` itself, with detection number = index + 1. Pairs are taken greedily, nearest first, one to one, within 2 × `person_radius_m` = 0.5 m.
- **Recall:** matched / visible, or None when no one is visible.
- **False:** an unmatched detection that is not within 0.5 m of an `exempt_xy` point. Exempt detections are counted apart. A sub-count gives the false detections within 0.5 m of a true person who fails `is_visible`.
- **Merge:** a detection with at least 2 visible true people within 0.5 m, of whom at most one is matched to any detection.
- **Neighbour pair:** two visible true people whose centres are less than 0.8 m apart. The pair is merged when the two are not both matched and one detection lies within 0.5 m of both. The merged share is merged pairs over neighbour pairs.
- **xy error:** the distance within each matched pair; the scorer reports the median.
- **Facing:** scored on matched pairs whose yaw is known. The error is e = |wrap(yaw_det − yaw_true)| when `facing="full"`, or min(e, 180° − e) when `facing="axis"`. A pair counts as within 45° when the error is at most 45°. The scorer reports known, within-45 and unknown counts. It always reports the axis version as well.
- **Missed people:** each missed visible person is listed with id, role, group and `lower_hidden`. `lower_hidden` means that at least 2 of the axis samples at z = 0.3, 0.5 and 0.7 m project into the image onto finite positive depth nearer than the sample by more than 0.3 m.
- **Staff behind counters:** missed people with a role who are `lower_hidden`.

**Stage A's exemption.** This reads the REPORT's "ignore blobs within 0.5 m of the target point". Stage A passes the anchors of every target in `truth.targets` that has no `person_id` as `exempt_xy`. Person targets are people and are scored as people.

## 5. Stage A evaluation and criteria
The bars are the REPORT's. Each is judged separately on E-real (42) and on E-ph (26).
- **Success:** recall ≥ 0.95 at 0.5 m, ≤ 0.10 false bodies per scene and ≤ 10% of neighbour pairs merged, in both domains.
- **Kill:** recall < 0.85, > 0.3 false bodies per scene or > 20% merged, in either domain, after the one fix.
- **Neither:** any other outcome. A domain with no neighbour pairs skips the merge bar.

**Report.** Per set (real, staffed, plain, placeholder), per domain, and pooled over the 68:
- the number of scenes and of visible people;
- recall as k/n, with a Wilson 95% interval;
- false, exempt and near-hidden counts;
- merges, and merged pairs as k/n;
- the median xy error, with n;
- axis facing as k/n;
- misses by class (staff behind counters, other staff, non-staff), with the share that are staff behind counters;
- recall by role (line member, group member, staff, other).

## 6. VLM people lists, scored like for like
**Answers.** One answer per (arm, scene, trial): the union of that key's records' people lists, for records that have one (as in `results/still-scenes/overview/phase-notes/phase1-files/a0_facing`).

**Runs** (all con):
- s19 `qwen38_low` and `glm53`, and s20 `dsv41`: the realistic arms plus their `ph` and `dev_seed0`–`2` runs.
- s23 `haiku55`, `sonnet55_low` and `gemini38_low`: realistic arms only.

**Scoring rules.**
- The scorer and exemption are the same as for blobs. Raw false counts are also given.
- Keys with no people list are counted and left out. In a sensitivity row they count as empty lists.
- Blob numbers are given once per scene, and again weighted to each model's answer keys.
- Facing: the VLM's full yaw, plus the VLM axis beside the blob axis.

## 7. Stage B arms (solver replays; no change to `src`)
**Records.**
- Primary: old wording on the realistic arms. That is s19 `qwen38_low` and `glm53`, and s20 `dsv41`, all con, with 252 records each.
- Extras: new wording, s21 `qwen38_low` and s23 `glm53`, `dsv41`, `haiku55`, `sonnet55_low` and `gemini38_low`, all con. Qwen's, GLM's and DeepSeek's people answers there are cache copies of s19/s20, so they are not new evidence on perception.
- Further extras: the placeholder C runs, s19 `ph` and `dev_seed0`–`2` (Qwen, GLM) and s20 `ph` and `dev_seed0`–`2` (DeepSeek).

**Solving.** Each solver step (`Record.solver_step`) is solved again with `H.resolve(rec, people=, constraints=)` and the recorded target, as in `results/still-scenes/replay-harness/repairs/repairs.py`. Every other record keeps its recorded verdict. A record without a people list never gets blobs. Methods never read situation or scene labels; labels are used only to tabulate.

**Terms.** Blobs are the frozen Stage A kept blobs of the record's scene, and distances are in xy.
- **Supported:** a detection with a blob within 0.5 m.
- **Unexplained:** a blob with no listed detection within 0.5 m.
- **Target blob:** a blob within 0.5 m of the record's grounded target point. It is never added, but it may support a detection or receive a relocated one.
- **Added person:** `DetectedPerson(number = max listed + 1, + 2, … in blob order, xy = blob position, yaw = None, pixel = projection of (x, y, 1.1 m), label = None)`. No merge or cut is applied; pairs closer than 0.25 m are counted.
- **Prune:** drop an unsupported detection and remove its number from every constraint's people list. The constraint itself is kept (`repairs.strip`). A detection is exempt when fewer than 3 of its 9 axis samples (z = 0.9–1.7 m) project into the image onto finite positive depth no nearer than the sample by more than 0.3 m. In other words, the depth cannot see that spot.

**Arms.**
- **B-add** (primary): add every unexplained blob that is not a target blob.
- **B-prune:** B-add, plus prune.
- **B-relocate** (data-informed, registered 17 Oct after I01): pair unsupported detections one to one with unexplained blobs whose projection at 1.0 m (`sgi.geometry.project`) lies within 40 px of the detection's recorded pixel. Pairs are taken greedily by increasing pixel distance; ties go to the lower number, then blob order. A paired detection moves to the blob's position and keeps its number, yaw, pixel and label. Then apply B-add to the blobs still unexplained.
- **B-relocate-prune:** B-relocate, then prune the detections still unsupported.

**Fixable offline.** A recorded failure that I01's P arm flips (`rows_old_*` and `rows_new_*`). For placeholder runs, P is computed with the unchanged `repairs.solver_row`.

## 8. Stage B criteria and report
The bars are the REPORT's. They apply to B-add, old wording, realistic arms: 252 records per model (lines 90, groups 54, nobody near 108).

**Success** requires all of:
- GLM lines ≥ 47 of 90 (recorded 42).
- DeepSeek lines ≥ 71 of 90 (recorded 68).
- Qwen, GLM and DeepSeek each lose no more than 1 net on non-line records.
- Qwen's total stays ≥ 234.
- At least half of the fixable line failures are recovered, pooled over the three models. The fixable count is 2 + 15 + 8 = 25, so at least 13.

**Kill** fires on any of:
- GLM's line gain is ≤ 2 and DeepSeek's line gain is ≤ 1.
- Total breaks ≥ total flips over the three models' 756 records.
- More than 60% of line failures are unfixable and the live check gains fewer than 2 of 10. The live check is not run in this batch, so this condition cannot fire.

**Known shares.** From I01, 21 of 46 line failures are unfixable (46%). Over all goal-step failures it is 69 of 102 (68%). A line failure is as defined in RERUN R01: a waiting-line solver step with the right target that fails because the place is outside the region or unreachable. I chose the line population knowing these shares, because the REPORT's bars and its "Worse than expected" text are about lines.

**Prediction** (inference): I01's R3 flips only 2 of 29 GLM and 2 of 15 DeepSeek line failures, so B-add will probably trip the first kill bar.

**Report.** Per arm and model:
- successes of 252, with flips and breaks against the recorded C;
- lines, groups and nobody near;
- fixable failures recovered, as k/n;
- the distinct people answers and layouts behind the flips;
- the step-0 reproduction count.

New wording and placeholder runs are reported as extras.

## 9. Order and honesty
1. This note and its hash.
2. The scorer, with self-checks on fit data: true people given as detections must give recall 1 and 0 false.
3. The fresh seeds.
4. The fit grid, `fit_selection.json` and `FIT.sha256`.
5. The fix addendum, if one is needed.
6. Stage A evaluation, the VLM table and Stage B, each scored once.

Every number comes with its n. Anything not declared here is labelled post hoc.
