# I09 fix: diagnosis of the false people and misses (fit-train, seeds 110–115)

Paths below are relative to `results/still-scenes/people-finder/early-detector/fix/`.

## 0. At a glance

| | Fit-train, seeds 110–115 (84 images) | A1 development E-real (42) | Confirmation (182, published aggregates) |
|---|---|---|---|
| Recall | 226/238 = 0.950 | 116/118 = 0.983 | 510/520 = 0.981 |
| Line members | 92/96 | 48/48 | 207/208 |
| Counter staff | 26/34 | 14/16 | 69/78 |
| False people per image | **17/84 = 0.202** | 3/42 = 0.071 | **44/182 = 0.242** |
| Exempt detections | 5 | 0 | 6 |
| Median position error | 0.059 m | 0.053 m | 0.061 m |
| Non-profile within 45 deg | 125/128 (median 12.6 deg) | 59/59 | 278/287 |
| Profile front/back correct | 98/98 | 57/57 | 223/223 |

- **The fit renders reproduce the confirmation's failure.** 0.202 false people per image, against 0.242 on the confirmation, with the same mix of causes. So the problem can be studied without touching test data.
- **Four mechanisms explain all 17 false people and all 12 misses.**

| Mechanism | False | Exempt | Misses | Confirmation class (aggregate) |
|---|---|---|---|---|
| M1. Occluded torso: the depth median lands on an object in front of a counter staff member (an espresso machine, a till, the counter end) | 9: 8 paired with a staff miss, 1 second detection beside a matched staff member | 0 | 8 staff | A: 16 (9 paired, 7 second detections) |
| M2. The 0.50 m duplicate radius drops a line neighbour | 0 | 0 | 4 line members | the 1 line miss |
| M3. Staff behind glass: the rendered depth sees the glass, the RGB sees the person, and the scorer's visibility rule calls the person hidden | 4 | 5 | 0 | B: 9; C: 4 |
| M4. Small far boxes on pictures of people (poster pictograms, figures seen through windows) | 4 | 0 | 0 | D: 15 |

- **Counts in more detail.**
  - The 17 false people by the confirmation's post hoc classes (`posthoc_false.py` rules, applied unchanged): A 9, B 4, C 0, D 4.
  - The 5 exempt detections are all M3: four are class B and one is class C at 0.502 m.
  - Facing is not a problem here.
- **Indicative what-ifs** (section 9; not a choice). Re-lifting occluded torsos from the head keypoints would remove all 9 M1 false people and all 8 staff misses, and it never fires on a matched person in any data set looked at.
  - Combined with an image-space duplicate test and A1's flat test, fit-train goes to 238/238 recall and 5/84 = 0.060 false people per image.
  - The same combination leaves A1's placeholder fit data unchanged and gives A1's development scenes 118/118 with 0 false.
  - What remains is M3, which is a scorer question, plus one tiny poster box.

## 1. What was done

1. **Manifest.** `manifest_fit.py` built `manifest_fit.json` (sha in `MANIFEST_FIT.sha256`): 140 images, 84 fit-train and 56 fit-validate.
   - Every scene folder was checked by the fit path guard (`pathguard_fit.py`: only `results/still-scenes/scene-seeds/phase3/realistic_train/seed110`–`119/eval/`), then by `H.guard_scene`, before its `scene.json` was read.
   - No banned name exists under the seeds folder.
2. **Detector and pose pass.** `sidecar/infer_fit.py` follows `results/still-scenes/people-finder/early-detector/phase2/confirm/sidecar/infer_confirm.py` and imports A1's `sidecar/infer.py` by path, with its sha256 asserted.
   - It calls A1's `run_pass("eval", …, ("rtdetr_v2_r50vd",))` unchanged. Only `guard` (the fit path guard plus manifest membership) and `wait_for_gpu` (≥ 3 GB GPU free and ≥ 3 GB RAM available) were replaced.
   - It loaded the pinned snapshots only: RT-DETRv2 `2824940…`, ViTPose `a93ac0c…`. It ran with `HF_HUB_OFFLINE=1`, `-I`, `nice -n 10` and 2 BLAS threads.
   - The pass took 18:34:51–18:35:08 for 140 images, with 15 MiB GPU in use beforehand and a peak of 799 MiB allocated.
   - It wrote 2,234 boxes (≥ 0.05) and 790 pose sets.
   - The sidecar's `pip freeze` (160 lines, sha `20f344bf…`, the same as at the confirmation) was unchanged. Hashes are in `RAW_FIT.sha256`; the log is `sidecar_pass.log`.
3. **Scoring.** `score_fit.py` scored fit-train with A1's frozen method and I03's scorer, unchanged.
   - Method: `P.candidates` and A1's steps 2–7, traced box by box and asserted equal to `P.run_image`; setting from `chosen.json`, facing F5.
   - Scorer: `C.score_scene` → `PS.score_people`, facing "full", `exempt_points(truth)`, default thresholds.
   - Outputs: `fit_train_rows.jsonl`, `fit_train_people.jsonl`, `fit_train_dets.jsonl` (every candidate ≥ 0.25 and its fate), `fit_train_summary.json`.
4. **Diagnosis.** `diagnose.py` writes `diag_dets.jsonl`, `diag_hidden.jsonl`, `diag_pairs.jsonl` and `diagnose_tables.txt`. Every table below is in `diagnose_tables.txt`.
   - From image, depth and keypoints only, it measures per box: metric box height, box-bottom height, flat share, head and shoulder depth, depth beside and below the box, and the shares of the torso region.
   - Using truth (diagnosis only) it adds: the post hoc class, which rendered part the region's pixels lie on (`render_job.json` parts and models, glass flagged), the displacement split along and across the camera ray, and every visibility sample of every hidden-by-rule person.
5. **What-ifs.** `whatif.py` → `whatif.txt`. Indicative what-ifs on fit-train, on A1's fit data (F-real 21 worked examples, F-ph 268 placeholder scenes; `fitdata.load`) and on A1's 42 development E-real scenes (through the harness, as `c2_repro.py` does).
   - Each base row reproduces A1's recorded numbers: F-real 36/39 and 3 false; development 116/118 and 3 false.
6. **Figures.** `figs.py` → `figs/` (fit-train only; the script refuses other images). Left panel RGB, right panel depth (near is bright).
   - Box colours: green = kept and matched, red = kept and false, orange = exempt, grey = dropped. Cyan is A1's torso hull, yellow dots are keypoints with confidence ≥ 0.30.
   - Circles mark each true person's labelled vertical axis (0–1.7 m): magenta for staff, white for others.

## 2. How A1 lifts a box (confirmed from `results/still-scenes/people-finder/early-detector/phase1c/a1/pipeline.py`)

- **Torso region** (`box_features`, lines 186–196).
  - When all four of left and right shoulder and hip have keypoint confidence ≥ 0.30: the pixels whose centres lie inside the convex hull of those four keypoints. This requires at least 20 pixels with valid depth.
  - Otherwise: the box's central rectangle (u 30–70 %, v 20–50 % of the box).
- **Statistic** (lines 202–210).
  - Z_med is the plain median of the camera-frame depth Z over the region's valid pixels (finite and > 0).
  - Pixels within ±0.30 m of Z_med are kept; at least 10 are needed.
  - The position `body_xy` is the median of the kept pixels' world x and, separately, of their world y.
- **Then** (`people_for`, lines 249–273):
  - the position is pushed 0.20 m away from the camera along the horizontal bearing;
  - the height band 1.00–2.30 m is tested on z_top, the box top lifted at Z_med;
  - in descending score order, a box whose pushed position lies within 0.50 m of an already kept one is dropped as a duplicate. The test is purely in world xy.

No step looks at whether the region's depth belongs to the person. A region that is mostly occluder gives the occluder's depth.

## 3. M1: occluded torso (displaced counter staff)

Table 3.1. Each class-A false detection on fit-train.

| Seed, row, number | Score | Box h px (bottom row) | Hip confidence L/R | Nearest head depth − Z_med | Share of region nearer than the staff axis − 0.4 m | What the kept pixels hit (top) | Z_med − staff axis Z | Offset before → after push | Pairs with |
|---|---|---|---|---|---|---|---|---|---|
| 110 eval_02 n1 | 0.770 | 72 (187) | 0.45/0.41 | +0.61 | 0.64 | till (plastic_dark) | −0.77 | 0.82 → 0.62 | staff miss |
| 110 eval_06 n2 | 0.890 | 237 (341) | 0.77/0.81 | +0.60 | 0.68 | machine (steel), counter body, counter top | −0.76 | 0.83 → 0.63 | staff miss |
| 110 eval_07 n3 | 0.685 | 142 (254) | 0.63/0.64 | +0.61 | 0.73 | plastic_dark, counter top, counter body | −0.76 | 0.84 → 0.64 | staff miss |
| 111 eval_06 n2 | 0.900 | 236 (340) | 0.79/0.81 | +0.60 | 0.67 | machine (steel), counter | −0.76 | 0.83 → 0.63 | staff miss |
| 111 eval_07 n3 | 0.863 | 226 (340) | 0.83/0.74 | +0.64 | 0.85 | machine (steel), counter | −0.77 | 0.83 → 0.63 | staff miss |
| 111 eval_08 n5 | 0.774 | 176 (297) | 0.75/0.75 | +0.63 | 0.83 | plastic_dark, counter body | −0.76 | 0.84 → 0.64 | staff miss |
| 112 eval_07 n3 | 0.776 | 220 (333) | 0.80/0.84 | +0.63 | 0.94 | machine (steel), counter | −0.77 | 0.83 → 0.63 | staff miss |
| 114 eval_07 n3 | 0.874 | 233 (340) | 0.80/0.82 | +0.61 | 0.89 | machine (steel), counter | −0.77 | 0.84 → 0.64 | staff miss |
| 114 eval_09 n8 | 0.366 | 167 (283) | 0.69/0.76 | +0.61 | 0.98 | machine (steel), counter | −0.76 | 0.85 → 0.65 | second detection (s0 matched by n7) |

Evidence:
- **All 9 use the keypoint hull.** ViTPose places both hips at confidence 0.41–0.84, so the hull is used even though the hips are behind the counter.
  - Knees are placed in 8 of 9 (0.46–0.87) and ankles in 7 of 9 (0.46–0.83), through the counter.
  - Seven boxes reach rows 283–341: the detector draws a full-height (amodal) box down over the counter.
  - The hips' depth lies 0.59–0.63 m nearer than the nearest head keypoint's.
- **64–98 % of the hull lies on something in front of the staff member.** It is an espresso machine or a till on the counter, or the counter's end and top.
  - So the median lands on that object: Z_med lies 0.76–0.77 m in front of the staff member's axis.
  - The ±0.30 m band then keeps only the object's pixels, at world height 0.96–1.39 m.
- **The displacement lies along the camera ray.** It is 0.82–0.85 m toward the camera, with at most 0.04 m sideways. The 0.20 m push takes off 0.20 m, which leaves 0.62–0.65 m. This is the confirmation's "0.63–0.66 m off".
- **The head is above the occluder, and its depth is right.** The head keypoints' depth lies 0.09–0.14 m in front of the axis, about the face surface.
  - Nearest head depth − Z_med is +0.60 to +0.64 for all 9 class-A boxes.
  - Over all 226 matched boxes it is −0.20 to +0.15.
- **Contrast with the matched staff.** The 24 matched staff members whose lower body is hidden have only 0.00–0.44 of their region on a nearer surface, so the median stays on the person; they are placed 0.02–0.13 m off.
  - In `figs/m_112_06.png` the camera–staff geometry is the same as in `figs/a_110_06.png`, but a narrower avatar keeps the hull clear of the machine.
  - So whether a staff member is displaced turns on the dressing and the avatar's width, not on the geometry alone.
- **Second detections and the duplicate rule** (`figs/a_114_09.png`). One staff member got two boxes:
  - box 6 (0.64): head and shoulders above the counter; no confident hips, so the central-rectangle region is used, which lies on the person; matched, 0.11 m off;
  - box 8 (0.37): full height; its hull is 98 % counter and machine.
  - Their pushed positions are 0.76 m apart, more than the 0.50 m radius, so both were kept.
  - In the image, box 6 lies 100 % inside box 8 and their nose keypoints are 1.7 px apart. The duplicate rule never looks at the image.
- **Three more boxes of the same staff members were removed as duplicates** (d 0.00–0.04 m). They had the same box top and the same hull, so they were displaced the same way.

![M1: espresso machine and counter end cover the hull](figs/a_110_06.png)
*`figs/a_110_06.png`. Seed 110, `eval_06`. The red box (0.89, full height) has its hull (cyan) half on the machine and counter end; Z_med lands there. The staff axis (magenta) is behind it.*

![M1: till in front of the torso](figs/a_110_02.png)
*`figs/a_110_02.png`. Seed 110, `eval_02`. A till on the counter covers the lower hull, and the depth shows it as the near (bright) block.*

![M1: second detection](figs/a_114_09.png)
*`figs/a_114_09.png`. Seed 114, `eval_09`. The green head-and-shoulders box is matched. The red full-height box has its hips and legs drawn through the counter; it lands 0.65 m off and is 0.76 m from the green one, so it is not removed as a duplicate.*

## 4. M2: the duplicate radius drops tight line neighbours

Table 4.1. Each duplicate drop on fit-train, and whether the two boxes are the same person. The identity is the true person whose axis projects into the box with the nearest depth; for diagnosis only.

| Kept → dropped | World distance | IoU | Containment (intersection / smaller box) | Nose distance px | Same person |
|---|---|---|---|---|---|
| 110 eval_09: p3 (0.94) → p4 (0.93) | 0.45 m | 0.00 | 0.00 | 55 | **no**: p4 missed |
| 110 eval_09: p2 (0.93) → p1 (0.93) | 0.48 m | 0.00 | 0.00 | 50 | **no**: p1 missed |
| 111 eval_13: p0 (0.93) → p1 (0.93) | 0.47 m | 0.00 | 0.00 | 60 | **no**: p1 missed |
| 115 eval_08: p1 (0.93) → p0 (0.93) | 0.48 m | 0.00 | 0.00 | 50 | **no**: p0 missed |
| 7 staff duplicates (110 eval_07, 111 eval_08, 112 eval_07, 113 eval_02, 114 eval_09 ×2) and 1 hidden staff member (114 eval_02) | 0.00–0.46 m | 0.23–0.62 | 0.99–1.00 | 0.1–2.0 | yes |
| 2 small head boxes inside a group member's box (110 eval_11, 112 eval_12) | 0.16–0.18 m | 0.02 | 0.83–0.90 | — | yes (viewed, `figs/x_110_11.png`: the same person's head) |

- **What happens.** The four missed line members stand in profile in lines whose true spacing is 0.50–0.53 m. Their lifted positions are 0.45–0.48 m apart, so the second box falls inside the 0.50 m radius and is dropped. The boxes are disjoint in the image (`figs/l_110_09.png`).
- **Image overlap separates the two cases on fit-train.**
  - Every same-person pair had containment ≥ 0.83, and ≥ 0.99 for those with a nose keypoint, with noses ≤ 2 px apart.
  - Every kept pair of different people within 1.2 m had containment ≤ 0.34 and noses ≥ 15 px apart.
  - World distance cannot separate them: the same person reached 0.76 m (the M1 second detection), and different people came within 0.45 m.

![M2: line neighbours dropped as duplicates](figs/l_110_09.png)
*`figs/l_110_09.png`. Seed 110, `eval_09`. The grey boxes (p4 and p1, 0.93 each) are separate people, dropped as duplicates of their neighbours' detections at 0.45 and 0.48 m.*

## 5. M3: staff behind glass ("hidden by rule")

- **What the visibility rule tests** (`sgi.gt.derive.is_visible`).
  - It takes 9 samples on the person's labelled vertical axis, at 0.9, 1.0, …, 1.7 m.
  - A sample is seen when it projects into the image onto a depth within 0.30 m of its own camera depth.
  - A person is visible when at least 3 samples are seen. The depth is the renderer's exact depth.

Table 5.1. Every fit-train person who fails `is_visible` with the 1.1 m mark in the picture. All 9 are counter staff.

| Scene | Samples seen | What the unseen samples hit | Depth − sample depth | Detection near them | Score, box h | Distance after push |
|---|---|---|---|---|---|---|
| 110 eval_09 s0 | 0 of 9 | glass 7, booth counter body 2 | −0.80 to −0.54 m | n5, false | 0.823, 64 px | 0.48 m |
| 111 eval_09 s0 | 0 | glass 7, counter body 2 | −0.80 to −0.54 | n7, exempt | 0.856, 75 px | 0.38 |
| 113 eval_07 s0 | 0 | glass 7, counter 2 | −0.81 to −0.63 | n3, false | 0.877, 95 px | 0.49 |
| 114 eval_02 s0 | 0 | glass 5, counter body 2, paper sign 2 | −0.87 to −0.38 | n1, exempt (class C) | 0.878, 76 px | 0.50 |
| 114 eval_08 s0 | 0 | glass 7, counter body 2 | −0.80 to −0.58 | n5, exempt | 0.861, 77 px | 0.38 |
| 115 eval_02 s0 | 0 | glass 5, counter body 2, paper sign 2 | −0.87 to −0.38 | n1, exempt | 0.832, 50 px | 0.21 |
| 115 eval_08 s0 | 0 | glass 7, counter body 2 | −0.80 to −0.58 | n4, false | 0.842, 68 px | 0.43 |
| 115 eval_09 s0 | 0 | glass 7, counter body 2 | −0.80 to −0.54 | n7, false | 0.791, 66 px | 0.40 |
| 115 eval_13 s0 | 0 | glass 7, counter body 2 | −0.81 to −0.60 | n4, exempt | 0.844, 89 px | 0.38 |

- **Where these staff stand.** All 9 are in glass-fronted booths (ticket-booth dressing, glass panes `sx` 0.01 m at 1.1–1.8 m height).
- **Why the rule calls them hidden.** The renderer's depth treats the glass as opaque, so every sample sees the glass or the booth's counter, 0.38–0.87 m in front of the sample. In the RGB the person is plainly visible through the glass (`figs/b_115_02.png`, `figs/b_110_09.png`), and the depth image shows no person silhouette at all.
- **What the detector does with them.**
  - The detector boxes the person (0.79–0.88, head and shoulders).
  - 82–100 % of the torso region lies more than 0.4 m in front of the axis, mostly on the glass (78–84 % of the kept pixels in 8 of 9). So the lift lands on the glass: 0.41–0.70 m in front of the axis, and 0.21–0.50 m after the push.
  - The head is behind the same glass, so nearest head depth − Z_med is only −0.16 to +0.19. The M1 signal does not fire here.
- **Whether the detection counts as false is arbitrary.** It depends on whether the glass point happens to lie within 0.5 m of the anchor of a target with no person:
  - 5 of 9 do and are exempt;
  - 4 do not and count as false.
- **Why the detector is not wrong.** The person is there and can be seen. The conflict is between the exact-depth visibility rule and the RGB. In the brief's words this is a scorer question.
  - (Inference) A real depth camera often returns no depth, or the depth behind, on clear glass, so opaque-glass depth is a renderer property, not a scene property.
- **If the scorer counted these staff as visible** (estimate): 8 of the 9 detections lie within the 0.5 m match radius (the ninth is at 0.502 m). That would give about +8 matched, −4 false and −4 exempt.

![M3: staff behind a glass pane](figs/b_115_02.png)
*`figs/b_115_02.png`. Seed 115, `eval_02`. The person is seen through the booth's glass. The depth shows only the flat glass. Orange = exempt.*

![M3: staff behind a glass pane, counted false](figs/b_110_09.png)
*`figs/b_110_09.png`. Seed 110, `eval_09`. The same mechanism; this detection is 0.48 m from the labelled point and counts as false.*

## 6. M4: small far boxes on pictures of people

Table 6.1. Each class-D false detection.

| Scene, number | Score | Box h px | Z_med | Metric height (h·Z/fx) | Box bottom above floor | Flat share | Depth beside the box − Z_med | Depth below − Z_med | Nearest person | What it is (viewed) |
|---|---|---|---|---|---|---|---|---|---|---|
| 110 eval_05 n2 | 0.294 | 9 | 7.24 | 0.12 m | 1.37 m | — (< 30 depth points) | −0.02 | +0.01 | 3.48 m | person pictogram on a poster on a noticeboard |
| 114 eval_05 n2 | 0.293 | 10 | 7.22 | 0.14 | 1.91 | 1.00 | −0.02 | +0.01 | 3.49 | the same kind of poster pictogram |
| 114 eval_07 n4 | 0.261 | 17 | 7.96 | 0.25 | 1.19 | 1.00 | 0.00 | +0.03 | 4.08 | a shape in the outdoor view through a window, beside a line member's head |
| 111 eval_10 n5 | 0.283 | 34 | 13.19 | 0.81 | 1.14 | 1.00 | +0.01 | +0.06 | 3.33 | a figure in a window view (depth is the far backdrop) |

- **What these boxes look like in depth.** The depth inside each box equals the depth around it: no silhouette, a flat surface, nothing nearer below a box that floats 1.1–1.9 m above the floor.
- **The matched people, for comparison.**
  - Score ≥ 0.56 (≥ 0.85 except staff whose lower body is hidden).
  - Flat share ≤ 0.53.
  - Every matched box floating more than 0.5 m above the floor has something at least 0.23 m nearer just below it: the counter.
- **Two simple cut-offs fail on staff behind counters.** Head-and-shoulders boxes of such staff are as short as 0.33–0.37 m (2 matched) and 37 px tall (3 matched under 40 px), so metric height and pixel height also catch real people.
- **Fewer of these than at the confirmation.** The confirmation's class D was 15 boxes, 9–57 px tall, scoring 0.25–0.53 (median 0.29), about 0.082 per image. Here there are 4, about 0.048 per image.

![M4: poster pictogram](figs/d_110_05.png)
*`figs/d_110_05.png`. A 9 px box on a poster pictogram; the depth is the flat wall.*

![M4: window figure](figs/d_111_10.png)
*`figs/d_111_10.png`. A figure in a window view; the box's depth is the far backdrop.*

## 7. Hidden structure: the camera–staff geometry repeats across seeds

`diagnose_tables.txt` ("STAFF CAMERA GEOMETRY") lists the staff member's projected (u, v, Z) at 1.2 m per row and seed. In the waiting-line rows it repeats exactly:
- `eval_06`: (480, 183, 4.19) in 4 of the 5 seeds with s0 in view;
- `eval_07`: (520 or 120, 183, 4.19), mirrored, in 5 of 5;
- `eval_09`: (553, 172, 5.71) ×4 and (94, 170, 6.03) ×2;
- `eval_08` and `eval_13`: similar.

The seed changes the room, the avatars and the stall dressing: an espresso counter, a counter with a till, or a glass booth. The same geometry then gives M1 with a machine, M3 with glass, or a clean match.
- The 9 M1 cases come from 5 geometries, and the 9 M3 cases from 5.
- (Inference) Fit-validate (116–119) and the spent test seeds very likely share these templates. A fix tuned on fit-train should therefore carry over to them, but that says little about stall geometries the generator does not make.

## 8. A side finding: head keypoints in profile and back views

On 2 of the 226 matched boxes (`figs/h_110_12.png`, `figs/h_113_09.png`, profile and back views), ViTPose puts the nose and eye keypoints on the edge of the head, over the background. The median of the head keypoints' depth then lies 3.9–4.7 m behind the torso.

The nearest head keypoint's depth does not have this problem: −0.20 to +0.15 m on all matched boxes. Any rule that uses head depth must use the nearest head keypoint, or something equally robust.

## 9. Candidate fixes per mechanism, with expected effect (not chosen)

The numbers are from `whatif.txt`. Each starts from A1's own candidates, changes one step, and scores with I03's scorer unchanged. They are indicative, not tuned or frozen; fit-validate was not read.

Table 9.1. Indicative what-ifs.

| What-if | Fit-train (84): recall; false | A1 F-real (21): recall; false | A1 F-ph placeholders (268): recall; false | A1 development E-real (42): recall; false |
|---|---|---|---|---|
| A1 as frozen | 226/238 (line 92/96, staff 26/34); 17 (0.202) | 36/39; 3 | 513/585 = 0.877; 0 | 116/118; 3 (0.071) |
| 1a re-lift occluded torsos from the head | 234/238 (staff 34/34); 8 (0.095) | 39/39; 1 | unchanged (never fires) | 118/118; 0 |
| 1e image-space duplicate added (containment ≥ 0.80 within 1.0 m) | 226/238; 16 | unchanged | unchanged | **115/118** (keeps the displaced box, drops the right one); 3 |
| 2a image-space duplicate only (containment ≥ 0.50 within 1.0 m) | 230/238 (line 96/96); 16 | 36/39; **5** (the glass ghost box returns) | unchanged | 115/118; 3 |
| 2b d_dup 0.30 m | 230/238 (line 96/96); 17 (exempt +1) | 36/39; **5** | unchanged | 116/118; 3 |
| 4a A1's flat test on | 226/238; 14 | unchanged | unchanged | unchanged |
| 4c score threshold 0.35 | 226/238; 13 | unchanged | **483/585 = 0.826** | **115/118**; 3 |
| 1a + 1e | 234/238; 8 | 39/39; 1 | unchanged | 118/118; 0 |
| 1a + 2a | 238/238; 8 (0.095) | 39/39; 3 | unchanged | 118/118; 0 |
| 1a + 2a + 4a | **238/238; 5 (0.060)** | 39/39; 1 | unchanged | 118/118; 0 |
| 1a + 2a + 4c | 238/238; 4 (0.048) | 39/39; 3 | 0.826 | 118/118; 0 |

The candidates for each mechanism follow.

**M1, occluded torso**
- **1a. Re-lift occluded torsos from the head.**
  - Rule: when the nearest confident head keypoint (nose, eyes, ears; 5×5 depth) lies more than 0.30 m behind Z_med, take the position from the head keypoints within 0.30 m of that depth, then push as usual.
  - Uses keypoints and depth only.
  - Expected effect:
    - fit-train: all 9 M1 false people and all 8 staff misses gone;
    - A1 F-real: 3 → 1 false, 39/39;
    - development: 3 → 0 false, 118/118;
    - placeholders: never fires.
  - Margin: 0.60–0.64 m for M1 against ≤ 0.15 m for matched boxes. On all four data sets the rule fired only on boxes that were unmatched before.
  - Risks: a head partly covered by something nearer is safe, since the rule only fires when the head is farther than the torso. Untested on real sensor depth, where depth at head edges is noisy. The head-median version would misfire (section 8).
- **1b. A depth-consistent hull.** Drop hip keypoints whose depth is more than 0.3 m nearer than the head or shoulders, and use the region above the occluder's edge. Expected to act like 1a, with more parameters. Not measured.
- **1c. A different statistic.** For example the farthest well-supported depth mode in the region instead of the median. Not measured. Risk: profile hulls can include background.
- **1e. An image-space duplicate test** (containment ≥ 0.80 and world distance ≤ 1.0 m). This removes M1 second detections: 1 here, 7 in the confirmation's aggregate.
  - Alone it is harmful: on the development scenes it kept a higher-scoring displaced box and dropped the correctly placed one (staff 14 → 13).
  - Safe only after 1a, or with a rule that keeps the member whose depth agrees with its head.

**M2, line merges**
- **2a. Duplicates decided in image space only** (containment ≥ 0.50 within 1.0 m). Line members go 92 → 96/96 and merged pairs 2 → 0.
  - Cost: A1 F-real +2 false. The shape behind a booth's frosted side glass is no longer merged by accident.
- **2b. d_dup 0.30 m** (A1's other grid value). The same gain and the same F-real cost.
- **Pairing.** Either one pairs with 4a: with 1a + 2a + 4a, F-real false is 1.

**M3, staff behind glass**
- **3b. Suppress the detection on the detector side.**
  - With the flat test: 1 of 9 removed (flat share 0.46–0.98).
  - With a "no silhouette in depth" test (side contrast −0.28 to −0.13 m): would remove them more fully.
  - **Flagged:** this would delete a person the robot can see, to please the scorer.
- **3c. Bold, unproven idea.** When a box's torso region is one plane with no person silhouette, place the person behind that plane rather than on it. Nothing here supports a particular offset.

**M4, small far boxes**
- **4a. A1's flat test** (share ≥ 0.80 within 0.04 m of the torso plane).
  - Removes 3 of 4. The 9 px box has fewer than 30 depth points, and the test then says "not flat".
  - Loses no matched person on any of the four data sets.
  - Also removes 1 M3 box and the F-real glass shape.
- **4b. A floating-box test.**
  - Rule: the box bottom is more than 0.5 m above the floor at Z_med, the box does not touch the image's bottom edge, and nothing at least 0.15 m nearer lies just below the box.
  - Fit-train: 4 of 4 removed, 0 of 226 matched lost.
  - The margin is thin: the closest matched staff member has the counter only 0.23 m nearer below.
  - Not run on A1's data.
- **4c. Score threshold 0.35.**
  - Fit-train: 4 of 4 removed.
  - Costs placeholder recall 0.877 → 0.826 and 1 development staff member.
  - The confirmation's class-D boxes scored up to 0.53, so a 0.35 cut would not have caught them all.
- **4d. Minimum metric height of 0.5 m.**
  - Removes 3 of 4, and 2 matched head-and-shoulders staff (0.33–0.37 m).
  - The evidence argues against it.

Two more notes:
- **Facing and positions.** Facing (F5) and median position error do not move in any what-if. Re-lifting a staff member also brings their facing into the counts: non-profile 125/128 → 132/136 within 45 deg on fit-train.
- **What remains after 1a + 2a + 4a on fit-train.** 5 false: the 4 glass detections of M3 that count as false, and the 9 px poster box.

## 10. Limits and deviations

Limits:
- Fit-train is 84 images. Its causes recur through repeated geometry (section 7), so the independent cases are fewer than the counts suggest.
- The what-ifs are indicative. They are measured on the same data that suggested them, and on A1's development scenes, which A1 used for its verdict.
- The part labels in section 3 are approximate: a point counts as on a part when it lies inside that part's box, enlarged by 3 cm. The identity of a box is a heuristic.

Deviations:
1. Before the scripts existed, I read one fit scene's `scene.json` and `render_job.json` (seed 110, `eval_07` and `eval_06`) with plain JSON reads, without `H.guard_scene`. These are fit data, the path holds no banned name, and every scripted read guards first. `render_job.json` has no harness reader; the scripts read it only after `H.guard_scene` passes.
2. `whatif.py` goes beyond a pure diagnosis, to give each candidate fix an expected effect. It is labelled indicative, and nothing is chosen or frozen.
4. I read the source of `results/still-scenes/people-finder/early-detector/phase2/confirm/posthoc_false.py` to reuse its class rules exactly. I did not read `posthoc_false.txt`, the confirmation's rows, or anything from seeds 200–219.
5. The detector pass covered fit-validate, because the brief says all new renders. Its 56 images were neither scored nor viewed.

Files (all in `results/still-scenes/people-finder/early-detector/fix/`):
- scripts: `pathguard_fit.py`, `fitlib.py`, `manifest_fit.py`, `sidecar/infer_fit.py`, `score_fit.py`, `diagnose.py`, `whatif.py`, `figs.py`;
- data: `manifest_fit.json`, `raw_fit/`, `fit_train_*.jsonl`, `fit_train_summary.json`, `diag_*.jsonl`;
- tables: `diagnose_tables.txt`, `whatif.txt`;
- figures: `figs/`;
- hashes: `MANIFEST_FIT.sha256`, `RAW_FIT.sha256`, `DIAG.sha256`.
