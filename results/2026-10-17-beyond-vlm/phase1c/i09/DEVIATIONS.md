# I09 A1: deviations and clarifications (part 1)

## D1. File locations (deviation in paths only, no change in method)
The dispatch puts every A1 file under `a1/`. The note's paths map as follows; all are relative to `phase1c/i09/`.

| Note | Used |
|---|---|
| `sidecar/infer.py` | `a1/sidecar/infer.py` |
| `manifest_fit.json` | `a1/manifest_fit.json` |
| `stageA/fit_grid.txt` | `a1/fit_grid.txt` |
| `stageA/fit_rows.jsonl` | `a1/fit_rows.jsonl` (192 setting rows), plus `a1/fit_people_chosen.jsonl` (per person, chosen setting) |
| `stageA/chosen.json` | `a1/chosen.json` |
| `stageA/FIT.sha256` | `a1/FIT.sha256`, with an identical copy at `a1/CHOSEN.sha256` (the dispatch's name). Both are verified before `manifest_eval.json` is written. |
| `manifest_eval.json` | `a1/manifest_eval.json` |

## D2. Evaluation pass runs the chosen detector only
The dispatch allows both detectors on evaluation images. The note (section 2, order guard) allows only the detector in `chosen.json`, and the note governs, so only that detector runs.

## Clarifications
These are details the note leaves open. I fixed them before the fit pass, from the note's text and the probe's API facts only.

**C1. Detector latency** (tie-break in section 5). Per image and per detector, the time runs from the decoded in-memory PIL image through pre-processing, forward and `post_process_object_detection` (threshold 0.05), until the boxes are on the CPU, with a CUDA synchronise. It is measured in the fit pass. Each detector first gets 5 warm-up passes on the first 5 F-real images in `fit_refs()` order. The median is then taken over all 289 fit images.

**C2. Fit-pass pose.** ViTPose runs separately on each detector's own boxes with score ≥ 0.10, as one batch per (image, detector).

**C3. Boxes** are the post-processed absolute xyxy values, unclipped. Pixel sets derived from a box are clipped to the image.

**C4. Pixel sets.** These use pixel centres (c + 0.5, r + 0.5).
- The hull region includes its boundary. A hull with fewer than 3 distinct vertices holds no pixels, so the box region is used instead.
- The box central region is the set of centres with x0 + 0.3w ≤ u ≤ x0 + 0.7w and y0 + 0.2h ≤ v ≤ y0 + 0.5h.
- Valid depth is finite and > 0 (`sgi.geometry._valid_depth`).
- Lifting follows `sgi.geometry.lift_all` arithmetic at the pixel centres.

**C5. Torso pixel** (output). This is (median u, median v) of step 2's kept pixel centres.

**C6. Flat test.**
- Q is the set of pixels whose centres lie inside the box (clipped), whose row and column indices are both even, and whose depth is valid.
- The plane is fitted to step 2's kept points in world coordinates: the centred points' SVD, with the normal as the right singular vector of the least singular value.

**C7. Duplicates** compare pushed ground points.

**C8. F3 and F4 bearings** use the output (pushed) position. A push along the camera ray changes neither bearing.

**C9. F2** takes the mean of the usable pair vectors, which is a single vector when only one pair is usable. It gives None when no pair is usable or when the mean is shorter than 0.05 m.

**C10. Keypoint use.**
- A keypoint is "used" when its score is ≥ 0.30, `sgi.geometry.lift(depth, cam, u, v, window=5)` is not None, and |Z_kp − Z_med| ≤ 0.40 m. Z_kp is the window median depth (`sgi.geometry._depth_at`, the value `lift` uses).
- The edge-on test needs both shoulders confident, with no depth condition. It uses ViTPose's u values, Z_med and the camera's fx.
- Ears for u_ref are taken when confident (score ≥ 0.30).

**C11. Unit check (fit data).** A None F1 yaw among the selected people counts as not under 90 deg. The share counting None as failure is the criterion; the share among known yaws is also printed.

**C12. Facing selection metric.** Over the chosen setting's scorer-matched fit people: within 45 deg of full yaw, with unknown counted as wrong. "Pooled" sums the two fit domains.

**C13. View class.** The angle is |wrap(bearing(true person → camera xy) − true yaw)|, with camera xy = `T_world_camera[:2, 3]` (as in `a0_facing.py`). The classes are frontal < 45, oblique [45, 75), profile [75, 105] and back > 105.

**C14. Fit check of section 10's bars** (one-general-fix trigger). The definitions are those of the evaluation: unknown counts as wrong, and as 180 deg in medians. They are applied to the chosen setting and variant, per fit domain. Line members are `people_scorer.role_classes` 'line'.

**C15. Runtime.**
- *Sidecar.* The time runs from the decoded in-memory PIL image to keypoints as numpy arrays on the CPU, after `torch.cuda.synchronize()`. File read, sha256 and decode are reported separately and not counted.
- *Project side.* Steps 1–7 plus the chosen facing variant, on boxes ≥ τ*, with depth, camera, boxes and keypoints already in memory, and 2 BLAS/OMP threads.
- *Warm-ups.* Both sides warm up on the same first 5 F-real images.
- *Keypoint agreement.* For each E-real image, the timing pass's keypoints for every box ≥ τ* are compared with the raw evaluation pass's for the same box. Boxes are paired by index, and the box coordinates are asserted equal within 1e-3 px.

**C16. Layouts** (E-real bootstrap). A layout is the scene id without its arm prefix (`real_`, `staffed_`, `plain_`). A draw resamples whole layouts, with all their arms and people.

**C17. `eval_people.json`.** For each evaluation scene, the frozen people come in descending score order: x, y, yaw (chosen variant; None allowed), score, box, torso pixel, and the projection of (x, y, `person_mark_height_m` = 1.1) as `mark_pixel`, which is None when behind the camera or outside the image. `eval_people.json` is keyed by scene folder and also lists the scene's listing keys.
