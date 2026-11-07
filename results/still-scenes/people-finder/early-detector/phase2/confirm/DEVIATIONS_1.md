# I09 unseen-layout confirmation: deviations and clarifications (part 1)

Written by the I09 confirmation compute agent on 18 Oct 2026, at about 04:40 +08:00. At that time:
- `FROZEN.md` verified against `FROZEN.sha256` (f55341c0…e751), and every code and data hash of FROZEN section 3 and section 7 verified (04:35:27);
- `manifest.json` did not exist, the compute agent had loaded no test scene, and no detector or pose model had processed any test image.

Its sha256 is in `DEVIATIONS_1.sha256`. This file is never edited. None of the items below changes the method, the scorer, the thresholds or the criteria.
## Part A. What the render step reported doing differently from FROZEN's text (disclosed; facts, not rulings)

## Part B. Clarifications by the compute agent (implementation details the note leaves open)

**C1. Manifest format for the sidecar.** `manifest.json` holds section 2's entries under `entries` and, under `images`, the five fields that `infer.run_pass` and `infer.load_images` read (`scene_id`, `domain` = `"test"`, `set` = `"real"`, `rgb`, `sha256` of `rgb.png`), so both run unchanged. Scene ids repeat across seeds, so raw lines are joined to manifest entries by their `rgb` path, which is unique, and the line's sha256 is asserted equal to the manifest's.

**C2. Sidecar driver.** `sidecar/infer_confirm.py` imports `infer.py` by path with bytecode writing switched off (nothing is written into `phase1c/`), replaces `infer.guard` and `infer.wait_for_gpu` only, and calls `infer.run_pass("eval", <manifest.json>, <raw/>, ("rtdetr_v2_r50vd",))` unchanged. In `eval` mode `run_pass` makes no warm-up, as A1's evaluation pass did.
- The replacement guard applies section 3's rules to every rgb path, as written and resolved. It also re-checks each path against `manifest.json` and requires `MANIFEST.sha256` to verify before any image is read.
- The replacement GPU wait implements section 10: start only when `nvidia-smi` shows at least 3,000 MiB free and `free -m` shows at least 3,000 MiB available; otherwise wait 300 s and poll again, with each wait logged. It returns `memory.used` before the start, so `meta.json`'s `gpu_used_before_MiB` keeps A1's meaning.

**C3. `pip freeze`.** It is run as `<sidecar python> -I -m pip freeze --disable-pip-version-check` immediately before and after the pass, saved to `pip_freeze_before.txt` and `pip_freeze_after.txt`, and asserted identical.

**C4. Loading test scenes.** Every test scene folder is passed through `H.guard_scene` and the section 3 path guard before `scene.json` is read; then `PS.FreshScene(load_scene(folder))`. The sha256 of `rgb.png`, `depth.npy` and `scene.json` is checked against `manifest.json` on every load (C3 and scoring).

**C5. Development comparators.** Section 7's numbers are computed from A1's recorded `people_rows.jsonl` and `rows.jsonl` (hash-checked) and printed beside the test values. C2 separately asserts that the confirmation's own code reproduces them.

**C6. Bootstraps.** Layout clusters are keyed `seed<k>/<scene id>` (182 single-image clusters for 13 seeds). Seed clusters are keyed by seed (13 clusters). Both use `C.boot_ci` with 2000 draws and seed 0, so `random.Random(0)`. The statistic is recomputed on each draw as on the full set: recall on scene rows; line recall, median position error and the facing statistics on per-person rows. As in A1, a draw where the statistic is undefined (for example, no matched profile person) is skipped by `C.boot_ci`.

**C7. Best raw box of a miss.** Among all boxes the sidecar wrote for the image (score ≥ 0.05, unclipped xyxy), it is the highest-scoring one whose closed rectangle contains the projection of (x, y, 1.1 m) of the missed true person. It is None when the projection is behind the camera, outside the image, or in no box.

**C8. False people.** Each is reported with its detector score, its distance to the nearest visible true person, and `near_hidden` (within 0.5 m of a true person who fails `is_visible`, as `PS.score_people` counts it). Exempt detections are listed apart.

**C9. Runtime (not judged).** Sidecar: `t_det_s + t_pose_s` from each raw line, with median and nearest-rank p90 (`C.p90`). Project side: the wall time of `P.run_image` per image in `score.py`, with 2 BLAS threads and depth, camera and raw output already in memory; the first image is not dropped.

**C11. Seed batches.** The batch 200–209 has 10 seeds and 140 layouts. The batch 210–219 has seeds 210–212 only, 42 layouts.

**C12. Order log.** `ORDER.txt` records each step's timestamps as it happens.
