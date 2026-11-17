# I03 Stage A: deviations and clarifications

The Stage A compute agent wrote this on 17 Oct 2026, after the fresh seeds were generated and the scorer self-check had run on fit data. It was written before the fit grid was scored and before any evaluation scene was loaded. FROZEN.md (sha256 8106f74a…b979a, checked before work started) is not edited.

## Deviations

**D1. F-real holds 39 visible true people, not 42.** FROZEN.md section 1 says the 21 realistic worked examples "hold 42 visible true people". The scorer counts 39 under `sgi.gt.derive.is_visible` with `configs/default.yaml`: 33 non-staff and 6 staff, out of 55 people in `truth.people`. Of those 55, 41 project into the image at some height in 0.2, 1.0 or 1.7 m. The other 14 are staff who stand outside the image, and two staff in the image (staffed and real `examples_03`, s0) have 0 of 9 axis samples seen.
- The figure is a description in the note, not a procedure, so no step changes.
- The verdict and the selection use the scorer's count. Every table prints n.

**D2. The matching check runs after the fit freeze.** The task's step list puts the "matching reproduces `_matched_people` on at least 20 recorded records" check in step 2, before the fit (step 3). But recorded records exist only on evaluation scenes, and FROZEN.md section 3 says no Stage A code loads an evaluation scene before `fit_selection.json` and `FIT.sha256` are written. So the check (`stageA/check_matching.py`) runs right after the freeze and before the blob detector is run on any evaluation scene. The note's own step 2 self-check (true people as detections give recall 1 and 0 false) runs on fit data before the fit.

**D3. File names.** `fit_selection.json` and `FIT.sha256` are written in `i03/`, as the note names them. `stageA/fit_grid.txt` and `stageA/chosen.json` are also written, as the task names them. `chosen.json` carries the sha256 of `fit_selection.json`, and `stageA/evaluate.py` refuses to run unless the two agree.

## Clarifications (points the note leaves open, fixed here before the fit)

1. **"Within 0.5 m"** means a distance ≤ 0.5 m everywhere: exemption, near-hidden, merge and merged pair. This follows the package's matching, which keeps a pair when its gap is ≤ 2 × `person_radius_m`. A neighbour pair is < 0.8 m, as the note says.
2. **Body filter bounds** are inclusive at both ends, with a 1e-9 m float tolerance on top_hi = z_hi − 0.05 and on both footprint bounds. Footprints are sums of multiples of 0.05 and can land a float hair outside an exact bound.
3. **Split window.** The window's half-width in cells is round(split_r / 0.05), which gives 3, 5 or 7 cells. The window maximum is taken over the component's own cells only (others count as −inf), as in the pilot. A cell equidistant from two seed cells joins the first seed cell in row-major order. A one-cell blob has a footprint of 0.05 m.
4. **Facing axis.**
   - The band variant is unknown (None) for a blob with fewer than 3 kept points at 1.0–1.6 m.
   - For choosing between the two variants on fit data, accuracy is the share of all matched pairs whose axis lies within 45° of the true yaw modulo 180°. An unknown axis counts as a miss.
   - The higher minimum over the two fit domains wins, and a tie goes to "all". The choice is made at the chosen threshold setting only.
5. **Role classes** (tabulation only), as in `results/still-scenes/overview/phase-notes/phase1-files/a0_facing`:
   - staff: the person has a role;
   - line member: the person's group is at a target of a waiting_line episode;
   - group member: the person's group is at a target of a conversation_group, audience or group_photo episode;
   - other: anyone else.
6. **Share of misses that are staff behind counters** = staff-behind-counters misses ÷ all misses in the group.
7. **Where the misses' depth went** is a descriptive breakdown that the note does not ask for (the task asks "where misses are"). Its categories are fixed here.
   - The missed person's blob is the blob (any blob, before the body filter) holding the most height-map cells within 0.25 m of the person.
   - The categories are checked in this order: no height-map cells within 0.25 m; kept blob matched to another person (a merge); kept blob unmatched, with its centre beyond 0.5 m; rejected, top above the clip (merged with tall structure); rejected, footprint above foot_hi; rejected, top below top_lo; rejected, footprint below foot_lo.
8. **VLM answers** are formed as in `results/still-scenes/overview/phase-notes/phase1-files/a0_facing`. One answer is the union of the people lists of a key's records. Identical entries are kept once, judged by xy to 1e-9 m and pixel to 1e-6 px, and the union is numbered in sorted order of that key.
   - "Blobs weighted to a model's answer keys" means each answer key that has a list adds its scene's blob counts once.
9. **Wilson intervals** use z = 1.96.
10. **The verdict** is computed on the two evaluation domains only, as the note says. The pooled-68 and per-set figures are reported but do not decide it.
