# I03 Stage A addendum: the one general fix (17 Oct 2026)

The Stage A compute agent wrote this after the original fit was frozen (`fit_selection.json`, sha256 197843df…181d9). It was written before the fix was implemented or run on any scene, and before any evaluation scene was loaded. Its sha256 is in `ADDENDUM_FIX.sha256`. FROZEN.md section 3 allows this step, because the chosen setting misses a success bar on fit data.

## Why a fix applies
The original choice is z_lo 0.40, z_hi 1.90, top_lo 1.50, split_r 0.35, foot_lo 0.30, foot_hi 0.70. On fit data it gives:
- F-real recall 33/39 = 0.846, under the 0.95 success bar (and the 0.85 kill line);
- F-ph recall 579/585;
- false bodies 0/21 (F-real) and 4/268 (F-ph);
- merged pairs 0/9 and 0/143.

All 8 feasible settings have F-real recall 33/39. The only settings that reach 39/39 use split_r 0.15, and they give 1.6–2.9 false bodies per scene.

## Fit-data failures (`stageA/fix_analysis.py`, fit scenes only)
- **F-real (6 misses).** All 6 are staff whose lower body is hidden (`lower_hidden`). Each one's height-map cells sit in a blob whose top is at the clip (1.90 m), with footprints of 0.68–1.68 m. The blob centre lies 0.16–0.38 m from the person.
  - Mechanism: the staff member's cells are 8-connected to a structure that reaches the clip. A cell of that structure lies inside the split window, so the head is not a seed. The staff cells join the structure's seed, and the blob is rejected as tall structure.
  - With split_r 0.15 the head becomes a seed, but walls and furniture then break into many body-sized pieces.
- **F-ph (6 misses).** Non-staff people seen narrow, with blob footprints of 0.27–0.30 m against foot_lo 0.30. The grid already holds foot_lo 0.15, so these need no detector change.

## The fix (one revision of the detector)
**Drop clip-reaching columns.** After step 3 (the height map), every cell whose highest kept point is at or above top_hi = z_hi − 0.05 is removed. The steps that follow use the remaining cells only: components, split, blob features, kept points for the facing axis, and the body filter, which is otherwise unchanged.
- **Reason.** The note's own rule is "a blob that reaches the clip is taller than the clip, so it is not a body". This applies the rule per cell instead of per blob. A body that stands against tall structure is then no longer absorbed by it, and tall structure can no longer be split into body-sized pieces.
- **Scope.** The fix names no situation, scene, role or label. It reads only the depth points and the setting's z_hi.
- **Expected side effects** (not tested before writing):
  - A person whose head reaches top_hi loses those cells, which matters at z_hi 1.90.
  - Structure whose top lies outside the image keeps its lower cells and may leave fragments.
  - The re-fit decides between the two detectors.

## Re-fit and adoption
- The re-fit uses the same 216-setting grid, the same fit data, the same metrics and the same rule as FROZEN.md section 3. Its rows go to `fit_selection_fix.json`, with the hash in `FIT_FIX.sha256`.
- **Adoption.** The selection rule runs on the union of the 432 rows, with the 216 original rows listed first. The fix replaces the original only if the rule picks one of the fixed rows.
  - A complete tie keeps the original, because the rule takes the first of equal keys.
  - The facing-axis variant is then re-chosen at the adopted setting with the same rule.
- `stageA/chosen.json` is rewritten with the adopted detector (original or fix), its setting, its facing variant and every hash. The original choice is kept as `stageA/chosen_original.json`.
- The Stage A kill bars are judged after this step.
- No other change to the detector is made, before or after evaluation.
