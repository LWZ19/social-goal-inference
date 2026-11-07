# I09 A1: deviations and clarifications (part 2)

Written by the I09 A1 compute agent on 17 Oct 2026, about 21:50 +08:00. This is after `a1/FIT.sha256`, `a1/ADDENDUM_FIX.md` and `a1/FIT_FIX.sha256`, and before `a1/manifest_eval.json` exists or any evaluation image is processed. Its sha256 is in `DEVIATIONS_2.sha256`. This file is never edited.

## D3. One general fix applied (see `a1/ADDENDUM_FIX.md`, hashed 21:42:43)
- **What.** The facing variant F5 is F4 without the one-sided face rule. The union rule over F1–F5 picked it on fit data: F-real 36/36, F-ph 505/513, within 45 with unknown counted as wrong.
- **Final frozen choice.** RT-DETRv2-r50vd, τ 0.25, band [1.00, 2.30] m, flat test off, d_dup 0.50 m, d_push 0.20 m, facing F5.
- **How chosen.** Through fallback 4b, since none of the 192 settings was feasible.
- **Time.** The addendum's header says "about 21:50". The hash was actually taken at 21:42:43.

## D4. Extra hashed fit file
`a1/fit_scenes_chosen.jsonl` holds the per-scene rows of the chosen setting. It is included in `a1/FIT.sha256`, in addition to the files the note names.

## Clarifications (reporting only; no method change)
**C18.** p90 is the nearest-rank 90th percentile: sorted[ceil(0.9 n) − 1].

**C19.** In the like-for-like table, the detector's full-yaw within 45 deg is given two ways: over its matched people with unknown counted as wrong, and among those with a known yaw. The VLMs' figure is the I03 table's: within 45 of known, with the number unknown.

**C20. A0-matched facing.**
- The distinct-instance figure is computed for the detector, where each instance has one yaw.
- The model's own value on the same subset is pair-weighted, as in A0.
- The layout bootstrap resamples A0's `layout` field.

**C21. Runtime check.** Feeding the timing pass's boxes and keypoints through the project side must give the same people as the evaluation path. A person counts as the same when xy agrees within 1e-3 m and yaw within 0.01 rad. The result is reported.

**C22. Post hoc lines.**
- The pre-fix choice F4 is also reported on E-real, both on matched people and A0-matched. These lines are labelled POST HOC and change no verdict.
- `people_rows.jsonl` and `eval_people.json` carry all five variants' yaws for transparency. Only F5 is scored.

**C23. Module names.** `results/still-scenes/perception/depth-blob-detector/stageA` contains `fit_grid.py` and `evaluate.py`. A1 code therefore loads its own `fit_grid.py` by path.
