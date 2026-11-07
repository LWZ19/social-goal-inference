
Written by the I09 A1 compute agent on 17 Oct 2026, about 21:50 +08:00. This is after `a1/FIT.sha256` (21:39:53) and before any evaluation manifest exists or any evaluation image is processed. Its sha256 is in `a1/ADDENDUM_FIX.sha256`. This file is never edited.

## Trigger
The fit choice misses section 10 success bars that can be measured on fit data. The choice is #5, RT-DETRv2-r50vd, τ 0.25, wide band, flat off, d_dup 0.50 m, d_push 0.20 m. No setting was feasible, so fallback 4b chose it. Its facing variant is F4.

| Fit domain | Missed bar | Value |
|---|---|---|
| F-real | recall ≥ 0.95 | 36/39 |
| F-real | false ≤ 0.10 per image | 3/21 |
| F-real | non-profile median ≤ 25 deg | 27.4 deg |
| F-real | non-profile within 45 ≥ 0.85 | 12/15 |
| F-ph | recall ≥ 0.95 | 513/585 |
| F-ph | non-profile median ≤ 25 deg | 130.8 deg |
| F-ph | non-profile within 45 ≥ 0.85 | 131/263 |

These bars are met in both domains:
- line recall (F-real 15/15, F-ph 208/215);
- median xy (0.060 and 0.055 m);
- profile front/back (21/21 and 246/250);
- false people on F-ph (0/268).

## Fit failures (fit data only: `fit_people_chosen.jsonl` and a diagnostic pass over the fit scenes)
**Facing.**
- F3's one-sided face rule fires on every matched back-view person: F-real 3 of 3, F-ph 140 of 140. ViTPose reports a confident nose and both eyes on people seen from behind.
- Each time, it turns a correct shoulder yaw into the camera half-plane. Within 45 before the flip: F1 3/3 and 137/140; F2 3/3 and 134/140.
- It never fires on a frontal or oblique person. Its other firings, 74 F-ph profile people, are then overridden by F4's profile rule.
- So across 549 matched fit people, the face rule corrected no one and broke 143.
- F4's profile rule is right on 21/21 F-real and 246/250 F-ph profile people.

**Detection.**
- F-ph: 71 of the 72 misses have a box over the person's chest point, but it scores 0.05–0.25, below every τ of the grid. A fix must keep the grid, so no revision can reach these misses.
- F-real: the 3 misses are staff whose lower body is hidden, in 2 layouts (`examples_02` real and staffed, and `staffed_examples_01`). Each is boxed at 0.84–0.86 but placed 0.61–0.63 m from the true position. Two of them also count as F-real's false people. The third false person stands within 0.42 m of a true person who fails `is_visible`.

## The fix: one revision, of facing
**F5 = F4 without the one-sided face rule.** That is, F2 followed by F4's profile rule, unchanged: the edge-on test, u_ref, r̂, the ±90 deg rule and the 1 px dead zone. F1–F4 are not changed.

It names no situation, scene or label, and it can create a yaw only where F4's profile rule already could.

**Why facing rather than detection.** The facing failure has one cause and accounts for 4 missed (bar, domain) cells in both fit domains. A detection revision could address at most F-real's recall and false bars: 2 cells, 2 layouts, 3 people. F-ph's recall misses lie below the grid's τ.

- I had read the section 8 input `results/still-scenes/facing/a0_pairs.jsonl` and counted the view classes of Gemini's evaluation pairs: 27 frontal, 67 oblique, 171 profile and 45 back.

So the choice of a facing revision over a detection revision is labelled **data-informed**. Its content, dropping the face rule, comes from fit failures only.

## Re-fit and adoption (same grid, same rule)
- Facing does not enter the detection metrics, so the 192 rows and the chosen setting stay as they are. The re-fit script asserts this by recomputing them.
- Facing selection runs again over the union F1–F5, with section 4's metric and rule: within 45 deg, unknown counted as wrong, over the chosen setting's matched fit people. The lower domain is maximised, and ties go to the higher pooled value, then the earlier variant.
- F5 is adopted only if this rule picks it.
- The results go to `a1/fit_fix.txt` and `a1/chosen_fix.json`, hashed into `a1/FIT_FIX.sha256` before the evaluation manifest is written.
- The kill bars are judged after the fix. The section 4 unit check is unaffected, because it concerns F1.
