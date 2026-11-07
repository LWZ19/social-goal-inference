# E2 result: detector people against C's people call on fresh static scenes (read by E2_FROZEN.md)

## Inputs

- **Design.** `E2_FROZEN.md` 38b2ef01 and its code (`E2.sha256`), frozen before any scene existed, plus
  `E2_ADDENDUM_1.md` 5f887334, written before any score was read.
- **Scenes and code.** Seeds 500-504, 8 people rows, 40 layouts. `manifest.csv` 6052e279, `e2_people.json` 7d3693c5.
  HEAD d50a7b3 with an empty diff throughout.
- **Runs.** `runs/e2_soclaas_{qwen38_low,dsv41,glm53}`, 160 units each.
  - GLM was resumed 13 times under item 19's stopping rule. 11 units were left out of every arm, which is within the
    16-unit limit, so GLM is not "partial".
- **Outputs.**
  - Frozen and fixed in advance: `e2_score_all_out.txt`, `e2_sens_all_out.txt`, and `e2_excluded_out.txt` (its
    script `e2_excluded.py` was hashed 85796fc2 before any GLM score was read).
  - Post hoc: `e2_pooled_worst.py` and `e2_pooled_worst_out.txt`.

## 1. The claim holds

"On fresh static scenes D beats C, faster at no loss of accuracy": A, G, S1 and S2 hold for each of Qwen, GLM and
DeepSeek. Success counts are over people-matter units.

| | A0 (current C) | A0b (C redrawn) | D (detector people) | gain / harm | floor rw | goal-step median, A0 → D | whole unit, A0 → D | D faster / slower | placebo p |
|---|---|---|---|---|---|---|---|---|---|
| Qwen 3.8 | 144/159 | 146 | **154** | 11 / 1 | 6 | 70.0 → 45.0 s | 107 → 83 s | 133 / 26 | 0.87 |
| DeepSeek V4.1 | 126/157 | 131 | **132** | 16 / 10 | 9 | 34.5 → 18.8 s | 66 → 50 s | 148 / 9 | 0.75 |
| GLM-5.3 | 100/143 | 93 | **125** | 26 / 1 | 23 | 122 → 89 s | 142 → 109 s | 107 / 36 | 1.00 |

- **Calls:** D makes one call fewer on every people-matter unit (4 → 3 with plan and context), and never calls for
  people.
- **The fresh-unit sensitivity** (addendum 1: no cached goal-step call) agrees on every model: D faster 42/11, 71/4
  and 35/16, with null placebos.

## 2. H2, more accurate: holds, pooled and for two of three models

- **Pooled over the three free models:** net +41, layout sign-flip p 0.0001.
- **Per model:**
  - GLM: net +25, p 0.0010 over 14 layouts; the row test gives p 0.0625 over 7 rows.
  - Qwen: net +10, p 0.031 over 6 layouts.
  - DeepSeek: net +6, p 0.41, not shown.
- **So "faster and more accurate" is stated pooled.** Per model it is stated only for GLM and Qwen.

## 3. GLM's 11 left-out units, and whether they could bias the result

- **Where they are.** All 11 are people-matter. 8 are conversation groups (cg3 5, cg4 3) and 3 are waiting lines.
  They sit in a few layouts: s501 cg3 4, s502 cg4 3, s500 cg3 2, s501 wl4 2.
- **What failed.**
  - Every failure is a constraints call that gave up after about 550 s on the same units every pass. So it is
    unit-specific, not an outage.
  - The failed arms are mostly C's: A0b alone in 6, A0 alone in 3, A0 + D in 1, all three in 1.
  - Units are left out whole, so no arm keeps a unit another arm lost.
- **How it could bias.**
  - If these hard conversation-group units favour C, leaving them out flatters D.
  - Worst case for D: every left-out unit counted as A0 right, D wrong, and D slower. Under it A, G and S2 still hold
    for GLM, and the pooled H2 still holds (net +30, p 0.016, post hoc).
  - GLM's own H2 does not survive the worst case (net +14, p 0.19).
- **So:** the claim and the pooled H2 cannot have been produced by the exclusions. GLM's per-model "more accurate"
  can be read only as the units scored.

## 4. Predictions (`E2_FROZEN.md` §7)

- **A and G held,** as predicted.
- **Missed:** "H2 is not shown at this size". It is shown pooled, and for GLM and Qwen.
- **Speed:**
  - Qwen saved 25 s, against the 20 s predicted.
  - DeepSeek saved 15.7 s, against 3-6 s predicted; the saving is larger than expected.
  - GLM's placebo did not fail, against the prediction that its gateway might be unreadable.
- **Detector:**
  - Missed: false people were 0.475 per image, against the 0.2-0.3 predicted (A1's unseen-layout rate was 0.242).
  - Recall of non-staff people was 159/160 and of staff 11/13, as predicted.
  - D was not harmed by the false people (harm 1, 10, 1).

## 5. Limits (`E2_FROZEN.md` §8)

- Layouts within a row share structure. The row-level tests are weaker: GLM p 0.0625, Qwen 0.5, DeepSeek 0.41.
- The test covers development situation types, the real variant, static scenes, and the free models.
- These scenes are now spent for any choice.
