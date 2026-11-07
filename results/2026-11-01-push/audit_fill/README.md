
All of these were re-scored from existing records. No model call and no simulator run was made. Each item reproduces
the figures it was checked against.

## Item 1: layout-level tests for the still-scene headline

Files: `item1_item2/layout_tests.py`, `layout_tests_out.txt`, `README.md`.

- **Test:** an exact two-sided sign-flip over the 14 development layouts. Records are paired by (variant, scene,
  episode, trial), 18 pairs per layout.
- **New wording (s21, s23):** C−R and C−E are positive for all six models, p 0.0039-0.0156. Each p is at the floor for
  its count of untied layouts, so it mostly reflects how many layouts are not tied (the six no_one layouts nearly
  always tie). No E−R is significant: smallest p 0.125.
- **Old wording:**
  - C−R: p 0.0049-0.0078.
  - C−E: Qwen p 0.031 (6 untied layouts: one more tie loses it), GLM 0.0078, DeepSeek 0.00098.
- **Checks:** every total and all 15 tests of the earlier `rerun/stats/stats_output.txt` reproduce exactly.

## Correction to EVIDENCE_AUDIT.md row A3

"No E−R difference is significant at the layout level" holds for the new wording only.
- Under the old wording, Qwen's E−R is +39, 8/0/6 layouts, p 0.0078.
- Among the older models, Gemini 3 Flash's E−R is +13, p 0.031.

## Item 2: E3's seven older models, layout-level (descriptive)

- C beats both R and E on five models: Gemini 3, 3.1 Lite, 3.5 Lite and 3.6, and Sonnet 5 (C−R p 0.047, C−E 0.031).
- Haiku 4.5: C−R p 0.036, C−E 0.023.
- **Sonnet 4.6 is the exception:** C−R −4 (p 0.875), C−E 0 (p 1).
- Totals reproduce `e3_models/results_all_out.txt`.
- Caveat: the s24 runs post-date the whitelist and were taken by the names in `run_static.sh`. Each has one folder of
  84 records, and its scenes match `scenes_dirs.yaml`.

## Item 3: CT0, CTo and CTt on seeds 0-2

Files: `item3_item6/item3_ct0_cto_ctt.py`, `item3_out.txt`. Development seeds, descriptive.

**Mean share, frozen / strict:**
- DeepSeek: C1 0.03/0.02, CT0 0.14/0.04, CT 0.18/0.12, CTo 0.29/0.21, CTt 0.32/0.27.
- Sonnet: C1 0.04/0.02, CT0 0.06/0.04, CT 0.20/0.16 (CTo and CTt were never run on Sonnet).
- x-test-3: C1 0.04/0.04, CT0 0.14/0.10, CT 0.12/0.09, CTo 0.13/0.10, CTt 0.15/0.13.

**What it shows, under strict scoring:**
- For Sonnet and DeepSeek, CT's gain over C1 is re-solving as people move (CT vs CT0: Sonnet 11/0/4, p 0.001; DeepSeek
  8/1/6, p 0.039), not better positions at answer time (CT0 vs C1 about +0.02).
- x-test-3 is the opposite: CT0 is about equal to CT.
- The true counter point (CTt) lifts CT a lot on DeepSeek (+0.15) and a little on x-test-3. The true roster (CTo) is
  inconsistent.

## Item 6: cut-ins normalised by exposure

Files: `item3_item6/item6_cutin_exposure.py`, `item6_out.txt`, `item6_cells_out.txt`.

- **Exposure** = samples within 1.0 m of the line's axis, the only place a cut-in can happen.
- **All four figures in SUMMARY §7 reproduce.**
- **CT2 vs R1, fresh cells, per model:**
  - Sonnet: the count gap is exposure; pooled rates 0.145 against 0.137 (rate test 9/4/4, p 0.27).
  - DeepSeek: CT2 lower (0.16 against 0.40, n.s.).
  - Haiku: CT2 lower (0.20 against 0.67; 2/12, p 0.013).
  - Gemini: CT2 higher (0.22 against 0.003; 11/1, p 0.006). Gemini's R1 almost never cuts in.
- **CT2 vs CT:** Sonnet's CT2 cuts in more per exposure (19/8, p 0.052); the others don't.
- **CT3 vs CT2, bystanders:**
  - Frozen sets: DeepSeek 655 against 502 (60 cells), Sonnet 137 against 88 (20 cells). Rate tests n.s.
  - SUMMARY §7's 1,036 against 881 and 138 against 89 use wider sets: the DeepSeek pilot plus the confirmation (75
    cells), and Sonnet with room (30 cells). 358 of each arm's DeepSeek cut-ins are 5 room cells where both arms drove
    through the line.
- **Caveat:** the rate test against R1 drops every `cafe_slots` cell, where R1 has no exposure.

## Item 7: marks outputs saved (in `marks/`)

- `m5_check_out.txt`: Gemini 29 of 38 wrong targets name a person (22 a person waiting in line); the other models 2-7.
- `m2e_glm_out.txt`: GLM layout net +22, p 0.0107.
- Neither script, nor anything it imports, makes a call.
