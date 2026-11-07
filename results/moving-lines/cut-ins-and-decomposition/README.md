# Audit items 3 and 6: zero-call re-scoring of logged moving-people episodes (6 Nov)

Read-only. No model call and no simulator run. Inputs are `results/data/moving/runs/<model>/<scenario>/seed<N>/<arm>/episode.json`
(91 samples at 1 Hz each). Scoring uses `results/moving-lines/design-and-scoring/summarise_fresh.py`'s `measures` and `sign_p`, imported unchanged.

| File | What |
|---|---|
| `item3_ct0_cto_ctt.py` → `item3_out.txt` | Item 3: CT0, CTo and CTt on seeds 0-2 |
| `item6_cutin_exposure.py` → `item6_out.txt` | Item 6: cut-ins normalised by exposure |
| `item6_cutin_exposure.py --cells` → `item6_cells_out.txt` | The same, plus every cell's counts |

Run with `.venv/bin/python <script> > <out>`.

## Definitions

- **Share (frozen, primary):** the share of samples with the logged `in_line` flag (SCORING.md).
- **Share (strict):** SCORING_STRICT.md, as in `summarise_fresh.in_line(x, "strict")`:
  - with someone ahead: frozen in line, and `side` ≤ 0.6 m;
  - with nobody ahead: frozen in line, and `to_slot0_m` ≤ 0.6 m.
- **Paired comparison (item 3):** A against B on the same (scenario, seed) cell.
  - A tie is |d| < 0.01.
  - Shares move in steps of 1/91 (0.011), so this rule equals exact ties. The script asserts this.
  - Sign p is `summarise_fresh.sign_p`: exact, two-sided, over untied pairs. Mean d is A − B.
- **Exposure (item 6):** the number of samples with `side` ≤ 1.0 m.
  - SCORING.md's cut-in needs `side` ≤ 1.0 m. The script asserts that no cut-in sample lies outside the exposure.
- **Cut-ins:** the number of samples where `cut_in` is true. This equals `measures(e)["cut"]`, which is asserted.
- **Rate:** cut-ins divided by exposure. It is undefined when exposure is 0.
- **Pooled rate:** the arm's summed cut-ins divided by its summed exposure.
- **Per-cell tests (item 6):**
  - Raw counts: more, fewer or equal, as integers. "More" means the first arm cut in more, which is worse.
  - Rate: only over cells where both arms have exposure > 0. Ties are |d| < 0.01, and an exact-tie version is given
    beside it.
  - Cells left out for zero exposure are counted by which arm had none.

## Arms (item 3; `sim/sgi_habitat/experiment_oracle.py` docstring)

All arms share C1's first answer; only the re-solves differ.

- **CT0:** one re-solve on the tracked people, then the goal is frozen.
  - CT0 against C1 measures the better people positions that tracking gives.
  - CT against CT0 measures re-solving as people move.
- **CTo (oracle ceiling, not a method):** CT, with the answer's people and constraints replaced from the first
  re-solve on. The replacement is the queue's true members when the answer was asked (one `come_after`). The model's
  target point is kept.
- **CTt (oracle ceiling, not a method):** CT, with the target point moved to the queue's true counter point (slot 0's
  face point). The model's people and constraints are kept.

## Cells

- **Item 3:**
  - Scenarios: `cafe`, `room_coffee_cart`, `cafe_line2_moving`, `cafe_line6_moving` and `sweet_spot_line2_moving`;
    seeds 0-2, so 15 cells per arm.
  - DeepSeek (`dsv41`) and x-test-3 (`x3`) have all five arms complete.
  - Sonnet (`sonnet55_low`) has CT0, C1, CT and R1. CTo and CTt were never run for it.
  - Seeds 0-2 are development seeds: CT2 was designed from their failure analysis. Read these results as descriptive.
- **Item 6(a):**
  - Scenarios: the five headline scenes (`cafe_slots`, `room_coffee_cart_slots`, `cafe_line2_moving`,
    `cafe_line6_moving`, `sweet_spot_line2_moving`).
  - Seeds: Sonnet 10-19; DeepSeek, `claude_haiku55` and `gemini_fast` 10-14; x3 10-12.
  - x3 has no fresh CT arm.
- **Item 6(b):**
  - The DeepSeek confirmation: 60 cells (cafe and sweet_spot bystander scenes, seeds 15-44).
  - The Sonnet addendum: 20 cells (the same two scenes, seeds 10-19).
  - The cell sets that SUMMARY §7 used are also scored: DeepSeek's pilot plus confirmation, 75 cells, and Sonnet's
    addendum plus room, 30 cells.
