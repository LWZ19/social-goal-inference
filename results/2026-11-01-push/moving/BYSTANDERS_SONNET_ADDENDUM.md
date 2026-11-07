# Bystanders, Sonnet addendum (frozen 2026-11-02, before any Sonnet cafe or sweet_spot bystander cell)

## What it adds

- BYSTANDERS_FROZEN.md's optional arm (Sonnet, room, seeds 10-19, CT2/CT3/Gl) is already running, with its reading
  unchanged: C2's capture bar, and share descriptive.
- This addendum adds a second model on the other two scenes, so that the pilot's capture and roles findings are not
  DeepSeek-only.

## Cells and arms

- **Sonnet 5.5 (low)** on `cafe_line2_bystanders` and `sweet_spot_line2_bystanders`, Habitat seeds 10-19: **20 cells**.
- **Arms:** CT2 (makes the answer), CT3, Gl, and R1.
  - R1 is reported only, as baseline context for the report.
- Arm definitions are as in the pilot (DEVIATIONS 12). CT3 has `co_motion_s = 5.0`, and CT3b is not run here.

## Bars (judged once, after all 30 Sonnet cells; `bystanders_sonnet_score.py`)

- **C2 (capture), cafe + sweet_spot:** CT3's pooled capture is at most a third of CT2's, and CT3 is captured (≥ 1 s) in
  at most 20% of cells. Room's 10 cells are judged with the same bar, separately, as the optional arm.
- **S2 (roster), all 30 cells:** Sonnet names a bystander as waiting in at most 6 answers. A standard-goal answer
  names nobody.
- **SR (roles), all 30 cells with a constraint answer:**
  - Gl's capture exceeds CT3's in at least 80% of the cells where either is captured;
  - CT3 against Gl strict share has wins > losses and sign p ≤ 0.05.
- **Reported only:** CT3 against CT2 share, and anything involving R1. A sign p ≤ 0.05 is stated as such, but it is
  not the confirmation's bar, which stays DeepSeek's C1.

## Predictions (inference)

- Sonnet gives constraint answers in at least 26 of 30 cells (10 of 10 on the original room).
- CT2 is captured in at least a third of cafe + sweet_spot cells.
- C2, S2 and SR pass.
- Sonnet's CT3 − CT2 share gain is larger than DeepSeek's (+0.023), because its answers arrive sooner, while the
  lines still have the states where capture happens (F2).

- After writing this note and its scorer, I ran the scorer as a format check with its model filter relaxed. It read
  DeepSeek cells for seeds 10-19, which included the first confirmation seeds (15-19) already on disk.
- So I saw an interim mix of pilot and partial-confirmation numbers. BYSTANDERS_CONFIRM_FROZEN.md and its scorer were
  locked before that and are unchanged.
- This note's text and bars were written before the format check and are not edited after it, apart from this
  section.
