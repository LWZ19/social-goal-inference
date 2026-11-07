# Round 7, part B: band-edge sensitivity (written before running b_band.py; description only, zero calls)

- CT4OT, with the true roster and the true target, scored 0.29.
- CT4PJT, with the model's roster and the true target, scored 0.62.
- The difference was the solver's 1.0 m standoff sitting against the band's 1.2 m top, not information.
- SUMMARY §1 already found one earlier result (CT2 over CT) to be a 1.2 m edge effect.

## The one sensitivity point, and why it is not chosen by looking

- **Frozen:** the in-line band is 0.5-1.2 m from the person ahead (SCORING.md, from the sim benchmark's RULES.md).
- **Sensitivity:** the band's top is 1.5 m. That is the sim's own threshold for counting the robot as joined behind
  the last person (`JOIN_GAP_M = (0.3, 1.5)`, `experiment.py`). It is the sim's definition of standing in the line
  behind someone, set before any of these runs.
- **Nothing else changes:** the bottom (0.5), side (0.6), heading, clearance and the front test.
- **Recomputed from the logged 1 Hz samples.** At 1.2 m the script must reproduce every episode's frozen strict
  share (it asserts this).
- **The frozen 1.2 m stays primary.** Nothing frozen is rescored. The confirmatory result stands as written.
- **Process note.**
  - The first version (note 9fa6bbdc, script b55a2961) stopped on its own reproduction check before printing any
    number.
  - The cause is that `gap_m` is logged to 1 mm, so a logged 1.2 can be a failed 1.2004 (1 sample on the pilot, a
    few on the confirm logs).
  - The script now uses the sim's logged flag at 1.2. At 1.5, a sample also passes when every other condition holds
    and 1.2 <= `gap_m` <= 1.5.
  - Nothing else changed.

## Questions

- **B1 (confirm logs, both models; spent, description only).** For the pairs CT4PJ − CT4P (H1's), CT4PJ − CT3F and
  CT4 − CT3F, at 1.2 and at 1.5:
  - wins, losses and ties (|d| < 0.01);
  - the exact sign test;
  - mean d;
  - the one-sided 95% paired-t lower bound.
- **B2.** Each arm's mean strict share at both tops, and the share of "too far" seconds at 1.2 that pass at 1.5.
- **B3.** The same on the pilot logs (design data).

## Readings, fixed now

- **E-none.** No B1 comparison changes its sign-test outcome at p 0.05, and no mean d changes sign. Then the band
  edge moves levels but does not drive the method comparisons.
- **E-flip.** Some B1 comparison changes its sign-test outcome or its mean d's sign.
  - The frozen result is not changed.
- **For ROUND7's decisions (T, WA, TG, PF):**
  - They are read at 1.2 m as fixed.
