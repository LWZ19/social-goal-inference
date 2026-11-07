# CT2-none and CT2-rat on all cells (post hoc, labelled; frozen 2026-11-02 ~05:45, before any episode of either)

**Post hoc.** CT2-nora and CT2-norv have been judged: removing readmit, or readmit with the vacated place, leaves CT2's share gain over CT intact (`norv_score_out.txt`).

## Arms
Both run on the same cached cells, so they make no model calls: Sonnet 10–19 (49 cells with an answer) and DeepSeek 10–14 (25).
- **CT2-none.** The CT2 code path with all five parts off:
  - readmit off, line span 1.0 m (CT's), no vacated place, no path memory;
  - no forward ratchet and no stood-on replace.
  - It is a sanity check that the CT2 code path with nothing switched on reproduces CT. **Run it first.**
- **CT2-rat.** CT2 without the forward ratchet (and its stood-on replace), as already defined. Only the scope widens, from 15 cells to all cells.

## Predictions (inference, written before any episode)
- **N1.** CT2-none against CT, paired share strict: mean within ±0.04 on each model (the timing floor: CT repeat +0.005, CT2 repeat +0.040).
- **R1.** CT2-rat's replace-rule samples (`analysis/failures.py`, all cells) come back to at least half of CT's: Sonnet ≥ 179 (CT 358), DeepSeek ≥ 60 (CT 119).
- **R2.** CT2-rat keeps at least half of CT2's gain over CT on each model: CT2-rat − CT ≥ +0.041 for Sonnet and ≥ +0.039 for DeepSeek. That is, the ratchet alone does not carry the gain.
  - This prediction is informed by the 15-cell CT2-rat run (+0.018 below CT2, with classes substituting), so it is not independent of that run.

## Reading rule, fixed now
- **N1 holds, R2 holds:** CT2's gain is redundant across its parts. No single part, nor readmit and the vacated place together, is necessary. Report that, with no more ablations before the report.
- **N1 holds, R2 fails:** the ratchet carries most of the gain.
- **N1 fails:** part of the gain comes from a difference between the CT and CT2 code paths beyond the five named parts. Report it and inspect the code before any claim about the parts.
