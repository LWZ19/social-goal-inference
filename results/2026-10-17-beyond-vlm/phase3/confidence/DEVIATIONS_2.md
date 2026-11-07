# Phase 3 confidence: deviation 2, R8 tolerance compared exactly (compute agent, 18 Oct 2026)

The compute agent wrote this on 18 Oct 2026, after pass 2's first attempt (17:43:14 +0800) stopped at R8 and before scoring resumed. Its sha256 is in `DEVIATIONS_2.sha256`, and pass 2 verifies it, with `DEVIATIONS.md`, on start. `DEVIATIONS.md` (sha256 afe30938…01cc2c) is unchanged.

## What happened

- R8 failed on one value of 26: S5b (1 − c_V1) for watch Qwen. The code gave 0.3125, and the note's expected value is 0.312.
- No other R8 value failed: the per-cell values, the 549-record pooled values and the five s23 values all pass.
- The pass stopped as section 6's stop rule says. The only numbers it printed were R8's baseline signals (S5b and the three single flags). No P1, P1b, S2, S3, S4, S6 or S6z number was computed or printed. The attempt is in `pass2.log`.

## Why this is the comparison, not the AUROC code

- **Phase 1d's own value is the same.** The raw value in `results/still-scenes/learned-models/soft-density/summary.json` (`auroc_all/watch_qwen/V1`) is 0.3125, identical to this code's. Watch Qwen has 4 failures and 32 successes, and 0.3125 = 40/128.
- **The published 0.312 is that value printed.** Phase 1d printed 0.3125 to 3 decimals, and Python's round-half-even formatting turns 0.3125 into "0.312".
- **The exact gap is inside the tolerance.** |0.3125 − 0.312| = 1/2000 = 0.0005 exactly, which is within the note's inclusive ±0.0005. In binary floating point, 0.3125 − 0.312 evaluates to 5.000000000000004e-4, because 0.312 has no exact binary form. That is why the float comparison `abs(got - expected) <= 0.0005` failed.

## The fix

- **One change in `code/tables.py`.** R8 now compares each value exactly: |Fraction(got) − Fraction(str(expected))| ≤ 5/10000, where `got` is the computed float taken exactly and `expected` is the published 3-decimal value as a decimal. This is the note's inclusive ±0.0005 in exact arithmetic.
- **Nothing else changes:** not the AUROC function, not the expected values, and not the 26 checks.
- **Hash verification.** `verify_frozen()` in `tables.py` now verifies every `DEVIATIONS*.sha256` present (both `DEVIATIONS.md` and this file).
