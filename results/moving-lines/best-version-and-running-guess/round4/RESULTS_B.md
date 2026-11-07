# Round 4 results so far (3 Nov, ~08:10; before the CT4P closed-loop score exists)

## PL, offline on the 24 pilot layouts (`b_pl_pilot_out.txt`): NOT promising, not proposed

ROUND4.md section 4 (cc30d5e0) needed all four of these, PL-path against W0 and CT5:

| Criterion | PL-path | Against | Result |
|---|---|---|---|
| far share | 0.430 | W0 0.340, CT5 0.492 | fails (only below CT5) |
| right share | 0.297 | W0 0.342 | fails |
| proxy goal in line | 0.197 | W0 0.232, CT5 0.172, CT4 0.241 | fails |
| switches per minute | 7.1 | W0 9.0 | holds |

Why the idea was wrong: P(last = j) for the true last is P(it waits) × the product of (1 − P(waits)) over everyone
farther back. Each uncertain person behind the line dilutes it, so a non-waiter at P 0.3-0.5 at the back can be the
mode. "The last" is an extreme of the waiting set. Every rule built on it inherits the same fragility to uncertain
people behind the line: threshold-then-farthest, expected cost, or the most probable last. The fix has to be in the
belief (who is behind the line), not in a rule over its extremes.

## CT4 against CT4P offline (`c_ct4p_offline_out.txt`; written after CT4P.md was frozen, so not a prediction of record)

Proxy goal in line, robot idle:

| Arm | high-error layouts, all ticks | high-error, right-last ticks | low-error, all ticks | low-error, right-last ticks |
|---|---|---|---|---|
| CT4 | 0.112 | 0.277 | 0.369 | 0.621 |
| CT4P | 0.149 | 0.391 | 0.354 | 0.591 |

- Offline, the direction helps where it should (high-error layouts) and not elsewhere. The effect is modest.
- Even with the path tangent, high-error layouts stay below low-error ones (0.39 against 0.59 on right-last ticks).
  The remembered path is itself bent by whom the answer named.
- In the closed loop, CT4P's Q1 and Q2 may well hold. Q3 (≥ 0.55 pooled on right-last, clean-roster ticks) is less
  likely: offline, the pooled right-last figure is about 0.52.
