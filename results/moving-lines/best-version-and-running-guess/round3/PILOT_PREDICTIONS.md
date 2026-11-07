
Offline replay on the 24 pilot layouts: robot-idle dumps, CT2's first answers, `g_pilot_predict_out.txt`. No CT3F,
CT4, W0 or CT5 output existed when this was written. This is a pilot on design data, so nothing here is a claim. It
tests whether offline replay predicts the closed loop now, after it failed to in the 3 Nov test.

| Offline | right | cut | far | empty | last switches / min | roster changes / tick |
|---|---|---|---|---|---|---|
| CT4 | 0.400 | 0.263 | 0.323 | 0.014 | 14.2 | 0.163 |
| W0 | 0.342 | 0.304 | 0.340 | 0.014 | 9.0 | 0.089 |
| CT5 | 0.312 | 0.197 | 0.492 | 0.000 | 8.5 | 0.089 |

Per layout, CT5 switches less than CT4 on 19 of 24.

## Predictions for the closed loop

Each is scored over layouts; "held" means more layouts go the stated way than the other way.
- **P1 (stability).** Goal jumps over 0.3 m per minute, the POSTHOC_1 Q5 measure:
  - CT5 < CT4 and W0 < CT4;
  - CT5 fewer than CT4 on at least 17 of 24 layouts.
- **P2 (stability).** Roster changes per re-solve: W0 and CT5 about half of CT4's, each below 0.6 × CT4's pooled
  rate.
- **P3 (cut-ins).** Cut-in samples: CT5 < CT4. W0 not below CT4, since offline its cut share is higher (0.304
  against 0.263).
- **P4 (the hypothesis that matters).** Strict share in line from the first service after delivery: CT5 > CT4 and
  W0 > CT4. The flicker diagnosis says stability buys time in line.
- **No prediction: CT5 against CT3F.** The offline loop cannot reproduce CT3F's closed loop faithfully
  (DEVIATION_1), so there is nothing to predict from. This comparison decides whether a confirmatory test is
  worth running.

## What each outcome means

- **P1-P3 hold, P4 fails:** stability is not what limits time in line. The diagnosis is incomplete, so look at
  where the robot goes (far choices) before any new test.
- **P4 holds, but CT5 is below CT3F:** CT5 is a better CT4, but not yet better than the best rules. Go back to
  design, and run no confirmatory test.
- **P4 holds and CT5 is at or above CT3F:** draft the confirmatory note: CT5 against CT3F, strict in line from
  the first service, on candidates 2000-2449.
