# Round 6 results: why CT4PJ trails CT3F (exploratory, zero calls; scored confirm runs, both models)

Inputs:
- questions and readings fixed beforehand in `ROUND6.md` and `ROUND6_B.md` (`ROUND6.sha256`).

Nothing here is a claim. Any design built from it must be confirmed on a fresh candidate range.

## Part A: goal jumps. Reading: supports (a), goal flicker costs the place, on both models

Pooled jumps per minute after the first service, by the sim's joined state:

| | DeepSeek CT3F | DeepSeek CT4PJ | Sonnet CT3F | Sonnet CT4PJ |
|---|---|---|---|---|
| switch of person, joined | 2.63 | 6.72 | 3.02 | 7.46 |
| same person, joined | 2.55 | 2.36 | 2.59 | 2.32 |
| switch of person, not joined | 5.34 | 10.85 | 4.54 | 10.46 |
| same person, not joined | 7.45 | 6.55 | 5.60 | 6.61 |

- **CT4PJ's excess is all switches of person**, both while joined (+4.1 and +4.4 a minute) and while not.
- **Jumps that follow the same person are level or lower.**
- **Switches before CT4PJ's leaves:** 0.76 in the 5 s before a leave, against 0.33 and 0.35 of joined seconds. That is a lift of 2.3 and 2.2, which meets the bar of 2.
- **Caveat: every arm's lift is high** (CT3F 4.5 and 3.1), because leaving needs a goal move. The evidence is in the rate. Per joined switch, a leave follows about as often for CT4PJ as for CT3F, roughly 1 in 8 to 1 in 10. CT4PJ leaves more because it switches more.
- **Per layout the link is weak.** CT4PJ jumps more than CT3F on 48 of 66 and 59 of 71 of all layouts, about the same share as on the layouts where it trails (28/34, 37/44). The rank correlations are −0.25 and −0.29.
- **Where the gap sits.** 34% (DeepSeek) and 42% (Sonnet) of the CT4PJ − CT3F gap is in layouts where CT4PJ's join test never fired. There CT4PJ behaves as CT4P, and flicker before joining (+5.5 and +5.9 switches a minute while not joined) is the likely part.
- **Under lag** (members wait 0-4 s before stepping up), the commitment rule's gain over CT4P is about 0 on both models: −0.010 and −0.021; +0.035 and −0.013. Without lag it is +0.08 to +0.17.

## Part B: frozen predecessor. Reading: "the roster at commitment is the problem"; the direction does not stand

Over re-solves where the arm's own join test is on:

| | not in line by the sim | right last, sim agrees | switches per minute joined |
|---|---|---|---|
| DeepSeek CT3F | 0.320 | 0.482 | 4.08 |
| DeepSeek CT4PJ | 0.394 | 0.493 | 8.08 |
| DeepSeek frozen (counterfactual on CT4PJ's logs) | 0.394 | 0.371 | 0.75 |
| Sonnet CT3F | 0.297 | 0.550 | 4.01 |
| Sonnet CT4PJ | 0.423 | 0.497 | 8.77 |
| Sonnet frozen | 0.423 | 0.320 | 0.79 |

- **Freezing the person ahead at commitment** takes switches below CT3F's but loses 0.12 and 0.18 of right last. The person CT4PJ follows when it commits is often wrong, and freezing keeps it. The wrong-place share is not over twice CT3F's (1.2 and 1.4 times).
- **Both arms often believe they have joined while the sim has them out of the line**: 30-42% of their joined re-solves.
- **While truly in line, both pick the right person about equally often**, about half the time (0.48 against 0.49; 0.55 against 0.50). CT3F switches half as often.

## What this adds up to

1. **CT3F's lead over the role belief is not better knowledge of whom it stands behind.** At equal accuracy it changes its mind half as often. Standing still near the end of the line keeps the robot in line even when the person it follows is wrong, so stability itself earns time in line.
2. **Maximal stability is worse.** The choice at commitment is right only about a third of the time, so locking it in costs more than flicker does.
3. **So the open design problem is the trade-off between stability and accuracy in the decision**, not the belief. CT3F sits at about 4 switches a minute and 0.48-0.55 right; CT4PJ at 8 and 0.49-0.50; frozen at under 1 and 0.32-0.37.

   A principled decision would weigh the expected gain of switching against the cost of moving, which includes giving up the current place. It would have to reach CT3F's corner or beyond. Whether any such rule can do that on these logs is the next offline question (part C). If none can, the gap is information that CT3F's anchored roster carries, not decision, and the role-belief line stops there.
