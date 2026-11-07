# Round 6, part C: can a decision on the role belief's evidence reach CT3F's corner? (written before running c_tradeoff.py)

- A threshold must come from a principle stated before looking.
- The confirm runs only describe; they choose nothing.

Part B (`RESULTS_1.md`) gave the corners, over the arm's joined re-solves (switches per minute joined; right last where the sim agrees the robot is in line):

| | CT3F | CT4PJ |
|---|---|---|
| pilot | 4.46, 0.595 | 8.89, 0.716 |
| confirm DeepSeek | 4.08, 0.482 | 8.08, 0.493 |
| confirm Sonnet | 4.01, 0.550 | 8.77, 0.497 |

**Which corners are compared.** Part C's own table scores every choice the same way: a choice of "none" counts as right when the sim's last is none. So the readings below compare with the CT3F row of that table, not with part B's numbers.

## The evidence, per re-solve on CT4PJ's logged joined stretches (as part B)

- **Candidates:** tracked people with a logged marginal P, not in CT4PJ's logged `behind`, and ahead of the robot along the line. "Ahead" uses s(x) = (x − target) · `line_direction`, which grows towards the back. The robot's position is the latest 1 Hz sample at or before the re-solve.
- **Order:** candidates are ordered nearest the robot first, by s descending.
- **q(k) = P(k is directly ahead)** = P_k × Π over candidates nearer the robot than k of (1 − P_j).
- **q(none)** = Π over all candidates of (1 − P_j).

These are the logged marginals treated as independent. CT4PJ does not log its set posterior.

## C1. The whole family of hold-unless-margin rules (description only)

- The choice starts at argmax q, at the stretch's first re-solve.
- It switches to b = argmax q when q(b) − q(current) > m. q(current) is 0 if the current person is no longer a candidate.
- Reported for m in {0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95}.
- **The question is about the whole curve:** does any point reach CT3F's corner, meaning switches at or below CT3F's and right last at or above it?
- No m is chosen from it.

## C2. The principled rule: CT5's decision after commitment, ROUND3 section 2 costs unchanged

The costs were fixed before any decision code ran; nothing is fitted. Unit: one service interval.
- **Cut:** 1 per waiting person the robot would stand in front of.
- **Far:** 0.25 per non-waiter between the true last ahead and the robot's choice.
- **Move:** 0.125 per metre (v 0.5 m/s, mean service 16 s).

The rule:
- **Expected cost of coming after j:** 1 × Σ P_i over candidates nearer the robot than j, plus 0.25 × Σ over candidates i not nearer than j of (1 − P_i) × Π (1 − P_l) over candidates l nearer than i.
- **Coming after none** (going to the front): 1 × Σ P_i.
- **Switching:** as CT5's `decide`, keep the current choice unless its expected cost exceeds the best one's plus 0.125 × |s(best) − s(current)|, with s(none) = 0.
- **Why this is the principle asked for:** it weighs following the wrong person (cut or far) against the cost of acting, with values set on 31 Oct for CT5 and not revisited.
- **What changes from CT5:** only the candidate set (people ahead of the robot once it has joined), and independent marginals in place of the set posterior.
- **Sensitivity, reported only and changing nothing:** ROUND3's own points, cut-to-far ratio 2 and 8, move cost 0 and doubled.

## Logs each step uses

| | pilot logs (`runs_pilot3/dsv41`, design 1000-1199) | confirm logs (`runs_confirm4/dsv41`, `sonnet55_low`; spent) |
|---|---|---|
| C1 curve | described | described |
| C2 point | the test of R2 | the test of R2, description only |
| anything fitted | nothing | nothing |

## Readings, fixed now

- **R1: no rule in the family reaches CT3F.** No m has switches at or below CT3F's and right last at or above CT3F's, on the pilot logs and on both confirm logs. Then the gap is information that CT3F's anchored roster has and this belief lacks, not decision. I recommend closing the role-belief line and writing it up.
- **R3: only a margin reaches.** Some m reaches the corner but C2 does not. That is reported as "no principled rule found", and the recommendation is as in R1.
- **Results that disagree across the three logs** are mixed. No principled rule is demonstrated, so the recommendation is as in R1.
- **Open loop:** the robot's path would differ under each rule, so even R2 is a choice-level check, not a prediction of time in line.
