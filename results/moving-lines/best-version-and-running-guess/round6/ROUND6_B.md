# Round 6, part B: would a frozen predecessor hold the right person? (written before running b_pred.py)

Exploratory, zero calls. Data:
- the scored confirm runs (both models; spent, so now design data);
- the pilot runs (DeepSeek), to test the code.

Part A (`RESULTS_1.md`) found two things:
- While joined by its own test, CT4PJ's goal switches person 2.5 times as often as CT3F's.
- Its leaves follow switches at over twice the base rate.

It also showed a second failure: CT4PJ "joined" behind non-members while the sim had it outside the line.

**The idea under test.** Once the robot has committed to a place, the decision needs one relation: who is directly ahead of me. In a line, that person changes only when they leave; people joining behind don't matter. Round 5 found leaving is detectable: the away bound fires on 81% of ex-member bouts and 1-6% of member bouts. Joining, which round 5 could not separate, no longer matters once the robot is in line.

## P-a. Committing in the wrong place (per arm, CT3F and CT4PJ, from their logs)

Among ok re-solves where the arm's own join test is on, report:
- the share where the sim says the robot is not in the line (`truth.joined` false);
- episodes with any such re-solve;
- the share whose `last` is the sim's right last (`truth.last`):
  - where the sim agrees the robot is in line;
  - over all of the arm's joined re-solves;
- identity switches of `last` per minute joined.

## P-b. Counterfactual frozen predecessor, on CT4PJ's logged joined stretches (open loop)

A stretch is a maximal run of consecutive ok re-solves with CT4PJ's joined flag on.

At the stretch's first re-solve:
- the predecessor p is CT4PJ's own `last`;
- the order ahead is CT4PJ's roster, present in the tracks, by distance to the answer's target.

At each later re-solve, p is replaced by the next person ahead of it in that order (or none) when either holds:
- p is no longer tracked;
- p's walking bout breaks round 5's away bound. That is speed over 0.3 m/s between re-solves, the bout starting from the position 1 s earlier, and the net distance to the target growing by more than 0.5 m.

Nothing else changes p. Reported on the same re-solves as P-a:
- right-last share;
- identity switches per minute joined.

The robot's path would differ under the rule, so this is a first check, not a prediction of time in line.

## How it will be read, fixed now

- **The direction stands, as a candidate for a design pilot:** on both models, the frozen predecessor's switches per minute joined are at or below CT3F's, and its right-last share (sim agrees) is at or above CT4PJ's.
- **The roster at commitment is the problem:** freezing lowers the right-last share, or CT4PJ's wrong-place share is over twice CT3F's on both models. Then freezing would lock errors in, and the design question moves to when to commit, not how to hold.
- Both can hold at once; then both are reported.
