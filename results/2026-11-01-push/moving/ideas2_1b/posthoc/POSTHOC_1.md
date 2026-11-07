
Written before computing anything below. C1 failed (`moving/sweep/score_dsv41_out.txt`, sha256 3791113d). Nothing
here changes that verdict or any frozen file. The 72 sweep layouts are spent for confirmatory use: whatever this
finds can only shape a new pre-registration on new layouts.

**Data.** `moving/runs_sweep/dsv41/sweep/*/seed*/<arm>/episode.json` only: per re-solve `truth`, `last`, `roster`,
`marginals` (CT4), `in_turn`, `joined`, `vacated`, `line_direction`, `tracks`; per 1-s sample `in_line`, `cut_in`,
`state`. I do not open `moving/sweep/human_dump/` or `moving/sweep/renders/`.

**Ticks.** As the scorer: re-solves with truth and someone ahead (`truth.ahead` non-empty).

## Questions

**Q1. Why do the wrong-last ticks go wrong?** Each scored tick, per arm, falls in exactly one class:
- right: `last == truth.last`, not vacated;
- vacated;
- empty: `last` is None;
- bystander: `last` is one of the episode's `bystanders`;
- ex-member: `last` is not in `truth.order` now but was earlier in the episode (served, or left);
- other non-member: not in `order`, never was, not a bystander;
- short: `last` is in `truth.ahead` but is not `truth.last` (a member ahead of the true last); split by whether
  `truth.last` is in the roster (listed but not chosen: an ordering failure) or not (dropped);
- behind the robot: `last` is in `order` but not in `ahead`.

**Q2. CT4's tail.** On scored ticks: P(truth.last); whether anyone in the roster stands farther back along
`line_direction` than `truth.last`; how many marginals sit in 0.35-0.65; the share of roster members that are in
`truth.order` (precision) by marginal bin (calibration).

**Q3. Does a right last reach the robot?** For each arm, the 1-s samples after the first delivery, each matched to
the latest re-solve sent at or before it:
- share in line, and share cut-in, when that re-solve's last was right versus wrong;
- share of right-last re-solves with `in_turn` False (the place behind the last was not used).

**Q4. Why is CT4 in line less than CT3F on "ours" layouts after the first service?** Same split as Q3, restricted
to the 40 "ours" layouts and to samples after the first service (the scorer's definition).

## What would change the next design

- If CT4's wrong ticks are mostly ex-members or bystanders behind the line, the gap is tail precision, which recall
  doesn't measure. The next estimator needs exit evidence (served people leave), not more recall.
- If right-last ticks often leave the robot out of line or cutting in, the gap is downstream of the roles (the
  solver's place in turn, or control), and a better role estimate cannot show up in time in line.
- If short ticks with the true last listed dominate, the problem is ordering, not membership.

## Added after seeing Q1-Q4 (still exploratory)

**Q5. Does the goal jitter?** Per arm, over re-solves after the first delivery: the share of consecutive re-solves
whose `pose_world` moved more than 0.3 m, and whose roster changed; and per 1-s sample, the share whose goal moved
more than 0.3 m from the previous sample.

## Results (3 Nov; `posthoc1_out.txt`, `posthoc1_q5_out.txt`; exploratory)

**Q1. Where wrong lasts go (all layouts, share of scored ticks).**

| | right | empty | bystander | ex-member | other non-member | short, last dropped | short, last listed | behind robot |
|---|---|---|---|---|---|---|---|---|
| CT3F | 0.382 | 0.209 | 0.110 | 0.036 | 0.038 | 0.148 | 0.012 | 0.054 |
| CT4 | 0.420 | 0.040 | 0.117 | 0.086 | 0.112 | 0.111 | 0.058 | 0.056 |

CT4 swaps CT3F's empties (-0.17) for people who are not in the queue: served people still on the roster, and people
who never queued (+0.12 together). Bystander errors are equal. Both drop the true last about as often.

**Q2. CT4's belief is overconfident in the middle.** People at 0.5-0.7 are queue members about a third of the time
(0.34, 0.32); at 0.8-0.9, 0.63; at 0.9-1.0, 0.85. There are 0.76 people per tick in 0.35-0.65. The true last is
usually on the roster (P >= 0.5 on 75% of ticks), but on 73% of ticks someone on the roster stands more than 0.2 m
farther back than the true last. High recall comes from a liberal roster, not a sharper one.

**Q3 and Q4. A right last does not reach the robot under CT4.**

| | in line when last right (all / "ours" after service) | right-last re-solves not using the place behind the last | robot moving |
|---|---|---|---|
| CT3F | 0.429 / 0.416 | 0.441 | 0.51 |
| CT4 | 0.227 / 0.279 | 0.686 | 0.67 |

**Q5. Why: the roster flickers and the goal follows it.** Roster changed between consecutive re-solves: CT4 0.172,
CT3F 0.051. Goal moved more than 0.3 m: CT4 0.198 of re-solves (median 22 times a minute), CT3F 0.127 (11).

**Reading.**
- The offline replay could not see this, because the robot never moved there. Right last measures perception; time
  in line measures behaviour, and the arms differ in the step between them.
- For the next design: decide with the belief (expected cost over its sets, with the cost of moving counted), on a
  calibrated belief. Develop it in closed loop on design layouts, not offline. Use a behavioural primary measure.
