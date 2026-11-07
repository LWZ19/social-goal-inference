# Round 4, part D: where the belief's false members come from (3 Nov; post hoc, pilot design data; `f_belief/`)

Written while CT4-PJ runs, before its result. Groundwork for CT4PJ.md's "J5 holds" branch.

## Non-waiters counted as waiting (P(waits) ≥ 0.5; for CT3F, its roster), by history (`exmember_out.txt`)

| Arm | ticks | bystanders the answer named | ex-members (served, walking off) | unnamed walkers | other |
|---|---|---|---|---|---|
| CT3F | 2,322 | 1,448 | 534 | 119 | 221 |
| CT4 | 3,415 | 1,453 | 1,171 | 592 | 199 |
| W0 | 3,412 | 1,368 | 1,345 | 526 | 173 |
| CT5 | 3,213 | 1,242 | 1,337 | 452 | 182 |

- **Ex-members** drop below 0.5 a median 4.4 s (CT4) and 6.1 s (W0) after leaving the queue. CT3F takes 2.6 s.
- **Bystanders the answer named.** The answer named 31 bystanders and 10 walkers as waiting, beside 54 waiters
  (`named_out.txt`). These come from the first answer, and CT3F and CT4 carry the same number of them.

## Which of them become the solver's last (share of ticks with someone ahead; `lasts_out.txt`)

| Arm | any non-waiter | named bystander | ex-member | unnamed walker | unnamed bystander |
|---|---|---|---|---|---|
| CT3F | 0.227 | 0.082 | 0.057 | 0.023 | 0.059 |
| CT4 | 0.343 | 0.133 | 0.106 | 0.082 | 0.009 |
| W0 | 0.372 | 0.120 | 0.151 | 0.076 | 0.015 |
| CT5 | 0.487 | 0.158 | 0.156 | 0.111 | 0.025 |

CT4's excess over CT3F (+0.116) splits roughly evenly three ways:
- ex-members: +0.049;
- unnamed walkers: +0.059;
- named bystanders as last: +0.051;
- partly offset by fewer unnamed bystanders absorbed into the line: −0.050.

## The idea this points to: a belief shaped by the activity's process

The role belief's dynamics are generic: independent per-person hazards (join 0.05/s after arriving and stopping,
leave 0.05/s while walking), plus co-motion. A queue is not a set of people with independent hazards. It is a FIFO
process, and that structure decides where people come and go:
- **Departures happen at the front, at service.** The person nearest the counter who walks away is being served,
  not drifting off at 0.05/s. The generic model keeps served customers at P ≥ 0.5 for 4-6 s while they walk past
  the line. That puts the robot's goal behind a moving person: 59% of CT5's goal jumps (PH-A).
- **Arrivals happen at the tail, along the line.** A joiner arrives behind the last along the line's direction. The
  static snapshot scores "one spacing from the person ahead" by distance only, so a walker pausing beside the line,
  about 1.1 m off the axis and level with the tail, passes as a joiner.
- **Step-ups are collective.** CT4 already models this (co-motion).
- **The observer is part of the order.** Once the robot joins, only people ahead of it matter (`RESULTS_C.md`,
  tested by CT4-PJ).

CT3F has a hand-written rule for each of these: `gone` for front departures, group follow for step-ups, the joined
filter, and the vacated place. The role belief was meant to replace hand-written rules with one model, but it
dropped the structure that the rules encode.

So the proposal is to keep one belief, but make its transition model the queue's process:
- front-of-line departures at service;
- arrivals at the tail along the line;
- collective step-ups;
- the robot's own place in the order.

This is not a fix for one model's weakness. It holds for any FIFO-like social process: counters, boarding, lifts.
The evidence above says which parts carry weight: departures and arrivals.

## Before any of this is built

- CT4-PJ decides the branch. If J5 fails (CT4-PJ reaches CT3F), the belief is not the bottleneck and this waits.
- The test bar would be offline first: lower ex-member and walker false-member ticks than CT4, with no loss of
  recall. Only then a closed-loop pilot.
