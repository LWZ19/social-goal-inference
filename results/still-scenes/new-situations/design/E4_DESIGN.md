# E4: two new test situations (decision, 6 Nov 2026). Written before any scene exists or any model call.

## 1. A waiting line that bends (situation `waiting_line`, new layouts)

- The line starts at a place (a counter or a desk) and, after 2 or 3 people, turns by about 90 degrees round
  something that gives the bend a reason (a pillar or a planter). Sizes: 4 people (bend after 2) and 6 people
  (bend after 3). Spacing as the existing lines (0.6 m).
- Success area: the existing waiting-line rule, unchanged (src/sgi/gt/regions.py `region_waiting_line`): the half
  ring 0.5-1.2 m round the last person that is not ahead of them, ahead meaning toward the person before them, so the
  direction is the second leg's; facing the way the line faces. Forbidden: the corridor along the whole bent line.
- What it tests: whether the robot joins the end of the second leg, not a straight extension of the first.

## 2. Someone using a machine (new situation `machine_use`, a held-out situation)

- One person stands at a cash machine or a ticket machine, facing it, about 0.5 m from its front, using it. Nobody
  else waits. Two variants: a cash machine and a ticket machine.
- Reported beside it, fixed now (sensitivity, no bar): the same rule with the near limit at 0.8 m.

## Scenes and runs

- Placeholder renderer, as the hidden lift and photo scenes. Fresh generator seeds 600, 601, 602 (unused: 3-99 sealed,
  100-219, 400-449, 500-509 used). Per seed: bent line of 4, bent line of 6, cash machine, ticket machine; two
  requests each (a grounded one and a high-level one), 3 trials: 36 records per situation, per method, per model.
- Scored by the package scorer; per situation and per model, success of each method; descriptive.
