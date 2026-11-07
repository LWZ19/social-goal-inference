# Round 4, part C: staying in line once joined (3 Nov, post hoc, pilot design data; `d_joined/`)

## Time in line is time spent joined

After the first service, strict in line is 0.22-0.35 while the sim has the robot joined, and 0.01-0.02 when it
does not. So time in line is mostly about joining and staying joined.

| Arm | joined share of samples | joins | leaves per join | median time joined before leaving |
|---|---|---|---|---|
| CT3F | 0.51 | 23 | 0.61 | 30.5 s |
| CT4 | 0.22 | 21 | 0.71 | 14.0 s |
| W0 | 0.21 | 17 | 0.76 | 16.0 s |
| CT5 | 0.14 | 14 | 0.86 | 8.5 s |

## A third package difference: CT3F stops counting people behind it once it has joined

- `Resolver.inputs` (`resolve.py` line 410): once the resolver counts itself as joined (in turn, near the place for
  1 s), people behind the robot along the line are left out of the solve. Nobody who queued after the robot can
  pull its goal back.
- `_resolve_with_belief` has nothing like this. It logs `"joined": False`, and every re-solve considers everyone.
- While the sim has the robot joined, the solver's last stands behind the robot on:

  | CT3F | CT4 | W0 | CT5 |
  |---|---|---|---|
  | 0.6% | 8.3% | 18.4% | 14.2% |

  - For the role arms, 79-90% of those people are waiting: they joined after the robot.
  - About a third of the role arms' leaves come within 3 s of such a re-solve (CT4 5/15, W0 5/13, CT5 4/12).
    CT3F: 1/14.

## What it means

- **The belief answers the wrong question once the robot is in line.** It asks "who is waiting?" and gets newcomers
  behind the robot right. The question that sets the goal is "who is waiting ahead of me?". So the role arms give
  their place to people who queued after them.
- **The robot is part of the structure it infers.** A queue is an order, and once the robot joins, its own place in
  that order is part of the state. A belief over sets of waiting people, with the robot outside them, cannot
  represent it.
  - CT3F handles this with a rule, "joined, so ignore people behind".
  - The principled version puts the robot into the order and decides on "the last person ahead of me". It reduces
    to the same filter in practice. The point is that it is part of the problem's structure, not a patch.
- **The packages compared in C1 and round 3 differ in at least three parts outside the belief over roles:** the
  direction rule, this commitment rule, and the vacated place (which turned out not to matter). A test of the belief
  itself needs the first two equalised. CT4P equalises the direction only.
