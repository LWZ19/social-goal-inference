
Written before running anything below on the confirm runs. Exploratory: the confirmatory test is scored
(`round4/CONFIRM_RESULT.md`), and nothing here is a claim.

Starting point:
- `round4/q_confirm_explore_out.txt` ec0459bd: CT3F's goal jumps 8.9 and 7.6 times a minute after the first service, against CT4PJ's 13.7 and 13.9.
- On the layouts where CT4PJ trails CT3F, CT4PJ jumps more on 28 of 34 and 37 of 44.
- Rank correlation of d(strict) with d(jumps): −0.25 and −0.29.

The cause could run either way:
- **(a)** the role belief's roster flickers, so the goal jumps and the robot steps out of the line;
- **(b)** once out of the line, CT4PJ's join rule and its "behind" rule are off, so its goal comes from the bare role belief and jumps more.

## Questions (both models' confirm runs; the pilot only to test the code)

- **J-a. Base rate.** On how many of all layouts does CT4PJ jump more than CT3F? Without it, "28 of 34" can't be read.
- **J-b. Kind of jump, per arm, from the first service on.** A jump is consecutive ok re-solves more than 0.5 m apart, as before. It is a:
  - **switch** when the re-solve's `last` changed;
  - **same person** when `last` stayed, and the person moved or the direction turned.

  Reported as pooled rates per minute.
- **J-c. Jumps by the sim's joined state.** Rates per minute while the robot is in the sim's line, and while it is not. Each jump takes the state of the latest 1 Hz sample at or before it.
- **J-d. Leaves after a jump.** For each of the sim's `left` events after the first service: is there a jump (any kind; switches alone) in the 5 s before it?
  - Compared with the base rate: the share of joined seconds with a jump in the 5 s before.
  - Their ratio is the lift.
- **J-e. Where the CT4PJ − CT3F gap sits.** The share of the total gap from layouts where CT4PJ's join test never fired, against those where it did.

## How each answer will be read, fixed now

- **Supports (a).** All three must hold:
  - CT4PJ's excess jumps over CT3F are mostly switches, and are present while joined;
  - the switch lift before CT4PJ's leaves is at least 2;
  - on both models.

  Then goal flicker is a plausible cause of losing the place, and the decision layer is the next design question.
- **Supports (b).** CT4PJ's excess jumps are mostly while not joined, and the lift is below 2. Then the jumps follow from being out of the line, and no design follows from jitter.
- **Anything else** is mixed and reported as such.
- J-a and J-e are context for those readings, not tests.
