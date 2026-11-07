# CT2: re-solving v2 (frozen 2026-11-02 ~01:00, before any fresh-seed run and before CT2 is built)

## CT2 = CT v1 plus

1. **Re-admit** (a bug fix, found on seeds 0-2): a person dropped from the waiting set whom the solver takes as joining
   behind the last in 2 solves running is waiting again (no longer dropped; their closest approach starts afresh).
2. **Line span 0.5 m** (implementation fix): the remembered line needs its front and back at least 0.5 m apart (was
   1.0 m, which no two-person line with 0.7 m spacing reached).
3. **Vacated place:** when nobody the robot came after is still waiting (all left), and the last of them has been
   away for at least 5 s (still more than 0.75 m from the remembered line), the goal is the last still pose (position
   and heading, at the last re-solve where they had moved under 0.10 m in 1 s) of the person who left last, instead of
   the solver's fallback to the answer's target. It follows from come_after: you come after X; X is gone; you take
   X's place.
4. **Forward ratchet** (sim side, replace rule): a new goal replaces the current one when it lies more than 0.08 m up
   the line (along the remembered direction, towards the front); moves in any other direction still need 0.3 m or
   25 deg; and a goal someone now stands on (a person's centre within 0.55 m) is replaced whatever the move.
5. **Path memory:** the line is remembered as the polyline through the waiting people's answer-time positions
   (front to back), not a straight axis: "off the line" is the distance to that polyline (extended past both ends),
   "farther back than the robot" is arc length along it, and the direction given to the solver is the polyline's
   tangent at the waiting person farthest back. For a straight line this equals the straight axis.

Not in CT2 (separate, labelled, optional): the stillness gate (dropping a named person who was moving at the answer),
which overrides the model's answer.

## Arms on fresh seeds

Primary: **CT2 against CT v1** (paired by scenario and seed; same cached first answer).
Ablations (no calls): **CT2-vac** (CT2 without 3) and **CT2-rat** (CT2 without 4).
Secondary: CTs (frozen separately), R1, C1. Redraw spread: CT and CT2 run twice on seeds 10-12.

## Scenarios

The 5 evaluation scenarios, with `room_coffee_cart` and `cafe` given extra queue slots (the seeds 0-2 runs had 4 and 5
slots, so a person rejoining was placed onto a robot queued at the back: a sim artefact). The extended copies are
`room_coffee_cart_slots.yaml` and `cafe_slots.yaml`; every arm on fresh seeds uses them.

## Scoring

Primary as frozen in SCORING.md, plus a **second scoring, frozen now**: first in line counts only within **0.6 m** of
slot 0 (was 1.6 m), everything else unchanged. The post hoc measures (Q9/Q10/Q12) and the event measure of
IDEAS_social_eval B2 (per service event with someone ahead: back in line within 10 s, without a cut-in) are reported
for every arm.

## Predictions (written before any run)

- CT2 against CT v1: more service events kept up, fewer cut-in samples (the "lost after joining" spans), more
  "reached the front in turn"; share in line higher under both scorings; the largest gains in the two-person lines.
- CT2-vac loses most of the cut-in reduction; CT2-rat loses most of the near-miss reduction (gap 1.2-1.5 m samples).
- Step-out case (cafe_line6_stepout, run on seeds 10-12 for CT and CT2): the stepped-aside person is 1.5 m off the
  line, so the leave rule drops them at once in both CT and CT2; the robot can move into their place only if they were
  the last person ahead of it. The 5 s vacate delay is shorter than the 7 s step-out, so if that happens CT2 may take
  their place and be counted cutting in when they return. Prediction: no more cut-in samples for CT2 than for CT in
  that case; any CT2 cut-in that starts during the step-out is reported as CT2's failure.
- Bent line (cafe_line_bent_moving, seeds 10-12, CT and CT2): CT2's share in line above CT's (CT 0.04 on seeds 0-2).
