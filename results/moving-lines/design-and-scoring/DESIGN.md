# Moving people: ask once, re-solve on tracks (design, 2026-11-01)

## Question

The constraint method's answer names people ("come after 4, 5, 6"). Once that answer exists, can the solver keep
the robot's goal right while people move (the line advances, the front person is served and leaves, people
rejoin), by re-solving on tracked positions with no further model call? How does that compare with asking again,
for the baseline and for the method, in time spent correctly in line, cut-ins, model calls and staleness?

Hypotheses (stated before any run):
- H1. C asked once and re-solved on tracks (CT) is in line at the end of an episode at least as often as every
  re-asked arm, with one answer's calls instead of one per re-ask.
- H3. The re-asked arms (Rq, Cq) lose time to latency: each re-ask is a fresh view answered 5-60 s later, by which
  time the line may have moved again; the slower the model, the larger the loss.
- What would surprise us: CT losing its place (taken by a rejoiner, or following a served person away), or Rq
  matching CT on the share in line at the same S.

## Arms (one episode = one scenario, one seed, one model)

| Arm | Goal method | Asked | After the answer |
|---|---|---|---|
| R1 | marked point (cot) | once, at t = 2 s | nothing |
| Rq | marked point (cot) | at t = 2 s, then every S s for the whole episode | each answer replaces the goal past 0.3 m / 25 deg |
| C1 | constraints | once | nothing |
| Cq | constraints | at t = 2 s, then every S s | as Rq |
| CT | constraints | once | re-solved on tracks every 0.5 s; same replace rule |

The first frame is fixed by scenario and seed, so C1, Cq and CT share their first answer (the cache), and R1 and
Rq share theirs: the arms are paired.

**Latency in sim time.** An answer is delivered at sim time t_sent + L, where L is the answer's reported total
time (`timings_s.total`; for a cached call, the stored latency of the call that was made). So a cached first
answer arrives as late as the live one did, whatever the order the arms ran in. A re-solve is delivered after its
own measured time. The sim never runs faster than real time.

## Tracking (oracle)

Every person's true position and heading from the sim, with a stable id, for those within 12 m of the robot
(the solver's floor range). This is an upper bound on tracking; a detector-based tracker is a stretch goal.

## Re-solve rules (`sgi.resolve`, a pure function with unit tests)

1. **First match.** At the first re-solve, the VLM's numbered people (positions lifted at t0) are matched one to
   one to the tracks as they stood at t0 (Hungarian, gate 0.75 m). A matched person keeps their number. A named
   person with no match is stripped from every constraint. A track with no match gets a new number, unlisted.
   The match rate is reported per model.
2. **Left the line.** A person named in a `come_after` whose distance to the target has grown by more than
   0.75 m beyond their closest approach is dropped from the constraints, for good. If they come back they are an
   unlisted person, whom the solver's place-in-turn rule absorbs when they stand behind the end.
3. **Joined.** Once the robot has stood within 0.35 m of a goal that used a place in turn for 1 s, it counts as
   the end of the line: from then on, people farther from the target than the robot are left out of the solve
   (they stay obstacles in the depth image). This stops a rejoiner behind the robot being absorbed as the new
   last and the robot being sent back.
4. **Solve.** `solver.solve` on the fresh observation (floor and seen face from the current depth), the target's
   world point from the first answer, the standard goal of the first answer, the tracked people and the stored
   constraints.

## Sim changes (all arms alike)

- **The robot as a queue member.** Once the robot has arrived within 0.3-1.5 m behind the last member, within
  0.6 m of the line's axis, it is put in the queue's member list (at the end), so a person who rejoins walks to
  the slot behind it rather than into it. It is never served. It leaves the list if it moves more than 1.5 m from
  where it joined other than up the line.
- `--requery-always`: keep asking every S s for the whole episode, not only until arrival.
- Answers delivered at t_sent + L (above). An episode log (`episode.jsonl`): every 1 s, the robot's pose, the
  queue members, the active goal, and every answer and re-solve.

## Scoring (frozen in SCORING.md before any model episode)

See SCORING.md.
