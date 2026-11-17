# Ideas from robotics, control, state estimation and tracking (2026-11-02)

## 0. Where CT's time goes (evaluation scenarios, CT, seeds 0-2, 15 episodes per model)

Method: each re-solve's logged goal is judged against the scorer's live last-ahead person. Their position is
rebuilt from each sample's `along`/`behind_m`/`gap_m` and the scenario's slot polyline. "Of ahead" means the share
of samples in which someone is ahead.

| CT, share of ... | dsv41 | sonnet55_low | x3 |
|---|---|---|---|
| all samples before the first answer arrives | 0.23 | 0.18 | **0.48** |
| ahead: latest re-solve has a place in turn and is valid | 0.33 | 0.43 | 0.45 |
| ahead: **no place in turn, robot never joined** (class A, a wrong or unusable first answer) | **0.38** | 0.21 | 0.30 |
| ahead: **no place in turn after joining** (class B, "the line was forgotten") | 0.19 | **0.25** | 0.12 |
| ahead: has a place in turn, but invalid (far, heading, off the axis) | 0.10 | 0.11 | 0.09 |

**Class B.** Lasting losses after joining (5 s or more): 7 / 8 / 5 episodes, **0.137 / 0.193 / 0.059 of all
samples** and **70 of 146 / 126 of 140 / 16 of 46 CT cut-in samples**. Traced (dsv41 cafe_line2_moving s1; sonnet
cafe_line2_moving s0-2, sweet_spot s0): the last named person ahead is served, the waiting set empties, and the
solver falls back to the answer's target pose, level with slot 0 but 0.8-1.0 m to its side. The robot walks there,
the simulator takes it out of the queue (side > 0.8 m), and returning people line up beside it: about 55 s scored as
cutting in.

**Class A** has three causes: no `come_after` (sonnet, 3 of 15); a `come_after` naming only the target person (dsv41
cafe_line6 s2, target point on the last person); or a place in turn off the first view's floor or a walker named in
the `come_after`. Offline, `sgi.solver.solve` on the stored `001/` inputs reproduces **45 of 45** first-answer poses
to within 6 cm. Dropping the model's `within_distance` on the target changes `turn_m` in **0 of 14** cases (so that
conflict is **not** the cause); cafe answers keep `turn_m` 2.1-6.8 m with `come_after` alone; and **8 of 45** first
answers name someone more than 0.8 m off the true line in their `come_after` (6 of 9 cafe answers, all the walker
"2" at about 1.1-1.25 m). Cafe is the lowest-scoring scenario for every arm.

---

## 1. Take the vacated place: what `come_after` means once the people ahead have gone (top pick)

- **Targets:** class B. At most it recovers 0.137 / 0.193 / 0.059 of all samples and 48% / 90% / 35% of CT's cut-in
  samples.
- **Mechanism (general).** `come_after [P...]` means "occupy each P's place once P is done". The resolver keeps each
  waiting person's last still pose: where they stood and the way they faced, after at least 2 s within 0.1 m. Once
  the robot has joined and everyone it came after has left (dropped or gone), its goal is the last still pose of the
  person who left most recently, facing their waiting heading, not the answer's target pose.
  - Guards: the pose is on the navigation map; it was held while the person counted as waiting (this excludes staff
    moving about behind a counter); nobody still waiting stands ahead of the robot.
  - It works the same for a counter, lift door, ticket machine, printer or Q&A microphone: the place of service is
    learned from the person who used it, not from the model's pixel.
- **Why it is new.** The resolver now falls back to the static standard goal. DEVIATIONS 3/v6 and Q5 leave "at the
  front it goes to the model's wrong target point" as a model-answer effect. LANDSCAPE lists no work with a terminal
  rule for a re-bound social spec.
- **Cheapest check (zero calls).** A CT variant on the 5 evaluation scenarios × seeds 0-2, for dsv41 and
  sonnet55_low: 30 episodes, about 50 min on 3 workers. Abort if any `cached` flag is false, so Claude is never
  called. Report beside share: cut-in samples, "reached the front in turn" (Q9; CT now 2 of 15 for dsv41) and "lost
  place". Already measured offline: in all 20 spans the robot had been at or just behind the front.
- **Prediction / refutation.** Prediction: cut-ins fall by ≥ 40% (dsv41) and ≥ 70% (sonnet), and "reached the front"
  rises to ≥ 5 of 15. Refuted if the line2 cells' cut-ins do not fall, or if the robot walks behind the counter.
- **Risk.** The share gain is inflated by the "never served" artefact (once first, it stays first), so lead with
  cut-ins and "reached the front", not share. **Effort:** about 2 h (a track-history buffer in `Resolver` and one
  branch in `serve.resolve`).

## 2. A forward ratchet and a centred goal: hysteresis that knows which way the line moves (top pick, measured)

- **Targets:** near misses that the 0.3 m rule freezes (Q13), and the redraw noise.
  - Settled samples at 1.2-1.5 m from the person ahead are 5.0% / 9.2% / 5.2% of all CT samples.
  - In them the robot is at its goal (median 0.09 m away), yet the goal is 1.13-1.20 m behind the person ahead.
  - The place in turn is aimed at `standoff_m` = 1.0, only 0.2 m inside the band's 1.2 m edge. A 0.24 m mid-step
    error lands outside the band, and is then held.
- **Mechanism (general).** Under `come_after` the robot's place only moves toward the front (people ahead only
  leave). Accept a new solve at once if it moves the goal more than 0.08 m forward: along the remembered line
  direction, or toward the target point when none is remembered. Otherwise keep the 0.3 m / 25 deg rule.
  Optionally aim re-solves at the middle of the 0.5-1.2 m acceptance band (0.85 m); the goal then stays valid
  longest as the person ahead creeps forward. The band is the static rule from RULES.md and the mentor paper, not
  fitted here.
- **Measured (offline replay).** Replaying the logged 0.3 m rule reproduces the logged goal in 99% / 98% / 97% of
  samples. Share of ahead samples with a valid goal (goal changes per episode in brackets):

  | Rule | dsv41 | sonnet | x3 |
  |---|---|---|---|
  | Logged 0.3 m rule | 0.299 (11.0) | 0.389 (11.9) | 0.381 (9.5) |
  | **Ratchet** | **0.334** (13.1) | **0.445** (14.7) | **0.454** (11.5) |
  | Ratchet + 0.15 m centring | **0.343** | **0.465** | **0.472** |
  | Always replace | 0.333 (132) | 0.435 (142) | 0.475 (90) |

  In overall share: about +0.02 / +0.04 / +0.04 for the ratchet, +0.03 / +0.06 / +0.05 with centring. CTv (dsv41)
  goes from 0.267 to 0.307. Caveat: the replay assumes re-solves do not depend on where the robot stands; its
  position enters only through tie-breaks and the "joined" filter.
- **Why it is new.** Q13 suggests "replace when the goal leaves its valid place", and CTs waits for settled tracks.
  This is an asymmetric deadband derived from the constraint's monotone progress: CTs freezes, the ratchet does not.
- **Next check (zero calls).** Run the `dsv41_repeat` cells (room_coffee_cart and cafe_line2_moving × seeds 0-2)
  twice with the ratchet.
- **Prediction / refutation.** Prediction: mean |Δshare| between repeats falls from 0.10 to ≤ 0.05 and the maximum
  from 0.50 to ≤ 0.25, with the mean share not lower. Refuted if the spread stays, or goal changes exceed 25 per
  episode (oscillation).
- **Risk:** low. **Effort:** 1 h (`_replace_goal`, plus the direction passed along with the re-solve).

## 3. The line is a place: remember its path and re-acquire members by where they stand (top pick)

- **Targets:** class B, plus three related failures.
  - **All named people dropped after a long latency.** x3 cafe s2: the answer arrives at 52.7 s, 2, 3 and 4 are
    dropped at the first re-solve, share 0.00. Also dsv41 sweet_spot s1 and x3 sweet_spot s2 (both 0.00).
  - **The bent line (share 0.04).** The remembered axis is a straight chord from front to back. The corner slot lies
    **1.15 m** off it, beyond `off_line_m` = 1.0, so everyone stepping through the corner is dropped for good. In
    dsv41 bent s0 all 7 are dropped by t = 75 s.
  - **Bent-line heading.** The chord points at −123°, the real segments at −146° and −90°: 23-33° off.
- **Mechanism (general).** Keep the path: a polyline through the waiting people's still positions, front to back,
  extended as people step up. A FIFO line shifts along its own path, so the path outlives its people.
  - At each re-solve, anyone still within 0.4 m of the path, between its front and the robot, counts as waiting,
    numbered by the model or not.
  - "Left the line" means more than 1 m from the path, not from a straight axis.
  - The place in turn extends the path's last segment; "up the line" is the path's local tangent.
  - The identity rules remain the first evidence; the path takes over when they run dry. It covers curved lines,
    lines round a pillar and lines along a wall.
- **Why it is new.** The resolver remembers a straight direction and identities only; here the constraint's referent
  becomes a place with memory. LANDSCAPE has nothing that re-binds a spec to a place.
- **Cheapest check (zero calls).** cafe_line_bent_moving and cafe_line6_stepout × seeds 0-2, dsv41 cached (6
  episodes), then a regression run on the 15 evaluation episodes. The chord distances above are measured offline.
- **Prediction / refutation.** Prediction: bent share rises from 0.04 to ≥ 0.25, no episode loses all its waiting
  people, and stepout cut-ins do not rise. Refuted if bent stays below 0.10, or walkers crossing the path are
  absorbed (watch `people` and the cut-ins).
- **Risk.** A walker who stops on the path is absorbed; idea 4's stillness gate bounds this. **Effort:** about 3 h.

## 4. A stillness gate on who waits and on the line's direction

- **Targets:** class A cases where `come_after` names a walker: 8 of 45 first answers (x3 cafe s0-2, sonnet cafe s0
  and s2, dsv41 cafe s1, one each in room_coffee_cart and line6). The walker becomes the solver's "last" and skews
  the remembered direction: (0.42, −0.91) in x3 cafe s2, 25° off the true axis. The models even label such people
  as walking (sonnet cafe s2 "walking"; a dsv41 cafe s0 `visible_to` "walking toward the counter").
- **Mechanism (general).** Waiting implies standing still.
  - A named `come_after` person moving faster than 0.25 m/s, over the 2 s around the photo or now, is "pending":
    not used as the last, nor for the direction.
  - They are promoted after standing still for 2 s within 0.4 m of the place in turn or the path.
  - The direction is re-estimated, from the still members only, whenever that set changes.
- **Why it is new.** It uses the tracker's velocities; the resolver today uses positions only.
- **Cheapest check (zero calls).** cafe × seeds 0-2 for all 3 models: 9 episodes.
- **Prediction / refutation.** Prediction: cafe CT share rises from 0.07 / 0.20 / 0.01 to ≥ 0.2 for at least 2
  models, with the remembered direction within 10° of the true axis. Refuted if the gate drops real members as they
  step up: steps also move at about 0.9 m/s, so let a step of at most one slot toward the front pass.
- **Risk:** low. **Effort:** 1-2 h.

## 5. Watch during the latency: learn the line's frame from service events before the answer arrives

- **Targets:** class A and B geometry (target point, front, direction, missing members), and the 18-48% of samples
  before the answer.
  - Services seen *during* the first answer's latency: median 1 (dsv41), 0 (sonnet), **2** (x3).
  - First service after delivery: median 8.2 / 4.3 / 12.6 s, and every episode has one.
  - Tracks from the photo onward are buffered anyway, for the matching at answer time.
- **Mechanism (general).** A *service event*: a person still for at least 5 s leaves (more than 1.5 m), and within
  6 s one or more still people each move 0.4-1.0 m in nearly the same direction, toward the spot that was left. The
  events give the **sink** (where the leaver stood), the **direction** (mean step-up vector), the **spacing**, the
  **members** (who stepped together) and the **service period**. When the answer arrives, its numbered people are
  matched to this frame, which feeds ideas 1, 3 and 4:
  - a `come_after` naming only the front gets everyone who stepped up with that person;
  - ordering and the front goal use the sink, not the model's pixel target. That pixel was 4 m off in DEVIATIONS
    3/v5 and on a person in dsv41 line6 s2;
  - facing uses the step-up direction, which fixes "face the target, 60° off" generally.

  Any FIFO turn-taking makes this signature; nothing is specific to counters or queues.
- **Why it is new.** LANDSCAPE (§1, D1 gap 2) proposes an event *taxonomy* for scoring; here events are the
  *measurement* used to estimate the spec's geometry, and the wait for the model becomes an observation window.
- **Cheapest check.**
  1. Zero calls: in simulation, log full tracks for the 15 evaluation episodes of one CT arm (cached answers). Score
     the detector against the simulator's member decrements: precision ≥ 0.9 and recall ≥ 0.8 on services.
  2. Then the frame-fed CT variant on the 45 evaluation episodes.
- **Prediction / refutation.** Prediction: class-A share halves, from 0.38 / 0.21 / 0.30 of ahead samples. Refuted if
  detector precision is below 0.8 with walkers present (cafe, sweet_spot).
- **Risk.** In a 2-person line a service can look like a reshuffle. **Effort:** 4-5 h.

## 6. BOLD A: a belief over line hypotheses, with the model's answer as the prior

- **Mechanism.** Keep 3-10 hypotheses h = (ordered members, sink, path). Seed them from the answer (its
  `come_after`, its target point) and from track-only alternatives (idea 5's chains, the stillness-gated set).
  - Weight each by a motion likelihood: members still between events; step-ups toward h's sink; departures from h's
    front; people outside h not stepping in sync.
  - The goal maximises P(valid) over the posterior: the cell that is a valid place under the most probability mass.
    This is "goal as a region", and it degrades to the CT point when the posterior is sharp.
  - Commit (dwell and count as joined) only when the top hypothesis has at least 0.8.
  - Re-ask once (4 calls) only when the answer-seeded hypotheses together fall below 0.2: a principled "the first
    answer is wrong" detector from tracks alone.
- **Targets:** class A (0.21-0.38 of ahead samples) and class B together.
- **Why it is new.** MIGU (mentor group) fuses VLM priors into a static grounding belief. Nobody updates a belief
  over a social spec with turn-taking dynamics, or uses its collapse as the re-ask trigger.
- **Cheapest check.**
  1. Zero calls: a 40-line kinematic simulator (no Habitat) driven by the scenario YAMLs (slots, `service_s`, wander
     areas), with each first answer's people and constraints from `001/answer.json`. Measure posterior accuracy
     over time against the true members.
  2. Then simulation with cached answers.
- **Prediction / refutation.** Prediction: the true hypothesis has more than 0.8 within one service in 80% of
  evaluation cells. Refuted if 2-person lines (line2, sweet_spot) never separate the hypotheses.
- **Score:** payoff 4/5, uncertainty 4/5, novelty 5/5, cost 10-15 h. Worth writing up as future work even if not
  built.

## 7. Staged departure: act on the plan stage while the constraints are pending (+ BOLD B)

- **Targets:** time before the first answer (0.23 / 0.18 / 0.48 of samples) and travel time.
  - The target point is known after the plan stage: median 4.5 of 19.2 s (dsv41), 3.9 of 15.1 s (sonnet), 9.2 of
    43.4 s (x3), i.e. 23-27% of C's latency. From answer to first in line, CT takes a median 8.6-9.4 s.
  - A crude bound, splicing R1's samples before C arrives with CT's after: 0.188 vs 0.182, 0.203 vs 0.198,
    **0.153 vs 0.118**. It matters only for slow models.
- **Mechanism (general).** On the plan stage, drive to a staging pose on the robot-target ray: 3 m from the target
  (`max_keep_m`), at least 1.2 m from every track, facing the target. CT then starts from there.
- **BOLD B.** If idea 5 has already seen a turn-taking structure whose sink is within 2 m of the plan-stage target,
  provisionally go to its back; the constraints confirm or override it. Score: payoff 3/5 (largest for x3),
  uncertainty 4/5 (it can join a line the model would rule out), cost 3 h.
- **Cheapest check (zero calls).** The 15 evaluation episodes for x3.
- **Prediction / refutation.** Prediction: median time to first in line falls by ≥ 5 s and x3 share rises by ≥ 0.03,
  with no cut-in samples before the answer. Refuted if staging adds cut-ins (approaching from the line's front
  side). **Effort:** 2 h.

## 8. Re-ask from a vantage point, only on surprise

- **Evidence a plain re-ask rarely fixes class A:** Cq's later answers are valid at delivery in 0.23 / 0.21 / 0.20 of
  cases overall, and 0.14 / 0.20 / 0.00 in the cells where CT never had a place in turn.
- **Mechanism (general).** On a monitor (idea 6's collapse, or no place in turn for 8 s while a still chain stands
  within 3 m of the sink): move about 3.5 m side-on to the estimated path so the whole chain is in view, re-ask
  once, and re-bind the new answer to tracks. New because Cq and Rq re-ask blind, from wherever the robot stands.
- **Check (needs calls):** 4 calls × class-A cells (5 / 3 / 5), free SoCLaaS models first, about 52 calls. Valid in
  ≥ 0.4 of cases would support it; ≤ 0.25 refutes it. **Effort:** 3 h.

## 9. Solve on where people will stop, not where they are (low priority, for variance only)

- After a detected service, solve on predicted rest positions (one spacing up the path) until the tracks settle: a
  feed-forward CTs that does not freeze. Little to gain on the mean: the post-service outage is median 0 samples
  (21 / 46 / 1 lost in total). Try it only if idea 2 does not cut the redraw spread; same repeat cells. 2 h.

---

## Order to try, and what to keep honest

1. Ideas 2 (1 h, measured), 1 (2 h) and 4 (1-2 h), all zero-call: they target 48-90% of CT's cut-ins, the frozen
   near-miss class and the cafe walker. Then idea 3, then idea 5 with its detector check.
2. Everything is post hoc on seeds 0-2: freeze one combined arm before seeds 10-12, as for CTs. For idea 1, report
   cut-ins and "reached the front in turn" beside share (the never-served artefact).
3. Every rule is tied to `come_after` and track kinematics (still, stepping, leaving), never to a counter or queue.
   Before freezing, check a lift door, a ticket machine, a curved line and a conversation group; for the group,
   ideas 1-5 must do nothing (no `come_after`, no service events).
