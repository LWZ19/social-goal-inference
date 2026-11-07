# Ideas: social behaviour and evaluation for the moving line (2026-11-02)

## Headline findings (they motivate the items)

1. **R1's DeepSeek lead is a scoring artefact.** 180 of R1's 270 in-line samples are taken while it stands first,
   never served, at a median 1.02 m from slot 0; none is within 0.6 m. With a 0.6 m first-in-line tolerance:
   dsv41 R1 0.20 -> 0.08 and CT 0.18 -> 0.12; Sonnet R1 0.15 -> 0.06 and CT 0.20 -> 0.18; x3 R1 0.14 -> 0.12 and
   CT 0.12 -> 0.09.
2. **CT's largest single failure is a near miss.** After the first settle, the commonest single failing test is
   "gap 1.2-1.5 m, all else fine": 81 / 82 / 146 samples (dsv41 / x3 / Sonnet), against 242 / 154 / 263 in line.
   It is built in: standoff 1.0 m + replace threshold 0.3 m > band edge 1.2 m, and a stale goal after a step-up
   always errs towards "too far".
3. **The robot believes it has joined when the line does not count it.** In CT, the resolver's `joined` is true
   while the sim does not count the robot as a member in 190 / 110 / 285 samples (of 1,365 per model); 189 / 100 /
   272 of them are not in line. Route (a): the robot becomes first, the waiting set empties, the solver drops the
   place in turn and walks to the model's target point about 0.9 m beside slot 0; the sim removes it from the queue
   and a rejoiner takes the front (dsv41 cafe_line2_moving s1, t=33-60). Route (b): it "joins" where the line never
   accepted it (dsv41 cafe s2; Sonnet cafe_line2_moving s0 and s1, where `joins` is empty).
4. **Cut-ins start when someone rejoins.** In the C arms, 47 of 57 cut-in onsets (82%, pooled over the 3 models)
   come within 10 s of a rejoin. The chance rate for a random settled sample is 0.49-0.53.

---

## A. Behaviour: mechanisms to borrow (none names a queue; each is a rule over `come_after` and its tracks)

### A1. Take the served person's place, and count as joined only where the line puts you
- **Target.** Finding 3. CT reached the front in turn in only 2 of 15 (dsv41). It became first and then left the
  line within 15 s in 2 of 6 episodes (dsv41) and 3 of 5 (Sonnet), losing 108 and 73 samples.
- **People.** Step into the footprints of the person just served. You know you are in line because the line and
  the person behind you treat you so, not because you stopped somewhere.
- **Mechanism.** (a) Keep the last stationary pose (where they stood, which way they faced) of the front-most
  `come_after` member. When the waiting set empties after joining, that pose is the place in turn until the robot's
  turn ends; the target point is used only for the act step. (b) The resolver's `joined` also requires the robot
  within 0.5 m of the chain's extension (the last two waiting people, or the remembered axis), behind the last.
  General: lift door, ticket desk, pharmacy counter.
- **Why new.** The brief names the failure but has no fix. Neither the brief nor LANDSCAPE separates
  self-declared joining from joining the line accepts.
- **Prediction / refutation.** Predicted: resolver-joined/sim-not samples fall below 20 per model, and "reached
  the front in turn" goes from 2/15 to at least 5/15 (dsv41). Refuted if the robot still leaves the corridor (the
  front pose was taken mid-step) or blocks the served person's exit.
- **Effort.** About 2 h in `resolve.py`.

### A2. Keep the line's own spacing, with room for staleness
- **Target.** Finding 2 and Q13's frozen near miss. dsv41 cafe_line6_moving s1 sits at 1.21-1.22 m for 13 s
  (t=39-51), then at 1.35 m for 16 s (t=71-86). Settled gaps are typically 1.01-1.12 m, 0.1 m from failing.
- **People.** Copy the spacing of the line they join; dense lines close up.
- **Mechanism.** The re-solve's standoff is the median spacing between consecutive tracked `come_after` members,
  clipped so that standoff + replace threshold <= band edge - 0.1 (about 0.8 m here). And/or replace the goal
  whenever its gap to the person ahead leaves the band (CTs's validity test covers clearance only). Put it in the
  resolver only: `standoff_m` is printed into the prompt, so a config change would change the cached answers.
- **Why new.** Q13 and CTs blame re-solve timing. Nobody noted that the three constants clash by construction,
  or took the spacing from the people.
- **Evidence (0 calls, done).** Counterfactual, robot 0.2 m closer on joined samples: CT 0.18 -> 0.24 (dsv41),
  0.12 -> 0.17 (x3), 0.20 -> 0.29 (Sonnet); C1 does not move (its failures are > 1.5 m). 1-D replay on 25 logged
  leader traces (1,244 s of CT/CTv joined runs; Spot 0.56 m/s), share in band: CT-like 0.71 (actual CT in the same
  runs 0.58, so the model is optimistic), standoff 0.8 m 0.98, validity replace 0.97, settled (CTs-like) 0.86.
- **Prediction / refutation.** Predicted: near-miss samples fall by at least 70%, keep-up is unchanged, and the
  redraw spread falls. Refuted if too-close samples (0 today) or cut-ins rise. Next test: sim, as in A1 (0 calls).
- **Effort.** 30 min plus runs.

### A3. Yield to whoever is heading for the end (intent from motion and facing, not distance)
- **Target.** Finding 4. Sonnet cafe_line2_moving s1, t=32: the robot "joins" 1.0 m behind the front person while
  a served person walks back. The rejoiner reaches the queue first, so the robot stands 1.87 m ahead of them.
- **People.** Read intent from gait and gaze: someone walking to the end and facing it is already in line
  ("after you").
- **Mechanism.** Before committing to (or declaring joined at) a place in turn, add to the waiting set any track
  that is moving towards that place (velocity within 30 deg of the bearing), faces roughly as the waiting people
  do, and would get there before the robot. Then re-solve behind them. This is the existing "joins behind the
  last" rule, made predictive. General: anyone hurrying to a lift door or a desk.
- **Why new.** LANDSCAPE Q4 foresees races for the end but offers commit policies only. "Who is last" is read
  from position everywhere in the pipeline.
- **Test.** Done (0 calls): the association. Next (0 calls): sim with cached answers on the cut-in cells (CT
  6 / 5 / 5 episodes), logging `tracks` per sample to measure intent directly.
- **Prediction / refutation.** Predicted: CT cut-in episodes halve; onsets after a rejoin fall to the chance rate.
  Refuted if the cut-ins come from the first answer (come_after missing people); check `answer.json` first.
- **Effort.** About 3 h.

### A4. Hold a place for someone who steps out; advance on flow, not on gaps
- **Target.** "Left for good." In dsv41 cafe_line6_stepout s1, the member who stepped 1.5 m aside at t=40 was
  dropped for good at t=42.5 (in CT and in CTv). Returning ahead of the last, they can never be re-absorbed. The
  predicted cut-in (Q7b) needs them to become the last ahead of the robot. The robot stood 2 or more places back
  and episodes end at 92 s, so the scripted case cannot show it: an evaluation gap.
- **People.** Keep a place for someone who steps aside briefly. Step up when the person in front does, never into
  a gap that opened sideways.
- **Mechanism.** Leaving needs a sustained departure: more than 3 m from their spot, or away for more than 10 s.
  Someone who comes back between members keeps their order. The robot advances only when the person directly
  ahead moves along the line. General: any ordered waiting.
- **Why new.** The mentor's group requires 3 frames of persistence. LANDSCAPE lists "member leaves" but has no
  hold rule.
- **Test.** A robot-relative event: the member directly ahead (variant: 2 ahead) steps 1.5 m aside for 7 s, 10 s
  after the robot joins. cafe_line6_moving's first frames are cached, so 0 calls; 3 seeds x {CT, CT+A4}. Check
  that served people still count as left within about 5 s.
- **Prediction / refutation.** Predicted: CT moves into the gap, or later cuts in ahead of the returner, in at
  least 2 of 3 seeds; CT+A4 does neither. Refuted if CT already holds its place.
- **Effort.** 2 h resolver + 1 h sim event.

### A5. [BOLD] Join the chain, then follow the person ahead
- **People.** Once in line, people stop working out where the end is. They follow the back of the person in front,
  starting about 0.5-1 s after them, and face them (queue dynamics, a cousin of car-following models).
- **Mechanism.** After joining, replace the 0.5 s pose re-solve and the 0.3 m replace rule with a following law
  on the person directly ahead in `come_after` order (gap about 0.85 m, 0.15 m dead band). Place the robot on the
  local tangent (that person's predecessor -> them), not the remembered axis, facing the person ahead. Hand over to
  the next person ahead when the leader leaves (A4); re-solve only on events (leader lost, nobody ahead -> A1,
  someone between).
- **Why new.** RT/RTw anchor to the person nearest R's point, with no order and no hand-over, so they follow the
  served anchor away. LANDSCAPE Q4 compares commit policies, not a following law.
- **Evidence (0 calls, done).** 1-D replay: in band 1.00 of the time against 0.71 for the CT-like policy.
  Idealised: a 0.13 calibration gap, and no turning. Next: a sim follower mode on oracle tracks, cached answers.
- **Prediction / refutation.** Predicted: out-of-line seconds per service (in line before) at most 3, redraw mean
  bring back RT's "follow the served person away".
- **Score.** Payoff 4, uncertainty 3, cost about 1 day. It changes the story from "the solver gives a pose" to
  "the solver gives a leader and a law".

## B. Evaluation: are we measuring the right thing?

### B1. A failure-cause audit: two scorer artefacts, and a line that differs by arm
- **Definition (episode.json).** For each sample after the first `state == "arrived"` that is not in line, list
  the failing tests and count single causes (combinations apart). With someone ahead: `behind_m < 0`,
  `gap_m < 0.5`, `gap_m > 1.2` (split at 1.5), `heading_off_deg > 45`, `nearest_m < 0.55` (split by `state`).
  First in line: `to_slot0_m > 1.6`, heading, clearance.
- **Results (dsv41).** CT: in line 242, near miss 81, ahead of L alone 60, heading 33, clearance while standing
  18; top combination arrived + ahead of L + far (142), finding 3's robot beside the counter. R1: ahead of L +
  clearance 148, far > 1.5 m 292, heading 0. C1: far > 1.5 m 419 (a stale goal).
- **Artefact 1: first-in-line tolerance (headline 1).** While first, R1 stands at a median 1.02 / 2.00 / 1.39 m
  from slot 0, and 0 of those samples are within 0.6 m.
- **Artefact 2: slot clamp.** room_coffee_cart (4 slots) and cafe (5) were not extended like the `_moving` copies.
  `_slot_of` clamps a rejoiner behind a robot at the last slot onto the robot's spot: `nearest_m` 0.28 m for 7+ s
  (dsv41 room_coffee_cart s0 CT, t=46-55). Nearly every C-arm "clearance-only, standing still" failure is there:
  CT 18 of 18 + CTv 16 (dsv41), CT 32 of 33, C1 11, Cq 11 (x3), CT 13 of 15 (Sonnet); the exception is RTw (14,
  cafe_line2_moving). It penalises standing correctly at the back.
- **The line differs by arm.** Seconds per episode with the robot first and the line frozen: R1 12.0, C1 13.7,
  Cq 11.3, CT 7.2, Rq0 1.5 (dsv41). After the first answer, "paired" cells no longer face the same line.
- **Proposal.** Publish the cause table per arm. Score v2 (post hoc, labelled): first-in-line tolerance 0.6 m, and
  exclude clearance failures where a person walks into a stationary robot (or extend the slots). Keep v1 frozen.
  Done.

### B2. Score service events, not seconds
- **Definition.** An event is a sample i with `joined` where `ahead` falls. Conditional on being in line at i-1:
  restored = in line at some sample in (i, i+10]; cost = not-in-line samples in [i, i+15); lag = first t with
  `along` <= along_i - 0.35; and any cut-in in the window.
- **Why new.** Q9's keep-up is motion-only. This one is outcome-based, a cost in seconds, and has a reference: an
  ideal follower costs about 0-2 s, since people take 1.5-2.5 s to step 0.7 m. LANDSCAPE named "lag in moving up";
  nothing was computed.
- **Results (in line before: restored, cost, median lag).**

  | Arm | dsv41 | x3 | Sonnet |
  |---|---|---|---|
  | CT | 7/9, 7.9 s, 3.0 s | 7/8, 5.9 s, 4.0 s | 11/14, 7.4 s, 3.0 s |
  | Cq | 3/5, 8.6 s, 14 s | 5/7, 13.3 s, 30 s | 10/15, 7.9 s, 8.5 s |
  | Rq | 4/6, 7.8 s, 9 s | 4/7, 11.4 s, 15 s | - |
  | C1 | 1/2, 13.0 s, never | 5/7, 13.3 s, never | 5/7, 12.7 s, never |

  R1 has only 2 / 3 / 0 such events; always report counts with rates.
- **Reading.** CT's edge is clear in lag. In cost it is about half what it could be (A2/A5). Done.

### B3. Two tiers ("in order" against "in place"), and tolerance curves
- **Results (mean share).** dsv41 strict: R1 0.20 > CT 0.18 > CTv 0.17 > Rq 0.16 > Cq 0.15. dsv41 in order:
  Cq 0.31 = CT 0.31 > Rq 0.28 = CTv 0.28 > RTw 0.26 ... R1 0.16 = C1 0.16. Sonnet in order: CT 0.39 > Cq 0.34 >
  R1 0.28 > C1 0.22 > Rq 0.14.
- **Noise (CT redraw pairs, n = 6).**

  | Measure | Mean abs diff | Max | Share of the CT-R1 paired variance |
  |---|---|---|---|
  | Strict | 0.10 | 0.49 | 0.18 |
  | Band <= 1.4 | 0.08 | 0.30 | 0.06 |
  | In order | 0.06 | 0.24 | 0.05 |

  The band edge drives much of the bimodal redraw.
- **Caveat.** The area under the tolerance curve rewards a lagging goal (C1 0.03 -> 0.09): a sensitivity only.
  Done.

### B4. Power, pseudo-replication, and the design a reviewer will ask for
- **Power.** CT-R1 paired sd 0.34 over 15 cells: the minimum detectable difference is about 0.25, and 80% power at
  0.10 needs about 91 cells. CT-C1 (sd 0.235, mean +0.16) needs about 18 and CT-RTw about 23: the only share
  contrasts within reach. Scenario-clustered bootstrap 95% CIs:

  | Model | CT-R1 | CT-C1 | CT-Rq |
  |---|---|---|---|
  | dsv41 | -0.02 [-0.24, +0.19] | +0.16 [+0.02, +0.32] | - |
  | Sonnet | - | +0.16 [+0.02, +0.30] | +0.15 [-0.01, +0.31] |

  Shares are zero-inflated (5-9 of 15 episodes are 0 for most arms): report per-cell ordinal outcomes (never /
  lost / kept / served in turn).
- **Pseudo-replication.** 20-83% of each arm's share variance is between scenarios (CT 0.50). With 3 seeds per
  scenario the design effect is about 1 + 2 x 0.5 = 2, so n_eff is about 7, not 15. x3 and Sonnet C1 first answers
  agree within 0.2 m across seeds in 4 of 5 scenarios; DeepSeek's differ by 0-5.9 m, which is model sampling, since
  seeds change only the walkers. Redraw noise is 18% of the paired variance: more scenarios beat more seeds, and
  both beat more redraws.
- **Effort.** Numbers done; a design note, 1 h.

### B5. [BOLD] A counter that serves and skips: the same line for every arm, "served in turn" as the primary
- **Target.** The never-served artefact, the stall that differs by arm (R1 12 s/episode against CT 7), and pairing
  that breaks once the robot changes the line.
- **Definition.** The queue serves the robot like anyone else when it is first, within 0.6 m of slot 0 and facing
  its face point. If it is not there within 5 s when its turn comes, the counter calls "next": the robot is skipped
  and must rejoin (lost place). Primary: served in turn, time to service, skips, cut-ins. Every arm faces the same
  schedule until its own turn.
- **Why new.** Q9 kept "serve the robot" as a sensitivity only. Skip-if-absent is what makes it a test of keeping
  one's place.
- **Test.** About 2 h in `experiment.py`. R1, C1, CT, CTv, RT and RTw re-run on cached first answers (0 calls).
  Rq, Cq and Rq0 see new frames: about 20-30 calls per episode on SoCLaaS.
- **Prediction / refutation.** Predicted: CT is served in turn more often than R1 on all 3 models (R1 stands
  1-2 m off slot 0 when first, so it is skipped). Refuted if R1 is served as often.
- **Score.** Payoff 4, uncertainty 2, cost half a day.

---

## Suggested order (under a week)
1. Put B1 (v2 scoring), B2 and B3 into the report as labelled post hoc tables (computed).
2. Freeze A1 + A2 (+ A3 if time allows) as one resolver variant before anyone looks at seeds 10-12. Run it there on
   CTs's cached answers (0 calls).
3. Build B5's serve-and-skip counter (0 calls for the asked-once arms).
4. A4 needs a robot-relative step-out event.
5. A5 last: the bold rewrite.
