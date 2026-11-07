# Ideas round 2: shortlist (run's synthesis, 2 Nov night)

Inputs:

No model calls were made. Every number below is post hoc, on cells that frozen tests have already judged. These
numbers motivate designs; they are not results.
## 0. Numbers re-checked by the run

Cut-in samples (`cut_in` flag per 1-s sample; SCORING.md: the robot more than 0.25 m ahead, along the line, of a
member who is ahead of it in queue order).

Fresh seeds of at least 10, the headline's five evaluation scenes. The first recount left out `cafe_slots` and
`room_coffee_cart_slots`; this is corrected per `research_session_challenge.md` section 0 and re-checked:

| Model | Arms | Pairs | Cut-in samples | First arm has more / fewer |
|---|---|---|---|---|
| Sonnet | CT2 vs R1 | 50 | 481 vs 162 | 25 / 4 |
| DeepSeek | CT2 vs R1 | 25 | 226 vs 504 | 9 / 10 |

For Sonnet this is exposure. CT2 is within 1 m of the axis in 0.73 of samples and R1 in 0.26, and each cuts in on
0.15 and 0.14 of those samples.

Bystander scenes:

| Model | Arms | Pairs | Cut-in samples | First arm has more / fewer |
|---|---|---|---|---|
| DeepSeek | CT3 vs CT2 | 75 | 1,036 vs 881 | 21 / 14 |
| Sonnet | CT3 vs CT2 | 30 | 138 vs 89 | 5 / 3 |

## 1. What converged

> **A VLM's social answer is a hypothesis, not a perception. It makes predictions about where the people it names
> stand and how they will move. The scene tests those predictions from the moment of the photo. The robot acts on
> what survives, and asks again only when the evidence turns against the answer.**

The measured facts behind it:

- **The model's role error is systematic, not random (N3; ml idea 1).**
  - Every bystander-naming answer names the lone person past the line's end (DeepSeek 11/11, Sonnet 13/13).
  - Sonnet names them in 0.78 of near-identical sweet_spot frames.
  - So voting makes it worse. This is bias, not variance.
- **The models contradict their own perception (ml idea 3; sociology B).**
  - When a model named the lone bystander as waiting, its own people step had him facing more than 90 degrees away in
    6/11 (DeepSeek) and 7/13 (Sonnet) cases.
  - Position overrides facing.
- **Motion is a much better role sensor than the model (estimation section 0).**
  - Bystanders followed a member's step 6 times in 480; named bystanders 0 times in 78.
  - Members followed 25 times in 26.
  - The model's naming has a likelihood ratio of 8.7 (DeepSeek) and 4.9 (Sonnet).
- **The evidence comes early, and CT3 wastes it.**
  - The line is served during the first answer's latency in 59/75 (DeepSeek) and 18/29 (Sonnet) cells (N1).
  - CT3's leftover capture mostly comes after two or more services the bystander sat out (N2).
  - CT3 uses one chain of events only: an advance by the person directly ahead.
- **Naive evidence counting fails, and how it fails is informative.**
  - "Drop after k non-follows" false-drops members: 283 of 307 of its non-follows happen while the member is being
    served or is away (N4). Roles change over time.
  - A per-person filter with independent beliefs does worse than CT3 on DeepSeek (estimation section 0). An advance
    is evidence only if the mover is in the line, so beliefs must be joint.
- **"The line ran out" is the other face of the same problem (far analogies section 0).**
  - CT3's waiting list is empty while a real line stands in 0.48 (DeepSeek) and 0.21 (Sonnet) of re-solves. It never
    recovers.
  - CT3 turns part of its capture into an empty roster: bystander as the last person 0.32 → 0.06, empty 0.33 → 0.48.
  - This is where the extra cut-ins come from.

## 2. Shortlist

### M1. The main bet: answers as tested hypotheses (working name CT4, a research option, off by default)

One mechanism that replaces CT2's re-admission and CT3's join gate, follow check and exemption. Its parts, and
which file each comes from:

1. **Predictions come from the constraint vocabulary, not from the model writing extra output.**
   - Static: "would I stand there if I were waiting": a leave-one-out solve for each named person
     (cognitive_science 1).
   - Dynamic: each relation kind predicts a coupling of motion. `come_after` is serial; a shared space is mutual;
     attention to a thing is joint facing (research_session 2, ml_reasoning 2, estimation_planning 2).
2. **Evidence starts at the photo.** Buffer the tracks from the request frame, then replay the wait when the answer
   arrives: the answer is fused as a measurement of the past (estimation_planning 1, N1).
3. **A joint belief over role assignments, with transitions.** States are waiting, being served, away and
   returning; never independent per-person beliefs (research_session 1, estimation_planning section 0).
4. **The structure is a view anchored at the target.** The named people are witnesses used to fit the relation, not
   the relation itself, so the line is re-derived when they leave or join. This is aimed at the line running out and
   the cut-ins (far_analogies 1, sociology E).
5. **A stale or unseen track is missing data, not "stayed put"** (research_session S4).
6. **Ask again, as a closed question about the one decisive person, only when the evidence turns against the
   answer** (research_session 3, ml_reasoning 4). This is optional for the first test.

**What a reader takes away:** for any slow model over a fast tracker, give the answer's vocabulary predictive
semantics, so that every answer can be checked for free. The model's latency then becomes an observation window.

### M2. Generality, without which M1 stays a queue result

- **New moving scenes outside queues, where geometry alone fails.**
  - A conversation group that re-forms when one member leaves, with a newcomer standing near it but outside it.
  - A cluster of people waiting at a lift door.
  - Check first, with zero calls, that geometry fails there.
- **An ambiguity sweep** (reviewer 4). Vary one factor at a time: bystander distance, facing, stillness, and a second
  formation nearby. Plot the model's added value against the geometry-only arm.

### F1. A cheap finding in its own right: the generator-validator gap for social roles

The question: is checking "is this ringed person waiting?" easier for VLMs than listing who waits?

- Research_session 3, ml_reasoning 3, sociology B.
- Kill: a false "yes" on the lone bystander at or above the roster's rate, or a "yes" rate above 90% overall.

### E1. Evaluation that a reviewer will demand (zero calls)

- **Scoring by events:** capture, cut-ins, false drops, served in turn. The harm ratio is stated, not hidden in
  seconds in band (cognitive_science 3, far_analogies 3, social_eval B2).
- **Stress replays S1-S4,** applied to CT3 as well: noise, lagging members, restless bystanders, and stale or switched
  tracks.

### Bold, needing the project
- **H1. Ask people, not the ruler (reviewer 1).** About 20-30 raters on about 100 clips calibrate "in line" and
  "cut in". It needs raters and probably an IRB exemption check.
- **D1. Hindsight co-motion labels distilled into a fast still-frame model on the A100 (ml_reasoning 5).** After M1.
- **Latency.** Two candidates: a contingent prefix, moving only as far as every plausible answer agrees
  (estimation_planning 4), and arrivals as demonstrations (cognitive_science 2). Both come after M1.

### Dropped or deferred

- **B2, roles by predicate:** it decides by geometry, which is exactly where geometry fails.
- **Self-consistency and union voting:** the error is a bias, so voting makes Sonnet worse.
- **Model-written expiry conditions:** they cost latency, and M1 derives its conditions instead.
- **Interface 7, "ask whether, not who" without a roster:** the roster carries information (CT3 vs Gl 24/5/22).

## 3. Plan

| Phase | What | Calls | Who |
|---|---|---|---|
| 1a | Event scorer, and a re-score of every frozen comparison with it | 0 | run |
| 1b | Offline design study on the logged tracks: the parts of M1 one at a time and together, under S1-S4; choose CT4's design and freeze its parameters on the development cells | 0 | analysis, plus a checks |
| 1c | Write the non-queue moving scenes and check that geometry fails there | 0 | run |
| 2 | Implement CT4 in `sgi.resolve` (TDD, off by default), then cache-only sim reruns of the 88 bystander and 45 original cells | 0 model calls | run |

**What would kill M1 before any call:** at phase 1b, no design halves CT3's capture with at most 2 of 244 false
drops and cut-ins no higher than CT2's, under S1-S3.
## 4. For the analysis to challenge

1. Is M1 one idea, or five fixes under one name? What is the smallest version that still carries the reframing?
2. Which development cells are design data, and which seeds stay fresh? Proposal: seeds 10-44 are design data;
   45-74 are fresh.
3. Does the "view anchored at the target" (part 4) conflict with the joint role belief (part 3), or is the view just
   the belief's most likely explanation?
4. Is a non-queue scene where geometry fails buildable with scripted humans, or does it need reactive ones?
5. Phase order: should F1 run first, because it is cheap and stands alone?
## 5. After the analysis's challenge (`research_session_challenge.md`), agreed

1. **The primary comparison is pre-registered as CT4-min against CT3-fix.** CT3-fix is CT3 + the group test +
   evidence from the photo + anchored re-derivation: the best bolt-on version of the same evidence.
   - Claim: CT4-min is no worse on clean tracks, and has fewer false drops at equal capture under S1-S4.
   - If CT3-fix matches it under perturbation too, the contribution is the unifying account only. This is stated now,
     before any data.
2. **CT4-min is one estimator, not five fixes.**
   - State: an exact belief over the waiting set near the anchor (n of 8 or fewer), with role hazards.
   - Evidence: the static leave-one-out residual, come_after co-motion measured along the line, and the answer as
     P(named | role).
   - Parameters are a priori or fitted on original seeds 0-2, never on bystander cells.
   - It keeps the ratchet, the line span and path memory.
   - It leaves out asking, relation kinds other than queues, and anchor estimation.
3. **A fresh test means new layouts, not new seeds.** Services run on a fixed clock, so the first service is at
   16 s in every seed, and the photo barely changes between seeds.
   - The test set comes from an ambiguity-sweep generator: bystander distance, side and facing varied, the service
     clock jittered, lines of 2-4.
   - Seeds 45-74 on the current layouts serve only as a replication.
4. **No lift scene:** `lift_wait` is a sealed held-out situation type. The generality scenes are, in order:
   - a non-waiter standing *on* the line's axis in the existing queue scenes (one YAML edit), since the static check
     works only because our bystanders stand off the axis;
   - a conversation group that re-closes on a schedule after one member leaves, with a non-member at the ring's edge.
5. **F1 runs on the new layouts, with POPE-style controls, not first.** On the current layout it would measure one
   bias in one place.
6. **The convergence was primed** (BRIEF listed BOLD A and B1), so "five of seven agreed" is not evidence.
7. **Phase 1b starts with one common scorer** that reconciles estimation Z6, ml's group test, N4 and K3. The cut-in
   bar moves from 1b to phase 2, because an offline replay cannot measure cut-ins.
