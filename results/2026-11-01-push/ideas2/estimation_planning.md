# Ideas round 2: estimation, planning and control under uncertainty

Lens: belief-space planning, multi-hypothesis tracking, out-of-sequence fusion, value of information, contingency
planning. Read-only; no model calls; no `keep/heldout/`, no eval_16-19. I take round 1's robotics BOLD A ("a belief
over line hypotheses with the model's answer as the prior") further, and ran zero-call checks on the 88 logged
bystander episodes that carry per-person tracks. **Some of those checks argue against the plain form of BOLD A**,
and that result shaped the ideas below.

## 0. What the logs say (zero calls; these numbers drive the ideas)

| # | Measurement (script) | DeepSeek | Sonnet |
|---|---|---|---|
| Z1 | The model as a role sensor: P(named in `come_after` given the person's role at the photo) (`vlm_sensor.py`) | member 93/120 = 0.78; bystander 16/180 = 0.09 | member 58/65 = 0.89; bystander 15/84 = 0.18 |
| Z2 | Likelihood ratio of "named" (member against bystander) | 8.7 | 4.9 |
| Z3 | Motion as a role sensor: a true member steps 0.4-1.2 m after standing still; does a still person within 3 m behind them move 0.25 m the same way within 8 s? (`info_rate.py`, both models pooled) | bystanders 6/480 = 0.01; members 28/43 = 0.65 (25/26 in the room line; my crude step detector probably also fires on served people walking off) | (pooled) |
| Z4 | Named bystanders in CT3's own episodes (`ct3_missed_tests.py`, 24 cells) | 31 bystanders in all: 30 tested at least once; 78 tests; **followed 0 of 78**; median first test **17 s** after the answer (range 5-58 s) | (pooled) |
| Z5 | Which of those 78 tests CT3 could use (the stepper is named and not yet dropped) | 41 of 78; **14 of 31 bystanders had no usable test**; CT3 never dropped 9 of 31 | (pooled) |
| Z6 | Bystander-seconds believed waiting (`compare_ct3.py`, `evidence_routing.py`): prior alone / CT3 as logged / any true member's step may test anyone behind it (oracle routing) | 993 / 618 / 577 (-7%) | 1083 / 642 / 438 (-32%) |
| Z7 | Same, if anyone's step may test (no oracle) | 452 s, but 40 false drops of named true members (plus 6 on plain scenes) | 339 s, 29 false drops (plus 25 on plain scenes) |
| Z8 | A soft per-person filter (log-odds; named prior 0.8, others 0.3; q = P(follow given waiting) 0.85; b = P(follow by chance) 0.2; all set a priori) (`role_filter.py`, `compare_ct3.py`) | **914 s (worse than CT3's 618)**; still in-queue members believed waiting only 0.27 of the time | 566 s (better than 642); member share 0.41 |
| Z9 | Post hoc filter variants: (q 0.65, b 0.02) / plus "a step is itself evidence of waiting" (x3) / (q 0.85, b 0.2, step x3) | 965 / 758 / 621 | 566 / 472 / 442 |
| Z10 | Third-party evidence: a walker stops within 2 m of a named person, nearer the target than them (`social_proof.py`) | named bystanders 19/31; named members 34/151; LR about 2.7; median first evidence 17.1 -> 16.3 s | (pooled) |
| Z11 | Named bystanders whose first test after the answer came sooner than the cell's own latency (a stationarity proxy for "a test happened while we waited") | **10 of 16** (median latency 25.5 s); those bystanders hold ~559 of CT3's 621 bystander-seconds | **8 of 15** (16.2 s); ~252 of 643 |
| Z12 | CT, evaluation scenes, 15 episodes per model: median latency / plan stage / start-to-first-goal / answer-to-arrival | 20.8 s / 4.8 s / 3.2 m / 8.2 s | 15.1 s / 3.9 s / 3.3 m / 9.3 s (x3: 43.2 s / 10.2 s / 3.2 m / 11.5 s) |

Note: CT3's figure is 618/642 s in `compare_ct3.py` and 621/643 s in `evidence_routing.py`; the two scripts start the
window a few tenths of a second apart. The first run of `role_filter.py` had a bug: it reset a person's belief after
cumulative drift, so step-ups reset members. I found it from a member recall of 0.15 and fixed it before any comparison with CT3 quoted here.

**Reading.** In this sim one service event is a near-perfect test of who waits: a no-follow LR of about 0.04 in a
clean line, against 4.9-8.7 for the model's naming. Bystanders never followed (0/78). So CT3's hard rule is close to
optimal for the evidence it uses, and a soft filter with a priori likelihoods does worse on DeepSeek and only a little better on Sonnet
(Z8). Better evidence routing
can remove only 7% (DeepSeek) or 32% (Sonnet) of CT3's leftover bystander time (Z6). **Most of what remains is
information latency.** The robot waits a median 17 s after the answer for the first informative step. For about
half the named bystanders, that step had probably already happened while the model was still thinking (Z11), and
CT3 had thrown that evidence away.

## 1. Intuitions from estimation and planning

1. **A delayed measurement belongs at its own timestamp.** In multi-sensor tracking, a measurement that arrives late
   is fused at the time it was taken, and the state is then rolled forward (retrodiction). This is the
   "negative-time update" (Bar-Shalom, "Update with out-of-sequence measurements in tracking: exact solution", IEEE
   TAES 38(3), 2002; UNVERIFIED, abstract seen only in search snippets, publisher page 403). Our answer is a
   measurement of the scene at the photo, delivered 15-43 s later.
2. **Group membership and motion must be estimated jointly** ("a chicken-and-egg problem"): Pellegrini, Ess and Van
   Gool, ECCV 2010, third-order graphical model (VERIFIED, abstract); Luber and Arras, RSS 2013, recursive
   multi-hypothesis social grouping (VERIFIED, abstract); Linder and Arras, FUSION 2014 (UNVERIFIED, metadata only).
   Z8 failed for exactly this reason: whether a stepper's step counts as evidence depends on their own role.
3. **Latent intention as the hidden part of the state, plus actions that elicit it.** Intention-aware motion
   planning treats a person's intention as the main partially observed variable of a MOMDP (Bandyopadhyay, Won,
   Frazzoli, Hsu, Lee, Rus, WAFR 2012 / STAR 86, 2013; UNVERIFIED, metadata and snippets). Active information
   gathering plans robot actions that probe a person's internal state, and beats passive estimation (Sadigh,
   Sastry, Seshia, Dragan, IROS 2016; VERIFIED, abstract).
5. **Ask when it pays.** KnowNo uses conformal prediction to decide when an LLM planner should ask for help (Ren et
   al., CoRL 2023; VERIFIED, abstract). UPS chooses between acting, asking and requesting an intervention (Yuan,
   Wu, Bajcsy, RSS 2026, arXiv 2602.22474; VERIFIED, abstract). Neither counts latency as the cost, and neither has
   a free competing source of evidence. I found no paper that decides when to query a VLM by value of information
   per second (search: "when to query a vision-language model ... value of information"; nothing close).

## 2. Ideas

### Idea 1. The answer is a measurement of the past: fuse it at the photo time and replay the wait

**In one sentence.** The robot buffers tracks from the moment of the photo. When the answer arrives, it binds the
answer's people at the photo time (as now), then runs the re-solver's rules forward over the buffered tracks
before it solves, so evidence seen during the wait (who stepped up, who did not follow, who left) applies at once.
- **Failure it attacks.** All of CT3's leftover capture comes from bystanders the model named (SUMMARY §5: 203 s
  DeepSeek, 322 s Sonnet). Z11: 10/16 and 8/15 of them were probably tested during their own latency. It also
  attacks stale answers (0.10 of failure time, FAILURES §1) and services missed during the wait ("the named front
  person is already gone", FAILURES §5A).
- **Borrowed from.** Out-of-sequence measurement fusion (intuition 1). It builds on round 1's "watch during the
  latency" (IDEAS_robotics 5) and "motion-revealed membership during the latency" (IDEAS_interface 4). Those
  learn a line frame or classify people before the answer. This replays the answer's own roster through the
  evidence. Today `Resolver._history` starts at delivery, so CT3 discards the window.
- **Why it is a way of thinking, not a patch.** Any system that pairs a slow model with a fast tracker has
  multi-second answers about a frame that is already gone. The rule to reuse: **timestamp the answer to its input
  frame, fuse it there, and roll forward through buffered evidence.** Verification can then finish before the
  answer is ever acted on. It applies to any identity-referencing answer, not only to queues.
- **Closest prior work.** TIC-VLA conditions a trained policy on delayed semantics plus latency metadata (VERIFIED in
  LANDSCAPE). Slow Brain, Fast Planner decays a stale choice by geometric similarity (VERIFIED in LANDSCAPE). OOSM
  filters (intuition 1) do the same for numeric measurements. What is new is retrodicting a symbolic,
  identity-referencing social answer and replaying its validity rules over the wait.
- **Risks.** (a) Scripted humans: co-motion is near-deterministic here (Z3), so replay looks better than it will on
  real people. (b) Detector tracks are 71% stale at any moment (SUMMARY §7). A person held at their last position
  "did not follow", which is a false drop unless missing data counts as no evidence. That is a design rule here,
  not an option. (c) It does nothing when no service falls in the window (fast models, slow services).
- **Scores.** Payoff 4, uncertainty 3, cost 1 day (a track buffer from `asked_t`, and a replay loop over
  `Resolver.inputs`, plus the rerun).

### Idea 2 [BOLD]. Social constraints as falsifiable motion predictions: a joint role belief, with the VLM as a calibrated sensor

**In one sentence.** Read each constraint the model writes as a hypothesis about a latent social state, with a
predicted motion signature. Keep a joint, multi-hypothesis belief over role assignments: the answer is a sensor
reading with a measured error model, the tracks are a second sensor, and the goal is chosen under the belief.
- **Failures it attacks.**
  - The model naming bystanders: 10/51 DeepSeek and 13/29 Sonnet answers (SUMMARY §5).
  - Omitted members, i.e. a `come_after` naming only the front (FAILURES §2) and cafe's occluded member (0.10 of
    failure time).
  - Two CT3 failure modes, which become special cases of one estimator: a wrong target point breaks the follow
    check (sweet s30), and a check can fire only from named, undropped steppers (Z5).
- **Borrowed from.** Joint tracking-and-grouping MHT (intuition 2), intention MOMDPs (intuition 3), and
  interacting-multiple-model filters for per-person modes (waiting, stepping, walking, shifting on the spot). For
  the IMM: Blom and Bar-Shalom, IEEE TAC 33(8), 1988 (UNVERIFIED, metadata only).
- **What round 1's BOLD A lacked, and the logs now show.**
  1. **Independent per-person beliefs fail.** A step counts as evidence only if the stepper waits, so the beliefs
     dilute each other. Z8 is worse than CT3 on DeepSeek, and leaves in-queue members believed waiting only
     27-41% of the time. The belief must be over joint hypotheses: ordered sets and their bindings.
  2. **The likelihood must be measured, not guessed.** b was 0.01, not 0.2 (Z3). The VLM's error model is
     per-model and **not in order of model strength**: DeepSeek's naming is the better role sensor (LR 8.7 against
     4.9, Z2).
  3. **Evidence sources rank as:** a co-motion test (LR about 25 against waiting on a no-follow), then the model's
     naming (5-9), then third-party placement (about 2.7, Z10). Social proof is weak here.
- **General across situations.** The signature comes from the constraint kind, never from the scene.
  - `come_after`: those named advance with the person ahead when the front is served, and stand still otherwise.
    This covers chains and also a **lift-door cluster**, which is not a chain (they all move at the service event,
    when the door opens). Geometry chains fail on such clusters (IDEAS_interface 7).
  - `keep_out_of_shared_space`: when one member moves or leaves, the others re-close the shared space. A passer-by
    does not take part.
  - `keep_out_of_view(target)` (an audience): still, facing the target, and turning together when it moves.
  - Number-to-track binding becomes part of each hypothesis. Two tracks in one gate means two hypotheses, not one
    Hungarian guess.
  - The goal maximises the probability of being valid. The robot commits, i.e. counts as joined, only when no
    hypothesis above δ makes the chosen place a cut-in.
- **Why it is a new way of thinking.** Today the VLM's answer is a command that rules later patch (CT2, CT3). Here
  it is **a hypothesis that predicts how people will move, so the world can falsify it**. A reader can reuse two
  things: "each constraint kind is a generative model of motion", and "report the VLM as a sensor (a confusion
  matrix over roles, plus a latency distribution), not as an accuracy".
- **Closest prior work.**
  - MIGU fuses VLM priors into a static grounding belief (VERIFIED in LANDSCAPE).
  - Capability-aware shared control Bayesian-filters a VLM's intent confidence (Du, Bowman, Zhang, arXiv 2609.25369;
    VERIFIED, abstract).
  - GUIDER uses VLM scores as a semantic prior on intent (arXiv 2508.11093; VERIFIED, abstract).
  - None of these gives a VLM's social constraints a motion likelihood, or updates role beliefs from co-motion.
  - Group MHT (Luber and Arras) infers groups from motion with no language prior and no goal.
- **Hypothesis, cheapest tests, kill.**
  1. Zero calls, partly done: Z1-Z10.
  3. Generality: three scripted motion scenes where chain geometry fails, each with a named passer-by:
     - a lift-door cluster that boards on a door event;
     - a 3-person group that re-forms when one leaves;
     - an audience that turns as the presenter moves.

- **Risks.**
  - Scripted motion makes every likelihood too sharp. Real queues creep and lag. I found no public queue-trajectory
    dataset to calibrate against, only patents and single-file walking experiments (UNVERIFIED snippets).
  - Per-kind signatures could drift into scene templates (a lift model that is really a queue model).
  - Joint hypotheses grow combinatorially in crowds.
  - The decision layer (δ, commit) has knobs a reviewer will call tuning.
- **Scores.** Payoff 5, uncertainty 4, cost 6-8 days.

### Idea 3. Information per second: the VLM competes with the world as a sensor

**In one sentence.** Before any re-ask (or any one-bit question), compare the expected information per second from
asking (the model's measured likelihood ratio and latency) with that from waiting (the expected time to the next
informative event and that event's likelihood ratio). Ask only for bits the world cannot reveal soon.
- **Failures it attacks.** Re-asking is a poor source here. R re-asked every 10 s scores below R asked once (0.14-0.16
  against 0.20, SUMMARY §1 controls). Cq's re-asks lose good first answers (323 s, FAILURES §2), and re-solving
  with no re-ask (CT 0.16) beats Cq (0.11). x3's re-asks take 43 s against 15-25 s services
  (stale_reask 0.27 of x3 Cq time). On the other side are errors no waiting can fix: in room, DeepSeek answered
  "people do not matter" 5/5 and drove through the line; 5/60 answers had no target; one target point was wrong
  (sweet s30).
- **Borrowed from.** Value of information and sensor scheduling, metareasoning about when to compute (Russell and
  Wefald, "Do the right thing", 1991; UNVERIFIED, memory), and asking for help (intuition 5).
- **Why it is new.** Calibrated asking (KnowNo, UPS) asks a human when the model is unsure. Here the cost is
  latency, the alternative is the scene's own free evidence, and the measured values decide.
  - For "who waits", the world wins on both counts: median 17 s to the first test (Z4) against a 15-43 s answer,
    and LR about 25 against 5-9. That explains, from first principles, why re-solving on tracks beat periodic re-asking.
  - It predicts where asking does pay: target identity, "do people matter", static groups with no events. Those
    are bits that motion never reveals. A reader can reuse the rule in any slow-model loop.
- **Builds on.** IDEAS_robotics 8 (re-ask from a vantage on surprise), IDEAS_interface 5 (one-bit questions),
  LANDSCAPE Q3 (triggers). This idea supplies the criterion that chooses among them.
- **Test.**
  1. Zero calls: on every Cq/Rq episode, take the service clock from the samples (`ahead` decrements) and apply the
     rule at each logged re-ask. Count the re-asks it suppresses, and how many of the 323 s that re-asks lost it
     would have saved.
  2. Then the structural-error cells (room DeepSeek 5, no_target 5, sweet s30): a track surprise, e.g. the planned
     path crosses still people who co-move. Re-ask once from a side view: about 11 cells x 4 calls = 44 free calls.
- **Risks.** Likelihood ratios measured in a scripted sim; the expected time to the next event depends on service
  time; the decision needs a horizon, so the episode length becomes a parameter.
- **Scores.** Payoff 4, uncertainty 3, cost 1.5 days.

### Idea 4. Contingent prefix: while the model thinks, move only as far as every plausible answer agrees

**In one sentence.** From the plan stage on (4-10 s, Z12), build a few track-only hypotheses (each still chain near
the target; "people do not matter"). Drive along the motion prefix they share, to a branch point that is harmless
under every hypothesis above δ (no cut-in, no shared space, outside view cones) and that minimises the expected
remaining travel. The answer, or an event, picks the branch.
- **Failure it attacks.** Waiting for the first answer (0.28 of failure time) plus walking there (0.11): answer to
  arrival is 8-12 s for 3.2 m (Z12).
- **Borrowed from.** Contingency and branch MPC (intuition 4). Intention-aware planning (intuition 3) supplies the
  commit rule.
- **Why it is new.** CT2m failed because it committed to **one** hypothesis, the target point, which is the counter
  and not the line's back (SUMMARY §4). Robotics BOLD B and SUMMARY's "speculative execution" proposed acting on a
  prior. The contingency view adds the constraint that makes this safe: the prefix stops where hypotheses diverge,
  and the stop is safe under all of them. For example, with a bystander it waits behind the bystander, never in
  front of them, then moves about 1 m. The idea to reuse: **latency is a branch point in a contingency plan, not
  dead time.**
- **Hypothesis and test.** Zero calls: a cached-answer sim arm CTc on the 15 evaluation cells x 3 models, plus the
  bystander cells.
  - Predicted: answer-to-arrival falls from 8-12 s to at most 4 s.
  - Strict share rises by at least +0.10 for x3 and +0.03 for DeepSeek and Sonnet. On plain scenes all chain
    hypotheses agree, so the robot is in line before the answer arrives.
  - No cut-in sample before the answer.
- **Risks.** Before the plan stage nothing is known; the plan target can be wrong (sweet s30); the approach path
  itself may cross the line (CT2m's wrong-side case); scripted people never react to an early robot.
- **Scores.** Payoff 3, uncertainty 3, cost 2 days.

### Idea 5. Report the identifiability floor: how soon the scene itself reveals the social state

**In one sentence.** For any moving-people benchmark, compute the time until an oracle-routed passive observer could
know each role (the first informative event), and report methods against that floor.
- **Failure it attacks.** Mis-reading results: CT3 vs CT2 is +0.018 near a floor (SUMMARY §5), and better
  estimators can recover at most 7-32% (Z6).
- **Why it is new.** It separates "the method is weak" from "the scene has not yet said". With passive evidence,
  CT3 is already within 7% of the floor on DeepSeek. Any further gain must come from a better prior, from replaying
  the latency window (idea 1), or from active sensing. Social-navigation papers report success per episode
  (LANDSCAPE §1); I found none that reports time to identifiability.
- **Scores.** Payoff 3, uncertainty 2, cost 0.5 day.

## 3. Limitations a CoRL or HRI reviewer would raise

1. **Scripted humans make motion evidence too clean.** Bystanders followed 6/480 member steps and members 25/26 in a
   clean line. CT3's gain and every idea here are upper bounds until the likelihoods have realistic lag, creep and
   shifting. Nobody reacts to the robot, so active probing (intuition 3) cannot be tested.
2. **Oracle tracks.** Every bystander result uses true tracks; CTd's tracks are 71% stale. Under co-motion rules,
   "unobserved" reads as "did not follow". Neither CT3 nor the ideas are validated with a detector.
3. **One dynamic situation.** All motion results are FIFO lines with a 15 s service. The claim that re-solving
   generalises to groups, audiences and lifts is argued from the vocabulary, not tested with motion. The generality
   scenes in idea 2 are the missing evidence.
4. **Designed errors.** The bystander layout (a lone person just past the end, off to its side) was built to fool
   the models. The rates 0.09 and 0.18 are properties of that layout, not of the models in the wild.
5. **Latency is the largest failure (0.28) and nothing claimed addresses it.** A reviewer will say the contribution
   is maintenance after an answer, not goal inference. Ideas 1 and 4 are the answer we have.
6. **No decision theory behind the thresholds.** Values such as 5 s co-motion, 0.25 m follow and 0.3 m replace carry
   no error-rate guarantee. The metric weighs a capture (too cautious) and absence alike, and counts cut-ins apart.
   A socially weighted cost (cut-in much worse than waiting behind a bystander) would change which policy wins.
7. **Small, post hoc evidence.** 24 named-bystander cells; all of section 0 is post hoc on the data it describes.

## 4. Top pick (five lines)

Idea 1, the answer as a measurement of the past: buffer tracks from the photo, bind there, and replay the
re-solver's rules through the wait before the first solve. All of CT3's leftover capture comes from named
bystanders. A proxy says 10/16 and 8/15 of them were tested while the model was still thinking. Better evidence
routing alone recovers only 7-32% (Z6). The test is zero calls (a cached rerun of 88 cells), with a clear kill bar
(< 15% fewer bystander-seconds). The lesson applies to any slow model over a fast tracker, and it is the first
working piece of the BOLD belief view (idea 2).
