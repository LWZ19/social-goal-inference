# Ideas round 2, cognitive science and developmental psychology lens (2 Nov 2026)

**The reframe in one paragraph.** Our pipeline is feed-forward. The VLM perceives roles, the solver places the robot,
and rules repair the roster afterwards (CT2, CT3). Cognitive science treats social-goal inference differently: it is
*joint* inference of other people's roles and one's own place. One generative model covers both questions: where
does a person with role R stand, and how do they move? People run that model forwards to plan their own place and
backwards to explain other people's poses and motion. What others say, here the VLM, counts as testimony to be
weighed, not as perception. Uncertainty is resolved actively, by watching, copying, asking or acting tentatively,
under a loss in which breaking a norm costs far more than being slow. Each idea below is one piece of that frame,
grounded in a failure we measured.

---

## 1. Intuitions from the field

**I1. People infer goals by inverse planning, using their own planner as the model of others.**
- Observers explain an action by asking which goal would make it rational. Baker, Saxe & Tenenbaum, "Action
  understanding as inverse planning", *Cognition* 113, 2009. VERIFIED.
- Others are assumed to maximise reward minus cost (the "naive utility calculus"). Jara-Ettinger, Gweon, Schulz &
  Tenenbaum, *TiCS* 20(8), 2016. VERIFIED.
- 10-month-olds infer how much an agent values a goal from the cost it pays to reach it. Liu, Ullman, Tenenbaum &
  Spelke, *Science* 358, 2017. VERIFIED.
- The "like me" account: one's own acts and others' acts share one code. Meltzoff, *Dev. Sci.* 10(1), 2007.
  VERIFIED.
- A robot that uses its own action system to infer a human's goals: Gray, Breazeal, Berlin, Brooks & Lieberman,
  "Action parsing and goal inference using self as simulator", RO-MAN 2005. VERIFIED (abstract).
- *Bearing on us.* A person who stands still near the counter yet does not walk up to it pays a cost that only
  "waiting my turn" explains. Our solver already encodes where a waiting agent stands. We use it only for the robot,
  never as a model of the people.

**I2. Groupness and roles are perceived from contingent motion, mostly when something perturbs the group.**
- Campbell named "common fate" the first index of whether people form a real group: "Common fate, similarity, and
  other indices of the status of aggregates of persons as social entities", *Behavioral Science* 3, 1958. VERIFIED
  (record and title).
- Moving in sync raises perceived rapport and groupness. Lakens & Stel, *Social Cognition* 29(1), 2011. VERIFIED.
- Chasing is seen directly from the contingency between two shapes. Gao, Newman & Scholl, *Cogn. Psych.* 59, 2009.
  VERIFIED.
- Trajectories alone, from aerial video of real people, are enough for interaction judgements. Shu, Peng, Fan, Lu
  & Zhu, *TopiCS* 10(1), 2018. VERIFIED.
- *Bearing on us.* CT3's co-motion check is Campbell's common fate. Our logs show its limit: evidence arrives only
  when the formation is perturbed, and then takes a while. Most of CT3's leftover capture falls *after* the first
  service but before the drop (A3).

**I3. Unsure people use cheap social information: they copy, infer the norm from one example, ask, or probe.**
- Copying should be selective, for example "copy when uncertain". Laland, "Social learning strategies", *Learning
  & Behavior* 32, 2004. VERIFIED.
- The winning strategy in a 104-entry tournament relied almost entirely on copying. Rendell et al., *Science* 328,
  2010. VERIFIED.
- 3-year-olds infer a norm from a single observed action, and enforce it on others. Schmidt, Butler, Heinz &
  Tomasello, *Psych. Sci.* 27, 2016. VERIFIED.
- Children explore more when the evidence is confounded. Schulz & Bonawitz, *Dev. Psych.* 43, 2007. VERIFIED.
- Some actions are taken to gain information, not progress ("epistemic actions"). Kirsh & Maglio, *Cogn. Sci.* 18,
  1994. VERIFIED.
- People asked a navigation robot about this exact uncertainty. When blind users met a line, their most common
  query was whether they were actually in it: 35% of line-type queries. Kamikubo et al., "Beyond Omakase", CHI
  2025, arXiv 2503.21997. VERIFIED (HTML, sections 5.1.2-5.1.3).
- *Bearing on us.* Our robot has one source of social information, the VLM, and one response to uncertainty: wait
  15-43 s for it.

**I4. Joining a group is graded, legible and governed by norms.**
- Joint action rests on shared representations, prediction of the partner, and "coordination smoothers" such as
  making oneself predictable. Sebanz, Bekkering & Knoblich, *TiCS* 10(2), 2006, VERIFIED; Vesper et al., *Neural
  Networks* 23, 2010, VERIFIED.
- Legible motion conveys intent. Dragan, Lee & Srinivasa, HRI 2013. VERIFIED.
- *Bearing on us.* Our robot freezes, then commits completely, and its scorer weighs a cut-in second the same as a
  second spent politely behind a non-waiter.

**I5. Perception is prediction, and a spike of surprise marks an event boundary where the model is updated.**
- Zacks, Speer, Swallow, Braver & Reynolds, "Event perception: a mind-brain perspective", *Psych. Bull.* 133, 2007.
  VERIFIED (abstract: "When transient errors in predictions arise, an event boundary is perceived").
- *Bearing on us.* An answer should be a forecast of how people will move; a wrong roster and a stale answer both
  show up as prediction error. A new answer should be *integrated* with the old one, not swapped in: Cq's re-asks
  threw away good first answers, 323 s of fa_noturn (FAILURES §2).

---

## 2. Ideas

### Idea 1. "Would I stand there if I were waiting?" The robot's own solver as the likelihood of everyone's role

**In one sentence.** Run the constraint solver leave-one-out for each person the answer gives a role: solve for an
agent who holds that role (for example come_after the named people ahead of them), and score how far the person's
actual pose, and later their response to events, is from what the solver predicts. The VLM's roster becomes the
prior, and this residual becomes the likelihood.

**Failure attacked.**
- All of CT3's leftover capture comes from bystanders the model named as waiting: DeepSeek 10/51 answers, 203 s;
  Sonnet 13/29 answers, 322 s (SUMMARY §5; `results/moving-lines/bystanders/post-hoc/README.md`).
- The follow check acts only after the line advances.

**Borrowed intuition.** I1: inverse planning with one's own planner, "like me", self as simulator. I5: the dynamic
half is prediction error under the same model.

**Zero-call evidence (post hoc; Appendix A1-A3).**
- A front-to-back parse at the first re-solve, on true tracks. A named, still person is kept only if the solver's
  place-in-turn geometry (`solver._behind`: standoff along the chain kept so far, or from the target point for the
  first) would put a waiting agent where they stand: lateral offset ≤ 0.3 m, and 0.3-1.6 m behind.
- **Sonnet.** In 10 of the 13 cells that name a bystander, every named bystander is rejected. 0 of 42 named true
  members are rejected. Those 10 cells hold 277 of CT3's 322 s of leftover capture (86%).
- **DeepSeek.** 6 of 11 cells, holding 134 of 212 s (63%). But 9 of 65 named true members are rejected, in 9 cells.
  The cause: DeepSeek's target points are off (one front member is 4.9 m from its target point; s30's target is a
  bystander), so the parse inherits the target point, as CT3's follow check does.
- **Thresholds.** I tried three tolerances on the same data, all reported. At 0.4 and 0.5 m Sonnet falls to 8/13
  and 6/13, and DeepSeek's member rejections fall to 8/65 and 7/65.
- **Timing.** CT3's leftover capture before the first service after the answer: 59 s (DeepSeek) and 102 s
  (Sonnet). Between that service and the drop: 137 s and 160 s. The static likelihood acts at the answer frame;
  the dynamic one has to wait about 10-12 s for a service, then for the co-motion window.

**Why it is a new way of thinking, not engineering.**
- **One model serves as the robot's policy and as its observation model of other people.** Each kind in the
  vocabulary then yields a role likelihood with no new code:
  - come_after: where a waiting person stands, and how they step up;
  - keep_out_of_shared_space plus face: where a group member stands, on the ring around the shared space, facing
    its centre;
  - keep_out_of_view(target): where a watcher stands, with a line of sight to the target.
- CT3's hand-written follow check becomes one instance of a derived prediction.
- Reusable takeaway: *any spec-then-solve pipeline can check its spec against the people it names by asking whether
  its own solver would have put them there.*

**Closest prior work, and what is new.**
- Baker 2009 and Ramírez & Geffner, "Probabilistic plan recognition using off-the-shelf classical planners", AAAI
  2010 (VERIFIED): goal inference from a planner's costs, in mazes and in classical planning.
- Gray & Breazeal 2005 (VERIFIED): self as simulator, for teammates' goals.
- Robotics BOLD A (`IDEAS_robotics.md`): a belief over line hypotheses with the model's answer as the prior.
- MIGU (LANDSCAPE, VERIFIED): VLM priors fused into a grounding belief.
- New here:
  - the forward model is the zero-shot *social constraint solver*, evaluated per person and per constraint kind;
  - static evidence exists at the first frame, before any event;
  - the evidence above is measured against the leftover failure of an existing motion check.
- My searches found no work that uses a robot's own social-placement planner to verify the roles an LLM or VLM
  assigned to people (UNVERIFIED as an absence).

**Hypothesis, cheapest test, kill.**
- H: in a frozen replay from cached answers (0 calls), a resolver that drops a named person whose leave-one-out
  residual is above τ meets three bars:
  - it cuts CT3's named-bystander capture by at least 50% on both models;
  - it causes at most 3 false member drops in the 60 DeepSeek cells;
  - it causes no harm on the original scenes (P6 bars).
- Before freezing, add a hard case: a non-waiter standing *on* the line's axis, facing the counter (someone
  waiting for a friend). The static term should fail there and the dynamic term should catch them. That is the
  complementarity claim.
- Kill: capture down by less than 30%; or more than 3 false member drops; or the on-axis scene shows the combined
  likelihood is no better than CT3.

**Risks.**
- The bystanders were placed 0.75-0.8 m off the axis by construction, and fidget at most 0.35 m, so position
  separates them by design.
- Positions come from oracle tracks.
- The axis of a two-person line is defined by the target point.
- For groups, the solver's own group placement is weak (`constraint_method.md`, "One distance for everyone, and
  groups"), so group residuals will be noisy.
- Real people mill about, use phones and chat in line.

**Scores.** Payoff 4, uncertainty 3, cost 1.5-2 days.

### Idea 2. Arrivals are demonstrations: while the model thinks, copy the last person who came for the same thing

**In one sentence.** From the moment the plan call names the target, watch for an *arrival*: someone who walks
towards the target and comes to rest among the people standing still there. Take the end state relative to them as
a provisional goal (come_after [arrival]) until the constraints answer arrives, which then confirms or overrides it.

**Failure attacked.**
- Waiting for the first answer is 0.28 of failure time, and walking there another 0.11 (FAILURES §1).
- Moving while thinking (CT2m) failed because the target point is the counter, not the place to stand (SUMMARY §4).

**Borrowed intuition.** I3: copy when uncertain, and when learning alone is costly. Here the costly individual
learner is the 15-43 s VLM call. A single observed action reveals a norm. Rational imitation copies the end state,
not the path: Gergely, Bekkering & Király, *Nature* 415, 2002, VERIFIED, but its robustness has been contested in
later replications (UNVERIFIED snippet).

**Zero-call evidence (A4).** A (re)join is complete before the first answer, counting the join's start plus a 5 s
walk:

| Model | Episodes with an arrival in time | Share of the pre-answer wait it cuts |
|---|---|---|
| x-test-3 | 29/33 | 30% (444 of 1,501 s) |
| DeepSeek | 27/127 | 13% |
| Sonnet | 2/100 | 1% |

With a 3-7 s walk the shares are 26-34%, 12-15% and about 1%. In our sim the first arrival comes about 25-30 s in
(a 15 s service plus 8-15 s of wandering), so the idea pays only for answers slower than that. **Its value scales
with answer latency divided by the time between arrivals.** That is a statement about situations, not about which
model is used.

**Why it is a new way of thinking.**
- CT2m and robotics idea 7 move towards the *target*. Robotics idea 5 learns the line's frame from *service*
  events. Copying an arrival imitates the act of joining itself: exactly the quantity the robot needs, the place,
  in an ambiguous scene.
- It is situation-agnostic. A latecomer to an audience, someone joining the crowd at a lift and someone joining a
  conversation each show where to join.
- Reusable takeaway: *under a slow model, the people who solve the same social problem in front of the robot are a
  fast, free oracle.*

**Closest prior work.** Learning from demonstration is usually done offline, from a teacher. I found nothing that
uses bystanders' arrivals as one-shot, third-person demonstrations of social placement (UNVERIFIED as an absence).
LANDSCAPE B5, "let the people propose the target", uses where people face, not what they do.

**Hypothesis, test, kill.**
- H: with an arm that adopts come_after [arrival] when an arrival settles before the answer, x-test-3's median time
  to first in line falls by at least 8 s, with no more cut-in samples. Test: sim replay of cached answers, 0 calls,
  33 x-test-3 and 30 DeepSeek cells.
- Kill: cut-ins rise. Or the arm copies a walker or bystander in more than 10% of cells. The bystander scenes
  include walkers, so that case can be measured.

**Risks.** Copying the wrong person. Scripted returners always re-join the same line and always correctly, so in
the sim every arrival is a true demonstration: **this works partly because our humans are scripted.** A quiet cafe
has no arrivals.

**Scores.** Payoff 3, uncertainty 3, cost 1 day.

### Idea 3. Errors are not equal: score and decide with an asymmetric social loss

**In one sentence.** Separate the *rude* error (cutting in, which breaks a norm) from the *polite* one (waiting
behind someone who isn't waiting, which only costs time). Measure the human cost ratio between them, then use the
same asymmetric loss in the decision rule: under doubt about a person's role, defer to them until evidence clears
them.

**Failure attacked.** How CT3 and Gl are evaluated on the bystander scenes (A5, frozen capture rule, the sim's
per-second cut-in flag):

| | Capture, CT2 → CT3 | Cut-in samples, CT2 → CT3 | Gl (geometry chain) |
|---|---|---|---|
| DeepSeek confirmation (60 cells) | 1,005 → 203 s | 502 → 655 | capture 2,258 s, cut-ins 211 |
| Sonnet (30 cells) | 938 → 322 s | 89 → 138 | capture 1,666 s, cut-ins 27 |

- On a weighted loss, CT3 beats CT2 only if a cut-in second costs less than **5.2** (DeepSeek) or **12.6**
  (Sonnet) captured seconds. CT3 beats the "dumb" geometry chain only below **4.6** and **12.1**.
- Some of this is exposure: per second near the line, CT3 cuts in less than CT2. Cut-in samples over (in line +
  cut-in) samples fall from 0.70 to 0.57 for DeepSeek, and from 0.25 to 0.20 for Sonnet. So the result is not
  "CT3 is ruder". It is that the ranking depends on a weight nobody has measured.

**Borrowed intuition.**
- Queue intrusion draws objections from the people just behind it: Milgram, Liberty, Toledo & Wackenhut, *JPSP*
  51, 1986, UNVERIFIED (secondary sources only).
- 3-year-olds enforce inferred norms on third parties (Schmidt 2016, VERIFIED).
- Under doubt, people choose the error that is cheap to repair: stand behind the person and ask.

**Why it is a new way of thinking.**
- Our share metric, and the social-navigation evaluation guidelines LANDSCAPE cites (Francis et al.), score time
  in place. Nothing weights errors by the norm a bystander would see broken.
- It reverses how capture is read for an ambiguous person: deferring to them is the polite default, not the main
  failure.
- It builds on `IDEAS_social_eval.md` B3 (the "in order" and "in place" tiers) and takes it to a measured cost
  ratio plus a decision rule.
- Reusable takeaway: *report rude and polite errors separately, and set a policy's asymmetry from a measured human
  cost ratio.*

**Hypothesis, test, kill.**
- Kill: a ratio below 2, which means the current share metric is adequate.

**Risks.** The cut-in flag includes sim artefacts: the robot beside slot 0, never served (class B). Exposure
confounds raw counts. Clips are not live presence.

**Scores.** Payoff 4, uncertainty 2, cost 1.5 days, mostly the rating study.

### Idea 4 [BOLD]. When testimony and observation disagree, ask the person: "Excuse me, are you in line?"

**In one sentence.** Treat the VLM's roster as testimony and idea 1's leave-one-out residual as the robot's own
observation. Only where they disagree about a person who decides the goal (the candidate last) does the robot ask
that person. The alternative, a one-bit question to the VLM, is interface idea 5. The question is weighed by value
of information against a social cost of asking.

**Failure attacked.**
- CT3's leftover capture is all from named bystanders, and naming them depends on the model, not on its strength
  (Sonnet 13/29 against DeepSeek 10/51; SUMMARY §5).
- DeepSeek answered "people do not matter" in all 5 room cells.

**Borrowed intuition.**
- I3: children explore when evidence is confounded (Schulz & Bonawitz 2007, VERIFIED).
- Children trust informants selectively by past accuracy: Koenig & Harris, *Child Dev.* 76, 2005, UNVERIFIED
  (snippet). Epistemic vigilance: Sperber et al., *Mind & Language* 2010, UNVERIFIED (memory).
- People at lines ask exactly this question (Kamikubo et al. CHI 2025, VERIFIED, 35%).
- Passers-by help robots that ask: Weiss et al., HRI 2010, VERIFIED (abstract; field trial of a robot that asked
  pedestrians for directions).
- People underestimate how readily others comply with requests: Flynn & Lake, *JPSP* 2008, UNVERIFIED (secondary).

**Zero-call evidence (from A1-A2; the reply is an oracle).**
- **Sonnet.** The disagreement set is exactly the named bystander in 10/13 cells, and no named member in 0/42.
  That is at most one question in 10 of 28 cells. A truthful reply would remove up to 277 of 322 s of capture.
- **DeepSeek.** At most 15 questions in 60 cells: 6 that catch bystanders (134 of 212 s) and 9 about true
  members.
- The key point: **asking turns idea 1's false rejections from harms into small social costs.** A member asked
  "are you in line?" says yes, and nobody is wrongly dropped.

**Why it is a new way of thinking.**
- Here the scene's own people become the oracle for their own roles, and the trigger is *disagreement between two
  sources*, not a confidence threshold.
- Reusable takeaway: *the cheapest and most accurate sensor of a person's social role is the person; ask only when
  your informant and your own eyes disagree.*
- I found no robot that asks a bystander about their own role in a queue or group (UNVERIFIED as an absence).

**Hypothesis, test, kill.**
- H: with an oracle reply from the scenario's bystander labels, sim replay from cached answers (0 calls) removes
  at least 70% of CT3's named-bystander capture, with at most 0.5 questions per cell on average and no false drops.
- Real test: a Wizard-of-Oz pilot or a video survey on whether a robot asking is acceptable, compared with being
  captured or cutting in (shares idea 3's raters).
- Kill: the trigger fires in more than a third of cells with no bystander (nagging); or raters find the question
  more annoying than the capture.

**Risks.**
- **Only testable with an oracle reply, because sim humans are scripted.**
- Speech in noisy cafes; asking from across a room is odd; the cost of asking falls on bystanders; cultures
  differ.
- It inherits idea 1's dependence on the target point.

**Scores.** Payoff 5, uncertainty 4, cost 1 day (oracle replay), weeks for a real HRI test.

### Idea 5 [BOLD]. Show the model motion, not a snapshot: a Heider-Simmel view of the last 15 s

**In one sentence.** Add one image to the constraints call (within the four-image cap): a top-down plot of tracked
people as numbered discs, each with a time-coloured trail of the last 15 s, the target marked and no appearance.
The model reads who moved with whom, as people do from moving shapes.

**Failure attacked.**
- Bystanders named as waiting (above).
- A snapshot cannot tell walking from waiting: 9 model-episodes (`IDEAS_interface.md` idea 4).

**Borrowed intuition.** I2: roles from motion alone (Heider & Simmel, *Am. J. Psych.* 57, 1944, UNVERIFIED,
memory). Interaction judged from decontextualised trajectories (Shu et al. 2018, VERIFIED). Synchrony gives
groupness (Lakens & Stel 2011, VERIFIED).

**Why it is a new way of thinking.**
- Interface idea 8 is a *static* plan view. This adds time, and moves the inference about motion into the model,
  where it applies to any formation.
- Either outcome is informative. If it works, VLM social roles come from motion abstractions. If it fails, the
  community learns that motion reasoning must stay outside the VLM (idea 1).
- A warning: Ying, Huang, Netanyahu, Barbu, Katz, Tenenbaum & Shu, "Grounding social perception in intuitive
  physics", arXiv 2603.27410, Mar 2026 (VERIFIED, abstract page) report that strong VLMs miss human judgements on
  Heider-Simmel-style animations, where a Bayesian inverse planner (SIMPLE; PHASE, AAAI 2021, VERIFIED) succeeds.

**Hypothesis, test, kill.**
- H: with trails that cover at least one service, DeepSeek names bystanders in at most half as many answers (from
  10/51 to 5 or fewer), with members kept.
- Cheapest test:
  - Re-run 15 bystander cells in the sim with tracks logged from t = 0 (0 calls) and render the plots.
  - About 30 free DeepSeek calls; the snapshot answers are already cached.
- A watch window costs time. Answer to first service is a median 10-12 s (A3), so use the trail on an
  event-triggered second call and *integrate* its roster with the first (I5); do not wait before the first call.
- Kill: bystander naming does not fall, or members are lost.

**Risks.** VLMs read diagrams poorly (SoM, VPBench in LANDSCAPE). The trail inherits tracker errors. Added call
latency.

**Scores.** Payoff 4, uncertainty 5, cost 1.5 days and 30-60 calls.

---

## 3. Limitations a CoRL or HRI reviewer with this background would raise

1. **The humans cannot respond.** Contingent reactions are the main evidence people use for social structure:
   who yields, who objects, who answers. The sim removes all of them. Probing and asking (idea 4) are untestable
   except through oracles. Returners re-join periodically and always correctly, which flatters any idea built on
   observing them (idea 2).
2. **The bystanders are separable by design.** They stand off the axis and face away, and their fidget (at most
   0.35 m) stays below the join gate's 1.0 m walk. CT3's gate and my static parse both use designed cues.
   - Missing: non-waiters *on* the axis (waiting for a friend, reading the menu), waiting people who drift or chat,
     and gapped lines.
3. **The pipeline holds no uncertainty.** One VLM roster is executed as fact; the solver returns a point; there is
   no "I don't know", no hedging and no question. Human social inference is probabilistic and active. A
   cognitive-science reviewer will see CT2 and CT3's rules as patches over a missing belief state.
4. **All errors cost the same.**
   - Cut-in, capture and latency seconds are pooled.
   - On DeepSeek's bystander scenes, cut-in time exceeds in-line time for every arm (A5).
   - CT3's lower capture comes with more cut-in samples.
   - No human rating anywhere supports the 1.2 m band or the equal weights (SUMMARY §1 already concedes the band).
5. **Snapshot social cognition.** People size up a scene in seconds and keep watching. Our model sees one frame,
   15-43 s before it acts, and never sees motion. "The model knows who queues" rests on one family of designed
   scenes, and depends on the model.
6. **The robot's own legibility is ignored.** In real lines people read a newcomer's intent: where they stand, which
   way they face, whether they look at the line. A robot standing off the axis or facing the counter may not be
   read as queuing, and others will step in front of it. The sim cannot show this; I4 says it matters.
7. **Generality is claimed, not shown.** Conversation groups, audiences and lifts are untested with motion, and the
   solver's group placement has known limits. Queue norms differ between cultures; the scorer assumes one
   front-to-back FIFO line.

## 4. Top pick: Idea 1, "would I stand there if I were waiting?"

It turns the constraint solver into an observation model of other people as well as the robot's policy, so the VLM's
roster becomes a testable prior instead of fact. With zero calls it already rejects the named bystanders in the
cells where CT3's motion check left capture: Sonnet 10/13 cells (277 of 322 s, no member rejected), DeepSeek 6/11
(134 of 212 s, though 9/65 members rejected via bad target points). It is general by construction, since every
constraint kind gives a role likelihood. It unifies the static check, CT3's co-motion check and the re-ask trigger.
It is the foundation that ideas 3 and 4 need.

---

## Appendix A. Zero-call checks (run 2 Nov 2026, from the repo root, `.venv/bin/python -I`)

- **A1. Static parse** (`selfparse2.py R 0.3|0.4|0.5`). Inputs:
  - the first re-solve that logs `tracks` and `answer_come_after`;
  - named people counted as matched within 0.5 m (the frozen rule);
  - "still" = moved less than 0.25 m/s over the next 2 s;
  - the target point from `001/answer.json`.

  | Model | Cells | Name a bystander | All named bystanders rejected (0.3 / 0.4 / 0.5) | Members rejected (0.3 / 0.4 / 0.5) |
  |---|---|---|---|---|
  | DeepSeek, pilot + confirmation | 60 constraint-answer cells with logged tracks | 11 | 6 / 6 / 6 | 9 / 8 / 7 of 65 |
  | Sonnet | 28 | 13 | 10 / 8 / 6 | 0 / 0 / 0 of 42 |

  The DeepSeek member rejections, traced:
  - cafe s14, s17 and s42, and sweet s25: lateral 0.64-1.25 m at 1.7-1.9 m behind the kept front member, whose axis
    runs from DeepSeek's target point;
  - sweet s15: lateral 0.32 m, just over the tolerance;
  - sweet s11: the front member 4.93 m from the target point;
  - sweet s17, s30 and s35: judged behind a named bystander kept ahead of them, so one roster error causes a second.
    The cause behind any trace was not checked further.
- **A2. Complementarity** (same script): CT3's capture in the cells naming a bystander, split by whether the parse
  rejects all of them. Sonnet 277.4 s (10 cells) against 44.8 s (3); DeepSeek 134.0 s (6) against 78.2 s (5). CT2
  in the same cells: 528.4 / 52.5 s and 213.0 / 140.2 s.
- **A3. Evidence latency** (`evidence_latency.py R`; service = the per-second `members` count falling). Answer to
  first service: median 12.1 s [0.3, 16.6] for DeepSeek, 9.6 s [1.3, 14.9] for Sonnet. Follow-check drop of the
  named bystander: 7/11 and 11/13 cells. CT3 capture before the service / to the drop / after: 59 / 137 / 17 s and
  102 / 160 / 60 s.
- **A4. Arrivals** (`arrivals.py R 3|5|7`; arm CT2, else CT; a join = `members` rising while the robot has not
  joined; the sim appends a returner when their 8-15 s wander ends, before the walk back, hence the added walk).
- **A5. Rude against polite** (`rude_polite.py R`). Capture by the frozen rule; cut-in = the sim's per-second flag
  (the robot within 1.0 m of the axis and more than 0.25 m ahead of a member who was waiting before it,
  `experiment.py:340`).

  | Model and seeds | Arm | Cut-in samples | Captured s | Frozen in-line share |
  |---|---|---|---|---|
  | DeepSeek 15-44 | CT2 | 502 | 1,005 | 0.039 |
  | | CT3 | 655 | 203 | 0.090 |
  | | CT3b | 668 | 190 | 0.101 |
  | | Gl | 211 | 2,258 | 0.015 |
  | Sonnet 10-19 | CT2 | 89 | 938 | 0.097 |
  | | CT3 | 138 | 322 | 0.204 |
  | | Gl | 27 | 1,666 | 0.018 |

  Break-even cut-in weight = Δcapture / Δcut-in.
