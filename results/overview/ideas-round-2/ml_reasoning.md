# Ideas round 2: machine learning and large-model reasoning (2 Nov 2026)

Lens: the VLM as a prior, not an oracle; generate-then-verify; self-consistency and calibration; falsifiable answers;
test-time compute against latency; distillation; motion as a label. No model calls; read-only apart from this file.
Every number below is post hoc on logged runs (seeds 10-44 of the bystander scenes, both models), so read them as
leads, not results.

the tracks logged every 0.5 s, and `queue_ids`), plus `CT2/001/answer.json` and the scenario YAMLs.
- `roster_consistency.py`: how often each person is named under `come_after`, per scene, over first answers whose
  frames are near-identical (seeds change only fidget and walkers).
- `named_facing.py`: the model's own people-stage heading for each person it named, against the bearing to its
  target point.
- `verify_latency.py`: for answers that named a bystander, when the first service came and when CT3 dropped him.
- `group_test.py`: an open-loop replay of "every advance tests every named person behind the advancer".
- `cost_named.py`: CT3's strict share in cells where the answer named a bystander, against cells where it did not.

## 1. Intuitions from ML and LLM reasoning

**I1. Voting removes variance, not bias.** Self-consistency samples several reasoning paths and keeps the most
consistent answer (Wang et al., arXiv 2203.11171, VERIFIED). Agreement across samples is a usable confidence signal,
and verbalised confidence is overconfident (Xiong et al., ICLR 2024, arXiv 2306.13063, VERIFIED). Both work only when
errors are random draws. A model that is consistently wrong gets more confident under voting. *For us:* which kind are
the bystander errors? Our logs answer this per model (idea 1).

**I2. LLM-Modulo: the model proposes, sound external verifiers dispose.**
- Kambhampati et al. (ICML 2024, arXiv 2402.01817, VERIFIED) argue that LLMs cannot plan or verify by themselves.
  They treat LLMs as approximate knowledge sources inside a loop with model-based verifiers.
- Self-critique collapses performance. A sound external verifier gives large gains, and simply re-prompting with that
  verifier keeps most of the benefit (Stechly et al., arXiv 2402.08115, VERIFIED abstract).
- Intrinsic self-correction often makes answers worse (Huang et al., arXiv 2310.01798, VERIFIED).
- *For us:* our sound verifier is physical: how the people move after the answer. CT3's follow check is a first,
  ad hoc version of one.

**I3. Models' generations disagree with their own judgement.** Even GPT-4 agrees with its own validation of its
answers only 76% of the time (Li et al., "Generator-Validator Consistency", arXiv 2310.01846, VERIFIED abstract;
venue not on the page). *For us:* the pipeline already asks the model twice about each person: what it sees (people
stage) and what role it gives them (constraints stage). Measured gap (`named_facing.py`):
- Where a model named the lone bystander as waiting, its own people stage had him facing more than 90 deg from the
  target in 6/11 (DeepSeek) and 7/13 (Sonnet) of those namings. For named true members the rates are 13/93 (14%) and
  15/58 (26%).
- Its own labels for those men include "facing away", "walking away", "seen from behind" and "walking in middle".
  The `because` texts are generic ("Persons 1, 2 and 3 are lined up waiting their turn").
- Naming follows distance. The lone man stands 0.92 m from the line's last and the pair 1.55-2.26 m. Lone man named:
  DeepSeek 0.12-0.26, Sonnet 0.10-0.78 by scene. Pair: 0-0.11 for both.
- *Reading (a hypothesis, idea 3):* VLMs assign waiting roles by proximity gestalt plus a script ("people near a
  counter queue"), not from the cues they themselves perceive.

**I4. Hypotheses tested on data, decided sequentially.**
- Hypothesis Search (Wang et al., ICLR 2024, arXiv 2309.05660, VERIFIED): the LLM states hypotheses, turns them into
  programs, and checks them on observed examples (ARC 17% -> 30%).
- Wald's sequential probability ratio test stops as soon as the evidence reaches preset error rates (classic;
  UNVERIFIED, from memory).
- *For us:* an answer of the constraint method is a hypothesis about how people will move. A line advance is a free
  experiment that tests every claim at once.

**I5. The future is a free annotator, and two views can teach each other.**
- Self-supervised field robotics labels images with the robot's later experience. Footholds projected into the camera
  label terrain without annotation (Wellhausen et al., RA-L 2019; UNVERIFIED: search snippet and OpenAlex record, the
  ETH page returned an error).
- Co-training uses two conditionally independent views to label for each other (Blum & Mitchell, COLT 1998;
  UNVERIFIED snippet).
- Students trained on a weaker supervisor's labels beat that supervisor (Burns et al., arXiv 2312.09390, VERIFIED).
- *For us:* a still frame (the VLM's view) and motion over time (the tracker's view) are nearly independent views of
  the same social roles.

- The value of computation (Russell & Wefald; UNVERIFIED snippet) is used to train LLMs to reason only when it pays
  (De Sabbata et al., arXiv 2410.05563, VERIFIED: 20-37% fewer tokens at equal performance).
- Compute-optimal test-time scaling depends on how hard the prompt is (Snell et al., arXiv 2408.03314, VERIFIED).
- *For us:* re-binding to tracks makes latency cost waiting time only, not staleness. A verifier in the loop changes
  which errors are worth extra thinking.

---

## 2. Ideas

### Idea 1. Bias or variance first: self-consistency is a weak-model fix for social roles

**One sentence.** Before choosing between more samples and an external verifier, split the model's per-person role
errors into variance (fixed by voting) and bias (made worse by voting). On our bystander scenes the stronger model's
errors are bias.

**Failure attacked.** Bystanders named as waiting (`SUMMARY.md` §5: DeepSeek 10/51, Sonnet 13/29). Also a direct
test of interface idea 2, "union over parallel draws", which I build on and argue against.

**Evidence (zero calls, `roster_consistency.py`).** Share of first constraint answers naming each person:

| Scene | DeepSeek: members / lone bystander / pair | Sonnet (low effort): members / lone / pair |
|---|---|---|
| cafe | 0.94, 0.64 / 0.12 / 0.00, 0.00 (33 answers) | 0.90, 0.90 / 0.10 / 0.00, 0.00 (10) |
| sweet_spot | 0.74, 0.78 / 0.26 / 0.11, 0.07 (27) | 1.00, 1.00 / **0.78** / 0.00, 0.00 (9) |
| room | no constraint answers (0/5) | 0.89, 0.78 (mirror), 0.78 / **0.56** / 0.11, 0.11 (9) |

- **DeepSeek:** its lone-man naming looks like noise, so majority-of-3 voting would cut it.
  - Sweet_spot, assuming independent draws: 0.26 -> 0.17.
  - It would also lift its weakest member: cafe, 0.64 -> 0.70.
- **Sonnet:** in sweet_spot the lone man is named about as often as the members. Voting would raise his rate
  (0.78 -> 0.88), and a union of two draws would include him in 95% of answers.
- **So:**
  - Interface idea 2 (union over draws) would make the strong model's bystander problem worse.

**Borrowed intuition.** I1 (Wang et al.; Xiong et al.).

**Why it is a way of thinking, not engineering.** It is a protocol to run before any test-time-compute fix in
VLM-in-the-loop robotics:
- measure the per-entity agreement between same-frame redraws;
- treat agreement on a wrong answer as a bias, not a confidence;
- send bias to verifiers that use other information, and variance to sampling.

The reusable claim is "in social scenes the stronger model's role errors can be consistent, so sampling-based fixes
favour weak models".

**Closest prior work.**
- Wang et al. (VERIFIED) and Xiong et al. (VERIFIED) give consistency as a fix and as a confidence signal.
- KnowNo (Ren et al., CoRL 2023, arXiv 2307.01928, VERIFIED) uses conformal prediction over LLM options to decide when
  to ask for help. It also assumes the model's distribution carries the uncertainty, which a consistent bias violates.
- I found no study that splits VLM social-role errors into bias and variance per person.

**Hypothesis, test, kill.**
- **Hypothesis:** on same-frame redraws, Sonnet names the sweet_spot lone man in at least 60% of draws, while
  DeepSeek's rate stays at or below 30%.
- **Zero-call part:** done above. Its caveat is that seeds are near-redraws, not true redraws: fidget moves the
  bystanders by up to 0.35 m.
- **Calls:** 5 frames per scene × 5 draws of the constraints call, with the people answer cached and a trial index in
  the cache key.
  - DeepSeek: 2 scenes × 25 = 50 free calls.
- **Kill:** Sonnet's same-frame rate for the lone man is at or below 0.3. Then its error is variance after all, and
  voting is a fair fix for both models.

**Risks.** Only one bystander template. Sonnet ran at low effort only (idea 6 tests effort). Frame-to-frame changes
may drive the rate more than sampling does; the same-frame test separates the two.

**Scores:** payoff 3, uncertainty 2, cost 0.5 day.

### Idea 2. Answers are forecasts: every constraint predicts motion, and the scene tests them all at once (top pick)

**One sentence.** Read the model's answer as a set of falsifiable claims about future motion. Each vocabulary kind
compiles to a forecast:
- `come_after [a,b]`: a and b advance when the person ahead of them advances, and they stay on the chain;
- `keep_out_of_shared_space [x,y]`: x and y stay mutually close and move together;
- `keep_out_of_view(target) [p]`: p keeps attending to the target.

Then use each natural event (a service, a group shift) as an experiment that tests every claim it touches, and decide
with a sequential test with error rates fixed beforehand, not hand thresholds.

**Failure attacked.**
- All of CT3's leftover capture comes from bystanders the model named (`SUMMARY.md` §5): 203 s for DeepSeek and 322 s
  for Sonnet, with 13 Sonnet cells still captured.
- The cause is that CT3's follow check is pairwise: only the next person behind the advancer is checked. Measured
  (`verify_latency.py`, `group_test.py`):
  - **Sonnet:** the first service comes a median 9.6 s after the answer. Yet CT3 drops the named lone man a median
    41.6 s after the answer, and never in 3 of 15 namings.
  - **DeepSeek:** 27.8 s, and never in 6 of 16.

**Evidence (zero calls, open loop, `group_test.py`).** Rule: "at every advance, everyone named behind the advancer must
come 0.25 m nearer within 5 s".
- **K = 1** (drop after one miss):
  - Bystanders dropped: Sonnet 15/15 at a median 10.4 s; DeepSeek 15/16 at 8.9 s.
  - Named people dropped while in the queue: 0. The pool is 59 + 61 named non-bystanders for Sonnet (bystander scenes +
    originals) and 96 + 28 for DeepSeek.
  - It also drops 11-28 named people who had already left the queue, served people walking back, whom readmission
    must restore.
- **K = 2:**
  - Bystanders dropped: Sonnet 15/15 at 27.7 s; DeepSeek 13/16 at 20.1 s.
  - Other drops: 0-2.
- The two settings are the operating points a sequential test chooses between.

**Borrowed intuition.**
- I4: hypothesis-and-test, Wald's SPRT.
- I2: an external, physical verifier (LLM-Modulo).
- The brief's "falsifiable answers": surprise, the accumulated log-likelihood against the answer, is the re-ask
  trigger (idea 4).

**Why it is a way of thinking, not engineering.** Three takeaways for readers:
1. A VLM's social answer is not ground truth to execute but a generative model of what people will do next. It can
   be falsified without asking the model for anything new.
2. The scene runs experiments for free: one service tests every waiting claim at once, and one group re-forming tests
   every shared-space claim.
3. Decisions come from a sequential test whose error rates (false drop α, missed bystander β) are set from human
   motion statistics, not from our sim. The per-model prior comes from each model's logged precision; for example,
   Sonnet's named-person precision on bystander scenes is about 58/74 = 0.78.

Situation-agnostic re-ask triggers (LANDSCAPE Q3) then become one quantity: the total surprise. It extends to
audiences and groups through their kinds' forecasts, with no queue code.

**Closest prior work.**
- CT3 (ours) is the pairwise, single-kind special case.
- Robotics BOLD A (`IDEAS_robotics.md` §6) weighs whole-line hypotheses by motion likelihood, queue-shaped. I build on
  it, but per claim and per kind.
- LANDSCAPE B1 has the model write its own expiry conditions. Here they are compiled from the semantics, with no new
  output burden.
- DoReMi and Code-as-Monitor (CVPR 2025, arXiv 2412.04455, VERIFIED) have VLM-written monitors on tracked constraint
  elements, for manipulation failures, not people's roles.
- T2-VLM (arXiv 2507.04789, VERIFIED abstract) keeps a VLM prior on subgoal completion, updated by Bayesian tracking,
  for manipulation rewards.
- TRACER (arXiv 2609.18776, VERIFIED abstract) keeps identity-bound beliefs over humans' response modes, updated from
  observed responses.
- Group detection from trajectories is old: Solera et al., TPAMI (UNVERIFIED, memory).
- New against all of them:
  - testing a zero-shot model's social spec claim by claim, with its own vocabulary compiled into forecasts;
  - natural events used as tests;
  - error-controlled decisions.

**Hypothesis, test, kill.**
- **Hypothesis (closed loop):** with forecasts for `come_after` and shared space and a two-miss sequential rule, each
  for both models:
  - leftover capture from named bystanders falls by at least 50% against CT3;
  - the bystander is decided within one service interval + 5 s in at least 80% of namings;
  - at most 1 false drop per 30 cells on the originals.
- **Cheapest test:** zero calls.
  - Run CT3 against CT3+forecasts in the sim from cache (`SGI_CACHE_ONLY=1`): DeepSeek's 60 confirmation cells,
    Sonnet's 30 bystander cells (cache only, a miss is an error), and the no-harm originals.
  - Then the robustness run, which is the key one: perturb the scripted members. Each skips a step-up with
    probability 0.2 and lags 0-6 s, and bystanders occasionally step 0.5 m towards the line. Fix α and β before
    looking.
- **Kill:**
  - under perturbed members, false drops exceed 2 per 30 cells at every setting that still halves capture; or
  - closed-loop capture does not fall by 50%, because CT3's leftover is then not about decision latency.

**Risks.**
- **"Only works because sim humans are scripted":** members step up crisply and bystanders never step. This is the main
  risk, and the perturbation run is how to measure it.
- Forecasts are uninformative while nothing happens: a still group stays at its prior, which is honest but gives no
  gain.
- An advance is measured towards the answer's target point. A wrong target (sweet s30) breaks it, so measure it
  along the remembered line where one exists.

**Scores:** payoff 4, uncertainty 3, cost 1.5-2 days.

### Idea 3. What the model saw against what it said: a generator-validator gap, and a cue study that explains it

**One sentence.** Use the gap between the model's own perception (people stage) and its role judgement (constraints
stage) as zero-call evidence against a named person. Then measure, with controlled re-renders, which cues VLMs
actually use to assign waiting roles.

**Failure attacked.** Bystanders named as waiting. In the bystander scenes a geometry chain is captured in 51/60
cells (`SUMMARY.md` §5), so roles are where the model must earn its place.

**Evidence (zero calls, I3).**
- The lone bystander faces more than 90 deg away by the model's own heading in 6/11 (DeepSeek) and 7/13 (Sonnet) of
  his namings, against 14% and 26% for named members: a likelihood ratio of about 2-4.
- His naming rate is ordered by distance to the line (lone at 0.92 m above the pair at 1.55-2.26 m) for both models.

**Borrowed intuition.** Generator-validator consistency (Li et al., VERIFIED). Factored evidence: say what you see,
then let a fixed rule turn facts into roles.

**Why it is a way of thinking, not engineering.**
- **Reusable test:** for any multi-stage VLM pipeline, check that later stages are consistent with the model's own
  earlier perceptual statements. Disagreements localise errors at zero cost.
- **Reusable science:** controlled re-renders in a simulator can attribute VLM social judgements to cues (distance,
  facing, being on the axis), like an ablation study of the model's social prior.
  - A finding of "proximity, not attention" would tell the community which verifier to pair with VLMs: the cue they
    ignore.

**Closest prior work.**
- Li et al. (VERIFIED) is text-only.
- EgoGroups (arXiv 2603.22249, VERIFIED abstract) finds VLMs zero-shot beat supervised group detection, with density
  and region as factors. It has no cue attribution.
- SocialNav-SUB (in LANDSCAPE, VERIFIED) is a question-answering benchmark.
- MINGLE (arXiv 2509.13484, UNVERIFIED snippet) fine-tunes a VLM on pairwise same-group decisions.
- I found no causal cue study of VLM queue or waiting roles.

**Hypothesis, test, kill.**
- **Hypothesis:** moving the lone man from 0.9 to 1.6 m changes Sonnet's naming rate at least twice as much as
  turning him from facing away to facing the target.
- **Test:**
  - Re-render the sweet_spot and room first frames, 2 × 2 (distance × facing), 5 draws each: 40 frame-draws per model.
  - Zero-call side: add the self-gap as a prior term in idea 2 and replay.
- **Kill:** facing changes the rate by at least 0.4, so the model does use attention. Then the gap is noise in its
  heading estimates, and the prior term should be dropped.

**Risks.**
- The mirror member (turned 90 deg, in line) shows facing alone misleads, so use it only as a soft term.
- Habitat avatars' facing may read differently to VLMs than real people do.
- The 8-direction heading lift is noisy.

**Scores:** payoff 4, uncertainty 3, cost 1.5 days.

### Idea 4. Evidence-conditioned re-asking, with a sycophancy control

**One sentence.** When the motion test falsifies the answer's structure, re-ask with a short, situation-free statement
of what people did, such as "since your answer, 2, 3 and 4 each moved about 0.7 m towards the cart within 3 s of 1
leaving it; 5 did not move". Measure whether the VLM revises because of the evidence or simply complies.

**Failure attacked.** Structural errors that no re-solving can fix:
- in room with bystanders, DeepSeek answered "people do not matter" 5/5 and drove through the line;
- 5/60 answers found no target;
- a wrong plan target (sweet s30);
- cafe's walker named in place of the occluded member (`SUMMARY.md` §5; FAILURES §2B, 0.10 of failure time).

Blind re-asks don't help:
- Cq's later answers were valid at delivery in only 0.20-0.23 of cases (`IDEAS_robotics.md` §8);
- Cq's re-asks threw away good first answers (323 s, FAILURES surprise 8).

**Borrowed intuition.**
- LLM-Modulo back-prompting with verifier critiques (VERIFIED).
- Inner Monologue: closed-loop language feedback improves embodied planning (arXiv 2207.05608, VERIFIED).
- Models accept coherent external evidence but show confirmation bias (Xie et al., ICLR 2024, arXiv 2305.13300,
  VERIFIED).
- Sycophancy, agreeing with what the prompt implies, is widespread (Sharma et al., arXiv 2310.13548, VERIFIED).

**Why it is a way of thinking, not engineering.** It asks a question the community needs answered before closing any
VLM loop with observations: do VLMs revise social readings in proportion to physical evidence?
- Control 1, false evidence about a true member, separates revising from complying.
- Control 2, a bare re-ask, separates evidence from simply getting a second draw.

Stechly et al. (VERIFIED) found that bare re-prompting with a sound verifier keeps most of the gain. Whether that holds
for social scenes decides whether the evidence text is worth writing.

**Closest prior work.** Inner Monologue, Code-as-Monitor and DoReMi feed back success or failure signals, not
person-level kinematic evidence about social roles. Interface idea 5 (one-bit questions) asks a fresh closed question
and does not test belief revision or compliance. I found no sycophancy-controlled study of VLM social-role revision.

**Hypothesis, test, kill.**
- **Hypothesis:**
  - true evidence makes the model drop the falsified person in at least 70% of re-asks, and turns room's "people do not
    matter" into a constraint answer in at least 3/5;
  - false evidence drops a true member in at most 30%;
  - a bare re-ask fixes at most 30%.
- **Test:** about 25 logged first frames, with the evidence text built from the logged tracks, × 3 arms (true, false,
  bare) on the constraints call only.
  - DeepSeek: 75 free calls.
- **Kill:** the false-evidence compliance rate is within 0.2 of the true-evidence revision rate (the model just obeys).
  Then motion stays a hard verifier and is never put into prompts.

**Risks.**
- The evidence sentence is a new shared prompt part and must use no queue words.
- Evidence is clean because humans are scripted.
- It only matters after idea 2 has decided something.

**Scores:** payoff 4, uncertainty 4, cost 1 day.

### Idea 5. [BOLD] Time is the teacher: hindsight co-motion labels distil social roles into a fast still-frame model

**One sentence.** Label every person in a first frame by what they did in the next 30-60 s:
- co-advanced with the chain: waiting;
- stayed mutually close and oriented: group;
- still and attending to the target: watching;
- otherwise: none.

Then train a small, fast model on the A100 to predict those labels from the still frame and the detector's people.
At t ≈ 0 the robot has roles that come from the scene's own outcome, not from a slow teacher's bias.

**Failure attacked.**
- Latency 0.28 + approach 0.11 of failure time; the first constraint answer takes 15-43 s.
- Bystander roles: Sonnet's lone-man rate of 0.78 would be copied by any student of Sonnet.

**Borrowed intuition.** I5: self-supervision from future experience, co-training of two views, weak-to-strong.

**Why it is a way of thinking, not engineering.** Existing distillation copies the teacher; the mentor group trains
on GPT-4o labels. Hindsight labels are free, scale with simulation or with a robot's own logs, and can beat the
teacher on exactly the cue it lacks, motion. The reusable claim is "in social navigation, what people do next is a
free annotator for what they are doing now". It applies to queues, groups, audiences and lift lobbies alike.

**Closest prior work.**
- SOPD-SocialNav (arXiv 2607.19850, VERIFIED abstract): entropy-selective on-policy distillation of a large VLM into a
  lightweight one; conversation and queue scenarios on a Scout Mini.
- ViLAM (arXiv 2503.09820, UNVERIFIED snippet): distils VLM reasoning into attention maps.
- "From Obstacles to Etiquette" (VERIFIED in LANDSCAPE): Qwen2.5-7B on GPT-4o labels.
- Self-supervised group relation embeddings exist, and trajectory clustering into groups is standard (search
  snippets; UNVERIFIED).
- I found no work that labels single frames for social roles by the people's future motion and uses that to replace
  a VLM's roster.

**Hypothesis, test, kill.**
- **Step 0 (zero calls):** validate the labeller.
  - In the logs, the two-miss group test of idea 2 already separates named bystanders from members with 0-2 errors.
    But that is the sim, so it is no evidence for real people.
  - On JRDB-Act (CVPR 2022, VERIFIED: per-person group IDs, actions, confidence tags), compute co-motion group labels
    from its trajectories and compare them with the annotated groups.
  - **Kill 1:** F1 below 0.7. Then hindsight labels are not social truth outside the sim.
- **Step 1:** generate about 2,000 randomised Habitat episodes (lines, groups, audiences, bystanders, walkers; new
  layouts kept out for test) and train a per-person classifier on detector crops plus relative geometry.
- **Hypothesis:** on held-out layouts, the student's named-person precision is at least Sonnet's (about 0.78 on
  bystander scenes) and above the geometry chain's, at under 0.5 s. CTe-style episodes then reach the line at least
  10 s sooner.
- **Kill 2:** precision on unseen layout templates falls below the chain's. Then the student learned the sim's
  templates.

**Risks.**
- Template leakage: our bystanders always stand alike.
- Scripted humans make sim labels perfect.
- The project's zero-shot claim is diluted.
- The earlier flow sampler failed on unseen situations (`SUMMARY.md` §6), though for circular labels, which hindsight
  labels avoid.
- JRDB access needs a login.

**Scores:** payoff 5, uncertainty 5, cost 5-8 days.

### Idea 6. A verifier changes where thinking pays: the value of computation under re-binding

**One sentence.** With answers re-bound to tracks and checked by motion, extra model deliberation on who waits is
worth less than its latency. Extra deliberation pays only on errors no verifier can catch: the target, whether people
matter, and occluded structure.

**Failure attacked.** Latency (0.28) against first-answer errors (0.39), traded through the reasoning-effort knob.

**Evidence (zero calls).**
- After the first answer, CT's in-line share is model-independent (0.22-0.24, FAILURES §5A), so each second of
  thinking costs about 0.2 s in line.
- A named bystander costs CT3 about 0.04 strict share (`cost_named.py`):
  - DeepSeek: 0.032 with a named bystander against 0.072 without;
  - Sonnet: 0.110 against 0.150. That is about 3.6 s of a 91 s episode.
- Sonnet names one in about 13/28 answers. So 10 s more thinking (about 2 s in line) pays for roles only if it fixes
  essentially every naming. After idea 2 the cost of a naming falls further.

**Borrowed intuition.** I6 (rational metareasoning; compute-optimal test-time scaling).

**Why it is a way of thinking, not engineering.** It gives a rule other teams can reuse when they pair a slow model
with a fast loop: put test-time compute on the claims the physical world cannot verify. The value of a model's
thinking is not fixed; it falls with every verifier added downstream.

**Closest prior work.**
- De Sabbata et al. (VERIFIED) and Snell et al. (VERIFIED) allocate compute by difficulty, not by downstream
  verifiability.
- TIC-VLA and Slow Brain, Fast Planner (in LANDSCAPE) compensate for latency rather than pricing it.

**Hypothesis, test, kill.**
- **Hypothesis:** high reasoning effort halves neither Sonnet's bystander naming nor the delay to fix it, but adds at
  least 8 s. It cuts the structural errors (no target, "people do not matter") by at least half.
- **Kill:** high effort removes at least 80% of bystander namings at 5 s extra or less. Then thinking does pay for
  roles.

**Risks.**
- Strict shares are near the floor on bystander scenes, so use capture seconds as well.
- The effort knob differs by provider.

**Scores:** payoff 3, uncertainty 3, cost 0.5 day.

---

## 3. Limitations a CoRL or HRI reviewer would raise (from this lens)

1. **The verifier's power is a property of scripted humans.**
   - Members step up crisply.
   - Bystanders only fidget (at most 0.35 m) and never step.
   - CT3's walk branch separates them by construction (BYSTANDERS_FROZEN "Known limits").
   - Zero false drops (idea 2) says nothing about people on their phones. Without a perturbation or real-trajectory
     test (JRDB), "motion corrects the VLM" is a sim claim.
2. **The VLM is handicapped by design.** It sees one still frame, while the verifier sees 70 s of tracks. A reviewer
   will ask for the obvious baseline: two to four frames across the latency window, within the four-image cap, so the
   VLM can see motion itself. If that baseline closes the gap, the story changes from "verify with motion" to "show
   the model motion".
3. **Pseudo-replication in the bystander evidence.** There is one bystander template in three scenes.
   - DeepSeek's 33 cafe answers are near-redraws of one picture.
   - Per-person rates (idea 1) are frame-and-sampling mixtures, not model properties.
5. **Small, near-floor effects.**
   - Bystander strict shares are 0.025-0.15.
   - CT3 against CT2 is +0.018 for DeepSeek and not replicated for Sonnet.
   - Capture is the substantive measure but is our own metric. Report time to service once the robot is served (social
     eval B5).
6. **Roles matter only where we built bystanders.** On the original scenes geometry picks the right people 15/15, so
   the model's social contribution is shown on scenes designed to need it.
7. **No calibration anywhere.** The pipeline acts on the roster as if certain. Nothing records the model's confidence,
   and no claim is made about when to trust it. Idea 1 is the first measurement.
8. **Latency is the gateway's, and the strongest competitor is trained.** "Where Should I Join?" answers in about 1 s.
   A reviewer will see a 15-43 s zero-shot pipeline as a non-starter for moving scenes unless something like idea 5,
   or the planned constraints-call distillation, closes it.
9. **Program synthesis was not pursued, deliberately.** The constraint vocabulary is already a program over tracks.
   Asking the model to write new predicates (LANDSCAPE B2) invites situation-specific drift. Idea 2 compiles fixed
   forecasts per kind instead. A reviewer may still ask whether model-written kinematic predicates would generalise
   better; that is untested.

## 4. Top pick: Idea 2, answers are forecasts

1. It attacks the largest error left after CT3, model-named bystanders (203 s / 322 s of capture), for both models.
   It does not help the weak model only: Sonnet gains most, because its errors are bias that voting cannot fix
   (idea 1).
2. A zero-call replay already shows the mechanism. Testing everyone behind each advance decides Sonnet's named
   bystanders at a median 10.4 s, against CT3's 41.6 s (15/15 against 12/15), with no named person dropped while in
   the queue.
3. The idea is reusable beyond queues. A VLM's social answer is a generative model of future motion; each vocabulary
   kind compiles to a forecast; natural events are free experiments; a sequential test sets the error rates; and the
   accumulated surprise is the situation-agnostic re-ask trigger.
4. It costs no calls to test closed-loop, from cache. The decisive check is a perturbed-member run that measures how
   much of the gain is the scripted sim.
5. It feeds ideas 3 (the model's own perception as a prior), 4 (falsification triggers an evidence-conditioned
   re-ask) and 5 (the same test, run in hindsight, is the label for distillation).
