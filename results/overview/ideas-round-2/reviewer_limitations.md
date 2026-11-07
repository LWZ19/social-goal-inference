# Through a demanding CoRL / HRI reviewer's eyes: limitations, the one contribution to add, and ideas (2 Nov 2026)

Lens: a reviewer who has read SUMMARY.md, FAILURES.md, the bystander and group4 READMEs, the report draft
(`docs/report/report.tex`, still static-only), `resolve.py`, `humans.py`, `experiment.py`, SCORING.md, DESIGN.md, the
three first-round ideas files and LANDSCAPE.md. Read-only, no model calls, nothing under keep/heldout or eval_16-19.

- **C1. Cut-ins go the other way for Sonnet** (`python3 -I cutins.py runs/sonnet55_low`).
  - Sonnet, CT2 against R1 over 50 pairs: CT2 has more cut-in samples in 25, fewer in 4, the same in 21.
    - Totals: 481 against 162.
    - Episodes with any cut-in: 26 against 10.
    - Where: the two-person lines (156 + 179) and cafe (119); none in the 6-person line.
  - DeepSeek over 25 pairs: 10 / 9 / 6, totals 226 against 504.
  - Part of the gap is exposure: R1 is rarely near the line, so it rarely gets a chance to cut in.
  - The 43/2 share headline is not reported next to this.
- **C2. The model's roster runs out on a predictable clock.**
  - Command: `roster_expiry.py RUNS ARM sim/scenarios`. "Runs out" means no place in turn for >= 3 s after having
    had one.
  - Sonnet CT; the answer arrives at t ~= 15 s.

    | Scene | People in line | Service every | Episodes where the roster ran out | Median, after the answer |
    |---|---|---|---|---|
    | The two two-person lines | 2 | 15 s | 17 / 20 | 20-22 s |
    | cafe | 3 | 20 s | 10 / 10 | 52 s |
    | room | 3 | 25 s | 5 / 10 | 70 s |
    | 6-person line | 6 | 15 s | 0 / 10 (CT2: 1 / 10) | - |

  - The prediction: the first service comes at about S, so the last named person is served at about n x S, and the
    roster runs out about n x S - 15 s after the answer. That gives 15, 45, 60 and 75 s.
  - Each measured value is the prediction plus 5-10 s of detection lag (the 0.75 m leave rule and the 3 s
    persistence).
  - For the 6-person line, 75 s after the answer falls at the episode's end (92 s), so its roster never runs out.
  - So **CT's maintenance failures occur only where the episode outlasts the roster**, and 92 s episodes hide them
    in long lines. Post hoc and explanatory.

---

## 1. Limitations a reviewer would raise, by severity

### Critical: each would sink a CoRL or HRI submission alone, if left unanswered

1. **"In line" is our own definition, and the solver aims at it.**
   - The place in turn is 1.0 m behind the last person. The scored band is 0.5-1.2 m.
   - The band comes from Fang et al. 2026, whose text says it derives from "social conventions ... and proxemics
     theory", without human calibration (VERIFIED, arXiv 2609.28467 HTML).
   - The effects that matter sit on the band's edges:
     - CT2 beats CT by +0.082, but by only +0.008 with the band's top at 1.5 m (SUMMARY §1).
     - Without the side check, Sonnet's CT2 leads R1 by +0.19, +0.06 and 0.00 at band tops of 1.2, 1.5 and
       2.0 m (SUMMARY §7).
   - The static report admits that "two shapes of the solver coincide with reference rules".
   - Rule and solver differ in exactly one held-out situation, a picture being taken; there every method scores
     0-2 of 36.
   - A reviewer writes: "success measures agreement with your own geometry."
   - **Answers: idea 1 (top pick) and idea 3.**
2. **Most of the evidence cannot see the VLM's social knowledge.**
   - On plain moving scenes, a geometry-only chain picks the right people in 15/15, 15/15 and 13/13 first frames
     (IDEAS_interface F2).
   - After the first answer, CT is model-independent: 0.22-0.24 for every model (FAILURES §6.6). So the headline
     against R1 compares "structured answer + solver + tracks" against "a pixel".
   - Where roles matter (bystanders), the share effect is +0.018 on a floor of 0.025 -> 0.043.
     - It passes for DeepSeek (p = 0.035) and does not replicate on Sonnet.
     - Sonnet names a bystander as waiting in 13 of 29 answers; DeepSeek in 10 of 51.
   - A reviewer asks: "Is this a VLM paper, or a queue tracker with a VLM-picked target?"
   - The honest answer today: maintenance and cost, plus roles under ambiguity on one model.
   - **Answer: idea 4.**
3. **The humans never react, and the sim grants membership by fiat.**
   - Walkers yield to anything for at most 2 s, then walk on (`humans.py` YIELD_M, MAX_YIELD_S). Nobody objects to
     a cut-in.
   - The robot is written into the queue's member list once it stops 0.3-1.5 m behind the last person, near the axis.
     The DESIGN note says within 0.6 m; the code's `CORRIDOR_M` is 0.8 m.
   - It is never served; for that reason 5.6% of CT's time is scored as in line (FAILURES §5E).
   - Service is lockstep FIFO to fixed slots.
     - CT3's follow check reads exactly that lockstep.
     - Real lines step up late, shuffle and let gaps open; there it is untested.
   - For HRI this is the first objection: no social claim stands without a social response.
   - **Answers: ideas 1, 5 and 6.**

### Major

4. **The harm metric is absent from the headline, and for Sonnet it goes the other way (C1).**
   - Report cut-ins per minute of exposure beside the share.
   - Add episode-level ordinal outcomes: never in line, lost its place, kept it, served in turn (asked for in
     IDEAS_social_eval B4).
   - **Answer: idea 3.**
5. **Few scenes, all of them queues, and the maintenance rules are queue-shaped.**
   - The scenes:
     - five straight FIFO lines on three Habitat stages;
     - a bent line and a member stepping out, n = 3 each;
     - three bystander variants;
     - 10 seeds.
   - n_eff is about 7 per comparison: half the variance lies between scenes (IDEAS_social_eval B4; Hurlbert,
     intuition 4).
   - The thesis is generality, yet no conversation group, audience, doorway or lift moves. The concert scenes exist
     but only served the detector check.
   - Every maintenance rule acts on `come_after`. Their wording is free of any situation; their effect applies to
     queues only.
   - C2 adds a design flaw: 92 s episodes are shorter than a 6-person roster's life.
   - **Answers: ideas 2 and 5.**
6. **The effects are small, and absolute performance is low.**
   - The best arm is in line 0.31 of the time (Sonnet CT2, strict scoring).
   - Redraw noise is about 0.1 per episode, and the timing floor leans 14/24.
   - Bystander gain: +0.018.
   - Detecting a 0.10 CT-R1 difference needs about 91 cells.
   - The headline counts (43/2, 32/8) survive this; most secondary claims don't, and the text must say so.
7. **Latency turns the robot into a statue for 15-43 s.**
   - That is longer than one service and 0.28 of all failure time. Real people would react to a robot standing
     still for 40 s.
   - Static runs take 1.4-1.9 times as long as the baseline; the proposal promised "a fraction of the time".
   - Moving while thinking failed. The mentor group's trained model answers in about 1 s (VERIFIED, LANDSCAPE).
   - A reviewer writes: "not deployable, and you don't show what latency would suffice." **Idea 2** gives that
     curve.
8. **Engineering or science, stated plainly.**
   - Science-grade claims:
     - constraints plus a solver beat pixel marking, including on one held-out situation;
     - re-binding beats asking once and re-asking (CT v1 against R1 +0.19, p < 0.001; Rq < R1);
     - model roles beat geometry under ambiguity (CT3 against Gl 24/5/22);
     - motion evidence halves the capture caused by the model's roster errors.
   - Engineering:
     - CT2's five rules, whose gain is the ratchet, a fix for the band edge;
     - the person snap (+3.7 points, narrow);
     - the seen-side solver rules (designed on the audience scenes);
     - the context-prompt rewording;
     - about a dozen thresholds.
   - The novelty, an "identity-referencing social spec", is thin against VoxPoser, ReKep, GSON and Code-as-Monitor
     (VERIFIED, LANDSCAPE), unless it carries a principle. **Idea 2** supplies one.
   - Lead with the frozen-before-data protocol: it is a real strength.

### Moderate

9. **Single view, everything in view.**
   - Every cafe first answer fails for every model, because an occluded member puts the place off the seen floor
     (FAILURES §5B).
   - The static scenes have exact depth and every relevant person in view.
   - **Answers:** idea 5 (JRDB is 360 deg) and IDEAS_interface 6.
10. **Baselines.**
    - R is known to be weak: zero-shot pose validity is 46-60% even for GPT-5.5 (Fang et al., VERIFIED).
    - Gl is the real competitor on plain scenes and belongs in the main table.
    - Missing:
      - a trained joining predictor;
      - a classical stand-in-line rule (Nakauchi & Simmons 2002, UNVERIFIED);
      - an oracle-roster upper bound (FAILURES 4c has the pieces).
11. **The models.**
    - Moving runs: Sonnet, DeepSeek and x-test-3. Static runs: Qwen and GLM.
    - No GPT or Gemini constraint runs.
    - "Role knowledge is not in order of model strength" rests on 29 and 51 answers.
12. **Sim and scoring artefacts.** They are documented and minor, but reviewers find them:
    - the 0.8 m corridor against the 0.6 m in DESIGN;
    - the 4-slot clamp in room;
    - R1's share, 46% of which comes while it is first and never served;
    - oracle tracks in most arms, although CTd shows no loss from that.

## 2. Intuitions from my field (evaluation methodology, HRI review practice)

1. **Construct validity.** A metric derived from the method's own rule measures conformance to it.
   - Francis et al. (THRI 2024, arXiv 2306.16740) call surveyed metrics "(arguably) the best way to measure social
     navigation success", and hand-crafted ones proxies that "cannot fully capture the desired property" (VERIFIED,
     HTML §VI-A1).
   - Video studies are a cheap stand-in: Woods et al. 2006 found moderate-to-high agreement between live and video
     HRI trials (VERIFIED, abstract as given in the repository listing).
   - SEAN-EP finds that perceptions differ between interactive simulation and video (VERIFIED, abstract, arXiv
     2012.12336).
2. **Simulation conventions get exploited.**
   - Our analogues: R1's share beside the line, the robot that is never served, membership by fiat.
3. **Age of Information, and of Incorrect Information.**
   - Kaul, Yates & Gruteser (INFOCOM 2012): under queueing there is an optimal update rate, and faster is not
     fresher (VERIFIED, catalogue abstract).
   - Maatouk et al. (arXiv 1907.06604): count only the time the receiver's estimate is wrong (VERIFIED, abstract).
   - Our "share in line" is 1 minus the goal's time-averaged Age of Incorrect Information.
4. **Pseudoreplication** (Hurlbert, Ecological Monographs 1984; VERIFIED, abstract via mirrors).
   - Seeds of one scene are not replicates. With a scene correlation of about 0.5 we have about 7 units, not 15-50.
5. **People enforce queues, mostly from behind.**
   - Milgram et al. (JPSP 1986): objections come mainly from those behind the intrusion (UNVERIFIED, from memory and
     a secondary summary).

## 3. If we could add only one contribution in a few days: ask people, not the ruler (idea 1)

The three numbers a reader will quote all hinge on what people count as being in line or cutting in:
- CT2's gain over CT, which is a band-edge effect;
- Sonnet's lead over the marked point, which vanishes without the side check at a 2.0 m band;
- Sonnet's tripled cut-ins (C1).

No further simulation can answer this, because every simulated result is scored by the same rule. A video-rating
study can.
- **Who and what:** 20-30 naive observers rate clips cut from our own logged episodes.
- **Cost:** 2-3 days and zero model calls.
- **Why it is worth it either way:** whichever way it comes out, it tells us which results are real.
- **What it leaves behind:** a reusable artefact, the first measured acceptance function for robot placement in a
  line (and, with the Blender renders, at a group). That function checks the 0.5-1.2 m / +-30 deg rule shared by the
  mentor group's benchmark and ours.
- **What it changes:** limitation 3 goes from "never tested with people" to "tested with people as observers", the
  usual first step at HRI.

The runners-up:
- **Idea 2:** zero calls and a real framing. Do it too.
- **Idea 5:** the biggest payoff, but 3-4 uncertain days.
- **More queue scenes:** needed, but they don't change what a reviewer believes.

## 4. Ideas

### Idea 1. Ask people, not the ruler: a human-calibrated acceptance function, then re-score everything

- **What.** Fit two functions to observers' judgments of short clips of our own robot:
  - P(seen as waiting in line | gap, lateral offset, heading, lag);
  - P(seen as cutting in | the same).

  Then re-score every frozen comparison with the expected acceptance.
- **Failure it attacks.** Limitation 1:
  - CT2 over CT: +0.082 -> +0.008 at 1.5 m;
  - Sonnet against R1 without the side check: +0.19 / +0.06 / 0.00;
  - SUMMARY §1 itself says "whether a 1.2-1.5 m gap reads as in line to people" can't be answered by these runs;
  - the cut-ins in C1.
- **Borrowed intuition.**
  - Intuition 1.
  - Psychophysics: fit a psychometric (logistic) curve along each dimension the scorer thresholds.
- **Why it is a new way of thinking.**
  - Success becomes a measured probability that people read the robot's behaviour socially, not a region drawn by
    the authors.
  - A benchmark can then state how uncertain its own success definition is.
  - Others can re-score their episodes with the released curve and clips, the mentor group's benchmark included.
  - The same protocol works for "is the robot intruding on this group?" on the static renders.
- **Closest prior work.**
  - Fang et al. 2026: a 15-person pilot where people clicked where they would join, compared by KL divergence; nobody
    rated robot poses (VERIFIED, HTML).
  - Kim et al. RA-L 2026: compared yielding, cutting in and FIFO with stakeholders (VERIFIED, LANDSCAPE).
  - SEAN-EP: crowdsourced feedback on social navigation (VERIFIED, abstract).
  - I found nothing that rates acceptance against the geometry that benchmarks threshold.
- **Hypotheses.**
  - **H1a.** Acceptance falls faster with lateral offset than with gap:
    - level with the last person but 0.8-1.2 m beside the line: P < 0.5;
    - on the axis, 1.2-1.5 m back: P > 0.7.

    If both hold, the side check is right and the 1.2 m edge is too tight. CT2's gain over CT shrinks, and the
    CT-vs-R1 headline holds.
  - **H1b.** Observers call at least half of CT2's logged cut-in samples a cut-in.
  - **Kill.** Beside-the-line positions get P > 0.6. Then Sonnet's lead over R is a scorer artefact, and we
    publish that.
- **Cheapest test (zero model calls, about 2.5 days).**
  1. Pick 80-120 five-second clips from CT, CT2 and R1 episodes, stratified by gap, side, heading, and moving or
     still. Include the cut-in samples.
  2. Replay them from cache and render the behind camera (agent 1 in `world.py`). This costs sim time only.
  3. Build a 10-minute form. Each rater sees 40 clips and answers "waiting in this line? yes / no / unsure" and
     "has it cut in?", with 3 attention checks.
  4. Fit a logistic model with a rater random effect, and re-score offline.
- **Risks.**
  - Watching a video is not standing in the line (SEAN-EP).
  - Spacing norms are cultural, and these raters are students in Singapore.
  - The avatars are artificial, and the people don't react.
  - Ethics: check that NUS IRB exempts an anonymous perception survey before collecting anything.
  - With fewer than about 15 raters, the slopes will be loose.
- **Scores.** Payoff 5, uncertainty 2, cost 2-3 days.

### Idea 2. Two clocks: geometric and semantic answer age, and a map of when to ask, re-ask or re-bind

- **What.** An answer about people ages on two clocks:
  - **geometric:** where people stand. It stays valid for tau_geo, about one service interval S.
  - **semantic:** who plays which role. It stays valid for tau_semantic, about n x S for a FIFO roster of n people.

  Re-binding resets the geometric clock continuously and leaves the semantic one running. The model's latency L
  then picks the regime:
  - **(a)** L well below tau_geo: ask once.
  - **(b)** L between tau_geo and tau_semantic: re-bind identities.
  - **(c)** L above tau_semantic: identities expire before the answer lands, so only role answers refreshed from
    motion, or a faster model, can work.
- **Failure it attacks.**
  - C1's 0.02-0.04 share, with stale goals 0.37 of its time.
  - Rq below R1: 0.14-0.16 against 0.20.
  - x-test-3's Cq: 0.27 of its time stale, with L = 43 s against services every 15-25 s (FAILURES §2, §5D).
  - CT's "the line ran out": 0.17 of CT's time (§5E).
  - C2's run-out at n x S - L plus 5-10 s, and Sonnet's cut-ins found only there (C1).
- **Borrowed intuition.** Intuition 3: share in line = 1 - time-averaged AoII.
- **Why it is a new way of thinking.** It turns the re-solve rules from a bag of fixes into parts of a theory:
  - CT2's re-admission and vacated place, and CT3's join gate and follow check, refresh the *semantic* clock from
    tracks without the model.
  - The ratchet is a fix for the *geometric* clock.
  - Any robot with a VLM in the loop can measure L, tau_geo and tau_semantic and read off its strategy.
  - It predicts that conversations (tau_semantic in minutes) fall in regime b, while doorways and concourses
    (seconds) fall in regime c. There, LANDSCAPE B2's role predicates or a faster model are needed, and re-binding
    by identity is futile.
- **Closest prior work.**
  - G2-Nav keeps 82% of its benefit at 4 s of staleness (VERIFIED, LANDSCAPE).
  - Slow Brain, Fast Planner decays a stale choice by geometry (VERIFIED, LANDSCAPE).
  - TIC-VLA conditions its actions on latency (VERIFIED, LANDSCAPE).
  - ReKep re-evaluates geometry with identities fixed.
  - LANDSCAPE Q1 (validity by event type) was proposed and never run.
  - What is new: two clocks, a regime map with computable boundaries, and maintenance rules recast as semantic
    refreshers.
- **Test (zero calls, about 3 h on 3 workers).**
  - **Setup.** Add a latency override (one line where `experiment.py` sets `latency`), and set service_s in the YAML
    (`cafe_line2_fast` already exists).
  - **Runs.** CT, CT2, C1 and R1 on cached first answers. L ∈ {1, 5, 10, 20, 40} s, S ∈ {6, 15, 25} s, on line2 and
    line6, 3 seeds: 360 episodes.
  - **Prediction.** C1 and R1 collapse once L exceeds about S/2. CT degrades only near n x S. The CT - C1 gap peaks
    in between.
  - **Kill.** CT degrades with L/S as fast as C1, or line6 degrades as early as line2.
- **Risks.**
  - Scripted FIFO makes both clocks clean. Real tau_semantic is shorter, which is why idea 5 measures it.
- **Scores.** Payoff 4, uncertainty 3, cost 1-1.5 days.

### Idea 3. Score what happens to the people: outcome metrics that share no geometry with the solver

- **What.** Read outcomes off the world's own state, per situation, and report each with its exposure:
  - **queues:** served in turn, skipped, or served out of turn; order inversions per minute within 1 m of the axis;
  - **groups:** seconds inside the group's shared space (o-space);
  - **audiences:** seconds of blocked line of sight between the watchers' eyes and the target, by ray casting;
  - **doorways:** delay imposed on people passing.
- **Failure it attacks.**
  - The never-served robot (FAILURES §5E).
  - The line stalls for a different time per arm: R1 12.0 s per episode against CT 7.2 (IDEAS_social_eval B1).
  - R1's share comes mostly from the scorer (FAILURES §6.1).
  - The reversed cut-ins in C1.
- **Borrowed intuition.**
  - Goodhart: measure outcomes, not proxies.
  - Queue fairness counts "slips and skips" (Larson, Operations Research 1987; UNVERIFIED, title only).
- **Why it is a new way of thinking.**
  - With no reference region, the metric cannot coincide with the solver's shape.
  - Benefit and harm share one unit: seconds of other people's time, and their turn.
  - It covers every situation the method claims.
  - It builds on IDEAS_social_eval B5 and B2, and takes them across situations and to harm per exposure.
- **Closest prior work.**
  - Francis et al. list politeness and agent understanding (VERIFIED).
  - HuNavSim has social metrics for passing (VERIFIED that it exists; its metric list UNVERIFIED).
  - Fang et al. report time violating personal space (VERIFIED).
  - None of them scores the consequences of choosing a goal.
- **Hypothesis.** Of the 50 Sonnet cells, CT2 is served in turn in at least 15 more than R1, and its cut-ins per
  exposure-minute are no higher than R1's.
  - **Kill:** its rate per exposure is higher. The honest headline then becomes "in line more, rude more once the
    roster runs out", which links to idea 2.
- **Test.**
  1. Exposure-normalised cut-ins from the logs: 1 hour, zero calls.
  2. A serve-and-skip counter (about 2 h in `experiment.py`), then R1, C1, CT, CT2 and CT3 on cached answers:
     about 250 episodes, zero calls.
  3. Audiences: the concert scenes, about 4 free calls per seed.
- **Risks.**
  - Service still needs a rule (within 0.6 m of slot 0): a point, but still ours.
  - The harm falls on scripted people.
- **Scores.** Payoff 4, uncertainty 2, cost 1.5 days.

### Idea 4. The geometry-gap curve: the VLM's marginal value along an ambiguity axis

- **What.** Build one-parameter families of scenes in which a geometry-only prior's error runs from 0 to 1, and
  plot the VLM method against Gl along that axis. The parameters, varied one at a time:
  - a non-participant's distance to the formation, 0.5-2.5 m;
  - their facing;
  - their stillness;
  - a second formation nearby: two lines, or a group beside a line.
- **Failure it attacks.** Limitation 2:
  - geometry is right in 15/15 on plain scenes;
  - Gl is captured in 51/60 bystander cells;
  - CT3 beats Gl 24/5/22;
  - Sonnet names bystanders in 13 of 29 answers;
  - DeepSeek answers "the people don't matter" in all 5 room cells.
- **Borrowed intuition.**
  - Benchmarks with calibrated difficulty (item-response theory).
  - Report against the strongest baseline that uses no model: SocialNav-SUB's rule-based baseline beat the best
    VLM (VERIFIED 17 Oct, LANDSCAPE).
- **Why it is a new way of thinking.** "Method beats baseline" becomes "from this level of ambiguity on, the model
  pays this much". Any VLM-for-social-robotics paper can copy the protocol. It also delivers the one-factor-at-a-time
  grid the proposal planned and never built.
- **Closest prior work.**
  - Our bystander scenes: a single point on the axis.
  - Fang et al.: GPT-5.5's grounding F1 is 0.83 (VERIFIED).
  - I found nothing like an ambiguity sweep.
- **Hypothesis.** Gl's capture rises monotonically as the bystander nears the line's end. The model's naming of
  bystanders rises more slowly, and CT3 > Gl becomes significant at 1.2 m or less.
  - **Kill:** the model's naming errors track geometry's capture. The claim then narrows to "the VLM gives the
    target and the situation; tracks give the people".
- **Cheapest test.**
  1. Zero calls: extend `bystanders_geometry.py` into the sweep.
  3. Replay the solver offline; run moving cells only at the extremes.
- **Risks.** Without the group-beside-line and two-formation families, the sweep stays queue-flavoured.
- **Scores.** Payoff 4, uncertainty 3, cost 2 days.

### Idea 5. [BOLD] Off the script: re-binding on real people's motion, scored against labels someone else wrote (JRDB)

- **What.** Run the ask-once-and-re-bind loop open-loop on JRDB, a mobile robot's 360 deg recordings of people,
  using its 3D tracks. Score the re-bound roster of a conversation group over time against JRDB-Act's group IDs,
  which nobody on this project wrote, and measure real tau_semantic.
- **Failures it attacks.** Limitations 1 (our own scorer), 3 (scripted lockstep), 5 (only queues move) and 9
  (single view).
- **Borrowed intuition.** Where closed-loop tests with people are impossible, replay real logs open-loop. Labels
  written by others break the circularity.
- **Why it is a new way of thinking.**
  - It tests our central claim, that a slow answer about people stays valid as they move, on real motion and on
    conversations.
  - It measures how fast real social structure goes stale, which is what idea 2 needs.
  - It gives the community an "answer half-life" benchmark for slow VLM social reasoning.
- **Closest prior work.**
  - JRDB-Act (Ehsanpour et al., CVPR 2022; VERIFIED, pp. 3-4):
    - 54 sequences, 64 minutes, on a campus;
    - five stereo cameras giving 360 deg, two 16-beam LiDARs, 3D boxes with track IDs;
    - group IDs per frame;
    - actions labelled at 7 fps, among them "talking to someone", "listening to someone", "looking at robot" and
      "interaction with door";
    - no "waiting in line" class.
  - GSON keeps group labels on tracks but scores them frame by frame, not as a slow answer's validity (VERIFIED,
    LANDSCAPE).
- **Hypothesis.** A single VLM roster of a group, re-bound on true tracks, keeps a Jaccard of at least 0.9 against
  JRDB's IDs for a median of at least 60 s. A frame-by-frame re-ask at L = 15 s does worse.
  - **Kill:** real tau_semantic for groups is shorter than L (15-20 s). Conversations would then sit in regime c,
    itself a striking negative result.
- **Cheapest test.**
  1. Zero calls: register (non-commercial licence), then compute how often group membership changes from the labels
     alone. That gives tau_semantic with no model.
  2. Then 40 keyframes x 2 free models x 2-3 calls = 160-240 free calls. Lift positions from LiDAR or the true 3D
     boxes (which tests roles only), re-bind, and score.
- **Risks.**
  - Registration may take time.
  - The pipeline expects a single RGB-D camera.
  - JRDB's "social group" is not our "shared space".
  - Its people react to the JackRabbot, not to our robot.
  - Which sequences contain lines or lifts: UNVERIFIED.
- **Scores.** Payoff 5, uncertainty 4, cost 3-4 days.

### Idea 6. [BOLD] People who answer back: their reactions as evidence that the answer is wrong

- **What.**
  - Give the scripted line one reaction that enforces its norm. When someone's place is taken, the person behind
    steps forward to close the gap and turns to face the intruder (Milgram).
  - Test a rule that names no situation: "a person who is not my target turns to me and comes within 1 m: yield, and
    re-solve behind them".
  - This extends CT3's evidence from motion, from "others' motion reveals their roles" to "others' motion towards
    me reveals that my own role is wrong".
- **Failure it attacks.**
  - Sonnet's 481 cut-in samples (C1).
  - 82% of cut-in onsets come within 10 s of someone rejoining (IDEAS_social_eval, finding 4).
  - The people never react (limitation 3).
- **Why it is a new way of thinking.**
  - The people become the supervisor: some wrong answers show only in how people respond to the robot.
    - a cut-in;
    - an intrusion into a group's shared space;
    - blocking an audience's view.
  - One signal serves lines, groups and audiences: "repair", as in conversation, applied to placement.
- **Closest prior work.** HuNavSim's reactive behaviours (VERIFIED that they exist; the list UNVERIFIED). I found
  nothing that uses reactions to revise a VLM's social goal.
- **Hypothesis.** With the reaction rate taken from the literature and not tuned, the rule removes at least 50% of
  the cut-in seconds, without lowering the strict share.
  - **Kill:** reactions can't be told from ordinary step-ups, so false yields outnumber true ones.
- **Test.** Zero calls.
  - About 3 h of work in `humans.py`.
  - Then CT2 and CT3, each with and without the rule, on the two-person lines: 80 episodes from cache.
- **Risks.** It is circular by construction, since we script both the reaction and the rule that reads it. It
  demonstrates a mechanism only. Real reactions are subtler: gaze, speech.
- **Scores.** Payoff 3, uncertainty 5, cost 2 days.

## 5. The earlier ideas: build now, or drop

| Earlier idea | Verdict | Why |
|---|---|---|
| social_eval B5, a counter that serves and skips | **Build now** (inside idea 3) | Removes the never-served artefact; zero calls for the asked-once and CT arms |
| LANDSCAPE Q1 (validity by event type) and Q3 (triggers) | **Build now** (the sweep in idea 2) | C2 already shows the semantic clock |
| interface 3, one tag per person | **Build next** (60 free calls, then Sonnet) | It is aimed at CT3's leftover error: bystanders the model names (10/51, 13/29), and the gap that no constraint kind says "not waiting" (BYSTANDERS F2). It must help most models |
| robotics BOLD A, a belief over line hypotheses | **Build only as a reformulation** | Worth it if it replaces CT2/CT3's dozen thresholds with fewer parameters (limitation 8), not for share |
| interface 6, ask from where the model can see | **Build next** (36 calls, cafe) | The fix for the single view: all 9 cafe first answers fail |
| interface 5, a one-bit question on an event | Defer | Yes-bias, and each question adds latency again. Keep it only for idea 2's regime c |
| interface 7, ask "whether", not "who" | **Drop as a method; keep as a finding** | The bystander test came out for "who": Gl captured 51/60, and CT3 beat Gl 24/5/22 |
| interface 1 (seed and close the chain); interface 2 (union of draws) | Drop | Seed-and-close is Gl, which bystanders capture. A union adds the errors that now dominate: bystanders named |
| interface 8, a plan-view diagram | Defer | A new image type confounds the test; it targets no top limitation |
| robotics 7 / BOLD B, leave before the answer | Drop | CT2m lost (12/27): the target is not the place |
| social_eval A5, follow the person ahead | Drop for the report | Control engineering on the band edge |
| LANDSCAPE B1, the answer states its own expiry | Drop now | Monitors written by the model are DoReMi / Code-as-Monitor ground, and C2 shows the expiry can be predicted from tracks |
| LANDSCAPE B2, roles not numbers | Drop now; revisit for regime c | Bystanders capture geometric role rules. Idea 2 says they are needed only where tau_semantic < L |
| LANDSCAPE B3, B6, B7; robotics 9 | Drop | Small payoff, or targets don't move |
| LANDSCAPE B4, tracker IDs as the numbers | Done in part (E1b, CTe) | Comparing re-asks is moot while we don't re-ask |
| LANDSCAPE B5, people propose the target | Defer | On moving lines target errors are rare: 1 wrong target and 5 with none of 60 confirmation answers |

## 6. Top pick, in five lines

1. Idea 1: an acceptance function fitted to people's ratings of clips of our own robot, then everything re-scored
   with it.
2. Our quotable results hinge on what people call "in line" or "cutting in": CT2 over CT is a band-edge effect, and
   Sonnet's lead over R vanishes without the side check.
3. Sonnet's CT2 also cuts in 3x as often as R1 (481 against 162 samples).
4. More simulation can't settle this, since our rule scores it and the solver aims at it. 20-30 raters on about 100
   clips can, in 2-3 days with zero calls, and the answer is useful either way.
5. It leaves the community the first measured acceptance curve for robot placement in a line, a check on the shared
   0.5-1.2 m rule.
