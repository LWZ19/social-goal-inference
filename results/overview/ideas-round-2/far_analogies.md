# Far analogies: what other fields already know about our failures (ideas round 2, 2 Nov 2026)

## 0. Zero-call checks run for this note (all post hoc, bystander scenes unless said)

- **K1. Per-person diagnostic table at the answer frame** (constraint answers; DeepSeek seeds 10-44, Sonnet 10-19).
  The model's `come_after` against the true roster at ask time, and Gl's 1 m geometry chain:

  | Source | Member sensitivity | Lone-bystander specificity | Pair specificity |
  |---|---|---|---|
  | DeepSeek roster (60 answers) | 93/120 = 0.78 (complete in 40) | 0.82 (named 11/60) | 0.96 (5/120) |
  | Sonnet roster (28 answers) | 58/65 = 0.89 (complete in 25) | **0.54** (named 13/28) | 0.96 (2/56) |
  | Geometry chain, DeepSeek frames (70) | 0.94 | 0.06 | 0.54 |
  | Geometry chain, Sonnet frames (29) | 0.96 | 0.03 | 0.38 |

- **K2. Motion's time to a verdict.** First service after the answer arrives: median 10.2 s (IQR 5-15). Named bystanders
  that CT3's follow check dropped: DeepSeek 10 of 16, at a median **27 s** after the answer (10-62); Sonnet 12 of 15,
  median **41 s** (15-70); the rest never.
- **K3. The maintained roster against the truth, every 0.5 s resolve after the answer** (solver's `last` against the
  true last person ahead of the robot):

  | | right | **empty while a line stands** | a bystander | other |
  |---|---|---|---|---|
  | DeepSeek CT2 / CT3 / Gl | 0.18 / 0.29 / 0.07 | **0.33 / 0.48** / 0.00 | 0.32 / 0.06 / 0.67 | rest |
  | Sonnet CT2 / CT3 / Gl | 0.19 / 0.39 / 0.05 | **0.14 / 0.21** / 0.00 | 0.47 / 0.16 / 0.80 | rest |

  In the bystander scenes every empty case has every named person dropped (original scenes, DeepSeek CT3: all but
  0.09 of resolves). Cells whose named set emptied while a line stood: DeepSeek CT2 26,
  CT3 41; Sonnet CT2 6, CT3 10; original scenes CT3 DeepSeek 9/13 and Sonnet 17/28. In **0 of these 109 arm-cells** did a
  place in turn ever come back. During empty resolves someone stands within 0.5 m of slot 0 in 85-87%. The robot is in
  line in 1-3% of them and the sample is a cut-in in 24-30%. **CT3's bystander fix turns capture into an empty roster**
  (DeepSeek: bystander 0.32 -> 0.06, empty 0.33 -> 0.48).
- **K4. Cut-in against capture, seconds per cell** (frozen primary sample flags; capture from `last`):
  DeepSeek, 75 cells: CT2 11.7 / 16.2, CT3 13.8 / 2.8, Gl 2.9 / 40.3. Sonnet, 30 cells: CT2 3.0 / 31.3, CT3 4.6 / 10.7,
  Gl 0.9 / 55.5. In-line s/cell: DeepSeek 3.8 / 8.6 / 1.2; Sonnet 8.9 / 18.5 / 1.6.
- **K5. The sim has no start-up wave.** Room scenes, 177 consecutive-member pairs after a service: the member behind
  starts moving a median **0.00 s** after the member ahead (IQR 0.00-0.53; 62% within one 0.5 s resolve).
- **K6. Why the lone man was named.** The model's own people description of him says "facing away" in 1 of 11 (DeepSeek)
  and 0 of 13 (Sonnet) such answers; his facing is a bare number ("facing -173 deg"). The `come_after` reasons are
  proximity words: "lined up heading to the counter", "gathered near the counter", "queued toward the cart".

## 1. Intuitions from other fields

1. **Databases: a stored result is not a query** (incremental view maintenance). A *materialized view* goes stale; a
   *view definition* is re-evaluated. For recursive views, deletions are the hard case. DRed ("delete and rederive")
   over-deletes, then re-derives from the base facts. If the base facts themselves are deleted, the closure is empty
   for good. Gupta, Mumick, Subrahmanian, SIGMOD 1993 (UNVERIFIED, search summary). KnowRob's "computable predicates"
   are the robotics cousin: facts computed on demand from perception (Tenorth & Beetz; UNVERIFIED).
2. **Linguistics: names versus descriptions.** A proper name ("person 6") keeps its referent when the role changes; a
   definite description ("the person at the end of the line") changes referent with the world. Artale, Kontchakov,
   Mazzullo, Wolter, arXiv 2405.07656 (VERIFIED, abstract) formalise non-rigid designators "that designate different
   objects (or none at all) at different states". Ship of Theseus: a line is a *continuant* that survives the total
   replacement of its members. Our resolver identifies the line with its members at t0.
3. **Ethology: proximity is not association; following is.** The "gambit of the group" treats animals seen together
   as associated (Franks, Ruxton, James, Behav Ecol Sociobiol 2010, VERIFIED, abstract). That it inflates association at
   shared resources (waterholes) is field lore (UNVERIFIED). Leadership is read from the *delay* with which one animal's
   heading follows another's (Nagy, Ákos, Biro, Vicsek, Nature 2010, VERIFIED, abstract). The same method works for
   walking humans (Lombardi, Warren, di Bernardo, arXiv 2006.03700, VERIFIED, abstract) and for single-file walking,
   where the delay depends on density (Xue et al., Chinese Physics B 2023, VERIFIED, abstract). Traffic says the same:
   queued vehicles start one after another (start-up lost time; HCM, UNVERIFIED).
4. **Law: possession is an act that tells others.** First possession counts because it is a clear act that the
   audience can read (Rose, U Chicago L Rev 1985; UNVERIFIED, citation and secondary summaries). A queue is a small
   social system of claims, held with place-markers and kept by continued presence (Mann, AJS 1969; UNVERIFIED).
5. **Epidemiology and formal methods: tests, verdict timing, harm ratios.** A test has sensitivity, specificity and a
   time to result. Serial testing gains only when the tests' errors are independent. Decision curves rank strategies by
   net benefit at a stated harm ratio (Vickers & Elkin, Med Decis Making 2006, VERIFIED, abstract). Runtime monitors
   return true, false or **inconclusive** on a partial trace, and "monitorability" asks when a verdict can come at all
   (Bauer, Leucker, Schallhart, TOSEM 2011, VERIFIED, abstract).

## 2. Ideas

### Idea 1. Views, not rosters: an answer is a query with witnesses, maintained by delete-and-rederive (top pick)

- **One sentence.** Store `come_after` as a view definition, not as people: an *anchor* (the target), a *relation*
  (what the kind says: "waits their turn at the target") and *witnesses* (the numbers named). The witnesses fit the
  relation's parameters and check it. Maintain the view by re-deriving it from the anchor, so it never runs dry while
  someone still satisfies the relation.
- **Failure attacked.**
  - The line running out: `ct_ran_out` is 0.17 of CT's time (FAILURES §6.2).
  - Stale answers: 0.10 of failure time.
  - K3, new: in the bystander scenes the roster is empty while a line stands in 0.33-0.48 (DeepSeek) and 0.14-0.21
    (Sonnet) of post-answer resolves, and in 0 of 109 arm-cells does it recover. That is CT3's largest remaining loss
    there, larger than bystander capture (0.06).
- **Borrowed intuition.** Intuitions 1 and 2. The resolver is a recursive view whose base case is a list of proper names,
  so deleting all of them (served) is a base-case loss that no re-derivation can repair. CT2's re-admission needs a
  current last to absorb behind, so it cannot restart an empty closure (K3).
  The dual anomaly is closure by proximity alone, the gambit of the group: Gl's chain takes the lone man in 94-97% (K1).
  Both are one maintenance problem: the base must be persistent (the anchor), and the step relation must test
  membership, not nearness.
- **Why a new way of thinking.** Two of our failure classes, "runs out" and "captured", are the two textbook anomalies of
  view maintenance: a base-case deletion, and an over-permissive recursive rule. What a reader reuses:
  - any VLM or LLM spec that names tracked entities (object IDs in manipulation, agent IDs in multi-robot work) should
    separate anchor, relation and witness;
  - the model's answer is a labelled example set: those named are positives, the unnamed people in view negatives;
  - **a correctness criterion for any re-binding scheme**: the maintained view equals the view recomputed from scratch on
    the current state, measured every tick (K3's "right" column: 0.05-0.39 today).

  The same split covers a conversation group (anchor: the group's shared centre; relation: faces it) and an audience
  (anchor: the thing watched).
- **Closest prior work, and what is new.**
  - B2 (the model writes a role predicate). Here the predicate is fixed by the kind, and the model supplies examples.
  - Interface idea 1, seed-and-close: a proximity closure from the person nearest the target. It flagged its own
    bystander risk; K1 measures that risk at 94-97%.
  - Robotics idea 3: the line is a place, with path memory.
  - CT3's join gate: an act-based step for newcomers.
  - KnowRob computables (UNVERIFIED) and open-world reference resolution (Williams & Scheutz, AAAI 2016; UNVERIFIED):
    descriptions evaluated on demand, but static, with no maintenance correctness.
  - DRed (UNVERIFIED): databases, never applied to social specs.

  New: the anchor/relation/witness split, DRed re-rooting at the anchor, and the per-tick view-correctness metric. I
  found nothing in social navigation that does this.
- **Hypothesis and test.**
  - *Variant CT-view = CT3 plus re-derivation.* When no named person still waits:
    - base: the still person nearest the target within the front gap the witnesses showed at t0 (± 0.5 m);
    - step: a still person behind a member within 1.5x the witnesses' spacing, who also passes CT3's join gate if they
      arrived after the answer;
    - then CT3's follow check as now.
  - *Zero calls:* cache-only sim reruns (`SGI_CACHE_ONLY=1`, abort on any miss) on the 60 DeepSeek confirmation cells,
    the 28 Sonnet bystander cells, and the original scenes for no-harm.
  - *Predictions:*
    - the empty share falls to ≤ 0.15 (DeepSeek, from 0.48) and ≤ 0.08 (Sonnet, from 0.21);
    - bystander-last stays ≤ 0.10 / 0.20;
    - cut-in seconds per cell fall by ≥ 30%;
    - "right" rises above 0.5.
- **Risks.**
  - The anchor is the model's target point. Where it is wrong (sweet s30, target on a bystander: 1 of 60), the base is
    wrong too.
  - A real counter has staff and people waiting for their order at the front, which would be false base cases. Our
    scripted front is always the next customer.
  - Bent lines need a path tangent for the step (robotics idea 3).
  - Generality to groups and audiences is argued, not tested on moving scenes.
- **Scores.** Payoff 5, uncertainty 2, cost 1.5 days (zero calls).

### Idea 2. Who waits is a historical fact: split newcomers from incumbents, and test reading against information

- **One sentence.** Membership is made by acts: walking to the end and stopping, then stepping up with the person ahead.
  One frame cannot show them, so:
  - decide *newcomers* (whose act the robot sees) by motion alone;
  - treat the model's verdict on *incumbents* (already standing when the robot looked) as provisional until their first
    act.

  First, measure whether the model's bystander errors are missing evidence or misread evidence.
- **Failure attacked.** Bystanders named as waiting: DeepSeek 10/51, Sonnet 13/29 (SUMMARY §5). By K1, lone-bystander
  specificity is 0.82 (DeepSeek) and 0.54 (Sonnet), against 0.96 for the pair. All of CT3's leftover capture comes from
  bystanders the model named.
- **Borrowed intuition.**
  - Intuition 4 (possession is a readable act).
  - Intuition 3: the model reasons like the gambit of the group. In K6, its reasons for naming the lone man are
    proximity words, and its own description almost never registers that he faces away.
  - The stronger model is the less specific one, which fits a bias in the inference, not a lack of capacity.
- **Why a new way of thinking.** It moves role inference from "who is in the picture" to "who did what since we
  looked". Two reusable outcomes:
  - **an allocation rule** by observability: acts are observable for newcomers, so the model is not needed; for
    incumbents, only present cues exist until the first advance (K2: median 27-41 s);
  - **an evidence-ablation protocol** for VLM social-role reading (cue present but ignored, against cue absent).
- **Closest prior work.**
  - Interface idea 4 (motion during latency) and interface idea 7 (ask whether, not who).
  - CT3's join gate, which already is the newcomer half.
  - "Where Should I Join?", which requires membership to persist for 3 frames (VERIFIED in LANDSCAPE).
  - Group detection from coherent motion, e.g. trajectory clustering for groups (UNVERIFIED, search).

  New: the incumbent/newcomer split as the rule for when to trust the model, and the information-against-reading test.
- **Hypothesis and test.**
  - *Zero calls first:* in the stored `001/inference.txt`, check whether the people list carries a usable facing cue for
    the lone man (K6 says only as a bare number).
  - *Then a quick probe on 10 bystander answer frames, three conditions:*
    - as now;
    - facing as a clause ("faces away from the counter");
    - plus one conduct line per person ("has stood within 0.3 m of one spot", or "stepped 0.7 m forward when #2 did"),
      written from the scenario.
  - *Prediction:* the conduct line halves lone-bystander naming without lowering member sensitivity, and the facing
    clause alone does not.
- **Risks.**
  - Scripted bystanders only fidget and members step in sync, so conduct lines are cleaner than real ones.
  - A friend walking up to chat with a member performs a "newcomer act".
  - The conduct line is extra prompt text, so it must stay situation-free ("stepped when the person ahead stepped").
- **Scores.** Payoff 4, uncertainty 3, cost 1 day plus 60 free calls.

### Idea 3. Score and decide like a screening programme: per-person Se/Sp, time to result, and a stated harm ratio

- **One sentence.** Treat each membership source as a diagnostic test with a sensitivity, a specificity and a time to
  result: the model's roster, the geometry chain, and co-motion. Combine them at an explicit cut-in-to-capture harm
  ratio, and report decision curves, not one in-line share.
- **Failure attacked.** Bystanders, and the hidden trade-off behind CT3's small win (strict share 0.025 -> 0.043).
  - By K4, CT3 trades capture for cut-ins: DeepSeek capture 16.2 -> 2.8 s per cell, cut-in 11.7 -> 13.8.
  - Gl has the fewest cut-ins of all (2.9).
  - If a cut-in second costs w capture seconds, Gl beats CT3 for DeepSeek once w > 3.4, and CT2 beats CT3 once
    w > 6.4. For Sonnet the crossings are at 12.1 and 12.9.
- **Borrowed intuition.** Intuition 5.
  - The tests are complementary: the chain is a high-sensitivity rule-out test (Se 0.94-0.96, lone Sp 0.03-0.06); the
    roster is a rule-in test for pairs (Sp 0.96).
  - The roster and the chain err together on the lone man, because both read proximity. Chaining them gains nothing
    there.
  - Only motion is an independent channel, and its result arrives late (K2).
- **Closest prior work.**
  - Robotics bold idea A: a belief over line hypotheses, maximising P(valid). It has no harm ratio and no time to
    result.
  - B3 in the social-evaluation round (two scoring tiers).
  - DCA (VERIFIED). I found no social-navigation paper that reports per-person role Se/Sp or ranks methods across
    harm ratios.
- **Hypothesis and test.**
  - *Done, zero calls:* K1 and K4.
  - *Next, zero calls:* a "doubt goes behind" resolver rule, run from cache on the same cells. A named person not yet
    confirmed by an advance, or an unnamed still person inside the chain, counts as waiting for the robot's place until
    motion decides.
    - Predict lower harm-weighted cost than both CT3 and Gl for every w ≥ 3.
  - *Then, no model calls:* a small rating study. About 30 raters watch paired clips (captured robot / cutting-in
    robot) to estimate w. If raters put w near 1, the plain share is the right primary.
- **Risks.**
  - Most of CT3's cut-ins come from the empty-roster fallback beside the counter (K3: 24-30% of empty resolves), so
    idea 1 moves this ranking too. Rerun K4 after it.
  - All of this is post hoc rescoring.
  - Capture is read from the solver's `last`, not from the robot's pose.
- **Scores.** Payoff 4, uncertainty 2, cost 0.5 day (rating study 2-3 more days).

### Idea 4. [BOLD] The advance is a wave: membership, order and hidden members from response timing

- **One sentence.** When the front person is served, the advance travels back through the line with a per-person
  reaction delay τ, as at a traffic light's queue or in a pigeon flock. So:
  - a member is whoever the wave reaches;
  - the order is the order of arrival;
  - **an occluded member shows up as a doubled delay** between two visible people.
- **Failure attacked.**
  - Occluded members: in cafe, every first answer fails because a hidden member puts the place off the seen floor
    (`fa_noturn` 0.097 of failure time, FAILURES §5B).
  - Bystanders, which no wave reaches.
  - It also gives CT3's follow check a principled form: a lag test in place of a fixed 5 s window.
- **Borrowed intuition.** Intuition 3: directional-delay leadership in pigeons; lagged cross-correlation in walking
  groups; the density-dependent delay of single-file walkers; start-up lost time in traffic.
- **Why a new way of thinking.** It turns membership into *signal timing*, and a hidden person can be inferred from
  timing alone, without seeing them. K5 shows our sim has no wave at all (median lag 0.00 s). So co-motion tests are
  easier in our sim than they will be with people, and that is itself a reviewer point.
- **Closest prior work.**
  - Nagy 2010; Lombardi et al. 2020; Xue et al. 2023 (all VERIFIED, abstracts).
  - Lemercier et al., CGF 2012: following and stop-and-go waves "when they queue" (UNVERIFIED, search summary).
  - Group and extended-object tracking (Mihaylova et al., DSP 2014; UNVERIFIED).

  I found nothing that uses response delay to decide queue membership, or to infer a hidden member, for a robot.
- **Hypothesis and test (zero model calls).**
  1. On the public Jülich single-file trajectories (archive DOI 10.34735/ped.da; access UNVERIFIED), estimate each
     follower's τ.
  2. Delete one walker's track and test whether a doubled gap detects the hidden walker. Target AUC ≥ 0.8.
  3. Add a synthetic fidgeting bystander 0.9 m off the file: the lag test should never admit him.
  4. Give the sim's step-ups a per-person delay of 0.5-1.5 s (a realism fix in `humans.py`). Rerun CT3 from cache:
     does its false-drop rate rise?

- **Risks.**
  - Real lines step up "accordion" style: people wait for a gap and ignore small ones.
  - The Jülich data are walking, not waiting for service.
  - The robot needs about 10 Hz tracks of the people behind the hidden one.
  - Only a weak test is possible in our sim until it has delays.
- **Scores.** Payoff 4, uncertainty 5, cost 2-3 days.

### Idea 5. Unknown is not false: three-valued social constraints whose "inconclusive" means look, not default

- **One sentence.** Today every stage turns "can't tell" into a default:
  - no usable `come_after` -> the standard goal;
  - unseen floor -> not floor (the closed-world assumption);
  - nobody named left -> no line;
  - membership undecided -> as the model said.

  Instead, return *inconclusive* with a reason (tail occluded; person undecided). Its action is to go and look, or to
  hold the cheap side, never the default. Log how long each verdict takes to come.
- **Failure attacked.** Occluded members (cafe 9/9 first answers; `fa_noturn` 0.097), the fallback beside the counter
  when the roster empties (K3: 24-30% cut-in samples), and motion verdicts that come 27-41 s late (K2).
- **Borrowed intuition.** Intuition 5 (inconclusive verdicts, monitorability), and negation as failure in databases
  (UNVERIFIED, memory).
- **Why a new way of thinking.** A social-goal solver normally returns one pose. A solver that says "inconclusive,
  because the tail is hidden" turns occlusion and latency into a perception action. It applies unchanged to a
  half-hidden conversation group or an audience behind a pillar.
- **Closest prior work.** Robotics ideas 6 and 8 (ask from where the model can see; re-ask from a vantage point on
  surprise) give the action; this gives the trigger's semantics. STL work in social navigation plans to specifications
  but does not monitor social roles (Ruo et al., arXiv 2404.00353; UNVERIFIED).
- **Hypothesis and test.**
  - *Zero calls:* on every stored first frame (`001/` depth plus answer), mark "inconclusive" when the floor 0.5-1.2 m
    behind the last named person is not observed in depth.
    - Predict it fires on the cafe first frames and on ≤ 10% of the others.
  - *Then, cache-only sim:* on "inconclusive", move side-on about 3 m to see the tail without entering the line's
    corridor, then re-derive (idea 1) with no new call.
    - Predict cafe CT share rises from 0.09.
- **Risks.** Looking costs time. Noisy depth could make "inconclusive" fire everywhere. Scripted people never block
  the robot's look.
- **Scores.** Payoff 3, uncertainty 3, cost 1 day.

### Analogies tried and dropped (decorative here, or already built)

Event- and self-triggered control (= LANDSCAPE T2-T4 and B1); the separation principle (= robotics bold A);
out-of-sequence measurements (Bar-Shalom, IEEE TAES 2002, UNVERIFIED: CT's ask-time matching already is one, a
related-work sentence); zipper merging and gap acceptance (nothing measured merges; A3 covers yielding); deixis (= A5);
empiric therapy while cultures grow (explains CT2m's loss as a narrow guess, but the hedge and the watching are robotics
ideas 7 and 5); pecking order (= A4).

## 3. Limitations a CoRL / HRI reviewer would raise

1. **The roster is rooted in identities.** In the bystander scenes the maintained roster is empty while a line stands
   in up to 0.48 of post-answer resolves, and it never recovers (K3). CT3's bystander result (+0.018 strict share,
   near the floor) partly swaps capture for that.
2. **The scorer treats capture and cut-in as the same failure.** Rankings flip at modest harm ratios (K4: Gl beats
   CT3 above 3.4 for DeepSeek). Report the ratio, or measure it with people.
3. **The sim's people make motion evidence too easy.**
   - Step-ups are simultaneous (K5).
   - Bystanders only fidget in place. There are no visitors, staff, or people who join and then give up.
   - Nobody reacts to the robot.
   - "Truth" is the sim's roster, which counts people still walking back.
4. **Bystander specificity rests on one placement per scene** (the lone man 0.92 m off the end, facing away), with
   28-60 answers. Sonnet's 13/28 could partly be redraw noise.
5. **Prompt confound.** Facing reaches the model as a bare angle (K6). A reviewer will ask whether the prompt or the
   model misses that the lone man faces away, and idea 2's facing condition answers it.
6. **Role accuracy and placement are mixed in one share.** Report per-person role Se/Sp (K1) apart from pose quality.
7. **Generality is argued, not tested, on moving groups and audiences.**

## 4. Top pick (five lines)

Idea 1, "views, not rosters", because it explains the biggest loss left in our newest scenes with zero-call numbers:
the roster is empty while a line stands in 0.33-0.48 of DeepSeek's resolves, with 0 recoveries in 109 arm-cells.
It shows "the line ran out" and "a bystander was taken in" are the two classic anomalies of maintaining a recursive view.
It gives the community a reusable split (anchor, relation, witnesses) and a per-tick correctness metric.
It can be tested this week at zero calls by cache-only reruns, with clear kill bars (bystander-last, cut-ins).
