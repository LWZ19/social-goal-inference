# Ideas round 2, lens: sociology, proxemics and HRI on social spaces (2 Nov 2026)

- `cd moving && python3 -I $S/ctx_tally.py runs`: the first answer's "people matter" verdict, by model and scene.
- `.venv/bin/python -I $S/roster_displays.py results/2026-11-01-push/moving/runs sim/scenarios` (repo
  root): who each bystander-scene answer named, and how it described their facing.
- `cd moving && python3 -I $S/contingency.py runs dsv41 cafe_line2_bystanders 15 44` (also sweet_spot 15-44, and
  Sonnet room 10-19): a "who moves after whom" graph with no target and no names.
- `cd moving && python3 -I $S/capture_runs.py runs`: bystander capture split into runs.

Tags: **VERIFIED** = I read the abstract or the paper's own page today. **UNVERIFIED** = search snippet, a secondary
source or memory. Books (Goffman, Hall, Kendon, Lewis) I know only through secondary sources here, so they are
UNVERIFIED even though they are standard.

## 0. Five zero-call findings that the ideas rest on

- **Z1. The models see the "not taking part" display and still name the person by nearness.** This is the lone man
  in the bystander scenes: 0.9 m past the line's end, off to the side, facing away.
  - Facing is judged as the reported heading against the bearing to the front place (slot 0).
  - DeepSeek, 60 constraint answers (pilot plus confirmation): it names him in 11.
    - Its people step reports him as less turned to the front than every listed line member in 48 of 60 answers.
    - **8 of the 11 answers that name him are among those 48.**
    - Named rate: 9 of 50 when reported as facing away (more than 60 deg off), 2 of 10 when reported as facing the
      front. Facing makes no difference to DeepSeek.
  - Sonnet, 29 answers: it names him in 13 (cafe 1 of 10, sweet_spot 7 of 9, room 5 of 10). Its facing is accurate
    (median error 7 deg; DeepSeek 27 deg).
    - Named rate: 8 of 21 when reported as facing away, 5 of 5 when reported as facing the front.
    - **7 of the 13 namings come where its own list shows him as the least oriented person near the line.**
  - So the information was in the prompt, and the role was assigned by position.
- **Z2. The formation is not seen in one picture, and the error runs both ways.**
  - DeepSeek answers "people do not matter" in **6 of 6** room bystander cells (seeds 0, 10-14), and the robot drives
    through the line. The same room and line got "yes" in 3 of 3 on the original scene and 4 of 5 on the slots
    variant. The bystander scene adds three people and turns one member 90 deg.
  - Sonnet says "yes" in 10 of 10, but once lists all seven people as queuing, three of whom it describes in the same
    answer as "facing one another in a conversation" (`sonnet55_low/room_coffee_cart_bystanders/seed12`).
- **Z3. Half a "with".**
  - Of DeepSeek's 4 answers that name a member of the talking pair (all in sweet_spot), 3 name only one of the two.
  - 2 of the 4 also list both pair members in a `keep_out_of_shared_space`.
  - The pair is treated as two people, not as one participation unit.
- **Z4. Capture comes in short runs, which one human repair would end.** Runs are consecutive re-solves whose last is
  a bystander. The bound assumes the bystander repairs the misreading T s after a run starts, the robot accepts it,
  and that bystander's later runs vanish.

  | Arm | Captured | Runs, cells | Median run | Left, repair at 5 s | Left, repair at 10 s |
  |---|---|---|---|---|---|
  | DeepSeek CT2 | 1,005 s | 74 runs, 28 cells | 8.7 s | 136 s | 233 s |
  | DeepSeek CT3 | 203 s | 26 runs, 9 cells | 5.3 s | 39 s | 58 s |
  | Sonnet CT2 | 938 s | 65 runs, 22 cells | 7.2 s | 91 s | 155 s |
  | Sonnet CT3 | 322 s | 41 runs, 13 cells | 5.3 s | 55 s | 91 s |

  - With a 10 s repair, CT2's capture (233 s) is about CT3's capture with no repair (203 s).
- **Z5. A target-free "who moves after whom" graph separates members from bystanders, but not from walkers.**
  - Edge i -> j: j moves 0.3 m or more within 5 s after i starts a move, within 2 m of i, in the same direction or
    into the place i left.
  - Share with at least one edge:

    | People | cafe | sweet_spot | room (Sonnet) |
    |---|---|---|---|
    | Line members, of those who moved | 47 of 56 | 44 of 46 | 27 of 27 |
    | Bystanders, of those present | 3 of 84 | 7 of 69 | 1 of 27 |
    | Walkers and served people, of those who moved | 101 of 112 | 89 of 92 | 17 of 35 |

  - The commonest edge is a served person (counted as "other") leaving and the next member stepping into their place:
    59, 55 and 11 edges.
  - So edges alone are not membership; the shape of the graph is needed (idea C).

## 1. Intuitions from the field

1. **A queue is an achieved order, seen in its progression, not in a snapshot.**
   - Garfinkel and Livingston (Visual Studies 18(1), 2003): in "formatted queues" people "incessantly" position
     themselves to exhibit an order of service. The order's properties (orientation, position, distance, direction) are
     produced and witnessable in conduct. VERIFIED (abstract).
   - Lee and Watson's 1993 Plan Urbain report on interaction in urban public space is the origin of the formatted-queue
     line. UNVERIFIED (citations only).
   - Mann (AJS 75(3), 1969) treats the queue as a "miniature social system" with place-keeping and push-in defence.
     UNVERIFIED (citation and secondary summaries).
   - Bears on: Z1, Z2, and "nobody joins without moving" (CT3).
2. **Participation status is relative to a focus, and the units are singles and "withs".**
   - Goffman separates ratified participants from unratified bystanders (overhearers, eavesdroppers): "Footing", Forms
     of Talk, 1981.
   - Relations in Public (1971) treats the "with" as a participation unit as basic as the single, and lists "the turn"
     (the queue at a ticket window) among the territories of the self, beside personal space and the stall.
   - Kendon: an F-formation is a maintained arrangement around an o-space to which members have equal, direct and
     exclusive access. Members keep it up by continual mutual adjustment.
   - All UNVERIFIED (secondary reviews: Blumer 1972, Gergen 1973; the emcawiki and Setti et al. summaries).
   - Bears on: Z1, Z3, and the vocabulary's missing "not taking part" kind (BYSTANDERS_FROZEN F2).
3. **Positions outlive their incumbents.**
   - Sacks's membership categories tie activities to positions in a device: "category-bound activities". Seeing the
     activity licenses seeing the category. UNVERIFIED (emcawiki and secondary tutorials).
   - A formatted queue has positions (front or service place, next, last, entry) that persons pass through.
   - Bears on: "the line ran out" (17% of CT time, FAILURES E). The answer named incumbents, and when they were all
     served, nothing was left of the line.
4. **Membership is negotiated and repaired, and errors are not symmetric.**
   - Conversation analysis: repair is built into interaction, with a preference for self-correction. Schegloff,
     Jefferson and Sacks, Language 53(2), 1977. UNVERIFIED (citation and summaries).
   - In queues, "are you in line?" and "who's last?" are routine repairs. The Soviet "who's last?" practice makes the
     order a chain of person-to-person links, not a spatial line. UNVERIFIED (a blog and novels; I found no
     ethnography that treats it as a category).
   - Milgram, Liberty, Toledo and Wackenhut (JPSP 51(4), 1986): objections to a line intruder come mostly from the
     person just behind. UNVERIFIED (secondary).
   - Standing behind a non-member is a cheap error that the non-member corrects; cutting in is sanctioned.
5. **Conventions show themselves as regularities, and leaders show themselves by who moves first.**
   - Lewis (Convention, 1969): a convention is a regularity, common knowledge, that solves a recurring coordination
     problem. UNVERIFIED (secondary).
   - Collective motion: pigeon flocks carry a stable hierarchy, read from directional choices and response timing.
     Nagy, Akos, Biro and Vicsek, Nature 464, 2010. VERIFIED (abstract). The time-delayed correlation method is
     UNVERIFIED (snippet).
   - Pedestrians: Kjaergaard et al. (PerCom 2013) detect following from time-lagged signal similarity, then leaders by
     link analysis on the directed graph. VERIFIED (abstract via OpenAlex).

## 2. The generic representation: a participation frame

A social situation is a set of **formations**. Each formation has:
- a **focus**: the service place, the o-space, the stage or screen, or the passage;
- a **format**: positions, an entry point and a progression;
- **incumbents**: each person, or each with, holds a status towards the focus (a participant in position p, a
  bystander, passing, or unclear), and the status is licensed by **displays**:
  - in a picture: orientation of the front towards the focus or the person ahead, and spacing in the format;
  - over time: co-regulated motion, i.e. moving when the formation moves.

| | Queue | Conversation | Audience | Doorway or lift |
|---|---|---|---|---|
| Focus | service place | o-space (mutual) | stage or screen | door or passage |
| Positions | front (use space), sequence, end | arc of the p-space | rows facing the focus | beside the door, exit lane kept clear |
| Entry | behind the last | a gap opened in the ring | the back edge, out of view lines | the side cluster |
| Progression | advance on each service | re-close when someone leaves or joins | little | batch on opening (weak arrival order) |
| Display of taking part | faces the focus or the back ahead; steps up | mutual orientation; compensating shifts | co-orientation; stillness | faces the door, still nearby |
| Turnover of incumbents | one service period | minutes | minutes | seconds |
| Our vocabulary | `come_after`, `face` | `keep_out_of_shared_space`, `visible_to` | `keep_out_of_view` | none for a lane |

**What it explains.**
- **Bystanders taken as members.** Three layers assign status by nearness instead of by display:
  - the model (Z1);
  - the solver's `_turn`: any still person within 1 m behind the last;
  - the 1 m chain G: 51 of 60 cells captured.

  The vocabulary cannot even state "not taking part" (F2). CT3's motion gate is the first display-based layer, and
  all of CT3's leftover capture comes from the model's layer.
- **"The line ran out".** The spec was bound to incumbents (come after 4, 5, 6), and incumbents turn over once per
  service. CT2's vacated place, re-admission and path memory are three ad hoc ways of binding to the format.
- **Room, "people do not matter" (Z2), and "all seven queue".** Both are gestalt errors on a picture in which the
  formation is not witnessable (a turned member and milling neighbours).
- **R's marks beside the line** (fa_off and fa_counter, 0.20 of failure time): a point with no format position.

**What it predicts that we have not run.** Incumbent-bound specs will fail wherever turnover is faster than the
answer's life:
- doorways and lifts, quickly;
- a conversation where one member is replaced (the o-space is computed from the named members who remain);
- an audience whose front row changes.

## 3. Ideas

### A. Witness before judging that there is no order
- **One sentence.** Before acting on "people do not matter", check whether others' conduct shows a convention in
  force. If it does, the claim is falsified: re-ask with the observation stated, or go to the constraint call anyway.
- **Failure attacked.**
  - DeepSeek room bystanders: 6 of 6 standard goals, and the robot crosses the line (SUMMARY section 5, Z2).
  - sweet_spot bystanders: 4 of 36 standard and 4 no-target answers (`ctx_tally.py`).
  - Nothing that re-solves can fix these.
- **Borrowed intuition.** Lewis (a convention is a regularity in a population), and Garfinkel and Livingston (order
  is witnessable in conduct).
- **Why it is a new way of thinking.** The model's social reading is a hypothesis that the observed social order can
  falsify at the level of the situation (is there a formation here?), not only at the level of who is in it. A
  reader can reuse "absence of order needs evidence too" for any VLM gate that decides whether people matter.
- **Closest prior work and what is new.**
  - Robotics BOLD A (a belief over line hypotheses) and LANDSCAPE B1 (expiry conditions) monitor a given answer. HUMA
    (arXiv 2607.10991) gates the VLM by proximity. VERIFIED in LANDSCAPE.
  - New: the gate's own negative verdict is checked against a generic formation detector, i.e. two or more still
    people co-oriented to a common focus near the target. The detector names no situation.
- **Hypothesis and tests.**
  - The detector (E1's RT-DETRv2 + ViTPose, shoulder line for facing) fires on all 6 room bystander frames, on 90% or
    more of the line frames, and on at most 10% of development still scenes whose reference says people do not
    matter.
  - Zero calls: run the detector on logged `rgb.png` first frames plus the development set.
  - Then re-ask DeepSeek on the 6 room and 4 sweet_spot standard frames with the observation in the context call:
    10 cells x 4 calls = 40 free calls.
  - **Kill:** false alarms above 20% on no-people-matter scenes, or DeepSeek still says "no" in half the re-asks.
- **Risks.**
  - The scripted line is unusually regular.
- **Scores:** payoff 3, uncertainty 3, cost 1 day.

### B. An accountable roster: participation status licensed by a display (builds on interface idea 3)
- **One sentence.** The constraints call states, for each listed person, or each with, a status towards the target:
  - taking part (waiting a turn, using it, watching it);
  - with #k in something else;
  - passing;
  - unclear.

  Each status cites the display that licenses it, from a closed list: front towards the target or towards the back
  of someone who does; standing in the others' arrangement; turned to another person; facing away; walking.
  Constraints are then compiled from the statuses. "Unclear" goes to motion evidence (CT3's follow check), and "not
  taking part" becomes a solver exemption, which closes F2's gap.
- **Failure attacked.** All of CT3's leftover capture comes from bystanders the model named (203 s DeepSeek, 322 s
  Sonnet; `bystanders_post/README.md`). In those namings the display was already perceived (Z1: 8 of 11, 7 of 13).
  Also Z3 (half a with, both "with" and "waiting").
- **Borrowed intuition.** Goffman's participation framework and withs, and the ethnomethodological point that
  membership is accountable, i.e. shown in conduct.
- **Why it is a new way of thinking.** It moves the model's task from deciding who is in the group (where it reasons
  by nearness, Z1) to reading what each person is doing towards the focus, where VLMs are strong. The status is
  licensed, not asserted. The same roster serves queues, conversations, audiences and doorways.
- **Closest prior work and what is new.**
  - Interface idea 3 (one exclusive tag per person, compiled) is the nearest; this adds licensing displays, withs as
    units, a three-valued status with "unclear" handed to motion, and the negative kind.
  - GCFF (Setti et al., PLoS ONE 2015, position and orientation only; VERIFIED abstract) is a geometric F-formation
    detector, not request-conditioned.
  - "Where Should I Join?" ranks member subsets with a trained model (VERIFIED in LANDSCAPE).
- **Hypothesis and tests.**
  - With the roster, DeepSeek names a bystander in at most 4 of 60 answers (now 11) and Sonnet in at most 6 of 29
    (now 13). Member recall on the 51 original first frames is not lower.
  - Development still scenes must stay at 84-93% (shared prompt).
  - **Kill:** bystander naming does not halve on two models, or member recall drops. That would turn mild capture into
    cut-ins (see D).
- **Risks.**
  - It is a prompt and schema change, so it must help most models.
  - Display lists can drift towards queue words.
  - It only works if the people step's facing is usable (DeepSeek's is 27 deg off, and members' facing is noisy).
- **Scores:** payoff 4, uncertainty 3, cost 1.5 days.

### C. [BOLD] Formations as "who moves after whom" graphs
- **One sentence.** From tracks alone, build a directed graph: j moves after i, the same way or into i's place, within
  a few seconds. Read membership, order and formation type from its shape:
  - a chain ending where people leave the focus is a turn-taking line;
  - mutual compensating edges are an F-formation;
  - no mutual edges but a common orientation response to the focus is an audience;
  - isolated nodes are bystanders.
- **Failures attacked.**
  - CT3's leftover capture: a named person with no incoming edge across two service events is dropped, whatever the
    model said.
  - The confirmation's sweet_spot s30 false drop: CT3 measures advances towards a wrong target point. Edges need no
    target.
  - "The line ran out": members are re-derived from the chain, not from names.
- **Borrowed intuition.** Leadership from delayed response in flocks (Nagy et al. 2010) and in pedestrians
  (Kjaergaard et al. 2013), and Kendon's formations kept up by co-adjustment.
- **Why it is a new way of thinking.** Membership becomes a property of contingency, not of position or naming. One
  observation model then covers formation types that robotics treats with separate detectors: F-formation detectors
  versus queue code.
- **Closest prior work and what is new.**
  - Group detection from trajectories groups people who walk together, by proximity and velocity: Ge, Collins and
    Ruback (TPAMI 2012, UNVERIFIED snippet) and Yamaguchi et al. (CVPR 2011, UNVERIFIED snippet).
  - Stationary crowd groups are modelled as regions (Yi, Li and Wang, CVPR 2015; UNVERIFIED snippet).
  - Queue detection is mostly patents that cluster trajectories or fit lines (UNVERIFIED snippets).
  - I found no work that infers stop-and-go formation membership, or formation type, from lagged contingency, and none
    in robot goal inference.
- **Hypothesis and tests.**
  - The zero-call part is done (Z5): members 84-100%, bystanders 4-10%, walkers mostly edged, so structure is needed.
  - Next, zero calls: a CT3 variant whose follow check is "an incoming edge from the person ahead, in any direction",
    run on the 60 confirmation cells from cache.
    - Prediction: capture at most CT3's, 0 false drops (s30 fixed), no harm on the originals.
  - The bold part needs real data, because our scripted scenes would be circular: score edge-based grouping against
    annotated F-formations in public mingling datasets with tracks (e.g. SALSA, MatchNMingle; UNVERIFIED, memory).
  - **Kill:** the graph variant loses to CT3 in capture or adds false drops, or on real data edge-based groups score
    below GCFF on F1.
- **Risks.**
  - Sim bystanders fidget on independent timers, so separation is easy here.
  - Real bystanders yield as the line moves: contingent but sideways, which needs a direction test.
  - Slow lines give few events.
- **Scores:** payoff 4, uncertainty 4, cost 2-3 days.

### D. [BOLD] Membership is negotiated: design and score for repair ("I'm not in line")
- **One sentence.** Three parts:
  - Treat other people as co-authors of the robot's reading.
  - Under ambiguity, choose the error people repair cheaply. Stand where the reading is legible, behind the ambiguous
    person on the axis, facing them, rather than the sanctionable one (cut in).
- **Failure attacked.** Bystander capture, and how we score it. Z4: one repair at 10 s would cut CT3's capture by
  71-72% and CT2's by 77-83%, making CT2 about as good as CT3 without repair.
- **Why it is a new way of thinking.** Social goal inference is usually posed as the robot inferring alone.
  Ethnomethodology says order is produced jointly. The research object becomes the loop:
  - legible commitment;
  - human repair;
  - an update.

  It also predicts that benchmarks with non-reactive humans overstate repairable errors and hide the cost of
  sanctionable ones.
- **Closest prior work and what is new.**
  - Interface idea 5 asks the VLM a one-bit question on events. Here the target is chosen by pivotality, and the human
    is the source.
  - "Social common sense modeling of a spread-out queue in public space for a service robot" (HRI 2017 companion, DOI
    10.1145/3029798.3038345): a service robot models a loose queue's unspoken order and advises a person who cuts in.
    UNVERIFIED (snippet).
  - Legibility of motion (Dragan et al.): UNVERIFIED (memory).
  - I found no robot that asks or reads "are you in line?" (search came up empty).
- **Hypothesis and tests.**
  - The zero-call bound is done (Z4).
  - Then add a scripted reaction to the sim: a non-member with the robot within 1.2 m behind, facing them, for 3-10 s
    turns to the robot and steps 0.7 m aside. Add a resolver rule that reads this, and run CT2 and CT3 with and
    without the rule on cached answers (0 calls).
    - Prediction: CT3 plus the reading leaves at most 30% of CT3's capture with no added cut-ins, and CT2 plus the
      reading comes within 1.3x of CT3's capture.
  - Pivotality: re-solve each first frame with each person's come_after membership flipped (0 calls).
    - Prediction: at most 2 pivotal people per frame, and a named bystander pivotal in at least 80% of namings.
  - **Kill:** the robot's ambiguous stops are seldom in the repair-eliciting configuration (behind and facing), or
    pivotal sets are large.
- **Risks.**
  - The reaction is our script, so the test measures the robot's use of repair, not whether people repair. That needs
    a small human study, the natural HRI follow-up.
  - Reframing capture as mild weakens part of the CT3 story; say so openly.
- **Scores:** payoff 5, uncertainty 4, cost 2 days.

### E. Bind the answer to the format, not the incumbents: a turnover law for specs (builds on LANDSCAPE B2, robotics 1 and 3)
- **One sentence.** Each constraint kind binds to a place in the format that persists:
  - the turn to the end, then the predecessor, then the front;
  - the shared space to the o-space as a place;
  - the view constraint to the focus's view cone.

  Current occupants are re-bound each tick. An identity-bound spec is predicted to expire after about one incumbent
  turnover.
- **Failure attacked.**
  - "The line ran out": 0.17 of CT time, 29 of 51 episodes (FAILURES E). CT2 fixed it with three ablation-redundant
    rules.
  - The cross-situation prediction in section 2.
- **Borrowed intuition.** Formatted queues and category-bound positions (Sacks), and Goffman's "turn" as a
  territory, distinct from the person.
- **Why it is a new way of thinking.**
  - It replaces CT2's queue rules with one principle that says which situations need it (fast turnover) and which do
    not (slow).
  - It gives a reusable law: the spec's life is about the incumbents' turnover. That is a principled expiry
    condition, a concrete form of LANDSCAPE B1.
- **Closest prior work.**
  - B2 (role predicates over tracks) re-evaluates who fits a description, while E keeps a format with positions.
  - Robotics idea 3 (the line as a place) is the queue case.
  - "Where Should I Join?" re-runs a trained model with 3-frame membership persistence (VERIFIED in LANDSCAPE).
  - None states turnover as the expiry.
- **Hypothesis and tests.**
  - Zero calls: drive `sgi.resolve.Resolver` and `sgi.solver.solve` with synthetic tracks (no Habitat), using
    reference answers.
    - A 4-person F-formation where one member is replaced, then a second.
    - An audience whose front row is replaced.
    - A line, as a control.
  - Prediction: the identity-bound CT puts the goal inside the current o-space or a view line after the second swap
    in at least half of the layouts. The format-bound variant does not, and matches CT2 on the line.
  - **Kill:** CT holds in the group and audience cases, i.e. turnover does not hurt where we predicted.
- **Risks.**
  - "Format" detection is the hard part; it is easy here because the reference answers give it.
  - It must stay free of situation names: the binding is per kind, not per scene.
- **Scores:** payoff 4, uncertainty 3, cost 2 days.

## 4. Limitations a CoRL or HRI reviewer would raise

1. **Nobody in the sim reacts, so nothing is negotiated.**
   - Capture is measured in a world where no one ever says "I'm not in line" (Z4).
   - Motion evidence is easy, because bystanders never respond to the line. Real bystanders yield, and withs move
     together; our pair fidgets independently.
2. **One bystander geometry.** A lone man facing away and a talking pair make up the whole test. Missing hard cases:
   - someone facing the counter but not waiting (reading the menu): display yes, conduct no;
   - a with in the line (two bodies, one turn);
   - a place-holder for a friend (Mann's markers);
   - staff at the front (sweet_spot has one behind the counter, never examined).
4. **The vocabulary has no non-participation status, no markers and no passage lane.** So the claim that it
   generalises to doorways and lifts is not supported by the kinds. The elevator norms work turns sociological norms
   into feature maps for waiting poses (Racca et al., RO-MAN 2025; VERIFIED abstract).
5. **One culture of queueing**: one FIFO line, fixed service. Spread-out queues (HRI 2017 above), weak first-come
   order in lift lobbies (UNVERIFIED), and relational "who's last?" queues are absent.
6. **"Knows who queues" is not ordered by model strength** (Sonnet names bystanders more than DeepSeek). Z1 suggests
   both use nearness as a heuristic. Avoid any claim that stronger VLMs understand social order.
7. **Snapshot cues are noisy in our renders.** Members' reported facing is a median 61 deg (DeepSeek) and 85 deg
   (Sonnet) off the bearing to the front. Some of that may be avatars still settling at t = 2 s, which I did not check.
8. **The robot never speaks.** A reviewer will ask why it doesn't simply ask. Idea D answers it with the pivotal
   person, but nothing is tested.

## 5. Top pick (five lines)

1. Adopt the participation frame (focus, format positions, incumbents whose status is licensed by displays) as the
   paper's generic representation.
2. It explains both named failures: bystanders are status by nearness, and "running out" is a spec bound to
   incumbents.
3. Test it first through **B, the accountable roster**. The models already perceive the lone man's "not taking part"
   display in 8 of 11 and 7 of 13 of their wrong namings, and CT3 cannot touch those.
5. Write up D (repair) as the bold reframing even if we cannot test it beyond scripted reactions.
