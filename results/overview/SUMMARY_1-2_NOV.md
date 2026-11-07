# Research push, 1-2 November 2026: what we found

## 1. Moving people: ask once, then re-solve as people move

Setup: Spot in Habitat 3.0 joins a waiting line that moves (front person served every 12-25 s, walks off, rejoins).
The constraint method's answer names people ("come after 4, 5, 6"); between answers the robot re-solves on tracked
people with no model call (CT), instead of asking once (C1/R1) or asking again every 10 s (Cq/Rq).

### Headline (fresh simulator seeds, nothing tuned on them; share of time correctly in line, strict scoring)

| Comparison (paired by scene and seed) | Sonnet 5.5, 50 cells | DeepSeek V4.1, 25 cells |
|---|---|---|
| **CT2 against the marked point R1 (the proposal's baseline)** | **43 wins / 2 losses / 5 ties, +0.27** | **19 / 3 / 3, +0.23** |
| CT2 against CT v1 (primary, CT2_FROZEN) | 32 / 8 / 10, +0.08, p < 0.001 | 18 / 3 / 4, +0.08, p = 0.001 |
| Detector tracking (CTd) against true tracks (CT2) | 5 / 6 / 4, -0.02 (no measurable loss) | 5 / 4 / 6, -0.02 |
| Move while thinking (CT2m) against CT2 | 12 / 27 / 11 (no gain; see section 4) | 7 / 10 / 8 |

Strict share in line: Sonnet CT2 0.31, CT 0.22, R1 0.03, C1 0.02; DeepSeek CT2 0.27, CT 0.20, R1 0.04.
Keep-up (back in line within 10 s of a service, no cut-in): Sonnet CT2 62/97, R1 4/12. The same ordering holds under
the frozen primary scoring; the strict scoring adds a side check and a 0.6 m first-in-line tolerance (both found to
flatter R in the failure analysis, frozen before the fresh seeds).

### What CT2 is (CT2_FROZEN.md) and why
parts off, CT2 reproduces CT (-0.009 / -0.011), so the gain lies in the named parts. Removing the forward ratchet
alone removes most of it (Sonnet -0.077, 12/30/7; DeepSeek -0.063, 4/13/8). Removing re-admission, or re-admission

**How to read CT2 against CT.** The ratchet is a goal-following fix: CT's symmetric 0.3 m / 25 deg replace threshold
left the robot behind as the line moved up in small steps, and the 1.0 m standoff plus that threshold produced near
misses past the scored 1.2 m band; an asymmetric rule (follow forward steps over 0.08 m at once) removes them. That is
control engineering found by failure analysis, not part of the method's idea, and its benefit shows as staying
inside a band our scorer defines. Measured (post hoc, `results/moving-lines/questions/ratchet_band_out.txt`):
- The ratchet keeps the robot about 0.1-0.3 m closer behind the person ahead. While joined, 30% of CT's gaps fall
  in 1.2-1.5 m, against 9% of CT2's; the gap's 75th percentile is 1.24 m against 1.14 m.
- With the band top at 1.5 m instead of 1.2 m, CT2 against CT falls to +0.008 (Sonnet) and +0.033 (DeepSeek), and
  the ratchet ablation's effect disappears.

So CT2 over CT is a 1.2 m band-edge effect. Whether a 1.2-1.5 m gap reads as "in line" to people is a social-distance
question these runs can't answer. The research claim does not depend on it: **CT v1 already beats the marked point
on the fresh seeds** (Sonnet 40/3/7, +0.19, p < 0.001; DeepSeek 11/3/11, +0.16, p = 0.06).

Break-it cases (Sonnet, n = 3 each): CT2 holds the bent line where CT fails (0.46 against 0.11); the parts were not
ablated there; path memory is the only part designed for it and is inert on the straight evaluation lines. A member
stepping out of line: CT2 0 cut-in samples against CT 5 (the frozen prediction holds).

### Controls on seeds 0-2 (DeepSeek, 15 episodes each; share under the frozen primary scoring)

- Tracking a point without roles fails: R's marked point moved with the nearest person (RTw) scores 0.08 against CT's
  0.18 and its goal leaves the line in 11 of 15 episodes (RT, the anchor-frame variant, 12; CT 7): it follows a served
  person away. Roles carry the gain.
- The navigation map is not the source: CT on the first picture's floor (CTv) 0.17 against CT 0.18.
- Re-asking is not free stability: R re-asked every 10 s or as fast as possible scores below R asked once (0.14-0.16
  against 0.20), at 20-30 calls an episode against CT's 4.
## 2. Efficiency: one model call fewer, same accuracy (E1, analysis's design)

- **E1b (static development scenes):** a pretrained detector (RT-DETRv2 + ViTPose, ~60 ms) in place of the VLM's
  people call. Sonnet 124 -> **132** of 143, DeepSeek 127 -> **127** of 151 (both non-inferior to the redraw floor);
  model time for the people-and-constraints step 9.0 -> 5.4 s (Sonnet), 12.7 -> 7.6 s (DeepSeek). Qwen 3.8:
  131 = 131 of 144, non-inferior, about 21 s of people call saved per record. GLM partial (105 records; the run stopped
  at 317 of 450 calls): 63 -> 73, gain 11 / harm 1, non-inferior. Across all four models the detector matches or
  beats the model's own people finding and removes the whole people call. Caveat: these are development scenes, and
  the detector setting (A1) was chosen on a split of them, so this is not a held-out test. On unseen layouts A1 had
  0.24 false people per image.
- **E1a:** on Habitat frames the detector finds 75/75 visible people, 0.08 false per image, 0.08 m error.
- **CTe (moving lines):** the full efficient pipeline, detector people for the first answer and detector tracking.
  First answer 10.6 s against 13.0 s (about 2.2 s faster after counting the detector), first in line 1.5 s earlier,
  quality the same as CTd within the redraw spread (9/4/2). One observed side effect: given detector people (positions
  only, no descriptions), Sonnet's constraints include keep-out-of-shared-space in 12 of 15 answers, against a 13 of 44
  base rate with its own people (stratified p 0.0009). Its effect on behaviour is not shown.

## 3. Failure analysis (analysis/FAILURES.md, 396 episodes, 31,554 s out of line)

Waiting for the first answer and walking there: 0.39 of failure time; a wrong or incomplete first answer: 0.39;
answers going stale: 0.10; re-solve rules: 0.07; sim and scoring: 0.01. x-test-3's lower score is all latency. Our
moving scenes are too easy to show the model's role knowledge: a geometry-only rule picks the right people in
15/15, 15/15 and 13/13 first frames, so the claim is maintenance (keep-up, cost, staying on the line), not that the
model knows who queues. Distractor scenes (a chatting pair by the counter, two lines) would make that testable.

## 4. What did not work, and why

- **Move while thinking (CT2m, Rm):** walk towards the target point once the plan call names it. No gain for any
  model, x-test-3 included. Diagnosis: the target point is the counter, the place is the line's back; the robot
  reaches the interim stop early and waits there, so the gain is a fixed ~1 m, not a share of the waiting time.
  Where the straight ray points the wrong way, the robot reaches the line later: beside the middle of a 6-person line,
  or round the wrong side of an obstacle. Sonnet was later in 8 of 10 and 8 of 8 cells in those two scenes, and earlier
  in 6 of 9 where the line's back lies on the ray.
  Future work (not run): speculative execution on a situation-agnostic prior; its ceiling from the logs is the
  plan-to-answer time (Sonnet 9.5 s, x-test-3 35.5 s).
- **Re-solving only on settled tracks (CTs):** no effect (5/7/3, 15 episodes, +0.01).
  215 -> 237, end success 155 -> 177, layout-level p = 0.011); nothing beyond run-to-run noise for DeepSeek, Qwen,

## 5. Beating CT2: bystanders and motion evidence (CT3), 2 Nov afternoon
the model's roles (geometry found the right people 15/15), so **bystander scenes** were built

**Found before any cell (zero calls):** CT2's solver takes any unnamed, still person standing just behind the last as
the new last (`solver._turn`), so CT2 can throw away a roster the model got right.

**CT3 = CT2 + motion evidence** (`sgi.resolve`, `co_motion_s` = 5): a named person who does not follow the person
ahead's advance within 5 s is dropped; nobody joins without moving (walked 1 m, or came a step nearer, within 20 s),
and those who may not join are exempt from the solver's place-in-turn rule. **Gl** = the geometry chain (no model
roles), timing-matched. Zero extra calls.

|---|---|---|---|
| Constraint answers naming a bystander as waiting | 1 / 9 | 10 / 51 | 13 / 29 |
| Time behind a bystander ("capture"), CT2 -> CT3 | 208 -> 9 s | 1005 -> 203 s | 938 -> 322 s |
| Cells captured, CT2 -> CT3 | 6 -> 1 | 26 -> 9 | 20 -> 13 |
| CT3 vs CT2, strict share (W/L/T) | 5/2/8, p 0.45 | **17/6/37, +0.018, p 0.035** | 11/5/14, p 0.21 (reported) |
| CT3 vs Gl on constraint-answer cells (roles from the model vs geometry) | 7/0/2 | 24/5/22, p 0.001 | 17/1/11, p < 0.001 |

- **Judged:** the frozen confirmation (`BYSTANDERS_CONFIRM_FROZEN.md`, DeepSeek, cafe and sweet_spot, seeds 15-44)
  passes both bars, so **"CT3 beats CT2 on bystander scenes" is claimed** for DeepSeek on the cafe and sweet_spot
  bystander scenes, as a small effect (+0.018 strict share, wins concentrated where CT2 was captured: 14/3/9, against
  3/3/28 elsewhere). It does not replicate on Sonnet by the frozen bars. On Sonnet (`BYSTANDERS_SONNET_ADDENDUM.md`) the
  capture bars and the roster bar fail; the roles bar passes. No harm on the original scenes (Sonnet 30 cells +0.015,
  DeepSeek 15 cells +0.007, no false drops).
- **How it works, on both models:** the join gate removes the solver's own capture completely (Sonnet: 0 s over its 17
  cells where the answer named no bystander; DeepSeek: none of its remaining 203 s); the follow check handles the
  model's own roster errors (it halves their capture; in Sonnet's 13 named-bystander cells, CT2's share is 0.000 and
  CT3's 0.110) but can act only when the line advances. All of CT3's remaining capture comes from bystanders the model
  named.
- **The model's roles matter:** the geometry chain takes bystanders into the line in 14/15 (pilot) and is captured in
  51/60 cells (confirmation); CT3 beats it on both models. "Knows who queues" is model-dependent, and not in order of
  model strength (Sonnet names bystanders more often than DeepSeek).
  substantive result is capture. In room with bystanders DeepSeek answered "people do not matter" 5/5 and drove
  through the line, a model-side failure nothing re-solved can fix; 5/60 confirmation answers found no target. Two
  With bystanders, CT2's lead over the marked point disappears for Sonnet (4/3/13): the headline in section 1 is

## 6. Still scenes: C's misses are a depth error; a person snap (2 Nov evening)

- **Diagnosis (zero calls, `results/still-scenes/group-of-four-case/README.md`):** on the development scenes C is right 84-93%, and its misses
  concentrate in one scene, a 4-person conversation group (Sonnet 7 of 10 misses, Qwen 8 of 13, DeepSeek 9 of 24). All
  six models click the right person, but the 5x5 depth lift at her torso reads the arm of a nearer person, so the target
  lands on him (23 of 28 misses). A majority vote of 3 helps little (Sonnet 0.61 -> 0.67), because the clicks scatter
  across the arm's edge. A detector "person snap" (take the detected person whose box holds the click, only when the
  depth lift is more than 0.5 m from them) fixed 24 and broke 0 over 1,512 records, post hoc, but 25 of the 27 target
  fixes are in that one scene.
- **Frozen test on new scenes** (`results/still-scenes/group-of-four-case/SNAP_FROZEN.md` and amendments 1-3): 24 new realistic scenes (generator seeds
  400-449, group bearings patched to make occlusion); primary set = 8 layouts with an occluder's edge within half a body
  width of the target's chest point, chosen from render geometry before any call; plan stage only, Qwen, GLM and
  DeepSeek x 3 trials, 432 free calls.
- **Judged:** H1 fails, so "not needed" and H2 is not judged: the depth lift is right on 86.8% of the 8 primary
  layouts against 95.4% on unhidden targets, a gap of 8.6 points against the 10-point bar. The claim is not made.
- **Descriptive (post hoc):** the failure needs the occluder's edge essentially at the chest point. Where it was
  (gap 0.04-0.05 body half-widths, 2 layouts, as in the development case), the lift was right 26/36 and the snap
  35/36 (9 of 10 misses fixed); at gaps 0.19-0.41 the lift was as good as on unhidden targets (99/108). The snap fired
  12 times in 432 records (11 fixes, 1 no change, 0 breaks); on the development records 32 times in 1,512 (24 fixed,
  0 broken). Size: +3.7 points over all new person-target records (300 -> 312 of 324), +7.6 on the primary set.
- **Reading:** a person snap at the lift step fixed the occluder-at-chest failure where it occurred (9 of 10 misses on
  new scenes; 25 of 28 target misses in the development scene) and broke nothing (0 of 432 new, 0 of 1,512 development
  records). The frozen test did not make the claim: the failure was rarer on fresh layouts (8.6 points against a
  10-point bar) and occurred in only 2 of 8 layouts (amendment 3's 0.5 threshold was too wide). A safe grounding fix
  with a small, narrow effect, not a general gain over C, and about turning a click into a position, not social
  reasoning.

18 Oct) lost to the recorded C (56.8% against 76.0%) and failed on unseen situations; its limits were circular labels
## 7. Caveats

- Every moving-people number is in simulation, five straight-line evaluation scenes, 3-10 seeds; per-episode share
  differences of ~0.1 are within CT's own redraw noise (same answer, timing jitter), so headline claims rest on
  paired counts, keep-up and calls. The timing-only floor over all cells (CT2-none, every part off, against CT; same
  method and answers): mean |d| about 0.06 per cell, mean shift under 0.011, yet a lean of 14/24 untied pairs
  (Sonnet, sign p about 0.14). So paired counts near that lean (e.g. CT2m against CT2, 12/27) need that context;
  the headline counts (43/2, 32/8) are far outside it.
- Scoring sensitivity of the headline (post hoc, `results/moving-lines/questions/headline_band_out.txt`):
  - With the side check (strict scoring), CT and CT2 beat R1 at every band top tried: 1.2, 1.5 and 2.0 m, Sonnet
    +0.28 to +0.31, DeepSeek +0.23.
  - Without the side check, the gain shrinks as the band widens. For Sonnet it is +0.19 at 1.2 m, +0.06 at 1.5 m and
    0.00 at 2.0 m (CT2 against R1, 21/26), because Sonnet's marked point puts the robot level with the line but
    beside it. DeepSeek still has +0.17 at 2.0 m.
  - So against R the claim rests on "in line means on the line's axis". That is a natural social criterion, frozen
    before the fresh seeds, but the report should state it.
  - Bystander scenes: CT3 has more than CT2 (DeepSeek 1,036 against 881, more in 21 of 75 pairs and fewer in 14;
- Tracking is oracle except in CTd/CTe. CTd's live tracks match a true person 89% of the time, with 2% id switches;
  71% of tracks at any moment are stale (out of view, held at the last position).
- Sim latency pauses for the request-frame detection (0.16 s warm), so it isn't counted. CTe's gain is stated after
  adding it back.
- The robot is never served in the sim; one Sonnet cell had a >90 s API stall (counted as failure, DEVIATIONS 11).
- Several items are post hoc and labelled so: the strict scoring's origin, the ablations, all diagnoses.

## 8. Ideas round 2: one role belief against the best rules (3 Nov)

- **CT3F**, the best bolt-on rules: CT3, plus the whole group following an advance, plus re-deriving the anchor, plus
  replaying the wait since the image.
- **CT4** (`sgi.roles.RoleBelief`): one exact joint belief over who waits (2^n subsets of the people seen when the
  answer was asked). The answer is its prior; motion, stillness and following are its evidence.

The claim was pre-registered (`results/moving-lines/best-version-and-running-guess/CT4_FROZEN.md`): CT4 is more robust than CT3F to people who are slow
to step up. It was tested on 72 new layouts drawn for the test, with stress flags (late steppers, a restless
bystander), on DeepSeek, scored once by a frozen scorer (`results/moving-lines/test-layouts/score_dsv41_out.txt`).

| Right last (share of re-solves where the robot comes after the right person) | clean | late steppers | restless | both |
|---|---|---|---|---|
| CT3 | 0.17 | 0.25 | 0.29 | 0.25 |
| CT3F (rules) | 0.49 | 0.45 | 0.33 | 0.41 |
| CT4 (role belief) | 0.45 | 0.42 | 0.36 | 0.53 |

- **The claim fails:** on the 36 layouts with late steppers CT4 and CT3F tie, 14/14/1, p 1.0.
- **What held:**
  - Both new arms roughly double CT3 on right last (CT4 41/18, p 0.004; CT3F 33/11, p 0.001; reported comparisons,
    named before the run).
  - CT4 knows who waits much better: recall 0.80-0.87 against 0.51-0.64 in every cell. Ticks with someone ahead but
    no one to follow: 0.02-0.05 against 0.14-0.25.
  - CT4's roster is liberal and overconfident in the middle. People it rates 0.5-0.7 are waiting only a third of the
  - Thresholded at 0.5, the roster flickers: it changes between consecutive re-solves 17% of the time against 5%. The
    goal jumps more than 0.3 m on 20% of re-solves against 13%.
- **Reading:** one belief seeded by the VLM's answer replaces the pile of rules and matches the best of them, with
  better perception of who waits. But turning that belief into a single roster throws its uncertainty away, and the
  robot pays for it in time in line.
- **Method lesson:** offline replay with synthetic stress predicted that the rules would collapse under late steppers.
  They did not. The offline lag was harsher than the sim's, and the offline loop never moved the robot.
- **Next (design data only, new layouts for any test):**
  - decide with the belief rather than its thresholded roster: an expected-cost choice of place, with asymmetric costs
    and a cost for moving, set from first principles and frozen before pilots;
  - calibrate the belief's evidence to the simulator, reported as such;
  - fix how scoring treats a place that a bystander physically blocks.

### 8b. The same comparison on Sonnet, and no-harm on the original scenes (3 Nov, reported, not claims)

  - right last, layouts with late steppers: CT4 against CT3F 5/15, p 0.041. CT3F is ahead here, so the claim fails
  - CT4 again knows who waits better (recall 0.92-0.93 in every cell, almost no ticks with no one to follow);
- **No-harm on the original scenes** (DeepSeek, 4 scenes x seeds 20-29; `results/data/moving/runs_noharm/dsv41/`):
  - CT3F beats CT3 on time in line (strict): 0.400 against 0.337, 25/7/8, p 0.002; after the first service 0.501
    against 0.421, p 0.001; fewer cut-ins (4.6 against 7.5);
  - CT4 is at 0.259 (18/21/1, p 0.75 against CT3).
- **Reading:** on both models and both scene sets, the role belief is better at who is waiting and worse at where
  to stand. The gap is in turning the belief into a place, which is what round 3 tests.

### 8c. Round 3: decide with the belief, not a thresholded roster (pilot, 3 Nov: the key prediction failed)

- **W0:** CT4 with the stillness evidence averaged over 5 s (the measured decorrelation time of a stance), so one
  odd frame does not flip the roster.
- **CT5:** W0 plus an expected-cost choice of place. Costs fixed in advance in service intervals: cutting in ahead
  of a waiter 1 each, standing behind a non-waiter 0.25 each, moving 0.125 a metre. The robot comes after anyone
  with at least a 0.2 chance of waiting, and keeps its place unless a switch saves more than the move costs.
- Pilot: 24 new design layouts (candidates 1000-1199), one seed each, DeepSeek, arms CT3F, CT4, W0 and CT5.
  Predictions were hashed blind from CT2's episodes before any of these arms ran; the key one is that CT5 and W0
  beat CT4 on strict time in line from the first service. If CT5 also reaches CT3F, a confirmatory test on fresh
  layouts (2000-2449) follows.
- **Result** (scored once, `results/moving-lines/best-version-and-running-guess/round3/h_pilot_score_out.txt`):
  - the 5 s window halves roster changes (W0 and CT5 at 0.59 and 0.58 of CT4) and CT5 cuts in less (6.4 against
    9.1 samples);
  - **but neither gives back time in line.** Strict, from the first service: CT3F 0.188, W0 0.072, CT4 0.068,
    CT5 0.036 (CT5 against CT4 2/11, p 0.022). The goal still jumps about twice as often as CT3F's;
  - no confirmatory test.
- **Where the robot goes** (exploratory): CT5 lines up behind a bystander on 49% of ticks with someone ahead,
  against 34% for CT4 and 23% for CT3F. Its rule comes after anyone with a 0.2 chance of waiting, the belief rates

### 8d. Why the belief stood in the wrong place: the robot is part of the queue it infers (3 Nov, pilot)

Two controls, each changing one thing outside the belief, on the same 24 pilot layouts (zero calls, scored by
frozen scorers):
- **CT4P**, CT4 with CT3F's line direction: 0.060 strict from the first service against CT4's 0.068. The
  direction was not the gap.
- **CT4PJ**, CT4P plus CT3F's commitment rule: once the robot has joined, people behind it no longer count.
  - **0.182 against CT3F's 0.188**: CT3F better on 8 layouts, worse on 11 (p 0.65). Against CT4P, better on 14 of
    18 untied layouts (p 0.031).
  - Time joined before leaving rises from 11 to 19 s.
- **Reading:** the belief answers "who is waiting?", including people who queue after the robot. The goal needs
  "who is waiting ahead of me?". Without the commitment rule, the robot keeps giving its place to newcomers.
- **Caveat:** the commitment rule needs the robot to reach its goal first. On one layout the belief kept the goal
  2-4 m from where the robot stood, and the rule never engaged. That is the same "who" error becoming a "where"
  error.
- **Next:**
  - a confirmatory test of CT4PJ against CT3F on fresh layouts (candidates 2000-2449), frozen before it runs;
  - offline: the belief's transitions as the queue's own process (leave at the front at service, arrive at the
    tail, step up together, the robot holding its own place), aimed at the ex-members and pausing walkers it
    still over-counts.

### 8e. The confirmatory test: the commitment effect did not replicate on the primary model (3 Nov)

| Model | CT3F | CT4PJ | CT4 | CT4P | H1: CT4PJ > CT4P | H2: CT4PJ within 0.05 of CT3F |
|---|---|---|---|---|---|---|
| DeepSeek (primary, n 66) | 0.208 | 0.159 | 0.119 | 0.107 | 27/16/23, p 0.126: **fails** | not claimed (lower bound -0.080) |
| Sonnet 5.5 low (reported, n 71) | 0.246 | 0.174 | 0.131 | 0.124 | 33/17/21, p 0.033: passes | fails (lower bound -0.105) |

- **What this says, by the rules fixed in advance:** on the primary model the pilot's commitment effect did not
  replicate. The direction held on both models: CT4PJ was ahead of CT4P by 0.05 on average, and it lasts longer
  in line before leaving (19 to 24 s on DeepSeek). But the effect is too small to clear the bar on DeepSeek.
- **Parity with CT3F failed on both models.** CT3F beat CT4PJ on 34 layouts against 17 (DeepSeek, p 0.024), and
  on 44 against 18 (Sonnet, p 0.001). The pilot's near-tie (0.182 against 0.188 on 24 layouts) was luck of the
  draw.
- **Reading:** the single role belief, even with the robot's own place in the order, does not yet replace the
  hand-built package. CT3F's other rules carry weight the belief does not reproduce. The best method stays CT3F
  (CT3 plus the group test, the gated anchor and the photo-time replay).
- **The next design fixed in §8 is already spent.** The process-structured belief (round 5, `results/moving-lines/best-version-and-running-guess/round5/`)
  was built offline on design data and failed its own written bars (`RESULTS_1.md`), so it never went to a run.
- **Exploratory follow-up** (`round4/q_confirm_explore.py`, questions written before any per-layout data;
  output `q_confirm_explore_out.txt` ec0459bd; not claims):
  - CT4PJ's gain over CT4P sits where its join test fires: +0.07 against about 0. It disappears under the lag
    flag.
  - The goal moves more than CT3F's: 14 jumps of over 0.5 m a minute against 8-9. Where CT4PJ trails CT3F, it
    jumps more on 28 of 34 layouts (DeepSeek) and 37 of 44 (Sonnet).
  - So CT3F's lead looks more like a steadier goal than a better choice of whom to follow; CT4PJ picks the right
    last slightly more often on DeepSeek. This is a lead, not a result.
  - Per-layout differences barely agree across the two models (rank correlations +0.24 and +0.06).

### 8f. Why CT3F wins: it knows when it is at the front (round 6, logs only, exploratory)

From `results/moving-lines/best-version-and-running-guess/round6/RESULTS_1.md` (d7bfb89a) and `RESULTS_2.md` (a95a01d7). Questions were fixed before
each step, and all 17 hashes in `ROUND6.sha256` were checked.
- **Correction:** part B's "same accuracy" (RESULTS_1) was a scoring artefact. It counted "nobody ahead" as wrong
  even when the sim agreed, and RESULTS_2 §1 withdraws it.
- **Scored correctly, while truly in line:**
  - with someone ahead, both pick the right person about equally often: CT3F 0.73 and 0.72, CT4PJ 0.71 and 0.64;
  - with nobody ahead (the robot at the front), CT3F knows it is first 0.52 and 0.66 of the time, against
    CT4PJ's 0.22 and 0.25. CT4PJ keeps aiming behind someone, most likely the customer just served, and takes 4-6
    s to drop them against CT3F's 2.6 s.
- **Stability:** CT4PJ also switches whom it follows 6.7 and 7.5 times a minute, against CT3F's 2.6 and 3.0.
  Freezing the choice at joining cuts switching below 1 a minute, but lowers accuracy.
- **Part C, is there a principled middle ground?** No.
  - A "hold unless someone else leads by m" rule never reaches CT3F's accuracy on the final-test logs: at best
    0.57 and 0.56, against 0.66 and 0.71.
  - On the pilot logs it does, at m 0.7. The pilot overstated the role belief a second time.
  - CT5's cost decision with its fixed costs switches 8-10 times a minute and is no more accurate.
- **Future work only:** round 5's "walking away" signal fires on 81% of ex-members' walks and 1-6% of members'.
### 8g. Is the test right? Ceilings and the band edge (3 Nov evening, project's question)

- **Band edge (rescoring, zero calls):** with the in-line band's top at 1.5 m instead of 1.2 m, every method rises
  by only 0.02-0.03. CT3F stays ahead of CT4PJ on both models. Most "too far" seconds are far beyond the edge,
  behind the wrong person or well back, not near-misses. The one reported result that depends on the edge is
  Sonnet's H1 sign test (p 0.033 at 1.2, p 0.134 at 1.5).
- **Ceilings (done; round7/RESULTS_3.md a5d758eb, read by ROUND7's fixed readings; description only).** Four
  test-only robots on the 72 confirm layouts, no calls, each told only who is ahead of the robot. DeepSeek, n 66,
  strict from the first service at band top 1.2 (1.5):
  - OW (ideal place) 0.640 (0.652); CT4OT (true who + counter) 0.330; CT4O (true who) 0.218; CT4PJT (true counter)
    0.207; CT3F 0.208; CT4PJ 0.159. CT3F and CT4PJ match the frozen scorer (bf2f51eb).
  - Gaps: where to stand given perfect information (OW − CT4OT) +0.310, mostly at the front (+0.249); counter
    point (CT4OT − CT4O) +0.111; who waits +0.059 with the model's counter, +0.123 with the true one.
  - The test is passable but not to 1.0: even the ideal place loses about a third, almost all while the robot walks
    or follows (the sim robot's own cost, the same for every arm).
  - At the front the solver stands 1.0 m from the counter point at about 42° off the axis, while the front spot is
    0.85-0.95 m straight down it (median 0.69 m from slot 0). Solver geometry, not information; not to be tuned.
  - Post hoc: the full truth about who waits, with the model's counter, only reaches CT3F (+0.010, p 0.78). So
    walking away passes its fixed bar against CT4PJ, but on its own it is unlikely to beat CT3F.
    first 43% of its window, the methods 12-20%. A question for the project, not a change.
  - Leads (flagged): where served customers stood, from the wait's tracks (one observation for both the counter
    point and the front place); a zero-call CT3FT cell (CT3F with the true counter) to separate the two.
  CT3F given the true counter point: 0.261 against CT3F's 0.208 (+0.053, lower bound +0.013, but 27 wins / 23
  losses, sign p 0.67; at band top 1.5 m +0.048, p 0.16). The rule's outcome is B at 1.2 m and C at 1.5 m, decided
  by the 0.05 bar's edge, so it goes to the project. Who is ahead, once the counter is right: CT4OT − CT3FT +0.068
  (36/17, p 0.013). For the best arm the counter point and who waits each add about 0.05-0.07; neither alone lifts
  rest is the solver's front place. Front shares of the strict score: OW 52%, CT4OT 25%, CT3FT 17%, CT3F 9%,
  CT4PJ 5%.

### 8h. Still scenes: detector people beat C's people call, faster at no loss (E2, frozen, 4-5 Nov)

- **Test** (`results/still-scenes/people-finder/E2_FROZEN.md` 38b2ef01; result `results/still-scenes/people-finder/E2_RESULTS.md` 8c7a3f5e): 40 fresh realistic Blender scenes
  (generator seeds 500-504, the 8 people rows), 2 episodes x 2 trials, three free models. Per unit the plan, target and
  context are shared; A0 = C, A0b = C drawn again (redraw floor and timing placebo), D = C with A1 detector people.
- **Claim holds on all three models:** D is non-inferior to C within the redraw floor, never worse by layout, makes one
  call fewer, and is faster (sign test, placebo null).
  - Qwen: right 144 -> 154 of 159; whole answer 107 -> 83 s; faster in 133 of 159.
  - DeepSeek: 126 -> 132 of 157; 66 -> 50 s; faster in 148 of 157.
  - GLM: 100 -> 125 of 143; 142 -> 109 s; faster in 107 of 143. 11 of 160 GLM units left out after 13 resume passes
    (constraints timeouts, 8 in conversation groups); in the worst case for D the claim still holds.
- **More accurate (H2):** pooled over the three models, net +41, layout p 0.0001; per model Qwen and GLM, not DeepSeek.
  GLM's own H2 depends on the left-out units (post hoc worst case p 0.19).
- **Caveats:** development situation types, the "real" variant, static scenes, free models; the detector's false
  people were 0.475 per image (twice its earlier rate) without harm to D. These scenes are now spent.
