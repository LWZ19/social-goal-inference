# Ideas through the interface lens: what we ask the model, and how its answer binds to geometry

Written 2026-11-02 by a brainstorm agent. No model calls; read-only; seeds 10-12 not opened. All numbers are
post hoc on evaluation episodes (seeds 0-2), so they are leads to confirm on fresh seeds, not results.

**Static check used below ("in line at the frame").** The solved first-frame goal, scored with SCORING.md's in-line
test against the true members at the frame (slots 0..k-1 at t = 2 s, headings towards the slot ahead). Floor: an
open map (everything navigable except behind the service edge), standing in for the navmesh CT re-solves on; it
ignores tables, so it is optimistic. Walkers keep the model's lifted position. It measures the answer, not the
episode.

---

## 0. Two findings made while checking ideas (read these first)

**F1. A served person who comes back can never wait again (resolver bug), and two-person lines never get a line
direction.** In `Resolver.record`, absorption skips `n in self._waiting`; a dropped person stays in `_waiting`, so
the solver's `joined_waiting` report for a rejoiner is ignored forever, and `inputs` strips them as `left`. DESIGN.md
says a returner is "an unlisted person, whom the solver's place-in-turn rule absorbs when they stand behind the
end"; the code does not do that.
A synthetic replay of `Resolver` reproduces it (rejoiner reported 3 times as joined, never re-added; once the other
member is served, `come_after` is `[]`). Separately, `_line_direction` needs the waiting people >= 1.0 m apart,
so a two-person line (0.7 m) never has one.
- Evidence: every two-person-line CT episode with a `come_after` (line2_moving, sweet_spot: 16 of 16 over DeepSeek,
  x-test-3, Sonnet) loses all its waiting people at t = 24-73 s; afterwards the place in turn is used in 0.00-0.04 of
  re-solves, and 10 of the 16 then log 16-36 cut-in samples (the robot heads for the target beside the line).
- So "8 of 15 usable first answers" overstates the first answer as the limiter: 5 of DeepSeek's 8 right answers are
  two-person lines that failed by maintenance (sweet_spot 0.18 / 0.00 / 0.00, line2 s1 0.10). Sonnet's line2 and
  sweet_spot runs (right answers, shares 0.03-0.15) are the same story.
- Fix (general, `come_after`-level): re-admit a dropped person whom the solver reports as joined (clear `dropped`,
  reset their closest approach). Verify with 0 model calls: rerun CT on the 6 two-person-line cells x 3 models from
  the cache (sim time only). Prediction: in-turn after the first two services rises from ~0 to > 0.5 wherever the
  rejoiner is back before the last waiting person is served. Label it a bug fix to the frozen design, post hoc.

**F2. A model-free control already solves the first frame in these scenes.** Take the tracked person nearest the
model's target point, close the chain of people within 1 m of them, order it by the members' tracked headings, and
solve with only that `come_after` (variant A00 in `replay.py`). In line at the frame: DeepSeek 15/15 (27/27 with
dev and break-it frames), x-test-3 15/15, Sonnet 13/13 with constraints. As answered (same floor): 9/15, 11/15,
10/13. Keeping the model's other constraints with that same membership (A0) loses 3 of 27 DeepSeek frames
(cafe_busy s0: `face` turns it 48 deg; bent s0/s1). Meaning: in these scenes the model's roster adds nothing once
geometry has the target, and its extra constraints sometimes subtract. A reviewer can make this point; idea 7 turns
it into a test design.

---

## 1. Seed-and-close: ask for one waiting person, let the tracks close the set

- **Failure class.** Membership under-listing and wrong line ends: come_after names only the front (DeepSeek line2
  s2, line6 s0), only the back person who is also the target (line6 s2), a member lost to a bad lift (room s0,
  cafe x3). 7 of 15 DeepSeek first answers; 4 of x-test-3's; 3 of Sonnet's.
- **Mechanism (general, tied to the `come_after` kind).** Treat the model's `come_after` people as seeds, not the
  roster. On the tracks at answer time: add everyone within 1 m of a member, repeatedly; keep the connected
  component nearest the target (drops a named passer-by); take the line's front-to-back direction from the members'
  tracked headings (people waiting their turn face where it is taken), not from the target point. In maintenance,
  re-derive the set every re-solve from the person nearest the target: identity-free, so F1's lock-out and the
  "everyone named was served" emptying both disappear.
- **Measured (zero calls, `replay.py` V0..V5).** In line at the frame:

  | | as answered (V0) | tracked people (V1) | + closure (V2) | + heading direction (V3) | + prune (V5) |
  |---|---|---|---|---|---|
  | DeepSeek | 9/15 | 10/15 | 12/15 | 14/15 | **15/15** |
  | x-test-3 | 11/15 | 11/15 | 11/15 | 12/15 | **14/15** |
  | Sonnet (13 with constraints) | 10/13 | 10/13 | 10/13 | 10/13 | **12/13** |

  Any single correctly named waiting member, alone as the seed, gives in line in 110 of 112 cases. Unchanged for
  D = 0.8, 1.2, 1.5 m. Fails on 2 of 3 bent-line frames, and A00 (same set, no other model constraint) passes them,
  so the model's other constraints, not the closure, break those two.
- **Why new.** B2 has the model write a role predicate; here the predicate is fixed by what `come_after` means and
  bound by tracks, and the model gives one seed. The prompt can then ask for less ("name at least the person who
  waits nearest the target"), which is easier for a weak model.
- **Cheapest next test.** 0 model calls: CT rerun from the cache with closure at first match and identity-free
  re-derivation, 15 eval cells x 3 models. Then fresh seeds (not 10-12; e.g. 13-15).
- **Prediction / refutation / risk / effort.** Predict CT DeepSeek share while someone is ahead >= 0.25 (now 0.14).
  Refuted if the episode share does not move although the frame check does (then maintenance dominates). Risk: the
  1 m chain is a line-shaped prior; bystanders standing next to a line get absorbed (test in idea 7's scenes).
  Effort: half a day.

## 2. Union over parallel draws (consistency aimed at omissions)

- **Failure class.** The models' membership errors are mostly omissions (front only, back only, a member missed);
  commissions (a passer-by named) are rarer (6 model-episodes, all in cafe; plus one phantom from a bad lift).
- **Mechanism.** Ask the constraints call k times in parallel (latency = the slowest, not the sum); membership =
  union of the draws' `come_after` (matched on tracks), other constraints by majority. No geometric prior, so it
  carries to groups and to waiting that is not a chain.
- **Measured (zero calls, proxy).** The three seeds of a scenario share the same line and differ only in walkers,
  so they act as near-redraws. Complete membership (every true member named): DeepSeek 8/15 one draw, 11/15 union
  of two, 12/15 union of three; x-test-3 11 -> 12 -> 12; Sonnet 10 -> 11 -> 12. Majority vote is worse (the right
  members are often named by one draw only, e.g. DeepSeek line6: [front], [all six], [back]).
- **Why new.** The landscape has no draw-merging; M2 measured redraw flips only as noise.
- **Cheapest next test.** 2 extra draws of the constraints call on the 15 eval first frames: 30 DeepSeek calls
  (cache key must include a trial index), merge offline, replay with `replay.py`.
- **Prediction / risk / effort.** Predict >= 11/15 complete for DeepSeek. Refuted if real draws at a fixed frame
  agree more than seeds do (then the proxy overstates diversity). Risk: union also unions passers-by; pair with
  idea 4's pruning. Effort: 2 hours plus calls.

## 3. Classification, not composition: one exclusive tag per tracked person

- **Failure class.** Vocabulary misuse that blocks the solver. 23 of 42 answers with a `come_after` also put
  waiting people in a `keep_out_of_shared_space` (DeepSeek 14/15, Sonnet 6/12, x-test-3 3/15); the solver never
  absorbs a person listed in a shared space (`apart` in `_turn`), which is why line6 s0's members 4 and 5 could not
  join. Also `within_distance(target, <= 1.2-1.8 m)` beside a `come_after` (DeepSeek cafe s0, line2 s1, sweet s1):
  harmless while people wait, but once the set empties it sends the robot to the counter (sweet s1's cut-in).
- **Mechanism.** The model writes no constraint list. For every numbered person it picks one tag from a fixed,
  situation-free set: uses the target now / waits for the target / with person #k / watches the target / passing /
  none, plus at most one free extra constraint. Constraints are compiled: uses + waits -> `come_after`; "with #k"
  pairs -> `keep_out_of_shared_space`; watches -> `keep_out_of_view(target)`. Exclusive tags make "waiting and in a
  conversation" impossible, and the forced pass over every person attacks omissions.
- **Why new.** Q9/B4 change how people are numbered; B2 changes how membership is expressed. This changes the task
  from composing constraints to labelling people, which weak models do more reliably.
- **Cheapest next test.** Zero-call part done (counts above). Then 15 DeepSeek + 15 x-test-3 calls on the eval first
  frames with a tag prompt, compiled and replayed offline (V0 scoring). Compare complete membership and in line at
  the frame against the 9/15 and 11/15 above.
- **Prediction / risk / effort.** Predict complete membership >= 12/15 for DeepSeek and zero shared-space misuse.
  Refuted if tags omit as many members as the lists do. Risk: a tag set is a new shared prompt that must not drift
  towards queue words; the compile table must be the same for every situation. Effort: 1 day.

## 4. Motion-revealed membership during the answer's latency

- **Failure class.** A snapshot cannot tell walking from waiting. Passers-by named as waiting: DeepSeek cafe s1,
  x-test-3 cafe s0-s2, Sonnet cafe s0, s2 ("about to join"); waiting people read as walking: Sonnet sweet s1/s2 (no `come_after` at all),
  DeepSeek line2 s2 (person 3 "walking right", left out). 9 model-episodes.
- **Mechanism (general).** The answer arrives 10-55 s after the ask, and CT has every track in between. Classify
  each person over that window by their stop-go pattern relative to the named people: still most of the time and
  stepping when a named member steps -> doing what they do (join the set); moving throughout -> not (drop from the
  set). The same rule serves groups that shift together and audiences that stay still.
- **Why new.** The resolver's leave rule acts only after a named person walks 0.75 m away; this uses the whole
  latency window, before the first solve, and also adds people.
- **Cheapest test.** 0 model calls, but the logs hold no per-person tracks: one cached CT rerun of the 45 eval cells
  with track logging, then the classifier offline. Prediction: separates walkers from waiting in every cell whose
  window contains a service. Risk: fast models (latency < service interval) see no step; fall back to stillness.
  Effort: half a day plus a sim batch.

## 5. Event-triggered one-bit questions instead of re-asks

- **Failure class.** Answers that tracks contradict, where re-solving can only keep the wrong reading: the waiting
  set empties while someone stands at the target (16/16 two-person-line episodes; F1), a tracked person stands in
  the chain but has no number (cafe: the model missed a member in 8 of 9 model-episodes), a named waiting person
  walks off without reaching the front.
- **Mechanism.** On such an event, mark one tracked person on the newest frame and ask a closed question with the
  request's own target words: "Is the marked person waiting their turn at <target>? yes/no". The answer flips one
  membership bit. Short output, so fast; any model, including a cheaper one.
- **Why new.** T1-T4 in the landscape re-run the whole pipeline; this edits the stored answer by one bit, so cost
  scales with contradictions, not with time.
- **Cheapest test.** Zero-call: count the events per episode from the logs (empties are counted; the unmatched-track
  event needs F4's track logs). Then about 2 calls x 18 two-person-line and cafe episodes, DeepSeek. Prediction:
  <= 3 calls per episode and in-turn after emptying > 0.5. Risk: yes-bias of closed questions; check with a
  marked walker as a control in each episode. Effort: 1 day.

## 6. Ask from where the model can see

- **Failure class.** Distance at the ask. Every true member is listed in 35 of 36 model-episodes when the line is
  within 3.5 m (line2, line6, room, sweet_spot), but in 1 of 9 in cafe (5.1-6.3 m); the cafe line is barely
  occluded (computed from the start pose), so size is the cause. Cafe is 3 of DeepSeek's 7 wrong answers and every
  model fails it.
- **Mechanism.** Before the single ask, use the tracks (no call) to find the people near the target; if they are
  farther than ~3.5 m or overlap in the image, drive to a side view about 3 m away, then ask. Latency (12-50 s)
  dwarfs the few seconds this costs.
- **Why new.** The landscape treats the frame as given; here the robot chooses the frame for the model.
- **Cheapest test.** Zero-call counts done. Then the 3 cafe seeds x 3 models with a pre-ask approach: 9 C answers
  (36 calls, Sonnet's announced first). Prediction: cafe complete membership >= 6/9. Refuted if the walkers still
  confuse the models from closer. Effort: half a day (sim side).

## 7. [BOLD] Ask whether, not who

- **Claim.** F2 says: give the model's target point and "people wait their turn here: yes" to geometry, and the
  first frame is right in 15/15, 15/15, 13/13. So ask the model only what geometry cannot know: the target, whether
  arriving means waiting for a turn, and who attends to whom or to what. Membership and order come from tracks.
- **Why bold.** It moves the model's job from people to situation, and it is only as general as the chain prior.
  The test that decides it is scenes where the prior fails: a pair chatting 0.8 m from the line's end, a staff member
  beside the counter, two lines to two targets, a waiting cluster that is not a chain. Write them as scenario YAMLs;
  first confirm offline (0 calls) that A00 fails there; then run C, CT and this variant (about 4 calls per first
  answer, 3 seeds x 3 scenes x 2 free models = 72 calls).
- **Outcomes.** If the model's answer beats A00 on those scenes, the paper has its missing evidence that the VLM's
  reading matters (today's scenes cannot show it). If not, the honest claim narrows to "VLM for target and
  situation, tracks for people". Score: payoff 5, uncertainty 4, effort 1.5 days.

## 8. [BOLD] A plan-view schematic as the constraints call's image

- **Claim.** Draw the tracked people as numbered discs with heading ticks on a top-down floor plan, with the target
  marked, and send it beside the camera view; skip the people call. It removes lifting errors (room s0, cafe),
  distance (idea 6) and facing misreads (the "face the robot" reading behind line6 s0's shared space).
- **Why new.** B4/Q9 mark tracked people on the camera image; a plan view is a different picture to reason over,
  and the line's order and spacing are explicit in it.
- **Cheapest test.** 15 DeepSeek + 15 x-test-3 calls on the eval first frames (two images, within the cap of four),
  compiled and replayed offline. Prediction: complete membership >= 13/15 for both. Refuted if models misread
  the diagram (wrong numbers, reversed order). Risk: departs from "the picture only" for the first answer (CT already
  uses tracks after it). Score: payoff 4, uncertainty 4, effort 1 day.

---

## Already in the landscape, not repeated

Expiry conditions written by the model (B1), role predicates (B2), extrapolated tracks (B3), tracker IDs as the
image's numbers (B4, Q9), trigger families T2-T4, people proposing the target (B5). Ideas 1, 4 and 5 are the
cheapest ways found here to get what B1/B2 aim at without asking the model for a new kind of output.

## Order suggested

F1 fix (0 calls) -> idea 1 in CT (0 calls) -> idea 7's scenes, offline A00 check (0 calls) -> ideas 2 and 3
(30 calls each) -> ideas 5, 6, 8.
