# Bystanders: does the model's roster keep non-waiting people out, and does motion evidence beat CT2? (frozen 2026-11-02 ~15:10, before any bystander cell)

## Scenes (fixed by hash)

`sim/scenarios/` sha256 prefixes: `cafe_line2_bystanders.yaml` f614baa44ec53ecb, `room_coffee_cart_bystanders.yaml`
e43e516af1504f43, `sweet_spot_line2_bystanders.yaml` 327e22f06050943d. Each is its moving or slots original plus three
standing people marked `bystander: true`, fidgeting (every 5-10 s, a new point within 0.35 m of their place):
- a lone person 0.92-0.93 m from the line's last, off to the side and facing away;
- a talking pair, 1.55-2.26 m from the last.

Mirror control in room: the middle member, female_0, is turned 90 deg but stays in line and steps up.

## Facts established with zero calls, from the scene files only (not predictions)

- **F1. G's 1 m chain takes bystanders when lines are full length** (`bystanders_geometry_out.txt`, fidget draws):
  - cafe, line of 2: the lone man, 61%;
  - room, line of 3: the lone man and the pair (29-61% each);
  - sweet_spot, line of 2: the lone man (57%) and through him the pair (21-24%).
- **F2. The solver's place-in-turn rule takes bystanders when lines are short.** `sgi.solver._turn` makes any
  unlisted, non-exempt person standing within 1.0 m of the place in turn, behind the last, the new last. This rule is
  shared by C1, CT, CT2, CT3 and G. Real solver code at the nominal places (`results/moving-lines/bystanders/pre/turn_check_out.txt`):
  - cafe, line of 1: the lone man;
  - cafe, line of 2: the pair's near member;
  - room, line of 2: the lone man and the pair's near member;
  - sweet_spot, line of 1: the lone man and the pair's near member.

  An answer can exempt people only by listing them under `keep_out_of_shared_space`. That frees the pair, never the
  lone man: no constraint kind says "this person is not waiting". So CT2 can discard role knowledge the model has.
- **F3. Share has little headroom on the two-person lines.** On the originals, DeepSeek CT2's strict share is 0.05
  (cafe_line2), 0.36 (room) and 0.05 (sweet_spot); Sonnet's is 0.11, 0.43 and 0.14.
  - Share comparisons therefore get their power mainly from room.
  - The primary measure below is capture, which does not depend on reaching the line.
- **F4. CT3's join gate leaks a little under fidget** (`results/moving-lines/bystanders/pre/gate_fidget_out.txt`). The step-up branch
  (0.55 m nearer the target within 20 s) is open about 1.5% of the time per bystander, and at least once in 17% of
  episodes. The walk branch (1.0 m) never opens, because fidget moves people at most 0.7 m.

## Arms (15 cells each: 3 scenes x seeds 10-14)

- **R1, C1, CT, CT2:** as frozen (CT2_FROZEN.md).
- **G:** the chain rule (IDEAS_interface F2, A00).
  - When the plan call returns the model's target point, take the tracked person nearest it.
  - Close the chain of tracked people within 1.0 m.
  - Use come_after = the chain, with no other constraint.
  - Re-solve every 0.5 s with CT2's resolver options.
- **CT3:** CT2 plus motion evidence for membership. The resolver uses `src/sgi/resolve.py` as it stands at this note
  (sha256 216d2fda5ad626cf), with:
  - `co_motion_s = 5.0`, `advance_m = 0.4`, `follow_m = 0.25`, `moved_m = 1.0`, `history_s = 20.0`;
  - `_ADVANCE_WITHIN_S 3.0`, `_ADVANCE_EVERY_S 5.0`, `_STEP_UP_M 0.55`.

  It adds one required change: **the join gate also binds the solver.**
  - Anyone the resolver would not take as joining (not waiting, and `_moved` false) is passed to `solver.solve` as
    exempt from `_turn`, like `apart`.
  - The current code gates only `Resolver.record` (admission for good). `_turn` still runs in every 0.5 s re-solve,
    so without this change a fidgeting bystander still captures the goal in each solve (F2).
  - Static solves are unchanged (default: nobody exempt).
- **CT3-j:** CT3 without the follow check (join gate and solver exemption only), for attribution.
- **Gl (optional, recommended):** G's chain taken on CT2's answer frame and delivered at CT2's answer time. This is a
  timing-matched control: roles from geometry against roles from the model, all else equal.

Calls: only CT's first answer (4 calls a cell) and R1 (3 a cell) are new, about 105 free DeepSeek calls. C1, CT2, CT3,
CT3-j, G and Gl reuse CT's cached plan and answer.

**No-harm runs on the originals, zero calls:** CT3 and CT3-j on the original three scenes, from cached answers:
- DeepSeek seeds 10-14: 15 cells.

## Required logging (no behaviour change)

- For each solve:
  - the solver's last, as a track id;
  - the `_turn` joined people, as track ids;
  - the exempt set;
  - for G, the chain.
- For each answer, its come_after people mapped to track ids (0.5 m at the answer frame).
- CT3's `non_followers`, with times.
- Bystander ids, from the scenario.

## Measures

- **M1. Capture (primary):** the share of time after an arm's first goal during which the solver's last is a
  bystander, so the robot's place lies behind a bystander. Reported per cell, with the number of cells captured for
  at least 1 s.
- **M2. Roster at the answer:** bystanders among the answer's come_after people (the same answer for C1, CT, CT2 and
  CT3), and bystanders in G's chain at its first goal. Zero calls.
- **M3. Strict share** (SCORING_STRICT.md), paired by cell; keep-up within 10 s of a service.
- **M4. False drops:** true members among CT3's `non_followers`. The mirror member female_0 is reported apart.
- **M5. No-harm:** CT3 and CT3-j against CT2 on the originals.

## Predictions (inference, written before any cell)

- **P1 (G's roster):** a bystander is in G's chain at its first goal in at least 6 of 15 cells.
- **P2 (the model's roster):** DeepSeek names a bystander as waiting in at most 3 of 15 answers.
- **P3 (CT2 discards it):** CT2 is captured for at least 1 s in at least 7 of 15 cells, including at least one cell in
  each scene. Most capture time comes from the lone man in short-line states.
- **P4 (CT3 against CT2, capture):** CT3's pooled capture time is at most a third of CT2's, and CT3 is captured in at
  most 4 of 15 cells.
- **P5 (CT3 against CT2, share; the judged bar for "beats CT2"):**
  - more paired wins than losses, and a mean strict-share difference above 0;
  - at least 4 of 5 room cells at or above CT2;
  - "Beats CT2" is claimed only with P4, and a sign p of 0.10 or less over untied cells. Otherwise it reads "not
    shown" (a 15-cell pilot with a floor in two scenes).
  - Expected size: +0.05 or less overall.
- **P6 (no harm):** on the originals, for each model:
  - CT3 against CT2 has a mean strict shift within plus or minus 0.02, and a per-cell mean |d| of 0.06 or less (the
    timing floor);
  - true-member false drops occur in at most 2 of 15 DeepSeek cells and at most 3 of 30 Sonnet cells.
- **P7 (mirror):** female_0 is a non-follower in at most 1 of 5 room cells (CT3).
- **P8 (attribution):** CT3 and CT3-j differ in capture in at most 4 of 15 cells. The follow check matters only where a
  bystander was admitted anyway: through F4's leak, or named by the model.

## How the role-knowledge question is read

- **The model's roles matter** if all of these hold:
  - P2 holds;
  - G's capture exceeds CT3's in at least 8 of the cells where either is captured;
  - CT3's strict share is at least G's (paired wins > losses; against Gl where it is run, since G starts sooner).
- **The model knows, but CT2 discards it** if P2 and P3 hold, and CT2's capture is not lower than G's in most
  captured cells.
- **Roles from the model don't matter here** if P2 fails: the model names bystanders about as often as G's chain
  takes them.

## Development protocol

- Seeds 0-2: render checks, plus one smoke cell per new arm (G, CT3, CT3-j) on DeepSeek seed 0 in each scene, to check
  the logging and false drops only.
- A smoke cell may veto `co_motion_s = 5.0` once, if a true member is dropped. The fallback is 10.0 s, recorded in
  DEVIATIONS.md. No other change is allowed.
- If G or the solver exemption is built differently from the text above, that is written in DEVIATIONS.md before the
  first judged cell.

## Known limits

- The lone-person trap needs particular line lengths (F1, F2). With service every 15 s, two-person lines often hold
  1 person.
- No pickup waiter whose motion is tied to service.
- Tracks are the sim's true positions.
- The 1.0 m walk branch of the gate separates bystanders from joiners by construction, because fidget is at most
  0.7 m. Real bystanders who wander more than 1 m would pass it. These scenes show the mechanism works and costs
  nothing elsewhere (P6), not that it generalises.

## Not judged here: a scene where meaning decides

A gapped line: one member standing 2.5 m behind the person ahead, facing the counter.
- G's 1 m chain stops at the gap.
- The solver's 1.0 m joining reach does not bridge it.
- So G puts the robot in the gap (a cut-in), while a model that names every member does not.

This tests role knowledge against geometry more sharply than bystanders, which mostly test what the solver does with
unnamed people.
