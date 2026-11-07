# I11 Stage 0 (phase 1b): deviations from FROZEN.md and clarifications

FROZEN.md sha256 `11f7b32746c7fc0be414883b430d59ddef98d2388fd89e082e40361b637a9165`, checked OK before any work, and checked again at the start of every script (`i11lib.check_pins`, which also checks the seven pinned code files; all matched).

Entries D1 to D7 were written after the S0 checks (synthetic and fit data only) and **before any evaluation record or scene was loaded**. Later entries carry the step they were written at.

**D1. The fit set has 29 scenes, not 28.**
- `H.examples()` lists 29 worked-example scenes: 7 each for real, staffed and plain, and 8 placeholder (the 8th is `examples_07_audience_4`). FROZEN.md says "the 28 worked-example scenes of `H.examples()`".
- The operative definition is `H.examples()`, so all 29 are used for the S0 fit checks.
- The largest target-footprint or obstacle diagonal over all 29 is 2.259 m, an obstacle in `examples_07_audience_4`. That matches the note's 2.26 m, so L_S = 4.6 m is unaffected. Nothing was re-fitted.

**D2. Unit check (a): a test-construction fix, not a code change.**
- My first synthetic box for check (a), "a box 0.5 m in front of a wall gives two components", was 1.0 m tall. It gave 4 components ≥ 25 px.
- Cause: seen from 1.5 m at 5 m, the box's top face sits at about 84° incidence. Its two far rows split into 1-row strips of 80 and 82 px. This is exactly the known cost FROZEN.md section 4 predicts for horizontal tops below the camera.
- The front face and the wall did separate correctly. The 20 m wall was structure (extent 6.5 m) and the 3 m wall was an object.
- Fix:
  - The test box is now 2.0 m tall, so no top face is visible.
  - The wall probe point was raised to 2.8 m, so its ray clears the box.
  - The 1.0 m case is kept as an informational line, (a'), in `s0_fit.txt`.
- No component, variant or trigger code was changed.

**D3. How two fit-check measures were read (section 10, step 2).** These are clarifications, not changes.
- **Structure share.** "The share of pixels within 0.3 m of the room's bounding polygon, or with z > 2.85 m, that are labelled structure" is computed over non-floor pixels (z ≥ 0.15 m). Floor pixels can never be structure by construction, so counting them would measure the floor's share instead. "Within 0.3 m" means distance to the polygon's boundary ≤ 0.3 m, or outside the polygon.
- **Ground-truth pixels.** The person, target and obstacle pixels are non-floor pixels at z ≤ 2.85 m:
  - person: lifting's person body mask, for every person;
  - target: lifting's body mask of every target;
  - obstacle: per-pixel lift within 0.05 m of an obstacle polygon (the same 0.05 m as the place body mask).
- **Observation (fit data, G1).** 54.5% of obstacle pixels are structure. By gap to the bounding wall:
  - all 66 obstacles that are mostly structure touch the wall (gap < 0.10 m; 74 obstacles do);
  - none of the 41 free-standing obstacles is.
  - Target pixels are 100.0% object and person pixels 99.9% object, with 0 structure. Walls and ceiling are 99.6% structure. 0 of 56 visible people are absorbed into structure.
  - So no section 4 definition fails visibly on fit data. **No ADDENDUM_1 was written**, and V1 is primary as frozen.

**D4. The reference reproductions write only into this folder.**
- `rerun/lifting/lifting.py` and `frontsnap.py` are re-run unchanged by importing them and calling their `main()`, with each module's `OUT` pointed at `phase1b/i11/repro_lifting/`. Their own `main()` would otherwise overwrite the original outputs in `rerun/lifting/`, which lies outside this folder.
- Their outputs are then diffed against the originals.
- The per-record replay of the front snap calls `frontsnap.front_snap` unchanged.

**D5. Verification is not independent.**
- `verify_indep.py` therefore recomputes the key primary numbers with separately written code. It shares no code with `i11lib.py` or `lifting.py` beyond `sgi` and the harness, and uses its own breadth-first components, its own masks and its own window logic.

**D6. Fresh seeds.** `phase1b/fresh_seeds/` did not exist when S0 ran. It is checked again when M4 runs. If it is still absent, M4 has no fresh-seed domain, and this is reported.

**D7. Two record-level readings.**
- **Moved.** For the oracle and V1 to V4, "moved" means the fix returned a new target. That target's pixel is a pixel centre, so it always differs from the recorded continuous pixel. This matches section 6 ("its target pixel or world point differs"). For the front-snap reference, "moved" means `front_snap` returned a point, as in frontsnap.py.
- **Replay.** A record with source neither `constraints` (solver step) nor `standard`, and with no recorded pose, counts as a failure in the full-success replay. lifting.py left it as None. Section 6 counts a missing pose as a failure.

**D8. Fresh seeds, and a discarded first pass** (written after the primary pass, before any fresh-seed result was read).
- `phase1b/fresh_seeds/` appeared while the primary pass was running.
- The primary pass's built-in M4 step then read 195 `scene.json` files under it. At that moment seed108 had no `DONE.json` yet, so some of what it read may have been half-written.
- That code took only folders named `seedNNN` with NNN in 100–109, and guarded every scene folder with `H.guard_tree`. Only seed100 to seed108 existed, and no eval_16–19 scene is in any fresh set.
- Its output (`m4_fresh.jsonl`) was deleted unread.
- M4 on fresh seeds was rerun as `run_eval.py --set fresh` once all of seed100 to seed109 carried `DONE.json`. That run reads only the scenes each `DONE.json` lists and skips any held-out situation (there were none). It covered 260 scenes, 10 seeds × (18 eval + 8 examples).
- The fresh scenes are placeholder-style renders, so they test the component definitions on new random layouts, not on the realistic look.

**D9. Labelled post hoc in the outputs.**
- The pooled X-new I33 line pools across models within X-new only; FROZEN.md asks for per model.
- T1 is reported under G2 components.
- Anything else is marked "(post hoc)" where it is printed.
