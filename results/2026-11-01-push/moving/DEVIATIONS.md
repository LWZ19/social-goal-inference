# Moving people: deviations from DESIGN.md / SCORING.md (hashes in SCORING.sha256 stay as frozen)

1. **2026-11-01, latency replay.** DESIGN.md says an answer is delivered at t_sent + `timings_s.total`. That field is
   the service's wall time, so a cached answer came back in 0.6 s instead of its stored 14.4 s (seen in the second
   development episode). The delivery time is now t_sent + the sum of the answer's stage times (plan, locate,
   context, goal, geometry), which replay the stored latency for a cached call. The intent is unchanged.
2. **2026-11-01, line-direction memory (a method change found on development data).** The first live episode
   (DeepSeek, CT, cafe_busy seed 0) cut in: with two people left at the front and the model's target point about
   4 m to the side, the solver's "one before" rule found nobody before the last person, and `_behind` ran the line
   from the target point, sideways. Added: the resolver keeps the direction of the waiting people's tracks at
   answer time (front = nearest the target, back = farthest, if at least 1.0 m apart) and passes it to
   `solver.solve(line_direction=...)`, which uses it only where nobody is before the last. Static solves are
   unchanged (default None). **cafe_busy is development data from here on** and is reported apart from the other
   scenarios.
4. **2026-11-01 ~22:35, re-ask bug: the re-ask arms of the first DeepSeek batches are invalid.** The episode loop
   decides whether to render a frame before delivering answers; when an answer is delivered and the next re-ask
   fires in the same tick (whenever latency exceeds the re-ask interval, nearly always with DeepSeek and x-test-3),
   the re-ask was sent with the last rendered frame, usually the start frame, and the cache answered it with the
   first answer. Found because Rq0's 128 answers all had the first answer's pose. Fixed in
   `Episode.requery` (renders a fresh view; checked: 75 re-asks, 75 different pictures). Arms affected: Rq, Cq,
   Rq0 (every re-asked answer). Not affected: R1, C1, CT, CTv, RT (their first frame and re-solves were rendered
   fresh). The affected DeepSeek episodes are kept apart in `runs/dsv41_invalid_requery/` and run again
   (queue_all.sh). x-test-3 had not started.
5. **2026-11-02 ~01:20, CT2 frozen (CT2_FROZEN.md, sha256 29808d8e) and the fresh-seed plan widened.** The waiting
   queue_fresh.sh (nothing started) is replaced by queue_freshA.sh / queue_freshB.sh: fresh seeds 10-19 for C1, R1, CT,
   CT2 (DeepSeek, Sonnet; x-test-3 if time), seeds 10-12 for CT2-vac, CT2-rat, CTs, repeats of CT and CT2, and the
   step-out and bent break-it cases. On fresh seeds `cafe` and `room_coffee_cart` are the `_slots` copies (extended
   queue slots; a sim-artefact fix) for every arm, CTs included (CTs_FROZEN.md named the 5 scenarios without this).
   Also found on seeds 0-2 by the brainstorm round: the resolver never re-admits a person who left and rejoined
   (`Resolver.record`), and two-person lines never got a line direction; both stay as they are in CT v1 and are fixed
   only in CT2 (opt-in options), so all CT v1 data are one consistent method.
6. **2026-11-02, found by the failure analyst (analysis/FAILURES.md).** The sim's queue-membership corridor in
   `experiment.py` is 0.8 m (`CORRIDOR_M`), while DESIGN.md says 0.6 m; all episodes so far ran with 0.8 m, which is
   kept for consistency and reported. The frozen scoring's in-line test has no side check (only cut-in uses the
   1.0 m corridor), so a robot level with the right gap but beside the line counts as in line; 38% of R1's in-line
   seconds (DeepSeek) are such. A strict scoring with a side check is proposed for the fresh seeds (to be frozen
   before any fresh-seed result).
8. **2026-11-02 01:25 (questioner):** CTs runs on Sonnet (seeds 10-12), not DeepSeek: with DeepSeek's first answers
   usable about half the time, CTs against CT would compare two ways of maintaining nothing. CTd runs on DeepSeek and
   Sonnet (Sonnet the informative half). Both CTs_FROZEN.md and E1D_FROZEN.md named three models; only the models
   change, no bar or parameter.
9. **2026-11-02 ~02:00, E1c (e1/e1c/E1C_FROZEN.md) as built (agreed with the questioner).** CTe numbers the request
   frame's detected people with the pipeline's own rule (`number_people`: merge, cut to max_people, left to right),
   not in ascending track id as the note said: it is the rule every people source goes through (E1b's D arm too),
   and the first match is then exact (identical detections at distance 0). Logged checks: the first-match rate
   (expected 1.0) and, per episode, live tracks left out of the answer by the merge or the max_people cut. Smoke
   (mock model, real detector): the first answer made plan, context and constraints calls only; 6 detected people
   in it; first-match rate 1.0.
11. **2026-11-02 03:40, one tail-latency cell (Sonnet, cafe_line2_moving seed 18).** CT, CT2 and CT2m got no first
    answer within the 92 s episode (answers [], 0 calls): the cell's first constraint-method call (made by CT) took
    longer than ~90 s with no error, and the cached answer replays that stored latency for CT2 and CT2m. R1 and Rm
    in the cell answered normally. Counted as failures, as frozen (latency is part of the method); the paired tables
    are reported with and without the cell.
12. **2026-11-02 ~16:30, BYSTANDERS_FROZEN.md as built (written before any judged bystander cell).**
    - **G is built only as Gl.** The early, plan-time G needs a solve without a stored answer, which `/resolve` has
      no path for. Only the timing-matched Gl is run: the chain (`serve._chain`: the tracked person nearest the
      answer's target point, then anyone within 1.0 m (strict) of anyone taken, repeated) is taken on the answer
      frame's tracks at the first `/resolve` and replaces the answer as `come_after` alone (the oracle path). It is
      re-solved with CT2's options and replace rules. P1 is read on Gl's chain; "G" in the reading rules means Gl.
    - **Solver exemption** (the note's required change): `Resolver.exempt()` = everyone with a track history who is
      not waiting (or was dropped or gone) and for whom `_moved` is false; `/resolve` passes it to `solve(...,
      exempt=)`, which unions it into `apart` in `_turn`. Empty unless `co_motion_s` is set.
    - **CT3-j** = `co_motion_follow: false`: the history, gate and exemption kept; `_follow_up` not run.
    - **Logging** (per `/resolve`, by track id): `last` (the solver's `_turn` last; None where no place in turn is
      used), `turn_joined`, `exempt`, `non_followers` {track id: t}; on the first: `answer_come_after` ([number,
      nearest answer-frame track, metres], so the 0.5 m rule is applied in scoring) and `chain` (Gl).
      `log["bystanders"]` = scenario indices with `bystander: true` (= track ids: tracks are the humans' indices).
    - **Cache-only replays:** `SGI_CACHE_ONLY=1` makes a cache miss a `vlm_error` with no call
      (`sgi.vlm.base`). Used for every Sonnet no-harm cell.
    - Resolver code at the judged runs differs from the hash in the note (216d2fda5ad626cf) only by `exempt()`,
      `track_of()`, `co_motion_follow` and `non_follower_t`; the hash at run time is recorded in the results.
    - **Gl where the answer leaves people out** (found in the seed-0 smoke, room: DeepSeek's context call said
      people do not matter, the answer was the standard goal and no session was stored, so Gl fell back to C1).
      The note's G needs only the target point, so Gl now sends `session_always: true` on `/goal`: the server keeps
      a session for any ok answer with a target (people and constraints empty when the answer had none), and Gl's
      chain replaces them at the first `/resolve` as before. Other arms do not send it and are unchanged.
    - **`queue_ids`** in every sample (the questioner's request): the scenario indices of the humans in the line at
      that tick, robot excluded. Logging only.
13. **2026-11-02, CT3b for BYSTANDERS_CONFIRM_FROZEN.md (d7dba68e), written before any confirmation cell.**
    `sgi.resolve.Resolver(co_motion_rejoin=True)`; sim arm `CT3b` = CT3's variant plus `"co_motion_rejoin": true`.
    Exact change (`src/sgi/resolve.py`, now sha256 30b7f44ce1d2af9f; default False, so CT2, CT3, CT3-j and Gl are
    unchanged):
    - `inputs()` stores its `t` as `self._now`;
    - `record()`, where a dropped person is taken back (`number in self.dropped` -> discarded), stores
      `self._readmitted_at[number] = self._now`;
    - `_follow_up()`, for each due check `(when, p, before)`: with `co_motion_rejoin`, skip it when
      `self._readmitted_at.get(p) > when - co_motion_s` (taken back after the advance that scheduled it).
    "Re-admitted or absorbed" in the note is this one path: a follow check is only ever scheduled for someone waiting
    at the advance, so being taken back after it always goes through the readmit branch. Test:
    `test_co_motion_rejoin_cancels_a_follow_check_made_before_someone_was_taken_back` (CT3 drops, CT3b does not).
14. **2026-11-02 ~17:45, after the confirmation was scored (post hoc diagnosis, logging only).** `Resolver.events`
    ((t, "advance", q, p, p_before, q_before, q_now) and (t, "drop", p, before, now)) and `/resolve`'s
    `co_motion_events` (people as track ids), logged per resolve by the sim. No behaviour change; resolve.py's hash
    changes. Used to replay sweet_spot s30 CT3 from cache (runs_debug/): the confirmation's one false drop comes from
    a wrong plan target (DeepSeek's plan targeted "the person at the counter" = a bystander at (16.88, 2.05), 3 m from
    the counter), so "advance towards the target" was measured towards the bystander pair; a returning member walking
    past them read as an advance and the true member behind was dropped.
