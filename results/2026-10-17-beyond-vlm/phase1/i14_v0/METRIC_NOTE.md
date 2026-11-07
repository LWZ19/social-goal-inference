# I14 metric note, v0 (frozen 2026-10-17, before any metric was computed)

## Records

- **Goal-step records**: `H.Record.solver_step` (source `constraints`, pose recorded, people and constraints
  recorded). Only these carry runners-up. Standard-path records are left out of every set metric; their count is
  reported.
- **Cells (model x set)**:
  - New wording, realistic (arms real, staffed, plain): s21 qwen38_low; s23 glm53, dsv41, haiku55, sonnet55_low,
    gemini38_low.
  - Old wording, realistic: s19 qwen38_low, s19 glm53, s20 dsv41.
  - Watching: s19 dev_seed0-2 qwen38_low and glm53, s20 dev_seed0-2 dsv41, situation `audience`, goal steps.
- **Primary cells for the criteria**: the brief's cells, i.e. the old-wording realistic cells (Qwen s19 n 143, GLM
  s19 n 141, DeepSeek s20 n 157) and the watching cells (n 36 each). The new wording is reported beside them.
- Cautions carried: s21 Qwen and s23 GLM/DeepSeek C people answers are cache copies of s19/s20 (not independent);
  the old and new rows of a model are therefore not two independent results.

## Places

- Place 1 = `result.pose_world` (the solver's top place). Places 2-5 = `result.candidates[i].pose_world` in
  recorded order (the runners-up), each with its recorded heading (the solver's own heading rule).
- **hit@k** (k = 1..5): any of places 1..k succeeds under the package scorer (`H.score_poses`, the record's
  recorded target, the run's recorded thresholds). hit@1 is the official top-1.
- **P(success)@k**: mean success over places 1..k (uniform mass: the solver gives no probabilities).
  **P(success) of the set** = P(success)@5.

## Floor: lift denominator and baseline floor

- **F(rec)** = `sgi.solver.floor(observation, the record's own people list, the run's recorded
  thresholds).selectable`: exactly the cells C's solver ranked for this record (0.05 m cells in the robot's frame,
  centres from `Floor.centres()`). Not the ground-truth grid, so the baselines get no oracle people. (The snap
  replay, `results/still-scenes/depth-fix/first-check/snap.py`, used the same floor model.) Check: all 5 recorded places must lie on F cells.
- **Disc**: the F cells whose centre is within 3.0 m (+1e-9) of the recorded target's lifted point
  (`result.target.world` x, y), the point the solver was given, not the true target.
- **Whole room**: all F cells.
- Success of any pose keeps the package's own components, including ground-truth reachability.

## Baseline poses

- **Heading rule**: face the recorded target point, `bearing(cell, target)`, the robot's yaw when closer than
  `heading_degenerate_m` (0.05 m): the R/E rule of `sgi.stages.goal._poses`.
- **Per-cell outcome**: success = target matched (record level) and reachable and inside the matched target's
  acceptable region and heading within tolerance; plus in-region alone, and each forbidden kind. Computed by a
  vectorised replica of `sgi.evaluate.score` (same formulas, same order), validated on every record against
  `H.score_poses` on a sample of disc cells (150 uniform, up to 100 inside the region, up to 50 just outside it).
  Every disagreement is reported.
- **Draws**: 1,000 per record, `numpy.random.default_rng([20261017, crc32(run folder/scene/episode/trial)])`.
  - **R5**, unconditional random-5: 5 distinct disc cells, uniform, without replacement.
  - **F5**, unconditional farthest-point-5: first cell uniform in the disc; each next one the disc cell farthest
    (largest minimum Euclidean distance) from those already picked; ties to the lowest grid index.
  - **AR**, anchored random: place 1 plus 4 distinct disc cells, uniform, excluding place 1's cell.
  - **AF**, anchored farthest: place 1 plus 4 farthest-point picks in the disc, starting from place 1
    (deterministic: all draws coincide).
  - random-k, farthest-k and the anchored variants for k < 5: the first k of each sequence.
  - The exact hypergeometric expectation of R5 and AR is reported as a check on the draws.

## Lift

- rho_disc(rec) = share of disc cells inside the matched target's acceptable region; rho_room(rec) = the same over
  all F cells. A wrong target gives 0 (and its top-1 is 0).
- **Lift_disc** = sum of top-1 successes / sum rho_disc; **Lift_room** = sum top-1 / sum rho_room (ratios of sums
  over the goal-step records; the same as over right-target records). **Chance ratio** = sum rho_disc / sum
  rho_room.
- Secondary: sigma_disc = share of disc cells whose face-target pose fully succeeds (45 deg); Lift over it.

## Other measures

- **Forbidden mass**, per kind present in the episode (`ahead_in_line`, `group_space`, `audience_view`): over the
  records whose episode has that kind, the mean share of the 5 places inside it; beside it the top-1 share and R5's
  expectation (the disc share of cells inside it).
- **Jitter**: each recorded place moved 5 cm in 8 world directions (0, 45, ..., 315 deg), heading unchanged,
  scored by the package. Top-1 success averaged over the 8 directions; top-1 flip rate = share of records where
  any direction changes the top-1 verdict; hit@5 with all 5 places moved by the same vector, mean and minimum
  over the 8 directions.
- **Heading tolerance**: the region's tolerance (45 deg in every episode) replaced by tau in {30, 45, 60}:
  success_tau = matched and reachable and in region and |wrap(theta - allowed)| <= tau. 45 is the scorer's;
  30 is the mentor's. For top-1, hit@5, P(success)@5, R5 and AR.
- In-region alone (heading aside) for R5 per situation, for the skeptic's guess (85-95% at group rings).

## Pre-registered criteria (copied from REPORT I14), with the operational reading fixed here

**Success**
- **S1** Exact reproduction: top-1 / any-of-5 = 134 / 134 of 143 (Qwen s19), 96 / 99 of 141 (GLM s19),
  109 / 123 of 157 (DeepSeek s20) on realistic goal steps; 32 / 35, 23 / 29, 23 / 33 of 36 watching.
- **S2** The anchored random-4 recovers at most 50% of the +14 (DeepSeek realistic) and +10 (watching) runner-up
  gains. Recovery = sum(E[hit_AR@5] - hit@1) / sum(hit@5 - hit@1), on both DeepSeek cells.
- **S3** The solver beats the anchored farthest-4: solver hit@5 > AF hit@5 on both DeepSeek cells.
- **S4** Top-1 lift is at least 3x disc chance (Lift_disc >= 3), and disc chance is at least 2x the whole-room
  chance (chance ratio >= 2). Met when every primary cell meets both.

**Kill** (the set story, not the code)
- **K1** The anchored baselines recover at least 80% on both DeepSeek cells. Evaluated for AR and AF separately;
  fires if either one recovers >= 80% on both cells.
- **K2** The unconditional random-5 is at least the solver's any-of-5 for GLM or DeepSeek: expected R5 hits >=
  hit@5 count on the old-wording realistic cell of GLM or DeepSeek (watching and new wording reported beside).

## Out of scope for v0

- I46's statistics (layout sign-flip tests, ICC, power): re-run already in RERUN.md R05 and R34.
- A proper score: needs human points (I35).

---

## Addendum after computing (2026-10-17; not part of the frozen note above)

- Nothing above was changed after computing. Results: `tables.txt` (T1-T10 as frozen; T11-T12 descriptive) and
  `posthoc_room.txt`.
- 6 records (wrong target lifted far off; all at eval_12, GLM and DeepSeek) have no disc cell, so no draws. Their top-1
  and any-of-5 are 0, so leaving them out of the baseline sums changes no comparison.
- Added afterwards, labelled as such: (a) a layout-clustered bootstrap of the recovery shares (T11, descriptive);
  (b) a post-hoc sensitivity drawing R5 and AR over the whole room instead of the 3 m disc (T12 in-region,
  `posthoc_room.txt` in success terms). Reason: the frozen disc leaves out about 90% of the watching region and
  about 16% of the realistic one (T1), which biases the disc baselines, low at watching. No new parameter: the
  whole room is the frozen lift denominator.
