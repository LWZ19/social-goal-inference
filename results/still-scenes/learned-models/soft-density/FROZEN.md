# I04 phase 1d (plan step 7): frozen protocol

The I04 lead wrote this on 18 Oct 2026, after the format check below and before any density, fold fit or metric was computed. Its sha256 is in `FROZEN.sha256`. This file is never edited. Any change goes into a dated addendum with its own hash, written before the data it affects is scored. Paths are relative to `results/still-scenes/learned-models/soft-density/` unless they start with `phase1/` or `src/`.

**Pre-freeze format check** (`prefreeze/fmt_check.py`, log `prefreeze/fmt_check.log`, 4.4 s). It reads the 36 runs of I14 v0's cells through the harness. It computes no region membership, success, density or score.
- Goal-step records match I14 v0. Old-wording realistic: Qwen 143, GLM 141, DeepSeek 157. Watching: 36 each. New wording: Qwen 144, GLM 141, DeepSeek 151, Haiku 146, Sonnet 143, Gemini 135. That is 549 primary and 860 new-wording records.
- Every goal-step record has `target.world`, a people list, `constraints[].constraint`, 4 candidates with `pose_world` and keys, `pose_world`, the three solver flags and `solver.keys`. The third key is `turn_m`, `face_m` or absent, never both.
- **Layouts.** The primary realistic records lie in 13 layouts: `eval_00`–`03` and `05`–`13`. **`eval_04` holds no goal-step record in any cell**, old or new. The 6 watching layouts are `dev_seed{0,1,2}` × `eval_14_audience_3` and `eval_15_audience_5`, with 18 records each. New-wording records lie in 11 layouts, all among the 13.
- The recorded thresholds are equal in all 36 runs: `heading_degenerate_m` 0.05, `robot_radius_m` 0.30, `grid_resolution_m` 0.05, `candidate_count` 4, `standoff_m` 1.0, `person_radius_m` 0.25.
- **Reproduction.** For the first goal-step record of each run, `sgi.solver._over_floor` was called on resolve()'s inputs. Picks 1–5 gave the recorded place and runners-up at 0.0 m. The heading rule below gave the recorded headings to within 0.0 rad.

**Exposure (disclosed).**
- the analysis knows I14 v0's results on these same records: the tables, the 3 m disc's poor coverage at watching, the runner-up recoveries (AR 96% and 33% for DeepSeek) and Lift_room per cell.
- The prediction in section 9 uses I14's published T5 only.

## 1. Records and cells
- Records: goal-step records (`H.Record.solver_step`) of the con runs, read only through `research_harness`, exactly as in `results/still-scenes/metrics-and-statistics/metric-v0/METRIC_NOTE.md` "Records".
- **Primary cells:**
  - old-wording realistic (arms real, staffed and plain): s19 `qwen38_low`, s19 `glm53` and s20 `dsv41`;
  - watching: s19 `dev_seed0`–`2` `qwen38_low` and `glm53`, and s20 `dev_seed0`–`2` `dsv41`, situation `audience`.
- **Reported beside, not judged:** the new wording, realistic: s21 `qwen38_low`; s23 `glm53`, `dsv41`, `haiku55`, `sonnet55_low` and `gemini38_low`.
- Caution carried from I14: the people answers of s21 Qwen and of s23 GLM and DeepSeek are cache copies of s19/s20. A model's old and new rows are therefore not independent.
- Standard-path records are left out, and their count is reported.
- No fresh seed, no held-out situation and no held-out run is read.

## 2. Floor and keys
**Solver inputs.** For each record they are rebuilt as `H.resolve` does (`replay_solver.solve_again`):
- `GroundedTarget(result.target)`;
- `DetectedPerson` for each recorded person;
- `[use.constraint for use in result.constraints]`;
- `standard_goal(observation, target, th)`;
- the run's recorded thresholds.

**Floor and keys.** `cells, settled = sgi.solver._over_floor(...)` gives F(rec) = `cells`. These are the selectable cells, in the robot's frame and grid order, the same as I14's floor. `settled.keys` are the keys, rounded to the mm as the solver compares them. `settled.faced_xy` and `settled.picks` are kept too.

**Asserts.**
- `picks[0]`, taken to the world frame, is the recorded place within 1e-6 m.
- Choice: `picks[1:5]` are also checked against the recorded runners-up.
- Every mismatch is reported. A record that mismatches is left out of every table, and the count is given.

## 3. Per-cell outcome
**Replica.** I14 v0's vectorised replica of `sgi.evaluate.score` (`results/still-scenes/metrics-and-statistics/metric-v0/i14.py` `replica()`) is applied to every F cell, with the cell's heading from section 4. Per cell it gives:
- matched (record level: the package's `matched_target_id` for the recorded place; a wrong target makes every cell 0);
- reachable;
- in the matched target's acceptable region;
- each forbidden kind;
- and, for each holding region, the allowed heading and tolerance (asserted 45 deg).

**Validation.**
- **Sample.** For each record: 150 uniform F cells, up to 100 in-region cells, and up to 50 cells outside the region within 0.15 m of it. The draw uses `default_rng([20261018, crc32(run folder/scene/episode/trial)])`.
- **Against the package.** Each sampled cell is scored with `H.score_poses` and compared on success, in-region, heading_ok, reachable and forbidden.
- **Offset check (choice).** The heading offset itself is checked by calling `sgi.evaluate._heading_within` with region copies whose tolerance is set to 30 and 60 deg.
- **Places.** The 5 recorded places and every density place (section 7) are also scored by the package.
- **Disagreements.** Any disagreement stops the run before the fold fits. The fix goes into a hashed deviation note.

**Official top-1.** This is the package's verdict on the recorded place. It is asserted equal to the recorded success.

## 4. Heading
**Heading rule.** Each cell's heading is the solver's own rule, as `poses_at` in `solve()` applies it:
- the robot faces `settled.faced_xy`;
- it keeps the robot's yaw when that point is nearer than `heading_degenerate_m`;
- the result is taken to the world frame by `make_poses`.

**Heading density.** A von Mises distribution centred on that heading, with κ ∈ {4, 8, 16, 32, ∞}.

**Success probability.** P_succ(c) = matched × reachable × in_region × P_vM(θ within 45 deg of an allowed heading).
- **Numerics** (numpy only; the venv has no scipy). For each finite κ, the CDF of exp(κ(cos y − 1)) on [−π, π] is tabulated:
  - 2^16 + 1 points, trapezoid rule, normalised by the same rule's total;
  - read by linear interpolation;
  - checked against a table 4 times finer (maximum difference reported; it must be below 1e-6).
  - Arcs are measured relative to the cell's heading and merged on the circle.

## 5. Variants (mass over F(rec); the reported top-1 is always the lexicographic place)
**V1, full soft (primary).**
- E(c) = Σ_k γ^(5−k) key_k(c) over five fixed slots, in the solver's order: (1) `excluded_m`, (2) `excluded_total_m`, (3) `turn_m` or `face_m`, (4) `preference_m`, (5) `standard_m`. So the weights are γ⁴, γ³, γ², γ, 1. An absent slot-3 key counts 0, and `robot_m` is left out.
- p(c) ∝ exp(−(E(c) − min E)/T), computed with log-sum-exp in float64.
- Grid: γ ∈ {1, 3, 10, 30}; T = `numpy.geomspace(0.005, 5, 25)` m; κ as in section 4. That is 500 settings.

**V2, exclusions hard (secondary).**
- Kept cells are those at the least `excluded_m` and then, among those, at the least `excluded_total_m`. Choice: this is exactly `_first`'s rule, on the mm-rounded keys with a 1e-9 tolerance. A cell 1 mm above the least is not kept.
- Every other cell gets zero mass. On the kept cells the energy is V1's (slots 3–5 carry the same weights), so V2 is V1 restricted to the kept cells.
- Same grid as V1.

**V3, Gaussian control.**
- p(c) ∝ exp(−|c − c_top|² / (2σ²)) over F(rec), where c_top = `cells[picks[0]]`.
- σ = `numpy.geomspace(0.05, 3, 25)` m; κ as in section 4. That is 125 settings.

**V0, reference row (choice; reported, not judged).** A point mass on the lexicographic place, at κ = ∞. Its S is official top-1.

## 6. Fit and folds
**Objective.** One setting per variant per fold, shared by all models and sets. It maximises the mean, over the fold's training records, of ln(1e-3 + S(rec)), where S(rec) = Σ_c p(c) P_succ(c).

**Folds.** Leave one layout out over the 549 primary records.
- `eval_04`'s fold has nothing to score and is not fitted.

**Scoring.** Each primary record is scored with the setting of the fold that left out its layout. A new-wording record uses the fold of its own layout and is in no fit pool.

**Ties.** Settings within 1e-12 of the best objective are tied. Choice of order among tied settings: the largest T or σ, then the smallest κ, then the smallest γ.

**S table (choice, declared).** S(rec, setting) is computed once for every record and setting. Each fold's choice reads only its training rows. The 19 fold choices go to `fits.json`, hashed in `FITS.sha256`, before any held-out metric is computed.

## 7. Metrics (I14 v0 conventions; per cell and pooled over the primary cells; held-out settings)
**Success-based measures.**
- **Mean S**: the density's P(success).
- **Lift_room** = Σ S / Σ ρ_room. ρ_room is the share of F cells inside the matched target's acceptable region (0 for a wrong target). The ratio is of sums, as in I14.
- **Lift_disc** = Σ S / Σ ρ_disc, using I14's disc: F cells within 3 m (+1e-9) of `result.target.world`.
- **Held-out log score**: the mean of ln(1e-3 + S).

**Mass measures.**
- **In-region mass** (choice: as I14's "in-region alone", i.e. matched × polygon, with heading and reachability aside).
- **Mass in each forbidden kind present.** Over the records whose episode has that kind, with the top place's share and the room share of F cells in the kind beside it.

**Density hit@k** (k = 1–5).
- Place 1 is the lexicographic place.
- Each next place is the highest-density cell that is at least 2 × `robot_radius_m` (−1e-9) from every earlier place, as the solver's runner-up rule measures it.
- Choice:
  - cells are ranked by unnormalised log density (V1 and V2 by ascending E, V3 by ascending distance to c_top), so underflow cannot create ties;
  - ties go to the lowest grid index;
  - V2 places come only from kept cells, and when none is left the list stops and hit@k carries forward.
- Each place takes the section 4 heading and is scored by the replica (and the package, section 3). κ does not enter.
- The lexicographic hit@1–5 (recorded places, package) is given beside it and must reproduce I14 S1: 134, 99 and 123 (old realistic) and 35, 29 and 33 (watching), with top-1 of 134, 96, 109, 32, 23 and 23. If it does not, the run stops.

**Concentration share** c(rec): the mass on cells within 0.3 m (+1e-9) of c_top. Its mean is reported, with the share of records where c > 0.9.

**Confidence AUROC** (section 8).

**Intervals.**
- Choice: a layout-cluster bootstrap, 2000 draws, `default_rng(20261018)`, stratified (the 13 realistic and 6 watching layouts are resampled separately), with percentile 95% intervals.
- They are given for V1 − V3, pooled and per cell (6 clusters at watching: crude), on the mean log score and on Lift_room. Choice: also V2 − V3.

**Fitted settings.** Reported per fold.

## 8. Confidence
**Score.** c(rec) as in section 7, under V1's held-out setting. It predicts official top-1 success: higher c means more confident. There is no sign flipping, and ties count half.

**Comparisons beside it:**
- V3's c;
- each recorded solver flag (`exclusion_broken`, `far_from_target`, `constraint_unusable`) as a 0/1 score. Choice: the score is 1 − flag, because a raised flag predicts failure.

**Reporting.** AUROC per primary cell; the pooled value is descriptive.

## 9. Criteria (verbatim REPORT I04, with the readings fixed here)
**Success** needs all of:
- (a) "Lift at least 10×": V1's Lift_room ≥ 10, pooled over the primary cells (each cell reported);
- (b) "beats V3": V1's pooled held-out mean log score is strictly higher than V3's (the interval and Lift_room are reported);
- (c) "hit@5 within 2 of the lexicographic hit@5" in every primary cell. Choice: this is read one-sided, V1 density hit@5 ≥ lexicographic hit@5 − 2. The literal two-sided outcome is reported beside it;
- (d) "AUROC at least 0.70 for 2 models": at least 2 of the 3 old-wording realistic cells.

**Kill** fires on either of:
- V1 does not beat V3, as in (b);
- c_V1(rec) > 0.9 in more than 80% of the 549 primary records.

**Precedence (choice).** A kill overrides success. Any other outcome is **neither**.

**Better than expected** (reported, not a verdict):
- V1 also beats V3 on pooled Lift_room;
- AUROC ≥ 0.75 in at least 2 of the 3 old-wording realistic cells (choice: the same population as (d)).

**V2.** It is judged against the same bars as a secondary result. V0 and the new wording are not judged.

**Prediction** (inference from I14 T5; no density computed):
- Over the primary cells, top-1 is 417 successes and Σ ρ_room is about 46.85, so the hard top place has a pooled Lift_room of about 8.9. Any-of-5 is 453.
- (a) needs Σ S ≥ 468.5 of 549. The right-target records number 513, so that means a mean S of about 0.91 over them, against the top place's 0.81.
- **(a) is therefore expected to fail on the pooled reading whatever the variant.** The watching cells' room share (0.15–0.16) drives this.
- The realistic cells (top-1 Lift_room 10.1–13.3) can pass per cell only if the density stays near the top place.

## 10. Order and honesty
1. This note and `FROZEN.sha256`.
2. **Per-record cache** (`cache/`):
   - cells, the five slot keys, the cell headings, the per-cell outcomes and P_vM per κ, the disc and forbidden masks;
   - the reproduction asserts, the replica validation and the S1 reproduction, logged.
3. **The S table**, then the fold fits, then `fits.json` and `FITS.sha256`.
4. **Held-out scoring and the tables**, each computed once.

**Rules.**
- No method reads a situation, scene, role or group label. Labels only split folds and tabulate.
- At most 2 CPU processes; numpy only.
- No provider or network call. Nothing in `src/`, `tests/`, `docs/` or `configs/` changes.
- Every number has its n. Anything not declared here is labelled post hoc, and anything chosen after seeing results on these data is labelled data-informed.

**What the scores mean.** Every score measures agreement with the project's own reference regions and heading rule, not social correctness.
