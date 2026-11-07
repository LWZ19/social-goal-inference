# C5 (I52) flow-matching pilot: frozen protocol (exploratory)

The pilot lead wrote this on 18 Oct 2026, after the format check below. No raster, label, model or evaluation metric existed yet. Its sha256 is in `FROZEN.sha256`. This file is never edited. Any change goes into a dated addendum with its own hash, written before the data it affects is built or scored. Paths are relative to `phase2/flow_pilot/` unless they start with `phase1`, `phase2`, `src/` or `tests/`.

- Gate 2 is still running in `phase1d/c5_gates/`. Its outcome is unknown now; the results will cite it.
- Gate 3: I04's V1 and V3 rows exist on the I14 harness (`phase1d/i04/`). Their verification is still running.

- the data: "fresh seeds" become seeds 110–199;
- the raster: "10 cm cells" become 64 × 64 of them;
- the model: "a small U-Net or transformer" becomes a CNN encoder with a FiLM MLP;
- the sampler: "10–20 ODE steps" become 16 Euler steps;
- the projection rule, the metrics and the readings of the criteria.

Anything beyond the rulings is marked (choice).

**Pre-freeze format check** (`prefreeze/fmt_check.py`, `fmt_check2.py`, logs beside them). It loaded 16 scenes of seed 100 and the first goal-step record of 4 primary runs. It computed no success, score or metric. It showed:
- **Episodes.** Every scene checked has one target and two episodes (e0, e1) with the same regions, so the two give identical rasters.
- **Headings.** An episode's regions share one heading rule: a fixed yaw (waiting_line) or a face point (the rest), with a 45 deg tolerance.
- **Seen face.** `_seen_face` gives None at the anchor's ground point (z = 0) in every scene.
  - At the 3-D anchor and at `oracle_target` it gives each place target's front, within 1 deg of front_yaw. Some person targets get a fitted face, and the object gets None.
  - The oracle point lies 0.00–0.15 m from the anchor's ground point; persons and objects account for 0.10–0.15 m.
- **Robot.** The robot starts 4.2–6.6 m from the target point in every scene and record checked.
- **Clipping.** The share of each region that falls inside the raster:
  - 1.00 for no_one, waiting lines of 1–3 and groups;
  - 0.88 for audience_3;
  - 0.56–0.62 for audience_4–5 and waiting_line_4;
  - **0.00 for waiting_line_6** (its region lies 4.3–5.0 m out).
- **Records.** `_over_floor` reproduces the recorded place at 0.0 m. The thresholds are the defaults. 1 of 530 listed people has no yaw.

These facts shaped the seen-face target, the label exclusion and section 10. Nothing else is data-informed.

## 1. Data and units (rulings 2, 3)
**Data.**
- Train: seeds 110–189.
- Validation, which is also the in-distribution test: seeds 190–199.
- All four development situations are included.
- The seed agent writes the seeds under `phase2/seeds/placeholder/`. This pilot never generates one.
- Seed 100 served the format check only. Seeds 3–109 are not used.

**Loading and guards.**
- Loading follows `phase1b/i03/stageA/fitdata.py`: `H.guard_scene`, then `load_scene`, then regions from `sgi.gt.derive.prepare` under the default thresholds.
- For every scene it is asserted that its situations are development situations and that it is no held-out row (group_photo, lift_wait, eval_16–19).

**Units.**
- A unit is one (scene, episode) with a non-empty acceptable region; the rest are counted and left out.
- Episodes with several targets are also left out and counted (choice).
- e0 and e1 are both kept, and every table gives both n episodes and n scenes.
- An episode with no free region cell inside the raster (section 3) gets no labels (choice). It is left out of training and of the validation loss, and counted by situation and row. It stays an evaluation unit in 6(i), where the flow fails it.

The development records and scenes are for evaluation only, and are read only through `research_harness`.

**Geometry.**
- 64 × 64 cells of 0.10 m.
- The origin o is the target point:
  - training: the true target's anchor ground point;
  - evaluation: `result.target.world` (x, y).
- The x axis runs from the robot's start to o; y is x turned +90 deg.
- Cell [r, c] is centred at ((c − 31.5)·0.1, (r − 31.5)·0.1) m (choice).

**Channels** (float32):
1. **Floor.** `sgi.solver.floor(obs, people, th).selectable`, read at each cell centre from the 0.05 m cell that holds it (`Floor.index`), and 0 off the grid (choice).
   - Training: the people are the true people `sgi.run.OraclePeople` keeps (visible by `is_visible`, with a projectable mark; choice), under the default thresholds.
   - Evaluation: the setting's people, under the run's recorded thresholds.
2. **People.** A disc of 0.25 m (+1e-9) round each person.
3. **Facing, cos.** cos of each person's yaw relative to the raster, inside their disc.
4. **Facing, sin.** The matching sin.
   - Both are 0 when the yaw is None.
   - Where discs overlap, the nearest person's values hold (choice).
5. **Target.** The target point as a disc of 0.15 m.
6. **Seen face.** The cells holding 51 points spaced 0.01 m along [o, o + 0.5 m · n], where n is `_seen_face`'s front normal in the raster frame.
   - The channel is empty when `_seen_face` gives None.
   - Training (choice, forced by the format check): `_seen_face` is given `oracle_target(obs, target, th)`. If that lift fails, it is given a target at the 3-D anchor, counted.
   - Evaluation: the record's GroundedTarget.
7. **Robot.** The start as a disc of 0.3 m.

Nothing else goes in: no situation, group, role, scene name, region, forbidden region or request text.

- **Region cells:** raster cells that are free in channel 1 and whose centre lies in the union of the target's acceptable polygons (`point_in_polygon`).
- **Poses:** 32 per episode.
  - Cells are drawn uniformly with replacement (choice).
  - Each pose is jittered uniformly within [−0.05, 0.05)² m.
  - The generator is `default_rng([20261018, crc32('seed<k>/<split>/<scene>/<episode>')])` (choice).
- **Heading rule**, as `sgi.evaluate._heading_within` evaluates it:
  - The first region, in list order, that holds the cell centre gives the rule.
  - The heading is `heading.yaw` when it is set. Otherwise it is `bearing(jittered pose, face_point)`.
  - The raster's yaw is subtracted, and the heading is stored as (cos, sin).
  - All regions of an episode are asserted to share one Heading; exceptions are counted.
- **Label:** (x/3.2, y/3.2, cos, sin), with x and y in metres.
- **Self-check** (training data only): the 6(i) replica scores every label pose. The shares in the region, within the heading tolerance and reachable are reported.

**Architecture.** The widths, norms, activations and the sum below are choices.
- **Encoder:** 4 blocks of conv 3×3 stride 2 → GroupNorm(8) → SiLU, with channels 7→32→64→128→256 (64² → 4²). Then flatten and Linear(4096→256).
- **Time embedding:** sinusoidal, with 64 frequencies exp(−ln(10⁴)·k/64) applied to 1000·t (sin and cos). Then an MLP 128→256→256 with SiLU.
- **Conditioning:** c = encoder embedding + time embedding.
- **Velocity MLP** over the 4-d pose: Linear(4→256), then 3 layers h ← SiLU(LayerNorm(Wh)·(1 + γ) + β). Each layer's (γ, β) = Linear(256→512)(SiLU(c)). Then Linear(256→4).
- PyTorch default initialisation.

**Loss.** Rectified flow: x_t = (1 − t)z + t·x, with z ~ N(0, I₄) and t ~ U(0, 1); loss = mean |v(x_t, t, raster) − (x − z)|².

**Optimisation.**
- AdamW, lr 1e-3, PyTorch's default betas, weight decay 0.01 (choice).
- Cosine decay to 0 over 30 epochs of steps, with no warm-up (choice).
- Batches of 256 label poses with their rasters. An epoch is every label pose once, shuffled (choice).
- fp32, `torch.manual_seed(0)`, no augmentation (choice).

**Main model.**
- Trained on all four situations, for at most 30 epochs or 40 min of wall time counted from the first step, validation passes included.
- The validation loss is taken after each epoch over all validation label poses. Their z and t are drawn once (generator seed 1) and then kept fixed (choice).
- The kept checkpoint is the one with the lowest validation loss, ties going to the earlier epoch (choice).
- No other early stopping and no hyperparameter search.

**Leave-one-situation-out (LOSO) models.**
- Three models, leaving out waiting_line, conversation_group or audience in turn.
- The same recipe, capped at 15 min each.
- Their training and validation loss use only their training situations, so the left-out situation enters no choice (choice).

**Checkpoints.** Their sha256 go into `CHECKPOINTS.sha256` before any evaluation sample is drawn.

## 5. Sampling and projection
**Sampling.**
- 64 samples per unit: z ~ N(0, I₄), then 16 Euler steps x ← x + v(x, k/16)/16 for k = 0..15.
- Position: 3.2·(x₁, x₂), taken to the world frame.
- Heading: atan2(x₄, x₃) plus the raster's yaw.
- Seed: crc32 of the unit key. A record's two settings share it (choice).

**Projection** (development records).
- Each sample snaps to the nearest F cell centre: Euclidean distance in the world frame, ties to the lowest grid index. Its heading is kept (choice).
- A sample is **valid** when its cell's `excluded_m` ≤ the lexicographic place's `excluded_m` + 0.001 m (+1e-9).
- The keys come from `sgi.solver._over_floor` on the setting's inputs, rounded to the mm as the solver rounds them.

**Density and top pose.**
- Each sample's density is Σ_j exp(−|p_i − p_j|²/(2·0.15²)).
  - Development records: the sum runs over the valid snapped positions.
  - Validation seeds: it runs over all 64 raw positions (choice).
- The top pose is the densest sample, with its own heading; ties go to the lowest index.
- On a development record with no valid sample, the solver's place is used instead (a fallback, counted).

The projection runs in numpy inside the sidecar, so the latency covers sampling and projection in one process (choice).

**(i) Validation seeds 190–199.** Per situation and pooled, with n episodes and n scenes.
- **Success** means all of:
  - inside an acceptable region of the target;
  - reachable (joined to the robot on `robot_grid(truth)`);
  - heading within 45 deg of the region's heading.
- **Scorer:** I14's vectorised replica (`phase1/i14_v0/i14.py`). It is checked against `sgi.evaluate.score` on a GoalResult with the oracle target, for the top pose and 8 random samples per episode. Any disagreement stops the scoring (choice).
- **Metrics:**
  - the share of the 64 samples that succeed;
  - region coverage: the share of region cells with a raw sample within 0.2 m (+1e-9). Choice: region cells are cells of the 0.10 m grid, extended beyond the raster, whose centre lies in the region and on the floor by channel 1's rule. Clipping therefore counts against the flow;
  - the top pose's success.
- **Models:**
  - the main model on every episode;
  - each LOSO model on its left-out situation, beside the main model on the same episodes.
- **Baselines (top-1):** `sgi.solver.solve` with the oracle people, the oracle target and `standard_goal`. Where the oracle target's lift fails, a target at the 3-D anchor is used, counted.
  - **B0:** no constraints (the solver alone).
  - **B1:** reference constraints. `src/` has none. The source is `tests/test_reference_constraints.reference_constraints` as copied in `phase1/i01_repairs/repairs.py`, written against the oracle people.
  - B1 is I01's P5/ALL configuration, not R5: generated scenes have no recorded people. It reads the situation and the true group, so it is circular and serves only as a reference.

**(ii) Development records, setting (b): each model's recorded people.**
- **Records:** goal-step records (`H.Record.solver_step`), 549 in all.
  - Realistic (arms real, staffed and plain): s19 qwen38_low, s19 glm53 and s20 dsv41, with n 143, 141 and 157.
  - Watching (audience): s19 dev_seed0–2 qwen38_low and glm53, and s20 dev_seed0–2 dsv41, with 36 each.
- **Inputs:** the observation, the recorded people (with recorded yaws) and `result.target.world`. F(rec) and the keys come from `_over_floor` with the recorded target, people and constraints. `picks[0]` is asserted to reproduce the recorded place within 1e-6 m.
- **Metrics,** by the package scorer `H.score_poses`, per cell, per model pooled, and over all 549:
  - **top-1 success;**
  - **P(success):** the mean success of the valid snapped samples. Under the fallback it is the success of the solver's place (choice). Beside it are the mean over all 64 snapped samples and the valid share;
  - **Lift_room** = Σ P(success) / Σ ρ_room, as I04 defines it. ρ_room is the share of F(rec) cells inside the matched target's region (0 for a wrong target), asserted equal to I04's within 1e-9. I14's Σ top-1 / Σ ρ_room is reported beside it;
  - **hit@5:** place 1 is the top pose. Each next place is the densest valid sample at least 2 × robot_radius_m (−1e-9) from every earlier place. When none is left, the list stops and hit@k carries forward. Under the fallback the list is the recorded places 1–5 (choice);
  - **latency** on the 4060, measured in the sidecar.
    - It runs from the arrays in host memory to the top pose and the hit@5 places: encoder, 16 steps over 64 samples, copy back, snap, validity, density and picks.
    - 10 untimed warm-up records, then each record once.
    - The median and p90 over the 549 are reported, with nvidia-smi memory and utilisation logged.
    - Reported apart (choice): painting the raster, `_over_floor` (solver work C already does), and the hand-off between the two environments.

**(iii) Setting (a): true people (upper bound).** As (ii), with these changes:
- The people and constraints are I01 P's, built by `repairs.py`'s own code: the visible true people at their true positions and yaws, and the recorded constraints with dropped numbers removed.
- F and ρ_room come from the floor with those people.
- The fallback is P's place and its runners-up.
- The comparison is P's own solver top-1. I01 has no watching rows, so P is recomputed for all 549 records. On realistic records it must reproduce I01's P verdicts (`phase1/i01_repairs/rows_old_*.jsonl`); any mismatch is reported.

**(iv) Comparisons.**
- Recorded C top-1 for (ii). By I14 S1 this is 134/143, 96/141 and 109/157 realistic, and 32, 23 and 23 of 36 watching.
- P's top-1 for (iii).
- I04's V1 and V3 per record, from `phase1d/i04/rows_old_*` and `rows_watch_*` (held-out settings), matched to the 549 records by key:
  - P(success) = mean S;
  - Lift_room = Σ S / Σ ρ_room;
  - hit@5 = `hit_pkg[4]`.
- The I04 rows are used once their verification has finished (a README or a verification summary in `phase1d/i04/`). Until then: poll every 10 min for at most 2 h. After that, report without them and say so.

**Secondary, deciding nothing** (choice):
- a layout-cluster bootstrap of flow(b) − C top-1, per model and pooled: 2,000 draws, `default_rng(20261018)`, realistic and watching strata apart, as in I04;
- the raw flow top pose on development records (density over all 64 unsnapped samples);
- each LOSO model on its left-out situation's development records, setting (b);
- tables by situation.

**Terms.**
- Models: Qwen 3.8, GLM-5.3-Flash and DeepSeek V4.1 Flash.
- "Pooled per model": over that model's realistic and watching cells, n 179, 177 and 193.
- "Points": percentage points of goal-step records.
- "Pooled": all 549 records.

**Success** needs all three:
- **S1** "(b) within 3 points of C pooled for 2 of 3 gateway models": flow(b) top-1 ≥ C − 3 for at least 2 models. Choice: read one-sided; the two-sided result is reported beside it.
- **S2** "beats I04 V1 on lift and hit@5": pooled, the flow's Lift_room and its hit@5 are each strictly above V1's.
- **S3** "sampling plus projection under 0.5 s": median latency below 0.5 s.

**Kill** fires on any one:
- **K1** "(a) more than 10 points below the solver with true people": flow(a) top-1 < P − 10 for every model.
- **K2** "(b) more than 10 points below C for every model": flow(b) top-1 < C − 10 for every model.
- **K3** "no better than I04 V3": pooled, flow(b)'s Lift_room ≤ V3's and its P(success) ≤ V3's mean S.

**Precedence** (choice): a kill overrides success; anything else is "neither". C5's "holds on fresh seeds" is reported from (i), not judged.

**Order.**
1. This note and its hash.
2. Rasters and labels for seeds 110–199 (project venv, at most 2 processes), with counts and the self-check; then the development arrays for (a) and (b), with the reproduction asserts.
3. The main model, then the three LOSO models; then `CHECKPOINTS.sha256`.
4. Sampling in the sidecar, then scoring in the project venv. Each table is computed once.
5. A README.

- It never imports sgi and installs nothing.
- It reads and writes only .npz and .json files under `phase2/flow_pilot/`.

**GPU.**
- One GPU process at a time.
- A job starts only when nvidia-smi shows at least 3,000 MiB free; otherwise it polls every 5 min and logs the wait.
- Each job logs the GPU, driver, torch and CUDA versions and peak memory.

**Rules.**
- No provider or network call.
- Nothing changes in `src/`, `tests/`, `docs/`, `configs/`, REPORT.md or earlier phases.
- No method reads a situation, scene, role or group label. Labels only split data and tabulate; B1 is a labelled reference.
- A code fix made after any number was read goes into a hashed deviation note.
- Every number has its n. Anything undeclared here is post hoc, and anything chosen after seeing these results is data-informed.

- **Exploratory.**
  - Gate 1 was overridden.
  - Training and test layouts share the generator's templates (22 configurations), so (i) measures fit to the generator.
  - The LOSO models are the only transfer test, and that test is partly ill-posed: each situation has its own rule.
  - C5's own held-out set, I46 (fresh seeds plus a situation written blind), does not exist yet.
- **Circular labels.** The labels are the project's reference rules. Agreement with them measures fit to those rules, not social correctness.
- **Optimistic test.** The validation seeds both pick the checkpoint (by flow loss) and form test (i), so (i) is mildly optimistic.
- **Unequal fitting.** I04's V1 and V3 settings were fitted on the development records themselves (leave one layout out). The flow never sees a development record.
- **Known handicaps in the rulings:**
  - the clipping (section 10);
  - a 0.00–0.15 m shift between the training centre (anchor ground point) and the evaluation centre (a lifted surface point);
  - channel 7 is almost always empty.

- **Raster extent.** At ±3.2 m:
  - waiting_line_6 has no region inside the raster (seed 100);
  - audience_4–5 and waiting_line_4 lose about 40% of theirs.
