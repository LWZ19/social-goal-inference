# Phase 2: the C5 (I52) flow-matching pilot, exploratory results (2026-10-18)

How it was made:
- **Protocol.** The pilot lead wrote `FROZEN.md` (sha256 `a60f1d62…767f`) at 01:18:12 and hashed it at 01:18:17, after a format check (`prefreeze/`: 16 scenes of seed 100 and the first goal-step record of 4 primary runs; no metric) and before any raster, label or model existed.
  - `DEVIATIONS.md` (hashed at 04:17:55: after `CHECKPOINTS.sha256` at 04:17:45, before the first evaluation sample at 04:17:57) records no deviation, 13 clarifications of mechanics, and a 15 s smoke run of the training code on the real data, deleted, from which nothing was chosen.
- **Two environments.** Rasters, labels, the solver's keys, scoring and tables ran in the torch-free project venv, reading recorded runs and scenes only through the step-0 harness. Training and sampling ran in the separate torch environment, used read-only as a sidecar (`-I`, `HF_HUB_OFFLINE=1`). The sidecar never imported the package and read and wrote only `.npz` and `.json` files in this folder. Its `pip freeze` was unchanged afterwards (verifier).
- **Evaluation inputs.** The validation seeds 190–199 (in distribution). The 549 goal-step records of the six primary cells (old wording, realistic and watching), read through the harness. I01's P (true people under the recorded constraints), recomputed with `phase1/i01_repairs/repairs.py`'s own code. I04's V1 and V3 rows from `phase1d/i04/`.
- **Verifier.** An independent verifier rebuilt the inputs with its own code, loaded the checkpoints into its own model class, retrained the main model, re-sampled with the protocol seeds and with independent seeds, re-scored with the package scorer and re-measured latency in its own sidecar. Where the compute agent and the verifier differ, this page uses the verifier's final value; where the verifier could not settle a point, both are given (section 6).

Paths below are relative to this folder (`phase2/flow_pilot/` in the research folder).
- Protocol: `FROZEN.md`, `FROZEN.sha256`, `DEVIATIONS.md`, `DEVIATIONS.sha256`, `CHECKPOINTS.sha256`, `prefreeze/` (the format check).
- Code (`code/`): `common.py`, `build_seeds.py`, `assemble.py`, `png_check.py`, `build_dev.py` (project venv); `flow.py`, `run_train.sh` (sidecar); `score_val.py`, `score_dev.py`, `tables.py`, `figs.py` (project venv).
- Data (`data/`, about 1 GB): `seeds/` (per-seed rasters and labels), `train.npz`, `val.npz`, `dev.npz`, `dev.json`, `manifest.json`, and `png/` (10 visual checks of rasters and labels).
- Models (`models/<name>/`): `best.npz`, `curve.json`, `env.json` for `main` and the three leave-one-situation-out (LOSO) models.
- Samples and results: `samples/` (the sidecar's samples and `latency.json`), `rows_val_*.jsonl`, `rows_{a,b}_{old,watch}_<model>.jsonl`, `rows_b_*_loso_*.jsonl`, `tables.txt` (tables D1–D2 and T1–T11), `summary.json`, `logs/`, `figs/`.
- Post hoc (written after the tables; decides nothing): `posthoc/by_row.py`/`.txt`, `posthoc/line_geometry.py`/`.txt`.
- Verifier: `verify/` (`v_*.py`, `out/`, `logs/`, `data/`, `samples/`).
- Seeds: `../seeds/` (`gen_placeholder.py`, `manifest_placeholder.json`, `placeholder/seed110`–`seed199`, the generation, verify and reproduction logs).

## 0. Verdicts at a glance (exploratory)

| Analysis | Outcome under C5's criteria (exploratory) | One-line reading (inference) |
|---|---|---|
| **C5 pilot overall** (FROZEN section 7) | **Kill.** K1, K2 and K3 fire; S1 and S2 not met; S3 met. A kill overrides success. The verifier gets the same verdicts with independent sampling seeds | On the development records the learned sampler loses to the solver it was meant to match, for every model. |
| (i) Validation seeds 190–199, in distribution (reported, not judged) | Top-1 500/520 (96.2%); 0.911 of all samples succeed; solver alone (B0) 300/520, solver with reference constraints (B1, circular) 500/520 | It fits the generator's own layouts as well as the reference constraints do; its 20 misses are clipped by the raster. |
| (i) Leave one situation out (reported, not judged) | Left-out lines 36/140, groups 0/100, audiences 0/60 (main model 120, 100, 60) | No transfer to an unseen situation. |
| (ii) Setting (b), each model's recorded people | 312/549 (56.8%) against recorded C's 417 (76.0%): Qwen −26.8, GLM −15.8, DeepSeek −15.0 pp | The loss is at lines (83/261 against 196). The watching records, where it wins, are generator rows (partly in distribution). |
| (iii) Setting (a), true people | 320/549 (58.3%) against I01 P's 461 (84.0%) | Better people barely help the flow (+8 records; +44 for the solver). |
| (iv) Against I04, pooled | Lift_room 6.66 against V1 7.85 and V3 8.78; hit@5 388 against 450 and 451; P(success) 0.568 against 0.670 and 0.749 | The solver's own soft density, and even its Gaussian control, are the better distributions here. |
| Latency (sampling plus projection, 4060) | Median 18.3 ms (n 549; verifier's re-measure 11.9 ms, n 183) against 0.5 s | Fast, but speed does not help while accuracy is this far below C. |

Shared cautions:
- **Two kinds of development record.** The 441 realistic records (s19 Qwen 143, s19 GLM 141, s20 DeepSeek 157) come from 13 layouts of the realistic scene set, not from placeholder-generator seeds. The 108 watching records (36 per model) are placeholder scenes of the generator's own audience rows (`eval_14_audience_3`, `eval_15_audience_5`) at development seeds 0–2, the rows the flow trained on at seeds 110–189. Post hoc (verifier), their rasters lie nearer the training rasters (median L1 distance 23–32) than the realistic records do (45–52); validation rasters lie at 5.
- **Sampling seeds.** One set of 64 samples per record was frozen (seed = crc32 of the unit key). With the verifier's independent seeds, pooled (b) top-1 moves from 312 to 321, and GLM's gap to C narrows from −15.8 to −11.3 pp, against K2's −10. The verdict does not change.
- **Fallbacks.** When no sample survives the projection, the solver's own place is used: 136 of 549 records in (b), 102 in (a). These records carry the solver's verdict, which lifts the flow towards C.
- **Unequal fitting.** I04's V1 and V3 settings were fitted on these 549 records (leave one layout out). The flow never saw a development record.
- **Optimistic test (i).** The validation seeds also picked the checkpoint (by flow loss), so (i) is mildly optimistic.
- **Clipping.** The ±3.2 m raster holds almost none of waiting_line_6's region (0.08% of its cells in training) and 56–62% of the audience rows' and waiting_line_4's. All 162 unlabelled units (142 train, 20 validation) are waiting_line_6.

---

## 1. What was done

**Data and units** (FROZEN sections 1–3; `code/build_seeds.py`, `assemble.py`).
- Train: seeds 110–189. Validation, which is also the in-distribution test: seeds 190–199. Scenes are loaded as `phase1b/i03/stageA/fitdata.py` does, after `H.guard_scene`, with an assert per scene that it holds development situations only.
- A unit is one (scene, episode) with one target and a non-empty acceptable region. e0 and e1 have the same regions and both are kept; tables give n episodes and n scenes.
- **Raster.** 64 × 64 cells of 0.10 m around the target point (training: the true target's anchor ground point; evaluation: the recorded `result.target.world`), with x from the robot's start towards the target. Seven channels: free floor (`sgi.solver.floor`), people discs (0.25 m), each person's facing as cos and sin, the target disc, the target's seen face (`_seen_face`, a 0.5 m segment), and the robot disc. No situation, group, role, region, constraint or request text goes in. At training the people are the visible true people (`OraclePeople`), and the seen face is computed at `oracle_target`, because it is None at the anchor's ground point.
- **Labels.** 32 poses per episode, drawn uniformly from the free region cells inside the raster, jittered within ±0.05 m, with the heading the region's rule gives. An episode with no free region cell inside the raster gets no labels and stays an evaluation unit, where the flow fails it.

**Model and training** (FROZEN section 4; `code/flow.py`).
- Encoder: 4 blocks of conv 3 × 3, stride 2, GroupNorm(8) and SiLU (channels 32, 64, 128, 256), then Linear(4096 → 256). A sinusoidal time embedding (64 frequencies) through an MLP 128 → 256 → 256. Conditioning is their sum. The velocity MLP over (x, y, cos θ, sin θ) has 3 FiLM layers of width 256. 2,134,148 parameters.
- Rectified flow, AdamW (lr 1e-3, weight decay 0.01), cosine decay to 0 over 30 epochs, batches of 256 label poses, no augmentation. The checkpoint with the lowest validation loss (fixed z and t) is kept. Caps: 40 min for the main model, 15 min for each LOSO model; none was reached.
- Three LOSO models leave out waiting_line, conversation_group or audience, from both training and validation loss.

**Sampling and projection** (FROZEN section 5).
- 64 samples per unit, z from N(0, I) seeded by crc32 of the unit key, 16 Euler steps.
- **Validation seeds:** the top pose is the densest of the 64 raw samples (Gaussian kernel, σ 0.15 m).
- **Development records:** each sample snaps to the nearest cell of the solver's floor F(rec) and keeps its heading. A sample is **valid** when its cell's largest exclusion violation (`excluded_m`) is no worse than the solver place's + 1 mm. The top pose is the densest valid sample; with no valid sample, the solver's place is used (a **fallback**). hit@5 takes the next densest valid samples at least a robot's width apart.
- The flow reads no constraint. The recorded constraints reach it only through the exclusion key in the validity test; the solver's place-in-turn and preference keys are not used.

**Evaluation** (FROZEN section 6), each table computed once.
- **Success** = inside the target's acceptable region, reachable on `robot_grid(truth)`, and heading within 45 deg. On validation seeds it is scored with I14's vectorised replica, checked against `sgi.evaluate.score` on the top pose and 8 random samples per unit (7,380 checks, 0 disagreements). On development records it is scored with the package scorer `H.score_poses` and the recorded target.
- **(i) Validation seeds:** the share of the 64 samples that succeed; region coverage (share of region cells, on a 0.10 m lattice extended beyond the raster, with a raw sample within 0.2 m, so clipping counts against the flow); top-1. Baselines: `sgi.solver.solve` with the true people, the oracle target and `standard_goal`, with no constraints (B0) or with the reference constraints of `tests/test_reference_constraints` (B1, I01's P5/ALL configuration; it reads the situation and the true group, so it is circular and only a reference).
- **(ii) Setting (b):** each model's recorded people (with their yaws), the recorded target and constraints. The comparison is recorded C, the solver's place with the same inputs.
- **(iii) Setting (a):** I01 P's people and constraints (the visible true people, the recorded constraints with dropped numbers removed). The comparison is P's own solver top-1. P was recomputed for all 549 records, because I01 has no watching rows.
- **(iv) Distribution metrics against I04**, matched by record key:
  - **P(success)**: the mean success of the valid snapped samples (under the fallback, the success of the solver's place). For I04 it is the mean S.
  - **ρ_room**: the share of F(rec) cells inside the matched target's region (0 for a wrong target), asserted equal to I04's.
  - **Lift_room** = Σ P(success) / Σ ρ_room (I04's form; I14's Σ top-1 / Σ ρ_room is in `tables.txt`).
  - **hit@5**: any of the first 5 places succeeds (under the fallback, the solver's places 1–5).
- **Latency**, in the sidecar: from arrays in host memory to the top pose and the hit@5 places (z draw, encoder, 16 Euler steps over 64 samples, copy back, snap, validity, density, picks); 10 untimed warm-ups, then each record once. Painting the raster, `_over_floor` and the hand-off between the two environments are reported apart.

**Criteria** (FROZEN section 7: C5's words, with the readings fixed there). Models: Qwen 3.8, GLM-5.3-Flash and DeepSeek V4.1 Flash; "pooled per model" means over the model's realistic and watching cells (n 179, 177, 193); "pooled" means all 549.
- **Success** needs all of S1: flow(b) top-1 ≥ C − 3 pp for at least 2 models (read one-sided; two-sided beside it); S2: pooled Lift_room and hit@5 each strictly above V1's; S3: median latency below 0.5 s.
- **Kill** on any of K1: flow(a) top-1 < P − 10 pp for every model; K2: flow(b) top-1 < C − 10 pp for every model; K3: pooled Lift_room ≤ V3's and P(success) ≤ V3's mean S.
- A kill overrides success; anything else is "neither". C5's "holds on fresh seeds" is reported from (i), not judged.

**Secondary, deciding nothing:** a layout-cluster bootstrap of flow(b) − C (2,000 draws, realistic and watching strata apart), the raw unprojected top pose, each LOSO model on its left-out situation's development records, and tables by situation.

---

## 2. Main tables

### Data and training

Table 2.1. Units (seeds 110–199). Unlabelled = no free region cell inside the raster.

| Situation | Train units (scenes) | Labelled (label poses) | Validation units (scenes) | Labelled | Region share inside the raster, by row (train) |
|---|---|---|---|---|---|
| no_one | 1,760 (880) | 1,760 (56,320) | 220 (110) | 220 | 1.0 for every row |
| waiting_line | 1,120 (560) | 978 (31,296) | 140 (70) | 120 | rows 1–3: 1.0; row 4: 0.58; row 6: 0.0008 |
| conversation_group | 800 (400) | 800 (25,600) | 100 (50) | 100 | 1.0 for every row |
| audience | 480 (240) | 480 (15,360) | 60 (30) | 60 | rows 3, 4, 5: 0.62, 0.61, 0.56 |
| **All** | **4,160 (2,080)** | **4,018 (128,576)** | **520 (260)** | **500** | |

- 0 episodes left out (none had several targets or no region); 0 oracle lift failures; every episode's regions share one heading rule. All 162 unlabelled units are waiting_line_6.
- Label self-check (training labels scored by the replica, n 128,576): 96.5% in the region, 96.5% within the heading tolerance, 99.2% reachable, 95.7% all three (by situation 95.0–96.4%). The misses come from the ±0.05 m jitter crossing a boundary. Not recomputed by the verifier, but the labels are identical on every unit it rebuilt.

Table 2.2. Training runs (RTX 4060; no cap reached; each ran its 30 epochs).

| Model | Left out | Train units (poses) | Validation poses | Best epoch | Best validation loss | Wall time | Peak allocation |
|---|---|---|---|---|---|---|---|
| main | none | 4,018 (128,576) | 16,000 | 30 | 0.2457 | 237 s | 2,030 MiB |
| LOSO | waiting_line | 3,040 (97,280) | 12,160 | 30 | 0.2898 | 179 s | 1,929 MiB |
| LOSO | conversation_group | 3,218 (102,976) | 12,800 | 24 | 0.1854 | 193 s | 1,930 MiB |
| LOSO | audience | 3,538 (113,216) | 14,080 | 28 | 0.2546 | 199 s | 1,970 MiB |

The four trainings took 13.5 GPU-minutes in all (C5's entry guessed about 1–2 GPU-hours for the probe).

### (i) Validation seeds 190–199, in distribution

Table 2.3. Main model by situation (protocol seeds). The verifier's independent sampling seeds give the same top-1 in every row (pooled sample success 0.915, coverage 0.789).

| Situation | Units (scenes) | Share of samples that succeed | Region coverage | Flow top-1 | B0, solver alone | B1, reference constraints (circular) |
|---|---|---|---|---|---|---|
| no_one | 220 (110) | 0.977 | 0.937 | **220/220 (100%)** | 200/220 (90.9%) | 200/220 (90.9%) |
| waiting_line | 140 (70) | 0.799 | 0.739 | **120/140 (85.7%)** | 0/140 | 140/140 |
| conversation_group | 100 (50) | 0.944 | 0.683 | **100/100** | 100/100 | 100/100 |
| audience | 60 (30) | 0.874 | 0.535 | **60/60** | 0/60 | 60/60 |
| **Pooled** | **520 (260)** | **0.911** | **0.788** | **500/520 (96.2%)** | 300/520 (57.7%) | 500/520 (96.2%) |

- The flow's 20 misses are exactly the 20 unlabelled waiting_line_6 units, whose region lies outside the raster. B1 gets all 20.
- Coverage is lowest at audiences (0.535) and groups (0.683): at audiences much of the region lies outside the raster.

Table 2.4. Leave one situation out: each LOSO model on its left-out situation, beside the main model on the same units.

| Left out | Units (scenes) | LOSO top-1 (protocol seeds) | LOSO top-1 (verifier's seeds) | Main top-1 | LOSO / main sample success | LOSO / main coverage | B0 |
|---|---|---|---|---|---|---|---|
| waiting_line | 140 (70) | **36/140** (25.7%) | 28/140 | 120/140 | 0.193 / 0.799 | 0.403 / 0.739 | 0/140 |
| conversation_group | 100 (50) | **0/100** | 0/100 | 100/100 | 0.088 / 0.944 | 0.055 / 0.683 | 100/100 |
| audience | 60 (30) | **0/60** | 0/60 | 60/60 | 0.000 / 0.874 | 0.000 / 0.535 | 0/60 |

- 20 of the 140 line units are unlabelled waiting_line_6, which both models fail; on the other 120 the LOSO model gets 36.
- Left without groups, the flow does worse than the solver alone (0 against 100 of 100).

### (ii)–(iv) Development records (549 goal-step records)

Table 2.5. Setting (b), each model's recorded people: flow top-1 against recorded C (the solver's place with the same people and constraints).

| Cell (n) | Recorded C top-1 | Flow top-1 | Flow − C (pp) | Fallbacks | Valid share | Raw flow top-1 (no projection) | Flow top-1, verifier's seeds |
|---|---|---|---|---|---|---|---|
| old_qwen (143) | 134 (93.7%) | 82 (57.3%) | −36.4 | 25 | 0.239 | 55 | 86 |
| old_glm (141) | 96 (68.1%) | 65 (46.1%) | −22.0 | 36 | 0.284 | 60 | 73 |
| old_deepseek (157) | 109 (69.4%) | 71 (45.2%) | −24.2 | 43 | 0.256 | 66 | 74 |
| watch_qwen (36) | 32 (88.9%) | 36 (100%) | +11.1 | 8 | 0.029 | 35 | 35 |
| watch_glm (36) | 23 (63.9%) | 26 (72.2%) | +8.3 | 14 | 0.078 | 27 | 26 |
| watch_deepseek (36) | 23 (63.9%) | 32 (88.9%) | +25.0 | 10 | 0.049 | 33 | 27 |
| **Qwen (179)** | 166 (92.7%) | 118 (65.9%) | **−26.8** | 33 | 0.196 | 90 | 121 (−25.1) |
| **GLM (177)** | 119 (67.2%) | 91 (51.4%) | **−15.8** | 50 | 0.242 | 87 | 99 (−11.3) |
| **DeepSeek (193)** | 132 (68.4%) | 103 (53.4%) | **−15.0** | 53 | 0.218 | 99 | 101 (−16.1) |
| **Pooled (549)** | **417 (76.0%)** | **312 (56.8%)** | **−19.1** | 136 | 0.219 | 276 (50.3%) | 321 (58.5%) |

- Realistic cells together (writer's sum): flow 218/441 (49.4%) against C 339/441 (76.9%). Watching cells: flow 94/108 (87.0%) against C 78/108 (72.2%).
- Layout-cluster bootstrap of flow − C (secondary; 13 realistic and 6 watching layouts): Qwen −26.8 pp [−47.5, −5.6], GLM −15.8 [−29.6, −3.7], DeepSeek −15.0 [−32.9, +2.5], pooled −19.1 [−34.5, −3.7].
- Only 0.22 of the samples survive the projection on average, and at the watching cells only 0.03–0.08.
- hit@1–5, pooled: 312, 359, 375, 387, 388.

Table 2.6. Setting (a), true people (I01 P's people and constraints): flow top-1 against P's own solver top-1.

| Cell (n) | P top-1 | Flow top-1 | Flow − P (pp) | Fallbacks | Valid share | Raw flow top-1 | Flow top-1, verifier's seeds |
|---|---|---|---|---|---|---|---|
| old_qwen (143) | 136 (95.1%) | 80 (55.9%) | −39.2 | 21 | 0.224 | 60 | 81 |
| old_glm (141) | 112 (79.4%) | 70 (49.6%) | −29.8 | 19 | 0.297 | 60 | 69 |
| old_deepseek (157) | 120 (76.4%) | 69 (43.9%) | −32.5 | 36 | 0.260 | 73 | 72 |
| watch_qwen (36) | 33 (91.7%) | 36 (100%) | +8.3 | 9 | 0.023 | 35 | 35 |
| watch_glm (36) | 29 (80.6%) | 32 (88.9%) | +8.3 | 13 | 0.042 | 32 | 31 |
| watch_deepseek (36) | 31 (86.1%) | 33 (91.7%) | +5.6 | 4 | 0.038 | 34 | 35 |
| **Qwen (179)** | 169 (94.4%) | 116 (64.8%) | **−29.6** | 30 | 0.184 | 95 | 116 (−29.6) |
| **GLM (177)** | 141 (79.7%) | 102 (57.6%) | **−22.0** | 32 | 0.245 | 92 | 100 (−23.2) |
| **DeepSeek (193)** | 151 (78.2%) | 102 (52.8%) | **−25.4** | 40 | 0.218 | 107 | 107 (−22.8) |
| **Pooled (549)** | **461 (84.0%)** | **320 (58.3%)** | **−25.7** | 102 | 0.216 | 294 (53.6%) | 323 (58.8%) |

- P reproduces I01's verdicts on all 441 realistic records (0 mismatches).
- True people move the flow from 312 to 320 and the solver from 417 (C) to 461 (P).

Table 2.7. Distribution metrics, setting (b), against I04's V1 (soft density) and V3 (Gaussian control), held-out settings matched by record key.

| Cell (n) | P(success): flow / V1 / V3 | Lift_room: flow / V1 / V3 | hit@5: flow / V1 / V3 / C (lexicographic) |
|---|---|---|---|
| old_qwen (143) | 0.575 / 0.869 / 0.918 | 8.13 / 12.29 / 12.99 | 106 / 134 / 135 / 134 |
| old_glm (141) | 0.467 / 0.536 / 0.669 | 6.94 / 7.97 / 9.94 | 85 / 99 / 98 / 99 |
| old_deepseek (157) | 0.451 / 0.604 / 0.674 | 6.98 / 9.35 / 10.43 | 96 / 122 / 122 / 123 |
| watch_qwen (36) | 0.979 / 0.778 / 0.906 | 6.02 / 4.78 / 5.57 | 36 / 35 / 35 / 35 |
| watch_glm (36) | 0.723 / 0.607 / 0.640 | 4.80 / 4.03 / 4.25 | 32 / 29 / 27 / 29 |
| watch_deepseek (36) | 0.880 / 0.647 / 0.671 | 5.43 / 3.99 / 4.14 | 33 / 31 / 34 / 33 |
| Qwen (179) | 0.656 / 0.850 / 0.916 | 7.36 / 9.53 / 10.26 | 142 / 169 / 170 / 169 |
| GLM (177) | 0.519 / 0.551 / 0.663 | 6.16 / 6.54 / 7.87 | 117 / 128 / 125 / 128 |
| DeepSeek (193) | 0.531 / 0.612 / 0.674 | 6.42 / 7.39 / 8.14 | 129 / 153 / 156 / 156 |
| **Pooled (549)** | **0.568 / 0.670 / 0.749** | **6.66 / 7.85 / 8.78** | **388 / 450 / 451 / 453** |

- ρ_room equals I04's on all 549 records (largest difference 0.0), with identical F sizes.
- With the verifier's seeds, the pooled flow values are 0.573, 6.72 and 397.
- The flow beats V1 and V3 on all three measures at the watching cells, and loses on all three at the realistic cells.

Table 2.8. By situation (secondary; labels used only to tabulate). Setting (b) unless marked.

| Situation (n) | C top-1 | Flow top-1 | P(success) | Fallbacks | Raw top-1 | (a): P / flow | LOSO model on its left-out situation (fallbacks) | Flow, verifier's seeds |
|---|---|---|---|---|---|---|---|---|
| no_one (18) | 10 | 8 | 0.408 | 4 | 11 | 10 / 8 | — | 7 |
| waiting_line (261) | 196 (75.1%) | **83 (31.8%)** | 0.345 | 88 | 33 | 220 / 85 | 106 (127) | 95 |
| conversation_group (162) | 133 (82.1%) | 127 (78.4%) | 0.749 | 12 | 137 | 138 / 126 | 122 (47) | 131 |
| audience (108) | 78 (72.2%) | **94 (87.0%)** | 0.861 | 32 | 95 | 93 / 101 | 78 (103) | 88 |

- At lines the projection and the fallback carry most of the flow's 83: its raw top pose succeeds in 33 of 261.
- The LOSO models on development records lean on the fallback (127, 47 and 103 records), so that column says little about transfer. LOSO-audience, with 103 fallbacks of 108, scores exactly C's 78 (inference: it is in effect C there).

Table 2.9. Post hoc, by generator row, setting (b) (`posthoc/by_row.txt`; written after the tables; decides nothing).

| Row (n) | C top-1 | Flow top-1 | Fallbacks | Valid share |
|---|---|---|---|---|
| waiting_line_1 (51) | 35 | **5** | 4 | 0.54 |
| waiting_line_2 (52) | 43 | **6** | 3 | 0.21 |
| waiting_line_3 (53) | 35 | 21 | 20 | 0.07 |
| waiting_line_4 (52) | 44 | 12 | 10 | 0.16 |
| waiting_line_6 (53) | 39 | 39 | 51 | 0.04 |
| conversation_group_2 / 3 / 4 (54 each) | 45 / 50 / 38 | 38 / 49 / 40 | 6 / 3 / 3 | 0.39 / 0.24 / 0.27 |
| audience_3 / 5 (54 each) | 46 / 32 | 50 / 44 | 12 / 20 | 0.05 / 0.05 |
| no_one place / deep place / person (15 / 2 / 1) | 7 / 2 / 1 | 5 / 2 / 1 | 4 / 0 / 0 | 0.68 / 1.00 / 0.95 |

- The short lines fail although their regions lie wholly inside the raster in training. waiting_line_6 equals C only because 51 of its 53 records fall back to the solver.
- **Post hoc line geometry** (`posthoc/line_geometry.txt`; not recomputed by the verifier): the angle, read from the rasters, between the target's seen-face direction and the direction to the nearest person. Generator validation lines: median 6 deg (IQR 3–13), all within 30 deg (n 140). Realistic line records, setting (b): median 128 deg (IQR 106–157), 9% within 30 deg (n 146; 115 more have no seen face or no person in the raster). (inference) The generator starts each line along the target's front; the realistic layouts do not.

### Latency

Table 2.10. Latency on the RTX 4060 (sampling plus projection in the sidecar; other steps reported apart).

| What | Who | n | Median | p90 | Max | Notes |
|---|---|---|---|---|---|---|
| **Sampling plus projection, setting (b)** (S3 reads this) | compute agent (frozen measurement) | 549 | **18.3 ms** | 32.5 ms | 71.8 ms | mean 21.7 ms; torch peak 18 MiB; GPU memory in use averaged 317 MiB (max 1,386), utilisation up to 27% |
| Sampling plus projection, setting (a) | compute agent | 549 | 20.4 ms | 46.9 ms | 80.2 ms | mean 28.2 ms; GPU memory in use averaged 898 MiB (max 1,586), utilisation up to 100% (mean 50%) |
| Sampling plus projection, setting (b) | verifier, own sidecar | 183 (every 3rd record) | 11.9 ms | 16.0 ms | 20.6 ms | idle GPU (7,794 MiB free, no other compute app); peak 17.6 MiB |
| Sampling plus projection, setting (a) | verifier | 183 | 12.2 ms | | | |
| Painting the raster (project venv) | compute agent | 549 | 78.5 ms | 100.3 ms | | outside S3's span |
| `_over_floor` (solver work C already does) | compute agent | 549 | 110.8 ms | 137.1 ms | | outside S3's span |
| Hand-off between the two environments | | | not timed | | | files on disk |

The sum of the three medians is about 0.21 s; the steps were not timed as one span.

---

## 3. Figures (`figs/`)

Validation figures show one unit of seed 190 in the raster frame: floor (grey), region cells (yellow), the 64 samples with their headings, the top pose (star), and the unit's label poses (not seen by the model). Development figures show, left, the setting-(b) raster (floor grey, people red, target blue, seen face green) and, right, the record in the world frame: the acceptable region (yellow), the 64 raw samples, the valid snapped samples, the flow's top pose, the solver's place (C) and the target point. Each pair of development figures shows the same layout drawn for two models.

- [`val_no_one_eval_00_no_one_place_e0.png`](figs/val_no_one_eval_00_no_one_place_e0.png): sample success 1.00, coverage 0.97, top pose succeeds.
- [`val_waiting_line_eval_06_waiting_line_1_e0.png`](figs/val_waiting_line_eval_06_waiting_line_1_e0.png): 0.94, 0.88, succeeds. The line's person stands along the target's seen face.
- [`val_conversation_group_eval_10_conversation_group_2_e0.png`](figs/val_conversation_group_eval_10_conversation_group_2_e0.png): 0.92, 0.76, succeeds. The samples spread round the whole group ring (several modes).
- [`val_audience_eval_14_audience_3_e0.png`](figs/val_audience_eval_14_audience_3_e0.png): 0.91, but coverage 0.39, because most of the region lies outside the raster.
- [`dev_waiting_line_s19_real_qwen38_low.png`](figs/dev_waiting_line_s19_real_qwen38_low.png) and [`dev_waiting_line_s20_real_dsv41.png`](figs/dev_waiting_line_s20_real_dsv41.png): `real_eval_06_waiting_line_1`. Flow fails, C succeeds (P(success) 0.12 and 0.13; valid shares 0.50 and 0.47). In the raster the seen face points well away from the line's person. The flow's valid samples gather outside the acceptable region behind that person, where C's place lies.
- [`dev_conversation_group_s19_real_qwen38_low.png`](figs/dev_conversation_group_s19_real_qwen38_low.png) and [`dev_conversation_group_s20_real_dsv41.png`](figs/dev_conversation_group_s20_real_dsv41.png): `real_eval_10_conversation_group_2`. The flow succeeds with Qwen's people (P(success) 1.00) and fails with DeepSeek's (0.50); C succeeds in both.
- [`dev_audience_s19_dev_seed0_qwen38_low.png`](figs/dev_audience_s19_dev_seed0_qwen38_low.png) and [`dev_audience_s20_dev_seed0_dsv41.png`](figs/dev_audience_s20_dev_seed0_dsv41.png): `eval_14_audience_3` at development seed 0, a generator row the flow trained on. Both succeed, with valid shares of only 0.06 and 0.16.

The figures are illustrations chosen by `code/figs.py`, not a sample; the verifier did not recompute them.

---

## 4. Checks

- **Hashes.** `FROZEN.md` matches its sha256 (`a60f1d62…767f`); `DEVIATIONS.md` (`b94749ee…681a`) and all four checkpoints match `DEVIATIONS.sha256` and `CHECKPOINTS.sha256` (verifier).
- **Order** (file times, +08:00): format check 01:10–01:11 → `FROZEN.md` 01:18:12, hash 01:18:17 → last seed `DONE.json` 03:01:24, seed manifest 03:01:25 → code 03:13–03:24 (scoring and table code by 03:24, `flow.py` 03:46) → seed rasters 03:14:35–03:38:41 → `train.npz`, `val.npz` 03:45:31 → development arrays 04:03:13 → trainings 04:03:37–04:17:11 → `CHECKPOINTS.sha256` 04:17:45 → `DEVIATIONS.md` and hash 04:17:54–55 → validation samples 04:17:57–04:18:06, development samples 04:18:17–04:19:01 → scoring about 04:18–04:29:51 (validation scoring began once its samples were written) → tables 04:30:01 → figures 04:30:14–18 → post hoc files 04:30:52–04:32:14 → compute draft 04:32:27 → verifier 04:39–06:05. The only files older than the note are `prefreeze/`.
- **Reproduction of recorded and earlier results.** In both settings, the solver's first pick on F(rec) reproduces the recorded place (b) and P's place (a) within 1e-6 m on all 549 records. The recorded C verdict equals a package rescore on all 549. P matches I01's `rows_old_*` on 441 of 441 realistic records. ρ_room equals I04's on 549 of 549.
- **Scorer.** The replica agreed with `sgi.evaluate.score` on 7,380 of 7,380 validation checks. The verifier scored all 64 samples of every validation unit with the package scorer and agreed per unit (520 of 520).
- **Independent rebuild** (verifier's own code, `verify/v_*.py`):
  - Training: 312 units (6 seeds: 110, 126, 142, 158, 174, 189) bitwise equal in all 7 channels, labels within 1e-6, identical to `data/train.npz`.
  - Validation: 520 of 520 units bitwise equal.
  - Development: 549 of 549 rasters bitwise equal in each setting; F cells, keys and places exact.
  - Checkpoints: all four load strictly into the verifier's model class written from FROZEN section 4 (2,134,148 parameters).
  - The verifier's own training loop on `train.npz` reproduced `models/main/best.npz` bit for bit (largest weight difference 0), with the same validation curve and best epoch 30. The LOSO models were checked from their logs, `env.json` and curves (best epochs 30, 24, 28 equal each curve's minimum), not retrained.
  - With the protocol seeds, every table figure and every per-record field (top-1, fallback, hit@1–5, P(success), raw top-1) agrees on 549 of 549 records in each setting, and every per-unit validation figure on 520 of 520.
- **Data separation.** `train.npz` holds exactly seeds 110–189 and `val.npz` exactly 190–199, with no shared key. Gradient steps use only `train.npz`; `val.npz` enters only the validation loss and the checkpoint choice. No development key is among the training or validation keys, and no development raster equals any training or validation raster (smallest L1 distance 7). 320 training keys carry the names `eval_14_audience_3` or `eval_15_audience_5`; these are the generator's template ids at seeds 110–189, a shared name, not shared data.
- **No held-out data.** `phase2/seeds/placeholder` holds exactly seed110–seed199. All 2,340 scenes and 4,680 episodes were scanned after `H.guard_scene`: 0 held-out or non-development episodes, no `eval_16`–`19`, group_photo or lift_wait name. Recorded runs and development scenes were read only through the harness.
- **Sidecar untouched.** An isolated `pip freeze` lists 160 packages, each at the version in I09's freeze of 17 Oct 21:32; no file in the environment is newer than that.
- **Not independently recomputed:** the label self-check (95.7%), the post hoc line-geometry angles and the figures.

---

## 5. C5's criteria and outcomes

FROZEN section 7 readings; all exploratory. "Verifier's seeds" are the verifier's independent sampling seeds (crc32 of `'verify-alt/' + key`); the frozen measurement uses the protocol seeds.

| Criterion | Value (n) | Outcome | Verifier's seeds |
|---|---|---|---|
| Gate 2: a KDE transfers in at least 2 of 3 LOSO folds | still running | No result | — |
| Gate 3: I04 V1 and V3 on the I14 harness | rows and verification summary exist | Met | — |
| S1: flow(b) ≥ C − 3 pp for at least 2 of 3 models (one-sided) | 0 of 3: −26.8, −15.8, −15.0 (n 179, 177, 193); two-sided also 0 of 3 | Not met | 0 of 3 (−25.1, −11.3, −16.1) |
| S2: pooled Lift_room and hit@5 each strictly above V1's | 6.66 against 7.85; 388 against 450 (n 549) | Not met | 6.72; 397 |
| S3: median sampling plus projection below 0.5 s | 18.3 ms (n 549) | Met | re-measure 11.9 ms (n 183) |
| K1: flow(a) < P − 10 pp for every model | −29.6, −22.0, −25.4 | **Fires** | −29.6, −23.2, −22.8 |
| K2: flow(b) < C − 10 pp for every model | −26.8, −15.8, −15.0 | **Fires** | −25.1, −11.3, −16.1 |
| K3: pooled Lift_room ≤ V3's and P(success) ≤ V3's mean S | 6.66 ≤ 8.78 and 0.568 ≤ 0.749 | **Fires** | 6.72 and 0.573 |
| **Overall** (a kill overrides success) | | **Kill (exploratory)** | Kill |
| C5's "holds on fresh seeds of the training situations" (reported from (i), not judged) | in distribution 500/520, sample success 0.911; LOSO 36/140, 0/100, 0/60 | Not judged | 500/520; LOSO 28/140, 0/100, 0/60 |

---

## 6. Where the compute agent and the verifier differ

Settled by the verifier (its value or reading is used above):
1. **The watching records are generator rows.** The compute draft says the flow beats C on the watching records, as a plain per-situation reading. The verifier points out that these records run on `keep/dev_new` seeds 0–2 of the generator's own audience rows, the rows the flow trained on. Post hoc, their rasters lie nearer the training rasters than the realistic records' do. So the win there is probably partly an in-distribution effect (inference), and this page says so.
2. **Sampling-seed sensitivity.** The compute draft reports one sampling-seed set. The verifier's independent seeds move pooled (b) top-1 from 312 to 321, setting (a) from 320 to 323, and LOSO-lines on validation from 36 to 28 of 140. GLM's K2 gap narrows from −15.8 to −11.3 pp against the −10 line. Every verdict holds, because K1 and K3 fire by wide margins under both. The protocol values stay the frozen result; the verifier's are given beside them.
3. **What entered training.** The verifier confirmed that gradient steps used only seeds 110–189, that seeds 190–199 served only for the checkpoint choice, and that no development record entered training. The 320 training keys that carry development-like scene names are generator template ids.

Not settled (both given; neither changes a verdict):
- **Not recomputed** by the verifier: the label self-check, the post hoc line geometry and the figures. The LOSO models were checked from logs and curves, not retrained.
- **Gate 2** had no result when the compute agent wrote, when the verifier checked (`inner_ceiling.log` updated at 04:52), or when this page was written (logs updated at 06:21).

Incidental, with no effect on any result:
- The compute agent ran a 15 s smoke test of the training code on the real data before the real runs and deleted its output; `DEVIATIONS.md` discloses it, and epoch 1 of the main model reproduced its losses.
- The compute draft notes that the two development line figures show the same layout. The writer notes the same for the group pair and the audience pair.

---

## 7. What it means (inference)

**Can a learned sampler give a useful distribution over goal poses here?**
- **In distribution, yes.** On fresh seeds of the generator's own templates, 500 of 520 top poses succeed, 0.911 of all samples succeed, and the samples cover 0.788 of the region cells (in the group figure they spread round the whole ring). That matches the solver with reference constraints (B1, circular) and beats the solver alone (300), with no situation label in the input. The 20 misses come from the raster's size, not the model.
- **On the recorded development records, no.** With recorded people it is 19 points below C pooled and 15–27 below for each model, and it is below I04's V1 and V3 on every pooled measure. Only 0.22 of its samples survive the projection, 136 of 549 records keep none, and its raw top pose succeeds in 276 of 549. As a distribution over goal poses on these layouts, the solver's own soft density (I04) is the better one, and it needs no training.
- **Where it does well.** At the watching records (94 of 108 against C's 78, and above V1 and V3 there) and near C at groups (127 against 133 of 162). The watching records are the generator's own audience rows at seeds 0–2, so that win is probably partly in distribution. At groups, the realistic layouts may simply resemble the generator's rings more than its lines; that was not measured.
- **People are not the bottleneck.** True people add 8 records to the flow (312 → 320) and 44 to the solver (417 → 461). The flow's loss is in placement, not perception.

**How fast?**
- Sampling plus projection takes a median 18.3 ms (verifier 11.9 ms), about 1/27 of C5's 0.5 s bar. Painting the raster (78.5 ms, in Python) and the solver's floor work (110.8 ms, which C already does) sit outside that span; the three medians sum to about 0.21 s.
- The constraints call it would replace takes 31.7 s for Qwen; stage 23's other models take a median 1.0–6.4 s per call per stage, except GLM's constraints call at 8.6 s (REPORT C5, Why). So the speed case holds by one to two orders of magnitude, but it matters only if accuracy holds, and here it does not.

**How well does it transfer to a left-out situation?**
- **Not at all.** 0 of 100 groups, 0 of 60 audiences and 36 of 140 lines (28 with other seeds), against the main model's 100, 60 and 120 on the same units. Left without groups it does worse than the solver alone (0 against 100). As C5 warned, leaving one situation out is partly ill-posed: each situation has its own rule, and nothing in the raster says which rule applies.
- **The more telling failure is within a situation**, from generator lines to realistic lines: 83 of 261 against C's 196. The short lines are worst (waiting_line_1: 5 of 51 against 35; post hoc). Post hoc, the geometry differs: generator lines start along the target's seen face (median 6 deg), realistic lines do not (median 128 deg). So the flow appears to have learned the generator's habit, standing behind the person who stands in front of the target's face. That is C5's "memorises the generator" risk.
- C gets its lines from `come_after` (`i01/README.md`, Table 1a, new wording: the solver with no constraints scores 0 of 90 at lines for every model, and `come_after` alone matches C for 4 of 6 models). The flow receives no constraint, only the exclusions in its projection, so at lines it must infer the queue and its end from the raster alone.

1. **Gate 1 on a wider generator, read for geometry.** I06's crowd variants give 92 configurations by the letter (`phase2/i06/README.md` section 7). But they lie within the existing situations, come from development seeds 0–2, and sit on the same 10 base structures, so they add no new line geometry. This pilot suggests the variety that matters is geometric: lines that do not start on the target's face, bent lines, lines beside a counter. A count would not show that. A check that the realistic layouts fall inside the generator's range on a measure frozen beforehand (the post hoc angle above, say) would.
2. **Gate 2's result**, still running.
4. **A raster that holds long structures** (FROZEN section 10: 128 × 128 at 0.10 m), with training centred on the lifted target point.
6. **Several sampling-seed sets per record**, since one alternative set moved GLM's K2 gap by 4.5 points.
7. **The circularity remains.** The labels stay the project's reference rules, so even a pass would measure fit to those rules. Only human data (C2's CongreG8) could make a learned sampler more correct than the rules.

---

## 8. Proposed REPORT edits (not applied)

Each item gives the exact old text from `REPORT.md` and the proposed new text. Each old text occurs exactly once in the current `REPORT.md` (checked with Python's `str.count`), and applying items 1–6 in order works (checked on a copy in memory; each old text still occurs exactly once when its turn comes).

### 1. "How to read": define [phase 2]

Old:
```text
Step 3b also ran a local detector and pose model on the GPU, on whitelisted scenes and those seeds only; no provider model was called. Parts registered after earlier results on the same runs are marked data-informed.
```
New:

### 2. C5, first probe: the exploratory pilot result

Old:
```text
  - Kill: (a) more than 10 points below the solver with true people; or (b) more than 10 points below C for every model; or no better than I04 V3, the Gaussian control.
```
New:

### 3. Section 6, step-8 row

Old:
New:

### 4. Revision note

Old:
```text
and add step-3b notes to the opening lines, Programmes A and C, section 6's cut order, I46's seeds, section 7's runs and section 8's render threat.
```
New:

### 5. Opening lines: the models run (optional, recommended)

Old:
```text
the pipeline itself is unchanged, and the only models run are I09's local detectors and pose model, offline on the GPU (RT-DETRv2 and ViTPose; D-FINE on fit images only).
```
New:
```text
the pipeline itself is unchanged, and the only models run are I09's local detectors and pose model, offline on the GPU (RT-DETRv2 and ViTPose; D-FINE on fit images only), and C5's exploratory flow-matching pilot, a small model trained and sampled on the GPU on generator seeds (`phase2/flow_pilot/README.md`).
```

### 6. Section 4.6: the generative experiment (optional)

Old:
```text
It would also need its own held-out design, fixed beforehand: fresh seeds plus at least one new situation written blind (I46). I04's soft density would be the baseline it must beat (guess at the gating thresholds).
```
New:
