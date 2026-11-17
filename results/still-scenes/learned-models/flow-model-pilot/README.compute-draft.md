# C5 (I52) flow-matching pilot: results (exploratory)

**Verdict under C5's criteria: KILL (exploratory).** K1, K2 and K3 all fire. S3 (latency) is met; S1 and S2 are not. A kill overrides success (FROZEN.md section 7).

The pilot is exploratory throughout:
- Training and test layouts come from the same generator templates.
- The labels are the project's own reference rules. Agreement with them measures fit to those rules, not social correctness.

Every number below has its n. All tables are in `tables.txt` and `summary.json`.

## What was done
The protocol is `FROZEN.md` (sha256 a60f1d62…767f, checked before work began). `DEVIATIONS.md` (hashed in `DEVIATIONS.sha256`) records no deviations and lists the clarifications. It was hashed after training and before any evaluation sample was drawn. The checkpoint hashes are in `CHECKPOINTS.sha256`, written before any sampling.

1. **Data** (`code/build_seeds.py`, `code/assemble.py`, `data/manifest.json`).
   - Seeds 110–199: 2,340 scenes and 4,680 units, i.e. (scene, episode) pairs; 0 left out.
   - Train is seeds 110–189: 4,160 units in 2,080 scenes, of which 4,018 carry labels (128,576 label poses).
   - Validation is seeds 190–199: 520 units in 260 scenes, 500 labelled.
   - Every unlabelled unit (142 train, 20 validation) is waiting_line_6, whose region lies outside the ±3.2 m raster.
   - The share of the extended region inside the raster: audience 0.56–0.62, waiting_line_4 0.58, waiting_line_6 0.001, every other row 1.0.
   - Label self-check (train, n 128,576): 95.7% of label poses meet all three conditions; the misses come from the ±0.05 m jitter crossing a boundary.
   - Visual checks are in `data/png/`.
2. **Development arrays** (`code/build_dev.py`, `data/dev.json`): 549 goal-step records.
   - In both settings, `_over_floor` reproduces the recorded place (b) or P's place (a) within 1e-6 m.
   - ρ_room(b) equals I04's within 1e-9.
   - P's verdict matches I01 on all 441 realistic records.
3. **Training** (`code/flow.py`, `models/`, one GPU process at a time): every model ran its 30 epochs.
   - Main: best epoch 30, validation loss 0.2457, 237 s, 2.0 GB peak.
   - LOSO waiting_line: epoch 30, 0.2898. LOSO conversation_group: epoch 24, 0.1854. LOSO audience: epoch 28, 0.2546.
4. **Sampling and scoring** (`samples/`, `code/score_val.py`, `code/score_dev.py`).
   - The I14 replica agreed with `sgi.evaluate.score` on all 7,380 validation checks.
   - The development records were scored with `H.score_poses`.

## Results
**(i) Validation seeds, in distribution (T1).** This set also picked the checkpoint, so it is mildly optimistic.

| | n | top-1 |
|---|---|---|
| Flow | 520 units / 260 scenes | 500 (96.2%) |
| B0, solver alone | 520 | 300 (57.7%) |
| B1, reference constraints (circular) | 520 | 500 (96.2%) |

- The flow's 20 misses are exactly the 20 unlabelled waiting_line_6 units.
- Across all units, 0.911 of the samples succeed. Region coverage is 0.788; for audience it is 0.535, because clipping counts against the flow.
- The flow does not transfer across situations. Each LOSO model on its left-out situation (T2), against the main model on the same units:

| Left out | n | LOSO top-1 | Main top-1 |
|---|---|---|---|
| waiting_line | 140 | 36 | 120 |
| conversation_group | 100 | 0 | 100 |
| audience | 60 | 0 | 60 |

**(ii) Development records, setting (b), the recorded VLM people (T3).** Pooled per model:

| Model | n | Flow top-1 | C top-1 | Difference |
|---|---|---|---|---|
| Qwen | 179 | 65.9% | 92.7% | −26.8 pp |
| GLM | 177 | 51.4% | 67.2% | −15.8 pp |
| DeepSeek | 193 | 53.4% | 68.4% | −15.0 pp |
| All | 549 | 312 (56.8%) | 417 (76.0%) | −19.1 pp |

- Layout bootstrap, all 549 (secondary): −19.1 pp [−34.5, −3.7].
- 136 of the 549 records had no valid sample and fell back to the solver's place.

**(iii) Setting (a), true people (T4).** Flow 320/549 (58.3%) against P's own solver 461/549 (84.0%). The gap per model is −29.6, −22.0 and −25.4 pp.

**(iv) Against I04 (T5), pooled over 549.** I04 was fitted on these records (leave one layout out); the flow never saw them.

| | Flow (b) | V1 | V3 |
|---|---|---|---|
| P(success) | 0.568 | 0.670 | 0.749 |
| Lift_room | 6.66 | 7.85 | 8.78 |
| hit@5 | 388 | 450 | 451 |

C's lexicographic hit@5 is 453.

**Latency (T6).** Setting (b), n 549: median 18.3 ms, p90 32.5 ms on the 4060, from arrays in host memory to the top pose and the hit@5 places. Two steps are reported apart, in the project venv: painting the raster (median 78.5 ms) and `_over_floor` (median 110.8 ms).

**Criteria (T7).**

| Criterion | Reading | Result |
|---|---|---|
| S1 | 0 of 3 models within 3 pp of C | not met |
| S2 | Lift_room 6.66 against V1's 7.85; hit@5 388 against 450 | not met |
| S3 | median 18.3 ms < 500 ms | met |
| K1 | every model more than 10 pp below P | fires |
| K2 | every model more than 10 pp below C | fires |
| K3 | Lift_room 6.66 ≤ V3's 8.78, and P(success) 0.568 ≤ V3's 0.749 | fires |

## Where it fails
These breakdowns come from T9 (secondary) and `posthoc/by_row.txt` (post hoc; it decides nothing).

| Situation, setting (b) | n | Flow top-1 | C top-1 |
|---|---|---|---|
| waiting_line | 261 | 83 | 196 |
| conversation_group | 162 | 127 | 133 |
| audience | 108 | 94 | 78 |

- Waiting lines carry the loss. The short lines fail on realistic layouts even though their regions lie wholly inside the raster in training:
  - waiting_line_1: 5 of 51, against C's 35;
  - waiting_line_2: 6 of 52, against 43;
  - waiting_line_4: 12 of 52, against 44.
- waiting_line_6 equals C (39 of 53) only because 51 of its 53 records fall back to the solver.
- A post-hoc reading of the rasters (`posthoc/line_geometry.txt`; it decides nothing) points to a geometry shift. On each raster I measured the angle between the seen-face direction and the direction to the nearest person.
  - Generator validation lines (n 140): median 6 deg, all within 30 deg. The generator always starts the line along the target's front normal.
  - Realistic line records, setting (b): median 128 deg, 9% within 30 deg, n 146. The other 115 of the 261 have no seen face or no person in the raster.
  - So the flow puts its mass where the generator would have put the line, which is the "memorises the generator" risk that REPORT C5 names (inference). The two example figures `figs/dev_waiting_line_*.png` happen to show the same layout (eval_06), drawn for two models.
- The flow beats C on watching records (audience), and is near C on groups. Those are per-situation readings on 108 and 162 records, from 6 and 13 layouts.

## Files
- `tables.txt`, `summary.json`.
- Rows:
  - `rows_<setting>_<cell>.jsonl` (main model, settings a and b, six primary cells);
  - `rows_b_<cell>_loso_*.jsonl`;
  - `rows_val_<model>.jsonl` and `rows_val_baselines.jsonl`.
- `samples/` (the sidecar's outputs and `latency.json`), `models/` (checkpoints, loss curves, environment), `figs/`, `data/` (rasters, labels and the manifest; about 1 GB, gitignored), `code/`, `logs/`, `posthoc/`.
