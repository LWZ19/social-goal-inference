# Phase 1d: I04's soft solver density and C5's zero-call gates (2026-10-18)

- **I04's first probe** (REPORT I04, "First probe") turns the solver's lexicographic keys into a density over its floor cells, with a von Mises heading, and tests it against a Gaussian control on the I14 version-0 harness.

How it was made:
- **Before hashing.** The I04 lead ran one format check through the harness: record counts, fields, thresholds and one solver call per run (4.4 s). The C5 lead loaded 6 scenes of seed 100, read the fresh-seed manifest and timed the kernel on random numbers. Neither computed a region statistic, a density or a score.
- **After the freeze.** I04 has one hashed file, `i04/DEVIATIONS.md`, written at 00:30:59, before the S table and any fit. It records no deviation, eight clarifications of mechanics, and one code bug that stopped the first cache run before any result was read; the cache was rebuilt from scratch. C5 wrote no deviation note. Its departures in file layout are listed in section 4.
- **Inputs.**
  - I04: the 1,409 goal-step records of I14 version 0's 12 cells (549 primary, 860 new wording), read through the step-0 harness.
  - C5 gate 1: the generator's rows in `src/sgi/synth/generate.py`. 26 rows belong to the development situations; the 4 held-out rows are named in the code and were never instantiated. The descriptive spread uses seeds 100–109.
  - C5 gate 2: the 260 scenes of fresh seeds 100–109, generated in phase 1b and read as `phase1b/i03/stageA/fitdata.py` reads them (not regenerated). The 68 whitelisted development scenes form a secondary test set that was never fitted.
- **Compute.** CPU only and numpy only (the venv has no scipy), at most 2 processes per track, `nice -n 10`. I04 took about 10 minutes of wall time. C5's gates took about 6.5 h on 2 processes, against an estimate of 5 CPU-hours, because the machine was shared.
- **Verifiers.** Each track had an independent verifier with its own code.
  - I04's verifier rebuilt the cache, refitted all 57 fold choices and matched every per-record held-out value to 4.2e-15.
  - C5's verifier recounted gate 1 under several readings, rebuilt the cells and features (equal to 6e-15), repeated the inner cross-validation in all 14 folds, and scored the primary folds under its own subsample draw and two more.
  - Where a compute agent and its verifier differ, this page uses the verifier's final value. Where the verifier did not settle a point, both are given (section 4).
- No model-provider or network call was made, no GPU was used and torch was not imported. No held-out scene, run or situation was read. Seeds 3–99 were not generated or opened, and no seed was generated. Nothing in `src/`, `tests/`, `docs/` or `configs/` changed.
- **Phase 2 used these outputs before this page.** C5's flow-matching pilot (`phase2/flow_pilot/README.md`) took the gate-1 count and I04's V1 and V3 rows from this folder, and I06's first probe recounted gate 1 on its crowd variants (`phase2/i06/README.md` section 7). Gate 2's tables were written at 07:02, after the pilot's page, which reports gate 2 as having no result. The REPORT cites the gate-1 count and the I04 rows in its phase-2 entries but has not entered phase 1d itself; section 6 proposes the edits.

Paths below are relative to this folder (`phase1d/` in the research folder).
- `i04/`: `FROZEN.md` and its hash, `DEVIATIONS.md` and its hash, `prefreeze/` (the format check), the scripts (`i04_cache.py`, `i04_check.py`, `i04_fit.py`, `i04_heldout.py`), `cache/` (per-record arrays and `check.txt`), `fits.json`, `folds.json` (a byte-identical copy) and `FITS.sha256`, `s_table.npz`, `tables.txt`, `summary.json`, per-record rows (`rows_<cell>.jsonl`, 12 files), the logs, `timings.log`, `posthoc/insample_grid.*`, and the verifier's `verify/`.
- `c5_gates/`: `FROZEN.md` and its hash, `precheck/`, the scripts (`gate1.py`, `common.py`, `cells.py`, `gate2.py`, `tables.py`, `run_chain_a.sh`, `run_chain_b.sh`), `gate1.txt` and `gate1.json`, `cells_check.txt`, `inner_cv.txt`, `folds.json`, `rows_gate2.jsonl` (956 per-episode rows), `tables.txt`, `work/`, `logs/`, `cache/` (245 MB), and the verifier's `verify/`.

## 0. Verdicts at a glance

| Analysis | Pre-registered outcome | One-line reading (inference) |
|---|---|---|
| **I04 V1** (full soft density; primary) | **Kill.** Kill 1 fires: held-out mean log score −1.563 against the Gaussian control V3's −1.452 (n 549). Of the success criteria only (c) holds | Spreading mass along the solver's keys does worse than almost no spread. The best density on these records is a lightly smoothed top place. |
| I04 V2 (exclusions kept hard; secondary) | Kill (−1.576) | The same picture. |
| I04 confidence (criterion d) | Not met: AUROC 0.559, 0.303, 0.540 on the old-wording realistic cells | The density's concentration does not tell when the top place fails. |
| **C5 gate 1** (configuration count) | **Not met.** 22 distinct configurations (15 under the coarse reading) against 30; C2's Source A kill fires (below 25) | The generator holds about 22 templates, which fresh seeds repeat with small jitter. |
| **C5 gate 2** (KDE transfer, leaving one situation out) | **Met.** 2 of 3 folds clear 0.10 nats per region cell: waiting_line +1.955, conversation_group −0.781, audience +4.113 (frozen draw); 2 of 3 in every subsample draw checked | A density learned from generator layouts covers the regions of an unseen situation better than a distance band, except at groups. The audience margin rests partly on a weak band. |
| **C5 gate 3** (I04 on the I14 harness; judged here) | **Met** | The baseline exists and is verified. V3, not V1, is the one to beat. |

Shared cautions:
- **What the scores measure.** Every score here is agreement with the project's own reference rules (regions and heading), not social correctness. For gate 2 the rules also generated the training data.
- **Not blind.** The I04 lead knew I14 version 0's results on the same records. The C5 lead read the generator's rows and the rule code before designing the features, and gate 1's counts were knowable from the code.
- **Few independent units.** I04 has 19 layout folds, only 6 of them watching. Gate 2 has 10 seeds whose layouts are near-replicas of each other (gate 1's spread), so its seed intervals are narrow and say little about new layouts.

---

## 1. I04: soft solver density (`i04/`)

### What was done

- **Records and cells** (as `phase1/i14_v0/METRIC_NOTE.md`): goal-step records of the con runs.
  - Primary (549): old-wording realistic, s19 Qwen 143, s19 GLM 141 and s20 DeepSeek 157; watching (dev seeds 0–2, audience), 36 per model.
  - Reported beside, not judged (860): the new wording, s21 Qwen and s23 GLM, DeepSeek, Haiku, Sonnet and Gemini. Their Qwen, GLM and DeepSeek people answers are cache copies of s19/s20, so a model's old and new rows are not independent.
- **Floor and keys.** Each record's solver inputs are rebuilt as `H.resolve` does. `sgi.solver._over_floor` then gives the floor F(rec), the selectable cells (I14's floor), and each cell's five keys, rounded to the millimetre as the solver compares them.
- **Per-cell success.** I14 version 0's vectorised replica of `sgi.evaluate.score` gives, for each cell, right target, reachable, inside the matched target's acceptable region, and each forbidden kind. The cell's heading follows the solver's own rule (face the faced point). P_succ(c) = matched × reachable × in-region × P(heading within 45 deg), with the heading drawn from a von Mises distribution centred on the solver's heading, κ ∈ {4, 8, 16, 32, ∞}.
- **Variants.** The reported top-1 is always the lexicographic place.
  - **V1, full soft (primary).** Energy E(c) = Σ γ^(5−k) key_k(c) over the solver's five keys in order (`excluded_m`, `excluded_total_m`, `turn_m` or `face_m`, `preference_m`, `standard_m`), and p(c) ∝ exp(−(E − min E)/T). Grid: γ ∈ {1, 3, 10, 30} × T from 0.005 to 5 m (25 steps) × κ, 500 settings.
  - **V2, exclusions hard (secondary).** V1 restricted to the cells at the least `excluded_m` and then the least `excluded_total_m`, by the solver's own tie rule.
  - **V3, Gaussian control.** p(c) ∝ exp(−|c − c_top|² / 2σ²) around the top place. Grid: σ from 0.05 to 3 m (25 steps) × κ, 125 settings.
  - **V0, reference row (not judged).** A point mass on the top place at κ ∞. Its S is official top-1.
- **Fit.** One setting per variant per fold, shared by all models and sets. It maximises the mean over the fold's training records of ln(1e-3 + S), where S = Σ_c p(c) P_succ(c) is the density's probability of success.
  - Folds leave one layout out. 19 layouts hold records: 13 realistic (an eval index pooled over arms and models) and 6 watching (dev seed × scene). `eval_04` has no goal-step record in any cell, so its fold is neither fitted nor scored.
  - S was computed once for every record and setting, and the 19 fold choices were hashed (`FITS.sha256`) before any held-out metric.
- **Metrics** (I14 version 0 conventions; every number held out):
  - mean S; Lift_room = ΣS / Σρ_room, where ρ_room is the share of F cells inside the matched target's region (0 for a wrong target); Lift_disc, with I14's 3 m disc; the mean log score ln(1e-3 + S);
  - in-region mass (heading and reachability aside) and the mass in each forbidden kind;
  - density hit@1–5: place 1 is the lexicographic place, each next place the densest cell at least 0.6 m from every earlier place, scored at the solver's heading;
  - concentration c = the mass within 0.3 m of the top place, and its AUROC for official top-1 success;
  - layout-cluster bootstrap intervals: 2,000 draws, stratified by 13 realistic and 6 watching layouts.
- **Criteria** (REPORT I04 verbatim; readings fixed in the note).
  - Success needs all of: (a) V1's pooled Lift_room ≥ 10; (b) V1's pooled mean log score strictly above V3's; (c) density hit@5 ≥ lexicographic hit@5 − 2 in every primary cell (read one-sided; the two-sided reading is reported); (d) AUROC(c_V1) ≥ 0.70 in at least 2 of the 3 old-wording realistic cells.
  - Kill on either: V1 does not beat V3 as in (b); c_V1 > 0.9 in more than 80% of the 549 records. A kill overrides success.
- **Prediction in the note**, from I14's published table T5 before any density: (a) fails pooled for every variant. The hard top place already has a pooled Lift_room of about 8.9, and 10 would need an expected 468.5 successes of 549, more than any-of-5 reaches (453).

### Checks (before any fit)

- **Reproduction.** All 1,409 records give their recorded place and all 4 runners-up at 0.0 m, and their headings at 0.0 rad. No record is left out.
- **Replica against the package.** 0 disagreements on about 402,577 sampled cells (success, in-region, heading, reachable; 394,670 forbidden checks), on all 7,045 recorded places and 24,308 density places, and on heading offsets with the tolerance set to 30 and 60 deg. The verifier's own sample of 134,706 cells also gave 0.
- **Numerics.** The von Mises table's error is 1.4e-8 against a table 4 times finer (bar 1e-6), and 1.47e-8 against an exact Bessel series (verifier).
- **I14's S1** reproduces in all 12 cells: lexicographic top-1 134, 96, 109, 32, 23, 23 and any-of-5 134, 99, 123, 35, 29, 33, with full hit@1–5 vectors matching. ρ_room and the F and disc sizes equal I14's (verifier).
- No right-target reachable cell lies in more than one acceptable region, so the note's union-of-arcs reading for heading never arose. The 13 heading-degenerate cells take the robot's yaw, as the solver does.

### Fitted settings

Table 1.1. Settings chosen on training records only (19 folds; no ties in any fold).

| Variant | Setting (folds of 19) |
|---|---|
| V1 | γ 3, T 0.281 m, κ 16 (12); γ 3, T 0.375 (3); γ 3, T 0.211 (2); γ 3, T 0.158 (1); γ 10, T 0.089, κ 32 (1: `dev_seed2/eval_14`) |
| V2 | γ 3, T 0.281, κ 16 (10); γ 3, T 0.211 (5); γ 3, T 0.375 (1); γ 3, T 0.158 (1); γ 10, T 0.211, κ 32 (1); γ 10, T 0.089, κ 32 (1) |
| V3 | σ 0.059 m, κ 32 (16); σ 0.050, κ 32 (2); σ 0.059, κ 16 (1) |

- V3's σ sits at or next to the bottom of its grid (0.05 m).
- The best and second settings differ by at least 1.3e-4 (V1) and 2.0e-4 (V3) in the training objective, above the worst effect of the von Mises table (3e-5), so no choice is a numerical accident (verifier).

### Main tables

Table 1.2. Success measures per primary cell (held out). S is the density's probability of success; V0 is the hard top place.

| Cell | n | Top-1 | Mean S: V1 / V2 / V3 | Lift_room: V0 / V1 / V2 / V3 | Mean log score: V0 / V1 / V2 / V3 |
|---|---|---|---|---|---|
| Old Qwen | 143 | 134 | 0.869 / 0.884 / 0.918 | 13.26 / 12.29 / 12.50 / 12.99 | −0.434 / −0.490 / −0.475 / −0.454 |
| Old GLM | 141 | 96 | 0.536 / 0.562 / 0.669 | 10.12 / 7.97 / 8.35 / 9.94 | −2.204 / −2.366 / −2.333 / −2.179 |
| Old DeepSeek | 157 | 109 | 0.604 / 0.603 / 0.674 | 10.74 / 9.35 / 9.34 / 10.43 | −2.111 / −2.005 / −2.085 / −1.867 |
| Watch Qwen | 36 | 32 | 0.778 / 0.784 / 0.906 | 5.46 / 4.78 / 4.82 / 5.57 | −0.767 / −0.446 / −0.438 / −0.317 |
| Watch GLM | 36 | 23 | 0.607 / 0.606 / 0.640 | 4.25 / 4.03 / 4.03 / 4.25 | −2.494 / −2.045 / −2.068 / −2.064 |
| Watch DeepSeek | 36 | 23 | 0.647 / 0.651 / 0.671 | 3.94 / 3.99 / 4.02 / 4.14 | −2.494 / −1.398 / −1.409 / −1.284 |
| **Pooled** | **549** | **417** (0.760) | **0.670 / 0.681 / 0.749** | **8.90 / 7.85 / 7.98 / 8.78** | **−1.660 / −1.563 / −1.576 / −1.452** |

- Pooled Lift_disc: V0 6.58, V1 5.80, V2 5.90, V3 6.49.
- Pooled, every density beats the hard top place on the log score (a point mass scores ln 1e-3 wherever it fails), and V3 beats it most. Per cell, the top place beats all three at old Qwen and beats V1 and V2 at old GLM. V1 is below V3 in 5 of 6 cells; the exception is watching GLM (+0.019).
- On mean S and Lift_room, V1 and V2 are below the hard top place in every realistic cell, and V3 is just below it.

Table 1.3. Layout-cluster bootstrap, 95% percentile intervals (2,000 draws; realistic cells resample 13 layouts, watching cells 6, the pooled row both).

| Cell | V1 − V3, mean log | V1 − V3, Lift_room | V2 − V3, mean log | V2 − V3, Lift_room |
|---|---|---|---|---|
| Old Qwen | −0.035 [−0.090, +0.007] | −0.70 [−1.20, −0.03] | −0.021 [−0.072, +0.011] | −0.48 [−0.92, −0.00] |
| Old GLM | −0.187 [−0.303, −0.062] | −1.97 [−2.92, −0.62] | −0.154 [−0.263, −0.052] | −1.59 [−2.34, −0.53] |
| Old DeepSeek | −0.137 [−0.376, +0.158] | −1.08 [−1.75, −0.44] | −0.218 [−0.389, −0.065] | −1.10 [−1.72, −0.54] |
| Watch Qwen | −0.129 [−0.174, −0.080] | −0.79 [−1.11, −0.53] | −0.121 [−0.171, −0.070] | −0.75 [−1.10, −0.45] |
| Watch GLM | +0.019 [−0.382, +0.465] | −0.22 [−0.35, −0.10] | −0.004 [−0.385, +0.402] | −0.23 [−0.37, −0.09] |
| Watch DeepSeek | −0.114 [−0.496, +0.225] | −0.15 [−0.46, +0.15] | −0.125 [−0.515, +0.227] | −0.12 [−0.43, +0.15] |
| **Pooled** | **−0.111 [−0.188, −0.035]** | **−0.93 [−1.31, −0.49]** | **−0.124 [−0.190, −0.060]** | **−0.80 [−1.17, −0.43]** |

- The watching cells have 6 clusters each, so their intervals are crude. The realistic layouts are very unequal in size (`eval_00`–`05` hold 1–6 records each).

Table 1.4. Density hit@k against the solver's recorded places (lexicographic).

| Cell | Lexicographic hit@5 | V1 | V2 | V3 |
|---|---|---|---|---|
| Old Qwen | 134 | 134 | 134 | 135 |
| Old GLM | 99 | 99 | 99 | 98 |
| Old DeepSeek | 123 | 122 | 121 | 122 |
| Watch Qwen | 35 | 35 | 35 | 35 |
| Watch GLM | 29 | 29 | 29 | 27 |
| Watch DeepSeek | 33 | 31 | 31 | 34 |
| **Pooled hit@5** | **453** | **450** | **449** | **451** |
| Pooled hit@1–5 | 417 / 438 / 445 / 452 / 453 | 417 / 427 / 436 / 446 / 450 | 417 / 424 / 435 / 445 / 449 | 417 / 428 / 444 / 450 / 451 |

- V2's list stops early in 10 records (no kept cell left), and hit@k carries forward.
- V3's hit@2–4 depend on how exact ties between equidistant cells are broken (section 4); its hit@5 does not.

Table 1.5. Confidence: AUROC for official top-1 success (held out; ties count half). For the flags the score is 1 − flag, because a raised flag predicts failure.

| Cell | n (successes) | c_V1 | c_V2 | 1 − exclusion_broken | 1 − far_from_target | 1 − constraint_unusable |
|---|---|---|---|---|---|---|
| Old Qwen | 143 (134) | 0.559 | 0.562 | 0.500 | 0.500 | 0.478 |
| Old GLM | 141 (96) | **0.303** | 0.334 | 0.511 | 0.511 | 0.659 |
| Old DeepSeek | 157 (109) | 0.540 | 0.480 | 0.531 | 0.500 | 0.496 |
| Watch Qwen | 36 (32) | 0.312 | 0.273 | 0.500 | 0.500 | 0.500 |
| Watch GLM | 36 (23) | 0.579 | 0.492 | 0.500 | 0.538 | 0.555 |
| Watch DeepSeek | 36 (23) | 0.627 | 0.647 | 0.500 | 0.500 | 0.435 |
| Pooled (descriptive) | 549 (417) | 0.470 | 0.457 | 0.515 | 0.508 | 0.563 |

- c_V3 lies between 0.999996 and 1 in every record, so its AUROCs (0.11–0.61) rank float noise and are uninterpretable. The compute agent and the verifier get different values for them (section 4).
- GLM's 0.303 means its V1 density is more concentrated on records where the top place fails. This is a data-informed observation, not a tested finding.

Table 1.6. Mass measures (held out).

| Cell | n | In-region mass: V0 / V1 / V2 / V3 | Mean c_V1 | c_V1 > 0.9 | c_V2 > 0.9 |
|---|---|---|---|---|---|
| Old Qwen | 143 | 0.937 / 0.881 / 0.896 / 0.925 | 0.832 | 91 (63.6%) | 95 (66.4%) |
| Old GLM | 141 | 0.681 / 0.556 / 0.581 / 0.674 | 0.706 | 65 (46.1%) | 67 (47.5%) |
| Old DeepSeek | 157 | 0.720 / 0.645 / 0.644 / 0.706 | 0.809 | 109 (69.4%) | 111 (70.7%) |
| Watch Qwen | 36 | 0.972 / 0.837 / 0.844 / 0.969 | 0.781 | 10 (27.8%) | 11 (30.6%) |
| Watch GLM | 36 | 0.694 / 0.634 / 0.632 / 0.670 | 0.724 | 11 (30.6%) | 14 (38.9%) |
| Watch DeepSeek | 36 | 0.833 / 0.784 / 0.790 / 0.816 | 0.891 | 26 (72.2%) | 27 (75.0%) |
| **Pooled** | **549** | **0.789 / 0.705 / 0.716 / 0.777** | **0.787** | **312 (56.8%)** | **325 (59.2%)** |

- V3 keeps more than 0.999 of its mass within 0.3 m of the top place in all 549 records.
- **Forbidden mass**, pooled, over the records whose episode has the kind (V1 / V3 / share of top places inside / room share of F cells):
  - `ahead_in_line` (n 261): 0.0043 / 0.0022 / 0.0038 / 0.0016;
  - `audience_view` (n 108): 0.0117 / 0.0185 / 0.0185 / 0.0916;
  - `group_space` (n 162): 0.0010 / 0.0012 / 0.0000 / 0.0030.
  - All densities stay out of the forbidden kinds about as well as the top place does. V1's ahead-in-line mass comes from old GLM (0.0136).

Table 1.7. New wording, reported beside (not judged; each record scored with the fold of its own layout; no new-wording record is in any fit pool).

| Cell | n | Top-1 | Mean log: V1 / V3 | Lift_room: V1 / V3 | Density hit@5: lexicographic / V1 | AUROC c_V1 |
|---|---|---|---|---|---|---|
| Qwen | 144 | 131 | −0.726 / −0.646 | 12.19 / 13.09 | 131 / 131 | 0.598 |
| GLM | 141 | 86 | −2.722 / −2.635 | 7.81 / 9.45 | 89 / 89 | 0.318 |
| DeepSeek | 151 | 106 | −1.905 / −1.846 | 9.44 / 10.62 | 118 / 118 | 0.607 |
| Haiku | 146 | 134 | −0.597 / −0.576 | 11.50 / 12.28 | 141 / 137 | 0.683 |
| Sonnet | 143 | 134 | −0.494 / −0.446 | 12.72 / 13.33 | 134 / 134 | 0.898 |
| Gemini | 135 | 109 | −1.377 / −1.339 | 11.94 / 12.49 | 109 / 109 | 0.532 |

- V1 is below V3 on the log score in all 6 new cells. Haiku's density hit@5 is 4 below the lexicographic one, which would miss (c)'s bar if it were judged. Sonnet's AUROC of 0.898 rests on 9 failures.

### Post hoc (not declared; changes no verdict)

- **In-sample diagnostic** (`posthoc/insample_grid.txt`; best setting over the whole grid on all 549 records, no folds). Best V1 −1.509 (γ 3, T 0.281, κ 16), best V2 −1.510, best V3 −1.446 (σ 0.059, κ 32), V0 −1.660. Even with the setting chosen on the test records, V1 trails V3, so the kill is not a fold-fit artefact. V3's score falls steadily as σ grows beyond 0.059 m: 0.139 m −1.483, 0.387 m −1.706, 1.078 m −2.077, 3.0 m −2.653.
- **By situation** (labels used only to tabulate).

| Situation | n | Top-1 | Mean S: V1 / V3 | Lift_room: V0 / V1 / V3 | Mean log: V1 / V3 | Mean c_V1 (share > 0.9) |
|---|---|---|---|---|---|---|
| waiting_line | 261 | 196 | 0.707 / 0.728 | 13.26 / 12.49 / 12.86 | −1.760 / −1.659 | 0.938 (0.893) |
| conversation_group | 162 | 133 | 0.637 / 0.815 | 9.36 / 7.26 / 9.29 | −1.254 / −1.088 | 0.549 (0.167) |
| audience | 108 | 78 | 0.677 / 0.739 | 4.56 / 4.28 / 4.67 | −1.297 / −1.222 | 0.799 (0.435) |
| no_one | 18 | 10 | 0.390 / 0.519 | 13.23 / 9.30 / 12.37 | −3.105 / −3.114 | 0.659 (0.278) |

- V1 loses most at conversation groups, where it spreads most (mean concentration 0.549). At lines its density stays close to the top place.

### Order

File times, +08:00: `FROZEN.md` 00:15:30 → `FROZEN.sha256` 00:15:53 → scripts last saved from 00:24:56 (after the fix for the aborted first cache run) → cache 00:25:54–00:30:26 → `DEVIATIONS.md` and its hash 00:30:59 → `s_table.npz`, `fits.json`, `folds.json` and `FITS.sha256` 00:31:30 → held-out rows, tables and summary 00:31:43 → post hoc grid 00:32:19. `i04_heldout.py` re-reads `FITS.sha256` before it scores anything. The verifier's refit, with each fold's training mask asserted disjoint from its test layout, gives the same 57 choices.

---

## 2. C5's zero-call gates (`c5_gates/`)

### 2.1 Gate 1: distinct generator configurations (C2 Step 0)

**What was done.**
- `gate1.py` takes the rows from the code (`_EVAL`, `_EXAMPLES`, `_EVAL_MORE` and `_EXAMPLES_MORE` in `src/sgi/synth/generate.py`) and keeps the 26 whose situation is a development situation. Rows of group_photo and lift_wait (`eval_16`–`19`) are named in the code and never instantiated.
- **PRIMARY** (decides): distinct (recipe, amount, kind, bystanders, second group) tuples. **LOWER**: distinct (situation, structure size, bystanders, second group), with structure size 0 for no_one.
- **EMPIRICAL** (descriptive): for each row, the positions of its people across seeds 100–109 (45 seed pairs), after removing the rigid move. The frame's rotation is the yaw of the bounds polygon's first edge, checked equal to every place target's front (180 of 180), with the target anchor as origin.

Table 2.1. Gate 1 counts.

| Reading | Count | Outcome |
|---|---|---|
| **PRIMARY** (decides) | **22**: no_one 9, waiting_line 5, conversation_group 5, audience 3 | **Not met** (below 25, not borderline) |
| LOWER | 15: no_one 2, waiting_line 5, conversation_group 5, audience 3 | Reported |
| The verifier's other readings | eval split only 18; examples split only 8; kind ignored 18; no_one view angle ignored 20; situation and size only 12; PRIMARY with the 4 held-out rows' tuples added (excluded by the note) 26 | Below 30 under every reading |
| **C2's Source A kill** | 22 < 25 | **Fires** under every reading the note allows |

- 9 of the 22 are nobody-near rows that differ in the target's kind (place, deep place, object, person, screen, lift), the robot's view angle (0, 30, 60 or 90 deg) or, in one row, a bystander. 13 have a people structure: lines of 1, 2, 3, 4 and 6; groups of 2–4, three of them with bystanders or a second group; audiences of 3–5.
- **Spread within a row** (EMPIRICAL). After the rigid move and the best one-to-one matching (mirror allowed), structure members move a median of 0.00–0.48 m per row and at most 0.09–1.00 m (15 rows with members, 45 seed pairs each). Matched by id, the audience-of-3 row reaches 2.08 m because its side flip mirrors the people. Bystanders and second groups move up to about 14 m, since they are placed 3–12 m from the robot at random bearings. The code lists 29 random parameters, among them the rigid yaw and shift, the side the robot views from, which of 4 places, ±5 cm position jitter, ±0.15 rad yaw jitter, bystander spots and furnishing.
- The verifier reproduced PRIMARY 22, LOWER 15 and the spread table (medians 0.003–0.48 m, maxima 0.091–1.005 m).

### 2.2 Gate 2: does a non-neural density transfer to a left-out situation?

**What was done.**
- **Units.** One (scene, episode). Fresh seeds: 260 scenes, 520 episodes, all eligible (no_one 220, waiting_line 140, conversation_group 100, audience 60). Development: 68 scenes, 136 episodes (42 realistic, 26 placeholder). The two episodes of a scene share one region set.
- **Cells.** A 0.10 m grid aligned to the scorer's 0.05 m grid. A cell is floor when its scorer cell is free in `sgi.freespace.robot_grid` and connected to the robot's start (the scorer's own reachability path, with ground-truth people and the default thresholds). A region cell is a floor cell whose centre lies in the union of the acceptable polygons. Position only; heading is not part of gate 2.
- **Features**, computed only from what C5's raster holds (no label, group, role, scene name or region):
  - f1 distance to the target's anchor; f2–f3 bearing relative to the target's front;
  - f4 distance to the nearest person (capped at 5 m); f5 that person's facing toward the cell; f6 people within 1.5 m;
  - f7 distance to the robot's start;
  - f8 people within 0.25 m of the segment from the cell to the target; f9 the target distance minus the farthest person within 2 m of the target.
  - Feature sets: FS1 = f1–f6; FS2 = FS1 + f7; FS3 = FS1 + f8, f9.
- **Density.** A ratio KDE r = p_region / p_floor with a Gaussian product kernel on standardised features (bandwidth h = m · n_r^(−1/(d+4))), turned into q over the test scene's floor cells with 1% uniform mixed in. A plain KDE (p_region alone) and a 1-D ratio KDE on f1 are reported beside it.
- **Baseline.** Band-uniform: uniform on the floor cells whose target distance lies between the 5th and 95th percentiles of the training region cells. Uniform over floor as a reference.
- **Score.** S = the mean over an episode's region cells of log q, in nats per region cell, then the mean over episodes. Δ = S_ratio − S_band.
- **Selection,** inside each fold's training scenes only: inner leave-one-seed-out over {FS1, FS2, FS3} × m ∈ {0.5, 1, 2, 4}.
- **Folds.** Primary: leave one people situation out (train on all no_one scenes plus the other two situations; training sizes 190, 210 and 230 scenes). Secondary: leave one seed out over all four situations (the in-distribution ceiling), and training on all 260 scenes with the 68 development scenes as the test.
- **Pass rule.** Δ ≥ 0.10 nats per region cell in at least 2 of the 3 primary folds.

**Configurations chosen.** In every fold (3 primary, 10 ceiling, 1 development), inner cross-validation chose FS3 with m = 0.5 for the ratio and the plain KDE, and m = 0.5 for the 1-D KDE. Every choice sits at the grid's edge and, as frozen, the grid was not extended. The ratio KDE's h is 0.215–0.220. The verifier made the same choice in all 14 folds.

Table 2.2. Primary folds (decide gate 2), frozen subsample draw.

| Left-out situation (episodes, scenes) | S: ratio | band | uniform | plain | 1-D | **Δ** [seed bootstrap 95%] | Seeds with Δ > 0 | Ratio lift (mass) |
|---|---|---|---|---|---|---|---|---|
| waiting_line (140, 70) | −7.129 | −9.084 | −8.953 | −9.322 | −9.316 | **+1.955** [+1.927, +1.981] | 10 of 10 | 16.27 (0.393) |
| conversation_group (100, 50) | −9.938 | −9.157 | −9.262 | −10.788 | −7.763 | **−0.781** [−0.806, −0.756] | 0 of 10 | 13.59 (0.699) |
| audience (60, 30) | −7.911 | −12.025 | −9.106 | −12.733 | −10.328 | **+4.113** [+3.794, +4.454] | 10 of 10 | 16.68 (0.650) |

- **Gate 2: 2 of 3 folds clear 0.10, so it is met.**
- At conversation groups the ratio KDE loses to the band and to uniform even though its lift is 13.6: a few region cells near the 1% floor dominate the log score. The 1-D ratio KDE beats the band there (+1.395).

Table 2.3. The primary folds under four training subsamples (configuration FS3, m = 0.5 throughout; the verifier's diagnostic, post hoc).

| Draw | waiting_line | conversation_group | audience | Folds ≥ 0.10 |
|---|---|---|---|---|
| Verifier, main | +2.843 | −0.771 | +3.787 | 2 |
| Verifier, salt 1 | +2.906 | −0.770 | +4.138 | 2 |
| Verifier, salt 2 | +2.874 | −0.773 | +4.263 | 2 |

- The seed bootstrap resamples test seeds only, so its intervals (0.054 wide at waiting_line) leave out this training-subsample variation of about 0.95 nats.

Table 2.4. In-distribution ceiling (leave one seed out, all four situations; secondary) and the transfer gap.

| Situation (episodes) | Δ, ceiling | S_ratio: ceiling / primary fold | Gap in S_ratio (frozen draw; verifier's draws) |
|---|---|---|---|
| no_one (220) | +3.465 | −5.270 / — | — |
| waiting_line (140) | +3.453 | −5.606 / −7.129 | +1.523 (+0.64 to +1.52) |
| conversation_group (100) | +2.168 | −6.655 / −9.938 | +3.283 (+3.27) |
| audience (60) | +2.909 | −6.271 / −7.911 | +1.640 (+1.64 to +1.98) |
| Pooled (520) | +3.148 | −5.742 / — | — |

- The verifier's ceiling values agree within 0.02.

Table 2.5. Development transfer: trained on all 260 fresh scenes, tested on the 68 development scenes (secondary; no bootstrap, since there are no seed clusters).

| Domain / situation | Episodes (scenes) | Δ | Ratio lift |
|---|---|---|---|
| **Realistic, all** | 84 (42) | **+2.109** (verifier +2.096) | 26.68 |
| realistic no_one | 36 (18) | +0.818 | 28.72 |
| realistic waiting_line | 30 (15) | +3.574 | 27.11 |
| realistic conversation_group | 18 (9) | +2.247 | 21.91 |
| **Placeholder, all** | 52 (26) | **+3.194** (verifier +3.195) | 32.89 |
| placeholder no_one | 24 (12) | +3.393 | 42.67 |
| placeholder waiting_line | 10 (5) | +3.759 | 32.34 |
| placeholder conversation_group | 6 (3) | +2.247 | 21.91 |
| placeholder audience | 12 (6) | +2.801 | 19.29 |

**Post hoc** (added after the scores were seen; descriptive, decides nothing):
- **Band coverage.** The band covers 0.859, 0.862 and 0.289 of the region cells in the waiting_line, conversation_group and audience folds. In the audience fold the band scores 2.919 nats below uniform, which inflates that fold's Δ.
- **Against uniform.** S_ratio − S_uniform is +1.823, −0.676 and +1.195 in the frozen draw, and +2.712, −0.667 and +0.868 in the verifier's draw: 2 of 3 positive either way.
- **Per episode.** Δ is positive in 124 of 140 waiting_line episodes, 0 of 100 conversation_group episodes and 60 of 60 audience episodes.
- **Development scenes by source.** The 42 realistic scenes are three arms (plain, real, staffed) of the 14 first-pass layouts, and the plain arm scores the same as the placeholder versions of those layouts, so the 68 scenes are far from 68 independent layouts. In the real and staffed nobody-near scenes the KDE loses to the band: Δ −0.433 and −0.432 (verifier −0.456 and −0.455; n 12 episodes each), against +3.318 for the plain arm. In all 12 of those scenes a person stands within 1.5 m of some floor cell (f6 > 0), against 2 of 6 plain scenes and 2 of 12 placeholder ones. The generator never puts staff near a nobody-near target place.

**Checks.**
- Every script asserts the note's hash before it runs.
- Point-in-polygon replica against the package: 287,304 fresh and 75,138 development cell tests, 0 disagreements. The verifier's own code: 292,012 and 76,230 tests, 0 disagreements.
- 0 development runs record a different grid, robot or person radius; 136 of 136 development episodes have the same regions under their recorded config and `default.yaml`; 0 fresh scenes have two region sets.
- **Order** (file times; the folder is git-ignored): precheck 00:15–00:16 → `FROZEN.md` 00:19:15 → `FROZEN.sha256` 00:19:39 → `gate1.json` 00:25:40 → cell cache 00:28–00:29 → self-test 00:30:43 → chains start 00:31:28 → inner cross-validation (primary) done 01:34:55 → outer primary 02:44 → ceiling 06:31 → development 07:00 → tables 07:02.
- **Data access.** Fresh seeds were loaded only by `H.guard_scene` then `load_scene`, and development scenes only through `H.distinct_scenes` and `H.scene`. No scene was generated. No held-out name appears in the code, cache, logs or rows; the fresh-seed manifest shows the held-out rows skipped before any random draw.

### 2.3 Gate 3 and the gate set (judged on this page)

**Gate 3: "I04's soft density (V1, with V3 as its control) exists on the I14 harness as the baseline to beat."** It passes if the I04 track built V1 and V3 on the I14 harness and its verifier reproduced them.
- **Built on the I14 harness.** I04 used I14 version 0's records, its floor F(rec), its vectorised scorer replica and its lift definitions. It reproduced I14's S1 exactly, and its F sizes, disc sizes and ρ_room equal I14's (verifier: 0 differences).
- **Reproduced.** The verifier refitted 57 of 57 fold choices and matched every per-record held-out value to 4.2e-15.
- **Gate 3: met.** The gate asks only that the baseline exist. V1 is itself a kill against V3, so in practice the baseline a C5 sampler must beat is V3, which C5's own kill clause already names ("no better than I04 V3, the Gaussian control").

**What the REPORT says must happen first.** For gate 1: "If it finds fewer, widen the generator first with I06's crowd builder" (C5, gate 1). Section 4.6 gives the reason: below about 30 configurations, a generative model could only memorise the generator.

**Where that remedy stands** (from `phase2/i06/README.md` section 7; [phase 2], not recomputed here).
- I06's first probe (plan step 4, 18 Oct) built crowd variants of the development layouts of seeds 0–2 (families A–E). Recounted on them, gate 1 gives 92 configurations (22 plus 70) on a reading that adds (family, level) to the tuple, 63 with A's six levels and E's two depths counted once per row, and 49 under the coarse reading. So the count is met by the letter on that widened set. Its verifier checked 92 and 63 by hand; 49 and the spread were not recomputed.

---

## 3. Pre-registered criteria and outcomes

### I04 (FROZEN.md section 9; judged on V1, 549 primary records)

| Criterion | Value | Outcome |
|---|---|---|
| Stop rule: picks 1–5 reproduce the recorded places | 1,409 of 1,409 at 0.0 m | Met |
| Stop rule: the replica agrees with the package | 0 disagreements | Met |
| Stop rule: the lexicographic hit@1–5 reproduces I14's S1 | 12 of 12 cells | Met |
| (a) V1's pooled Lift_room ≥ 10 | 7.85 (V2 7.98, V3 8.78, top place 8.90) | Not met (as the note predicted) |
| (b) V1's pooled mean log score strictly above V3's | −1.563 against −1.452; V1 − V3 = −0.111 [−0.188, −0.035] | Not met |
| (c) density hit@5 ≥ lexicographic hit@5 − 2 in every primary cell | worst 122 against 123 and 31 against 33 | Met (the two-sided reading is met too) |
| (d) AUROC(c_V1) ≥ 0.70 in at least 2 of 3 old-wording realistic cells | 0.559, 0.303, 0.540 | Not met (0 of 3) |
| Kill 1: V1 does not beat V3 on the pooled mean log score | as (b) | **Fires** |
| Kill 2: c_V1 > 0.9 in more than 80% of the 549 records | 312 of 549 (56.8%) | Does not fire |
| **V1 overall** (a kill overrides success) | | **Kill** |
| Better than expected: V1's pooled Lift_room above V3's | 7.85 against 8.78 | No (reported, not a verdict) |
| Better than expected: AUROC ≥ 0.75 in 2 of 3 old realistic cells | 0 of 3 | No (reported, not a verdict) |
| V2, secondary, same bars | log −1.576 (V2 − V3 −0.124 [−0.190, −0.060]); (a) 7.98; (c) met (121 against 123, 31 against 33); (d) 0.562, 0.334, 0.480; concentration 59.2% | **Kill** |
| Prediction in the note: (a) fails pooled for every variant | V1 7.85, V2 7.98, V3 8.78 | Held |

### C5's gates (C5 FROZEN.md sections 1–2; gate 3 judged on this page)

| Criterion | Value | Outcome |
|---|---|---|
| Gate 1: PRIMARY ≥ 30 distinct configurations (25–29 borderline, not met) | 22 (LOWER 15) | **Not met** |
| C2's Source A kill: PRIMARY < 25 | 22 | **Fires** |
| Gate 2: ratio KDE Δ ≥ 0.10 nats per region cell in at least 2 of 3 leave-one-situation-out folds | waiting_line +1.955 (n 140), conversation_group −0.781 (n 100), audience +4.113 (n 60); 2 of 3 in all four draws | **Met** |
| Gate 3: I04's V1 and V3 on the I14 harness, reproduced by a verifier | built on I14 version 0's records, floors and replica; 57 of 57 fold choices and every per-record value reproduced | **Met** |
| **The gate set** (REPORT: the probe starts only if all three pass) | gate 1 fails | **Not passed: the probe may not start** |

---

Every criterion verdict agrees. The I04 verifier reproduced every reported number and every per-record held-out value. The C5 verifier reproduced gate 1, the cells, the features and every inner choice; it differs only where a random training subsample enters.

Settled by the verifier (its value is used above):
1. **I04, V3's density hit@2–4** (not judged). They depend on how exact ties between equidistant grid cells are broken. With float distance in place of integer grid distance, pooled V3 hit@2–4 would be 434, 440 and 444 instead of 428, 444 and 450 (26 records differ); hit@5 is unchanged in every cell. The integer reading, declared in DEVIATIONS clarification 3, is the faithful one, and the tables use it.
2. **I04, the in-sample wording.** The compute text says V3's in-sample score "falls steadily as σ grows". V3 peaks at σ 0.0593 (−1.4464), slightly above σ 0.05 (−1.447), so it falls steadily from 0.059 m on. Section 1 says so.
3. **I04, `cache/check.txt`.** Its file time (00:32:35) is after the held-out scoring and after `DEVIATIONS.md`, which cites it. The unchanged `i04_check.py` was evidently re-run later. Its content depends only on the cache and the I14 rows, and the verifier reproduces it exactly, so there is no effect.
4. **C5, the overall verdict.** The compute agent calls the outcome "pre-registered mixed". The note registers per-gate rules only. Under the REPORT's rule (all three gates), gate 1's failure means the gate set does not pass. "Mixed" is right as a description.
5. **C5, "a person near the target".** The development post hoc says the real and staffed nobody-near scenes have a person near the target. What was measured is a person within 1.5 m of some floor cell (f6 > 0); the counts (6 of 6 in each arm) are confirmed.
6. **C5, the bootstrap intervals.** They resample test seeds only and leave out the training-subsample variation (section 2.2), so they understate the uncertainty. The compute caveats do not say this.

Not settled (both readings given; neither changes a verdict):
- **C5, the numbers derived from the primary folds.** The transfer gap (waiting_line +1.523 against the verifier's +0.640; audience +1.640 against +1.982) and S_ratio − S_uniform (waiting_line +1.823 against +2.712; audience +1.195 against +0.868) move with the draw.
- **C5, development transfer by source** (post hoc): real and staffed nobody-near Δ −0.433 and −0.432 (compute) against −0.456 and −0.455 (verifier), and realistic overall +2.109 against +2.096; the by-situation values agree within 0.02.
- **I04, c_V3's AUROCs**, and c_V2's at the third decimal in 3 cells. They depend on float summation noise in concentrations within 1e-12 of 1. Old Qwen c_V3 is 0.353 (compute), 0.358 (verifier) and 0.387 with ties counted within 1e-12; new Qwen 0.495, 0.505 and 0.458. c_V1 is identical except watching DeepSeek, which moves from 0.627 to 0.629 under the 1e-12 tolerance. No criterion uses these values, and c_V3 is uninterpretable anyway.

Not independently re-checked:
- I04's post hoc pooled Lift_room intervals for V1 alone [6.39, 9.45] and V3 alone [7.44, 10.15], which are descriptive.
- I04's specific 402,577-cell validation sample. The verifier used its own 134,706-cell sample and checked the heading offsets directly against the package's formula, with 0 disagreements in both.
- The post hoc lines on V1's low-temperature mode (section 1).

---

## 5. What it means for the plan (inference)

**I04, the soft density.**
- The kill is clear and not a fitting accident: V1 loses to V3 out of sample (−0.111, interval below 0) and in sample (−1.509 against −1.446). Spreading mass along the solver's keys does worse than almost no spread.
- Why, as a reading: the top place succeeds in 417 of 549 records, so mass moved off it costs on most records. Spreading can only help on the 96 right-target records where the top place fails and a nearby cell succeeds. V3's fitted σ of about 6 cm buys a small hedge (−1.452 against the point mass's −1.660). V1's spread follows the keys, so it moves mass onto cells the solver itself ranks worse, and (post hoc) at low temperature its mode is not always the top place.
- **The confidence result matters more.** V1's concentration does not separate failures from successes (0.303–0.559 on the old realistic cells), and the solver's flags do no better (0.435–0.659). One likely reason, untested: many failures are not about where on the floor to stand. 36 of the 549 records have the wrong target, where every density over that floor scores 0, and others carry wrong people. A density over one record's floor cannot see either. A useful confidence signal would need spread from upstream uncertainty, for example I12's Monte Carlo over perception hypotheses, I02's fusing of trials, or I09's detector people, not from the solver's keys.
- **Knock-on.** The parked I15 (conformal goal sets) was to be revived only if "I14/I04 give a confidence AUROC of at least 0.70"; neither does. I14 version 0 already declined to license the set story on runner-up knowledge, and I04 now adds no density case on these records.
- **What it does not show.** That no density could help. It covers V1, V2 and V3 on 549 development records with the v0 metrics, scored against the project's rules.

**C5, the flow-matching bet.**
- The gates do not pass, so a non-exploratory probe may not start. Gate 1's count (22 templates, near-replica seeds) and the pilot's failure (500 of 520 in distribution, 15–27 points below C on the development records, no transfer to a left-out situation) point the same way: a model trained on this generator learns its templates.
- **Gate 2's pass is modest evidence.** A non-neural density on rule-shaped features beats a distance band in 2 of 3 unseen-situation folds and transfers to the development scenes (realistic Δ +2.1). Against that:
  - it fails at conversation groups, probably because no training situation has their ring geometry (untested);
  - its waiting_line margin moves by about 0.9 nats with the training subsample, because test lines put up to 6 people in the corridor to the target, while no training region cell has more than 2;
  - its audience margin rests partly on a band that covers 29% of the region;
  - FS3's f8 and f9 mirror the rules' geometry, as the note disclosed;
  - leaving a situation out costs 1.5–3.3 nats in S_ratio against the in-distribution ceiling (frozen draw).
- So gate 2 shows that a density learned from generator layouts can reach some unseen situations when its inputs are rule-shaped features. The pilot's raster input did not (36 of 140 lines, 0 of 100 groups, 0 of 60 audiences when left out). Inputs, targets and scores all differ between the two, so the contrast is untested as a cause.

**What gate 1's failure implies for plan step 4 (I06).**
- C5's remedy routes through I06: "widen the generator first with I06's crowd builder". The builder exists (I06's first probe, 18 Oct), and its recount meets the count by the letter (92).
- Even then the widening adds no situation, so it would not help C5's leave-one-situation-out weakness. The pilot's page suggests that the variety that matters is geometric, for example lines that do not start at the target's face, bent lines and lines beside a counter, which the crowd builder does not vary.

- **Yes, mechanically, and with zero calls.** Three now exist. I04's soft density gives a density over the solver's floor cells with a von Mises heading for every recorded goal step, in about 0.4 s per record on the CPU (cache build). Gate 2's KDE gives a density over floor cells (position only) for a new scene, learned from generator layouts. C5's pilot gives sampled poses from a flow model (phase 2).
- **Not yet a useful one, by the pre-registered standards.**
  - On the development records the best-scoring density is a near-point mass on the solver's top place (σ about 6 cm). Spreading mass along the solver's keys makes it worse, and the flow sampler is worse still (pilot: Lift_room 6.66 against V1's 7.85 and V3's 8.78).
  - None of these densities says when the top place is wrong.
  - As a robot output, a distribution would today add a small jitter around the solver's pick and no confidence signal.
- **Two different questions hide in "a distribution over goal poses"**, and the two tracks measured one each:
  1. How likely is the chosen pose to be acceptable? (I04's score: P(success) under the density.) This rewards concentration whenever the solver is usually right, which is why V3 collapsed.
  2. Where are all the acceptable poses? (Gate 2's score: the mean log density over the region's cells.) This rewards covering the whole region. Here a learned density clearly beats a distance band and uniform within the training situations (ceiling Δ +2.2 to +3.5), and partly transfers to unseen ones.
  - Question 2 is what a planner would use to pick a fallback stop, avoid a crowd or trade off path length. Question 1 is what a confidence or abstain signal needs.
- What would make question 1 answerable (untested): uncertainty that comes from perception, not from the solver's keys (I12, I02, I09's detector people).
- Every number here is agreement with the project's own rules, so a "good" distribution means one that matches the rules' regions, not what people find acceptable. Only human data (C2's CongreG8 check, I35's click pilot) could test that.

**The metric list (I14 v1).** Before any further density work, a v1 list should say which of the two questions a density answers and choose its score to match. I04's kill line ("more than 90% of the mass within 0.3 m of the top place in over 80% of records") would fire on its own best control (V3: 549 of 549), so a v1 should also say whether collapse is a failure or the right answer when the solver is usually right.

**Rulings this raises.**
- Nothing here changes the pipeline: no solver, prompt or scorer code was touched, and none is proposed.

---

## 6. Proposed REPORT edits (not applied)

Each item gives the exact old text from `REPORT.md` and the proposed new text. Each old text occurs exactly once in the report as it stands (checked with Python's `str.count`), and applying items 1–21 in order works: on a copy in memory, each old text still occurs exactly once when its turn comes, and each new text occurs once afterwards. Items 1–17 alone, in order, also apply cleanly. Items 18–21 are optional. The tag [phase 1] keeps its meaning; item 1 widens its definition to this page. Item 8 brings in I06's gate-1 recount, which the REPORT's phase-2 checking pass held until phase 1d was applied; it keeps that page's [phase 2] tag. Before applying, keep a copy of the report as `REPORT.before-phase1d.md` (item 17 names it).

### 1. "How to read": the [phase 1] tag covers phase 1d

Old:
```text
- **[phase 1]** marks a phase-1 result (17 Oct): computed through the step-0 harness from whitelisted development runs and recomputed by an independent verifier with its own code (`phase1/README.md`, `phase1b/README.md` for the second batch, or `phase1c/README.md` for step 3b, I09 A1–A2, with its section). The second batch (I03 Stages A–B, I11 Stage 0, I33's Stage B trigger) and step 3b also froze and hashed a protocol note before any evaluation scoring. Their only input outside the whitelist is fresh seeds 100–109, generated in the second batch (fit data for I03 and I09 A1, and an extra domain for I11's component check).
```
New:

### 2. Opening status line: the list of phase-1 pages

Old:
```text
`phase1c/README.md`: I09 A1–A2) and the phase-2 steps of 18 Oct
```
New:
```text
`phase1c/README.md`: I09 A1–A2; `phase1d/README.md`: I04's first probe and C5's gates 1–3) and the phase-2 steps of 18 Oct
```

### 3. Opening status line: what phase 1 answers

Old:
```text
I03 Stages A–B, I11 Stage 0 and I33's Stage B trigger. [phase 2] Phase 2 (18 Oct) answers
```
New:
```text
I03 Stages A–B, I11 Stage 0, I33's Stage B trigger, I04's first probe (kill) and C5's three zero-call gates (gate 1 not met, gates 2 and 3 met, so the gates do not pass). [phase 2] Phase 2 (18 Oct) answers
```

### 4. Section 1.2, item 9 (done)

Old:
```text
9. Optional: I04's V1 and V3 variants, once I14's metric note is frozen.
```
New:

### 5. Section 1.2, the distribution paragraph

Old:
```text
The first produced distribution would be I04's soft density (V1, with V3 as its control).
```
New:

### 6. I04, first probe: the phase-1d result

Old:
```text
  - Kill: does not beat V3; more than 90% of the mass within 0.3 m of the top place in over 80% of records.
```
New:

### 7. C2, first probe: the Step 0 result

Old:
```text
  - Kill: Source A has fewer than about 30 configurations; KDE gains under 0.1 nats; parsing takes more than 8 h. Keep Step 1 whatever happens.
```
New:

### 8. C5, gates before any training: the three results

Old:
```text
  1. C2 Step 0 finds at least about 30 distinct generator configurations. If it finds fewer, widen the generator first with I06's crowd builder.
  2. A non-neural conditional density (a KDE over reference-region cells relative to the target and the people) transfers, in at least 2 of 3 leave-one-situation-out folds, against a band-uniform baseline.
  3. I04's soft density (V1, with V3 as its control) exists on the I14 harness as the baseline to beat.
```
New:

### 9. C5, the exploratory pilot: the phase-1d outputs are now entered

Old:
```text
The gate-1 count and the I04 rows are phase-1d outputs not yet entered in this report; the kill does not rest on them, since K1 (against P) and K2 (against C) fire on their own.
```
New:
```text
The gate-1 count and the I04 rows are phase-1d outputs, now entered with the gates above (`phase1d/README.md`) [phase 1]; the kill does not rest on them, since K1 (against P) and K2 (against C) fire on their own. Gate 2's result (met) came after the pilot.
```

### 10. C5, the exploratory pilot: gate 2 now has a result

Old:
```text
A non-exploratory test would need gate 1 on a wider generator that includes realistic line geometry, gate 2's result, a raster that holds long lines
```
New:
```text
A non-exploratory test would need gate 1 on a wider generator that includes realistic line geometry, gate 2's result (since met [phase 1]), a raster that holds long lines
```

### 11. Section 4.6, where it names the gates

Old:
```text
- C2 Step 0 finds at least about 30 distinct configurations;
- the simple density above transfers;
- I04's soft density exists as the baseline (plan step 7).
```
New:
```text
- C2 Step 0 finds at least about 30 distinct configurations; [phase 1] not met: 22 on the generator's development rows;
- the simple density above transfers; [phase 1] met: a ratio KDE trained on fresh generator seeds beats the band-uniform baseline in 2 of 3 leave-one-situation-out folds, and loses at conversation groups;
- I04's soft density exists as the baseline (plan step 7). [phase 1] Met: V1 and V3 exist on the I14 harness and were verified; V1 is a kill against V3, so V3, a near-point mass on the solver's top place, is the baseline to beat.

[phase 1] So the gates do not pass (`phase1d/README.md` section 2), and by C5's rule the probe waits until the generator is widened. (inference) Gate 2 is the first evidence that a simple density learned from generator layouts reaches a situation it was not trained on, measured against the project's rules; I04's kill says the solver's own keys give no better distribution than a lightly smoothed top place.
```

### 12. Section 6, step 7 (done)

Old:
New:

### 13. Section 6, step 8: the gate results

Old:
```text
Then the probe, only if all three pass. Not counted in the do-first total. **Exploratory pilot run 18 Oct**
```
New:
```text
Then the probe, only if all three pass. Not counted in the do-first total. **Gates run 18 Oct** [phase 1] (`phase1d/README.md` section 2): gate 1 not met (22 configurations; C2's Source A kill fires), gate 2 met (2 of 3 leave-one-situation-out folds), gate 3 met (I04's V1 and V3 verified on the I14 harness). So the gates do not pass, and C5 says to widen the generator first with I06's crowd builder; I06's recount on its development-seed crowd variants gives 92 by the letter (`phase2/i06/README.md` section 7), but training scenes would need fresh seeds. **Exploratory pilot run 18 Oct**
```

### 14. Section 6, step 8: effort and what a further step needs

Old:
New:

### 15. Section 7: the metric list

Old:
```text
the 3 m disc leaves out about 90% of the watching region, and the top place lies outside it in 470 of 790 line records.
```
New:
```text
the 3 m disc leaves out about 90% of the watching region, and the top place lies outside it in 470 of 790 line records. [phase 1] I04's first probe (`phase1d/README.md` sections 1 and 5) adds two points for a v1. Its fit score, the mean of ln(1e-3 + P(success)), rewards concentration whenever the top place usually succeeds: the fitted Gaussian control keeps more than 0.999 of its mass within 0.3 m of the top place in every record, so I04's own collapse line would fire on the control. And C5's gate 2 scored a different question, how well a density covers the whole acceptable region (the mean log density over region cells). (inference) A v1 should say which question a density answers, the chance that the chosen pose is acceptable or where all acceptable poses lie, and choose its score to match.
```

Old:
New:

### 17. Revision note

Old:
```text
The repair probe's edits came after this pass (report before them: `REPORT.before-repair.md`) and are checked with phase 1d's.
```
New:

### 18. Summary item 5: C5's gates (optional)

Old:
New:

### 19. Section 4.5, the I15 parked row (optional)

Old:
```text
| I14/I04 give a confidence AUROC of at least 0.70 |
```
New:
```text
| I14/I04 give a confidence AUROC of at least 0.70. [phase 1] I04's first probe did not (0.303–0.559 on the old-wording realistic cells, `phase1d/README.md` section 1) |
```

### 20. Section 4.7, the O-1 row (optional)

Old:
```text
| O-1 Distribution over poses | I04 (with I14, I12) | b |
```
New:
```text
| O-1 Distribution over poses | I04 (with I14, I12); [phase 1] I04's first probe is a kill against its Gaussian control (`phase1d/README.md` section 1) | b |
```

### 21. Section 6, the cut order (optional)

Old:
```text
cut in this order: the optional I04 step, then I35 Part A, then I21's A1 implementation (keep its offline replay). I09's A2, once last on this list, is done (17 Oct).
```
New:
```text
cut in this order: I35 Part A, then I21's A1 implementation (keep its offline replay). I09's A2, once last on this list, is done (17 Oct), and so is the optional I04 step, once first (18 Oct, `phase1d/README.md`) [phase 1].
```
