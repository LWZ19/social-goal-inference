# Phase 3, confidence from perception disagreement: frozen protocol

The protocol author wrote this on 18 Oct 2026. It was written after the format check below and before any signal was computed against any outcome. Its sha256 is in `FROZEN.sha256`, and both files are read-only. This file is never edited. Any change goes into a dated `DEVIATIONS.md` with its own hash (`DEVIATIONS.sha256`), written before the data it affects is scored. Paths are relative to `phase3/confidence/` in the research folder (`results/2026-10-17-beyond-vlm/`) unless they start with `phase1`, `phase1c`, `phase1d`, `src/` or `docs/`.

**Pre-freeze format check** (`prefreeze/format_check.py`, output `prefreeze/format_check.txt`, about 1 s, one process). It was run twice: the second run added two counts (D place and target numbers present in every A2 row; pose and target co-occurring in every trial), and its output replaced the first. It reads the 33 runs of section 2 through the harness. It never reads a success, a score, `target_correct`, a detector distance or a re-solved place, and it computes no signal.
- Goal-step records per cell: old Qwen 143, old GLM 141, old DeepSeek 157; watching 36 each; s23 GLM 141, DeepSeek 151, Haiku 146, Sonnet 143, Gemini 135. That makes 1,265 in total. The key set (run folder/scene/episode/trial) equals I14 v0's rows (`phase1/i14_v0/rows_oldwatch.jsonl`, plus `rows_new.jsonl` without s21).
- Every goal-step record joins I04's rows (`phase1d/i04/rows_*.jsonl`) by key. Its three recorded solver flags equal I04's `flags` in 1,265 of 1,265. Every realistic goal-step record (1,157) joins an A2 row (`phase1c/i09/a2/rows_<old|new>_<model>.jsonl`) that is a solver step with a `D_d` block. All 1,157 of those rows carry a D place (`D_d.moved_from_C_m` present) and both target numbers (`tnum_C`, `D_d.tnum`).
- Every goal-step record has `target.world`, the three flags, unique person numbers and a detector-people listing in `phase1c/i09/a1/eval_people.json`. That file has 68 scenes: 42 realistic, 12 of `keep/dev_new/seed0–2` (the watching scenes) and 14 placeholder.
- 12 records have an empty people list. 1,257 name at least one person in a constraint. 10 records name 29 numbers that are absent from their own people list.
- **Trials.** 438 (run, scene, episode) groups hold a goal-step record, and each has exactly 3 trials (0, 1, 2). 434 groups have a place in all 3 trials. In 4 groups, one trial has status `no_target` and no place. 45 non-goal-step trials are standard-path records with a place. In every trial, a place and `target.world` are either both present or both absent.
- **Layouts** (I04's `layout` field, which equals the `eval_NN` of the scene id in all 1,157 realistic records). Primary: `eval_00`–`03` and `05`–`13`, 13 in all (Qwen 8, GLM 11, DeepSeek 12). s23: `eval_00`, `01`, `03`, `06`–`13`, 11 in all. Watching: `dev_seed{0,1,2}/eval_14_audience_3` and `/eval_15_audience_5`, 6 in all.
- Thresholds in all 33 runs: `person_radius_m` 0.25 (so the matching width is 0.5 m) and `match_tolerance_m` 0.3. One `H.resolve` call takes about 0.25 s.
- The venv has numpy 2.5.3, with no scipy and no sklearn.
- While choosing fields, the author printed one I04 row in full (one old-Qwen record, including its `top1` and V1 concentration) and the identity fields of one A2 solver-step row per file. Nothing else per record was displayed.

**Exposure (disclosed). The design is data-informed.**
- They also know A2's results (phase1c README, Tables 2.1–2.2 and the post hoc lines). D flips 2, 16 and 18 recorded failures (Qwen, GLM, DeepSeek) to successes and breaks 1 DeepSeek success. So at least those 37 records have a D result that differs from C's. Some of the DeepSeek flips came from positions and yaws only, at moves of 0.07–0.28 m.
- From I14's published tables: primary failures are 9 (Qwen), 45 (GLM) and 48 (DeepSeek), 102 in all, of which 34 have the wrong target (6, 14, 14).

## 1. Question and unit

- **Question.** Can a zero-call signal built from perception disagreement predict when C (people call + constraints call + solver) fails, well enough to abstain or ask for help?
- **Unit.** One goal-step record of C (`H.Record.solver_step`). The label is its official success, the recorded `score.success`. Failure is the positive class.
- **Primary cells** (judged): the three old-wording realistic cells. These are s19 `qwen38_low` (n 143), s19 `glm53` (n 141) and s20 `dsv41` (n 157), over arms real, staffed and plain: 441 records.
- **Secondary cells** (reported, not judged):
  - watching: s19 `dev_seed0`–`2` `qwen38_low` and `glm53`, and s20 `dev_seed0`–`2` `dsv41`, situation `audience`, 36 records each;
  - the s23 new-wording con runs on the realistic arms: `glm53`, `dsv41`, `haiku55`, `sonnet55_low` and `gemini38_low` (716 records).
- **Caution carried from I14.** The s23 GLM and DeepSeek people answers are cache copies of s19/s20. Their S2, S3 and P1 inputs are therefore not independent of the primary cells.

## 2. Records

- **Runs.** Old and watching are `i14.plan('oldwatch')` (`phase1/i14_v0/i14.py`, imported unchanged). s23 is `H.runs(stage='s23', arm=a, model=m, method='con')` for a in (real, staffed, plain), with exactly one run each. The folders are:
  - old Qwen: `20261011-160334_s19_real_qwen38_low_con`, `20261011-160344_s19_staffed_…`, `20261011-160527_s19_plain_…`;
  - old GLM: `20261011-160500_s19_real_glm53_con`, `20261011-160352_s19_staffed_…`, `20261011-160452_s19_plain_…`;
  - old DeepSeek: `20261013-191337_s20_real_dsv41_con`, `20261013-191345_s20_staffed_…`, `20261013-191402_s20_plain_…`;
  - watching: the nine s19/s20 `dev_seed0`–`2` con runs that `i14.plan` lists;
  - s23: the 15 folders in `prefreeze/format_check.txt`.
- **Records.** Each run's allowed records are read with `H.records`, keeping those with `rec.solver_step`. In the watching runs, only situation `audience` is kept, as I14 does. The key is `i14.key(rec)` = `<run folder>/<scene>/<episode>/<trial>`.
- **Counts required:**

| Cell | n | successes (I14 top-1) | wrong target (I14 T1) |
|---|---|---|---|
| old Qwen / GLM / DeepSeek | 143 / 141 / 157 | 134 / 96 / 109 | 6 / 14 / 14 |
| watching Qwen / GLM / DeepSeek | 36 / 36 / 36 | 32 / 23 / 23 | 0 / 2 / 0 |
| s23 GLM / DeepSeek / Haiku / Sonnet / Gemini | 141 / 151 / 146 / 143 / 135 | 86 / 106 / 134 / 134 / 109 | 22 / 15 / 1 / 7 / 26 |

- **Right target** is the recorded `score.target_correct`. It is used only to choose subsets for analysis and never inside a signal.
- **Layout of a record.** For a realistic record, the `eval_NN` in its scene id. For a watching record, `dev_seedK/<scene id>`. Both equal I04's `layout` field.

## 3. Inputs (read unchanged; hashes verified on every load)

- **Detector people.** `phase1c/i09/a1/eval_people.json`, loaded with A2's `load_detector_people()` (`phase1c/i09/a2/replay_d.py`, imported by path, unchanged). A record's detector people are those of the scene whose `listed` holds (str(run.scenes_dir), scene_id). That is all of the scene's entries in file order; no score cut is applied (choice: the file is A1's frozen output). Each person's xy is (x, y), in the world frame.
- Hashes are verified with `replay_d.verify_hashes()` (I09 FROZEN.md, `a1/eval_people.json` against `a1/EVAL.sha256`, A2 CLARIFICATIONS.md) before any signal is computed.
- **Recorded data.** People are `rec.people` as `DetectedPerson`. Constraints are `[u["constraint"] for u in rec.constraints]`, whatever their recorded status. The place is `rec.pose_world`, the target `rec.target`, the flags `rec.result["flags"]` and the label `rec.score["success"]`.
- **I04 rows.** `phase1d/i04/rows_<cell>.jsonl` (cells `old_qwen`, `old_glm`, `old_deepseek`, `watch_*` and `new_{glm,deepseek,haiku,sonnet,gemini}`), joined by `key`. Read for `V1.conc` and `flags` only.
- **A2 rows** (checks only). `phase1c/i09/a2/rows_old_{qwen38_low,glm53,dsv41}.jsonl` and `rows_new_{glm53,dsv41,haiku55,sonnet55_low,gemini38_low}.jsonl`, joined by (run folder, scene, episode, trial).
- **I14 rows** (checks only). `phase1/i14_v0/rows_oldwatch.jsonl` and `rows_new.jsonl` (s21 rows ignored): `rec_success` and `target_ok`.

## 4. Signals (higher = less confident)

Every signal comes from recorded data, the frozen detector people and the package's solver only. None uses ground truth.

**P1 (PRIMARY): re-solve sensitivity.** Realistic records only (old and s23). It is not computed at watching, as the design rules.
- **D's people and constraints.** These are exactly A2's setting D, built with A2's functions, unchanged. `obs, th` come from `repairs.scene_info(rec)` (I01, unchanged).
  - `det` = the recorded people as `DetectedPerson`; `cons` = the recorded constraints; `M = max(person numbers + constraint numbers, default=0)`.
  - `pairs = replay_d.match_xy(det, dets)`: one to one, distance ≤ 0.5 m, greedy on (distance, VLM number, detector index), as `sgi.run._matched_people`.
  - `people, out_cons, … = replay_d.build(det, cons, dets, pairs, False, M, obs, th)`. Matched detections take the detector's xy and yaw, keeping their own yaw where the detector's is None. Unmatched detections are dropped and their numbers stripped from every constraint (`repairs.strip`). Unmatched detector people are added as M+1, M+2, … in detector order. A record with an empty list gets every detector person added.
- **D's place.** `H.resolve(rec, people=[p.model_dump() for p in people], constraints=out_cons)`, with the recorded target. The solver re-derives the target person by position.
- **P1** = |xy of `rec.pose_world` − xy of D's `pose_world`| in metres, rounded to 1e-6 m before ranking (choice). Heading is not part of P1.
- **When P1 is +inf.** The design gives +inf when "D's target or verdict path differs so that no place exists". The reading fixed here is that P1 = +inf exactly when D's re-solve gives no place: `H.resolve` raises, or it returns no `pose_world`. Each such record is listed with its exception. The target point is the recorded one in both C and D, so it cannot differ. A change of target person alone does not make P1 +inf (choice; see P1b). The format check found a D place in every A2 row, so +inf is expected 0 times.

**P1b (secondary, not judged): P1 with the target-person reading.** As P1, but also +inf when the target person differs. The target person is `sgi.solver.target_person(obs, GroundedTarget(rec.target), people, th)`, taken for C's people (`det`) and for D's people and compared by number (None equals None). Each must equal A2's `tnum_C` and `D_d.tnum` (check R5).

**S2: people disagreement.** S2 = 1 − F1 = 1 − 2m / (n_v + n_d), where:
- n_v is the number of VLM people in the record, and n_d the number of detector people in its scene;
- m = |`replay_d.match_xy(det, dets)`|, the same one-to-one greedy matching within 0.5 m (inclusive), as `sgi.run._matched_people` does it.
- S2 = 0 when n_v + n_d = 0 (choice). It is computed as an exact fraction and then converted to float.

**S3: named-person support.** N is the set of distinct person numbers in the `people` of every recorded constraint (all statuses; a missing `people` counts as []).
- A number in N is **unsupported** when the record's people list has no person with that number (choice: 29 such numbers in 10 records, and every named number of an empty list), or when that person's xy has no detector person within 0.5 m (inclusive).
- Support is literal: any detector person within 0.5 m counts, not only a one-to-one match (choice).
- S3 = |unsupported| / |N| as an exact fraction, and 0 when N is empty.
- Available for realistic and watching records alike, since detector people exist for all 54 of these scenes. At watching, the detector's recall on placeholder images is 0.845 (phase1c Table 3.2).

**S4: trial disagreement.** This is a self-consistency signal. At deployment it costs two extra full pipeline runs per goal step, and every table reports it as such.
- **The trial group** G of a record is every allowed record of the same run folder with the same scene_id and episode_id: trials 0, 1 and 2, on any path. Standard-path trials with a place are included. In watching runs every member is an `audience` record anyway.
- **S4 = +inf** if any member of G has no `pose_world`.
- Otherwise **S4 = +inf** if any pair of members has recorded target ground points (`target.world` x, y) more than 1.0 m apart. This is the deployable reading of "targets differ" (choice). The scorer's `matched_target_id` would need ground truth and is not used.
- Otherwise S4 = the largest pairwise xy distance between the members' `pose_world`, rounded to 1e-6 m.
- Every goal-step record of G gets the same value. All pairs among the trials present are used (3 pairs for 3 trials). A group with a single trial would give 0 and be counted (none is expected).
- **S4-place (descriptive).** S4 without the target rule: +inf only for a missing place.

**S5: baselines.**
- **S5a, flags (judged baseline).** 1 if any of `exclusion_broken`, `far_from_target` or `constraint_unusable` is raised in the recorded flags, else 0. The recorded flags must equal I04's `flags` (check R6).
  - The three single flags are reported too (descriptive), as in phase 1d.
- **S5b, I04 V1's concentration.** 1 − `V1.conc` from I04's held-out row, raw, with no rounding, as phase 1d's compute agent used it.

**S6 (secondary): leave-one-layout-out logistic combination of P1, S2, S3 and S4.**
- **Features**, fixed before fitting (choice):
  - f1 = ln(1 + min(P1, 10)), with +inf mapped to ln 11;
  - f2 = 1{P1 = +inf};
  - f3 = S2; f4 = S3;
  - f5 = ln(1 + min(S4, 10)), with +inf mapped to ln 11;
  - f6 = 1{S4 = +inf}.
- **Standardising.** Each feature is standardised by the training fold's mean and population SD. A feature whose training SD is below 1e-12 gets weight 0. f2 is expected to be constant.
- **Model** (choice). Minimise Σ_i [ln(1 + e^{z_i}) − y_i z_i] + (λ/2) Σ_j w_j², where z_i = b + x_i·w, y = 1 for failure and λ = 1.0. The intercept b is not penalised.
  - Newton–Raphson from zero, stopping when max |gradient| < 1e-9 or after 100 iterations. Any non-convergence is reported.
- **Folds** (choice).
  - Primary: one model pooled over the 441 primary records. For each of the 13 layouts, fit on the other 12 and score the held-out layout's records with the predicted failure probability. No model indicator is used.
  - s23: the same over its 716 records and 11 layouts, fitted separately.
  - S6 is not computed at watching (no P1).
  - Out-of-fold probabilities are pooled across folds as they are. The bootstrap treats them as fixed and does not refit.
  - The full-data coefficients are reported (descriptive).
- **S6z (descriptive; choice).** The same with f1–f4 only. This is the zero-call combination: P1, S2 and S3 need no extra model call.

**Deployment cost**, reported beside every signal:
- P1 and P1b: the detector (I09 A1, about 56 ms on the 4060) plus one extra solver run (about 0.25 s CPU);
- S2 and S3: the detector only;
- S4: two extra full pipeline runs;
- S5: free;
- S6: all of the above; S6z: the detector plus one solver run.

## 5. Measures

**AUROC for failure.** For n1 failures and n0 successes, AUROC = (R1 − n1(n1 + 1)/2) / (n1·n0), where R1 is the sum of the failures' midranks in the joint ascending ranking of the signal. So ties count half. +inf values tie with each other and rank above every finite value. The AUROC is undefined, and printed as "—", when n1 = 0 or n0 = 0.
- **Per cell**, for every signal.
- **Pooled (judged):** one ranking over the 441 primary records together (choice). Beside it, descriptive:
  - the stratified pooled AUROC: concordant failure–success pairs within a cell (ties half), summed over cells, divided by Σ n1·n0 within cells;
  - the mean of the three cell AUROCs.
- **Pooled, secondary (descriptive):** over the 716 s23 records; over the 108 watching records; and over all 549 I14 primary records, for the signals defined there (S2–S5), for comparison with phase 1d.
- **Right-target records only.** The subset with `target_correct` True: 407 primary records, of which 68 are failures. AUROC per cell and pooled, with the bootstrap.
- **Wrong-target AUROC** (for the prediction; descriptive). Positives are the wrong-target records (all failures; 34 primary), negatives the successes (339). Pooled primary and per cell.
- **Comparison precision.** AUROCs are compared at full precision and printed to 3 decimals.

**Layout-cluster bootstrap.** 2,000 draws.
- `rng = np.random.default_rng(20261018)`. Three draw matrices are made in this order:
  1. primary: `rng.integers(0, 13, size=(2000, 13))` over the 13 primary layouts, sorted as strings;
  2. s23: `rng.integers(0, 11, size=(2000, 11))`;
  3. watching: `rng.integers(0, 6, size=(2000, 6))`.
- The same matrix serves every signal and every cell of its group, so the draws are paired.
- A cell's resample is all of its records in the drawn layouts, with multiplicity. The per-cell resamples use the group's whole layout list (I04's convention), so a cell's resample size varies by draw.
- A draw in which a cell's resample has no failure or no success is dropped for that cell, and the number of dropped draws is reported.
- Intervals are 95% percentile intervals (numpy `percentile`, default linear, at 2.5 and 97.5).
- Also reported: the paired difference P1 − S5a for the pooled primary AUROC, per draw, with its interval (descriptive; the criterion uses point values).

**Selective accuracy** (point values, all records of a cell, and pooled primary).
- Records are sorted by signal ascending, most confident first. At coverage q ∈ {0.8, 0.5} the kept mass is K = q·n.
- Records strictly below the tie group that straddles K are kept whole, and records above it are dropped. Each member of that group (size g, with c records below it) gets weight (K − c)/g. This is the expected value under random tie-breaking (choice).
- Selective accuracy = Σ w_i y_i / K (y = 1 for success). It is reported beside the base rate (mean y) and the difference.

**Failures caught at 20% abstention.**
- The abstained weight is 1 − w_i at q = 0.8, ties handled as above.
- Caught = Σ (1 − w_i)(1 − y_i), as an expected count to 1 decimal, and as a share of the cell's failures, beside the random expectation 0.2 × failures.
- Caught failures are split into wrong-target and right-target failures.

**Descriptive counts:**
- records with P1 = 0, > 0 and +inf;
- records with P1 = 0 whose D heading differs from the recorded heading by more than 1 deg (wrapped difference);
- records whose P1b is +inf;
- S4 +inf records by path (missing place, or targets more than 1.0 m apart);
- S3 = 0 records, and records naming no one.

## 6. Reproduction checks (all must pass before any AUROC, selective accuracy or criterion is computed)

- **R1, I14's goal-step counts.** The per-cell n of section 2 holds, and the key set equals I14's rows (s21 excluded), key by key.
- **R2, recorded success of every record.** For every one of the 1,265 records, `rec.score.success` = `H.rescore(rec).success` = I14's `rec_success` = I04's `top1`. The per-cell success counts equal section 2's table.
- **R3, right target.** `score.target_correct` equals I14's `target_ok` for every record, and the per-cell wrong-target counts equal section 2's.
- **R4, step 0.** For every realistic record (1,157), `H.resolve(rec)` gives the recorded place within 1e-6 m (`SAME_PLACE_M`).
- **R5, A2's D per record.**
  - For every realistic record, the package's success of the fresh D re-solve equals A2's `D`.
  - |P1 − A2's `D_d.moved_from_C_m`| ≤ 1e-4 m. A2 rounded to 4 decimals and measured from the step-0 place.
  - C's and D's target numbers equal A2's `tnum_C` and `D_d.tnum`.
- **R6, joins.** Every record joins its I04 row, and its recorded flags equal I04's `flags`. Every realistic record joins its A2 row as a solver step. Every record has detector people.
- **R7, hashes.** `replay_d.verify_hashes()` passes. `FROZEN.sha256` verifies at the start of every script.
- **R8, AUROC code check** (the first thing the scoring pass prints). S5b and the three single flags must reproduce phase1d Table 1.5 to ±0.0005:
  - `exclusion_broken` 0.500 / 0.511 / 0.531 / 0.500 / 0.500 / 0.500;
  - `far_from_target` 0.500 / 0.511 / 0.500 / 0.500 / 0.538 / 0.500;
  - `constraint_unusable` 0.478 / 0.659 / 0.496 / 0.500 / 0.555 / 0.435;
  - the values pooled over the 549 records: c_V1 0.470, `exclusion_broken` 0.515, `far_from_target` 0.508 and `constraint_unusable` 0.563;
  - the s23 c_V1 values of phase1d Table 1.7: GLM 0.318, DeepSeek 0.607, Haiku 0.683, Sonnet 0.898, Gemini 0.532.
  - (AUROC for failure with signal s equals phase 1d's AUROC for success with 1 − s.)

## 7. Pre-registered criteria for P1 (judged on the 441 primary records; point estimates)

- **Success**, when both hold:
  - P1's AUROC is at least 0.70 in at least 2 of the 3 primary cells;
  - P1's pooled AUROC is strictly above the pooled AUROC of the flags baseline (S5a, any flag raised).
- **Kill:** P1's pooled AUROC is below 0.60.
- **Otherwise:** neither. If the success and kill conditions both hold, the verdict is kill (choice, as in I04).
- P1b, S2, S3, S4, S5, S6 and S6z are reported, not judged. The bootstrap intervals are reported beside the verdict and do not enter it.

> P1 helps for GLM and DeepSeek, whose people answers carry phantoms that the detector does not confirm, and barely for Qwen, whose people answers are nearly perfect; S4 (trial disagreement) is the strongest single signal but costs extra calls; wrong-target failures stay invisible to P1-S3.

How it is read (choice; descriptive, not a criterion):
- **Part 1** holds when P1's AUROC is at least 0.70 in both old GLM and old DeepSeek and below 0.60 in old Qwen. Qwen's AUROC rests on 9 failures.
- **Part 2** holds when S4 has the strictly highest pooled primary AUROC among the single signals P1, S2, S3, S4, S5a and S5b. P1b, the single flags, S4-place, S6 and S6z are not in that comparison.
- **Part 3** holds when P1, S2 and S3 each have a pooled primary wrong-target AUROC below 0.60.

## 9. Compute rules, order and outputs

**Rules.**
- Runs and scenes are read only through `docs/model_notes/tools/research_harness.py`. Nothing held out is read: no s18 or s20_held run, nothing under `keep/heldout`, no `eval_16`–`19` scene, no synth seeds 3–99 and nothing of seeds 200–219 (`phase2/seeds/realistic`, the `phase2/i09/confirm` rows; and, to be safe, nothing under `phase3/seeds`).
- CPU only. `nice -n 10`, 2 BLAS threads (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` = 2), at most 2 worker processes, and `free -m` must show at least 3 GB available first. No pkill or killall.
- `.venv/bin/python -B` with `PYTHONPATH` empty. numpy only.
- Nothing in `src/`, `tests/`, `configs/`, `docs/`, REPORT.md or any earlier phase folder is edited. Earlier code is imported unchanged (`i14.py`, `repairs.py`, `replay_d.py`, and I04's rows as data). Output goes only to this folder.

**Order.**
1. `FROZEN.md` → `FROZEN.sha256`, both read-only.
2. **Pass 1** (`confidence.py`). It computes every signal and runs checks R1–R7, writing `rows.jsonl` (one row per record: key, cell, layout, label, target_correct, P1, P1b, S2, S3, S4, S4-place and their parts, flags, c_V1, D heading, check fields; +inf written as JSON `Infinity`), `checks.txt` and `pass1.log`. It computes no AUROC, rate or criterion. It may run as 2 processes split by cell group, with results independent of the split.
3. **If R1–R7 pass:** any `DEVIATIONS.md` (with its hash) is written now.
4. **Pass 2** (`tables.py`). It re-verifies `FROZEN.sha256` (and `DEVIATIONS.sha256` if one exists) and runs R8 first. It then fits S6/S6z and computes every measure of section 5, writing `tables.txt`, `summary.json` and `pass2.log`.
   - The verdict of section 7 and the prediction reading of section 8 are printed last.
   - Any analysis not in this note is labelled post hoc and changes no verdict.

**Tables in `tables.txt`:**
- T0 checks R1–R8;
- T1 records per cell: n, successes, failures, wrong target, base rate;
- T2 AUROC per cell and pooled, with intervals and dropped draws, for every signal, with its cost;
- T3 right-target AUROC;
- T4 wrong-target AUROC;
- T5 selective accuracy at 80% and 50%, and failures caught at 20% abstention (split wrong-target and right-target);
- T6 the secondary cells (watching, s23), as T2–T5;
- T7 S6/S6z coefficients (full-data fit, and the per-fold mean and range);
- T8 the descriptive counts of section 5;
- T9 the criteria and the prediction.

## 10. Cautions and limits (stated now)

- **What the label measures.** Every label is agreement with the project's own reference rules (regions, heading, reachability), not social correctness.
- **Few independent units.** There are 13 primary layouts, unequal in size, and 3 trials share each episode, so S4 is constant within a trial group. Qwen has 9 failures. Intervals are crude, especially at watching (6 layouts).
- **D's people are not what a deployed pipeline would see.** P1 keeps the VLM's constraints, written against the VLM's own people. A pipeline that showed the model the detector's numbered people would get different constraints (phase1c section 2), and P1 does not test that.
- **The detector at watching.** It is weak on placeholder images (recall 0.845, against 0.983 realistic), so S2 and S3 at watching mix the detector's misses with the VLM's errors.
- **This is a development-set probe.** No fresh scene, held-out run or new seed is read. A positive result would need confirming on data the design did not see.
