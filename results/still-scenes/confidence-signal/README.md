# Phase 3: confidence from perception disagreement (2026-10-18)

Phase 1d found that no confidence signal tried so far separates C's failures from its successes. C is the people call plus the constraints call plus the solver. I04's density concentration gave AUROC 0.559, 0.303 and 0.540 on the old-wording realistic cells, and the solver's flags gave 0.435–0.659 (`results/still-scenes/overview/phase-notes/phase1d.md` sections 1 and 5). That page suggested, untested, that a useful signal would have to come from upstream perception uncertainty, not from the solver's keys. This probe tests that suggestion with zero calls. It compares the VLM's people with I09's frozen detector people and asks whether the disagreement predicts when C fails, well enough to abstain or ask for help.

How it was made:
- **Before hashing.** One format check through the harness (`prefreeze/format_check.py`, about 1 s, run twice; the second run added two counts). It read no success, score, `target_correct`, detector distance or re-solved place, and computed no signal. While choosing fields, the author printed one I04 row in full (one old-Qwen record, with its top-1 verdict and V1 concentration).
- **After the freeze.** Two hashed deviation notes, neither changing a signal, label, record set or criterion.
  - `DEVIATIONS.md` (hashed 17:43:07), after pass 1's checks R1–R7 passed and before pass 2: file layout and reporting clarifications only (one row file per cell, exact fractions for selective accuracy, standardised S6 coefficients, point values only for the 549-record pool and the wrong-target AUROC).
  - `DEVIATIONS_2.md` (hashed 17:43:57), after the first pass-2 attempt stopped at check R8 on one value at the tolerance boundary: R8 now compares in exact arithmetic (section 2). That attempt printed only R8's baseline numbers (S5b and the single flags), no P1–S4 or S6 number (`pass2.log`).
- **Inputs.**
  - The 1,265 goal-step records of I14 version 0's cells other than s21 Qwen, read through the step-0 harness (`docs/model_notes/tools/research_harness.py`). Primary (judged): the 441 old-wording realistic records. Secondary: watching (108) and the five s23 new-wording con runs on the realistic arms (716).
  - I09 A1's frozen detector people (`results/still-scenes/people-finder/early-detector/phase1c/a1/eval_people.json`, hash-checked), loaded with A2's own loader.
  - A2's setting D, built with A2's functions (`results/still-scenes/people-finder/early-detector/phase1c/a2/replay_d.py`) and I01's `repairs.scene_info`, imported unchanged.
  - I04's rows (`phase1d/i04/rows_*.jsonl`) for the flags and V1's concentration, and I14's rows for the checks.
- **Compute.** CPU only and numpy only, at most 2 processes, `nice -n 10`, 2 BLAS threads. Pass 1 (every signal and checks R1–R7) took about 2.2 min on 2 processes; pass 2 (every table) about 1 s.
- The full structured outputs of the protocol author, the compute agent and the verifier are in `agent_outputs.json`, unedited.

Paths below are relative to this folder (`results/still-scenes/confidence-signal/` in the research folder).
- `FROZEN.md` and `FROZEN.sha256`; `DEVIATIONS.md`, `DEVIATIONS_2.md` and their hashes; `prefreeze/` (the format check).
- `code/confidence.py` (pass 1 and the `checks` mode) and `code/tables.py` (pass 2).
- `rows_<cell>.jsonl` (11 files, one row per record), `checks.txt`, `pass1.log`, `pass2.log`, `tables.txt` (tables T0–T9) and `summary.json`.
- `verify/`: the verifier's scripts (`v_pass1.py`, `v_tables.py`, `v_compare.py`, `v_break.py`, `v_exposure.py`), its rows, `vtables.out`, `vcompare.out`, `vbreak.json` and `vexposure.out`.

## 0. Verdicts at a glance

| Analysis | Pre-registered outcome | One-line reading (inference) |
|---|---|---|
| **P1, re-solve shift under the detector's people** (judged) | **Neither.** AUROC for failure 0.645 (old Qwen), 0.738 (old GLM), 0.688 (old DeepSeek): 1 of 3 cells reaches 0.70, and 2 are needed. Pooled 0.708 [0.564, 0.828], above the flags' 0.595; the kill line (below 0.60) is not crossed | A modest signal, and the verdict sits on the line between neither and success. Most of its ranking comes from the 37 records A2 had already shown would change. |
| S2, people disagreement with the detector (reported) | Not judged. Pooled 0.741; cells 0.869 / 0.801 / 0.650 | The detector-only signal ranks failures better than P1 and needs no solver run. |
| S4, trial disagreement (reported) | Not judged. Pooled 0.774 | Strongest single signal, but it costs two extra full pipeline runs per goal step. |
| S6z, zero-call combination of P1, S2 and S3 (reported) | Not judged. Pooled 0.752 | Abstaining on the least confident 20% raises C's success among the kept records from 0.769 to 0.878 and catches 59.0 of 102 failures (random 20.4). |
| S6, the combination with S4 (reported) | Not judged. Pooled 0.849 | The best ranking measured here, at the cost of two extra runs. |
| S5b, I04 V1's concentration (reported) | Not judged. Pooled 0.459 | Reproduces phase 1d's null. |
| Secondary cells (reported) | s23: P1 0.641 pooled, but Haiku 0.197, Sonnet 0.562, Gemini 0.501. Watching: S2 0.717, S4 0.695 | Perception signals carry little for the models whose failures are not perception (the API models). |

Shared cautions:
- **What the label measures.** Every label is agreement with the project's own reference rules (regions, heading, reachability), not social correctness.
- **Not blind.** The design was written knowing phase 1d's and A2's results on the same records (above, and section 5's exposure check).
- **Few independent units.** 13 primary layouts, very unequal in size. Qwen has 9 failures. The 3 trials of an episode share one S4 value. Intervals are wide: P1's pooled interval runs from 0.564 to 0.828.
- **Fragile verdict.** Dropping any one of 3 layouts, or using trial 1 or trial 2 alone, turns P1's verdict into success, and the success condition holds in half the bootstrap draws (section 5).
- **Pooling across models.** Which model answered is itself a signal: each cell's failure rate, used as the score, gives a pooled AUROC of 0.658, above the flags' 0.595. So "pooled above the flags" is a low bar. P1's stratified value (0.703) is close to its one-ranking value (0.708), so pooling barely inflates P1 itself.
- **Development records only.** No fresh scene, held-out run or new seed was read. Any positive reading needs confirming on records the design did not see.

---

## 1. What was done

- **Question.** Can a zero-call signal built from perception disagreement predict when C fails, well enough to abstain or ask for help?
- **Unit and label.** One goal-step record of C (`H.Record.solver_step`). The label is the recorded official success; failure is the positive class. Every signal is "higher = less confident".
- **Cells.**
  - Primary (judged): the old-wording realistic cells, s19 `qwen38_low` (n 143), s19 `glm53` (141) and s20 `dsv41` (157), over the real, staffed and plain arms: 441 records, 102 failures, 34 of them with the wrong target.
  - Secondary (reported): watching (s19/s20 `dev_seed0`–`2`, audience, 36 records per model) and the s23 new-wording con runs on the realistic arms for GLM, DeepSeek, Haiku, Sonnet and Gemini (716). s21 Qwen is not in the design's list and was left out. The s23 GLM and DeepSeek people answers are cache copies of s19/s20, so their perception signals repeat the primary cells' inputs.

**Signals** (each from recorded data, the frozen detector people and the package's solver only; none uses ground truth):

| Signal | Definition | Deployment cost |
|---|---|---|
| **P1** (judged) | Distance in metres from C's recorded place to the place the solver gives under A2's setting D, rounded to 1e-6 m. D moves each VLM detection onto the detector person within 0.5 m (keeping its number), drops the unmatched detections and strips their numbers from the constraints, and adds the unmatched detector people. +inf only if D gives no place (0 records). Realistic records only | Detector (about 56 ms on the 4060) + one extra solver run (about 0.25 s CPU) |
| P1b | P1, also +inf when the solver's target person changes between C's and D's people | As P1 |
| S2 | 1 − F1 between the VLM's people and the detector's, with the same one-to-one greedy matching within 0.5 m | Detector only |
| S3 | Share of the people named in the constraints that have no detector person within 0.5 m (a named number missing from the people list counts as unsupported) | Detector only |
| S4 | Largest pairwise place distance among the 3 trials of the same episode; +inf if a trial has no place or two trials' target ground points lie more than 1.0 m apart | **Two extra full pipeline runs** |
| S4-place | S4 without the target rule | As S4 |
| S5a (baseline) | Any of the three recorded solver flags raised; the three single flags are reported too | Free |
| S5b | 1 − I04 V1's held-out concentration | Free |
| S6 | Logistic combination of P1, S2, S3 and S4 (six fixed features, λ = 1), fitted leaving one layout out over the pooled records, no model indicator | All of the above |
| S6z | The same with P1, S2 and S3 only | Detector + one solver run |

**Measures.** AUROC for failure (midranks, so ties count half; +inf values tie with each other and rank highest), per cell and pooled as one ranking over the 441 records (judged), with a stratified pooled value and the mean of the cells beside it. 95% layout-cluster bootstrap intervals (2,000 paired draws, seed 20261018; draws with no failure or no success are dropped and counted). The right-target subset (407 records, 68 failures). A wrong-target AUROC (34 wrong-target records against 339 successes). Selective accuracy at 80% and 50% coverage, and failures caught at 20% abstention, with ties at the cutoff split by their expected value.

**Criteria for P1** (FROZEN section 7; point estimates):
- Success when both hold: P1's AUROC is at least 0.70 in at least 2 of the 3 primary cells, and P1's pooled AUROC is strictly above S5a's.
- Kill: P1's pooled AUROC is below 0.60. A kill overrides success. Otherwise neither.
- Every other signal is reported, not judged, and the intervals do not enter the verdict.

## 2. Checks (all before any AUROC, rate or criterion)

| Check | Result |
|---|---|
| R1, I14's counts and keys | 1,265 records; key sets equal I14's key by key; per-cell n 143 / 141 / 157, 36 × 3 at watching, 141 / 151 / 146 / 143 / 135 for s23 |
| R2, recorded success | Recorded = fresh rescore = I14's = I04's top-1 in 1,265 of 1,265 |
| R3, right target | Equal to I14's in 1,265 of 1,265; wrong-target counts 6 / 14 / 14 (primary) |
| R4, step 0 | All 1,157 realistic records give their recorded place (largest move 0.0 m) |
| R5, A2's D per record | D's success equals A2's in 1,157 of 1,157 (976 D successes on each side); P1 within 1e-4 m of A2's distance in every record (largest gap 5.0e-5 m); C's and D's target numbers equal A2's; no P1 is +inf |
| R6, joins | Every record joins its I04 row with equal flags, cell and layout; every realistic record joins its A2 solver-step row; every record has detector people |
| R7, hashes | `FROZEN.sha256` and A2's `verify_hashes()` pass; A2's script and 8 row files match `A2.sha256` |
| R8, AUROC code | Reproduces phase 1d's Tables 1.5 and 1.7 (V1 concentration and the single flags, per cell, pooled over 549, and s23) within ±0.0005, compared exactly |

- **R8's first attempt** stopped on one value: watch Qwen's V1 concentration gave 0.3125 against the published 0.312. Phase 1d's own raw value is also 0.3125 (`results/still-scenes/learned-models/soft-density/summary.json`), printed as "0.312". The exact gap is 1/2000, inside the inclusive tolerance, but in floating point it evaluates to 5.000000000000004e-4. `DEVIATIONS_2.md` switched R8 to exact comparison before scoring resumed; the verifier judged the fix legitimate.
- **The verifier** reran R1–R8 with its own code: all pass. Its own rebuild of D's people and constraints equals A2's `build` in 1,157 of 1,157, and its own matcher equals A2's in 1,265 of 1,265.
- **Leak check** (verifier). P1 uses only the re-solved place; D's success is stored for R5 only. The visible-truth people and oracle that `repairs.scene_info` returns are unpacked and discarded. `eval_people.json` holds no truth field. S6 and S6z standardise on the training fold, and the verifier's own leave-one-layout-out fit matches every out-of-fold AUROC to 1e-12. `target_correct` only chooses subsets.

**Order** (file times, +08:00): `prefreeze/format_check.py` and `.txt` 17:33:21–28 → `FROZEN.md` (created 17:33:43, read-only 17:34:11) → `FROZEN.sha256` 17:34:16 → `code/confidence.py` (created 17:38:42) → pass 1 17:39:07–17:41:22 → `checks.txt` 17:42:19 → `DEVIATIONS.md` 17:43:02 and its hash 17:43:07 → first pass-2 attempt, stopped at R8, 17:43:14 → `DEVIATIONS_2.md` and its hash 17:43:57 → `code/tables.py` saved 17:44:10 → pass 2 17:44:15–16 → verifier 17:48–17:56. Pass 2 re-verifies every hash before it runs.

## 3. Main tables

Table 3.1. Records per cell (T1).

| Cell | n | Successes | Failures | Wrong target | Base rate |
|---|---|---|---|---|---|
| Old Qwen | 143 | 134 | 9 | 6 | 0.937 |
| Old GLM | 141 | 96 | 45 | 14 | 0.681 |
| Old DeepSeek | 157 | 109 | 48 | 14 | 0.694 |
| **Pooled primary** | **441** | **339** | **102** | **34** | **0.769** |
| Watching (Qwen / GLM / DeepSeek) | 36 / 36 / 36 | 32 / 23 / 23 | 4 / 13 / 13 | 0 / 2 / 0 | 0.722 pooled |
| s23 (GLM / DeepSeek / Haiku / Sonnet / Gemini) | 141 / 151 / 146 / 143 / 135 | 86 / 106 / 134 / 134 / 109 | 55 / 45 / 12 / 9 / 26 | 22 / 15 / 1 / 7 / 26 | 0.795 pooled |

Table 3.2. AUROC for failure, primary cells (T2). Brackets are 95% layout-cluster bootstrap intervals; old Qwen's intervals drop 75 of 2,000 draws (no failure in the resample).

| Signal | Old Qwen (9 failures) | Old GLM (45) | Old DeepSeek (48) | **Pooled, one ranking** | Stratified | Mean of cells |
|---|---|---|---|---|---|---|
| **P1** (judged) | 0.645 [0.392, 1.000] | 0.738 [0.508, 0.951] | 0.688 [0.593, 0.782] | **0.708 [0.564, 0.828]** | 0.703 | 0.690 |
| P1b | 0.850 [0.458, 0.951] | 0.690 [0.467, 0.917] | 0.696 [0.599, 0.780] | 0.691 [0.564, 0.803] | 0.711 | 0.745 |
| S2 | 0.869 [0.419, 1.000] | 0.801 [0.666, 0.906] | 0.650 [0.554, 0.741] | 0.741 [0.644, 0.807] | 0.735 | 0.773 |
| S3 | 0.602 [0.475, 1.000] | 0.789 [0.695, 0.860] | 0.601 [0.517, 0.713] | 0.691 [0.602, 0.773] | 0.677 | 0.664 |
| S4 (2 extra runs) | 0.840 [0.756, 0.976] | 0.715 [0.534, 0.853] | 0.669 [0.588, 0.740] | 0.774 [0.707, 0.841] | 0.707 | 0.742 |
| S4-place (2 extra runs) | 0.826 [0.608, 0.984] | 0.703 [0.529, 0.854] | 0.727 [0.637, 0.818] | 0.795 [0.740, 0.857] | 0.729 | 0.752 |
| **S5a**, any flag (baseline) | 0.478 [0.457, 0.492] | 0.670 [0.552, 0.772] | 0.517 [0.451, 0.604] | **0.595 [0.526, 0.678]** | 0.574 | 0.555 |
| `exclusion_broken` | 0.500 | 0.511 | 0.531 | 0.520 [0.500, 0.547] | 0.520 | 0.514 |
| `far_from_target` | 0.500 | 0.511 | 0.500 | 0.505 [0.500, 0.515] | 0.504 | 0.504 |
| `constraint_unusable` | 0.478 | 0.659 | 0.496 | 0.581 [0.516, 0.664] | 0.559 | 0.544 |
| S5b, 1 − V1 concentration | 0.559 [0.000, 0.791] | 0.303 [0.176, 0.461] | 0.540 [0.427, 0.628] | 0.459 [0.360, 0.568] | 0.447 | 0.467 |
| S6 (includes S4) | 0.882 [0.701, 1.000] | 0.871 [0.767, 0.973] | 0.747 [0.685, 0.812] | 0.849 [0.775, 0.907] | 0.812 | 0.833 |
| S6z (zero extra calls) | 0.825 [0.446, 1.000] | 0.780 [0.602, 0.948] | 0.689 [0.570, 0.780] | 0.752 [0.632, 0.844] | 0.741 | 0.765 |

- Paired difference P1 − S5a, pooled: +0.112 [−0.088, +0.273] (descriptive; the criterion uses point values).
- Pooled over all 549 I14 primary records (old and watching; point values only): S2 0.742, S3 0.700, S4 0.759, S4-place 0.771, S5a 0.578, S5b 0.470; single flags 0.515, 0.508, 0.563.
- (Post hoc, this page's reading of the table; not a verdict.) Read against P1's bars, S2 (0.869, 0.801), S4 (0.840, 0.715) and S6z (0.825, 0.780) each reach 0.70 in 2 of 3 cells with a pooled value above S5a's, and S6 reaches it in 3 of 3. S3 and P1b reach it in 1 of 3. None of them was the judged signal, all were designed knowing A2's mechanism, and checking several signals against one bar favours finding a pass.

Table 3.3. Right-target records only (T3; 407 records, 68 failures: old Qwen 3, GLM 31, DeepSeek 34). The subset is chosen by an outcome field, so it is descriptive.

| Signal | Old Qwen | Old GLM | Old DeepSeek | Pooled [95%] |
|---|---|---|---|---|
| P1 | 0.944 | 0.757 | 0.717 | 0.750 [0.538, 0.904] |
| P1b | 0.869 | 0.701 | 0.711 | 0.698 [0.512, 0.847] |
| S2 | 0.961 | 0.838 | 0.646 | 0.753 [0.612, 0.858] |
| S3 | 0.830 | 0.823 | 0.619 | 0.730 [0.613, 0.825] |
| S4 | 0.927 | 0.645 | 0.609 | 0.741 [0.671, 0.818] |
| S4-place | 0.979 | 0.704 | 0.771 | 0.827 [0.771, 0.875] |
| S5a | 0.478 | 0.674 | 0.530 | 0.610 [0.532, 0.690] |
| S5b | 0.333 | 0.313 | 0.538 | 0.455 [0.335, 0.605] |
| S6 | 0.971 | 0.869 | 0.746 | 0.858 [0.776, 0.926] |
| S6z | 0.954 | 0.813 | 0.714 | 0.784 [0.619, 0.909] |

- Old Qwen's right-target intervals drop 241 draws; its 3 failures make those values unstable.

Table 3.4. Wrong-target AUROC (T4; positives are the wrong-target records, all failures; negatives the successes; right-target failures left out; point values).

| Signal | Old Qwen (6) | Old GLM (14) | Old DeepSeek (14) | Pooled (34 against 339) |
|---|---|---|---|---|
| P1 | 0.495 | 0.694 | 0.616 | 0.624 |
| P1b | 0.840 | 0.666 | 0.660 | 0.678 |
| S2 | 0.823 | 0.719 | 0.659 | 0.716 |
| S3 | 0.489 | 0.713 | 0.558 | 0.613 |
| S4 | 0.797 | 0.870 | 0.817 | **0.841** |
| S4-place | 0.750 | 0.700 | 0.623 | 0.730 |
| S5a | 0.478 | 0.662 | 0.484 | 0.566 |
| S5b | 0.672 | 0.281 | 0.545 | 0.468 |
| S6 | 0.837 | 0.876 | 0.749 | 0.833 |
| S6z | 0.761 | 0.705 | 0.631 | 0.689 |

- S4's target rule (+inf when trials' targets lie more than 1 m apart) is what sees wrong targets: without it (S4-place) the value falls from 0.841 to 0.730.

Table 3.5. Abstaining on the least confident records, pooled primary (T5). Base rate 0.769; 102 failures (34 wrong-target, 68 right-target).

| Signal | Success among kept, 80% coverage | 50% coverage | Failures caught at 20% abstention (share) | Wrong-target caught (of 34) | Right-target caught (of 68) |
|---|---|---|---|---|---|
| Random | 0.769 | 0.769 | 20.4 (0.200) | 6.8 | 13.6 |
| **P1** | 0.868 | 0.853 | 55.3 (0.542) | 11.0 | 44.3 |
| P1b | 0.853 | 0.859 | 50.2 (0.492) | 16.0 | 34.2 |
| S2 | 0.856 | 0.873 | 51.1 (0.501) | 14.4 | 36.7 |
| S3 | 0.846 | 0.846 | 47.7 (0.467) | 11.0 | 36.7 |
| S4 | 0.816 | 0.933 | 37.0 (0.363) | 23.8 | 13.2 |
| S4-place | 0.827 | 0.941 | 41.1 (0.403) | 8.0 | 33.1 |
| S5a | 0.809 | 0.809 | 34.4 (0.338) | 9.6 | 24.8 |
| S5b | 0.753 | 0.791 | 15.0 (0.147) | 2.0 | 13.0 |
| S6 | 0.871 | 0.955 | 56.6 (0.555) | 13.6 | 43.0 |
| **S6z** | **0.878** | 0.891 | **59.0 (0.578)** | 12.0 | 47.0 |

- P1 per cell at 80% / 50% coverage, with failures caught: old Qwen 0.957 / 0.958, 4.1 of 9; old GLM 0.832 / 0.810, 26.0 of 45; old DeepSeek 0.780 / 0.803, 20.4 of 48.
- S4 is +inf in 100 of the 441 primary records (22.7%), so its 20% cut falls inside that tie group and the expected-value tie rule spreads the abstention over all 100; this is why its gain at 80% coverage is small next to its gain at 50%.

Table 3.6. Secondary cells, AUROC for failure (T6; reported, not judged).

| Signal | Watch Qwen (4) | Watch GLM (13) | Watch DeepSeek (13) | **Watching pooled (30)** |
|---|---|---|---|---|
| S2 | 0.711 | 0.841 | 0.610 | 0.717 [0.553, 0.802] |
| S3 | 0.711 | 0.808 | 0.610 | 0.702 [0.481, 0.797] |
| S4 | 0.781 | 0.706 | 0.545 | 0.695 [0.533, 0.798] |
| S5a | 0.500 | 0.594 | 0.435 | 0.524 [0.450, 0.637] |
| S5b | 0.312 | 0.579 | 0.627 | 0.500 [0.293, 0.704] |

| Signal | s23 GLM (55) | DeepSeek (45) | Haiku (12) | Sonnet (9) | Gemini (26) | **s23 pooled (147)** |
|---|---|---|---|---|---|---|
| P1 | 0.725 | 0.737 | 0.197 | 0.562 | 0.501 | 0.641 [0.500, 0.789] |
| P1b | 0.670 | 0.728 | 0.190 | 0.550 | 0.489 | 0.620 [0.493, 0.744] |
| S2 | 0.759 | 0.704 | 0.478 | 0.525 | 0.579 | 0.692 [0.636, 0.738] |
| S3 | 0.755 | 0.617 | 0.500 | 0.500 | 0.510 | 0.646 [0.583, 0.697] |
| S4 | 0.765 | 0.736 | 0.863 | 0.877 | 0.568 | 0.803 [0.754, 0.859] |
| S5a | 0.611 | 0.583 | 0.430 | 0.529 | 0.558 | 0.564 [0.511, 0.618] |
| S5b | 0.318 | 0.607 | 0.683 | 0.898 | 0.532 | 0.531 [0.405, 0.672] |
| S6 | 0.880 | 0.844 | 0.677 | 0.938 | 0.513 | 0.824 [0.785, 0.882] |
| S6z | 0.760 | 0.774 | 0.139 | 0.657 | 0.414 | 0.665 [0.552, 0.785] |

- P1 is not computed at watching, so S6 and S6z are not either. At watching the detector's recall is 0.845 (placeholder images), so S2 and S3 mix its misses with the VLM's errors.
- The s23 GLM and DeepSeek people answers are copies of s19/s20, so their perception signals are not new evidence.
- Haiku's P1 of 0.197 ranks its 12 failures (11 right-target) below its successes. Gemini's 26 failures are all wrong targets, which I11 Stage 0 classed as choice errors; no perception signal sees them (P1 0.501, S2 0.579). Sonnet's intervals drop 223 draws.
- s23 selective accuracy (T6.s23 d): S6z keeps 0.888 at 80% coverage against a base of 0.795 and catches 83.0 of 147 failures; P1 keeps 0.881 and catches 78.7.

Table 3.7. S6 and S6z coefficients, primary (T7; full-data fit on standardised features, λ = 1, intercept unpenalised).

| Model | Intercept | P1 (log) | S2 | S3 | S4 (log) | S4 = +inf |
|---|---|---|---|---|---|---|
| S6 | −1.640 | +0.817 | +0.899 | −0.164 | +0.965 | −0.128 |
| S6z | −1.425 | +0.996 | +0.911 | −0.204 | — | — |

- Every fold converged within 6 Newton iterations. The P1 = +inf indicator is constant (no record has it), so it gets weight 0. S3's weight is small and negative once S2 is in (inference: the two overlap).

**Descriptive counts** (T8).
- P1 is exactly 0 in 7 / 27 / 14 primary records (old Qwen / GLM / DeepSeek), 48 in all (14 failures, 34 successes), and at least 1 cm in every other record: 122 between 1 and 10 cm, 207 between 10 cm and 1 m, and 64 at 1 m or more (verifier). It is +inf in 0.
- P1b is +inf (target person changes) in 27 / 13 / 19 primary records.
- S4 is +inf in 202 records overall: 8 for a missing place (all old GLM; 4 trial groups with a `no_target` trial) and 194 for trials whose targets lie more than 1 m apart (92 of them primary).

## 4. Pre-registered criteria and outcomes (FROZEN section 7; 441 primary records)

| Criterion | Value | Outcome |
|---|---|---|
| Stop rule: checks R1–R7 | All pass | Met |
| Stop rule: R8 reproduces phase 1d's values | All pass after the exact-comparison fix (`DEVIATIONS_2.md`) | Met |
| Success (a): P1's AUROC ≥ 0.70 in at least 2 of 3 primary cells | 0.645, 0.738, 0.688: 1 of 3 | Not met |
| Success (b): P1's pooled AUROC strictly above S5a's | 0.708 against 0.595 | Met |
| Kill: P1's pooled AUROC below 0.60 | 0.708 | Does not fire |
| **P1 overall** | (a) fails, kill does not fire | **Neither** |
| Prediction part 1: P1 ≥ 0.70 in old GLM and old DeepSeek, < 0.60 in old Qwen | GLM 0.738, DeepSeek 0.688, Qwen 0.645 | Does not hold |
| Prediction part 2: S4 strictly highest pooled among P1, S2, S3, S4, S5a, S5b | S4 0.774; S2 0.741, P1 0.708, S3 0.691, S5a 0.595, S5b 0.459 | Holds |
| Prediction part 3: pooled wrong-target AUROC below 0.60 for P1, S2 and S3 | 0.624, 0.716, 0.613 | Does not hold |

## 5. Post hoc (the verifier's; changes no verdict)

The verifier added these checks after seeing the results. Some use subsets that depend on the outcome, so they are diagnostics.

- **+inf reading.** No P1 is +inf, because D gave a place in all 1,157 records. Under the alternative reading P1b, cells 0.850 / 0.690 / 0.696 and pooled 0.691: still 1 of 3, still neither.
- **Rounding.** Irrelevant at the frozen 1e-6 m, at 1 mm and with the step-0 reference place (identical AUROCs). Raw floats give 0.633 / 0.735 / 0.688 and 1 cm gives 0.656 / 0.738 / 0.688: neither either way.
- **Ties decide the margin.** With ties broken all against or all for P1, the AUROC ranges are Qwen [0.616, 0.673], GLM [0.709, 0.767] and DeepSeek [0.670, 0.706]. If every tie broke in P1's favour, DeepSeek would reach 0.706 and the verdict would be success. The frozen rule (ties count half) gives 0.688.
- **Right-target subset.** On right-target records alone P1 would meet both success conditions (0.944 / 0.757 / 0.717, pooled 0.750 against S5a's 0.610). The subset is chosen by an outcome field, so it cannot be a criterion.
- **Single layouts.** Dropping `eval_01` (DeepSeek 0.718), `eval_11` (GLM 0.722, DeepSeek 0.707) or `eval_12` (Qwen 0.824, GLM 0.773) turns the verdict into success: 3 of 13 layouts. Across all single-layout drops the cells range over Qwen 0.5515–0.8238, GLM 0.6821–0.8692 and DeepSeek 0.6535–0.7184, and the pooled value over 0.6784–0.7671.
- **Single trials.** Trial 0 alone gives neither (0.654 / 0.712 / 0.648); trial 1 alone (0.764 / 0.765 / 0.672) and trial 2 alone (0.213 / 0.764 / 0.724) each give success.
- **Bootstrap.** The success condition holds in 50.1% of the frozen-seed draws and the kill in 6.7%. Each cell reaches 0.70 in 34.4% (Qwen), 65.2% (GLM) and 41.2% (DeepSeek) of draws. The data cannot tell neither from success.
- **Exposure.** The 37 records where D's verdict differs from C's (Qwen 2, GLM 16, DeepSeek 19, one of them a break) were known before the freeze. All 37 have P1 of at least 0.539 m. Used as a score, that flip indicator (it uses the outcome) gives 0.611 / 0.678 / 0.683, pooled 0.675, against P1's 0.645 / 0.738 / 0.688 and 0.708. On the 404 records where D's verdict equals C's, P1 gives 0.543 / 0.594 / 0.532, pooled 0.557. So most of P1's ranking comes from records A2 had already shown would change. This is not a leak in the code, but the test is not independent of what the designers knew. The same check was not run for S2 or S3.
- **Model identity.** Each cell's failure rate as the score gives 0.658 pooled, above S5a's 0.595, so success clause (b) is a low bar.

## 6. Where the compute agent and the verifier differ

Every criterion verdict and every number agrees. The verifier reproduced all 1,265 records signal by signal (0 differences) and all 2,725 table numbers to 1e-12 with its own AUROC method (brute-force pairwise concordance gathered into layout-by-layout matrices).

Settled (the verifier's value or this page's count is used above):
1. **Undefined dropped-draw counts.** For s23 right-target Gemini (0 failures) the compute agent records 2,000 dropped draws and the verifier records none (undefined), in 13 fields. Same meaning; the AUROC is undefined there.
2. **`DEVIATIONS.md`'s time.** Its text says "about 17:45"; its file was written at 17:43:02 and hashed at 17:43:07, before pass 2's first attempt (17:43:14). The compute agent disclosed this too. The order is unaffected.
3. **R8's number of values.** `DEVIATIONS_2.md` speaks of one failure "of 26" values and the verifier of "all 34". The check lists 33 values in both `tables.txt` and `verify/vtables.out` (24 per cell, 4 pooled over 549, 5 for s23). Nothing else depends on the count.
4. **"P1 ≥ 0.54 m" for the flip records.** The verifier's summary says all 37 flip records have P1 of at least 0.54 m; its own output gives a smallest value of 0.5385 m (`verify/vexposure.out`). Section 5 says 0.539 m.

Not independently re-checked:

Disclosed housekeeping (neither changes a number):

---

## 7. What it means (inference)

**For P1, the judged signal.**
- The pre-registered verdict is neither, and it stands as registered. But it is knife-edge: one layout, one trial or the tie rule moves it to success, and half the bootstrap draws pass. The honest reading is "between neither and success", not "fails".
- P1 is also not independent evidence. Its ranking rests mostly on the 37 records where A2's repair changes the verdict, which the designers knew about; on the other 404 records P1 is close to chance (0.557, an outcome-dependent subset).
- So P1's information largely overlaps with what A2's setting D already repairs. If the pipeline took D (or the detector) as its people source, most of the failures P1 flags would be fixed, not flagged, and a confidence signal for that pipeline would have to come from elsewhere.

**For the distribution and confidence question** (phase 1d section 5, question 1: how likely is the chosen pose to be acceptable?).
- **Phase 1d's suggestion holds in direction.** Signals built from perception disagreement rank C's failures well above anything earlier on the same records: S2 0.741, S6z 0.752 and P1 0.708 pooled, against 0.595 for the flags and 0.459 for I04's density concentration. The stratified values (0.735, 0.741, 0.703) say this is not just which model answered. These are descriptive numbers for signals designed after seeing A2, and only P1 was judged.
- **But only where failures are perception failures.** For GLM and DeepSeek, whose people answers carry unconfirmed detections, S2 and P1 work. For the new-wording API models they carry little: P1 0.197 / 0.562 / 0.501 and S2 0.478 / 0.525 / 0.579 for Haiku, Sonnet and Gemini. Their failures are placement (Haiku, 11 of 12 right-target) or target choice (Gemini, 26 of 26 wrong targets; I11 Stage 0 called them choice errors). This matches I01's split of failures into target finding and placement.
- **As an abstain rule.** On these development records, abstaining on the least confident 20% by S6z (detector plus one solver run, no extra model call) raises C's success among the kept records from 0.769 to 0.878 and catches 59.0 of 102 failures, against 20.4 by chance. That is the first zero-call number in this project that would make "ask for help" worth considering. It is untested on unseen records.
- **A distribution is still not the answer.** The useful signals here are scalars from comparing two perception sources or three trials, not spread over the solver's floor. (inference) P1 is the one-hypothesis case of I12's Monte Carlo over perception hypotheses, with the detector as the only alternative. A density built from many perception hypotheses would be the natural distribution-shaped version, and it is untested. Its value would have to show on the records the detector repair does not change, where P1 has little left.

**What it does not show.**
- That P1 fails: the data cannot separate neither from success.
- That S2 or S6z work on new records: they were not judged, and they were designed knowing A2's mechanism.
- That these signals survive the detector's behaviour on unseen layouts. Phase 2's confirmation found 0.242 false people per image there, against 0.071 on these scenes (`results/still-scenes/people-finder/early-detector/phase2/README.md` section 1). S2 counts every false detector person as disagreement, so on new layouts it would likely rise on successes too (untested).
- Anything about social correctness: the labels are the project's own rules.

**For other entries.**
- **I12.** P1's pooled 0.708 sits between I12's kill line (0.65) and its success bar (0.72), on a different and simpler test (no K = 32 sampling, no net-gain criteria).
- **I09.** The detector gains a second possible use, as a confidence check on the VLM's people, at about 56 ms with no extra model call. That use inherits the detector's open false-people question.

3. **Scope of a confidence output.** Is an abstain or ask-for-help output in scope for the pipeline at all, beyond the (x, y, θ) goal? Possibly with the mentor.
4. **Trial disagreement.** Is S4's cost (two extra full pipeline runs per goal step) acceptable for the cases only it sees (wrong targets, the API models' placement failures)? This ties into I02's and I39's open cost and panel questions.
5. **I15.** Does a perception-disagreement AUROC of 0.708–0.741 meet the parked row's revival condition, which names I14 or I04?
6. **A zero-call follow-up** this page did not run: S2's and S3's AUROC on the records whose verdict D does not change (the exposure check above covered P1 only). It would show whether the detector comparison keeps any value once D's repairs are taken out.

---

## 9. Proposed REPORT edits (not applied)

Each item gives the exact old text from `REPORT.md` as it stands now and the proposed new text. Each old text occurs exactly once in the report (checked with Python's `str.count`). Applying items 1–12 in order works: on a copy in memory, each old text still occurs exactly once when its turn comes, and each new text occurs once afterwards. Items 1–8 alone, in order, also apply cleanly. Items 9–12 are optional. New text carries the tag [phase 3]; item 1 defines it. Before applying, keep a copy of the report as `REPORT.before-phase3-confidence.md` (item 8 names it).

**Merging with another phase-3 page.** Another phase-3 page may propose its own [phase 3] definition, opening-line entry and revision-note bullet.
- Item 1: keep one [phase 3] bullet in "How to read", right after the [phase 2] bullet. If another page's bullet is already there, do not add a second one. Add this page to that bullet's list of pages as "confidence from perception disagreement (`results/still-scenes/confidence-signal/README.md`)", and carry over only what that bullet lacks from item 1: zero calls, CPU only, no new input (whitelisted development runs plus earlier phases' frozen outputs), and that this page's design is data-informed. If both pages are applied together, write one bullet that names both.
- Item 2: list every phase-3 page in one parenthesis in the opening line.
- Item 8: the revision-note bullet goes at the very end of the revision note. If another bullet has been appended after this item's old text since this page was written, put this bullet after the last bullet instead, using that bullet's final sentence as the old text.
- Items 3–7 and 9–12 touch sentences no other phase-3 page is expected to touch. If an old text no longer occurs exactly once, re-anchor on the same sentence.

### 1. "How to read": define [phase 3]

Old:
New:

### 2. Opening status line: the list of pages

Old:
```text
C5's exploratory pilot), nothing has been built
```
New:
```text
C5's exploratory pilot) and phase 3's zero-call confidence probe (`results/still-scenes/confidence-signal/README.md`) [phase 3], nothing has been built
```

### 3. Opening status line: what phase 3 answers

Old:
New:

### 4. I04: the phase-3 follow-up on confidence

Old:
New:

### 5. I09: a second use, as a confidence check

Old:
New:

### 6. I12: the one-hypothesis case

Old:
```text
  - Kill: net gain of +2 or less; more than 3 Qwen records lost; drift above 0.25 m; AUROC below 0.65.
```
New:
```text
  - Kill: net gain of +2 or less; more than 3 Qwen records lost; drift above 0.25 m; AUROC below 0.65.
  - [phase 3] A one-hypothesis version has been measured (`results/still-scenes/confidence-signal/README.md`): the shift of C's place when the solver re-runs with I09's detector people in place of the VLM's (A2's setting D) predicts C's failure with a pooled AUROC of 0.708 on the 441 old-wording realistic goal-step records (0.645, 0.738 and 0.688 by model; its own pre-registered verdict: neither). That sits between this entry's kill line (0.65) and success bar (0.72), on a simpler test with no sampling and no net-gain criteria. On the 404 records where D does not change the verdict it falls to 0.557 (post hoc, outcome-dependent subset). (inference) A K-sample version would have to show its value on those records.
```

### 7. Section 7: a decision on confidence output

Old:
```text
- Per-situation comparisons reported as descriptive only, since 3–5 layouts cannot reach p < 0.05 (I46).
```
New:

### 8. Revision note

Old:
```text
The report before them is kept as `REPORT.before-phase1d-check.md`.
```
New:
```text
The report before them is kept as `REPORT.before-phase1d-check.md`.
- **Phase 3, confidence from perception disagreement (18 Oct).** A zero-call probe asked whether perception disagreement predicts when C fails (`results/still-scenes/confidence-signal/README.md`), on the CPU, from whitelisted development runs and earlier phases' frozen outputs. The protocol note was frozen and hashed before any signal was computed against an outcome (data-informed: written knowing phase 1d's and A2's results), and an independent verifier reproduced every signal on all 1,265 records and all 2,725 table numbers with its own code. The judged signal P1, the shift of C's place when the solver re-runs with the VLM's people moved onto I09's detector people and the unsupported ones dropped (A2's setting D), is neither: AUROC for failure 0.645, 0.738 and 0.688 on the old-wording realistic cells (2 of 3 at 0.70 needed), pooled 0.708 against the flags' 0.595, kill line 0.60 not crossed. The verdict is fragile (dropping any one of 3 layouts, or using trial 1 or 2 alone, turns it into success; the success condition holds in half the bootstrap draws), and most of P1's ranking comes from the 37 records A2 had already shown would change (0.557 on the others; post hoc). Reported, not judged: people disagreement with the detector 0.741 pooled, trial disagreement 0.774 (two extra pipeline runs), and a zero-call combination 0.752 that keeps 0.878 success at 80% coverage; for the new-wording API models the perception signals carry little (P1 0.197–0.562). "How to read" defines [phase 3], and the opening lines, I04, I09, I12 and section 7 carry the result; optionally also section 1.2 (item 9), section 3.6, I39 and the I15 parked row. The report before these changes is kept as `REPORT.before-phase3-confidence.md`.
```

### 9. Section 1.2, item 9 (optional)

Old:
```text
(AUROC 0.303–0.559 on the old-wording realistic cells).
- **Decision gate**
```
New:
```text
(AUROC 0.303–0.559 on the old-wording realistic cells). [phase 3] A zero-call follow-up tested perception disagreement as the confidence signal instead (`results/still-scenes/confidence-signal/README.md`): the shift of C's place under I09's detector people gives a pooled AUROC of 0.708 for failure, a fragile pre-registered neither, and the people disagreement with the detector gives 0.741 (reported).
- **Decision gate**
```

### 10. Section 3.6, "For us" (optional)

Old:
```text
**For us:** beat plain voting and random picks first. Disagreement across trials and across models is a cheap signal.
```
New:
```text
**For us:** beat plain voting and random picks first. Disagreement across trials and across models is a cheap signal. [phase 3] Measured on the 441 old-wording realistic goal-step records (`results/still-scenes/confidence-signal/README.md`): disagreement across C's three trials predicts failure with a pooled AUROC of 0.774 (but costs two extra runs), and disagreement between the VLM's people and a local detector's 0.741, against 0.595 for the solver's flags.
```

### 11. I39: trial disagreement on the old records (optional)

Old:
```text
    - The pitched version that reuses GLM's plan tops out near 207.
```
New:
```text
    - The pitched version that reuses GLM's plan tops out near 207.
    - [phase 3] Phase 3's S4 (the largest place distance among an episode's three C trials, infinite when a trial has no place or their targets lie more than 1 m apart) gives AUROC 0.715 on old GLM, 0.840 on old Qwen and 0.669 on old DeepSeek, and 0.863 and 0.877 on s23 Haiku and Sonnet (`results/still-scenes/confidence-signal/README.md`, reported, not judged). It is the only signal there that sees wrong targets well (0.841 pooled).
```

### 12. Section 4.5, the I15 parked row (optional)

Old:
```text
[phase 1] I04's first probe did not (0.303–0.559 on the old-wording realistic cells, `results/still-scenes/overview/phase-notes/phase1d.md` section 1) |
```
New:

---

## 10. Follow-up after this page: the exposure check for S2, S3 and S6z (post hoc; the research lead, 18 Oct)

AUROC for failure, pooled over the three primary cells (one ranking), with 95% layout-cluster bootstrap intervals from
the frozen draw matrix (old Qwen / GLM / DeepSeek in brackets where useful):

| Signal | All 441 records | The 404 where D's verdict equals C's (66 failures, 34 of them wrong-target) |
|---|---|---|
| P1 (judged) | 0.708 | 0.557 [0.455, 0.673] |
| S2, people disagreement | 0.741 | **0.694 [0.587, 0.743]** (0.832 / 0.751 / 0.597) |
| S3, named-person support | 0.691 | 0.625 [0.557, 0.690] |
| S4, trial disagreement (2 extra runs) | 0.774 | 0.784 [0.711, 0.871] |
| S6z, frozen zero-call combination | 0.752 | 0.635 [0.515, 0.721] |
| S5a, flags | 0.595 | 0.622 [0.525, 0.721] |
| Model identity (pooled only) | 0.658 | 0.651 |

- **S2 keeps most of its value; P1 does not.** S2 stays above 0.5 pooled in every bootstrap draw on the 404 records.
  As an abstain rule it keeps 0.897 success at 80% coverage (base 0.837) and catches 32.8 of 66 failures (random 13.2),
  level with S4 (0.889, 30.2), which costs two extra runs.
- **But it is not clearly better than the flags there:** S2 − S5a is +0.071 [−0.083, +0.182]. Its edge is in
  wrong-target failures (0.716 against the flags' 0.566); on the 32 right-target failures it is 0.671 against 0.683.
  Per cell, S2 − S5a is +0.354 (Qwen, 7 failures), +0.045 (GLM) and +0.052 (DeepSeek), every interval including 0.
- **Two readings corrected by the verifier.** (1) The compute agent wrote that adding P1 makes the combination worse.
  The gap comes from the leave-one-layout-out fit itself: S2 alone fitted the same way scores 0.623 on the 404, and
  adding P1 to it changes nothing measurable (+0.009 [−0.058, +0.077]). Raw S2 remains the best zero-call score as
  measured. (2) "Its edge comes from GLM" does not hold for the edge over the flags (GLM's is the smallest); GLM does
  carry S2's own AUROC (0.751 [0.652, 0.874]; the Qwen and DeepSeek intervals include 0.5).
- **The 37 flips** hold 36 failures and 1 success, so no AUROC there is meaningful.
- The verifier reproduced all 531 printed numbers at printed precision; on the full 441 records every value equals
  `summary.json` exactly.
