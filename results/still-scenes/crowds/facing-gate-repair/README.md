# Facing gate on joining a line (repair probe on the crowd scenes)

**Question.** Does a rule that lets a non-listed person join a waiting line only if they face into the line's front half-plane (RF90) fix the crowd failures where the solver absorbs a bystander, without breaking cases that already work?

**Result.** On the crowd scenes RF90 turns 29 of the 54 judged line failures into successes (27 were needed) and breaks none of the 30 bases or 46 group and audience variants; 24 of the 29 are the A1 and A2 variants, which the gate removes by construction, and A3 stays at 0 of 15 under every rule. On 549 primary development goal steps RF90 has 0 breaks in each of six cells (nets 0, 0, +1, 0, 0, 0), but only 4 steps had an absorption to change. On simulated people (arm 1b) it raises mean line success by 0.10 to 0.14 per model with 0 breaks. The baseline rule is reproduced exactly everywhere (229 of 229 scenes, 10,080 of 10,080 noisy draws, 1,265 of 1,265 goal steps).

**Files.**
- `FROZEN.md`, `DEVIATIONS.md`, `logs/`: protocol, the one change after it, hash and order logs.
- `code/`: `rules.py` (the four rules R0, RF90, RF60, RN), `common.py`, `crowd.py` (arm 1), `crowd_noisy.py` (arm 1b), `records.py` (arm 2), `tables.py`, `run_*.sh`.
- `tables.txt`, `summary.json`, `rows_crowd.jsonl`, `rows_noisy_*.jsonl`, `rows_records_*.jsonl`: results; `out/r0_*.txt`: baseline reproductions; `work/`: per-pass rows.
- `precheck/format_check.py`: baseline format check before the freeze.
- `verify/`: independent re-implementation (`vturn.py`, `v_*.py`) with its reports in `out/`.

**Reproduce.** From the repository root with `PYTHONPATH=src`: `bash results/still-scenes/crowds/facing-gate-repair/code/run_crowd_r0.sh`, then `run_r0_rest.sh`, `run_repair.sh`, and `python results/still-scenes/crowds/facing-gate-repair/code/tables.py`. These use the step-0 harness (`results/still-scenes/analysis-scripts/harness/`) and the development scenes in `data/dev_scenes/` (release dev-scenes-v1), which are in this repository, but also the rendered crowd-variant images, which are not (see `../README.md`), and `common.start()` asserts the sha256 of the original `FROZEN.md` and `src/sgi/solver.py`, which no longer match the files here. The verifier analyses run here and reproduce their stored reports: `python results/still-scenes/crowds/facing-gate-repair/verify/v_arm1.py` (also `v_arm1b.py`, `v_arm2.py`).

**Data.** `data/dev_scenes/` (release dev-scenes-v1); `results/logs/research-data.tar.gz` (row files, `work/`, `verify/out/`); the baseline rows come from `../score/` and `../../replay-harness/repairs/`.

## Notes

# Phase 2: I06 repair probe, a facing gate on joining a line (2026-10-18)

How it was made:
- **After the freeze.** `DEVIATIONS.md` (sha256 `7a909449…1624a`) holds two entries. D1 moves files only (code in `code/`, final rows at the top of this folder). D2: arm 1b's RF90 pass simulates each draw again with I06's seed and asserts that the simulated people hash to R0's. Both were hashed at 06:43:26, 14 s before the first pass; the time the entries state (06:45) is wrong (section 6).
- **This page.** The writer ran no arm. A few counts were tallied from recorded rows (the verifier's `verify/out/crowd_seed*.jsonl` and `verify/out/records_old_dsv41.jsonl`); each is marked "writer's tally". Numbers computed from the tables are marked "writer's arithmetic". The writer also re-checked both hash files and every output hash in `logs/output_hashes.txt` (all still match), and the stray file of section 6 (by its file time).
- **How the rules run.** Each rule is research code installed at run time, in the probe's own processes, as a replacement of `sgi.solver._turn`; the package was not edited. Every script asserted the sha256 of FROZEN.md and of `src/sgi/solver.py` at start, and `rules.py` had the same hash (`352d0e7a…`) at all 35 script starts.
- **Inputs.** I06's 229 kept crowd scenes (seeds 0–2), loaded with I06's `lib.load`, which calls the harness's scene guard; I06's scoring code and noise model, unchanged (hashes logged). Development runs, read through the step-0 harness: s19 `qwen38_low`, s19 `glm53` and s20 `dsv41` (arms real, staffed and plain, and the audience goal steps of dev_seed0–2), and the new-wording s23 runs of `glm53`, `dsv41`, `haiku55`, `sonnet55_low` and `gemini38_low` (arms real, staffed and plain). I01's and A2's row files as R0 targets, and I09's frozen detector people through A2's `replay_d.py`.
- **Compute.** `nice -n 10`, 2 BLAS threads, at most 2 processes, with at least 11 GB of memory free throughout. About 1 h 20 min of wall time for the compute agent and about 1 h for the verifier.
- The structured outputs of the compute agent and the verifier (and the protocol author's) are in `agent_outputs.json`, unedited.

Paths below are relative to this folder (`results/still-scenes/crowds/facing-gate-repair/` in the research folder).
- Protocol: `FROZEN.md`, `DEVIATIONS.md` (each hashed; `logs/deviations_hash_history.txt`); `precheck/` (`format_check.py`, `format_check.log`).
- `code/`: `rules.py` (the four rules), `common.py`, `crowd.py` (arm 1), `crowd_noisy.py` (arm 1b), `records.py` (arm 2), `tables.py`, `run_crowd_r0.sh`, `run_r0_rest.sh`, `run_repair.sh`.
- Results: `tables.txt`, `summary.json`, `rows_crowd.jsonl`, `rows_noisy_{qwen,glm,dsv}.jsonl`, `rows_records_<cell>.jsonl` (11 cells); the R0 reproduction files `out/r0_*.txt`; per-pass rows in `work/`; `logs/` (`ORDER.txt`, `output_hashes.txt`, `git_status_before.txt`, `git_status_after.txt`, run logs).
- Verifier: `verify/` (`vturn.py`, `v_*.py`, `run_*.sh`, `job.sh`; `out/arm1_report.txt`, `out/arm1b_report.txt`, `out/arm2_report.txt`, its own rows; `logs/`).

## 0. Verdicts at a glance

| Analysis | Pre-registered outcome | One-line reading (inference) |
|---|---|---|
| R0 reproductions | **Full** everywhere: I06's arm 3 on 229 of 229 scenes; I06's noisy draws 10,080 of 10,080; recorded success 1,265 of 1,265 goal steps; I01's P 1,265 of 1,265; A2's D 1,157 of 1,157 | Every comparison is against the exact baseline. |
| S1, crowds (arm 1: I06's scenes, oracle C, L1) | **Holds.** RF90 turns 29 of the 54 judged line failures into successes (27 needed) and breaks none of the 30 bases or 46 group and audience variants; no break anywhere | Narrow: 24 of the 29 are A1–A2, which the gate turns away by construction. A3 stays 0 of 15 under every rule; RF90 fixes 5 of the 15 B–D failures. |
| K2 | Does not fire (29; fires at 13 or fewer) | |
| S2, real records (arm 2, setting (a)) | **Holds.** 0 breaks in each of the six primary cells; nets 0, 0, +1, 0, 0, 0 | Barely tested: R0 absorbed someone in only 4 of the 549 primary goal steps. |
| K1 | Does not fire (most breaks in a cell: 0; pooled net +1) | |
| Not judged: RF60, RN, settings (b) and (c), new wording, arm 1b | Crowds: RF60 fixes 30 and RN 35 of 54, with no break. Records: in (b) and (c) RF90 changes nothing in any cell; RN loses up to 5 successes per cell. Arm 1b: RF90 raises mean line L1 by 0.10–0.14 per model, with 0 per-draw breaks | Real lines need some absorption, and the people absorbed there face the line's front, so the gate keeps them. |

Shared cautions:
- **Development records.** Already studied in I01, I14 and A2. The watching cells have n 36. Judgements rest on point estimates.
- **Only absorptions can change.** The gate only removes candidates, so a scene or record can change only where R0 absorbed someone (asserted on every solve). That is 99 of I06's 229 scenes, all R0 failures, so arm 1 could show fixes but no breaks; on the records it is 4 of 549 primary goal steps in setting (a).
- **Yaw sources differ.** Arm 1: true yaw. Arm 1b: simulated yaw. Arm 2: (a) the model's own yaw; (b) true yaw; (c) the I09 detector's yaw, falling back to the model's.
- **Unit.** Arm 1: a scene (e0's verdict; both episodes agree under every rule, asserted). Arm 1b: a draw. Arm 2: a goal step, keyed by (run arm, scene, episode, trial).

---

## 1. The rules (`code/rules.py`; FROZEN section 2)

Verbatim (lead):
- R0: the package's `_turn` unchanged (baseline; must reproduce I06's arm 3 and the recorded verdicts exactly).
- RF90 (primary): a non-listed person may join only if their facing points into the line's front half-plane: with u = the unit vector from that person to the current last member, and f = the person's facing unit vector, require f · u ≥ cos(90°) = 0 (strictly: f · u > −1e-9). A person with unknown yaw joins as in R0.
- RF60 (secondary): the same with f · u ≥ cos(60°).
- RN (ablation): nobody joins (only listed waiting people form the line).
- Everything else in the solver is unchanged.

Readings (the protocol author's choices):
- `turn_copy` is a line-for-line copy of `_turn` with one extra condition on the candidate test. "Current last member" is the loop's current last, so it can be someone who joined in an earlier iteration. The cosines are the constants 0.0 and 0.5, with tolerance 1e-9. A person at zero distance from the last member joins as in R0. The gate only removes candidates; ordering, ties and the loop are R0's.
- The solver works in the robot's frame, a rotation, so f · u is the same as in the world frame.

Checks, all passed:
- **Copy check.** On every call in the R0 passes, the research copy with no gate returned what the package's `_turn` returned (compute agent: 0 disagreements; verifier, with its own copy: 0 of about 15,400 calls).
- **Static check.** `rules.py` contains none of the tokens situation, role or group (the verifier also checked label, truth and members). The rule reads only body positions and yaws, the waiting list, the `apart` set and the target's number.
- **Built-in assertions on every solve.** Where R0 absorbs nobody, every rule gives R0's place exactly and R0's verdict; RN joins nobody; under oracle people every yaw is known. No solve raised.

---

## 2. Arm 1: I06's crowd variants under oracle C (S1, K2)

### What was done

- I06's arm 3, unchanged (`lib.solve_c`): the oracle target, the reference context check, oracle people numbered as the pipeline numbers them, I01 R5's reference constraints, then `solver.solve`. On all 229 kept scenes (30 bases, 199 variants), both episodes, under each rule. Labels as I06 scored them: L1 is the package's success; L2 is L1 plus I06's check of a 0.3 m body gap and other groups' circles.
- Family A's confuser stands on the line's axis behind the last member: A1 1.9 m, 180° off the line's facing; A2 1.6 m, 135°; A3 1.3 m, 90°; A4 1.0 m, 60°; A5 0.8 m, 30°; A6 0.6 m, 0° (I06's FROZEN section 4).

### Main tables

Table 2.1. L1 success by family and level, per rule (seeds pooled; `tables.txt` Table 1.1). L2 equals L1 in every cell except E1, which is 18 of 21 under L2 for every rule, as in I06.

| Family, level | n | R0 | RF90 | RF60 | RN | Median place-to-target distance, R0 → RF90 |
|---|---|---|---|---|---|---|
| Bases | 30 | 30 | 30 | 30 | 30 | 3.05 → 3.05 m |
| A1 | 12 | 0 | **12** | 12 | 12 | 4.21 → 2.72 m |
| A2 | 12 | 0 | **12** | 12 | 12 | 4.19 → 2.57 m |
| A3 | 15 | 0 | **0** | 0 | 0 | 4.31 → 3.31 m |
| A4 | 15 | 0 | 0 | 0 | 0 | 4.06 → 4.06 m |
| A5 | 15 | 0 | 0 | 0 | 0 | 3.80 → 3.80 m |
| A6 | 15 | 0 | 0 | 0 | 0 | 3.60 → 3.60 m |
| B1 | 21 | 13 | **15** | 15 | 17 | 3.30 → 2.99 m |
| C1 | 20 | 17 | **19** | 20 | 20 | 2.72 → 2.72 m |
| D2 | 29 | 25 | **26** | 26 | 29 | 3.30 → 3.28 m |
| D5 | 8 | 8 | 8 | 8 | 8 | 1.02 → 1.02 m |
| E1 | 21 | 21 | 21 | 21 | 21 | 3.05 → 3.05 m |
| E2 | 16 | 12 | 12 | 12 | 12 | 3.15 → 3.15 m |

- **Paired drop against the scene's own base** (L1, levels pooled): A 100.0 under R0, 71.4 under RF90, RF60 and RN (n 84); B 38.1, then 28.6 (RF90, RF60) and 19.0 (RN) (n 21); C 15.0, then 5.0 (RF90) and 0.0 (RF60, RN) (n 20); D 10.8, then 8.1 (RF90, RF60) and 0.0 (RN) (n 37); E 10.8 under every rule (n 37).
- **By situation** (`tables.txt` Table 1.2): every group and audience scene succeeds under every rule (bases 9 and 6; variants 38 and 8). All changes are at lines: B1 4 of 12 under R0, then 6, 6 and 8 under RF90, RF60 and RN; C1 9 of 12, then 11, 12 and 12; D2 11 of 15, then 12, 12 and 15.
- A4's median (4.06 m) is the verifier's; `tables.txt` shows 4.05 m (section 6).

Table 2.2. The judged set J: successes per rule (`tables.txt` Table 1.3; the verifier's are identical).

| Rule | J under L1 (L2 the same) | A1 (12) | A2 (12) | A3 (15) | B1 (8) | C1 (3) | D2 (4) |
|---|---|---|---|---|---|---|---|
| R0 | 0/54 | 0 | 0 | 0 | 0 | 0 | 0 |
| **RF90** | **29/54 = 53.7%** | 12 | 12 | 0 | 2 | 2 | 1 |
| RF60 | 30/54 = 55.6% | 12 | 12 | 0 | 2 | 3 | 1 |
| RN | 35/54 = 64.8% | 12 | 12 | 0 | 4 | 3 | 4 |

- **Flips and breaks** (`tables.txt` Table 1.4). Every success in J is a flip (an R0 failure turned into a success). Breaks (an R0 success turned into a failure) are 0 for every rule, under L1 and L2: at the 30 bases, the 46 group and audience variants and every line variant outside J. This was expected: only R0's 99 scenes with a joiner can change, and all of them are R0 failures.
- RF60's one extra flip over RF90 is `i06_s1_C1_eval_07_waiting_line_2`, whose joiner has f · u +0.18. RN's six extra flips over RF90 are at B1 (2), C1 (that same scene) and D2 (3).
- (writer's arithmetic) RN removes every unlisted joiner, so it bounds what any gate on joining can fix in J: 35 of 54. RF90 reaches 29 of those 35. Without A1–A2, RF90 fixes 5 of 30 (A3 and B–D) and RN 11 of 30.

Table 2.3. Failure (i), a non-member in the line, firing in a variant and not in its base, over I06's 153 relevant variants (`tables.txt` Table 1.6; the verifier's are identical).

| Rule | Fires | A (84) | B (12) | C (12) | D (15) | E (30) |
|---|---|---|---|---|---|---|
| R0 | 99 (64.7%) | 84 | 8 | 3 | 4 | 0 |
| RF90 | 60 (39.2%) | 52 | 4 | 1 | 3 | 0 |
| RF60 | 42 (27.5%) | 35 | 4 | 0 | 3 | 0 |
| RN | 0 | 0 | 0 | 0 | 0 | 0 |

### What the gate sees

Table 2.4. The people R0 absorbed (e0), with the gate's f · u towards the current last member (`tables.txt` Table 1.7; the verifier's are identical). Every yaw is known under oracle people.

| Family, level | Absorbed | f · u: min / median / max | RF90 lets in | RF60 lets in |
|---|---|---|---|---|
| A1 | 12 | −1.000 / −0.997 / −0.988 | 0 | 0 |
| A2 | 12 | −0.783 / −0.690 / −0.594 | 0 | 0 |
| A3 | 15 | −0.161 / −0.029 / +0.175 | 7 | 0 |
| A4 | 15 | +0.348 / +0.485 / +0.627 | 15 | 5 |
| A5 | 15 | +0.805 / +0.857 / +0.941 | 15 | 15 |
| A6 | 15 | +0.982 / +0.997 / +1.000 | 15 | 15 |
| B1 | 13 | −0.995 / +0.668 / +0.988 | 7 | 7 |
| C1 | 5 | −0.924 / +0.178 / +0.922 | 3 | 2 |
| D2 | 4 | −0.587 / +0.666 / +0.949 | 3 | 3 |

- **A3 sits on RF90's boundary**, as the note predicted: the confuser faces 90° off the line, with up to ±0.15 rad of jitter. RF90 lets 7 of 15 in, so the jitter decides the gate there.

| Level | Confuser's distance from the place in turn | RN success |
|---|---|---|
| A1 | 0.86–0.94 m | 12/12 |
| A2 | 0.55–0.65 m | 12/12 |
| A3 | 0.26–0.35 m | 0/15 |
| A4 | 0.03–0.08 m | 0/15 |
| A5 | 0.15–0.26 m | 0/15 |
| A6 | 0.36–0.44 m | 0/15 |

- (inference) On these 84 scenes, success after the confuser is turned away needs it to stand at least 0.55 m from the place in turn; at 0.44 m or less every rule fails. A plausible mechanism, not checked: the solver keeps 1.0 m from every person, so a confuser that close pushes the stop out of the region. Either way, at A3–A6 no rule about who joins can succeed under L1.
- **At B–D, facing cannot separate the confusers.** Many B1 and D2 confusers face the line's front (f · u +0.59 to +0.99), as waiting people do. RF90 fixes 5 of the 15 B–D failures; RN fixes 11.

Table 2.5. The 15 B–D failures in J, scene by scene (`tables.txt` Table 1.8; the verifier's are identical). Joiners are given as number: f · u (towards the current last member in R0's loop); S success, F failure under L1.

| Scene | R0 joiners | RF90 | RF60 | RN |
|---|---|---|---|---|
| `i06_s0_B1_eval_08_waiting_line_4` | 6: −0.82, 7: +0.88 | none, S | none, S | none, S |
| `i06_s0_B1_eval_13_waiting_line_3` | 2: +0.67 | 2, F | 2, F | none, F |
| `i06_s1_B1_eval_06_waiting_line_1` | 3: +0.99, 4: −0.81 | 3, F | 3, F | none, S |
| `i06_s1_B1_eval_07_waiting_line_2` | 2: −0.94 | none, F | none, F | none, F |
| `i06_s1_B1_eval_08_waiting_line_4` | 2: +0.79, 1: +0.79 | 2 and 1, F | 2 and 1, F | none, F |
| `i06_s2_B1_eval_06_waiting_line_1` | 2: −0.98 | none, F | none, F | none, F |
| `i06_s2_B1_eval_07_waiting_line_2` | 4: −0.39, 5: +0.70 | none, S | none, S | none, S |
| `i06_s2_B1_eval_13_waiting_line_3` | 2: +0.82, 1: −1.00 | 2, F | 2, F | none, S |
| `i06_s0_C1_eval_13_waiting_line_3` | 2: −0.63, 1: +0.92 | none, S | none, S | none, S |
| `i06_s1_C1_eval_07_waiting_line_2` | 1: +0.18 | 1, F | none, S | none, S |
| `i06_s1_C1_eval_08_waiting_line_4` | 2: −0.92, 1: +0.78 | none, S | none, S | none, S |
| `i06_s1_D2_eval_06_waiting_line_1` | 3: +0.59 | 3, F | 3, F | none, S |
| `i06_s2_D2_eval_06_waiting_line_1` | 1: −0.59 | none, S | none, S | none, S |
| `i06_s2_D2_eval_08_waiting_line_4` | 6: +0.74 | 6, F | 6, F | none, S |
| `i06_s2_D2_eval_13_waiting_line_3` | 1: +0.95 | 1, F | 1, F | none, S |

- Where R0 took two people and the first faces away, turning the first away can also drop the second (for example `i06_s0_B1_eval_08`): with a different last member, the second no longer qualifies.
- Under RN, 4 of the 15 still fail, all at B1 (writer's tally from the table).

### Not judged

- A4, A5 and A6 are 0 of 15 each under every rule, RN included.
- E2's 4 R0 failures stay failures under every rule. Hidden floor is not about who joins.

---

## 3. Arm 1b: I06's noisy arm (secondary, not judged)

- **What was done.** The 168 kept line scenes (15 bases, 153 variants); I06's noise model per model; 20 draws per scene with I06's own seeds; R0 and RF90 on the same simulated people (D2: re-simulated in a second process, with all 10,080 people hashes equal to R0's). R0 matches I06's recorded draws in 10,080 of 10,080, so I06's seeding reproduces.

Table 3.1. Mean L1 over draws, then over scenes, R0 → RF90 (`tables.txt` section 2; the verifier's are identical).

| Family, level | n | Qwen 3.8 | GLM-5.3 | DeepSeek V4.1 |
|---|---|---|---|---|
| Bases | 15 | 0.940 → 0.940 | 0.783 → 0.783 | 0.770 → 0.770 |
| A1 | 12 | 0.233 → 0.896 | 0.362 → 0.754 | 0.371 → 0.804 |
| A2 | 12 | 0.033 → 0.854 | 0.133 → 0.613 | 0.163 → 0.717 |
| A3 | 15 | 0.027 → 0.083 | 0.117 → 0.193 | 0.080 → 0.143 |
| A4 | 15 | 0.007 → 0.010 | 0.047 → 0.077 | 0.023 → 0.023 |
| A5 | 15 | 0.003 → 0.003 | 0.017 → 0.027 | 0.020 → 0.023 |
| A6 | 15 | 0.000 → 0.000 | 0.037 → 0.037 | 0.023 → 0.023 |
| B1 | 12 | 0.342 → 0.463 | 0.346 → 0.467 | 0.254 → 0.367 |
| C1 | 12 | 0.771 → 0.879 | 0.667 → 0.758 | 0.675 → 0.775 |
| D2 | 15 | 0.623 → 0.733 | 0.550 → 0.627 | 0.553 → 0.630 |
| E1 | 15 | 0.937 → 0.937 | 0.733 → 0.737 | 0.777 → 0.777 |
| E2 | 15 | 0.620 → 0.623 | 0.490 → 0.493 | 0.490 → 0.500 |
| **All 168** | | **0.380 → 0.518** | **0.355 → 0.451** | **0.349 → 0.448** |
| Per-draw flips / breaks (of 3,360 draws) | | 463 / 0 | 320 / 0 | 334 / 0 |
| Share of draws with failure (i) | | 0.539 → 0.322 | 0.390 → 0.221 | 0.410 → 0.244 |

- Mean L2 over all 168 scenes: Qwen 0.374 → 0.510, GLM 0.322 → 0.415, DeepSeek 0.318 → 0.415 (verifier).
- (inference) Under each model's simulated noise, the gate keeps most of its oracle effect at A1–A2 and B–D and never breaks a draw. The noise is i.i.d. and fitted in-sample (I06 section 4), so this says nothing about real VLM yaw errors in crowds.

---

## 4. Arm 2: development records (S2, K1)

### What was done

- Every goal step (a record where the solver ran) of 11 cells, re-solved under each rule (FROZEN Table 5.1). Primary, old wording, I14's six cells: realistic (arms real, staffed and plain) for s19 Qwen 3.8 low (143 goal steps), s19 GLM-5.3 (141) and s20 DeepSeek V4.1 Flash (157), and watching (the audience goal steps of dev_seed0–2, 36 each) for the same three. Secondary: the s23 new-wording realistic runs of GLM (141), DeepSeek (151), Haiku 5.5 (146), Sonnet 5.5 low (143) and Gemini 3.8 low (135).
- **Settings.** (a) The record's own inputs (target, people, constraints) through the harness. (b) True people with the recorded constraints (I01's P). (c) The frozen I09 detector's people with the recorded constraints (A2's D), realistic cells only.
- Flips and breaks are counted against R0 of the same setting.

Table 4.1. Successes per cell and setting, and each rule's flips / breaks against R0 (`tables.txt` Table 3.1; the verifier's are identical). "Absorbed" counts the goal steps where R0 took someone into the line (in brackets: of them, R0 successes). Only setting (a) of the six primary cells is judged.

| Cell | Setting | n | Absorbed (R0 ok) | R0 | RF90 f / b | RF60 f / b | RN f / b |
|---|---|---|---|---|---|---|---|
| **Primary** | | | | | | | |
| Qwen realistic | (a) | 143 | 0 | 134 | 0 / 0 | 0 / 0 | 0 / 0 |
| | (b) | 143 | 0 | 136 | 0 / 0 | 0 / 0 | 0 / 0 |
| | (c) | 143 | 0 | 136 | 0 / 0 | 0 / 0 | 0 / 0 |
| GLM realistic | (a) | 141 | 0 | 96 | 0 / 0 | 0 / 0 | 0 / 0 |
| | (b) | 141 | 3 (3) | 112 | 0 / 0 | 0 / 0 | 0 / 0 |
| | (c) | 141 | 3 (3) | 112 | 0 / 0 | 0 / 0 | 0 / 0 |
| DeepSeek realistic | (a) | 157 | 3 (1) | 109 | **1 / 0** | 1 / 1 | 1 / 1 |
| | (b) | 157 | 8 (6) | 120 | 0 / 0 | 0 / 0 | 0 / 5 |
| | (c) | 157 | 8 (7) | 126 | 0 / 0 | 0 / 0 | 0 / 5 |
| Qwen watching | (a) | 36 | 0 | 32 | 0 / 0 | 0 / 0 | 0 / 0 |
| | (b) | 36 | 0 | 33 | 0 / 0 | 0 / 0 | 0 / 0 |
| GLM watching | (a) | 36 | 0 | 23 | 0 / 0 | 0 / 0 | 0 / 0 |
| | (b) | 36 | 0 | 29 | 0 / 0 | 0 / 0 | 0 / 0 |
| DeepSeek watching | (a) | 36 | 1 (1) | 23 | 0 / 0 | 0 / 0 | 0 / 0 |
| | (b) | 36 | 1 (1) | 31 | 0 / 0 | 0 / 0 | 0 / 0 |
| **Secondary, new wording** | | | | | | | |
| GLM | (a) | 141 | 0 | 86 | 0 / 0 | 0 / 0 | 0 / 0 |
| | (b) | 141 | 4 (4) | 107 | 0 / 0 | 0 / 0 | 0 / 0 |
| | (c) | 141 | 3 (3) | 106 | 0 / 0 | 0 / 0 | 0 / 0 |
| DeepSeek | (a) | 151 | 2 (1) | 106 | 0 / 0 | 0 / 0 | 0 / 0 |
| | (b) | 151 | 9 (5) | 116 | 0 / 0 | 0 / 0 | 0 / 1 |
| | (c) | 151 | 9 (6) | 119 | 0 / 0 | 0 / 0 | 0 / 2 |
| Haiku | (a) | 146 | 3 (3) | 134 | 0 / 0 | 0 / 0 | 0 / 0 |
| | (b) | 146 | 3 (3) | 135 | 0 / 0 | 0 / 0 | 0 / 0 |
| | (c) | 146 | 3 (3) | 134 | 0 / 0 | 0 / 0 | 0 / 0 |
| Sonnet | (a) | 143 | 0 | 134 | 0 / 0 | 0 / 0 | 0 / 0 |
| | (b) | 143 | 0 | 134 | 0 / 0 | 0 / 0 | 0 / 0 |
| | (c) | 143 | 0 | 134 | 0 / 0 | 0 / 0 | 0 / 0 |
| Gemini | (a) | 135 | 15 (0) | 109 | 0 / 0 | 0 / 0 | 0 / 0 |
| | (b) | 135 | 15 (0) | 109 | 0 / 0 | 0 / 0 | 0 / 0 |
| | (c) | 135 | 16 (0) | 109 | 0 / 0 | 0 / 0 | 0 / 0 |

- **S2's evidence is thin.** In setting (a), R0 absorbed someone in only 4 of the 549 primary goal steps (DeepSeek realistic 3, DeepSeek watching 1). Elsewhere the gate cannot act, by construction.

Table 4.2. The four primary goal steps with an absorption in setting (a) (`tables.txt` Table 3.4b; the verifier's rows).

| Record (stage, arm, scene, episode, trial) | Joiner: f · u with the model's yaw | R0 | RF90 | RF60 | RN |
|---|---|---|---|---|---|
| s20 real, `real_eval_09_waiting_line_6`, e0, 1 | #1: −0.338 | S | S (let #2 in instead; place moved 0.54 m) | **F (break)** | **F (break)** |
| s20 real, `real_eval_13_waiting_line_3`, e1, 2 | #1: −0.389 | F | **S (flip)** | S (flip) | S (flip) |
| s20 staffed, `staffed_eval_13_waiting_line_3`, e1, 1 | #1: +0.218 | F | F | F | F |
| s20 dev_seed2, `eval_15_audience_5`, e0, 2 | #3: +0.986 | S | S | S | S |

### What RN costs, and who is absorbed on real records

Table 4.3. RN's breaks, that is how often absorption is needed (`tables.txt` Table 3.5).

| Cell | (a) | (b) | (c) |
|---|---|---|---|
| DeepSeek realistic, old wording | 1 break, 1 flip | 5 breaks | 5 breaks |
| DeepSeek, new wording | 0 | 1 break | 2 breaks |
| Every other cell | 0 | 0 | 0 |

- The joiners in RN's breaks in (b) and (c) have f · u 0.93–1.00: real waiting people the waiting list missed, standing and facing as members do. RF90 keeps every one of them. They come from three layouts: `eval_08_waiting_line_4` (staffed arm), `eval_09_waiting_line_6` and `eval_13_waiting_line_3` (real arm).
- **Who is absorbed** (`tables.txt` Tables 3.4 and 3.4b). In (b) and (c), every person R0 absorbed in any cell has a known yaw and f · u ≥ 0.629 (verifier; the compute agent wrote "nearly every"). In (a), with the models' own yaws: DeepSeek old 3 (f · u −0.389, −0.338, +0.218), DeepSeek new 2 (+0.454, +0.985), Haiku 3 (+0.466 to +0.999), Gemini 15 (+0.968 to +0.994) and DeepSeek watching 1 (+0.986); none in the Qwen, GLM or Sonnet cells.
- **Gemini's 15 absorptions in (a)** are all R0 failures and stay failures under every rule, RN included, so removing the joiner does not rescue them (inference: those records fail for another reason).
- All absorptions are at lines, except the one audience goal step (DeepSeek watching) and one or two conversation-group goal steps in each DeepSeek cell in (b) and (c).

---

## 5. Criteria and verdict (verbatim, FROZEN section 6; readings there)

| Criterion | Value | Outcome |
|---|---|---|
| S1 "under oracle C and L1, RF90 turns at least 50% of R0's line failures in families A1-A3, B, C and D into successes (R0: A1-A3 39 failures, B 8, C 3, D 4 at lines; recount from the rows), and adds no failure in any I06 base or group/audience variant." | 29 of 54 in J (27 needed). Breaks: 30 bases 0; 46 group and audience variants 0 (line variants outside J also 0, reported) | **Holds** (by 2 scenes) |
| S2 "in setting (a), RF90 has at most 1 break in each of the six primary cells, and its net (flips - breaks) is >= 0 in every primary cell." | Breaks 0 in every cell. Nets 0, 0, +1, 0, 0, 0 (Qwen, GLM, DeepSeek realistic; Qwen, GLM, DeepSeek watching) | **Holds** (4 goal steps could change) |
| K1 "RF90 has 3 or more breaks in any primary cell in setting (a), or a negative pooled net in (a)." | Most breaks in a cell: 0; pooled net +1 | Does not fire |
| K2 "RF90 fixes fewer than 25% of the judged I06 failures." | 29 of 54; fires at 13 or fewer | Does not fire |
| R0 reproduction in arm 1 and setting (a) (otherwise no verdict) | Full (229 of 229 scenes; 1,265 of 1,265 goal steps) | Met |

- Not judged, as the design says: RF60 and RN, settings (b) and (c), the new-wording runs and arm 1b (sections 2–4).

---

## 6. Where the compute agent and the verifier differ

Every criterion value, every R0 reproduction and every row agree. The verifier found 0 differences against `rows_crowd.jsonl`, `rows_noisy_*.jsonl` and `rows_records_*.jsonl`. What differs is bookkeeping, one rounded median and one explanation.

Settled by the verifier (its value is used above):
1. **A write outside the probe folder.** The compute agent reported that nothing was written outside `results/still-scenes/crowds/facing-gate-repair/`. Its records pass, which imports A2's `replay_d.py`, wrote the bytecode cache `phase1c/i09/a2/__pycache__/replay_d.cpython-312.pyc` at 06:47:42, creating that `__pycache__` folder in an earlier phase folder; the writer confirmed the file time (06:47:42.05). No content file changed. The verifier ran Python with `-B` and wrote nothing outside `verify/`.
2. **The time in DEVIATIONS.md.** D1 and D2 are dated "06:45 +08:00, before any pass ran". The file was created at 06:43:25 and hashed at 06:43:26 (`logs/deviations_hash_history.txt`), and the first R0 pass started at 06:43:40. So both entries predate every pass, as stated, but the stated time is wrong, and it would fall after the first pass started.
4. **Why A3 fails under every rule.** Compute agent: the confuser's own 1.0 m space covers the place in turn. Verifier: that also holds at A1 and A2, where RN succeeds 24 of 24; at A3 the confuser stands 0.26–0.35 m from the place in turn, against 0.55–0.94 m at A1–A2, and that closeness is what separates it. The counts are not in dispute. Section 2 uses the verifier's reading and extends its figures to A4–A6 (writer's tally).

Also settled, a wording point: the compute agent wrote that "nearly every" person absorbed in (b) and (c) faces the line's front; it is every one, with f · u ≥ 0.629 (verifier).

Not a disagreement:
- **S2's 0 breaks rested once on a second candidate** (the verifier's nuance; section 4, Table 4.2).

- D1 (file places) and D2 (arm 1b re-simulated with a hash check; all 10,080 people hashes equal). The verifier followed FROZEN section 4 literally (one simulation per draw, both rules in one process); the two routes agree draw by draw.
- In arm 1 the verifier computed failure (i)'s detector with its own one-line formula (equal to I06's on all 229 scenes), and it wrote its own P-people construction. It reused I06's `lib.py` and `noise.py`, I01's `repairs.py` and A2's `replay_d.py` unchanged, as FROZEN specifies.
- Not re-derived by the verifier: the per-cell situation splits (`tables.txt` Table 3.3), which follow from the rows it matched.

---

## 7. What it means (inference)

**For failure (i).**
- A facing gate fixes the clear cases: a confuser facing away from the line (A1–A2, 24 of 24) and some B–D cases. It breaks nothing on I06's crowds, and nothing on the development records in any setting.
- Never absorbing is not an option. On real records, with true or detector people, RN loses up to 5 successes per DeepSeek cell, because the waiting list misses real waiting people who stand and face as members do (f · u 0.93–1.00). The facing gate keeps every one of them. So the gate sits between R0 and RN: it keeps the absorptions real lines need and drops the ones that face away.

**How strong the evidence is.**
- S1 passed by 2 scenes, and 24 of its 29 fixes are A1–A2, which the gate turns away by construction. Of the J failures that removing every joiner fixes (35), RF90 fixes 29 (writer's arithmetic).
- S2 passed on 4 goal steps. Absorption is rare on real records in setting (a) (4 of 549 primary goal steps), so S2 shows that the gate is harmless on these records more than that it helps. The one record where the gate turned a real joiner away used the model's wrong yaw, and a second person saved it.
- Everything is development data, with oracle people or in-sample i.i.d. noise. No crowd test with real VLM perception has been run.

**For the pipeline.**
- It needs a facing for every detected person, which ties it to the heading question in REPORT section 7. With a model's own yaw it can turn real joiners away (one DeepSeek record here). With I09's detector yaw (setting (c)) every person absorbed on the records faced the line's front, and that detector is within 45° for 0.969 of non-profile people on unseen layouts (`results/still-scenes/people-finder/early-detector/phase2/README.md` section 1). So the gate and I09's people source fit together.
- A test on records not used to design the rule should come first. Because the gate only re-solves recorded inputs, it can be replayed on any later run's records at zero extra calls, for example on the VLM crowd run's (Programme B step 8).
- For I19 (geometry computes the members): a chain consistent with facing for `come_after` should be compared with RF90, not only with the current `_turn`, on crowded lines.

1. Accept the verdict, method candidate, given the four differences in section 6, none of which changes a criterion value?
2. Adopt RF90 in `sgi.solver._turn` now, or first replay it on records not used to design it (for example the VLM crowd run's, at zero extra calls)?
3. Which facing should the gate use: the model's own, as recorded, or I09's detector's, if I09 becomes the people source? One record here shows a model's wrong yaw turning a real joiner away.
5. Housekeeping: delete the stray bytecode cache `phase1c/i09/a2/__pycache__/` (harmless, not a content change), and append a dated correction of the stated time to `DEVIATIONS.md` (its entries are never edited)?

---

## 8. Proposed REPORT edits (not applied)

Each item gives the exact old text from `REPORT.md` and the proposed new text. Each old text occurs exactly once in the current `REPORT.md` (checked with Python's `str.count`), and applying items 1–15 in order works (checked on a copy in memory: each old text still occurs exactly once when its turn comes). New text is tagged [phase 2]. That tag is already defined in "How to read", so item 1 adds this page to the definition's list of pages and item 2 adds the probe's inputs to the same definition; neither redefines it, and neither needs its own tag. Item 4 extends a sentence that already starts with [phase 2]. Item 14 appends the revision-note bullet after the existing "- **Phase 2, C5 flow-matching pilot" bullet. Item 15 is optional. If another page's edit changes one of these anchors first, re-check its count before applying.

### 1. "How to read": add the repair probe to [phase 2]'s list of pages

Old:
```text
I06's first crowd probe (`results/still-scenes/crowds/README.md`) and C5's flow-matching pilot (`results/still-scenes/learned-models/flow-model-pilot/README.md`). Each step froze
```
New:
```text
I06's first crowd probe (`results/still-scenes/crowds/README.md`), I06's repair probe, a facing gate on joining a line (`results/still-scenes/crowds/facing-gate-repair/README.md`), and C5's flow-matching pilot (`results/still-scenes/learned-models/flow-model-pilot/README.md`). Each step froze
```

### 2. "How to read": the repair probe's inputs, in the same definition

Old:
New:

### 3. Opening lines: the list of phase-2 steps

Old:
```text
I06's first crowd probe, C5's exploratory pilot)
```
New:
```text
I06's first crowd probe and its repair probe [phase 2], C5's exploratory pilot)
```

### 4. Opening lines: what phase 2 answers

Old:
```text
I06's first crowd probe for families A–E (neither) and C5's exploratory pilot (kill);
```
New:

### 5. Finding 2: the solver's silent crowd failures

Old:
```text
post hoc because the frozen families kept added people at least 1.2 m from the members and outside the structure's space (`results/still-scenes/crowds/README.md` section 3).
```
New:
```text
post hoc because the frozen families kept added people at least 1.2 m from the members and outside the structure's space (`results/still-scenes/crowds/README.md` section 3). [phase 2] A facing gate on joining (an unlisted person joins only when facing into the line's front half-plane; research code, zero calls) then turned 29 of the 54 judged absorption failures into successes and broke nothing on the crowds or the development records. What it leaves is unlisted people facing the line's front near its end, and confusers on or near the place in turn, which no join rule fixes (`results/still-scenes/crowds/facing-gate-repair/README.md`).
```

### 6. Section 1.2, phase-2 step 6

Old:
```text
(the other 4 post hoc from hidden floor; `results/still-scenes/crowds/README.md`).
```
New:
```text
(the other 4 post hoc from hidden floor; `results/still-scenes/crowds/README.md`). [phase 2] A repair probe the same day, a facing gate on joining (zero calls), is a method candidate: it fixes 29 of 54 judged absorption failures and breaks nothing on the crowds or the development records (`results/still-scenes/crowds/facing-gate-repair/README.md`).
```

### 7. Section 1.3: a new decision

Old:
```text
- Who could write rules for new held-out situations, and the real deadline.
```
New:
```text
- Who could write rules for new held-out situations, and the real deadline.
- [phase 2] Whether to adopt I06's facing gate on joining a line in the solver (a method candidate, `results/still-scenes/crowds/facing-gate-repair/README.md`), which facing it should use, and whether to replay it first on records not used to design it.
```

### 8. I06: the repair probe's result, after the first probe's result

Old:
```text
The zero-call curves do not answer RQ2; that still needs the VLM crowd run.
```
New:

### 9. I19: the payoff on crowded lines

Old:
```text
- **Payoff.** −1/1.5/5 pp on the existing sets (guess). On crowded lines, +5–20 pp (guess).
```
New:
```text
- **Payoff.** −1/1.5/5 pp on the existing sets (guess). On crowded lines, +5–20 pp (guess). [phase 2] On I06's crowd variants (oracle people), a one-condition facing gate on the current `_turn` already turns 29 of 54 judged absorption failures into successes with no break (`results/still-scenes/crowds/facing-gate-repair/README.md`). (inference) So I19's chain consistent with facing should be compared with that gate, not only with `_turn`; what the gate leaves is unlisted people facing the line's front near its end.
```

### 10. Section 6, the decision-gate row

Old:
New:

### 11. Programme B: experiment 2

Old:
```text
groups and audiences held in every variant built (`results/still-scenes/crowds/README.md`).
```
New:
```text
groups and audiences held in every variant built (`results/still-scenes/crowds/README.md`). [phase 2] A repair probe the same day, a facing gate on joining, is a method candidate: 29 of 54 judged absorption failures fixed and no break on the crowds or the development records, though S2 rested on the 4 goal steps where absorption happened (`results/still-scenes/crowds/facing-gate-repair/README.md`).
```

### 12. Section 7: crowd labels and the cue threshold

Old:
```text
the probe's criteria come out the same without A5–A6.
```
New:
```text
the probe's criteria come out the same without A5–A6. [phase 2] I06's repair probe shows that no join rule decides A3–A6: they fail under every rule tested, nobody joining included, because the confuser stands 0.03–0.44 m from the place in turn, against 0.55–0.94 m at A1–A2, where the rules succeed (`results/still-scenes/crowds/facing-gate-repair/README.md` section 2). (inference) So those scenes turn on the cue threshold, that is on what L1 counts when someone stands at or near the place in turn.
```

### 13. Section 7: a new item after the heading question

Old:
```text
where the models score 0.547–0.906 on the same pairs (`results/still-scenes/overview/phase-notes/phase1c.md` section 1).
```
New:
```text
where the models score 0.547–0.906 on the same pairs (`results/still-scenes/overview/phase-notes/phase1c.md` section 1).
- [phase 2] Whether the solver's joining rule should get I06's facing gate (an unlisted person joins a line only when facing into its front half-plane; method candidate, `results/still-scenes/crowds/facing-gate-repair/README.md`). It needs a facing for every detected person, so it ties into the heading question above: with a model's own yaw, one DeepSeek record had a real joiner turned away (a second person saved the success); with true or I09-detector yaw, every person absorbed on the records faced the line's front. Adopting it changes `src/sgi/solver.py` and its tests. (inference) A replay on records not used to design it should come first.
```

### 14. Revision note, after the C5 flow-matching pilot bullet

Old:
```text
"How to read" defines [phase 2], and C5 and section 6 (step 8) carry the result, and so do the opening lines and section 4.6.
```
New:

### 15. Section 5, "Which to start with": Programme B's condition (optional)

Old:
New:
