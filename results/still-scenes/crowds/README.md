# Crowded scenes: how many extra people break goal inference (still scenes)

**Question.** When bystanders, a second group or a person who looks like a line member are added around a target group, how often does the zero-call solver, given the true people and the reference constraints, still choose a correct place?

**Result.** On 229 crowd scenes (199 variants and 30 base layouts, 3 seeds) the full oracle succeeds on 229 of 229. With the reference constraints (oracle C) the success rate falls by 100 points when a confuser stands behind a line (family A, from 1.3 m), by 38.1 (n 21) with a nearby cluster of three (B), 15.0 (n 20) with a second group (C), and 13.8 (n 29) with bystanders (D2); an added person in the camera line (E) does not hurt on L1. Only one of the pre-registered failure modes fired (bystanders absorbed behind a line, 99 of 153 relevant variants). A noise model fitted on three models' recorded perception errors reproduces their recorded success within +0.3 (Qwen), +5.4 (GLM) and +6.2 (DeepSeek) points, and does not show crowds amplifying noise. A C5 gate-1 recount on the widened generator gives 92 configurations against 30.

**Files.**
- `FROZEN.md`, `DEVIATIONS.md`: the protocol written before scoring and the one change made after it.
- `precheck/`: format and depth checks run before the freeze.
- `builder/`: the crowd generator (`crowd.py`), base reproduction, variant generation, label checks, byte-identity proof, self-test, contact sheets.
- `build/`: manifests, label counts, per-seed base and variant tables, `byte_identity.txt`, `self_test.txt`, `readings.txt`.
- `scenes/seed0..2/`: the 229 kept scenes (`scene.json` only; the rendered `rgb.png` and `depth.npy` are not in the repository).
- `score/`: arms, noise model, validation, noisy arm, tables (`tables.txt`, `summary.json`), C5 recount, and the row files `rows_*.jsonl`.
- `verify_i06_build/`, `verify_i06_score/`: independent re-implementations and their outputs in `out/`.
- `facing-gate-repair/`: a follow-up experiment (own README).

**Reproduce.** Run from the repository root with `PYTHONPATH=src`.
- Scoring: `bash results/still-scenes/crowds/score/run_arms.sh`, `run_validate.sh MODEL...`, `run_noisy.sh MODEL`, then `python results/still-scenes/crowds/score/tables.py`.
- Build: `python results/still-scenes/crowds/builder/reproduce_bases.py SEED`, `generate_variants.py SEED`, `label_checks.py rows SEED`, then `label_checks.py summary`.
- The step-0 harness is in this repository (`results/still-scenes/analysis-scripts/harness/`); the development scenes it reads come from `data/dev_scenes/` (release dev-scenes-v1). The rendered images of the crowd variants (`rgb.png`, `depth.npy` in `scenes/`) are not in the repository: regenerate them with `builder/reproduce_bases.py` and `builder/generate_variants.py`, which takes more than 10 minutes per seed (the script resumes from the manifest if stopped).
- The scoring and re-check scripts check the sha256 of the original `FROZEN.md` and `src/sgi/solver.py`; the files in this repository no longer match those pinned values, so they stop at the assertion (the pinned values are left as they are).
- `builder/reproduce_bases.py SEED` needs only the package generator. Run with its output redirected to a scratch folder, it reproduces `build/bases_seed{0,1,2}.jsonl` and every base `scene.json` (and the recorded file hashes) exactly.
- Run here unchanged: `precheck/depth_check.py`, `precheck/format_check.py`, `builder/byte_identity.py compare` (after regenerating the scene images), `builder/base_check.py` (differs from the stored text only in the printed scene folders), `verify_i06_build/v_summary.py`, `v_selftest.py`, `v_compare_bases.py` and `builder/self_test.py` (the last two need the regenerated base layouts).
- What does run here: `python results/still-scenes/crowds/verify_i06_score/v_compare_noise.py qwen glm dsv` (same text as `verify_i06_score/out/compare_validation_lead.txt`) and `v_boot30.py`.

**Data.** `data/dev_scenes/` (release dev-scenes-v1) for the development scenes; `results/logs/research-data.tar.gz` (row and work files beside the notes) and `results/logs/still-runs.tar.gz` (the `runs/` folders that the noise fit reads). The seed folders and rendered images are not part of either.

## Notes

# Phase 2, step 4: I06 first probe, crowd variants with zero-call solver curves (2026-10-18)

Plan step 4 (REPORT section 6; REPORT I06, "First probe"): crowd variants of the development layouts of seeds 0–2, scored with zero calls. The solver is run with oracle people and reference constraints under two labels (L1, L2), then with people simulated from a noise model fitted on development runs. The page also recounts C5's gate 1 on the widened generator. Its outputs are solver-degradation curves, not RQ2 curves: R and E were not run, and C with real VLM perception needs the VLM crowd run (Programme B step 8).

How it was made:
- **After the freeze.** `DEVIATIONS.md` (sha256 `a54a7d2e…e70c`) holds one entry, D1, which moves output files only. It was written at 01:54:04, during generation and before any scoring. `build/readings.txt` (01:53:45) lists the builder's readings of points the note leaves open (draw order, D's band sampling, the bisection root, where pre-checks c4–c7 are measured from); none changes a rule.
- **This page.** The writer ran no arm, fit or simulation. A few counts were tallied from the recorded rows (`score/rows_oracle.jsonl`) to settle a discrepancy or split a table; each is marked "writer's tally". Differences computed from numbers in `score/tables.txt` are marked "writer's arithmetic".
- **Order** (file times, 18 Oct): FROZEN.md 01:34:19 → first base scene 01:45:21 → byte-identity proof 01:48:02 → `base_check.txt` 01:49:55 and `self_test.txt` 01:49:57 → `builder/crowd.py` last saved 01:51:41 → first variant 01:51:59 → `readings.txt` 01:53:45, `DEVIATIONS.md` 01:54:04 → label scoring 02:06:25 → first score file (`score/work/arms_seed0.jsonl`) 03:26:48 → tables and figures 05:21:40. Every hash file verifies.
- **Inputs.** The 30 base layouts were reproduced in memory with the package's own generator and streams: rows eval_06–13 from `default_rng(seed)` (eval_00–05 drawn only to advance the stream), and eval_14–15 from `default_rng([seed, 1])`, stopping before the group-photo row eval_16. Every crowd addition draws from a new stream `[seed, 2, family, row, level]`, so the existing scenes stay byte-identical (proved, section 1). Noise was fitted on the old-wording C runs s19 `qwen38_low`, s19 `glm53` and s20 `dsv41` (arms real, staffed, plain, ph, dev_seed0–2), read through the step-0 harness.

Paths below are relative to this folder (`results/still-scenes/crowds/` in the research folder).
- Protocol: `FROZEN.md`, `DEVIATIONS.md` (each hashed); `precheck/` (`format_check.py`, `depth_check.py`, logs).
- `builder/`: `crowd.py`, `reproduce_bases.py`, `base_check.py` and `base_check.txt`, `self_test.py`, `byte_identity.py`, `generate_variants.py`, `label_checks.py`, `contact_sheets.py`, `run_guarded.sh`.
- `build/`: `manifest.json` and `manifest.jsonl`, `labels.txt`, `labels_rows_seed{0,1,2}.jsonl`, `counts_by_row.txt`, `readings.txt`, `self_test.txt`, `byte_identity.txt` (and `byte_identity/`), `base_tables_seed*.json`, `base_layouts/`, `contact_A.png`–`contact_E.png`, `logs/`.
- `scenes/seed{0,1,2}/`: the 229 kept scenes (199 variants, 30 bases).
- `score/`: scripts (`lib.py`, `arms.py`, `noise.py`, `validate.py`, `noisy.py`, `tables.py`, `c5_recount.py`, `run_*.sh`, `explore_runs.py`); results (`tables.txt`, `summary.json`, `rows_oracle.jsonl`, `rows_validation.jsonl`, `rows_noisy_{qwen,glm,dsv}.jsonl`, `noise_model.json`, `noise/`, `c5_recount.txt`, `curves_oracle.png`, `curves_noisy.png`, `curves_by_situation.png`); `work/` and `logs/`.
- Verifiers: `verify_i06_build/`, `verify_i06_score/`.

## 0. Verdicts at a glance

| Analysis | Pre-registered outcome | One-line reading (inference) |
|---|---|---|
| Build: criterion (d), K3, label checks | **Met.** 199 of 270 planned variants kept (73.7%); full oracle 229 of 229 under L1; every label check 229 of 229; K3 does not fire (26.3% rejected) | The builder and the labels hold. Which family levels exist depends on the situation (D5 only at groups; audiences mostly rejected). |
| Oracle curves: criterion (a) | **Met.** Family drops A 100.0 (n 84), B 38.1 (n 21), C 15.0 (n 20); D 10.8 and E 10.8 (n 37 each) | Oracle C fails only at lines; every group and audience variant succeeds (46 of 46). |
| Documented failures: criterion (b) | **Not met.** Only (i), bystanders absorbed behind a line, fires: 99 of 153 relevant variants (64.7%). (ii), (iii) and (iv) fire 0 times | The frozen families kept non-members out of the structure's space, so (ii)–(iv) had little chance to fire. |
| L1/L2: criterion (c), K2 | **Met.** Oracle C disagrees in 3 of 199 (1.5%); L2-empty 0 of 199; K2 does not fire | For the solver the agreement is built in. L1's own reference point (the full oracle's stop) fails L2 in 85 of 199 (42.7%). |
| Noise model: K4 | **Valid for all three models.** Simulated minus recorded C: Qwen +0.3, GLM +5.4, DeepSeek +6.2 points (n 408 each); K4 does not fire | Against people noise alone (R5) the gaps are −1.2, −2.5 and −4.5. The score verifier's own validation did not finish. |
| Noisy separation: criterion (e) | **Not met.** Top level (n 80): Qwen +2.2, GLM +8.0, DeepSeek +8.9 points | In this i.i.d. model crowds do not amplify noise: GLM and DeepSeek separate by 17–18 points on the bases already. |
| K1 | Does not fire (lowest oracle-C cell is 0%) | |
| **I06 first probe overall** | **Neither.** (a), (c) and (d) met; (b) and (e) not met; no kill line fires. the project decides | One documented failure confirmed at a high rate in generated crowds, plus a post hoc hidden-floor mechanism; no failure inside the region was tested. |
| C5 gate-1 recount | **Met by the letter.** 92 configurations (63 and 49 under the stricter readings), against 30 | The widening adds diversity within the existing situations only. |

Shared cautions:
- **Labels are the project's own rules.** L1 is the members-only reference rule; L2 is this probe's stricter variant. Neither is social ground truth. The reference constraints of the oracle-C arm read the situation label and the true group, so that arm is circular (REPORT section 8).
- **Unit and n.** One scene is one episode (both episodes give the same result under every arm, asserted 229 of 229), so n is scenes. The layouts are 10 base structures × 3 seeds = 30; bootstrap intervals resample those 30 layouts and are descriptive only.
- **Family and situation are mixed.** D5 exists only at conversation groups, B and C have no variant at the line of 6 or the audience of 5, A1–A2 none at the line of 6, and 8 of 36 planned audience variants were kept.
- **Scenes.** Placeholder renders, with each variant keeping its base room. Family F was deferred (section 1).

---

## 1. The build (`builder/`, `build/`, `scenes/`)

### What was built

- **Base layouts** (FROZEN section 2): 10 development rows per seed, 30 in all. Lines `eval_06` (1 person), `eval_07` (2), `eval_08` (4), `eval_09` (6), `eval_13` (3); conversation groups `eval_10` (2, with 2 bystanders), `eval_11` (3, with a second group), `eval_12` (4, with 1 bystander and a second group); audiences `eval_14` (3) and `eval_15` (5).
- **Variants.** Each is the base layout plus added people, all non-members (role none, never in the target's group), made with the package's own `_person` (±0.05 m and ±0.15 rad jitter) and `_shirts`. The room, furnishings, robot and camera stay as in the base, so the added people are the only change. Each candidate passes pre-checks c1–c8 (room clearance, body spacing, distance from the robot, in the picture, clear sight lines, not before the target, outside the structure's space; FROZEN section 3), then the package's own scene path (`build_scene`, `_rejection`, `_with_regions`), with at most 5 redraws.
- **The families** (FROZEN section 4). "Members" are the people of the target's group.

| Family | What is added | Where | Levels |
|---|---|---|---|
| A | A confuser on the line's axis behind the last member | Lines only | A1 1.9 m, 180 deg off the line's facing; A2 1.6 m, 135; A3 1.3 m, 90; A4 1.0 m, 60; A5 0.8 m, 30; A6 0.6 m, 0 (faces like a member). Every level is inside the solver's 2.0 m absorption distance |
| B | 3 people on an equilateral triangle with 1.2 m sides, random yaws | All rows | B1: nearest cluster person exactly 1.2 m from the nearest member |
| C | A second group of 3 (the package's conversation geometry), new group id | All rows | C1: nearest pair exactly 1.5 m apart |
| D | Bystanders, random yaw, 1.2–2.5 m from the nearest member, at least 1.2 m from everyone, off the target's sight line | All rows | D2 (2 people), D5 (5 people) |
| E | One person on the ground segment from the camera to the representative point of the members' region | All rows | E1: 1.5 m from that point; E2: 2.0 m from the camera |

- **Labels.** L1 = the package's regions, derived from the members only; every kept variant's polygons must equal its base's. L2 = L1 success plus a body gap of at least 0.3 m to every non-member and a stop outside every other group's circle, scored as a pose-level check (FROZEN section 5).

### Counts

Table 1.1. Kept of planned, by family and level (seeds 0–2 pooled; `build/labels.txt`).

| Family, level | Planned | Kept | Rejected (reason) |
|---|---|---|---|
| A1 | 15 | 12 | 3 (visibility check 8 on all 6 attempts; the line of 6) |
| A2 | 15 | 12 | 3 (the same) |
| A3–A6 | 15 each | 15 each | 0 |
| B1 | 30 | 21 | 9 (no spot) |
| C1 | 30 | 20 | 10 (no spot) |
| D2 | 30 | 29 | 1 (no spot) |
| D5 | 30 | 8 | 22 (no spot) |
| E1 | 30 | 21 | 9 (no spot) |
| E2 | 30 | 16 | 14 (no spot) |
| **All** | **270** | **199 (73.7%)** | **71: 65 no spot, 6 visibility check (36 attempts)** |

- **First-attempt pass rate = pass rate** (199 of 270). No variant was accepted after a redraw; all 36 rejected attempts were check 8 on A1/A2 at the line of 6, a fixed position that fails every time.
- **Where the rejections fall** (`build/counts_by_row.txt`, 3 seeds per row): D5 kept 0 of 15 at lines, 0 of 6 at audiences and 8 of 9 at groups. E kept 0 of 12 at audiences, and E2 1 of 9 at groups (every E2 no-spot failed c6, the target sight line, on 200 of 200 draws). B1 and C1 kept 0 of 3 at the line of 6 and at the audience of 5.
- **By situation** (writer's tally): lines 153 of 180 planned kept, groups 38 of 54, audiences 8 of 36.
- **Added people** (after jitter): A follows its gaps (A1 1.86–1.94 m down to A6 0.56–0.64 m from the last member); B1 at least 1.16 m, C1 at least 1.45 m, D 1.21–2.52 m and E1 1.17–1.58 m from the nearest member. People per kept variant: 2–12 (the scene limit is 40).

### Label checks

All on the 229 kept scenes (199 variants + 30 bases), from `build/labels.txt` and the per-seed rows; the build verifier reproduced each.

| Check | Result |
|---|---|
| Base layouts reproduced, no layout rejected on the way | 30 of 30 |
| Bases equal to the whitelisted scenes (people, targets, camera, robot, regions, representative points, rgb and depth arrays) | 14 of 14 checkable, maximum difference 0 (seed 0 eval_06–13 against `data/placeholder/eval`; seeds 0–2 eval_14–15 against `keep/dev_new/seedN`). 16 (seeds 1–2, eval_06–13) have no copy on disk |
| Bases stay byte-identical with the builder loaded | Pass: 10 scenes written by a package-only process and by a builder process that built variants in between are sha256-equal file for file; harness fingerprints match the whitelisted folders |
| Pre-checks accept the people the package placed (eval_10–12) | 21 of 21 |
| Scene check passes (no error; no warning but 9) | 229 of 229 |
| Full oracle succeeds on both episodes under L1 (0 model calls) | 229 of 229 |
| Every acceptable region non-empty and reachable | 229 of 229 |
| Every added person a non-member | 229 of 229 |
| No held-out situation, row or scene | 229 of 229 |
| L1 invariance (polygons and headings equal the base's) | 229 of 229 (199 variants) |
| L2-empty | 0 of 199 variants, 0 of 30 bases |
| Representative point moved from the base's (allowed) | 102 of 199, at most 1.645 m |
| E: visible fraction and pixel share recorded | 37 of 37 kept E variants |

- **E hides almost nothing.** Every relevant member kept all its visible samples in 37 of 37 kept E variants, and pixel share fell in one case only (seed 1, E2, `eval_09`, person p5: 1.00 to 0.73). So E tests occupied or hidden floor near the stopping point, not occlusion of members.
- Many members show 8 of 9 visibility samples in the bases too (shorter figures' heads fall below the 1.7 m sample); the additions do not cause this.
- **Stream replay** (build verifier): all 270 planned variants were replayed from their streams with the verifier's own pre-checks. Every attempt's outcome matches the manifest, and the 199 kept scenes re-render byte for byte.

### Contact sheets

---

## 2. Oracle curves under L1 and L2 (`score/`)

### What was done

- **Arms** (FROZEN section 6), on every kept variant and base:
  1. Full oracle: the package's `run_episode` with oracle grounding, the context check always yes and the oracle goal.
  2. Solver alone: `run_episode` with oracle people and no constraints.
  3. **Oracle C (primary):** the oracle target, the reference context check, the visible true people numbered as the pipeline numbers them, I01 R5's reference constraints (lines: `come_after(members)`; groups: `keep_out_of_shared_space`, `visible_to` and `face`; audiences: `keep_out_of_view` and `face` towards the target), then `solver.solve`.
  4. Noisy C: arm 3 with simulated people (sections 4–5).
- **Built-in checks.** Both episodes agree under every arm: 229 of 229. Arm 3 with an empty constraint list lands on arm 2's place in 229 of 229 scenes (458 episode pairs), at a largest distance of 0.0 m. Arms 1–3 made 0 model calls. The score verifier recomputed arms 1–3 with its own code: 0 differences in verdicts, places, detector values and cell counts.
- **Drop** = mean over a level's kept variants of [success of its base − success of the variant] × 100, so each variant is paired with its own base and a rejection removes the pair.
- In 24 scenes (`eval_11` of every seed and `eval_12` of seed 2, bases and their variants) the reference context check finds that people do not matter, so arm 3 returns the standard goal there.

### Main tables

Table 2.1. Success by family and level (seeds pooled). Level 0 = the bases the family applies to.

| Family, level | n | Oracle C, L1 | Oracle C, L2 | Paired drop, L1 [95% CI] | Solver alone, L1 | Full oracle, L1 / L2 | Median distance, oracle-C place to target |
|---|---|---|---|---|---|---|---|
| A, bases (lines) | 15 | 15/15 | 15/15 | — | 0/15 | 15/15 / 15/15 | 3.05 m |
| A1 | 12 | 0/12 | 0/12 | 100 [100, 100] | 0/12 | 12/12 / 12/12 | 4.21 m |
| A2 | 12 | 0/12 | 0/12 | 100 [100, 100] | 0/12 | 12/12 / **0/12** | 4.19 m |
| A3 | 15 | 0/15 | 0/15 | 100 [100, 100] | 0/15 | 15/15 / **0/15** | 4.31 m |
| A4 | 15 | 0/15 | 0/15 | 100 [100, 100] | 0/15 | 15/15 / **0/15** | 4.05 m |
| A5 | 15 | 0/15 | 0/15 | 100 [100, 100] | 0/15 | 15/15 / **0/15** | 3.80 m |
| A6 | 15 | 0/15 | 0/15 | 100 [100, 100] | 0/15 | 15/15 / **0/15** | 3.60 m |
| B–E, bases (all rows) | 30 | 30/30 | 30/30 | — | 9/30 | 30/30 / 30/30 | 3.05 m |
| B1 | 21 | 13/21 | 13/21 | 38.1 [17.6, 57.9] | 7/21 | 21/21 / 13/21 | 3.30 m |
| C1 | 20 | 17/20 | 17/20 | 15.0 [0.0, 33.3] | 7/20 | 20/20 / 20/20 | 2.72 m |
| D2 | 29 | 25/29 | 25/29 | 13.8 [3.3, 26.7] | 9/29 | 29/29 / 25/29 | 3.30 m |
| D5 | 8 | 8/8 | 8/8 | 0.0 [0.0, 0.0] | 8/8 | 8/8 / 7/8 | 1.02 m |
| E1 | 21 | 21/21 | 18/21 | 0.0 [0.0, 0.0] (L2: 14.3) | 6/21 | 21/21 / 21/21 | 3.05 m |
| E2 | 16 | 12/16 | 12/16 | 25.0 [5.9, 47.1] | 1/16 | 16/16 / 16/16 | 3.15 m |

- **Family drops** (criterion (a); arm 3, L1, levels pooled): A 100.0 (n 84) [100, 100]; B 38.1 (n 21) [17.6, 57.9]; C 15.0 (n 20) [0.0, 33.3]; D 10.8 (n 37) [2.4, 22.2]; E 10.8 (n 37) [2.6, 21.1]. Under L2, E's drop is 18.9 and the others are unchanged.
- No `far_from_target` or `exclusion_broken` flag was raised in any arm-3 scene.

Table 2.2. Oracle C (arm 3, L1) by situation, seeds pooled (`score/tables.txt`, "By situation"; `curves_by_situation.png`).

| Family, level | Lines | Conversation groups | Audiences |
|---|---|---|---|
| Bases | 15/15 | 9/9 | 6/6 |
| A1–A6 | **0/84** | — | — |
| B1 | **4/12** | 7/7 | 2/2 |
| C1 | **9/12** | 7/7 | 1/1 |
| D2 | **11/15** | 9/9 | 5/5 |
| D5 | — | 8/8 | — |
| E1 | 15/15 | 6/6 | — |
| E2 | **11/15** | 1/1 | — |

- Every failure is at a line. Every group and audience variant succeeds: 46 of 46 (38 group, 8 audience).
- **The solver alone** (arm 2; writer's tally, consistent with Table 2.1): it succeeds in every conversation-group scene (9 of 9 bases, 38 of 38 variants) and in no line or audience scene (lines 0 of 15 bases and 0 of 153 variants; audiences 0 of 6 and 0 of 8). So at groups the reference constraints add nothing, even with second groups, clusters and bystanders; at audiences the watcher constraints are needed. Arm 2 stands at least 110 deg off the target person's facing in 4 group variants (descriptive), and all 4 still succeed under L1.

![Oracle curves](score/curves_oracle.png)

Figure 2.1 (`score/curves_oracle.png`). Oracle C under L1 and L2, the solver alone and the full oracle, per family. A drops to 0 at A1 and stays there; B, C and E2 drop; D5 sits at 100% because it exists only at groups.

![Oracle curves by situation](score/curves_by_situation.png)

Figure 2.2 (`score/curves_by_situation.png`). Oracle C under L1, by situation: every drop is in the lines panel; the group and audience panels are flat at 100%.

### L1 and L2

- **Disagreement** (L1 success with L2 failure, per scene):
  - Oracle C (arm 3): 3 of 199 variants (1.5%), 0 of 30 bases. All three are E1 at `eval_11` (one per seed), where people do not matter and the standard goal lies 0.226, 0.253 and 0.270 m (body gap) from the added E person.
  - Solver alone (arm 2): 3 of 199 (1.5%).
  - **Full oracle (arm 1): 85 of 199 (42.7%)**: all 72 variants of A2–A6 (12 + 4 × 15), 8 of B1, 4 of D2 and 1 of D5 (writer's tally from `score/rows_oracle.jsonl`; it matches the verifier's table). In every case the full oracle's stop, L1's own representative point, lies 0.53–0.81 m from an added person, inside L2's 0.85 m (score verifier).
  - Noisy C (arm 4, mean over draws; not independently verified): Qwen 2.1%, GLM 4.5%, DeepSeek 4.1%.
- **L2-empty:** 0 of 199 variants and 0 of 30 bases.
- **Why the solver agrees.** It keeps 1.0 m from every person, beyond L2's 0.85 m centre distance, so its agreement with L2 is built in (disclosed in FROZEN section 9(c)). The full-oracle row shows what L2 does to a stop that does not keep that distance.

---

## 3. The documented failures (criterion (b))

### What was done

A failure **fires** in a variant when its detector is true there and false in the variant's base. A variant is **relevant** when the failure's precondition holds and its base does not fire. Detectors are read on arm 3 (FROZEN section 7).
- (i) Bystanders behind a line taken as waiting: the solver's `joined_waiting` holds a non-member. Relevant: line variants where `come_after` is used.
- (ii) A crowd keeps the robot out: the L1 cells at least 1.0 m from every person (F) are empty in the variant but not in the base, the place is outside L1, and nothing is excluded.
- (iii) Large-circle off-facing: at group rows, the place is at least 110 deg off the target person's facing.
- (v) A named cause, not one of the four: no L1 cell is standable on the solver's own floor and the place is outside L1.

### Results

Table 3.1. Firing on arm 3.

| Failure | Fires / relevant | By family |
|---|---|---|
| (i) bystanders absorbed behind a line | **99/153 = 64.7%** | A 84/84, B 8/12, C 3/12, D 4/15, E 0/30 |
| (ii) a crowd keeps the robot out | 0/199 | 0 in every family |
| (iii) large-circle off-facing | 0/38 | B 0/7, C 0/7, D 0/17, E 0/7 |
| (iv) hindmost-watcher rule dropped | 0/8 | B 0/2, C 0/1, D 0/5; the rule was applied in all 8 |
| (v) region floor not standable (named cause only) | 0/199 | 0 in every family |

- **Sensitivity** (disclosed in advance): without A5–A6, (i) fires in 69 of 123 (56.1%); without family A, in 15 of 69 (21.7%). (ii)–(iv) stay at 0.
- **What absorption does.** Example: in `i06_s0_A1_eval_06_waiting_line_1` the solver takes the confuser (p1, 1.9 m behind with its back to the line) as waiting and stops 2.93 m from the target, outside the region. In the 15 B, C and D failures the solver absorbed one or two added people and stopped 2.8–5.5 m from the target point (writer's tally).
- **Named causes of the arm-3 failures.** All 99 failures in A–D carry cause (i). The 4 E2 failures at lines carry none.
- **Post hoc: hidden floor (E2).** In those 4 variants (`s0 eval_08`, `s1 eval_07`, `s1 eval_09`, `s2 eval_08`), the person 2 m from the camera hides most of the floor round the place in turn: standable L1 cells fall from 648–662 in the bases to 50–207 (of 706–720). The solver then stops 0.35–0.46 m from the place in turn, outside the region and with its heading out of tolerance, with no flag. Detector (v) needs every L1 cell unstandable, so it does not fire. Over all 30 E variants at lines, the median variant-to-base ratio of standable L1 cells is 0.53.
- **Post hoc: why (ii)–(iv) do not fire.** The builder keeps every added person at least 1.2 m from the members and 0.3 m outside the structure's space (c8). So F never empties (smallest 33 cells; lowest variant-to-base ratio 0.12); group circles stay as built (largest off-facing 99.4 deg, as in the group bases); and the watcher rule applied in all 8 relevant audience variants, while most audience variants were rejected at build time.
- Arm 2, descriptive: (ii) 0, (iii) 4, (v) 0 of 199.

---

## 4. The noise model and its validation

### What was done (FROZEN section 8)

- **Outcomes per visible person:** located (matched within 0.5 m, as `run._matched_people`); mislocated (unmatched, but paired with an unmatched detection within 40 px of its mark, I03's data-informed radius); or missed. Unpaired detections are false people.
- **Parameters:** p_miss by role (members, others, staff); q_mis = P(mislocated | listed); empirical located and mislocation offsets in the camera-ray frame; resampled false-person counts and offsets; yaw error and unknown share by A0 view class. Pooled over models where a model has fewer than 10 samples.
- **Simulation:** i.i.d. per person and per answer, 20 draws per scene and model, applied identically to bases and variants with no situation branch. The reference constraints keep listed numbers (mislocated people included) and drop missed ones.
- **Validation (K4):** solver-step records are re-solved 20 times with simulated people, reference constraints and the recorded target; other records keep their verdict. Gap = simulated − recorded C, pooled over the seven runs. A model with |gap| > 10 points gets no noisy curve; K4 fires only when every model misses. Diagnostic: R5, each answer's actual people with the reference constraints.

### Fit

Table 4.1. Noise parameters (`score/noise/fit.txt`, `score/noise_model.json`).

| Model (answers; visible people) | p_miss member | p_miss other | p_miss staff | q_mis | False people per answer | Located offset, median | Mislocation radial offset, median (n) | Yaw within 45 deg: frontal / oblique / profile / back |
|---|---|---|---|---|---|---|---|---|
| Qwen 3.8 (114; 480) | 2/372 | 1/84 | 1/24 | 11/476 | 0.009 (offsets pooled from all three) | 0.104 m | −0.82 m (11) | 0.972 / 0.778 / 0.780 / 0.869 |
| GLM-5.3 (118; 485) | 49/372 | 8/86 | 2/27 | 23/426 | 0.305 | 0.116 m | +4.07 m (23) | 0.657 / 0.533 / 0.485 / 0.814 |
| DeepSeek V4.1 (128; 499) | 15/372 | 7/89 | 0/38 | 32/477 | 0.102 | 0.119 m | +3.36 m (32) | 0.867 / 0.606 / 0.574 / 0.810 |

- **Cross-check.** Recall computed I03's way equals `results/still-scenes/perception/depth-blob-detector/stageA/vlm_rows.jsonl` for all three: 465/480, 403/485, 445/499.
- GLM's and DeepSeek's mislocated people sit 3–4 m farther along the camera ray, which matches the phantom people on the back wall in REPORT finding 8.
- The score verifier's own fit gives identical parameters, including every offset list and the placeholder-only fit.

### Validation

Table 4.2. Simulated against recorded C on the development records (n 408 per model; 0 records differ between the recorded and re-scored verdicts).

| Model (solver steps) | Recorded C | Simulated C | Gap (K4) | R5 | Simulated − R5 | R5 − recorded |
|---|---|---|---|---|---|---|
| Qwen 3.8 (227) | 93.4 | 93.7 | **+0.3** | 94.9 | −1.2 | +1.5 |
| GLM-5.3 (225) | 73.3 | 78.7 | **+5.4** | 81.1 | −2.5 | +7.8 |
| DeepSeek V4.1 (237) | 73.0 | 79.3 | **+6.2** | 83.8 | −4.5 | +10.8 |

- **All three models are valid; K4 does not fire.** R5 reproduces I01's rows on 756 of 756 realistic records.
- **FROZEN point 2 (validate against R5?) is moot here:** the gaps are within 10 points against either target.
- **By situation** (gap against recorded C, then writer's arithmetic against R5): no_one (n 180) Qwen 0.0 / 0.0, GLM 0.0 / 0.0, DeepSeek +1.7 / 0.0; lines (n 120) −2.5 / −4.1, +13.6 / −7.2, +1.0 / −10.6; groups (n 72) +1.9 / +1.9, +2.7 / −0.1, +18.5 / −1.0; audiences (n 36) +8.1 / −3.1, +10.0 / −3.9, +21.9 / −14.2.
- **Limits.** The fit is in-sample, the noise is i.i.d. and does not grow with density, and the realistic and placeholder domains are pooled. "i.i.d. noise is too gentle" (REPORT I06) still applies to crowds.

---

## 5. The noisy curves

Table 5.1. Noisy C (arm 4) under L1: mean over 20 draws per scene, then over scenes. Separation = oracle C − noisy C, in points.

| Family, level | n | Oracle C | Qwen 3.8 | GLM-5.3 | DeepSeek V4.1 |
|---|---|---|---|---|---|
| A, bases | 15 | 100 | 94.0 | 78.3 | 77.0 |
| A1 | 12 | 0 | 23.3 | 36.3 | 37.1 |
| A2 | 12 | 0 | 3.3 | 13.3 | 16.2 |
| A3 | 15 | 0 | 2.7 | 11.7 | 8.0 |
| A4 | 15 | 0 | 0.7 | 4.7 | 2.3 |
| A5 | 15 | 0 | 0.3 | 1.7 | 2.0 |
| A6 | 15 | 0 | 0.0 | 3.7 | 2.3 |
| B–E, bases | 30 | 100 | 95.8 | 81.7 | 82.8 |
| B1 | 21 | 61.9 | 61.9 | 57.4 | 52.9 |
| C1 | 20 | 85.0 | 85.5 | 77.5 | 77.8 |
| D2 | 29 | 86.2 | 80.0 | 68.3 | 71.9 |
| D5 | 8 | 100 | 98.1 | 93.8 | 96.9 |
| E1 | 21 | 100 | 95.2 | 80.2 | 82.9 |
| E2 | 16 | 75.0 | 64.4 | 50.0 | 50.9 |
| **Top level (A6, B1, C1, D5, E2)** | **80** | **62.5** | **60.3 (+2.2 [−0.5, 4.5])** | **54.5 (+8.0 [3.7, 12.4])** | **53.6 (+8.9 [4.4, 13.0])** |
| Bases | 30 | 100 | 95.8 (+4.2) | 81.7 (+18.3) | 82.8 (+17.2) |
| Top level without A6 (post hoc) | 65 | 76.9 | 74.2 (+2.7) | 66.2 (+10.7) | 65.5 (+11.5) |

- **Both episodes agree in every draw** for every model (229 scenes each, none missing).
- **At A, noise can only help.** Oracle C is 0%, and a missed or displaced confuser is not absorbed, so noise raises success at A1 to 23–37%.
- **Crowds do not amplify the noise here** (writer's reading of Table 5.1). At B1, C1, D2, D5 and E1, no model's separation exceeds its separation on the bases by more than 2 points (largest excess: Qwen at D2, +6.2 against +4.2). E2 is the exception for all three: +10.6, +25.0 and +24.1, against +4.2, +18.3 and +17.2 on the bases.
- **Secondary arm** (rates fitted on the placeholder answers alone; decides nothing): top level +1.2, +7.8, +8.2; bases +1.3, +18.7, +15.3.
- **Detectors under noise** (descriptive, not differenced against the base; % of variant draws): (i) is true in 45.5 (Qwen), 32.9 (GLM) and 34.6 (DeepSeek); (iii) in 1.4, 2.0 and 2.4; (iv) and (v) in 0.
- The score verifier checked the separations' arithmetic from the scoring check's per-scene rows (top level +2.19, +8.00, +8.88; bases +4.17, +18.33, +17.17; without A6 +2.69, +10.69, +11.46; A1 23.3, 36.3, 37.1%). Its own re-simulation did not finish.

![Noisy curves](score/curves_noisy.png)

Figure 5.1 (`score/curves_noisy.png`). Oracle C and each model's noisy C, per family. The noisy lines sit below oracle C everywhere except at A, where they rise above 0 at the weak-cue levels.

---

## 6. Criteria (a)–(e) and the kill lines (REPORT I06 verbatim; readings in FROZEN section 9)

| Criterion | Value | Outcome |
|---|---|---|
| (a) "Oracle C drops at least 15 points in at least 2 families, or falls monotonically along family D or A with named causes." | Families at 15 or more: A 100.0 (n 84), B 38.1 (n 21), C 15.0 (n 20). Monotone along A: 100 at every level, and 15 of 15 A6 failures carry cause (i). D is not monotone (13.8, then 0.0; D5 exists only at groups). Without A5–A6: met; without A: met (B, C) | **Met** (by construction through A; also met without A) |
| (b) "At least 2 of the 4 documented failures fire in at least 10% of relevant variants." | (i) 99/153 = 64.7%; (ii) 0/199; (iii) 0/38; (iv) 0/8. Without A5–A6 or without A, only (i) clears 10% | **Not met** |
| (c) "L1 and L2 verdicts differ in at most 5% of episodes. … L2 empty in at most 2%." | Arm 3: 3/199 = 1.5%; L2-empty 0/199. Partly built in (solver standoff 1.0 m, L2 0.85 m) | **Met** |
| (d) "At least 70% of variants pass `check_scene`, and full oracle scores 100%." | 199/270 = 73.7% (no spot counted as a rejection); full oracle 199/199 variants, 30/30 bases | **Met** (3.7 points above the line) |
| (e) "Noisy perception separates from oracle by at least 10 points at the top level." | Top level (n 80): Qwen +2.2, GLM +8.0, DeepSeek +8.9; 3 valid models, none at 10 | **Not met** |
| K1 "Oracle C at least 95% everywhere, and noise within 5 points." | Lowest arm-3 cell 0% (A1–A6) | Does not fire |
| K2 "L1/L2 disagreement above 30%, or L2 empty above 10%." | 1.5%, 0.0% | Does not fire |
| K3 "More than 50% of layouts rejected." (read as variants) | 71/270 = 26.3% | Does not fire |
| K4 "The noise model misses the recorded C by more than 10 points." (fires only when every model misses) | +0.3, +5.4, +6.2 | Does not fire (rests on the scoring check's validation) |
| **Verdict** (build the full pass when (a)–(e) hold and no kill fires) | (a), (c), (d) met; (b), (e) not met; no kill | **Neither; the project decides** |
| Post hoc, reported only: (e) without A6 | GLM +10.7, DeepSeek +11.5, Qwen +2.7 (n 65) | Would be met for 2 models; changes no verdict |

- (a) can be met through A by construction, since A was built around failure (i) (disclosed in FROZEN section 9).
- The verdict cannot hinge on K4 in practice: (b) fails on verified data and K1–K3 cannot fire, so the verdict is neither unless K4 fired for all three models, which the identical noise parameters make implausible (score verifier).

---

## 7. C5 gate-1 recount on the widened generator (`score/c5_recount.txt`)

- **Before.** C5's gate 1 asks C2 Step 0 for at least about 30 distinct generator configurations. The phase-1d count over the 26 development rows gave PRIMARY 22 and LOWER 15: gate 1 not met, and C2's Source A kill fires (`results/still-scenes/learned-models/flow-model-gates/gate1.txt`).
- **Readings** (FROZEN section 10, extending the phase-1d note): PRIMARY' adds (family, level) to C5's tuple and counts every (row, family, level) with at least one kept variant in seeds 0–2; STRUCTURE counts A's six levels and E's two depths once per row; LOWER' counts (situation, structure size, bystanders including added ungrouped people, second-group presence).

| Reading | Count | Gate (≥ 30) |
|---|---|---|
| PRIMARY' (decides) | **92** = 22 development tuples + 70 variant configurations (A 28, B 8, C 8, D 13, E 13) | Met |
| STRUCTURE | 63 | Met |
| LOWER' | 49 (phase 1d: 15) | Met |

- **EMPIRICAL** (descriptive; not recomputed by the verifier): in C5's frame, after the best one-to-one matching with or without the mirror, the added people move a median 0.13 m between seeds (median over 68 configurations with at least two seeds). A and E configurations sit at fixed offsets from the members and move about 0.02–0.32 m; B, C and D configurations move 0.21–3.39 m (writer's reading of the per-configuration medians). Base members move 0.02–0.09 m at lines and groups and 0.33–0.51 m at audiences.
- **Distinctness, as a judgement** (FROZEN section 10): every configuration sits on the same 10 base structures and adds no situation. So the widening adds diversity within situations and does nothing for C5's leave-one-situation-out folds. A's levels and E's depths are near-duplicates for training. The configurations come from development seeds 0–2; training data would need fresh seeds, which belong to other steps.
- The score verifier checked PRIMARY' 92 and STRUCTURE 63 by hand; LOWER' 49 and EMPIRICAL were not recomputed (section 8).

---

## 8. Where the compute checks and the verifiers differ
Settled by the verifier (its value is used above):
1. **Base labels in the build manifest.** `builder/label_checks.py` merges label rows by scene id alone, and base ids repeat across seeds. So the 20 base entries of seeds 0 and 1 in `build/manifest.json` and `manifest.jsonl` carry seed 2's label fields (`l1_cells_e0`, `l2_cells_e0`, `rep_point_moved_m`; for example seed 0 `eval_06`: 708 stored, 719 true), and `build/labels.txt`'s base lines are seed 2's 10 rows counted three times. All 30 bases do pass, on the per-seed rows (`labels_rows_seed{0,1,2}.jsonl`) and in the verifier's own runs. The scoring read the per-seed rows keyed by (seed, scene id), so no number on this page is affected. The manifest is left as written; anyone reading base labels from it should re-key it first.
3. **Order.** The build's `self_test.txt` (01:49:57) and `base_check.txt` (01:49:55) predate the last edit of `builder/crowd.py` (01:51:41) by about two minutes, so they tested an earlier version. No effect: the verifier's own checks accept 21 of 21 and reproduce the final builder's decision at every attempt. `readings.txt` (01:53:45) and `DEVIATIONS.md` (01:54:04) were written about two minutes after variant generation began (01:51:59), before any scoring. The note allows addenda before the affected data are scored, and the readings change no rule.

Found while writing (the tables' values are used above):
5. **"0/69".** Both the scoring check's and the score verifier's summaries say the full oracle fails L2 in "A2–A6 (0/69)", and their situation lines say oracle C scores "A 0/69" at lines. `score/tables.txt`, the verifier's own `verify_i06_score/out/summary_oracle.txt` and `score/rows_oracle.jsonl` give 0 of 72 for A2–A6 (12 + 4 × 15) and 0 of 84 for A at lines (every A variant is at a line). The arm-1 disagreement total, 85 = 72 + 8 + 4 + 1, agrees with the tables. This looks like a slip in the summaries; it changes no verdict.

- **Arm-4 L1/L2 disagreement** (2.1%, 4.5%, 4.1%): not recomputed.
- **C5 recount:** PRIMARY' 92 and STRUCTURE 63 verified by hand; LOWER' 49 and the EMPIRICAL 0.13 m median not recomputed (the verifier's `v_c5.py` was written but not run).
- **The seed-9 incident** can only be checked as an absence of artifacts; none was found.

- **FROZEN step 9's README** is this page.

---

## 9. What it means (inference)

**For crowded scenes.**
- Under oracle people and the reference constraints, the solver held in every conversation-group and audience variant built (46 of 46), and failed only at lines. Of the 103 line failures, 99 come from one documented mechanism: non-members behind or beside the line taken as waiting (failure (i)), at a rate of 64.7% of relevant line variants (21.7% without family A, which was built to trigger it). So generated crowds confirm that bug at a high rate; they found no new one.
- One new mechanism appeared post hoc: hidden floor. A person 2 m from the camera hid most of the region's floor in 4 of 15 E2 line variants, and the solver stopped 0.35–0.46 m short, outside the region, with no flag. It is general (any occluder near the robot) and rests on 4 scenes.
- Audiences are thinly tested (8 kept variants of 36 planned), and D5 exists only at groups. A full pass would need families whose top levels exist in every situation, or reporting by situation.
- Perception noise costs GLM and DeepSeek 17–18 points on the plain bases, and in this i.i.d. model crowds add little to that, except E2. The probe therefore gives no zero-call reason to expect C to degrade faster in crowds than in plain scenes. That is a property of the noise model (density-independent, in-sample) as much as of the solver; real VLM people answers in crowds are untested.

- **The L1/L2 decision, with the mentor.** Criterion (c) passes for the solver (1.5%), but that agreement is built in. L1's own representative point fails L2 in 85 of 199 variants (42.7%), including every A2–A6 variant. So the decision matters for methods that aim near the region's reference point: R's and E's marked points, and labels sampled for C5. It does not matter for the solver. The cue threshold decides how family A is read. At A6 the confuser stands on the place in turn that L1 marks as correct, facing like a member; L1 counts joining behind it as a failure. The probe's criteria come out the same without A5–A6, so the threshold changes the reading of A, not the verdict.
- **The occlusion decision for family F.** The probe gives no occlusion evidence. Family E never hid a relevant member (37 of 37 kept their visible samples), because the scene check's visibility rule held and E aimed at the region's reference point. F's partly hidden target, and any variant that hides a member, still need the decision, and so does live check L4. The hidden-floor mechanism above suggests what such scenes would test besides grounding: whether the solver can stand on floor the depth image does not show.
- **The VLM crowd run's size.** The REPORT's guess is about 2,000 calls: 14 layouts × 3 crowd levels × 3 trials × about 5.3 calls per episode × 3 models. The probe suggests where the levels should go. Oracle C bends only at lines, so the line levels are where C can be compared with R and E: for example A1 (a clear "not waiting" cue the solver ignores), A6, and B1 or D2. Whether R and E, which see the image, read a confuser's facing better than the solver does is an open question that family A can answer. Groups and audiences do not bend under oracle C, but R and E were never tested there in crowds.
  - A lines-only version at the same cost per episode would be 15 line layouts (5 rows × 3 seeds) × 3 levels × 3 trials × 5.3 × 3 models ≈ 2,100 calls (guess).
  - The variants are placeholder renders with unchanged rooms. A realistic run would need Blender re-renders, and possibly re-furnished rooms (FROZEN point 4).
  - The calls are free gateway calls but a big run, so the announcement rule applies.

**For C5.**

2. The L1/L2 rule and the cue threshold, with the mentor, now that L1's own stop fails L2 in 42.7% of variants.
4. For any full pass: define family top levels by situation. (e)'s pooled top level includes A6, where oracle C is 0% and noise can only help; without it, GLM and DeepSeek separate by 10.7 and 11.5 points (post hoc, reported only).
5. For the VLM crowd run: lines only or lines plus groups, placeholder renders or realistic re-renders, and how many calls.
6. FROZEN point 2 (validate the noise against R5?) is moot: all three models pass against either target.

---

## 10. Proposed REPORT edits (not applied)

### 1. "How to read": define [phase 2]

Old:
```text
Step 3b also ran a local detector and pose model on the GPU, on whitelisted scenes and those seeds only; no provider model was called. Parts registered after earlier results on the same runs are marked data-informed.
```
New:

### 2. I06, first probe: the result

Old:
```text
  - The noise model misses the recorded C by more than 10 points. Then report oracle curves only.
```
New:
  - Scope. Families A–E, zero calls. Family F was deferred: it needs generator support for a second same-kind target and the project's occlusion decision, so L4 still waits on it. Base layouts: the 10 development rows of seeds 0–2 that a family applies to (lines `eval_06`–`09` and `13`, groups `eval_10`–`12`, audiences `eval_14`–`15`), reproduced with the package's generator; the 14 with a whitelisted copy are identical to it. Variants keep the base room, add non-members only, and leave the L1 regions unchanged (199 of 199).
  - Build. 199 of 270 planned variants kept (73.7%; 65 found no spot, 6 failed the visibility check), plus the 30 bases. Every label check passes on all 229 scenes, and full oracle scores 100% under L1. Which levels exist depends on the situation: D5 only at groups (8 of 9; 0 of 21 elsewhere), B and C none at the line of 6 or the audience of 5, and 8 of 36 planned audience variants were kept. E as built never hid a relevant member (visible samples unchanged in 37 of 37), so it tests hidden floor, not occlusion.
  - Oracle C (oracle people, reference constraints) under L1: bases 30 of 30; A 0 of 84, at every level; B1 13 of 21 (drop 38.1); C1 17 of 20 (15.0); D2 25 of 29 (13.8); D5 8 of 8; E1 21 of 21; E2 12 of 16 (25.0). Every failure is at a line: all 46 group and audience variants succeed, and the solver with no constraints also succeeds in every group scene.
  - Documented failures. (i), bystanders behind a line taken as waiting, fires in 99 of 153 relevant line variants (64.7%; without family A, 15 of 69) and explains all 99 failures in A–D. (ii), (iii) and (iv) fire 0 times (0 of 199, 0 of 38, 0 of 8), post hoc because the frozen families keep added people at least 1.2 m from the members and outside the structure's space. E2's 4 line failures have no named cause; post hoc, the added person hides most of the floor at the place in turn, and the solver stops 0.35–0.46 m short, outside the region, with no flag.
  - L1/L2. Oracle C disagrees in 3 of 199 (1.5%; all E1 at one group layout where the standard goal is used), and L2 is never empty. That agreement is built in (the solver keeps 1.0 m, L2 needs 0.85 m): the full oracle's own stop fails L2 in 85 of 199 (42.7%), including all 72 variants of A2–A6.
  - Noise model (fitted in-sample on the seven development runs per model; i.i.d.). Simulated minus recorded C: +0.3 (Qwen), +5.4 (GLM), +6.2 (DeepSeek) points, n 408 each, so all three are valid; against I01's R5 construction the gaps are −1.2, −2.5 and −4.5. Separation from oracle C at the top level (A6, B1, C1, D5, E2; n 80): +2.2, +8.0 and +8.9 points; on the bases +4.2, +18.3 and +17.2. Post hoc, without A6 (where oracle C is 0% and noise can only help): +2.7, +10.7 and +11.5.
  - Criteria: (a) met (A 100.0, B 38.1, C 15.0; also met without A); (b) not met (only (i)); (c) met; (d) met (73.7%, full oracle 100%); (e) not met. No kill line fires (K3: 26.3% rejected; K4: no model misses). Verdict: neither, and the project decides.

### 3. Section 6, step-4 row

Old:
```text
| 4 | Days 14–17 | I06 probe with families A–F, about 400 placeholder scenes (about 200 CPU-minutes, 10–20 min wall on 20 CPUs). Family F's zero-call label check runs before family F is used. | First result: about 3 d; family F about 1 d more (guess). The full pass is about 14 person-days, more than 2 weeks in the worse case. | 0 | Rules L1/L2 and family F written first; announce the shared CPU |
```
New:
| 4 | Days 14–17 | I06 probe with families A–F, about 400 placeholder scenes (about 200 CPU-minutes, 10–20 min wall on 20 CPUs). Family F's zero-call label check runs before family F is used. **First probe done 18 Oct** (`results/still-scenes/crowds/README.md`) [phase 2]: families A–E only (F deferred), 199 of 270 planned variants kept plus 30 bases, under a protocol note frozen and hashed before any variant was built. Verdict neither: (a), (c) and (d) met, (b) and (e) not met, no kill fired. Oracle C fails only at lines, by bystanders absorbed behind the line (99 of 153 relevant variants); the noise separation at the top level is 2.2–8.9 points. | First result: about 3 d; family F about 1 d more (guess). The full pass is about 14 person-days, more than 2 weeks in the worse case. The first probe's build and scoring ran in about 4 h of wall time (18 Oct). | 0 | Rules L1/L2 and family F written first; announce the shared CPU. Family F still needs the project's occlusion decision and generator support for a second same-kind target |

### 4. Section 6, the decision-gate row

Old:
| Gate | About day 17 | **Decision gate** with the project, after the I06 probe returns: results so far; choose a programme; confirm the metric list; approve the live checks. | — | — | project |
New:
| Gate | About day 17 | **Decision gate** with the project, after the I06 probe returns: results so far; choose a programme; confirm the metric list; approve the live checks. [phase 2] The I06 first probe returned on 18 Oct as neither (`results/still-scenes/crowds/README.md` section 9). It brings to the gate: the programme choice (B's case was conditional on the probe passing); the L1/L2 decision and the cue threshold, with the mentor (the solver agrees with L2 by construction, while L1's own reference point fails L2 in 42.7% of variants); the occlusion decision for family F (family E never hid a member, so the probe has no occlusion evidence); whether to add a family that crowds the region itself; and the VLM crowd run's size and levels. | — | — | project |

### 5. Programme B: experiment 2

Old:
```text
2. I06 probe.
3. I35 Part A.
```
New:
```text
2. I06 probe. [phase 2] First probe done 18 Oct, families A–E: neither ((b) and (e) not met, no kill). Oracle C fails only at lines, by bystanders absorbed behind the line; groups and audiences held in every variant built (`results/still-scenes/crowds/README.md`).
3. I35 Part A.
```

### 6. Section 5, "Which to start with": Programme B's condition

Old:
```text
- B may be the main contribution of the term, but only if the I06 probe passes (guess). That is why the decision gate waits for the probe.
```
New:
- B may be the main contribution of the term, but only if the I06 probe passes (guess). That is why the decision gate waits for the probe. [phase 2] The first probe did not pass: it is neither, with no kill line fired (`results/still-scenes/crowds/README.md`). (inference) Its zero-call curves add one measured failure rate (bystanders absorbed behind a line, 64.7% of relevant line variants) and a post hoc hidden-floor mechanism, but no failure inside the region, since the frozen families kept non-members out of it. B's remaining case rests on the VLM crowd run (step 8, the only route to RQ2 for R and E), or on a second frozen probe that crowds the region after the L1/L2 decision.

### 7. Finding 9: the other constraint kinds in crowds

Old:
```text
So the current scenes barely need them. That raises the cheaper routes to that fact (I19, I24, I32, I05) and the scenes where the other kinds would matter (I06).
```
New:
```text
So the current scenes barely need them. That raises the cheaper routes to that fact (I19, I24, I32, I05) and the scenes where the other kinds would matter (I06). [phase 2] I06's first probe (oracle people, reference constraints) did not change this at conversation groups: the solver with no constraints succeeds in all 38 group crowd variants (second groups, clusters, bystanders, a person on the line of sight) and all 9 group bases, as oracle C does. At the placeholder audience rows, which the realistic sets lack, the watcher constraints are needed (no constraints 0 of 14, reference constraints 14 of 14; 6 bases, 8 variants). At lines `come_after` is needed, and it is also where every crowd failure comes from, by absorbing bystanders behind the line (`results/still-scenes/crowds/README.md` sections 2–3).
```

### 8. C5, gate 1

Old:
```text
  1. C2 Step 0 finds at least about 30 distinct generator configurations. If it finds fewer, widen the generator first with I06's crowd builder.
```
New:

### 9. Section 7: crowd labels (L1/L2) and the cue threshold

Old:
```text
- Crowd labels: L1 (members only), L2 (minus a 0.3 m gap to non-members and minus other groups' circles), or L3 (L2 with a fallback). The cue-strength threshold for ambiguous episodes (I06, I18). *Mentor input wanted.*
```
New:
- Crowd labels: L1 (members only), L2 (minus a 0.3 m gap to non-members and minus other groups' circles), or L3 (L2 with a fallback). The cue-strength threshold for ambiguous episodes (I06, I18). *Mentor input wanted.* [phase 2] I06's first probe measured both on 199 crowd variants (`results/still-scenes/crowds/README.md` section 2). For the solver they barely differ: oracle C disagrees in 3 of 199 (1.5%) and L2 is never empty, but that agreement is built in, since the solver keeps 1.0 m from people and L2 needs 0.85 m. L1's own reference point (the full oracle's stop) fails L2 in 85 of 199 (42.7%), including all 72 variants with a confuser 0.6–1.6 m behind a line (A2–A6). (inference) So the decision matters for methods that aim near the region's reference point (R's and E's marked points, labels sampled for C5), not for the solver. The cue threshold decides how family A is read: at A5–A6 the confuser stands 0.6–0.8 m behind the line, facing within 30 deg of the line's facing, and L1 counts joining behind it as a failure; the probe's criteria come out the same without A5–A6.
### 10. Section 7: the occlusion decision

Old:
```text
- Whether "all relevant people visible" may be relaxed for occlusion scenes (I06 family F's partly occluded target, I30, I17), and whether bystanders near a line change its acceptable region.
```
New:
- Whether "all relevant people visible" may be relaxed for occlusion scenes (I06 family F's partly occluded target, I30, I17), and whether bystanders near a line change its acceptable region. [phase 2] I06's first probe gives no occlusion evidence: its family E puts a person on the line of sight to the region's reference point, but the scene check's visibility rule held, and no relevant member lost a visible sample (37 of 37 E variants). Post hoc, it found that hidden floor matters: with a person 2 m from the camera, 4 of 15 line variants failed because that person hid most of the floor at the place in turn and the solver stopped 0.35–0.46 m short, outside the region, with no flag. Family F, and any variant that hides a member, still need this decision (`results/still-scenes/crowds/README.md` sections 1 and 3). On bystanders near a line: the probe kept each line's L1 region unchanged, and the solver took non-members about 0.6–2.5 m from a line as waiting in 99 of 153 relevant variants.

### 11. Revision note

Old:
```text
and add step-3b notes to the opening lines, Programmes A and C, section 6's cut order, I46's seeds, section 7's runs and section 8's render threat.
```
New:
- **Phase 2, I06 first probe (18 Oct).** I06's first crowd probe is done (`results/still-scenes/crowds/README.md`), with zero calls: families A–E on the development layouts of seeds 0–2 (199 of 270 planned variants kept, plus 30 bases), built from new `[seed, 2, …]` streams so the existing scenes stay byte-identical. Family F was deferred (generator support and the project's occlusion decision). The protocol note was frozen and hashed before any variant was built, and independent verifiers reproduced the build from its streams and the oracle arms scene for scene; the score verifier's own noise validation did not finish. Verdict neither: (a), (c) and (d) met, (b) and (e) not met, no kill fired. Oracle C fails only at lines, by bystanders absorbed behind the line (99 of 153 relevant variants); the solver's agreement with L2 is built in, while L1's own stop fails L2 in 42.7% of variants; noisy perception separates from oracle C by 2.2–8.9 points at the top level. The C5 gate-1 recount on the widened generator gives 92 configurations. "How to read" defines [phase 2], and I06, section 6 (step 4 and the decision gate), Programme B and section 5's start recommendation, finding 9, C5's gate 1 and section 7 (crowd labels, occlusion) carry the results; optionally also section 1.2 (step 6) and section 8's circularity bullet.

### 12. Section 1.2, phase-2 step 6 (optional)

Old:
```text
6. I06 crowd probe, with the distractor family F: L1 vs L2 labels, true and noisy people. Its zero-call output is solver-degradation curves, not RQ2 curves for R and E.
```
New:
```text
6. I06 crowd probe, with the distractor family F: L1 vs L2 labels, true and noisy people. Its zero-call output is solver-degradation curves, not RQ2 curves for R and E. [phase 2] First probe done 18 Oct for families A–E (F deferred): neither, with no kill fired; oracle C fails only at lines, by bystanders absorbed behind the line (`results/still-scenes/crowds/README.md`).
```

### 13. Section 8, circularity: L2 and the solver (optional)

Old:
```text
  - L2 labels mirror the solver's own-space rule.
```
New:
```text
  - L2 labels mirror the solver's own-space rule. [phase 2] Measured in I06's first probe: on 199 crowd variants, oracle C and the solver alone each disagree between L1 and L2 in 1.5%, the full oracle's stop in 42.7% (`results/still-scenes/crowds/README.md` section 2).
```

---

## 11. After this page was written: the score verifier's noise runs (the research lead, 18 Oct)

  78.7 (GLM) and 79.4 (DeepSeek) against recorded 93.4, 73.3 and 73.0: gaps +0.3, +5.4 and +6.4 (scoring step +0.3,
  record, and R5 equals I01's rows on 252 of 252 per model. All three models are valid; K4 does not fire.
  L1/L2 disagreement for DeepSeek is 4.0% (scoring step 4.1%).