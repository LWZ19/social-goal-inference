# I06 repair probe: a facing gate on joining a line (frozen protocol)

**Sources.**
- `phase2/i06/README.md` (sections 1–3), `phase2/i06/FROZEN.md` (family A's geometry, section 4) and `phase2/i06/score/` (`lib.py`, `arms.py`, `noisy.py`, `noise.py`, `tables.py`, `rows_oracle.jsonl`, `work/`, `noise_model.json`).
- `phase1/i01_repairs/repairs.py` and its `rows_*.jsonl`; `phase1/README.md` section 4 (I14's cells) and `phase1/i14_v0/i14.py`.
- `phase1c/i09/a2/replay_d.py`, `rows_*.jsonl` and `base_ph_*.jsonl`; `phase1c/README.md`; `phase2/i09/README.md` section 1.
- Package code: `src/sgi/solver.py` (sha256 `10ece70f521eff2b1224267fa0b91bfac546d5740f85622073989a16c6c5d800`; `_turn` at lines 514–541). Tools: `docs/model_notes/tools/research_harness.py` (`resolve`) and `replay_solver.py` (`solve_again`).

## 0. What the protocol author read and ran before hashing

**Code read:** the solver (`_bodies`, `_line_end`, `_behind`, `_turn`, `_settle`, `solve`); I06's scoring code; I01's `repairs.py`; A2's `replay_d.py`; the harness's `resolve`, which calls `sgi.solver.solve` through `replay_solver.solve_again`; and I14's record selection.

**Row files read**, for counts and keys only. These are R0 or recorded values; no repair outcome exists yet.
- `phase2/i06/score/rows_oracle.jsonl`.
  - R0 line failures under oracle C and L1: A1 12, A2 12, A3 15, A4 15, A5 15, A6 15, B1 8, C1 3, D2 4, E2 4, which is 103. There are none at groups or audiences.
  - 99 scenes have a non-empty `joined_waiting`, and every joiner is a non-member (`det['i']`). All 99 are R0 failures, and none is a base.
  - 153 variants are relevant for (i): A 84, B 12, C 12, D 15, E 30.
- `phase1/i01_repairs/rows_{old,new}_*.jsonl`, `phase1c/i09/a2/rows_{old,new}_*.jsonl` and `phase1c/i09/a2/base_ph_*.jsonl`. These give the goal-step counts and R0 targets in Table 5.1. The watching cells' goal steps are all audience records of `eval_14_audience_3` and `eval_15_audience_5`, 12 per dev seed. The fields used later are `C_rec`, `C`, `P`, `D` and `D_d.joined`.
- The keys of `phase2/i06/score/rows_noisy_glm.jsonl` and of `work/noisy_models_glm_seed0.jsonl`.

**Ran:** `precheck/format_check.py` (log in `precheck/format_check.log`), 06:30:57–06:31:00 on 18 Oct. It is R0 only: the package's own `_turn`, wrapped in a pass-through call counter. It found:
- A function installed as `sgi.solver._turn` at run time is the one the solver calls. That holds both through I06's `lib.solve_c` and through `research_harness.resolve` (1 call in each line solve checked).
- Oracle C, R0, seed 0, on `eval_06_waiting_line_1` (L1 true, joined []) and `i06_s0_A1_eval_06_waiting_line_1` (L1 false, joined [2]): equal to I06's arm 3 in L1, L2, xy and `joined_waiting`.
- The I06 noisy arm, R0, model glm, draws 0–1 of the same two scenes: L1, L2 and xy (4 dp) are equal to I06's `work/noisy_models_glm_seed0.jsonl`. So I06's seeding reproduces on these 4 draws.
- One recorded goal step with a `come_after` (`20261011-160500_s19_real_glm53_con`, `real_eval_06_waiting_line_1`, e0, trial 1) is reproduced by `H.resolve` (moved 0.0 m), with success false as recorded.

**Not done before hashing.**
- No repair rule was written or run.
- No one's facing was looked at, in any scene or record, for anyone the solver absorbed.
- Nothing was read from `keep/heldout`, s18 or s20_held runs, eval_16–19, synth seeds 3–99 or seeds 200–212, and no group_photo or lift_wait row was instantiated.
- Nothing outside `phase2/i06_repair/` was written.

## 1. Question and scope

- **Question (lead):** does a facing gate on joining remove most absorption failures without breaking real lines?
- **Background (lead).**
  - Under oracle people and reference constraints, the solver fails only at lines. 99 of the 103 line failures come from failure (i): `sgi.solver._turn` lets a detected non-listed person join the line when their own space covers the place in turn and they stand behind the last member, whatever way they face.
  - I09's frozen detector gives facing within 45 deg for 0.969 of non-profile people on unseen layouts (`phase2/i09/README.md` section 1). So a facing-aware rule is usable with a local people source.
- **Zero calls.** No model-provider or network call, and no GPU. Nothing in the package changes: the repair is research code installed at run time in the probe's own process (section 2).

## 2. The rules

**Verbatim (lead; fixed now; no other variant may be chosen from outcomes):**
- R0: the package's `_turn` unchanged (baseline; must reproduce I06's arm 3 and the recorded verdicts exactly).
- RF90 (primary): a non-listed person may join only if their facing points into the line's front half-plane: with u = the unit vector from that person to the current last member, and f = the person's facing unit vector, require f . u >= cos(90 deg) = 0 (strictly: f . u > -1e-9). A person with unknown yaw joins as in R0.
- RF60 (secondary): the same with f . u >= cos(60 deg).
- RN (ablation): nobody joins (only listed waiting people form the line).
- Everything else in the solver is unchanged.

**Implementation (choice, except where the design fixes it).**
- `rules.py` holds `turn_copy(waiting, apart, bodies, target_number, target_xy, robot_xy, th, *, rule)`. It is a line-for-line copy of `src/sgi/solver.py` `_turn` (sha256 above), using the solver module's own `_line_end`, `_behind`, `_Turn`, `_EPS` and `_MM`. It adds one condition to the candidate test:

```python
# inside the loop over bodies, after R0's own skips (waiting, target, apart):
if gap < th.standoff_m - _EPS and behind and rule != "RN" and gate(body, bodies[last], COS_MIN[rule]):
    taken.append((round(gap, _MM), number))

COS_MIN = {"R0": None, "RF90": 0.0, "RF60": 0.5, "RN": None}

def gate(body, last_body, cos_min):
    if cos_min is None or body.yaw is None:      # no gate (R0's test), or unknown yaw: joins as in R0
        return True
    to_last = last_body.xy - body.xy              # from the person to the current last member
    dist = float(np.hypot(*to_last))
    if dist < 1e-12:                              # (choice) no direction defined: as R0
        return True
    f = np.array([math.cos(body.yaw), math.sin(body.yaw)])
    return float(f @ to_last) / dist > cos_min - 1e-9
```

- **Readings fixed by that code.**
  - "Current last member" is the loop's `last` at that iteration. That person may have joined in an earlier iteration (choice; the design says "current").
  - The solver works in the robot's frame (positions by `to_robot_xy`, yaw minus the robot's yaw). That is a rotation, so f · u is the same as in the world frame.
  - The gate removes candidates only. Ordering, ties and the loop are R0's.
- **Installing a rule.** A context manager sets `sgi.solver._turn` and restores the package's function on exit. For R0 it installs the original function object (`ORIG = sgi.solver._turn`, captured at import before any patch). For RF90, RF60 and RN it installs `functools.partial(turn_copy, rule=...)`.
- **Copy check (before any repair rule runs).** In every R0 pass (section 8), the installed function returns `ORIG`'s result. It also calls `turn_copy(..., rule="R0")`, and the two must agree on every call: `last`, `joined` and `last_from_target` equal, and `place` equal element for element. Any disagreement stops the work.
- **Static check.** `rules.py` contains none of the tokens `situation`, `role` or `group`. It reads only the positions and yaws of detected bodies, the waiting list, the `apart` set and the target's number. In arm 1, the reference constraints read the situation label and the true group, and in arm 1b the noise model's roles read group membership. Both are I06's oracle arms, outside the rule, and circular as in I06 (REPORT section 8).
- **Built-in consequences, asserted on every solve.**
  1. Where R0's `joined_waiting` is empty, every rule gives R0's place exactly (distance 0) and R0's verdict. The gate can only remove candidates, so the first iteration takes no one.
  2. Under RN, `joined_waiting` is empty.
  3. Under oracle people (arm 1) every body has a known yaw.

  A failed assertion is an implementation fault: the work stops and it is reported.

## 3. Arm 1: I06 crowd variants, oracle C (primary for S1 and K2)

- **Scenes.** The 229 kept scenes, from `lib.kept_scenes()`: 30 bases and 199 variants, seeds 0–2. They are loaded with `lib.load`, which uses `H.guard_scene`. I06's `score/lib.py`, `noise.py` and `noise_model.json` are imported from `phase2/i06/score/` unchanged, and their sha256 values are recorded at start. Thresholds come from `lib.load_config()` (the default config, as in I06).
- **The arm** is I06's arm 3, unchanged: `lib.solve_c(inputs, truth, et, th)`. That means the oracle target, the reference context check (`reference_people_matter`), `OraclePeople` numbered by `number_people`, I01 R5's reference constraints, and `solver.solve`.
  - It runs per scene, per episode (e0, e1) and per rule (R0, RF90, RF60, RN).
  - In the 24 scenes where people do not matter, the standard goal is used under every rule, as in I06.
- **Labels, as I06 scored them.**
  - L1 = `sgi.evaluate.score(...).success`.
  - L2 = L1 and `lib.l2_ok(xy, truth, truth.group_for(eid, "t0"), th)`.
  - Both episodes must agree in L1, L2 and xy under every rule (asserted, as in I06). The scene's verdict is e0's (choice, as I06's tables).
- **Detector (i).** `lib.detectors(...)` runs on e0 under each rule. It gives `det_i` (a non-member in `joined_waiting`), `come_after_used`, `joined_waiting` and `joined_sources`, and all of its other fields are kept as descriptive values.
  - Under a rule, (i) fires in a variant when `det_i` is true there and false in its base under the same rule.
  - The denominator is I06's 153 relevant variants (`relevant['i']` in `rows_oracle.jsonl`) (choice). Under R0 no base has a joiner, so the base side is unchanged under every rule (consequence 1).
- **R0 reproduction (step 1 of section 8).** All 229 scenes and both episodes must match, or nothing further is computed in this arm:
  - L1 and L2 equal I06's arm 3 in `score/work/arms_seed{0,1,2}.jsonl`;
  - xy is within 1e-9 m;
  - `note.joined_waiting` is equal;
  - `det_i` equals `rows_oracle.jsonl` `det['i']`.
  - The R0 recount of line failures by family and level must equal section 0's 103, with no failure at groups or audiences.
- **The judged set J (S1, K2).**
  - J is the R0 L1 failures among line variants of A1, A2, A3, B1, C1 and D2: 12 + 12 + 15 + 8 + 3 + 4 = **54** (recounted from the rows; D5 has no line variant).
- **Measures, per rule.**
  - L1 and L2 success by family and level, and by situation (lines, groups, audiences) by family and level, laid out like I06's Tables 2.1–2.2.
  - Flips (R0 fail → success) and breaks (R0 success → fail) under L1, and also under L2. Each comes with its scene id, seed, row, family and level, paired by scene id within a seed.
  - Failure (i)'s firing rate overall and by family.
  - The paired drop against the scene's own base under the same rule, by family (descriptive).
  - The median distance from the place to the target, by family and level (descriptive).

## 4. Arm 1b: I06's noisy arm (secondary, not judged)

- **Scenes.** The 168 kept line scenes: rows eval_06, 07, 08, 09 and 13, which is 15 bases and 153 variants, seeds 0–2.
- **Models.** `qwen`, `glm` and `dsv`, from `noise_model.json['models']` (`check_i03` must be true, as in `noisy.py`) through `noise.Model`.
- **Draws.** d = 0…19, with `rng = noise.scene_rng(model, scene_id, d)` = `default_rng([20261018, {qwen 0, glm 1, dsv 2}, crc32(scene id), d])`. These are I06's seeds.
  - `found = noise.simulate(model, rng, truth, obs, th, roles)`, with `roles = {p.id: noise.role_of(p, truth.group_for("e0", "t0"))}`.
  - It is simulated once per draw, and the same `found` is passed to both rules (choice).
- **Per episode and rule (R0, RF90)**, exactly as in `noisy.py`:
  - `lib.solve_c(..., people_found=found)`, reusing e0's solve for e1 when the situation and group are the same;
  - L1, L2, and xy rounded to 4 dp.
  - `det_i` is computed with `lib.detectors`' own formula from the solve's `note.joined_waiting` and `source_of` (members from `truth.group_for("e0", "t0")`). The floor-based detectors are not run, to save time (choice).
- **R0 reproduction.** Every draw's e0 `{l1, l2, xy}` must equal `score/work/noisy_models_<model>_seed<s>.jsonl` `draws[d]['e0']`, its `det_i` must equal `draws[d]['det']['det_i']`, and its `agree` must be true. The total is 168 × 20 × 3 = 10,080 draws.
  - If all match, I06's seeding is "reproducible" and RF90 is compared with I06's R0.
  - If any differ, they are counted and listed, and RF90 is compared with this probe's own R0 on the same simulated people, flagged "I06's draws not reproduced" (choice).
- **Measures, per model.**
  - Mean L1 (and L2) over draws per scene, then over scenes, by family and level, under R0 and RF90, with the difference.
  - Per-draw flips and breaks pooled by family and level.
  - `det_i`'s rate under each rule.

## 5. Arm 2: development records (S2 and K1 in setting (a))

**Records.**
- One run per (stage, arm, model), method `con`, from `H.runs(...)`; exactly one must exist. Records are read with `H.records(ref)`, sorted by (scene, episode, trial).
- A goal step is a record with `Record.solver_step` true. That is the harness's definition, I01's "solver steps" and I14's "goal steps".
- Records that are not goal steps are left out of every count here. They cannot change under any rule (choice).
- The record key is (run arm, scene, episode, trial), as I01 and A2 key them.

Table 5.1. Cells, goal steps and R0 targets (counts from the row files named in section 0).

| Cell | Stage, model | Arms | Goal steps | Recorded success (a) | P (b) | D (c) |
|---|---|---|---|---|---|---|
| **Primary** | | | | | | |
| Qwen realistic | s19 `qwen38_low` | real, staffed, plain | 143 | 134 | 136 | 136 |
| GLM realistic | s19 `glm53` | real, staffed, plain | 141 | 96 | 112 | 112 |
| DeepSeek realistic | s20 `dsv41` | real, staffed, plain | 157 | 109 | 120 | 126 |
| Qwen watching | s19 `qwen38_low` | dev_seed0–2, audience goal steps | 36 | 32 | 33 | — |
| GLM watching | s19 `glm53` | dev_seed0–2, audience goal steps | 36 | 23 | 29 | — |
| DeepSeek watching | s20 `dsv41` | dev_seed0–2, audience goal steps | 36 | 23 | 31 | — |
| **Secondary (new wording)** | | | | | | |
| GLM new | s23 `glm53` | real, staffed, plain | 141 | 86 | 107 | 106 |
| DeepSeek new | s23 `dsv41` | real, staffed, plain | 151 | 106 | 116 | 119 |
| Haiku new | s23 `haiku55` | real, staffed, plain | 146 | 134 | 135 | 134 |
| Sonnet new | s23 `sonnet55_low` | real, staffed, plain | 143 | 134 | 134 | 134 |
| Gemini new | s23 `gemini38_low` | real, staffed, plain | 135 | 109 | 109 | 109 |

- These are the six primary cells of I14 (`phase1/README.md` section 4). The watching cells keep only audience goal steps, as `i14.py` does.
- The secondary runs are the design's list, with the realistic arms, as in I01, I14 and A2 (choice). s21 `qwen38_low` is not in the design's list, so it is left out (choice).

**Settings.** Every goal step is re-solved under R0, RF90, RF60 and RN in each setting that applies.
- **(a) The record's own inputs:** `H.resolve(rec)`, with the recorded target, people and constraints. All 11 cells.
- **(b) True people with the recorded constraints:** I01's P.
  - The people are the matched detections moved to the true position and yaw. Matching uses `sgi.run._matched_people` against the visible true people under the run's thresholds, at 0.5 m, one to one.
  - Unmatched detections are dropped.
  - Missed visible true people whose marks project are added as `OraclePeople` makes them, numbered max + 1, + 2, … in ground-truth order.
  - The constraints are `repairs.strip(recorded constraints, dropped numbers)`.
  - `phase1/i01_repairs/repairs.py` is imported unchanged, for `scene_info` and `strip`. The lines of `solver_row` that build `p_p` and `c_strip` are copied verbatim (choice: copied rather than calling `solver_row`, which runs 15 arms).
  - All 11 cells.
- **(c) The frozen I09 detector people with the recorded constraints:** A2's D.
  - `phase1c/i09/a2/replay_d.py` is imported unchanged, and its `verify_hashes()` must pass.
  - The detector people are `load_detector_people()[(str(ref.scenes_dir), rec.scene_id)]`, with `M` computed as in `arms_step`.
  - `people, cons = build(det, recorded constraints, dets, match_xy(det, dets), False, M, obs, th)`, then `H.resolve(rec, people=people, constraints=cons)`.
  - Realistic cells only: the 3 primary realistic cells and the 5 secondary cells. The design says "on the realistic records where A2 has them", so D is not computed for the watching cells (choice).
  - Matched people take the detector's yaw, or the record's own where the detector's is None. Added detector people whose yaw is None join as in R0.
- **Outcome.** `score.success` from `H.resolve`, the package scorer's official top-1 success, as in I01 and A2.

**R0 reproduction (step 1 of section 8)**, per goal step, record for record, before any repair rule runs on that setting:
- (a): `reproduced` is true (place within the harness's 1e-6 m), success equals the recorded success (`C_rec`), and, for realistic records, it equals I01's `C`.
- (b): success equals I01's `P` (`phase1/i01_repairs/rows_{old,new}_<model>.jsonl`) for realistic records, and A2's `P` (`phase1c/i09/a2/base_ph_<model>.jsonl`, computed by I01's unchanged `solver_row`) for watching records (choice).
- (c): success equals A2's `D`, and `joined_waiting` equals A2's `D_d['joined']` (`phase1c/i09/a2/rows_{old,new}_<model>.jsonl`).
- Any mismatch stops that setting. No repair outcome is computed there until a dated DEVIATIONS entry, written before any repair outcome of that setting exists, says what is done.
- A rule's solve that raises counts as a failure under that rule and is listed. An R0 solve that raises is a reproduction failure (choice).

**Measures, per cell, setting and rule.**
- n and the success count.
- Flips and breaks against R0 of the same setting (choice: (b) against P under R0, (c) against D under R0), paired by record key, and the net (flips − breaks).
- Every flip and break is listed with its run, arm, scene, layout (`layout_of`: the scene id without its `real_`/`staffed_`/`plain_` prefix), episode, trial and situation.
- Splits by situation.
- The number of goal steps in which R0 absorbed at least one person, and how many of those succeed under R0.
- **What RN costs ("how often absorption is needed"):** RN's breaks per cell and setting.
- For every person R0 absorbed (descriptive): whether their yaw is known, and the gate's f · u.

## 6. Pre-registered criteria (verbatim) and their readings

**Verbatim (lead; judged on point estimates):**
- S2 (no harm on real records): in setting (a), RF90 has at most 1 break in each of the six primary cells, and its net (flips - breaks) is >= 0 in every primary cell. The same is reported for (b) and (c), and for the new-wording runs, not judged.
- K1: RF90 has 3 or more breaks in any primary cell in setting (a), or a negative pooled net in (a).
- K2: RF90 fixes fewer than 25% of the judged I06 failures.
- Report RF60 and RN beside RF90 (not judged), and say what RN costs on real records (how often absorption is needed).

**Readings.**
- **S1**, on e0's L1 verdict. Both parts must hold.
  1. The number of scenes in J (n 54) that succeed under RF90 is at least 0.5 × |J|, which is 27 of 54.
  2. None of the 30 bases or 46 group and audience variants succeeds under R0 and fails under RF90.
  - Breaks at line variants outside J are reported beside S1, not judged in it, on the design's literal wording (choice). Those are breaks at A4–A6, at E, and at R0 successes at B–D lines.
  - S1 is judged on L1. L2 is reported.
- **K2** fires when the RF90 successes in J are fewer than 0.25 × |J|, that is 13 or fewer of 54.
- **S2** holds when, in setting (a), each of the six primary cells has RF90 breaks ≤ 1 and RF90 flips − breaks ≥ 0.
- **K1** fires when, in setting (a), any primary cell has 3 or more RF90 breaks, or the pooled net is negative. The pooled net is the sum over the six primary cells of (flips − breaks).
- **Verdict.**
  - Method candidate when S1 and S2 hold and neither K1 nor K2 fires; kill when K1 or K2 fires; neither otherwise.
  - S2 excludes K1 and S1 excludes K2, so the three verdicts cannot overlap.
  - If any R0 reproduction in arm 1 or in setting (a) fails and is not resolved under section 5's rule, no verdict is given.
  - `tables.py` computes the verdict from these readings.
- **Not judged.** RF60 and RN everywhere, settings (b) and (c), the new-wording runs, and arm 1b.

> RF90 fixes almost all A1-A3 failures by construction and some B-D failures; breaks on real records are rare because real waiting people mostly face the line's front, but VLM yaw errors (GLM within-45 about 0.55 for oblique/profile views) could turn real joiners away in setting (a).

**The protocol author's notes on the prediction.** These are not part of it and change no criterion. They are reasoned from the code and I06's family spec, not from any repair outcome.
1. **A3 sits on RF90's boundary.**
   - The A3 confuser stands on the line's axis behind the last member and faces 90 deg off the line's facing. On top of that come `_person`'s jitters: up to ±0.15 rad of yaw and ±0.05 m of position.
   - So f · u is about cos(90 deg ± 8.6 deg), between about −0.15 and 0.15 (a little wider with the position jitter), and the jitter decides the gate there.
   - A1 (f · u ≈ −1) and A2 (≈ −0.71) are clearly turned away. A4 (≈ 0.5), A5 (≈ 0.87) and A6 (≈ 1) are not, and A4 sits on RF60's boundary.
2. **The gate is not success.**
   - A confuser that is turned away still keeps its own space (`standoff_m` 1.0 m). At A1–A3 it stands about 0.9, 0.6 and 0.3 m from the place in turn.
   - So the robot cannot stop at the place in turn itself, and whether its stop lands in L1's region was not checked.
   - "By construction" holds for the gate's decision at A1–A2, not for L1 success.
3. **Only scenes and records with a joiner can change.**
   - Only scenes in which R0 absorbed someone can change (consequence 1). In I06 those are 99 scenes, all R0 failures, so arm 1 can show flips and no breaks.
   - On records, likewise, only goal steps where R0 absorbed someone can change.

## 8. Order of work, outputs, compute and guards

**Order.**
1. This note is hashed (`sha256sum FROZEN.md > FROZEN.sha256`) and made read-only before any repair rule exists.
2. `rules.py` is written, and the static check is run.
3. **R0 passes**, each over its whole source, with the copy check active:
   - arm 1;
   - arm 1b per model;
   - arm 2 per setting, over all of that setting's cells.

   Each writes `out/r0_<source>.txt`, holding the reproduction counts, every mismatch and the file time, and stops on a mismatch.
4. **Repair passes:** RF90, RF60 and RN in arms 1 and 2, and RF90 in arm 1b. They run only on sources whose R0 file shows full reproduction. Every scene, draw and record is solved under every rule; nothing is skipped by consequence 1, which is asserted instead.
5. `tables.py` writes `tables.txt` and `summary.json`, with the verdict computed by code.
6. `README.md` is the results page. Its writer runs no new arm, and any tally it makes is marked.

**Files** (choice), all under `phase2/i06_repair/`:
- `precheck/`;
- `rules.py`;
- `crowd.py` → `out/crowd_rows.jsonl`, one row per scene and rule, with both episodes inside;
- `crowd_noisy.py` → `out/noisy_<model>.jsonl`, one row per scene and rule, with the per-draw values;
- `records.py` → `out/records_<cell>.jsonl`, one row per goal step, holding every setting and rule. The cells are `old_qwen38_low`, `old_glm53`, `old_dsv41`, `watch_qwen38_low`, `watch_glm53`, `watch_dsv41`, `new_glm53`, `new_dsv41`, `new_haiku55`, `new_sonnet55_low` and `new_gemini38_low`;
- `out/r0_*.txt`, `tables.py`, `tables.txt`, `summary.json`, `logs/`, `run_*.sh`, `README.md`;
- `DEVIATIONS.md` and its hash, if needed.

**Compute.**
- CPU only, `nice -n 10`, `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=2`, and at most 2 worker processes.
- `free -m` must show at least 3,000 MB available before each heavy job (wait and poll, as in I06's `run_noisy.sh`). No `pkill` or `killall`.
- Use `.venv/bin/python` with `PYTHONPATH` empty.
- Expected cost (a guess from I06's logs): arm 1 under an hour, arm 1b 1–2 hours on 2 workers, arm 2 about an hour.

**Guards.**
- Zero model or network calls.
- Recorded runs and scenes are read only through the step-0 harness. The I06 scenes are loaded by `lib.load`, which uses `H.guard_scene`.
- Nothing is read from `keep/heldout`, any s18 or s20_held run, eval_16–19 scene data, or synth seeds 3–99 or 200–212. No group_photo or lift_wait row is instantiated.
- Writes go only under `phase2/i06_repair/`. Nothing in `src/`, `tests/`, `configs/`, `docs/`, `REPORT.md` or any earlier phase folder changes.

## 9. Cautions

- **Data-informed and not blind.**
  - The rule was chosen after I06 showed failure (i), and the S1 scenes were built to trigger it.
  - The protocol author has seen the R0 counts in Table 5.1 and section 0, but no joiner's facing.
- **Circular.** Oracle C reads the situation label and the true group (REPORT section 8). Arm 1 tests the gate under oracle inputs, not a method.
- **The records are development data**, already studied in I01, I14 and A2. The watching cells are small (n 36), and judgements rest on point estimates.
- **Yaw sources differ by setting.** In (a) yaw is the model's own, and it may be unknown; GLM is within 45 deg for 0.533 of oblique and 0.485 of profile views (I06 Table 4.1). In (b) yaw is true. In (c) it is the I09 detector's, falling back to the model's.
