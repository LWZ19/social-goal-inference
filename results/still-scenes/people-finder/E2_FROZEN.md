# E2: detector people against C's people call, on fresh static scenes (frozen 4 Nov 2026, before any scene exists)

## 1. Question

Does C with detector people (D) beat the current C (A0) on static scenes, in both speed and accuracy?

- E1b (`results/moving-lines/own-camera/detector-on-sim-frames/RESULTS.md`) found D level with or better than A0, with one call fewer.
- But that was on the development scenes, where the detector A1 was fitted on part of them. Its development check was
  not blind.
- This test repeats the comparison once, on scenes that do not exist yet, with everything fixed in advance.

## 2. Arms (paired: all in the same unit, on the same plan, target and context check)

A unit is one (scene, episode, trial) on one model. Per unit:
- The plan call runs once.
- The target is the plan's point, lifted as C lifts it.
- The context call runs once.

All of this is shared by the arms. If the model's context check says people do not matter, the goal is C's standard
goal for every arm, so the arms are identical by construction. Otherwise each arm runs its goal step:
- **A0, the current C:** the VLM people call, then the constraints call, at trial 50 + t.
- **A0b, a second draw of the current C:** the same two calls at trial 150 + t, so they are fresh draws (the cache
  key holds the trial).
  - It gives the redraw floor for accuracy (section 6), and a placebo for timing.
- **D:** I09 A1's frozen detector people, then the constraints call at trial 50 + t. There is no people call.
  - The constraints call sees the detector's numbered people.
  - This is the D-live / E1b people source (`DetectorPeople`), unchanged.

The goal step's inputs are the model's own context check, as in C. The arms run in a rotating order per unit (A0, D,
A0b; then D, A0b, A0; then A0b, A0, D), so drift in the gateway falls on all three alike.

**Detector setting: A1, as frozen** (rtdetr_v2_r50vd, tau 0.25, band wide, d_dup 0.5, d_push 0.2, F5 facing; served
by `results/moving-lines/own-camera/detector-on-sim-frames/e1d/detector_service.py`). Why A1 and not fix 2:
1. E1b's evidence is A1's. A confirmatory test runs the configuration that produced the evidence. Switching would test
   two changes at once.
3. On the development scenes fix 2 differs from A1 by only 2 missed staff and 3 false people (`E1_FROZEN.md`).

Fix 2 is not used anywhere in E2.

## 3. Scenes (built by the run; logged in `E2_DEVIATIONS.md` before any model call)
- **Rows: the 8 people rows of the eval split.**
  - `eval_06_waiting_line_1`, `eval_07_waiting_line_2`, `eval_08_waiting_line_4`, `eval_09_waiting_line_6`,
    `eval_13_waiting_line_3`;
  - `eval_10_conversation_group_2`, `eval_11_conversation_group_3`, `eval_12_conversation_group_4`.
  - The 6 no_one rows are left out: C makes no people call there, so the arms are identical and cannot inform the
    comparison. On the development sets, exactly the 8 people rows were the people-matter records (48 of 84 per run).
- **Layouts:** 5 seeds × 8 rows = 40 layouts, each with the generator's 2 episodes, so 80 scene-episodes.
- **Render, per seed N** (from `.-scenes`):

      PYTHONPATH= .venv/bin/python -m sgi.synth.realistic.generate --variant real --seed N \
          --out <E2>/renders/seed<N> --only eval_06_waiting_line_1 eval_07_waiting_line_2 eval_08_waiting_line_4 \
          eval_09_waiting_line_6 eval_10_conversation_group_2 eval_11_conversation_group_3 \
          eval_12_conversation_group_4 eval_13_waiting_line_3

  - At about 10 images a scene, 40 scenes is about 50 min, plus any spare seed.
- **Assemble** (`e2_assemble.py`; no GPU, zero calls):
  - copies each used seed's 8 scenes to `run_scenes/s<N>_real_eval_<row>`, with scene_id rewritten to that name;
  - writes `manifest.csv`: seed, row, scene_id, and sha256 of `rgb.png` and `depth.npy`.
- **Detector pass** (`e2_detect.py`; GPU about 60 ms an image, zero calls):
  - A1's people for every scene, written to `e2_people.json` in `eval_people.json`'s format, with each image's
    detection and pose times.
  - It is the same path as the person-snap test's `snap_detect.py`.
- **Logged before any model call:** the sha256 of `manifest.csv` and of `e2_people.json`. The driver refuses a
  people file whose sha256 differs from the one it is given.

## 4. Calls (the run runs them)
- **Models, free:** Qwen 3.8 low (`soclaas_qwen38_low`), GLM-5.3 (`soclaas_glm53`), DeepSeek V4.1
  (`soclaas_dsv41`).
  - Config: `sgi.serve.make_config(model)`, the current C, the same as E1b. No worked examples, because C's goal step
    takes none.
- **Trials:** 2 per scene-episode (t = 0, 1), so 160 units a model.
- **Calls per unit:**
  - plan 1 + context 1, then on people-matter units A0 2 + A0b 2 + D 1, which is at most 7.
  - At most 1,120 a model and 3,360 in all, free. The development share of people-matter units in these rows is
    about 0.95, so about 1,080 a model is expected.
- **Running:**
  - Units are written whole, one line each. `--resume` skips units already written, and re-runs any written with a
    `vlm_error`.
- **Code freeze.**
  - The driver writes `run_meta.json`: git HEAD, the sha256 of `git diff HEAD -- src prompts`, the config, the
    people file's sha256, and the arguments.
  - The scorer refuses shards of one model whose meta differ.
- **Dry run:** `--dry` uses the mock back end with the cache off. Zero calls.

## 5. Measures (`e2_score.py`)

Per model:
- **Units, and people-matter units.** A people-matter unit is one where the model's own context check says people
  matter. It is the same for all arms.
- **Success** (sgi.evaluate.score, as C's records are scored) of A0, A0b and D: on people-matter units, and on all
  units.
- **Redraw floor:** right→wrong (rw) and wrong→right (wr) between A0 and A0b on people-matter units.
- **D against A0:** gain, harm and net.
  - An exact two-sided sign-flip test over layouts on the per-layout net.
  - The same over the 8 rows (the net summed over seeds), reported for robustness.
- **Speed:**
  - calls per people-matter unit, per arm;
  - goal-step model seconds per people-matter unit: A0 and A0b are the people plus constraints call latencies; D is
    the constraints latency plus the detector's warm time per image.
    - The warm time is the median of detection plus pose over the detection pass, without its first image, which is
      cold.
  - the whole unit's seconds (plan + context + goal step), reported.
  - **Latency** is each call's `latency_s`: the provider's time for the answer that counted. The client's pacing wait
    is not in it, so the shared 85 a minute does not enter. Validation and transient retries are logged per call and
    reported per arm; a retry's earlier attempt is not in `latency_s`.
- **Statuses.**
  - A unit with a `vlm_error` in any shared or arm call after the last resume is left out of every arm and counted.
  - `parse_failed` is the model's own answer and counts as a failure.
- **Detector check** (zero calls, reported, not a bar): A1's recall of visible truth people, matched one-to-one within
  0.5 m, split by staff and others, and false people per image on these scenes.

## 6. Bars (judged once per free model, after all calls)

- **A, accuracy, non-inferior within the redraw floor:** harm − gain ≤ rw on the people-matter units.
  - This is E1b's bar, with the fresh second draw A0b in place of the recorded C.
- **G, guard:** the layout sign-flip test of D against A0 must not show D worse (fails when p ≤ 0.05 and the net is
  below 0).
- **S1, calls:** D makes no people call on any people-matter unit. Wherever A0's people call is ok, D's goal step
  makes one call fewer than A0's.
- **S2, time:**
  - On people-matter units where all three arms are ok, D's goal-step seconds are below A0's in more units than not.
    The exact two-sided sign test must give p ≤ 0.05.
  - **Placebo:** the same test of A0b against A0 must give p > 0.05. If it does not, the gateway's timing is not
    readable for that model, and S2 is reported as "not judgeable" there.
- **The claim:**
  - "On fresh static scenes D beats C, faster at no loss of accuracy" is made only if A, G, S1 and S2 hold for each of
    the three free models.
  - If a model's S2 is not judgeable, the claim there is for calls only, and says so.
- **H2, more accurate (secondary, judged only if the claim holds):** the layout sign-flip test of D against A0 gives
  p ≤ 0.05 with net above 0.
  - It is judged per model, and pooled over the three free models (per-layout net summed over models).
  - "Faster and more accurate" is said only with H2.
- **Sonnet:** the same numbers, reported; no bar.

## 7. Predictions (inference, written before any scene)

- **A and G hold for all three free models.**
  - D is level with A0 for Qwen and DeepSeek, and level or better for GLM, as in E1b.
  - H2 is not shown at this size.
- **S1 and S2 hold.**
  - The goal step is about 20 s faster per people-matter unit for Qwen, and 3-6 s for DeepSeek.
  - GLM's gateway may fail the placebo, as E1b found its times too noisy to read.
- **Detector check:**
  - false people about 0.2-0.3 per image, as A1's unseen-layout 0.242;
  - recall of visible non-staff people at least 0.95;
  - counter staff missed more often.
- **Risk:** in waiting lines a false person can enter the constraints call's line and move the goal. That is where
  harm is expected, if anywhere.

## 8. What this does not cover

- **Layouts within a row share structure.** New seeds of a row repeat much of its geometry. So the layout-level test
  generalises over the generator's seeds of 8 situation rows, not over new situation templates. The row-level test is
  reported beside it.
- Development situation types only, the "real" variant only, and static scenes only.
- No held-out situation, and no real photographs.
- The detector's time is measured on the shared RTX 4060. A robot's own GPU would differ.

## 9. Code, and what was checked before freezing (zero calls)

- **Scripts:**
  - `e2_assemble.py`: seeds in order, the 8 rows, the scene check, the copy with a unique scene_id, and the manifest.
  - `e2_detect.py`: `snap_detect.py`'s A1 path with arguments, and each image's order kept.
  - `e2_driver.py`: the units, shared steps, arms, rotation, shards, resume and meta.
  - `e2_score.py`: section 5's measures and section 6's bars.
- **The driver's shared steps are sgi's own.**
  - `Pipeline.plan` and `Pipeline.infer_goal` run the plan, the lift and the context call.
  - A stand-in method (`Capture`) receives exactly what `infer_goal` hands a method.
  - Each arm's method then runs on it, and its result is merged as `infer_goal`'s last lines do.
  - The plan call is added to each result as `sgi.run.run_episode` adds it, and scoring is `sgi.evaluate.score`.
- **Dry run** with the mock back end, cache off, on the spent person-snap scenes (`results/still-scenes/group-of-four-case/snapset/run_scenes_v2`) and
  their A1 people file. Checked:
  - two shards split the units;
  - the order rotates;
  - A0 and A0b make people + constraints calls, and D makes constraints only, with the detector's people;
  - resume skips finished units;
  - a wrong people-file sha256 and a changed run meta are refused.
  - The scorer ran on the dry records end to end. Its sign-flip and sign tests were checked against hand values.
- **`e2_assemble.py`** was tested on a fake render folder: a complete seed was used and an incomplete one passed
  over. The assembled scenes load through `sgi.run._load_scenes`.
- **Not run before freezing:** `e2_detect.py`, which needs the detector service on the GPU. It is `snap_detect.py`,
  which ran for the person-snap test, with the scene folder and output as arguments.
- **Order of work:**
  1. CT3FT finishes.
  3. Render, one Blender at a time, on the shared GPU.
  4. Assemble.
  5. Detector pass.
  6. Log the manifest and people hashes.
  7. A smoke of 2 units per free model, then the full run.
