# Item 12: the still-scene headline (C vs R vs E) on fresh layouts of the development rows (frozen before any scene exists; amendment 1 before any total)

## Question

Does C beat R (the model marks the goal) and E (R with worked examples) on layouts no prompt was chosen on?

Every still-scene method result except E2 is on the same 14 development layouts, where the context prompt was chosen.
There C−R and C−E are positive for all six models at the layout level (`results/still-scenes/method-comparison/layout-level-tests/layout_tests_out.txt`).

## Scenes (the run renders; the analysis never opens them before scoring)
- **Rows:** the 14 first-pass eval rows:
  - eval_00 to eval_02 (no_one place, at 0°, 30° with a bystander, and 60°), eval_03 (deep place), eval_04 (object),
    eval_05 (person);
  - eval_06 to eval_09 and eval_13 (waiting lines of 1, 2, 4, 6 and 3);
  - eval_10 to eval_12 (conversation groups of 2, 3 and 4).
  - The held-out situations cannot be made by this generator.
- **Assembly:** `i12_assemble.py` (`e2_assemble.py` with these rows, seeds and one folder per seed; diff
  `i12_assemble.diff`). It writes `run_scenes/seed<N>/s<N>_real_<row>` with unique scene ids, plus `manifest.csv`.
- **Logged before any model call:** the sha256 of `manifest.csv`.
- **Storage:** everything stays under `.superpowers/`, which git ignores, so rendered sets never enter git.

## Runs

- **Methods:** R (`cot`), E (`fsc`) and C (`con`), with the s23 settings of E3's real-variant configs. The only change
  is 2 trials in place of 3 (`configs/*.yaml`; diffs in this note's hash set).
  - E's worked examples are the development examples, as in s23.
  - R runs first; E and C reuse its plan and context answers through the cache, as `results/still-scenes/hidden-situations-and-more-models/run_static.sh`.
- **Models:**
  - free: Qwen 3.8 low, GLM-5.3, DeepSeek V4.1;
- **Records:** 70 layouts x 2 episodes x 2 trials = 280 per method per model.

## Bars (judged once per model, after all calls; `i12_score.py`)

- **Per model:**
  - **I12-R:** C − R per-layout net (summed over episodes and trials), exact two-sided sign-flip over the 70 layouts,
    net > 0 and p ≤ 0.05.
  - **I12-E:** the same for C − E.
  - If both hold for a model: "C beats R and E on fresh layouts of the development rows" for that model.
- **Pooled over the six models, and over the three free models:** the same two tests, reported beside the per-model
  bars.
- **Reported:** E − R; success by situation; the comparison with the development numbers.

## Amendment 1 (7 Nov, before any item 12 total): records that keep failing

- **What happened.** GLM's C runs have 53 vlm_error records, about 10 per run: the gateway's 180 s request timeout.
  R and E are clean. The other five models have none.
- **The rule (E2's, item 19).**
  - Passes continue until the records are clean, or until two passes in a row clear none.
- **Records still in vlm_error after the last pass** are left out of every method for that model, so the comparison
  stays paired, and counted.
  - If more than 10% of a model's records are left out, its result is reported as "partial".
- **Worst case.** The scorer also prints I12-R and I12-E with every left-out record counted as a C loss. If a bar
  holds only without them, that is stated.
- **The 180 s timeout is unchanged** (superseded by amendment 1b below).

## Predictions (inference)

- I12-R and I12-E hold for every model. The development margins were +32 to +92 per 252.
- The fresh margins may be somewhat smaller, as the context prompt was chosen on the development layouts.
- E − R is near zero, with no significant layout-level difference, as on the new-wording development runs.
- GLM's C − R is the weakest. On the 14 real-variant development layouts it is p 0.125 (`i12_score_test_out.txt`); 70
  layouts give far more power.

## What this does not cover

These are new seeds of the same 14 templates, not new templates written blind. The result generalises over the
generator's seeds of the development rows.

## Run (run)
    # 1. render (scenes folder; one Blender at a time; about 67 s a scene, about 16 min a seed, about 80 min for 5 seeds)
    cd .-scenes
    I=results/still-scenes/fresh-scenes
    ROWS="eval_00_no_one_place eval_01_no_one_place eval_02_no_one_place eval_03_no_one_deep_place eval_04_no_one_object eval_05_no_one_person eval_06_waiting_line_1 eval_07_waiting_line_2 eval_08_waiting_line_4 eval_09_waiting_line_6 eval_10_conversation_group_2 eval_11_conversation_group_3 eval_12_conversation_group_4 eval_13_waiting_line_3"
    for N in 510 511 512 513 514; do
      PYTHONPATH= .venv/bin/python -m sgi.synth.realistic.generate --variant real --seed $N --out $I/renders/seed$N --only $ROWS > $I/render_seed$N.log 2>&1; echo "$(date +%H:%M) seed $N exit $?"
    done
    # (render 515-519 in order only if the assembly passes a seed over)
    # 2. assemble (zero calls, no GPU), then log manifest.csv's sha256 before any call
    cd .
    PYTHONPATH= .venv/bin/python $I/i12_assemble.py --renders $I/renders --out $I/run_scenes
    sha256sum $I/manifest.csv
    # 3. runs, per model M (each seed folder in parallel; R first, then E and C)
    # 4. score (zero calls)
    PYTHONPATH= .venv/bin/python $I/i12_score.py runs/*_s27_i12_* > $I/i12_score_out.txt

## Cost

- **Calls:** 280 records x 3 methods x about 2.8 calls, so about 2,350 per model.
  - Free: about 7,000 calls. Wall time: about 3 h for Qwen and about 6 h for GLM, with the 5 seed folders in parallel.
- **Rendering:** about 80 min of Blender on the shared GPU, competing with the simulator.

## Scorer and tests

- `i12_score.py` (per model and pooled sign-flips; success by situation). It refuses duplicate records, so one model
  and method must not cover a scene twice.
  - Tested on the development s21/s23 real-variant runs (`i12_score_test_out.txt`): the totals match those runs
    (e.g. Qwen R/E/C 62/65/79 of 84).
- `i12_assemble.py`: tested on a fake render folder (a complete seed used, the rest reported as not rendered).
