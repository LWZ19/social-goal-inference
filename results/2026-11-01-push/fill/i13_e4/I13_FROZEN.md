# Item 13: E4, a bent waiting line and someone using a machine (frozen before any E4 model call; amendment 1 before scoring)

## 1. What is fixed by hash

- **The design.** `e4/E4_DESIGN.md`, sha256 `5ebf7fafbef4f974...` (full hash in `I13.sha256`).

## 2. Scenes, models, methods

- **Why 12 layouts:** the bar is a layout-level sign-flip. At 6 layouts its smallest possible p is 0.031, reached only
  if every layout favours C. On the development data, 5 layouts gave a floor of 0.0625, so even a clean sweep could
  not pass (`e4_score_test_out.txt`). At 12 layouts the floor is about 0.0005.
- **Models:** the six current ones (Qwen 3.8, GLM-5.3, DeepSeek V4.1, Sonnet 5.5 low, Haiku 5.5, Gemini 3.8).
- **Methods:** R (`cot`), E (`fsc`), C (`con`), with prompts and code unchanged. The configs are `e4/configs/*.yaml`
  (trials 3, final_test true).

`e4_geometry.py` scores four geometry baselines with sgi's own scorer and a perfect click:
- **G0, C's people-blind standard goal:** do people change the goal?
- **G1, local extension:** 0.85 m behind the last member along the last two members' direction.
- **G2, straight extension:** along the first leg's axis.
- **G3, 1.5 m straight behind the machine user.**

Rules, fixed now:
- **If G0 passes at least 50% of a family's episodes,** that family is reported as "people do not change the goal
  here". The C vs R comparison there is reported but not claimed as social reasoning.
- **If G2 passes at least 50% of the bent-line episodes,** "the bend is not tested". The bent-line result is reported
  as an ordinary waiting line, and no claim about bends is made.
- **If G1 or G3 fails on more than 10% of episodes,** the success area is blocked or unreachable in those scenes. The
  builder fixes them, logged, before any call, and the check is re-run.
- **Tested** on the development waiting lines (straight). G0 0/10 and G1 8/8 pass; G2 8/8, so "bend not tested", which
  is correct for straight lines. `--selftest` puts G3 inside the machine rule's area and outside its forbidden zones at
  four orientations.

## 4. Bars (judged once per family, after all calls; `e4_score.py`)

- **E4-R, primary per family.** Pooled over the six models (per-layout net of C minus R summed over models, episodes
  and trials), an exact two-sided sign-flip over the 12 layouts must give net > 0 and p ≤ 0.05.
  - If it holds: "C beats pointing on bent lines" or "on machine use".
- **E4-E.** The same with E in place of R.
- **If Sonnet runs on seeds 600-602 only** (see the cost), it is left out of the pooled bars and reported separately.
  The bars then pool the other five models over 12 layouts.

## 4b. Amendment 1 (7 Nov, before E4 is scored): records that keep failing

- **What happened.** GLM's E4 runs have vlm_error records: 31 in C and 1 in E, all the gateway's 180 s timeout. The
  other five models are clean.
- **The rule (E2's, as item 12's amendment 1).**
  - Passes continue until clean, or until two passes in a row clear none.
- **Records still in vlm_error after the last pass** are left out of every method for that model, keeping the
  comparison paired, and counted.
  - Over 10% of a model's records in a family means "partial" for that model.
- **Worst case.** `e4_score.py` also prints the pooled E4-R and E4-E with every left-out record counted as a C loss in
  its layout. If a bar holds only without them, that is stated.

## 5. Predictions (inference)

- **The validity check:** G0 fails both families (people matter). G1 and G3 pass. G2 fails on the bent lines, unless
  the bend's geometry lets a straight extension land within the half ring with the right heading.
- **Bent line:** E4-R holds. C follows the last member's direction, as the rule does, while pointing models often
  mark the counter or the first leg.

    cd .
    E=results/2026-11-01-push/e4
    F=results/2026-11-01-push/fill/i13_e4
    # 0. render seeds 603-605 with the builder's command for 600-602 (placeholder renderer, seconds a scene)
    # 1. validity check, zero calls (before any call):
    PYTHONPATH= .venv/bin/python $F/e4_geometry.py $E/scenes > $F/e4_geometry_out.txt
    # 2. runs, per model M (free models first; seeds in parallel as e3_models/run_held.sh):
    # 3. score (zero calls):
    PYTHONPATH= .venv/bin/python $F/e4_score.py runs/*_s26_e4_* > $F/e4_score_out.txt

## 7. Cost

- **Records:** 24 layouts x 2 episodes x 3 trials = 144 per method per model, 432 per model at about 2.8 calls each.
  That is about 1,200 calls per model, about 7,300 in all.
- **Free models:** about 3,600 calls.
  - Qwen and GLM are slow: seeds run in parallel, about 6-10 h of wall time per model at the gateway's pace.
- **Rendering:** 12 placeholder scenes, seconds each, on the CPU.

## 8. Scorers

- `e4_geometry.py`: the validity check; tested as in section 3.
- `e4_score.py`: the bars; tested on the development s21/s23 runs (`e4_score_test_out.txt`).
