# Extra X1: the counter point and the ceilings on four models' answers (frozen before any X1 episode)

## Question

On DeepSeek's answers, CT3F given the true counter point (CT3FT) was +0.053 over CT3F, with lower bound +0.013, but
only 27 layouts better against 23 worse (`round7/RESULTS_4.md`). Does that hold on Sonnet's, Haiku's and Gemini's
answers, and pooled?

This decides the order of the two future-work leads: the counter point (lead 1) vs who waits (lead 2).

## Arms, models, layouts

- **Stage A (the reading):** CT3FT and CT3F, re-run together in one invocation (cache-only) into
  `results/data/moving/runs_x1/<model>/`, so both arms share the machine's load (the item 10 lesson).
  - Models: DeepSeek (re-run too, for the same reason), Sonnet 5.5, Haiku 5.5, Gemini 3.8.
  - The 72 confirmation layouts (`sweep/jobs_confirm.txt`), each model's own cached CT2 first answers
    (`runs_confirm4/<model>`). 576 episodes.
- **Stage B (reported, if the simulator has room):** OW, CT4OT, CT4O, CT4PJT and CT4PJ in the same folders.
  - `round7/c_gaps.py` then gives ROUND7's full decomposition per model (place and follow, counter point, who).
  - 1,440 more episodes.

## Readings (judged once after stage A; strict share from the first service at band top 1.2 m)

- **X1-consistent:** per-layout mean of (CT3FT − CT3F) over the four models; the sign test over layouts (ties
  |mean| < 0.01) gives wins > losses and p ≤ 0.05.
- **X1-size:** the pooled mean is at least 0.05 (ROUND7_D's bar), with a paired-t lower bound above 0.
- **Outcomes:**
  - Both hold: the true counter point lifts the best arm consistently. Lead 1 (the observed serving spot) goes first,
    as ROUND7_D's outcome A or B.
  - Neither holds: the counter point alone does not lift CT3F, and who and target are one design question.
- **Per model:** the same numbers, reported.
- **The 1.5 m band top** is not repeated here; ROUND7_B showed it moves levels by 0.02-0.03.

## Predictions (inference)

- On the other models CT3FT − CT3F is positive, about 0.03-0.07.
- X1-size holds.
- X1-consistent is borderline: the counter point matters only where the model's target is off, which varies by model
  (Gemini's targets were worst on still scenes).

    # stage B (reported), per model, then round7's decomposition:
    #   a <entry> <dir> "OW CT4OT CT4O CT4PJT CT4PJ" B
    #   cd results/moving-lines/design-and-scoring/ideas2_1b && $P round7/c_gaps.py sweep/confirm_selection.json runs_x1/<dir> runs_x1/<dir>

## Cost

- **Calls:** zero.
- **Stage A:** 576 episodes at about 96 s, so about 7.7 h at 2 workers or about 3.9 h at 4.
- **Stage B:** 1,440 more episodes, about 19 h at 2 workers. Run it only if the simulator is otherwise idle.

## Mechanics

- Each episode's first answer is the model's cached CT2 answer (the same pose), with no `vlm_error`.
- CT3FT gets the true counter point (`target_xy`) as before (`check_ct3ft.py`).

## Scorer

`x1_score.py`, reusing round7's `c_gaps.load`. Tested with `--test` on the earlier DeepSeek pair, it reproduces
`e_ct3ft_out.txt`: +0.053, lower bound +0.013, 27/23/16, p 0.672.
