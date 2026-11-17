# Hidden situations and more models

**Question.** Does the constraint method's lead hold on seven older Claude and Gemini models, and on the two
situations kept hidden during development (waiting for a lift, keeping out of a photo), and do Haiku 5.5 and
Gemini 3.8 follow the other models in moving lines?

**Result.** On the still scenes (of 252, pointing / with examples / constraint method) the method leads on nine of ten
Claude and Gemini models, e.g. Gemini 3.5 Flash Lite 102 / 101 / 215 and Sonnet 5 139 / 144 / 194; Sonnet 4.6 is
level (166 / 162 / 162). At the lift (of 36) it is best for Sonnet 5.5 (36), Haiku 5.5 (32) and Gemini 3.8 (28); the
photo stays hard for every method (0-9 of 36). In moving lines, re-solving (`CT2`) beats deciding once (`R1`) for
Haiku 5.5 (0.30 against 0.01 of the time in the right spot, 19 better, 2 worse) and Gemini 3.8 (0.21 against 0.09).

**Files.**
- `PLAN.md`: the plan and the running log of results and diagnoses (Sonnet 4.6, Haiku 4.5).
- `results_all_out.txt`: all still-scene and hidden-situation totals. `still_gemini_out.txt`: the still-scene table
  of the older Gemini models beside the current models.
- `moving_headline_haiku_out.txt`, `moving_headline_gemini_out.txt`: the moving-line headline (strict share, seeds
  10-14, five scenarios).
- `confirm_score_haiku55_out.txt`, `confirm_score_gemini_out.txt`: the final moving-line test, scored by
  `results/moving-lines/best-version-and-running-guess/round4/o_confirm_score.py`.
- `calib_*.txt`, `calib_config.yaml`, `smoke_with_config.py`: the point-format calibration of each older model
  (`sgi.smoke --calibrate`).
- `configs/`, `scenes_dirs.yaml`: the still-scene configs and scene sets; `held_configs/`, `held_scenes_dirs.yaml`:
  the hidden-situation ones. `new_entries.yaml`: the older models' entries.
- `run_static.sh`, `run_held.sh`, `queue.sh`, `sim_track.sh`, `sim_model.sh`, `sim_headline.sh`: how the runs were
  started.

**Reproduce.** The still-scene totals of the older models are re-counted, run by run, by
`results/still-scenes/method-comparison/layout-level-tests/layout_tests.py`. The final moving-line test, from the
repository root:

```bash
PYTHONPATH=src python results/moving-lines/best-version-and-running-guess/round4/o_confirm_score.py \
    runs_confirm4/claude_haiku55 test-layouts/confirm_selection.json "Haiku 5.5 (E3, descriptive)"
```

(and `runs_confirm4/gemini_fast` with `"Gemini 3.8 Flash (E3, descriptive)"`). Both reproduce their outputs; only the
selection path in the first line changed (it was `sweep/`). The other tables were printed by short commands that were
not kept.

**Data.** `results/logs/still-runs.tar.gz` (the runs `runs/*_s24_*` and `runs/*_s25_held_*`);
`results/logs/moving-runs.tar.gz` (`runs/`, `runs_confirm4/` in `results/data/moving/`).
