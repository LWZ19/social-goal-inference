# Layout-level tests of the method comparison

**Question.** Is the constraint method's (C) lead over pointing (R) and pointing with examples (E) still there when
the layout, not the try, is the unit of the test (14 development layouts, 18 paired records each)?

**Result.** New wording: C−R and C−E are positive for all six models, p 0.0039-0.0156 (each at the floor for its count
of untied layouts); no E−R is significant. Old wording: C−R p 0.0049-0.0078; C−E Qwen 0.031, GLM 0.0078, DeepSeek
0.00098. Seven older models (descriptive): C ahead on six, not on Sonnet 4.6 (C−R −4, p 0.875). Every total and all
15 earlier layout-level tests reproduce.

**Files.**
- `layout_tests.py`: the test. It reads finished runs (`results.jsonl`, `config.yaml`) and makes no model call.
- `layout_tests_out.txt`: its output.
- The definitions, runs and checks are below.

**Reproduce.** From the repository root:
`PYTHONPATH=src python results/still-scenes/method-comparison/layout-level-tests/layout_tests.py` (about 3 s). It
writes `layout_tests_out.txt` beside itself. It reads the run list
`results/still-scenes/analysis-scripts/harness/research_runs.txt`, the runs in `runs/`, and, for its last check, the
full output of the earlier statistics run. The copy in
`results/still-scenes/metrics-and-statistics/stats-rerun/stats_output.txt` holds only that file's last 38 lines, so
the last check stops with a KeyError; with the full file the output is reproduced exactly.

**Data.** `results/logs/still-runs.tar.gz` (the runs in `runs/`).

## Notes

`layout_tests.py` reads finished runs and writes `layout_tests_out.txt`. It makes no model or provider call and
imports no model or pipeline code.

### Definitions

- **Methods.** R = the model marks the goal point (`cot` runs). E = R with worked examples (`fsc` runs). C = the
  constraint method (`con` runs).
- **Records.** Each method has 3 variants (real, staffed, plain) x 14 layouts x 2 episodes x 3 trials = 252
  records. Success is `score.success` from `results.jsonl`.
- **Layout.** A layout is the scene id with its variant prefix removed. For example, `real_eval_07_waiting_line_2`,
  `staffed_eval_07_waiting_line_2` and `plain_eval_07_waiting_line_2` are all layout `eval_07_waiting_line_2`. There
  are 14 layouts, with 18 records each per method.
- **Pairing.** Records are paired across methods by (variant, scene, episode, trial). The script stops unless all
  three methods have the same 252 keys.
- **Net.** For a layout, net(A−B) = sum over its 18 paired records of success(A) − success(B).
- **W/L/T.** The number of layouts with net > 0, net < 0 and net = 0.
- **Test.** An exact two-sided sign-flip test with the layout as the unit. Let k be the number of layouts with a
  non-zero net. Then p = #{sign patterns s : |Σ s_l·|net_l|| ≥ |Σ net_l|} / 2^k.
  - The script counts p exactly by dynamic programming over integer sums. It also checks the count by listing all
    2^k patterns.
  - Layouts with net 0 are dropped. Keeping them gives the same p.
  - The output prints p both as a decimal and as hits/2^k.
- **Smallest attainable p.** With k non-zero layouts the smallest possible p is 2/2^k, or 1 when k ≤ 1. That
  happens when every non-zero layout has the same sign.
- **Held-out guard.** These records are skipped and counted:
  - records of a held-out situation: `group_photo`, `lift_wait`, `machine_use`, plus any list the run's config
    records;
  - records of scenes eval_16 to eval_19.

### Runs

- **Item 1, new wording.** Qwen 3.8 low uses `s21_{variant}_qwen38_low_{cot,fsc,con}`. GLM-5.3, DeepSeek V4.1,
  Haiku 5.5, Sonnet 5.5 low and Gemini 3.8 low use `s23_{variant}_<short>_{cot,fsc,con}`.
- **Item 1, old wording.** Qwen and GLM take R and E from s16b and C from s19. DeepSeek takes R, E and C from s20,
  never s20_held. These are the runs of the earlier `results/still-scenes/metrics-and-statistics/stats-rerun/stats.py`.
- **Item 1 source.** Every item 1 run comes from
  `results/still-scenes/analysis-scripts/harness/research_runs.txt`. The script checks each run's recorded scenes
  folder against the whitelist.
- **Item 2.** These are E3 runs `s24_{variant}_<entry>_{cot,fsc,con}`, the names `results/still-scenes/hidden-situations-and-more-models/run_static.sh` gives,
  for seven entries: gemini3_flash, gemini31_flash_lite, gemini35_flash_lite, gemini36_flash, claude_haiku45,
  claude_sonnet46_low and claude_sonnet5_low.
  - These runs come after the frozen whitelist, so they are not on it. Each name has exactly one folder.
  - The script checks each run's scenes folder against `results/still-scenes/hidden-situations-and-more-models/scenes_dirs.yaml`.
  - The script never opens held runs or `held_configs/`.

### Checks (all pass; see the end of the output)

- **Item 1 totals.** R/E/C totals match the earlier report (`EVIDENCE_AUDIT.md` A1 and A2,
  `docs/results/README.md`) for all 9 model x wording rows.
- **Item 2 totals.** R/E/C totals match `results/still-scenes/hidden-situations-and-more-models/results_all_out.txt` for the 7 older models. The 3 current
  models in that file also match their s23 runs.
- **Earlier stats.** All 15 layout-level tests of the earlier `results/still-scenes/metrics-and-statistics/stats-rerun/stats_output.txt` are reproduced: old
  wording C−E for 3 models, and new wording C−E and C−R for 6 models. The C and X totals, the two-sided p and all 14
  per-layout nets match.
- **New tests.** Old-wording C−R and E−R, and new-wording E−R, are new here.
