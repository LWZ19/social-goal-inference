# Still scenes: the October report and its re-check

**Question.** For still scenes, what could improve on a pipeline that leaves everything to the vision-language model,
and which of those ideas hold up against the recorded development runs?

**Result.**
- For the weaker models, most remaining loss is perception. For example, 56 of GLM's 74 missed line members were never
  found by the people call. A perception repair from exact depth fixes 15 of GLM's 29 line failures under the
  constraint method.
- Counted with the layout as the unit, the constraint method beats worked examples for all three models: exact
  sign-flip p 0.031 (Qwen), 0.0078 (GLM) and 0.00098 (DeepSeek).
- The re-check confirmed 53 of the report's 76 claims and corrected 19. It refuted none, and 3 could not be
  reproduced.

**Files.**
- `REPORT.md`: the report. A literature landscape, about fifty scored ideas, and the first zero-call findings.
- `RERUN.md`: the ledger that recomputes every number in the report, with the final verdict per claim.
- `phase-notes/phase1.md`, `phase1b.md`, `phase1c.md`, `phase1d.md`: results of each later batch of zero-call
  analyses. Each analysis's plan, scripts and outputs are in its own topic folder under `results/still-scenes/`.

**Reproduce.**
- These are notes; the scripts are in the topic folders.
- Most of those scripts read the runs and scenes through the step-0 harness, which is in
  `results/still-scenes/analysis-scripts/harness/`.
- The development scene sets go in `data/dev_scenes/` (`realistic_final/{plain,real,staffed}`, `dev_new/seed0-2`,
  `placeholder_first14`; release dev-scenes-v1).
- Scripts that check the sha256 of their frozen files stop at that check here: the pins refer to the original files.

**Data.** None for this folder. The topic folders use `results/logs/still-runs.tar.gz` (the recorded runs in `runs/`)
and `results/logs/research-data.tar.gz`.
