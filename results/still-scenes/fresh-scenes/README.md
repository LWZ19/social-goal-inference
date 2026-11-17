# Fresh scenes

**Question.** Does the constraint method (C) still beat pointing (R) and pointing with worked examples (E) on 70 new
layouts (new seeds 510-514 of the 14 development rows) that no prompt was chosen on?

**Result.** Yes, on all six models: per model C−R and C−E are positive with p ≤ 0.021, and pooled over the six models
C wins 42 layouts, loses none and ties 28, both against R (net +424) and against E (net +361), p < 0.0001. For
example Haiku 5.5 gets 151 / 156 / 260 of 280 (R / E / C), Sonnet 5.5 198 / 194 / 257 and Qwen 3.8 191 / 227 / 246.
GLM-5.3 lost 20 records to failed calls; counting each as a C loss, its C−R still holds (p 0.041), its C−E does not.

**Files.**
- `I12_FROZEN.md`: the plan and the bars, written before any scene existed.
- `i12_score.py`: the scorer. `i12_score_six_out.txt`: all six models. `i12_score_five_out.txt`: the five models
  without GLM, scored first. `i12_score_test_out.txt`: a check on the development runs (s21/s23, real variant).
- `RESULTS_I12_INTERIM.md`: the write-up of the five-model result.
- `i12_assemble.py`: builds the run scenes and `manifest.csv` from the renders, with `assemble_log.txt`
  (`i12_assemble.diff`: its differences from `../people-finder/e2_assemble.py`). `manifest_full.csv` is the same
  manifest. Seeds 510-511 were assembled and run first (`run_scenes/`, `manifest_part1_510_511.csv`,
  `assemble_log_part1.txt`); the full set is `run_scenes_b/` with its seeds 510-511 in `run_scenes_b_already_run/`.
- `configs/`: the three method configs, with their differences from the earlier runs' configs (`*.diff`).
- `render_i12.sh`, `render_when_gpu_free.sh`, `run_i12.sh`: how the scenes were rendered and the runs started.
- `renders/`: the render jobs and scene files.

**Reproduce.** From the repository root (each prints the stored output exactly):

```bash
PYTHONPATH=src python results/still-scenes/fresh-scenes/i12_score.py runs/*_s27_i12_*                  # six models
PYTHONPATH=src python results/still-scenes/fresh-scenes/i12_score.py $(ls -d runs/*_s27_i12_* | grep -v glm53)
PYTHONPATH=src python results/still-scenes/fresh-scenes/i12_score.py runs/*_s21_real_qwen38_low_* runs/*_s23_real_*
```

`i12_assemble.py --renders results/still-scenes/fresh-scenes/renders --out <new folder>` needs the rendered pictures
and depth images, which are not in the repository.

**Data.** `results/logs/still-runs.tar.gz` (the runs in `runs/`); `results/logs/research-data.tar.gz` (the scene
files in `renders/` and `run_scenes*/`).
