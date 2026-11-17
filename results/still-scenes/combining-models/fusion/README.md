# Fusion of trials and models

**Question.** How many episodes are solved by at least one, a majority, or all of several trials or models?

**Result.** Old wording, 84 realistic episodes: GLM total 186/252 (any 78, majority 64, all 44); DeepSeek 182/252 (any 79, majority 65, all 38); the 6 samples of both reach any 83, at least 3 in 78, all 27. Rescoring the recorded runs now gives the same counts.

**Files.**
- `fusion.py`: the counts, ports of three earlier analysis scripts (`../../analysis-scripts/`).
- `fusion_out.txt`: its output.
- `verify/`: independent re-checks `v1_list.py` ... `v11_baseline.py` and their `*_out.txt`.

**Reproduce.** `PYTHONPATH=src python results/still-scenes/combining-models/fusion/fusion.py` (about 4 minutes on an idle machine). It reads runs through the step-0 harness (`results/still-scenes/analysis-scripts/harness/`) and `data/dev_scenes/`. Re-run here, `fusion.py` and all `verify/*.py` give the stored counts; only the printed elapsed seconds differ. `verify/v3_fusion_out.txt` is not in this repository but is reproduced by `verify/v3_fusion.py`.

**Data.** `results/logs/still-runs.tar.gz` and `data/dev_scenes/` (release dev-scenes-v1).