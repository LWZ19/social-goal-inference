# Target found versus placement

**Question.** Of the failures, how many are wrong targets and how many are placement errors given the right target?

**Result.** For Qwen with the marked-point method, the target is found in 107 of 108 nobody-near records and 87 of 90 line records, but success given the right target is 50.6% at lines (44 of 87). So at lines the loss is placement, not target finding.

**Files.**
- `split.py / split_out.txt`: the split tables per model, method and situation
- `tally.py / tally_out.txt`: tallies of goal steps
- `inspect_split.py, inspect2.py and their `_out.txt``: record-level listings
- `explore.py`: first look
- `verify/`: an independent re-computation and its outputs

**Reproduce.** From the repo root: `SGI_CACHE_ONLY=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src python results/still-scenes/replay-harness/split/split.py`. The scripts import the harness in `results/still-scenes/analysis-scripts/harness/` and read the development scenes from `data/dev_scenes/` (release dev-scenes-v1). `verify/v_rescore.py` takes more than 10 minutes.

**Data.** `results/logs/still-runs.tar.gz` (unpacks to `runs/` at the repo root); development scenes in `data/dev_scenes/` (release dev-scenes-v1).
