# Zero-call gates for the flow model (C5)

**Question.** Are there enough distinct configurations, and does a kernel density on scene features help, to justify training a flow model?

**Result.** Gate 1 is not met: 22 distinct configurations against the 30 required, so the first kill source fires. Gate 2 is met: the ratio KDE gains at least 0.1 nats per region cell in 2 of 3 leave-one-situation-out folds.

**Files.**
- `FROZEN.md` : the frozen protocol.
- `gate1.py`, `gate1.json`, `gate1.txt` : the configuration count.
- `cells.py`, `cells_check.txt`, `common.py` : cells and shared code.
- `gate2.py`, `tables.py`, `rows_gate2.jsonl`, `folds.json`, `inner_cv.txt`, `tables.txt`, `work/` : gate 2.
- `run_chain_a.sh`, `run_chain_b.sh` : the two run chains.
- `precheck/`, `verify/`, `cache/` : pre-freeze checks, independent re-implementation (`kde_v.py` and others), cache index.

**Reproduce.** `bash results/still-scenes/learned-models/flow-model-gates/run_chain_a.sh` and `run_chain_b.sh`, then `python .../tables.py` from the repo root. Uses the step-0 replay harness in `results/still-scenes/analysis-scripts/harness/` and the development scenes in `data/dev_scenes/` (release dev-scenes-v1). The cache and verifier cache hold only index files, and every script here stops at its FROZEN.md hash check (the pinned hash no longer matches the note after its paths were updated), so none could be re-run.

**Data.** `logs/still-runs.tar.gz`, `logs/bulk.tar.gz`.
