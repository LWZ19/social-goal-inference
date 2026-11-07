# Metric v0 (I14)

**Question.** What do the solver's top place and its four runners-up score under the package scorer, and how much of the free floor do they cover?

**Result.** For every goal-step record the scorer gives hit@1 to hit@5, P(success) of the set and a lift over the floor. For example the old-wording realistic cells have 143 (Qwen), 141 (GLM) and 157 (DeepSeek) goal-step records; the full tables are in `tables.txt`.

**Files.**
- `METRIC_NOTE.md` : the frozen v0 definitions (records, places, floor, lift).
- `i14.py` : the scorer; writes one row per record to `rows_<group>.jsonl`.
- `tables.py`, `tables.txt` : tables from the rows (needs `run_new.log` and `run_oldwatch.log`, which are not in the repository).
- `posthoc_room.py`, `posthoc_room.txt`, `rows_posthoc_room.jsonl` : a post hoc room-based variant.
- `rows_new.jsonl`, `rows_oldwatch.jsonl` : stored rows.
- `explore_*.py`, `probe_*.py` : listing, format and timing probes.
- `verify/` : an independent re-computation (`v_main.py`, `v_tables.py`) with its rows and `v_tables.txt`.

**Reproduce.** From the repo root: `PYTHONPATH=src python results/still-scenes/metrics-and-statistics/metric-v0/i14.py GROUP [LIMIT]`, then `.../tables.py`. Uses the step-0 replay harness in `results/still-scenes/analysis-scripts/harness/` and the development scenes in `data/dev_scenes/` (release dev-scenes-v1). `verify/v_tables.py` reads only stored rows and gives identical output. `posthoc_room.py` was re-run and reproduces its rows and table exactly. `i14.py` takes more than 10 minutes per group; the first rows it writes match the stored ones.

**Data.** `logs/still-runs.tar.gz` and `logs/research-data.tar.gz`.
