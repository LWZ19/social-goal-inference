# Latency of the call stages

**Question.** How long do the plan, context, people and constraints calls take, and how much would overlapping them save?

**Result.** For Qwen on the three realistic sets, serial time is 61.2 s per episode against 49.3 s on the critical path (19.5% saved); GLM is 9.8 s against 8.0 s (19.0%). Qwen's people call takes 16.0 s and its constraints call 31.7 s on average.

**Files.**
- `latency.py / latency_out.txt`: the replay and the tables
- `explore.py, explore2.py / explore2_out.txt`: first look at the runs
- `verify/`: independent checks (stages, pacer, key fix)

**Reproduce.** From the repo root: `SGI_CACHE_ONLY=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src python results/still-scenes/speed/latency/latency.py`. The scripts import the harness in `results/still-scenes/analysis-scripts/harness/` and read the development scenes from `data/dev_scenes/` (release dev-scenes-v1).

**Data.** `results/logs/still-runs.tar.gz` (unpacks to `runs/` at the repo root) (latencies are those recorded in the runs).
