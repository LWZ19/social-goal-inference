# Numbered spots in place of pointing

**Question.** If the picture shows numbered marks on the candidate objects and people, and the VLM names a number
instead of pointing at a pixel, does it pick the right target more often?

**Result.** Only for the model that points worst. On GLM-5.3 marks raise the right target from 215 to 237 of 252
(30 gained, 8 lost, against 26/26 changes between two unmarked draws; +22 over layouts, p 0.011). For DeepSeek, Qwen,
Qwen 3.8 Next and Gemini the change is within that redraw noise, and DeepSeek's located success falls (195 → 168)
because its click is drawn towards the labels. A perfect pick among the marks would have gained only +13 records
(the frozen bar was +36), and on the simulator's frames marks did not help either model tested.

**Files.**
- `RESULTS.md`: all results (M1, M1b, M1c, M2, M2P, M2d, M4) and what they mean.
- `M1_FROZEN.md`, `M2_FROZEN.md`, `M2P_FROZEN.md`, `M4_FROZEN.md`: the plans.
- M1, coverage and the perfect-pick ceiling: `m1_coverage.py` → `m1_out.txt`, `m1_instances.jsonl`; post hoc
  `m1b_diagnose.py` → `m1b_out.txt`, `m1b_rows.jsonl` and `m1c_split.py` → `m1c_out.txt`, `m1c_rows.jsonl`.
  `buggy_idcache/`: the outputs of the first M1c run, whose candidate cache was keyed wrongly (fixed; M1 and M1b
  unchanged by the fix); `fix_done.txt`.
- M2, marked plan calls: `m2lib.py` (marks and drawing), `m2_driver.py` (the calls), `prompts_marks/`, `m2_score.py`
  → `runs/m2_<model>/m2_out.txt` and `m2_rows.jsonl`; post hoc `m2b_diagnose.py` → `m2b_out.txt`, `m2c_noise.py` →
  `m2c_out.txt`, `m2d_combine.py` (the two-call design), `m2e_glm.py` → `m2e_glm_out.txt`, `m5_check.py` →
  `m5_check_out.txt` (which wrong targets name a person).
- M2P, label attraction: `m2p_driver.py`, `m2p_episodes.json`, `m2p_score.py` → `runs/m2p_dsv41/m2p_out.txt`.
- M4, marks on the simulator's frames: `m4_driver.py`, `m4_score.py` → `runs/m4_<model>/m4_out.txt`.

**Reproduce.** From the repository root (no model call):

```bash
PYTHONPATH=src python results/still-scenes/numbered-spots/m5_check.py
PYTHONPATH=src python results/still-scenes/numbered-spots/m2c_noise.py results/still-scenes/numbered-spots/runs/m2_glm53
PYTHONPATH=src python results/still-scenes/numbered-spots/m2b_diagnose.py results/still-scenes/numbered-spots/runs/m2_glm53
```

These reproduce their outputs exactly (`m2b`/`m2c` write `m2b_out.txt`/`m2c_out.txt` into the run folder given).
The other scorers (`m1_coverage.py` and everything that imports it) first check the sha256 of `M1_FROZEN.md` and of
the lifting study's frozen note (`results/still-scenes/perception/lifting/`); those notes were edited for sharing, so
the checks stop them. The drivers make model calls.

**Data.** `results/logs/research-data.tar.gz` (`runs/`, the `*.jsonl` files); the development scenes in
`data/dev_scenes/` (see `results/README.md`) for the M1 and M2 scorers; `results/logs/moving-runs.tar.gz` for M4's
frames (`runs/dsv41` in `results/data/moving/`).
