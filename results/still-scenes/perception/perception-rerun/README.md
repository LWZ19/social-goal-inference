# Perception rerun: what the people call misses

**Question.** When the model misses people in a line or group, did its people call never find them, or did it find them and then lose them later?

**Result.** For GLM (s19, realistic) 74 members were missed over the records with a measured recall; 56 (76%) were never found by the people call. At line goal steps 34 of 296 detections are more than 1 m from every true person, all within 0.3 m of the wall line.

**Files.**
- `perception.py` -> `perception_out.txt`: the main recount (claims P1-P3 and new rows).
- `p1_variants.py`, `p1_variants2.py`, `i03_why.py`, `phantom_pixels.py`: variants and follow-ups, each with `*_out.txt`.
- `explore.py`, `explore2.py`, `explore3.py`: exploratory scans.
- `verify/`: independent re-counts (`v_main.py`, `v_n6.py`, `v_p1alt.py`, `v_p1scan.py`, `v_p2tok.py`).

**Reproduce.** `PYTHONPATH=src python results/still-scenes/perception/perception-rerun/perception.py > perception_out.txt`. It reads runs and scenes through the step-0 harness (`results/still-scenes/analysis-scripts/harness/`) and `data/dev_scenes/`; re-run here it reproduces `perception_out.txt` byte for byte, as do `p1_variants*.py`, `i03_why.py`, `phantom_pixels.py` and `verify/*.py`.

**Data.** `results/logs/still-runs.tar.gz` (the `runs/` folder at the repo root) and the development scenes in `data/dev_scenes/` (release dev-scenes-v1).