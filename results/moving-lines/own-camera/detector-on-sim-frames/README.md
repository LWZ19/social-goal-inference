# People detector: simulator frames, still scenes and the tracker

**Question.** Is a frozen people detector (A1: RT-DETRv2 boxes and ViTPose keypoints) good enough to replace the
model's people call, and to track people from the robot's camera in the simulator?

**Result.** On the 13 simulator first frames it finds 75 of 75 visible people, with 1 false person and a median
position error of 0.077 m (E1a). On the still development scenes, detector people in place of the people call are
non-inferior on every model: Sonnet 132 against 124 of 143 records, DeepSeek 127 against 127 of 151, Qwen 131 against
131 of 144, GLM 73 against 63 of 105 (a partial run) (E1b). The tracker built on it (E1d) passes its 6 unit tests;
`CTd` in the parent folder uses it.

**Files.**
- `RESULTS.md`: the write-up of E1a, E1b, E1d and E1c (E1c's files are in `../../people-finder/first-check/`).
- `E1_FROZEN.md`: the E1b plan. `e1_driver.py`: the E1b runs (model calls). `e1_score.py`: the E1b scorer, output
  `runs/e1_<model>/e1_out.txt`. `runs/e1_<model>/records.jsonl`: one line per record and arm.
- `e1a/`: `e1a_sidecar.py` (the detector pass on the GPU, output `raw_sim/`), `e1a_score.py` (output `e1a_out.txt`),
  `manifest_sim.json` (the 13 frames).
- `e1d/`: `E1D_FROZEN.md` (the plan), `detector_service.py` (the detector as a local HTTP service), `tracker.py`,
  `test_tracker.py`.

**Reproduce.** From the repository root:
- `PYTHONPATH=src python results/moving-lines/own-camera/detector-on-sim-frames/e1_score.py results/moving-lines/own-camera/detector-on-sim-frames/runs/e1_dsv41`,
  one run folder per model. It rewrites that folder's `e1_out.txt`; all four reproduce exactly.
- `PYTHONPATH=src python -m pytest results/moving-lines/own-camera/detector-on-sim-frames/e1d/test_tracker.py`.
- `e1a_score.py` needs the 13 simulator frames (request, truth and depth), which are not in this repository; with
  them it reproduces `e1a_out.txt` exactly. `e1_driver.py` makes model calls and needs the still-scene harness
  (`research_harness`), which is not in this repository. `e1a_sidecar.py` and `detector_service.py` need a GPU and
  the detector weights. The sha256 checks in these three scripts fail on this repository's copies of `infer.py` and
  `eval_people.json`, which were cleaned for sharing.

**Data.** `results/logs/research-data.tar.gz` (`runs/e1_*/records.jsonl`, `e1a/raw_sim/`, `e1a/manifest_sim.json`).
