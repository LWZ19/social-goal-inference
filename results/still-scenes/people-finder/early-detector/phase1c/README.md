# Early people detector: first evaluation (I09 A1 and A2)

**Question.** Can a local person detector with a pose model (RT-DETRv2 plus ViTPose, depth-lifted) replace the model's people list: does it find, place and face people on the development scenes (A1), and does handing its people to the solver in place of the model's raise goal success (A2)?

**Result.** Both pre-registered steps are a success, resting on one general fix to facing (variant F5, which drops a face rule that reversed every back-view person).
- A1, realistic images (42 scenes, 118 visible people): recall 116/118 = 0.983, 3 false people (0.071 per image), median position error 0.053 m, facing within 45 deg for 111/116 matched people (non-profile 59/59). On Gemini's A0 pairs the detector's facing is 0.951 against Gemini's own 0.906. Median run time 55.6 ms per image.
- A1, placeholder scenes (26, reported, not judged): recall 49/58 = 0.845; all 9 misses are group members of one layout seen side-on or from behind.
- A2, arm D (primary, 252 records each): Qwen 236, GLM 202, DeepSeek 199, against recorded 234, 186 and 182 and bars 232, 199 and 190. No kill clause fired.
- The independent re-computation reproduced the raw detector outputs bit for bit on 97 images and every fit row and evaluation person.

**Files.**
- `FROZEN.md`, `DEVIATIONS.md`, `DEVIATIONS_2.md`: the plan and the later changes.
- `probe/`: the one-image format probe made before the plan was fixed.
- `a1/`: the detector and pose sidecar (`sidecar/infer.py`), the project side (`pipeline.py`, `common.py`), manifests (`manifest_fit.py`, `manifest_eval.py`), the 192-setting fit (`fit_grid.py`, `fit_grid.txt`), the one fix (`ADDENDUM_FIX.md`, `fit_fix.py`, `fit_fix.txt`), the evaluation (`evaluate.py`, `tables.txt`, `rows.jsonl`, `people_rows.jsonl`, `summary.json`, `eval_people.json`), timing (`project_timing.py`, `timing/`), `selfcheck.py`/`.txt`, `ORDER.txt`.
- `a2/`: the solver replays (`replay_d.py`, `base_*.jsonl`, `rows_*.jsonl`), `tables.py` (`tables.txt`, `summary.json`), `selfcheck.py`/`.txt`, `ORDER.txt`.
- `verify_i09_a1/`, `verify_i09_a2/`: independent re-computation with their own sidecar, pipeline and replay (`compare_*.txt`, `veval.txt`, `vfit.txt`, `vtime.txt`, `vtables.txt` and others).
- The full write-up: `results/still-scenes/overview/phase-notes/phase1c.md`.

**Reproduce.** From the repository root, with `PYTHONPATH=src`:
- From stored rows, with `D=results/still-scenes/people-finder/early-detector/phase1c`:
  - `python $D/a2/tables.py` rebuilds `a2/tables.txt` and `a2/summary.json`;
  - `python $D/verify_i09_a2/vtables.py` rebuilds `vtables.txt` (the order of equal-count keys in a few lines varies from run to run);
  - `python $D/verify_i09_a1/compare_raw.py` prints `compare_raw.txt`.
- Synthetic checks: `python $D/a1/selfcheck.py --no-scorer` rebuilds `a1/selfcheck.txt`; `python $D/a2/selfcheck.py` prints `a2/selfcheck.txt`.
- Through the step-0 harness (`results/still-scenes/analysis-scripts/harness/`) and the development scenes (`data/dev_scenes/`):
  - `python $D/verify_i09_a1/veval.py` rebuilds `veval.txt` and its rows, and `python $D/verify_i09_a1/veph_misses.py` prints `veph_misses.txt`;
  - `python $D/verify_i09_a2/cache_check.py` rebuilds `cache_check.txt`; `explore.py` lists the planned runs;
  - `python $D/verify_i09_a1/manifest.py` and `python $D/probe/probe_manifest.py` rebuild their manifests, which then name the scene folders where they lie now;
  - `python $D/a1/project_timing.py` and `python $D/verify_i09_a1/vtime.py` (with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to 2) time the project side again; only the timings change.
- Still blocked here:
  - the fit (`a1/fit_grid.py`, `a1/manifest_fit.py`, `verify_i09_a1/vfit.py`, `verify_i09_a1/vselfcheck.py`, and the scorer part of `a1/selfcheck.py`) needs the images and depth of fresh seeds 100–109, which are not stored (`../../../perception/fresh_seeds/README.md` says how to render them);
  - `a1/evaluate.py`, `a1/fit_fix.py`, `a1/manifest_eval.py` and `a2/replay_d.py` check `*.sha256` files that are not in this repository, and `verify_i09_a2/vreplay.py` checks the hash of `FROZEN.md`, which has been edited since;
  - the sidecars need a GPU and the pinned model snapshots.

**Data.** `results/logs/research-data.tar.gz` (the manifests, raw detector outputs and `*.jsonl`/`*.json` rows beside these notes). The development scenes are in `data/dev_scenes/` (release `dev-scenes-v1`), and the recorded runs in `runs/` (`results/logs/still-runs.tar.gz`); both are read through the harness. Inputs also come from `../../../perception/depth-blob-detector/`, `../../../replay-harness/repairs/` and `../../../facing/`.
