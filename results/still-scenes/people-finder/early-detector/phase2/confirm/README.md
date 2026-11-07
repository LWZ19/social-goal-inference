# Unseen-layout confirmation of the early people detector

**Question.** Scored once, with nothing re-tuned, does the frozen phase-1c detector meet its pre-registered bars on
182 realistic layouts it has never seen? The detector is RT-DETRv2-r50vd at tau 0.25 with ViTPose facing, lifted
with depth.

**Result.** The verdict is **neither** (`tables.txt`).
- Six of seven success bars were met: recall 510/520 = 0.981 [Wilson 0.965, 0.990], line members 207/208, median
  position error 0.061 m, non-profile facing median 10.3 deg, and profile front/back 223/223.
- False people were 44/182 = 0.242 per image against a bar of 0.10 (layout bootstrap 0.176–0.313).
- No kill rule fired.
- Post hoc (`posthoc_false.txt`):
  - 16 false people are displaced counter staff;
  - 13 sit by staff whom the visibility rule calls hidden;
  - 15 are small far boxes.
  - Without the first two groups, the rate would be 15/182 = 0.082 per image.
- Checks C2 (development reproduction, exact) and C3 (truth self-check, 182 of 182) pass (`c2_repro.txt`,
  `c3_truth.txt`).

**Files.**
- `FROZEN.md`: the protocol, frozen before rendering. `DEVIATIONS_1.md` and `DEVIATIONS_2.md` record the
  departures.
- `render/render_seeds.py`: renders seeds 200–219 under a 3 h clock. It stopped after seed 212. Its log is
  `render/render_events.jsonl`, with `render/state.json` and `render/manifest.json`.
- `manifest.py` writes `manifest.json`, the test manifest (13 seeds, 182 layouts).
- `pathguard.py`: the path guard. Only `results/still-scenes/scene-seeds/phase2/realistic/seed200`–`219/eval/` may be
  read, and held-out names are refused.
- `confirm_lib.py`: the shared code. It pins phase-1c's code and data by sha256.
- `c2_repro.py` writes `c2_repro.txt`. `c3_truth.py` writes `c3_truth.txt`.
- `sidecar/infer_confirm.py` writes `raw/`, the one detector and pose pass (torch, GPU).
- `score.py` writes `tables.txt`, `rows.jsonl`, `people_rows.jsonl` and `summary.json`.
- `posthoc_false.py` writes `posthoc_false.txt`, the post hoc breakdown of the false people.
- `ORDER.txt` is empty in this copy.
- The independent re-check is in `../verify_i09_confirm/`.

**Reproduce.** Run from the repository root, in this order:

```bash
C=results/still-scenes/people-finder/early-detector/phase2/confirm
PYTHONPATH=src python $C/render/render_seeds.py --trial            # Blender; then --seeds 200-219, then --manifest
PYTHONPATH=src python $C/c2_repro.py
PYTHONPATH=src python $C/manifest.py
PYTHONPATH=src python $C/c3_truth.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I $C/sidecar/infer_confirm.py   # torch sidecar, GPU
PYTHONPATH=src python $C/score.py                                  # --dry-dev: the development dry run only
PYTHONPATH=src python $C/posthoc_false.py
```

The step-0 harness they import is in this repo (`results/still-scenes/analysis-scripts/harness/`), and the
development scenes go in `data/dev_scenes/`. They still cannot be re-run here:
- `confirm_lib.py` first checks `FROZEN.sha256` and the `DEVIATIONS_*.sha256` files, which were left out;
- several pinned code and data hashes no longer match, because cleaning removed absolute path prefixes and edited
  notes (the pinned values are kept as they were);
- they need the rendered test scenes.

**Data.**
- `research-data.tar.gz`: `manifest.json`, `raw/`, `rows.jsonl`, `people_rows.jsonl`, `summary.json` and the
  `render/` logs.
- The development scenes: `data/dev_scenes/`, from the `dev-scenes-v1` release.
- Not included: the test scenes (pictures and depth).
