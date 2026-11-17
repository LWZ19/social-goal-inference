# Scene seeds for the still-scene experiments

**Question.** Which generated scenes did the still-scene experiments fit on and test on, and how were they made?

**Result.** Two seed sets, kept apart by purpose:
- training seeds 110-199, used to fit models and rules;
- test seeds 200-219, kept for confirmation tests.

The notes that use them record no held-out situation in either set.

**Files.**
- `phase2/`:
  - the placeholder seeds 110-199 used by the flow-model pilot, with their generator and manifest;
  - the realistic seeds 200-212, the test data of the first people-detector confirmation (now spent).

  It is described here from the notes that use it (`learned-models/flow-model-pilot/README.md` and
  `people-finder/early-detector/fix/FIX_PROTOCOL.md`).
- `phase3/realistic_train/`: realistic training seeds 110-119, the fit data of the people-detector fix. It holds:
  - each scene's `scene.json` and `render_job.json`;
  - `manifest.json`;
  - `render_log.jsonl`, the generation commands;
  - one `seed<k>.stdout.txt` per seed.

**Reproduce.**
- One generation command per seed is in `phase3/realistic_train/render_log.jsonl`. It is of the form
  `python -m sgi.synth.realistic.generate --variant real --seed <k> --out <folder> --only <scene ids>`.
- Rendering needs Blender.

**Data.**
- The stdout logs are in git. The scene files, manifest and render log come from `results/logs/research-data.tar.gz`.
- The rendered images (about 589 MB for seeds 110-119) are not in this repository.
