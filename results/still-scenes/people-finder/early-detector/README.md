# Early people detector (I09)

**Question.** Can a local person detector with a pose model, lifted with depth, replace the model's people answer on still scenes, and does it hold up on layouts it has never seen?

**Result.** Three stages, in order:
- [`phase1c/`](phase1c/README.md), first evaluation. RT-DETRv2 plus ViTPose, fitted on worked examples and fresh seeds 100–109, then scored once on the 68 development scenes. On realistic images it finds 116/118 people (0.983) with 0.071 false people per image. Its facing is within 45 deg on 0.951 of Gemini's A0 pairs, against Gemini's own 0.906. Handing its people to the solver raises GLM from 186 to 202 and DeepSeek from 182 to 199 of 252. Placeholder recall is 0.845.
- [`phase2/`](phase2/README.md), unseen-layout confirmation. The unchanged detector on 182 realistic layouts (seeds 200–212) meets 6 of 7 bars (recall 0.981), but places 0.242 false people per image against a 0.10 bar, so the pre-registered verdict is "neither". The folder also holds the build of a live check, which has no live result.
- [`fix/`](fix/README.md), diagnosis and fix. On new training seeds four causes explain the false people. Fix C07, chosen by a frozen rule, cuts them on the held-back seeds 116–119 from 0.214 to 0.036 per image, with recall 164/164. A second confirmation on unseen layouts is frozen but was not run, so by its own standard the detector is not yet a method.

**Files.** One folder per stage, each with its own README. The first evaluation's full write-up is `../../overview/phase-notes/phase1c.md`. The later people-finder study one folder up (`../E2_FROZEN.md`) uses the first evaluation's detector, not the fix.

**Reproduce.** See each stage's README. Scripts run from the repository root with `PYTHONPATH=src`. They read recorded runs and development scenes through the step-0 harness in `results/still-scenes/analysis-scripts/harness/`. Some steps are still blocked here, by missing `*.sha256` files, by code and data pins that no longer match, or by scene images that are not stored. The detector passes also need a GPU and the pinned model snapshots.

**Data.**
- `results/logs/research-data.tar.gz`: the data files beside the notes.
- `results/logs/still-runs.tar.gz`: the recorded runs in `runs/` that the replays read.
- `data/dev_scenes/` (release `dev-scenes-v1`): the development scenes.
- `results/logs/bulk.tar.gz`: the test sandboxes of `phase2/`.
