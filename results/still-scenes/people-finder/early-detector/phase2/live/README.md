# Live check: the early detector as the people source (build and dry runs)

**Question.** Take the recorded C runs at trial t0 and make one change: the frozen detector's numbered people replace
the VLM's people call. Does the pipeline then keep A2's gain? The frozen bar is at least 66 (GLM) and 68 (DeepSeek)
successes of 84.

**Result.** The live run was not made.
- The driver passed all 17 zero-call dry-run checks (`dryrun/*_result.json`).
- After two fix rounds, a count-only pass was made on 18 Oct (`runs/count_only_*.json`, driver `e5e79093…`). It found
  48 GLM and 54 DeepSeek constraints misses, and 0 plan, locate or context misses, as the go-live rule needs.
- There are no live run folders, so there is no D-live reading.

**Files.**
- `FROZEN.md`: the protocol (question, baselines, reading rule, call cap), frozen before the driver existed.
- `DEVIATIONS_1.md` (21 Oct, before any live call): a DeepSeek live result would mix two model versions, because the
  gateway's DeepSeek model changed after the recorded runs.
- `driver.py`: the driver. It runs `sgi.run.run_episode` from `src_202c165/src`, with `DetectorPeople` in place of
  the people call. It has a call ledger and a hard cap (120 in all, 60 per model), and `--count-only` and `--mock`
  modes.
- `before_fix_e6e73181/`, `before_fix2_84bab2e4/`, `before_fix3_41940c3c/`: frozen earlier builds of `driver.py`.
  They read `src_202c165/` and `FROZEN.md` beside themselves, so they run only when copied back into `live/`.
- `dryrun.py` writes `dryrun/`: zero-call checks a (mock end to end), b and b2 (count-only), and c (the cap).
  `dryrun/a_numbered_*.txt` describe two numbered images.
- `runs/`: the two count-only reports (`.json` and `.jsonl`).
- `src_202c165/`: the package at commit 202c165, an archived copy (git-ignored).

**Reproduce.** Run from the repository root:

```bash
L=results/still-scenes/people-finder/early-detector/phase2/live
PYTHONPATH=src python $L/dryrun.py a        # then b, b2, c; zero calls
PYTHONPATH=src python $L/driver.py --model glm53 --trial 0 --out $L/runs --cache-dir <absolute cache folder> --count-only
PYTHONPATH=src python $L/driver.py --model glm53 --trial 0 --out $L/runs --cache-dir <absolute cache folder> \
    --env-file <absolute key file>          # the live run: model calls; then --model dsv41
```

The driver loads the step-0 harness, which is in this repo (`results/still-scenes/analysis-scripts/harness/`), and
the development scenes, which go in `data/dev_scenes/`. It still cannot be re-run here:
- it checks `FROZEN.sha256`, which was left out;
- it checks the pinned hash of phase-1c's `eval_people.json`, which cleaning changed (absolute path prefixes were
  removed);
- `dryrun.py` builds mock clients, and the live run calls the model gateway.

**Data.**
- `bulk.tar.gz`: `src_202c165/`.
- `still-runs.tar.gz`: the six recorded runs, `s19_{real,staffed,plain}_glm53_con` and
  `s20_{real,staffed,plain}_dsv41_con`.
- `research-data.tar.gz`: `runs/`, the data files in `dryrun/`, and
  `results/still-scenes/people-finder/early-detector/phase1c/a1/eval_people.json`.
- The development scenes: `data/dev_scenes/`, from the `dev-scenes-v1` release.
- Not included: the recorded-answer cache.
