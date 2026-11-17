# Depth fix on Claude and Gemini

**Question.** The person snap (move a click that lands on someone in front onto the detected person behind) was
tested on the free models only; does it help Sonnet 5.5, Haiku 5.5 and Gemini 3.8 on the same 24 scenes, and does it
ever break a right answer?

**Result.** These models rarely need it: the plain depth lift already gets the target right in 142 of 144 records on
the 8 hidden-target layouts (the snap 144), so the frozen bar (a gap of at least 10 points against the unhidden
controls) fails at 1.4 points and the claim is not made. The snap fired 3 times in 432 records, fixed 3 (all Haiku)
and broke none.

**Files.**
- `X5_FROZEN.md`: the plan: the snap test of `results/still-scenes/group-of-four-case/` (`SNAP_FROZEN.md`,
  amendments 1-3) unchanged except the models, plan call only.
- `x5_score_out.txt`: the result, scored by `results/still-scenes/group-of-four-case/snap_score.py`.

**Reproduce.** From the repository root:

```bash
G=results/still-scenes/group-of-four-case
PYTHONPATH=src python $G/snap_score.py --runs $(ls -d runs/*_snap_plan_claude_* runs/*_snap_plan_gemini_fast) \
    --scenes $G/snapset/run_scenes_v2 --people $G/snapset/snap_people.json
```

It reads the scenes' depth images, which are not in the repository; with them it reproduces `x5_score_out.txt`
exactly. The runs were made with `python -m sgi.run --config $G/snapset/plan_only.yaml --model <model>
--scenes $G/snapset/run_scenes_v2 --name snap_plan_<model>` (model calls).

**Data.** `results/logs/still-runs.tar.gz` (`runs/*_snap_plan_*`); `results/logs/research-data.tar.gz` (the scene
files and `snap_people.json`).
