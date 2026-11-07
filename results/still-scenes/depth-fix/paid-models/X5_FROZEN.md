
## Question

The frozen person-snap test (`results/still-scenes/group-of-four-case/SNAP_FROZEN.md` and amendments 1-3) ran only on the free models. Its H1 failed
(8.6 points against a 10-point bar), so its claim was not made.

## Design: the frozen test's, unchanged except the models

- **Scenes:** the same 24 scenes (`results/still-scenes/group-of-four-case/snapset/run_scenes_v2`) and the same A1 detector people
  (`results/still-scenes/group-of-four-case/snapset/snap_people.json`).
- **Calls:** plan only. The config `results/still-scenes/group-of-four-case/snapset/plan_only.yaml` sets context_check always_no, so each record makes
  only the plan call. 3 trials.
- **Models:** Sonnet 5.5 low (`claude_sonnet55_low`), Haiku 5.5 (`claude_haiku55`), Gemini 3.8 (`gemini_fast`).
  These are the entries of the s23 runs, all in `plan_only.yaml`'s model table.

## Reading (reported, no bar)

- Per model: target_correct with the depth lift against with the snap; fixed and broken; detector recall of the target.
- H3, no harm, is the line that matters: broken records on the controls, and none on object targets. The free models
  had 0 breaks in 432.

    cd .
    G=results/still-scenes/group-of-four-case
    X=results/still-scenes/depth-fix/paid-models
    for M in claude_haiku55 gemini_fast claude_sonnet55_low; do
      PYTHONPATH= .venv/bin/python -m sgi.run --config $G/snapset/plan_only.yaml --model $M --scenes $G/snapset/run_scenes_v2 --name snap_plan_$M > $X/log_$M.log 2>&1; echo "$(date +%H:%M) $M exit $?"
    done
    cd $G && python3 snap_score.py --runs $(ls -d runs/*_snap_plan_claude_* runs/*_snap_plan_gemini_fast) --scenes snapset/run_scenes_v2 --people snapset/snap_people.json > ../fill/x5_snap_paid/x5_score_out.txt

## Cost

- **Calls:** 24 scenes x 2 episodes x 3 trials = 144 plan calls per model.
