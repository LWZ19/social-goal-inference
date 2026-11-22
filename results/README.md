# Results

This folder holds the scoring scripts and stored outputs behind the numbers in the project presentation. Slide numbers
below refer to that presentation, and B1 to B10 are its backup slides.

- **Pointing:** the VLM marks the spot in the picture. In the outputs it is `R`, or `E` when it is shown worked examples.
- **Our method:** the VLM names the people and the rules, and the solver in `src/sgi/solver.py` finds the spot. In the
  outputs it is `C`.
- **Good spot (still scenes):** inside the region where a polite person would stand, facing within 45 degrees.
- **Time in the right spot (moving lines):** the share of a 92 s run in which the robot is 0.5 to 1.2 m behind the last
  person waiting, within 0.6 m of the line, facing the front and clear of everyone, or at the serving spot once it is
  first. The exact rule is in [`SCORING_STRICT.md`](moving-lines/design-and-scoring/SCORING_STRICT.md).

## Still scenes

| Slide | Finding | Script | Stored output |
|---|---|---|---|
| 12, 13 | 70 new scenes, six VLMs: pointing with examples 63.3%, ours 85.1%. Waiting lines 41% and 81%. Better on 42 scenes, worse on none | [`i12_score.py`](still-scenes/fresh-scenes/i12_score.py) | [`i12_score_six_out.txt`](still-scenes/fresh-scenes/i12_score_six_out.txt) |
| 14 | At lines, all rules 77%, only "who to stand behind" 77%, people with no rules 0% | [`i01_baselines.py`](still-scenes/replay-harness/baselines/i01_baselines.py), [`tables.py`](still-scenes/replay-harness/baselines/tables.py) | [`tables.txt`](still-scenes/replay-harness/baselines/tables.txt) |
| 15 | Held out: lift 54% to 88%, group photo 4% to 8% | [`held_out_count.py`](still-scenes/hidden-situations-and-more-models/held_out_count.py) | [`held_out_count_out.txt`](still-scenes/hidden-situations-and-more-models/held_out_count_out.txt) |
| 15 | New situations: bent line 2% to 69%, machine 14% to 62% | [`e4_score.py`](still-scenes/new-situations/e4_score.py) | [`e4_score_six_out.txt`](still-scenes/new-situations/e4_score_six_out.txt) |
| 16, B8 | 229 crowd scenes, five VLMs: ours 7.4 points ahead of pointing with examples, and drops 23.4 points more when people are added | [`crowd_score.py`](still-scenes/crowds/vlm-run/crowd_score.py) | [`crowd_score_five_out.txt`](still-scenes/crowds/vlm-run/crowd_score_five_out.txt) |
| 16, B9 | Someone just behind a line: ours 3.5%, pointing with examples 18.7% | [`behind_line_split.py`](still-scenes/crowds/vlm-run/behind_line_split.py) | [`behind_line_split_out.txt`](still-scenes/crowds/vlm-run/behind_line_split_out.txt) |
| B10 | Stricter joining check: 1.3 m or more behind the line, 7% to 52% on a new seed; no one added, unchanged at 81% | [`fresh_score.py`](still-scenes/crowds/fresh-seed3/fresh_score.py), [`gate_replay.py`](still-scenes/crowds/joining-gate/gate_replay.py) | [`fresh_score_out.txt`](still-scenes/crowds/fresh-seed3/fresh_score_out.txt), [`gate_replay_out.txt`](still-scenes/crowds/joining-gate/gate_replay_out.txt) |
| B1 | Older and newer models: 12 of 13 pick more good spots with our method | [`layout_tests.py`](still-scenes/method-comparison/layout-level-tests/layout_tests.py) | [`layout_tests_out.txt`](still-scenes/method-comparison/layout-level-tests/layout_tests_out.txt) |
| B2 | The VLM finds the right target in 95% of tries | [`target_found.py`](still-scenes/target-found/target_found.py) | [`target_found_out.txt`](still-scenes/target-found/target_found_out.txt) |
| B3 | From an 8-frame tour, 839 of 840 tries pick a frame that shows the target | [`tour_score.py`](still-scenes/tour-grounding/tour_score.py) | [`tour_score_out.txt`](still-scenes/tour-grounding/tour_score_out.txt) |
| B4 | Each answer takes 1.2 to 1.8 times as long as pointing | [`answer_time.py`](still-scenes/speed/answer_time.py) | [`answer_time_out.txt`](still-scenes/speed/answer_time_out.txt) |
| B4, B7 | A full solve takes 130 ms | [`solver_time.py`](still-scenes/speed/solver_time.py) | [`solver_time_out.txt`](still-scenes/speed/solver_time_out.txt) |
| B4 | A small people finder saves 21 to 25% of the time on still scenes | [`e2_score.py`](still-scenes/people-finder/e2_score.py) | [`e2_score_six_out.txt`](still-scenes/people-finder/e2_score_six_out.txt) |
| B7 | The solver is right on all 229 crowd scenes with the true people and rules | | [`tables.txt`](still-scenes/crowds/score/tables.txt) |
| B6, B7 | Learned replacements for the solver: 67% and 57%, against the solver's 76% | | [soft density](still-scenes/learned-models/soft-density/tables.txt), [flow model](still-scenes/learned-models/flow-model-pilot/tables.txt) |
| B6 | Picking from numbered marks helps GLM by 22 good spots and costs DeepSeek 27 | | [GLM](still-scenes/numbered-spots/m2_glm53_out.txt), [DeepSeek](still-scenes/numbered-spots/m2_dsv41_out.txt) |

The held-out and new situations (slide 15) and the crowd scenes use simpler renders of the layouts. The pictures on
those slides are the same layouts redrawn in Blender. The 85.1% leaves out 20 GLM calls that failed. Counting them as
misses gives 84.0%.

## Moving lines

| Slide | Finding | Script | Stored output |
|---|---|---|---|
| 18 | Claude Sonnet: decide once by pointing 6%, ask once and track 29% | [`i9_score.py`](moving-lines/school-models/i9_score.py) | [`i9_score_e3b_out.txt`](moving-lines/school-models/i9_score_e3b_out.txt) |
| 18 | Asking again and again: 2 to 10%, with 22 to 29 calls against 4 | [`i10_score.py`](moving-lines/re-asking/i10_score.py) | [`i10_score_out.txt`](moving-lines/re-asking/i10_score_out.txt) |
| 19, 21, 22 | Hold-place rule on 64 untuned layouts: Sonnet 25% to 47%, Gemini 21% to 37%, DeepSeek 21% to 31%, Haiku 20% to 28% | [`hold_extended_score.py`](moving-lines/hold-place/hold_extended_score.py) | [`hold_extended_score_out.txt`](moving-lines/hold-place/hold_extended_score_out.txt) |
| 19, 22 | A robot given the ideal place every half second: 65% | [`ceiling_score.py`](moving-lines/hold-place/ceiling/ceiling_score.py) | [`ceiling_score_out.txt`](moving-lines/hold-place/ceiling/ceiling_score_out.txt) |
| 22 | With the robot's own camera in place of exact positions | [`i5_score.py`](moving-lines/own-camera/i5_score.py) | [`i5_score_out.txt`](moving-lines/own-camera/i5_score_out.txt) |
| B5 | Time closer than 0.55 m to someone: 2.3% for tracking, 7.1% and 10.2% for the others | [`bumps.py`](moving-lines/bumps/bumps.py) | [`bumps_out.txt`](moving-lines/bumps/bumps_out.txt) |
| B5 | Cutting in, per second near the line: 14.5% and 13.7% | [`item6_cutin_exposure.py`](moving-lines/cut-ins-and-decomposition/item6_cutin_exposure.py) | [`item6_out.txt`](moving-lines/cut-ins-and-decomposition/item6_out.txt) |
| B6 | A people finder in moving lines: DeepSeek 29% to 10% | [`x3_score.py`](moving-lines/people-finder/x3_score.py) | [`x3_score_out.txt`](moving-lines/people-finder/x3_score_out.txt) |
| B6 | Walking before the answer arrives: 27 runs worse, 12 better | | [`results_fresh_sonnet55_low.md`](moving-lines/updating-vs-once/results_fresh_sonnet55_low.md) |
| B6 | A rule-based fix for the front of the line: no change | [`front_score.py`](moving-lines/front-of-line-fix/front_score.py) | [`front_score_out.txt`](moving-lines/front-of-line-fix/front_score_out.txt) |

## Reading the outputs

The stored outputs are exactly as the scripts printed them, so they keep the labels used during the runs.

- `R`, `E`, `C` are pointing, pointing with examples, and our method.
- `R1` decides once by pointing. `C1` decides once with our method. `Rq0` and `Cq` ask the VLM again during the run.
- `CT2` asks once and tracks the people. `CT3F` adds a check that people move with the line. `CT3FL` adds the hold-place
  rule. `OW` is a robot given the ideal place.
- "Frozen" or "bar" marks a scoring rule or pass mark fixed before the run. The planning notes themselves are not
  included. W/L/T is wins, losses and ties, paired by layout, or by scene and seed.
- Script names such as `i12`, `e4`, `x3` and `item6` are experiment numbers from the project.

## Supporting material

- [`moving-lines/roles-vs-marked-spot/`](moving-lines/roles-vs-marked-spot/): remembering who to stand behind against following the marked spot.
- [`moving-lines/test-robots/`](moving-lines/test-robots/): test robots given the true serving spot, which show the largest loss is at the front of the line.
- [`moving-lines/test-layouts/`](moving-lines/test-layouts/) and [`moving-lines/best-version-and-running-guess/`](moving-lines/best-version-and-running-guess/): the layouts of the final test and the code that breaks down where time is lost.
- [`still-scenes/run-to-run-noise/`](still-scenes/run-to-run-noise/): the gap between methods against the noise between repeated runs.
- [`still-scenes/analysis-scripts/harness/`](still-scenes/analysis-scripts/harness/): the tool that solves stored answers again, used by the replay analyses.
- [`still-scenes/metrics-and-statistics/`](still-scenes/metrics-and-statistics/): earlier statistics that the layout tests compare with.

## Re-running a script

The scripts read stored run logs and make no model calls. The logs are in the release
[`run-logs-v1`](https://github.com/LWZ19/social-goal-inference/releases/tag/run-logs-v1). From the repository root:

```bash
gh release download run-logs-v1 --repo LWZ19/social-goal-inference
tar xzf run-logs.tar.gz            # unpacks runs/ and results/data/moving/
PYTHONPATH=src python results/moving-lines/hold-place/hold_extended_score.py
```

A few scripts also need the rendered scenes, which are too large to include: `fresh_score.py`, `gate_replay.py`,
`i01_baselines.py`, `layout_tests.py` and `solver_time.py`. Their stored outputs are the record.

Three stored outputs were saved before the last runs finished or hold an extra check, so a re-run differs from them in
rows the talk does not use: the Qwen rows of `i10_score.py`, one GLM line of `i9_score.py`, and a closeness section of
`front_score.py`.
