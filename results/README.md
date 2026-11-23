# Results

This folder holds the scoring scripts and stored outputs behind the numbers in the project presentation. Slide numbers refer to that presentation, and A1 to A14 are its additional slides.

- **Pointing.** The VLM marks the spot in the picture. In the outputs it is `R`, or `E` when it is shown worked examples.
- **Our method.** The VLM names the people and the rules, and the solver in [`src/sgi/solver.py`](../src/sgi/solver.py) finds the spot. In the outputs it is `C`.
- **Good spot (still scenes).** Inside the region where a polite person would stand, facing within 45 degrees.
- **Time in the right spot (moving lines).** The share of a 92 s run in which the robot is 0.5 to 1.2 m behind the last person waiting, within 0.6 m of the line, facing the front and clear of everyone, or at the serving spot once it is first. The exact rule is in [`SCORING_STRICT.md`](moving-lines/design-and-scoring/SCORING_STRICT.md).
- **Scenes.** The final test uses the realistic Blender scenes. The held-out, new-situation and crowd tests are scored on scenes from the project's generator (`sgi.synth`), and the slides illustrate the same layouts with realistic renders.

## Still scenes

| Slide | Finding | Script | Stored output |
|---|---|---|---|
| 12, 13 | 70 new scenes, six VLMs: pointing with examples 63.3%, ours 85.1%. Waiting lines 41% and 81%. Better on 42 scenes, worse on none | [`final_test_score.py`](still-scenes/fresh-scenes/final_test_score.py) | [`final_test_score_out.txt`](still-scenes/fresh-scenes/final_test_score_out.txt) |
| 14 | At lines, all rules 77%, only "who to stand behind" 77%, people with no rules 0% | [`baselines.py`](still-scenes/replay-harness/baselines/baselines.py), [`tables.py`](still-scenes/replay-harness/baselines/tables.py) | [`tables.txt`](still-scenes/replay-harness/baselines/tables.txt) |
| 15 | Held out: lift 54% to 88%, group photo 4% to 8% | [`held_out_count.py`](still-scenes/hidden-situations-and-more-models/held_out_count.py) | [`held_out_count_out.txt`](still-scenes/hidden-situations-and-more-models/held_out_count_out.txt) |
| 15 | New situations: bent line 2% to 69%, machine 14% to 62% | [`new_situations_score.py`](still-scenes/new-situations/new_situations_score.py) | [`new_situations_score_out.txt`](still-scenes/new-situations/new_situations_score_out.txt) |
| 16, A8 | 229 crowd scenes, five VLMs: ours 7.4 points ahead of pointing with examples, and drops 23.4 points more when people are added | [`crowd_score.py`](still-scenes/crowds/vlm-run/crowd_score.py) | [`crowd_score_five_out.txt`](still-scenes/crowds/vlm-run/crowd_score_five_out.txt) |
| 16, A9 | Someone just behind a line: ours 3.5%, pointing with examples 18.7% | [`behind_line_split.py`](still-scenes/crowds/vlm-run/behind_line_split.py) | [`behind_line_split_out.txt`](still-scenes/crowds/vlm-run/behind_line_split_out.txt) |
| A10 | Stricter joining check: 1.3 m or more behind the line, 7% to 52% on a new seed. No one added, unchanged at 81% | [`fresh_score.py`](still-scenes/crowds/fresh-seed3/fresh_score.py), [`gate_replay.py`](still-scenes/crowds/joining-gate/gate_replay.py) | [`fresh_score_out.txt`](still-scenes/crowds/fresh-seed3/fresh_score_out.txt), [`gate_replay_out.txt`](still-scenes/crowds/joining-gate/gate_replay_out.txt) |
| A1 | Older and newer models: 12 of 13 pick more good spots with our method | [`layout_tests.py`](still-scenes/method-comparison/layout-level-tests/layout_tests.py) | [`layout_tests_out.txt`](still-scenes/method-comparison/layout-level-tests/layout_tests_out.txt) |
| A2 | The VLM finds the right target in 95% of tries | [`target_found.py`](still-scenes/target-found/target_found.py) | [`target_found_out.txt`](still-scenes/target-found/target_found_out.txt) |
| A3 | From an 8-frame tour, 839 of 840 tries pick a frame that shows the target | [`tour_score.py`](still-scenes/tour-grounding/tour_score.py) | [`tour_score_out.txt`](still-scenes/tour-grounding/tour_score_out.txt) |
| A4 | Each answer takes 1.2 to 1.8 times as long as pointing | [`answer_time.py`](still-scenes/speed/answer_time.py) | [`answer_time_out.txt`](still-scenes/speed/answer_time_out.txt) |
| A4, A7 | A full solve takes 130 ms | [`solver_time.py`](still-scenes/speed/solver_time.py) | [`solver_time_out.txt`](still-scenes/speed/solver_time_out.txt) |
| A4 | A small people finder saves 21 to 25% of the time on still scenes | [`people_finder_score.py`](still-scenes/people-finder/people_finder_score.py) | [`people_finder_score_out.txt`](still-scenes/people-finder/people_finder_score_out.txt) |
| A7 | The solver is right on all 229 crowd scenes with the true people and rules | | [`tables.txt`](still-scenes/crowds/score/tables.txt) |
| A6, A7 | Learned replacements for the solver: 67% and 57%, against the solver's 76% | | [soft density](still-scenes/learned-models/soft-density/tables.txt), [flow model](still-scenes/learned-models/flow-model-pilot/tables.txt) |
| A6 | Picking from numbered marks helps GLM by 22 good spots and costs DeepSeek 27 | | [GLM](still-scenes/numbered-spots/numbered_spots_glm_out.txt), [DeepSeek](still-scenes/numbered-spots/numbered_spots_deepseek_out.txt) |

The 85.1% is over the 1,660 tries with an answer from the model. Counting the 20 failed GLM calls as misses gives 84.0%.

## Moving lines

| Slide | Finding | Script | Stored output |
|---|---|---|---|
| 18 | Claude Sonnet: decide once by pointing 6%, ask once and track 29% | [`track_score.py`](moving-lines/track-vs-decide-once/track_score.py) | [`track_score_out.txt`](moving-lines/track-vs-decide-once/track_score_out.txt) |
| 18 | Asking again and again: 2 to 10%, with 22 to 29 calls against 4 | [`re_asking_score.py`](moving-lines/re-asking/re_asking_score.py) | [`re_asking_score_out.txt`](moving-lines/re-asking/re_asking_score_out.txt) |
| 19, 21, 22 | Hold-place rule on 64 untuned layouts: Sonnet 25% to 47%, Gemini 21% to 37%, DeepSeek 21% to 31%, Haiku 20% to 28% | [`hold_extended_score.py`](moving-lines/hold-place/hold_extended_score.py) | [`hold_extended_score_out.txt`](moving-lines/hold-place/hold_extended_score_out.txt) |
| 19, 22 | A robot given the ideal place every half second: 65% | [`ceiling_score.py`](moving-lines/hold-place/ceiling/ceiling_score.py) | [`ceiling_score_out.txt`](moving-lines/hold-place/ceiling/ceiling_score_out.txt) |
| 22 | With the robot's own camera in place of exact positions | [`own_camera_score.py`](moving-lines/own-camera/own_camera_score.py) | [`own_camera_score_out.txt`](moving-lines/own-camera/own_camera_score_out.txt) |
| A5 | Time closer than 0.55 m to someone: 2.3% for tracking, 7.1% and 10.2% for the others | [`bumps.py`](moving-lines/bumps/bumps.py) | [`bumps_out.txt`](moving-lines/bumps/bumps_out.txt) |
| A5 | Cutting in, per second near the line: 14.5% and 13.7% | [`cut_in_exposure.py`](moving-lines/cut-ins/cut_in_exposure.py) | [`cut_in_exposure_out.txt`](moving-lines/cut-ins/cut_in_exposure_out.txt) |
| A6 | A people finder in moving lines: DeepSeek 29% to 10% | [`moving_people_finder_score.py`](moving-lines/people-finder/moving_people_finder_score.py) | [`moving_people_finder_score_out.txt`](moving-lines/people-finder/moving_people_finder_score_out.txt) |
| A6 | Walking before the answer arrives: 27 runs worse, 12 better | | [`results_fresh_sonnet55_low.md`](moving-lines/updating-vs-once/results_fresh_sonnet55_low.md) |
| A6 | A rule-based fix for the front of the line: no change | [`front_score.py`](moving-lines/front-of-line-fix/front_score.py) | [`front_score_out.txt`](moving-lines/front-of-line-fix/front_score_out.txt) |

## Labels in the outputs

| Label | Meaning |
|---|---|
| `R`, `E`, `C` | Pointing, pointing with examples, our method |
| `cot`, `fsc`, `con` | The same three in run folder names |
| `R1`, `C1` | Decide once by pointing, decide once with our method |
| `Rq0`, `Cq` | Ask the VLM again during the run |
| `CT2` | Ask once and track the people |
| `CT3F` | Tracking, plus a check that people move with the line |
| `CT3FL` | Tracking, plus the hold-place rule |
| `OW` | A robot given the ideal place |
| `real`, `staffed`, `plain` | The three Blender versions of a scene |
| `W/L/T` | Wins, losses and ties, paired by layout or by scene and seed |
| `strict`, `looser` | The scoring rule in `SCORING_STRICT.md` and the wider first rule of the moving-line runs (`share_frozen` in the data) |
| `sN`, new or old wording | A batch of still-scene runs, and whether it used the final prompts |

The other moving-line arms are defined at the top of [`experiment_oracle.py`](../sim/sgi_habitat/experiment_oracle.py).

## Supporting analyses

- [`moving-lines/roles-vs-marked-spot/`](moving-lines/roles-vs-marked-spot/): remembering who to stand behind against following the marked spot.
- [`moving-lines/test-robots/`](moving-lines/test-robots/): test robots given the true serving spot, which show the largest loss is at the front of the line.
- [`moving-lines/where-time-is-lost/`](moving-lines/where-time-is-lost/) and [`moving-lines/test-layouts/`](moving-lines/test-layouts/): where time in the right spot is lost, and the layouts of the final moving-line test.
- [`moving-lines/cut-ins/`](moving-lines/cut-ins/): how often people cut in near the line.
- [`still-scenes/run-to-run-noise/`](still-scenes/run-to-run-noise/): the gap between methods against the noise between repeated runs.
- [`still-scenes/analysis-scripts/harness/`](still-scenes/analysis-scripts/harness/): the tool that solves stored answers again, used by the replay analyses.
- [`still-scenes/metrics-and-statistics/`](still-scenes/metrics-and-statistics/): the statistics that the layout tests compare with.

## Re-running a script

The scripts read stored run logs and make no model calls. The logs are in the release [`run-logs-v1`](https://github.com/LWZ19/social-goal-inference/releases/tag/run-logs-v1). From the repository root:

```bash
gh release download run-logs-v1 --repo LWZ19/social-goal-inference
tar xzf run-logs.tar.gz            # unpacks runs/ and results/data/moving/
PYTHONPATH=src python results/moving-lines/hold-place/hold_extended_score.py
```

`fresh_score.py`, `gate_replay.py`, `baselines.py`, `layout_tests.py` and `solver_time.py` also need the rendered scenes. The development scenes are in the release [`dev-scenes-v1`](https://github.com/LWZ19/social-goal-inference/releases/tag/dev-scenes-v1).
