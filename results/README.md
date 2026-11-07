# Results: start here

This folder holds the experiments behind the project: the test plans, the scoring scripts, every result table, and
the run logs. It is laid out so that a person or an AI agent can explore it, check any number, and explain how it was
reached.

For a plain-language overview with charts, open
[`2026-11-01-push/progress_page/robot-stand.html`](2026-11-01-push/progress_page/robot-stand.html) in a browser.

## The question

A robot gets a request ("get me a coffee") and a camera picture, and has to choose where to stand: join the end of a
waiting line, wait behind a group at a lift, stop at the edge of a conversation, stay out of a photo. A vision-language
model (VLM) reads the picture. Two ways of choosing are compared:

- **Pointing:** the VLM marks the spot in the picture (strategies `cot`, and `few_shot_cot` with worked examples;
  written R and E in the notes).
- **Constraint method (ours):** the VLM names the target, the people and the rule, for example "come after people
  4, 5 and 6"; a geometry solver (`src/sgi/solver.py`) finds the exact spot (strategy `con`; written C in the notes).

Two test beds:

1. **Still scenes:** realistic 3D scenes rendered in Blender, scored automatically against hand-written regions of
   good spots (`src/sgi/gt/`).
2. **Moving lines:** Habitat 3.0 with Spot and animated people (`sim/sgi_habitat/`). The front person is served
   every 12-25 s, walks off and rejoins at the back. The score is the **share of time in the right spot** over a
   92-second run: every second, is the robot 0.5-1.2 m behind the last person waiting, within 0.6 m of the line,
   facing the front and at least 0.55 m from everyone (or, once first, within 0.6 m of the serving spot). See
   `2026-11-01-push/moving/SCORING.md` and `SCORING_STRICT.md` (the "strict" scoring used in the results).

## Main findings

| Finding | Numbers | Where |
|---|---|---|
| The constraint method picks good spots far more often than pointing, on every model | Claude Sonnet 5.5: 96% vs 71%; Claude Haiku 5.5: 95% vs 60%; Qwen 3.8: 90% vs 78% (of 252 tries each) | `2026-11-01-push/EVIDENCE_AUDIT.md` row A1; `../docs/results/` |
| It holds on 70 new scenes never used in development | e.g. Haiku at waiting lines: 92 of 100 vs 4 of 100 | `2026-11-01-push/fill/i12_still/` |
| It holds on situations kept hidden until the end (lift, photo) and on two new ones (a bent line, someone at a machine) | wins on all 12 layouts of both new tests | `2026-11-01-push/e3_models/`, `2026-11-01-push/fill/i13_e4/`, `2026-11-01-push/e4/` |
| A small people detector can replace one VLM question: faster on all six models, never less accurate | e.g. Qwen 107 → 83 s per answer | `2026-11-01-push/e2/` |
| Moving lines: asking once and re-solving the spot as people move beats deciding once | 11 of 13 models; Sonnet 29% vs 6% of the time in the right spot | `2026-11-01-push/moving/`, `2026-11-01-push/e3_models/`, `2026-11-01-push/fill/i9_qwen_glm/` |
| Re-solving beats asking the VLM again and again, with far fewer calls | Sonnet 32% vs 2-10%, 4 calls vs 22-29 per run | `2026-11-01-push/fill/i10_reask/RESULTS_I10.md` |
| Remembering who to stand behind beats following the marked spot | 87 of 125 runs better | `2026-11-01-push/fill/i4_rtw/` |
| Tracking people with the robot's own camera costs nothing measurable | mean change +0.4 points over 125 runs | `2026-11-01-push/fill/i5_ctd/RESULTS_I5.md` |
| The biggest remaining loss is at the front of the line: where the counter serves | a test robot told the serving spot gains 4-13 points | `2026-11-01-push/fill/x1_ceilings/` |

The full list, about 110 findings with their strength (conclusive, suggestive, quick probe) and their files, is
[`2026-11-01-push/EVIDENCE_AUDIT.md`](2026-11-01-push/EVIDENCE_AUDIT.md). The latest results are in
[`2026-11-01-push/fill/RESULTS_7NOV.md`](2026-11-01-push/fill/RESULTS_7NOV.md) and the `RESULTS_*.md` file of each
`fill/` item.

**Still running:** GLM-5.3's runs for the fresh scenes (`fill/i12_still`), the two new tests (`fill/i13_e4`) and the
repeat-run noise check (`fill/x4_noise`). Their tables show the other five models.

## Layout

| Folder | What is in it |
|---|---|
| `2026-10-17-beyond-vlm/` | Still-scene analyses and ideas tested from 17 Oct: replays, ablations, statistics, perception, learned models, crowds. `REPORT.md` is the overview; `phase1*/`, `phase2/`, `phase3/` hold each step with its own README. |
| `2026-11-01-push/` | From 1 Nov: moving lines, numbered marks, the people detector, more models, new tests, and the final checks. `SUMMARY.md` covers 1-2 Nov; `EVIDENCE_AUDIT.md` indexes everything. |
| `2026-11-01-push/moving/` | Moving-line design, scoring, frozen test plans (`*_FROZEN.md`), deviations, and the run script `run_moving.py`. |
| `2026-11-01-push/fill/` | The final checks of 6-7 Nov, one folder per item (see the glossary). |
| `2026-11-01-push/progress_page/` | The plain-language results page and its pictures. |
| `logs/` | The run logs, compressed (Git LFS). |

## Run logs

```bash
git lfs install && git lfs pull                       # fetches results/logs/*.tar.gz
tar xzf results/logs/moving-runs.tar.gz -C results     # simulator runs -> results/2026-11-01-push/moving/runs*/
tar xzf results/logs/research-data.tar.gz -C results   # data files and other run folders of the analyses
tar xzf results/logs/still-runs.tar.gz                 # still-scene runs -> runs/
```

They extract to the paths the notes and scoring scripts refer to.

- **A simulator run** (`<scenario>/seed<N>/<arm>/`): `episode.json` holds the setup (scene, model, request, arm),
  the VLM answer with its timing, every re-solve of the spot (about twice a second), and a per-second record of where
  the robot is, who is in line, how many are ahead, the gap to the last person, the heading, whether it counts as in
  the right spot, and cut-ins. Each VLM question has a folder with `request.json`, `answer.json` (plan, people, rules,
  spot) and `inference.txt` (the same, readable).
- **A still-scene run** (`runs/<id>/`): `config.yaml` (model and settings), `prompts/` (the exact prompts), and
  `results.jsonl` (one line per try: scene, request, plan, chosen spot, score and why it passed or failed).
- **Not included:** camera pictures, depth images and videos (about 20 GB), and the rendered scene sets. Re-scoring
  still scenes from scratch (`python -m sgi.rescore`) needs the scene sets; the stored scores in `results.jsonl` do
  not.

## How to read the notes

- **Each test plan was written before its data were run** (`*_FROZEN.md`, or `I*_FROZEN.md` / `X*_FROZEN.md` in
  `fill/`), with a pass rule ("bar"). Later changes are numbered amendments that give their reason and say whether
  they came before the totals were seen. Results that were not planned are marked post hoc.
- **Held-out scenes** (the lift and photo situations, the machine-use region, and the held-out seeds) were not used
  while developing the method.
- `*_score.py` are the scorers, `*_score_out.txt` their outputs, `RESULTS*.md` / `results_*.md` the write-ups.
- Comparisons are paired by scene and seed and reported as wins/losses/ties with a sign test, and as a mean
  difference with a 95% lower bound.
- The notes have been cleaned for sharing: some planning text was removed, and the original hash files were left
  out because cleaned files no longer match them.

## Glossary

**Models** (run-folder names): `sonnet55_low` Claude Sonnet 5.5; `claude_haiku55` Claude Haiku 5.5; `gemini_fast`
Gemini 3.8 Flash; `dsv41` DeepSeek V4.1 Flash; `soclaas_qwen38_low` Qwen 3.8; `soclaas_glm53` GLM-5.3-Flash; `x3`
Qwen 3.8 Flash Next. The free models run on the school's model gateway. Older Claude and Gemini models are in
`configs/default.yaml`.

**Moving-line arms:**

| Arm | Meaning |
|---|---|
| `R1` / `C1` | Decide once: ask once (pointing / constraint method), walk to that spot, stay |
| `CT` | Keep updating, first version: ask once, then re-solve the spot on tracked people with no further VLM call |
| `CT2` | Improved updating: people who leave and rejoin, taking the place of the last person served, only stepping forward, the line's path |
| `CT2-vac`, `CT2-rat`, ... | `CT2` with one part switched off |
| `CT3` | `CT2` plus the bystander check: someone counts as waiting only if they step up with the line |
| `CT3F` | Best version: `CT3` with the check on everyone behind a step, and re-finding the line from the target |
| `CT3FT` | Test robot: `CT3F` told the true serving spot |
| `CTo`, `CTt`, `CT0` | Test robots: told the true waiting people / the true serving spot; one re-solve then frozen |
| `CT4*` | A running probability of who is waiting, instead of rules |
| `CTd` | Tracking with the robot's own camera (people detector and tracker) |
| `CTe` | The detector's people given to the VLM instead of asking it who is there |
| `CTs` | Re-solve only when the tracks have settled |
| `CTv` | Re-solve on the first camera view instead of the floor map |
| `CT2m`, `Rm` | Start walking while the VLM is still answering |
| `RT`, `RTw` | Follow the VLM's marked spot with the tracked person nearest it, instead of the roles it named |
| `Rq0`, `Cq` | Ask the VLM again after every answer / every 10 s |

**Final-check items** (`fill/`): `i4_rtw` roles against the marked spot; `i5_ctd` own camera; `i9_qwen_glm` Qwen and
GLM with longer runs; `i10_reask` re-asking; `i11_ct3f` best version on new layouts; `i12_still` 70 new still scenes;
`i13_e4` the bent line and machine tests; `x1_ceilings` test robots given part of the answer; `x2_bystanders_qwen`
the bystander check on Qwen; `x3_cte` the people detector in moving lines; `x4_noise` repeat-run noise; `x5_snap_paid`
the depth fix on Claude and Gemini.

## Questions worth asking

- Why does the robot leave its place at the front of the line in some runs? (`fill/x1_ceilings/`, and any
  `episode.json` where `ahead` drops to 0.)
- Which still-scene situations still fail with the constraint method, and is the cause the VLM's answer or the
  solver? (`results.jsonl` scores and `answer.json` rules.)
- How large is each model's run-to-run noise against the gaps between methods? (`fill/x4_noise/`.)
