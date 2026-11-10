# Results: start here

Every experiment behind the project, grouped by what it tests: the plan written before each run, the scoring script,
its output, the notes, and the run logs. A person or an AI agent can explore it, re-check any number, and see how it
was reached. For a plain-language overview with charts, open [`page/robot-stand.html`](page/robot-stand.html) in a
browser.

## The question

A robot gets a request ("get me a coffee") and a camera picture, and has to choose where to stand: join the end of a
waiting line, wait behind a group at a lift, stop at the edge of a conversation, stay out of a photo. A vision-language
model (VLM) reads the picture. Two ways of choosing are compared:

- **Pointing:** the VLM marks the spot in the picture (strategies `cot`, and `few_shot_cot` with worked examples;
  written R and E in the notes).
- **Constraint method (ours):** the VLM names the target, the people and the rule, for example "come after people
  4, 5 and 6"; a geometry solver (`src/sgi/solver.py`) finds the exact spot (strategy `con`; written C).

Two test beds:

1. **Still scenes:** realistic 3D scenes rendered in Blender, scored automatically against hand-written regions of
   good spots (`src/sgi/gt/`).
2. **Moving lines:** Habitat 3.0 with Spot and animated people (`sim/sgi_habitat/`). The front person is served
   every 12-25 s, walks off and rejoins at the back. The score is the **share of time in the right spot** over a
   92-second run: every second, is the robot 0.5-1.2 m behind the last person waiting, within 0.6 m of the line,
   facing the front and at least 0.55 m from everyone (or, once first, within 0.6 m of the serving spot). See
   [`moving-lines/design-and-scoring/`](moving-lines/design-and-scoring/).

## Main findings

| Finding | Numbers | Where |
|---|---|---|
| The constraint method picks good spots far more often than pointing, on every model | Claude Sonnet 5.5: 96% vs 71%; Claude Haiku 5.5: 95% vs 60%; Qwen 3.8: 90% vs 78% (of 252 tries each) | [`still-scenes/method-comparison/`](still-scenes/method-comparison/), [`../docs/results/`](../docs/results/) |
| It holds on 70 new scenes never used in development, on all six models | e.g. Claude Haiku at waiting lines: 92 of 100 vs 4 of 100 | [`still-scenes/fresh-scenes/`](still-scenes/fresh-scenes/) |
| It holds on situations kept hidden until the end, and on two new ones (a bent line, someone at a machine) | wins on all 12 layouts of both new tests over the six models together; GLM-5.3 is level on the machine test | [`still-scenes/hidden-situations-and-more-models/`](still-scenes/hidden-situations-and-more-models/), [`still-scenes/new-situations/`](still-scenes/new-situations/) |
| The gap between methods is larger than the run-to-run noise | over three repeats, our method leads pointing by +65 to +93 good spots (of 252) on DeepSeek and +24 to +38 on GLM-5.3 | [`still-scenes/run-to-run-noise/`](still-scenes/run-to-run-noise/) |
| A small people detector can replace one VLM question: faster on all six models, never less accurate | e.g. Qwen 107 → 83 s per answer | [`still-scenes/people-finder/`](still-scenes/people-finder/) |
| Moving lines: asking once and re-solving the spot as people move beats deciding once | 11 of 13 models; Claude Sonnet 29% vs 6% of the time in the right spot | [`moving-lines/updating-vs-once/`](moving-lines/updating-vs-once/), [`moving-lines/school-models/`](moving-lines/school-models/) |
| Re-solving beats asking the VLM again and again, with far fewer calls | Claude Sonnet 32% vs 2-10%, 4 calls vs 22-29 per run | [`moving-lines/re-asking/`](moving-lines/re-asking/) |
| Remembering who to stand behind beats following the marked spot | 87 of 125 runs better | [`moving-lines/roles-vs-marked-spot/`](moving-lines/roles-vs-marked-spot/) |
| Tracking people with the robot's own camera costs nothing measurable | mean change +0.4 points over 125 runs | [`moving-lines/own-camera/`](moving-lines/own-camera/) |
| The updating robot comes close to people least often | within 0.55 m of someone about 2% of the time, against 7-10% for the other methods | [`moving-lines/bumps/`](moving-lines/bumps/) |
| Holding the place once in line, and going straight to the serving spot when first, lifts the best version on every model | on 64 unseen layouts: Claude Sonnet 25% → 47%, Gemini 21% → 37%, DeepSeek 21% → 31%, Claude Haiku 20% → 28% | [`moving-lines/hold-place/`](moving-lines/hold-place/) |
| Before that, the biggest loss was at the front of the line | a test robot told the serving spot gained 4-13 points; a simpler rule-based fix did not | [`moving-lines/test-robots/`](moving-lines/test-robots/), [`moving-lines/front-of-line-fix/`](moving-lines/front-of-line-fix/) |

The full list, about 110 findings with their strength (conclusive, suggestive, quick probe) and their files, is
[`overview/EVIDENCE_AUDIT.md`](overview/EVIDENCE_AUDIT.md); the final round of checks is summarised in
[`overview/RESULTS_7NOV.md`](overview/RESULTS_7NOV.md). Every topic folder has its own `README.md`: the question, the
result, the files, and how to reproduce it.

## Layout

| Folder | What is in it |
|---|---|
| [`still-scenes/`](still-scenes/) | Everything on the still 3D scenes: the method comparison, fresh scenes, new and hidden situations, more models, the people finder, numbered spots, perception, the depth fix, facing, metrics and statistics, learned models, crowds, the confidence signal, combining models, speed, thinking levels, run-to-run noise, and the replay harness the analyses use. |
| [`moving-lines/`](moving-lines/) | Everything in the simulator: design and scoring, updating against deciding once, re-asking, roles, the robot's own camera, the people finder, the school's models, bystanders, test robots, the test layouts, the best version, the front-of-line fix, bumps, cut-ins, failure analysis and open questions. |
| [`overview/`](overview/) | The index of every finding, the summaries, and the second ideas round. |
| [`page/`](page/) | The plain-language results page and its pictures. |
| [`literature/`](literature/) | The related work found for both directions. |
| [`_lib/`](_lib/) | `paths.py` (where everything is; scripts import it) and `mcompare.py` (paired comparison of simulator runs). |
| [`logs/`](logs/) | The compressed run logs, data files and bulk (Git LFS). |
| `data/` | Created when the logs are unpacked: the simulator run folders. Not committed. |

## Run logs and data

```bash
git lfs install && git lfs pull                    # fetches results/logs/*.tar.gz
for a in moving-runs research-data bulk; do tar xzf results/logs/$a.tar.gz -C results; done
tar xzf results/logs/still-runs.tar.gz              # still-scene runs -> runs/ at the repository root
```

- **moving-runs:** every simulator run, unpacked to `results/data/moving/` (`runs`, `runs_i4`, `runs_i10`, ...). A run
  folder (`<scenario>/seed<N>/<arm>/`) holds `episode.json` (the setup, the VLM answer and its timing, every re-solve
  of the spot, and a per-second record of the robot, the line, the gap to the last person, the heading, whether it is
  in the right spot, and cut-ins) and one folder per VLM question (`request.json`, `answer.json`, `inference.txt`).
- **still-runs:** every still-scene run, unpacked to `runs/`: `config.yaml`, `prompts/` and `results.jsonl` (one line
  per try, with the chosen spot, the score and why it passed or failed).
- **research-data:** the data files of the analyses (json and jsonl), unpacked beside their notes in each topic folder.
- **bulk:** generated layout candidates, test sandboxes and run-console logs, unpacked back into their topic folders.
- **Development scenes:** the 89 scenes the October still-scene analyses replay (three realistic versions, three
  development seeds, early placeholder scenes) are a release download, not in git:
  `gh release download dev-scenes-v1 --repo LWZ19/social-goal-inference && tar xzf dev-scenes-v1.tar.gz` (unpacks to
  `data/dev_scenes/`). Held-out scenes are not included.
- **Not included:** camera pictures, depth images and videos (about 20 GB), and the scenes of the later tests (fresh
  scenes, new situations). The stored scores do not need them.

Run scripts from the repository root with `PYTHONPATH=src python <script>`; they find every location through
`results/_lib/paths.py`.

## How to read the notes

- **Each test was planned before its data were run** (`*_FROZEN.md`), with a pass rule ("bar"). Later changes are
  numbered amendments that give their reason and say whether they came before the totals were seen. Results that
  were not planned are marked post hoc or exploratory.
- **Held-out scenes** (the lift and photo situations, the machine-use region, and the held-out seeds) were not used
  while developing the method.
- `*_score.py` are scorers, `*_out.txt` their outputs, `RESULTS*.md` / `results_*.md` the write-ups.
- Comparisons are paired by scene and seed and reported as wins/losses/ties with a sign test, and as a mean
  difference with a 95% lower bound.
- The notes were cleaned for sharing (some planning text removed), and the original hash files were left out because
  cleaned files no longer match them. Older notes may name the earlier folder names (item numbers such as `i10` or
  `x1`); the glossary below maps them.
- **Frozen hash checks:** many scripts check the sha256 of their frozen plan or code before running. Those values
  refer to the original files; the copies here were cleaned (paths, wording), so those checks stop such scripts. The
  stored outputs are the record, and every script that could run here was re-run against its stored output.
- **Old folder names inside run settings and data:** run `config.yaml` files and some data files still name the
  original folders of the development scene sets. Those sets are in `data/dev_scenes/`, and the replay harness
  (`still-scenes/analysis-scripts/harness/`) maps the old names there.
- **Simulator scenarios:** the generated scenario folders `sim/scenarios/{sweep,pilot3,confirm4,i11}` are not
  committed; `moving-lines/test-layouts/select_{sweep,pilot,confirm,i11}.py` rebuild them byte for byte, so run those
  first before re-running moving-line scripts that read the line's shape.

## Glossary

**Models** (run-folder names): `sonnet55_low` Claude Sonnet 5.5; `claude_haiku55` Claude Haiku 5.5; `gemini_fast`
Gemini 3.8 Flash; `dsv41` DeepSeek V4.1 Flash; `soclaas_qwen38_low` Qwen 3.8; `soclaas_glm53` GLM-5.3-Flash; `x3`
Qwen 3.8 Flash Next. The free models run on the school's model gateway. Older Claude and Gemini models are in
`configs/default.yaml`.

**Moving-line versions (arms):**

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
| `CT2f`, `CT3Ff` | The front-of-line fix on `CT2` / `CT3F` |
| `CT4*` | A running probability of who is waiting, instead of rules |
| `CTd` | Tracking with the robot's own camera (people detector and tracker) |
| `CTe` | The detector's people given to the VLM instead of asking it who is there |
| `CTs` | Re-solve only when the tracks have settled |
| `CTv` | Re-solve on the first camera view instead of the floor map |
| `CT2m`, `Rm` | Start walking while the VLM is still answering |
| `RT`, `RTw` | Follow the VLM's marked spot with the tracked person nearest it, instead of the roles it named |
| `Rq0`, `Cq` | Ask the VLM again after every answer / every 10 s |

**Earlier item names** (in older notes): `i4` roles-vs-marked-spot; `i5` own-camera; `i9` school-models; `i10`
re-asking; `i11` best-version-new-layouts; `i12` fresh-scenes; `i13`/`E4` new-situations; `x1` test-robots; `x2`
bystanders/qwen; `x3` moving-lines/people-finder; `x4` run-to-run-noise; `x5` depth-fix/paid-models; `E2`
still-scenes/people-finder; `E3` hidden-situations-and-more-models; `marks` numbered-spots; `group4`
group-of-four-case; phase folders and `I01`-`I52` ideas of the October analyses are listed in
[`still-scenes/overview/`](still-scenes/overview/).

