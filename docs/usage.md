# Usage

The [front page](../README.md) gives the overview and the results. This page is the full reference.

**On this page:** [How the pipeline works](#how-the-pipeline-works) · [Setup](#setup) · [Commands](#commands) · [Switching models](#switching-models) · [Reading a run folder](#reading-a-run-folder) · [Run modes](#run-modes)

## How the pipeline works

`sgi` uses vision-language models to turn a vague human request and one RGB-D view of a scene into a socially appropriate navigation goal for a robot.

The input is a request such as "get me a coffee", optional context text, and one RGB-D observation with its camera and robot pose. The output is an ordered list of subtasks. For the navigation subtask whose target is in view, the output also holds a goal pose (x, y, θ) in the robot frame.

Five stages produce it:

- The **plan** stage splits the request into subtasks and points at each navigation target in the image.
- **Locate** marks a target that the plan left unmarked.
- The **context check** decides whether the people in view matter for where the robot stops.
- If they do not, the **standard goal** stops about a metre in front of the side of the target that the robot sees, facing it. It uses depth alone.
- If they do, the **point-marking baseline** has the model mark a point on the floor, and depth lifts it to the goal pose.

The scene format, for anyone who exports scenes, is in [`docs/scene_format.md`](scene_format.md).

The last step has a second method, the **constraint method**, chosen with `goal: constraints`. The people in view get numbers. A model states what they require of the place as constraints over those numbers, and a geometric solver finds the place. [Run the constraint method](#run-the-constraint-method) shows how to use it. [`docs/constraint_method.md`](constraint_method.md) explains the method, how to run it and its known failure cases.

## Setup

Python 3.12 is needed. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
[ -e .env ] || cp .env.example .env
.venv/bin/pytest -q
```

`.env` holds `GEMINI_API_KEY`, `OPENAI_API_KEY`, `SOCLAAS_API_KEY` and `ANTHROPIC_API_KEY`, and git ignores it. The third line creates it once, from the example. It leaves an existing `.env` alone, since that file may hold a real key, so the block can be run again. Fill in the key of each provider you are going to call.

A key is sent only to its provider's own endpoint. The shell's `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_GEMINI_BASE_URL` and Vertex switches are not followed. The key for SoCLaaS, an OpenAI-compatible model gateway, goes only to the address its model entries give in `configs/default.yaml`.

The tests and every mock run need no key. No test reads `.env` or opens a network connection. The test suite takes 7 to 10 minutes, most of it spent generating test scenes.

Nothing the shell holds for OpenAI goes to that address with the key. Every `openai_compatible` entry is called through the `openai` package. That package would send the contents of `OPENAI_CUSTOM_HEADERS`, `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` as request headers, and an `Authorization` line in the first of them would replace the key. So while the shell sets one of the three, a command on such an entry stops before any request, with one line that names the variable. To run the command, put `env -u` and that name in front of it.

## Commands

Every command runs from the repository root. `data/`, `runs/` and `.cache/` are git-ignored, so a fresh clone has none of them.

### Generate the test scenes

```bash
.venv/bin/python -m sgi.synth.generate --out data/placeholder
```

This writes 22 scenes to `data/placeholder/eval` and 8 to `data/placeholder/examples` in a minute or two. The scenes are the same every time. They form a small test set for trying the pipeline. Each scene has exact depth and ground truth, and holds:

- one target (a coffee stall, a ticket counter, a help desk, a print kiosk, a parcel on a table, a big screen, a lift, or a person in red),
- people, in one of six situations (nobody at the target, a waiting line, a conversation group, people watching a screen, a picture being taken, people waiting at a lift),
- a request that names the target and one that only says what is wanted,
- a tour of eight frames.

The in-context examples of the goal stage are taken from the scenes in `data/placeholder/examples`.

### Check a scene

```bash
.venv/bin/python -m sgi.check_scene docs/example_scene
```

This prints one verdict per check, twelve in all (`OK`, `WARNING` or `ERROR`), then each episode's derived `people_matter` and factors. It also writes `check.png` in the scene folder. That picture shows the image with the projected people, anchors and regions, beside a top-down plot. The exit code is 1 when a check is an error. The command needs no key and no simulator. It works on any scene folder in the format, generated or exported. [`docs/scene_format.md`](scene_format.md) says what each check catches.

### Run on the mock provider

```bash
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --model mock --name plumbing
```

The keyless path is `--model mock`, the `mock` entry of the model table in `configs/default.yaml`. The mock answers every point with the image centre, so the success rate this run prints means nothing. The run shows that the whole path works, from the scene check and the selection of examples through every stage to the scorer and the outputs. It writes `runs/<timestamp>_<name>/`.

Before it starts on the episodes, a run logs every warning of the scene check, with the scene and the check. Only an error stops it. On the generated test scenes no warning is expected. On an exported scene, look at every warning before reading the numbers.

- A warning from check 10 or 11 matters most. A map that does not agree with the depth image means every goal is tested for reachability against the wrong map.
- A warning from check 9 says that the standard goal fails in an episode with nobody near the target. That is a problem with the scene or the view, and it has nothing to do with the people. It has two causes. In both, the stop is taken on the robot's line of sight and ends up beside a target that is seen from the side.
  - The target's depth shows no flat upright surface round the target point. A person seen in profile is one such target.
  - The target has such a surface, but no place in front of it is clear, on floor the picture shows, and as wide as the robot. The place is hidden behind something, or lies at the edge of the picture or outside it. Or the robot stands so near that the picture ends on the target itself, not on floor.

With the camera of the generated test scenes, 1.5 m up and tilted 10 degrees down, that last case begins nearer than about 2.5 m from a place seen 45 degrees off its front, and 2.7 m from one seen 58 degrees off. The robots in those scenes stand 4.3 to 6.2 m from their targets.

### Run with a real model

```bash
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --model gemini_fast --name baseline_gemini
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --model openai_fast --name baseline_openai
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --model claude_haiku55 --name baseline_claude
```

`--model` puts every stage on one entry of the model table. [Switching models](#switching-models) lists the entries and the other ways to choose them. Each command needs its provider's key in `.env`. Without it, the command prints one line that names the missing variable and exits with code 2. Like the mock run, each writes `runs/<timestamp>_<name>/`.

Of the 22 scenes, 18 are of the development situations and 4 of the two held-out ones (a picture being taken, people waiting at a lift). A run as above evaluates the 18 and skips the 4. That is 36 episodes of up to three model calls each (plan, context check, goal). With `--final-test` it evaluates all 22, which is 44 episodes.

Answers are cached under `.cache/vlm`. The same command run again repeats only the calls that got no answer, and `--no-cache` asks every call afresh.

The exit code is 0 for a complete run. It is 1 when some record ended without a model answer, and the command should then be run again. It is 2 when the run was refused or stopped by a configuration, scene, missing-key or provider error. A run also stops with code 2 when the provider's allowance is used up. It keeps what it has done, and [Reading a run folder](#reading-a-run-folder) says how to go on.

### Smoke and calibrate

```bash
.venv/bin/python -m sgi.smoke --model gemini_fast
.venv/bin/python -m sgi.smoke --model gemini_fast --calibrate
```

Both make real requests and never use the cache.

The first command sends the first episode of the first scene in `data/placeholder/eval`, or of the folder given with `--scene`. It makes the plan call and then the goal call. A locate call comes between them when the plan left the target unmarked. For each call it prints five lines:

- the provider and model with their settings, and the size of the image as sent,
- the latency,
- the token counts,
- the keys of the raw answer, in the order the model wrote them,
- the parsed answer.

When those keys are not in the order of the schema that was sent, a line starting `WARNING:` stands under the fourth line. It names the call and the first object of the answer whose keys differ. A `result:` line follows with the status and the goal pose in the robot frame. The image with the target and the goal drawn on it is written to `runs/smoke.png`.

When the scene has a tour, as every generated scene does, one more plan call is made on the tour's first frame alone. It is reported on a `[tour]` line, with the same warning under it when its keys are out of order. A plan on a single tour frame is the one request whose schema holds the JSON-schema keyword `const`, so this call shows whether the provider accepts it. That makes three calls on a generated test scene, and four when locate runs.

The exit code is 0 when the result and the tour plan are `ok` and no answer drew that warning. It is 1 for any other status, for an answer with its keys in another order, and for a request the provider refused (one `error:` line). It is 2 for a missing key, an unknown entry, a missing scene or a provider whose allowance is used up.

The order matters because a model writes its answer key by key and can build only on what it has already written. The reasoning has to come before the point it argues for. So every schema lists its keys in the order they are to be written in. That order is also their sorted order. A provider that sorts the keys of an answer whatever the schema says (SoCLaaS does) and one that keeps the schema's order then write the same answer. The keys of each answer, in that order:

| Stage | Keys of the answer |
|---|---|
| plan | `reasoning`, `subtasks`; each subtask `description`, `target`, `target_point` (on a tour `tour_frame` in its place), `type` |
| locate | `reasoning`, `target_point` |
| context | `reasoning`, `relevant_people`, `stop_elsewhere` |
| goal | `analysis` with its parts `arrangement`, `expectations` and `placement`, then `goal_point`, `look_at_point` |
| people | `analysis`, `people`; each person `appearance`, `body_point`, `faced_point`, `facing` |
| constraints | `analysis` with the same three parts, then `constraints`; each constraint `because`, `kind`, `people`, `range_m` (null, or `at_least` and `at_most`), `target` |

The last two are the answers of the constraint method. The reasoning keys (`reasoning`, and `analysis` in the goal, people and constraints answers) appear only under `cot` and `few_shot_cot`. `look_at_point` appears only with `heading_mode: vlm_look_at`.

Three keys have another name in the stage outputs that `results.jsonl` records, because the name there would sort out of its place. `stop_elsewhere` is recorded as `people_matter`, `analysis` as the goal mark's `reasoning`, and `arrangement` as its `people`, each with the same meaning. A call's `raw_text` in that file is the answer as the model wrote it.

A `WARNING:` line on an entry means that its answers come in some other order, for example the point before the reasoning. `cot` and `few_shot_cot` then measure nothing on that entry.

With `--calibrate` the command asks the model to point at four targets, once in each point format. The four come from the place and object targets of every scene in the folder. A person target is skipped, and so is one whose known position is outside the image. Of the rest, the command takes the four whose known positions project farthest from the image centre. It lists them, one line each with that pixel and its distance from the centre. It then prints two lines per format, such as `xy_1000: median error 12.3 px (4 of 4 points answered)` and `xy_1000: image centre 161.0 px (the median error of the image centre over the points answered)`.

The error is the distance in pixels between the answer and the projection of the target's known position. The second line is what a model would score on the same points by answering the middle of the image every time. It costs no call.

A point that got no answer is listed under its format's two lines, with one of these reasons:

- `null answer` means the model said the target is not in the image.
- `parse_failed` comes with the validation error. A point outside the image means the format is wrong for the model.
- `vlm_error` means no answer came. Run the command again.

A calibration on the default folder is eight calls, and no picture is written.

An entry whose smaller median is not clearly below its image-centre number is not pointing at the targets. Run no baseline on it. The points are taken far from the centre to keep those two numbers apart. A target is large in the image, and its known position is one spot on it. A model that points at another part of the right target can therefore be some tens of pixels off. On a target the robot faces, the image centre is no further off than that. On the default folder the image centre scores about 160 px. `.venv/bin/python -m sgi.smoke --model mock --calibrate` needs no key. It shows the four points and that number before any model call.

Run the smoke command once for each real entry before its first use. Before any baseline number is reported, set that entry's `point_format` in `configs/default.yaml` to the format with the smaller median error.

### Run the constraint method

```bash
cp configs/default.yaml constraints.yaml      # then set goal: constraints under run:
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --config constraints.yaml --model mock --name constraints_plumbing
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --config constraints.yaml --model gemini_fast --name constraints_gemini
.venv/bin/python -m sgi.smoke --model gemini_fast --goal constraints --scene data/placeholder/eval/eval_08_waiting_line_4
```

The constraint method places a goal in the same episodes as the baseline, those where the context check says that the people matter. So the two are compared on the same episodes. It works in three steps.

1. **People.** Every person in view gets a floor position, a heading where it can be told, and a number. With `people: vlm`, the default, one model call of the stage `people` lists them on the camera image. Depth lifts each body point to the floor. The request holds nothing of an episode, so the episodes of a scene share one cached answer. With `people: oracle` the visible people of the ground truth are used instead, and no model is asked. That setting is for studying the next step alone. At most `max_people` are kept, those nearest the target, and they are numbered from the left of the image.
2. **Constraints.** One model call of the stage `constraints` is shown the request, a list of the people with distances measured from depth, and the image with the target ringed and each number in a yellow disc on its person. It answers with constraints that refer to people by number or to the target. There are six kinds: `keep_out_of_shared_space`, `keep_out_of_view`, `come_after`, `within_distance`, `visible_to` and `face`. With `constraints: none` this call is not made.
3. **The solver.** It takes the floor from depth. Without being told, it keeps the robot `standoff_m` from every person found and about that far from the target. Where depth shows a flat upright face round the target point, it also keeps the robot on the side of that face it saw. It turns each usable constraint into a field over the floor. It returns the best place, the heading there and up to `candidate_count` runners-up. It makes no model call. With `constraints: none` it runs on its own rules alone. A run with constraints has to be read beside that run, because what the model's constraints add is the difference between the two.

Like every run mode, the three settings go under `run:` in a copy of `configs/default.yaml`. They are `goal: constraints`, and with it `people` and `constraints`. [Run modes](#run-modes) lists their values. The two stages take a model entry and a strategy like the others, with `--stage-model people=ENTRY` or `--strategy constraints=zero_shot`. Left alone, both use the entry of the `goal` stage, `people` with `zero_shot` and `constraints` with `cot`. Their prompts are `prompts/people.md` and `prompts/constraints.md`. A run makes up to four model calls an episode, for the plan, the context check, the people and the constraints.

The run on `--model mock` needs no key and checks the path. The mock lists one person, at the image centre, and states one constraint about a number that nobody has. So the solver runs alone in every episode.

The `sgi.smoke` command is a check by eye. Make it before trusting a model's numbers under this method. It shows whether the model refers to people by the numbers drawn on them. It sends two real requests for the first episode of the scene, the people call and the constraints call. The target is taken from the ground truth, so no plan is asked for. It prints each call as a smoke run does, with the answer as the model wrote it. Under the people call comes the list of people as the model is then shown it. Under the constraints call comes each constraint with what the solver made of it: `used`, `unusable` with the reason, `superseded` (a second `face`) or `cut` (beyond `max_constraints`).

The image the constraints call was shown is written to `runs/smoke_numbered.png`. Compare the numbers in the answer with the numbers on the people. Name a scene with people in view with `--scene`, as the command above does, and run the check on a few such scenes. The scene the command takes by itself, the first of the generated test scenes, has nobody in view, so the check would say nothing.

[Reading a run folder](#reading-a-run-folder) says what a record of the method holds and what is measured of it. [`docs/constraint_method.md`](constraint_method.md) lists the method's known failure cases.

### Read a constraint run beside the solver alone

```bash
cp constraints.yaml solver_alone.yaml         # then set constraints: none under run: as well
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --config constraints.yaml --model gemini_fast --name constraints_gemini
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --config solver_alone.yaml --model gemini_fast --name solver_alone_gemini
```

To see what the constraints add, compare with the solver alone. Without being told, the solver does four things:

- it keeps the robot `standoff_m` from every person found,
- it stops the robot about that far from the target,
- where depth shows a face of the target, it keeps the robot on the side of that face it saw,
- among places that are otherwise equal, it takes the one nearest the standard goal.

So every run of the method is made and reported as a pair. Both runs use the same scenes, the same model entry and the same config, once as it is and once with `constraints: none`, the solver alone. Nothing but that one setting may differ between the two config files.

With the cache on, the second run asks for nothing new. Its plan, context-check and people calls repeat the requests of the first run and are replayed. It makes no constraints call. Both runs therefore place the goal in the same episodes, from the same people. Like every run on a real entry, the second run still reads the entry's key from `.env` before it starts.

Read the two `summary.csv` files row by row. Compare `success_rate` and its four components, in the `overall` row and in the row of each `situation`. What the model's constraints add is the difference between the two. In a row where the two runs agree, the successes are the solver's own. In the first run, `flag_no_usable_constraint_rate` is the share of records whose answer held no constraint the solver could use. In those records the solver ran alone in the first run as well.

Two more runs stand beside the pair.

- The baseline is the same command with no `--config`, or with `goal: direct_marking`. The method places a goal only where the baseline does, in the episodes where the context check says that the people matter. With the cache on and the same entry, the plan and the context check are the same answers in both runs.
- A run with `people: oracle` gives the constraints call the people of ground truth. It gives the upper bound of the constraints step.

Compare methods with `people: vlm`, where success and latency come from the same run.

### Score a finished run again

```bash
.venv/bin/python -m sgi.rescore runs/20261009-101500_baseline_gemini
.venv/bin/python -m sgi.synth.generate --out data/regenerated
.venv/bin/python -m sgi.rescore runs/20261009-101500_baseline_gemini --scenes data/regenerated/eval
```

`sgi.rescore` takes a run folder, here one with the kind of name `sgi.run` gives. It scores again what the run recorded, with no model call and no key. It loads the scenes the run used and derives their ground truth with the current code, under the thresholds the run's `config.yaml` records. It then scores every record of `results.jsonl` against that ground truth.

It writes `rescored/` inside the run folder and leaves everything else there as it was. `rescored/` holds `results.jsonl` and `summary.csv`, as a run writes them, and `note.txt`. The note says when the rescoring was done, at which commit of the code, from which scenes folder, and in how many records `success` changed. The last line printed is the one `sgi.run` prints, with that number after it. In a record, `score`, `hidden_relevant_people` and the context-check measures are worked out again. Everything else is carried over as it was recorded.

`--scenes` names the folder of the scenes, in place of the one `config.yaml` records. A record whose scene or episode is not in the folder stops the command with one `error:` line and exit code 2, and nothing is written.

A scene that carries its own acceptable regions keeps them, and the generated test scenes are written with theirs. So after changing a reference rule, generate the test scenes again before rescoring. Write them into their own folder or, as above, into another one that `--scenes` then names.

With people at the target, a goal is acceptable 0.5 to 1.2 m from the person the robot stands by:

- the last person of a waiting line, where the robot must not be ahead of them and has to face the way the line faces,
- the nearest member of a conversation group, outside the group's circle and on any side of it, facing the person asked for,
- a person nobody is with, in front of them.

The distances are those of "Where Should I Join? Robot Group Joining via Language-Guided Goal Prediction" (arXiv 2609.28467). A place or an object with nobody near keeps its region of 0.6 to 1.6 m. Section 7 of [`docs/scene_format.md`](scene_format.md) has the rules in full.

A region says where a goal is in place. Whether the robot can stand there is a component of `success` of its own. `in_region` and `reachable` are separate fields of a record's `score`. The scorer keeps the robot's centre 0.55 m from a person's. That is a person's radius of 0.25 m plus the robot's of 0.30 m, the defaults. So where a region is measured from a person, the part of it nearer to them than about 0.55 m is `in_region` and cannot succeed. A count of poses by the distances alone includes that strip, and so comes out higher there than a rescored run.

Changing a reference rule leaves the pictures as they are, so cached model answers still apply. The one exception is a goal request under `few_shot` or `few_shot_cot`. That request shows the goal point of each worked example, and those points follow the regions. So such a request is a new one to the cache. A worked example's goal point is the grid point deepest inside its region. Around a conversation group, that point can be behind the person asked for or across the group from them.

### Drive a robot among moving people

```bash
.venv/bin/python -m sgi.serve --model gemini_fast --goal constraints --port 8765
```

`sgi.serve` runs the pipeline as a local HTTP service. It is given one live RGB-D view, the camera, the robot's pose and a request. It answers with the goal pose of the plan's first navigation subtask. It takes `--model`, `--stage-model` and `--strategy` as `sgi.run` does. `sim/` holds a simulator built on it, with Spot in a Habitat 3.0 flat where five people walk about and queue at a kitchen counter. Habitat needs Python 3.9, so the simulator has a conda environment of its own. [`docs/habitat.md`](habitat.md) covers the setup, the known limits, and `sim/demo.sh`, which starts both.

## Switching models

A model entry is one line of the `models` table in `configs/default.yaml`. It gives a provider, a model ID, a `reasoning_effort` and a `point_format`. Each of the four stages (`plan`, `locate`, `context`, `goal`) names the entry it uses, and by default all four name `mock`. The two stages of the constraint method, `people` and `constraints`, may be named there in the same way. Left out, they use the entry of the `goal` stage. There are three ways to choose other entries.

```bash
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --model soclaas_qwen38 --name qwen38
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --model soclaas_qwen38 --stage-model context=soclaas_glm53_max --name qwen38_glm53
cp configs/default.yaml my_models.yaml      # then edit the entry names under stages:
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --config my_models.yaml --name my_models
```

1. **The whole run.** `--model ENTRY` puts every stage on one entry.
2. **One stage.** `--stage-model STAGE=ENTRY` puts one stage on an entry of its own. It may be given several times, and it is applied after `--model` wherever it stands on the line. The second command above uses one model to point at the target and mark the goal, and another to judge whether the people matter. A value that is not `STAGE=ENTRY`, an unknown stage or an unknown entry stops the command with one `error:` line and exit code 2. This happens before anything is written or any model is asked.
3. **A config file.** In a copy of `configs/default.yaml`, set `model` for each stage under `stages:` and pass the copy with `--config`. A new entry, for another model or another level of thinking, is one more line in the copy's `models` table.

`sgi.smoke` takes `--model` and `--stage-model` in the same way, and always reads `configs/default.yaml`. It never calls the `context` stage, so a `--stage-model` or a `--strategy` for that stage would change nothing. Either one stops the command with one `error:` line and exit code 2. The command needs a key only for the entry of a stage it calls. A smoke run calls `plan`, `locate` and `goal`, and with `--goal constraints` it calls `people` and `constraints` in their place. With `--calibrate` it calls `locate` alone and calibrates the entry of that stage. Whichever way is used, the run's `config.yaml` records the entry of every stage.

Two commands list what there is to choose from.

```bash
.venv/bin/python -m sgi.run --list-entries
.venv/bin/python -m sgi.smoke --model soclaas_qwen38 --list-models
```

`--list-entries` prints one line per entry of the model table. The line gives its name, provider, model ID, reasoning effort and point format. It ends with `key set` or `key missing` and the name of the entry's variable in `.env`, or with `no key needed`. The command prints no key, makes no request and runs nothing. With `--config` it lists the table of that file. `sgi.smoke --list-entries` prints the same lines for `configs/default.yaml`.

`--list-models` asks the provider of the `--model` entry which model IDs its key may use. It prints them sorted, one per line. It makes a real request to the provider, and no model is asked anything. It works for every provider, for a `gemini`, an `openai`, an `anthropic` and an `openai_compatible` entry alike. For an `openai_compatible` entry, a model ID must be one of those the service lists. The service may say more of a model than its ID, owner and date, such as an alias, prices or capabilities. The line then carries that after the ID, as compact JSON cut at 300 characters. The `mock` entry has nothing to list, and the command says so in one line.

| Provider | `provider` | Entries | Key in `.env` | State |
|---|---|---|---|---|
| Gemini | `gemini` | `gemini_fast`, `gemini_strong`, `gemini_pointing`; older models for comparison: `gemini3_flash`, `gemini31_flash_lite`, `gemini35_flash_lite`, `gemini36_flash` | `GEMINI_API_KEY` | Used in live calls and full runs (`gemini_fast`). |
| SoCLaaS | `openai_compatible` | the seventeen `soclaas_*` entries | `SOCLAAS_API_KEY` | A university gateway to open-weight models, in the OpenAI format. It needs a university key. Used in full runs. The prices are in the model table. |
| OpenAI | `openai` | `openai_fast`, `openai_strong` | `OPENAI_API_KEY` | Implemented and tested offline. Not used for the reported results. |
| Claude | `anthropic` | `claude_opus55`, `claude_opus55_low`, `claude_sonnet55`, `claude_sonnet55_low`, `claude_haiku55`, `claude_haiku55_low`; older models for comparison: `claude_haiku45`, `claude_sonnet46_low`, `claude_sonnet5_low` | `ANTHROPIC_API_KEY` | Anthropic's own API, used for live runs. The three older models are paced at 25 requests a minute. The prices are in the model table. |
| Mock | `mock` | `mock` | none | Answers the image centre. For plumbing runs and tests. |

An `openai_compatible` entry is for any service that speaks the OpenAI chat-completions format. Besides the fields every entry has, it gives the service's address as `base_url`. It also gives, as `api_key_env`, the name of the variable in `.env` that holds its key. No entry of another provider may set either. `structured_output` says how the answer's shape is asked for. It is `json_schema` (the default, which sends the schema in strict mode), `json_object`, or `none`. The answer is validated in every case.

`reasoning_effort` is sent as written. The level names a model accepts differ from model to model, and the comment above the `soclaas_*` entries in `configs/default.yaml` lists them. A level the model rejects stops the run with the provider's message. There is one entry per model and level worth comparing, such as `soclaas_qwen38` (thinking off), `soclaas_qwen38_low` and `soclaas_qwen38_medium`.

For an `anthropic` entry, `reasoning_effort` is the model's effort level, one of `low`, `medium`, `high`, `xhigh` or `max`. Thinking is adaptive, up to what the level allows, and it is billed as output. `none` is not one of those levels. It stands for thinking switched off. By Anthropic's documentation Claude Haiku 5.5 accepts it, and Claude Opus 5.5 and Claude Sonnet 5.5 refuse it, which stops the run with the provider's message. Such an entry may set neither `temperature` nor `image_detail`. The models refuse a sampling parameter, and the back end has nowhere to send either, so an entry that sets one is refused with an `error:` line that names it. An answer may be as long as `max_output_tokens`, thinking included, or 16000 tokens where the entry gives no number. The stage's schema asks for the shape of the answer. It is sent without the bounds of a point's coordinates, because the API takes no bound on a number, and the answer is still checked against those bounds. A request the model declines is recorded as refused, with the category the API gives. It is never handed to another model, because a run measures the model its entry names.

What has been measured with real models is in [`results/README.md`](../results/README.md).

### Strategies and prompt folders

A prompting strategy and the prompt texts are chosen per run, like a model.

```bash
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --model soclaas_qwen38 --strategy goal=few_shot_cot --name qwen38_few_shot_cot
cp -r prompts my_prompts      # then edit the files in it
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --model soclaas_qwen38 --prompts my_prompts --name qwen38_my_prompts
```

`--strategy STAGE=NAME` gives one stage its strategy. `plan`, `locate`, `context`, `people` and `constraints` take `zero_shot` or `cot`, and `goal` also takes `few_shot` and `few_shot_cot`. The option may be given several times and is applied after `--model` and `--stage-model`. The command stops with one `error:` line and exit code 2 on a value that is not `STAGE=NAME`, an unknown stage, an unknown strategy, or a strategy the stage does not accept. It stops before anything is written or any model is asked. In a config file the same setting is `strategy` for each stage under `stages:`.

`--prompts DIR` reads the stage prompts from `DIR` instead of the repository's `prompts/`. The files are `plan.md`, `locate.md`, `context.md` and `goal.md`. Under `goal: constraints` they also include `constraints.md` and, with `people: vlm`, `people.md`. A relative `DIR` is taken from the working directory.

A prompt file is a series of blocks. Each block is a line `## name` with its text under it. A file has to keep every block of the repository's file of the same name. Each block has to keep the `{placeholders}` of the repository's block, and no others.

The command stops as it does for a bad strategy when a file the run reads is missing, when a file lacks one of those blocks, or when a block has a placeholder too many or too few. A file the run does not read is not looked at, so a folder with the four baseline files stays valid for a baseline run. A `{lower_case_word}` in braces is read as a placeholder in every block, the system block included. So a folder whose text needs literal braces round a word is refused. The wording and answer keys inside a prompt are not checked.

In a config file the setting is `prompts_dir` under `run:`, relative to the repository root unless it is absolute.

`sgi.smoke` takes both options as well. It shows no examples, whatever the strategy. A run's `config.yaml` records the strategy of every stage and `prompts_dir`. The run folder keeps a copy of the files the run reads in `prompts/`, so a number can be traced to the text that produced it. The cache key covers the text that is sent. A changed prompt is asked afresh, and an unchanged one is replayed.

### Pacing, prices and cost

An entry may give `max_requests_per_minute`. All the clients of one provider and address in a command then share one pacer. The pacer lets at most that many requests start in any 60 seconds. A request beyond that waits for its turn, the first attempt and every retry alike. An answer from the cache never waits. The wait counts in a call's `latency_total_s`, and so in the latency columns of `summary.csv`. It does not count in the call's `latency_s`, which stays the provider's own time. Where entries of one address name different limits, the lowest holds. The SoCLaaS entries are set to 80 a minute. Each command paces only itself, so run commands one after another, not side by side, or lower the number.

An entry may also give `price_input_per_mtok` and `price_output_per_mtok`, in US dollars per million tokens. It gives both or neither. An entry with one of them alone is refused with an `error:` line that names it. The SoCLaaS entries carry the prices of the service's model list, and the Claude entries carry Anthropic's. The Gemini and OpenAI entries have none. A call costs its input tokens at the first price and its output and thinking tokens at the second.

`summary.csv` ends with three columns:

- `calls`, the model calls of the row's records,
- `cached_calls`, those answered from the cache,
- `cost_usd`, what the calls cost. It is empty when one of them used an entry without prices.

A call answered from the cache counts at what it cost when it was made. So `cost_usd` is the cost of a run's answers, which can be more than the run added to the bill. The last line a run prints carries the same numbers, as in `(28 records, 0 ended in vlm_error; 73 calls, 17 from the cache, about $0.04)`.

The cost is an estimate from the token counts the provider reports for the request that gave a call its answer. A call asked a second time, because its first answer was not valid, counts only that second request. Where that happened, the figure is a little low. Neither the limit nor a price is part of the cache key, so changing one keeps every cached answer.

## Reading a run folder

`runs/<timestamp>_<name>/` holds five outputs.

- `config.yaml` holds the exact configuration used, under `config`. That covers the model entries, the strategy of each stage, the thresholds and the `run` settings, among them `examples_dir`, `prompts_dir`, `trials`, `leave_out_situation` and `final_test`. Beside it are the run's `name`, its `scenes_dir`, `example_ids` (the in-context examples every episode was shown), `examples_missing` (the situations that needed such an example and had none, see [Run modes](#run-modes)), `skipped_held_out`, `vlm_errors` and `incomplete`. A run that the provider's allowance stopped also has `stopped_by`.
- `prompts/` is a copy of the stage prompts the run read. It holds four files, and under `goal: constraints` also `constraints.md` and, with `people: vlm`, `people.md`.
- `results.jsonl` has one record per episode and trial. In a record, `plan` is the plan as parsed. `result` is the goal result, with its `status`, `pose_robot`, the flags, the stage outputs and `calls`, which keeps every model call with its raw response text. Of the flags, `standard_goal_frontal` is set when a standard goal was taken in front of the surface the target point lies on. Without it, a standard goal stands on the robot's line of sight to the target. That happens when depth showed no flat upright surface round the target point, or no place in front of it that is clear, on floor the image shows, and as wide as the robot. Floor just under the picture counts as shown where the picture ends on floor above it. The six `surface_*` thresholds in `configs/default.yaml` say what counts as such a surface. `score` has `success` and its four components (`target_correct`, `reachable`, `in_region`, `heading_ok`) with the supplementary measures. `measures` has the context-check and decomposition measures. `labels` has the episode's request type, situation and factors, and `example_ids` the examples shown. A record of a tour-mode run also has `tour`.
- A result of the constraint method has `source: constraints` and four more fields, each null in any other result. `people` lists the people found. Each has its `number`, its floor position `xy` and heading `yaw` in the world frame, the body point `pixel` and, from a model, a `label`. `constraints` has one entry per constraint of the answer, in its order. Each entry holds the `constraint` as it was written, a `status` (`used`, `unusable`, `superseded` or `cut`), the `reason` when it is not used, and `broken_m`, by how much the place breaks it. `candidates` are the runners-up, each a `pose_world` with its `keys`. `solver` is the solver's note on the place. It holds the `keys` the place was chosen by, `own_space_broken_m` per person, `joined_waiting` (people taken as waiting whom the answer did not list), `not_ahead_of` (people who watch the target, whom the place is kept level with or behind), `not_ahead_reason` (why nobody is in that list, where the answer stated a view of the target) and `target_distance_m`. With these and the scene, a record can be solved again without a model call. Six flags belong to the method: `no_usable_constraint` (the solver ran alone, which is no failure), `constraint_unusable`, `constraints_cut`, `exclusion_broken` (every place broke an exclusion, and the one taken breaks the least), `people_cut` and `far_from_target`. A people call or a constraints call that brought no valid answer ends the result with that call's status and no pose. `goal_mark` is null. `goal_pixel` is the place as the image shows it, or null when it lies outside the image.
- Six measures compare the constraint method with ground truth. They are the last six entries of a record's `measures`. Each is null where the record does not measure it, and all six are null in a record of another method. `people_found_share` is the share of the people of ground truth in view who have a person found within twice `person_radius_m` of them, 0.5 m by default. The pairs are taken nearest first, and a person found counts for one person only. `people_unmatched` is the number of people found that are left over. Both are measured under `people: vlm` alone, and there is no share where nobody is in view. `members_precision` and `members_recall` are measured where the reference of the episode has people waiting their turn at the target and a model stated the constraints. `members_precision` is the share of the numbers the answer lists under `come_after` that are members of that group in view. `members_recall` is the share of those members that the answer lists. Both are read from the answer as it was written, before the solver takes anyone else as waiting, and a number that nobody has counts as listed and wrong. `constraints_unusable` is the number of constraints the solver could not use, `unusable` or `superseded`. `exclusions_broken` is the number of exclusions the place breaks. It counts the own space of each person the place is inside, each `keep_out_of_shared_space` and `keep_out_of_view` it breaks, and `come_after` once. The last two are measured for every result the solver made, the solver alone included.
- `summary.csv` starts with the `overall` row. One row follows per value of `request_type`, of `situation` and of each factor. The factors are `density`, `group_size` (binned as 0, 1-2, 3-5 and 6-10), `bystanders`, `other_groups` and `view_angle_deg` (up to 30 degrees, and above). An episode without a factor is counted under `unlabelled`. Each row gives `episodes`, `records`, `vlm_errors`, then `success_rate`, the rate of each success component and the supplementary measures. An empty cell means the row did not measure that column. With more than one trial, a measure is the mean over the trials. Its minimum and maximum across them follow, in the columns ending in `_min` and `_max`. The six measures of the constraint method are the last measure columns, under the names they have in a record. Each is the mean over the row's records that measured it, and all six are empty in every row of a baseline run. The six flags of the method have their rates among the other `flag_*_rate` columns. The last three columns, `calls`, `cached_calls` and `cost_usd`, are totals over the row's records and trials. [Pacing, prices and cost](#pacing-prices-and-cost) says how to read them.
- `viz/<scene>_<episode>.png` is one picture per episode. It shows the image with the target and the goal marked, beside a top-down plot of the people with their headings, the target, the acceptable and forbidden regions, the robot and the goal pose. For a result of the constraint method, both halves also show the number of each person found and the runners-up. The plot then also shades the floor the robot may stand on by the solver's order, and the darker a cell, the better the solver ranks it. That floor is solved again from the record, with the default thresholds.

A record whose status is `vlm_error` got no model answer at all. The provider was unreachable or kept failing through three attempts. Such episodes stay out of the success rate and of every other rate, and are counted in `vlm_errors`. A run with any of them is incomplete. `config.yaml` says `incomplete: true`, and the command exits with code 1. Run the same command again. The cache replays every call that was answered, and only the rest are asked.

A run also stops when the provider's allowance is used up. With an `openai` or an `openai_compatible` entry, a 429 whose message speaks of a quota or a budget is taken as a spent allowance. Waiting brings no answer, and every later call would fail the same way. The run stops at that call and makes no other. `results.jsonl` keeps the records finished before it, and `summary.csv` is written over those records. `config.yaml` says `incomplete: true` and gives the provider's message as `stopped_by`. The command prints one `error:` line with that message and exits with code 2. When there is allowance again, run the same command again. The cache replays every call that was answered, so the run goes on from where it stopped. A run made with `--no-cache` stored nothing and starts over. Any other 429 is taken for a rate limit and asked again. A `gemini` entry has no such stop, and each of its 429s is asked again.

A plan answer that could not be parsed is still the model's own answer. It counts as a wrong plan in the decomposition columns (`plan_exact_rate`, `navigate_recall` and `later_subtask_hit_rate`).

A `rescored/` folder inside a run folder is written by `sgi.rescore`, after the run. It holds the run's records scored again, as [Score a finished run again](#score-a-finished-run-again) says.

In `summary.csv`, the columns `latency_s` (per episode) and `latency_<stage>_s` (per stage, for example `latency_plan_s`) include failed attempts and backoff sleeps. Read them next to the row's `transient_retry_share`, the share of the row's model calls that needed a transient retry. That share counts the calls of `vlm_error` records too, is one value over all trials, and is empty when the row made no call. Quote a latency number together with the `model` and the `reasoning_effort` of the entry used. `config.yaml` records both under `config`, `models`. Per call, `results.jsonl` has `latency_total_s`, the per-query figure, with `transient_retries`, `model` and `reasoning_effort`. A call answered from the cache keeps the latencies of the request that was stored.

## Run modes

The harness can put a value derived from ground truth in place of a stage's output. Stage code never sees ground truth.

| Setting | Values | Meaning |
|---|---|---|
| `grounding` | `vlm`, `oracle` | `oracle` skips the plan stage and locate: the target point is the projection of the target's known position, lifted with depth like a model's point. |
| `context_check` | `vlm`, `oracle`, `always_yes`, `always_no` | `oracle` uses the reference answer: people matter exactly when the standard goal fails. `always_no` takes the standard goal in every episode. `always_yes` marks a point in every episode. |
| `goal` | `direct_marking`, `oracle`, `constraints` | `direct_marking` is the baseline: the model marks the goal. `oracle` marks the representative point of the matched target's acceptable region and takes the heading that region allows. `constraints` is the constraint method, a second method for the same step: the people in view are numbered, a model states constraints over them, and a solver finds the place. It is shown no worked examples. |
| `people` | `vlm`, `oracle` | Read under `goal: constraints` only: who finds the people. `vlm` asks a model, and `oracle` takes the visible people of the ground truth, with their true positions and headings and no model call. |
| `constraints` | `vlm`, `none` | Read under `goal: constraints` only: who states the constraints. `vlm` asks a model. With `none` no model is asked for them and the solver runs on its own rules alone: the run to read a run with constraints beside. |
| `grounding_source` | `view`, `tour` | `view` plans on the scene's observation. `tour` plans on the scene's tour frames, then locates every navigation target in the observation. It needs a scene with a tour and cannot be combined with `grounding: oracle`. |
| `trials` | integer, default 1 | Each episode is run this many times, with the trial number in the cache key. |
| `leave_out_situation` | a development situation, default unset | For example selection only, that situation is treated as held out. Its episodes are still evaluated. |
| `--final-test` | flag | Includes the episodes of held-out situations, which are skipped otherwise. Use it once, at the end. A held-out episode that has no acceptable region stops the run before it starts: no goal could succeed in it. |

The first eight are settings. They go under `run:` in a copy of `configs/default.yaml`, and the copy is passed to `sgi.run` with `--config`. `--final-test` is a flag of the command.

`examples_dir` is set the same way, and `--scenes` does not set it. The worked examples of the goal stage come from `run.examples_dir`. It is `data/placeholder/examples` by default, whatever folder `--scenes` names. To run delivered scenes, set `examples_dir` in the copy to the delivered `examples/` folder. Give an absolute path, or one relative to the repository root (not to the working directory). Then pass the copy with `--config`, beside `--scenes` and the delivered `eval/` folder. `example_ids` in the run's `config.yaml` shows which examples were used.

A situation needs a worked example when one of its evaluated episodes can reach the goal stage. The examples folder may hold no eligible episode of such a situation. The run then goes on. It logs one warning that names the situation, uses the examples there are, and lists the situation under `examples_missing` in `config.yaml`. That list is empty otherwise. With no eligible example at all, the run goes on under the same strategy without examples.

Under the default context check, an episode with nobody near the target is eligible only where the standard goal fails in it. The generated examples have one such scene (`examples_06_no_one_person`), so `no_one` is not listed there. In an examples folder without such a scene, `no_one` is the situation most likely to be listed. One setting still stops the run with an `error:` line. That is `k`, when it is set too small to reach a situation that needs an example and has one.

The model `qwen3.8:27b` accepts at most four images in a prompt, which is three worked examples and the camera view. For it, set `leave_out_situation` or `k: 3` under `run:`.

The oracle run is `grounding: oracle`, `context_check: always_yes` and `goal: oracle`. It makes no model call and has to score 100% on the generated test scenes. That score shows that the generator, the geometry and the scorer agree.

`context_check: always_no` is the lower bound. It is a robot that ignores the people and takes the standard goal everywhere.

Use one trial for a plumbing run and three or more for any number that is compared or reported.
