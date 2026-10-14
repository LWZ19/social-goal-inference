# Social goal inference pipeline

`sgi` turns a vague human request and one RGB-D view of a scene into a socially appropriate navigation goal for a robot, using vision-language models. What goes in is a request such as "get me a coffee", optional context text, and one RGB-D observation with its camera and robot pose. What comes out is an ordered list of subtasks and, for the navigation subtask whose target is in view, a goal pose (x, y, θ) in the robot frame. Five stages produce it. The **plan** stage splits the request into subtasks and points at each navigation target in the image. **Locate** marks a target that the plan left unmarked. The **context check** decides whether the people in view matter for where the robot stops. If they do not, the **standard goal** stops about a metre in front of the side of the target that the robot sees, facing it, from depth alone. If they do, the **point-marking baseline** has the model mark a point on the floor, and depth lifts it to the goal pose. The scene format, for anyone who exports scenes, is in [`docs/scene_format.md`](docs/scene_format.md).

That last step has a second method, the **constraint method**, chosen with `goal: constraints`: the people in view get numbers, a model states what they require of the place as constraints over those numbers, and a geometric solver finds the place. [Run the constraint method](#run-the-constraint-method) says how to use it. [`docs/constraint_method.md`](docs/constraint_method.md) is the page to read first: what the method does, how to run it against the baseline, its known limits, and what has and has not been measured.

Comments in the code that cite a "spec" or "the design" by section refer to the project's design documents, which are not part of this repository.

Numbers measured on placeholder scenes are not research results. They only show the pipeline works.

## Setup

Python 3.12 is needed. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
[ -e .env ] || cp .env.example .env
.venv/bin/pytest -q
```

`.env` holds `GEMINI_API_KEY`, `OPENAI_API_KEY`, `SOCLAAS_API_KEY` and `ANTHROPIC_API_KEY` and is git-ignored. The third line creates it once, from the example: an existing `.env`, which may hold a real key, is left alone, so the block can be run again. Fill in the key of each provider you are going to call. A key is sent only to its provider's own endpoint: `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_GEMINI_BASE_URL` and the Vertex switches of the shell are not followed. The SoCLaaS key goes only to the address its model entries give in `configs/default.yaml`. The tests and every mock run need no key: no test reads `.env` or opens a network connection. The test suite takes 7 to 10 minutes, most of it spent generating placeholder scenes.

Nothing the shell holds for OpenAI goes to that address with the key. The `openai` package, which every `openai_compatible` entry is called through, would send what `OPENAI_CUSTOM_HEADERS`, `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` hold as request headers, and an `Authorization` line in the first of them in place of the key. So while the shell sets one of the three, a command on such an entry stops with one line that names it, before any request. Put `env -u` and that name in front of the command to run it.

## Commands

Every command runs from the repository root. `data/`, `runs/` and `.cache/` are git-ignored, so a fresh clone has none of them.

### Generate the placeholder scenes

```bash
.venv/bin/python -m sgi.synth.generate --out data/placeholder
```

This writes 22 scenes to `data/placeholder/eval` and 8 to `data/placeholder/examples`, the same ones every time, in a minute or two. They stand in for the Isaac Sim scenes until those arrive. Each scene is a furnished room drawn by a small ray caster, with exact depth and ground truth. It has one target (a coffee stall, a ticket counter, a help desk, a print kiosk, a parcel on a table, a big screen, a lift, or a person in red), people drawn with bodies and faces, one of six situations (nobody at the target, a waiting line, a conversation group, people watching a screen, a picture being taken, people waiting at a lift), a request that names the target and one that only says what is wanted, and a tour of eight frames. `data/placeholder/examples` holds the scenes that the in-context examples of the goal stage are taken from.

### Check a scene

```bash
.venv/bin/python -m sgi.check_scene docs/example_scene
```

This prints one verdict per check, twelve in all (`OK`, `WARNING` or `ERROR`), then each episode's derived `people_matter` and factors. It writes `check.png` in the scene folder: the image with the projected people, anchors and regions drawn on it, beside a top-down plot. The exit code is 1 when a check is an error. The command needs no key and no simulator, and it works on any scene folder in the format, generated or exported. [`docs/scene_format.md`](docs/scene_format.md) says what each check catches.

### Run on the mock provider

```bash
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --model mock --name plumbing
```

The keyless path is `--model mock`: the `mock` entry of the model table in `configs/default.yaml`. This is what the spec calls `--provider mock`; no command here has a `--provider` option. The mock answers every point with the image centre, so the success rate this run prints means nothing. What the run shows is that the whole path works: the scene check, the selection of examples, every stage, the scorer and the outputs. It writes `runs/<timestamp>_<name>/`.

A run logs every warning of the scene check, with the scene and the check, before it starts on the episodes; only an error stops it. On the placeholder scenes no warning is expected. Any warning on an exported scene is worth a look before the numbers are read, above all one from check 10 or 11: a map that does not agree with the depth image means every goal is tested for reachability against the wrong map. A warning from check 9 says that the standard goal fails in an episode with nobody near the target, which is a matter of the scene or of the view and not of people. It has two causes, and in both the stop is taken on the robot's line of sight and ends up beside a target that is seen from the side. One is a target whose depth shows no flat upright surface round the target point (a person seen in profile is one). The other is a target that has such a surface but no place in front of it that is clear and on floor the picture shows, as wide as the robot is: the place is hidden behind something, lies at the edge of the picture or beside it, or the robot stands so near that the picture ends on the target itself and not on floor. With the camera of the placeholder scenes, 1.5 m up and tilted 10 degrees down, that last case begins nearer than about 2.5 m from a place seen 45 degrees off its front and 2.7 m from one seen 58 degrees off; the placeholder robots stand 4.3 to 6.2 m from their targets.

### Run with a real model

```bash
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --model gemini_fast --name baseline_gemini
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --model openai_fast --name baseline_openai
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --model claude_haiku55 --name baseline_claude
```

`--model` puts every stage on one entry of the model table; [Switching models](#switching-models) has the other ways to choose entries, and the entries there are. Each command needs its provider's key in `.env`; without it the command prints one line that names the missing variable and exits with code 2. Like the mock run, each writes `runs/<timestamp>_<name>/`.

Of the 22 scenes, 18 are of the development situations and 4 of the two held-out ones (a picture being taken, people waiting at a lift). A run as above evaluates the 18 and skips the 4: that is 36 episodes of up to three model calls each (plan, context check, goal). With `--final-test` it evaluates all 22, which is 44 episodes. Answers are cached under `.cache/vlm`, so the same command run again repeats only the calls that got no answer; `--no-cache` asks every call afresh. The exit code is 0 for a complete run, 1 when some record ended without a model answer (run the command again), and 2 when the run was refused or stopped by a configuration, scene, missing-key or provider error. A run also stops with code 2 when the provider's allowance is used up. It keeps what it has done, and [Reading a run folder](#reading-a-run-folder) says how to go on.

### Smoke and calibrate

```bash
.venv/bin/python -m sgi.smoke --model gemini_fast
.venv/bin/python -m sgi.smoke --model gemini_fast --calibrate
```

Both make real requests and never use the cache.

The first command sends the first episode of the first scene in `data/placeholder/eval`, or of the folder given with `--scene`. It makes the plan call and then the goal call, with a locate call between them when the plan left the target unmarked. For each call it prints five lines: the provider and model with their settings and the size of the image as sent; the latency; the token counts; the keys of the raw answer in the order the model wrote them; and the parsed answer. When those keys are not in the order of the schema that was sent, a line starting `WARNING:` stands under the fourth, naming the call and the first object of the answer whose keys differ. A `result:` line follows with the status and the goal pose in the robot frame, and the image with the target and the goal drawn on it is written to `runs/smoke.png`. When the scene has a tour, as every generated scene does, one more plan call is made on the tour's first frame alone and reported on a `[tour]` line, with the same warning under it when its keys are out of order. A plan on a single tour frame is the one request whose schema holds the JSON-schema keyword `const`, so this call shows whether the provider accepts it. That is three calls on a placeholder scene, and four when locate runs. The exit code is 0 when the result and the tour plan are `ok` and no answer drew that warning, 1 for any other status, for an answer with its keys in another order and for a request the provider refused (one `error:` line), and 2 for a missing key, an unknown entry, a missing scene or a provider whose allowance is used up.

The order matters because a model writes its answer key by key and can build only on what it has already written: the reasoning has to come before the point it argues for. Every schema therefore lists its keys in the order they are to be written in, and that order is also their sorted order, so that a provider which sorts the keys of an answer whatever the schema says (SoCLaaS does) and one which keeps the schema's order write the same answer. The answers, in that order:

| Stage | Keys of the answer |
|---|---|
| plan | `reasoning`, `subtasks`; each subtask `description`, `target`, `target_point` (on a tour `tour_frame` in its place), `type` |
| locate | `reasoning`, `target_point` |
| context | `reasoning`, `relevant_people`, `stop_elsewhere` |
| goal | `analysis` with its parts `arrangement`, `expectations` and `placement`, then `goal_point`, `look_at_point` |
| people | `analysis`, `people`; each person `appearance`, `body_point`, `faced_point`, `facing` |
| constraints | `analysis` with the same three parts, then `constraints`; each constraint `because`, `kind`, `people`, `range_m` (null, or `at_least` and `at_most`), `target` |

The last two are the answers of the constraint method. The reasoning keys (`reasoning`, and `analysis` in the goal, people and constraints answers) are there only under `cot` and `few_shot_cot`, and `look_at_point` only with `heading_mode: vlm_look_at`. Three keys have another name in the answer than in the stage outputs that `results.jsonl` records, because the name there would sort out of its place: `stop_elsewhere` is recorded as `people_matter`, `analysis` as the goal mark's `reasoning`, and `arrangement` as its `people`, each with the same meaning. A call's `raw_text` in that file is the answer as the model wrote it. A `WARNING:` line on an entry means that its answers are written in some other order, for example the point before the reasoning: `cot` and `few_shot_cot` then measure nothing on that entry.

With `--calibrate` the command asks the model to point at four targets, once in each point format. The four are taken from the place and object targets of every scene in the folder (a person target, or one whose known position is outside the image, is skipped): they are the ones whose known positions project farthest from the image centre. The command lists them, one line each with that pixel and its distance from the centre, and then prints two lines per format, such as `xy_1000: median error 12.3 px (4 of 4 points answered)` and `xy_1000: image centre 161.0 px (the median error of the image centre over the points answered)`. The error is the distance in pixels between the answer and the projection of the target's known position. The second line is what a model would score on the same points by answering the middle of the image every time; it costs no call. A point that got no answer is listed under its format's two lines with the reason: `null answer` (the model said the target is not in the image), `parse_failed` with the validation error (a point outside the image means the format is wrong for the model), or `vlm_error` (no answer came; run the command again). A calibration on the default folder is eight calls, and no picture is written.

An entry whose smaller median is not clearly below its image-centre number is not pointing at the targets, and no baseline should be run on it. The points are taken far from the centre to keep those two numbers apart. A target is large in the image and its known position is one spot on it, so a model that points at another part of the right target can be some tens of pixels off; on a target the robot faces, the image centre is no further off than that. On the default folder the image centre scores about 160 px. `.venv/bin/python -m sgi.smoke --model mock --calibrate` needs no key and shows the four points and that number before any call is paid for.

Run the smoke command once for each real entry before its first use, and set that entry's `point_format` in `configs/default.yaml` to the format with the smaller median error before any baseline number is reported.

### Run the constraint method

```bash
cp configs/default.yaml constraints.yaml      # then set goal: constraints under run:
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --config constraints.yaml --model mock --name constraints_plumbing
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --config constraints.yaml --model gemini_fast --name constraints_gemini
.venv/bin/python -m sgi.smoke --model gemini_fast --goal constraints --scene data/placeholder/eval/eval_08_waiting_line_4
```

The constraint method places the goal where the baseline does, in the episodes where the context check says that the people matter, so the two are compared on the same episodes. It works in three steps.

1. **People.** Every person in view gets a floor position, a heading where it can be told, and a number. With `people: vlm`, the default, one model call of the stage `people` lists them on the camera image, and depth lifts each body point to the floor. Nothing of an episode is in that request, so the episodes of a scene share one cached answer. With `people: oracle` the visible people of the ground truth are taken in its place and no model is asked: this is for studying the next step alone. At most `max_people` are kept, those nearest the target, and they are numbered from the left of the image.
2. **Constraints.** One model call of the stage `constraints` is shown the image with the target ringed and each number in a yellow disc on its person, the request, and a list of the people with distances measured from depth. It answers with constraints of six kinds that refer to people by number or to the target: `keep_out_of_shared_space`, `keep_out_of_view`, `come_after`, `within_distance`, `visible_to` and `face`. With `constraints: none` this call is not made.
3. **The solver.** It takes the floor from depth, keeps the robot `standoff_m` from every person found, about that far from the target and, where depth shows a flat upright face round the target point, on the side of that face it saw, all without being told. It turns each usable constraint into a field over the floor and returns the best place, the heading there and up to `candidate_count` runners-up. It makes no model call. With `constraints: none` it runs on its own rules alone, and that run is what a run with constraints has to be read beside: what the model's constraints add is the difference between the two.

The three settings go under `run:` in a copy of `configs/default.yaml`, as every run mode does: `goal: constraints`, and with it `people` and `constraints`; [Run modes](#run-modes) lists their values. The two stages take a model entry and a strategy like the others, with `--stage-model people=ENTRY` or `--strategy constraints=zero_shot`. Left alone, both use the entry of the `goal` stage, `people` with `zero_shot` and `constraints` with `cot`. Their prompts are `prompts/people.md` and `prompts/constraints.md`. A run is up to four model calls an episode: plan, context check, people and constraints.

The run on `--model mock` is the keyless one. The mock points at the image centre for the one person it lists and states one constraint about a number that nobody has, so the solver runs alone in every episode: the run shows that the path works and nothing else.

The `sgi.smoke` command is the check to make by eye before a model's numbers under this method are trusted: that the model refers to people by the numbers drawn on them. It sends two real requests for the first episode of the scene, the people call and the constraints call, with the target taken from the ground truth, so no plan is asked for. It prints each call as a smoke run does, with the answer as the model wrote it. Under the people call comes the list of people as the model is then shown it, and under the constraints call each constraint with what the solver made of it: `used`, `unusable` with the reason, `superseded` (a second `face`) or `cut` (beyond `max_constraints`). The image the constraints call was shown is written to `runs/smoke_numbered.png`: compare the numbers in the answer with the numbers on the people. Name a scene with people in view with `--scene`, as the command above does, and run it on a few of them: the scene the command takes by itself, the first of the placeholder set, has nobody in view, and the check would say nothing.

[Reading a run folder](#reading-a-run-folder) says what a record of the method holds and what is measured of it. [`docs/constraint_method.md`](docs/constraint_method.md) has the method's known limits: layouts in which the place is wrong and no flag says so.

### Read a constraint run beside the solver alone

```bash
cp constraints.yaml solver_alone.yaml         # then set constraints: none under run: as well
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --config constraints.yaml --model gemini_fast --name constraints_gemini
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --config solver_alone.yaml --model gemini_fast --name solver_alone_gemini
```

A success of a run with constraints is not yet a success of the constraints. The solver keeps the robot `standoff_m` from every person found, stops it about that far from the target, keeps it on the side of the target's face it saw where depth shows such a face and, among places that are otherwise equal, takes the one nearest the standard goal, all without being told. So every run of the method is made as a pair and reported as a pair: the same scenes, the same model entry and the same config, once as it is and once with `constraints: none`, the solver alone. Nothing but that one setting may differ between the two config files.

With the cache on, the second run asks for nothing new. Its plan, context-check and people calls are the requests of the first run and are replayed, and it makes no constraints call. Both runs therefore place the goal in the same episodes, from the same people. The second run still reads the entry's key from `.env` before it starts, as every run on a real entry does.

Read the two `summary.csv` files row by row: `success_rate` and its four components, in the `overall` row and in the row of each `situation`. What the model's constraints add is the difference between the two. In a row where the two runs agree, the successes are the solver's own. In the first run, `flag_no_usable_constraint_rate` is the share of records in which the answer held no constraint the solver could use: in those the solver ran alone in the first run as well.

Two more runs stand beside the pair. The baseline is the same command with no `--config`, or with `goal: direct_marking`: the method places a goal only where the baseline does, in the episodes where the context check says that the people matter, and with the cache on and the same entry the plan and the context check are the same answers in both. And `people: oracle` gives the constraints call the people of ground truth: that run is the upper bound of the constraints step, not a result of the method, and its latency is not the method's. The comparison that counts uses `people: vlm`, where success and latency come from the same run.

### Score a finished run again

```bash
.venv/bin/python -m sgi.rescore runs/20261009-101500_baseline_gemini
.venv/bin/python -m sgi.synth.generate --out data/regenerated
.venv/bin/python -m sgi.rescore runs/20261009-101500_baseline_gemini --scenes data/regenerated/eval
```

`sgi.rescore` takes a run folder, here one with the kind of name `sgi.run` gives, and scores what the run recorded again, with no model call and no key. It loads the scenes the run used, derives their ground truth with the code as it is now, under the thresholds the run's `config.yaml` records, and scores every record of `results.jsonl` against it. It writes `rescored/` inside the run folder and leaves everything else there as it was: `results.jsonl` and `summary.csv` as a run writes them, and `note.txt`, which says when this was done, at which commit of the code, from which scenes folder, and in how many records `success` changed. The last line printed is the one `sgi.run` prints, with that number after it. In a record, `score`, `hidden_relevant_people` and the context-check measures are worked out again; everything else is carried over as it was recorded.

`--scenes` names the folder of the scenes, in place of the one `config.yaml` records. A record whose scene or episode is not in the folder stops the command with one `error:` line and exit code 2, and nothing is written.

A scene that carries its own acceptable regions keeps them, and the placeholder scenes are written with theirs. So after a change of the reference rules, generate the placeholder scenes again before rescoring, into their own folder or, as above, into another one that `--scenes` then names. This is how a run made before 9 October 2026 is brought to the regions of that day. With people at the target, a goal is now acceptable 0.5 to 1.2 m from the person the robot stands by: the last person of a waiting line, where it must not be ahead of them and has to face the way the line faces; the nearest member of a conversation group, outside the group's circle and on any side of it, facing the person asked for; and a person nobody is with, in front of them. The distances are those of "Where Should I Join? Robot Group Joining via Language-Guided Goal Prediction" (arXiv 2609.28467, 2026). A place or an object with nobody near keeps its region of 0.6 to 1.6 m. Section 7 of [`docs/scene_format.md`](docs/scene_format.md) has the rules in full.

A region says where a goal is in place. Whether the robot can stand there is a component of `success` of its own: `in_region` and `reachable` are separate fields of a record's `score`. The scorer keeps the robot's centre 0.55 m from a person's, which is a person's radius of 0.25 m and the robot's of 0.30 m, the defaults. So where a region is measured from a person, the part of it nearer to them than about 0.55 m is `in_region` and cannot succeed. A count of poses by the distances alone takes that strip as well, and comes out higher there than a rescored run.

That change left every picture of the placeholder set as it was, so the cached answers of the models still belong to the scenes generated again, with one exception. A goal request under `few_shot` or `few_shot_cot` shows the goal point of each worked example, and those points moved with the regions, so with the examples generated again such a request is a new one to the cache. Look at what the examples show before a few-shot run on them. A worked example's goal point is the grid point deepest inside its region. In the band round a conversation group that can be behind the person asked for, or across the group from them, while the reasoning text of those examples says to stand just outside the group, facing that person, and names no side. The examples generated before the change are no way round this: the goal points of their waiting lines and groups lie outside the new regions.

## Switching models

A model entry is one line of the `models` table in `configs/default.yaml`: a provider, a model ID, a `reasoning_effort` and a `point_format`. Each of the four stages (`plan`, `locate`, `context`, `goal`) names the entry it uses, and by default all four name `mock`. The two stages of the constraint method, `people` and `constraints`, may be named there in the same way; left out, they use the entry of the `goal` stage. There are three ways to choose other entries.

```bash
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --model soclaas_qwen38 --name qwen38
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --model soclaas_qwen38 --stage-model context=soclaas_glm53_max --name qwen38_glm53
cp configs/default.yaml my_models.yaml      # then edit the entry names under stages:
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --config my_models.yaml --name my_models
```

1. **The whole run.** `--model ENTRY` puts every stage on one entry.
2. **One stage.** `--stage-model STAGE=ENTRY` puts one stage on an entry of its own. It may be given several times and is applied after `--model`, wherever it stands on the line: the second command points at the target and marks the goal with one model and has another judge whether the people matter. A value that is not `STAGE=ENTRY`, an unknown stage and an unknown entry each stop the command with one `error:` line and exit code 2, before anything is written or any model is asked.
3. **A config file.** In a copy of `configs/default.yaml`, set `model` for each stage under `stages:` and pass the copy with `--config`. A new entry, for another model or another level of thinking, is one more line in the copy's `models` table.

`sgi.smoke` takes `--model` and `--stage-model` in the same way, and always reads `configs/default.yaml`. It never calls the `context` stage, so a `--stage-model` or a `--strategy` for that stage would change nothing: it stops the command with one `error:` line and exit code 2. The command needs a key only for the entry of a stage it calls. A smoke run calls `plan`, `locate` and `goal`, and with `--goal constraints` it calls `people` and `constraints` in their place. With `--calibrate` it calls `locate` alone, and the entry of that stage is the one it calibrates. Whichever way is used, the run's `config.yaml` records the entry of every stage.

Two commands list what there is to choose from.

```bash
.venv/bin/python -m sgi.run --list-entries
.venv/bin/python -m sgi.smoke --model soclaas_qwen38 --list-models
```

`--list-entries` prints one line per entry of the model table: its name, provider, model ID, reasoning effort and point format, then `key set` or `key missing` with the name of the entry's variable in `.env`, or `no key needed`. It prints no key, makes no request and runs nothing; with `--config` it lists the table of that file, and `sgi.smoke --list-entries` prints the same lines for `configs/default.yaml`.

`--list-models` asks the provider of the `--model` entry which model IDs its key may use and prints them, sorted, one per line. It is a real request, and no model is asked. It works for every provider: a `gemini`, an `openai`, an `anthropic` and an `openai_compatible` entry alike. For an `openai_compatible` entry, a model ID must be one of those the service lists. Where the service says more of a model than its ID, owner and date (an alias, prices, capabilities), the line carries that after the ID, as compact JSON cut at 300 characters. The `mock` entry has nothing to list, and the command says so in one line.

| Provider | `provider` | Entries | Key in `.env` | State |
|---|---|---|---|---|
| Gemini | `gemini` | `gemini_fast`, `gemini_strong`, `gemini_pointing` | `GEMINI_API_KEY` | Confirmed by a few live calls, on the flash model (`gemini_fast`) only. The key is paid. |
| SoCLaaS | `openai_compatible` | the fifteen `soclaas_*` entries | `SOCLAAS_API_KEY` | The school's service for open-weight models, internal to the school. Confirmed by full runs on 2026-10-08. A key is allowed 90 requests a minute and has a daily and a monthly allowance. The prices are in the model table. |
| OpenAI | `openai` | `openai_fast`, `openai_strong` | `OPENAI_API_KEY` | Built and tested offline, never called: there is no key. |
| Claude | `anthropic` | `claude_opus55`, `claude_opus55_low`, `claude_sonnet55`, `claude_sonnet55_low`, `claude_haiku55`, `claude_haiku55_low` | `ANTHROPIC_API_KEY` | Anthropic's own API. Built and tested offline, never called: there is no key. No request has confirmed a model ID, a level or the shape of an answer. The prices are in the model table. |
| Mock | `mock` | `mock` | none | Answers the image centre. For plumbing runs and tests. |

An `openai_compatible` entry is for any service that speaks the OpenAI chat-completions format. Besides the fields every entry has, it gives the service's address as `base_url` and, as `api_key_env`, the name of the variable in `.env` that holds its key; no entry of another provider may set either. `structured_output` says how the answer's shape is asked for: `json_schema` (the default: the schema is sent, strict), `json_object`, or `none`. The answer is validated in every case.

`reasoning_effort` is sent as written, and the level names a model accepts differ from model to model: the comment above the `soclaas_*` entries in `configs/default.yaml` lists them. A level the model rejects stops the run with the provider's message. There is one entry per model and level worth comparing, such as `soclaas_qwen38` (thinking off), `soclaas_qwen38_low` and `soclaas_qwen38_medium`.

For an `anthropic` entry, `reasoning_effort` is the model's effort level: `low`, `medium`, `high`, `xhigh` or `max`. The model thinks by itself, as much as the level allows, and its thinking is billed as output. `none` is not one of those levels: it stands for thinking switched off. By Anthropic's documentation Claude Haiku 5.5 takes that, and Claude Opus 5.5 and Claude Sonnet 5.5 refuse it, which stops the run with the provider's message. Such an entry may set neither `temperature` nor `image_detail`: the models refuse a sampling parameter and the back end has nowhere to send either, so an entry that sets one is refused with an `error:` line that names it. An answer may be as long as `max_output_tokens`, thinking included, and 16000 tokens where the entry gives no number. The shape of an answer is asked for with the stage's schema, sent without the bounds of a point's coordinates because the API takes no bound on a number; the answer is checked against the bounds all the same. A request the model declines is recorded as refused, with the category the API gives, and is never handed to another model: a run measures the model its entry names.

What has been measured with real models so far is in [`docs/results/README.md`](docs/results/README.md).

### Strategies and prompt folders

A prompting strategy and the prompt texts are chosen per run, like a model.

```bash
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --model soclaas_qwen38 --strategy goal=few_shot_cot --name qwen38_few_shot_cot
cp -r prompts my_prompts      # then edit the four files in it
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --model soclaas_qwen38 --prompts my_prompts --name qwen38_my_prompts
```

`--strategy STAGE=NAME` gives one stage its strategy: `zero_shot` or `cot` for `plan`, `locate`, `context`, `people` and `constraints`, and for `goal` also `few_shot` and `few_shot_cot`. It may be given several times and is applied after `--model` and `--stage-model`. A value that is not `STAGE=NAME`, an unknown stage, an unknown strategy and a strategy the stage does not accept each stop the command with one `error:` line and exit code 2, before anything is written or any model is asked. In a config file the same setting is `strategy` for each stage under `stages:`.

`--prompts DIR` reads the stage prompts from `DIR` in place of the repository's `prompts/`: `plan.md`, `locate.md`, `context.md` and `goal.md`, and under `goal: constraints` also `constraints.md` and, with `people: vlm`, `people.md`. A relative `DIR` is a folder of the working directory. A prompt file is a series of blocks, each a line `## name` with its text under it, and each file has to keep every block that the repository's file of the same name has, each block with the `{placeholders}` of the repository's block and no other. A folder without one of the files the run reads, a file without one of those blocks, or a block with a placeholder too many or too few stops the command in the same way as a bad strategy. A file the run does not read is not looked at, so a folder with the four baseline files stays valid for a baseline run. A `{lower_case_word}` in braces is read as a placeholder in every block, the system block included, so a folder whose text needs literal braces round a word is refused. The keys of an answer that a prompt names in double quotes are not checked: a folder copied before 2026-10-09 still says `"people_matter"` in `context.md`, and `"reasoning"` and `"people"` in the `reasoning` block of `goal.md`, where the answers now have `"stop_elsewhere"`, `"analysis"` and `"arrangement"`, and has to be changed by hand. Nor is the wording checked: such a folder also still tells the context stage that the standard goal is a stop short of the target on the side the robot is coming from, where it is now a stop in front of the side of the target that the robot sees. In a config file the setting is `prompts_dir` under `run:`, relative to the repository root unless it is absolute.

`sgi.smoke` takes both options as well; it shows no examples, whatever the strategy. A run's `config.yaml` records the strategy of every stage and `prompts_dir`, and the run folder keeps a copy of the files the run reads in `prompts/`, so a number can be traced to the text that produced it. The cache key covers the text that is sent: a changed prompt is asked afresh, and an unchanged one is replayed.

### Pacing, prices and cost

An entry may give `max_requests_per_minute`. All the clients of one provider and address in a command then share one pacer, which lets at most that many requests start in any 60 seconds: a request beyond that waits for its turn, the first attempt and every retry alike, and an answer from the cache never waits. The wait counts in a call's `latency_total_s`, and so in the latency columns of `summary.csv`, but not in the call's `latency_s`, which stays the provider's own time. Where entries of one address name different limits, the lowest holds. The SoCLaaS entries are set to 80 a minute, because the service allows 90 a minute per key. Everything that uses the key shares those 90, and each command paces only itself: run commands one after another, not side by side, or lower the number.

An entry may also give `price_input_per_mtok` and `price_output_per_mtok`, in US dollars per million tokens: both or neither, and an entry with one of them alone is refused with an `error:` line that names it. The SoCLaaS entries carry the prices of the service's model list and the Claude entries those of Anthropic's; the Gemini and OpenAI entries have none. A call costs its input tokens at the first price and its output and thinking tokens at the second. `summary.csv` ends with three columns: `calls`, the model calls of the row's records; `cached_calls`, those answered from the cache; and `cost_usd`, what the calls cost, which is empty when one of them used an entry without prices. A call answered from the cache counts at what it cost when it was made, so `cost_usd` is the cost of a run's answers, not what the run added to the bill. The last line a run prints carries the same numbers, as in `(28 records, 0 ended in vlm_error; 73 calls, 17 from the cache, about $0.04)`. The cost is an estimate from the token counts the provider reports for the request that gave a call its answer. A call that was asked a second time because its first answer was not valid counts that second request only, so the figure is a little low where that happened. Neither the limit nor a price is part of the cache key, so changing one keeps every cached answer.

## Reading a run folder

`runs/<timestamp>_<name>/` holds five outputs.

- `config.yaml` is the exact configuration used, under `config`: the model entries, the strategy of each stage, the thresholds and the `run` settings, among them `examples_dir`, `prompts_dir`, `trials`, `leave_out_situation` and `final_test`. Beside it are the run's `name`, its `scenes_dir`, `example_ids` (the in-context examples every episode was shown), `examples_missing` (the situations that needed such an example and had none; see [Run modes](#run-modes)), `skipped_held_out`, `vlm_errors` and `incomplete`. A run that the provider's allowance stopped also has `stopped_by`.
- `prompts/` is a copy of the stage prompts the run read: four files, and under `goal: constraints` also `constraints.md` and, with `people: vlm`, `people.md`.
- `results.jsonl` has one record per episode and trial. `plan` is the plan as parsed. `result` is the goal result: its `status`, `pose_robot`, the flags, the stage outputs and `calls`, which keeps every model call with its raw response text. Of the flags, `standard_goal_frontal` is set when a standard goal was taken in front of the surface the target point lies on. A standard goal without it stands on the robot's line of sight to the target, because depth showed no flat upright surface round the target point, or no place in front of it that is clear and on floor the image shows, as wide as the robot is. Floor just under the picture counts as shown where the picture ends on floor above it. The six `surface_*` thresholds in `configs/default.yaml` say what counts as such a surface. `score` has `success` and its four components (`target_correct`, `reachable`, `in_region`, `heading_ok`) with the supplementary measures. `measures` has the context-check and decomposition measures, `labels` the episode's request type, situation and factors, and `example_ids` the examples shown. A record of a tour-mode run also has `tour`.
- A result of the constraint method has `source: constraints` and four more fields, each null in any other result. `people` lists the people found, each with its `number`, its floor position `xy` and heading `yaw` in the world frame, the body point `pixel` and, from a model, a `label`. `constraints` has one entry per constraint of the answer, in its order: the `constraint` as it was written, a `status` (`used`, `unusable`, `superseded` or `cut`), the `reason` when it is not used, and `broken_m`, by how much the place breaks it. `candidates` are the runners-up, each a `pose_world` with its `keys`. `solver` is the solver's note on the place: the `keys` it was chosen by, `own_space_broken_m` per person, `joined_waiting` (people taken as waiting whom the answer did not list), `not_ahead_of` (people who watch the target, whom the place is kept level with or behind), `not_ahead_reason` (why nobody is in that list, where the answer stated a view of the target) and `target_distance_m`. With these and the scene, a record can be solved again without a model call. Six flags belong to the method: `no_usable_constraint` (the solver ran alone, which is no failure), `constraint_unusable`, `constraints_cut`, `exclusion_broken` (every place broke an exclusion, and the one taken breaks the least), `people_cut` and `far_from_target`. A people call or a constraints call that brought no valid answer ends the result with that call's status and no pose. `goal_mark` is null, and `goal_pixel` is the place as the image shows it, null when it lies outside the image.
- Six measures hold the constraint method against ground truth. They are the last six entries of a record's `measures`, each null where the record does not measure it, and every one of them in a record of another method. `people_found_share` is the share of the people of ground truth in view who have a person found within twice `person_radius_m` of them, 0.5 m by default: the pairs are taken nearest first, and a person found counts for one person only. `people_unmatched` is the number of people found that are left over. Both are measured under `people: vlm` alone, and there is no share where nobody is in view. `members_precision` and `members_recall` are measured where the reference of the episode has people waiting their turn at the target and a model stated the constraints: of the numbers the answer lists under `come_after`, the share that are members of that group in view, and of those members, the share the answer lists. They are read from the answer as it was written, before the solver takes anyone else as waiting, and a number that nobody has is listed and wrong. `constraints_unusable` is the number of constraints the solver could not use, `unusable` or `superseded`, and `exclusions_broken` the number of exclusions the place breaks: the own space of each person it is inside, each `keep_out_of_shared_space` and `keep_out_of_view` it breaks, and `come_after` once. The last two are measured for every result the solver made, the solver alone included.
- `summary.csv` starts with the `overall` row, followed by one row per value of `request_type`, of `situation` and of each factor: `density`, `group_size` (binned as 0, 1-2, 3-5 and 6-10), `bystanders`, `other_groups` and `view_angle_deg` (up to 30 degrees, and above). An episode without a factor is counted under `unlabelled`. Each row gives `episodes`, `records`, `vlm_errors`, then `success_rate`, the rate of each success component and the supplementary measures. An empty cell means the row did not measure that column. With more than one trial a measure is the mean over the trials, followed by its minimum and maximum across them in the columns ending in `_min` and `_max`. The six measures of the constraint method are the last measure columns, under the names they have in a record: each is the mean over the row's records that measured it, and they are empty in every row of a baseline run. The six flags of the method have their rates among the other `flag_*_rate` columns. The last three columns, `calls`, `cached_calls` and `cost_usd`, are totals over the row's records and trials; [Pacing, prices and cost](#pacing-prices-and-cost) says how to read them.
- `viz/<scene>_<episode>.png` is one picture per episode: the image with the target and the goal marked, beside a top-down plot of the people with their headings, the target, the acceptable and forbidden regions, the robot and the goal pose. For a result of the constraint method both show the number of each person found and the runners-up as well, and the plot shades the floor the robot may stand on by the solver's order: the darker a cell, the better the solver ranks it. That floor is solved again from the record, with the default thresholds.

A record whose status is `vlm_error` got no model answer at all: the provider was unreachable or kept failing through three attempts. Such episodes stay out of the success rate and of every other rate, and are counted in `vlm_errors`. A run with any of them is incomplete: `config.yaml` says `incomplete: true` and the command exits with code 1. Run the same command again; the cache replays every call that was answered and only the rest are asked.

A run also stops when the provider's allowance is used up. With an `openai` or an `openai_compatible` entry, a 429 whose message speaks of a quota or a budget is not a rate limit: no wait brings an answer, and every later call would fail the same way. The run stops at that call and makes no other. `results.jsonl` keeps the records finished before it, `summary.csv` is written over those records, and `config.yaml` says `incomplete: true` and gives the provider's message as `stopped_by`. The command prints one `error:` line with that message and exits with code 2. When there is allowance again, run the same command again: the cache replays every call that was answered, so the run goes on from where it stopped. A run made with `--no-cache` stored nothing and starts over. Any other 429 is taken for a rate limit and asked again. A `gemini` entry has no such stop: each of its 429s is asked again.

A plan answer that could not be parsed is the model's own answer, so it counts as a wrong plan in the decomposition columns (`plan_exact_rate`, `navigate_recall` and `later_subtask_hit_rate`), not as an unmeasured one.

A folder `rescored/` in a run folder is not written by the run: it holds the run's records scored again by `sgi.rescore`, as [Score a finished run again](#score-a-finished-run-again) says.

Latency is read as follows. In `summary.csv` the columns `latency_s` (per episode) and `latency_<stage>_s` (per stage, for example `latency_plan_s`) include failed attempts and backoff sleeps, so read them next to the row's `transient_retry_share`: the share of the row's model calls that needed a transient retry. It counts the calls of `vlm_error` records too, is one value over all trials, and is empty when the row made no call. Quote a latency number together with the `model` and the `reasoning_effort` of the entry used, which `config.yaml` records under `config`, `models`. Per call, `results.jsonl` has `latency_total_s`, the per-query figure, with `transient_retries`, `model` and `reasoning_effort`. A call answered from the cache keeps the latencies of the request that was stored.

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

The first eight are settings: they go under `run:` in a copy of `configs/default.yaml`, and the copy is passed to `sgi.run` with `--config`. `--final-test` is a flag of the command.

`examples_dir` is set the same way, and `--scenes` does not set it. The worked examples of the goal stage come from `run.examples_dir`, which is `data/placeholder/examples` by default whatever folder `--scenes` names. To run delivered scenes, set `examples_dir` in the copy to the delivered `examples/` folder (an absolute path, or one relative to the repository root, not to the working directory) and pass the copy with `--config` beside `--scenes` and the delivered `eval/` folder. `example_ids` in the run's `config.yaml` shows which examples were used.

A situation needs a worked example when one of its evaluated episodes can reach the goal stage. Where the examples folder holds no eligible episode of such a situation, the run is not stopped: it logs one warning that names the situation, goes on with the examples there are, and lists the situation under `examples_missing` in `config.yaml`, which is an empty list otherwise. With no eligible example at all, the run goes on under the same strategy without examples. Under the default context check an episode with nobody near the target is eligible only where the standard goal fails in it. The placeholder examples have one such scene (`examples_06_no_one_person`), so `no_one` is not listed there; on an examples folder without such a scene it is the situation most likely to be listed. Only `k`, when it is set too small to reach a situation that needs an example and has one, still stops the run with an `error:` line. One of the school's models, `qwen3.8:27b`, accepts at most four images in a prompt, which is three worked examples and the camera view: for it set `leave_out_situation` or `k: 3` under `run:`.

The oracle run is `grounding: oracle`, `context_check: always_yes` and `goal: oracle`. It makes no model call and has to score 100% on the placeholder scenes. That shows the generator, the geometry and the scorer agree with one another. It says nothing about an exported scene; the scene check covers that.

`context_check: always_no` is the lower bound: a robot that ignores the people and takes the standard goal everywhere. A method is worth its model calls only by what it gains over that.

Use one trial for a plumbing run and three or more for any number that is compared or reported.
