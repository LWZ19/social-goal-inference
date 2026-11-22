# The pipeline in Habitat, among moving people

This page is for anyone who wants to watch the pipeline drive a robot: Boston Dynamics' Spot in a Habitat 3.0 flat where five people walk about and queue at the kitchen counter. Type a request such as "get me a cup from the kitchen counter". The robot's current RGB-D view goes to the pipeline with it, the people keep moving while the models answer, and Spot then walks to the goal pose that comes back.

The demo on this page is not a measurement. It is for seeing the pipeline at work. In [the results](../results/README.md), the still-scene numbers come from `python -m sgi.run` on scene folders, and the moving-line numbers from `sgi_habitat.experiment`, launched by `results/moving-lines/design-and-scoring/run_moving.py`.

## Two environments

Habitat-Sim is built for Python 3.9 only, and `sgi` needs Python 3.12, so they run in two conda environments and talk over HTTP on the local machine:

| Environment | Python | Runs | Package |
|---|---|---|---|
| `sgi` | 3.12 | `python -m sgi.serve`: the pipeline as a local service | `src/sgi` |
| `sgi-habitat` | 3.9 | `python -m sgi_habitat.run`: Habitat-Sim 0.3.3, habitat-lab 0.3.3, the people, Spot, the window | `sim/sgi_habitat` |

The service is given what a robot has (the image, the depth, the camera and the robot's pose) and answers with the goal pose. It reads no ground truth and scores nothing.

## Setup

From the repository root, once:

```bash
conda env create -f envs/sgi.yml
conda env create -f envs/sgi-habitat.yml
conda run -n sgi-habitat python -m habitat_sim.utils.datasets_download \
    --uids habitat_humanoids hab_spot_arm replica_cad_dataset --data-path data/habitat --no-replace
```

The download is about 370 MB: twelve humanoids with their walking motion, Spot, and the ReplicaCAD flats. `data/` is git-ignored. A real model needs its key in `.env`, as the [README](../README.md#setup) says; the `mock` entry needs none.

Rendering needs an OpenGL-capable GPU. It was set up and checked on an RTX 5060 Laptop (8 GB), where the simulation runs in real time at 30 frames a second with three 640 × 480 cameras.

## Run

One command starts both sides and stops the service when the simulator closes:

```bash
sim/demo.sh                                   # the mock model: the whole path works, the goals mean nothing
sim/demo.sh --model soclaas_qwen38_low        # a real model, with the constraint method
sim/demo.sh --model gemini_fast --goal direct_marking -- --record runs/habitat/demo.mp4
```

`--model` and `--goal` go to `python -m sgi.serve`. Everything after `--` goes to the simulator. Started by hand, in two terminals, it is:

```bash
conda activate sgi && python -m sgi.serve --model soclaas_qwen38_low
conda activate sgi-habitat && python -m sgi_habitat.run
```

A window opens. On the left is a camera that follows Spot. In the middle are Spot's RGB and depth, which are what the pipeline is sent. On the right is the map: grey where the robot can walk, red dots for the people in the queue, green for the people walking about, the robot in blue, its path in yellow and its goal in red, each with an arrow for its heading. The bottom line shows the robot's state and the last answer.

Type a request in the terminal and press Enter. In the window, `q` quits, `r` sends the last request again, `c` cancels the robot's goal, and `p` pauses the people and the robot.

The row under the panels shows what the pipeline made of the last request; `i` hides or shows it. On the left is the image it was sent, with the target ringed and named, the people it found numbered as the model saw them (under the constraint method), the goal with its heading, and the solver's runners-up as small rings. On the right is each stage's output: the plan and its reasoning, the target's world point, the context check (whether the people matter, who, and why), the people found with their distance from the target and the way they face, each constraint with what the solver made of it, the goal pose, any flags, and each call's status and time. The map adds the target (a magenta cross), the people as the model placed them (numbered rings, beside the true people), and the runners-up. The same picture and text are saved as `inference.png` and `inference.txt` in the request's folder.

### A model key

The `mock` entry needs no key. It answers every question with the centre of the image, so what the row shows is the shape of an answer, not an inference. For a real model, put a key in `.env` at the repository root (`cp .env.example .env` makes the file) and name the model entry:

| Provider | Key in `.env` | How to get it | Entry to try |
|---|---|---|---|
| Google Gemini | `GEMINI_API_KEY` | aistudio.google.com: sign in, then "Get API key" (there is a free tier) | `gemini_fast` |
| SoCLaaS, the school's gateway | `SOCLAAS_API_KEY` | from the course or SoC; the measured results use it | `soclaas_qwen38_low` |
| OpenAI | `OPENAI_API_KEY` | platform.openai.com | `openai_fast` |
| Anthropic | `ANTHROPIC_API_KEY` | console.anthropic.com | `claude_haiku55`: the fastest, about 15 s a request; its points are calibrated (pixels) |

Check the key with one real request before starting the simulator. The smoke check needs the placeholder scenes (`python -m sgi.synth.generate --out data/placeholder`, about two minutes). Then start the simulator with the same entry:

```bash
conda run -n sgi python -m sgi.smoke --model gemini_fast
sim/demo.sh --model gemini_fast -- --scenario sim/scenarios/cafe_line_side.yaml
```

A request costs a few model calls: about four under the constraint method (plan, context, people, constraints), fewer when the people do not matter. The first answer can take 10 to 40 seconds; the people keep moving meanwhile.

You can also send Spot somewhere by clicking, with no model involved. It walks there along a shortest path on the navmesh, as it does to a goal from the pipeline, and stops for people in its way.

- **On the map**: Spot goes to the place clicked. Press and drag to choose the heading it ends in; a plain click leaves it facing the way it travelled.
- **On the robot's RGB image**: the depth at that pixel lifts it to a point in the world, and Spot goes there. A point on a person, a counter or a wall sends it to the nearest place it can stand, and the status line says how far off that is.

A click made while the models are still answering wins: the answer, when it comes, is printed but not followed.

The simulator's options:

| Option | What it does |
|---|---|
| `--scenario FILE` | the scene, the people, the queue and the robot's start (default `sim/scenarios/apt_0_kitchen.yaml`) |
| `--request TEXT`, `--request-at S` | send this request after S seconds of simulated time (default 2), without typing it |
| `--context TEXT` | context text sent with every request |
| `--goal X Y YAW` | drive Spot to this pose at the start, without the pipeline: a check of the motion alone |
| `--headless`, `--record FILE` | no window; write what the window would show to an `.mp4` |
| `--seconds S`, `--exit-on-arrival` | stop after S seconds, or once Spot reaches its goal |
| `--sgi-url URL` | where the service listens (default `http://127.0.0.1:8765`) |
| `--out DIR` | where requests are kept (default `runs/habitat/<time>`) |

For a run without a window:

```bash
sim/demo.sh -- --headless --request "Get me a cup from the kitchen counter" --seconds 60 --exit-on-arrival --record runs/habitat/cup.mp4
```

## A test set: nobody, two or six in line

Six scenarios cover two cafés, each with three lengths of line:

| Café | Nobody | Two | Six |
|---|---|---|---|
| the simple café | `cafe_line0.yaml` | `cafe_line2.yaml` | `cafe_line6.yaml` |
| Sweet Spot (with a barista) | `sweet_spot_line0.yaml` | `sweet_spot_line2.yaml` | `sweet_spot_line6.yaml` |

The three of a café differ only in how many people wait. Spot starts from the same place facing the same way, the line stands still (nobody is served), and two people walk about beyond the line, in a `wander_area` that never comes between Spot and the line. The starts were found by search: Spot was placed in turn on every point of a 0.5 m grid with the six-person line standing, and each place was kept or dropped by what its camera saw. A person counts as seen when the depth image shows them, not something in front of them, at two of three heights (hips, chest, head). In all six scenarios, every person in the line and the counter were seen at every check over 30 s with the walkers moving.

```bash
sim/demo.sh --model claude_haiku55 -- --scenario sim/scenarios/sweet_spot_line6.yaml
```

The barista of the Sweet Spot café stands at the edge of Spot's view from there, and can fall just outside it.

## A concert: crowded or sparse at the front

`concert_crowded.yaml` and `concert_sparse.yaml` put Spot on an open ground 30 × 32 m, facing a concert stage across the ground. Two 8 m barricades stand 1.4 m in front of the stage, three performers stand on its deck, 1.2 m up, and two people walk about at the back. The two scenes differ only in the audience behind the barricades. In the crowded one the whole front row is taken and a second row stands behind it (41 people). In the sparse one four people stand scattered along the barricades. Asked "Take me near the stage, I want to watch the concert", the right place is behind the crowd in the first scene and a free stretch of barricade in the second.

In one run of each with Claude Haiku, the model told the two apart. In the sparse scene it named "the stage front barrier railing", kept out of the view of the people watching, and Spot stopped at the barricade in a gap between two of them. In the crowded scene it found 31 people and stated that the robot comes after them and keeps out of the crowd, and Spot stayed where it started, about 6 m behind the back row, rather than closing up behind it.

The stage came as an FBX file with separate textures, which nothing in `sgi-habitat` can convert. It was converted with Assimp in a small environment of its own: `conda env create -f envs/sgi-tools.yml`, then `conda run -n sgi-tools python sim/fbx_to_glb.py IN.fbx OUT.glb`. Its textures were then put on the GLB as one glTF material by `python -m sgi_habitat.texture_model`: base colour, metallic-roughness (packed from the roughness and metallic maps), and normal, each at 2048 px. The metallic factor was lowered to 0.35, since without reflections a fully metal surface renders black. Its units are centimetres, hence `scale: 0.01`. The barricade came as an OBJ in metres and went in through `import_model`. Neither download came with a licence file. Their Sketchfab pages give both as CC BY 4.0: the stage is [Simple Concert Stage](https://sketchfab.com/3d-models/simple-concert-stage-d5c7733d06d24947bf60b3a0fe203f69) by gluttonia (its 6,458 faces match the download; later re-uploads under the same name have 6,452), and the barricade [barigate 8 met besi hitam](https://sketchfab.com/3d-models/barigate-8-met-besi-hitam-07ba7df39b3a4947bb5f0910b9fbc530) by eventdesignws. Neither is in this repository; `sim/setup.sh` builds both from the downloaded `.zip` files.

## Moving among moving people

Replanning happens at two levels, at two very different rates.

**The path, twice a second, without the pipeline.** Spot plans on a navmesh of its own, cut back 0.45 m from every wall and piece of furniture (`clearance_m` under `robot:` in a scenario changes it). The people walk on one cut back 0.3 m. While anyone is within 3 m of Spot or of its path, the rest of the way is planned again every 0.5 s with A* on a 0.1 m grid of that navmesh. Each person is a disc 0.8 m in radius on the grid (their own 0.25 m, the robot's clearance and a margin), and the path is then pulled straight where the grid allows. So Spot goes round a line and round people walking about instead of stopping for them. It still stops for anyone within 0.65 m ahead, and when no way round is open it waits for them to move. On the map, the yellow path shows each new plan. In one run across the busy café (`cafe_busy.yaml`), Spot went round the moving line and two walkers without stopping, never left its navmesh, and came no nearer than 0.65 m to a person's centre.

**The goal, every few seconds, through the pipeline.** A goal is inferred once, on the view at the moment of the request, and people move after it. With `--requery S`, the simulator sends the same request again on a fresh view every S seconds while Spot carries it out, and once more after it arrives (twice at most). An answer whose goal moved more than 0.3 m or turned more than 25° replaces the goal. Any other answer, including one with no goal, leaves it as it is. A click stops the asking. Each time costs one pipeline answer, about 4 to 15 s with Claude Haiku, so a few seconds apart is as often as it can usefully go:

```bash
sim/demo.sh --model claude_haiku55 -- --scenario sim/scenarios/cafe_busy.yaml --requery 6
```

## What a request leaves behind

Every request gets a folder `runs/habitat/<time>/NNN/` holding:

- `rgb.png` and `depth.npy`: the view that was sent (depth is camera Z in metres, 0 where nothing was hit);
- `request.json`: the request, the context, the camera (intrinsics and `T_world_camera`) and the robot's pose;
- `answer.json`: the service's answer;
- `view.png`: the follow camera when the answer came.

These are live observations of moving people. With ground truth written for them they could become scene folders for `python -m sgi.run`, but no tool here writes that ground truth.

## The service

`python -m sgi.serve` takes `--model`, `--stage-model`, `--strategy` and `--goal` as the other commands do, and `--host` and `--port`. The default goal method is `constraints`. Under `direct_marking`, the goal stage is put on `cot` unless `--strategy` names its strategy, because the service has no example scenes to show. It answers:

- `GET /health`: the goal method and the model entry of each stage;
- `POST /goal`: a JSON body with `request`, `context`, `rgb_png` and `depth_npy` (base64 PNG and `.npy`), `camera` (fx, fy, cx, cy, width, height, `T_world_camera`) and `robot` (x, y, yaw). The answer holds `status`, the plan's `subtasks`, the `source` of the goal, `pose_world` and `pose_robot` as [x, y, θ], `target_world`, the `calls` made and `timings_s`. Its docstring, in [`src/sgi/serve.py`](../src/sgi/serve.py), describes each field.
- `POST /plan`: the plan alone, so a robot can start towards the target while the rest of the answer is made;
- `POST /resolve`: solves a stored `/goal` answer again on tracked people and a new observation, with no model call. The moving-line experiments use it.

Any simulator or robot can use it, not only this one.

## Frames

The pipeline's world is z-up with the floor at z = 0, and its cameras are optical frames: x right, y down, z forward ([`src/sgi/geometry.py`](../src/sgi/geometry.py)). Habitat's world is y-up, and its cameras look along −z. The two worlds share x. The pipeline's y is Habitat's −z, and its z is Habitat's y above the floor. A yaw is the same angle in both. [`sim/sgi_habitat/frames.py`](../sim/sgi_habitat/frames.py) does the conversion. A check run while this was built projected each person in the queue to a pixel whose depth was the front of their body (about 0.13 m before its centre), and lifted a floor pixel to z = 0.000.

Positions in a scenario file are in the pipeline's world, so a goal pose from the service, the map and the scenario all use the same numbers.

## The scenario file

[`sim/scenarios/apt_0_kitchen.yaml`](../sim/scenarios/apt_0_kitchen.yaml) is commented. It names the scene (a scene of a dataset, or a `.glb` under `stage:`, as in the next section), the robot's start, the robot's camera, the queues and the people. `humanoids` and `spot_urdf` default to where the download above puts them. An `avatar` is one of the twelve folders there: `female_0` to `female_3`, `male_0` to `male_3`, and `neutral_0` to `neutral_3`, which are untextured. A queue is a list of slots, the first at the counter. Its members stand in the slots in order: the first faces the counter and each other one faces the person ahead. Every `service_s` seconds the person at the front is served, leaves, walks about for 8 to 15 s, and joins the end of the line again. A person with no queue walks to a random place 1.5 to 8 m away, waits 1 to 4 s, and goes on. The robot's camera defaults to 1.2 m up, 0.35 m ahead of Spot's body, pitched 10° down, with a 90° field of view, 640 × 480.

## Your own scene, from Blender or elsewhere

A scenario can name a `.glb` file under `stage:` in place of `dataset:` and `scene:`. Habitat loads it, the navmesh is computed from its geometry at start-up, and the people and Spot work on it as they do in the flat. A café built this way comes with the repository:

```bash
conda run -n sgi-habitat python -m sgi_habitat.make_cafe          # writes data/habitat/stages/cafe.glb
sim/demo.sh -- --scenario sim/scenarios/cafe.yaml
```

It is a 10 × 8 m room with a coffee counter, a machine and a menu board along the north wall, four tables with chairs, and three people queueing at the counter while two walk about. Request "Get me a coffee".

To make a scene in Blender:

1. **Model in metres, with the floor's top at z = 0.** Leave the ceiling out. The follow camera looks down from up to 4 m above Spot, and with a ceiling it has to come in close.
2. **Leave room to walk.** The navmesh is cut for a body 0.3 m in radius, so a gap narrower than about 0.6 m between furniture, or a doorway that narrow, is closed to the robot and the people.
3. **Export with File → Export → glTF 2.0**: format *glTF Binary (.glb)*, *+Y Up* ticked (the default), *Apply Modifiers* ticked, materials exported. Put the file under `data/habitat/stages/`.
4. **Write a scenario** like `sim/scenarios/cafe.yaml`, with `stage:` naming the file. With *+Y Up*, Blender's x and y are the pipeline's x and y, so the robot's start, the queue slots and the start positions of the people can be read straight off Blender's 3D cursor.

The first time a `.glb` is loaded, a `<name>.stage_config.json` is written beside it. Without it, Habitat 0.3.3 takes a bare `.glb` to be z-up and lays the scene on its side. The file is not overwritten, so it can be edited. For a heavy scene, for instance, `collision_asset` can name a simpler mesh of the same room, since the navmesh and the clicks' ray casts use that mesh.

### A scene from separate models

A scene can also be put together from single models, with no modelling at all: an empty room as the stage, and the models placed by the scenario.

```bash
conda activate sgi-habitat
python -m sgi_habitat.make_room                                        # data/habitat/stages/room.glb: 10 x 8 m, a doorway
python -m sgi_habitat.fetch_models --list                              # the models of the repository's asset manifest
python -m sgi_habitat.fetch_models coffee_brewer plant_tall cooler_water
sim/demo.sh --model claude_haiku55 -- --scenario sim/scenarios/room_coffee_cart.yaml
```

`fetch_models` downloads models listed in `assets/realistic/manifest/models.json` into `data/habitat/objects/<id>/`, checking each file's SHA-256, and prints the file to name.

A model downloaded from anywhere else (Sketchfab, Poly Haven, CGTrader, a Blender export) goes in through `import_model`:

```bash
python -m sgi_habitat.import_model ~/Downloads/coffee-machine.zip --name coffee_machine
python -m sgi_habitat.import_model ~/Downloads/kiosk.obj --z-up
```

It takes a `.glb`, `.gltf`, `.obj`, `.stl`, `.ply`, `.off` or `.dae` file, or a `.zip` or folder holding one, as most sites give them. It writes one `.glb` with the textures inside to `data/habitat/objects/<name>/`, prints the model's size, and prints the line to paste under `objects:`. When the size looks like centimetres or millimetres, that line scales the model to metres with `height_m`. When the model is much flatter than it is wide, it says so: a model made z up, as many `.obj` and `.stl` files are, lies on its back until it is imported again with `--z-up`. A `.fbx` or `.blend` file has to be exported as glTF Binary (`.glb`) from Blender first. Each entry of `objects:` takes:

| Key | Meaning |
|---|---|
| `model` | the `.glb` or `.gltf` (y up, as glTF has it) |
| `x`, `y` | where the middle of the model goes, in the pipeline's world |
| `yaw_deg` | a turn about the vertical, counter-clockwise seen from above (default 0) |
| `scale` or `height_m` | a scale factor (default 1), or the height in metres to scale the model to. Use `height_m` for a model whose units are not metres: the water cooler's are centimetres. |
| `z` | metres above the floor, for something standing on something else (default 0) |

Each model is set on the stage's floor, its bounding box centred on (x, y), and is cut out of the navmesh, so people and Spot walk round it. As with a stage, an `<name>.object_config.json` is written beside a model the first time it is loaded.

In `sim/scenarios/room_coffee_cart.yaml`, Poly Haven's coffee cart (2.2 × 1.7 m) stands against the north wall with a line of three in front of it. Asked "Get me a coffee", Claude Haiku found the cart, stated that the robot comes after the people waiting, and Spot stopped 1.27 m behind the last of them, 0.30 m to the side of the line. That was one run, not a measurement.

### A downloaded scene without colours

Some downloads hold shape only. The Sweet Spot café from Sketchfab (CC BY 4.0) is one: its author uploaded an `.stl` file, so every format Sketchfab offers for it (glTF, GLB, OBJ, USDZ) renders white. `paint_stage` gives such a scene the two tones of Sketchfab's clay view. The floor and every top become a pale warm grey, and everything else terracotta:

```bash
python -m sgi_habitat.paint_stage data/habitat/downloads/sweet_spot_cafe/scene.gltf data/habitat/stages/sweet_spot_cafe.glb \
    --floor-y 0.15 --region 7.5 21.0 -0.3 9.3 --counter 11.0 16.7 4.9 8.85
sim/demo.sh --model claude_haiku55 -- --scenario sim/scenarios/sweet_spot_cafe.yaml
```

`--floor-y` is the floor's height in the file, `--region` the room in the pipeline's x and y, and `--counter` a counter's footprint. The ceiling is dropped, so the follow camera can look in from above. `--style colours` guesses a colour for each kind of surface instead (wood floor, stone counter tops, wall paint, and so on). Stages loaded from a file get soft light from above and from all four sides, and a light grey background where no ceiling is hit. With Habitat's default lighting, the faces turned away from its one light would be dark.

In the clay café, Claude Haiku names "the coffee counter at the back of the room" and finds it. In the two recorded runs it judged that the people queueing at the counter's other end did not matter, and stopped at the end of the counter nearest the robot, beside the line rather than behind it.

The realistic layouts in `src/sgi/synth/realistic/` are built in Blender 4.5 (`BLENDER_SERIES` in `blender.py`), so with Blender installed they could be exported in the same way. No exporter for them is written yet.

## Known limits

- **A fixed set of scenes.** The scenarios in `sim/scenarios/` use the environments listed in [`sim/README.md`](../sim/README.md). A ticket desk, a lift or a screen with an audience would each be a scene and a scenario of their own.
- **Spot's base glides.** The base moves kinematically along its path, as in Habitat 3.0's social navigation task, and the legs play a trot over it; the feet are not planted, so they can slide.
- **People only yield.** A walker waits up to 2 s for someone within 0.7 m ahead and then walks on, so two people can pass through each other, or walk into Spot's way. Spot plans round them (above), but people do not plan round Spot.
- **The local planner sees people as round.** It knows where each person is, not where they are heading, so someone walking towards Spot can come nearer than the 0.8 m it plans for before the next plan, half a second later.
- **The floor is taken to be flat.** The people's feet and Spot's base are held at the measured floor height, and the navmesh, which lies a few centimetres above it, gives only x and z.
- **The goal is snapped to the navmesh.** A goal pose closer to furniture or a wall than the navmesh allows (0.3 m) is moved to the nearest place the robot can stand. When that is more than 0.2 m away, the status line says how far.
- **One request at a time.** A request typed while the models are still answering is refused. The robot stands still while it waits, unless a click sends it somewhere.
- **The watching camera has no depth.** Clicks are taken on the map and on the robot's image only.
- **The mock's goals mean nothing.** It points at the centre of the image, so `sim/demo.sh` without `--model` only shows that the whole path works.
