# The realistic scene sets

The realistic scenes are Blender renders of indoor public spaces, with lit rooms, real furniture, varied people, and staff behind service points. They use the same scene format as every other set (see [scene_format.md](scene_format.md)), so `sgi.check_scene`, `sgi.run` and the scorer take them unchanged.

Every scene is built from a generated layout. The people and their places, the robot pose and camera, the targets, anchors and bounds, and so the acceptable and forbidden regions, all come from that layout.

## The three sets

| Set | Folder | Scene ids | Signage | Staff | Small things |
|---|---|---|---|---|---|
| `real` | `data/realistic` | `real_<twin id>` | natural | yes | yes |
| `plain` | `data/realistic_plain` | `plain_<twin id>` | large text labels | no | no |
| `staffed` | `data/realistic_staffed` | `staffed_<twin id>` | large text labels | yes | no |

In the scene ids, `<twin id>` is the scene id of the generated layout.

Each folder holds `eval/` and `examples/`, with 14 evaluation layouts and 7 example layouts (`eval_00` to `eval_13` and `examples_00` to `examples_06`). The generated test scenes after those have no realistic version. They are people watching a screen, a picture being taken, people waiting at a lift, and the lift and the screen with nobody at them.

A scene keeps its layout's episode ids and request texts. So rows of two sets pair up by the layout's scene id (the id without its `real_`, `plain_` or `staffed_` prefix) and the episode id.

Each comparison isolates one change:

| Compare | What it shows |
|---|---|
| `plain` against `staffed` | the staff, with one behind each staffed service point and a vendor at the second stall (seen only round the tour) |
| `staffed` against `real` | natural signage and small things |

**Why natural signage.** A big text sign over a desk makes the target obvious, and real places usually have none. In `real`, a place shows what a real one would, such as an invented logo, a pictogram, a small plate, a menu board or a screen. Nothing spells a target out in large letters, and no real brand is used. The two control sets use large text labels.

## Get a set

Rendered sets are not in git. Each one is a single archive attached to a release of this repository. A manifest in the repository lists the path, size and SHA-256 of every file. To get the `real` set:

```bash
gh release download scenes-realistic-v1 --pattern real.tar.gz
.venv/bin/python -m sgi.synth.realistic.package fetch --archive real.tar.gz --into data/realistic --variant real
.venv/bin/python -m sgi.synth.realistic.package verify --set data/realistic --variant real
```

The release `scenes-realistic-v1` of the repository holds the three final sets:

- `real.tar.gz` goes into `data/realistic`.
- `staffed.tar.gz` goes into `data/realistic_staffed` (`--variant staffed`).
- `plain.tar.gz` goes into `data/realistic_plain` (`--variant plain`).

Download an archive with the GitHub command line tool and hand it to `fetch` as a file.

`--variant` names the manifest, here `assets/realistic/sets/real.json`. `fetch` checks the archive against it, unpacks into a temporary folder beside `--into`, verifies every file there, and only then moves the set into place. It refuses an existing `--into` unless you add `--force`. An archive at a public `https` address can be given with `--url` instead of `--archive`.

`verify` exits with 1 when the set is not the one the manifest names, and lists each file that is missing, changed or unexpected.

These commands upload nothing.

## Make a set

Rendering needs Blender 4.5 LTS and the assets. The test suite does not, because the tests that use Blender are skipped when it is absent.

**1. Set up (once).**

```bash
.venv/bin/python -m sgi.synth.realistic.assets blender
.venv/bin/python -m sgi.synth.realistic.assets fetch
.venv/bin/python -m sgi.synth.realistic.assets prepare
.venv/bin/python -m sgi.synth.realistic.assets verify
```

- `blender` puts Blender into `.tools/`. To use another binary of the 4.5 series, set `SGI_BLENDER` to it.
- `fetch` downloads every asset into `assets/realistic/_cache/` and checks its size and SHA-256. That is about 3.3 GB, and none of it is committed.
- `prepare` has Blender normalise the models and bake each avatar in each pose into a still mesh, and writes a stamp. The generator runs only when the stamp covers the whole manifest.
- `verify` says what is missing or wrong.

`--only ID ...` limits `fetch`, `prepare` and `verify` to the named assets. `--force` does them again where they are already done.

**2. Generate.**

```bash
.venv/bin/python -m sgi.synth.realistic.generate
.venv/bin/python -m sgi.synth.realistic.generate --variant plain
.venv/bin/python -m sgi.synth.realistic.generate --variant staffed
```

| Flag | Meaning |
|---|---|
| `--variant` | `real` (default), `plain` or `staffed` |
| `--out` | the set's folder (default: the folder in the table above) |
| `--seed` | the seed of the generated layouts to replay (default 0) |
| `--config` | a configuration file, for the thresholds |
| `--renderer` | `blender` (default) or `stub`, which needs neither Blender nor assets and is used by the tests |
| `--only ID ...` | make only these scenes, named with or without the variant prefix |
| `--force` | draw a scene again though its stored job is unchanged |

The command replays the accepted layouts of the generated test scenes for the seed. It dresses each as a scene, renders it, runs the scene check, derives the regions with `sgi.gt` and builds the tour. It draws no layout of its own.

Progress goes to stderr, one line per scene. The one line on stdout reads `<out>: N scenes rendered, N skipped as unchanged, N dressed again`.

A scene that fails the scene check is dressed again, up to five times. The command then stops and names the reason. All three variants of a scene must pass for a dressing to be kept, so the three sets always show the same dressing of the same layout.

Budget about 16 s a scene on an idle GPU and 30 to 45 s on a shared one, which is some minutes a set, and about 4.5 MB a scene.

A scene folder holds `scene.json`, `rgb.png`, `depth.npy`, `tour/000.png` to `tour/007.png`, and `render_job.json`. The last records what was rendered, which is the variant, seed, kept attempt, renderer and its version, the scene description, the thresholds, and a digest of each variant's scene.

**When a scene is skipped.** A scene whose stored `render_job.json` equals the job about to be run is skipped. Pictures are not bit-reproducible on a GPU, and a model's cached answers are keyed on the pixels, so a scene is redrawn only for a reason. The reasons are a change of its description, camera, episodes, ground truth, any threshold, the renderer or its version. A log line says which entries differ.

A change the job does not record, such as code, prepared asset files or render settings, is not seen. Use `--force` or `--only` with `--force` for those. A run never deletes a folder.

**3. Pack (when a set is final).**

```bash
.venv/bin/python -m sgi.synth.realistic.package pack --set data/realistic --out real.tar.gz
```

This writes the archive and the manifest `assets/realistic/sets/<variant>.json`. The variant is read from the scenes. The same files give the same archive bytes. Delete a `check.png` that `sgi.check_scene` left in a scene folder first, or it is packed too.

## Run a model on a set

`--scenes` takes the set's `eval/` folder. The worked examples of the goal stage come from `run.examples_dir`, which `--scenes` does not set, so point a copy of `configs/default.yaml` at the set's `examples/` folder. [Run modes](usage.md#run-modes) says how.

```bash
.venv/bin/python -m sgi.check_scene data/realistic/eval/real_eval_02_no_one_place
.venv/bin/python -m sgi.run --scenes data/realistic/eval --config my_realistic.yaml --model mock --name realistic_plumbing
```

Here `my_realistic.yaml` is that copy, with `examples_dir: data/realistic/examples` under `run:`.

## How a layout is dressed

- **Staff.** In `real` and `staffed`, each staffed service point (a stall, desk or ticket counter) has one member of staff behind it, facing out of the front.
  - The second stall of a scene (fruit, flowers or sweets), which is never a target, has a vendor too. Every scene of seed 0 holds such a stall. A scene has two members of staff where its target is a staffed place, and one, the vendor, where it is not.
  - In seed 0 that stall stands outside the observation picture in all 21 scenes. The vendor shows only round the tour, in one or two of the eight frames in 12 of the 21 scenes. In the picture a run reads by default, the place asked for is therefore still the only one with someone behind it.
  - Staff are in the ground truth as people with `role: "staff"` and no group. They count in no line, group or bystander total, and a `density` label equals the generated layout's. The role is documented in [scene_format.md](scene_format.md) and is never sent to a model.
- **Obstacles under staffed places and the second stall.** To leave a strip of free floor for the staff member, the counter is built shallower than its footprint. For these places the counter's own outline (counter, back wall, chair, posts) is the scene's obstacle instead of the footprint. The target's `footprint`, anchor and front face are unchanged. This holds in all three variants, so the free space is the same in each.
- **One rationale text.** Where nobody waits at a target that a member of staff stands behind, the layout's sentence "Nobody is waiting at the target or standing close to it" would be false of the picture. In `real` and `staffed`, such an episode's `rationale.people` says instead that nobody is waiting on the visitors' side and that a member of staff stands behind it. This is the only field of an episode that may differ from the layout's. Ids, requests, contexts, request types, plans and the other rationale texts are the layout's.
- **Signage.** Described above. The sign rectangles are recorded in both modes, because the tour's rule for whether a frame shows a target reads lettering from them.
- **Small things** (`real` only). Cups, a monitor, books and the like stand on counters and tables. Things on walls are no deeper than 5 cm or sit above the camera. They are placed where they change no free space.
- **People.** The height comes from the layout. The avatar, clothes, build and pose are chosen when the layout is dressed. A target "person in red" is cast from an avatar in red, and nobody else in the scene wears red or pink.
- **Tour.** A scene has eight tour frames, the observation among them (`tour/003.png` is `rgb.png`). The `tour_frames` listed for a target may differ between the three variants of one scene.
- **Depth.** `depth.npy` is exact planar camera depth with 0 for no hit, and is the depth the scene was judged with. It agrees with `rgb.png` except for a few pixels on hair.

## Licences and attribution

Everything downloaded is listed in `assets/realistic/manifest/*.json` with its source, licence as the source states it, SHA-256 and size. [`assets/realistic/ATTRIBUTION.md`](../assets/realistic/ATTRIBUTION.md) is written from that list, so change the manifest and not that file. To write it again:

```bash
.venv/bin/python -m sgi.synth.realistic.assets attribution
```

The avatars and poses are Microsoft Rocketbox (MIT). Models and materials are mostly CC0 (Poly Haven) and CC BY 4.0 (Objaverse and Amazon Berkeley Objects). The fonts are under the SIL Open Font License. Pictures that use CC BY assets need the attribution to travel with them. The manifest and the attribution file are committed. The assets themselves are not.

## Known limits

- **Pictures are not bit-reproducible on a GPU.** Depth is. Two renders of one scene differ in a few percent of the pixels by one level of 255. Everyone who scores the same pixels therefore needs the same files, which is why the sets travel as archives.
- **A redraw invalidates a model's cache.** A scene is drawn again whenever its description, camera, episodes, ground truth or thresholds change, and new pixels make a model's cached answers for that scene useless.
- **People.** There are no children, no walking poses and no gaze.
- **The assets are fetched by script.** They take about 3.3 GB under `assets/realistic/_cache/`, none of it committed.
- **Blender 4.5 LTS is needed to render**, not to run the tests. The Blender tests are skipped without it.
- **14 evaluation and 7 example layouts.** There are no lines with uneven spacing, seated people, an audience or held-out situations, and no camera noise or lens blur.
