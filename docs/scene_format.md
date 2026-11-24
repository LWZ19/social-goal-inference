# Scene format v0.1

A scene folder is what the social goal inference pipeline reads, and any simulator can export one. This page gives the files of a folder, the meaning of every field of `scene.json`, and how to check a folder before using it. The scene check (section 9) tells you whether an export is right.

A scene is static. It holds one RGB-D view of a place, the people and targets in it, and one or more requests a person might make there. The pipeline is given the view and a request and returns a goal pose for the robot. The harness scores that pose against the ground truth in the same file.

Every number on this page that is called a default is a threshold in `configs/default.yaml` and can be changed there.

## 1. What to deliver

**One folder per scene.**

```text
<scene_id>/
  scene.json
  rgb.png            H x W x 3, 8 bits per channel (an alpha channel is dropped on load)
  depth.npy          H x W, float32, metres
  occupancy.png      optional, see section 5
  tour/000.png ...   optional, see the tour frame table in section 4
```

`scene.json` names its image, depth, occupancy and tour files, relative to the folder, so other file names work too. Name the folder after its `"scene_id"`.

**Every scene has its own `scene_id`.** Results, pictures and example ids are keyed by it. The run harness refuses a folder of scenes in which two folders carry the same id. A scene folder copied to make a new scene, `docs/example_scene` included, needs a new `"scene_id"` in its `scene.json`.

**Two folders of scenes, `eval/` and `examples/`, with no scene in both.** `eval/` holds the scenes that are evaluated. `examples/` holds the scenes that the model's worked examples are taken from. It needs:

- at least one scene, and preferably two, for each development situation (`no_one`, `waiting_line`, `conversation_group`, `audience`),
- no scene from a held-out situation (`group_photo`, `lift_wait`, `machine_use`),
- ideally `rationale` filled in on every episode (section 4), so the examples can show their reasoning,
- for `no_one`, at least one scene in which the plain answer is wrong for a reason that has nothing to do with people. The plain answer stops about 1 m in front of the surface the target point lies on (section 4, `people_matter`), and an example is eligible only when that answer fails in it. For a place this is rare. The stop is in front of the place's face from any angle, as long as the floor before it is in view. Where there is no such surface, the stop falls on the robot's line of sight. That fails when the target has a front and is seen well off it (from about 45 degrees). The clear case is a person standing alone, seen in profile (about 90 degrees off their front). No plane fits the depth points of a body seen from the side, so the stop falls on the line of sight, beside the person and outside the region in front of them. The generated test set has that scene, `examples_06_no_one_person`. Warning 9 of the scene check is not raised on it, because the person counts as relevant.

The pipeline is pointed at the two folders separately. `--scenes` names the `eval/` folder, and the `run.examples_dir` setting names the `examples/` folder (see [Run modes](usage.md#run-modes)). Putting them side by side does not connect them.

**Inputs and ground truth are kept apart.** The inputs are `observation`, `tour`, and each episode's `request` and `context`. Everything under a `ground_truth` key is ground truth. The loader returns the two as separate objects, and the pipeline gets only the inputs. Nothing that identifies the answer may appear in an input. Write a `request` the way a person would say it, without ids or coordinates. A person's `role` is ground truth too, and it is never sent to a model.

**A key this page does not list is an error.** `scene.json` is read strictly, which is how a misspelt field name is caught. Extra files in the folder are ignored.

**Optional fields.** These may be left out: `source`, `tour`, `context`, `footprint`, `match_radius` (required for a `place` target without a `footprint`), `front_yaw`, `person_id` (required for a `person` target), `group`, `role`, `tour_frames`, `group_at_target`, `acceptable_regions`, `forbidden_regions`, `representative_point`, `target_id` (required on every region of an episode with several targets), `tolerance_deg`, `reference_plan`, `optional`, `factors`, `rationale`, `people_matter`, `version`. All of them except `tour`, `tolerance_deg` and `optional` may also be written as `null`. The lists `people` and `obstacles`, and the `target_ids` of a plan step, may be left out when they are empty. Of the two forms of free space, `occupancy` and `bounds` with `obstacles`, one is required (section 5). Everything else is required.

**Writing a scene.** Write the v0.1 folder directly. Fill in `intrinsics` and `T_world_camera` (for an Isaac Sim export, with the recipe of section 8), then run the scene check on the folder. An Isaac Sim export should keep the writer's `camera_params` file in the folder, so that the conversion can be redone.

## 2. Worked example

`docs/example_scene/` is a complete scene. It holds `scene.json`, `rgb.png`, `depth.npy` and eight tour images, `tour/000.png` to `tour/007.png`. It also holds `check.png`, which the scene check wrote and which is not part of the format. It is one of the generated test scenes. Two people wait in line at a help desk, and the robot looks at the line from the side. Open `check.png` next to this section.

Its `scene.json`, from top to bottom:

- `"format_version"` is `"0.1"`, and `"scene_id"` is `"eval_07_waiting_line_2"`. `"source"` records the generator, with `"simulator": "sgi-synth"` and no `"version"`.
- `"observation"` names `"rgb.png"` and `"depth.npy"`, both 640 wide and 480 high (`"width"`, `"height"`). `"intrinsics"` has `"fx"` and `"fy"` of 554.3 and `"cx"`, `"cy"` of 320 and 240, the image centre. `"T_world_camera"` puts the camera at (3.83, 0.81, 1.50), 1.5 m above the floor. Its third row, `[0, -0.985, -0.174, 1.5]`, says the camera's down axis points almost straight down in the world and its forward axis dips by 10 degrees. `"robot_pose_world"` is `"x"` 3.88, `"y"` 1.01, `"yaw"` -1.826. The robot's centre is 0.2 m behind the camera, facing the same way.
- `"tour"` has eight entries, each with its `"rgb"` file and its own `"T_world_camera"`. They were taken on a walk round the desk. Entry 3 is the observation itself.
- `"episodes"` has two entries for the same target. `e0` has the `"request"` "take me to the help desk", with `"request_type"` `"grounded"`, because it names the target. `e1` has the request "I need to report a lost item", with `"request_type"` `"high_level"`. It does not name the target, so the model has to work out that the help desk serves it. Neither has a `"context"`.
- Each episode's `"ground_truth"` has `"target_ids"` `["t0"]`, `"situation"` `"waiting_line"` and `"group_at_target"` `"g0"`, the group of the two people in line.
  - `"acceptable_regions"` holds one region, the half of the ring 0.5 to 1.2 m round the last person in line, `p1`, that is not ahead of them. It is one polygon of 182 vertices, from x 2.82 to 4.34 and y -4.54 to -2.18, which `check.png` draws as a half ring behind the line. Its `"heading"` is a fixed `"yaw"` of 2.87 rad, the way the line faces, with `"face_point"` null, and its `"tolerance_deg"` is 45. The robot may stand anywhere in the half ring and has to face along the line. Its `"representative_point"` is (3.95, -3.55), 0.83 m from `p1`, and its `"target_id"` is `"t0"`.
  - `"forbidden_regions"` holds two rectangles of `"kind"` `"ahead_in_line"`, each 1.2 m wide. One runs from the last person to the first, and one from the first person to the desk. A goal there is counted as cutting in.
  - `"reference_plan"` is the expected decomposition. For `e0` it is one step of `"type"` `"navigate"` to `t0`. For `e1` it is that step, an `"act"` step, and then a navigate step and an act step marked `"optional": true`, for going back to the requester. The navigate step back has empty `"target_ids"`, because the requester is not in the scene.
  - `"factors"` has only `"density"` entered, as `"low"`. `"group_size"`, `"bystanders"`, `"other_groups"` and `"view_angle_deg"` are null, so the harness derives them. The derived values are a group of 2, no bystander, no other group, and a view 77 degrees off the desk's front.
  - `"rationale"` gives the three texts `"people"`, `"expectations"` and `"placement"` that a worked example would show. The `"placement"` reads "Stand just behind the last person, at arm's length or a little more, facing the way the line faces." `"people_matter"` is null, so the reference value is derived.
- The scene's `"ground_truth"` comes last.
  - `"people"` has `p0` at (2.54, -3.17) and `p1` at (3.14, -3.34), each with a `"yaw"` of about 3.0 rad, facing the desk, and both in `"group"` `"g0"`. `"role"` is null for both, because they are waiting in line, not working at the desk.
  - `"targets"` has one target, with `"id"` `"t0"`, `"kind"` `"place"` and `"label"` "help desk". Its `"anchor"` is (1.80, -3.00, 1.0), the middle of the desk's front panel, 1 m up. Its `"footprint"` is the desk's ground rectangle and its `"front_yaw"`, -0.256 rad, points out of the front of the desk, along the line. `"match_radius"` and `"person_id"` are null. `"tour_frames"` lists all eight frames, because the desk can be seen in each.
  - `"bounds"` is the rectangle of the room's floor. `"obstacles"` has 18 polygons, the desk's footprint first and then the ground rectangle of every furnishing along the walls. `"occupancy"` is null, because this scene gives its free space as polygons.

The generator writes out every optional field, with `null` where it has no value. An exporter can leave those out. The reference rules of section 7 filled in the regions and representative points of this file, so an exporter does not have to write them either.

## 3. Frames and units

All ground-truth coordinates are in the world frame, in metres and radians.

1. **World:** right-handed, Z up, metres, ground at z = 0. This is the Isaac Sim default.
2. **Robot:** x forward, y left, z up, origin on the ground under the robot. A yaw is the rotation about +Z from world +X to the robot's +X.
3. **Camera (optical frame):** +Z forward, +X right, +Y down. Isaac Sim calls this camera axis convention `ros`.
4. **`T_world_camera`:** a 4 x 4 nested list, row by row, for column vectors: `p_world = T_world_camera @ p_camera`.
5. **Pixels:** u to the right, v down. Pixel (i, j) has its centre at (u, v) = (i + 0.5, j + 0.5), which matches `cx` = W / 2. A continuous (u, v) reads the arrays at row floor(v), column floor(u).
6. **Depth:** the distance to the image plane along camera +Z, in metres. This is Isaac's `distance_to_image_plane`, not `distance_to_camera`. A value that is not finite or not positive means no depth there.
7. **Angles:** radians. The loader wraps a yaw into (-π, π]. A yaw above π + 0.001 in magnitude is rejected, because that is what degrees look like.

With these, the world point seen at pixel (u, v) with depth Z is

```text
p_camera = [(u - cx) * Z / fx, (v - cy) * Z / fy, Z]
p_world  = T_world_camera @ p_camera
```

Three more rules:

- `robot_pose_world` and `T_world_camera` are independent. The camera does not have to sit at robot height on the robot. If there is no robot body, write the camera's ground projection as `x`, `y` and the yaw of its optical axis as `yaw`.
- The camera is a plain pinhole with square pixels and no lens distortion.
- `rgb` and `depth` have exactly `height` rows and `width` columns.

## 4. Field reference

In the tables, "number" is a JSON number, and a point `[x, y]` or `[x, y, z]` is a list of numbers in world metres. A polygon is a list of `[x, y]` points in order round its boundary, without repeating the first.

**Top level**

| Field | Type | Required | Meaning |
|---|---|---|---|
| `format_version` | string | yes | Always `"0.1"`. |
| `scene_id` | string | yes | The scene's name, different in every scene delivered. |
| `source` | object | no | Where the scene comes from. |
| `observation` | object | yes | The one RGB-D view the pipeline is given. |
| `tour` | list of tour frames | no | Images from an earlier pass through the space. |
| `episodes` | list of episodes | yes | The requests made in this scene, each with its ground truth. |
| `ground_truth` | object | yes | The scene's people, targets and free space. |

**`source`**

| Field | Type | Required | Meaning |
|---|---|---|---|
| `simulator` | string | yes | For example `"isaac-sim"`. |
| `version` | string | no | The simulator's version, for example `"6.1.0"`. The API names of section 8 depend on it. |

**`observation`**

| Field | Type | Required | Meaning |
|---|---|---|---|
| `rgb` | string | yes | File name of the colour image, relative to the scene folder. |
| `depth` | string | yes | File name of the depth array: one `.npy` array, metres. |
| `width` | integer | yes | Image width W in pixels. |
| `height` | integer | yes | Image height H in pixels. |
| `intrinsics` | object | yes | The pinhole parameters, below. |
| `T_world_camera` | 4 x 4 list | yes | The camera's pose in the world, for the optical frame of section 3. |
| `robot_pose_world` | object | yes | The robot's pose on the ground, below. |

**`intrinsics`**

| Field | Type | Required | Meaning |
|---|---|---|---|
| `fx` | number | yes | Focal length along u, in pixels. |
| `fy` | number | yes | Focal length along v, in pixels. Equal to `fx`: the pixels are square. |
| `cx` | number | yes | The u of the principal point, in pixels. |
| `cy` | number | yes | The v of the principal point, in pixels. |

**`robot_pose_world`**

| Field | Type | Required | Meaning |
|---|---|---|---|
| `x` | number | yes | World x of the point on the ground under the robot, in metres. |
| `y` | number | yes | World y of that point. |
| `yaw` | number | yes | The robot's heading in radians. |

**Tour frame** (an entry of `tour`)

| Field | Type | Required | Meaning |
|---|---|---|---|
| `rgb` | string | yes | File name of the frame's image, relative to the scene folder. |
| `T_world_camera` | 4 x 4 list | yes | The camera's pose in the world when the frame was taken. |

A frame's index is its position in the list, counted from 0. A tour is needed only for runs that ground the plan on the tour in place of the observation. A tour image needs no depth, and its size may differ from the observation's. The model is shown at most 16 frames, spread evenly over the tour.

**Episode** (an entry of `episodes`)

| Field | Type | Required | Meaning |
|---|---|---|---|
| `episode_id` | string | yes | Different for every episode of the scene. |
| `request` | string | yes | What the person says, for example "get me a coffee". |
| `context` | string | no | Free text the person would add, for example how to recognise someone they name. |
| `ground_truth` | object | yes | The episode's ground truth, below. |

**Episode `ground_truth`**

| Field | Type | Required | Meaning |
|---|---|---|---|
| `request_type` | string | yes | `"grounded"` when the request names the target ("go to stall 4"), `"high_level"` when it only says what is wanted ("get me a coffee"). |
| `target_ids` | list of strings | yes | The `id` of every target that is an acceptable answer for the first place the robot has to go, and that is visible in the observation. At least one. |
| `situation` | string | yes | What is going on at the target. One of `no_one`, `waiting_line`, `conversation_group`, `audience`, `group_photo`, `lift_wait`, `machine_use`. Any other label is an error. |
| `group_at_target` | string, object or null | no | The `group` id of the people at the target. With several targets, an object that maps each target id to a group id or null. |
| `acceptable_regions` | list of regions | no | Where the robot may stand, and how it must face. Filled in from `situation` when left out (section 7). |
| `forbidden_regions` | list of forbidden regions | no | Where standing is a specific mistake. Built together with the acceptable regions when both are left out (section 7). |
| `reference_plan` | list of plan steps | no | The expected decomposition of the request. |
| `factors` | object | no | Labels the results are grouped by, below. |
| `rationale` | object | no | The reasoning a worked example shows, below. |
| `people_matter` | boolean | no | Your own judgement of whether the people change where the robot should stop. |

`situation` and `factors` apply to every target in `target_ids`. An episode whose acceptable targets are in different situations cannot be written in v0.1. Split it into separate scenes.

**Region** (an entry of `acceptable_regions`)

| Field | Type | Required | Meaning |
|---|---|---|---|
| `polygon` | polygon | yes | The floor positions of the robot's centre that are appropriate. |
| `heading` | object | yes | The direction the robot has to face there, below. |
| `tolerance_deg` | number | no | How far the robot's heading may be off, in degrees. 45 when left out. |
| `representative_point` | point `[x, y]` | no | One good standing point inside the region (section 7). Chosen for you when left out. |
| `target_id` | string | when the episode has several targets | The target this region belongs to. A region without one applies to every target. |

**`heading`** has exactly one of its two fields.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `face_point` | point `[x, y]` | this or `yaw` | A ground point the robot has to face from wherever it stands in the region. |
| `yaw` | number | this or `face_point` | A fixed world heading in radians. |

**Forbidden region** (an entry of `forbidden_regions`)

| Field | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | What the mistake is: `group_space`, `audience_view`, `ahead_in_line`, `photo_frame`, `over_shoulder`, `beside_user` or a label of your own. Used for reporting only. |
| `polygon` | polygon | yes | The floor positions concerned. |
| `target_id` | string | when the episode has several targets | As for a region. |

**Plan step** (an entry of `reference_plan`)

| Field | Type | Required | Meaning |
|---|---|---|---|
| `type` | string | yes | `"navigate"` (go to a target) or `"act"` (do something there). |
| `target_ids` | list of strings | no | For a navigate step, the targets that would each do. Empty for a step whose target is not in this scene, such as going back to the requester. |
| `optional` | boolean | no | True when a correct plan may leave the step out. False when left out. |

**`factors`**

| Field | Type | Required | Meaning |
|---|---|---|---|
| `density` | string | no | `"low"`, `"medium"` or `"high"`: how crowded you judge the scene to be. |
| `group_size` | integer | no | The number of people in the group at the target. |
| `bystanders` | integer | no | The number of people who belong to no group and have no `role`. |
| `other_groups` | integer | no | The number of groups other than the one at the target. |
| `view_angle_deg` | number | no | The angle between the target's front and the direction from the target to the robot. Always computed for a target that has a front, so there is no need to write it. |

**`rationale`**

| Field | Type | Required | Meaning |
|---|---|---|---|
| `people` | string | yes | The people relevant to where the robot may stop: how they are arranged and what each person or group is doing. |
| `expectations` | string | yes | What they would expect of someone arriving to do what the request asks. |
| `placement` | string | yes | Where to stand and face, and why. |

**Scene `ground_truth`**

| Field | Type | Required | Meaning |
|---|---|---|---|
| `people` | list of people | no | Everyone in the scene. Empty when left out. |
| `targets` | list of targets | yes | Every place, object or person an episode names. |
| `bounds` | object | one form of free space | An object with one `polygon`: the outline of the floor. |
| `obstacles` | list of objects | no | Each an object with one `polygon`: the ground outline of a static thing. |
| `occupancy` | object | one form of free space | An occupancy map (section 5). |

**Person** (an entry of `people`)

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | Different for every person of the scene. |
| `x` | number | yes | World x of the ground point under the centre of the person's body, standing or seated. |
| `y` | number | yes | World y of that point. |
| `yaw` | number | yes | The direction the person faces: atan2(f_y, f_x) of their horizontal facing direction f in the world frame, in radians. |
| `group` | string | no | An id shared by people who are together. Leave it out, or null, for someone alone. |
| `role` | string | no | What the person is there for, when it is not to wait or to talk: `"staff"` for someone who works at a counter or a stall. Leave it out, or null, for an ordinary person. A person with a role is not a bystander and is not relevant to an episode unless one of its targets is that person. Any non-empty string counts as a role, and the person must have no `group`: both together is an error. |

**Target** (an entry of `targets`)

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | Different for every target of the scene. |
| `kind` | string | yes | `"place"` (a location or fixture), `"object"` or `"person"`. |
| `label` | string | yes | A short neutral name a visitor would use, such as "coffee stall" or "woman in a red jacket". |
| `anchor` | point `[x, y, z]` | yes | The 3D point a person would point at to show the target. |
| `footprint` | polygon | a `place` needs this or `match_radius` | The target's ground outline, for example from its world bounding box. |
| `match_radius` | number | a `place` needs this or `footprint` | Without a footprint: the radius in metres, from the anchor's ground projection, that counts as the target. 0.75 by default for an `object` or a `person`. |
| `front_yaw` | number | no | The world heading that points out of the side people approach, in radians. |
| `person_id` | string | for a `person` | The `id` of this person in `people`. |
| `tour_frames` | list of integers | no | The indices of the tour frames in which the target is visible. |

**Rules for people**

- Read every pose back from the stage at the captured frame. Do not copy it from the script or the spawn settings that placed the character. The simulated pose is what the image shows.
- Facing matters. The reference rules place regions by where people face, and no automatic check tests it. Section 9 says how to check it by eye.

**Rules for targets**

- The `label` is sent to the model, under oracle grounding and by the calibration command, so keep it neutral. It must not mention the people around the target, the situation or any id.
- The `anchor` lies on the target's visible surface in the image, or at most 0.3 m behind it. For a person, use about the centre of the chest. For a place, use the middle of the side people approach (the counter front, a door, a sign). For a small object, use its centre, and for a deeper one, the middle of its visible face. An anchor at the centre of a counter's volume is hidden inside it and fails check 5.
- A point marked by the model counts as "on the target" when its ground projection is within 0.3 m of the `footprint`, or, for a target without one, within `match_radius` of the anchor's ground projection. For a `person` target that person must also be the nearest person to the point.
- Only the reference rules and `view_angle_deg` use `front_yaw`. For a `person` target it defaults to that person's `yaw`. Leave it out for something that can be approached from any side, such as an object on a free-standing table.
- A `tour_frames` index has to be an index of `tour`.

**Rules for episodes**

- Every id an episode uses has to exist. That covers the ids in `target_ids`, in a region's `target_id`, in `group_at_target` and in the plan steps.
- A region's `target_id` has to be one of the episode's own `target_ids`. An acceptable region that names another target of the scene would never be tested, so no goal could succeed in the episode. The loader rejects it.
- When `target_ids` has more than one entry, `group_at_target` is written as an object, for example `{"stall_1": "g0", "stall_2": null}`, and every region carries its `target_id`.
- In a `reference_plan`, the first navigate step that has targets must list the same ids as the episode's `target_ids`. That step is the one the pipeline's goal is scored for.
- `density` is always your own label. When you leave out `group_size`, `bystanders` and `other_groups`, they are counted from `people` and `group_at_target`. Leaving them out is safer. The scene check warns when a number you entered differs from the count.
- `people_matter` is optional, and the reference value is derived. People matter exactly when the plain answer fails the success test. The plain answer stops about 1 m in front of the surface the target point lies on, on the side the robot sees it from, facing the target. It does so when a plane fits the depth points round that point and, within half a metre of that stop, there is a place clear of everything standing whose floor under the whole robot is in view. Otherwise (no plane, nothing clear there, or floor out of view) it stops 1 m short of the target on the robot's line of sight. On either line it slides away from the target by up to 0.5 m past what is in the way. The scene check prints the derived value per episode and says so when your label disagrees.

## 5. Free space

To test that a goal is reachable, the harness needs to know where the robot can drive. Give static free space as `bounds` with `obstacles`, as `occupancy`, or as both. At least one form is required. With both, a spot is free only when both say so.

**Polygons.** `bounds` is the outline of the floor. Everything outside it is not free. Each entry of `obstacles` is the ground outline of one static thing, such as a counter, a table, a pillar or a plant. Include everything solid that the robot cannot drive through, including what is lower than the camera.

**Occupancy map.**

```json
"occupancy": {"image": "occupancy.png", "resolution": 0.05, "origin": [-4.0, 0.0]}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `image` | string | yes | File name of the map image, relative to the scene folder. |
| `resolution` | number | yes | Metres per pixel. |
| `origin` | point `[x, y]` | yes | The world position of the outer corner of the bottom-left pixel. |

- The image is 8-bit greyscale. A colour image is converted to grey on load. A cell is free only if its pixel is 250 or brighter. Anything darker is not free, and so is everything outside the image.
- Columns run along world +X and rows along world -Y, so the top row of the image has the largest y.
- There is no rotation. A third `origin` element is accepted only when it is 0.
- The map holds static geometry only. Generate it with the people and the robot removed, over the height range of the robot's body, so that tables and chairs count as occupied.

The harness adds people as discs of radius 0.25 m and grows everything by the robot's radius, 0.30 m, on a 0.05 m grid. A person with a `role` is a disc like anyone else.

**A target's `footprint` is not an obstacle by itself.** Only what you list as `obstacles` or draw in `occupancy` blocks the floor. Take a counter whose footprint is 1.0 m deep and whose body is 0.4 m deep. It leaves a strip of free floor behind it, and a person (a member of staff, say) may stand inside the footprint on a free cell. When the two differ, list the counter's own outline as the obstacle, not its footprint.

## 6. Visibility and camera placement

The pipeline sees one image, so the image has to show what the answer depends on.

**Who is relevant.** A person is relevant to an episode if they are within 2 m of a target of the episode or of any of its acceptable or forbidden regions, or if they share a `group` with such a person. Distance and group never make a person with a `role` (staff) relevant. Such a person is relevant only to an episode one of whose targets is that person (`person_id`).

**When a person counts as visible.** Nine sample points are taken on the person's vertical axis, at z = 0.9, 1.0, ... 1.7 m above their `x`, `y`. A sample is seen when it projects inside the image at a pixel whose depth is within 0.3 m of the sample's own camera-frame Z, so that nothing nearer hides it. The person is visible if at least 3 samples are seen. Every relevant person must be visible. The scene check warns when one is not (warning 8).

**Where to put the camera.**

- Mount it about 1.5 m high and pitch it about 10 degrees down. That keeps whole standing people in frame, from feet to head, at 2.5 to 7 m.
- Do not use a lower camera with a steeper pitch. It cuts off people's heads, and in a model run this made them hard to recognise as people.
- Stand off the axis of any line of people. A camera below head height that looks along a line hides everyone but the last person.
- Keep the target's `anchor` in the image and unhidden (check 5), and keep some bare floor in view between the robot and the target. The scene check needs at least 2,000 floor pixels (check 3), and part of every acceptable region has to be visible floor (section 7).

The generated test scenes use exactly this mount. The camera is 0.2 m ahead of the robot's centre, 1.5 m high and pitched 10 degrees down, with a 640 x 480 image and `fx` = `fy` = 554.3, which gives a horizontal field of view of 60 degrees.

## 7. Regions and the reference rules

**What a region is.** An acceptable region is the set of floor positions for the robot's centre that are appropriate for the situation at its target, with the heading the robot must have there. A goal succeeds when it is on the right target, reachable, inside an acceptable region of that target, and within the region's `tolerance_deg` of its `heading`.

- A `polygon` is a simple polygon. A region with a hole is written as one polygon whose vertex list runs round the outer boundary and then back round the inner one.
- A `heading` is either `{"face_point": [x, y]}`, to face a point from wherever the robot stands, or `{"yaw": radians}`, a fixed direction.
- A forbidden region marks a specific mistake by its `kind`: `group_space` (standing inside a group), `audience_view` (blocking what people are watching), `ahead_in_line` (cutting into a line, or stepping in before people who wait for a turn), `photo_frame` (standing in a picture that is being taken), `over_shoulder` and `beside_user` (crowding someone who uses a machine), or any other string. It does not change success. The share of goals that land in each kind is reported.

**What you may leave out.** `acceptable_regions`, `forbidden_regions` and `representative_point` are all optional. When an episode has no acceptable region, the harness builds both lists from the reference rules below. The rules use only the scene's facts, which are `people`, `targets`, `situation` and `group_at_target`. If you write `acceptable_regions` yourself, write the matching `forbidden_regions` too. Nothing is added to an episode that brings its own acceptable regions. A `forbidden_regions` list you write is always kept as it is, even an empty one.

**The reference rules.** They define where the robot may stand. Distances are in metres, measured on the ground from the anchor's ground projection unless stated otherwise.

- **`no_one`.** For a target with a `front_yaw`, the region is a rectangle in front of the target along that direction, from 0.6 to 1.6 m from the anchor and 1.2 m wide. The robot faces the anchor. For a `person` target the rectangle is as wide and runs from 0.5 to 1.2 m from the person, and the person's own `yaw` is the front. For a target without a front, the region is the full ring from 0.6 to 1.6 m round the anchor, facing the anchor. Nothing is forbidden. A person in no group at the target changes none of this, however near they stand. An example is a person at work behind the counter.
- **`waiting_line`.** The members of the group at the target are ordered by distance from the anchor. The line's direction is the unit vector from the second-to-last member to the last, or from the anchor to the member when there is only one. The region is the part of the ring 0.5 to 1.2 m round the last member that is not ahead of them. A point is ahead of the last member when it lies on the target's side of the line drawn through that member across the line's direction. Wherever in the region the robot stands, it faces the way the line faces, from the last member towards the one before (towards the anchor when there is only one). This heading is written as a `yaw`. The distances are those of "Where Should I Join? Robot Group Joining via Language-Guided Goal Prediction" (arXiv 2609.28467). They also hold in the other rules wherever a person is the one stood by. A goal has to be reachable as well as in a region. The robot's centre cannot come nearer than 0.55 m to a person's (a person's disc of 0.25 m grown by the robot's 0.30 m). So where a region is measured from a person, the part of it nearer to them than about 0.55 m is in the region, and no goal in it succeeds.
  - A corridor 1.2 m wide, centred on the path from the last person through the members in order to the anchor, is forbidden as `ahead_in_line`.
  - The rule assumes a line that does not bend sharply. For one that does, supply the polygons.
  - The rule still holds for a line that turns, as long as each person stands further from the anchor than the one before. The corridor then follows the turn, and the region lies behind the last member along the last leg. The bent lines of the generated test scenes, which turn by 90 degrees in two steps of 45, are such lines.
- **`conversation_group`**, with a member of the group as the target. The group's circle is centred on the centroid of the members' positions. Its radius is the largest distance of a member from that centroid. The region is every point outside that circle that is 0.5 to 1.2 m from the nearest member, on any side of the group. The robot faces the target person. The circle's interior is forbidden as `group_space`. The rule needs at least two members, one of them the target person. Only the members count, and a person of another group or of none shapes nothing.

- **`audience`**, people who stand together and watch the target, which is not one of them. The space between the people and what they watch is the convex hull of the members' positions and the target's `footprint`. For a target without a footprint, a disc of 0.5 m round the anchor stands in for it, so that a person or a small thing has room of its own. The region is every point that is 0.5 to 1.2 m from the nearest member, at least 0.3 m outside that space, and no nearer the target than the member nearest it. That clearance is half a body's width, so that no part of the robot's body stands between a member and what they watch. Nearer is measured from the anchor along the target's `front_yaw`. A point belongs when it lies at least as far out along that direction as the member who is least far out. For a target without a front, nearer is plain distance from the anchor. In words, the robot joins the people behind or beside them, at arm's length or a little more from the nearest of them. No part of its body comes between any of them and the thing, and it is no nearer the thing than the nearest of them. The robot faces the anchor. The space between is forbidden as `audience_view`. The rule needs at least one member. A `front_yaw` that is given must be a finite angle, and with one that is not, the episode is left without a region.
- **`lift_wait`**, people who wait at the target for a turn, not in single file. The rule is that of `audience`. Here the space between the people and the target is forbidden as `ahead_in_line`.
- **`group_photo`**, a picture being taken, with the one who takes it or one of those posing as the target. All of them are in the group at the target. The picture-taker is the one member who has every other member within 60 degrees of the way they face, and whom every other member has within 60 degrees of the way they face. Exactly one member must pass this test. Ahead of the picture-taker is the direction from them to the centroid of those posing. The picture-taker's own `yaw` is used for the test and for nothing else. For either kind of target the region is the half of the ring 0.5 to 1.2 m round the picture-taker that is not ahead of them. It ends on the line through them across that direction. In words, the robot waits beside or behind the person taking the picture, at arm's length or a little more from them. The robot faces the target person, who is the picture-taker or the one posing whom it came for. The strip from the camera through those posing and on behind them is forbidden as `photo_frame`. This strip is the convex hull of discs round the picture-taker, round each of those posing and round the point 1.2 m behind each as the picture-taker sees them, so that it reaches 0.5 m to each side of the outermost of them. Places beside the people posing are not counted, because from a normal lens they are in the picture, or they are too far from anyone.

These last three rules are this project's own first definitions. They take the distances of the paper named above, 0.5 to 1.2 m from the person stood by, and nothing else from it. A disc in them is a polygon of 48 vertices on its circle, the default of `sgi.polygons.disc`. A scene they do not suit should supply its polygons. Their limits are these:

- A place level with the foremost of the people is accepted by the `audience` and `lift_wait` rule. Beside the one nearest the target, the robot has passed everyone else.
- The whole `footprint` counts as the thing. For a lift that is the shaft and not the doors, so the space between is as wide as the shaft and "beside the people" begins further out.
- The rule suits a compact group. Between two members more than 2.4 m apart the middle is too far from both. The region behind a row comes apart before that, from about 2.3 m between neighbours. The clearance takes the first 0.3 m behind the row, and a part of the band less than 0.1 m deep is left out. For people spread wide, supply the polygons.
- A space between that has no inside forbids nothing. An example is one member and a thing whose footprint is a line, on one line with them.
- For a target without a `front_yaw`, where nearer is plain distance from the anchor, the rule assumes that the anchor's ground point lies in the space between or within 0.3 m of it. A target without a footprint has it there, and so has a footprint that holds its own anchor. For an anchor further outside, supply the polygons.
- With one of those posing as the target, `group_photo` still places the robot by the picture-taker, where it faces the target across the picture. It does not go up to them.
- A picture taken of one person alone has no rule, because two people who face each other could each be the one taking it.
- Near the 60 degree limit one of those posing can be the only member who passes the picture-taker's test, for example when the one at the camera has turned far from the row. The region is then round that posing member.
- For a wide row the half ring and the forbidden strip of the picture overlap in a sliver at each inner corner of the ring, beside the picture-taker. The sliver lies within 0.6 m of them for a row that reaches 30 degrees to each side as they see it. A goal there is a success and is counted in the strip as well.

- **`machine_use`**, one person using the target, a machine with a front such as a cash machine or a ticket machine, with nobody else waiting. The user is the one member of the group at the target. Ahead of the user is the way they face, their `yaw`, and straight behind is opposite it. The region is every point 1 to 2 m from the user that is not ahead of them and lies within 60 degrees of straight behind them, which is a third of the ring round them. In words, the robot waits a metre or more behind the person at the machine, and not level beside them. The robot faces the anchor. Within 1 m of the user is forbidden as `over_shoulder`. The strip before the machine beside the user, where a second person would stand at it with them, is forbidden as `beside_user`. It runs from the line through the anchor across the target's `front_yaw` out to 0.5 m past the user, and across the target's `footprint` and 0.5 m past each side of it. The rule needs exactly one member, who is not the target, and a `front_yaw` that is a finite angle. It is this project's own, set before any scene of it was made. The near limit follows banks, which commonly ask the next customer to wait about a metre back. The far limit of 2 m keeps the robot clearly waiting for this machine. The bound of 60 degrees lets a place slightly off straight behind count as polite. The same rule with its near limit and the disc over the shoulder at 0.8 m (`region_machine_use(..., near=0.8)`) is reported beside it, with no pass mark.

A rule never fills in a missing input. All but `no_one` need `group_at_target`. Without it the episode is left without a region, and the scene check reports that as an error, or as a warning in a held-out situation.

**The representative point.** Every acceptable region needs one point that a method which marks a point on the image can hit. The test follows the path such a mark takes. The point is projected into the image, rounded to the 0..1000 grid the model answers on, and lifted back with depth. What comes back must be on the floor (within 0.15 m of z = 0), inside the same region, and in a free cell the robot can reach.

- When you leave `representative_point` out, the harness picks it from the points of the region's 0.05 m grid that pass this test. It considers those whose distance from the region's boundary is within one grid step of the largest such distance. When the region's `heading` is a `face_point`, it keeps those within one grid step of the nearest to that point and takes the one of them nearest the robot. When the heading is a `yaw`, it takes the one least to the side of the line through the polygon's centroid along that direction. Of two as far to the side, it takes the one nearer the centroid.
- A point you supply that fails the test is an error. It is never replaced.
- A region in which no point passes is an error too, because a point-marking method could not succeed there. So part of every acceptable region has to be bare floor that the image shows and the robot can reach.

## 8. Exporting from Isaac Sim

The scenes in this repository come from its own generators. The mapping below follows NVIDIA's documentation and has not been tested on a real export. API names depend on the Isaac Sim version, so record yours in `source`, as `"version"`. Where the scene check disagrees with this section, follow the check.

**Camera, from the Replicator `BasicWriter` output** (`rgb`, `distance_to_image_plane`, `camera_params`). Converting the camera takes more than a transpose. `cameraViewTransform` is world-to-camera, in row-vector form, in USD camera axes (-Z forward, +Y up) and in stage units.

```text
M = reshape(cameraViewTransform, 4, 4)
T = inverse(transpose(M))
T[:3, 3] *= metersPerSceneUnit
T_world_camera = T @ diag(1, -1, -1, 1)
fx = fy = W * cameraFocalLength / cameraAperture[0];  cx = W / 2;  cy = H / 2
```

**Camera, from the `isaacsim.sensors.camera.Camera` class** (deprecated since 6.0, with no direct replacement for these helpers). `get_intrinsics_matrix()` gives the 3 x 3 matrix that holds `fx`, `fy`, `cx` and `cy`. `get_world_pose(camera_axes="ros")` gives a position and a w-x-y-z quaternion, from which you assemble the 4 x 4. The default axes, `world`, are wrong here.

**Depth.** Use the `distance_to_image_plane` annotator. It is already in metres whatever the stage units. Zero or infinity means no hit.

**Near clipping.** The default near plane can be 1 m in a metre stage, which removes the floor and the people close to the camera. Set it to a few centimetres.

**People.** Take the position and the facing direction at the captured frame from the people-simulation output. Do not take them from the spawn settings, whose orientations are relative to each asset's default facing. With the Replicator Agent extension, this output is reported to be `metro_agent_data` in `IRABasicWriter`'s `object_detection.json`. Its `world_position` gives `x`, `y`, and its `world_facing_direction` is the vector f that `yaw` is computed from.

**Free space.** From the Occupancy Map generator take the image, the cell size (`resolution`) and the world position of the image's bottom-left corner (`origin`). The generator's own Origin setting is the start point of its flood fill, not that corner. Free cells must be white.

The pipeline exchanges scene files only and never runs inside the simulator's process, so its Python version is independent of Isaac Sim's.

## 9. Checking a scene

The scene check needs only the repository installed (see [Setup](usage.md#setup)). It needs no simulator, model or API key. Run it on every scene folder before using it. Here it runs on the example:

```bash
.venv/bin/python -m sgi.check_scene docs/example_scene
```

It prints one verdict per check, then one line per episode with the derived `people_matter` and factors, and exits with code 1 when a check is an error:

```text
[ 1] OK      structure:
[ 2] OK      pose sanity: camera height 1.50 m
[ 3] OK      floor depth: median relative error 0.00% over 106987 floor pixels
[ 4] OK      landmarks: 3 matched, 0 occluded, 0 skipped
[ 5] OK      anchors:
[ 6] OK      regions:
[ 7] OK      robot pose:
[ 8] OK      visibility:
[ 9] OK      standard goal:
[10] OK      map alignment:
[11] OK      free-space agreement: floor on free cells: 97.2% of 133002 points, obstacles on cells that are not free: 100.0% of 57625 points
[12] OK      factors:
e0: people_matter=True group_size=2 bystanders=0 other_groups=0 view_angle_deg=77.2
e1: people_matter=True group_size=2 bystanders=0 other_groups=0 view_angle_deg=77.2
```

Checks 1 to 7 are errors, because a scene that fails one cannot be scored correctly. Checks 8 to 12 are warnings. When check 1 or check 2 fails, the later checks are skipped, because they would only add noise.

| # | Name | Level | What it tests | The export mistake it catches |
|---|---|---|---|---|
| 1 | structure | error | `scene.json` follows this page: known fields only, required ones present, array shapes equal to `width` and `height`, ids unique and resolvable, known `situation` labels, yaws within π, a `place` with a `footprint` or `match_radius`. | A misspelt or missing field, a yaw in degrees, an id that points nowhere, an image of another size. |
| 2 | pose sanity | error | `T_world_camera` is a rigid transform (orthonormal rotation, determinant +1, bottom row 0 0 0 1), camera +Y points down in the world, the optical axis is at least 30 degrees from vertical, and the camera is 0.2 to 3.0 m high. | A transposed or inverted matrix, a camera left in USD axes, a translation in stage units. |
| 3 | floor depth | error | On floor that nothing can hide, the depth image matches the depth of the plane z = 0: a median relative error of at most 2%, over at least 2,000 pixels. | Range depth in place of plane depth, wrong units, wrong tilt or height, a wrong axis convention. Too little visible floor is reported as that, not as a mismatch. |
| 4 | landmarks | error | Each person, at 1.0 m above their `x`, `y`, and each `anchor` is projected. Where one lands in the image, the depth there must match its distance within 0.3 m, or be nearer (hidden). At least one must match. | A wrong camera yaw or position, which the floor alone cannot show, or people or targets placed from the script and not from the captured frame. |
| 5 | anchors | error | The `anchor` of every target an episode names is in the image, on the surface the depth image shows, within 0.3 m. | A target out of view, or an anchor inside or behind the target or hidden by something else. |
| 6 | regions | error | After the rules of section 7, every episode has an acceptable region, and each has a representative point that passes its test. | A missing `group_at_target`, a situation without a rule and without polygons, a region out of view or out of reach, a supplied point that a person stands on. A missing region is only a warning in a held-out situation. |
| 7 | robot pose | error | The robot does not stand inside an obstacle or a person. | A wrong `robot_pose_world`. |
| 8 | visibility | warning | Every relevant person is visible (section 6). | A camera placed so that people hide one another. |
| 9 | standard goal | warning | In an episode with nobody relevant, the plain answer (section 4, `people_matter`) succeeds. | A target the plain answer cannot stand before, for a reason that has nothing to do with people: a target with a front, seen well off it, whose anchor lies on no flat surface or on a place seen from so near that the floor before it is not in view. |
| 10 | map alignment | warning | Nobody stands on a cell that is not free. | A shifted or flipped map, or a map generated with the people still in it. |
| 11 | free-space agreement | warning | Depth points on the floor fall on free cells, and depth points above the floor and below the camera's height, more than 0.4 m from every person, fall on cells that are not free. Both shares are printed, and below 90% is a warning. | A shifted, flipped or wrongly scaled map, or an obstacle that the map or the polygons leave out. |
| 12 | factors | warning | A `group_size`, `bystanders` or `other_groups` you entered equals the count. | A hand-entered number that no longer matches the scene. |

**`check.png`.** The check also writes `check.png` into the scene folder. Its left half is the image with every person, every anchor and every region outline projected onto it. Each person is a dot 1 m above their `x`, `y`, with a line 0.5 m long along their `yaw`. The right half is a top-down plot of the free space, the people with their heading arrows, the targets, the regions and the robot. If the dots sit on the people and the outlines lie flat on the floor, the camera is right.

**Facing is checked by eye.** No check can tell which way a character faces. Compare each person's heading line in `check.png` with the way the image shows them facing, at least once for every character asset you use. An asset whose default facing differs from the others shows up as a line pointing out of the person's side or back.

**Warning 9 is not expected on any scene of the generated test set.** The plain answer stops before the surface it sees. A place seen at an angle therefore keeps its stop wherever the floor before it is in view. The one `no_one` example in which the plain answer fails (section 1) has a person at the target, who counts as relevant. A scene of yours raises the warning only when nobody is relevant and the plain answer still fails. One example is a place seen at an angle from so near that the floor before it is not in view. Another is a target with a front, seen well off it, whose anchor is on a thin post or an edge. When the warning appears, look at the scene again. Move the camera back, or, if the view is as intended, keep it as a scene in which the plain answer fails for no social reason.

**The harness runs the same check.** `sgi.run` checks each scene once and refuses a scene with an error, unless it is run with `--skip-scene-check`. Warnings never stop a run. Each is logged with its scene and check when the run starts. The run adds one test of its own. With `--final-test`, a held-out episode that has no acceptable region is refused, because no goal could succeed in it. Give every `group_photo`, `lift_wait` and `machine_use` episode its `group_at_target`, or supply its polygons where the reference rule does not suit it.
