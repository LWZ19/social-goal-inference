# The constraint method

The pipeline has one step where people decide the outcome. That step chooses where the robot stops when the people at the target matter. The baseline has a model mark that place as a point on the image. The constraint method splits the step in two:

- A model says what the people require of the place, in a small fixed vocabulary.
- Geometry finds the place.

Choose the method with `goal: constraints`. It runs in the same episodes as the baseline, the ones in which the context check says that the people matter. Everything before that step is the same in both. The [usage page](usage.md) explains the commands.

Every figure for the solver on this page comes from a test in `tests/` or from recorded runs solved again without a model. "Known limits" and "What has been measured" say which.

## What it does, in four steps

1. **The people are found.** With `people: vlm`, the default, one call of the `people` stage takes the camera image as it is and lists the people in it. For each person it gives:

   - a few words of appearance,
   - a point on the body,
   - a point on what the person faces, when that is in view,
   - the way the person faces as seen in the image, one of eight directions or `unclear`.

   Depth lifts the body point to the floor. The centre of the body is placed one person's radius farther from the camera than that point. The heading is the direction to the faced point, or else the stated facing turned into the world frame. An entry whose point does not lift, or lifts onto the floor, is left out.

   The request holds nothing specific to an episode, so all episodes of a scene share one cached answer. With `people: oracle`, the people are the ground-truth people in view, with their true positions and headings, and no model is called.

2. **They are numbered and shown.** The people found are merged, dropping any one within a person's radius of another. The `max_people` nearest the target are kept and numbered from 1, starting at the left of the image. The model of the next step is shown:

   - the image, with the target ringed and each number in a yellow disc on its person,
   - the request and the subtask,
   - what the context check wrote about the people, when a model made that check,
   - a list with one line per person, computed from depth. Each line gives the person's distance from the robot and from the target, which third of the image they are in, and whom or what they face, with distances.

   A person who is the target is marked as such.

3. **A model states constraints.** One call of the `constraints` stage returns a list of constraints. Each constraint has a one-sentence reason, one of six kinds, the numbers of the people it concerns, and whether it concerns the target. The one kind that takes a range in metres also has that range. At most `max_constraints` are read. The prompt tells the model what the robot does without being asked, and asks it to state nothing for that. With `constraints: none` this call is not made.

4. **The solver finds the place.** It reads the floor from the depth image on a grid of `grid_resolution_m` cells in the robot's own frame. It marks where the image shows floor, where something nearer hides it, and where something stands. Each constraint becomes a field over that floor, in metres of violation. The cells the robot can stand on and reach are compared key by key, in this order:

   1. the largest violation of any one exclusion,
   2. the sum of the exclusions' violations,
   3. the distance to the place in turn, when someone is waited behind,
   4. how far behind the seen face of the target a cell lies (when the target has such a face, is not a person, nobody is waited behind, and the answer does not say the robot came for a person),
   5. the sum of the preferences' violations,
   6. the distance to the standard goal,
   7. the distance to the robot.

   The best cell is the place. The robot faces what a `face` constraint names. Without one, it faces the last person it waits behind, or else the target. Up to `candidate_count` runners-up are kept. The solver calls no model and reads no ground truth.

Without any constraint asking for it, the solver keeps the robot `standoff_m` from every person found. That distance is the person's own space. The solver also stops the robot about `standoff_m` from the target.

Where the depth image shows a flat upright surface round the target point, the solver takes it as the seen face. It keeps the robot on the side of that face that the robot saw. Among the cells that break the exclusions least, those in front of the face come first. So people standing in front of a thing do not send the robot round behind it.

The solver ignores that face when the answer says the robot came for a person. The answer says so when a used `face` constraint names somebody and a usable `visible_to` lists the same person. The place is then chosen by that person's front, and a face fitted from depth does not rank above it. A `visible_to` alone, or one about somebody the robot is not told to face, leaves the seen face in use.

Where the answer holds no usable constraint, these rules and the standard goal decide the place alone. The flag `no_usable_constraint` records this.

## The six kinds

| Kind | The place | In the solver |
|---|---|---|
| `keep_out_of_shared_space` | is outside the space between the listed people, who attend to one another. Two or more are listed. | an exclusion |
| `keep_out_of_view` | is outside what the listed people watch. That is towards the target with `target: true`, and straight ahead of each of them otherwise. With `target: true` it is also not ahead of those of them who stand on the side of the target the robot was sent to. It is level with the hindmost of them, or behind, where the robot can stand on some such place without breaking another exclusion more than it does ahead of them. | an exclusion |
| `come_after` | is one taken after all the listed people, who wait their turn at the target. | an exclusion, and the place in turn |
| `within_distance` | is between `at_least` and `at_most` metres from each listed person, or from the target when nobody is listed and `target` is true. | a preference |
| `visible_to` | is where each listed person can see it without turning round. | a preference |
| `face` | is not affected. The robot faces the listed person, the middle of several, or the target when nobody is listed and `target` is true. | the heading |

An exclusion is never traded for a preference. If some cell breaks no exclusion, the place breaks none.

Fields a kind does not use are ignored. A constraint the solver cannot use is recorded with the reason. Examples are a constraint that lists only numbers nobody has, a `within_distance` without a range, and a second `face`.

No kind names a situation, and the solver has no branch for one.

## Run it against the baseline

Every command runs from the repository root. The first three lines make the config files. Each is a copy of `configs/default.yaml` that differs only in the settings named.

```bash
cp configs/default.yaml constraints.yaml      # then set goal: constraints under run:
cp constraints.yaml solver_alone.yaml         # then set constraints: none under run: as well
cp constraints.yaml given_people.yaml         # then set people: oracle under run: as well
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --config constraints.yaml --model mock --name constraints_plumbing
.venv/bin/python -m sgi.smoke --model gemini_fast --goal constraints --scene data/placeholder/eval/eval_08_waiting_line_4
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --model gemini_fast --name baseline_gemini
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --config constraints.yaml --model gemini_fast --name constraints_gemini
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --config solver_alone.yaml --model gemini_fast --name solver_alone_gemini
.venv/bin/python -m sgi.run --scenes data/placeholder/eval --config given_people.yaml --model gemini_fast --name given_people_gemini
```

1. The run on `--model mock` needs no key. It checks that the path works.
2. The `sgi.smoke` line is a check by eye. It shows whether the model refers to people by the numbers drawn on them, and writes the image the model was shown to `runs/smoke_numbered.png`. Run it on a few scenes with people in view before relying on any number from a model under this method.
3. The four runs then give four numbers to read together.

| Run | Settings under `run:` | What it is |
|---|---|---|
| `baseline_gemini` | `goal: direct_marking` | The baseline. The model marks the place. |
| `constraints_gemini` | `goal: constraints` | The method. This is the run compared with the baseline. Its people are found by the model, so its success and its latency come from the same calls. |
| `solver_alone_gemini` | `goal: constraints`, `constraints: none` | The solver on its own rules. What the model's constraints add is the difference between this run and the one above. Every result of the method is reported beside it. |
| `given_people_gemini` | `goal: constraints`, `people: oracle` | The constraints step with the people known exactly. This is its upper bound. |

Keep the cache on and use one model entry for all four runs. The plan and context-check answers are then the same in every run, so all four place a goal in the same episodes. The solver-alone run makes no call of its own. It replays the plan, the context check and the people from the run with constraints.

Compare the `summary.csv` files row by row on `success_rate` and its components, in the `overall` row and in each `situation` row. Use three or more trials for a reported number, and leave `--final-test` out until the end.

`context_check: always_no` is the lower bound for all of them. It is a robot that takes the standard goal everywhere.

## What a record shows

A result of the method in `results.jsonl` has `source: constraints`. Besides the fields every result has, it holds:

- `people` lists the people found. Each has its `number`, its floor position `xy` and heading `yaw` in the world frame, the body point `pixel`, and the `label` a model gave.
- `constraints` has one entry per constraint, in the answer's order. Each holds the `constraint` as written, a `status` (`used`, `unusable`, `superseded` or `cut`), the `reason` when it was not used, and `broken_m`, how much the place breaks it.
- `candidates` lists the runners-up, each a `pose_world` with the `keys` it was ordered by.
- `solver` holds the following:
  - `keys`, the keys of the place. Among them, `face_m` is how far behind the seen face the place lies. It is 0.0 for a place in front of the face and positive for one that had to go behind it. It is null where the target has no such face or is a person, where a place in turn was used, or where the answer names a person to face and to be visible to.
  - `own_space_broken_m`, per person.
  - `joined_waiting`, the people the solver took as waiting though the answer did not list them.
  - `not_ahead_of`, the people watching the target whom the place is kept level with or behind. It is empty where that rule was not applied.
  - `not_ahead_reason`, which says why that list is empty when the answer has a usable `keep_out_of_view` towards the target. The reasons are:
    - a place in turn is used,
    - the target is a person,
    - none of the listed people stands on the side the robot was sent to,
    - that side, or the direction they watch in, cannot be told,
    - no place the robot can stand on is level with the hindmost of them or behind,
    - every such place breaks the other exclusions more than some place ahead of them.
  - `target_distance_m`.
- Six flags are set or clear: `no_usable_constraint`, `constraint_unusable`, `constraints_cut`, `exclusion_broken`, `people_cut` and `far_from_target`.

With these fields and the scene, a record can be solved again without a model call. `calls` keeps the two calls, with the answers as the model wrote them. If the people call or the constraints call brought no valid answer, the result ends with that call's status and no pose, and it is scored as a failure.

When a place looks wrong, read the record in this order:

1. `people` against the picture. Was everyone found, and where?
2. `constraints`. What was stated, and what was unusable?
3. `joined_waiting`, `not_ahead_of` and `not_ahead_reason`.
4. The flags.

`far_from_target` means the robot did not get to what it came for, whatever the cause. It is not set where the robot waited its turn or stood back with people watching the target, however far away that is. `target_distance_m` gives the distance.

Check by hand any record with a non-empty `not_ahead_of` and a `target_distance_m` beyond `max_keep_m` (3 m). No flag is set there. The robot may be standing back with one distant onlooker, or with someone who has nothing to do with the target (see "Known limits").

The `broken_m` of a `keep_out_of_view` with `target: true` is the larger of two values: its zone's violation, and how far the place is ahead of the people in `not_ahead_of`.

`people_cut` means more people were found than were kept. The ones dropped are the farthest from the target, which in a long line is its end. So a result with this flag and a `come_after` cannot be relied on.

## What the picture shows

`viz/<scene>_<episode>.png` shows the camera image beside a top-down plot. For a result of the method, both show the number of each person found, the place and the runners-up. Like every picture, they also show the target, the ground-truth people with their headings, and the acceptable and forbidden regions the scorer uses.

The plot shades the floor the robot may stand on by the solver's order. The darker a cell, the earlier it comes, and the place is among the darkest. That floor is solved again from the record with the default thresholds, so for a run with other thresholds the shading can differ from what the run chose on. The place and the numbers are drawn from the record.

The regions in the picture are the scorer's. The solver never sees them.

## What is measured of a run

The method uses the baseline's scorer and reference regions, so its goals are scored exactly like marked ones. It adds six measures. They are the last entries of a record's `measures`. In `summary.csv` they follow every other measure and come before `calls`, `cached_calls` and `cost_usd`. A column holds the mean over the row's records that measured it. It is empty where none did, which is every row of a baseline run.

| Column | What it is | Measured when |
|---|---|---|
| `people_found_share` | Of the people of ground truth in view, the share that a person found stands within twice `person_radius_m` of. Pairs are taken nearest first, and a person found counts for one person only. | `people: vlm`, and somebody is in view |
| `people_unmatched` | The number of people found that are left over, because nobody of ground truth stands there or the one who does was matched already. | `people: vlm` |
| `members_precision` | Of the numbers the answer lists under `come_after`, the share that are people waiting their turn at the target, in view. A number nobody has is listed and wrong. | the reference has people waiting at the target, a model stated the constraints, and the answer lists someone |
| `members_recall` | Of the people waiting their turn at the target, in view, the share the answer lists under `come_after`. | the reference has people waiting at the target, one of them is in view, and a model stated the constraints |
| `constraints_unusable` | The number of constraints the solver could not use, either `unusable` or `superseded`. | every result the solver made |
| `exclusions_broken` | The number of exclusions the place breaks. It counts the own space of each person it is inside, each `keep_out_of_shared_space` and `keep_out_of_view` it breaks, and `come_after` once. | every result the solver made |

The two measures about members are read from the answer as the model wrote it, before the solver takes anyone else as waiting.

The mean of a column counts each record once. So a scene counts once for each of its episodes, and a scene with one person counts as much as a scene with seven.

The six flags have their own columns, `flag_<name>_rate`. `sgi.rescore` carries all of these over as the run recorded them.

Latency is in `latency_s` and the `latency_<stage>_s` columns, as for the baseline. The method's two calls count under `goal`. Lifting the people and the solver count under `geometry`.

## Known limits

In the cases below, the solver picks a wrong place or its choice flips on a small difference. Most of them raise no flag. The cases with a test are in `tests/test_solver.py`.

**Who is last in a line is read from distance alone.** The last of the waiting people is the one farthest from the marked target point. No heading is read for this. So a line is ordered correctly only where its end is farther from the mark than its head.

- A line along the front of a wide thing, with the mark at its middle, is taken to end at its head. The robot is sent a metre beyond the person being served. In the test, with three people 0.8 m apart, it stands 2.6 m from the person who is really last.
- A line that turns back towards the target is ordered wrongly in the same way.
- In a single line that turns a corner with one person past it, the last two people are equally far from the mark. The robot then stands beside the end of the line instead of behind it, 1.0 to 1.5 m from the person who is really last.
- The scorer cannot detect the first two cases. Its reference rule for a waiting line also orders people by distance from the target, so in such a scene it would score the wrong place as right and the right place as wrong. Such a scene needs regions drawn by the exporter.

**Whoever stands behind the end of a line is taken as waiting.** A person the answer does not list joins the line as its new last member when they stand behind the last waiting person and within `standoff_m` of the place in turn. The test then repeats from them. So an answer that leaves out the last person does no harm.

The same rule also takes in a bystander, or a chain of them. With three passers-by one behind another behind a line of four, all three are taken, and the robot stands about 4.3 m behind the last person who waits. `joined_waiting` names those taken, and no flag is set. People the answer lists as a shared space are not taken. People it lists under a view are.

- A waiting person left out of the answer who stands too far behind the others is not taken. The robot then stands in the gap, ahead of them. In the tests the person is taken at 0.6 to 1.9 m between people, and not at 2.2 m.
- An unrelated person listed under `come_after`, farther from the target than the rest, becomes the end of the line.
- Two people who talk while they wait, given as a shared space and left out of `come_after`, are not taken as waiting. The robot stands beside the line, ahead of them.

**People who wait in a loose arc.** The place is behind one of those farthest from the target, and which one it is can change with centimetres.

**People who watch the target.** With a `keep_out_of_view` towards the target, the robot stands level with the hindmost of the listed people on the side of the target it was sent to, or behind them. That side is the one in front of the seen face. For a target without a seen face, it is the side the standard goal stands on.

Neither this rule nor the seen face applies where the place in turn is used, or where the target is a person. So where the answer states `come_after` for people who watch, the place in turn is kept and neither rule acts.

The counts below come from the 90 recorded runs that hold solver steps, solved again from their records (4344 records, 2295 solver steps, see "What has been measured").

- **The hindmost listed person decides, with no bound.** An unrelated person listed with a row of watchers, 2.7 m behind the row, keeps the robot back with them. In the test, seen from behind them all, the robot stands level with that person, 1.2 m beside them and about 5.6 m from the thing. `far_from_target` is not set where people are joined, so only `not_ahead_of` and `target_distance_m` show it.
- **Where the robot can stand on no place level with that person or behind, nobody is joined.** The rule cannot be met there, so it is not applied at all, to the row or to the person behind it. In the test the same four people are seen from the side, and the picture shows no floor level with the person at the back. The robot stands about 2.0 m from the thing, ahead of the row, inside nobody's own space and with no flag. `not_ahead_reason` names the person. `far_from_target` is judged there as if the rule did not exist. In the recorded runs this happens in 4 of the 72 steps on the scenes of people watching a thing. In each of them the model had placed people on the far wall. All four fail.
- **A sliver of floor.** Where floor level with the hindmost person is in view but all of it lies within `standoff_m` of them, nobody is joined either. The rule applies only where the robot can stand level with them or behind without breaking another exclusion more than it would ahead of them. So where some place breaks no exclusion, the rule never puts the robot inside a person's own space. In the test one person stands 4 m along a wall from a mark on it. The robot stands a metre in front of the mark, out of their view and more than 4 m ahead of them, with nothing broken and no flag. `not_ahead_reason` says why. None of the 63 recorded steps that join people breaks an exclusion, and none of the 2295 recorded steps has this reason.
- **One person is enough.** The rule treats every view of the target as people to join. It does not look at how many they are, how far off they stand, or whether the answer also asks to be near the target. In the test one onlooker stands 4 m in front of a thing. The robot stands 1.1 m beside them and 4.2 m from the thing. By default it stops a metre in front of the thing. Nothing is flagged. A `within_distance` about the target changes nothing, because it is a preference. Outside the scenes of people watching a thing, 14 of 2223 recorded steps join people, 10 of them more than 3 m from the target. Of the 14, 11 fail.
- **A false person on the watched thing.** When the people step places a person on the target itself, the solver takes that person for the target person. Neither rule then applies, and the robot goes round behind the thing. The record shows this by a null `face_m` and by `not_ahead_reason`. It happens in 3 of the 36 recorded steps of one model on the scenes of people watching a thing.
- **A target person the people step missed.** When the list of people lacks the person the robot needs, or places them more than a body's radius from where they stand, the solver treats that person as a thing. The seen face can then be fitted on their body. The face is used unless the answer names that person both to face and to be visible to. So with a `visible_to` of them and no `face` of them, it is used. In the test the robot then stays beside the person, at the plane of that face, and does not come round to where they look. The rule about people who watch applies as it would to any thing. In the test four people stand in a circle, and the one needed is listed 0.67 m off. The three others are joined, and the robot stands across the circle from the one it needs, 1.9 m from the point marked on them. In the recorded runs this acts in 6 steps of conversation groups, where it loses 1 success and gains 1.
- **A person behind a thing, named to face and to be visible to.** The exception above depends on the answer alone and does not look at the target. So where the target is a real thing and the answer names a person behind it to face and to be visible to, the seen face is ignored and the robot can go behind the thing. In the test someone stands behind a low thing and looks away from the robot. The robot goes round, 0.85 m behind the face and about 1.7 m from the mark, with no flag and a null `face_m`. When that person looks at the robot over the thing instead, the same answer leaves the robot in front of the face, with nobody else in front of the thing. The exception acts wherever the named person stands. Take three people talking in front of a free-standing thing, and an answer that states their shared space, asks for the place to be visible to the first and gives `face` 1. The robot goes 0.94 m behind the face (1.08 m from the mark), where the thing hides it from them. In the recorded runs, 24 steps state the two together where the face would otherwise be used. In 16 of them, ground truth has the target as a thing.
- **The mark on a narrow side.** A robot at the flank of a watched thing gets its target point on the thing's narrow side face. The people in front of the true front are then joined only where their centre lies in front of the plane of that side face. In the test, 4 cm decide. With the nearest of a row 2 cm in front of that plane, the robot stands behind the row, 4.7 m from the mark. With them 2 cm behind it, the robot stays a metre in front of the side face.
- People who watch the target and are stated with `target: false` are not joined. In the test the robot stands about 2.1 m from the thing, more than a metre ahead of the row.
- A thing without a seen face, such as a low one marked on its top, takes its side from the standard goal, which follows the robot's own line of approach. Take three people on one side of such a thing and two on the other, with the robot coming along the gap between the two sides. One person of each side is joined, and the robot stops about 1.77 m from the thing, nearer than any of the five.
- The seen face only orders the places. It removes none. Where every place in front of the face breaks more of an exclusion than some place behind it, the robot still goes behind, and `face_m` says how far.
- The fit takes the near side of a round post of 0.2 m radius for a seen face. Its front then points at the camera, wherever the robot stands.
- At a symmetric row, which end the robot stands at is a near-tie. The standard goal settles it to the millimetre. In the test, with hand-picked positions and no picture, 2 mm of the standard goal decide it, whichever side the robot comes from.

**One distance for everyone, and groups.** Own space is `standoff_m` round every person found, including the neighbours of the person the robot needs. People in a group stand closer together than that leaves room for, so own space wins over `visible_to`. The robot stands at arm's length beside the person it needs, well round from where they look.

In the tests, with six, eight and ten people in a circle, the robot stands at least 110, 120 and 125 degrees off the facing of the one it needs, behind their shoulder, with no flag set. The space between people adds to this at any spacing. Six people 2 m apart also leave the robot 105 to 120 degrees off. Expect `visible_to` to go unmet in any group of three or more.

**A crowd keeps the robot out.** Where people stand closer together than 1.41 m (the square root of 2 times `standoff_m`), no cell among them is free of every own space. A cell that breaks no exclusion beats every cell that breaks one, so the robot stays outside, metres from the person or thing it needs.

In the test, with the person needed in the middle of 25 people 1.4 m apart or less, the robot stops 3.2 to 3.7 m away with no exclusion broken. `far_from_target` is set there. The flag is set only beyond `max_keep_m`, so a somewhat tighter crowd can leave the robot just inside that distance with nothing flagged.

**Near-ties.** When two places are equally good to within a cell, where the grid's cells fall decides between them, and the grid lies in the robot's frame. So the place can move from one side of a person to the other when the robot itself stands a few centimetres elsewhere. Both places break no exclusion and rank the same on preferences, so either is right. With four or more people listed as one shared space, which pairs count as neighbours can also turn on centimetres.

**Smaller cases.**

- If the people posing for a picture are listed under `keep_out_of_view` together with the person taking it, the floor behind those posing is not excluded. Listing the picture-taker alone excludes it.
- A `within_distance` of nearly zero about a thing is obeyed only as far as the floor allows. About a person, own space keeps the robot at `standoff_m`.
- The floor is what one depth image shows. The robot can cross floor the image hides, but not stand on it.

**What the method does not include.** The method has no worked examples for the constraints step, and no person detector other than the model. It has no reference constraints for a picture being taken or for people waiting at a lift. Both situations have a reference rule ([`scene_format.md`](scene_format.md), section 7) and generated test scenes, so both are scored. A separate layer, `sgi.resolve`, handles moving people. It solves again on tracked people with no new model call.

## What has been measured

**By the tests, with no model.** The suite runs the solver on hand-made layouts, field by field and key by key, and on the generated test scenes.

`tests/test_reference_constraints.py` checks the solver with reference constraints. It covers every generated test scene with nobody at the target, a waiting line, a conversation group or people watching a screen. For each of these scenes it writes the constraints a careful answer would hold, from who is in the group at the target. For a scene of another situation it gives none. It takes the people from ground truth and asserts the solver's own properties:

- The place breaks no exclusion where some cell breaks none.
- No cell the robot could stand on comes before it in the order.
- No such cell lies outside the scene's static free space.

Where people watch a screen, the test also asserts that the view is not broken, by the solver's own figure. It checks one more thing on the place itself, without the solver's figures and without the scorer. The robot must stand level with the hindmost of them or behind.

The test then prints what the scorer says of those places, per situation, as a count with no pass mark:

```bash
.venv/bin/pytest tests/test_reference_constraints.py -rP
```

This count checks the solver with reference constraints, with no model involved. For a waiting line, the place in turn is built the same way as the reference region, from the last person by distance. With the right members it lies inside that region by construction. For people watching a screen, the reference rule also keeps the robot no nearer the watched thing than those who watch it, as the solver does.

**On recorded runs, with no model.** A record of the method holds what the solver was given: the target point, the people found and the constraints as the model wrote them. A replay tool, `results/still-scenes/analysis-scripts/harness/replay_solver.py`, solves every such record again with the current solver, on the scene the run used, and scores the new place. Run over the recorded runs listed in `replay_runs.txt` beside it, the tool is the regression gate for any change to the solver.

Per situation and in all, the tool prints the successes as recorded and as replayed, how many records gained or lost success, and every record that lost it. It fails when a record that is not on its list of accepted losses has lost its success. That list holds one known loss, the record named under "A target person the people step missed" above.

`replay_runs.txt` names 90 runs with 4344 records and 2295 solver steps. The runs and their scenes are not in git. The tool skips any record of a held-out situation without opening its scene, whether the run's own config or the current default holds the situation out. The 90 runs have no such record.

Replayed, 3244 of the 4344 records succeed:

| Situation | Records that succeed |
|---|---|
| Nobody at the target | 1521 of 1872 |
| Waiting lines | 897 of 1500 |
| Conversation groups | 771 of 900 |
| People watching a thing | 55 of 72 |

These counts replay the solver on recorded answers. Where people watch a thing, the solver's rule has the same shape as the reference rule, so a success there follows partly from that shape.

**On real models.** The success rate of the method, its latency and the comparison with the baseline are in [`results/README.md`](../results/README.md).

**Coverage of the kinds and situations.**

- No test applies `within_distance` to a scored episode. `keep_out_of_view` is applied to one only as a view of the target. That happens in the three generated test scenes of people watching a screen, which the test above solves with reference constraints. Otherwise the two kinds are covered by the hand-made layouts of the solver's tests.
- People watching a screen, a picture being taken and people waiting at a lift each have a reference rule and generated test scenes. Seven of these scenes have people, and the lift and the screen appear once each with nobody at them. The four scenes of a picture being taken and of people waiting at a lift have no reference constraints, and the test above solves them with none.
- Two of the held-out situations, a picture being taken and people waiting at a lift, have reference regions and no worked example, so the goal stage is shown no example of either. Their results are in [`results/README.md`](../results/README.md).

The baseline's results on the same scenes are in [`results/README.md`](../results/README.md).
