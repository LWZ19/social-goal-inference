# I06 first probe (crowd variants, zero calls): frozen protocol

## 0. Rulings, choices and what was run before hashing

1. Scope: the first probe, zero calls, families A–E. Family F is deferred (section 1).
2. Base layouts: the development rows of seeds 0, 1 and 2 that a family applies to, 10 per seed, 30 in all, reproduced with the package's generator and checked against the whitelisted scenes. The REPORT said only "dev layouts (seeds 0–2)".
3. The five families' frame (A lines only; B a 3-person cluster at 1.2 m spacing; C a second group 1.5 m away; D 0, 2 or 5 bystanders; E one person on the line of sight at 2 depths), non-member people only, regions from sgi.gt unchanged (L1), the package's scene path, streams rooted at `[seed, 2, family, row, level]`, at most 5 redraws. The REPORT gave the families in one line each.
4. L2 = L1 minus cells within a 0.3 m body gap of any non-member and minus other groups' circles, scored as a pose-level check on top of the package scorer. L2-empty episodes are reported.
5. Oracle scoring: full-oracle C (oracle grounding, oracle context, oracle people, reference constraints, the package solver) under L1 and L2; also the solver alone and full oracle.
6. The four documented failures: three named, the fourth chosen here (section 7).
7. The perception-noise model: its data, parameters, simulation and validation (section 8).
8. Criteria (a)–(e) and the kill lines verbatim, with "drop" fixed as against the family's base scenes, pooled over seeds, per family.
9. The C5 gate-1 recount (section 10).

**What I ran before hashing** (in `precheck/`, each with its log):
- `format_check.py` reproduced the seed-0 base layout `eval_06_waiting_line_1`. It drew rows eval_00 to eval_06 of the main stream `default_rng(0)` through the package's accept step (`generate._accept`) in a temporary folder under `precheck/`, then removed the folder. It compared the result with the whitelisted `data/placeholder/eval/eval_06_waiting_line_1`, loaded through the harness. People (ids, positions, yaws, groups, roles), the target (anchor, footprint, front yaw), `T_world_camera`, the robot pose, and both episodes' acceptable regions and representative points are all equal (maximum difference 0). No layout was rejected on the way.
  - The realistic `real_`, `staffed_` and `plain_eval_06_waiting_line_1` have the same members and anchor (difference 0). `real` and `staffed` add one person with the role `staff`. So `realistic_final` holds seed 0's first pass. It is not used here: every base and variant is rendered with the placeholder renderer.
  - The whitelisted development scenes on disk are: `data/placeholder/eval` (eval_00–13; seed 0, first pass; `keep/placeholder_first14` links to the same 14), `keep/realistic_final/{real,staffed,plain}/scenes` (seed 0, first pass, Blender renders), and `keep/dev_new/seed{0,1,2}/scenes` (eval_14, eval_15, eval_20, eval_21 of seeds 0, 1, 2; second pass).
  - Solver note fields: `keys` (`excluded_m`, `excluded_total_m`, `turn_m`, `face_m`, `preference_m`, `standard_m`), `own_space_broken_m`, `joined_waiting`, `not_ahead_of`, `not_ahead_reason`, `target_distance_m`. Flags: `marked_point_off_floor`, `standard_goal_blocked`, `standard_goal_frontal`, `heading_fallback`, `heading_degenerate`, `no_usable_constraint`, `constraint_unusable`, `constraints_cut`, `exclusion_broken`, `people_cut`, `far_from_target`.
  - For that one layout (a line of 1), a body axis on the line's axis is in the image (at least 3 of the 9 visibility heights project inside it, occlusion aside) up to 2.32 m behind the last member.
- `depth_check.py` repeated the reproduction and found the rgb and depth images byte-identical to the whitelisted scene's.
- Thresholds (default config): person radius 0.25 m, robot radius 0.30 m, standoff 1.0 m, personal gap 0.3 m, relevant radius 2.0 m, visibility 3 of 9 samples, `max_people` 40, view half-angle 35 deg, `max_keep_m` 3.0 m, grid 0.05 m.

**What that informed (data-informed, disclosed).** The 2.32 m figure, together with the recipe's camera rule read from code, set family A's largest gap at 1.9 m (section 4). By that rule, the in-view limit is about 2.0–2.1 m for lines of 2–4 people and about 1.4 m for the line of 6. Only base-layout geometry was looked at. No variant was built and nothing was scored.

## 1. Scope

- I06's first probe, zero calls, families A–E. No model-provider or network call, no GPU, no torch.
- Outputs are solver-degradation curves under oracle inputs and simulated perception noise. They are not RQ2 curves: R and E are not run, and C with real VLM perception needs the VLM crowd run (Programme B step 8).

## 2. Base layouts

- **Rows** (index in the eval split; recipe, amount, bystanders, second group), applied to seeds 0, 1 and 2:
  - lines: eval_06 (1), eval_07 (2), eval_08 (4), eval_09 (6), eval_13 (3);
  - conversation groups: eval_10 (2; 2 bystanders), eval_11 (3; a second group), eval_12 (4; 1 bystander and a second group);
  - watching: eval_14 (audience 3), eval_15 (audience 5).
  - 10 per seed, 30 in all. Nobody-near rows and examples rows are not used.
- **Reproduction**, in memory or temporary folders under `builder/`, with the package's own functions and streams:
  - First pass: `default_rng(seed)`. Rows eval_00 to eval_13 go in order through `generate._accept`. eval_00–05 are drawn only to advance the stream and are then discarded. Drawing stops after eval_13, so no examples row is drawn.
  - Second pass: `default_rng([seed, generate._MORE_STREAM])`. Rows eval_14 and eval_15 go through `_accept`, then drawing stops. The group_photo row eval_16, and every row after it, is never instantiated. So eval_20 and eval_21 are never base layouts here.
  - Each base gets its regions through `generate._with_regions`. No tour is built: no arm here reads a tour.
- **Equality checks, before any variant is built.** Each check is exact (people ids, x, y, yaw, group and role; targets; `T_world_camera`; robot pose; both episodes' acceptable polygons and representative points; the rgb and depth arrays):
  - seed 0, eval_06–13, against `data/placeholder/eval`;
  - seeds 0, 1 and 2, eval_14–15, against `keep/dev_new/seed{N}/scenes`.
  - All whitelisted scenes are loaded through the harness (`H.scene` with a listed run of that folder).
  - eval_06–13 of seeds 1 and 2 have no whitelisted counterpart on disk. Their correctness rests on the same code path, which the seed-0 first-pass checks and the second-pass checks of all three seeds confirm.
  - If any check fails, nothing is generated and the failure is reported.
- **Level 0.** For every family, level 0 is the reproduced base scene, scored by the same arms as the variants. For seed 0 it is byte-identical to the whitelisted scene.

## 3. The builder (`builder/`; research code that imports the package and changes nothing in it)

**Streams.**
- Every attempt of a variant continues the same generator. No variant draws from a base stream, so the base scenes stay byte-identical.

**People added** (all families):
- Each is made with the package's `recipes._person(rng, index, x, y, yaw, shirt, group)`. That gives ±0.05 m position jitter and ±0.15 rad yaw jitter, and the figure's look comes from `appearance(index)`.
- Ids continue the scene's numbering: `p<len(people)>`, and so on.
- Shirts come from `recipes._shirts(rng, k, taken=colours already worn)`, as `_on_free_spot` does. Red, the target's colour, is never drawn.
- Every added person has role None. Group is None except in family C, whose group gets the first free id among `g1`, `g2`, …. All added people are non-members: none belongs to `group_at_target`.

- It is the base `Layout` with the added `(Person, Figure)` pairs appended: `figures[i]` still draws `people[i]`.
- The room, its bounds, the furnishings, the target's parts, the robot and the camera all stay as the base has them. The package's `_with_more_people` re-furnishes the room from the stream instead. I do not use it, so that the added people are the only change between a variant and its base, and a drop can be laid on them.
- Consequence: the generator's picture rules for furnishings (none partly hidden behind a person) can be broken in a variant. That matters only to a later VLM run, which may need re-furnished scenes. It is not a question for this probe.

**Pre-checks on each added person, before rendering.** They use the base geometry plus the people already placed:
- (c1) Room: the centre is inside the bounds and ≥ 0.5 m from them, and ≥ 0.30 m (person radius + 0.05) from every obstacle polygon (target footprints and furnishings).
- (c2) Bodies: ≥ 0.5 m (two person radii) from every other person's centre. Each family adds its own stricter spacing.
- (c3) Robot: ≥ 1.5 m from the robot's centre.
- (c4) In the picture: the bearing from the camera's ground position, off the robot's heading, is within `recipes._VIEW_INSIDE_RAD`. That is the package's limit for what a picture holds whole: 4 deg inside the image's side, room for a body's width.
- (c5) Clear sight, the package's `recipes._in_clear_view` rules without its 22-deg cone, re-implemented in `builder/`:
  - (a) the robot's ground line of sight to the person passes no other person, existing or added, within `_SIGHT_GAP_M` (0.3 m), and no target footprint within 0.3 m;
  - (b) no other person's line of sight passes within 0.3 m of the person;
  - (c) the person is not in the bearing wedge between a place's anchor and the first in line, widened by 0.3 m over the distance.
- (c6) Target sight line: the camera-to-anchor ground segment passes no added person within 0.3 m.
- (c7) Not before the target: no added person stands nearer the camera than the target footprint's farthest corner while inside the footprint's bearing span, widened by 0.3 m over the distance.
- (c8) Not in the structure's space: ≥ 0.3 m outside every forbidden polygon of the base episodes (line corridor, group circle, the space between an audience and the screen), and ≥ 1.0 m from the target footprint.
- Which apply: B, C and D use c1–c8. A uses c1–c3, c5 and c6 (the scene check decides whether the confuser is in view). E uses c1–c3 and c6, plus c5(a). E waives c5(b), because partly hiding relevant people is the point; the scene check's visibility rule decides.

**Sampling inside one attempt.**
- B, C and D, like the package's `_on_free_spot` and `_add_extras`: up to 200 position draws per person (or per cluster or group), people placed one after another. If any draw fails, the whole set starts again, up to 20 times. A and E: up to 200 jitter draws.
- If the draws run out, the variant is rejected at once, reason `no_spot`. The inner loops have already tried.

**The package's scene path.**
1. `generate.build_scene(layout, scene_id)` renders rgb and depth with the placeholder renderer.
2. `save_scene` writes it.
3. `generate._rejection(dir, cfg)` checks it. It rejects on any error, or on any warning other than check 9, exactly as the generator does. Check 8, "every relevant person visible", is among them. The standard-goal reachability test applies only to no_one episodes, and there are none here.
4. If accepted: `generate._with_regions`, then `save_scene` again. No tour.
- If the scene check rejects, the variant is redrawn from the same stream, at most 5 times (at most 6 attempts). If all fail, the variant is recorded as rejected with each attempt's reason.
- Scene ids: `i06_s<seed>_<family><level>_eval_<row>_<name>`. Folders: `scenes/seed<seed>/`.

**Records** (`manifest.jsonl`, one line per variant and per base):
- seed, row, family, level, attempts and reasons, accepted or not;
- the added people (id, x, y, yaw, group) and the stream key;
- for every person: whether relevant (`sgi.gt.derive.relevant_people`) and the visible fraction, as the samples seen of the 9 on the axis (`derive._sample_seen`), in the variant and in its base;
- for E, and for any person under 9 of 9 in another family: the pixel share shown (`generate._share_shown` on that person's figure).

**Self-tests before generation.**
- c1–c8, with c5 re-implemented, must accept every bystander and second-group member the package placed in eval_10–12 of the three seeds. A refusal is investigated and reported before any variant is generated.

## 4. Families (each level is one scene per base layout where the family applies)

"Members" are the people of `group_at_target` of episode e0. Distances are centre to centre on the floor.

**A, a confuser extending the line (lines only: eval_06–09, eval_13).**
- Geometry, from the members as the reference rule reads them:
  - the last member L is the farthest from the target's anchor;
  - the one before, B, is the second farthest (the anchor itself for a line of 1);
  - the line's axis u = (L − B)/|L − B| points away from the target, and the line faces f = −u.
- The confuser stands at L + g·u and faces yaw(f) + s·φ, where s = ±1 is drawn from the stream. Both then get `_person`'s jitter.
- Levels, in rising cue strength:

| Level | gap g behind the last member | facing φ off the line's facing | Reading |
|---|---|---|---|
| A1 | 1.9 m | 180 deg (back to the target) | clearly not waiting |
| A2 | 1.6 m | 135 deg | |
| A3 | 1.3 m | 90 deg (side-on) | |
| A4 | 1.0 m | 60 deg | |
| A5 | 0.8 m | 30 deg | |
| A6 | 0.6 m | 0 deg (as the members) | looks like the next member |

- Why these:
  - 0.6 m is the line's own spacing (`_LINE_SPACING_M`), so A6 is a seventh member in all but the label.
  - At 1.9 m, plus the 0.05 m jitter, a body axis stays in view for lines of up to 4 people (section 0).
  - The solver takes an unlisted body behind the last member as waiting when it stands within `standoff_m` of the place in turn, 1.0 m behind the last; on the axis, that is any gap under 2.0 m (constraint_method.md: taken at 0.6–1.9 m, not at 2.2 m). So **every A level lies inside the solver's absorption distance, and the solver reads no facing.** A tests whether the solver ignores a clear "not waiting" cue, not where its distance limit lies. A level safely beyond 2.0 m (past the jitter) would be out of view for most lines: the in-view limits are about 2.3 m for 1 person, 2.0–2.1 m for 2–4 and 1.4 m for 6.
- Expected rejections, which are counted: A1 and A2 for the line of 6 (out of view; check 8).

**B, a 3-person cluster (all 10 rows).**
- Three non-members on the corners of an equilateral triangle with 1.2 m sides, so every pair is under √2 · `standoff_m` = 1.41 m. Each faces a yaw drawn from U(−π, π), as the package's bystanders do.
- Placement:
  - draw a bearing θ ~ U(−π, π) from the members' centroid, and the triangle's turn ψ ~ U(0, 2π/3);
  - bisect along θ for the centroid distance at which the cluster's nearest person is exactly 1.2 m from the nearest member;
  - then apply the jitter and checks c1–c8;
  - every cluster person is also ≥ 1.2 m from every non-member already in the scene.
- **Distance from the structure: 1.2 m, nearest pair.** That is the reference rules' far join distance (`_JOIN_FAR_M`), so no cluster person stands in the members' region. The cluster and the structure then form one crowd with no neighbouring gap above 1.41 m.

**C, a second group (all 10 rows, where it fits).**
- **Size 3**, laid out by the package's own conversation geometry for 3: neighbours 1.2 m apart (`_GROUP_NEIGHBOUR_M`), circle radius 0.6/sin 60° = 0.693 m, each member facing the circle's centre. The group gets a new group id.
- Placement:
  - bearing θ ~ U(−π, π) from the members' centroid, turn ψ ~ U(0, 2π/3);
  - bisect for the circle-centre distance at which the nearest pair (a second-group member and a member) is exactly **1.5 m** apart;
  - every second-group member is also ≥ 1.2 m from every other non-member;
  - then c1–c8.

**D, bystanders (all 10 rows; levels 0, 2 and 5; level 0 is the base).**
- k = 2 or 5 non-members, each facing U(−π, π), placed one after another.
- Rule for each:
  - distance to the nearest member in **[1.2, 2.5] m**, drawn uniformly over that band's area (rejection sampling in the members' bounding box grown by 2.5 m);
  - **≥ 1.2 m from every other person**: members, existing extras, earlier bystanders;
  - **not on the line of sight to the target** (c6), nor before it (c7); then c1–c8.
- Why: 1.2 m is the join band's far edge, so nobody stands in the members' region. 2.5 m keeps every bystander within the relevant radius (2.0 m) of the region, so the scene check's visibility rule covers them, and it matches the REPORT's "confusers sit 1–2.5 m from structures". 1.2 m spacing lets ordinary draws make pairs under 1.41 m apart, with members or with each other.

**E, one person on a line of sight (all 10 rows; two depths).**
- One non-member facing U(−π, π), on the ground segment from the camera's ground position (the translation of `T_world_camera`) to the **representative point of the base episode e0's first acceptable region**. That is the point sgi.gt derives from the members, where the full oracle stops.
  - **E1, near the structure:** 1.5 m from the representative point, towards the camera.
  - **E2, near the robot:** 2.0 m from the camera's ground position.
- Then the jitter, and c1–c3, c5(a) and c6. The position is otherwise fixed, so a failed check rejects the variant.

**Planned variants.** Per seed: A 30, B 10, C 10, D 20, E 20, so 90. Over three seeds: 270 variants plus 30 bases.

## 5. Labels

- **L1** (the project's own rule, as today): the acceptable regions of the variant's episodes, derived by `sgi.gt` (`_with_regions`, from `group_at_target`'s members only). Scored by `sgi.evaluate.score`: target matched, reachable, in region, heading within tolerance.
  - **Invariance check:** every accepted variant's acceptable polygons equal its base's. The members are unchanged, so they must. Representative points may move: people block the reach grid, and E hides floor.
- **L2** (a pose-level check on top of L1; the project has no polygon library). An episode succeeds under L2 when it succeeds under L1 and its goal (x, y):
  - has a body gap ≥ `personal_gap_m` (0.3 m) to every non-member. Body gap = centre distance − person radius − robot radius, exactly as `evaluate.score` computes `min_body_gap_m`;
  - lies outside every other group's circle. For each group id other than `group_at_target` with ≥ 2 members, existing or added: the disc whose centre is the members' mean and whose radius is the largest member distance from it (the conversation rule's own circle, `regions.region_conversation_group`).
  - Non-members are everyone outside `group_at_target`, including the bystanders and second groups the base scenes already hold (eval_10–12).
- **L1 cells:** points of the 0.05 m lattice (`sgi.polygons.grid_points_in`) inside e0's acceptable polygons, on a free cell of the scorer's robot grid (`freespace.robot_grid`) joined to the robot (`freespace.connected`). Heading is not used.
- **L2-empty:** a scene with L1 cells of which none meets the two L2 conditions. Reported for every accepted variant and base.
- **Disagreement:** an episode whose L1 and L2 verdicts differ. Only L1 success with L2 failure can occur.
- **Unit.** Both episodes of a scene have one target and one region set. Under every arm's inputs they get the same result: asserted per scene, and any mismatch is reported. So rates are given per scene (= per episode), with n = scenes. Episode counts are twice that.

## 6. Scoring arms (every accepted variant and every base)

1. **Full oracle.** The package's `run.run_episode` with grounding oracle, `context_check: always_yes`, goal oracle and the mock model (README's full oracle). Scored under L1 and L2.
2. **Solver alone.** `run_episode` with grounding oracle, `context_check: oracle`, goal constraints, people oracle and constraints none (no call).
3. **Oracle C (primary).**
   - Target: `derive.oracle_target(observation, t0, th)`, the point `run_episode`'s oracle grounding lifts.
   - Context: oracle, `reference_people_matter(et, "t0")`. If False, the standard goal is the result.
   - People: `run.OraclePeople`, the visible true people, then `stages.people.number_people`.
   - Constraints: the reference constraints from the truth, copied as I01 R5 copied them (`results/still-scenes/replay-harness/repairs/repairs.py`, from `tests/test_reference_constraints.py`):
     - waiting_line: `come_after(members)`;
     - conversation_group: `keep_out_of_shared_space(members)`, `visible_to(needed)`, `face(needed)`;
     - audience: `keep_out_of_view(members, target=True)`, `face(target=True)`;
     - members are the numbers of the visible members of `group_at_target`, and needed is the number of the target person.
   - Then `solver.solve(observation, target, people, constraints, standard_goal(...), th)`, and a `GoalResult` built as `ConstraintInferrer.infer` builds it.
   - The reference constraints read the situation label and the true group. They are the oracle answer, not a method, and circular (REPORT section 8). The solver and the scorer have no situation branch.
   - **Equivalence check:** arm 3 with an empty constraint list must give arm 2's place on every scene, to 1e-9 m.
4. **Noisy C.** Arm 3, but the people are simulated with each model's noise (section 8). The reference constraints keep the numbers of listed people, mislocated ones included, and drop the numbers of missed ones. 20 draws per scene per model.

**People numbers to identities** (for the detectors): in arm 3 each number's person is the true person at that exact position. In arm 4 it is the simulated person's source; false people have none.

## 7. The documented failures (criterion b) and the named causes

A failure **fires in a variant** when its detector is true in the variant and false in the variant's base: it is then caused by the added people. Detectors are read on arm 3, and reported descriptively for arms 2 and 4. A variant is **relevant** to a failure when the failure's precondition holds and its base does not fire.

- **(i) Bystanders behind a line absorbed as waiting** (BRIEF 3.5; constraint_method.md "Whoever stands behind the end of a line is taken as waiting").
  - Detector: the note's `joined_waiting` holds a number whose person is a non-member.
  - Relevant: accepted variants where a `come_after` is used (the line rows), all families.
  - A triggers it by design. B, C, D and E can trigger it wherever an added person stands behind the last member.
- **(ii) "A crowd keeps the robot out"** (people under 1.41 m apart; documented stop 3.2–3.7 m away).
  - Let F be the L1 cells at ≥ `standoff_m` from every visible person.
  - Detector: F is empty in the variant and not in the base, the solver's place is outside L1 (`in_region` false), and `keys.excluded_m` = 0 (it found a place that breaks nothing, elsewhere).
  - Recorded: `target_distance_m`, and the smallest centre distance between two people whose own spaces cover L1 cells.
  - Relevant: all accepted variants whose base has F non-empty. Every family can trigger it.
- **(iii) Large-circle off-facing** (the robot stands 110–125 deg off the facing of the person it needs).
  - Detector, where the target is a person in a group of ≥ 2 (the conversation-group rows): |wrap(bearing(target person → place) − target person's yaw)| ≥ 110 deg.
  - Relevant: accepted variants of those rows whose base is under 110 deg (families B–E; A cannot, it adds to lines only).
  - The families do not enlarge circles (members are fixed by L1). They can push the robot round only through non-members' own space and hidden floor.
  - This is not an L1 failure: the region rings the whole group, and the robot faces the person anyway.
- **(iv) Chosen: the brief's row "the hindmost of those who watch decides, with no bound"** (constraint_method.md, "People who watch the target"), read through that section's floor cases.
  - The rule is dropped silently when no cell level with or behind the hindmost listed watcher is free, or every such cell breaks another exclusion more than some cell ahead. The robot then stands ahead of the row, with no flag.
  - Detector, where a `keep_out_of_view` towards the target is used (the watching rows): `not_ahead_of` is empty and `not_ahead_reason` starts with "no cell is level with or behind" or "every cell level with or behind" in the variant, while the base has `not_ahead_of` non-empty. Also recorded: whether the place is ahead of the hindmost member, and its L1 verdict.
  - Relevant: accepted watching variants of B–E whose base applies the rule. Bystanders behind or beside the row take the floor there; E hides it.
  - **Why this row:**
    - "Who is last in a line is read from distance to the target alone" cannot be triggered here: the members, and so the straight line and its order, are fixed. Non-members reach the line only by being absorbed, which is (i). Under noise, a grossly mislocated listed member can trigger it, but at a rate that does not depend on the family.
    - The row's own example (an unrelated *listed* person far behind) needs a listed non-member, which the reference constraints never hold.
    - The `max_people` cut cannot fire: at most 12 people in any variant (eval_12 with D5), against 40.
    - Near-ties are not a failure of the place.
- **(v) Named cause, not one of the four:** the region's floor is not standable. Every L1 cell is not `standable` on the solver's own floor (`solver.floor(observation, people, th)`), and the place is outside L1. This is E's mechanism. It counts only as a named cause for criterion (a).

## 8. The perception-noise model (fitted on development runs only)

**Fit data.**
- The old-wording C people answers of three models, each through the harness: `H.runs(stage, arm, model, method="con")`, then `H.records`, then `H.scene`.
  - Qwen: s19 `qwen38_low`, arms real, staffed, plain, ph, dev_seed0–2;
  - GLM: s19 `glm53`, the same arms;
  - DeepSeek: s20 `dsv41`, the same arms.
- An answer is the people list of a solver-step record (`Record.solver_step`), one per (run, scene, trial). The other episode's list must be equal; if it is not, both are kept, counted.
- Truth: the visible true people (`derive.is_visible`, under each run's recorded thresholds).
- **Matching:** `run._matched_people` (nearest first, one to one, within 0.5 m), as I03's scorer does, against all visible people.
- **Role groups:** members of the group at the record's target (`truth.group_for(episode, "t0")`); others (non-staff non-members); staff (role set). Staff do not occur in the variants. Their rates are used only to simulate the development scenes in the validation.

**Per visible person, one outcome:**
- **located:** matched;
- **mislocated:** not matched, but paired with an unmatched detection. Pairing is greedy, nearest first, one to one, by the distance between the detection's recorded pixel and the person's mark projection (`person_mark_height_m` = 1.1 m, as `OraclePeople`), within **40 px**. That is I03 Stage B's registered relocation radius: data-informed on these runs, and I add no new radius. This is I01's mechanism: unmatched detections treated as mislocated people;
- **missed:** neither.
- Unpaired unmatched detections are **false people**.

- p_miss by role group;
- q_mis = P(mislocated | listed), pooled over roles;
- located xy error: the empirical (radial, lateral) offsets in the camera-ray frame. Origin at the camera's ground position; radial runs along the ray through the true person, away from the camera; lateral is to its left. Pooled over roles;
- mislocation offset: empirical (radial, lateral) of the paired detection from its person. Taken from the three models pooled if a model has fewer than 10 pairs; its rates stay its own;
- false people per answer: the empirical count distribution. Their positions are (radial, lateral) offsets from the visible person whose mark projects nearest in pixels (pooled over models if fewer than 10). Their yaw is None in the model's observed share and U(−π, π) otherwise;
- yaw, by A0 view class: frontal < 45, oblique 45–75, profile 75–105, back > 105 deg. The angle is |wrap(bearing(true person → camera xy) − true yaw)|. Per class, P(yaw unknown) and the empirical signed error on matched pairs. Pooled over models for a class with fewer than 10 known pairs. Mislocated people take the same yaw model.
- **Cross-check:** recall computed I03's way (all visible, staff included) must equal `results/still-scenes/perception/depth-blob-detector/stageA/vlm_rows.jsonl`'s pooled recall for the same runs. Any mismatch is reported and stops the noisy arm until explained.

**Simulation of one draw on a scene.**
- For each visible true person (OraclePeople's list): missed with p_miss[model, role]; if listed, mislocated with q_mis[model], else located.
- Located and mislocated people keep the true person's mark projection as their pixel, and get a resampled offset (applied in their own ray frame) and a yaw draw.
- Add false people: the count is resampled; each is anchored to a uniformly drawn visible person, with a resampled offset and the pixel of its own position (dropped if that does not project in front of the camera).
- Then `number_people`: merge, cut, number.
- Reference constraints: the numbers of listed members, and needed when the target person is listed. Otherwise `visible_to` and `face` are omitted, as in I01. `come_after` with nobody is kept, and the solver treats it as unusable.
- RNG `default_rng([20261018, model, crc32(scene key), draw])`, with model 0, 1, 2 for Qwen, GLM, DeepSeek; draws 0–19. The scene key is the scene id for bases and variants, and "run folder/scene id/trial" for development records, so both episodes of one trial share their draws, as they share one people answer. These are noise seeds, not generator seeds.
- Applied **identically to bases and variants**. The noise model has no branch on situation, scene name, family or level. Its role groups read group membership only.

**Validation (the kill line).**
- On every allowed record of the model's seven fit runs (development scenes, realistic and placeholder):
  - a solver-step record is re-solved 20 times with `H.resolve(rec, people=simulated, constraints=reference)`, keeping the recorded target. Its simulated success is the mean over draws;
  - every other record keeps its verdict;
  - simulated C = the mean over all the records.
- Recorded C = the same records' success under the current scorer (`H.rescore`). Records whose recorded success differs from the rescore are counted.
- **Gap = simulated − recorded, in points**, pooled over the seven runs. Also given per domain and per situation, descriptively.
- Diagnostic, outside the rule: the same records with each answer's actual people and the reference constraints (I01's R5 construction, over all seven runs). Simulated − R5 isolates the people-noise fidelity. R5 − recorded is the part due to the model's own constraints.
  - Expected bias, stated in advance: on realistic records, I01 found R5 above recorded C by 20 (GLM) and 24 (DeepSeek) of 252 records, 7.9 and 9.5 points. So the gap may exceed 10 points for those models with the people noise modelled well.
- Secondary arm (descriptive, decides nothing): the same simulation with rates fitted on the placeholder answers alone (ph and dev_seed0–2), on the top-level variants (section 9) and their bases.

## 9. Criteria and kill lines (REPORT I06, verbatim, with the readings fixed here)

"Build the full pass only if all of":
- **(a) "Oracle C drops at least 15 points in at least 2 families, or falls monotonically along family D or A with named causes."**
  - Drop of family f = the mean, over its accepted variants v, of [success(base(v)) − success(v)] × 100. That is arm 3 under L1, levels and seeds pooled: paired to each variant's own base, so a rejection removes the pair.
  - Monotone fall along D (levels 0, 2, 5) or A (0, A1, …, A6, in rising cue strength):
    - the per-level paired drop is ≥ 0 at the first level and non-decreasing level after level;
    - the top level's paired drop is ≥ 10 points;
    - ≥ 50% of the top level's failures carry a named cause, (i)–(v).
- **(b) "At least 2 of the 4 documented failures fire in at least 10% of relevant variants."** On arm 3, by section 7's definitions.
- **(c) "L1 and L2 verdicts differ in at most 5% of episodes. Between 5% and 20% passes, with side-by-side reporting and an project decision. L2 empty in at most 2%."**
- **(d) "At least 70% of variants pass `check_scene`, and full oracle scores 100%."**
  - Pass rate = accepted / planned variants (270), where a variant is accepted within its attempts. Rejections of every kind count against it, `no_spot` included.
  - Full oracle (arm 1) must score 100% under L1 on every accepted variant and every base.
  - First-attempt pass rates are reported too.
- **(e) "Noisy perception separates from oracle by at least 10 points at the top level."**
  - Top level = A6, B, C, D5 and E2, pooled.
  - Separation of a model = arm-3 success − arm-4 success (mean over draws) on those variants, × 100.
  - Met when ≥ 10 points for at least 2 of the models with a valid noise model. Not met if fewer than 2 are valid.
  - The separation on the bases is reported beside it, since the noise also hits scenes with no crowd.

"Kill, if any of":
- **K1 "Oracle C at least 95% everywhere, and noise within 5 points."** Arm-3 success ≥ 95% at every (family, level), pooled over seeds, and every valid model's separation ≤ 5 points at every family's top level. With no valid noise model, the first half alone decides.
- **K2 "L1/L2 disagreement above 30%, or L2 empty above 10%."** Arm 3, as in (c).
- **K3 "More than 50% of layouts rejected."** Read as variants. No base layout can be rejected: all 30 are reproduced.
- **K4 "The noise model misses the recorded C by more than 10 points. Then report oracle curves only."** As read in section 8.

**Sensitivity, disclosed in advance; it does not change the verdict.**
- (a)'s monotone clause can be met through A by construction, because A was built around failure (i).

**Reporting.**
- Per (family, level): n accepted and n rejected, by reason; arm 1–3 success under L1 and L2, and each model's arm 4 (mean over draws); paired drops; the detectors; flags (`far_from_target`, `exclusion_broken`); `target_distance_m`.
- Layout-clustered bootstrap 95% intervals (resampling the 30 base layouts, 2,000 draws, `default_rng(0)`) for drops and separations. These are descriptive only: decisions use point estimates.
- Per-base-layout tables.
- Anything added after results is labelled post hoc.

## 10. C5 gate-1 recount on the widened generator

Under `results/still-scenes/learned-models/flow-model-gates/FROZEN.md` (it exists; its readings are kept and extended):
- **PRIMARY' (decides, as PRIMARY did):** the number of distinct (recipe, amount, kind, bystanders, second group, family, level), counted over:
  - the 26 development rows of both splits and both passes, with family and level empty. These are C5's 22 distinct tuples;
  - every (row, family, level) with at least one accepted variant in seeds 0–2.
  - Gate 1 is met at ≥ 30, borderline at 25–29.
- **LOWER':** distinct (situation, structure size, bystanders, second-group presence).
  - Bystanders = the base's count plus the added people with no group id (A 1, B 3, D 2 or 5, E 1).
  - Second-group presence is true if the base has one or the variant is C.
- **EMPIRICAL (descriptive):** for each (row, family, level), the added people's displacement between the three seed pairs. The frame is C5's, p' = R(−ψ)(p − a), with rigid removal; people are matched by order, and by the best one-to-one matching with and without the mirror. Given as median and maximum, beside the base members' displacement (about 0.05 m in C5's table).
- **Reading of distinctness, given as a judgement:**
  - Crowd additions are random within bands, so within one configuration the seeds differ in where the added people stand, unlike the base rows, whose members barely move.
  - But every configuration sits on the same 10 base structures (5 line lengths, 3 group sizes, 2 audiences) and adds no situation. So it widens diversity within a situation and does nothing for C5's leave-one-situation-out folds.
  - A's levels and E's depths are near-duplicates of each other for training purposes.
  - These configurations come from development seeds 0–2. Training data for C5 would need fresh seeds that belong to other steps.

## 11. Order, outputs and compute

**Order.**
1. This note and its hash.
2. `builder/`, its self-tests and the base reproduction checks (`builder/base_check.txt`).
3. Generation (`manifest.jsonl`, `scenes/`, `logs/`).
4. Label checks: L1 invariance, L2 cells and L2-empty, full oracle at 100% (`labels.txt`).
5. Scoring: arms 1–3, the equivalence check, the detectors (`score_rows.jsonl`, `tables_oracle.txt`).
6. Noise fit on development runs and validation (`noise/fit.json`, `noise/validation.txt`).
7. Noisy curves (`noise/rows_*.jsonl`, `tables_noise.txt`).
8. C5 recount (`c5_recount.txt`).
9. `README.md` with the verdict.

**Compute.**
- At most 2 worker processes of mine, `nice -n 10`, 2 threads each, torch-free.
- `free -m` before each heavy job. A job starts only with ≥ 3,000 MiB available; otherwise it polls every 5 minutes and logs the wait.
- Estimates (guess): generation 270 variants × up to 6 attempts, about 3–6 s an attempt, under 1 h on 2 workers; arms 1–3 minutes; noise about 33,000 solves, about 1–2 h on 2 workers.
- Scenes take about 0.5 GB under `scenes/`, which the research folder's `.gitignore` keeps out of git.
## 12. Disclosures

- **Designed to trigger known failures.** I know the documented failures (BRIEF 3.5, constraint_method.md), and A–E are built around them: A around (i); B and D around (ii) and (iv); E around hidden floor. The firing rates measure known bugs in generated crowds; they do not discover new ones. A is expected to fail at nearly every level under L1.
- **Labels are the project's own rules.** L1 is the members-only reference rule; L2 is this probe's stricter variant. Neither is social ground truth, and L2 mirrors the solver's own-space rule. The reference constraints are circular (they read the situation and the true group).
- **What I read before writing:** the generator rows, the recipes, the region rules, the solver's line, watcher and own-space code, the scorer, I01's repairs and I03's scorer, plus the C5 gate-1 table (22 configurations; it fires C2's Source A kill).
  - The families' distances come from the package's constants (0.6, 1.2, 1.41, 2.0 m) and the REPORT's 1–2.5 m. A's gap range is data-informed by one base layout's geometry (section 0).
  - No variant was built and no number was scored before this note.
- **One code path differs from the package's.** Arm 3 re-implements the constraint inferrer's last step, so that it can pass reference constraints; the equivalence check guards it. The builder re-implements `_in_clear_view` without its 22-deg cone; the self-test guards it.
- **Not generated or opened:** seeds 3–219; any held-out row, scene or run (group_photo, lift_wait, eval_16–19, s18, s20_held, the replay and pre-flight folders); eval_20 and eval_21 as base layouts.

## 13. Points for the project (the defaults above stand until a dated addendum changes them, before the affected data are scored)
2. **Noise validation target.** The REPORT's kill compares the simulated C with the recorded C. The recorded C also holds the models' own constraint errors (I01: R5 − C = +7.9 and +9.5 points for GLM and DeepSeek). Keep the REPORT's line (my default), or validate against R5 (people noise only)?
4. **Room unchanged.** Variants keep the base room and furnishings, so the drop is the people's. A later VLM crowd run may want re-furnished scenes instead (the package's `_with_more_people`).
