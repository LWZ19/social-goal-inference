# C5 (I52) zero-call gates 1 and 2: frozen protocol

The C5 gates lead wrote this on 18 Oct 2026. It was written before any configuration was counted, any density fitted or any fold scored. Its sha256 is in `FROZEN.sha256`. This file is never edited. Any change goes into a dated addendum with its own hash, written before the data it affects is scored. Paths are relative to `results/still-scenes/learned-models/flow-model-gates/`, unless they start with `phase1/`, `phase1b/` or `src/`.

**Sources.** REPORT.md C5 [BOLD] I52 (gates 1–3), C2 [BOLD] I27 (Step 0 and kill), I04, I14, section 4.6 and section 6 steps 7–8; `results/still-scenes/metrics-and-statistics/metric-v0/METRIC_NOTE.md` and `i14.py`. This note borrows from them the vectorised point-in-polygon replica and the rule that a cell stands for its centre.

- Gate 1's sources and its three readings, and its 30 / 25–29 line.
- Gate 2's data (fresh seeds 100–109), cells, the feature rule, the ratio KDE, the band-uniform baseline, the score, the folds and the pass rule (0.1 nats, C2's kill line).

**What I ran before hashing** (all in `precheck/`, logs beside the scripts):
- `format_check.py` loaded 6 scenes of seed 100 (one each of no_one place, object and person, waiting_line_4, conversation_group_4 and audience_5). It listed their targets, people, episodes and region polygons, and their 0.10 m floor and region cell counts.
- `bench_synthetic.py` timed the kernel sum on random numbers only: about 6.5 ns per (evaluation, training) pair, with 2 threads.
- A one-liner read `results/still-scenes/perception/fresh_seeds/manifest.json`. Every seed has 26 scenes: 11 no_one, 7 waiting_line, 5 conversation_group and 3 audience. There are no rejections, and the 4 held-out rows were skipped.

**What the format check showed:**
- Each scene has one target `t0` and two episodes, e0 and e1, with the same region cell set.
- Each scene has 4,944–15,048 floor cells and 83–690 region cells.
- The first edge of the bounds polygon has the same yaw as a place's front.

These facts shaped the episode weighting, the subsample sizes and gate 1's frame. I counted no configuration and fitted nothing.

## 1. Gate 1: does the generator hold at least about 30 distinct configurations? (C2 Step 0 and its Source A kill)

`gate1.py` takes the rows from the code: `_EVAL`, `_EXAMPLES`, `_EVAL_MORE` and `_EXAMPLES_MORE` of `src/sgi/synth/generate.py`, kept only when `row[0]` is in `cfg.dev_situations`. Rows of group_photo and lift_wait are never instantiated. No scene is generated.

**PRIMARY (generous).** The number of distinct tuples (recipe, amount, kind, bystanders, second group), by exact equality, across both splits and both passes.

- Structure size is the row's amount for waiting_line, conversation_group and audience (the target person counts as a member).
- It is 0 for no_one, whose lone target person is the target, not a structure.
- The robot's viewpoint and the target's kind are ignored.

**EMPIRICAL (descriptive).** For each of the 26 rows, matched across seeds 100–109 by (split, index), with each scene loaded after `H.guard_scene`:
- **Displacement.** Over the 45 seed pairs, the distance between each person's two positions, reported as the median and maximum per row (m).
  - (a) **Primary:** people matched by id (the generator's order), with rigid removal only.
  - (b) **Secondary:** the smallest over one-to-one matchings, with and without the mirror y' -> -y'. Audience rows mirror their people by `side`, which is not rigid.
  - Each reading is given for structure members alone and for everyone. Group ids are used here for tabulation only. The robot's start position is reported the same way.
- **Random parameters.** A list, from the code, of the parameters drawn at random within a row, with their ranges.

**Rules.**
- Gate 1 is met when PRIMARY ≥ 30. A count of 25–29 is reported as borderline, not met.
- All three readings are reported. Only PRIMARY decides.

## 2. Gate 2: does a non-neural conditional density transfer across situations?

**Data.** The 260 fresh-seed scenes, loaded as `results/still-scenes/perception/depth-blob-detector/stageA/fitdata.py` does: `H.guard_scene`, then `sgi.scene.load_scene`. Their regions come from `sgi.gt.derive.prepare` under `configs/default.yaml`.
- **Unit:** one (scene, episode) with exactly one target and at least one acceptable region.
- **Excluded and counted:** episodes with several targets, episodes with no region cell, and any episode of a held-out situation (asserted absent).
- **Secondary test set, never fitted:** the 68 scenes of `H.distinct_scenes()`, each loaded with `H.scene(first listed run, scene_id)`. A scene with no listed run is skipped and counted. Domain comes from the folder: `realistic_final` is realistic (42), anything else placeholder (26).

- The 0.10 m grid's origin is the scorer grid's origin minus 0.025 m in x and y, so every 0.10 m cell centre is the centre of a 0.05 m scorer cell.
- A cell is **floor** when that scorer cell is free in `sgi.freespace.robot_grid(truth, robot, th)` and joined to the robot's start by `freespace.connected`. This is the scorer's own reachability path (`evaluate.score`). It uses ground-truth people and the **default** thresholds for every scene, including development scenes. Runs whose recorded grid, robot or person radius differ are counted.
- A **region cell** is a floor cell whose centre lies in the union of the episode's acceptable-region polygons. The test is `sgi.polygons.point_in_polygon`, through I14's vectorised `pip` replica, checked against the package on every region cell plus 300 random floor cells per scene. Every disagreement is reported.
- **Heading is not part of gate 2.** Position only.

- f1. d_t = |c - a|.
- f2, f3. cos and sin of wrap(bearing(a -> c) - front). The front is `front_yaw`, else the target person's yaw (person targets), else none, giving (0, 0).
- f4. d_p = min(distance to the nearest person, 5 m). It is 5 m when there are no people.
- f5. The cosine between that person's facing and the unit vector from them to c. It is 0 when f4 = 5 m.
- f6. The number of people within 1.5 m of c.
- f7. The distance from c to the robot's start.
- f8. The number of people, the target person excluded, whose centre lies within 0.25 m (person_radius_m) of the segment [c, a].
- f9. d_t minus the largest |p - a| among people within 2.0 m of a. With nobody there, the max is 0 and f9 = d_t.

**Feature sets:**
- **FS1** = f1–f6;
- **FS2** = FS1 + f7 (robot);
- **FS3** = FS1 + f8, f9 (people at the target).

- **Weighting.** Every training episode weighs the same. An episode is one (scene, region set), and identical sets in one scene are pooled by summing their weights.
- **Subsamples** (fixed seed, `default_rng([20261018, crc32(seed/split/scene_id)])`, drawn once per scene): at most 100 of its region cells and at most 200 of its non-region floor cells. Training scenes whose episodes have no region cell are dropped.
- **Standardisation.** Features are standardised by the floor KDE's weighted mean and SD (an SD of 0 becomes 1).
- **Kernel.** A Gaussian product kernel with one bandwidth h = m · n_r^(-1/(d+4)) in every standardised dimension. n_r is the number of region sample points and d the set's dimension.
- **Numerator.** p_reg is the region-sample KDE.
- **Denominator.** p_floor = Σ_e w_e [π_e · KDE(region sample_e) + (1 - π_e) · KDE(non-region sample_e)], where π_e is the episode's region share of floor. This is a stratified estimate of the floor-cell KDE. It keeps r = p_reg / p_floor at or below 1/min π_e.
- **Distribution.** q(c) = 0.99 · r(φ(c)) / Σ_floor r + 0.01 / N_floor, over the test scene's floor cells. Everything is computed in log space (logsumexp), in float64, with numpy only (scipy is absent).
- **Plain KDE (secondary).** p_reg replaces r.
- **1-D ratio KDE (secondary baseline).** The same construction on f1 alone.

**Band-uniform baseline** (from C2).
- B = the test scene's floor cells with d_t in [q05, q95] of every training region cell's d_t. Pooled over cells, numpy linear quantiles.
- q = 0.99 · 1[B] / |B| + 0.01 / N_floor. If B is empty, q is uniform.

**Score.**
- Per test episode, S = the mean over its region cells of log q(c), in nats per region cell: the negated cross-entropy of the uniform distribution on the region against q.
- Also the region mass M = Σ_region q, and lift = M / (n_region / N_floor).
- Fold value = the mean over the fold's test episodes, with n episodes and n scenes. Lift and mass are means of per-episode values.
- Δ = S_KDE - S_band.

**Selection, inside each fold's training data only.**
- **Grid:** {FS1, FS2, FS3} × m ∈ {0.5, 1, 2, 4}.
- **Inner CV:** leave one seed out over the fold's training scenes. The score is S averaged over the held-out seed's episodes, all training situations pooled, equal weight per episode.
- **Pick:** the highest mean inner score. Ties go to the earlier feature set, then the larger m. An optimum at a grid edge is reported, never extended.
- The plain KDE and the 1-D KDE (m only) are each selected by their own inner CV on the same grid.
- The outer evaluation is exact, over every floor and region cell.

**Folds.**
- **Primary: leave one situation out** over waiting_line, conversation_group and audience. Train on all 10 seeds' no_one scenes plus the other two situations, and test on all 10 seeds' scenes of the left-out one.
- **In-distribution ceiling (secondary):** leave one seed out over all four situations. The inner CV runs over the 9 training seeds. Results are reported per situation, together with the transfer gap (ceiling minus primary fold).
- **Development transfer (secondary):** train on all 260 scenes, with the configuration chosen by inner leave-one-seed-out over all 10 seeds. Test on the 68 scenes, by situation, with realistic and placeholder reported apart.

**Gate 2 is met** when Δ ≥ 0.10 nats per region cell in at least 2 of the 3 primary folds, with the KDE being the ratio KDE. The plain KDE and the 1-D baseline are reported beside it and decide nothing.

## 3. Order, outputs and compute

**Order.**
1. This note and its hash.
2. `gate1.py`, which writes `gate1.txt`.
3. `cells.py`, which caches the floor cells, f1–f9 and the region masks per scene and episode (`cache/`) and runs the checks above.
4. `gate2.py`: inner CV and the outer evaluation, first the primary folds, then the ceiling, then the development scenes. It writes `tables.txt` and `inner_cv.txt`.
5. README.

**Compute.** At most 2 processes, `nice -n 10`, 2 threads each. Estimated at about 5 CPU-hours, most of it the ceiling's inner CV. A longer run changes nothing in this protocol.

- **What the reference regions are.** They are the project's own rules (`src/sgi/gt/regions.py`). Every gate-2 number measures agreement with those rules, not social correctness. Fresh seeds come from the same generator, and the 68 development scenes were authored alongside the rules.
- **My exposure.** I read the generator's rows and recipes and the rule code before designing the features, so every feature is informed by them. Specifically:
  - The three people rules share the 0.5–1.2 m join distance from the nearest member, which f4 measures.
  - Each rule's region lies behind or beside the people's facing (behind the last member, outside the circle, outside the screen–audience space), which f5 and f8 measure.
  - Some transfer will therefore come from constants and geometry the rules share by construction, not from a learned regularity.
  - f7 may carry the generator's situation-specific camera placement, so FS2 is a deliberate test of that artefact.
  - No feature branches on a situation, and each means the same in every situation.
- **Prior use of the data.** Seeds 100–109 were I03's, I11's and I09's fit data, for people detection, not regions. No density or region statistic has been computed on them before this protocol.
- **The precheck.** The format check informed the episode weighting, the subsample sizes and gate 1's frame. Nothing else here is data-informed.
- **Labelling of results.** Anything added after results is labelled post hoc. Every number is given with its n.
