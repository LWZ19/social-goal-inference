
**Status.** Draft, revised 2026-10-18 after three reviews (`phase3/metrics_v1/CHANGELOG.md`; first draft kept as `METRIC_LIST_v1_DRAFT.first.md`), then fact-checked (`FACTCHECK.md`), then updated with the confidence probe's results (`A2_UPDATE.md`). Not frozen or agreed. Version 0 (`phase1/i14_v0/METRIC_NOTE.md`) stays in force until rows of the table below are agreed (REPORT §7).
- **How to agree.** Rows can be agreed one at a time. An agreed row replaces the matching v0 convention. A row nobody answers takes its default. A number marked *placeholder* is not up for agreement yet.
- **Labels.** "Drafter's arithmetic": worked out here from cited numbers. "Reviewer's simulation": from a review, not re-run here. Paths are relative to `results/2026-10-17-beyond-vlm/`.

---

## 0. Summary

**What v1 changes from v0**
1. **Top-1 success stays the only primary metric.** v1 adds units, clusters, which interval decides, and when a difference is worth acting on (§1).
2. **A distribution study names its question first:** Q1, "how likely is the chosen pose acceptable?", or Q2, "where are all the acceptable poses?" Each has its own scores and bars (§2).
3. **Collapse is judged by the question.** Q1 has no collapse line; Q2's region log-density score penalises collapse itself (§3).
4. **Confidence is judged within each model's cell and against the solver's flags,** so a signal cannot pass by knowing which model ran (§2.1).
5. **Seeds of one generator row count as one template,** not as independent layouts (§1).
6. **Random baselines are drawn over the whole room,** not a 3 m disc (§2.4). New scope: crowd labels (§4), people perception (§5), speed and held-out data (appendices).

**Words used below**
- **Top place, top-1.** The goal pose a method picks (for C, the solver's pick). Top-1 success: that pose is acceptable.
- **R, E, C.** The methods. In R and E the VLM marks the goal itself (E with worked examples). In C the VLM lists people and writes constraints, and a solver picks the pose.
- **Old and new wording.** Two answer schemas. For gateway models the prompt text is the same, and new-wording people answers are cache copies of old ones.
- **Cell.** One model × one set (× method, wording). Primary cells: old-wording realistic (s19 Qwen, s19 GLM, s20 DeepSeek) and watching. s19, s20, s21, s23 are recorded run batches.
- **Layout, template.** A layout is one realistic scene, dressed three ways. A template is one generator row; seeds of a row are near-copies (§1).
- **Situations.** Nobody near, waiting line, conversation group, watching (an audience).
- **V0, V1, V3.** V0: all mass on the top place. V1: I04's soft density over the solver's ranking keys. V3: a Gaussian around the top place, fitted to σ ≈ 0.06 m (a "smoothed point mass").
- **F(rec), band-uniform.** F(rec): the floor cells the solver could choose for a record, built with the record's own VLM people. Band-uniform: uniform over floor cells at the usual distance from the target (gate 2's baseline).
- **Family A.** I06's crowd variants with one confuser behind a line, at six levels of how much it looks like it is waiting.
- **L1, L2.** Crowd labels. L1: only group members shape the acceptable region (today). L2: L1 minus cells within a 0.3 m body gap of non-members and minus other groups' circles.
- **AUROC.** The chance a random failure gets lower confidence than a random success; 0.5 is no signal. **Nats:** natural-log units of a log score.
- **MDD.** Minimum detectable difference: the true difference a design finds 80% of the time at two-sided 0.05. A planning number.
- **SEWA.** Smallest effect worth acting on, fixed before a study.
- **Holm's correction.** Over m models: test the smallest p at 0.05/m, the next at 0.05/(m − 1), and so on.
- **Visibility rule.** The package's `is_visible`: whether a person counts as visible in the image.
- **Confidence probe.** The phase-3 study (`phase3/confidence/README.md`, 18 Oct): confidence from disagreement between the VLM's people and a local detector's (§2.1). Signals: P1, the shift of C's place when the solver re-runs with the detector's people (the detector repair, I09 step A2's setting D); S2, people disagreement with the detector; S4, disagreement among C's trials (two extra runs); S6z, a zero-call combination including P1 and S2. **The flags:** any of the solver's three flags raised.
- **Idea codes** (REPORT §4). I04 soft solver density; I06 crowd variants; I09 local people detector; I14 this scoreboard; I46 statistics and sealed test set; A0, I09's step that measured the VLMs' facing accuracy; C5 flow-matching sampler, whose "gate 2" tested a simple density on unseen situations.

| # | Question | Recommendation; default if no answer | Who | Why (evidence) |
|---|---|---|---|---|
| D16 | External checks | Ask about sharing the 42 group-joining scenarios or the predictor's outputs (I51), and ethics review for a click pilot (I35). *Default:* none | Mentor | Every label is the project's own rule (REPORT §8) |

---

## 1. Primary metric: top-1 success

**Definition** (as v0, `phase1/i14_v0/METRIC_NOTE.md`). One goal pose per episode succeeds when the method picked the right target, the pose is reachable on the ground-truth grid (robot radius 0.30 m, person radius 0.25 m), it lies inside the matched target's acceptable region, and its heading is within 45 deg of the allowed heading (D1, D2). The package scorer decides; a wrong target scores 0.

**Units.** Every table names its unit. Goal-step counts read "of N goal-step records" and are never mixed with records of 252.

| Unit | What it is | Size | Clusters |
|---|---|---|---|
| Episode record | One (arm, scene, episode, trial); the unit for top-1 | 252 per model and method per realistic set (14 layouts × 3 arms × 2 episodes × 3 trials) | 14 layouts of 18 records |
| Goal-step record | An episode record where C reached the solver; the unit for set metrics, Q1, and Q2 on development records | Old wording: Qwen 143, GLM 141, DeepSeek 157 (441); watching 36 per model; 549 in all | 13 realistic layouts (`eval_04` has none; per cell Qwen 8, GLM 11, DeepSeek 12) plus 2 watching templates |
| Generator episode | One (scene, episode) of a generator seed; the unit for Q2 there, as in gate 2 | 520 on seeds 100–109 | Templates |
| Template | One generator row or base structure; seeds of a row repeat it with small jitter (members move a median 0.00–0.48 m; `phase1d/README.md` §2.1) | 22 among the development rows; the 14 realistic layouts are 14 rows | Cluster and leave out by template |

- A **layout** is a scene id without the arm (real, staffed and plain dress one layout). Template counts: watching 2 (rows `eval_14`, `eval_15`, seeds 0–2), I06 10 base structures, I09 14 rows. Watching cells are descriptive only; give the 6-scene version as a sensitivity.
- **Per-cell reporting.** k of n per cell. Pool only independent sets: s21 Qwen's and s23 GLM's and DeepSeek's C people answers are cache copies of s19/s20.

**Intervals and tests.** Records in a layout are correlated: the share of variance between layouts (ICC) is 0.08–0.75, so 252 records carry the information of 18–107 independent ones (`phase1/README.md` §5).
- **Top-1 differences on episode records.** The two-sided exact sign-flip test by layout decides whether there is a difference: how often random sign flips of the 14 layout differences give a total this large. The t interval over the 14 layout mean differences gives the size. No record-level binomial, McNemar or Wald tests.
- **Record-weighted means and ratios on goal-step records** (log score, Brier, mean S, lift, recovery). Layouts are very unequal in size (1–6 records at `eval_00`–`05`), so an unweighted t over layout means estimates a different quantity. Use the record-weighted value with a cluster-robust standard error from cluster totals and t(G − 1) critical values, G the number of clusters. This interval decides.
- **AUROC and selective accuracy.** Layout-cluster bootstrap, 2,000 draws, percentile; it runs narrow with 13 clusters, so say so. As the probe froze (`FROZEN.md` §5): paired draws, midrank AUROC (ties half), expected-value ties in selective accuracy. Fix tie rules in advance: P1's DeepSeek AUROC spans 0.670–0.706 by tie-breaking.
- **Fitted parts** are scored leaving one template out.

**A difference worth acting on** (D3).
- **Before the study,** the note names its design, the matching MDD row and its SEWA. If the MDD exceeds the SEWA, it says the study can detect only effects of at least the MDD.

| Design | MDD, records of 252 (`phase1/i46_stats/mdd_table.txt`) | Noise covered |
|---|---|---|
| Claim about new layouts (offline replay or live) | Over 14 layouts: 18–28 (E−R-like), 33–79 (C−E-like) | Layout to layout |
| Paired live A/B, one stage, same trials | 11–21 per 3-trial batch | Run to run, these layouts |
| Two independent live batches | 24–32; 22 / 18 / 16 with 2 / 3 / 4 batches each | Run to run, these layouts |
| Offline replay, claim about these records only | None: margins stand for those records | — |

- **After the study.** (1) *A difference:* the pre-registered exact test passes at 0.05 or its Holm level. That needs enough non-tied layouts: 6 for one comparison (floor 0.031), 7 for 3 models under Holm (0.016), 8 for 6 models (0.0078). (2) *Worth acting on:* also, the estimate ≥ SEWA. (3) *Strong claim:* also, the interval's lower end ≥ SEWA.

**Multiplicity** (D13). Each study freezes and hashes a note before its test data exist or anything is scored, naming one primary metric, one primary comparison, unit, bar, interval and SEWA. An "m of k" bar ("in 2 of 3 cells") is part of that comparison and is judged on point estimates, as I04 and the confidence probe do. "Every model" claims use Holm with the floors above. All else is secondary; post hoc and data-informed parts are labelled.

---

## 2. Distribution metrics: two questions

**Proper scores.** A score is proper when a forecaster does best on average by stating what it believes. The binary log score and the Brier score (§2.1 (b)) are proper. Q2's region log density is proper for the target "uniform over the region". Q1 (a)'s mean ln(ε + S) is not; it is a decision score.

### 2.1 Q1: "How likely is the chosen pose acceptable?"

**(a) A density over floor cells.** S = Σ p(cell) × P(success at the cell), the expected success of one random draw.
- **Decision score:** mean ln(ε + S), ε = 1e-3, as I04 fitted. A point mass on the best cell maximises it, and ε alone sets its penalty for failure: V0's −1.660 is (132/549) × ln 1e-3 plus a negligible term, and V0 scores −1.10 at ε = 1e-2 and −2.21 at 1e-4 (drafter's arithmetic). So verdicts must hold at all three ε.
- **Also reported:** mean S; the hedging share (top-place failures where S > 0); Lift_room in two forms, I04's ΣS / Σρ_room and v0's Σ top-1 / Σρ_room (ρ_room: the region's share of the floor); hit@k with places at least 0.6 m apart; the set success share (v0's "P(success)@5").

**(b) A probability p̂ stated for the chosen pose,** computed without labels (method code reads only `result.*`; REPORT §8, "Leakage").
- **Skill (judged).** Brier score: mean (p̂ − outcome)². Brier skill score (BSS): 1 − Brier / Brier of a reference that states the base rate of the record's own cell (leave one template out). Computed within each cell; the pooled figure is the record-weighted mean of the cell values. A pooled reference hands skill to any signal that knows the model: per-cell rates alone score +0.08 pooled and +0.32 at old Qwen (+0.35 if the base rate is pooled over all 549 primary records; drafter's arithmetic).
- **Descriptive.** Binary log score (p̂ clipped to [0.001, 0.999]); a reliability table (5 equal-count bins); ECE, the record-weighted gap between mean p̂ and success share; calibration slope and intercept from a logistic refit, with intervals. ECE is biased upward on small clustered samples: a perfectly calibrated forecaster would often show 0.03–0.06 here (reviewer's simulation).

**(c) Any confidence score, used for ranking.** **Stratified AUROC for failure** (judged): failure–success pairs are compared only within a cell, summed over the counting cells (Q1 bars). Never one ranking pooled across cells. Reported beside, each next to the flags': AUROC per cell with its failure count, right-target and wrong-target AUROC. 34 of the 102 failures among the 441 realistic records are wrong targets, which no density over one record's floor can see (`phase3/confidence/FROZEN.md`).

**(d) Abstention,** for a study claiming that use. (c) stays the primary bar for any confidence signal; an abstention claim must also pass (d). Abstain (ask or look again) on the 20% of a cell's records with the lowest confidence. **Selective accuracy** is top-1 over the rest; the gain is that minus the cell's base rate (random abstention's expected value), then the record-weighted mean over counting cells. Report failures caught beside the random 20%, and 50% abstention as a secondary rate. For scale on the 441 realistic records: a perfect ranker gains +6.3, +17.0 and +17.4 points (Qwen, GLM, DeepSeek); a signal with AUROC 0.70 about +1.9, +6.0 and +5.9, a mean of +4.6 (binormal model, equal spreads; reviewer's arithmetic, re-run here). Within cells on the probe's records (all three): P1 +8.5, S2 +8.1, S6z +8.6, the flags +3.0 (+4.6 over GLM and DeepSeek alone; drafter's arithmetic from its `tables.txt` T5).

| Output | (a) Density bar | (b) Skill bar | (c) Ranking bar | (d) Abstention bar |
|---|---|---|---|---|
| Density over floor cells | Judged | — | If it claims a confidence (for example S) | If claimed |
| Stated probability for the chosen pose | — | Judged | Judged | If claimed |
| Score that is not a probability (for example a distance) | — | — | Judged | If claimed |

**Q1 bars** (D7).
- **(a)** Beat V3 on mean ln(ε + S) at every ε in {1e-2, 1e-3, 1e-4}, with the cluster-robust interval of the difference excluding zero at 1e-3; beat V0 the same way at 1e-3. V0 and V3 are recomputed, and V3 refitted, on the study's own records and folds (I04's −1.660 and −1.452 hold only for its 549 records). If the density changes the top pose, its top-1 stays within 2 records of the solver's in every primary cell (I04 (c)'s tolerance, one-sided). This is stricter than I04's (b): it adds V0, an interval and the ε check.
- **(b)** Pooled within-cell BSS above 0, the cluster-robust interval excluding zero, and positive in at least 2 of 3 old-wording realistic cells (point estimates). So a combined signal, such as the probe's S6, cannot pass by learning the model.
- **(c)** A cell counts with at least 20 failures and 20 successes; others are reported, not judged; a claim needs at least 2 (old Qwen's 9 failures gave P1 an interval of 0.392–1.000). *Pass:* stratified AUROC over counting cells ≥ 0.70, bootstrap lower end ≥ 0.60; no counting cell below 0.60 (point); the paired difference from the flags' stratified AUROC has its lower end above 0. *Kill:* stratified AUROC below 0.60, or at or below the flags'. The bar reads all records; a placement-only claim may pre-register the right-target AUROC instead. A signal overlapping a measured repair is also reported where that repair changes nothing. *Planning:* the probe's paired intervals against the flags ran ±0.13–0.18, so 400–440 records need a gap near 0.15–0.2.
- *Set after the probe's results, so valid only on records it did not use:* the counting-cell rule, the per-cell floor, both flag comparisons and the repair-overlap report. *Set before:* 0.70, 0.60, 20% abstention, bars (a) and (b).

| Item | Confidence probe | v1 | Why |
|---|---|---|---|
| Success | AUROC ≥ 0.70 in 2 of 3 cells, and pooled AUROC (one ranking over 441) above the any-flag baseline's | Bar (c): stratified over counting cells, with a paired win over the flags | Qwen's cell has 9 failures; model identity alone clears "pooled above the flags" (0.658 against 0.595); P1 − flags +0.112 [−0.088, +0.273] |
| Kill | Pooled AUROC < 0.60 | Bar (c) | As above |
| Abstention | Coverage 0.8 and 0.5, point values | Bar (d); 50% secondary | One model runs at deployment |

### 2.2 Q2: "Where are all the acceptable poses?"

Q2 asks whether a density covers the whole region and keeps mass elsewhere low. Its ideal is uniform over the region, if every pose there is equally acceptable (D8).

**Floor, grid and unit.** Values on the two floors are never compared.
- **Development records:** F(rec), the solver's selectable cells on its 0.05 m grid, built with the record's own VLM people, as v0 froze (so baselines get no oracle people). A region cell is a cell whose centre lies in an acceptable polygon. A region cell outside F(rec), such as one blocked by a phantom person, counts at the 1% floor level, ln(0.01 / |F(rec)|); it is not dropped. Unit: goal-step record.
- **Generator seeds where the method sees true people:** the ground-truth floor connected to the robot's start, 0.10 m grid, as gate 2. Unit: generator episode; clusters: templates.

**Metrics.**
- **Region log density (primary).** The mean over an episode's region cells of ln q (nats per region cell), then the mean over episodes. q is the method's density over the floor with 1% uniform mixed in. That weight sets the cost of a missed cell (about ln 0.01 ≈ −4.6 nats against uniform), so verdicts must also hold at 0.1% and 10%.
- **Baselines and control,** on the same cells: band-uniform (uniform on floor cells whose target distance lies between the 5th and 95th percentiles of training region cells); uniform over the floor; and a control, uniform over floor cells outside every forbidden kind.
- **Coverage (diagnostic, always beside in-region mass).** The share of region cells where q is strictly above the uniform level 1/|floor cells| (relative tolerance 1e-9). For samplers: 64 samples per record, the share of region cells within 0.2 m of a sample, and the in-region share of samples. Uniform over the floor maximises both, so coverage alone rewards spreading.
- **Mass outside the region.** In-region mass, and mass in each forbidden kind (`ahead_in_line`, `group_space`, `audience_view`) over records or episodes that have that kind. Heading: the share of in-region mass within 45 deg.

**Q2 bars** (D8).
- Δ ≥ 0.10 nats per region cell against the **stronger** of band-uniform and uniform, fold by fold. Band-uniform alone is weak: uniform beats it by 0.131 and 2.919 nats in gate 2's waiting-line and audience folds, so uniform would clear 0.10 against the band in 2 of 3 folds (`phase1d/README.md` Table 2.2).
- In at least 2 of 3 left-out people situations (gate 2's rule), each passing fold judged on its **worst of at least 3 training draws**: in gate 2 the training subsample alone moved a fold's Δ by about 0.9 nats (+1.955 against +2.843 to +2.906; Table 2.3). Report per-situation Δ, the share of episodes with Δ > 0, and a template-cluster interval per fold (3–5 templates, so descriptive).
- The control must fail under the same folds, or the study makes no Q2 claim. (At I04's room shares, dropping the forbidden cells gains at most about 0.096 nats over uniform, so the control fails, narrowly, at audiences; drafter's arithmetic.)
- The verdict holds at mixing weights 0.1%, 1% and 10%.
- Forbidden mass at most 0.02 in each kind, beside uniform's level (the room share) and the solver's top place's share. 0.02 allows smoothing over a group's ring (V1 and V3: 0.001) and the top place's own rate under VLM people (0.0185 in audience view), and catches uniform's 0.0916 in audience view (`phase1d/README.md` Table 1.6). Where the room share is below 0.02 (lines 0.0016, groups 0.0030), only the log-density bar catches spreading.

### 2.3 Earlier results under v1

| Result | What it scored | Question | Verdict under v1 |
|---|---|---|---|
| I14 v0 runners-up (`phase1/README.md` §4) | hit@k, recovery, set success share | A list question, closest to Q1 | **Same:** runner-up knowledge not shown. With whole-room draws, AR recovers 57% and 69%, above the 50% line in both DeepSeek cells. GLM's runners-up fall below random and farthest additions (99 against 104.0 and 107.0) |
| I04 (`phase1d/README.md` §1) | Mean ln(1e-3 + S); concentration AUROC | Q1 (a), (c) | **Same (kill):** V1 −1.563 against V3 −1.452 (n 549). The ε check was not computed; stratified AUROC 0.447 (recomputed by the confidence probe); every cell AUROC (0.559, 0.303, 0.540) is below 0.70 |
| C5 gate 2 (`phase1d/README.md` §2.2) | Region log density against band-uniform | Q2 | **Same on the main clause:** against the stronger baseline Δ is +1.823, −0.781, +1.195 (frozen draw) and 2 of 3 in the verifier's draw too. Not computable from the page: a third draw against uniform, mixing weights, the control, template intervals. Δ was positive in 0 of 100 conversation-group episodes |
| Flow pilot (`phase2/flow_pilot/README.md` §5) | Top-1 (K1 against P, the solver with true people; K2 against C); Lift_room and P(success) against V3 (K3) | Top-1, then Q1 | **Same (kill):** K1 (22–30 points below P) and K2 (15–27 below C) fire on top-1 alone. K3 compares with V3, a near-point mass |
| I06 L1/L2 (`phase2/i06/README.md` §2) | Disagreement on oracle C | Label validity (§4) | **Same:** 1.5% passes. The reference stop's 42.7% reads as label dependence for reference-point methods |
| I09 confirmation (`phase2/i09/README.md` §1) | Detector bars | People perception (§5) | **Same (neither):** 0.242 false people per image, or 0.192 / 0.170 under D12, against 0.10 |

### 2.4 Chance region (D9)

- **(a) Lift denominator.** v1 keeps the floor F(rec), as v0 froze. On shared records ρ_room does not depend on the method, so a lift ratio equals a success ratio: V3/V0's Lift_room of 8.78/8.90 matches their mean S of 0.749/0.760 (`phase1d/README.md` Table 1.2). Lift only restates success against chance. The region is 1.6–3.8% of reachable floor (REPORT I14), so lifts are large by nature; compare them across methods, not with a fixed number.
- **(b) Random baselines.** v0 drew R5 (5 random cells) and AR (the top place plus 4 random cells) in a 3 m disc around the target. The disc leaves out about 90% of the watching region, and the top place lies outside it in 470 of 790 line records (`phase1/README.md` §4). v1 draws over F(rec), the disc beside. AR's recovery moves from 96% to 57% (DeepSeek realistic) and from 33% to 69% (watching).
- **Why not a region around the acceptable region or the top place** (REPORT §7's option). Around the region uses the label, so it is not chance. Around the top place depends on the method's own pick, which the anchored baselines already cover (AR, and AF: the top place plus 4 farthest-apart cells), and it needs a radius nobody has chosen. The whole floor adds no parameter.

---

## 3. Collapse

**What happened.** I04's kill 2: "more than 90% of the mass within 0.3 m of the top place in over 80% of records" (REPORT I04). V1 crossed 0.9 in 312 of 549 records (56.8%), so it did not fire. But V3 kept more than 0.999 of its mass within 0.3 m in every record, so the line would fire on the study's best density (`phase1d/README.md` Tables 1.1, 1.6). The top place succeeds in 417 of 549 goal-step records, so under a Q1 score moving mass off it costs more than it gains.

**v1 rule** (D6).
- **Q1: concentration is not a failure.** When the solver is usually right, nearly all mass on its top place is the right answer. No collapse line; beat V0 and V3. Phase 1d's reading, untested: a useful spread would need upstream uncertainty, such as perception, which the solver's keys cannot see (`phase1d/README.md` §5).
- **Q2: concentration is a failure,** caught by the region log-density score itself, since a near-point mass leaves most region cells at the 1% floor. Coverage, read with in-region mass, shows it too.
- **Diagnostics, never judged:** mass within 0.3 m of the top place; effective support (exp of the density's entropy, in m²) beside the region's area.
- A method claiming both questions passes each question's bars separately.

---

## 4. Crowd labels

L1 and L2 as in the glossary; L3 (L2 with a fallback) was not measured (REPORT §7).

**What I06 found** (`phase2/i06/README.md` §2). One scene is one episode, so n is scenes. The 199 variants come from 10 base structures × 3 seeds: 10 templates.
- Oracle C (true people, reference constraints, then the solver) disagrees between labels in 3 of 199 (1.5%), all at one row (`eval_11`, one per seed); L2 is never empty. This is built in: the solver keeps 1.0 m from people and L2 needs 0.85 m. With simulated people, its disagreement is 2.1–4.5% (DeepSeek 4.0% in the verifier's run; §11 there), close to the 5% line.
- L1's own reference stop fails L2 in 85 of 199 (42.7%), including all 72 variants of family A levels 2–6.

**v1 rule** (D10, D11).
- **For a method:** report its L1 and L2 success and its disagreement rate, with a template-cluster interval. Above 5%, its claims are stated under both labels; a claim that holds under one label only is "label-dependent". These lines never kill a method.
- **Affected methods:** those aiming near the reference point, that is R's and E's marked points and labels sampled for learned samplers such as C5 (REPORT §7). The choice also moves Q2's region cells in crowds. It does not touch the perception metrics.

---

## 5. People-perception metrics

For detector studies (I09) and VLM people stages, with the project's people scorer (from I03) unchanged (`phase2/i09/README.md` §1). **Detector bars:** I09 step A1's. The unseen-layout confirmation used all but the last two rows, since no VLM ran on its scenes and runtime was not judged.

| Metric | Definition | Bar |
|---|---|---|
| Recall | Matched / visible; greedy nearest-first one-to-one matching within 0.5 m; visible by the visibility rule | ≥ 0.95; kill < 0.90 |
| Line-member recall | Recall over waiting-line members | ≥ 0.95 |
| False people per image | Unmatched detections. Exempt (counted apart): within 0.5 m of the anchor of a target with no person. Frozen: near a person failing the visibility rule, false and flagged. Proposed (D12): within 0.5 m of such a person, counted apart | ≤ 0.10; kill > 0.30 |
| Position error | Median over matched pairs, m | ≤ 0.15 m |
| Facing by view class | Angle between true yaw and bearing to camera: frontal < 45, oblique 45 to < 75, profile 75–105, back > 105 deg; unknown yaw is wrong, 180 deg in medians | Non-profile median ≤ 25 deg; non-profile within 45 ≥ 0.85; profile front/back right ≥ 0.90; kill: all matched within 45 < 0.70 |
| Facing on A0's pairs (when scored on them) | Within 45 on A0's (arm, scene, person) pairs | Kill below 0.806 (Gemini's 0.906 minus 10 points), or below 0.70 with the school panel only |
| Runtime | Warm deployed path on the 4060: median detector-plus-pose time plus median project-side time | ≤ 1.0 s |

- **Intervals.** Cluster by row (14 templates): a cluster-robust interval with t(13) for recall and for false people per image (a rate that can exceed 1, not a proportion); a row-cluster bootstrap as a check. Wilson intervals ignore clustering and are descriptive only.
- Report recall by role class (staff was weakest: 69 of 78 on unseen layouts; `phase2/i09/README.md` Table 1.2).
- **The hidden-by-rule question** (D12). On seeds 200–212, 44 false people gave 0.242 per image. 9 lie within 0.5 m of counter staff the visibility rule calls hidden, though the staff's marked point projects into the picture; 4 more lie 0.52–0.61 m away (post hoc, `phase2/i09/README.md` Table 1.5). Without the 9 the rate is 35/182 = 0.192; without all 13, 31/182 = 0.170 (drafter's arithmetic). Both stay above 0.10, so I09's "neither" stands. "Apart" fits better than hit or false: the rule and the image disagree, so there is no visible person to match, yet a real person is there; it is a scorer question (I09 §3, §4 question 4). The radius is fixed at 0.5 m now, because the 0.61 m edge came from the post hoc table.

---

## Appendix A. Speed metrics (D15)

Label every goal-step latency with one quantity (REPORT I21; Qwen C, 143 realistic goal-step records):

| Quantity | Includes | Example |
|---|---|---|
| Record time | The record's `latency_s`: calls, waits in the rate limiter (pacer), wall time outside calls | 61.6 s |
| Provider time | Serial sum of per-call provider times | 61.2 s |
| Critical path | Longest chain of dependent calls, independent calls run together (replay) | 49.3 s |
| As the benchmark runs | People answers cached per scene: serial, then critical path | 53.2 s, 47.4 s |

- Median and p90 with n. Fresh calls and cache hits apart. State the load (stage 23 ran 6 gateway jobs under the 85/min cap; REPORT §1.1, finding 8).
- C and E are compared only under the same per-stage thinking policy (REPORT I22), paired on matched records in the same load window, with a scene-cluster bootstrap interval for the median difference. "C faster than E" is claimed only when that interval excludes zero.
- **Local models:** name the timed span and the load. I09: 55.6 ms on the warm deployed path (step A1's development images; verifier 61.9 ms); 215.1 ms un-warmed, every box ≥ 0.10 posed, on a shared machine (verifier 127.0 ms on the same images; `phase2/i09/README.md` §1). The flow pilot's 18.3 ms covered sampling and projection only (verifier 11.9 ms, idle GPU), with the raster (78.5 ms) and floor work (110.8 ms) timed apart (`phase2/flow_pilot/README.md` Table 2.10). Neither pair is settled.

---

## Appendix B. Held-out data and seeds

| Data | Fit or tune? | Test? | Source |
|---|---|---|---|
| Development: whitelisted runs and the 68 development scenes (42 realistic, 26 placeholder) | Yes, leaving one template out | Yes, but a method chosen on it needs fresh-template confirmation (D14) | REPORT §8, "Tuning on the reporting sets" |
| Fit seeds 100–109 | Yes (I03, I11, I09 step A1, gate 2) | Not for methods fitted on them | REPORT I46; gate 2: `phase1d/README.md` §2.2 |
| Training seeds 110–199 | Training only (110–189 trained the pilot, 190–199 chose its checkpoint) | No | `phase2/flow_pilot/README.md` §1, §7 |
| I46's sealed bank: seeds 3–99 (never generated or opened) and new situations written blind | No | I46's one final run, about 6,000 calls | REPORT I46, §6 |
| `eval_16`–`19` (group photo, lift wait), runs s18 and s20_held | No; never read | Spent for clean claims; a new method needs its own held-out design | REPORT §8, "Held-out discipline" |

- The REPORT does not say whether the `keep/heldout` test and I46's final run are one run or two; the call estimate is I46's.
- Noise models are fitted on development runs only (REPORT §8 (d)). Other fitted parts use development data or the fit and training seed ranges above.
