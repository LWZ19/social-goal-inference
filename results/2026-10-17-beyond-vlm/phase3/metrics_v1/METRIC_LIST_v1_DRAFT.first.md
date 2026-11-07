
---

## 0. Summary

**What v1 changes from v0**
1. **Top-1 success stays the only primary metric.** v1 writes down its units, cell reporting, intervals and the size of difference worth acting on (§1).
2. **Distribution metrics answer one of two questions.** Q1: "how likely is the chosen pose acceptable?" Q2: "where are all the acceptable poses?" Each has its own metrics and bars, and each study names its question before it runs (§2).
3. **Collapse is judged by the question.** I04's collapse kill line would have fired on its own best control. v1 drops it from Q1 and lets Q2's coverage score catch collapse (§3).
4. **Chance region.** The whole-room floor replaces v0's 3 m disc. Q2 adds a band-uniform baseline (§2.4).
5. **New scope:** crowd labels (§4), people perception (§5), speed (§6), held-out data and multiplicity (§7).

| # | Question | Recommended answer | Why |
|---|---|---|---|
| D1 | Keep top-1 success at 45 deg as the one primary metric? | Yes, with 30 deg (the mentor's value) as a sensitivity | All earlier results use it. At 30 deg DeepSeek's realistic top-1 falls from 109 to 101 (`phase1/README.md` §4) |
| D2 | Must each distribution study name Q1 or Q2, with that question's bars? | Yes | I04's Q1 score rewarded a near-point mass; gate 2's Q2 score rewards spread (`phase1d/README.md` §1, §2.2) |
| D3 | Drop the collapse kill line from Q1 studies? | Yes; require beating a point mass and a smoothed point mass instead | I04's best control kept more than 0.999 of its mass within 0.3 m of the top place in every record (`phase1d/README.md` §1, Table 1.6) |
| D4 | Which crowd label is primary? | L1 until the mentor rules; L2 always beside it | L2 mirrors the solver's own spacing rule, so as primary it would build in a solver advantage (REPORT §8, "Circularity") |
| D5 | Whole-room floor instead of the 3 m disc as the chance region? | Yes; the disc stays secondary | The disc leaves out about 90% of the watching region; the top place lies outside it in 470 of 790 line records (`phase1/README.md` §4) |
| D6 | How to score a detection of a person the visibility rule calls hidden? | Count it apart, as neither hit nor false; keep the frozen count beside it; apply to development numbers too | 13 of I09's 44 unseen-layout false people are of this kind (`phase2/i09/README.md` §1, Table 1.5) |
| D7 | How many primary comparisons per study? | One, pre-registered; Holm's correction for "every model" claims | 14 layouts give little power (`phase1/i46_stats/HOW_TO_REPORT.md`, "Test") |
| D8 | Bar for a confidence signal? | AUROC for failure ≥ 0.70 in 2 of 3 gateway models' realistic cells; selective-accuracy bar after A2 | 0.70 is I04's criterion (d) and the parked I15's revival line (REPORT I04; §4.5) |

---

## 1. Primary metric: top-1 success

**Units.** Every table names its unit.

| Unit | What it is | Size |
|---|---|---|
| Episode record | One (arm, scene, episode, trial); the unit counted for top-1 | 252 per model and method per realistic set (14 layouts × 3 arms × 2 episodes × 3 trials) |
| Goal-step record | An episode record where C reached the solver; only these carry runners-up; used for set and distribution metrics | Old wording: Qwen 143, GLM 141, DeepSeek 157; watching 36 per model; 549 in all |
| Layout | A scene id without the arm; the unit for intervals and tests | 14 per realistic set (6 nobody near, 5 lines, 3 groups) |

Sources: `phase1/i46_stats/HOW_TO_REPORT.md`, "Unit"; `phase1/i14_v0/METRIC_NOTE.md`, "Records".

**Per-cell reporting.** A cell is model × set × method (and wording). Give k of n per cell. Pool only independent sets: the C people answers of s21 Qwen and s23 GLM and DeepSeek are cache copies of s19/s20 (`phase1/README.md` §0). Per-situation numbers are descriptive only: lines cannot reach p ≤ 0.05 (floor 0.0625), groups neither (0.25) (`phase1/i46_stats/HOW_TO_REPORT.md`).

**Intervals and tests.** Records in a layout are correlated. The ICC (share of variance between layouts) is 0.08–0.75, so 252 records carry the information of 18–107 independent ones (`phase1/README.md` §5, Table 5.1).
- Success differences: t-intervals over the 14 layout mean differences, and a two-sided exact sign-flip test by layout (it asks how often random sign flips of the layout differences give a total this large). No record-level binomial, McNemar or Wald tests (`phase1/i46_stats/HOW_TO_REPORT.md`).
- Ratios of sums (lift, recovery) and per-record scores (log score, Brier): a layout-cluster bootstrap, 2,000 draws, stratified by realistic and watching layouts, as I04 did (`phase1d/README.md` §1, "Metrics"). It runs a little narrow with 14 clusters and is crude with 6 (REPORT §8), so per-record score differences also get the t-interval over layout means.
- Anything fitted on development records is scored leaving one layout out, as in I04's 19 folds.

**A difference worth acting on** (proposal). All three must hold:
1. It exceeds the minimum detectable difference (MDD: the smallest difference found 80% of the time at two-sided 0.05) for the design. In records of 252 (`phase1/README.md` §5, Table 5.4): two independent live 3-trial batches 24–32; live, paired within one stage 11–21; over 14 layouts 18–28 (E−R-like) or 33–79 (C−E-like).
2. The layout t-interval excludes zero.
3. At least 6 layouts move; fewer cannot reach p ≤ 0.05.

---

## 2. Distribution metrics: two questions

### 2.1 Q1: "How likely is the chosen pose acceptable?"

**(a) A pose drawn from the method's distribution.** Score its expected success S = Σ p(cell) × P(success at the cell): mean S and mean ln(1e-3 + S), as I04 did (`phase1d/README.md` §1, "Fit"). This score is best when all mass sits on the pose most likely to succeed. v0's "P(success)@5" (uniform mass over 5 places) becomes the "set success share", so "P(success)" means only a probability.

**(b) A probability p̂ the method states for its chosen pose,** computed without labels (method code reads only `result.*`; REPORT §8, "Leakage").
- *Calibration.* Binary log score: mean ln(probability given to what happened), p̂ clipped to [0.001, 0.999]; higher is better. Brier score: mean (p̂ − outcome)²; lower is better. Brier skill score: 1 − Brier / Brier of a base-rate forecaster that always says the training success rate (leave one layout out); at 0.760 that forecaster scores 0.182 (drafter's arithmetic, from 417 of 549; `phase1d/README.md` Table 1.2). Reliability: 5 equal-count bins by p̂, mean p̂ against success share in each; ECE (expected calibration error) is the record-weighted mean gap.
- *Ranking.* AUROC for failure: the chance a random failure gets lower confidence than a random success (ties half); 0.5 is no signal, below 0.5 points the wrong way (GLM's 0.303 in I04; `phase1d/README.md` Table 1.5). Selective accuracy: abstain (ask or look again) on the share a of records with the lowest confidence and score top-1 on the rest, at a = 0.10, 0.20, 0.30. For scale, on the solver's 549 records random abstention stays at 0.760 and a perfect ranker at a = 0.20 reaches 0.950 (drafter's arithmetic: 417 successes in 439 kept).

**Q1 bars** (proposal; inherited lines named):
- (a) Beat the point mass on the top place (V0, pooled mean log −1.660) and the Gaussian control (V3, −1.452) on pooled mean ln(1e-3 + S), layout interval excluding zero, with top-1 not lower: I04's criterion (b) and kill 1 (`phase1d/README.md` Table 1.2, §3).
- (b) Calibration: Brier skill score above 0, interval excluding zero, pooled and in 2 of 3 gateway models; ECE ≤ 0.05 pooled (placeholder).
- (b) Ranking: AUROC for failure ≥ 0.70 in 2 of 3 old-wording realistic cells (I04's (d)), and above the solver's own flags in the same cell (0.435–0.659; `phase1d/README.md` Table 1.5).
- Selective accuracy: **placeholder.** A2's confidence probe (`phase3/confidence/README.md`) had not reported when this was written; its folder held only a pre-freeze check. A2's result should set this bar. Until then: at least +5 points over no abstention at a = 0.20, pooled, interval excluding zero.

### 2.2 Q2: "Where are all the acceptable poses?"

Q2 asks whether a density covers the whole region and keeps mass elsewhere low. Its ideal is uniform over the region, if every pose there is equally acceptable (a question for the mentor, §8).

- **Region log density (primary).** The mean over an episode's region cells of ln q (nats per region cell), then the mean over episodes. q is the method's density over floor cells with 1% uniform mixed in, so no cell is zero. Report Δ against a baseline on the same cells, never raw values across grids. Baselines: band-uniform (uniform on floor cells whose target distance lies between the 5th and 95th percentiles of training region cells) and uniform over the floor. This is gate 2's score (`phase1d/README.md` §2.2).
- **Coverage.** Share of region cells where q is at least the uniform level (proposal). For samplers, also the share of region cells within 0.2 m of a sample, as the flow pilot measured (`phase2/flow_pilot/README.md` §1).
- **Mass outside the region.** In-region mass and mass in each forbidden kind (`phase1d/README.md` Table 1.6).
- **Secondary.** Lift_room; hit@k with places at least 0.6 m apart (I04's rule). Position only, as gate 2; heading reported as the share of in-region mass within 45 deg.

**Q2 bars** (proposal):
- Δ against band-uniform ≥ 0.10 nats per region cell (gate 2's line) *and* Δ against uniform above 0. The new clause: in gate 2's audience fold the band covered 0.289 of the region and scored 2.919 nats below uniform, which inflated Δ (`phase1d/README.md` §2.2, post hoc).
- For transfer, in at least 2 of 3 left-out people situations (gate 2's rule).
- Coverage at least the band-uniform's; forbidden mass no higher than the top places' share.

### 2.3 Which question earlier results answered

| Result | What it scored | Question | Reading under v1 |
|---|---|---|---|
| I14 v0 runners-up (`phase1/README.md` §4) | hit@k, recovery, set success share | A list question, closest to Q1 | Top place plus 4 random disc cells recovered 96% of DeepSeek's +14, so runner-up knowledge was not shown |
| I04 (`phase1d/README.md` §1) | Mean ln(1e-3 + S); concentration AUROC | Q1 (a); Q1 (b) ranking | Right test for Q1. A near-point mass won (V1 −1.563, V3 −1.452, n 549); AUROC 0.559, 0.303, 0.540 |
| C5 gate 2 (`phase1d/README.md` §2.2) | Region log density against band-uniform | Q2 | Met in 2 of 3 folds (+1.955, −0.781, +4.113) |
| Flow pilot (`phase2/flow_pilot/README.md` §5) | Top-1 (K1, K2); Lift_room and P(success) against V3 (K3); coverage on validation seeds, not judged | Top-1, then Q1 | A sampler judged on Q1 against a point mass. The verdict stands: K1 and K2 fire on top-1 alone (15–27 points below C) |

### 2.4 Chance region

v0's 3 m disc fails away from groups; at watching the disc-to-room chance ratio is 0.27–0.31 (`phase1/README.md` §4, Table 4.2). v1 uses the floor F(rec) as the main lift denominator, which v0 already froze (`phase1/i14_v0/METRIC_NOTE.md`, "Floor"). The region is 1.6–3.8% of reachable floor (REPORT I14, "Why"), so lifts are large by nature. Compare them across methods, not with a fixed number: I04's "lift ≥ 10×" was predicted to fail for every variant (`phase1d/README.md` §1, "Prediction").

---

## 3. Collapse

**What happened.** I04's kill 2: "more than 90% of the mass within 0.3 m of the top place in over 80% of records" (REPORT I04). V1 crossed 0.9 in 312 of 549 records (56.8%), so it did not fire. But V3 fitted σ ≈ 0.06 m and kept more than 0.999 of its mass within 0.3 m in every record: the line would fire on the study's best density (`phase1d/README.md` Tables 1.1, 1.6). The top place succeeds in 417 of 549 records, so under a Q1 score moving mass off it costs more than it gains (§5 there).

**v1 rule** (proposal):
- **Q1: concentration is not a failure.** When the solver is usually right, nearly all mass on its top place is the correct answer. No collapse line; beat V0 and V3 instead. Phase 1d's reading, untested: a useful spread would need upstream uncertainty, such as perception, which the solver's keys cannot see (`phase1d/README.md` §5).
- **Q2: concentration is a failure,** caught by the score itself: a near-point mass leaves most region cells at the 1% floor.
- **Diagnostics, never judged:** mass within 0.3 m of the top place, and effective support (exp of the density's entropy, in m²) beside the region's area. Where the region is small, concentration and coverage coincide.
- A method claiming both questions passes each question's bars separately.

---

## 4. Crowd labels

L1: members-only regions, as today. L2: L1 minus cells within a 0.3 m body gap of non-members and minus other groups' circles. L3 (L2 with a fallback) was not measured (REPORT I06).

**What I06 found** (`phase2/i06/README.md` §2, "L1 and L2"). One scene equals one episode, so n is scenes; 199 variants come from 30 layouts (§0 there).
- Oracle C disagrees between labels in 3 of 199 (1.5%); L2 is never empty. This is built in: the solver keeps 1.0 m from people and L2 needs 0.85 m.
- L1's own reference stop fails L2 in 85 of 199 (42.7%), all 72 A2–A6 variants included.

**v1 rule.**
- **L1 is primary pending the mentor** (D4). A claim that holds under one label only is reported as label-dependent.
- **Affected methods:** those aiming near the region's reference point, that is R's and E's marked points and labels sampled for learned samplers such as C5 (REPORT §7). The choice also moves Q2's region cells in crowds. It does not affect the solver or the perception metrics.
- **Cue threshold.** At A5–A6 the confuser stands 0.6–0.8 m behind the line, facing within 30 deg of it, and L1 counts joining behind it as failure (REPORT §7). Until the mentor rules, report family A with and without A5–A6; I06's verdict is the same either way (`phase2/i06/README.md` §9). No join rule fixes A3–A6, where the confuser stands 0.03–0.44 m from the place in turn (`phase2/i06_repair/README.md` §2).

---

## 5. People-perception metrics

For detectors (I09) and VLM people stages, with I03's people scorer unchanged (`phase2/i09/README.md` §1, "The method and the scorer"). Bars are I09 A1's, verbatim.

| Metric | Definition | Bar |
|---|---|---|
| Recall | Matched / visible; greedy nearest-first one-to-one matching within 0.5 m; visible by `is_visible` | ≥ 0.95; kill < 0.90 |
| Line-member recall | Recall over waiting-line members | ≥ 0.95 |
| False people per image | Unmatched detections. Exempt (counted apart): within 0.5 m of the anchor of a target with no person. Near a person failing `is_visible`: false, flagged | ≤ 0.10; kill > 0.30 |
| Position error | Median over matched pairs, m | ≤ 0.15 m |
| Facing within 45 deg by view class | Angle between true yaw and bearing to camera: frontal < 45, oblique 45 to < 75, profile 75–105, back > 105 deg; unknown yaw is wrong, 180 deg in medians | Non-profile median ≤ 25 deg; non-profile within 45 ≥ 0.85; profile front/back right ≥ 0.90; kill: all matched within 45 < 0.70 |

- Also report recall by role class (staff was weakest: 69 of 78 on unseen layouts; `phase2/i09/README.md` Table 1.2).
- Detector unit: one (seed, row) image; Wilson 95% intervals, a layout bootstrap and a seed bootstrap. VLM unit: one answer per (arm, scene, trial), cache copies not counted (`phase1/README.md` §6, A0). A VLM's facing is compared with A0's bar of 0.806 (`phase1/README.md` §0).
- **Open scorer question** (D6). On seeds 200–212, 44 false people gave 0.242 per image. 13 sit 0.42–0.61 m from counter staff the visibility rule calls hidden, though the staff's marked point is in the picture (`phase2/i09/README.md` Table 1.5). Without them the rate is 31 of 182 = 0.170 (drafter's arithmetic), still above 0.10, so "neither" stands. Recommendation: a third class, "detected but hidden by rule", reported apart, with the frozen count beside it, applied to development numbers too (§3 there).

---

## 6. Speed metrics

Label every goal-step latency with one quantity (REPORT I21, "Why"; Qwen C, 143 realistic goal-step records):

| Quantity | Includes | Example |
|---|---|---|
| Record time | The record's `latency_s`: calls, pacer waits, wall time outside calls | 61.6 s |
| Provider time | Serial sum of per-call provider times | 61.2 s |
| Critical path | Longest chain of dependent calls with independent calls run together (replay) | 49.3 s |
| As the benchmark runs | People answers cached per scene: serial, then critical path | 53.2 s, 47.4 s |

Rules (proposal): median and p90 with n; fresh calls and cache hits apart; load stated (stage 23 ran 6 gateway jobs under the 85/min cap; REPORT §1.1, finding 8). C and E compared only under the same per-stage thinking policy, and "C faster than E" claimed only if C's matched time is at or below E's (REPORT I22). Local models: warm deployed path apart from a cold pass (I09: 55.6 ms warm, 215.1 ms cold confirmation pass; `phase2/i09/README.md` §1, "Runtime"). Name the timed span: the flow pilot's 18.3 ms covered sampling and projection only, with the raster (78.5 ms) and the floor work (110.8 ms) timed apart (`phase2/flow_pilot/README.md` Table 2.10).

---

## 7. Held-out data and multiplicity

| Data | Tune or fit? | Test? | Source |
|---|---|---|---|
| Development: whitelisted runs and the 68 development scenes (42 realistic, 26 placeholder) | Yes, leaving one layout out | Yes, but a method chosen on it needs fresh-layout confirmation | REPORT §8, "Tuning on the reporting sets" |
| Fit seeds 100–109 | Yes (I03, I11, I09 A1, gate 2) | Not for methods fitted on them | REPORT I46 |
| Training seeds 110–199 | Training only (110–189 trained the pilot, 190–199 chose its checkpoint) | No | `phase2/flow_pilot/README.md` §7 |
| Seeds 200–212 (182 layouts) | Proposal: development use only | No: spent as I09's test | `phase2/i09/README.md` §3 |
| Sealed seeds 3–99 | No; never generated or opened | Only as I46's sealed set | REPORT I46 |
| `keep/heldout`, `eval_16`–`19` (group photo, lift wait), runs s18 and s20_held | No; never read | Only the one final test (about 6,000 calls) | REPORT §8, "Held-out discipline"; I46 |

**Multiplicity** (proposal):
- Each study freezes and hashes a note before its test data exist or anything is scored, as phases 1d and 2 did, naming one primary metric, one primary comparison, unit, bar and interval method.
- One primary comparison per study. "Every model" claims use Holm's correction (test the smallest p at α/m, the next at α/(m−1), and so on).
- All else is secondary and descriptive. Per-situation results are descriptive. Post hoc and data-informed parts are labelled. Noise models and fitted parts use development data only (REPORT §8 (d)).

---

## 8. Open questions for the mentor

1. **Crowd labels.** L1, L2 or L3, and what cue threshold makes a nearby person count as waiting? L1's own stop fails L2 in 42.7% of I06's variants (`phase2/i06/README.md` §2).
2. **Heading tolerance.** 45 deg (ours) or 30 deg (yours) for the primary metric?
3. **Inner band edge.** 0.50 m (the human norm) or 0.55 m (the closest this robot can stand; REPORT §1.3)?
4. **Q2's ideal.** Are all poses in an acceptable region equally acceptable? If not, Q2 needs graded labels, for example human clicks (I35 Part B).
5. **Abstention.** Is "abstain and ask" a fair use of confidence in this course, and which abstention rates matter?
6. **Hidden-but-detected people.** Count as false, exempt, or report apart (§5)?
7. **External checks.** Could the 42 group-joining scenarios or the predictor's outputs be shared (I51), and does a click pilot need ethics review (I35)? Both would test our rules against something other than themselves (REPORT §8, "Circularity").
