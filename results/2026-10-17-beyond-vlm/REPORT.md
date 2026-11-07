# Beyond the VLM-only pipeline: research report

## How to read this report

- **Other work** carries a source and a status tag from the claim check (Appendix A):
  - [verified] means the checkers confirmed the claim against the source.
  - [partly] means the checkers corrected part of the claim; the corrected text is used here.
  - [contested] marks a figure the checks disagree on. One claim cited here carries it (the 17.75 KL figure in section 3.1). No claim cited here is tagged [verified-single].
  - [unchecked] marks a pointer to a source outside the checked set.
  - [source-checked] marks an external fact outside Appendix A that this revision confirmed directly at the source page on 2026-10-17 (a dataset record, a model card, an arXiv page). One reader checked each; there was no second checker.
- **(inference)** marks our reading of a source or of the brief, not something the source states.
- **(searcher note, unchecked)** marks a searcher's summary or open-question note. These notes were not checked.
- **Ids.** I01–I51 are the scored ideas (section 4). O-1 … O-5c and P-1 … P-15 are the brief's directions (brief section 6).
- **Held-out records** (group photo, lift) are used for nothing in this report.

---

## 1. Summary

### 1.1 What we found

1. **For the weaker models, most remaining loss is perception, and much of it looks repairable from exact depth at no model cost.**
   - All 29 GLM line failures under C have `members_recall` below 1 (12 of them 0). `people_found_share` is below 1 in 25, and 8 state no `come_after` (brief section 3.5).
   - [re-run] Of GLM's 74 missed line members (s19 realistic, 83 line goal steps), 56 (76%) are people the people call never found. On failed goal steps only it is about 80% (51 of 64 at lines; 55 of 68 with group members). (Caution: only 17 of the 74 have no detection within 1.0 m; most have a nearby detection that the one-to-one matching gave to a neighbour. With "found" meaning any detection within 0.5 m, the share is 0.716.)
   - [re-run] There are 15 wrong-target records whose target is a person. In all of them, the true target's pixels lie within 15 px of the model's point, and the depth under the point landed 1.0–12.3 m away: 9 on the occluding person in front, 6 on the background behind (2 on the wall, 1 at its foot, 3 on the floor short of it). The re-run confirms these are edge near-misses, not choice errors: the point falls up to 14 px outside the silhouette, and an oracle snap-and-lift fixes 15 of 15. [phase 1] Without the true mask, I11 Stage 0's zero-call depth-component snap fixes 5 of the 15 (all background leaks) and breaks no right-target record (`phase1b/README.md` section 3).
   - [phase 1] Phase 1 qualifies this finding. Under the recorded constraints, the full perception repair fixes 15 of GLM's 29 and 8 of DeepSeek's 15 line failures. All 15 of GLM's come from dropping detections 3.7–4.6 m from every true person (probably mislifted members); adding the missed people fixes 2 of 29. The perception ceilings are +6.3 (GLM), +4.4 (DeepSeek) and +0.8 pp (Qwen), lower bounds below the guessed +10–11.5, +6 and +1. Reference constraints alone give +7.9 and +9.5 pp (circular). (inference) Mislocated people and unstated `come_after` matter more than missed people.
2. **The headline numbers come from scenes built to avoid the solver's known crowd failure modes.** Extras stand at least 4.0 m from the target, at least 2.5 m from everyone, and off lines of sight. No line has a bystander, and occluded layouts are rejected (brief sections 3.3, 3.8, 4.6). The solver is not infallible there: C at groups is 48, 51 and 34 of 54 (brief section 3.2), and the solver alone with true people scored 75 of 90 at groups on the panel (brief section 3.1).
   - **RQ2 has only thin, confounded data.** There is one layout per line length (1–6) or group size (2–4), at most 2 bystanders and 1 other group, and none at lines. The brief's tables (sections 3.2–3.3: by line length, group size, density low/medium/high, 0–2 bystanders, 0–1 other groups) are the RQ2 baseline. Read at face value, they show success *rising* with bystanders, because those rows hold only nobody-near and group scenes (brief section 3.3).
3. **The literature has the parts but not this combination.**
   - ReKep has a VLM write constraints for an optimiser, in manipulation. Its automatic constraints reach 44.3%, against 68.6% for hand-annotated ones [verified].
   - No paper was found that scores sets or calibration for social stop poses. No benchmark was found that labels acceptable standing regions in crowds (searcher notes, unchecked).
   - The mentor's group-joining paper uses the same 0.5–1.2 m band, and its human check is a 15-person click pilot [verified]. That this pilot is its only human check is our inference. Our circularity problem is shared by the closest published work.
4. **Latency can fall only if the slow calls change.**
   - Decode was 93.3% of runtime in LocalNav [verified]. "Caveman" prompting cut only 10.7% of tokens [verified].
   - Setting thinking per stage might bring a Qwen C goal step to about 22 s, against 61.6 s today (guess, from small placeholder samples). Under the same policy, E's own 14.5 s goal call would shrink too, and it has never been measured at "none". So whether C beats E on time is open until both are measured under one policy (I22, live check L2).
6. **The honest statistical unit is the layout.**
   - [re-run] Two-sided exact sign-flip tests clustered by layout (statistic: the sum of the 14 layout mean differences) keep C > E for all three models: p = 0.031 (Qwen), 0.0078 (GLM), 0.00098 (DeepSeek).
   - Qwen's 0.031 is already the smallest value possible with 6 non-tied layouts (2/64). Any rescoring that ties or flips one layout pushes it above 0.05. So probe criteria in this report gate on margins and on layout counts, not on this p (I35 Part A).
   - No two-sided comparison within lines or within groups can reach p < 0.05 (nobody-near, with 6 layouts, could reach 0.031). Lines have 5 layouts, so the smallest possible p is 0.0625. Groups have 3, so it is 0.25.
7. **About fifteen ideas share one offline harness.** It loads recorded answers, re-solves them with the current solver and scores them with the package scorer.

       | Variant (near, far, clearance in m) | Qwen | GLM | DeepSeek | Haiku | Sonnet | Gemini |
       |---|---|---|---|---|---|---|
       | adopted (0.5, 1.2, 0.55) | +32 | +38 | +59 | +89 | +64 | +41 |
       | far 1.5 | +23 | +35 | +58 | +83 | +62 | +39 |
       | far 2.0 | +21 | +34 | +53 | +83 | +62 | +39 |
       | near 0.55 | +36 | +40 | +60 | +89 | +64 | +41 |
       | near 0.3 | same as adopted | | | | | |
       | far 1.0 (the solver's own standoff) | −40 | −11 | −5 | +20 | −15 | −17 |

       - Under far 1.0, C at lines falls to 2 / 1 / 14 / 9 / 0 / 0 of 90.
       - "Clear 0.50" (+25 / +36 / +59 / +89 / +62 / +31) is not a fair variant: it scores poses that overlap a body as reachable.
       - (inference) C's lead survives every plausible band, but only while the band's far edge lies beyond the solver's fixed 1.0 m standoff. This is the line caveat above in numbers. I35 Part A builds on this table (section 4.2).
     - **In-sample.** The context prompt was chosen on these same sets by C's success. All five lines are seen side-on at 4.4–6.2 m from the robot pose (4.2–6.0 m from the camera).
   - **(inference) What this changes here:**
     - **I01 moves to the centre.** The solver-alone row, the people-blind standard-goal row, and the split of every score into "target found" and "placement given the right target" are now the main offline result, for all six models.
     - **The scenes test less than the headline suggests.** Groups barely test people-awareness (the people-blind goal passes 38–48 of 54), and lines test recognition more than placement. This raises the value of I06's crowd variants and of I35's rule-sensitivity envelope. It also makes the "within 3 points of C" criteria of I05, I32, I13 and C5 easy to meet at groups, so they must be read at lines and on I06's crowd variants.
     - **Programme A has direct evidence.** The phantom people on the back wall are probably mislifted real members (13 of 20 sit within 25 px of a member who was never found; plausible, not proven), which favours relocating people (I11's snap-and-lift, or an I03 audit that moves people) over pruning. [phase 1] For GLM, pruning does as well: dropping unmatched detections alone (R4) fixes the same 15 line failures as the full repair (finding 9). [phase 1] With real depth blobs (I03 Stage B, secondary arms), relocating fixes 15 of GLM's 29 and 3 of DeepSeek's 15 line failures and add-plus-prune 13 and 6, so relocating leads only for GLM, and its 40 px radius is data-informed (`phase1b/README.md` section 2). The `eval_12` artefact supports I11's snap-and-lift. The too-close marked points support I34's Stage 0 snap control (Gemini: 62 of 99 failures fixed); the low points of Sonnet and Haiku sit beside the line on free floor, where the snap fixes 0 of 126 and 8 of 164.
     - The speed problem (Programme C) is mostly Qwen's thinking: fresh calls take a median 1.0–6.4 s per stage for the API models and DeepSeek (DeepSeek mostly under 2 s; GLM's constraints call 8.6 s), and Qwen's constraints call 35.8 s. Load during stage 23: 6 gateway jobs under the 85/min cap, 4 Anthropic jobs at 40/min, 3 Gemini jobs at 60/min.
     - Pooling caution: the s23 GLM and DeepSeek C people answers repeat the s19/s20 draws, and s21 Qwen's repeat s19's (143 of 144 identical), so they must not be counted as new samples. Within a stage, E and C also reuse R's plan and context answers, so the target is one draw shared by R, E and C.

### 1.2 Recommended first moves (offline first)

Phase 1 (working days 1–13), no model calls:

Phase 2 (working days 14–22 or so), no model calls except the live checks:
7. I35 Part A: rule-sensitivity envelope.
8. I21 and I22 offline replays.

- The secondary metrics for a set or density output. Top-1 stays official. [phase 3] A draft metric list v1 sets these and related questions out as decisions D1–D16, each with a recommendation and a default (`phase3/metrics_v1/METRIC_LIST_v1_DRAFT.md`; not agreed, so v0 stays in force).
- The labelling rule for non-members in crowded scenes (L1, L2 or L3), ideally with the mentor.
- Generality rulings: do geometric attention or relation tables (I05, I32, I13) count as situation-specific? May a prior learned from one situation's data (I27 source B) be used as a general preference key?
- Whether to ask the mentor for the group-joining benchmark and the predictor's outputs (I51).
- Who could write rules for new held-out situations, and the real deadline.
- [phase 2] Whether to adopt I06's facing gate on joining a line in the solver (a method candidate, `phase2/i06_repair/README.md`), which facing it should use, and whether to replay it first on records not used to design it.

**Do-first ideas:** I01, I14, I46 (statistics half), I06 (probe), I03, I11 (Stage 0), I09 (A1–A2, moved up 17 Oct), I21, I22, I35 (Part A).

---

## 2. How this was done

### 2.1 Angles searched

| Core angles (the brief's themes) | Gap angles (added to catch what the core would miss) |
|---|---|
| Goals as distributions, heatmaps or candidate sets | VLMs writing constraints, cost maps or value functions for social navigation |
| Non-VLM perception: people, heading, groups | VLM spatial placement: candidates, marks, verifiers, pointing models |
| Crowded and confusing social scenes | Reinforcement fine-tuning with verifiable spatial rewards |
| Faster inference for robot decisions | Worked examples, and optimising one shared prompt set across VLMs |
| Diffusion, flow matching, energy-based models | Agentic VLMs with tools (zoom, detectors, code over geometry) |
| Small models trained on generated scenes | Human judgments of where to stand; metrics for goal distributions |
| Uncertainty, calibration, selection | |
| Target grounding under clutter and distractors | |

### 2.2 Sources and claim checking

- 112 sources; 0 with a mismatched title or URL.
- 372 claims checked: 313 verified, 59 partly right and corrected. Refuted claims were dropped.
- The searchers' summaries, surprises and open questions were not checked.

### 2.3 How ideas were generated

- **Eight lenses:** offline-first, crowds, hybrid perception, speed, distribution output, contrarian, cross-domain, generative moonshots.
- **The brief's 22 directions** (O-1, O-2, O-3, O-4, O-5a, O-5b, O-5c, P-1 to P-15) were included as ideas.
- **Merging:** near-duplicates were merged, keeping the variants as notes.
- **Completeness critic:** this pass added I46–I51.
- The result is 51 ideas.

### 2.4 How ideas were scored

- **Feasibility:** effort, calls, compute, dependencies, rule risks, blockers; scored 1–5.
- **Payoff:** gain in pp (low, mid, high), latency, failure classes reached, novelty, report value; scored 1–5.
- **Skeptic:** risks, evidence against, confounds, worst likely outcome.
- **Steelman:** the strongest honest case, plus a probe with success and kill criteria and a call count.

### 2.5 Limits

- **Probabilities are uncalibrated guesses.** Priority differences under about 0.1 mean little.
- **Feasibility multiplies priority, so cheap zero-call ideas rise.** That is a deliberate bias toward offline work, but it is still a bias.
- **No human has reviewed the ideas or the scores.**

---

## 3. The landscape by theme

Appendix A has every source. This section keeps what bears on decisions.

### 3.1 Distribution outputs

**Now:** methods predict a floor map, then score its best point.
- **The mentor's group-joining paper (arXiv:2609.28467):**
  - It outputs an energy map with cos/sin heading maps and takes the minimum-energy cell [verified].
  - It samples 10 poses per scenario and reports KL divergence to 15 participants' choices: 14.87 for its method, against 19.60 for GPT-5.5 and 19.51 for few-shot GPT-5.5 [verified]. A figure of 17.75 for Gemini 3 Flash is [contested]: one check confirmed it, and another did not find it in the fetched table.
  - Pose validity is 89.76% at 1.108 s. Few-shot GPT-5.5 gets 59.76% at 49.25 s, and few-shot Gemini 3 Flash 53.57% at 27.80 s [verified].
  - With oracle groups it reaches 94.76%; a hand-written rule reaches 64.29% [verified].
- **Dual-Heatmap (arXiv:2605.19420):** an MPPI planner integrates the heatmap, with no single goal point [verified]. Affordance rate is 0.7265 vs 0.2492 for Molmo [verified]. Simulation only [verified].
- **RoboMAP (arXiv:2510.10912):** scores only its peak point and calls this "overly simplistic" [verified].
- **VoxPoser:** affordance map ×2 plus avoidance map ×1 [verified]. This is a scalar cousin of our lexicographic solver.
- **VLFM:** confidence-weighted fusion gives success rate (SR) 52.5, vs 44.5 for replacing the map [verified].
- **Nav2 NavigateThroughPoses:** visits poses in order; it takes no alternatives [verified].
- **Best-of-k is not a strictly proper score.** minADE falls as k grows even for an unchanged prediction (Shahroudi et al., ICML 2024) [verified].

**For us:** a set or density from the solver is cheap to produce. It has to be scored against chance and against spread-out baselines, and something has to consume it.

**Open (searcher notes):** no metric for set, coverage or mass-in-region for social stop poses, and no conformal method for a continuous goal region. Ideas: I14, I04, I12, I15.

### 3.2 Other models helping or replacing VLM steps

- **The mentor's pipeline:** YOLOv10 plus RTMW for orientation, spectral partitioning, a SigLIP2 ranker; grounding takes 0.936 s [verified].
- **SAM 3:** 30 ms per image with 100+ objects on an H200 [verified]; LVIS box AP 53.6, COCO 56.4 [verified]; short noun-phrase prompts only [partly].
- **MEBOW:** facing MAE 8.393 deg, 96.9% within 30 deg, on real crops with ground-truth boxes [verified].
- **RTMW:** 35.3–47.6 ms per crop on CPU [verified]. Top-down, so it suits scenes of about 6 people or fewer [verified].
- **GCFF:** F1 0.89 (T=2/3) and 0.71 (T=1) [partly], with parameters tuned per dataset [verified].
- **DANTE vs GCFF on Cocktail Party:** 0.73 vs 0.64 at T=1, not significant (p = 0.28) [verified].
- **Mesh models** (Multi-HMR 2, SAM 3D Body) report no facing accuracy [verified].
- **"Seeing Is Not Measuring":** tools help a 4B VLM only with a good detector; weak detectors score below having no tools [verified].
- **VADAR:** zero-shot programs score 40.4 vs 42.9 for direct GPT-4o. On CLEVR, programs are right 83.0% of the time but executed answers only 53.6% [verified].

**For us:** with exact depth, a body's position is nearly free once something finds it. Facing is the weak link, and a bad tool is worse than none.

**Open:** no measurement of real-photo detectors on avatar renders. Ideas: I03, I09, I33, I10. [phase 1] I09 A1 made one: a COCO-trained RT-DETRv2 finds 116 of 118 visible people on the 42 realistic development images, but 49 of 58 on the placeholder scenes, where 9 audience members of one situation, seen side-on or from behind, score below its threshold (`phase1c/README.md` section 1).

### 3.3 Crowded and confusing scenes

- **SocialNav-SUB:** about 6.65 people per scene. The best VLM scores 0.62, a rule-based baseline 0.64, and the average human 0.60 [verified].
- **HumanRef:** Qwen2.5-VL-7B has DensityF1 56.2 and correct rejection of 7.1% [verified].
- **CrowdCue:** counting error grows from MAE 8.99 (0–50 people) to 448.56 (>1,000) [verified]. A detector count given as digits helped (62.65 vs 81.64); a heatmap given as an image hurt (96.26) [partly].
- **GSON:** prompted grouping is correct 77.8% of the time for GPT-4o and 53.6% for Qwen2.5-VL-7B [verified].
- **HabiCrowd:** density 0.1–0.5 people per m², scored by trajectories [verified]; success rate falls with density [partly].
- **CongreG8:** 380 human and 38 robot approach trials, groups of three or four [verified, partly].
- **Procedural cost map (arXiv:2407.10547):** 95.75–98.67% in distribution; fails on bent queues and at about 3 m spacing [verified].

**For us:** a crowd generator with exact goal regions would be new. People recall is likely the first thing to fail as crowds grow.

**Open:** no benchmark with standing regions in crowds. Ideas: I06, I18, I19, I40.

### 3.4 Faster inference

- **LocalNav [verified]:**
  - Decode is 93.3% of runtime.
  - A brevity reward cut output tokens from 535.2 to 149.37 and runtime by 71.8%.
  - IQ4_XS quantisation raised speed from 17.58 to 39.43 tok/s.
  - "Caveman" prompting cut tokens by 10.7%.
  - Suppressing chain of thought cut tokens by 91.82%, but SPL fell 3.9 points and queries rose 1.65×.
  - A 4B student trained on about 500 traces scored 34.5% vs 39.7% for its teacher.
- **Dual-process navigation:** the fast path takes 0.7–0.9 s and the slow path 17–19 s [verified]. The source adds that "it handled 53.6% of all decisions", where "it" could mean either path (inference). So which path handles 53.6% is unclear.
- **Prompt caps on reasoning length are ineffective** (arXiv:2511.21397) [verified].
- **Speculative decoding** reaches about 2× only with trained drafters; EAGLE-3 gives 0.96× [verified].
- **G2-Nav:** a 2B VLM with short outputs answers in about 1.5 s [verified].
- **The mentor's distillation paper** reports no latency figures [verified].

**For us:** in rough order of cost, the levers are concurrency, thinking set per stage, a shorter or skipped constraints call, a geometric gate, and only then local models.

**Open:** no measured VLM latency on an RTX 4060. Ideas: I21, I22, I23, I39, I24, I44.

### 3.5 Diffusion, flow and energy models

- **Diffusion-CCSP:** summed per-constraint energies generalise to new combinations. It needs 10k–30k instances per task, and its labels come from hand-coded rules [verified].
- **SoGuDiff:** without its projection layer, success drops from 99% to 70%. Joint guidance fails for some combinations [verified].
- **FLUX:** 216.5 ms vs 468.4 ms for NoMaD [partly]; the authors say it lacks explicit social reasoning [partly].
- **Dream2Real:** about 6 minutes per instruction; its CLIP scorer reads spatial relations like a bag of words [verified].

**For us:** the mentor's energy map shows a learned goal head works without diffusion. Compositional energies are the learned twin of our solver.

**Open:** no generative model that outputs a stop pose from a request plus RGB-D. Ideas: I07, I27, I28, I29, I30, I31, I42. Section 4.6 answers directly whether diffusion can be used somewhere.

### 3.6 Uncertainty and selection

- **KnowNo:** hardware task success 0.74 with help on 0.58 of steps, vs 0.72 and 0.72 for a simple prediction set. Its guarantee assumes a fully grounded scene [verified].
- **ENCP:** plain conformal prediction undercovers (0.854 vs a 0.90 target) [partly]. The guarantee does not survive a shift from seen to unseen [verified].
- **GUI-RC:** spatial voting moves 86.48 to 88.52 for a model that outputs points [verified].
- **Argus:** uncertainty-method rankings do not transfer to API models (rho about +0.08) [verified].
- **Decoding-format audit:** a selector's gain vanishes against a format-matched majority vote [verified].
- **Probing world models:** a random verifier gets 58.67–67.33%, vs 63.33–67.33% for a heuristic verifier [verified].
- **CUSP:** collective uncertainty across an ensemble of small models gives AUROC 0.764 for error detection. With commercial models, disagreement measured as JSD is the strongest signal, at AUROC 0.819 [verified].

**For us:** beat plain voting and random picks first. Disagreement across trials and across models is a cheap signal. [phase 3] Measured on the 441 old-wording realistic goal-step records (`phase3/confidence/README.md`): disagreement across C's three trials predicts failure with a pooled AUROC of 0.774 (but costs two extra runs), and disagreement between the VLM's people and a local detector's 0.741, against 0.595 for the solver's flags. Both are reported, not judged, and were designed knowing A2's mechanism; the judged signal, the shift of C's place under the detector's people, is a fragile neither (pooled 0.708).

**Open:** no conformal method for continuous goal poses. Ideas: I02, I12, I15, I36, I39.

### 3.7 Target grounding

- **Set-of-Mark:** 25.7 → 86.4 for GPT-4V. Ground-truth masks give 90.1 vs 75.6 with proposals. Open models "can hardly interpret the marks" [verified].
- **Ref-Adv:** Qwen2.5-VL-72B falls from 92.7% to 58.3% with same-kind distractors; the best model reaches 63.7 [verified]. Chain of thought helps unevenly [partly].
- **"VLMs Can Describe, But Not Measure":** direct VLM depth error is 192.0 mm vs 12.1 mm from mask plus measured depth. The mask median beats the box centre, 13.2 vs 15.8 mm [verified].
- **Think-Act-Build:** numbered SAM3 masks with an ID, 0 or −1 answer [verified], but no ablation of that step [verified].
- **LocateBench:** picking a letter beats giving coordinates for most models, but Claude-3 Opus reverses (38.6 vs 31.3) [verified].

**For us:** separate choosing the target from lifting it to 3D. Lift with the median depth over the mask. Check that the gateway models can read numbered marks.

**Open:** no study of lift error at occluder edges. Ideas: I11, I33, I47.

### 3.8 Training on synthetic data

- **The mentor's goal predictor:** 10k train and 2k validation simulated samples from 29 F-formation transitions [verified].
- **Mentor's distillation:** GPT-4o labels on 4,851 pairs [verified].
- **2026 survey:** synthetic data is promising but uncertain, and lightweight VLMs reason worse in socially complex scenes [verified].

**For us:** feasible, but labels from our own rules are circular.

**Open:** no test of a learned social-placement model on a situation it was not trained on. Ideas: I13, I26, I31, I37.

### 3.9 Gap angle: VLMs that write constraints or costs

- **ReKep:** the optimiser contributes little error; the point tracker and the VLM contribute most [verified].
- **G2-Nav:** without VLM scoring, error is 1.12 m L2 vs 0.55 m with it [partly].
- **The mentor's path-selection paper:** only the current frame is seen, so long queues are missed. Simultaneous activities confuse the choice [verified].
- **Vazquez et al.:** personal-space weighting hampered finding gaps in larger groups [verified].

**For us:** our design is a known pattern applied to a new problem. No one has measured, for social placement, the gap between constraints the VLM writes and hand-written ones. I01's oracle-constraint arm measures it.

### 3.10 Gap angle: VLM spatial placement

- **RoboPoint:** 46.77% vs 29.06% for GPT-4o [verified].
- **RoboSpatial context questions:** GPT-4o 5.7%, Molmo 0.1% [verified].
- **VLMnav:** 50.4% with the geometric proposer, 33.2% without it, 29.8% as text only, 24.6% for PIVOT [verified].
- **PIVOT:** links correct reasoning to the wrong mark [verified].
- **PointArena:** chain of thought and examples did not reliably help [verified].

**For us:** the geometric proposer matters more than the marks. Free-space pointing is weak (I24, I34, I41).

### 3.11 Gap angle: reinforcement learning with verifiable rewards

- **Visual-RFT:** RFT beats SFT on tiny data [verified].
- **Chu et al.:** RL generalises, but needs an SFT warm-up [verified].
- **Lu et al.:** difficulty-filtered SFT 62.10 vs GRPO 59.48 out of distribution at 7B [verified].
- **Embodied-R1:** 8 A100 GPUs for about 48 h per phase; reward hacking observed [verified].
- **GUI-G2:** a dense reward scores 93.3 vs 87.4 for a sparse one [verified].

**For us:** out of reach this term (the RL extension of I50 and I31 are parked).

### 3.12 Gap angle: worked examples and a shared prompt set

- **Demonstration text carries the signal** (arXiv:2407.00902) [verified].
- **More shots can hurt**, and open models often gain nothing [verified].
- **Exemplar optimisation:** above 11%, vs about 8% for instruction optimisation [verified].
- **MAGE:** fixed prompts beat optimisers at about 30 training items [verified].
- **Prompt transfer** between two 4B models helped in 3 of 8 cases [verified].

**For us:** see I08 (and I50, parked).

### 3.13 Gap angle: agentic VLMs with tools

- **Zoom search without training** helps small models, but needs logits [verified].
- **An untrained agent mode** scored 62.83%, against 80.63% for answering directly [verified].
- **An untrained 8B planner** got 30.7 vs 31.1 [verified].

**For us:** only a fixed second look driven by the host is safe. No idea here depends on an agent loop.

### 3.14 Gap angle: human judgments and metrics

- **The mentor's validity rule:** 0.5–1.2 m from the relevant members, heading within ±30 deg [verified]. Benchmark: 15 conversation, 15 queue, 12 audience scenes [partly].
- **SocNav1:** weighted kappa 0.61–0.71 [verified].
- **MANNERS-DB:** 15 raters per scene [verified].
- **Francis et al.:** report distributions and a battery of metrics [verified].
- **Bylinskii et al.:** KL, SIM and IG punish zero bins [verified].
- **Survey with N = 181:** preferred approach distance differs by country [verified].

**For us:** a click study on our renders would be stronger than the closest work's (I35, I27).

---

## 4. Idea portfolio

### 4.1 Score table (sorted by priority)

- "probe calls" counts the first probe only. Conditional live stages that follow a passed probe, such as I03's L3 (about 80–100 calls) and I11's L4 (about 180–210), are in section 6.
- I25's gain (5/20/40 pp) is measured against keeping a stale pose on a new paired set, not against current C. It is not comparable with the other rows.

| id | title | themes | bold | feas | payoff | gain pp (lo/mid/hi) | novelty | p_disappoint | p_upside | EV | priority | probe calls | offline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I06 | Crowd generator pass with factor sweeps and zero-call solver degradation curves | crowds, data, evaluation | incremental | 4 | 4 | 0/3/15 | 3 | 0.45 | 0.25 | 2.2 | 1.76 | 0 | yes |
| I14 | Proper scoreboard for goal sets and densities with chance baselines | evaluation, distribution | incremental | 5 | 3 | 0/0/0 | 3 | 0.5 | 0.2 | 1.5 | 1.5 | 0 | yes |
| I21 | Pipeline as a dependency graph: concurrent people call and early constraints | speed | incremental | 5 | 2 | -1/0/0 | 1 | 0.35 | 0.1 | 1.3 | 1.3 | 80 | no |
| I03 | Depth-blob audit that completes the VLM people list | other_models, crowds | incremental | 5 | 3 | -1/3.5/7 | 2 | 0.6 | 0.18 | 1.2 | 1.2 | 0 | yes |
| I09 | Local person detector plus keypoints as the people source | other_models, speed, evaluation | ambitious | 4 | 3 | -3/3/7 | 2 | 0.55 | 0.22 | 1.35 | 1.08 | 0 | yes |
| I35 | Rule-sensitivity rescoring and a human click study of the polygons | evaluation, distribution, data | ambitious | 4 | 3 | 0/0/2 | 3 | 0.57 | 0.2 | 1.29 | 1.03 | 0 | yes |
| I11 | Numbered candidate masks for target grounding against occluders and look-alikes | grounding, crowds, other_models | ambitious | 4 | 3 | 0/4/8 | 3 | 0.6 | 0.25 | 1.2 | 0.96 | 0 | yes |
| I18 | Adversarial or falsification search for crowd layouts that break the solver | crowds, evaluation, data | ambitious | 4 | 3 | 0/2/8 | 3 | 0.6 | 0.25 | 1.2 | 0.96 | 0 | yes |
| I05 | Geometric attention structure (chains, rings, shared focus) that derives or informs constraints | speed, other_models, crowds | ambitious | 4 | 3 | -2/0.5/3 | 2 | 0.62 | 0.2 | 1.14 | 0.91 | 0 | yes |
| I33 | Depth proposes people and target candidates; VLM only verifies | grounding, other_models, speed, crowds | ambitious | 4 | 3 | -1/4/9 | 2 | 0.62 | 0.35 | 1.14 | 0.91 | 84 | no |
| I02 | Fusing trials and models: early fusion of perceptions vs late fusion of goal maps | distribution, other_models, evaluation | incremental | 5 | 3 | 0/3/8 | 2 | 0.7 | 0.17 | 0.9 | 0.9 | 0 | yes |
| I47 | Finding the side people use, for the standard goal when the surface fit fails | grounding, other | incremental | 5 | 2 | 0.4/1.6/3.2 | 2 | 0.55 | 0.2 | 0.9 | 0.9 | 0 | yes |
| I49 | Sensor-realism replay: re-lift every recorded answer through noisy, monocular and miscalibrated depth | evaluation, grounding, other_models | incremental | 5 | 2 | 0/1/6 | 2 | 0.55 | 0.3 | 0.9 | 0.9 | 0 | yes |
| I19 | VLM states kinds and an anchor; geometric expanders compute membership | crowds, speed, other | ambitious | 4 | 3 | -1/1.5/5 | 3 | 0.63 | 0.12 | 1.11 | 0.89 | 0 | yes |
| I12 | Monte Carlo over perception hypotheses through the solver for a robust pose and spread | distribution, other_models, evaluation | ambitious | 5 | 3 | -1/1.5/4 | 3 | 0.73 | 0.15 | 0.81 | 0.81 | 0 | yes |
| I25 | Constraints as a standing program re-bound to fresh detections | speed, crowds, data | ambitious | 4 | 3 | 5/20/40 | 3 | 0.68 | 0.25 | 0.96 | 0.77 | 0 | yes |
| I34 | Pointing on a metric top-down map (R-BEV) | grounding, other | incremental | 4 | 3 | -10/-3/3 | 3 | 0.68 | 0.2 | 0.96 | 0.77 | 120 | no |
| I39 | Model-escalation cascade driven by uncertainty without ground truth | speed, other_models, evaluation | ambitious | 4 | 3 | 3/7/12 | 2 | 0.68 | 0.1 | 0.96 | 0.77 | 0 | yes |
| I23 | Fast-slow cascade: solver-alone pose unless a geometric conflict needs the constraints call | speed, distribution, crowds | ambitious | 5 | 2 | -3/0.5/3 | 2 | 0.62 | 0.15 | 0.76 | 0.76 | 0 | yes |
| I32 | Attention edges in the people call replace the constraints call | other, speed, crowds | ambitious | 4 | 3 | -4/1/5 | 3 | 0.7 | 0.2 | 0.9 | 0.72 | 0 | yes |
| I46 | Sealed confirmatory benchmark: new situations written blind, fresh seeds sized by a power analysis, and layout-clustered tests | evaluation, data | incremental | 3 | 3 | 0/0/0 | 2 | 0.6 | 0.22 | 1.2 | 0.72 | 0 | yes |
| I04 | Soft solver: lexicographic keys turned into a calibrated density over poses | distribution, evaluation | incremental | 5 | 2 | 0/0/0 | 2 | 0.65 | 0.2 | 0.7 | 0.7 | 0 | yes |
| I08 | Solver-verified constraint search for worked examples and language coverage | data, evaluation, other_models | ambitious | 4 | 3 | -1/3/8 | 3 | 0.72 | 0.15 | 0.84 | 0.67 | 0 | yes |
| I24 | Choose, don't write: solver enumerates constraint hypotheses, VLM picks a numbered stop | speed, distribution, evaluation | ambitious | 4 | 3 | -4/1/6 | 3 | 0.72 | 0.15 | 0.84 | 0.67 | 0 | yes |
| I10 | Geometric gate in place of the context call | speed, other_models | incremental | 4 | 2 | -1.5/0.8/2.8 | 2 | 0.6 | 0.12 | 0.8 | 0.64 | 0 | yes |
| I51 | Real-image check and benchmark swap with the mentor's group-joining system | evaluation, data, other_models | ambitious | 3 | 3 | 0/0/0 | 3 | 0.66 | 0.17 | 1.02 | 0.61 | 0 | yes |
| I15 | Conformal goal sets that decide when to look again or ask | distribution, evaluation, speed, grounding | ambitious | 4 | 3 | 0/2/6 | 3 | 0.78 | 0.13 | 0.66 | 0.53 | 0 | yes |
| I48 | Solver-diagnosed revision: one targeted re-ask, only when geometry flags the answer | other, crowds | incremental | 4 | 2 | -0.5/2/4 | 2 | 0.68 | 0.2 | 0.64 | 0.51 | 0 | yes |
| I13 | Small learned network that produces solver constraints from people geometry | speed, data, other_models, crowds | moonshot | 3 | 3 | -6/4/11 | 4 | 0.72 | 0.15 | 0.84 | 0.5 | 0 | yes |
| I38 | Facing- and density-dependent personal space and transactional segments in the solver | grounding, other | incremental | 4 | 2 | -1.5/0.4/2.5 | 2 | 0.72 | 0.12 | 0.56 | 0.45 | 0 | yes |
| I36 | Diverse candidate slate with a cautious pairwise VLM re-ranker | distribution, evaluation, other_models | incremental | 4 | 2 | -1/1.5/4.5 | 2 | 0.75 | 0.15 | 0.5 | 0.4 | 0 | yes |
| I41 | Numbered free-floor candidates for the VLM to choose, filtered by reachability |  | - | 4 | 2 | -2/4/10 | 2 | 0.75 | 0.2 | 0.5 | 0.4 | 30 | no |
| I16 | Path-aware choice of stop from the goal set | distribution, crowds, evaluation | ambitious | 4 | 2 | -2/0/1 | 2 | 0.77 | 0.12 | 0.46 | 0.37 | 0 | yes |
| I20 | Posterior over attention graphs giving constraints and a pose distribution | crowds, distribution, other | moonshot | 3 | 3 | -2/3/7 | 4 | 0.8 | 0.15 | 0.6 | 0.36 | 0 | yes |
| I27 | Masked-person prior: robot stands where one more person would | diffusion_generative, distribution, data, speed, evaluation | ambitious | 3 | 3 | 0/1/4 | 3 | 0.8 | 0.15 | 0.6 | 0.36 | 0 | yes |
| I37 | Candidate quality scorer trained on generator scenes with perception noise | data, distribution, speed | ambitious | 3 | 3 | -0.5/2/5 | 3 | 0.8 | 0.15 | 0.6 | 0.36 | 0 | yes |
| I40 | Crowd simulator as a crowded-scene generator and independent newcomer label | data, evaluation, crowds | moonshot | 3 | 3 | 0/0/3 | 4 | 0.8 | 0.12 | 0.6 | 0.36 | 0 | yes |
| I50 | Solver-verified optimisation of the constraints prompt across the model panel, with RL fine-tuning as the bold extension | other, evaluation, other_models | ambitious | 3 | 2 | 0/3/7 | 3 | 0.72 | 0.12 | 0.56 | 0.34 | 0 | yes |
| I29 | Depth-exact robot compositing at candidates with VLM forced choice | other_models, distribution, speed, evaluation, diffusion_generative | ambitious | 4 | 2 | -1.5/1/4 | 3 | 0.8 | 0.12 | 0.4 | 0.32 | 60 | no |
| I07 | Learned BEV goal energy composed with solver/VLM constraints, tested leave-one-situation-out | diffusion_generative, distribution, data, other_models | moonshot | 3 | 2 | -2/1/5 | 3 | 0.8 | 0.13 | 0.4 | 0.24 | 0 | yes |
| I44 | A faster mixture-of-experts model |  | - | 4 | 1 | -5/1/5 | 1 | 0.75 | 0.3 | 0.25 | 0.2 | 124 | no |
| I30 | Generative completion of hidden or missed people, marginalised through the solver | crowds, diffusion_generative, distribution, grounding | moonshot | 3 | 2 | -1/2/5 | 3 | 0.85 | 0.2 | 0.3 | 0.18 | 0 | yes |
| I28 | Learned compositional energies per constraint kind | diffusion_generative, distribution, crowds | ambitious | 3 | 2 | -2/1/5 | 4 | 0.87 | 0.08 | 0.26 | 0.16 | 0 | yes |
| I43 | Short frame sequences or video |  | - | 2 | 2 | 0/3/10 | 2 | 0.8 | 0.15 | 0.4 | 0.16 | 0 | yes |
| I26 | Distil people and constraint answers into a small local VLM | speed, data, other_models | moonshot | 2 | 2 | -15/-5/2 | 3 | 0.82 | 0.08 | 0.36 | 0.14 | 0 | yes |
| I17 | Active second view to resolve a spread goal or uncertain crowd | distribution, crowds, grounding | moonshot | 2 | 2 | 0/1.5/6 | 4 | 0.85 | 0.12 | 0.3 | 0.12 | 0 | yes |
| I31 | End-to-end trained goal predictor on a small local VLM | diffusion_generative, speed, data, distribution, other_models | moonshot | 2 | 2 | -12/3/15 | 3 | 0.85 | 0.1 | 0.3 | 0.12 | 0 | yes |
| I45 | Tour-based grounding (Mobility VLA style) |  | - | 4 | 1 | -2/0/1 | 1 | 0.85 | 0.12 | 0.15 | 0.12 | 306 | no |
| I42 | Generative image models to make crowded or realistic training/test images |  | - | 2 | 1 | 0/0/1 | 2 | 0.85 | 0.1 | 0.15 | 0.06 | 60 | no |

**Tier moves against priority.**
- I18 (0.96, the same priority as I11) stays in tier (b). It waits on the I06 probe, whose crowd builder it shares, so it cannot start before phase 2 ends.
- I25 (0.77) and I32 (0.72) go from tier (b) priority to bold bets. Their priority matches tier-(b) ideas, but each has a large upside if it works (I25: a re-binding path to temporal updates; I32: removing the constraints call) and p_disappoint of 0.68 and 0.70.
- I27 (0.36) and I13 (0.50) go to bold bets. Their first probes cost no calls, and they are the only routes to human-grounded labels and to a leave-one-situation-out test.
- I10 (0.64) goes to parked. Its latency gain needs I03's blobs, and with the VLM people call it slows Qwen's nobody-near episodes.

### 4.2 Tier (a): do first

Most of these share **step 0**, the offline harness in section 6. All are zero-call except the small live checks in I21 and I22.

**What.**
- Replay every recorded C goal step from the dev runs through the unchanged solver, repairing one input at a time:
  - R0: constraints removed (solver alone).
  - R1: true positions. R2: true facing.
  - R3: missed people added as unlisted bodies. R4: spurious people dropped.
  - R5: reference constraints (harness-only; a circular upper bound).
  - R6: the oracle target.
- Score each with the package scorer.
- Add VLM facing accuracy in 45-deg bins (this is I09's step A0).

**Why.**
- No solver-alone run existed on the realistic sets until step 0, though `README.md:129` asks that every C run be paired with one (brief section 3.8). The replay now gives it offline for all six models (`i01/README.md`, Table 1a: 0 of 90 at lines, at least C at groups).
- Oracle people were measured only on the old panel, where they gave +22 of 420 (brief section 3.4).
- All GLM line failures have recall below 1 (brief section 3.5).
- ReKep reports its auto vs annotated gap only in aggregate (44.3 vs 68.6) [verified]. The mentor's paper gives only oracle groups [verified].

**Payoff.**
- 0 pp directly.
- Indirect gain from steering effort: 0/1/4 pp (guess).

**Better than expected.** The repairs, calibrated on the placeholder panel against the real true-people reruns (`s8_conorc`: GLM +5 and qwen3vl +11 of 84), recover most of the real gain. That would make this a trusted zero-call test harness for any people source. R0 might also show that constraints hurt groups. On the panel, the solver alone scored 75 vs 69 for C (brief section 3.1).

**Worse than expected.**
- R3 flips a moderate share of failures, with wide intervals.
- R5 comes out near 100% by construction.
- The table mostly restates brief section 3.9.
- The repairs keep constraints written against the model's own people, so perception gains are lower bounds.

- Step 0: reproduce the recorded places with `--expect-reproduced`. s19 and s20 are not on the gate list.
- Step 1: R3 and P (R1–R4) on GLM's 29 and DeepSeek's 15 line failures, plus any successes the repairs break.
- Step 2: the s8 calibration.
- Success, all of:
  - 100% of places reproduced.
  - P flips at least 15 of 29 (GLM) and at least 8 of 15 (DeepSeek); R3 alone flips at least 10 of 29; at most 2 successes lost per model.
  - The calibration recovers at least half of the real gain, with at least 80% agreement per record and at most 10% ambiguous matches.
- Kill, if any of:
  - P flips at most 8 of 29 and at most 4 of 15, or R3 flips at most 5 while breaking 3 or more. Then perception is not the lever under the recorded constraints; run R5 before blaming the constraints.
  - The calibration recovers less than a third, or agreement is below 65%. Then the harness is not trusted for allocating effort.
  - More than 20% of matches are ambiguous.
- **Phase-1 result (17 Oct; `phase1/README.md` sections 1–2) [phase 1].**
  - Step 0: 1,301 of 1,301 recorded solver places reproduce (s19 and s20 included).
  - Step 1, old wording, line failures flipped (GLM n 29 / DeepSeek n 15): P 15 / 8, exactly on the bars; R3 alone 2 / 2; R4 alone (dropping unmatched detections) 15 / 4; R5 (circular) 19 / 9; true people with reference constraints 29 / 15. Over all solver steps, P loses 0 (GLM), 4 (DeepSeek) and 0 (Qwen) successes, and R3 loses 0; over line steps only, DeepSeek P loses 1.
  - Step 1 verdict: neither success nor kill. R3 misses its bar badly. DeepSeek's 4 losses miss "at most 2" if the clause covers all situations (it is met if it covers lines only). The two P bars rest on few independent answers: GLM's 15 flips come from 9 people answers in 3 layouts, DeepSeek's 8 from 4.
  - Mechanism: all 15 GLM P flips come through R4. Each has a `come_after` that lists a detection 3.7–4.6 m from every true person, probably a member whose depth fell through (plausible, not proven). So perception helps here by removing mislocated detections that `come_after` names, not by adding missed people.
  - Step 2 (s8, five placeholder models, 420 records): P recovers +22 of the real +22 oracle-people gain (share 1.00, 95% CI about 0.65–2.1), agrees with `s8_conorc` on 90.6% of 213 goal steps (95.2% of 420), with 0.5% ambiguous matches. Success met and kill not met, pooled. Caveats: doing nothing already agrees on 80.3% of goal steps; the net match is partly cancellation (24 of 32 real wins and 0 of 10 real losses reproduced); the criteria do not say pooled or per model, and by the letter ornith15 and qwen38 (real gain +2 each, CI includes 0) meet the kill's share clause.
  - New wording (extra rows; Qwen, GLM and DeepSeek people answers are cache copies): perception repairs give at most +0.4 pp for Haiku, Sonnet and Gemini. Gemini's loss is the target (oracle target +10.3 pp). R5 fixes all 11 of Haiku's line failures, and P fixes 1.
  - (inference) For GLM and DeepSeek, effort should go to removing or relocating mislocated people and to stating `come_after` correctly, more than to adding missed people. A pixel-space match or depth-consistency filter for these detections is the natural next arm (not pre-registered). [phase 1] I03 Stage B then ran both as secondary arms with real blobs (each adding the remaining blobs): relocating unsupported detections to a blob within 40 px gives GLM 57 and DeepSeek 71 line successes of 90, and pruning them gives 55 and 74, both passing Stage B's bars as secondary results; the 40 px radius was registered after I01 on these runs (`phase1b/README.md` section 2).

- Where the code lives.
- The optional pipeline-level oracle-people arm: about 150 constraints calls per model, about 450 for three if plan and context come from the cache, and up to about 2,300 if not. This is a big run.
- Report wording: perception repairs are lower bounds; R5 is an upper bound and circular.

#### I14: Scoreboard for sets and densities, with chance baselines (plus the statistics half of I46)

**What.**
- Secondary metrics, computed on the harness side. Official top-1 is unchanged.
  - P(success) over a set.
  - Lift over the region's share of floor.
  - Mass in each forbidden kind.
  - hit@k for k = 1–5, against random-k and farthest-point-k baselines.
  - Stability under 5 cm jitter.
  - A heading-tolerance curve (45 deg vs the mentor's 30 deg).
- From I46: layout-clustered exact sign-flip tests, the intra-layout correlation (ICC), and a power curve.

**Why.**
- Candidates are recorded but never scored (brief sections 2.6, 3.8).
- Runner-up headroom: 123 vs 109 of 157 (DeepSeek, realistic) and 33 vs 23 of 36 (watching) (brief section 3.4).
- The region covers 1.6–3.8% of reachable floor (brief section 4.7).
- Min-of-N is not a proper score [verified].

**Payoff.** 0 pp on top-1 by construction. It decides whether the set ideas (I04, I24, I36, I15) are worth anything.

**Better than expected.**
- An anchored random baseline recovers less than a third of the solver's runner-up gain. That would show the runners-up carry knowledge, which licenses P-4 and O-1.
- A fair chance level near the target lets every top-1 result be restated as lift (guess).

**Worse than expected.**
- [Skeptic's rough area estimate, guess] Random-5 inside a 3 m disc may hit 85–95% of group rings before the heading check. That would kill the set story.
- The proper score cannot be defined without human points.

- Freeze a metric note first.
- Re-score the recorded 5 places. Reproduce 134/96/109 and 32/23/23.
- 1,000 draws per record for: unconditional random-5 and farthest-5 in a 3 m disc; anchored baselines (the top place plus 4 random or 4 farthest); two heading rules; lift against the whole room and against the disc.
- I46 statistics: reproduce p = 0.031, 0.0078, 0.00098 and the power table. [re-run] Power at 14 fresh layouts is 0.62 (Qwen), 0.91 (GLM), 1.0 (DeepSeek) (post-hoc Monte Carlo power from the observed layout effects, B = 300; an exact-test resampling gives 0.60 / 0.92 / 1.00; optimistic for a sealed set).
- Success:
  - Exact reproduction.
  - The anchored random-4 recovers at most 50% of the +14 (DeepSeek realistic) and +10 (watching) runner-up gains.
  - The solver beats the anchored farthest-4.
  - Top-1 lift is at least 3× disc chance, and disc chance is at least 2× the whole-room chance.
- Kill (the set story, not the code): the anchored baselines recover at least 80% on both DeepSeek cells, or the unconditional random-5 is at least the solver's any-of-5 for GLM or DeepSeek.
- **Version-0 result (17 Oct; metric note frozen before computing, `phase1/i14_v0/METRIC_NOTE.md`; `phase1/README.md` section 4) [phase 1].**
  - S1 met: all six runner-up cells reproduce exactly. The I46 p values and the old power table were confirmed earlier (`RERUN.md` R05, R34), not recomputed here.
  - K1 and K2 do not fire. Random-5 in the 3 m disc expects 61.5 (GLM) and 63.1 (DeepSeek) hits, against the solver's any-of-5 of 99 and 123. No anchored baseline recovers 80% on both DeepSeek cells.
  - S2 fails. The top place plus 4 random disc cells recovers 96% of DeepSeek's realistic +14 (layout bootstrap about 65–146%) and 33% of the watching +10 (about 4–60%). Post hoc, drawing over the whole room: 57% and 69%.
  - S3 is met on the frozen reading (DeepSeek: 123 vs 113.0 realistic, 33 vs 24.0 watching). GLM's runners-up do worse than random or farthest additions (99 vs 104.0 and 107.0), so S3 fails for GLM on the entry's unqualified wording.
  - S4 fails. Top-1 lift is 4.9–19.9× disc chance in all 6 primary cells, but disc chance is at least 2× whole-room chance only for old GLM (2.06); Qwen and DeepSeek give 1.91–1.92, watching 0.27–0.31.
  - The skeptic's 85–95% comes in at 77–80%: random-5 in the disc lands in the region (heading aside) in about 78% of right-target group records.
  - The 3 m disc is a poor chance region away from groups. It leaves out about 90% of the watching region and 10–17% of the realistic one, and the top place lies outside it in 470 of 790 line records.
  - Other measures: the solver's 5 places put at most 0.019 of their mass in a forbidden kind (random disc places: 0.24 in audience view); 5 cm jitter flips the top-1 verdict in at most 7 records per cell; at 30 deg, DeepSeek's realistic top-1 falls from 109 to 101.

- Frozen conventions: lift denominator; baseline floor (the solver's selectable cells, not the ground-truth grid, to avoid handing oracle people to the baselines); heading rule; k; jitter size.
- An allow-list that excludes the held-out runs.
- The click study is a separate decision (I35).

#### I06 (probe): Crowd variants with zero-call solver curves (covers O-3)

**What.**
- Add crowd variants to dev layouts (seeds 0–2; never group photo or lift), in six families:
  - A: a confuser extending the line, at 6 cue strengths (gap, facing).
  - B: a 3-person cluster at 1.2 m spacing.
  - C: a second group 1.5 m away.
  - D: 0, 2 or 5 bystanders near the structure.
  - E: a person on the line of sight, with visible fraction recorded.
  - F (for RQ1, grounding in clutter): same-kind distractors and partly hidden targets, on development situations only (nobody near, waiting line, conversation group, watching). The rules are written and fixed before any scene is generated:
    - a second same-kind target: a second counter or place of the same kind, or a second person matching the request's description;
    - multi-target `target_ids` where the request does not single one out, and exactly one target where it does (by position, colour or name);
  - A zero-call label check runs before any scene is used. Every family-F scene must pass `check_scene` and score 100% under full oracle. Every target in `target_ids` must have a non-empty, reachable acceptable region. In the single-target variants, exactly one object must match the request under the generator's own attributes. Ambiguous variants are dropped, not relabelled.
  - Family F has no solver curve of its own. It supplies the clutter scenes for I11 Stage 1 (live check L4) and Programme A's RQ1 claim.
- Score the solver with oracle people and reference constraints under:
  - L1: members-only labels, as today.
  - L2: L1 minus cells within a 0.3 m body gap of non-members and minus other groups' circles.
- Repeat with perception noise fitted only on development runs.

**Why.**
- At most 2 bystanders and 1 other group anywhere, none at lines, and factors confounded with situation (brief sections 3.3, 3.8).
- Generator blockers (brief section 4.6), including item 5: no same-kind distractor places or people exist, except the second parcel on the parcel table.
- Silent solver failures: bystanders absorbed (about 4.3 m too far back), people under 1.41 m apart (robot kept 3.2–3.7 m away), and 110–125 deg off-facing in large circles (brief section 3.5).
- HabiCrowd varies density but scores trajectories [verified].
- (inference) Finding 9: on the current sets C's outcome rests on `come_after`. The other kinds add at most one record (Haiku, at lines), and at groups `come_after` alone scores at least C for all six models. These crowd variants are where the other kinds would matter.

**Payoff.**
- 0 pp on the existing sets.
- 0/3/15 pp on a crowd set, through solver fixes the probe exposes (guess).
- Main value: the first **solver-degradation curves** under oracle inputs and simulated perception noise, plus a regression gate.
- These are not RQ2 curves. RQ2 asks how direct VLM goal prediction (R and E) degrades. R and E curves, and C with real VLM perception, need the VLM crowd run (Programme B step 8; about 2,000 calls, guess). The brief's thin tables (sections 3.2–3.3) remain the RQ2 baseline until then.

**Better than expected.**
- The documented failures fire in ordinary generated crowds, each traceable to a named general cause.
- L1 and L2 disagree in under 5% of episodes.
- The noise model predicts C per model within about 5 points, so model runs can target the 2–3 crowd levels where the curves bend.

**Worse than expected.**
- The generator work takes 2+ weeks.
- L1 and L2 disagree often. This is likely, since confusers sit 1–2.5 m from structures.
- The curves only quantify known bugs, and i.i.d. noise is too gentle.

- Build the full pass only if all of:
  - (a) Oracle C drops at least 15 points in at least 2 families, or falls monotonically along family D or A with named causes.
  - (b) At least 2 of the 4 documented failures fire in at least 10% of relevant variants.
  - (d) At least 70% of variants pass `check_scene`, and full oracle scores 100%.
  - (e) Noisy perception separates from oracle by at least 10 points at the top level.
- Kill, if any of:
  - Oracle C at least 95% everywhere, and noise within 5 points.
  - L1/L2 disagreement above 30%, or L2 empty above 10%.
  - More than 50% of layouts rejected.
  - The noise model misses the recorded C by more than 10 points. Then report oracle curves only.
- **Phase-2 result, first probe (18 Oct; protocol frozen and hashed before any variant was built, `phase2/i06/FROZEN.md`; `phase2/i06/README.md` sections 1–7) [phase 2].**
  - Build. 199 of 270 planned variants kept (73.7%; 65 found no spot, 6 failed the visibility check), plus the 30 bases. Every label check passes on all 229 scenes, and full oracle scores 100% under L1. Which levels exist depends on the situation: D5 only at groups (8 of 9; 0 of 21 elsewhere), B and C none at the line of 6 or the audience of 5, and 8 of 36 planned audience variants were kept. E as built never hid a relevant member (visible samples unchanged in 37 of 37), so it tests hidden floor, not occlusion.
  - Oracle C (oracle people, reference constraints) under L1: bases 30 of 30; A 0 of 84, at every level; B1 13 of 21 (drop 38.1); C1 17 of 20 (15.0); D2 25 of 29 (13.8); D5 8 of 8; E1 21 of 21; E2 12 of 16 (25.0). Every failure is at a line: all 46 group and audience variants succeed, and the solver with no constraints also succeeds in every group scene.
  - Documented failures. (i), bystanders behind a line taken as waiting, fires in 99 of 153 relevant line variants (64.7%; without family A, 15 of 69) and explains all 99 failures in A–D. (ii), (iii) and (iv) fire 0 times (0 of 199, 0 of 38, 0 of 8), post hoc because the frozen families keep added people at least 1.2 m from the members and outside the structure's space. E2's 4 line failures have no named cause; post hoc, the added person hides most of the floor at the place in turn, and the solver stops 0.35–0.46 m short, outside the region, with no flag.
  - L1/L2. Oracle C disagrees in 3 of 199 (1.5%; all three E1 at `eval_11`, one per seed, where the standard goal is used), and L2 is never empty. That agreement is built in (the solver keeps 1.0 m, L2 needs 0.85 m): the full oracle's own stop fails L2 in 85 of 199 (42.7%), including all 72 variants of A2–A6.
  - Noise model (fitted in-sample on the seven development runs per model; i.i.d.). Simulated minus recorded C: +0.3 (Qwen), +5.4 (GLM), +6.2 (DeepSeek) points, n 408 each, so all three are valid; against I01's R5 construction the gaps are −1.2, −2.5 and −4.5. Separation from oracle C at the top level (A6, B1, C1, D5, E2; n 80): +2.2, +8.0 and +8.9 points; on the bases +4.2, +18.3 and +17.2. Post hoc, without A6 (where oracle C is 0% and noise can only help): +2.7, +10.7 and +11.5.
  - (inference) The probe confirms one documented failure at a high rate in generated crowds and adds a post hoc hidden-floor mechanism; it did not test crowding inside the region. Under oracle perception, groups and audiences held in every variant built. In this noise model crowds do not amplify perception noise: separations at the crowd levels stay within 2 points of the bases' or below, except at E2. The zero-call curves do not answer RQ2; that still needs the VLM crowd run.
- **Phase-2 result, repair probe: a facing gate on joining a line (18 Oct; protocol frozen and hashed before any repair rule was coded, `phase2/i06_repair/FROZEN.md`; `phase2/i06_repair/README.md`) [phase 2].**
  - Rules, zero calls. Each is a research copy of `sgi.solver._turn`, installed at run time in the probe's own process; the package is unchanged. R0 is the current rule. RF90 (primary): an unlisted person may join only if facing into the line's front half-plane (f · u ≥ 0, with u pointing from that person to the current last member; unknown yaw joins as before). RF60: f · u ≥ cos 60°. RN: no unlisted person joins. The rules read only positions, yaws, the waiting list and the target's number.
  - Crowds (I06's 229 scenes, oracle C, L1). Of the 54 judged line failures (A1–A3, B1, C1, D2), RF90 turns 29 into successes (A1 12 of 12, A2 12 of 12, A3 0 of 15, B1 2 of 8, C1 2 of 3, D2 1 of 4; 27 needed) and breaks nothing anywhere, though here no rule could break a scene: R0 absorbed someone in only 99 scenes, all of them failures, and where R0 absorbs nobody every rule gives R0's place. RF60 fixes 30 and RN 35, also with no break. Failure (i) fires in 60 of 153 relevant variants under RF90 (R0 99). A3–A6 stay at 0 under every rule, RN included: there the confuser stands 0.03–0.44 m from the place in turn, against 0.55–0.94 m at A1–A2. Several B and D confusers face the line's front (f · u up to +0.99), so RF90 fixes 5 of the 15 B–D failures, where RN fixes 11.
  - Development records (I14's six primary cells, 549 goal steps; setting (a), each record's own inputs). RF90 has 0 breaks in every cell and a pooled net of +1 (one DeepSeek flip). But R0 absorbed someone in only 4 of the 549, so the test is thin. With true people (I01's P) or the I09 detector's people (A2's D), RF90 changes nothing in any cell, old or new wording: every person absorbed there faces the line's front (f · u ≥ 0.63), and RF90 keeps them. RN loses up to 5 successes per DeepSeek cell there, so real lines need some absorption.
  - I06's noisy arm (not judged): RF90 raises mean line L1 from 0.380 to 0.518 (Qwen), 0.355 to 0.451 (GLM) and 0.349 to 0.448 (DeepSeek), with 0 per-draw breaks.

- L1, L2 or L3, and the cue threshold, with the mentor.
- A pre-registered crowd held-out design (fresh seeds, a withheld crowd level, a new situation).
- A `[seed, 2]` stream, so existing scenes stay byte-identical.
- Score L2 as "inside L1 and outside the circles". The project has no polygon library.
- A full VLM crowd run (about 22,700 calls) is a big run. A modest run comes first: Programme B step 8, on 2–3 crowd levels. That is about 2,000 calls (guess: 14 layouts × 3 levels × 3 trials × about 5.3 calls per episode shared across R, E and C × 3 models). It is the first run that gives RQ2 curves for R and E.

#### I03: Depth-blob audit of the VLM people list

**What.**
- In numpy: lift the depth and keep points 0.15–2.1 m above the floor. Build a height map on the 5 cm grid, take components, and split them at local maxima. Keep body-sized blobs. The pilot used a top height of 1.2–2.0 m and a footprint of 0.2–0.8 m. Those values may have been set in the same pilot that scanned held-out scenes, so they are not used. Stage A re-derives every threshold from scratch.
- Add unexplained blobs as people with unknown facing, and optionally prune listed people that no blob supports.
- Implement as `AuditedPeople(inner)` behind `PeopleSource`.
- The solver's `_turn` already joins unlisted bodies standing behind the last waiting person.

**Why.**
- GLM finds 0.826 of people (lines 0.774). Its single-person line: found 0.529, C 3 of 18 (brief section 3.4).
- The generator rejects occluded relevant people, so every relevant person is unoccluded by construction (brief section 4.2). Counter staff can still be partly hidden. In the whitelist re-run every miss was a staff member. In the settings that reproduce the pilot's 39 and 37, 4 of the 8 misses are hidden behind a counter-like surface and 4 are fully visible staff whose blob merges with tall structure.
- VLMs miss and invent people (HumanRef) [verified].
- A geometric proposer mattered most in VLMnav [verified].
- Depth clustering for RGB-D people detection is an old method (Munaro & Menegatti 2014) [unchecked: outside the checked landscape].
- [re-run] Counting people inside the image, depth blobs find 34 of 34 on realistic_plain, 36–42 of 42 on real and 34–39 of 42 on staffed (the range spans split rules the pilot did not state; 39 and 37 reproduce under some of them), and 42 of 42 on placeholder dev scenes (eval_00–13 plus one dev seed of eval_14/15), not 41. Every non-staff person is found. Real and staffed hold 56 people each; 14 staff per set stand outside the image. With the pilot's thresholds, false body-sized blobs run about 0.6–2.6 per realistic image.
- (inference) Finding 9: the audit should relocate people more than prune them. 13 of the 20 phantom people on the back wall sit within 25 px of a member who was never found, so they are probably real members lifted to the wrong place (plausible, not proven).
- [phase 1] I01 Step 1 (17 Oct, `phase1/README.md` section 1): under the recorded constraints, removing the detections that lie 3.7–4.6 m from every true person (and their numbers in `come_after`) fixes 15 of GLM's 29 line failures, and the full repair, which also puts the members where they really stand, fixes the same 15. Adding the missed people alone fixes 2 of 29. (inference) For these models the prune-or-relocate half of the audit is the one that pays, and for GLM pruning alone is enough; this qualifies finding 9's line above. Adding unexplained blobs as new people pays little (see the [phase 1] note under Stage B's success bar below).

**Payoff.**
- −1/3.5/7 pp, mostly GLM at lines (guess).

**Better than expected.** [re-run] The facing axis (modulo 180 deg) came within 45 deg for 0.68–0.70 of people on the realistic sets with a torso-band PCA (0.82–0.84 with all blob points), against a chance level of 0.50. Blobs could become the people list itself, which would shorten the people call and give a free recall meter for RQ2.

**Worse than expected.**
- Of GLM's 29 line failures, 7 found nobody, 8 have no `come_after`, and 12 have recall 0. These cannot be fixed offline.
- Pruning may remove real people who were merely placed wrong.
- A gain of +4–8 GLM line records, clustered by layout, may not be significant.
- In crowds, blobs merge.

- Stage B: replay with add-only (primary) and add-plus-prune (stated rule). Tag each failure as fixable offline or not.
- Success:
  - Stage A: recall at least 0.95 at 0.5 m, at most 0.10 false bodies per scene, at most 10% of neighbours merged.
  - Stage B: GLM lines up by at least 5 of 90 (42 to 47+), DeepSeek up by at least 3 (68 to 71+). No model loses more than 1 net elsewhere, and Qwen does not fall. At least half of the fixable failures recovered.
- Kill, if any of:
  - Recall below 0.85, false bodies above 0.3 per scene, or merges above 20% after one general fix.
    - [re-run] With the pilot's thresholds, false body-sized blobs run 0.6–2.6 per realistic image (0 on placeholder dev seeds), above the 0.3 kill line, so Stage A must re-derive them.
  - Add-only gains at most 2 (GLM) and at most 1 (DeepSeek).
  - Losses at least equal to gains.
  - More than 60% of failures unfixable offline, and the optional 10–20-call check gains fewer than 2 of 10.
- **Phase-1 result, Stages A–B (17 Oct; protocol frozen and hashed before scoring, `phase1b/i03/FROZEN.md`; `phase1b/README.md` sections 1–2) [phase 1].**
  - Stage A fit, on fit data only (21 realistic and 8 placeholder worked examples, 260 fresh-seed scenes; 216 settings): keep 0.40–1.90 m, top at least 1.50 m, split radius 0.35 m, footprint 0.30–0.70 m. The one allowed general fix (dropping clip-height cells) was tried on fit data and not adopted: every fixed setting had at least 1.72 false bodies per scene in one fit domain. The grid's ranges were shaped by the summaries of the blob pilot re-run, which had already run on all 68 evaluation scenes (disclosed in the note); the thresholds were still chosen on fit data.
  - Stage A, scored once on the 68 development scenes. Realistic (42 images): recall 105 of 118 = 0.890 (Wilson 95% 0.821–0.934), 0.143 false bodies per scene, 0 of 33 neighbour pairs merged, median xy error 0.077 m. Placeholder (26 scenes): 58 of 58, 0 false, 0 of 11 merged.
  - Every non-staff person is found (line members 64 of 64, group members 60 of 60, others 36 of 36). All 13 misses are counter staff whose blob joins tall structure at the 1.90 m clip (staff recall 3 of 16). The 6 false bodies are two blobs of one layout, repeated in its three arms (post hoc).
  - Stage A verdict: neither. The realistic set misses the recall and false-body success bars and crosses no kill bar; the placeholder set meets every success bar. The margin to the kill line is thin: the realistic recall interval spans 0.85, and the 42 images are 14 layouts × 3 arms with shared staff positions.
  - Like for like, on the same answers (realistic, answers with a people list): GLM recall 0.842 with 0.342 false people per answer, DeepSeek 0.891 / 0.337, Qwen 0.955 / 0.167, Sonnet 0.962 / 0.097, Haiku 0.981 / 0.081, Gemini 0.994 / 0.028; blobs 0.906–0.933 / 0.21–0.25. Blobs beat GLM and DeepSeek on both counts and trail the other four. The blob facing axis (modulo 180 deg) is within 45 deg for 0.686 (realistic) and 0.983 (placeholder), against a chance level of 0.50.
  - Stage B (old wording, 252 records per model; step 0 reproduces all 1,301 realistic solver places, old and new wording). Add-only: GLM lines 42 to 44 (bar 47), DeepSeek 68 to 70 (bar 71), Qwen 234 unchanged, 2 of 25 fixable line failures recovered (bar 13), 4 flips and 1 break over 756. Verdict: neither. Kill 1 misses by one record (DeepSeek's +2 comes from one people answer), so the note's prediction that it would fire did not hold. The flips are I01 R3's, record for record: the solver uses an added body only when it joins the line, which happened in 2 of the 24 GLM line failures that received real missed people.
  - Secondary arms, same bars, all met with no kill: add-plus-prune GLM 55 / DeepSeek 74 lines (18 of 25 fixable recovered); relocate (data-informed: its 40 px radius was registered after I01 on these runs) 57 / 71 (16 of 25); relocate-plus-prune 57 / 74 (18 of 25). The gain is I01's R4 mechanism: the moved detections lie a median 3.5–4.2 m from their blobs. Pruning also drops 34–44 detections per model (summed over solver steps) that are real staff the blobs miss, and it breaks 2 records at the new wording (DeepSeek, Haiku).
  - Post hoc: when a model lists nobody but its constraints name numbers, added people numbered 1, 2, … take those numbers. This caused the one add-only break. A later version should number added people above every number a constraint names.

- Whether pruning is allowed in the method.
- A live confirmation of about 2,400 calls (about 1,700 on goal steps only) is a big run.
- The module lives in `stages/`, under the lint and import rules.
- It must not silently turn a failed people call into a blob list.

#### I11 (Stage 0): Where wrong targets come from, and snap-and-lift

**What.**
- Classify every wrong-target C record (E's target points are the same cached answers in 756 of 756 pairs, so E adds no new targets), using approximate visible masks from ground truth (cylinders and footprints checked against depth):
  - occluder capture;
  - background leak;
  - floor slip;
  - choice error.
- Replay two fixes:
  - snap the point to the nearest oracle visible mask and lift the median depth;
  - snap to the nearest depth component (no ML).
- Measure how well depth components recover target masks.

**Why.**
- Wrong target [re-run]: Qwen 7 (5 in the group of 4), GLM 28 (24 with a point, the brief's figure), DeepSeek 22 (`RERUN.md` R39). The parcel case: GLM 8 of 18, DeepSeek 7 (brief section 3.5).
- The mask median beats the centre pixel, and direct VLM depth error averages 192.0 mm [verified].
- Proposals cap mark-based grounding (75.6 vs 90.1) [verified].
- [re-run] Of 57 wrong-target C records, 53 have a point (4 GLM no_target records have none). Recoverable by an oracle snap-and-lift:
  - person targets: 15 of 15 recoverable by lifting;
  - parcel: 6 of 15;
  - place targets: 5 of 23 (6 of 23 with the package's match mask; 0 of 23 only with a floor-only mask that cannot see the desk or kiosk).
- The evidence rests on 2 layouts: the cluttered group of 4 and the parcel table. No same-kind distractor scenes exist (brief section 4.6, item 5). I06 family F builds them.
- (inference) Finding 9: an oracle snap-and-lift fixes 15 of 15 person wrong targets and recovers 5 of 23 place targets, but the zero-call front-surface snap fixed 8 and broke 37 of 752 C records. So the zero-call rule must be designed, not assumed.
- [phase 1] I01's oracle-target arm (17 Oct): at the new wording, Gemini's remaining loss is the target (+10.3 pp with the oracle target, 0 from any perception repair); on goal steps the oracle target fixes 6 of 6 Qwen, 7 of 14 GLM and 10 of 14 DeepSeek wrong targets at the old wording (`phase1/README.md` section 1).

**Payoff.**
- 0/4/8 pp, mostly GLM and DeepSeek; Qwen +1–2 pp (guess).
- Larger on a distractor set (guess). Family F of the I06 pass would be that set.

**Better than expected.**
- A zero-call, class-agnostic snap-and-lift ships as a general fix.
- If marks are needed, the gateway models probably read them, since the constraints call already uses a numbered image.

**Worse than expected.**
- Depth components merge the occluder with the person behind it.
- Parcel cases are choice errors.
- The gain rests on 2 layouts.

- Success, all of:
  - Lift classes are at least 40% of the GLM and DeepSeek C wrong targets (at least about 18 of 46).
  - Oracle snap-and-lift fixes at least 15 of 46 and breaks at most 2 of about 400 correct records.
  - Replay turns at least 8 of those into full successes.
  - Depth components reach at least 70% of the oracle gain.
- [re-run] Oracle snap-and-lift fixes 20 of 46 GLM + DeepSeek wrong targets (16 full successes), meeting three of the four criteria; depth components were not tested (old wording).
- **Stage 0 result (17 Oct; protocol frozen and hashed before scoring, `phase1b/i11/FROZEN.md`; `phase1b/README.md` section 3) [phase 1].**
  - `rerun/lifting` reproduces line for line. Lift classes are 20 of 46 (43.5%; exactly 40.0% if the 4 no-point records count as non-lift): 9 occluder captures, 11 background leaks, 0 floor slips. Choice errors are 26 of 46 (56.5%).
  - Oracle snap-and-lift, applied to every record with a point: fixes 20 of 46 (every occluder capture and background leak, no choice error), 16 of them full successes, and breaks 0 of 454 right-target records (0 of 699 over all three models; the front-surface snap broke 37). S1–S3 met.
  - Depth components, no ML (V1: 4-neighbour components joined across 3D gaps of at most 0.10 m, snapping only when the lift lands on floor or structure): 9 of 46 fixes, all background leaks, and 0 breaks, so 0.45 of the oracle's gain against the 0.70 bar. S4 not met. No variant reaches 0.70 (always snap 0.40; a step back to the object behind 0.50, data-informed; a depth-relative join 0.45). No kill clause fires. Verdict: neither.
  - Components recover people (12 of 12 person targets at IoU ≥ 0.7), but the parcel always merges with its table (3 of 3; 10 of 10 on fresh seeds), and 14 of 27 place targets merge with people standing against them.
  - V1 broke no right-target record in any set (3,716 records, new wording and placeholder runs included). At the new wording it fixes 12 of GLM's 33 wrong targets. Gemini's 34 are all choice errors on place targets, and even the oracle fixes none of them.
  - Choice errors at 56.5% meet the first half of Stage 1's gate; family F is still needed. The evidence is clustered: the oracle's 20 fixes come from 6 layouts, V1's 9 from 4.
- Kill: lift classes under 20%, fewer than 6 fixes, or more than 5 breaks.
- Stage 1, the marks probe (live check L4), runs only if both hold:
  - Stage 0 shows choice errors of at least 30%, or the parcel failures persist;
  - family F of the I06 pass exists and has passed its zero-call label check.
- Stage 1 then covers the 2 recorded layouts (90 calls, plus 30 optional controls) and about 6 family-F scenes × 3 models × 5 permuted trials (about 90 calls). Without family F, any RQ1 result from Stage 1 is a two-layout case study and must be reported as one.

- Report any change to how `match_target` scores a target separately.
- Fix in advance how an "ask" outcome is scored, if Stage 1 runs.
- A changed plan or locate stage means paired re-runs (a big run).

#### I21: Concurrency, and critical-path latency (covers O-4)

**What.**
- Step A1:
  - Start the people call in a worker thread alongside plan, target lift and context.
  - Lock the Pacer.
  - Join before numbering.
  - Report the critical path alongside the serial sum and provider-only time.
- Step A2 (constraints alongside context) is dropped unless people become fast.

**Why.**
- The people call takes 16.0 s for Qwen and carries no episode text. The package has no concurrency, and the Pacer has no lock (brief sections 3.6, 5d).
- [re-run] Replay:
  - Qwen, three realistic sets: 61.2 to 49.3 s (n = 143; the real set alone, n = 48, gives 62.9 to 50.6 s).
  - GLM: 9.8 to 8.0 s (n = 141; real set 9.8 to 7.7 s).
  - As the benchmark runs, with people answers cached per scene: 53.2 to 47.4 s for Qwen (real set 54.6 to 48.7 s).
  - A2 saves 1.2 s for Qwen (up to 1.65 s on one set) and 0.9 s for GLM at s19; 2.8 s for Qwen at stage 21 and 0.6–3.2 s for the s23 models.

**Payoff.**
- 0 pp by construction.
- About −20% per deployed episode in replay, assuming the gateway overlaps requests fully [re-run]. Live, the saving may be only 5–12% (see "Worse than expected").
- C stays at about 1.8× E (1.77× per goal-step episode, 1.66× at stage 21) for Qwen, in replay [re-run].

**Better than expected.** The gateway overlaps requests fully. The thread-safe pacer also lets evaluation runs go concurrent, giving several-fold throughput for every later study.

**Worse than expected.** One key's requests queue at the gateway. The live saving comes out at 5–12%, with an interval that includes zero.

**First probe.**
- Step 1: offline replay over all non-held C runs, plus the E runs.
- Step 2: implement A1. Identity check: the mock model, plus one cached run (`s19_real_glm53_con`) must be byte-identical with zero cache misses. A pacer stress test with a fake clock.
- Step 3: live check L1 (section 6), 80 calls.
- Success, all of:
  - Offline A1 at most 0.85× serial for Qwen and GLM on every dev set.
    - [re-run] On the three s19 realistic sets, Qwen and GLM reach 0.79–0.83× serial, and s21 Qwen 0.81×. At s23, GLM is 0.89× (18.3 → 16.3 s, n 141), above the bar. Caveat: its people latencies are carried over from s19, under a different load.
  - 100% identical results, with no pacer violations.
  - Live median at least 12% below serial for Qwen. The concurrent people call at most 1.3× its serial time, with no rise in 429 errors.
- Kill: live saving under 5%, or the people call slows more than 1.5×. Then report only the simulated critical path. Any mismatch in replayed records blocks adoption. Drop the claim that concurrency rescues "reduce latency" if C stays above 1.5× E.

- The plan call is launched in `run.py`, so the thread starts there.
- Keep the serial timing fields and add per-call time stamps.

#### I22: Thinking set per stage (covers P-10)

**What.** For Qwen, keep plan and context at low thinking and test people (and constraints) at "none". One shared prompt set; the thinking level is a per-entry API setting.

**Why.**
- Qwen C: 65 of 84 at none (s8, median record time 8.4 s), against 74 at low at 26.2 s (brief section 3.1; the low cell is s12, which is not on the whitelist and was not re-checked). Constraints call 31.7 s and 2,351 tokens; people call 16.0 s and 1,199 tokens (brief section 3.6).
- [re-run]
  - About 73% (people) and 83% (constraints) of Qwen's output tokens in these stages is hidden thinking, estimated from text length because `reasoning_tokens` is null on every Qwen call.
  - On s10 (one trial, 28 records), right-target goal steps succeeded 11 of 11 at none and 11 of 11 at low. Most of the all-none loss came from plan and context.
- Prompt-level reasoning caps do not work [verified].

**Payoff.**
- −6/0/1 pp (Qwen C) (guess).
- People at none saves about 12 s run serially, about 3.5 s once concurrent (guess).
- Constraints at none might bring a Qwen C goal step to about 22 s (guess). That is not yet a comparison with E. E's 28.1 s is at low thinking, and E's goal call (14.5 s at low; brief section 3.6) would shrink under the same per-stage policy.
- A matched comparison sets plan and context at low and every goal-step call at "none", for both methods. Under that policy, E is about 13.6 s of plan and context (28.1 − 14.5) plus its goal call at none, which has never been measured. (Stage 21, with three jobs sharing the pacing, has fresh medians of 7.6 s plan, 6.9 s context, 17.2 s for E's goal call and 35.8 s for constraints. For Qwen its prompts are the same text as before f1a488a, since schema field descriptions never reach the gateway models.) If that call shrank by the same factor C's goal-step calls are guessed to shrink (from about 48 s to about 8 s), E would be near 16 s and still faster than C's 22 s (guess). L2 therefore measures E's goal call at "none" on the same episodes.

**Better than expected.** Constraints at none hold within 2 points, *and* C under the matched policy comes in at or below E under the same policy. Only then does the report's "slower, not faster" line flip. Both figures are guesses until L2 runs.

**Worse than expected.** Constraints at none lose 5–10 points at lines and groups, leaving only people at none. Facing errors rise unseen by the found-share metric.

**First probe.**
- Offline ladder: s10 none vs low; s8 vs s19_ph, re-solved.
- People cross-wire.
- Token attribution.
- Then live check L2: about 74–106 calls, including E's goal call at "none" (section 6 on commits and baselines).
- Gate (offline), all of:
  - None within 2 of low on pooled right-target goal steps (49 of 52 vs 51 of 52 as pooled [re-run], but 11 none-side draws are counted twice. Without them it is 38 of 41 vs 40 of 41, or 37 of 40 vs 40 of 40, depending on which low draw trial 0 is paired with, so the gate sits at its boundary. The s8/s19_ph half also changes the context prompt.)
  - The cross-wire changes at most 1 result.
  - Facing within 10 deg.
  - Tokens at none at most 40% of low.
    - [re-run] On s10 matched calls (12 pairs each), tokens at none are 22.6% (people) and 18.2% (constraints) of low, inside the 40% (the plan call, which I22 keeps at low, is 45.1% on 28 pairs). Caveat: one trial (s10), so only 12 pairs per stage.
- Live success:
  - Found share at least low −0.03; unmatched at most low +0.1 per scene; facing within 10 deg.
  - The re-solve over about 143 records loses at most 3.
  - Constraints at none: at least the low-thinking baseline count −1 on 16 episodes, at a mean of at most 8 s. The baseline uses the same prompt text (stage 21 or earlier records, since schema descriptions do not reach Qwen), at matched load.
  - Report C and E goal-step time under the same per-stage policy. Claim "C faster than E" only if C's matched time is at or below E's matched time.
- Kill:
  - "None" not honoured (tokens above 60% of low).
  - People at none: found share drops by more than 0.05, facing worsens by more than 15 deg, or the re-solve loses 6 or more.
  - Constraints at none: the gate loses 4 or more, or the live check loses 3 or more of 16.

- Whether the headline uses the mixed setting.
- Claude Opus and Sonnet cannot turn thinking off (config notes), so the recipe is per model.
- A full evaluation is about 2,100 calls per configuration.

#### I35 (Part A): Rule-sensitivity envelope (covers P-13)

**What.**
- Rescore all recorded dev R/E/C runs under a pre-registered family of plausible rule variants:
  - inner radius 0.3/0.5/0.85 m; outer radius 1.2/1.5/2.0 m;
  - heading 30/45/60 deg;
  - lines: behind the last person, or near any member outside the corridor;
  - groups: near the target, or near any member;
  - pose-only scoring when the target is wrong.
- Report, per model, the share of plausible variants where C keeps a margin over E (defined below), the smallest margin, and the margin under the mentor's rule.
- Part B, the human click pilot, is in section 4.3.

**Why.**
- Rules are self-authored and coincide with solver shapes at lines, watching and lift. The heading tolerance is 45 deg against the mentor's 30 (brief section 5f).
- 0.5–0.85 m counts both as "in region" and as "intrusion" (brief section 5a).
- The mentor uses the same band with ±30 deg [verified].

**Payoff.** 0/0/2 pp (guess). It makes the C > E claim defensible against the first reviewer objection.

**Better than expected.** C > E holds in at least 90% of variants with a margin of at least +8, and holds under the mentor's rule.

**Worse than expected.** Reassuring but foreseeable. Forking paths if the variant family is chosen after seeing results.

**First probe** (about 5 h for the rescoring; about 2.5 person-days including the new polygon builders; 0 calls). Section 6 books 2.5 days.
- Test: the layout-clustered exact sign-flip test (as in I46), with the layout as the unit. Its p is reported for every variant and model but gates nothing. For Qwen it cannot gate: 0.031 is already the smallest value possible with 6 non-tied layouts, so a variant that ties or flips one layout pushes it above 0.05. A record-level test would pass trivially and contradict section 1.1 item 6.
- Success (margin-based, fixed beforehand): for each model, in at least 80% of the frozen plausible variants, both hold:
  - C − E is at least +5 points pooled;
  - C − E is positive in at least k of the n non-tied layouts, with k = ⌈0.75 n⌉.
- A clean warning also counts as a result: a variant that drops the pooled lead below +5, or the layout count below k, for 2 of 3 models.
- Stop at one table if every variant keeps the pooled lead at +10 or more.

- The template script `photo_rule_variants.py` reads held-out data, so the new script must filter run names, including `s20_held`.
- The variant builders stay on the harness side.

### 4.3 Tier (b): strong bets

- **What.**
  - A sidecar in the separate torch environment runs RT-DETRv2 or D-FINE, plus ViTPose (or RTMW).
  - The project side lifts with depth: position from the torso median, facing from the shoulder-and-hip plane, sign from face keypoints.
  - Flat surfaces are rejected; each person gets a confidence.
- **Why.**
  - The oracle-people gain sits in the weak perceivers (brief section 3.4). Qwen's people call takes 16.0 s (brief section 3.6). VLM facing accuracy is now measured (A0, 17 Oct, [phase 1]): 0.55–0.91 of matched people within 45 deg by model; Qwen, DeepSeek, Haiku and GLM do worst on people seen in profile (Gemini and Sonnet do not).
  - The mentor uses YOLOv10 + RTMW [verified]. MEBOW: 8.393 deg MAE on real crops [verified].
  - A bad detector is worse than none [verified].
  - The Hub metadata of `ustc-community/dfine-large-coco`, `PekingU/rtdetr_v2_r50vd` and `usyd-community/vitpose-base-simple` gives the licence as apache-2.0 [source-checked]. The licence files themselves were not read.
- **Payoff.**
  - −3/3/7 pp (GLM, DeepSeek); Qwen about 0 or slightly negative (guess).
  - −16 s for Qwen's people call, though I21 gets most of that.
- **Better / worse.**
  - Better: near-oracle recall in these clean scenes; continuous facing helps groups.
  - Worse: [re-run] every line member in the realistic eval sets stands 74.5–107.0 deg from facing the robot pose (74.1–107.3 deg from the camera; 16 distinct people), in profile, where shoulder planes are edge-on. Facing flips. Extra bystanders trigger crowd failures.
- **First probe.**
  - A0, the VLM facing error, runs inside I01 at no cost. Done 17 Oct (`phase1/README.md` section 3) [phase 1]. Within 45 deg, on the yaw the pipeline uses: Gemini 0.906 (n 310 matched pairs, 36 people, 8 layouts; 95% CI 0.81–0.98), Sonnet 0.809, Qwen 0.765, DeepSeek 0.616, Haiku 0.616, GLM 0.547. Weaker models lose most on people seen in profile, mainly line members (GLM 0.463, Haiku 0.488).
  - So the facing kill bar below is 0.806 (10 points below Gemini), or the absolute 0.70 if only the school panel counts. Score A1 on the same matched (arm, scene, person) instances (`phase1/a0_facing/a0_pairs.jsonl`), since Gemini's 0.906 comes only from the scenes where it ran a people call.
  - Post hoc: the faced point replaces the model's own 8-bin label for 65–91% of people, and for 5 of 6 models this costs 4–23 points (GLM 0.547 against 0.779 with the label alone). A heading-from-label change would need pre-registering and testing on other sets; the placeholder panel splits 2 to 2 on it.
  - Within I01, true facing alone (R2) flips 0 of GLM's 29 and 0 of DeepSeek's 15 line failures; over all 252 records it nets +1 (GLM) and +2 (DeepSeek: 4 flips, 2 breaks). (inference) On these sets facing errors cost C little, so a detector's value would lie mostly in recall, position and speed.
  - A1 (after the OK, about 6 h): freeze on the example layouts, report once on 42 eval images.
    - [phase 1] I03's people scorer is ready for A1 (17 Oct): `phase1b/i03/people_scorer.py`, `score_people(people, scene, th=None, *, exempt_xy=(), facing='full')`. It calls the package's own matching (`sgi.run._matched_people`) and visibility test, and returns recall, false people, merged neighbour pairs, xy error, full-yaw and axis facing, and misses by role. The rows to compare against, on the same scenes: the depth blobs (`phase1b/i03/stageA/rows.jsonl`; realistic recall 0.890, 0.143 false per image, every miss counter staff, axis facing 0.686) and the six models' recorded people lists (`phase1b/i03/stageA/vlm_rows.jsonl`; realistic recall 0.842–0.994, 0.028–0.342 false per answer; `phase1b/README.md` section 1). (inference) A1 could freeze on I03's fit data (the worked examples and fresh seeds 100–109), so that all three people sources share one split.
  - A2: replay with constraints re-addressed to the detected people.
  - Success: recall at least 0.95 overall and on line members; at most 0.10 false people per image; median position error at most 0.15 m. Facing for non-profile people: median at most 25 deg, at least 85% within 45 deg. Profile front/back correct at least 90%. In A2, GLM and DeepSeek within 3 records of ground-truth people and at least the recorded C; Qwen loses at most 2. Warm runtime at most 1.0 s on the 4060.
  - 0 calls offline. Live: about 30 (up to about 120), then a full run of about 1,400.
  - **Phase-1 result, A1–A2 (17 Oct; protocol frozen and hashed before any evaluation image, `phase1c/i09/FROZEN.md`; `phase1c/README.md` sections 1–4) [phase 1].**
    - Setup. A sidecar in the torch environment (used read-only; ViTPose only, no RTMW) runs the detector and the pose model; the torch-free project side lifts with depth. Fit data as I03: 21 realistic and 8 placeholder worked examples plus 260 fresh-seed scenes (289 images). A 192-setting grid covered the two detectors, score threshold, height band, a flat-surface test, a duplicate radius and a depth push. No setting met the false-people bar in both fit domains, so the stated fallback chose RT-DETRv2 at score 0.25 (band 1.00–2.30 m, flat test off).
    - Facing came from 4 declared variants. Fit data chose shoulders plus hips, with a face rule for the sign and a nose-against-ears rule for edge-on (profile) people; the profile rule is data-informed (its author knew every realistic line member stands side-on). That choice failed on fit data: the face rule turned every back-view person around (143 of 143), because ViTPose reports confident face keypoints on people seen from behind. The one allowed general fix, written and hashed before evaluation, drops the face rule (fit: 36 of 36 and 505 of 513 within 45 deg). Choosing a facing fix over a detection fix is data-informed: its author had counted the view classes of the A0 evaluation pairs.
    - A1, scored once on the 42 realistic images: recall 116 of 118 = 0.983 (Wilson 95% 0.940–0.995; line members 48 of 48; counter staff 14 of 16, against the blobs' 3 of 16), 0.071 false people per image (3 in 42; layout bootstrap 0.000–0.190), 0 of 33 neighbour pairs merged, median position error 0.053 m. Both misses are counter staff with the lower body hidden, boxed but placed 0.62–0.65 m off.
    - Facing, all 116 matched people, none unknown: non-profile 59 of 59 within 45 deg (median 6.4 deg); profile front/back right for 57 of 57 (52 of 57 within 45). On the A0 pairs: 292 of 307 detector-matched Gemini pairs = 0.951, against Gemini's own 0.906 on the same pairs and the 0.806 kill bar; 0.949–0.958 on every other model's pairs, where the models score 0.547–0.807.
    - Warm runtime on the 4060: median 55.6 ms (the verifier re-measured 61.9 ms), p90 about 0.11 s, peak GPU memory 627 MiB.
    - A1 verdict: success; no kill clause fired. Post hoc: without the fix, the facing kill would have fired (0.775 on Gemini's pairs) and the non-profile bar would have failed (41 of 59). Placeholder scenes (reported, not judged): recall 49 of 58 = 0.845, below the 0.90 kill line. All 9 misses are audience members of one situation (`eval_15_audience_5`, dev seeds 0–2), seen side-on or from behind and boxed below the threshold; the blobs found 58 of 58 there.
    - A2 (old wording, 252 records per model; step 0 reproduces all 1,301 realistic places, and I01's P, R3 and R4 reproduce record for record). The primary arm D moves each VLM detection onto the detector person within 0.5 m (keeping its number), drops the unmatched detections and strips their numbers from the constraints, and adds the unmatched detector people numbered above every constraint number. Qwen 236 (+2/−0; bar 232), GLM 202 (+16/−0; bar 199; ground-truth people 202), DeepSeek 199 (+18/−1; bar 190; ground-truth people 193). Lines 88, 57 and 78 of 90 (recorded 86, 42, 68); all 25 fixable line failures recovered. Verdict: success; no kill.
    - GLM's 16 flips are exactly I01 P's, from 10 people answers in 4 layouts. DeepSeek's surplus over P is post hoc traced to position and yaw differences only (D and P differ in 8 records, D wins 7); without its 7 position-and-yaw-only flips against C, DeepSeek still scores 192 (bar 190). Secondary: matching by pixel (a data-informed arm) also passes (236, 203, 199); keeping the unsupported detections instead of dropping them does not (GLM 189); the new wording passes bars taken relative to its own C and P (230, 190, 198), a reading that is an interpretation, since Qwen and GLM would miss the primary bars (232, 199). On the placeholder runs D meets none of the same relative bars (Qwen 139, GLM 117, DeepSeek 123 of 156): GLM and DeepSeek still gain over C (113 and 116), but Qwen loses 8 net (C 147), because D drops real audience members the detector missed.
    - Overall I09: success (A1 and A2 both success; no kill clause fired). Not blind: the note's author knew I01's mechanism, D is in effect I01's P with the detector's people in place of the true people, and D matched the note's prediction for GLM and Qwen.
    - (inference) On these realistic images the detector is the best people source measured: recall level with Gemini's list (0.990 against 0.994 on the same answers) with about three times its false-people rate (0.083 against 0.028), and better facing than every model. Its weak spots are counter staff and placeholder avatars. A2's gain is mostly I01's R4 mechanism (post hoc: dropping a detection a constraint named accounts for 2 of Qwen's 2, 15 of GLM's 16 and 9 of DeepSeek's 18 flips), and D still needed the VLM's list to bind the constraint numbers, so the −16 s people-call saving is not yet shown: the next step is a live check in which the constraints call sees the detector's numbered people. Calling it a method would also need a confirmation on unseen layouts, with nothing re-tuned.
  - **Phase-2 result, unseen-layout confirmation (18 Oct; protocol frozen and hashed before any test scene existed, `phase2/i09/confirm/FROZEN.md`; `phase2/i09/README.md` section 1) [phase 2].**
    - Verdict: neither. Six of seven success bars are met and no kill clause fires. Recall 510 of 520 = 0.981 (Wilson 95% 0.965–0.990; development 0.983); line members 207 of 208; median position error 0.061 m; non-profile facing median 10.3 deg, with 278 of 287 within 45 deg; profile front/back right for 223 of 223; all matched people within 45 deg 494 of 510 = 0.969 (kill line 0.70). False people: 44 in 182 images = 0.242 per image (layout bootstrap 0.176–0.313) against the 0.10 bar. That is about 3.4 times the development rate (0.071) and above the note's predicted 0.03–0.15. It stays under the 0.30 kill line, though the bootstrap's upper end crosses it.
    - Misses: 9 of the 10 are counter staff with the lower body hidden, boxed but placed 0.63–0.65 m off (A1's known failure); the tenth is one merged line member. Post hoc (from truth and rows only; changes no verdict): 29 of the 44 false people relate to counter staff. 16 are displaced staff detections, and 13 lie 0.42–0.61 m from counter staff whom the package's visibility rule calls hidden though their marked point is in the picture. The other 15 are small low-score boxes 3.2–4.5 m from every true person. Without the three staff classes the rate would be 0.082. 12 of the 13 seeds alone exceed 0.10.
    - (inference) Recall, position and facing carry over to unseen layouts of the same rows, and the facing fix holds (back views 76 of 77 within 45 deg). False people do not: the 14 development layouts under-sampled them. So by its own pre-registered standard the detector is not yet a method, and no kill fired. Any change the breakdown suggests (counter-staff placement, the duplicate rule, the threshold, or how the scorer treats people hidden by rule) is data-informed. It would need a new pre-registration and fresh unseen layouts; seeds 200–212 are spent as test data.
  - **Phase-2 status, live check (18 Oct; protocol frozen before the driver existed, `phase2/i09/live/FROZEN.md`; `phase2/i09/README.md` sections 2 and 6) [phase 2].**
    - Pre-registered reading on successes of 84: consistent with A2 if GLM ≥ 66 and DeepSeek ≥ 68 (recorded C t0 64 and 65; A2's D replay 69 and 71); contradicts A2 if GLM ≤ 63 and DeepSeek ≤ 64; otherwise mixed. One trial resolves only large effects (recorded C per trial: GLM 64/65/57, DeepSeek 65/60/57).
  - **Phase-3 use as a confidence signal (18 Oct; zero calls; `phase3/confidence/README.md`) [phase 3].** Comparing the VLM's people with A1's frozen detector people predicts C's failures on the 441 old-wording realistic goal-step records: 1 − F1 between the two lists gives a pooled AUROC of 0.741 (0.869, 0.801 and 0.650 for Qwen, GLM and DeepSeek; reported, not judged), and the shift of C's place under A2's setting D gives 0.708 (judged: neither, fragile). Most of that shift's ranking comes from the 37 records whose verdict D changes, which were known before the design (on the other 404 it gives 0.557; post hoc). The people disagreement keeps more there, 0.694 [0.587, 0.743], but not clearly above the solver's flags (+0.071 [−0.083, +0.182]), and its edge over them is in wrong-target failures (post hoc; `phase3/confidence/README.md` section 10). (inference) As a check, the detector costs about 56 ms and no model call, but it inherits the detector's false people on unseen layouts (0.242 per image in the confirmation), which would raise the disagreement on successes too (untested).
  - **Phase-3 fix and second-confirmation protocol (18 Oct; selection rule frozen and hashed before any candidate was scored, `phase3/i09_fix/FIX_PROTOCOL.md`; second confirmation frozen in `phase3/i09_fix/CONFIRM2_FROZEN.md`; `phase3/i09_fix/README.md`) [phase 3].**
    - Diagnosis: four mechanisms explain all 17 false people and all 12 misses. (1) Occluded torsos: for counter staff, the pose model places the hips behind the counter, the torso hull falls on an espresso machine, a till or the counter end, and the depth median lands 0.76–0.77 m in front of the person (9 false, 8 staff misses). (2) The 0.50 m duplicate radius drops line neighbours lifted 0.45–0.48 m apart (4 misses). (3) Staff in glass booths: the rendered depth treats glass as opaque, so the visibility rule calls them hidden while the picture shows them, and their detections count as false or exempt by chance (4 false, 5 exempt); this is a scorer question. (4) Small boxes on poster pictograms and window figures (4 false). The camera–staff geometry repeats across the seeds of a row, so the seeds are less independent than their count.
    - (inference) On the fit data the fix removes the displaced-staff, line-merge and far-box failures without losing a person, and what remains is mostly the scorer question about glass. But fit-validate shares the fit-train geometry templates, and the second confirmation would test new layouts of the same 14 rows, not new stall geometries. I09 stays "not yet a method" until that confirmation has run.

#### I33: Depth proposes people and target candidates; the VLM only verifies

- **What.**
  - Depth blobs (as in I03) give the people.
  - One montage of numbered crops; per crop the VLM answers "person?" yes or no, and facing (toward, away, left, right).
  - The plan's point snaps to proposals. When candidates are ambiguous, a multiple-choice question.
  - Lift with the mask median.
- **Why.**
  - The same evidence as I03 and I11.
  - [re-run] The facing axis from PCA came within 45 deg for 0.97–0.98 of people on placeholder scenes and 0.68–0.70 on realistic ones (torso-band PCA, 1.0–1.6 m), against a chance level of 0.50 for an axis taken modulo 180 deg (0.52–0.59 with shuffled yaws). The 0.24–0.28 is a 'faces the camera' baseline, not chance, so the realistic gain over chance is about +0.2, not +0.45.
- **Payoff.**
  - −1/4/9 pp (guess).
  - Replaces the 16 s Qwen people call with a short verification call of about 3–6 s (guess).
- **Better / worse.**
  - Worse: 4-way facing replaces 8-way; weak models confirm false positives (a screen on a stand); the right reasoning gets tied to the wrong number.
- **First probe.**
  - Stage A: offline bound under 5 people conditions.
  - Stage B: target-snapping trigger.
  - Stage C: live, 84 calls.
  - Success:
    - Stage A: recall at least 0.95; xy median at most 0.08 m; the perfect-verifier condition within 3 points of oracle people.
      - [re-run] In the whitelist re-run of the pilot, matched blobs have a median xy error of 0.075–0.077 m on the realistic sets, just inside the 0.08 m bar. Caveat: the re-run kept the pilot's thresholds, which may come from the pass that scanned held-out scenes, so Stage A re-derives them (I03, What). [phase 1] I03 Stage A's re-derived detector (thresholds fitted on fit data only, none taken from the pilot) has a median xy error of 0.077 m on the realistic images and 0.067 m on placeholder scenes, still inside this bar (`phase1b/README.md` section 1).
    - Stage B: the trigger catches at least 12 of 15 person wrong-targets, with at most 10% false triggers.
      - [phase 1] Measured on 17 Oct (`phase1b/README.md` section 4; trigger frozen in `phase1b/i11/FROZEN.md`). The trigger fires when the lift lands on floor or structure, or the 5×5 lift window spans two depth components. On the old wording it catches 11 of 15 person wrong targets (all 6 background leaks; 5 of 9 occluder captures, exactly those whose window holds target pixels) and fires on 12 of 201 right-target person records (6.0%). Verdict: neither (the catch bar misses by one; the kill line of 10 is not crossed). The note's author had read the per-record diagnostics and predicted 11, so the blind check is the new wording: 20 of 31 pooled (post hoc pooling), with false triggers of 0–19.4% by model (Gemini 19.4%, Sonnet 12.3%). The 15 come from 3 layouts, 13 at `eval_12`. The trigger flags 11.3% of records with a point, 47% of them wrong targets.
    - Stage C: yes/no at least 95% correct; facing at least 85% within 45 deg; multiple choice at least 7 of 9, with at most 1 of 9 controls flipped; latency at most half the people call.
  - Kill:
    - Recall below 0.90.
      - [phase 1] (inference) I03 Stage A's frozen detector finds 0.890 of visible people on the realistic images (105 of 118; every miss is counter staff) and 1.000 on placeholder scenes. If Stage A here reused it unchanged, this kill line would fire on the realistic set by 0.01 (`phase1b/README.md` section 1). [phase 1] (inference) I09 A1's frozen detector finds 0.983 on the realistic images, with front-and-back facing, but 0.845 on the placeholder scenes. As Stage A's proposer it would clear this line on the realistic set and cross it on the placeholder one; swapping it in would need its own pre-registration (`phase1c/README.md` sections 1 and 6).
    - The perfect-verifier condition more than 5 points below oracle.
    - More than 20 false positives per scene.
    - The trigger catches fewer than 10 of 15.
    - Stage C fails for all models.

#### I18: Falsification search for crowd layouts that break the solver

- **What.** Random search over extras (number K, positions, facing) on dev layouts. Oracle people plus reference constraints. Shrink each failure to a minimal scene (delta debugging), cluster by solver flag, and tag cases where the rule itself is ambiguous.
- **Why.**
  - The silent failures are pinned only by hand tests (brief section 3.5).
  - P-14 says measure before adding rules.
  - The procedural cost map's limits were found by hand [verified].
- **Payoff.** 0/2/8 pp, indirect (guess).
- **Better / worse.**
  - Better: new failure clusters, or a monotone failure envelope by K and spacing, which gives a crowd test that can be fixed in advance.
  - Worse: everything collapses onto the known modes, or most failures are rule-ambiguous.
- **First probe** (about 18 h, after the I06 probe; shares its builder):
  - Success: rule-clear failures in at least 10% of layouts at K ≤ 6, plus at least 2 new clusters (each with at least 5 scenes from at least 3 layouts). Or failures rise monotonically from under 5% at K = 1 to at least 25% at K = 8. At least 20 minimal scenes.
  - Kill: under 3% fail; more than 60% of failures rule-ambiguous; more than 90% fall into documented modes with no envelope; the harness is not working within 2 days.
  - 0 calls. A later stage 2 on rendered minimal scenes is about 500–700 calls.

#### I05: Geometric attention structure gives constraints (also the first probe for bold bet C1)

- **What.**
  - A numpy graph built from people and their facing:
    - chains toward the target give `come_after`, ordered by the chain;
    - mutual cliques give `keep_out_of_shared_space` plus `visible_to`/`face`;
    - shared focus on one thing gives `keep_out_of_view`.
  - Used as a replacement, a gate, or a cross-check.
- **Why.**
  - With true people and no constraints, the solver alone scores 75 of 90 at groups but 0 of 150 at lines (brief section 3.1).
  - GLM states no `come_after` in 8 of 29 line failures (brief section 3.5).
  - The mentor's hand rule with oracle groups scores 64.29% [verified]. GCFF needs tuning per dataset [verified].
  - (inference) Finding 9: `come_after` alone matches C on these sets, so a cheaper route to "who the robot comes after" may carry C's whole gain here. Read this idea's success criteria at lines and on I06's crowd variants.
- **Payoff.**
  - −2/0.5/3 pp (guess).
  - Latency: Qwen C from about 62 s to 28–30 s where the gate fires, or 16–20 s with concurrency (guess).
- **Better / worse.**
  - Better: matches C at lines and watching; disagreement with the VLM works as a free confidence signal.
- **First probe** (about 14 h, 0 calls). Replay with four constraint sources (recorded, graph, none, hybrid), plus oracle people with a reference ceiling, plus a robustness sweep.
  - Success: graph within 3 records of C for at least 2 models and within 2 of 36 on watching, and at least 30 records above solver-alone at lines. The gate covers at least 60% of Qwen goal steps, with the hybrid at least as good as C. Oracle within 3% of reference. 8-bin quantisation costs at most 5 points.

#### I02: Fusing trials and models

- **What.** Offline, from the 3 recorded trials:
  - early fusion (cluster people, vote on constraints);
  - late fusion (sum the per-cell maps, vote on places);
  - fusion across models (GLM plus DeepSeek).
- **Why.**
  - Any-of-3 vs all-3 under C: GLM 78/44, DeepSeek 79/38 (brief section 3.4).
  - [re-run]
    - Majority-of-3 succeeds in 64 (GLM) and 65 (DeepSeek) of 84, against per-trial means of about 62 and 60.7.
    - With 6 GLM+DeepSeek samples, at least 3 succeed in 78 of 84 episodes (ties count; a strict majority, at least 4 of 6, holds in 56).
  - A format-matched majority vote erases selector gains [verified].
  - (inference) Finding 9: the variance between trials sits mostly in the people answer (nearly even for DeepSeek at s23, 11 vs 10), so the fusion target should be people positions first.
- **Payoff.** 0/3/8 pp (guess). Deployment costs k× the calls.
- **Better / worse.**
  - Better: a cross-model vote reaches Qwen-level success from two fast models (guess).
  - Worse: no better than plain majority; union fusion adds false people.
- **First probe** (about 10 h, 0 calls; scripts `bounds.py`, `mixedC.py` and `cross.py` exist):
  - Success: at least 68/84 (GLM) and at least 67/84 (DeepSeek); at least 3 above the place-majority vote; the gain over the per-trial mean is positive in at least ⌈0.75 n⌉ of the n non-tied layouts (the layout-clustered sign-flip p is reported, not gated; no record-level bootstrap); watching no worse; cross-model at least 74/84.
  - [re-run] At s23, GLM + DeepSeek reach 71 of 84 (at least 3 of 6 succeed; strict majority 59), below the 74. Caveat: the s23 people answers are copies of s19/s20, so only the constraints are new draws (old wording: 78).
  - Kill: under +3 over the per-trial mean, or within 2 of the majority vote, or loses more than 2 on watching.

#### I47: Finding the side people use, for the standard goal

- **What.** Near the target, fit up to k planes by RANSAC. Rank them by free floor, face width and visibility, and re-centre the stop on the face. Pure geometry, shared by R, E and C.
- **Why.**
  - The standard goal lands outside the region in 8 (Qwen), 5 (GLM) and 14 (DeepSeek) records (brief section 3.5). For Qwen this ties with grounding as the largest failure class (brief section 3.9). On the oblique `eval_02`, DeepSeek gets 3 of 18.
  - (inference) Finding 9: most of Qwen's 14-record C gap to Sonnet is the standard goal built from its located point at nobody-near scenes (s21, new wording: outside the region in 9 records, unreachable in 1; the 8 above are s19, old wording), which is this idea's target.
- **Correction (skeptic, from the code).**
  - The pitch read `standard_goal_frontal` backwards. True means the surface fit worked. The flag is documented at `types.py:87`, computed at `goal.py:116` (`frontal = goal is not None`), stored at `goal.py:123`, and described at `goal.py:83`.
  - [re-run] The fallback accounts for 3 of 8, 4 of 5 and 6 of 14 misses (13 of 27); 14 misses have the surface fit working, so a fallback-only fix reaches at most 13, below the probe's 17.
  - The 2 GLM "context missed" records (brief section 3.5) are not standard-goal geometry failures, so this idea cannot fix them. The denominator is 27 (8 + 5 + 14), not 29.
  - The reference `people_matter` label is derived with the same `standard_goal()` (`derive.py:285-296`). Decouple them first.
- **Payoff.** 0.4/1.6/3.2 pp (Qwen); DeepSeek 0–5.6 pp, mostly from `eval_02` (guess). It does not change C vs E.
- **First probe** (about 8 h, 0 calls). Variant a must reproduce. Then b1 (robust plane), b2 (b1 plus re-centring), b3 (full ranking). A regression check, and fresh seeds with sampled target pixels.
  - Success: at least 60% recovered (at least 17 of the 27 standard-goal misses under C), at most 1 new failure, fresh-seed gain of at least 15 points at view angles of 30 deg or more.
  - Kill: under 30% recovered; at least 2 regressions per 3 fixes; fresh-seed gain under 5 points. If the misses need the point moved, fold them into I11.

#### I49: Sensor-realism replay

- **What.** Corrupt depth and camera pose:
  - axial noise growing with z²;
  - holes;
  - flying pixels;
  - pitch error of ±1–3 deg and height error of ±5 cm.
  Re-lift the recorded pixel answers, redo the standard goal and the solver, and score against clean ground truth.
- **Why.**
  - Depth is exact and there is no noise model (brief section 5b). The acceptable band is 23, 12 and 7 pixel rows tall at 4, 6 and 8 m (brief section 3.5).
  - VLMnav: SR 50.4% with true depth vs 39.1% with ZoeDepth [verified].
- **Payoff.** 0/1/6 pp (protective; guess).
- **Better / worse.**
  - Worse: a flat curve, worth one robustness paragraph.
- **First probe** (about 8 h, 0 calls). Qwen and GLM, real arm, 5 noise seeds.
  - Success: the replay reproduces at least 99%; a realistic condition moves some model and method by at least 5 points, or halves the C−E gap at lines.
  - Kill: every mild condition stays within 2 points.

#### I19: The VLM states kinds and an anchor; geometry computes the members

- **What.** `expand(kind, anchor, people)`:
  - a chain consistent with facing, for `come_after`;
  - o-space voting, for shared space;
  - ray tests, for views.
- **Why.**
  - GLM members recall is 0.661 (brief section 3.4). Last-in-line is read from distance only (brief section 3.5).
  - [re-run] Most of GLM's member misses (56 of 74 at lines, 76%) are people never found, so expanders mostly cannot recover them (caution: only 17 of the 74 have no detection within 1.0 m; most have a nearby detection the matching gave to a neighbour). Member lists are about 1% of the characters but about 4% (values) to 7% (with keys) of the constraints call's output tokens, so the latency claim stays dropped.
  - (inference) Finding 9: `come_after` alone matches C on these sets, so a cheaper route to "who the robot comes after" may carry C's whole gain here. Read this idea's success criteria at lines and on I06's crowd variants.
- **Payoff.** −1/1.5/5 pp on the existing sets (guess). On crowded lines, +5–20 pp (guess). [phase 2] On I06's crowd variants (oracle people), a one-condition facing gate on the current `_turn` already turns 29 of 54 judged absorption failures into successes with no break (`phase2/i06_repair/README.md`). (inference) So I19's chain consistent with facing should be compared with that gate, not only with `_turn`; what the gate leaves is unlisted people facing the line's front near its end.
- **Better / worse.**
  - Better: success stays flat from N = 3 to 15 on synthetic crowds, and one mechanism fixes three silent failures.
  - Worse: o-space voting is fragile with 8-bin facing; bent lines break the chain; circular with the line rule.
- **First probe** (about 16 h, 0 calls). A synthetic crowd harness on `tests/solver_layouts.py`, plus a replay.
  - Success: last-in-line accuracy with bystanders at the tail at least 15 points above the current `_turn`; group F1 at least 0.85; success at N ≥ 10 at least 10 points higher; replay loses at most 2 per model.
  - Kill: F1 below 0.75; within 5 points of `_turn`; N ≥ 10 gain under 5 points.

#### I12: Monte Carlo over perception hypotheses

- **What.** Sample K = 32 variants of the perceived people (drops, jitter, facing noise, phantom people at unexplained blobs). Solve each, and return the place most often accepted, a density, and an agreement-based confidence.
- **Why.**
  - Near-ties flip with centimetres (brief section 3.5). There is runner-up headroom (brief section 3.4) and no scalar score (brief section 2.5).
  - Nothing found propagates perception uncertainty into a goal solver (searcher note, unchecked).
- **Payoff.** −1/1.5/4 pp (guess); the main value is the confidence and the density.
- **Better / worse.**
  - Better: phantoms (from I03's blobs) move the robot back when a person may be missing.
  - Worse: the robust pose equals the top place in 90–95% of records. Independent drops push the robot forward, the wrong way. The confidence may detect the situation, not the error.
- **First probe** (about 8 h, 0 calls; a rider on I03 and I14).
  - Success: net at least +8 of 298 (GLM + DeepSeek), fixed:broken at least 3:1, and the net gain positive in at least ⌈0.75 n⌉ of the n layouts where it is non-zero (the layout-clustered sign-flip p is reported, not gated); Qwen loses at most 2; drift at most 0.15 m; AUROC at least 0.72.
  - Kill: net gain of +2 or less; more than 3 Qwen records lost; drift above 0.25 m; AUROC below 0.65.
  - [phase 3] A one-hypothesis version has been measured (`phase3/confidence/README.md`): the shift of C's place when the solver re-runs with I09's detector people in place of the VLM's (A2's setting D) predicts C's failure with a pooled AUROC of 0.708 on the 441 old-wording realistic goal-step records (0.645, 0.738 and 0.688 by model; its own pre-registered verdict: neither). That sits between this entry's kill line (0.65) and success bar (0.72), on a simpler test with no sampling and no net-gain criteria. On the 404 records where D does not change the verdict it falls to 0.557 (post hoc, outcome-dependent subset). (inference) A K-sample version would have to show its value on those records.

#### I23: Fast-slow cascade

- **What.** Solve with no constraints. Make the constraints call only if geometric conflict features fire: a view sector, someone between the robot and the target, a near-tie, people near the target.
- **Why.** The panel's solver-alone result with true people was 75 at groups, 131 nobody near, 0 at lines (brief section 3.1). No solver-alone run existed on the realistic sets until step 0 (brief section 3.8); the replay now gives it offline for all six models (`i01/README.md`, Table 1a: 0 of 90 at lines, at least C at groups). Dual-process gating [verified].
- **Payoff.** −3/0.5/3 pp (guess). About −12 s per Qwen goal step if every group episode takes the fast path (guess).
- **Better / worse.**
  - Better: 35–45% of episodes take the fast path at a loss of at most 1%.
  - Worse: the gate degenerates into "the target is a person" (a hidden situation detector).
- **First probe** (about 8 h, 0 calls). Its first table is I01's R0. Freeze the gate on Qwen/GLM s19; test on DeepSeek s20 and watching.
  - Success: at least 30% fast, at most 1% loss, at most 5% of line episodes fast.
  - Kill: fast-path groups more than 5 points below C for 2 of 3 models; no gate reaches 15% fast within 1% loss; AUROC below 0.70.

#### I39: Model-escalation cascade (GLM to Qwen)

- **What.** Run GLM C. Read uncertainty signals that need no ground truth (disagreement across samples, solver ambiguity, unexplained blobs). Re-run uncertain episodes with Qwen.
- **Why.**
  - GLM C: 11.6 s, 74%. Qwen C: 61.6 s, 93% (brief sections 3.1, 3.6).
  - CUSP: ensemble uncertainty detects errors (collective uncertainty, AUROC 0.764, small models; JSD disagreement, AUROC 0.819, commercial models) [verified]. Argus: confidence signals do not transfer across model families [verified].
  - [re-run]
    - Qwen succeeds on 42 of GLM's 45 goal-step failures.
    - An oracle gate gives 228 of 252 at 30% escalation.
    - Disagreement with the other two trials (three samples) reaches AUROC 0.73 and 215–216 of 252 at 50% escalation, depending on a tie. Disagreement between two samples reaches AUROC 0.66–0.75, depending on which second trial is drawn, and 215–218 of 252. Random escalation gives 203.4 on average (201 for one seed).
    - The pitched version that reuses GLM's plan tops out near 207.
    - [phase 3] Phase 3's S4 (the largest place distance among an episode's three C trials, infinite when a trial has no place or their targets lie more than 1 m apart) gives AUROC 0.715 on old GLM, 0.840 on old Qwen and 0.669 on old DeepSeek, and 0.863 and 0.877 on s23 Haiku and Sonnet (`phase3/confidence/README.md`, reported, not judged). It is the only signal there that sees wrong targets well (0.841 pooled).
- **Payoff.** 3/7/12 pp over GLM C, at about 40–55% of Qwen's latency (guess).
- **First probe** (about 5 h, 0 calls).
  - Success: at least 218 of 252 (Qwen's plan run concurrently) or at least 210 (GLM's plan reused); goal step at most 31 s; at least +10 over random; AUROC at least 0.70.
  - [re-run] From s23 GLM (170 of 252), the strong model alone beats every cascade, the oracle gate included: Qwen 228 against at most 217, Sonnet 242 against 222, Gemini 214 against 205. GLM's 27 failures outside the goal step never escalate. Caveat: latencies there are not load-matched.
  - Kill: AUROC below 0.65, or less than +5 over random.

#### I34: Pointing on a metric top-down map (R-BEV)

- **What.** A second image: a top-down render at 2.5 cm per pixel, with people as discs with facing arrows and numbers, the target, and the robot's footprint. The VLM marks a point on it, and the point maps back to the floor exactly.
- **Why.**
  - The acceptable band is 23, 12 and 7 pixel rows tall at 4, 6 and 8 m. Failed E points at lines (s16b/s20) lie a median 0.34, 0.88 and 0.35 m from the region (R: 0.51, 0.60 and 0.74 m). Pointing aids drawn in the camera view did not help (brief section 3.5).
  - BEACON: points marked in the image land inside static structure (SIR) 57.25% of the time for GPT-4o, vs 2.13% for its trained top-down output [partly].
  - [re-run] Failures of E at groups are mostly near misses that land on a body (28 of 44: within 0.3 m of the region and within 0.3 m of a body disc; 25 of 44 overlap a body; Qwen 3 of 5, GLM 13 of 16, DeepSeek 12 of 23).
- **Payoff.** −10/−3/3 pp against C (guess). +3–20 pp against E (guess). Qwen latency about 45–50 s (guess).
- **First probe.**
  - Stage 0, 0 calls: a snap control (move recorded E points off bodies onto free floor), a geometry figure, and a dry run of the renderer.
  - Stage 1: 120 calls on 6 placeholder episodes.
  - Gate: snapping fixes under 60% of near misses, or far misses at lines are at least 15%.
  - Success: at least 15 points above E and above snapped E for GLM and DeepSeek, and within 7 points of C.
  - Kill: snapping already recovers at least 70% of near misses. Then report the snap fix and drop R-BEV.

#### I04: Soft solver density (covers O-1, P-3)

- **What.** Turn the per-cell lexicographic keys into an energy with a temperature, giving a density over cells (plus a von Mises distribution over heading). Top-1 is set to the lexicographic pick explicitly, because the banded keys mean the soft argmax can differ.
- **Why.** No scalar score (brief section 2.5); runner-up headroom (brief section 3.4). VoxPoser's weighted sum [verified]. RoboMAP reduces its map to the argmax point [verified]. [phase 1] I14 version 0 (17 Oct) finds that the top place plus 4 random disc cells recovers 96% of DeepSeek's realistic runner-up gain (+14), though only 33% of its watching +10, where the 3 m disc is a poor chance region. (inference) So runner-up headroom alone does not justify I04; its case rests on producing a density and a confidence signal.
- **Payoff.** 0 pp on top-1 by design.
- **Better / worse.**
  - Better: beats a Gaussian blob around top-1; confidence AUROC at least 0.75.
  - Worse: the temperature collapses (all mass on one cell) or spreads (trivial lift).
- **First probe** (about 8 h, 0 calls). Variants V1 (full soft), V2 (exclusions kept hard), V3 (Gaussian control); leave one layout out. V1 and V3 are offered as an optional phase-2 add-on (section 6, step 7) once I14's metric note is frozen. They share I14's harness, and they would be the first distribution the project actually produces.
  - Success: lift at least 10× and beats V3; hit@5 within 2 of the lexicographic hit@5; AUROC at least 0.70 for 2 models.
  - Kill: does not beat V3; more than 90% of the mass within 0.3 m of the top place in over 80% of records.
  - **Phase-3 follow-up on confidence (18 Oct; protocol frozen and hashed before any signal was computed against an outcome, `phase3/confidence/FROZEN.md`; `phase3/confidence/README.md`) [phase 3].** On the 441 old-wording realistic goal-step records, signals from perception disagreement rank C's failures far better than the density's concentration (pooled AUROC 0.459 there): the judged signal P1, the shift of C's place when the solver re-runs with I09's detector people (A2's setting D), gives 0.645, 0.738 and 0.688 by model and 0.708 pooled, a pre-registered neither (2 of 3 cells at 0.70 needed; flags 0.595; kill line 0.60). Reported, not judged: the people disagreement with the detector (1 − F1) 0.741, trial disagreement 0.774 (two extra pipeline runs), and a zero-call combination 0.752, which keeps 0.878 success at 80% coverage against a base of 0.769. Post hoc, on the 404 records whose verdict A2's setting D does not change, P1 falls to 0.557 and the zero-call combination to 0.635, while the people disagreement keeps 0.694, not clearly above the flags' 0.622 there (section 10 of that page). (inference) The confidence that a soft density lacked comes, on these records, from comparing perception sources. It is data-informed (designed knowing A2's results), P1's verdict is fragile, and for the new-wording API models, whose failures are not perception, the perception signals carry little (P1 0.197–0.562).

#### I08: Solver-verified worked examples for the constraints step (covers P-5)

- **What.** An offline search for small constraint sets that the solver verifies on example episodes. These become worked examples, and the search also shows what the six kinds cannot express. Then a small probe with examples given as text only.
- **Why.**
  - Examples lifted Qwen's marked points from 69% to 85% (brief section 3.1), but none exist for the constraints step (brief section 3.8). GLM writes 0.27 unusable constraints per record (brief section 3.4).
  - Optimising exemplars beats optimising instructions [verified]. Demonstration text carries the signal [verified].
- **Payoff.** −1/3/8 pp for GLM and DeepSeek (guess).
- **First probe** (0 calls). Part A is the oracle-constraint ceiling; run it once, shared with I01's R5. [phase 1] R5 has run (17 Oct, `phase1/README.md` section 1; circular). With each model's own people, reference constraints add +7.9 (GLM) and +9.5 pp (DeepSeek) of 252 over recorded C at the old wording, and +9.9 and +9.5 at the new. With true people on both sides (P5 against P), the old-wording gap is +5.6 and +7.9. So the kill does not fire; on the own-people reading the 8-point bar is met for DeepSeek, and for GLM only at the new wording. The share from reasoning errors is not yet measured. Part B is group coverage.
  - Success: a gap between reference and recorded constraints of at least 8 points for GLM or DeepSeek, with at least half of it from reasoning errors; robust sets for at least 80% of examples. Then a 40–60-call probe.
  - Kill: gap under 4 points for both models.

#### I24: Choose, don't write

- **What.** The solver enumerates constraint hypotheses over clusters of people, solves each, and de-duplicates to at most 6 numbered stops. The VLM picks one.
- **Why.**
  - Qwen's constraints output is 2,351 tokens (brief section 3.6).
  - LocateBench: picking a letter beats coordinates, except for Claude-3 Opus [verified]. A random verifier recovers most of the gain [verified].
  - (inference) Finding 9: `come_after` alone matches C on these sets, so a cheaper route to "who the robot comes after" may carry C's whole gain here. Read this idea's success criteria at lines and on I06's crowd variants.
- **Payoff.** −4/1/6 pp (guess). Qwen goal step about 25–35 s (guess).
- **First probe** (about 8 h, 0 calls). An offline coverage study.
  - Success: coverage@6 at least 10 percentage points above C's top-1 share of goal-step records, and above any-of-5. That means DeepSeek at least 125 of 157 (top-1 109, any-of-5 123) and GLM at least 111 of 141 (top-1 96, any-of-5 99). Qwen at least 134 of 143; median k at most 5.
  - Kill: coverage no better than any-of-5. If a model-free consensus pick comes within 3 points of coverage, keep the consensus and drop the VLM call.

#### I51: Real images and a benchmark swap with the mentor's system

- **What.**
  - Ask the mentor for the 42 scenarios (with depth, free space and relevant members) and for the predictor's outputs on our scene files.
  - Zero-call part: rescore recorded runs under the mentor's written rule. This overlaps I35.
  - Fallback: 15–25 staged RGB-D scenes with rules fixed before capture.
- **Why.**
  - Every number so far is simulated and scored by self-written rules (brief section 3.8).
  - The mentor's system: 89.76% vs 46–60% for prompted GPT-5.5 and Gemini variants; Qwen3.5 with thinking scored 39.52% and Molmo2 10.71% [verified].
  - The arXiv abstract page lists no code, project page or data release [source-checked]. The PDF was not checked for a link.
- **Payoff.** 0 pp on our sets; new numbers on new sets.

#### I46 (sealed bank): New situations written blind, plus fresh seeds

- **What.**
  - Someone other than the designer (the teammate or the mentor) writes 2–4 new situations as a new generator pass, sealed under a hash.
  - One `--final-test` run.
- **Why.** The held-out sets are spent (brief section 4.5), and rules are self-authored.
- **Payoff.** 0 pp. It enables a clean RQ3 claim of generalisation.
- **First probe.** The fresh seeds can be generated now, without a blind author.
  - Success: a protocol file plus fresh seeds at 100% under full oracle, committed before any method freeze; a blind author commits at least 2 situations by about 2026-10-31.
  - Kill: no blind author by then; more than 40 layouts needed per situation.
  - The final run is about 6,000 calls.
- **Statistics half (17 Oct; `phase1/i46_stats/`, `phase1/README.md` section 5) [phase 1].**
  - Success is clustered by layout: ICC 0.08–0.75 over 27 model × method cells, design effect 2.4–13.7, so 252 records carry the information of 18–107 independent ones. C is the least clustered method by median.
  - At the new wording, C beats E and R for all six models. Every t(13) interval on the 14 layout differences excludes zero (C − E from +32, Qwen, to +92, Haiku, of 252). No E − R difference is significant.
  - Power of a sealed set of fresh layouts (two-sided exact sign-flip by layout; post-hoc and optimistic): 0.98–1.00 at 20 layouts for C − E and C − R, all six models; 0.79–0.97 at 14; 0.14–0.41 at 8. Plan for at least 20. [phase 3] (inference) That figure resampled the 14 distinct rows' layout differences, so the 20 should be distinct layouts too, not new seeds of the same rows: new seeds of a row repeat much of its geometry (in the waiting-line rows the staff member's projected position is identical across seeds; `phase3/i09_fix/README.md` section 1). The draft metric list v1's D14 (not agreed) would count only a distinct template as a fresh layout.
  - The kill line "more than 40 layouts needed per situation" does not fire: 0 of 24 lines and groups cells need more than 40 (6, or 14 where a layout is tied or negative). That design resamples 3–5 values and is degenerate, and 6 per situation is the test's floor, not a powered size.
  - The one-page reporting note is `phase1/i46_stats/HOW_TO_REPORT.md` (with the verifier's two fixes applied: R and E are only nearly identical at nobody-near, since DeepSeek differs in 3 of 108; and 6 layouts per situation is a floor).

#### I35 (Part B): Human click pilot

- **What.** 5–8 naive raters click where the robot should stop on 14 realistic dev renders and 6 watching scenes. Clicks are lifted with depth.
- **Why.**
  - The mentor's human check is a 15-person KL pilot [verified]. That it is the only one is our inference. SocNav1 shows moderate agreement (kappa 0.61–0.71) [verified].
  - [Steelman, guess] Raters may put group joiners outside the ring.
- **First probe.**
  - Success: at least 70% of clicks inside the polygon at lines and watching; the human mode inside the polygon in at least 11 of 14 layouts; median click-to-centroid distance at most 0.6 m at lines. A clean warning (under 50% of clicks inside the group ring) also counts.
  - Kill: median distance above 1.0 m; fewer than 5 raters within a week; ethics review cannot finish by about week 3.

### 4.4 Tier (c): bold bets (flagged)

#### C1 [BOLD] I32: Attention edges replace the constraints call (I20 is the probabilistic variant)

- **What.**
  - For each person, "attends to what" plus "waiting for a turn", from the people call or from geometry alone.
  - One fixed objective: own space; do not block anyone's line of attention; follow the turn order along chains; be visible to the needed person. It feeds the unchanged lexicographic solver.
  - In the zero-call version, edges come from the recorded people's geometry, the same experiment as I05's replay.
- **Why.**
  - The 31.7 s, 2,351-token constraints call (brief section 3.6); unusable constraints (brief section 3.4); groups needed no constraints with true people (brief section 3.1); the silent failures (brief section 3.5).
  - The mentor's three attention topologies [verified]. G2-Nav: a few numbers per object [verified].
  - **Note.** The pitch's "wait elsewhere" motivation came from held-out picture records (LEDGER:479). It cannot motivate the method and is set aside here. The case rests on development evidence, and generality needs a new pre-registered situation.
  - (inference) Finding 9: `come_after` alone matches C on these sets, so a cheaper route to "who the robot comes after" may carry C's whole gain here. Read this idea's success criteria at lines and on I06's crowd variants.
- **Payoff.**
  - −4/1/5 pp (guess).
  - Qwen about 33–38 s with a richer people call (guess). The zero-call geometric version is about the length of the people call alone (guess).
- **What would make it pay off.** Geometric edges come within 3 points of C pooled, and at least match C at groups. Then the constraints call can go with no prompt change. Stress layouts fixed at least 16 of 20.
- **First probe** (0 calls).
  - Stage A: oracle edges against reference constraints, plus about 20 stress layouts.
  - Stage B: geometric edges from recorded people.
  - Success: Stage A at most one record lower per situation (placeholder) or within 2 points (realistic); Stage B pooled within 3 points, lines within 5.

#### C2 [BOLD] I27: Masked-person prior and real joiners

- **What.**
  - Learn "where would one more person stand" from generator layouts (source A) and from real newcomers in CongreG8 (source B).
  - Use it as a preference key below the exclusions.
  - Probe 1: score CongreG8's final newcomer positions against our conversation-group ring. This would be the first human-grounded check of a reference rule.
- **Why.**
  - Groups: Qwen R/E/C 39/44/48 of 54, DeepSeek C 34 (brief section 3.2). 110–125 deg off-facing in large circles (brief section 3.5). Circularity (brief section 5f).
  - CongreG8: 380 human trials, groups of three or four [verified, partly].
  - The data are on Zenodo, record 4537811, under CC BY 4.0 [source-checked]. The human CSV zip is 1.7 GB, the robot CSV zip 133.1 MB, and the whole record (with FBX files and labels) 2.3 GB [source-checked].
- **Payoff.** 0/1/4 pp (guess).
- **What would make it pay off.**
  - At least 60% of real joiners inside the ring and at least 80% facing within 45 deg. That would be human support from about 380 trials.
  - Or a systematic offset, which would be data behind changing the 1.0 m radius.
  - A real-joiner KDE that beats band-uniform and loses nothing to the solver alone.
- **Worse.**
  - Real joiners stand inside the 0.85 m intrusion line. That is still a finding.
- **First probe** (0 calls).
  - Step 0, 1 h: count configurations.
  - Step 1, 6–8 h: the CongreG8 ring check.
  - Step 2, 6 h: KDE vs baselines, plus a dev replay.
  - Kill: Source A has fewer than about 30 configurations; KDE gains under 0.1 nats; parsing takes more than 8 h. Keep Step 1 whatever happens.

#### C3 [BOLD] I13: A learned relation parser, tested by leaving one situation out (I07 and I31 are the heavier variants)

- **What.** A tiny set network trained on geometry-only generator layouts, with perception noise. It predicts relations (same structure, ahead of, attends to the target). A fixed table turns relations into the six kinds, and the unchanged `solve()` runs.
- **Why.**
  - Latency (brief section 3.6); group structure is largely geometric (brief section 3.1).
  - The mentor's paper does not discuss out-of-distribution generalisation [partly]. That its model was tested only in distribution is our inference. The procedural cost map failed on bent queues [verified]. Difficulty-filtered SFT is a strong control [verified].
- **Payoff.** −6/4/11 pp for GLM and DeepSeek with a detector (guess). Goal step about 0.1–2 s (guess).
- **What would make it pay off.** A relation table that names no situation reproduces the reference constraints in at least 95% of episodes. Hand-coded geometric rules under the measured noise keep at least 80% of the success and beat the recorded GLM/DeepSeek C. Leaving watching out of training costs at most 10 points.
- **Worse.** Leaving one situation out is partly ill-posed, because each relation type lives in one situation. Labels are circular. A logistic regression may match the network.
- **First probe** (about 5 h, 0 calls).
  - Step A: table ceiling.
  - Step B: hand-coded extractor under noise, plus leave-one-situation-out.
  - Step C: logistic regression, only if A and B pass.
  - Kill: the table needs the situation label in more than 10% of episodes; more than 10 points below C with true people; drop above 25 points when a situation is left out.

#### C4 [BOLD] I25: Constraints as a standing program, re-bound to fresh detections

- **What.**
  - After one constraints answer, match new detections to the old people, rewrite the constraint arguments, and re-solve in about 0.1 s.
  - A staleness trigger (matching cost, unmatched people, infeasibility) calls the VLM again.
  - Evaluated on generated scene pairs (before and after).
- **Why.**
  - The solver takes 0.1 s, against 30–60 s for calls (brief section 3.6). The proposal's week-7 tracking milestone is unbuilt (brief section 3.8).
  - ReKep re-solves at about 10 Hz [verified]. G2-Nav keeps 82% of the benefit with 4 s-stale scores [verified]. The mentor keeps a goal if it stays within 0.4 m of the top-5 new candidates [verified].
- **Payoff.**
  - On a new paired set: 5/20/40 pp against keeping the stale pose (guess). This baseline differs from every other row's (current C), so the numbers are not comparable with the rest of the table.
  - 0 to −10 pp against asking again, at 100–300× lower update cost (guess).
- **What would make it pay off.**
  - Part A: re-binding recorded constraints to better people lifts line success by at least 5 points. That would make it a tool for swapping the people source.
  - Part B: at least 90% of the success of fresh constraints; trigger recall at least 75%.
- **Worse.** A flat curve by construction (the band is wide, and `_turn` absorbs people joining at the end). Role changes slip past the trigger.
- **First probe** (about 18 h, 0 calls).
  - Kill: within 5 points of the stale pose; under 70% of fresh in 2 types; noise costs more than 15 points.

- **What.**
  - The VLM still reads the request and grounds the target. Plan, locate and context stay unchanged.
  - A conditional flow-matching (rectified-flow) model samples goal poses (x, y, cos θ, sin θ). It is conditioned on a top-down raster around the target (inference: 10 cm cells), with channels for:
    - free floor;
    - people as discs with a facing mark;
    - the target's footprint and front;
    - the robot's footprint.
  - The solver checks every sample against its exclusions, as a projection step. The top pose is the valid sample of highest density. If no sample survives, the solver's own top pose is used. The sample set is the distribution output.
  - I07 (a learned top-down energy) and I28 (energies per constraint kind) stay parked. C5 is the sampler version of I07, with the solver as its projection.
- **Why.**
  - No paper was found with a generative model that outputs a stop pose from a request plus RGB-D (section 3.5, open; searcher note, unchecked).
  - SoGuDiff drops from 99% to 70% without its projection layer [verified]. Here the solver plays that role.
  - The mentor's energy map shows that a learned goal head can work without a VLM in the loop (section 3.1) [verified].
  - Speed: sampling should take milliseconds on the 4060 (guess), against Qwen's constraints call of 31.7 s (brief section 3.6). Stage 23's other models take a median 1.0–6.4 s a call per stage, except GLM's constraints call at 8.6 s (section 1.1, finding 8).
- **Payoff (lead researcher's guess).**
  - Top-1 against C with the same people: −10/0/+3 pp.
  - The goal step after the people source drops from tens of seconds to under 1 s.
  - A native multimodal density for the I14 scoreboard, for example mass on several gaps in a group.
  - p_disappoint about 0.8.
- **What would make it pay off.**
  - With each model's recorded VLM people, it comes within 3 points of C pooled for 2 of 3 gateway models.
  - It beats I04's soft density on lift and hit@5.
  - It holds on fresh seeds of the training situations.
- **Worse than expected.**
  - It memorises the generator.
  - It learns the reference rules back. Its labels are those rules, so a "success" measures speed and smoothness, not social correctness. Only human data (CongreG8, C2) could make it more correct than the rules.
  - It fails on crowd variants outside its training (I06).
  - Its samples collapse to one mode.
  - Leaving one situation out is partly ill-posed (as for C3): each situation has its own rule.
- **Gates before any training** (all zero-call, about 1–2 days):
  2. A non-neural conditional density (a KDE over reference-region cells relative to the target and the people) transfers, in at least 2 of 3 leave-one-situation-out folds, against a band-uniform baseline. [phase 1] Met (`phase1d/README.md` section 2). On fresh seeds 100–109, a ratio KDE over target- and people-relative features beats the band-uniform baseline by at least 0.10 nats per region cell in 2 of 3 folds: waiting_line +1.955 (n 140 episodes), conversation_group −0.781 (n 100), audience +4.113 (n 60). The verifier's three other training subsamples give +2.84 to +2.91 and +3.79 to +4.26 for the passing folds; 2 of 3 pass in every draw. The band covers only 29% of the audience region (post hoc), and two of the chosen features mirror the rules' geometry.
  3. I04's soft density (V1, with V3 as its control) exists on the I14 harness as the baseline to beat. [phase 1] Met (judged in `phase1d/README.md` section 2): V1 and V3 were built on I14 version 0's records, floors and scorer replica, and an independent verifier reproduced them. V1 is a kill against V3, so V3, a near-point mass on the solver's top place, is the baseline to beat.
  - [phase 1] Outcome: gate 1 fails, so the gates do not pass and, by this entry's rule, the probe may not start. The remedy the entry names is to widen the generator first with I06's crowd builder; the recount above meets the count by the letter, but on development seeds and with no new situation or line geometry.

### 4.5 Tier (d): parked

| id | Title (brief direction) | Why parked | What would revive it |
|---|---|---|---|
| I10 | Geometric gate in place of the context call | Little accuracy headroom (context misses 1/3/1, unneeded goal steps 0/4/14; brief section 3.5). With the VLM people call, Qwen's nobody-near episodes get slower (16 vs 12.4 s). Agreement with the reference is circular. | I03's blobs pass Stage A (I03 Stage A, 17 Oct [phase 1]: not met; neither success nor kill, realistic recall 0.890 with 13 of 16 staff missed) |
| I48 | Re-ask when the solver flags the answer | [re-run] On DeepSeek's realistic set, flags fire on 19 successes (10.4%) and 10 failures (14.3%); 25 vs 14 is flag firings pooled over all seven s20 DeepSeek C runs. For GLM the flags do separate failures (5.4% of successes against 30.3% of failures). Self-check record: 4 fixed, 11 broken (brief section 3.5). | The flag table (a zero-call rider on the test harness) separates failures, and I01 finds failures a constraint change can fix |
| I36 | Diverse slate plus pairwise re-ranker (P-4) | Ceiling: Qwen 0, GLM 3, DeepSeek 14 (brief section 3.4). Self-check record. | I14 shows the runner-up headroom is knowledge, not spread (I14 version 0, 17 Oct [phase 1]: not shown; random additions recover 96% of DeepSeek's realistic gain) |
| I41 | Numbered free-floor candidates (P-1, P-2) | Errors are reasoning (28% cut in at the front), not precision. Coverage trades off against legibility. Overlaps I24/I36. | The I24 coverage study and the I34 snap control show a large near-miss share |
| I20 | Posterior over attention graphs | The probabilistic form of C1. Perception is the bottleneck. 2–3 weeks of MCMC. | C1 Stage A/B pass and a crowd set exists |
| I50 | Cross-panel prompt optimisation, plus RL | Low-data warning [verified]. Weak transfer [verified]. Tunes on the headline sets. RL is infeasible on 8 GB. | I08 shows a constraint-writing gap of at least 8 points, and a fresh confirmation set exists |
| I29 | Robot compositing plus forced choice (O-5c) | Tiny ceiling. Self-check record. Random-verifier result [verified]. | I14 shows the runners-up carry knowledge |
| I28 | Learned compositional energies | Relearns the analytic fields. Sums lose the exclusion ordering (SoGuDiff 99% → 70% without projection) [verified]. | A crowd set shows analytic fields fail beyond a patch |
| I43 | Frame sequences (P-7) | No motion scenes, no walking poses, and the 4-image cap competes with examples. | Isaac Sim exports per-frame people (do C4 first) |
| I17 | Active second view | The task is one image. Scenes are built to be fully visible. Doubles latency. | Occlusion scenes exist and multi-view is in scope |
| I31 | End-to-end trained predictor (P-8) | Inside C3. Leave-one-situation-out likely fails. 24 person-days. | The C3 structured probe transfers in at least 2 of 3 folds |
| I45 | Tour grounding (P-15) | The target is always in view: tour frame 3 is the observation, and keeping k = 4 frames drops it. A target found only in the tour cannot be scored. | Out-of-view scenes are wanted. Note that a two-call split of grounding is cheaper |

|---|---|---|---|
| O-5c: "imagine the robot there" (I29) | Composite or inpaint the robot at the solver's candidates, then ask the VLM to choose | The ceiling is tiny (runner-up headroom: Qwen 0, GLM 3, DeepSeek 14 goal-step records; brief section 3.4). The self-check record is poor (4 fixed, 11 broken). A random verifier recovers most of a verifier's gain [verified]. Depth-exact compositing needs no diffusion at all. | Wait for I14. Only if the runners-up carry knowledge beyond spread, run I29's 60-call probe with a random-of-4 baseline. [phase 1] I14 version 0 (17 Oct) did not show it: the top place plus 4 random disc cells recovers 96% of DeepSeek's realistic runner-up gain. The watching figure (33%) rests on a disc that leaves out about 90% of the region (69% over the whole room, post hoc). |

**Recommendation: no diffusion model this term.** Four reasons:
1. Labels would come from the project's own rules (circularity, section 8).
2. A clean generalisation test needs new situations written blind (I46), and none exist yet.
3. The solver already gives a cheap distribution to beat first (I04, I12).

- C2 Step 0 finds at least about 30 distinct configurations; [phase 1] not met: 22 on the generator's development rows;
- the simple density above transfers; [phase 1] met: a ratio KDE trained on fresh generator seeds beats the band-uniform baseline in 2 of 3 leave-one-situation-out folds, and loses at conversation groups;
- I04's soft density exists as the baseline (plan step 7). [phase 1] Met: V1 and V3 exist on the I14 harness and were verified; V1 is a kill against V3, so V3, a near-point mass on the solver's top place, is the baseline to beat.

[phase 1] So the gates do not pass (`phase1d/README.md` section 2), and by C5's rule the probe waits until the generator is widened. (inference) Gate 2 is the first evidence that a simple density learned from generator layouts reaches a situation it was not trained on, measured against the project's rules; I04's kill says the solver's own keys give no better distribution than a lightly smoothed top place.

(An earlier draft gated it on C3's probe instead. C5 does not use C3's relation table, so that gate is dropped.)

### 4.7 Coverage of the brief's directions (brief section 6)

| Direction | Covered by | Tier |
|---|---|---|
| O-1 Distribution over poses | I04 (with I14, I12); [phase 1] I04's first probe is a kill against its Gaussian control (`phase1d/README.md` section 1) | b |
| O-2 Other models helping the VLM | I09 (with I03, I33) | a (moved up 17 Oct; I03 is a) |
| O-3 Bystander-heavy, confusing scenes | I06 (with I18) | a (probe) |
| O-4 Faster inference | I21 (with I22, I23) | a |
| O-5a Generative model over goal poses | I52 (bold bet C5, added 17 Oct); I07 parked; see 4.6 | c |
| O-5b Generative images | I42; see 4.6 | d |
| O-5c "Imagine the robot there" | I29; see 4.6 | d |
| P-1 Numbered free-floor candidates | I41 (I24 is the solver-based form) | d (I24 b) |
| P-2 Depth/reachability filter | I41 | d |
| P-3 Continuous heatmap | I04 | b |
| P-4 Re-rank runners-up | I36 (revived only if I14 shows runner-up knowledge; version 0 did not [phase 1]) | d |
| P-5 Worked examples for constraints | I08 | b |
| P-6 Non-VLM person detector | I09 | a (moved up 17 Oct) |
| P-7 Frame sequences | I43 | d |
| P-8 Training for pointing | I31 (inside C3) | d |
| P-9 Faster MoE model | I44 | d |
| P-10 Thinking off where latency matters | I22 | a |
| P-11 Lower bounds, oracle ablations | I01 | a |
| P-12 Semantic baseline | I01 (`always_no` arm) | a |
| P-14 Solver rule changes | I38 (measured first by I06, I18, I19) | d |
| P-15 Tour grounding | I45 | d |

---

## 5. Research programmes

### Programme A: Geometry audits the VLM's perception

**Phase-1 check (17 Oct) [phase 1].** Partly true. Under the recorded constraints, full perception repair gives GLM +6.3 and DeepSeek +4.4 pp of 252 (lower bounds), and reference constraints give more (+7.9 and +9.5; circular). (inference) Inside perception, the lever is mostly mislocated people that `come_after` names, not missed people: adding the missed people alone fixes 2 of GLM's 29 and 2 of DeepSeek's 15 line failures. For the top tier, perception repair is worth at most +0.8 pp, and Gemini's loss is the target (oracle target +10.3 pp), which is I11's ground (`phase1/README.md` sections 1 and 8). [phase 1] I11 Stage 0 then classified Gemini's 34 wrong targets at the new wording: all are choice errors on place targets, and even the oracle snap-and-lift fixes none of them. (inference) So they are a choice problem for I11 Stage 1's marks, not a lift problem. For GLM and DeepSeek, a snap to the true target's mask fixes all 20 lift errors among their 46 wrong targets with no breaks, but zero-call depth components reach only 0.45 of that gain. With real depth blobs, relocating unsupported detections and adding the rest (a data-informed secondary arm) fixes 15 of GLM's 29 and 3 of DeepSeek's 15 line failures, and adding the blobs alone fixes 2 and 2 (`phase1b/README.md` sections 2–3). [phase 1] I09 A1–A2 then measured a learned people source: a local detector plus pose model, lifted with depth, beats the blobs on the realistic images (recall 0.983 against 0.890; counter staff 14 of 16 against 3 of 16; facing within 45 deg for 0.949–0.958 of the A0 pairs it matched for each model) but not on the placeholder ones (49 of 58 against 58 of 58). Moving the VLM's detections onto its people and dropping the unsupported ones gives GLM +16 and DeepSeek +17 of 252 under the recorded constraints: GLM's equals full perception repair, and DeepSeek's passes it through position and yaw differences alone (post hoc) (`phase1c/README.md` sections 1–3). (inference) The gain is again mostly I01's R4 mechanism, dropping detections that no reference supports, and on realistic images the best reference measured is a learned detector lifted with depth, not depth geometry alone.

**Ideas.** I01, I03, I11, I09, I33, I47, I49, I12; I10 if the blobs work.

**Experiments, in order.**
2. I03 Stages A and B, and I11 Stage 0.
3. I09 A1–A2 (moved up 17 Oct): the learned comparator, on the same people scorer.
4. I47 offline.
5. Live check L3. Live check L4 only once I06 family F exists (shared with Programme B).
6. I49: sensor-noise sweep.
7. One announced confirmation on fresh seeds (about 2,400 calls).

**What the report could claim if it works.**
- RQ1, on the family-F distractor and occlusion scenes plus the 2 recorded clutter layouts: wrong targets split into lifting errors vs choice errors; median-over-mask lifting; whether the gateway models read marks. Without family F, RQ1 rests on 2 layouts and is a case study, not a reliability estimate.
- A robustness bound under sensor noise.

**Against the closest work.**
- The mentor feeds learned perception (YOLOv10 + RTMW) into a learned map [verified].
- ReKep and VADAR split their oracles only in aggregate [verified].
- Exact depth flatters these methods, so the I49 sensor replay bounds the gap.

### Programme B: Constraints under crowd stress, scored honestly

**Thesis.** Current results show C beating direct pointing in scenes built to avoid the solver's known crowd failure modes. RQ2 asks how direct VLM goal prediction degrades with density, group size, bystanders and other groups. It has only thin, confounded data: one layout per line length or group size, at most 2 bystanders and 1 other group, none at lines (brief sections 3.2–3.3, the RQ2 baseline). The programme:
- builds crowded scenes with exact labels;
- maps where the solver breaks under perfect perception and simulated noise (solver-degradation curves, zero calls);
- then measures how R, E and C degrade with real VLM perception (step 8, about 2,000 calls, guess);
- reports layout-clustered statistics, a rule-sensitivity envelope, and labels written blind.

**Ideas.** I06, I18, I19, I14, I46, I35, I05; I38 only after measurement.

**Experiments, in order.**
1. I46 statistics and I14 version 0.
2. I06 probe. [phase 2] First probe done 18 Oct, families A–E: neither ((b) and (e) not met, no kill). Oracle C fails only at lines, 99 of its 103 failures by bystanders absorbed behind the line; groups and audiences held in every variant built (`phase2/i06/README.md`). [phase 2] A repair probe the same day, a facing gate on joining, is a method candidate: 29 of 54 judged absorption failures fixed and no break on the crowds or the development records, though S2 rested on the 4 goal steps where absorption happened (`phase2/i06_repair/README.md`).
3. I35 Part A.
5. I06 full pass, with a held-out design fixed in advance.
6. I18 search on that pass.
7. I19 synthetic harness.
8. An announced VLM run on 2–3 crowd levels, with R, E and C on three models: about 2,000 calls (guess: 14 layouts × 3 levels × 3 trials × about 5.3 calls per episode shared across methods × 3 models). This is the step that answers RQ2 for R and E.
9. The sealed-bank test, if a blind author is available.

**What the report could claim.**
- From the zero-call steps (2–7): solver-degradation curves under oracle and simulated perception, splitting solver failures from perception failures. These are not RQ2 answers on their own.
- RQ2, only after step 8: degradation curves for R, E and C with real VLM perception, set against the brief's thin baseline tables.
- RQ3: a test on harder scenes with a design fixed in advance.
- Label and rule sensitivity as honest validity statements.

**Against the closest work.**
- HabiCrowd scores trajectories [verified].
- The mentor's benchmark has 42 scenarios grouped by situation [partly].
- Procedural success hides narrow generalisation [verified].
- No crowd benchmark with goal regions was found (searcher note).

### Programme C: Make the constraint method fast

**Thesis.** Latency is output tokens in two serial calls: people 16.0 s and constraints 31.7 s for Qwen. Scheduling, thinking set per stage and geometric gating can remove most of that without changing what the solver sees. The honest target is C at or below the time of a marked point, with both methods under the same per-stage thinking policy (I22). [phase 1] I09 A1 measured a local people source at a median 55–62 ms warm on an idle 4060, against 16.0 s for Qwen's people call. It would take that call off the critical path only if the constraints call can address the detector's numbered people, which A2 did not test; the I09 live check would (`phase1c/README.md` section 6).

**Ideas.** I21, I22, I23, I05/I32, I39, I24.

**Experiments, in order.**
1. I21 offline replay plus the identity check; the I22 gate.
2. Live checks L1 and L2.
3. I23's table (from I01 R0).
4. I05 replay.
5. An announced latency run.

**What the report could claim.**
- RQ3 latency: critical path vs serial sum vs provider-only time.
- Where thinking pays, stage by stage.
- Whether C can beat a marked point on time when both run under one per-stage policy.

**Against the closest work.** Decode-bound latency (LocalNav) [verified]; dual-process gating [verified]; the mentor's 1.1 s trained model [verified].

### Which to start with

**Recommendation: start with Programme A's zero-call probes in phase 1 (days 1–13).**
- They are the cheapest and the most certain to give reportable numbers.

**Run Programme B's I06 probe first thing in phase 2 (days 14–17).**
- RQ2 has only thin, confounded data.

**Programme C's I21 and I22 are hygiene** that fit alongside either.

---

## 6. Plan for the next four weeks (two phases)

**Phase 1 (days 1–13): Programme A's core**

|---|---|---|---|---|---|
| 1 | Days 2–4 | I01 Steps 0–2, with R0 (I23's table) and facing accuracy (I09 A0). Moved first after stage 23 (finding 8): the solver-alone and people-blind standard-goal rows, the split into target finding and placement given the right target, and the R/E points replayed through C's clearance snap (I34 Stage 0), for all six models (s21, s23 and the older runs). Deliverables 1–3 done 17 Oct (`i01/README.md`). Steps 0–2 and A0 done 17 Oct (`phase1/README.md`): 1,301 of 1,301 places reproduce; Step 1 is neither success nor kill; the s8 calibration passes pooled. The optional pipeline oracle-people arm is not run (a big run, listed below). | Whole: about 3 person-days | 0 | — |

**Phase 2 (days 14–22 or so): crowd probe, rule envelope, speed hygiene**

|---|---|---|---|---|---|
| 5 | Days 17–19 | I35 Part A | About 2.5 person-days with the polygon builders | 0 | Variant family and margin rule frozen first |
| 6 | Days 20–22 | I21 offline replay, A1 implementation and identity check (mock model plus one cached run); I22 offline gate | I21 first result about 0.5 d (whole about 4 d; book 1.5 d); I22 about 1 d | 0 | — |

**Shared CPU.** Steps are sequential for the one worker. Two background CPU jobs overlap other work:
- the fresh-seed generation for I03 Stage A (seeds 100–109, moved from 3–12 on 17 Oct because seeds 3–99 are reserved for the I46 sealed set; done 17 Oct: 260 scenes at a mean of 4.1 s each, no held-out situation written);
- the I06 scene generation, started once its builder exists (about day 15), while the family-F label check is written.

- So for these models, old and new records come from the same prompt text. The only effect of the change is that recorded answers no longer hit the cache under the new schema.

|---|---|---|---|---|

The live-check total is about 410–500 calls, all free (L1 80, L2 74–106, L3 80–100, L4 180–210). L2's range depends on whether it runs at stage 21's load or re-measures its own baseline.

- Send the request to the mentor (I51).
- Agree the metric list (I14) and the L1/L2 crowd labelling rule (I06; not the live checks of the same names).

- I01's pipeline oracle-people arm: about 450 calls, up to about 2,300.
- The VLM crowd run (Programme B step 8; R, E and C on 2–3 crowd levels, three models): about 2,000 calls (guess). This is the run that answers RQ2 for R and E.
- I21 full latency run: about 2,700.
- I03 confirmation: about 2,400.
- I34 Stage 1: 120.
- I09 live: about 30–120, then about 1,400.
- I46 final test: about 6,000.

---

**Metrics and reporting**
- Whether the pipeline should take a person's heading from its 8-bin facing label instead of the faced point (A0, post hoc [phase 1]). The faced point sets the heading for 65–91% of people and, for 5 of 6 models, costs 4–23 points within 45 deg. True facing alone (R2) moves C by at most +2 net of 252 on these sets. A change would need pre-registering and testing on other sets; the placeholder panel splits 2 to 2. [phase 1] (inference) If I09's detector supplies the yaw, this question matters less: on the same A0 pairs its yaw is within 45 deg for 0.949–0.958 of the pairs it matched for each model (it misses 0–4 per model), where the models score 0.547–0.906 on the same pairs (`phase1c/README.md` section 1).
- [phase 2] Whether the solver's joining rule should get I06's facing gate (an unlisted person joins a line only when facing into its front half-plane; method candidate, `phase2/i06_repair/README.md`). It needs a facing for every detected person, so it ties into the heading question above: with a model's own yaw, one DeepSeek record had a real joiner turned away (a second person saved the success); with true or I09-detector yaw, every person absorbed on the records faced the line's front. Adopting it changes `src/sgi/solver.py` and its tests. (inference) A replay on records not used to design it should come first.
- How to read three criteria that phase 1 found open [phase 1]: whether I01's "at most 2 successes lost" covers all situations (DeepSeek P loses 4: not met) or only lines (loses 1: met); whether I01's calibration criteria apply pooled (met) or per model (ornith15 and qwen38 meet the kill's share clause by the letter, on +2 gains within noise); and whether I14's S3 covers only the DeepSeek cells (met) or GLM too (fails, 99 vs 107). No verdict changes overall under either reading.
- Per-situation comparisons reported as descriptive only, since 3–5 layouts cannot reach p < 0.05 (I46).
- Whether the Qwen headline may use mixed per-stage thinking (I22), and whether that counts as an allowed API setting.

**Data and labels**
- New held-out situations, with rules written by someone other than the designer (the teammate or the mentor), sealed before any method freezes (I46). Fresh seeds of existing situations, sized by the power analysis (17 Oct: at least 20 fresh layouts; I46 [phase 1]).

**Generality rulings** (the rule: no situation-specific code or prompts; brief section 7.3)

**Dependencies and compute**
- Shared GPU time for any Blender renders (crowd or fresh seeds), coordinated with the teammate.

**Runs**
- Whether a two-model method (I39) is acceptable under the panel rule.
- Whether to reopen the panel for qwen3.6:35b (I44). Recommend no.

**Scope and time**
- The real deadline (currently "late November, unconfirmed").
- Whether real images (I51) and a temporal extension (C4, the week-7 milestone) are in scope.
- Where the research code lives: a harness tool on build/pipeline, or the research branch. Whether `reference_constraints` moves out of `tests/`.
- Whether I11's zero-call depth-component snap (V1) may go into the pipeline [phase 1]. It broke no right-target record in 3,716 records and nets +8 successes for GLM and DeepSeek at the old wording (9 of their 46 wrong targets fixed, all background leaks), but it missed its pre-registered bar (0.45 of the oracle's gain against 0.70), so adopting it would need its own test (`phase1b/README.md` sections 3 and 7).

**For the mentor**
- Can the 42 group-joining scenarios, or the predictor's outputs on our scene files, be shared (I51)?
- Is a course click pilot subject to ethics review (I35)?
- Confirm or adjust the reference rules for lines, groups and watching, and the heading tolerance (45 vs 30 deg).

---

## 8. Risks to validity

- **Circularity.**
  - The reference rules are the project's own, and solver shapes coincide with them at lines, watching and lift (brief sections 4.1, 5e, 5f).
  - Oracle-constraint arms (I01 R5) are circular. [phase 1] Measured on 17 Oct, they do not score near 100%: with each model's own people, R5 gives 206 of 252 for GLM and for DeepSeek (old wording) and flips 19 of 29 and 9 of 15 line failures. Only with true people too (P5) does it flip every line failure, and wrong targets still fail.
  - L2 labels mirror the solver's own-space rule. [phase 2] Measured in I06's first probe: on 199 crowd variants, oracle C and the solver alone each disagree between L1 and L2 in 1.5%, the full oracle's stop in 42.7% (`phase2/i06/README.md` section 2).
  - Learned models trained on these labels (I13, I27 source A, I37, I07) inherit the circularity.
  - *Mitigation:* report L1 and L2 side by side; label oracle arms as upper bounds; rescore under the mentor's rule (I35 A, I51); human clicks (I35 B, the I27 CongreG8 check); new situations written blind (I46).
- **Few layouts.**
  - 14 layouts per set, re-dressed three times, with 3 correlated trials. The effective sample is about 14 clusters, and gains often rest on 3–5 layouts.
  - *Mitigation:* cluster by layout; report per layout; size fresh seeds with the power analysis.
- **Synthetic renders.**
  - Exact depth, no occlusion, 30 reused avatars.
  - The depth-based ideas (I03, I11, I33, I10, I47) look best exactly here. [phase 1] I09 rests on these conditions too: its people are lifted with exact depth, and its detector was measured only on renders, finding 0.983 of visible people with the realistic avatars and 0.845 with the placeholder ones (`phase1c/README.md` section 1). [phase 3] I09's fix found two more render effects and one untested dependence (`phase3/i09_fix/README.md` sections 1 and 5): the rendered depth treats glass as opaque, so the package's visibility rule calls staff in glass booths hidden while the picture shows them; the generator reuses its camera–staff geometry across the seeds of a row, so new seeds of the same rows are weaker evidence than their count; and the fix re-lifts occluded staff from 5×5 depth at head keypoints, which a real sensor may make noisy at head edges (untested).
  - *Mitigation:* the I49 sensor replay; say "simulation only" plainly; real images through I51 if possible.
- **Self-authored rules and tolerances.**
  - 45 deg here vs the mentor's 30 deg; an inconsistent inner band.
  - *Mitigation:* the I14 tolerance curve and the I35 envelope.
- **Held-out discipline.**
  - The picture and lift sets are spent, and a new method needs its own held-out design.
- **Tuning on the reporting sets.** Most probes use the realistic dev sets that carry the headline. Any method chosen this way needs confirmation on fresh seeds.
- **Anything a gateway model must know has to be in the prompt text.** An idea that adds fields to an answer (for example I32's attention edges in the people call) must explain them in the prompt. A field description alone would be a no-op for the gateway models, and a field name alone only constrains the decoding.
- **Examples in E.** On the realistic sets E shows 2 worked examples (`waiting_line_3/e1` and `no_one_person/e1`) and no conversation-group example. The placeholder sets get 3.
- **Leakage in offline scripts.** Labels and scores sit in the same records as model outputs. Method-side code must read only `result.*`, and be covered by the leakage test.

---

## Appendix A. The checked landscape

See `APPENDIX_LANDSCAPE.md` in this folder. It is complete, with all 14 angle sections plus a final section listing the claims the checkers corrected, disputed or rejected. It covers 112 sources and 372 claims (313 verified, 59 partly), with each claim's status tag, plus each angle's searcher summary and open questions (unchecked).

**I01.** Feasibility 5: about 3 person-days, every hook exists, zero calls. Payoff 3: diagnostic only. Perception repairs are lower bounds and R5 is circular. Skeptic: the cells are small; the result largely restates brief section 3.9; R3 cannot fix records with no `come_after` or with recall 0. Steelman: the s8 calibration against real true-people reruns turns this into a trusted zero-call test harness for any people source.

**I02.** Feasibility 5, payoff 3. The majority ceiling is small (+2.0/+4.3/+2.0 episodes over the per-trial mean). Skeptic: errors are correlated across trials; union fusion adds false people; a format-matched vote probably matches it. Steelman [re-run]: the 6-sample pool has at least 3 successes in 78 of 84 (a strict majority, at least 4 of 6, in 56). The re-run's swap test puts most of the deciding variance in the people answer, not the constraints.

**I03.** Feasibility 5, payoff 3. Skeptic: `_turn` absorbs only bodies near the next place in the line; 7 nobody-found and 8 no-`come_after` records cannot be fixed offline; pruning may drop people who were only misplaced; the scenes flatter depth clustering. Steelman [re-run]: pilot recall is near 1 except for staff; the blobs could become the people list itself.

**I04.** Feasibility 5, payoff 2. Top-1 is unchanged by design, and the banded keys mean the soft argmax can differ from the lexicographic pick. Skeptic: the temperature collapses or spreads, and lift is trivial. Steelman: a density plus a confidence; mass-ranked hit@3 may match the lexicographic hit@5.

**I06.** Feasibility 4 (about 14 person-days for the full pass; about 3 days to a first result), payoff 4 (the top payoff score). It is an instrument, not a method. Skeptic: it does not answer RQ2 for R and E (its zero-call output is solver-degradation curves); the oracle curves are near-circular; i.i.d. noise is too gentle. Steelman: all the parts exist; first curves within days.

**I07.** Feasibility 3, payoff 2. Skeptic: latency does not change while the VLM calls remain; the gain is capped by ranking headroom; leaving one situation out reduces to one weak fold (36 records). Steelman: watching combines line and group geometry, so it might transfer.

**I08.** Feasibility 4, payoff 3. The search may only rediscover `reference_constraints()`, and the example pool is thin. Steelman: it measures the constraint-writing gap, the analogue of ReKep's [verified], and may show that groups of 3 or more cannot be expressed.

**I10.** Feasibility 4, payoff 2. Little accuracy headroom. The latency gain needs a body source with no model; with the VLM people call, Qwen's nobody-near episodes slow down. Agreement with the reference is circular.

**I11.** Feasibility 4, payoff 3. Skeptic: depth components merge the occluder with the person; the gain rests on 2 layouts; models may misread marks. Steelman [re-run]: the person wrong-targets are edge near-misses that an oracle snap-and-lift fixes (15 of 15). The one zero-call rule tried (a front-surface snap) fixes 8 and breaks 37 of 752 C records, so a zero-call fix is not yet shown. The evidence rests on 2 layouts until I06 family F exists.

**I12.** Feasibility 5, payoff 3, p_disappoint 0.73. Skeptic: drops push the robot forward; the robust pose usually equals the top place; the confidence may detect the situation rather than the error. Steelman: phantom people from the blobs; an AUROC of 0.80 or more would give the first scalar confidence.

**I13.** Feasibility 3, payoff 3, novelty 4. Skeptic: little accuracy room; leaving one situation out is either trivial or impossible; circular; a logistic regression may match it. Steelman: `reference_constraints` is already a relation table with a situation label attached.

**I14.** Feasibility 5, payoff 3. An enabler. Skeptic: a proper score is undefined without human points; lift is trivial; random baselines inside 3 m may be high (guess). Steelman: honest lift claims, and evidence that the runners-up carry knowledge.

**I15.** Feasibility 4, payoff 3, p_disappoint 0.78. Coverage is likely vacuous for the weak models; set size stands in for the situation; the ask arm uses oracle answers. Steelman: ranks do not need a density; disagreement across trials is the best gate.

**I16.** Feasibility 4, payoff 2. Approaches are clear by construction (3.3% of C goal poses cross a forbidden polygon, 0.3% of C successes [re-run]); path cost pulls the stop to the near edge; execution is out of scope.

**I17.** Feasibility 2, payoff 2, novelty 4. The task is one image; the scenes are built to be fully visible; latency doubles. Steelman: render hooks exist; turning in place may give most of the gain.

**I18.** Feasibility 4, payoff 3. Skeptic: it rediscovers the known modes; labels are undefined in crowds; every candidate needs a depth render. Steelman: new clusters, and an envelope that can be fixed in advance.

**I19.** Feasibility 4, payoff 3. Skeptic [re-run]: most member misses are people never found, and the latency claim is dropped (about 4–7% of output tokens). Steelman: a synthetic crowd harness with no rendering; one mechanism fixes three failures.

**I20.** Feasibility 3, payoff 3, novelty 4. Wrong bottleneck; self-authored priors; no crowd data; weeks of MCMC. Steelman: a geometry-only variant drops the constraints call.

**I21.** Feasibility 5, payoff 2, novelty 1. Neutral on accuracy by construction, with a bounded saving; it does not flip RQ3. Steelman [re-run]: 61.2 → 49.3 s over the three realistic sets (62.9 → 50.6 s on the real set), and the thread-safe pacer also enables concurrent evaluation.

**I22.** Feasibility 5, payoff 2. Skeptic: earlier loss at "none"; underpowered tests; facing not measured. Steelman [re-run]: the all-none loss came from plan and context.

**I23.** Feasibility 5, payoff 2. Skeptic: the group advantage is noise; the gate becomes a line-vs-group switch. Steelman: it delivers the missing solver-alone pairing.

**I24.** Feasibility 4, payoff 3. Skeptic: coverage vs pruning; weak VLM choice; Qwen loses. Steelman: coverage may exceed any-of-5, and a consensus pick may make the VLM call unnecessary.

**I25.** Feasibility 4, payoff 3. Skeptic: the probe is near-tautological; role changes slip past the trigger; the speed claim needs a detector. Steelman: re-binding at t0 is a way to swap the people source.

**I27.** Feasibility 3, payoff 3. Skeptic: Qwen's group failures are wrong targets; Use 1 only breaks ties; source A is circular. Steelman: the CongreG8 ring check from about 380 trials.

**I28.** Feasibility 3, payoff 2, novelty 4. Relearns the analytic fields and loses the exclusion ordering. Steelman: a parametric fit first.

**I29.** Feasibility 4, payoff 2. Small ceiling; the candidates look alike; a random-of-4 baseline is needed. Steelman: the renderer gives a figure whatever happens.

**I30.** Feasibility 3, payoff 2. No occluded scenes; I03 covers the gain on current scenes; phantoms push the robot back.

**I31.** Feasibility 2, payoff 2. Knowledge enters through weights; leaving one situation out likely fails; 24 person-days. Steelman: about 1–3 s of prefill.

**I32.** Feasibility 4, payoff 3. Skeptic: reasoning moves into the people call; chains break; the oracle probe is circular; a held-out motivation (set aside). Steelman: geometric edges with no prompt change.

**I33.** Feasibility 4, payoff 3. Skeptic: coarser facing; false positives; scenes that flatter it. Steelman [re-run]: pilot recall is near 1 except for staff, and an oracle snap-and-lift fixes the 15 person wrong-targets; a zero-call fix is not yet shown. A held-out scan in the pilot is disclosed above.

**I34.** Feasibility 4, payoff 3. Skeptic: it inherits the missing people; reasoning errors dominate; diagrams get misread. Steelman [re-run]: E's group failures are near misses on a body.

**I35.** Feasibility 4 (Part A about 2.5 person-days with the polygon builders), payoff 3. Skeptic: the variants keep the shared core, so circularity remains; clicking on a perspective image is hard. Steelman: robust under the mentor's rules; humans may disagree with the group ring.

**I36.** Feasibility 4, payoff 2. Tiny ceiling; the self-check record; best-of-k is not proper.

**I37.** Feasibility 3, payoff 3. Ceiling about 27 records; circular; mismatched noise model.

**I39.** Feasibility 4, payoff 3. Skeptic: the offline and live designs differ; a trivial gate is the real baseline; the panel rule. Steelman [re-run]: an oracle gate gives 228/252.

**I40.** Feasibility 3, payoff 3, novelty 4. Lines need a queue rule; the label mirrors the constraint kinds.

**I41.** Feasibility 4, payoff 2. Errors are reasoning, not precision; duplicates I24/I36.

**I42.** Feasibility 2, payoff 1. 2D generators break the ground truth.

**I43.** Feasibility 2, payoff 2. No motion scenes; the 4-image cap.

**I44.** Feasibility 4, payoff 1. The panel is fixed; the speed comes mostly from thinking off.

**I45.** Feasibility 4, payoff 1. The target is always in view; the tour frame selection drops the observation.

**I46.** Feasibility 3, payoff 3. The statistics half is cheap and essential; the sealed bank needs a blind author.

**I47.** Feasibility 5, payoff 2. The flag was misread (corrected from the code); the reference coupling must be removed; 2–3 layouts.

**I48.** Feasibility 4, payoff 2. The flags fire on a similar share of DeepSeek successes and failures (10.4% vs 14.3%), but separate GLM failures (5.4% vs 30.3%); the gate can be gamed.

**I49.** Feasibility 5, payoff 2. Frozen constraints bias the comparison; renumbering artefacts. Steelman: a cliff caused by pitch error.

**I50.** Feasibility 3, payoff 2. Underpowered; tunes on the headline sets; RL infeasible.

**I51.** Feasibility 3, payoff 3. Depends on the mentor; conversion costs. Steelman: rescoring under the mentor's rule is free.

---

## Revision note (2026-10-17, after review)

- **RQ2.** It is now described as thin and confounded, with the brief's tables 3.2–3.3 as its baseline. "Hardly fail" is replaced. I06's zero-call output is labelled solver-degradation curves. The VLM crowd run (about 2,000 calls, guess) is named as the step that answers RQ2 for R and E.
- **RQ1.** I06 has a distractor family F with rules fixed in advance and a zero-call label check. I11 Stage 1 / L4 depends on it.
- **Statistics.** I35 Part A names the layout-clustered sign-flip test and gates on margins and layout counts, not p. I02 and I12 follow the same rule.
- **Latency.** I22 compares C and E under one per-stage policy, and L2 measures E's goal call at "none". The "flip" is conditional.
- **Pilot numbers.** They are moved out of the summary and tagged "pending whitelist re-run". I03's thresholds are re-derived from scratch, and step 0 re-runs the pilot.
- **Section 7.** Added the generality rulings (I05, I32, I13; I27 source B) and the P-13 / V1 status.
- **Tags.** Re-tagged the landscape claims the review flagged. Confirmed the CongreG8 licence and sizes, the model-card licences and the arXiv page directly at the source ([source-checked]).
- **Score table.** It now opens with a guess disclaimer and column notes, and the tier criteria and tier moves (I18, I25, I32) are explained.
- **Held-out list (lead researcher's check, after the revision).** The first revision's whitelist rule named `eval_16` and `eval_18`–`20` and left out `eval_17`, which is a group-photo scene. Step 0 and section 8 now exclude `eval_16`–`19`. `eval_20` is a development scene and stays in.
- **Step 0 re-run (same day).** The ledger is `RERUN.md`; the I01 tables are in `i01/README.md`. The report before these changes is kept as `REPORT.before-rerun.md`.
  - Main corrections:
    - The latency replay's 62.9 → 50.6 s is the real set alone. Over the three realistic sets it is 61.2 → 49.3 s, the same saving of about 20%.
    - In I02, the deciding variance between trials sits mostly in the people answer, not the constraints.
    - In finding 8, the C gaps among the top models are placement and target errors, not target finding at nobody-near scenes (1 of 44 gap records). Its group rows now count all 54 records.
    - The blob pilot's realistic recall depends on a split rule it did not state (real 36–42, staffed 34–39 of 42; placeholder 42 of 42). Its facing gain over chance is about +0.2, not +0.45.
    - I11's place targets are 5 of 23 recoverable, not 0.
    - The parked rows I48, I16 and I44 are rescoped.
    - I22's low-thinking cell (s12), and I44's "4 wins, 9 losses" and "136 vs 127", cannot be reproduced.
  - Finding 9 is added to section 1.1. I02, I03, I05, I06, I11, I19, I24, I32, I34 and I47 each carry one (inference) line pointing to it.
  - An independent checker's pass (same day) corrected the rest. The claim count now names its sixth class (R33, re-measured). Finding 9 names its sources and the three unreproduced rows, says "mostly" for the people-answer variance, tags the phantom mechanism as plausible, splits Qwen's gap fully, and tags the Gemini–I11 link as untested. Finding 8's tier gaps add up. The "unreviewed" lines (sections 2.5 and 8) and the plan rows for step 0, step 1 and I01 now say what is done. I01 and I23 note the offline solver-alone row. Section 5's phase days match section 6. Optional [re-run] lines sit under the criteria of I02, I21, I22, I33 and I39.
  - A second checker pass (same day) fixed 12 more points. The opening lines say what is built. Finding 8 limits its target-finding clause to the nobody-near scenes, splits Qwen's gap in full and hedges the phantom mechanism. Finding 9 and I06 say what the other constraint kinds add, and finding 9 and I02 note DeepSeek's near-even swap at s23. [re-run] lines now sit under I03 Stage A, I11's first probe and I47. I11, I22, I44, I47 and C5 get small wording fixes.
- **Phase 1, first results (17 Oct).** I01 Steps 0–2, A0, I14 version 0 and the I46 statistics are done, zero-call, each recomputed by an independent verifier (`phase1/README.md`). "How to read" defines [phase 1]. I01, I09, I14, I46, sections 1.2 and 6, findings 1, 8 and 9, section 8 and the opening status line carry the results. The report before these changes is kept as `REPORT.before-phase1.md`. I01 Step 1 is neither success nor kill, and its calibration passes when pooled. A checking pass (three checkers, each fix adjudicated) then corrected wording and added short notes to findings 8 and 9, I03's Stage B bar, I08, the O-5c and P-4 rows, section 6 step 7, section 7 (metric list, the heading question, seeds, pruning, open criteria readings) and section 8's circularity bullet; the version before that pass is `REPORT.before-phase1-check.md`. I14's set story survives its kill tests but fails S2 and S4. A0 sets I09's facing bar at 0.806. I46 sizes the sealed set at 20 or more fresh layouts. Short [phase 1] notes also go into Programme A's thesis, I03, I04, I11 and the I36 parked row.
- **Phase 2, checking pass (18 Oct).** Three checkers (numbers, stale claims elsewhere, structure), each fix adjudicated against the sources, proposed 51 fixes; 40 were accepted and merged into 21 edits, which also bring phase 2 into the opening lines, section 1.2, I46's seeds and section 7. The report before phase 2 is kept as `REPORT.before-phase2.md`, before the flow pilot as `REPORT.before-phase2b.md`, and before the checking pass as `REPORT.before-phase2-check.md`. Phase 1d (I04 and C5's gates 1–3) was not yet entered, and I06's C5 gate-1 recount waited for it; both are entered now (the next bullet). The repair probe's edits came after this pass (report before them: `REPORT.before-repair.md`) and are checked with phase 1d's.
- **Phase 3, confidence from perception disagreement (18 Oct).** A zero-call probe asked whether perception disagreement predicts when C fails (`phase3/confidence/README.md`), on the CPU, from whitelisted development runs and earlier phases' frozen outputs. The protocol note was frozen and hashed before any signal was computed against an outcome (data-informed: written knowing phase 1d's and A2's results), and an independent verifier reproduced every signal on all 1,265 records and all 2,725 table numbers with its own code. The judged signal P1, the shift of C's place when the solver re-runs with the VLM's people moved onto I09's detector people and the unsupported ones dropped (A2's setting D), is neither: AUROC for failure 0.645, 0.738 and 0.688 on the old-wording realistic cells (2 of 3 at 0.70 needed), pooled 0.708 against the flags' 0.595, kill line 0.60 not crossed. The verdict is fragile (dropping any one of 3 layouts, or using trial 1 or 2 alone, turns it into success; the success condition holds in half the bootstrap draws), and most of P1's ranking comes from the 37 records A2 had already shown would change (0.557 on the others; post hoc). Reported, not judged: people disagreement with the detector 0.741 pooled, trial disagreement 0.774 (two extra pipeline runs), and a zero-call combination 0.752 that keeps 0.878 success at 80% coverage; for the new-wording API models the perception signals carry little (P1 0.197–0.562). "How to read" defines [phase 3], and the opening lines, I04, I09, I12 and section 7 carry the result; optionally also section 1.2 (item 9), section 3.6, I39 and the I15 parked row. The report before these changes is kept as `REPORT.before-phase3-confidence.md`.
