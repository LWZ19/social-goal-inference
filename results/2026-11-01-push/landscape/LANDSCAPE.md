# Landscape for two research directions: moving people, and choosing from numbered candidates

Date: 2026-11-01. Scope: a targeted search for two directions that start now.
- **D1, moving people:** ask once, then re-solve on tracked people.
- **D2, choosing the target:** propose numbered candidates, and the model picks one.

The 17 Oct sweep (`cs6244_project-scenes/results/2026-10-17-beyond-vlm/REPORT.md`) already covered SoM, PIVOT, RoboPoint, VLMnav, ReKep, GSON, SocialNav-SUB and G2-Nav. Facts reused from it are tagged **VERIFIED (17 Oct)**. Its bet C4/I25 ("constraints as a standing program, re-bound to fresh detections") is the seed of D1. This file adds what that bet lacked:
- an event taxonomy;
- metrics over time;
- latency compensation;
- the newest related work, the mentor group's Sept 2026 preprint.

Tags:
- **VERIFIED:** I opened the arXiv abstract or HTML page today and checked the claim.
- **VERIFIED (17 Oct):** checked in the earlier sweep.
- **UNVERIFIED:** seen only in search snippets or recalled from memory.

---

## 1. Short answers

### D1: Moving people ("ask once, re-solve on tracks")

- **The pattern exists in manipulation.** The model writes a symbolic spec once, and the system re-evaluates it at high rate on tracked state.
  - VoxPoser caches the LLM code per sub-task and re-plans at 5 Hz.
  - ReKep re-solves its constraints at about 10 Hz on keypoints tracked at 20 Hz, and backtracks a stage when a constraint is violated.
  - The model is re-asked only on a monitored violation, as in DoReMi and Code-as-Monitor.
- **Social navigation copes with slow models in other ways:**
  - group labels kept on tracks, with the model asked at most once a second (GSON);
  - stale scores kept with defaults (G2-Nav keeps about 82% of the benefit at 4 s of staleness);
  - a stale choice turned into a decaying bonus (Slow Brain, Fast Planner);
  - a trained fast policy that conditions on the latency (TIC-VLA, DualVLN).
- **The closest work comes from the mentor's group:** "Where Should I Join?" (Fang, Wang, Lee, Hsu, arXiv 2609.28467, Sept 2026).
  - It predicts joining poses for conversations, queues and audiences.
  - It re-runs a trained sub-second predictor continuously on tracked people.
  - It keeps the current goal if it lies within 0.4 m of the top-5 new candidates, and accepts a change in who belongs to the group only after three frames.
  - Its only dynamic test is one queue with one newcomer, over 5 real trials: 4/5 for their method, 2/5 for GPT-5.5 sampling-and-selection.
- **The gap, in four parts. No paper we found:**
  1. re-binds a zero-shot model's identity-referencing social spec (come_after [4,5,6]) to tracks in order to keep a goal pose valid;
  2. sorts social events by whether they leave that spec valid (the served person leaves, a newcomer joins the end, someone leaves mid-line, a group shifts) or make it stale;
  3. defines situation-agnostic re-ask triggers;
  4. measures goal quality over time (time in the correct place, cut-ins, lost place, lag in moving up). All papers we found report success per episode only.
- **One capability the marked point (R) lacks by construction:** C can bind its answer to the people at the moment of the photo, then re-solve on where those people are when the answer arrives 5–60 s later. That turns latency into a tracking problem.

### D2: Choosing the target from numbered candidates ("propose and select")

- **Choosing from numbered candidates is common in robotics, but mostly for actions and paths, not for the target a request refers to.**
  - Actions and paths: PIVOT, VLMnav, CoNVOI, VL-TGS, and the mentor's RA-L 2026 path selection (3–6 numbered red paths, read by a fine-tuned Qwen2.5-7B).
  - Keypoints: MOKA.
- **The evidence on its weak points is consistent:**
  - Open-weight models read marks poorly unless trained to (SoM; SoM-LLaVA; MMIR finds "marginal gains").
  - Results are fragile to marker style: changing red to blue reorders leaderboards (VPBench).
  - Too many marks hurt: 5 marks beat 8 in Coarse Correspondences.
  - Correct reasoning gets attached to the wrong mark (PIVOT).
  - Models favour certain option labels as tokens (Zheng et al., ICLR 2024).
  - The candidates cap accuracy: on SoM, ground-truth masks give 90.1 against 75.6 with proposals; the mentor group's grounding misses a queue member who stands at twice the spacing.
- **The GUI-agent literature shows the trade-off sharply.** SeeAct finds SoM "not effective" on web pages. UGround finds direct pixel grounding from a trained model beats choosing among candidates.
- **The mentor group's numbers point to the split we already use:** GPT-5.5 grounds the right group well (F1 0.83, against 0.81 for their trained model), but places poses poorly (46–60% valid, against 90%).
- **The gap.** No study we found compares, for request-conditioned target grounding in social scenes:
  - picking a numbered target, against writing a pixel, against writing constraints;
  - on the same open and closed models;
  - with the harm on currently-correct cases measured;
  - with errors split into binding errors (right object, wrong number) and true misreads.
  Our 111/233 wrong-target failures are the right test set. Coverage-aware designs (recall@K, a "none of these" option with a pixel fallback) are also not studied in this setting.

---

## 2. Landscape facts

### 2.1 D1: queues, approaching groups, goals that update

- **Y. Nakauchi, R. Simmons, "A Social Robot that Stands in Line", Autonomous Robots 12(3):313–324, 2002.** Earlier versions appeared at IROS 2000. https://www.ri.cmu.edu/?p=8134
  - Stereo vision finds the line; personal space is modelled from measured human line-standing; the robot enters at the end and waits; demonstrated buying coffee.
  - UNVERIFIED: the citation and claims come from the CMU RI listing in search results (the page itself 404'd). "14 of 20 trials" is from a news piece only.
- **Z. Fang, Z. Wang, G. H. Lee, D. Hsu, "Where Should I Join? Robot Group Joining via Language-Guided Goal Prediction", arXiv 2609.28467, Sept 2026.** VERIFIED (HTML).
  - **Method.** Candidate member subsets come from recursive spectral partitioning. A SigLIP2 + GeoEncoder ranker scores them against the instruction. A goal predictor outputs an energy-orientation map.
  - **Validity rule.** A queue pose must be 0.5–1.2 m from the last member, with heading within ±30° of the queue direction.
  - **Offline pose validity (Table II):**

    | Model | Valid poses |
    |---|---|
    | Ours | 89.76 ± 4.21% |
    | GPT-5.5 few-shot | 59.76% |
    | Gemini 3 Flash few-shot | 53.57% |
    | GPT-5.5 zero-shot | 45.95% |
    | Qwen3.5-397B (thinking) | 39.52% |
    | Qwen3.5-397B (no thinking) | 22.62% |
    | Molmo2 | 10.71% |

  - **Grounding F1:** GPT-5.5 few-shot 0.831, ours 0.813.
  - **Latency:** about 1.1 s per goal query for ours, against 49.25 s for GPT-5.5 few-shot; on the real robot, "<0.7 s".
  - **Dynamics.** Inference "runs continuously and asynchronously" on globally tracked people. The goal is kept if membership is unchanged and it lies within 0.4 m of any of the top-5 new candidates. A membership change must persist for 3 frames.
  - **Real-robot results:**

    | Method | Static queue | Dynamic queue (one person joins during the approach) |
    |---|---|---|
    | Ours | 4/5 | 4/5 |
    | Sampling-and-selection (GPT-5.5, indexed poses on a top-down view) | 4/5 | 2/5 |

  - **Stated limits:** the correct group can be absent from the candidates (a customer at twice the spacing); audience goals are too conservative.
  - **For us:** this is the closest work and the reviewer's frame of reference. Its dynamics are handled by re-running a trained model, not by re-binding a zero-shot spec. It has no metrics over time.
- **Z. Fang, A. Xiao, D. Hsu, G. H. Lee, "From Obstacles to Etiquette: Robot Social Navigation with VLM-Informed Path Selection", RA-L 2026, arXiv 2602.09002.** VERIFIED (HTML).
  - A* candidates are clustered to 3–6 representatives, drawn as numbered red lines. A Qwen2.5-7B fine-tuned on GPT-4o labels (4,851 pairs) returns a "Chosen Path" number.
  - The goal is given, not inferred. Tracking uses a 40-frame buffer, plus 2.4 s motion prediction.
  - Latency is "a concern" but not measured. When an answer arrives, the closest current candidate is retrieved.
  - Stated limits: the model sees only the current frame (long queues are missed), and post-processing can remove the ideal path.
- **S. B. Banisetty, S. Forer, L. Yliniemi, M. Nicolescu, D. Feil-Seifer, "Socially-Aware Navigation: A Non-linear Multi-Objective Optimization Approach", ACM TiiS (in press), arXiv 1911.04037.** VERIFIED (abstract).
  - Waiting in a queue is one of its test scenarios. It is a planner with social objectives, not goal inference.
- **A. Eirale, M. Leonetti, M. Chiaberge, "Learning Social Cost Functions for Human-Aware Path Planning", IROS 2024, arXiv 2407.10547.** Queue etiquette is a learned cost. It fails on bent queues and at about 3 m spacing. VERIFIED (17 Oct).
- **N. Kim, H. Kim, S. Lee, C. Nam, "Hey, Robot! Don't Cut in Line: Designing Queuing Behaviors of Delivery Robots with Multiple Stakeholders", IEEE RA-L 11(6):7332–7339, 2026, DOI 10.1109/LRA.2026.3686647.** VERIFIED (repository abstract).
  - **For us:** human-side support for scoring cut-ins as failures.
- **S. Satake, T. Kanda, D. Glas, M. Imai, H. Ishiguro, N. Hagita, "How to Approach Humans? Strategies for Social Robots to Initiate Interaction", HRI 2009, pp. 109–116.**
  - It predicts pedestrians' walking, selects a target person and plans the approach. It is the classic for approaching people who are moving.
  - UNVERIFIED: from search snippets and dblp.
- **"Let me join you! Real-time F-formation recognition by a socially aware robot", arXiv 2008.10078, 2020.** Predicts the formation and the approach angle from an ego camera stream. UNVERIFIED (search snippet; authors not checked).
- **A. Francis et al., "Principles and Guidelines for Evaluating Social Robot Navigation Algorithms", arXiv 2306.16740.** Eight principles plus metric guidance. We found no metric for keeping a goal correct over time. UNVERIFIED (search snippet).

### 2.2 D1: slow model, fast loop (how a stale answer is handled, and when the model is asked again)

- **W. Huang, C. Wang, R. Zhang, Y. Li, J. Wu, L. Fei-Fei, "VoxPoser", CoRL 2023, arXiv 2307.05973.** VERIFIED (HTML).
  - The LLM code "remains the same throughout" a sub-task and is cached. The value maps "can be re-evaluated to reflect the latest environment observation", and the trajectory is re-planned at 5 Hz.
  - Masks are tracked with XMem.
  - The model is never re-asked mid-task. Most real-world errors came from perception.
- **W. Huang, C. Wang, Y. Li, R. Zhang, L. Fei-Fei, "ReKep", arXiv 2409.01652 (CoRL 2024, venue not on page).** VERIFIED (HTML).
  - Constraints are written once. Keypoints are tracked at 20 Hz and the solver runs at about 10 Hz. A violated path constraint makes it "backtrack to a previous stage".
  - Re-planning with new skeletons would need the model "at a high frequency", which the paper calls a challenge.
  - The tracker causes the largest share of errors. Automatic constraints reach 44.3%, against 68.6% with annotated ones; under disturbance, 26.7% against 46.7%.
  - **For us:** the closest structural analogue of D1. Note that the staleness it handles is geometric only: identities are fixed keypoints.
- **J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, A. Zeng, "Code as Policies", arXiv 2209.07753 (ICRA 2023).** Generated code can "express … feedback loops that process perception outputs". VERIFIED (abstract).
- **W. Yu et al., "Language to Rewards for Robotic Skill Synthesis", arXiv 2306.08647 (CoRL 2023).** The LLM writes a reward that MuJoCo MPC optimises in real time; 90% of tasks, against 50% for Code-as-Policies primitives. VERIFIED (abstract).
- **Y. Guo, Y.-J. Wang, L. Zha, J. Chen, "DoReMi", IROS 2024, arXiv 2307.00329.** The LLM writes constraints signalling misalignment; a VLM checks them "continuously"; a violation triggers recovery. VERIFIED (abstract).
  - **For us:** precedent for the model writing its own monitor conditions, alongside the plan.
- **E. Zhou et al., "Code-as-Monitor", CVPR 2025, arXiv 2412.04455.**
  - VLM-written code monitors spatio-temporal constraints on tracked "constraint elements", both reactively and proactively.
  - Under severe disturbance: +28.7% success and −31.8% execution time against baselines.
  - VERIFIED (abstract).
- **Y. Liao, Y. Lan, Y. Yang, R. Liu, X. Xu, S. Yuan, L. Xie, "G2-Nav", arXiv 2607.16956 (listed as CoRL 2026).** VERIFIED (HTML).
  - The VLM scores objects asynchronously, taking 1.5 s with a local model and 4 s with GPT-5.4-Nano. Objects not yet scored get a default score of 3.
  - It retains "82% (L2) and 81% (MAOE) of VLM's benefit … at the deployed 4s latency".
- **Z. Peng, H. He, Q. Li, Y. Ma, B. Zhou, "Slow Brain, Fast Planner: Latency-Resilient VLM-Augmented Urban Navigation", arXiv 2606.20458, June 2026.** VERIFIED (abstract).
  - The VLM picks a candidate index in 1–3 s. A stale choice becomes a scoring bonus "via geometric similarity with exponential decay". It keeps over 80% success at delays of up to 5 s.
  - **For us:** carrying a stale choice forward, by geometry instead of by identity.
- **Z. Huang, Y. Zhang, J. Liu, R. Song, C. Tang, J. Ma, "TIC-VLA", ICML 2026, arXiv 2602.02459.** Actions condition on delayed semantics plus latency metadata; delays are injected in training. VERIFIED (abstract).
- **M. Wei et al., "Ground Slow, Move Fast" (DualVLN), arXiv 2512.08186.** The slow VLM predicts pixel waypoints; a fast diffusion policy runs on them. VERIFIED (abstract); the 2 Hz and 30 Hz rates are from secondary summaries, so UNVERIFIED.
- **S. Luo, P. Sun, J. Zhu, Y. Deng, C. Yu, A. Xiao, X. Wang, "GSON", RA-L, arXiv 2409.18084.** VERIFIED (HTML).
  - People are marked with box, mask and tracking ID. The large model is asked only when a 1 s keyframe updates. Group labels are stored "in our tracking system's data structure".
  - Latency: about 1 s for Qwen2.5-VL-7B, about 4 s for GPT-4o and about 8 s for Gemini 2.5 Pro and Claude 3.7.
  - Grouping is accurate 77.8% of the time for GPT-4o and 53.6% for Qwen2.5-VL-7B. ID switches are not addressed.
  - **For us:** prior art for keeping a model's answer on tracks, but for avoiding groups along a path, not for a goal.
- **A. J. Sathyamoorthy et al., "CoNVOI", IROS 2024, arXiv 2403.15637.**
  - Numbered markers sit on free space taken from the occupancy map. The reference path is extrapolated while a CLIP context label is unchanged, and the large model is asked again when it changes.
  - UNVERIFIED (search snippet of the abstract).
- **A. Ahmadi et al., "Think When It Matters" (HUMA), arXiv 2607.10991, July 2026 (CoRL 2026 submission).** The VLM is called only when a person enters the proximity zone, and an RL policy handles the rest. VERIFIED (abstract); no latency is given.
- **D. Martinez-Baselga et al., "Hey Robot! Personalizing Robot Navigation through MPC with a Large Language Model", arXiv 2409.13393 (ICRA 2025 per search).** The LLM writes MPC cost code, and the solver is regenerated per instruction. UNVERIFIED (search).
- **Also seen, not opened.** All UNVERIFIED.
  - Hi Robot (Shi et al., ICML 2025, arXiv 2502.19417): a hierarchical VLM over π0.
  - RoboDual (arXiv 2410.08001) and Fast-in-Slow (arXiv 2506.01953): dual-system policies with a fixed ratio between slow and fast updates.
  - Vi-LAD (arXiv 2503.09820): distils VLM attention into an MPC costmap.
  - OLiVia-Nav (arXiv 2409.13675), Social-LLaVA (arXiv 2501.09024), VLM-Social-Nav (Song et al., 2024).
  - SocialNav-SUB (Munje et al., CoRL 2025, arXiv 2509.08757): VERIFIED (abstract), a question-answering benchmark only. The best VLM is below a rule-based baseline, per the 17 Oct check.

### 2.3 D2: choosing from numbered candidates, pointing, and how marks fail

- **J. Yang, H. Zhang, F. Li, X. Zou, C. Li, J. Gao, "Set-of-Mark Prompting…", arXiv 2310.11441.** VERIFIED (abstract) and VERIFIED (17 Oct).
  - SEEM/SAM regions are overlaid with marks.
  - GPT-4V goes from 25.7 to 86.4. Ground-truth masks give 90.1 against 75.6 with proposals, which is the coverage effect.
  - Open models "can hardly interpret the marks".
- **A. Yan et al., "List Items One by One" (SoM-LLaVA), COLM 2024, arXiv 2404.16375.**
  - "Other MLLMs struggle to understand these visual tags". Training on 10k–30k tagged images fixes it. VERIFIED (abstract).
- **Q. Yan et al., "MMIR", arXiv 2502.16033.** SoM "yields marginal gains". VERIFIED (abstract). The claim that it confuses non-GPT models is from a snippet, so UNVERIFIED.
- **H. Feng, L. Lian, L. Dunlap, J. Shu, X. Wang, R. Wang, T. Darrell, A. Suhr, A. Kanazawa, "Visually Prompted Benchmarks Are Surprisingly Fragile" (VPBench), arXiv 2512.17875.** VERIFIED (abstract).
  - Changing a marker from red to blue "can completely change rankings". A slightly larger marker lifts InternVL3-8B to the level of Gemini 2.5 Pro.
  - The effect is "substantially larger" on visually prompted tasks.
  - The figure of up to about 13% for Qwen2.5-VL is from a snippet only, so UNVERIFIED.
- **B. Liu et al., "Coarse Correspondences…", arXiv 2408.00754.** VERIFIED (HTML).
  - Marks with stable IDs across frames come from a tracker. "adding all the correspondences reduces the MLLM's performance".
  - On ScanQA with GPT-4V, K=5 marks beats K=8 on every metric (CIDEr 79.2 against 74.0).
  - **For us:** a direct number on the cost of too many marks, and precedent for IDs that stay the same across frames.
- **X. Fu et al., "BLINK", ECCV 2024, arXiv 2404.12390.** Humans score 95.7%, GPT-4V 51.3% and Gemini 45.7%, only 13.2 and 7.6 points above chance, on perception tasks that include visual prompts. VERIFIED (abstract).
- **C. Zheng, H. Zhou, F. Meng, J. Zhou, M. Huang, "Large Language Models Are Not Robust Multiple Choice Selectors", ICLR 2024, arXiv 2309.03882.**
  - The selection bias is a token bias towards particular option IDs. The PriDe fix estimates the prior by permuting options on a few samples.
  - UNVERIFIED (search snippet of the abstract). It transfers directly to choosing a number.
- **S. Nasiriany et al., "PIVOT", arXiv 2402.07872 (ICML 2024).** VERIFIED (abstract) and VERIFIED (17 Oct).
  - Candidates are drawn as arrows and refined over iterations; "far from perfect".
  - It links correct reasoning to the wrong mark.
- **F. Liu, K. Fang, P. Abbeel, S. Levine, "MOKA", arXiv 2403.03174 (RSS 2024).** VERIFIED (HTML).
  - Candidates are GroundedSAM masks, with K contour points by farthest-point sampling plus the centre. Waypoints use a 5×5 tile grid.
  - It turns "directly generating coordinates into multiple-choice questions", but has no ablation against direct coordinates.
- **D. Goetting, H. G. Singh, A. Loquercio, "VLMnav", arXiv 2411.05755.** VERIFIED (abstract) and VERIFIED (17 Oct).
  - 50.4% with the depth-based navigability proposer, 33.2% without it, 29.8% as text only, and 24.6% for PIVOT.
- **D. Song, J. Liang, X. Xiao, D. Manocha, "VL-TGS", arXiv 2408.02454.** A CVAE generates candidate trajectories and the VLM picks one by visual prompting; +20.81% on traversability constraints. VERIFIED (abstract).
- **W. Yuan et al., "RoboPoint", arXiv 2406.10721 (CoRL 2024).** VERIFIED (abstract).
  - +21.8% over GPT-4o and PIVOT on spatial affordance. 46.77% against 29.06% for GPT-4o: VERIFIED (17 Oct).
  - That the Where2Place benchmark comes from this paper is UNVERIFIED.
- **E. Zhou et al., "RoboRefer" (RefSpatial, RefSpatial-Bench), NeurIPS 2025, arXiv 2506.04308.** A 3D-aware VLM for spatial referring; the RFT version beats Gemini-2.5-Pro by 17.4% on RefSpatial-Bench. VERIFIED (abstract).
- **L. Cheng et al., "PointArena", arXiv 2505.09990.** VERIFIED (abstract).
  - Molmo-72B is best, and supervision aimed at pointing "significantly enhances" performance.
  - "Chain-of-thought lowers pointing accuracy" is from a snippet, so UNVERIFIED. It is relevant to R's reasoning field.
- **C. Clark et al., "MolmoPoint: Better Pointing for VLMs with Grounding Tokens", arXiv 2603.28069, Mar 2026.** VERIFIED (abstract).
  - Instead of writing coordinates, it selects image tokens, then a subpatch, then a location. 70.7% on PointBench.
  - **For us:** choosing inside the model beats writing coordinates, but it needs training.
- **B. Zheng, B. Gou, J. Kil, H. Sun, Y. Su, "SeeAct: GPT-4V(ision) is a Generalist Web Agent, if Grounded", arXiv 2401.01614 (ICML 2024).** VERIFIED (abstract).
- **H.-T. L. Chiang et al., "Mobility VLA", arXiv 2407.07775.** A long-context VLM picks a goal frame from a tour video, so it selects from candidates. VERIFIED (abstract).
- **"SeeGround" / "Zero-Shot 3D Visual Grounding from Vision-Language Models", arXiv 2412.04383 / 2505.22429.** The VLM predicts an object ID on rendered views, then retrieves the 3D box. UNVERIFIED (search).
- **M. Lu, A. Xiao, D. Hsu, "MIGU", arXiv 2609.24995, Sept 2026.** VERIFIED (abstract).
  - VLM priors over candidate objects and regions are fused into a grounding belief, which either drives planning or asks for clarification.
  - From the mentor's group; precedent for a belief over candidates instead of a single pick.
- **N. Carion et al., "SAM 3: Segment Anything with Concepts", arXiv 2511.16719, Nov 2025.** VERIFIED (abstract).
  - Detects, segments and tracks every instance of a noun-phrase concept, with identities. It is a candidate source for both D2 and D1.
- **Molmo/PixMo (Deitke et al., arXiv 2409.17146).** Open pointing model. UNVERIFIED (memory).

---

## 3. Research questions and probes

- SoCLaaS models (Qwen 3.8, GLM-5.3-Flash, DeepSeek) are free, paced at 85/min.
- A quick probe uses 5–10 trials per condition.
- At most four images per prompt.
- One shared prompt for every model.
- No coffee- or queue-specific code.
- "0 calls" probes replay answers already on disk, or use `reference_constraints()` as an oracle for a fresh answer.

### 3.1 D1: moving people

#### Metrics to define first

This is new: no paper we found reports any of these.

| Metric | Definition |
|---|---|
| Correct at arrival | Whether the goal is correct at the moment the robot arrives. |
| Cut-in | The robot's final place is ahead of someone who was already waiting when the robot started. |
| Lost place | Someone who arrived after the robot ends ahead of it. |
| Lag in moving up | Seconds from the gap ahead opening by more than one spacing to the goal moving up. |
| Goal churn | Metres of goal motion per minute while nothing happens in the scene. |
| Re-asks per minute | How often the model is asked again. |

#### Event taxonomy

Scripted on ground-truth tracks, written in general terms:
1. **Front leaves:** the served person leaves and everyone advances.
2. **Newcomer before arrival:** a newcomer joins the end before the robot arrives.
3. **Newcomer after the robot:** a newcomer joins after the robot has taken its place.
4. **Member leaves:** someone in the middle leaves.
5. **Role change:** a waiting person turns away and becomes a bystander.
6. **Group shifts:** a conversation group re-forms or a member leaves.
7. **Watcher moves:** a person watching moves across the line of sight between the audience and the target.

**Q1. How long does each answer stay valid, per event type?**
- **Hypothesis.** R's frozen point and C's frozen pose fail once the line advances by one spacing (event 1).
  - C re-solved on tracks survives events 1, 3, 4 and 7.
  - Re-solved C fails only on events that change who is involved (2 and 5).
- **Cheapest probe (0 calls, about 1 day).**
  - Take recorded C answers and R points from existing runs on the generated set, e.g. 40 line, group and audience scenes × 3 SoCLaaS models.
  - Associate the model's people numbers to ground-truth people at t0 by nearest match.
  - Script events 1–7 forward at 10 Hz for 30 s. Feed the solver the tracked ground truth plus noise (σ 0.1 m, and 5% ID swaps as a stress condition), with the numbering kept.
  - Compare five conditions:
    - R frozen;
    - C frozen;
    - C re-solved;
    - C with an oracle fresh answer (`reference_constraints()` on the new state);
    - the oracle with a delay of L = 5, 20 or 60 s.
- **Surprise.** Re-solved C does worse than frozen C on event 1, for example because the solver's line-end logic jumps when the named first person leaves. Or R's frozen point survives because the correct region is wide compared with the spacing.
- **Why new.** ReKep and VoxPoser re-solve geometry with fixed identities. "Where Should I Join?" re-runs a trained model and reports one dynamic queue, success out of 5. Nobody splits staleness into geometric versus identity for a zero-shot social spec.

**Q2. Does re-binding at answer time recover the accuracy lost to latency?**
- **Hypothesis.** In the Habitat cafés, people move while the model answers. Solving C on positions at answer time, not at ask time, recovers most of the loss. R cannot do this, because its pixel belongs to the old frame.
- **Probe.**
  - Run `cafe_line2` and `cafe_line6` with animated people, 10 episodes each, on 1–2 SoCLaaS models: about 20–40 pipeline runs, roughly 80–160 calls.
  - Use the simulator's ground-truth tracks as an oracle tracker, with IDs.
  - Solve the same answer two ways: at ask time and at answer time. This costs no extra calls.
  - Report correctness at arrival, against the measured latency.
- **Surprise.** No gain, because people barely move during 5–60 s in these scenarios. That would be a finding about the scenarios. Then script a line that advances.
- **Why new.** TIC-VLA and Slow Brain, Fast Planner compensate for latency with training or with geometric similarity. Compensating by identity, using the symbolic answer, has not been shown.

**Q3. Which situation-agnostic trigger tells us to ask again?**
- **Triggers to compare** (no trigger names a situation type):
  - T0: never ask again.
  - T1: ask every N seconds.
  - T2: a named track is lost, or a new track appears within r of the goal or the named people.
  - T3: the re-solved pose jumps more than d in one tick, or the solver becomes infeasible.
  - T4: the answer's own expiry conditions fail (see B1 below).
- **Probe (0 calls).** On the Q1 replays, plot TICP against re-asks per minute, with each re-ask served by the oracle answer after a delay L.
- **Hypothesis.** T2 or T3 reaches at least 90% of the "always fresh" TICP with at most 1 re-ask/min, while T1 needs at least 6/min.
- **Surprise.** T3 misses role changes (event 5) entirely, because the pose barely moves.

**Q4. How should a moving goal be committed to?**
- **Policies to compare:**
  - update every tick;
  - the mentor group's rule (keep the goal if it is within 0.4 m of the top-5 candidates and membership is stable for 3 frames), with the solver's top-k cells as the candidates;
  - commit within 1.5 m of arrival, after which only moving up is allowed.
- **Probe.** Habitat, one answer per episode, 10 episodes per policy. Score cut-ins, lost place, goal churn and path length.
- **Surprise.** Updating every tick causes cut-ins, because the robot races a newcomer to a moving end of the line.

#### Bold ideas for D1

**B1. [BOLD] The answer carries its own expiry conditions.**
- **What.** In the same constraints call, the model also writes 1–3 validity conditions in the existing vocabulary, for example "within 1.5 m of [4,5,6]" or "visible_to [2]", or "the answer holds while …". The re-solver checks them each tick; a failure triggers a re-ask.
- **Why it is new.** It is a self-authored monitor, like DoReMi or Code-as-Monitor, but in one call and over people.
- **Probe.** 10 scenes × 3 SoCLaaS models × 1 call = 30 calls, then replay as in Q3 to score T4.
- **Score:** payoff 4, uncertainty 4 (models may write trivial or always-true conditions), cost about 1 day and ~30–60 calls.

**B2. [BOLD] Arguments by role, not by number.**
- **What.** Instead of come_after [4,5,6], the model names a role defined from the existing constraint types: "people who face the target within X m". The solver re-evaluates the role on tracks each tick, so newcomers who fit the role join automatically and event 2 becomes absorbable.
- **The risk.** Roles drift towards situation-specific predicates, which goes against the rule that the pipeline must generalise. The vocabulary must stay the shared one.
- **Probe.** 0 calls on oracle roles first, to check whether roles can be expressed for line, group and audience in the existing vocabulary; then 30 calls.
- **Score:** payoff 4, uncertainty 4, cost 1–2 days.

**B3. Solve on where people will be at arrival.**
- **What.** Re-solve on tracks extrapolated by the robot's estimated travel time; the mentor's path paper uses 2.4 s of prediction. Expected to reduce stale ends at arrival on event 1.
- **Probe.** 0 calls, on the Q1 replays.
- **Score:** payoff 2, uncertainty 2, cost half a day. A cheap add-on to Q1, not bold.

**B4. [BOLD] Tracker IDs as the model's people numbers.**
- **What.** Draw the tracked detections as numbered marks for the people call, so the model's numbering equals the tracker IDs. That removes the association step and makes every later re-ask diffable against the last answer: if the same people are named, apart from known events, the answer is consistent; otherwise flag it. This links D1 and D2.
- **Probe.** It shares calls with D2 Q8.
- **Score:** payoff 3, uncertainty 4. It meets the open models' weakness at reading marks. Cost 1 day.

### 3.2 D2: choosing the target from numbered candidates

**Q5. Where do the 111 wrong-target failures come from, and would candidates cover them?**
- **Probe (0 calls, half a day).**
  - Label the 111 failures as place, object or person.
  - Compute recall@K of the true target for three proposers, at K = 3, 6 and 10:
    - the generator's ground-truth object list (an oracle upper bound);
    - class-agnostic masks or depth-connected floor-standing components;
    - SAM 3 concept prompts from the plan's noun phrase plus 2–3 alternatives the plan step already writes, if any.
- **Hypothesis.** Most wrong targets are interpretation errors between visible alternatives (two counters, or a counter against a menu board). These are covered at K = 6 with recall of at least 0.9.
- **Surprise.** Many failures are misreads of a person (the wrong member of a group). Candidates then belong in the people call, not the plan call.

**Q6. Does picking a number beat writing a pixel for the target, per model family, including harm?**
- **Probe.**
  - Take 30 cases: 15 drawn from the wrong-target failures and 15 currently right.
  - Run only the target step, as pixel (current) against numbered marks, with K ≤ 6, ground-truth candidates, one fixed marker style and the same prompt for every model.
- **Hypothesis.** Closed models gain. Open models gain less or lose, because they misread marks (SoM, SoM-LLaVA).
- **Surprise.** Open models gain more, if their failure is writing pixels rather than understanding the scene. That would turn around what the SoM literature implies.
- **Why new.** No head-to-head of numbered choice against pixels for request-conditioned target grounding in social scenes, split by model family and with harm on currently-correct cases. Our pipeline also has a third arm that the literature lacks: the constraint method's own grounding.

**Q7. Are the errors in binding or in understanding, and is there a bias towards some numbers?**
- **Probe.**
  - Same 30 cases × 3 random number permutations × 3 SoCLaaS models = 270 calls.
  - Parse the reasoning text for the object it describes, and compare that with the mark it chose. A mismatch is a binding error.
  - Measure whether choices follow the number rather than the object, and apply a PriDe-style correction: estimate the prior over numbers from permuted samples, then divide it out.
- **Hypothesis.** Binding errors are at least 30% of the open models' errors with marks.
- **Surprise.** Permuting the numbers flips more than 20% of an open model's choices, which would be a label bias.
- **Why new.** PIVOT reports binding errors qualitatively. We found no measurement of them, or of number-token bias, for visual marks in robotics.

**Q8. How do coverage and clutter trade off, and does a "none of these" option work?**
- **Probe.**
  - K = 3, 6 or 10, with the true target present or removed, and an explicit "none of these, describe it" option that falls back to the pixel step.
  - 20 cases × 3 values of K × 2 presence conditions × 2 models = 240 SoCLaaS calls.
- **Hypothesis.** Accuracy peaks around K = 5–6 (Coarse Correspondences found 5 better than 8). Abstention recall is at least 0.7 when the target is removed, with no more than 10% false abstentions when it is present.
- **Surprise.** Models never abstain. Coverage would then have to be guaranteed upstream, which favours the oracle-style proposers from Q5.

**Q9. Can marking the people remove a call?**
- **What.** This attacks C's 1.4–1.9× latency. Draw detector- or tracker-proposed people as numbered marks, and merge the people call into the constraints call. The model selects the relevant people and writes constraints referring to their numbers.
- **Probe.** 30 cases × 2 SoCLaaS models = 60 calls, with ground-truth boxes plus noise as an oracle detector. Compare success and wall time with the current two calls.
- **How it differs from "adding missed people".** That earlier idea tried to raise coverage. This one tries to cut a call, and it gives D1 its tracker IDs.
- **Surprise.** Accuracy holds with one fewer call. That would close most of the speed gap promised in the proposal.

#### Bold ideas for D2

**B5. [BOLD] Let the people propose the target.**
- **What.** Candidate targets are the points where people's facing directions converge: the service point a line faces, a stage, the centre of a group. These are ranked alongside detector candidates, and the model picks a number.
- **Why it might work.** It is general across lines, audiences and groups. It attacks wrong targets in busy scenes, where context is strongest.
- **Probe.** 0 calls first: recall@K on the 111 failures, as in Q5. Then reuse the Q6 calls.
- **How it differs.** It is a new use of the facing estimates, not a better facing estimate (already tried).
- **Score:** payoff 4, uncertainty 3, cost 1 day.

**B6. [BOLD] The target's identity is tracked too.**
- **What.** Once the model picks target #k, its candidate ID enters the constraint call ("face #k"), and SAM 3 tracks it. The whole answer is then a set of IDs plus constraint types, which can be re-solved on tracks for both people and target. This unifies D1 and D2.
- **Score:** payoff 3, uncertainty 3, cost 1–2 days. It matters only if targets move or are re-detected.

**B7. [BOLD] A belief over candidates, then a cheap action.**
- **What.** Ask for the top-2 candidates with a margin, in the style of MIGU. When the margin is small, check whether the two candidates' solved goals are within 0.5 m. If they are, the choice does not matter. If not, ask once for clarification, or take the people-weighted candidate.
- **The overlap.** This is close to "conformal goal sets" (already tried). It differs because the set is over target identities, not poses.
- **Score:** payoff 2, uncertainty 3, cost about 0.5 day. Run it only if Q6 shows many near-ties.

### 3.3 Suggested order, given under a week

- **D1:** Q1 (0 calls) → Q3 (0 calls, the same replays) → Q2 (about 100 free calls) → B1 if there is time.

Q5 and Q1 decide whether the rest is worth running, so do them first.

---

## 4. Things to avoid

- **Asking the model every frame, or at a fixed fast rate.**
  - GSON's large models take 1–8 s and it asks at most once a second.
  - The mentor group's GPT-5.5 sampling-and-selection was "impractical" for timely updates: 49 s per query, 2/5 on the dynamic queue.
  - Re-binding plus a trigger is the point; periodic re-asks are only the T1 baseline.
- **Model-selected stop poses as the main placement method.**
  - Zero-shot pose validity is 46–60% even for GPT-5.5, against 90% for geometry.
  - Keep numbered choices for which target and which people, and keep placement in the solver.
- **Many marks.** More than about 6–8 marks hurts (5 beat 8 in Coarse Correspondences). Tile grids on the image, as in MOKA, are also out: drawing a grid was already tried and did not help.
- **Assuming open models read marks.** SoM, SoM-LLaVA and MMIR all say they may not. Test on SoCLaaS before building anything on marks.
- **Ranking models from one marker style.** VPBench shows that colour and size reorder models. Fix one style for every model, and use a second style only as a robustness check.
- **PIVOT-style loops that refine over several rounds.** They add calls and latency, and PIVOT scored 24.6 against 50.4 in VLMnav's comparison.
- **Trained dual-system or latency-conditioned policies** (DualVLN, TIC-VLA, RoboDual, Fast-in-Slow, Vi-LAD). They need training data and time we do not have, and they go against the zero-shot claim.
- **Situation-specific re-binding rules,** such as "append whoever stands behind the last named person". They are queue-specific code. Use the generic triggers T2–T4 or role arguments built from the shared vocabulary (B2).
- **From the already-tried list, still to avoid:**
  - flow-matching or generative pose samplers;
  - soft solver densities and conformal goal sets (B7 is allowed only as a target-identity tie check);
  - adding missed people (Q9 is about cutting a call, not about coverage);
  - better facing estimates (B5 uses facing for proposals, not accuracy);
  - grids drawn on the image;
  - a second look at the model's own point;
  - weak-to-strong cascades;
  - fast/slow gating by a situation detector (HUMA's proximity gate is in the same family; our triggers concern whether an existing answer has gone stale, not which method to run).
- **Claiming novelty for "re-solve a VLM spec in closed loop" in general.** VoxPoser and ReKep did it. Claim instead:
  - identity-referencing social specs;
  - the event taxonomy and metrics over time;
  - latency compensation by identity;
  - zero-shot operation across six models.
