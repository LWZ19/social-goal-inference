
## About this audit

An index of every experiment and idea, with how strong its evidence is and where its files are. The follow-up runs it called for were done on 6-7 Nov; their results are in `fill/` (start with `results/overview/RESULTS_7NOV.md`).

- **Read-only.** No model or simulator call was made for it.
  
**Strength:**
- **conclusive**: a frozen bar met or failed with adequate n;
- **suggestive**: consistent, but exploratory, post hoc or modest n;
- **quick probe**: small n, one model, old scoring, or superseded code.

**Priority:**
- **needed**: the report's claims depend on it;
- **nice**: strengthens or widens a claim;
- **not worth it**: closed, superseded, or off the report's path.

**Cost basis.**
- Free school models (Qwen 3.8, GLM-5.3, DeepSeek V4.1) cost nothing, paced inside 85 requests a minute.
- A moving episode is about 90-96 s of wall time, so 2-4 workers give 40-110 episodes an hour.
- A realistic scene renders in about 70 s; a placeholder scene in about 4 s.

## Part 2. The rows

### A. Still scenes: the method comparison and prompts (RD, s16b-s23)

- **A1. C vs R vs E, new wording, six models (s21, s23)**
  - Question: does C beat R and E for every model?
  - Data: Qwen 3.8 low, GLM-5.3, DeepSeek V4.1, Haiku 5.5, Sonnet 5.5 low, Gemini 3.8 low; 14 development layouts x 3
    realistic variants x 2 episodes x 3 trials = 252 records per method per model.
  - Frozen: no. Planned runs, but the context prompt was chosen on these sets.
  - Result (R/E/C): Qwen 179/196/228, GLM 132/123/170, DeepSeek 120/126/185, Haiku 151/148/240, Sonnet 178/178/242,
    Gemini 164/173/214.
    - Layout-level C−E is +32 to +92, p 0.0039-0.0156, each the floor for its untied count. C is below E in 0 of 84
      model x layout cells.
    - **Conclusive in-sample; suggestive for generalisation.**
    - Caveats: the gateway models' C people answers are cache copies of s19/s20 (one people draw each). C's line lead is
      partly built into the band (A9).
  - To conclude: Part 1 #12 (fresh seeds, cheap), or a sealed set of at least 20 new templates written blind (strong;
    needs new rows written).
  - Priority: **needed** (the headline). Paths: `results/still-scenes/overview/REPORT.md` §1.1, `results/still-scenes/overview/RERUN.md`, `results/still-scenes/replay-harness/sets-and-definitions/README.md`.
- **A2. Old-wording gateway headline (s16b R and E; s19 and s20 C)**
  - Data: Qwen 175/214/234, GLM 129/145/186, DeepSeek 113/120/182 (R/E/C of 252).
  - Frozen: no.
  - Result: layout-level C−E is Qwen +20, p 0.031 (6/0/8 layouts: one more tie loses it), GLM p 0.0078, DeepSeek
    p 0.00098. **Conclusive in-sample for GLM and DeepSeek, fragile for Qwen.**
  - To conclude: as A1.
  - Priority: **needed**: switch report.tex to layout-level tests (Part 1 #1). Path: `RD/audit/SYNTHESIS.md` A6.
- **A3. E vs R (worked examples)**
  - Result: pooled E−R is +3 (−9 to +9 by model). Qwen's 9-draw pool gives +24 per 252, borderline. None is
    significant at the layout level. **Suggestive null** (Qwen's gain suggestive).
  - To conclude: 20 or more fresh layouts with R and E paired within a stage; free.
  - Priority: **needed** as a plain statement that examples add little.
- **A4. Run-to-run noise (Qwen, 9-draw pool)**
  - Result: batch SD 7.93 records (df 4); smallest detectable difference 24-32 between batches and 11-21 paired.
    GLM's two same-prompt C runs differ by 16. **Suggestive** (one model).
  - To conclude: 2-3 repeat batches for GLM and DeepSeek (700-2,000 free calls each).
  - Priority: **needed** as the reading bar; extending it is nice.
- **A5. Context-step rewording (10 Oct)**
  - Data: Qwen and GLM, 252 records; three rewordings screened on these sets.
  - Result: C Qwen 224→234, GLM 172→186. Default-stop errors at lines fell from 11 to 0 (Qwen) and from 5 to 2 (GLM).
    **Suggestive** (selected in-sample; totals about 1.3 batch SD).
  - To conclude: a paired old-vs-new context call on fresh layouts (about 500 free calls a model).
  - Priority: nice; disclose as in-sample.
- **A6. Earlier pipeline changes (placeholder panel, 5 models)**
  - Reasoning before the point: about +20. Stopping in front of the target's face: +50 of 180. Examples: about +20.
    Goal rewordings: nothing.
  - **Quick probe** (old code, single runs).
  - Priority: nice as one line of history; not worth re-running. Path: `docs/results/README.md`.
- **A7. Schema field descriptions (commit f1a488a)**
  - Priority: **needed** as a methods footnote.
- **A8. Model tiers**
  - Result: two tiers (Qwen, Sonnet, Gemini, plus Haiku on C; then GLM and DeepSeek). Nothing orders the top tier.
    **Suggestive** (one batch each).
  - Priority: nice; report tiers, not a ranking.
- **A9. Band sensitivity of the still-scene result**
  - Result: C's lead holds with the far edge at 1.5-2.0 m. At 1.0 m (the solver's own standoff) it reverses on 5 of 6
    models. The 0.55 m start changes no conclusion. **Conclusive** as an exact rescore: C's line lead is partly built
    into the band.
  - To conclude: freeze the I35 Part A variant family and margin rule (zero calls).
  - Priority: **needed** as a caveat.
- **A10. Watching (audience) situation**
  - Data: placeholder, development seeds 0-2, 36 per cell, 2 templates.
  - Result: R/E/C after the solver fix: Qwen 18/25/32, GLM 4/5/23, DeepSeek 4/9/23. The fix was designed on these
    scenes. **Quick probe.**
  - Priority: nice.
- **A11. Realistic variants (real, staffed, plain) and placeholder vs realistic**
  - Result: C by variant is within a few records (e.g. Qwen 77/76/81 of 84). No test. **Quick probe / suggestive.**
  - To conclude: a layout-clustered breakdown of the existing records (zero calls).
  - Priority: nice. Claim only "no large realism effect".

### B. Still scenes: replays, decomposition and ablations (RD)

- **B1. Replay harness and re-run ledger**
  - Result: 76 claims checked: 53 confirmed, 19 corrected, 0 refuted. The harness reproduces 4,536 of 4,536 verdicts.
    It is exact only for changes made after both model calls. **Conclusive.**
  - Priority: **needed** briefly (replay is not live evidence).
- **B2. Solver alone, standard goal, come_after only (finding 9)**
  - Result: at lines, the solver alone and the standard goal score 0 of 90 for every model, and come_after alone equals
    C for 4 of 6 models. At groups, the solver alone is at least C. **Conclusive** for these sets.
  - Priority: **needed**: C's gain at lines is recognising whom to come after.
- **B4. I34 Stage 0: R and E through C's clearance snap**
  - Result: fixes 168 and breaks 0. C's margin over the best of R and E shrinks to +14 to +72 (Gemini: 9).
    **Conclusive** as an exact replay. Stage 1 (R-BEV, 120 calls) not run.
  - Priority: **needed** as an ablation row.
- **B5-B6. I01 counterfactual repairs and calibration**
  - Result: perception repairs flip exactly at the bars (GLM 15, DeepSeek 8); perception ceilings +0.8 to +6.3 pp.
    Calibration recovers the real oracle-people gain pooled (share 1.00) but 2 of 5 models per model.
    **Suggestive.**
  - Priority: nice.
- **B7. A0: VLM facing accuracy**
  - Result: within 45°: Gemini 0.906 down to GLM 0.547. True facing flips no line failure. **Conclusive** as a
    measurement.
  - Priority: nice.

### C. Still scenes: metrics and statistics (RD)

- **C1. I14 scoreboard version 0** (frozen `results/still-scenes/metrics-and-statistics/metric-v0/METRIC_NOTE.md`)
  - Result: S2 and S4 fail; runners-up carry no knowledge beyond spread. **Conclusive negative.**
  - Priority: nice.
- **C2. I46 statistics and power**
  - Result: ICC 0.08-0.75; 252 records carry the information of 18-107 independent ones. Power 0.98-1.00 at 20 distinct
    layouts, 0.14-0.41 at 8. **Conclusive as method**; the power figures are optimistic.
  - Priority: **needed** (how to report).
- **C3. I14 metric list v1 draft**
  - Not agreed, so version 0 stays.
  - Priority: nice. Only D1, D2, D13 and D14 bear on the write-up.

### D. Perception and grounding on still scenes

- **D1. I03 depth-blob detector** (frozen)
  - Realistic recall 0.890 against a 0.95 bar; neither. **Conclusive-ish**, thin margin. Superseded by the I09
    detector.
  - Priority: not worth it.
- **D2. I03 Stage B blob replays**
  - Add-only (primary): fails, **conclusive negative**. Relocate and prune arms: **suggestive**, data-informed.
  - Priority: nice.
- **D3. I11 lift error vs choice error; the zero-call depth snap** (frozen)
  - Result: 20 lift and 26 choice errors. The snap fixes 0.45 of the oracle's gain against a 0.70 bar, with 0 breaks.
    Neither.
  - Priority: nice. Superseded by the person snap (D10) and marks (D12).
- **D4. I09 A1: detector plus pose model on development scenes** (frozen)
  - Result: recall 0.983, 0.071 false per image, facing 0.951. **Conclusive on development scenes, not blind.**
  - Priority: nice. Superseded by D6 and E2.
- **D5. I09 A2: arm D replay**
  - Result: Qwen 236, GLM 202, DeepSeek 199 against bars 232, 199, 190. **Suggestive.** Superseded by E2.
  - Priority: nice.
- **D6. I09 confirmation 1: unseen layouts** (seeds 200-212, 182 layouts; frozen)
  - Result: recall 0.981, facing 0.969, false people 0.242 per image against a 0.10 bar. **Conclusive**: A1 fails its
    false-people bar.
  - Priority: **needed** if the detector is in the write-up (it is, via E2). E2 shows those false people do no harm.
- **D7. I09 fix 2 and confirmation 2**
  - Result: 0.036 false per image on its fit-validate seeds. Confirmation 2 was frozen but never ran, and fix 2 can't be
    scored at current code. **Suggestive.**
  - Priority: nice (E2 uses A1).
- **D8. I09 live check (D-live)**
  - Frozen but never ran (low power); superseded by E2.
  - Priority: not worth it.
- **D9. Confidence from perception disagreement** (frozen)
  - Result: P1 AUROC 0.708 pooled; neither. S2 and S4 suggestive; near chance for API models.
  - Priority: not worth it.
- **D10. Person snap** (`group4/`; frozen `SNAP_FROZEN.md` 8251bf9e + amendments 1-3)
  - Data: development replay of 1,512 records (24 fixed, 0 broken, 25 of 27 in one scene). Frozen test: 24 new layouts
    (seeds 400-449), 8 primary occluded; plan call only; Qwen, GLM, DeepSeek x 3 trials (432 free calls).
  - Result:
    - H1 fails (gap 8.6 against a 10-point bar), so "not needed". **Conclusive** that the failure is rare on fresh
      layouts.
    - Post hoc: with the occluder's edge at the chest (2 layouts) the snap fixed 9 of 10 misses, with 0 breaks in 432.
      **Suggestive** that it fixes the case.
  - Priority: nice. A grounding fix, not social reasoning.
- **D11. Detector people (E1a, E1b, E2)**
  - **E1a**, 13 sim frames: recall 75/75, 0.08 false per image. **Quick probe**, plain margin. Not worth more.
  - **E1b**, development scenes (frozen `results/moving-lines/own-camera/detector-on-sim-frames/E1_FROZEN.md`):
    - Result: non-inferior on 4 models (Sonnet 124→132, DeepSeek 127=127, Qwen 131=131, GLM partial 63→73).
      **Suggestive**: the detector's own design data.
    - Superseded by E2.
  - **E2**, fresh static scenes (frozen `results/still-scenes/people-finder/E2_FROZEN.md` 38b2ef01):
    - Data: 40 layouts (seeds 500-504, 8 people rows) x 2 episodes x 2 trials. Three free models with bars; Sonnet,
      Haiku and Gemini as extras.
    - Limits: weak row-level tests, the "real" variant only, development types only.
    - Priority: **needed**, done. Widening it is nice (Part 1c).
- **D12. Numbered target marks** (`marks/`)
  - **M1, coverage and the perfect-pick ceiling** (frozen `M1_FROZEN.md`): the ceiling gate fails (+13 against +36);
    Sonnet −11, Haiku −17. **Conclusive.** Priority nice: one sentence.
  - **M2, harm on the free models** (frozen `M2_FROZEN.md`; Qwen, GLM, DeepSeek, x-test-3; 2,016 calls):
    - The frozen harm bar fails for all four. But it was judged against 0, not a redraw. Against the post hoc redraw
      floor the marks are inside noise for DeepSeek, Qwen and x-test-3.
    - **Conclusive as frozen but uninformative.** Priority nice: show the floor beside it.
  - **M2P, prompt vs drawn marks** (frozen; DeepSeek, 60 cells): the drawn labels pull the click. **Suggestive.**
    Not worth more.
  - **M2d, two-call design** (zero calls): no gain. **Quick probe.**
  - **M4, marks on sim first frames** (frozen; DeepSeek and GLM, 180 calls): both fail the go bar. **Conclusive** as a
    gate, small.

### E. Distributions and learned models (RD)

- **E1. I04 soft solver density** (frozen)
  - Result: worse than a Gaussian control (log score −1.563 vs −1.452); kill. **Conclusive negative.**
  - Priority: nice. It answers the proposal's distribution question.
- **E2'. C5 gates**
  - Result: gate 1 not met (22 generator configurations against about 30). **Conclusive.**
  - Priority: not worth it. Keep "22 templates" for sealed-set design.

### F. Crowds (RD)

- **F1. I06 first crowd probe** (frozen; oracle people and reference constraints)
  - Result: oracle C fails only at lines (absorbed bystanders); the noise separation misses the bar. Neither.
    **Suggestive.** There is no VLM run, so R and E are not compared.
  - To conclude: a VLM crowd run, R/E/C at 2-3 crowd levels (about 2,000 free calls).
  - Priority: **needed** if RQ2 (crowds) stays in the report; otherwise nice.
- **F2. I06 repair: facing gate (RF90)** (frozen, data-informed)
  - Result: fixes 29 of 54 (bar 27), 24 of them built in by how the scenes were made. **Suggestive.**
  - To conclude: replay the gate on runs s8-s17 (zero calls, minutes).
  - Priority: nice.

### G. Speed, fusion and untested still-scene ideas (RD)

- **G1. I21, concurrent people call**: replay only, −10 to −20% of goal-step time. **Quick probe.** Live check 80 free
  calls. Priority nice.
- **G2. I22, thinking per stage**: placeholder, old code, gate at its boundary. **Quick probe.** About 106 free calls.
  Priority nice.
- **G3. I02 fusion, I39 cascade, I44**: majority voting +1 to +5; the strong model alone beats every cascade.
  **Quick probes.** Priority not worth it.
- **G4. I08, solver-verified worked examples for constraints**: only the circular ceiling measured (+2 to +25).
  **Untested.** Priority nice (a lever for Haiku and GLM).
- **G5. I47, standard-goal geometry**: **untested**; targets 45 of C's 233 failures. A zero-call replay of about 8 h.
  Priority nice (the top zero-call gap).

One cell is scene x seed x model, with one episode per arm.

- **H1. Headline: CT2 and CT v1 against R1 on fresh seeds**
  - Data:
    - 5 straight-line scenes; Sonnet seeds 10-19 (50 cells), DeepSeek seeds 10-14 (25), x-test-3 seeds 10-12 (15).
    - E3 adds Haiku 5.5 and Gemini 3.8 (25 cells each).
  - Frozen: arms, scenes and strict scoring frozen before the fresh seeds (`CT2_FROZEN.md`, `SCORING_STRICT.md`).
    **There is no pass/fail bar for CT2 vs R1.**
  - Result (strict):
    - CT2 vs R1: Sonnet 43/2/5, +0.27; DeepSeek 19/3/3; x-test-3 9/0/6 (6/5/4 under primary); Haiku 19/2/4; Gemini 18/6/1.
    - CT v1 vs R1: Sonnet 40/3/7; DeepSeek 11/3/11, p 0.06.
  - Strength: **conclusive in practice** for Sonnet and DeepSeek (the counts are far outside the timing floor), but with
    no bar, and it rests on the side check (H22). Haiku strong. Gemini moderate (p about 0.02; weaker under primary).
    x-test-3 suggestive.
  - To conclude: Part 1 #9 (Qwen and GLM, frozen bars).
  - Priority: **needed**.
- **H2. CT2 vs CT v1 (CT2_FROZEN's primary)**
  - Result: Sonnet 32/8/10, DeepSeek 18/3/4. Two frozen predictions failed (cut-ins; where the gain sits). Post hoc,
    it is a 1.2 m band-edge effect (+0.008 and +0.033 at a 1.5 m top).
  - Strength: **conclusive at the frozen band, not robust** to it.
- **H3. CT2 ablations: 15-cell frozen (vac, rat), then all-cell post hoc (nora, norv, none, rat)**
  - Result: the 15-cell "no ratchet effect" was overturned at 49 cells. The ratchet carries the gain; re-admission and
    the vacated place carry none.
  - Strength: 15-cell **quick probe, superseded**; all-cell **suggestive-to-strong** (hashed predictions on known data).
  - Priority: nice. One paragraph, no runs.
- **H4. Move while thinking (CT2m, Rm)** (frozen `CT2m_FROZEN.md`)
  - Result: no gain on 3 models (Sonnet 12/27/11; DeepSeek 7/10/8; x-test-3 3/5/7). **Conclusive negative** for "no
    gain"; harm not shown.
  - Priority: **needed** as a negative result; no runs.
- **H5. CTs, re-solving only on settled tracks** (frozen `CTs_FROZEN.md`)
  - Result: Sonnet only, 15 cells: 5/7/3. Its redraw-spread prediction was never tested. **Quick probe**, superseded
    by the ratchet.
  - Priority: not worth it.
- **H6. Re-asking (Rq, Rq0, Cq)**
  - Data: seeds 0-2, 15 episodes each, on 4 models (Rq0 on 2); compared with CT v1; primary scoring.
  - Result:
    - CT ahead of Cq on all four, every p above 0.4. Calls 4 against 20-30.
    - Under strict rescoring (pooled, partial) Rq0 is about level with CT.
  - Strength: **quick probe.** Only the difference in calls is solid.
  - To conclude: Part 1 #10.
  - Priority: **needed**.
- **H7. Tracking a point without roles (RT, RTw)**
  - Data: seeds 0-2, 15 each; primary scoring.
  - Result:
    - DeepSeek: CT 0.18 vs RTw 0.08; the goal leaves the line in 11 of 15.
    - x-test-3 and Sonnet: not shown (RTw 0.18 vs CT 0.12; 6/7/2).
    - Strict, pooled: CT ahead.
  - Strength: **quick probe.**
  - To conclude: Part 1 #4 (zero calls).
  - Priority: **needed**.
- **H8. Navigation map (CTv)**
  - Result: no difference on 3 models (seeds 0-2, 15 each). **Quick probe**, consistent.
  - To conclude: CT2 without the map on cached fresh cells (zero calls).
  - Priority: nice.
- **H9. CT0, CTo, CTt (decomposition and oracle ceilings)**
  - Status: ran on seeds 0-2, never written up.
  - Recount (primary): CTt vs CT 12/2/1 (DeepSeek) and 8/1/6 (x-test-3); CT vs CT0 11/1/3 on Sonnet.
  - Strength: **quick probe, unreported.**
  - Priority: **needed** to score them (Part 1 #3). Extending them is nice.
- **H10. Detector tracking (CTd)** (frozen `results/moving-lines/own-camera/detector-on-sim-frames/e1d/E1D_FROZEN.md`, no bar)
  - Data: Sonnet and DeepSeek, 15 cells each.
  - Result: −0.018 and −0.017 (5/6/4, 5/4/6). Tracks matched 0.89-0.93; 67-71% of tracks stale. **Quick probe.**
  - To conclude: Part 1 #5 (zero calls).
  - Priority: **needed**.
- **H11. CTe, detector people plus tracking** (frozen `results/moving-lines/people-finder/first-check/E1C_FROZEN.md`, no bar)
  - Data: Sonnet only, 15 cells.
  - Result:
    - First answer about 2.2 s faster.
    - Quality vs CTd 9/4/2, inside the spread of near-identical arms.
    - Sonnet adds keep-out-of-shared-space in 12/15 vs a 13/44 base rate (p 0.0009).
  - Strength: latency mechanical; quality a **quick probe.**
  - Priority: nice (E2 carries the efficiency claim).
- **H12. Break-it cases (bent line, member stepping out, fast service)**
  - Data: n = 3 each; DeepSeek seeds 0-2, Sonnet seeds 10-12.
  - Result: Sonnet bent line CT2 0.46 vs CT 0.11; step-out 0 vs 5 cut-ins. Path memory never ablated. **Quick probe.**
  - Priority: nice, unless the report claims bent lines (E4 may cover them on still scenes).
- **H13. Failure analysis** (396 episodes, seeds 0-2, CT v1 era)
  - Result: late start 0.39, wrong first answer 0.39, stale 0.10, rules 0.07, scoring 0.01. A geometry rule picks the
    right people in 15/15, 15/15 and 13/13 first frames. **Suggestive.**
  - To conclude: re-run on the fresh CT2 and R1 episodes (zero calls).
  - Priority: nice.
- **H14. Timing (redraw) floor**
  - Result: CT varies by mean |d| about 0.10, CT2 by 0.05; CT2-none vs CT leans 14/24. Adequate as a reading aid.
  - Priority: **needed** in the report.
- **H15. Bystanders pilot, CT3** (frozen `BYSTANDERS_FROZEN.md`; DeepSeek, 15 cells)
  - Result: capture 208 → 9 s (bar met); share 5/2/8, not shown. **Conclusive for capture**, small.
  - Priority: **needed** as the pilot; done.
- **H16. Bystanders confirmation** (frozen `BYSTANDERS_CONFIRM_FROZEN.md`; DeepSeek, 60 cells, seeds 15-44)
  - Result: share 17/6/37, +0.018, p 0.035; capture 1,005 → 203 s. Both bars pass. **Conclusive**, small effect, one
    model.
  - To conclude more widely: a school model (about 240 calls, 2.5 sim-h).
  - Priority: nice (CT3F has replaced CT3).
- **H18. CT3 no-harm on the original scenes**: Sonnet 30 cells +0.015; DeepSeek 15 cells +0.007; 0 false drops.
  **Conclusive.** Priority **needed** (one line).
- **H19. CT3b**: 18/8/34 vs CT2, p 0.076. **Suggestive**, no gain. Priority not worth it.
- **H20. Geometry chain (Gl): do the model's roles matter?**
  - Result: Gl is captured in 51/60 cells. CT3 beats it 24/5/22 (DeepSeek) and 17/1/11 (Sonnet).
  - Strength: **conclusive on 2 models**, but the scenes were built so a 1 m chain fails.
  - Priority: **needed**: the only moving evidence that the model's roles matter.
- **H21. Sweep: geometry-only arm vs CT3** (sweep layouts, DeepSeek)
  - Result: CT3 beats geometry 34 to 5. The planned "model's added value along the ambiguity axis" plot was never
    drawn.
  - Priority: nice (zero calls).
- **H22. Scoring variants (side check, band top)**
  - Result: with the side check CT and CT2 beat R1 at every band top. Without it, Sonnet's gain goes to 0.00 at 2.0 m.
    **Conclusive** as a sensitivity analysis.
  - Priority: **needed** as a caveat.
- **H23. Cut-ins**
  - Result: Sonnet CT2 481 vs R1 162 samples (an exposure effect); DeepSeek 226 vs 504; CT3 more than CT2.
    **Suggestive.**
  - To conclude: Part 1 #6.
  - Priority: **needed**.
- **H24. Keep-up**
  - Result: Sonnet CT2 62/97 vs R1 4/12. It is counted only while joined, a selection effect. **Suggestive.**
  - Priority: nice (an unconditional paired measure, zero calls).
- **H25. x-test-3 latency**
  - Result: first answer C 43 s vs R 23 s; its deficit is latency (a counterfactual). **Suggestive.**
  - Priority: nice.
- **H26. Moving ideas never run** (re-ask triggers, expiry conditions, roles as arguments, speculative execution,
  gapped-line and two-line scenes, scenes outside queues)
  - No evidence.
  - Priority: not worth it before the report. Future work. State that every moving result is queue-only.

- **I1. CT4 vs CT3F under late steppers** (frozen `CT4_FROZEN.md` 2df880ff)
  - Data: DeepSeek; 72 sweep layouts; 6 arms.
  - Result: the claim fails, 14/14/1 on the 36 lag layouts. Reported: both new arms about double CT3 on right last; CT4
    recall 0.80-0.87 vs 0.51-0.64. **Conclusive negative.**
  - Priority: **needed** as a stated failed prediction (written up). No runs.
- **I3. No-harm of CT3F and CT4 on the original scenes** (DeepSeek, 40 cells; comparisons named before the run)
  - Result: CT3F beats CT3, 0.400 vs 0.337 (25/7/8, p 0.002), with fewer cut-ins; CT4 0.259.
    **Suggestive-to-conclusive**, one model.
  - Priority: nice (see Part 1 #11 for CT3F's own frozen test).
- **I4. Round 3: W0 and CT5** (predictions hashed blind; DeepSeek, 24 pilot layouts)
  - Result: the key prediction fails. CT3F 0.188, W0 0.072, CT4 0.068, CT5 0.036. **Quick probe (negative).**
  - Priority: not worth it (round 6 shows the gap is information, not decision).
- **I5. CT4P and CT4PJ pilot** (24 layouts)
  - Result: CT4PJ 0.182 vs CT3F 0.188. Overstated (winner's curse; see I6). **Suggestive.**
  - Priority: report only as the pilot behind I6.
- **I6. Confirmatory test: CT4PJ vs CT4P (H1) and vs CT3F (H2)** (frozen `round4/CONFIRM_CT4PJ.md`; 72 fresh
  layouts)
  - Result:
    - H1 fails on DeepSeek (27/16/23, p 0.126) and passes on Sonnet (p 0.033). Sonnet's pass flips at a 1.5 m band top.
    - Parity fails on both (CT3F ahead 34/17 and 44/18).
    - E3 adds Haiku 5.5 and Gemini 3.8 (description only): the same order CT3F > CT4PJ > CT4P, H1 not met (p 0.17 and
      0.085).
    - **Conclusive**: the commitment effect is about +0.05 and too small for the bar; CT3F stays best.
  - Priority: **needed**; done.
- **I7. Round 5: process belief and the walking-away signal** (offline, frozen bars `round5/ROUND5.md`)
  - Result: v1 fails check 1; v2-v4 fail the belief bars. The walking-away signal fires on 81% of ex-member bouts and
    1-6% of member bouts. **Quick probe (negative)**, offline only.
  - Priority: not worth it as a belief. The signal is future work.
- **I8. Round 6: why CT3F wins** (logs only; questions fixed before each step; 17 hashes)
  - Result: CT4PJ switches person 2x as often; freezing loses accuracy; no hold/switch rule reaches CT3F on fresh logs;
    CT3F knows it is first 2.4-2.6x as often. One correction published. **Suggestive.**
  - Priority: report as the explanation, labelled exploratory.
- **I9. Ceilings: OW, CT4OT, CT4O, CT4PJT, CT3FT** (cache-only; DeepSeek's answers only; readings fixed before
  totals: `round7/ROUND7.md`, `ROUND7_D.md`)
  - Result:
    - OW 0.640 (reachable with a following cost).
    - Gaps: place and follow +0.31 (mostly at the front); counter point +0.11; who +0.06 (model's counter) and +0.12
      (true counter).
    - CT3FT − CT3F +0.053 (27/23, p 0.67): outcome B at 1.2 m, C at 1.5 m.
    - **Suggestive** (descriptive, one model's answers).
  - To conclude: the same arms on Sonnet's, Haiku's and Gemini's cached answers (zero calls, about 6-8 sim-h).
  - Priority: nice, cheap (Part 1c).
- **I10. Band-edge sensitivity** (zero calls; frozen readings `ROUND7_B.md`)
  - Result: methods rise 0.02-0.03; CT3F's lead holds; only Sonnet's H1 sign test flips. **Conclusive** as a
    sensitivity analysis.
  - Priority: **needed** (one line); done.
- **I11. Not tried: the observed serving spot; walking away after commitment**
  - Motivation: the serving spot targets the two largest gaps (front place +0.25, counter point +0.11). Walking away's
    room over CT3F with the model's counter is about +0.01.
  - Never built.

- **J1. Latest models on lift and picture-taking** (Sonnet 5.5, Haiku 5.5, Gemini 3.8; 36 records per situation per
  method; `--final-test`)
  - Result (pointing / with examples / C):
    - lift: Sonnet 33/31/36, Haiku 7/7/32, Gemini 15/20/28;
    - photo: Sonnet 2/2/4, Haiku 0/1/9, Gemini 6/3/0.
  - Strength: **suggestive** for lift. Weak for photo, whose success rule is debatable.

### K. More models (E3; descriptive)

- **K1. Point-format calibration** (`e3_models/calib_*.txt`)
  - Result: 4 targets x 2 formats per model; the gaps are large, so the choice is clear. **Quick probe**, chosen on
    targets inside the scored scenes.
  - To conclude: re-calibrate on about 10 targets outside the scored set (about 20 calls a model).
  - Priority: nice; state as a caveat.
- **K2. Older models on still scenes** (7 models x 252 records x 3 methods; reproduction check exact)
  - Result (pointing / examples / C):
    - Gemini 3.6 Flash 123/142/232; 3.1 Flash Lite 131/122/227; 3.5 Flash Lite 102/101/215; 3 Flash 119/132/176;
    - Sonnet 5 139/144/194; Sonnet 4.6 166/162/162; Haiku 4.5 24/26/57.
    - C beats pointing on 6 of 7 (12 of 13 models in all).
  - Strength: **suggestive** (no layout test; development scenes).
  - To conclude: Part 1 #2 (zero calls).
  - Priority: **needed**.
- **K3. Haiku 5.5 and Gemini 3.8 on the moving headline**: see H1. Priority **needed** (exact p, labelled a
  replication).
- **K4. Haiku 5.5 and Gemini 3.8 on the final moving test**: see I6. Priority nice (one sentence).

### L. E4: new still-scene situations (`results/still-scenes/new-situations/design/E4_DESIGN.md`)

- **L1. Bent line and someone using a machine**
  - Status: 12 scenes rendered (seeds 600-602 x 4 situations); no calls. The design is written, but not hashed, and
    the region code is uncommitted.
  - Strength: no result yet.
  - To conclude: Part 1 #13.

### M. Ideas round 2 shortlist (`results/overview/ideas-round-2/SHORTLIST.md`): what was tried

| idea | tried? | where, and the strength |
|---|---|---|
| The answer as a hypothesis that motion tests (one role belief, CT4) | yes, extensively | I1-I9: conclusive negative against CT3F |
| CT3-fix (CT3F), the agreed comparator | yes; the best moving method | I3, I6: no frozen claim of its own (Part 1 #11) |
| Moving scenes outside queues | no | a limitation: every moving result is queue-only |
| Ambiguity sweep | yes, as the test-set generator | H21: the added-value plot never drawn |
| Is checking a role easier than listing roles? | no | not worth it |
| Event-based scoring and stress replays | partly | the stress replays did not transfer; a harm-ratio scorer was never written (nice, zero calls) |
| Distilling co-motion labels into a fast model | no | future work |
| Latency ideas (contingent prefix, arrivals as demonstrations) | no; related CT2m failed | future work |
| Ask again, closed question, when the evidence turns | no | not worth it |
| The accountable roster (sociology's top pick) | no | not worth it (a prompt change that must help most models) |
| Roles by predicate, voting, expiry conditions, "ask whether, not who" | dropped at the shortlist | voting dropped on zero-call evidence the error is a bias |
