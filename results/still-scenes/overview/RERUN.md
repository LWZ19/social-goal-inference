
Ledger for REPORT.md, written 2026-10-17. Every number below was recomputed from the recorded development runs through the step-0 harness (`docs/model_notes/tools/research_harness.py`), then checked a second time by an independent verifier with its own code. No model or network call was made and nothing was tuned.

- **RD** = `results/2026-10-17-beyond-vlm`. Script paths below are relative to RD.
- **Sets.** "Realistic" means the arms real, staffed and plain, 84 records per run, 3 trials, so 252 records per model and method. Old wording: s16b (Qwen, GLM, R and E), s19 (Qwen, GLM, C), s20 (DeepSeek). New wording: s21 (Qwen), s23 (GLM, DeepSeek, Haiku, Sonnet low, Gemini low).
- **Methods.** R = marked point (cot), E = R plus worked examples (fsc), C = people call, constraints call and solver (con).
- The three I01 deliverables are in `results/still-scenes/replay-harness/sets-and-definitions/README.md`. This ledger covers the report's own claims. Some finding-8 claims are checked in both places.

---

## Summary

### Counts

76 report claims were checked (one row per claim; a claim that the report repeats in several places is one row).

| Final verdict | Rows |
|---|---|
| confirmed | 53 (3 of them are this ledger's arithmetic on verified values: R22, R23, R76) |
| corrected | 19 |
| refuted | 0 |
| cannot reproduce | 3 (R50, R74, R75) |
| disputed | 0 |

### Most important corrections

3. **Finding 8, two tiers.** The C gaps between the top models do not come from target finding at nobody-near scenes (1 of 44 gap records). Qwen loses placement of the standard goal there; Gemini loses wrong targets at lines.
4. **Finding 8, groups.** The solver-alone and C group rows count goal steps but say "of 54". Over all 54 records: SA 46/49/44/53/47/54, C 46/48/34/53/47/54. The conclusion stands.
5. **Blob pilot (I03, I33).** Real and staffed recall depend on a split rule the pilot did not state (36–42 and 34–39 of 42). Placeholder is 42 of 42, not 41. Only about half the staff misses are behind counters. The facing null of 0.24–0.28 is not chance: chance is 0.50 for an axis taken modulo 180 deg.
6. **I11 places.** Place targets are 5 of 23 recoverable, not 0 of 23. Only a floor-only mask that cannot see the desk gives 0.
7. **Parked rows.** I48's 25 vs 14 is flag firings over seven runs, not the realistic set (19 vs 10 records). I16's 3.9% cannot be reproduced (3.3% under the natural definition). I44's 1.7× is 1.8× on fresh calls, and its "4 wins, 9 losses" and "136 vs 127" cannot be reproduced from listed runs.
8. **Cannot reproduce.** I22's "74 of 84 at 26.2 s" comes from s12, which is not on the whitelist.

### (A) Numbers in REPORT that must change

Each item gives the exact old fragment (with its REPORT line number at the time of writing) and the proposed new text.

**A1. Line 36 (1.1 item 1). Row R02.**
- New: "[re-run] Of GLM's 74 missed line members (s19 realistic, 83 line goal steps), 56 (76%) are people the people call never found. On failed goal steps only it is about 80% (51 of 64 at lines; 55 of 68 with group members)."

**A2. Line 38 (1.1 item 1). Row R03.**
- New: "[re-run] There are 15 wrong-target records" … "and the depth under the point landed 1.0–12.3 m away: 9 on the occluding person in front, 6 on the background behind (2 on the wall, 1 at its foot, 3 on the floor short of it). The re-run confirms these are edge near-misses, not choice errors: the point falls up to 14 px outside the silhouette, and an oracle snap-and-lift fixes 15 of 15."

**A3. Line 48 (1.1 item 4). Row R41.**

**A4. Line 65 (1.1 item 8, groups). Rows R14, R15.**
- Old: "scores 46 / 48 / 44 / 53 / 47 / 53 of 54, against C's 46 / 47 / 34 / 53 / 47 / 53."
- New: "scores 46 / 49 / 44 / 53 / 47 / 54 of 54, against C's 46 / 48 / 34 / 53 / 47 / 54 (all 54 records; GLM's and Gemini's one standard-path group record keeps its recorded success)."

**A5. Line 67 (1.1 item 8, two tiers). Row R21.**
- Old: "and the gaps in the totals come from target finding at the nobody-near scenes."
- New: "and the gaps in the totals do not come from target finding: only 1 of the 44 gap records is a wrong target at a nobody-near scene. Qwen's 14-record gap to Sonnet is 11 records at the nobody-near scenes, where the target is right in 10 and the standard goal built from Qwen's located point lands outside the region (9) or is unreachable (1). Haiku's 2 are placement at the lines. Gemini's 28 are wrong targets at the lines (34, offset by +7 at groups)."

**A6. Line 95 (1.1 item 8) and line 1270 (C5). Row R33.**
- Old (line 95): "fresh calls take 2–6 s for the API models and DeepSeek"
- New: "fresh calls take a median 1.0–6.4 s per stage for the API models and DeepSeek (DeepSeek mostly under 2 s; GLM's constraints call 8.6 s)"
- Old (line 1270): "Stage 23's other models take 2–6 s a call"
- New: "Stage 23's other models take a median 1.0–6.4 s a call per stage"

**A7. Line 618 (I03 Why). Row R37.**
- Old: "Counter staff can still be partly hidden: all 8 misses in the pilot were staff behind counters (pending whitelist re-run)."
- New: "Counter staff can still be partly hidden. In the whitelist re-run every miss was a staff member. In the settings that reproduce the pilot's 39 and 37, 4 of the 8 misses are hidden behind a counter-like surface and 4 are fully visible staff whose blob merges with tall structure."

**A8. Line 622 (I03 Why). Row R36.**
- New: "[re-run] Counting people inside the image, depth blobs find 34 of 34 on realistic_plain, 36–42 of 42 on real and 34–39 of 42 on staffed (the range spans split rules the pilot did not state; 39 and 37 reproduce under some of them), and 42 of 42 on placeholder dev scenes (eval_00–13 plus one dev seed of eval_14/15), not 41. Every non-staff person is found. Real and staffed hold 56 people each; 14 staff per set stand outside the image. With the pilot's thresholds, false body-sized blobs run about 0.6–2.6 per realistic image."

**A9. Line 628 (I03 Better than expected). Row R56.**
- New: "[re-run] The facing axis (modulo 180 deg) came within 45 deg for 0.68–0.70 of people on the realistic sets with a torso-band PCA (0.82–0.84 with all blob points), against a chance level of 0.50."

**A10. Lines 671–674 (I11 Why). Row R40.**
- New: "[re-run] Of 57 wrong-target C records, 53 have a point (4 GLM no_target records have none). Recoverable by an oracle snap-and-lift:" … "place targets: 5 of 23 (6 of 23 with the package's match mask; 0 of 23 only with a floor-only mask that cannot see the desk or kiosk)." (Person 15 of 15 and parcel 6 of 15 stand.)

**A11. Lines 721–724 (I21 Why). Rows R41–R44.**
- Old: "Qwen realistic: 62.9 to 50.6 s." New: "Qwen, three realistic sets: 61.2 to 49.3 s (n = 143; the real set alone, n = 48, gives 62.9 to 50.6 s)."
- Old: "GLM: 9.8 to 7.7 s." New: "GLM: 9.8 to 8.0 s (n = 141; real set 9.8 to 7.7 s)."
- Old: "54.6 to 48.7 s for Qwen." New: "53.2 to 47.4 s for Qwen (real set 54.6 to 48.7 s)."
- Old: "A2 saves at most 0.9 s with VLM people." New: "A2 saves 1.2 s for Qwen (up to 1.65 s on one set) and 0.9 s for GLM at s19; 2.8 s for Qwen at stage 21 and 0.6–3.2 s for the s23 models."

**A12. Line 725 (I21 Why). Row R45.**
- Old: "The two differ in accounting and possibly in subset. The replay script was not saved, so the exact subset is unconfirmed. Step 0 rewrites it."
- New: "The gap is mostly subset: 62.9 s is the real set alone (48 records). On the same 143 three-set records, provider time is 61.2 s and record time 61.6 s. The 0.4 s is wall time outside the calls; Qwen has no pacer time. The replay is now `results/still-scenes/speed/latency/latency.py`."

**A13. Line 756 (I22 Why). Row R50.**
- Old: "Qwen C: 65 (none) vs 74 (low) of 84, at 8.4 vs 26.2 s (brief section 3.1)."
- New: "Qwen C: 65 of 84 at none (s8, median record time 8.4 s), against 74 at low at 26.2 s (brief section 3.1; the low cell is s12, which is not on the whitelist and was not re-checked)."

**A14. Line 778 (I22 gate). Row R53.**
- New: "(49 of 52 vs 51 of 52 as pooled, but 11 none-side draws are counted twice. Without them it is 38 of 41 vs 40 of 41, or 37 of 40 vs 40 of 40, depending on which low draw trial 0 is paired with, so the gate sits at its boundary. The s8/s19_ph half also changes the context prompt.)"

**A15. Line 888 (I33 Why). Row R56.**
- Old: "came within 45 deg for 0.98 of people on placeholder scenes and 0.68–0.72 on realistic ones, against a null baseline of 0.24–0.28."
- New: "came within 45 deg for 0.97–0.98 of people on placeholder scenes and 0.68–0.70 on realistic ones (torso-band PCA, 1.0–1.6 m), against a chance level of 0.50 for an axis taken modulo 180 deg (0.52–0.59 with shuffled yaws). The 0.24–0.28 is a 'faces the camera' baseline, not chance, so the realistic gain over chance is about +0.2, not +0.45."

**A16. Line 962 (I02 Why). Row R59.**
- Old: "while the constraint kinds differ in 17 of 19 and 21 of 22. The variance sits in the constraints."

**A17. Line 1018 (I19 Why) and line 1682 (Appendix B, I19). Rows R02, R63.**
- Old (1018): "Most of GLM's member misses are people never found, so expanders cannot recover them. Member lists are only 1–3% of the visible constraints output, so the latency claim is dropped."
- New: "Most of GLM's member misses (56 of 74 at lines, 76%) are people never found, so expanders cannot recover them. Member lists are about 1% of the characters but about 4% (values) to 7% (with keys) of the constraints call's output tokens, so the latency claim stays dropped."
- Old (1682): "(1–3% of output)"
- New: "(about 4–7% of output tokens)"

**A18. Line 1063 (I39 Why). Row R67.**
- Old: "A gate built on disagreement between two samples reaches AUROC 0.73–0.75, and 216 of 252 at 50% escalation, against 201 for random escalation."
- New: "Disagreement with the other two trials (three samples) reaches AUROC 0.73 and 215–216 of 252 at 50% escalation, depending on a tie. Disagreement between two samples reaches AUROC 0.66–0.75, depending on which second trial is drawn, and 215–218 of 252. Random escalation gives 203.4 on average (201 for one seed)."

**A19. Line 1311 (I48 parked row) and line 1740 (Appendix B, I48). Row R71.**
- Old (1311): "Flags fire on more DeepSeek successes (25) than failures (14)."
- New: "On DeepSeek's realistic set, flags fire on 19 successes (10.4%) and 10 failures (14.3%); 25 vs 14 is flag firings pooled over all seven s20 DeepSeek C runs. For GLM the flags do separate failures (5.4% of successes against 30.3% of failures)."
- Old (1740): "The flags fire on successes"
- New: "The flags fire on a similar share of DeepSeek successes and failures (10.4% vs 14.3%), but separate GLM failures (5.4% vs 30.3%)"

**A20. Line 1315 (I16 parked row) and line 1676 (Appendix B, I16). Row R72.**
- Old (1315): "About 3.9% of straight-line approaches cross a forbidden polygon today."
- New: "The straight approach to 3.3% of C goal poses (25 of 752, s19/s20 realistic) crosses a forbidden polygon, and 0.3% for C successes (2 of 602); the earlier 3.9% could not be reproduced."
- New: "(3.3% of C goal poses cross a forbidden polygon, 0.3% of C successes [re-run])"

**A21. Line 1322 (I44 parked row). Rows R73–R75.**
- New: "In its favour, on the listed runs qwen3.6:35b leads Qwen 3.8 (both thinking off) with C by 69 vs 65 (convlm) and 71 vs 67 (conorc) of 84, not significant; the brief's 136 vs 127 of 168 pools an unlisted run and was not checked. It produces about 1.8× the output tokens per second on fresh calls (1.7× counting cached copies) and writes about 1.2× the tokens, so a fresh call takes about 0.6× as long. Against it: the '4 wins, 9 losses' could not be reproduced; against Qwen 3.8 low C on the same 84 placeholder records it is 0 wins, 10 losses, confounded by thinking level and prompt version."

**A22. Line 1648 (Appendix B, I02). Rows R58, R59, R60.**
- Old: "(+2/+5/+2 episodes, by the skeptic's arithmetic)"
- New: "(+2.0/+4.3/+2.0 episodes over the per-trial mean)"
- New: "Steelman [re-run]: the 6-sample pool has at least 3 successes in 78 of 84 (a strict majority, at least 4 of 6, in 56). The re-run's swap test puts most of the deciding variance in the people answer, not the constraints."

**A23. Line 1666 (Appendix B, I11). Rows R03, R40.**
- New: "Steelman [re-run]: the person wrong-targets are edge near-misses that an oracle snap-and-lift fixes (15 of 15). The one zero-call rule tried (a front-surface snap) fixes 8 and breaks 37, so a zero-call fix is not yet shown."

**A24. Line 1686 (Appendix B, I21). Row R41.**
- New: "Steelman [re-run]: 61.2 → 49.3 s over the three realistic sets (62.9 → 50.6 s on the real set)"

### (B) Wording that should change (the numbers stand)

- **Line 56 (R05).** "Exact sign-flip tests clustered by layout" → "Two-sided exact sign-flip tests clustered by layout (statistic: the sum of the 14 layout mean differences)".
- **Line 58 (R07).** "No comparison within one situation can reach p < 0.05." → "No two-sided comparison within lines or within groups can reach p < 0.05 (nobody-near, with 6 layouts, could reach 0.031)."
- **Line 532 (R34).** After "1.0 (DeepSeek)" add: "(post-hoc Monte Carlo power from the observed layout effects, B = 300; an exact-test resampling gives 0.60 / 0.92 / 1.00; optimistic for a sealed set)".
- **Line 60 (finding 8 header).** "Not yet re-checked by this track's harness" → "Re-checked by this track's harness on 17 Oct (RERUN.md)".
- **Line 62 (R10).** "(largest p = 0.016, permutation)" → "(largest p = 0.016, two-sided exact sign-flip test by layout)".
- **Line 64 (R13).** "With `come_after` it scores 87 of 87 (Sonnet), 78 of 80 (Haiku) and 85 of 90 (Qwen)." → add "(over the line records that state a `come_after`)".
- **Line 65 (R18).** "because the reference says people do not matter at `eval_10` and `eval_11`." → "because at `eval_10` and `eval_11` the standard goal already lies in the acceptable region (212 of 216 records pass there, against 40 of 108 at `eval_12`); the reference's people-matter flag is defined by that same test."
- **Line 84 (R27).** "All five lines are seen side-on at 4.4–6.2 m." → add "from the robot pose (4.2–6.0 m from the camera)".
- **Line 89 (R31).** After "(20 of 161 detections lie more than 1 m from anyone)" add "(distinct answers; the s23 answers are copies of s19's, so this is one sample)".
- **Line 90 (R32).** "look like a depth-lifting artefact at an occlusion edge (plausible, not proven)" → "are edge near-misses: the point falls 1–14 px outside the target's silhouette (0 of 13 centre pixels on the target), so the lift takes the occluder's or the background's depth; a snap fixes 13 of 13".
- **Line 94.** "The phantom people on the back wall support I03's pruning of listed people that no depth blob supports." → "The phantom people on the back wall are mostly mislifted real members (13 of 20 sit within 25 px of a member who was never found), which favours relocating people (I11's snap-and-lift, or an I03 audit that moves people) over pruning." Also "The low and too-close marked points support I34's Stage 0 snap control." → "The too-close marked points support I34's Stage 0 snap control (Gemini: 62 of 99 failures fixed); the low points of Sonnet and Haiku sit beside the line on free floor, where the snap fixes 0 of 126 and 8 of 164."
- **Line 96 (R26).** "the s23 GLM and DeepSeek C people answers repeat the s19/s20 draws" → add "and s21 Qwen's repeat s19's (143 of 144 identical). Within a stage, E and C also reuse R's plan and context answers, so the target is one draw shared by R, E and C."
- **Line 657 (I11 What).** "Classify every wrong-target C and E record" → "Classify every wrong-target C record (E's target points are the same cached answers in 756 of 756 pairs, so E adds no new targets)".
- **Line 730 (R47).** "C stays at about 1.8× E" → "C stays at about 1.8× E (1.77× per goal-step episode, 1.66× at stage 21)".
- **Line 758 (R51), optional.** "About 65–87%" → "About 73% (people) and 83% (constraints), estimated from text length because `reasoning_tokens` is null on every Qwen call,".
- **Line 865 (R55).** "stands 75–107 deg from facing the camera" → "stands 74.5–107.0 deg from facing the robot pose (74.1–107.3 deg from the camera; 16 distinct people)".
- **Line 963 (R60).** "at least 3 succeed in 78 of 84 episodes" → add "(ties count; a strict majority, at least 4 of 6, holds in 56)".
- **Line 981 (R61).** "The fallback accounts for only about 3, 4 and 6 records." → "The fallback accounts for 3 of 8, 4 of 5 and 6 of 14 misses (13 of 27); 14 misses have the surface fit working, so a fallback-only fix reaches at most 13."
- **Line 1075 (R69).** "Failed marked points at lines lie a median 0.34, 0.88 and 0.35 m from the region." → "Failed E points at lines (s16b/s20) lie a median 0.34, 0.88 and 0.35 m from the region (R: 0.51, 0.60 and 0.74 m)."
- **Line 1077 (R70).** "Failures of E at groups are mostly near misses that land on a body." → add "(28 of 44: within 0.3 m of the region and within 0.3 m of a body disc; 25 of 44 overlap a body; Qwen 3 of 5, GLM 13 of 16, DeepSeek 12 of 23)".

---

## Ledger

### Section 1.1: Summary findings

**R01. GLM line failures** (1.1 item 1; also I03 "Worse than expected", line 632)
- REPORT: "All 29 GLM line failures under C have `members_recall` below 1 (12 of them 0). `people_found_share` is below 1 in 25, and 8 state no `come_after` (brief section 3.5)." I03: "Of GLM's 29 line failures, 7 found nobody, 8 have no `come_after`, and 12 have recall 0."
- Re-run: 29 line failures out of 83 line goal steps (252 records). Recall below 1 in 29 (0 in 12). Found share below 1 in 25 (0 in 7). Unmatched people in 23. No `come_after` in 8. All 29 outside the region, 4 unreachable. DeepSeek (s20) for the brief's companion figure: 15 line failures, all with recall below 1 (n 89 line goal steps).
- Definition: s19 glm53 con, realistic. Line failure = waiting_line solver step with the right target, not a success, place outside the region or unreachable (brief section 3.5). Recorded measures.
- Verdict: **confirmed**.
- Scripts: `results/still-scenes/perception/perception-rerun/perception.py`; `results/still-scenes/perception/perception-rerun/verify/v_main.py`.

**R02. 81% never found** (1.1 item 1; I19 Why, line 1018)
- REPORT: "About 81% of GLM's missed group members are people the people call never found."
- Re-run: depends on the subset. All realistic line goal steps: 0.757 (56 of 74 missed members, 83 steps). Failed line goal steps: 0.797 (51 of 64). Failed goal steps with conversation-group members: 0.809 (55 of 68). Failed line steps with the ph arm pooled in: 0.812 (69 of 85). The 29 line failures of R01: 0.857 (42 of 49). Lines and groups, all steps: 0.733 (74 of 101). DeepSeek s20: 0.606 (20 of 33).
- Definition: the package's one-to-one matching (`sgi.run._matched_people`, 0.5 m). A missed member has no matched `come_after` number; never found means no detected person is matched to them. The recomputed found share and members recall equal the recorded ones in 141 of 141 and 83 of 83 records. Caution: only 17 of the 74 missed members have no detection at all within 1.0 m; most have a nearby detection that one-to-one matching gave to a neighbour. With "found" = any detection within 0.5 m, the share is 0.716.
- Scripts: `results/still-scenes/perception/perception-rerun/perception.py`, `p1_variants.py`, `p1_variants2.py`; `results/still-scenes/perception/perception-rerun/verify/v_p1alt.py`, `v_p1scan.py`.

**R03. Person wrong targets: 15 px, 1.0–12.3 m** (1.1 item 1)
- REPORT: "There are 15 wrong-target records whose target is a person. In all of them, the true target's pixels lie within 15 px of the model's point, and the depth under the point landed 1.0–12.3 m away, on an occluder or the wall."
- Re-run: 15 of 15 within 15 px (max 13.95 px; 8 within 3 px). Landed 1.03–12.34 m from the true person: 9 on the occluding person p3 in front (1.03–1.11 m), 6 behind the target (2 on the wall, 1 at the foot of the wall, 3 on the floor 1.65–3.07 m short of the wall). The centre pixel is off the target in 15 of 15. n = 15 (Qwen 5, GLM 2, DeepSeek 8; eval_12 13, eval_10 1, eval_11 1).
- Definition: the 15 person-target C wrong targets of R40. Body mask = each pixel's own depth lifted, inside a 0.30 m cylinder (0.25 and 0.35 m give the same result). "Landed" = horizontal distance from the recorded lifted point to the true person.
- Scripts: `results/still-scenes/perception/lifting-rerun/lifting.py`, `crops.py`; `results/still-scenes/perception/lifting-rerun/verify/v_lift.py`.

**R04. C at groups, old wording** (1.1 item 2)
- REPORT: "C at groups is 48, 51 and 34 of 54 (brief section 3.2)".
- Re-run: 48 / 51 / 34 of 54 (Qwen s19, GLM s19, DeepSeek s20, con, realistic).
- Verdict: **confirmed**.
- Scripts: `results/still-scenes/replay-harness/baselines/tables.py` (tables.txt, old wording); `results/still-scenes/replay-harness/baselines/verify/v3_tables.py`.

**R05. Layout sign-flip p values** (1.1 item 6; I14 line 532)
- Re-run: two-sided 0.03125 (2/64) / 0.0078125 (8/1024) / 0.00097656 (2/2048). One-sided values are half. Successes C vs E: 234 vs 214, 186 vs 145, 182 vs 120 of 252. Layouts favouring C / E / tied: 6/0/8, 9/1/4, 11/0/3. n = 252 pairs and 14 layouts per model. Rescoring all 1,512 records with the current scorer changes 0 verdicts.
- Definition: E = s16b fsc (Qwen, GLM), s20 fsc (DeepSeek); C = s19/s20 con; paired on (scene, episode, trial). Layout = scene id without the arm prefix. Statistic = sum of the 14 layout mean differences. All 2^14 sign patterns enumerated. clus.py uses |sum|, so the report's values are two-sided.
- Verdict: **confirmed**. The report should say "two-sided".
- Scripts: `results/still-scenes/metrics-and-statistics/stats-rerun/stats.py`; `results/still-scenes/metrics-and-statistics/stats-rerun/verify/v_stats.py`, `v_rescore.py`.

**R06. Qwen's 0.031 is the minimum** (1.1 item 6)
- REPORT: "Qwen's 0.031 is already the smallest value possible with 6 non-tied layouts (2/64). Any rescoring that ties or flips one layout pushes it above 0.05."
- Re-run: 6 non-tied layouts, all for C (eval_06 +3, eval_07 +6, eval_08 +1, eval_09 +6, eval_10 +2, eval_12 +2). p = 2/64 two-sided, the minimum. One tie gives 0.0625; one flip gives at least 0.0625.
- Verdict: **confirmed**. Holds only for the two-sided test (one-sided, one tie leaves 0.031).
- Scripts: as R05.

**R07. Within-situation p floors** (1.1 item 6)
- REPORT: "No comparison within one situation can reach p < 0.05. Lines have 5 layouts, so the smallest possible p is 0.0625. Groups have 3, so it is 0.25."
- Re-run: lines 5 layouts (eval_06, 07, 08, 09, 13), smallest two-sided p 0.0625; groups 3 (eval_10, 11, 12), 0.25. Nobody-near has 6 layouts, smallest p 0.031. One-sided, lines can reach 0.031, and DeepSeek's old-wording lines do.
- Scripts: as R05.

**R08. New-wording totals** (1.1 item 8)
- REPORT: "Successes of 252 at the new wording, 3 trials (R / E / C): Qwen 179 / 196 / 228 (stage 21); GLM 132 / 123 / 170; DeepSeek 120 / 126 / 185; Haiku 151 / 148 / 240; Sonnet (low) 178 / 178 / 242; Gemini (low) 164 / 173 / 214."
- Re-run: identical, all 18 numbers. n = 252 per cell; 0 not-ok results.
- Verdict: **confirmed** (checked by three clusters).
- Scripts: `results/still-scenes/replay-harness/split/split.py`, `results/still-scenes/replay-harness/baselines/i01_baselines.py`, `results/still-scenes/metrics-and-statistics/stats-rerun/stats.py`; verifiers in each `verify/`.

**R09. Records, rescore, solver replay** (1.1 item 8)
- REPORT: "4,536 records and 12,512 calls, all ok. The rescore reproduces every verdict, and the solver replay reproduces 860 of 860 C poses."
- Verdict: **confirmed** (12,512 calls not checked).
- Scripts: `results/still-scenes/replay-harness/split/split.py`; `results/still-scenes/replay-harness/split/verify/v_rescore.py`; `results/still-scenes/replay-harness/baselines/verify/v2_recorded.py`.

**R10. Largest p = 0.016** (1.1 item 8)
- REPORT: "C beats R and E for all six models at the layout level (largest p = 0.016, permutation)."
- Re-run: all 12 comparisons favour C. Largest two-sided exact sign-flip p = 0.015625 (2/128), for Qwen C vs R, GLM C vs R, Sonnet C vs E and Sonnet C vs R. Largest one-sided p 0.0078. Each of the four sits at the minimum for 7 non-tied layouts.
- Verdict: **confirmed**. Say "two-sided exact sign-flip test".
- Scripts: `results/still-scenes/metrics-and-statistics/stats-rerun/stats.py`; `results/still-scenes/metrics-and-statistics/stats-rerun/verify/v_stats.py`.

**R11. 0 of 84 cells** (1.1 item 8)
- REPORT: "C is below E in 0 of 84 model × layout cells."
- Re-run: 0 of 84 (36 tied, 48 above). For comparison C is below R in 2 of 84 (both DeepSeek, −1 each). Old wording: C below E in 1 of 42 (GLM eval_08_waiting_line_4, 12 vs 13).
- Verdict: **confirmed**.
- Scripts: as R10.

**R12. Solver alone at lines** (1.1 item 8)
- REPORT: "With no constraints the solver scores 0 of 90 at lines."
- Re-run: 0 of 90 for each of the six models (goal steps 90/88/90/90/89/82; the 2 + 1 + 8 standard-path line records of GLM, Sonnet and Gemini failed as recorded). Old wording: 0 of 90 for Qwen, GLM, DeepSeek.
- Verdict: **confirmed**.
- Scripts: `results/still-scenes/replay-harness/baselines/i01_baselines.py`; `results/still-scenes/replay-harness/baselines/verify/v1_compute.py`.

**R13. Solver with come_after** (1.1 item 8)
- REPORT: "With `come_after` it scores 87 of 87 (Sonnet), 78 of 80 (Haiku) and 85 of 90 (Qwen)."
- Re-run: 87 of 87, 78 of 80 and 85 of 90, both as recorded C and when re-solved with only the `come_after` constraints. The denominators are the line records that state a `come_after` (Sonnet 87 of 89 goal steps, Haiku 80 of 90, Qwen 90 of 90). Haiku's full C at lines is 79 of 90.
- Verdict: **confirmed**. Name the subset (list B).
- Scripts: as R12.

**R14. Solver alone at groups** (1.1 item 8)
- REPORT: "The solver alone, given the recorded people, scores 46 / 48 / 44 / 53 / 47 / 53 of 54".
- Re-run: all 54 group records, standard-path records keeping their recorded verdict: 46 / 49 / 44 / 53 / 47 / 54. Goal steps only: 46 / 48 / 44 / 53 / 47 / 53 of 54 / 53 / 54 / 54 / 54 / 53.
- Definition: H.resolve(rec, constraints=[]) on every C record with a recorded target and people; [] reaches the solver as no constraints (checked).
- Verdict: **corrected** (denominator). GLM and Gemini each have one group record that took the standard path and succeeded.
- Scripts: as R12, plus `results/still-scenes/replay-harness/baselines/check_empty.py`.

**R15. C at groups, new wording** (1.1 item 8)
- REPORT: "against C's 46 / 47 / 34 / 53 / 47 / 53."
- Re-run: all 54 records: 46 / 48 / 34 / 53 / 47 / 54. Goal steps only: 46 / 47 / 34 / 53 / 47 / 53.
- Verdict: **corrected** (same denominator slip).
- Scripts: as R12.

**R16. Constraints add nothing at groups** (1.1 item 8)
- REPORT: "**At groups, the constraints add nothing.**"
- Re-run: SA − C = 0 / +1 / +10 / 0 / 0 / 0 (Qwen, GLM, DeepSeek, Haiku, Sonnet, Gemini), on all 54 and on goal steps alike. Old wording: −1 / +1 / +11 (Qwen, GLM, DeepSeek).
- Verdict: **confirmed**. Caution: the GLM and DeepSeek old and new rows share people answers (134 of 134 and 144 of 144 identical), so DeepSeek's +10 and +11 are not two independent results.
- Scripts: as R12.

**R17. People-blind standard goal at groups** (1.1 item 8; line 93)
- REPORT: "The people-blind standard goal passes 42 / 41 / 38 / 48 / 43 / 40 of 54" and "(the people-blind goal passes 38–48 of 54)".
- Re-run: 42 / 41 / 38 / 48 / 43 / 40 of 54 (all 54 records). Goal steps only it would be 42 / 40 / 38 / 48 / 43 / 39. The standard goal recomputed for 963 standard-path records lands exactly on the recorded pose.
- Definition: `sgi.stages.goal.standard_goal` on the recorded target with the run's thresholds, scored with H.score_pose.
- Verdict: **confirmed**.
- Scripts: as R12.

**R18. Why the people-blind goal passes** (1.1 item 8)
- REPORT: "because the reference says people do not matter at `eval_10` and `eval_11`."
- Re-run: SG passes 212 of 216 at eval_10 + eval_11 and 40 of 108 at eval_12 (six models pooled). `sgi.gt.derive.reference_people_matter` is False for every eval_10 and eval_11 group episode in all arms, and True for every eval_12 and line episode.
- Verdict: **confirmed**. But that flag is defined as "the truth standard goal fails", so the clause restates the result (list B).
- Scripts: `results/still-scenes/replay-harness/baselines/tables.py`; `results/still-scenes/replay-harness/baselines/verify/v4b_refpm.py`.

**R19. R at lines given the right target** (1.1 item 8)
- REPORT: "Given the right target, R succeeds 57% (Gemini), 51% (Qwen), 28% (Sonnet), 21% (GLM), 15% (DeepSeek) and 7% (Haiku)."
- Re-run: 32/56 = 57.1%, 44/87 = 50.6%, 25/90 = 27.8%, 15/72 = 20.8%, 13/85 = 15.3%, 6/90 = 6.7% (n = right-target line records of 90).
- Verdict: **confirmed** (all six).
- Scripts: `results/still-scenes/replay-harness/split/split.py`; `results/still-scenes/replay-harness/split/verify/v_split.py`.

**R20. C at the people scenes** (1.1 item 8)
- REPORT: "On C, the people scenes give Sonnet 134, Haiku 132 and Qwen 131 of 144".
- Re-run: Sonnet 87 + 47 = 134, Haiku 79 + 53 = 132, Qwen 85 + 46 = 131 of 144 (lines 90 + groups 54).
- Verdict: **confirmed**. Qwen's s21 people answers are copies of s19's (R26).
- Scripts: as R19.

**R21. Where the C gaps come from** (1.1 item 8)
- REPORT: "and the gaps in the totals come from target finding at the nobody-near scenes."
- Re-run: C gap to Sonnet (242). Qwen −14 = nobody-near −11 + lines −2 + groups −1. Of Qwen's 11 nobody-near failures, 1 is a wrong target, 9 are right-target standard goals 0.055–0.587 m outside the region and 1 is in the region but unreachable. Haiku −2 = lines −8 + groups +6. Gemini −28 = nobody-near −4 (right target, 0.033–0.043 m outside) + lines −31 (34 wrong targets) + groups +7. Over all 44 gap records, 1 is a wrong target at a nobody-near scene.
- Definition: target found = score.target_correct (as in decomp.py); failures split in order wrong target, out of region, unreachable, heading.
- Scripts: `results/still-scenes/replay-harness/split/split.py`, `inspect2.py`; `results/still-scenes/replay-harness/split/verify/v_split.py`, `v_extra.py`.

**R22. The 0.50–0.55 m strip counts** (1.1 item 8)
- REPORT: "Of the in-region, right-heading but unreachable R/E poses, Gemini's 22 of 24 and Qwen's 11 of 15 overlap the person's body by up to 5 cm (`score.min_body_gap_m` between −0.05 and 0); the rest sit at a gap of 0.00–0.02."
- Re-run (verifier's per-method counts, summed here): Gemini R 13 + E 9 = 22 in the strip, of 14 + 10 = 24 in-region unreachable. Qwen R 5 + E 6 = 11, of 7 + 8 = 15 (the 15 includes 1 R and 1 E record at nobody-near scenes). The rest of those listed for Gemini and Qwen sit at +0.002 to +0.015 m.
- Verdict: **confirmed** (ledger arithmetic on the verifier's counts). The "12 R/E verdicts" figure was not checked.
- Scripts: `results/still-scenes/replay-harness/split/verify/v_extra.py` (v_extra_out.txt).

**R23. Region table, adopted row** (1.1 item 8)
- REPORT: "adopted (0.5, 1.2, 0.55) | +32 | +38 | +59 | +89 | +64 | +41".
- Re-run: C − max(R, E) from the confirmed totals (R08): +32, +38, +59, +89, +64, +41.
- Verdict: **confirmed** (ledger arithmetic). The other rows of the table (far 1.5, 2.0, 1.0; near 0.55) were not re-run.

**R24. Sonnet did not think** (1.1 item 8)
- REPORT: "Sonnet at "low" reported 0 reasoning tokens in every call, so it did not think."
- Re-run: `usage.reasoning_tokens` is 0 on every Sonnet call in s23. It is positive only for Gemini in s23 (925 of 2,052 calls, 62% of output tokens) and GLM at low in s10.
- Verdict: **confirmed**.
- Scripts: `results/still-scenes/thinking-levels/thinking.py`; `results/still-scenes/thinking-levels/verify/v2_extra.py`.

**R25. E gains within one batch** (1.1 item 8)
- REPORT: "E gains nothing within a single batch (pooled +3); only Qwen's 9-trial pool gives +24".
- Re-run: E − R at s23: GLM −9, DeepSeek +6, Haiku −3, Sonnet 0, Gemini +9, sum +3 (Qwen s21 +17). Qwen 9-draw pool: R 542, E 614 of 756, so E − R = +72 over 9 draws = +24 per 252.
- Verdict: **confirmed**. The Holm statement was not checked.
- Scripts: `results/still-scenes/replay-harness/split/split.py`; `results/still-scenes/replay-harness/split/verify/v_split.py`.

**R26. Cached people answers** (1.1 item 8, lines 83 and 96)
- REPORT: "The gateway models' C people answers are cache copies of s19/s20, not fresh draws." and "the s23 GLM and DeepSeek C people answers repeat the s19/s20 draws".
- Re-run: s23 GLM people lists equal s19's in 134 of 134 paired records (people calls cached 141 of 141); DeepSeek equals s20 in 144 of 144 (149 of 151 cached). Also s21 Qwen's C people are cached 144 of 144 and 143 of 144 identical to s19 for the same key. Within a stage, E and C plan and context calls are cache hits of R's (252 of 252), so the target is identical across R, E and C.
- Verdict: **confirmed**. Line 96 should also name Qwen s21 (list B).
- Scripts: `results/still-scenes/combining-models/fusion/fusion.py`; `results/still-scenes/combining-models/fusion/verify/v8_people_match.py`; `results/still-scenes/replay-harness/split/verify/v_cache.py`, `v_cache2.py`; `results/still-scenes/combining-models/cascade/verify/cache_check.py`.

**R27. Lines seen side-on at 4.4–6.2 m** (1.1 item 8)
- REPORT: "All five lines are seen side-on at 4.4–6.2 m."
- Re-run: 5 distinct line layouts (15 scene instances). Line axis 85.3–89.7 deg from the camera ray. Distance from the robot base pose 4.385–6.229 m (n = 48 member instances, 16 distinct); from the camera optical centre 4.19–6.04 m. All members project into the image.
- Verdict: **confirmed**. The numbers match the robot pose, 0.2 m behind the camera.
- Scripts: `results/still-scenes/speed/profile/profile_angles.py`; `results/still-scenes/speed/profile/verify/v_profile.py`.

**R28. Sonnet's R point lands low** (1.1 item 8, mechanisms)
- REPORT: "Sonnet's R point lands a median 54 px low, 1–1.7 m toward the camera, although its words say "behind"."
- Re-run: Sonnet R line marked points with the right target (n 89): median 54.6 px below the region's representative point; ground range median −1.35 m (IQR −1.61 to −1.23). Failures (n 64): 65 px, −1.47 m. All 64 failures lie more than 0.3 m to the camera's side of the line axis (median 1.38 m). Against the nearest region point the offset is 2 px (all 89) or 9 px (failures): the points sit beside the line, not short of it. Not unique to Sonnet: GLM 41 px, Haiku 45 px, Qwen 25 px.
- Verdict: **confirmed**. "Its words say behind" was not checked.
- Scripts: `results/still-scenes/depth-fix/first-check/snap.py`, `summarize.py` (T5a, T5b); `results/still-scenes/depth-fix/first-check/verify/v4_extra.py`.

**R29. Haiku picks the counter end** (1.1 item 8, mechanisms)
- REPORT: "Haiku picks the counter or the front of the line."
- Re-run: Haiku R line failures with the right target (n 84): 75 between the target anchor and the first member along the line axis, 5 more in the front half (median t 0.06), about 1.2 m off the axis toward the camera; 64 are more than 1 m from the region. E (n 80): 49 at the counter end, 14 in the front half.
- Verdict: **confirmed** (along the axis; by straight-line radius only 9 of 84 are nearer the anchor than the first member).
- Scripts: as R28.

**R30. Gemini stands too close** (1.1 item 8, mechanisms)
- REPORT: "Gemini picks the right spot but stands too close (median 0.61 m)."
- Re-run: Gemini R line marked points with the right target (n 56): median 0.613 m from the last member (E 0.621, n 56). Successes 0.65 (R) / 0.68 (E); failures 0.53 / 0.53. Of 45 R + E line failures, 35 are behind the last member, and 34 of those are unreachable (14 under 0.5 m, 18 at 0.50–0.55 m).
- Verdict: **confirmed**. "Too close" is carried by the failures (0.53 m).
- Scripts: as R28.

**R31. GLM's phantom people** (1.1 item 8, mechanisms)
- REPORT: "GLM's C at lines suffers phantom people lifted onto the back wall (20 of 161 detections lie more than 1 m from anyone), and the solver then queues behind them."
- Re-run: 20 of 161 detections over 45 distinct line people answers lie more than 1 m from every true person. Per record: 39 of 310 at 88 line goal steps (s23); 34 of 296 at 83 (s19). All 39 far detections lie within 0.3 m of the wall along the camera ray. 36 of 39 are listed under `come_after`; in 23 of 24 records listing one, it is the farthest listed person, the one the solver queues behind. 26 of 27 records holding a phantom fail. DeepSeek 8 of 166; Qwen s19 2 of 165.
- Verdict: **confirmed**. The s23 answers are s19 copies, so this is one sample, not a replication.
- Scripts: `results/still-scenes/perception/perception-rerun/perception.py`, `phantom_pixels.py`; `results/still-scenes/perception/perception-rerun/verify/v_main.py`, `v_n6.py`.

**R32. eval_12 artefact** (1.1 item 8, mechanisms)
- REPORT: "At `eval_12`, the wrong targets look like a depth-lifting artefact at an occlusion edge (plausible, not proven)."
- Re-run: 13 C wrong targets at eval_12 (Qwen 5, GLM 2, DeepSeek 6). All 13 points sit 1–14 px outside the red person's silhouette (7 within 3 px). 9 are on the left edge, where the lift lands on the occluder p3; 4 are on the right edge, where it lands 10–12.3 m away. The centre pixel is on the target in 0 of 13. An oracle snap fixes 13 of 13, and so does the pipeline's own lift at the snapped pixel.
- Verdict: **confirmed**, with a refinement: it is pointing imprecision at a depth edge, not the window median smearing an on-target pixel.
- Scripts: `results/still-scenes/perception/lifting-rerun/lifting.py`, `crops.py`; `results/still-scenes/perception/lifting-rerun/verify/v_lift.py`.

**R33. Fresh calls take 2–6 s** (1.1 item 8, line 95; C5 line 1270)
- REPORT: "fresh calls take 2–6 s for the API models and DeepSeek, and Qwen's constraints call 35.8 s."
- Re-run: s23 fresh-call medians per stage span 1.0–6.4 s for DeepSeek and the API models (DeepSeek 1.0–3.1 s; Anthropic 1.6–4.6 s; Gemini's R goal call 6.4 s). GLM's constraints call is 8.6 s, about 2× its s19 value. Means run 1.5–2× the medians for the Anthropic models. Full table in "New rows". Qwen's 35.8 s is confirmed in R54.
- Scripts: `results/still-scenes/speed/latency/latency.py`; `results/still-scenes/speed/latency/verify/stages.py`.

### Section 4.2: Tier (a)

**R34. Power at 14 fresh layouts** (I14, line 532)
- REPORT: "Power at 14 fresh layouts is 0.62 (Qwen), 0.91 (GLM), 1.0 (DeepSeek)."
- Re-run: a seeded replay of clus.py's loop gives exactly 0.62 / 0.91 / 1.0. An exact-test resampling design (10,000 draws) gives two-sided 0.60 / 0.92 / 1.00 (Qwen 0.603 and 0.599 under two seeds) and one-sided 0.79 / 0.97 / 1.00.
- Definition: clus.py: random.seed(0); B = 300 resamples of layout means; 400 Monte Carlo sign flips; reject at p < 0.05. SE about 0.028 at 0.62.
- Verdict: **confirmed**. Post-hoc and optimistic: it resamples the observed effects of the same 14 layouts.
- Scripts: `results/still-scenes/metrics-and-statistics/stats-rerun/stats.py`; `results/still-scenes/metrics-and-statistics/stats-rerun/verify/v_stats.py`.

**R35. GLM people found and member recall** (I03 Why; I19 Why)
- REPORT: "GLM finds 0.826 of people (lines 0.774). Its single-person line: found 0.529, C 3 of 18 (brief section 3.4)." and "GLM members recall is 0.661 (brief section 3.4)."
- Re-run: 0.826 (n 141 goal steps); lines 0.774 (n 83); single-person line found 0.529 (n 17), C 3 of 18; members recall 0.661 (n 83).
- Verdict: **confirmed**.
- Scripts: `results/still-scenes/perception/perception-rerun/i03_why.py`; `results/still-scenes/perception/perception-rerun/verify/v_main.py`.

**R36. Blob pilot recall** (I03 Why, line 622)
- REPORT: "The pilot found 34 of 34 people on realistic_plain, 39 of 42 on real, 37 of 42 on staffed and 41 of 42 on placeholder dev scenes."
- Re-run: in-image people, across split rules not stated by the pilot: plain 34 of 34 always; real 36–42 of 42; staffed 34–39 of 42; placeholder 42 of 42 always (eval_00–13 plus any one dev seed of eval_14/15; 58 of 58 over three seeds). The pilot's 39 and 37 reproduce under some settings (for example watershed r = 0.25 m with match 0.6 m). Real and staffed hold 56 people each; 14 staff per set are outside the image. Every non-staff person is found under every rule.
- Definition: lift depth, keep 0.15–2.1 m, 5 cm max-height map, 8-connected components, split at local maxima (rule varied), keep top 1.2–2.0 m and footprint 0.2–0.8 m, greedy one-to-one match within 0.25 / 0.4 / 0.6 m. 68 distinct development scenes.
- Verdict: **corrected**. The placeholder miss does not appear on any whitelisted subset.
- Scripts: `results/still-scenes/perception/depth-blob-pilot/blobpilot.py`, `diag.py`; `results/still-scenes/perception/depth-blob-pilot/verify/v_blobs.py`, `v_match.py`.

**R37. All 8 misses were staff behind counters** (I03 Why, line 618)
- REPORT: "all 8 misses in the pilot were staff behind counters (pending whitelist re-run)."
- Re-run: every miss under every rule is a staff member (the counter staff s0); no non-staff person is missed. In the settings that give exactly 8 misses (and reproduce 39 and 37), 4 have the lower body hidden by a nearer surface (counter-like) and 4 are fully visible staff whose sub-blob merges with tall structure (top 2.0–2.1 m, footprint up to 1.6 m).
- Verdict: **corrected**. "Staff" holds; "behind counters" holds for half. The report's 8 also leaves out its own placeholder miss.
- Scripts: as R36.

**R38. Blob pilot denominators** (I03, implicit "of 42")
- Re-run: truth people per set: plain 34; real 56 (22 staff); staffed 56 (22 staff); placeholder eval_00–13 34; 8 per dev seed. In-image: 34 / 42 / 42 / 34 / 8.
- Verdict: **confirmed** (the pilot's 42 is the in-image count).
- Scripts: `results/still-scenes/perception/depth-blob-pilot/blobpilot.py`; `results/still-scenes/perception/depth-blob-pilot/verify/v_list.py`, `v_blobs.py`.

**R39. Which 57 wrong targets** (I11 Why)
- REPORT: "Wrong target: Qwen 7 (5 in the group of 4), GLM 24, DeepSeek 22." and "Over 57 wrong-target C records".
- Re-run: 57 = Qwen 7 + GLM 28 + DeepSeek 22 with target_correct False (s19/s20 con, realistic). 53 have a point (GLM 24); the other 4 are GLM no_target with no point. Qwen has 5 at eval_12 (the group of 4). Classes of the 53: person 15, parcel 15, place 23. The placeholder set would add 15.
- Verdict: **confirmed**. The report mixes the two denominators (57 and 53).
- Scripts: `results/still-scenes/perception/lifting-rerun/lifting.py`; `results/still-scenes/perception/lifting-rerun/verify/v_counts.py`.

**R40. Recoverable by lifting: 15/15, 6/15, 0/23** (I11 Why)
- REPORT: "person targets: 15 of 15 recoverable by lifting; parcel: 6 of 15; place targets: 0 of 23."
- Re-run: person 15 of 15, parcel 6 of 15, place 5 of 23 (6 of 23 with the package's match mask; 0 of 23 only with a floor-only footprint mask). Radius sweep (person / parcel / place): r3 8/0/2, r10 12/5/5, r15 15/6/5, r30 15/11/7. The 5 places are near-misses 1.5–8.6 px off the kiosk or desk silhouette whose lift lands 3.4–4.5 m behind.
- Definition: recoverable = a body-mask pixel within 15 px of the recorded pixel AND the oracle snap (nearest body-mask pixel, median body depth in its 5×5 window) grounds the true target under `sgi.evaluate.match_target`. H.target_at reproduces every recorded lift exactly.
- Verdict: **corrected** (place only).
- Scripts: `results/still-scenes/perception/lifting-rerun/lifting.py`; `results/still-scenes/perception/lifting-rerun/verify/v_lift.py`.

**R41. Concurrency replay, Qwen** (I21 Why; 1.1 item 4; Appendix B I21)
- REPORT: "Qwen realistic: 62.9 to 50.6 s." and "cuts a Qwen C goal step from 62.9 to 50.6 s in replay".
- Re-run: s19 real set alone (n 48): 62.9 → 50.6 s (−19.6%). Three realistic sets (n 143): 61.2 → 49.3 s (−19.5%), medians 58.8 → 46.6. Staffed 61.5 → 49.0 (n 47), plain 59.1 → 48.2 (n 48). Mean pre 13.4 s, people 16.0 s, constraints 31.7 s; people slower than pre in 102 of 143.
- Definition: cp.py port. Per record with a people and a constraints call: sums of call latency_s per stage. Serial = plan + locate + context + people + constraints; concurrent = max(plan + locate + context, people) + constraints. All s19 calls are cache hits carrying the original calls' latencies.
- Verdict: **corrected** (subset). cp.py prints one line per folder; the quoted line is s19_real.
- Scripts: `results/still-scenes/speed/latency/latency.py`; `results/still-scenes/speed/latency/verify/verify.py`.

**R42. Concurrency replay, GLM** (I21 Why)
- REPORT: "GLM: 9.8 to 7.7 s."
- Re-run: real set (n 50) 9.8 → 7.7 s (−21.5%). Three sets (n 141): 9.8 → 8.0 s (−19.0%). Staffed 10.8 → 8.7, plain 8.9 → 7.4. Mean pre 3.1 s, people 2.1 s; people slower than pre in 28 of 141.
- Verdict: **corrected** (subset).
- Scripts: as R41.

**R43. Replay with people cached per scene** (I21 Why; 1.1 item 4)
- REPORT: "As the benchmark runs, with people answers cached per scene: 54.6 to 48.7 s for Qwen." and "the saving is about 10–13%".
- Re-run: real set 54.6 → 48.7 s (−11.0%, n 48, 24 firsts). Three sets 53.2 → 47.4 s (−10.9%, n 143, 72 firsts); Qwen per set 10.5–11.3%. GLM pooled 8.9 → 7.9 (−11.2%), per set 9.2–12.5%.
- Definition: the first record of each (scene, trial) pays serial or concurrent; later records of that (scene, trial) pay plan + context + constraints in both columns. The data agree with that sharing unit.
- Verdict: **corrected** (the absolute numbers are the real set; the 10–13% holds).
- Scripts: as R41.

**R44. A2 saves at most 0.9 s** (I21 Why)
- REPORT: "A2 saves at most 0.9 s with VLM people."
- Re-run: real set Qwen 0.89 s, GLM 0.62 s. Three sets at s19: Qwen 0.89 / 0.97 / 1.65 s, pooled 1.17 s (A1 49.25, A2 48.08, n 143); GLM 0.62 / 1.05 / 1.06, pooled 0.90 s (n 141). Other s19 runs 0.47–1.14 s; s20 DeepSeek 0.81–1.04 s; s21 Qwen 2.83 s (n 144); s23 DeepSeek 0.56, Haiku 1.70, Sonnet 2.24, GLM 2.26, Gemini 3.15 s.
- Definition: cp2.py port. A1 = max(p + x, pe) + k; A2 = max(p, pe) + max(x, k); p plan, x context, pe people, k constraints.
- Verdict: **corrected**. The decision to drop A2 still holds for Qwen at s19 (about 2.4% of A1).
- Scripts: as R41.

**R45. What 62.9 s measures; the brief's 61.6 s** (I21 Why, line 725)
- REPORT: "The brief's 61.6 s is a different quantity: the mean record `latency_s` (pacer waits included) over all realistic goal-step episodes (brief section 3.6). The two differ in accounting and possibly in subset."
- Re-run: 61.6 s (median 59.3) reproduces exactly from s19 {real, staffed, plain} qwen38_low con, n = 143 goal-step records. measures.latency_s = sum(timings_s) in 10,896 of 10,896 listed records. Provider-only time on the same 143 records is 61.2 s; the 0.4 s gap is wall time outside the calls. Qwen has 0.00 s of pacer time (GLM 1.29 s per goal-step record: 11.6 vs 9.8 s). Real set alone: record 63.3 s, provider 62.9 s.
- Verdict: **confirmed** for 61.6 s and its definition. The sentence "differ in accounting and possibly in subset" is the reverse: the gap is mostly subset (list A12).
- Scripts: `results/still-scenes/speed/latency/latency.py`; `results/still-scenes/speed/latency/verify/verify.py`, `pacer.py`.

**R46. Cached calls carry the original latency** (I21 / step-0 definition)
- Claim checked: a cached call carries the latency of the call that first produced it, and measures.latency_s counts cached calls.
- Re-run: 0 mismatches in latency_s, latency_total_s and usage across every chain checked (s21 fsc/con plan vs cot 504 of 504; context 502 of 502; s23 plan/context vs cot; s21 people = s19 144 of 144; s23 GLM people = s19 141 of 141; s23 DeepSeek people = s20 149 of 149; s19 plan/context = s16b cot 252 of 252). Record time minus all calls' latency_total_s: median 0.39 s (s19 Qwen), against 44.9 s if cached calls were left out.
- Scripts: `results/still-scenes/speed/latency/explore2.py`; `results/still-scenes/speed/latency/verify/l7_keyfix.py`.

**R47. About −20% and C at 1.8× E** (I21 Payoff)
- Re-run: −19.5% per goal-step record over three sets (n 143); −19.6% on the real set. Over all 252 C records, including 109 standard-goal records, −17.1% (39.5 → 32.7 s). E (s16b fsc, 143 goal-step records) serial 27.9 s; C concurrent / E = 49.3 / 27.9 = 1.77× (same 143 episodes); 1.73× on the real set; serial C/E 2.19×. Stage 21: 57.7 / 34.7 = 1.66×.
- Verdict: **confirmed**. "Per deployed episode" means per goal-step episode with no people cache.
- Scripts: as R41.

**R48. Call times and tokens** (I22 Why; I21 Why; Programme C; C1)
- REPORT: "Constraints call 31.7 s and 2,351 tokens; people call 16.0 s and 1,199 tokens (brief section 3.6)."
- Re-run: s19 realistic Qwen C (252 records): people 143 calls, mean 16.03 s (median 17.0), 1,199.5 tokens; constraints 143 calls, 31.70 s (median 29.4), 2,351.9 tokens. Plan 6.4 s / 443 tokens, context 6.0 s / 417 tokens.
- Verdict: **confirmed** (2,351 is 2,351.9 truncated).
- Scripts: `results/still-scenes/thinking-levels/thinking.py`; `results/still-scenes/thinking-levels/verify/v1_main.py`.

**R49. E's 28.1 s and its 14.5 s goal call** (I22 Payoff; 1.1 item 4)
- REPORT: "E's 28.1 s is at low thinking, and E's goal call (14.5 s at low; brief section 3.6)".
- Re-run: Qwen R and E goal-step record time 28.1 s (s16b); E's goal call mean 14.5 s (median 13.4), 143 goal-step records.
- Verdict: **confirmed**.
- Scripts: `results/still-scenes/speed/latency/latency.py`; `results/still-scenes/speed/latency/verify/verify.py`.

**R50. Qwen C at none vs low** (I22 Why, line 756)
- REPORT: "Qwen C: 65 (none) vs 74 (low) of 84, at 8.4 vs 26.2 s (brief section 3.1)."
- Re-run: none half: s8_convlm_qwen38 gives 65 of 84, median record time 8.4 s (mean 8.9). Low half: the 3-trial low cell is s12_qwen38_low_con, which the harness does not list (H.runs(stage='s12') is empty). Nearest listed: s10_qwen38_low_con 27 of 28 (median 31.7 s); s19_ph_qwen38_low_con 79 of 84 (median 34.1 s, later prompt and solver).
- Verdict: **cannot reproduce** (the low half). 8.4 is a median, so 26.2 is probably a median too.
- Scripts: `results/still-scenes/thinking-levels/thinking.py`; `results/still-scenes/thinking-levels/verify/v1_main.py`, `v3_cfg.py`.

**R51. 65–87% hidden thinking** (I22 Why)
- REPORT: "About 65–87% of Qwen's output tokens in these stages is hidden thinking."
- Re-run: `reasoning_tokens` is null on all 9,338 Qwen 3.8 calls, so the share is estimated from text length. People: 73.1–74.7% across s10, s19 ph, s19 realistic (n 72) and s21 realistic (n 72). Constraints: 82.7–83.8% (n up to 144). Per-call IQR: people 67–80%, constraints 78–86%. The s10 none/low token ratio gives 77.4% and 81.8% (12 pairs each).
- Definition: visible tokens = len(raw_text) / characters per token, calibrated on s8 at effort none (people 2.56, constraints 3.71). Assumes none has no hidden thinking; if it has some, these are lower bounds.
- Verdict: **confirmed** (loose; about 73% and 83% is the supportable figure).
- Scripts: `results/still-scenes/thinking-levels/thinking.py`; `results/still-scenes/thinking-levels/verify/v1_main.py`.

**R52. 11 of 11 vs 11 of 11** (I22 Why)
- REPORT: "On s10 (one trial, 28 records), right-target goal steps succeeded 11 of 11 at none and 11 of 11 at low. Most of the all-none loss came from plan and context."
- Re-run: matched right-target goal steps 11 of 11 vs 11 of 11 (unmatched 12 of 12 vs 15 of 15). Records 21 of 28 vs 27 of 28. The 6 records lost at none: 3 wrong target, 2 waiting lines where context missed the people, 1 nobody-near standard stop 0.99 m outside; 0 from the goal step.
- Verdict: **confirmed** (5 of 6 from plan or context).
- Scripts: as R51.

**R53. Gate: 49 of 52 vs 51 of 52** (I22 First probe, line 778)
- Re-run: 49 of 52 vs 51 of 52, as recorded and after re-solving (0 places moved). Pool = s10 none vs low matched (11) + s8 vs s19_ph matched (41). But s10 none is byte-identical to s8 trial 0 (28 of 28 records), so 11 none draws are counted twice. Duplicate-free: 38 of 41 vs 40 of 41 (s8 vs s19_ph alone, passes at the boundary) or 37 of 40 vs 40 of 40 (fails by one). The swing is one record, eval_10 e0 t0. The s8/s19_ph half also changes the context prompt (+80 input tokens).
- Scripts: as R51; `results/still-scenes/thinking-levels/verify/v4_ctx.py`, `v5_inp.py`.

**R54. Stage-21 fresh-call medians** (I22 Payoff; 1.1 item 8 "35.8 s")
- REPORT: "Stage 21 … has fresh medians of 7.6 s plan, 6.9 s context, 17.2 s for E's goal call and 35.8 s for constraints."
- Re-run: plan 7.6 s (n 252, all in cot folders), context 6.9 s (251), goal R 16.0 s (144), goal E 17.2 s (144), constraints 35.8 s (144); no fresh people calls (144 copies of s19).
- Verdict: **confirmed**.
- Scripts: `results/still-scenes/speed/latency/latency.py`; `results/still-scenes/speed/latency/verify/stages.py`.

### Section 4.3: Tier (b)

**R55. Line members in profile** (I09 Better/worse)
- REPORT: "every line member in the realistic eval sets stands 75–107 deg from facing the camera, in profile".
- Re-run: n = 48 instances (16 distinct people × 3 identical sets). From the robot base pose: 74.51–106.97 deg, median 89.0; 48 of 48 inside 75–107 after rounding (45 strictly). From the camera optical centre: 74.1–107.3 deg (42 strictly inside). Largest distance from 90 deg: 17.3. 27 turn slightly toward the camera, 21 slightly away.
- Scripts: `results/still-scenes/speed/profile/profile_angles.py`; `results/still-scenes/speed/profile/verify/v_profile.py`, `v_yawconv.py`.

**R56. Blob facing axis vs null** (I33 Why, line 888; I03 line 628)
- REPORT: "The facing axis from PCA came within 45 deg for 0.98 of people on placeholder scenes and 0.68–0.72 on realistic ones, against a null baseline of 0.24–0.28."
- Re-run: torso-band PCA (points at 1.0–1.6 m): realistic 0.676–0.703, placeholder 0.971–0.976. All blob points: realistic 0.82–0.84, placeholder 0.86–0.88. Cell centres: 0.74–0.76 and 0.93–0.95. Nulls on the same people: uniform axis modulo 180 = 0.50; random direction on the full circle 0.25; "faces the camera" axis 0.235–0.297 (realistic); PCA axis against shuffled true yaws 0.52–0.59.
- Verdict: **corrected**. The facing values reproduce with a torso band; the null is wrong for a modulo-180 test. The realistic gain over chance is about +0.2, not about +0.44.
- Scripts: `results/still-scenes/perception/depth-blob-pilot/blobpilot.py`; `results/still-scenes/perception/depth-blob-pilot/verify/v_extra.py`.

**R57. Any-of-3 vs all-3** (I02 Why)
- REPORT: "Any-of-3 vs all-3 under C: GLM 78/44, DeepSeek 79/38 (brief section 3.4)."
- Re-run: GLM 78 / 44, DeepSeek 79 / 38 of 84 episodes (s19 glm53 con, s20 dsv41 con). Rescored: identical.
- Verdict: **confirmed**.
- Scripts: `results/still-scenes/combining-models/fusion/fusion.py`; `results/still-scenes/combining-models/fusion/verify/v3_fusion.py`.

**R58. Majority-of-3** (I02 Why)
- REPORT: "Majority-of-3 succeeds in 64 (GLM) and 65 (DeepSeek) of 84, against per-trial means of about 62 and 60.7."
- Re-run: majority 64 and 65 of 84; per-trial means 62.00 (trials 64/65/57) and 60.67 (65/60/57). Gains +2.00 and +4.33 (Qwen s19 +2.00).
- Verdict: **confirmed**. The skeptic's "+2/+5/+2" is the gain rounded with the mean rounded down (list A22).
- Scripts: as R57.

**R59. People vs constraint variance across trials** (I02 Why; Appendix B I02)
- REPORT: "In episodes where trials disagree, the number of people differs in only 2 of 19 (GLM) and 7 of 22 (DeepSeek), while the constraint kinds differ in 17 of 19 and 21 of 22. The variance sits in the constraints."
- Re-run: 19 and 22 mixed episodes; people count differs 2 of 19 and 7 of 22 (matches). mixedC.py reads `entry.get('kind')` on wrapped entries, so its "kinds" test compares the number of constraints (17 of 19, 21 of 22). True kinds differ in 19 of 19 and 22 of 22, and also in 22 of 24 (GLM) and 26 of 26 (DeepSeek) all-success episodes, so the statistic does not separate success from failure. Swap test through H.resolve: re-solving the failing trial with the succeeding trial's people fixes 16 of 19 (GLM) and 17 of 22 (DeepSeek) episodes; with its constraints, 2 of 19 and 11 of 22. Without a swap, every failing trial reproduces its place (31 of 31, 42 of 42). Under GLM a mis-lifted person is often about 4 m off.
- Verdict: **corrected**. The numbers 2/19 and 7/22 stand; "kinds" is a count, and the conclusion is reversed: the deciding variance is mostly in the people answer.
- Scripts: `results/still-scenes/combining-models/fusion/fusion.py`; `results/still-scenes/combining-models/fusion/verify/v10_swap.py`, `v11_baseline.py`, `v7_usable_kinds.py`.

**R60. 6-sample cross-model vote** (I02 Why; Appendix B I02)
- REPORT: "With 6 GLM+DeepSeek samples, at least 3 succeed in 78 of 84 episodes."
- Re-run: 78 of 84 (at least 3 of 6). Any 83, at least 4 (strict majority) 56, all 27. Rescored: identical.
- Verdict: **confirmed**. 78 counts ties as wins; the strict majority (56) is below either model's own majority-of-3.
- Scripts: as R57.

**R61. Standard-goal misses and the fallback** (I47 Why and Correction)
- REPORT: "The standard goal lands outside the region in 8 (Qwen), 5 (GLM) and 14 (DeepSeek) records" and "The fallback accounts for only about 3, 4 and 6 records." and "The denominator is 27 (8 + 5 + 14)".
- Re-run: standard-goal misses 8 / 5 / 14, all outside the region and reachable, none blocked. Fallback (standard_goal_frontal False): 3 of 8, 4 of 5, 6 of 14 (13 of 27). Strictly "no surface normal found": 3 / 3 / 6. GLM's 2 waiting-line standard misses would make it 5 of 7. The brief's whole C failure table reproduces for all three models. Recomputing standard_goal on the recorded target gives the same flag and pose in 29 of 29.
- Verdict: **confirmed**.
- Scripts: `results/still-scenes/other-checks/flags/flags_misc.py` (m2); `results/still-scenes/other-checks/flags/verify/v_m1_m2.py`, `v_m2b.py`.

**R62. DeepSeek on eval_02** (I47 Why)
- REPORT: "On the oblique `eval_02`, DeepSeek gets 3 of 18."
- Re-run: 3 of 18 (s20 C, realistic). The 15 failures are 12 standard-goal misses and 3 solver places outside the region. 12 of DeepSeek's 14 standard-goal misses are on eval_02 (view angle 58.2 deg).
- Verdict: **confirmed**.
- Scripts: as R61.

**R63. Member lists are 1–3% of the output** (I19 Why; Appendix B I19)
- REPORT: "Member lists are only 1–3% of the visible constraints output, so the latency claim is dropped."
- Re-run: GLM s19 / DeepSeek s20 distinct constraints answers (141 / 157; reasoning_tokens 0, so raw text is the whole output). Characters: list contents 0.72% / 0.91% without brackets, 1.16% / 1.38% with brackets, 3.33% / 3.71% with the "people" key. Against recorded output tokens: about 3.7–3.9% (values) and 7.1–7.5% (with key). s23 is similar (3.8–3.9% and 7.3%).
- Scripts: `results/still-scenes/perception/perception-rerun/perception.py`; `results/still-scenes/perception/perception-rerun/verify/v_p2tok.py`.

**R64. GLM C and Qwen C time and success** (I39 Why)
- REPORT: "GLM C: 11.6 s, 74%. Qwen C: 61.6 s, 93% (brief sections 3.1, 3.6)."
- Re-run: GLM 186 of 252 = 73.8%, 11.6 s mean record time over 141 goal-step records (median 10.1). Qwen 234 of 252 = 92.9%, 61.6 s over 143 (median 59.3). Provider-only: 9.8 and 61.2 s.
- Verdict: **confirmed**.
- Scripts: `results/still-scenes/combining-models/cascade/cascade.py`; `results/still-scenes/combining-models/cascade/verify/verify.py`.

**R65. Qwen fixes 42 of GLM's 45** (I39 Why)
- REPORT: "Qwen succeeds on 42 of GLM's 45 goal-step failures."
- Re-run: 42 of 45 (GLM goal steps n 141 of 252 pairs; 3 fail for both). GLM also has 21 failures outside the goal step, which the cascade never escalates.
- Verdict: **confirmed**.
- Scripts: as R64.

**R66. Oracle gate 228 at 30%** (I39 Why; Appendix B I39)
- REPORT: "An oracle gate gives 228 of 252 at 30% escalation."
- Re-run: 228 of 252 = 186 + 42 gains; 30% of 141 rounds to 42.
- Verdict: **confirmed**.
- Scripts: as R64.

**R67. Disagreement gate** (I39 Why, line 1063)
- REPORT: "A gate built on disagreement between two samples reaches AUROC 0.73–0.75, and 216 of 252 at 50% escalation, against 201 for random escalation."
- Scripts: `results/still-scenes/combining-models/cascade/cascade.py`; `results/still-scenes/combining-models/cascade/verify/robust.py`, `verify.py`.

**R68. Plan reuse tops out near 207** (I39 Why)
- REPORT: "The pitched version that reuses GLM's plan tops out near 207."
- Verdict: **confirmed**.
- Scripts: `results/still-scenes/combining-models/cascade/cascade.py`; `results/still-scenes/combining-models/cascade/verify/replay_s19.py`.

**R69. Failed marked points at lines: 0.34, 0.88, 0.35 m** (I34 Why)
- REPORT: "Failed marked points at lines lie a median 0.34, 0.88 and 0.35 m from the region."
- Re-run: E (fsc) failures with the right target at lines: Qwen s16b 0.341 m (n 18), GLM s16b 0.878 m (n 55; 0.875 for the 53 marked points), DeepSeek s20 0.346 m (n 62). R gives 0.51, 0.60 and 0.74 m.
- Verdict: **confirmed**. These are E's numbers; the report should say so (list B).
- Scripts: `results/still-scenes/depth-fix/first-check/n1n2.py`; `results/still-scenes/depth-fix/first-check/verify/v1_n1n2.py`.

**R70. E's group failures are near misses on a body** (I34 Why; Appendix B I34)
- REPORT: "Failures of E at groups are mostly near misses that land on a body."
- Re-run: 44 E failures at groups with the right target (Qwen 5, GLM 16, DeepSeek 23). Near (within 0.3 m of the region, 0 inside it): 34. Near and within 0.3 m of a body disc: 28 of 44 (64%): Qwen 3 of 5, GLM 13 of 16, DeepSeek 12 of 23. Near and overlapping a body (gap < 0): 25 of 44 (57%).
- Definition: near.py port. "On a body" in near.py is min_body_gap_m < 0.3 m (personal-space intrusion), not overlap.
- Verdict: **confirmed** (weakest for DeepSeek, 52%). New wording: 67 of 93 (72%).
- Scripts: `results/still-scenes/depth-fix/first-check/n1n2.py`, `n1_s23.py`; `results/still-scenes/depth-fix/first-check/verify/v1_n1n2.py`.

### Section 4.5: Parked

**R71. I48: flags fire on successes** (I48 row; Appendix B I48)
- REPORT: "Flags fire on more DeepSeek successes (25) than failures (14)."
- Re-run: s20 DeepSeek C realistic (182 successes, 70 failures): records with any of the six solver flags 19 successes vs 10 failures (19 vs 11 firings); 10.4% of successes vs 14.3% of failures. Only constraint_unusable (19 / 8) and exclusion_broken (0 / 3) fire. 25 vs 14 appears exactly as flag firings pooled over all seven s20 DeepSeek C runs (realistic, placeholder and three dev seeds, n 408).
- Verdict: **corrected** (scope; and the raw-count framing hides a higher failure rate).
- Scripts: `results/still-scenes/other-checks/flags/flags_misc.py` (m1, m1b); `results/still-scenes/other-checks/flags/verify/v_m1_m2.py`.

**R72. I16: 3.9% of approaches cross a forbidden polygon** (I16 row; Appendix B I16)
- REPORT: "About 3.9% of straight-line approaches cross a forbidden polygon today."
- Re-run: robot start to recorded C goal pose, against every forbidden region of the episode's truth. Segment meets a polygon (ending inside counts): 25 of 752 = 3.3% (Qwen 0, GLM 14 of 248, DeepSeek 11). Pass-through only: 24 of 752 = 3.2%. C successes: 2 of 602 = 0.3%. Episodes with a forbidden region: 25 of 428 = 5.8%. 20 of the 25 cross ahead-in-line regions, 5 group space. Robot-body sweep (0.30 m): 9.2%. No reading tried by either agent gives 3.9% (range 0.1–6.0%).
- Verdict: **corrected**: use 3.3% (25 of 752) and 0.3% for successes; 3.9% cannot be reproduced.
- Scripts: `results/still-scenes/other-checks/flags/flags_misc.py` (m3), `m3_variants.py`; `results/still-scenes/other-checks/flags/verify/v_m3.py`, `v_m3b.py`, `v_m3c.py`.

**R73. I44: 1.7× faster per token, more tokens** (I44 row)
- REPORT: "it decodes about 1.7× faster per token but writes more tokens".
- Re-run: fresh calls only, pooled over s8 none/convlm/conorc: median 123.3 vs 68.6 output tokens per second, 1.80× (n 111 vs 108; people 1.90×, constraints 1.79×; matched calls 1.79–1.92×). Every call including cached copies (lat.py's reading): 1.74× pooled, 1.65–1.77× per run. Output tokens per fresh call: median 424 vs 352 (1.20×). Net fresh-call time 3.10 vs 5.19 s (0.60×). Both thinking off.
- Verdict: **corrected** (1.8× on fresh calls). The measure covers the whole call (prefill included), so "decodes" overstates it.
- Scripts: `results/still-scenes/other-checks/i44/i44.py`; `results/still-scenes/other-checks/i44/verify/v1.py`, `v7_matched.py`.

**R74. I44: 4 wins, 9 losses against Qwen low** (I44 row)
- REPORT: "paired 4 wins, 9 losses against Qwen low."
- Verdict: **cannot reproduce**. The direction against Qwen low C holds and is stronger, but it mixes thinking level and prompt version.
- Scripts: `results/still-scenes/other-checks/i44/search_q2.py`; `results/still-scenes/other-checks/i44/verify/v2_q2.py`, `v3_allpairs.py`.

**R75. I44: 136 vs 127 of 168** (I44 row)
- REPORT: "qwen3.6:35b scored 136 against 127 of 168 with C against Qwen 3.8 (brief section 6.2, P-9)."
- Re-run: no pair of listed runs gives it. Listed: convlm 69 vs 65 of 84 (13 / 9, p 0.52); conorc 71 vs 67 (10 / 6, p 0.45); both 140 vs 132 of 168 (23 / 15, p 0.26). 127 = 65 + 62, and docs/results/README.md:130 records Qwen 3.8 at 62 in an unlisted reasoning-first context run (inferred).
- Scripts: `results/still-scenes/other-checks/i44/i44.py`; `results/still-scenes/other-checks/i44/verify/v1.py`.

### Section 8: Risks to validity

**R76. Central estimates for Qwen** (Section 8, run-to-run noise)
- REPORT: "Central estimates for Qwen 3.8 at low thinking, each the mean of 9 draws with the same model and examples: R 180.7 of 252 (72%), E 204.7 of 252 (81%)."
- Re-run: Qwen 9-draw pool (s16b + s22 trials 3–5 + s21): R 542 of 756 and E 614 of 756, that is 180.7 and 204.7 per 252.
- Verdict: **confirmed** (ledger arithmetic on the verified pool). The pool mixes old and new wording; s22 trials 0–2 were dropped as copies of s16b.
- Scripts: `results/still-scenes/replay-harness/split/split.py` (TABLE 3); `results/still-scenes/replay-harness/split/verify/v_split.py`.

### Not re-checked

---

## New rows

### New wording, six models

Qwen is s21; the other five are s23. Realistic arms, 252 records per model and method.

**C vs E and C vs R, layout-clustered** (two-sided exact sign-flip p; layouts favouring C / the other / tied; t(13) 95% CI on the difference in records of 252)

| Model | C − E | p | Layouts | t CI | C − R | p | Layouts | t CI |
|---|---|---|---|---|---|---|---|---|
| Qwen | +32 | 0.0078 | 8/0/6 | +8.4 to +55.6 | +49 | 0.0156 | 7/0/7 | +11.9 to +86.1 |
| GLM | +47 | 0.0078 | 8/0/6 | +17.3 to +76.7 | +38 | 0.0156 | 7/0/7 | +8.2 to +67.8 |
| DeepSeek | +59 | 0.0078 | 8/0/6 | +20.4 to +97.6 | +65 | 0.0078 | 9/2/3 | +22.4 to +107.6 |
| Haiku | +92 | 0.0039 | 9/0/5 | +38.1 to +145.9 | +89 | 0.0039 | 9/0/5 | +32.4 to +145.6 |
| Sonnet | +64 | 0.0156 | 7/0/7 | +14.4 to +113.6 | +64 | 0.0156 | 7/0/7 | +8.0 to +120.0 |
| Gemini | +41 | 0.0078 | 8/0/6 | +11.0 to +71.0 | +50 | 0.0078 | 8/0/6 | +16.5 to +83.5 |

Every p is the smallest possible for its count of non-tied layouts. Old wording, C − E: Qwen +20.0 (t CI +2.4 to +37.6, t p 0.029), GLM +41.0 (+11.0 to +71.0, 0.011), DeepSeek +62.0 (+27.0 to +97.0, 0.0021). Caveats: GLM and DeepSeek C reuse s19/s20 people answers; the context prompt was chosen in-sample. Script `results/still-scenes/metrics-and-statistics/stats-rerun/stats.py`.

**Fresh-call median latency per stage, s** (n calls; cached copies excluded)

| Model | plan | context | goal R | goal E | people | constraints |
|---|---|---|---|---|---|---|
| Qwen (s21) | 7.6 (252) | 6.9 (251) | 16.0 (144) | 17.2 (144) | none fresh (144 copies of s19) | 35.8 (144) |
| GLM | 2.7 (252) | 1.5 (252) | 4.5 (141) | 4.3 (141) | none fresh (141 copies of s19) | 8.6 (141) |
| DeepSeek | 1.5 (252) | 1.0 (249) | 2.3 (149) | 1.6 (149) | 0.6 (2; 149 copies of s20) | 3.1 (149) |
| Haiku | 2.1 (246) | 1.6 (246) | 2.7 (140) | 2.4 (140) | 2.2 (71) | 3.6 (140) |
| Sonnet | 2.9 (246) | 2.2 (246) | 3.6 (137) | 3.1 (137) | 2.8 (69) | 4.6 (137) |
| Gemini | 3.3 (246) | 3.9 (246) | 6.4 (129) | 3.7 (129) | 4.2 (69) | 4.4 (129) |

**Concurrency replay of C** (cp.py definition; mean s over goal-step records)

| Model | n | Serial → concurrent | People shared per scene | A2 extra saving |
|---|---|---|---|---|
| Qwen (s21) | 144 | 71.2 → 57.7 (−19.0%) | 63.2 → 56.5 (−10.6%) | 2.83 s |
| GLM | 141 | 18.3 → 16.3 (−11.0%) | 17.3 → 16.2 (−5.9%) | 2.26 s |
| DeepSeek | 151 | 11.3 → 8.7 (−22.9%) | 9.7 → 8.3 (−13.6%) | 0.56 s |
| Haiku | 146 | 14.9 → 12.6 (−15.6%) | 13.7 → 12.5 (−8.6%) | 1.70 s |
| Sonnet | 143 | 15.6 → 12.8 (−18.0%) | 14.2 → 12.8 (−9.9%) | 2.24 s |
| Gemini | 135 | 17.1 → 12.8 (−25.5%) | 15.0 → 12.7 (−15.2%) | 3.15 s |

Qwen s21 C concurrent / E = 57.7 / 34.7 = 1.66×. GLM and DeepSeek people latencies are carried from s19/s20 (a different load), and a few API-model calls carry pre-flight latencies. Haiku's medians run well below its means (serial median 10.2 s). Script `results/still-scenes/speed/latency/latency.py`.

**Fusion of trials under C** (any / majority / all of 84 episodes; mean successes per trial)

| Model | Any | Majority | All | Mean | Majority − mean |
|---|---|---|---|---|---|
| Qwen (s21) | 83 | 80 | 65 | 76.00 | +4.0 |
| GLM | 76 | 58 | 36 | 56.67 | +1.3 |
| DeepSeek | 79 | 67 | 39 | 61.67 | +5.3 |
| Haiku | 83 | 82 | 75 | 80.00 | +2.0 |
| Sonnet | 84 | 82 | 76 | 80.67 | +1.3 |
| Gemini | 74 | 71 | 69 | 71.33 | −0.3 |

GLM + DeepSeek at s23 (6 samples): at least 3 succeed in 71 of 84 (strict majority 59), below the I02 probe's 74. All six models pooled (18 samples): at least 10 succeed in 84 of 84, at least 12 in 79, all 18 in 17; a vote of per-model majorities (at least 4 of 6) in 82. Swap test at s23: GLM's succeeding people fix 14 of 18 mixed episodes, its constraints 3; DeepSeek 11 and 10 of 19. Old wording, Qwen s19: 84 / 80 / 70, mean 78.00. Script `results/still-scenes/combining-models/fusion/fusion.py`.

**Model-escalation cascade from s23 GLM** (GLM alone 170 of 252; 141 goal-step records, 55 failures)

| Strong model | Alone | Fixes / loses | dis AUROC fail / gain | At 50%: dis / dis1 / random (expected) | Escalate all | Plan reuse (proxy / re-solve) | True oracle |
|---|---|---|---|---|---|---|---|
| Qwen (s21) | 228 | 47 / 5 | 0.782 / 0.748 | 206 / 206 / 190.9 | 212 | 196 / 200 | 212 at 30%, 217 at 40% |
| Sonnet | 242 | 52 / 7 | 0.782 / 0.780 | 208 / 211 / 192.3 | 215 | 198 / 201 | 222 at 40–50% |
| Gemini | 214 | 35 / 13 | 0.782 / 0.699 | 188 / 190 / 181.0 | 192 | 180 / 199 | 205 at 30% |

**GLM and DeepSeek perception at s23** (people answers are s19/s20 copies; only constraints are new draws)

| Model | Missed members never found | Line failures | Recall < 1 (= 0) | Found < 1 (= 0) | No come_after | Mean members recall |
|---|---|---|---|---|---|---|
| GLM | 0.726 (61 of 84; 88 line steps) | 32 | 32 (10) | 28 (5) | 5 | 0.635 |
| DeepSeek | 0.513 (20 of 39; 90 line steps) | 17 | 16 (2) | 9 (0) | 1 | 0.858 |

At s23, one DeepSeek line failure has recall 1, so "all line failures have recall below 1" no longer holds for DeepSeek. Script `results/still-scenes/perception/perception-rerun/perception.py`.

The I01 new-wording rows (solver alone, people-blind goal, target found vs placement, clearance snap) are in `results/still-scenes/replay-harness/sets-and-definitions/README.md`.

### Other new findings (old wording or placeholder)

- **Thinking fields.** `reasoning_tokens` is null on all 9,338 Qwen 3.8 calls (no `<think>` tags). At none, output tokens are 22.6% (people) and 18.2% (constraints) of low on s10 matched calls, under the I22 gate's 40% (plan 45.1%). `results/still-scenes/thinking-levels/thinking.py`.
- **I48 flags per model.** GLM: 10 of 186 successes (5.4%) vs 20 of 66 failures (30.3%) flagged; on solver steps 10.4% vs 44.4%. Qwen: 6 vs 0. `results/still-scenes/other-checks/flags/flags_misc.py`.
- **I47.** Among nobody-near standard goals with the right target, the fallback succeeds 22 of 25 (Qwen), 9 of 13 (GLM), 12 of 18 (DeepSeek); the fitted face 78 of 83, 81 of 82, 61 of 69. A fallback-only fix reaches at most 13 of the 27 misses, below the probe's 17.
- **I16 context.** Pass-through crossings: C 3.2%, R 1.5%, E 0.9% (n 752 each). R and E instead end inside forbidden regions (38 and 35 records, against 1 for C).
- **I11 Stage 0 probe** (old wording). C and E target points are identical in 756 of 756 pairs. The oracle snap fixes 26 of 53 wrong targets (22 become full successes); GLM + DeepSeek 20 of 46 (16 full), meeting three of the four first-probe criteria (the depth-component criterion was not tested). The one zero-call rule tried (front-surface snap) fixes 8 and breaks 37 of 752. Placeholder adds 15 wrong targets (5 fixed). `results/still-scenes/perception/lifting-rerun/`.
- **GLM phantom mechanism.** 13 of the 20 distinct phantoms sit within 25 px (median 15.7 px) of a line member who was never found; matched detections sit a median 5.1 px from their own person. DeepSeek behaves the same way (7 of 8). Plausible, not proven.
- **Blob pilot extras.** False body-sized blobs: 0.6–2.6 per realistic image across split rules, about 1 at a middle rule (Stage A's kill line is 0.3); 0 on dev seeds. Median xy error of matched blobs 0.075–0.077 m (realistic), just inside I33's 0.08 m bar. `results/still-scenes/perception/depth-blob-pilot/`.
- **I44.** Turning thinking off cuts the C goal-step call about 4× (0.23–0.27×, mostly by writing about 5× fewer tokens); swapping Qwen 3.8 for qwen3.6 at thinking off cuts it about 1.7× (0.59–0.60×). This supports Appendix B's "the speed comes mostly from thinking off". `results/still-scenes/other-checks/i44/`.

---

### Reads beyond the plan

- Source code under `src/sgi` (goal.py, types.py, evaluate.py, polygons.py, gt) for definitions (flags_misc, snap and others).
- `replay_runs.txt` (a harness list file) was read by the thinking verifier to locate the s12 run name.

Procedure notes:

- The perception agent once ran the venv Python without the `nice` and thread-cap prefix, to apply a pure text patch to its own script. It imported no harness and read no data.
- The first blob agent cleared the harness's private scene cache (`H._SCENES`) between loads. Harmless.
- No s22 trials 0–2 records were pooled (they are exact copies of s16b: 252 of 252 cot and fsc). Only s22 trials 3–5 enter the Qwen 9-draw pool.

### Held-out note

- The blob pilot was re-run from the 68 whitelisted development scenes only. Its "42 placeholder people" must be eval_00–13 plus eval_14/15. The original pilot probably used the base renders in `data/placeholder/eval`, a folder that also holds the held-out eval_16–19. The re-run did not touch that folder, and its one placeholder miss could not be reproduced on any whitelisted subset.
