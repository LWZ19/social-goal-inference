# Results, 7 Nov: items 4 and 11, extras X1-X4, E3b, item 9's GLM secondary and the E4 interim

## Item 4: the model's roles against tracking its marked point (`i4_rtw/`)

- **Frozen bar H4 holds.** Pooled over 125 cells, on whole-episode strict share, CT2 beats RTw 87/31/7, p < 0.0001,
  mean +0.197. Under the primary scoring it is 87/36/2, +0.225, also p < 0.0001.
- **Per model:**

  | model | CT2 vs RTw strict | strict levels, CT2 vs RTw |
  |---|---|---|
  | Sonnet | 40/7/3, p < 0.0001 | 0.318 vs 0.036 |
  | DeepSeek | 17/6/2, p 0.035 | 0.278 vs 0.067 |
  | Haiku | 20/3/2, p 0.0005 | 0.320 vs 0.071 |
  | Gemini | 10/15/0, p 0.42, not shown | 0.218 vs 0.255 |
  | x-test-3 (reported) | 7/4/4, p 0.55 | 0.118 vs 0.103 |

- **Why RTw fails:** its anchor left the queue in 117 of 140 episodes. It follows the person the point was
  marked beside, and that person is served and walks off.
- **Repeat checks:** these runs against the earlier runs differ by mean |d| ≤ 0.037.
- **Qwen and GLM are not in item 4.** Their RTw arms were never run.
- **Post hoc, the secondary RTw vs R1** (`i4_posthoc_r1.py`):
  - The frozen scorer looked for R1 in the new folders, where it was never re-run. That was my slip; the secondary
    printed empty.
  - Reading R1 from the earlier runs (a different machine load), RTw beats R1 a little on strict share: 46/5/74,
    +0.051, p < 0.0001. On the primary scoring it is −0.003.
  - So tracking a point helps only slightly over asking once. Most of the gain comes from the roles.

## Item 11: the best moving method on fresh layouts (`i11_ct3f/`)

- **H11a is not shown,** so H11b (CT3F vs CT3) is not judged, as the order fixes.
  - CT3F − CT2, strict from the first service: 9/2/12, p 0.065, n 23. Mean +0.137, lower bound +0.064.
- **Cause: the slow gateway.**
  - The live first answer was missing from the episode in 33 of 72 layouts, and took a median of 72.6 s where it came.
    On the confirmation layouts (2-3 Nov) the median was 12.6 s, with none missing.
  - So only 23 layouts had a service after delivery.
  - Amendment 2 (before scoring) left out the one layout whose replays missed the cache.
- **Reported:**
  - On whole-episode strict share CT3F beats CT2 12/2/57, p 0.013. CT3F − CT3 is 7/2/14 from the first service.
  - Repeat check: live CT2 0.139 against replayed CT2 0.134.
- **Reading:** the direction matches the confirmation layouts (CT3F ahead on every model there), but the fresh test is
  underpowered by latency. Not claimed.
- **Option (for the pending list):** re-run item 11 on a fresh range when DeepSeek's gateway is back to normal (the
  rolling-median rule of item 10). About 290 free calls and 3.8 sim-h. A 240 s replay of these answers would be post
  hoc only.

## X1: the true counter point on four models (`x1_ceilings/`)

- **Both readings hold.** Given the true counter point, CT3F's strict share from the first service rises:
  - pooled over 72 layouts: +0.079, lower bound +0.052;
  - 46/14/12 layouts, p < 0.0001.
- **Per model:**

  | model | mean d | W/L/T, p | front part |
  |---|---|---|---|
  | DeepSeek | +0.042 | 30/19/16, p 0.15 | +0.021 |
  | Sonnet | +0.086 | 34/19/18, p 0.053 | +0.095 |
  | Haiku | +0.049 | 25/14/16, p 0.11 | +0.051 |
  | Gemini | +0.128 | 40/12/19, p 0.0001 | +0.091 |

  The gain is mostly at the front of the line, where the robot must pick the serving spot itself.
- **Outcome, as frozen:** future-work lead 1 (observe the serving spot) goes before lead 2 (who waits).
- **Stage B** (the full decomposition) has not been run.

## X2: motion evidence and bystanders on Qwen (`x2_bystanders_qwen/`)

- **At 92 s it is latency-bound,** under item 9's rule: 59 of 60 CT2 episodes had no first answer in time. Every
  comparison ties.
- **The 240 s secondary (labelled) passes C1 and C2,** so "CT3 beats CT2 on bystander scenes" is claimed for Qwen as a
  secondary, as for DeepSeek:
  - **C1:** CT3 beats CT2 on strict share, 25/4/31, +0.028, p < 0.001.
  - **C2:** capture time falls from 2,065 s to 259 s. Cells captured: 36 to 6 of 60.
  - CT2 still had no answer within 240 s in 15 of 60.
- **Bar R is not judged:** the frozen secondary runs CT2 and CT3 only, so it has no Gl arm.

## X3: the people detector in place of the people call, moving scenes (`x3_cte/`)

- **Both bars fail.**
  - **X3-speed:** CTe's first answer came later, not earlier. DeepSeek median 47 s against 19 s, later in 21 of 25.
    Qwen ties.
  - **X3-quality:** pooled −0.096, lower bound −0.138.
- **DeepSeek:**
  - CTe is worse on strict share, 0.102 against 0.287 (2/16/7, p 0.001).
  - Measured only from each arm's own delivery (post hoc), it is still worse: 5/10/5, −0.144. So the loss is not just
    lateness.
- **Qwen is latency-bound in both arms** (20 and 21 of 25 with no answer in the episode). Nothing is measurable.
- **Design flaw (mine).** The speed bar is not a fair race:
  - CTd replays CT2's stored latency from its original run (19 s; wall time 0.24 s).
  - CTe calls live, at 35 s of wall time on DeepSeek.
  - So X3-speed's failure is not clean evidence that CTe is slower. The quality loss is the solid finding.
- **Contrast with E2:** on still scenes the detector's people beat the people call, pooled; for DeepSeek that was +6,
  p 0.41. On moving scenes, for DeepSeek, it does worse. Not a contradiction, but the efficient pipeline is not shown
  for moving scenes.

## X4: run-to-run noise, DeepSeek (`x4_noise/`); GLM is still running

| batch | R | E | C (of 252) | C − R |
|---|---|---|---|---|
| s23 | 120 | 126 | 185 | +65 |
| noise1 | 125 | 138 | 210 | +85 |
| noise2 | 126 | 137 | 219 | +93 |

- **Batch SD:**
  - R 3.2, E 6.7, C 17.6, against Qwen's 7.9.
  - C's spread is mostly s23. Its people answers were cache copies (the frozen caveat), and the two fresh batches
    differ by 9.
- **Smallest detectable difference** (2.8 × SD): about 49 records for C with s23, or about 18 from the two fresh
  batches.
- **C − R is +65 to +93 in every batch,** well above the noise.
- **Flips between batches:** 19-34% of records per method change between right and wrong, in both directions.

## E3b: the older models on the moving headline (`results/moving-lines/school-models/i9_score_e3b_out.txt`)

- **Re-solving (CT2) beats asking once (R1) on strict share on 5 of 7 older models:**

  | model | CT2 vs R1 strict |
  |---|---|
  | Sonnet 4.6 | 14/3/8, p 0.013 |
  | Sonnet 5 | 17/3/5, p 0.003 |
  | Gemini 3.1 Flash Lite | 22/2/1, p < 0.0001 |
  | Gemini 3.5 Flash Lite | 20/2/3, p 0.0001 |
  | Gemini 3.6 Flash | 18/5/2, p 0.011 |

- **Not shown:**
  - Haiku 4.5, 5/2/18: 10 of its 25 constraint answers were unusable (5 could not be turned into a goal, 5 named no
    target).
  - Gemini 3 Flash, 13/11/1.

## Item 9: GLM's 240 s secondary (`results/moving-lines/school-models/i9_score_long_out.txt`)

- **Still latency-bound at 240 s:** 24 of 25 constraint episodes had no answer in time. Median first answer 217 s (C)
  and 173 s (R). CT2 vs R1 is 0/3/22.
- **GLM cannot be judged on the moving protocol** at today's gateway speed. `RESULTS_I9.md` is updated.

- **The validity check passed** before any call:
  - the people-blind goal fails every scene, so people matter;
  - a straight extension fails every bent line, so the bend is tested;
  - the success areas can be stood in.
- **Both bars hold in both families on the five finished models.** C wins on every one of the 12 layouts. p 0.0005 is
  the floor for 12 layouts.

  | | C − R | C − E | C (of 72 per model) | R |
  |---|---|---|---|---|
  | machine use | +211 | +203 | 31-66 | 0-20 |
  | bent line | +253 | +256 | 40-66 | 0-4 |

  - Per model, every C − R and C − E has p ≤ 0.0078.
- **The 0.8 m sensitivity agrees,** for machine use:
  - C − R +223, 12/0/0; C − E +194, 11/1/0.
  - Every method gains. Sonnet's C − E falls to 7/1/4, p 0.078.
- **Prediction check:** I expected machine use to hold more weakly than the bent line. It held about as strongly.
- **GLM** adds at most ±6 records per layout. Its rows and the six-model bars come after its re-runs, under
  amendments 1 and 1b.
