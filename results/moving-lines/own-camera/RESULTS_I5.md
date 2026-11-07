# Item 5 result: tracking people from the robot's own camera (CTd) against the sim's true positions (CT2)

## Inputs

- **Frozen design:** `I5_FROZEN.md`, scored by `i5_score.py` into `i5_score_out.txt` (13eba76d).
- **Post hoc description,** written after reading that output: `i5_posthoc.py` and `i5_posthoc_out.txt`.
- **Run:** 125 cells (Sonnet 50; DeepSeek, Haiku and Gemini 25 each). Both arms ran in one invocation, cache-only.
  - 0 timeouts and 0 vlm_error. CT2's first pose matches the earlier runs in all 125.

## 1. The frozen bar holds

- **H5:** pooled strict mean(CTd − CT2) is +0.004, and the one-sided 95% lower bound is −0.023, above −0.05.
  - So "detector tracking costs no more than 0.05 of strict time in line, on average" is claimed for these scenes.
- **Per model, strict:**

  | model | mean d | lower bound | W/L/T |
  |---|---|---|---|
  | Sonnet | −0.012 | −0.046 | 10/31/9, p 0.001 |
  | DeepSeek | +0.002 | −0.050 | 6/13/6 |
  | Haiku | +0.031 | −0.049 | 11/12/2 |
  | Gemini | +0.012 | −0.065 | 6/13/6 |

## 2. But the average is not "no effect" (reported beside the bar)

- **Most cells lose, a little.** CTd is worse in 69 of 125 cells and better in 33 (sign p < 0.001). The per-cell median
  is −0.011, with quartiles −0.060 and +0.011.
- **The near-zero mean is a cancellation across scenes** (post hoc):

  | scene | mean d | worse / better cells |
  |---|---|---|
  | sweet_spot_line2_moving | **+0.189** | 2 / 16 |
  | cafe_line2_moving | +0.002 | 16 / 5 |
  | cafe_line6_moving | −0.025 | 17 / 3 |
  | room_coffee_cart_slots | −0.032 | 15 / 5 |
  | cafe_slots | **−0.114** | 19 / 4 |

  - In one scene, tracking from the camera beats the true positions by a lot. In the other four it is a little or
    clearly worse.
  - Why sweet_spot favours CTd is not diagnosed. One plausible cause: true positions include people the camera
    cannot see, and the solver acts on them.
- **Under the primary scoring** (in line without the side check) the mean is −0.034 and the lower bound −0.062, which
  misses the margin (reported). CTd loses more of the "level with the line but beside it" time that the primary
  scoring counts.
- **Repeat check (timing only):** CT2 re-run against its earlier run differs by a mean |d| of 0.024-0.065, with a mean
  shift of +0.003 to +0.018. CTd's repeat on the old 30 cells differs more (|d| 0.13 on Sonnet, 0.05 on DeepSeek), so
  CTd is more timing-sensitive than CT2.

## 3. Tracking quality

| model | live tracks matched to a true person | id switches per live track-frame | stale share of tracks | first-match rate | detector median ms |
|---|---|---|---|---|---|
| Haiku | 0.88 | 3.1% | 0.72 | 0.99 | 121 |
| Sonnet | 0.91 | 2.4% | 0.69 | 0.99 | 154 |
| Gemini | 0.94 | 2.6% | 0.63 | 0.99 | 123 |
| DeepSeek | 0.94 | 2.5% | 0.65 | 0.96 | 134 |

There were no detection errors in about 20,000 frames. About two thirds of tracks at any moment are stale: out of
view and held at the last position.

## 4. Reading for the report

- **The claim:** with the robot's own camera tracking people, time in line is on average within 0.02 of using the true
  positions (lower bound −0.023, against a −0.05 margin).
- **How to state it:** say "no loss on average, slightly worse in most cells and scenes, better in one", with the
  scene table.
- **Under the primary scoring** the loss is larger, at −0.034.
- **What it removes:** "every moving result assumes oracle tracking" is no longer the main validity threat. Its cost
  is small on average and scene-dependent.
