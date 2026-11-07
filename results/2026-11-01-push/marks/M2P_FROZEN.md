# Marks probe M2P: what moves DeepSeek's click, the prompt or the drawn marks? (frozen before any call, 2 Nov 2026)

## Why
- **The finding.** In M2, DeepSeek's replayed success fell from 195 to 168 although its target choice improved. In 28 pairs both clicks were on the right target, A0's replay succeeded and A2's failed; A2's click had moved a median 98 px on a place target.
- **The cause is not occlusion.** Only 1 of 28 A0 clicks lay under a label.
- **x-test-3 had none of these.**
- **Two causes are possible:**
  - the marks prompt: its target_mark instruction and "do not point at a label";
  - the labels drawn on the image.

## Design: a fresh 2x2 (prompt × image), DeepSeek only
- **Arms** (all fresh, at trials 6, 7 and 8):

  | Arm | Prompt | Image |
  |---|---|---|
  | P0U | the pipeline's | unmarked (= M2's A0) |
  | P2M | the marks prompt and schema | marked (= M2's A2) |
  | P2U | the marks prompt and schema | unmarked |
  | P0M | the pipeline's | marked |

- **Cells.** The 20 (scene, episode) behind the 28 pairs (`m2p_episodes.json`) × 3 trials = 60 cells, 240 calls.
- **Why fresh draws for all four arms.** The cells were picked because A2 lost there and A0 won. Re-using M2's A0 and A2 answers would regress to the mean in M2's favour. Fresh draws in every arm avoid that.
- **Code.** `m2p_driver.py` imports M2's frozen driver and library unchanged. Marks, numbering, prompts and style are M2's.

## Measures (`m2p_score.py`; zero calls)
Per arm, over the 60 cells, using the click alone, since that is where the change is:
- target right;
- replayed success against DeepSeek's s23 C record with trial − 6, so trials 6, 7 and 8 map to 0, 1 and 2;
- the click's distance from P0U's click at the same trial (median and 75th percentile);
- answers that carry a mark. For P2U, where no label is visible, this shows whether the model invents numbers.

## Decision rule (judged once)
- **The loss to explain:** L = replay(P0U) − replay(P2M).
- **If L ≤ 2:** "no loss to explain". On fresh draws, M2's drop was noise or selection.
- **Otherwise,** with L_prompt = replay(P0U) − replay(P2U) and L_image = replay(P0U) − replay(P0M):
  - **the prompt** is the cause if L_prompt ≥ L/2 and L_image < L/2;
  - **the drawn marks,** if the reverse;
  - **both,** if both are ≥ L/2;
  - **only the two together,** if neither is.

## Size

## Pins (sha256)
See `M2P.sha256`, written with this note.
