# CT2-nora: CT2 without readmit (post hoc, labelled; frozen 2026-11-02 before any CT2-nora episode)

**Post hoc.** CT2's fresh-seed results are known. The log attribution in `ct2_attrib*.py` motivated this ablation.

## Arm
- **CT2-nora** is CT2 with part 1 (readmit) off and everything else unchanged.
- It runs on the cached cells only, so it makes no model calls:
  - Sonnet seeds 10–19: the 5 evaluation scenarios, 49 cells with a first answer;
  - DeepSeek seeds 10–14: 25 cells.
- It is paired with CT2 and CT in the same cells.

## Split, fixed in advance from CT2's own logs (readmit fired = a person id leaves CT2's dropped list)
- **Readmit cells:** Sonnet 25, DeepSeek 15.
- **No-readmit cells:** Sonnet 24, DeepSeek 10.
- In the no-readmit cells the ablation changes nothing in the method. CT2-nora against CT2 there is therefore a timing-only floor measured in the same run.

## Measures
- Share in line, strict (SCORING_STRICT.md), paired.
- "Line ran out" samples, from `analysis/failures.py` classify (frozen in-line test).

## Predictions (inference, written before any episode)
- **P1, readmit cells.** CT2-nora's "line ran out" samples come back at least halfway from CT2 to CT.
  - Sonnet: CT 385, CT2 217, so CT2-nora ≥ 301.
  - DeepSeek: CT 220, CT2 89, so CT2-nora ≥ 155.
- **P2, DeepSeek readmit cells.** CT2-nora's share is below CT2's by at least 0.07, half of CT2's +0.155 over CT there. More paired losses than wins.
- **P3, Sonnet readmit cells.** The share drop is smaller than DeepSeek's (mean |drop| < 0.07), because the failure classes substitute.
- **P4, no-readmit cells.** CT2-nora against CT2 is within the timing floor: mean difference within ±0.04, mean |d| ≤ 0.10.

## Not covered
- Path memory: all 5 evaluation lines are straight (every slot segment points the same way), and for a straight line path memory equals the straight axis by construction. It can act only in the bent break-it case.
- The 0.5 m span: an optional separate ablation on the two-person lines.
