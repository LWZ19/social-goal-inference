# CT2-norv: CT2 without readmit and without the vacated place (post hoc, labelled; frozen 2026-11-02 ~04:55, before any CT2-norv episode)

**Post hoc.** CT2-nora (CT2_NORA_POSTHOC.md) has been judged, and its P1 and P2 failed (`nora_score_out.txt`).

## What CT2-nora showed
- In the readmit cells, removing readmit did not bring "line ran out" back.
  - Sonnet: CT 385, CT2 217, CT2-nora 184.
  - DeepSeek: CT 220, CT2 89, CT2-nora 137.
- Share stayed at CT2's level: Sonnet +0.024, DeepSeek −0.028 (7/7/1).
- On seeds 10–12, CT2-vac (no vacated place) also kept "line ran out" low: 98, against CT2 163 and CT 207.

## Hypothesis H
Readmit and the vacated place are two redundant fixes for the same failure: the waiting set empties while people still queue.
- With readmit off, a returning person stays dropped. The waiting set then empties, and the vacated place sends the robot to the leaver's spot.
- With the vacated place off, readmit keeps the waiting set filled.

## Arm
- **CT2-norv** is CT2 with part 1 (readmit) and part 3 (vacated place) off, and everything else unchanged.
- It runs on the same cached cells as CT2-nora, so it makes no model calls: Sonnet 10–19 (49 cells with an answer) and DeepSeek 10–14 (25).
- The readmit split is as in CT2_NORA_POSTHOC.md: Sonnet 25/24, DeepSeek 15/10.

## Predictions (inference, written before any episode)
- **Q1, readmit cells, "line ran out"** (`analysis/failures.py`): CT2-norv comes back at least halfway from CT2 to CT. Sonnet ≥ 301; DeepSeek ≥ 155.
- **Q2, readmit cells, share strict against CT2:** the drop is at least half of CT2's gain over CT there. Sonnet ≤ −0.042; DeepSeek ≤ −0.078. Losses outnumber wins on each model.
- **Q3, all cells, share strict:** CT2-norv against CT keeps less than half of CT2's gain over CT on each model (CT2 − CT over these cells: Sonnet +0.082, DeepSeek +0.078).

## If Q1 to Q3 fail
CT2's gain does not come from handling a line that runs out. The candidates left are the 0.5 m span and the forward ratchet, singly or together; path memory is inert on these straight lines. That would be reported as found, with no further ablation before the report.
