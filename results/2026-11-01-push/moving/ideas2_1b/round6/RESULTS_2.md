
Inputs:
- `c_tradeoff_pilot_out.txt` and `c_tradeoff_confirm_out.txt`; questions and readings in `ROUND6_C.md`.
- `d_none_pilot_out.txt` and `d_none_confirm_out.txt`, a scoring check added after part C. It was not planned in advance.
- All are in `ROUND6.sha256`.

**Process note.** `c_tradeoff.py` ran once on the pilot logs before the questions and script were hashed. Nothing was changed after that run. The confirm logs ran after hashing.

## 1. Correction to part B

Part B scored a choice of "nobody ahead" as wrong even when the sim agreed nobody was ahead. Scored correctly, over joined re-solves where the sim has the robot in line:

| | someone ahead: n, CT3F right | n, CT4PJ right | nobody ahead: n, CT3F right | n, CT4PJ right |
|---|---|---|---|---|
| pilot | 800, 0.859 | 771, 0.848 | 355, 0.296 | 142, 0.254 |
| confirm DeepSeek | 2306, 0.732 | 2040, 0.712 | 1197, 0.517 | 904, 0.217 |
| confirm Sonnet | 3337, 0.720 | 2422, 0.643 | 1030, 0.658 | 708, 0.251 |

- **Withdrawn from `RESULTS_1.md`:** "while truly in line, both pick the right person about equally often" and "CT3F doesn't know better whom it's behind".
- **What holds instead:**
  - With someone ahead, the two are about equal on the pilot and DeepSeek; CT3F is better on Sonnet.
  - At the front of the line, CT3F knows nobody is ahead 2.4 and 2.6 times as often on the confirm runs.
  - CT4PJ keeps aiming behind someone who is no longer in the line. Round 4 (`RESULTS_D.md`) found the role belief takes 4-6 s to drop a served customer, against CT3F's 2.6 s.
  - CT3F is also at the front more often: 1197 against 904, and 1030 against 708 re-solves.
- Part A's switch rates and part B's frozen-predecessor result are unchanged.

## 2. Part C: the decision's trade-off on CT4PJ's joined stretches

Right last where the sim agrees, scored as in section 1, and switches per minute joined:

| | CT3F | CT4PJ | best point of the margin curve (m) | at CT3F's switch rate | C2 (CT5's costs) |
|---|---|---|---|---|---|
| pilot | 0.686, 4.46 | 0.756, 8.89 | 0.754, 5.38 (0.5) | 0.748, 4.45 (0.7) | 0.691, 8.75 |
| confirm DeepSeek | 0.659, 4.08 | 0.560, 8.08 | 0.569, 4.61 (0.6) | 0.568, 4.19 (0.7) | 0.527, 7.99 |
| confirm Sonnet | 0.705, 4.01 | 0.554, 8.77 | 0.563, 5.50 (0.6) | 0.558, 4.56 (0.7) | 0.519, 10.09 |

- **C1, the whole margin family.**
  - On the confirm logs, no margin brings the role belief's evidence near CT3F's accuracy. The curve tops out at 0.57 and 0.56, against 0.66 and 0.71.
  - Holding longer buys stability at no gain in accuracy.
  - On the pilot logs, m 0.7 reaches CT3F's corner, because there CT4PJ's evidence was already more accurate than CT3F's.
- **C2, the principled rule, misses on all three logs.**
  - It is CT5's decision after commitment, with ROUND3's costs unchanged.
  - The move cost (0.125 per metre) is too small to steady the choice: 8-10 switches a minute.
  - The cheap "far" cost lowers accuracy.
  - ROUND3's sensitivity points don't change this.
- **Reading, as fixed in `ROUND6_C.md`:**
  - The pilot shows a margin reaching the corner; the confirm logs show none; C2 reaches on none.
  - So the result is mixed: no principled rule is demonstrated. The recommendation is R1's: close the role-belief line and write it up.

CT3F's lead over the role belief is information, not decision.
- No hold-or-switch rule on the role belief's evidence reaches CT3F's accuracy on fresh layouts, whatever its threshold.
- The principled cost rule does worse than the role arm's own choice.
- Most of the accuracy gap is at the front of the line. CT3F's rules drop the person just served almost at once; the role belief keeps them for seconds and aims behind them.
- The design pilot overstated the role belief twice: once in time in line (`CONFIRM_RESULT.md`), and again here in accuracy (0.756 against CT3F's 0.686 on the pilot, 0.56 against 0.66-0.71 on fresh layouts).

## 4. One open lead, flagged for the write-up's future work and not proposed

Round 5's "walking away from the counter" signal is specific: it fires on 81% of ex-member bouts and 1-6% of member bouts. It failed as a whole-roster belief only because joining can't be separated before a service, and after commitment joining no longer matters. Applied only after commitment, it targets the front-of-line failure above.

It came from spent data. It would need:
- design-data checks first (the pilot shows only a small front-of-line gap, 0.296 against 0.254);
- a fresh range.
