# Bystanders confirmation: does CT3 beat CT2 on share? (frozen 2026-11-02, after the pilot, before any new cell)

## Why

BYSTANDERS_FROZEN.md's pilot (DeepSeek, seeds 10-14) found:
- **Capture removed:** CT3 8.9 s pooled, against CT2 208.1 s (P4 PASS).
- **The model's roster:** DeepSeek named a bystander in 1 of 15 answers.
- **Share:** CT3 against CT2 was 5/2/8, +0.023, sign p 0.45, so "beats CT2" is NOT SHOWN.

Post hoc, about the pilot (`bystanders_post/constraint_cells_out.txt`):
- 6 of the 15 cells were standard-goal answers: all 5 of room, and sweet_spot s14. In those, DeepSeek's context call
  said people don't matter, so nothing re-solved, every C arm was identical, and the robot drove through the line
  to the cart.
- On the original room, DeepSeek did this in 1 of 5 cells. Room adds both the bystanders and the turned member, so
  the cause can't be separated.
- On the 9 constraint-answer cells: CT3 against CT2 was 5/2/2, +0.038.

This note asks the share question once, with more cells, where the answers are constraint answers.

## Cells

- **DeepSeek V4.1:** `cafe_line2_bystanders` and `sweet_spot_line2_bystanders`, Habitat seeds 15-44, giving **60
  cells**. Scenario files as hashed in BYSTANDERS_FROZEN.md.
- **Room is left out for DeepSeek.** Its pilot answers were 5 of 5 standard goals, and ties carry no sign-test
  information.
- Standard-goal cells in cafe and sweet_spot stay in, as ties. They are the method's behaviour.

## Arms

- **CT2 and CT3**, exactly as in the pilot (DEVIATIONS 12; `co_motion_s = 5.0`; resolver hash 15abba1663f531a1 plus
  logging only).
- **Gl:** zero extra calls, to replicate the role-knowledge reading.
- **CT3b (post hoc, reported, never part of a bar):** CT3, plus one change. A pending follow check is cancelled for
  anyone re-admitted to the waiting set (readmitted or absorbed) after the advance that scheduled it. This fixes the
  pilot's one false drop (cafe s11).
- **Calls:** CT2 makes the cell's first answer (4 free calls), and the others replay it. 60 × 4 = **240 free
  calls**; sim time for 4 arms × 60 cells.

## Bars (judged once, after all cells; `bystanders_confirm_score.py`)

- **C1 (share):** CT3 against CT2 strict share (SCORING_STRICT.md), paired by cell. Wins > losses, and exact sign p ≤
  0.05 over untied cells.
- **C2 (capture):** CT3's pooled capture is at most a third of CT2's, and CT3 is captured (≥ 1 s) in at most 20% of
  cells.
- **"CT3 beats CT2 on bystander scenes"** is claimed only with C1 and C2.
- **R (roles, replication):** on constraint-answer cells, Gl's capture exceeds CT3's in at least 80% of the cells
  where either is captured, and CT3 against Gl strict share has sign p ≤ 0.05.

## Power, stated before the data

- At the pilot's win rate among untied cells (5 of 7 = 0.71), with about 70% of cells untied (about 42), the sign
  test has power of about 0.79. At 0.65 it is about 0.48.
- The pilot's rate rests on 7 cells, so a failed C1 reads "not shown", not "no effect".
- The capture result (C2) and the pilot's P4 are the mechanism. C1 asks whether it shows in time in line, which our
  two-person lines hold near the floor: DeepSeek CT2 0.04-0.05.

- Room is where shares have headroom (Sonnet CT2 0.43 on the original). Sonnet answered with constraints there in
  10 of 10, and F2 predicts lone-man capture at a line of 2.
- Reported with C2's capture bar, and descriptively for share. No share claim from 10 cells unless the sign p is
  ≤ 0.05.

## Code

- `bystanders_score.py` (sha256 91d21d01): `--seeds` and `--scenes` added. The judged pilot output is reproduced
  line for line. The judged version is kept as `bystanders_score_c7532cd2.py`.
- `bystanders_confirm_score.py` is hashed with this note.
