# Bystanders: post hoc diagnosis (after the frozen judgements; zero calls)

Files:
- `constraint_cells.py`: the pilot restricted to constraint answers;
- `confirm_diag.py`: the confirmation's no-answer cells, false drops, where the leftover capture comes from, and wins
  split by capture.

## Pilot (seeds 10-14)

- **Answers.** 6 of 15 first answers were standard goals: room 5/5, sweet s14. In those, nothing re-solves and every C
  arm is identical (the robot drives through the line to the cart). On the 9 constraint-answer cells:
  - CT2 is captured in 6 of 9;
  - Gl's capture is above CT3's in 8 of 8;
  - CT3 against CT2 is 5/2/2.

## Confirmation (seeds 15-44, cafe and sweet_spot)

- **No answer.** 5 of 60 first answers were `no_target` (cafe s37; sweet s23, 26, 41, 43). Every arm scores 0 there.
- **Wins track capture.** CT3 against CT2 is 14/3/9 (+0.045) in the 26 cells where CT2 was captured, and 3/3/28
  (−0.003) in the 34 where it was not. Mean strict share: CT2 0.025, CT3 0.043, CT3b 0.049, Gl 0.004.
- **Leftover capture.** All of CT3's 203 s come from bystanders the model named as waiting, where the gate does not
  apply. DeepSeek named a bystander in 10 of 51 constraint answers. The F4 step-up leak contributed 0 s.

## M4: CT3's false drops, two causes

1. **Pilot, cafe s11.** A served member walked back up the line, and a follow check that was due from an advance made
   before she was re-admitted dropped her. CT3b (DEVIATIONS 13) cancels such checks.
