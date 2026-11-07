# Round 7: readings for the ceiling check, fixed before the totals (analysis, 3 Nov)

- **OW:** no solver. The goal is the sim's ideal place: 0.85 m behind the true last person ahead on the axis, or
  slot 0.
- **CT4OT:** CT4PJ with the true members ahead as the roster at every re-solve, and the counter's true point.
- **CT4O:** CT4OT with the model's target point kept.

## 1. Already described, before the totals (`a_loss.py`, zero calls)

Outputs: `a_loss_pilot_out.txt` (design data), and `a_loss_confirm_dsv41_out.txt` and
`a_loss_confirm_sonnet55_low_out.txt` (spent; description only). The script reproduces every episode's frozen
strict share; it asserts equality with `score_sweep.episode_scores`.

**Where the window's seconds go** (confirm, DeepSeek):
- About 40-48% of the window is lost before the robot's first strict second ("arrival"), and 36-39% after it.
- The largest primary reasons are:
  - moving off the axis (side > 0.6 m): CT4PJ 24% of window seconds;
  - too far behind the person ahead (gap > 1.2 m): 19% for both arms;
  - ahead of the last person: 15-16%;
  - far from slot 0 when first: 11-13%.

**The chain with someone ahead** (confirm, DeepSeek, per re-solve):

| | CT3F | CT4PJ |
|---|---|---|
| right last | 0.46 | 0.47 |
| goal right, given right last | 0.70 | 0.55 |
| robot there, given goal right | 0.59 | 0.54 |

- "Goal right" means the goal would pass the strict geometry if the robot stood on it.
- Each link loses 30-50%.

**At the front** (the robot first; the sim never serves it, so this lasts to the end of the episode):
- This is 10-16% of window seconds.
- Strict there is 0.12 for DeepSeek CT3F, and 0.00-0.08 for every other arm and model.
- On the pilot, while nobody was ahead, CT3F still named someone ahead at 335 of 440 re-solves.
- When it did know nobody was ahead, its goal was about 0.85 m from slot 0, outside the strict 0.6 m.
- The cause is the target point: the model's target is 0.3-0.7 m off the counter point. The solver's place stays
  1.0 m from the target (`standoff_m`), while slot 0 is 0.85-0.95 m from the counter point.

**A convention mismatch inside OW − CT4OT:**
- The solver stands 1.0 m behind the person ahead (`standoff_m`, a shared pipeline setting). OW stands 0.85 m
  behind, the scored band's middle.
- The goal-error median of +0.16 m is this mismatch.
- At 1.0 m the robot is 0.2 m from the band's far edge (1.2 m), so a small step forward by the person ahead puts it
  out.
- This part of the gap is a convention, not a research problem. It must not be closed by tuning `standoff_m` to
  the test.

## 2. Predictions, fixed now

- **P1. OW** reaches a mean strict share of at least 0.7.
- **P2. At the front:** CT4OT's strict share is at least 0.5. Its target is right and its roster has no ghost. CT4O
  is lower at the front than CT4OT.
- **P3. With someone ahead:** CT4OT's strict share is 0.25-0.45. Right last is near 1, but goal and following
  still lose. So OW − CT4OT is the largest of the three gaps below.

## 3. Readings, fixed now

The quantities are means of paired per-layout differences over layouts with all values, on DeepSeek. CT4PJ comes
from `results/data/moving/runs_confirm4/dsv41`, which uses the same cached answers.
- **Place and follow:** OW − CT4OT.
- **Target:** CT4OT − CT4O.
- **Who:** CT4O − CT4PJ.

The split depends on the order of the steps. With a wrong target, who barely matters (cafe_1090).

**T. Is the test reachable?**
- **OW at least 0.7:** yes. The methods' 0.1-0.25 is the methods'.
- **OW 0.5-0.7:** reachable, with an inherent following cost. Each method is reported also as a share of OW.
- **Layouts with OW below 0.5** are listed with their reasons and reported separately. They are not dropped from the
  frozen confirm result.

**WA. Is walking away worth trying?**
- Walking away changes only who waits, after commitment. Its gain cannot exceed who: CT4O has the full truth ahead
  at every re-solve, which walking away never gets.
- **Who below 0.05:** walking away cannot reach a testable effect. 0.05 is the smallest effect with useful power at
  72 layouts (`CONFIRM_CT4PJ.md` §3). It stays future work.
- **Who at least 0.05:** design checks on the pilot logs, then a proposal and a fresh range, as agreed.

**TG. The target point.**
- If target is at least 0.05 and mostly at the front, the counter point is the first answer's job, not the
  re-solve's.
- One lead, flagged and not proposed: where served customers stood is observed in the wait's tracks.

**PF. Place and follow.**
- If OW − CT4OT is at least 0.10, where to stand given whom, and getting there, is the largest open part.
- `a_loss.py` on CT4OT splits it into the goal (geometry, including the 1.0 against 0.85 convention) and following
  (robot there, given goal right).

**Mixed:** all shares are reported. The next step goes to the largest share that is at least 0.05.

## 4. One request to the run (zero calls)
The 2 × 2 has one cell missing: CT4PJ with the true target and the model's roster ("CT4PJ-T", as CTt for CT3).
- Without it, who is measured only with the model's target, and on the layouts where the target is wrong it reads as
  zero.
- With it, who is also CT4OT − CT4PJ-T, and the order of the steps no longer decides the split.
- It is cache-only, like the other ceilings.

## 5. After the totals
