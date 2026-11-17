# Round 7 results: the ceiling check, read by ROUND7.md (description only; for the project first)

Inputs:
- Runs: `results/data/moving/runs_ceiling4/dsv41`, 72 of 72 each of OW, CT4OT, CT4O and CT4PJT, all cache-only with zero calls. The
- CT4PJ and CT3F come from `results/data/moving/runs_confirm4/dsv41`, which replays the same cached answers.
- Each arm has 66 layouts with a window: 5 first answers have no target, and 1 episode has no service.
- Scripts and readings were hashed before the totals: `ROUND7.md`, `ROUND7_B.md`, `c_gaps.py`.
- Outputs: `c_gaps_ceiling_out.txt`, `a_loss_ceiling_out.txt`, `b_band_ceiling_out.txt`. `d_post.py` and
  `d_post_out.txt` were written after reading them and are post hoc description.

Nothing here picks a method or changes the frozen confirm result.

## 1. Levels and the 2 × 2 (strict share in line from the first service, DeepSeek, n 66)

Band top 1.2 m, with 1.5 m in brackets:

| | model's roster | true roster ahead |
|---|---|---|
| model's target | CT4PJ 0.159 (0.183) | CT4O 0.218 (0.252) |
| true counter point | CT4PJT 0.207 (0.238) | CT4OT 0.330 (0.372) |

- OW, the ideal place: 0.640 (0.652).
- CT3F: 0.208 (0.241).
- **Each fix alone adds about 0.05.**
  - Who: CT4O − CT4PJ +0.059.
  - Target: CT4PJT − CT4PJ +0.048.
- **Together they add 0.171.** The interaction is +0.064: knowing who is ahead pays twice as much when the target
  is right.

## 2. The fixed readings (identical decisions at 1.2 and 1.5 m)

Paired gaps:

| gap | mean at 1.2 (1.5) | front part | behind part | wins/losses/ties | sign p | lower bound |
|---|---|---|---|---|---|---|
| place and follow, OW − CT4OT | +0.310 (+0.280) | +0.249 | +0.061 | 52/12/2 | <0.001 | +0.235 |
| target, CT4OT − CT4O | +0.111 (+0.119) | +0.070 | +0.041 | 32/17/17 | 0.044 | +0.064 |
| who with the model's target, CT4O − CT4PJ | +0.059 (+0.069) | +0.005 | +0.054 | 35/14/17 | 0.004 | +0.021 |
| who with the true target, CT4OT − CT4PJT | +0.123 (+0.134) | +0.066 | +0.057 | 43/13/10 | <0.001 | +0.071 |

- **T: reachable, with a following cost.**
  - OW scores 0.640, not the 0.9 of the cafe_1090 smoke test.
  - As a share of OW: CT3F 0.33, CT4PJ 0.25, CT4OT 0.52.
  - 22 of 66 layouts have OW below 0.5 (listed with reasons in `d_post_out.txt` §1). They are reported, not dropped.
    - 17 of these 22 have lag (people wait 0-4 s before stepping up), against 16 of the other 44.
    - In 4 of them OW is never in line during the window.
  - OW's own loss is almost all while the robot is moving: walking up to slot 0 9%, following 8%, off the axis on
    the way 6%. Another 7% is people within 0.55 m while it is parked. This is the sim robot's following cost, not
    information.
- **WA: walking away may go on to pilot design checks.**
  - Who is at least 0.05 both ways (+0.059 and +0.123) and at both band tops, so the two measures agree.
  - See §3 for what this bound means against CT3F.
- **TG: met.**
  - Target is +0.111, mostly at the front (+0.070).
  - So the counter point is the first answer's job.
- **PF: met, and the largest gap.**
  - Place and follow is +0.310, mostly at the front (+0.249).
- **Predictions:**
  - P1 not held: OW 0.640, below 0.7.
  - P2 not held: CT4OT at the front is 0.422, below 0.5. CT4O is lower there (0.117), as predicted.
  - P3 held: CT4OT with someone ahead is 0.314, and place and follow is the largest gap.

## 3. Post hoc description (`d_post.py`, written after reading the above)

- **Against CT3F:**
  - CT4O − CT3F +0.010, p 0.78;
  - CT4PJT − CT3F −0.002, p 0.79;
  - CT4OT − CT3F +0.121, p 0.018, lower bound +0.064.
- **What this means for who-waits work:**
  - With the model's target, the full truth about who is ahead only reaches CT3F's level.
  - So walking away, or any other fix to who waits, can at best match the existing hand-built arm unless the target
    point is also right.
  - The bound in WA is real against CT4PJ, but not against CT3F.
- **Why CT4OT loses at the front even with perfect information:**
  - Its goal when first is a median 0.69 m from slot 0 (quartiles 0.66 / 0.98). Only 10.5% are within the strict
    0.6 m. The heading is right.
  - The solver's front place is 1.0 m from the counter point (`standoff_m`), about 42° off the line's axis. Slot 0
    is 0.85-0.95 m from the counter point, straight down the axis.
  - Like the 1.0 m against 0.85 m gap behind a person, this is the solver's geometry, not information. It must not
    be tuned to the test.
- **Why the front matters so much: a property of the test.**
  - The sim never serves the robot, so once it is first it stays first to the end of the episode.
  - OW is first for 43% of its window and passes 0.78 there. The methods are first for 12-20% of theirs.
- **Following is the same for every arm.** Given a right goal, the robot is there at about 0.57-0.60 of re-solves
  for every arm. OW passes 0.54 with someone ahead. Right goals don't fix following.
## 4. What this suggests (flagged leads, not proposals; each needs the project, design checks on pilot logs, and a fresh range)

1. **Where to stand at the front, and the target point, have one observation in common: where served customers
   stood.**
   - It is seen in the wait's tracks before the robot moves.
   - It would replace a guessed counter point and a geometric standoff with an observed serving spot.
   - It addresses the two largest gaps (PF front +0.249, TG +0.111). Walking away addresses neither.
   - It is not specific to coffee: "where people are served" is the same kind of observation as "where people wait".
2. **Walking away** passes its fixed bar against CT4PJ. Its ceiling over CT3F with the model's target is about
   +0.01, so on its own it is unlikely to improve the pipeline.
3. **One more zero-call cell would settle 1 against 2:** CT3F with the true counter point ("CT3FT", as CTt). If CT3FT
   is near CT4OT, the target point alone lifts the existing best arm and who-waits work adds little.

## 5. For the project: questions about the test, not changes
- **The front phase.** Should the never-served robot keep being scored, to the end of the episode, on 0.6 m from
  slot 0? This decides much of every good arm's score.
- **Sonnet's H1 sign test** passes at 1.2 m but not at 1.5 m (`ROUND7_B.md`; the mean is unchanged).
- Any change applies to every arm and is fixed before rescoring.
