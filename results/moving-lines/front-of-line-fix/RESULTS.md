# Front-of-line fix: an exploratory check (7 Nov)

**Question.** The moving-line video shows the robot, once first in line, back away round people and wait beside the
counter. Two causes were found in its log: a backward goal move was accepted (only small backward moves are blocked),
and the "take the place of the last person served" rule was cancelled because that person came back and rejoined
behind the robot.

**Change (`CT2f`, and `CT3Ff` for the best version, not run here).** Two rules for any waiting line:
1. once the robot is in line, a new spot farther back along the line is ignored while its current spot is free;
2. `vacate_rejoined` (`src/sgi/resolve.py`): someone who comes back to the line behind the robot no longer cancels
   taking the place of the last person served.

**Test.** `CT2f` against unchanged `CT2` on the same 25 layouts per model (five scenarios, seeds 10-14), DeepSeek V4.1
and Claude Sonnet 5.5, replayed from stored answers (no model calls). Not planned in advance: exploratory only.

**Result** (`front_score_out.txt`):

| | DeepSeek | Sonnet |
|---|---|---|
| Time in the right spot, CT2f vs CT2 | 0.288 vs 0.277 (10/8/7) | 0.331 vs 0.324 (12/9/4) |
| Seconds stepping back after joining | 6 vs 17 | 2 vs 23 |
| At the front, time in the right spot | 0.052 vs 0.017 | 0.000 vs 0.047 |

Over both models: time within 0.55 m of someone 2.0% vs 2.9%, runs with a moment within 0.3 m 6 vs 10.

**Reading.** Rule 1 works as meant: far fewer backward steps, and the robot comes close to people less often. The
score does not change measurably, because the robot still stands in the wrong place at the front: its idea of where
the counter serves comes from the model's first answer, and the place of the last person served is often off the line
too. The test robot told the true serving spot (`x1_ceilings`) remains the fix that matters.
