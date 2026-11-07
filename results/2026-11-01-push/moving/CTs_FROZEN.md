# CTs: re-solve on settled tracks (frozen 2026-11-01 ~23:55, before any x-test-3 or Claude CT result was looked at)

Origin: the redraw-floor check (QUESTIONS Q13) on DeepSeek evaluation cells showed CT's outcome depends on whether a
re-solve falls while people are still stepping up, and that the 0.3 m replace rule can then hold a near miss. The
fix below was suggested by those cells, so it is judged only on fresh simulator seeds (10, 11, 12; the Habitat
random seed, unrelated to the scene-generator seeds reserved elsewhere).

## Definition (general; no situation logic)

- **Settled re-solve.** At each re-solve time (every 0.5 s, as CT), the resolver looks at every tracked person within
  3.0 m of any person currently in the waiting set (or, before the first solve, of any matched person). If any of them
  has moved more than 0.10 m over the last 1.0 s of tracks, the solve is skipped and the last solved goal is returned
  unchanged ("held"). Otherwise it solves exactly as CT does. The first solve is never held.
- **Replace rule.** The robot's goal is replaced when the new place differs by more than 0.3 m or 25 deg (as CT), or
  when the current goal is no longer valid: some tracked person's centre is within 0.55 m of it (person radius +
  robot radius).
- Everything else is CT: same first answer (cached), same resolver rules, same floor (the navigation map), same
  scoring (SCORING.md and the post hoc measures of Q9/Q10/Q12).

## Runs

## Prediction (questioner's, written before the run)

CTs keeps CT's keep-up, has fewer lost-place events, and a smaller redraw spread (mean |share difference| between
the two runs of a cell) than CT.

## Reading

Reported per model and pooled: keep-up, lost place, goal left the line, share in line, redraw spread of CT and CTs.
"Fewer lost place" and "smaller spread" are judged descriptively (paired counts over 15 cells per model); no
significance claim beyond what the counts carry.
