# CT4-PJ: CT4P plus CT3F's joined rule (a control, not a candidate)

## Why

`RESULTS_C.md` (87d25e3d):
- Strict in line is almost all time spent joined.
- CT3F's Resolver stops counting people behind the robot once it has joined (`resolve.py` lines 366-372 and 410).
  The role arms have no such rule. While joined, their solver last stands behind the robot on 8-18% of re-solves
  (CT3F 0.6%), and 79-90% of those people joined after the robot.

CT4-PJ adds that rule to CT4P, so the role arm and CT3F share both parts outside the belief: the direction rule and
the commitment rule.

## The arm

- CT4P exactly (`CT4P.md`, as ported: `CT4P_DEVIATION.md`), plus the following.
- **Join test, as `Resolver._inputs`:**
  - the previous re-solve was in turn (its logged `in_turn`);
  - the robot is within `join_radius_m` (0.35 m) of that re-solve's `pose_world`;
  - this has held for `join_dwell_s` (1.0 s);
  - once joined, joined for the rest of the episode, as the Resolver.
- **Once joined:** tracked people farther back than the robot are left out of the solve: out of `people`,
  `come_after`, and the points for the direction.
  - "Farther back" is measured as in `Resolver.inputs` line 410: arc length on the remembered path (`_along` of the
    geometry Resolver) when there is a path, otherwise distance to the target point.
  - The belief itself is not changed; it keeps stepping on all tracks.
- **Logged:** `joined` (the arm's own flag) and the ids left out as behind.
- No new parameter: the join radius, dwell and "farther back" rule are the Resolver's.
- Zero model calls: the 24 pilot layouts, CT2's cached first answers, DeepSeek.

## Predictions (24 pilot layouts; CT4P is the comparison)

- **J1 (mechanism).** While the sim has the robot joined, the share of re-solves whose solver last stands more than
  0.25 m farther back along the sim's line than the robot (`d_joined/behind.py`'s definition): CT4-PJ below 0.03.
  CT4 was 0.083 and CT3F 0.006 in the pilot.
- **J2.** Median time joined before leaving (sim `joins`): CT4-PJ above CT4P.
- **J3.** Joined share of samples after the first service: CT4-PJ above CT4P on more layouts than below.
- **J4 (outcome).** Strict share in line from the first service: CT4-PJ above CT4P on more layouts than below
  (ties: |d| < 0.01).
- **J5 (the question, a direction stated).** CT4-PJ against CT3F on the same measure: CT3F ahead on more layouts
  than not. Before joining, the belief's extra non-waiters (walkers, off-axis bystanders) still cost it.

## Readings, fixed now

- **J1 fails:** the port is wrong; fix it before reading anything else.
- **J1 holds, J4 holds, J5 fails** (CT4-PJ at or above CT3F on more layouts): the belief's "where" deficit was the
  package parts. Draft a confirmatory note for CT4-PJ against CT3F on candidates 2000-2449.
- **J1 holds, J4 holds, J5 holds:** commitment matters. What is left is the belief's own errors before joining,
  about who stands behind the line. Next: the belief's evidence about people behind the line, designed offline.
- **J1 holds, J4 fails:** commitment is not what limits the role arm. Look at joining (J3) and the time before the
  first join.
- Every outcome is reported. Nothing is tuned.

## Scorer and checks

- `m_ct4pj_score.py` (to be frozen alongside this note).

## Port clarifications (run's two notes, added at freezing; predictions and readings unchanged)
1. **Where the join test's in-turn flag and place come from.** In belief mode `Resolver.record()` is never called.
   So each re-solve records the solver's own `in_turn` (`turn_m` ≤ 0.3, as logged) and the solved `pose_world`,
   as the Resolver's `record()` would. The next re-solve's join test reads those.
2. **"Behind the robot".** `_along` on the geometry Resolver's remembered path when it has a path, as
   `resolve.py` line 410 does with `path_memory`; otherwise distance to the target point. A person is left out
   when their measure exceeds the robot's (the Resolver keeps those with `_along(p) <= mine + eps`).
