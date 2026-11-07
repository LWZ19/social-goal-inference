
Written before any round 5 belief was run. Motivation: `round4/RESULTS_D.md` (c5f99d64). The role belief's excess
false members over CT3F are:
- ex-members: served customers walking off, kept at P(waits) ≥ 0.5 for a median 4-6 s;
- unnamed walkers: half of them are moving at the time.

Its dynamics are generic: leaving at 0.05/s while walking, joining at 0.05/s after arriving and stopping. A queue is
a process with a kinematic signature: a person waiting moves only in short step-ups towards the front.

## 1. The change (QB = W0's belief plus the process's kinematics)

**A walking bout** is consecutive ticks at over `WALK_MPS` (0.3 m/s, the existing threshold). It runs from the
person's position 1 s before the bout's first walking tick.

**A step-up, from first principles:**
- one or two places forward (`STEP_MAX` 1.5 m, the existing parameter), towards the front;
- so the path length stays within `STEP_MAX` plus a 0.5 m margin for tracking and bends, which is 2.0 m;
- and the person does not end up farther from the target point than where they started by more than half a
  spacing, 0.5 m.

A bout breaks either bound when its path exceeds 2.0 m, or its net change in distance to the target exceeds
+0.5 m. From that tick until the bout ends, the leave hazard is `H_LEAVE` = 1.0/s, so about 0.4 a tick at 0.5 s.
Within the bounds, and while standing, the leave hazard is `H_STILL` (0.0001/s, existing): a step-up is not a way of
leaving.

Joining is unchanged: `H_MOVE` 0.05/s, after arriving and stopping. Everything else is W0's (W4 options:
`STATIC_WIN_S` 5, `WALKERS_STATIC` True, `LR_EXIT` 1.0).

New parameters, set a priori and not fitted:
- `BOUT_PATH_M` 2.0;
- `BOUT_AWAY_M` 0.5;
- `H_LEAVE` 1.0/s.

The descriptive bout table (`a_bouts_out.txt`) was computed before this note. It shows the separation exists:
- members' bouts: a median 0.7-0.9 m path and −0.6 to −0.7 m net;
- ex-members': +1.1 m net;
- never-members': about 2 m paths.

The bounds above come from the step definition, not from that table.

## 2. Why this is not E2 again

The first check below is exactly this false-fire rate on members.

## 3. Offline checks, in order (design data: round 3 design cells × 4 conditions, and the 24 pilot layouts, robot idle)

1. **False fires:** the share of "member, stays" bouts that break a bound, against ex-member and never-member bouts.
   - Bar: at most 5% of member bouts.
   - If it fails, stop here and report.
2. **Belief and choices**, QB against W0 (same decision: threshold roster, solver turn rule) and against CT4:
   - false-member ticks (non-waiters at P ≥ 0.5) split as `exmember.py` does: ex-member, named bystander, walker;
   - recall (true waiters at P ≥ 0.5);
   - right, cut, far and empty shares; last switches per minute;
   - pilot layouts only: proxy goal in line (`round4/a_proxy`).

## 4. What counts as promising, fixed now

All of these, QB against W0, on the pilot layouts and pooled over the design cells' 4 conditions:
- ex-member false-member ticks at least 40% fewer;
- walker false-member ticks fewer;
- recall within 0.02 of W0's;
- far share lower, and right share at or above W0's;
- empty share at most W0's + 0.01 (E2's failure mode);
- pilot proxy goal in line at or above W0's.

If all hold, it is worth a closed-loop pilot as QB plus CT4PJ's direction and joined rules (QBPJ), against CT4PJ and
CT3F. That pilot would be proposed only after the confirmatory result is in. If any fails, report which, and do not
propose it.
