# Round 3 pilot follow-ups (3 Nov, written after the scorer's output, before these numbers)

## The outcome, and which reading applies

Scored once (`h_pilot_score_out.txt`, sha256 5f8b7d52). The outcome is **P4 fails with P1 split**:
- P2 and P3 held, and P5 held;
- P1 held for W0 (17/3/4) but not for CT5 (12/11/1);
- P4 failed for both, and CT5 is worse than CT4 (2/11/11, p 0.022).

That is not exactly the "P1-P3 hold, P4 fails" case that section 11 planned for. Halving roster changes did not
halve goal jumps for CT5 (21.1 against 22.6 a minute; CT3F 10.1). So the flicker account is incomplete twice over:
- CT5's goal moves for some reason other than roster changes;
- W0's goal does steady down (19.0, p 0.003), but its time in line does not rise (5/7/12).

The planned reading still applies: look at where the robot goes before any new test. No confirmatory test.

## X1 and X2, as fixed in ROUND3.md section 11 (3dde29fe) before any output

- X1: CT5 minus CT3F, strict share from the first service, on the 8 restless and the 8 not-restless "ours" layouts
  separately.
- X2: for CT4 and CT5 (CT3F and W0 added for reference), samples from the first service on that are not strictly in
  line. Each takes the newest re-solve sent at or before it, so the goal the robot was heading for. Split four
  ways, in this order:
  - (a) the solver's last is waiting and nearer the front than the true last (a cut);
  - (b) the solver's last is not waiting (a bystander or a walker);
  - (c) the solver's last is waiting and stands more than 0.6 m off the line's axis, as the sim's `Line.project`
    measures it;
  - (d) other: no last, or the true last on the axis. These are split by the first failing sim condition: side
    over 0.6 m, behind the last, gap outside 0.5-1.2 m, heading over 45 degrees, nearest person under 0.55 m.
  A cross-tab of waiting lasts (right, nearer the front, behind the robot) × (on or off the axis) is printed too, so
  the order of (a) and (c) does not hide anything.

## Post hoc, added after seeing the output (labelled as such)

- PH-A, why the goal jumps: over consecutive ok re-solves whose pose moved more than 0.3 m, the share where
  - (1) the solver's last changed;
  - (2) the last is the same and moved more than 0.15 m;
  - (3) the last is the same and still, and the roster or exempt set changed;
  - (4) none of these: the other people's positions moved the solver's answer.
  For every arm.
- PH-B, who the robot follows: for ticks where the solver's last is a bystander, how far that bystander stands from
  the axis and behind the true last along the line, and how long the robot stays behind the same bystander (the
  median run length, in seconds).

## Post hoc, added after X1, X2, PH-A and PH-B (`i_followups_out.txt`)

- PH-C, the goal or the getting there: for samples from the first service on, is the goal itself (the newest
  re-solve's `pose_world`) a strictly in-line place? Test it against that re-solve's tracked positions and the
  sim's rules: behind the last by distance along the line; gap 0.5-1.2 m to them; side at most 0.6 m; heading at
  most 45 degrees off up the line; nobody nearer than 0.55 m. Then, given a goal in line, is the robot strictly in
  line? Strict in line is about P(goal in line) × P(robot there | goal in line), so this says which factor each
  arm loses on.

## Results (3 Nov; `i_followups_out.txt`, `j_decompose_out.txt`, `k_geometry/*_out.txt`)

**X1 (fixed in advance).** On "ours" layouts, every role arm falls further behind CT3F on restless layouts than on
not-restless ones. Mean d in strict share from the first service:

| Arm | restless (n 8) | not restless (n 8) |
|---|---|---|
| CT5 | -0.21 | -0.10 |
| W0 | -0.16 | -0.06 |
| CT4 | -0.18 | -0.06 |

No role arm beats CT3F on any restless layout. With n 8, all sign p values are 0.06 or more.

**X2 (fixed in advance).** Samples from the first service on that are not strictly in line, by the goal's last:

| Arm | strict in line | (a) cut | (b) not waiting | (c) waiter off axis | (d) other |
|---|---|---|---|---|---|
| CT3F | 0.191 | 0.226 | 0.228 | 0.011 | 0.535 |
| CT4 | 0.072 | 0.227 | 0.354 | 0.056 | 0.364 |
| W0 | 0.075 | 0.246 | 0.385 | 0.042 | 0.327 |
| CT5 | 0.039 | 0.175 | 0.494 | 0.093 | 0.238 |

- CT5 trades cuts for following people who are not waiting.
- The "member stepped out" class (c) is real but small: 9% for CT5, 1% for CT3F.
- (d) is mostly "gap": the right last, but the robot is more than 1.2 m back.
  - For CT3F the robot is parked near its goal (median 0.24 m away), so the goal itself is too far back.
  - For the role arms the robot is mostly still moving, about 1.5 m from its goal (`gapcheck_out.txt`).

**PH-A (post hoc), why the goal jumps.** 59% of CT5's jumps are the same last moving more than 0.15 m. CT3F: 32%,
CT4: 29%. CT5's choice is steady, but it often follows someone who moves.

**PH-B (post hoc), who the role arms follow.**
- CT5's non-waiting lasts are mostly walkers: 1,055 walkers against 638 bystanders.
- They stand a median 1.54 m off the axis, against 0.57 m for CT3F's.
- CT5 stays behind the same one for a median 3.5 s (CT3F 2.1 s).

**PH-C (post hoc), the goal or the getting there.**

| Arm | goal in line | robot in line given goal in line |
|---|---|---|
| CT3F | 0.291 | 0.720 |
| CT4 | 0.156 | 0.354 |
| W0 | 0.142 | 0.379 |
| CT5 | 0.078 | 0.383 |

- Even CT3F's goal is a strictly in-line place under a third of the time.
- The first failing condition is mostly side: the goal is more than 0.6 m off the axis (CT3F 0.47, CT5 0.78), then
  gap over 1.2 m (0.11-0.13 for every arm).

**PH-D (post hoc), offline replay against the closed loop.** Offline replay predicted each role arm's choice classes
to within 0.06: right, cut (offline) or "a waiter nearer the front" (closed loop), and far (offline) or "a
bystander" (closed loop). CT5 far: 0.492 offline, 0.487 closed loop. What it got wrong was the consequence. The
frozen costs priced "behind a non-waiter" at 0.25 of a service, as if it meant standing a little farther back in
the line. In the closed loop that person is usually a walker or someone off the line, so the robot is out of line.
Offline replay can be trusted for what a decision rule will choose, but not yet for what that choice costs.

**Geometry (post hoc; `k_geometry/`).**
- With the right last and no non-waiter in the roster, goals are in line on:

  | CT3F | CT4 | W0 | CT5 |
  |---|---|---|---|
  | 0.64 | 0.40 | 0.43 | 0.24 |

  The "who" is the same on those ticks, so the gap is about "where" (`extras_out.txt`).
- **The direction of the line.**
  - CT3F aims along the tangent of the answer's remembered path at its back-most active member (`resolve.py`
    line 401), updated as people move.
  - The role arms aim along `RoleBelief.dir`: a straight chord from the first named person to the last, fixed at
    the photo (`roles.py` `_direction`; CT4_FROZEN section 2, "CT4 passes its own `dir`").
  - On those same ticks, goals are in line when the role arms' direction is under 10° off the true axis at the
    last: 0.52-0.63 for the role arms, like CT3F's 0.58. At 25° or more off: CT4 0.16, W0 0.21, CT5 0.01; CT3F
    0.64 (`geom_out.txt`).
  - By layout (`tiltlay_out.txt`), strict share from the first service:

    | Layouts | CT3F | CT4 | W0 | CT5 |
    |---|---|---|---|---|
    | 12 with direction error under 15° | 0.232 | 0.115 | 0.110 | 0.064 |
    | 12 with direction error 15° or more | 0.144 | 0.021 | 0.033 | 0.008 |

  - In 3 layouts the chord points the wrong way (150-180°).
  - So part of the C1 and pilot gap on "where" comes from the role arms' direction estimate, a part outside the
    belief over roles, and part does not: at small direction errors CT4 still has half of CT3F's time in line.
- **The vacated place is not the reason.** CT3F used it on 32 of 2,683 ticks, and none were in line.
- **Withdrawn: "once off the line, the goal stays off."** Pooled, goals are in line more often when the robot is on
  the axis. Within an arm and layout, at the first right tick after a wrong one, the robot being on the axis does
  not raise the share (9 higher, 12 lower, 16 tied; `transit_out.txt`). The pooled link is the robot following its
  goal, plus differences between layouts.
- **8 of 24 layouts are at 0.04 or less for every arm** (`zeros_out.txt`). Reasons are mixed:
  - goals more than 1.2 m back, with the right last (cafe_1008, sweet_1112, sweet_1003);
  - lasts that are not waiting (sweet_1092);
  - the robot ahead of the last (cafe_1090, cafe_1001).
  The strict measure has a floor here that no arm reaches past.

## What this means

- **The role belief's "who" is not what costs it most.** Time in line is lost at four separate steps:
  1. a wrong last, which for the role arms is often a walker or an off-axis bystander;
  2. the line's direction at its tail (role arms only, through the fixed chord);
  3. the distance behind the last (every arm: 11-13% of samples have goals more than 1.2 m back);
  4. getting there: the role arms' robots reach a good goal half as often as CT3F's.
- **CT5's cost model was wrong about geometry, not about probability.** The action space is places, not people.
  Coming after someone is a stand-in for a place, and it is only valid when that person stands on the line.
- **C1 and round 3 compared packages.** CT4, W0 and CT5 against CT3F differs in the direction rule as well as in
  who waits. The pre-registered results stand as package comparisons, but "the belief is worse at where to stand"
  should not be read as a property of the belief until the direction is equalised.
