# Moving line: where the time out of line goes (failure taxonomy, 2026-11-02)

Read-only analysis of the episode logs: no model calls, nothing run but `sgi.solver.solve` replays on logged answers.
Seeds 10-12, `dsv41_invalid_requery/` and `dsv41_repeat/` were not opened. Script: `failures.py` next to this file
(`python failures.py` prints every table here but section 2; `--matrix` prints section 2; `--episode
dsv41/cafe/seed0/CT` prints a case's timeline, one line a second, with each answer and re-solve between).
**Snapshot:** 00:40 on 2 Nov, 396 episodes (5
evaluation scenarios + cafe_busy; DeepSeek complete, x-test-3 and Sonnet still running, so their CTv/RT/RTw/Rq0/CT0
rows are partial), 36,036 one-second samples, 31,554 of them not in line (88 %). Re-running the script on more
episodes moves shares by a point or two; CT, C1, Cq, R1 and Rq are complete for all three models (51 episodes each).

## How a second is classified

Unit: a 1-s sample from the request to the end that is not in line by the frozen test (SCORING.md). A failure period
is a run of such samples. Each sample gets its most proximate cause from a fixed decision list (`classify`):

1. Before the first answer arrives: **latency**. From then until the robot first arrives somewhere: **approach**.
2. Nobody ahead (first): goal > 1.6 m from slot 0 or facing off the counter: **fa_front_goal** (the answer's target
   point; ask-once arms whose answer was right when asked: **stale_once**); goal fine but not there yet:
   **front_move**; there and facing right but someone within 0.55 m: **front_clear**.
3. Someone ahead. L (the member ahead farthest back) is rebuilt from the sample's gap and behind (exact along the
   line, side up to a sign). Robot in the band but facing > 45 deg off: **heading_goal** (arrived: its goal faces
   that way) or **heading_turn** (still turning). In the band, facing right, someone < 0.55 m: **clearance**.
4. L is > 0.5 m off the line's axis (walking back from being served) while the robot stands right behind the line as
   it physically is: **rejoiner_transit** (the sim counts the walker as the end of the line).
5. The robot's goal is a valid place (behind L, 0.5-1.2 m from L, **within 0.6 m of the axis**, heading within 45
   deg): **moving_valid** / **blocked** / **edge_stop** (stopped 0.1 m short, just outside the band) / **lag** (a dip
   of at most 4 s after the line moved, recovered).
6. A goal offered in the last 2 s that would be in line now was refused by the 0.3 m / 25 deg replace rule:
   **hysteresis**.
7. Otherwise the goal is wrong, and who set it decides. An answer (first or re-asked) is judged against the line as
   it stood in the frame it was asked on. Right when asked: **stale_once** (R1, C1, CT0) / **stale_reask** (Rq, Rq0,
   Cq). Wrong when asked: a C answer with no usable place in turn (turn_m > 0.3 or no come_after) is **fa_noturn**;
   otherwise **fa_** + where the goal is: *counter* (beside the line, level with or ahead of its end), *ahead* (in
   the line's corridor ahead of the last: a cut-in spot), *off* (beside the line, behind its end), *far* (on the axis,
   > 1.2 m back), *close* (< 0.5 m). CT family, goal from a re-solve: no place in turn after having one,
   **ct_ran_out** (**ct_ran_out_front** if the robot had already been first); with one: 1.2-1.5 m behind L
   **near_miss**, farther **ct_wrong_back**, ahead of L **ct_wrong_front**, beside **ct_off**. R-track whose first
   answer was right when asked: **rt_** + where.

## 1. All arms and models: seconds of failure by class

| Class | Group | Seconds | Share of failure time | Share after the first answer | Episodes with it |
|---|---|---|---|---|---|
| latency | late start | 8,880 | 0.281 | - | 396 |
| fa_off (beside the line, behind its end) | model answer | 3,495 | 0.111 | 0.154 | 134 |
| approach | late start | 3,483 | 0.110 | 0.154 | 395 |
| fa_noturn (C: no usable place in turn) | model answer | 3,070 | 0.097 | 0.135 | 70 |
| fa_counter (beside the line, at its front) | model answer | 2,813 | 0.089 | 0.124 | 122 |
| stale_once | arm design | 2,094 | 0.066 | 0.092 | 42 |
| stale_reask | arm design | 1,207 | 0.038 | 0.053 | 57 |
| fa_ahead (cut-in spot) | model answer | 1,179 | 0.037 | 0.052 | 68 |
| ct_ran_out | re-solve rules | 898 | 0.028 | 0.040 | 33 |
| fa_far | model answer | 842 | 0.027 | 0.037 | 44 |
| hysteresis | re-solve / replace rule | 565 | 0.018 | 0.025 | 40 |
| fa_front_goal | model answer | 467 | 0.015 | 0.021 | 31 |
| heading_turn | execution | 391 | 0.012 | 0.017 | 101 |
| ct_ran_out_front | re-solve rules | 368 | 0.012 | 0.016 | 10 |
| clearance | sim / scoring | 285 | 0.009 | 0.013 | 58 |
| heading_goal | model answer | 269 | 0.009 | 0.012 | 24 |
| front_move, moving_valid, lag, edge_stop, blocked | execution | 619 | 0.020 | 0.027 | 49-75 each |
| fa_close | model answer | 179 | 0.006 | 0.008 | 23 |
| near_miss, ct_wrong_back/front, ct_off, ct_front_turn | re-solve rules | 228 | 0.007 | 0.010 | 6-30 each |
| rejoiner_transit, front_clear | sim / scoring | 132 | 0.004 | 0.006 | 34, 16 |
| rt_* (R-track rule) | R-track rule | 90 | 0.003 | 0.004 | 8 |

By group (share of all failure time): late start 0.39 (latency 0.28), model answer 0.39, arm design (stale) 0.10,
re-solve and replace rules 0.07, execution 0.03, sim / scoring 0.01, R-track 0.003. There are 917 failure periods
(runs); by their dominant class: latency 257, fa_off 100, fa_counter 70, lag 65 (short dips), fa_noturn 60, stale
88, front_move 38, clearance 37, hysteresis 27, ct_ran_out 27 (+10 front), fa_ahead 27. The largest class after the
first answer, per episode: fa_off 74, fa_noturn 59, fa_counter 47, stale 70, approach 31, ct_ran_out 26 (+8),
hysteresis 21.

## 2. Per model and arm: share of episode time (episodes with the class in brackets)

Columns: lat = latency, app = approach, noturn = fa_noturn, beside = fa_off, counter = fa_counter, ahead = fa_ahead,
far = fa_far+fa_close, front = fa_front_goal, head = heading_goal, stale = stale_once+stale_reask, ran out =
ct_ran_out+_front, repl = hysteresis, rs = other re-solve classes, exec = execution, sim = sim / scoring. Seconds =
share x episodes x 91 (every episode has 91 samples).

| Model | Arm | Eps | In line | lat | app | noturn | beside | counter | ahead | far | front | head | stale | ran out | repl | rs | exec | sim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | CT | 51 | 0.16 | 0.30 | 0.11 | 0.08 (9) | - | - | - | - | 0.00 | 0.02 (8) | - | 0.17 (29) | 0.06 (22) | 0.03 (25) | 0.04 | 0.02 |
| all | CTv | 45 | 0.15 | 0.29 | 0.09 | 0.19 (13) | - | - | - | - | - | 0.02 (7) | - | 0.12 (14) | 0.06 (17) | 0.02 | 0.03 | 0.02 |
| all | C1 | 51 | 0.03 | 0.30 | 0.07 | 0.18 (15) | - | - | - | 0.02 | 0.00 | 0.01 | 0.37 (34) | - | - | - | 0.00 | 0.00 |
| all | Cq | 51 | 0.11 | 0.30 | 0.07 | 0.19 (30) | 0.01 | 0.02 | 0.00 | 0.00 | 0.00 | 0.01 | 0.21 (40) | - | - | - | 0.04 | 0.01 |
| all | R1 | 51 | 0.16 | 0.19 | 0.09 | - | 0.28 (28) | 0.09 (19) | 0.05 (7) | 0.06 | 0.03 | - | 0.05 (5) | - | - | - | 0.00 | 0.01 |
| all | Rq | 51 | 0.13 | 0.19 | 0.08 | - | 0.13 (36) | 0.25 (43) | 0.08 (18) | 0.04 | 0.02 | 0.00 | 0.03 (9) | - | 0.00 | - | 0.03 | 0.01 |
| all | Rq0 | 27 | 0.16 | 0.21 | 0.08 | - | 0.17 (20) | 0.17 (23) | 0.10 (13) | 0.01 | 0.02 | - | 0.04 (8) | - | - | - | 0.03 | 0.02 |
| all | RT | 30 | 0.13 | 0.22 | 0.18 | - | 0.20 (21) | 0.09 (17) | 0.05 (16) | 0.03 | 0.00 | 0.01 | - (rt 0.03) | - | - | - | 0.06 | 0.01 |
| all | RTw | 27 | 0.12 | 0.15 | 0.14 | - | 0.21 (21) | 0.18 (14) | 0.09 (12) | 0.04 | 0.02 | - | - | - | - | - | 0.04 | 0.01 |
| dsv41 | CT | 18 | 0.16 | 0.23 | 0.10 | 0.08 (4) | - | - | - | - | 0.00 | 0.02 | - | 0.23 (11) | 0.06 (6) | 0.04 | 0.05 | 0.01 |
| dsv41 | CTv | 15 | 0.17 | 0.23 | 0.09 | 0.22 (5) | - | - | - | - | - | 0.01 | - | 0.17 (5) | 0.06 (5) | 0.01 | 0.03 | 0.01 |
| dsv41 | C1 | 18 | 0.02 | 0.24 | 0.08 | 0.24 (6) | - | - | - | 0.06 | 0.01 | 0.00 | 0.35 (10) | - | - | - | 0.00 | 0.00 |
| dsv41 | Cq | 18 | 0.14 | 0.23 | 0.08 | 0.25 (15) | 0.01 | 0.05 | - | 0.00 | - | 0.02 | 0.15 (13) | - | - | - | 0.06 | 0.01 |
| dsv41 | R1 | 18 | 0.18 | 0.16 | 0.12 | - | 0.17 (6) | 0.04 | 0.13 (7) | 0.13 (6) | 0.04 | - | - | - | - | - | 0.00 | 0.03 |
| dsv41 | Rq | 18 | 0.15 | 0.16 | 0.11 | - | 0.13 (12) | 0.23 (16) | 0.06 (7) | 0.05 | 0.03 | 0.00 | 0.01 | - | - | - | 0.05 | 0.01 |
| dsv41 | RTw | 15 | 0.08 | 0.16 | 0.15 | - | 0.23 (11) | 0.08 (5) | 0.14 (8) | 0.07 | 0.02 | - | - | - | - | - | 0.05 | 0.02 |
| x3 | CT | 18 | 0.12 | 0.48 | 0.10 | 0.03 (2) | - | - | - | - | - | 0.01 | - | 0.13 (9) | 0.05 (9) | 0.01 | 0.05 | 0.02 |
| x3 | CTv | 15 | 0.14 | 0.48 | 0.09 | 0.10 (3) | - | - | - | - | - | 0.01 | - | 0.06 (4) | 0.06 (7) | 0.01 | 0.03 | 0.03 |
| x3 | C1 | 18 | 0.04 | 0.48 | 0.07 | 0.10 (4) | - | - | - | - | - | - | 0.31 (14) | - | - | - | - | 0.01 |
| x3 | Cq | 18 | 0.04 | 0.48 | 0.07 | 0.11 (5) | - | 0.00 | - | 0.01 | 0.01 | - | 0.27 (14) | - | - | - | 0.01 | 0.01 |
| x3 | R1 | 18 | 0.14 | 0.27 | 0.07 | - | 0.28 (11) | 0.02 | - | 0.04 | 0.03 | - | 0.13 (5) | - | - | - | 0.00 | 0.00 |
| x3 | Rq | 18 | 0.18 | 0.27 | 0.07 | - | 0.15 (12) | 0.13 (13) | 0.01 | 0.04 | 0.03 | - | 0.07 (7) | - | - | - | 0.03 | 0.01 |
| sonnet | CT | 15 | 0.20 | 0.18 | 0.11 | 0.14 (3) | - | - | - | - | 0.00 | 0.03 | - | 0.15 (9) | 0.09 (7) | 0.05 | 0.03 | 0.02 |
| sonnet | CTv | 15 | 0.16 | 0.18 | 0.10 | 0.25 (5) | - | - | - | - | - | 0.03 | - | 0.12 (5) | 0.08 (5) | 0.04 | 0.03 | 0.02 |
| sonnet | C1 | 15 | 0.04 | 0.18 | 0.07 | 0.22 (5) | - | - | - | - | - | 0.02 | 0.47 (10) | - | - | - | - | 0.01 |
| sonnet | Cq | 15 | 0.17 | 0.18 | 0.07 | 0.22 (10) | 0.03 | 0.01 | 0.01 | 0.01 | - | 0.01 | 0.21 (13) | - | - | - | 0.06 | 0.01 |
| sonnet | R1 | 15 | 0.15 | 0.12 | 0.07 | - | 0.42 (11) | 0.24 (12) | - | - | - | - | - | - | - | - | - | - |
| sonnet | Rq | 15 | 0.05 | 0.12 | 0.06 | - | 0.11 (12) | 0.43 (14) | 0.18 (9) | 0.01 | - | - | 0.02 | - | 0.00 | - | 0.01 | 0.00 |
| sonnet | RTw | 12 | 0.16 | 0.13 | 0.13 | - | 0.18 (10) | 0.30 (9) | 0.02 | 0.02 | 0.03 | - | - | - | - | - | 0.03 | 0.00 |

CT by scenario (share of CT time): cafe in line 0.09 (fa_noturn 0.21, ran out 0.14); room_coffee_cart 0.32
(hysteresis 0.12, clearance 0.08); cafe_line2_moving 0.17 (ran out 0.19, heading_goal 0.11); cafe_line6_moving 0.21
(hysteresis 0.17); sweet_spot_line2_moving 0.04 (ran out 0.19, fa_noturn 0.16); cafe_busy 0.10 (ran out 0.17).

### The first answers (one per model, scenario and seed)

| Model | C answers | come_after has the last | names only the front | no come_after | usable place in turn | C right when asked | R right when asked | people matched to members | target point on a person | latency C / R (median s) |
|---|---|---|---|---|---|---|---|---|---|---|
| dsv41 | 18 | 13 | 4 | 0 | 11 | 10 | 0 of 18 | 55 of 60 | 2 | 20.0 / 12.5 |
| x3 | 18 | 14 | 1 | 0 | 14 | 14 | 5 of 18 | 56 of 60 | 0 | 43.4 / 23.3 |
| sonnet | 15 | 10 | 0 | 3 | 10 | 10 | 0 of 15 | 46 of 48 | 0 | 15.2 / 11.3 |

fa_noturn seconds by what the first answer's come_after said: CT 384 = no come_after 195, names a non-member (the
cafe walker) 107, front only 39, has the last but unusable 43; CTv 783 = 199 / 424 / 52 / misses the last 65 / 43;
C1 844 = 154 / 511 / 114 / 65 / 0. Cq adds 323 s whose first answer was fine: its re-asked answers lose the place in
turn (views from beside the line).

## 3. Upper bounds by where the fix lives (share of episode time that would be in line if fixed perfectly)

Not additive: fixing one class exposes the next (e.g. a kept line meets the replace rule).

| Arm | In line | (a) model: answer | (a') model: latency | (b) re-solve / replace rules | (c) sim / scoring classes | arm design (stale) | approach (travel) | execution |
|---|---|---|---|---|---|---|---|---|
| CT | 0.16 | 0.11 | 0.30 | 0.27 (ran out 0.17, replace 0.06, other 0.03) | 0.02 | - | 0.11 | 0.04 |
| CTv | 0.15 | 0.21 | 0.29 | 0.20 (0.12, 0.06, 0.02) | 0.02 | - | 0.09 | 0.03 |
| C1 | 0.03 | 0.21 | 0.30 | - | 0.00 | 0.37 | 0.07 | 0.00 |
| Cq | 0.11 | 0.24 | 0.30 | 0.00 | 0.01 | 0.21 | 0.07 | 0.04 |
| R1 | 0.16 | 0.51 | 0.19 | - | 0.01 | 0.05 | 0.09 | 0.00 |
| Rq | 0.13 | 0.52 | 0.19 | 0.00 | 0.01 | 0.03 | 0.08 | 0.03 |
| RTw | 0.12 | 0.54 | 0.15 | 0.00 (R-track) | 0.01 | - | 0.14 | 0.04 |

(c) is small as a cause of failure, but the scoring tolerances move the in-line share itself by more than most arm
differences (table 4a): a 1.5 m band adds +0.08 (CT) to +0.11 (R1); counting only robots on the axis and "first"
only within 0.6 m of slot 0 removes 0.04 (CT) to 0.13 (R1).

## 4. Counterfactual checks

**4a. Rescoring the logged samples** (all models; "strict" = on the axis within 0.6 m when someone is ahead, and
first only within 0.6 m of slot 0 facing its face point):

| Arm | In line (frozen) | band to 1.5 m | heading 60 deg | first within 0.6 m | on the axis | both strict | in-line time while first | in-line time beside the line |
|---|---|---|---|---|---|---|---|---|
| CT | 0.158 | 0.236 | 0.184 | 0.126 | 0.150 | 0.118 | 0.21 | 0.05 |
| CTv | 0.154 | 0.229 | 0.176 | 0.114 | 0.151 | 0.111 | 0.27 | 0.02 |
| C1 | 0.032 | 0.054 | 0.039 | 0.032 | 0.025 | 0.025 | 0.00 | 0.24 |
| Cq | 0.115 | 0.142 | 0.119 | 0.083 | 0.106 | 0.073 | 0.48 | 0.08 |
| R1 | 0.158 | 0.264 | 0.159 | 0.085 | 0.097 | 0.024 | 0.46 | 0.38 |
| Rq | 0.130 | 0.187 | 0.140 | 0.111 | 0.098 | 0.079 | 0.23 | 0.25 |
| Rq0 | 0.164 | 0.216 | 0.170 | 0.150 | 0.133 | 0.120 | 0.20 | 0.19 |
| RT | 0.126 | 0.181 | 0.131 | 0.119 | 0.062 | 0.056 | 0.08 | 0.50 |
| RTw | 0.117 | 0.165 | 0.123 | 0.112 | 0.041 | 0.035 | 0.05 | 0.65 |

With the sim's own corridor (0.8 m) instead of 0.6 m, "both strict" is CT 0.124, R1 0.030, Rq0 0.132, RT 0.101.

**4b. Goal-stream replays (CT family).** The logged first answer and every logged re-solve are replayed through a
changed rule and the goal the robot would hold is judged against each sample's line (robot assumed at its goal, so
an upper bound that ignores travel). Share of post-answer samples whose goal is a place in line:

| Arm | Samples | As run (0.3 m / 25 deg) | replace past 0.1 m / 10 deg | always replace | standoff 0.85 m (pose 0.15 m up the line) | both |
|---|---|---|---|---|---|---|
| CT | 3,226 | 0.365 | 0.422 | 0.422 | 0.418 | 0.435 |
| CTv | 2,887 | 0.361 | 0.411 | 0.412 | 0.421 | 0.425 |

**4c. `sgi.solver.solve` re-run on the logged first answers** (C1 frames; reproduces all 51 logged poses exactly):
right when asked, as logged / standoff 0.85 m / come_after = the true members (unseen ones added where they stood) /
no `face` constraint / last two together: dsv41 10 / 11 / 12 / 11 / 13 of 18; x3 14 / 14 / 15 / 14 / 15 of 18;
Sonnet 10 / 11 / 12 / 10 / 12 of 15. Of the 17 wrong C first answers, the 8 outside cafe are each fixed by one change
(6 by the full come_after, 1 by dropping `face`, 1 by the 0.85 m standoff); none of the 9 cafe answers is fixed by
any of them (below). R first marks snapped onto the line's axis at the same distance along it: right when asked
dsv41 0 -> 4, x3 5 -> 7, Sonnet 0 -> 2: R's errors are mostly in the order, not just sideways.

## 5. The top classes, with cases

**A. Late start (latency 0.28 + approach 0.11 of all failure time).** The robot cannot move for the first answer's
latency (median C: x3 43 s, DeepSeek 20 s, Sonnet 15 s), longer than one 15 s service.
- `x3/cafe_line2_moving/seed0/CT`: first answer after 33.8 s (t = 35.8); after it the robot is in line 48 of 57 s.
- `dsv41/sweet_spot_line2_moving/seed2/CT`: answer at 27.6 s, after the first service; the named front person is
  already gone, the approach takes 23 s and ends beside the counter (case E3): 0 of 65 s.
- Counterfactual: a faster first answer. Check (computed, not simulated): CT's in-line share after the first answer
  is the same for all three models (DeepSeek 0.22, x3 0.23, Sonnet 0.24), so x3 at Sonnet's latency would gain about
  (0.82 - 0.52) x 0.23 = +0.07 of episode time. Travel itself (approach 0.09-0.11) is mostly irreducible at 0.6 m/s.

**B. No usable place in turn in the C answer (fa_noturn: 0.10 of failure time, 70 episodes).**
- `dsv41/cafe/seed0/C1` (and CTv, CT): from 6.5 m a walker hides the third member; come_after [4, 5] names the front
  person and someone behind the counter (not a member); turn_m 6.8; the solver keeps the robot at its start (8.31,
  -6.77) for 63 s (CT, with the map: 53 s). Should: queue at about (5.0, -3.8). Replay with come_after = all three
  true members: turn_m still 4.3, same pose. All 9 cafe first answers (3 models x 3 seeds) fail this way: the place
  in turn is not on the floor the first view shows, whatever the constraints say. Fix: perception of the occluded
  member and the map floor (CT has the map but its come_after still includes the walker).
- `sonnet55_low/sweet_spot_line2_moving/seed2/CT`: no come_after at all (Sonnet: 3 of 15 first answers); the robot
  stands beside the front person at the counter (14.78, 4.33) for 70 s. Replay with come_after = the two members:
  0.94 m behind the last, facing up the line: right when asked.
- `dsv41/cafe_line6_moving/seed2/CT`: the plan's target is "person at the back of the queue", so the target point is
  on the last person, 4.5 m from the counter; come_after [1] names only that person and is "unusable: nobody is
  listed but the target person". The first pose happens to fall 1.1 m behind that person, but no re-solve ever has
  a place in turn: the CT robot (answer at 45.7 s) stands 1 m beside the line facing that person (`face [1]`, 135 deg
  off) for 43 s. Replay with the target point moved to the counter (CTt-style): the come_after becomes usable
  (turn_m 0.07), 1.12 m behind the last, 3 deg off. A solver rule that reads "come after [the target person]" as
  "join behind them" would do the same without an oracle.

**C. R's marked point beside the line (fa_off 0.11 + fa_counter 0.09 of failure time; 0.29-0.39 of each R arm's time).**
- `sonnet55_low/cafe_line6_moving/seed0/R1` (seeds 1, 2 alike): mark at (6.19, -4.09), 1.2 m to the side of the fifth
  person, ahead of the end; 73 s, never in line.
- `sonnet55_low/sweet_spot_line2_moving/seed2/Rq`, t = 61: a re-asked mark puts the robot at (12.21, 4.40), 3.6 m to
  the side, level with the counter.
- `dsv41/sweet_spot_line2_moving/seed2/R1`: mark at (14.59, 2.93), 1.2 m beside the line; it scores in line 60 of 74 s
  because the in-line test checks distance to L and order, never the side. Strict scoring removes it.
- Counterfactual: snap the mark to the axis (4c): +4 / +2 / +2 first marks only. R's marks are right when asked in 5
  of 51 first answers (C's: 34 of 51). This class is the model's, and the R arms' in-line share mostly comes from the
  lax scorer (4a, surprise 1).

**D. Stale goal (stale_once 0.07 + stale_reask 0.04 of failure time).** Right when asked, the line moved.
- `dsv41/cafe_line2_moving/seed0/C1`: first answer (5.09, -3.14), 0.99 m behind the second person when asked;
  delivered at 12.4 s; service at 16 s; the robot settles 1.54 m behind and is never updated: in line 3 of 80 s.
  The same answer re-solved (CT) is in line 0.66 of the episode.
- x3 Cq: re-asked answers take ~43 s against 15-25 s services; stale_reask is 0.27 of x3 Cq time (14 of 18 episodes).
- Counterfactual: re-solve on tracks (that is CT). Check, paired by (model, scenario, seed) and second: the CT robot
  is in line in 29 % of C1's stale seconds (498 of 1,727) and 30 % of Cq's (289 of 973). Re-solving fixes staleness
  where it is the only problem; the rest of those seconds go to E, F and late start.

**E. CT: the line ran out (ct_ran_out 0.126 + ct_ran_out_front 0.044 of CT time; 29 of 51 CT episodes).** Every
person the answer named has been served and dropped, while people still queue: a person who comes back stays a
dropped number, and the solver can only absorb newcomers behind a current last, so an empty come_after has no line.
The solver falls back to the standard goal by the model's target point, beside the counter.
- `dsv41/sweet_spot_line2_moving/seed0/CT`: in line 20-33 s; first at 33 s; the re-solve at 34.8 s has nobody to come
  after and sends the robot to (14.77, 4.41), 0.95 m beside slot 0; past 0.8 m from the axis the sim drops it from the
  queue (39 s), so the people who come back count as ahead: 54 s out of line. Should: step up to slot 0, face the
  counter.
- `dsv41/cafe_line2_moving/seed1/CT`: the same at 35.8 s, goal (5.92, -1.49), dropped from the queue at 39 s: 49 s.
- `dsv41/sweet_spot_line2_moving/seed2/CT` (not first): late answer (27.6 s); the front named person was served
  before it, came back at 28 s as a dropped number; the second was served at 33 s; at 35.1 s the come_after is empty
  and the robot, never joined, goes beside the counter while the returned person stands at slot 0: 42 s.
- Counterfactuals: (i) once first with nobody left, hold the front's place facing the target: upper bound 262 s, 5.6 %
  of CT time (13 episodes), all scored in line because the robot is never served (an artefact, so report it apart).
  (ii) When the come_after empties, re-seed it with the tracked people standing on the remembered axis ahead of the
  robot: upper bound 12.6 % of CT time; in 72 % of these seconds (CT and CTv) the last person ahead does stand on the
  axis and in 95 % the robot is beside the line, so there is a line to join. Not replayable: per-person tracks are not logged.

**F. The replace rule (hysteresis 0.064 of CT time, CTv 0.065; 40 episodes).**
- `x3/cafe_line6_moving/seed1/CT`, 69-87 s: a re-solve at 69.4 s, while the person ahead was still stepping up, put the
  goal 1.37 m behind; every re-solve from 70 s offered (5.05, -4.70), 1.1 m behind, but 0.25 m from the held goal,
  under the 0.3 m rule: 17 s at 1.37 m. Replay (4b) with a 0.1 m / 10 deg rule: goal in line 41 of 42 post-answer
  seconds instead of 19.
- `dsv41/room_coffee_cart/seed2/CT`, 28-54 s: goal 1.12 m behind (valid), the robot stops 0.09 m short at 1.21 m
  (waypoint tolerance 0.10 m); re-solves at 1.0 m are 0.11 m away and refused: 27 s just outside the band. A 0.85 m
  standoff puts the stop at about 1.06 m.
- Over all CT episodes (4b): goal in line 0.365 -> 0.422 with the finer rule, 0.418 with the 0.85 m standoff (1.0 m
  sits 0.2 m from the band's 1.2 m edge), 0.435 with both. The near miss proper (gap 1.2-1.5 m from a mid-step re-solve,
  no better offer) is only 0.5 % of CT time.

Smaller classes, for the candidates the brief named: heading_goal 0.9 % of failure time (CT 2.1 %, mostly
cafe_line2_moving: the face constraint); wrong membership in the resolver (near_miss + ct_wrong_* + ct_off: CT 3.0 %;
e.g. `dsv41/cafe_line2_moving/seed0/CT` 19-28 s, just after joining, someone not in the line (the person just
served, walking off, or a walker) is taken as the new last and the goal goes 0.5 m off the axis for 8 s); moving/chasing and blocked 0.6 %; people missed or mislocated: 157 of 168 queue
members matched within 0.75 m, the misses are cafe's occluded member.

## 6. Surprises and contradictions with the brief

1. **R1's share is mostly the scorer.** 46 % of R1's in-line seconds are the robot standing first (never served,
   1.6 m tolerance) and 38 % are beside the line (the test never checks the side; R marks land 0.7-1.3 m beside the
   last person). Strict: R1 0.158 -> 0.024, CT 0.158 -> 0.118, Rq0 0.164 -> 0.120. "R1 0.20 vs CT 0.18" is not a
   like-for-like baseline result.
2. **CT's biggest own loss is the line running out (17 % of CT time), not hysteresis (6 %)**, and the brief does not
   list it. It hits the two-person lines (and cafe_busy's fast service) hardest, and slow answers feed it.
3. **The near miss is a standoff-vs-band problem as much as timing**: the place in turn is 1.0 m behind, the band ends
   at 1.2 m, and the robot stops up to 0.1 m short.
4. **"Face the target" is small as a proximate cause** (0.9 % of failure time; a 60 deg tolerance adds +0.026 to CT).
   The dev episode overstated it.
5. **cafe fails for every model on visibility, not constraint content**: the last member is occluded and the place
   behind them is off the seen floor; a perfect come_after does not move the first pose.
6. **After the first answer, CT is model-independent** (0.22-0.24 in line); x3's lower CT share is all latency.
7. **Sim inconsistencies**: the robot leaves the queue past 0.8 m from the axis (`CORRIDOR_M`; DESIGN says 0.6) while
   "first" is scored within 1.6 m, so a first robot 0.95 m beside slot 0 is in line one moment and out of the queue the
   next; room_coffee_cart has 4 slots, so a person returning behind the robot is sent to the robot's own slot
   (nearest 0.19 m; clearance 8 % of CT time there); the join rule takes the farthest-back member, including one still
   walking back, so a robot correctly behind the line can be refused membership (rejoiner_transit, 0.3 %).
8. **Cq's re-asks lose a good first answer**: 323 s of Cq fa_noturn come from re-asked answers after a first answer
   with a usable place in turn.
