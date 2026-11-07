# Moving people: scoring (written 2026-11-01, before any model episode; hashed in SCORING.sha256)

Follows the sim benchmark's RULES.md (2026-10-21) and `src/sgi/gt/regions.py`, applied to where the robot
actually stands, once a second, against the live line.

## The line at a sample

The line's way runs along the queue's slot polyline from slot 0 (served next) backwards. A point's `along` is its
arc length on that polyline (extended straight past either end), and its `side` its distance from it. The
members are the queue's people at that moment (humans only; the robot is never a member for scoring). A member
counts as **ahead of the robot** when their latest joining of the queue came before the robot's (before the
robot has joined, every member does). A served person leaves the queue; when they come back they join again, so
someone who rejoins after the robot joined is behind it.

## Per sample (every 1.0 s of sim time from the request to the end of the episode)

- **In line**, when some member is ahead of the robot: with L the member ahead of the robot that is farthest back
  along the line,
  - the robot is not ahead of L along the line (along(robot) >= along(L)), and
  - 0.5 m <= distance(robot, L) <= 1.2 m, and
  - the robot's heading is within 45 deg of facing up the line (the tangent of the polyline at the robot's
    along, pointing towards slot 0), and
  - the robot is at least 0.55 m from every person.
- **In line**, when no member is ahead of the robot (it is first): within 1.6 m of slot 0 and facing within
  45 deg of the bearing to slot 0's face point, at least 0.55 m from every person.
- **Cut in**: side(robot) <= 1.0 m and the robot is more than 0.25 m ahead along the line of some member who is
  ahead of it in queue order.

## Per episode

Primary:
- **In line at the end**: the last sample is in line.
- **Share in line**: the share of samples in line, from the request to the end.

Secondary: time from the request to the first sample in line (or none); cut-in samples (any, and their share);
lost place (an in-line sample followed by 3 or more consecutive not-in-line samples, after the first in-line
sample, counted per episode); model calls; the answers' latencies; re-solves; the first-match rate (CT).

## Comparisons

Arms are paired by (scenario, seed, model). Reported per arm: counts over episodes for "in line at the end" and
the mean share in line, with the per-episode table and traces. Few scenarios exist with moving lines, so results
are descriptive; a paired sign test over (scenario, seed) is reported for CT against each other arm, without a
claim of significance beyond what the count supports.

## Fixed parameters

Episode: request at t = 2 s, end at t = 92 s (90 s after the request). Re-ask interval S = 10 s (the sim's
fastest useful re-ask, below the shortest service time of 12 s). Re-solve every 0.5 s. Goal replace rule
0.3 m / 25 deg for every arm.
