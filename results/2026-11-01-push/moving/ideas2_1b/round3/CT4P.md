# CT4-P: CT4 with CT3F's line direction (a control, not a candidate). Written 3 Nov before any CT4-P run.

## Why

`FOLLOWUPS.md` (dcbbe5d5) found that CT4, W0 and CT5 against CT3F differ in two things: who waits (the role
belief), and the direction the solver lines up along.
- CT3F uses the tangent of the answer's remembered path at its back-most active member (`resolve.py` line 401).
- The role arms use `RoleBelief.dir`, a straight chord fixed at the photo.

With the right last and a clean roster, the role arms' goals are in line far less often when the chord is 25° or
more off the true axis. Half of the pilot layouts have a direction error of 15° or more. CT4-P equalises the
direction, so that CT4 against CT3F measures the belief.

## The arm

- CT4 exactly: `role_belief` True, no window, roster = P(waits) ≥ 0.5, `come_after` = roster, everyone else
  exempt.
- Only change: the solver's `line_direction`.
  - It is the remembered path's tangent (`Resolver._on_path(xy)[2]`) at the roster member farthest back by path arc
    length (`Resolver._along`), among those tracked now.
  - The path is the answer's named waiting people at the answer, as CT3F builds it. So the role arm's Resolver is
    built with `path_memory` True and `line_span_m` 0.5, which `_path` needs.
  - Fallback to `RoleBelief.dir` when there is no path or no roster member on it.
- Logged per re-solve: `solve_direction` and which fallback was used, if any.
- Zero model calls: the 24 pilot layouts, CT2's cached first answers, DeepSeek. Sim time only, about one arm's worth.

## Groups, fixed now from `k_geometry/tiltlay_out.txt`

- High direction error (CT4's chord 15° or more off the true axis at the true last, median over ticks):
  cafe_1002, cafe_1000, cafe_1001, cafe_1160, cafe_1031, sweet_1112, sweet_1130, cafe_1081, sweet_1011, cafe_1183,
  cafe_1090, sweet_1129.
  - Informative, meaning CT3F is at 0.10 or more there: cafe_1002, cafe_1000, cafe_1160, cafe_1031, sweet_1130.
- Low direction error: the other 12.

## Predictions (measure: strict share in line from the first service, `episode_scores`' `after_service_strict`)

- **Q1.** On the 5 informative high-error layouts, CT4-P > CT4 on at least 4.
- **Q2.** On the 12 low-error layouts, |mean(CT4-P − CT4)| < 0.03: the direction barely matters when it is already
  right.
- **Q3.** Goals in line with the right last and no non-waiter in the roster (`k_geometry/extras.py`'s
  definition), pooled after the first service: CT4-P ≥ 0.55 (CT4 0.40, CT3F 0.64).
- **Q4.** Overall (24 layouts), CT3F stays ahead of CT4-P on more layouts than not. The belief's extra
  non-waiters and the getting-there cost remain.

## Readings, fixed now

- **Q1 and Q3 hold, Q4 holds:** the direction rule explains the high-error part of the gap. What is left is the
  belief's own cost: following walkers and off-axis people, and transit. Next is the place-based decision
  (choose places, not people), designed offline.
- **Q1 fails:** the direction is not what separates the high-error layouts; they differ in other ways. The "where"
  deficit is about whom the belief follows. Go straight to the place-based decision.
- **Q4 fails (CT4-P at or above CT3F on more layouts):** once the geometry is equal, the belief is not worse at
  where to stand. Draft a confirmatory note for CT4-P against CT3F on candidates 2000-2449.
- Every outcome is reported. Nothing here is tuned. CT4-P has no parameter of its own.

- `check_mech` clean; every CT4-P episode replays CT2's first answer from the cache with the same pose.
- `l_ct4p_score.py` prints the share of re-solves using the fallback. Its direction check: the median angle
  between `solve_direction` and the true axis at the true last is below 15° on the high-error layouts. If not, the
  port is wrong; stop and fix it.
- Logging: today's sim episodes record `line_direction` but have `solve_direction` as None (checked on CT4 and
  CT3F pilot episodes). CT4-P episodes must record `solve_direction` (the path tangent actually used) and
  `direction_source` ("path" or "belief"). The scorer falls back to `line_direction` only so the dry run works.
- Dry runs of the scorer with CT4 and CT3F standing in for CT4-P both print a direction-check FAIL (36.8°, the
  chord), as they should. They are code tests, not results.
