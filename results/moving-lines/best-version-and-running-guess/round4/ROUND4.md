# Round 4: choose a place, not a person to follow (analysis, 3 Nov; design data only, zero calls)

Written before any round 4 decision rule was run. Evidence: `results/moving-lines/best-version-and-running-guess/round3/FOLLOWUPS.md` (dcbbe5d5).

## 1. The idea

The robot's action is a place. Today's rules choose people, and the solver turns those people into a place by
taking the farthest of them along a direction. That is a maximum twice over:
- CT4 takes everyone with P(waits) ≥ 0.5, then the farthest of them;
- CT5 comes after anyone with P(waits) at about 0.2 or more (cut 1 against far 0.25).

A maximum is as fragile as a statistic can be: one false member farther back moves the goal. That is the pilot's
failure. CT5 followed a walker or an off-axis bystander on 49% of ticks; CT4 on 34%.

The question the robot faces is "where is the end of the line?", not "who might be waiting?". Each waiting set W
in the belief implies one end, its last member last(W), and so one place behind it. So the posterior over sets
gives a posterior over places. The robot should go where that posterior is most concentrated: a mode, not a
maximum. A walker who is a member in a few sets adds a little mass to one more place. That moves the mode only if
those sets together outweigh the rest.

## 2. The rule (PL), fixed now

- **Belief:** W0's (`ct5.CT5` with W4's options: `STATIC_WIN_S` 5, `WALKERS_STATIC` True, `LR_EXIT` 1.0). CT4's
  belief is reported as a check.
- **Candidates:** "first" (nobody present waits) and each person tracked now.
- **Score:** P(last = j) = the sum of P(W) over sets W whose farthest present member (by `dist_now`) is j. For
  "first", the sum over sets with no present member.
- **Choice:** the candidate with the largest score. Keep the current one unless the best beats it by more than
  c_move × |d_best − d_current|, with c_move = 0.125 per metre (round 3's figure, from first principles: a 16 s
  horizon at 0.5 m/s). No other parameter.
- **What the solver gets:** as CT5. `come_after` = the choice plus roster members nearer the target than it;
  everyone else exempt.
- **Direction:** two variants, both reported:
  - PL-chord: `RoleBelief.dir`, as CT4 and CT5;
  - PL-path: the remembered path's tangent at the choice, as CT4P's rule. This is the variant meant for the
    sim, if any.

## 3. Offline measures (design data)

On the 24 pilot layouts (robot-idle dumps, CT2's first answers; `g_pilot_predict.pilot_cells`), and the round 3
design cells × 4 conditions (`r3lib.design_cells`, `CONDITIONS`):
- right / cut / far / empty shares (`r3lib.classify`), and last switches per minute;
- **pilot layouts only, goal in line (proxy):** the place 1.0 m behind the chosen last along the given direction,
  judged by the sim's rules against the true last (`round4/a_proxy.place_check`).
  - On the pilot's closed-loop re-solves, this proxy agrees with the solver's real goal on 0.87-0.93 of re-solves
    and is about 0.06 optimistic (`a_proxy.py`).
  - It does not model the robot getting there.

## 4. What counts as promising, fixed now

Offline on the 24 pilot layouts, PL-path against the same belief's threshold rule (W0) and against CT5:
- far share below W0's and CT5's;
- right share at or above W0's;
- proxy goal in line above W0's, CT5's and CT4's;
- switches per minute at or below W0's.

If all four hold, a closed-loop pilot of PL-path is worth proposing. CT4P's result decides whether the comparison
in the sim is against CT4P or CT4. If any fails, say which, and do not propose it.

Known limit: offline replay predicted choice classes well in round 3 (within 0.06). It did not predict time in line,
because it has no robot motion. The proxy covers goal placement only.
