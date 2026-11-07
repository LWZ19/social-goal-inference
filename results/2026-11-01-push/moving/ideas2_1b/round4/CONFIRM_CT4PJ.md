# Confirmatory test: the observer's place in the line (CT4PJ), on candidates 2000-2449

## 1. What the pilot found (design data, candidates 1000-1199; `n_ct4pj_release_out.txt` 882c7a71)

The role belief (CT4) answers "who is waiting?". Once the robot is in the line, its goal needs "who is waiting ahead
of me?". Without that rule the role arm gives its place to people who queued after it (`RESULTS_C.md`).

Adding CT3F's commitment rule (CT4PJ):
- strict share in line from the first service: CT4P 0.060 → 0.182, above on 14 layouts, below 4, tied 6, p 0.031;
- CT3F: 0.188, against which CT4PJ is better on 11 layouts, worse 8, tied 5, p 0.648.

Paired differences per layout:
- CT4PJ − CT4P: mean +0.122, sd 0.159;
- CT4PJ − CT3F: mean −0.006, sd 0.138.

Not robust, so not claimed: cut-ins. CT4PJ's mean is lower (8.9 against 12.1), but per layout it has more cut-ins
than CT3F on 10 and fewer on 6.

## 2. Hypotheses, tested in a fixed order (no multiplicity correction needed; a later one is claimed only if every earlier one passes)

- **H1 (the mechanism).** Adding the observer's place in the line raises time in line: CT4PJ > CT4P on the strict
  share in line from the first service.
  - Per layout d = CT4PJ − CT4P; ties |d| < 0.01.
  - Passes when wins > losses and the exact two-sided sign test over untied layouts gives p ≤ 0.05 (C1's test).
- **H2 (parity with the best hand-built package), tested only if H1 passes.** CT4PJ is not worse than CT3F by more
  than δ = 0.05 on the same measure.
  - Per layout d = CT4PJ − CT3F, over all layouts with both values.
  - Passes when the one-sided 95% lower confidence bound of the mean d, from the paired t interval with n − 1
    degrees of freedom, is above −0.05.
  - Layouts where both arms score 0 enter with d = 0. That is legitimate: on those layouts the two do equally, and
    the claim is about the average over this layout distribution. The scorer prints how many there are.
  - The scorer prints the lower bound itself, so a reader can apply a tighter δ.
- **Not tested:** CT4PJ > CT3F. The pilot shows parity, so a superiority test would be underpowered. Reported only.

## 3. Why δ = 0.05, and power

- At 72 layouts, with the pilot's sd of 0.138, the standard error is 0.0163 and t at 0.95 with 71 degrees of freedom
  is about 1.667. So the lower bound sits about 0.027 below the observed mean d.
- **H2's power** (true mean d −0.006):
  - about 0.85 at δ = 0.05;
  - about 0.43 at δ = 0.03.

  So 0.05 is the smallest margin with useful power. It is also about a quarter of CT3F's pilot level (0.188), and
  well under half of the deficit the rule removed (0.122). Fixed now, before any draw.
- **H1's power:** about 0.99 at 72 layouts, at an untied win rate of 0.78 with about 25% ties. With about 54 untied
  layouts it needs 35 wins.
- Per-layout shares are zero-heavy: in the pilot, 8 of 24 layouts were at 0.04 or less for every arm. At n 72 the t
  interval is adequate; the bootstrap (section 7) is the check.

- **Candidates:** `sweep/confirm_candidates`, from `make_sweep.py --start 2000 --per-base 450`.
- **Quotas:** the pilot's mix times 3: ours 48, chain 12, mixed 6, easy 6, half from each base (72 layouts, the same
  count as the 3 Nov sweep).
- **Stress flags** (lag, restless) are crossed 2 × 2 and assigned in turn along the distance order, balanced where
  the count allows. Mixed and easy have 3 per base, so they cannot be balanced within those cells.
- **Outputs:**
  - `sim/scenarios/confirm4`, `sweep/confirm_selection.json`, `sweep/jobs_confirm.txt`;
  - robot-idle dumps in `sweep/confirm_dump`;
  - runs in `runs_confirm4/dsv41` and `runs_confirm4/sonnet55_low`.

## 5. Arms and models

- **Primary model:** DeepSeek V4.1 Flash (free gateway), as in the pilot.
- **Arms:**
  - CT2: the first answers, cached and replayed by every other arm;
  - CT3F, CT4P, CT4PJ;
  - CT4: reported reference only, the original package.
- Every arm is as frozen and ported for the pilot: CT4P.md with CT4P_DEVIATION.md, and CT4PJ.md with its port
  clarifications. No parameter changes; CT4PJ has no parameter of its own.

3. DeepSeek: CT2 (first answers), then CT3F, CT4P, CT4PJ and CT4, replaying the cache.
4. Mechanical checks:
   - `check_mech`, and every arm replaying CT2's first answer with the same pose;
   - `round4/p_port_checks.py RUN_ROOT sweep/confirm_selection.json sweep/confirm_dump`: totals only, bars as in
     its docstring (0.99 match for CT4P's direction and CT4PJ's "behind", boundary cases allowed; no "behind"
     last; no fallback with a path).
   - Below a bar: stop and fix the port; scores unread.
5. Sonnet: announce, then the same steps 3 and 4.
6. `round4/o_confirm_score.py RUN_ROOT sweep/confirm_selection.json LABEL` once per model. Output saved, hashed and
   read-only, then sent.

**Missing values.** A layout whose CT2 first answer has no usable result, or which has no strict value for an arm of
a pair, drops out of that comparison. The scorer prints the count and the arms involved, never names. Per-layout
names are shared only after scoring.

A mechanical fault found is fixed and stated in a deviation note before scoring. Scores are never a reason to rerun.

## 7. Reported with the result, none claimed

- Means per arm; H1 and H2 by stratum.
- Right last, cut-in samples, joined share after the first service, median time joined before leaving.
- How often CT4PJ's own join test fires: episodes with at least one join, and re-solves joined. This is the scope
  limit from the pilot: the rule cannot engage where the role belief keeps the goal away from the robot
  (cafe_1031).
- The CT4PJ − CT3F sign test, and a paired bootstrap of the mean d (10,000 resamples, seed 2000) as a sensitivity
  check for H2.

## 8. What each outcome means, fixed now

- **H1 and H2 pass:** confirmed on fresh layouts. Without the observer's own place in the order, a belief over who
  waits loses time in line; with it, the single role belief is within 0.05 of the best hand-built package.
- **H1 passes, H2 fails:** the mechanism is confirmed, but parity is not. CT3F's other rules carry weight the role
  belief does not replace.
- **H1 fails:** the pilot's commitment effect did not replicate. H2 is reported only.
- Whatever the outcome, the next design is the process-structured belief (`RESULTS_D.md`), kept offline on design
  data until this result is in.

## 9. Frozen files

This note, `o_confirm_score.py` and `p_port_checks.py`. Dry runs of both on the pilot reproduce its numbers:
- H1 14/4/6, p 0.031;
- H2 mean d −0.006, lower bound −0.054 at n 24;
- port totals 3070/3070 and 1570/1577.

They are code tests, not results. Hashes are in `CONFIRM.sha256`.
