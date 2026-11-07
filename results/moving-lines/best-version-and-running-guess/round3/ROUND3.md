# Round 3: decide with the belief (analysis, 3 Nov 2026)

## 1. What step A found (`a_flicker.py`, `a_flicker_out.txt`)

On 129 design cells, offline, frozen CT4-min reproduces the sweep's failure:
- the roster changes between consecutive ticks 0.155 (clean) and 0.168 (lag matched to the sim's, S2m), against
  0.172 in the sweep's closed loop;
- without the static snapshot, the dynamic belief alone changes 0.060 and 0.055, about CT3F's 0.051 in the sweep;
- calibration is overconfident in the middle as in the sweep: people at 0.5-0.7 are queue members 0.27-0.35 of the time.

So offline replay is a fair testbed for roster stability and calibration, though not for time in line. The static
snapshot makes about two thirds of the flicker.

Unit: one service interval (the time the front person takes; the sweep drew 12-20 s). Costs count waiting time, and
the robot's own time counts the same as anyone else's.
- **Cutting in ahead of a person who waits: 1 per person.** Each person cut waits one more service, for certain.
- **Standing behind a non-waiter: 0.25 per non-waiter between the true last and the robot.** The robot loses time
  only if someone takes the gap before the next advance shows the non-waiter doesn't follow (one service interval
  later at most). I take that chance as 0.25: one arrival per four services, a moderate line. The ratio 4 puts the
  robot behind a person who might be last unless P(that person waits) is below 0.2 (the newsvendor quantile
  0.25 / 1.25).
- **Moving: d / (v · s) per move of d metres**, with v = 0.5 m/s (the robot's speed) and s = 16 s (the sweep's mean
  service), so 0.125 per metre. The robot changes its place only when the expected saving beats this. That is a
  Bayes decision with a cost of acting, not a tuned ratchet.
- Offline, I report sensitivity to the ratio (2 and 8) and to the move cost (0 and double). Those runs don't change
  these values. If closed-loop pilots ever change them, every variant tried is logged and counted.

## 3. The estimator (CT5), changed only where the post-hoc points

- **E1. Static evidence reads standing people only.** A walker's current position is not a standing configuration.
  This puts back the original rule. `WALKERS_STATIC` was added in 1b because served people walking away kept a high
  P; E2 deals with them directly.
- **E2. Exit evidence, phrased by the target.** Someone who reached the target (within 1.6 m of the target point)
  and is now walking away from it (0.5 m farther than their nearest point there, walking) is evidence of not
  waiting. One likelihood ratio, counted once per exit.
- **E3. Calibration by counting, not by search.** Each likelihood term is estimated as its own frequency on design
  data with the dump's truth:
  - the answer's reliability, P(named | waits) and P(named | not);
  - the base rate;
  - P(follow | both wait) and P(follow | not both);
  - the exit ratio;
  - the static residual distributions for waiters and non-waiters.
  Fit on clean plus S2m; check reliability on S3 (restless bystanders) and S23m, whose stress the fit never saw.
  Fitted values are reported next to the hand-set ones. The write-up says "calibrated to the simulator".

## 4. The decision

The sets are exact (n ≤ 8, at most 256). For each candidate j, "come after j": every tracked person, ordered by
distance from the target point, plus "first". For each set W:
- last(W) = the member of W farthest from the target point;
- cost(j | W) = 1 × |members of W farther back than j| + 0.25 × |non-members between last(W) and j|.

The robot takes the j with the least expected cost under the posterior, unless its current j costs less than that
plus the move. The solver then gets come_after = W's members up to j (P ≥ 0.5) and j itself, with everyone else
exempt, so its place-in-turn geometry is unchanged.

## 5. Offline measures (proxy; the closed loop decides)

Per tick from delivery, against the dump's truth:
- right (j is the true last);
- cut (j is ahead of the true last: the robot would stand ahead of a waiter);
- far (j is behind the true last);
- empty;
- switches per minute;
- calibration.

Conditions: clean, S2m, S3, S23m. Compared: frozen CT4-min with the 0.5 threshold, CT5 with the threshold, and CT5
with the decision.
## 6. Measurement rule for a blocked place in turn (draft for the run; caution 3)

The sim's `in_line` needs a spot behind the last that is 0.5-1.2 m from them, with heading within 45° and no human
within 0.55 m. When a non-waiter stands there, no such spot may exist, and no arm can be "in line".

**Draft rule, per 1-s sample with someone ahead:**
- **`band_free`.** True if some point on a 0.1 m grid satisfies all of:
  - behind the last along the line;
  - 0.5-1.2 m from the last;
  - within the line corridor (|side| ≤ 0.6 m of the line through the last);
  - at least 0.55 m from every human;
  - on the navigable floor.
- **`acceptable`.**
  - If `band_free`: `acceptable` = `in_line`, unchanged.
  - If not: `acceptable` = the robot stands behind the last along the line, does not cut in, has heading within 45°
    and no human within 0.55 m, and is 0.5-1.2 m behind the nearest person who blocks the band. That is the first
    free spot behind the blocker.
- **Primary measure for the next test:** share of samples `acceptable`, from the first service after delivery to
  the end. Reported beside it: plain `in_line`, the `band_free` share, and cut-in samples.

## 7. Offline results (3 Nov; design data; every variant tried is listed)

Files: `b_calibrate*_out.txt`, `c_eval_out.txt` (V0-V4), `c_eval_window_out.txt` (W0-W4), `c_eval_sensw4_out.txt`
(cost sensitivity), `d_flips_out.txt`, `e_acf_out.txt`, `f_exit_diag.py`.

**Variants tried (11, plus 4 cost-sensitivity runs reported only):**
- V0: CT4 frozen, threshold;
- V1: CT5 hand-set, E1 + E2;
- V2: CT5 calibrated by counting;
- V3: V2 + cost;
- V4: CT4-like + cost;
- W0: CT4-like + window;
- W1: W0 + exit;
- W2: calibrated with walkers + window;
- W3: W2 + cost;
- W4: W0 + cost.

**What the flicker is (step D).** 94% of crossings of 0.5 are driven by the static snapshot. By who crosses:
- walking non-members 39%;
- still non-members 30%;
- members 24%.

**What did not work, and is not adopted:**
- **E1 (static for standing people only)** adds a new on/off flicker as people start and stop walking: roster
  changes rise from 0.148 to 0.185.
- **E2 (exit evidence as defined)** fires on waiting members near the target (25 wrong exits on the fit set). Each
  costs ×1/65, so empty ticks rise from 0.03 to 0.10, and 95% of the extra empties follow a wrong exit. Dropped, not
  patched.
- **E3 (calibration by counting)** improves the calibration error (0.109 to 0.068) but worsens decisions. Cut ticks
  rise from 0.13 to 0.25-0.29, because the fitted static table and P(named | not) 0.073 drop true members who stand
  a little off the predicted spot. Reported as a negative result, and not adopted.
  - Fitted against hand-set: P0 0.352 (0.3); P_NAMED_W 0.858 (0.85); P_NAMED_N 0.073 (0.15); Q 0.931 (0.85);
    B 0.041 (0.05).
  - The static ratio is about 5 below 0.3 m, then falls to about 0.2 beyond 0.45 m. The hand-set curve stays at or
    above 1 up to 0.8 m.

**What worked:**
- **The window (W0).** The static log ratio's measured 1/e decorrelation time is 5.0 s (`e_acf_out.txt`). Reading
  the snapshot as the mean over that window halves roster changes (0.148 to 0.076 clean; 0.231 to 0.101 under S3)
  and last switches (16 to 10 a minute). The cost is a later admission of the newest joiner: right last 0.49 to
  0.39 on clean fit seeds.
  The price is many "far" ticks (0.60-0.76): the robot comes after anyone whose P(waits) is at least 0.2.
- **Cost sensitivity, reported only.** The frozen point sits mid-range: ratio 8 and a doubled move cost score
  slightly lower on their own costs, and ratio 2 and no move cost score higher. Nothing is changed.

**Candidate for the closed-loop design pilots: CT5 = W4.**
- CT4-min's a priori belief, unchanged except the 5 s window on the static snapshot.
- The cost decision with the frozen costs.

Whether "far" (standing behind a possible non-waiter) costs time in line, and whether halving the flicker buys time
in line, are closed-loop questions.
## 8. Port (for the `src/sgi/roles.py` and the CT arm in `serve.py`)

1. **`RoleBelief`: a `STATIC_WIN_S` parameter**, 0 for CT4 and 5.0 for CT5.
   - In `step`, after the static vector is built: append `(t, np.log(static))` to a history, drop entries older than
     `STATIC_WIN_S`, and set `self.static = np.exp(mean of the kept logs, axis 0)`. This is `ct5.CT5.step`.
   - Also store `self.dist_now = dist`: each person's distance from the target point, as `step` computes it.
2. **A decision function**, `ct5.decide(belief, present_ids, current)`.
   - Candidates: every present person tracked at the photo, plus None ("first").
   - For each set W: last(W) is the member farthest from the target point. The cost of coming after j is
     1 × |members of W farther than j| + 0.25 × |non-members between last(W) and j|.
   - Pick the least expected cost under `b × static` (normalised). Keep `current` unless its expected cost exceeds
     the best one's plus 0.125 × |dist(best) − dist(current)|.
   - `current` is per session: unset at the first decision; a fresh choice when that person is no longer tracked.
3. **Solver inputs.**
   - `come_after` = {roster members (P ≥ 0.5) nearer the target than the choice} ∪ {the choice}.
   - `exempt` = everyone else; choice None means `come_after` is empty.
   - Line direction and the CT2 control parts as for CT4.
4. **Log per re-solve:** `choice`, the expected cost of each candidate, `marginals`, `roster`, and the history
   frames as for CT4. This allows a reproduction check against `ct5.py` (bar 0.95 on the choice, as in
   CT4_FROZEN §5).
## 9. Update to section 6 (3 Nov, after the band check, `results/moving-lines/best-version-and-running-guess/round3_check/`)

- **A blocked place almost never happens from the first service on.**
  - Robot idle, `band_free` is false on 1.4% of design frames and 0.4% of spent-sweep "ours" frames.
  - In the closed loop, the band is free in 0.992-0.999 of samples for every arm.
  - The best clearance is at least 0.85 m in 79% of "ours" frames.
  - The fallback credited no sample in any arm.
- **Adopted for the next test:** the primary measure is the strict share in line from the first service after
  delivery (`summarise_fresh.in_line(x, "strict")`), with `band_free` reported as a check. The fallback is not
  used.
- **Wording fixes, recorded:**
  - "the last" in the band rule is the last by distance along the line among those ahead, as the sim's `in_line`
    uses. Right-last scoring uses `truth.last` (queue order). The two differ on 444 of 12,480 design frames, mostly
    in long, bent and step-out lines.
  - "acceptable" would have been `in_line` OR the fallback.
  - The corridor is measured from the scene's line axis, as strict scoring does.
- **Before the first service**, the solver's 1.0 m own-space exclusion blocks the place in turn in nearly every
  "ours" layout. It is the same for every arm. A flexible standoff is a separate, shared solver change, kept for
  later.
- **The no-harm run agrees with the flicker diagnosis on scenes it never saw** (original scenes, seeds 20-29,
  DeepSeek, strict share in line):
  - overall: CT3F 0.400, CT3 0.337, CT4 0.259;
  - from the first service: CT3F 0.501, CT3 0.421, CT4 0.310.
  CT3F is the bar a role method has to clear on time in line.

## 10. Pilot bookkeeping (3 Nov)

- Predictions: `PILOT_PREDICTIONS.md` (cb29fc67) and `g_pilot_predict_out.txt` (97d12fe0), hashed before stage 2.
- Scorer and checker, frozen before stage 2: `h_pilot_score.py` (8bc3e123) and `check_port_ct5.py` (34cc1529).
- The src port matches `ct5.py` offline: 70,876 of 70,876 choices on 129 design cells × 4 conditions.

## 11. Before stage 2 output (3 Nov, ~07:00)

  (recall 0.92-0.93, almost no empty ticks), but it stands in line less. Strict share from the first service:
  CT3F 0.223, CT4 0.121. On overall strict share, CT3F is ahead in 7 of 8 model × flag cells (DeepSeek's "both" is
- **The flicker carries to Sonnet** (`results/moving-lines/best-version-and-running-guess/posthoc/REPLICATE_SONNET.md`, expectation hashed before running). Roster
  changes per re-solve: CT4 0.138, CT3F 0.053. Pose moves per minute: 23.1 against 9.2.
- **Follow-up questions, fixed now** in case the pilot gives the "P1-P3 hold, P4 fails" outcome. They are
  exploratory and reported only:
  - X1: CT5 minus CT3F, strict share from the first service, separately on the 8 restless and the 8 not-restless
    "ours" layouts.
  - X2: for CT4 and CT5, the ticks after the first service where strict in_line is false, split four ways:
    (a) ahead of the true last (cut); (b) behind someone who is not waiting (far); (c) behind a waiting member who
    stands more than 0.6 m off the line axis (a member has stepped out); (d) other (gap, heading, crowding).
  - Why: a member's place in the order is not where they are standing. A belief can be right about who waits and
    still send the robot to the wrong spot if it lines up behind a member who has stepped away.
