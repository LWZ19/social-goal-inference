
Frozen after phase 1b, before any fresh call. Design data only: seeds 10-44 of the bystander and original scenes,
offline, zero calls. I have not opened `moving/sweep/human_dump/`, `moving/sweep/renders/`, or any fresh output. The
1b record is `INTERIM.md` and the `*_out.txt` files in this folder.

## 1. Frozen code (this folder; sha256)

| File | sha256 | Holds |
|---|---|---|
| `ct4.py` | a8176bf930a3fcc6f80252e288d74c010d720e6b9d7447d61af80da26041012e | CT4-min: `class CT4`, `DEFAULTS`, `replay_ct4` (the reference decision loop) |
| `variants.py` | b06ee3e7c0b1df31714c5d4bd75de4fb1dea416158faf2e46aa626476c194af2 | CT3-fix-gated: `class CT3FixGated(AnchoredGated, GroupFollow, Resolver)` |
| `replay_lib.py` | 1fd5b49ca40f8f39e14b8ea8f691d78bd950149231379d63b88d39fd51157074 | `replay` (the reference loop for Resolver arms, including the photo-time start), `solve_roles`, `score` |
| `stress.py` | 5e8f4b899d1ea9ae5d66b506019887add885b25afa28427c93d29fb2706cf93c | S1-S4 (offline only; not used in the fresh test) |

## 2. The two arms

**CT3-fix-gated (the best bolt-on of the same evidence)**
- `sgi.resolve.Resolver` with CT3's options: `readmit` True, `line_span_m` 0.5, `vacate_after_s` 5.0, `path_memory`
  True, `co_motion_s` 5.0, `co_motion_follow` True.
- **The group test (`GroupFollow._follow_up`).** At each advance, every waiting person behind the advancer gets a
  follow check, not only the next one.
- **The anchor (`Anchored.inputs`, gated).** When nobody named still waits, the waiting people are re-derived from the
  target point:
  - base: the still person nearest it, within 0.5 m of the front gap the named people showed at the answer;
  - step: still people behind a member, within 1.5 × their mean spacing;
  - never a follow-check non-follower;
  - anyone the answer didn't name must pass CT3's `_moved` gate. "Still" means within 0.10 m over 1 s.
- **The photo-time start.** Before the first live solve, every buffered 0.5 s tick from the photo to delivery goes
  through `inputs`, the turn rule and `record`, with the robot at its start (`replay_lib.replay(start="photo")`).

**CT4-min (one estimator of who waits; `ct4.py` docstring)**
- **State.** An exact belief over the set of people waiting at the target point, among those tracked at the photo
  (2^n sets, n ≤ 8).
- **Prior.** The answer is fused at the photo: P(named | waits) 0.85, P(named | not) 0.15. The base rate is 0.3.
- **Hazards, gated by motion.** Joining only after arriving (1 m in 20 s) and stopping: 0.05/s. Leaving only while
  walking (over 0.3 m/s): 0.05/s. Otherwise 0.0001/s.
- **Co-motion.** A step up is: still, then 0.5-1.5 m nearer the target point, then still. Every still person behind
  the stepper and within 3 m must, if both wait, come nearer by at least half the step (and at least 0.25 m) within
  5 s, and still be there 2 s later. Q 0.85, B 0.05. The step itself counts ×3 towards the stepper waiting.
- **Static snapshot.** Applied to everyone, walkers included, at their current position:
  - one spacing (0.4-1.4 m) behind the waiting person just ahead, continuing the chain from the person or target point
    ahead of them;
  - or, if first, 0.3-1.6 m from the target point;
  - Gaussian 0.35 m, neutral at 0.8 m, likelihood ratio bounded to [1/5, 5];
  - multiplied in once, at decision time; never accumulated.
- **Decision.** Roster = P(waits) ≥ 0.5. The solver gets `come_after` = the roster, `exempt` = everyone else, and
  `line_direction` = `CT4.dir`.
- **Parameters.** Every one is in `DEFAULTS`, set a priori or as modelling choices found on design data (INTERIM §3,
  and the round-2 ablation in `ct4_ablate_out.txt`). None was fitted to the sweep.

## 3. What design data predicts (offline replay; all ticks from delivery; `frozen_*_out.txt`, `percell_out.txt`)

Right last share, bystander cells, DeepSeek / Sonnet:

| Condition | CT3 | CT3-fix-gated | CT4-min | Per cell, CT4-min vs CT3-fix-gated, W/L/T |
|---|---|---|---|---|
| clean | 0.18 / 0.21 | 0.37 / 0.34 | 0.40 / 0.44 | 27/30/3 and 15/12/1 (a tie) |
| lagging members (S2) | 0.11 / 0.17 | 0.25 / 0.24 | 0.37 / 0.38 | 45/15/0 and 22/6/0 (untied 0.75, 0.79) |
| restless bystanders (S3) | 0.15 / 0.17 | 0.28 / 0.25 | 0.30 / 0.30 | 30/27/3 and 15/13/0 (a tie) |
| both (S2 + S3) | 0.11 / 0.18 | 0.20 / 0.21 | 0.29 / 0.28 | 43/17/0 and 16/9/3 (0.72, 0.64) |

- **CT4-min keeps the line.** Recall is 0.74-0.88 against 0.39-0.62; "empty while a line stands" is 0.01-0.05 against
  0.19-0.31.
- **CT3-fix-gated captures less in every condition.** Clean: 38 / 74 s against 210 / 234 s. S3: 571 / 392 s against
  908 / 724 s.
- **Original scenes:** CT4-min wins 10-12 of 13 cells (DeepSeek) and 28 of 28 (Sonnet) in every condition.
- **Restless bystanders defeat both** (capture rises several-fold). Neither design solves them.

Data: the 72 frozen sweep layouts (`moving/sweep/selection.json`, scenario sha256 047f49a5), one seed each.
- Ties are |d| < 0.01. Tests are exact two-sided sign tests over layouts.

**Primary measure, right last.** At each re-solve from the first delivery to the episode's end, scored at ticks where
someone queues ahead of the robot:
- the solver's last (track id) equals the true last;
- the true last is the member farthest back, in the sim's queue order, among those ahead of the robot when the robot
  has joined; otherwise the queue's last member;
- the denominator excludes ticks with nobody queuing ahead, so values will run slightly above the offline table,
  whose denominator was all ticks.

**C1 (the claim).** On the 36 lag-flagged layouts (lag only, plus lag and excursion), DeepSeek, CT4-min beats
CT3-fix-gated on right last: wins > losses and p ≤ 0.05.
- Design-data untied win rate 0.72-0.79. Power for 36 untied layouts is 0.71-0.90; 25 of 36 wins are needed.

**Reported with stated predictions (not claimed):**
- excursion-only layouts (18): a tie (untied win rate in 0.35-0.65);
- clean layouts (18): a tie;
- the "ours" stratum's lag-flagged layouts (20): CT4-min ahead (power about 0.5-0.7);
- capture: CT3-fix-gated lower in every flag cell;
- recall and empty: CT4-min better in every flag cell;
- cut-ins and strict share in line: no prediction. This is the open question of which roster failure costs more;
- the Sonnet subset;
- no-harm on fresh original-scene seeds 20-29.

**Why lag-flagged layouts and not all flagged ones.**
- Design data predicts a tie under excursion alone.
- All 54 flagged layouts would carry an expected untied win rate of about 0.66, power about 0.63; the 30 flagged
  "ours" layouts about 0.40.
- The 2×2 stays balanced, so the interaction is estimated honestly, and the clean and excursion-only cells are
  reported against their predictions.

**Readings fixed in advance:**
- *C1 passes:* "a joint role estimator seeded by the VLM's answer is more robust to lagging line members than the best
  rules built on the same evidence. It keeps the line (recall, never empty) at the cost of slower bystander drops.
  Neither is robust to restless bystanders."
- *C1 fails:* no robustness claim. M1 reduces to a unifying account that ties the bolt-ons, and the trade-off is
  reported as measured.

- **The closed-loop reproduction check.** On at least 6 design bystander cells (3 DeepSeek, 3 Sonnet), cache-only,
  each ported arm's logged per-re-solve last must agree in at least 0.95 of re-solves with this folder's offline code
  fed that run's own logged tracks. CT3's offline replay agreed 0.975-0.995 with its logs.
- **The CT4-min port needs four inputs:**
  - the tracks at the photo;
  - 2 Hz tracks from the photo (the `history` plumbing);
  - the answer's people and constraints and the target point;
  - the robot's start (the direction fallback).
- **The CT4-min port replaces the Resolver's role logic**, but not CT2's control parts (ratchet, span, path memory).
  Path memory: CT4 passes its own `dir`. The ratchet and replace rule stay as in CT2.
