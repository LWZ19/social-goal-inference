
## Method

- **The replay.** Offline replays of the role part of re-solving, on the people dumps (2 Hz from the photo, robot
  idle; truth roles per frame) and, for checking, on the runs' logged tracks.
  - It drives the real `sgi.resolve.Resolver` as `sgi.serve.resolve` does. `sgi.solver._turn` stands in for the
    full solve, whose only effect on roles is whom the turn rule takes as joining and as last.
  - Variants are subclasses in `variants.py`; CT4-min is in `ct4.py`; the perturbations S1-S4 are in `stress.py`.
- **Fidelity (`validate_replay.py`).** On each run's own logged tracks, the replay gives the same solver last as the
  log in:
  - CT3: 97.5% of 8,389 re-solves (DeepSeek) and 99.5% of 7,245 (Sonnet);
  - CT2: 95.5% and 98.0%.
- **The cells.** Every CT3 cell with a recorded constraint answer, seeds 10-44:
  - DeepSeek: 60 bystander cells and 13 original;
  - Sonnet: 28 bystander and 28 original.
- **Measures per 0.5 s tick from delivery:**
  - *right last:* the solver's last is the true last of the line;
  - *empty while a line stands:* no place in turn while someone queues;
  - *capture:* a bystander is the last;
  - *member recall:* the share of in-queue people on the roster;
  - *false drops:* named people dropped by a role rule while in the queue, for the hard rules only. CT4's soft roster
    flickers at 0.5, so its event count isn't comparable; use member recall for it.

## 1. The cheap fixes, on clean tracks (`ladder_clean_out.txt`, `ladder_gated_out.txt`)

Capture on the bystander scenes, in seconds:

| Rung | DeepSeek | Sonnet | Note |
|---|---|---|---|
| CT3 (offline) | 214 | 290 | logged: 212 and 322 |
| + group test (every waiting person behind an advancer is tested) | 185 | 231 | |
| + photo-time start (the wait replayed through the resolver) | 150 | 262 | |
| + group + photo | 13 | 74 | every named bystander dropped at delivery: 16/16 and 15/15 |
| CT3-fix gated (+ the anchor re-derivation; only named people skip the join gate) | 38 | 74 | empty 0.50 → 0.23 (DeepSeek) |

- **Each fix alone is modest; together they're decisive.** The evidence is worth most early, while the answer's
  roster is still intact. That reconciles estimation's Z6 (routing alone, about -7%) with ml's group test (fast drops
  in open loop).
- **The ungated anchor (far_analogies' rule) brings capture back** (79 s), as predicted in Q3. Gating unnamed people
  halves that.
- **Sonnet's whole clean leftover (73 s) is in room.** The first service there is at 26 s, after the answer. Only
  evidence at the photo, or asking, can help there.
- **Hard-rule false drops on clean tracks:** DeepSeek 3-4 (sweet s30 and s35: wrong target points, so advances are
  measured the wrong way), Sonnet 0-1.

## 2. Under stress, the cheap fixes collapse (`ladder_S*_out.txt`)

Capture on the bystander scenes, DeepSeek / Sonnet, in seconds:

| | CT2 | CT3 | + group + photo | CT3-fix gated |
|---|---|---|---|---|
| clean | 890 / 808 | 214 / 290 | 13 / 74 | 38 / 74 |
| S1, 0.10 m noise | 756 / 852 | 241 / 224 | 56 / 62 | 63 / 62 |
| S2, lagging members | 948 / 745 | 219 / 354 | 46 / 104 | 52 / 104 |
| S3, restless bystanders | 644 / 700 | 288 / 313 | 242 / 335 | 571 / 392 |
| S4, stale and switched tracks | 924 / 826 | 268 / 342 | 21 / 100 | 108 / 100 |
| S1234, all combined | 794 / 506 | 234 / 287 | 200 / 321 | 641 / 435 |

- **S3 defeats every hard rule.** One 0.8 m excursion towards the counter gets a bystander past the join gate (the
  0.55 m step-up) and past the anchor's gate.
- **Under the combined stresses, the full bolt-on version is worse than CT3.** Its empty roster also returns: 0.38
  and 0.28.
- **σ 0.20 m white noise (S1b) breaks even CT2's leaving rule** (empty 0.7-0.98). I treat it as extreme; real
  trackers smooth over time.

## 3. CT4-min, the first working draft (`ct4.py`; `ladder_ct4_*_out.txt`, `ct4_variant_out.txt`)

**The design:**
- an exact belief over waiting sets (at most 256);
- the answer fused at the photo;
- hazards gated by motion (joining only after arriving and stopping, leaving only while walking, about zero
  otherwise);
- co-motion: every still person behind a step-up must take up at least half of it, and keep it for 2 s; Q 0.85,
  B 0.05;
- a static leave-one-out snapshot used once at decision time, bounded at a likelihood ratio of 5;
- stale tracks give no evidence.

**Three design errors found on design data, each a modelling point, not a tuned number:**
1. Fitting the line's axis to the named people bends it through a named bystander. Fixed by continuing the chain
   from the anchor.
2. Static evidence accumulated every tick counted one observation about 20 times over. Fixed: a snapshot, used once.
3. A symmetric "still" hazard pulled every low belief back up (+422 log-odds summed over the DeepSeek cells). Fixed:
   nobody starts waiting without moving.

**Results:**

| | Bystander scenes, right last (DS / Son) | capture s (DS / Son) | recall (DS / Son) | empty (DS / Son) | Original scenes, right last (DS / Son) |
|---|---|---|---|---|---|
| clean: CT3-fix gated | 0.37 / 0.34 | 38 / 74 | 0.56 / 0.62 | 0.23 / 0.21 | 0.42 / 0.34 |
| clean: CT4-min | 0.35 / 0.41 | 200 / 197 | 0.80 / 0.88 | 0.03 / 0.02 | 0.46 / 0.51 |
| S1234: CT3-fix gated | 0.16 / 0.17 | 641 / 435 | 0.31 / 0.43 | 0.38 / 0.28 | 0.15 / 0.21 |
| S1234: CT4-min | 0.21 / 0.26 | 234 / 246 | 0.84 / 0.85 | 0.03 / 0.03 | 0.28 / 0.35 |
| S3: CT3-fix gated | 0.28 / 0.25 | 571 / 392 | 0.51 / 0.59 | 0.22 / 0.19 | 0.42 / 0.34 |
| S3: CT4-min | 0.28 / 0.30 | 772 / 636 | 0.81 / 0.89 | 0.03 / 0.01 | 0.46 / 0.51 |
| S4: CT3-fix gated | 0.29 / 0.29 | 108 / 100 | 0.50 / 0.57 | 0.25 / 0.26 | 0.33 / 0.32 |
| S4: CT4-min | 0.19 / 0.20 | 90 / 264 | 0.88 / 0.88 | 0.02 / 0.03 | 0.32 / 0.35 |

**Reading:**
- **CT4-min keeps the line.** Recall is 0.80-0.93 against 0.31-0.67, and the empty roster almost disappears in every
  condition. On right last it ties or beats the bolt-ons clean, and beats them under the combined stresses.
- **It captures more on clean tracks.** It drops named bystanders later (median 8 s against 0.5 s), because it waits
  for the persistence look.
- **Restless bystanders defeat it too.** The static snapshot reads where a person stands at that instant, so during
  an excursion he looks like a member.
- **Under S4 its right last is lower than the bolt-ons'.** Not yet diagnosed.

## 4. Where this leaves the shortlist

- **The kill bar as written:** halve CT3's capture with at most 2 of 244 false drops, under S1-S3. No design meets it
  under S3.
  - On clean tracks the bolt-on halves capture easily, but has 4 false drops; both cases are wrong target points.
  - CT4-min doesn't halve clean capture.
- **What is measured is a trade-off, not a winner.** The bolt-ons win clean capture. The estimator wins line-keeping
  (recall, empty) and every measure under the combined stresses.
- **Restless bystanders are a limit for every motion-evidence method so far.** Telling an excursion from joining
  needs a notion of where a person stands over time, or evidence beyond position (facing, attention).
- **Open choices:**
  1. Continue CT4 on design data: a standing place over time for the snapshot, stale handling, hysteresis on the
     roster. Each is a modelling change, and the fresh layouts guard against over-fitting.
  2. Or freeze both CT3-fix gated and CT4-min as they are, and let the fresh layouts (with their built-in lag and
     excursion flags) decide, pre-registering the trade-off.
  3. The primary measure for phase 2 should be right last (with capture, recall and cut-ins as secondaries), not
     capture alone. Capture alone rewards dropping everyone.
