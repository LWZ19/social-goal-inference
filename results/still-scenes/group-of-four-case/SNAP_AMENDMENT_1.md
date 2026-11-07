# SNAP_FROZEN.md, amendment 1 (2026-11-02, before any render or plan call on the new scenes)

- At the generator's fixed group bearings, every 15-40% candidate is a conversation_group_4 near-copy of the
  development case.
- Nothing reaches 40-70% with the target's chest clear. Scene check 5 rejects the layout when the chest is covered.
- No layout has an object target with a person within 1 m.

The rule, the measures, H1 and H2 are unchanged. This amendment changes only which scenes are drawn and the harm
controls.

## Scenes, as amended

- **Seeds 400-449** (unused; the scan verified this).
- **Bearing override:** the driver sets `recipes._GROUP_BEARING_DEG` in its own process only (deterministic, no
  tracked file changed). `render_subset.py` writes the bearing into its own manifest, because render_job.json does not
  record it, and DEVIATIONS records it.
- **Allowed bearings** come from `bearing_sweep_out.txt`, chosen by a fixed rule: in 5 deg steps, the median hidden
  share falls in the bin, and the chest is covered in at most 10 of 20 sweep layouts. This gives:
  - **15-40%:**
    - cg2: 20 deg;
    - cg3: 15 deg and 55 deg;
    - cg4: 30 deg, 70 deg (default) and 75 deg (default).
  - **40-55%** (the 40-70% bin, renamed for what check 5 lets through):
    - cg2: 15 deg;
    - cg3: 50 deg;
    - cg4: 35 deg and 10 deg.
- **Occluded set: 12 layouts, 6 per bin.** Each bin has 2 layouts each of cg2, cg3 and cg4. At most 2 layouts come
  from one (situation, bearing) pair, and at most 1 from one seed. A layout enters a bin by its cylinder share
  (`occlusion.py`) and must pass the generator's own checks. Selection goes in seed order, and the next qualifying
  seeds are the spares. If a (bin, situation) has too few, its places go to the other situations of that bin, and
  this is recorded.
- **Harm set A, unhidden person targets: 6 layouts.** These are cg2, cg3 and no_one_person at the default bearings,
  with the target under 5% hidden, as selected.csv already has them.
- **Harm set B, target in front: 6 layouts.** This replaces "object near a person". The target is under 5% hidden, and
  the target itself hides at least 15% of another person behind them.
  - This is the case where the snap's box-centre rule can break a right answer: a click on the target that lands
    nearer the hidden person's box centre. It was never present in the development data.
  - B layouts are found from the same truth scan, at any allowed or default bearing, with the same seed-order rule.
  - **Fallback**, only if fewer than 4 B layouts exist: B is made up to 6 with the scan's "place with a visitor within
    0.75-0.85 m" layouts. Those are labelled weak by construction, because the anchor pixel is in no visitor's box in
    0 of 98, and staff stand inside the footprint.

## Pixel share and spares (rule fixed now)

- After rendering, and before any plan call, Blender's pixel-level hidden share is recorded next to the cylinder share.
- A layout whose pixel share falls outside its bin, widened by 0.05 on each side, is replaced by the next spare of the
  same bin and situation:
  - 15-40% becomes 10-45%;
  - 40-55% becomes 35-60%;
  - A and B: target under 10%.
  - B also needs the hidden person 10% or more hidden.
- With no spare left, the layout stays in, labelled, and is reported in both its cylinder bin and its pixel bin.

## Calls

Unchanged: 24 layouts × 2 episodes × 3 trials × 3 free models = 432 plan-only calls. These come from
`snapset/plan_only.yaml` (`run.context_check: always_no`), whose plan prompt is identical to the recorded s21/s23 runs.

## Bars and predictions, as amended

- **H1 and H2:** unchanged. The occluded set is the one defined above.
- **H3 (no harm), replacing the old H3:**
  - At most 1 of the 108 set-A records is broken.
  - At most 3 of the 108 set-B records are broken.
  - Every break is listed with its click and the boxes containing it.
  - If B used the fallback, its share of H3 is reported but carries no weight, and the claim says so.
- **P2:** "fewer in 40-70%" becomes "fewer in 40-55%".
- **P3:** set A has no breaks.
- **P5 (new):** set B has a few breaks, but no more than 3, all of them clicks on the target closer to the hidden
  person's box centre than to the target's own.
- **The claim "the snap beats C's lift on person targets"** still needs H1, H2 and H3.

## Not taken

A reduced replication (default-bearing cg4 at 15-36% only) costs the same 432 calls but would only restate the
development case. Mixed situations at both bins test whether the failure and the fix generalise.
