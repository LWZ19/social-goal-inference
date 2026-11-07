# SNAP_FROZEN.md, amendment 3 (2026-11-02, after rendering, before any plan call or detector run on the new scenes)

**Why.** The 24 scenes are rendered (`snapset/run_scenes_v2/`, `assembled_v2.csv`). Blender's pixel hidden shares are
far below the cylinder shares, and only 4 of 12 "occluded" scenes stay in their pixel bins. But hidden share is not
the development failure's mechanism.
- **The failure needs an occluder edge at the target's chest point,** where models click. In conversation_group_4,
  a nearer surface lies **1.0 px** from the chest point (0.04 body half-widths), and the clicks scatter about it.
- **Measured from the render depth by `anchor_gap.py`** (truth only, zero calls):
  - the unhidden development and control targets have no nearer surface within 0.69 half-widths (or none at all);
  - the new occluded scenes split into 8 at 0.04-0.41, s407 at 0.54, and 3 at 0.83-1.19, as far as the controls.
- Pixel share misses this. s406 has an edge 1.2 px from the chest but only 0.28 of the person hidden.

I chose the threshold below after seeing this geometry. No model output and no detector output exist for these
scenes.

- **Primary occluded set (H1, H2) = occluded scenes whose nearer-surface gap at the chest point is at most 0.5 body
  half-widths** (`anchor_gap.py`: depth more than 0.3 m nearer; half-width = fx × 0.22 m / depth). That gives
  **8 scenes**:
  - s403 c3b50, s404 c4b10, s405 c2b20, s406 c3b50, s410 c4b75, s411 c4b70, s412 c2b15 and s415 c4b10;
  - that is cg2 ×2, cg3 ×2 and cg4 ×4.
- **The other 4 occluded scenes** (s401, s407, s408, s409) are still called, and reported as a far-occluder set.
- **No more renders.** The 24 scenes and 432 calls stand.
- **H1 (amended):** baseline target_correct on the 8 primary scenes is at least 10 points below set A (unhidden),
  pooled over the three models.
- **H2 (amended):** on the 8, fixed − broken > 0, with an exact two-sided sign-flip test over the 8 layouts at p ≤ 0.05.
  The smallest attainable p is 2/256.
- **H3:** unchanged for set A (at most 1 of 108 broken). Set B is the 6 weak waiting-line fallbacks (both real B
  layouts failed the pixel rule), so it carries no weight. Its breaks are listed.
- **The pixel-share and cylinder-share bins become descriptive.** P2's bin predictions are replaced by a per-scene
  report: chest-point gap, pixel share, detector recall of the target, and fixed/broken.
- **The claim** still needs H1, H2 and H3. With 8 layouts, a pass is a first replication on new layouts, not a
  general result.
