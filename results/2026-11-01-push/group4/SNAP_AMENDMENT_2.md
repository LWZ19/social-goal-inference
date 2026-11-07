# SNAP_FROZEN.md, amendment 2 (2026-11-02, before any render or plan call on the new scenes)

**Why.** The selection agent (`snapset/select_v2_out.txt`, `bearing_exact_out.txt`; nothing rendered, no calls) found
that amendment 1's bearing rule and its own bearing list disagree. The error is mine:
- **cg3 55 deg** was listed for 15-40%, but its median share is 0.14.
- **cg3 50 deg** was listed for 40-55%, but its median is 0.36, which is 15-40%.
- **cg4 30 deg** (median 0.14) and **75 deg** (0.01) were listed for 15-40%, while **cg4 15 deg** (0.17) was left out.
- **cg4 35 deg** is exactly 0.396, and check 8 rejects it in 47 of 50 seeds.

The rule as written ("median share in the bin") gives 40-55% only 4 layouts and no cg3.

**The selection is `snapset/selected_v2_literal.csv` (sha256 prefix 56cd80f7): amendment 1's bearing list, as frozen.**
- **The binding criterion is each layout's own cylinder share,** as amendment 1 says ("a layout enters a bin by its
  cylinder share"), together with the generator's own checks and the caps. The median-based sentence described how I
  meant to derive the list. It was wrong, and it is withdrawn. Every selected layout lies in its bin by its own share.
- **Both bins have all three group sizes:**
  - 40-55%: cg2 15 deg ×2, cg3 50 deg ×2, cg4 10 deg ×2;
  - 15-40%: cg2 20 deg ×2, cg3 15 deg ×2, cg4 70 deg and 75 deg ×1 each.
  - At most 2 layouts per (situation, bearing), with spares as listed.
- **Set B:** only 2 layouts qualify (seed 436 cg2 at default, seed 412 cg4 75 deg; the largest share hidden by the
  target is 0.20). So B is made up to 6 with the visitor-place fallback, labelled weak.
  - B's part of H3 carries no weight, as amendment 1 says.
  - The 2 real B layouts are reported apart, with every break listed (click and the boxes containing it). They are
    the only test of the box-centre rule's failure case, and 2 layouts settle nothing.
- Everything else in SNAP_FROZEN.md and amendment 1 stands: the rule, the measures, H1-H3, the pixel-share spares and
  the calls.
