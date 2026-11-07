# Person-snap test: deviations and recipe (SNAP_FROZEN.md 8251bf9e, SNAP_AMENDMENT_1.md c6f8ad17, _2.md 5cecbc28)

1. **2026-11-02 ~18:30, the scene recipe, logged before any render or plan call.**
   - Generator seeds 400-449 (placeholder generator `sgi.synth`, first-pass `accepted_layouts`; no synth seed above
     219 was used anywhere before; 3-99 sealed, 100-219 spent). These 50 seeds are spent by this test.
   - Selection: `snapset/selected_v2_literal.csv` (sha256 56cd80f7..., `select_v2.py --bearings literal`), as
     SNAP_AMENDMENT_2 rules: occluded 15-40 and 40-55 bins with cg2/cg3/cg4 each, set A (unhidden person targets),
     set B (2 real "target in front" layouts, 436 and 412, reported apart; 4 weak visitor-place fallbacks, no weight).
   - Group bearings are patched in the driver's process (`sgi.synth.recipes._GROUP_BEARING_DEG`) per (seed,
     configuration), recorded in `manifest_v2.jsonl` (render_job.json does not hold it); no tracked file changed.
