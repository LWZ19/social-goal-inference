# I04 phase 1d: deviations and clarifications

The I04 compute agent wrote this on 18 Oct 2026. It was written after the per-record cache and its checks (`cache/check.txt`: no stop condition), and before the S table, the fold fits and any held-out score. FROZEN.md (sha256 8e0aac98…1882, verified before work started) is not edited. Its hash is in `DEVIATIONS.sha256`.

## Deviations

**None.** Every step of FROZEN.md ran as written. The checks gave:
- reproduction 1,409 of 1,409 records (picks 1–5 at 0.0 m, headings at 0.0 rad), so no record is left out;
- 0 replica disagreements (sample cells, recorded places, density places, offset at 30 and 60 deg);
- the von Mises table check at most 1.4e-8 (bar 1e-6);
- S1 reproduced in all 12 cells.

## Clarifications (mechanics FROZEN.md leaves open; fixed here before the S table)

1. **File names.** `fits.json` is written as the note names it. `folds.json`, which the task names, is a byte-identical copy. Both hashes go into `FITS.sha256`.
2. **Energies in integer millimetres.** E_mm = Σ γ^(5−k) key_k(mm) is computed exactly in int64, and E = E_mm / 1000 m.
   - Every slot key is the solver's mm-rounded value. On every cell of every record, key_mm / 1000 == key holds exactly.
   - So this is the note's E without float noise, and ranks and ties are exact. γ ∈ {1, 3, 10, 30} are integers.
   - V2's kept cells use `_first`'s own code on the solver's float keys, which is the same as equality in mm.
3. **V3's ranking for the density places** uses the integer squared grid distance to c_top. Ties are exact and go to the lowest grid index. The order is the note's "ascending distance". The density weights use the float |c − c_top|².
4. **Validation sample.** The draw is the union of three sets, drawn in order from one generator:
   - 150 uniform F cells;
   - up to 100 in-region cells (replica: matched and inside the polygon);
   - up to 50 cells outside the region whose centre is within 0.15 m (+1e-9) of the nearest in-region cell's centre, as in I14 v0.
5. **Bootstrap mechanics.** One set of 2,000 draws serves every row. From `default_rng(20261018)` the realistic stratum (2000 × 13 layout indices) is drawn first, then the watching stratum (2000 × 6). A realistic cell uses only the realistic draws, a watching cell only the watching draws, and the pooled row uses both.
7. **V0's mass measures.** In-region mass is 1 when the top cell is in the region (replica), and forbidden mass is 1 when it lies in the kind. On the recorded places the replica equals the package (checked). Concentration is 1.
8. **A code fix before any result.** The first cache run stopped at a range assert on an empty array (a wrong-target record has no success cells). The assert was guarded and the cache rebuilt from scratch. No number was read from the aborted run.
9. **Added, labelled post hoc, not part of any verdict:**
   - a by-situation table of the primary records (labels used only to tabulate);
   - pooled Lift_room bootstrap intervals for V1 and V3 alone, on the same draws.
