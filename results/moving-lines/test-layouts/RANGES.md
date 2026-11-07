# Candidate index ranges of the sweep generator (declared 3 Nov 2026, before any of the new ranges is drawn)

`make_sweep.py` seeds candidate k of base b with sha256("sweep-{b}-{k}"), so ranges never share a draw.

| Indices (per base) | Use | Who may look |
|---|---|---|
| 0-449 | the 3 Nov test (CT4_FROZEN.md); scored, now spent | anyone |
| 1000-1199 | round 3 design pilots (ROUND3.md): declared design data before drawing | anyone, the analysis included |
| 2000-2449 | reserved for the next confirmatory test; not drawn until its note is frozen | the run only; never renders, dumps or per-layout results to the analysis before scoring |
| 3000-3449 | item 11 (CT3F vs CT2, then CT3F vs CT3; results/moving-lines/best-version-new-layouts) | run only until scored |

Nothing from 1000-1199 may be reused in a confirmatory test. A pilot result never moves a layout between ranges.
