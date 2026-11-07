# Candidate index ranges of the sweep generator (declared 3 Nov 2026, before any of the new ranges is drawn)

`make_sweep.py` seeds candidate k of base b with sha256("sweep-{b}-{k}"), so ranges never share a draw.

| Indices (per base) | Use | Who may look |
|---|---|---|
| 0-449 | the 3 Nov test (CT4_FROZEN.md); scored, now spent | anyone |

Nothing from 1000-1199 may be reused in a confirmatory test. A pilot result never moves a layout between ranges.
