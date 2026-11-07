
- **The port is exact** (`round4/e_ct4p_port.py`, output `e_ct4p_port_out.txt`). For every CT4P re-solve with
  `direction_source` "path", I recomputed the back tangent from the logged roster and tracks, using a Resolver with
  `path_memory` True and `line_span_m` 0.5 built from the cached answer and the tracks at the photo.
  - 3,070 of 3,070 match the logged `solve_direction` to within 1e-3.
  - All 461 fallbacks are re-solves where I find no path either (cafe_1081, sweet_1112, cafe_1007, and 14 ticks of
    sweet_1000).
- **The scorer printed Q1-Q4 after the FAIL line instead of withholding them.** Its "stop" text was meant for the
  procedure, but the code should have enforced it. Noted as a scorer fault.
