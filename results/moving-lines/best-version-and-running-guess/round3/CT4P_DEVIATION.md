# CT4P.md deviation (run's port, accepted by the analysis, 3 Nov, before any CT4P output)

CT4P.md said to build the role arm's Resolver with `path_memory` True and `line_span_m` 0.5. That Resolver's
`inputs()` still drops people from the answer's other constraints, and its drop rule depends on `line_direction`
and the path (`_off_line`). So CT4P would have differed from CT4 in more than the direction.

The port instead keeps CT4's own Resolver, and builds a second one with {`path_memory` True, `line_span_m` 0.5} from
the same people, constraints, target point and answer-time tracks, used for geometry only:
- `Resolver.back_tangent(points)` = `_on_path(max(points, key=_along))[2]`;
- points = the tracked roster members (P(waits) ≥ 0.5, tracked now);
- fallback to `RoleBelief.dir`.

`_path` is set only at construction (`resolve.py` lines 178-182), as in CT3F, so the geometry Resolver needs no
updates. This matches CT4P.md's intent ("only change: the solver's `line_direction`") more exactly than its wording
did. The predictions, groups and scorer are unchanged.
