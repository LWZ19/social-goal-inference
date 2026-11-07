# CT4_FROZEN.md, deviation 1 (3 Nov 2026): the section 5 reproduction check

**What section 5 says.** On at least 6 design cells, each ported arm's per-re-solve last must agree in at least 0.95
of re-solves with this folder's offline code, fed that run's own logged tracks.

- **CT4: 1,694 / 1,698 = 0.998. PASS.** The roster was the same in 1,694 re-solves, and the largest marginal
  difference was 0.028.
- **CT3F: 1,497 / 1,661 = 0.901. FAIL by the letter.** Nine cells agree 100%. Five carry long runs of disagreement
  (DeepSeek cafe s44, sweet s17 and s30; Sonnet cafe s15 and sweet s15).

**The cause (diagnosed before any fresh call).**
- At the first disagreement in each failing cell, the resolver's `joined` flag differs between the offline loop and
  the run. Once the robot counts as joined, `Resolver.inputs` drops the people behind it, so the last changes.
- `joined` depends on the place the solve recorded and on `in_turn`. The offline loop's stand-in for the full solve
  records the place in turn with `in_turn` True, while the live solve records the cell it chose and its own `in_turn`.
  These differ where the place in turn is blocked or off the floor.
- **Check.** `check_port_solverfed.py` keeps the same frozen role logic and the same tracks, but passes the run's own
  logged `pose_world` and `in_turn` to `record`. Output `check_port_solverfed_out.txt`: **1,634 / 1,648 = 0.992**,
  with 13 of 14 cells at 100%.
- The remaining 14 re-solves are DeepSeek sweet s30, the cell whose target point lies on a bystander. The live solve
  derives `target_number` from the target point; the offline stand-in passes None.
- **So the port's role logic matches the frozen design. The shortfall is the offline loop's approximation of the
  full solve, not the port.**
- CT4 is unaffected, because its role logic never records a place: the roster comes from the belief.

**Consequences, stated before the fresh run.**
1. The ported CT3F and CT4 stand as faithful implementations of the frozen designs.
2. The design-data predictions in CT4_FROZEN.md section 3 came from the same offline loop. For CT3F they carry this
   approximation, because `joined` timing differs from closed loop. The closed-loop fresh numbers may therefore depart
   from the predicted values more for CT3F than for CT4. Claim C1, its test and its bars are unchanged.
3. Nothing in CT4_FROZEN.md is edited. This note and both check outputs are added beside it.
