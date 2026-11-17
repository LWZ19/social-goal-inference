# CT2m and Rm: move while thinking (frozen 2026-11-02 ~01:45, before it is built and before any fresh-seed result)
is known at 23-27% of the constraint method's latency (after the plan call). Critiqued by the analysis.

## Method (latency hiding; names no situation)

As soon as the plan call has answered (the pipeline's own first call, which the full answer reuses from the cache, so
no extra model call), the robot gets an **interim goal** on the straight ray from where it stands to the plan's target
point, facing the target:
- 2.5 m short of the target point, or 1.5 m before the first tracked person within 0.6 m of that ray (the nearest to
  the robot along it), whichever is farther from the target;
- no interim goal when that point is behind the robot or within 0.3 m of it, or when the plan gives no target point.
The full answer replaces the interim goal as usual when it arrives (CT2's replace rule for CT2m; the base rule for
Rm). Timing: the interim goal is delivered at the request time plus the plan stage's reported time (the stored latency
for a cached call), the full answer at the request time plus all its stages' time, as for every arm.

## Arms (fresh seeds 10-19, sharing each cell's cached answers)

- **CT2m** = CT2 + move while thinking. Clean latency comparison: **CT2m against CT2**.
- **Rm** = R1 + move while thinking (parity: R's plan call yields the target point as early). **Rm against R1**.
- Headline across: CT2m against Rm.

## Predictions (written before any run)

- Time from the request to first in line falls, for both, by roughly the gap between the plan's and the full answer's
  delivery less what the approach would have cost anyway; the largest gain for the slowest model (x-test-3).
- Keep-up unchanged against the arm without it (it acts only before the full answer).
- Cut-in samples during the approach (before the full answer) about 0, because of the 1.5 m stop before people on the ray.
- Share in line (all three scorings) higher than without it, mostly through the earlier start.
