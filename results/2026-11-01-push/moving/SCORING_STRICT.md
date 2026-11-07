# SCORING_STRICT: a third scoring for the fresh seeds (frozen 2026-11-02 ~01:45, before any fresh-seed result)

Origin: post hoc, from the failure analysis of seeds 0-2 (analysis/FAILURES.md): the frozen in-line test has no
side check (38% of R1's in-line seconds on DeepSeek are beside the line), and the first-in-line tolerance (1.6 m)
counts a robot standing about 1 m short of the serving spot, never served. Frozen here before any result on fresh
seeds 10-19 is looked at; reported beside SCORING.md (primary, as frozen) and CT2_FROZEN.md's 0.6 m-first scoring.

A sample is **in line (strict)** when it is in line under SCORING.md and, in addition:
- with someone ahead of the robot: the robot is within **0.6 m** of the queue's slot polyline (`side` <= 0.6 m; the
  0.6 m of DESIGN.md's corridor; the sim's own membership corridor in code is 0.8 m, logged in DEVIATIONS 6);
- with nobody ahead (first): within **0.6 m** of slot 0 (`to_slot0_m` <= 0.6), facing and clearance as in SCORING.md.

Everything else (sampling, episode measures, comparisons) is as in SCORING.md and the post hoc measures of Q9, Q10,
Q12. Computed from the logged samples; no re-run.
