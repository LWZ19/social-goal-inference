# CT4P read (analysis, 3 Nov, after the go; scorer output 855b679c)

- **Q1 NOT held** (3/1/1, 4 needed); **Q2 held** (−0.027); **Q3 NOT held** (0.42 against the 0.55 bar; CT4 0.40,
  CT3F 0.64); **Q4 held** (CT3F ahead 14/3/7, p 0.013).
- Means over layouts, strict share from the first service: CT3F 0.188, CT4 0.068, CT4P 0.060.
- **The reading that applies** is CT4P.md's "Q1 fails": the direction is not what separates the high-error layouts.
  They differ in other ways. The direction check failed on its premise (`CT4P_CHECK.md`), so the direction rule
  repaired the angle on only about 4 of the 12 high-error layouts.

## A correction to FOLLOWUPS.md and the messages that followed

- FOLLOWUPS.md said part of the role arms' "where" deficit comes from their direction estimate. That came from
  ticks and layouts grouped by chord error (`k_geometry/geom_out.txt`, `tiltlay_out.txt`). It was post hoc and
  confounded: the high-error layouts are hard for every arm (CT3F 0.144 against 0.232).
- Equalising the direction rule in the closed loop moved CT4 by −0.008. So **the direction rule is not a material
  part of the gap** in its CT3F form, on these layouts. What equalising it did change is small: Q3 0.40 → 0.42.
  Offline replay predicted the same (`results/moving-lines/best-version-and-running-guess/round4/RESULTS_B.md`: 0.50 → 0.52 pooled on right-last ticks).
- The withdrawn part is "direction". "C1 and round 3 compared packages" still holds as a description. The open
  candidate for a package part that matters is the joined rule (`results/moving-lines/best-version-and-running-guess/round4/RESULTS_C.md`), which CT4-PJ tests. With
  CT4P ≈ CT4, CT4-PJ's comparison with CT4P is in effect a comparison with CT4.
