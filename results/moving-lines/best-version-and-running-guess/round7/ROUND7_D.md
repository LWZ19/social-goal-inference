
## Background

- **CT3FT** is CT3F with only the target point moved to the true counter point, as in CTt and CT4PJT. It runs with
  zero calls on DeepSeek's cached CT2 answers and the 72 confirm layouts, into `runs_ceiling4/dsv41`. It is
  description only.

## Gaps

Same measure and pairing as `c_gaps.py`:
- strict share in line from the first service;
- paired per-layout means over layouts where both arms have a window;
- front and behind parts;
- wins, losses and ties (|d| < 0.01), the exact sign test, and the one-sided 95% paired-t lower bound;
- band top 1.2 m (the decision) and 1.5 m (reported beside it).

The gaps:
- **G1, target for the best arm:** CT3FT − CT3F (CT3F from `runs_confirm4/dsv41`, same cached answers).
- **G2, who once the target is right:** CT4OT − CT3FT. This is the room left for who-waits work over CT3F's rules
  when both have the true counter point.
- **Reported, no rule:**
  - CT3FT − CT4PJT (CT3F's rules against the role belief, both with the true target);
  - each arm's front part and its strict share at the front;
  - CT3FT as a share of OW.

## One identity

On the same 66 layouts, G2 = (CT4OT − CT3F) − G1 = 0.121 − G1 at 1.2 m. The 0.121 is from `d_post_out.txt`, and
the script recomputes it. So the outcome depends on G1 alone, cut at 0.05 and at 0.071 (0.121 − 0.05). The bar 0.05
is the smallest effect with useful power at 72 layouts (`CONFIRM_CT4PJ.md` §3), as in ROUND7.

## Outcomes, fixed now (read at 1.2 m)

| outcome | G1 | G2 | lead 1: where served customers stood (target point and front place) | lead 2: walking away / who waits |
|---|---|---|---|---|
| A | above 0.071 | below 0.05 | **first.** The counter point alone lifts the existing best arm to within 0.05 of the true-everything arm | **closed** (future work). With a right target, CT3F's rules leave under 0.05 for who-waits work |
| B | 0.05 to 0.071 | 0.05 to 0.071 | **first** | **second, only after lead 1.** Who-waits work pays only once the target is right |

Further rules:
- **Uncertainty.** If G1's lower bound is below 0, outcome A or B is reported as "point estimate only". With 66
  layouts, a mean of 0.05 has a lower bound near 0.02, so this flags only a weak result.
- **Where the gain sits.** If G1's front part is more than half of G1, the gain is at the front. That supports the
  serving-spot form of lead 1 over a better counter guess alone. The result is reported and decides nothing.

## Script

`e_ct3ft.py` prints all of the above mechanically from `runs_ceiling4/dsv41` and `runs_confirm4/dsv41`. It is
hashed with this note. It was checked on pilot logs with stand-in arms before the totals.
