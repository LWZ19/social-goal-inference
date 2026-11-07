
Inputs:
- The rule and script were hashed before the totals: `ROUND7_D.md` 919218e9, `e_ct3ft.py` 19798987.
- Outputs:
  - `e_ct3ft_out.txt`;
  - `a_loss_ct3ft_out.txt` and `a_loss_ct3ft.csv`;
  - `f_front.py` and `f_front_out.txt`, written after reading the above, post hoc description.

Nothing here picks a method or changes the frozen result or the test.

| | 1.2 m (the decision) | 1.5 m |
|---|---|---|
| G1, CT3FT − CT3F (the counter point for the best arm) | +0.053, lower bound +0.013; 27/23/16, sign p 0.67 | +0.048, lower bound +0.008; 31/20/15, p 0.16 |
| G2, CT4OT − CT3FT (who, once the target is right) | +0.068, lower bound +0.018; 36/17/13, p 0.013 | +0.082, lower bound +0.033; 36/22/8, p 0.087 |

- **G1 sits on the 0.05 bar** at both tops, so the letter is decided by the bar's edge, not by the data.
  - Its t lower bound is above 0, but the sign test is not close. The true counter point helps on 27 layouts and
    hurts on 23.
  - So the average gain comes from fewer, larger wins, not from a consistent lift.
- **G1 splits about half and half** between the front (+0.026) and behind (+0.027 at 1.2 m).
- **The identity holds:** CT4OT − CT3F = +0.121, so G1 + G2 = 0.121.

## 2. What the two outcomes have in common

- For CT3F, the counter point and who is ahead contribute about equally to the gap to the true-everything arm: +0.053
  and +0.068.
- Neither alone is a clear, consistent win. Together they give +0.121 (39/20/7, p 0.018).
- Under B the order would be: target first, then who. Under C both are one design question. Either way, a who-waits
  method alone (walking away) cannot improve the best arm much while the counter point is the model's guess.

## 3. The front, where the counter point acts (post hoc, `f_front_out.txt`, `a_loss_ct3ft_out.txt`)

| at re-solves with nobody ahead | CT3F | CT3FT |
|---|---|---|
| knows nobody is ahead | 0.50 | 0.63 |
| goal to slot 0, median | 1.45 m | 0.71 m |
| goal within the strict 0.6 m | 0.21 | 0.35 |
| strict at the front (seconds) | 0.12 | 0.25 |
| OW | | 0.78 |

- **The true counter point roughly doubles CT3F's front score,** but the front goal still sits at a median 0.71 m
  from slot 0, just outside 0.6 m. CT4OT's was 0.69 m. That is the solver's front place, 1.0 m from the counter point
  and off the axis, not information.
- **With someone ahead and the right person named,** CT3FT's goal is right 0.75 of the time (CT3F 0.70), and the
  robot is there 0.60 of that, like every arm.

| arm | strict | front part | share from the front |
|---|---|---|---|
| OW (ideal place) | 0.640 | 0.333 | 52% |
| CT4OT (true roster + true counter) | 0.330 | 0.083 | 25% |
| CT3FT (CT3F + true counter) | 0.261 | 0.044 | 17% |
| CT4O (true roster) | 0.218 | 0.013 | 6% |
| CT3F | 0.208 | 0.018 | 9% |
| CT4PJT (role belief + true counter) | 0.207 | 0.017 | 8% |
| CT4PJ | 0.159 | 0.008 | 5% |

The front is half of the ideal robot's score, and a twentieth to a quarter of the methods'. The methods reach the
front less (12-20% of their window, against OW's 43%) and pass there far less often.

The fixed rule sends this to you because B and C split on the bar's edge. My reading:
- The counter point and the front place (lead 1, where served customers stood) are the next thing to try.
- Who-waits work comes after, measured on top of a right counter point.

That is B's order. It is also consistent with C, if C's single design question is answered with lead 1 first.
