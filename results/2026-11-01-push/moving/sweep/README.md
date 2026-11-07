# Sweep layouts: the fresh test set for ideas round 2 (3 Nov, zero calls)

Why: the bystander scenes' seeds are not independent tests. The service clock is fixed (the first service comes at
16 s in every cafe and sweet_spot seed), and the photo barely changes between seeds; Sonnet names the lone man in
0.78 of near-identical frames. A fresh test needs new layouts (`ideas2/research_session_challenge.md` Q2;
`ideas2/SHORTLIST.md` section 5).

## Steps (all zero calls)

1. **`make_sweep.py`** (sim Python) draws candidates from two bases, cafe_line2_bystanders and
   sweet_spot_line2_bystanders. 450 per base, indices 0-449.
   - Drawn per candidate:
     - line length 2-4;
     - a lone non-member 0-0.9 m off the axis and 0.4-2.2 m past the last member, facing the counter, sideways or
       away (facing is stratified);
     - a talking pair whose centre is 1.0-2.5 m from the line's end and outside its corridor;
     - service time [12, 20] s;
     - avatars, randomised, with walkers filling up to the 8 dressed avatars.
   - Kept only if every condition holds:
     - everyone stands on navigable floor;
     - everyone is at least 0.6 m apart;
     - the robot's and returning people's slots are navigable;
     - the pair is at least 1.8 m from the robot;
     - every member, the non-member and the pair are each at least 40% visible in the first picture;
     - a place in line exists, at least 0.6 m from everyone.
   - Slots from n + 1 that lie near or behind the non-member jog 0.9 m sideways.
   - Room is left out: DeepSeek answered "people do not matter" in 6/6 room bystander cells. cafe_line6 shares
     cafe_line2's stage, axis and robot start.
   - The first picture of `sweet_001` in the first draw had the pair filling the view, which led to the 1.8 m rule.
     That draw was discarded and redrawn whole.
2. **`sweep_report.py`** checks, for each candidate, which model-free rules take the lone non-member at the photo
   frame:
   - chain: 1.0 m chain from the person nearest the target;
   - absorb: the solver's `_turn`;
   - static: the leave-one-out check, at most 0.3 m off the axis and 0.3-1.6 m behind.

   Strata:

   | Stratum | Meaning | Count (cafe + sweet) |
   |---|---|---|
   | ours | absorb and static take him, chain doesn't | 96 (50 + 46) |
   | chain | the chain takes him or a pair member | 167 |
   | mixed | any other rule takes him | 371 |
   | easy | no rule takes him | 266 |

   **No layout lets all three rules take him while a place in line exists.** Within 1.0 m and on the axis, no point
   0.5-1.2 m behind the last member is 0.6 m clear. So the "ours" stratum is the hardest one that exists: both of
   the pipeline's own geometric rules are fooled, and only motion or the model can decide.
3. **`select_sweep.py`** makes the frozen selection: ours 40, chain 12, mixed 8, easy 12, half from each base.
   - Within each stratum and base, layouts are spread over the distance past the last member (equal bins), evening
     out facing and line length.
   - Stress flags are crossed 2 × 2 and balanced within each stratum:
     - lag: step-up delay U(0, 4) s, with 20% lazy;
     - restless: one excursion towards the counter at U(10, 60) s, of U(0.75, 1.15) m.
   - Output: `sim/scenarios/sweep/*.yaml` (72 files; sha256 of the concatenated files, in selection order, `047f49a5`)
     and `selection.json`. One seed (0) per layout. The unit of analysis is the layout.
   - Output summary: `select_out.txt`.

- At the photo, the only place in line is 0.5-0.75 m behind the last member. The solver's 1.0 m standoff is blocked
  by the non-member in every "ours" layout (`standoff_valid` 0 of 40).
- A man on the axis 1.2-1.6 m behind the last member reads as waiting to a person too, so squeezing in front of him
  would look like a cut-in to an observer.
- **Primary measures on "ours" cells:**
  - time from the answer's delivery to a right roster;
  - share in line after the first service that follows delivery.
- Not share from the start: it would reward rushing in or luck.

## Simulator options used (off by default; existing scenes replay byte-identically)

| Option | Behaviour |
|---|---|
| queue `service_s: [min, max]` | Each service takes a drawn time; the clock runs only while the front person stands at the front. |
| queue `step_lag: [min, max]`, `lazy_share: p` | Members wait a drawn time before stepping up. Lazy members also wait for two free places, unless first. Nobody steps past the place of the person ahead. |
| standing person `excursion: {at, dist, hold, toward}` | One walk part way towards a point, stopping 0.5 m short of anyone; hold; go back. |
