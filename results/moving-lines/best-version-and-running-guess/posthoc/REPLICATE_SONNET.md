# Post-hoc 1, Q5 on Sonnet episodes: written before running (3 Nov)

Expected if the mechanism is general and not a DeepSeek quirk:
- roster changed per re-solve pair: CT4 at least 2x CT3F (DeepSeek: 0.172 vs 0.051);
- pose moves per minute (median over episodes): CT4 above CT3F (DeepSeek: 22.3 vs 11.1).
If either fails, the flicker account of CT4's lost time in line does not carry to Sonnet.

## Result (`posthoc1_q5_sonnet_out.txt`, run after the text above was hashed: 336ddddf)

| Arm | roster changed / re-solve pair | pose moves / min (median) | goal moved > 0.3 m / 1-s sample |
|---|---|---|---|
| CT3F | 0.053 | 9.2 | 0.168 |
| CT4 | 0.138 | 23.1 | 0.318 |

- Roster changes: CT4 / CT3F = 2.6, so "at least 2x" held.
- Pose moves per minute: CT4 above CT3F, so it held.
The flicker carries to a second first-answer source. It is still a post-hoc account: these numbers explain
CT4's lost time in line on two models, but they do not show that removing the flicker restores it. The round 3
pilot (P4) tests that.
