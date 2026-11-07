
- `o_confirm_score.py` f536382f, run once per model;
- `sweep/confirm_score_dsv41_out.txt` bf2f51eb;
- `sweep/confirm_score_sonnet55_low_out.txt` 74d6b929.

Mechanical checks passed for both models before scoring:
- DeepSeek: port check 0.9999 and 0.9938.
- Sonnet: port check 9914/9914 and 0.9913.

## 1. The verdict, as frozen

- **DeepSeek V4.1 Flash (primary): H1 fails.** 27/16/23, p 0.126, mean d +0.052, n 66.
  - Section 8: "the pilot's commitment effect did not replicate. H2 is reported only."
  - H2 would have failed anyway: mean d −0.049, lower bound −0.080.
- **Sonnet 5.5 low (second model, reported, not claimed): H1 passes, H2 fails.**
  - H1: 33/17/21, p 0.033, mean d +0.050, n 71.
  - H2: mean d −0.072, lower bound −0.105.
  - Section 8: "the mechanism is confirmed, but parity is not".
- **Nothing from this test is claimed.** The primary model decides, and the second model was declared as reported only.

## 2. What the numbers show (description, not claims)

Strict share in line from the first service, mean over the same layouts for every arm:

| | CT4 | CT4P | CT4PJ | CT3F |
|---|---|---|---|---|
| pilot, DeepSeek (n 24, design data) | - | 0.060 | 0.182 | 0.188 |
| confirm, DeepSeek (n 66) | 0.119 | 0.107 | 0.159 | 0.208 |
| confirm, Sonnet (n 71) | 0.131 | 0.124 | 0.174 | 0.246 |

1. **The commitment rule's direction held on both models, but its size halved.**
   - Mean d was +0.122 in the pilot, +0.052 (DeepSeek) and +0.050 (Sonnet) here.
   - Among untied layouts, the win share fell from 0.78 to 0.63 and 0.66.
   - Ties rose from 25% to 35% and 30%.
   - The design assumed the pilot's rate, so H1 had less power than planned (section 3 put it at about 0.99).
2. **The pilot's parity with CT3F did not hold.**
   - CT3F leads by 0.049 and 0.072. The sign test favours CT3F on both models: 17/34/15, p 0.024; 18/44/9, p 0.001.
   - The pilot never showed parity at δ 0.05. Its lower bound was −0.054 at n 24, so it showed only a mean near zero with a wide interval.
3. **Why the pilot overstated.**
   - CT4PJ was designed on the pilot layouts, after diagnosing CT4P's failures there.
   - CT4P's pilot score of 0.060 was a low point; here it is 0.107 and 0.124.
   - CT3F's held or rose (0.188 to 0.208 and 0.246).
   - The pattern fits a fix chosen on the same data it was first measured on: a winner's curse. The fresh draw was there to catch exactly this.
4. **The share of the CT4P-to-CT3F gap the rule closes:** 0.95 in the pilot, 0.51 (DeepSeek), 0.41 (Sonnet).
5. **Direction rule:** CT4P is no better than CT4 (0.107 against 0.119; 0.124 against 0.131). This repeats round 3's null.
6. **The scope limit is not the main story.** CT4PJ's own join test fired in 48 of 72 and 52 of 72 episodes.
7. **Whom it follows, against holding the place.**
   - DeepSeek: CT4PJ picks the right last more often than CT3F (0.470 against 0.450), yet has the lower share in line.
   - Both models: CT4PJ is joined less after the first service (0.474 against 0.516; 0.469 against 0.605) and has more stays ended by leaving (26 against 17; 43 against 34).
   - This suggests, unclaimed, that CT3F's lead is more about holding its place than about whom it follows.
8. **Cut-ins.** CT3F's lead comes with more cut-in samples on both models: 4.46 against 3.72, and 9.04 against 4.01. Time in line was the primary measure, so this only qualifies "CT3F is better": it holds more line time and also cuts in more.

## 3. What changes

- **Withdrawn:** "CT4PJ ties CT3F" (`CT4PJ.md` pilot reading, `n_ct4pj_release_out.txt`). The best estimate of the commitment rule's effect drops from +0.122 to about +0.05.
- **Stands, as description only:** the observer's own place in the order is the largest single part of the role belief's deficit found so far, and it is positive on both models. It is significant only on the second model, which was not the test.
- **Section 8's next design is already spent.** The process-structured belief was built offline in round 5 (`round5/ROUND5.md`, `RESULTS_1.md`) and failed its own written bar:
  - v1 failed false fires;
  - v2 to v4 lost recall or raised empty;
  - on the design cells, v4 only traded far for cut.

  Nothing goes forward from section 8, and no closed-loop run is proposed.

## 4. Next: exploratory follow-ups on the scored runs (zero calls; nothing claimed)

`q_confirm_explore.py`. Its questions were written before any per-layout confirm data was seen:
- **E1:** where the CT4PJ − CT3F gap sits, split by whether CT4PJ's join test ever fired.
- **E2:** holding the place:
  - the sim's leaves per minute joined, and goal jumps per minute;
  - whether the layouts where CT4PJ trails CT3F are the ones where it leaves faster or jumps more.
- **E3:** the gap by stress flags (lag × restless). Do the hand-built rules' gains shrink when people stop following the sim's script?
- **E4:** whether each layout's d agrees across the two models, which separates layout effects from model effects.

The dry run on the pilot (design data) only tests the code. It reproduces 14/4/6 and 11/8/5. It also gives a lead, not a result:
- goal jumps per minute: CT3F 7.2, CT4PJ 12.5, CT4P 17.6;
- rank correlation of d(strict) with d(jumps), CT4PJ − CT3F: −0.40.

If the confirm runs show the same, the open question is how a belief should become a goal that doesn't jitter. That is about the decision layer, not the belief.

Any arm built from E1-E4 needs:
- a fresh candidate range declared in `sweep/RANGES.md` before drawing (for example 3000-3449);
- a frozen note;
- the same mechanical checks.
