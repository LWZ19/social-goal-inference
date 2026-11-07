# Item 10 result: asking once and re-solving (CT2) against re-asking (Rq0, Cq), read by I10_FROZEN.md with amendments 1-3

## Inputs

- **Runs:** `moving/runs_i10/{dsv41, sonnet55_low}`, 75 episodes each (CT2, Rq0 and Cq x 25 cells).
  - 0 failed first or later answers, and 0 HTTP 429s.
- **DeepSeek's amendment 3 run condition met:** the rolling median of its own calls peaked at 16.4 s (run 1) and 4.2 s
  (run 2), against 36 s (`gateway_check_dsv41_run{1,2}.txt`).
- **Qwen:** incomplete (9 and 8 pairs) after a slow-gateway stop; reported only.
- **Scored:** `i10_score_out.txt`.

## 1. The frozen reading

- **Gateway check.** Amendment 3's ratio guard flags DeepSeek as "gateway-confounded".
  - Its re-asks ran at 0.81 of its replayed first answers' latency, for both Rq0 and Cq. That is above 1.5 x its
    normal ratio (0.49 and 0.51, from the 1 Nov seeds 0-2 runs).
  - So by the rule, DeepSeek is reported only and left out of the pooled bar.
  - Qwen is latency-bound (item 9's rule: 6 of 9 CT2 episodes without a first answer in time), so it is left out
    too.
- **The pooled bar is therefore Sonnet's 25 cells. Both bars hold, by a wide margin:**
  - **H10-Rq0:** CT2 − Rq0 strict mean +0.305, lower bound +0.225 (needs > −0.05). Superior too: 22/2/1, p < 0.0001.
  - **H10-Cq:** CT2 − Cq +0.228, lower bound +0.156. Superior: 21/4/0, p 0.0009.
  - **The claim holds:** "asking once and re-solving keeps the robot in line at least as well as re-asking, at a
    fraction of the calls." On these cells it does better.

## 2. Per model (reported)

| model | CT2 − Rq0 (lower bound; W/L/T) | CT2 − Cq (lower bound; W/L/T) | calls per episode CT2 / Rq0 / Cq |
|---|---|---|---|
| Sonnet 5.5 | +0.305 (+0.225; 22/2/1) | +0.228 (+0.156; 21/4/0) | 4.0 / 29.0 / 22.4 |
| DeepSeek V4.1 (flagged) | +0.174 (+0.067; 19/5/1) | +0.220 (+0.129; 16/5/4) | 3.8 / 15.2 / 9.2 |
| Qwen 3.8 (latency-bound, partial) | +0.005 (9 pairs) | +0.015 (8 pairs) | 1.3 / 2.3 / 1.5 |

- **DeepSeek, though left out of the bar, agrees strongly.** Its lower bounds are well above zero.
- **The guard is in the anti-conservative direction for CT2,** so it cannot have produced DeepSeek's direction. It
  can only have enlarged its size.
- **Qwen at 92 s barely acts in any arm** (about 1-2 calls an episode), as item 9 found.
- **Repeat check:** CT2 re-run against its earlier run, mean |d| 0.027 (DeepSeek) and 0.042 (Sonnet).

## 3. A note on the guard (post hoc observation; the rule is not changed)

- **The two speed checks disagree for DeepSeek.** The direct check, the gateway's per-call medians during the lane,
  was far under the threshold. The ratio guard fired.
- **The guard's "normal" ratio comes from the 1 Nov seeds 0-2 runs,** where DeepSeek's re-asks were unusually fast
  relative to its first answers (5.8 s against 11.8 s). Today's re-asks (10.1 s) are close to today's replayed first
  answers (12.4 s), the 0.8-0.9 seen for the other models.
- So the flag probably reflects an unrepresentative anchor, not a slow gateway. It is kept as frozen, and DeepSeek
  stays reported only.

## 4. Reading for the report

- **Re-asking does not buy stability.** On Sonnet, asking once and re-solving beats re-asking every 10 s (Cq) and
  re-asking as fast as answers come (Rq0) by 0.23-0.31 of time in line, with 4 calls an episode against 22-29.
- **DeepSeek shows the same,** by 0.17-0.22 with 4 calls against 9-15. It is reported beside the bar because of the
  guard.
- **This replaces the old 15-episode, primary-scoring evidence,** under which strict rescoring had Rq0 about level
  with CT v1.
