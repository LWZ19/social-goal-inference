# Item 9 result: the moving headline on Qwen 3.8 and GLM-5.3 (read by I9_FROZEN.md with amendments 1-3 and 2b)

## Inputs

- **Runs:** `results/data/moving/runs/soclaas_qwen38_low` and `results/data/moving/runs/soclaas_glm53`, 90 of 90 each.
  - GLM needed 3 passes: 13 victim timeouts, all re-run, none left (`timeouts.txt` empty).
  - 0 vlm_error first answers.
- **Qwen's 240 s secondary:** `results/data/moving/runs_i9long/soclaas_qwen38_low`, complete, with 0 cache misses.
- **GLM's 240 s secondary has not run.** The queue was stopped by the 18:56 out-of-memory event; 2 stray episodes
  exist.
- **Scored:** `i9_score_out.txt` (exact-429 sensitivity), `i9_score_conservative_out.txt` (all-retries sensitivity),
  `i9_score_long_out.txt` (secondary).

## 1. The frozen bar at 92 s: both models are latency-bound, so H9 fails for both

| model | strict R1 / CT / CT2 | C episodes with no first answer in time | CT2 vs R1 strict (W/L/T, p) |
|---|---|---|---|
| Qwen 3.8 | 0.03 / 0.01 / 0.01 | 20 of 25 | 2/5/18, p 0.45 |
| GLM-5.3 | 0.00 / 0.00 / 0.00 | 25 of 25 | 0/1/24 |

- **Both models are labelled latency-bound** by the rule fixed in advance (more than half of the CT2 episodes without a
  first answer within the episode). The 92 s comparison measures how late the constraint method's answer arrives, not
  the method.
- **R1 is slow too:** its first answer has a median of 57 s (Qwen) and 74 s (GLM), so R1 also scores near zero.
- **Retry sensitivities (amendments 2 and 2b) change nothing:** GLM is 0/1/22 without the 2 cells with an exact 429,
  and 0/0/5 without the 20 conservative candidates.

## 2. With time for the answer to arrive (240 s, cache-only, labelled secondary)

| model | strict R1 / CT2 | first answer median, C / R | CT2 vs R1 strict | CT2 vs R1 primary |
|---|---|---|---|---|
| Qwen 3.8 | 0.04 / 0.13 | 123 s / 67 s | **19/1/5, p < 0.0001** | 19/5/1, p 0.007 |
| GLM-5.3 | 0.01 / 0.00 | 217 s / 173 s | 0/3/22, p 0.25 | 0/3/22, p 0.25 |

- Once Qwen's answer arrives, asking once and re-solving beats the marked point decisively, the same direction as
  every other model (Sonnet 21/2/2, DeepSeek 19/3/3, Haiku 19/2/4, Gemini 18/6/1 at 92 s).
- Qwen's whole constraint answer takes a median of about 2 minutes. At 92 s most answers had not arrived; the 81 s
  median in the 92 s table counts only those that did.

## 3. Reading for the report

- **On the school's free gateway models, the 92 s moving protocol is latency-bound.** Qwen's and GLM's
  constraint-method answers take about 2 minutes or more, so in a 92 s episode the robot mostly waits. That is a
  latency result, not a method result.
- **Given time, Qwen confirms the headline** (CT2 over R1, 19/1/5).
- **GLM stays latency-bound even at 240 s** (7 Nov): 24 of 25 constraint episodes had no answer in time, so GLM
  cannot be judged on the moving protocol at the gateway's current speed. One request ran 17 minutes before failing.
- **What it pointed to:** the efficient pipeline (CTe, extra X3), which drops the people call. Tested 7 Nov: on Qwen
  both arms stayed latency-bound, and on DeepSeek CTe did worse (`results/overview/RESULTS_7NOV.md`).

## E3b (the descriptive run; complete 7 Nov)
All 7 older models are in `i9_score_e3b_out.txt`. CT2 beats R1 on strict share on 5 of 7: Sonnet 4.6 14/3/8, Sonnet
5 17/3/5, Gemini 3.1 Flash Lite 22/2/1, Gemini 3.5 Flash Lite 20/2/3, Gemini 3.6 Flash 18/5/2. Not shown: Haiku 4.5
(5/2/18; 10 of its 25 constraint answers unusable) and Gemini 3 Flash (13/11/1). See `results/overview/RESULTS_7NOV.md`.
