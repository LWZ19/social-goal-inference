# Item 12, interim: the still-scene headline on fresh layouts, five of six models (GLM pending its re-run passes)

## Inputs

- **Runs:** `runs/*_s27_i12_*` for Qwen 3.8, DeepSeek V4.1, Haiku 5.5, Gemini 3.8 and Sonnet 5.5. That is 70 fresh
  layouts (seeds 510-514, the 14 development rows), 280 records per method per model.
- **Scored:** `i12_score.py` (amendment 1 included; there are no vlm_error records in these five) into
  `i12_score_five_out.txt`.
- **GLM** is excluded until its C re-run passes end (amendment 1). So the pooled bars, which need all six models and
  the three free ones, are judged then. The "pooled" lines in the output are the five models only, and are not yet
  the bar.

## Per-model bars: I12-R and I12-E hold for all five

| model | R / E / C (of 280) | C−R: W/L/T layouts, net, p | C−E | E−R (reported) |
|---|---|---|---|---|
| Qwen 3.8 | 191 / 227 / 246 | 29/2/39, +55, p < 0.0001 | 20/6/44, +19, p 0.021 | **+36, p 0.0006** |
| DeepSeek V4.1 | 141 / 141 / 228 | 38/0/32, +87, p < 0.0001 | 36/1/33, +87 | 0, p 1.0 |
| Haiku 5.5 | 151 / 156 / 260 | 35/0/35, +109, p < 0.0001 | 38/2/30, +104 | +5 |
| Gemini 3.8 | 167 / 174 / 237 | 32/2/36, +70, p < 0.0001 | 28/1/41, +63 | +7 |
| Sonnet 5.5 | 198 / 194 / 257 | 26/2/42, +59, p < 0.0001 | 27/2/41, +63 | −4 |

- **C beats both R and E on every model,** on layouts no prompt was chosen on. Across five models C loses on 6
  layouts in all, and only for Qwen against E.
- **The margins are not smaller than on the development layouts,** so the in-sample objection does not apply. C−R as
  a share of records:

  | model | development | fresh |
  |---|---|---|
  | Qwen | 19% | 20% |
  | DeepSeek | 26% | 31% |
  | Haiku | 35% | 39% |
  | Gemini | 20% | 25% |
  | Sonnet | 25% | 21% |

- **Where the gain is (success by situation):** waiting lines above all.
  - Pointing models mostly fail them: R gets Haiku 4 of 100, DeepSeek 19, Sonnet 33. C gets 71-92.
  - Conversation groups gain moderately.
  - No-one scenes are equal across methods, since people don't matter there.
- **Worked examples (E − R) help Qwen significantly** (+36, p 0.0006), as under the old wording, and no other model.
  On the new-wording development runs no model's E − R was significant.

## Still to come

GLM's per-model bars, and the pooled bars over the six and over the three free models, once GLM's vlm_error re-runs
end (amendment 1, with the worst-case bound).
