
## Inputs

- **Runs:** `runs/e2_claude_sonnet55_low`, `runs/e2_claude_haiku55` and `runs/e2_gemini_fast`.
  - 160 of 160 units each, 0 vlm_error.
  - The same design, scenes, people file and code: HEAD d50a7b3, empty diff (`E2_DEVIATIONS.md` items 46-48).
- **Scoring:** the frozen `e2_score.py` and `e2_sens.py` over all six folders, giving `e2_score_six_out.txt` and
  `e2_sens_six_out.txt`.
  - The three free models' numbers are identical to `e2_score_all_out.txt`.
  - The claim and the pooled H2 use the free three only, and are unchanged (`E2_RESULTS.md`).

## All six models (success over people-matter units)

| model | A0 (current C) | A0b (C redrawn) | D (detector people) | gain / harm | floor rw | goal-step median A0 → D | whole unit A0 → D | D faster / slower | placebo p |
|---|---|---|---|---|---|---|---|---|---|
| Qwen 3.8 (free) | 144/159 | 146 | 154 | 11 / 1 | 6 | 70.0 → 45.0 s | 107 → 83 s | 133 / 26 | 0.87 |
| DeepSeek V4.1 (free) | 126/157 | 131 | 132 | 16 / 10 | 9 | 34.5 → 18.8 s | 66 → 50 s | 148 / 9 | 0.75 |
| GLM-5.3 (free) | 100/143 | 93 | 125 | 26 / 1 | 23 | 122 → 89 s | 142 → 109 s | 107 / 36 | 1.00 |
| Sonnet 5.5 low | 155/160 | 157 | 155 | 4 / 4 | 2 | 7.4 → 4.8 s | 12.9 → 10.2 s | 156 / 3 | 0.20 |
| Haiku 5.5 | 143/158 | 137 | 147 | 9 / 5 | 12 | 5.9 → 3.8 s | 9.9 → 7.7 s | 157 / 1 | 0.58 |
| Gemini 3.8 Flash low | 129/158 | 129 | 132 | 3 / 0 | 2 | 9.9 → 4.9 s | 19.1 → 14.3 s | 157 / 1 | **0.005** |

- **Gemini's timing placebo is not null:** A0b is faster than A0 in 97 units and slower in 61, p 0.005.
  - By the frozen rule that makes its timing comparison "not judgeable".
  - On the fresh units (addendum 1) its placebo is p 0.054, and D is faster in 78 of 78.
  - D's lead (157 of 158) is far larger than the drift the placebo shows. But as written, Gemini's speed result is
    the call saving only.

## Reading (description only)

Across all six models the detector people source never costs accuracy beyond the redraw floor, and always saves
one call.
- On the three free models it is also significantly more accurate when pooled (`E2_RESULTS.md`).
- The time saved is largest where the people call is slow: the free gateway models.
