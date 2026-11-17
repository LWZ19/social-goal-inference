# Holding the place in line (CT3FL)

**Question.** Does the best moving-line version (CT3F) keep its place better if it locks the set of people ahead of
it once it has joined, never steps back down the line, and goes straight to the serving spot when it is first?

**Result.** On 8 layouts of the confirm set, 2 runs each, Claude Sonnet 5.5, both versions in the updated simulator:
time in the right spot (strict) rose from 32% (CT3F) to 55% (CT3FL), better in 15 of 16 runs (sign test p = 0.0005);
cut-in time fell from 170 s to 6 s.

**It holds on unseen layouts and other models.** On the other 64 layouts of the 72-layout final test, which the
front-of-line rule was not tuned on, one run each, both versions in the updated simulator:

| Model | CT3F | CT3FL | better / worse / same | cut-in time |
|---|---|---|---|---|
| Claude Sonnet 5.5 | 25% | 47% | 49 / 8 / 7 (p < 0.0001) | 492 → 138 s |
| Gemini 3.8 Flash | 21% | 37% | 42 / 11 / 11 (p < 0.0001) | 698 → 412 s |
| DeepSeek V4.1 | 21% | 31% | 35 / 12 / 17 (p 0.001) | 324 → 209 s |
| Claude Haiku 5.5 | 20% | 28% | 32 / 11 / 21 (p 0.002) | 245 → 249 s |

The simulator fixes alone raise CT3F by 3-5 points on these layouts (significant for Sonnet only), so most of the gain
comes from holding the place. Runs where the model's single answer found no target (17 for Haiku, 5 DeepSeek, 1 Gemini)
are the same in both versions.

**Files.**
- `CHANGES.md`: what changed in the method (`hold_place` in `src/sgi/resolve.py` and `src/sgi/serve.py`) and in the
  simulator (`sim/sgi_habitat/robot.py`, `planner.py`, `humans.py`), with the result table.
- `videos/`: CT3F on the left and CT3FL on the right, one video per layout and run (Git LFS).
- Tests: the `hold_place` tests in `tests/test_resolve.py`.
- `hold_extended_score.py`: the unseen-layout and four-model check; `hold_extended_score_out.txt`: its output.

**Reproduce.** Run the arm `CT3FL` (and `CT3F` for comparison) with
`results/moving-lines/design-and-scoring/run_moving.py --module sgi_habitat.experiment_oracle`.

**Data.** The four-model runs are in `results/data/moving/runs_hold/` (CT3F and CT3FL, 72 layouts each).
