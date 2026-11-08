# Holding the place in line (CT3FL)

**Question.** Does the best moving-line version (CT3F) keep its place better if it locks the set of people ahead of
it once it has joined, never steps back down the line, and goes straight to the serving spot when it is first?

**Result.** On 8 layouts of the confirm set, 2 runs each, Claude Sonnet 5.5, both versions in the updated simulator:
time in the right spot (strict) rose from 32% (CT3F) to 55% (CT3FL), better in 15 of 16 runs (sign test p = 0.0005);
cut-in time fell from 170 s to 6 s. The front-of-line rule was tuned on these layouts, so a test on layouts it has
not seen, and on more models, follows.

**Files.**
- `CHANGES.md`: what changed in the method (`hold_place` in `src/sgi/resolve.py` and `src/sgi/serve.py`) and in the
  simulator (`sim/sgi_habitat/robot.py`, `planner.py`, `humans.py`), with the result table.
- `videos/`: CT3F on the left and CT3FL on the right, one video per layout and run (Git LFS).
- Tests: the `hold_place` tests in `tests/test_resolve.py`.

**Reproduce.** Run the arm `CT3FL` (and `CT3F` for comparison) with
`results/moving-lines/design-and-scoring/run_moving.py --module sgi_habitat.experiment_oracle`.

**Data.** The runs are not archived yet.
