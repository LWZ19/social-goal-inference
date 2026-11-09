# What changed (hold-place branch)

## 1. New arm `CT3FL` = CT3F + hold place (`src/sgi/resolve.py`, option `hold_place`)

- **Lock the people ahead on joining.** This covers the waiting people ahead of the robot, plus anyone standing within 0.6 m of the line ahead. After that, the set only shrinks: no joiners, no re-admits, no anchor re-derive. The usual drops (served, off the line, not following) still apply.
- **Forward only.** Once locked, a new goal further back down the line is ignored (`src/sgi/serve.py`).
- **Front of line.** When the set is empty, the robot goes at once to the still pose nearest the target where the waiting people stood (the serving spot), facing up the line. The model's target point is no longer used for this.

## 2. Simulator, used by both arms

- **Robot (`robot.py`, `planner.py`):**
  - squeezes through a narrower gap when no comfortable path exists, and backs off after 3 s stuck;
  - takes no long detours round people (more than 1.5× the straight distance, or 1 m longer);
  - steps directly to goals under 1 m away, instead of turning round.
- **People (`humans.py`):**
  - rejoin a line only on reaching its end, so they no longer hold a place from across the room;
  - never walk through the robot; they wait for it, then step round.

## Result

Claude Sonnet 5.5, 8 confirm4 layouts, 2 runs each, both arms in the same updated simulator:

| | CT3F | CT3FL |
|---|---|---|
| Strict | 32% | **55%** (15/16 runs better, sign test p = 0.0005) |
| Primary | 44% | **61%** |
| Cut-in | 170 s | **6 s** |

Not comparable with `results/2026-11-01-push`: the simulator changes affect both arms. The front-of-line rule was tuned on these same layouts.

`videos/` has CT3F on the left and CT3FL on the right, one video per layout and run. Tests are in `tests/test_resolve.py` (the `hold_place` tests).
