# E2 addendum 1: people calls shared across a scene's two episodes (written before any E2 score is read, 4 Nov)

## Why this addendum

- C's people call depends only on the image and the trial. So at one trial, the second episode of a scene is served
  from the cache with the first episode's answer and its stored `latency_s`.
- This holds for A0 and for A0b: DeepSeek 73 and 71 such calls, Qwen 61 and 64.
- D's detector people are per image anyway.
- This is the pipeline's own behaviour, as in the development runs.

## What changes, and what doesn't

- **Nothing in the frozen bars.** Success, A, G, S1 and S2 use every people-matter unit, as written.
  - A cached call counts with its stored `latency_s`, the provider's time for the answer that counted.
  - That is the honest cost per request: a new request on a new picture would need the people found again.
- **One sensitivity, reported and not a bar** (`e2_sens.py`). S2 and its placebo are re-computed on the "fresh"
  people-matter units: those where no goal-step call (people or constraints) of A0, A0b or D was served from the
  cache.
  - These units count no latency twice. On all units, two units of a scene and trial share one people latency, and
    the sign test assumes independent units.
- **Also reported:** cached goal-step calls per arm.
