# E1b: detector people in place of the VLM people call, static benchmark (frozen before any call, 2 Nov 2026)

## Question (the project's efficiency track)
Can a local detector replace C's people call, making the pipeline faster and one call cheaper without losing accuracy? The proposal itself names this design. It was never tested live; the D-live check was dropped as underpowered.

## Design
- **Records.** Every new-wording C record whose goal step called people (the people-matter records), on the realistic sets.
  - Qwen 3.8 low s21: 144 records. GLM s23: 141. DeepSeek s23: 151.
  - Records without a people call are identical in both arms and are not run.
- **What is held fixed.** Each record's own recorded plan, evaluated subtask, target point and context check are passed to `pipeline.infer_goal`. A zero-call check confirmed that the injected inputs reproduce the recorded target world point in 60 of 60 records. Only the constraint method runs, at a fresh trial (30 + the recorded trial).
- **Arms:**
  - **A0:** the VLM people call plus the constraints call, i.e. the current C, freshly drawn.
  - **D:** I09 **A1**'s frozen detector people (`results/still-scenes/people-finder/early-detector/phase1c/a1/eval_people.json`, sha256 `ffa59545…`), passed as the people source exactly as the D-live driver does, plus the constraints call.
    - There is no people call.
    - The constraints call sees the detector's numbered people.
    - Fix 2 is not used. Its unseen-layout confirmation never ran, and on these scenes it differs from A1 only by 2 missed staff and 3 false people.
- **Scoring** (`e1_score.py`, sha256 in `E1.sha256`). The harness's `score_pose` on each arm's pose and target, as the recorded runs are scored.
- **Code.** `e1_driver.py`; dry-run with the mock in the main venv.

## Measures
- **Success** of the recorded C, A0 and D on the people-matter records.
- **Redraw floor:** the flips between the recorded C and A0 (same plan, context and target; fresh people and constraints draws).
  - For DeepSeek the floor also includes the version change to 1017, so it is conservative.
- **D against A0:** gain, harm and net, and an exact two-sided sign-flip test over layouts.
- **Latency per record:** A0's people-call and constraints-call latencies, D's constraints latency, plus the detector's 0.06 s (A1 warm, measured on the RTX 4060). Calls per record: A0 2, D 1.

## Judged bar (per model, once, after all calls)
- **D is non-inferior to A0 within the redraw floor** if (harm − gain) ≤ the right→wrong flips between the recorded C and A0.
- **The efficiency claim holds for the free models** if every free model meets the bar. It is then stated with the measured people-call time saved per record.

## Prediction (inference, written before any call)
- **Accuracy:** D is level with or better than A0 for GLM and DeepSeek, whose people errors the A2 replays found (I01's mechanism), and level for Qwen.
- **Risks:**
  - the 2 counter staff the detector misses (constraints naming staff);
  - the 3 false people;
  - a constraints call that writes worse with unfamiliar numbering.
- **Time saved per people-matter record:** about 16 s for Qwen, 2–4 s for GLM and DeepSeek.

## Calls (the run runs them)
- **Free:** Qwen 432, GLM 423, DeepSeek 453; 1,308 in all, at most 13 a minute per process.

## What comes next
- **E1a (zero calls, GPU about 1 min):** the detector on the Habitat sim frames against truth.json (`keep/sim_bench/frozen/*/001`). Habitat humanoids are a new domain. It is announced before the GPU is used.
- **E1c:** a detector-people CT on the moving lines, only if E1b holds.
