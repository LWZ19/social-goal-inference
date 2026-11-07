# Extra X3: the efficient moving pipeline (CTe) on DeepSeek and Qwen (frozen before any X3 episode)

## Question

CTe takes the detector's people for the first answer, which drops the people call, and tracks with the detector.
Does it answer sooner than CTd (the model's people call, then the same detector tracking) without losing time in line?

So far there are only 15 Sonnet cells. On Qwen, dropping the people call is a large share of its slow first answer
(item 9), so it is the case that matters.

## Arms, models, cells

- **CTd and CTe run together** in one invocation into `results/data/moving/runs_x3/<model>/`, with the detector and tracker
  services up (as item 5), so both share the machine's load.
  - CTd replays the cell's cached CT2 first answer.
  - CTe makes a live constraints call (plus plan and context) on the detector's people.
- **Models:** DeepSeek V4.1 and Qwen 3.8 (free; Qwen after item 9 has cached its answers).
- **Cells:** the 5 evaluation scenes, seeds 10-14, so 25 cells per model.

- **X3-speed (per model):** CTe's first answer arrives earlier than CTd's in more cells than not, exact two-sided sign
  test p ≤ 0.05.
  - The sim pauses for the request-frame detection, so its warm time (about 0.16 s) is added to CTe's when stating
    the gain, as E1c did.
- **X3-quality (pooled over both models):** the one-sided 95% paired-t lower bound of mean(CTe − CTd) on
  whole-episode strict share is above −0.05.
- **Reported:** per-model quality; the primary scoring; the constraint mix (keep-out-of-shared-space share, as E1c
  found).
- **Qwen:** if item 9 labels it latency-bound, its 92 s quality comparison is reported, not pooled. X3-speed still
  applies, as that is the point for Qwen.

## Predictions (inference)

- X3-speed holds on both models: DeepSeek a few seconds earlier, Qwen 20-40 s earlier.
- X3-quality holds; CTe matched CTd within the spread on Sonnet (+0.07, 9/4/2).
- On Qwen, CTe may turn a latency-bound arm into one that acts within the episode.

    cd results/moving-lines/design-and-scoring
    P=.venv/bin/python
    EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
    F=../fill/x3_cte
    # start the detector service (port 8899) and tracker_service.py (port 8897) exactly as item 5, with its health guard
    # stop both services by PID; then score (zero calls):
    cd .. && $P results/moving-lines/people-finder/x3_score.py > results/moving-lines/people-finder/x3_score_out.txt

## Cost

- **Calls:** CTe's first answer is about 3 calls (plan, context, constraints), about 75 per model; CTd makes none. All
  free.
- **Simulator:** 100 episodes at about 96 s, so about 1.3 h at 2 workers. The detector uses about 1.3 GB of GPU.

## Scorer

`x3_score.py`, using `results/_lib/mcompare.py`. Tested with `--test` on the earlier Sonnet cells, it reproduces E1c: CTe
0.316 and CTd 0.247, 9/4/2; first answer 10.6 and 13.0 s.
