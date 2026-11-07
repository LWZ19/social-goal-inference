# Item 4: do the model's roles carry the gain, or does tracking a point do as well? (RTw on fresh cells; frozen before any RTw episode on these cells; amendment 1 before any total)

## Question

CT2 re-solves the model's answer ("come after 4, 5, 6") on tracked people. RTw takes R1's marked point and keeps its
offset from the nearest person, in world coordinates, with the answer's heading. That is the strongest tracker that
uses no roles.

The question: does CT2 keep the robot in line better than RTw on the fresh headline cells?

The old evidence is 15 episodes per model on seeds 0-2. It is significant only pooled and only under strict scoring
(the `--test` run of the scorer reproduces it: CT vs RTw 24/10/11, p 0.024 strict; 18/16/11 primary).

## Amendment 1 (6 Nov; before any RTw total was sent or read)

- **The fix.** RTw and CT2 are re-run together in one invocation (`--arms CT2 RTw`, into `moving/runs_i4/<model>/`).
  Both are cache-only, at zero calls. **H4 is judged on these.**
- **Repeat checks.** The first RTw run and the earlier CT2 in `runs/<model>/` are read only as repeat checks (mean d,
  mean |d|).
- **Cost.** 280 episodes in all (RTw and CT2 per cell), zero calls, about 3.7 h at 2 workers.

## Arms and cells (zero model calls: RTw replays R1's cached first answer)

- **RTw** is new on these cells. **CT2** and **R1** already ran and are not re-run.
- **Scenes:** the 5 evaluation scenes (cafe_slots, room_coffee_cart_slots, cafe_line2_moving, cafe_line6_moving,
  sweet_spot_line2_moving).
- **Models and seeds:**
  - Pooled for the bar: Sonnet 5.5 low (seeds 10-19, 50 cells), DeepSeek V4.1 (10-14, 25), Haiku 5.5 (10-14, 25),
    Gemini 3.8 (10-14, 25). That is 125 cells.
  - Reported only: x-test-3 (10-12, 15). Qwen 3.8 and GLM-5.3 (10-14, 25 each) are added once item 9 has cached
    their R1 answers.

- **H4 (primary).** Pooled over the 125 cells, on whole-episode strict share in line (SCORING_STRICT.md, as the
  headline): CT2 wins > losses, and the exact two-sided sign test gives p ≤ 0.05.
  - Ties are exact equality, as in `summarise_fresh.paired`.
  - If H4 holds: "the model's roles carry the gain over tracking a point" is claimed for these scenes.
- **Under the primary scoring:** the same test, reported beside it.
  - If it disagrees, that is stated with the reason: the primary scoring counts time beside the line, where RTw
    stands.
- **Per model:** the same test, reported and labelled. A per-model claim is made only where p ≤ 0.05.
- **Secondary, reported:** RTw vs R1 (does tracking the point help over asking once?).
- **Descriptive:** the RTw episodes whose anchor left the queue, and those with no anchor (the point more than 3 m from
  anyone).

- One RTw episode per cell, exit 0.
- No answer with a `vlm_error`: a cache miss would show as one.
- R1's answer is the cached one, with the same pose.
- The scorer prints missing cells and vlm_error counts. If the pooled pairs are not 125, it flags them.

## Predictions (inference)

- H4 holds under strict scoring, with CT2 ahead by about 0.1-0.2.
- Under the primary scoring the gap is smaller, and may not be significant, as on seeds 0-2.
- The anchor leaves the queue in most RTw episodes: it follows the served person away.

    cd results/2026-11-01-push/moving
    P=.venv/bin/python
    EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
    M=sgi_habitat.experiment_oracle
    PB=8760    # any free base; run_moving refuses a busy port
    # amendment 1: RTw and CT2 together, into runs_i4/<model> (the judged run)
    SGI_CACHE_ONLY=1 $P run_moving.py --model claude_sonnet55_low --out runs_i4/sonnet55_low --workers 2 --port-base $PB --module $M --scenarios $EVALF --seeds 10 11 12 13 14 15 16 17 18 19 --arms CT2 RTw
    SGI_CACHE_ONLY=1 $P run_moving.py --model soclaas_dsv41 --out runs_i4/dsv41 --workers 2 --port-base $PB --module $M --scenarios $EVALF --seeds 10 11 12 13 14 --arms CT2 RTw
    SGI_CACHE_ONLY=1 $P run_moving.py --model claude_haiku55 --out runs_i4/claude_haiku55 --workers 2 --port-base $PB --module $M --scenarios $EVALF --seeds 10 11 12 13 14 --arms CT2 RTw
    SGI_CACHE_ONLY=1 $P run_moving.py --model gemini_fast --out runs_i4/gemini_fast --workers 2 --port-base $PB --module $M --scenarios $EVALF --seeds 10 11 12 13 14 --arms CT2 RTw
    SGI_CACHE_ONLY=1 $P run_moving.py --model soclaas_qwen38_next_low --out runs_i4/x3 --workers 2 --port-base $PB --module $M --scenarios $EVALF --seeds 10 11 12 --arms CT2 RTw
    # after item 9 (reported only):
    SGI_CACHE_ONLY=1 $P run_moving.py --model soclaas_qwen38_low --out runs_i4/soclaas_qwen38_low --workers 2 --port-base $PB --module $M --scenarios $EVALF --seeds 10 11 12 13 14 --arms CT2 RTw
    SGI_CACHE_ONLY=1 $P run_moving.py --model soclaas_glm53 --out runs_i4/soclaas_glm53 --workers 2 --port-base $PB --module $M --scenarios $EVALF --seeds 10 11 12 13 14 --arms CT2 RTw
    # score (zero calls):
    cd .. && $P fill/i4_rtw/i4_score.py > fill/i4_rtw/i4_score_out.txt

## Cost

- **Calls:** zero (cache-only).
- **Simulator:** 140 episodes, about 96 s each, so about 1.9 h at 2 workers or about 1 h at 4.
  - Add 50 episodes after item 9.

## Scorer

`i4_score.py`, using `fill/mcompare.py`, which uses summarise_fresh's `measures` and `sign_p`. Tested on old data with
`--test` (seeds 0-2, CT in place of CT2). It reproduces the earlier figures, e.g. Sonnet RTw 0.155 and CT 0.198
primary; DeepSeek CT vs RTw 8/2/5 strict.
