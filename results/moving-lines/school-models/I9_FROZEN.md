# Item 9: the moving headline on Qwen 3.8 and GLM-5.3 (frozen before any of their moving episodes; amendments 1-3 and 2b before any score)

## Question

Does asking once and then re-solving on tracked people (CT2) keep the robot in line better than the marked point asked
once (R1), with the school's free panel models?

The headline already covers Sonnet, DeepSeek, Haiku 5.5, Gemini 3.8 and x-test-3, and E3b is adding the older models.
Qwen 3.8 and GLM-5.3 are missing.

## Protocol (unchanged from the other models, so the table stays one table)

- **Scenes:** the 5 evaluation scenes. **Seeds** 10-14 (25 cells). Episodes of 92 s.
- **Arms:** C1 on seeds 10-12 only, and R1, CT and CT2 on 10-14, as `results/still-scenes/hidden-situations-and-more-models/sim_headline.sh`.
  - C1, CT and CT2 share one first answer per cell through the cache; R1 has its own.
- **Why seeds 10-14 and not new seeds:** C1, R1, CT and CT2 were fixed before seeds 10-14 were first used, and their
  code is unchanged. Nothing was ever tuned on these models.
- **Models:** `soclaas_qwen38_low` and `soclaas_glm53` (free), with the current C prompts and settings.

## Bars (judged once per model, after its episodes and the mechanics check)
- **H9 (per model).** CT2 vs R1, whole-episode strict share in line, ties exact as in `summarise_fresh.paired`. CT2 wins
  > losses, and the exact two-sided sign test gives p ≤ 0.05.
- **Reported:** the same under the primary scoring; CT vs R1; C1; and the first answer's latency.
- **Latency label (fixed now).** E2 measured whole C answers at a median 107 s for Qwen and 142 s for GLM (`e2/`,
  still scenes), against a 92 s episode.
  - If more than half of a model's CT2 episodes have no first answer within the episode, its 92 s result is reported
    as "latency-bound": the comparison then measures latency, not the method. The scorer prints this.
- **Outages.** A first answer that ended in `vlm_error` (gateway outage, no model answer) is not the model's answer.
  - A cell still failing after them is left out of both arms of the comparison and counted.
  - `parse_failed` is the model's own answer and stays.

## Amendment 1: sim runs that time out (6 Nov, before any item 9 total; the question)
- **What happened.** Some GLM episodes never finished: run_moving's subprocess limit (488 s) ended them while the sim
  was still waiting for the first answer, so no episode.json exists.
- **Cached latency.** A cache hit replays the stored answer's own latency (experiment.py receive()), so a re-run cell
  gets no speed advantage.
## Amendment 2: rate-limit retries on GLM's answers (6 Nov, before any item 9 total; the disclosure)

- **What happened.** From 15:58, DeepSeek and Qwen still-scene calls (items 12 and 13) ran alongside GLM's live moving
  runs. GLM's serve logs show 10 HTTP 429s (16:02-16:26; none before 15:58), each retried after a short wait. The
  answers in flight were a few seconds late, against GLM's roughly 140 s.
- **The rule, fixed now:**
  - **Primary H9 and the latency label are unchanged:** every cell counts.
  - **Sensitivity (reported, no bar):** H9 and the label also without the cells whose CT2 or R1 first answer had
    `transient_retries > 0`.

## Amendment 2b: which retry list (6 Nov, after the runs' mechanics, before any score was read)

## Amendment 3: why the timeouts happen, and the rulings it changes (6 Nov, before any item 9 total)

## Secondary: the same cells with 240 s episodes (cache-only, zero calls; reported, labelled)

- **What:** R1 and CT2, re-run with `--seconds 240` into `results/data/moving/runs_i9long/<model>`. Their first answers come from
  the 92 s run's cache, because the request frame at 2 s is the same.
- **Why:** it separates the method from the model's latency. The answer, once it arrives, is used for 150 s or more.
- **Reading:** the same test as H9, labelled "secondary: with time for a slow model's answer to arrive". It does not
  replace H9.

## Predictions (inference)

- **Qwen:** most first answers arrive in about 60-110 s, so many CT2 episodes are latency-bound at 92 s. H9 may fail
  for latency, while the 240 s secondary favours CT2.
- **GLM:** slower, with more outages. Likely latency-bound at 92 s, with a secondary in CT2's favour if its answers
  are usable.
- **R1** (plan + marking, two calls) arrives sooner, so the 92 s comparison is tilted towards R1 for these models.
## Run (run; from `moving/`; live free calls; 2 workers per model, one model at a time)

    cd results/moving-lines/design-and-scoring
    P=.venv/bin/python
    EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
    M=sgi_habitat.experiment_oracle
    F=results/moving-lines/school-models
    PB=8800
    for m in soclaas_qwen38_low soclaas_glm53; do
      $P run_moving.py --model $m --out runs/$m --workers 2 --port-base $PB --module $M --scenarios $EVALF --seeds 10 11 12 --arms C1 R1 CT CT2 > $F/run_${m}_a.log 2>&1; echo "$(date +%H:%M) $m 10-12 exit $?"
      $P run_moving.py --model $m --out runs/$m --workers 2 --port-base $PB --module $M --scenarios $EVALF --seeds 13 14 --arms R1 CT CT2 > $F/run_${m}_b.log 2>&1; echo "$(date +%H:%M) $m 13-14 exit $?"
    done
    # outage passes: delete only the arm folders whose first answer has status vlm_error, then re-run the block (<= 3 passes)
    # secondary, after the bar's runs (cache-only):
    for m in soclaas_qwen38_low soclaas_glm53; do
      SGI_CACHE_ONLY=1 $P run_moving.py --model $m --out runs_i9long/$m --workers 2 --port-base $PB --module $M --scenarios $EVALF --seeds 10 11 12 13 14 --arms R1 CT2 --seconds 240 > $F/run_${m}_long.log 2>&1
    done
    cd .. && $P results/moving-lines/school-models/i9_score.py > results/moving-lines/school-models/i9_score_out.txt && $P results/moving-lines/school-models/i9_score.py --long > results/moving-lines/school-models/i9_score_long_out.txt

## Cost

- **Calls:** about 7 per cell (CT2's first answer: plan, context, people, constraints; R1: plan, context, marking).
  That is about 175 free calls per model, plus outage re-runs. The secondary makes zero calls.
- **Simulator:** 90 episodes per model at about 96 s, so about 1.2 h at 2 workers per model. The secondary is 50
  episodes per model at about 245 s, so about 1.7 h at 2 workers.
- **Rate:** a few calls a minute, inside the shared 85.

## Scorer

- `i9_score.py`, using `results/_lib/mcompare.py`. It prints every model's same 25 cells (E3b's models as they finish), the
  first-answer latency, and the episodes with no answer in time.
- Tested on the existing runs, it reproduces the E3 table: CT2 vs R1 strict Sonnet 21/2/2, DeepSeek 19/3/3, Haiku
  19/2/4, Gemini 18/6/1.
