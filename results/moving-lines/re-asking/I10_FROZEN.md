# Item 10: asking once and re-solving against re-asking (CT2 vs Rq0 and Cq; fresh cells; frozen before any re-asking episode on these cells; amendments 1 and 2 before any episode; amendment 3 as disclosed in it)

## Question (the design's H1, `results/moving-lines/design-and-scoring/DESIGN.md`)

Is asking once and then re-solving on tracked people (CT2, about 4 calls an episode) as good at keeping the robot in
line as asking again and again?
- **Rq0:** the marked point, asked again as soon as an answer arrives.
- **Cq:** the constraint method, asked again every 10 s.

The re-asking arms make about 6-37 calls an episode.

**Old evidence:** 15 episodes per model on seeds 0-2, against CT v1, under the primary scoring. Under strict rescoring
Rq0 was about level with CT. The `--test` run of the scorer on those episodes gives lower bounds of −0.07 to −0.12 at
n 15: too few cells to decide.
## Amendment 1 (6 Nov, before any item 10 episode; the point)

- **The issue.** An answer lands in the sim at its send time plus its wall latency, and that latency includes
  sgi.serve's local work (`experiment.py` 271-285). The machine is heavily loaded today (load 18-28 on 20 cores). New
  Rq0 and Cq episodes would get slower local turnaround than the CT2 episodes recorded earlier on a quieter machine.
  That tilts H10 towards CT2: the anti-conservative direction for a non-inferiority claim about CT2.
- **The fix.** CT2 is re-run in the same invocation as Rq0 and Cq (`--arms CT2 Rq0 Cq`, into `results/data/moving/runs_i10/<model>/`).
  It replays its cached first answer, and its re-solves are local, so it costs zero calls.
- **H10 is judged on this concurrent CT2.** The earlier CT2 in `runs/<model>/` is read only for a repeat check (mean d
  and mean |d|), as in item 5.

## Amendment 2 (6 Nov, before any item 10 episode): gateway contention

- **The issue.** Rq0 and Cq make live calls during the episode, while CT2 replays its cached first answer with the
  latency stored when it was made.
  - If the school gateway is also serving other DeepSeek or Qwen jobs while item 10 runs, only the re-asking arms'
    calls slow down (and may hit 429 retries).
  - That tilts H10 towards CT2, the anti-conservative direction for its non-inferiority claim.
  - Machine load is handled by amendment 1. This is the gateway.
- **Run condition.** Item 10's DeepSeek lane and its Qwen lane each run while no other job calls the same model on the
  gateway (pause or finish them first, as was done for GLM in item 9). Sonnet's lane uses the Anthropic API and is
  unaffected.

## Amendment 3 (7 Nov about 01:40): an external gateway slowdown

- **The issue.** From about 00:10 the school gateway slowed about 3x for every model, with our own load unchanged
  (DeepSeek's median call went from 29 s to 70 s).
  - In item 10 only the re-asking arms' later answers are live. All first answers, and all of CT2, are cache replays
    with latencies stored at normal speed.
  - A slow gateway therefore handicaps only Rq0 and Cq: the anti-conservative direction for H10, as in amendment 2.
- **Check in the scorer, applied whatever the run conditions.**
  - For each model and re-asking arm: the median latency of its live re-asks over the median latency of its first
    answers (cache replays).
  - Under normal speed, on the old seeds 0-2 runs, re-asks are faster than first answers: DeepSeek 0.49 and 0.51,
    Sonnet 0.79 (Cq), x-test-3 0.91 and 0.93 (`i10_score_test_out.txt`). So the check is relative to each model's
    normal ratio, not absolute. Qwen 3.8 has no old re-asking runs, so x-test-3, a Qwen model on the same gateway,
    stands in.
  - If either arm's ratio exceeds 1.5 x its model's normal ratio, that model is "gateway-confounded": reported only
    and left out of the pooled bar.

## Arms and cells

- **Rq0 and Cq are new on these cells.** Each replays the cell's cached first answer: R1's for Rq0, CT2's for Cq.
  Every later answer is a fresh call.
- **CT2 is re-run concurrently,** cache-only (amendment 1).
- **Scenes:** the 5 evaluation scenes, seeds 10-14 (25 cells per model).
## Bars (judged once, after all episodes and the mechanics check)

- **H10-Rq0 (primary).** Pooled over the models' cells, the one-sided 95% paired-t lower bound of mean(CT2 − Rq0) on
  whole-episode strict share must be above −0.05 (CT2 non-inferior).
- **H10-Cq (primary).** The same for CT2 − Cq.
- **If both hold:** "asking once and re-solving keeps the robot in line as well as re-asking, at a fraction of the
  calls" is claimed for these scenes. The calls per episode are reported beside it.
- **Why −0.05:** the same margin as item 5 and the confirmatory test (the smallest margin with useful power), equal to
  CT2's own per-cell repeat noise.
- **Pooled set:** DeepSeek and Sonnet (50 cells), plus Qwen unless it is latency-bound. Item 9's rule applies: more
  than half of its CT2 episodes without a first answer within 92 s. Then both arms barely act, and their tie would
  flatter non-inferiority, so Qwen is reported only. The scorer applies this.
- **Reported:** per model; the primary scoring; superiority (sign test, wins > losses, p ≤ 0.05); Rq0 vs R1 (does
  re-asking help the marked point?); calls per episode, mean and maximum.

## Mechanics (the run checks before sending totals)
- One episode per arm per cell, exit 0.
- The first answer is the cached one (the same pose as R1's or CT2's).
- Re-asked answers that ended in `vlm_error` are counted per episode (an outage mid-episode is part of re-asking's
  cost and is not re-run). An episode whose first answer failed is re-run, as in item 9.

## Predictions (inference)

- H10-Cq holds: re-asking the constraint method gains little, because its answers arrive late and reset the goal.
- H10-Rq0 is the closer call. Re-asking the marked point every few seconds lets R follow the line, and the old strict
  data had Rq0 about level with CT. A pass is likely but not certain.
- Superiority is not shown for Rq0.
- Calls: CT2 4, Rq0 about 25-35 (DeepSeek, Sonnet), Cq about 20.
## Run (run; from `moving/`; 1 worker per model, as the re-asking arms call often)

    cd results/moving-lines/design-and-scoring
    P=.venv/bin/python
    EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
    M=sgi_habitat.experiment_oracle
    F=results/moving-lines/re-asking
    $P run_moving.py --model soclaas_dsv41 --out runs_i10/dsv41 --workers 1 --port-base 8840 --module $M --scenarios $EVALF --seeds 10 11 12 13 14 --arms CT2 Rq0 Cq --max-rpm 30 > $F/run_dsv41.log 2>&1
    $P run_moving.py --model claude_sonnet55_low --out runs_i10/sonnet55_low --workers 1 --port-base 8840 --module $M --scenarios $EVALF --seeds 10 11 12 13 14 --arms CT2 Rq0 Cq --max-rpm 30 > $F/run_sonnet.log 2>&1
    # after item 9:
    $P run_moving.py --model soclaas_qwen38_low --out runs_i10/soclaas_qwen38_low --workers 1 --port-base 8840 --module $M --scenarios $EVALF --seeds 10 11 12 13 14 --arms CT2 Rq0 Cq --max-rpm 30 > $F/run_qwen.log 2>&1
    cd .. && $P results/moving-lines/re-asking/i10_score.py > results/moving-lines/re-asking/i10_score_out.txt

## Cost

- **DeepSeek:** about 25 cells x (about 30 + 21) calls, so about 1,300 free calls.
- **Qwen:** fewer, limited by latency; a few hundred free calls.
- **Rate:** up to about 20-30 calls a minute per worker while it runs, inside the shared 85 at 1 worker.
- **Simulator:** 75 episodes per model (CT2, Rq0, Cq) at about 96 s, so about 2 h per model at 1 worker. The
  concurrent CT2 adds zero calls.

## Scorer

`i10_score.py`, using `results/_lib/mcompare.py`. Tested with `--test` on the old seeds 0-2 episodes (CT in place of CT2). It
reproduces their calls per episode: DeepSeek Rq0 30.4 and Cq 20.8; x-test-3 10.5 and 5.9.
