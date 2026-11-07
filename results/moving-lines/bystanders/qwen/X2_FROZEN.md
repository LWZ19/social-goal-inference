# Extra X2: the bystander confirmation on a school model (Qwen 3.8; frozen before any X2 episode)

## Question

Does motion evidence (CT3) stop the robot from lining up behind a bystander, on Qwen 3.8 as on DeepSeek? And does it
raise time in line?

The frozen confirmation passed on DeepSeek (`results/moving-lines/bystanders/BYSTANDERS_CONFIRM_FROZEN.md`). Its Sonnet addendum failed the
capture bars. No school-panel model has run it.

## Design: the confirmation's, unchanged except the model

- **Scenes:** cafe_line2_bystanders and sweet_spot_line2_bystanders, seeds 15-44. That is 60 cells.
- **Arms:** CT2, CT3 and Gl, in one invocation per cell into `results/data/moving/runs_x2/soclaas_qwen38_low/`.
  - CT2 makes the live first answer. CT3 and Gl replay it through the cache, as run_moving groups them.
- **Model:** `soclaas_qwen38_low` (free). 92 s episodes.
- **Scorer:** `results/moving-lines/bystanders/bystanders_confirm_score.py`, the frozen confirmation scorer, unchanged. Its bars are C1 (share,
  CT3 vs CT2), C2 (capture) and R (roles, CT3 vs Gl).

## Bars

- The confirmation's C1, C2 and R, read as frozen, for Qwen.
- **Latency.** Item 9's rule applies: if more than half of Qwen's CT2 episodes have no first answer within 92 s, the
  92 s result is labelled latency-bound.
  - CT2 and CT3 are then re-run with `--seconds 240`, cache-only (zero calls), into
    `results/data/moving/runs_x2/soclaas_qwen38_low_long/`.
  - The bars are read there as a labelled secondary.

- **What happened.** `bystanders_confirm_score.py` raised IndexError on its first pass over Qwen's runs: in a CT2
  episode with no answer at all, `answers` is an empty list and `[0]` fails. DeepSeek's confirmation never had one.
- **The fix.** `x2_score.py` is a copy with only those two lines guarded (`(… .get("answers") or [{}])[0]`) and its
  import path; the diff is in the hash set. On DeepSeek's confirmation runs it reproduces the frozen output exactly.
- **Bar R in the 240 s secondary.** The frozen secondary re-runs CT2 and CT3 only, so it has no Gl episodes; R is
  therefore not judged there (the scorer's FAIL line for R is an artefact of the missing arm).

## Predictions (inference)

- **Capture:** CT3 cuts capture sharply (bar C2), as on both other models. The join gate does not depend on the model.
- **Share (bar C1):** small, as on DeepSeek (+0.018). It may not reach p ≤ 0.05 at n 60.
- **Roles (bar R):** CT3 beats the geometry chain if Qwen's rosters name few bystanders. Sonnet named them in 13 of
  29 answers, DeepSeek in 10 of 51.

    cd results/moving-lines/design-and-scoring
    P=.venv/bin/python
    F=results/moving-lines/bystanders/qwen
    $P run_moving.py --model soclaas_qwen38_low --out runs_x2/soclaas_qwen38_low --workers 2 --port-base 8900 \
      --module sgi_habitat.experiment_oracle --scenarios cafe_line2_bystanders sweet_spot_line2_bystanders \
      --seeds $(seq 15 44) --arms CT2 CT3 Gl > $F/run.log 2>&1
    python3 bystanders_confirm_score.py runs_x2/soclaas_qwen38_low > $F/x2_score_out.txt
    # only if latency-bound (item 9's rule), cache-only secondary:
    SGI_CACHE_ONLY=1 $P run_moving.py --model soclaas_qwen38_low --out runs_x2/soclaas_qwen38_low_long --workers 2 --port-base 8900 \
      --module sgi_habitat.experiment_oracle --scenarios cafe_line2_bystanders sweet_spot_line2_bystanders \
      --seeds $(seq 15 44) --arms CT2 CT3 --seconds 240 > $F/run_long.log 2>&1
    python3 bystanders_confirm_score.py runs_x2/soclaas_qwen38_low_long > $F/x2_score_long_out.txt

## Cost

- **Calls:** about 4 per cell for CT2's first answer, about 240 free calls.
- **Simulator:** 180 episodes at about 96 s, so about 2.4 h at 2 workers. The secondary, if needed: 120 episodes at
  about 245 s, so about 4 h at 2 workers.

## Check before running

`bystanders_confirm_score.py` reads cells by scenario and seed from the roots given. Run against
`runs_x2/soclaas_qwen38_low`, it scores only those episodes. Its hash is in `X2.sha256`.
