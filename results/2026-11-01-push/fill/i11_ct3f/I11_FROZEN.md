# Item 11: a frozen claim for CT3F, the best moving method (CT3F vs CT2, then vs CT3; fresh layouts; frozen before any candidate in the range is drawn; amendment 1 likewise; amendment 2 before scoring)

## Why, and what is already known

CT3F (CT3 plus the group follow test, the gated anchor and the photo-time replay) is the best moving method, but it
never had a frozen claim of its own.

**Description only (spent data):** on the 72 confirmation layouts (candidates 2000-2449), where CT3F and CT2 both ran
with arms fixed before those layouts were drawn, CT3F beat CT2 on every model (`i11_score_test_out.txt`). Strict share
from the first service:

| model | CT3F | CT2 | layouts W/L/T | lower bound |
|---|---|---|---|---|
| DeepSeek | 0.208 | 0.051 | 44/8/14 | +0.10 |
| Sonnet | 0.246 | 0.026 | 56/5/10 | +0.17 |
| Haiku | 0.232 | 0.078 | 39/5/11 | +0.10 |
| Gemini | 0.231 | 0.051 | 43/5/23 | +0.13 |

- CT2 is low on these layouts because their lone bystander captures it.
- So H11a below is expected to pass. **H11b, CT3F vs CT3, is the informative test:** it isolates CT3F's own rules. Its
  prior is the 3 Nov sweep ("ours" layouts, DeepSeek, spent): strict from the first service, CT3F 0.226 against CT3
  0.118.

- **Generate:** `make_sweep.py --start 3000 --per-base 450`.
- **Select:** `select_i11.py`, a copy of `select_confirm.py` with only the docstring, OUT, CAND, the header label and the
  output file changed (the diff is in this note's hash set).
  - Quotas as the confirmatory test: ours 48, chain 12, mixed 6, easy 6, half from each base.
  - The stress flags are crossed 2 x 2. That gives 72 layouts, one seed each.
- **Model:** DeepSeek V4.1 (free).
- **Arms:** CT2 (live first answer), then CT3F and CT3, replayed from CT2's cached first answer (cache-only).

## Amendment 1 (6 Nov, before any candidate in the range is drawn; machine load)

- **The issue.** An answer and every re-solve land at their send time plus wall latency, including sgi.serve's local
  work (`experiment.py` 271-285). So CT2's live stage and a later replay stage would run under different machine
  loads.
- **The fix.** CT2 runs live once into `runs_i11/dsv41_live/`, to fill the cache. Then **CT2, CT3F and CT3 run
  together**, cache-only, into `runs_i11/dsv41/`, and the bars are judged there.
- **Repeat check.** The live CT2 is scored as a second root, for a repeat check of CT2 (live vs replayed).

## Amendment 2 (7 Nov, before scoring; the replays' first-answer errors)

- **What happened.** In 1 layout (sweet_3016) all three replays' first answer ended in vlm_error: a cache-only miss.
  The live CT2 run had no answer within its episode, so nothing was cached.
  - In many other layouts the live CT2's first answer arrived only after its 92 s episode ended: 33 of 72 had no
    answer within the episode, and where it came the median was 72.6 s. On the confirmation layouts (2-3 Nov) the
    median was 12.6 s, with none missing (measured from the episode files, zero calls). All three arms share that
    late answer through the cache, so they tie there: fair, but uninformative.
- **The rule.** A layout whose first answer ended in vlm_error in either compared arm is left out of that comparison
  and counted. The scorer also prints, per arm, how many episodes had no first answer within the episode.

## Bars (fixed order; a later one is judged only if every earlier one passes; no multiplicity correction)

The measure is strict share in line from the first service after delivery (`score_sweep.episode_scores`
`after_service_strict`), as the confirmatory test. Ties are |d| < 0.01, with the exact two-sided sign test over layouts
with both values.
- **H11a:** CT3F − CT2 has wins > losses and p ≤ 0.05.
- **H11b:** CT3F − CT3 has wins > losses and p ≤ 0.05. If it holds, "CT3F's rules beat CT3's on fresh layouts" is
  claimed for DeepSeek.
- **Reported:** whole-episode strict share, cut-in samples, CT3 − CT2, mean d and the paired-t lower bound.

- 72 layouts per arm, exit 0, no `vlm_error`.
- CT3F's and CT3's first answer is CT2's (the same pose).
- `check_mech.py` faults none. Re-run a CT2 episode that had a gateway outage (`vlm_error` first answer) before the
  replays.

## Predictions (inference)

- H11a passes by a wide margin (as on the confirmation layouts).
- H11b passes, with CT3F ahead by about 0.08-0.12 and more wins than losses by about 2 to 1.
- CT3F has more cut-in samples than CT3, as CT3F had more than CT4PJ on the confirmation layouts.

    cd .
    PYTHONPATH=sim ~/miniforge3/envs/sgi-habitat/bin/python results/2026-11-01-push/moving/sweep/make_sweep.py --start 3000 --per-base 450 --out results/2026-11-01-push/moving/sweep/i11_candidates
    cp results/2026-11-01-push/fill/i11_ct3f/select_i11.py results/2026-11-01-push/moving/sweep/select_i11.py
    cd results/2026-11-01-push/moving/sweep && python3 select_i11.py > i11_select_out.txt
    python3 -c "import json;[print(f\"i11/{l['name']} {l['seed']}\") for l in json.load(open('i11_selection.json'))['layouts']]" > jobs_i11.txt
    cd .. && P=.venv/bin/python
    $P run_moving.py --model soclaas_dsv41 --out runs_i11/dsv41_live --workers 2 --port-base 8820 --module sgi_habitat.experiment_oracle --jobs sweep/jobs_i11.txt --arms CT2 > ../fill/i11_ct3f/run_ct2_live.log 2>&1
    SGI_CACHE_ONLY=1 $P run_moving.py --model soclaas_dsv41 --out runs_i11/dsv41 --workers 2 --port-base 8820 --module sgi_habitat.experiment_oracle --jobs sweep/jobs_i11.txt --arms CT2 CT3F CT3 > ../fill/i11_ct3f/run_replay.log 2>&1
    cd .. && $P fill/i11_ct3f/i11_score.py sweep/i11_selection.json runs_i11/dsv41 runs_i11/dsv41_live > fill/i11_ct3f/i11_score_out.txt
    # the bars are read from the runs_i11/dsv41 section; the dsv41_live section is the CT2 repeat check

## Cost and order

- **Calls:** about 4 per layout for CT2's first answer, about 290 free calls. The replays make zero calls.
- **Simulator:** 288 episodes (72 live CT2, plus 72 x 3 replays) at about 96 s, so about 3.8 h at 2 workers.
- **Suggestion:** given the spent-data evidence for H11a, run this after items 12 and 10 if simulator time is short.
  H11b is what it adds.

## Scorer

`i11_score.py`, tested on the confirmation logs of four models (the table above, `i11_score_test_out.txt`).
