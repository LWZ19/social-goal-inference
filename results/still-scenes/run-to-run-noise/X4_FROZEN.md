# Extra X4: run-to-run noise for GLM and DeepSeek on the still scenes (frozen before any X4 call)

## Question

How much do the R, E and C totals move between independent batches of the same runs?

The noise bar every still-scene comparison is read against is measured for Qwen only: batch SD 7.93 records of 252,
df 4. GLM's two same-prompt C runs differed by 16.

## Design

- **Two fresh batches** (noise1, noise2) of the s23 runs, for GLM-5.3 and DeepSeek V4.1.
  - The 3 realistic development sets (plain, staffed, real) x R, E and C, at 3 trials: 252 records per method per
    batch.
- **Configs:** E3's s23 configs (`results/still-scenes/hidden-situations-and-more-models/configs/{variant}_{method}.yaml`) with one change, a batch-specific cache
  folder (`.cache/vlm_noise1`, `.cache/vlm_noise2`; `configs/`, with the diff in the hash set).
  - Every call is fresh: no cache entry exists there.
  - Within a batch E and C still reuse R's plan and context answers through that batch's cache, as s23 did. So the
    within-batch comparisons stay paired.
- **Measures** (`x4_score.py`), per model and method:
  - the three batch totals (s23, noise1, noise2), their mean and SD;
  - C − R and C − E within each batch;
  - record-level flips between batches.

- GLM's per-call timeout is 600 s (`timeout_s: 600.0` on soclaas_glm53 only) in this item's 18 configs, for the X4
  GLM runs not yet started. GLM runs already running keep 180 s; their vlm_errors are re-run later at 600 s.
- timeout_s is not in the cache key.
- The configs' diff against E3's now shows the cache folder and that timeout.

## Reading (reported)

- The SD gives each model's batch noise, beside Qwen's 7.93. The report states the smallest detectable difference
  between independent batches as about 2.8 x SD.
- **Caveat:** in s23, the gateway models' C people answers were cache copies of s19 and s20, so s23's C is not a fully
  independent draw of the people call. The two new batches are.

## Cost

- **Calls:** about 2,100 per batch per model (756 records x about 2.8), so about 8,400 free calls in all.
- **Wall time:** DeepSeek about 1.5 h a batch with the variants in parallel; GLM about 4-6 h a batch (its gateway).
  Run it when the gateway is otherwise quiet.

## Scorer

`x4_score.py`. On the existing s23 runs it reproduces their totals: GLM 132/123/170 and DeepSeek 120/126/185.
