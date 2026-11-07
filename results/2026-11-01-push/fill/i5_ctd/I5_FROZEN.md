# Item 5: does tracking people from the robot's own camera cost time in line? (CTd vs CT2, non-inferiority; frozen before any new episode)

## Question

Every moving result except CTd and CTe uses the sim's true people positions. CTd is CT2 on tracks from the robot's own
camera: I09 A1's detector every 0.5 s, plus the E1d tracker.

Is CTd non-inferior to CT2 within 0.05 of strict share in line?

The old evidence is 15 cells each on Sonnet and DeepSeek: −0.018 and −0.017, with no bar. Pooled at 30 cells, the lower
bound is −0.054, so too few cells to exclude a loss of 0.05.

## Arms and cells (zero model calls: both arms replay the cell's cached first answer)

- **The earlier CT2 and CTd episodes** in `moving/runs/<model>/` are read only for a repeat check: the same arm and
  answer, earlier run vs this run. That is a timing-only floor on up to 125 cells for CT2 and 30 for CTd.
- **Scenes:** the 5 evaluation scenes.
- **Models and seeds:**
  - Pooled for the bar: Sonnet 5.5 low (seeds 10-19, 50 cells), DeepSeek V4.1, Haiku 5.5 and Gemini 3.8 (10-14, 25
    each). That is 125 cells.
  - Reported only: Qwen 3.8 and GLM-5.3 after item 9.

- **H5 (primary).** Pooled over the 125 cells, the one-sided 95% paired-t lower bound of mean(CTd − CT2) on
  whole-episode strict share must be above −0.05.
  - If it holds: "detector tracking costs no more than 0.05 of time in line" is claimed for these scenes.
  - Why −0.05: it is the smallest margin with useful power in the confirmatory test (`CONFIRM_CT4PJ.md` §3). It is also
    CT2's own per-cell repeat noise (mean |d| 0.052). It is about a sixth of CT2's level.
  - Power: the old pairs' spread (SD about 0.12) gives a lower bound about 0.018 below the mean at 125 cells. A true
    loss of 0.02 passes.
- **Under the primary scoring:** the same bound, reported.
- **Reported, no bar:** per model; the sign test; the repeat check; tracking quality (live tracks matched to a true
  person, id switches, stale share, first-match rate, detector ms), as `summarise_fresh.tracking` prints it.

- Both arms per cell, exit 0, no `vlm_error` (a cache miss would show as one).
- The same first-answer pose as the cell's earlier CT2.
- The detector service answers throughout: detection frames with errors are counted in the tracking table.
- The scorer flags pooled pairs ≠ 125.

## Predictions (inference)

- H5 holds, with a mean of about −0.02 and a lower bound of about −0.035.
- Under the primary scoring the bound is lower, but probably above −0.05.
- Matched tracks about 0.9; about two thirds of tracks stale (out of view, held at the last position).

- The detector service needs about 1.3 GB of GPU, alongside the sim workers.
- The tracker service keys each episode by its output path, so parallel workers and successive models never share a
  tracker.

      # after item 9 (reported only):
      run soclaas_qwen38_low soclaas_qwen38_low 10 11 12 13 14
      run soclaas_glm53 soclaas_glm53 10 11 12 13 14
      kill $TP $DP
      cd .. && $P fill/i5_ctd/i5_score.py > fill/i5_ctd/i5_score_out.txt     # zero calls

## Cost

- **Calls:** zero (cache-only).
- **Simulator:** 250 episodes (125 cells x 2 arms), about 96 s each, so about 3.3 h at 2 workers. Add 100 after item 9.
- **GPU:** about 1.3 GB for the detector service while it runs.

## Scorer

- `i5_score.py`, using `fill/mcompare.py` and `summarise_fresh.tracking`.
- Tested with `--test` on the earlier 30 cells. It reproduces −0.018 and −0.017; pooled lower bound −0.054; tracking
  matched 0.89 and 0.93, stale 0.71 and 0.67.
