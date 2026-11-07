# How to report a comparison on the realistic sets (I46)

Numbers come from `phase1/i46_stats/` (`i46_stats.py`, tables `*.txt`) and RERUN.md's "New rows" table. Everything is offline, from the recorded development runs.

## Unit: the layout

- A realistic set has 14 layouts: 6 nobody-near, 5 lines and 3 groups. Each layout gives 18 records (3 arms x 2 episodes x 3 trials), so there are 252 records per model and method.
- Records within a layout are correlated. The ICC of success by layout runs from 0.08 to 0.75 over the 27 model x method cells at both wordings (design effect 1 + 17 ICC, from 2.4 to 13.7). So 252 records carry the information of about 18 to 107 independent ones (`icc_table.txt`).
  - For the paired C-E and C-R differences the ICC runs from 0.10 to 0.73 (design effect 2.7 to 13.3).
  - E-R at the new wording is the exception: 0.01 to 0.05.
- Never use a record-level binomial test, McNemar test or Wald interval. Say "14 layouts", not "252 trials".

## Test: two-sided exact sign-flip by layout

- **Statistic.** For each layout l, d_l is the mean over its 18 paired records of (success A - success B). T = sum of the d_l. The p value is the share of the 2^14 sign patterns with |T*| >= |T| (`rerun/stats/stats.py`, `signflip_int`).
- **Report** the p value, the layouts favouring A / B / tied, and the floor 2 / 2^k for k non-tied layouts. Example: Qwen new wording, C vs E: p = 0.0078, 8/0/6, which is at the floor.
- **At least 6 layouts must move** before p <= 0.05 is possible. On these sets no claim within one situation can reach it: the floor is 0.0625 for lines (5 layouts) and 0.25 for groups (3).
- A claim made "for every model" needs Holm's correction across the models.

## Intervals: t over the 14 layout differences

- Give mean(d) +/- t(0.975, 13) x sd(d) / sqrt(14), times 252, in records. Example: Qwen C-E +32 (+8.4 to +55.6). The 12 C-E and C-R rows are in RERUN.md and match `tci_table.txt`.
- Do not use the percentile bootstrap: with 14 clusters it runs narrow.

## Dependence to state every time

- **Shared target draw.** Within a stage, R, E and C share one target draw (the plan and context calls). So a comparison inside one stage is paired by trial, and R and E are nearly identical at nobody-near (standard goal): identical for Qwen, GLM at the new wording, Haiku, Sonnet and Gemini, but DeepSeek differs in 3 of 108 records (s23) and 7 of 108 (s20), and old-wording GLM in 1.
- **Different stages.** Old-wording Qwen and GLM C (s19) against R/E (s16b) come from different stages and share no target draw.
- **Cached people.** The C people answers of s21 Qwen and s23 GLM and DeepSeek are cache copies of s19/s20. New against old C for those three is not an independent replication of perception.
- **Duplicates.** s22 trials 0-2 duplicate s16b; drop them.
- **In-sample prompt.** The context prompt was chosen in-sample on these sets.
- **Run-to-run noise.**
  - One 3-trial batch has an SD of 6-8 records.
  - The within-batch estimators give 6.0-6.2 for Qwen and 3.9-6.2 for the other five models (item binomial, pooled over R, E and C). They miss batch-to-batch shifts. Qwen E's batch totals 214 / 204 / 196 (s16b / s22 / s21) suggest such shifts exist.

## Minimum detectable differences for live A/B runs (80% power, two-sided 0.05)

The table reads two ways. "Run noise" covers these layouts only. "Over layouts" covers generalising to new layouts.

| Design | SD used | MDD (records of 252) |
|---|---|---|
| Two independent 3-trial batches, nothing shared (run noise) | sqrt(2) x 6-8 | 24-32 (9-13 pp); Qwen 7.93 gives 31 (12.5 pp) |
| The same with 2 / 3 / 4 batches per arm | / sqrt(b) | 22 / 18 / 16 |
| Paired in one stage, same trials and target draws (run noise) | within-batch paired SD 3.9-7.3 | 11-21; Qwen pool E-R 20 |
| Over 14 layouts, t test, E-R-like contrast | sd_layout 0.09-0.14 | 18-28 |
| Over 14 layouts, C-E / C-R-like contrast | sd_layout 0.16-0.39 | 33-79 |

- **Small gates.** Gates with margins of 1-3 records sit inside the noise. Read them on paired records that share cached perception, or widen them.
- **Unsettled single batches.** A single batch cannot settle an E-R-sized effect. Qwen's E-R ran +39, +16 and +17 over three batches, and at the new wording E-R is -9 to +17 across models (none significant).

## Sizing a sealed set of fresh layouts (post-hoc, optimistic)

- **Power curve.** Resampling the observed new-wording layout effects gives this two-sided exact-test power (`power_table.txt`):

  | Fresh layouts | Power for C-E and C-R, six models |
  |---|---|
  | 8 | 0.14-0.42 |
  | 14 | 0.78-0.97 |
  | 20 | 0.98-1.00 |
  | 30 | 1.00 |

- **Why it is optimistic.** It reuses the same 14 layouts that showed the effect. Because nearly every observed non-zero layout effect favours C, power here measures little more than the chance of drawing at least 6 non-tied layouts.
- **Recommendation.** Plan for at least 20 fresh layouts. For any per-situation claim, at least 6 layouts in that situation is the floor of the test, not a powered size: where one of 3 observed group layouts is tied, power at 6 is about 0.09 and reaches 0.8 only at 14. Plan for 14 per situation wherever a layout may tie.
