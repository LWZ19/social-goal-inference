# Update of the metric list v1 draft after the confidence probe's results (2026-10-18)

**What this is.** The draft `METRIC_LIST_v1_DRAFT.md` was revised before the confidence probe's results page existed. Its confidence bars and decisions D4 and D7 still said they waited for it. This update brings them in line with the results. The text before the update is kept as `METRIC_LIST_v1_DRAFT.before-a2.md`.

**Naming.** The task called the probe "A2". The draft keeps "the confidence probe", because A2 also names I09's step A2, whose setting D the probe uses. The draft now calls setting D "the detector repair".

**Sources** (paths relative to `results/2026-10-17-beyond-vlm/`):
- `results/still-scenes/confidence-signal/FROZEN.md`: AUROC by midranks, the layout-cluster bootstrap, the selective-accuracy grid.
- `results/still-scenes/confidence-signal/followup/exposure.txt` (Tables B, F, G, H).
- `results/still-scenes/confidence-signal/tables.txt` T5 and `summary.json` T5, for the per-cell abstention gains.
- The draft, `CHANGELOG.md` and `FACTCHECK.md`.

**Length.** 6,548 words before, 6,881 after (+333).

## Changes

| # | Where | Old (summary) | New (summary) | Source |
|---|---|---|---|---|
| U1 | Status line | Revised after three reviews, then fact-checked | Adds: then updated with the probe's results (this file) | This update |
| U2 | Summary item 4 | Confidence is judged within each model's cell | Within each model's cell **and against the solver's flags** | README §5 (model identity 0.658), §10 |
| U3 | Glossary, "Confidence probe" | Its results page appeared at 18:05, after the revision | Points to §2.1. Defines P1, S2, S4 (two extra runs), S6z, the detector repair (I09 step A2's setting D) and the flags, so D4 and D7 stand alone | README §1 signals table; FROZEN §4 |
| U4 | D4 | "The probe has tested it (judged signal: neither)" | P1 is neither, on a fragile margin (cells 0.645, 0.738, 0.688; pooled 0.708). Reported: S2 0.741, S6z 0.752, S4 0.774. Confirming needs records the probe did not use. Q1 stays the recommendation | README §0, §4, §8 question 1 |
| U5 | D7 | Abstention number a *placeholder* until revisited against the results page. Why: model-only AUROC 0.658, Brier skill +0.08 | Yes, as revised: cells with at least 20 failures, a paired win over the flags, abstention judged against the flags too. Valid only on records the probe did not use. +4 points stays a *placeholder* until the project rules on the scope of an abstain output. Why adds the flags' 0.595 and the flags' own abstention gain of +3.0 points | README Table 3.2, §5, §8 question 3; `tables.txt` T5 (drafter's arithmetic) |
| U6 | §1, "AUROC and selective accuracy" | Layout-cluster bootstrap, 2,000 draws, percentile, narrow with 13 clusters | Adds the probe's frozen readings: paired draws, midrank AUROC (ties half), expected-value ties at the selective-accuracy cutoff. Tie rules are fixed in advance, since P1's DeepSeek AUROC spans 0.670–0.706 by tie-breaking | FROZEN §5; README §5 ("Ties decide the margin") |
| U7 | §2.1 (c), description | Summed over the three old-wording realistic cells; per-cell counts 9 / 45 / 48 | Summed over the counting cells. Per-cell, right-target and wrong-target AUROC are each reported beside the flags' | README §10 (S2's edge over the flags is in wrong-target failures) |
| U8 | §2.1 (d), description | Gain = selective accuracy minus base rate, mean over cells; binormal scale (+4.6 at AUROC 0.70) | Base rate named as what random abstention keeps on average; mean over counting cells. Adds the measured within-cell gains over all three cells: P1 +8.5, S2 +8.1, S6z +8.6, flags +3.0 (flags +4.6 over GLM and DeepSeek alone) | `tables.txt` T5; drafter's arithmetic (record-weighted mean of each cell's gain at 80% coverage) |
| U9 | Q1 bar (b) | "…such as the confidence probe's S6, cannot pass by learning which model produced the record" | Same meaning, shorter | — |
| U10 | Q1 bar (c) | Stratified AUROC ≥ 0.70, lower end ≥ 0.60, strictly above the any-flag baseline; kill below 0.60; right-target option | **Counting cells:** at least 20 failures and 20 successes; other cells reported, not judged; a claim needs at least 2. **Pass:** stratified AUROC over counting cells ≥ 0.70 with lower end ≥ 0.60 (kept); no counting cell below 0.60 (point); paired difference from the flags' stratified AUROC with lower end above 0. **Kill:** below 0.60, or at or below the flags'. Right-target option kept. New: a signal that overlaps a measured repair is also reported where that repair changes nothing. Planning note: the paired intervals against the flags ran ±0.13–0.18, so 400–440 records need a gap near 0.15–0.2 | README Table 3.2 (old Qwen [0.392, 1.000]), §3 (P1 − flags +0.112 [−0.088, +0.273]), §5, §7; README §10 and `exposure.txt` Table F (S2 − flags +0.071 [−0.083, +0.182]), Table B (P1 0.557) |
| U11 | Q1 bar (d) | *Placeholder:* gain ≥ +4 points at 20% abstention, interval excluding zero | Within counting cells: gain over random with its interval excluding zero, and above the flags' gain (point). +4 points kept as a *placeholder*, with the reason | `tables.txt` T5; README §8 question 3 |
| U12 | Q1 bars, new last line | — | Says which parts were set after the probe's results (so they hold only on records it did not use) and which were set before | Task; README §0 ("Not blind") |
| U13 | §2.1, "The confidence probe" paragraph | The README did not exist; a fact-check note gave P1's cells and pooled value; 0.70, 0.60 and +4 "not yet revisited" | States the results: P1 neither on a fragile margin (cells, pooled 0.708 against the flags' 0.595; dropping one of 3 particular layouts, or trial 1 or 2 alone, makes it a success). Reported S2 0.741, S6z 0.752, S4 0.774, all designed knowing the repair's results. Post hoc on the 404 records the repair does not change: P1 0.557; S2 0.694 [0.587, 0.743]; S2 − flags +0.071 [−0.083, +0.182]; edge mostly on wrong-target failures (0.716 against 0.566; right-target 0.671 against 0.683) | README §0, §4, §5, §10; `exposure.txt` Tables B, F, H |
| U14 | §2.1, table "Where v1 differs from FROZEN.md" | v1 column restated the old bars. Why: 9 failures; one pooled ranking rewards the model (0.658) | v1 column points to bars (c) and (d). Why adds: model identity alone clears "pooled above the flags" (0.658 against 0.595); P1 − flags +0.112 [−0.088, +0.273] | README §3, §5 |
| U15 | §2.3, I04 row | "The ε check and stratified AUROC were not computed" | Stratified AUROC 0.447, recomputed by the probe (its S5b); the ε check is still not computed | README Table 3.2 |

## Placeholders

- **Settled and removed.** D7's "placeholder until revisited against the results page" is gone, because the revisit is done. The (d) bar's interval clause is now a firm part of the bar, not a placeholder.

## Which numbers depend on the probe's results

- **Set after seeing them** (so they hold only for studies on records the probe did not use): the 20-failure counting-cell rule, the per-cell floor of 0.60, the paired comparison with the flags in (c), the comparison with the flags in (d), and the repair-overlap report.
- **Set before them** (kept unchanged): the 0.70 level, the 0.60 kill, 20% abstention, and bars (a) and (b).

## Flagged, not changed

- **20 failures is a choice.** It is a round number, and it leaves out cells like old Qwen (9 failures). Abstaining gains little in such cells anyway: a perfect ranker gains +6.3 points at old Qwen.
- **Bar (b)'s "2 of 3 cells" clause was not revisited.** Brier skill uses every record, so few failures matter less there. It still counts old Qwen.
- **CHANGELOG.md and FACTCHECK.md were not edited**, since only the draft and this file were in scope. FACTCHECK's C7 note ("bar numbers not yet revisited") is now superseded by this update.
- **The arithmetic in U8** was not re-checked by an independent verifier. It is labelled "drafter's arithmetic" in the draft.
