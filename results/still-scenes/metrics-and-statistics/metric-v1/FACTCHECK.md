# Fact-check: I14 metric list v1 draft (2026-10-18)

**Result.** 147 claims checked. 140 confirmed as written. 7 corrected in the draft: 1 number, 2 wordings that the sources do not support, 3 incomplete citations, and 1 status update (made in four places). Three more points are flagged below but not changed.

Paths are relative to `results/2026-10-17-beyond-vlm/`.

## Corrections made in the draft

| # | Where | Was | Now | Why |
|---|---|---|---|---|
| C1 | §2.1 (b) | "per-cell rates alone score +0.08 pooled and +0.35 at old Qwen" | "+0.32 at old Qwen (+0.35 if the base rate is pooled over all 549 primary records)" | Recomputed. With the base rate pooled over the 441 realistic records (0.769), old Qwen's Brier skill is +0.325. The +0.348 appears only with the rate pooled over all 549 primary records (0.760). The pooled +0.077 holds either way |
| C2 | §4, label validity | "I06's lines, measured on oracle C only", then five bands | "I06's lines (criterion (c): at most 5%, L2 empty at most 2%; K2: above 30% or above 10%), with the bands between them added here" | I06 sets only the outer lines (`results/still-scenes/crowds/README.md` §6). The 5–20%, 20–30% and 2–10% bands are this draft's additions |
| C3 | D15, Why | "Same work, different load: 18.3 against 11.9 ms; 215.1 against 127.0 ms" | "The same span timed twice under different conditions, causes not settled" | I09 calls its pair the same machine under different load (`results/still-scenes/people-finder/early-detector/phase2/README.md` §1). For the flow pilot the verifier names two other likely causes, an in-process GPU poller and a different projection code path, and timed only every third record (`results/still-scenes/learned-models/flow-model-pilot/README.md` §6) |
| C4 | Appendix B, fit seeds row | Source "REPORT I46" | "REPORT I46; gate 2: `results/still-scenes/overview/phase-notes/phase1d.md` §2.2" | REPORT I46 names I03, I11 and I09 A1 as users of seeds 100–109, not gate 2 |
| C5 | Appendix B, training seeds row | `results/still-scenes/learned-models/flow-model-pilot/README.md` §7 | §1, §7 | The 110–189 / 190–199 split is stated in §1; §7 says the seeds are spent as training data |
| C6 | Appendix B, seeds 213–219 row | `results/still-scenes/people-finder/early-detector/phase2/README.md` §3 | §1, §3 | a decision (seeds 200–219 a test set, never for fitting) and the 14 rows are in §1 |
| C7 | Glossary; D4; D7; §2.1 "The confidence probe" | "now running", "tests it now", "until the confidence probe reports" | Says the results page appeared at 18:05, after the revision; adds a dated note: P1 neither (cells 0.645, 0.738, 0.688; pooled 0.708 against the flags' 0.595), model-only 0.658 confirmed, bar numbers not yet revisited | `results/still-scenes/confidence-signal/README.md` §0, §4, §5 |

The Status line also now points to this file.

## Flagged, not changed

- **CHANGELOG.md** still gives "+0.35 at old Qwen" (row 1.3) and "+0.348 at old Qwen" (Checks line). It is the revision's record, so it was left as written. See C1.
- **Holm floors (§1, D13).** "7 for 3 models, 8 for 6" is the number of non-tied layouts the *smallest* p in the family needs. Rows after it may have fewer: Sonnet C−E, with 7 non-tied layouts (p 0.0156), still passes Holm over 6 models. The wording is correct as a requirement; it is not a per-row floor.
- **D13, "no per-situation claim can reach p ≤ 0.05".** This copies `results/still-scenes/metrics-and-statistics/statistics/HOW_TO_REPORT.md`, which argues it for lines (floor 0.0625) and groups (0.25). Nobody-near has 6 layouts (floor 0.031) but R and E are nearly identical there. The draft matches its source.

## Every claim checked

"Confirmed" means the source gives the same number or statement. "Recomputed" means the arithmetic was redone here.

### Summary and glossary

| # | Claim | Source | Result |
|---|---|---|---|
| 1 | V3 fitted to σ ≈ 0.06 m | `results/still-scenes/overview/phase-notes/phase1d.md` Table 1.1 (σ 0.059 in 16 of 19 folds) | Confirmed |
| 2 | Band-uniform is gate 2's baseline, uniform at the usual target distance | `results/still-scenes/overview/phase-notes/phase1d.md` §2.2 | Confirmed |
| 3 | Family A: one confuser behind a line, six levels | `results/still-scenes/crowds/README.md` §1 | Confirmed |
| 4 | L2 = L1 minus a 0.3 m body gap to non-members and other groups' circles | `results/still-scenes/crowds/README.md` §1; REPORT §7 | Confirmed |
| 5 | Gateway models get the same prompt text under both wordings; new-wording people answers are cache copies | REPORT §8 "Wording change f1a488a"; `results/still-scenes/overview/phase-notes/phase1.md` cautions | Confirmed |
| 6 | Primary cells: s19 Qwen, s19 GLM, s20 DeepSeek, and watching | `results/still-scenes/metrics-and-statistics/metric-v0/METRIC_NOTE.md` "Records" | Confirmed |
| 7 | MDD = difference found 80% of the time at two-sided 0.05 | `results/still-scenes/metrics-and-statistics/statistics/mdd_table.txt` | Confirmed |
| 8 | Holm's procedure as stated | `HOW_TO_REPORT.md` (Holm across models) | Confirmed |
| 9 | Visibility rule = the package's `is_visible` | `results/still-scenes/people-finder/early-detector/phase2/README.md` §1 | Confirmed |
| 10 | Confidence probe: disagreement between the VLM's people and the detector's | `results/still-scenes/confidence-signal/FROZEN.md` §1 | Confirmed; status corrected (C7) |
| 11 | Idea codes (I04, I06, I09, I14, I46, A0, C5 gate 2) | REPORT §4; `results/still-scenes/overview/phase-notes/phase1.md` §3; `results/still-scenes/overview/phase-notes/phase1d.md` | Confirmed |

### Decision table

| # | Claim | Source | Result |
|---|---|---|---|
| 12 | D1: at 30 deg DeepSeek realistic top-1 falls 109 → 101 of 157; most cells flat | `results/still-scenes/overview/phase-notes/phase1.md` §4 | Confirmed |
| 13 | D2: 0.50 m is the human-norm edge; 0.55 m the closest this robot can stand; 0.55 m changes 12 R/E verdicts, no conclusion | REPORT §1.1 finding 8, §1.3 | Confirmed |
| 14 | D3: SEWA 13 of 252 ≈ 5 points | Recomputed (5.2 points) | Confirmed |
| 15 | D3: an MDD bar rejects 6 of 12 C−E/C−R rows | `mdd_table.txt` M3 against `tci_table.txt`: Qwen C−E, Qwen C−R, GLM C−R, Sonnet C−E, Sonnet C−R, Gemini C−E fall below their MDD | Confirmed |
| 16 | D3: Qwen C−E +32 against MDD 33.1, p 0.0078 | `tci_table.txt`; `mdd_table.txt` | Confirmed |
| 17 | D3: all 12 rows pass the proposed rule, Holm over 6 models included | `tci_table.txt` p values, Holm within each comparison type | Confirmed |
| 18 | D3: 7 rows are strong (lower end ≥ 13) | `tci_table.txt`: GLM C−E, DeepSeek C−E and C−R, Haiku both, Sonnet C−E, Gemini C−R | Confirmed |
| 19 | D4: Q2 serves a planner, Q1 a confidence or abstain signal | `results/still-scenes/overview/phase-notes/phase1d.md` §5 | Confirmed; status corrected (C7) |
| 20 | D5: I04's score rewarded a near-point mass; gate 2's rewards spread | `results/still-scenes/overview/phase-notes/phase1d.md` §1, §2.2, §5 | Confirmed |
| 21 | D6: I04's collapse line would fire on V3 in 549 of 549 | `results/still-scenes/overview/phase-notes/phase1d.md` Table 1.6 note, §5 | Confirmed |
| 22 | D7: a model-only signal scores pooled AUROC 0.658 | Recomputed (0.6578); `results/still-scenes/confidence-signal/README.md` §5 | Confirmed |
| 23 | D7: it beats a pooled base rate, Brier skill +0.08 | Recomputed (+0.077) | Confirmed |
| 24 | D8: uniform beats band-uniform by ≥ 0.10 nats in 2 of 3 gate-2 folds | `results/still-scenes/overview/phase-notes/phase1d.md` Table 2.2 (+0.131, −0.105, +2.919) | Confirmed |
| 25 | D9: AR recovers 57% and 69% over the room, 96% and 33% in the disc | `results/still-scenes/overview/phase-notes/phase1.md` §4 | Confirmed |
| 26 | D10: L2 mirrors the solver's 1.0 m standoff | REPORT §8 "Circularity" (own-space rule); `results/still-scenes/crowds/README.md` §2 (1.0 m) | Confirmed |
| 27 | D11: no joining rule succeeds at levels 3–6 | `results/still-scenes/crowds/facing-gate-repair/README.md` §2 | Confirmed |
| 28 | D12: 0.192 (0.170 with 4 more) against 0.242; above the 0.10 bar | `results/still-scenes/people-finder/early-detector/phase2/README.md` Tables 1.1, 1.5; recomputed 35/182, 31/182, 44/182 | Confirmed |
| 29 | D13: per-situation claims cannot reach p ≤ 0.05; E−R-sized effects need 18–28 records over 14 layouts | `HOW_TO_REPORT.md` | Confirmed (see flagged points) |
| 30 | D14: seeds repeat a row with small jitter | `results/still-scenes/overview/phase-notes/phase1d.md` §2.1 | Confirmed |
| 31 | D14: I46's 20 came from resampling 14 layouts | `results/still-scenes/overview/phase-notes/phase1.md` §5; `HOW_TO_REPORT.md` | Confirmed |
| 32 | D14: seeds 213–219 hold the same 14 rows | `results/still-scenes/people-finder/early-detector/phase2/README.md` §1 (each seed renders the same 14 ids) | Confirmed |
| 33 | D15: 18.3 against 11.9 ms; 215.1 against 127.0 ms | `results/still-scenes/learned-models/flow-model-pilot/README.md` Table 2.10, §6; `results/still-scenes/people-finder/early-detector/phase2/README.md` §1 | Numbers confirmed; wording corrected (C3) |
| 34 | D16: the 42 group-joining scenarios or the predictor's outputs (I51); ethics review for a click pilot (I35) | REPORT §7 "For the mentor" | Confirmed |
| 35 | D16: every label is the project's own rule | REPORT §8 "Circularity" | Confirmed |
| 36 | Seeds 200–212: decision, never for fitting, spent as I09's test | `results/still-scenes/people-finder/early-detector/phase2/README.md` §1, §3; REPORT "How to read" | Confirmed |

### §1 Primary metric

| # | Claim | Source | Result |
|---|---|---|---|
| 37 | Robot radius 0.30 m, person radius 0.25 m, heading within 45 deg | `configs/default.yaml` (folder); `METRIC_NOTE.md` | Confirmed |
| 38 | 252 = 14 layouts × 3 arms × 2 episodes × 3 trials; 18 records per layout | `results/still-scenes/overview/phase-notes/phase1.md` §5; `HOW_TO_REPORT.md` | Confirmed |
| 39 | Goal-step records 143, 141, 157 (441); 36 per watching cell; 549 | `METRIC_NOTE.md`; `results/still-scenes/overview/phase-notes/phase1d.md` §1 | Confirmed |
| 40 | 13 realistic goal-step layouts, none at `eval_04`; per cell 8, 11, 12; 2 watching rows | `results/still-scenes/confidence-signal/FROZEN.md` pre-freeze check; `results/still-scenes/overview/phase-notes/phase1d.md` §1 | Confirmed |
| 41 | 520 generator episodes on seeds 100–109 | `results/still-scenes/overview/phase-notes/phase1d.md` §2.2 | Confirmed |
| 42 | 22 templates among the development rows; members move a median 0.00–0.48 m | `results/still-scenes/overview/phase-notes/phase1d.md` §2.1 | Confirmed |
| 43 | Templates: watching 2 (`eval_14`, `eval_15`, seeds 0–2), I06 10, I09 14 | `FROZEN.md` (layouts); `results/still-scenes/crowds/README.md` §1; `results/still-scenes/people-finder/early-detector/phase2/README.md` §1 | Confirmed |
| 44 | s21 Qwen and s23 GLM and DeepSeek C people answers are cache copies | `results/still-scenes/overview/phase-notes/phase1.md` cautions | Confirmed |
| 45 | ICC 0.08–0.75; 252 records carry 18–107 independent ones | `results/still-scenes/overview/phase-notes/phase1.md` §5 | Confirmed |
| 46 | Exact sign-flip test by layout; t interval over 14 layout means; no binomial, McNemar or Wald | `HOW_TO_REPORT.md` | Confirmed |
| 47 | `eval_00`–`05` hold 1–6 goal-step records each | `results/still-scenes/overview/phase-notes/phase1d.md` note under Table 1.3 | Confirmed |
| 48 | The percentile bootstrap runs narrow with few clusters | `HOW_TO_REPORT.md`; REPORT §8 | Confirmed |
| 49 | MDD over 14 layouts 18–28 (E−R-like), 33–79 (C−E-like) | `HOW_TO_REPORT.md`; `mdd_table.txt` M3 | Confirmed |
| 50 | Paired live, one stage: 11–21 per batch | `HOW_TO_REPORT.md`; `mdd_table.txt` M2 | Confirmed |
| 51 | Two independent batches: 24–32; 22 / 18 / 16 with 2 / 3 / 4 batches | `HOW_TO_REPORT.md`; `mdd_table.txt` M1 (from Qwen's 7.93) | Confirmed |
| 52 | Floors 6 → 0.031; 7 → 0.016 (3 models); 8 → 0.0078 (6 models) | Recomputed: 2/2^k against 0.05/m | Confirmed (see flagged points) |
| 53 | Live margins of 1–3 records sit in a batch SD of 6–8; the stage-23 check | REPORT §8 "Run-to-run noise" | Confirmed |
| 54 | "m of k" bars on point estimates, as I04 and the probe do | `results/still-scenes/overview/phase-notes/phase1d.md` §1 criterion (d); `FROZEN.md` §7 | Confirmed |

### §2.1 Q1

| # | Claim | Source | Result |
|---|---|---|---|
| 55 | Binary log, Brier and region log density proper; mean ln(ε + S) is not | Standard reasoning, no project source | Consistent; not a source claim |
| 56 | ε = 1e-3, as I04 fitted | `results/still-scenes/overview/phase-notes/phase1d.md` §1 | Confirmed |
| 57 | V0 −1.660 = (132/549) × ln 1e-3 plus a negligible term | Recomputed (−1.6601; 132 = 549 − 417) | Confirmed |
| 58 | V0 −1.10 at ε 1e-2 and −2.21 at 1e-4 | Recomputed (−1.0997, −2.2144) | Confirmed |
| 59 | Lift_room in I04's form ΣS/Σρ_room and v0's Σ top-1/Σρ_room | `results/still-scenes/overview/phase-notes/phase1d.md` §1; `METRIC_NOTE.md` "Lift" | Confirmed |
| 60 | hit@k places at least 0.6 m apart | `results/still-scenes/overview/phase-notes/phase1d.md` §1 | Confirmed |
| 61 | Set success share = v0's P(success)@5 | `METRIC_NOTE.md` "Places" | Confirmed |
| 62 | Method code reads only `result.*` | REPORT §8 "Leakage" | Confirmed |
| 63 | Per-cell rates alone: +0.08 pooled | Recomputed (+0.077) | Confirmed |
| 64 | Per-cell rates alone: +0.35 at old Qwen | Recomputed (+0.325 with the 441-record rate) | Corrected (C1) |
| 65 | ECE 0.03–0.06 for a calibrated forecaster | Reviewer's simulation, labelled as such | Not checkable here; label is accurate |
| 66 | Cell AUROCs rest on 9, 45 and 48 failures | 143−134, 141−96, 157−109 | Confirmed |
| 67 | 34 of the 102 realistic failures are wrong targets | `FROZEN.md` exposure; `results/still-scenes/confidence-signal/README.md` §1 | Confirmed |
| 68 | Perfect ranker at 20% abstention: +6.3, +17.0, +17.4 | Recomputed | Confirmed |
| 69 | Binormal AUROC 0.70: +1.9, +6.0, +5.9, mean +4.6 | Recomputed (+1.87, +5.98, +5.89; mean 4.58, record-weighted 4.61) | Confirmed |
| 70 | Q1 bar (a): within 2 records is I04 (c)'s tolerance, one-sided | `results/still-scenes/overview/phase-notes/phase1d.md` §1 criterion (c) (a hit@5 tolerance, applied here to top-1) | Confirmed |
| 71 | I04's −1.660 and −1.452 hold for its 549 records | `results/still-scenes/overview/phase-notes/phase1d.md` Table 1.2 | Confirmed |
| 72 | Bar (a) is stricter than I04's (b) | I04 (b) = V1 above V3 pooled (`results/still-scenes/overview/phase-notes/phase1d.md` §1) | Confirmed |
| 73 | The any-flag baseline = any of the solver's three flags | `FROZEN.md` §4, S5a | Confirmed |
| 74 | S6 is a combined signal | `FROZEN.md` §4 | Confirmed |
| 75 | FROZEN.md sha256 `2a3a890a…cb4b3` | `FROZEN.sha256`; `sha256sum` of the file | Confirmed |
| 76 | Probe success: AUROC ≥ 0.70 in 2 of 3 cells and pooled above the any-flag baseline; kill pooled < 0.60; coverage 0.8 and 0.5, point values | `FROZEN.md` §5, §7; neither deviation note changes them | Confirmed |
| 77 | Qwen's cell has 9 failures; pooling rewards knowing the model (0.658) | `FROZEN.md`; `results/still-scenes/confidence-signal/README.md` §5 | Confirmed |
| 78 | The probe's README did not exist at revision time | README file time 18:05, after the 17:56 revision | Confirmed; dated note added (C7) |

### §2.2 Q2

| # | Claim | Source | Result |
|---|---|---|---|
| 79 | F(rec): the solver's selectable cells, 0.05 m grid, the record's own VLM people | `METRIC_NOTE.md` "Floor" | Confirmed |
| 80 | Generator floor: ground truth, connected to the robot's start, 0.10 m grid | `results/still-scenes/overview/phase-notes/phase1d.md` §2.2 | Confirmed |
| 81 | Region log density with 1% uniform mixed in | `results/still-scenes/overview/phase-notes/phase1d.md` §2.2 | Confirmed |
| 82 | ln 0.01 ≈ −4.6 nats | Recomputed (−4.605) | Confirmed |
| 83 | Band from the 5th to 95th percentile of training region-cell distances | `results/still-scenes/overview/phase-notes/phase1d.md` §2.2 | Confirmed |
| 84 | Sampler coverage: 64 samples, within 0.2 m | `results/still-scenes/learned-models/flow-model-pilot/README.md` §1 | Confirmed |
| 85 | Forbidden kinds `ahead_in_line`, `group_space`, `audience_view` | `METRIC_NOTE.md` | Confirmed |
| 86 | Uniform beats band by 0.131 and 2.919 nats | `results/still-scenes/overview/phase-notes/phase1d.md` Table 2.2, recomputed | Confirmed |
| 87 | Training subsample moved Δ about 0.9 nats (+1.955 against +2.843 to +2.906) | `results/still-scenes/overview/phase-notes/phase1d.md` Table 2.3 | Confirmed |
| 88 | Gate 2's rule: 2 of 3 left-out situations | `results/still-scenes/overview/phase-notes/phase1d.md` §2.2 | Confirmed |
| 89 | 3–5 templates per fold | Gate 1 per situation: lines 5, groups 5, audiences 3 (`results/still-scenes/overview/phase-notes/phase1d.md` Table 2.1) | Confirmed |
| 90 | The control gains at most about 0.096 nats over uniform at audiences | Recomputed from room share 0.0916: 0.0961 without the mix, 0.0952 with 1% | Confirmed |
| 91 | Group-ring mass V1 and V3 0.001; top place 0.0185 in audience view; uniform 0.0916; room shares lines 0.0016, groups 0.0030 | `results/still-scenes/overview/phase-notes/phase1d.md` Table 1.6 forbidden-mass lines | Confirmed |

### §2.3 Earlier results

| # | Claim | Source | Result |
|---|---|---|---|
| 92 | I14 v0: AR over the room 57% and 69%, above 50%; GLM's runners-up 99 against 104.0 and 107.0 | `results/still-scenes/overview/phase-notes/phase1.md` §4, §6 | Confirmed |
| 93 | I04: V1 −1.563 against V3 −1.452 (n 549); cell AUROCs 0.559, 0.303, 0.540 | `results/still-scenes/overview/phase-notes/phase1d.md` §1, §3 | Confirmed |
| 94 | C5 gate 2 against the stronger baseline: +1.823, −0.781, +1.195 | `results/still-scenes/overview/phase-notes/phase1d.md` Table 2.2 and post hoc "against uniform" (uniform stronger at lines and audiences, band at groups) | Confirmed |
| 95 | 2 of 3 in the verifier's draw too | `results/still-scenes/overview/phase-notes/phase1d.md` §2.2 (+2.712, −0.771, +0.868) | Confirmed |
| 96 | Δ positive in 0 of 100 conversation-group episodes | `results/still-scenes/overview/phase-notes/phase1d.md` §2.2 post hoc | Confirmed |
| 97 | Flow pilot: K1 22–30 points below P, K2 15–27 below C; K3 against V3 | `results/still-scenes/learned-models/flow-model-pilot/README.md` §5 | Confirmed |
| 98 | I06: 1.5% and 42.7% | `results/still-scenes/crowds/README.md` §2 | Confirmed |
| 99 | I09: 0.242, or 0.192 / 0.170, against 0.10 | `results/still-scenes/people-finder/early-detector/phase2/README.md` §1 | Confirmed |

### §2.4 Chance region

| # | Claim | Source | Result |
|---|---|---|---|
| 100 | Lift_room 8.78/8.90 matches mean S 0.749/0.760 | `results/still-scenes/overview/phase-notes/phase1d.md` Table 1.2; ratios 0.987 and 0.986 | Confirmed |
| 101 | The region is 1.6–3.8% of reachable floor | REPORT I14 | Confirmed |
| 102 | The disc leaves out about 90% of the watching region; top place outside it in 470 of 790 line records | `results/still-scenes/overview/phase-notes/phase1.md` §4 | Confirmed |
| 103 | AR recovery 96% → 57% and 33% → 69% | `results/still-scenes/overview/phase-notes/phase1.md` §4 | Confirmed |
| 104 | AF = top place plus 4 farthest-apart cells | `METRIC_NOTE.md` "Draws" | Confirmed |
| 105 | REPORT §7 suggests a region around the region or the top place | REPORT §7 "Metrics and reporting" | Confirmed |

### §3 Collapse

| # | Claim | Source | Result |
|---|---|---|---|
| 106 | I04's kill 2 wording | REPORT I04 | Confirmed |
| 107 | V1 above 0.9 in 312 of 549 (56.8%) | `results/still-scenes/overview/phase-notes/phase1d.md` Table 1.6 | Confirmed |
| 108 | V3 keeps more than 0.999 within 0.3 m in every record | `results/still-scenes/overview/phase-notes/phase1d.md` note under Table 1.6 | Confirmed |
| 109 | Top place succeeds in 417 of 549 | `results/still-scenes/overview/phase-notes/phase1d.md` Table 1.2 | Confirmed |
| 110 | A useful spread would need upstream uncertainty (untested) | `results/still-scenes/overview/phase-notes/phase1d.md` §5 | Confirmed |

### §4 Crowd labels

| # | Claim | Source | Result |
|---|---|---|---|
| 111 | L3 was not measured | REPORT §7; I06 measured L1 and L2 only | Confirmed |
| 112 | 199 variants on 10 base structures × 3 seeds | `results/still-scenes/crowds/README.md` cautions | Confirmed |
| 113 | Oracle C disagrees in 3 of 199, all `eval_11`, one per seed; L2 never empty | `results/still-scenes/crowds/README.md` §2 | Confirmed |
| 114 | Solver keeps 1.0 m; L2 needs 0.85 m | `results/still-scenes/crowds/README.md` §2 | Confirmed |
| 115 | Simulated people: 2.1–4.5%; DeepSeek 4.0% in the verifier's run (§11) | `results/still-scenes/crowds/README.md` §2, §11 | Confirmed |
| 116 | Reference stop fails L2 in 85 of 199 (42.7%), all 72 of levels 2–6 | `results/still-scenes/crowds/README.md` §2 | Confirmed |
| 117 | Label-validity bands are I06's lines | `results/still-scenes/crowds/README.md` §6 | Corrected (C2) |
| 118 | Affected: R's and E's marked points, labels for C5 | REPORT §7 | Confirmed |
| 119 | Levels 5–6: 0.6–0.8 m behind, facing within 30 deg | `results/still-scenes/crowds/README.md` §1; REPORT §7 | Confirmed |
| 120 | No rule succeeds at levels 3–6 (0.03–0.44 m); levels 4–6 left unjudged | `results/still-scenes/crowds/facing-gate-repair/README.md` §2 | Confirmed |

### §5 People perception

| # | Claim | Source | Result |
|---|---|---|---|
| 121 | The I03 people scorer is used unchanged | `results/still-scenes/people-finder/early-detector/phase2/README.md` §1 | Confirmed |
| 122 | The confirmation used every A1 bar but the runtime and best-VLM clauses | `results/still-scenes/people-finder/early-detector/phase2/README.md` §1 | Confirmed |
| 123 | Recall: greedy nearest-first one-to-one within 0.5 m; ≥ 0.95, kill < 0.90 | `results/still-scenes/people-finder/early-detector/phase2/README.md` §1, Table 1.1 | Confirmed |
| 124 | Line-member recall ≥ 0.95 | `results/still-scenes/people-finder/early-detector/phase2/README.md` Table 1.1 | Confirmed |
| 125 | False people: definition; ≤ 0.10, kill > 0.30 | `results/still-scenes/people-finder/early-detector/phase2/README.md` §1, Table 1.1 | Confirmed |
| 126 | Position error ≤ 0.15 m | `results/still-scenes/people-finder/early-detector/phase2/README.md` Table 1.1 | Confirmed |
| 127 | View classes and facing bars (25 deg, 0.85, 0.90, kill 0.70) | `results/still-scenes/people-finder/early-detector/phase2/README.md` §1, Table 1.1 | Confirmed |
| 128 | Facing kill 0.806 (Gemini 0.906 minus 10 points), or 0.70 with the school panel | `results/still-scenes/overview/phase-notes/phase1.md` §3; `results/still-scenes/overview/phase-notes/phase1c.md` §1 | Confirmed |
| 129 | Runtime: warm deployed path, median sidecar plus median project side ≤ 1.0 s | `results/still-scenes/overview/phase-notes/phase1c.md` §1 | Confirmed |
| 130 | Staff recall 69 of 78 | `results/still-scenes/people-finder/early-detector/phase2/README.md` Table 1.2 | Confirmed |
| 131 | VLM people unit: one answer per (arm, scene, trial), cache copies not counted | `results/still-scenes/overview/phase-notes/phase1.md` §3 | Confirmed |
| 132 | 44 false people; 9 within 0.5 m of hidden staff; 4 at 0.52–0.61 m; 35/182, 31/182 | `results/still-scenes/people-finder/early-detector/phase2/README.md` Table 1.5; recomputed | Confirmed |
| 133 | "A scorer question" (I09 §3, §4 question 4) | `results/still-scenes/people-finder/early-detector/phase2/README.md` §3, §4 | Confirmed |

### Appendix A Speed

| # | Claim | Source | Result |
|---|---|---|---|
| 134 | Qwen C, 143 records: 61.6, 61.2, 49.3, 53.2 and 47.4 s | REPORT I21 | Confirmed |
| 135 | Stage 23: 6 gateway jobs under the 85/min cap (§1.1, finding 8) | REPORT §1.1, finding 8 | Confirmed |
| 136 | C and E compared under the same per-stage thinking policy | REPORT I22 | Confirmed |
| 137 | I09: 55.6 ms warm (verifier 61.9); 215.1 ms un-warmed, every box ≥ 0.10 posed, shared machine (verifier 127.0) | `results/still-scenes/people-finder/early-detector/phase2/README.md` §1; `results/still-scenes/overview/phase-notes/phase1c.md` §1 | Confirmed |
| 138 | Flow pilot: 18.3 ms (verifier 11.9, idle GPU); raster 78.5 ms; floor 110.8 ms | `results/still-scenes/learned-models/flow-model-pilot/README.md` Table 2.10 | Confirmed |

### Appendix B Held-out data and seeds

| # | Claim | Source | Result |
|---|---|---|---|
| 139 | 68 development scenes: 42 realistic, 26 placeholder | `results/still-scenes/overview/phase-notes/phase1d.md` inputs | Confirmed |
| 140 | Tuning on the reporting sets needs fresh confirmation | REPORT §8 "Tuning on the reporting sets" | Confirmed |
| 141 | Seeds 100–109 fit I03, I11, I09 A1 and gate 2 | REPORT I46; `results/still-scenes/overview/phase-notes/phase1d.md` §2.2 | Confirmed; citation completed (C4) |
| 142 | Seeds 110–189 trained the pilot; 190–199 chose its checkpoint; training only | `results/still-scenes/learned-models/flow-model-pilot/README.md` §1, §7; REPORT I46 | Confirmed; citation completed (C5) |
| 143 | Seeds 200–212: 182 images of the 14 rows; never fit; spent; a second confirmation needs its own pre-registration, other fit data and fresh layouts | `results/still-scenes/people-finder/early-detector/phase2/README.md` §1, §3 | Confirmed |
| 144 | Seeds 213–219: 98 images, unrendered, the same 14 rows | `results/still-scenes/people-finder/early-detector/phase2/README.md` §1, §3 | Confirmed; citation completed (C6) |
| 145 | I46's sealed bank: seeds 3–99 and blind situations; final run about 6,000 calls | REPORT I46; §6 ("I46 final test: about 6,000") | Confirmed |
| 146 | `keep/heldout/seed0-2` for the final held-out test; `eval_16`–`19`, s18, s20_held spent; REPORT does not say whether that test and I46's run are one or two | REPORT §8 "Held-out discipline", disclosure (a); REPORT I46 | Confirmed |
| 147 | Noise models fitted on development runs only | REPORT §8 (d) | Confirmed |

## A2 update check (2026-10-18)

### Claims

| # | Where | Claim | Source | Result |
|---|---|---|---|---|
| A1 | Status | Updated with the probe's results (`A2_UPDATE.md`) | `A2_UPDATE.md` exists | Confirmed |
| A2 | Summary 4 | Confidence judged within cells and against the flags | Bars (c), (d) in the draft | Confirmed; agrees with the body |
| A3 | Glossary | Probe: `results/still-scenes/confidence-signal/README.md`, 18 Oct; VLM people against a detector's | README §1 | Confirmed |
| A4 | Glossary | P1: shift of C's place when the solver re-runs under I09 step A2's setting D | README §1 signals table; FROZEN §4 | Confirmed |
| A5 | Glossary | S2: people disagreement with the detector | README §1 (1 − F1) | Confirmed |
| A6 | Glossary | S4: disagreement among C's trials, two extra runs | README §1 | Confirmed |
| A7 | Glossary | S6z: zero-call combination including P1 and S2 | README §1 (P1, S2, S3) | Confirmed |
| A8 | Glossary | The flags: any of the solver's three flags | FROZEN §4 (S5a) | Confirmed |
| A9 | D4 | P1 is neither | README §0, §4 | Confirmed |
| A10 | D4 | On a fragile margin | README §0 cautions, §5 | Confirmed (see F2) |
| A11 | D4 | Cells 0.645, 0.738, 0.688; pooled 0.708 | README Table 3.2 | Confirmed |
| A12 | D4 | Reported S2 0.741, S6z 0.752, S4 0.774 | README §0, Table 3.2 | Confirmed; marked "reported" |
| A13 | D4 | Confirming needs records the probe did not use | README §0 cautions, §8 q1 | Confirmed |
| A14 | D7 | Revised bars: cells with ≥ 20 failures, paired win over the flags, abstention against the flags | Draft bars (c), (d) | Confirmed (see F1) |
| A15 | D7 | Valid only on records the probe did not use | Draft bars, last line | Confirmed |
| A16 | D7 | +4 stays a placeholder until the project rules on an abstain output | README §8 q3 | Confirmed |
| A17 | D7 | Model identity alone: pooled AUROC 0.658 | README §0, §5; `exposure.txt` Table A | Confirmed |
| A18 | D7 | Flags' pooled AUROC 0.595 | README Table 3.2 | Confirmed |
| A19 | D7 | Brier skill +0.08 against a pooled base rate | Earlier check #23 (+0.077) | Confirmed, unchanged |
| A20 | D7 | Flags gain +3.0 points at 20% abstention within cells | `summary.json` T5 S5a, record-weighted d80: +3.004 | Recomputed; confirmed |
| A21 | §1 | Paired draws, midrank AUROC, expected-value ties in selective accuracy | FROZEN §5 | Confirmed |
| A22 | §1 | P1's DeepSeek AUROC spans 0.670–0.706 by tie-breaking | README §5 ("Ties decide the margin") | Confirmed (see F2) |
| A23 | §2.1 (c) | Summed over counting cells; per-cell, right- and wrong-target AUROC beside the flags' | Draft bar (c) | Consistent; no number |
| A24 | §2.1 (d) | Base rate = what random abstention keeps on average | FROZEN §5; T5 random row equals base | Confirmed |
| A25 | §2.1 (d) | P1 +8.5 within cells, all three | T5: +1.97, +15.07, +8.60; weighted +8.519 | Recomputed; confirmed |
| A26 | §2.1 (d) | S2 +8.1 | T5: +5.08, +12.70, +6.81; weighted +8.130 | Recomputed; confirmed |
| A27 | §2.1 (d) | S6z +8.6 | T5: +3.67, +13.48, +8.60; weighted +8.560 | Recomputed; confirmed |
| A28 | §2.1 (d) | Flags +3.0; +4.6 over GLM and DeepSeek | T5: −0.28, +8.69, +0.89; +3.004 and +4.577 | Recomputed; confirmed |
| A29 | Bar (c) | Counting cell: ≥ 20 failures and ≥ 20 successes; a claim needs 2 | Rule set after results; labelled so | Confirmed as labelled |
| A30 | Bar (c) | Old Qwen's 9 failures gave P1 0.392–1.000 | README Table 3.2 | Confirmed |
| A31 | Bar (c) | Paired intervals against the flags ran ±0.13–0.18 | README §3 (P1 − flags [−0.088, +0.273], half-width 0.181); §10 / Table F (S2 − flags [−0.083, +0.182], 0.133) | Recomputed; confirmed (see F4) |
| A32 | Bar (c) | 400–440 records need a gap near 0.15–0.2 | Lower half-widths 0.154 (S2, 404) and 0.200 (P1, 441) | Recomputed; confirmed (see F4) |
| A33 | Bar (d) | Scope of an abstain output awaits the project | README §8 q3 | Confirmed |
| A34 | Bars, last line | Set before: 0.70, 0.60, 20% abstention, bars (a), (b) | 0.70 in `METRIC_LIST_v1_DRAFT.first.md` (17:41); 0.60 and 20% in CHANGELOG 1.5, 1.7 (17:56 revision, before the README at 18:05) | Confirmed (see F8) |
| A35 | §2.1 probe | FROZEN.md sha256 `2a3a890a…cb4b3` | `sha256sum` of `FROZEN.md` | Confirmed |
| A36 | §2.1 probe | P1 neither; cells; 2 of 3 needed; pooled 0.708 against 0.595 | README §0, §4 | Confirmed |
| A37 | §2.1 probe | Dropping one of 3 layouts, or trial 1 or 2 alone, makes it a success | README §5 (`eval_01`, `eval_11`, `eval_12`; trials 1 and 2) | Confirmed (see F2) |
| A38 | §2.1 probe | Reported, not judged: S2 0.741, S6z 0.752, S4 0.774 (two extra runs) | README §0, Table 3.2 | Confirmed |
| A39 | §2.1 probe | All designed knowing the repair's results | README §0 cautions; note under Table 3.2 | Confirmed |
| A40 | §2.1 probe | Post hoc (§10): 404 records the repair does not change | `exposure.txt` counts (b): 404, 66 failures | Confirmed |
| A41 | §2.1 probe | P1 0.557 on the 404 | `exposure.txt` Table B; README §5 | Confirmed |
| A42 | §2.1 probe | S2 0.694 [0.587, 0.743] | `exposure.txt` Table B | Confirmed |
| A43 | §2.1 probe | S2 minus the flags +0.071 [−0.083, +0.182] | `exposure.txt` Table F | Confirmed |
| A44 | §2.1 probe | Edge mostly on wrong targets: 0.716 against 0.566; right-target 0.671 against 0.683 | `exposure.txt` Table H | Confirmed. On right targets S2 is below the flags, so "mostly" understates; harmless |
| A45 | Differences table | Success → bar (c) with a paired win over the flags | Draft bar (c) | Consistent |
| A46 | Differences table | Qwen's cell has 9 failures | README Table 3.1 | Confirmed |
| A47 | Differences table | Model identity clears "pooled above the flags" (0.658 against 0.595) | README §0, §5 | Confirmed |
| A48 | Differences table | P1 − flags +0.112 [−0.088, +0.273] | README §3; `summary.json` T2 | Confirmed |
| A49 | Differences table | Kill → bar (c); abstention → bar (d), 50% secondary | Draft bars and (d) text | Consistent |
| A50 | §2.3 I04 | Stratified AUROC 0.447, recomputed by the probe | README Table 3.2 (S5b); `summary.json` 0.4469 | Confirmed (see F5) |
| A51 | §2.3 I04 | ε check still not computed | Earlier check #93; the probe did not compute it | Confirmed |

**Agreement.** D4 and the §2.1 paragraph give the same numbers. D7 matches bars (c) and (d), except F1. Summary item 4 matches the bars. The differences table matches the bars. `A2_UPDATE.md`'s word counts (6,548 and 6,881) match `wc -w`.

### Flagged, not changed

- **F1. D7 against bar (c).** D7 says "cells with ≥ 20 failures". Bar (c) says at least 20 failures *and* 20 successes. The success clause does not bind on the current cells (96 or more successes each). Suggest adding "and 20 successes" to D7.
- **F2. Post hoc readings not labelled.** The layout and trial drops (§2.1 paragraph) and the tie range 0.670–0.706 (§1) are the verifier's post hoc checks (README §5). The draft labels only the §10 numbers as post hoc. Suggest "post hoc (its §5)" in both places.
- **F3. "Development records" not said.** The §2.1 paragraph and the measured gains in (d) do not say these are development records, untested on unseen ones (README §0, §7). The bars' last line ("valid only on records it did not use") covers part of this. Suggest one word: "on its 441 development records".
- **F4. The planning note is approximate.** Its intervals are pooled one-ranking paired differences. Bar (c) uses the stratified difference over counting cells, which drops old Qwen (about 300 records on this set), so intervals would likely be wider. It is drafter's arithmetic and not labelled so.
- **F5. I04's 0.447 is over all three cells.** Under v1's counting cells (GLM, DeepSeek) the stratified value would be 0.433 (checker's arithmetic from `summary.json` T2). Kill either way.
- **F6. S6z on the 404 records.** D4 and the §2.1 paragraph list S6z 0.752 but not its post hoc fall to 0.635 [0.515, 0.721] on the 404 (flags 0.622; `exposure.txt` Table B). S2 falls less (0.694). For balance, suggest giving both.
- **F8. "Set before" list.** The interval clause of bar (d) predates the probe's results (it was in the earlier placeholder) but is in neither list. Minor.

**F1–F8 applied by the research lead (18 Oct)** as the checker suggested, with no number changed except the added F5, F6 and F7 values, which are the checker's arithmetic or come from `exposure.txt`. The draft before them is kept as `METRIC_LIST_v1_DRAFT.before-flags.md`.
