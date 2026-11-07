# I01 deliverables: re-run tables for the run
# Sets and definitions of the replay harness

**Question.** What are the sets, definitions and headline tables the replay analyses share?

**Result.** Three deliverables: solver-alone and people-blind standard-goal rows for six models, every score split into target found and placement given the right target, and R and E marked points replayed through the clearance snap. The old text below holds all tables.

**Files.**
- `README.md`: this page and the full tables (under Notes)

**Reproduce.** The tables come from `replay-harness/baselines/`, `replay-harness/split/` and `depth-fix/first-check/`. The scripts import the harness in `results/still-scenes/analysis-scripts/harness/` and read the development scenes from `data/dev_scenes/` (release dev-scenes-v1).

**Data.** `results/logs/still-runs.tar.gz` (unpacks to `runs/` at the repo root); development scenes in `data/dev_scenes/` (release dev-scenes-v1).

## Notes


1. The solver-alone row and the people-blind standard-goal row, for all six models.
2. Every score split into "target found" and "placement given the right target".
3. R and E marked points replayed through C's clearance snap (I34 Stage 0).

Every number was computed through the step-0 harness from recorded development runs, then recomputed by an independent verifier with its own code. The two agree record by record unless a row says otherwise. No model or network call was made and nothing was tuned.

RD = `results/2026-10-17-beyond-vlm`. Paths below are relative to RD. The ledger for the report's other claims is `RERUN.md`.

## Sets and shared definitions

- **Realistic arms**: real, staffed and plain, pooled. 84 records per run, 3 trials, so 252 records per model and method: nobody near 108, lines 90, groups 54.
- **New wording**: Qwen 3.8 low from s21; GLM-5.3, DeepSeek V4.1 Flash, Haiku 5.5, Sonnet 5.5 low and Gemini 3.8 low from s23.
- **Old wording**: Qwen and GLM R and E from s16b, C from s19; DeepSeek R, E and C from s20.
- **Methods**: R = marked point (cot); E = R plus 2 worked examples (fsc); C = people call, constraints call and solver (con).
- **Success** is the recorded `score.success`. Rescoring with the current package scorer gives the same verdict in every record checked (4,536 of 4,536 new-wording records by the verifier; 7,308 of 7,308 by the first check).
- **Situation names**: "nobody near" = no_one; "lines" = waiting_line; "groups" = conversation_group. No audience situation exists in these sets.

---

## Deliverable 1: solver alone and the people-blind standard goal

### Definitions

- **C**: recorded C success.
- **SA (solver alone)**: `H.resolve(rec, constraints=[])` on every C record that has a recorded target and a recorded people list (an empty list counts). It keeps the model's own target and people. The empty list reaches the solver as "no constraints" (checked, `check_empty.txt`).
- **SG (people-blind standard goal)**: `sgi.stages.goal.standard_goal` on the recorded target with the run's recorded thresholds, scored with `H.score_pose`. Computed for every record with a target. Recomputing it for the 963 standard-path records lands exactly on the recorded pose.
- **CA (come_after only)**: the solver with only the recorded `come_after` constraints kept, whatever status the solver gave them. A goal step with no `come_after` gets SA.
- Records with no goal step (standard path) keep their recorded verdict in SA and CA. New wording: Qwen 108, GLM 111, DeepSeek 101, Haiku 106, Sonnet 109, Gemini 117 such records, nearly all at nobody-near scenes (lines: GLM 2, Sonnet 1, Gemini 8; groups: GLM 1, Gemini 1).
- R and E are the recorded successes of the same stage, for reference.

### Table 1a. New wording, successes by model and situation

Nobody near (n = 108 per model)

| Model | R | E | C | SA | SG | CA |
|---|---|---|---|---|---|---|
| Qwen | 97 | 97 | 97 | 97 | 97 | 97 |
| GLM | 83 | 83 | 83 | 83 | 83 | 83 |
| DeepSeek | 81 | 82 | 83 | 85 | 85 | 85 |
| Haiku | 106 | 106 | 108 | 108 | 108 | 108 |
| Sonnet | 108 | 108 | 108 | 108 | 108 | 108 |
| Gemini | 104 | 104 | 104 | 104 | 104 | 104 |

Lines (n = 90 per model)

| Model | R | E | C | SA | SG | CA |
|---|---|---|---|---|---|---|
| Qwen | 44 | 61 | 85 | 0 | 0 | 85 |
| GLM | 15 | 10 | 39 | 0 | 0 | 39 |
| DeepSeek | 13 | 21 | 68 | 0 | 0 | 72 |
| Haiku | 6 | 10 | 79 | 0 | 0 | 78 |
| Sonnet | 25 | 27 | 87 | 0 | 0 | 87 |
| Gemini | 32 | 35 | 56 | 0 | 0 | 56 |

Groups (n = 54 per model)

| Model | R | E | C | SA | SG | CA |
|---|---|---|---|---|---|---|
| Qwen | 38 | 38 | 46 | 46 | 42 | 46 |
| GLM | 34 | 30 | 48 | 49 | 41 | 49 |
| DeepSeek | 26 | 23 | 34 | 44 | 38 | 42 |
| Haiku | 39 | 32 | 53 | 53 | 48 | 53 |
| Sonnet | 45 | 43 | 47 | 47 | 43 | 47 |
| Gemini | 28 | 34 | 54 | 54 | 40 | 54 |

All situations (n = 252 per model)

| Model | R | E | C | SA | SG | CA |
|---|---|---|---|---|---|---|
| Qwen | 179 | 196 | 228 | 143 | 139 | 228 |
| GLM | 132 | 123 | 170 | 132 | 124 | 171 |
| DeepSeek | 120 | 126 | 185 | 129 | 123 | 199 |
| Haiku | 151 | 148 | 240 | 161 | 156 | 239 |
| Sonnet | 178 | 178 | 242 | 155 | 151 | 242 |
| Gemini | 164 | 173 | 214 | 158 | 144 | 214 |

What it shows:

- At nobody-near scenes SA = SG = C, except DeepSeek (85 vs 83): the solver alone fixes 2 of DeepSeek's 7 nobody-near goal steps where it wrongly saw people.
- At lines, both people-blind baselines score 0 of 90 for every model. C's line success comes from `come_after`: CA equals C for 4 of 6 models, beats it by 4 for DeepSeek and loses 1 for Haiku.
- At groups, SA ≥ C for all six models (SA − C = 0 / +1 / +10 / 0 / 0 / 0). Only DeepSeek gains from dropping the constraints.
- Pooled, SA and SG sit below C and below R and E, because they lose every line episode.

### Table 1b. Groups, goal-step records only

| Model | Goal steps | C | SA | SG | CA |
|---|---|---|---|---|---|
| Qwen | 54 | 46 | 46 | 42 | 46 |
| GLM | 53 | 47 | 48 | 40 | 48 |
| DeepSeek | 54 | 34 | 44 | 38 | 42 |
| Haiku | 54 | 53 | 53 | 48 | 53 |
| Sonnet | 54 | 47 | 47 | 43 | 47 |
| Gemini | 53 | 53 | 53 | 39 | 53 |

### Table 1c. Groups by layout (six models pooled, new wording)

| Layout | n | SG | SA | C |
|---|---|---|---|---|
| eval_10 (group of 2) | 108 | 106 | 106 | 101 |
| eval_11 (group of 3) | 108 | 106 | 107 | 101 |
| eval_12 (group of 4) | 108 | 40 | 80 | 80 |

SG per model at eval_10 + eval_11 (of 36) / eval_12 (of 18): Qwen 36 / 6, GLM 34 / 7, DeepSeek 34 / 4, Haiku 36 / 12, Sonnet 36 / 7, Gemini 36 / 4. The reference flag `reference_people_matter` is False at every eval_10 and eval_11 group episode in all three arms, and True at every eval_12 and every line episode. That flag is defined as "the truth standard goal fails", so it restates this table rather than explaining it.

### Table 1d. Lines, the come_after subset

| Model | Goal steps | State a come_after | C on those | CA on those | Right target among goal steps | C given right target and a come_after |
|---|---|---|---|---|---|---|
| Qwen | 90 | 90 | 85 | 85 | 87 | 85 of 87 |
| GLM | 88 | 76 | 39 | 39 | 71 | 39 of 66 |
| DeepSeek | 90 | 88 | 68 | 72 | 85 | 68 of 84 (CA 72) |
| Haiku | 90 | 80 | 78 | 78 | 90 | 78 of 80 |
| Sonnet | 89 | 87 | 87 | 87 | 89 | 87 of 87 |
| Gemini | 82 | 82 | 56 | 56 | 56 | 56 of 56 |

### Table 1e. Old wording

| Model | Situation | n | R | E | C | SA | SG | CA |
|---|---|---|---|---|---|---|---|---|
| Qwen | nobody near | 108 | 100 | 100 | 100 | 100 | 100 | 100 |
| Qwen | lines | 90 | 36 | 70 | 86 | 0 | 0 | 86 |
| Qwen | groups | 54 | 39 | 44 | 48 | 47 | 42 | 47 |
| Qwen | all | 252 | 175 | 214 | 234 | 147 | 142 | 233 |
| GLM | nobody near | 108 | 90 | 91 | 93 | 93 | 93 | 93 |
| GLM | lines | 90 | 13 | 18 | 42 | 0 | 0 | 42 |
| GLM | groups | 54 | 26 | 36 | 51 | 52 | 44 | 52 |
| GLM | all | 252 | 129 | 145 | 186 | 145 | 137 | 187 |
| DeepSeek | nobody near | 108 | 80 | 75 | 80 | 83 | 83 | 81 |
| DeepSeek | lines | 90 | 12 | 22 | 68 | 0 | 0 | 72 |
| DeepSeek | groups | 54 | 21 | 23 | 34 | 45 | 35 | 43 |
| DeepSeek | all | 252 | 113 | 120 | 182 | 128 | 118 | 196 |

GLM has 4 line records with no target (no_target); they keep their recorded failure in SG too. At the old wording SA − C at groups is −1 / +1 / +11.

### Checks against the figures (REPORT 1.1, finding 8)
| run's figure | Re-run | Result |
|---|---|---|
| Totals R / E / C: Qwen 179/196/228; GLM 132/123/170; DeepSeek 120/126/185; Haiku 151/148/240; Sonnet 178/178/242; Gemini 164/173/214 | identical, 18 of 18 | agree |
| Solver alone at groups 46/48/44/53/47/53 of 54 | 46/49/44/53/47/54 of 54 (all records); 46/48/44/53/47/53 over goal steps (n 54/53/54/54/54/53) | differ: denominator |
| C at groups 46/47/34/53/47/53 | 46/48/34/53/47/54 of 54; 46/47/34/53/47/53 over goal steps | differ: denominator |
| People-blind goal at groups 42/41/38/48/43/40 of 54 | 42/41/38/48/43/40 of 54 | agree |
| People-blind goal passes 38–48 of 54 at groups | 38–48 | agree |
| "because the reference says people do not matter at eval_10 and eval_11" | flag False at all eval_10/11 group episodes, True at eval_12; SG 212 of 216 vs 40 of 108 | agree (but circular, see Table 1c) |
| Solver with no constraints at lines: 0 of 90 | 0 of 90 for all six | agree |
| With come_after: 87 of 87 (Sonnet), 78 of 80 (Haiku), 85 of 90 (Qwen) | 87 of 87, 78 of 80, 85 of 90 | agree (denominator = records stating a come_after) |
| Rescore reproduces every verdict; solver replay 860 of 860 C poses | 4,536 of 4,536; 860 of 860 (moved 0 m) | agree |
| Region table, adopted row: C − max(R, E) = +32/+38/+59/+89/+64/+41 | same, from the totals above | agree |
| With the band's far edge at 1.0 m, C trails max(R, E) for 5 of 6 | not re-run | not checked |

---

## Deliverable 2: target found vs placement given the right target

### Definitions

- **Found** = `score.target_correct` (a not-ok result counts as not found; there are none at the new wording).
- **Success given found** = successes among records with the right target (a success always has the right target).
- **Failure split**, in order: wrong target; right target but outside the region; in the region but unreachable; reachable but heading wrong. A pose both outside the region and unreachable counts as outside.
- Within one stage the target is a single draw shared by R, E and C: the target point and `target_correct` are identical in 252 of 252 keys for all six models (and for the old-wording s16b + s19 pairing). E and C reuse R's plan and context calls from the cache. So "found" cannot differ by method, and R/E/C differences are paired comparisons of placement.

### Table 2a. Target found, new wording (same for R, E and C)

| Model | Nobody near (of 108) | Lines (of 90) | Groups (of 54) | All (of 252) |
|---|---|---|---|---|
| Qwen | 107 | 87 | 46 | 240 (95.2%) |
| GLM | 98 | 72 | 49 | 219 (86.9%) |
| DeepSeek | 99 | 85 | 44 | 228 (90.5%) |
| Haiku | 108 | 90 | 53 | 251 (99.6%) |
| Sonnet | 108 | 90 | 47 | 245 (97.2%) |
| Gemini | 108 | 56 | 54 | 218 (86.5%) |

### Table 2b. Success given the right target, new wording

| Model | Method | Nobody near | Lines | Groups | All |
|---|---|---|---|---|---|
| Qwen | R | 97/107 | 44/87 (51%) | 38/46 | 179/240 (74.6%) |
| Qwen | E | 97/107 | 61/87 (70%) | 38/46 | 196/240 (81.7%) |
| Qwen | C | 97/107 | 85/87 (98%) | 46/46 | 228/240 (95.0%) |
| GLM | R | 83/98 | 15/72 (21%) | 34/49 | 132/219 (60.3%) |
| GLM | E | 83/98 | 10/72 (14%) | 30/49 | 123/219 (56.2%) |
| GLM | C | 83/98 | 39/72 (54%) | 48/49 | 170/219 (77.6%) |
| DeepSeek | R | 81/99 | 13/85 (15%) | 26/44 | 120/228 (52.6%) |
| DeepSeek | E | 82/99 | 21/85 (25%) | 23/44 | 126/228 (55.3%) |
| DeepSeek | C | 83/99 | 68/85 (80%) | 34/44 | 185/228 (81.1%) |
| Haiku | R | 106/108 | 6/90 (7%) | 39/53 | 151/251 (60.2%) |
| Haiku | E | 106/108 | 10/90 (11%) | 32/53 | 148/251 (59.0%) |
| Haiku | C | 108/108 | 79/90 (88%) | 53/53 | 240/251 (95.6%) |
| Sonnet | R | 108/108 | 25/90 (28%) | 45/47 | 178/245 (72.7%) |
| Sonnet | E | 108/108 | 27/90 (30%) | 43/47 | 178/245 (72.7%) |
| Sonnet | C | 108/108 | 87/90 (97%) | 47/47 | 242/245 (98.8%) |
| Gemini | R | 104/108 | 32/56 (57%) | 28/54 | 164/218 (75.2%) |
| Gemini | E | 104/108 | 35/56 (63%) | 34/54 | 173/218 (79.4%) |
| Gemini | C | 104/108 | 56/56 (100%) | 54/54 | 214/218 (98.2%) |

At the nobody-near scenes every method produces the same standard goal from the target point (identical poses in 108 of 108 keys for Qwen, GLM, Sonnet and Gemini), so the three methods tie there except where DeepSeek and Haiku marked a few points.

### Table 2c. Failure split, new wording, all situations

| Model | Method | Failures | Wrong target | Outside region | Unreachable | Heading |
|---|---|---|---|---|---|---|
| Qwen | R | 73 | 12 | 54 | 7 | 0 |
| Qwen | E | 56 | 12 | 36 | 8 | 0 |
| Qwen | C | 24 | 12 | 11 | 1 | 0 |
| GLM | R | 120 | 33 | 84 | 3 | 0 |
| GLM | E | 129 | 33 | 91 | 5 | 0 |
| GLM | C | 82 | 33 | 48 | 1 | 0 |
| DeepSeek | R | 132 | 24 | 107 | 1 | 0 |
| DeepSeek | E | 126 | 24 | 99 | 3 | 0 |
| DeepSeek | C | 67 | 24 | 39 | 3 | 1 |
| Haiku | R | 101 | 1 | 100 | 0 | 0 |
| Haiku | E | 104 | 1 | 101 | 2 | 0 |
| Haiku | C | 12 | 1 | 11 | 0 | 0 |
| Sonnet | R | 74 | 7 | 66 | 1 | 0 |
| Sonnet | E | 74 | 7 | 65 | 2 | 0 |
| Sonnet | C | 10 | 7 | 3 | 0 | 0 |
| Gemini | R | 88 | 34 | 40 | 14 | 0 |
| Gemini | E | 79 | 34 | 35 | 10 | 0 |
| Gemini | C | 38 | 34 | 4 | 0 | 0 |

On R and E, failures outside the region dominate for every model. On C, wrong targets are 50% (Qwen), 70% (Sonnet) and 89% (Gemini) of what is left. Heading is the cause once in all 4,536 records. The per-situation split is TABLE 1 of `results/still-scenes/replay-harness/split/split_out.txt`.

Where the wrong targets are (R records, shared by E and C): Sonnet 7 of 7 at eval_12. Qwen 12: 8 at eval_12, 3 at eval_09 (line of 6), 1 at eval_00. DeepSeek 24: 9 at eval_12, 7 at eval_04 (parcel), 8 elsewhere. Haiku 1, at eval_12. GLM 33: 11 at eval_06 (line of 1), 8 at eval_04, the rest spread. Gemini 34 of 34 at lines (eval_06, 08, 09 and 13 in every arm; eval_07 in staffed only).

### Table 2d. C gap to Sonnet (242), new wording

Gap = people-scene gap + nobody-near gap; the nobody-near part is split into wrong target and placement with the right target.

| Model | C | Gap | People scenes | Nobody near | of which wrong target | of which placement |
|---|---|---|---|---|---|---|
| Qwen | 228 | −14 | −3 | −11 | −1 | −10 |
| GLM | 170 | −72 | −47 | −25 | −10 | −15 |
| DeepSeek | 185 | −57 | −32 | −25 | −9 | −16 |
| Haiku | 240 | −2 | −2 | 0 | 0 | 0 |
| Gemini | 214 | −28 | −24 | −4 | 0 | −4 |

Details for the top tier: Qwen's 10 right-target nobody-near misses are standard goals built from its located point that land 0.055–0.587 m outside the region (9) or in the region but unreachable (1). Haiku's 2 are placement at the lines (11 outside the region against Sonnet's 3), offset by 6 fewer wrong targets at groups. Gemini's people-scene gap is −31 at lines (34 wrong targets) and +7 at groups; its 4 nobody-near misses lie 0.033–0.043 m outside the region.

### Table 2e. Old wording

| Model | Method | Found: nobody / lines / groups / all | Nobody near | Lines | Groups | All |
|---|---|---|---|---|---|---|
| Qwen | R | 108 / 88 / 49 / 245 | 100/108 | 36/88 (41%) | 39/49 | 175/245 |
| Qwen | E | same | 100/108 | 70/88 (80%) | 44/49 | 214/245 |
| Qwen | C | same | 100/108 | 86/88 (98%) | 48/49 | 234/245 |
| GLM | R | 99 / 73 / 52 / 224 | 90/99 | 13/73 (18%) | 26/52 | 129/224 |
| GLM | E | same | 91/99 | 18/73 (25%) | 36/52 | 145/224 |
| GLM | C | same | 93/99 | 42/73 (58%) | 51/52 | 186/224 |
| DeepSeek | R | 100 / 84 / 46 / 230 | 80/100 | 12/84 (14%) | 21/46 | 113/230 |
| DeepSeek | E | same | 75/100 | 22/84 (26%) | 23/46 | 120/230 |
| DeepSeek | C | same | 80/100 | 68/84 (81%) | 34/46 | 182/230 |

GLM's 4 not-ok records are at lines, the same 4 keys in R, E and C. C at the people scenes: Qwen 134, GLM 93, DeepSeek 102 of 144.

### Table 2f. Qwen pools across stages

| Pool | Records | Successes | Found | Success given found | Lines given found |
|---|---|---|---|---|---|
| R, 9 draws (s16b + s22 trials 3–5 + s21) | 756 | 542 | 728 | 74.5% | 130/261 (49.8%) |
| E, 9 draws (same) | 756 | 614 | 728 | 84.3% | 195/261 (74.7%) |
| C, 6 draws (s19 + s21) | 504 | 462 | 485 | 95.3% | 171/175 (97.7%) |

### Checks against the figures
| run's figure | Re-run | Result |
|---|---|---|
| Given the right target, R succeeds at lines 57% (Gemini), 51% (Qwen), 28% (Sonnet), 21% (GLM), 15% (DeepSeek), 7% (Haiku) | 32/56, 44/87, 25/90, 15/72, 13/85, 6/90 | agree |
| On C, the people scenes give Sonnet 134, Haiku 132, Qwen 131 of 144 | 134, 132, 131 | agree |
| "the gaps in the totals come from target finding at the nobody-near scenes" | 1 of 44 gap records is a wrong target at a nobody-near scene (Table 2d) | differ (corrected; the first check called it refuted) |
| E gains nothing within a single batch (pooled +3); Qwen's 9-trial pool gives +24 | s23 E − R: −9 / +6 / −3 / 0 / +9 = +3; Qwen pool +24 per 252 | agree |
| Central estimates R 180.7, E 204.7 of 252 (section 8) | 542/3 = 180.7, 614/3 = 204.7 | agree |
| Strip: of in-region, right-heading, unreachable R/E poses, Gemini 22 of 24 and Qwen 11 of 15 overlap the body by up to 5 cm; the rest at 0.00–0.02 | Gemini R 13 + E 9 = 22 of 14 + 10; Qwen R 5 + E 6 = 11 of 7 + 8 (the 15 includes one R and one E at nobody-near scenes); the rest +0.002 to +0.015 m | agree |
| The 0.55 m band start changes 12 R/E verdicts | not re-run | not checked |

---

## Deliverable 3: R and E points through C's clearance snap

### Definitions

- **Snap rule**: C's own floor model (`sgi.solver.floor`). C may stop only on a selectable 5 cm cell: seen as floor, not within 0.40 m of a raised depth point, and joined to the robot through passable cells. A marked point (source direct_marking) already on a selectable cell stays; otherwise it moves to the nearest selectable cell centre. Heading faces the target (`sgi.stages.goal._poses`, which reproduces all 1,720 recorded headings).
- **snapD**: the floor with no people list (depth only, zero extra calls).
- **snapP**: plus the person discs (0.55 m) from the same model's C record for the same arm, scene, episode and trial. It costs C's people call.
- Only marked points are snapped. Standard goals (context said people do not matter) are left alone, as C also uses the standard goal there.
- **Classes of a failed pose**: near = outside the region by at most 0.3 m; mid = 0.3–1.0 m; far = over 1.0 m; in-region = inside but unreachable or heading wrong. "Overlaps a body" = `min_body_gap_m` < 0 (robot disc within 0.55 m of a person's centre).
- **Fixed** = success after the snap. **Broken** = a success that fails after the snap: **0 in every cell**.

### Table 3a. Right-target marked-point failures and what the snap fixes

Each cell: marked failures with the right target → fixed by snapD (fixed by snapP). Standard-goal failures are counted separately and never snapped.

| Model | Method | Nobody near | Lines | Groups | Successes before → snapD (snapP) | C |
|---|---|---|---|---|---|---|
| Qwen | R | 0 marked (10 standard) | 43 → 7 (7) | 8 → 4 (4) | 179 → 190 (190) | 228 |
| Qwen | E | 0 marked (10 standard) | 26 → 7 (8) | 8 → 4 (4) | 196 → 207 (208) | 228 |
| GLM | R | 0 marked (15 standard) | 56 → 0 (0), +1 standard | 15 → 8 (8) | 132 → 140 (140) | 170 |
| GLM | E | 0 marked (15 standard) | 61 → 2 (2), +1 standard | 19 → 12 (14) | 123 → 137 (139) | 170 |
| DeepSeek | R | 5 → 0 (0), +13 standard | 72 → 6 (8) | 18 → 3 (5) | 120 → 129 (133) | 185 |
| DeepSeek | E | 4 → 0 (0), +13 standard | 64 → 9 (11) | 21 → 12 (12) | 126 → 147 (149) | 185 |
| Haiku | R | 2 → 0 (0) | 84 → 0 (0) | 14 → 8 (8) | 151 → 159 (159) | 240 |
| Haiku | E | 2 → 0 (0) | 80 → 8 (8) | 21 → 12 (12) | 148 → 168 (168) | 240 |
| Sonnet | R | 0 | 64 → 0 (0), +1 standard | 2 → 1 (1) | 178 → 179 (179) | 242 |
| Sonnet | E | 0 | 62 → 0 (0), +1 standard | 4 → 3 (3) | 178 → 181 (181) | 242 |
| Gemini | R | 0 marked (4 standard) | 24 → 17 (17) | 26 → 19 (22) | 164 → 200 (203) | 214 |
| Gemini | E | 0 marked (4 standard) | 21 → 14 (16) | 20 → 12 (16) | 173 → 199 (205) | 214 |

Totals of fixes: snapD 168 (Qwen 22, GLM 22, DeepSeek 30, Haiku 28, Sonnet 4, Gemini 62); snapP 186 (23, 24, 36, 28, 4, 71). The snap moved 10–31 successes per model (Qwen 21, GLM 18, DeepSeek 10, Haiku 14, Sonnet 31, Gemini 28) without breaking any.

What it shows:

- The snap closes most of the gap to C only for Gemini (R 164 → 200 against C's 214).
- At groups it fixes 64–65% of near misses (snapD) and 71–72% with C's people.
- At lines it fixes about a third of near misses and does almost nothing for Sonnet (0 of 126), GLM (2 of 117) and Haiku R (0 of 84).

### Table 3b. Line failures that survive snapD (R and E together)

| Model | Still failing | Near | Mid | Far | In region | Behind the last member | Notes |
|---|---|---|---|---|---|---|---|
| Qwen | 55 | 18 | 23 | 13 | 1 | 18 | none overlaps a body |
| GLM | 115 | 26 | 45 | 44 | 0 | 2 | 110 of the 117 line failures lie more than 0.3 m to the camera's side of the line axis |
| DeepSeek | 121 | 39 | 53 | 25 | 4 | 26 | 12 still overlap a body (5 with snapP) |
| Haiku | 156 | 2 | 48 | 106 | 0 | 0 | mostly at the counter end, far |
| Sonnet | 126 | 51 | 70 | 5 | 0 | 3 | all 126 on the camera side, on free floor; only 8 were moved |
| Gemini | 14 | 8 | 2 | 1 | 3 | 4 | 2 overlap a body |

With the looser "within 0.3 m of a body disc" (`min_body_gap_m` < 0.3), the surviving line poses near a body are Qwen 5, GLM 36, DeepSeek 53, Haiku 33, Sonnet 0, Gemini 6.

### Table 3c. The I34 Stage 0 gate

Gate: the snap fixes under 60% of near misses, or far misses at lines are at least 15%. Kill: the snap recovers at least 70% of near misses. Near miss = near or in-region before the snap.

| Reading | Near misses fixed, snapD | snapP | Far share of line failures |
|---|---|---|---|
| All six models, R, lines and groups together | 71 of 159 (45%) | 49% | 116 of 343 (34%) |
| All six models, E, lines and groups together | 92 of 202 (46%) | 50% | 93 of 314 (30%) |
| GLM E, together | 14 of 27 (52%) | 56% | 26 of 61 (43%) |
| DeepSeek E, together | 18 of 44 (41%) | 45% | 10 of 64 (16%) |
| All six, R, lines only / groups only | 30 of 95 (32%) / 41 of 64 (64%) | 34% / 72% | |
| All six, E, lines only / groups only | 38 of 119 (32%) / 54 of 83 (65%) | 36% / 71% | |
| GLM E, groups only (s23) | 12 of 16 (75%) | 81% | |
| DeepSeek E, groups only (s23) | 11 of 15 (73%) | 73% | |
| Old E sets (s16b/s20), groups only: Qwen / GLM / DeepSeek | 3 of 4 / 8 of 15 (53%) / 10 of 15 (67%) | | |

Reading: as I34 words it (no split by situation), **the gate opens and the kill is not met** for every reading. Taken alone, groups at s23 reach the kill line for GLM and DeepSeek E (75%, 73%), but not on the old E sets where I34's evidence came from. Per model and method values are in T4 of `results/still-scenes/depth-fix/first-check/snap_tables.txt`.

### The 0.50–0.55 m strip

### Checks against the figures (mechanisms, REPORT 1.1 finding 8) and I34
| Figure | Re-run | Result |
|---|---|---|
| Sonnet's R point lands a median 54 px low, 1–1.7 m toward the camera | n 89: 54.6 px below the region's representative point; −1.35 m (IQR −1.61 to −1.23). Failures (n 64): 65 px, −1.47 m, all beside the line on the camera side | agree ("its words say behind" not checked) |
| Haiku picks the counter or the front of the line | 80 of 84 R line failures project onto the counter end (75) or the front half (5); E 63 of 80 | agree |
| Gemini picks the right spot but stands too close (median 0.61 m) | R 0.613 m (n 56), E 0.621; failures 0.53 m, successes 0.65–0.68 m | agree |
| The low and too-close marked points support I34's Stage 0 snap control | too close (Gemini): snap fixes 62 of 99; low/beside (Sonnet, Haiku): 0 of 126 and 8 of 164 at lines | partly |
| I34: failures of E at groups are mostly near misses that land on a body (old wording) | 28 of 44 (64%) near and within 0.3 m of a body; 25 of 44 overlap | agree |
| I34: failed marked points at lines lie a median 0.34, 0.88, 0.35 m from the region | E failures (s16b/s20): 0.34 (n 18), 0.88 (n 55), 0.35 (n 62); R gives 0.51, 0.60, 0.74 | agree (these are E's) |
| New wording, E group failures near and within 0.3 m of a body | 67 of 93 (72%); snapD fixes 54 of them (snapP 59) | new |
Verifier corrections to the first check here: GLM's camera-side count is 110 of 117 (not 107), and the gate reading above replaces the first check's "kill met at groups for GLM and DeepSeek E".

---

## Caveats

- **Cached people answers.** The s23 GLM and DeepSeek C people answers are cache copies of s19 and s20 (134 of 134 and 144 of 144 identical where both have a list). The s21 Qwen C people answers are copies of s19 too (143 of 144 identical; 144 of 144 cached). So C, SA and CA at the people scenes for Qwen, GLM and DeepSeek reuse old people draws, and the old- and new-wording rows for these models are not independent. snapP for GLM and DeepSeek uses the same copies.
- **Shared targets.** Within a stage, R, E and C share one target draw (Deliverable 2 definitions). Within a stage, the C people answer is also one per (arm, scene, trial), shared by both episodes (Qwen 72 of 72 group answers, Haiku 72 of 72, Sonnet 71 of 71, Gemini 63 of 63). Treating C people-scene episodes as independent overcounts.
- **s22 duplicates.** s22 trials 0–2 are exact copies of s16b (252 of 252 cot and fsc). They were not used. Only s22 trials 3–5 enter the Qwen pool, which also mixes old wording (s16b, s22) with new (s21).
- **Stages.** Old-wording R and E for Qwen and GLM come from s16b, a different stage from s19's C. Every s19 call is a cache hit.
- **What SA is.** SA keeps the target and the people the model wrote. It is not "the solver with true people", and the people were written for a run with constraints.
- **Denominators.** For the group rows, use all 54 records with recorded verdicts kept (SG and the stage totals already do). Over goal steps only, GLM and Gemini have 53.
- **Noise.** These are replays of recorded answers, so the replay adds no sampling noise. Each 252-record cell still carries one 3-trial batch's noise (SD about 6–8 records).
- **In-sample.** The context prompt was chosen on these sets by C's success.
- **Not part of this run.** I01's oracle repairs (R1–R6), VLM facing accuracy (A0), the band-at-1.0 m comparison and the other region-variant rows.

## Scripts and outputs

Deliverable 1:
- `results/still-scenes/replay-harness/baselines/i01_baselines.py`, `tables.py`, `check_empty.py`; outputs `tables.txt`, `records.jsonl`, `re_records.jsonl`, `check_empty.txt`, `run.log`.

Deliverable 2:
- `results/still-scenes/replay-harness/split/split.py`, `inspect_split.py`, `inspect2.py`, `tally.py`; outputs `split_out.txt` (TABLES 1–3 and claim checks), `inspect_out.txt`, `inspect2_out.txt`, `tally_out.txt`.
- Verifier: `results/still-scenes/replay-harness/split/verify/v_split.py`, `v_extra.py`, `v_cache.py`, `v_cache2.py`, `v_rescore.py` (with their `_out.txt` files).

Deliverable 3:
- `results/still-scenes/depth-fix/first-check/snap.py`, `summarize.py`, `n1n2.py`, `n1_s23.py`; outputs `snap_tables.txt` (T1–T7), `snap_records.jsonl`, `n1n2.txt`, `n1_s23.txt`.
- Verifier: `results/still-scenes/depth-fix/first-check/verify/v1_n1n2.py`, `v2_snap.py`, `v3_summary.py`, `v4_extra.py`, `v5_more.py` (with their `.txt` outputs and `v2_snap_records.jsonl`).
