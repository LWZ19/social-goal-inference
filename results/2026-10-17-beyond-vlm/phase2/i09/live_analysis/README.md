# I09 live check: the offline analysis (`live_analysis/`)

`analyse.py` is the analysis that `live/FROZEN.md` sections 5–6 say runs after the live run of arm D-live. It was
written, tested with zero calls, and hashed on 18 Oct before any live run folder existed.

- **Hash.** `ANALYSE.sha256` holds the sha256 of `analyse.py` (`45b83a6e70577acb…2566e`), `make_synthetic.py` and
  `run_tests.py`. They were hashed at 2026-10-18T05:42:50+0800.
- **Before hashing.** At 05:42:50 `live/runs/` did not exist (`test -e` false, `ls` "No such file or directory"). It
  was still absent right after hashing and after the tests were re-run on the hashed code (`ORDER.txt`).
- **Reads.** The recorded runs only through the step-0 harness. Nothing under `live/` was written.

## What it does

1. **FROZEN.md.** It checks `FROZEN.md` against `FROZEN.sha256` and its own pinned hash. It parses from the note the
   baselines (section 4), the thresholds (section 6), the trial spread (section 6), the recorded goal steps
   (section 2) and the detector time (section 5.6). Each must equal the constant in the script.
2. **Baselines, derived again.** If any differs from FROZEN, it stops with exit 2.
   - Recorded C, from the six recorded runs through the harness: t0 in all, per situation and per trial, plus the
     goal steps.
   - The D replay, from `phase1c/i09/a2/rows_old_*` (checked against `A2.sha256`).
   - P, from `phase1/i01_repairs/rows_old_*` (pinned sha256). It is asserted equal to A2's P column record by record,
     and A2's `C_rec` is asserted equal to the records' `score.success`.
3. **The folders.** In `--runs` it takes, per (arm, model), the latest folder `<time>_s24_i09_<arm>_<model>_con/`
   whose `config.yaml` says `incomplete: false` (REQUEST.md section 2). It refuses (exit 2) when any of the six is
   missing or incomplete.
4. **Validity.** Any failed gating check makes the reading "not applicable".
   - Each folder is complete, in live mode, made from the right recorded run, at trial 0.
   - 28 records per arm (84 per model), with exactly the recorded t0 keys and situations.
   - Every record has `config.people_source == detector_i09_a1_frozen`, and no record holds a people call.
   - The check files' upstream comparison has `all_equal` for every record. The same comparison is also redone here
     from the records, with the driver's rule.
   - The check-file cache counts have no people stage and no refused upstream request.
   - The driver sha256 in each `config.yaml` and check file equals `BUILD.sha256`'s driver line at analysis time.
   - The FROZEN sha256 recorded in `config.yaml` equals FROZEN.md now.
   - The ledger has no refused line, stays within 120 in all and 60 per model, and sends nothing but constraints
     requests.
   - At most one re-run per model (two live check files at most).
   - Reported but not gating: the count-only reports, any invocation that stopped, a leftover `ledger.lock`, and
     `BUILD.sha256` against `driver.py` now.
5. **The reading and the measures.** It prints the pre-registered reading exactly as FROZEN.md section 6 states it
   (GLM ≥ 66 and DeepSeek ≥ 68: consistent; GLM ≤ 63 and DeepSeek ≤ 64: contradicts; mixed otherwise). It then
   prints every section 5 measure, each labelled (the measures section below).
   - Successes are the package's own `score.success` in the records. Nothing is rescored.
   - Output: `tables.txt` and `summary.json` in `--out`. It refuses an `--out` inside `live/`.

### The measures and their labels

| Measure | Label | What is reported |
|---|---|---|
| Successes of 84 (M1) and the reading | Pre-registered (FROZEN 6, 5.1) | Successes per model; vlm_error count; statuses |
| Line successes and by situation (M2) | Pre-registered measure, no role in the reading (5.2) | D-live, C, D, P per situation |
| Flips and breaks (M3) | Pre-registered measure, no role in the reading (5.3) | Against C, D and P (P secondary). Records, distinct scenes and layouts, net by layout |
| Constraint usability (M4) | Pre-registered measure, no role in the reading (5.4) | `ConstraintUse.status` counts; `no_usable_constraint` steps; `constraints_unusable` and `exclusions_broken` (sum, mean); constraints per goal step |
| Numbers named that exist in the list (M5) | Pre-registered measure, no role in the reading (5.5) | Share of named numbers in the list shown; share of goal steps naming one not in it |
| Goal-step latency (M6) | Descriptive (FROZEN 5.6 calls it so) | Medians and p90: all goal steps, paired, call parts alone; plus 55.6 ms (and 61.9 ms) detector constant |
| Extras (M7) | Descriptive (5.7) | People per goal step against the VLM's list size; frozen detector count per scene; `people_found_share`, `people_unmatched`, `members_precision`, `members_recall`; retries; calls made against the ledger |
| Per-arm successes; distinct numbers per goal step | Descriptive, added by the analysis (not in FROZEN) | — |

## How to run it

```bash
cd .-scenes
LA=results/2026-10-17-beyond-vlm/phase2/i09/live_analysis
(cd $LA && sha256sum -c ANALYSE.sha256)
env PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 .venv/bin/python $LA/analyse.py \
    --runs results/2026-10-17-beyond-vlm/phase2/i09/live/runs --out $LA/result
```

**Exit codes:**
- 0: written, and the reading applies;
- 3: written, but the reading is "not applicable" (the reasons are printed);
- 2: refused.

**Flags:**
- `--allow-incomplete`: analyse what is there; the reading is then not applicable.
- `--count-vlm-errors-as-failures`: see decision 2.
- `--accept-mock` (tests only): also read the driver's `_con_mock` folders.

It takes about 2 s.

## Tests (zero calls; `run_tests.py`, results in `test_out/TESTS.json`): 25 of 25 pass

**(i) The driver's dry-run folders (parsing, refusals, validity).**
- `live/dryrun/a_mock`:
  - without flags it refuses (exit 2: mock folders are not read);
  - with `--accept-mock --allow-incomplete` it reads the 10 GLM records and exits 3. Failed checks: mode mock, the
    first build's driver sha256, 10 of 84 records, upstream unequal, no check file, no ledger.
- `live/dryrun/a_pythonpath_mock` (2 DeepSeek records) and `verify_i09_live_build/a_mock` (16 GLM records; each folder
  matched to one check-file entry): exit 3.
- `live/dryrun/c_cap/c6_main` (fake-gateway live format, both models stopped by the cap): exit 3. The 2 refused ledger
  lines and the stopped invocations are flagged.
- `c6_main_full`: refused strict (exit 2); exit 3 with `--allow-incomplete`.
- `b2_count_only_filled`: the latest of three complete GLM real folders is taken (013304).

**(ii) Synthetic copies of recorded data** (`make_synthetic.py` → `synthetic/<variant>/runs`).
- How they are made: the six runs' recorded t0 records, read through the harness, are written in the driver's
  format.
  - Each record gets `config.people_source` set and its people call taken out of `result.calls`.
  - Check files carry the upstream comparison made with the driver's rule. There is also a ledger with one sent line
    per constraints call (48 + 54) and count-only reports.
  - Every `config.yaml` says `i09_live.synthetic`, and the analysis prints a "TEST INPUT, NOT A RESULT" banner.
- `c_copy`: GLM 64, DeepSeek 65, **mixed**, exit 0.
  - 0 flips and 0 breaks against C.
  - Against D: GLM 0/5, DeepSeek 0/6.
  - M4, M5 and M7 are identical to recorded C's: 102 goal steps, numbers named 100% in the list.
- `d_replay` (`score.success` set to A2's D at t0): GLM 69, DeepSeek 71, **consistent with A2**.
  - 0 flips and 0 breaks against D.
  - Against C: +5 and +6, the A2 t0 gains.
- The rule's boundaries:
  - 63/64: contradicts;
  - 66/68: consistent;
  - 66/67: mixed;
  - 63/65: mixed.
- Refusals and validity failures:
  - a missing arm: exit 2, or 3 with `--allow-incomplete`;
  - a cap stop with a refused ledger line: exit 2, or 3;
  - one changed context answer: both upstream checks fail, exit 3;
  - the first build's driver sha256 in one `config.yaml`: exit 3;
  - one record keeping its people call: exit 3;
  - a vlm_error left after one re-run: exit 2 strict, or with `--count-vlm-errors-as-failures` GLM 63, mixed, exit 0,
    with a note;
  - a later incomplete folder: the earlier complete one is taken, with a note;
  - `--out` inside `live/`: exit 2, nothing created.

## Decisions (where FROZEN.md left a choice)

1. **Folder choice.**
   - Default: the latest complete folder per (arm, model), by the time in its name, as REQUEST.md says.
   - With `--allow-incomplete`: the latest folder of any state, and the reading is not applicable.
2. **A vlm_error left after the one re-run.** FROZEN 5.1 counts it as a failure, and FROZEN 6 defines "complete" as
   "no cap stop". The dispatch's summary, and REQUEST.md's "latest complete folder", would instead withhold the
   reading.
   - Default: strict. Such a folder is incomplete, so the analysis refuses.
3. **What gates the reading.** Every check in step 4 except those listed as reported.
   - A driver sha256 that differs from `BUILD.sha256` gates.
   - The count-only reports do not gate. In live mode the driver refuses every upstream send, and the decisive test is
     the post-run upstream comparison.
   - Upstream equality must hold both in the check file and when redone here.
4. **Goal step.**
   - D-live: a record with a constraints call.
   - Recorded C: a record with a people call (FROZEN 2).
   - "With a constraints answer": `result.constraints` and `result.solver` are both set, so the solver ran on a valid
     answer.
5. **M5.**
   - It counts every number in every constraint of the answer, `cut` ones included, as occurrences ("every number").
     Distinct numbers per goal step are reported as an addition.
   - "The list shown" is the numbers of `result.people`; the script checks that they run 1..N.
6. **M6.**
   - Per goal step: the summed `latency_total_s` of the named stage's calls (a missing call adds 0), plus
     `timings_s.geometry`.
   - The quantile is numpy's default linear one.
   - "Paired" means the records that are goal steps in both arms. Its differences are per step.
   - The detector time is added as a constant, 55.6 ms, with a second line at 61.9 ms.
7. **M7.**
   - People per goal step is the length of the numbered list shown, after the package merges and cuts it. The raw
     frozen count per scene comes from the check files.
   - Retries are split into those of fresh calls and those carried by cached calls.
   - The ledger's sends are compared with the attempts of fresh calls. They are equal only when one invocation made
     every folder taken.
8. **Other.**
   - The live folders are not on the harness lists, so they are read as files. Each path is guarded with
     `H.guard_file`, and each record's situation and scene are guarded before its other fields are used.
   - `--out` is a folder.

**Files:**
- `analyse.py`, `ANALYSE.sha256`, `ORDER.txt`;
- `make_synthetic.py` and `synthetic/` (with `VARIANTS.json`);
- `run_tests.py` and `test_out/`.

See `LEAD_DECISIONS.md`: two validity checks changed (a stopped invocation now gates; a folder left by an invocation stopped from outside is reported), the lock line is information only, the tests were re-run against driver e5e79093 (25 of 25 pass), and the result is computed with `--count-vlm-errors-as-failures` as FROZEN.md 5.1 says. `ANALYSE.sha256` now also holds `LEAD_DECISIONS.md`.
