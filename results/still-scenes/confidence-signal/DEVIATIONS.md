# Phase 3 confidence: deviations and clarifications (compute agent, 18 Oct 2026)

The compute agent wrote this on 18 Oct 2026 at about 17:45 +0800. That was after pass 1 (`code/confidence.py`), whose checks R1–R7 all passed (`checks.txt`), and before pass 2 (`code/tables.py`) computed any AUROC, rate or criterion. Its sha256 is in `DEVIATIONS.sha256`, and pass 2 verifies it on start. No signal was looked at against an outcome before this file was hashed. None of the points below changes a signal, a label, a record set, the judged pooled AUROC or the criterion of FROZEN.md section 7.

## 1. Output layout (FROZEN.md section 9)

- **Rows.** Pass 1 writes one file per cell, `rows_<cell>.jsonl` (11 files: `old_{qwen,glm,deepseek}`, `watch_{qwen,glm,deepseek}`, `new_{glm,deepseek,haiku,sonnet,gemini}`), not a single `rows.jsonl`. The note allows 2 processes split by cell group, and the task names the files `rows_*.jsonl`. Row content is as the note specifies, and a row does not depend on how the cells were split. The split used was {old ×3, watching ×3, new_glm} and {new_deepseek, new_haiku, new_sonnet, new_gemini}.
- **Checks.** `checks.txt` is written by a separate mode, `confidence.py checks`, which runs after both processes because R1's key-set test needs every cell. That mode re-verifies FROZEN.md and `replay_d.verify_hashes()` itself.
- **Log.** Both processes append to the one `pass1.log`, and each line carries its process id. For a watching run, the logged "allowed" count is the number of audience records kept (12 of each run's 24 allowed records).
- **Code** is in `code/`, as the task asks.

## 2. Readings fixed in code (no change to the note's definitions)

- **S4 trial groups at watching.** These are built from the run's allowed `audience` records. The note says every member is an audience record anyway, and pass 1 asserts that every trial group, in every run, has a single situation.
- **Selective accuracy.** K = q·n is computed as an exact fraction (q = 4/5 and 1/2), and so are the tie weights. The weights are summed to K exactly (asserted) and then converted to float.
- **S6 / S6z coefficients (T7)** are reported on the standardised scale of each fit (the full-data fit uses the full data's mean and population SD), with the intercept. A feature with weight 0 because its training SD is below 1e-12 is printed as inactive.
- **Pooled AUROC over the 549 I14 primary records** (old plus watching) is printed as point values only. The note defines no draw matrix that spans both realistic and watching layouts (its three matrices are separate groups), so no interval is invented for it.
- **Wrong-target AUROC (T4, T6 c)** is printed as point values only. The note calls it descriptive and asks for no bootstrap for it.
- **Extra descriptive columns.** The stratified pooled AUROC and the mean of the cell AUROCs are also printed for the right-target (T3) and secondary (T6) blocks, beside the one-ranking pooled value.
- **A self-check added to pass 2.** The bootstrap's weighted AUROC (pairs weighted w_i·w_j, which equals the midrank AUROC of the resample with each record repeated w times) is asserted equal to the midrank AUROC at unit weights, within 1e-12, for S2, S3, S4 and S5b in every cell.

## 3. Extra provenance lines in checks.txt (informational, not part of R1–R7)

- The note's section 2 statement that each record's layout equals I04's `layout` field is checked: 1,265 of 1,265.
- `replay_d.py` and the eight A2 row files used are checked against `phase1c/i09/a2/A2.sha256`: all match.
- The sha256 of every other input file (I04 rows, I14 rows, `i14.py`, `repairs.py`, `eval_people.json`) and of every pass-1 row file is listed.
