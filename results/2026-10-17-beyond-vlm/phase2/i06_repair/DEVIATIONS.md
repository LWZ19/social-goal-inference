# I06 repair probe: deviations from FROZEN.md

FROZEN.md sha256 `02432f62320c0ebb0924c0f0e5b4b2089d508eb4a85fd28cf56641c25e27560f` (verified with `sha256sum -c FROZEN.sha256` before any work). Entries are appended and never edited. The file's current hash is in `DEVIATIONS.sha256`. Earlier hashes are listed in `logs/deviations_hash_history.txt`.

## D1. File locations (18 Oct 2026, 06:45 +08:00, before any pass ran)

- Code goes in `code/` (`rules.py`, `common.py`, `crowd.py`, `crowd_noisy.py`, `records.py`, `tables.py`, `run_*.sh`). FROZEN put it at the top of `phase2/i06_repair/`.
- The final rows go at the top of `phase2/i06_repair/`, not in `out/`:
  - `rows_crowd.jsonl` (FROZEN: `out/crowd_rows.jsonl`);
  - `rows_noisy_<model>.jsonl` (FROZEN: `out/noisy_<model>.jsonl`);
  - `rows_records_<cell>.jsonl` (FROZEN: `out/records_<cell>.jsonl`).
- Each pass's own rows go in `work/`, and a merge step builds the final rows from them.
- `out/r0_*.txt`, `tables.txt`, `summary.json` and `logs/` stay where FROZEN puts them.
- The compute agent does not write `README.md`. FROZEN section 8 step 6 leaves the results page to "its writer", who runs no new arm.

## D2. Arm 1b: "simulated once per draw" (18 Oct 2026, 06:45 +08:00, before any pass ran)

FROZEN section 8 runs the R0 passes over the whole source before any repair pass. So the RF90 pass of arm 1b runs in a separate process after R0's.
- The RF90 pass simulates each draw again with I06's same seeded generator, `noise.scene_rng(model, scene_id, d)`.
- The R0 pass stores a sha256 of each draw's `found` list: every person's xy, yaw, pixel and label, plus the source id. The RF90 pass asserts that its own `found` has the same hash.
- So both rules get the same `found` content for every draw, which is what FROZEN section 4 asks for. They do not get the same Python objects.
- A hash mismatch stops the RF90 pass.
