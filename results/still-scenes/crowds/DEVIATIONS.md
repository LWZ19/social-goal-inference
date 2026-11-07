# I06 deviations from FROZEN.md

FROZEN.md sha256 a83c0ffe0cf5fafd9f007fee478d5899c9f858d2eb77629f01885cbeccef693e. Each entry is dated and written
before the data it affects is scored. The sha256 of this file is in DEVIATIONS.sha256; whoever adds an entry re-hashes.
## D1 (18 Oct 2026, build step, during generation, before any scoring): output places

- FROZEN.md section 11 names `manifest.jsonl`, `logs/` and `labels.txt` (relative to `results/still-scenes/crowds/`) and
  `builder/base_check.txt`. The build task puts logs, the manifest and the checks under `build/`.
- Done: `build/manifest.jsonl` and `build/manifest.json` (the task's name; same entries plus counts), `build/labels.txt`
  (with per-scene rows in `build/labels_rows_seed<N>.jsonl`), `build/logs/`, `builder/base_check.txt` with a copy in
  `build/base_check.txt`; scenes in `scenes/seed<N>/` as the note says.
- Effect on data: none. No rule, stream, threshold or label changes.

Readings of points the note leaves open (draw order, D's band sampling, the bisection root, c5-c7 measuring points) are
not deviations; they are listed in `build/readings.txt`.
