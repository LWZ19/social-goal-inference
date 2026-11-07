# People finder in place of the people question

**Question.** On 40 fresh still scenes, does the constraint method with the people found by a small detector (D) do
as well as the current method that asks the VLM who is there (A0), faster and with one call fewer?

**Result.** Yes on all six models: D is never less accurate than A0 beyond a second draw of A0, and it is faster on
every model, e.g. Qwen 3.8 107 → 83 s per answer and 144 → 154 of 159 right, GLM-5.3 142 → 109 s and 100 → 125 of 143,
Sonnet 5.5 12.9 → 10.2 s at 155 of 160 both ways. Pooled over the three free models D is also more accurate: net
+41, p 0.0001 over layouts (+30, p 0.016, if each of GLM's 11 failed units counts against D).

**Files.**
- `E2_FROZEN.md`: the plan, written before any scene existed; `E2_ADDENDUM_1.md` (before any score);
  `E2_DEVIATIONS.md`: what happened during the runs.
- `E2_RESULTS.md`: the result on the three free models; `E2_RESULTS_EXTRAS.md`: all six models.
- `e2_score.py`: the scorer. `e2_score_all_out.txt` (the three free models), `e2_score_qd_out.txt` (Qwen and
  DeepSeek, scored first), `e2_score_six_out.txt` (all six).
- `e2_sens.py`: the timing check on units with no cached answer; `e2_sens_all_out.txt`, `e2_sens_qd_out.txt`,
  `e2_sens_six_out.txt`.
- `e2_excluded.py`: GLM's failed units and the worst case for D; `e2_excluded_out.txt`.
- `e2_pooled_worst.py` (post hoc): the pooled result with those units against D; `e2_pooled_worst_out.txt`.
- `check_e2_mech.py`: the mechanical check of a run folder; `mech_soclaas_*.txt` (`mech_soclaas_glm53.txt` is after
  GLM's first pass, `mech_soclaas_glm53_final.txt` after its last); `mech_paid.txt`: the Claude and Gemini runs.
- `e2_driver.py`: runs the three arms (model calls). `e2_detect.py`, `run_detect.sh`: the detector pass, written to
  `e2_people.json`. `e2_assemble.py`: builds `run_scenes/` and `manifest.csv` from the renders; `assemble_log.txt`.
- `runs/`, `runs_smoke/`: each model's records. `early-detector/`: how the detector was chosen (its own README).

**Reproduce.** From the repository root, with `P=results/still-scenes/people-finder` and
`R=$P/runs/e2_soclaas`:

```bash
PYTHONPATH=src python $P/e2_score.py --people $P/e2_people.json --scenes $P/run_scenes ${R}_qwen38_low ${R}_dsv41 ${R}_glm53
PYTHONPATH=src python $P/e2_sens.py  --people $P/e2_people.json --scenes $P/run_scenes ${R}_qwen38_low ${R}_dsv41 ${R}_glm53
PYTHONPATH=src python $P/e2_excluded.py --people $P/e2_people.json --scenes $P/run_scenes ${R}_qwen38_low ${R}_dsv41 ${R}_glm53
PYTHONPATH=src python $P/e2_pooled_worst.py ${R}_qwen38_low ${R}_dsv41 ${R}_glm53
PYTHONPATH=src python $P/check_e2_mech.py ${R}_dsv41
```

Add `$P/runs/e2_claude_sonnet55_low $P/runs/e2_claude_haiku55 $P/runs/e2_gemini_fast` for the six-model files, or
keep only Qwen and DeepSeek for `*_qd_*`. `e2_score.py` also writes `e2_score_out.txt` into the first run folder. The
scorers read the scenes' pictures and depth images (for the detector check), which are not in the repository; with
them every output is reproduced exactly. `e2_pooled_worst.py` and `check_e2_mech.py` need only the repository.

**Data.** `results/logs/research-data.tar.gz` (`runs/`, `runs_smoke/`, `e2_people.json`, the scene files in
`run_scenes/` and `renders/`).
