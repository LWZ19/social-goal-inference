# Round 7: the ceiling, and where time in line is lost

**Question.** Where is strict time in line lost, and how much would the robot gain from knowing the true people
ahead, the true counter point, or the ideal place itself?

**Result.** On 66 confirm layouts (DeepSeek, strict share in line from the first service, band top 1.2 m): the ideal
place (OW) scores 0.640, the true people ahead plus the true counter point (CT4OT) 0.330, CT3F 0.208 and CT4PJ 0.159.
Each fix alone adds about 0.05 and both together 0.171; giving CT3F only the true counter point (CT3FT) adds +0.053
(27/23/16, sign p 0.67), so the counter point and where to stand at the front matter as much as who waits.
Descriptive only; it does not change the confirm result.

**Files.**
- `ROUND7.md` (the ceiling arms, readings and predictions), `ROUND7_B.md` (band top 1.5 m as a sensitivity point),
  `ROUND7_D.md` (the rule for CT3FT): written before the totals.
- `a_loss.py` -> `a_loss_pilot_out.txt`, `a_loss_confirm_dsv41_out.txt`, `a_loss_confirm_sonnet55_low_out.txt`,
  `a_loss_ceiling_out.txt`, `a_loss_ct3ft_out.txt` and the matching `.csv` files: where strict seconds are lost, per
  arm.
- `b_band.py` -> `b_band_*_out.txt`: band top 1.2 m against 1.5 m.
- `c_gaps.py` -> `c_gaps_ceiling_out.txt`: the ceiling gaps read by the fixed rules.
- `d_post.py` -> `d_post_out.txt`: post hoc description (layouts where OW is low, arms against CT3F, CT4OT's front
  goals).
- `e_ct3ft.py` -> `e_ct3ft_out.txt`: CT3FT read by its fixed rule.
- `f_front.py` -> `f_front_out.txt`: post hoc, re-solves with nobody ahead.
- `RESULTS_3.md` (the ceiling) and `RESULTS_4.md` (CT3FT).

**Reproduce.** From the repo root (selection files and CSV paths relative to `results/moving-lines/`, run folders
relative to `results/data/moving/`, scenario folders relative to the repo); `D` is
`results/moving-lines/best-version-and-running-guess/round7`:

    PYTHONPATH=src .venv/bin/python $D/a_loss.py test-layouts/confirm_selection.json sim/scenarios/confirm4 \
        runs_confirm4/dsv41 CT3F CT4PJ CT4P CT4 --csv best-version-and-running-guess/round7/a_loss_confirm_dsv41.csv
    PYTHONPATH=src .venv/bin/python $D/a_loss.py test-layouts/confirm_selection.json sim/scenarios/confirm4 \
        runs_ceiling4/dsv41 OW CT4OT CT4O CT4PJT --csv best-version-and-running-guess/round7/a_loss_ceiling.csv
    PYTHONPATH=src .venv/bin/python $D/b_band.py test-layouts/confirm_selection.json runs_ceiling4/dsv41 \
        OW CT4OT CT4O CT4PJT --pairs OW-CT4OT,CT4OT-CT4O,CT4OT-CT4PJT
    PYTHONPATH=src .venv/bin/python $D/c_gaps.py test-layouts/confirm_selection.json runs_ceiling4/dsv41 runs_confirm4/dsv41
    PYTHONPATH=src .venv/bin/python $D/d_post.py test-layouts/confirm_selection.json sim/scenarios/confirm4 \
        runs_ceiling4/dsv41 runs_confirm4/dsv41
    PYTHONPATH=src .venv/bin/python $D/e_ct3ft.py test-layouts/confirm_selection.json runs_ceiling4/dsv41 runs_confirm4/dsv41
    PYTHONPATH=src .venv/bin/python $D/f_front.py

The other outputs use the same scripts with the run folder and arms in their first line: the pilot with
`test-layouts/pilot_selection.json sim/scenarios/pilot3 runs_pilot3/dsv41 CT3F CT4PJ CT4P CT2`, Sonnet with
`runs_confirm4/sonnet55_low`, CT3FT with `runs_ceiling4/dsv41 CT3FT`; `b_band.py` on the pilot and confirm runs with
`--pairs CT4PJ-CT4P,CT4PJ-CT3F,CT4-CT3F` (pilot arms `CT3F CT4PJ CT4P CT4 CT2 CT5 W0`, confirm arms
`CT3F CT4PJ CT4P CT4`).

**Data.** `logs/moving-runs.tar.gz` (`runs_pilot3/dsv41`, `runs_confirm4/dsv41`, `runs_confirm4/sonnet55_low`,
`runs_ceiling4/dsv41`) and `logs/research-data.tar.gz` (the two selection files). `a_loss.py`, `d_post.py` and
`f_front.py` also read the generated layout files in `sim/scenarios/pilot3/` and `sim/scenarios/confirm4/`, which are
not in the repo: `test-layouts/select_pilot.py` and `test-layouts/select_confirm.py` write them from the layout
candidates in `logs/bulk.tar.gz`.
