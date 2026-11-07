# First check on Sonnet (E1c)

**Question.** On 15 Sonnet episodes (five scenes, seeds 10-12), does giving the model the detector's people for its
first answer (`CTe`) bring the first answer sooner than `CTd`, and does it change time in line?

**Result.** The first answer came 2.4 s sooner by the median (about 2.2 s once the detection is counted). Time in
line did not differ beyond the spread of nearly identical arms: `CTe` − `CTd` +0.070 (sign-flip p 0.12), while `CT2`
re-run on the same stored answers moved by +0.040 (mean |d| 0.052). Detector people made Sonnet ask for
keep_out_of_shared_space far more often: 12 of 15 first answers, against 13 of 44 with its own people.

**Files.**
- `E1C_FROZEN.md`: the plan. The write-up is the E1c part of `../../own-camera/detector-on-sim-frames/RESULTS.md`.
- `cte_pairs.py`: per episode, the four arms side by side, output `cte_pairs_out.txt`.
- `cte_people.py`: people positions and constraints in each arm's first answer, output `cte_people_out.txt`.
- `cte_stats.py`: per-arm totals and paired differences, output `cte_stats_out.txt`.
- `cte_floor.py`: the redraw floor and the shared-space counts, output `cte_floor_out.txt`. Its last two lines (the
  Fisher and stratified exact p) were computed separately and are not printed by the script.

**Reproduce.** From the repository root:
`PYTHONPATH=src python results/moving-lines/people-finder/first-check/cte_stats.py`, and the other three the same
way. The scripts read every episode in `runs/sonnet55_low`, which has grown since 2 Nov (bystander scenes and more
seeds), so `cte_pairs`, `cte_stats` and part (2) of `cte_floor` now print more rows. Limited to the episodes that
existed when they were run, all four reproduce their stored outputs (`cte_people` reproduces as is).

**Data.** `results/logs/moving-runs.tar.gz` (`runs/sonnet55_low` and `runs/sonnet55_low_fresh_repeat` in
`results/data/moving/`).
