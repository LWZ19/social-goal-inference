# Bystanders: people standing near the line who are not waiting

**Question.** When people stand near the line without waiting (a lone person beside its end, a talking pair), does
the robot wrongly line up behind them, and does motion evidence for who is waiting (CT3: CT2 plus a join gate and a
follow check) prevent it and raise time in line?

**Result.** DeepSeek V4.1 Flash, confirmation (60 cells, seeds 15-44, frozen bars): CT3 beats CT2 on strict share,
17/6/37 wins/losses/ties, +0.018, sign p 0.035, and cuts time spent behind a bystander from 1,005 s to 203 s; both
bars pass. The pilot (15 cells) cut it from 208 s to 9 s. On Claude Sonnet 5.5 the capture bars fail (it names
bystanders as waiting in 13 of 30 answers), but CT3 still beats the geometry-only chain Gl, 17/1/11, +0.115. On
Qwen 3.8 (in `qwen/`) only the 240 s secondary is judged: it passes both bars (25/4/31; 2,065 s to 259 s).

**Files.**
- Plans: `BYSTANDERS_FROZEN.md` (the pilot), `BYSTANDERS_CONFIRM_FROZEN.md` (the confirmation),
  `BYSTANDERS_SONNET_ADDENDUM.md` (a second model).
- Scorers: `bystanders_score.py` (pilot; `bystanders_score_c7532cd2.py` is the judged version, kept byte for byte),
  `bystanders_confirm_score.py`, `bystanders_sonnet_score.py`.
- Outputs: `bystanders_score_out.txt`, `bystanders_confirm_score_out.txt`, `bystanders_sonnet_score_out.txt`.
- Scene checks with no sim: `bystanders_geometry.py` (`bystanders_geometry_out.txt`: which model-free rules take a
  bystander), `bystanders_visibility.py` (what the robot sees from its start).
- `pre/`: zero-call checks written before the pilot (`turn_check.py`: the solver's place-in-turn rule takes a
  bystander on short lines; `gate_fidget.py`: how often a fidgeting bystander passes the join gate).
- `post-hoc/`: diagnosis after the judgements (its own README).
- `qwen/`: the confirmation on Qwen 3.8 (its own README).

**Reproduce.** From the repository root (`R=results/data/moving/runs`, `B=results/moving-lines/bystanders`):

    PYTHONPATH=src .venv/bin/python $B/bystanders_confirm_score.py $R/dsv41
    PYTHONPATH=src .venv/bin/python $B/bystanders_sonnet_score.py $R/sonnet55_low
    PYTHONPATH=src .venv/bin/python $B/bystanders_score.py $R/dsv41 --nh $R/dsv41 $R/sonnet55_low
    PYTHONPATH=src .venv/bin/python $B/bystanders_geometry.py

The current `bystanders_score.py` also prints the seeds and the first-answer sources in its header. The judged
version, `bystanders_score_c7532cd2.py`, is kept unedited, so it still has its old folder's import path and repository
root; with those two lines set as in `bystanders_score.py`, it gives `bystanders_score_out.txt` exactly.

**Data.** `logs/moving-runs.tar.gz` (`results/data/moving/runs/`, and `runs_x2/` for Qwen).
