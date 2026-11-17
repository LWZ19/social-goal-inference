# Bystanders on Qwen 3.8

**Question.** Does the bystander confirmation (CT3 against CT2 on the cafe and sweet_spot bystander scenes, seeds
15-44) hold on Qwen 3.8?

**Result.** At the standard 92 s the test is latency-bound: in `x2_score_out.txt` none of the 60 CT2 episodes has a
first answer, so every comparison ties. In the labelled 240 s secondary (cache only), CT3 beats CT2 on strict share, 25/4/31
wins/losses/ties, +0.028, sign p < 0.001, and time spent behind a bystander falls from 2,065 s to 259 s (cells
captured 36 to 6 of 60): both bars pass. The roles bar is not judged there, as the secondary has no Gl arm.

**Files.**
- `X2_FROZEN.md`: the plan, bars and predictions, frozen before any episode; amendment 1 explains the scorer copy.
- `x2_score.py`: the confirmation scorer with two lines guarded for an episode with no answer; `x2_score.diff` is
  its difference from `../bystanders_confirm_score.py`.
- Outputs: `x2_score_out.txt` (92 s), `x2_score_long_out.txt` (240 s), `x2_score_test_out.txt` (the copy on
  DeepSeek's confirmation runs, equal to `../bystanders_confirm_score_out.txt`).

**Reproduce.** From the repository root:

    PYTHONPATH=src .venv/bin/python results/moving-lines/bystanders/qwen/x2_score.py results/data/moving/runs_x2/soclaas_qwen38_low
    PYTHONPATH=src .venv/bin/python results/moving-lines/bystanders/qwen/x2_score.py results/data/moving/runs_x2/soclaas_qwen38_low_long
    PYTHONPATH=src .venv/bin/python results/moving-lines/bystanders/qwen/x2_score.py results/data/moving/runs/dsv41

**Data.** `logs/moving-runs.tar.gz` (`results/data/moving/runs_x2/`, `results/data/moving/runs/dsv41/`).
