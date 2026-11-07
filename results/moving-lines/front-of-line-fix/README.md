# Front-of-line fix

**Question.** Once first in line, the robot sometimes backs away round people and waits beside the counter. Do two
rule fixes (`CT2f`: ignore a spot farther back once in line, and keep taking the place of the last person served when
that person rejoins behind) raise time in the right spot over `CT2`?

**Result.** No measurable change, on 25 cells per model replayed from stored answers: DeepSeek 0.288 against 0.277
(10/8/7), Sonnet 0.331 against 0.324 (12/9/4). The fixes do what they are meant to: far fewer backward steps after
joining (6 against 17, and 2 against 23) and fewer close approaches (within 0.55 m 2.0% against 2.9% of the time).
Exploratory, not planned in advance.

**Files.**
- `RESULTS.md`: the write-up.
- `front_score.py`: the scorer, output `front_score_out.txt`. The output's last block (closeness to people) was
  computed separately with `../bumps/bumps.py`'s rules on the same 50 cells; the script does not print it.

**Reproduce.** From the repository root: `PYTHONPATH=src python results/moving-lines/front-of-line-fix/front_score.py`.
It reproduces the stored output exactly, apart from that last block, which recomputes to the same numbers.

**Data.** `results/logs/moving-runs.tar.gz` (`runs_front/` and `runs_i10/` in `results/data/moving/`).
