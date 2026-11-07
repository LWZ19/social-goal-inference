"""Verifier: compare my validation and noisy rows with the compute agent's (read as data) and my R5 with I01's."""
import json
import sys
from pathlib import Path

import numpy as np

I06 = Path(__file__).resolve().parents[1]
OUT = I06 / 'verify_i06_score/out'
I01 = I06.parent / 'replay-harness/repairs'
NAME = {'qwen': 'qwen38_low', 'glm': 'glm53', 'dsv': 'dsv41'}


def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x]


for key in sys.argv[1:]:
    mine = rows(OUT / f'validate_{key}.jsonl')
    theirs = rows(I06 / f'score/noise/validation_{key}.jsonl')
    k = lambda r: (r['run'], r['scene'], r['episode'], str(r['trial']))  # noqa: E731
    T = {k(r): r for r in theirs}
    M = {k(r): r for r in mine}
    assert set(T) == set(M), (len(T), len(M))
    d_res = sum(M[x]['rescore'] != T[x]['rescore'] for x in M)
    d_r5 = sum(M[x]['r5'] != T[x]['r5'] for x in M)
    steps = [x for x in M if M[x]['step']]
    dsim = np.array([M[x]['sim'] - T[x]['sim_mean'] for x in steps])
    print(f'{key}: records {len(M)}, steps {len(steps)}; rescore differs {d_res}; R5 differs {d_r5}; '
          f'sim per-step diff mean {100 * dsim.mean():+.2f} pts, |diff| max {np.abs(dsim).max():.2f}; '
          f'mine {100 * np.mean([M[x]["sim"] for x in M]):.2f} theirs {100 * np.mean([T[x]["sim_mean"] for x in M]):.2f}')
    i01 = rows(I01 / f'rows_old_{NAME[key]}.jsonl')
    agree = n = 0
    for r in i01:
        x = (r['run'], r['scene'], r['episode'], str(r['trial']))
        if x in M:
            n += 1
            agree += bool(r['R5']) == bool(M[x]['r5'])
    print(f'   R5 vs I01 rows_old: {agree}/{n}')
