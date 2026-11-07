"""Post hoc, zero calls: time from the full answer's delivery to the first re-solve with a usable place in turn
(in_turn), and from there to first in line (strict), for CT2 and CT2m, Sonnet seeds 10-19 and DeepSeek 10-14."""
import json, statistics, sys
from collections import defaultdict
from pathlib import Path
M = Path('results/2026-11-01-push/moving')
sys.path.insert(0, str(M)); import summarise_fresh as S
for root, label in ((M / 'runs/sonnet55_low', 'Sonnet'), (M / 'runs/dsv41', 'DeepSeek')):
    for arm in ('CT2', 'CT2m'):
        per = defaultdict(list)
        for e in S.load([root]):
            if e['arm'] != arm or e['scenario'] in S.BREAK or not e['answers']: continue
            T = e['answers'][0]['delivered_t']
            r = next((r['sent_t'] for r in e['resolves'] if r.get('in_turn')), None)
            f = next((x['t'] for x in e['samples'] if S.in_line(x, 'strict')), None)
            per[e['scenario']].append((None if r is None else r - T, None if (r is None or f is None) else f - r,
                                       e['resolves'][0].get('in_turn') if e['resolves'] else None))
        print(f'\n{label} {arm}: median s from answer to first in-turn re-solve | from it to first in line (strict) | first re-solve already in turn')
        for sc, rs in sorted(per.items()):
            a = [x[0] for x in rs if x[0] is not None]; b = [x[1] for x in rs if x[1] is not None]
            print(f'  {sc:24s} n {len(rs):2d}  {statistics.median(a) if a else float("nan"):5.1f} | {statistics.median(b) if b else float("nan"):5.1f} | {sum(bool(x[2]) for x in rs)}/{len(rs)}')
