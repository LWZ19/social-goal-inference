"""Post hoc, zero calls: the most any interim goal could gain over CT2. If the robot already stood at its place when the
full answer arrived, first in line could come at the answer at best, so the saving is at most (first in line - answer).
Share-in-line ceiling = that saving / samples (91), counted only where CT2 reached the line. Per model."""
import statistics, sys
from pathlib import Path
M = Path('results/2026-11-01-push/moving')
sys.path.insert(0, str(M)); import summarise_fresh as S
for root, label in (('sonnet55_low', 'Sonnet 10-19'), ('dsv41', 'DeepSeek 10-14'), ('x3', 'x-test-3 10-12')):
    sav, n, reached = [], 0, 0
    for e in S.load([M / 'runs' / root]):
        if e['arm'] != 'CT2' or e['scenario'] in S.BREAK or not e['answers']: continue
        n += 1
        f = next((x['t'] for x in e['samples'] if S.in_line(x, 'strict')), None)
        if f is None: sav.append(0.0); continue
        reached += 1
        sav.append(max(0.0, f - e['answers'][0]['delivered_t']))
    fin = [s for s in sav if s > 0]
    print(f'{label:16s} cells {n:2d}, CT2 reached {reached:2d}: answer->first in line median {statistics.median(fin):.1f} s; '
          f'share ceiling mean {statistics.mean(sav) / 91:+.3f} (perfect guess, every cell)')

print('\nSpeculative ceiling (a guess right at the plan stage): the robot could be in line from (plan time + travel) instead of')
print('(answer + travel), so the saving is at most the full answer time minus the plan stage time (CT2m logs both).')
for root, label in (('sonnet55_low', 'Sonnet 10-19'), ('dsv41', 'DeepSeek 10-14'), ('x3', 'x-test-3 10-12')):
    gaps = []
    for e in S.load([M / 'runs' / root]):
        if e['arm'] != 'CT2m' or e['scenario'] in S.BREAK or not e['answers']: continue
        iv = next((x for x in e.get('events', []) if x['event'] == 'interim'), None)
        if iv: gaps.append(e['answers'][0]['delivered_t'] - iv['t'])
    print(f'{label:16s} cells {len(gaps):2d}: answer minus plan stage median {statistics.median(gaps):.1f} s -> share ceiling about {statistics.median(gaps) / 91:+.2f}')
