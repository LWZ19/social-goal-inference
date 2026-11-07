"""Post hoc, zero calls: is CT2m's interim goal nearer the plan's target point than the place the robot finally took
(i.e. level with or ahead of the line's back)? Sonnet seeds 10-19, by scenario."""
import json, math, re, statistics, sys
from collections import defaultdict
from pathlib import Path
M = Path('results/2026-11-01-push/moving')
sys.path.insert(0, str(M)); import summarise_fresh as S
root = M / 'runs/sonnet55_low'
per = defaultdict(list)
for f in sorted(root.glob('*/seed*/CT2m/episode.json')):
    m = json.loads(f.read_text())
    if m['seed'] < 10 or m['scenario'] in S.BREAK or not m['answers']: continue
    b = json.loads((f.parent.parent / 'CT2' / 'episode.json').read_text())
    fb = next((x['t'] for x in b['samples'] if S.in_line(x, 'strict')), None)
    if fb is None: continue
    place = next(x['robot'] for x in b['samples'] if x['t'] >= fb)
    tg = re.search(r'TARGET at \((-?[\d.]+), (-?[\d.]+)', (f.parent / '001' / 'inference.txt').read_text())
    tx, ty = float(tg.group(1)), float(tg.group(2))
    g = next(x['goal'] for x in m['events'] if x['event'] == 'interim')
    st = m['samples'][0]['robot']
    per[m['scenario']].append((math.hypot(g[0] - tx, g[1] - ty), math.hypot(place[0] - tx, place[1] - ty),
                               math.hypot(st[0] - tx, st[1] - ty)))
for sc, rs in sorted(per.items()):
    print(f'{sc:24s} n {len(rs):2d}  interim->target {statistics.median(r[0] for r in rs):.1f} m, place->target '
          f'{statistics.median(r[1] for r in rs):.1f} m, start->target {statistics.median(r[2] for r in rs):.1f} m; '
          f'interim nearer the target than the place: {sum(r[0] < r[1] - 0.3 for r in rs)}/{len(rs)}')
