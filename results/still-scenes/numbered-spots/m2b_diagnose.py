"""POST HOC diagnosis of M2 (not in M2_FROZEN.md): why replayed success drops although target choice improves, and
what the harm flips are. Reads calls.jsonl and m2_rows.jsonl of a run; zero calls.
    m2b_diagnose.py runs/m2_dsv41
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
calls = [json.loads(l) for l in (run / 'calls.jsonl').read_text().splitlines() if l.strip()]
rows = {(r['scene'], r['episode'], r['trial']): r for r in
        (json.loads(l) for l in (run / 'm2_rows.jsonl').read_text().splitlines() if l.strip())}
by = collections.defaultdict(dict)
for c in calls:
    by[(c['scene_id'], c['episode'], c['trial'])][c['arm']] = c


def first_nav(subs):
    for s in subs:
        if s['type'] == 'navigate' and (s.get('point') is not None or s.get('mark') is not None):
            return s
    return None


out = []
say = out.append
moves, label_hits = [], 0
lost_loc = []          # both right, A0 replay ok, A2-point replay not ok
for key, r in rows.items():
    a0, a2 = by[key].get('A0'), by[key].get('A2')
    if not a0 or not a2 or a0['status'] != 'ok' or a2['status'] != 'ok':
        continue
    s0, s2 = first_nav(a0['subtasks']), first_nav(a2['subtasks'])
    if not s0 or not s2 or s0.get('point') is None or s2.get('point') is None:
        continue
    d = math.hypot(s0['point'][0] - s2['point'][0], s0['point'][1] - s2['point'][1])
    # how far the unmarked click lay from the nearest label centre (a label hides what is under it)
    dl0 = min(math.hypot(s0['point'][0] - c['mark'][0], s0['point'][1] - c['mark'][1]) for c in a2['cands'])
    if r.get('A0_right') and r.get('A2_point_right'):
        moves.append(d)
        if r.get('A0_replay') and not r.get('A2_point_replay'):
            lost_loc.append((key, round(d, 1), round(dl0, 1)))
say(f'run {run.name}: pairs with both clicks right: {len(moves)}; A2 click moved from A0 click: '
    f'median {np.median(moves):.1f} px, p75 {np.percentile(moves, 75):.1f}, p90 {np.percentile(moves, 90):.1f}')
say(f'both right, A0 replay succeeds, A2-point replay fails: {len(lost_loc)}')
if lost_loc:
    dl = [x[2] for x in lost_loc]
    say(f'  for these, the unmarked click lay a median {np.median(dl):.1f} px from the nearest label centre '
        f'(<= 12 px, i.e. under a label: {sum(1 for x in dl if x <= 12)}/{len(dl)}); click moved median '
        f'{np.median([x[1] for x in lost_loc]):.1f} px')
    say('  by scene: ' + ', '.join(f'{k} {v}' for k, v in collections.Counter(x[0][0].split('_', 1)[1] for x in lost_loc).most_common(8)))
# all pairs: unmarked clicks under a label
under = [min(math.hypot(first_nav(by[k]['A0']['subtasks'])['point'][0] - c['mark'][0],
                        first_nav(by[k]['A0']['subtasks'])['point'][1] - c['mark'][1]) for c in by[k]['A2']['cands']) <= 12
         for k in rows if by[k].get('A0', {}).get('status') == 'ok' and by[k].get('A2', {}).get('status') == 'ok'
         and first_nav(by[k]['A0']['subtasks']) and first_nav(by[k]['A0']['subtasks']).get('point') is not None]
say(f'unmarked clicks lying under a label (<= 12 px of its centre): {sum(under)}/{len(under)}')
# harm and gain flips (rule both)
for name, pick in (('HARM', lambda r: r.get('A0_right') and not r.get('A2_both_right')),
                   ('GAIN', lambda r: not r.get('A0_right') and r.get('A2_both_right'))):
    rr = [(k, r) for k, r in rows.items() if pick(r)]
    say(f'\n== {name} ({len(rr)})')
    for k, r in sorted(rr):
        a2 = by[k].get('A2', {})
        s2 = first_nav(a2.get('subtasks', [])) if a2.get('status') == 'ok' else None
        s0 = first_nav(by[k]['A0']['subtasks']) if by[k].get('A0', {}).get('status') == 'ok' else None
        say(f"  {k[0]:<36} {k[1]} t{k[2]}  A0 '{s0 and s0.get('target')}' -> A2 '{s2 and s2.get('target')}' "
            f"mark {r.get('mark')} ({r.get('mark_kind')}) conflict {r.get('binding_conflict')} status {a2.get('status')}")
text = '\n'.join(out)
print(text)
(run / 'm2b_out.txt').write_text(text + '\n')
