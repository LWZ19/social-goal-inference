"""Post hoc, zero calls: CT2m against CT2 for x-test-3 (seeds 10-12): scene split, and where the robot stands at the
full answer relative to the place finally taken."""
import json, math, statistics, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths
M = paths.MOVING_RUNS
sys.path.insert(0, str(paths.DESIGN)); import summarise_fresh as S
def fs(e): return next((x['t'] for x in e['samples'] if S.in_line(x, 'strict')), None)
def at(s, t):
    b = s[0]
    for x in s:
        if x['t'] <= t + 1e-6: b = x
    return b
eps = {(e['scenario'], e['seed'], e['arm']): e for e in S.load([M / 'runs/x3']) if e['scenario'] not in S.BREAK}
per = defaultdict(list)
for k, m in sorted(eps.items()):
    if k[2] != 'CT2m' or k[:2] + ('CT2',) not in eps: continue
    b = eps[k[:2] + ('CT2',)]
    if not m['answers'] or not b['answers']:
        per[k[0]].append(None); continue
    T = m['answers'][0]['delivered_t']; fm, fb = fs(m), fs(b)
    iv = next((x for x in m.get('events', []) if x['event'] == 'interim'), None)
    place = at(b['samples'], fb)['robot'] if fb else None
    st, rT = m['samples'][0]['robot'], at(m['samples'], T)['robot']
    per[k[0]].append(dict(seed=k[1], T=round(T, 1), t_int=iv and iv['t'], fb=fb, fm=fm,
                          d0=place and round(math.hypot(place[0] - st[0], place[1] - st[1]), 1),
                          dT=place and round(math.hypot(place[0] - rT[0], place[1] - rT[1]), 1),
                          shb=round(S.measures(b)['share_strict'], 2), shm=round(S.measures(m)['share_strict'], 2)))
for sc, rs in sorted(per.items()):
    print(sc)
    for r in rs:
        print('   ', r)
