import json, sys, itertools, statistics, random
from pathlib import Path
sys.path.insert(0, 'results/2026-11-01-push/moving')
import summarise_fresh as S
R = Path('results/2026-11-01-push/moving/runs/sonnet55_low')
eps = [e for e in S.load([R]) if e['seed'] in (10, 11, 12) and e['scenario'] not in S.BREAK]
by = {}
for e in eps:
    by.setdefault((e['scenario'], e['seed']), {})[e['arm']] = S.measures(e)
arms = ['CT', 'CT2', 'CTd', 'CTe', 'CT2m', 'CT2-vac', 'CT2-rat']
print('seeds 10-12, 15 episodes per arm')
for a in arms:
    m = [v[a] for v in by.values() if a in v]
    if not m: continue
    print(f"{a:8s} n {len(m)} share {statistics.mean(x['share_strict'] for x in m):.3f} end {sum(x['end_strict'] for x in m)} "
          f"back {sum(x['back10'] for x in m)}/{sum(x['events'] for x in m)} cut {sum(x['cut'] for x in m)} "
          f"cut eps>0 {sum(x['cut']>0 for x in m)} max-cut-ep {max(x['cut'] for x in m)}")
def flip_p(d):
    obs = abs(statistics.mean(d)); n = len(d); c = 0
    for signs in itertools.product((1, -1), repeat=n):
        c += abs(sum(s * x for s, x in zip(signs, d)) / n) >= obs - 1e-12
    return c / 2 ** n
print()
for a, b in [('CTe', 'CTd'), ('CTe', 'CT2'), ('CTd', 'CT2'), ('CT2', 'CT'), ('CTe', 'CT'), ('CTd', 'CT'), ('CT2m', 'CT2'), ('CT2-vac', 'CT2'), ('CT2-rat', 'CT2')]:
    ks = [k for k, v in by.items() if a in v and b in v]
    d = [by[k][a]['share_strict'] - by[k][b]['share_strict'] for k in ks]
    dd = [by[k][a]['share_strict'] - by[k][b]['share_strict'] for k in ks if k != ('room_coffee_cart_slots', 10)]
    print(f"{a:7s}-{b:7s} n {len(d)} mean {statistics.mean(d):+.3f} mean|d| {statistics.mean(abs(x) for x in d):.3f} "
          f"sd {statistics.pstdev(d):.3f} flip-p {flip_p(d):.2f}  without room_slots s10: {statistics.mean(dd):+.3f} flip-p {flip_p(dd):.2f}")
