"""Post hoc, zero calls: CT2m against CT2 by scenario (share strict W/L/T, first-in-line strict later/same/earlier),
next to the timing-only floor (CT2 repeat against CT2, Sonnet seeds 10-12)."""
import json, statistics, sys
from collections import defaultdict
from pathlib import Path
M = Path('results/2026-11-01-push/moving')
sys.path.insert(0, str(M)); import summarise_fresh as S
def fs(e): return next((x['t'] for x in e['samples'] if S.in_line(x, 'strict')), None)
def load(root):
    return {(e['scenario'], e['seed'], e['arm']): e for e in S.load([root]) if e['scenario'] not in S.BREAK}
def compare(A, B, a, b, title, seeds=None):
    per = defaultdict(lambda: [0, 0, 0, 0, 0, 0, []])
    for k in A:
        if k[2] != a or (seeds and k[1] not in seeds): continue
        kb = k[:2] + (b,)
        if kb not in B: continue
        x, y = A[k], B[kb]
        d = S.measures(x)['share_strict'] - S.measures(y)['share_strict']
        p = per[k[0]]
        p[0 if d > 1e-9 else 1 if d < -1e-9 else 2] += 1
        fx, fy = fs(x), fs(y)
        if fx is not None and fy is not None:
            p[3 if fx < fy - 0.5 else 5 if fx > fy + 0.5 else 4] += 1
        p[6].append(d)
    print(f'\n{title}')
    print(f'{"scenario":26s} share W/L/T   mean d    first-in-line earlier/same/later')
    tot = [0] * 6; alld = []
    for sc, p in sorted(per.items()):
        print(f'{sc:26s} {p[0]:2d}/{p[1]:2d}/{p[2]:2d}   {statistics.mean(p[6]):+.3f}    {p[3]}/{p[4]}/{p[5]}')
        tot = [t + v for t, v in zip(tot, p[:6])]; alld += p[6]
    print(f'{"all":26s} {tot[0]:2d}/{tot[1]:2d}/{tot[2]:2d}   {statistics.mean(alld):+.3f}    {tot[3]}/{tot[4]}/{tot[5]}')
    small = [d for d in alld if -0.05 <= d < -1e-9]; big = [d for d in alld if d > 0.05]
    print(f'   losses smaller than 0.05: {len(small)} of {sum(d < -1e-9 for d in alld)}; wins over 0.05: {len(big)} of {sum(d > 1e-9 for d in alld)}')
son, rep, ds = load(M / 'runs/sonnet55_low'), load(M / 'runs/sonnet55_low_fresh_repeat'), load(M / 'runs/dsv41')
compare(son, son, 'CT2m', 'CT2', 'Sonnet CT2m vs CT2, seeds 10-19')
compare(ds, ds, 'CT2m', 'CT2', 'DeepSeek CT2m vs CT2, seeds 10-14', seeds=set(range(10, 20)))
compare(son, son, 'CT2m', 'CT2', 'Sonnet CT2m vs CT2, seeds 10-12 only', seeds={10, 11, 12})
compare(rep, son, 'CT2', 'CT2', 'Floor: Sonnet CT2 repeat vs CT2, seeds 10-12')
compare(rep, son, 'CT', 'CT', 'Floor: Sonnet CT repeat vs CT, seeds 10-12')
