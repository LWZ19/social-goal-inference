"""Post hoc, zero calls: per scenario, where the interim goal leaves CT2m's robot when the full answer arrives, against
CT2's robot (still at the start): straight-line distance to the place finally taken, path driven from the answer to first
in line, and time from the answer to first in line. Sonnet seeds 10-19."""
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
def plen(s, t0, t1):
    L = 0
    for a, b in zip(s, s[1:]):
        if a['t'] >= t0 - 1e-6 and b['t'] <= t1 + 1e-6:
            L += math.hypot(b['robot'][0] - a['robot'][0], b['robot'][1] - a['robot'][1])
    return L
def ang(a, b): return abs((a - b + math.pi) % (2 * math.pi) - math.pi) * 180 / math.pi
root = M / 'runs/sonnet55_low'
eps = {(e['scenario'], e['seed'], e['arm']): e for e in S.load([root]) if e['scenario'] not in S.BREAK}
per = defaultdict(list)
for k, m in eps.items():
    if k[2] != 'CT2m' or not m['answers']: continue
    b = eps[k[:2] + ('CT2',)]
    fm, fb = fs(m), fs(b)
    if fm is None or fb is None: continue
    T = m['answers'][0]['delivered_t']
    place = at(b['samples'], fb)['robot']
    st = m['samples'][0]['robot']; rT = at(m['samples'], T)['robot']
    # bearing from robot to place vs robot heading at T (turn needed); and whether the line (people) lies between
    brg_b = math.atan2(place[1] - st[1], place[0] - st[0]); brg_m = math.atan2(place[1] - rT[1], place[0] - rT[0])
    per[k[0]].append(dict(
        d_b=math.hypot(place[0] - st[0], place[1] - st[1]), d_m=math.hypot(place[0] - rT[0], place[1] - rT[1]),
        L_b=plen(b['samples'], T, fb), L_m=plen(m['samples'], T, fm), t_b=fb - T, t_m=fm - T,
        turn_b=ang(st[2], brg_b), turn_m=ang(rT[2], brg_m),
        cut_after_b=sum(x['cut_in'] for x in b['samples'] if T < x['t'] <= fb), cut_after_m=sum(x['cut_in'] for x in m['samples'] if T < x['t'] <= fm)))
print('per scenario, medians: straight distance to the place at the answer (CT2 | CT2m), path driven answer->first in line, '
      'time answer->first in line, turn needed at the answer (deg), cut-in samples answer->first in line (sums)')
for sc, rs in sorted(per.items()):
    md = lambda key: statistics.median(r[key] for r in rs)
    print(f"{sc:24s} n {len(rs):2d} dist {md('d_b'):.1f} | {md('d_m'):.1f}  path {md('L_b'):.1f} | {md('L_m'):.1f}  "
          f"time {md('t_b'):.1f} | {md('t_m'):.1f}  turn {md('turn_b'):.0f} | {md('turn_m'):.0f}  cut {sum(r['cut_after_b'] for r in rs)} | {sum(r['cut_after_m'] for r in rs)}  "
          f"path/dist {statistics.median(r['L_b']/max(r['d_b'],.1) for r in rs):.2f} | {statistics.median(r['L_m']/max(r['d_m'],.1) for r in rs):.2f}")
