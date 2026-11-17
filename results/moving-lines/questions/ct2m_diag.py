"""Post hoc, zero calls: why CT2m (move while thinking) did not bring first-in-line earlier than CT2.
Per paired cell (same cached answers): the interim goal, where it left the robot when the full answer arrived relative
to the place it ended up taking, travel, cut-ins before/after the full answer, and whether the first re-solve differs."""
import json, math, statistics, sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths
M = paths.MOVING_RUNS
sys.path.insert(0, str(paths.DESIGN))
import summarise_fresh as S

def at(samples, t):
    best = samples[0]
    for s in samples:
        if s['t'] <= t + 1e-6: best = s
    return best

def path_len(samples, t_end):
    L = 0.0
    for a, b in zip(samples, samples[1:]):
        if b['t'] > t_end + 1e-6: break
        L += math.hypot(b['robot'][0] - a['robot'][0], b['robot'][1] - a['robot'][1])
    return L

def first_strict(s):
    return next((x['t'] for x in s if S.in_line(x, 'strict')), None)

def run(root, label):
    eps = {}
    for e in S.load([root]):
        if e['scenario'] in S.BREAK: continue
        eps[(e['scenario'], e['seed'], e['arm'])] = e
    cells = sorted({k[:2] for k in eps if k[2] == 'CT2m' and k[:2] + ('CT2',) in eps})
    rows = []
    for c in cells:
        m, b = eps[c + ('CT2m',)], eps[c + ('CT2',)]
        if not m['answers'] or not b['answers']:
            rows.append(dict(cell=c, none=True)); continue
        T = m['answers'][0]['delivered_t']
        start = m['samples'][0]['robot'][:2]
        ev = [x for x in m.get('events', []) if x.get('event') == 'interim']
        iv = ev[0] if ev else None
        fb, fm = first_strict(b['samples']), first_strict(m['samples'])
        # the place: where CT2's robot first stood in line (strict), else its first in-turn re-solve pose, else first pose
        if fb is not None:
            place = at(b['samples'], fb)['robot'][:2]
        else:
            rr = next((r for r in b['resolves'] if r.get('in_turn') and r.get('pose_world')), None) or \
                 next((r for r in b['resolves'] if r.get('pose_world')), None)
            place = (rr['pose_world'] if rr else b['answers'][0]['pose_world'])[:2]
        rT = at(m['samples'], T)['robot'][:2]
        d0 = math.hypot(start[0] - place[0], start[1] - place[1])
        dT = math.hypot(rT[0] - place[0], rT[1] - place[1])
        r0b = next((r['pose_world'] for r in b['resolves'] if r.get('pose_world')), b['answers'][0]['pose_world'])
        r0m = next((r['pose_world'] for r in m['resolves'] if r.get('pose_world')), m['answers'][0]['pose_world'])
        tend = max(fb or 92, fm or 92)
        cutpre = sum(x['cut_in'] for x in m['samples'] if x['t'] <= T)
        rows.append(dict(cell=c, none=False, interim=bool(iv and iv.get('status') == 'ok'), stop=(iv or {}).get('stop_m'),
                         t_int=(iv or {}).get('t'), T=T, d0=d0, dT=dT, fb=fb, fm=fm,
                         Lb=path_len(b['samples'], fb or 92), Lm=path_len(m['samples'], fm or 92),
                         cutpre=cutpre, cut_m=sum(x['cut_in'] for x in m['samples']), cut_b=sum(x['cut_in'] for x in b['samples']),
                         r0diff=math.hypot(r0b[0] - r0m[0], r0b[1] - r0m[1]),
                         sh_m=S.measures(m)['share_strict'], sh_b=S.measures(b)['share_strict'],
                         arrive_b=next((x['t'] for x in b['samples'] if x['state'] == 'arrived'), None)))
    ok = [r for r in rows if not r['none']]
    wi = [r for r in ok if r['interim']]
    print(f'\n## {label}: {len(rows)} paired cells, {len(rows) - len(ok)} with no first answer')
    print(f'interim goal given: {len(wi)}/{len(ok)}; statuses other than ok: {[ (r["cell"], r["stop"]) for r in ok if not r["interim"]][:6]}')
    stops = [r['stop'] for r in wi if r['stop'] is not None]
    print(f'stop_m: median {statistics.median(stops):.2f}; person stop fired (stop_m > 2.51): {sum(s > 2.51 for s in stops)}/{len(stops)}')
    prog = [r['d0'] - r['dT'] for r in wi]
    print(f'progress towards the place taken by the time the full answer arrives (d_start - d_T): median {statistics.median(prog):+.2f} m; '
          f'closer {sum(p > 0.3 for p in prog)}, about same {sum(abs(p) <= 0.3 for p in prog)}, farther {sum(p < -0.3 for p in prog)}')
    print(f'distance to the place at answer time: CT2 (at start) median {statistics.median(r["d0"] for r in wi):.2f} m, CT2m median {statistics.median(r["dT"] for r in wi):.2f} m')
    both = [r for r in ok if r['fb'] is not None and r['fm'] is not None]
    dd = [r['fm'] - r['fb'] for r in both]
    print(f'first in line strict, both reached ({len(both)}): CT2m - CT2 median {statistics.median(dd):+.1f} s; earlier {sum(x < -0.5 for x in dd)}, same {sum(abs(x) <= 0.5 for x in dd)}, later {sum(x > 0.5 for x in dd)}')
    print(f'   reached: CT2 {sum(r["fb"] is not None for r in ok)}, CT2m {sum(r["fm"] is not None for r in ok)}')
    lag = [r['fb'] - r['arrive_b'] for r in both if r['arrive_b'] is not None]
    print(f'CT2: first "arrived" to first in line strict: median {statistics.median(lag):.1f} s (0 = in line on arrival); >2 s in {sum(x > 2 for x in lag)}/{len(lag)}')
    print(f'travel until first in line (both reached): CT2 median {statistics.median(r["Lb"] for r in both):.1f} m, CT2m {statistics.median(r["Lm"] for r in both):.1f} m; CT2m longer by >1 m in {sum(r["Lm"] - r["Lb"] > 1 for r in both)}/{len(both)}')
    print(f'cut-in samples: CT2 {sum(r["cut_b"] for r in ok)}, CT2m {sum(r["cut_m"] for r in ok)}, of which before the full answer {sum(r["cutpre"] for r in ok)} (in {sum(r["cutpre"] > 0 for r in ok)} cells)')
    print(f'first re-solve pose differs by > 0.3 m: {sum(r["r0diff"] > 0.3 for r in ok)}/{len(ok)}')
    # by direction of progress: share diff
    for name, sel in (('closer', lambda p: p > 0.3), ('same', lambda p: abs(p) <= 0.3), ('farther', lambda p: p < -0.3)):
        sub = [r for r in wi if sel(r['d0'] - r['dT'])]
        if sub:
            d = [r['sh_m'] - r['sh_b'] for r in sub]
            f = [r['fm'] - r['fb'] for r in sub if r['fm'] is not None and r['fb'] is not None]
            print(f'   {name:7s} n {len(sub):2d}: share CT2m-CT2 mean {statistics.mean(d):+.3f} W/L/T {sum(x > 1e-9 for x in d)}/{sum(x < -1e-9 for x in d)}/{sum(abs(x) <= 1e-9 for x in d)}; first-in-line diff median {statistics.median(f) if f else float("nan"):+.1f} s')
    return rows

for root, label in ((M / 'runs/sonnet55_low', 'Sonnet'), (M / 'runs/dsv41', 'DeepSeek')):
    run(root, label)
