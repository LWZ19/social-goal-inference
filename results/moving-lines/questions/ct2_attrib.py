"""Post hoc, zero calls: what carries CT2's gain over CT v1 on fresh seeds?
(1) the run's failure classes (analysis/failures.py classify, frozen in-line test) summed per arm over paired cells:
    which classes shrink from CT to CT2.
(2) Direct evidence of each CT2 part from the re-solve logs:
    readmit fired = a person id leaves the re-solve's dropped list; span = a line direction exists in CT2 but not in CT.
    Gain (share strict, CT2 - CT) in cells where readmit fired against cells where it did not."""
import collections, json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths
M = paths.MOVING_RUNS
sys.path.insert(0, str(paths.DESIGN)); sys.path.insert(0, str(paths.MOVING / 'failure-analysis'))
import summarise_fresh as S
import failures as F
GROUP = {"latency": "latency", "approach": "approach", "fa_noturn": "first answer", "ct_ran_out": "line ran out",
         "ct_ran_out_front": "line ran out", "hysteresis": "replace rule", "near_miss": "near miss"}
def grp(c):
    if c is None: return None
    if c.startswith("fa_"): return "first answer"
    return GROUP.get(c, "other re-solve" if c.startswith("ct_") else c)

def readmits(e):
    seen, n = set(), 0
    for r in e['resolves']:
        cur = set(r.get('dropped') or [])
        n += len(seen - cur)
        seen = cur
    return n
def has_dir(e): return sum(1 for r in e['resolves'] if r.get('line_direction')) / max(len(e['resolves']), 1)

def main():
    for root, label in ((M / 'runs/sonnet55_low', 'Sonnet seeds 10-19'), (M / 'runs/dsv41', 'DeepSeek seeds 10-14')):
        eps = {}
        for e in S.load([root]):
            if e['scenario'] in S.BREAK: continue
            e['_dir'] = e['_path']
            eps[(e['scenario'], e['seed'], e['arm'])] = e
        cells = sorted(k[:2] for k in eps if k[2] == 'CT2' and k[:2] + ('CT',) in eps
                       and eps[k]['answers'] and eps[k[:2] + ('CT',)]['answers'])
        sums = {a: collections.Counter() for a in ('CT', 'CT2')}
        rows = []
        for c in cells:
            for a in ('CT', 'CT2'):
                for cl in F.classify(eps[c + (a,)]):
                    sums[a][grp(cl) or 'in line'] += 1
            ct, ct2 = eps[c + ('CT',)], eps[c + ('CT2',)]
            rows.append(dict(cell=c, gain=S.measures(ct2)['share_strict'] - S.measures(ct)['share_strict'],
                             ra=readmits(ct2), ra_ct=readmits(ct), dir_ct=has_dir(ct), dir_ct2=has_dir(ct2)))
        n = len(cells)
        print(f'\n## {label}: {n} paired cells; samples per class (frozen in-line test), CT -> CT2 (difference)')
        keys = sorted(set(sums['CT']) | set(sums['CT2']), key=lambda k: -sums['CT'][k])
        for k in keys:
            print(f'  {k:16s} {sums["CT"][k]:5d} -> {sums["CT2"][k]:5d}  ({sums["CT2"][k] - sums["CT"][k]:+d})')
        fired = [r for r in rows if r['ra'] > 0]; nof = [r for r in rows if r['ra'] == 0]
        print(f'  readmit fired in CT2: {len(fired)}/{n} cells ({sum(r["ra"] for r in fired)} events); ids leaving dropped in CT (should be 0): {sum(r["ra_ct"] for r in rows)}')
        for name, sub in (('readmit fired', fired), ('no readmit', nof)):
            if sub:
                g = [r['gain'] for r in sub]
                print(f'    {name:14s} n {len(sub):2d}: CT2-CT share strict mean {statistics.mean(g):+.3f}, W/L/T {sum(x > 1e-9 for x in g)}/{sum(x < -1e-9 for x in g)}/{sum(abs(x) <= 1e-9 for x in g)}')
        by = collections.defaultdict(list)
        for r in rows: by[r['cell'][0]].append(r)
        for sc, rs in sorted(by.items()):
            print(f'    {sc:24s} gain {statistics.mean(r["gain"] for r in rs):+.3f}  readmit cells {sum(r["ra"] > 0 for r in rs)}/{len(rs)}  '
                  f'line direction share of re-solves CT {statistics.mean(r["dir_ct"] for r in rs):.2f} -> CT2 {statistics.mean(r["dir_ct2"] for r in rs):.2f}')


if __name__ == '__main__':
    main()
