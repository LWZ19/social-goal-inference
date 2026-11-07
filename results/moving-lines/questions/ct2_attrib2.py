"""Post hoc, zero calls: failure classes on Sonnet seeds 10-12 (15 cells) for CT, CT2, CT2-vac, CT2-rat and the
two repeats; plus 'line ran out' CT -> CT2 split by whether readmit fired, both models."""
import collections, json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths
M = paths.MOVING_RUNS
sys.path.insert(0, str(paths.DESIGN)); sys.path.insert(0, str(paths.MOVING / 'failure-analysis')); sys.path.insert(0, str(Path(__file__).parent))
import summarise_fresh as S, failures as F
from ct2_attrib import grp, readmits
def load(root):
    out = {}
    for e in S.load([root]):
        if e['scenario'] in S.BREAK or not e['answers']: continue
        e['_dir'] = e['_path']; out[(e['scenario'], e['seed'], e['arm'])] = e
    return out
def main():
    son, rep = load(M / 'runs/sonnet55_low'), load(M / 'runs/sonnet55_low_fresh_repeat')
    cells = sorted({k[:2] for k in son if k[1] in (10, 11, 12) and k[2] == 'CT2-vac'})
    cols = [('CT', son, 'CT'), ('CT rep', rep, 'CT'), ('CT2', son, 'CT2'), ('CT2 rep', rep, 'CT2'), ('CT2-vac', son, 'CT2-vac'), ('CT2-rat', son, 'CT2-rat')]
    sums = {}
    for name, src, arm in cols:
        c = collections.Counter()
        for cell in cells:
            if cell + (arm,) in src:
                for cl in F.classify(src[cell + (arm,)]): c[grp(cl) or 'in line'] += 1
        sums[name] = c
    keys = ['in line', 'line ran out', 'replace rule', 'other re-solve', 'approach', 'latency', 'first answer']
    print(f'Sonnet seeds 10-12, {len(cells)} cells: samples per class')
    print(f'{"class":16s}' + ''.join(f'{n:>9s}' for n, *_ in cols))
    for k in keys + ['rest']:
        vals = [(sum(v for kk, v in sums[n].items() if kk not in keys) if k == 'rest' else sums[n][k]) for n, *_ in cols]
        print(f'{k:16s}' + ''.join(f'{v:9d}' for v in vals))
    print('\nline ran out, CT -> CT2, by whether readmit fired in CT2')
    for root, label in ((M / 'runs/sonnet55_low', 'Sonnet'), (M / 'runs/dsv41', 'DeepSeek')):
        eps = load(root)
        acc = collections.defaultdict(lambda: [0, 0, 0])
        for k, e in eps.items():
            if k[2] != 'CT2' or k[:2] + ('CT',) not in eps: continue
            tag = 'readmit fired' if readmits(e) else 'no readmit'
            a = acc[tag]; a[2] += 1
            a[0] += sum(grp(c) == 'line ran out' for c in F.classify(eps[k[:2] + ('CT',)]))
            a[1] += sum(grp(c) == 'line ran out' for c in F.classify(e))
        for tag, (a, b, n) in sorted(acc.items()):
            print(f'  {label:8s} {tag:14s} cells {n:2d}: {a} -> {b} ({b - a:+d})')


if __name__ == '__main__':
    main()
