"""Post hoc, zero calls: Sonnet cells where readmit never fired: CT2 - CT by scenario and which classes shrink."""
import collections, statistics, sys
from pathlib import Path
M = Path('results/2026-11-01-push/moving')
sys.path.insert(0, str(M)); sys.path.insert(0, str(M / 'analysis')); sys.path.insert(0, str(Path(__file__).parent))
import summarise_fresh as S, failures as F
from ct2_attrib import grp, readmits, has_dir
from ct2_attrib2 import load
for root, label in ((M / 'runs/sonnet55_low', 'Sonnet'), (M / 'runs/dsv41', 'DeepSeek')):
    eps = load(root)
    by = collections.defaultdict(lambda: [[], collections.Counter()])
    for k, e in eps.items():
        if k[2] != 'CT2' or k[:2] + ('CT',) not in eps or readmits(e): continue
        ct = eps[k[:2] + ('CT',)]
        b = by[k[0]]
        b[0].append(S.measures(e)['share_strict'] - S.measures(ct)['share_strict'])
        for c in F.classify(ct): b[1][grp(c) or 'in line'] -= 1
        for c in F.classify(e): b[1][grp(c) or 'in line'] += 1
    print(f'\n{label}, cells with no readmit: CT2 - CT')
    for sc, (g, c) in sorted(by.items()):
        top = ', '.join(f'{k} {v:+d}' for k, v in sorted(c.items(), key=lambda kv: kv[1])[:3])
        print(f'  {sc:24s} n {len(g):2d} share {statistics.mean(g):+.3f} W/L/T {sum(x>1e-9 for x in g)}/{sum(x<-1e-9 for x in g)}/{sum(abs(x)<=1e-9 for x in g)}  in line {c["in line"]:+d}; biggest falls: {top}')
