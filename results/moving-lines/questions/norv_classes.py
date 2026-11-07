"""Post hoc, zero calls: failure classes in CT2's readmit cells for CT, CT2, CT2-nora, CT2-norv (where did the returning
'line ran out' time come from), and CT2-norv / CT2-rat against CT by scenario."""
import collections, statistics, sys
from pathlib import Path
M = Path('results/2026-11-01-push/moving')
sys.path.insert(0, str(M)); sys.path.insert(0, str(M / 'analysis')); sys.path.insert(0, str(Path(__file__).parent))
import summarise_fresh as S, failures as F
from ct2_attrib import grp, readmits
from ct2_attrib2 import load
ARMS = ('CT', 'CT2', 'CT2-nora', 'CT2-norv')
for root, label in ((M / 'runs/sonnet55_low', 'Sonnet'), (M / 'runs/dsv41', 'DeepSeek')):
    eps = load(root)
    cells = [k[:2] for k in eps if k[2] == 'CT2' and all(k[:2] + (a,) in eps for a in ARMS)]
    for tag, sub in (('readmit cells', [c for c in cells if readmits(eps[c + ('CT2',)])]), ('all cells', cells)):
        sums = {a: collections.Counter(grp(x) or 'in line' for c in sub for x in F.classify(eps[c + (a,)])) for a in ARMS}
        keys = sorted(set().union(*sums.values()), key=lambda k: -sums['CT'][k])[:9]
        print(f'\n{label}, {tag} ({len(sub)}): samples per class')
        print(f'  {"class":16s}' + ''.join(f'{a:>10s}' for a in ARMS))
        for k in keys:
            print(f'  {k:16s}' + ''.join(f'{sums[a][k]:10d}' for a in ARMS))
    by = collections.defaultdict(list)
    for c in cells:
        by[c[0]].append(tuple(S.measures(eps[c + (a,)])['share_strict'] for a in ARMS))
    print(f'\n{label}: mean share strict by scenario  ' + ' / '.join(ARMS))
    for sc, v in sorted(by.items()):
        print(f'  {sc:24s} ' + ' / '.join(f'{statistics.mean(x[i] for x in v):.3f}' for i in range(len(ARMS))))
