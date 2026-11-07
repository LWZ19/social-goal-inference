"""Post hoc, zero calls: is the ratchet's benefit an artefact of the 0.5-1.2 m band? Share in line (frozen-style test
with the strict side check) under band tops 1.2 / 1.5 / 2.0 m, plus band-free behaviour: the gap to the person ahead
while the robot is behind someone, on the line axis (side <= 0.6 m) and joined, for CT, CT2, CT2-rat."""
import statistics, sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths
M = paths.MOVING_RUNS
sys.path.insert(0, str(paths.DESIGN)); sys.path.insert(0, str(paths.MOVING / 'failure-analysis')); sys.path.insert(0, str(Path(__file__).parent))
import summarise_fresh as S, failures as F
from ct2_attrib2 import load
ARMS = ('CT', 'CT2', 'CT2-rat')
def ok(x, hi):
    if not F.rescore(x, band_hi=hi, first_m=0.6): return False
    return x['ahead'] == 0 or x['side'] <= 0.6
for root, label in ((M / 'runs/sonnet55_low', 'Sonnet'), (M / 'runs/dsv41', 'DeepSeek')):
    eps = load(root)
    cells = [k[:2] for k in eps if k[2] == 'CT2' and all(k[:2] + (a,) in eps for a in ARMS)]
    print(f'\n{label} ({len(cells)} cells)')
    for hi in (1.2, 1.5, 2.0):
        sh = {a: statistics.mean(sum(ok(x, hi) for x in eps[c + (a,)]['samples']) / len(eps[c + (a,)]['samples']) for c in cells) for a in ARMS}
        print(f'  band top {hi:.1f} m: share ' + ', '.join(f'{a} {sh[a]:.3f}' for a in ARMS) +
              f'   CT2 - rat {sh["CT2"] - sh["CT2-rat"]:+.3f}, rat - CT {sh["CT2-rat"] - sh["CT"]:+.3f}')
    for a in ARMS:
        g = [x['gap_m'] for c in cells for x in eps[c + (a,)]['samples']
             if x['ahead'] and x.get('gap_m') is not None and x['joined'] and x['side'] <= 0.6 and x['behind_m'] >= 0]
        if g:
            q = sorted(g)
            print(f'  {a:8s} gap behind the person ahead (joined, on axis; {len(g)} samples): median {statistics.median(g):.2f} m, '
                  f'p75 {q[int(.75 * len(q))]:.2f}, p90 {q[int(.9 * len(q))]:.2f}; share in 1.2-1.5 m {sum(1.2 < x <= 1.5 for x in g) / len(g):.2f}, over 1.5 m {sum(x > 1.5 for x in g) / len(g):.2f}')
