"""Post hoc, zero calls: does the headline (CT / CT2 against R1) depend on the 1.2 m band top or the side check?
Share under band tops 1.2 / 1.5 / 2.0 m, with and without the strict side check, paired W/L."""
import statistics, sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths
M = paths.MOVING_RUNS
sys.path.insert(0, str(paths.DESIGN)); sys.path.insert(0, str(paths.MOVING / 'failure-analysis')); sys.path.insert(0, str(Path(__file__).parent))
import summarise_fresh as S, failures as F
from ct2_attrib2 import load
ARMS = ('R1', 'CT', 'CT2')
def ok(x, hi, side):
    if not F.rescore(x, band_hi=hi, first_m=0.6): return False
    return (not side) or x['ahead'] == 0 or x['side'] <= 0.6
for root, label in ((M / 'runs/sonnet55_low', 'Sonnet'), (M / 'runs/dsv41', 'DeepSeek')):
    eps = load(root)
    cells = [k[:2] for k in eps if k[2] == 'CT2' and all(k[:2] + (a,) in eps for a in ARMS)]
    print(f'\n{label} ({len(cells)} cells)')
    for side in (True, False):
        for hi in (1.2, 1.5, 2.0):
            v = {a: [sum(ok(x, hi, side) for x in eps[c + (a,)]['samples']) / len(eps[c + (a,)]['samples']) for c in cells] for a in ARMS}
            def wl(a, b):
                d = [x - y for x, y in zip(v[a], v[b])]
                return f'{sum(x > 1e-9 for x in d)}/{sum(x < -1e-9 for x in d)} {statistics.mean(d):+.3f}'
            print(f'  side check {"on " if side else "off"} band {hi:.1f}: ' + ', '.join(f'{a} {statistics.mean(v[a]):.3f}' for a in ARMS) +
                  f' | CT2 vs R1 {wl("CT2", "R1")} | CT vs R1 {wl("CT", "R1")} | CT2 vs CT {wl("CT2", "CT")}')
