"""Scorer for CT2_NORV_POSTHOC.md (sha256 165654e6), zero calls. Judges Q1-Q3 exactly as written."""
import statistics, sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths
M = paths.MOVING_RUNS
sys.path.insert(0, str(paths.DESIGN)); sys.path.insert(0, str(paths.MOVING / 'failure-analysis')); sys.path.insert(0, str(Path(__file__).parent))
import summarise_fresh as S, failures as F
from ct2_attrib import grp, readmits
from ct2_attrib2 import load
A = 'CT2-norv'
Q1 = {'Sonnet': 301, 'DeepSeek': 155}; Q2 = {'Sonnet': -0.042, 'DeepSeek': -0.078}; Q3 = {'Sonnet': 0.082, 'DeepSeek': 0.078}
def ranout(e): return sum(grp(c) == 'line ran out' for c in F.classify(e))
def sh(e): return S.measures(e)['share_strict']
for root, label in ((M / 'runs/sonnet55_low', 'Sonnet'), (M / 'runs/dsv41', 'DeepSeek')):
    eps = load(root)
    cells = sorted(k[:2] for k in eps if k[2] == 'CT2' and k[:2] + ('CT',) in eps)
    have = [c for c in cells if c + (A,) in eps]
    print(f'\n## {label}: {A} present in {len(have)}/{len(cells)} cells')
    if not have: continue
    ra = [c for c in have if readmits(eps[c + ('CT2',)])]
    ro = {a: sum(ranout(eps[c + (a,)]) for c in ra) for a in ('CT', 'CT2', 'CT2-nora', A) if all(c + (a,) in eps for c in ra)}
    d = [sh(eps[c + (A,)]) - sh(eps[c + ('CT2',)]) for c in ra]
    w, l = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
    print(f'  readmit cells {len(ra)}: ran out {ro}')
    print(f'    Q1 {A} ran out >= {Q1[label]}: {ro.get(A)} -> {"PASS" if ro.get(A, 0) >= Q1[label] else "FAIL"}')
    print(f'    Q2 share vs CT2 <= {Q2[label]} and losses > wins: mean {statistics.mean(d):+.3f}, W/L {w}/{l} -> '
          f'{"PASS" if statistics.mean(d) <= Q2[label] and l > w else "FAIL"}')
    kept = statistics.mean(sh(eps[c + (A,)]) - sh(eps[c + ('CT',)]) for c in have)
    print(f'    Q3 {A} - CT over all cells {kept:+.3f}, less than half of {Q3[label]:+.3f}? -> {"PASS" if kept < Q3[label] / 2 else "FAIL"}')
