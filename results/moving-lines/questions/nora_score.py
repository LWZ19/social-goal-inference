"""Scorer for CT2_NORA_POSTHOC.md (sha256 3d428ebe), zero calls. Readmit split from CT2's own logs (a person id
leaving CT2's dropped list). Judges P1-P4 exactly as written."""
import collections, statistics, sys
from pathlib import Path
M = Path('results/2026-11-01-push/moving')
sys.path.insert(0, str(M)); sys.path.insert(0, str(M / 'analysis')); sys.path.insert(0, str(Path(__file__).parent))
import summarise_fresh as S, failures as F
from ct2_attrib import grp, readmits
from ct2_attrib2 import load
BAR = {'Sonnet': (385, 217), 'DeepSeek': (220, 89)}       # CT, CT2 ran-out samples in readmit cells (frozen note)
def ranout(e): return sum(grp(c) == 'line ran out' for c in F.classify(e))
def sh(e): return S.measures(e)['share_strict']
for root, label in ((M / 'runs/sonnet55_low', 'Sonnet'), (M / 'runs/dsv41', 'DeepSeek')):
    eps = load(root)
    cells = sorted(k[:2] for k in eps if k[2] == 'CT2' and k[:2] + ('CT',) in eps)
    have = [c for c in cells if c + ('CT2-nora',) in eps]
    print(f'\n## {label}: CT2-nora present in {len(have)}/{len(cells)} cells')
    if not have: continue
    for tag in ('readmit', 'no readmit'):
        sub = [c for c in have if bool(readmits(eps[c + ('CT2',)])) == (tag == 'readmit')]
        if not sub: continue
        ro = {a: sum(ranout(eps[c + (a,)]) for c in sub) for a in ('CT', 'CT2', 'CT2-nora')}
        d = [sh(eps[c + ('CT2-nora',)]) - sh(eps[c + ('CT2',)]) for c in sub]
        dct = [sh(eps[c + ('CT2-nora',)]) - sh(eps[c + ('CT',)]) for c in sub]
        w, l = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
        print(f'  {tag:10s} cells {len(sub):2d}: ran out CT {ro["CT"]} / CT2 {ro["CT2"]} / CT2-nora {ro["CT2-nora"]}; '
              f'share nora-CT2 mean {statistics.mean(d):+.3f} mean|d| {statistics.mean(abs(x) for x in d):.3f} W/L/T {w}/{l}/{len(d)-w-l}; '
              f'nora-CT mean {statistics.mean(dct):+.3f}')
        if tag == 'readmit':
            ct, ct2 = BAR[label]
            bar = (ct + ct2) / 2
            full = len(sub) == {'Sonnet': 25, 'DeepSeek': 15}[label]
            print(f'    P1 ran out >= {bar:.0f} (halfway; frozen on the full set): {ro["CT2-nora"]} -> '
                  f'{"PASS" if ro["CT2-nora"] >= bar else "FAIL"}{"" if full else "  (partial set: not judged)"}')
            if label == 'DeepSeek':
                print(f'    P2 share drop >= 0.07 and losses > wins: mean {statistics.mean(d):+.3f}, {l} losses / {w} wins -> '
                      f'{"PASS" if statistics.mean(d) <= -0.07 and l > w else "FAIL"}')
            else:
                print(f'    P3 |mean drop| < 0.07: {abs(statistics.mean(d)):.3f} -> {"PASS" if abs(statistics.mean(d)) < 0.07 else "FAIL"}')
        else:
            m, a = statistics.mean(d), statistics.mean(abs(x) for x in d)
            print(f'    P4 mean within +-0.04 and mean|d| <= 0.10: {m:+.3f}, {a:.3f} -> {"PASS" if abs(m) <= 0.04 and a <= 0.10 else "FAIL"}')
