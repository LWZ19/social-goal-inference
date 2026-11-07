"""V3 density hit@k with float-distance ranking instead of exact integer grid distance (verifier sensitivity)."""
import sys
import numpy as np
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1d/i04/verify')
from v_analysis import load, PRIMARY, NEW
rows, *_ = load()
for c in PRIMARY + ['POOLED'] + NEW:
    rr = [r for r in rows if (r['cell'] in PRIMARY if c == 'POOLED' else r['cell'] == c)]
    a = np.sum([r['dens'][('V3', None)]['hit'] for r in rr], axis=0).tolist()
    b = np.sum([r['dens'][('V3f', None)]['hit'] for r in rr], axis=0).tolist()
    print(f'{c:15s} n {len(rr):4d} integer {a} float {b}')
