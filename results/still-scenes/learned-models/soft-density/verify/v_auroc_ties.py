"""Sensitivity of the confidence AUROCs to float noise in c near 1 (verifier; post hoc check of numerics only)."""
import json, pickle, sys
import numpy as np
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1d/i04/verify')
from v_analysis import load, auroc, GAMMAS, PRIMARY, NEW
V = 'results/2026-10-17-beyond-vlm/phase1d/i04/verify'
C = 'results/2026-10-17-beyond-vlm/phase1d/i04'
rows, *_ = load()
fits = json.load(open(f'{V}/v_summary.json'))['fits']
theirs = {}
for c in PRIMARY + NEW:
    for line in open(f'{C}/rows_{c}.jsonl'):
        d = json.loads(line); theirs[d['key']] = d
for c in PRIMARY + NEW:
    rr = [r for r in rows if r['cell'] == c]
    y = np.array([r['top1'] for r in rr])
    out = []
    for v in ('V1', 'V2', 'V3'):
        if v == 'V3':
            mine = np.array([r['out']['V3']['conc'][fits[r['layout']]['V3'][0]] for r in rr])
        else:
            mine = np.array([r['out'][v]['conc'][fits[r['layout']][v][0], fits[r['layout']][v][1]] for r in rr])
        th = np.array([theirs[r['key']][v]['conc'] for r in rr])
        out.append(f"{v} mine {auroc(mine, y):.3f} theirs {auroc(th, y):.3f} | rounded 1e-12: mine {auroc(np.round(mine, 12), y):.3f} theirs {auroc(np.round(th, 12), y):.3f} | n(c==1 exactly) {int((mine == 1).sum())}/{int((th == 1).sum())}")
    print(f'{c:15s} ' + ' || '.join(out))
