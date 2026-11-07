"""Compare my cell/feature cache with the compute agent's cache (read-only comparison)."""
import json
from pathlib import Path
import numpy as np
V = Path('results/2026-10-17-beyond-vlm/phase1d/c5_gates/verify')
C = V.parent / 'cache'
mine = json.loads((V / 'vcache' / 'index.json').read_text())
theirs = json.loads((C / 'index.json').read_text())
print(theirs[0].keys())
tk = {}
for r in theirs:
    tk[(r['kind'], r['scene_id'], r.get('seed'))] = r
maxdiff = np.zeros(9); nmis = 0; nsc = 0; regmis = 0
for kind in ('fresh', 'dev'):
    for m in mine[kind]:
        key = (kind, m['scene_id'], m['seed'])
        if key not in tk:
            # dev keys may differ
            cands = [r for r in theirs if r['kind'] == kind and r['scene_id'] == m['scene_id']]
            if len(cands) != 1:
                print('no match', key, len(cands)); continue
            r = cands[0]
        else:
            r = tk[key]
        zm = np.load(V / m['file']); zt = np.load(C / r['file'])
        Fm = zm['F']
        Ft = [zt[k] for k in zt.files if k.startswith('F__')]
        assert len(Ft) == 1
        Ft = Ft[0]
        nsc += 1
        if Fm.shape != Ft.shape:
            nmis += 1; print('shape', key, Fm.shape, Ft.shape); continue
        d = np.abs(Fm - Ft).max(0)
        maxdiff = np.maximum(maxdiff, d)
        regs = [zt[k] for k in zt.files if k.startswith('reg__')]
        mm = np.nonzero(zm['masks'][0])[0]
        for rg in regs:
            if not np.array_equal(np.sort(rg), mm):
                regmis += 1
print('scenes compared', nsc, 'shape mismatches', nmis, 'region-set mismatches', regmis)
print('max |diff| per feature', maxdiff)
