"""Compare my rebuilt seed arrays (verify/data/vseed<k>.npz) with the compute agent's (data/seeds/seed<k>.npz and the
joined data/train.npz / data/val.npz rows with the same keys). Exact for rasters (float32), labels within 1e-6,
coverage lattices as sets (rounded to 1e-6 m)."""
import json, sys
from pathlib import Path
import numpy as np
P = Path('results/2026-10-17-beyond-vlm/phase2/flow_pilot')
out = {}
joined = {}
for name in ('train', 'val'):
    z = np.load(P / 'data' / f'{name}.npz')
    joined[name] = (z, {str(k): i for i, k in enumerate(z['keys'])})
tot = dict(units=0, raster_exact=0, raster_maxdiff=0.0, raster_cells_diff=0, labels_ok=0, haslabel_ok=0, ext_same=0,
           joined_raster_exact=0, joined_label_ok=0, ch_diff={})
bad = []
for a in sys.argv[1:]:
    k = int(a)
    mine = np.load(P / 'verify' / 'data' / f'vseed{k}.npz')
    theirs = np.load(P / 'data' / 'seeds' / f'seed{k}.npz')
    assert list(mine['keys']) == list(theirs['keys']), k
    jz, jidx = joined['train' if k < 190 else 'val']
    for i, key in enumerate(mine['keys']):
        key = str(key)
        tot['units'] += 1
        a_, b_ = mine['rasters'][i], theirs['rasters'][i]
        d = np.abs(a_.astype(np.float64) - b_)
        if d.max() == 0:
            tot['raster_exact'] += 1
        else:
            tot['raster_maxdiff'] = max(tot['raster_maxdiff'], float(d.max()))
            for ch in range(7):
                n = int((d[ch] > 1e-6).sum())
                if n:
                    tot['ch_diff'][ch] = tot['ch_diff'].get(ch, 0) + n
            tot['raster_cells_diff'] += int((d > 1e-6).sum())
            bad.append((key, [int((d[ch] > 1e-6).sum()) for ch in range(7)], float(d.max())))
        tot['haslabel_ok'] += int(mine['has_label'][i] == theirs['has_label'][i])
        if np.abs(mine['labels'][i] - theirs['labels'][i]).max() <= 1e-6:
            tot['labels_ok'] += 1
        e1 = mine['ext_xy'][mine['ext_off'][i]:mine['ext_off'][i + 1]]
        e2 = theirs['ext_xy'][theirs['ext_off'][i]:theirs['ext_off'][i + 1]]
        s1 = {tuple(np.round(r, 6)) for r in e1}
        s2 = {tuple(np.round(r, 6)) for r in e2}
        tot['ext_same'] += int(s1 == s2)
        j = jidx[key]
        tot['joined_raster_exact'] += int(np.array_equal(jz['rasters'][j], a_))
        tot['joined_label_ok'] += int(np.abs(jz['labels'][j] - mine['labels'][i]).max() <= 1e-6)
print(json.dumps(tot))
for b in bad[:20]:
    print('DIFF', b)
