"""Compare my rebuilt development arrays (verify/data/vdev_*) with the compute agent's data/dev.npz and dev.json,
record by record: rasters (exact; else counts of differing cells per channel), F (exact), excluded_m keys (exact),
place_ex, places (1e-9), frame, rho_room, C and P verdicts."""
import json
from pathlib import Path

import numpy as np

P = Path('results/2026-10-17-beyond-vlm/phase2/flow_pilot')
th = np.load(P / 'data' / 'dev.npz')
tj = {m['key']: m for m in json.loads((P / 'data' / 'dev.json').read_text())['records']}
tk = {str(k): i for i, k in enumerate(th['keys'])}
res = {}
diffs = []
for s in ('b', 'a'):
    res[s] = dict(n=0, raster_exact=0, raster_ch_cells={}, F_exact=0, ex_exact=0, place_ex_eq=0, places_eq=0, rho_eq=0)
res['frame_maxdiff'] = 0.0
res['C_eq'] = res['P_eq'] = res['order_eq'] = 0
for part in sorted((P / 'verify' / 'data').glob('vdev_*.npz')):
    z = np.load(part)
    meta = json.loads(part.with_suffix('.json').read_text())
    for j, m in enumerate(meta):
        key = m['key']
        i = tk[key]
        res['order_eq'] += int(i == m['idx'])
        res['frame_maxdiff'] = max(res['frame_maxdiff'], float(np.abs(z['frame'][j] - th['frame'][i]).max()))
        res['C_eq'] += int(m['C_rec'] == tj[key]['C_top1'])
        res['P_eq'] += int(m['P_top1'] == tj[key]['P_top1'])
        for s in ('b', 'a'):
            r = res[s]
            r['n'] += 1
            a, b = z[f'raster_{s}'][j], th[f'raster_{s}'][i]
            if np.array_equal(a, b):
                r['raster_exact'] += 1
            else:
                d = np.abs(a.astype(np.float64) - b)
                per = [int((d[c] > 1e-6).sum()) for c in range(7)]
                for c in range(7):
                    if per[c]:
                        r['raster_ch_cells'][c] = r['raster_ch_cells'].get(c, 0) + per[c]
                diffs.append((s, key, per, float(d.max())))
            o1, o2 = z[f'off_{s}'], th[f'off_{s}']
            F1, F2 = z[f'F_{s}'][o1[j]:o1[j + 1]], th[f'F_{s}'][o2[i]:o2[i + 1]]
            r['F_exact'] += int(F1.shape == F2.shape and np.array_equal(F1, F2))
            e1, e2 = z[f'ex_{s}'][o1[j]:o1[j + 1]], th[f'ex_{s}'][o2[i]:o2[i + 1]]
            r['ex_exact'] += int(e1.shape == e2.shape and np.array_equal(e1, e2))
            r['place_ex_eq'] += int(z[f'place_ex_{s}'][j] == th[f'place_ex_{s}'][i])
            r['places_eq'] += int(np.abs(z[f'places_{s}'][j] - th[f'places_{s}'][i]).max() <= 1e-9)
            r['rho_eq'] += int(abs(m[f'rho_{s}'] - tj[key][f'rho_{s}']) <= 1e-12)
out = dict(res, n_raster_diffs=len(diffs), diffs_head=diffs[:30])
(P / 'verify' / 'out' / 'cmp_dev.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
