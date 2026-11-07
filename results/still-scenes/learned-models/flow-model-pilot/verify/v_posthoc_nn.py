"""POST HOC (decides nothing): nearest training raster to each development raster (setting b), L1 over the floor and
people channels on a 4x-downsampled copy, by cell; and by validation unit for comparison."""
import json
from pathlib import Path
import numpy as np
P = Path('results/still-scenes/learned-models/flow-model-pilot')
tr = np.load(P / 'data' / 'train.npz'); va = np.load(P / 'data' / 'val.npz'); dv = np.load(P / 'data' / 'dev.npz')
meta = {m['key']: m for m in json.loads((P / 'data' / 'dev.json').read_text())['records']}
f = lambda R: R[:, :2, ::4, ::4].reshape(len(R), -1).astype(np.float32)
T = f(tr['rasters'])
def nn(Q):
    out = []
    for i in range(0, len(Q), 4):
        out += list(np.abs(Q[i:i + 4, None, :] - T[None]).sum(-1).min(1))
    return np.array(out)
dmin = nn(f(dv['raster_b'])); vmin = nn(f(va['rasters']))
res = {'val (generator seeds 190-199)': dict(n=len(vmin), median=float(np.median(vmin)), p90=float(np.percentile(vmin, 90)))}
cells = np.array([meta[str(k)]['cell'] for k in dv['keys']])
for c in sorted(set(cells)):
    x = dmin[cells == c]
    res[c] = dict(n=int(len(x)), median=float(np.median(x)), p90=float(np.percentile(x, 90)))
(P / 'verify' / 'out' / 'posthoc_nn.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
