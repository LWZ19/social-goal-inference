"""Post hoc: in the waiting_line fold, how many sampled training points have f8 >= 3, per subsample draw."""
import json, sys, zlib
import numpy as np
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1d/c5_gates/verify')
import kde_v as K
fresh = K.load('fresh')
train = [s for s in fresh if s.situation != 'waiting_line']
T = K.V.parent / 'cache'
tidx = {r['key']: r for r in json.loads((T / 'index.json').read_text()) if r['kind'] == 'fresh'}
def count(get):
    nr = nn = 0
    for s in train:
        r, n = get(s)
        nr += int((s.F[r, 7] >= 3).sum()); nn += int((s.F[n, 7] >= 3).sum())
    return nr, nn
print('verifier main draw (region, non-region with f8>=3):', count(lambda s: (s.tr_reg[0], s.tr_non[0])))
def theirs(s):
    z = np.load(T / tidx[f"seed{s.seed}/{s.meta['split']}/{s.name}"]['file']); return z['trreg__0'], z['trnon__0']
print('compute agent draw:', count(theirs))
for salt in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
    def g(s, salt=salt):
        rng = np.random.default_rng([20261018, zlib.crc32(f"{s.seed}/{s.meta['split']}/{s.name}".encode()), salt])
        R = s.sets[0]; non = np.setdiff1d(np.arange(s.N), R)
        return rng.choice(R, size=min(100, R.size), replace=False), rng.choice(non, size=min(200, non.size), replace=False)
    print(f'salt {salt}:', count(g))
