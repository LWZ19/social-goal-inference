"""Check the powering path of kde_v.Model.logs against one exp per config (all 16 grid configs, 40 points)."""
import sys
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1d/c5_gates/verify')
import numpy as np
import kde_v as K
fresh = K.load('fresh')
tr = [s for s in fresh if s.seed != 101]
te = [s for s in fresh if s.seed == 101]
m = K.Model(tr)
cfgs, names = K.grid_configs(m)
for s in te[::7]:
    idx = np.concatenate([s.ev_flo[:20], s.ev_reg[0][:20]])
    LN, LD = m.logs(s.F[idx], cfgs)
    worst = 0
    for ci, c in enumerate(cfgs):
        ln, ld = m.logs(s.F[idx], [c])
        worst = max(worst, np.abs(ln[0] - LN[ci]).max(), np.abs(ld[0] - LD[ci]).max())
    print(s.name, 'max |powered - direct| over 16 configs', worst)
