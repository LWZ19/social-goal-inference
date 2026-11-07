"""Post hoc verifier diagnostic: f8 (people in the way to the target) in the waiting_line test region cells against
the training data of the waiting_line fold (all floor cells, and my subsample / the compute agent's subsample)."""
import json, sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1d/c5_gates/verify')
import kde_v as K
fresh = K.load('fresh')
T = K.V.parent / 'cache'
tidx = {r['key']: r for r in json.loads((T / 'index.json').read_text()) if r['kind'] == 'fresh'}
for sit in K.PEOPLE_SITS:
    test = [s for s in fresh if s.situation == sit]
    train = [s for s in fresh if s.situation != sit]
    reg = collections.Counter(); allf = collections.Counter(); allr = collections.Counter()
    mine_r = collections.Counter(); mine_n = collections.Counter(); th_r = collections.Counter(); th_n = collections.Counter()
    for s in test:
        reg.update(s.F[s.sets[0], 7].astype(int).tolist())
    for s in train:
        allf.update(s.F[:, 7].astype(int).tolist())
        allr.update(s.F[s.sets[0], 7].astype(int).tolist())
        mine_r.update(s.F[s.tr_reg[0], 7].astype(int).tolist()); mine_n.update(s.F[s.tr_non[0], 7].astype(int).tolist())
        z = np.load(T / tidx[f"seed{s.seed}/{s.meta['split']}/{s.name}"]['file'])
        th_r.update(s.F[z['trreg__0'], 7].astype(int).tolist()); th_n.update(s.F[z['trnon__0'], 7].astype(int).tolist())
    print(f'== fold leaving {sit} out: f8 value counts')
    print('  test region cells        ', dict(sorted(reg.items())))
    print('  training floor cells     ', dict(sorted(allf.items())))
    print('  training region cells    ', dict(sorted(allr.items())))
    print('  my subsample region/non  ', dict(sorted(mine_r.items())), dict(sorted(mine_n.items())))
    print('  their subsample region/non', dict(sorted(th_r.items())), dict(sorted(th_n.items())))
