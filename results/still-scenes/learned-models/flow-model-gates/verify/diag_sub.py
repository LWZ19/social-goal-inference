"""Verifier diagnostics (post hoc, decide nothing): why my primary-fold S_ratio differs from the compute agent's.

    python diag_sub.py their <situation> <seed,seed,...>
        Primary fold leaving <situation> out, trained on the compute agent's OWN training subsample indices (read from
        its cache npz files, data only), FS3 m=0.5 as both inner CVs chose; test on that fold's scenes of the listed
        seeds. Compared per episode with its rows_gate2.jsonl. Also my own subsample on the same scenes.
    python diag_sub.py draw <salt>
        All three primary folds with a fresh training subsample (rng key [20261018, crc32(key), salt]); configuration
        fixed at FS3 m=0.5 (the choice of every inner CV, both mine and the compute agent's); ratio and band only.
"""
from __future__ import annotations

import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, 'results/still-scenes/learned-models/flow-model-gates/verify')
import kde_v as K  # noqa: E402

V = K.V
THEIR = V.parent / 'cache'


def rows_for(model, test, fold):
    cfg = [(K.FS['FS3'], model.h(8, 0.5))]
    lo, hi = model.band
    out = []
    for s in test:
        LN, LD = model.logs(s.F, cfg)
        lr = K.logq(LN[0] - LD[0], s.N)
        dt = s.F[:, 0]
        B = (dt >= lo) & (dt <= hi)
        lb = np.log(0.99 * B / B.sum() + 0.01 / s.N)
        for e in s.eps:
            R = s.sets[e['set']]
            out.append({'fold': fold, 'scene': s.name, 'seed': s.seed, 'split': s.meta['split'], 'episode': e['episode'],
                        'situation': s.situation, 'S_ratio': float(lr[R].mean()), 'S_band': float(lb[R].mean()),
                        'M_ratio': float(np.exp(lr[R]).sum()), 'n_region': int(R.size), 'N': s.N,
                        'frac_floor_like': float(np.mean(lr[R] < math.log(0.02 / s.N)))})
    return out


def main():
    mode = sys.argv[1]
    fresh = K.load('fresh')
    if mode == 'their':
        sit = sys.argv[2]
        seeds = [int(x) for x in sys.argv[3].split(',')]
        tidx = {r['key']: r for r in json.loads((THEIR / 'index.json').read_text()) if r['kind'] == 'fresh'}
        train = [s for s in fresh if s.situation != sit]
        test = [s for s in fresh if s.situation == sit and s.seed in seeds]
        mine_model = K.Model(train)
        mine_rows = rows_for(mine_model, test, f'LOSO_{sit}')
        for s in train:
            r = tidx[f"seed{s.seed}/{s.meta['split']}/{s.name}"]
            z = np.load(THEIR / r['file'])
            s.tr_reg = [np.sort(z['trreg__0'])]
            s.tr_non = [np.sort(z['trnon__0'])]
        their_model = K.Model(train)
        their_rows = rows_for(their_model, test, f'LOSO_{sit}')
        ref = {}
        for line in (V.parent / 'rows_gate2.jsonl').read_text().splitlines():
            t = json.loads(line)
            if t['fold'] == f'LOSO_{sit}':
                ref[(t['key'], t['eid'])] = t
        L = [f'LOSO_{sit}, test seeds {seeds}: n_r mine {mine_model.n_r}, with their subsample {their_model.n_r}; '
             f'band mine {mine_model.band}, theirs-subsample {their_model.band}']
        d_their, d_mine, d_band = [], [], []
        for a, b in zip(their_rows, mine_rows):
            t = ref[(f"seed{a['seed']}/{a['split']}/{a['scene']}", a['episode'])]
            d_their.append(a['S_ratio'] - t['ratio']['S'])
            d_mine.append(b['S_ratio'] - t['ratio']['S'])
            d_band.append(a['S_band'] - t['band']['S'])
            L.append(f"  {a['scene']:32s} {a['episode']}  theirs {t['ratio']['S']:9.4f}  me+their-subsample "
                     f"{a['S_ratio']:9.4f}  me+my-subsample {b['S_ratio']:9.4f}  floor-like share their/mine "
                     f"{a['frac_floor_like']:.3f}/{b['frac_floor_like']:.3f}")
        L.append(f'  max |me with their subsample - theirs| S_ratio {np.max(np.abs(d_their)):.2e}; S_band '
                 f'{np.max(np.abs(d_band)):.2e}; mean (my subsample - theirs) {np.mean(d_mine):+.3f} (n {len(d_mine)})')
        txt = '\n'.join(L)
        print(txt)
        (V / f'diag_their_{sit}.txt').write_text(txt + '\n')
    elif mode == 'draw':
        salt = int(sys.argv[2])
        for s in fresh:
            key = f"{s.seed}/{s.meta['split']}/{s.name}"
            rng = np.random.default_rng([20261018, zlib.crc32(key.encode()), salt])
            R = s.sets[0]
            non = np.setdiff1d(np.arange(s.N), R)
            s.tr_reg = [np.sort(rng.choice(R, size=min(100, R.size), replace=False))]
            s.tr_non = [np.sort(rng.choice(non, size=min(200, non.size), replace=False))]
        out = {'salt': salt, 'rows': []}
        for sit in K.PEOPLE_SITS:
            t0 = time.time()
            train = [s for s in fresh if s.situation != sit]
            test = [s for s in fresh if s.situation == sit]
            model = K.Model(train)
            rows = rows_for(model, test, f'LOSO_{sit}')
            out['rows'].extend(rows)
            d = np.mean([r['S_ratio'] - r['S_band'] for r in rows])
            K.log(f'salt {salt} LOSO_{sit}: S_ratio {np.mean([r["S_ratio"] for r in rows]):.3f} Delta {d:+.3f} '
                  f'(n {len(rows)}), {time.time() - t0:.0f} s')
            (V / f'out_draw_{salt}.json').write_text(json.dumps(out))


if __name__ == '__main__':
    main()
