"""Independent verifier, C5 gate 2: ratio KDE, plain KDE, 1-D ratio KDE, band-uniform baseline, inner CV and outer
folds (FROZEN.md section 2). Reads only verify/vcache (written by cells_v.py). numpy only, float64.

    python kde_v.py primary | ceiling | dev | bench
Each stage writes verify/out_<stage>.json."""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np

V = Path('results/2026-10-17-beyond-vlm/phase1d/c5_gates/verify')
FROZEN = V.parent / 'FROZEN.md'
SHA = '8b7d3ae294ad81a810c70c23706247674bbfed22cc0247445e91f6615426419c'
assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == SHA
CACHE = V / 'vcache'
SEEDS = list(range(100, 110))
PEOPLE_SITS = ['waiting_line', 'conversation_group', 'audience']
FS = {'FS1': [0, 1, 2, 3, 4, 5], 'FS2': [0, 1, 2, 3, 4, 5, 6], 'FS3': [0, 1, 2, 3, 4, 5, 7, 8]}
FS_ORDER = ['FS1', 'FS2', 'FS3']
MS = [0.5, 1.0, 2.0, 4.0]
LOG99, LOG01 = math.log(0.99), math.log(0.01)
CHUNK = 16


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ------------------------------------------------------------------------------------------------ data
class Scene:
    __slots__ = ('kind', 'meta', 'seed', 'situation', 'domain', 'F', 'N', 'sets', 'set_w', 'eps', 'tr_reg', 'tr_non',
                 'ev_reg', 'ev_flo', 'name')


def load(kind):
    index = json.loads((CACHE / 'index.json').read_text())
    out = []
    for meta in index[kind]:
        z = np.load(V / meta['file'])
        s = Scene()
        s.kind, s.meta, s.seed = kind, meta, meta['seed']
        s.domain = meta['domain']
        s.F = z['F']
        s.N = int(meta['N_floor'])
        masks = z['masks']
        el = [e for e in meta['episodes'] if e['eligible']]
        sits = {e['situation'] for e in el}
        assert len(sits) <= 1
        s.situation = sits.pop() if sits else None
        s.eps = el
        s.sets = [np.nonzero(m)[0] for m in masks]
        s.set_w = [sum(1 for e in el if e['set'] == k) for k in range(len(s.sets))]
        s.name = meta['scene_id']
        if kind == 'fresh':
            key = f"{meta['seed']}/{meta['split']}/{meta['scene_id']}"
            rng = np.random.default_rng([20261018, zlib.crc32(key.encode())])
            s.tr_reg, s.tr_non, s.ev_reg = [], [], []
            for R in s.sets:                     # one set per fresh scene (checked by cells_v)
                non = np.setdiff1d(np.arange(s.N), R)
                s.tr_reg.append(np.sort(rng.choice(R, size=min(100, R.size), replace=False)))
                s.tr_non.append(np.sort(rng.choice(non, size=min(200, non.size), replace=False)))
            for R in s.sets:
                s.ev_reg.append(np.sort(rng.choice(R, size=min(200, R.size), replace=False)))
            s.ev_flo = np.sort(rng.choice(s.N, size=min(600, s.N), replace=False))
        out.append(s)
    return out


# ------------------------------------------------------------------------------------------------ model
class Model:
    """Training set of one fold: subsampled points, weights, standardisation."""

    def __init__(self, scenes):
        E = sum(sum(s.set_w) for s in scenes)
        Xr, Xn, ar, br, bn = [], [], [], [], []
        band = []
        band_sets = []
        for s in scenes:
            for k, R in enumerate(s.sets):
                if s.set_w[k] == 0:
                    continue
                w = s.set_w[k] / E
                pi = R.size / s.N
                nr, nn = s.tr_reg[k].size, s.tr_non[k].size
                Xr.append(s.F[s.tr_reg[k]])
                Xn.append(s.F[s.tr_non[k]])
                ar.append(np.full(nr, w / nr))
                br.append(np.full(nr, w * pi / nr))
                bn.append(np.full(nn, w * (1 - pi) / nn))
                for _ in range(s.set_w[k]):
                    band.append(s.F[R, 0])
                band_sets.append(s.F[R, 0])
        self.Xr, self.Xn = np.concatenate(Xr), np.concatenate(Xn)
        self.ar, self.br, self.bn = np.concatenate(ar), np.concatenate(br), np.concatenate(bn)
        assert abs(self.ar.sum() - 1) < 1e-9 and abs(self.br.sum() + self.bn.sum() - 1) < 1e-9
        X = np.concatenate([self.Xr, self.Xn])
        b = np.concatenate([self.br, self.bn])
        mu = (b[:, None] * X).sum(0) / b.sum()
        sd = np.sqrt((b[:, None] * (X - mu) ** 2).sum(0) / b.sum())
        sd = np.where(sd > 0, sd, 1.0)
        self.mu, self.sd = mu, sd
        self.Zr, self.Zn = (self.Xr - mu) / sd, (self.Xn - mu) / sd
        self.n_r = self.Xr.shape[0]
        bd = np.concatenate(band)
        self.band = (float(np.quantile(bd, 0.05)), float(np.quantile(bd, 0.95)))
        bs = np.concatenate(band_sets)
        self.band_sets = (float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95)))
        self.AB = np.stack([self.ar, self.br], 1)

    def h(self, d, m):
        return m * self.n_r ** (-1.0 / (d + 4))

    def logs(self, F, configs):
        """For raw features F (n, 9), and configs [(dims, h)], arrays (len(configs), n) of log numerator and log
        denominator (up to the shared Gaussian constant)."""
        Z = (F - self.mu) / self.sd
        n = Z.shape[0]
        LN = np.empty((len(configs), n))
        LD = np.empty((len(configs), n))
        groups = {}
        for ci, (dims, h) in enumerate(configs):
            groups.setdefault(tuple(dims), []).append((ci, h))
        base = [0, 1, 2, 3, 4, 5]
        Zr_b, Zn_b = self.Zr[:, base], self.Zn[:, base]
        nr2, nn2 = (Zr_b ** 2).sum(1), (Zn_b ** 2).sum(1)
        for lo in range(0, n, CHUNK):
            zc = Z[lo:lo + CHUNK]
            need_base = any(len(d) > 1 for d in groups)
            if need_base:
                zb = zc[:, base]
                z2 = (zb ** 2).sum(1)[:, None]
                Br = np.maximum(z2 + nr2[None, :] - 2.0 * zb @ Zr_b.T, 0.0)
                Bn = np.maximum(z2 + nn2[None, :] - 2.0 * zb @ Zn_b.T, 0.0)
            for dims, lst in groups.items():
                if list(dims) == base:
                    Dr, Dn = Br, Bn
                elif len(dims) == 1:
                    j = dims[0]
                    Dr = (zc[:, j:j + 1] - self.Zr[None, :, j]) ** 2
                    Dn = (zc[:, j:j + 1] - self.Zn[None, :, j]) ** 2
                else:
                    extra = [j for j in dims if j not in base]
                    assert [j for j in dims if j in base] == base
                    Dr, Dn = Br.copy(), Bn.copy()
                    for j in extra:
                        Dr += (zc[:, j:j + 1] - self.Zr[None, :, j]) ** 2
                        Dn += (zc[:, j:j + 1] - self.Zn[None, :, j]) ** 2
                mr, mn = Dr.min(1), Dn.min(1)
                Rr = Dr - mr[:, None]
                Rn = Dn - mn[:, None]
                c_prev, Er, En = None, None, None
                for ci, h in sorted(lst, key=lambda t: -t[1]):          # widest kernel first
                    c = 1.0 / (2.0 * h * h)
                    if c_prev is not None and c == 4.0 * c_prev:       # exp(-4cR) = exp(-cR)^4, exactly the next m
                        Er *= Er
                        Er *= Er
                        En *= En
                        En *= En
                    else:
                        Er = np.exp(Rr * (-c))
                        En = np.exp(Rn * (-c))
                    c_prev = c
                    sab = Er @ self.AB                       # (C, 2): numerator, region part of denominator
                    sn = En @ self.bn
                    LN[ci, lo:lo + CHUNK] = -c * mr + np.log(sab[:, 0])
                    LD[ci, lo:lo + CHUNK] = np.logaddexp(-c * mr + np.log(sab[:, 1]), -c * mn + np.log(sn))
        return LN, LD


def lse(x):
    m = x.max()
    return m + math.log(np.exp(x - m).sum())


def logq(lr, N):
    """log q over a scene's floor cells from log r (or log p) at all floor cells."""
    return np.logaddexp(LOG99 + lr - lse(lr), LOG01 - math.log(N))


# ------------------------------------------------------------------------------------------------ inner CV
def grid_configs(model):
    cfgs, names = [], []
    for fs in FS_ORDER:
        for m in MS:
            cfgs.append((FS[fs], model.h(len(FS[fs]), m)))
            names.append((fs, m))
    for m in MS:
        cfgs.append(([0], model.h(1, m)))
        names.append(('F1D', m))
    return cfgs, names


def inner_cv(train, tag):
    """Leave one seed out over the training scenes. Returns the mean inner score per config for ratio, plain, 1-D."""
    seeds = sorted({s.seed for s in train})
    sums = None
    n_ep = 0
    for k in seeds:
        tr = [s for s in train if s.seed != k]
        te = [s for s in train if s.seed == k]
        model = Model(tr)
        cfgs, names = grid_configs(model)
        acc_r = np.zeros(len(cfgs))
        acc_p = np.zeros(len(cfgs))
        for s in te:
            idx = np.concatenate([s.ev_flo] + s.ev_reg)
            LN, LD = model.logs(s.F[idx], cfgs)
            nf = s.ev_flo.size
            for which, L, acc in (('ratio', LN - LD, acc_r), ('plain', LN, acc_p)):
                logZ = math.log(s.N) + np.array([lse(L[c, :nf]) for c in range(len(cfgs))]) - math.log(nf)
                off = nf
                for kk, R in enumerate(s.ev_reg):
                    sl = L[:, off:off + R.size]
                    off += R.size
                    lq = np.logaddexp(LOG99 + sl - logZ[:, None], LOG01 - math.log(s.N))
                    acc += s.set_w[kk] * lq.mean(1)
            n_ep += sum(s.set_w)
        sums = (acc_r, acc_p) if sums is None else (sums[0] + acc_r, sums[1] + acc_p)
        log(tag, 'inner seed', k, 'done')
    mr, mp = sums[0] / n_ep, sums[1] / n_ep
    return names, mr, mp, n_ep


def pick(names, scores, allowed):
    best = None
    for i, (fs, m) in enumerate(names):
        if fs not in allowed:
            continue
        key = (scores[i], -allowed.index(fs), m)       # higher score, then earlier FS, then larger m
        if best is None or key > best[0]:
            best = (key, i)
    i = best[1]
    return names[i], float(scores[i])


# ------------------------------------------------------------------------------------------------ outer
def outer(train_model, test, choice):
    """Per test episode rows for ratio, plain, ratio1d, band, uniform."""
    (fr, mr_), (fp, mp_), (f1, m1) = choice['ratio'], choice['plain'], choice['ratio1d']
    same = (fr, mr_) == (fp, mp_)
    cfgs = [(FS[fr], train_model.h(len(FS[fr]), mr_)), ([0], train_model.h(1, m1))]
    if not same:
        cfgs.append((FS[fp], train_model.h(len(FS[fp]), mp_)))
    ip = 0 if same else 2
    rows = []
    lo, hi = train_model.band
    for s in test:
        LN, LD = train_model.logs(s.F, cfgs)
        lq = {'ratio': logq(LN[0] - LD[0], s.N), 'plain': logq(LN[ip], s.N), 'ratio1d': logq(LN[1] - LD[1], s.N)}
        dt = s.F[:, 0]
        B = (dt >= lo) & (dt <= hi)
        if B.sum() == 0:
            lq['band'] = np.full(s.N, -math.log(s.N))
        else:
            lq['band'] = np.log(0.99 * B / B.sum() + 0.01 / s.N)
        lo2, hi2 = train_model.band_sets           # alternative reading: each region cell of a scene once
        B2 = (dt >= lo2) & (dt <= hi2)
        lq['bandu'] = np.log(0.99 * B2 / B2.sum() + 0.01 / s.N) if B2.sum() else np.full(s.N, -math.log(s.N))
        lq['uniform'] = np.full(s.N, -math.log(s.N))
        for e in s.eps:
            R = s.sets[e['set']]
            row = {'scene': s.name, 'seed': s.seed, 'situation': s.situation, 'domain': s.domain,
                   'episode': e['episode'], 'n_region': int(R.size), 'N': s.N,
                   'band_cover': float(B[R].mean()), 'folder': s.meta.get('folder')}
            for meth, v in lq.items():
                q = np.exp(v[R])
                row[f'S_{meth}'] = float(v[R].mean())
                row[f'M_{meth}'] = float(q.sum())
                row[f'lift_{meth}'] = float(q.sum() / (R.size / s.N))
            rows.append(row)
    return rows


def run_fold(tag, train, test, inner_train=None):
    t0 = time.time()
    names, mr, mp, n_ep = inner_cv(train if inner_train is None else inner_train, tag)
    choice = {'ratio': pick(names, mr, FS_ORDER)[0], 'plain': pick(names, mp, FS_ORDER)[0],
              'ratio1d': pick(names, mr, ['F1D'])[0]}
    model = Model(train)
    info = {'tag': tag, 'choice': choice, 'inner_names': names, 'inner_ratio': mr.tolist(), 'inner_plain': mp.tolist(),
            'inner_n_episodes': n_ep, 'n_r': model.n_r, 'n_train_points': int(model.Xr.shape[0] + model.Xn.shape[0]),
            'h_ratio': model.h(len(FS[choice['ratio'][0]]), choice['ratio'][1]),
            'h_1d': model.h(1, choice['ratio1d'][1]), 'band': model.band, 'band_sets': model.band_sets,
            'mu': model.mu.tolist(), 'sd': model.sd.tolist(),
            'n_train_scenes': len(train), 'n_test_scenes': len(test)}
    log(tag, 'choice', choice, 'inner done', f'{time.time() - t0:.0f} s')
    rows = outer(model, test, choice)
    for r in rows:
        r['fold'] = tag
    info['seconds'] = time.time() - t0
    log(tag, 'outer done', f'{time.time() - t0:.0f} s')
    return info, rows


def main(stage):
    fresh = load('fresh')
    assert len(fresh) == 260
    out = {'stage': stage, 'folds': [], 'rows': []}
    if stage == 'bench':
        tr = [s for s in fresh if s.seed != 100]
        te = [s for s in fresh if s.seed == 100][:3]
        model = Model(tr)
        cfgs, names = grid_configs(model)
        for C in (16, 48, 128):
            global CHUNK
            CHUNK = C
            t0 = time.time()
            idx = np.concatenate([te[0].ev_flo] + te[0].ev_reg)
            model.logs(te[0].F[idx], cfgs)
            dt = time.time() - t0
            pairs = idx.size * (model.Xr.shape[0] + model.Xn.shape[0])
            log('chunk', C, f'{dt:.2f} s', f'{dt / pairs / len(cfgs) * 1e9:.2f} ns per pair-config')
        # exactness check vs a direct logsumexp on a few points
        idx = te[1].ev_flo[:5]
        cfg = [(FS['FS3'], model.h(8, 0.5)), ([0], model.h(1, 1.0))]
        LN, LD = model.logs(te[1].F[idx], cfg)
        Z = (te[1].F[idx] - model.mu) / model.sd
        for ci, (dims, h) in enumerate(cfg):
            for i in range(len(idx)):
                dr = ((Z[i, dims] - model.Zr[:, dims]) ** 2).sum(1)
                dn = ((Z[i, dims] - model.Zn[:, dims]) ** 2).sum(1)
                ln = lse(np.log(model.ar) - dr / (2 * h * h))
                ld = lse(np.concatenate([np.log(model.br) - dr / (2 * h * h), np.log(model.bn) - dn / (2 * h * h)]))
                log('check', ci, i, LN[ci, i] - ln, LD[ci, i] - ld)
        return
    if stage == 'primary':
        for sit in PEOPLE_SITS:
            train = [s for s in fresh if s.situation != sit]
            test = [s for s in fresh if s.situation == sit]
            assert all(s.situation != sit for s in train)
            info, rows = run_fold(f'LOSO_{sit}', train, test)
            out['folds'].append(info)
            out['rows'].extend(rows)
            (V / f'out_{stage}.json').write_text(json.dumps(out))
    elif stage.startswith('ceiling'):
        seeds = SEEDS if stage == 'ceiling' else [int(x) for x in stage.split(':')[1].split(',')]
        for k in seeds:
            train = [s for s in fresh if s.seed != k]
            test = [s for s in fresh if s.seed == k]
            info, rows = run_fold(f'LOSeed_{k}', train, test)
            out['folds'].append(info)
            out['rows'].extend(rows)
            (V / f"out_{stage.replace(':', '_').replace(',', '-')}.json").write_text(json.dumps(out))
    elif stage == 'dev':
        devs = load('dev')
        info, rows = run_fold('DEV', fresh, devs)
        out['folds'].append(info)
        out['rows'].extend(rows)
        (V / f'out_{stage}.json').write_text(json.dumps(out))
    log('stage', stage, 'finished')


if __name__ == '__main__':
    main(sys.argv[1])
