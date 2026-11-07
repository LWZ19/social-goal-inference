"""C5 gate 2, step 4 of FROZEN.md: inner CV and the outer evaluation of the ratio KDE, the plain KDE, the 1-D ratio
KDE, the band-uniform baseline and uniform, on the cached cells (cells.py). numpy only, float64, log space.

Stages (each writes work/<stage>.json; outer stages wait for the earlier outer stage, so the outer scores are
computed in the frozen order: primary folds, then the ceiling, then the development scenes):
    inner_primary | inner_ceiling | inner_dev | outer_primary | outer_ceiling | outer_dev | selftest

    cd WT && PYTHONPATH= OMP_NUM_THREADS=2 ... nice -n 10 .venv/bin/python <this> STAGE
No model, no network. Nothing here reads a situation label inside a method: labels only split folds and tabulate.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1d/c5_gates')
from common import CACHE, FEATURE_SETS, FROZEN_SHA, HERE, M_GRID, PEOPLE_SITS, SEEDS  # noqa: E402

WORK = HERE / 'work'
CHUNK = 256
MIX = 0.01
METHOD_GRID = {'ratio': [(fs, m) for fs in FEATURE_SETS for m in M_GRID],
               'plain': [(fs, m) for fs in FEATURE_SETS for m in M_GRID],
               'ratio1d': [('F1D', m) for m in M_GRID]}
FS_ORDER = {'FS1': 0, 'FS2': 1, 'FS3': 2, 'F1D': 0}


def cols(fs):
    return (0,) if fs == 'F1D' else FEATURE_SETS[fs]


# --------------------------------------------------------------------------------------------------- data
class SceneData:
    def __init__(self, rec):
        self.rec = rec
        self.kind = rec['kind']
        self.seed = rec['seed']
        self.key = rec['key']
        self.domain = rec['domain']
        self.n_floor = rec['n_floor']
        z = np.load(CACHE / rec['file'])
        self.F = {k[3:]: z[k] for k in z.files if k.startswith('F__')}
        self.reg = {k[5:]: z[k] for k in z.files if k.startswith('reg__')}
        self.eps = [e for e in rec['episodes'] if e.get('eligible')]
        sits = {e['situation'] for e in rec['episodes']}
        assert len(sits) == 1, (self.key, sits)
        self.situation = next(iter(sits))
        self.sets = []
        if self.kind == 'fresh':
            for st in rec['sets']:
                j = st['set']
                self.sets.append({'eids': st['eids'], 'target': st['target'], 'pi': st['pi'],
                                  'n_region': st['n_region'], 'reg': self.reg[st['eids'][0]],
                                  'trreg': z[f'trreg__{j}'], 'trnon': z[f'trnon__{j}'], 'vareg': z[f'vareg__{j}']})
            self.vafloor = z['vafloor']


def load(kind):
    index = json.loads((CACHE / 'index.json').read_text())
    return [SceneData(r) for r in index if r['kind'] == kind]


# --------------------------------------------------------------------------------------------------- density
def train_arrays(scenes, fs):
    """Region sample (numerator weights a, denominator weights b) and non-region sample (denominator weights),
    every training episode weighing the same (identical sets in a scene pooled by summing), standardised by the
    floor KDE's weighted mean and SD."""
    c = cols(fs)
    XR, aR, bR, XN, bN = [], [], [], [], []
    n_dropped = 0
    for s in scenes:
        if not s.sets:
            n_dropped += 1
            continue
        for st in s.sets:
            F = s.F[st['target']]
            w = float(len(st['eids']))
            r, n = st['trreg'], st['trnon']
            XR.append(F[r][:, c])
            aR.append(np.full(len(r), w / len(r)))
            bR.append(np.full(len(r), w * st['pi'] / len(r)))
            if len(n):
                XN.append(F[n][:, c])
                bN.append(np.full(len(n), w * (1.0 - st['pi']) / len(n)))
    XR, aR, bR = np.vstack(XR), np.concatenate(aR), np.concatenate(bR)
    XN, bN = np.vstack(XN), np.concatenate(bN)
    tot = aR.sum()
    aR, bR, bN = aR / tot, bR / tot, bN / tot
    X = np.vstack([XR, XN])
    b = np.concatenate([bR, bN])
    mu = (b[:, None] * X).sum(axis=0) / b.sum()
    sd = np.sqrt((b[:, None] * (X - mu) ** 2).sum(axis=0) / b.sum())
    sd = np.where(sd > 0, sd, 1.0)
    return {'XR': (XR - mu) / sd, 'XN': (XN - mu) / sd, 'aR': aR, 'bR': bR, 'bN': bN, 'mu': mu, 'sd': sd,
            'n_r': int(len(XR)), 'd': len(c), 'cols': c, 'n_dropped': n_dropped}


def bandwidth(T, m):
    return m * T['n_r'] ** (-1.0 / (T['d'] + 4))


def log_terms(Xe, T, hs):
    """For each h: (logA, logB) at every row of Xe (standardised), where A = sum_i aR_i K_h and
    B = sum_i bR_i K_h + sum_j bN_j K_h; the kernel's constant is dropped (it cancels)."""
    XR, XN, aR, bR, bN = T['XR'], T['XN'], T['aR'], T['bR'], T['bN']
    tR, tN = (XR ** 2).sum(axis=1), (XN ** 2).sum(axis=1)
    ne = len(Xe)
    out = {h: (np.empty(ne), np.empty(ne)) for h in hs}
    for lo in range(0, ne, CHUNK):
        b = Xe[lo:lo + CHUNK]
        b2 = (b ** 2).sum(axis=1)
        DR = b @ XR.T
        DR *= -2.0
        DR += b2[:, None]
        DR += tR[None, :]
        np.maximum(DR, 0.0, out=DR)
        DN = b @ XN.T
        DN *= -2.0
        DN += b2[:, None]
        DN += tN[None, :]
        np.maximum(DN, 0.0, out=DN)
        minR, minN = DR.min(axis=1), DN.min(axis=1)
        minA = np.minimum(minR, minN)
        ER, EN = np.empty_like(DR), np.empty_like(DN)
        for h in hs:
            c = 0.5 / (h * h)
            np.subtract(DR, minR[:, None], out=ER)
            ER *= -c
            np.exp(ER, out=ER)
            sA, sBR = ER @ aR, ER @ bR
            np.subtract(DN, minA[:, None], out=EN)
            EN *= -c
            np.exp(EN, out=EN)
            sBN = EN @ bN
            logA = -c * minR + np.log(sA)
            logB = -c * minA + np.log(sBR * np.exp(-c * (minR - minA)) + sBN)
            out[h][0][lo:lo + CHUNK] = logA
            out[h][1][lo:lo + CHUNK] = logB
    return out


def logsumexp(x):
    m = np.max(x)
    if not np.isfinite(m):
        return m
    return m + math.log(np.exp(x - m).sum())


def log_q(logr_cells, log_sum_r, n_floor):
    """log q = log(0.99 r / sum_floor r + 0.01 / N_floor)."""
    return np.logaddexp(math.log(1.0 - MIX) + logr_cells - log_sum_r, math.log(MIX / n_floor))


def outcome(lq_region, n_floor):
    n_reg = len(lq_region)
    M = float(np.exp(lq_region).sum())
    return {'S': float(lq_region.mean()), 'M': M, 'lift': M / (n_reg / n_floor)}


# --------------------------------------------------------------------------------------------------- inner CV
def inner_cv(train_scenes, label):
    """Leave one seed out over the training scenes; per method and grid point, the mean over held-out seeds of the
    per-seed mean S over that seed's eligible episodes (estimated: <= 200 region cells per episode, sum_floor r from
    <= 600 random floor cells per scene)."""
    seeds = sorted({s.seed for s in train_scenes})
    per = {meth: {f'{fs}|{m}': {} for fs, m in grid} for meth, grid in METHOD_GRID.items()}
    t0 = time.monotonic()
    for s_out in seeds:
        tr = [s for s in train_scenes if s.seed != s_out]
        va = [s for s in train_scenes if s.seed == s_out and s.sets]
        for fs in ('FS1', 'FS2', 'FS3', 'F1D'):
            T = train_arrays(tr, fs)
            c = T['cols']
            blocks, Xe = [], []
            off = 0
            for s in va:
                F = s.F[s.sets[0]['target']] if len({st['target'] for st in s.sets}) == 1 else None
                assert F is not None
                idx = [st['vareg'] for st in s.sets] + [s.vafloor]
                parts = []
                for ii in idx:
                    parts.append((off, off + len(ii)))
                    off += len(ii)
                    Xe.append((F[ii][:, c] - T['mu']) / T['sd'])
                blocks.append((s, parts))
            Xe = np.vstack(Xe)
            hs = {m: bandwidth(T, m) for m in M_GRID}
            terms = log_terms(Xe, T, list(hs.values()))
            for m, h in hs.items():
                logA, logB = terms[h]
                for meth in (('ratio1d',) if fs == 'F1D' else ('ratio', 'plain')):
                    logr = logA - logB if meth != 'plain' else logA
                    scores = []
                    for s, parts in blocks:
                        fl0, fl1 = parts[-1]
                        n_fl = fl1 - fl0
                        lsr = math.log(s.n_floor / n_fl) + logsumexp(logr[fl0:fl1])
                        for st, (r0, r1) in zip(s.sets, parts[:-1]):
                            S = float(log_q(logr[r0:r1], lsr, s.n_floor).mean())
                            scores.extend([S] * len(st['eids']))
                    per[meth][f'{fs}|{m}'][str(s_out)] = float(np.mean(scores))
        print(f'[{label}] inner seed {s_out} done ({len(tr)} train, {len(va)} val scenes), '
              f'{time.monotonic() - t0:.0f} s', flush=True)
    chosen = {}
    for meth, grid in METHOD_GRID.items():
        best = None
        for fs, m in grid:
            v = float(np.mean(list(per[meth][f'{fs}|{m}'].values())))
            k = (v, -FS_ORDER[fs], m)
            if best is None or k > best[0]:
                best = (k, fs, m, v)
        _, fs, m, v = best
        grid_ms = [mm for f2, mm in grid if f2 == fs]
        chosen[meth] = {'fs': fs, 'm': m, 'inner_mean': v, 'edge': m in (min(grid_ms), max(grid_ms))}
    return {'seeds': seeds, 'per_seed': per,
            'means': {meth: {k: float(np.mean(list(v.values()))) for k, v in per[meth].items()} for meth in per},
            'chosen': chosen, 'n_train_scenes': len(train_scenes),
            'n_train_episodes': int(sum(len(s.eps) for s in train_scenes))}


# --------------------------------------------------------------------------------------------------- outer
def band_limits(train_scenes):
    d = []
    for s in train_scenes:
        cells = {}
        for st in s.sets:
            for i in st['reg']:
                cells[(st['target'], int(i))] = True
        for (tid, i) in cells:
            d.append(s.F[tid][i, 0])
    d = np.array(d)
    return float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95)), int(len(d))


def outer_eval(train_scenes, test_scenes, chosen, label):
    """Exact evaluation over every floor and region cell of each test scene."""
    q05, q95, n_band = band_limits(train_scenes)
    configs = {}
    for meth, ch in chosen.items():
        configs.setdefault(ch['fs'], set()).add(ch['m'])
    trained = {fs: train_arrays(train_scenes, fs) for fs in configs}
    info = {'band': {'q05': q05, 'q95': q95, 'n_cells': n_band},
            'h': {meth: bandwidth(trained[ch['fs']], ch['m']) for meth, ch in chosen.items()},
            'n_r': {fs: T['n_r'] for fs, T in trained.items()},
            'n_dropped_train_scenes': {fs: T['n_dropped'] for fs, T in trained.items()},
            'standardise': {fs: {'mu': T['mu'].tolist(), 'sd': T['sd'].tolist()} for fs, T in trained.items()}}
    rows = []
    t0 = time.monotonic()
    for s in test_scenes:
        if not s.eps:
            continue
        for tid in sorted({e['target'] for e in s.eps}):
            F = s.F[tid]
            N = s.n_floor
            terms = {}
            for fs, ms in configs.items():
                T = trained[fs]
                Xe = (F[:, T['cols']] - T['mu']) / T['sd']
                hs = [bandwidth(T, m) for m in sorted(ms)]
                for (m, h), v in zip(zip(sorted(ms), hs), log_terms(Xe, T, hs).values()):
                    terms[(fs, m)] = v
            lq = {}
            for meth, ch in chosen.items():
                logA, logB = terms[(ch['fs'], ch['m'])]
                logr = logA if meth == 'plain' else logA - logB
                lq[meth] = log_q(logr, logsumexp(logr), N)
            band = (F[:, 0] >= q05) & (F[:, 0] <= q95)
            nb = int(band.sum())
            if nb:
                qb = (1.0 - MIX) * band / nb + MIX / N
            else:
                qb = np.full(N, 1.0 / N)
            lq['band'] = np.log(qb)
            lq['uniform'] = np.full(N, -math.log(N))
            for e in s.eps:
                if e['target'] != tid:
                    continue
                reg = s.reg[e['eid']]
                row = {'fold': label, 'kind': s.kind, 'seed': s.seed, 'key': s.key, 'scene_id': s.rec['scene_id'],
                       'eid': e['eid'], 'situation': e['situation'], 'domain': s.domain, 'target': tid,
                       'n_floor': N, 'n_region': int(len(reg)), 'band_cells': nb}
                for meth, v in lq.items():
                    row[meth] = outcome(v[reg], N)
                row['delta'] = row['ratio']['S'] - row['band']['S']
                rows.append(row)
        print(f'[{label}] {s.key} done, {time.monotonic() - t0:.0f} s', flush=True)
    return rows, info


# --------------------------------------------------------------------------------------------------- stages
def folds_primary(fresh):
    out = []
    for left in PEOPLE_SITS:
        train = [s for s in fresh if s.situation != left]
        test = [s for s in fresh if s.situation == left]
        out.append((f'LOSO_{left}', train, test))
    return out


def folds_ceiling(fresh):
    return [(f'LOSeed_{k}', [s for s in fresh if s.seed != k], [s for s in fresh if s.seed == k]) for k in SEEDS]


def ceiling_part(fresh, inner, seeds):
    rows, infos = [], {}
    for name, train, test in folds_ceiling(fresh):
        if int(name.split('_')[1]) not in seeds:
            continue
        r, i = outer_eval(train, test, inner[name]['chosen'], name)
        rows += r
        infos[name] = i
    return rows, infos


def wait_for(name):
    p = WORK / f'{name}.json'
    while not p.exists():
        time.sleep(30)
    return json.loads(p.read_text())


def write(name, obj):
    WORK.mkdir(exist_ok=True)
    tmp = WORK / f'{name}.json.tmp'
    tmp.write_text(json.dumps(obj))
    tmp.rename(WORK / f'{name}.json')


def selftest():
    """The kernel sums against a direct computation on random numbers (no data)."""
    rng = np.random.default_rng(1)
    XR, XN = rng.normal(size=(50, 3)), rng.normal(size=(70, 3)) + 3.0
    aR = rng.random(50)
    bR, bN = rng.random(50), rng.random(70)
    T = {'XR': XR, 'XN': XN, 'aR': aR, 'bR': bR, 'bN': bN}
    Xe = rng.normal(size=(600, 3)) * 2
    for h in (0.05, 0.3, 2.0):
        logA, logB = log_terms(Xe, T, [h])[h]
        dR = ((Xe[:, None, :] - XR[None]) ** 2).sum(-1)
        dN = ((Xe[:, None, :] - XN[None]) ** 2).sum(-1)

        def lse(e):
            mx = e.max(1, keepdims=True)
            return (mx + np.log(np.exp(e - mx).sum(1, keepdims=True)))[:, 0]
        rA = lse(np.log(aR)[None] - dR / (2 * h * h))
        rB = lse(np.hstack([np.log(bR)[None] - dR / (2 * h * h), np.log(bN)[None] - dN / (2 * h * h)]))
        print(f'h {h}: max |logA err| {np.abs(logA - rA).max():.2e}, '
              f'max |logB err| {np.abs(logB - rB).max():.2e} (direct stable logsumexp reference)')


def main():
    assert hashlib.sha256((HERE / 'FROZEN.md').read_bytes()).hexdigest() == FROZEN_SHA
    stage = sys.argv[1]
    if stage == 'selftest':
        selftest()
        return
    fresh = load('fresh')
    assert len(fresh) == 260, len(fresh)
    t0 = time.monotonic()
    if stage == 'inner_primary':
        write(stage, {name: inner_cv(train, name) for name, train, _ in folds_primary(fresh)})
    elif stage == 'inner_ceiling':
        write(stage, {name: inner_cv(train, name) for name, train, _ in folds_ceiling(fresh)})
    elif stage == 'inner_dev':
        write(stage, {'DEV': inner_cv(fresh, 'DEV')})
    elif stage == 'outer_primary':
        inner = wait_for('inner_primary')
        rows, infos = [], {}
        for name, train, test in folds_primary(fresh):
            r, i = outer_eval(train, test, inner[name]['chosen'], name)
            rows += r
            infos[name] = i
        write(stage, {'rows': rows, 'info': infos})
    elif stage == 'outer_ceiling':
        # Split across the two processes for speed (same computation): this process takes the seed folds
        # 100-104, the outer_dev process takes 105-109 while it waits, and this one merges them.
        inner = wait_for('inner_ceiling')
        wait_for('outer_primary')
        rows, infos = ceiling_part(fresh, inner, SEEDS[:5])
        write('outer_ceiling_A', {'rows': rows, 'info': infos})
        other = wait_for('outer_ceiling_B')
        infos.update(other['info'])
        write(stage, {'rows': rows + other['rows'], 'info': infos})
    elif stage == 'outer_dev':
        inner = wait_for('inner_dev')
        inner_c = wait_for('inner_ceiling')
        wait_for('outer_primary')
        rows, infos = ceiling_part(fresh, inner_c, SEEDS[5:])
        write('outer_ceiling_B', {'rows': rows, 'info': infos})
        wait_for('outer_ceiling')
        dev = load('dev')
        rows, info = outer_eval(fresh, dev, inner['DEV']['chosen'], 'DEV')
        write(stage, {'rows': rows, 'info': {'DEV': info}})
    else:
        raise SystemExit(f'unknown stage {stage}')
    print(f'{stage} done in {time.monotonic() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
