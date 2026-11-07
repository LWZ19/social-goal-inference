"""I04 step 3 (FROZEN.md sections 5 and 6): the S table for every record and setting, then the 19 leave-one-layout-out
fold fits on training records only, written to fits.json (and folds.json, the same content) and hashed in FITS.sha256
before any held-out metric is computed.

S(rec, setting) = sum_c p(c) P_succ(c), with p over F(rec):
  V1  p ~ exp(-(E - min E)/T), E = sum_k gamma^(5-k) key_k (integer millimetres, exact), gamma x T x kappa = 500;
  V2  V1 restricted to the kept cells (least excluded_m, then least excluded_total_m), 500 settings;
  V3  p ~ exp(-|c - c_top|^2 / (2 sigma^2)), sigma x kappa = 125.
P_succ(c) = matched x reachable x in-region x P_vM(heading within 45 deg of a holding region), kappa inf = indicator.
Objective per fold: mean over training records of ln(1e-3 + S). Ties within 1e-12: largest T or sigma, then smallest
kappa, then smallest gamma. No held-out row is read by a fold's choice.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time

import numpy as np

HERE = 'results/2026-10-17-beyond-vlm/phase1d/i04'
PRIMARY = ['old_qwen', 'old_glm', 'old_deepseek', 'watch_qwen', 'watch_glm', 'watch_deepseek']
GAMMAS = (1, 3, 10, 30)
TS = np.geomspace(0.005, 5, 25)
SIGMAS = np.geomspace(0.05, 3, 25)
KAPPAS = (4.0, 8.0, 16.0, 32.0, math.inf)
FLOOR = 1e-3
TIE = 1e-12


def load_meta():
    rows = []
    for s in (0, 1):
        with open(f'{HERE}/cache/meta_shard{s}.jsonl') as f:
            rows += [json.loads(line) for line in f]
    rows.sort(key=lambda r: r['key'])
    return rows


def load_npz(r):
    z = np.load(f'{HERE}/cache/npz/{r["digest"]}.npz')
    return {k: z[k] for k in z.files}


def cells_of(r, z):
    offsets = np.arange(-r['half'], r['half'] + 1) * r['res']
    return np.stack([offsets[z['col'].astype(np.int64)], offsets[z['row'].astype(np.int64)]], axis=1)


def psucc(z):
    """(n_succ, 5) P_succ on the succ cells for kappa 4, 8, 16, 32, inf."""
    return np.concatenate([z['pvm'], z['hok_succ'].astype(np.float64)[:, None]], axis=1)


def energy_mm(z, g):
    w = np.array([g ** 4, g ** 3, g ** 2, g, 1], dtype=np.int64)
    return z['kmm'].astype(np.int64) @ w


def s_soft(E_mm, mask, succ_idx, P, temps):
    """S for each T (rows) and kappa (columns): p ~ exp(-(E - min E)/T) on the cells of mask (others 0)."""
    Em = E_mm[mask]
    e = (Em - Em.min()).astype(np.float64) / 1000.0          # metres, >= 0, least = 0 (log-sum-exp shift)
    W = np.exp(-e[None, :] / temps[:, None])                 # (nT, n_mask)
    full = np.zeros((len(temps), len(E_mm)))
    full[:, mask] = W
    Z = full.sum(axis=1)
    return (full[:, succ_idx] @ P) / Z[:, None]


def s_gauss(d2, succ_idx, P, sigmas):
    W = np.exp(-d2[None, :] / (2.0 * sigmas[:, None] ** 2))
    return (W[:, succ_idx] @ P) / W.sum(axis=1)[:, None]


def s_table(rows):
    n = len(rows)
    S1 = np.zeros((n, len(GAMMAS), len(TS), len(KAPPAS)))
    S2 = np.zeros_like(S1)
    S3 = np.zeros((n, len(SIGMAS), len(KAPPAS)))
    for i, r in enumerate(rows):
        z = load_npz(r)
        N = r['n_cells']
        P = psucc(z)
        succ = z['succ_idx'].astype(np.int64)
        kept = np.zeros(N, dtype=bool)
        kept[z['kept']] = True
        allc = np.ones(N, dtype=bool)
        for gi, g in enumerate(GAMMAS):
            E = energy_mm(z, g)
            S1[i, gi] = s_soft(E, allc, succ, P, TS)
            S2[i, gi] = s_soft(E, kept, succ, P, TS)
        cells = cells_of(r, z)
        top = int(z['picks'][0])
        d2 = (cells[:, 0] - cells[top, 0]) ** 2 + (cells[:, 1] - cells[top, 1]) ** 2
        S3[i] = s_gauss(d2, succ, P, SIGMAS)
    return S1, S2, S3


def choose(obj, settings, order_key):
    """Best setting; ties within TIE broken by order_key (smallest first)."""
    best = obj.max()
    tied = [s for s, o in zip(settings, obj) if o >= best - TIE]
    tied.sort(key=order_key)
    return tied[0], best, len(tied)


def main():
    t0 = time.monotonic()
    rows = [r for r in load_meta() if r['repro_ok']]
    S1, S2, S3 = s_table(rows)
    t1 = time.monotonic()
    keys = np.array([r['key'] for r in rows])
    np.savez_compressed(f'{HERE}/s_table.npz', keys=keys, S1=S1, S2=S2, S3=S3, gammas=np.array(GAMMAS), Ts=TS,
                        sigmas=SIGMAS, kappas=np.array(KAPPAS))
    prim = np.array([r['cell'] in PRIMARY for r in rows])
    layout = np.array([r['layout'] for r in rows])
    folds = sorted(set(layout[prim]))
    L1 = np.log(FLOOR + S1)
    L2 = np.log(FLOOR + S2)
    L3 = np.log(FLOOR + S3)
    set12 = [(gi, ti, ki) for gi in range(len(GAMMAS)) for ti in range(len(TS)) for ki in range(len(KAPPAS))]
    set3 = [(si, ki) for si in range(len(SIGMAS)) for ki in range(len(KAPPAS))]
    key12 = lambda s: (-TS[s[1]], KAPPAS[s[2]], GAMMAS[s[0]])       # largest T, smallest kappa, smallest gamma
    key3 = lambda s: (-SIGMAS[s[0]], KAPPAS[s[1]])                  # largest sigma, smallest kappa
    fits = dict(note='FROZEN.md sha256 8e0aac9801eadd6a0c420efa5a96d2f629d8d854498081657d67a6cf6c981882',
                objective='mean over the fold\'s training records (primary cells, other layouts) of ln(1e-3 + S)',
                grids=dict(gamma=list(GAMMAS), T=TS.tolist(), sigma=SIGMAS.tolist(),
                           kappa=['inf' if math.isinf(k) else k for k in KAPPAS]),
                n_primary=int(prim.sum()), folds={})
    kap = lambda k: 'inf' if math.isinf(KAPPAS[k]) else KAPPAS[k]
    for L in folds:
        train = prim & (layout != L)
        test = prim & (layout == L)
        o1 = L1[train].mean(axis=0)
        o2 = L2[train].mean(axis=0)
        o3 = L3[train].mean(axis=0)
        (g1, t1_, k1), b1, n1 = choose(np.array([o1[s] for s in set12]), set12, key12)
        (g2, t2_, k2), b2, n2 = choose(np.array([o2[s] for s in set12]), set12, key12)
        (s3, k3), b3, n3 = choose(np.array([o3[s] for s in set3]), set3, key3)
        fits['folds'][L] = dict(
            n_train=int(train.sum()), n_test=int(test.sum()),
            V1=dict(gamma=GAMMAS[g1], T=float(TS[t1_]), kappa=kap(k1), idx=[g1, t1_, k1], train_obj=float(b1),
                    n_tied=n1),
            V2=dict(gamma=GAMMAS[g2], T=float(TS[t2_]), kappa=kap(k2), idx=[g2, t2_, k2], train_obj=float(b2),
                    n_tied=n2),
            V3=dict(sigma=float(SIGMAS[s3]), kappa=kap(k3), idx=[s3, k3], train_obj=float(b3), n_tied=n3))
    for name in ('fits.json', 'folds.json'):
        with open(f'{HERE}/{name}', 'w') as f:
            json.dump(fits, f, indent=1)
    lines = []
    for name in ('fits.json', 'folds.json'):
        with open(f'{HERE}/{name}', 'rb') as f:
            lines.append(f'{hashlib.sha256(f.read()).hexdigest()}  {name}')
    with open(f'{HERE}/FITS.sha256', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    t2 = time.monotonic()
    print(f'S table: {len(rows)} records x (500 + 500 + 125) settings in {t1 - t0:.1f} s; fits: {len(folds)} folds in '
          f'{t2 - t1:.1f} s', flush=True)
    for L, v in fits['folds'].items():
        print(f'{L:30s} train {v["n_train"]:4d} test {v["n_test"]:3d} | V1 g {v["V1"]["gamma"]:2d} T {v["V1"]["T"]:.4f} '
              f'k {v["V1"]["kappa"]} ({v["V1"]["n_tied"]}) | V2 g {v["V2"]["gamma"]:2d} T {v["V2"]["T"]:.4f} '
              f'k {v["V2"]["kappa"]} ({v["V2"]["n_tied"]}) | V3 s {v["V3"]["sigma"]:.4f} k {v["V3"]["kappa"]} '
              f'({v["V3"]["n_tied"]})')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
