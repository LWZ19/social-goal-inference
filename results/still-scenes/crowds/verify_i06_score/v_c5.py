"""Verifier: the C5 gate-1 recount on the widened generator (FROZEN 10), my own counting.

    v_c5.py  -> out/c5.txt, out/c5.json
"""
from __future__ import annotations

import collections
import itertools
import json
import math

import numpy as np

import vlib as V
from sgi.synth import generate as G

ADDED_UNGROUPED = {'A': 1, 'B': 3, 'C': 0, 'E': 1}


def dev_rows(cfg):
    out = []
    for rws in (G._EVAL, G._EXAMPLES, G._EVAL_MORE, G._EXAMPLES_MORE):
        out += [tuple(r) for r in rws if r[0] in cfg.dev_situations]
    return out


def frame(truth):
    b = truth.bounds
    psi = math.atan2(b[1][1] - b[0][1], b[1][0] - b[0][0])
    t = truth.targets[V.T0]
    return psi, np.array(t.anchor[:2], float)


def to_local(pts, psi, a):
    d = np.asarray(pts, float).reshape(-1, 2) - a
    c, s = math.cos(psi), math.sin(psi)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)


def best(P, Q):
    n = len(P)
    bestsum, bestd = None, None
    for perm in itertools.permutations(range(n)):
        d = np.hypot(*(P - Q[list(perm)]).T)
        if bestsum is None or d.sum() < bestsum - 1e-15:
            bestsum, bestd = d.sum(), d
    return bestd


def main():
    cfg = V.load_config()
    rws = dev_rows(cfg)
    man = V.manifest()
    spec = {e['row']: tuple(e['row_spec']) for e in man}
    acc = [e for e in man if e['kind'] == 'variant' and e['accepted']]
    confs = sorted({(e['row'], e['family'], e['level']) for e in acc})
    prim = {r + (None, None) for r in rws} | {spec[r] + (f, l) for r, f, l in confs}
    struct = {r + (None, None) for r in rws} | {spec[r] + (f, None if f in 'AE' else l) for r, f, l in confs}

    def low_row(r):
        return (r[0], 0 if r[0] == 'no_one' else int(r[1]), int(r[3]), bool(r[4]))

    def low_var(r, f, l):
        s = spec[r]
        add = l if f == 'D' else ADDED_UNGROUPED[f]
        return (s[0], 0 if s[0] == 'no_one' else int(s[1]), int(s[3]) + add, bool(s[4]) or f == 'C')
    lower = {low_row(r) for r in rws} | {low_var(*c) for c in confs}
    lines = [f'dev rows {len(rws)}, distinct {len(set(rws))}; variant configurations {len(confs)}',
             f"PRIMARY' {len(prim)} -> {'met' if len(prim) >= 30 else 'borderline' if len(prim) >= 25 else 'not met'}",
             f'STRUCTURE {len(struct)}', f"LOWER' {len(lower)}"]
    # EMPIRICAL
    by = collections.defaultdict(dict)
    for e in acc:
        by[(e['row'], e['family'], e['level'])][e['seed']] = e
    meds = []
    emp = {}
    for c in confs:
        locs = {}
        for s, e in by[c].items():
            inputs, truth = V.load_entry(e)
            psi, a = frame(truth)
            ids = [x['id'] for x in e['added']]
            pos = {p.id: (p.x, p.y) for p in truth.people}
            locs[s] = to_local([pos[i] for i in ids], psi, a)
        ds = []
        for i, j in itertools.combinations(sorted(locs), 2):
            P, Q = locs[i], locs[j]
            m0 = best(P, Q)
            m1 = best(P, Q * np.array([1.0, -1.0]))
            ds += list(m0 if m0.sum() <= m1.sum() else m1)
        if ds:
            meds.append(float(np.median(ds)))
            emp[f'{c}'] = [float(np.median(ds)), float(np.max(ds)), len(ds)]
    lines.append(f'EMPIRICAL: configurations with >= 2 seeds {len(meds)}; median of per-configuration medians '
                 f'{np.median(meds):.3f} m; range {min(meds):.3f}-{max(meds):.3f}')
    base_d = []
    bases = {(e['seed'], e['row']): e for e in man if e['kind'] == 'base'}
    for row in sorted({e['row'] for e in bases.values()}):
        L = {}
        for s in (0, 1, 2):
            inputs, truth = V.load_entry(bases[(s, row)])
            psi, a = frame(truth)
            g = truth.group_for('e0', V.T0)
            L[s] = to_local([(p.x, p.y) for p in truth.people if p.group == g], psi, a)
        d = [x for i, j in itertools.combinations((0, 1, 2), 2) for x in np.hypot(*(L[i] - L[j]).T)]
        base_d.append((row, float(np.median(d))))
    lines.append('base members by id, median per row: ' + ', '.join(f'eval_{r:02d} {m:.3f}' for r, m in base_d))
    (V.OUT / 'c5.txt').write_text('\n'.join(lines) + '\n')
    (V.OUT / 'c5.json').write_text(json.dumps({'primary': len(prim), 'structure': len(struct), 'lower': len(lower),
                                               'confs': len(confs), 'emp': emp, 'lower_tuples': sorted(map(list, lower))},
                                              indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
