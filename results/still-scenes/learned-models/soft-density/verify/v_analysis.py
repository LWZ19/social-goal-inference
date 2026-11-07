"""I04 independent verifier, steps 2-4: checks, fold fits (training records only), held-out scoring, tables and the
criterion verdicts, from vcache_shard*.pkl (v_cache.py). No compute-agent code is imported; fits.json and
summary.json of the compute agent are read only at the end, to compare.

    cd WT && PYTHONPATH= ... .venv/bin/python v_analysis.py > v_analysis.txt
"""
from __future__ import annotations

import collections
import json
import math
import pickle

import numpy as np

V = 'results/still-scenes/learned-models/soft-density/verify'
C = 'results/still-scenes/learned-models/soft-density'
I14D = 'results/still-scenes/metrics-and-statistics/metric-v0'
GAMMAS = (1, 3, 10, 30)
TS = np.geomspace(0.005, 5, 25)
SIGMAS = np.geomspace(0.05, 3, 25)
KAPPAS = (4.0, 8.0, 16.0, 32.0, math.inf)
FLOOR = 1e-3
PRIMARY = ['old_qwen', 'old_glm', 'old_deepseek', 'watch_qwen', 'watch_glm', 'watch_deepseek']
NEW = ['new_qwen', 'new_glm', 'new_deepseek', 'new_haiku', 'new_sonnet', 'new_gemini']
OLDREAL = PRIMARY[:3]
S1_EXPECT = {'old_qwen': (134, 134), 'old_glm': (96, 99), 'old_deepseek': (109, 123), 'watch_qwen': (32, 35),
             'watch_glm': (23, 29), 'watch_deepseek': (23, 33)}


def load():
    rows, counts, mism, std, other = [], collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    for k in (0, 1):
        with open(f'{V}/vcache_shard{k}.pkl', 'rb') as f:
            d = pickle.load(f)
        rows += d['rows']
        counts.update(d['counts'])
        mism.update(d['mism'])
        std.update(d['std'])
        other.update(d['other_sit'])
    rows.sort(key=lambda r: r['key'])
    return rows, counts, mism, std, other


def auroc(score, label):
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    npos, nneg = label.sum(), (~label).sum()
    if npos == 0 or nneg == 0:
        return float('nan')
    # direct pairwise count (ties half)
    sp, sn = score[label], score[~label]
    gt = (sp[:, None] > sn[None, :]).sum()
    eq = (sp[:, None] == sn[None, :]).sum()
    return float((gt + 0.5 * eq) / (npos * nneg))


def main():
    rows, counts, mism, std, other = load()
    P = print
    P('I04 independent verification (verifier code only; FROZEN.md sha256 8e0aac98...1882)')
    P(f'records: {len(rows)}; distinct keys {len(set(r["key"] for r in rows))}')
    P(f'standard-path left out per cell: {dict(std)}; other situations left out: {dict(other)}')
    P(f'check counts: {dict(counts)}')
    P(f'mismatches: {dict(mism)}')
    rep_ok = [r for r in rows if r['repro']['ok']]
    P(f'reproduction: {len(rep_ok)} of {len(rows)}; worst move {max(r["repro"]["worst_m"] for r in rows):.3e} m; '
      f'worst heading {max(r["repro"]["worst_h"] for r in rows):.3e} rad; picks {collections.Counter(r["repro"]["npicks"] for r in rows)}')
    P(f'official top-1 == recorded success: {sum(r["top1"] == r["rec_success"] for r in rows)} of {len(rows)}; '
      f'in-region top replica == package: {sum(r["inreg_top_rep"] == r["inreg_top_pkg"] for r in rows)}')
    P(f'multi-region right-target reachable cells: {sum(r["n_multi"] for r in rows)}; degenerate-heading cells: '
      f'{sum(r["n_degen"] for r in rows)}')
    rows = rep_ok
    cellof = np.array([r['cell'] for r in rows])
    layout = np.array([r['layout'] for r in rows])
    prim = np.isin(cellof, PRIMARY)
    P('\n== records per cell ==')
    for c in PRIMARY + NEW:
        m = cellof == c
        P(f'{c:15s} n {m.sum():4d}  layouts {len(set(layout[m]))}')
    P(f'primary n {prim.sum()}, new n {(~prim).sum()}')

    # ---- S1 reproduction
    P('\n== S1 (lexicographic hit@1..5, package) ==')
    lex = {}
    for c in PRIMARY + NEW:
        m = [r for r in rows if r['cell'] == c]
        h = np.sum([r['lex_hit'] for r in m], axis=0).tolist()
        lex[c] = h
        exp = S1_EXPECT.get(c)
        P(f'{c:15s} n {len(m):4d} hit {h}' + (f'  expect top1/any5 {exp} -> {"match" if (h[0], h[4]) == exp else "DIFF"}' if exp else ''))

    # ---- cross-check against I14 v0 rows
    i14 = {}
    for fn in ('rows_oldwatch.jsonl', 'rows_new.jsonl'):
        with open(f'{I14D}/{fn}') as f:
            for line in f:
                d = json.loads(line)
                i14[f"{d['run']}/{d['scene']}/{d['ep']}/{d['trial']}"] = d
    nm = dn = 0
    mx_room = mx_disc = 0.0
    for r in rows:
        d = i14.get(r['key'])
        if d is None:
            continue
        nm += 1
        dn += int(d['n_sel'] != r['N'] or d['n_disc'] != r['n_disc'])
        mx_room = max(mx_room, abs(d['rho_room'] - r['rho_room']))
        mx_disc = max(mx_disc, abs(d['rho_disc'] - r['rho_disc']))
    P(f'\nI14 cross-check: matched {nm} of {len(rows)}; F/disc size differs {dn}; max |rho_room diff| {mx_room:.2e}; '
      f'max |rho_disc diff| {mx_disc:.2e}')
    P(f'primary top-1 {sum(r["top1"] for r in rows if r["cell"] in PRIMARY)}, right target '
      f'{sum(r["target_ok"] for r in rows if r["cell"] in PRIMARY)}, sum rho_room '
      f'{sum(r["rho_room"] for r in rows if r["cell"] in PRIMARY):.3f}')

    # ---- S table
    S1 = np.array([r['out']['V1']['S'] for r in rows])          # (n, 4, 25, 5)
    S2 = np.array([r['out']['V2']['S'] for r in rows])
    S3 = np.array([r['out']['V3']['S'] for r in rows])          # (n, 25, 5)
    L1, L2, L3 = np.log(FLOOR + S1), np.log(FLOOR + S2), np.log(FLOOR + S3)
    # tie order: largest T/sigma, then smallest kappa, then smallest gamma
    order12 = sorted([(g, t, k) for g in range(4) for t in range(25) for k in range(5)],
                     key=lambda s: (-TS[s[1]], KAPPAS[s[2]], GAMMAS[s[0]]))
    order3 = sorted([(s, k) for s in range(25) for k in range(5)], key=lambda s: (-SIGMAS[s[0]], KAPPAS[s[1]]))

    def pick(obj, order):
        best = max(obj[s] for s in order)
        tied = [s for s in order if obj[s] >= best - 1e-12]
        return tied[0], best, len(tied)

    folds = sorted(set(layout[prim]))
    P(f'\n== fold fits ({len(folds)} folds; training = primary records of the other layouts) ==')
    fit = {}
    for L in folds:
        train = prim & (layout != L)
        assert not np.any(train & (layout == L))
        o1, o2, o3 = L1[train].mean(axis=0), L2[train].mean(axis=0), L3[train].mean(axis=0)
        f1, b1, n1 = pick(o1, order12)
        f2, b2, n2 = pick(o2, order12)
        f3, b3, n3 = pick(o3, order3)
        # second best margin for robustness
        def margin(obj, best_s, order):
            vals = sorted((obj[s] for s in order), reverse=True)
            return vals[0] - vals[1]
        fit[L] = dict(V1=f1, V2=f2, V3=f3, b=(b1, b2, b3), n=(n1, n2, n3),
                      margin=(margin(o1, f1, order12), margin(o2, f2, order12), margin(o3, f3, order3)),
                      ntrain=int(train.sum()), ntest=int((prim & (layout == L)).sum()))
        P(f'{L:30s} train {train.sum():4d} test {(prim & (layout == L)).sum():3d} | V1 g {GAMMAS[f1[0]]:2d} '
          f'T {TS[f1[1]]:.4f} k {KAPPAS[f1[2]]} obj {b1:.6f} tied {n1} margin {fit[L]["margin"][0]:.2e} | V2 g '
          f'{GAMMAS[f2[0]]:2d} T {TS[f2[1]]:.4f} k {KAPPAS[f2[2]]} tied {n2} | V3 s {SIGMAS[f3[0]]:.4f} '
          f'k {KAPPAS[f3[1]]} obj {b3:.6f} tied {n3} margin {fit[L]["margin"][2]:.2e}')
    # compare with the compute agent's fits.json
    with open(f'{C}/fits.json') as f:
        cf = json.load(f)
    diff = 0
    for L in folds:
        c = cf['folds'][L]
        mine = fit[L]
        for var in ('V1', 'V2'):
            if tuple(c[var]['idx']) != tuple(mine[var]):
                diff += 1
                P(f'  FIT DIFF {L} {var}: compute {c[var]["idx"]} mine {mine[var]}')
        if tuple(c['V3']['idx']) != tuple(mine['V3']):
            diff += 1
            P(f'  FIT DIFF {L} V3: compute {c["V3"]["idx"]} mine {mine["V3"]}')
        for j, var in enumerate(('V1', 'V2', 'V3')):
            if abs(c[var]['train_obj'] - mine['b'][j]) > 1e-9:
                P(f'  OBJ DIFF {L} {var}: {c[var]["train_obj"]} vs {mine["b"][j]}')
    P(f'fold settings differing from fits.json: {diff} (of {3 * len(folds)}); folds in fits.json: {sorted(cf["folds"]) == folds}')

    # ---- held-out per record
    n = len(rows)
    lay_fold = {L: fit[L] for L in folds}
    missing = sorted(set(layout) - set(folds))
    P(f'layouts with no fold (records there cannot be scored): {missing}')
    hs = {v: np.zeros(n) for v in ('V0', 'V1', 'V2', 'V3')}
    hin = {v: np.zeros(n) for v in ('V0', 'V1', 'V2', 'V3')}
    hco = {v: np.zeros(n) for v in ('V1', 'V2', 'V3')}
    hfb = {v: [dict() for _ in range(n)] for v in ('V0', 'V1', 'V2', 'V3')}
    hhit = {v: np.zeros((n, 5), dtype=int) for v in ('V1', 'V2', 'V3')}
    for i, r in enumerate(rows):
        f = lay_fold[r['layout']]
        g, t, k = f['V1']
        hs['V1'][i] = r['out']['V1']['S'][g, t, k]
        hin['V1'][i] = r['out']['V1']['inreg'][g, t]
        hco['V1'][i] = r['out']['V1']['conc'][g, t]
        hfb['V1'][i] = {kd: r['out']['V1']['forb'][kd][g, t] for kd in r['forb_kinds']}
        hhit['V1'][i] = r['dens'][('V1', GAMMAS[g])]['hit']
        g2, t2, k2 = f['V2']
        hs['V2'][i] = r['out']['V2']['S'][g2, t2, k2]
        hin['V2'][i] = r['out']['V2']['inreg'][g2, t2]
        hco['V2'][i] = r['out']['V2']['conc'][g2, t2]
        hfb['V2'][i] = {kd: r['out']['V2']['forb'][kd][g2, t2] for kd in r['forb_kinds']}
        hhit['V2'][i] = r['dens'][('V2', GAMMAS[g2])]['hit']
        s3, k3 = f['V3']
        hs['V3'][i] = r['out']['V3']['S'][s3, k3]
        hin['V3'][i] = r['out']['V3']['inreg'][s3]
        hco['V3'][i] = r['out']['V3']['conc'][s3]
        hfb['V3'][i] = {kd: r['out']['V3']['forb'][kd][s3] for kd in r['forb_kinds']}
        hhit['V3'][i] = r['dens'][('V3', None)]['hit']
        hs['V0'][i] = float(r['top1'])
        hin['V0'][i] = float(r['inreg_top_rep'])
        hfb['V0'][i] = {kd: float(r['forb_top_pkg'][kd]) for kd in r['forb_kinds']}
    v3f_diff = sum(r['dens'][('V3', None)]['hit'] != r['dens'][('V3f', None)]['hit'] for r in rows)
    v3f_pl = sum(r['dens'][('V3', None)]['places'] != r['dens'][('V3f', None)]['places'] for r in rows)
    P(f'V3 density places, integer vs float distance ranking: place lists differ in {v3f_pl} records, hit lists in {v3f_diff}')
    top1 = np.array([r['top1'] for r in rows])
    rho_room = np.array([r['rho_room'] for r in rows])
    rho_disc = np.array([r['rho_disc'] for r in rows])
    hl = {v: np.log(FLOOR + hs[v]) for v in hs}

    def stats(m):
        o = {}
        for v in ('V0', 'V1', 'V2', 'V3'):
            o[v] = dict(meanS=hs[v][m].mean(), lift_room=hs[v][m].sum() / rho_room[m].sum(),
                        lift_disc=hs[v][m].sum() / rho_disc[m].sum(), logs=hl[v][m].mean(), inreg=hin[v][m].mean())
        for v in ('V1', 'V2', 'V3'):
            o[v]['conc'] = hco[v][m].mean()
            o[v]['conc09'] = int((hco[v][m] > 0.9).sum())
            o[v]['conc0999'] = int((hco[v][m] > 0.999).sum())
            o[v]['hit'] = hhit[v][m].sum(axis=0).tolist()
        return o

    groups = [(c, cellof == c) for c in PRIMARY] + [('POOLED', prim)] + [(c, cellof == c) for c in NEW]
    res = {}
    P('\n== held-out: n, top1 | mean S V0/V1/V2/V3 | Lift_room | Lift_disc | mean log ==')
    for name, m in groups:
        o = stats(m)
        res[name] = o
        f = lambda key, fmt: ' '.join(format(o[v][key], fmt) for v in ('V0', 'V1', 'V2', 'V3'))
        P(f'{name:15s} {m.sum():4d} {int(top1[m].sum()):4d} | {f("meanS", ".3f")} | {f("lift_room", ".2f")} | '
          f'{f("lift_disc", ".2f")} | {f("logs", ".3f")}')
    P('\n== held-out mass: in-region V0/V1/V2/V3 | conc V1/V2/V3 | conc>0.9 V1/V2/V3 | V3 conc>0.999 ==')
    for name, m in groups:
        o = res[name]
        P(f'{name:15s} {m.sum():4d} | ' + ' '.join(f'{o[v]["inreg"]:.3f}' for v in ('V0', 'V1', 'V2', 'V3')) + ' | ' +
          ' '.join(f'{o[v]["conc"]:.3f}' for v in ('V1', 'V2', 'V3')) + ' | ' +
          ' '.join(f'{o[v]["conc09"]} ({100 * o[v]["conc09"] / m.sum():.1f}%)' for v in ('V1', 'V2', 'V3')) +
          f' | {o["V3"]["conc0999"]}')
    P('\n== forbidden mass (records whose episode has the kind) V1 V2 V3 | top room ==')
    for name, m in groups:
        idx = np.flatnonzero(m)
        kinds = sorted({kd for i in idx for kd in rows[i]['forb_kinds']})
        for kd in kinds:
            ii = [i for i in idx if kd in rows[i]['forb_kinds']]
            vals = [np.mean([hfb[v][i][kd] for i in ii]) for v in ('V1', 'V2', 'V3', 'V0')]
            room = np.mean([rows[i]['forb_room'][kd] for i in ii])
            P(f'{name:15s} {kd:15s} n {len(ii):4d} | {vals[0]:.4f} {vals[1]:.4f} {vals[2]:.4f} | {vals[3]:.4f} {room:.4f}')
    P('\n== density hit@1..5: lexicographic | V1 | V2 | V3 ==')
    for name, m in groups:
        lx = np.sum([rows[i]['lex_hit'] for i in np.flatnonzero(m)], axis=0).tolist()
        o = res[name]
        res[name]['lex'] = lx
        P(f'{name:15s} {m.sum():4d} | {lx} | {o["V1"]["hit"]} | {o["V2"]["hit"]} | {o["V3"]["hit"]}')
    v2short = sum(len(r['dens'][('V2', GAMMAS[lay_fold[r['layout']]['V2'][0]])]['places']) < 5 for r in rows if r['cell'] in PRIMARY)
    P(f'V2 short lists (primary, held-out gamma): {v2short}')

    # ---- AUROC
    P('\n== AUROC for official top-1: c_V1 c_V2 c_V3 | 1-excl 1-far 1-unusable ==')
    au = {}
    for name, m in groups:
        y = top1[m]
        a = [auroc(hco[v][m], y) for v in ('V1', 'V2', 'V3')]
        fl = [auroc(1 - np.array([rows[i]['flags'][f] for i in np.flatnonzero(m)], dtype=float), y)
              for f in ('exclusion_broken', 'far_from_target', 'constraint_unusable')]
        au[name] = a
        P(f'{name:15s} {m.sum():4d} {int(y.sum()):4d} | ' + ' '.join(f'{x:.3f}' for x in a) + ' | ' +
          ' '.join(f'{x:.3f}' for x in fl))
    c3 = hco['V3'][prim]
    P(f'c_V3 range over primary: [{c3.min():.7f}, {c3.max():.7f}]')

    # ---- bootstrap
    real_l = sorted(set(layout[prim & np.isin(cellof, OLDREAL)]))
    watch_l = sorted(set(layout[prim & ~np.isin(cellof, OLDREAL)]))
    P(f'\nbootstrap strata: {len(real_l)} realistic, {len(watch_l)} watching layouts')
    rng = np.random.default_rng(20261018)
    B = 2000
    RD = rng.integers(0, len(real_l), size=(B, len(real_l)))
    WD = rng.integers(0, len(watch_l), size=(B, len(watch_l)))
    lidx = {L: i for i, L in enumerate(real_l)}
    widx = {L: i for i, L in enumerate(watch_l)}

    def weights(b, m):
        cr = np.bincount(RD[b], minlength=len(real_l))
        cw = np.bincount(WD[b], minlength=len(watch_l))
        w = np.zeros(n)
        for i in np.flatnonzero(m):
            L = layout[i]
            w[i] = cr[lidx[L]] if L in lidx else cw[widx[L]]
        return w

    P('== bootstrap 95%: V1-V3 mean log | V1-V3 Lift_room | V2-V3 mean log | V2-V3 Lift_room ==')
    boot = {}
    for name, m in groups[:7]:
        vals = np.zeros((B, 4))
        for b in range(B):
            w = weights(b, m)
            sw = w.sum()
            rr = (w * rho_room).sum()
            vals[b, 0] = (w * (hl['V1'] - hl['V3'])).sum() / sw
            vals[b, 1] = (w * (hs['V1'] - hs['V3'])).sum() / rr
            vals[b, 2] = (w * (hl['V2'] - hl['V3'])).sum() / sw
            vals[b, 3] = (w * (hs['V2'] - hs['V3'])).sum() / rr
        lo, hi = np.percentile(vals, [2.5, 97.5], axis=0)
        pt = [hl['V1'][m].mean() - hl['V3'][m].mean(), res[name]['V1']['lift_room'] - res[name]['V3']['lift_room'],
              hl['V2'][m].mean() - hl['V3'][m].mean(), res[name]['V2']['lift_room'] - res[name]['V3']['lift_room']]
        boot[name] = (pt, lo.tolist(), hi.tolist())
        P(f'{name:15s} ' + ' | '.join(f'{pt[j]:+.3f} [{lo[j]:+.3f}, {hi[j]:+.3f}]' for j in range(4)))

    # ---- criteria
    P('\n== criteria ==')
    verdicts = {}
    for var in ('V1', 'V2'):
        o = res['POOLED']
        a = o[var]['lift_room'] >= 10
        b = o[var]['logs'] > o['V3']['logs']
        c_one = all(res[c][var]['hit'][4] >= res[c]['lex'][4] - 2 for c in PRIMARY)
        c_two = all(abs(res[c][var]['hit'][4] - res[c]['lex'][4]) <= 2 for c in PRIMARY)
        aus = [auroc(hco[var][cellof == c], top1[cellof == c]) for c in OLDREAL]
        d = sum(x >= 0.70 for x in aus) >= 2
        k1 = not b
        share = o[var]['conc09'] / prim.sum()
        k2 = share > 0.80
        verdict = 'kill' if (k1 or k2) else ('success' if (a and b and c_one and d) else 'neither')
        verdicts[var] = verdict
        P(f'{var}: (a) Lift_room {o[var]["lift_room"]:.3f} >= 10: {a}; (b) logs {o[var]["logs"]:.4f} > V3 '
          f'{o["V3"]["logs"]:.4f}: {b}; (c) one-sided {c_one} two-sided {c_two} '
          f'{[(res[c][var]["hit"][4], res[c]["lex"][4]) for c in PRIMARY]}; (d) {["%.3f" % x for x in aus]}: {d}; '
          f'kill1 {k1}; kill2 share {o[var]["conc09"]}/{prim.sum()} = {share:.3f}: {k2}; better-than-expected '
          f'Lift_room > V3: {o[var]["lift_room"] > o["V3"]["lift_room"]}, AUROC>=0.75 in 2 of 3: '
          f'{sum(x >= 0.75 for x in aus) >= 2} -> {verdict.upper()}')

    # ---- post hoc in-sample (no folds)
    o1, o3 = L1[prim].mean(axis=0), L3[prim].mean(axis=0)
    b1 = np.unravel_index(np.argmax(o1), o1.shape)
    b3 = np.unravel_index(np.argmax(o3), o3.shape)
    P(f'\npost hoc in-sample: best V1 {o1[b1]:.4f} at g {GAMMAS[b1[0]]} T {TS[b1[1]]:.4f} k {KAPPAS[b1[2]]}; best V3 '
      f'{o3[b3]:.4f} at s {SIGMAS[b3[0]]:.4f} k {KAPPAS[b3[1]]}; V0 {np.log(FLOOR + top1[prim]).mean():.4f}')
    P('V3 in-sample best-kappa log score by sigma: ' + ', '.join(f'{SIGMAS[s]:.3f}:{o3[s].max():.3f}' for s in (0, 6, 12, 18, 24)))

    # ---- by situation (post hoc)
    sit = np.array([r['sit'] for r in rows])
    P('\npost hoc by situation (primary): n | mean S V1/V3')
    for s_ in sorted(set(sit[prim])):
        m = prim & (sit == s_)
        P(f'  {s_:20s} {m.sum():4d} | {hs["V1"][m].mean():.3f} {hs["V3"][m].mean():.3f}')

    with open(f'{V}/v_summary.json', 'w') as f:
        json.dump(dict(res={k: {v: {kk: vv for kk, vv in d.items()} for v, d in o.items() if v != 'lex'} | {'lex': o['lex']}
                            for k, o in res.items()},
                       auroc=au, boot=boot, verdicts=verdicts,
                       fits={L: dict(V1=list(map(int, fit[L]['V1'])), V2=list(map(int, fit[L]['V2'])),
                                     V3=list(map(int, fit[L]['V3'])), obj=list(fit[L]['b']), tied=list(fit[L]['n']),
                                     margin=list(fit[L]['margin'])) for L in folds}),
                  f, indent=1, default=float)

    # ---- compare with the compute agent's per-record rows
    P('\n== per-record comparison with the compute agent rows_<cell>.jsonl ==')
    theirs = {}
    for c in PRIMARY + NEW:
        with open(f'{C}/rows_{c}.jsonl') as f:
            for line in f:
                d = json.loads(line)
                theirs[d['key']] = d
    P(f'their rows {len(theirs)}; keys in common {len(set(theirs) & set(r["key"] for r in rows))}')
    mx = collections.defaultdict(float)
    where = {}
    ndiff_hit = collections.Counter()
    for i, r in enumerate(rows):
        d = theirs[r['key']]
        assert d['layout'] == r['layout'] and d['cell'] == r['cell']
        for v in ('V0', 'V1', 'V2', 'V3'):
            for fld, mine in (('S', hs[v][i]), ('inreg_mass', hin[v][i])):
                e = abs(d[v][fld] - mine)
                if e > mx[(v, fld)]:
                    mx[(v, fld)], where[(v, fld)] = e, r['key']
        for v in ('V1', 'V2', 'V3'):
            e = abs(d[v]['conc'] - hco[v][i])
            if e > mx[(v, 'conc')]:
                mx[(v, 'conc')], where[(v, 'conc')] = e, r['key']
            if list(d[v]['hit']) != list(hhit[v][i]):
                ndiff_hit[v] += 1
            for kd in r['forb_kinds']:
                e = abs(d[v]['forb'][kd] - hfb[v][i][kd])
                mx[(v, 'forb')] = max(mx[(v, 'forb')], e)
        for fld in ('rho_room', 'rho_disc', 'top1', 'lex_hit', 'n_cells', 'n_kept', 'matched'):
            mine = {'n_cells': r['N']}.get(fld, r.get(fld))
            if d[fld] != mine and not (isinstance(mine, float) and abs(d[fld] - mine) < 1e-12):
                ndiff_hit[fld] += 1
    P('max |their - mine| per variant/field: ' + ', '.join(f'{k[0]}.{k[1]} {v:.2e}' for k, v in sorted(mx.items())))
    P(f'records whose hit list or scalar field differs: {dict(ndiff_hit)}')
    return rows, hs, hco, theirs


if __name__ == '__main__':
    main()
