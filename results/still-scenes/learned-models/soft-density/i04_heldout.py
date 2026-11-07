"""I04 steps 4-5 (FROZEN.md sections 7-9): held-out scoring with each record's fold setting, the tables, the
criteria and the verdict. Refuses to run unless fits.json and folds.json match FITS.sha256 (written before this).

Writes rows_<cell>.jsonl (one line per record) and tables.txt. No fitting happens here.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import sys
import time

import numpy as np

HERE = 'results/2026-10-17-beyond-vlm/phase1d/i04'
sys.path.insert(0, HERE)
from i04_fit import GAMMAS, KAPPAS, SIGMAS, TS, cells_of, energy_mm, load_meta, load_npz, psucc  # noqa: E402

PRIMARY = ['old_qwen', 'old_glm', 'old_deepseek', 'watch_qwen', 'watch_glm', 'watch_deepseek']
OLD = PRIMARY[:3]
NEW = ['new_qwen', 'new_glm', 'new_deepseek', 'new_haiku', 'new_sonnet', 'new_gemini']
VARS = ('V0', 'V1', 'V2', 'V3')
FLOOR = 1e-3
B = 2000
BOOT_SEED = 20261018
FLAGS = ('exclusion_broken', 'far_from_target', 'constraint_unusable')


def verify_fits():
    with open(f'{HERE}/FITS.sha256') as f:
        for line in f:
            digest, name = line.split()
            with open(f'{HERE}/{name}', 'rb') as g:
                assert hashlib.sha256(g.read()).hexdigest() == digest, f'{name} does not match FITS.sha256'
    with open(f'{HERE}/fits.json') as f:
        return json.load(f)


def hits_of(succ):
    h = np.logical_or.accumulate(np.array(succ, dtype=bool)).astype(int).tolist()
    return h + [h[-1]] * (5 - len(h))               # a short list carries its last value forward


def score_record(r, fold, s_tab):
    z = load_npz(r)
    N = r['n_cells']
    P = psucc(z)
    succ = z['succ_idx'].astype(np.int64)
    reg = z['reg_idx'].astype(np.int64)
    near = z['near_top'].astype(np.int64)
    forb = {kd: z[f'forb__{kd}'].astype(np.int64) for kd in r['kinds']}
    kept = np.zeros(N, dtype=bool)
    kept[z['kept']] = True
    cells = cells_of(r, z)
    top = int(z['picks'][0])
    out = dict(key=r['key'], cell=r['cell'], set=r['set'], model=r['model'], arm=r['arm'], run=r['run'],
               scene=r['scene'], ep=r['ep'], trial=r['trial'], sit=r['sit'], layout=r['layout'], fold=r['layout'],
               n_cells=N, n_kept=r['n_kept'], matched=r['matched'], target_ok=r['target_ok'], top1=r['top1'],
               lex_hit=r['lex_hit'], rho_room=r['rho_room'], rho_disc=r['rho_disc'], n_disc=r['n_disc'],
               flags=r['flags'], kinds=r['kinds'], top_forb=r['top_forb'], room_forb=r['room_forb'])
    for V in ('V1', 'V2', 'V3'):
        st = fold[V]
        if V == 'V3':
            si, ki = st['idx']
            d2 = (cells[:, 0] - cells[top, 0]) ** 2 + (cells[:, 1] - cells[top, 1]) ** 2
            w = np.exp(-d2 / (2.0 * SIGMAS[si] ** 2))
            p = w / w.sum()
            dens = r['dens']['V3']
            setting = dict(sigma=float(SIGMAS[si]), kappa=st['kappa'])
            s_ref = s_tab['S3'][si, ki]
        else:
            gi, ti, ki = st['idx']
            E = energy_mm(z, GAMMAS[gi])
            mask = kept if V == 'V2' else np.ones(N, dtype=bool)
            e = (E[mask] - E[mask].min()).astype(np.float64) / 1000.0
            w = np.exp(-e / TS[ti])
            p = np.zeros(N)
            p[mask] = w / w.sum()
            dens = r['dens'][f'{V}_g{GAMMAS[gi]}']
            setting = dict(gamma=GAMMAS[gi], T=float(TS[ti]), kappa=st['kappa'])
            s_ref = s_tab[f'S{V[1]}'][gi, ti, ki]
        S = float(p[succ] @ P[:, ki])
        assert abs(S - s_ref) <= 1e-9, (r['key'], V, S, s_ref)
        out[V] = dict(setting=setting, S=S, log=math.log(FLOOR + S), inreg_mass=float(p[reg].sum()),
                      conc=float(p[near].sum()), forb={kd: float(p[ix].sum()) for kd, ix in forb.items()},
                      hit=hits_of(dens['succ_rep']), hit_pkg=hits_of(dens['succ_pkg']), n_places=len(dens['places']))
    in_reg_top = bool(np.isin(top, reg))
    out['V0'] = dict(setting='point mass on the lexicographic place, kappa inf', S=float(r['top1']),
                     log=math.log(FLOOR + float(r['top1'])), inreg_mass=float(in_reg_top), conc=1.0,
                     forb={kd: float(np.isin(top, ix)) for kd, ix in forb.items()})
    return out


# --------------------------------------------------------------------------------------------------- aggregates
def agg(rs, V):
    n = len(rs)
    S = np.array([x[V]['S'] for x in rs])
    rr = np.array([x['rho_room'] for x in rs])
    rd = np.array([x['rho_disc'] for x in rs])
    lg = np.array([x[V]['log'] for x in rs])
    im = np.array([x[V]['inreg_mass'] for x in rs])
    cc = np.array([x[V]['conc'] for x in rs])
    return dict(n=n, sumS=S.sum(), meanS=S.mean(), lift_room=S.sum() / rr.sum(), lift_disc=S.sum() / rd.sum(),
                log=lg.mean(), inreg=im.mean(), conc=cc.mean(), conc90=float((cc > 0.9).mean()),
                n_conc90=int((cc > 0.9).sum()))


def auroc(score, label):
    score, label = np.asarray(score, dtype=np.float64), np.asarray(label, dtype=bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def boot_setup(prim_rows):
    real = sorted({x['layout'] for x in prim_rows if x['set'] == 'old'})
    watch = sorted({x['layout'] for x in prim_rows if x['set'] == 'watch'})
    rng = np.random.default_rng(BOOT_SEED)
    rd = rng.integers(0, len(real), size=(B, len(real)))      # drawn first: the realistic stratum
    wd = rng.integers(0, len(watch), size=(B, len(watch)))    # then the watching stratum
    cr = np.stack([np.bincount(d, minlength=len(real)) for d in rd])
    cw = np.stack([np.bincount(d, minlength=len(watch)) for d in wd])
    return real, watch, cr, cw


def boot_diff(rs, real, watch, cr, cw, A, Bv):
    """Percentile 95% intervals of (A - B) on the mean log score and on Lift_room, layout clusters, stratified."""
    def per_layout(lays):
        M = np.zeros((len(lays), 6))
        for i, L in enumerate(lays):
            sub = [x for x in rs if x['layout'] == L]
            M[i] = [len(sub), sum(x[A]['log'] for x in sub), sum(x[Bv]['log'] for x in sub),
                    sum(x[A]['S'] for x in sub), sum(x[Bv]['S'] for x in sub), sum(x['rho_room'] for x in sub)]
        return M
    tot = cr @ per_layout(real) + cw @ per_layout(watch)
    ok = (tot[:, 0] > 0) & (tot[:, 5] > 0)
    dlog = (tot[ok, 1] - tot[ok, 2]) / tot[ok, 0]
    dlift = (tot[ok, 3] - tot[ok, 4]) / tot[ok, 5]
    return (np.percentile(dlog, [2.5, 97.5]), np.percentile(dlift, [2.5, 97.5]), int((~ok).sum()))


def boot_lift(rs, real, watch, cr, cw, V):
    def per_layout(lays):
        M = np.zeros((len(lays), 2))
        for i, L in enumerate(lays):
            sub = [x for x in rs if x['layout'] == L]
            M[i] = [sum(x[V]['S'] for x in sub), sum(x['rho_room'] for x in sub)]
        return M
    tot = cr @ per_layout(real) + cw @ per_layout(watch)
    ok = tot[:, 1] > 0
    return np.percentile(tot[ok, 0] / tot[ok, 1], [2.5, 97.5])


def main():
    t0 = time.monotonic()
    fits = verify_fits()
    meta = load_meta()
    n_meta = len(meta)
    rows = [r for r in meta if r['repro_ok']]
    st = np.load(f'{HERE}/s_table.npz')
    pos = {k: i for i, k in enumerate(st['keys'].tolist())}
    S1, S2, S3 = st['S1'], st['S2'], st['S3']
    scored = []
    for r in rows:
        fold = fits['folds'][r['layout']]
        i = pos[r['key']]
        scored.append(score_record(r, fold, dict(S1=S1[i], S2=S2[i], S3=S3[i])))
    by = collections.defaultdict(list)
    for x in scored:
        by[x['cell']].append(x)
    for c, xs in by.items():
        with open(f'{HERE}/rows_{c}.jsonl', 'w') as f:
            for x in xs:
                f.write(json.dumps(x) + '\n')
    t1 = time.monotonic()
    prim = [x for x in scored if x['cell'] in PRIMARY]
    groups = [(c, by[c]) for c in PRIMARY] + [('POOLED primary', prim)] + [(c, by[c]) for c in NEW]

    out = []
    p = out.append
    p('I04 phase 1d tables. Protocol: FROZEN.md (sha256 8e0aac98...1882), fits: fits.json (FITS.sha256, written')
    p('before any held-out metric). Every number is a held-out number: each record is scored with the setting of the')
    p('fold that left out its layout (new-wording records: the fold of their own layout; they are in no fit pool).')
    p('Every score measures agreement with the project\'s own reference regions and heading rule, not social')
    p('correctness. Primary cells: old-wording realistic (s19 Qwen, s19 GLM, s20 DeepSeek) and watching (s19/s20')
    p('dev_seed0-2, audience). New wording is reported beside, not judged; its people answers for Qwen, GLM and')
    p('DeepSeek are cache copies of s19/s20, so old and new rows of a model are not independent.')
    p(f'Records: {n_meta} goal-step records in the cache; {len(rows)} reproduce (left out: {n_meta - len(rows)}).')

    # ---- T1 fitted settings
    p('\n== T1. Fitted setting per fold (training records only; 19 folds; eval_04 has no record and no fold) ==')
    p(f'{"fold (left-out layout)":30s} {"train":>5s} {"test":>4s} | {"V1 gamma T kappa (tied)":28s} | '
      f'{"V2 gamma T kappa (tied)":28s} | V3 sigma kappa (tied)')
    for L, v in fits['folds'].items():
        a, b, c = v['V1'], v['V2'], v['V3']
        p(f'{L:30s} {v["n_train"]:5d} {v["n_test"]:4d} | {a["gamma"]:3d} {a["T"]:8.4f} {str(a["kappa"]):>4s} '
          f'({a["n_tied"]:2d})       | {b["gamma"]:3d} {b["T"]:8.4f} {str(b["kappa"]):>4s} ({b["n_tied"]:2d})       | '
          f'{c["sigma"]:.4f} {str(c["kappa"]):>4s} ({c["n_tied"]})')
    p('T in metres of energy, sigma in metres; (tied) = settings within 1e-12 of the best training objective.')
    cnt = collections.Counter((v['V1']['gamma'], round(v['V1']['T'], 4), v['V1']['kappa']) for v in fits['folds'].values())
    p('V1 settings over the 19 folds: ' + '; '.join(f'gamma {g} T {t} kappa {k}: {n}' for (g, t, k), n in cnt.most_common()))
    cnt = collections.Counter((v['V3']['sigma'], v['V3']['kappa']) for v in fits['folds'].values())
    p('V3 settings over the 19 folds: ' + '; '.join(f'sigma {s:.4f} kappa {k}: {n}' for (s, k), n in cnt.most_common()))

    # ---- T2 success-based measures
    p('\n== T2. Success-based measures (held out): mean S, Lift_room, Lift_disc, mean log score ==')
    p(f'{"cell":16s} {"n":>4s} {"top1":>4s} | {"mean S V0/V1/V2/V3":30s} | {"Lift_room V0/V1/V2/V3":30s} | '
      f'{"Lift_disc V0/V1/V2/V3":30s} | mean ln(1e-3+S) V0/V1/V2/V3')
    for name, rs in groups:
        if not rs:
            continue
        A = {V: agg(rs, V) for V in VARS}
        p(f'{name:16s} {len(rs):4d} {sum(x["top1"] for x in rs):4d} | '
          + ' '.join(f'{A[V]["meanS"]:6.3f}' for V in VARS) + '   | '
          + ' '.join(f'{A[V]["lift_room"]:6.2f}' for V in VARS) + '   | '
          + ' '.join(f'{A[V]["lift_disc"]:6.2f}' for V in VARS) + '   | '
          + ' '.join(f'{A[V]["log"]:7.3f}' for V in VARS))
        if name == 'POOLED primary':
            p('-' * 60)
    p('S = the density\'s P(success) (V0 = the lexicographic place at kappa inf, i.e. official top-1). Lift_room = sum S')
    p('/ sum rho_room (rho_room = share of F cells in the matched target\'s acceptable region, 0 for a wrong target);')
    p('Lift_disc = sum S / sum rho_disc (F cells within 3 m of the recorded target point). Ratios of sums, as I14.')

    # ---- T3 mass measures
    p('\n== T3. Mass measures (held out): in-region mass, concentration share within 0.3 m of the top place ==')
    p(f'{"cell":16s} {"n":>4s} | {"in-region mass V0/V1/V2/V3":30s} | {"mean conc V1/V2/V3":22s} | '
      f'records with conc > 0.9: V1 / V2 / V3')
    for name, rs in groups:
        if not rs:
            continue
        A = {V: agg(rs, V) for V in VARS}
        p(f'{name:16s} {len(rs):4d} | ' + ' '.join(f'{A[V]["inreg"]:6.3f}' for V in VARS) + '   | '
          + ' '.join(f'{A[V]["conc"]:6.3f}' for V in ('V1', 'V2', 'V3')) + ' | '
          + ' / '.join(f'{A[V]["n_conc90"]} ({100 * A[V]["conc90"]:.1f}%)' for V in ('V1', 'V2', 'V3')))
        if name == 'POOLED primary':
            p('-' * 60)
    p('in-region mass = mass on F cells inside the matched target\'s acceptable region (heading, reachability aside).')

    # ---- T4 forbidden mass
    p('\n== T4. Mass in each forbidden kind present (held out), records whose episode has the kind ==')
    p(f'{"cell":16s} {"kind":15s} {"n":>4s} | {"V1":>7s} {"V2":>7s} {"V3":>7s} | {"top place":>9s} {"room share":>10s}')
    for name, rs in groups:
        kinds = sorted({kd for x in rs for kd in x['kinds']})
        for kd in kinds:
            sub = [x for x in rs if kd in x['kinds']]
            p(f'{name:16s} {kd:15s} {len(sub):4d} | '
              + ' '.join(f'{np.mean([x[V]["forb"][kd] for x in sub]):7.4f}' for V in ('V1', 'V2', 'V3'))
              + f' | {np.mean([x["top_forb"][kd] for x in sub]):9.4f} {np.mean([x["room_forb"][kd] for x in sub]):10.4f}')
    p('top place = share of recorded top places inside the kind (package); room share = mean share of F cells in it.')

    # ---- T5 density hit@k
    p('\n== T5. Density hit@k, k = 1..5 (held-out gamma for V1/V2; V3 has no parameter in its ranking) ==')
    p(f'{"cell":16s} {"n":>4s} | {"lexicographic (package)":24s} | {"V1 density":24s} | {"V2 density":24s} | '
      f'{"V3 density":24s} | V2 short lists')
    for name, rs in groups:
        if not rs:
            continue
        lex = [sum(x['lex_hit'][k] for x in rs) for k in range(5)]
        h = {V: [sum(x[V]['hit'][k] for x in rs) for k in range(5)] for V in ('V1', 'V2', 'V3')}
        hp = {V: [sum(x[V]['hit_pkg'][k] for x in rs) for k in range(5)] for V in ('V1', 'V2', 'V3')}
        assert hp == h
        short = sum(x['V2']['n_places'] < 5 for x in rs)
        p(f'{name:16s} {len(rs):4d} | ' + ' '.join(f'{v:4d}' for v in lex) + ' | '
          + ' | '.join(' '.join(f'{v:4d}' for v in h[V]) for V in ('V1', 'V2', 'V3')) + f' | {short}')
        if name == 'POOLED primary':
            p('-' * 60)
    p('Place 1 = the lexicographic place; each next = the highest-density cell >= 0.6 m (-1e-9) from every earlier place,')
    p('at the solver\'s heading; scored by the replica, equal to the package\'s verdict on every density place (checked).')

    # ---- T6 AUROC
    p('\n== T6. Confidence AUROC for official top-1 success (held out; ties count half) ==')
    p(f'{"cell":16s} {"n":>4s} {"succ":>4s} | {"c_V1":>6s} {"c_V2":>6s} {"c_V3":>6s} | ' +
      ' '.join(f'{"1-" + f[:12]:>15s}' for f in FLAGS))
    au = {}
    for name, rs in groups:
        if not rs:
            continue
        lab = [x['top1'] for x in rs]
        a = {V: auroc([x[V]['conc'] for x in rs], lab) for V in ('V1', 'V2', 'V3')}
        a.update({f: auroc([1 - int(x['flags'][f]) for x in rs], lab) for f in FLAGS})
        au[name] = a
        p(f'{name:16s} {len(rs):4d} {sum(lab):4d} | ' + ' '.join(f'{a[V]:6.3f}' for V in ('V1', 'V2', 'V3')) + ' | '
          + ' '.join(f'{a[f]:15.3f}' for f in FLAGS) + ('   (descriptive)' if name == 'POOLED primary' else ''))
        if name == 'POOLED primary':
            p('-' * 60)
    p('c_V = mass within 0.3 m of the top place under the held-out setting; 1-flag = 0/1 score (a raised flag predicts')
    p('failure). nan = one class empty.')

    # ---- T7 bootstrap
    real, watch, cr, cw = boot_setup(prim)
    p(f'\n== T7. Layout-cluster bootstrap, {B} draws, default_rng({BOOT_SEED}), stratified ({len(real)} realistic and '
      f'{len(watch)} watching layouts resampled separately), percentile 95% ==')
    p(f'{"cell":16s} {"n":>4s} | {"V1-V3 mean log [95%]":32s} | {"V1-V3 Lift_room [95%]":32s} | '
      f'{"V2-V3 mean log [95%]":32s} | {"V2-V3 Lift_room [95%]":32s}')
    boots = {}
    for name, rs in [(c, by[c]) for c in PRIMARY] + [('POOLED primary', prim)]:
        A = {V: agg(rs, V) for V in VARS}
        (l13, f13, d13) = boot_diff(rs, real, watch, cr, cw, 'V1', 'V3')
        (l23, f23, d23) = boot_diff(rs, real, watch, cr, cw, 'V2', 'V3')
        boots[name] = (l13, f13, l23, f23)
        p(f'{name:16s} {len(rs):4d} | {A["V1"]["log"] - A["V3"]["log"]:+7.3f} [{l13[0]:+7.3f}, {l13[1]:+7.3f}]       | '
          f'{A["V1"]["lift_room"] - A["V3"]["lift_room"]:+7.2f} [{f13[0]:+7.2f}, {f13[1]:+7.2f}]       | '
          f'{A["V2"]["log"] - A["V3"]["log"]:+7.3f} [{l23[0]:+7.3f}, {l23[1]:+7.3f}]       | '
          f'{A["V2"]["lift_room"] - A["V3"]["lift_room"]:+7.2f} [{f23[0]:+7.2f}, {f23[1]:+7.2f}]'
          + (f'  (dropped draws {d13})' if d13 else ''))
    p('A realistic cell resamples the 13 realistic layouts and a watching cell the 6 watching layouts (6 clusters: crude);')
    p('the pooled row resamples both strata. The same 2,000 draws serve every row (realistic drawn first, then watching).')
    lb = boot_lift(prim, real, watch, cr, cw, 'V1')
    lb3 = boot_lift(prim, real, watch, cr, cw, 'V3')
    p(f'Post hoc, descriptive (not declared): pooled Lift_room V1 95% [{lb[0]:.2f}, {lb[1]:.2f}], V3 [{lb3[0]:.2f}, '
      f'{lb3[1]:.2f}] (same draws).')

    # ---- T8 post hoc by situation
    p('\n== T8. Post hoc, descriptive (not declared): primary records by situation (labels used only to tabulate) ==')
    p(f'{"situation":20s} {"n":>4s} {"top1":>4s} | {"mean S V1/V3":14s} | {"Lift_room V0/V1/V3":22s} | '
      f'{"mean log V1/V3":16s} | mean conc V1 / share > 0.9')
    for sit in sorted({x['sit'] for x in prim}):
        rs = [x for x in prim if x['sit'] == sit]
        A = {V: agg(rs, V) for V in VARS}
        p(f'{sit:20s} {len(rs):4d} {sum(x["top1"] for x in rs):4d} | {A["V1"]["meanS"]:.3f} {A["V3"]["meanS"]:.3f}  | '
          f'{A["V0"]["lift_room"]:6.2f} {A["V1"]["lift_room"]:6.2f} {A["V3"]["lift_room"]:6.2f} | '
          f'{A["V1"]["log"]:7.3f} {A["V3"]["log"]:7.3f} | {A["V1"]["conc"]:.3f} / {A["V1"]["conc90"]:.3f}')

    # ---- T9 criteria
    def criteria(V):
        P_ = {V2_: agg(prim, V2_) for V2_ in VARS}
        a = P_[V]['lift_room'] >= 10
        b = P_[V]['log'] > P_['V3']['log']
        c_cells = {}
        for c in PRIMARY:
            rs = by[c]
            dh = sum(x[V]['hit'][4] for x in rs)
            lh = sum(x['lex_hit'][4] for x in rs)
            c_cells[c] = (dh, lh, dh >= lh - 2, abs(dh - lh) <= 2)
        cc = all(v[2] for v in c_cells.values())
        cc2 = all(v[3] for v in c_cells.values())
        aus = {c: auroc([x[V]['conc'] for x in by[c]], [x['top1'] for x in by[c]]) for c in OLD}
        d = sum(v >= 0.70 for v in aus.values()) >= 2
        k1 = not b
        share90 = P_[V]['conc90']
        k2 = share90 > 0.80
        verdict = 'kill' if (k1 or k2) else ('success' if (a and b and cc and d) else 'neither')
        better_lift = P_[V]['lift_room'] > P_['V3']['lift_room']
        better_au = sum(v >= 0.75 for v in aus.values()) >= 2
        return dict(P=P_, a=a, b=b, c=cc, c2=cc2, c_cells=c_cells, d=d, aus=aus, k1=k1, k2=k2, share90=share90,
                    verdict=verdict, better_lift=better_lift, better_au=better_au)

    res = {}
    for V in ('V1', 'V2'):
        C = criteria(V)
        res[V] = C
        P_ = C['P']
        tag = 'primary' if V == 'V1' else 'secondary'
        p(f'\n== T9{"a" if V == "V1" else "b"}. Criteria for {V} ({tag}; FROZEN.md section 9), n {len(prim)} primary records ==')
        p(f'(a) Lift_room >= 10 pooled: {P_[V]["lift_room"]:.2f} (V0 {P_["V0"]["lift_room"]:.2f}, V3 '
          f'{P_["V3"]["lift_room"]:.2f}) -> {"MET" if C["a"] else "NOT MET"}')
        p('    per cell: ' + '; '.join(f'{c} {agg(by[c], V)["lift_room"]:.2f}' for c in PRIMARY))
        p(f'(b) pooled mean log score {V} {P_[V]["log"]:.4f} > V3 {P_["V3"]["log"]:.4f}: '
          f'{"MET" if C["b"] else "NOT MET"} (interval in T7)')
        p(f'(c) density hit@5 >= lexicographic hit@5 - 2 in every primary cell: {"MET" if C["c"] else "NOT MET"}; '
          f'literal two-sided |diff| <= 2: {"MET" if C["c2"] else "NOT MET"}')
        p('    ' + '; '.join(f'{c} {v[0]} vs {v[1]}' for c, v in C['c_cells'].items()))
        p(f'(d) AUROC(c_{V}) >= 0.70 in >= 2 of 3 old-wording realistic cells: '
          + ', '.join(f'{c} {v:.3f}' for c, v in C['aus'].items()) + f' -> {"MET" if C["d"] else "NOT MET"}')
        p(f'Kill 1 ({V} does not beat V3 on the pooled mean log score): {"FIRES" if C["k1"] else "does not fire"}')
        p(f'Kill 2 (c_{V} > 0.9 in more than 80% of the {len(prim)} primary records): {P_[V]["n_conc90"]} of {len(prim)} '
          f'({100 * C["share90"]:.1f}%) -> {"FIRES" if C["k2"] else "does not fire"}')
        p(f'Better than expected (reported, not a verdict): pooled Lift_room {V} > V3: '
          f'{"yes" if C["better_lift"] else "no"}; AUROC >= 0.75 in >= 2 of 3 old realistic cells: '
          f'{"yes" if C["better_au"] else "no"}')
        p(f'VERDICT ({V}): {C["verdict"].upper()}')
    p(f'\nPrediction in FROZEN.md section 9 (from I14 T5): (a) fails pooled for every variant -> observed: V1 '
      f'{res["V1"]["P"]["V1"]["lift_room"]:.2f}, V2 {res["V2"]["P"]["V2"]["lift_room"]:.2f}, V3 '
      f'{res["V1"]["P"]["V3"]["lift_room"]:.2f}.')
    text = '\n'.join(out)
    with open(f'{HERE}/tables.txt', 'w') as f:
        f.write(text + '\n')
    print(text)
    summ = dict(verdict_V1=res['V1']['verdict'], verdict_V2=res['V2']['verdict'],
                pooled={V: {k: float(v) for k, v in agg(prim, V).items()} for V in VARS},
                criteria={V: {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in res[V].items()
                              if k in ('a', 'b', 'c', 'c2', 'd', 'k1', 'k2', 'share90', 'better_lift', 'better_au')}
                          for V in ('V1', 'V2')},
                aus={V: res[V]['aus'] for V in ('V1', 'V2')}, auroc_all=au,
                boots={k: [list(map(float, x)) for x in v] for k, v in boots.items()},
                c_cells={V: {c: list(map(lambda q: q if isinstance(q, int) else bool(q), v))
                             for c, v in res[V]['c_cells'].items()} for V in ('V1', 'V2')})
    with open(f'{HERE}/summary.json', 'w') as f:
        json.dump(summ, f, indent=1, default=float)
    print(f'\nheld-out scoring {t1 - t0:.1f} s; tables {time.monotonic() - t1:.1f} s')


if __name__ == '__main__':
    main()
