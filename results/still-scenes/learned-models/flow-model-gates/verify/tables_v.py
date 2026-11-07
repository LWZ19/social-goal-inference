"""Independent verifier, C5 gate 2: fold values, Delta, gate verdict, and a per-episode comparison with the compute
agent's rows_gate2.jsonl (read only to compare). Writes verify/tables_v.txt and verify/summary_v.json."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import numpy as np

V = Path('results/still-scenes/learned-models/flow-model-gates/verify')
SHA = '8b7d3ae294ad81a810c70c23706247674bbfed22cc0247445e91f6615426419c'
assert hashlib.sha256((V.parent / 'FROZEN.md').read_bytes()).hexdigest() == SHA
METH = ('ratio', 'plain', 'ratio1d', 'band', 'bandu', 'uniform')
SITS = ('no_one', 'waiting_line', 'conversation_group', 'audience')


def boot(rows, f, draws=2000):
    by = collections.defaultdict(list)
    for r in rows:
        by[r['seed']].append(f(r))
    keys = sorted(by)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(draws):
        pick = rng.integers(0, len(keys), len(keys))
        vals.append(np.mean(np.concatenate([by[keys[i]] for i in pick])))
    pos = sum(np.mean(by[k]) > 0 for k in keys)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)), int(pos), len(keys)


def block(rows, title, L, bootstrap=True):
    out = {'n_episodes': len(rows), 'n_scenes': len({(r['scene'], r.get('folder'), r['seed']) for r in rows})}
    L.append(f"{title}: n = {out['n_episodes']} episodes, {out['n_scenes']} scenes")
    for m in METH:
        S = float(np.mean([r[f'S_{m}'] for r in rows]))
        M = float(np.mean([r[f'M_{m}'] for r in rows]))
        lift = float(np.mean([r[f'lift_{m}'] for r in rows]))
        out[m] = {'S': S, 'M': M, 'lift': lift}
        L.append(f'  {m:8s} S {S:9.3f}  M {M:7.3f}  lift {lift:7.2f}')
    d = out['ratio']['S'] - out['band']['S']
    du = out['ratio']['S'] - out['bandu']['S']
    out['delta'] = d
    out['delta_bandu'] = du
    line = f'  Delta = S_ratio - S_band = {d:+.3f} (with the each-cell-once band: {du:+.3f})'
    if bootstrap and all(r['seed'] is not None for r in rows):
        lo, hi, pos, nk = boot(rows, lambda r: r['S_ratio'] - r['S_band'])
        out['boot'] = (lo, hi, pos, nk)
        line += f'  boot95 {lo:+.3f} to {hi:+.3f}; seeds Delta>0 {pos}/{nk}'
    L.append(line)
    for a, b in (('plain', 'band'), ('ratio1d', 'band'), ('ratio', 'uniform'), ('band', 'uniform')):
        L.append(f"  S_{a} - S_{b} = {out[a]['S'] - out[b]['S']:+.3f}")
    out['band_cover'] = float(np.mean([r['band_cover'] for r in rows]))
    out['delta_pos_episodes'] = int(sum(r['S_ratio'] - r['S_band'] > 0 for r in rows))
    out['delta_median'] = float(np.median([r['S_ratio'] - r['S_band'] for r in rows]))
    L.append(f"  band covers {out['band_cover']:.3f} of region cells; per-episode Delta median {out['delta_median']:+.3f},"
             f" positive in {out['delta_pos_episodes']} of {len(rows)}")
    return out


def their_rows():
    rows = [json.loads(line) for line in (V.parent / 'rows_gate2.jsonl').read_text().splitlines()]
    out = {}
    for r in rows:
        key = (r['fold'], r['key'], r['eid'])
        out[key] = r
    return out


def my_key(r):
    if r['seed'] is not None:
        # their key: seed{k}/{split}/{scene_id}; split = prefix of scene_id
        split = r['scene'].split('_')[0]
        return (r['fold'], f"seed{r['seed']}/{split}/{r['scene']}", r['episode'])
    parts = Path(r['folder']).parts[-3:]
    return (r['fold'], f"{r['domain']}/" + '/'.join(parts), r['episode'])


def main():
    L = ['C5 gate 2: independent verifier tables (verify/kde_v.py). Agreement with the project reference rules only.']
    summary = {}
    outs = {}
    for p in sorted(V.glob('out_*.json')):
        if p.stem.startswith('out_draw'):
            continue
        outs[p.stem] = json.loads(p.read_text())
    folds, rows = {}, []
    for o in outs.values():
        for f in o['folds']:
            folds[f['tag']] = f
        rows.extend(o['rows'])
    L.append(f'folds available: {sorted(folds)}; rows {len(rows)}')
    # ---------------- inner choices
    L.append('\n== Inner-CV choices (verifier) ==')
    theirs_f = json.loads((V.parent / 'folds.json').read_text())
    for tag in sorted(folds):
        f = folds[tag]
        t = theirs_f.get(tag, {})
        tc = {k: (v['fs'], v['m']) for k, v in t.get('chosen', {}).items()}
        mc = {k: tuple(v) for k, v in f['choice'].items()}
        names = [tuple(n) for n in f['inner_names']]
        ir = dict(zip(names, f['inner_ratio']))
        top = sorted(((v, k) for k, v in ir.items() if k[0] != 'F1D'), reverse=True)[:2]
        L.append(f"{tag:26s} mine {mc}  theirs {tc}  same {mc == tc}; n_r mine {f['n_r']} theirs "
                 f"{t.get('chosen', {}).get('ratio', {}).get('n_r')}; h {f['h_ratio']:.4f}; band mine "
                 f"[{f['band'][0]:.4f}, {f['band'][1]:.4f}] (each-cell-once [{f['band_sets'][0]:.4f}, {f['band_sets'][1]:.4f}])"
                 f" theirs [{t.get('band', {}).get('q05', float('nan')):.4f}, {t.get('band', {}).get('q95', float('nan')):.4f}];"
                 f" inner top-2 ratio {[(k, round(v, 4)) for v, k in top]}")
        summary.setdefault('inner', {})[tag] = {'mine': mc, 'theirs': tc, 'same': mc == tc, 'n_r': f['n_r'],
                                               'h_ratio': f['h_ratio'], 'band': f['band'], 'band_sets': f['band_sets']}
    # ---------------- primary
    L.append('\n== PRIMARY: leave one people situation out ==')
    passes = 0
    prim = {}
    for sit in ('waiting_line', 'conversation_group', 'audience'):
        rr = [r for r in rows if r['fold'] == f'LOSO_{sit}']
        if not rr:
            continue
        o = block(rr, f'LOSO_{sit}', L)
        o['meets'] = o['delta'] >= 0.10
        passes += o['meets']
        prim[sit] = o
    summary['primary'] = prim
    if len(prim) == 3:
        summary['gate2'] = {'folds_passing': passes, 'met': passes >= 2}
        L.append(f'Gate 2: {passes} of 3 folds with Delta >= 0.10 -> {"MET" if passes >= 2 else "NOT MET"}')
    # ---------------- ceiling
    cr = [r for r in rows if r['fold'].startswith('LOSeed_')]
    if cr:
        L.append(f"\n== CEILING (seed folds available: {sorted({r['fold'] for r in cr})}) ==")
        summary['ceiling'] = {}
        for sit in SITS:
            rr = [r for r in cr if r['situation'] == sit]
            if rr:
                summary['ceiling'][sit] = block(rr, f'ceiling {sit}', L)
        summary['ceiling']['all'] = block(cr, 'ceiling all', L)
        for sit in ('waiting_line', 'conversation_group', 'audience'):
            if sit in prim and sit in summary['ceiling']:
                g = summary['ceiling'][sit]['ratio']['S'] - prim[sit]['ratio']['S']
                L.append(f'  transfer gap S_ratio {sit}: {g:+.3f}')
    # ---------------- dev
    dr = [r for r in rows if r['fold'] == 'DEV']
    if dr:
        L.append('\n== DEV transfer ==')
        summary['dev'] = {}
        for dom in ('realistic', 'placeholder'):
            for sit in SITS:
                rr = [r for r in dr if r['domain'] == dom and r['situation'] == sit]
                if rr:
                    summary['dev'][f'{dom}/{sit}'] = block(rr, f'{dom}/{sit}', L, bootstrap=False)
            rr = [r for r in dr if r['domain'] == dom]
            summary['dev'][f'{dom}/all'] = block(rr, f'{dom}/all', L, bootstrap=False)
        # post hoc by source
        L.append('  post hoc, by source (Delta, n episodes):')
        def src(r):
            f = r['folder']
            if 'realistic_final' in f:
                return f.split('realistic_final/')[1].split('/')[0]
            return 'ph_devnew' if 'dev_new' in f else 'ph_first14'
        for key in sorted({(src(r), r['situation']) for r in dr}):
            rr = [r for r in dr if (src(r), r['situation']) == key]
            d = np.mean([r['S_ratio'] - r['S_band'] for r in rr])
            L.append(f'    {key[0]:11s} {key[1]:20s} n {len(rr):3d}  Delta {d:+.3f}')
            summary.setdefault('dev_by_source', {})[f'{key[0]}/{key[1]}'] = (float(d), len(rr))
    # ---------------- per-episode comparison with the compute agent
    th = their_rows()
    diffs = collections.defaultdict(list)
    missing = 0
    for r in rows:
        k = my_key(r)
        t = th.get(k)
        if t is None:
            missing += 1
            continue
        for m in ('ratio', 'plain', 'ratio1d', 'band', 'uniform'):
            diffs[(r['fold'][:4], m)].append(r[f'S_{m}'] - t[m]['S'])
        diffs[(r['fold'][:4], 'bandu')].append(r['S_bandu'] - t['band']['S'])
        diffs[(r['fold'][:4], 'n_region')].append(r['n_region'] - t['n_region'])
        diffs[(r['fold'][:4], 'N')].append(r['N'] - t['n_floor'])
    L.append(f'\n== Per-episode comparison with the compute agent (mine - theirs, S in nats/region cell); unmatched {missing} ==')
    comp = {}
    for k in sorted(diffs):
        a = np.array(diffs[k])
        comp[f'{k[0]}|{k[1]}'] = {'n': int(a.size), 'mean': float(a.mean()), 'maxabs': float(np.abs(a).max())}
        L.append(f'  {k[0]:5s} {k[1]:9s} n {a.size:4d}  mean diff {a.mean():+.4f}  max |diff| {np.abs(a).max():.4f}')
    summary['per_episode_vs_compute'] = comp
    summary['unmatched'] = missing
    (V / 'tables_v.txt').write_text('\n'.join(L) + '\n')
    (V / 'summary_v.json').write_text(json.dumps(summary, indent=1, default=str))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
