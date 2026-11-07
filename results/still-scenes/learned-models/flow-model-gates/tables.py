"""C5 gates: tables (FROZEN.md sections 1-2). Reads gate1.json, cells_check.txt, cache/index.json and work/*.json;
writes tables.txt, inner_cv.txt, rows_gate2.jsonl and folds.json. No model, no network, nothing fitted here."""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, 'results/still-scenes/learned-models/flow-model-gates')
from common import FEATURE_NAMES, FEATURE_SETS, FROZEN_SHA, HERE, M_GRID, PEOPLE_SITS, SEEDS  # noqa: E402

WORK = HERE / 'work'
METHODS = ('ratio', 'plain', 'ratio1d', 'band', 'uniform')
SITS = ('no_one', 'waiting_line', 'conversation_group', 'audience')
PASS_NATS = 0.10


def boot(rows, value, draws=2000):
    """Seed-cluster bootstrap 95% interval of the episode mean of value(row) (default_rng(0) per call)."""
    by = {k: [value(r) for r in rows if r['seed'] == k] for k in SEEDS}
    by = {k: v for k, v in by.items() if v}
    keys = sorted(by)
    rng = np.random.default_rng(0)
    out = []
    for _ in range(draws):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        vals = np.concatenate([by[keys[i]] for i in pick])
        out.append(vals.mean())
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975)), sum(np.mean(v) > 0 for v in by.values()), len(keys)


def mean(rows, f):
    return float(np.mean([f(r) for r in rows])) if rows else float('nan')


def block(rows, title, L, with_boot=True):
    n_ep = len(rows)
    n_sc = len({r['key'] for r in rows})
    L.append(f'{title}: n = {n_ep} episodes, {n_sc} scenes')
    L.append(f'  {"method":9s} {"S (nats/region cell)":>21s} {"mass M":>8s} {"lift":>7s}')
    for m in METHODS:
        L.append(f'  {m:9s} {mean(rows, lambda r: r[m]["S"]):21.3f} {mean(rows, lambda r: r[m]["M"]):8.3f} '
                 f'{mean(rows, lambda r: r[m]["lift"]):7.2f}')
    d = mean(rows, lambda r: r['ratio']['S'] - r['band']['S'])
    line = f'  Delta = S_ratio - S_band = {d:+.3f}'
    if with_boot:
        lo, hi, pos, nk = boot(rows, lambda r: r['ratio']['S'] - r['band']['S'])
        line += f'  (seed-cluster bootstrap 95% {lo:+.3f} to {hi:+.3f}; seeds with Delta > 0: {pos} of {nk})'
    L.append(line)
    for a, b in (('plain', 'band'), ('ratio1d', 'band'), ('ratio', 'ratio1d'), ('ratio', 'uniform'), ('band', 'uniform')):
        L.append(f'  S_{a} - S_{b} = {mean(rows, lambda r: r[a]["S"] - r[b]["S"]):+.3f}')
    return d


def main():
    assert hashlib.sha256((HERE / 'FROZEN.md').read_bytes()).hexdigest() == FROZEN_SHA
    g1 = json.loads((HERE / 'gate1.json').read_text())
    inner = {k: json.loads((WORK / f'inner_{k}.json').read_text()) for k in ('primary', 'ceiling', 'dev')}
    outer = {k: json.loads((WORK / f'outer_{k}.json').read_text()) for k in ('primary', 'ceiling', 'dev')}
    index = json.loads((HERE / 'cache' / 'index.json').read_text())

    # ------------------------------------------------------------------------- rows and folds files
    with open(HERE / 'rows_gate2.jsonl', 'w') as fh:
        for k in ('primary', 'ceiling', 'dev'):
            for r in outer[k]['rows']:
                fh.write(json.dumps({'fold_type': k, **r}) + '\n')
    folds = {}
    for k in ('primary', 'ceiling', 'dev'):
        for name, inn in inner[k].items():
            info = outer[k]['info'][name]
            ch = {}
            for meth, c in inn['chosen'].items():
                cols = (0,) if c['fs'] == 'F1D' else FEATURE_SETS[c['fs']]
                ch[meth] = {**c, 'features': [FEATURE_NAMES[i] for i in cols], 'h': info['h'][meth],
                            'n_r': info['n_r'][c['fs']]}
            folds[name] = {'fold_type': k, 'n_train_scenes': inn['n_train_scenes'],
                           'n_train_episodes': inn['n_train_episodes'], 'inner_seeds': inn['seeds'],
                           'chosen': ch, 'band': info['band'], 'standardise': info['standardise'],
                           'n_dropped_train_scenes': info['n_dropped_train_scenes'], 'inner_means': inn['means']}
    (HERE / 'folds.json').write_text(json.dumps(folds, indent=1))

    # ------------------------------------------------------------------------- inner_cv.txt
    I = ['C5 gate 2: inner CV (leave one seed out inside each fold\'s training data). Mean over held-out seeds of the '
         'per-seed mean S (nats per region cell; estimated: <= 200 region cells per episode, sum_floor r from <= 600 '
         'floor cells per scene). * = chosen (ties: earlier feature set, then larger m).', '']
    for name, f in folds.items():
        I.append(f'{name} ({f["fold_type"]}; {f["n_train_scenes"]} training scenes, {f["n_train_episodes"]} episodes)')
        for meth in ('ratio', 'plain', 'ratio1d'):
            means = f['inner_means'][meth]
            c = f['chosen'][meth]
            fss = ('F1D',) if meth == 'ratio1d' else tuple(FEATURE_SETS)
            for fs in fss:
                cells = []
                for m in M_GRID:
                    v = means[f'{fs}|{m}']
                    star = '*' if (fs == c['fs'] and m == c['m']) else ' '
                    cells.append(f'm={m:<3}: {v:8.4f}{star}')
                I.append(f'  {meth:8s} {fs:4s} ' + '  '.join(cells))
        I.append('')
    (HERE / 'inner_cv.txt').write_text('\n'.join(I) + '\n')

    # ------------------------------------------------------------------------- tables.txt
    L = ['C5 (I52) zero-call gates 1 and 2: tables. Protocol: FROZEN.md (sha256 ' + FROZEN_SHA + ').',
         'Every gate-2 number measures agreement with the project\'s own reference rules (src/sgi/gt/regions.py), '
         'not social correctness.', '']
    L.append('== T1. Gate 1: configuration count (C2 Step 0) ==')
    L.append(f'PRIMARY (decides) = {g1["PRIMARY"]["count"]} distinct (recipe, amount, kind, bystanders, second group) over '
             f'{len(g1["rows_kept"])} development rows (both splits, both passes).')
    L.append(f'LOWER = {g1["LOWER"]["count"]} distinct (situation, structure size, bystanders, second group).')
    L.append(f'Gate 1 (>= 30 met; 25-29 borderline, not met): {g1["gate1_verdict"].upper()}.  C2 Source A kill: '
             f'{g1["c2_source_a_kill"].upper()}.')
    emp = g1['EMPIRICAL']
    mem = [e['members']['b_matching_or_mirror']['median_m'] for e in emp if e['members']['b_matching_or_mirror']]
    mem_max = [e['members']['b_matching_or_mirror']['max_m'] for e in emp if e['members']['b_matching_or_mirror']]
    L.append(f'EMPIRICAL (descriptive; seeds 100-109, 45 pairs per row): structure members, best matching with mirror: '
             f'median per row {min(mem):.2f}-{max(mem):.2f} m, maximum per row {min(mem_max):.2f}-{max(mem_max):.2f} m '
             f'over {len(mem)} rows with members. Full table: gate1.txt.')
    L.append('')
    L.append('== T2. Gate 2 data and cells ==')
    for line in (HERE / 'cells_check.txt').read_text().splitlines()[2:]:
        if line.startswith('time'):
            continue
        L.append(line)
    L.append('')

    L.append('== T3. Configurations chosen by inner CV (training data only) ==')
    L.append(f'{"fold":24s} {"ratio KDE":28s} {"plain KDE":28s} {"1-D ratio":18s}')
    for name, f in folds.items():
        cells = []
        for meth in ('ratio', 'plain', 'ratio1d'):
            c = f['chosen'][meth]
            cells.append(f'{c["fs"]} m={c["m"]} h={c["h"]:.3f}{" EDGE" if c["edge"] else ""}')
        L.append(f'{name:24s} {cells[0]:28s} {cells[1]:28s} {cells[2]:18s}')
    L.append('EDGE = optimum at the end of the m grid {0.5, 1, 2, 4}: reported, not extended.')
    L.append('')

    L.append('== T4. PRIMARY: leave one people situation out (decides gate 2) ==')
    deltas = {}
    prim = outer['primary']['rows']
    for left in PEOPLE_SITS:
        rows = [r for r in prim if r['fold'] == f'LOSO_{left}']
        deltas[left] = block(rows, f'Fold: test {left} (train no_one + the other two)', L)
        L.append(f'  meets Delta >= {PASS_NATS}: {deltas[left] >= PASS_NATS}')
        L.append('')
    # post hoc, descriptive (added after the primary scores were seen): how much of each region the band covers,
    # and the spread of the per-episode Delta
    L.append('Post hoc (added after the primary scores were seen; descriptive, decides nothing):')
    for left in PEOPLE_SITS:
        name = f'LOSO_{left}'
        rows = [r for r in prim if r['fold'] == name]
        b = outer['primary']['info'][name]['band']
        cover = []
        for r in rows:
            z = np.load(HERE / 'cache' / next(x['file'] for x in index if x['key'] == r['key']))
            dt = z[f'F__{r["target"]}'][z[f'reg__{r["eid"]}'], 0]
            cover.append(float(((dt >= b['q05']) & (dt <= b['q95'])).mean()))
        dl = np.array([r['delta'] for r in rows])
        L.append(f'  {left:20s} band [{b["q05"]:.2f}, {b["q95"]:.2f}] m covers {np.mean(cover):.3f} of the region cells '
                 f'(mean over episodes); per-episode Delta median {np.median(dl):+.3f}, positive in {int((dl > 0).sum())} '
                 f'of {len(dl)} episodes; S_ratio - S_uniform {mean(rows, lambda r: r["ratio"]["S"] - r["uniform"]["S"]):+.3f}')
    L.append('')
    n_met = sum(d >= PASS_NATS for d in deltas.values())
    g2 = 'MET' if n_met >= 2 else 'NOT MET'
    L.append(f'Gate 2 (Delta >= {PASS_NATS} nats per region cell in at least 2 of 3 folds, ratio KDE): {n_met} of 3 -> {g2}')
    L.append('')

    L.append('== T5. In-distribution ceiling: leave one seed out over all four situations (secondary) ==')
    ceil = outer['ceiling']['rows']
    cdel = {}
    for sit in SITS:
        rows = [r for r in ceil if r['situation'] == sit]
        cdel[sit] = block(rows, f'Situation {sit} (pooled over the 10 seed folds)', L)
        L.append('')
    block(ceil, 'All situations pooled', L)
    L.append('')
    L.append('Transfer gap (ceiling minus primary fold, same test episodes):')
    for left in PEOPLE_SITS:
        pr = [r for r in prim if r['fold'] == f'LOSO_{left}']
        ce = [r for r in ceil if r['situation'] == left]
        L.append(f'  {left:20s} S_ratio {mean(ce, lambda r: r["ratio"]["S"]):8.3f} - {mean(pr, lambda r: r["ratio"]["S"]):8.3f}'
                 f' = {mean(ce, lambda r: r["ratio"]["S"]) - mean(pr, lambda r: r["ratio"]["S"]):+.3f};  Delta '
                 f'{cdel[left]:+.3f} - {deltas[left]:+.3f} = {cdel[left] - deltas[left]:+.3f}  (n = {len(pr)} episodes)')
    L.append('')

    L.append('== T6. Development transfer: train on all 260 fresh scenes, test on the 68 development scenes (secondary) ==')
    dev = outer['dev']['rows']
    for dom in ('realistic', 'placeholder'):
        for sit in SITS:
            rows = [r for r in dev if r['domain'] == dom and r['situation'] == sit]
            if rows:
                block(rows, f'{dom} / {sit}', L, with_boot=False)
                L.append('')
        rows = [r for r in dev if r['domain'] == dom]
        block(rows, f'{dom} / all', L, with_boot=False)
        L.append('')
    L.append('(No bootstrap for the development scenes: they have no fresh-seed clusters, and the note fixes the '
             'bootstrap by seed.)')
    L.append('')
    L.append('Post hoc (added after the development scores were seen; descriptive, decides nothing): by source. The 42 '
             'realistic scenes are three arms (plain, real, staffed) of the 14 layouts of placeholder/eval (ph_first14); '
             'ph_devnew = the 12 scenes of keep/dev_new seed0-2. The plain arm scores the same as ph_first14 here.')
    L.append(f'  {"source":11s} {"situation":20s} {"n ep":>4s} {"S_ratio":>8s} {"S_band":>8s} {"Delta":>7s} {"lift_ratio":>10s}')
    groups = {}
    for r in dev:
        if r['domain'] == 'realistic':
            src = r['key'].split('/')[1]
        else:
            src = 'ph_first14' if '/placeholder/eval/' in r['key'] else 'ph_devnew'
        groups.setdefault((src, r['situation']), []).append(r)
    for (src, sit), rows in sorted(groups.items()):
        L.append(f'  {src:11s} {sit:20s} {len(rows):4d} {mean(rows, lambda r: r["ratio"]["S"]):8.3f} '
                 f'{mean(rows, lambda r: r["band"]["S"]):8.3f} {mean(rows, lambda r: r["delta"]):+7.3f} '
                 f'{mean(rows, lambda r: r["ratio"]["lift"]):10.2f}')
    n_people_noone = {}
    for x in index:
        if x['kind'] == 'dev' and x['episodes'][0]['situation'] == 'no_one':
            src = x['key'].split('/')[1] if x['domain'] == 'realistic' else 'placeholder'
            z = np.load(HERE / 'cache' / x['file'])
            n_people_noone.setdefault(src, []).append(int(z[f'F__{x["episodes"][0]["target"]}'][:, 5].max() > 0))
    L.append('  no_one scenes with at least one person within 1.5 m of some floor cell (f6 > 0), by source: '
             + ', '.join(f'{k} {sum(v)} of {len(v)}' for k, v in sorted(n_people_noone.items())))
    L.append('')
    L.append('Definitions. S = mean over an episode\'s region cells of log q (nats per region cell), then the mean over '
             'episodes. M = sum of q over the region cells; lift = M / (n_region / N_floor); both means of per-episode '
             'values. q mixes 1% uniform into every density method; band = 0.99 uniform on floor cells with d_t in '
             '[q05, q95] of the training region cells; uniform = 1 / N_floor. e0 and e1 of a scene share one region set '
             'and one q, so they carry identical values; n is given as episodes and scenes.')
    (HERE / 'tables.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    summary = {'gate1': {'primary': g1['PRIMARY']['count'], 'lower': g1['LOWER']['count'],
                         'verdict': g1['gate1_verdict'], 'kill': g1['c2_source_a_kill']},
               'gate2': {'deltas': deltas, 'n_met': n_met, 'verdict': g2}}
    (WORK / 'summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
