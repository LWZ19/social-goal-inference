"""Verifier: every table, criterion and kill line from my own outputs (out/), then a comparison with the compute
agent's summary.json numbers (read as data, not imported).

    v_summary.py  -> out/summary.txt, out/summary.json
"""
from __future__ import annotations

import collections
import json
import math

import numpy as np

import vlib as V

L = []


def say(s=''):
    print(s, flush=True)
    L.append(s)


def pct(k, n):
    return f'{k}/{n} = {100 * k / n:.1f}%' if n else f'{k}/0'


def load_arms():
    rows = []
    for s in (0, 1, 2):
        rows += [json.loads(x) for x in (V.OUT / f'arms_seed{s}.jsonl').read_text().splitlines() if x]
    return rows


def boot(pairs_by_layout, rng_seed=0, n=2000):
    """Layout-clustered bootstrap of a mean of per-item values: {layout: [values]}."""
    keys = sorted(pairs_by_layout)
    rng = np.random.default_rng(rng_seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(len(keys), size=len(keys))
        xs = [v for i in pick for v in pairs_by_layout[keys[i]]]
        vals.append(np.mean(xs) if xs else np.nan)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def main():
    rows = load_arms()
    by = {(r['seed'], r['scene_id']): r for r in rows}
    bases = [r for r in rows if r['kind'] == 'base']
    var = [r for r in rows if r['kind'] == 'variant']
    man = V.manifest()
    planned = [e for e in man if e['kind'] == 'variant']
    out = {}
    say(f'kept scenes {len(rows)} (bases {len(bases)}, variants {len(var)}); planned variants {len(planned)}')
    assert len(rows) == sum(e['accepted'] for e in man)

    def b_of(r):
        return by[(r['seed'], r['base_scene_id'])]

    e0 = lambda r: r['eps']['e0']  # noqa: E731
    # ---- checks
    same = sum(all(r['same_eps'].values()) for r in rows)
    eq = [r['eps'][k]['equiv_m'] for r in rows for k in r['eps']]
    say(f'both episodes same under every arm: {pct(same, len(rows))}')
    say(f'equivalence arm3(no constraint) vs arm2: max {max(x for x in eq if x is not None):.2e} m; None pairs '
        f'{sum(x is None for x in eq)}; within 1e-9: {sum(x is not None and x <= 1e-9 for x in eq)}/{len(eq)}')
    say(f'model calls: {sum(r["calls"] for r in rows)}')
    say(f'L1 invariance (variant polygons == base polygons): {sum(r["l1_invariant"] for r in var)}/{len(var)}')
    a1v = sum(e0(r)['arm1']['l1'] for r in var)
    a1b = sum(e0(r)['arm1']['l1'] for r in bases)
    say(f'arm 1 L1: variants {pct(a1v, len(var))}, bases {pct(a1b, len(bases))}')
    out['checks'] = {'same': same, 'n': len(rows), 'arm1_var': a1v, 'arm1_base': a1b}
    # ---- curves
    applies = lambda fam, r: (r['situation'] == 'waiting_line') if fam == 'A' else True  # noqa: E731
    levels = {'A': [1, 2, 3, 4, 5, 6], 'B': [1], 'C': [1], 'D': [2, 5], 'E': [1, 2]}
    say('\nfam lvl kept | arm3 L1  L2 | drop3 L1 [CI] L2 | arm2 L1 L2 | arm1 L1 L2')
    table = {}
    for fam, lvls in levels.items():
        bs = [r for r in bases if applies(fam, r)]
        say(f'{fam} 0 {len(bs):3d} | {sum(e0(r)["arm3"]["l1"] for r in bs)}/{len(bs)} '
            f'{sum(e0(r)["arm3"]["l2"] for r in bs)}/{len(bs)} | - | {sum(e0(r)["arm2"]["l1"] for r in bs)} '
            f'{sum(e0(r)["arm2"]["l2"] for r in bs)} | {sum(e0(r)["arm1"]["l1"] for r in bs)} {sum(e0(r)["arm1"]["l2"] for r in bs)}')
        for lv in lvls:
            vs = [r for r in var if r['family'] == fam and r['level'] == lv]
            n = len(vs)
            s3 = sum(e0(r)['arm3']['l1'] for r in vs)
            s3b = sum(e0(r)['arm3']['l2'] for r in vs)
            d1 = [int(e0(b_of(r))['arm3']['l1']) - int(e0(r)['arm3']['l1']) for r in vs]
            d2 = [int(e0(b_of(r))['arm3']['l2']) - int(e0(r)['arm3']['l2']) for r in vs]
            byl = collections.defaultdict(list)
            for r, d in zip(vs, d1):
                byl[(r['seed'], r['row'])].append(100 * d)
            ci = boot(byl) if vs else (math.nan, math.nan)
            table[(fam, lv)] = {'n': n, 'arm3_l1': s3, 'arm3_l2': s3b, 'drop_l1': 100 * np.mean(d1) if n else None,
                                'drop_l2': 100 * np.mean(d2) if n else None,
                                'arm2_l1': sum(e0(r)['arm2']['l1'] for r in vs), 'arm2_l2': sum(e0(r)['arm2']['l2'] for r in vs),
                                'arm1_l1': sum(e0(r)['arm1']['l1'] for r in vs), 'arm1_l2': sum(e0(r)['arm1']['l2'] for r in vs)}
            t = table[(fam, lv)]
            say(f'{fam} {lv} {n:3d} | {s3}/{n} {s3b}/{n} | {t["drop_l1"]:.1f} [{ci[0]:.1f}, {ci[1]:.1f}] {t["drop_l2"]:.1f} | '
                f'{t["arm2_l1"]} {t["arm2_l2"]} | {t["arm1_l1"]} {t["arm1_l2"]}')
    out['table'] = {f'{k[0]}{k[1]}': v for k, v in table.items()}
    # by situation
    say('\nby situation, arm 3 L1 success (n): ')
    for fam, lvls in levels.items():
        for lv in lvls:
            vs = [r for r in var if r['family'] == fam and r['level'] == lv]
            parts = []
            for sit in ('waiting_line', 'conversation_group', 'audience'):
                q = [r for r in vs if r['situation'] == sit]
                parts.append(f'{sit[:5]} {sum(e0(r)["arm3"]["l1"] for r in q)}/{len(q)}')
            say(f'  {fam}{lv}: ' + '; '.join(parts))
    say('arm 2 L1 bases: ' + pct(sum(e0(r)['arm2']['l1'] for r in bases), len(bases)))

    # ---- detectors
    def det_rows(rset):
        res = []
        for r in rset:
            b = b_of(r)
            d, bd = e0(r)['arm3']['det'], e0(b)['arm3']['det']
            fires = {k: bool(d[k] and not bd[k]) for k in ('i', 'ii', 'iii', 'iv', 'v')}
            rel = {'i': d['ca_used'] and not bd['i'],
                   'ii': bd['F'] > 0 and not bd['ii'],
                   'iii': d['iii_pre'] and bd['off_deg'] is not None and bd['off_deg'] < 110.0,
                   'iv': r['family'] in 'BCDE' and d['iv_pre'] and bool(bd['not_ahead_of']) and not bd['iv'],
                   'v': not bd['v']}
            res.append((r, fires, rel))
        return res

    D = det_rows(var)

    def rates(Dset):
        o = {}
        for k in ('i', 'ii', 'iii', 'iv', 'v'):
            rel = [x for x in Dset if x[2][k]]
            fi = [x for x in rel if x[1][k]]
            o[k] = (len(fi), len(rel))
        return o

    rt = rates(D)
    say('\n(b) detectors on arm 3 (fires / relevant):')
    for k, (f, n) in rt.items():
        fam_parts = []
        for fam in 'ABCDE':
            rel = [x for x in D if x[2][k] and x[0]['family'] == fam]
            fam_parts.append(f'{fam} {sum(x[1][k] for x in rel)}/{len(rel)}')
        say(f'  ({k}) {pct(f, n)}  ' + ', '.join(fam_parts))
    # any firing outside "relevant"?
    for k in ('i', 'ii', 'iii', 'iv', 'v'):
        extra = [x[0]['scene_id'] for x in D if x[1][k] and not x[2][k]]
        if extra:
            say(f'  ({k}) fires in {len(extra)} non-relevant variants: {extra[:5]}')
    b_met = [k for k in ('i', 'ii', 'iii', 'iv') if rt[k][1] and rt[k][0] / rt[k][1] >= 0.10]
    say(f'  (b) failures >= 10%: {b_met} -> {"MET" if len(b_met) >= 2 else "NOT MET"}')
    offs = [e0(r)['arm3']['det']['off_deg'] for r, f, rel in D if rel['iii']]
    say(f'  (iii) max off-facing in relevant variants {max(offs):.1f} deg (n {len(offs)})')
    Fmin = min(e0(r)['arm3']['det']['F'] for r in var)
    say(f'  (ii) smallest F in a variant {Fmin}; variants with F == 0: {sum(e0(r)["arm3"]["det"]["F"] == 0 for r in var)}')
    iv_rel = [(r['scene_id'], e0(r)['arm3']['det']['not_ahead_of'], e0(r)['arm3']['det']['not_ahead_reason'])
              for r, f, rel in D if rel['iv']]
    say(f'  (iv) relevant {len(iv_rel)}; rule applied (not_ahead_of non-empty) in {sum(bool(x[1]) for x in iv_rel)}')
    # arm 2 descriptive
    a2 = {k: 0 for k in ('ii', 'iii', 'v')}
    for r in var:
        d, bd = e0(r)['arm2']['det'], e0(b_of(r))['arm2']['det']
        for k in a2:
            a2[k] += bool(d[k] and not bd[k])
    say(f'  arm 2 fires over all variants: {a2}')
    out['b'] = {k: list(v) for k, v in rt.items()}
    # named causes of arm-3 L1 failures
    say('\narm-3 L1 failures and named causes:')
    cause_tab = {}
    for fam, lvls in levels.items():
        for lv in lvls:
            xs = [x for x in D if x[0]['family'] == fam and x[0]['level'] == lv and not e0(x[0])['arm3']['l1']]
            named = [x for x in xs if any(x[1].values())]
            causes = collections.Counter(k for x in xs for k, v in x[1].items() if v)
            cause_tab[(fam, lv)] = (len(named), len(xs))
            if xs:
                say(f'  {fam}{lv}: failures {len(xs)}, named {len(named)}, causes {dict(causes)}')
                for x in xs:
                    if not any(x[1].values()):
                        d = e0(x[0])['arm3']['det']
                        bd = e0(b_of(x[0]))['arm3']['det']
                        say(f'     unnamed {x[0]["scene_id"]}: in_region {d["in_region"]} heading_ok {d["heading_ok"]} '
                            f'standable {d["standable"]}/{x[0]["n_l1_cells"]} (base {bd["standable"]}) F {d["F"]}')

    # ---- (a)
    def crit_a(vset, Dset, fams, lv_map):
        drops = {}
        for fam in fams:
            vs = [r for r in vset if r['family'] == fam]
            if vs:
                drops[fam] = (100 * np.mean([int(e0(b_of(r))['arm3']['l1']) - int(e0(r)['arm3']['l1']) for r in vs]), len(vs))
        ge = [f for f, (d, n) in drops.items() if d >= 15]
        mono = {}
        for fam in ('A', 'D'):
            if fam not in fams:
                continue
            lv = lv_map[fam]
            per = []
            for l in lv:
                vs = [r for r in vset if r['family'] == fam and r['level'] == l]
                per.append(100 * np.mean([int(e0(b_of(r))['arm3']['l1']) - int(e0(r)['arm3']['l1']) for r in vs]) if vs else None)
            ok = per and per[0] is not None and per[0] >= 0 and all(
                per[i + 1] is not None and per[i] is not None and per[i + 1] >= per[i] - 1e-12 for i in range(len(per) - 1))
            top = lv[-1]
            xs = [x for x in Dset if x[0]['family'] == fam and x[0]['level'] == top and not e0(x[0])['arm3']['l1']]
            named = sum(any(x[1].values()) for x in xs)
            mono[fam] = {'per': per, 'nondecr': bool(ok), 'top': per[-1], 'named': (named, len(xs)),
                         'met': bool(ok and per[-1] is not None and per[-1] >= 10 and xs and named / len(xs) >= 0.5)}
        met = len(ge) >= 2 or any(m['met'] for m in mono.values())
        return drops, ge, mono, met

    say('\n(a):')
    full = crit_a(var, D, 'ABCDE', {'A': [1, 2, 3, 4, 5, 6], 'D': [2, 5]})
    for name, res in (('all', full),
                      ('without A5-A6', crit_a([r for r in var if not (r['family'] == 'A' and r['level'] >= 5)],
                                               [x for x in D if not (x[0]['family'] == 'A' and x[0]['level'] >= 5)],
                                               'ABCDE', {'A': [1, 2, 3, 4], 'D': [2, 5]})),
                      ('without A', crit_a([r for r in var if r['family'] != 'A'], [x for x in D if x[0]['family'] != 'A'],
                                           'BCDE', {'D': [2, 5]}))):
        drops, ge, mono, met = res
        say(f'  [{name}] drops ' + ', '.join(f'{f} {d:.1f} (n {n})' for f, (d, n) in drops.items()) +
            f'; >=15: {ge}; monotone {json.dumps(mono)} -> {"MET" if met else "NOT MET"}')
    for name, keep in (('without A5-A6', lambda x: not (x[0]['family'] == 'A' and x[0]['level'] >= 5)),
                       ('without A', lambda x: x[0]['family'] != 'A')):
        rr = rates([x for x in D if keep(x)])
        say(f'  (b) [{name}] ' + ', '.join(f'({k}) {pct(*v)}' for k, v in rr.items() if k != 'v'))
    # family drop CIs
    for fam in 'ABCDE':
        byl = collections.defaultdict(list)
        for r in var:
            if r['family'] == fam:
                byl[(r['seed'], r['row'])].append(100 * (int(e0(b_of(r))['arm3']['l1']) - int(e0(r)['arm3']['l1'])))
        lo, hi = boot(byl)
        l2d = 100 * np.mean([int(e0(b_of(r))['arm3']['l2']) - int(e0(r)['arm3']['l2']) for r in var if r['family'] == fam])
        say(f'  family {fam}: drop CI [{lo:.1f}, {hi:.1f}]; L2 drop {l2d:.1f}')
    out['a'] = {'drops': {f: v[0] for f, v in full[0].items()}, 'met': full[3]}

    # ---- (c)
    say('\n(c):')
    for arm in ('arm3', 'arm2', 'arm1'):
        dis = [r for r in var if e0(r)[arm]['l1'] and not e0(r)[arm]['l2']]
        disb = [r for r in bases if e0(r)[arm]['l1'] and not e0(r)[arm]['l2']]
        say(f'  {arm}: disagreement variants {pct(len(dis), len(var))}, bases {pct(len(disb), len(bases))}')
        if arm == 'arm3':
            for r in dis:
                say(f'     {r["scene_id"]}: {e0(r)["arm3"]["l2_reasons"]}')
            out['c_dis'] = len(dis)
    l2e = [r for r in var if r['n_l1_cells'] > 0 and r['n_l2_cells'] == 0]
    l2eb = [r for r in bases if r['n_l1_cells'] > 0 and r['n_l2_cells'] == 0]
    say(f'  L2-empty: variants {pct(len(l2e), len(var))}, bases {pct(len(l2eb), len(bases))}; no L1 cells: '
        f'{sum(r["n_l1_cells"] == 0 for r in rows)}')
    pm_false = sorted({(r['seed'], r['row']) for r in rows if not e0(r)['arm3']['matter']})
    say(f'  people do not matter (arm 3 standard goal) in {sum(not e0(r)["arm3"]["matter"] for r in rows)} scenes, '
        f'layouts {pm_false}')
    out['l2_empty'] = len(l2e)
    # ---- (d)
    acc = sum(e['accepted'] for e in planned)
    first = sum(e['accepted'] and e['attempts'] == 1 for e in planned)
    say(f'\n(d): pass {pct(acc, len(planned))}; first attempt {pct(first, len(planned))}; rejected reasons '
        f'{dict(collections.Counter(e["rejected_reason"] for e in planned if not e["accepted"]))}; full oracle L1 '
        f'variants {pct(a1v, len(var))}, bases {pct(a1b, len(bases))}')
    out['d'] = {'acc': acc, 'planned': len(planned)}
    # ---- K1, K3
    low = min(table[k]['arm3_l1'] / table[k]['n'] for k in table if table[k]['n'])
    say(f'K1 first half: lowest arm-3 success over (family, level) {100 * low:.1f}% -> {"holds" if low >= 0.95 else "fails"}')
    say(f'K3: rejected {pct(len(planned) - acc, len(planned))}')
    (V.OUT / 'summary_oracle.txt').write_text('\n'.join(L) + '\n')
    (V.OUT / 'summary_oracle.json').write_text(json.dumps(out, default=float, indent=1))


if __name__ == '__main__':
    main()
