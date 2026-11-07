"""I06 repair probe: tables and the verdict (FROZEN.md sections 3-6). Reads only this probe's rows and R0 files and
I06's rows_oracle.jsonl (relevant['i']). Writes tables.txt and summary.json. No solve is run here.

    tables.py
"""
from __future__ import annotations

import collections
import json
import statistics
import sys

import common as C

C.start()

RULES = ('R0', 'RF90', 'RF60', 'RN')
REP = ('RF90', 'RF60', 'RN')
LINE = 'waiting_line'
GROUPISH = ('conversation_group', 'audience')
J_CELLS = {('A', 1), ('A', 2), ('A', 3), ('B', 1), ('C', 1), ('D', 2)}
MODELS = ('qwen', 'glm', 'dsv')
CELLS = ('old_qwen38_low', 'old_glm53', 'old_dsv41', 'watch_qwen38_low', 'watch_glm53', 'watch_dsv41',
         'new_glm53', 'new_dsv41', 'new_haiku55', 'new_sonnet55_low', 'new_gemini38_low')
PRIMARY = CELLS[:6]
SITS = {'waiting_line': 'line', 'conversation_group': 'group', 'audience': 'watch', 'no_one': 'no_one'}

out = []


def p(*a):
    out.append(' '.join(str(x) for x in a))


def frac(k, n):
    return f'{k}/{n} = {100 * k / n:.1f}%' if n else f'{k}/0 = -'


def r0_ok(name):
    path = C.OUT / f'r0_{name}.txt'
    return path.exists() and 'FULL REPRODUCTION: yes' in path.read_text()


# ======================================================================================================== arm 1
def arm1(summary):
    rows = C.read_jsonl(C.HERE / 'rows_crowd.jsonl')
    oracle = {r['key']: r for r in C.read_jsonl(C.I06_SCORE / 'rows_oracle.jsonl')}
    by = {}
    for r in rows:
        by[(r['seed'], r['scene_id'], r['rule'])] = r
    scenes = sorted({(r['seed'], r['scene_id']) for r in rows})
    assert len(scenes) == 229 and len(rows) == 229 * 4
    meta = {k: by[(*k, 'R0')] for k in scenes}
    base_of = {k: (k[0], meta[k]['base_scene_id']) for k in scenes}
    variants = [k for k in scenes if meta[k]['kind'] != 'base']
    bases = [k for k in scenes if meta[k]['kind'] == 'base']
    fl = lambda k: (meta[k]['family'], meta[k]['level'])  # noqa: E731
    sit = lambda k: meta[k]['situation']  # noqa: E731
    l1 = lambda k, rule: by[(*k, rule)]['l1']  # noqa: E731
    l2 = lambda k, rule: by[(*k, rule)]['l2']  # noqa: E731

    J = [k for k in variants if sit(k) == LINE and fl(k) in J_CELLS and not l1(k, 'R0')]
    grp = [k for k in variants if sit(k) in GROUPISH]
    a46 = [k for k in variants if sit(k) == LINE and meta[k]['family'] == 'A' and meta[k]['level'] >= 4]
    e2 = [k for k in variants if sit(k) == LINE and fl(k) == ('E', 2) and not l1(k, 'R0')]
    line_fail = [k for k in variants if sit(k) == LINE and not l1(k, 'R0')]

    p('== 1. Arm 1: I06 crowd variants, oracle C (I06 arm 3), 229 kept scenes, both episodes; verdict = e0 ==')
    p(f'R0 reproduction (out/r0_crowd.txt): {"FULL" if r0_ok("crowd") else "NOT FULL"}')
    p(f'R0 line failures (L1): {len(line_fail)}; judged set J (A1-A3, B1, C1, D2 line variants failing under R0): '
      f'{len(J)} = ' + ', '.join(f'{f}{l} {sum(fl(k) == (f, l) for k in J)}' for f, l in sorted(J_CELLS)))
    p(f'bases {len(bases)}, group and audience variants {len(grp)}, A4-A6 line variants {len(a46)}, '
      f'E2 R0 failures {len(e2)}')
    p()

    # --- table 1.1 by family and level
    p('Table 1.1. Success by family and level, per rule (L1 and L2; seeds pooled; level 0 = bases). drop = paired L1 '
      'drop against the scene\'s own base under the same rule, mean x 100. tdist = median place-to-target distance (m).')
    head = 'fam lvl   n |' + '|'.join(f' {r:>4}: L1      L2   drop tdist ' for r in RULES)
    p(head)
    groups = collections.defaultdict(list)
    for k in scenes:
        groups[fl(k)].append(k)
    t11 = {}
    for (f, l) in sorted(groups, key=lambda x: (x[0] != 'base', x[0], x[1])):
        ks = groups[(f, l)]
        cells = []
        for rule in RULES:
            a = sum(l1(k, rule) for k in ks)
            b = sum(l2(k, rule) for k in ks)
            if f == 'base':
                drop = 0.0
            else:
                drop = 100 * statistics.mean(int(l1(base_of[k], rule)) - int(l1(k, rule)) for k in ks)
            td = [by[(*k, rule)]['det']['target_distance_m'] for k in ks]
            td = [x for x in td if x is not None]
            med = statistics.median(td) if td else None
            cells.append(f' {a:3d}/{len(ks):<3d} {b:3d}/{len(ks):<3d} {drop:5.1f} {"-" if med is None else f"{med:5.2f}"} ')
            t11[f'{f}{l} {rule}'] = {'n': len(ks), 'l1': a, 'l2': b, 'drop_l1': round(drop, 1), 'tdist_med': med}
        p(f'{f:4s} {l:2d} {len(ks):4d} |' + '|'.join(cells))
    p('Paired L1 drop against the scene\'s own base, by family (levels and seeds pooled), per rule:')
    for fam in sorted({meta[k]['family'] for k in variants}):
        ks = [k for k in variants if meta[k]['family'] == fam]
        p(f'  {fam}: n {len(ks)}; ' + ', '.join(
            f'{rule} {100 * statistics.mean(int(l1(base_of[k], rule)) - int(l1(k, rule)) for k in ks):.1f}'
            for rule in RULES))
    p()

    # --- table 1.2 by situation
    p('Table 1.2. L1 success by situation, family and level, per rule (counts; seeds pooled).')
    p('fam lvl |' + '|'.join(f' {SITS[s]:>5}: n ' + ' '.join(f'{r:>4}' for r in RULES) + ' ' for s in (LINE,) + GROUPISH))
    for (f, l) in sorted(groups, key=lambda x: (x[0] != 'base', x[0], x[1])):
        cells = []
        for s in (LINE,) + GROUPISH:
            ks = [k for k in groups[(f, l)] if sit(k) == s]
            if ks:
                cells.append(f' {SITS[s]:>5}: {len(ks):2d} ' + ' '.join(f'{sum(l1(k, r) for k in ks):4d}' for r in RULES) + ' ')
            else:
                cells.append(f' {SITS[s]:>5}:  0 ' + ' '.join(f'{"-":>4}' for r in RULES) + ' ')
        p(f'{f:4s} {l:2d} |' + '|'.join(cells))
    p()

    # --- judged set and S1 / K2
    p('Table 1.3. The judged set J (54 R0 L1 failures at A1-A3, B1, C1, D2 lines): successes under each rule.')
    jres = {}
    for rule in RULES:
        per = {f'{f}{l}': sum(l1(k, rule) for k in J if fl(k) == (f, l)) for f, l in sorted(J_CELLS)}
        jres[rule] = {'total': sum(l1(k, rule) for k in J), 'by_cell': per,
                      'l2_total': sum(l2(k, rule) for k in J)}
        p(f'  {rule:4s}: L1 {frac(jres[rule]["total"], len(J))}  ({", ".join(f"{a} {b}" for a, b in per.items())}); '
          f'L2 {jres[rule]["l2_total"]}/{len(J)}')
    p()
    p('Table 1.4. Breaks (R0 L1 success -> failure) and flips (R0 failure -> success), per rule, by where they fall.')
    brk = {}
    for rule in REP:
        b_base = [k for k in bases if l1(k, 'R0') and not l1(k, rule)]
        b_grp = [k for k in grp if l1(k, 'R0') and not l1(k, rule)]
        b_line_other = [k for k in variants if sit(k) == LINE and k not in J and l1(k, 'R0') and not l1(k, rule)]
        flips = [k for k in scenes if not l1(k, 'R0') and l1(k, rule)]
        breaks = [k for k in scenes if l1(k, 'R0') and not l1(k, rule)]
        flips2 = [k for k in scenes if not l2(k, 'R0') and l2(k, rule)]
        breaks2 = [k for k in scenes if l2(k, 'R0') and not l2(k, rule)]
        brk[rule] = {'bases': len(b_base), 'group_audience': len(b_grp), 'line_outside_J': len(b_line_other),
                     'flips_l1': len(flips), 'breaks_l1': len(breaks), 'flips_l2': len(flips2),
                     'breaks_l2': len(breaks2),
                     'flips_l1_by_fl': dict(collections.Counter(f'{fl(k)[0]}{fl(k)[1]}' for k in flips)),
                     'list_breaks_l1': [f'{k[0]}/{k[1]}' for k in breaks]}
        p(f'  {rule}: L1 flips {len(flips)}, breaks {len(breaks)} (bases {len(b_base)}, group/audience variants '
          f'{len(b_grp)}, line variants outside J {len(b_line_other)}); L2 flips {len(flips2)}, breaks {len(breaks2)}')
        p(f'        L1 flips by family/level: {dict(sorted(brk[rule]["flips_l1_by_fl"].items()))}')
        for k in breaks:
            p(f'        L1 break: seed {k[0]} {k[1]} row {meta[k]["row"]} {fl(k)[0]}{fl(k)[1]} {sit(k)}')
        for k in breaks2:
            if k not in breaks:
                p(f'        L2-only break: seed {k[0]} {k[1]} row {meta[k]["row"]} {fl(k)[0]}{fl(k)[1]} {sit(k)}')
        for k in sorted(flips, key=lambda k: (fl(k), k)):
            p(f'        L1 flip: seed {k[0]} {k[1]} row {meta[k]["row"]} {fl(k)[0]}{fl(k)[1]} {sit(k)}'
              + ('' if l2(k, rule) else ' (L2 still fails)'))
        for k in flips2:
            if k not in flips:
                p(f'        L2-only flip: seed {k[0]} {k[1]} row {meta[k]["row"]} {fl(k)[0]}{fl(k)[1]} {sit(k)}')
    p()
    p('Table 1.5. Not judged: A4-A6 (cue-threshold ruling) and E2, L1 successes per rule.')
    for name, ks in (('A4', [k for k in a46 if fl(k) == ('A', 4)]), ('A5', [k for k in a46 if fl(k) == ('A', 5)]),
                     ('A6', [k for k in a46 if fl(k) == ('A', 6)]), ('E2 (R0 failures)', e2)):
        p(f'  {name:16s} n {len(ks):2d}: ' + ', '.join(f'{r} {sum(l1(k, r) for k in ks)}' for r in RULES))
    p()

    # --- failure (i)
    rel = [k for k in variants if oracle[f'{k[0]}/{k[1]}']['relevant']['i']]
    assert len(rel) == 153, len(rel)
    p(f'Table 1.6. Failure (i) firing: det_i in the variant and not in its base under the same rule; denominator I06\'s '
      f'{len(rel)} relevant variants (relevant[i]).')
    fires = {}
    for rule in RULES:
        f_all = [k for k in rel if by[(*k, rule)]['det']['det_i'] and not by[(*base_of[k], rule)]['det']['det_i']]
        byfam = collections.Counter(meta[k]['family'] for k in f_all)
        nfam = collections.Counter(meta[k]['family'] for k in rel)
        fires[rule] = {'n': len(f_all), 'of': len(rel), 'by_family': {f: [byfam.get(f, 0), nfam[f]] for f in sorted(nfam)}}
        p(f'  {rule:4s}: {frac(len(f_all), len(rel))}; by family ' +
          ', '.join(f'{f} {byfam.get(f, 0)}/{nfam[f]}' for f in sorted(nfam)))
    p()

    # --- the gate's view of R0's joiners
    p('Table 1.7. Every person R0 absorbed (e0), with the gate\'s f . u towards the current last, by family and level; '
      'and what each repair rule did in those scenes.')
    agg = collections.defaultdict(list)
    for k in scenes:
        for a in by[(*k, 'R0')]['episodes']['e0']['absorbed']:
            agg[fl(k)].append(a['fu'])
    for key in sorted(agg):
        fus = [x for x in agg[key] if x is not None]
        p(f'  {key[0]}{key[1]}: absorbed {len(agg[key])}, yaw unknown {sum(x is None for x in agg[key])}, f.u min '
          f'{min(fus):.3f} median {statistics.median(fus):.3f} max {max(fus):.3f}; f.u > -1e-9 (RF90 admits) '
          f'{sum(x > -1e-9 for x in fus)}, f.u > 0.5 - 1e-9 (RF60 admits) {sum(x > 0.5 - 1e-9 for x in fus)}')
    p()
    p('Table 1.8. J, scene by scene: R0 joiners (number: f.u), then per repair rule the joined list and L1.')
    for k in sorted(J, key=lambda k: (fl(k), k)):
        r0 = by[(*k, 'R0')]['episodes']['e0']
        ab = ' '.join(f'{a["number"]}:{a["fu"]:+.2f}' if a['fu'] is not None else f'{a["number"]}:?' for a in r0['absorbed'])
        rest = ' | '.join(f'{r} {by[(*k, r)]["episodes"]["e0"]["joined"]} {"S" if l1(k, r) else "F"}' for r in REP)
        p(f'  s{k[0]} {fl(k)[0]}{fl(k)[1]} {k[1][:44]:44s} R0 joined [{ab}] | {rest}')
    p()

    s1_fix = jres['RF90']['total']
    s1a = s1_fix >= 0.5 * len(J)
    s1b = brk['RF90']['bases'] == 0 and brk['RF90']['group_audience'] == 0
    k2 = s1_fix < 0.25 * len(J)
    summary['arm1'] = {'r0_full': r0_ok('crowd'), 'J': len(J), 'J_by_cell': {f'{f}{l}': sum(fl(k) == (f, l) for k in J)
                       for f, l in sorted(J_CELLS)}, 'J_success': jres, 'breaks': brk, 'fires_i': fires,
                       'table_1_1': t11, 'S1_part1': s1a, 'S1_part2': s1b, 'S1': s1a and s1b, 'K2': k2,
                       'RF90_fixes': s1_fix, 'need_S1': 0.5 * len(J), 'K2_below': 0.25 * len(J)}
    p(f'S1 reading: RF90 successes in J {s1_fix}/{len(J)} (>= {0.5 * len(J):g} needed): {s1a}; RF90 breaks at bases '
      f'{brk["RF90"]["bases"]} and at group/audience variants {brk["RF90"]["group_audience"]} (0 needed): {s1b}. '
      f'S1 holds: {s1a and s1b}.')
    p(f'K2 reading: RF90 successes in J {s1_fix} < {0.25 * len(J):g}: {k2}.')
    p()


# ======================================================================================================== arm 1b
def arm1b(summary):
    p('== 2. Arm 1b: I06 noisy arm on the 168 line scenes, 20 draws, R0 and RF90 (secondary, not judged) ==')
    res = {}
    for m in MODELS:
        path = C.HERE / f'rows_noisy_{m}.jsonl'
        if not path.exists():
            p(f'  {m}: no rows')
            continue
        rows = C.read_jsonl(path)
        by = {(r['seed'], r['scene_id'], r['rule']): r for r in rows}
        scenes = sorted({(r['seed'], r['scene_id']) for r in rows})
        meta = {k: by[(*k, 'R0')] for k in scenes}
        full = r0_ok(f'noisy_{m}')
        p(f'Model {m}: R0 reproduction of I06\'s draws (out/r0_noisy_{m}.txt): '
          f'{"FULL (compared with I06\'s R0)" if full else "NOT FULL: I06\'s draws not reproduced; RF90 compared with this probe\'s own R0"}')
        groups = collections.defaultdict(list)
        for k in scenes:
            groups[(meta[k]['family'], meta[k]['level'])].append(k)
        p('  fam lvl   n | R0 L1   RF90 L1   diff | R0 L2  RF90 L2 | draws: flips breaks | det_i R0 RF90')
        tab = {}
        for key in sorted(groups, key=lambda x: (x[0] != 'base', x[0], x[1])):
            ks = groups[key]
            a = statistics.mean(by[(*k, 'R0')]['l1_mean'] for k in ks)
            b = statistics.mean(by[(*k, 'RF90')]['l1_mean'] for k in ks)
            a2 = statistics.mean(by[(*k, 'R0')]['l2_mean'] for k in ks)
            b2 = statistics.mean(by[(*k, 'RF90')]['l2_mean'] for k in ks)
            fl_ = br_ = di0 = di1 = 0
            for k in ks:
                for x, y in zip(by[(*k, 'R0')]['draws'], by[(*k, 'RF90')]['draws']):
                    fl_ += (not x['e0']['l1']) and y['e0']['l1']
                    br_ += x['e0']['l1'] and not y['e0']['l1']
                    di0 += x['det_i']
                    di1 += y['det_i']
            nd = 20 * len(ks)
            tab[f'{key[0]}{key[1]}'] = {'n': len(ks), 'R0_l1': round(a, 4), 'RF90_l1': round(b, 4),
                                        'R0_l2': round(a2, 4), 'RF90_l2': round(b2, 4), 'flips': fl_, 'breaks': br_,
                                        'det_i_R0': di0, 'det_i_RF90': di1, 'draws': nd}
            p(f'  {key[0]:4s} {key[1]:2d} {len(ks):4d} | {a:.3f}   {b:.3f}   {b - a:+.3f} | {a2:.3f}  {b2:.3f} | '
              f'{fl_:5d} {br_:5d} of {nd} | {di0 / nd:.3f} {di1 / nd:.3f}')
        alld = sum(v['draws'] for v in tab.values())
        tot = {'flips': sum(v['flips'] for v in tab.values()), 'breaks': sum(v['breaks'] for v in tab.values()),
               'det_i_R0': sum(v['det_i_R0'] for v in tab.values()) / alld,
               'det_i_RF90': sum(v['det_i_RF90'] for v in tab.values()) / alld,
               'R0_l1': statistics.mean(by[(*k, 'R0')]['l1_mean'] for k in scenes),
               'RF90_l1': statistics.mean(by[(*k, 'RF90')]['l1_mean'] for k in scenes)}
        p(f'  all 168 scenes: mean L1 R0 {tot["R0_l1"]:.3f}, RF90 {tot["RF90_l1"]:.3f}; per-draw flips {tot["flips"]}, '
          f'breaks {tot["breaks"]} of {alld}; det_i rate R0 {tot["det_i_R0"]:.3f}, RF90 {tot["det_i_RF90"]:.3f}')
        res[m] = {'r0_full': full, 'by_family_level': tab, 'total': tot}
        p()
    summary['arm1b'] = res


# ======================================================================================================== arm 2
def arm2(summary):
    p('== 3. Arm 2: development records, every goal step re-solved under each rule ==')
    p('Settings: (a) the record\'s own inputs; (b) I01\'s P (true people, recorded constraints stripped); '
      '(c) A2\'s D (frozen I09 detector people), realistic cells only. Flips and breaks against R0 of the same setting.')
    full = {s: r0_ok(f'records_{s}') for s in 'abc'}
    p(f'R0 reproduction: (a) {full["a"]}, (b) {full["b"]}, (c) {full["c"]} (out/r0_records_*.txt)')
    p()
    res = {}
    lists = []
    absorbed_rows = []
    for cell in CELLS:
        path = C.HERE / f'rows_records_{cell}.jsonl'
        rows = C.read_jsonl(path)
        res[cell] = {}
        for s in ('a', 'b', 'c'):
            if s not in rows[0]['settings']:
                continue
            have = [r for r in RULES if r in rows[0]['settings'][s]]
            d = {'n': len(rows)}
            r0s = [r['settings'][s]['R0'] for r in rows]
            d['R0'] = sum(x['success'] for x in r0s)
            d['absorbed_steps'] = sum(bool(x.get('joined')) for x in r0s)
            d['absorbed_steps_R0_success'] = sum(bool(x.get('joined')) and x['success'] for x in r0s)
            d['errors'] = {}
            for rule in have:
                errs = [r for r in rows if 'error' in r['settings'][s][rule]]
                if errs:
                    d['errors'][rule] = [(r['run'], r['scene'], r['episode'], r['trial'], r['settings'][s][rule]['error'])
                                         for r in errs]
            for rule in have[1:]:
                xs = [r['settings'][s][rule] for r in rows]
                fl_ = [r for r, x, y in zip(rows, r0s, xs) if not x['success'] and y['success']]
                br_ = [r for r, x, y in zip(rows, r0s, xs) if x['success'] and not y['success']]
                d[rule] = {'success': sum(y['success'] for y in xs), 'flips': len(fl_), 'breaks': len(br_),
                           'net': len(fl_) - len(br_),
                           'by_situation': {SITS[sit]: [sum(y['success'] for r, y in zip(rows, xs) if r['situation'] == sit),
                                                        sum(x['success'] for r, x in zip(rows, r0s) if r['situation'] == sit),
                                                        sum(r['situation'] == sit for r in rows)]
                                            for sit in sorted({r['situation'] for r in rows})}}
                for kind, lst in (('flip', fl_), ('break', br_)):
                    for r in lst:
                        lists.append((cell, s, rule, kind, r['run'], r['arm'], r['scene'], r['layout'], r['episode'],
                                      r['trial'], r['situation'], r['settings'][s]['R0'].get('joined'),
                                      r['settings'][s][rule].get('joined'),
                                      [a['fu'] for a in r['settings'][s]['R0'].get('absorbed', [])]))
            for r in rows:
                for a in r['settings'][s]['R0'].get('absorbed', []):
                    absorbed_rows.append((cell, s, r['situation'], a['yaw_known'], a['fu'], r['settings'][s]['R0']['success'],
                                          r['run'], r['scene'], r['episode'], r['trial'], a['number'],
                                          {rule: r['settings'][s][rule]['success'] for rule in have}))
            res[cell][s] = d
    # table 3.1
    p('Table 3.1. Successes per cell, setting and rule; flips/breaks/net against R0 of the same setting. abs = goal steps '
      'in which R0 absorbed someone (of them R0 successes).')
    p('cell              set   n  abs(ok) |   R0 | RF90  f  b  net | RF60  f  b  net |   RN  f  b  net')
    for cell in CELLS:
        for s, d in res[cell].items():
            line = f'{cell:17s}  ({s}) {d["n"]:3d} {d["absorbed_steps"]:3d}({d["absorbed_steps_R0_success"]:3d}) | {d["R0"]:4d} |'
            for rule in REP:
                if rule in d:
                    x = d[rule]
                    line += f' {x["success"]:4d} {x["flips"]:2d} {x["breaks"]:2d} {x["net"]:+4d} |'
                else:
                    line += '    -  -  -    - |'
            p(line)
    p()
    p('Table 3.2. Every flip and break: cell, setting, rule, kind, run, arm, scene, layout, episode, trial, situation, '
      'R0 joined, rule joined, f.u of R0\'s joiners.')
    for x in lists:
        p('  ' + ' '.join(str(v) for v in x))
    if not lists:
        p('  none')
    p()
    p('Table 3.3. Successes by situation (rule / R0 / n), per cell and setting, RF90.')
    for cell in CELLS:
        for s, d in res[cell].items():
            if 'RF90' in d:
                p(f'  {cell:17s} ({s}) ' + ', '.join(f'{k} {v[0]}/{v[1]}/{v[2]}' for k, v in d['RF90']['by_situation'].items()))
    p()
    p('Table 3.4. Every person R0 absorbed on the records: yaw known and f.u, per cell and setting '
      '(f.u buckets: <= -0.5, (-0.5, 0], (0, 0.5], > 0.5; unknown yaw separate).')
    for cell in CELLS:
        for s in res[cell]:
            xs = [a for a in absorbed_rows if a[0] == cell and a[1] == s]
            if not xs:
                p(f'  {cell:17s} ({s}) none absorbed')
                continue
            fus = [a[4] for a in xs if a[4] is not None]
            b = [sum(f <= -0.5 for f in fus), sum(-0.5 < f <= -1e-9 for f in fus), sum(-1e-9 < f <= 0.5 - 1e-9 for f in fus),
                 sum(f > 0.5 - 1e-9 for f in fus)]
            p(f'  {cell:17s} ({s}) absorbed {len(xs)} (yaw unknown {sum(not a[3] for a in xs)}); f.u buckets {b}; '
              f'situations {dict(collections.Counter(a[2] for a in xs))}')
    p()
    p('Table 3.4b. Each person R0 absorbed on the records: cell, setting, run, scene, episode, trial, situation, number, '
      'yaw known, f.u, success under R0/RF90/RF60/RN.')
    for a in absorbed_rows:
        fu = '?' if a[4] is None else f'{a[4]:+.3f}'
        p(f'  {a[0]} ({a[1]}) {a[6]} {a[7]} {a[8]} {a[9]} {a[2]} #{a[10]} yaw_known {a[3]} f.u {fu} '
          + ' '.join(f'{rule} {"S" if v else "F"}' for rule, v in a[11].items()))
    p()
    p('Table 3.5. What RN costs ("how often absorption is needed"): RN breaks per cell and setting.')
    for cell in CELLS:
        p(f'  {cell:17s} ' + ', '.join(f'({s}) {d["RN"]["breaks"]} breaks, {d["RN"]["flips"]} flips' for s, d in res[cell].items()
                                       if 'RN' in d))
    errs = [(c, s, d['errors']) for c in CELLS for s, d in res[c].items() if d['errors']]
    p()
    p(f'Solves that raised (count as failures): {errs or "none"}')
    p()
    # S2 / K1
    s2_cells = {}
    pooled = 0
    k1_cell = False
    for cell in PRIMARY:
        d = res[cell]['a']['RF90']
        s2_cells[cell] = {'breaks': d['breaks'], 'flips': d['flips'], 'net': d['net'],
                          'ok': d['breaks'] <= 1 and d['net'] >= 0}
        pooled += d['net']
        k1_cell |= d['breaks'] >= 3
    s2 = all(v['ok'] for v in s2_cells.values())
    k1 = k1_cell or pooled < 0
    p('S2 reading (setting (a), RF90, primary cells): ' + '; '.join(
        f'{c} breaks {v["breaks"]} net {v["net"]:+d} {"ok" if v["ok"] else "FAILS"}' for c, v in s2_cells.items()))
    p(f'S2 holds: {s2}. K1 reading: any primary cell with >= 3 breaks: {k1_cell}; pooled net {pooled:+d} (< 0 fires): '
      f'{pooled < 0}. K1 fires: {k1}.')
    p()
    summary['arm2'] = {'r0_full': full, 'cells': res, 'S2_cells': s2_cells, 'S2': s2, 'K1': k1,
                       'K1_cell_3_breaks': k1_cell, 'pooled_net_a': pooled, 'n_flip_break_listed': len(lists)}


def main():
    summary = {'frozen_sha256': C.FROZEN_SHA, 'written': C.now(),
               'deviations_sha256': C.sha256(C.HERE / 'DEVIATIONS.md')}
    p(f'I06 repair probe: tables (FROZEN.md sha256 {C.FROZEN_SHA[:8]}...; DEVIATIONS.md sha256 '
      f'{summary["deviations_sha256"][:8]}...). Generated by code/tables.py {C.now()} from rows_*.jsonl and out/r0_*.txt.')
    p('Rules: R0 the package\'s _turn; RF90 (primary) joins only with f . u > -1e-9; RF60 f . u > 0.5 - 1e-9; RN nobody '
      'joins. Judged on point estimates; only RF90 is judged.')
    p()
    arm1(summary)
    arm1b(summary)
    arm2(summary)
    a1, a2 = summary['arm1'], summary['arm2']
    repro = a1['r0_full'] and a2['r0_full']['a']
    if not repro:
        verdict = 'incomplete'
    elif a2['K1'] or a1['K2']:
        verdict = 'kill'
    elif a1['S1'] and a2['S2']:
        verdict = 'method_candidate'
    else:
        verdict = 'neither'
    summary['verdict'] = verdict
    summary['criteria'] = {'S1': a1['S1'], 'S2': a2['S2'], 'K1': a2['K1'], 'K2': a1['K2'],
                           'R0_reproduced_arm1_and_a': repro}
    p('== 4. Verdict (FROZEN section 6, computed) ==')
    p(f'S1 {a1["S1"]} (RF90 fixes {a1["RF90_fixes"]}/{a1["J"]}; bases/group-audience breaks '
      f'{a1["breaks"]["RF90"]["bases"]}/{a1["breaks"]["RF90"]["group_audience"]}); S2 {a2["S2"]}; '
      f'K1 {a2["K1"]} (pooled net {a2["pooled_net_a"]:+d}); K2 {a1["K2"]}; R0 reproduced (arm 1 and setting (a)) {repro}')
    p(f'VERDICT: {verdict}')
    (C.HERE / 'tables.txt').write_text('\n'.join(out) + '\n')
    (C.HERE / 'summary.json').write_text(json.dumps(summary, indent=1, default=str) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
