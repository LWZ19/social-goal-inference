"""Verifier: arm 2 analysis from verify/out/records_*.jsonl; R0 reproduction against I01, A2 and the records; S2, K1;
comparison with the agent's rows_records_*.jsonl. Writes verify/out/arm2_v.json.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

RD = Path('results/2026-10-17-beyond-vlm')
HERE = Path(__file__).resolve().parent
AG = RD / 'phase2/i06_repair'
I01 = RD / 'phase1/i01_repairs'
A2 = RD / 'phase1c/i09/a2'
RULES = ("R0", "RF90", "RF60", "RN")
CELLS = ['old_qwen38_low', 'old_glm53', 'old_dsv41', 'watch_qwen38_low', 'watch_glm53', 'watch_dsv41',
         'new_glm53', 'new_dsv41', 'new_haiku55', 'new_sonnet55_low', 'new_gemini38_low']
PRIMARY = CELLS[:6]
TABLE51 = {'old_qwen38_low': (143, 134, 136, 136), 'old_glm53': (141, 96, 112, 112), 'old_dsv41': (157, 109, 120, 126),
           'watch_qwen38_low': (36, 32, 33, None), 'watch_glm53': (36, 23, 29, None), 'watch_dsv41': (36, 23, 31, None),
           'new_glm53': (141, 86, 107, 106), 'new_dsv41': (151, 106, 116, 119), 'new_haiku55': (146, 134, 135, 134),
           'new_sonnet55_low': (143, 134, 134, 134), 'new_gemini38_low': (135, 109, 109, 109)}


def keyed(path):
    return {(r['arm'], r['scene'], r['episode'], r['trial']): r for r in map(json.loads, open(path))}


def joiner_fu(log, joined):
    """f.u of each joiner, from the iteration in which it joined (one _turn call expected)."""
    if not joined:
        return []
    assert len(log) == 1, len(log)
    it = log[0]['iters']
    out = []
    for i, j in enumerate(joined):
        hit = [c for c in it[i] if c[0] == j]
        assert len(hit) == 1 and hit[0][4]
        out.append((j, hit[0][3]))
    return out


report = {}
all_flips = []
mism_total = collections.Counter()
diff_agent = []
ONLY = sys.argv[1:]
for cell in CELLS:
    if ONLY and cell not in ONLY:
        continue
    kind, model = cell.split('_', 1)
    rows = [json.loads(l) for l in open(HERE / 'out' / f'records_{cell}.jsonl')]
    key = lambda r: (r['arm'], r['scene'], r['episode'], r['trial'])  # noqa: E731
    if kind == 'watch':
        ref_b = keyed(A2 / f'base_ph_{model}.jsonl')
        ref_i01, ref_c = None, None
    else:
        w = 'old' if kind == 'old' else 'new'
        ref_i01 = keyed(I01 / f'rows_{w}_{model}.jsonl')
        ref_b = ref_i01
        ref_c = keyed(A2 / f'rows_{w}_{model}.jsonl')
    # R0 reproduction
    mism = collections.Counter()
    for r in rows:
        k = key(r)
        a0 = r['a']['R0']
        if not a0['reproduced']:
            mism['a_reproduced'] += 1
        if a0['ok'] != r['C_rec']:
            mism['a_vs_recorded'] += 1
        if ref_i01 is not None:
            o = ref_i01[k]
            assert o['solver_step']
            if a0['ok'] != o['C'] or r['C_rec'] != o['C_rec']:
                mism['a_vs_I01_C'] += 1
        if r['b']['R0']['ok'] != ref_b[k]['P']:
            mism['b_vs_P'] += 1
        if ref_c is not None:
            o = ref_c[k]
            if r['c']['R0']['ok'] != o['D']:
                mism['c_vs_D'] += 1
            if r['c']['R0']['joined'] != o['D_d']['joined']:
                mism['c_joined'] += 1
    mism_total.update(mism)
    n = len(rows)
    t = TABLE51[cell]
    counts = (n, sum(r['C_rec'] for r in rows), sum(r['b']['R0']['ok'] for r in rows),
              sum(r['c']['R0']['ok'] for r in rows) if kind != 'watch' else None)
    print(f"{cell}: n {n}; recorded {counts[1]}; R0 (b) {counts[2]}; R0 (c) {counts[3]}; Table 5.1 {t} "
          f"{'OK' if counts == t else 'DIFF'}; R0 mismatches {dict(mism)}")
    rep = {'n': n, 'table51_ok': counts == t, 'mism': dict(mism)}
    for s in ('a', 'b', 'c'):
        if s == 'c' and kind == 'watch':
            continue
        r0 = [r[s]['R0']['ok'] for r in rows]
        absorbed = [r for r in rows if r[s]['R0']['joined']]
        line = {'n': n, 'R0': sum(r0), 'abs': len(absorbed), 'abs_ok': sum(r[s]['R0']['ok'] for r in absorbed)}
        for rule in RULES[1:]:
            fl = [r for r in rows if not r[s]['R0']['ok'] and r[s][rule]['ok']]
            br = [r for r in rows if r[s]['R0']['ok'] and not r[s][rule]['ok']]
            raised = [r for r in rows if r[s][rule].get('error')]
            line[rule] = {'ok': sum(r[s][rule]['ok'] for r in rows), 'f': len(fl), 'b': len(br),
                          'net': len(fl) - len(br), 'raised': len(raised)}
            for kind_, lst in (('flip', fl), ('break', br)):
                for r in lst:
                    fus = joiner_fu(r[s]['R0']['log'], r[s]['R0']['joined'])
                    all_flips.append((cell, s, rule, kind_, r['run'], r['scene'], r['layout'], r['episode'],
                                      r['trial'], r['situation'], r[s]['R0']['joined'], r[s][rule]['joined'],
                                      [round(f, 3) if f is not None else None for _, f in fus]))
            # consequence 1 and RN
            for r in rows:
                if not r[s]['R0']['joined'] and (r[s][rule]['ok'] != r[s]['R0']['ok'] or r[s][rule]['xy'] != r[s]['R0']['xy']):
                    print('  CONSEQUENCE 1 VIOLATED', cell, s, rule, key(r))
                if rule == 'RN' and r[s][rule]['joined']:
                    print('  RN JOINED', cell, s, key(r))
        fus = []
        for r in absorbed:
            fus += [f for _, f in joiner_fu(r[s]['R0']['log'], r[s]['R0']['joined'])]
        line['fu'] = fus
        rep[s] = line
        print(f"  ({s}) R0 {line['R0']} abs {line['abs']}({line['abs_ok']}) | " + ' | '.join(
            f"{rule} {line[rule]['ok']} f{line[rule]['f']} b{line[rule]['b']} net{line[rule]['net']:+d}"
            + (f" raised {line[rule]['raised']}" if line[rule]['raised'] else '') for rule in RULES[1:])
            + f" | f.u of absorbed {[round(f, 3) if f is not None else None for f in fus]}")
    report[cell] = rep
    # compare with the agent's rows
    ag = {(a['arm'], a['scene'], a['episode'], a['trial']): a for a in map(json.loads, open(AG / f'rows_records_{cell}.jsonl'))}
    if set(ag) != {key(r) for r in rows}:
        diff_agent.append((cell, 'keys differ', len(ag), n))
    for r in rows:
        a = ag.get(key(r))
        if a is None:
            continue
        for s in ('a', 'b', 'c'):
            if s not in r:
                if s in a['settings']:
                    diff_agent.append((cell, key(r), s, 'agent has setting'))
                continue
            for rule in RULES:
                m, x = r[s][rule], a['settings'][s][rule]
                if m['ok'] != x['success'] or m['joined'] != x['joined']:
                    diff_agent.append((cell, key(r), s, rule, (m['ok'], m['joined']), (x['success'], x['joined'])))
print('R0 mismatches total', dict(mism_total))
print('flips and breaks:')
for f in all_flips:
    print(' ', *f)
if ONLY:
    print('differences vs agent rows_records:', len(diff_agent), diff_agent[:10])
    sys.exit(0)
# S2 / K1
s2 = all(report[c]['a']['RF90']['b'] <= 1 and report[c]['a']['RF90']['net'] >= 0 for c in PRIMARY)
k1_breaks = any(report[c]['a']['RF90']['b'] >= 3 for c in PRIMARY)
pooled = sum(report[c]['a']['RF90']['net'] for c in PRIMARY)
print('S2', s2, 'per cell (breaks, net)', [(c, report[c]['a']['RF90']['b'], report[c]['a']['RF90']['net']) for c in PRIMARY])
print('K1 any >=3 breaks', k1_breaks, 'pooled net', pooled, 'K1 fires', k1_breaks or pooled < 0)
print('primary goal steps with an R0 absorption in (a):', {c: report[c]['a']['abs'] for c in PRIMARY},
      'total', sum(report[c]['a']['abs'] for c in PRIMARY))
print('RN cost (breaks, flips) per cell and setting:')
for c in CELLS:
    print(' ', c, {s: (report[c][s]['RN']['b'], report[c][s]['RN']['f']) for s in ('a', 'b', 'c') if s in report[c]})
print('differences vs agent rows_records:', len(diff_agent), diff_agent[:10])
json.dump({'report': report, 'flips': all_flips, 'S2': s2, 'K1_breaks': k1_breaks, 'pooled': pooled,
           'diff_agent': len(diff_agent), 'mism': dict(mism_total)}, open(HERE / 'out' / 'arm2_v.json', 'w'), indent=1)
