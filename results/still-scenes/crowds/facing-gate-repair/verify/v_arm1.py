"""Verifier: arm 1 analysis from verify/out/crowd_seed*.jsonl; compares with I06's arm 3 and the agent's rows_crowd.jsonl.
Prints a report and writes verify/out/arm1_v.json.
"""
from __future__ import annotations

import collections
import json
import math
import statistics
from pathlib import Path

RD = Path('results/2026-10-17-beyond-vlm')
HERE = Path(__file__).resolve().parent
I06 = RD / 'phase2/i06/score'
AG = RD / 'phase2/i06_repair'
RULES = ("R0", "RF90", "RF60", "RN")

rows = []
for s in (0, 1, 2):
    rows += [json.loads(l) for l in open(HERE / 'out' / f'crowd_seed{s}.jsonl')]
by = {(r['seed'], r['scene_id']): r for r in rows}
assert len(rows) == 229 == len(by), len(rows)
i06 = {}
for s in (0, 1, 2):
    for l in open(I06 / 'work' / f'arms_seed{s}.jsonl'):
        r = json.loads(l)
        i06[(r['seed'], r['scene_id'])] = r
orc = {(r['seed'], r['scene_id']): r for r in map(json.loads, open(I06 / 'rows_oracle.jsonl'))}
assert set(i06) == set(by) == set(orc)


def sit(r):
    s = r['situation']
    assert len(s) == 1, s
    return {'waiting_line': 'line', 'conversation_group': 'group', 'audience': 'watch'}.get(s[0], s[0])


def fl(r):
    return 'base' if r['kind'] == 'base' else f"{r['family']}{r['level']}"


# ---------------------------------------------------------------- R0 reproduction
mism = []
for k, r in by.items():
    for eid in ('e0', 'e1'):
        mine = r['rules']['R0'][eid]
        ref = i06[k]['episodes'][eid]['arm3']
        jref = [] if ref['note'] is None else ref['note']['joined_waiting']
        bad = []
        if mine['l1'] != ref['l1']:
            bad.append('l1')
        if mine['l2'] != ref['l2']:
            bad.append('l2')
        if (mine['xy'] is None) != (ref['xy'] is None) or (
                mine['xy'] is not None and max(abs(a - b) for a, b in zip(mine['xy'], ref['xy'])) > 1e-9):
            bad.append('xy')
        if mine['joined'] != jref:
            bad.append('joined')
        if bad:
            mism.append((k, eid, bad))
    if r['rules']['R0']['e0']['det_i'] != orc[k]['det']['i']:
        mism.append((k, 'det_i', r['rules']['R0']['e0']['det_i'], orc[k]['det']['i']))
print('R0 reproduction mismatches:', len(mism), mism[:10])

# ---------------------------------------------------------------- consequences and agreement
viol = []
for k, r in by.items():
    for rule in RULES:
        a, b = r['rules'][rule]['e0'], r['rules'][rule]['e1']
        if (a['l1'], a['l2'], a['xy']) != (b['l1'], b['l2'], b['xy']):
            viol.append((k, rule, 'episodes disagree'))
        for eid in ('e0', 'e1'):
            x, r0 = r['rules'][rule][eid], r['rules']['R0'][eid]
            if not r0['joined'] and (x['xy'] != r0['xy'] or x['l1'] != r0['l1'] or x['l2'] != r0['l2']):
                viol.append((k, rule, eid, 'consequence 1'))
            if rule == 'RN' and x['joined']:
                viol.append((k, rule, eid, 'RN joined'))
            if x['yaws_known'] is False:
                viol.append((k, rule, eid, 'unknown yaw'))
print('consequence/agreement violations:', len(viol), viol[:10])

# ---------------------------------------------------------------- counts
L = lambda r, rule: r['rules'][rule]['e0']['l1']  # noqa: E731
L2 = lambda r, rule: r['rules'][rule]['e0']['l2']  # noqa: E731
line_fail = collections.Counter(fl(r) for r in rows if sit(r) == 'line' and not L(r, 'R0'))
fail_other = [k for k, r in by.items() if sit(r) != 'line' and not L(r, 'R0')]
print('R0 line failures', sum(line_fail.values()), dict(sorted(line_fail.items())), 'failures off lines', fail_other)
bases = [r for r in rows if r['kind'] == 'base']
ga = [r for r in rows if r['kind'] != 'base' and sit(r) in ('group', 'watch')]
linev = [r for r in rows if r['kind'] != 'base' and sit(r) == 'line']
print('bases', len(bases), 'group/audience variants', len(ga), 'line variants', len(linev))


def in_J_family(r):
    return r['kind'] != 'base' and sit(r) == 'line' and (
        (r['family'] == 'A' and r['level'] in (1, 2, 3)) or r['family'] in ('B', 'C', 'D'))


J = [r for r in rows if in_J_family(r) and not L(r, 'R0')]
print('|J|', len(J), dict(collections.Counter(fl(r) for r in J)))
out = {'r0_mismatch': len(mism), 'violations': len(viol), 'J': len(J), 'line_fail': dict(line_fail)}
for rule in RULES:
    s1 = collections.Counter(fl(r) for r in J if L(r, rule))
    s2 = sum(L2(r, rule) for r in J)
    flips = [r for r in rows if not L(r, 'R0') and L(r, rule)]
    breaks = [r for r in rows if L(r, 'R0') and not L(r, rule)]
    flips2 = [r for r in rows if not L2(r, 'R0') and L2(r, rule)]
    breaks2 = [r for r in rows if L2(r, 'R0') and not L2(r, rule)]
    br_cat = collections.Counter('base' if r['kind'] == 'base' else ('ga' if sit(r) != 'line' else 'line')
                                 for r in breaks)
    print(f"{rule}: J L1 {sum(s1.values())}/{len(J)} {dict(sorted(s1.items()))}; J L2 {s2}; flips L1 {len(flips)} "
          f"{dict(collections.Counter(fl(r) for r in flips))}; breaks L1 {len(breaks)} {dict(br_cat)}; "
          f"flips L2 {len(flips2)} breaks L2 {len(breaks2)}")
    out[rule] = {'J_l1': sum(s1.values()), 'J_by': dict(s1), 'J_l2': s2, 'flips': sorted(f"{r['seed']}/{r['scene_id']}" for r in flips),
                 'breaks': sorted(f"{r['seed']}/{r['scene_id']}" for r in breaks), 'flips_l2': len(flips2),
                 'breaks_l2': len(breaks2)}

# A4-A6, E2
for f in ('A4', 'A5', 'A6'):
    rs = [r for r in rows if fl(r) == f and sit(r) == 'line']
    print(f, len(rs), {rule: sum(L(r, rule) for r in rs) for rule in RULES})
e2f = [r for r in rows if fl(r) == 'E2' and sit(r) == 'line' and not L(r, 'R0')]
print('E2 R0 failures', len(e2f), {rule: sum(L(r, rule) for r in e2f) for rule in RULES})

# Table 1.1: by family level
print('family/level L1 L2 tdist-median per rule')
groups = collections.defaultdict(list)
for r in rows:
    groups[fl(r)].append(r)
t11 = {}
for key in sorted(groups):
    rs = groups[key]
    cells = []
    for rule in RULES:
        td = [r['rules'][rule]['e0']['tdist'] for r in rs if r['rules'][rule]['e0']['tdist'] is not None]
        cells.append(f"{rule} {sum(L(r, rule) for r in rs)}/{sum(L2(r, rule) for r in rs)}/{len(rs)} "
                     f"td {statistics.median(td) if td else float('nan'):.2f}")
        t11[f'{key}/{rule}'] = [sum(L(r, rule) for r in rs), sum(L2(r, rule) for r in rs), len(rs),
                                round(statistics.median(td), 2) if td else None]
    print(' ', key, ' | '.join(cells))
out['t11'] = t11
# Table 1.2 by situation
t12 = {}
for key in sorted(groups):
    for s in ('line', 'group', 'watch'):
        rs = [r for r in groups[key] if sit(r) == s]
        if rs:
            t12[f'{key}/{s}'] = [len(rs)] + [sum(L(r, rule) for r in rs) for rule in RULES]
print('by situation', t12)
out['t12'] = t12

# paired drop by family
for fam in 'ABCDE':
    rs = [r for r in rows if r['family'] == fam]
    drops = {rule: 100 * statistics.mean(int(L(by[(r['seed'], r['base_scene_id'])], rule)) - int(L(r, rule)) for r in rs)
             for rule in RULES}
    print('paired drop', fam, len(rs), {k: round(v, 1) for k, v in drops.items()})

# det (i) firing
rel = [k for k, r in orc.items() if r['kind'] != 'base' and r['relevant']['i']]
print('relevant (i)', len(rel), dict(collections.Counter(orc[k]['family'] for k in rel)))
out['det_i'] = {}
for rule in RULES:
    fires = [k for k in rel if by[k]['rules'][rule]['e0']['det_i']
             and not by[(k[0], by[k]['base_scene_id'])]['rules'][rule]['e0']['det_i']]
    fam = collections.Counter(by[k]['family'] for k in fires)
    print(f"det_i {rule}: {len(fires)}/{len(rel)} {dict(sorted(fam.items()))}")
    out['det_i'][rule] = [len(fires), dict(fam)]
n_join = [k for k, r in by.items() if r['rules']['R0']['e0']['joined']]
print('R0 scenes with a joiner', len(n_join), 'of them R0 failures', sum(not L(by[k], 'R0') for k in n_join),
      'bases among them', sum(by[k]['kind'] == 'base' for k in n_join))


# f.u of R0 joiners (e0)
def joiners_fu(entry_log, joined):
    """[(number, fu)] for each joiner, from the iteration it was chosen in."""
    assert len(entry_log) == 1, entry_log
    it = entry_log[0]['iters']
    outl = []
    for i, j in enumerate(joined):
        hit = [c for c in it[i] if c[0] == j]
        assert len(hit) == 1 and hit[0][4]
        outl.append((j, hit[0][3]))
    return outl


fu_by = collections.defaultdict(list)
for k, r in by.items():
    x = r['rules']['R0']['e0']
    if x['joined']:
        for j, fu in joiners_fu(x['log'], x['joined']):
            fu_by[fl(r)].append(fu)
print('f.u of R0 joiners by family/level (min, median, max, n, RF90 admits, RF60 admits):')
for key in sorted(fu_by):
    v = fu_by[key]
    kn = [x for x in v if x is not None]
    print(f"  {key}: n {len(v)} unknown {len(v) - len(kn)} min {min(kn):.3f} med {statistics.median(kn):.3f} "
          f"max {max(kn):.3f} RF90 {sum(x > -1e-9 for x in kn)} RF60 {sum(x > 0.5 - 1e-9 for x in kn)}")
out['fu'] = {k: v for k, v in fu_by.items()}

# J scene by scene
print('J scene by scene:')
for r in sorted(J, key=lambda r: (fl(r), r['row'], r['seed'])):
    x = r['rules']['R0']['e0']
    j0 = ' '.join(f"{j}:{fu:+.2f}" for j, fu in joiners_fu(x['log'], x['joined'])) if x['joined'] else ''
    print(f"  s{r['seed']} {fl(r)} {r['scene_id']:45s} R0 [{j0}] | " + ' | '.join(
        f"{rule} {r['rules'][rule]['e0']['joined']} {'S' if L(r, rule) else 'F'}" for rule in RULES[1:]))

# ---------------------------------------------------------------- compare with the agent's rows_crowd.jsonl
ag = {}
for l in open(AG / 'rows_crowd.jsonl'):
    a = json.loads(l)
    ag[(a['seed'], a['scene_id'], a['rule'])] = a
diff = []
for k, r in by.items():
    for rule in RULES:
        a = ag[(k[0], k[1], rule)]
        for eid in ('e0', 'e1'):
            m, x = r['rules'][rule][eid], a['episodes'][eid]
            if m['l1'] != x['l1'] or m['l2'] != x['l2']:
                diff.append((k, rule, eid, 'label', (m['l1'], m['l2']), (x['l1'], x['l2'])))
            if (m['xy'] is None) != (x['xy'] is None) or (m['xy'] is not None and max(abs(p - q) for p, q in zip(m['xy'], x['xy'])) > 1e-9):
                diff.append((k, rule, eid, 'xy', m['xy'], x['xy']))
        if r['rules'][rule]['e0']['det_i'] != a['det']['det_i']:
            diff.append((k, rule, 'det_i', r['rules'][rule]['e0']['det_i'], a['det']['det_i']))
        if r['rules'][rule]['e0']['joined'] != a['det']['joined_waiting']:
            diff.append((k, rule, 'joined', r['rules'][rule]['e0']['joined'], a['det']['joined_waiting']))
print('differences vs agent rows_crowd.jsonl:', len(diff), diff[:10])
out['diff_agent'] = len(diff)
json.dump(out, open(HERE / 'out' / 'arm1_v.json', 'w'), indent=1, default=str)
