"""I06 step 8 (FROZEN.md section 10): the C5 gate-1 recount on the widened generator. Reuses phase1d/c5_gates/gate1.py's
row reader, frame, matching and summaries (imported, unchanged). Crowd scenes are this probe's own (guarded, loaded
with sgi.scene.load_scene). No scene is generated.

    c5_recount.py   -> score/c5_recount.txt, score/c5_recount.json
"""
from __future__ import annotations

import collections
import importlib.util
import itertools
import json

import numpy as np

import lib as L
from sgi.config import DEFAULT_CONFIG, load_config

spec = importlib.util.spec_from_file_location('gate1', L.RES / 'learned-models/flow-model-gates/gate1.py')
G1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G1)

ADDED_BYSTANDERS = {'A': 1, 'B': 3, 'C': 0, 'D': None, 'E': 1}     # D: the level (2 or 5)


def main():
    import hashlib
    assert hashlib.sha256((L.I06 / 'FROZEN.md').read_bytes()).hexdigest() == L.FROZEN_SHA
    cfg = load_config(DEFAULT_CONFIG)
    rws, skipped = G1.rows(cfg)
    assert len(rws) == 26, len(rws)
    entries = L.manifest_entries()
    kept = [e for e in entries if e['kind'] == 'variant' and e['accepted']]
    cfgs = sorted({(e['row'], e['family'], e['level']) for e in kept})
    spec_of = {e['row']: tuple(e['row_spec']) for e in entries}
    base_rows = [tuple(r[3]) + (None, None) for r in rws]
    var_rows = [spec_of[r] + (f, l) for r, f, l in cfgs]
    primary = sorted(set(base_rows) | set(var_rows), key=repr)
    base_distinct = len(set(base_rows))
    structure = sorted(set(base_rows) | {spec_of[r] + (f, l if f not in ('A', 'E') else None) for r, f, l in cfgs},
                       key=repr)

    def lower_v(r, f, l):
        recipe, amount, kind, byst, second = spec_of[r]
        add = l if f == 'D' else ADDED_BYSTANDERS[f]
        size = 0 if recipe == 'no_one' else int(amount)
        return (recipe, size, int(byst) + add, bool(second) or f == 'C')
    lower = sorted({G1.lower_key(r[3]) for r in rws} | {lower_v(*c) for c in cfgs}, key=repr)

    def verdict(n):
        return 'met' if n >= 30 else ('borderline (not met)' if n >= 25 else 'not met')

    # EMPIRICAL: added people's displacement between the three seed pairs, C5's frame
    by_cfg = collections.defaultdict(dict)
    for e in kept:
        by_cfg[(e['row'], e['family'], e['level'])][e['seed']] = e
    bases = {(e['seed'], e['row']): e for e in entries if e['kind'] == 'base'}
    emp = []
    for c in cfgs:
        seeds = by_cfg[c]
        locs = {}
        for s, e in sorted(seeds.items()):
            inputs, truth = L.load(e)
            psi, a, _ = G1.frame_of(truth)
            added = [a_['id'] for a_ in e['added']]
            P = G1.local([(p.x, p.y) for p in truth.people if p.id in added], psi, a)
            locs[s] = P
        da, db, dm = [], [], []
        for i, j in itertools.combinations(sorted(locs), 2):
            Pi, Pj = locs[i], locs[j]
            if len(Pi) == 0 or len(Pi) != len(Pj):
                continue
            da.extend(np.hypot(*(Pi - Pj).T).tolist())
            m0, m1 = G1.best_matching(Pi, Pj), G1.best_matching(Pi, G1.mirror(Pj))
            db.extend(m0.tolist())
            dm.extend((m0 if m0.sum() <= m1.sum() else m1).tolist())
        emp.append({'row': c[0], 'family': c[1], 'level': c[2], 'seeds': sorted(locs), 'n_added': len(next(iter(locs.values()))),
                    'a_by_order': G1.summarise(da), 'b_matching': G1.summarise(db), 'b_matching_or_mirror': G1.summarise(dm)})
    # base members' displacement over the same seed pairs (C5's comparison: about 0.05 m)
    base_emp = []
    for row in sorted({e['row'] for e in entries if e['kind'] == 'base'}):
        locs = {}
        for s in (0, 1, 2):
            e = bases[(s, row)]
            inputs, truth = L.load(e)
            psi, a, _ = G1.frame_of(truth)
            g = truth.group_for('e0', 't0')
            locs[s] = G1.local([(p.x, p.y) for p in truth.people if p.group == g], psi, a)
        da = []
        for i, j in itertools.combinations((0, 1, 2), 2):
            da.extend(np.hypot(*(locs[i] - locs[j]).T).tolist())
        base_emp.append({'row': row, 'members_a_by_id': G1.summarise(da)})

    out = {'PRIMARY_prime': {'count': len(primary), 'base_rows_distinct': base_distinct,
                             'variant_configurations': len(cfgs), 'tuples': [list(t) for t in primary]},
           'STRUCTURE': {'count': len(structure)}, 'LOWER_prime': {'count': len(lower), 'tuples': [list(t) for t in lower]},
           'gate1_verdict_primary_prime': verdict(len(primary)), 'EMPIRICAL': emp, 'base_members': base_emp}
    (L.SCORE / 'c5_recount.json').write_text(json.dumps(out, indent=1))
    Lw = []
    Lw.append('C5 gate-1 recount on the widened generator (FROZEN.md section 10; phase1d/c5_gates/FROZEN.md readings kept)')
    Lw.append(f'Development rows (both splits, both passes): {len(rws)}, {base_distinct} distinct tuples (C5 PRIMARY 22)')
    Lw.append(f'Variant configurations (row, family, level) with >= 1 accepted variant in seeds 0-2: {len(cfgs)}')
    Lw.append(f'PRIMARY\' (decides): {len(primary)} distinct (recipe, amount, kind, bystanders, second group, family, level)'
              f' -> gate 1 {verdict(len(primary))} (>= 30 met; 25-29 borderline)')
    Lw.append(f'STRUCTURE (A\'s six levels and E\'s two depths once per row): {len(structure)} -> {verdict(len(structure))}')
    Lw.append(f'LOWER\' (situation, structure size, bystanders incl. added ungrouped people, second-group presence): '
              f'{len(lower)} -> {verdict(len(lower))}')
    for t in lower:
        Lw.append(f'    {t}')
    Lw.append('')
    Lw.append('EMPIRICAL (descriptive): added people\'s displacement between seed pairs (0-1, 0-2, 1-2 where accepted), '
              'C5 frame p\' = R(-psi)(p - a); (a) by order, (b) best one-to-one, (b+m) best of (b) and mirrored. '
              'median/max m (n person-pairs)')

    def f(s):
        return '-' if s is None else f'{s["median_m"]:.2f}/{s["max_m"]:.2f} ({s["n"]})'
    for x in emp:
        Lw.append(f'  eval_{x["row"]:02d} {x["family"]}{x["level"]} seeds {x["seeds"]} added {x["n_added"]}: '
                  f'a {f(x["a_by_order"])}; b {f(x["b_matching"])}; b+m {f(x["b_matching_or_mirror"])}')
    Lw.append('Base members, same frame, by id over the three seed pairs:')
    for x in base_emp:
        Lw.append(f'  eval_{x["row"]:02d}: {f(x["members_a_by_id"])}')
    allb = [x['b_matching_or_mirror']['median_m'] for x in emp if x['b_matching_or_mirror']]
    Lw.append(f'Median over configurations of the (b+m) median: {np.median(allb):.2f} m (n configurations {len(allb)})')
    (L.SCORE / 'c5_recount.txt').write_text('\n'.join(Lw) + '\n')
    print('\n'.join(Lw))


if __name__ == '__main__':
    main()
