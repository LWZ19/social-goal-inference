"""I06 scoring tables (FROZEN.md sections 7, 9 and 11): oracle curves, the documented failures, the noise validation and
noisy curves, criteria (a)-(e), the kill lines, the C5 recount summary, and the verdict. Reads only score/work, score/noise,
build/ and the C5 recount output; computes nothing new from scenes.

    tables.py   -> score/tables.txt, score/rows_oracle.jsonl, score/rows_noisy_<model>.jsonl, score/rows_validation.jsonl,
                   score/curves_*.png, score/summary.json
"""
from __future__ import annotations

import collections
import json
import math

import numpy as np

import lib as L

FAMILIES = ('A', 'B', 'C', 'D', 'E')
LEVELS = {'A': (1, 2, 3, 4, 5, 6), 'B': (1,), 'C': (1,), 'D': (2, 5), 'E': (1, 2)}
TOP = {'A': 6, 'B': 1, 'C': 1, 'D': 5, 'E': 2}
LINE_ROWS = (6, 7, 8, 9, 13)
SITS = ('waiting_line', 'conversation_group', 'audience')
SIT_SHORT = {'waiting_line': 'line', 'conversation_group': 'group', 'audience': 'watch'}
MODELS = [('qwen', 'Qwen 3.8'), ('glm', 'GLM-5.3'), ('dsv', 'DeepSeek V4.1')]
BOOT = 2000


def pct(k, n):
    return f'{k}/{n} = {100.0 * k / n:.1f}%' if n else f'{k}/0'


def fam_applies(f, row):
    return row in LINE_ROWS if f == 'A' else True


def load_arms():
    rows = []
    for s in (0, 1, 2):
        rows += [json.loads(l) for l in (L.SCORE / 'work' / f'arms_seed{s}.jsonl').read_text().splitlines() if l]
    return rows


def boot_ci(pairs, layouts, rng_seed=0, n=BOOT):
    """Layout-clustered bootstrap of a ratio mean: pairs = [(layout, value)], resampling the 30 base layouts."""
    by = collections.defaultdict(list)
    for lay, v in pairs:
        by[lay].append(v)
    sums = np.array([sum(by[l]) if l in by else 0.0 for l in layouts])
    cnts = np.array([len(by[l]) if l in by else 0 for l in layouts])
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(len(layouts), size=(n, len(layouts)))
    s, c = sums[idx].sum(axis=1), cnts[idx].sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        vals = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    import hashlib
    assert hashlib.sha256((L.I06 / 'FROZEN.md').read_bytes()).hexdigest() == L.FROZEN_SHA
    out = []

    def say(t=''):
        out.append(t)

    manifest = L.manifest_entries()
    planned = [e for e in manifest if e['kind'] == 'variant']
    labels = {}
    for s in (0, 1, 2):
        for l in (L.BUILD / f'labels_rows_seed{s}.jsonl').read_text().splitlines():
            if l:
                r = json.loads(l)
                labels[(r['seed'], r['scene_id'])] = r
    arms = load_arms()
    by_id = {r['scene_id']: r for r in arms}
    base_of = {}
    for r in arms:
        if r['kind'] == 'base':
            base_of[(r['seed'], r['scene_id'])] = r
    layouts = sorted({(r['seed'], r['row']) for r in arms if r['kind'] == 'base'})
    assert len(layouts) == 30
    variants = [r for r in arms if r['kind'] == 'variant']
    bases = [r for r in arms if r['kind'] == 'base']
    assert len(variants) == 199 and len(bases) == 30, (len(variants), len(bases))

    # ------------------------------------------------------------------------------------------- per-scene rows
    rows = []
    checks = collections.Counter()
    for r in arms:
        e0, e1 = r['episodes']['e0'], r['episodes']['e1']
        checks['episodes_agree'] += all(r['episodes_agree'].values())
        checks['equiv_ok'] += all(ep['equiv_m'] is not None and ep['equiv_m'] <= 1e-9 for ep in (e0, e1))
        checks['model_calls'] += r['model_calls']
        sit = r['situation'][0]
        assert len(r['situation']) == 1
        d3, d2 = e0['arm3']['det'], e0['arm2']['det']
        b = base_of[(r['seed'], r['base_scene_id'])]
        bd3, bd2 = b['episodes']['e0']['arm3']['det'], b['episodes']['e0']['arm2']['det']
        row = {k: r[k] for k in ('scene_id', 'kind', 'seed', 'row', 'family', 'level', 'base_scene_id', 'n_people')}
        row['situation'] = sit
        row['key'] = f"{r['seed']}/{r['scene_id']}"
        row['layout'] = [r['seed'], r['row']]
        for arm in ('arm1', 'arm2', 'arm3'):
            row[f'{arm}_l1'] = e0[arm]['l1']
            row[f'{arm}_l2'] = e0[arm]['l2']
            row[f'base_{arm}_l1'] = b['episodes']['e0'][arm]['l1']
            row[f'base_{arm}_l2'] = b['episodes']['e0'][arm]['l2']
        row['arm3_in_region'] = e0['arm3']['in_region']
        row['arm3_heading_ok'] = e0['arm3']['heading_ok']
        row['arm3_reachable'] = e0['arm3']['reachable']
        row['arm3_l2_fail'] = e0['arm3'].get('l2_fail')
        row['arm3_flags'] = e0['arm3']['flags']
        row['people_matter'] = e0['arm3']['people_matter']
        note = e0['arm3']['note']
        row['target_distance_m'] = None if note is None else round(note['target_distance_m'], 3)
        row['keys'] = None if note is None else note['keys']
        row['joined_waiting'] = d3['joined_waiting']
        row['joined_sources'] = d3['joined_sources']
        row['F_cells'] = d3['F_cells']
        row['base_F_cells'] = bd3['F_cells']
        row['min_pair_covering_m'] = d3['min_pair_covering_m']
        row['off_facing_deg'] = d3['off_facing_deg']
        row['base_off_facing_deg'] = bd3['off_facing_deg']
        row['not_ahead_of'] = d3['not_ahead_of']
        row['not_ahead_reason'] = d3['not_ahead_reason']
        row['base_not_ahead_of'] = bd3['not_ahead_of']
        row['ahead_of_hindmost_m'] = d3['ahead_of_hindmost_m']
        row['l1_cells_e0'] = r['l1_cells_e0']
        row['l1_cells_standable'] = d3['l1_cells_standable']
        row['equiv_m'] = max(e0['equiv_m'], e1['equiv_m'])
        row['episodes_agree'] = all(r['episodes_agree'].values())
        isvar = r['kind'] == 'variant'
        # detectors on arm 3 (FROZEN section 7): fires = true in the variant and false in its base
        det_ii_v = d3['F_cells'] == 0 and d3['det_ii_core']
        det_ii_b = bd3['F_cells'] == 0 and bd3['det_ii_core']
        row['det'] = {'i': d3['det_i'], 'ii': det_ii_v, 'iii': d3['det_iii'], 'iv': d3['det_iv_core'], 'v': d3['det_v']}
        row['base_det'] = {'i': bd3['det_i'], 'ii': det_ii_b, 'iii': bd3['det_iii'], 'iv': bd3['det_iv_core'], 'v': bd3['det_v']}
        row['relevant'] = {
            'i': isvar and d3['come_after_used'] and not bd3['det_i'],
            'ii': isvar and bd3['F_cells'] > 0 and not det_ii_b,
            'iii': isvar and d3['iii_applies'] and bd3['off_facing_deg'] is not None and bd3['off_facing_deg'] < L.OFF_FACING_DEG,
            'iv': isvar and r['family'] in ('B', 'C', 'D', 'E') and d3['iv_applies'] and bool(bd3['not_ahead_of']) and not bd3['det_iv_core'],
            'v': isvar and not bd3['det_v']}
        row['fires'] = {k: bool(isvar and row['det'][k] and not row['base_det'][k]) for k in row['det']}
        row['named_cause'] = any(row['fires'].values())
        # arm 2 detectors (descriptive)
        a2_ii_v = d2['F_cells'] == 0 and d2['det_ii_core']
        a2_ii_b = bd2['F_cells'] == 0 and bd2['det_ii_core']
        row['arm2_fires'] = {'ii': bool(isvar and a2_ii_v and not a2_ii_b),
                             'iii': bool(isvar and d2['det_iii'] and not bd2['det_iii']),
                             'v': bool(isvar and d2['det_v'] and not bd2['det_v'])}
        row['drop_arm3_l1'] = (int(row['base_arm3_l1']) - int(row['arm3_l1'])) if isvar else 0
        row['drop_arm3_l2'] = (int(row['base_arm3_l2']) - int(row['arm3_l2'])) if isvar else 0
        row['drop_arm2_l1'] = (int(row['base_arm2_l1']) - int(row['arm2_l1'])) if isvar else 0
        row['drop_arm1_l1'] = (int(row['base_arm1_l1']) - int(row['arm1_l1'])) if isvar else 0
        lab = labels[(r['seed'], r['scene_id'])]
        row['l2_empty'] = lab['l2_empty']
        row['rep_point_moved_m'] = lab['rep_point_moved_m']
        rows.append(row)
    with open(L.SCORE / 'rows_oracle.jsonl', 'w') as fh:
        for row in rows:
            fh.write(json.dumps(row) + '\n')
    R = {row['key']: row for row in rows}
    V = [row for row in rows if row['kind'] == 'variant']
    B = [row for row in rows if row['kind'] == 'base']

    say('I06 first probe: scoring tables (FROZEN.md sha256 a83c0ffe...693e; DEVIATIONS.md D1 only). Generated by '
        'score/tables.py from score/work (arms.py, noisy.py), score/noise (noise.py, validate.py) and build/.')
    say('Unit: one scene = one episode (both episodes share one target and one region set; asserted below). n = scenes.')
    say('Labels are the project\'s own rules: L1 = the members-only reference regions (sgi.gt), L2 = this probe\'s '
        'stricter variant (L1 + body gap >= 0.3 m to every non-member + outside other groups\' circles). Neither is '
        'social ground truth; the reference constraints of arms 3-4 read the situation label and the true group '
        '(circular, REPORT section 8).')
    say('')
    say('== 0. Checks ==')
    say(f'  kept scenes scored: {len(rows)} (variants {len(V)}, bases {len(B)}); planned variants {len(planned)}')
    say(f'  every arm gives both episodes the same result (L1, L2, place): {pct(checks["episodes_agree"], len(rows))}')
    say(f'  equivalence check (arm 3 with no constraint = arm 2, place within 1e-9 m, both episodes): '
        f'{pct(checks["equiv_ok"], len(rows))}; max distance {max(row["equiv_m"] for row in rows):.2e} m')
    say(f'  model calls made by arms 1-3: {checks["model_calls"]}')
    a1v = sum(row['arm1_l1'] for row in V)
    a1b = sum(row['arm1_l1'] for row in B)
    say(f'  arm 1 (full oracle) under L1: variants {pct(a1v, len(V))}, bases {pct(a1b, len(B))} '
        f'(the build\'s label check found the same: 229/229)')
    say('')

    # ------------------------------------------------------------------------------------------- 1. oracle curves
    say('== 1. Oracle curves (arms 1-3; L1 and L2; seeds pooled). Level 0 = the base scenes the family applies to '
        '(A: the 5 line rows x 3 seeds = 15; B-E: all 10 rows x 3 seeds = 30). ==')
    say('Paired drop at a level = mean over its accepted variants of [success(base) - success(variant)] x 100 (pairs '
        'removed with rejections). Arm 3 = oracle C (primary); arm 2 = solver alone; arm 1 = full oracle.')
    hdr = (f'  {"fam":3s} {"lvl":>3s} {"plan":>4s} {"kept":>4s} {"rej(reason)":18s} | {"arm3 L1":>14s} {"arm3 L2":>14s} '
           f'{"drop3 L1 [95% CI]":>24s} {"drop3 L2":>8s} | {"arm2 L1":>13s} {"arm2 L2":>13s} {"drop2":>6s} | '
           f'{"arm1 L1":>9s} {"arm1 L2":>9s} | {"far":>3s} {"exbr":>4s} {"tdist med":>9s}')
    say(hdr)
    curve = {}
    for f in FAMILIES:
        bl = [row for row in B if fam_applies(f, row['row'])]
        n = len(bl)
        say(f'  {f:3s} {0:>3d} {n:>4d} {n:>4d} {"-":18s} | {pct(sum(x["arm3_l1"] for x in bl), n):>14s} '
            f'{pct(sum(x["arm3_l2"] for x in bl), n):>14s} {"0.0":>24s} {"0.0":>8s} | '
            f'{pct(sum(x["arm2_l1"] for x in bl), n):>13s} {pct(sum(x["arm2_l2"] for x in bl), n):>13s} {"0.0":>6s} | '
            f'{sum(x["arm1_l1"] for x in bl):>4d}/{n:<4d} {sum(x["arm1_l2"] for x in bl):>4d}/{n:<4d} | '
            f'{sum(bool(x["arm3_flags"].get("far_from_target")) for x in bl):>3d} '
            f'{sum(bool(x["arm3_flags"].get("exclusion_broken")) for x in bl):>4d} '
            f'{np.median([x["target_distance_m"] for x in bl if x["target_distance_m"] is not None]):>9.2f}')
        curve[(f, 0)] = {'n': n, 'arm3_l1': sum(x['arm3_l1'] for x in bl) / n, 'arm3_l2': sum(x['arm3_l2'] for x in bl) / n,
                         'arm2_l1': sum(x['arm2_l1'] for x in bl) / n, 'arm1_l1': sum(x['arm1_l1'] for x in bl) / n,
                         'drop3': 0.0, 'drop3_l2': 0.0, 'drop2': 0.0}
        for lv in LEVELS[f]:
            pl = [e for e in planned if e['family'] == f and e['level'] == lv]
            vs = [row for row in V if row['family'] == f and row['level'] == lv]
            rej = collections.Counter(e['rejected_reason'] for e in pl if not e['accepted'])
            k = len(vs)
            d3 = 100 * np.mean([x['drop_arm3_l1'] for x in vs]) if vs else float('nan')
            d3l2 = 100 * np.mean([x['drop_arm3_l2'] for x in vs]) if vs else float('nan')
            d2 = 100 * np.mean([x['drop_arm2_l1'] for x in vs]) if vs else float('nan')
            ci = boot_ci([(tuple(x['layout']), 100 * x['drop_arm3_l1']) for x in vs], layouts)
            td = [x['target_distance_m'] for x in vs if x['target_distance_m'] is not None]
            say(f'  {f:3s} {lv:>3d} {len(pl):>4d} {k:>4d} {str(dict(rej)) if rej else "-":18s} | '
                f'{pct(sum(x["arm3_l1"] for x in vs), k):>14s} {pct(sum(x["arm3_l2"] for x in vs), k):>14s} '
                f'{d3:>6.1f} [{ci[0]:5.1f}, {ci[1]:5.1f}]      {d3l2:>8.1f} | '
                f'{pct(sum(x["arm2_l1"] for x in vs), k):>13s} {pct(sum(x["arm2_l2"] for x in vs), k):>13s} {d2:>6.1f} | '
                f'{sum(x["arm1_l1"] for x in vs):>4d}/{k:<4d} {sum(x["arm1_l2"] for x in vs):>4d}/{k:<4d} | '
                f'{sum(bool(x["arm3_flags"].get("far_from_target")) for x in vs):>3d} '
                f'{sum(bool(x["arm3_flags"].get("exclusion_broken")) for x in vs):>4d} '
                f'{np.median(td) if td else float("nan"):>9.2f}')
            curve[(f, lv)] = {'n': k, 'arm3_l1': sum(x['arm3_l1'] for x in vs) / k, 'arm3_l2': sum(x['arm3_l2'] for x in vs) / k,
                              'arm2_l1': sum(x['arm2_l1'] for x in vs) / k, 'arm1_l1': sum(x['arm1_l1'] for x in vs) / k,
                              'drop3': d3, 'drop3_l2': d3l2, 'drop2': d2, 'ci': ci,
                              'base_arm3_l1': sum(x['base_arm3_l1'] for x in vs) / k}
    say('  (far = far_from_target flags; exbr = exclusion_broken flags; tdist = solver place to target point, m)')
    say('')
    # family drops, pooled over levels
    say('Family drop (criterion (a)): mean over the family\'s accepted variants of [success(base) - success(v)] x 100, '
        'arm 3, levels and seeds pooled; layout-clustered bootstrap 95% CI (30 layouts, 2,000 draws, default_rng(0); '
        'descriptive).')
    fam_drop = {}
    for f in FAMILIES:
        vs = [x for x in V if x['family'] == f]
        for lab in ('l1', 'l2'):
            d = 100 * np.mean([x[f'drop_arm3_{lab}'] for x in vs])
            ci = boot_ci([(tuple(x['layout']), 100 * x[f'drop_arm3_{lab}']) for x in vs], layouts)
            fam_drop[(f, lab)] = (d, ci, len(vs))
        d2 = 100 * np.mean([x['drop_arm2_l1'] for x in vs])
        say(f'  {f}: n {len(vs)}; arm 3 L1 drop {fam_drop[(f, "l1")][0]:.1f} [{fam_drop[(f, "l1")][1][0]:.1f}, '
            f'{fam_drop[(f, "l1")][1][1]:.1f}]; L2 drop {fam_drop[(f, "l2")][0]:.1f}; arm 2 L1 drop {d2:.1f}; '
            f'arm 3 L1 success {pct(sum(x["arm3_l1"] for x in vs), len(vs))} vs paired bases '
            f'{pct(sum(x["base_arm3_l1"] for x in vs), len(vs))}')
    say('')
    # by situation
    say('By situation (arm 3, seeds pooled): success L1 / L2 and paired L1 drop, per family and level')
    say(f'  {"fam":3s} {"lvl":>3s} | ' + ' | '.join(f'{SIT_SHORT[s]:>5s}: {"n":>2s} {"L1":>5s} {"L2":>5s} {"drop":>5s}' for s in SITS))
    by_sit = {}
    for f in FAMILIES:
        for lv in (0,) + LEVELS[f]:
            parts = []
            for s in SITS:
                if lv == 0:
                    xs = [x for x in B if x['situation'] == s and fam_applies(f, x['row'])]
                else:
                    xs = [x for x in V if x['situation'] == s and x['family'] == f and x['level'] == lv]
                n = len(xs)
                if n:
                    l1 = 100 * np.mean([x['arm3_l1'] for x in xs])
                    l2 = 100 * np.mean([x['arm3_l2'] for x in xs])
                    dr = 100 * np.mean([x['drop_arm3_l1'] for x in xs])
                    parts.append(f'{SIT_SHORT[s]:>5s}: {n:>2d} {l1:>5.0f} {l2:>5.0f} {dr:>5.0f}')
                    by_sit[(f, lv, s)] = (n, l1, l2, dr)
                else:
                    parts.append(f'{SIT_SHORT[s]:>5s}: {0:>2d} {"-":>5s} {"-":>5s} {"-":>5s}')
            say(f'  {f:3s} {lv:>3d} | ' + ' | '.join(parts))
    say('  (rows a family never reaches at a level are empty: D5 exists only at conversation groups, B/C have no line_6 '
        'or audience_5 variant, A1/A2 no line_6; build/counts_by_row.txt)')
    say('')
    # per base layout
    say('Per base layout (arm 3 L1; base, then each kept variant: family+level = 1 success / 0 failure):')
    for lay in layouts:
        b = next(x for x in B if (x['seed'], x['row']) == lay)
        vs = sorted([x for x in V if (x['seed'], x['row']) == lay], key=lambda x: (x['family'], x['level']))
        say(f'  s{lay[0]} eval_{lay[1]:02d} {b["situation"][:5]:5s}: base {int(b["arm3_l1"])} | ' +
            ' '.join(f'{x["family"]}{x["level"]}={int(x["arm3_l1"])}' for x in vs))
    say('')
    # L1/L2
    say('L1/L2 disagreement (criterion (c)): an episode that succeeds under L1 and fails under L2; per scene (= episode).')
    dis = {}
    for arm in ('arm1', 'arm2', 'arm3'):
        k = sum(x[f'{arm}_l1'] and not x[f'{arm}_l2'] for x in V)
        kb = sum(x[f'{arm}_l1'] and not x[f'{arm}_l2'] for x in B)
        dis[arm] = (k, len(V))
        say(f'  {arm}: variants {pct(k, len(V))}; bases {pct(kb, len(B))}')
    for f in FAMILIES:
        vs = [x for x in V if x['family'] == f]
        k = sum(x['arm3_l1'] and not x['arm3_l2'] for x in vs)
        say(f'    arm 3, family {f}: {pct(k, len(vs))}')
    for x in V:
        if x['arm3_l1'] and not x['arm3_l2']:
            say(f'    disagreement: {x["scene_id"]}: {x["arm3_l2_fail"]}')
    l2e = sum(x['l2_empty'] for x in V)
    say(f'  L2-empty (variants): {pct(l2e, len(V))}; bases {pct(sum(x["l2_empty"] for x in B), len(B))}')
    say('')

    # ------------------------------------------------------------------------------------------- 2. failures
    say('== 2. The documented failures (criterion (b)), on arm 3; fires = detector true in the variant and false in its '
        'base; relevant = precondition holds and the base does not fire (FROZEN section 7) ==')
    names = {'i': '(i) bystanders behind a line absorbed as waiting',
             'ii': '(ii) "a crowd keeps the robot out"',
             'iii': '(iii) large-circle off-facing (>= 110 deg)',
             'iv': '(iv) the hindmost-watcher rule dropped silently',
             'v': '(v) named cause: the region\'s floor not standable (not one of the four)'}
    rates = {}
    for k in ('i', 'ii', 'iii', 'iv', 'v'):
        rel = [x for x in V if x['relevant'][k]]
        fire = [x for x in rel if x['fires'][k]]
        rates[k] = (len(fire), len(rel))
        say(f'  {names[k]}: fires in {pct(len(fire), len(rel))} relevant variants')
        for f in FAMILIES:
            rf = [x for x in rel if x['family'] == f]
            ff = [x for x in rf if x['fires'][k]]
            if rf:
                by_lv = collections.Counter(x['level'] for x in ff)
                say(f'      {f}: {pct(len(ff), len(rf))}' + (f'  by level {dict(sorted(by_lv.items()))}' if ff else ''))
        ex = fire[:4]
        for x in ex:
            if k == 'i':
                detail = (f'joined {x["joined_waiting"]} (sources {x["joined_sources"]}), place '
                          f'{x["target_distance_m"]} m from the target, L1 {x["arm3_l1"]}')
            elif k == 'ii':
                detail = (f'F cells {x["F_cells"]} (base {x["base_F_cells"]}), smallest covering pair '
                          f'{x["min_pair_covering_m"]} m, place {x["target_distance_m"]} m from the target, L1 {x["arm3_l1"]}')
            elif k == 'iii':
                detail = f'off-facing {x["off_facing_deg"]} deg (base {x["base_off_facing_deg"]}), L1 {x["arm3_l1"]}'
            elif k == 'iv':
                detail = (f'reason "{x["not_ahead_reason"]}", base not_ahead_of {x["base_not_ahead_of"]}, ahead of the '
                          f'hindmost member {x["ahead_of_hindmost_m"]} m, L1 {x["arm3_l1"]}')
            else:
                detail = f'L1 cells {x["l1_cells_e0"]}, standable 0, L1 {x["arm3_l1"]}'
            say(f'      e.g. {x["scene_id"]}: {detail}')
        if k == 'iv':
            nf = [x for x in rel if not x['fires'][k]]
            reasons = collections.Counter(str(x['not_ahead_reason']) if not x['not_ahead_of'] else 'applied' for x in nf)
            say(f'      relevant, not firing: {dict(reasons)}')
    met_b = [k for k in ('i', 'ii', 'iii', 'iv') if rates[k][1] and rates[k][0] / rates[k][1] >= 0.10]
    say(f'  criterion (b) counts the four documented failures (i)-(iv): firing in >= 10% of relevant variants: '
        f'{met_b or "none"} -> {"MET" if len(met_b) >= 2 else "NOT MET"}')
    fmin = min(x['F_cells'] for x in V)
    fratio = min(x['F_cells'] / x['base_F_cells'] for x in V if x['base_F_cells'])
    offs = [x['off_facing_deg'] for x in V if x['relevant']['iii'] and x['off_facing_deg'] is not None]
    boffs = [x['base_off_facing_deg'] for x in B if x['base_off_facing_deg'] is not None]
    say('  POST HOC (descriptive) why (ii)-(iv) do not fire: (ii) F (L1 cells >= 1.0 m from every visible person) never '
        f'empties: smallest F in a variant {fmin} cells, smallest variant/base ratio {fratio:.2f} (n {len(V)}); the '
        'builder keeps every added person >= 1.2 m from the members and 0.3 m outside the structure\'s space (c8). '
        f'(iii) off-facing in relevant group variants: max {max(offs):.1f} deg (n {len(offs)}); in group bases max '
        f'{max(boffs):.1f} deg (n {len(boffs)}); the families keep the members, so circles stay as built. (iv) the '
        f'rule was applied in all {rates["iv"][1]} relevant watching variants (audience rows are mostly rejected: '
        'B, C, E and D5 kept 0-2 per audience row).')
    pm = [x for x in rows if not x['people_matter']]
    say(f'  Note: the reference context check (arm 3) finds that people do not matter, so the standard goal is the '
        f'result, in {len(pm)} scenes: ' + ', '.join(sorted({f"s{x['seed']} eval_{x['row']:02d}" for x in pm})) +
        ' (bases and their variants).')
    say('  Arm 2 (solver alone; descriptive), fires over all variants: ' + ', '.join(
        f'{k} {sum(x["arm2_fires"][k] for x in V)}/{len(V)}' for k in ('ii', 'iii', 'v')))
    say('')
    # failure explanation
    say('Arm-3 L1 failures by named cause (variant fails, base succeeds or fails; cause = any detector firing):')
    for f in FAMILIES:
        for lv in LEVELS[f]:
            vs = [x for x in V if x['family'] == f and x['level'] == lv and not x['arm3_l1']]
            if not vs:
                continue
            causes = collections.Counter('+'.join(k for k in ('i', 'ii', 'iii', 'iv', 'v') if x['fires'][k]) or 'none'
                                         for x in vs)
            say(f'  {f}{lv}: {len(vs)} failures; causes {dict(causes)}')
            for x in vs:
                if not x['named_cause']:
                    say(f'      unnamed: {x["scene_id"]}: base L1 {x["base_arm3_l1"]}, in_region {x["arm3_in_region"]}, '
                        f'heading_ok {x["arm3_heading_ok"]}, reachable {x["arm3_reachable"]}, keys {x["keys"]}, '
                        f'joined {x["joined_waiting"]} {x["joined_sources"]}, not_ahead {x["not_ahead_of"]} '
                        f'({x["not_ahead_reason"]}), F {x["F_cells"]}, standable {x["l1_cells_standable"]}/{x["l1_cells_e0"]}')
    say('  POST HOC (added after the results; descriptive): the unnamed failures are E2 at lines. The E2 person, 2 m '
        'from the camera on the line of sight to the representative point, hides the floor round the place in turn; '
        'detector (v) needs every L1 cell unstandable and does not fire. Standable L1 cells on the solver\'s floor, '
        'variant vs base:')
    for x in V:
        if not x['arm3_l1'] and not x['named_cause']:
            b = next(y for y in B if y['seed'] == x['seed'] and y['scene_id'] == x['base_scene_id'])
            say(f'      {x["scene_id"]}: {x["l1_cells_standable"]}/{x["l1_cells_e0"]} (base {b["l1_cells_standable"]}/'
                f'{b["l1_cells_e0"]}); place {x["keys"]["turn_m"] if x["keys"] else None} m from the place in turn, '
                f'outside the region, heading not within tolerance: {not x["arm3_heading_ok"]}')
    eok = [x for x in V if x['family'] == 'E' and x['situation'] == 'waiting_line']
    say(f'      all E variants at lines (n {len(eok)}): standable L1 cells variant/base median '
        f'{np.median([x["l1_cells_standable"] / max(1, next(y for y in B if y["seed"] == x["seed"] and y["scene_id"] == x["base_scene_id"])["l1_cells_standable"]) for x in eok]):.2f}')
    say('')

    # ------------------------------------------------------------------------------------------- (a)
    def criterion_a(Vsel, fams=FAMILIES, label=''):
        drops, ns = {}, {}
        for f in fams:
            vs = [x for x in Vsel if x['family'] == f]
            if vs:
                drops[f] = float(100 * np.mean([x['drop_arm3_l1'] for x in vs]))
                ns[f] = len(vs)
        big = [f for f, d in drops.items() if d >= 15]
        mono = {}
        for f in ('A', 'D'):
            if f not in fams:
                continue
            lvls = [lv for lv in LEVELS[f] if any(x['family'] == f and x['level'] == lv for x in Vsel)]
            if not lvls:
                continue
            per = [float(100 * np.mean([x['drop_arm3_l1'] for x in Vsel if x['family'] == f and x['level'] == lv])) for lv in lvls]
            nondec = per[0] >= 0 and all(b >= a for a, b in zip(per, per[1:]))
            top = lvls[-1]
            tv = [x for x in Vsel if x['family'] == f and x['level'] == top]
            fails = [x for x in tv if not x['arm3_l1']]
            named = sum(x['named_cause'] for x in fails)
            ok = nondec and per[-1] >= 10 and (len(fails) > 0 and named / len(fails) >= 0.5)
            mono[f] = {'levels': lvls, 'per_level_drop': [round(p, 1) for p in per], 'nondecreasing': nondec,
                       'top_level': top, 'top_drop': round(per[-1], 1), 'top_failures': len(fails),
                       'top_named': named, 'met': ok}
        met = len(big) >= 2 or any(m['met'] for m in mono.values())
        return {'drops': {f: round(d, 1) for f, d in drops.items()}, 'n': ns, 'families_15': big, 'monotone': mono, 'met': met}

    crit_a = criterion_a(V)
    crit_a_noA56 = criterion_a([x for x in V if not (x['family'] == 'A' and x['level'] in (5, 6))])
    crit_a_noA = criterion_a([x for x in V if x['family'] != 'A'], fams=('B', 'C', 'D', 'E'))
    # (b) sensitivity
    def crit_b(Vsel):
        rr = {}
        for k in ('i', 'ii', 'iii', 'iv'):
            rel = [x for x in Vsel if x['relevant'][k]]
            rr[k] = (sum(x['fires'][k] for x in rel), len(rel))
        met = [k for k, (a, b) in rr.items() if b and a / b >= 0.10]
        return rr, met
    b_noA56 = crit_b([x for x in V if not (x['family'] == 'A' and x['level'] in (5, 6))])
    b_noA = crit_b([x for x in V if x['family'] != 'A'])

    # ------------------------------------------------------------------------------------------- 3. noise
    say('== 3. Perception-noise model (fit on development runs only; in-sample; i.i.d. per person and answer) ==')
    say('  Fit parameters: score/noise/fit.txt (summarised in score/noise_model.json). The fit report is appended at the '
        'end of this file.')
    nm = json.loads((L.SCORE / 'noise_model.json').read_text())
    val = {}
    valid = []
    vrows_all = []
    for key, label in MODELS:
        p = L.SCORE / 'noise' / f'validation_{key}.jsonl'
        if not p.exists():
            say(f'  {label}: validation missing')
            continue
        vr = [json.loads(l) for l in p.read_text().splitlines() if l]
        vrows_all += vr
        n = len(vr)
        rec_c = 100 * np.mean([x['rescore'] for x in vr])
        rec_raw = 100 * np.mean([x['recorded'] for x in vr])
        sim_c = 100 * np.mean([x['sim_mean'] for x in vr])
        r5_c = 100 * np.mean([x['r5'] for x in vr])
        mism = sum(x['recorded'] != x['rescore'] for x in vr)
        steps = sum(x['solver_step'] for x in vr)
        gap = sim_c - rec_c
        ok = abs(gap) <= 10
        if ok:
            valid.append(key)
        val[key] = {'n_records': n, 'solver_steps': steps, 'recorded_C': rec_c, 'simulated_C': sim_c, 'R5': r5_c,
                    'gap': gap, 'valid': ok, 'recorded_vs_rescore_mismatch': mism}
        say(f'  {label}: records {n} (solver steps {steps}); recorded C (rescore) {rec_c:.1f}% (as recorded {rec_raw:.1f}%; '
            f'{mism} records differ); simulated C {sim_c:.1f}%; gap {gap:+.1f} points -> '
            f'{"valid (|gap| <= 10)" if ok else "MISSES (|gap| > 10): no noisy curve"}')
        say(f'      diagnostic: R5 (actual people + reference constraints) {r5_c:.1f}%; simulated - R5 {sim_c - r5_c:+.1f}; '
            f'R5 - recorded {r5_c - rec_c:+.1f}')
        for dom in ('realistic', 'placeholder'):
            xs = [x for x in vr if x['domain'] == dom]
            say(f'      [{dom}] n {len(xs)}: recorded {100 * np.mean([x["rescore"] for x in xs]):.1f}, simulated '
                f'{100 * np.mean([x["sim_mean"] for x in xs]):.1f}, R5 {100 * np.mean([x["r5"] for x in xs]):.1f}, gap '
                f'{100 * (np.mean([x["sim_mean"] for x in xs]) - np.mean([x["rescore"] for x in xs])):+.1f}')
        for s in ('no_one', 'waiting_line', 'conversation_group', 'audience'):
            xs = [x for x in vr if x['situation'] == s]
            if xs:
                say(f'      [{s}] n {len(xs)}: recorded {100 * np.mean([x["rescore"] for x in xs]):.1f}, simulated '
                    f'{100 * np.mean([x["sim_mean"] for x in xs]):.1f}, R5 {100 * np.mean([x["r5"] for x in xs]):.1f}, '
                    f'gap {100 * (np.mean([x["sim_mean"] for x in xs]) - np.mean([x["rescore"] for x in xs])):+.1f}')
    with open(L.SCORE / 'rows_validation.jsonl', 'w') as fh:
        for x in vrows_all:
            fh.write(json.dumps(x) + '\n')
    # I01 cross-check of R5 on the realistic arms
    i01 = {}
    for key, fname in (('qwen', 'qwen38_low'), ('glm', 'glm53'), ('dsv', 'dsv41')):
        p = L.RES / 'phase1/i01_repairs' / f'rows_old_{fname}.jsonl'
        for l in p.read_text().splitlines():
            if l:
                r = json.loads(l)
                i01[(key, r['run'], r['scene'], r['episode'], r['trial'])] = r['R5']
    agree = tot = 0
    for x in vrows_all:
        k = (x['model'], x['run'], x['scene'], x['episode'], x['trial'])
        if k in i01:
            tot += 1
            agree += i01[k] == x['r5']
    say(f'  R5 cross-check against phase1/i01_repairs rows_old_* (realistic arms): {agree}/{tot} records agree')
    k4 = len(val) == 3 and not valid
    say(f'  K4 reading (FROZEN section 8): a model with |gap| > 10 has no noisy curve; the kill fires when every model '
        f'misses. Valid models: {[dict(MODELS)[k] for k in valid] or "none"} -> K4 {"FIRES" if k4 else "does not fire"}')
    say('')

    # noisy curves
    noisy = {}
    for key, label in MODELS:
        rr = []
        for s in (0, 1, 2):
            p = L.SCORE / 'work' / f'noisy_models_{key}_seed{s}.jsonl'
            if p.exists():
                rr += [json.loads(l) for l in p.read_text().splitlines() if l]
        if rr:
            noisy[key] = {f"{r['seed']}/{r['scene_id']}": r for r in rr}
    sep = {}
    if noisy:
        say('Noisy curves (arm 4; mean over 20 draws per scene, then over scenes; L1). Only models with a valid noise model '
            'are reported (K4 reading).')
        for key, label in MODELS:
            if key not in noisy or key not in valid:
                continue
            nz = noisy[key]
            missing = [x['key'] for x in rows if x['key'] not in nz]
            agree_all = all(nz[s]['agree_all'] for s in nz)
            say(f'  {label}: scenes {len(nz)} (missing {len(missing)}); both episodes agree in every draw: {agree_all}')
            with open(L.SCORE / f'rows_noisy_{key}.jsonl', 'w') as fh:
                for s, r in nz.items():
                    o = R[s]
                    fh.write(json.dumps({'scene_id': r['scene_id'], 'family': r['family'], 'level': r['level'], 'kind': r['kind'],
                                         'seed': r['seed'], 'row': r['row'], 'situation': o['situation'],
                                         'arm3_l1': o['arm3_l1'], 'arm3_l2': o['arm3_l2'], 'arm4_l1_mean': r['l1_mean'],
                                         'arm4_l2_mean': r['l2_mean'],
                                         'arm4_det_rates': {k: float(np.mean([d['det'][kk] for d in r['draws']]))
                                                            for k, kk in (('i', 'det_i'), ('iii', 'det_iii'),
                                                                          ('iv', 'det_iv_core'), ('v', 'det_v'))},
                                         'draws_l1': [d['e0']['l1'] for d in r['draws']]}) + '\n')
            say(f'    {"fam":3s} {"lvl":>3s} {"n":>3s} | {"arm3 L1":>7s} {"arm4 L1":>7s} {"sep":>6s} [95% CI]       | '
                f'{"arm4 L2":>7s} {"dis4":>5s}')
            for f in FAMILIES:
                for lv in (0,) + LEVELS[f]:
                    if lv == 0:
                        xs = [x for x in B if fam_applies(f, x['row'])]
                    else:
                        xs = [x for x in V if x['family'] == f and x['level'] == lv]
                    xs = [x for x in xs if x['key'] in nz]
                    if not xs:
                        continue
                    a3 = 100 * np.mean([x['arm3_l1'] for x in xs])
                    a4 = 100 * np.mean([nz[x['key']]['l1_mean'] for x in xs])
                    a4l2 = 100 * np.mean([nz[x['key']]['l2_mean'] for x in xs])
                    dis4 = 100 * np.mean([np.mean([d['e0']['l1'] and not d['e0']['l2'] for d in nz[x['key']]['draws']])
                                          for x in xs])
                    ci = boot_ci([(tuple(x['layout']), 100 * (x['arm3_l1'] - nz[x['key']]['l1_mean'])) for x in xs],
                                 layouts)
                    sep[(key, f, lv)] = (a3 - a4, len(xs))
                    say(f'    {f:3s} {lv:>3d} {len(xs):>3d} | {a3:>7.1f} {a4:>7.1f} {a3 - a4:>+6.1f} [{ci[0]:+5.1f}, {ci[1]:+5.1f}] | '
                        f'{a4l2:>7.1f} {dis4:>5.1f}')
            topv = [x for x in V if x['level'] == TOP[x['family']] and x['key'] in nz]
            a3 = 100 * np.mean([x['arm3_l1'] for x in topv])
            a4 = 100 * np.mean([nz[x['key']]['l1_mean'] for x in topv])
            ci = boot_ci([(tuple(x['layout']), 100 * (x['arm3_l1'] - nz[x['key']]['l1_mean'])) for x in topv], layouts)
            bb = [x for x in B if x['key'] in nz]
            b3 = 100 * np.mean([x['arm3_l1'] for x in bb])
            b4 = 100 * np.mean([nz[x['key']]['l1_mean'] for x in bb])
            sep[(key, 'top')] = (a3 - a4, len(topv))
            sep[(key, 'bases')] = (b3 - b4, len(bb))
            say(f'    top level pooled (A6, B1, C1, D5, E2): n {len(topv)}; arm 3 {a3:.1f}, arm 4 {a4:.1f}, separation '
                f'{a3 - a4:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}] points; bases n {len(bb)}: arm 3 {b3:.1f}, arm 4 {b4:.1f}, '
                f'separation {b3 - b4:+.1f}')
            a4dis = 100 * np.mean([np.mean([d['e0']['l1'] and not d['e0']['l2'] for d in nz[x['key']]['draws']])
                                   for x in V if x['key'] in nz])
            say(f'    arm 4 L1/L2 disagreement over variants: {a4dis:.1f}% (mean over draws)')
            dets = {k: 100 * np.mean([np.mean([d['det'][kk] for d in nz[x['key']]['draws']]) for x in V if x['key'] in nz])
                    for k, kk in (('i', 'det_i'), ('iii', 'det_iii'), ('iv', 'det_iv_core'), ('v', 'det_v'))}
            say('    arm 4 detectors true (descriptive, not differenced; % of variant draws): ' +
                ', '.join(f'{k} {v:.1f}' for k, v in dets.items()))
        say('')
        # secondary arm
        sec = {}
        for key, label in MODELS:
            rr = []
            for s in (0, 1, 2):
                p = L.SCORE / 'work' / f'noisy_placeholder_only_{key}_seed{s}_top.jsonl'
                if p.exists():
                    rr += [json.loads(l) for l in p.read_text().splitlines() if l]
            if rr:
                sec[key] = {f"{r['seed']}/{r['scene_id']}": r for r in rr}
        if sec:
            say('Secondary arm (descriptive, decides nothing): rates fitted on the placeholder answers alone, on the '
                'top-level variants and the bases (L1)')
            for key, label in MODELS:
                if key not in sec:
                    continue
                nz = sec[key]
                topv = [x for x in V if x['level'] == TOP[x['family']] and x['key'] in nz]
                bb = [x for x in B if x['key'] in nz]
                a3 = 100 * np.mean([x['arm3_l1'] for x in topv])
                a4 = 100 * np.mean([nz[x['key']]['l1_mean'] for x in topv])
                b3 = 100 * np.mean([x['arm3_l1'] for x in bb])
                b4 = 100 * np.mean([nz[x['key']]['l1_mean'] for x in bb])
                say(f'  {label}: top n {len(topv)}: arm 3 {a3:.1f}, arm 4 {a4:.1f}, separation {a3 - a4:+.1f}; bases n '
                    f'{len(bb)}: arm 3 {b3:.1f}, arm 4 {b4:.1f}, separation {b3 - b4:+.1f}')
            say('')

    # ------------------------------------------------------------------------------------------- 4. criteria
    say('== 4. Criteria (a)-(e) and the kill lines (REPORT I06, verbatim; readings fixed in FROZEN section 9) ==')
    say(f'(a) "Oracle C drops at least 15 points in at least 2 families, or falls monotonically along family D or A '
        f'with named causes."')
    def show_a(c, tag):
        say(f'    {tag}: family drops (arm 3 L1, n): ' + ', '.join(
            f'{f} {d:.1f} (n {c["n"][f]})' for f, d in c['drops'].items()) +
            f'; families >= 15: {c["families_15"]}')
        for f, m in c['monotone'].items():
            say(f'      monotone {f}: levels {m["levels"]} per-level drop {m["per_level_drop"]}, non-decreasing '
                f'{m["nondecreasing"]}, top {f}{m["top_level"]} drop {m["top_drop"]} (>= 10), named causes '
                f'{m["top_named"]}/{m["top_failures"]} failures (>= 50%) -> {"met" if m["met"] else "not met"}')
        say(f'      -> {"MET" if c["met"] else "NOT MET"}')
    show_a(crit_a, 'all families and levels')
    show_a(crit_a_noA56, 'sensitivity, without A5-A6')
    show_a(crit_a_noA, 'sensitivity, without family A')
    say(f'(b) "At least 2 of the 4 documented failures fire in at least 10% of relevant variants."')
    say('    ' + ', '.join(f'({k}) {pct(*rates[k])}' for k in ('i', 'ii', 'iii', 'iv')) +
        f' -> {"MET" if len(met_b) >= 2 else "NOT MET"} ({met_b})')
    say('    sensitivity without A5-A6: ' + ', '.join(f'({k}) {pct(*v)}' for k, v in b_noA56[0].items()) +
        f' -> {"met" if len(b_noA56[1]) >= 2 else "not met"}')
    say('    sensitivity without family A: ' + ', '.join(f'({k}) {pct(*v)}' for k, v in b_noA[0].items()) +
        f' -> {"met" if len(b_noA[1]) >= 2 else "not met"}')
    d3k, d3n = dis['arm3']
    dshare = d3k / d3n
    l2share = l2e / len(V)
    c_level = 'MET (<= 5%)' if dshare <= 0.05 else ('PASSES WITH A DECISION (5-20%)' if dshare <= 0.20 else 'NOT MET (> 20%)')
    c_met = dshare <= 0.20 and l2share <= 0.02
    say(f'(c) "L1 and L2 verdicts differ in at most 5% of episodes. Between 5% and 20% passes, with side-by-side reporting '
        f'and a decision. L2 empty in at most 2%."')
    say(f'    arm 3 disagreement {pct(d3k, d3n)} -> {c_level}; L2-empty {pct(l2e, len(V))} (<= 2%) -> '
        f'{"MET" if c_met else "NOT MET"}' + ('' if dshare <= 0.05 else ' (decision needed)'))
    say('    disclosed: for the solver arms agreement is partly built in (standoff 1.0 m > L2\'s 0.85 m centre distance)')
    pass_rate = sum(e['accepted'] for e in planned) / len(planned)
    d_met = pass_rate >= 0.70 and a1v == len(V) and a1b == len(B)
    say(f'(d) "At least 70% of variants pass check_scene, and full oracle scores 100%."')
    say(f'    pass rate {pct(sum(e["accepted"] for e in planned), len(planned))}; first attempt '
        f'{pct(sum(e["accepted"] and e["attempts"] == 1 for e in planned), len(planned))}; full oracle L1 variants '
        f'{pct(a1v, len(V))}, bases {pct(a1b, len(B))} -> {"MET" if d_met else "NOT MET"}')
    say(f'(e) "Noisy perception separates from oracle by at least 10 points at the top level."')
    e_models = [k for k in valid if (k, 'top') in sep and sep[(k, 'top')][0] >= 10]
    e_met = len(valid) >= 2 and len(e_models) >= 2
    for k in valid:
        if (k, 'top') in sep:
            say(f'    {dict(MODELS)[k]}: separation at the top level {sep[(k, "top")][0]:+.1f} points (n {sep[(k, "top")][1]}); '
                f'bases {sep[(k, "bases")][0]:+.1f} (n {sep[(k, "bases")][1]})')
    say(f'    valid models {len(valid)}; models >= 10 points: {[dict(MODELS)[k] for k in e_models]} -> '
        f'{"MET" if e_met else "NOT MET"}')
    for k in valid:
        nz = noisy.get(k, {})
        nonA = [x for x in V if x['family'] != 'A' and x['level'] == TOP[x['family']] and x['key'] in nz]
        if nonA:
            a3 = 100 * np.mean([x['arm3_l1'] for x in nonA])
            a4 = 100 * np.mean([nz[x['key']]['l1_mean'] for x in nonA])
            say(f'    POST HOC (descriptive): {dict(MODELS)[k]} top level without A6 (B1, C1, D5, E2; n {len(nonA)}): '
                f'arm 3 {a3:.1f}, arm 4 {a4:.1f}, separation {a3 - a4:+.1f}. At A6 oracle C is 0%, so noise can only '
                'raise success there (a missed or displaced confuser is not absorbed).')
    say('Kill lines:')
    cells = [(f, lv) for f in FAMILIES for lv in LEVELS[f]] + [(f, 0) for f in FAMILIES]
    min_cell = min(curve[c]['arm3_l1'] for c in cells)
    k1_first = all(curve[c]['arm3_l1'] >= 0.95 for c in cells)
    k1_second = all(abs(sep.get((k, f, TOP[f]), (0, 0))[0]) <= 5 for k in valid for f in FAMILIES) if valid else True
    k1 = k1_first and k1_second
    say(f'  K1 "Oracle C at least 95% everywhere, and noise within 5 points.": lowest arm-3 success over (family, level) '
        f'{100 * min_cell:.1f}% -> first half {"holds" if k1_first else "fails"}; -> K1 {"FIRES" if k1 else "does not fire"}')
    k2 = dshare > 0.30 or l2share > 0.10
    say(f'  K2 "L1/L2 disagreement above 30%, or L2 empty above 10%.": {100 * dshare:.1f}%, {100 * l2share:.1f}% -> '
        f'{"FIRES" if k2 else "does not fire"}')
    rej = 1 - pass_rate
    k3 = rej > 0.5
    say(f'  K3 "More than 50% of layouts rejected." (read as variants): {100 * rej:.1f}% -> {"FIRES" if k3 else "does not fire"}')
    say(f'  K4 "The noise model misses the recorded C by more than 10 points. Then report oracle curves only.": '
        + ', '.join(f'{dict(MODELS)[k]} gap {v["gap"]:+.1f}' for k, v in val.items()) +
        f' -> {"FIRES" if k4 else "does not fire"} (fires only when every model misses)')
    kills = {'K1': k1, 'K2': k2, 'K3': k3, 'K4': k4}
    crits = {'a': crit_a['met'], 'b': len(met_b) >= 2, 'c': c_met, 'd': d_met, 'e': e_met}
    if any(kills.values()):
        verdict = 'kill'
    elif all(crits.values()):
        verdict = 'success'
    else:
        verdict = 'neither'
    say('')
    say(f'VERDICT: criteria {crits}; kills {kills} -> '
        + {'success': 'BUILD THE FULL PASS (all of (a)-(e) hold, no kill fires)',
           'kill': 'KILL (a kill line fires)',
           'neither': 'NEITHER: no kill fires, but not all of (a)-(e) hold outright; the project decides'}[verdict])
    if all(crits.values()) and 0.05 < dshare <= 0.20 and not any(kills.values()):
        say('  (c) holds only in its 5-20% band, which "passes, with side-by-side reporting and a decision": the '
            'full pass is licensed pending that decision.')
    say('')

    # ------------------------------------------------------------------------------------------- 5. C5
    say('== 5. C5 gate-1 recount (FROZEN section 10; full table in score/c5_recount.txt) ==')
    c5p = L.SCORE / 'c5_recount.txt'
    if c5p.exists():
        for line in c5p.read_text().splitlines()[:7]:
            say('  ' + line)
    say('')
    say('== Appendix: noise fit report (score/noise/fit.txt) ==')
    fitp = L.SCORE / 'noise' / 'fit.txt'
    if fitp.exists():
        out.extend(fitp.read_text().splitlines())
    (L.SCORE / 'tables.txt').write_text('\n'.join(out) + '\n')
    summary = {'verdict': verdict, 'criteria': crits, 'kills': kills, 'crit_a': crit_a, 'crit_a_noA56': crit_a_noA56,
               'crit_a_noA': crit_a_noA, 'rates': rates, 'met_b': met_b, 'b_noA56': b_noA56, 'b_noA': b_noA,
               'disagreement_arm3': [d3k, d3n], 'disagreement': dis, 'l2_empty': [l2e, len(V)], 'pass_rate': pass_rate,
               'full_oracle': [a1v, len(V), a1b, len(B)], 'validation': val, 'valid_models': valid,
               'separation': {f'{k[0]}|{k[1]}' + (f'|{k[2]}' if len(k) > 2 else ''): v for k, v in sep.items()},
               'fam_drop': {f'{f}|{lab}': [d, list(ci), n] for (f, lab), (d, ci, n) in fam_drop.items()},
               'curve': {f'{f}|{lv}': v for (f, lv), v in curve.items()}, 'checks': dict(checks),
               'min_cell_arm3': min_cell, 'i01_r5_agree': [agree, tot]}
    (L.SCORE / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    print('\n'.join(out[:400]))
    figures(V, B, curve, noisy, valid, layouts)


def figures(V, B, curve, noisy, valid, layouts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), sharey=True)
    for ax, f in zip(axes, FAMILIES):
        lv = (0,) + LEVELS[f]
        xs = list(range(len(lv)))
        for key, lab, st in (('arm3_l1', 'oracle C, L1', '-o'), ('arm3_l2', 'oracle C, L2', '--s'),
                             ('arm2_l1', 'solver alone, L1', ':^'), ('arm1_l1', 'full oracle, L1', '-.d')):
            ax.plot(xs, [100 * curve[(f, l)][key] for l in lv], st, label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels([f'{f}{l}' if l else 'base' for l in lv])
        ax.set_title(f'family {f} (n per level: ' + ', '.join(str(curve[(f, l)]['n']) for l in lv) + ')', fontsize=9)
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('success, % of scenes')
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('I06 oracle curves (seeds 0-2 pooled; zero calls). Labels are the project\'s own rules.', fontsize=10)
    fig.tight_layout()
    fig.savefig(L.SCORE / 'curves_oracle.png', dpi=110)
    plt.close(fig)
    # by situation
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, s in zip(axes, SITS):
        for f in FAMILIES:
            lv = (0,) + LEVELS[f]
            ys, xs = [], []
            for i, l in enumerate(lv):
                pool = [x for x in (B if l == 0 else V) if x['situation'] == s and (l == 0 and fam_applies(f, x['row'])
                                                                                    or l and x['family'] == f and x['level'] == l)]
                if pool:
                    xs.append(i)
                    ys.append(100 * np.mean([x['arm3_l1'] for x in pool]))
            if xs:
                ax.plot(xs, ys, '-o', label=f)
        ax.set_title(f'{s} (arm 3, L1)', fontsize=9)
        ax.set_xlabel('level index (0 = base; A: A1..A6; D: D2, D5; E: E1, E2)')
        ax.grid(alpha=0.3)
        ax.set_ylim(-5, 105)
    axes[0].set_ylabel('success, % of scenes')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(L.SCORE / 'curves_by_situation.png', dpi=110)
    plt.close(fig)
    if noisy and valid:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), sharey=True)
        for ax, f in zip(axes, FAMILIES):
            lv = (0,) + LEVELS[f]
            xs = list(range(len(lv)))
            ax.plot(xs, [100 * curve[(f, l)]['arm3_l1'] for l in lv], '-ok', label='oracle C (arm 3)')
            for key, lab in MODELS:
                if key not in valid or key not in noisy:
                    continue
                nz = noisy[key]
                ys = []
                for l in lv:
                    pool = [x for x in (B if l == 0 else V) if (l == 0 and fam_applies(f, x['row'])) or
                            (l and x['family'] == f and x['level'] == l)]
                    ys.append(100 * np.mean([nz[x['key']]['l1_mean'] for x in pool if x['key'] in nz]))
                ax.plot(xs, ys, '--s', label=f'noisy C, {lab}')
            ax.set_xticks(xs)
            ax.set_xticklabels([f'{f}{l}' if l else 'base' for l in lv])
            ax.set_title(f'family {f}', fontsize=9)
            ax.set_ylim(-5, 105)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel('success under L1, % (arm 4: mean of 20 draws)')
        axes[0].legend(fontsize=8, loc='lower left')
        fig.suptitle('I06 noisy curves (simulated people, fitted in-sample on development runs; i.i.d. noise)', fontsize=10)
        fig.tight_layout()
        fig.savefig(L.SCORE / 'curves_noisy.png', dpi=110)
        plt.close(fig)


if __name__ == '__main__':
    main()
