"""Build out_primary.txt, out_extras.txt, tables.txt and rows.jsonl from run_eval.py's outputs (FROZEN.md
section 11). Pure bookkeeping: no scene or run is read here except the original rerun/lifting outputs for the
reproduction comparison."""
import collections
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
RD = D.parents[1]
GD = ('glm53', 'dsv41')
MODELS3 = ('qwen38_low', 'glm53', 'dsv41')
FIXES = ('oracle', 'V1', 'V2', 'V3', 'V4', 'front')
CLASSES = ('choice error', 'occluder capture', 'background leak', 'floor slip', 'near-depth miss')
LIFT = ('occluder capture', 'background leak', 'floor slip')
KINDS = ('person', 'parcel', 'place')


def load(name):
    p = D / name
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


class Out:
    def __init__(self):
        self.lines = []

    def __call__(self, *a):
        self.lines.append(' '.join(str(x) for x in a))

    def text(self):
        return '\n'.join(self.lines) + '\n'


def lay(r):
    s = r['scene']
    k = s.find('eval_')
    return s[k:k + 7] if k >= 0 else s


def cl(rows):
    rows = list(rows)
    return f"[layouts {len({lay(r) for r in rows})}, model x layout {len({(r['model'], lay(r)) for r in rows})}]"


def pct(a, b):
    return f"{a}/{b} ({100.0 * a / b:.1f}%)" if b else f"{a}/0"


def pts(rows):
    return [r for r in rows if r.get('has_point') and not r.get('skip')]


def wrong(rows):
    return [r for r in pts(rows) if not r['right_before']]


def right(rows):
    return [r for r in pts(rows) if r['right_before']]


def fixes(rows, f):
    return [r for r in wrong(rows) if r['fx'][f]['right_after']]


def breaks(rows, f):
    return [r for r in right(rows) if not r['fx'][f]['right_after']]


def net(rows, f):
    """(gains, losses, not computed) of full successes over records with a point."""
    g = lo = nc = 0
    for r in pts(rows):
        a = r['fx'][f]['succ_after']
        if a is None:
            nc += 1
            continue
        if a and not r['rec_success']:
            g += 1
        if r['rec_success'] and not a:
            lo += 1
    return g, lo, nc


def fixed_success(rows, f):
    fs = fixes(rows, f)
    return sum(1 for r in fs if r['fx'][f]['succ_after']), len(fs)


def counts_table(rows, say, models):
    say(f"  {'':28s} " + ' '.join(f'{m:>11s}' for m in models) + f" {'all':>6s}")
    for name, fn in (('records', lambda rs: rs), ('with a point', pts), ('wrong target', None),
                     ('wrong target with a point', wrong), ('right target with a point', right)):
        vals = []
        for m in list(models) + [None]:
            rs = [r for r in rows if m is None or r['model'] == m]
            if fn is None:
                rs = [r for r in rs if not r['rec_target_correct']]
                vals.append(len(rs))
            else:
                vals.append(len(fn(rs)))
        say(f"  {name:28s} " + ' '.join(f'{v:11d}' for v in vals[:-1]) + f" {vals[-1]:6d}")


# ---------------------------------------------------------------- primary
def primary_report(say):
    rows = load('rows_primary.jsonl')
    meta = json.loads((D / 'meta_primary.json').read_text())
    say('=' * 110)
    say('I11 STAGE 0 (phase 1b): PRIMARY RUNS (old wording, realistic arms, con)')
    say('=' * 110)
    say('runs: ' + ', '.join(meta['runs']))
    say(f"records: {meta['n_records']} (allowed); per-run tallies: "
        + '; '.join(f"{k.split('_', 1)[1]} {v}" for k, v in meta['tallies'].items()))
    say('')
    say('A. Data and reproduction checks')
    counts_table(rows, say, MODELS3)
    exp = dict(records=(252, 252, 252), point=(252, 248, 252), wrong=(7, 28, 22), wpt=(7, 24, 22), rpt=(245, 224, 230))
    got = dict(records=tuple(len([r for r in rows if r['model'] == m]) for m in MODELS3),
               point=tuple(len(pts([r for r in rows if r['model'] == m])) for m in MODELS3),
               wrong=tuple(len([r for r in rows if r['model'] == m and not r['rec_target_correct']]) for m in MODELS3),
               wpt=tuple(len(wrong([r for r in rows if r['model'] == m])) for m in MODELS3),
               rpt=tuple(len(right([r for r in rows if r['model'] == m])) for m in MODELS3))
    say(f"  expected (FROZEN.md section 1) vs got: " + '; '.join(f"{k} {exp[k]} -> {got[k]} "
                                                              f"{'OK' if exp[k] == got[k] else 'DIFF'}" for k in exp))
    w46 = [r for r in wrong(rows) if r['model'] in GD]
    r454 = [r for r in right(rows) if r['model'] in GD]
    w53, r699 = wrong(rows), right(rows)
    say(f"  GLM + DeepSeek wrong with a point {len(w46)} (expect 46); right with a point {len(r454)} (expect 454); "
        f"all three wrong {len(w53)} (53), right {len(r699)} (699)")
    say(f"  lift reproduction: max |H.target_at(pixel) - recorded world| over {len(pts(rows))} records = "
        f"{max(r['lift_repro_m'] for r in pts(rows)):.1e} m (asserted <= 1e-9)")
    say(f"  recorded target_correct vs match_target now: disagreements {sum(r.get('tc_disagree', False) for r in pts(rows))} "
        f"of {len(pts(rows))} (asserted 0)")
    vm = sum(r.get('vec_mismatch', 0) for r in w53)
    vc = sum(r.get('vec_checked', 0) for r in w53)
    say(f"  section 9: vectorised match mask vs match_target on sampled pixels of the 53 wrong-target records: "
        f"{vm} mismatches of {vc}")
    say(f"  targets per episode: {dict(collections.Counter(r['ntargets'] for r in rows))}")
    # comparison with the original lifting_records.tsv
    orig = {}
    with open(RD / 'rerun/lifting/lifting_records.tsv') as f:
        for o in csv.DictReader(f, delimiter='\t'):
            if o['method'] == 'con' and o['arm'] in ('real', 'staffed', 'plain'):
                orig[(o['model'], o['arm'], o['scene'], o['episode'], o['trial'])] = o
    allw = [r for r in rows if not r['rec_target_correct']]
    agree = collections.Counter()
    diffs = []
    for r in allw:
        o = orig.get((r['model'], r['arm'], r['scene'], r['episode'], str(r['trial'])))
        if o is None:
            diffs.append(('missing in lifting_records', r['scene'], r['episode'], r['trial']))
            continue
        agree['matched'] += 1
        if not r.get('has_point'):
            agree['no point both' if o['has_point'] == 'False' else 'point mismatch'] += 1
            continue
        agree['class same' if o['lift_class'] == r['lift_class'] else 'class differs'] += 1
        agree['surface same' if o['surface'] == r['surface'] else 'surface differs'] += 1
        fo = o['fix_oracle'] == 'True'
        agree['oracle fix same' if fo == r['fx']['oracle']['right_after'] else 'oracle fix differs'] += 1
        if fo:
            fs = o['fix_success'] == 'True'
            agree['oracle success same' if fs == bool(r['fx']['oracle']['succ_after']) else 'oracle success differs'] += 1
        if abs(float(o['d_body_px']) - r['d_body_px']) > 1e-9:
            agree['d_body differs'] += 1
    say(f"  per-record agreement with rerun/lifting/lifting_records.tsv on the {len(allw)} wrong-target records: "
        f"{dict(agree)}; {len(diffs)} unmatched {diffs[:5]}")
    say(f"  reproduction of rerun/lifting itself (repro_lifting/diff.txt): "
        + '; '.join((D / 'repro_lifting/diff.txt').read_text().splitlines()[k] for k in (0, 1, 2)))
    # front snap in frontsnap.py's own format
    say('  front-surface snap (reference row), replayed per record with frontsnap.front_snap, in frontsnap.py\'s format:')
    tally = collections.Counter()
    per_class = collections.defaultdict(collections.Counter)
    for r in pts(rows):
        d = r['fx']['front']
        key = ('right' if r['right_before'] else 'wrong') + '->' + ('right' if d['right_after'] else 'wrong') + \
              (' (moved)' if d['moved'] else '')
        tally[(r['model'], key)] += 1
        per_class[r['kind']][key] += 1
    for m in MODELS3:
        say(f"    {m:11s} " + str({k[1]: v for k, v in sorted(tally.items()) if k[0] == m}))
    for c in KINDS:
        say(f"    {c:7s} " + str(dict(sorted(per_class[c].items()))))
    ff = sum(v for k, v in tally.items() if k[1].startswith('wrong->right'))
    fb = sum(v for k, v in tally.items() if k[1].startswith('right->wrong'))
    fm = sum(v for k, v in tally.items() if k[1] == 'right->right (moved)')
    say(f"    total fixes {ff}, breaks {fb}, correct records moved but still right {fm}  -> reference check "
        f"(8 fixes, 37 breaks of 752): {'REPRODUCED' if (ff, fb) == (8, 37) else 'NOT REPRODUCED'}")
    werr = [r['fx']['front'].get('world_err', 0.0) for r in pts(rows) if r['fx']['front']['moved']]
    say(f"    front-snap target rebuilt with H.target_at(world=...): max world error {max(werr) if werr else 0:.1e} m")

    # B. error classes
    say('')
    say('B. Error classes of the wrong targets with a point (lifting.py rules and precedence; FROZEN.md section 2)')
    say(f"  {'model':11s} {'n':>3s} " + ' '.join(f'{c[:12]:>12s}' for c in CLASSES) + '  lift classes')
    for m in list(MODELS3) + ['GLM+DS', 'all']:
        rs = [r for r in w53 if (m == 'all') or (m == 'GLM+DS' and r['model'] in GD) or r['model'] == m]
        cc = collections.Counter(r['lift_class'] for r in rs)
        say(f"  {m:11s} {len(rs):3d} " + ' '.join(f'{cc[c]:12d}' for c in CLASSES)
            + f"  {sum(cc[c] for c in LIFT):3d} {cl(rs)}")
    say('  by kind (all three models):')
    for k in KINDS:
        rs = [r for r in w53 if r['kind'] == k]
        say(f"    {k:7s} n={len(rs):2d} {dict(collections.Counter(r['lift_class'] for r in rs))}")
    say('  by kind (GLM + DeepSeek):')
    for k in KINDS:
        rs = [r for r in w46 if r['kind'] == k]
        say(f"    {k:7s} n={len(rs):2d} {dict(collections.Counter(r['lift_class'] for r in rs))}")
    say('  descriptive cross-tab, class x lifting.surface() of the recorded lift (all 53):')
    for c in CLASSES:
        rs = [r for r in w53 if r['lift_class'] == c]
        if rs:
            surf = collections.Counter(('person' if r['surface'].startswith(('person', 'target person')) else
                                        'footprint' if r['surface'].startswith('footprint') else r['surface'])
                                       for r in rs)
            say(f"    {c:16s} n={len(rs):2d} {dict(surf)}")
    nlift46 = sum(r['lift_class'] in LIFT for r in w46)
    nchoice46 = sum(r['lift_class'] == 'choice error' for r in w46)
    say(f"  lift classes among the 46: {pct(nlift46, 46)} {cl([r for r in w46 if r['lift_class'] in LIFT])}; over 50 "
        f"(4 GLM no-point records as non-lift): {pct(nlift46, 50)}")
    say(f"  choice errors among the 46 (Stage 1 gate, >= 30%): {pct(nchoice46, 46)}; near-depth misses "
        f"{sum(r['lift_class'] == 'near-depth miss' for r in w46)}")

    # C. per fix
    say('')
    say('C. Fixes (FROZEN.md sections 3-6). fixes = wrong -> right; breaks = right -> wrong (match_target).')
    say('   46 = GLM+DeepSeek wrong with a point; 454 = their right with a point; 53 / 699 = all three models.')
    say(f"  {'fix':6s} {'mv R/W (GD)':>12s} {'fix/46':>7s} {'brk/454':>8s} {'fix/53':>7s} {'brk/699':>8s} "
        f"{'succ|fix46':>10s} {'net GD (+/-)':>13s} {'net all (+/-)':>14s} {'gain':>5s} {'share':>6s} "
        f"{'share3':>6s} {'shareNet':>8s}")
    res = {}
    for f in FIXES:
        mvR = sum(r['fx'][f]['moved'] for r in r454)
        mvW = sum(r['fx'][f]['moved'] for r in w46)
        f46, b454 = len(fixes(w46, f)), len(breaks(r454, f))
        f53, b699 = len(fixes(w53, f)), len(breaks(r699, f))
        s46, n46 = fixed_success(w46, f)
        gGD = net([r for r in rows if r['model'] in GD], f)
        gAll = net(rows, f)
        res[f] = dict(mvR=mvR, mvW=mvW, f46=f46, b454=b454, f53=f53, b699=b699, s46=s46, gain=f46 - b454,
                      gain3=f53 - b699, netGD=gGD[0] - gGD[1], netAll=gAll[0] - gAll[1], gGD=gGD, gAll=gAll)
    og = res['oracle']['gain']
    og3 = res['oracle']['gain3']
    ogn = res['oracle']['netGD']
    for f in FIXES:
        x = res[f]
        sh = x['gain'] / og if og > 0 else math.nan
        sh3 = x['gain3'] / og3 if og3 > 0 else math.nan
        shn = x['netGD'] / ogn if ogn > 0 else math.nan
        x.update(share=sh, share3=sh3, shareNet=shn)
        say(f"  {f:6s} {x['mvR']:5d}/{x['mvW']:<6d} {x['f46']:7d} {x['b454']:8d} {x['f53']:7d} {x['b699']:8d} "
            f"{x['s46']:4d}/{x['f46']:<5d} {x['netGD']:+5d} ({x['gGD'][0]}/{x['gGD'][1]}) "
            f"{x['netAll']:+5d} ({x['gAll'][0]}/{x['gAll'][1]}) {x['gain']:5d} {sh:6.2f} {sh3:6.2f} {shn:8.2f}")
    nc = {f: net(rows, f)[2] for f in FIXES}
    say(f"  replays not computed: {nc} (0 = every moved record was replayed)")
    say("  share = (fix/46 - brk/454) / oracle's; share3 = (fix/53 - brk/699) / oracle's; shareNet = net GD successes / "
        "oracle's (secondary)")
    # losses whose baseline does not reproduce
    for f in FIXES:
        lo = [r for r in pts(rows) if r['rec_success'] and r['fx'][f]['succ_after'] is False]
        nb = [r for r in lo if r['fx'][f].get('baseline_reproduces') is False]
        if lo:
            say(f"    {f}: losses {len(lo)} (right->right moved {sum(r['right_before'] and r['fx'][f]['right_after'] for r in lo)}, "
                f"breaks {sum(r['right_before'] and not r['fx'][f]['right_after'] for r in lo)}); of the losses, replay with the "
                f"recorded target does not reproduce the recorded success in {len(nb)}")
    say('')
    say('  per model (fix/wrong-with-point, break/right-with-point, moves on right/wrong):')
    for f in FIXES:
        parts = []
        for m in MODELS3:
            rs = [r for r in rows if r['model'] == m]
            parts.append(f"{m} {len(fixes(rs, f))}/{len(wrong(rs))} brk {len(breaks(rs, f))}/{len(right(rs))} "
                         f"mv {sum(r['fx'][f]['moved'] for r in right(rs))}/{sum(r['fx'][f]['moved'] for r in wrong(rs))}")
        say(f"    {f:6s} " + ' | '.join(parts))
    say('')
    say('  M1 fixes by error class (GLM+DS 46 | all 53):')
    say(f"    {'fix':6s} " + ' '.join(f'{c[:14]:>16s}' for c in CLASSES))
    say(f"    {'n':6s} " + ' '.join(f"{sum(r['lift_class'] == c for r in w46):7d} | {sum(r['lift_class'] == c for r in w53):5d}" for c in CLASSES))
    for f in FIXES:
        say(f"    {f:6s} " + ' '.join(f"{sum(r['lift_class'] == c for r in fixes(w46, f)):7d} | "
                                       f"{sum(r['lift_class'] == c for r in fixes(w53, f)):5d}" for c in CLASSES))
    say('  M1 fixes by kind (GLM+DS 46 | all 53):')
    say(f"    {'n':6s} " + ' '.join(f"{k:>7s} {sum(r['kind'] == k for r in w46):3d}|{sum(r['kind'] == k for r in w53):<3d}" for k in KINDS))
    for f in FIXES:
        say(f"    {f:6s} " + ' '.join(f"{'':7s} {sum(r['kind'] == k for r in fixes(w46, f)):3d}|"
                                       f"{sum(r['kind'] == k for r in fixes(w53, f)):<3d}" for k in KINDS))
    say('  M2 breaks by kind (GLM+DS 454 | all 699):')
    say(f"    {'n':6s} " + ' '.join(f"{k:>7s} {sum(r['kind'] == k for r in r454):3d}|{sum(r['kind'] == k for r in r699):<3d}" for k in KINDS))
    for f in FIXES:
        say(f"    {f:6s} " + ' '.join(f"{'':7s} {sum(r['kind'] == k for r in breaks(r454, f)):3d}|"
                                       f"{sum(r['kind'] == k for r in breaks(r699, f)):<3d}" for k in KINDS))
    say('  clustering of the key counts:')
    for f in FIXES:
        say(f"    {f:6s} fixes/46 {cl(fixes(w46, f))}; breaks/454 {cl(breaks(r454, f))}; breaks/699 {cl(breaks(r699, f))}")
    say('  V1-V4 moves on right-target records by what s (the lift surface) was (all 699):')
    for f in ('V1', 'V2', 'V3', 'V4'):
        key = 's_cls_G2' if f == 'V4' else 's_cls'
        say(f"    {f}: moved {sum(r['fx'][f]['moved'] for r in r699)} {dict(collections.Counter(r[key] for r in r699 if r['fx'][f]['moved']))}; "
            f"surface classes over all 699 {dict(collections.Counter(r[key] for r in r699))}")
    say('  V1 on the 53 wrong targets, by class: lift surface s and outcome')
    for c in CLASSES:
        rs = [r for r in w53 if r['lift_class'] == c]
        if rs:
            say(f"    {c:16s} n={len(rs):2d} s {dict(collections.Counter(r['s_cls'] for r in rs))}; V1 moved "
                f"{sum(r['fx']['V1']['moved'] for r in rs)}, fixed {sum(r['fx']['V1']['right_after'] for r in rs)}; "
                f"V3 moved {sum(r['fx']['V3']['moved'] for r in rs)}, fixed {sum(r['fx']['V3']['right_after'] for r in rs)}")
    say('  replay paths used (moved records, all fixes): '
        + str(dict(collections.Counter(r['fx'][f].get('path') for r in pts(rows) for f in FIXES if r['fx'][f]['moved']))))

    # D. criteria
    say('')
    say('D. I11 Stage 0 criteria (REPORT.md I11 "First probe", verbatim in FROZEN.md section 7)')
    o = res['oracle']
    v1 = res['V1']
    S1 = nlift46 >= 19
    S2 = o['f46'] >= 15 and o['b454'] <= 2
    S3 = o['s46'] >= 8
    S4 = (v1['share'] >= 0.70) if not math.isnan(v1['share']) else False
    K1 = nlift46 <= 9
    K2 = o['f46'] < 6
    K3 = o['b454'] > 5
    say(f"  S1 lift classes >= 40% of 46 (>= 19): {pct(nlift46, 46)} -> {'MET' if S1 else 'NOT MET'} "
        f"(over 50: {pct(nlift46, 50)})")
    say(f"  S2 oracle fixes >= 15 of 46 and breaks <= 2 of 454: fixes {o['f46']}/46, breaks {o['b454']}/454 -> "
        f"{'MET' if S2 else 'NOT MET'}{'' if o['f46'] >= 15 else ' (fixes short)'}{'' if o['b454'] <= 2 else ' (breaks over)'}")
    say(f"  S3 >= 8 of the oracle's fixed records among the 46 become full successes: {o['s46']}/{o['f46']} -> "
        f"{'MET' if S3 else 'NOT MET'}")
    say(f"  S4 V1 reaches >= 70% of the oracle gain: V1 gain {v1['gain']} (fixes {v1['f46']} - breaks {v1['b454']}) / "
        f"oracle gain {o['gain']} (fixes {o['f46']} - breaks {o['b454']}) = {v1['share']:.2f} -> {'MET' if S4 else 'NOT MET'}")
    say(f"  Kill: lift classes < 20% of 46 (<= 9): {nlift46} -> {'FIRES' if K1 else 'no'}; oracle fixes < 6: "
        f"{o['f46']} -> {'FIRES' if K2 else 'no'}; oracle breaks > 5 of 454: {o['b454']} -> {'FIRES' if K3 else 'no'}")
    verdict = 'KILL' if (K1 or K2 or K3) else ('SUCCESS' if (S1 and S2 and S3 and S4) else 'NEITHER')
    say(f"  VERDICT (I11 Stage 0): {verdict}")
    v1kill = v1['f46'] < 6 or v1['b454'] > 5
    say(f"  V1 as a zero-call rule: fixes {v1['f46']}/46 (kill if < 6), breaks {v1['b454']}/454 (kill if > 5) -> "
        f"{'V1 KILLED (zero-call snap route ends)' if v1kill else 'V1 not killed'}; S4 {'met' if S4 else 'not met'}"
        f"{' but the kill wins' if (S4 and v1kill) else ''}")
    for f in ('V2', 'V3', 'V4'):
        x = res[f]
        say(f"  {f} (no verdict): fixes {x['f46']}/46, breaks {x['b454']}/454, share {x['share']:.2f}; against the bars: "
            f"share >= 0.70 {'yes' if x['share'] >= 0.70 else 'no'}, fixes < 6 {'yes' if x['f46'] < 6 else 'no'}, "
            f"breaks > 5 {'yes' if x['b454'] > 5 else 'no'}")
    x = res['front']
    say(f"  front-snap reference: fixes {x['f46']}/46, breaks {x['b454']}/454, share {x['share']:.2f}")

    # E. I33
    say('')
    say('E. I33 Stage B trigger (FROZEN.md section 8). T1: lift on floor or structure, or the 5x5 window holds >= 2')
    say('   labels (floor, structure, object components; fragments ignored). T2: T1 or another object within 15 px.')
    p15 = [r for r in w53 if r['kind'] == 'person']
    d1 = [r for r in r699 if r['kind'] == 'person']
    t1c = sum(r['T1'] for r in p15)
    t1fp = sum(r['T1'] for r in d1)
    say(f"  person wrong targets: n={len(p15)} ({dict(collections.Counter(r['model'] for r in p15))}) {cl(p15)}")
    say(f"    T1 catches {t1c}/{len(p15)}; T2 catches {sum(r['T2'] for r in p15)}/{len(p15)}")
    say(f"    T1 by class: " + str({c: f"{sum(r['T1'] for r in p15 if r['lift_class'] == c)}/{sum(r['lift_class'] == c for r in p15)}"
                                    for c in CLASSES if any(r['lift_class'] == c for r in p15)}))
    say(f"  D1 (decisive) right-target person records: n={len(d1)}; T1 false triggers {pct(t1fp, len(d1))}; T2 "
        f"{pct(sum(r['T2'] for r in d1), len(d1))}  {cl([r for r in d1 if r['T1']])}")
    say(f"  D2 all right-target records with a point: n={len(r699)}; T1 {pct(sum(r['T1'] for r in r699), len(r699))}; "
        f"T2 {pct(sum(r['T2'] for r in r699), len(r699))}")
    for k in KINDS:
        rs = [r for r in r699 if r['kind'] == k]
        say(f"    right {k:7s} n={len(rs):3d} T1 {pct(sum(r['T1'] for r in rs), len(rs))}  T2 {pct(sum(r['T2'] for r in rs), len(rs))}")
    ok = t1c >= 12 and t1fp <= 0.10 * len(d1)
    kill = t1c < 10
    say(f"  I33 Stage B: success needs T1 >= 12/15 and <= 10% of D1 ({0.10 * len(d1):.1f}); kill if < 10/15 -> "
        f"{'SUCCESS' if ok else ('KILL' if kill else 'NEITHER')} (T1 {t1c}/{len(p15)}, D1 {pct(t1fp, len(d1))})")
    say('  T1 / T2 on all 53 wrong targets, by class:')
    for c in CLASSES:
        rs = [r for r in w53 if r['lift_class'] == c]
        if rs:
            say(f"    {c:16s} n={len(rs):2d} T1 {sum(r['T1'] for r in rs):2d}  T2 {sum(r['T2'] for r in rs):2d}")
    allp = pts(rows)
    fl = [r for r in allp if r['T1']]
    say(f"  T1 flags {pct(len(fl), len(allp))} of all records with a point (the share of goal steps that would carry the "
        f"verification call); of the flagged, wrong targets {pct(sum(not r['right_before'] for r in fl), len(fl))}")
    fl2 = [r for r in allp if r['T2']]
    say(f"  T2 flags {pct(len(fl2), len(allp))}; of the flagged, wrong targets {pct(sum(not r['right_before'] for r in fl2), len(fl2))}")
    say(f"  (post hoc, not judged) T1 under G2 components: person wrong {sum(r['T1_G2'] for r in p15)}/{len(p15)}, D1 "
        f"{pct(sum(r['T1_G2'] for r in d1), len(d1))}")

    # F. M4
    say('')
    say('F. M4 component recovery (FROZEN.md section 6)')
    m4 = [r for r in load('m4_primary.jsonl') if r['table'] == 'm4']
    m4w = [r for r in load('m4_primary.jsonl') if r['table'] == 'm4_wrong']
    m4tab(m4, say, f"  primary scenes (realistic): distinct (scene folder, scene, true target) n = {len(m4) // 2}")
    say('  true targets of the 53 wrong-target records, by error class (G1 | G2):')
    for c in CLASSES:
        for join in ('G1', 'G2'):
            rs = [r for r in m4w if r['lift_class'] == c and r['join'] == join and not r['empty']]
            if rs:
                iou = np.array([r['IoU'] for r in rs])
                say(f"    {join} {c:16s} n={len(rs):2d} med IoU {np.median(iou):.2f}, IoU>=.5 {int((iou >= .5).sum())}, merged "
                    f"{sum(r['merged'] for r in rs)}, merged w/ body {sum(r['merged_body'] for r in rs)}, split "
                    f"{sum(r['split'] for r in rs)}, structure {sum(r['structure'] for r in rs)}")
    fr = load('m4_fresh.jsonl')
    mf = D / 'meta_fresh.json'
    say(f"  fresh seeds 100-109: {json.loads(mf.read_text())['note'] if mf.exists() else 'not run'}")
    if fr:
        m4tab(fr, say, f"  fresh seeds: n = {len(fr) // 2}")

    # predictions
    say('')
    say('Predictions written in FROZEN.md section 12 (before scoring) against the outcomes:')
    say(f"  V1 'at most about 11 of 46, share at most about 0.55; S4 likely fails': V1 {res['V1']['f46']}/46, share "
        f"{res['V1']['share']:.2f}; S4 {'met' if res['V1']['share'] >= 0.7 else 'not met'}")
    say(f"  V3 'may add up to about 5 occluder captures': V3 fixes occluder captures "
        f"{sum(r['lift_class'] == 'occluder capture' for r in fixes(w46, 'V3'))} of the 46 and "
        f"{sum(r['lift_class'] == 'occluder capture' for r in fixes(w53, 'V3'))} of the 53; V3 breaks {res['V3']['b454']}/454, "
        f"{res['V3']['b699']}/699")
    say(f"  T1 'about 11 of 15 (all 6 background leaks and 5 straddling occluder captures)': T1 {t1c}/{len(p15)} (background "
        f"leaks {sum(r['T1'] for r in p15 if r['lift_class'] == 'background leak')}/{sum(r['lift_class'] == 'background leak' for r in p15)}, "
        f"occluder captures {sum(r['T1'] for r in p15 if r['lift_class'] == 'occluder capture')}/{sum(r['lift_class'] == 'occluder capture' for r in p15)})")

    # G. per record
    say('')
    say('G. The 53 wrong targets, one row each (R = right after the fix)')
    say(f"  {'model':10s} {'arm':7s} {'scene':36s} {'ep':3s} {'t':1s} {'kind':6s} {'class':16s} {'dB':>5s} {'s':9s} "
        f"{'T1':2s} {'T2':2s} orc V1 V2 V3 V4 frt  orcS")
    for r in sorted(w53, key=lambda r: (r['kind'], r['model'], r['scene'], r['episode'], r['trial'])):
        say(f"  {r['model']:10s} {r['arm']:7s} {r['scene']:36s} {r['episode']:3s} {r['trial']} {r['kind']:6s} "
            f"{r['lift_class']:16s} {r['d_body_px'] if r['d_body_px'] is not None else float('inf'):5.1f} {r['s_cls']:9s} "
            f"{'T' if r['T1'] else '.':2s} {'T' if r['T2'] else '.':2s} "
            + ' '.join(f"{'R' if r['fx'][f]['right_after'] else ('m' if r['fx'][f]['moved'] else '.'):>3s}" for f in FIXES)
            + f"  {str(r['fx']['oracle']['succ_after'])[0] if r['fx']['oracle']['right_after'] else '-'}")
    say('  (R right after; m moved but still wrong; . not moved)')
    say('')
    say('  right-target records broken by V1 (all 699):')
    for r in breaks(r699, 'V1'):
        say(f"    {r['model']:10s} {r['arm']:7s} {r['scene']:36s} {r['episode']} t{r['trial']} {r['kind']:6s} s={r['s_cls']} "
            f"d={r['fx']['V1'].get('d')}")
    say('  right-target records broken by the oracle (all 699):')
    for r in breaks(r699, 'oracle'):
        say(f"    {r['model']:10s} {r['arm']:7s} {r['scene']:36s} {r['episode']} t{r['trial']} {r['kind']:6s} "
            f"d={r['fx']['oracle'].get('d')}")
    return dict(res=res, nlift46=nlift46, nchoice46=nchoice46, verdict=verdict, v1kill=v1kill, S=(S1, S2, S3, S4),
                K=(K1, K2, K3), t1c=t1c, n15=len(p15), t1fp=t1fp, nd1=len(d1), i33=('SUCCESS' if ok else ('KILL' if kill else 'NEITHER')),
                t2c=sum(r['T2'] for r in p15), d2=sum(r['T1'] for r in r699), flag752=len(fl), nall=len(allp),
                flagged_wrong=sum(not r['right_before'] for r in fl))


def m4tab(rows, say, title):
    say(title)
    say(f"    {'join':4s} {'kind':7s} {'n':>4s} {'empty':>5s} {'medIoU':>6s} {'IoU>=.5':>8s} {'IoU>=.7':>8s} "
        f"{'merged':>6s} {'mBody':>5s} {'split':>5s} {'struct':>6s}  medCov medPur")
    for join in ('G1', 'G2'):
        for kind in ('person', 'parcel', 'place', 'ALL'):
            rs = [r for r in rows if r['join'] == join and (kind == 'ALL' or r['kind'] == kind)]
            if not rs:
                continue
            ne = [r for r in rs if not r['empty']]
            n = len(ne)
            if not n:
                say(f"    {join:4s} {kind:7s} {0:4d} {len(rs):5d}")
                continue
            iou = np.array([r['IoU'] for r in ne])
            say(f"    {join:4s} {kind:7s} {n:4d} {len(rs) - n:5d} {np.median(iou):6.2f} {int((iou >= .5).sum()):3d}/{n:<4d} "
                f"{int((iou >= .7).sum()):3d}/{n:<4d} {sum(r['merged'] for r in ne):6d} {sum(r['merged_body'] for r in ne):5d} "
                f"{sum(r['split'] for r in ne):5d} {sum(r['structure'] for r in ne):6d}  "
                f"{np.median([r['coverage'] for r in ne]):.2f}   {np.median([r['purity'] for r in ne]):.2f}")


# ---------------------------------------------------------------- extras
def extras_report(say):
    out = {}
    for setname, title in (('xnew', 'X-new (new wording, realistic arms, con): the blind replication'),
                           ('xph', 'X-ph (placeholder ph and dev_seed0-2 arms, con)'),
                           ('xph2', 'X-ph2 (s10 con, s8 convlm and conorc; placeholder eval scenes)')):
        rows = load(f'rows_{setname}.jsonl')
        if not rows:
            say(f'{title}: not run')
            continue
        meta = json.loads((D / f'meta_{setname}.json').read_text())
        say('=' * 110)
        say(title)
        say('=' * 110)
        say(f"runs ({len(meta['runs'])}): " + ', '.join(meta['runs']))
        skipped = collections.Counter()
        for k, v in meta['tallies'].items():
            for kk, vv in v.items():
                if kk != 'allowed':
                    skipped[kk] += vv
        say(f"records allowed {meta['n_records']}; skipped by the harness {dict(skipped) or 0}")
        nopix = [r for r in rows if r.get('no_pixel')]
        rep = [r for r in rows if r.get('skip')]
        tcd = [r for r in pts(rows) if r.get('tc_disagree')]
        say(f"records with a target but no pixel: {len(nopix)} (runs {sorted({r['run'] for r in nopix})}); lift not "
            f"reproduced (skipped): {len(rep)}; recorded target_correct vs match_target now disagree: {len(tcd)}")
        if tcd:
            say('   disagreements (match_target now is used): ' + '; '.join(
                f"{r['run'].split('_', 1)[1]} {r['scene']} {r['episode']} t{r['trial']} rec {r['rec_target_correct']} now {r['right_before']}"
                for r in tcd[:12]))
        multi = [r for r in pts(rows) if r['ntargets'] > 1]
        say(f"records in episodes with several targets: {len(multi)}")
        if setname == 'xph2':
            keyc = collections.Counter((r['model'], r['scene'], r['episode'], r['trial'], round(r['u'], 3), round(r['v'], 3))
                                       for r in pts(rows))
            say(f"repeated draws across runs (same model, scene, episode, trial and pixel): {sum(c - 1 for c in keyc.values() if c > 1)} "
                f"extra copies in {sum(1 for c in keyc.values() if c > 1)} keys")
        vm = sum(r.get('vec_mismatch', 0) for r in wrong(rows))
        vc = sum(r.get('vec_checked', 0) for r in wrong(rows))
        say(f"section 9 vectorised mask check on single-target wrong records: {vm} mismatches of {vc} sampled pixels")
        models = sorted({r['model'] for r in rows}, key=lambda m: [r['model'] for r in rows].index(m))
        say('')
        counts_table(rows, say, models)
        say('')
        say('per model (pooled over that model\'s runs in this set; never pooled with the primary):')
        for m in models:
            rs = [r for r in rows if r['model'] == m]
            w, rt = wrong(rs), right(rs)
            cc = collections.Counter(r['lift_class'] for r in w)
            say(f"  {m} [{', '.join(sorted({r['arm'] for r in rs}))}]: with a point {len(pts(rs))}, wrong {len(w)}, right {len(rt)}; "
                f"classes {dict(cc)}; lift classes {sum(cc[c] for c in LIFT)}/{len(w)} {cl(w)}")
            say(f"    wrong by kind {dict(collections.Counter(r['kind'] for r in w))}")
            for f in FIXES:
                g, lo, nc = net(rs, f)
                s_, n_ = fixed_success(rs, f)
                say(f"    {f:6s} fixes {len(fixes(rs, f)):3d}/{len(w):<3d} breaks {len(breaks(rs, f)):3d}/{len(rt):<4d} "
                    f"moves R/W {sum(r['fx'][f]['moved'] for r in rt):4d}/{sum(r['fx'][f]['moved'] for r in w):<3d} "
                    f"fixed->success {s_}/{n_}  net {g - lo:+d} ({g}/{lo}{'' if not nc else f', {nc} not replayed'})  "
                    f"fixes by class {dict(collections.Counter(r['lift_class'] for r in fixes(rs, f)))}")
            p = [r for r in w if r['kind'] == 'person']
            d1 = [r for r in rt if r['kind'] == 'person']
            say(f"    I33: person wrong targets {len(p)} {cl(p)}: T1 catches {sum(r['T1'] for r in p)}, T2 {sum(r['T2'] for r in p)}; "
                f"D1 n={len(d1)} T1 false {pct(sum(r['T1'] for r in d1), len(d1))}; D2 n={len(rt)} T1 "
                f"{pct(sum(r['T1'] for r in rt), len(rt))}; T1 on all wrong {sum(r['T1'] for r in w)}/{len(w)}")
            out[(setname, m)] = dict(p=len(p), t1=sum(r['T1'] for r in p), d1=len(d1), d1fp=sum(r['T1'] for r in d1),
                                     w=len(w), lift=sum(cc[c] for c in LIFT),
                                     orc=len(fixes(rs, 'oracle')), orcb=len(breaks(rs, 'oracle')),
                                     v1=len(fixes(rs, 'V1')), v1b=len(breaks(rs, 'V1')), rt=len(rt))
        say('')
        say('per run (fixes/wrong, breaks/right) for the oracle and V1, and T1 on person wrong targets:')
        for run in meta['runs']:
            rs = [r for r in rows if r['run'] == run]
            w, rt = wrong(rs), right(rs)
            p = [r for r in w if r['kind'] == 'person']
            say(f"  {run:48s} pts {len(pts(rs)):3d} wrong {len(w):3d} | oracle {len(fixes(rs, 'oracle')):2d}/{len(w):<3d} "
                f"brk {len(breaks(rs, 'oracle')):2d} | V1 {len(fixes(rs, 'V1')):2d} brk {len(breaks(rs, 'V1')):2d} | V2 "
                f"{len(fixes(rs, 'V2')):2d} brk {len(breaks(rs, 'V2')):2d} | V3 {len(fixes(rs, 'V3')):2d} brk "
                f"{len(breaks(rs, 'V3')):2d} | V4 {len(fixes(rs, 'V4')):2d} brk {len(breaks(rs, 'V4')):2d} | front "
                f"{len(fixes(rs, 'front')):2d} brk {len(breaks(rs, 'front')):2d} | T1 person {sum(r['T1'] for r in p)}/{len(p)}")
        if setname == 'xnew':
            pw = [r for r in wrong(rows) if r['kind'] == 'person']
            pd1 = [r for r in right(rows) if r['kind'] == 'person']
            say('')
            say(f"(post hoc summary, pooled over the X-new models only, never with the primary) person wrong targets "
                f"n={len(pw)} {cl(pw)}: T1 {pct(sum(r['T1'] for r in pw), len(pw))}, T2 {pct(sum(r['T2'] for r in pw), len(pw))}; "
                f"by class " + str({c: f"{sum(r['T1'] for r in pw if r['lift_class'] == c)}/{sum(r['lift_class'] == c for r in pw)}"
                                    for c in CLASSES if any(r['lift_class'] == c for r in pw)})
                + f"; D1 n={len(pd1)} T1 false {pct(sum(r['T1'] for r in pd1), len(pd1))}; D2 n={len(right(rows))} T1 "
                f"{pct(sum(r['T1'] for r in right(rows)), len(right(rows)))}; excluding Gemini (no person wrong targets) D1 "
                f"{pct(sum(r['T1'] for r in pd1 if r['model'] != 'gemini38_low'), sum(r['model'] != 'gemini38_low' for r in pd1))}")
            say('')
            say('X-new person wrong targets, one row each (the blind I33 test):')
            for r in sorted([r for r in wrong(rows) if r['kind'] == 'person'], key=lambda r: (r['model'], r['scene'])):
                say(f"  {r['model']:12s} {r['arm']:7s} {r['scene']:36s} {r['episode']} t{r['trial']} {r['lift_class']:16s} "
                    f"s={r['s_cls']:9s} T1={'T' if r['T1'] else '.'} T2={'T' if r['T2'] else '.'} oracle "
                    f"{'R' if r['fx']['oracle']['right_after'] else '.'} V1 {'R' if r['fx']['V1']['right_after'] else '.'}")
        if setname == 'xph':
            m4 = [r for r in load('m4_xph.jsonl') if r['table'] == 'm4']
            if m4:
                say('')
                m4tab(m4, say, f"M4 on X-ph scenes: distinct (scene folder, scene, true target) n = {len(m4) // 2}")
    return out


def main():
    p = Out()
    prim = primary_report(p)
    (D / 'out_primary.txt').write_text(p.text())
    e = Out()
    ext = extras_report(e)
    (D / 'out_extras.txt').write_text(e.text())
    # tables.txt: everything in one file
    parts = ['I11 STAGE 0 (phase 1b) TABLES', 'FROZEN.md sha256 11f7b32746c7fc0be414883b430d59ddef98d2388fd89e082e40361b637a9165',
             '', '#' * 110, 'S0 (s0_fit.txt)', '#' * 110, (D / 's0_fit.txt').read_text(),
             '#' * 110, 'Reproduction of rerun/lifting (repro_lifting/diff.txt)', '#' * 110,
             (D / 'repro_lifting/diff.txt').read_text(), p.text(), e.text()]
    v = D / 'verify_out.txt'
    if v.exists():
        parts += ['#' * 110, 'Verification (verify_out.txt; same agent, separate code: DEVIATIONS.md D5)', '#' * 110, v.read_text()]
    (D / 'tables.txt').write_text('\n'.join(parts) + '\n')
    with open(D / 'records.tsv', 'w') as f:                  # FROZEN.md section 10 step 4: one row per record x fix
        header_done = False
        for s_ in ('primary', 'xnew', 'xph', 'xph2'):
            pth = D / f'records_{s_}.tsv'
            if not pth.exists():
                continue
            lines = pth.read_text().splitlines()
            if not header_done:
                f.write(lines[0] + '\n')
                header_done = True
            f.write('\n'.join(lines[1:]) + '\n')
    with open(D / 'rows.jsonl', 'w') as f:
        for s in ('primary', 'xnew', 'xph', 'xph2'):
            for r in load(f'rows_{s}.jsonl'):
                f.write(json.dumps(r) + '\n')
    json.dump({'primary': {k: v for k, v in prim.items() if k != 'res'}, 'res': prim['res'],
               'extras': {f'{k[0]}|{k[1]}': v for k, v in ext.items()}},
              open(D / 'summary.json', 'w'), indent=1, default=str)
    print(p.text())


if __name__ == '__main__':
    main()
