"""Tables of the snap replay from snap_records.jsonl (written by snap.py). Prints and saves snap_tables.txt."""
import collections
import json
import statistics as st

HERE = 'results/still-scenes/depth-fix/first-check'
MODELS = ['qwen38_low', 'glm53', 'dsv41', 'haiku55', 'sonnet55_low', 'gemini38_low']
SHORT = {'qwen38_low': 'Qwen', 'glm53': 'GLM', 'dsv41': 'DeepSeek', 'haiku55': 'Haiku', 'sonnet55_low': 'Sonnet',
         'gemini38_low': 'Gemini'}
SITS = ['no_one', 'waiting_line', 'conversation_group']
SS = {'no_one': 'nobody', 'waiting_line': 'lines', 'conversation_group': 'groups'}
MS = {'cot': 'R', 'fsc': 'E'}
rows = [json.loads(l) for l in open(f'{HERE}/snap_records.jsonl')]
out = []


def p(s=''):
    print(s)
    out.append(s)


def med(v, f='{:.2f}'):
    return f.format(st.median(v)) if v else '-'


def is_marked(r):
    return r['source'] == 'direct_marking' and r['status'] == 'ok'


p(f"{len(rows)} records: " + ', '.join(f"{SHORT[m]} {sum(r['model'] == m for r in rows)}" for m in MODELS))
p("Sets: realistic arms real, staffed, plain; Qwen s21 cot/fsc, the others s23 cot/fsc; 84 records a run, 3 trials.")
p("Scores: the package's scorer now, on the recorded pose and on each snapped pose (score_poses).")
mis = sum(r['rec_success'] != r['success'] for r in rows)
p(f"Recorded success vs score now: {mis} of {len(rows)} differ.")
p("snapD = the clearance snap on the solver's floor without people (depth only); snapP = with the person discs of the "
  "same model's C record (same arm, scene, episode, trial).")
p("Classes of a failed pose: near = outside the region within 0.3 m; mid = 0.3-1.0 m; far = more than 1.0 m; inreg = "
  "inside the region but unreachable or heading wrong; body = min_body_gap_m < 0 (robot disc overlaps a true body disc, "
  "radius 0.25 + 0.30 m); line place: behind = level with or behind the last member (the region's own test), front = "
  "nearer the target anchor than the first member, beside = beside or inside the line.")


def tally(sel, snap='snapD'):
    c = collections.Counter()
    for r in sel:
        if not r['target_ok']:
            continue
        c['right'] += 1
        c['ok'] += r['success']
        if not r['success']:
            c['fail'] += 1
            if r['source'] == 'standard':
                c['std'] += 1
            elif r['status'] != 'ok':
                c['nopose'] += 1
            else:
                c['marked'] += 1
                c['fixed'] += r[snap]['success']
                c['still'] += not r[snap]['success']
        elif is_marked(r):
            c['broken'] += not r[snap]['success']
        if is_marked(r):
            c['ok_snap'] += r[snap]['success']
        else:
            c['ok_snap'] += r['success']
    # successes over every record (right target or not) for the totals line
    c['all'] = len(sel)
    c['succ_all'] = sum(r['success'] for r in sel)
    c['succ_all_snap'] = sum((r[snap]['success'] if is_marked(r) else r['success']) for r in sel)
    c['wrong'] = sum(not r['target_ok'] for r in sel)
    return c


for snap in ('snapD', 'snapP'):
    p(f"\n=== T1 ({snap}): per model x situation x method; counts are records with the right target unless named ===")
    p(f"{'model':9s} {'sit':7s} {'m':2s} {'all':>4s} {'wrong':>5s} {'right':>5s} {'succ':>5s} {'fail':>5s} | {'std':>4s} {'nopose':>6s} {'marked':>6s} | {'fixed':>5s} {'still':>5s} {'broken':>6s} | {'succ->':>6s} {'C':>4s}")
    for model in MODELS:
        for sit in SITS + ['all']:
            for method in ('cot', 'fsc', 'both'):
                sel = [r for r in rows if r['model'] == model and (sit == 'all' or r['sit'] == sit)
                       and (method == 'both' or r['method'] == method)]
                c = tally(sel, snap)
                csucc = sum(bool(r['c_success']) for r in sel) if method != 'both' else '-'   # one C record per (episode, trial)
                mlabel = MS.get(method, 'RE')
                p(f"{SHORT[model]:9s} {SS.get(sit, 'all'):7s} {mlabel:2s} {c['all']:4d} {c['wrong']:5d} {c['right']:5d} {c['ok']:5d} {c['fail']:5d} | {c['std']:4d} {c['nopose']:6d} {c['marked']:6d} | {c['fixed']:5d} {c['still']:5d} {c['broken']:6d} | {c['succ_all']:3d}->{c['succ_all_snap']:<3d} {csucc!s:>4s}")
        p('')
    p("std = the standard goal (context said people do not matter), not a marked point, so not snapped; nopose = no pose "
      "(lift failed or call failed). succ-> = successes over all records before and after the snap (wrong targets "
      "included). C = the same model's C successes on the same episodes and trials (recorded score), for reference.")

# ---------------------------------------------------------------- T2 / T3: why still failing; what the snap acts on
for snap in ('snapD', 'snapP'):
    p(f"\n=== T2 ({snap}): marked-point failures with the right target, before ('now') and after the snap, by class ===")
    p(f"{'model':9s} {'sit':7s} {'m':2s} {'stage':6s} {'n':>4s} {'near':>5s} {'mid':>5s} {'far':>5s} {'inreg':>5s} {'body':>5s} | {'behind':>6s} {'beside':>6s} {'front':>6s} | {'behind&near':>11s} {'snapped':>7s} {'med move':>8s}")
    for model in MODELS:
        for sit in ('waiting_line', 'conversation_group', 'no_one'):
            for method in ('cot', 'fsc', 'both'):
                base = [r for r in rows if r['model'] == model and r['sit'] == sit and (method == 'both' or r['method'] == method)
                        and r['target_ok'] and not r['success'] and is_marked(r)]
                for stage in ('now', snap):
                    sel = base if stage == 'now' else [r for r in base if not r[snap]['success']]
                    c = collections.Counter(r[stage]['cls'] for r in sel)
                    body = sum(1 for r in sel if r[stage]['gap'] is not None and r[stage]['gap'] < 0)
                    lp = collections.Counter(r[stage].get('line') for r in sel)
                    bn = sum(1 for r in sel if r[stage].get('line') == 'behind' and r[stage]['cls'] in ('near', 'inreg'))
                    mv = [r['moved_d' if snap == 'snapD' else 'moved_p'] for r in base]
                    snapped = sum(1 for x in mv if x > 0)
                    extra = f"{snapped:7d} {med([x for x in mv if x > 0]):>8s}" if stage == 'now' else ''
                    lines_part = f"{lp['behind']:6d} {lp['beside']:6d} {lp['front']:6d}" if sit == 'waiting_line' else f"{'':6s} {'':6s} {'':6s}"
                    bnp = f"{bn:11d}" if sit == 'waiting_line' else f"{'':11s}"
                    p(f"{SHORT[model]:9s} {SS[sit]:7s} {MS.get(method, 'RE'):2s} {stage:6s} {len(sel):4d} {c['near']:5d} {c['mid']:5d} {c['far']:5d} {c['inreg']:5d} {body:5d} | {lines_part} | {bnp} {extra}")
        p('')
    p("'now' rows: the marked-point failures before the snap; snapped = how many of them the snap moved (the rest lie "
      "on a selectable cell and keep their pose); med move in metres over those moved. The snap rows: those still "
      "failing after it.")

# ---------------------------------------------------------------- T4: I34 Stage 0 gate quantities
p("\n=== T4: I34 Stage 0 gate quantities (marked-point failures with the right target, people scenes) ===")
p("near miss = within 0.3 m of the region or inside it (class near or inreg) before the snap. Gate: snapping fixes "
  "under 60% of near misses, or far misses at lines >= 15%. Kill: snapping recovers >= 70% of near misses.")
for snap in ('snapD', 'snapP'):
    p(f"-- {snap}")
    for model in MODELS + ['pooled']:
        for method in ('cot', 'fsc'):
            parts = []
            for sit in ('waiting_line', 'conversation_group'):
                base = [r for r in rows if (model == 'pooled' or r['model'] == model) and r['sit'] == sit and r['method'] == method
                        and r['target_ok'] and not r['success'] and is_marked(r)]
                nm = [r for r in base if r['now']['cls'] in ('near', 'inreg')]
                fx = sum(r[snap]['success'] for r in nm)
                far = sum(r['now']['cls'] == 'far' for r in base)
                parts.append(f"{SS[sit]}: near {fx}/{len(nm)} fixed ({100 * fx / len(nm) if nm else 0:.0f}%), far {far}/{len(base)} ({100 * far / len(base) if base else 0:.0f}%)")
            p(f"{SHORT.get(model, model):9s} {MS[method]}  " + ';  '.join(parts))

# ---------------------------------------------------------------- T5: mechanisms
p("\n=== T5a: where the marked point sits in the picture at lines (marked points with the right target) ===")
p("dv = v(marked pixel) - v(region point projected at floor height), positive = lower in the image; drange = the pose's "
  "ground range from the camera minus the region point's, negative = toward the camera. rep = the region's "
  "representative point; nearest = the region point nearest the pose.")
for method in ('cot', 'fsc'):
    for model in MODELS:
        for which in ('fail', 'all'):
            sel = [r for r in rows if r['model'] == model and r['method'] == method and r['sit'] == 'waiting_line'
                   and r['target_ok'] and is_marked(r) and 'picture' in r and (which == 'all' or not r['success'])]
            g = lambda k: [r['picture'][k] for r in sel if k in r['picture']]
            dvr, dvn, drr, drn = g('dv_rep'), g('dv_nearest'), g('drange_rep'), g('drange_nearest')
            q = lambda v: (f"[{st.quantiles(v, n=4)[0]:.2f}, {st.quantiles(v, n=4)[2]:.2f}]" if len(v) >= 4 else '')
            p(f"{SHORT[model]:9s} {MS[method]} {which:4s} n {len(sel):3d}  dv_rep med {med(dvr, '{:.0f}')} px  dv_nearest med {med(dvn, '{:.0f}')} px  drange_rep med {med(drr)} m IQR {q(drr)}  drange_nearest med {med(drn)} m")

p("\n=== T5b: line failures (marked point, right target): where the pose stands along the line ===")
p("behind/beside/front as defined above. Along the axis from the target anchor (t = 0) to the last member (t = 1): "
  "counter = t below the first member's t (between the anchor and the first member, or past the anchor); front half = "
  "from there to t 0.5; back half = t 0.5 to 1; past = t > 1. Across: cam = more than 0.3 m off the axis on the "
  "camera's side, on = within 0.3 m of it, far = more than 0.3 m on the other side.")
for model in MODELS:
    for method in ('cot', 'fsc'):
        sel = [r for r in rows if r['model'] == model and r['method'] == method and r['sit'] == 'waiting_line'
               and r['target_ok'] and not r['success'] and is_marked(r) and 't' in r['now']]
        lp = collections.Counter(r['now'].get('line') for r in sel)
        def along(n):
            if n['t'] < n['t_first']:
                return 'counter'
            return 'front half' if n['t'] < 0.5 else ('back half' if n['t'] <= 1 else 'past')
        def across(n):
            return 'cam' if n['perp_cam'] > 0.3 else ('far' if n['perp_cam'] < -0.3 else 'on')
        al = collections.Counter(along(r['now']) for r in sel)
        ac = collections.Counter(across(r['now']) for r in sel)
        p(f"{SHORT[model]:9s} {MS[method]} n {len(sel):3d}  behind {lp['behind']:3d} beside {lp['beside']:3d} front {lp['front']:3d} | "
          f"counter {al['counter']:3d} front-half {al['front half']:3d} back-half {al['back half']:3d} past {al['past']:3d} | "
          f"cam {ac['cam']:3d} on {ac['on']:3d} far {ac['far']:3d} | t med {med([r['now']['t'] for r in sel])} "
          f"across med {med([r['now']['perp_cam'] for r in sel])} m  d_anchor med {med([r['now']['d_anchor'] for r in sel])} m")

p("\n=== T5d: distance from the last member of the line, every marked point with the right target at lines ===")
for model in MODELS:
    for method in ('cot', 'fsc'):
        sel = [r for r in rows if r['model'] == model and r['method'] == method and r['sit'] == 'waiting_line'
               and r['target_ok'] and is_marked(r) and 'd_last' in r['now']]
        ok = [r['now']['d_last'] for r in sel if r['success']]
        bad = [r['now']['d_last'] for r in sel if not r['success']]
        p(f"{SHORT[model]:9s} {MS[method]} all: med {med([r['now']['d_last'] for r in sel])} m (n {len(sel)}); successes {med(ok)} (n {len(ok)}); failures {med(bad)} (n {len(bad)})")

p("\n=== T6: marked points the snap moves, successes included ===")
for snap, mv in (('snapD', 'moved_d'), ('snapP', 'moved_p')):
    for model in MODELS:
        sel = [r for r in rows if r['model'] == model and is_marked(r)]
        succ = [r for r in sel if r['success']]
        moved_s = [r for r in succ if r[mv] > 0]
        broke = sum(1 for r in succ if not r[snap]['success'])
        fails = [r for r in sel if not r['success']]
        moved_f = [r for r in fails if r[mv] > 0]
        fixed_wrong = sum(1 for r in sel if not r['target_ok'] and r[snap]['success'])
        p(f"{snap} {SHORT[model]:9s} marked {len(sel):3d}: successes {len(succ):3d}, moved {len(moved_s):3d}, broken {broke}; failures {len(fails):3d}, moved {len(moved_f):3d}, med move {med([r[mv] for r in moved_f])} m; wrong-target records made successes {fixed_wrong}")

p("\n=== T5c: line failures (marked point, right target) behind the last member: how close, and which component fails ===")
for model in MODELS:
    for method in ('cot', 'fsc', 'both'):
        sel = [r for r in rows if r['model'] == model and (method == 'both' or r['method'] == method) and r['sit'] == 'waiting_line'
               and r['target_ok'] and not r['success'] and is_marked(r)]
        beh = [r for r in sel if r['now'].get('line') == 'behind']
        dl = [r['now']['d_last'] for r in beh if 'd_last' in r['now']]
        dl_all = [r['now']['d_last'] for r in sel if 'd_last' in r['now']]
        unre = sum(not r['now']['reach'] for r in beh)
        inr = sum(r['now']['inreg'] for r in beh)
        hd = sum(r['now']['inreg'] and not r['now']['heading'] for r in beh)
        close = sum(x < 0.5 for x in dl)
        strip = sum(0.5 <= x < 0.55 for x in dl)
        far = sum(x > 1.2 for x in dl)
        p(f"{SHORT[model]:9s} {MS.get(method, 'RE'):2s} fails {len(sel):3d}, behind {len(beh):3d}: d_last med {med(dl)} m (all fails {med(dl_all)}); <0.5 m {close}, 0.50-0.55 m {strip}, >1.2 m {far}; unreachable {unre}, in region {inr} (heading wrong there {hd}); fixed by snapD {sum(r['snapD']['success'] for r in beh)}, snapP {sum(r['snapP']['success'] for r in beh)}")

open(f'{HERE}/snap_tables.txt', 'w').write('\n'.join(out) + '\n')

out_extra = []
p("\n=== T7: what the snap fixes, by the class of the pose before it (marked-point failures with the right target) ===")
p("strip = the recorded pose lies 0.50-0.55 m from the nearest true person's centre: inside the region's 0.5 m inner edge "
  "but inside the scorer's 0.55 m reachability disc (the scorer mismatch of finding 8), so it can never score.")
for snap in ('snapD', 'snapP'):
    for model in MODELS + ['pooled']:
        parts = []
        for sit in ('waiting_line', 'conversation_group'):
            sel = [r for r in rows if (model == 'pooled' or r['model'] == model) and r['sit'] == sit and r['target_ok']
                   and not r['success'] and is_marked(r) and r[snap]['success']]
            c = collections.Counter(r['now']['cls'] for r in sel)
            strip = sum(1 for r in sel if r['now'].get('d_person') is not None and 0.5 <= r['now']['d_person'] < 0.55)
            over = sum(1 for r in sel if r['now']['gap'] is not None and r['now']['gap'] < 0)
            parts.append(f"{SS[sit]}: fixed {len(sel):3d} (inreg {c['inreg']:2d}, near {c['near']:2d}, mid {c['mid']:2d}, far {c['far']:2d}; strip {strip:2d}; on a body {over:2d})")
        p(f"{snap} {SHORT.get(model, model):9s} " + ';  '.join(parts))
open(f'{HERE}/snap_tables.txt', 'w').write('\n'.join(out) + '\n')
