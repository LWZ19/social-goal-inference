"""I14 v0 tables from rows_new.jsonl and rows_oldwatch.jsonl (i14.py). Definitions: METRIC_NOTE.md. Writes tables.txt."""
from __future__ import annotations

import collections
import json
import re

HERE = 'results/still-scenes/metrics-and-statistics/metric-v0'
MODELS = ['qwen38_low', 'glm53', 'dsv41', 'haiku55', 'sonnet55_low', 'gemini38_low']
SHORT = dict(qwen38_low='Qwen', glm53='GLM', dsv41='DeepSeek', haiku55='Haiku', sonnet55_low='Sonnet',
             gemini38_low='Gemini')
SETS = [('old', 'old wording, realistic (s19/s20)'), ('watch', 'watching, dev seeds (s19/s20 dev_seed*, audience)'),
        ('new', 'new wording, realistic (s21/s23)')]
BRIEF = {('old', 'qwen38_low'): (134, 134, 143), ('old', 'glm53'): (96, 99, 141), ('old', 'dsv41'): (109, 123, 157),
         ('watch', 'qwen38_low'): (32, 35, 36), ('watch', 'glm53'): (23, 29, 36), ('watch', 'dsv41'): (23, 33, 36)}
SITS = ['waiting_line', 'conversation_group', 'no_one', 'audience']
OUT = []


def p(line=''):
    OUT.append(line)


def load():
    rows = []
    for name in ('new', 'oldwatch'):
        with open(f'{HERE}/rows_{name}.jsonl') as f:
            rows += [json.loads(line) for line in f]
    return rows


def std_counts():
    """Standard-path (non-goal-step) records per (set, model), from the run logs."""
    out = collections.Counter()
    for name in ('new', 'oldwatch'):
        for line in open(f'{HERE}/run_{name}.log'):
            m = re.search(r'\s(new|old|watch)\s+\S+_(s\d+)_(?:real|staffed|plain|dev_seed\d)_(\S+)_con\s+goal steps (\d+)\s+standard (\d+)', line)
            if m:
                out[(m.group(1), m.group(3))] += int(m.group(5))
    return out


def cells(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r['set'], r['model'])].append(r)
    return by


def S(rs, f):
    return sum(f(r) for r in rs)


def hasbase(r):
    return bool(r['base'])


def agg(rs, tau='45'):
    """Counts over a cell's records: solver hit@k, P5; baselines' expected hit@k; gains."""
    n = len(rs)
    d = dict(n=n)
    d['hit'] = [S(rs, lambda r: r['solver'][tau]['hit'][k] if k < len(r['solver'][tau]['hit']) else r['solver'][tau]['hit'][-1]) for k in range(5)]
    d['p5'] = S(rs, lambda r: r['solver'][tau]['p5'])
    nb = [r for r in rs if hasbase(r)]
    d['nb'] = len(nb)
    for b in ('r5', 'f5', 'ar', 'af'):
        d[b] = [S(nb, lambda r: r['base'][tau][b][k]) for k in range(5)]
    d['r5_p5'] = S(nb, lambda r: r['base'][tau]['r5_p5'])
    d['f5_p5'] = S(nb, lambda r: r['base'][tau]['f5_p5'])
    d['ex_r5'] = S(nb, lambda r: r['base'][tau]['ex_r5'])
    d['ex_ar'] = S(nb, lambda r: r['base'][tau]['ex_ar'])
    # solver over the records with baselines (for like-for-like recovery)
    d['hit_nb'] = [S(nb, lambda r: r['solver'][tau]['hit'][k]) for k in range(5)]
    return d


def pct(a, b):
    return float('nan') if b == 0 else 100.0 * a / b


def main():
    rows = load()
    by = cells(rows)
    std = std_counts()
    p('I14 v0 tables. Definitions: METRIC_NOTE.md (frozen before computing). Counts are records; baseline values are')
    p('expected counts (sum over records of the mean over 1,000 draws). Goal-step records only (they carry runners-up).')
    p('Tolerance 45 deg (the scorer\'s) unless a table says otherwise. Solver verdicts at 45 are the package\'s.')
    p()

    # validation
    mism = collections.Counter()
    nval = 0
    for r in rows:
        nval += r['n_val']
        for k, v in r['mism'].items():
            mism[k] += v
    p('== Checks ==')
    p(f'records {len(rows)}; recorded places on the solver\'s selectable cells: '
      f'{sum(all(r["on_sel"]) for r in rows)} of {len(rows)} records (all 5 places)')
    p(f'replica vs package: {nval} sampled disc cells + {sum(r["nplace"] * 9 for r in rows)} places and jittered places; '
      f'disagreements: {dict(mism) or 0}')
    p(f'recorded success vs package top-1 now: {sum(r["rec_success"] == bool(r["solver"]["45"]["hit_pkg"][0]) for r in rows)} of {len(rows)} agree')
    p(f'records with fewer than 5 places: {sum(r["nplace"] < 5 for r in rows)}; with fewer than 5 disc cells (no draws): '
      f'{sum(not hasbase(r) for r in rows)}')
    p()

    # T1 records
    p('== T1. Records per cell ==')
    p(f'{"set":6s} {"model":9s} {"goal":>5s} {"std":>4s} {"rightT":>6s} {"top in disc":>11s} {"disc cells":>10s} {"F cells":>8s} '
      f'{"region cells outside disc (share)":>34s}')
    for s, _ in SETS:
        for m in MODELS:
            rs = by.get((s, m))
            if not rs:
                continue
            right = [r for r in rs if r['target_ok']]
            reg_all = S(right, lambda r: r['region_sel'])
            reg_d = S(right, lambda r: r['region_disc'])
            p(f'{s:6s} {SHORT[m]:9s} {len(rs):5d} {std[(s, m)]:4d} {len(right):6d} '
              f'{S(rs, lambda r: r["top_in_disc"]):5d} ({pct(S(rs, lambda r: r["top_in_disc"]), len(rs)):3.0f}%) '
              f'{S(rs, lambda r: r["n_disc"]) / len(rs):10.0f} {S(rs, lambda r: r["n_sel"]) / len(rs):8.0f} '
              f'{reg_all - reg_d:>12d} of {reg_all:6d} ({pct(reg_all - reg_d, reg_all):4.1f}%)')
    p('goal = goal-step records; std = standard-path records left out; rightT = right target; top in disc = solver top')
    p('place within 3 m of the recorded target point; disc / F cells = mean selectable cells in the disc / in all F.')
    p()
    # top outside disc by situation
    p('Top place outside the 3 m disc, by situation (all cells pooled):')
    c = collections.Counter((r['sit'], r['top_in_disc']) for r in rows)
    for sit in SITS:
        p(f'  {sit:20s} outside {c[(sit, False)]:4d} of {c[(sit, False)] + c[(sit, True)]:4d}')
    p()

    # T2 reproduction
    p('== T2. Reproduction of the brief (section 3.4): top / any-of-5 of n, package scorer ==')
    p(f'{"cell":24s} {"brief":>16s} {"now":>16s} {"match":>6s}')
    for (s, m), (t, a, n) in BRIEF.items():
        rs = by[(s, m)]
        top = S(rs, lambda r: r['solver']['45']['hit_pkg'][0])
        any5 = S(rs, lambda r: r['solver']['45']['hit_pkg'][-1])
        p(f'{s + " " + SHORT[m]:24s} {f"{t} / {a} of {n}":>16s} {f"{top} / {any5} of {len(rs)}":>16s} '
          f'{"yes" if (t, a, n) == (top, any5, len(rs)) else "NO":>6s}')
    p('"134/96/109" = top-1 successes (Qwen s19, GLM s19, DeepSeek s20) on realistic goal steps (n 143/141/157);')
    p('"32/23/23" = top-1 successes on the watching goal steps (n 36 each); "123 vs 109 of 157" and "33 vs 23 of 36" =')
    p('DeepSeek any-of-5 vs top-1 (realistic, watching).')
    p()

    # T3 hit@k
    for s, title in SETS:
        p(f'== T3. hit@k, k = 1..5 ({title}), 45 deg: solver vs baselines (expected counts) ==')
        p(f'{"model":9s} {"n":>4s} | {"solver hit@1..5":28s} | {"R5 @1..5":34s} | {"F5 @1..5":34s} | {"AR @1..5":34s} | {"AF @1..5":30s} | P(succ)@5 solver / R5 / F5')
        for m in MODELS:
            rs = by.get((s, m))
            if not rs:
                continue
            d = agg(rs)
            f = lambda xs: ' '.join(f'{x:6.1f}' for x in xs)
            p(f'{SHORT[m]:9s} {d["n"]:4d} | {" ".join(f"{x:4d}" for x in d["hit"]):28s} | {f(d["r5"]):34s} | {f(d["f5"]):34s} | '
              f'{f(d["ar"]):34s} | {f(d["af"]):30s} | {d["p5"]:.1f} / {d["r5_p5"]:.1f} / {d["f5_p5"]:.1f}  (of {d["n"]}; nb {d["nb"]})')
        p()
    p('R5 = random-5, F5 = farthest-point-5 (both in the 3 m disc, facing the recorded target); AR = top place + 4')
    p('random; AF = top place + 4 farthest (deterministic). P(succ)@5 = sum over records of the mean success of the 5')
    p('places (uniform mass). nb = records with draws (>= 5 disc cells).')
    p()

    # T4 recovery
    p('== T4. Runner-up gain and how much the anchored baselines recover (45 deg) ==')
    p(f'{"cell":20s} {"n":>4s} {"top":>5s} {"any5":>5s} {"gain":>5s} | {"AR@5":>7s} {"AR gain":>7s} {"recov":>7s} | {"AF@5":>6s} {"AF gain":>7s} {"recov":>7s} | {"R5@5":>7s} {"F5@5":>6s} | exact R5 / AR')
    recov = {}
    for s, _ in SETS:
        for m in MODELS:
            rs = by.get((s, m))
            if not rs:
                continue
            d = agg(rs)
            top, any5 = d['hit'][0], d['hit'][4]
            gain = any5 - top
            arg, afg = d['ar'][4] - d['hit_nb'][0], d['af'][4] - d['hit_nb'][0]
            gain_nb = d['hit_nb'][4] - d['hit_nb'][0]
            recov[(s, m)] = (gain, pct(arg, gain_nb), pct(afg, gain_nb), d)
            p(f'{s + " " + SHORT[m]:20s} {d["n"]:4d} {top:5d} {any5:5d} {gain:+5d} | {d["ar"][4]:7.1f} {arg:+7.1f} {pct(arg, gain_nb):6.0f}% | '
              f'{d["af"][4]:6.1f} {afg:+7.1f} {pct(afg, gain_nb):6.0f}% | {d["r5"][4]:7.1f} {d["f5"][4]:6.1f} | {d["ex_r5"]:.1f} / {d["ex_ar"]:.1f}')
    p('gain = any-of-5 minus top-1 (solver); AR/AF gain = expected AR/AF any-of-5 minus top-1; recov = AR or AF gain as a')
    p('share of the solver\'s gain (over records with draws; nan where the solver gains nothing). exact = hypergeometric')
    p('expectation of R5@5 and AR@5 (a check on the draws).')
    p()

    # T5 lift
    p('== T5. Lift: top-1 over the region\'s share of selectable cells (ratio of sums) ==')
    p(f'{"cell":20s} {"n":>4s} {"top1":>5s} {"rate":>6s} | {"rho_disc":>8s} {"rho_room":>8s} | {"Lift_disc":>9s} {"Lift_room":>9s} {"chance ratio":>12s} | {"sigma_disc":>10s} {"Lift_sigma":>10s}')
    lifts = {}
    for s, _ in SETS:
        for m in MODELS:
            rs = by.get((s, m))
            if not rs:
                continue
            top = S(rs, lambda r: r['solver']['45']['hit_pkg'][0])
            rd, rr, sg = S(rs, lambda r: r['rho_disc']), S(rs, lambda r: r['rho_room']), S(rs, lambda r: r['sigma_disc'])
            ld, lr, cr = top / rd, top / rr, rd / rr
            lifts[(s, m)] = (ld, lr, cr)
            n = len(rs)
            p(f'{s + " " + SHORT[m]:20s} {n:4d} {top:5d} {top / n:6.3f} | {rd / n:8.4f} {rr / n:8.4f} | {ld:9.2f} {lr:9.2f} {cr:12.2f} | {sg / n:10.4f} {top / sg:10.2f}')
    p('rho_disc / rho_room = mean share of the disc\'s / all F cells inside the matched target\'s acceptable region (0 for a')
    p('wrong target); Lift_disc = sum top-1 / sum rho_disc; Lift_room = sum top-1 / sum rho_room; chance ratio = sum')
    p('rho_disc / sum rho_room. sigma_disc = share of disc cells whose face-target pose succeeds (45 deg).')
    p()
    p('Lift by situation (all cells of a set pooled):')
    for s, title in SETS:
        for sit in SITS:
            rs = [r for r in rows if r['set'] == s and r['sit'] == sit]
            if not rs:
                continue
            top = S(rs, lambda r: r['solver']['45']['hit_pkg'][0])
            rd, rr = S(rs, lambda r: r['rho_disc']), S(rs, lambda r: r['rho_room'])
            right = [r for r in rs if r['target_ok']]
            p(f'  {s:6s} {sit:20s} n {len(rs):4d} top1 {top:4d}  rho_disc {rd / len(rs):.4f} (right target {S(right, lambda r: r["rho_disc"]) / max(len(right), 1):.4f})  '
              f'rho_room {rr / len(rs):.4f} (right target {S(right, lambda r: r["rho_room"]) / max(len(right), 1):.4f})  Lift_disc {top / rd:.2f}  Lift_room {top / rr:.2f}  ratio {rd / rr:.2f}')
    p()

    # T6 forbidden
    p('== T6. Mass in each forbidden kind: share of the 5 places inside it, the top place, and R5 (disc share) ==')
    p(f'{"cell":20s} {"kind":14s} {"n":>4s} {"set mass":>9s} {"top":>6s} {"R5":>7s}')
    for s, _ in SETS:
        for m in MODELS:
            rs = by.get((s, m))
            if not rs:
                continue
            kinds = sorted({k for r in rs for k in r['forb']})
            for kd in kinds:
                ks = [r for r in rs if kd in r['forb']]
                p(f'{s + " " + SHORT[m]:20s} {kd:14s} {len(ks):4d} {S(ks, lambda r: r["forb"][kd]["set"]) / len(ks):9.4f} '
                  f'{S(ks, lambda r: r["forb"][kd]["top"]) / len(ks):6.3f} {S(ks, lambda r: r["forb"][kd]["r5"]) / len(ks):7.4f}')
    p('set mass = mean over the records whose episode has that kind of the share of the 5 places inside it; top = share')
    p('of top places inside it; R5 = expected share for random-5 in the disc (= the disc share of cells inside it).')
    p()

    # T7 jitter
    p('== T7. Stability under 5 cm jitter (8 directions, heading kept; package verdicts at 45 deg) ==')
    p(f'{"cell":20s} {"n":>4s} {"top1":>5s} {"top1 jit mean":>13s} {"top flips":>9s} {"top kept all 8":>14s} | {"hit@5":>5s} {"hit@5 jit mean":>14s} {"hit@5 worst dir":>15s} {"any place flips":>15s}')
    for s, _ in SETS:
        for m in MODELS:
            rs = by.get((s, m))
            if not rs:
                continue
            j = lambda f: S(rs, lambda r: r['jitter']['45'][f])
            p(f'{s + " " + SHORT[m]:20s} {len(rs):4d} {S(rs, lambda r: r["solver"]["45"]["hit_pkg"][0]):5d} {j("top_mean"):13.1f} {j("top_flip"):9d} {j("top_all_kept"):14d} | '
              f'{S(rs, lambda r: r["solver"]["45"]["hit_pkg"][-1]):5d} {j("hit5_mean"):14.1f} {j("hit5_min"):15d} {j("set_flip"):15d}')
    p('top1 jit mean = sum over records of the mean top-1 success over the 8 moved copies; top flips = records where any')
    p('direction changes the top-1 verdict; top kept all 8 = top-1 successes that stay successes in all 8; hit@5 jit mean')
    p('/ worst dir = any-of-5 with all places moved by the same vector, mean / sum of per-record minimum over directions;')
    p('any place flips = records where some place\'s verdict changes in some direction.')
    p()

    # T8 heading curve
    p('== T8. Heading-tolerance curve: 30 / 45 / 60 deg ==')
    p(f'{"cell":20s} {"n":>4s} | {"top1 30/45/60":>16s} | {"hit@5 30/45/60":>16s} | {"P@5 30/45/60":>20s} | {"R5@5 30/45/60":>20s} | {"AR@5 30/45/60":>20s} | AR recov 30/45/60')
    for s, _ in SETS:
        for m in MODELS:
            rs = by.get((s, m))
            if not rs:
                continue
            ds = {t: agg(rs, t) for t in ('30', '45', '60')}
            rc = []
            for t in ('30', '45', '60'):
                d = ds[t]
                g = d['hit_nb'][4] - d['hit_nb'][0]
                rc.append(pct(d['ar'][4] - d['hit_nb'][0], g))
            p(f'{s + " " + SHORT[m]:20s} {len(rs):4d} | {"/".join(str(ds[t]["hit"][0]) for t in ds):>16s} | {"/".join(str(ds[t]["hit"][4]) for t in ds):>16s} | '
              f'{"/".join(f"{ds[t]["p5"]:.1f}" for t in ds):>20s} | {"/".join(f"{ds[t]["r5"][4]:.1f}" for t in ds):>20s} | '
              f'{"/".join(f"{ds[t]["ar"][4]:.1f}" for t in ds):>20s} | {"/".join(f"{x:.0f}%" for x in rc)}')
    p('Solver at 45 here is the replica\'s verdict on the recorded headings, equal to the package\'s (T2 checks).')
    p()

    # T9 per situation
    p('== T9. By situation (cells of a set pooled): solver vs R5, and in-region alone (heading aside) ==')
    p(f'{"set":6s} {"situation":20s} {"n":>4s} {"rightT":>6s} | {"top1":>5s} {"any5":>5s} | {"R5@5":>7s} {"F5@5":>6s} {"AR@5":>7s} {"AF@5":>6s} | in-region: {"solver any5":>11s} {"R5@5":>7s} {"R5@5 / rightT":>13s}')
    for s, _ in SETS:
        for sit in SITS:
            rs = [r for r in rows if r['set'] == s and r['sit'] == sit]
            if not rs:
                continue
            d = agg(rs)
            nb = [r for r in rs if hasbase(r)]
            right = sum(r['target_ok'] for r in rs)
            ir5 = S(nb, lambda r: r['base']['inreg']['r5'][4])
            p(f'{s:6s} {sit:20s} {len(rs):4d} {right:6d} | {d["hit"][0]:5d} {d["hit"][4]:5d} | {d["r5"][4]:7.1f} {d["f5"][4]:6.1f} {d["ar"][4]:7.1f} {d["af"][4]:6.1f} | '
              f'{S(rs, lambda r: r["solver_inreg_hit"][-1]):11d} {ir5:7.1f} {pct(ir5, right):12.1f}%')
    p('in-region: any of the 5 places inside the acceptable region (target matched), heading and reachability aside.')
    p()
    p('Per model at groups, R5 in-region any-of-5 as a share of right-target group records (the skeptic\'s 85-95% guess):')
    for s, _ in SETS:
        for m in MODELS:
            rs = [r for r in by.get((s, m), []) if r['sit'] == 'conversation_group' and hasbase(r)]
            if not rs:
                continue
            right = sum(r['target_ok'] for r in rs)
            ir5 = S(rs, lambda r: r['base']['inreg']['r5'][4])
            s5 = S(rs, lambda r: r['base']['45']['r5'][4])
            p(f'  {s:6s} {SHORT[m]:9s} n {len(rs):3d} right {right:3d}  R5 in-region {ir5:6.1f} ({pct(ir5, right):5.1f}%)  R5 success {s5:6.1f} ({pct(s5, right):5.1f}%)  '
              f'solver any5 {S(rs, lambda r: r["solver"]["45"]["hit"][-1]):3d}')
    p()

    # T10 criteria
    p('== T10. Pre-registered criteria (METRIC_NOTE.md) ==')
    s1 = all((t, a, n) == (S(by[k], lambda r: r['solver']['45']['hit_pkg'][0]), S(by[k], lambda r: r['solver']['45']['hit_pkg'][-1]), len(by[k]))
             for k, (t, a, n) in BRIEF.items())
    p(f'S1 exact reproduction (6 cells, top and any-of-5 and n): {"MET" if s1 else "NOT MET"}')
    for k in (('old', 'dsv41'), ('watch', 'dsv41')):
        g, ra, rf, d = recov[k]
        p(f'   {k[0]} DeepSeek: gain {g:+d}; AR recovers {ra:.0f}%; AF recovers {rf:.0f}%; solver any5 {d["hit"][4]} vs AF@5 {d["af"][4]:.1f}')
    s2 = all(recov[k][1] <= 50 for k in (('old', 'dsv41'), ('watch', 'dsv41')))
    p(f'S2 AR recovers <= 50% on both DeepSeek cells: {"MET" if s2 else "NOT MET"}')
    s3 = all(recov[k][3]['hit'][4] > recov[k][3]['af'][4] for k in (('old', 'dsv41'), ('watch', 'dsv41')))
    p(f'S3 solver any-of-5 > AF any-of-5 on both DeepSeek cells: {"MET" if s3 else "NOT MET"}')
    prim = [k for k in lifts if k[0] in ('old', 'watch')]
    s4a = [k for k in prim if lifts[k][0] >= 3]
    s4b = [k for k in prim if lifts[k][2] >= 2]
    p(f'S4 Lift_disc >= 3 in {len(s4a)} of {len(prim)} primary cells; chance ratio >= 2 in {len(s4b)} of {len(prim)}: '
      f'{"MET" if len(s4a) == len(prim) and len(s4b) == len(prim) else "NOT MET"}')
    for k in sorted(lifts, key=lambda k: (k[0], MODELS.index(k[1]))):
        p(f'   {k[0]:6s} {SHORT[k[1]]:9s} Lift_disc {lifts[k][0]:.2f}  Lift_room {lifts[k][1]:.2f}  chance ratio {lifts[k][2]:.2f}')
    k1_ar = all(recov[k][1] >= 80 for k in (('old', 'dsv41'), ('watch', 'dsv41')))
    k1_af = all(recov[k][2] >= 80 for k in (('old', 'dsv41'), ('watch', 'dsv41')))
    p(f'K1 an anchored baseline recovers >= 80% on both DeepSeek cells: AR {"yes" if k1_ar else "no"}, AF {"yes" if k1_af else "no"} '
      f'-> {"FIRES" if k1_ar or k1_af else "does not fire"}')
    k2 = []
    for k in (('old', 'glm53'), ('old', 'dsv41')):
        d = recov[k][3]
        k2.append(d['r5'][4] >= d['hit'][4])
        p(f'   {k[0]} {SHORT[k[1]]}: R5@5 {d["r5"][4]:.1f} vs solver any5 {d["hit"][4]} (n {d["n"]}, draws on {d["nb"]})')
    p(f'K2 R5 >= solver any-of-5 for GLM or DeepSeek (old-wording realistic): {"FIRES" if any(k2) else "does not fire"}')
    p('   beside it (not the pre-registered cells):')
    for k in (('new', 'glm53'), ('new', 'dsv41'), ('watch', 'glm53'), ('watch', 'dsv41')):
        d = recov[k][3]
        p(f'   {k[0]} {SHORT[k[1]]}: R5@5 {d["r5"][4]:.1f} vs solver any5 {d["hit"][4]}; AR recovers {recov[k][1]:.0f}%, AF {recov[k][2]:.0f}% of gain {recov[k][0]:+d}')
    p()

    # T11 descriptive uncertainty (not pre-registered)
    import numpy as np
    p('== T11. Descriptive only (not pre-registered): layout-clustered bootstrap of the recovery shares, 45 deg ==')
    p('Resample the cell\'s layouts (scene id without its arm; a watching scene is its own layout) with replacement (B 2000, seed 14); recovery = sum(AR or AF any5 - top1) / sum(solver')
    p('any5 - top1) over the resampled records; resamples with no solver gain are dropped (count shown).')
    for k in (('old', 'dsv41'), ('watch', 'dsv41'), ('old', 'glm53'), ('watch', 'glm53'), ('new', 'dsv41'), ('new', 'glm53'),
              ('watch', 'qwen38_low')):
        rs = [r for r in by.get(k, []) if hasbase(r)]
        if not rs:
            continue
        # layout = scene id without its arm (rerun/stats); a development seed's scene is its own layout (the seeds
        # reuse scene ids for different layouts)
        lay = lambda r: (r['arm'] + '/' + r['scene']) if r['arm'].startswith('dev') else re.sub(r'^(real|staffed|plain)_', '', r['scene'])
        scenes = sorted({lay(r) for r in rs})
        idx = {sc: [i for i, r in enumerate(rs) if lay(r) == sc] for sc in scenes}
        g = np.array([r['solver']['45']['hit'][4] - r['solver']['45']['hit'][0] for r in rs], dtype=float)
        ga = np.array([r['base']['45']['ar'][4] - r['solver']['45']['hit'][0] for r in rs])
        gf = np.array([r['base']['45']['af'][4] - r['solver']['45']['hit'][0] for r in rs])
        rng = np.random.default_rng(14)
        ra, rf, dropped = [], [], 0
        for _ in range(2000):
            pick = rng.choice(len(scenes), size=len(scenes), replace=True)
            ii = np.concatenate([idx[scenes[j]] for j in pick])
            if g[ii].sum() <= 0:
                dropped += 1
                continue
            ra.append(ga[ii].sum() / g[ii].sum())
            rf.append(gf[ii].sum() / g[ii].sum())
        if ra:
            p(f'  {k[0]:6s} {SHORT[k[1]]:9s} layouts {len(scenes):3d} records {len(rs):4d} gain {int(g.sum()):+d}: AR recovery '
              f'{100 * ga.sum() / max(g.sum(), 1e-9):5.1f}% [{100 * np.percentile(ra, 2.5):5.1f}, {100 * np.percentile(ra, 97.5):5.1f}]  '
              f'AF {100 * gf.sum() / max(g.sum(), 1e-9):5.1f}% [{100 * np.percentile(rf, 2.5):5.1f}, {100 * np.percentile(rf, 97.5):5.1f}]  dropped {dropped}')
        else:
            p(f'  {k[0]:6s} {SHORT[k[1]]:9s} no solver gain in any resample')
    p()

    # T12 post-hoc sensitivity (not pre-registered): draws over the whole room instead of the 3 m disc
    p('== T12. Post-hoc sensitivity, NOT pre-registered: the 3 m disc against the whole room (all F cells), in-region proxy ==')
    p('Why: the frozen disc leaves out much of the region where it lies far from the target point (T1: 90% at watching, 16%')
    p('at realistic; the top place lies outside the disc in 100 of 108 watching and 470 of 790 line records). No new')
    p('parameter: the whole room is the frozen lift denominator. Measured in-region (target matched, heading and')
    p('reachability aside), because per-cell success was computed in the disc only; T5 shows sigma_disc ~ rho_disc.')
    p('Whole-room values are exact hypergeometric expectations from the region and floor cell counts.')
    p(f'{"cell":20s} {"n":>4s} | solver in-region: {"top":>4s} {"any5":>4s} {"gain":>5s} | {"R5 disc":>7s} {"R5 room":>7s} | '
      f'{"AR disc":>7s} {"recov":>6s} | {"AR room":>7s} {"recov":>6s}')

    def none_exact(n, s, k):
        q = 1.0
        for i in range(k):
            if n - i <= 0:
                return 0.0
            q *= max(n - s - i, 0) / (n - i)
        return q
    for s_, _ in SETS:
        for m in MODELS:
            rs = [r for r in by.get((s_, m), []) if hasbase(r)]
            if not rs:
                continue
            top = S(rs, lambda r: r['solver_inreg_hit'][0])
            any5 = S(rs, lambda r: r['solver_inreg_hit'][-1])
            r5d = S(rs, lambda r: r['base']['inreg']['r5'][4])
            ard = S(rs, lambda r: r['base']['inreg']['ar'][4])
            r5r = S(rs, lambda r: 1 - none_exact(r['n_sel'], r['region_sel'], 5))
            arr = S(rs, lambda r: 1.0 if r['solver_inreg_hit'][0] else 1 - none_exact(r['n_sel'] - 1, r['region_sel'], 4))
            g = any5 - top
            p(f'{s_ + " " + SHORT[m]:20s} {len(rs):4d} | {"":17s} {top:4d} {any5:4d} {g:+5d} | {r5d:7.1f} {r5r:7.1f} | '
              f'{ard:7.1f} {pct(ard - top, g):5.0f}% | {arr:7.1f} {pct(arr - top, g):5.0f}%')
    with open(f'{HERE}/tables.txt', 'w') as f:
        f.write('\n'.join(OUT) + '\n')
    print('\n'.join(OUT))


if __name__ == '__main__':
    main()
