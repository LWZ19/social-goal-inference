"""I46 statistics, re-computed through research_harness only (no model calls, no tuning).

Claims S1-S4 (old wording, the judge's clus.py subsets):
  Qwen  s16b {arm}_qwen38_low_fsc (E) vs s19 {arm}_qwen38_low_con (C)
  GLM   s16b {arm}_glm53_fsc      (E) vs s19 {arm}_glm53_con      (C)
  DS    s20  {arm}_dsv41_fsc      (E) vs s20 {arm}_dsv41_con      (C)
  arms real, staffed, plain; records paired by (scene_id, episode_id, trial); layout = scene id without its arm prefix.
NEW rows (new wording): Qwen s21 cot/fsc/con; GLM, DeepSeek, Haiku, Sonnet, Gemini s23 cot/fsc/con; C vs E and C vs R.

Test: exact sign-flip (randomisation) test with the layout as the unit. Per layout, d_l = mean over its paired records
of (success_C - success_X). Statistic T = sum_l d_l (equivalently the mean over layouts). All 2^L sign patterns
are enumerated (L = 14 here; zero layouts are kept, which leaves p unchanged versus dropping them).
  two-sided p = #{s : |sum s_l d_l| >= |T|} / 2^L   (the judge's clus.py definition)
  one-sided p = #{s : sum s_l d_l >= T} / 2^L        (alternative C > X)
Every layout has 18 pairs (3 arms x 2 episodes x 3 trials), so d_l * 18 is an integer and the layout-sum and
layout-mean statistics give the same p.
"""
import collections
import itertools
import math
import random
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402

OUT = Path(__file__).resolve().parent
ARMS = ('real', 'staffed', 'plain')
LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    LINES.append(s)


_CACHE = {}


def load(stage, arm, model, method):
    key = (stage, arm, model, method)
    if key not in _CACHE:
        rs = H.runs(stage=stage, arm=arm, model=model, method=method)
        assert len(rs) == 1, (key, rs)
        tally = collections.Counter()
        recs = list(H.records(rs[0], tally=tally))
        assert set(tally) == {'allowed'}, (key, tally)
        _CACHE[key] = (rs[0].folder, recs)
    return _CACHE[key]


def layout_of(scene_id):
    return re.sub(r'^[a-z]+_', '', scene_id)


def success(r):
    return int(bool(r.score['success']))


def pairs(spec_c, spec_x):
    """spec = (stage, model, method). Pairs (layout, situation, arm, sC, sX) over the three arms."""
    out, folders = [], []
    for arm in ARMS:
        fc, C = load(spec_c[0], arm, spec_c[1], spec_c[2])
        fx, X = load(spec_x[0], arm, spec_x[1], spec_x[2])
        folders += [fc, fx]
        kx = {(r.scene_id, r.episode_id, r.trial): r for r in X}
        assert len(kx) == len(X)
        n_unpaired = 0
        for r in C:
            k = (r.scene_id, r.episode_id, r.trial)
            if k not in kx:
                n_unpaired += 1
                continue
            assert kx[k].situation == r.situation
            out.append((layout_of(r.scene_id), r.situation, arm, success(r), success(kx[k])))
        assert n_unpaired == 0 and len(C) == len(X), (spec_c, spec_x, arm, n_unpaired, len(C), len(X))
    return out, folders


SIGNS = {}


def sign_matrix(n):
    if n not in SIGNS:
        SIGNS[n] = np.array(list(itertools.product((1, -1), repeat=n)), dtype=np.int64)
    return SIGNS[n]


def signflip_int(v):
    """Exact sign-flip p (two-sided, one-sided C>X) for integer layout statistics v (DP over sums: exact for any n)."""
    v = [int(x) for x in v]
    T = sum(v)
    tot = sum(abs(x) for x in v)
    cnt = np.zeros(2 * tot + 1, dtype=np.float64)   # index = sum + tot
    cnt[tot] = 1.0
    for x in v:
        x = abs(x)
        if x == 0:
            cnt = cnt * 2.0
            continue
        new = np.zeros_like(cnt)
        new[x:] += cnt[:len(cnt) - x]
        new[:len(cnt) - x] += cnt[x:]
        cnt = new
    sums = np.arange(-tot, tot + 1)
    allp = 2.0 ** len(v)
    p2 = cnt[np.abs(sums) >= abs(T)].sum() / allp
    p1 = cnt[sums >= T].sum() / allp
    return p2, p1


def signflip_enum(d):
    """Same test by full enumeration over float layout means (cross-check of signflip_int, the judge's way)."""
    d = np.asarray(d, dtype=float)
    S = sign_matrix(len(d)) @ d
    T = d.sum()
    return float(np.mean(np.abs(S) >= abs(T) - 1e-12)), float(np.mean(S >= T - 1e-12))


def binom_sign(npos, nneg):
    n = npos + nneg
    if n == 0:
        return 1.0, 1.0
    tail = lambda k: sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n  # noqa: E731
    p1 = tail(npos)
    p2 = min(1.0, 2 * tail(max(npos, nneg)))
    return p2, p1


def t_pdf(x, df):
    return math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi) * (1 + x * x / df) ** (-(df + 1) / 2)


def t_cdf(x, df, steps=20000):
    if x == 0:
        return 0.5
    a, b = 0.0, abs(x)
    h = (b - a) / steps
    s = t_pdf(a, df) + t_pdf(b, df) + sum((4 if i % 2 else 2) * t_pdf(a + i * h, df) for i in range(1, steps))
    area = s * h / 3
    return 0.5 + area if x > 0 else 0.5 - area


def t_quant(q, df):
    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if t_cdf(mid, df, 4000) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def icc_oneway(groups):
    """The judge's ICC(1) of a 0/1 outcome by layout (one-way ANOVA)."""
    g = list(groups.values())
    k = st.mean(len(v) for v in g)
    gm = st.mean(x for v in g for x in v)
    msb = sum(len(v) * (st.mean(v) - gm) ** 2 for v in g) / (len(g) - 1)
    msw = sum((x - st.mean(v)) ** 2 for v in g for x in v) / (sum(len(v) for v in g) - len(g))
    den = msb + (k - 1) * msw
    return (msb - msw) / den if den > 0 else float('nan')


T975_13 = None


def analyse(name, P, rng_boot=None):
    """Layout-clustered statistics of pairs P. Returns a dict."""
    lays = sorted(set(p[0] for p in P))
    per = {l: [p for p in P if p[0] == l] for l in lays}
    npairs = {l: len(v) for l, v in per.items()}
    sit = {}
    for l in lays:
        ss = set(p[1] for p in per[l])
        assert len(ss) == 1, (l, ss)
        sit[l] = ss.pop()
    dsum = {l: sum(p[3] - p[4] for p in per[l]) for l in lays}           # integer per layout
    csum = {l: sum(p[3] for p in per[l]) for l in lays}
    xsum = {l: sum(p[4] for p in per[l]) for l in lays}
    dm = {l: dsum[l] / npairs[l] for l in lays}
    equal = len(set(npairs.values())) == 1
    nz = [l for l in lays if dsum[l] != 0]
    npos = sum(dsum[l] > 0 for l in lays)
    nneg = sum(dsum[l] < 0 for l in lays)
    p2e, p1e = signflip_enum([dm[l] for l in lays])
    if equal:
        p2, p1 = signflip_int([dsum[l] for l in lays])
        assert abs(p2 - p2e) < 1e-12 and abs(p1 - p1e) < 1e-12, (p2, p2e, p1, p1e)
    else:
        p2, p1 = p2e, p1e
    sp2, sp1 = binom_sign(npos, nneg)
    vals = np.array([dm[l] for l in lays])
    L = len(vals)
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1))
    tq = t_quant(0.975, L - 1)
    se = sd / math.sqrt(L)
    tci = (mean - tq * se, mean + tq * se)
    tstat = mean / se if se > 0 else float('inf')
    tp2 = 2 * (1 - t_cdf(abs(tstat), L - 1)) if se > 0 else 0.0
    out = dict(name=name, n=len(P), L=L, lays=lays, sit=sit, dsum=dsum, csum=csum, xsum=xsum, dm=dm,
               npairs=npairs, nz=len(nz), npos=npos, nneg=nneg, p2=p2, p1=p1, sp2=sp2, sp1=sp1,
               C=sum(p[3] for p in P), X=sum(p[4] for p in P), mean=mean, sd=sd, tq=tq, tci=tci, tstat=tstat,
               tp2=tp2, min_p2=2 / 2 ** len(nz) if nz else 1.0, min_p1=1 / 2 ** len(nz) if nz else 1.0,
               scale=sum(npairs.values()))
    if rng_boot is not None:
        bs = rng_boot.choice(vals, size=(10000, L), replace=True).mean(axis=1)
        out['pboot'] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    out['icc_c'] = icc_oneway({l: [p[3] for p in per[l]] for l in lays})
    out['icc_x'] = icc_oneway({l: [p[4] for p in per[l]] for l in lays})
    return out


def fmt_p(p):
    return f'{p:.5g}'


def report_test(a, xname):
    say(f"  pairs n={a['n']} layouts L={a['L']} (pairs per layout: {sorted(set(a['npairs'].values()))}); "
        f"C={a['C']} {xname}={a['X']} (C-{xname} = {a['C'] - a['X']:+d} of {a['n']})")
    say(f"  layouts C>{xname}: {a['npos']}, C<{xname}: {a['nneg']}, tied: {a['L'] - a['nz']}; non-tied = {a['nz']}")
    say(f"  exact sign-flip, statistic = sum of layout mean diffs (2^{a['L']} patterns): two-sided p = {fmt_p(a['p2'])} "
        f"(= {round(a['p2'] * 2 ** a['nz'])}/{2 ** a['nz']} over non-tied), one-sided (C>{xname}) p = {fmt_p(a['p1'])}; "
        f"smallest possible with {a['nz']} non-tied: two-sided {fmt_p(a['min_p2'])}, one-sided {fmt_p(a['min_p1'])}")
    say(f"  sign test on layouts: two-sided {fmt_p(a['sp2'])}, one-sided {fmt_p(a['sp1'])}")
    lo, hi = a['tci']
    s = a['scale']
    say(f"  layout mean diff = {a['mean']:+.4f} (x{s} = {a['mean'] * s:+.1f} records), sd {a['sd']:.4f}, "
        f"t(df={a['L'] - 1}) 95% CI [{lo:+.4f}, {hi:+.4f}] = [{lo * s:+.1f}, {hi * s:+.1f}] records, "
        f"t = {a['tstat']:.2f}, two-sided t p = {fmt_p(a['tp2'])}")
    if 'pboot' in a:
        bl, bh = a['pboot']
        say(f"  percentile bootstrap of the layout mean (10,000, seed 1): [{bl:+.4f}, {bh:+.4f}] = [{bl * s:+.1f}, {bh * s:+.1f}] records")
    say(f"  ICC(1) by layout (judge's one-way ANOVA): C {a['icc_c']:.3f}, {xname} {a['icc_x']:.3f}")
    say('  per layout (situation: C-' + xname + ' summed over its pairs): ' +
        ', '.join(f"{l}[{a['sit'][l]}]:{a['dsum'][l]:+d}" for l in a['lays']))


def power_exact(dvals_int, n, draws, rng, alpha=0.05):
    """Resample n layouts with replacement from the observed integer layout differences; exact sign-flip test."""
    vals = np.asarray(dvals_int)
    rej2 = rej1 = 0
    for _ in range(draws):
        smp = rng.choice(vals, size=n, replace=True)
        p2, p1 = signflip_int(smp)
        rej2 += p2 <= alpha
        rej1 += p1 <= alpha
    return rej2 / draws, rej1 / draws


def judge_power_replay(order, results):
    """clus.py's power loop replayed exactly: random.seed(0) once, models in its order, n in (8,14,20,30), B=300
    resamples of layouts, 400 Monte Carlo sign flips each, two-sided, reject when p_MC < 0.05."""
    random.seed(0)
    out = {}
    for m in order:
        a = results[m]
        lays = a['lays']
        dm = a['dm']
        for n in (8, 14, 20, 30):
            pw = 0
            B = 300
            for _ in range(B):
                smp = [dm[random.choice(lays)] for _ in range(n)]
                o = abs(sum(smp))
                c = 0
                for _ in range(400):
                    if abs(sum(v * random.choice((1, -1)) for v in smp)) >= o - 1e-12:
                        c += 1
                pw += (c / 400) < 0.05
            out[(m, n)] = round(pw / B, 2)
    return out


def main():
    say('I46 statistics re-run (research_harness only). Definitions in the module docstring of stats.py.')
    say('=' * 110)
    say('PART 1. Old wording, C (con) vs E (fsc), the judge clus.py subsets')
    old = {'qwen': (('s19', 'qwen38_low', 'con'), ('s16b', 'qwen38_low', 'fsc')),
           'glm': (('s19', 'glm53', 'con'), ('s16b', 'glm53', 'fsc')),
           'ds': (('s20', 'dsv41', 'con'), ('s20', 'dsv41', 'fsc'))}
    res_old = {}
    rng_b = np.random.default_rng(1)
    for m, (sc, sx) in old.items():
        P, folders = pairs(sc, sx)
        say(f'-- {m}: runs ' + ', '.join(folders))
        a = analyse(m, P, rng_b)
        res_old[m] = a
        report_test(a, 'E')

    say('')
    say('S3. Layouts per situation (old-wording subsets; the same 14 layouts in every run)')
    a = res_old['qwen']
    by_sit = collections.defaultdict(list)
    for l in a['lays']:
        by_sit[a['sit'][l]].append(l)
    for s_, ls in sorted(by_sit.items()):
        k = len(ls)
        say(f'  {s_}: {k} layouts -> smallest exact sign-flip p (all {k} non-tied, all one sign): two-sided {fmt_p(2 / 2 ** k)}, '
            f'one-sided {fmt_p(1 / 2 ** k)}  [{", ".join(ls)}]')
    say('  Per-situation C vs E within each model (descriptive):')
    for m, a in res_old.items():
        parts = []
        for s_, ls in sorted(by_sit.items()):
            v = [a['dsum'][l] for l in ls]
            p2, p1 = signflip_int(v)
            parts.append(f'{s_}: diffs {v} p2={fmt_p(p2)} p1={fmt_p(p1)}')
        say(f'   {m}: ' + '; '.join(parts))
    for m, a in res_old.items():
        for l in a['lays']:
            assert a['sit'][l] == res_old['qwen']['sit'][l]

    say('')
    say('S4. Power at n fresh layouts. Design A (this re-run): draw n layouts with replacement from the 14 observed '
        'per-layout differences (integer sums over 18 pairs), exact sign-flip test, alpha 0.05, 10,000 draws, '
        'numpy default_rng(seed 0); reject when p <= 0.05.')
    rng = np.random.default_rng(0)
    powA = {}
    for m, a in res_old.items():
        v = [a['dsum'][l] for l in a['lays']]
        row = []
        for n in (8, 14, 20, 30):
            p2pow, p1pow = power_exact(v, n, 10000, rng)
            powA[(m, n)] = (p2pow, p1pow)
            row.append(f'n={n}: two-sided {p2pow:.3f}, one-sided {p1pow:.3f}')
        say(f'  {m}: ' + ' | '.join(row))
    say('  Design B (the judge clus.py loop replayed exactly: random.seed(0), B=300 resamples, 400 MC flips, '
        'two-sided, p_MC < 0.05):')
    powB = judge_power_replay(('qwen', 'glm', 'ds'), res_old)
    for m in ('qwen', 'glm', 'ds'):
        say(f'  {m}: ' + ' | '.join(f'n={n}: {powB[(m, n)]}' for n in (8, 14, 20, 30)))

    say('')
    say('=' * 110)
    say('PART 2 (NEW rows). New wording: C (con) vs E (fsc) and C vs R (cot); Qwen s21, others s23')
    new = {'qwen38_low': 's21', 'glm53': 's23', 'dsv41': 's23', 'haiku55': 's23', 'sonnet55_low': 's23',
           'gemini38_low': 's23'}
    res_new = {}
    rng_b = np.random.default_rng(1)
    for m, stg in new.items():
        for xm, xname in (('fsc', 'E'), ('cot', 'R')):
            P, folders = pairs((stg, m, 'con'), (stg, m, xm))
            say(f'-- {m} C vs {xname}: runs ' + ', '.join(folders))
            a = analyse(f'{m} C vs {xname}', P, rng_b)
            res_new[(m, xname)] = a
            report_test(a, xname)
    say('')
    say('Totals of 252 (R / E / C) per model at the new wording:')
    for m in new:
        R = res_new[(m, 'R')]['X']
        E = res_new[(m, 'E')]['X']
        C = res_new[(m, 'E')]['C']
        assert C == res_new[(m, 'R')]['C']
        say(f'  {m}: {R} / {E} / {C}')
    say('')
    say('Summary of layout-clustered exact sign-flip p (new wording):')
    say(f"  {'model':14s} {'cmp':6s} {'C-X':>5s} {'+/-/0':>8s} {'p two':>9s} {'p one':>9s} {'t 95% CI (records of 252)':>28s}")
    for (m, xn), a in res_new.items():
        lo, hi = a['tci']
        say(f"  {m:14s} C vs {xn:2s} {a['C'] - a['X']:+5d} {a['npos']:>2d}/{a['nneg']}/{a['L'] - a['nz']:<2d} "
            f"{fmt_p(a['p2']):>9s} {fmt_p(a['p1']):>9s}   [{lo * 252:+.1f}, {hi * 252:+.1f}]")
    say(f"  largest two-sided p over the 12: {fmt_p(max(a['p2'] for a in res_new.values()))}; "
        f"largest one-sided p: {fmt_p(max(a['p1'] for a in res_new.values()))}")
    for xn in ('E', 'R'):
        cells_below = [(m, l, res_new[(m, xn)]['csum'][l], res_new[(m, xn)]['xsum'][l])
                       for m in new for l in res_new[(m, xn)]['lays']
                       if res_new[(m, xn)]['csum'][l] < res_new[(m, xn)]['xsum'][l]]
        ties = sum(res_new[(m, xn)]['csum'][l] == res_new[(m, xn)]['xsum'][l] for m in new for l in res_new[(m, xn)]['lays'])
        ncells = sum(len(res_new[(m, xn)]['lays']) for m in new)
        say(f'  C below {xn} in {len(cells_below)} of {ncells} model x layout cells (successes summed over 18 pairs), '
            f'tied in {ties}: {cells_below}')
    say('')
    say('Old-wording C below E cells (for comparison):')
    cb = [(m, l, a['csum'][l], a['xsum'][l]) for m, a in res_old.items() for l in a['lays'] if a['csum'][l] < a['xsum'][l]]
    say(f'  {len(cb)} of {sum(len(a["lays"]) for a in res_old.values())}: {cb}')

    (OUT / 'stats_output.txt').write_text('\n'.join(LINES) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
