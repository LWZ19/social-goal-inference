"""Independent check of i46_stats.py (own loading, own formulas; does not import rerun/stats/stats.py).

1 Qwen 9-draw pool batch totals and the pooled batch SD (R, E).
2 ICC(1) by layout from per-layout success counts (binary-data form of the one-way ANOVA), all 27 cells.
3 t(13) intervals for C-E and C-R at the new wording, with a t quantile from the regularised incomplete beta.
4 Power at n = 14 for every new-wording C-E / C-R cell, by full enumeration of sign patterns, python random, 2,000
  draws (agreement within Monte Carlo error with power_table.txt).
"""
import collections
import itertools
import math
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402

OUT = Path(__file__).resolve().parent
LINES = []


def say(s=''):
    print(s, flush=True)
    LINES.append(s)


def recs(stage, model, method, trials=None):
    out = []
    for arm in ('real', 'staffed', 'plain'):
        rs = H.runs(stage=stage, arm=arm, model=model, method=method)
        assert len(rs) == 1
        for r in H.records(rs[0]):
            if trials is None or r.trial in trials:
                out.append((arm, r.scene_id, r.episode_id, r.trial, int(bool(r.score['success']))))
    return out


def lay(sid):
    return re.sub(r'^[a-z]+_', '', sid)


# 1
say('1. Qwen 9-draw pool')
var = []
for meth in ('cot', 'fsc'):
    tots = [sum(x[4] for x in recs('s16b', 'qwen38_low', meth)),
            sum(x[4] for x in recs('s22', 'qwen38_low', meth, {3, 4, 5})),
            sum(x[4] for x in recs('s21', 'qwen38_low', meth))]
    m = sum(tots) / 3
    v = sum((t - m) ** 2 for t in tots) / 2
    var.append(v)
    say(f'  {meth}: totals {tots}, SD {math.sqrt(v):.3f}')
say(f'  pooled SD {math.sqrt(sum(var) / 2):.3f}')

# 2
say('2. ICC(1) from per-layout counts')


def icc_counts(cnt, k=18):
    p = np.array(cnt, dtype=float) / k
    g = len(p)
    pb = p.mean()
    msb = k * ((p - pb) ** 2).sum() / (g - 1)
    msw = (k * p * (1 - p)).sum() / (g * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw)


cells = {'old': {'Qwen': [('s16b', 'qwen38_low', 'cot'), ('s16b', 'qwen38_low', 'fsc'), ('s19', 'qwen38_low', 'con')],
                 'GLM': [('s16b', 'glm53', 'cot'), ('s16b', 'glm53', 'fsc'), ('s19', 'glm53', 'con')],
                 'DeepSeek': [('s20', 'dsv41', m) for m in ('cot', 'fsc', 'con')]},
         'new': {lab: [(st, mod, m) for m in ('cot', 'fsc', 'con')] for lab, mod, st in
                 (('Qwen', 'qwen38_low', 's21'), ('GLM', 'glm53', 's23'), ('DeepSeek', 'dsv41', 's23'),
                  ('Haiku', 'haiku55', 's23'), ('Sonnet', 'sonnet55_low', 's23'), ('Gemini', 'gemini38_low', 's23'))}}
DATA = {}
for w, tab in cells.items():
    for lab, specs in tab.items():
        out = []
        for M, sp in zip('REC', specs):
            rr = recs(*sp)
            DATA[(w, lab, M)] = rr
            c = collections.Counter()
            for x in rr:
                c[lay(x[1])] += x[4]
            assert len(c) == 14 or len([k for k in c]) <= 14
            allays = sorted(set(lay(x[1]) for x in rr))
            out.append(f'{M} {icc_counts([c[l] for l in allays]):.3f}')
        say(f'  {w} {lab}: ' + ', '.join(out))


# 3
def betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1) < 1e-15:
            break
    return h


def ibeta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * betacf(a, b, x) / a
    return 1 - math.exp(lbt) * betacf(b, a, 1 - x) / b


def tcdf(t, df):
    x = df / (df + t * t)
    p = 0.5 * ibeta(df / 2, 0.5, x)
    return 1 - p if t > 0 else p


def tq(q, df):
    lo, hi = 0.0, 60.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tcdf(mid, df) < q else (lo, mid)
    return (lo + hi) / 2


say('3. t(13) intervals, new wording')
t975 = tq(0.975, 13)
say(f'  t0.975,13 = {t975:.5f}, t0.80,13 = {tq(0.80, 13):.5f}')
LD = {}
for lab in cells['new']:
    for a, b in (('C', 'E'), ('C', 'R')):
        A = {x[:4]: x[4] for x in DATA[('new', lab, a)]}
        B = {x[:4]: x[4] for x in DATA[('new', lab, b)]}
        assert A.keys() == B.keys()
        dsum = collections.Counter()
        for k in A:
            dsum[lay(k[1])] += A[k] - B[k]
        ls = sorted(set(lay(k[1]) for k in A))
        d = np.array([dsum[l] / 18 for l in ls])
        LD[(lab, a + '-' + b)] = [dsum[l] for l in ls]
        m, s = d.mean(), d.std(ddof=1)
        h = t975 * s / math.sqrt(14)
        say(f'  {lab} {a}-{b}: {sum(A.values()) - sum(B.values()):+d}, CI [{(m - h) * 252:+.1f}, {(m + h) * 252:+.1f}]')

# 4
say('4. Power at n = 14, full enumeration, python random.Random(7), 2,000 draws')
signs = np.array(list(itertools.product((1, -1), repeat=14)), dtype=np.int64)
rnd = random.Random(7)
for key, v in LD.items():
    rej = 0
    for _ in range(2000):
        smp = np.array([rnd.choice(v) for _ in range(14)], dtype=np.int64)
        T = abs(int(smp.sum()))
        p = float(np.mean(np.abs(signs @ smp) >= T))
        rej += p <= 0.05
    say(f'  {key[0]} {key[1]}: {rej / 2000:.3f}')

(OUT / 'verify_output.txt').write_text('\n'.join(LINES) + '\n', encoding='utf-8')
