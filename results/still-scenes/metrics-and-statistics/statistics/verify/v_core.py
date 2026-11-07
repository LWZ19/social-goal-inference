"""Independent verifier for i46_stats: ICC, design effect, t intervals, sign-flip p, batch SD, MDD.
Reads runs only through the step-0 harness. Written without reading i46_stats.py."""
import sys, collections, itertools, math, pickle, os
import numpy as np
from statistics import NormalDist

def _simpson(f, a, b, n=20000):
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(a + i * h)
    return s * h / 3

def _tpdf(x, df):
    return math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2)) * (1 + x * x / df) ** (-(df + 1) / 2)

def t_ppf(q, df):
    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        c = 0.5 + _simpson(lambda x: _tpdf(x, df), 0, mid, 4000)
        lo, hi = (mid, hi) if c < q else (lo, mid)
    return (lo + hi) / 2

def chi2_ppf(q, df):
    pdf = lambda x: 0.0 if x <= 0 else x ** (df / 2 - 1) * math.exp(-x / 2) / (2 ** (df / 2) * math.gamma(df / 2))
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        c = _simpson(pdf, 0, mid, 4000)
        lo, hi = (mid, hi) if c < q else (lo, mid)
    return (lo + hi) / 2

class _St:
    class t:
        ppf = staticmethod(t_ppf)
    class chi2:
        ppf = staticmethod(chi2_ppf)
    class norm:
        ppf = staticmethod(lambda q: NormalDist().inv_cdf(q))
st = _St
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H

OUT = os.path.dirname(os.path.abspath(__file__))
ARMS = ('real', 'staffed', 'plain')
M2 = {'R': 'cot', 'E': 'fsc', 'C': 'con'}

def load(stage, model, method, trials=None):
    d = {}
    for r in H.runs(stage=stage, model=model, method=method):
        if r.arm not in ARMS:
            continue
        for rec in H.records(r):
            if trials is not None and rec.trial not in trials:
                continue
            k = (r.arm, rec.scene_id, rec.episode_id, rec.trial)
            assert k not in d
            d[k] = 1 if rec.score.get('success') else 0
    return d

def layout(arm, sid):
    assert sid.startswith(arm + '_'), (arm, sid)
    return sid[len(arm) + 1:]

CELLS = {
 'old': {'Qwen': {'R': ('s16b', 'qwen38_low'), 'E': ('s16b', 'qwen38_low'), 'C': ('s19', 'qwen38_low')},
         'GLM': {'R': ('s16b', 'glm53'), 'E': ('s16b', 'glm53'), 'C': ('s19', 'glm53')},
         'DeepSeek': {m: ('s20', 'dsv41') for m in 'REC'}},
 'new': {'Qwen': {m: ('s21', 'qwen38_low') for m in 'REC'},
         'GLM': {m: ('s23', 'glm53') for m in 'REC'},
         'DeepSeek': {m: ('s23', 'dsv41') for m in 'REC'},
         'Haiku': {m: ('s23', 'haiku55') for m in 'REC'},
         'Sonnet': {m: ('s23', 'sonnet55_low') for m in 'REC'},
         'Gemini': {m: ('s23', 'gemini38_low') for m in 'REC'}},
}

cache = os.path.join(OUT, 'data.pkl')
if os.path.exists(cache):
    D = pickle.load(open(cache, 'rb'))
else:
    D = {}
    for w, models in CELLS.items():
        for mname, meths in models.items():
            for m, (stage, model) in meths.items():
                D[(w, mname, m)] = load(stage, model, M2[m])
    # Qwen pool
    for m in 'RE':
        D[('pool', 's16b', m)] = load('s16b', 'qwen38_low', M2[m])
        D[('pool', 's22a', m)] = load('s22', 'qwen38_low', M2[m], trials={0, 1, 2})
        D[('pool', 's22b', m)] = load('s22', 'qwen38_low', M2[m], trials={3, 4, 5})
        D[('pool', 's21', m)] = load('s21', 'qwen38_low', M2[m])
    pickle.dump(D, open(cache, 'wb'))

def by_layout(d):
    g = collections.defaultdict(list)
    for (arm, sid, ep, tr), v in d.items():
        g[layout(arm, sid)].append(v)
    return g

def icc1(groups):
    gs = [np.asarray(v, float) for v in groups.values()]
    k = len(gs[0]); assert all(len(x) == k for x in gs)
    G = len(gs); N = G * k
    grand = np.mean(np.concatenate(gs))
    ssb = k * sum((x.mean() - grand) ** 2 for x in gs)
    ssw = sum(((x - x.mean()) ** 2).sum() for x in gs)
    msb = ssb / (G - 1); msw = ssw / (N - G)
    return (msb - msw) / (msb + (k - 1) * msw), k, G

lines = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); lines.append(s)

P('== counts and ICC(1) by layout ==')
icc_res = {}
for w in ('old', 'new'):
    for mname in CELLS[w]:
        row = []
        for m in 'REC':
            d = D[(w, mname, m)]
            assert len(d) == 252, (w, mname, m, len(d))
            icc, k, G = icc1(by_layout(d))
            icc_res[(w, mname, m)] = icc
            row.append(f"{m} n={len(d)} succ={sum(d.values())} icc={icc:.3f} deff={1+(k-1)*icc:.2f} neff={252/(1+(k-1)*icc):.1f} (k={k},G={G})")
        P(w, mname, ' | '.join(row))
vals = list(icc_res.values())
P('ICC range', round(min(vals), 3), round(max(vals), 3), 'deff range', round(1 + 17 * min(vals), 2), round(1 + 17 * max(vals), 2))
for w in ('old', 'new'):
    for m in 'REC':
        P('median ICC', w, m, round(float(np.median([icc_res[(w, mn, m)] for mn in CELLS[w]])), 3))

def paired(a, b):
    assert set(a) == set(b), (len(set(a) ^ set(b)))
    return {k: a[k] - b[k] for k in a}

def signflip_p(sums):
    sums = np.asarray(sums)
    obs = abs(sums.sum())
    tot = 0; cnt = 0
    for signs in itertools.product((1, -1), repeat=len(sums)):
        s = abs((np.asarray(signs) * sums).sum())
        cnt += s >= obs - 1e-9; tot += 1
    return cnt / tot

P('\n== paired contrasts: diff, layouts +/-/0, sign-flip p two-sided, t CI (records), ICC of d ==')
tq = st.t.ppf(0.975, 13)
P('t0.975,13 =', round(tq, 5), ' t0.80,13 =', round(st.t.ppf(0.80, 13), 4))
LAY = {}
for w in ('old', 'new'):
    for mname in CELLS[w]:
        for (A, B) in (('C', 'E'), ('C', 'R'), ('E', 'R')):
            dd = paired(D[(w, mname, A)], D[(w, mname, B)])
            g = by_layout(dd)
            keys = sorted(g)
            sums = np.array([sum(g[k]) for k in keys])
            LAY[(w, mname, A + '-' + B)] = dict(zip(keys, sums))
            means = sums / 18.0
            m = means.mean() * 252; sd = means.std(ddof=1)
            half = tq * sd / math.sqrt(14) * 252
            p = signflip_p(sums)
            icd, _, _ = icc1(g)
            P(f"{w} {mname:8s} {A}-{B}: {sums.sum():+d} lay {sum(sums>0)}/{sum(sums<0)}/{sum(sums==0)} p={p:.5f} CI {m-half:+.1f} to {m+half:+.1f} sd_layout={sd:.4f} iccd={icd:.3f} deff_d={1+17*icd:.2f} MDD14={(tq+st.t.ppf(0.8,13))*sd/math.sqrt(14)*252:.1f}")
pickle.dump(LAY, open(os.path.join(OUT, 'layouts.pkl'), 'wb'))

P('\n== Qwen pool batch totals ==')
for m in 'RE':
    a = D[('pool', 's16b', m)]; b = D[('pool', 's22a', m)]
    same = sum(a[k] == b[k] for k in a)
    P(m, 's16b vs s22 trials 0-2 identical verdicts', same, 'of', len(a), 'keys equal', set(a) == set(b))
tot = {}
for m in 'RE':
    tot[m] = [sum(D[('pool', b, m)].values()) for b in ('s16b', 's22b', 's21')]
    P(m, 'batch totals s16b/s22b/s21', tot[m], 'n', [len(D[('pool', b, m)]) for b in ('s16b', 's22b', 's21')], 'SD', round(np.std(tot[m], ddof=1), 3))
pooled = math.sqrt(np.mean([np.var(tot[m], ddof=1) for m in 'RE']))
P('pooled batch SD', round(pooled, 3))
lo = math.sqrt(4 * pooled ** 2 / st.chi2.ppf(0.975, 4)); hi = math.sqrt(4 * pooled ** 2 / st.chi2.ppf(0.025, 4))
P('exact chi2 95% CI (df 4)', round(lo, 2), round(hi, 2))
# single-trial totals x sqrt(3)
vv = {}
for m in 'RE':
    tt = []
    for b in ('s16b', 's22b', 's21'):
        d = D[('pool', b, m)]
        trs = sorted(set(k[3] for k in d))
        for t in trs:
            tt.append(sum(v for k, v in d.items() if k[3] == t))
    vv[m] = np.var(tt, ddof=1)
    P(m, 'single-trial totals', tt, 'sd x sqrt3', round(math.sqrt(3 * vv[m]), 3))
P('pooled (b)', round(math.sqrt(3 * np.mean(list(vv.values()))), 3))
# item binomial
ib = {}
for m in 'RE':
    items = collections.defaultdict(list)
    for b in ('s16b', 's22b', 's21'):
        for (arm, sid, ep, tr), v in D[('pool', b, m)].items():
            items[(arm, sid, ep)].append(v)
    assert len(items) == 84 and all(len(v) == 9 for v in items.values())
    s2 = sum(np.var(v, ddof=1) for v in items.values())
    ib[m] = 3 * s2
    P(m, 'item binomial sqrt(3 sum s_i^2)', round(math.sqrt(3 * s2), 3))
P('pooled (c)', round(math.sqrt(np.mean(list(ib.values()))), 3))
# paired E-R within pool
er = []
for b in ('s16b', 's22b', 's21'):
    er.append(sum(D[('pool', b, 'E')].values()) - sum(D[('pool', b, 'R')].values()))
P('E-R per batch', er, 'mean', np.mean(er), 'SD', round(np.std(er, ddof=1), 2))
items = collections.defaultdict(list)
for b in ('s16b', 's22b', 's21'):
    for k in D[('pool', b, 'E')]:
        items[k[:3]].append(D[('pool', b, 'E')][k] - D[('pool', b, 'R')][k])
s2 = sum(np.var(v, ddof=1) for v in items.values())
P('item-level paired SD (3 trials)', round(math.sqrt(3 * s2), 2), 'vs sqrt2 x pooled', round(math.sqrt(2) * pooled, 2))

P('\n== MDD ==')
z = st.norm.ppf(0.975) + st.norm.ppf(0.8)
P('factor normal 2-sample', round(z * math.sqrt(2), 3), 'paired', round(z, 3))
for sd in (5, 6, 7, pooled, 8):
    P('batch SD', round(sd, 2), 'MDD', round(z * math.sqrt(2) * sd, 1), 'pp', round(100 * z * math.sqrt(2) * sd / 252, 1))
for b in (2, 3, 4):
    P('b batches', b, round(z * math.sqrt(2) * pooled / math.sqrt(b), 1))
P('paired Qwen pool E-R MDD', round(z * math.sqrt(3 * s2), 1), ' with across-batch SD', round(z * np.std(er, ddof=1), 1))

open(os.path.join(OUT, 'v_core_output.txt'), 'w').write('\n'.join(lines) + '\n')
