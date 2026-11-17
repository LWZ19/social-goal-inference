"""Independent power check: exact sign-flip p via integer DP (no 2^n enumeration), two-sided P(|S|>=|obs|),
draws of n layouts with replacement from the 14 observed integer layout differences. Also per-situation power and
within-batch noise for the six new-wording models."""
import sys, os, pickle, math, collections
import numpy as np
from statistics import NormalDist
OUT = os.path.dirname(os.path.abspath(__file__))
LAY = pickle.load(open(os.path.join(OUT, 'layouts.pkl'), 'rb'))
D = pickle.load(open(os.path.join(OUT, 'data.pkl'), 'rb'))
lines = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); lines.append(s)

_memo = {}
def p_two(vals):
    key = tuple(sorted(int(v) for v in vals))
    if key in _memo:
        return _memo[key]
    a = [abs(v) for v in key if v != 0]
    obs = abs(sum(key))
    tot = sum(a)
    dist = np.zeros(2 * tot + 1); dist[tot] = 1.0
    for v in a:
        new = np.zeros_like(dist)
        new[v:] += dist[:len(dist) - v] if v else dist
        new[:len(dist) - v] += dist[v:] if v else dist
        dist = new / 2
    s = np.arange(-tot, tot + 1)
    p = dist[np.abs(s) >= obs].sum()
    _memo[key] = p
    return p

def sit(l):
    n = int(l.split('_')[1])
    return 'lines' if n in (6, 7, 8, 9, 13) else 'groups' if n in (10, 11, 12) else 'nobody'

# sanity vs exact enumeration on observed values
for k in (('new', 'Qwen', 'C-E'), ('new', 'Sonnet', 'C-R'), ('old', 'GLM', 'C-E')):
    v = list(LAY[k].values())
    P('observed p (DP)', k, round(p_two(v), 5), 'values', v)

rng = np.random.default_rng(12345)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
P(f'\n== power, two-sided, alpha 0.05 (p <= 0.05), B={B} ==')
for w, contrasts in (('new', ('C-E', 'C-R')), ('old', ('C-E',))):
    models = ['Qwen', 'GLM', 'DeepSeek', 'Haiku', 'Sonnet', 'Gemini'] if w == 'new' else ['Qwen', 'GLM', 'DeepSeek']
    for c in contrasts:
        for m in models:
            v = np.array(list(LAY[(w, m, c)].values()))
            row = []
            for n in (8, 14, 20, 30):
                rej = 0
                for _ in range(B):
                    rej += p_two(rng.choice(v, size=n, replace=True)) <= 0.05
                row.append(f"{rej/B:.3f}")
            P(w, c, f"{m:8s}", ' / '.join(row), ' nonzero layouts', int((v != 0).sum()))

P('\n== tied layouts by situation, new wording ==')
for c in ('C-E', 'C-R'):
    for m in ['Qwen', 'GLM', 'DeepSeek', 'Haiku', 'Sonnet', 'Gemini']:
        d = LAY[('new', m, c)]
        by = collections.defaultdict(list)
        for l, v in sorted(d.items()):
            by[sit(l)].append(int(v))
        P(c, f"{m:8s}", dict(by))

P('\n== per-situation power (n in 6,8,14,20,30,40), B=2000 ==')
for c in ('C-E', 'C-R'):
    for m in ['Qwen', 'GLM', 'DeepSeek', 'Haiku', 'Sonnet', 'Gemini']:
        d = LAY[('new', m, c)]
        for s in ('lines', 'groups'):
            v = np.array([x for l, x in d.items() if sit(l) == s])
            row = []
            first = None
            for n in (6, 8, 14, 20, 30, 40):
                rej = sum(p_two(rng.choice(v, size=n, replace=True)) <= 0.05 for _ in range(2000)) / 2000
                row.append(f"{rej:.2f}")
                if first is None and rej >= 0.8:
                    first = n
            P(c, f"{m:8s}", s, v.tolist(), ' '.join(row), 'first n>=0.8:', first)

P('\n== within-batch noise, new wording, six models ==')
import importlib.util
NEW = {'Qwen': ('s21', 'qwen38_low'), 'GLM': ('s23', 'glm53'), 'DeepSeek': ('s23', 'dsv41'), 'Haiku': ('s23', 'haiku55'),
       'Sonnet': ('s23', 'sonnet55_low'), 'Gemini': ('s23', 'gemini38_low')}
z = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.8)
for m in NEW:
    ib = []; tt = []
    for meth in 'REC':
        d = D[('new', m, meth)]
        items = collections.defaultdict(list)
        for k, x in d.items():
            items[k[:3]].append(x)
        assert len(items) == 84 and all(len(x) == 3 for x in items.values())
        ib.append(3 * sum(np.var(x, ddof=1) for x in items.values()))
        tots = [sum(x for k, x in d.items() if k[3] == t) for t in (0, 1, 2)]
        tt.append(3 * np.var(tots, ddof=1))
    pr = []
    for (A, Bm) in (('C', 'E'), ('C', 'R'), ('E', 'R')):
        items = collections.defaultdict(list)
        for k in D[('new', m, A)]:
            items[k[:3]].append(D[('new', m, A)][k] - D[('new', m, Bm)][k])
        sd = math.sqrt(3 * sum(np.var(x, ddof=1) for x in items.values()))
        pr.append(f"{A}-{Bm} {sd:.2f} (MDD {z*sd:.1f})")
    P(f"{m:8s} item-binomial pooled {math.sqrt(np.mean(ib)):.2f}  trial-total pooled {math.sqrt(np.mean(tt)):.2f}  paired: " + '; '.join(pr))
open(os.path.join(OUT, 'v_power_output.txt'), 'w').write('\n'.join(lines) + '\n')
