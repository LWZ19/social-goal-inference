"""Independent verifier for I46 cluster statistics. Reads runs only through research_harness."""
import sys, re, math, collections, itertools, random
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
import numpy as np

OUT = []
def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s); OUT.append(s)

ARMS = ('real', 'staffed', 'plain')

def one(stage, arm, model, method):
    rs = H.runs(stage=stage, arm=arm, model=model, method=method)
    assert len(rs) == 1, (stage, arm, model, method, rs)
    return rs[0]

def load(stage, model, method):
    """{(scene, episode, trial): (layout, situation, success)} over the three arms; checks keys are unique."""
    out = {}
    tallies = {}
    for arm in ARMS:
        ref = one(stage, arm, model, method)
        tal = collections.Counter()
        for r in H.records(ref, tally=tal):
            k = (r.scene_id, r.episode_id, r.trial)
            assert k not in out, ('dup key', k)
            assert r.scene_id.startswith(arm + '_'), r.scene_id
            lay = r.scene_id[len(arm) + 1:]
            out[k] = (lay, r.situation, int(bool(r.score['success'])))
        tallies[arm] = dict(tal)
    return out, tallies

def pair(A, B):
    """A is C, B the comparator. Returns list of (layout, situation, a, b) and unmatched counts."""
    ka, kb = set(A), set(B)
    common = sorted(ka & kb)
    pr = []
    for k in common:
        la, sa, a = A[k]; lb, sb, b = B[k]
        assert la == lb and sa == sb
        pr.append((la, sa, a, b))
    return pr, len(ka - kb), len(kb - ka)

SIGNS = np.array(list(itertools.product([1, -1], repeat=14)), dtype=np.int64)  # 16384 x 14

def signflip(d):
    """d: integer per-layout summed differences (len 14). Exact two- and one-sided p (sum statistic)."""
    d = np.asarray(d, dtype=np.int64)
    assert len(d) == 14
    S = SIGNS @ d
    T = d.sum()
    two = np.mean(np.abs(S) >= abs(T))
    one_ = np.mean(S >= T)
    return float(two), float(one_)

# Student t quantile without scipy
def t_pdf(x, v):
    return math.gamma((v + 1) / 2) / (math.sqrt(v * math.pi) * math.gamma(v / 2)) * (1 + x * x / v) ** (-(v + 1) / 2)
def t_cdf(x, v, n=20000):
    # integrate pdf from 0 to |x| with Simpson
    a, b = 0.0, abs(x)
    if b == 0: return 0.5
    h = (b - a) / n
    s = t_pdf(a, v) + t_pdf(b, v)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * t_pdf(a + i * h, v)
    area = s * h / 3
    return 0.5 + area if x > 0 else 0.5 - area
def t_q(pr, v):
    lo, hi = 0.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if t_cdf(mid, v, 4000) < pr: lo = mid
        else: hi = mid
    return (lo + hi) / 2
TQ = t_q(0.975, 13)

def analyse(label, pr, want_detail=True):
    lays = sorted(set(x[0] for x in pr))
    sit = {l: next(x[1] for x in pr if x[0] == l) for l in lays}
    npl = collections.Counter(x[0] for x in pr)
    d = [sum(x[2] - x[3] for x in pr if x[0] == l) for l in lays]
    two, one_ = signflip(d)
    pos = sum(v > 0 for v in d); neg = sum(v < 0 for v in d); tie = sum(v == 0 for v in d)
    A = sum(x[2] for x in pr); B = sum(x[3] for x in pr)
    # t interval on layout mean differences, scaled to records of 252 (n pairs)
    dm = np.array(d, float) / np.array([npl[l] for l in lays], float)
    m = dm.mean(); sd = dm.std(ddof=1); se = sd / math.sqrt(len(dm))
    scale = len(pr)
    lo, hi = (m - TQ * se) * scale, (m + TQ * se) * scale
    tstat = m / se if se > 0 else float('inf')
    tp = 2 * (1 - t_cdf(abs(tstat), 13)) if se > 0 else 0.0
    p(f"{label}: n={len(pr)} layouts={len(lays)} per-layout pairs={sorted(set(npl.values()))} "
      f"C={A} other={B} diff={A-B} | +{pos}/-{neg}/0:{tie} | sign-flip two-sided p={two:.6f} one-sided p={one_:.6f} "
      f"| t95 [{lo:+.1f},{hi:+.1f}] t p={tp:.4f}")
    if want_detail:
        p('    per-layout C-other (summed):', ' '.join(f"{l[:7]}:{v:+d}" for l, v in zip(lays, d)))
    return dict(lays=lays, sit=sit, d=d, dm=dm, two=two, one=one_, pos=pos, neg=neg, tie=tie, A=A, B=B, n=len(pr))

# ---------------- old wording ----------------
OLD = {
    'qwen': (('s16b', 'qwen38_low', 'fsc'), ('s19', 'qwen38_low', 'con')),
    'glm': (('s16b', 'glm53', 'fsc'), ('s19', 'glm53', 'con')),
    'ds': (('s20', 'dsv41', 'fsc'), ('s20', 'dsv41', 'con')),
}
p('=== OLD WORDING: C vs E (S1/S2/S3) ===')
p(f't(0.975,13) = {TQ:.5f}')
old = {}
for m, (e, c) in OLD.items():
    E, te = load(*e); C, tc = load(*c)
    pr, ua, ub = pair(C, E)
    p(f'[{m}] E records={len(E)} C records={len(C)} unmatched C={ua} unmatched E={ub} tallies E={te} C={tc}')
    old[m] = analyse(f'[{m}] C vs E', pr)
    old[m]['pairs'] = pr

# S2: minimum p per number of non-tied layouts
p('\n=== S2 ===')
for m, r in old.items():
    k = r['pos'] + r['neg']
    p(f"[{m}] non-tied={k} min two-sided p=2/2^{k}={2/2**k:.6f}; observed {r['two']:.6f}; min one-sided {1/2**k:.6f}")
# fragility for Qwen: tie or flip each nonzero layout
r = old['qwen']
worst_tie = []; worst_flip = []
for i, v in enumerate(r['d']):
    if v == 0: continue
    dt = list(r['d']); dt[i] = 0; worst_tie.append(signflip(dt)[0])
    df = list(r['d']); df[i] = -v; worst_flip.append(signflip(df)[0])
p(f"[qwen] tie one layout -> two-sided p in {sorted(set(round(x,5) for x in worst_tie))}; flip one -> {sorted(set(round(x,5) for x in worst_flip))}")
# a 'flip' meaning one layout's sign goes against C by -1 record
dfm = []
for i, v in enumerate(r['d']):
    if v == 0: continue
    dd = list(r['d']); dd[i] = -1; dfm.append(signflip(dd)[0])
p(f"[qwen] set one non-tied layout to -1 -> two-sided p in {sorted(set(round(x,5) for x in dfm))}")

# S3: layouts per situation
p('\n=== S3 ===')
for m, r in old.items():
    cnt = collections.Counter(r['sit'][l] for l in r['lays'])
    p(f"[{m}] layouts per situation: {dict(cnt)}")
cnt = collections.Counter(old['qwen']['sit'][l] for l in old['qwen']['lays'])
for s, k in cnt.items():
    p(f"  {s}: k={k} min two-sided {2/2**k:.4f}, one-sided {1/2**k:.4f}")
# within-situation observed for each model
for m, r in old.items():
    for s in cnt:
        ds_ = [v for l, v in zip(r['lays'], r['d']) if r['sit'][l] == s]
        k = len(ds_)
        S = np.array(list(itertools.product([1, -1], repeat=k))) @ np.array(ds_)
        T = sum(ds_)
        p(f"  [{m}] {s}: d={ds_} two-sided p={np.mean(np.abs(S) >= abs(T)):.4f} one-sided={np.mean(S >= T):.4f}")

# ---------------- S4 power ----------------
p('\n=== S4 power ===')
def exact_two_batch(D):
    """D: draws x 14 integer matrix -> two-sided exact p per draw."""
    S = D @ SIGNS.T  # draws x 16384
    T = D.sum(1, keepdims=True)
    return (np.abs(S) >= np.abs(T)).mean(1)
def exact_one_batch(D):
    S = D @ SIGNS.T
    T = D.sum(1, keepdims=True)
    return (S >= T).mean(1)
rng = np.random.default_rng(12345)
for m, r in old.items():
    d = np.array(r['d'], dtype=np.int64)
    rej2 = 0; rej1 = 0; N = 10000
    for start in range(0, N, 500):
        idx = rng.integers(0, 14, size=(500, 14))
        D = d[idx]
        rej2 += int((exact_two_batch(D) <= 0.05).sum())
        rej1 += int((exact_one_batch(D) <= 0.05).sum())
    p(f"[{m}] design V (draw 14 layouts w/ replacement from observed d, exact sign-flip, alpha .05, 10000 draws, seed 12345): "
      f"power two-sided={rej2/N:.3f} (MC SE {math.sqrt(rej2/N*(1-rej2/N)/N):.3f}) one-sided={rej1/N:.3f}")
    # strict p<0.05 variant
random.seed(0)
for m in ('qwen', 'glm', 'ds'):
    r = old[m]
    lays = r['lays']
    dm = {l: v / 18 for l, v in zip(lays, r['d'])}
    # clus.py consumes no random numbers before the power loop for this model (the test is exhaustive)
    res = []
    for n in (8, 14, 20, 30):
        pw = 0; B = 300
        for _ in range(B):
            smp = [dm[random.choice(lays)] for _ in range(n)]
            o = abs(sum(smp)); c = 0
            for _ in range(400):
                if abs(sum(v * random.choice((1, -1)) for v in smp)) >= o - 1e-12: c += 1
            pw += (c / 400) < 0.05
        res.append((n, round(pw / B, 2)))
    p(f"[{m}] judge replay (clus.py loop): {res}")

# ---------------- ICC (as in clus.py) ----------------
p('\n=== ICC (one-way, by layout) ===')
import statistics as st
for m, r in old.items():
    pr = r['pairs']; lays = r['lays']
    def icc(idx):
        g = {l: [x[idx] for x in pr if x[0] == l] for l in lays}
        k = st.mean(len(v) for v in g.values()); gm = st.mean(x for v in g.values() for x in v)
        msb = sum(len(v) * (st.mean(v) - gm) ** 2 for v in g.values()) / (len(g) - 1)
        msw = sum((x - st.mean(v)) ** 2 for v in g.values() for x in v) / (sum(len(v) for v in g.values()) - len(g))
        return (msb - msw) / (msb + (k - 1) * msw)
    p(f"[{m}] ICC_C={icc(2):.3f} ICC_E={icc(3):.3f}")

# percentile bootstrap vs t for old
p('\n=== N3: old wording t-interval vs percentile bootstrap (records of 252) ===')
rngb = np.random.default_rng(7)
for m, r in old.items():
    dm = r['dm']; n = r['n']
    bs = dm[rngb.integers(0, 14, size=(10000, 14))].mean(1) * n
    p(f"[{m}] mean {dm.mean()*n:+.1f}; percentile bootstrap [{np.percentile(bs,2.5):+.1f},{np.percentile(bs,97.5):+.1f}]")

# ---------------- new wording ----------------
p('\n=== NEW WORDING: C vs E (N1), C vs R (N2) ===')
NEW = {'qwen': ('s21', 'qwen38_low'), 'glm': ('s23', 'glm53'), 'ds': ('s23', 'dsv41'),
       'haiku': ('s23', 'haiku55'), 'sonnet': ('s23', 'sonnet55_low'), 'gemini': ('s23', 'gemini38_low')}
newres = {}
cells_E = collections.Counter(); cells_R = collections.Counter()
for m, (stg, mod) in NEW.items():
    R, _ = load(stg, mod, 'cot'); E, _ = load(stg, mod, 'fsc'); C, _ = load(stg, mod, 'con')
    tot = lambda X: sum(v[2] for v in X.values())
    p(f"[{m}] records R/E/C = {len(R)}/{len(E)}/{len(C)}; successes R/E/C = {tot(R)}/{tot(E)}/{tot(C)}")
    prE, ua, ub = pair(C, E); assert ua == ub == 0, (ua, ub)
    prR, ua, ub = pair(C, R); assert ua == ub == 0, (ua, ub)
    aE = analyse(f'[{m}] C vs E', prE)
    aR = analyse(f'[{m}] C vs R', prR)
    newres[m] = (aE, aR)
    for l, v in zip(aE['lays'], aE['d']):
        cells_E['below' if v < 0 else 'tie' if v == 0 else 'above'] += 1
        if v < 0: p(f"    C below E: {m} {l} {v}")
    for l, v in zip(aR['lays'], aR['d']):
        cells_R['below' if v < 0 else 'tie' if v == 0 else 'above'] += 1
        if v < 0: p(f"    C below R: {m} {l} {v}")
p(f"cells C vs E (new, 6x14): {dict(cells_E)}")
p(f"cells C vs R (new, 6x14): {dict(cells_R)}")
allp2 = [(m, w, (newres[m][0] if w == 'E' else newres[m][1])['two']) for m in NEW for w in ('E', 'R')]
allp1 = [(m, w, (newres[m][0] if w == 'E' else newres[m][1])['one']) for m in NEW for w in ('E', 'R')]
p('largest two-sided p over 12 new comparisons:', max(x[2] for x in allp2), [x[:2] for x in allp2 if x[2] == max(y[2] for y in allp2)])
p('largest one-sided p over 12 new comparisons:', max(x[2] for x in allp1))
p('all 12 favour C:', all((newres[m][0]['A'] > newres[m][0]['B']) and (newres[m][1]['A'] > newres[m][1]['B']) for m in NEW))
# old-wording cell check
ob = [(m, l, v) for m, r in old.items() for l, v in zip(r['lays'], r['d']) if v < 0]
p('old wording C below E cells (of 42):', ob)

open('results/2026-10-17-beyond-vlm/rerun/stats/verify/v_stats.txt', 'w').write('\n'.join(OUT) + '\n')
