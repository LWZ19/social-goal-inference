"""Independent verifier for fusion claims F1-F4 and N1-N7. Reads only through research_harness."""
import sys, collections, itertools, io
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H

OUT = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s); OUT.write(s + '\n')

ARMS = ['real', 'staffed', 'plain']

def load(stage, model, method='con'):
    """{(arm, scene, ep): {trial: Record}}"""
    eps = collections.defaultdict(dict)
    tally = collections.Counter()
    for arm in ARMS:
        rs = H.runs(stage=stage, arm=arm, model=model, method=method)
        assert len(rs) == 1, (stage, arm, model, rs)
        for rec in H.records(rs[0], tally=tally):
            k = (arm, rec.scene_id, rec.episode_id)
            assert rec.trial not in eps[k], ('dup trial', k, rec.trial)
            eps[k][rec.trial] = rec
    return eps, tally

def succ(eps):
    return {k: [bool(v[t].score['success']) for t in sorted(v)] for k, v in eps.items()}

def summary(name, S):
    n = len(S)
    tri = collections.Counter(len(v) for v in S.values())
    per_trial = [sum(v[i] for v in S.values()) for i in range(3)]
    tot = sum(map(sum, S.values()))
    anyc = sum(any(v) for v in S.values())
    majc = sum(sum(v) >= 2 for v in S.values())
    allc = sum(all(v) for v in S.values())
    p(f'{name}: n={n} trials/ep={dict(tri)} any={anyc} maj={majc} all={allc} per-trial={per_trial} total={tot} mean={tot/3:.2f} gain={majc - tot/3:+.2f}')
    return dict(n=n, any=anyc, maj=majc, all=allc, mean=tot/3)

def plen(p_):
    if p_ is None: return None
    return len(p_) if isinstance(p_, list) else len(p_.get('people', p_))

def judge_kinds(c):
    if c is None: return None
    L = c if isinstance(c, list) else c.get('constraints', [])
    return tuple(sorted(str(y.get('kind')) for y in L if isinstance(y, dict)))

def true_kinds(c):
    if c is None: return None
    out = []
    for y in c:
        if isinstance(y, dict) and isinstance(y.get('constraint'), dict):
            out.append(str(y['constraint'].get('kind')))
        elif isinstance(y, dict):
            out.append('UNWRAPPED:' + str(y.get('kind')))
    return tuple(sorted(out))

def mixed(name, eps):
    nm = pc = jk = tk = tks = 0; ncount = 0; pc_same_tk = 0; allfail = 0
    unwrapped = 0
    for k, v in eps.items():
        cs = [v[t] for t in sorted(v) if v[t].result.get('source') == 'constraints' and v[t].score['target_correct']]
        if len(cs) < 2: continue
        s = [bool(x.score['success']) for x in cs]
        if any(s) and not all(s):
            nm += 1
            pcs = [plen(x.people) for x in cs]
            jks = [judge_kinds(x.constraints) for x in cs]
            tkm = [true_kinds(x.constraints) for x in cs]
            cnt = [len(x.constraints) if x.constraints is not None else None for x in cs]
            unwrapped += sum(any(t.startswith('UNWRAPPED') for t in (tt or ())) for tt in tkm)
            if len(set(pcs)) > 1: pc += 1
            if len(set(jks)) > 1: jk += 1
            if len(set(tkm)) > 1: tk += 1
            if len(set(frozenset(t) if t is not None else None for t in tkm)) > 1: tks += 1
            if len(set(cnt)) > 1: ncount += 1
            if len(set(pcs)) == 1 and len(set(tkm)) > 1: pc_same_tk += 1
        elif not any(s):
            allfail += 1
    p(f'{name}: mixed={nm} allfailC={allfail} people-count differs={pc} judge-literal kinds differs={jk} constraint-count differs={ncount} '
      f'true kind multiset differs={tk} true kind set differs={tks} same people count & true kinds differ={pc_same_tk} unwrapped entries={unwrapped}')
    return nm, pc, jk, tk

def mixed_loose(name, eps):
    nm = n2 = pc = tk = 0
    for k, v in eps.items():
        s = [bool(v[t].score['success']) for t in sorted(v)]
        if not (any(s) and not all(s)): continue
        nm += 1
        cs = [v[t] for t in sorted(v) if v[t].result.get('source') == 'constraints']
        if len(cs) < 2: continue
        n2 += 1
        if len(set(plen(x.people) for x in cs)) > 1: pc += 1
        if len(set(true_kinds(x.constraints) for x in cs)) > 1: tk += 1
    p(f'{name} loose: mixed(all 3 trials)={nm} with>=2 C steps={n2} people differs={pc} true kinds differ={tk}')

def pooled(name, Ss, thresholds):
    keys = set(Ss[0])
    for S in Ss: assert set(S) == keys, name
    pool = {k: sum((S[k] for S in Ss), []) for k in keys}
    m = len(next(iter(pool.values())))
    assert all(len(v) == m for v in pool.values())
    parts = [f'any={sum(any(v) for v in pool.values())}']
    for t in thresholds:
        parts.append(f'>={t}of{m}={sum(sum(v) >= t for v in pool.values())}')
    parts.append(f'all={sum(all(v) for v in pool.values())}')
    parts.append(f'mean/sample={sum(map(sum, pool.values()))/m:.2f}')
    p(f'{name} (n={len(pool)}): ' + ' '.join(parts))
    return pool

p('=== OLD wording: s19 GLM, s20 DeepSeek, s19 Qwen (C)')
g19, tg = load('s19', 'glm53'); d20, td = load('s20', 'dsv41'); q19, tq = load('s19', 'qwen38_low')
p('tallies', dict(tg), dict(td), dict(tq))
Sg, Sd, Sq = succ(g19), succ(d20), succ(q19)
summary('F1/F4 GLM s19', Sg); summary('F1/F4 DS s20', Sd); summary('Qwen s19', Sq)
p('=== F2 mixedC')
mixed('F2 GLM s19', g19); mixed('F2 DS s20', d20)
mixed_loose('GLM s19', g19); mixed_loose('DS s20', d20)
p('=== F3 cross')
pooled('F3 GLM+DS old', [Sg, Sd], [3, 4])
p('=== N7 old 3-model')
pooled('N7 Qwen+GLM+DS old', [Sq, Sg, Sd], [3, 4, 5])

p('=== NEW wording: s23 GLM, DS; s21 Qwen; s23 others (C)')
models = [('s21', 'qwen38_low'), ('s23', 'glm53'), ('s23', 'dsv41'), ('s23', 'haiku55'), ('s23', 'sonnet55_low'), ('s23', 'gemini38_low')]
E = {}; S = {}
for st, m in models:
    E[m], t = load(st, m); S[m] = succ(E[m])
    p(m, st, 'tally', dict(t))
    summary(f'N5 {m} {st}', S[m])
p('=== N2 mixedC new')
mixed('N2 GLM s23', E['glm53']); mixed('N2 DS s23', E['dsv41'])
mixed_loose('GLM s23', E['glm53']); mixed_loose('DS s23', E['dsv41'])
p('=== N3 cross new')
pooled('N3 GLM+DS s23', [S['glm53'], S['dsv41']], [3, 4])
p('=== N6 6-model pooled')
pooled('N6 6 models', [S[m] for _, m in models], [9, 10, 12])
keys = list(S['glm53'])
majv = {k: [sum(S[m][k]) >= 2 for _, m in models] for k in keys}
p(f'N6 per-model-majority vote: >=4of6={sum(sum(v)>=4 for v in majv.values())} >=3of6={sum(sum(v)>=3 for v in majv.values())} n={len(majv)}')
for (_, a), (_, b) in itertools.combinations(models, 2):
    pooled(f'pair {a}+{b}', [S[a], S[b]], [3, 4])

p('=== cache-copy check: s23 people vs s19/s20 people (same arm, scene, episode, trial)')
for new, old in [('glm53', g19), ('dsv41', d20)]:
    both = same = 0; cachedp = callsp = 0; cachedc = callsc = 0
    for k, v in E[new].items():
        for t, rec in v.items():
            tp = rec.timings.get('people') or {}
            callsp += tp.get('calls', 0); cachedp += tp.get('cached', 0)
            tc = rec.timings.get('constraints') or {}
            callsc += tc.get('calls', 0); cachedc += tc.get('cached', 0)
            o = old.get(k, {}).get(t)
            if o is None: continue
            if rec.people is not None and o.people is not None:
                both += 1; same += rec.people == o.people
    p(f'{new}: people lists both present={both} identical={same}; people calls cached {cachedp}/{callsp}; constraints calls cached {cachedc}/{callsc}')

open('results/2026-10-17-beyond-vlm/rerun/fusion/verify/v3_fusion_out.txt', 'w').write(OUT.getvalue())
