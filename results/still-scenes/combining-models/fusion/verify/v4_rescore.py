"""Rescore subset: s19 glm53 con and s20 dsv41 con (F1-F4 basis), and s23 glm53/dsv41 con (N1-N4 basis). Recorded vs now."""
import sys, collections, io, time
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
OUT = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n')
ARMS = ['real', 'staffed', 'plain']
store = {}
for st, m in [('s19', 'glm53'), ('s20', 'dsv41'), ('s23', 'glm53'), ('s23', 'dsv41')]:
    t0 = time.time(); mism = 0; S = collections.defaultdict(list); n = 0; tcm = 0
    for arm in ARMS:
        r = H.runs(stage=st, arm=arm, model=m, method='con')[0]
        for rec in H.records(r):
            sc = H.rescore(rec)
            now = bool(sc.success); n += 1
            if now != bool(rec.score['success']): mism += 1
            if bool(sc.target_correct) != bool(rec.score['target_correct']): tcm += 1
            S[(arm, rec.scene_id, rec.episode_id)].append(now)
    store[(st, m)] = S
    tot = sum(map(sum, S.values()))
    p(f'{st} {m}: records={n} success mismatches={mism} target_correct mismatches={tcm} rescored any={sum(any(v) for v in S.values())} maj={sum(sum(v)>=2 for v in S.values())} all={sum(all(v) for v in S.values())} mean={tot/3:.2f} ({time.time()-t0:.0f}s)')
for a, b, name in [(('s19','glm53'), ('s20','dsv41'), 'old'), (('s23','glm53'), ('s23','dsv41'), 'new')]:
    A, B = store[a], store[b]
    pool = {k: A[k] + B[k] for k in A}
    p(f'{name} rescored GLM+DS 6: any={sum(any(v) for v in pool.values())} >=3={sum(sum(v)>=3 for v in pool.values())} >=4={sum(sum(v)>=4 for v in pool.values())} all={sum(all(v) for v in pool.values())}')
open('results/still-scenes/combining-models/fusion/verify/v4_rescore_out.txt', 'w').write(OUT.getvalue())
