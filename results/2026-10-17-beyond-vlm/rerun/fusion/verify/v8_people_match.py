"""F2/N2 caveat: in mixed episodes with equal people count, are the people the same places? Optimal matching of xy
between every pair of C trials; worst matched distance per episode."""
import sys, collections, io, itertools, math
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
import numpy as np
OUT = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n')
def load(stage, model):
    eps = collections.defaultdict(list)
    for arm in ['real', 'staffed', 'plain']:
        for rec in H.records(H.runs(stage=stage, arm=arm, model=model, method='con')[0]):
            eps[(arm, rec.scene_id, rec.episode_id)].append(rec)
    return eps
def worst(a, b):
    A = np.array([q['xy'][:2] for q in a]); B = np.array([q['xy'][:2] for q in b])
    if len(A) == 0 or len(B) == 0: return 0.0 if len(A) == len(B) else math.inf
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    if len(A) != len(B): return math.inf
    best = math.inf
    for perm in itertools.permutations(range(len(B))):
        best = min(best, max(D[i, j] for i, j in enumerate(perm)))
    return float(best)
for st, m in [('s19', 'glm53'), ('s20', 'dsv41'), ('s23', 'glm53'), ('s23', 'dsv41')]:
    eps = load(st, m); ws = []; samecount = 0
    for k, v in eps.items():
        cs = [x for x in v if x.result.get('source') == 'constraints' and x.score['target_correct']]
        if len(cs) < 2: continue
        s = [bool(x.score['success']) for x in cs]
        if not (any(s) and not all(s)): continue
        if len(set(len(x.people) for x in cs)) != 1: continue
        samecount += 1
        ws.append(max(worst(a.people, b.people) for a, b in itertools.combinations(cs, 2)))
    ws.sort()
    p(st, m, 'mixed eps with same people count', samecount, 'worst matched xy gap: median %.2f m' % ws[len(ws)//2],
      '<=0.25m', sum(w <= 0.25 for w in ws), '<=0.5m', sum(w <= 0.5 for w in ws), '>1m', sum(w > 1 for w in ws))
open('results/2026-10-17-beyond-vlm/rerun/fusion/verify/v8_people_match_out.txt', 'w').write(OUT.getvalue())
