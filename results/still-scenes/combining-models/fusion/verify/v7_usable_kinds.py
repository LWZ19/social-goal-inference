"""Sensitivity for F2/N2: kind multiset over usable constraints only; status values; people-list identity across trials."""
import sys, collections, io
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
OUT = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n')
def load(stage, model):
    eps = collections.defaultdict(list)
    for arm in ['real', 'staffed', 'plain']:
        for rec in H.records(H.runs(stage=stage, arm=arm, model=model, method='con')[0]):
            eps[(arm, rec.scene_id, rec.episode_id)].append(rec)
    return eps
for st, m in [('s19', 'glm53'), ('s20', 'dsv41'), ('s23', 'glm53'), ('s23', 'dsv41')]:
    eps = load(st, m)
    status = collections.Counter()
    c = collections.Counter()
    for k, v in eps.items():
        v.sort(key=lambda r: r.trial)
        cs = [x for x in v if x.result.get('source') == 'constraints' and x.score['target_correct']]
        for x in cs:
            for y in x.constraints: status[y.get('status')] += 1
        if len(cs) < 2: continue
        s = [bool(x.score['success']) for x in cs]
        if not (any(s) and not all(s)): continue
        c['mixed'] += 1
        uk = [tuple(sorted(y['constraint']['kind'] for y in x.constraints if y.get('status') != 'unusable')) for x in cs]
        c['usable kinds differ'] += len(set(uk)) > 1
        c['people list identical across C trials'] += len(set(repr(x.people) for x in cs)) == 1
        c['people xy rounded 0.3m same'] += len(set(tuple(sorted((round(q['xy'][0] / 0.3), round(q['xy'][1] / 0.3)) for q in x.people)) for x in cs)) == 1
    p(st, m, dict(c), 'statuses', dict(status))
open('results/still-scenes/combining-models/fusion/verify/v7_usable_kinds_out.txt', 'w').write(OUT.getvalue())
