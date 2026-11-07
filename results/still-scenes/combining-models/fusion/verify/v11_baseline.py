"""Base rate for F2's statistics: in episodes whose C goal steps (right target, >=2) ALL succeed or ALL fail, how often
do the people count and the true constraint-kind multiset differ across trials? Same measures as for mixed episodes."""
import sys, collections, io
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
OUT = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n')
for st, m in [('s19', 'glm53'), ('s20', 'dsv41'), ('s23', 'glm53'), ('s23', 'dsv41')]:
    eps = collections.defaultdict(list)
    for arm in ['real', 'staffed', 'plain']:
        for rec in H.records(H.runs(stage=st, arm=arm, model=m, method='con')[0]):
            eps[(arm, rec.scene_id, rec.episode_id)].append(rec)
    c = collections.Counter()
    for k, v in eps.items():
        cs = [x for x in v if x.result.get('source') == 'constraints' and x.score['target_correct']]
        if len(cs) < 2: continue
        s = [bool(x.score['success']) for x in cs]
        g = 'mixed' if any(s) and not all(s) else ('allsucc' if all(s) else 'allfail')
        c[g] += 1
        c[g + ' people-count differs'] += len(set(len(x.people) for x in cs)) > 1
        c[g + ' true kinds differ'] += len(set(tuple(sorted(y['constraint']['kind'] for y in x.constraints)) for x in cs)) > 1
        c[g + ' n constraints differ'] += len(set(len(x.constraints) for x in cs)) > 1
    p(st, m, dict(sorted(c.items())))
open('results/2026-10-17-beyond-vlm/rerun/fusion/verify/v11_baseline_out.txt', 'w').write(OUT.getvalue())
