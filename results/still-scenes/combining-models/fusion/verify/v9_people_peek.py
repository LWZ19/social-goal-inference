"""Look at people lists of two GLM s19 mixed episodes with equal counts."""
import sys, collections, io
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
OUT = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n')
eps = collections.defaultdict(list)
for arm in ['real', 'staffed', 'plain']:
    for rec in H.records(H.runs(stage='s19', arm=arm, model='glm53', method='con')[0]):
        eps[(arm, rec.scene_id, rec.episode_id)].append(rec)
shown = 0
for k, v in eps.items():
    cs = [x for x in v if x.result.get('source') == 'constraints' and x.score['target_correct']]
    if len(cs) < 2: continue
    s = [bool(x.score['success']) for x in cs]
    if not (any(s) and not all(s)) or len(set(len(x.people) for x in cs)) != 1: continue
    p('==', k, v[0].situation)
    for x in cs:
        p('  trial', x.trial, 'success', x.score['success'], 'target', x.target.get('world')[:2] if x.target else None)
        for q in x.people:
            p('    #%s xy=(%.2f,%.2f) px=%s %s' % (q.get('number'), q['xy'][0], q['xy'][1], q.get('pixel'), str(q.get('label'))[:60]))
    shown += 1
    if shown >= 3: break
open('results/still-scenes/combining-models/fusion/verify/v9_people_peek_out.txt', 'w').write(OUT.getvalue())
