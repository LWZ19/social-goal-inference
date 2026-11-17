import sys, collections, json
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
r = H.runs(stage='s20', arm='real', model='dsv41', method='con')[0]
recs = list(H.records(r))
print(len(recs))
c = collections.Counter((x.situation, x.solver_step, x.result.get('source'), x.result.get('status')) for x in recs)
for k, v in sorted(c.items(), key=str): print(k, v)
rec = next(x for x in recs if x.solver_step and x.situation == 'conversation_group')
print(json.dumps(rec.candidates, indent=1)[:2000])
print('pose', rec.pose_world)
print('target', json.dumps(rec.target)[:600])
print('keys', list(rec.result.keys()))
print('solver', json.dumps(rec.solver)[:1500])
print('score', rec.score)
for st in ('s19','s20'):
    for rr in H.runs(stage=st, method='con'):
        if rr.arm.startswith('dev'):
            cc = collections.Counter((x.situation, x.solver_step) for x in H.records(rr))
            print(rr.folder, dict(cc))
