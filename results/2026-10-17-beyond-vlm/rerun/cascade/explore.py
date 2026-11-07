import sys, collections, json
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
for st in ('s19','s21','s23'):
    for r in H.runs(stage=st):
        if r.method=='con' or st!='s19':
            print(st, r.folder, r.arm, r.model, r.method)
r = H.runs(stage='s19', arm='real', model='glm53', method='con')[0]
recs = list(H.records(r))
print(len(recs))
x = recs[0]
print(x.scene_id, x.episode_id, x.trial, x.situation, x.labels)
print('result keys', list(x.result.keys()))
print('score keys', x.score)
print('raw keys', list(x.raw.keys()))
print('measures', x.measures)
print('timings', json.dumps(x.timings, indent=0)[:1500])
print('calls', [ (c['stage'], c.get('cached'), c.get('latency_s')) for c in x.result.get('calls')])
print('solver', type(x.solver), str(x.solver)[:300])
print('cands', str(x.candidates)[:500])
print('constraints', str(x.constraints)[:500])
print(collections.Counter((y.situation, y.solver is not None, y.solver_step, y.result.get('source')) for y in recs))
