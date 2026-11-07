import sys, json, collections
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
for r in H.runs(stage='s8'):
    print(r.name, r.model, r.method, r.arm)
r = H.runs(stage='s8', method='convlm', model='glm53')[0]
yaml, cfg = H.recorded_config(r)
print(cfg.run)
recs = list(H.records(r))
print(len(recs))
rec = next(x for x in recs if x.solver_step)
print(rec.scene_id, rec.episode_id, rec.trial, rec.situation)
print(list(rec.raw.keys()))
print(rec.raw.get('config'))
print(json.dumps(rec.result.get('people'), indent=0)[:800])
print(json.dumps(rec.result.get('constraints'), indent=0)[:1500])
print(json.dumps(rec.result.get('target'), indent=0)[:500])
print(rec.score)
print(list(rec.result.keys()))
print(rec.raw.get('measures'))
calls = rec.raw.get('calls') or rec.result.get('calls')
print(type(calls), len(calls) if calls else None)
if calls: print([ (c.get('stage'), c.get('cached')) for c in calls])
