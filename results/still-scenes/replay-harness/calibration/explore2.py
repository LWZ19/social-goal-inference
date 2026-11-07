import sys, collections, json
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
r = H.runs(stage='s8', model='glm53', method='convlm')[0]
recs = list(H.records(r))
rec = next(x for x in recs if x.solver_step and x.situation == 'waiting_line')
print(rec.scene_id, rec.episode_id, rec.trial)
print('result keys', list(rec.result.keys()))
print('raw keys', list(rec.raw.keys()))
for c in rec.result['calls']:
    print({k: c.get(k) for k in ('stage', 'cached', 'model', 'latency_s')}, list(c.keys()))
print(json.dumps(rec.result['people'], indent=0)[:800])
print(json.dumps([u['constraint'] for u in rec.result['constraints']], indent=0)[:1500])
print('target', rec.result['target'])
print('context', rec.result.get('context_check'))
print('measures', rec.measures)
print('score', rec.score)
