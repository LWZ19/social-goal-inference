import sys, json, time
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
r = H.runs(stage='s8', method='conorc', model='glm53')[0]
yaml, cfg = H.recorded_config(r)
print(cfg.run)
recs = list(H.records(r))
rec = next(x for x in recs if x.solver_step)
c = rec.result['calls'][0]
print(list(c.keys()))
print({k: (str(v)[:200]) for k, v in c.items()})
print(rec.raw['plan'] if isinstance(rec.raw['plan'], (str,)) else json.dumps(rec.raw['plan'])[:400])
t=time.time(); rs = H.resolve(rec); print('resolve', time.time()-t, rs.reproduced, rs.score.success)
t=time.time(); rs = H.resolve(rec); print('resolve2', time.time()-t)
print(rec.result['people'][:2])
