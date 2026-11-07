import sys, json, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
rs = H.runs(stage='s20', model='dsv41', method='con')
for r in rs: print(r.folder, r.arm)
rs19 = H.runs(stage='s19')
for r in rs19: print(r.folder, r.arm, r.model, r.method)
r = H.runs(stage='s20', model='dsv41', method='con', arm='real')[0]
recs = list(H.records(r))
print(len(recs))
rec = recs[0]
print(rec.scene_id, rec.episode_id, rec.situation, rec.labels)
print(json.dumps({k: v for k, v in rec.result.items() if k not in ('calls','people','constraints','candidates','solver')}, default=str)[:2000])
print(json.dumps(rec.score, default=str)[:2000])
print(rec.result.get('solver') and json.dumps(rec.result['solver'], default=str)[:1000])
