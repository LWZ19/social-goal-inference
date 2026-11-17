import sys, json
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
for st in ['s16b','s19','s20','s21','s23','s8']:
    for r in H.runs(stage=st):
        print(st, r.arm, r.model, r.method, r.folder)
r = [x for x in H.runs(stage='s19', arm='real') if x.method=='con'][0]
recs = list(H.records(r))
rec = recs[0]
print(rec.scene_id, rec.episode_id, rec.trial, rec.situation)
print(json.dumps(rec.people)[:800])
print(list(rec.raw.keys()))
print(list(rec.result.keys()))
for k,v in rec.raw.items():
    if k not in ('result',):
        print(k, str(v)[:400])
