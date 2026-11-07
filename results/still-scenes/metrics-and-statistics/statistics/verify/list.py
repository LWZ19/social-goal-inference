import sys
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
for st in ('s16b','s19','s20','s21','s22','s23'):
    for r in H.runs(stage=st):
        if r.arm not in ('real','staffed','plain'): continue
        print(st, r.arm, r.model, r.method, r.name)
r = [x for x in H.runs(stage='s21') if x.arm=='real'][0]
recs = H.records(r)
print(len(recs)); print(recs[0].scene_id, recs[0].episode_id, recs[0].trial, recs[0].situation, recs[0].score)
