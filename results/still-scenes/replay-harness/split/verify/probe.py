import sys, json
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
rs = H.runs(stage='s23')
print(len(rs))
for r in rs[:3]: print(r.name, r.arm, r.model, r.method)
r = H.runs(stage='s23', arm='real', model='gemini38_low', method='con')[0]
recs = list(H.records(r))
print(len(recs))
rec = recs[0]
print(json.dumps(rec.score, indent=1))
print(rec.result.keys())
print(rec.result.get('target'))
print(rec.result.get('source'), rec.result.get('status'))
print(rec.timings)
print(rec.labels)
print(rec.scene_id, rec.episode_id, rec.trial, rec.situation)
for s in ['s16b','s19','s20','s21','s22','s23']:
    print(s, sorted(set((x.arm,x.model,x.method) for x in H.runs(stage=s))))
