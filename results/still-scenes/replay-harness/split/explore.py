import sys, collections, json
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
for st in ['s16b','s19','s20','s21','s22','s23']:
    for r in H.runs(stage=st):
        print(st, r.arm, r.model, r.method, r.folder)
r = H.runs(stage='s23', arm='real', model='haiku55', method='cot')[0]
recs = list(H.records(r))
print(len(recs))
rec = recs[0]
print(json.dumps(rec.score, indent=1, default=str))
print(rec.labels)
print({k: (v if not isinstance(v,(list,dict)) else type(v).__name__) for k,v in rec.result.items()})
print(collections.Counter(x.situation for x in recs))
print(collections.Counter(x.trial for x in recs))
print(collections.Counter(x.result.get('status') for x in recs))
