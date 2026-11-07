import sys, json, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
for st in ('s19','s20'):
    for r in H.runs(stage=st):
        print(st, r.arm, r.model, r.method, r.folder)
run = H.runs(stage='s20', arm='real', model='dsv41', method='con')[0]
recs = list(H.records(run))
print(len(recs))
r = recs[0]
print({k: (v if not isinstance(v,(list,dict)) else type(v).__name__) for k,v in r.result.items()})
print(r.result.get('flags'))
print(r.score)
print(r.labels)
print(collections.Counter((x.result.get('source'), x.situation) for x in recs))
fl = collections.Counter()
for x in recs:
    for k,v in (x.result.get('flags') or {}).items():
        if v: fl[k]+=1
print(fl)
