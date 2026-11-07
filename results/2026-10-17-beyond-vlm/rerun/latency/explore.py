import sys, json, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
rs = H.runs()
print(len(rs))
for r in rs:
    if r.stage in ('s19','s21','s16b','s23','s20','s22'):
        print(r.stage, r.arm, r.model, r.method, r.folder)
ref = H.runs(stage='s19', arm='real', model='qwen38_low', method='con')[0]
recs = list(H.records(ref))
r = recs[0]
print(len(recs))
print(json.dumps({k: (v if k!='result' else None) for k, v in r.raw.items()}, default=str)[:3000])
res = r.result
print(sorted(res.keys()))
print(res.get('timings_s'))
for c in res.get('calls', []):
    print({k: v for k, v in c.items() if k not in ('request','response','text','raw','prompt','messages')})
print(r.measures)
print(r.timings)
