import sys, json
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
rs = H.runs(stage='s19', arm='real', model='glm53', method='con')
print([r.folder for r in rs])
recs = list(H.records(rs[0]))
print(len(recs))
r = next(x for x in recs if x.situation.startswith('line') or 'line' in x.situation or 'queue' in x.situation) if any('line' in x.situation or 'queue' in x.situation for x in recs) else recs[0]
print(sorted(set(x.situation for x in recs)))
print('raw keys', sorted(r.raw.keys()))
print('result keys', sorted(r.result.keys()))
print('score', json.dumps(r.score, indent=1)[:3000])
print('measures', json.dumps(r.measures, indent=1)[:3000])
print('labels', r.labels)
