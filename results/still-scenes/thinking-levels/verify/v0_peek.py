import sys, json
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
recs = list(H.records('s10_qwen38_low_con'))
r = recs[0]
print(len(recs))
print('raw keys', list(r.raw.keys()))
print('result keys', list(r.result.keys()))
print('score', json.dumps(r.score)[:1500])
print('measures', json.dumps(r.measures)[:1500])
c = r.result['calls'][0]
print('call keys', list(c.keys()))
print(json.dumps({k:(v if not isinstance(v,str) or len(v)<200 else v[:200]) for k,v in c.items()}, default=str)[:2000])
for c in r.result['calls']:
    print(c['stage'], c.get('cached'), c.get('latency_s'), c.get('usage'))
print(r.timings)
