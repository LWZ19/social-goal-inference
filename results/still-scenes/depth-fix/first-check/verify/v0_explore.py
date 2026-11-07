import sys, collections
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
for st in ['s16b','s20','s21','s23','s19']:
    rs = H.runs(stage=st)
    print(st, len(rs))
    for r in rs: print('  ', r.folder, r.arm, r.model, r.method)
r = H.runs(stage='s23', arm='real', model='sonnet55_low', method='cot')[0]
recs = list(H.records(r))
rec = recs[0]
print(rec.situation, rec.labels)
print(sorted(rec.result.keys()))
print(sorted(rec.score.keys()))
print({k: rec.result[k] for k in rec.result if k not in ('calls',)})
print(collections.Counter(x.situation for x in recs))
print(collections.Counter((x.result.get('source'), x.result.get('status')) for x in recs))
cfg = H.recorded_config(r)[1]
print(cfg.thresholds)
