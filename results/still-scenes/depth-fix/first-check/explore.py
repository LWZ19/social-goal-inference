import sys, collections, json
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
for st in ['s16b','s20','s21','s23']:
    rs = H.runs(stage=st)
    print(st, len(rs), sorted({(r.arm, r.model, r.method) for r in rs}))
r = H.runs(stage='s21', arm='real', method='cot')[0]
print(r)
tally = collections.Counter()
recs = list(H.records(r, tally=tally))
print(tally, len(recs))
rec = recs[0]
print('labels', rec.labels)
print('score', rec.score)
print('result keys', list(rec.result.keys()))
print({k: rec.result[k] for k in rec.result if k not in ('calls',)})
print('raw keys', list(rec.raw.keys()))
print(collections.Counter(x.situation for x in recs))
print(collections.Counter(x.scene_id for x in recs))
cfg = H.recorded_config(r)[1]
print(cfg.run.heading_mode, cfg.thresholds)
