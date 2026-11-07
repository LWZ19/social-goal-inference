import sys, json
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
r = [x for x in H.runs(stage='s23', arm='real', model='gemini38_low') if x.method=='con'][0]
for rec in H.records(r):
    if rec.people:
        print(rec.scene_id, rec.episode_id, rec.trial, rec.situation, rec.result['status'])
        print(json.dumps(rec.people)[:600])
        for c in rec.result['calls']:
            print({k:(str(v)[:300]) for k,v in c.items()})
        break
cfgraw, cfg = H.recorded_config(r)
print(cfg.thresholds)
print(json.dumps(cfgraw)[:1500])
