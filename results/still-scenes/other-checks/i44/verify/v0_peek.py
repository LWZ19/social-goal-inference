import sys, collections, json
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
r=H.runs(stage='s8', model='qwen36_35', method='convlm')[0]
recs=list(H.records(r))
x=recs[0]
print(x.scene_id, x.episode_id, x.trial, x.situation)
print('score', x.score)
for c in x.result.get('calls',[]):
    print({k:(v if k not in ('raw_text','prompt','messages','text') else str(v)[:60]) for k,v in c.items()})
print(x.timings)
print(sorted(x.raw.keys()))
print(x.raw.get('config',{}) if 'config' in x.raw else '')
