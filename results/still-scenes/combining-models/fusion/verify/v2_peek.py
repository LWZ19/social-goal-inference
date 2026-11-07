import sys, collections, json
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
r = H.runs(stage='s19', arm='real', model='glm53', method='con')[0]
recs = list(H.records(r))
print(len(recs))
for x in recs:
    if x.people and x.constraints:
        print('keys result', sorted(x.result.keys()))
        print('score', x.score)
        print('people', type(x.people).__name__, json.dumps(x.people)[:400])
        print('constraints', type(x.constraints).__name__, json.dumps(x.constraints)[:900])
        print('trial', x.trial, x.scene_id, x.episode_id, x.situation)
        print('timings', json.dumps(x.timings)[:600])
        break
print(collections.Counter(x.result.get('source') for x in recs))
print(collections.Counter(type(x.people).__name__ for x in recs))
print(collections.Counter(type(x.constraints).__name__ for x in recs))
