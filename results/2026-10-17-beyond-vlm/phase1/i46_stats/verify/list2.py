import sys, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
for st in ('s21','s22','s16b'):
    r = [x for x in H.runs(stage=st, model='qwen38_low', method='cot') if x.arm=='real'][0]
    recs = list(H.records(r))
    print(st, len(recs)); print(recs[0].scene_id, recs[0].episode_id, recs[0].trial, recs[0].situation, recs[0].score)
    print(collections.Counter(x.trial for x in recs))
    print(sorted(set(x.scene_id for x in recs)))
