import sys
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
a = H.runs(stage='s8', model='qwen36_35', method='convlm')[0]
others = [H.runs(stage='s19', arm='ph', model='qwen38_low')[0], H.runs(stage='s10', model='qwen38_low', method='con')[0], H.runs(stage='s16b', arm='ph', model='qwen38_low', method='fsc')[0]]
lk = {}
for s in H.distinct_scenes():
    for k in s.listed: lk[k] = s.folder
ids = sorted({x.scene_id for x in H.records(a)})
print('s8 scenes_dir', a.scenes_dir, len(ids))
for b in others:
    bid = sorted({x.scene_id for x in H.records(b)})
    same = [lk.get((a.scenes_dir, i)) is not None and lk.get((a.scenes_dir, i)) == lk.get((b.scenes_dir, i)) for i in bid]
    eps_a = {(x.scene_id, x.episode_id) for x in H.records(a)}; eps_b = {(x.scene_id, x.episode_id) for x in H.records(b)}
    print(b.name, b.scenes_dir, 'n scenes', len(bid), 'same folder (non-None):', all(same), sum(same), 'episodes equal:', eps_a == eps_b)
