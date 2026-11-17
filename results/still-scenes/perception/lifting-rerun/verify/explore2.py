import sys
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
r = H.runs(stage='s20', arm='real', model='dsv41', method='con')[0]
seen=set()
for rec in H.records(r):
    if rec.score['target_correct']: continue
    sc, gt, th, _ = H._against(rec)
    for tid in gt.target_ids:
        t = sc.truth.targets[tid]
        key=(rec.scene_id,tid)
        if key in seen: continue
        seen.add(key)
        print(rec.scene_id, rec.episode_id, tid, t.kind, t.label, 'anchor', t.anchor, 'r', t.match_radius, 'fp', t.footprint is not None, 'pid', t.person_id)
sc = H.scene(r, 'real_eval_12_conversation_group_4') if False else None
rec0 = [x for x in H.records(r) if 'eval_12' in x.scene_id][0]
S = H.scene(r, rec0.scene_id)
print('bounds', S.truth.bounds)
print('obstacles', type(S.truth.obstacles), (S.truth.obstacles[:3] if hasattr(S.truth.obstacles,'__getitem__') else S.truth.obstacles))
print('people', S.truth.people)
print(th)
