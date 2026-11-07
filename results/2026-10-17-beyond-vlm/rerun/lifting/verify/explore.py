import sys, json, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
for st in ('s19','s20'):
    for r in H.runs(stage=st, method='con'):
        print(st, r.name, r.arm, r.model, r.method)
r = H.runs(stage='s19', arm='real', model='glm53', method='con')[0]
recs = list(H.records(r))
print(len(recs))
rec = [x for x in recs if not x.score['target_correct']][0]
print(rec.scene_id, rec.episode_id, rec.trial, rec.situation)
print('result keys', list(rec.result.keys()))
print('target', rec.result.get('target'))
print('status', rec.result.get('status'))
print('score', rec.score)
print('calls', [(c['stage'], c.get('cached')) for c in rec.result.get('calls') or []])
print('raw keys', list(rec.raw.keys()))
sc = H.scene(r, rec.scene_id)
print(type(sc.inputs), [a for a in dir(sc.inputs) if not a.startswith('_')])
print(type(sc.observation), [a for a in dir(sc.observation) if not a.startswith('_')])
print(type(sc.truth), [a for a in dir(sc.truth) if not a.startswith('_')])
et = sc.truths[rec.episode_id]
print(type(et), [a for a in dir(et) if not a.startswith('_')])
print('gt', [a for a in dir(et.gt) if not a.startswith('_')])
print('target_ids', et.gt.target_ids)
for tid in et.gt.target_ids:
    print(tid, sc.truth.targets[tid])
print('people', sc.truth.people[:2])
print('depth', sc.observation.depth.shape, sc.observation.depth.dtype)
print('cam', sc.observation.camera)
