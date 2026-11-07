import sys, collections, time
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
import numpy as np
from sgi import solver
from sgi.types import DetectedPerson
run = H.runs(stage='s20', arm='real', model='dsv41', method='con')[0]
recs = [r for r in H.records(run) if r.solver_step]
print(len(recs), collections.Counter(r.situation for r in recs))
r = recs[0]
print(r.result['target'])
print(r.candidates[0])
print(r.people[:1])
sc = H.scene(run, r.scene_id)
th = H.recorded_config(run.__class__ and run)[1].thresholds
t=time.time(); fl = solver.floor(sc.observation, [DetectedPerson.model_validate(p) for p in r.people], th); print('floor', time.time()-t, fl.selectable.sum())
ep = sc.truths[r.episode_id].gt
print(type(ep), ep.acceptable_regions[0] if ep.acceptable_regions else None)
print([ (f.kind) for f in ep.forbidden_regions or []])
cen = fl.centres()[fl.selectable]
tx, ty = r.result['target']['world'][:2]
d = np.hypot(cen[:,0]-tx, cen[:,1]-ty)
disc = cen[d<=3.0]
print('disc', len(disc))
poses = [(x, y, float(np.arctan2(ty-y, tx-x))) for x, y in disc[:500]]
t=time.time(); s = H.score_poses(r, poses); print('score 500', time.time()-t)
for st in ('s19',):
  for rr in H.runs(stage=st, arm='dev_seed0', method='con'):
    print(collections.Counter(x.situation for x in H.records(rr) if x.solver_step))
