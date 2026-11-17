import sys, time, math, collections
import numpy as np
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
from sgi import solver
from sgi.types import DetectedPerson
r = H.runs(stage='s20', arm='real', model='dsv41', method='con')[0]
th = H.recorded_config(r)[1].thresholds
print('th', th.robot_radius_m, th.person_radius_m, th.grid_resolution_m, th.floor_range_m, th.candidate_count, th.heading_degenerate_m)
recs = [x for x in H.records(r) if x.solver_step][:6]
for rec in recs:
    t0 = time.time()
    sc = H.scene(rec.ref, rec.scene_id)
    t1 = time.time()
    people = [DetectedPerson.model_validate(p) for p in rec.people]
    fl = solver.floor(sc.observation, people, th)
    t2 = time.time()
    cen = fl.centres()[fl.selectable]
    tx, ty = rec.target['world'][:2]
    d = np.hypot(cen[:,0]-tx, cen[:,1]-ty)
    disc = cen[d <= 3.0]
    th_ = np.arctan2(ty - disc[:,1], tx - disc[:,0])
    poses = [(float(a), float(b), float(c)) for (a, b), c in zip(disc, th_)]
    t3 = time.time()
    s = H.score_poses(rec, poses)
    t4 = time.time()
    gt = H._against(rec)[1]
    print(rec.situation, 'sel', len(cen), 'disc', len(disc), f'scene {t1-t0:.2f} floor {t2-t1:.2f} score {t4-t3:.2f} s = {1e6*(t4-t3)/len(disc):.0f} us/pose',
          'succ', sum(x.success for x in s), 'inreg', sum(x.in_region for x in s), 'forb', collections.Counter(k for x in s for k, v in x.forbidden.items() if v),
          'tol', [g.tolerance_deg for g in gt.acceptable_regions], [g.heading for g in gt.acceptable_regions][:1], [(g.kind) for g in gt.forbidden_regions or []])
