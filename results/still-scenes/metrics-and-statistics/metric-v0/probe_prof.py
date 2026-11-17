import sys, time, cProfile, pstats
import numpy as np
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
r = H.runs(stage='s20', arm='real', model='dsv41', method='con')[0]
rec = next(x for x in H.records(r) if x.solver_step and x.situation == 'waiting_line')
sc = H.scene(rec.ref, rec.scene_id)
gt = H._against(rec)[1]
print([len(g.polygon) for g in gt.acceptable_regions], [len(g.polygon) for g in gt.forbidden_regions or []], len(sc.truth.people))
tx, ty = rec.target['world'][:2]
xs = np.linspace(tx-2, tx+2, 40)
poses = [(float(a), float(b), 0.0) for a in xs for b in np.linspace(ty-2, ty+2, 25)]
cProfile.run('H.score_poses(rec, poses)', '/tmp/prof.out')
pstats.Stats('/tmp/prof.out').sort_stats('cumulative').print_stats(15)
