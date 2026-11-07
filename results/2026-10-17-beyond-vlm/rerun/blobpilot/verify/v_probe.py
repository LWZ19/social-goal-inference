import sys, numpy as np
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
from sgi.geometry import lift_all
ds = H.distinct_scenes()
for s in ds:
    if s.scene_id not in ('real_eval_09_waiting_line_6', 'eval_09_waiting_line_6', 'eval_15_audience_5'): continue
    sc = H.scene(s.runs[0], s.scene_id)
    obs = sc.observation
    cam = obs.camera
    print(s.scene_id, obs.depth.shape, obs.depth.dtype, cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height)
    print('cam pos', cam.T_world_camera[:3,3])
    P = lift_all(obs.depth, cam)
    print('N', len(P), 'z pct', np.percentile(P[:,2], [0,1,5,50,95,99,100]).round(3))
    print('x range', P[:,0].min(), P[:,0].max(), 'y range', P[:,1].min(), P[:,1].max())
    for p in sc.truth.people: print('  ', p.id, round(p.x,2), round(p.y,2), round(p.yaw,2), p.group, p.role)
    print('targets', {k:(v.kind, v.anchor) for k,v in sc.truth.targets.items()})
    print([f for f in dir(obs) if not f.startswith('_')])
