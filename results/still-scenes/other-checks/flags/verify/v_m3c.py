"""M3 variant: the start taken at the camera's floor position (T_world_camera x, y) instead of observation.robot."""
import sys, collections, math
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
from sgi.polygons import point_in_polygon
from sgi.geometry import to_world_xy
src = open(__file__.replace('v_m3c.py', 'v_m3.py')).read()
exec(src.split("p('== M3 primary")[0]); out.clear()
C = runs_for([('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41')])
recs = []
for ref in C: recs += list(H.records(ref))
recs.sort(key=lambda r: (str(r.ref.scenes_dir), r.scene_id))
n = A = B = S = SA = 0; per = collections.Counter()
for r in recs:
    pr = r.result.get('pose_robot')
    if r.result.get('status') != 'ok' or pr is None: continue
    sc = H.scene(r.ref, r.scene_id); T = sc.observation.camera.T_world_camera
    a = (float(T[0, 3]), float(T[1, 3])); b = to_world_xy(sc.observation.robot, pr[0], pr[1])
    regs = sc.truths[r.episode_id].gt.forbidden_regions or []
    hit = any(seg_meets_poly(a, b, g.polygon) for g in regs)
    endin = any(point_in_polygon(b, [(float(x), float(y)) for x, y in g.polygon]) for g in regs)
    n += 1; A += hit; B += hit and not endin; per[r.model] += hit
    if r.score.get('success'): S += 1; SA += hit
p(f'camera start: n={n} A={A} ({100*A/n:.2f}%) B={B} ({100*B/n:.2f}%) successes {SA}/{S} per model {dict(per)}')
open(__file__.replace('.py', '.txt'), 'w').write('\n'.join(out) + '\n')
