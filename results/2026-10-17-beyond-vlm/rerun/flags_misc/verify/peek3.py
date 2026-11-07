import sys, math, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
seen = {}
for arm in ('real','staffed','plain'):
    ref = H.runs(stage='s20', model='dsv41', method='con', arm=arm)[0]
    for rec in H.records(ref):
        if (arm, rec.scene_id) in seen: continue
        sc = H.scene(ref, rec.scene_id)
        T = sc.observation.camera.T_world_camera
        rb = sc.observation.robot
        seen[(arm, rec.scene_id)] = math.dist((rb.x, rb.y), (float(T[0,3]), float(T[1,3])))
        if rec.scene_id.startswith(('real_eval_02', 'real_eval_03')):
            print(rec.scene_id, rec.labels.get('factors', {}).get('view_angle_deg'))
print('scenes', len(seen), 'max camera-robot xy offset', max(seen.values()))
