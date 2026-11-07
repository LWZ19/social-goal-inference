"""Probe: look at one solver-step record of s19 glm53 con real (structure only)."""
import json
import sys
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H

ref = H.runs(stage='s19', arm='real', model='glm53', method='con')[0]
print(ref.folder)
th = H.recorded_config(ref)[1].thresholds
print('max_people', th.max_people, 'person_radius', th.person_radius_m, 'standoff', th.standoff_m)
n = 0
for rec in H.records(ref):
    if rec.solver_step and rec.situation == 'waiting_line':
        print(rec.scene_id, rec.episode_id, rec.trial, rec.situation)
        print(' people', json.dumps(rec.people)[:600])
        print(' constraints', json.dumps(rec.constraints)[:900])
        print(' target', rec.target)
        print(' score', {k: v for k, v in rec.score.items()})
        sc = H.scene(ref, rec.scene_id)
        gt = sc.truths[rec.episode_id].gt
        print(' gt target_ids', gt.target_ids, 'situation', gt.situation)
        print(' truth people', [(p.id, round(p.x, 2), round(p.y, 2), round(p.yaw, 2), p.group, p.role) for p in sc.truth.people])
        r = H.resolve(rec)
        print(' resolve moved', r.moved_m, r.score.success)
        n += 1
        if n >= 1:
            break
