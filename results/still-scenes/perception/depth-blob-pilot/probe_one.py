"""Probe: load a few scenes through the harness and print the people truth, image size and load time."""
import sys, time
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H

want = {'staffed_eval_06_waiting_line_1', 'eval_06_waiting_line_1', 'real_eval_12_conversation_group_4'}
for s in H.distinct_scenes():
    if s.scene_id not in want:
        continue
    t = time.time()
    sc = H.scene(s.runs[0], s.scene_id)
    obs = sc.observation
    print(s.scene_id, 'load %.1fs' % (time.time() - t), obs.depth.shape, obs.depth.dtype, 'cam z %.2f' % obs.camera.T_world_camera[2, 3],
          'tour', len(sc.inputs.tour))
    for p in sc.truth.people:
        print('   ', p.id, round(p.x, 2), round(p.y, 2), round(p.yaw, 2), p.group, p.role)
    print('   targets', [(t.id, t.kind, t.person_id) for t in sc.truth.targets.values()])
    H._SCENES.clear()
