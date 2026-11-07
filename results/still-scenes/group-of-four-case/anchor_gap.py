"""Zero calls, render truth only. How close a nearer surface comes to a person target's chest point in the picture:
the pixel distance from the target's anchor pixel (its 1.1 m point, where models click) to the nearest pixel whose
depth is more than 0.3 m nearer than the target's own depth there. The development failure (conversation_group_4)
needs an occluder edge near the chest point, not a large hidden share.  python3 anchor_gap.py SCENE_DIR [...]"""
import json, sys
import numpy as np

for d in sys.argv[1:]:
    sc = json.load(open(d + '/scene.json'))
    t = next((t for t in sc['ground_truth']['targets'] if t['kind'] == 'person'), None)
    if t is None:
        print(f"{sc['scene_id']:55} no person target")
        continue
    ob = sc['observation']; K = ob['intrinsics']
    T = np.array(ob['T_world_camera']); Tinv = np.linalg.inv(T)
    c = Tinv @ np.array([t['anchor'][0], t['anchor'][1], t['anchor'][2], 1.0])
    u, v = K['fx'] * c[0] / c[2] + K['cx'], K['fy'] * c[1] / c[2] + K['cy']
    dep = np.load(d + '/depth.npy')
    H, W = dep.shape
    # the camera's z-depth of the target at its anchor; nearer = depth < z - 0.3
    near = (dep > 0) & (dep < c[2] - 0.3)
    ys, xs = np.nonzero(near[max(0, int(v) - 60):int(v) + 60, max(0, int(u) - 60):int(u) + 60])
    if xs.size:
        gap = float(np.min(np.hypot(xs + max(0, int(u) - 60) - u, ys + max(0, int(v) - 60) - v)))
    else:
        gap = float('inf')
    half_w = K['fx'] * 0.22 / c[2]              # a body's half width in pixels at that depth
    print(f"{sc['scene_id']:55} anchor ({u:.0f},{v:.0f}) depth {c[2]:.2f} m; nearer surface {gap:.1f} px from the chest point "
          f"({gap / half_w:.2f} body half-widths)")
