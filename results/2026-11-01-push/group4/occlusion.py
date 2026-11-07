"""Zero calls. How much of each truth person's image silhouette a nearer person hides, from scene truth alone:
people as upright cylinders (radius 0.25 m, 0-1.8 m tall), sampled on their camera-facing surface, projected; a
sample is hidden when a nearer person's cylinder covers the same pixel. Used to select and bin person targets for
SNAP_FROZEN.md before any model call.  python3 occlusion.py SCENE_DIR [...]"""
import json, math, sys
import numpy as np
R, H = 0.25, 1.8


def project(Tinv, K, p):
    c = Tinv @ np.array([p[0], p[1], p[2], 1.0])
    if c[2] <= 0.1:
        return None
    return K['fx'] * c[0] / c[2] + K['cx'], K['fy'] * c[1] / c[2] + K['cy'], c[2]


def silhouettes(scene):
    ob = scene['observation']
    T = np.array(ob['T_world_camera']); Tinv = np.linalg.inv(T); K = ob['intrinsics']
    cam = T[:3, 3]
    out = {}
    for p in scene['ground_truth']['people']:
        to_cam = np.array([cam[0] - p['x'], cam[1] - p['y']]); to_cam /= np.linalg.norm(to_cam)
        side = np.array([-to_cam[1], to_cam[0]])
        pts = []
        for a in np.linspace(-0.95, 0.95, 15):           # across the visible half-cylinder
            off = R * (a * side + math.sqrt(1 - a * a) * to_cam)
            for z in np.linspace(0.05, H - 0.05, 18):
                q = project(Tinv, K, (p['x'] + off[0], p['y'] + off[1], z))
                if q and 0 <= q[0] < ob['width'] and 0 <= q[1] < ob['height']:
                    pts.append(q)
        out[p['id']] = dict(pts=pts, xy=(p['x'], p['y']))
    return out, Tinv, K


def covers(Tinv, K, person, u, v, depth):
    """Whether this person's cylinder covers pixel (u, v) nearer than `depth` (approximate: its projected box)."""
    a = project(Tinv, K, (person['xy'][0], person['xy'][1], 0.0))
    b = project(Tinv, K, (person['xy'][0], person['xy'][1], H))
    if not a or not b:
        return False
    half = K['fx'] * R / a[2]
    return a[2] < depth and abs(u - a[0]) <= half and min(a[1], b[1]) <= v <= max(a[1], b[1])


def hidden_share(scene, pid):
    s, Tinv, K = silhouettes(scene)
    pts = s[pid]['pts']
    if not pts:
        return None, None
    hid = sum(any(covers(Tinv, K, s[o], u, v, d) for o in s if o != pid) for u, v, d in pts)
    return hid / len(pts), [o for o in s if o != pid and any(covers(Tinv, K, s[o], u, v, d) for u, v, d in pts)]


if __name__ == '__main__':
    for d in sys.argv[1:]:
        sc = json.load(open(d + '/scene.json'))
        for t in sc['ground_truth']['targets']:
            if t['kind'] == 'person':
                share, by = hidden_share(sc, t['person_id'])
                print(f"{sc['scene_id']:40} target {t['id']} = {t['person_id']} ({t['label']}): hidden {share:.0%} by {by}")
