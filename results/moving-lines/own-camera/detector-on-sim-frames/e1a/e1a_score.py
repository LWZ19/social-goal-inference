"""E1a scorer, zero calls: A1's frozen detector people  on the 13 Habitat
sim-benchmark first frames, against truth.json. A true person counts as visible when the point 1.1 m above them
projects into the image within 12 m and the depth there is no nearer than their distance minus 0.5 m. Matching is
one to one within 0.5 m (nearest first); a detection with no true person (visible or not) within 0.5 m is false.
Facing is checked for queue members against the bearing to their slot's face point.
    e1a_score.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RD = Path('results/2026-10-17-beyond-vlm')
SIM = Path('.superpowers/sdd/2026-10-07-social-goal-inference-pipeline/keep/sim_bench/frozen')
sys.dont_write_bytecode = True


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    P = load('a1pipeline', RD / 'phase1c/i09/a1/pipeline.py')
    setting = json.loads((RD / 'phase1c/i09/a1/chosen.json').read_text())
    s = setting.get('setting', setting)
    from sgi.geometry import Camera
    from sgi.geometry import project
    raw = {json.loads(l)['scene_id']: json.loads(l) for l in (HERE / 'raw_sim/rtdetr_v2_r50vd.jsonl').read_text().splitlines()}
    tot = {k: dict(vis=0, hit=0, false=0, imgs=0, err=[], face_ok=0, face_n=0) for k in ('A1',)}
    lines = []
    for case, r in sorted(raw.items()):
        d = SIM / case / '001'
        req = json.loads((d / 'request.json').read_text())
        truth = json.loads((d / 'truth.json').read_text())
        c = req['camera']
        cam = Camera(fx=float(c['fx']), fy=float(c['fy']), cx=float(c['cx']), cy=float(c['cy']), width=int(c['width']),
                     height=int(c['height']), T_world_camera=np.asarray(c['T_world_camera'], dtype=np.float64))
        depth = np.load(d / 'depth.npy').astype(np.float32)
        cam_xy = cam.T_world_camera[:2, 3]
        vis = []
        for h in truth['humans']:
            x, y = h['xy']
            pr = project(cam, (x, y, 1.1))
            if pr is None:
                continue
            u, v, z = pr
            if not (0 <= u < cam.width and 0 <= v < cam.height) or math.hypot(x - cam_xy[0], y - cam_xy[1]) > 12:
                continue
            dz = float(depth[int(v), int(u)])
            if np.isfinite(dz) and dz > 0 and dz < z - 0.5:
                continue
            vis.append(h)
        faces = {}
        for q in truth.get('queues', {}).values():
            for sl in q['slots']:
                if sl.get('face') is not None:
                    faces[(round(sl['x'], 2), round(sl['y'], 2))] = sl['face']
        people = {'C00': P.run_image(r, depth, cam, s, 'F5')}
        for key, pid in (('A1', 'C00'),):
            ppl = people[pid]
            t = tot[key]
            t['imgs'] += 1
            t['vis'] += len(vis)
            used = set()
            pairs = sorted(((math.hypot(p['xy'][0] - h['xy'][0], p['xy'][1] - h['xy'][1]), i, j)
                            for i, p in enumerate(ppl) for j, h in enumerate(vis)), key=lambda x: x[0])
            mi, mj = set(), set()
            for dist, i, j in pairs:
                if dist <= 0.5 and i not in mi and j not in mj:
                    mi.add(i); mj.add(j); t['hit'] += 1; t['err'].append(dist)
                    h = vis[j]
                    near = min(faces, key=lambda k: math.hypot(k[0] - h['xy'][0], k[1] - h['xy'][1])) if faces else None
                    if h.get('role') == 'queue' and near and math.hypot(near[0] - h['xy'][0], near[1] - h['xy'][1]) < 0.3:
                        fx, fy = faces[near]
                        want = math.atan2(fy - h['xy'][1], fx - h['xy'][0])
                        yaw = ppl[i].get('yaw')
                        t['face_n'] += 1
                        if yaw is not None and abs((yaw - want + math.pi) % (2 * math.pi) - math.pi) <= math.radians(45):
                            t['face_ok'] += 1
            for i, p in enumerate(ppl):
                if i in mi:
                    continue
                if not any(math.hypot(p['xy'][0] - h['xy'][0], p['xy'][1] - h['xy'][1]) <= 0.5 for h in truth['humans']):
                    t['false'] += 1
            if key == 'A1':
                lines.append(f'  {case:<20} visible {len(vis):>2}  A1 people {len(ppl):>2}  matched {len(mj):>2}')
    out = ['E1a: A1 detector people on the 13 Habitat sim-benchmark first frames, against truth.json'] + lines
    for k, t in tot.items():
        out.append(f'{k}: recall {t["hit"]}/{t["vis"]} = {t["hit"] / max(t["vis"], 1):.3f}; false people {t["false"]} '
                   f'({t["false"] / t["imgs"]:.2f} per image); median xy error {np.median(t["err"]):.3f} m; '
                   f'queue members facing within 45 deg {t["face_ok"]}/{t["face_n"]}')
    text = '\n'.join(out)
    print(text)
    (HERE / 'e1a_out.txt').write_text(text + '\n')


if __name__ == '__main__':
    main()
