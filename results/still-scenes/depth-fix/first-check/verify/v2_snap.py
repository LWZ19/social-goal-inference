"""Independent snap recompute (verifier). R/E marked points (Qwen s21, five others s23; real/staffed/plain; cot/fsc)
moved onto the nearest selectable cell of sgi.solver.floor (C's clearance rule), heading face_target, rescored."""
import sys, collections, json, math, time
import numpy as np
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
from sgi.solver import floor as c_floor
from sgi.stages.goal import _poses
from sgi.types import DetectedPerson
from sgi.geometry import bearing, wrap_angle

V = 'results/2026-10-17-beyond-vlm/rerun/snap/verify/'
MODELS = [('qwen38_low', 's21'), ('glm53', 's23'), ('dsv41', 's23'), ('haiku55', 's23'), ('sonnet55_low', 's23'), ('gemini38_low', 's23')]
ARMS = ['real', 'staffed', 'plain']
FLOORS = {}

def floor_of(ref, scene_id, people=()):
    sc = H.scene(ref, scene_id)
    th = H.recorded_config(ref)[1].thresholds
    if people:
        return c_floor(sc.observation, people, th), sc, th
    key = (str(ref.scenes_dir), scene_id, th.model_dump_json())
    if key not in FLOORS:
        FLOORS[key] = c_floor(sc.observation, [], th)
    return FLOORS[key], sc, th

def snap_xy(fl, x, y):
    """Keep (x, y) when its cell is selectable, else the nearest selectable cell centre (first in grid order on a tie)."""
    cell = fl.index(x, y)
    if cell is not None and fl.selectable[cell]:
        return (x, y), 0.0
    cen = fl.centres()[fl.selectable]               # (N,2), grid (row-major) order
    d = np.hypot(cen[:, 0] - x, cen[:, 1] - y)
    i = int(np.argmin(d))
    return (float(cen[i, 0]), float(cen[i, 1])), float(d[i])

def klass(s):
    d = s.distance_to_region_m
    if d is None: return 'none'
    if s.in_region: return 'inreg'
    if d <= 0.3: return 'near'
    if d <= 1.0: return 'mid'
    return 'far'

out = open(V + 'v2_snap_records.jsonl', 'w')
t0 = time.time()
hd_bad = 0; hd_n = 0
for model, stage in MODELS:
    cidx = {}
    for arm in ARMS:
        (cref,) = H.runs(stage=stage, arm=arm, model=model, method='con')
        for c in H.records(cref):
            cidx[(arm, c.scene_id, c.episode_id, c.trial)] = c
    for meth in ['cot', 'fsc']:
        for arm in ARMS:
            (ref,) = H.runs(stage=stage, arm=arm, model=model, method=meth)
            for rec in H.records(ref):
                row = dict(model=model, method=meth, arm=arm, scene=rec.scene_id, ep=rec.episode_id, trial=rec.trial,
                           sit=rec.situation, source=rec.result.get('source'), status=rec.result.get('status'),
                           ok=bool(rec.score.get('success')), tc=bool(rec.score.get('target_correct')),
                           d=rec.score.get('distance_to_region_m'), gap=rec.score.get('min_body_gap_m'),
                           inreg=rec.score.get('in_region'), reach=rec.score.get('reachable'), head=rec.score.get('heading_ok'))
                c = cidx.get((arm, rec.scene_id, rec.episode_id, rec.trial))
                row['c_ok'] = None if c is None else bool(c.score.get('success'))
                if rec.result.get('source') == 'direct_marking' and rec.result.get('status') == 'ok' and rec.pose_world is not None:
                    fl, sc, th = floor_of(ref, rec.scene_id)
                    x, y, th0 = rec.pose_world
                    tw = rec.target['world']
                    robot = sc.observation.robot
                    # recorded heading is face_target?
                    pw, _, _ = _poses(robot, (x, y), tw[:2], th)
                    hd_n += 1
                    if abs(wrap_angle(pw[2] - th0)) > 1e-6: hd_bad += 1
                    (sx, sy), dm = snap_xy(fl, x, y)
                    pwD, _, _ = _poses(robot, (sx, sy), tw[:2], th)
                    poses = [tuple(pwD)]
                    row['movedD'] = dm
                    # snapP: C's people (same arm/scene/episode/trial) added as discs, computed for failures only
                    if (not row['ok']) and row['tc'] and c is not None and c.people:
                        people = [DetectedPerson.model_validate(p) for p in c.people]
                        flP, _, _ = floor_of(ref, rec.scene_id, people)
                        (px, py), dp = snap_xy(flP, x, y)
                        pwP, _, _ = _poses(robot, (px, py), tw[:2], th)
                        poses.append(tuple(pwP)); row['movedP'] = dp
                    scs = H.score_poses(rec, poses)
                    sD = scs[0]
                    row.update(okD=bool(sD.success), dD=sD.distance_to_region_m, gapD=sD.min_body_gap_m, inregD=sD.in_region,
                               reachD=sD.reachable, headD=sD.heading_ok)
                    if len(scs) > 1:
                        row.update(okP=bool(scs[1].success), dP=scs[1].distance_to_region_m, gapP=scs[1].min_body_gap_m)
                    else:
                        row.update(okP=bool(sD.success), dP=sD.distance_to_region_m, gapP=sD.min_body_gap_m, P_eq_D=True)
                    # geometry for mechanisms: line axis etc.
                    gt = sc.truths[rec.episode_id].gt
                    if rec.situation == 'waiting_line' and row['tc']:
                        tid = rec.score.get('matched_target_id')
                        gid = sc.truth.group_for(rec.episode_id, tid)
                        mem = [p for p in sc.truth.people if p.group == gid]
                        a = sc.truth.targets[tid].anchor[:2]
                        line = sorted(((m.x, m.y) for m in mem), key=lambda q: math.dist(a, q))
                        last = line[-1]; before = line[-2] if len(line) > 1 else a
                        fx, fy = (before[0] - last[0]), (before[1] - last[1]); nf = math.hypot(fx, fy); fx, fy = fx / nf, fy / nf
                        row['not_ahead'] = (x - last[0]) * fx + (y - last[1]) * fy <= 1e-9
                        row['not_aheadD'] = (sx - last[0]) * fx + (sy - last[1]) * fy <= 1e-9
                        ax_, ay_ = last[0] - a[0], last[1] - a[1]; L = math.hypot(ax_, ay_)
                        ux, uy = ax_ / L, ay_ / L
                        row['t'] = ((x - a[0]) * ux + (y - a[1]) * uy) / L
                        row['t_first'] = ((line[0][0] - a[0]) * ux + (line[0][1] - a[1]) * uy) / L
                        lat = (x - a[0]) * (-uy) + (y - a[1]) * ux    # signed lateral offset from axis
                        cam = sc.observation.camera.T_world_camera[:2, 3]
                        latcam = (cam[0] - a[0]) * (-uy) + (cam[1] - a[1]) * ux
                        row['lat_camside'] = lat * (1 if latcam > 0 else -1)   # positive = toward the camera's side
                        row['d_last'] = math.dist((x, y), last)
                        row['d_anchor'] = math.dist((x, y), a)
                        row['d_first'] = math.dist((x, y), line[0])
                        # pixel offset to representative point and ground range to the camera
                        reg = [r for r in gt.acceptable_regions if r.target_id in (None, tid)]
                        rp = reg[0].representative_point
                        from sgi.geometry import project
                        cam_ = sc.observation.camera
                        rpix = project(cam_, (rp[0], rp[1], 0.0))
                        gp = rec.result.get('goal_pixel')
                        if rpix is not None and gp is not None:
                            row['dv'] = gp[1] - rpix[1]
                        row['drange'] = math.dist((x, y), cam) - math.dist(rp, cam)
                    # nearest true person centre distance
                    row['d_person'] = min((math.dist((x, y), (p.x, p.y)) for p in sc.truth.people), default=None)
                out.write(json.dumps(row) + '\n')
        print(model, meth, 'done', f'{time.time()-t0:.0f}s', flush=True)
out.close()
print('heading face_target check: bad', hd_bad, 'of', hd_n)
