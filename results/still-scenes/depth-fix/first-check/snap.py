"""SNAP (I01 deliverable 4, I34 Stage 0): R and E marked points moved by C's clearance snap, and rescored.

Reads runs and scenes only through research_harness. No model call, nothing tuned.

C's clearance rule (src/sgi/solver.py `floor`): the constraint method may stop only on a *selectable* cell of the
solver's floor grid (0.05 m cells in the robot's frame): a cell the picture shows as floor (seen) or the strip under
the picture, not *blocked* (within standard_goal_clearance_m = 0.40 m of a depth point that stands between the floor
tolerance and the camera's height, i.e. walls, furniture and bodies as depth sees them; and within person_radius_m +
robot_radius_m = 0.55 m of a listed person's centre), and joined to the robot through passable cells. The solver only
ever picks among those cells, so every C pose passes this rule.

The snap: a marked point (source direct_marking, status ok) whose cell is selectable is left exactly where it is;
otherwise it moves to the nearest selectable cell centre (Euclidean; ties by the grid's row-major order, as the
solver's last tie-break). The heading is the method's own rule, face_target: the bearing from the new position to the
target's lifted point (sgi.stages.goal._poses, with its degenerate fallback). Two variants:
  snapD  depth only: the floor with no people list (R and E have none). Bodies still block through depth.
  snapP  plus C's people: the person discs of the same model's C record of the same arm, scene, episode and trial
         (s21 con for Qwen, s23 con for the others; for GLM and DeepSeek those people answers are cache copies of
         s19/s20). Where that C record has no people (call failed), snapP falls back to snapD (counted).
Every marked-point record is snapped, successes included, so the snap's breakages are counted too.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import math
import statistics as st
import sys
import time

import numpy as np

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402
from sgi import solver  # noqa: E402
from sgi.geometry import project, to_robot_xy, wrap_angle  # noqa: E402
from sgi.stages.goal import _poses  # noqa: E402
from sgi.types import DetectedPerson  # noqa: E402

HERE = 'results/still-scenes/depth-fix/first-check'
MODELS = [('qwen38_low', 's21'), ('glm53', 's23'), ('dsv41', 's23'), ('haiku55', 's23'), ('sonnet55_low', 's23'),
          ('gemini38_low', 's23')]
ARMS = ('real', 'staffed', 'plain')
METHODS = ('cot', 'fsc')
SITS = ('no_one', 'waiting_line', 'conversation_group')
NEAR_M, FAR_M = 0.3, 1.0

BASE: dict = {}                                    # (scenes_dir, scene_id, thresholds) -> Floor without people
CHECKED_FLOOR = [0]                                # with_people checked against solver.floor this many times


def base_floor(rec, sc, th):
    key = (str(rec.ref.scenes_dir), rec.scene_id, th.model_dump_json())
    if key not in BASE:
        BASE[key] = solver.floor(sc.observation, [], th)
    return BASE[key]


def with_people(fl, obs, people, th):
    """The floor of `solver.floor(obs, people, th)`, built from the floor without people: the same last steps."""
    if not people:
        return fl
    x, y = np.meshgrid(fl.offsets, fl.offsets)
    blocked = fl.blocked.copy()
    for person in people:
        ahead, left = to_robot_xy(obs.robot, *person.xy)
        blocked |= np.hypot(x - ahead, y - left) <= th.person_radius_m + th.robot_radius_m + solver._EPS
    passable = fl.own | ((fl.seen | fl.hidden | fl.strip) & ~blocked)
    standable = fl.own | ((fl.seen | fl.strip) & ~blocked)
    half = len(fl.offsets) // 2
    selectable = standable & solver._joined(passable, (half, half))
    return dataclasses.replace(fl, blocked=blocked, passable=passable, standable=standable, selectable=selectable)


def snap(fl, xy):
    """(new xy, moved m): xy unchanged on a selectable cell, else the nearest selectable cell centre."""
    cell = fl.index(*xy)
    if cell is not None and fl.selectable[cell]:
        return (float(xy[0]), float(xy[1])), 0.0
    sel = fl.centres()[fl.selectable]              # row-major order of the grid
    d = np.round(np.hypot(sel[:, 0] - xy[0], sel[:, 1] - xy[1]), 6)
    i = int(np.argmin(d))
    return (float(sel[i, 0]), float(sel[i, 1])), float(d[i])


def line_geometry(sc, ep_id, matched):
    """(anchor, members sorted nearest the target first, last, front unit) of a waiting line, or None."""
    truth = sc.truth
    gid = truth.group_for(ep_id, matched)
    if gid is None:
        return None
    target = truth.targets[matched]
    a = np.array(target.anchor[:2], dtype=float)
    members = sorted((np.array([p.x, p.y]) for p in truth.people if p.group == gid), key=lambda q: float(np.hypot(*(q - a))))
    if not members:
        return None
    last = members[-1]
    before = members[-2] if len(members) > 1 else a
    front = (before - last) / max(float(np.hypot(*(before - last))), 1e-9)
    return a, members, last, front


def line_place(geo, xy):
    """Where a pose stands along a waiting line: behind (level with or behind the last member, the region's own
    test), front (nearer the target anchor than the first member: at the counter or cutting in at the front), or
    beside (beside or inside the line)."""
    a, members, last, front = geo
    p = np.asarray(xy, dtype=float)
    if float((p - last) @ front) <= 1e-9:
        return 'behind'
    if float(np.hypot(*(p - a))) < float(np.hypot(*(members[0] - a))):
        return 'front'
    return 'beside'


def miss_class(s):
    """near / mid / far / inreg of a failed score (distance_to_region_m: 0 inside the region)."""
    d = s.distance_to_region_m
    if d is None:
        return 'nodist'
    if s.in_region:
        return 'inreg'
    return 'near' if d <= NEAR_M else ('far' if d > FAR_M else 'mid')


def nearest_on_polygon(p, poly):
    best, bq = math.inf, None
    vs = [(float(v[0]), float(v[1])) for v in poly]
    for (ax, ay), (bx, by) in zip(vs, vs[1:] + vs[:1]):
        dx, dy = bx - ax, by - ay
        L = dx * dx + dy * dy
        t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L))
        q = (ax + t * dx, ay + t * dy)
        d = math.hypot(p[0] - q[0], p[1] - q[1])
        if d < best:
            best, bq = d, q
    return bq


def main():
    started = time.monotonic()
    rows = []
    for arm in ARMS:
        for model, stage in MODELS:
            con = H.runs(stage=stage, arm=arm, model=model, method='con')
            assert len(con) == 1, (stage, arm, model, con)
            c_people, c_success = {}, {}
            for crec in H.records(con[0]):
                key = (crec.scene_id, crec.episode_id, crec.trial)
                c_people[key] = None if crec.people is None else [DetectedPerson.model_validate(p) for p in crec.people]
                c_success[key] = bool(crec.score.get('success'))
            for method in METHODS:
                refs = H.runs(stage=stage, arm=arm, model=model, method=method)
                assert len(refs) == 1, (stage, arm, model, method, refs)
                ref = refs[0]
                cfg = H.recorded_config(ref)[1]
                assert cfg.run.heading_mode == 'face_target', (ref.folder, cfg.run.heading_mode)
                tally = collections.Counter()
                for rec in H.records(ref, tally=tally):
                    rows.append(one(rec, cfg.thresholds, c_people, c_success))
                print(f"{time.monotonic() - started:7.1f} s  {ref.folder}  {dict(tally)}", flush=True)
    with open(f'{HERE}/snap_records.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f"{len(rows)} records, {CHECKED_FLOOR[0]} people floors checked against solver.floor, "
          f"{time.monotonic() - started:.1f} s")


def one(rec, th, c_people, c_success):
    sc = H.scene(rec.ref, rec.scene_id)
    obs = sc.observation
    robot = obs.robot
    res = rec.result
    key = (rec.scene_id, rec.episode_id, rec.trial)
    row = dict(model=rec.model, arm=rec.arm, method=rec.method, run=rec.run, scene=rec.scene_id, ep=rec.episode_id,
               trial=rec.trial, sit=rec.situation, source=res.get('source'), status=res.get('status'),
               rec_success=bool(rec.score.get('success')), c_success=c_success.get(key))
    marked = res.get('source') == 'direct_marking' and res.get('status') == 'ok' and rec.pose_world is not None
    if res.get('status') == 'ok' and rec.pose_world is not None:
        poses = [tuple(rec.pose_world)]
        if marked:
            faced = res['target']['world'][:2]
            # the method's heading rule reproduces the recorded heading
            again = _poses(robot, rec.pose_world[:2], faced, th)[0]
            assert abs(wrap_angle(again[2] - rec.pose_world[2])) < 1e-9, (rec.run, key)
            fl = base_floor(rec, sc, th)
            xy_d, moved_d = snap(fl, rec.pose_world[:2])
            people = c_people.get(key)
            row['c_people'] = None if people is None else len(people)
            flp = with_people(fl, obs, people or [], th)
            if people and CHECKED_FLOOR[0] < 6:    # the shortcut is solver.floor exactly
                full = solver.floor(obs, people, th)
                assert np.array_equal(full.selectable, flp.selectable) and np.array_equal(full.blocked, flp.blocked)
                CHECKED_FLOOR[0] += 1
            xy_p, moved_p = snap(flp, rec.pose_world[:2])
            poses += [_poses(robot, xy_d, faced, th)[0], _poses(robot, xy_p, faced, th)[0]]
            row.update(moved_d=moved_d, moved_p=moved_p)
        scores = H.score_poses(rec, poses)
        now = scores[0]
        target_ok = now.target_correct
    else:
        scores = [H.rescore(rec)]
        now = scores[0]
        forced = H.score_pose(rec, (robot.x, robot.y, robot.yaw), force_ok=True)
        target_ok = forced.target_correct
        row['matched'] = forced.matched_target_id
    row.update(target_ok=bool(target_ok), success=bool(now.success))
    if now.matched_target_id is not None:
        row['matched'] = now.matched_target_id
    names = ['now', 'snapD', 'snapP'][:len(scores)]
    geo = None
    if rec.situation == 'waiting_line' and row.get('matched'):
        geo = line_geometry(sc, rec.episode_id, row['matched'])
    gt = sc.truths[rec.episode_id].gt
    for name, s, pose in zip(names, scores, poses if res.get('status') == 'ok' and rec.pose_world is not None else [None]):
        row[name] = dict(success=bool(s.success), target=bool(s.target_correct), reach=bool(s.reachable),
                         inreg=bool(s.in_region), heading=bool(s.heading_ok), dist=s.distance_to_region_m,
                         gap=s.min_body_gap_m, cls=miss_class(s))
        if pose is not None and geo is not None:
            a, members, last, front = geo
            row[name]['line'] = line_place(geo, pose[:2])
            row[name]['d_last'] = float(np.hypot(pose[0] - last[0], pose[1] - last[1]))
            row[name]['d_first_anchor'] = float(np.hypot(*(members[0] - a)))
            row[name]['d_anchor'] = float(np.hypot(pose[0] - a[0], pose[1] - a[1]))
            row[name]['n_members'] = len(members)
            # along the axis from the target anchor (t = 0) to the last member (t = 1), and across it (metres,
            # positive on the camera's side of the axis)
            axis = last - a
            L2 = float(axis @ axis)
            q = np.array(pose[:2]) - a
            unit = axis / math.sqrt(L2)
            cam_xy = np.array(obs.camera.T_world_camera[:2, 3])
            side = 1.0 if float(unit[0] * (cam_xy - a)[1] - unit[1] * (cam_xy - a)[0]) >= 0 else -1.0
            row[name]['t'] = float(q @ axis) / L2
            row[name]['t_first'] = float((members[0] - a) @ axis) / L2
            row[name]['perp_cam'] = side * float(unit[0] * q[1] - unit[1] * q[0])
        if pose is not None and sc.truth.people:
            row[name]['d_person'] = min(float(math.hypot(pose[0] - p.x, pose[1] - p.y)) for p in sc.truth.people)
    # where the marked point sits in the picture against the region (mechanism checks)
    if marked and row.get('matched'):
        cam = obs.camera
        cam_xy = cam.T_world_camera[:2, 3]
        regions = [g for g in gt.acceptable_regions or [] if g.target_id in (None, row['matched'])]
        gp = res.get('goal_pixel')
        if regions and gp is not None:
            goal = rec.pose_world[:2]
            reg = min(regions, key=lambda g: math.dist(goal, nearest_on_polygon(goal, g.polygon)))
            near_q = nearest_on_polygon(goal, reg.polygon)
            rep = reg.representative_point
            out = {}
            for tag, q in (('rep', rep), ('nearest', near_q)):
                if q is None:
                    continue
                shown = project(cam, (q[0], q[1], 0.0))
                if shown is None:
                    continue
                out[f'dv_{tag}'] = float(gp[1] - shown[1])
                out[f'drange_{tag}'] = float(math.dist(goal, cam_xy) - math.dist(q, cam_xy))
            row['picture'] = out
    return row


if __name__ == '__main__':
    main()
