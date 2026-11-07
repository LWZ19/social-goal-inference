"""C5 flow pilot, pre-freeze format check (written before FROZEN.md is hashed). No training, no metric, no
success or score is computed; nothing is tuned. No model, no network, no torch.

Part A: 7 scenes of fresh seed 100 (read-only fit seed; one per row kind) loaded as phase1b/i03/stageA/fitdata.py
does (H.guard_scene, then sgi.scene.load_scene; regions from sgi.gt.derive.prepare under configs/default.yaml).
It lists the targets, episodes, region polygons and their heading rule, the visible people, the robot's distance
to the target, anchor ground point vs the oracle-grounded target point, _seen_face at both, and how much of the
region (0.05 m sample points) falls inside a 64 x 64 x 0.10 m raster centred on each point, and how much of it lies
on the solver floor's selectable cells.

Part B: the first goal-step record of 4 primary runs, through research_harness: fields present, yaw None counts,
_over_floor reproduces the recorded place (key excluded_m at the place), the seen face, the robot's distance.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402
from sgi.config import load_config  # noqa: E402
from sgi.geometry import to_world_xy  # noqa: E402
from sgi.gt.derive import is_visible, oracle_target, prepare  # noqa: E402
from sgi.polygons import point_in_polygon  # noqa: E402
from sgi.scene import load_scene  # noqa: E402
from sgi import solver  # noqa: E402
from sgi.stages.goal import standard_goal  # noqa: E402
from sgi.types import DetectedPerson, GroundedTarget  # noqa: E402

FRESH = Path('results/still-scenes/perception/fresh_seeds')
SCENES = [('eval', 'eval_00_no_one_place'), ('eval', 'eval_04_no_one_object'), ('eval', 'eval_05_no_one_person'),
          ('eval', 'eval_08_waiting_line_4'), ('eval', 'eval_12_conversation_group_4'), ('eval', 'eval_15_audience_5'),
          ('eval', 'eval_21_no_one_screen')]
HALF = 3.2


def frame(origin, robot_xy):
    d = np.array(origin[:2]) - np.array(robot_xy[:2])
    yaw = math.atan2(d[1], d[0])
    return yaw, float(np.hypot(*d))


def in_raster(pts, origin, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    d = pts - np.array(origin[:2])
    x = c * d[:, 0] + s * d[:, 1]
    y = -s * d[:, 0] + c * d[:, 1]
    return (np.abs(x) < HALF) & (np.abs(y) < HALF)


def region_points(regions):
    out = []
    for r in regions:
        v = np.array(r.polygon)
        lo, hi = v.min(axis=0), v.max(axis=0)
        xs = np.arange(lo[0], hi[0] + 1e-9, 0.05)
        ys = np.arange(lo[1], hi[1] + 1e-9, 0.05)
        for x in xs:
            for y in ys:
                if point_in_polygon((x, y), r.polygon):
                    out.append((x, y))
    return np.unique(np.round(np.array(out), 6), axis=0) if out else np.zeros((0, 2))


def part_a():
    cfg = load_config()
    th = cfg.thresholds
    print(f'thresholds: floor_range_m {th.floor_range_m}, grid {th.grid_resolution_m}, robot_radius {th.robot_radius_m}, '
          f'person_radius {th.person_radius_m}, standoff {th.standoff_m}')
    for split, sid in SCENES:
        folder = FRESH / 'seed100' / split
        H.guard_scene(folder / sid)
        inputs, truth = load_scene(folder / sid)
        obs = inputs.observation
        truths = prepare(inputs, truth, cfg)
        vis = [p for p in truth.people if is_visible(p, obs, th)]
        print(f'\n== seed100/{split}/{sid}: {len(truth.people)} people, {len(vis)} visible; robot ({obs.robot.x:.2f}, '
              f'{obs.robot.y:.2f}, yaw {obs.robot.yaw:.2f})')
        for tid, t in truth.targets.items():
            ot = oracle_target(obs, t, th)
            a = np.array(t.anchor[:2])
            print(f'  target {tid}: kind {t.kind}, anchor {tuple(round(v, 2) for v in t.anchor)}, front_yaw {t.front_yaw}, '
                  f'footprint {"yes" if t.footprint else "no"}, person_id {t.person_id}')
            if ot is None:
                print('    oracle_target: None (lift fails)')
            else:
                w = np.array(ot.world)
                print(f'    oracle_target world {tuple(round(float(v), 2) for v in w)}; ground distance to anchor '
                      f'{float(np.hypot(*(w[:2] - a))):.3f} m')
            yaw_a, dist_a = frame(a, (obs.robot.x, obs.robot.y))
            print(f'    robot start to anchor ground point: {dist_a:.2f} m (robot disc in raster: {dist_a < HALF - 0.0})')
            for label, gt_world in (('anchor ground', (a[0], a[1], 0.0)), ('anchor 3d', tuple(t.anchor)),
                                    ('oracle', None if ot is None else ot.world)):
                if gt_world is None:
                    continue
                g = GroundedTarget(point=ot.point if ot else None, pixel=ot.pixel if ot else (0.0, 0.0),
                                   world=tuple(float(v) for v in gt_world)) if ot else None
                if g is None:
                    continue
                n = solver._seen_face(obs, g, th)
                if n is None:
                    print(f'    _seen_face at {label}: None')
                else:
                    c, s = math.cos(obs.robot.yaw), math.sin(obs.robot.yaw)
                    nw = (c * n[0] - s * n[1], s * n[0] + c * n[1])
                    print(f'    _seen_face at {label}: world yaw {math.degrees(math.atan2(nw[1], nw[0])):.1f} deg '
                          f'(front_yaw {None if t.front_yaw is None else round(math.degrees(t.front_yaw), 1)})')
        people = [DetectedPerson(number=i + 1, xy=(p.x, p.y), yaw=p.yaw, pixel=(0.0, 0.0)) for i, p in enumerate(vis)]
        fl = solver.floor(obs, people, th)
        for eid, et in truths.items():
            gt = et.gt
            regs = gt.acceptable_regions or []
            kinds = sorted({('yaw' if r.heading.yaw is not None else 'face_point') for r in regs})
            tol = sorted({r.tolerance_deg for r in regs})
            print(f'  episode {eid}: situation {gt.situation}, targets {gt.target_ids}, {len(regs)} acceptable '
                  f'polygons, heading rule {kinds}, tol {tol}, forbidden {[f.kind for f in gt.forbidden_regions or []]}')
            if not regs:
                continue
            pts = region_points(regs)
            t = truth.targets[gt.target_ids[0]]
            a = np.array(t.anchor[:2])
            yaw_a, _ = frame(a, (obs.robot.x, obs.robot.y))
            ins = in_raster(pts, a, yaw_a)
            sel = np.array([(lambda ix: ix is not None and bool(fl.selectable[ix]))(fl.index(x, y)) for x, y in pts])
            print(f'    region sample points (0.05 m): {len(pts)}; inside raster around anchor: {ins.mean():.3f}; '
                  f'on selectable floor (true visible people): {sel.mean():.3f}; both: {(ins & sel).mean():.3f}')
            far = float(np.hypot(*(pts - a).T).max())
            print(f'    farthest region point from anchor ground point: {far:.2f} m')


def part_b():
    picks = [('s19', 'real', 'qwen38_low'), ('s20', 'staffed', 'dsv41'), ('s19', 'dev_seed0', 'glm53'),
             ('s20', 'dev_seed2', 'dsv41')]
    for stage, arm, model in picks:
        (ref,) = H.runs(stage=stage, arm=arm, model=model, method='con')
        th = H.recorded_config(ref)[1].thresholds
        n_goal = 0
        first = None
        yaw_none = 0
        n_people = 0
        for rec in H.records(ref):
            if not rec.solver_step:
                continue
            n_goal += 1
            yaw_none += sum(p.get('yaw') is None for p in rec.people)
            n_people += len(rec.people)
            if first is None:
                first = rec
        rec = first
        sc = H.scene(rec.ref, rec.scene_id)
        obs = sc.observation
        tgt = GroundedTarget.model_validate(rec.target)
        people = [DetectedPerson.model_validate(p) for p in rec.people]
        cons = [u['constraint'] for u in rec.constraints]
        from sgi.types import Constraint
        cons = [Constraint.model_validate(c) for c in cons]
        cells, settled = solver._over_floor(obs, tgt, people, cons, standard_goal(obs, tgt, th), th)
        p0 = cells[settled.picks[0]]
        pw = to_world_xy(obs.robot, *p0)
        moved = math.dist(pw, rec.pose_world[:2])
        ex = settled.keys['excluded_m']
        front = solver._seen_face(obs, tgt, th)
        _, dist = frame(tgt.world, (obs.robot.x, obs.robot.y))
        print(f'\n== {ref.folder}: {n_goal} goal-step records; people yaw None {yaw_none} of {n_people}')
        print(f'  first: {rec.scene_id} {rec.episode_id} t{rec.trial} ({rec.situation}); keys {list(settled.keys)}; '
              f'F cells {len(cells)}; place reproduced at {moved:.2e} m; excluded_m at place {ex[settled.picks[0]]:.3f}; '
              f'cells at <= place + 1 mm: {int((ex <= ex[settled.picks[0]] + 0.001 + 1e-9).sum())}; '
              f'seen face {"none" if front is None else "yes"}; robot to target point {dist:.2f} m; '
              f'thresholds equal default: {th == load_config().thresholds}')


if __name__ == '__main__':
    part_a()
    part_b()
