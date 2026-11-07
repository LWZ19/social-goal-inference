"""C5 gates, pre-freeze format check (allowed before FROZEN.md is hashed): load a few fresh-seed scenes of seed 100,
list their episodes, targets, people, region polygons, and the 0.10 m floor and region cell counts. Counts no
configuration, fits no density, computes no score. No model, no network. Seeds 100-109 only (fitdata.py's way)."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402
from sgi.config import load_config  # noqa: E402
from sgi.freespace import connected, robot_grid  # noqa: E402
from sgi.gt.derive import prepare  # noqa: E402
from sgi.polygons import point_in_polygon  # noqa: E402
from sgi.scene import load_scene  # noqa: E402

FRESH = Path('results/2026-10-17-beyond-vlm/phase1b/fresh_seeds')
PICK = ['eval/eval_00_no_one_place', 'eval/eval_04_no_one_object', 'eval/eval_05_no_one_person',
        'eval/eval_08_waiting_line_4', 'eval/eval_12_conversation_group_4', 'eval/eval_15_audience_5']


def coarse_floor(truth, robot, th):
    """0.10 m cells whose centre is a 0.05 m scorer cell centre (origin shifted -0.025 m), free and joined."""
    grid = robot_grid(truth, robot, th)
    joined = connected(grid, (robot.x, robot.y))
    sub = joined[0::2, 0::2]                      # fine cells (2j, 2i): centres x0 + 0.025 + 0.10 i
    ny, nx = sub.shape
    xs = grid.x0 + 0.025 + 0.10 * np.arange(nx)
    ys = grid.y0 + 0.025 + 0.10 * np.arange(ny)
    iy, ix = np.nonzero(sub)
    return np.stack([xs[ix], ys[iy]], axis=1), grid, joined


def main():
    cfg = load_config()
    th = cfg.thresholds
    print('thresholds: grid', th.grid_resolution_m, 'robot r', th.robot_radius_m, 'person r', th.person_radius_m)
    for rel in PICK:
        folder = FRESH / 'seed100' / rel
        H.guard_scene(folder)
        t0 = time.monotonic()
        inputs, truth = load_scene(folder)
        ets = prepare(inputs, truth, cfg)
        t1 = time.monotonic()
        robot = inputs.observation.robot
        cells, grid, joined = coarse_floor(truth, robot, th)
        t2 = time.monotonic()
        print(f'\n== {inputs.scene_id}  load+prepare {t1 - t0:.1f}s floor {t2 - t1:.2f}s')
        print(f'  robot ({robot.x:.2f},{robot.y:.2f},{math.degrees(robot.yaw):.0f}deg); fine grid {grid.free.shape};'
              f' joined fine cells {int(joined.sum())}; 0.10 m floor cells {len(cells)}')
        b = truth.bounds
        print(f'  bounds first edge yaw {math.degrees(math.atan2(b[1][1] - b[0][1], b[1][0] - b[0][0])):.2f}deg;'
              f' obstacles {len(truth.obstacles)}; occupancy {truth.occupancy is not None}')
        for t in truth.targets.values():
            fy = None if t.front_yaw is None else round(math.degrees(t.front_yaw), 2)
            print(f'  target {t.id} kind {t.kind} anchor ({t.anchor[0]:.2f},{t.anchor[1]:.2f}) front {fy}'
                  f' footprint {t.footprint is not None} person {t.person_id}')
        print(f'  people {len(truth.people)}: ' + ', '.join(f'{p.id}(g={p.group},r={p.role})' for p in truth.people))
        sets = {}
        for eid, et in ets.items():
            gt = et.gt
            acc = gt.acceptable_regions or []
            inside = np.zeros(len(cells), dtype=bool)
            for reg in acc:
                inside |= np.array([point_in_polygon((float(x), float(y)), reg.polygon) for x, y in cells])
            sets[eid] = frozenset(np.nonzero(inside)[0].tolist())
            print(f'  {eid}: situation {gt.situation} targets {gt.target_ids} group {gt.group_at_target};'
                  f' acceptable {len(acc)} (vertices {[len(r.polygon) for r in acc]}, target_ids'
                  f' {[r.target_id for r in acc]}, tol {[r.tolerance_deg for r in acc]});'
                  f' forbidden {[f.kind for f in gt.forbidden_regions or []]}; region cells {int(inside.sum())};'
                  f' problems {list(et.problems)}')
        print(f'  episodes share one region-cell set: {len(set(sets.values())) == 1}')


if __name__ == '__main__':
    main()
