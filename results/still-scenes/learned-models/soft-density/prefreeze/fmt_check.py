"""I04 pre-freeze format check (run before FROZEN.md is hashed). No density, no fold fit, no metric.

Reads runs and scenes only through research_harness. No model call, no network.
Checks, per run of the I14 v0 cells:
  - records allowed / goal-step records (H.Record.solver_step); for the watching runs, goal steps of situation audience;
  - key names present in each goal-step record (target.world, people, constraints[].constraint, 4 candidates with
    pose_world and keys, pose_world, flags with the three solver flags, solver.keys);
  - the layouts: realistic eval index (00-13) from the scene id, watching (dev seed, scene id);
  - the recorded thresholds the protocol uses, equal across runs;
  - for the FIRST goal-step record of each run: rebuild the solver's inputs as H.resolve does, call
    sgi.solver._over_floor, and check that picks[0] gives the recorded place (<= 1e-6 m), picks[1:5] the recorded
    runners-up, the solver's heading rule (faced_xy) the recorded headings, and list settled.keys names.
It does not compute any region membership, success, density or score.

    cd WT && PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python <this>
"""
from __future__ import annotations

import collections
import math
import re
import sys
import time

import numpy as np

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402
from sgi import solver  # noqa: E402
from sgi.geometry import to_world_xy, wrap_angle  # noqa: E402
from sgi.stages.goal import standard_goal  # noqa: E402
from sgi.types import Constraint, DetectedPerson, GroundedTarget  # noqa: E402

NEW = [('s21', 'qwen38_low'), ('s23', 'glm53'), ('s23', 'dsv41'), ('s23', 'haiku55'), ('s23', 'sonnet55_low'),
       ('s23', 'gemini38_low')]
OLD = [('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41')]
REAL_ARMS = ('real', 'staffed', 'plain')
DEV_ARMS = ('dev_seed0', 'dev_seed1', 'dev_seed2')
TH_FIELDS = ('heading_degenerate_m', 'robot_radius_m', 'grid_resolution_m', 'candidate_count', 'floor_range_m',
             'standoff_m', 'person_radius_m')
EVAL_IDX = re.compile(r'(?:^|_)eval_(\d\d)_')


def plan():
    out = []
    for stage, model in OLD:
        for arm in REAL_ARMS:
            out.append(('old', stage, model, arm))
    for stage, model in OLD:
        for arm in DEV_ARMS:
            out.append(('watch', stage, model, arm))
    for stage, model in NEW:
        for arm in REAL_ARMS:
            out.append(('new', stage, model, arm))
    return out


def key_names(rec):
    r = rec.result
    names = []
    t = r.get('target') or {}
    names.append('target.world' if t.get('world') is not None else 'NO target.world')
    names.append(f"people[{'list' if isinstance(r.get('people'), list) else 'NONE'}]")
    cons = r.get('constraints')
    names.append('constraints[].constraint' if isinstance(cons, list) and all('constraint' in c for c in cons)
                 else 'BAD constraints')
    cands = r.get('candidates') or []
    names.append(f'candidates={len(cands)}' + ('' if all('pose_world' in c and 'keys' in c for c in cands)
                                                 else ' BAD'))
    names.append('pose_world' if r.get('pose_world') is not None else 'NO pose_world')
    fl = r.get('flags') or {}
    names.append('flags:' + ','.join(f for f in ('exclusion_broken', 'far_from_target', 'constraint_unusable')
                                      if f in fl))
    sk = (r.get('solver') or {}).get('keys') or {}
    names.append('solver.keys:' + ','.join(k for k, v in sk.items() if v is not None))
    return tuple(names)


def layout(label, rec):
    if label == 'watch':
        return f'{rec.arm}/{rec.scene_id}'
    m = EVAL_IDX.search(rec.scene_id)
    return f'eval_{m.group(1)}' if m else f'?{rec.scene_id}'


def reproduce(rec, th):
    sc = H.scene(rec.ref, rec.scene_id)
    obs = sc.observation
    robot = obs.robot
    target = GroundedTarget.model_validate(rec.target)
    people = [DetectedPerson.model_validate(p) for p in rec.people]
    constraints = [Constraint.model_validate(c['constraint']) for c in rec.constraints]
    std = standard_goal(obs, target, th)
    cells, settled = solver._over_floor(obs, target, people, constraints, std, th)
    recorded = [tuple(rec.pose_world)] + [tuple(c['pose_world']) for c in rec.candidates]
    out = dict(n_cells=len(cells), n_picks=len(settled.picks), keys=list(settled.keys))
    moved, dth = [], []
    for i, pk in enumerate(settled.picks[:len(recorded)]):
        wx, wy = to_world_xy(robot, *cells[pk])
        moved.append(math.hypot(wx - recorded[i][0], wy - recorded[i][1]))
        toward = settled.faced_xy - cells[pk]
        deg = float(np.hypot(*toward)) < th.heading_degenerate_m
        theta = robot.yaw if deg else robot.yaw + math.atan2(toward[1], toward[0])
        dth.append(abs(wrap_angle(wrap_angle(theta) - recorded[i][2])))
    out['moved_m'] = moved
    out['dtheta'] = dth
    return out


def main():
    t0 = time.monotonic()
    cells = collections.defaultdict(lambda: dict(n=0, std=0, names=collections.Counter(), layouts=collections.Counter(),
                                                 situations=collections.Counter(), trials=collections.Counter()))
    th_seen = {}
    repro = []
    for label, stage, model, arm in plan():
        refs = H.runs(stage=stage, arm=arm, model=model, method='con')
        assert len(refs) == 1, (stage, arm, model, refs)
        ref = refs[0]
        th = H.recorded_config(ref)[1].thresholds
        th_seen[ref.folder] = tuple(getattr(th, f) for f in TH_FIELDS)
        tally = collections.Counter()
        c = cells[(label, model)]
        first = None
        for rec in H.records(ref, tally=tally):
            if label == 'watch' and rec.situation != 'audience':
                continue
            if not rec.solver_step:
                c['std'] += 1
                continue
            c['n'] += 1
            c['names'][key_names(rec)] += 1
            c['layouts'][layout(label, rec)] += 1
            c['situations'][rec.situation] += 1
            c['trials'][rec.trial] += 1
            if first is None:
                first = rec
        if first is not None:
            r = reproduce(first, th)
            repro.append((ref.folder, first.scene_id, first.episode_id, first.trial, r))
        print(f'{time.monotonic() - t0:7.1f} s  {label:5s} {ref.folder}  {dict(tally)}', flush=True)

    print('\n== goal-step records per cell (model x set) ==')
    for (label, model), c in cells.items():
        print(f'{label:5s} {model:13s} goal {c["n"]:4d}  standard {c["std"]:4d}  layouts {len(c["layouts"]):2d}  '
              f'situations {dict(c["situations"])}  trials {dict(sorted(c["trials"].items(), key=str))}')
    print('\n== key names per goal-step record (distinct sets per cell) ==')
    for (label, model), c in cells.items():
        for names, k in c['names'].most_common():
            print(f'{label:5s} {model:13s} {k:4d}  {" | ".join(names)}')
    print('\n== layouts ==')
    for lab in ('old', 'watch', 'new'):
        lay = collections.Counter()
        for (label, model), c in cells.items():
            if label == lab:
                lay.update(c['layouts'])
        print(f'{lab:5s} {len(lay)} layouts: ' + ', '.join(f'{k} ({v})' for k, v in sorted(lay.items())))
    print('\n== recorded thresholds (fields: ' + ', '.join(TH_FIELDS) + ') ==')
    for vals, k in collections.Counter(th_seen.values()).items():
        print(f'{k} runs: {vals}')
    print('\n== solver reproduction, first goal-step record of each run (picks vs recorded places) ==')
    worst_m = worst_th = 0.0
    for folder, scene_id, ep, trial, r in repro:
        worst_m = max(worst_m, max(r['moved_m']))
        worst_th = max(worst_th, max(r['dtheta']))
        print(f'{folder:52s} {scene_id:42s} {ep} t{trial}  F {r["n_cells"]:5d}  picks {r["n_picks"]}  '
              f'max moved {max(r["moved_m"]):.2e} m (place1 {r["moved_m"][0]:.2e})  max dtheta {max(r["dtheta"]):.2e}  '
              f'keys {",".join(r["keys"])}')
    print(f'records reproduced: {len(repro)}; worst place move {worst_m:.3e} m; worst heading diff {worst_th:.3e} rad')
    print(f'done in {time.monotonic() - t0:.1f} s')


if __name__ == '__main__':
    main()
