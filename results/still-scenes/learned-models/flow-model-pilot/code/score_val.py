"""C5 flow pilot 6(i): validation seeds 190-199, scored in the project venv (FROZEN.md section 6(i)).

For every validation unit and every model's samples (samples/val_<model>.npz, from the sidecar):
  - success of each of the 64 raw samples and of the top pose (density over all 64 raw positions) by I14's
    vectorised replica (phase1/i14_v0/i14.py replica), with the oracle target (a target at the 3-D anchor where
    the oracle lift fails, counted) and the matched target id the package gives it;
  - a check against sgi.evaluate.score on a GoalResult with that target for the top pose and 8 random samples per
    unit (rng default_rng([7, crc32(key)]), choice), on success, in_region, heading_ok and reachable; any
    disagreement stops the scoring;
  - region coverage: the share of the unit's extended region cells (data/seeds, common.region_cells_extended) with a
    raw sample within 0.2 m (+1e-9);
  - baselines (top-1, once per unit): sgi.solver.solve with the oracle people, the oracle target and standard_goal;
    B0 no constraints, B1 reference constraints (repairs.reference_constraints, written against the oracle people:
    circular, a reference only), scored by sgi.evaluate.score.
Writes rows_val_<model>.jsonl and rows_val_baselines.jsonl. No model call, no network, no torch.
"""
from __future__ import annotations

import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, 'results/still-scenes/replay-harness/repairs')
import common as C  # noqa: E402
import research_harness as H  # noqa: E402
import i14  # noqa: E402
import repairs  # noqa: E402
from build_seeds import anchor_target  # noqa: E402
from sgi.config import load_config  # noqa: E402
from sgi.evaluate import match_target, score  # noqa: E402
from sgi.freespace import connected, robot_grid  # noqa: E402
from sgi.geometry import make_poses, project  # noqa: E402
from sgi.gt.derive import is_visible, oracle_target, prepare  # noqa: E402
from sgi.run import OraclePeople  # noqa: E402
from sgi.scene import load_scene  # noqa: E402
from sgi.solver import solve  # noqa: E402
from sgi.stages.goal import standard_goal  # noqa: E402
from sgi.types import Constraint, DetectedPerson, GoalResult  # noqa: E402

OUT = Path(C.OUT)
MODELS = ('main', 'loso_waiting_line', 'loso_conversation_group', 'loso_audience')
COVER_M = 0.2


def oracle_people_with_ids(truth, obs, th):
    """OraclePeople's list, numbered 1.. in truth order, with the true ids (same filter as sgi.run.OraclePeople)."""
    found = OraclePeople(truth, th)(obs).found
    ids = []
    for person in truth.people:
        if not is_visible(person, obs, th):
            continue
        if project(obs.camera, (person.x, person.y, th.person_mark_height_m)) is not None:
            ids.append(person.id)
    assert len(ids) == len(found)
    people = []
    for i, (pid, f) in enumerate(zip(ids, found)):
        p = truth.person(pid)
        assert (p.x, p.y) == tuple(f.xy)
        people.append(DetectedPerson(number=i + 1, xy=f.xy, yaw=f.yaw, pixel=f.pixel, label=None))
    return people, {pid: i + 1 for i, pid in enumerate(ids)}


def pkg_score(obs, truth, gt, th, target, pose):
    pw, pr = make_poses(obs.robot, pose[:2], pose[2])
    res = GoalResult(status='ok', source='constraints', pose_world=pw, pose_robot=pr, target=target)
    return score(res, obs, truth, gt, th)


def main():
    cfg = load_config()
    th = cfg.thresholds
    samples = {m: np.load(OUT / 'samples' / f'val_{m}.npz') for m in MODELS}
    by_key = {m: {str(k): i for i, k in enumerate(samples[m]['keys'])} for m in MODELS}
    rows = {m: [] for m in MODELS}
    base_rows = []
    checks = collections.Counter()
    t0 = time.monotonic()
    for k in range(190, 200):
        info = json.loads((OUT / 'data' / 'seeds' / f'seed{k}.json').read_text())
        z = np.load(OUT / 'data' / 'seeds' / f'seed{k}.npz')
        ext_xy, ext_off = z['ext_xy'], z['ext_off']
        by_scene = collections.defaultdict(list)
        for ui, u in enumerate(info['units']):
            by_scene[(u['split'], u['scene'])].append((ui, u))
        for (split, scene_id), units in by_scene.items():
            folder = Path(C.SEEDS_DIR) / f'seed{k}' / split / scene_id
            H.guard_scene(folder)
            inputs, truth = load_scene(folder)
            truths = prepare(inputs, truth, cfg)
            C.assert_dev_scene(scene_id, [et.gt.situation for et in truths.values()])
            obs = inputs.observation
            robot = obs.robot
            grid = robot_grid(truth, robot, th)
            joined = connected(grid, (robot.x, robot.y))
            people, number_of = oracle_people_with_ids(truth, obs, th)
            for ui, u in units:
                key = u['key']
                gt = truths[u['episode']].gt
                tid = u['target_id']
                target_t = truth.targets[tid]
                ot = oracle_target(obs, target_t, th)
                lift_failed = ot is None
                tg = anchor_target(obs, target_t.anchor) if lift_failed else ot
                matched = match_target(tg.world, truth, gt, th)
                ext = ext_xy[ext_off[ui]:ext_off[ui + 1]]
                # baselines
                std = standard_goal(obs, tg, th)
                group = truth.group_for(u['episode'], tid)
                member_ids = [] if group is None else [p.id for p in truth.group(group) if p.id in number_of]
                members = [number_of[m] for m in member_ids]
                needed = number_of.get(target_t.person_id) if target_t.person_id else None
                ref = [Constraint.model_validate(c) for c in repairs.reference_constraints(gt.situation, members, needed)]
                brow = dict(key=key, seed=k, split=split, scene=scene_id, episode=u['episode'], sit=gt.situation,
                            lift_failed=lift_failed, matched=matched, n_ref=len(ref))
                for name, cons in (('B0', []), ('B1', ref)):
                    try:
                        sol = solve(obs, tg, people, cons, std, th)
                        brow[name] = bool(pkg_score(obs, truth, gt, th, tg, sol.pose_world).success)
                    except ValueError as e:
                        brow[name] = False
                        brow[name + '_error'] = str(e)
                base_rows.append(brow)
                for m in MODELS:
                    if key not in by_key[m]:
                        continue
                    i = by_key[m][key]
                    s = samples[m]
                    pw, hd, pr, top = s['raw_world'][i], s['head'][i], s['raw_raster'][i], int(s['top'][i])
                    rep = i14.replica(robot, gt, matched, grid, joined, pw, hd)
                    succ = rep['success']
                    rng = np.random.default_rng([7, C.crc(key)])
                    check_idx = [top] + sorted(rng.choice(len(pw), size=8, replace=False).tolist())
                    for j in check_idx:
                        sc = pkg_score(obs, truth, gt, th, tg, (float(pw[j, 0]), float(pw[j, 1]), float(hd[j])))
                        for f, a, b in (('success', bool(rep['success'][j]), sc.success),
                                        ('inreg', bool(rep['inreg'][j]), sc.in_region),
                                        ('heading', bool(rep['hok'][j]), sc.heading_ok),
                                        ('reach', bool(rep['reach'][j]), sc.reachable)):
                            checks[f] += 1
                            if a != b:
                                raise SystemExit(f"STOP: replica disagrees with sgi.evaluate.score on {f}: {key} {m} "
                                                 f"sample {j}: replica {a}, package {b}")
                    if len(ext):
                        d = np.hypot(ext[:, None, 0] - pr[None, :, 0], ext[:, None, 1] - pr[None, :, 1]).min(axis=1)
                        cover = float(np.mean(d <= COVER_M + C.EPS))
                    else:
                        cover = None
                    inr = np.abs(pr) < C.HALF * C.RES
                    rows[m].append(dict(key=key, seed=k, split=split, scene=scene_id, episode=u['episode'],
                                        sit=gt.situation, row=u['row'], labelled=u['selfcheck'] is not None,
                                        lift_failed=lift_failed, matched=matched,
                                        share_success=float(succ.mean()), share_inreg=float(rep['inreg'].mean()),
                                        share_reach=float(rep['reach'].mean()), top_success=bool(succ[top]),
                                        coverage=cover, n_region_ext=int(len(ext)),
                                        share_samples_in_raster=float(np.mean(inr[:, 0] & inr[:, 1]))))
        print(f"{time.monotonic() - t0:7.1f} s  seed{k} scored; checks {dict(checks)}", flush=True)
    for m in MODELS:
        with open(OUT / f'rows_val_{m}.jsonl', 'w') as f:
            for r in rows[m]:
                f.write(json.dumps(r) + '\n')
    with open(OUT / 'rows_val_baselines.jsonl', 'w') as f:
        for r in base_rows:
            f.write(json.dumps(r) + '\n')
    (OUT / 'logs' / 'score_val_checks.json').write_text(json.dumps(dict(checks=dict(checks), disagreements=0)))
    print(f"done: checks {dict(checks)}, 0 disagreements, {time.monotonic() - t0:.1f} s", flush=True)


if __name__ == '__main__':
    main()
