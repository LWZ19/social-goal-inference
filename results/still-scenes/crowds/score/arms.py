"""I06 scoring step 5 (FROZEN.md section 11): arms 1-3, the equivalence check (arm 3 with no constraint = arm 2, to
1e-9 m) and the detectors' raw values, on every kept scene (30 bases + accepted variants). No model call.

    arms.py SEED [LIMIT]  -> score/work/arms_seed<SEED>.jsonl (one row per scene; both episodes inside)
"""
from __future__ import annotations

import json
import math
import sys
import time

import lib as L
from sgi.evaluate import score
from sgi.gt.derive import prepare
from sgi.pipeline import build_pipeline
from sgi.run import run_episode
from sgi.types import GoalResult


def main(seed: int, limit: int | None = None):
    import hashlib
    assert hashlib.sha256((L.I06 / 'FROZEN.md').read_bytes()).hexdigest() == L.FROZEN_SHA
    cfg, c1, c2 = L.config_variants()
    th = cfg.thresholds
    p1, p2 = build_pipeline(c1), build_pipeline(c2)
    entries = [e for e in L.kept_scenes() if e['seed'] == seed]
    if limit:
        entries = entries[:limit]
    out = L.SCORE / 'work' / f'arms_seed{seed}.jsonl'
    out.parent.mkdir(exist_ok=True)
    started = time.perf_counter()
    with open(out, 'w') as fh:
        for e in entries:
            t0 = time.perf_counter()
            inputs, truth = L.load(e)
            obs = inputs.observation
            ets = prepare(inputs, truth, cfg)
            row = {k: e[k] for k in ('scene_id', 'kind', 'seed', 'row', 'family', 'level', 'base_scene_id')}
            row['situation'] = sorted({ep.situation for ep in truth.episodes.values()})
            row['n_people'] = len(truth.people)
            gid = truth.group_for('e0', L.TARGET_ID)
            cells = L.l1_cells(truth, obs, ets['e0'].gt.acceptable_regions, th)
            row['l1_cells_e0'] = int(len(cells))
            eps = {}
            calls = 0
            for episode in inputs.episodes:
                eid = episode.episode_id
                et = ets[eid]
                g = truth.group_for(eid, L.TARGET_ID)
                ep = {}
                # arm 1 and arm 2 through the package's run_episode
                for name, pipe, conf in (('arm1', p1, c1), ('arm2', p2, c2)):
                    rec = run_episode(pipe, inputs, truth, et, episode, conf, 0)
                    calls += len(rec['result'].get('calls') or [])
                    res = GoalResult.model_validate(rec['result'])
                    xy = L.goal_xy(res, obs)
                    s1 = bool(rec['score']['success'])
                    ep[name] = {'l1': s1, 'l2': bool(s1 and L.l2_ok(xy, truth, g, th)),
                                'xy': None if xy is None else [float(xy[0]), float(xy[1])],
                                'theta': None if res.pose_world is None else float(res.pose_world[2]),
                                'in_region': rec['score'].get('in_region'), 'status': res.status,
                                'source': res.source, 'people_matter': rec['measures']['reference_people_matter'],
                                'min_body_gap_m': rec['score'].get('min_body_gap_m'),
                                'flags': {k: v for k, v in res.flags.model_dump().items() if v}}
                    if name == 'arm2':
                        c = {'result': res, 'note': res.solver, 'uses': res.constraints, 'people': res.people,
                             'source_of': {}, 'target': res.target}
                        if res.people is not None:
                            by_xy = {(p.x, p.y): p.id for p in truth.people}
                            c['source_of'] = {d.number: by_xy.get(tuple(d.xy)) for d in res.people}
                        ep['arm2']['det'] = L.detectors(inputs, truth, et, th, c, cells)
                        ep['arm2']['note'] = None if res.solver is None else res.solver.model_dump()
                # arm 3 and its no-constraint twin
                for name, mode in (('arm3', 'reference'), ('arm3e', 'empty')):
                    c = L.solve_c(inputs, truth, et, th, constraints=mode)
                    res = c['result']
                    xy = L.goal_xy(res, obs)
                    sc = score(res, obs, truth, et.gt, th)
                    ep[name] = {'l1': bool(sc.success), 'l2': bool(sc.success and L.l2_ok(xy, truth, g, th)),
                                'xy': None if xy is None else [float(xy[0]), float(xy[1])],
                                'theta': None if res.pose_world is None else float(res.pose_world[2]),
                                'in_region': sc.in_region, 'heading_ok': sc.heading_ok, 'reachable': sc.reachable,
                                'target_correct': sc.target_correct, 'min_body_gap_m': sc.min_body_gap_m,
                                'people_matter': c['people_matter'],
                                'flags': {k: v for k, v in res.flags.model_dump().items() if v}}
                    if name == 'arm3':
                        ep['arm3']['l2_fail'] = [] if not sc.success else L.l2_fail_reason(xy, truth, g, th)
                        ep['arm3']['det'] = L.detectors(inputs, truth, et, th, c, cells)
                        ep['arm3']['note'] = None if c['note'] is None else c['note'].model_dump()
                        ep['arm3']['constraints'] = c['constraints']
                        ep['arm3']['uses'] = None if c['uses'] is None else [
                            {'kind': u.constraint.kind, 'status': u.status, 'reason': u.reason, 'broken_m': u.broken_m}
                            for u in c['uses']]
                        ep['arm3']['people'] = None if c['people'] is None else [
                            {'number': d.number, 'xy': list(d.xy), 'source': c['source_of'][d.number]}
                            for d in c['people']]
                        ep['arm3']['members'] = c.get('members')
                        ep['arm3']['needed'] = c.get('needed')
                # equivalence: arm3 with no constraint gives arm2's place
                a, b = ep['arm3e'], ep['arm2']
                if a['xy'] is None or b['xy'] is None:
                    ep['equiv_m'] = None if (a['xy'] is None and b['xy'] is None) else float('inf')
                else:
                    ep['equiv_m'] = math.dist(a['xy'], b['xy'])
                ep['equiv_theta'] = None if a['theta'] is None or b['theta'] is None else abs(a['theta'] - b['theta'])
                ep['equiv_l1'] = a['l1'] == b['l1']
                eps[eid] = ep
            row['episodes'] = eps
            row['model_calls'] = calls
            # one target, one region set: every arm gives both episodes the same result
            same = {}
            for arm in ('arm1', 'arm2', 'arm3', 'arm3e'):
                vals = [(eps[k][arm]['l1'], eps[k][arm]['l2'], eps[k][arm]['xy']) for k in eps]
                same[arm] = all(v == vals[0] for v in vals)
            row['episodes_agree'] = same
            row['secs'] = round(time.perf_counter() - t0, 2)
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            print(f"seed {seed} {e['scene_id']}: arm1 {eps['e0']['arm1']['l1']} arm2 {eps['e0']['arm2']['l1']} "
                  f"arm3 {eps['e0']['arm3']['l1']}/{eps['e0']['arm3']['l2']} equiv {eps['e0']['equiv_m']} "
                  f"agree {all(same.values())} [{row['secs']} s, {time.perf_counter() - started:.0f} s]", flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else None)
