"""Verifier: arms 1-3 (and arm 3 with no constraint) on every kept scene of one seed; L1, L2, detectors.

    v_arms.py SEED  -> out/arms_seed<SEED>.jsonl
"""
from __future__ import annotations

import json
import math
import sys
import time

import vlib as V
from sgi.gt.derive import prepare
from sgi.pipeline import build_pipeline
from sgi.run import run_episode
from sgi.types import GoalResult


def configs():
    cfg = V.load_config()
    for name, st in cfg.stages.items():
        assert cfg.models[st.model].provider == 'mock', (name, st.model)
    run = cfg.run
    upd1 = dict(grounding='oracle', context_check='always_yes', goal='oracle', use_cache=False, trials=1)
    upd2 = dict(grounding='oracle', context_check='oracle', goal='constraints', people='oracle', constraints='none',
                use_cache=False, trials=1)
    c1 = cfg.model_copy(update={'run': run.model_copy(update=upd1)})
    c2 = cfg.model_copy(update={'run': run.model_copy(update=upd2)})
    return cfg, c1, c2


def main(seed):
    cfg, c1, c2 = configs()
    th = cfg.thresholds
    pipe1, pipe2 = build_pipeline(c1), build_pipeline(c2)
    ents = [e for e in V.kept() if e['seed'] == seed]
    base_polys = {}
    out = V.OUT / f'arms_seed{seed}.jsonl'
    t_start = time.perf_counter()
    with open(out, 'w') as fh:
        for e in ents:
            t0 = time.perf_counter()
            inputs, truth = V.load_entry(e)
            obs = inputs.observation
            ets = prepare(inputs, truth, cfg)
            polys = [[list(map(float, pt)) for pt in reg.polygon] for reg in ets['e0'].gt.acceptable_regions]
            if e['kind'] == 'base':
                base_polys[e['row']] = polys
            row = {k: e[k] for k in ('scene_id', 'kind', 'seed', 'row', 'family', 'level', 'base_scene_id')}
            row['situation'] = ets['e0'].gt.situation
            row['l1_invariant'] = polys == base_polys.get(e['row'])
            g0 = truth.group_for('e0', V.T0)
            cells = V.lattice_l1(truth, obs, ets['e0'].gt.acceptable_regions, th)
            row['n_l1_cells'] = len(cells)
            row['n_l2_cells'] = V.l2_cells(cells, truth, g0, th)
            row['calls'] = 0
            eps = {}
            for ep in inputs.episodes:
                eid = ep.episode_id
                et = ets[eid]
                g = truth.group_for(eid, V.T0)
                res = {}
                for arm, pipe, conf in (('arm1', pipe1, c1), ('arm2', pipe2, c2)):
                    rec = run_episode(pipe, inputs, truth, et, ep, conf, 0)
                    row['calls'] += len(rec['result'].get('calls') or [])
                    gr = GoalResult.model_validate(rec['result'])
                    xy = V.goal_of(gr, obs)
                    l1 = bool(rec['score']['success'])
                    res[arm] = {'l1': l1, 'l2': bool(l1 and V.l2_pass(xy, truth, g, th)),
                                'xy': None if xy is None else [float(xy[0]), float(xy[1])],
                                'th': None if gr.pose_world is None else float(gr.pose_world[2])}
                    if arm == 'arm2':
                        src = {}
                        if gr.people is not None:
                            idx = {(p.x, p.y): p.id for p in truth.people}
                            src = {d.number: idx.get((d.xy[0], d.xy[1])) for d in gr.people}
                        c = {'result': gr, 'note': gr.solver, 'uses': gr.constraints or [], 'people': gr.people or [],
                             'src': src}
                        res[arm]['det'] = V.detect(inputs, truth, et, th, c, cells)
                for arm, wc in (('arm3', True), ('arm3e', False)):
                    c = V.oracle_c(inputs, truth, et, th, with_constraints=wc)
                    gr = c['result']
                    xy = V.goal_of(gr, obs)
                    sc = V.pkg_score(gr, obs, truth, et.gt, th)
                    l1 = bool(sc.success)
                    res[arm] = {'l1': l1, 'l2': bool(l1 and V.l2_pass(xy, truth, g, th)),
                                'xy': None if xy is None else [float(xy[0]), float(xy[1])],
                                'th': None if gr.pose_world is None else float(gr.pose_world[2]),
                                'matter': c['matter']}
                    if arm == 'arm3':
                        res[arm]['det'] = V.detect(inputs, truth, et, th, c, cells)
                        res[arm]['members'] = c.get('members')
                        res[arm]['needed'] = c.get('needed')
                        res[arm]['cons'] = c['cons']
                        res[arm]['l2_reasons'] = V.l2_reasons(xy, truth, g, th) if xy is not None else None
                a, b = res['arm3e']['xy'], res['arm2']['xy']
                res['equiv_m'] = None if (a is None and b is None) else (math.inf if (a is None or b is None)
                                                                         else math.dist(a, b))
                eps[eid] = res
            row['eps'] = eps
            same = {}
            for arm in ('arm1', 'arm2', 'arm3', 'arm3e'):
                v = [(eps[k][arm]['l1'], eps[k][arm]['l2'], eps[k][arm]['xy']) for k in sorted(eps)]
                same[arm] = all(x == v[0] for x in v)
            row['same_eps'] = same
            row['secs'] = round(time.perf_counter() - t0, 2)
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            e0 = eps['e0']
            print(f"s{seed} {e['scene_id']}: a1 {e0['arm1']['l1']} a2 {e0['arm2']['l1']} a3 {e0['arm3']['l1']}/"
                  f"{e0['arm3']['l2']} eq {e0['equiv_m']} same {all(same.values())} [{row['secs']} s, "
                  f"{time.perf_counter() - t_start:.0f} s]", flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
