"""Verifier: arm 4 (noisy C), 20 draws per scene and model, my simulation (v_noise.simulate).

    v_noisy.py MODEL SEED [all|top] [models|placeholder_only]  -> out/noisy_<which>_<MODEL>_s<SEED>_<scope>.jsonl
'top' = the bases and the top-level variants A6, B1, C1, D5, E2.
"""
from __future__ import annotations

import json
import sys
import time

import vlib as V
import v_noise as N
from sgi.gt.derive import prepare, reference_people_matter

DRAWS = 20


def main(key, seed, scope='all', which='models'):
    cfg = V.load_config()
    th = cfg.thresholds
    nz = N.load_noise(which)[key]
    ents = [e for e in V.kept() if e['seed'] == seed]
    if scope == 'top':
        ents = [e for e in ents if e['kind'] == 'base' or (e['family'], e['level']) in V.TOP]
    out = V.OUT / f'noisy_{which}_{key}_s{seed}_{scope}.jsonl'
    t_start = time.perf_counter()
    with open(out, 'w') as fh:
        for e in ents:
            t0 = time.perf_counter()
            inputs, truth = V.load_entry(e)
            obs = inputs.observation
            ets = prepare(inputs, truth, cfg)
            cells = V.lattice_l1(truth, obs, ets['e0'].gt.acceptable_regions, th)
            g0 = truth.group_for('e0', V.T0)
            roles = {p.id: N.role(p, g0) for p in truth.people}
            sig = {eid: (et.gt.situation, truth.group_for(eid, V.T0), reference_people_matter(et, V.T0))
                   for eid, et in ets.items()}
            draws = []
            for d in range(DRAWS):
                found = N.simulate(nz, N.rng_for(key, e['scene_id'], d), truth, obs, th, roles)
                per = {}
                c0 = None
                for ep in inputs.episodes:
                    eid = ep.episode_id
                    et = ets[eid]
                    c = c0 if (c0 is not None and sig[eid] == sig['e0']) else V.oracle_c(inputs, truth, et, th, found=found)
                    if eid == 'e0':
                        c0 = c
                    xy = V.goal_of(c['result'], obs)
                    sc = V.pkg_score(c['result'], obs, truth, et.gt, th)
                    g = truth.group_for(eid, V.T0)
                    per[eid] = (bool(sc.success), bool(sc.success and V.l2_pass(xy, truth, g, th)))
                det = V.detect(inputs, truth, ets['e0'], th, c0, cells)
                draws.append({'l1': per['e0'][0], 'l2': per['e0'][1], 'agree': all(v == per['e0'] for v in per.values()),
                              'n_true': sum(s is not None for _, s in found), 'n_false': sum(s is None for _, s in found),
                              'det': {k: det[k] for k in ('i', 'ii', 'iii', 'iv', 'v', 'F', 'off_deg', 'joined_src')}})
            row = {k: e[k] for k in ('scene_id', 'kind', 'seed', 'row', 'family', 'level', 'base_scene_id')}
            row.update(model=key, which=which, draws=draws, l1=sum(x['l1'] for x in draws) / DRAWS,
                       l2=sum(x['l2'] for x in draws) / DRAWS, agree=all(x['agree'] for x in draws),
                       secs=round(time.perf_counter() - t0, 2))
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            print(f"{which} {key} s{seed} {e['scene_id']}: L1 {row['l1']:.2f} L2 {row['l2']:.2f} [{row['secs']} s, "
                  f"{time.perf_counter() - t_start:.0f} s]", flush=True)


if __name__ == '__main__':
    a = sys.argv
    main(a[1], int(a[2]), a[3] if len(a) > 3 else 'all', a[4] if len(a) > 4 else 'models')
