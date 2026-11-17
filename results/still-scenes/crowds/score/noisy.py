"""I06 arm 4, noisy C (FROZEN.md section 6 item 4 and section 8): arm 3 with each model's simulated people, 20 draws
per scene per model, on every kept scene (bases and accepted variants). The reference constraints keep the numbers of
listed people (mislocated ones included) and drop the missed. RNG default_rng([20261018, model, crc32(scene id), draw]).
The noise model has no branch on situation, scene, family or level; its role groups read group membership only.

    noisy.py MODEL[,MODEL] SEED [models|placeholder_only] [top]   -> score/work/noisy_<which>_<model>_seed<SEED>.jsonl
    ('top' restricts to the top-level variants A6, B1, C1, D5, E2 and the bases: the secondary arm's scope)
"""
from __future__ import annotations

import json
import sys
import time

import lib as L
import noise as N
from sgi.evaluate import score
from sgi.gt.derive import prepare

DRAWS = 20
TOP = {('A', 6), ('B', 1), ('C', 1), ('D', 5), ('E', 2)}


def main(keys, seed, which='models', top=False):
    import hashlib
    assert hashlib.sha256((L.I06 / 'FROZEN.md').read_bytes()).hexdigest() == L.FROZEN_SHA
    doc = json.loads((L.SCORE / 'noise_model.json').read_text())
    assert doc['check_i03']
    cfg = L.load_config()
    th = cfg.thresholds
    entries = [e for e in L.kept_scenes() if e['seed'] == seed]
    if top:
        entries = [e for e in entries if e['kind'] == 'base' or (e['family'], e['level']) in TOP]
    for key in keys:
        model = N.Model(doc[which][key])
        out = L.SCORE / 'work' / f'noisy_{which}_{key}_seed{seed}{"_top" if top else ""}.jsonl'
        started = time.perf_counter()
        with open(out, 'w') as fh:
            for e in entries:
                t0 = time.perf_counter()
                inputs, truth = L.load(e)
                obs = inputs.observation
                ets = prepare(inputs, truth, cfg)
                cells = L.l1_cells(truth, obs, ets['e0'].gt.acceptable_regions, th)
                g0 = truth.group_for('e0', L.TARGET_ID)
                roles = {p.id: N.role_of(p, g0) for p in truth.people}
                draws = []
                for d in range(DRAWS):
                    rng = N.scene_rng(key, e['scene_id'], d)
                    found = N.simulate(model, rng, truth, obs, th, roles)
                    per_ep, c_prev, inputs_prev = {}, None, None
                    for episode in inputs.episodes:
                        eid = episode.episode_id
                        et = ets[eid]
                        g = truth.group_for(eid, L.TARGET_ID)
                        assert g == g0, (e['scene_id'], eid)
                        same = (et.gt.situation, g)
                        if c_prev is not None and inputs_prev == same:
                            c = c_prev
                        else:
                            c = L.solve_c(inputs, truth, et, th, people_found=found)
                        c_prev, inputs_prev = c, same
                        res = c['result']
                        xy = L.goal_xy(res, obs)
                        sc = score(res, obs, truth, et.gt, th)
                        per_ep[eid] = {'l1': bool(sc.success), 'l2': bool(sc.success and L.l2_ok(xy, truth, g, th)),
                                       'xy': None if xy is None else [round(float(xy[0]), 4), round(float(xy[1]), 4)]}
                    det = L.detectors(inputs, truth, ets['e0'], th, c_prev, cells)
                    n_true = sum(s is not None for _, s in found)
                    n_false = sum(s is None for _, s in found)
                    draws.append({'d': d, 'e0': per_ep['e0'], 'agree': all(v == per_ep['e0'] for v in per_ep.values()),
                                  'n_found_true': n_true, 'n_found_false': n_false,
                                  'members': c_prev.get('members'), 'needed': c_prev.get('needed'),
                                  'det': {k: det[k] for k in ('det_i', 'det_i_false_person', 'joined_sources', 'F_cells',
                                                              'det_ii_core', 'excluded_m', 'target_distance_m',
                                                              'iii_applies', 'off_facing_deg', 'det_iii', 'iv_applies',
                                                              'not_ahead_of', 'not_ahead_reason', 'det_iv_core',
                                                              'l1_cells_standable', 'det_v', 'in_region',
                                                              'far_from_target', 'exclusion_broken')}})
                row = {k: e[k] for k in ('scene_id', 'kind', 'seed', 'row', 'family', 'level', 'base_scene_id')}
                row.update(model=key, which=which, draws=draws,
                           l1_mean=sum(x['e0']['l1'] for x in draws) / DRAWS,
                           l2_mean=sum(x['e0']['l2'] for x in draws) / DRAWS,
                           agree_all=all(x['agree'] for x in draws), secs=round(time.perf_counter() - t0, 2))
                fh.write(json.dumps(row) + '\n')
                fh.flush()
                print(f"{which} {key} seed {seed} {e['scene_id']}: L1 {row['l1_mean']:.2f} L2 {row['l2_mean']:.2f} "
                      f"agree {row['agree_all']} [{row['secs']} s, {time.perf_counter() - started:.0f} s]", flush=True)


if __name__ == '__main__':
    which = sys.argv[3] if len(sys.argv) > 3 else 'models'
    top = len(sys.argv) > 4 and sys.argv[4] == 'top'
    main(sys.argv[1].split(','), int(sys.argv[2]), which, top)
