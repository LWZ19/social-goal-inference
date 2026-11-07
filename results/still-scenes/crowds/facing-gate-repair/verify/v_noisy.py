"""Verifier arm 1b: I06's noisy arm on the kept line scenes, 20 draws, R0 and RF90 on the same simulated people,
with my own _turn (vturn.py). I06's lib.py and noise.py imported unchanged.

    v_noisy.py MODEL SEED  -> verify/out/noisy_<model>_seed<SEED>.jsonl
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
RD = paths.STILL
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(RD / 'crowds/score'))
sys.path.insert(0, str(HERE))
import lib as L  # noqa: E402
import noise as N  # noqa: E402
from sgi.evaluate import score  # noqa: E402
from sgi.gt.derive import prepare  # noqa: E402
import vturn as V  # noqa: E402

DRAWS = 20
RULES = ("R0", "RF90")
FROZEN_SHA = '02432f62320c0ebb0924c0f0e5b4b2089d508eb4a85fd28cf56641c25e27560f'
SOLVER_SHA = '10ece70f521eff2b1224267fa0b91bfac546d5740f85622073989a16c6c5d800'


def main(key, seed):
    assert hashlib.sha256((RD / 'crowds/facing-gate-repair/FROZEN.md').read_bytes()).hexdigest() == FROZEN_SHA
    assert hashlib.sha256(Path(V.S.__file__).read_bytes()).hexdigest() == SOLVER_SHA
    doc = json.loads((L.SCORE / 'noise_model.json').read_text())
    assert doc['check_i03']
    model = N.Model(doc['models'][key])
    cfg = L.load_config()
    th = cfg.thresholds
    entries = [e for e in L.kept_scenes() if e['seed'] == seed]
    t_start = time.time()
    n_line = 0
    with open(HERE / 'out' / f'noisy_{key}_seed{seed}.jsonl', 'w') as fh:
        for e in entries:
            inputs, truth = L.load(e)
            sits = sorted({ep.situation for ep in truth.episodes.values()})
            if sits != ['waiting_line']:
                continue
            n_line += 1
            obs = inputs.observation
            ets = prepare(inputs, truth, cfg)
            g0 = truth.group_for('e0', L.TARGET_ID)
            roles = {p.id: N.role_of(p, g0) for p in truth.people}
            members = {p.id for p in truth.people if g0 is not None and p.group == g0}
            draws = []
            for d in range(DRAWS):
                rng = N.scene_rng(key, e['scene_id'], d)
                found = N.simulate(model, rng, truth, obs, th, roles)
                out = {'d': d}
                for rule in RULES:
                    per_ep, c_prev, prev = {}, None, None
                    V.take_log()
                    for episode in inputs.episodes:
                        eid = episode.episode_id
                        et = ets[eid]
                        g = truth.group_for(eid, L.TARGET_ID)
                        same = (et.gt.situation, g)
                        if c_prev is not None and prev == same:
                            c = c_prev
                        else:
                            with V.rule_on(rule):
                                c = L.solve_c(inputs, truth, et, th, people_found=found)
                        c_prev, prev = c, same
                        res = c['result']
                        xy = L.goal_xy(res, obs)
                        sc = score(res, obs, truth, et.gt, th)
                        per_ep[eid] = {'l1': bool(sc.success), 'l2': bool(sc.success and L.l2_ok(xy, truth, g, th)),
                                       'xy': None if xy is None else [round(float(xy[0]), 4), round(float(xy[1]), 4)]}
                    log = V.take_log()
                    note = c_prev['note']
                    joined = [] if note is None else [int(n) for n in note.joined_waiting]
                    src = c_prev.get('source_of') or {}
                    jsrc = [src.get(n) for n in joined]
                    out[rule] = {'e0': per_ep['e0'], 'agree': all(v == per_ep['e0'] for v in per_ep.values()),
                                 'joined': joined, 'det_i': any(s is not None and s not in members for s in jsrc),
                                 'log': log}
                draws.append(out)
            row = {k: e[k] for k in ('scene_id', 'kind', 'seed', 'row', 'family', 'level', 'base_scene_id')}
            row['model'] = key
            row['draws'] = draws
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            print(f"{key} {seed} {e['scene_id']}: R0 {sum(x['R0']['e0']['l1'] for x in draws)} RF90 "
                  f"{sum(x['RF90']['e0']['l1'] for x in draws)} [{time.time() - t_start:.0f} s]", flush=True)
    print(f"done {key} seed {seed}: {n_line} line scenes; shadow {V.STATS}", flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
