"""Verifier arm 1: I06 arm 3 (oracle C) on the 229 kept scenes, per rule, with my own _turn (vturn.py).

    v_crowd.py SEED  -> verify/out/crowd_seed<SEED>.jsonl
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

RD = Path('results/2026-10-17-beyond-vlm')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(RD / 'phase2/i06/score'))
sys.path.insert(0, str(HERE))
import lib as L  # noqa: E402  (I06's, unchanged)
from sgi.evaluate import score  # noqa: E402
from sgi.gt.derive import prepare  # noqa: E402
import vturn as V  # noqa: E402

RULES = ("R0", "RF90", "RF60", "RN")
FROZEN_SHA = '02432f62320c0ebb0924c0f0e5b4b2089d508eb4a85fd28cf56641c25e27560f'
SOLVER_SHA = '10ece70f521eff2b1224267fa0b91bfac546d5740f85622073989a16c6c5d800'


def main(seed):
    assert hashlib.sha256((RD / 'phase2/i06_repair/FROZEN.md').read_bytes()).hexdigest() == FROZEN_SHA
    assert hashlib.sha256(Path(V.S.__file__).read_bytes()).hexdigest() == SOLVER_SHA
    cfg = L.load_config()
    th = cfg.thresholds
    entries = [e for e in L.kept_scenes() if e['seed'] == seed]
    out = HERE / 'out' / f'crowd_seed{seed}.jsonl'
    t_start = time.time()
    with open(out, 'w') as fh:
        for e in entries:
            inputs, truth = L.load(e)
            obs = inputs.observation
            ets = prepare(inputs, truth, cfg)
            row = {k: e[k] for k in ('scene_id', 'kind', 'seed', 'row', 'family', 'level', 'base_scene_id')}
            row['situation'] = sorted({ep.situation for ep in truth.episodes.values()})
            per = {}
            for rule in RULES:
                eps = {}
                for episode in inputs.episodes:
                    eid = episode.episode_id
                    et = ets[eid]
                    g = truth.group_for(eid, L.TARGET_ID)
                    V.take_log()
                    with V.rule_on(rule):
                        c = L.solve_c(inputs, truth, et, th)
                    log = V.take_log()
                    res = c['result']
                    xy = L.goal_xy(res, obs)
                    sc = score(res, obs, truth, et.gt, th)
                    note = c['note']
                    joined = [] if note is None else [int(n) for n in note.joined_waiting]
                    src = c.get('source_of') or {}
                    members = {p.id for p in truth.people if g is not None and p.group == g}
                    jsrc = [src.get(n) for n in joined]
                    yaws_known = None if c['people'] is None else all(p.yaw is not None for p in c['people'])
                    eps[eid] = {'l1': bool(sc.success),
                                'l2': bool(sc.success and L.l2_ok(xy, truth, g, th)),
                                'xy': None if xy is None else [float(xy[0]), float(xy[1])],
                                'joined': joined, 'joined_sources': jsrc,
                                'det_i': any(s is not None and s not in members for s in jsrc),
                                'tdist': None if note is None else float(note.target_distance_m),
                                'yaws_known': yaws_known, 'log': log}
                per[rule] = eps
            row['rules'] = per
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            print(f"{seed} {e['scene_id']}: " + ' '.join(f"{r}:{int(per[r]['e0']['l1'])}" for r in RULES)
                  + f" [{time.time() - t_start:.0f} s]", flush=True)
    print(f"done seed {seed}; shadow {V.STATS}", flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
