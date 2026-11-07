"""Marks probe M2P scorer (M2P_FROZEN.md): per arm, over the 60 (episode, trial) cells, the click's target right and
its replayed success (the click alone, against the DeepSeek s23 C record with trial - 6), and the click's distance
from the fresh pipeline arm's click (P0U) at the same trial. Zero calls.   m2p_score.py runs/m2p_dsv41
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

import m1_coverage as M
import m2lib as L

X, H = M.X, M.H
ARMS = ('P0U', 'P2M', 'P2U', 'P0M')


def main(run: Path):
    M.check_frozen()
    from sgi.evaluate import match_target
    wl = H.whitelist()
    rows = [json.loads(l) for l in (run / 'calls.jsonl').read_text().splitlines() if l.strip()]
    recs, cache, cells = {}, {}, collections.defaultdict(dict)
    for r in rows:
        cells[(r['scenes_dir'], r['scene_id'], r['episode'], r['trial'])][r['arm']] = r
    res = collections.defaultdict(lambda: dict(n=0, right=0, replay=0, marks=0, dist=[]))
    for (sdir, sid, eid, t), arms in sorted(cells.items()):
        key = (sdir, sid)
        if key not in cache:
            sref = next(s for s in H.distinct_scenes(wl=wl) if str(s.scenes_dir) == sdir and s.scene_id == sid)
            sc = H.scene(sref.runs[0], sid, wl=wl)
            cache[key] = (sc, X.Geom(sc.observation.depth, sc.observation.camera))
        sc, g = cache[key]
        ep = sc.truth.episodes[eid]
        arm_name = sid.split('_')[0]
        if arm_name not in recs:
            run_ = H.runs(stage='s23', method='con', model='dsv41', arm=arm_name, wl=wl)[0]
            recs[arm_name] = {(x.scene_id, x.episode_id, x.trial): x for x in H.records(run_, wl=wl)}
        rec = recs[arm_name].get((sid, eid, t - 6))
        th = H.recorded_config(rec.ref)[1].thresholds
        base = None
        p0 = arms.get('P0U')
        if p0 and p0['status'] == 'ok':
            s0 = next((s for s in p0['subtasks'] if s['type'] == 'navigate' and s.get('point')), None)
            base = None if s0 is None else s0['point']
        for arm in ARMS:
            a = arms.get(arm)
            if a is None:
                continue
            d = res[arm]
            d['n'] += 1
            if a['status'] != 'ok':
                continue
            s = next((s for s in a['subtasks'] if s['type'] == 'navigate' and s.get('point')), None)
            d['marks'] += any(x.get('mark') is not None for x in a['subtasks'])
            if s is None:
                continue
            tgt = L._lift_click(g, *s['point'])
            right = tgt is not None and match_target(tgt.world, sc.truth, ep, th) is not None
            d['right'] += right
            d['replay'] += bool(right and X.replay_success(rec, tgt)[0])
            if base is not None and arm != 'P0U':
                d['dist'].append(math.hypot(s['point'][0] - base[0], s['point'][1] - base[1]))
    lines = [f'M2P (DeepSeek, {len(cells)} cells): prompt x image, the click alone']
    lines.append(f'{"arm":<5}{"prompt":>8}{"image":>10}{"cells":>7}{"right":>7}{"replay":>8}{"marks":>7}'
                 f'{"move from P0U (px) median / p75":>34}')
    for arm, (pr, im) in zip(ARMS, (('pipe', 'unmarked'), ('marks', 'marked'), ('marks', 'unmarked'),
                                     ('pipe', 'marked'))):
        d = res[arm]
        mv = f'{np.median(d["dist"]):.1f} / {np.percentile(d["dist"], 75):.1f}' if d['dist'] else '-'
        lines.append(f'{arm:<5}{pr:>8}{im:>10}{d["n"]:>7}{d["right"]:>7}{d["replay"]:>8}{d["marks"]:>7}{mv:>34}')
    loss_a2 = res['P0U']['replay'] - res['P2M']['replay']
    loss_p = res['P0U']['replay'] - res['P2U']['replay']
    loss_i = res['P0U']['replay'] - res['P0M']['replay']
    if loss_a2 <= 2:
        verdict = 'no loss to explain: A2 (P2M) does not lose replayed success against a fresh P0U on these cells'
    else:
        p, i = loss_p >= loss_a2 / 2, loss_i >= loss_a2 / 2
        verdict = {(True, False): 'the prompt', (False, True): 'the drawn marks', (True, True): 'both, each alone',
                   (False, False): 'only the two together'}[(p, i)]
    lines.append(f'replay loss against P0U: P2M {loss_a2}, P2U {loss_p}, P0M {loss_i} -> cause: {verdict}')
    text = '\n'.join(lines)
    print(text)
    (run / 'm2p_out.txt').write_text(text + '\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
