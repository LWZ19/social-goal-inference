"""POST HOC diagnosis of M1's surprise (not in M1_FROZEN.md): the proposer covers every target, yet a perfect pick
replays below the models' own clicks for Haiku, Sonnet and Qwen. For every new-wording C record whose outcome
changes under the perfect pick, it reports why. It also replays a variant M1b-snap: the model's own click is snapped
into the candidate component that the perfect pick chose. That keeps where on the target the model pointed, which is
the information a single mark per component throws away. Zero calls, CPU only.
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import numpy as np

import m1_coverage as M

X, H = M.X, M.H
HERE = Path(__file__).resolve().parent


def snap_into(g, cp, c, u, v):
    """The model's click moved into component c: the c pixel nearest (u, v), lifted with c's own depths."""
    ii, jj = np.nonzero(cp.label == c)
    d = (ii + 0.5 - v) ** 2 + (jj + 0.5 - u) ** 2
    k = int(np.argmin(d))
    return X.comp_lift(g, cp, c, (int(ii[k]), int(jj[k])))


def main():
    M.check_frozen()
    from sgi.evaluate import match_target
    wl = H.whitelist()
    runs = H.runs(stage='s21', method='con', wl=wl) + H.runs(stage='s23', method='con', wl=wl)
    rows, cache = [], {}
    for run in runs:
        for rec in H.records(run, wl=wl):
            if rec.target is None:
                continue
            sc = H.scene(rec.ref, rec.scene_id, wl=wl)
            ep = sc.truth.episodes[rec.episode_id]
            th = H.recorded_config(rec.ref)[1].thresholds
            key = (str(rec.ref.scenes_dir), rec.scene_id)   # was id(sc.observation): ids are reused after the harness evicts a scene
            if key not in cache:
                g = X.Geom(sc.observation.depth, sc.observation.camera)
                cache[key] = (g, M.proposer(g))
            g, cands = cache[key]
            cp = g.comps('G1')
            rec_ok = bool(rec.score.get('success'))
            rec_world = rec.target['world']
            u, v = rec.target['pixel']
            rec_match = match_target(rec_world, sc.truth, ep, th)
            # perfect pick, as M1 judged it (first 20)
            pick, pick_ok, pick_why = None, False, None
            matching = []
            for rank, (c, cls, size, q, tgt) in enumerate(cands[:M.JUDGED_CAP]):
                if match_target(tgt.world, sc.truth, ep, th) is None:
                    continue
                ok, why = X.replay_success(rec, tgt)
                matching.append((rank, c, cls, size, tgt, ok, why))
                if ok and not pick_ok:
                    pick, pick_ok, pick_why = (rank, c, cls, size, tgt), True, why
            if pick is None and matching:
                pick = matching[0][:5]
                pick_why = matching[0][6]
            # M1b-snap: the click moved into the component the pick (or the first matching candidate) chose
            snap_ok = None
            if pick is not None:
                st = snap_into(g, cp, pick[1], u, v)
                if match_target(st.world, sc.truth, ep, th) is not None:
                    snap_ok, _ = X.replay_success(rec, st)
                else:
                    snap_ok = False
            dist = None if pick is None else math.hypot(pick[4].world[0] - rec_world[0], pick[4].world[1] - rec_world[1])
            t = sc.truth.targets[rec_match] if rec_match else sc.truth.targets[ep.target_ids[0]]
            rows.append(dict(model=rec.model, scene=rec.scene_id, episode=rec.episode_id, trial=rec.trial,
                             situation=rec.situation, kind=t.kind, source=rec.result.get('source'),
                             rec_ok=rec_ok, rec_target_right=rec_match is not None, pick_ok=pick_ok,
                             pick_class=None if pick is None else pick[2], pick_size=None if pick is None else pick[3],
                             pick_why=pick_why, n_matching=len(matching), dist_m=dist, snap_ok=snap_ok))
    out = []
    say = out.append
    say('M1b (POST HOC): why the perfect pick replays below the models\' own clicks')
    lost = [r for r in rows if r['rec_ok'] and not r['pick_ok']]
    won = [r for r in rows if not r['rec_ok'] and r['pick_ok']]
    say(f'records with a target: {len(rows)}; lost by the pick (success -> failure): {len(lost)}; '
        f'won (failure -> success): {len(won)}')
    for name, rr in (('LOST', lost), ('WON', won)):
        say(f'\n== {name}: by model / situation / target kind / recorded source / picked class')
        for field in ('model', 'situation', 'kind', 'source', 'pick_class'):
            say(f'  {field}: ' + ', '.join(f'{k} {v}' for k, v in collections.Counter(r[field] for r in rr).most_common()))
        d = [r['dist_m'] for r in rr if r['dist_m'] is not None]
        if d:
            say(f'  distance pick point -> model point (m): median {np.median(d):.2f}, p10 {np.percentile(d, 10):.2f}, '
                f'p90 {np.percentile(d, 90):.2f}')
        say('  pick replay reasons: ' + ', '.join(f'{k} {v}' for k, v in collections.Counter(r['pick_why'] for r in rr).most_common()))
    say('\n== M1b-snap (click moved into the picked component): successes per model')
    say(f'{"model":<14}{"recorded":>9}{"pick":>6}{"snap":>6}')
    by = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        b = by[r['model']]
        b[0] += r['rec_ok']
        b[1] += r['pick_ok']
        b[2] += bool(r['snap_ok']) if r['snap_ok'] is not None else r['rec_ok']
    for m, (a, b_, c) in sorted(by.items()):
        say(f'{m:<14}{a:>9}{b_:>6}{c:>6}   (records with a target only)')
    say(f'pooled: recorded {sum(v[0] for v in by.values())}, pick {sum(v[1] for v in by.values())}, '
        f'snap {sum(v[2] for v in by.values())}')
    text = '\n'.join(out)
    print(text)
    (HERE / 'm1b_out.txt').write_text(text + '\n')
    (HERE / 'm1b_rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))


if __name__ == '__main__':
    main()
