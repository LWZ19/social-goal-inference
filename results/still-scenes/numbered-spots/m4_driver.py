"""Marks probe M4 driver (M4_FROZEN.md): marked against unmarked plan calls on the first frames of D1's 5 evaluation
moving scenarios x seeds 0-2 (15 frames), trials 20-22. Candidates are depth components only (no detector or tracked
people: the sim frames carry neither). Run by the run.
    python -B m4_driver.py --models soclaas_glm53 --out runs/m4_glm53 --rate 13 [--dry]
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m2_driver as D  # noqa: E402
import m2lib as L  # noqa: E402
import paths  # noqa: E402  (results/_lib/paths.py, on sys.path through m2_driver)

M, X = D.M, D.X
FRAMES = paths.MOVING_RUNS / 'runs/dsv41'
SCENARIOS = ('cafe', 'cafe_line2_moving', 'cafe_line6_moving', 'room_coffee_cart', 'sweet_spot_line2_moving')
SEEDS = (0, 1, 2)


def frames():
    """(scenario, seed, folder, request, context, observation) of each first frame (C1's, shared by every arm)."""
    from sgi.serve import observation_from
    for sc in SCENARIOS:
        for s in SEEDS:
            f = FRAMES / sc / f'seed{s}' / 'C1' / '001'
            req = json.loads((f / 'request.json').read_text())
            body = dict(rgb_png=base64.b64encode((f / 'rgb.png').read_bytes()).decode(),
                        depth_npy=base64.b64encode((f / 'depth.npy').read_bytes()).decode(),
                        camera=req['camera'], robot=req['robot'])
            yield sc, s, f, req['request'], req.get('context'), observation_from(body)


def candidates(g) -> list:
    """Depth components only (M1's proposer), largest first, mark point below 2.5 m, at most 8."""
    out = []
    for c, cls, size, q, tgt in M.proposer(g):
        if len(out) >= L.MAX_MARKS:
            break
        if tgt.world[2] > L.CEILING_Z:
            continue
        out.append(L.Cand('component', (q[1] + 0.5, q[0] + 0.5), comp=c))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--trials', default='20,21,22')
    ap.add_argument('--out', required=True)
    ap.add_argument('--rate', type=float, default=13.0)
    ap.add_argument('--max-calls', type=int, default=90)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args(argv)
    assert D.sha(D.PROMPTS_A0 / 'plan.md') == D.PLAN_SHA_A0 and D.sha(D.PROMPTS_A2 / 'plan.md') == D.PLAN_SHA_A2
    from sgi.imaging import encode_png
    from sgi.geometry import point_to_pixel
    from sgi.prompts import build_plan_request
    from sgi.schemas import _to_point
    from sgi.serve import make_config
    from sgi.stages.plan import run_plan
    from sgi.vlm.factory import make_client
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / 'calls.jsonl').open('a')
    n, stamps = 0, []
    fr = list(frames())
    for model in a.models.split(','):
        cfg = make_config(model)
        stage = cfg.stages['plan']
        backend = None
        if a.dry:
            from sgi.vlm.mock import MockBackend, centre_responder
            backend = MockBackend(centre_responder)
        client = make_client(stage.model, cfg, backend=backend)
        if a.dry:
            client._cache = None
        pf = client.entry.point_format
        for sc, s, f, request, context, obs in fr:
            g = X.Geom(obs.depth, obs.camera)
            cands = candidates(g)
            for t in (int(x) for x in a.trials.split(',')):
                nums = L.numbering(len(cands), f'{sc}|{s}', t)
                for arm in ('U', 'M'):
                    if n >= a.max_calls:
                        return 3
                    now = time.time()
                    stamps = [x for x in stamps if now - x < 60.0]
                    if len(stamps) >= a.rate:
                        time.sleep(60.0 - (now - stamps[0]) + 0.05)
                    stamps.append(time.time())
                    n += 1
                    if arm == 'U':
                        img = encode_png(obs.rgb, cfg.run.max_image_side)
                        plan, resp = run_plan(client, stage.strategy, request, context, img, None, trial=t,
                                              prompts_dir=D.PROMPTS_A0)
                        subs = [] if plan is None else [dict(type=x.type, target=x.target, mark=None,
                                                             point=None if x.target_point is None else
                                                             list(point_to_pixel(x.target_point.x, x.target_point.y,
                                                                                 obs.camera.width, obs.camera.height)))
                                                        for x in plan.subtasks]
                    else:
                        img = encode_png(L.render(obs.rgb, cands, nums), cfg.run.max_image_side)
                        system, parts = build_plan_request(stage.strategy, request, context, img, None, pf,
                                                           prompts_dir=D.PROMPTS_A2)
                        size = (img.width, img.height)
                        resp = client.generate(parts, D.marked_plan_schema(stage.strategy, pf, size, nums),
                                               system=system, trial=t)
                        subs = []
                        if resp.status == 'ok':
                            for x in resp.parsed['subtasks']:
                                p = _to_point(x.get('target_point'), pf, size)
                                subs.append(dict(type=x['type'], target=x['target'], mark=x.get('target_mark'),
                                                 point=None if p is None else
                                                 list(point_to_pixel(p.x, p.y, obs.camera.width, obs.camera.height))))
                    row = dict(model=model, arm=arm, scenario=sc, seed=s, frame=str(f), trial=t, status=resp.status,
                               cached=bool(getattr(resp, 'cached', False)), subtasks=subs, nums=nums,
                               cands=[dict(kind=c.kind, mark=list(c.mark)) for c in cands])
                    log.write(json.dumps(row, default=str) + '\n')
                    log.flush()
                    print(f'{n:>4} {model} {arm} {sc} s{s} t{t} {resp.status}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
