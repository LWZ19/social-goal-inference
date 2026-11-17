"""Marks probe M2P driver (M2P_FROZEN.md): a fresh 2x2 of prompt (pipeline / marks) x image (unmarked / marked) for
DeepSeek on the 20 episodes behind M2's 28 moved clicks, trials 6, 7, 8. Run by the run.

    python -B m2p_driver.py --out runs/m2p_dsv41 --rate 13 [--dry]
Arms: P0U = the pipeline's plan call (as M2's A0); P2M = M2's A2; P2U = A2's prompt and schema on the unmarked image;
P0M = the pipeline's prompt on the marked image. Imports M2's frozen driver and library unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m2_driver as D  # noqa: E402
import m2lib as L  # noqa: E402

ARMS = ('P0U', 'P2M', 'P2U', 'P0M')
MODEL = 'soclaas_dsv41'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--trials', default='6,7,8')
    ap.add_argument('--rate', type=float, default=13.0)
    ap.add_argument('--max-calls', type=int, default=240)
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
    wanted = {tuple(x) for x in json.loads((HERE / 'm2p_episodes.json').read_text())}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / 'calls.jsonl').open('a')
    cfg = make_config(MODEL)
    stage = cfg.stages['plan']
    backend = None
    if a.dry:
        from sgi.vlm.mock import MockBackend, centre_responder
        backend = MockBackend(centre_responder)
    client = make_client(stage.model, cfg, backend=backend)
    if a.dry:
        client._cache = None
    pf = client.entry.point_format
    n, stamps = 0, []
    for sref, sc, eid, request, context in D.episodes():
        if (sref.scene_id, eid) not in wanted:
            continue
        obs = sc.observation
        g = D.X.Geom(obs.depth, obs.camera)
        cands = L.candidates(str(sref.folder), g)
        for t in (int(x) for x in a.trials.split(',')):
            nums = L.numbering(len(cands), sref.scene_id, t)
            for arm in ARMS:
                if n >= a.max_calls:
                    return 3
                now = time.time()
                stamps = [x for x in stamps if now - x < 60.0]
                if len(stamps) >= a.rate:
                    time.sleep(60.0 - (now - stamps[0]) + 0.05)
                stamps.append(time.time())
                n += 1
                marked_img = arm in ('P2M', 'P0M')
                img = encode_png(L.render(obs.rgb, cands, nums) if marked_img else obs.rgb, cfg.run.max_image_side)
                if arm in ('P0U', 'P0M'):
                    plan, resp = run_plan(client, stage.strategy, request, context, img, None, trial=t,
                                          prompts_dir=D.PROMPTS_A0)
                    subs = [] if plan is None else [dict(type=s.type, target=s.target, mark=None,
                                                         point=None if s.target_point is None else
                                                         list(point_to_pixel(s.target_point.x, s.target_point.y,
                                                                             obs.camera.width, obs.camera.height)))
                                                    for s in plan.subtasks]
                else:
                    system, parts = build_plan_request(stage.strategy, request, context, img, None, pf,
                                                       prompts_dir=D.PROMPTS_A2)
                    size = (img.width, img.height)
                    resp = client.generate(parts, D.marked_plan_schema(stage.strategy, pf, size, nums),
                                           system=system, trial=t)
                    subs = []
                    if resp.status == 'ok':
                        for s in resp.parsed['subtasks']:
                            p = _to_point(s.get('target_point'), pf, size)
                            subs.append(dict(type=s['type'], target=s['target'], mark=s.get('target_mark'),
                                             point=None if p is None else
                                             list(point_to_pixel(p.x, p.y, obs.camera.width, obs.camera.height))))
                row = dict(model=MODEL, arm=arm, scenes_dir=str(sref.scenes_dir), scene_id=sref.scene_id, episode=eid,
                           trial=t, status=resp.status, cached=bool(getattr(resp, 'cached', False)), subtasks=subs,
                           nums=nums, cands=[dict(kind=c.kind, mark=list(c.mark)) for c in cands])
                log.write(json.dumps(row, default=str) + '\n')
                log.flush()
                print(f'{n:>4} {arm} {sref.scene_id} {eid} t{t} {resp.status}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
