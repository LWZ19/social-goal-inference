"""Marks probe M2 driver (M2_FROZEN.md): paired plan calls, unmarked (A0) and marked (A2), on the 42 realistic
development scenes x 2 episodes, at fresh trial indices. Run separately; it is never run live here.

    .venv/bin/python -B m2_driver.py --models soclaas_qwen38_low,soclaas_glm53,soclaas_dsv41 --trials 3,4,5 \
        --out runs/<name> --max-calls 1600 --rate 40            (live: run by the run)
    .venv/bin/python -B m2_driver.py --dry --models soclaas_glm53 --trials 3 --episodes 4 --out runs/dry
        (zero calls: a mock back end answers, no key is read)

Each call is one line of <out>/calls.jsonl, written as it returns. The order is model, then (scene, episode),
then trial, then A0 before A2, so the two arms of a pair are asked minutes apart at most. A0 uses the repository's
prompts/ unchanged (sha256 checked); A2 uses prompts_marks/, whose plan.md differs only in its point_format block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Literal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m1_coverage as M  # noqa: E402
import m2lib as L  # noqa: E402

X, H = M.X, M.H
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / '_lib' / 'paths.py').exists()) / '_lib'))
import paths  # noqa: E402  (results/_lib/paths.py)
WT = paths.REPO                     # the repository root
PROMPTS_A0 = WT / 'prompts'
PROMPTS_A2 = HERE / 'prompts_marks'
PLAN_SHA_A0 = '00f2fae826be7433b56de529c8e13d3eb4d8b25da83bb70c0e398ca527933c38'   # prompts/plan.md as in s21/s23
PLAN_SHA_A2 = '102494e09ab5ccb1734949c2fb908b7c28ad30154dba28ea40606623909f430c'   # prompts_marks/plan.md, frozen in M2_FROZEN.md


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def marked_plan_schema(strategy, point_format, size, nums):
    from sgi.schemas import _PLAN_DESC, _reasoning_field, _wire_model, _wire_point
    sub = _wire_model('SubtaskWire', _PLAN_DESC, description=str, target=str | None,
                      target_mark=Literal[tuple(sorted(nums))] | None,
                      target_point=_wire_point(point_format, size) | None, type=Literal['navigate', 'act'])
    return _wire_model('PlanWire', _PLAN_DESC, **_reasoning_field(strategy), subtasks=list[sub])


def episodes():
    """(scene ref, run, scene id, episode id, request, context) for the realistic development scenes."""
    wl = H.whitelist()
    for sref in H.distinct_scenes(wl=wl):
        if not M.is_realistic(sref.scene_id):
            continue
        sc = H.scene(sref.runs[0], sref.scene_id, wl=wl)
        for ep in sc.inputs.episodes:
            yield sref, sc, ep.episode_id, ep.request, ep.context


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--trials', default='3,4,5')
    ap.add_argument('--arms', default='A0,A2')
    ap.add_argument('--episodes', type=int, default=0, help='first N episodes only (smoke)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-calls', type=int, default=1600)
    ap.add_argument('--rate', type=float, default=40.0, help='calls per minute at most, over this driver')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args(argv)

    assert sha(PROMPTS_A0 / 'plan.md') == PLAN_SHA_A0, 'prompts/plan.md is not the s21/s23 text'
    if PLAN_SHA_A2:
        assert sha(PROMPTS_A2 / 'plan.md') == PLAN_SHA_A2, 'prompts_marks/plan.md changed after the freeze'
    for f in ('constraints', 'context', 'goal', 'locate', 'people'):
        assert sha(PROMPTS_A2 / f'{f}.md') == sha(PROMPTS_A0 / f'{f}.md')

    from sgi.imaging import encode_png
    from sgi.prompts import build_plan_request
    from sgi.schemas import _to_point
    from sgi.geometry import point_to_pixel
    from sgi.serve import make_config
    from sgi.stages.plan import run_plan
    from sgi.vlm.factory import make_client

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / 'calls.jsonl').open('a')
    trials = [int(t) for t in a.trials.split(',')]
    arms = a.arms.split(',')
    n_calls, t_last = 0, []
    eps = list(episodes())
    if a.episodes:
        eps = eps[:a.episodes]
    for model in a.models.split(','):
        cfg = make_config(model)
        stage = cfg.stages['plan']
        backend = None
        if a.dry:
            from sgi.vlm.mock import MockBackend, centre_responder
            backend = MockBackend(centre_responder)
        client = make_client(stage.model, cfg, backend=backend)
        if a.dry:
            client._cache = None                   # a dry run writes nothing into any cache
        pf = client.entry.point_format
        for sref, sc, eid, request, context in eps:
            obs = sc.observation
            g = X.Geom(obs.depth, obs.camera)
            cands = L.candidates(str(sref.folder), g)
            for t in trials:
                nums = L.numbering(len(cands), sref.scene_id, t)
                for arm in arms:
                    if n_calls >= a.max_calls:
                        print(f'stop: --max-calls {a.max_calls} reached', flush=True)
                        return 3
                    now = time.time()                  # the driver's own pace, on top of the client's pacer
                    t_last = [x for x in t_last if now - x < 60.0]
                    if len(t_last) >= a.rate:
                        time.sleep(60.0 - (now - t_last[0]) + 0.05)
                    t_last.append(time.time())
                    n_calls += 1
                    row = dict(model=model, arm=arm, scenes_dir=str(sref.scenes_dir), scene_id=sref.scene_id,
                               episode=eid, trial=t, strategy=stage.strategy, point_format=pf, started=time.time())
                    if arm == 'A0':
                        image = encode_png(obs.rgb, cfg.run.max_image_side)
                        plan, resp = run_plan(client, stage.strategy, request, context, image, None, trial=t,
                                              prompts_dir=PROMPTS_A0)
                        subs = [] if plan is None else [dict(type=s.type, target=s.target,
                                                             point=None if s.target_point is None else
                                                             list(point_to_pixel(s.target_point.x, s.target_point.y,
                                                                                 obs.camera.width, obs.camera.height)),
                                                             mark=None) for s in plan.subtasks]
                    else:
                        marked = L.render(obs.rgb, cands, nums)
                        image = encode_png(marked, cfg.run.max_image_side)
                        system, parts = build_plan_request(stage.strategy, request, context, image, None, pf,
                                                           prompts_dir=PROMPTS_A2)
                        size = (image.width, image.height)
                        resp = client.generate(parts, marked_plan_schema(stage.strategy, pf, size, nums),
                                               system=system, trial=t)
                        subs = []
                        if resp.status == 'ok':
                            for s in resp.parsed['subtasks']:
                                p = _to_point(s.get('target_point'), pf, size)
                                subs.append(dict(type=s['type'], target=s['target'], mark=s.get('target_mark'),
                                                 point=None if p is None else
                                                 list(point_to_pixel(p.x, p.y, obs.camera.width, obs.camera.height))))
                    row.update(status=resp.status, cached=bool(getattr(resp, 'cached', False)),
                               latency_s=getattr(resp, 'latency_total_s', None), subtasks=subs,
                               reasoning=(resp.parsed or {}).get('reasoning') if resp.status == 'ok' else None,
                               nums=nums, cands=[dict(kind=c.kind, mark=list(c.mark)) for c in cands],
                               usage=getattr(resp, 'usage', None) and resp.usage.model_dump())
                    log.write(json.dumps(row, default=str) + '\n')
                    log.flush()
                    print(f'{n_calls:>5} {model} {arm} {sref.scene_id} {eid} t{t} {resp.status}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
