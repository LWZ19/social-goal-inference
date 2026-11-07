"""Verifier step 1b: rebuild the development-record arrays for settings (b) and (a) with my own code, all 549
goal-step records, read only through research_harness.

(b) recorded people (recorded yaws), recorded constraints, recorded target; raster origin result.target.world.
(a) I01 P's people and constraints, taken from phase1/i01_repairs/repairs.py's OWN solver_row: H.resolve is
    intercepted for the duration of one solver_row call and the 8th call's arguments (arm P: people=p_p,
    constraints=c_strip) are captured; the real H.resolve is then run once on them for P's place and verdict.
For each setting: _over_floor (F, excluded_m keys as the solver rounds them, picks), the floor, my raster, rho_room
(share of F cells in the matched target's acceptable regions; 0 for a wrong target). Asserts: picks[0] reproduces
the recorded place (b) / P's place (a) within 1e-6 m.

    project venv: v_build_dev.py SHARD NSHARD
"""
from __future__ import annotations

import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1/i01_repairs')
import v_common as V  # noqa: E402
import research_harness as H  # noqa: E402
import repairs  # noqa: E402
from sgi import solver  # noqa: E402
from sgi.evaluate import match_target  # noqa: E402
from sgi.geometry import to_world_xy  # noqa: E402
from sgi.stages.goal import standard_goal  # noqa: E402
from sgi.types import Constraint, DetectedPerson, GroundedTarget  # noqa: E402

OUT = Path(V.VER) / 'data'
OLD = [('s19', 'qwen38_low', 'qwen'), ('s19', 'glm53', 'glm'), ('s20', 'dsv41', 'deepseek')]


def record_list():
    out = []
    for stage, model, short in OLD:
        for arm in ('real', 'staffed', 'plain'):
            (ref,) = H.runs(stage=stage, arm=arm, model=model, method='con')
            out.append((f'old_{short}', ref, False))
    for stage, model, short in OLD:
        for arm in ('dev_seed0', 'dev_seed1', 'dev_seed2'):
            (ref,) = H.runs(stage=stage, arm=arm, model=model, method='con')
            out.append((f'watch_{short}', ref, True))
    recs = []
    for cell, ref, watch in out:
        for rec in H.records(ref):
            if watch and rec.situation != 'audience':
                continue
            if not rec.solver_step:
                continue
            recs.append((cell, rec))
    return recs


class _Stub:
    score = types.SimpleNamespace(success=False, target_correct=False)
    moved_m = 0.0
    reproduced = True
    uses = []


def p_arm_inputs(rec):
    calls = []
    real = repairs.H.resolve

    def fake(r, **kw):
        calls.append(kw)
        return _Stub()
    repairs.H.resolve = fake
    try:
        repairs.solver_row(rec, {})
    finally:
        repairs.H.resolve = real
    kw = calls[7]
    assert set(kw) == {'people', 'constraints'}, kw.keys()
    assert calls[8]['people'] == kw['people']                     # Ps: same people, constraints as written
    return kw['people'], kw['constraints']


def one_setting(obs, target, people, cons, standard, th, fr, face, regions):
    cells, st = solver._over_floor(obs, target, people, cons, standard, th)
    fl = solver.floor(obs, people, th)
    assert len(cells) == int(fl.selectable.sum())
    ras = V.make_raster(obs, people, fl, fr, face)
    Fw = V.floor_cells_world(cells, obs.robot)
    ex = np.asarray(st.keys['excluded_m'], float)
    pick0 = to_world_xy(obs.robot, *cells[st.picks[0]])
    rho = float(V.union_hold(regions, Fw[:, 0], Fw[:, 1]).mean()) if regions else 0.0
    return dict(raster=ras, F=Fw, ex=ex, place_ex=float(ex[st.picks[0]]), pick0=pick0, rho=rho)


def main(shard, nshard):
    recs = record_list()
    assert len(recs) == 549, len(recs)
    mine = [(i, c, r) for i, (c, r) in enumerate(recs) if i % nshard == shard]
    col = {k: [] for k in ('idx', 'keys', 'raster_b', 'raster_a', 'F_b', 'F_a', 'ex_b', 'ex_a', 'place_ex_b', 'place_ex_a',
                           'places_b', 'places_a', 'frame', 'robot_radius')}
    meta = []
    t0 = time.monotonic()
    for n, (i, cell, rec) in enumerate(mine):
        key = f"{rec.ref.folder}/{rec.scene_id}/{rec.episode_id}/{rec.trial}"
        th = H.recorded_config(rec.ref)[1].thresholds
        sc = H.scene(rec.ref, rec.scene_id)
        obs = sc.observation
        gt = sc.truths[rec.episode_id].gt
        target = GroundedTarget.model_validate(rec.target)
        standard = standard_goal(obs, target, th)
        fr = V.RFrame(target.world[:2], (obs.robot.x, obs.robot.y))
        face = solver._seen_face(obs, target, th)
        matched = match_target(target.world, sc.truth, gt, th)
        regions = [] if matched is None else [g for g in (gt.acceptable_regions or []) if g.target_id in (None, matched)]
        C_rec = bool(rec.score.get('success'))
        C_now = bool(H.score_pose(rec, rec.pose_world).success)
        # (b)
        pb = [DetectedPerson.model_validate(p) for p in rec.people]
        cb = [Constraint.model_validate(u['constraint']) for u in rec.constraints]
        b = one_setting(obs, target, pb, cb, standard, th, fr, face, regions)
        mb = math.dist(b['pick0'], rec.pose_world[:2])
        assert mb <= 1e-6, (key, mb)
        places_b = [list(rec.pose_world)] + [list(c['pose_world']) for c in rec.candidates]
        # (a)
        pp, cs = p_arm_inputs(rec)
        resP = H.resolve(rec, people=pp, constraints=cs)
        pa = [DetectedPerson.model_validate(p) for p in pp]
        ca = [Constraint.model_validate(c) for c in cs]
        a = one_setting(obs, target, pa, ca, standard, th, fr, face, regions)
        ma = math.dist(a['pick0'], resP.pose_world[:2])
        assert ma <= 1e-6, (key, ma)
        places_a = [list(resP.pose_world)] + [list(c.pose_world) for c in resP.candidates]
        assert len(places_b) == 5 and len(places_a) == 5, key
        col['idx'].append(i)
        col['keys'].append(key)
        for s, d in (('b', b), ('a', a)):
            col[f'raster_{s}'].append(d['raster'])
            col[f'F_{s}'].append(d['F'])
            col[f'ex_{s}'].append(d['ex'])
            col[f'place_ex_{s}'].append(d['place_ex'])
        col['places_b'].append(places_b)
        col['places_a'].append(places_a)
        col['frame'].append([fr.o[0], fr.o[1], fr.yaw])
        col['robot_radius'].append(th.robot_radius_m)
        meta.append(dict(idx=i, key=key, cell=cell, model=rec.model, arm=rec.arm, sit=rec.situation, scene=rec.scene_id,
                         matched=matched, C_rec=C_rec, C_now=C_now, P_top1=bool(resP.score.success),
                         rho_b=b['rho'], rho_a=a['rho'], nF_b=len(b['F']), nF_a=len(a['F']), face_none=face is None,
                         n_people_b=len(pb), n_people_a=len(pa), moved_b=mb, moved_a=ma))
        if n % 25 == 0:
            print(f'{time.monotonic() - t0:7.1f} s  shard {shard}: {n + 1}/{len(mine)} {key}', flush=True)
    offb = np.concatenate([[0], np.cumsum([len(f) for f in col['F_b']])])
    offa = np.concatenate([[0], np.cumsum([len(f) for f in col['F_a']])])
    np.savez(OUT / f'vdev_{shard}of{nshard}.npz', idx=np.array(col['idx']), keys=np.array(col['keys']),
             raster_b=np.stack(col['raster_b']), raster_a=np.stack(col['raster_a']),
             F_b=np.concatenate(col['F_b']), ex_b=np.concatenate(col['ex_b']), off_b=offb,
             F_a=np.concatenate(col['F_a']), ex_a=np.concatenate(col['ex_a']), off_a=offa,
             place_ex_b=np.array(col['place_ex_b']), place_ex_a=np.array(col['place_ex_a']),
             places_b=np.array(col['places_b'], float), places_a=np.array(col['places_a'], float),
             frame=np.array(col['frame']), robot_radius=np.array(col['robot_radius']))
    (OUT / f'vdev_{shard}of{nshard}.json').write_text(json.dumps(meta))
    print(f'done shard {shard}: {len(mine)} records, {time.monotonic() - t0:.1f} s', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
