"""Independent verifier: per-record computation for I11 Stage 0, from FROZEN.md only.

usage: vcompute.py primary|xnew|xph|xph2
Writes vrows_<set>.jsonl in this folder. Reads runs and scenes only through research_harness.
"""
from __future__ import annotations

import collections
import json
import math
import sys
import time

import numpy as np

from vlib import (H, OUT, SNAP_R, FRONT_MARGIN, CLS_NAME, CLS_FLOOR, CLS_STRUCT, CLS_OBJECT, FLOOR_LABEL,
                  SceneGeo, check_pins, check_th, nearest_in, window_median, surface_label, window_labels,
                  comp_lift, nearest_object_component, make_target, front_snap, lift_pixel_at, win_slice,
                  match_target)
from sgi.stages.goal import standard_goal

REAL = ('real', 'staffed', 'plain')
PRIMARY_FOLDERS = {
    '20261011-160334_s19_real_qwen38_low_con', '20261011-160344_s19_staffed_qwen38_low_con',
    '20261011-160527_s19_plain_qwen38_low_con', '20261011-160500_s19_real_glm53_con',
    '20261011-160352_s19_staffed_glm53_con', '20261011-160452_s19_plain_glm53_con',
    '20261013-191337_s20_real_dsv41_con', '20261013-191345_s20_staffed_dsv41_con',
    '20261013-191402_s20_plain_dsv41_con'}


def one(stage, arm, model, method):
    rs = H.runs(stage=stage, arm=arm, model=model, method=method)
    assert len(rs) == 1, (stage, arm, model, method, rs)
    return rs[0]


def select(setname):
    if setname == 'primary':
        runs = [one(s, a, m, 'con') for s, m in (('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41'))
                for a in REAL]
        assert {r.folder for r in runs} == PRIMARY_FOLDERS
        return runs
    if setname == 'xnew':
        return [one(s, a, m, 'con') for s, m in (('s21', 'qwen38_low'), ('s23', 'glm53'), ('s23', 'dsv41'),
                                                  ('s23', 'haiku55'), ('s23', 'sonnet55_low'),
                                                  ('s23', 'gemini38_low')) for a in REAL]
    if setname == 'xph':
        return [one(s, a, m, 'con') for s, m in (('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41'))
                for a in ('ph', 'dev_seed0', 'dev_seed1', 'dev_seed2')]
    if setname == 'xph2':
        return (H.runs(stage='s10', method='con') + H.runs(stage='s8', method='convlm')
                + H.runs(stage='s8', method='conorc'))
    raise ValueError(setname)


_GEO = collections.OrderedDict()


def geo_for(rec, th):
    key = (str(rec.ref.scenes_dir), rec.scene_id, th.model_dump_json())
    if key not in _GEO:
        _GEO[key] = SceneGeo(H.scene(rec.ref, rec.scene_id), th)
        while len(_GEO) > 4:
            _GEO.popitem(last=False)
    _GEO.move_to_end(key)
    return _GEO[key]


_REPLAY = {}


def replay(rec, geo, th, nt):
    key = (rec.run, rec.scene_id, rec.episode_id, rec.trial, tuple(nt.pixel), tuple(nt.world))
    if key in _REPLAY:
        return _REPLAY[key]
    if rec.solver_step:
        ok, path = bool(H.resolve(rec, target=nt).score.success), 'solver'
    elif rec.result.get('source') == 'standard':
        sg = standard_goal(geo.sc.observation, nt, th)
        if sg.pose_world is None:
            ok, path = False, 'standard (no pose)'
        else:
            ok, path = bool(H.score_pose(rec, sg.pose_world, target=nt).success), 'standard'
    elif rec.pose_world is not None:
        ok, path = bool(H.score_pose(rec, rec.pose_world, target=nt).success), 'recorded pose'
    else:
        ok, path = False, 'no pose'
    _REPLAY[key] = (ok, path)
    return ok, path


def is_right(world, truth, gt, th):
    return match_target(world, truth, gt, th) is not None


def process(rec, setname, rng):
    cfg = H.recorded_config(rec.ref)[1]
    th = cfg.thresholds
    check_th(th)
    sc = H.scene(rec.ref, rec.scene_id)
    gt = sc.truths[rec.episode_id].gt
    row = dict(set=setname, run=rec.run, stage=rec.ref.stage, arm=rec.arm, model=rec.model, method=rec.method,
               scene=rec.scene_id, episode=rec.episode_id, trial=rec.trial, situation=rec.situation,
               ntargets=len(gt.target_ids), rec_success=bool(rec.score.get('success')),
               rec_tc=bool(rec.score.get('target_correct')), source=rec.result.get('source'),
               solver_step=bool(rec.solver_step), has_point=rec.target is not None)
    layout = rec.scene_id
    for a in ('real_', 'staffed_', 'plain_'):
        if layout.startswith(a):
            layout = layout[len(a):]
    row['layout'] = '_'.join(layout.split('_')[:2])
    if rec.target is None:
        # still record the true-target kind for denominators
        t = sc.truth.targets[gt.target_ids[0]]
        row['kind'] = 'parcel' if t.kind == 'object' else t.kind
        return row
    geo = geo_for(rec, th)
    truth = sc.truth
    u, v = (float(x) for x in rec.target['pixel'])
    row['u'], row['v'] = u, v
    world_rec = np.asarray(rec.target['world'], dtype=np.float64)
    again = H.target_at(rec, pixel=(u, v))
    row['lift_err'] = None if again is None else float(np.max(np.abs(np.asarray(again.world) - world_rec)))
    right = is_right(world_rec, truth, gt, th)
    row['right'] = right
    row['tc_disagree'] = right != row['rec_tc']
    # true target: the one whose body mask is nearest the point
    dists = []
    for tid in gt.target_ids:
        q = nearest_in(geo.body(tid), u, v)
        dists.append((math.inf if q is None else q[2], tid))
    dists.sort(key=lambda x: x[0])
    tid = dists[0][1] if dists[0][0] < math.inf else gt.target_ids[0]
    t = truth.targets[tid]
    row['tid'] = tid
    row['kind'] = 'parcel' if t.kind == 'object' else t.kind
    body = geo.body(tid)
    union = np.zeros_like(body)
    for t2 in gt.target_ids:
        union |= geo.body(t2)
    # vectorised match mask vs match_target on sampled pixels (wrong-target records)
    if not right:
        m = geo.match(tid)
        iv = rng.integers(0, geo.h, 400)
        iu = rng.integers(0, geo.w, 400)
        on = np.argwhere(m)
        if len(on):
            pick = on[rng.integers(0, len(on), min(200, len(on)))]
            iv = np.concatenate([iv, pick[:, 0]])
            iu = np.concatenate([iu, pick[:, 1]])
        bad = 0
        if len(gt.target_ids) == 1:
            for a, b in zip(iv, iu):
                got = match_target(geo.W[a, b], truth, gt, th) == tid
                bad += int(got != bool(m[a, b]))
        row['vec_bad'], row['vec_n'] = int(bad), int(len(iv))
    # error class
    i, j = math.floor(v), math.floor(u)
    Zpt = window_median(geo.Z, geo.valid, i, j)
    qb = nearest_in(body, u, v)
    row['d_body'] = None if qb is None else qb[2]
    if qb is not None:
        ii, jj = np.nonzero(body)
        dd = np.hypot(jj + 0.5 - u, ii + 0.5 - v)
        near = dd <= SNAP_R
        Zt = float(np.median(geo.Z[ii[near], jj[near]])) if near.any() else float(geo.Z[qb[0], qb[1]])
    else:
        Zt = math.nan
    row['Zpt'], row['Zt'] = Zpt, Zt
    if not right:
        if qb is None or qb[2] > SNAP_R:
            lc = 'choice'
        elif Zpt < Zt - FRONT_MARGIN:
            lc = 'occluder'
        elif Zpt > Zt + FRONT_MARGIN:
            lc = 'background'
        elif world_rec[2] < 0.05:
            lc = 'floor slip'
        else:
            lc = 'near-depth'
        row['lclass'] = lc
    # surfaces and triggers
    for g in ('G1', 'G2'):
        lab_s, cls_s, _ = surface_label(geo, g, u, v)
        labs = window_labels(geo, g, u, v)
        on_env = cls_s in (CLS_FLOOR, CLS_STRUCT)
        t1 = on_env or len(labs) >= 2
        other = None
        if cls_s == CLS_OBJECT:
            other = nearest_object_component(geo, g, u, v, exclude=lab_s)
        t2 = t1 or (cls_s == CLS_OBJECT and other is not None)
        row[f's_{g}'] = CLS_NAME[cls_s]
        row[f'nlab_{g}'] = len(labs)
        row[f'T1_{g}'] = bool(t1)
        row[f'T2_{g}'] = bool(t2)
    # fixes
    fx = {}

    def finish(name, moved, nt):
        if not moved:
            fx[name] = dict(moved=False, right_after=right, succ_after=row['rec_success'])
            return
        ra = is_right(nt.world, truth, gt, th)
        sa, path = replay(rec, geo, th, nt)
        fx[name] = dict(moved=True, right_after=bool(ra), succ_after=bool(sa), path=path,
                        pixel=list(nt.pixel), world=list(nt.world))

    def did_move(pix, world):
        return (abs(pix[0] - u) > 0 or abs(pix[1] - v) > 0 or
                float(np.max(np.abs(np.asarray(world) - world_rec))) > 0)

    # oracle
    qo = nearest_in(union, u, v, SNAP_R)
    if qo is None:
        finish('oracle', False, None)
    else:
        Zm = window_median(geo.Z, geo.valid, qo[0], qo[1], mask=union)
        uq, vq = qo[1] + 0.5, qo[0] + 0.5
        w_ = lift_pixel_at(geo.cam, uq, vq, Zm)
        nt = make_target(geo.cam, uq, vq, w_)
        finish('oracle', did_move((uq, vq), w_), nt)
    # V1, V2, V4 (V4 = V1 under G2), V3
    for name, g, always in (('V1', 'G1', False), ('V2', 'G1', True), ('V4', 'G2', False)):
        _, cls_s, _ = surface_label(geo, g, u, v)
        trig = always or cls_s in (CLS_FLOOR, CLS_STRUCT)
        res = None
        if trig:
            oc = nearest_object_component(geo, g, u, v)
            if oc is not None:
                res = comp_lift(geo, g, oc[0], u, v)
        if res is None:
            finish(name, False, None)
        else:
            (uq, vq), w_ = res
            finish(name, did_move((uq, vq), w_), make_target(geo.cam, uq, vq, w_))
    # V3
    g = 'G1'
    lab_s, cls_s, _ = surface_label(geo, g, u, v)
    res = None
    why = ''
    if cls_s in (CLS_FLOOR, CLS_STRUCT):
        oc = nearest_object_component(geo, g, u, v)
        if oc is not None:
            res = comp_lift(geo, g, oc[0], u, v)
        why = 'V1'
    elif cls_s == CLS_OBJECT:
        lab = geo.comp[g]['lab']
        cmap = geo.comp[g]['cmap']
        rs, cs = win_slice(i, j, geo.h, geo.w)
        wl = lab[rs, cs]
        wc = cmap[rs, cs]
        zw = geo.Z[rs, cs]
        vm = geo.valid[rs, cs]
        zn = float(np.median(zw[(wl == lab_s) & vm]))
        cands = []
        for c in np.unique(wl[(wc == CLS_OBJECT) & vm]):
            c = int(c)
            if c == lab_s:
                continue
            zf = float(np.median(zw[(wl == c) & vm]))
            if zf > zn:
                q = nearest_in(lab == c, u, v)
                cands.append((q[2], c))
        if cands:
            cands.sort()
            res = comp_lift(geo, g, cands[0][1], u, v)
            why = 'back-step'
    row['V3_why'] = why
    if res is None:
        finish('V3', False, None)
    else:
        (uq, vq), w_ = res
        finish('V3', did_move((uq, vq), w_), make_target(geo.cam, uq, vq, w_))
    # front snap reference
    fsr = front_snap(geo, u, v)
    if fsr is None:
        finish('front', False, None)
    else:
        (uq, vq), w_ = fsr
        finish('front', True, make_target(geo.cam, uq, vq, w_))
    row['fx'] = fx
    # baseline replay for records that lose success under oracle or V1 (does the recorded target reproduce?)
    for name in ('oracle', 'V1'):
        f = fx[name]
        if f['moved'] and row['rec_success'] and not f['succ_after']:
            from sgi.types import GroundedTarget
            nt0 = GroundedTarget.model_validate(rec.target)
            row[f'{name}_loss_baseline'] = replay(rec, geo, th, nt0)[0]
    return row


def main():
    setname = sys.argv[1]
    for line in check_pins():
        print(line, flush=True)
    runs = select(setname)
    print(f'{setname}: {len(runs)} runs', flush=True)
    rng = np.random.default_rng(12345)
    t0 = time.time()
    rows = []
    tallies = {}
    for run in runs:
        tally = collections.Counter()
        recs = list(H.records(run, tally=tally))
        tallies[run.folder] = dict(tally)
        # sort by scene to keep the scene cache warm
        recs.sort(key=lambda r: (r.scene_id, r.episode_id, r.trial if r.trial is not None else -1))
        has_pixel = all((r.target is None) or ('pixel' in r.target and r.target['pixel'] is not None) for r in recs)
        if not has_pixel:
            print(f'  {run.folder}: records lack a target pixel, skipped', flush=True)
            continue
        for rec in recs:
            rows.append(process(rec, setname, rng))
        print(f'  {run.folder}: {len(recs)} records, {time.time() - t0:.0f} s, tally {dict(tally)}', flush=True)
    with open(OUT / f'vrows_{setname}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    (OUT / f'vmeta_{setname}.json').write_text(json.dumps(dict(runs=[r.folder for r in runs], tallies=tallies,
                                                               n=len(rows), seconds=time.time() - t0), indent=1))
    print(f'done {len(rows)} rows in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
