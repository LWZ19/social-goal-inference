"""Independent verifier: M4 component recovery (FROZEN.md section 6), primary scenes, X-ph scenes, fresh seeds.

usage: vm4.py primary|xph|fresh
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
import time

import numpy as np

from vlib import H, OUT, RD, SceneGeo, check_pins, check_th, CLS_STRUCT, FLOOR_BAND
from vcompute import select


def m4_target(geo, g, tid):
    truth = geo.sc.truth
    B = geo.body(tid) & (geo.W[..., 2] >= FLOOR_BAND)
    lab = geo.comp[g]['lab']
    nB = int(B.sum())
    if nB == 0:
        return dict(nB=0)
    labs_in = lab[B]
    labs_in = labs_in[labs_in >= 0]
    if labs_in.size == 0:
        return dict(nB=nB, none=True)
    cnt = np.bincount(labs_in)
    c = int(np.argmax(cnt))            # ties: lowest id
    inter = int(cnt[c])
    cm = lab == c
    size = int(cm.sum())
    iou = inter / (size + nB - inter)
    cov = inter / nB
    pur = inter / size
    # other bodies
    t = truth.targets[tid]
    others = []
    for k, p in enumerate(geo.people):
        if t.kind == 'person' and p.id == t.person_id:
            continue
        others.append(geo.pmasks[k])
    for t2 in truth.targets:
        if t2 == tid:
            continue
        others.append(geo.body(t2))
    mo = max([int((cm & o).sum()) for o in others], default=0)
    mo_union = int((cm & np.logical_or.reduce(others)).sum()) if others else 0
    return dict(nB=nB, iou=iou, cov=cov, pur=pur, merged=pur < 0.5, split=cov < 0.5,
                merged_other=mo >= 25, merged_other_union=mo_union >= 25,
                structure=int(geo.comp[g]['ccls'][c]) == CLS_STRUCT)


def kind_of(t):
    return 'parcel' if t.kind == 'object' else t.kind


def run_set(setname):
    rows = []
    if setname in ('primary', 'xph'):
        seen = {}
        for run in select(setname):
            th = H.recorded_config(run)[1].thresholds
            check_th(th)
            for rec in H.records(run):
                k = (str(rec.ref.scenes_dir), rec.scene_id)
                if k not in seen:
                    seen[k] = (run, rec)
        for (sd, sid), (run, rec) in sorted(seen.items()):
            th = H.recorded_config(run)[1].thresholds
            sc = H.scene(run, sid)
            geo = SceneGeo(sc, th)
            tids = sorted({t for et in sc.truths.values() for t in et.gt.target_ids})
            for tid in tids:
                for g in ('G1', 'G2'):
                    r = m4_target(geo, g, tid)
                    r.update(set=setname, scenes_dir=sd, scene=sid, tid=tid, kind=kind_of(sc.truth.targets[tid]), g=g)
                    rows.append(r)
    elif setname == 'fresh':
        from sgi.config import load_config
        from sgi.scene import load_scene
        cfg = load_config()
        th = cfg.thresholds
        check_th(th)
        held = set(cfg.held_out_situations)
        base = RD / 'fresh_seeds'
        for seed in range(100, 110):
            folder = base / f'seed{seed}'
            listing = json.loads((folder / 'DONE.json').read_text())
            assert listing['seed'] == seed
            for ent in listing['scenes']:
                assert not (set(ent.get('situations') or []) & held), ent
                sf = folder / ent['split'] / ent['scene_id']
                H.guard_tree(sf)
                inputs, truth = load_scene(sf)
                assert not any(ep.situation in held for ep in truth.episodes.values())
                sc = H.Scene(inputs, inputs.observation, truth, None)
                geo = SceneGeo(sc, th)
                tids = sorted({t for ep in truth.episodes.values() for t in ep.target_ids})
                for tid in tids:
                    for g in ('G1', 'G2'):
                        r = m4_target(geo, g, tid)
                        r.update(set='fresh', scenes_dir=f'seed{seed}/{ent["split"]}', scene=ent['scene_id'], tid=tid,
                                 kind=kind_of(truth.targets[tid]), g=g)
                        rows.append(r)
    return rows


def summarise(rows, title):
    out = [title]
    for g in ('G1', 'G2'):
        for k in ('person', 'parcel', 'place'):
            rs = [r for r in rows if r['g'] == g and r['kind'] == k and 'iou' in r]
            if not rs:
                continue
            ious = [r['iou'] for r in rs]
            out.append(f"  {g} {k:7s} n={len(rs):3d} medIoU {statistics.median(ious):.3f} IoU>=.5 "
                       f"{sum(i >= .5 for i in ious)} IoU>=.7 {sum(i >= .7 for i in ious)} merged {sum(r['merged'] for r in rs)} "
                       f"merged_other {sum(r['merged_other'] for r in rs)} (union {sum(r['merged_other_union'] for r in rs)}) "
                       f"split {sum(r['split'] for r in rs)} structure {sum(r['structure'] for r in rs)} "
                       f"medPur {statistics.median(r['pur'] for r in rs):.3f}")
        nb = [r for r in rows if r['g'] == g and 'iou' not in r]
        if nb:
            out.append(f'  {g} without B or component: {len(nb)}')
    return out


if __name__ == '__main__':
    check_pins()
    s = sys.argv[1]
    t0 = time.time()
    rows = run_set(s)
    with open(OUT / f'vm4_{s}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    lines = summarise(rows, f'M4 {s}: {len({(r["scenes_dir"], r["scene"]) for r in rows})} scenes, '
                            f'{len(rows) // 2} targets, {time.time() - t0:.0f} s')
    print('\n'.join(lines))
    (OUT / f'vm4_{s}.txt').write_text('\n'.join(lines) + '\n')
