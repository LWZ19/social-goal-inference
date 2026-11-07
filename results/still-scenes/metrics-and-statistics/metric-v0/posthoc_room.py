"""I14 v0, POST-HOC sensitivity (not pre-registered): R5 and AR drawn over the whole room (all F cells) instead of the
3 m disc, in success terms (face-target heading, 45 deg), for the six primary cells (old-wording realistic and
watching). Exact hypergeometric expectations, no draws. The replica of the scorer is i14.py's, checked here again
against H.score_poses on 100 uniform F cells and up to 50 in-region F cells per record.

Why: the frozen 3 m disc leaves out 90% of the watching region and 16% of the realistic one (tables.txt T1).

    cd WT && OMP_NUM_THREADS=2 ... nice -n 10 .venv/bin/python <this>
"""
from __future__ import annotations

import collections
import json
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1/i14_v0')
import i14  # noqa: E402
from i14 import H, exact_none, face_target, key, replica, solver, DetectedPerson  # noqa: E402

HERE = i14.HERE


def one(label, rec, th, mism):
    sc = H.scene(rec.ref, rec.scene_id)
    obs, robot = sc.observation, sc.observation.robot
    _, gt, _, cache = H._against(rec)
    top = tuple(rec.pose_world)
    s_top = H.score_pose(rec, top)
    grid, joined = cache['reach']
    matched = s_top.matched_target_id
    fl = solver.floor(obs, [DetectedPerson.model_validate(p) for p in rec.people], th)
    cen = fl.centres()[fl.selectable]
    tx, ty = (float(v) for v in rec.target['world'][:2])
    hd = face_target(robot, cen, (tx, ty), th)
    rep = replica(robot, gt, matched, grid, joined, cen, hd)
    succ = rep['success']
    rng = np.random.default_rng([11, zlib.crc32(key(rec).encode())])
    samp = set(rng.choice(len(cen), size=min(100, len(cen)), replace=False).tolist())
    inr = np.flatnonzero(rep['inreg'])
    if len(inr):
        samp |= set(rng.choice(inr, size=min(50, len(inr)), replace=False).tolist())
    samp = sorted(samp)
    for i, s in zip(samp, H.score_poses(rec, [(float(cen[i, 0]), float(cen[i, 1]), float(hd[i])) for i in samp])):
        mism['checked'] += 1
        if bool(succ[i]) != s.success or bool(rep['inreg'][i]) != s.in_region:
            mism['disagree'] += 1
    cell = fl.index(top[0], top[1])
    flat_sel = np.flatnonzero(fl.selectable.ravel())
    pos = np.flatnonzero(flat_sel == cell[0] * fl.selectable.shape[1] + cell[1])
    S = int(succ.sum())
    s_pool = S - (int(succ[pos[0]]) if len(pos) else 0)
    n = len(cen)
    hit1 = bool(s_top.success)
    hit5 = bool(rec.score.get('success')) or any(s.success for s in H.score_poses(rec, [tuple(c['pose_world']) for c in rec.candidates]))
    return dict(set=label, model=rec.model, arm=rec.arm, scene=rec.scene_id, sit=rec.situation, n_sel=n, s_room=S,
                sigma_room=S / n if n else 0.0, hit1=hit1, hit5=hit5,
                r5_room=1 - exact_none(n, S, 5), ar_room=1.0 if hit1 else 1 - exact_none(n - 1, s_pool, 4))


def main():
    started = time.monotonic()
    mism = collections.Counter()
    rows = []
    for label, ref in i14.plan('oldwatch'):
        th = H.recorded_config(ref)[1].thresholds
        for rec in H.records(ref):
            if label == 'watch' and rec.situation != 'audience':
                continue
            if not rec.solver_step:
                continue
            rows.append(one(label, rec, th, mism))
        print(f'{time.monotonic() - started:7.1f} s {label} {ref.folder}', flush=True)
    with open(f'{HERE}/rows_posthoc_room.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    out = ['I14 v0 POST-HOC sensitivity (NOT pre-registered): R5 and AR over the whole room (all F cells), success at 45 deg,',
           'face-target heading for the drawn cells; exact hypergeometric expectations. Primary cells only.',
           f'replica vs package on sampled F cells: {mism["checked"]} checked, {mism["disagree"]} disagree.', '',
           f'{"cell":20s} {"n":>4s} {"top1":>5s} {"any5":>5s} {"gain":>5s} | {"R5 room":>7s} {"AR room":>7s} {"AR recov":>8s} | {"sigma_room":>10s} {"Lift_room(succ)":>15s}']
    by = collections.defaultdict(list)
    for r in rows:
        by[(r['set'], r['model'])].append(r)
    for (s, m), rs in by.items():
        t1 = sum(r['hit1'] for r in rs)
        a5 = sum(r['hit5'] for r in rs)
        r5 = sum(r['r5_room'] for r in rs)
        ar = sum(r['ar_room'] for r in rs)
        sg = sum(r['sigma_room'] for r in rs)
        g = a5 - t1
        rec_ = f'{100 * (ar - t1) / g:7.0f}%' if g else '     nan'
        out.append(f'{s + " " + m:20s} {len(rs):4d} {t1:5d} {a5:5d} {g:+5d} | {r5:7.1f} {ar:7.1f} {rec_:>8s} | {sg / len(rs):10.4f} {t1 / sg:15.2f}')
    out.append('')
    out.append('recov = (AR room any-of-5 - top-1) / (solver any-of-5 - top-1). Compare tables.txt T4 (disc).')
    text = '\n'.join(out) + '\n'
    with open(f'{HERE}/posthoc_room.txt', 'w') as f:
        f.write(text)
    print(text)
    print(f'{time.monotonic() - started:.1f} s')


if __name__ == '__main__':
    main()
