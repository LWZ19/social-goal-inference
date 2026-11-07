"""Separate-code verification of the key primary numbers (FROZEN.md section 10 step 5; DEVIATIONS.md D5: same agent,
so this checks the implementation, not the reading of the note). Shares no code with i11lib.py or lifting.py: its
own per-pixel lift (sgi.geometry.lift_all), its own breadth-first components, its own structure test, its own body
masks from the ground truth, its own windows, its own replay calls. Reads runs and scenes only through the harness.

Recomputes on the 9 primary runs: the error-class counts, oracle M1/M2/M3, V1 M1/M2/M5, T1 catches and false
triggers; then compares with summary.json (report.py) and writes verify_out.txt."""
import collections
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402
from sgi.evaluate import match_target  # noqa: E402
from sgi.geometry import lift_all, pixel_to_point  # noqa: E402
from sgi.types import GroundedTarget, Point  # noqa: E402

FLOOR, GAP, MINPX, LS, RAD = 0.15, 0.10, 25, 4.6, 15.0


def world_image(depth, cam):
    pts = lift_all(depth, cam)                       # (N, 3), row-major over valid pixels
    ok = np.isfinite(depth) & (depth > 0)
    assert ok.all(), 'the verifier assumes no invalid depth'
    return pts.reshape(depth.shape[0], depth.shape[1], 3)


def bfs_components(Wimg):
    """Labels (component index from 0, -1 floor) by breadth-first search over 4-neighbours within GAP in 3D."""
    h, w, _ = Wimg.shape
    nonfloor = Wimg[..., 2] >= FLOOR
    lab = np.full((h, w), -1, dtype=np.int64)
    right = np.zeros((h, w), bool)
    down = np.zeros((h, w), bool)
    right[:, :-1] = (np.sqrt(((Wimg[:, 1:] - Wimg[:, :-1]) ** 2).sum(-1)) <= GAP) & nonfloor[:, 1:] & nonfloor[:, :-1]
    down[:-1, :] = (np.sqrt(((Wimg[1:, :] - Wimg[:-1, :]) ** 2).sum(-1)) <= GAP) & nonfloor[1:, :] & nonfloor[:-1, :]
    right_l, down_l, nf = right.tolist(), down.tolist(), nonfloor.tolist()
    lab_l = lab.tolist()
    n = 0
    for i in range(h):
        for j in range(w):
            if not nf[i][j] or lab_l[i][j] >= 0:
                continue
            lab_l[i][j] = n
            dq = deque([(i, j)])
            while dq:
                a, b = dq.popleft()
                if b + 1 < w and right_l[a][b] and lab_l[a][b + 1] < 0:
                    lab_l[a][b + 1] = n
                    dq.append((a, b + 1))
                if b > 0 and right_l[a][b - 1] and lab_l[a][b - 1] < 0:
                    lab_l[a][b - 1] = n
                    dq.append((a, b - 1))
                if a + 1 < h and down_l[a][b] and lab_l[a + 1][b] < 0:
                    lab_l[a + 1][b] = n
                    dq.append((a + 1, b))
                if a > 0 and down_l[a - 1][b] and lab_l[a - 1][b] < 0:
                    lab_l[a - 1][b] = n
                    dq.append((a - 1, b))
            n += 1
    lab = np.array(lab_l, dtype=np.int64)
    kind = {}
    sizes = np.bincount(lab[lab >= 0], minlength=n)
    for c in range(n):
        if sizes[c] < MINPX:
            kind[c] = 'frag'
            continue
        xy = Wimg[..., :2][lab == c]
        xy = xy - xy.mean(0)
        _, _, vt = np.linalg.svd(xy, full_matrices=False)
        pr = xy @ vt.T
        ext = max(np.percentile(pr[:, 0], 98) - np.percentile(pr[:, 0], 2), np.percentile(pr[:, 1], 98) - np.percentile(pr[:, 1], 2))
        kind[c] = 'struct' if ext > LS else 'obj'
    return lab, kind


def cls_of(lab, kind, i, j):
    c = lab[i, j]
    return 'floor' if c < 0 else kind[c]


def win(i, j, h, w):
    return range(max(i - 2, 0), min(i + 3, h)), range(max(j - 2, 0), min(j + 3, w))


def surface(Z, u, v):
    h, w = Z.shape
    i, j = int(math.floor(v)), int(math.floor(u))
    rr, cc = win(i, j, h, w)
    vals = [(Z[a, b], a, b) for a in rr for b in cc]
    med = float(np.median([x[0] for x in vals]))
    best = min(vals, key=lambda t: (abs(float(t[0]) - med), math.hypot(t[2] + 0.5 - u, t[1] + 0.5 - v), t[1], t[2]))
    return best[1], best[2]


def nearest(mask, u, v):
    ii, jj = np.nonzero(mask)
    if not len(ii):
        return None, math.inf
    d = np.hypot(jj + 0.5 - u, ii + 0.5 - v)
    k = int(np.argmin(d))                             # np.nonzero is row-major: ties to row-major
    return (int(ii[k]), int(jj[k])), float(d[k])


def lift_px(cam, i, j, Zv):
    u, v = j + 0.5, i + 0.5
    pc = np.array([(u - cam.cx) * Zv / cam.fx, (v - cam.cy) * Zv / cam.fy, Zv, 1.0])
    wpt = cam.T_world_camera @ pc
    x, y = pixel_to_point(u, v, cam.width, cam.height)
    return GroundedTarget(point=Point(x=x, y=y), pixel=(u, v), world=(float(wpt[0]), float(wpt[1]), float(wpt[2])))


def seg_dist(X, Y, poly):
    v = np.asarray(poly, float)
    best = np.full(X.shape, np.inf)
    inside = np.zeros(X.shape, bool)
    for k in range(len(v)):
        a, b = v[k - 1], v[k]
        ab = b - a
        t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / max(ab @ ab, 1e-12), 0, 1)
        best = np.minimum(best, np.hypot(X - a[0] - t * ab[0], Y - a[1] - t * ab[1]))
        crosses = (a[1] > Y) != (b[1] > Y)
        with np.errstate(divide='ignore', invalid='ignore'):
            xi = a[0] + (Y - a[1]) * ab[0] / ab[1]
        inside ^= crosses & (X < xi)
    return np.where(inside | (best <= 1e-9), 0.0, best)


def body_mask(Wimg, truth, tid, th):
    t = truth.targets[tid]
    X, Y, Zw = Wimg[..., 0], Wimg[..., 1], Wimg[..., 2]
    if t.footprint is not None:
        return (seg_dist(X, Y, t.footprint) <= 0.05) & (Zw > 0.05)
    r = th.default_match_radius_m if t.match_radius is None else t.match_radius
    if t.kind != 'person':
        return (np.hypot(X - t.anchor[0], Y - t.anchor[1]) <= r) & (Zw > 0.05)
    ds = np.stack([np.hypot(X - p.x, Y - p.y) for p in truth.people], -1)
    k = [p.id for p in truth.people].index(t.person_id)
    return (ds[..., k] <= 0.30) & (Zw > 0.05) & (Zw < 2.0) & (np.argmin(ds, -1) == k)


def replay(rec, nt):
    if rec.solver_step:
        return bool(H.resolve(rec, target=nt).score.success)
    if rec.result.get('source') == 'standard':
        from sgi.stages.goal import standard_goal
        sc = H.scene(rec.ref, rec.scene_id)
        pw = standard_goal(sc.observation, nt, H.recorded_config(rec.ref)[1].thresholds).pose_world
        return pw is not None and bool(H.score_pose(rec, pw, target=nt).success)
    if rec.pose_world is not None:
        return bool(H.score_pose(rec, rec.pose_world, target=nt).success)
    return False


def main():
    out = []
    say = lambda s: (print(s, flush=True), out.append(s))  # noqa: E731
    runs = []
    for stage, model in (('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41')):
        for arm in ('real', 'staffed', 'plain'):
            r = H.runs(stage=stage, arm=arm, model=model, method='con')
            assert len(r) == 1
            runs.append(r[0])
    recs = [rec for run in runs for rec in H.records(run)]
    recs.sort(key=lambda r: (r.scene_id, r.run))
    cache = {}
    rows = []
    for rec in recs:
        if rec.target is None:
            continue
        key = rec.scene_id
        sc = H.scene(rec.ref, rec.scene_id)
        th = H.recorded_config(rec.ref)[1].thresholds
        if key not in cache:
            cache.clear()
            Wimg = world_image(sc.observation.depth, sc.observation.camera)
            lab, kind = bfs_components(Wimg)
            cache[key] = (Wimg, lab, kind, {})
        Wimg, lab, kind, bodies = cache[key]
        Z = sc.observation.depth.astype(np.float64)
        cam = sc.observation.camera
        h, w = Z.shape
        gt = sc.truths[rec.episode_id].gt
        tid = gt.target_ids[0]
        if tid not in bodies:
            bodies[tid] = body_mask(Wimg, sc.truth, tid, th)
        body = bodies[tid]
        u, v = rec.target['pixel']
        wr = rec.target['world']
        ok = match_target(wr, sc.truth, gt, th) is not None
        t = sc.truth.targets[tid]
        kindname = 'parcel' if t.kind == 'object' else t.kind
        row = dict(model=rec.model, kind=kindname, right=ok, rec_success=bool(rec.score['success']))
        # class
        q, d = nearest(body, u, v)
        i, j = int(math.floor(v)), int(math.floor(u))
        rr, cc = win(i, j, h, w)
        zpt = float(np.median([Z[a, b] for a in rr for b in cc]))
        if not ok:
            ii, jj = np.nonzero(body)
            dd = np.hypot(jj + 0.5 - u, ii + 0.5 - v)
            sel = dd <= RAD
            zt = float(np.median(Z[ii[sel], jj[sel]])) if sel.any() else float(Z[q])
            if d > RAD:
                c = 'choice'
            elif zpt < zt - 0.25:
                c = 'occluder'
            elif zpt > zt + 0.25:
                c = 'background'
            elif wr[2] < 0.05:
                c = 'floor slip'
            else:
                c = 'near'
            row['cls'] = c
        # oracle
        if q is not None and d <= RAD:
            r2, c2 = win(q[0], q[1], h, w)
            zs = [Z[a, b] for a in r2 for b in c2 if body[a, b]]
            nt = lift_px(cam, q[0], q[1], float(np.median(zs)))
            row['orc_right'] = match_target(nt.world, sc.truth, gt, th) is not None
            row['orc_succ'] = replay(rec, nt)
        else:
            row['orc_right'] = ok
            row['orc_succ'] = row['rec_success']
        # surface, V1, T1
        s = surface(Z, u, v)
        scls = cls_of(lab, kind, *s)
        labels = set()
        for a in rr:
            for b in cc:
                k = cls_of(lab, kind, a, b)
                if k == 'floor':
                    labels.add('F')
                elif k in ('struct', 'obj'):
                    labels.add(int(lab[a, b]))
        row['T1'] = scls in ('floor', 'struct') or len(labels) >= 2
        row['V1_right'] = ok
        row['V1_moved'] = False
        if scls in ('floor', 'struct'):
            objmask = np.zeros((h, w), bool)
            objs = [c for c, k in kind.items() if k == 'obj']
            objmask = np.isin(lab, objs)
            q2, d2 = nearest(objmask, u, v)
            if q2 is not None and d2 <= RAD:
                cmp_ = lab[q2]
                r3, c3 = win(q2[0], q2[1], h, w)
                zs = [Z[a, b] for a in r3 for b in c3 if lab[a, b] == cmp_]
                nt = lift_px(cam, q2[0], q2[1], float(np.median(zs)))
                row['V1_moved'] = True
                row['V1_right'] = match_target(nt.world, sc.truth, gt, th) is not None
        rows.append(row)
    GDm = ('glm53', 'dsv41')
    w46 = [r for r in rows if not r['right'] and r['model'] in GDm]
    r454 = [r for r in rows if r['right'] and r['model'] in GDm]
    w53 = [r for r in rows if not r['right']]
    r699 = [r for r in rows if r['right']]
    cls46 = collections.Counter(r['cls'] for r in w46)
    lift46 = cls46['occluder'] + cls46['background'] + cls46['floor slip']
    of46 = sum(r['orc_right'] for r in w46)
    ob454 = sum(not r['orc_right'] for r in r454)
    os46 = sum(r['orc_succ'] for r in w46 if r['orc_right'])
    v1f = sum(r['V1_right'] for r in w46)
    v1b = sum(not r['V1_right'] for r in r454)
    p15 = [r for r in w53 if r['kind'] == 'person']
    d1 = [r for r in r699 if r['kind'] == 'person']
    mine = dict(n46=len(w46), n454=len(r454), n53=len(w53), n699=len(r699), lift46=lift46,
                choice46=cls46['choice'], cls53=dict(collections.Counter(r['cls'] for r in w53)),
                orc_fix46=of46, orc_brk454=ob454, orc_brk699=sum(not r['orc_right'] for r in r699),
                orc_fix53=sum(r['orc_right'] for r in w53), orc_succ46=os46,
                v1_fix46=v1f, v1_brk454=v1b, v1_share=(v1f - v1b) / (of46 - ob454) if of46 > ob454 else None,
                t1_catch=sum(r['T1'] for r in p15), n15=len(p15), t1_d1=sum(r['T1'] for r in d1), nd1=len(d1),
                t1_d2=sum(r['T1'] for r in r699))
    say('verify_indep.py: separate-code recomputation on the 9 primary runs (same agent; DEVIATIONS.md D5)')
    for k, val in mine.items():
        say(f'  {k}: {val}')
    S = json.loads((HERE / 'summary.json').read_text())
    res, pr = S['res'], S['primary']
    theirs = dict(lift46=pr['nlift46'], choice46=pr['nchoice46'], orc_fix46=res['oracle']['f46'],
                  orc_brk454=res['oracle']['b454'], orc_brk699=res['oracle']['b699'], orc_fix53=res['oracle']['f53'],
                  orc_succ46=res['oracle']['s46'], v1_fix46=res['V1']['f46'], v1_brk454=res['V1']['b454'],
                  v1_share=res['V1']['share'], t1_catch=pr['t1c'], n15=pr['n15'], t1_d1=pr['t1fp'], nd1=pr['nd1'],
                  t1_d2=pr['d2'])
    say('comparison with run_eval.py / report.py (summary.json):')
    bad = 0
    for k, tv in theirs.items():
        mv = mine[k]
        same = (abs(mv - tv) < 1e-9) if isinstance(tv, float) and mv is not None else mv == tv
        bad += not same
        say(f"  {k:12s} verifier {mv!s:>8s}  main {tv!s:>8s}  {'AGREE' if same else 'DISAGREE'}")
    say(f'  {"ALL AGREE" if not bad else f"{bad} DISAGREE"}')
    (HERE / 'verify_out.txt').write_text('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
