"""Independent verifier for I11 Stage 0 (lifting vs choice). Reads only through research_harness.

Masks are built from ground truth checked against depth: each pixel is lifted with its own depth at its centre
(sgi.geometry.lift_all semantics), and a pixel is on the target when its world point lies in the target's volume.
Only pixels within WIN px of the model's point are examined (enough for radii up to 30 px).
"""
import sys, math, collections, statistics as stt
import numpy as np
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
from sgi.evaluate import match_target, score as pkg_score
from sgi.polygons import point_in_polygon, distance_to_polygon, distance_to_boundary
from sgi.geometry import _depth_at
from sgi.types import GroundedTarget, Point, GoalResult
from sgi.geometry import pixel_to_point
from sgi.stages.goal import standard_goal

D = 'results/still-scenes/perception/lifting-rerun/verify/'
WIN = 32
SETS = sys.argv[1] if len(sys.argv) > 1 else 'realistic'
ARMS = {'realistic': ('real', 'staffed', 'plain'), 'ph': ('ph',)}[SETS]
SEL = [('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41')]
lines = []


def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    lines.append(s)


def unproject(cam, u, v, Z):
    pc = np.array([(u - cam.cx) * Z / cam.fx, (v - cam.cy) * Z / cam.fy, Z, 1.0])
    return (cam.T_world_camera @ pc)[:3]


def window_world(depth, cam, u, v):
    """world points (h, w, 3), depth (h, w), and pixel-centre coords for pixels within WIN of floor(u), floor(v)."""
    r0, c0 = math.floor(v), math.floor(u)
    rs, re_ = max(r0 - WIN, 0), min(r0 + WIN + 1, depth.shape[0])
    cs, ce = max(c0 - WIN, 0), min(c0 + WIN + 1, depth.shape[1])
    Z = depth[rs:re_, cs:ce].astype(np.float64)
    vv, uu = np.meshgrid(np.arange(rs, re_) + 0.5, np.arange(cs, ce) + 0.5, indexing='ij')
    valid = np.isfinite(Z) & (Z > 0)
    pc = np.stack([(uu - cam.cx) * Z / cam.fx, (vv - cam.cy) * Z / cam.fy, Z, np.ones_like(Z)], axis=-1)
    W = pc @ cam.T_world_camera.T
    return W[..., :3], Z, uu, vv, valid, (rs, cs)


def masks_for(truth, gt, th, W, valid, person_r=0.30):
    """body mask (visible surface), floor-only mask, match mask (pixel's own lift passes match_target)."""
    h, w = valid.shape
    body = np.zeros((h, w), bool)
    floor = np.zeros((h, w), bool)
    match = np.zeros((h, w), bool)
    people = truth.people
    pxy = np.array([[q.x, q.y] for q in people], dtype=float).reshape(-1, 2)
    X, Y, Zw = W[..., 0], W[..., 1], W[..., 2]
    # nearest person index per pixel
    if len(pxy):
        d2 = (X[..., None] - pxy[:, 0]) ** 2 + (Y[..., None] - pxy[:, 1]) ** 2
        nearest = np.argmin(d2, axis=-1)
    else:
        nearest = np.full(X.shape, -1)
    for tid in gt.target_ids:
        t = truth.targets[tid]
        if t.kind == 'person':
            k = [q.id for q in people].index(t.person_id)
            hd = np.hypot(X - pxy[k, 0], Y - pxy[k, 1])
            body |= valid & (hd <= person_r) & (Zw > 0.05) & (Zw < 2.0) & (nearest == k)
        elif t.footprint is not None:
            for i in range(h):
                for j in range(w):
                    if not valid[i, j]:
                        continue
                    pt = (X[i, j], Y[i, j])
                    dist = distance_to_polygon(pt, t.footprint)
                    if dist <= 0.05:
                        if Zw[i, j] > 0.05:
                            body[i, j] = True
                        else:
                            floor[i, j] = True
        else:
            rad = th.default_match_radius_m if t.match_radius is None else t.match_radius
            hd = np.hypot(X - t.anchor[0], Y - t.anchor[1])
            body |= valid & (hd <= rad) & (Zw > 0.05)
    for i in range(h):
        for j in range(w):
            if valid[i, j] and match_target((X[i, j], Y[i, j], Zw[i, j]), truth, gt, th) is not None:
                match[i, j] = True
    return body, floor, match


def snap(mask, uu, vv, u, v):
    if not mask.any():
        return None, math.inf
    d = np.hypot(uu - u, vv - v)
    d = np.where(mask, d, np.inf)
    idx = np.unravel_index(np.argmin(d), d.shape)
    return idx, float(d[idx])


def oracle_lift(cam, Zwin, mask, idx, uu, vv):
    i, j = idx
    sub_m = mask[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3]
    sub_z = Zwin[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3]
    Z = float(np.median(sub_z[sub_m]))
    return unproject(cam, uu[i, j], vv[i, j], Z)


def landed_surface(truth, wpt):
    x, y, z = wpt
    if z <= 0.05:
        surf = 'floor'
    elif not truth.people:
        surf = 'other'
    else:
        near = min(truth.people, key=lambda q: math.hypot(x - q.x, y - q.y))
        if math.hypot(x - near.x, y - near.y) <= 0.4 and z < 2.0:
            surf = 'person:' + near.id
        else:
            surf = 'other'
    dw = distance_to_boundary((x, y), truth.bounds)
    return surf, dw


def run():
    rows = []
    for st, m in SEL:
        for arm in ARMS:
            for r in H.runs(stage=st, arm=arm, model=m, method='con'):
                for rec in H.records(r):
                    if rec.score['target_correct'] or rec.target is None:
                        continue
                    rows.append(rec)
    p(SETS, 'wrong-target C records with a point:', len(rows))
    out = []
    for rec in rows:
        sc, gt, th, _ = H._against(rec)
        cam = sc.observation.camera
        depth = sc.observation.depth
        u, v = rec.target['pixel']
        # check the recorded lift reproduces
        again = H.target_at(rec, point=(rec.target['point']['x'], rec.target['point']['y']))
        lift_err = float(np.linalg.norm(np.array(again.world) - np.array(rec.target['world'])))
        W, Zwin, uu, vv, valid, _ = window_world(depth, cam, u, v)
        kinds = sorted({sc.truth.targets[t].kind for t in gt.target_ids})
        kind = '/'.join(kinds)
        res = dict(model=rec.model, arm=rec.arm, scene=rec.scene_id, ep=rec.episode_id, trial=rec.trial,
                   kind=kind, source=rec.result.get('source'), u=u, v=v, lift_err=lift_err)
        body, floor, match = masks_for(sc.truth, gt, th, W, valid)
        for name, mk in (('body', body), ('floor', floor), ('match', match)):
            idx, dpx = snap(mk, uu, vv, u, v)
            res[name + '_px'] = dpx
            ok_or = ok_pipe = ok_med15 = False
            if idx is not None and dpx <= 15:
                w_or = oracle_lift(cam, Zwin, mk, idx, uu, vv)
                ok_or = match_target(w_or, sc.truth, gt, th) is not None
                gtp = H.target_at(rec, pixel=(uu[idx], vv[idx]))
                ok_pipe = gtp is not None and match_target(gtp.world, sc.truth, gt, th) is not None
                near15 = mk & (np.hypot(uu - u, vv - v) <= 15)
                w_m15 = unproject(cam, uu[idx], vv[idx], float(np.median(Zwin[near15])))
                ok_med15 = match_target(w_m15, sc.truth, gt, th) is not None
                if name == 'body':
                    res['fixed_world'] = tuple(float(c) for c in w_or)
                    res['fixed_pixel'] = (float(uu[idx]), float(vv[idx]))
            res[name + '_ok'] = ok_or
            res[name + '_okpipe'] = ok_pipe
            res[name + '_okmed15'] = ok_med15
        # depth comparison, landed distance and surface
        Zpt = _depth_at(depth, cam, u, v, th.lift_window)
        near15 = body & (np.hypot(uu - u, vv - v) <= 15)
        Ztg = float(np.median(Zwin[near15])) if near15.any() else None
        res['Zpt'] = Zpt
        res['Ztg'] = Ztg
        if Ztg is None:
            res['cls'] = 'choice'
        elif Zpt < Ztg - 0.25:
            res['cls'] = 'occluder'
        elif Zpt > Ztg + 0.25:
            res['cls'] = 'background'
        else:
            res['cls'] = 'same-depth'
        wrec = rec.target['world']
        surf, dwall = landed_surface(sc.truth, wrec)
        res['surf'] = surf
        res['dwall'] = dwall
        res['z'] = wrec[2]
        if kind == 'person':
            tq = [q for q in sc.truth.people if q.id == sc.truth.targets[gt.target_ids[0]].person_id][0]
            res['landed_m'] = math.hypot(wrec[0] - tq.x, wrec[1] - tq.y)
            # person-radius sensitivity
            for pr in (0.25, 0.35):
                b2, _, _ = masks_for(sc.truth, gt, th, W, valid, person_r=pr)
                res[f'body{pr}_px'] = snap(b2, uu, vv, u, v)[1]
        # window stats (for W3)
        r0, c0 = math.floor(v), math.floor(u)
        # window index inside the local arrays
        rs = max(r0 - WIN, 0); cs = max(c0 - WIN, 0)
        i0, j0 = r0 - rs, c0 - cs
        wz = Zwin[max(i0 - 2, 0):i0 + 3, max(j0 - 2, 0):j0 + 3]
        wv = valid[max(i0 - 2, 0):i0 + 3, max(j0 - 2, 0):j0 + 3]
        wb = body[max(i0 - 2, 0):i0 + 3, max(j0 - 2, 0):j0 + 3]
        res['win_range'] = float(wz[wv].max() - wz[wv].min()) if wv.any() else None
        res['win_has_target'] = bool(wb.any())
        res['centre_on_target'] = bool(body[i0, j0])
        # side: point u vs mean u of body pixels within 15 px
        res['side'] = None if not near15.any() else ('left' if u < float(uu[near15].mean()) else 'right')
        res['rec'] = rec
        out.append(res)
    return out


def summarise(out):
    p('max recorded-lift reproduction error (m):', max(r['lift_err'] for r in out))
    by = collections.defaultdict(list)
    for r in out:
        by[r['kind']].append(r)
    p('\n== W1: recoverable (mask pixel within 15 px AND oracle snap+mask-median lift matches) ==')
    for name in ('body', 'floor', 'match'):
        p(name, {k: f"{sum(x[name + '_ok'] for x in v)}/{len(v)}" for k, v in sorted(by.items())},
          'pipeline-lift-at-snap', {k: sum(x[name + '_okpipe'] for x in v) for k, v in sorted(by.items())},
          'median15', {k: sum(x[name + '_okmed15'] for x in v) for k, v in sorted(by.items())})
    p('by model (body):', {(k, mm): f"{sum(x['body_ok'] for x in v if x['model'] == mm)}/{sum(1 for x in v if x['model'] == mm)}"
                           for k, v in sorted(by.items()) for mm in ('qwen38_low', 'glm53', 'dsv41')})
    p('radius sweep (body mask within r px, any lift):')
    for rr in (3, 5, 10, 15, 20, 30):
        p(' r', rr, {k: sum(x['body_px'] <= rr for x in v) for k, v in sorted(by.items())})
    p('radius sweep (body mask within r px AND oracle lift matches) is body_ok for r=15 only')
    p('classes (body, 15 px):', {k: dict(collections.Counter(x['cls'] for x in v)) for k, v in sorted(by.items())})
    p('\nplace records:')
    for x in by.get('place', []):
        p(' ', x['model'], x['scene'], x['ep'], x['trial'], 'body_px %.1f' % x['body_px'], 'floor_px %.1f' % x['floor_px'],
          'match_px %.1f' % x['match_px'], 'ok', x['body_ok'], x['floor_ok'], x['match_ok'], 'cls', x['cls'],
          'Zpt %.2f' % x['Zpt'], 'Ztg', None if x['Ztg'] is None else round(x['Ztg'], 2), 'surf', x['surf'], 'z %.2f' % x['z'])
    p('\nobject records:')
    for x in by.get('object', []):
        p(' ', x['model'], x['scene'], x['ep'], x['trial'], 'body_px %.1f' % x['body_px'], 'ok', x['body_ok'], x['body_okpipe'],
          'cls', x['cls'], 'surf', x['surf'])
    P = by.get('person', [])
    p('\n== W2: person wrong targets ==', len(P), dict(collections.Counter(x['model'] for x in P)),
      dict(collections.Counter(x['scene'] for x in P)))
    if P:
        p('within 15 px:', sum(x['body_px'] <= 15 for x in P), 'max px %.2f' % max(x['body_px'] for x in P),
          'within 3 px:', sum(x['body_px'] <= 3 for x in P))
        p('r=0.25 max px %.2f, r=0.35 max px %.2f, match mask max px %.2f' % (
            max(x['body0.25_px'] for x in P), max(x['body0.35_px'] for x in P), max(x['match_px'] for x in P)))
        p('landed m: min %.2f max %.2f' % (min(x['landed_m'] for x in P), max(x['landed_m'] for x in P)))
        for x in sorted(P, key=lambda x: x['landed_m']):
            p(' ', x['model'], x['arm'], x['scene'], x['ep'], x['trial'], 'px %.1f' % x['body_px'], 'landed %.2f' % x['landed_m'],
              'cls', x['cls'], 'surf', x['surf'], 'z %.2f' % x['z'], 'dwall %.2f' % x['dwall'],
              'Zpt %.2f Ztg %s' % (x['Zpt'], None if x['Ztg'] is None else round(x['Ztg'], 2)), 'side', x['side'], 'centre_on', x['centre_on_target'],
              'win_range %.2f' % x['win_range'], 'win_has_tgt', x['win_has_target'], 'ok', x['body_ok'], x['body_okpipe'])
    E = [x for x in out if 'eval_12' in x['scene']]
    p('\n== W3: eval_12 wrong targets ==', len(E), dict(collections.Counter(x['model'] for x in E)))
    if E:
        p('px<=3', sum(x['body_px'] <= 3 for x in E), 'px<=10', sum(x['body_px'] <= 10 for x in E), 'max %.2f' % max(x['body_px'] for x in E))
        p('side', dict(collections.Counter((x['side'], x['cls'], x['surf']) for x in E)))
        p('window range > 0.5 m:', sum((x['win_range'] or 0) > 0.5 for x in E), 'window holds target px:',
          sum(x['win_has_target'] for x in E), 'centre on target:', sum(x['centre_on_target'] for x in E))
        p('oracle fixes', sum(x['body_ok'] for x in E), 'pipeline lift at snap fixes', sum(x['body_okpipe'] for x in E))
    p('\n== N6: lifts at the snapped pixel (body mask) ==')
    rec_ = [x for x in out if x['body_ok'] or x['body_okpipe'] or x['body_okmed15']]
    p('oracle', sum(x['body_ok'] for x in out), 'pipeline', sum(x['body_okpipe'] for x in out), 'median15',
      sum(x['body_okmed15'] for x in out), 'any', len(rec_))
    return by


def n3(out):
    p('\n== N3: oracle fixes -> full successes ==')
    full = collections.Counter()
    fixes = collections.Counter()
    for x in out:
        if not x['body_ok']:
            continue
        rec = x['rec']
        fixes[(x['model'], x['kind'])] += 1
        u, vv_ = x['fixed_pixel']
        sc, gt, th, _ = H._against(rec)
        cam = sc.observation.camera
        px, py = pixel_to_point(u, vv_, cam.width, cam.height)
        tgt = GroundedTarget(point=Point(x=px, y=py), pixel=(u, vv_), world=x['fixed_world'])
        if rec.result.get('source') == 'constraints':
            rs = H.resolve(rec, target=tgt)
            ok = rs.score.success
        elif rec.result.get('source') == 'standard':
            g = standard_goal(sc.observation, tgt, th)
            ok = H.score_pose(rec, g.pose_world, target=tgt).success
        else:
            ok = None
        x['full'] = ok
        if ok:
            full[(x['model'], x['kind'])] += 1
    p('fixes', dict(fixes))
    p('full successes', dict(full))
    for mm in ('qwen38_low', 'glm53', 'dsv41'):
        p(' ', mm, 'fixes', sum(v for k, v in fixes.items() if k[0] == mm), 'full', sum(v for k, v in full.items() if k[0] == mm))
    for kk in ('person', 'object', 'place'):
        p(' ', kk, 'fixes', sum(v for k, v in fixes.items() if k[1] == kk), 'full', sum(v for k, v in full.items() if k[1] == kk))
    gd = [x for x in out if x['model'] in ('glm53', 'dsv41')]
    p('GLM+DeepSeek with a point', len(gd), 'lift classes (target px within 15)', sum(x['cls'] != 'choice' for x in gd),
      'oracle fixes', sum(x['body_ok'] for x in gd), 'full', sum(bool(x.get('full')) for x in gd))
    p('sources of wrong records', dict(collections.Counter(x['source'] for x in out)))


if __name__ == '__main__':
    out = run()
    summarise(out)
    n3(out)
    with open(D + f'v_lift_{SETS}.tsv', 'w') as f:
        keys = ['model', 'arm', 'scene', 'ep', 'trial', 'kind', 'source', 'u', 'v', 'body_px', 'floor_px', 'match_px',
                'body_ok', 'body_okpipe', 'body_okmed15', 'floor_ok', 'match_ok', 'cls', 'Zpt', 'Ztg', 'surf', 'z',
                'dwall', 'landed_m', 'side', 'win_range', 'win_has_target', 'centre_on_target', 'full']
        f.write('\t'.join(keys) + '\n')
        for x in out:
            f.write('\t'.join(str(x.get(k)) for k in keys) + '\n')
    open(D + f'v_lift_{SETS}.txt', 'w').write('\n'.join(lines) + '\n')
