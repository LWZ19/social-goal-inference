"""N4: a zero-call front-surface snap over every C record with a point (realistic, three models), and, as an extra,
the oracle snap-and-lift applied to every record (fixes and breaks). Reads only through research_harness."""
import sys, math, collections
import numpy as np
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
from sgi.evaluate import match_target
from sgi.geometry import _depth_at

D = 'results/2026-10-17-beyond-vlm/rerun/lifting/verify/'
WIN = 16
lines = []


def p(*a):
    s = ' '.join(str(x) for x in a); print(s); lines.append(s)


def unproject(cam, u, v, Z):
    pc = np.array([(u - cam.cx) * Z / cam.fx, (v - cam.cy) * Z / cam.fy, Z, 1.0])
    return (cam.T_world_camera @ pc)[:3]


def poly_dist(X, Y, poly):
    """distance of each (X, Y) to the polygon (0 inside), vectorised."""
    P = np.asarray(poly, float)
    n = len(P)
    inside = np.zeros(X.shape, bool)
    dmin = np.full(X.shape, np.inf)
    for k in range(n):
        ax, ay = P[k]; bx, by = P[(k + 1) % n]
        cond = ((ay > Y) != (by > Y)) & (X < (bx - ax) * (Y - ay) / ((by - ay) if by != ay else 1e-12) + ax)
        inside ^= cond
        dx, dy = bx - ax, by - ay
        L = dx * dx + dy * dy
        t = np.clip(((X - ax) * dx + (Y - ay) * dy) / (L if L > 0 else 1), 0, 1)
        dmin = np.minimum(dmin, np.hypot(X - (ax + t * dx), Y - (ay + t * dy)))
    return np.where(inside, 0.0, dmin)


def local(depth, cam, u, v):
    r0, c0 = math.floor(v), math.floor(u)
    rs, re_ = max(r0 - WIN, 0), min(r0 + WIN + 1, depth.shape[0])
    cs, ce = max(c0 - WIN, 0), min(c0 + WIN + 1, depth.shape[1])
    Z = depth[rs:re_, cs:ce].astype(np.float64)
    vv, uu = np.meshgrid(np.arange(rs, re_) + 0.5, np.arange(cs, ce) + 0.5, indexing='ij')
    valid = np.isfinite(Z) & (Z > 0)
    pc = np.stack([(uu - cam.cx) * Z / cam.fx, (vv - cam.cy) * Z / cam.fy, Z, np.ones_like(Z)], axis=-1)
    W = (pc @ cam.T_world_camera.T)[..., :3]
    return W, Z, uu, vv, valid, r0 - rs, c0 - cs


def body_mask(truth, gt, th, W, valid):
    X, Y, Zw = W[..., 0], W[..., 1], W[..., 2]
    people = truth.people
    pxy = np.array([[q.x, q.y] for q in people], float).reshape(-1, 2)
    nearest = np.argmin((X[..., None] - pxy[:, 0]) ** 2 + (Y[..., None] - pxy[:, 1]) ** 2, axis=-1) if len(pxy) else np.full(X.shape, -1)
    body = np.zeros(valid.shape, bool)
    for tid in gt.target_ids:
        t = truth.targets[tid]
        if t.kind == 'person':
            k = [q.id for q in people].index(t.person_id)
            body |= valid & (np.hypot(X - pxy[k, 0], Y - pxy[k, 1]) <= 0.30) & (Zw > 0.05) & (Zw < 2.0) & (nearest == k)
        elif t.footprint is not None:
            body |= valid & (poly_dist(X, Y, t.footprint) <= 0.05) & (Zw > 0.05)
        else:
            rad = th.default_match_radius_m if t.match_radius is None else t.match_radius
            body |= valid & (np.hypot(X - t.anchor[0], Y - t.anchor[1]) <= rad) & (Zw > 0.05)
    return body


def main():
    tally = collections.Counter()
    oracle = collections.Counter()
    by_kind = collections.Counter()
    n = 0
    for st, m in (('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41')):
        for arm in ('real', 'staffed', 'plain'):
            for run in H.runs(stage=st, arm=arm, model=m, method='con'):
                for rec in H.records(run):
                    if rec.target is None:
                        continue
                    n += 1
                    sc, gt, th, _ = H._against(rec)
                    cam, depth = sc.observation.camera, sc.observation.depth
                    u, v = rec.target['pixel']
                    was = bool(rec.score['target_correct'])
                    kind = '/'.join(sorted({sc.truth.targets[t].kind for t in gt.target_ids}))
                    W, Z, uu, vv, valid, i0, j0 = local(depth, cam, u, v)
                    # zero-call front snap
                    Zm = _depth_at(depth, cam, u, v, th.lift_window)
                    dpx = np.hypot(uu - u, vv - v)
                    cand = valid & (dpx <= 15) & (Z <= Zm - 0.5)
                    if cand.any():
                        dd = np.where(cand, dpx, np.inf)
                        i, j = np.unravel_index(np.argmin(dd), dd.shape)
                        z0 = Z[i, j]
                        sub = Z[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3]
                        sv = valid[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3]
                        keep = sv & (np.abs(sub - z0) <= 0.3)
                        w = unproject(cam, uu[i, j], vv[i, j], float(np.median(sub[keep])))
                        now = match_target(w, sc.truth, gt, th) is not None
                        moved = True
                    else:
                        now, moved = was, False
                    if moved:
                        key = ('fix' if (now and not was) else 'break' if (was and not now) else 'moved_still_ok' if now else 'moved_still_wrong')
                        tally[key] += 1
                        tally[(key, kind)] += 1
                    else:
                        tally['unmoved'] += 1
                    # oracle snap on every record
                    body = body_mask(sc.truth, gt, th, W, valid)
                    bd = np.where(body, dpx, np.inf)
                    if np.isfinite(bd).any() and bd.min() <= 15:
                        i, j = np.unravel_index(np.argmin(bd), bd.shape)
                        sub = Z[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3]
                        sm = body[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3]
                        w = unproject(cam, uu[i, j], vv[i, j], float(np.median(sub[sm])))
                        now2 = match_target(w, sc.truth, gt, th) is not None
                    else:
                        now2 = was
                    oracle[('fix' if now2 and not was else 'break' if was and not now2 else 'same')] += 1
                    by_kind[(kind, was)] += 1
    p('C records with a point (realistic, 3 models):', n)
    p('zero-call front snap:', {k: v for k, v in tally.items() if isinstance(k, str)})
    p('  by kind:', {k: v for k, v in tally.items() if not isinstance(k, str)})
    p('oracle snap-and-lift on every record:', dict(oracle))
    p('records by (kind, recorded target_correct):', dict(by_kind))
    open(D + 'v_n4.txt', 'w').write('\n'.join(lines) + '\n')


main()
