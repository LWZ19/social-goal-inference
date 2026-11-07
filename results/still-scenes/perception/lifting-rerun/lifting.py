"""I11 Stage 0 re-run: wrong targets of C (and, as an extra, E), lifting vs choice.

Reads runs and scenes only through research_harness. No model call, nothing tuned. Prints tables and writes
them to lifting_out.txt and a per-record table to lifting_records.tsv beside this file (crops.py draws the
person and near-miss place cases into crops_wrong_targets.png; frontsnap.py tests a zero-call snap).

Definitions (fixed before looking at the counts; the alternatives are printed beside them):
  wrong target  : a record whose recorded score has target_correct False. "with a point" = its result holds a
                  grounded target (status ok); the no_target ones have no point and cannot be lifted.
  per-pixel lift: every pixel's own depth lifted at its centre (window 1) to a world point.
  match mask    : pixels whose per-pixel lift the package's own target test (sgi.evaluate.match_target, here
                  vectorised and checked against it) accepts as the true target: "pixels that would ground it".
  body mask     : the target's approximate visible surface from ground truth, checked against depth:
                  person = per-pixel lift within R_PERSON m (horizontal) of the person and 0.05 < z < 2.0 m, and
                           that person is the nearest one;
                  place  = per-pixel lift inside the footprint (within 0.05 m) and z > 0.05 m;
                  object (no footprint) = per-pixel lift within match_radius (horizontal) of the anchor, z > 0.05.
  snap distance : pixel distance from the model's continuous pixel to the nearest mask pixel centre.
  recoverable by lifting (main) : the body mask has a pixel within SNAP_R=15 px of the model's point AND the oracle
                  snap-and-lift grounds the true target: snap to the nearest body-mask pixel, take the median depth
                  of the body-mask pixels in the 5x5 lift window round it, lift that pixel at that depth, and the
                  package's match_target accepts it.
  pipeline snap : the same snap, then the pipeline's own lift (5x5 median of all depth; H.target_at).
  landed        : the recorded lifted point (the result's target.world): its horizontal distance to the true
                  target's anchor, and what it lies on (another person, an obstacle, the bounding wall, the floor).
  front/behind  : the camera depth under the point (5x5 median) against the median depth of the true target's
                  body-mask pixels within SNAP_R of the point (0.25 m margin): occluder capture if in front,
                  background leak if behind, floor slip if the lifted z < 0.05 m; choice error when no body-mask
                  pixel is within SNAP_R.
"""
import collections
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402
from sgi.evaluate import match_target  # noqa: E402
from sgi.geometry import lift  # noqa: E402
from sgi.types import GroundedTarget, Point  # noqa: E402
from sgi.geometry import pixel_to_point  # noqa: E402

OUT = Path(__file__).resolve().parent
SNAP_R = 15.0
RADII = (3, 5, 10, 15, 20, 30)
R_PERSON = 0.30
R_PERSON_ALT = (0.25, 0.35)
Z_FLOOR = 0.05
H_PERSON = 2.0
FRONT_MARGIN = 0.25
REAL_ARMS = ('real', 'staffed', 'plain')

LOG = []


def out(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ---------------------------------------------------------------- per-scene data, cached (a few scenes at most)
_SD = collections.OrderedDict()


def scene_data(rec):
    key = (str(rec.ref.scenes_dir), rec.scene_id, rec.ref.folder)
    if key in _SD:
        _SD.move_to_end(key)
        return _SD[key]
    sc = H.scene(rec.ref, rec.scene_id)
    cam = sc.observation.camera
    depth = sc.observation.depth
    Z = depth.astype(np.float64)
    h, w = Z.shape
    vv, uu = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing='ij')
    pc = np.stack([(uu - cam.cx) * Z / cam.fx, (vv - cam.cy) * Z / cam.fy, Z, np.ones_like(Z)], axis=-1)
    W = (pc @ np.asarray(cam.T_world_camera).T)[..., :3]
    valid = np.isfinite(Z) & (Z > 0)
    people = sc.truth.people
    P = np.array([[p.x, p.y] for p in people]) if people else np.zeros((0, 2))
    if len(P):
        d2 = (W[..., 0, None] - P[:, 0]) ** 2 + (W[..., 1, None] - P[:, 1]) ** 2
        nearest = np.argmin(d2, axis=-1)
        nearest_d = np.sqrt(np.min(d2, axis=-1))
    else:
        nearest = np.full(Z.shape, -1)
        nearest_d = np.full(Z.shape, np.inf)
    sd = dict(sc=sc, cam=cam, depth=depth, Z=Z, W=W.astype(np.float64), valid=valid, people=people, P=P,
              nearest=nearest, nearest_d=nearest_d, masks={})
    _SD[key] = sd
    while len(_SD) > 3:
        _SD.popitem(last=False)
    return sd


def poly_dist(X, Y, poly):
    """Vectorised sgi.polygons.distance_to_polygon: 0 inside, else distance to the boundary."""
    v = np.asarray(poly, dtype=np.float64)
    n = len(v)
    inside = np.zeros(X.shape, bool)
    dmin = np.full(X.shape, np.inf)
    for i in range(n):
        ax, ay = v[i - 1]
        bx, by = v[i]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = np.clip(((X - ax) * dx + (Y - ay) * dy) / L2, 0, 1) if L2 > 0 else 0.0
        dmin = np.minimum(dmin, np.hypot(X - (ax + t * dx), Y - (ay + t * dy)))
        cross = (ay > Y) != (by > Y)
        with np.errstate(divide='ignore', invalid='ignore'):
            xc = ax + (Y - ay) * (bx - ax) / (by - ay)
        inside ^= cross & (xc > X)
    inside |= dmin <= 1e-9
    return np.where(inside, 0.0, dmin)


def target_masks(sd, tid, th, r_person=R_PERSON):
    key = (tid, r_person)
    if key in sd['masks']:
        return sd['masks'][key]
    truth = sd['sc'].truth
    t = truth.targets[tid]
    W, valid = sd['W'], sd['valid']
    X, Y, Zw = W[..., 0], W[..., 1], W[..., 2]
    if t.footprint is not None:
        dist = poly_dist(X, Y, t.footprint)
        match = dist <= th.match_tolerance_m
        body = (dist <= 0.05) & (Zw > Z_FLOOR)
    else:
        radius = th.default_match_radius_m if t.match_radius is None else t.match_radius
        dh = np.hypot(X - t.anchor[0], Y - t.anchor[1])
        match = dh <= radius
        if t.kind == 'person':
            pid = [p.id for p in sd['people']].index(t.person_id)
            near_ok = sd['nearest'] == pid
            match &= near_ok
            per = sd['people'][pid]
            dp = np.hypot(X - per.x, Y - per.y)
            body = (dp <= r_person) & (Zw > Z_FLOOR) & (Zw < H_PERSON) & near_ok
        else:
            body = (dh <= radius) & (Zw > Z_FLOOR)
    match &= valid
    body &= valid
    sd['masks'][key] = (match, body)
    return match, body


def check_vectorised(rec, sd, tid, th, gt, rng):
    """The vectorised match mask against the package's match_target on sampled pixels; returns mismatches."""
    match, _ = target_masks(sd, tid, th)
    h, w = match.shape
    iv = rng.integers(0, h, 400)
    iu = rng.integers(0, w, 400)
    on = np.argwhere(match)
    if len(on):
        pick = on[rng.integers(0, len(on), min(200, len(on)))]
        iv = np.concatenate([iv, pick[:, 0]])
        iu = np.concatenate([iu, pick[:, 1]])
    bad = 0
    for i, j in zip(iv, iu):
        got = match_target(sd['W'][i, j], sd['sc'].truth, gt, th) == tid
        bad += got != bool(match[i, j])
    return bad, len(iv)


def nearest_pixel(mask, u, v):
    idx = np.argwhere(mask)
    if not len(idx):
        return None, math.inf
    d = np.hypot(idx[:, 1] + 0.5 - u, idx[:, 0] + 0.5 - v)
    k = int(np.argmin(d))
    return (int(idx[k, 0]), int(idx[k, 1])), float(d[k])


def window(arr, i, j, half=2):
    return arr[max(i - half, 0):i + half + 1, max(j - half, 0):j + half + 1]


def lift_at_depth(cam, u, v, Z):
    p = np.array([(u - cam.cx) * Z / cam.fx, (v - cam.cy) * Z / cam.fy, Z, 1.0])
    return (np.asarray(cam.T_world_camera) @ p)[:3]


def wall_distance(sd, x, y):
    truth = sd['sc'].truth
    if truth.bounds is None:
        return math.inf
    poly = truth.bounds.polygon if hasattr(truth.bounds, 'polygon') else truth.bounds
    v = np.asarray(poly, dtype=np.float64)
    p = np.array([x, y])
    best = math.inf
    for i in range(len(v)):
        a, b = v[i - 1], v[i]
        t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + t * (b - a)))))
    return best


def surface(sd, world, tid):
    """What a lifted world point lies on: another person, an obstacle, a target footprint, the bounding wall, the
    foot of the wall, the floor, or other."""
    truth = sd['sc'].truth
    x, y, z = (float(c) for c in world)
    t = truth.targets[tid]
    wd = wall_distance(sd, x, y)
    if z < Z_FLOOR:
        return 'wall base (floor)' if wd <= 0.30 else 'floor'
    best = None
    for p in sd['people']:
        d = math.hypot(x - p.x, y - p.y)
        if d <= 0.45 and z < H_PERSON and (best is None or d < best[0]):
            best = (d, p.id)
    if best is not None:
        return ('target person ' if (t.kind == 'person' and best[1] == t.person_id) else 'person ') + best[1]
    for o in truth.obstacles:
        poly = o.polygon if hasattr(o, 'polygon') else o
        if poly_dist(np.array([x]), np.array([y]), poly)[0] <= 0.10:
            return 'obstacle'
    for k2, tt in truth.targets.items():
        if tt.footprint is not None and poly_dist(np.array([x]), np.array([y]), tt.footprint)[0] <= 0.10:
            return f'footprint {k2}({tt.label})'
    if wd <= 0.30:
        return 'wall'
    return 'other'


def analyse(rec, rng, do_resolve=True):
    sd = scene_data(rec)
    sc = sd['sc']
    th = H.recorded_config(rec.ref)[1].thresholds
    gt = sc.truths[rec.episode_id].gt
    tid = gt.target_ids[0]
    t = sc.truth.targets[tid]
    row = dict(stage=rec.ref.stage, arm=rec.arm, model=rec.model, method=rec.method, scene=rec.scene_id,
               episode=rec.episode_id, trial=rec.trial, situation=rec.situation, tid=tid, kind=t.kind,
               label=t.label, ntargets=len(gt.target_ids), status=rec.result.get('status'))
    cls = 'parcel' if (t.kind == 'object') else t.kind
    row['class'] = cls
    if rec.target is None:
        row['has_point'] = False
        return row
    row['has_point'] = True
    u, v = rec.target['pixel']
    row['u'], row['v'] = float(u), float(v)
    cam = sd['cam']
    i, j = math.floor(v), math.floor(u)
    world_rec = np.asarray(rec.target['world'])
    # reproduction of the pipeline's lift
    again = H.target_at(rec, pixel=(u, v))
    row['lift_repro_m'] = float(np.linalg.norm(np.asarray(again.world) - world_rec))
    row['rec_match'] = match_target(world_rec, sc.truth, gt, th)
    bad, nchk = check_vectorised(rec, sd, tid, th, gt, rng)
    row['vec_mismatch'] = bad
    row['vec_checked'] = nchk
    match, body = target_masks(sd, tid, th)
    qm, dm = nearest_pixel(match, u, v)
    qb, db = nearest_pixel(body, u, v)
    row['d_match_px'] = dm
    row['d_body_px'] = db
    for rp in R_PERSON_ALT:
        if t.kind == 'person':
            _, b2 = target_masks(sd, tid, th, r_person=rp)
            row[f'd_body_px_r{rp}'] = nearest_pixel(b2, u, v)[1]
    row['body_pixels'] = int(body.sum())
    # landed
    ax, ay = t.anchor[0], t.anchor[1]
    if t.kind == 'person':
        per = [p for p in sd['people'] if p.id == t.person_id][0]
        ax, ay = per.x, per.y
    row['landed_m'] = math.hypot(world_rec[0] - ax, world_rec[1] - ay)
    row['landed_z'] = float(world_rec[2])
    row['surface'] = surface(sd, world_rec, tid)
    row['wall_dist_m'] = wall_distance(sd, world_rec[0], world_rec[1])
    row['source'] = rec.result.get('source')
    if t.footprint is not None:
        # variants of a place's mask, to see whether any gives the report's 0 of 23
        X, Y, Zw = sd['W'][..., 0], sd['W'][..., 1], sd['W'][..., 2]
        dist = poly_dist(X, Y, t.footprint)
        floor_in = (dist <= 0.0) & (Zw <= Z_FLOOR) & sd['valid']
        row['d_place_floor_only_px'] = nearest_pixel(floor_in, u, v)[1]
        strict = (dist <= 0.0) & (Zw > Z_FLOOR) & sd['valid']
        row['d_place_strict_px'] = nearest_pixel(strict, u, v)[1]
    if t.kind == 'person':
        from sgi.geometry import project
        per = [p for p in sd['people'] if p.id == t.person_id][0]
        pr = project(cam, (per.x, per.y, t.anchor[2]))
        bi_ = np.argwhere(body)
        if pr is not None and len(bi_):
            cu, cv = bi_[:, 1].mean() + 0.5, bi_[:, 0].mean() + 0.5
            row['d_amodal_centre_px'] = math.hypot(u - pr[0], v - pr[1])
            row['d_visible_centroid_px'] = math.hypot(u - cu, v - cv)
            row['amodal_u'] = pr[0]
            row['visible_u'] = cu
    # depths
    Zwin = window(sd['Z'], i, j)
    Zpt = float(np.median(Zwin[np.isfinite(Zwin) & (Zwin > 0)]))
    row['Z_point'] = Zpt
    row['Z_centre_px'] = float(sd['Z'][i, j])
    row['win_range_m'] = float(np.nanmax(Zwin) - np.nanmin(Zwin))
    row['win_body_px'] = int(window(body, i, j).sum())
    row['win_match_px'] = int(window(match, i, j).sum())
    row['centre_px_match'] = bool(match[i, j])
    # target depth near the point
    idx = np.argwhere(body)
    if len(idx):
        dd = np.hypot(idx[:, 1] + 0.5 - u, idx[:, 0] + 0.5 - v)
        near = idx[dd <= SNAP_R]
        if len(near):
            Zt = float(np.median(sd['Z'][near[:, 0], near[:, 1]]))
        else:
            Zt = float(sd['Z'][qb[0], qb[1]])
    else:
        Zt = math.nan
    row['Z_target'] = Zt
    row['dZ_point_minus_target'] = Zpt - Zt
    # class of the error: depth relation first, then the floor
    if db > SNAP_R:
        lc = 'choice error'
    elif Zpt < Zt - FRONT_MARGIN:
        lc = 'occluder capture'
    elif Zpt > Zt + FRONT_MARGIN:
        lc = 'background leak'
    elif world_rec[2] < Z_FLOOR:
        lc = 'floor slip'
    else:
        lc = 'near-depth miss'
    row['lift_class'] = lc
    # fixes
    row['fix_oracle'] = False
    row['fix_pipe'] = False
    row['fix_oracle_any_r'] = None
    if qb is not None:
        bi, bj = qb
        wb = window(body, bi, bj)
        wz = window(sd['Z'], bi, bj)
        Zm = float(np.median(wz[wb]))
        uq, vq = bj + 0.5, bi + 0.5
        w_or = lift_at_depth(cam, uq, vq, Zm)
        ok_or = match_target(w_or, sc.truth, gt, th) == tid
        row['fix_oracle_any_r'] = ok_or
        row['fix_oracle'] = bool(ok_or and db <= SNAP_R)
        w_pipe = lift(sd['depth'], cam, uq, vq, window=th.lift_window)
        ok_pipe = w_pipe is not None and match_target(w_pipe, sc.truth, gt, th) == tid
        row['fix_pipe_any_r'] = ok_pipe
        row['fix_pipe'] = bool(ok_pipe and db <= SNAP_R)
        # median over the whole body mask near the point (variant), lifted at the snapped pixel
        if len(idx):
            near = idx[np.hypot(idx[:, 1] + 0.5 - u, idx[:, 0] + 0.5 - v) <= SNAP_R]
            if len(near):
                Zall = float(np.median(sd['Z'][near[:, 0], near[:, 1]]))
                w_v = lift_at_depth(cam, uq, vq, Zall)
                row['fix_median_near'] = match_target(w_v, sc.truth, gt, th) == tid
            else:
                row['fix_median_near'] = False
        # full success of the oracle fix (main definition)
        row['fix_success'] = None
        if row['fix_oracle'] and do_resolve:
            x, y = pixel_to_point(uq, vq, cam.width, cam.height)
            gtg = GroundedTarget(point=Point(x=x, y=y), pixel=(uq, vq), world=tuple(float(c) for c in w_or))
            if rec.solver_step:
                rs = H.resolve(rec, target=gtg)
                row['fix_success'] = bool(rs.score.success)
                row['fix_path'] = 'solver'
            elif rec.result.get('source') == 'standard':
                from sgi.stages.goal import standard_goal
                sg = standard_goal(sc.observation, gtg, th)
                if sg.pose_world is not None:
                    s_ = H.score_pose(rec, sg.pose_world, target=gtg)
                    row['fix_success'] = bool(s_.success)
                row['fix_path'] = 'standard goal again'
            elif rec.pose_world is not None:
                # E/R: the pose is the model's own mark; only the target changes
                s_ = H.score_pose(rec, rec.pose_world, target=gtg)
                row['fix_success'] = bool(s_.success)
                row['fix_path'] = 'recorded pose, target replaced (' + str(rec.result.get('source')) + ')'
    return row


def table_by_class(rows, title):
    out('')
    out('=' * 100)
    out(title)
    cls_order = ['person', 'parcel', 'place']
    out(f"{'class':8s} {'n':>3s} {'n_pt':>4s} {'body<=15':>8s} {'match<=15':>9s} {'fix_oracle':>10s} {'fix_pipe':>8s} "
        f"{'fix_med':>7s} {'fix->success':>12s}")
    for c in cls_order + ['ALL']:
        rs = [r for r in rows if (c == 'ALL' or r['class'] == c)]
        p = [r for r in rs if r['has_point']]
        if not rs:
            continue
        fs = [r for r in p if r.get('fix_oracle')]
        succ = sum(1 for r in fs if r.get('fix_success'))
        out(f"{c:8s} {len(rs):3d} {len(p):4d} {sum(r['d_body_px'] <= SNAP_R for r in p):8d} "
            f"{sum(r['d_match_px'] <= SNAP_R for r in p):9d} {sum(bool(r['fix_oracle']) for r in p):10d} "
            f"{sum(bool(r['fix_pipe']) for r in p):8d} {sum(bool(r.get('fix_median_near')) for r in p):7d} "
            f"{succ:5d} of {len(fs):3d}")
    out('  radius sweep (body mask within r px | oracle fix within r px):')
    for c in cls_order:
        p = [r for r in rows if r['class'] == c and r['has_point']]
        if not p:
            continue
        s1 = ' '.join(f"r{r_}:{sum(r['d_body_px'] <= r_ for r in p)}|{sum(bool(r['fix_oracle_any_r']) and r['d_body_px'] <= r_ for r in p)}" for r_ in RADII)
        out(f"   {c:7s} n={len(p):2d} {s1}")
    out('  lift class:')
    for c in cls_order:
        p = [r for r in rows if r['class'] == c and r['has_point']]
        if p:
            out(f"   {c:7s} {dict(collections.Counter(r['lift_class'] for r in p))}")
    nopt = [r for r in rows if not r['has_point']]
    if nopt:
        out(f"  no point (no_target): {len(nopt)} -> {dict(collections.Counter((r['model'], r['class'], r['label']) for r in nopt))}")


def main():
    rng = np.random.default_rng(0)
    sel = [('s19', 'qwen38_low', 'con'), ('s19', 'glm53', 'con'), ('s20', 'dsv41', 'con')]
    rows = []
    for stage, model, method in sel:
        for arm in REAL_ARMS + ('ph',):
            runs = H.runs(stage=stage, arm=arm, model=model, method=method)
            assert len(runs) == 1, (stage, arm, model, method, runs)
            run = runs[0]
            recs = [r for r in H.records(run) if not r.score['target_correct']]
            recs.sort(key=lambda r: r.scene_id)
            for rec in recs:
                rows.append(analyse(rec, rng))
            out(f"{stage} {arm} {model} {method}: {len(recs)} wrong-target records ({run.folder})")
    real = [r for r in rows if r['arm'] in REAL_ARMS]
    ph = [r for r in rows if r['arm'] == 'ph']
    out('')
    out('counts per model, realistic sets (with point / all):')
    for m in ('qwen38_low', 'glm53', 'dsv41'):
        rs = [r for r in real if r['model'] == m]
        out(f"  {m:11s} with point {sum(r['has_point'] for r in rs):2d}  all {len(rs):2d}  "
            f"{dict(collections.Counter(r['class'] for r in rs if r['has_point']))}")
    out(f"  sanity: lift reproduction max error {max(r['lift_repro_m'] for r in rows if r['has_point']):.2e} m; "
        f"recorded points matching a target now: {sum(r['rec_match'] is not None for r in rows if r['has_point'])}; "
        f"vectorised mask vs match_target mismatches {sum(r['vec_mismatch'] for r in rows if r['has_point'])} of "
        f"{sum(r['vec_checked'] for r in rows if r['has_point'])} sampled pixels; targets per episode "
        f"{set(r['ntargets'] for r in rows)}")
    table_by_class(real, 'W1 main: C wrong targets, s19 Qwen/GLM + s20 DeepSeek, realistic sets (real, staffed, plain)')
    table_by_class(ph, 'Extra: same models, placeholder set (ph)')
    table_by_class(real + ph, 'Extra: realistic + placeholder')
    out('')
    out('place mask variants (realistic, places with a point): within 15 px of the model point')
    pl = [r for r in real if r['class'] == 'place' and r['has_point']]
    out(f"  body (footprint+0.05 m, z>0.05): {sum(r['d_body_px'] <= SNAP_R for r in pl)} of {len(pl)}; "
        f"strict footprint (z>0.05): {sum(r['d_place_strict_px'] <= SNAP_R for r in pl)}; match mask "
        f"(package test): {sum(r['d_match_px'] <= SNAP_R for r in pl)}; floor-only inside footprint: "
        f"{sum(r['d_place_floor_only_px'] <= SNAP_R for r in pl)}")
    for r in pl:
        if r['d_body_px'] <= SNAP_R or r['d_match_px'] <= SNAP_R:
            out(f"   {r['model']:10s} {r['arm']:7s} {r['scene']:34s} {r['episode']} t{r['trial']} {r['label']:12s} "
                f"dBody {r['d_body_px']:.1f} dMatch {r['d_match_px']:.1f} dFloorOnly {r['d_place_floor_only_px']:.1f} "
                f"landed {r['landed_m']:.2f} m on {r['surface']} ({r['lift_class']}); oracle fix {r['fix_oracle']} "
                f"-> success {r.get('fix_success')}")
    out('  fix paths used for the success replay: ' + str(dict(collections.Counter(r.get('fix_path') for r in rows if r.get('fix_oracle')))))

    # W2 detail table
    out('')
    out('=' * 100)
    out('W2: person wrong targets (realistic), one row each')
    hdr = (f"{'model':10s} {'arm':7s} {'scene':36s} {'ep':3s} {'t':>1s} {'pixel':>13s} {'dBody':>6s} {'dB.25':>6s} "
           f"{'dB.35':>6s} {'dMatch':>6s} {'landed_m':>8s} {'z':>5s} {'Zpt':>5s} {'Ztgt':>5s} {'winR':>5s} {'winB':>4s} "
           f"{'ctr':>4s} {'class':16s} {'surface':22s} {'fixO':>4s} {'fixP':>4s} {'succ':>4s}")
    out(hdr)
    pers = [r for r in real if r['class'] == 'person' and r['has_point']]
    for r in pers:
        out(f"{r['model']:10s} {r['arm']:7s} {r['scene']:36s} {r['episode']:3s} {r['trial']:1d} "
            f"({r['u']:5.0f},{r['v']:5.0f}) {r['d_body_px']:6.1f} {r.get('d_body_px_r0.25', math.nan):6.1f} "
            f"{r.get('d_body_px_r0.35', math.nan):6.1f} {r['d_match_px']:6.1f} {r['landed_m']:8.2f} {r['landed_z']:5.2f} "
            f"{r['Z_point']:5.2f} {r['Z_target']:5.2f} {r['win_range_m']:5.2f} {r['win_body_px']:4d} "
            f"{str(r['centre_px_match'])[0]:>4s} {r['lift_class']:16s} {r['surface'][:22]:22s} "
            f"{str(r['fix_oracle'])[0]:>4s} {str(r['fix_pipe'])[0]:>4s} {str(r.get('fix_success'))[0]:>4s}")
    if pers:
        out(f"  n={len(pers)}; body mask within 15 px: {sum(r['d_body_px'] <= SNAP_R for r in pers)}; "
            f"max d_body {max(r['d_body_px'] for r in pers):.1f} px; max d_match {max(r['d_match_px'] for r in pers):.1f} px; "
            f"landed range {min(r['landed_m'] for r in pers):.2f}-{max(r['landed_m'] for r in pers):.2f} m; "
            f"|dZ| range {min(abs(r['dZ_point_minus_target']) for r in pers):.2f}-{max(abs(r['dZ_point_minus_target']) for r in pers):.2f} m")
        out(f"  surfaces: {dict(collections.Counter(r['surface'] for r in pers))}")
        out(f"  classes:  {dict(collections.Counter(r['lift_class'] for r in pers))}")
        out(f"  scenes:   {dict(collections.Counter(r['scene'].split('_', 1)[1] for r in pers))}")
        out(f"  in front of the target (occluder): {sum(r['dZ_point_minus_target'] < -FRONT_MARGIN for r in pers)}; "
            f"behind it (background): {sum(r['dZ_point_minus_target'] > FRONT_MARGIN for r in pers)} -> surfaces behind "
            f"{dict(collections.Counter(r['surface'] for r in pers if r['dZ_point_minus_target'] > FRONT_MARGIN))}; "
            f"wall distance of the background ones (m): "
            f"{[round(r['wall_dist_m'], 2) for r in pers if r['dZ_point_minus_target'] > FRONT_MARGIN]}")
        out(f"  landed (horizontal distance of the recorded lift from the true person): occluder ones "
            f"{min(r['landed_m'] for r in pers if r['dZ_point_minus_target'] < 0):.2f}-"
            f"{max(r['landed_m'] for r in pers if r['dZ_point_minus_target'] < 0):.2f} m, background ones "
            f"{min(r['landed_m'] for r in pers if r['dZ_point_minus_target'] > 0):.2f}-"
            f"{max(r['landed_m'] for r in pers if r['dZ_point_minus_target'] > 0):.2f} m")

    # W3: eval_12
    out('')
    out('=' * 100)
    out('W3: eval_12 (conversation_group_4) wrong targets, realistic + placeholder, edge diagnostics')
    e12 = [r for r in rows if 'eval_12' in r['scene'] and r['has_point']]
    for r in e12:
        out(f"  {r['model']:10s} {r['arm']:7s} {r['episode']} t{r['trial']} dBody={r['d_body_px']:.1f}px "
            f"window(5x5): depth range {r['win_range_m']:.2f} m, target px {r['win_body_px']}/25, centre px on target "
            f"{r['centre_px_match']}; Zpt {r['Z_point']:.2f} vs Ztgt {r['Z_target']:.2f}; class {r['lift_class']}; "
            f"on {r['surface']}; landed {r['landed_m']:.2f} m; oracle fix {r['fix_oracle']} pipe fix {r['fix_pipe']}")
    if e12:
        out(f"  n={len(e12)}: within 3 px of target {sum(r['d_body_px'] <= 3 for r in e12)}, within 5 px "
            f"{sum(r['d_body_px'] <= 5 for r in e12)}, within 10 px {sum(r['d_body_px'] <= 10 for r in e12)}; window straddles "
            f"a >0.5 m depth jump {sum(r['win_range_m'] > 0.5 for r in e12)}; window holds target pixels "
            f"{sum(r['win_body_px'] > 0 for r in e12)}; classes {dict(collections.Counter(r['lift_class'] for r in e12))}")

    if e12:
        am = [r for r in e12 if 'd_amodal_centre_px' in r]
        out(f"  amodal check (eval_12): model point nearer the projected 3D centre of the target (at anchor height) "
            f"than the centroid of its visible mask in {sum(r['d_amodal_centre_px'] < r['d_visible_centroid_px'] for r in am)} "
            f"of {len(am)}; projected centre u={am[0]['amodal_u']:.0f}, visible centroid u={am[0]['visible_u']:.0f}; "
            f"point u values {sorted(round(r['u']) for r in am)}")
        out(f"  occluder side (point on p3) {sum(r['surface'].startswith('person') for r in e12)}, background side "
            f"{sum(r['dZ_point_minus_target'] > FRONT_MARGIN for r in e12)}")

    # Extra: E (worked examples) wrong targets, realistic sets, same analysis
    erows = []
    for stage, model, method in [('s16b', 'qwen38_low', 'fsc'), ('s16b', 'glm53', 'fsc'), ('s20', 'dsv41', 'fsc')]:
        for arm in REAL_ARMS:
            runs = H.runs(stage=stage, arm=arm, model=model, method=method)
            assert len(runs) == 1, (stage, arm, model, method, runs)
            recs = [r for r in H.records(runs[0]) if not r.score['target_correct']]
            recs.sort(key=lambda r: r.scene_id)
            for rec in recs:
                erows.append(analyse(rec, rng))
            out(f"{stage} {arm} {model} {method}: {len(recs)} wrong-target records ({runs[0].folder})")
    table_by_class(erows, 'Extra: E (fsc) wrong targets, s16b Qwen/GLM + s20 DeepSeek, realistic sets')
    for m in ('qwen38_low', 'glm53', 'dsv41'):
        rs = [r for r in erows if r['model'] == m]
        out(f"  {m:11s} E wrong targets {len(rs)} (with point {sum(r['has_point'] for r in rs)}): "
            f"{dict(collections.Counter(r['class'] for r in rs if r['has_point']))}; person recoverable "
            f"{sum(bool(r.get('fix_oracle')) for r in rs if r['class'] == 'person')}")
    rows = rows + erows

    # per-record TSV
    keys = sorted({k for r in rows for k in r})
    with open(OUT / 'lifting_records.tsv', 'w') as f:
        f.write('\t'.join(keys) + '\n')
        for r in rows:
            f.write('\t'.join(str(r.get(k, '')) for k in keys) + '\n')
    (OUT / 'lifting_out.txt').write_text('\n'.join(LOG) + '\n')
    return rows


if __name__ == '__main__':
    main()
