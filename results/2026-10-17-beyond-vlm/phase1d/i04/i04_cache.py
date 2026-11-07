"""I04 step 1 (FROZEN.md sections 2-4 and the density places of section 7): the per-record cache.

Reads runs and scenes only through research_harness. No model call, no network, nothing fitted here.
Per goal-step record of the 12 cells (6 primary, 6 new wording):
  - the solver's inputs rebuilt as H.resolve does; cells, settled = sgi.solver._over_floor(...);
  - asserts: picks[0] in the world frame is the recorded place within 1e-6 m; picks[1:5] the recorded runners-up;
    the solver's heading rule gives the recorded headings;
  - the five slot keys in integer millimetres (exactly the solver's mm-rounded keys);
  - every F cell's heading (the solver's own rule), the I14 v0 replica (phase1/i14_v0/i14.py replica()) on every F
    cell, the von Mises probability of a heading within 45 deg of any holding region (kappa 4, 8, 16, 32; kappa inf
    is the replica's indicator), the forbidden kinds, the 3 m disc and the region shares;
  - the replica checked against H.score_poses on a sample (150 uniform, up to 100 in-region, up to 50 just outside),
    plus the heading offset through sgi.evaluate._heading_within at 30 and 60 deg;
  - the density places for V1 and V2 (each gamma) and V3, each scored by the replica and the package.

    cd WT && PYTHONPATH= OMP_NUM_THREADS=2 ... nice -n 10 .venv/bin/python <this> SHARD [LIMIT]
    SHARD: 0 or 1 (runs alternate between the two shards)
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import re
import sys
import time
import zlib

import numpy as np

WT = '.-scenes'
HERE = f'{WT}/results/2026-10-17-beyond-vlm/phase1d/i04'
sys.path.insert(0, f'{WT}/docs/model_notes/tools')
import research_harness as H  # noqa: E402
sys.path.insert(0, f'{WT}/results/2026-10-17-beyond-vlm/phase1/i14_v0')
import i14  # noqa: E402  (replica, pip, wrap: I14 v0's frozen scorer replica)
from sgi import solver  # noqa: E402
from sgi.evaluate import _heading_within  # noqa: E402
from sgi.geometry import make_poses, to_world_xy, wrap_angle  # noqa: E402
from sgi.polygons import point_in_polygon  # noqa: E402
from sgi.stages.goal import standard_goal  # noqa: E402
from sgi.types import Constraint, DetectedPerson, GroundedTarget  # noqa: E402

SEED = 20261018
EPS = 1e-9
KAPPAS = (4.0, 8.0, 16.0, 32.0)                    # finite kappas; kappa inf is the replica's indicator
GAMMAS = (1, 3, 10, 30)
DISC_M = 3.0
CONC_M = 0.3
TOL_DEG = 45.0
NTAB = 2 ** 16 + 1
SAME_PLACE_M = 1e-6

REAL_ARMS = ('real', 'staffed', 'plain')
DEV_ARMS = ('dev_seed0', 'dev_seed1', 'dev_seed2')
CELLS = [('old_qwen', 'old', 's19', 'qwen38_low'), ('old_glm', 'old', 's19', 'glm53'),
         ('old_deepseek', 'old', 's20', 'dsv41'),
         ('watch_qwen', 'watch', 's19', 'qwen38_low'), ('watch_glm', 'watch', 's19', 'glm53'),
         ('watch_deepseek', 'watch', 's20', 'dsv41'),
         ('new_qwen', 'new', 's21', 'qwen38_low'), ('new_glm', 'new', 's23', 'glm53'),
         ('new_deepseek', 'new', 's23', 'dsv41'), ('new_haiku', 'new', 's23', 'haiku55'),
         ('new_sonnet', 'new', 's23', 'sonnet55_low'), ('new_gemini', 'new', 's23', 'gemini38_low')]
EVAL_IDX = re.compile(r'(?:^|_)eval_(\d\d)_')
FLAGS = ('exclusion_broken', 'far_from_target', 'constraint_unusable')


def plan():
    out = []
    for cell, label, stage, model in CELLS:
        for arm in (DEV_ARMS if label == 'watch' else REAL_ARMS):
            refs = H.runs(stage=stage, arm=arm, model=model, method='con')
            assert len(refs) == 1, (stage, arm, model, refs)
            out.append((cell, label, refs[0]))
    return out


def key(rec):
    return f"{rec.ref.folder}/{rec.scene_id}/{rec.episode_id}/{rec.trial}"


def layout_of(label, rec):
    if label == 'watch':
        return f'{rec.arm}/{rec.scene_id}'
    m = EVAL_IDX.search(rec.scene_id)
    assert m, rec.scene_id
    return f'eval_{m.group(1)}'


# ------------------------------------------------------------------------------------------------- von Mises tables
def vm_table(kappa, n=NTAB):
    """(y, cdf) of exp(kappa (cos y - 1)) on [-pi, pi]: trapezoid rule, normalised by the same rule's total."""
    y = np.linspace(-math.pi, math.pi, n)
    f = np.exp(kappa * (np.cos(y) - 1.0))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(y))])
    return y, cum / cum[-1]


def vm_check():
    """Max |F_coarse - F_fine| over the fine grid, per kappa (fine table 4 times finer)."""
    out = {}
    for k in KAPPAS:
        yc, Fc = vm_table(k)
        yf, Ff = vm_table(k, 4 * (NTAB - 1) + 1)
        out[k] = float(np.max(np.abs(np.interp(yf, yc, Fc) - Ff)))
    return out


TABLES = {k: vm_table(k) for k in KAPPAS}


def F(k, x):
    y, c = TABLES[k]
    return np.interp(x, y, c)


def arc_single(k, d, tol):
    """P(y in [d - tol, d + tol] mod 2 pi) for y ~ vM(0, k), d in (-pi, pi], tol < pi."""
    a, b = d - tol, d + tol
    plain = F(k, b) - F(k, a)
    low = F(k, b) + 1.0 - F(k, a + 2 * math.pi)
    high = 1.0 - F(k, a) + F(k, b - 2 * math.pi)
    return np.where(a < -math.pi, low, np.where(b > math.pi, high, plain))


def arc_union(k, ds, tol):
    """P(y in the union of the arcs [d - tol, d + tol] mod 2 pi), y ~ vM(0, k): pieces in [-pi, pi], merged."""
    pieces = []
    for d in ds:
        a, b = d - tol, d + tol
        if a < -math.pi:
            pieces += [(-math.pi, b), (a + 2 * math.pi, math.pi)]
        elif b > math.pi:
            pieces += [(a, math.pi), (-math.pi, b - 2 * math.pi)]
        else:
            pieces.append((a, b))
    pieces.sort()
    merged = [list(pieces[0])]
    for a, b in pieces[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return float(sum(F(k, b) - F(k, a) for a, b in merged))


# ----------------------------------------------------------------------------------------------------- one record
def roundtrip(robot, xy, theta_world):
    """The world pose the scorer sees after make_poses and evaluate's way back (the first lines of i14.replica)."""
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    gx, gy = xy[:, 0].astype(np.float64), xy[:, 1].astype(np.float64)
    dx, dy = gx - robot.x, gy - robot.y
    rx, ry = c * dx + s * dy, c * dy - s * dx
    rth = i14.wrap(theta_world - robot.yaw)
    return robot.x + c * rx - s * ry, robot.y + s * rx + c * ry, rth + robot.yaw


def pip_box(px, py, poly):
    """i14.pip on the points inside the polygon's bounding box (+1e-6); False elsewhere."""
    v = np.array(poly, dtype=np.float64)
    lo, hi = v.min(axis=0) - 1e-6, v.max(axis=0) + 1e-6
    box = (px >= lo[0]) & (px <= hi[0]) & (py >= lo[1]) & (py <= hi[1])
    out = np.zeros(len(px), dtype=bool)
    if box.any():
        out[box] = i14.pip(px[box], py[box], poly)
    return out


def density_places(order, cells, top, rr):
    """Place 1 = top; each next = the first cell of `order` at least 2 rr (-1e-9) from every earlier place."""
    places = [int(top)]
    free = solver._distance(cells, cells[top]) >= 2 * rr - EPS
    while len(places) < 5:
        ok = free[order]
        if not ok.any():
            break
        p = int(order[int(np.argmax(ok))])
        places.append(p)
        free &= solver._distance(cells, cells[p]) >= 2 * rr - EPS
    return places


def one(cell, label, rec, th, npz_dir):
    t0 = time.monotonic()
    sc = H.scene(rec.ref, rec.scene_id)
    obs = sc.observation
    robot = obs.robot
    _, gt, _, cache = H._against(rec)
    places = [tuple(rec.pose_world)] + [tuple(c['pose_world']) for c in rec.candidates]
    assert len(places) == 5, len(places)
    pkg = H.score_poses(rec, places)
    matched = pkg[0].matched_target_id
    assert all(s.matched_target_id == matched for s in pkg)
    rec_success = bool(rec.score.get('success'))
    grid, joined = cache['reach']

    # ---- section 2: the solver's inputs, floor and keys
    target = GroundedTarget.model_validate(rec.target)
    people = [DetectedPerson.model_validate(p) for p in rec.people]
    constraints = [Constraint.model_validate(c['constraint']) for c in rec.constraints]
    std = standard_goal(obs, target, th)
    cells, settled = solver._over_floor(obs, target, people, constraints, std, th)
    N = len(cells)
    res = th.grid_resolution_m
    half = math.floor(th.floor_range_m / res + 1e-9)
    offsets = np.arange(-half, half + 1) * res
    col = np.rint(cells[:, 0] / res).astype(np.int64) + half
    row = np.rint(cells[:, 1] / res).astype(np.int64) + half
    assert np.array_equal(offsets[col], cells[:, 0]) and np.array_equal(offsets[row], cells[:, 1])
    flat = row * (2 * half + 1) + col
    assert np.all(np.diff(flat) > 0), 'cells not in grid order'
    K = settled.keys
    s3 = [n for n in ('turn_m', 'face_m') if n in K]
    assert len(s3) <= 1, s3
    slot3 = s3[0] if s3 else None
    assert set(K) == {'excluded_m', 'excluded_total_m', 'preference_m', 'standard_m', 'robot_m'} | set(s3), list(K)
    names = ['excluded_m', 'excluded_total_m', slot3, 'preference_m', 'standard_m']
    kmm = np.zeros((N, 5), dtype=np.int64)
    for j, nm in enumerate(names):
        if nm is None:
            continue
        kmm[:, j] = np.rint(K[nm] * 1000.0).astype(np.int64)
        assert np.array_equal(kmm[:, j] / 1000.0, K[nm]), nm
    assert kmm.min() >= 0
    # V2's kept cells: _first's rule on the first two keys (no band, 1e-9 tolerance)
    kept = np.ones(N, dtype=bool)
    for nm in ('excluded_m', 'excluded_total_m'):
        least = K[nm][kept].min()
        kept &= K[nm] <= least + EPS
    picks = list(settled.picks)
    top = int(picks[0])
    assert kept[top]

    # ---- world positions and the solver's heading at every cell (poses_at)
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    cx = robot.x + c * cells[:, 0] - s * cells[:, 1]
    cy = robot.y + s * cells[:, 0] + c * cells[:, 1]
    toward = settled.faced_xy[None, :] - cells
    deg = np.hypot(toward[:, 0], toward[:, 1]) < th.heading_degenerate_m
    theta = np.where(deg, robot.yaw, robot.yaw + np.arctan2(toward[:, 1], toward[:, 0]))
    theta_w = i14.wrap(theta)
    for p in picks:                                 # the scalar path of the package, exactly
        wx, wy = to_world_xy(robot, *cells[p])
        assert (wx, wy) == (float(cx[p]), float(cy[p]))

    # ---- reproduction asserts (picks vs recorded places and headings)
    moved, dth = [], []
    for i in range(5):
        if i < len(picks):
            moved.append(math.hypot(cx[picks[i]] - places[i][0], cy[picks[i]] - places[i][1]))
            dth.append(abs(wrap_angle(float(theta_w[picks[i]]) - places[i][2])))
        else:
            moved.append(math.inf)
            dth.append(math.inf)
    repro_ok = moved[0] <= SAME_PLACE_M and max(moved) <= SAME_PLACE_M and max(dth) <= 1e-9 and len(picks) == 5

    # ---- section 3: the replica on every F cell, at the cell's heading
    xy = np.stack([cx, cy], axis=1)
    rep = i14.replica(robot, gt, matched, grid, joined, xy, theta_w)
    wx, wy, th_rt = roundtrip(robot, xy, theta_w)
    regions = [] if matched is None else [g for g in gt.acceptable_regions or [] if g.target_id in (None, matched)]
    for g in regions:
        assert g.tolerance_deg == TOL_DEG, g.tolerance_deg
    holds, dmat = [], []
    for g in regions:
        h = pip_box(wx, wy, g.polygon)
        if g.heading.yaw is not None:
            allowed = np.full(N, float(g.heading.yaw))
        else:
            fx, fy = g.heading.face_point
            allowed = i14.wrap(np.arctan2(fy - wy, fx - wx))
        holds.append(h)
        dmat.append(i14.wrap(allowed - th_rt))
    inreg = np.zeros(N, dtype=bool)
    for h in holds:
        inreg |= h
    assert np.array_equal(inreg, rep['inreg'])
    tol = math.radians(TOL_DEG)
    hok_mine = np.zeros(N, dtype=bool)
    for h, d in zip(holds, dmat):
        hok_mine |= h & (np.abs(d) <= tol)
    hok_agree = int(np.sum(hok_mine == rep['hok']))   # boundary |wrap(-x)| vs |wrap(x)| only at pi; reported
    succ_mask = (matched is not None) & rep['reach'] & rep['inreg']
    succ_idx = np.flatnonzero(succ_mask)
    pvm = np.zeros((len(succ_idx), len(KAPPAS)))
    n_multi = 0
    if len(succ_idx):
        H_ = np.stack([h[succ_idx] for h in holds], axis=1)
        D_ = np.stack([d[succ_idx] for d in dmat], axis=1)
        nh = H_.sum(axis=1)
        assert nh.min() >= 1
        single = nh == 1
        n_multi = int((~single).sum())
        dsingle = D_[single, np.argmax(H_[single], axis=1)]
        for j, k in enumerate(KAPPAS):
            pvm[single, j] = arc_single(k, dsingle, tol)
            for r in np.flatnonzero(~single):
                pvm[r, j] = arc_union(k, D_[r, H_[r]].tolist(), tol)
    hok_succ = rep['hok'][succ_idx]
    assert np.array_equal(rep['success'], np.isin(np.arange(N), succ_idx) & rep['hok'])
    assert pvm.size == 0 or (pvm.min() >= -1e-12 and pvm.max() <= 1 + 1e-12)

    reg_idx = np.flatnonzero(rep['inreg'])
    kinds = sorted({g.kind for g in gt.forbidden_regions or []})
    forb_idx = {kd: np.flatnonzero(rep['forb'][kd]) for kd in kinds}
    tx, ty = (float(v) for v in rec.target['world'][:2])
    in_disc = np.hypot(cx - tx, cy - ty) <= DISC_M + EPS
    n_disc = int(in_disc.sum())
    rho_disc = float(rep['inreg'][in_disc].mean()) if n_disc else 0.0
    rho_room = float(rep['inreg'].mean())
    d_top = solver._distance(cells, cells[top])
    near_top = d_top <= CONC_M + EPS
    dd_int = (col - col[top]) ** 2 + (row - row[top]) ** 2
    assert np.array_equal(near_top, dd_int <= 36)

    # ---- density places (section 7): V1, V2 for each gamma; V3
    dens = {}
    rr = th.robot_radius_m
    for g in GAMMAS:
        w = np.array([g ** 4, g ** 3, g ** 2, g, 1], dtype=np.int64)
        E = kmm @ w
        order = np.argsort(E, kind='stable')
        dens[f'V1_g{g}'] = density_places(order, cells, top, rr)
        order_k = order[kept[order]]
        dens[f'V2_g{g}'] = density_places(order_k, cells, top, rr)
    order3 = np.argsort(dd_int, kind='stable')
    dens['V3'] = density_places(order3, cells, top, rr)
    uniq = sorted({p for v in dens.values() for p in v})
    upkg = H.score_poses(rec, [(float(cx[i]), float(cy[i]), float(theta_w[i])) for i in uniq])
    pk_of = dict(zip(uniq, upkg))
    val = collections.Counter()
    mism = collections.Counter()

    def compare(tag, rep_, i, sco):
        for f, a, b in (('success', bool(rep_['success'][i]), sco.success),
                        ('inreg', bool(rep_['inreg'][i]), sco.in_region),
                        ('heading', bool(rep_['hok'][i]), sco.heading_ok),
                        ('reach', bool(rep_['reach'][i]), sco.reachable)):
            val[f'{tag}.{f}'] += 1
            if a != b:
                mism[f'{tag}.{f}'] += 1
        for kd, flag in sco.forbidden.items():
            val[f'{tag}.forb'] += 1
            if bool(rep_['forb'].get(kd, np.zeros(len(rep_['success']), bool))[i]) != flag:
                mism[f'{tag}.forb'] += 1
        assert sco.matched_target_id == matched

    for i in uniq:
        compare('dens', rep, i, pk_of[i])
    dens_out = {}
    for name, pl in dens.items():
        dens_out[name] = dict(places=pl, succ_rep=[bool(rep['success'][p]) for p in pl],
                              succ_pkg=[bool(pk_of[p].success) for p in pl])
    # the recorded places by the replica, at their recorded headings
    pxy = np.array([p[:2] for p in places], dtype=np.float64)
    pth = np.array([p[2] for p in places], dtype=np.float64)
    prep = i14.replica(robot, gt, matched, grid, joined, pxy, pth)
    for i, sco in enumerate(pkg):
        compare('places', prep, i, sco)

    # ---- validation sample against the package (section 3)
    vrng = np.random.default_rng([SEED, zlib.crc32(key(rec).encode())])
    samp = set(vrng.choice(N, size=min(150, N), replace=False).tolist())
    inr = reg_idx
    if len(inr):
        samp |= set(vrng.choice(inr, size=min(100, len(inr)), replace=False).tolist())
        outs = np.flatnonzero(~rep['inreg'])
        lo = np.array([cx[inr].min(), cy[inr].min()]) - 0.15
        hi = np.array([cx[inr].max(), cy[inr].max()]) + 0.15
        box = outs[(cx[outs] >= lo[0]) & (cx[outs] <= hi[0]) & (cy[outs] >= lo[1]) & (cy[outs] <= hi[1])]
        if len(box):
            dmin = np.full(len(box), np.inf)
            for b0 in range(0, len(inr), 500):
                q = inr[b0:b0 + 500]
                dmin = np.minimum(dmin, np.hypot(cx[box, None] - cx[None, q], cy[box, None] - cy[None, q]).min(axis=1))
            near = box[dmin <= 0.15 + EPS]
            if len(near):
                samp |= set(vrng.choice(near, size=min(50, len(near)), replace=False).tolist())
    samp = sorted(samp)
    vs = H.score_poses(rec, [(float(cx[i]), float(cy[i]), float(theta_w[i])) for i in samp])
    for i, sco in zip(samp, vs):
        compare('sample', rep, i, sco)
    # heading offset through the package's own _heading_within, tolerance 30 and 60 deg
    for i in samp:
        pose_world, pose_robot = make_poses(robot, (float(cx[i]), float(cy[i])), float(theta_w[i]))
        goal = to_world_xy(robot, pose_robot[0], pose_robot[1])
        thg = pose_robot[2] + robot.yaw
        holding = [g for g in regions if point_in_polygon(goal, g.polygon)]
        for tau in (30.0, 60.0):
            a = any(_heading_within(g.model_copy(update={'tolerance_deg': tau}), goal, thg) for g in holding)
            b = bool(rep['inreg'][i] and rep['err'][i] <= math.radians(tau))
            val[f'offset{int(tau)}'] += 1
            if a != b:
                mism[f'offset{int(tau)}'] += 1

    top1 = bool(pkg[0].success)
    assert top1 == rec_success, (top1, rec_success)
    assert bool(rep['success'][top]) == top1 or not repro_ok

    digest = hashlib.sha1(key(rec).encode()).hexdigest()[:16]
    np.savez_compressed(f'{npz_dir}/{digest}.npz', row=row.astype(np.int16), col=col.astype(np.int16),
                        kmm=kmm.astype(np.int32), kept=np.flatnonzero(kept).astype(np.int32),
                        picks=np.array(picks, dtype=np.int32), succ_idx=succ_idx.astype(np.int32), pvm=pvm,
                        hok_succ=hok_succ, reg_idx=reg_idx.astype(np.int32),
                        near_top=np.flatnonzero(near_top).astype(np.int32),
                        **{f'forb__{kd}': v.astype(np.int32) for kd, v in forb_idx.items()})
    flags = rec.result.get('flags') or {}
    meta = dict(key=key(rec), digest=digest, cell=cell, set=label, stage=rec.ref.stage, model=rec.model, arm=rec.arm,
                run=rec.run, scene=rec.scene_id, ep=rec.episode_id, trial=rec.trial, sit=rec.situation,
                layout=layout_of(label, rec), res=res, half=half, n_cells=N, n_kept=int(kept.sum()), slot3=slot3,
                matched=matched, target_ok=bool(pkg[0].target_correct), rec_success=rec_success, top1=top1,
                lex_succ=[bool(x.success) for x in pkg],
                lex_hit=np.logical_or.accumulate([bool(x.success) for x in pkg]).astype(int).tolist(),
                flags={f: bool(flags[f]) for f in FLAGS}, repro_ok=bool(repro_ok), moved=moved, dtheta=dth,
                n_picks=len(picks), rho_room=rho_room, rho_disc=rho_disc, n_disc=n_disc, n_reg=int(len(reg_idx)),
                n_succ_cells=int(len(succ_idx)), n_multi=n_multi, n_deg=int(deg.sum()), hok_agree=hok_agree,
                kinds=kinds, top_forb={kd: bool(pkg[0].forbidden.get(kd, False)) for kd in kinds},
                room_forb={kd: float(len(forb_idx[kd]) / N) for kd in kinds},
                top_rep_success=bool(rep['success'][top]), dens=dens_out, n_val=len(samp), val=dict(val),
                mism=dict(mism), secs=round(time.monotonic() - t0, 3))
    return meta


def main():
    shard = int(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    t0 = time.monotonic()
    import os
    npz_dir = f'{HERE}/cache/npz'
    os.makedirs(npz_dir, exist_ok=True)
    if shard == 0:
        print('vM table check (max |F_coarse - F_fine|):', vm_check(), flush=True)
    out = f'{HERE}/cache/meta_shard{shard}.jsonl'
    skipped = collections.Counter()
    n_all = 0
    with open(out, 'w') as f:
        for idx, (cell, label, ref) in enumerate(plan()):
            if idx % 2 != shard:
                continue
            th = H.recorded_config(ref)[1].thresholds
            tally = collections.Counter()
            n = std = other = 0
            for rec in H.records(ref, tally=tally):
                if label == 'watch' and rec.situation != 'audience':
                    other += 1
                    continue
                if not rec.solver_step:
                    std += 1
                    continue
                meta = one(cell, label, rec, th, npz_dir)
                f.write(json.dumps(meta) + '\n')
                n += 1
                if limit and n >= limit:
                    break
            f.flush()
            n_all += n
            skipped[(cell, 'standard')] += std
            skipped[(cell, 'other_situation')] += other
            print(f'{time.monotonic() - t0:8.1f} s  {cell:15s} {ref.folder}  goal {n}  standard {std}  '
                  f'other-situation {other}  {dict(tally)}', flush=True)
    with open(f'{HERE}/cache/skipped_shard{shard}.json', 'w') as f:
        json.dump({f'{c}|{w}': v for (c, w), v in skipped.items()}, f, indent=1)
    print(f'done: {n_all} records in {time.monotonic() - t0:.1f} s', flush=True)


if __name__ == '__main__':
    main()
