"""I14 v0: the scoreboard for sets and densities, with chance baselines. Definitions: METRIC_NOTE.md (frozen first).

Reads runs and scenes only through research_harness. No model call, no network, nothing tuned. One process per
group of runs (at most 2 at once); writes one JSON line per goal-step record to rows_<group>.jsonl.

Per record:
  - the 5 recorded places (top + 4 runners-up) and their 40 jittered copies (5 cm, 8 directions, heading kept) are
    scored by the package (H.score_poses);
  - F = sgi.solver.floor(observation, the record's people, the run's thresholds).selectable; the disc = F cells
    within 3 m of the recorded target point; each disc cell, facing the recorded target, is scored by a vectorised
    replica of sgi.evaluate.score (in region, heading error, reachable, forbidden kinds), checked against
    H.score_poses on a sample of 150 uniform + up to 100 in-region + up to 50 just-outside cells;
  - 1,000 draws of R5, F5, AR; AF once (deterministic).

    cd WT && OMP_NUM_THREADS=2 ... nice -n 10 .venv/bin/python <this> GROUP [LIMIT]
    GROUP: new | oldwatch | test
"""
from __future__ import annotations

import json
import math
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402
from sgi import solver  # noqa: E402
from sgi.types import DetectedPerson  # noqa: E402

HERE = 'results/2026-10-17-beyond-vlm/phase1/i14_v0'
SEED = 20261017
DRAWS = 1000
DISC_M = 3.0
JITTER_M = 0.05
JITTER_DIRS = 8
TAUS = (30.0, 45.0, 60.0)
K = 5
EPS = 1e-9                                         # the package's boundary tolerance (sgi.polygons._EPS)
CHUNK = 250                                        # draws per block (memory)

NEW = [('s21', 'qwen38_low'), ('s23', 'glm53'), ('s23', 'dsv41'), ('s23', 'haiku55'), ('s23', 'sonnet55_low'),
       ('s23', 'gemini38_low')]
OLD = [('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41')]
REAL_ARMS = ('real', 'staffed', 'plain')
DEV_ARMS = ('dev_seed0', 'dev_seed1', 'dev_seed2')


def plan(group):
    """(set label, RunRef) in processing order."""
    out = []
    if group in ('new', 'test'):
        for stage, model in NEW:
            for arm in REAL_ARMS:
                refs = H.runs(stage=stage, arm=arm, model=model, method='con')
                assert len(refs) == 1, (stage, arm, model, refs)
                out.append(('new', refs[0]))
    if group in ('oldwatch', 'test'):
        for stage, model in OLD:
            for arm in REAL_ARMS:
                refs = H.runs(stage=stage, arm=arm, model=model, method='con')
                assert len(refs) == 1, (stage, arm, model, refs)
                out.append(('old', refs[0]))
        for stage, model in OLD:
            for arm in DEV_ARMS:
                refs = H.runs(stage=stage, arm=arm, model=model, method='con')
                assert len(refs) == 1, (stage, arm, model, refs)
                out.append(('watch', refs[0]))
    return out


# ------------------------------------------------------------------------------------------- the scorer's replica
def wrap(a):
    """sgi.geometry.wrap_angle, elementwise."""
    return math.pi - np.mod(math.pi - a, 2 * math.pi)


def pip(px, py, poly):
    """sgi.polygons.point_in_polygon, elementwise: even-odd, the boundary within EPS inside."""
    v = np.array([(float(p[0]), float(p[1])) for p in poly], dtype=np.float64)
    a = np.roll(v, 1, axis=0)                       # v[i - 1]
    ax, ay, bx, by = a[:, 0], a[:, 1], v[:, 0], v[:, 1]
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    out = np.zeros(len(px), dtype=bool)
    for lo in range(0, len(px), 2000):
        X, Y = px[lo:lo + 2000, None], py[lo:lo + 2000, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = ((X - ax) * dx + (Y - ay) * dy) / len2
            t = np.maximum(0.0, np.minimum(1.0, t))
            d = np.hypot(X - (ax + t * dx), Y - (ay + t * dy))
            d = np.where(len2 == 0.0, np.hypot(X - ax, Y - ay), d)
            onb = d.min(axis=1) <= EPS
            cond = (ay > Y) != (by > Y)
            x_cross = ax + (Y - ay) * (bx - ax) / (by - ay)
            cross = cond & (x_cross > X)
        out[lo:lo + 2000] = onb | (cross.sum(axis=1) % 2 == 1)
    return out


def replica(robot, gt, matched, grid, joined, xy, theta_world):
    """What sgi.evaluate.score gives each pose (xy (N,2), theta_world (N,)) after make_poses: dict of arrays
    inreg, err (min heading error over holding regions, radians; inf outside), reach, forb {kind: bool}, and
    heading_ok / success at the regions' own tolerance."""
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    gx, gy = xy[:, 0].astype(np.float64), xy[:, 1].astype(np.float64)
    dx, dy = gx - robot.x, gy - robot.y
    rx, ry = c * dx + s * dy, c * dy - s * dx
    rth = wrap(theta_world - robot.yaw)
    wx, wy = robot.x + c * rx - s * ry, robot.y + s * rx + c * ry
    theta = rth + robot.yaw
    n = len(gx)
    regions = [] if matched is None else [g for g in gt.acceptable_regions or [] if g.target_id in (None, matched)]
    inreg = np.zeros(n, dtype=bool)
    err = np.full(n, np.inf)
    hok = np.zeros(n, dtype=bool)
    for g in regions:
        hold = pip(wx, wy, g.polygon)
        if g.heading.yaw is not None:
            allowed = np.full(n, float(g.heading.yaw))
        else:
            fx, fy = g.heading.face_point
            allowed = wrap(np.arctan2(fy - wy, fx - wx))
        e = np.abs(wrap(theta - allowed))
        inreg |= hold
        err = np.where(hold, np.minimum(err, e), err)
        hok |= hold & (e <= math.radians(g.tolerance_deg))
    ix = np.floor((wx - grid.x0) / grid.resolution).astype(np.int64)
    iy = np.floor((wy - grid.y0) / grid.resolution).astype(np.int64)
    ny, nx = grid.free.shape
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    reach = np.zeros(n, dtype=bool)
    reach[ok] = joined[iy[ok], ix[ok]]
    forb = {}
    for g in gt.forbidden_regions or []:
        forb[g.kind] = forb.get(g.kind, np.zeros(n, dtype=bool)) | pip(wx, wy, g.polygon)
    success = (matched is not None) & reach & inreg & hok
    return dict(inreg=inreg, err=err, reach=reach, forb=forb, hok=hok, success=success)


def success_tau(rep, matched, tau):
    return (matched is not None) & rep['reach'] & rep['inreg'] & (rep['err'] <= math.radians(tau))


def face_target(robot, xy, target_xy, th):
    """The heading of sgi.stages.goal._poses at each cell: the bearing to the target, the robot's yaw when closer
    than heading_degenerate_m."""
    tx, ty = target_xy
    deg = np.hypot(tx - xy[:, 0], ty - xy[:, 1]) < th.heading_degenerate_m
    return np.where(deg, robot.yaw, wrap(np.arctan2(ty - xy[:, 1], tx - xy[:, 0])))


# ------------------------------------------------------------------------------------------------------- draws
def random_order(rng, n_draws, pool, k):
    """(n_draws, k) indices into range(pool): k distinct, uniform, in random order."""
    out = np.empty((n_draws, k), dtype=np.int64)
    for lo in range(0, n_draws, CHUNK):
        b = min(CHUNK, n_draws - lo)
        keys = rng.random((b, pool))
        part = np.argpartition(keys, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(keys, part, axis=1), axis=1)
        out[lo:lo + b] = np.take_along_axis(part, order, axis=1)
    return out


def fps(cells, start_d, k):
    """Farthest-point picks: start_d (B, N) distances from the start set; k picks, ties to the lowest index."""
    mind = start_d.copy()
    picks = np.empty((mind.shape[0], k), dtype=np.int64)
    for j in range(k):
        nxt = np.argmax(mind, axis=1)
        picks[:, j] = nxt
        p = cells[nxt]
        mind = np.minimum(mind, np.hypot(cells[None, :, 0] - p[:, None, 0], cells[None, :, 1] - p[:, None, 1]))
    return picks


def any_by_k(succ_picks):
    """(B, k) bool -> (k,) mean over draws of 'any of the first j' for j = 1..k."""
    return np.logical_or.accumulate(succ_picks, axis=1).mean(axis=0)


def exact_none(n, s, k):
    """P(no success among k distinct uniform draws from n cells holding s successes)."""
    p = 1.0
    for i in range(k):
        if n - i <= 0:
            return 0.0
        p *= max(n - s - i, 0) / (n - i)
    return p


# --------------------------------------------------------------------------------------------------- one record
def one(label, rec, th, val_counts):
    sc = H.scene(rec.ref, rec.scene_id)
    obs = sc.observation
    robot = obs.robot
    _, gt, _, cache = H._against(rec)
    places = [tuple(rec.pose_world)] + [tuple(c['pose_world']) for c in rec.candidates]
    nplace = len(places)
    angles = [2 * math.pi * d / JITTER_DIRS for d in range(JITTER_DIRS)]
    jit = [(p[0] + JITTER_M * math.cos(a), p[1] + JITTER_M * math.sin(a), p[2]) for p in places for a in angles]
    pkg = H.score_poses(rec, places + jit)
    grid, joined = cache['reach']
    matched = pkg[0].matched_target_id
    assert all(s.matched_target_id == matched for s in pkg)
    pl_pkg, jit_pkg = pkg[:nplace], pkg[nplace:]

    people = [DetectedPerson.model_validate(p) for p in rec.people]
    fl = solver.floor(obs, people, th)
    sel_idx = np.flatnonzero(fl.selectable.ravel())             # row-major grid order
    cen = fl.centres().reshape(-1, 2)[sel_idx]
    on_sel = []
    for p in places:
        cell = fl.index(p[0], p[1])
        on_sel.append(cell is not None and bool(fl.selectable[cell]))
    tx, ty = (float(v) for v in rec.target['world'][:2])
    dist = np.hypot(cen[:, 0] - tx, cen[:, 1] - ty)
    in_disc = dist <= DISC_M + EPS
    disc = cen[in_disc]
    N = len(disc)
    disc_flat = sel_idx[in_disc]

    # the recorded places by the replica (their recorded headings), checked against the package
    pxy = np.array([p[:2] for p in places], dtype=np.float64)
    pth = np.array([p[2] for p in places], dtype=np.float64)
    prep = replica(robot, gt, matched, grid, joined, pxy, pth)
    jxy = np.array([p[:2] for p in jit], dtype=np.float64)
    jth = np.array([p[2] for p in jit], dtype=np.float64)
    jrep = replica(robot, gt, matched, grid, joined, jxy, jth)
    mism = {}
    for name, rep, sco in (('places', prep, pl_pkg), ('jitter', jrep, jit_pkg)):
        for i, s in enumerate(sco):
            for f, a, b in (('success', bool(rep['success'][i]), s.success), ('inreg', bool(rep['inreg'][i]), s.in_region),
                            ('heading', bool(rep['hok'][i]), s.heading_ok), ('reach', bool(rep['reach'][i]), s.reachable)):
                val_counts[f] += 1
                if a != b:
                    mism[f'{name}.{f}'] = mism.get(f'{name}.{f}', 0) + 1
            for kind, flag in s.forbidden.items():
                val_counts['forb'] += 1
                if bool(rep['forb'].get(kind, np.zeros(len(sco), bool))[i]) != flag:
                    mism[f'{name}.forb'] = mism.get(f'{name}.forb', 0) + 1

    # the disc cells facing the recorded target, by the replica
    dth = face_target(robot, disc, (tx, ty), th)
    drep = replica(robot, gt, matched, grid, joined, disc, dth)
    # validation sample against the package
    vrng = np.random.default_rng([7, zlib.crc32(key(rec).encode())])
    samp = set(vrng.choice(N, size=min(150, N), replace=False).tolist()) if N else set()
    inr = np.flatnonzero(drep['inreg'])
    if len(inr):
        samp |= set(vrng.choice(inr, size=min(100, len(inr)), replace=False).tolist())
        outs = np.flatnonzero(~drep['inreg'])
        lo, hi = disc[inr].min(axis=0) - 0.15, disc[inr].max(axis=0) + 0.15
        box = outs[np.all((disc[outs] >= lo) & (disc[outs] <= hi), axis=1)]
        if len(box):
            dmin = np.full(len(box), np.inf)
            for b0 in range(0, len(inr), 500):
                q = disc[inr[b0:b0 + 500]]
                dmin = np.minimum(dmin, np.hypot(disc[box, None, 0] - q[None, :, 0], disc[box, None, 1] - q[None, :, 1]).min(axis=1))
            near = box[dmin <= 0.15 + EPS]
            if len(near):
                samp |= set(vrng.choice(near, size=min(50, len(near)), replace=False).tolist())
    samp = sorted(samp)
    vs = H.score_poses(rec, [(float(disc[i, 0]), float(disc[i, 1]), float(dth[i])) for i in samp])
    for i, s in zip(samp, vs):
        for f, a, b in (('success', bool(drep['success'][i]), s.success), ('inreg', bool(drep['inreg'][i]), s.in_region),
                        ('heading', bool(drep['hok'][i]), s.heading_ok), ('reach', bool(drep['reach'][i]), s.reachable)):
            val_counts[f] += 1
            if a != b:
                mism[f'disc.{f}'] = mism.get(f'disc.{f}', 0) + 1
        for kind, flag in s.forbidden.items():
            val_counts['forb'] += 1
            if bool(drep['forb'].get(kind, np.zeros(N, bool))[i]) != flag:
                mism['disc.forb'] = mism.get('disc.forb', 0) + 1

    # whole room: region share of all F cells (bbox first, then the replica's point-in-polygon)
    room_inreg = 0
    if matched is not None:
        regions = [g for g in gt.acceptable_regions or [] if g.target_id in (None, matched)]
        hold = np.zeros(len(cen), dtype=bool)
        for g in regions:
            v = np.array(g.polygon, dtype=np.float64)
            lo, hi = v.min(axis=0) - 1e-6, v.max(axis=0) + 1e-6
            boxm = np.all((cen >= lo) & (cen <= hi), axis=1)
            hold[boxm] |= pip(cen[boxm, 0], cen[boxm, 1], g.polygon)
        room_inreg = int(hold.sum())
        disc_inreg_check = int(hold[in_disc].sum())
        assert disc_inreg_check == int(drep['inreg'].sum()), (disc_inreg_check, int(drep['inreg'].sum()))

    # outcomes per tolerance
    S = {tau: success_tau(drep, matched, tau) for tau in TAUS}
    P = {tau: success_tau(prep, matched, tau) for tau in TAUS}
    J = {tau: success_tau(jrep, matched, tau) for tau in TAUS}
    assert np.array_equal(P[45.0], prep['success'])
    row = dict(set=label, stage=rec.ref.stage, arm=rec.arm, model=rec.model, run=rec.run, scene=rec.scene_id,
               ep=rec.episode_id, trial=rec.trial, sit=rec.situation, matched=matched,
               target_ok=bool(pkg[0].target_correct), nplace=nplace, on_sel=on_sel,
               rec_success=bool(rec.score.get('success')), n_sel=int(len(cen)), n_disc=int(N),
               top_in_disc=bool(math.hypot(places[0][0] - tx, places[0][1] - ty) <= DISC_M + EPS),
               region_sel=room_inreg, region_disc=int(drep['inreg'].sum()),
               rho_disc=(float(drep['inreg'].mean()) if N else 0.0),
               rho_room=(room_inreg / len(cen) if len(cen) else 0.0),
               sigma_disc=(float(S[45.0].mean()) if N else 0.0), n_val=len(samp), mism=mism)
    # the solver's set (package verdicts at 45; replica for 30 and 60)
    pk = np.array([s.success for s in pl_pkg])
    pin = np.array([s.in_region for s in pl_pkg])
    row['solver'] = {str(int(t)): dict(hit=np.logical_or.accumulate(P[t]).astype(int).tolist(),
                                     p5=float(P[t].mean())) for t in TAUS}
    row['solver']['45']['hit_pkg'] = np.logical_or.accumulate(pk).astype(int).tolist()
    row['solver_inreg_hit'] = np.logical_or.accumulate(pin).astype(int).tolist()
    # forbidden: the set, the top, R5's expectation (the disc share)
    kinds = sorted({g.kind for g in gt.forbidden_regions or []})
    row['forb'] = {kd: dict(set=float(np.mean([s.forbidden.get(kd, False) for s in pl_pkg])),
                            top=bool(pl_pkg[0].forbidden.get(kd, False)),
                            r5=(float(drep['forb'][kd].mean()) if N else 0.0)) for kd in kinds}
    # jitter (package at 45; replica at 30/60)
    jp = np.array([s.success for s in jit_pkg]).reshape(nplace, JITTER_DIRS)
    jt = {}
    for t in TAUS:
        js = jp if t == 45.0 else J[t].reshape(nplace, JITTER_DIRS)
        top0 = bool(P[t][0])
        per_dir_hit = js.any(axis=0)
        jt[str(int(t))] = dict(top_mean=float(js[0].mean()), top_flip=int(bool((js[0] != top0).any())),
                               top_all_kept=int(bool(top0 and js[0].all())),
                               hit5_mean=float(per_dir_hit.mean()), hit5_min=int(per_dir_hit.min()),
                               set_flip=int(bool((js != P[t][:, None]).any())))
    row['jitter'] = jt

    # draws
    rng = np.random.default_rng([SEED, zlib.crc32(key(rec).encode())])
    base = {}
    if N >= K:
        r5 = random_order(rng, DRAWS, N, K)
        first = rng.integers(N, size=DRAWS)
        f5 = np.empty((DRAWS, K), dtype=np.int64)
        for lo in range(0, DRAWS, CHUNK):
            st = first[lo:lo + CHUNK]
            d0 = np.hypot(disc[None, :, 0] - disc[st, None, 0], disc[None, :, 1] - disc[st, None, 1])
            f5[lo:lo + CHUNK, 0] = st
            f5[lo:lo + CHUNK, 1:] = fps(disc, d0, K - 1)
        top_cell = fl.index(places[0][0], places[0][1])
        top_flat = None if top_cell is None else top_cell[0] * fl.selectable.shape[1] + top_cell[1]
        pos = np.flatnonzero(disc_flat == top_flat) if top_flat is not None else np.array([], dtype=np.int64)
        top_d = int(pos[0]) if len(pos) else None
        pool = np.arange(N) if top_d is None else np.delete(np.arange(N), top_d)
        ar = pool[random_order(rng, DRAWS, len(pool), K - 1)]
        af = fps(disc, np.hypot(disc[:, 0] - places[0][0], disc[:, 1] - places[0][1])[None, :], K - 1)[0]
        row['top_disc_idx'] = top_d
        for t in TAUS:
            s, top_t = S[t], bool(P[t][0])
            ex_r5 = 1.0 - exact_none(N, int(s.sum()), K)
            if top_t:
                ex_ar = 1.0
            else:
                s_pool = int(s.sum()) - (int(s[top_d]) if top_d is not None else 0)
                ex_ar = 1.0 - exact_none(len(pool), s_pool, K - 1)
            ar_hit = np.concatenate([np.full((DRAWS, 1), top_t), s[ar]], axis=1)
            af_hit = np.concatenate([[top_t], s[af]])
            base[str(int(t))] = dict(
                r5=any_by_k(s[r5]).tolist(), f5=any_by_k(s[f5]).tolist(), ar=any_by_k(ar_hit).tolist(),
                af=np.logical_or.accumulate(af_hit).astype(float).tolist(),
                r5_p5=float(s[r5].mean()), f5_p5=float(s[f5].mean()), ex_r5=ex_r5, ex_ar=ex_ar)
        ir = drep['inreg']
        base['inreg'] = dict(r5=any_by_k(ir[r5]).tolist(), f5=any_by_k(ir[f5]).tolist(),
                             ar=any_by_k(np.concatenate([np.full((DRAWS, 1), bool(pin[0])), ir[ar]], axis=1)).tolist())
    row['base'] = base
    return row


def key(rec):
    return f"{rec.ref.folder}/{rec.scene_id}/{rec.episode_id}/{rec.trial}"


def main():
    group = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    started = time.monotonic()
    import collections
    val_counts = collections.Counter()
    out_path = f'{HERE}/rows_{group}.jsonl'
    n_all = 0
    with open(out_path, 'w') as f:
        for label, ref in plan(group):
            th = H.recorded_config(ref)[1].thresholds
            tally = collections.Counter()
            n = std = 0
            for rec in H.records(ref, tally=tally):
                if label == 'watch' and rec.situation != 'audience':
                    continue
                if not rec.solver_step:
                    std += 1
                    continue
                row = one(label, rec, th, val_counts)
                f.write(json.dumps(row) + '\n')
                n += 1
                if limit and n >= limit:
                    break
            f.flush()
            n_all += n
            print(f"{time.monotonic() - started:8.1f} s  {label:5s} {ref.folder}  goal steps {n}  standard {std}  "
                  f"{dict(tally)}", flush=True)
    print(f"done: {n_all} rows, checks {dict(val_counts)}, {time.monotonic() - started:.1f} s", flush=True)


if __name__ == '__main__':
    main()
