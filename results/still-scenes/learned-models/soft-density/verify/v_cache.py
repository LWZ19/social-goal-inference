"""I04 independent verifier, step 1: per-record rebuild of floor, keys, headings, per-cell outcomes, and every
per-setting quantity the frozen note needs (FROZEN.md sha256 8e0aac98...1882).

Written by the verifier, not importing the compute agent's code. Imports: the package, the research harness and
I14 v0's replica (phase1/i14_v0/i14.py: replica(), pip(), wrap()).

Per record it computes, for every setting of V1 (gamma x T x kappa), V2 (same grid) and V3 (sigma x kappa):
S, in-region mass, concentration within 0.3 m of the top place and mass in each forbidden kind; plus V1/V2 density
place lists per gamma and V3's, scored by the replica and by the package. Writes vcache_shard<k>.pkl.

    cd WT && PYTHONPATH= OMP_NUM_THREADS=2 ... nice -n 10 .venv/bin/python v_cache.py SHARD NSHARDS [LIMIT]
"""
from __future__ import annotations

import collections
import importlib.util
import math
import pickle
import re
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402
from sgi import solver as S  # noqa: E402
from sgi.geometry import make_poses, wrap_angle  # noqa: E402
from sgi.stages.goal import standard_goal  # noqa: E402
from sgi.types import GoalResult  # noqa: E402

OUT = "/tmp" if "--test" in sys.argv else "results/2026-10-17-beyond-vlm/phase1d/i04/verify"
spec = importlib.util.spec_from_file_location(
    'i14v0', 'results/2026-10-17-beyond-vlm/phase1/i14_v0/i14.py')
I14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(I14)

EPS = 1e-9
GAMMAS = (1, 3, 10, 30)
TS = np.geomspace(0.005, 5, 25)
SIGMAS = np.geomspace(0.05, 3, 25)
KAPPAS = (4.0, 8.0, 16.0, 32.0, math.inf)
TOL_DEG = 45.0
DISC_M = 3.0
CONC_M = 0.3

CELLS = [('old_qwen', 's19', 'qwen38_low', 'real'), ('old_glm', 's19', 'glm53', 'real'),
         ('old_deepseek', 's20', 'dsv41', 'real'),
         ('watch_qwen', 's19', 'qwen38_low', 'watch'), ('watch_glm', 's19', 'glm53', 'watch'),
         ('watch_deepseek', 's20', 'dsv41', 'watch'),
         ('new_qwen', 's21', 'qwen38_low', 'real'), ('new_glm', 's23', 'glm53', 'real'),
         ('new_deepseek', 's23', 'dsv41', 'real'), ('new_haiku', 's23', 'haiku55', 'real'),
         ('new_sonnet', 's23', 'sonnet55_low', 'real'), ('new_gemini', 's23', 'gemini38_low', 'real')]


def plan():
    out = []
    for cell, stage, model, kind in CELLS:
        arms = ('real', 'staffed', 'plain') if kind == 'real' else ('dev_seed0', 'dev_seed1', 'dev_seed2')
        for arm in arms:
            refs = H.runs(stage=stage, arm=arm, model=model, method='con')
            assert len(refs) == 1, (stage, arm, model, refs)
            out.append((cell, kind, refs[0]))
    return out


# ------------------------------------------------------------------ von Mises arc mass (the note's numerics)
def vm_table(kappa, n):
    y = np.linspace(-math.pi, math.pi, n)
    f = np.exp(kappa * (np.cos(y) - 1.0))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(y))])
    return y, cum / cum[-1]


TABLES = {k: vm_table(k, 2 ** 16 + 1) for k in KAPPAS if math.isfinite(k)}


def cdf(kappa, x):
    y, F = TABLES[kappa]
    return np.interp(x, y, F)


def arc_mass(kappa, lo, hi):
    """vM(0, kappa) mass on the arc [lo, hi] (hi - lo <= 2 pi), lo/hi any reals."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    # shift so that lo lies in [-pi, pi)
    k = np.floor((lo + math.pi) / (2 * math.pi))
    lo2 = lo - 2 * math.pi * k
    hi2 = hi - 2 * math.pi * k
    inside = np.minimum(hi2, math.pi)
    m = cdf(kappa, inside) - cdf(kappa, lo2)
    over = hi2 > math.pi
    m = np.where(over, m + cdf(kappa, np.minimum(hi2 - 2 * math.pi, math.pi)) - 0.0, m)
    return m


def bessel_ratio_series_cdf_check():
    """Independent check of the table: F(x) = (x+pi)/(2pi) + (1/pi) sum_n (I_n/I_0) sin(nx)/n, ratios by Miller's
    backward recurrence. Returns max |table - series| over a grid of x, per kappa."""
    xs = np.linspace(-math.pi, math.pi, 4001)
    out = {}
    for kappa in TABLES:
        N = 400
        r = np.zeros(N + 2)
        for n in range(N, 0, -1):
            r[n] = 1.0 / (2.0 * n / kappa + r[n + 1])          # r[n] = I_n / I_{n-1}
        ratios = np.cumprod(r[1:N + 1])                         # I_n / I_0
        n = np.arange(1, N + 1)
        series = (xs + math.pi) / (2 * math.pi) + (np.sin(np.outer(xs, n)) / n * ratios).sum(axis=1) / math.pi
        out[kappa] = float(np.max(np.abs(cdf(kappa, xs) - series)))
    fine = {}
    for kappa in TABLES:
        y4, F4 = vm_table(kappa, 4 * 2 ** 16 + 1)
        fine[kappa] = float(np.max(np.abs(cdf(kappa, xs) - np.interp(xs, y4, F4))))
    return out, fine


# ---------------------------------------------------------------------------------------------- helpers
def layout_of(kind, rec):
    if kind == 'watch':
        return f'{rec.arm}/{rec.scene_id}'
    m = re.search(r'(eval_\d+)', rec.scene_id)
    return m.group(1)


def key_of(rec):
    return f'{rec.ref.folder}/{rec.scene_id}/{rec.episode_id}/{rec.trial}'


def lse_mass(neg_e_over_t):
    """Normalised mass from log-weights (float64 log-sum-exp)."""
    m = neg_e_over_t.max()
    w = np.exp(neg_e_over_t - m)
    return w / w.sum()


def density_places(order, cells, first, allowed=None, k=5, min_d=0.6):
    """Place 1 = first; then the first cell in `order` (best density first) at least min_d - 1e-9 from all
    earlier places (and in `allowed` if given). Stops early if none is left."""
    places = [int(first)]
    free = np.hypot(cells[:, 0] - cells[first, 0], cells[:, 1] - cells[first, 1]) >= min_d - EPS
    if allowed is not None:
        free &= allowed
    while len(places) < k:
        cand = order[free[order]]
        if len(cand) == 0:
            break
        nxt = int(cand[0])
        places.append(nxt)
        free &= np.hypot(cells[:, 0] - cells[nxt, 0], cells[:, 1] - cells[nxt, 1]) >= min_d - EPS
    return places


# --------------------------------------------------------------------------------------------- one record
def one(cell_label, kind, rec, counts, mism):
    sc, gt, th, cache = H._against(rec)
    obs = sc.observation
    robot = obs.robot
    result = GoalResult.model_validate(rec.result)
    target = result.target
    people = result.people
    constraints = [use.constraint for use in result.constraints]
    std = standard_goal(obs, target, th)
    cells, settled = S._over_floor(obs, target, people, constraints, std, th)
    N = len(cells)
    keys = settled.keys
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    wx = robot.x + c * cells[:, 0] - s * cells[:, 1]
    wy = robot.y + s * cells[:, 0] + c * cells[:, 1]
    # the solver's heading rule at every cell
    toward = settled.faced_xy[None, :] - cells
    degen = np.hypot(toward[:, 0], toward[:, 1]) < th.heading_degenerate_m
    theta_raw = np.where(degen, robot.yaw, robot.yaw + np.arctan2(toward[:, 1], toward[:, 0]))
    theta_w = math.pi - np.mod(math.pi - theta_raw, 2 * math.pi)        # make_poses' wrap_angle
    # reproduction: picks vs recorded places
    picks = list(settled.picks)
    rec_places = [tuple(rec.pose_world)] + [tuple(cc['pose_world']) for cc in rec.candidates]
    worst_m, worst_h = 0.0, 0.0
    for i, p in enumerate(picks[:5]):
        pw, _ = make_poses(robot, (wx[p], wy[p]), theta_raw[p])
        # also via the solver's own scalar path for the heading
        if i < len(rec_places):
            worst_m = max(worst_m, math.dist(pw[:2], rec_places[i][:2]))
            worst_h = max(worst_h, abs(wrap_angle(pw[2] - rec_places[i][2])))
    repro = dict(npicks=len(picks), nrec=len(rec_places), worst_m=worst_m, worst_h=worst_h,
                 ok=(len(picks) >= len(rec_places) and worst_m <= 1e-6))
    top = picks[0]
    # package verdicts on the recorded places
    pkg_places = H.score_poses(rec, rec_places)
    matched = pkg_places[0].matched_target_id
    assert all(p.matched_target_id == matched for p in pkg_places)
    top1 = bool(pkg_places[0].success)
    grid, joined = cache['reach']
    # replica over all F cells at the solver heading
    xy = np.stack([wx, wy], axis=1)
    rep = I14.replica(robot, gt, matched, grid, joined, xy, theta_w)
    # own per-region hold and signed offsets (for the arcs)
    rth = I14.wrap(theta_w - robot.yaw)
    rx, ry = c * (wx - robot.x) + s * (wy - robot.y), c * (wy - robot.y) - s * (wx - robot.x)
    gx, gy = robot.x + c * rx - s * ry, robot.y + s * rx + c * ry               # the scorer's re-transformed goal
    th_s = rth + robot.yaw
    regions = [] if matched is None else [g for g in gt.acceptable_regions or [] if g.target_id in (None, matched)]
    holds, offs = [], []
    for g in regions:
        assert abs(g.tolerance_deg - TOL_DEG) < 1e-12, g.tolerance_deg
        h = I14.pip(gx, gy, g.polygon)
        if g.heading.yaw is not None:
            allowed = np.full(N, float(g.heading.yaw))
        else:
            fx, fy = g.heading.face_point
            allowed = I14.wrap(np.arctan2(fy - gy, fx - gx))
        holds.append(h)
        offs.append(I14.wrap(allowed - th_s))                                  # d: allowed relative to heading
    nhold = np.sum(holds, axis=0) if holds else np.zeros(N, dtype=int)
    inreg = nhold > 0
    assert np.array_equal(inreg, rep['inreg'])
    tolr = math.radians(TOL_DEG)
    if holds:
        err_own = np.min([np.where(h, np.abs(d), np.inf) for h, d in zip(holds, offs)], axis=0)
        assert np.allclose(np.where(inreg, err_own, 0), np.where(inreg, rep['err'], 0), atol=1e-12, rtol=0)
    base = (matched is not None) & rep['reach'] & inreg
    multi = base & (nhold > 1)
    counts['multi_region_cells'] += int(multi.sum())
    counts['multi_region_cells_any'] += int((nhold > 1).sum())
    psucc = {}
    for kappa in KAPPAS:
        p = np.zeros(N)
        if holds:
            if math.isinf(kappa):
                ok = np.zeros(N, dtype=bool)
                for h, d in zip(holds, offs):
                    ok |= h & (np.abs(d) <= tolr)
                p = ok.astype(np.float64)
            else:
                single = base & (nhold == 1)
                dsel = np.zeros(N)
                for h, d in zip(holds, offs):
                    dsel = np.where(h, d, dsel)
                p[single] = arc_mass(kappa, dsel[single] - tolr, dsel[single] + tolr)
                for i in np.flatnonzero(multi):                           # union of arcs, by fine sampling
                    yy = np.linspace(-math.pi, math.pi, 2 ** 18 + 1)
                    f = np.exp(kappa * (np.cos(yy) - 1.0))
                    cov = np.zeros(len(yy), dtype=bool)
                    for h, d in zip(holds, offs):
                        if h[i]:
                            cov |= np.abs(I14.wrap(yy - d[i])) <= tolr
                    p[i] = np.trapezoid(f * cov, yy) / np.trapezoid(f, yy)
        psucc[kappa] = np.where(base, p, 0.0)
    assert np.array_equal(psucc[math.inf] > 0, rep['success']), 'kappa-inf indicator vs replica success'
    counts['kinf_eq_replica'] += 1
    # replica success at the top cell == official top-1
    counts['top_replica_eq_official'] += int(bool(rep['success'][top]) == top1)
    counts['official_eq_recorded'] += int(top1 == bool(rec.score.get('success')))
    # ---------------------------------------------------------------- validation sample against the package
    vr = np.random.default_rng([4242, zlib.crc32(key_of(rec).encode())])
    samp = set(vr.choice(N, size=min(50, N), replace=False).tolist())
    inr = np.flatnonzero(inreg)
    if len(inr):
        samp |= set(vr.choice(inr, size=min(30, len(inr)), replace=False).tolist())
        outs = np.flatnonzero(~inreg)
        lo, hi = cells[inr].min(axis=0) - 0.16, cells[inr].max(axis=0) + 0.16
        box = outs[np.all((cells[outs] >= lo) & (cells[outs] <= hi), axis=1)]
        if len(box):
            dmin = np.full(len(box), np.inf)
            for b0 in range(0, len(inr), 500):
                q = cells[inr[b0:b0 + 500]]
                dmin = np.minimum(dmin, np.hypot(cells[box, None, 0] - q[None, :, 0],
                                                 cells[box, None, 1] - q[None, :, 1]).min(axis=1))
            near = box[dmin <= 0.15 + EPS]
            if len(near):
                samp |= set(vr.choice(near, size=min(20, len(near)), replace=False).tolist())
    samp = sorted(samp)
    sco = H.score_poses(rec, [(float(wx[i]), float(wy[i]), float(theta_w[i])) for i in samp])
    from sgi.geometry import bearing
    for i, sp in zip(samp, sco):
        for f, a, b in (('success', bool(rep['success'][i]), sp.success), ('inreg', bool(inreg[i]), sp.in_region),
                        ('heading', bool(rep['hok'][i]), sp.heading_ok), ('reach', bool(rep['reach'][i]), sp.reachable),
                        ('kinf', bool(psucc[math.inf][i] > 0), sp.success)):
            counts[f'samp.{f}'] += 1
            if a != b:
                mism[f'samp.{f}'] += 1
        for kd, flag in sp.forbidden.items():
            counts['samp.forb'] += 1
            if bool(rep['forb'].get(kd, np.zeros(N, bool))[i]) != flag:
                mism['samp.forb'] += 1
        # the heading offset itself, by the package's own scalar formula
        goal = (float(gx[i]), float(gy[i]))
        for g, h, d in zip(regions, holds, offs):
            if h[i]:
                allowed = g.heading.yaw if g.heading.yaw is not None else bearing(goal, g.heading.face_point)
                e = abs(wrap_angle(float(th_s[i]) - allowed))
                counts['samp.offset'] += 1
                if abs(e - abs(float(d[i]))) > 1e-9:
                    mism['samp.offset'] += 1
    # recorded places: replica at recorded headings vs package
    pxy = np.array([p[:2] for p in rec_places])
    pth = np.array([p[2] for p in rec_places])
    prep = I14.replica(robot, gt, matched, grid, joined, pxy, pth)
    for i, sp in enumerate(pkg_places):
        counts['places'] += 1
        if bool(prep['success'][i]) != sp.success or bool(prep['inreg'][i]) != sp.in_region:
            mism['places'] += 1
    lex_hit = np.logical_or.accumulate([p.success for p in pkg_places]).astype(int).tolist()
    # ------------------------------------------------------------------------------ keys and energies
    k3 = keys.get('turn_m', keys.get('face_m'))
    assert not ('turn_m' in keys and 'face_m' in keys)
    slots = [keys['excluded_m'], keys['excluded_total_m'], np.zeros(N) if k3 is None else k3,
             keys['preference_m'], keys['standard_m']]
    slots_mm = [np.rint(np.asarray(v) * 1000).astype(np.int64) for v in slots]
    for v, vm in zip(slots, slots_mm):
        assert np.array_equal(vm / 1000.0, np.asarray(v, dtype=np.float64)), 'mm round trip'
    # V2 kept cells: _first's rule over the first two keys
    ex, ext = slots[0], slots[1]
    kept = ex <= ex.min() + EPS
    kept &= ext <= ext[kept].min() + EPS
    assert kept[top]
    # distances
    dtop = np.hypot(cells[:, 0] - cells[top, 0], cells[:, 1] - cells[top, 1])
    near_top = dtop <= CONC_M + EPS
    d2top = dtop ** 2
    res = th.grid_resolution_m
    gidx = np.rint(cells / res).astype(np.int64)
    assert np.allclose(gidx * res, cells, atol=1e-9)
    d2int = ((gidx - gidx[top]) ** 2).sum(axis=1)
    forb_kinds = sorted({g.kind for g in gt.forbidden_regions or []})
    forb = {kd: rep['forb'][kd].astype(np.float64) for kd in forb_kinds}
    inreg_f = inreg.astype(np.float64)
    P = np.stack([psucc[k] for k in KAPPAS], axis=1)                         # (N, 5)
    # ---------------------------------------------------------------------- V1/V2 per setting
    out = {}
    for var in ('V1', 'V2'):
        Sx = np.zeros((len(GAMMAS), len(TS), len(KAPPAS)))
        IR = np.zeros((len(GAMMAS), len(TS)))
        CO = np.zeros((len(GAMMAS), len(TS)))
        FB = {kd: np.zeros((len(GAMMAS), len(TS))) for kd in forb_kinds}
        for gi, gm in enumerate(GAMMAS):
            emm = sum((gm ** (4 - j)) * slots_mm[j] for j in range(5))      # int64, exact
            E = emm / 1000.0
            sel = np.ones(N, dtype=bool) if var == 'V1' else kept
            Es = E[sel]
            for ti, T in enumerate(TS):
                p = lse_mass(-(Es - Es.min()) / T)
                Sx[gi, ti] = p @ P[sel]
                IR[gi, ti] = p @ inreg_f[sel]
                CO[gi, ti] = p[near_top[sel]].sum()
                for kd in forb_kinds:
                    FB[kd][gi, ti] = p @ forb[kd][sel]
        out[var] = dict(S=Sx, inreg=IR, conc=CO, forb=FB)
    Sx = np.zeros((len(SIGMAS), len(KAPPAS)))
    IR = np.zeros(len(SIGMAS))
    CO = np.zeros(len(SIGMAS))
    FB = {kd: np.zeros(len(SIGMAS)) for kd in forb_kinds}
    for si, sg in enumerate(SIGMAS):
        p = lse_mass(-d2top / (2 * sg * sg))
        Sx[si] = p @ P
        IR[si] = p @ inreg_f
        CO[si] = p[near_top].sum()
        for kd in forb_kinds:
            FB[kd][si] = p @ forb[kd]
    out['V3'] = dict(S=Sx, inreg=IR, conc=CO, forb=FB)
    # --------------------------------------------------------------------------- density places
    idx = np.arange(N)
    dens = {}
    allpl = set()
    for var in ('V1', 'V2'):
        for gm in GAMMAS:
            emm = sum((gm ** (4 - j)) * slots_mm[j] for j in range(5))
            order = np.lexsort((idx, emm))
            pl = density_places(order, cells, top, allowed=(kept if var == 'V2' else None))
            dens[(var, gm)] = pl
            allpl.update(pl)
    order3 = np.lexsort((idx, d2int))
    order3f = np.lexsort((idx, d2top))
    dens[('V3', None)] = density_places(order3, cells, top)
    dens[('V3f', None)] = density_places(order3f, cells, top)
    allpl.update(dens[('V3', None)])
    allpl = sorted(allpl)
    psc = H.score_poses(rec, [(float(wx[i]), float(wy[i]), float(theta_w[i])) for i in allpl])
    pk = {i: sp.success for i, sp in zip(allpl, psc)}
    for i in allpl:
        counts['dens'] += 1
        if bool(rep['success'][i]) != pk[i]:
            mism['dens'] += 1
    dens_hit = {}
    for kk, pl in dens.items():
        succ = [bool(rep['success'][i]) for i in pl]
        hit = np.logical_or.accumulate(succ).astype(int).tolist()
        hit = hit + [hit[-1]] * (5 - len(hit))                                   # carry forward
        dens_hit[kk] = dict(places=pl, hit=hit)
    # -------------------------------------------------------------------------- scalars
    tx, ty = (float(v) for v in rec.target['world'][:2])
    disc = np.hypot(wx - tx, wy - ty) <= DISC_M + EPS
    rho_room = float(inreg.mean()) if N else 0.0
    rho_disc = float(inreg[disc].mean()) if disc.any() else 0.0
    flags = rec.result.get('flags') or {}
    row = dict(key=key_of(rec), cell=cell_label, kind=kind, layout=layout_of(kind, rec), sit=rec.situation,
               scene=rec.scene_id, arm=rec.arm, run=rec.run, matched=matched, target_ok=matched is not None,
               top1=top1, rec_success=bool(rec.score.get('success')), N=N, n_disc=int(disc.sum()),
               rho_room=rho_room, rho_disc=rho_disc, region_cells=int(inreg.sum()),
               inreg_top_rep=bool(inreg[top]), inreg_top_pkg=bool(pkg_places[0].in_region),
               forb_kinds=forb_kinds,
               forb_top_pkg={kd: bool(pkg_places[0].forbidden.get(kd, False)) for kd in forb_kinds},
               forb_room={kd: float(forb[kd].mean()) for kd in forb_kinds},
               flags={f: bool(flags.get(f, False)) for f in ('exclusion_broken', 'far_from_target', 'constraint_unusable')},
               repro=repro, lex_hit=lex_hit, dens=dens_hit, out=out, n_kept=int(kept.sum()),
               n_degen=int(degen.sum()), n_multi=int(multi.sum()), n_val=len(samp))
    return row


def main():
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "--test" else None
    t0 = time.monotonic()
    counts, mism = collections.Counter(), collections.Counter()
    rows = []
    std_counts = collections.Counter()
    other_sit = collections.Counter()
    pl = plan()
    for j, (cell, kind, ref) in enumerate(pl):
        if j % nsh != shard:
            continue
        n = 0
        for rec in H.records(ref):
            if kind == 'watch' and rec.situation != 'audience':
                other_sit[cell] += 1
                continue
            if not rec.solver_step:
                std_counts[cell] += 1
                continue
            rows.append(one(cell, kind, rec, counts, mism))
            n += 1
            if limit and n >= limit:
                break
        print(f'{time.monotonic() - t0:8.1f} s {cell:15s} {ref.folder} goal {n}', flush=True)
    if shard == 0:
        series, fine = bessel_ratio_series_cdf_check()
        print('vM table vs Bessel series (max abs):', series, flush=True)
        print('vM table vs 4x finer table (max abs):', fine, flush=True)
    with open(f'{OUT}/vcache_shard{shard}.pkl', 'wb') as f:
        pickle.dump(dict(rows=rows, counts=dict(counts), mism=dict(mism), std=dict(std_counts),
                         other_sit=dict(other_sit)), f)
    print(f'done {len(rows)} rows in {time.monotonic() - t0:.1f} s; counts {dict(counts)}; mism {dict(mism)}; '
          f'standard {dict(std_counts)}; other-situation {dict(other_sit)}', flush=True)


if __name__ == '__main__':
    main()
