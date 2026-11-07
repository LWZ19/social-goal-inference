"""Independent verifier for i14_v0. Written before reading i14.py.

Per goal-step record: the 5 recorded places scored by the package (H.score_poses), jitter, heading curve; the
solver's selectable floor (sgi.solver.floor with the record's own people, recorded thresholds); every selectable
cell scored by the package with the face-target heading (no replica); then exact hypergeometric R5 / AR (disc and
room), deterministic AF, Monte Carlo F5 (200 draws), lift shares. Writes verify/rows_<group>.jsonl.

usage: v_main.py <group>   group in {old, watch, new_a, new_b, smoke}
"""
import sys, json, math, time, zlib
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
import numpy as np
from math import comb
from sgi import solver
from sgi.types import DetectedPerson
from sgi.geometry import bearing, wrap_angle
from sgi.polygons import point_in_polygon

OUT = 'results/still-scenes/metrics-and-statistics/metric-v0/verify'
REAL = ('real', 'staffed', 'plain')
SEEDS = ('dev_seed0', 'dev_seed1', 'dev_seed2')


def cells_for(group):
    out = []
    if group in ('old', 'smoke'):
        for st, m in (('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41')):
            for a in REAL:
                out.append(('old', m, H.runs(stage=st, arm=a, model=m, method='con')))
    if group == 'watch':
        for st, m in (('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41')):
            for a in SEEDS:
                out.append(('watch', m, H.runs(stage=st, arm=a, model=m, method='con')))
    if group == 'new_a':
        for st, m in (('s21', 'qwen38_low'), ('s23', 'glm53'), ('s23', 'dsv41')):
            for a in REAL:
                out.append(('new', m, H.runs(stage=st, arm=a, model=m, method='con')))
    if group == 'new_b':
        for st, m in (('s23', 'haiku55'), ('s23', 'sonnet55_low'), ('s23', 'gemini38_low')):
            for a in REAL:
                out.append(('new', m, H.runs(stage=st, arm=a, model=m, method='con')))
    return out


def hyper_any(N, S, k):
    """P(at least one success among k distinct uniform draws from N cells with S successes)."""
    if N <= 0:
        return 0.0
    k = min(k, N)
    return 1.0 - comb(N - S, k) / comb(N, k) if N - S >= k else 1.0


def face_heading(xy, txy, robot_yaw, deg_m):
    d = np.hypot(txy[0] - xy[:, 0], txy[1] - xy[:, 1])
    th = np.arctan2(txy[1] - xy[:, 1], txy[0] - xy[:, 0])
    th = np.where(d < deg_m, robot_yaw, th)
    return th


def heading_ok_tau(ep, matched, goal, theta, tau_deg):
    if matched is None:
        return False, False
    regs = [g for g in (ep.acceptable_regions or []) if g.target_id in (None, matched)]
    holding = [g for g in regs if point_in_polygon(goal, g.polygon)]
    ok = False
    for g in holding:
        allowed = g.heading.yaw if g.heading.yaw is not None else bearing(goal, g.heading.face_point)
        if abs(wrap_angle(theta - allowed)) <= math.radians(tau_deg) + 1e-12:
            ok = True
    return bool(holding), ok


def greedy_far(start_xy_list, pts, n):
    picked = []
    mind = np.full(len(pts), np.inf)
    for s in start_xy_list:
        mind = np.minimum(mind, np.hypot(pts[:, 0] - s[0], pts[:, 1] - s[1]))
    for _ in range(n):
        i = int(np.argmax(mind))
        picked.append(i)
        mind = np.minimum(mind, np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1]))
    return picked


def one(rec, th):
    sc = H.scene(rec.ref, rec.scene_id)
    obs = sc.observation
    ep = sc.truths[rec.episode_id].gt
    places = [tuple(rec.pose_world)] + [tuple(c['pose_world']) for c in (rec.candidates or [])]
    row = {'key': f'{rec.run}/{rec.scene_id}/{rec.episode_id}/{rec.trial}', 'run': rec.run, 'arm': rec.arm,
           'scene': rec.scene_id, 'episode': rec.episode_id, 'trial': rec.trial, 'situation': rec.situation,
           'ncand': len(rec.candidates or []), 'rec_success': bool(rec.score.get('success'))}
    sp = H.score_poses(rec, places)
    row['succ'] = [bool(s.success) for s in sp]
    row['inreg'] = [bool(s.in_region) for s in sp]
    row['matched'] = sp[0].matched_target_id
    row['forb_places'] = [{k: bool(v) for k, v in s.forbidden.items()} for s in sp]
    row['kinds'] = sorted({g.kind for g in (ep.forbidden_regions or [])})
    # heading curve on the recorded places
    for tau in (30, 45, 60):
        vals = []
        for s, p in zip(sp, places):
            hold, ok = heading_ok_tau(ep, s.matched_target_id, p[:2], p[2], tau)
            vals.append(bool(s.matched_target_id is not None and s.reachable and hold and ok))
        row[f'succ_tau{tau}'] = vals
    # jitter
    jit = []
    for k in range(8):
        a = math.radians(45 * k)
        dx, dy = 0.05 * math.cos(a), 0.05 * math.sin(a)
        js = H.score_poses(rec, [(p[0] + dx, p[1] + dy, p[2]) for p in places])
        jit.append([bool(s.success) for s in js])
    row['jit'] = jit
    # floor
    people = [DetectedPerson.model_validate(p) for p in rec.people]
    fl = solver.floor(obs, people, th)
    sel = fl.selectable
    cen = fl.centres()[sel]                         # grid (row-major) order
    idx_grid = -np.ones(sel.shape, dtype=int)
    idx_grid[sel] = np.arange(sel.sum())
    on_sel = []
    place_cell = []
    for p in places:
        ij = fl.index(p[0], p[1])
        ok = ij is not None and bool(sel[ij])
        on_sel.append(ok)
        place_cell.append(int(idx_grid[ij]) if ok else -1)
    row['on_sel'] = on_sel
    tw = rec.result['target']['world']
    txy = (float(tw[0]), float(tw[1]))
    dist = np.hypot(cen[:, 0] - txy[0], cen[:, 1] - txy[1])
    disc = dist <= 3.0 + 1e-9
    theta = face_heading(cen, txy, obs.robot.yaw, th.heading_degenerate_m)
    poses = [(float(x), float(y), float(t)) for (x, y), t in zip(cen, theta)]
    cs = H.score_poses(rec, poses)
    succ = np.array([s.success for s in cs])
    inreg = np.array([s.in_region for s in cs])
    kinds = row['kinds']
    forb = {k: np.array([bool(s.forbidden.get(k, False)) for s in cs]) for k in kinds}
    Nr, Sr = int(len(cen)), int(succ.sum())
    Nd, Sd = int(disc.sum()), int(succ[disc].sum())
    row.update(N_room=Nr, S_room=Sr, I_room=int(inreg.sum()), N_disc=Nd, S_disc=Sd, I_disc=int(inreg[disc].sum()))
    row['top_in_disc'] = bool(place_cell[0] >= 0 and disc[place_cell[0]])
    row['forb_disc_share'] = {k: (float(forb[k][disc].mean()) if Nd else None) for k in kinds}
    row['forb_room_share'] = {k: float(forb[k].mean()) for k in kinds}
    # exact R5 and AR
    top = row['succ'][0]
    row['R5_disc'] = hyper_any(Nd, Sd, 5)
    row['R1_disc'] = Sd / Nd if Nd else 0.0
    row['R5_room'] = hyper_any(Nr, Sr, 5)
    # in-region versions (skeptic)
    row['R5_disc_inreg'] = hyper_any(Nd, int(inreg[disc].sum()), 5)

    def ar(mask):
        pool = mask.copy()
        if place_cell[0] >= 0:
            pool[place_cell[0]] = False
        N1, S1 = int(pool.sum()), int(succ[pool].sum())
        return 1.0 if top else hyper_any(N1, S1, 4), N1, S1
    row['AR_disc'], row['AR_disc_N'], row['AR_disc_S'] = ar(disc)
    row['AR_room'], _, _ = ar(np.ones(len(cen), dtype=bool))
    # AF deterministic in the disc, from place 1
    if Nd >= 4:
        dpts = cen[disc]
        dsucc = succ[disc]
        picks = greedy_far([places[0][:2]], dpts, 4)
        afs = [bool(dsucc[i]) for i in picks]
        row['AF_seq'] = afs
        row['AF5'] = bool(top or any(afs))
        # F5 Monte Carlo, 200 draws
        rng = np.random.default_rng([14, zlib.crc32(row['key'].encode())])
        hits = 0
        hits1 = 0
        for _ in range(200):
            f = int(rng.integers(Nd))
            pk = [f] + greedy_far([dpts[f]], dpts, 4)
            hits += any(dsucc[i] for i in pk)
            hits1 += bool(dsucc[f])
        row['F5_disc'] = hits / 200
    else:
        row['AF_seq'] = None
        row['AF5'] = None
        row['F5_disc'] = None
    return row


def main():
    group = sys.argv[1]
    out = open(f'{OUT}/rows_{group}.jsonl', 'w')
    t0 = time.time()
    n = 0
    for tag, model, refs in cells_for(group):
        for ref in refs:
            th = H.recorded_config(ref)[1].thresholds
            for rec in H.records(ref):
                if not rec.solver_step:
                    continue
                if tag == 'watch' and rec.situation != 'audience':
                    continue
                row = one(rec, th)
                row['tag'], row['model'] = tag, model
                out.write(json.dumps(row) + '\n')
                out.flush()
                n += 1
                if group == 'smoke' and n >= 3:
                    print('smoke', time.time() - t0)
                    return
    print(group, n, 'records', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
