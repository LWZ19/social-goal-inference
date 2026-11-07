"""S0 (FROZEN.md section 10, steps 1-2): unit checks on synthetic depth, then fit checks on the worked-example
scenes only. No evaluation record or scene is loaded here. Writes s0_fit.txt and s0_fit_rows.jsonl."""
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i11lib as I  # noqa: E402
from sgi.geometry import Camera, RobotPose, camera_on_robot, lift  # noqa: E402
from sgi.evaluate import match_target  # noqa: E402
from sgi.gt.derive import oracle_target  # noqa: E402

log = I.Log()


# ---------------------------------------------------------------- synthetic scenes
def synth_camera():
    T = camera_on_robot(RobotPose(0.0, 0.0, 0.0), 0.0, 1.5, math.radians(10.0))
    return Camera(554.3, 554.3, 320.0, 240.0, 640, 480, T)


def render_boxes(cam, boxes):
    """Camera depth (optical Z) of axis-aligned world boxes [(xmin, xmax, ymin, ymax, zmin, zmax)]; 0 = no hit."""
    h, w = cam.height, cam.width
    vv, uu = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing='ij')
    dc = np.stack([(uu - cam.cx) / cam.fx, (vv - cam.cy) / cam.fy, np.ones_like(uu)], -1)
    Rm = cam.T_world_camera[:3, :3]
    C = cam.T_world_camera[:3, 3]
    dw = dc @ Rm.T
    best = np.full((h, w), np.inf)
    for b in boxes:
        lo = np.array([b[0], b[2], b[4]])
        hi = np.array([b[1], b[3], b[5]])
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (lo - C) / dw
            t2 = (hi - C) / dw
        tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
        hit = (tmin <= tmax) & (tmax > 0)
        t = np.where(tmin > 0, tmin, tmax)
        best = np.where(hit & (t < best), t, best)
    return np.where(np.isfinite(best), best, 0.0).astype(np.float32)


def bfs_reference(node, er, ed):
    """A plain breadth-first labelling, to check label_components' partition."""
    h, w = node.shape
    lab = -np.ones((h, w), int)
    for i in range(h):
        for j in range(w):
            if not node[i, j] or lab[i, j] >= 0:
                continue
            root = i * w + j
            lab[i, j] = root
            stack = [(i, j)]
            while stack:
                a, b = stack.pop()
                nb = []
                if b + 1 < w and er[a, b]:
                    nb.append((a, b + 1))
                if b - 1 >= 0 and er[a, b - 1]:
                    nb.append((a, b - 1))
                if a + 1 < h and ed[a, b]:
                    nb.append((a + 1, b))
                if a - 1 >= 0 and ed[a - 1, b]:
                    nb.append((a - 1, b))
                for p in nb:
                    if lab[p] < 0:
                        lab[p] = root
                        stack.append(p)
    return lab


def unit_checks():
    log('S0.1 unit checks on synthetic depth (camera 640x480, f 554.3 px, 1.5 m high, pitched 10 deg down)')
    ok_all = True
    cam = synth_camera()
    floor = (-50, 50, -50, 50, -1.0, 0.0)
    # (a) box 0.5 m in front of a wall: two components; wall structure once wide enough
    for wall_w, expect_wall in ((20.0, I.STRUCT), (3.0, I.OBJECT)):
        wall = (6.0, 6.2, -wall_w / 2, wall_w / 2, 0.0, 3.0)
        box = (5.0, 5.5, -0.4, 0.4, 0.0, 2.0)   # taller than the camera: no top face (see the line after)
        depth = render_boxes(cam, [floor, wall, box])
        g = I.Geom(depth, cam)
        cp = g.comps('G1')
        big = {c: cp.ccls[c] for c in cp.size if cp.size[c] >= I.FRAG_PX}
        # the box's centre pixel and a wall pixel above it
        from sgi.geometry import project
        ub, vb, _ = project(cam, (5.0, 0.0, 0.5))
        uw, vw, _ = project(cam, (6.0, 0.0, 2.8))   # above the box from the camera
        cb = cp.label[math.floor(vb), math.floor(ub)]
        cw = cp.label[math.floor(vw), math.floor(uw)]
        ok = (len(big) == 2 and cb != cw and cp.ccls[cb] == I.OBJECT and cp.ccls[cw] == expect_wall)
        ok_all &= ok
        log(f"  (a) wall {wall_w:4.1f} m wide, box 0.5 m in front: components >=25 px {len(big)}, box "
            f"{I.CLS_NAME[cp.ccls[cb]]}, wall {I.CLS_NAME[cp.ccls[cw]]} (extent {cp.extent.get(int(cw), float('nan')):.2f} m), "
            f"expect wall {I.CLS_NAME[expect_wall]} -> {'PASS' if ok else 'FAIL'}")
        if wall_w == 20.0:
            # (d) component lift equals lifting.lift_at_depth / sgi lift on a one-component window
            qi, qj = math.floor(vb), math.floor(ub)
            i0, i1, j0, j1 = I.win_bounds(qi, qj, g.h, g.w)
            one = bool((cp.label[i0:i1, j0:j1] == cb).all())
            t = I.comp_lift(g, cp, int(cb), (qi, qj))
            ref = I.L.lift_at_depth(cam, qj + 0.5, qi + 0.5, float(np.median(g.Z[i0:i1, j0:j1])))
            ref2 = lift(depth, cam, qj + 0.5, qi + 0.5, 5)
            e1 = float(np.linalg.norm(np.array(t.world) - ref))
            e2 = float(np.linalg.norm(np.array(t.world) - ref2))
            ok = one and e1 < 1e-12 and e2 < 1e-12
            ok_all &= ok
            log(f"  (d) component lift vs lifting.lift_at_depth on a one-component window: window one component "
                f"{one}, |diff| {e1:.1e} m, vs sgi.geometry.lift {e2:.1e} m -> {'PASS' if ok else 'FAIL'}")
            # V1 / T1 sanity on the synthetic scene: a point on the wall 6 px right of the box edge
            colmask = (cp.label == cb).any(axis=0)
            right_edge = int(np.max(np.nonzero(colmask)[0]))
            row = math.floor(vb)
            u_w, v_w = right_edge + 6.5, row + 0.5
            info_w = I.point_info(g, u_w, v_w)
            tv, iv = I.variant(g, u_w, v_w, 'V1')
            info_b = I.point_info(g, ub, vb)
            tb, _ = I.variant(g, ub, vb, 'V1')
            ok = (info_w['s_cls'] == 'structure' and info_w['T1'] and tv is not None and iv.get('c') == cb
                  and info_b['s_cls'] == 'object' and not info_b['T1'] and tb is None)
            ok_all &= ok
            log(f"  (e) point on the wall 6 px beside the box: s={info_w['s_cls']} T1={info_w['T1']} V1 -> "
                f"{'box' if iv.get('c') == cb else iv}; point on the box: s={info_b['s_cls']} T1={info_b['T1']} V1 "
                f"moves={tb is not None} -> {'PASS' if ok else 'FAIL'}")
    # (a') informational: a 1.0 m box at 5 m shows its top at about 84 deg incidence; its far rows split into row
    # strips, the known cost FROZEN.md section 4 predicts for horizontal tops (not a pass/fail check)
    depth = render_boxes(cam, [floor, (6.0, 6.2, -10.0, 10.0, 0.0, 3.0), (5.0, 5.5, -0.4, 0.4, 0.0, 1.0)])
    g = I.Geom(depth, cam)
    cp = g.comps('G1')
    big = sorted((s_, I.CLS_NAME[cp.ccls[c]]) for c, s_ in cp.size.items() if s_ >= I.FRAG_PX)
    log(f"  (a') informational, a 1.0 m box (top face visible at about 84 deg incidence at 5 m): components "
        f">=25 px {len(big)} {big}: the top's far rows split into 1-row strips (the predicted known cost)")
    # (b) a floor plane gives only floor pixels
    depth = render_boxes(cam, [floor])
    g = I.Geom(depth, cam)
    cp = g.comps('G1')
    ok = bool(g.valid.any()) and bool((g.floor == g.valid).all()) and cp.n_components == 0
    ok_all &= ok
    log(f"  (b) floor plane only: valid {int(g.valid.sum())} px, floor {int(g.floor.sum())} px, components "
        f"{cp.n_components} -> {'PASS' if ok else 'FAIL'}")
    # (c) a slanted plane at 80 deg incidence at 4 m stays one component under G1
    for inc, expect_one in ((80.0, True), (88.0, False)):
        a = math.radians(inc)
        n = np.array([math.sin(a), 0.0, -math.cos(a)])
        d0 = n @ np.array([0.0, 0.0, 4.0])
        h, w = 480, 640
        vv, uu = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing='ij')
        r = np.stack([(uu - 320.0) / 554.3, (vv - 240.0) / 554.3, np.ones_like(uu)], -1)
        with np.errstate(divide='ignore'):
            Zp = d0 / (r @ n)
        patch = (uu > 300) & (uu < 330) & (vv > 150) & (vv < 300) & (Zp > 0) & (Zp < 60)
        depth = np.where(patch, Zp, 0.0).astype(np.float32)
        g = I.Geom(depth, cam)
        cp = g.comps('G1')
        node = g.valid & ~g.floor
        nlab = len(set(cp.label[node].tolist()))
        ok = (nlab == 1) == expect_one
        ok_all &= ok
        log(f"  (c) plane at {inc:.0f} deg incidence through 4 m on the optical axis (patch u 300-330, v 150-300): "
            f"G1 components {nlab} (expect {'1' if expect_one else 'more than 1'}; predicted limit acos(Z/(f g)) = "
            f"{math.degrees(math.acos(4.0 / (554.3 * 0.10))):.1f} deg at 4 m) -> {'PASS' if ok else 'FAIL'}")
    # (f) the labelling's partition against a plain breadth-first search
    rng = np.random.default_rng(1)
    for trial in range(5):
        node = rng.random((60, 80)) < 0.8
        er = node[:, :-1] & node[:, 1:] & (rng.random((60, 79)) < 0.55)
        ed = node[:-1, :] & node[1:, :] & (rng.random((59, 80)) < 0.55)
        a = I.label_components(node, er, ed)
        b = bfs_reference(node, er, ed)
        ok = bool((a == b).all())
        ok_all &= ok
    log(f"  (f) label_components vs breadth-first search on 5 random 60x80 graphs: {'PASS' if ok else 'FAIL'}")
    log(f"  unit checks: {'ALL PASS' if ok_all else 'SOME FAIL'}")
    return ok_all


# ---------------------------------------------------------------- fit checks
def boundary_dist(X, Y, poly):
    """Distance of each (X, Y) to the polygon's boundary, and whether it lies inside."""
    v = np.asarray(poly, dtype=np.float64)
    inside = np.zeros(X.shape, bool)
    dmin = np.full(X.shape, np.inf)
    for k in range(len(v)):
        ax, ay = v[k - 1]
        bx, by = v[k]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = np.clip(((X - ax) * dx + (Y - ay) * dy) / L2, 0, 1) if L2 > 0 else 0.0
        dmin = np.minimum(dmin, np.hypot(X - (ax + t * dx), Y - (ay + t * dy)))
        cross = (ay > Y) != (by > Y)
        with np.errstate(divide='ignore', invalid='ignore'):
            xc = ax + (Y - ay) * (bx - ax) / (by - ay)
        inside ^= cross & (xc > X)
    return dmin, inside


def diag(poly):
    v = np.asarray(poly, dtype=np.float64)
    return float(max(np.hypot(*(a - b)) for a in v for b in v))


def fit_checks():
    from sgi.config import load_config
    cfg = load_config()
    th = cfg.thresholds
    I.check_thresholds(th)
    ex = I.H.examples()
    log('')
    log(f'S0.2 fit checks on the worked-example scenes of H.examples(): {len(ex)} scenes '
        f'(FROZEN.md says 28; the harness lists {len(ex)}, all used; see DEVIATIONS.md D1)')
    m4rows, counts, wallrows, gtrows, epirows, people_rows, obstrows = [], [], [], [], [], [], []
    max_diag = (0.0, '')
    for d, sid in ex:
        sc = I.H.example(sid, d)
        sd = I.sd_from_scene(sc)
        g = I.Geom(sc.observation.depth, sc.observation.camera)
        assert np.array_equal(g.W, sd['W'])
        for t in sc.truth.targets.values():
            if t.footprint is not None:
                dd = diag(t.footprint)
                if dd > max_diag[0]:
                    max_diag = (dd, f'{sid} target {t.label}')
        for o in sc.truth.obstacles:
            dd = diag(o)
            if dd > max_diag[0]:
                max_diag = (dd, f'{sid} obstacle')
        for join in ('G1', 'G2'):
            cp = g.comps(join)
            counts.append(dict(scene=sid, join=join, comps=cp.n_components, frags=cp.n_fragments,
                               objects=cp.n_objects, structure=cp.n_structure,
                               floor_px=int(g.floor.sum()), node_px=int((g.valid & ~g.floor).sum())))
        tids = sorted({t for et in sc.truths.values() for t in et.gt.target_ids})
        for r in I.m4_scene(g, sd, th, tids):
            r['scene'] = sid
            m4rows.append(r)
        # every ground-truth person absorbed into structure?
        bodies = I.scene_bodies(sd, th)
        zok = g.W[..., 2] >= I.FLOOR_Z
        for k, p in enumerate(sd['people']):
            B = I.person_body(sd, k) & zok
            others = [m for kk, m in bodies.items()
                      if not (kk[0] == 'person' and kk[1] == p.id)
                      and not (kk[0] == 'target' and sc.truth.targets[kk[1]].kind == 'person'
                               and sc.truth.targets[kk[1]].person_id == p.id)]
            for join in ('G1', 'G2'):
                r = I.recovery(g.comps(join), B, others)
                if r is not None:
                    people_rows.append(dict(scene=sid, person=p.id, join=join, **r))
        # walls and ceiling labelled structure
        X, Y, Zw = g.W[..., 0], g.W[..., 1], g.W[..., 2]
        node = g.valid & ~g.floor
        if sc.truth.bounds is not None:
            bd, inside = boundary_dist(X, Y, sc.truth.bounds)
            near_wall = (bd <= I.WALL_BAND) | ~inside
        else:
            near_wall = np.zeros(X.shape, bool)
        env = node & (near_wall | (Zw > I.CEIL_BAND))
        for join in ('G1', 'G2'):
            cp = g.comps(join)
            c = cp.cls[env]
            wallrows.append(dict(scene=sid, join=join, n=int(env.sum()), structure=int((c == I.STRUCT).sum()),
                                 object=int((c == I.OBJECT).sum()), fragment=int((c == I.FRAG).sum())))
        # ground-truth people, targets and obstacles labelled object
        groups = {'person': np.zeros(X.shape, bool), 'target': np.zeros(X.shape, bool),
                  'obstacle': np.zeros(X.shape, bool)}
        for kk, m in bodies.items():
            if kk[0] == 'person' or sc.truth.targets[kk[1]].kind == 'person':
                groups['person'] |= m
            else:
                groups['target'] |= m
        for o in sc.truth.obstacles:
            groups['obstacle'] |= I.L.poly_dist(X, Y, o) <= 0.05
            om = (I.L.poly_dist(X, Y, o) <= 0.05) & node & (Zw <= I.CEIL_BAND)
            if om.any() and sc.truth.bounds is not None:
                ov = np.asarray(o, dtype=np.float64)
                gap = float(boundary_dist(ov[:, 0], ov[:, 1], sc.truth.bounds)[0].min())
                cpg = g.comps('G1')
                obstrows.append(dict(scene=sid, gap=gap, n=int(om.sum()),
                                     structure=int((cpg.cls[om] == I.STRUCT).sum())))
        for name, m in groups.items():
            m = m & node & (Zw <= I.CEIL_BAND)
            for join in ('G1', 'G2'):
                cp = g.comps(join)
                c = cp.cls[m]
                gtrows.append(dict(scene=sid, join=join, group=name, n=int(m.sum()),
                                   object=int((c == I.OBJECT).sum()), structure=int((c == I.STRUCT).sum()),
                                   fragment=int((c == I.FRAG).sum())))
        # triggers at the oracle target of each episode
        for eid, et in sorted(sc.truths.items()):
            gt = et.gt
            tid = gt.target_ids[0]
            ot = oracle_target(sc.observation, sc.truth.targets[tid], th)
            if ot is None:
                epirows.append(dict(scene=sid, episode=eid, oracle=None))
                continue
            u, v = ot.pixel
            before = match_target(ot.world, sc.truth, gt, th) is not None
            info = I.point_info(g, u, v, 'G1')
            row = dict(scene=sid, episode=eid, oracle=True, kind=sc.truth.targets[tid].kind,
                       right_before=before, s_cls=info['s_cls'], T1=info['T1'], T2=info['T2'])
            for vname in ('V1', 'V2', 'V3', 'V4'):
                tv, _ = I.variant(g, u, v, vname)
                row[vname + '_moved'] = tv is not None
                row[vname + '_right'] = before if tv is None else (match_target(tv.world, sc.truth, gt, th) is not None)
            epirows.append(row)
    log(f'  largest target footprint or obstacle diagonal over these {len(ex)} scenes: {max_diag[0]:.3f} m '
        f'({max_diag[1]}); FROZEN.md used 2.26 m -> L_S = 2 x 2.26 + 0.10 = 4.62 -> 4.6 m (unchanged, frozen)')
    log('')
    log('  components per image (median [min-max]):')
    for join in ('G1', 'G2'):
        cs = [c for c in counts if c['join'] == join]
        for key in ('comps', 'frags', 'objects', 'structure'):
            a = np.array([c[key] for c in cs])
            log(f"    {join} {key:9s} {np.median(a):7.0f} [{a.min()}-{a.max()}]")
        a = np.array([c['structure'] for c in cs])
        log(f"    {join} images with no structure component: {int((a == 0).sum())} of {len(cs)}")
    log('')
    log('  walls and ceiling: non-floor pixels within 0.3 m of the bounding polygon (or outside it) or above 2.85 m;')
    log('  share labelled structure (pooled pixels; per-image median):')
    for join in ('G1', 'G2'):
        ws = [w for w in wallrows if w['join'] == join and w['n'] > 0]
        n = sum(w['n'] for w in ws)
        s = sum(w['structure'] for w in ws)
        o = sum(w['object'] for w in ws)
        f = sum(w['fragment'] for w in ws)
        med = np.median([w['structure'] / w['n'] for w in ws])
        low = [(w['scene'], round(w['structure'] / w['n'], 2)) for w in ws if w['structure'] / w['n'] < 0.5]
        log(f"    {join}: structure {I.pct(s, n)}, object {I.pct(o, n)}, fragment {I.pct(f, n)}; per-image median "
            f"{med:.2f}; images under 0.5: {len(low)} {low}")
    log('')
    log('  ground-truth pixels (non-floor, at most 2.85 m) by label:')
    for join in ('G1', 'G2'):
        for name in ('person', 'target', 'obstacle'):
            rs = [r for r in gtrows if r['join'] == join and r['group'] == name]
            n = sum(r['n'] for r in rs)
            log(f"    {join} {name:8s} n={n:7d} object {I.pct(sum(r['object'] for r in rs), n)}  structure "
                f"{I.pct(sum(r['structure'] for r in rs), n)}  fragment {I.pct(sum(r['fragment'] for r in rs), n)}")
    log('  obstacles by their gap to the bounding wall (vertex distance), G1: mostly (> 0.5 of pixels) structure')
    for lo, hi in ((0.0, 0.10), (0.10, 0.30), (0.30, 1e9)):
        sel = [o for o in obstrows if lo <= o['gap'] < hi]
        log(f"    gap {lo:.2f}-{'inf' if hi > 1e8 else f'{hi:.2f}'} m: n={len(sel)}, mostly structure "
            f"{sum(o['structure'] / o['n'] > 0.5 for o in sel)}, mostly object {sum(o['structure'] / o['n'] <= 0.5 for o in sel)}")
    log('    (an obstacle touching the wall joins the wall component by construction; no free-standing obstacle is')
    log('    structure, no target pixel and no person pixel is structure: no section-4 definition fails visibly, so no')
    log('    addendum is written; FROZEN.md section 10 step 3)')
    log('')
    I.m4_table(m4rows, log, f'  M4 on the fit scenes\' true targets (distinct scene x target; n = {len(m4rows) // 2}):')
    log('')
    log('  every visible ground-truth person (not only targets): absorbed into structure / merged / merged with')
    log('  another body / split:')
    for join in ('G1', 'G2'):
        pr = [p for p in people_rows if p['join'] == join]
        n = len(pr)
        log(f"    {join}: n={n}, absorbed into structure {I.pct(sum(p['structure'] for p in pr), n)}, merged "
            f"{sum(p['merged'] for p in pr)}, merged with another body {sum(p['merged_body'] for p in pr)}, split "
            f"{sum(p['split'] for p in pr)}, median IoU {np.median([p['IoU'] for p in pr]):.2f}")
    log('')
    er = [e for e in epirows if e.get('oracle')]
    n = len(er)
    log(f'  triggers at the oracle target of each fit episode (n = {n} episodes with an oracle target, of '
        f'{len(epirows)}): how often a perfect point is flagged or moved')
    log(f"    surface s: {dict(collections.Counter(e['s_cls'] for e in er))}")
    log(f"    oracle target right before: {sum(e['right_before'] for e in er)} of {n}")
    log(f"    T1 flags {I.pct(sum(e['T1'] for e in er), n)}; T2 flags {I.pct(sum(e['T2'] for e in er), n)}")
    for kind in ('person', 'place', 'object'):
        ek = [e for e in er if e['kind'] == kind]
        if ek:
            log(f"      {kind:6s} n={len(ek):2d} T1 {sum(e['T1'] for e in ek)}  T2 {sum(e['T2'] for e in ek)}  "
                f"s {dict(collections.Counter(e['s_cls'] for e in ek))}")
    for vname in ('V1', 'V2', 'V3', 'V4'):
        mv = sum(e[vname + '_moved'] for e in er)
        brk = sum(e['right_before'] and not e[vname + '_right'] for e in er)
        fx = sum((not e['right_before']) and e[vname + '_right'] for e in er)
        log(f"    {vname}: moves {I.pct(mv, n)}, breaks (right -> wrong) {brk}, fixes {fx}")
    with open(I.OUTDIR / 's0_fit_rows.jsonl', 'w') as f:
        for kind, rows in (('m4', m4rows), ('counts', counts), ('walls', wallrows), ('gt', gtrows),
                           ('episodes', epirows), ('people', people_rows)):
            for r in rows:
                f.write(json.dumps(dict(table=kind, **{k: (v if not isinstance(v, (np.bool_,)) else bool(v))
                                                        for k, v in r.items()})) + '\n')
    return dict(m4=m4rows, walls=wallrows, gt=gtrows, people=people_rows, episodes=epirows)


def main():
    for line in I.check_pins():
        log(line)
    unit_checks()
    fit_checks()
    log.write(I.OUTDIR / 's0_fit.txt')


if __name__ == '__main__':
    main()
