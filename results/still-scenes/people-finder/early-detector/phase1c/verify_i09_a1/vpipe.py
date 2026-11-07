"""I09 A1 verifier: the project-side pipeline, re-implemented from FROZEN.md sections 3-4 (torch-free).

Two layers:
  box_base(obs, box, kp)        everything about one box that no grid parameter changes: the torso region, Z_med,
                                the kept count, the unpushed median ground point, z_top, the flat share, and the
                                facing ingredients (lateral vectors, face and profile cues).
  people(bases, setting, var)   the grid's filters (tau, band, flat, d_dup, d_push) and the facing variants.
"""
from __future__ import annotations

import math

import numpy as np

from sgi.geometry import _depth_at, bearing, lift, wrap_angle

CONF = 0.30
LS, RS, LH, RH = 5, 6, 11, 12
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
BANDS = {"wide": (1.00, 2.30), "narrow": (1.30, 2.10)}


def _hull(pts):
    """Convex hull (monotone chain), counter-clockwise, distinct vertices only."""
    P = sorted(set((float(a), float(b)) for a, b in pts))
    if len(P) < 3:
        return P

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def _centres_in(lo, hi, n):
    """Indices i in [0, n) whose centre i + 0.5 lies in [lo, hi]."""
    a = max(0, math.ceil(lo - 0.5))
    b = min(n - 1, math.floor(hi - 0.5))
    return np.arange(a, b + 1) if b >= a else np.arange(0)


def _valid(z):
    return np.isfinite(z) & (z > 0)


def _region(box, kp, depth):
    """(rows, cols) of the torso region's pixels (all of them, valid depth or not), and which rule made it."""
    H, W = depth.shape
    x0, y0, x1, y1 = box[:4]
    if kp is not None and all(kp[j][2] >= CONF for j in (LS, RS, LH, RH)):
        hull = _hull([(kp[j][0], kp[j][1]) for j in (LS, RS, LH, RH)])
        if len(hull) >= 3:
            us = [p[0] for p in hull]
            vs = [p[1] for p in hull]
            cols = _centres_in(min(us), max(us), W)
            rows = _centres_in(min(vs), max(vs), H)
            if len(cols) and len(rows):
                rr, cc = np.meshgrid(rows, cols, indexing="ij")
                u, v = cc + 0.5, rr + 0.5
                inside = np.ones(u.shape, bool)
                n = len(hull)
                for i in range(n):
                    (ax, ay), (bx, by) = hull[i], hull[(i + 1) % n]
                    inside &= (bx - ax) * (v - ay) - (by - ay) * (u - ax) >= -1e-9
                r, c = rr[inside], cc[inside]
                if int(_valid(depth[r, c]).sum()) >= 20:
                    return r, c, "hull"
    w, h = x1 - x0, y1 - y0
    cols = _centres_in(x0 + 0.3 * w, x0 + 0.7 * w, W)
    rows = _centres_in(y0 + 0.2 * h, y0 + 0.5 * h, H)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return rr.ravel(), cc.ravel(), "box"


def _lift_px(cam, r, c, Z):
    u, v = c + 0.5, r + 0.5
    pc = np.stack([(u - cam.cx) * Z / cam.fx, (v - cam.cy) * Z / cam.fy, Z, np.ones_like(Z)], axis=1)
    return (pc @ cam.T_world_camera.T)[:, :3]


def box_base(obs, box, kp):
    """Grid-independent quantities of one box; reason set when steps 1-2 reject it."""
    depth = obs.depth
    cam = obs.camera
    H, W = depth.shape
    out = {"score": float(box[4]), "box": [float(b) for b in box[:4]]}
    r, c, rule = _region(box, kp, depth)
    out["region"] = rule
    z = depth[r, c].astype(np.float64)
    ok = _valid(z)
    if not ok.any():
        out["reject"] = "no valid depth"
        return out
    r, c, z = r[ok], c[ok], z[ok]
    zmed = float(np.median(z))
    keep = np.abs(z - zmed) <= 0.30
    if int(keep.sum()) < 10:
        out["reject"] = "band < 10"
        return out
    r, c, z = r[keep], c[keep], z[keep]
    P = _lift_px(cam, r, c, z)
    out["zmed"] = zmed
    out["xy0"] = (float(np.median(P[:, 0])), float(np.median(P[:, 1])))
    out["torso_px"] = (float(np.median(c + 0.5)), float(np.median(r + 0.5)))
    x0, y0, x1, y1 = box[:4]
    uc = 0.5 * (x0 + x1)
    pt = np.array([(uc - cam.cx) * zmed / cam.fx, (y0 - cam.cy) * zmed / cam.fy, zmed, 1.0])
    out["ztop"] = float((cam.T_world_camera @ pt)[2])
    out["y0"] = float(y0)
    # flat share
    cols = _centres_in(x0, x1, W)
    rows = _centres_in(y0, y1, H)
    cols, rows = cols[cols % 2 == 0], rows[rows % 2 == 0]
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    zq = depth[rr, cc].astype(np.float64)
    vq = _valid(zq)
    rr, cc, zq = rr[vq], cc[vq], zq[vq]
    if len(zq) < 30:
        out["flat_share"] = None
    else:
        Q = _lift_px(cam, rr, cc, zq)
        cen = P.mean(axis=0)
        _, _, vt = np.linalg.svd(P - cen, full_matrices=False)
        n = vt[-1]
        d = np.abs((Q - cen) @ n)
        out["flat_share"] = float((d <= 0.04).mean())
    # facing ingredients
    def used(j):
        if kp is None or kp[j][2] < CONF:
            return None
        p = lift(depth, cam, kp[j][0], kp[j][1], 5)
        if p is None:
            return None
        zk = _depth_at(depth, cam, kp[j][0], kp[j][1], 5)
        if abs(zk - zmed) > 0.40:
            return None
        return p
    vecs = {}
    for name, (a, b) in (("sh", (LS, RS)), ("hip", (LH, RH))):
        pa, pb = used(a), used(b)
        if pa is not None and pb is not None:
            vecs[name] = (float(pa[0] - pb[0]), float(pa[1] - pb[1]))
    out["vecs"] = vecs
    conf = (lambda j: kp is not None and kp[j][2] >= CONF)
    out["face"] = bool(conf(NOSE) and conf(LEYE) and conf(REYE))
    edge = conf(LS) and conf(RS) and abs(kp[LS][0] - kp[RS][0]) * zmed / cam.fx < 0.12
    prof = None
    if edge and conf(NOSE):
        ears = [kp[j][0] for j in (LEAR, REAR) if conf(j)]
        uref = sum(ears) / len(ears) if ears else 0.5 * (kp[LS][0] + kp[RS][0])
        prof = float(kp[NOSE][0] - uref)
    out["edge"] = bool(edge)
    out["prof_du"] = prof
    out["both_sh_conf"] = bool(conf(LS) and conf(RS))
    return out


def _yaw_of(v):
    if v is None or math.hypot(*v) < 0.05:
        return None
    return math.atan2(-v[0], v[1])


def facing(b, xy, cam_xy):
    """{F1..F5: yaw or None} for a kept box b at its output position xy."""
    vecs = b["vecs"]
    f1 = _yaw_of(vecs.get("sh"))
    vs = list(vecs.values())
    f2 = _yaw_of((sum(v[0] for v in vs) / len(vs), sum(v[1] for v in vs) / len(vs))) if vs else None

    def face(y):
        if y is None or not b["face"]:
            return y
        if math.cos(y - bearing(xy, cam_xy)) < 0:
            return wrap_angle(y + math.pi)
        return y

    def prof(y):
        du = b["prof_du"]
        if du is None or abs(du) < 1.0:
            return y
        ang = math.atan2(xy[1] - cam_xy[1], xy[0] - cam_xy[0])
        return wrap_angle(ang - math.pi / 2 if du > 0 else ang + math.pi / 2)
    f3 = face(f2)
    return {"F1": f1, "F2": f2, "F3": f3, "F4": prof(f3), "F5": prof(f2)}


def people(bases, cam_xy, *, tau, band, flat, d_dup, d_push):
    """The kept people in descending score: dicts with x, y, yaws, score, box, base index."""
    lo, hi = BANDS[band] if isinstance(band, str) else band
    kept = []
    for i, b in enumerate(bases):                      # bases are in descending score
        if b["score"] < tau or "reject" in b:
            continue
        zt = b["ztop"]
        if zt > hi or (b["y0"] >= 2 and zt < lo):
            continue
        if flat and b["flat_share"] is not None and b["flat_share"] >= 0.80:
            continue
        x, y = b["xy0"]
        dx, dy = x - cam_xy[0], y - cam_xy[1]
        L = math.hypot(dx, dy)
        if L > 0:
            x, y = x + d_push * dx / L, y + d_push * dy / L
        if any(math.hypot(x - k["x"], y - k["y"]) <= d_dup for k in kept):
            continue
        kept.append({"x": x, "y": y, "i": i, "score": b["score"], "box": b["box"]})
    for k in kept:
        k["yaw"] = facing(bases[k["i"]], (k["x"], k["y"]), cam_xy)
    return kept
