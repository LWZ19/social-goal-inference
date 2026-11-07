"""Verifier's own re-implementation of A1's frozen project side (phase1c/i09/FROZEN.md sections 3-4 with
ADDENDUM_FIX.md's F5), written from the protocol text; it imports none of pipeline.py / common.py.

people(raw, depth, cam, tau=0.25, band=(1.00, 2.30), d_dup=0.50, d_push=0.20) -> list of dicts
    {xy, yaw (F5, radians or None), score, box, index}

Setting: chosen.json + chosen_fix.json (RT-DETRv2, tau 0.25, wide band, flat off, d_dup 0.50, d_push 0.20, F5).
"""
from __future__ import annotations

import math

import numpy as np

CONF = 0.30
NOSE, LEAR, REAR, LS, RS, LH, RH = 0, 3, 4, 5, 6, 11, 12


def wrap(a):
    """into (-pi, pi]"""
    a = math.fmod(a + math.pi, 2 * math.pi)
    if a <= 0:
        a += 2 * math.pi
    return a - math.pi


def idx_range(lo, hi, n):
    """integer pixel indices i whose centre i + 0.5 lies in [lo, hi], clipped to [0, n-1]"""
    a = max(0, int(math.ceil(lo - 0.5)))
    b = min(n - 1, int(math.floor(hi - 0.5)))
    return a, b


def lift_px(depth, cam, cols, rows):
    Z = depth[rows, cols].astype(np.float64)
    ok = np.isfinite(Z) & (Z > 0)
    u = cols[ok] + 0.5
    v = rows[ok] + 0.5
    Z = Z[ok]
    P = np.stack([(u - cam.cx) * Z / cam.fx, (v - cam.cy) * Z / cam.fy, Z, np.ones_like(Z)], 1)
    W = P @ np.asarray(cam.T_world_camera, dtype=np.float64).T
    return W[:, :3], u, v, Z


def hull(pts):
    """convex hull, counter-clockwise in (u, v) coordinates, collinear points dropped (Andrew)."""
    p = sorted(set((float(a), float(b)) for a, b in pts))
    if len(p) < 3:
        return p

    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for q in p:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(p):
        while len(up) >= 2 and cr(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    return lo[:-1] + up[:-1]


def pixels_in_hull(poly, H, W):
    if len(poly) < 3:
        return np.zeros(0, int), np.zeros(0, int)
    us = [q[0] for q in poly]
    vs = [q[1] for q in poly]
    c0, c1 = idx_range(min(us), max(us), W)
    r0, r1 = idx_range(min(vs), max(vs), H)
    if c1 < c0 or r1 < r0:
        return np.zeros(0, int), np.zeros(0, int)
    cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
    u = cc + 0.5
    v = rr + 0.5
    m = np.ones(u.shape, bool)
    for i in range(len(poly)):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % len(poly)]
        m &= ((bx - ax) * (v - ay) - (by - ay) * (u - ax)) >= -1e-9
    return cc[m], rr[m]


def pixels_in_rect(u0, u1, v0, v1, H, W):
    c0, c1 = idx_range(u0, u1, W)
    r0, r1 = idx_range(v0, v1, H)
    if c1 < c0 or r1 < r0:
        return np.zeros(0, int), np.zeros(0, int)
    cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
    return cc.ravel(), rr.ravel()


def window_depth(depth, cam, u, v, win=5):
    H, W = depth.shape
    if not (0 <= u < cam.width and 0 <= v < cam.height):
        return None
    r, c, h = int(math.floor(v)), int(math.floor(u)), win // 2
    patch = depth[max(r - h, 0):r + h + 1, max(c - h, 0):c + h + 1].astype(np.float64)
    good = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(good)) if good.size else None


def lift_kp(depth, cam, u, v):
    Z = window_depth(depth, cam, u, v)
    if Z is None:
        return None, None
    p = np.array([(u - cam.cx) * Z / cam.fx, (v - cam.cy) * Z / cam.fy, Z, 1.0])
    return (np.asarray(cam.T_world_camera, dtype=np.float64) @ p)[:3], Z


def yaw_f5(kp, depth, cam, zmed, pxy, cxy):
    if kp is None:
        return None
    used = {}
    for j in (LS, RS, LH, RH):
        u, v, s = kp[j]
        if s < CONF:
            used[j] = None
            continue
        P, Z = lift_kp(depth, cam, u, v)
        used[j] = P if (P is not None and abs(Z - zmed) <= 0.40) else None
    lats = []
    if used[LS] is not None and used[RS] is not None:
        lats.append(used[LS][:2] - used[RS][:2])
    if used[LH] is not None and used[RH] is not None:
        lats.append(used[LH][:2] - used[RH][:2])
    yaw = None
    if lats:
        m = np.mean(lats, axis=0)
        if math.hypot(m[0], m[1]) >= 0.05:
            yaw = math.atan2(-m[0], m[1])
    # profile rule (edge-on shoulders, nose against ears)
    if kp[LS][2] >= CONF and kp[RS][2] >= CONF:
        width = abs(kp[LS][0] - kp[RS][0]) * zmed / cam.fx
        if width < 0.12 and kp[NOSE][2] >= CONF:
            ears = [kp[j][0] for j in (LEAR, REAR) if kp[j][2] >= CONF]
            uref = float(np.mean(ears)) if ears else 0.5 * (kp[LS][0] + kp[RS][0])
            d = kp[NOSE][0] - uref
            if abs(d) >= 1.0:
                b = math.atan2(pxy[1] - cxy[1], pxy[0] - cxy[0])
                yaw = wrap(b - math.pi / 2) if d > 0 else wrap(b + math.pi / 2)
    return yaw


def people(raw, depth, cam, tau=0.25, band=(1.00, 2.30), d_dup=0.50, d_push=0.20):
    H, W = depth.shape
    T = np.asarray(cam.T_world_camera, dtype=np.float64)
    cxy = (float(T[0, 3]), float(T[1, 3]))
    kps = {p["box_index"]: p["kp"] for p in raw["pose"]}
    out = []
    for i, b in enumerate(raw["boxes"]):
        x0, y0, x1, y1, s = b
        if s < tau:
            break
        kp = kps.get(i)
        w, h = x1 - x0, y1 - y0
        pts = None
        if kp is not None and all(kp[j][2] >= CONF for j in (LS, RS, LH, RH)):
            cc, rr = pixels_in_hull(hull([kp[j][:2] for j in (LS, RS, RH, LH)]), H, W)
            if len(cc):
                pts, u, v, Z = lift_px(depth, cam, cc, rr)
                if len(Z) < 20:
                    pts = None
        if pts is None:
            cc, rr = pixels_in_rect(x0 + 0.3 * w, x0 + 0.7 * w, y0 + 0.2 * h, y0 + 0.5 * h, H, W)
            if len(cc) == 0:
                continue
            pts, u, v, Z = lift_px(depth, cam, cc, rr)
        if len(Z) == 0:
            continue
        zmed = float(np.median(Z))
        keep = np.abs(Z - zmed) <= 0.30
        if keep.sum() < 10:
            continue
        kp3 = pts[keep]
        bxy = (float(np.median(kp3[:, 0])), float(np.median(kp3[:, 1])))
        uc = 0.5 * (x0 + x1)
        ztop = float(T[2] @ np.array([(uc - cam.cx) * zmed / cam.fx, (y0 - cam.cy) * zmed / cam.fy, zmed, 1.0]))
        if ztop > band[1]:
            continue
        if not (y0 < 2.0) and ztop < band[0]:
            continue
        dx, dy = bxy[0] - cxy[0], bxy[1] - cxy[1]
        L = math.hypot(dx, dy)
        xy = (bxy[0] + d_push * dx / L, bxy[1] + d_push * dy / L) if L > 0 and d_push else bxy
        if any(math.dist(xy, o["xy"]) <= d_dup for o in out):
            continue
        out.append({"xy": xy, "yaw": yaw_f5(kp, depth, cam, zmed, bxy, cxy), "score": s, "box": [x0, y0, x1, y1],
                    "index": i})
    return out
