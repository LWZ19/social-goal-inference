"""I09 A1 project side (torch-free): lift, filters and facing of detector boxes (FROZEN.md sections 3-4; DEVIATIONS.md
C3-C10). No situation, scene or label is read.

    candidates(raw, depth, cam, tau_min) -> per box with score >= tau_min (descending score): the setting-independent
        features (depth band, position before the push, z_top, flat flag, F1-F4 yaws, torso pixel, diagnostics).
    people_for(cands, setting, cam_xy, variant) -> the output people of one grid setting: steps 2-7.
    run_image(raw, depth, cam, setting, variant) -> the deployed path for one image (runtime and evaluation).

World yaw as truth: radians, anticlockwise from +x. Keypoint L/R are the person's own (COCO; probe in FROZEN.md).
"""
from __future__ import annotations

import math

import numpy as np

from sgi.geometry import _depth_at, lift, wrap_angle

CONF = 0.30                      # a confident keypoint
NOSE, LEYE, REYE, LEAR, REAR, LS, RS, LH, RH = 0, 1, 2, 3, 4, 5, 6, 11, 12
BAND_M = 0.30                    # step 2 depth band around Z_med
MIN_KEPT = 10
HULL_MIN_VALID = 20
KP_DEPTH_M = 0.40                # a used keypoint's depth within this of Z_med
KP_WINDOW = 5
LAT_MIN_M = 0.05                 # |lateral| below this: yaw None
EDGE_ON_M = 0.12                 # F4 projected shoulder width
NOSE_DEAD_PX = 1.0
FLAT_TOL_M = 0.04
FLAT_SHARE = 0.80
FLAT_MIN_Q = 30
TOP_TOUCH_PX = 2.0
VARIANTS = ("F1", "F2", "F3", "F4")                # frozen (FROZEN.md section 4)
FIX_VARIANTS = ("F5",)                              # ADDENDUM_FIX.md: F4 without the face rule = F2 + profile rule
ALL_VARIANTS = VARIANTS + FIX_VARIANTS
BANDS = {"wide": (1.00, 2.30), "narrow": (1.30, 2.10)}


# ------------------------------------------------------------------------------------------------ pixels and lifting
def lift_pixels(depth, cam, cols, rows):
    """World points of pixels (integer cols, rows) at their centres, valid depth only (sgi.geometry.lift_all
    arithmetic). Returns (points (N, 3), u, v, Z)."""
    cols = np.asarray(cols, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    Z = depth[rows, cols].astype(np.float64)
    keep = np.isfinite(Z) & (Z > 0)
    u, v, Z = cols[keep] + 0.5, rows[keep] + 0.5, Z[keep]
    pc = np.stack([(u - cam.cx) * Z / cam.fx, (v - cam.cy) * Z / cam.fy, Z, np.ones_like(Z)], axis=1)
    return (pc @ cam.T_world_camera.T)[:, :3], u, v, Z


def centre_range(lo, hi, n):
    """Integer pixel indices whose centre (i + 0.5) lies in [lo, hi], clipped to [0, n - 1]."""
    a = max(0, math.ceil(lo - 0.5))
    b = min(n - 1, math.floor(hi - 0.5))
    return a, b


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    """Monotone chain; collinear points dropped. Fewer than 3 vertices for a degenerate set."""
    pts = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(pts) < 3:
        return pts
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def hull_pixels(poly, H, W):
    """(cols, rows) of pixels whose centres lie in the convex polygon (boundary included)."""
    if len(poly) < 3:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    us, vs = [p[0] for p in poly], [p[1] for p in poly]
    c0, c1 = centre_range(min(us), max(us), W)
    r0, r1 = centre_range(min(vs), max(vs), H)
    if c1 < c0 or r1 < r0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    rr, cc = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
    u, v = cc + 0.5, rr + 0.5
    inside = np.ones(u.shape, bool)
    n = len(poly)
    for i in range(n):
        (ax, ay), (bx, by) = poly[i], poly[(i + 1) % n]
        inside &= (bx - ax) * (v - ay) - (by - ay) * (u - ax) >= -1e-9
    return cc[inside], rr[inside]


def rect_pixels(u0, u1, v0, v1, H, W, stride=1):
    c0, c1 = centre_range(u0, u1, W)
    r0, r1 = centre_range(v0, v1, H)
    if c1 < c0 or r1 < r0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    cs = np.arange(c0, c1 + 1)
    rs = np.arange(r0, r1 + 1)
    if stride > 1:
        cs = cs[cs % stride == 0]
        rs = rs[rs % stride == 0]
    rr, cc = np.meshgrid(rs, cs, indexing="ij")
    return cc.ravel(), rr.ravel()


# ------------------------------------------------------------------------------------------------ facing
def yaw_from_lateral(lat):
    """Facing = the lateral vector L - R turned by -90 deg; None when shorter than LAT_MIN_M."""
    if lat is None or math.hypot(lat[0], lat[1]) < LAT_MIN_M:
        return None
    return math.atan2(-lat[0], lat[1])


def used_point(kp, j, depth, cam, zmed):
    """World point of keypoint j when it is used (C10): confident, lifts (5x5), depth within KP_DEPTH_M of Z_med."""
    u, v, s = kp[j]
    if s < CONF:
        return None
    P = lift(depth, cam, u, v, window=KP_WINDOW)
    if P is None:
        return None
    Z = _depth_at(depth, cam, u, v, KP_WINDOW)
    if abs(Z - zmed) > KP_DEPTH_M:
        return None
    return P


def facing(kp, depth, cam, zmed, person_xy, cam_xy):
    """F1-F4 yaws (None allowed) and diagnostics for one box's keypoints."""
    out = {v: None for v in ALL_VARIANTS}
    diag = {"sh_used": False, "hip_used": False, "edge_on": None, "f3_flip": False, "f4_rule": None}
    if kp is None:
        return out, diag
    P = {j: used_point(kp, j, depth, cam, zmed) for j in (LS, RS, LH, RH)}
    lat_sh = None if P[LS] is None or P[RS] is None else (P[LS][:2] - P[RS][:2])
    lat_hip = None if P[LH] is None or P[RH] is None else (P[LH][:2] - P[RH][:2])
    diag["sh_used"], diag["hip_used"] = lat_sh is not None, lat_hip is not None
    out["F1"] = yaw_from_lateral(lat_sh)
    vecs = [x for x in (lat_sh, lat_hip) if x is not None]
    f2 = yaw_from_lateral(np.mean(vecs, axis=0)) if vecs else None
    out["F2"] = f2
    f3 = f2
    to_cam = math.atan2(cam_xy[1] - person_xy[1], cam_xy[0] - person_xy[0])
    if f3 is not None and all(kp[j][2] >= CONF for j in (NOSE, LEYE, REYE)) and math.cos(f3 - to_cam) < 0:
        f3 = wrap_angle(f3 + math.pi)
        diag["f3_flip"] = True
    out["F3"] = f3
    f4 = f3
    f5 = f2                                         # ADDENDUM_FIX.md: the profile rule on F2 (no face rule)
    if kp[LS][2] >= CONF and kp[RS][2] >= CONF:
        width = abs(kp[LS][0] - kp[RS][0]) * zmed / cam.fx
        diag["edge_on"] = bool(width < EDGE_ON_M)
        diag["sh_width_m"] = width
        if width < EDGE_ON_M and kp[NOSE][2] >= CONF:
            ears = [kp[j][0] for j in (LEAR, REAR) if kp[j][2] >= CONF]
            u_ref = float(np.mean(ears)) if ears else 0.5 * (kp[LS][0] + kp[RS][0])
            d = kp[NOSE][0] - u_ref
            if abs(d) >= NOSE_DEAD_PX:
                base = math.atan2(person_xy[1] - cam_xy[1], person_xy[0] - cam_xy[0])
                f4 = wrap_angle(base - math.pi / 2) if d > 0 else wrap_angle(base + math.pi / 2)
                f5 = f4
                diag["f4_rule"] = "right" if d > 0 else "left"
            else:
                diag["f4_rule"] = "dead zone"
    out["F4"] = f4
    out["F5"] = f5
    return out, diag


# ------------------------------------------------------------------------------------------------ one box
def box_features(box, kp, depth, cam, cam_xy):
    """Steps 1-5's setting-independent parts for one box; a dict with 'reject' set when step 2 rejects it."""
    H, W = depth.shape
    x0, y0, x1, y1, score = box
    w, h = x1 - x0, y1 - y0
    f = {"box": [x0, y0, x1, y1], "score": score, "reject": None}
    region = "box"
    pts = None
    if kp is not None and all(kp[j][2] >= CONF for j in (LS, RS, RH, LH)):
        cols, rows = hull_pixels(convex_hull([(kp[j][0], kp[j][1]) for j in (LS, RS, RH, LH)]), H, W)
        if len(cols):
            pts, u, v, Z = lift_pixels(depth, cam, cols, rows)
            if len(Z) >= HULL_MIN_VALID:
                region = "hull"
            else:
                pts = None
    if pts is None:
        cols, rows = rect_pixels(x0 + 0.3 * w, x0 + 0.7 * w, y0 + 0.2 * h, y0 + 0.5 * h, H, W)
        pts, u, v, Z = lift_pixels(depth, cam, cols, rows) if len(cols) else (np.zeros((0, 3)),) + (np.zeros(0),) * 3
    f["region"] = region
    f["region_valid"] = int(len(Z))
    if len(Z) == 0:
        f["reject"] = "no valid depth in the torso region"
        return f
    zmed = float(np.median(Z))
    keep = np.abs(Z - zmed) <= BAND_M
    f["zmed"] = zmed
    f["kept"] = int(keep.sum())
    if keep.sum() < MIN_KEPT:
        f["reject"] = "fewer than 10 points in the depth band"
        return f
    kpts = pts[keep]
    f["body_xy"] = (float(np.median(kpts[:, 0])), float(np.median(kpts[:, 1])))
    f["torso_pixel"] = (float(np.median(u[keep])), float(np.median(v[keep])))
    uc = 0.5 * (x0 + x1)
    p = np.array([(uc - cam.cx) * zmed / cam.fx, (y0 - cam.cy) * zmed / cam.fy, zmed, 1.0])
    f["z_top"] = float(cam.T_world_camera[2] @ p)
    f["top_touch"] = bool(y0 < TOP_TOUCH_PX)
    # flat test inputs: Q = valid pixels in the box, even rows and columns; plane through the kept points
    qc, qr = rect_pixels(x0, x1, y0, y1, H, W, stride=2)
    Q = lift_pixels(depth, cam, qc, qr)[0] if len(qc) else np.zeros((0, 3))
    f["n_Q"] = int(len(Q))
    if len(Q) < FLAT_MIN_Q:
        f["flat"] = False
        f["flat_share"] = None
    else:
        c = kpts.mean(axis=0)
        n = np.linalg.svd(kpts - c, full_matrices=False)[2][-1]
        share = float(np.mean(np.abs((Q - c) @ n) <= FLAT_TOL_M))
        f["flat_share"] = share
        f["flat"] = bool(share >= FLAT_SHARE)
    yaws, diag = facing(kp, depth, cam, zmed, f["body_xy"], cam_xy)
    f["yaw"] = yaws
    f["facing_diag"] = diag
    return f


def candidates(raw, depth, cam, tau_min):
    """Features of every box with score >= tau_min, in the sidecar's descending-score order."""
    cam_xy = cam.T_world_camera[:2, 3]
    kp_of = {p["box_index"]: p["kp"] for p in raw["pose"]}
    out = []
    for i, box in enumerate(raw["boxes"]):
        if box[4] < tau_min:
            break
        f = box_features(box, kp_of.get(i), depth, cam, cam_xy)
        f["index"] = i
        out.append(f)
    return out


def push(xy, cam_xy, d):
    away = (xy[0] - cam_xy[0], xy[1] - cam_xy[1])
    length = math.hypot(*away)
    if length <= 0 or d == 0:
        return (float(xy[0]), float(xy[1]))
    return (float(xy[0] + d * away[0] / length), float(xy[1] + d * away[1] / length))


def people_for(cands, s, cam_xy, variant="F1"):
    """Steps 2-7 for one setting: s has tau, band ('wide'|'narrow'), flat (bool), d_dup, d_push."""
    lo, hi = BANDS[s["band"]]
    kept = []
    for c in cands:
        if c["score"] < s["tau"]:
            break
        if c["reject"] is not None:
            continue
        if c["z_top"] > hi or (not c["top_touch"] and c["z_top"] < lo):
            continue
        if s["flat"] and c["flat"]:
            continue
        xy = push(c["body_xy"], cam_xy, s["d_push"])
        if any(math.dist(xy, k["xy"]) <= s["d_dup"] for k in kept):
            continue
        kept.append({"xy": xy, "yaw": c["yaw"][variant], "yaws": c["yaw"], "score": c["score"], "box": c["box"],
                     "torso_pixel": c["torso_pixel"], "index": c["index"], "facing_diag": c["facing_diag"]})
    return kept


def run_image(raw, depth, cam, s, variant):
    """The deployed path for one image: candidates at tau, then the setting's filters, one facing variant."""
    return people_for(candidates(raw, depth, cam, s["tau"]), s, cam.T_world_camera[:2, 3], variant)
