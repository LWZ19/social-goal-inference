"""C5 gates: shared pieces (FROZEN.md section 2). The vectorised point-in-polygon replica is I14 v0's (phase1/i14_v0/
i14.py pip), copied unchanged; the floor rule is the precheck's coarse_floor. No model, no network."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

HERE = Path('results/still-scenes/learned-models/flow-model-gates')
FRESH = Path('results/still-scenes/perception/fresh_seeds')
CACHE = HERE / 'cache'
FROZEN_SHA = '8b7d3ae294ad81a810c70c23706247674bbfed22cc0247445e91f6615426419c'
SEEDS = tuple(range(100, 110))
SUB_SEED = 20261018
EPS = 1e-9
PEOPLE_SITS = ('waiting_line', 'conversation_group', 'audience')
DEV_SITS = ('no_one', 'waiting_line', 'conversation_group', 'audience')
FEATURE_NAMES = ('f1_dt', 'f2_cos_front', 'f3_sin_front', 'f4_dp', 'f5_face_cos', 'f6_n15', 'f7_robot',
                 'f8_in_way', 'f9_dt_minus_rmax')
FEATURE_SETS = {'FS1': (0, 1, 2, 3, 4, 5), 'FS2': (0, 1, 2, 3, 4, 5, 6), 'FS3': (0, 1, 2, 3, 4, 5, 7, 8)}
M_GRID = (0.5, 1.0, 2.0, 4.0)
CAP_DP = 5.0
NEAR_N_M = 1.5
F9_RADIUS_M = 2.0


def wrap(a):
    """sgi.geometry.wrap_angle, elementwise."""
    return math.pi - np.mod(math.pi - a, 2 * math.pi)


def pip(px, py, poly):
    """sgi.polygons.point_in_polygon, elementwise: even-odd, the boundary within EPS inside (I14 v0 replica)."""
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


def in_union(xy, polys):
    """Cells whose centre lies in any polygon (bbox prefilter, then the replica)."""
    hold = np.zeros(len(xy), dtype=bool)
    for poly in polys:
        v = np.array(poly, dtype=np.float64)
        lo, hi = v.min(axis=0) - 1e-6, v.max(axis=0) + 1e-6
        box = np.all((xy >= lo) & (xy <= hi), axis=1) & ~hold
        if box.any():
            idx = np.flatnonzero(box)
            hold[idx] |= pip(xy[idx, 0], xy[idx, 1], poly)
    return hold


def coarse_floor(truth, robot, th):
    """0.10 m cells whose centre is a 0.05 m scorer cell centre (origin shifted -0.025 m), free in robot_grid and
    joined to the robot's start by freespace.connected (the scorer's reachability path)."""
    from sgi.freespace import connected, robot_grid
    grid = robot_grid(truth, robot, th)
    assert abs(grid.resolution - 0.05) < 1e-12, grid.resolution
    joined = connected(grid, (robot.x, robot.y))
    sub = joined[0::2, 0::2]
    ny, nx = sub.shape
    xs = grid.x0 + 0.025 + 0.10 * np.arange(nx)
    ys = grid.y0 + 0.025 + 0.10 * np.arange(ny)
    iy, ix = np.nonzero(sub)
    return np.stack([xs[ix], ys[iy]], axis=1)


def features(xy, truth, target, robot, th):
    """f1-f9 (FROZEN.md section 2, Features) at every cell centre of xy (N, 2)."""
    a = np.array(target.anchor[:2], dtype=np.float64)
    d = xy - a
    dt = np.hypot(d[:, 0], d[:, 1])
    front = target.front_yaw
    if front is None and target.kind == 'person':
        front = truth.person(target.person_id).yaw
    if front is None:
        c2 = np.zeros(len(xy))
        s2 = np.zeros(len(xy))
    else:
        rel = wrap(np.arctan2(d[:, 1], d[:, 0]) - front)
        c2, s2 = np.cos(rel), np.sin(rel)
    people = list(truth.people)
    n = len(xy)
    if people:
        P = np.array([(p.x, p.y) for p in people], dtype=np.float64)
        Y = np.array([p.yaw for p in people], dtype=np.float64)
        D = np.hypot(xy[:, None, 0] - P[None, :, 0], xy[:, None, 1] - P[None, :, 1])      # (N, people)
        k = np.argmin(D, axis=1)
        dmin = D[np.arange(n), k]
        f4 = np.minimum(dmin, CAP_DP)
        ux, uy = (xy[:, 0] - P[k, 0]) / dmin, (xy[:, 1] - P[k, 1]) / dmin
        f5 = np.cos(Y[k]) * ux + np.sin(Y[k]) * uy
        f5 = np.where(dmin >= CAP_DP, 0.0, f5)
        f6 = (D <= NEAR_N_M + EPS).sum(axis=1).astype(np.float64)
        # f8: people other than the target person within person_radius_m of the segment [c, a]
        f8 = np.zeros(n)
        seg = a[None, :] - xy                                                               # c -> a
        L2 = (seg ** 2).sum(axis=1)
        for p in people:
            if target.kind == 'person' and p.id == target.person_id:
                continue
            pc = np.array([p.x, p.y]) - xy
            with np.errstate(divide='ignore', invalid='ignore'):
                t = np.where(L2 > 0, (pc * seg).sum(axis=1) / L2, 0.0)
            t = np.clip(t, 0.0, 1.0)
            q = xy + t[:, None] * seg
            dd = np.hypot(p.x - q[:, 0], p.y - q[:, 1])
            f8 += (dd <= th.person_radius_m + EPS)
        ra = np.hypot(P[:, 0] - a[0], P[:, 1] - a[1])
        near = ra[ra <= F9_RADIUS_M + EPS]
        rmax = float(near.max()) if near.size else 0.0
    else:
        f4 = np.full(n, CAP_DP)
        f5 = np.zeros(n)
        f6 = np.zeros(n)
        f8 = np.zeros(n)
        rmax = 0.0
    f7 = np.hypot(xy[:, 0] - robot.x, xy[:, 1] - robot.y)
    f9 = dt - rmax
    return np.stack([dt, c2, s2, f4, f5, f6, f7, f8, f9], axis=1)
