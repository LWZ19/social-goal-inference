"""Independent verifier library for I11 Stage 0 (phase 1b).

Written from FROZEN.md (sha256 11f7b327...) only. It imports the package and the research harness, never the
compute agent's i11lib / run_eval / verify_indep, nor rerun/lifting (lifting.py is used only in an optional
cross-check, check_masks_vs_lifting, which compares masks and is not used for any number).

Components: hook-and-compress union-find on the 4-neighbour edge list (a different method from the compute
agent's label propagation and its BFS).
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import sys  # noqa: E402
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
WT = paths.REPO
RD = paths.STILL / 'perception'                  # holds lifting/ and lifting-rerun/
OUT = RD / 'lifting/verify_i11_stage0'
sys.path.insert(0, str(WT / 'results/still-scenes/analysis-scripts/harness'))
import research_harness as H  # noqa: E402
from sgi.evaluate import match_target  # noqa: E402
from sgi.geometry import pixel_to_point  # noqa: E402
from sgi.types import GroundedTarget, Point  # noqa: E402

PINS = {
    RD / 'lifting/FROZEN.md': '11f7b32746c7fc0be414883b430d59ddef98d2388fd89e082e40361b637a9165',
    WT / 'src/sgi/evaluate.py': '198f11ae6adf6e8a7eda4bb2f8c78f982baaf38f15cbe6f8ff6276eb66086ca1',
    WT / 'src/sgi/geometry.py': '99af62055b6998faafbf6748b7eece13adb427227c71821b5190fdcdbd4b4db2',
    WT / 'src/sgi/stages/goal.py': 'edce50f09990b80e5413c41caf19b543747f33a912e6e14c93aed5db1f7ba600',
    WT / 'results/still-scenes/analysis-scripts/harness/research_harness.py': '391483778ec1c5a232909292862926724a6580fc5f9c307b1f3f9c789cee455c',
    WT / 'results/still-scenes/analysis-scripts/harness/replay_solver.py': 'eed1dcc1bb0287208e98a4b150236486b7c23d12613859b1dc92890b7d3af243',
    RD / 'lifting-rerun/lifting.py': '10ec43fe1871ab89f749dfa3991498c03c856de9b125eeddcf9561ee9c80fe63',
    RD / 'lifting-rerun/frontsnap.py': 'a45aae92673562fab43fd2f4dee9cecfac5c3b0b214fe641254a24d358671307',
}

# constants from FROZEN.md
SNAP_R = 15.0
FRONT_MARGIN = 0.25
Z_BODY = 0.05            # body masks: z > 0.05
H_PERSON = 2.0
R_PERSON = 0.30
FLOOR_BAND = 0.15        # floor_tolerance_m
G1_GAP = 0.10
FRAG = 25
L_S = 4.6
HALF = 2                 # 5x5 window
# front snap reference
FS_R, FS_JUMP, FS_AGREE = 15.0, 0.5, 0.3

CLS_INVALID, CLS_FLOOR, CLS_STRUCT, CLS_OBJECT, CLS_FRAG = 0, 1, 2, 3, 4
CLS_NAME = {0: 'invalid', 1: 'floor', 2: 'structure', 3: 'object', 4: 'fragment'}
FLOOR_LABEL = -2
INVALID_LABEL = -1


def check_pins():
    lines = []
    for p, h in PINS.items():
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == h, (p, got, h)
        lines.append(f'{p.relative_to(WT)} OK')
    return lines


def check_th(th):
    assert th.floor_tolerance_m == 0.15, th.floor_tolerance_m
    assert th.lift_window == 5, th.lift_window
    assert th.seen_depth_ratio == 0.03, th.seen_depth_ratio
    assert th.seen_depth_margin_m == 0.02, th.seen_depth_margin_m
    assert th.match_tolerance_m == 0.3, th.match_tolerance_m
    assert th.default_match_radius_m == 0.75, th.default_match_radius_m


# ------------------------------------------------------------------ geometry
def per_pixel_world(depth, cam):
    Z = np.asarray(depth, dtype=np.float64)
    h, w = Z.shape
    u = (np.arange(w) + 0.5)[None, :].repeat(h, 0)
    v = (np.arange(h) + 0.5)[:, None].repeat(w, 1)
    T = np.asarray(cam.T_world_camera, dtype=np.float64)
    Xc = (u - cam.cx) * Z / cam.fx
    Yc = (v - cam.cy) * Z / cam.fy
    Wx = T[0, 0] * Xc + T[0, 1] * Yc + T[0, 2] * Z + T[0, 3]
    Wy = T[1, 0] * Xc + T[1, 1] * Yc + T[1, 2] * Z + T[1, 3]
    Wz = T[2, 0] * Xc + T[2, 1] * Yc + T[2, 2] * Z + T[2, 3]
    valid = np.isfinite(Z) & (Z > 0)
    return Z, np.stack([Wx, Wy, Wz], -1), valid


def lift_pixel_at(cam, u, v, Z):
    T = np.asarray(cam.T_world_camera, dtype=np.float64)
    p = np.array([(u - cam.cx) * Z / cam.fx, (v - cam.cy) * Z / cam.fy, Z, 1.0])
    return (T @ p)[:3]


def win_slice(i, j, h, w, half=HALF):
    return slice(max(i - half, 0), min(i + half + 1, h)), slice(max(j - half, 0), min(j + half + 1, w))


def polygon_distance(X, Y, poly):
    """0 inside (even-odd, boundary inside), else distance to the boundary; vectorised, my own code."""
    P = np.asarray(poly, dtype=np.float64)
    n = len(P)
    best = np.full(X.shape, np.inf)
    crossings = np.zeros(X.shape, dtype=np.int64)
    for k in range(n):
        x1, y1 = P[k]
        x2, y2 = P[(k + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        L = ex * ex + ey * ey
        if L > 0:
            t = ((X - x1) * ex + (Y - y1) * ey) / L
            t = np.minimum(np.maximum(t, 0.0), 1.0)
        else:
            t = np.zeros(X.shape)
        d = np.sqrt((X - (x1 + t * ex)) ** 2 + (Y - (y1 + t * ey)) ** 2)
        best = np.minimum(best, d)
        straddle = (y1 > Y) != (y2 > Y)
        with np.errstate(divide='ignore', invalid='ignore'):
            xc = x1 + (Y - y1) * ex / np.where(ey == 0, np.nan, ey)
        crossings += (straddle & (xc > X)).astype(np.int64)
    inside = (crossings % 2 == 1) | (best <= 1e-9)
    return np.where(inside, 0.0, best)


# ------------------------------------------------------------------ components (hook-and-compress union-find)
def connected(n, ea, eb):
    parent = np.arange(n, dtype=np.int64)
    if len(ea) == 0:
        return parent
    while True:
        ra, rb = parent[ea], parent[eb]
        lo, hi = np.minimum(ra, rb), np.maximum(ra, rb)
        m = lo != hi
        if not m.any():
            break
        np.minimum.at(parent, hi[m], lo[m])
        while True:
            pp = parent[parent]
            if np.array_equal(pp, parent):
                break
            parent = pp
    return parent


def components(Z, W, valid, mode):
    """label map: -1 invalid, -2 floor, >=0 component id (ids ordered by first pixel in row-major order)."""
    h, w = Z.shape
    floor = valid & (W[..., 2] < FLOOR_BAND)
    node = valid & ~floor
    idx = np.arange(h * w).reshape(h, w)
    eas, ebs = [], []
    for (a_sl, b_sl) in (((slice(None), slice(0, w - 1)), (slice(None), slice(1, w))),
                         ((slice(0, h - 1), slice(None)), (slice(1, h), slice(None)))):
        both = node[a_sl] & node[b_sl]
        if mode == 'G1':
            d = np.sqrt(((W[a_sl] - W[b_sl]) ** 2).sum(-1))
            ok = d <= G1_GAP
        elif mode == 'G2':
            za, zb = Z[a_sl], Z[b_sl]
            ok = np.abs(za - zb) <= 0.03 * np.minimum(za, zb) + 0.02
        else:
            raise ValueError(mode)
        j = both & ok
        eas.append(idx[a_sl][j])
        ebs.append(idx[b_sl][j])
    ea, eb = np.concatenate(eas), np.concatenate(ebs)
    root = connected(h * w, ea, eb).reshape(h, w)
    lab = np.full((h, w), INVALID_LABEL, dtype=np.int64)
    lab[floor] = FLOOR_LABEL
    roots = root[node]
    # compact ids in row-major order of first appearance
    uniq, first, inv = np.unique(roots, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    lab[node] = rank[inv]
    return lab


def robust_extent(xy):
    c = xy - xy.mean(0)
    if len(c) < 2:
        return 0.0
    cov = c.T @ c / len(c)
    _, vec = np.linalg.eigh(cov)
    best = 0.0
    for k in range(2):
        pr = c @ vec[:, k]
        best = max(best, float(np.percentile(pr, 98) - np.percentile(pr, 2)))
    return best


def classify(lab, W):
    """class per component id and a per-pixel class map."""
    n = int(lab.max()) + 1 if (lab >= 0).any() else 0
    flat = lab.ravel()
    on = flat >= 0
    sizes = np.bincount(flat[on], minlength=n)
    ccls = np.full(n, CLS_FRAG, dtype=np.int64)
    ext = np.zeros(n)
    big = np.nonzero(sizes >= FRAG)[0]
    Wf = W.reshape(-1, 3)
    if len(big):
        order = np.argsort(flat[on], kind='stable')
        pix = np.nonzero(on)[0][order]
        labs_sorted = flat[on][order]
        starts = np.searchsorted(labs_sorted, np.arange(n))
        ends = np.searchsorted(labs_sorted, np.arange(n), side='right')
        for c in big:
            pp = pix[starts[c]:ends[c]]
            e = robust_extent(Wf[pp, :2])
            ext[c] = e
            ccls[c] = CLS_STRUCT if e > L_S else CLS_OBJECT
    cmap = np.full(lab.shape, CLS_INVALID, dtype=np.int64)
    cmap[lab == FLOOR_LABEL] = CLS_FLOOR
    m = lab >= 0
    cmap[m] = ccls[lab[m]]
    return ccls, sizes, ext, cmap


# ------------------------------------------------------------------ masks
def person_masks(W, valid, people):
    """per person index: body mask (within 0.30 m horizontally, 0.05<z<2.0, nearest person), and nearest index."""
    X, Y, Zw = W[..., 0], W[..., 1], W[..., 2]
    if not people:
        return [], None
    D = np.stack([np.hypot(X - p.x, Y - p.y) for p in people], 0)
    near = np.argmin(D, 0)
    out = []
    for k, p in enumerate(people):
        out.append((D[k] <= R_PERSON) & (Zw > Z_BODY) & (Zw < H_PERSON) & (near == k) & valid)
    return out, near


def target_body(W, valid, truth, tid, th, pmasks):
    t = truth.targets[tid]
    X, Y, Zw = W[..., 0], W[..., 1], W[..., 2]
    if t.footprint is not None:
        d = polygon_distance(X, Y, t.footprint)
        return (d <= 0.05) & (Zw > Z_BODY) & valid
    if t.kind == 'person':
        pid = [p.id for p in truth.people].index(t.person_id)
        return pmasks[pid]
    radius = th.default_match_radius_m if t.match_radius is None else t.match_radius
    return (np.hypot(X - t.anchor[0], Y - t.anchor[1]) <= radius) & (Zw > Z_BODY) & valid


def target_match_mask(W, valid, truth, tid, th, near):
    t = truth.targets[tid]
    X, Y = W[..., 0], W[..., 1]
    if t.footprint is not None:
        return (polygon_distance(X, Y, t.footprint) <= th.match_tolerance_m) & valid
    radius = th.default_match_radius_m if t.match_radius is None else t.match_radius
    m = (np.hypot(X - t.anchor[0], Y - t.anchor[1]) <= radius) & valid
    if t.kind == 'person':
        pid = [p.id for p in truth.people].index(t.person_id)
        m &= near == pid
    return m


# ------------------------------------------------------------------ scene cache
class SceneGeo:
    def __init__(self, sc, th):
        self.sc = sc
        self.cam = sc.observation.camera
        self.depth = sc.observation.depth
        self.Z, self.W, self.valid = per_pixel_world(self.depth, self.cam)
        self.h, self.w = self.Z.shape
        self.people = list(sc.truth.people)
        self.pmasks, self.near = person_masks(self.W, self.valid, self.people)
        self.th = th
        self._body = {}
        self._match = {}
        self.comp = {}
        for g in ('G1', 'G2'):
            lab = components(self.Z, self.W, self.valid, g)
            ccls, sizes, ext, cmap = classify(lab, self.W)
            self.comp[g] = dict(lab=lab, ccls=ccls, sizes=sizes, ext=ext, cmap=cmap)

    def body(self, tid):
        if tid not in self._body:
            self._body[tid] = target_body(self.W, self.valid, self.sc.truth, tid, self.th, self.pmasks)
        return self._body[tid]

    def match(self, tid):
        if tid not in self._match:
            self._match[tid] = target_match_mask(self.W, self.valid, self.sc.truth, tid, self.th, self.near)
        return self._match[tid]


def nearest_in(mask, u, v, rmax=None):
    """(i, j, distance) of the mask pixel whose centre is nearest (u, v); ties row-major; None if none."""
    ii, jj = np.nonzero(mask)        # row-major order
    if len(ii) == 0:
        return None
    d = np.hypot(jj + 0.5 - u, ii + 0.5 - v)
    k = int(np.argmin(d))            # first minimum = row-major tie-break
    if rmax is not None and d[k] > rmax:
        return None
    return int(ii[k]), int(jj[k]), float(d[k])


def window_median(Z, valid, i, j, mask=None):
    rs, cs = win_slice(i, j, Z.shape[0], Z.shape[1])
    z = Z[rs, cs]
    m = valid[rs, cs]
    if mask is not None:
        m = m & mask[rs, cs]
    vals = z[m]
    return float(np.median(vals)) if vals.size else None


def surface_label(geo, g, u, v):
    """(label, class) of s: the window pixel whose valid depth is nearest the window median depth; ties: pixel centre
    nearest (u, v), then row-major."""
    i, j = math.floor(v), math.floor(u)
    rs, cs = win_slice(i, j, geo.h, geo.w)
    z = geo.Z[rs, cs]
    vm = geo.valid[rs, cs]
    med = float(np.median(z[vm]))
    ii, jj = np.nonzero(vm)
    ii = ii + rs.start
    jj = jj + cs.start
    dz = np.abs(geo.Z[ii, jj] - med)
    dp = np.hypot(jj + 0.5 - u, ii + 0.5 - v)
    order = np.lexsort((np.arange(len(ii)), dp, dz))   # primary dz, then dp, then row-major index
    k = order[0]
    lab = int(geo.comp[g]['lab'][ii[k], jj[k]])
    cls = int(geo.comp[g]['cmap'][ii[k], jj[k]])
    return lab, cls, med


def window_labels(geo, g, u, v):
    i, j = math.floor(v), math.floor(u)
    rs, cs = win_slice(i, j, geo.h, geo.w)
    lab = geo.comp[g]['lab'][rs, cs]
    cmap = geo.comp[g]['cmap'][rs, cs]
    keep = (cmap == CLS_FLOOR) | (cmap == CLS_STRUCT) | (cmap == CLS_OBJECT)
    return set(int(x) for x in np.unique(lab[keep]))


def comp_lift(geo, g, c, u, v):
    """component lift onto component c: (pixel centre (uq, vq), world) or None when c has no pixel within 15 px."""
    lab = geo.comp[g]['lab']
    i, j = math.floor(v), math.floor(u)
    r = 17
    i0, i1, j0, j1 = max(i - r, 0), min(i + r + 1, geo.h), max(j - r, 0), min(j + r + 1, geo.w)
    sub = lab[i0:i1, j0:j1] == c
    q = nearest_in(sub, u - j0, v - i0, SNAP_R)
    if q is None:
        return None
    qi, qj = q[0] + i0, q[1] + j0
    Zc = window_median(geo.Z, geo.valid, qi, qj, mask=(lab == c))
    uq, vq = qj + 0.5, qi + 0.5
    return (uq, vq), lift_pixel_at(geo.cam, uq, vq, Zc)


def nearest_object_component(geo, g, u, v, exclude=None):
    lab, cmap = geo.comp[g]['lab'], geo.comp[g]['cmap']
    i, j = math.floor(v), math.floor(u)
    r = 17
    i0, i1, j0, j1 = max(i - r, 0), min(i + r + 1, geo.h), max(j - r, 0), min(j + r + 1, geo.w)
    m = cmap[i0:i1, j0:j1] == CLS_OBJECT
    if exclude is not None:
        m &= lab[i0:i1, j0:j1] != exclude
    q = nearest_in(m, u - j0, v - i0, SNAP_R)
    if q is None:
        return None
    return int(lab[q[0] + i0, q[1] + j0]), q[2]


def make_target(cam, uq, vq, world):
    x, y = pixel_to_point(uq, vq, cam.width, cam.height)
    return GroundedTarget(point=Point(x=x, y=y), pixel=(uq, vq), world=tuple(float(c) for c in world))


def front_snap(geo, u, v):
    """Independent re-implementation of frontsnap.py's rule (R=15, JUMP=0.5, AGREE=0.3)."""
    i, j = math.floor(v), math.floor(u)
    zpt = window_median(geo.Z, geo.valid, i, j)
    r = 16
    i0, i1, j0, j1 = max(i - r, 0), min(i + r + 1, geo.h), max(j - r, 0), min(j + r + 1, geo.w)
    sub = geo.Z[i0:i1, j0:j1]
    cand = geo.valid[i0:i1, j0:j1] & (sub < zpt - FS_JUMP)
    q = nearest_in(cand, u - j0, v - i0, FS_R)
    if q is None:
        return None
    qi, qj = q[0] + i0, q[1] + j0
    zq = geo.Z[qi, qj]
    rs, cs = win_slice(qi, qj, geo.h, geo.w)
    ww = geo.Z[rs, cs]
    vm = geo.valid[rs, cs] & (np.abs(ww - zq) <= FS_AGREE)
    zm = float(np.median(ww[vm]))
    uq, vq = qj + 0.5, qi + 0.5
    return (uq, vq), lift_pixel_at(geo.cam, uq, vq, zm)
