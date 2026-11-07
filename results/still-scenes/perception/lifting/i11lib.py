"""I11 Stage 0 (phase 1b) library: depth components, the snap variants V1-V4, the oracle snap-and-lift, the I33
triggers T1/T2, component recovery (M4) and the full-success replay.

Every definition follows FROZEN.md (sha256 11f7b32746c7fc0be414883b430d59ddef98d2388fd89e082e40361b637a9165).
Components, variants and triggers read only the depth image and the camera (no ground truth, no ML). The oracle,
the masks and the error classes are rerun/lifting/lifting.py's, imported unchanged. Runs and evaluation scenes are
read only through research_harness. No model or network call is made anywhere.
"""
from __future__ import annotations

import collections
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

WT = Path('.-scenes')
RD = WT / 'results/2026-10-17-beyond-vlm'
OUTDIR = RD / 'phase1b/i11'
sys.path.insert(0, str(WT / 'docs/model_notes/tools'))
sys.path.insert(0, str(RD / 'rerun/lifting'))
import research_harness as H  # noqa: E402
import lifting as L  # noqa: E402
from sgi.evaluate import match_target  # noqa: E402
from sgi.geometry import pixel_to_point  # noqa: E402
from sgi.types import GroundedTarget, Point  # noqa: E402

FROZEN_SHA = '11f7b32746c7fc0be414883b430d59ddef98d2388fd89e082e40361b637a9165'
PINNED = {
    'src/sgi/evaluate.py': '198f11ae6adf6e8a7eda4bb2f8c78f982baaf38f15cbe6f8ff6276eb66086ca1',
    'src/sgi/geometry.py': '99af62055b6998faafbf6748b7eece13adb427227c71821b5190fdcdbd4b4db2',
    'src/sgi/stages/goal.py': 'edce50f09990b80e5413c41caf19b543747f33a912e6e14c93aed5db1f7ba600',
    'docs/model_notes/tools/research_harness.py': '391483778ec1c5a232909292862926724a6580fc5f9c307b1f3f9c789cee455c',
    'docs/model_notes/tools/replay_solver.py': 'eed1dcc1bb0287208e98a4b150236486b7c23d12613859b1dc92890b7d3af243',
    str((RD / 'rerun/lifting/lifting.py').relative_to(WT)): '10ec43fe1871ab89f749dfa3991498c03c856de9b125eeddcf9561ee9c80fe63',
    str((RD / 'rerun/lifting/frontsnap.py').relative_to(WT)): 'a45aae92673562fab43fd2f4dee9cecfac5c3b0b214fe641254a24d358671307',
}

# ---------------------------------------------------------------- frozen parameters (FROZEN.md sections 2-5)
FLOOR_Z = 0.15            # floor band: floor_tolerance_m
G1_GAP = 0.10             # 3D join gap (G1)
RATIO, MARGIN = 0.03, 0.02  # depth-relative join (G2): seen_depth_ratio, seen_depth_margin_m
FRAG_PX = 25              # fragments: components under the lift window's area
L_S = 4.6                 # structure: robust horizontal extent above this (m)
R_SNAP = 15.0             # search radius (px), rerun/lifting SNAP_R
HALF = 2                  # 5x5 lift window
CEIL_BAND = 2.85          # fit check: pixels above this height count as ceiling
WALL_BAND = 0.30          # fit check: within this of the bounding polygon counts as wall

INVALID, FLOOR, STRUCT, OBJECT, FRAG = 0, 1, 2, 3, 4
CLS_NAME = {INVALID: 'invalid', FLOOR: 'floor', STRUCT: 'structure', OBJECT: 'object', FRAG: 'fragment'}
FIXES = ('oracle', 'V1', 'V2', 'V3', 'V4', 'front')


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_pins() -> list[str]:
    """Stop when FROZEN.md or any pinned file changed since the freeze (FROZEN.md section 1)."""
    lines = []
    got = sha256(OUTDIR / 'FROZEN.md')
    if got != FROZEN_SHA:
        raise SystemExit(f'FROZEN.md changed: {got}')
    lines.append(f'FROZEN.md sha256 {got} OK')
    for rel, want in PINNED.items():
        got = sha256(WT / rel)
        if got != want:
            raise SystemExit(f'pinned file changed: {rel} {got} != {want}')
        lines.append(f'{rel} sha256 OK')
    return lines


def check_thresholds(th) -> None:
    """The recorded thresholds FROZEN.md section 1 asserts for every record."""
    assert th.floor_tolerance_m == 0.15, th.floor_tolerance_m
    assert th.lift_window == 5, th.lift_window
    assert th.seen_depth_ratio == 0.03, th.seen_depth_ratio
    assert th.seen_depth_margin_m == 0.02, th.seen_depth_margin_m
    assert th.match_tolerance_m == 0.3, th.match_tolerance_m
    assert th.default_match_radius_m == 0.75, th.default_match_radius_m


# ---------------------------------------------------------------- geometry of one image
class Geom:
    """Per-pixel lift (identical arithmetic to lifting.scene_data) and the depth components of one image."""

    def __init__(self, depth: np.ndarray, cam):
        self.cam = cam
        self.depth = depth
        Z = depth.astype(np.float64)
        self.Z = Z
        h, w = Z.shape
        self.h, self.w = h, w
        vv, uu = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing='ij')
        pc = np.stack([(uu - cam.cx) * Z / cam.fx, (vv - cam.cy) * Z / cam.fy, Z, np.ones_like(Z)], axis=-1)
        self.W = (pc @ np.asarray(cam.T_world_camera).T)[..., :3].astype(np.float64)
        self.valid = np.isfinite(Z) & (Z > 0)
        self.floor = self.valid & (self.W[..., 2] < FLOOR_Z)
        self._comps = {}

    def comps(self, join: str = 'G1') -> 'Comps':
        if join not in self._comps:
            self._comps[join] = Comps(self, join)
        return self._comps[join]


def label_components(node: np.ndarray, er: np.ndarray, ed: np.ndarray) -> np.ndarray:
    """Connected components of the graph whose vertices are the `node` pixels and whose edges are the joined
    right-neighbour (er, shape h x w-1) and down-neighbour (ed, shape h-1 x w) pairs. Each component's label is
    the row-major index of its first pixel (hook-to-minimum with full pointer jumping, to a fixed point); -1 off
    the nodes. The partition does not depend on the method."""
    h, w = node.shape
    idx = np.arange(h * w).reshape(h, w)
    src = np.concatenate([idx[:, :-1][er], idx[:-1, :][ed]])
    dst = np.concatenate([idx[:, 1:][er], idx[1:, :][ed]])
    parent = np.arange(h * w)
    while True:
        while True:
            pp = parent[parent]
            if np.array_equal(pp, parent):
                break
            parent = pp
        pa, pb = parent[src], parent[dst]
        diff = pa != pb
        if not diff.any():
            break
        np.minimum.at(parent, np.maximum(pa[diff], pb[diff]), np.minimum(pa[diff], pb[diff]))
    lab = parent.reshape(h, w).copy()
    lab[~node] = -1
    return lab


def robust_extent(xy: np.ndarray) -> float:
    """Over the two principal axes of the points' (x, y), the larger 98th-minus-2nd-percentile range."""
    c = xy - xy.mean(axis=0)
    cov = c.T @ c / max(len(c), 1)
    _, vecs = np.linalg.eigh(cov)
    proj = c @ vecs
    return float(max(np.percentile(proj[:, k], 98) - np.percentile(proj[:, k], 2) for k in range(2)))


class Comps:
    """Depth components of one image under one join (FROZEN.md section 4)."""

    def __init__(self, g: Geom, join: str):
        self.join = join
        node = g.valid & ~g.floor
        a_r = node[:, :-1] & node[:, 1:]
        a_d = node[:-1, :] & node[1:, :]
        if join == 'G1':
            jr = np.linalg.norm(g.W[:, :-1] - g.W[:, 1:], axis=-1) <= G1_GAP
            jd = np.linalg.norm(g.W[:-1, :] - g.W[1:, :], axis=-1) <= G1_GAP
        elif join == 'G2':
            Za, Zb = g.Z[:, :-1], g.Z[:, 1:]
            jr = np.abs(Za - Zb) <= RATIO * np.minimum(Za, Zb) + MARGIN
            Za, Zb = g.Z[:-1, :], g.Z[1:, :]
            jd = np.abs(Za - Zb) <= RATIO * np.minimum(Za, Zb) + MARGIN
        else:
            raise ValueError(join)
        lab = label_components(node, a_r & jr, a_d & jd)
        self.label = lab                                    # component id (row-major first pixel), -1 elsewhere
        roots = lab[node]
        uniq, counts = np.unique(roots, return_counts=True)
        self.size = dict(zip(uniq.tolist(), counts.tolist()))
        self.ccls, self.extent = {}, {}
        big = uniq[counts >= FRAG_PX]
        flat_lab = lab.ravel()
        order = np.argsort(flat_lab, kind='stable')
        sorted_lab = flat_lab[order]
        W2 = g.W[..., :2].reshape(-1, 2)
        for c in uniq.tolist():
            self.ccls[c] = FRAG
        for c in big.tolist():
            lo = np.searchsorted(sorted_lab, c, 'left')
            hi = np.searchsorted(sorted_lab, c, 'right')
            ext = robust_extent(W2[order[lo:hi]])
            self.extent[c] = ext
            self.ccls[c] = STRUCT if ext > L_S else OBJECT
        cls_map = np.full(lab.shape, INVALID, dtype=np.int8)
        cls_map[g.floor] = FLOOR
        if len(uniq):
            lut_keys = np.array(list(self.ccls.keys()))
            lut_vals = np.array([self.ccls[k] for k in lut_keys], dtype=np.int8)
            pos = np.searchsorted(lut_keys, roots)            # uniq is sorted, so lut_keys is sorted
            cls_map[node] = lut_vals[pos]
        self.cls = cls_map
        self.n_components = len(uniq)
        self.n_fragments = int(sum(1 for c in uniq.tolist() if self.ccls[c] == FRAG))
        self.n_objects = int(sum(1 for c in big.tolist() if self.ccls[c] == OBJECT))
        self.n_structure = int(sum(1 for c in big.tolist() if self.ccls[c] == STRUCT))

    def t1_label(self, i: int, j: int):
        """The label a pixel carries for the straddle test: 'F' for floor, the component id for structure and
        object components, None for fragments and invalid pixels."""
        k = self.cls[i, j]
        if k == FLOOR:
            return 'F'
        if k in (STRUCT, OBJECT):
            return int(self.label[i, j])
        return None


# ---------------------------------------------------------------- helpers on a pixel
def win_bounds(i: int, j: int, h: int, w: int, half: int = HALF):
    return max(i - half, 0), min(i + half + 1, h), max(j - half, 0), min(j + half + 1, w)


def surface_pixel(g: Geom, u: float, v: float):
    """The pixel of the 5x5 window (clipped) at (floor(v), floor(u)) whose valid depth is nearest to the window's
    median valid depth; ties to the centre nearest (u, v), then row-major (FROZEN.md section 4, 's')."""
    i, j = math.floor(v), math.floor(u)
    i0, i1, j0, j1 = win_bounds(i, j, g.h, g.w)
    Zw = g.Z[i0:i1, j0:j1]
    vw = g.valid[i0:i1, j0:j1]
    if not vw.any():
        return None
    med = float(np.median(Zw[vw]))
    best, best_key = None, None
    for a, b in np.argwhere(vw):
        key = (abs(float(Zw[a, b]) - med), math.hypot(j0 + b + 0.5 - u, i0 + a + 0.5 - v), int(a), int(b))
        if best_key is None or key < best_key:
            best_key, best = key, (i0 + int(a), j0 + int(b))
    return best


def nearest_in(mask: np.ndarray, u: float, v: float, R: float = R_SNAP):
    """The pixel of `mask` whose centre is nearest (u, v) within R px (ties row-major), and its distance; (None, inf)
    when there is none."""
    h, w = mask.shape
    i, j = math.floor(v), math.floor(u)
    r = int(math.ceil(R)) + 1
    i0, i1, j0, j1 = max(i - r, 0), min(i + r + 1, h), max(j - r, 0), min(j + r + 1, w)
    sub = mask[i0:i1, j0:j1]
    if not sub.any():
        return None, math.inf
    vv, uu = np.meshgrid(np.arange(i0, i1) + 0.5, np.arange(j0, j1) + 0.5, indexing='ij')
    d = np.where(sub, np.hypot(uu - u, vv - v), np.inf)
    k = int(np.argmin(d))
    dk = float(d.flat[k])
    if dk > R:
        return None, math.inf
    return (i0 + k // (j1 - j0), j0 + k % (j1 - j0)), dk


def lift_at_depth(cam, u, v, Z):
    return L.lift_at_depth(cam, u, v, Z)


def make_target(cam, qi: int, qj: int, world) -> GroundedTarget:
    uq, vq = qj + 0.5, qi + 0.5
    x, y = pixel_to_point(uq, vq, cam.width, cam.height)
    return GroundedTarget(point=Point(x=x, y=y), pixel=(uq, vq), world=tuple(float(c) for c in world))


def comp_lift(g: Geom, cp: Comps, c: int, q) -> GroundedTarget:
    """Component lift onto component c at its pixel q: the median depth of c's pixels in the 5x5 window centred on
    q, lifted at q's centre (FROZEN.md section 4)."""
    qi, qj = q
    i0, i1, j0, j1 = win_bounds(qi, qj, g.h, g.w)
    m = (cp.label[i0:i1, j0:j1] == c) & g.valid[i0:i1, j0:j1]
    Zc = float(np.median(g.Z[i0:i1, j0:j1][m]))
    return make_target(g.cam, qi, qj, lift_at_depth(g.cam, qj + 0.5, qi + 0.5, Zc))


def nearest_object(g: Geom, cp: Comps, u: float, v: float, exclude=None):
    """(component, pixel, distance) of the object-component pixel nearest (u, v) within R px; None when none."""
    mask = cp.cls == OBJECT
    if exclude is not None:
        mask = mask & ~np.isin(cp.label, list(exclude))
    q, d = nearest_in(mask, u, v)
    if q is None:
        return None
    return int(cp.label[q]), q, d


# ---------------------------------------------------------------- per-point analysis (no ground truth)
def point_info(g: Geom, u: float, v: float, join: str = 'G1') -> dict:
    """s, its class, the straddle labels of the lift window, and the T1/T2 triggers, under one join."""
    cp = g.comps(join)
    s = surface_pixel(g, u, v)
    out = {}
    if s is None:
        out.update(s=None, s_cls='invalid', s_label=None, win_labels=0, T1=False, T2=False)
        return out
    s_cls = int(cp.cls[s])
    i, j = math.floor(v), math.floor(u)
    i0, i1, j0, j1 = win_bounds(i, j, g.h, g.w)
    labels = set()
    for a in range(i0, i1):
        for b in range(j0, j1):
            if g.valid[a, b]:
                lb = cp.t1_label(a, b)
                if lb is not None:
                    labels.add(lb)
    t1 = s_cls in (FLOOR, STRUCT) or len(labels) >= 2
    t2 = t1
    if not t2 and s_cls == OBJECT:
        t2 = nearest_object(g, cp, u, v, exclude={int(cp.label[s])}) is not None
    out.update(s=s, s_cls=CLS_NAME[s_cls], s_label=int(cp.label[s]) if s_cls in (STRUCT, OBJECT, FRAG) else None,
               win_labels=len(labels), T1=bool(t1), T2=bool(t2))
    return out


def variant(g: Geom, u: float, v: float, which: str):
    """(moved target or None, info) of V1, V2, V3 or V4 at the continuous pixel (u, v) (FROZEN.md section 5)."""
    join = 'G2' if which == 'V4' else 'G1'
    cp = g.comps(join)
    s = surface_pixel(g, u, v)
    if s is None:
        return None, {'why': 'no valid depth'}
    s_cls = int(cp.cls[s])
    if which in ('V1', 'V4', 'V3') and s_cls in (FLOOR, STRUCT):
        no = nearest_object(g, cp, u, v)
        if no is None:
            return None, {'why': f'on {CLS_NAME[s_cls]}, no object within R'}
        c, q, d = no
        return comp_lift(g, cp, c, q), {'why': f'on {CLS_NAME[s_cls]} -> object', 'c': c, 'd': d}
    if which == 'V2':
        no = nearest_object(g, cp, u, v)
        if no is None:
            return None, {'why': 'no object within R'}
        c, q, d = no
        return comp_lift(g, cp, c, q), {'why': 'always', 'c': c, 'd': d}
    if which == 'V3' and s_cls == OBJECT:
        cn = int(cp.label[s])
        i, j = math.floor(v), math.floor(u)
        i0, i1, j0, j1 = win_bounds(i, j, g.h, g.w)
        wl = cp.label[i0:i1, j0:j1]
        wc = cp.cls[i0:i1, j0:j1]
        wz = g.Z[i0:i1, j0:j1]
        wv = g.valid[i0:i1, j0:j1]
        zn = float(np.median(wz[(wl == cn) & wv]))
        quals = []
        for cf in sorted(set(wl[(wc == OBJECT) & wv].tolist()) - {cn}):
            zf = float(np.median(wz[(wl == cf) & wv]))
            if zf > zn:
                q, d = nearest_in(cp.label == cf, u, v)
                if q is not None:
                    quals.append((d, cf, q))
        if quals:
            d, cf, q = min(quals, key=lambda t: (t[0], t[1]))
            return comp_lift(g, cp, cf, q), {'why': 'amodal back-step', 'c': cf, 'd': d}
        return None, {'why': 'on object, nothing behind in window'}
    return None, {'why': f'on {CLS_NAME[s_cls]}'}


# ---------------------------------------------------------------- ground-truth side (lifting's, unchanged)
def sd_from_scene(sc) -> dict:
    """The dict lifting.scene_data builds, from a loaded Scene (for fit and fresh scenes, which have no record).
    Same arithmetic as lifting.scene_data, line for line."""
    cam = sc.observation.camera
    depth = sc.observation.depth
    Z = depth.astype(np.float64)
    h, w = Z.shape
    vv, uu = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing='ij')
    pc = np.stack([(uu - cam.cx) * Z / cam.fx, (vv - cam.cy) * Z / cam.fy, Z, np.ones_like(Z)], axis=-1)
    W = (pc @ np.asarray(cam.T_world_camera).T)[..., :3]
    valid = np.isfinite(Z) & (Z > 0)
    people = sc.truth.people
    P = np.array([[p.x, p.y] for p in people]) if people else np.zeros((0, 2))
    if len(P):
        d2 = (W[..., 0, None] - P[:, 0]) ** 2 + (W[..., 1, None] - P[:, 1]) ** 2
        nearest = np.argmin(d2, axis=-1)
        nearest_d = np.sqrt(np.min(d2, axis=-1))
    else:
        nearest = np.full(Z.shape, -1)
        nearest_d = np.full(Z.shape, np.inf)
    return dict(sc=sc, cam=cam, depth=depth, Z=Z, W=W.astype(np.float64), valid=valid, people=people, P=P,
                nearest=nearest, nearest_d=nearest_d, masks={})


def person_body(sd: dict, pid_index: int, r_person: float = L.R_PERSON) -> np.ndarray:
    """lifting's person body mask for any person (target or not)."""
    W = sd['W']
    per = sd['people'][pid_index]
    dp = np.hypot(W[..., 0] - per.x, W[..., 1] - per.y)
    return (dp <= r_person) & (W[..., 2] > L.Z_FLOOR) & (W[..., 2] < L.H_PERSON) & (sd['nearest'] == pid_index) \
        & sd['valid']


def true_target(sd: dict, gt, th, u=None, v=None) -> str:
    """gt.target_ids[0]; with several targets and a point, the one whose body mask is nearest to the point."""
    if len(gt.target_ids) == 1 or u is None:
        return gt.target_ids[0]
    best = None
    for tid in gt.target_ids:
        _, body = L.target_masks(sd, tid, th)
        _, d = L.nearest_pixel(body, u, v)
        if best is None or d < best[0]:
            best = (d, tid)
    return best[1]


def body_union(sd: dict, gt, th) -> np.ndarray:
    m = None
    for tid in gt.target_ids:
        _, body = L.target_masks(sd, tid, th)
        m = body.copy() if m is None else (m | body)
    return m


def oracle_fix(sd: dict, body: np.ndarray, u: float, v: float):
    """lifting.py's oracle snap-and-lift, unchanged: snap to the nearest body pixel within SNAP_R, median of the
    body pixels' depths in its 5x5 window, lift its centre there. (target or None, distance)."""
    qb, db = L.nearest_pixel(body, u, v)
    if qb is None or db > L.SNAP_R:
        return None, db
    bi, bj = qb
    wb = L.window(body, bi, bj)
    wz = L.window(sd['Z'], bi, bj)
    Zm = float(np.median(wz[wb]))
    w_or = L.lift_at_depth(sd['cam'], bj + 0.5, bi + 0.5, Zm)
    return make_target(sd['cam'], bi, bj, w_or), db


def error_class(sd: dict, tid: str, th, u: float, v: float, world_rec) -> dict:
    """lifting.py's error class of a wrong target, same rules and precedence (FROZEN.md section 2)."""
    _, body = L.target_masks(sd, tid, th)
    qb, db = L.nearest_pixel(body, u, v)
    i, j = math.floor(v), math.floor(u)
    Zwin = L.window(sd['Z'], i, j)
    Zpt = float(np.median(Zwin[np.isfinite(Zwin) & (Zwin > 0)]))
    idx = np.argwhere(body)
    if len(idx):
        dd = np.hypot(idx[:, 1] + 0.5 - u, idx[:, 0] + 0.5 - v)
        near = idx[dd <= L.SNAP_R]
        Zt = float(np.median(sd['Z'][near[:, 0], near[:, 1]])) if len(near) else float(sd['Z'][qb[0], qb[1]])
    else:
        Zt = math.nan
    if db > L.SNAP_R:
        lc = 'choice error'
    elif Zpt < Zt - L.FRONT_MARGIN:
        lc = 'occluder capture'
    elif Zpt > Zt + L.FRONT_MARGIN:
        lc = 'background leak'
    elif world_rec[2] < L.Z_FLOOR:
        lc = 'floor slip'
    else:
        lc = 'near-depth miss'
    return dict(lift_class=lc, d_body_px=db, Z_point=Zpt, Z_target=Zt,
                win_range_m=float(np.nanmax(Zwin) - np.nanmin(Zwin)), win_body_px=int(L.window(body, i, j).sum()))


LIFT_CLASSES = ('occluder capture', 'background leak', 'floor slip')


# ---------------------------------------------------------------- M4: component recovery
def recovery(cp: Comps, B: np.ndarray, others: list[np.ndarray]) -> dict | None:
    """M4 for one target: B is its body mask restricted to z >= 0.15 m; others are the body masks of the other
    people and the other targets of the scene. None when B is empty."""
    nb = int(B.sum())
    if nb == 0:
        return None
    labs = cp.label[B]
    labs = labs[labs >= 0]
    if not len(labs):
        return dict(nB=nb, IoU=0.0, coverage=0.0, purity=0.0, merged=False, merged_body=False, split=True,
                    structure=False, cstar_cls='none', cstar_px=0)
    uniq, counts = np.unique(labs, return_counts=True)
    k = int(np.argmax(counts))                       # largest overlap; ties to the lower label
    c = int(uniq[k])
    inter = int(counts[k])
    csz = cp.size[c]
    cm = cp.label == c
    iou = inter / (csz + nb - inter)
    cov = inter / nb
    pur = inter / csz
    mb = any(int((cm & o).sum()) >= FRAG_PX for o in others)
    return dict(nB=nb, IoU=iou, coverage=cov, purity=pur, merged=pur < 0.5, merged_body=bool(mb), split=cov < 0.5,
                structure=cp.ccls[c] == STRUCT, cstar_cls=CLS_NAME[cp.ccls[c]], cstar_px=int(csz))


def scene_bodies(sd: dict, th) -> dict:
    """Body masks of every person (key ('person', id)) and every target (key ('target', id)) of a scene; a
    person target's body is that person's, so it is keyed once, as the target."""
    truth = sd['sc'].truth
    out = {}
    person_targets = {t.person_id: tid for tid, t in truth.targets.items() if t.kind == 'person'}
    for tid in truth.targets:
        out[('target', tid)] = L.target_masks(sd, tid, th)[1]
    for k, p in enumerate(sd['people']):
        if p.id in person_targets:
            continue
        out[('person', p.id)] = person_body(sd, k)
    return out


def m4_scene(g: Geom, sd: dict, th, tids, joins=('G1', 'G2')) -> list[dict]:
    """M4 rows for the given targets of one scene, under each join."""
    bodies = scene_bodies(sd, th)
    zok = g.W[..., 2] >= FLOOR_Z
    truth = sd['sc'].truth
    rows = []
    for tid in tids:
        t = truth.targets[tid]
        B = bodies[('target', tid)] & zok
        others = [m for k, m in bodies.items() if k != ('target', tid)]
        for join in joins:
            r = recovery(g.comps(join), B, others)
            row = dict(tid=tid, kind=('parcel' if t.kind == 'object' else t.kind), label=t.label, join=join)
            if r is None:
                row['empty'] = True
            else:
                row['empty'] = False
                row.update(r)
            rows.append(row)
    return rows


def m4_table(rows: list[dict], say, title: str) -> None:
    say(title)
    say(f"  {'join':4s} {'kind':7s} {'n':>4s} {'empty':>5s} {'medIoU':>6s} {'IoU>=.5':>8s} {'IoU>=.7':>8s} "
        f"{'merged':>6s} {'mBody':>5s} {'split':>5s} {'struct':>6s}  medCov medPur")
    for join in ('G1', 'G2'):
        for kind in ('person', 'parcel', 'place', 'ALL'):
            rs = [r for r in rows if r['join'] == join and (kind == 'ALL' or r['kind'] == kind)]
            if not rs:
                continue
            ne = [r for r in rs if not r['empty']]
            n = len(ne)
            if n == 0:
                say(f"  {join:4s} {kind:7s} {0:4d} {len(rs) - n:5d}")
                continue
            iou = np.array([r['IoU'] for r in ne])
            say(f"  {join:4s} {kind:7s} {n:4d} {len(rs) - n:5d} {np.median(iou):6.2f} "
                f"{int((iou >= 0.5).sum()):3d}/{n:<4d} {int((iou >= 0.7).sum()):3d}/{n:<4d} "
                f"{sum(r['merged'] for r in ne):6d} {sum(r['merged_body'] for r in ne):5d} "
                f"{sum(r['split'] for r in ne):5d} {sum(r['structure'] for r in ne):6d}  "
                f"{np.median([r['coverage'] for r in ne]):.2f}   {np.median([r['purity'] for r in ne]):.2f}")


# ---------------------------------------------------------------- replay (lifting.py's paths)
_REPLAY = {}


def replay_success(rec, target: GroundedTarget) -> tuple[bool, str]:
    """Full success of a moved record with a new target, by lifting.py's path; cached per (record, target)."""
    key = (rec.ref.folder, rec.scene_id, rec.episode_id, rec.trial, tuple(target.pixel), tuple(target.world))
    if key in _REPLAY:
        return _REPLAY[key]
    if rec.solver_step:
        rs = H.resolve(rec, target=target)
        res = (bool(rs.score.success), 'solver')
    elif rec.result.get('source') == 'standard':
        from sgi.stages.goal import standard_goal
        sc = H.scene(rec.ref, rec.scene_id)
        th = H.recorded_config(rec.ref)[1].thresholds
        sg = standard_goal(sc.observation, target, th)
        if sg.pose_world is None:
            res = (False, 'standard: no pose')
        else:
            res = (bool(H.score_pose(rec, sg.pose_world, target=target).success), 'standard')
    elif rec.pose_world is not None:
        res = (bool(H.score_pose(rec, rec.pose_world, target=target).success),
               'recorded pose (' + str(rec.result.get('source')) + ')')
    else:
        res = (False, 'no pose')
    _REPLAY[key] = res
    return res


# ---------------------------------------------------------------- small reporting helpers
def layout(scene_id: str) -> str:
    """eval_XX without the arm prefix."""
    k = scene_id.find('eval_')
    return scene_id[k:k + 7] if k >= 0 else scene_id


def clusters(rows) -> str:
    rows = list(rows)
    lay = {layout(r['scene']) for r in rows}
    ml = {(r['model'], layout(r['scene'])) for r in rows}
    return f"[layouts {len(lay)}, model x layout {len(ml)}]"


class Log:
    def __init__(self):
        self.lines = []

    def __call__(self, *a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        self.lines.append(s)

    def write(self, path: Path):
        Path(path).write_text('\n'.join(self.lines) + '\n')


def pct(a, b) -> str:
    return f"{a}/{b} ({100.0 * a / b:.1f}%)" if b else f"{a}/0"


Counter = collections.Counter
