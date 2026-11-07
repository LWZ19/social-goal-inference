"""Independent verifier for I03 Stage A: detector and scorer re-implemented from FROZEN.md sections 2 and 4.

Written without importing the compute agent's blobs.py or people_scorer.py. Uses only numpy and the package
(sgi.geometry.lift_all / project, sgi.gt.derive.is_visible, which the note names as definitions).

Detector differences in implementation (not in definition) from the compute agent, on purpose:
- cells are held sparsely (np.unique over floor(xy / 0.05)), components by hook-and-jump union over neighbour edges;
- the local-maximum test is done globally by scanning every offset of the (2k+1)^2 window, restricted to cells of
  the same component (no bounding-box max filter).
Conventions taken from the note and its pre-fit clarifications (DEVIATIONS.md): half-width k = round(split_r/0.05);
touching seed cells are 8-connected; ties for nearest seed go to the first seed cell in (ix, iy) row-major order;
body filter inclusive with 1e-9 tolerance on top_hi and both footprint bounds; band 1.0 <= z <= 1.6 with >= 3 points.
"""
from __future__ import annotations

import math
import statistics

import numpy as np

from sgi.geometry import lift_all, project
from sgi.gt.derive import is_visible

RES = 0.05
EPS = 1e-9
NEAR = 0.5
NEIGH = 0.8
TOL45 = 45.0
FWD = ((0, 1), (1, -1), (1, 0), (1, 1))      # forward half of the 8-neighbourhood


def _cc(n, ea, eb):
    """Connected-component root per node (roots are the smallest index of each component)."""
    parent = np.arange(n)
    if len(ea) == 0:
        return parent
    while True:
        pa, pb = parent[ea], parent[eb]
        lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
        m = lo != hi
        if not m.any():
            return parent
        np.minimum.at(parent, hi[m], lo[m])
        while True:
            pp = parent[parent]
            if np.array_equal(pp, parent):
                break
            parent = pp


def _edges(ij, mask=None):
    """Index pairs of 8-adjacent cells among ij (N,2) (optionally only among cells where mask)."""
    idx = np.arange(len(ij)) if mask is None else np.flatnonzero(mask)
    sub = ij[idx]
    if len(sub) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    lo = sub.min(0) - 1
    shape = sub.max(0) - lo + 2
    grid = -np.ones(tuple(shape), dtype=np.int64)
    g = sub - lo
    grid[g[:, 0], g[:, 1]] = idx
    ea, eb = [], []
    for dx, dy in FWD:
        nb = grid[g[:, 0] + dx, g[:, 1] + dy]
        ok = nb >= 0
        ea.append(idx[ok])
        eb.append(nb[ok])
    return np.concatenate(ea), np.concatenate(eb)


class HeightMap:
    """Cells of the height map of the points in z_lo < z < z_hi (optionally with clip-reaching cells dropped)."""

    def __init__(self, pts, z_lo, z_hi, drop_clip=False):
        p = pts[(pts[:, 2] > z_lo) & (pts[:, 2] < z_hi)]
        ij = np.floor(p[:, :2] / RES).astype(np.int64)
        if len(p):
            cells, inv = np.unique(ij, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            top = np.full(len(cells), -np.inf)
            np.maximum.at(top, inv, p[:, 2])
        else:
            cells, inv, top = np.zeros((0, 2), np.int64), np.zeros(0, np.int64), np.zeros(0)
        if drop_clip and len(cells):
            keep = top < (z_hi - 0.05)
            remap = -np.ones(len(cells), np.int64)
            remap[keep] = np.arange(keep.sum())
            pk = remap[inv] >= 0
            p, inv = p[pk], remap[inv][pk]
            cells, top = cells[keep], top[keep]
        self.p, self.inv, self.cells, self.top = p, inv, cells, top
        n = len(cells)
        ea, eb = _edges(cells)
        root = _cc(n, ea, eb)
        _, self.comp = np.unique(root, return_inverse=True)
        self.comp = self.comp.reshape(-1)
        # dense lookup for offsets
        if n:
            self.lo = cells.min(0)
            shp = cells.max(0) - self.lo + 1
            self.grid = -np.ones(tuple(shp), np.int64)
            g = cells - self.lo
            self.grid[g[:, 0], g[:, 1]] = np.arange(n)
            self.g = g

    def blobs(self, split_r):
        """List of blobs: dict(cells=index array, x, y, top, foot, axis_all, axis_band, n_band)."""
        n = len(self.cells)
        if n == 0:
            return []
        k = int(round(split_r / RES))
        g, grid, top, comp = self.g, self.grid, self.top, self.comp
        W, H = grid.shape
        seed = np.ones(n, bool)
        for dx in range(-k, k + 1):
            for dy in range(-k, k + 1):
                if dx == 0 and dy == 0:
                    continue
                x, y = g[:, 0] + dx, g[:, 1] + dy
                inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                nb = np.full(n, -1, np.int64)
                nb[inb] = grid[x[inb], y[inb]]
                ok = nb >= 0
                ok[ok] = comp[nb[ok]] == comp[ok]
                bigger = np.zeros(n, bool)
                bigger[ok] = top[nb[ok]] > top[ok]
                seed &= ~bigger
        # seed groups: 8-connected among seed cells
        ea, eb = _edges(self.cells, seed)
        sroot = _cc(n, ea, eb)
        label = -np.ones(n, np.int64)
        out = []
        nb_id = 0
        order = np.argsort(comp, kind="stable")
        bounds = np.flatnonzero(np.diff(comp[order])) + 1
        for members in np.split(order, bounds):
            members = np.sort(members)                    # (ix, iy) row-major because cells are lexsorted
            sc = members[seed[members]]
            groups = sroot[sc]
            ug = np.unique(groups)
            if len(ug) <= 1:
                label[members] = nb_id
                nb_id += 1
                continue
            gid = {int(r): i for i, r in enumerate(ug)}
            d = ((self.cells[members][:, None, :] - self.cells[sc][None, :, :]) ** 2).sum(-1)
            j = d.argmin(1)                               # first minimum = first seed cell in row-major order
            label[members] = nb_id + np.array([gid[int(r)] for r in groups[j]])
            nb_id += len(ug)
        pt_label = label[self.inv]
        for b in range(nb_id):
            cidx = np.flatnonzero(label == b)
            cen = (self.cells[cidx] + 0.5) * RES
            t = float(top[cidx].max())
            if len(cidx) >= 2:
                c = cen - cen.mean(0)
                w, v = np.linalg.eigh(c.T @ c / len(cen))
                pr = cen @ v[:, 1]
                foot = float(pr.max() - pr.min() + RES)
            else:
                foot = RES
            out.append({"cells": cidx, "x": float(cen[:, 0].mean()), "y": float(cen[:, 1].mean()), "top": t,
                        "foot": foot, "_b": b})
        self._pt_label = pt_label
        return out

    def axes(self, blob):
        pts = self.p[self._pt_label == blob["_b"]]
        res = {}
        for name, sel in (("all", np.ones(len(pts), bool)), ("band", (pts[:, 2] >= 1.0) & (pts[:, 2] <= 1.6))):
            q = pts[sel]
            if len(q) < (3 if name == "band" else 2):
                res[name] = None
                continue
            c = q[:, :2] - q[:, :2].mean(0)
            w, v = np.linalg.eigh(c.T @ c / len(q))
            res[name] = math.atan2(v[1, 0], v[0, 0])
        return res


def body(blob, s, z_hi):
    return (s["top_lo"] <= blob["top"] <= z_hi - 0.05 + EPS) and (s["foot_lo"] - EPS <= blob["foot"] <= s["foot_hi"] + EPS)


# ------------------------------------------------------------------------------------------------- scorer
def wrapdeg(a):
    d = math.degrees(a) % 360.0
    return d - 360.0 if d > 180.0 else d


def match(dets, vis):
    """Greedy nearest-first one-to-one within 0.5 m; ties lower number then earlier visible. number = index + 1."""
    pairs = sorted((math.dist(d[:2], (p.x, p.y)), i + 1, j) for i, d in enumerate(dets) for j, p in enumerate(vis))
    m, taken = {}, set()
    for gap, num, j in pairs:
        if gap <= NEAR and num not in m and j not in taken:
            m[num] = j
            taken.add(j)
    return m


def lower_hidden(p, obs):
    cam, depth = obs.camera, obs.depth
    hits = 0
    for z in (0.3, 0.5, 0.7):
        pr = project(cam, (p.x, p.y, z))
        if pr is None:
            continue
        u, v, zc = pr
        if not (0 <= u < cam.width and 0 <= v < cam.height):
            continue
        d = float(depth[int(math.floor(v)), int(math.floor(u))])
        if math.isfinite(d) and d > 0 and d < zc - 0.3:
            hits += 1
    return hits >= 2


def role_class(truth):
    sits = {}
    for ep in truth.episodes.values():
        g = ep.group_at_target
        for gid in (g.values() if isinstance(g, dict) else [g]):
            if gid is not None:
                sits.setdefault(gid, set()).add(ep.situation)
    out = {}
    for p in truth.people:
        if p.role:
            out[p.id] = "staff"
        else:
            s = sits.get(p.group, set()) if p.group is not None else set()
            out[p.id] = "line" if "waiting_line" in s else ("group" if s & {"conversation_group", "audience", "group_photo"} else "other")
    return out


class Truth:
    def __init__(self, obs, truth, th):
        assert th.person_radius_m == 0.25 and th.visibility_min_samples == 3 and th.visibility_depth_tolerance_m == 0.3
        self.vis = [p for p in truth.people if is_visible(p, obs, th)]
        self.hid = [p for p in truth.people if not is_visible(p, obs, th)]
        self.exempt = [(t.anchor[0], t.anchor[1]) for t in truth.targets.values() if t.person_id is None]
        self.roles = role_class(truth)
        self.lh = {p.id: lower_hidden(p, obs) for p in self.vis}


def score(dets, T, exempt=None):
    """dets: list of (x, y, yaw or None). Returns counts."""
    vis = T.vis
    exempt = T.exempt if exempt is None else exempt
    m = match(dets, vis)
    mj = set(m.values())
    false = ex = nh = 0
    for i, d in enumerate(dets):
        if i + 1 in m:
            continue
        if any(math.dist(d[:2], e) <= NEAR for e in exempt):
            ex += 1
            continue
        false += 1
        if any(math.dist(d[:2], (p.x, p.y)) <= NEAR for p in T.hid):
            nh += 1
    merges = 0
    for d in dets:
        near = [j for j, p in enumerate(vis) if math.dist(d[:2], (p.x, p.y)) <= NEAR]
        if len(near) >= 2 and sum(j in mj for j in near) <= 1:
            merges += 1
    npairs = mpairs = 0
    for a in range(len(vis)):
        for b in range(a + 1, len(vis)):
            pa, pb = vis[a], vis[b]
            if math.dist((pa.x, pa.y), (pb.x, pb.y)) < NEIGH:
                npairs += 1
                if not (a in mj and b in mj) and any(math.dist(d[:2], (pa.x, pa.y)) <= NEAR and
                                                     math.dist(d[:2], (pb.x, pb.y)) <= NEAR for d in dets):
                    mpairs += 1
    errs, known, unknown, wf, wa = [], 0, 0, 0, 0
    for num, j in m.items():
        d, p = dets[num - 1], vis[j]
        errs.append(math.dist(d[:2], (p.x, p.y)))
        if d[2] is None:
            unknown += 1
            continue
        known += 1
        e = abs(wrapdeg(d[2] - p.yaw))
        wf += e <= TOL45
        wa += min(e, 180 - e) <= TOL45
    misses = [{"id": p.id, "role": p.role, "cls": T.roles[p.id], "lh": T.lh[p.id],
               "staff_behind": bool(p.role) and T.lh[p.id]} for j, p in enumerate(vis) if j not in mj]
    people = [(T.roles[p.id], j in mj) for j, p in enumerate(vis)]
    return {"n_det": len(dets), "vis": len(vis), "matched": len(m), "false": false, "exempt": ex, "near_hidden": nh,
            "merges": merges, "npairs": npairs, "mpairs": mpairs, "errs": errs, "known": known, "unknown": unknown,
            "w45_full": wf, "w45_axis": wa, "misses": misses, "people": people,
            "matches": {num: vis[j].id for num, j in m.items()}}


SUMK = ("n_det", "vis", "matched", "false", "exempt", "near_hidden", "merges", "npairs", "mpairs", "known",
        "unknown", "w45_full", "w45_axis")


def pool(rows):
    out = {k: 0 for k in SUMK}
    out.update(scenes=0, errs=[], misses=[], people=[])
    for r in rows:
        out["scenes"] += 1
        for k in SUMK:
            out[k] += r[k]
        out["errs"] += r["errs"]
        out["misses"] += r["misses"]
        out["people"] += r["people"]
    out["recall"] = out["matched"] / out["vis"] if out["vis"] else None
    out["fb"] = out["false"] / out["scenes"] if out["scenes"] else None
    out["mp"] = out["mpairs"] / out["npairs"] if out["npairs"] else None
    out["med_err"] = statistics.median(out["errs"]) if out["errs"] else None
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))
