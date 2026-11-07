"""I03 phase 1b: the Stage A depth-blob detector (FROZEN.md section 2), numpy only.

Fixed steps: (1) lift every valid depth pixel with sgi.geometry.lift_all (floor z = 0); (2) keep z_lo < z < z_hi;
(3) height map on the package's 0.05 m grid, each cell its highest kept point; (4) 8-connected components;
(5) split each component at its local maxima: a seed is a cell equal to the component's maximum within a square
window of half-width split_r (round(split_r / 0.05) cells); touching seed cells (8-connected) are one seed; every
cell joins its nearest seed cell (Euclidean in the grid; ties to the seed met first in row-major order).

Blob features: top = the highest cell; footprint = extent of the cell centres along their first principal axis
+ 0.05 m (0.05 m for a one-cell blob); position = mean of the cell centres.
Body filter: top_lo <= top <= z_hi - 0.05 and foot_lo <= footprint <= foot_hi.
Facing axis (reported only): the minor principal axis of the xy of the blob's kept points, from all of them ("all")
or from those with 1.0 <= z <= 1.6 m ("band"); an angle in (-pi, pi], read modulo pi. The band axis is None when the
blob has fewer than 3 points in the band.

One general fix (ADDENDUM_FIX.md, drop_clip=True): after the height map, every cell whose highest kept point is at
or above z_hi - 0.05 is removed before components; everything after uses the remaining cells only.

Implementation follows rerun/blobpilot/blobpilot.py (components, max_filter, split_component, pca_axes) with the
frozen thresholds as parameters. No model, no network.
"""
from __future__ import annotations

import collections
import itertools
import math

import numpy as np

from sgi.geometry import lift_all

RES = 0.05
NEIGH8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
BAND = (1.0, 1.6)

GRID = {
    "z_lo": (0.10, 0.40),
    "z_hi": (1.90, 2.05, 2.20),
    "top_lo": (1.10, 1.30, 1.50),
    "split_r": (0.15, 0.25, 0.35),
    "foot_lo": (0.15, 0.30),
    "foot_hi": (0.70, 1.00),
}
KEYS = ("z_lo", "z_hi", "top_lo", "split_r", "foot_lo", "foot_hi")


def settings():
    """The 216 settings as dicts, in a fixed order."""
    return [dict(zip(KEYS, combo)) for combo in itertools.product(*(GRID[k] for k in KEYS))]


def setting_key(s) -> str:
    return " ".join(f"{k}={s[k]:.2f}" for k in KEYS)


def components(occ: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected labels of a boolean grid (0 = empty), by breadth-first search in row-major seed order."""
    lab = np.zeros(occ.shape, dtype=np.int32)
    n = 0
    rows, cols = occ.shape
    for r0, c0 in zip(*np.nonzero(occ)):
        if lab[r0, c0]:
            continue
        n += 1
        lab[r0, c0] = n
        q = collections.deque([(r0, c0)])
        while q:
            r, c = q.popleft()
            for dr, dc in NEIGH8:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and occ[rr, cc] and not lab[rr, cc]:
                    lab[rr, cc] = n
                    q.append((rr, cc))
    return lab, n


def max_filter(a: np.ndarray, half: int) -> np.ndarray:
    """Square-window maximum (window 2*half+1), separable, padding with -inf."""
    w = 2 * half + 1
    p = np.pad(a, ((half, half), (0, 0)), constant_values=-np.inf)
    a1 = np.lib.stride_tricks.sliding_window_view(p, w, axis=0).max(axis=-1)
    p = np.pad(a1, ((0, 0), (half, half)), constant_values=-np.inf)
    return np.lib.stride_tricks.sliding_window_view(p, w, axis=1).max(axis=-1)


def split_component(hsub: np.ndarray, mask: np.ndarray, half: int) -> np.ndarray:
    """Sub-labels (1..k, 0 outside) of one component's cells, split at its local maxima (nearest seed cell)."""
    h = np.where(mask, hsub, -np.inf)
    seeds = mask & (h >= max_filter(h, half))
    seed_lab, k = components(seeds)
    out = np.zeros(mask.shape, dtype=np.int32)
    if k <= 1:
        out[mask] = 1
        return out
    cells = np.argwhere(mask)
    seed_cells = np.argwhere(seed_lab > 0)
    seed_ids = seed_lab[seed_lab > 0]
    best = np.full(len(cells), np.inf)
    who = np.zeros(len(cells), dtype=np.int32)
    for start in range(0, len(seed_cells), 512):
        sc = seed_cells[start:start + 512]
        d = ((cells[:, None, :] - sc[None, :, :]) ** 2).sum(-1)
        j = d.argmin(1)
        dj = d[np.arange(len(cells)), j]
        better = dj < best
        best[better] = dj[better]
        who[better] = seed_ids[start:start + 512][j[better]]
    out[cells[:, 0], cells[:, 1]] = who
    return out


def pca_axes(xy: np.ndarray):
    """(major unit vector, minor unit vector) of 2-D points."""
    c = xy - xy.mean(0)
    cov = c.T @ c / max(len(xy), 1)
    w, v = np.linalg.eigh(cov)                       # ascending
    return v[:, 1], v[:, 0]


def lifted(obs) -> np.ndarray:
    return lift_all(obs.depth, obs.camera)


def blobs_band(pts: np.ndarray, z_lo: float, z_hi: float, split_rs, *, facing=True, keep_cells=False,
               drop_clip=False) -> dict:
    """Every blob of the height map of the points in z_lo < z < z_hi, per split radius: a list of dicts with id, x, y,
    top, footprint, cells, and (facing=True) axis_all, axis_band, n_points, n_band."""
    p = pts[(pts[:, 2] > z_lo) & (pts[:, 2] < z_hi)]
    out = {}
    if len(p) == 0:
        return {f"{r:.2f}": [] for r in split_rs}
    ij = np.floor(p[:, :2] / RES).astype(np.int64)
    lo = ij.min(0)
    ij -= lo
    shape = tuple(ij.max(0) + 1)
    hmap = np.full(shape, -np.inf)
    np.maximum.at(hmap, (ij[:, 0], ij[:, 1]), p[:, 2])
    if drop_clip:                                  # ADDENDUM_FIX.md: drop every cell at or above top_hi = z_hi - 0.05
        hmap[hmap >= z_hi - 0.05] = -np.inf
    occ = np.isfinite(hmap)
    if not occ.any():
        return {f"{r:.2f}": [] for r in split_rs}
    lab, n = components(occ)
    idx = np.argwhere(lab > 0)
    comp_ids = lab[idx[:, 0], idx[:, 1]]
    order = np.argsort(comp_ids, kind="stable")
    idx, comp_ids = idx[order], comp_ids[order]
    groups = np.split(idx, np.flatnonzero(np.diff(comp_ids)) + 1)
    for split_r in split_rs:
        half = int(round(split_r / RES))
        blob_lab = np.zeros(shape, dtype=np.int32)
        nb = 0
        for cells in groups:
            comp = lab[cells[0, 0], cells[0, 1]]
            (r0, c0), (r1, c1) = cells.min(0), cells.max(0) + 1
            mask = lab[r0:r1, c0:c1] == comp
            sub = split_component(hmap[r0:r1, c0:c1], mask, half)
            region = blob_lab[r0:r1, c0:c1]
            region[mask] = sub[mask] + nb
            nb += int(sub.max())
        cell_idx = np.argwhere(blob_lab > 0)
        cell_ids = blob_lab[cell_idx[:, 0], cell_idx[:, 1]]
        order = np.argsort(cell_ids, kind="stable")
        cell_idx, cell_ids = cell_idx[order], cell_ids[order]
        starts = np.concatenate([[0], np.flatnonzero(np.diff(cell_ids)) + 1, [len(cell_ids)]])
        if facing:
            point_blob = blob_lab[ij[:, 0], ij[:, 1]]
            porder = np.argsort(point_blob, kind="stable")
            pstarts = np.searchsorted(point_blob[porder], np.arange(1, nb + 2))
        blobs = []
        for a, b in zip(starts[:-1], starts[1:]):
            bid = int(cell_ids[a])
            cells = cell_idx[a:b]
            centres = (cells + lo + 0.5) * RES
            top = float(hmap[cells[:, 0], cells[:, 1]].max())
            if len(cells) >= 2:
                major, _ = pca_axes(centres)
                proj = centres @ major
                foot = float(proj.max() - proj.min() + RES)
            else:
                foot = RES
            blob = {"id": bid, "x": float(centres[:, 0].mean()), "y": float(centres[:, 1].mean()), "top": top,
                    "footprint": foot, "cells": int(len(cells))}
            if keep_cells:                                 # diagnostics only: the blob's cell centres
                blob["cell_xy"] = centres
            if facing and 1.0 <= top and foot <= 1.0 + 1e-9:   # a candidate body under some grid setting
                bp = p[porder[pstarts[bid - 1]:pstarts[bid]]]
                _, minor = pca_axes(bp[:, :2])
                blob["axis_all"] = float(math.atan2(minor[1], minor[0]))
                band = bp[(bp[:, 2] >= BAND[0]) & (bp[:, 2] <= BAND[1])]
                if len(band) >= 3:
                    _, minor_b = pca_axes(band[:, :2])
                    blob["axis_band"] = float(math.atan2(minor_b[1], minor_b[0]))
                else:
                    blob["axis_band"] = None
                blob["n_points"] = int(len(bp))
                blob["n_band"] = int(len(band))
            blobs.append(blob)
        out[f"{split_r:.2f}"] = blobs
    return out


def kept(blobs, s) -> list:
    """The blobs that pass the body filter of setting s, in blob order."""
    top_hi = s["z_hi"] - 0.05
    return [b for b in blobs if s["top_lo"] <= b["top"] <= top_hi + 1e-9 and s["foot_lo"] - 1e-9 <= b["footprint"]
            <= s["foot_hi"] + 1e-9]


def detect(obs, s, *, pts=None, keep_cells=False, drop_clip=False) -> tuple[list, list]:
    """(kept blobs, all blobs) of one observation under setting s; drop_clip applies ADDENDUM_FIX.md."""
    pts = lifted(obs) if pts is None else pts
    allb = blobs_band(pts, s["z_lo"], s["z_hi"], (s["split_r"],), keep_cells=keep_cells,
                      drop_clip=drop_clip)[f"{s['split_r']:.2f}"]
    return kept(allb, s), allb
