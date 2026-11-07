"""Independent verifier of the C5 flow pilot: my own raster, label and region-lattice code, written from FROZEN.md
(sha256 a60f1d62...767f) sections 2-3 and 6(i), plus DEVIATIONS.md's clarification 1 (label rng order) so that labels
can be compared exactly. It does NOT import the compute agent's code (phase2/flow_pilot/code/). Project venv only;
no torch, no network, no model call.

Raster frame: origin o (world x, y); unit x axis u = (o - robot) / |o - robot|; y axis = u turned +90 deg.
Cell [r, c] centre = ((c - 31.5) * 0.1, (r - 31.5) * 0.1) in that frame (c along x, r along y).
"""
from __future__ import annotations

import math
import re
import sys
import zlib

import numpy as np

from pathlib import Path  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402,F401
from sgi import solver  # noqa: E402
from sgi.geometry import bearing, wrap_angle  # noqa: E402
from sgi.polygons import point_in_polygon  # noqa: E402

WT = str(paths.REPO)
PILOT = str(paths.STILL / 'learned-models' / 'flow-model-pilot')
VER = PILOT + '/verify'
SEEDS = str(paths.STILL / 'scene-seeds' / 'phase2' / 'placeholder')
SOFT_DENSITY = str(paths.STILL / 'learned-models' / 'soft-density')  # old phase1d/i04

NC = 64
CELL = 0.10
DEV_SITS = ('no_one', 'waiting_line', 'conversation_group', 'audience')
HELD_SITS = ('group_photo', 'lift_wait')
_HELD_ROW = re.compile(r'eval_1[6-9](?![0-9])')

# raster-frame centres
IDX = np.arange(NC)
CTR = (IDX - 31.5) * CELL                                   # coordinate of index i
GX = np.tile(CTR[None, :], (NC, 1))                        # GX[r, c] = x of column c
GY = np.tile(CTR[:, None], (1, NC))                        # GY[r, c] = y of row r


def check_dev_scene(scene_id, situations):
    assert not _HELD_ROW.search(scene_id.lower()), scene_id
    for h in HELD_SITS:
        assert h not in scene_id.lower(), scene_id
    for s in situations:
        assert s in DEV_SITS, (scene_id, s)


def crc(key: str) -> int:
    return zlib.crc32(key.encode('utf-8'))


class RFrame:
    def __init__(self, o_xy, robot_xy):
        self.o = np.array([float(o_xy[0]), float(o_xy[1])])
        d = self.o - np.array([float(robot_xy[0]), float(robot_xy[1])])
        self.dist = float(np.linalg.norm(d))
        self.ux = d / self.dist                             # x axis (world)
        self.uy = np.array([-self.ux[1], self.ux[0]])       # y axis = x turned +90 deg
        self.yaw = math.atan2(self.ux[1], self.ux[0])

    def world(self, x, y):
        """raster (x, y) -> world (wx, wy)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return self.o[0] + x * self.ux[0] + y * self.uy[0], self.o[1] + x * self.ux[1] + y * self.uy[1]

    def local(self, wx, wy):
        dx = np.asarray(wx, float) - self.o[0]
        dy = np.asarray(wy, float) - self.o[1]
        return dx * self.ux[0] + dy * self.ux[1], dx * self.uy[0] + dy * self.uy[1]


def floor_lookup(fl, wx, wy):
    """Floor.index rule, vectorised: the 0.05 m robot-frame cell holding each world point; selectable there, 0 off
    the grid."""
    wx = np.asarray(wx, float)
    wy = np.asarray(wy, float)
    c, s = math.cos(fl.robot.yaw), math.sin(fl.robot.yaw)
    dx, dy = wx - fl.robot.x, wy - fl.robot.y
    ahead = c * dx + s * dy
    left = c * dy - s * dx
    step = fl.offsets[1] - fl.offsets[0]
    half = len(fl.offsets) // 2
    row = np.floor(left / step + 0.5).astype(np.int64) + half
    col = np.floor(ahead / step + 0.5).astype(np.int64) + half
    n = len(fl.offsets)
    ok = (row >= 0) & (row < n) & (col >= 0) & (col < n)
    out = np.zeros(wx.shape, dtype=bool)
    out[ok] = fl.selectable[row[ok], col[ok]]
    return out


def make_raster(obs, people, fl, fr: RFrame, face_robot):
    """7 channels (float32): floor, people, facing cos, facing sin, target, seen face, robot."""
    R = np.zeros((7, NC, NC), np.float32)
    wx, wy = fr.world(GX, GY)
    R[0] = floor_lookup(fl, wx, wy)
    nearest = np.full((NC, NC), np.inf)
    for p in people:
        d = np.hypot(wx - p.xy[0], wy - p.xy[1])
        disc = d <= 0.25 + 1e-9
        R[1][disc] = 1.0
        upd = disc & (d < nearest)
        nearest[upd] = d[upd]
        if p.yaw is None:
            R[2][upd] = 0.0
            R[3][upd] = 0.0
        else:
            rel = float(p.yaw) - fr.yaw
            R[2][upd] = math.cos(rel)
            R[3][upd] = math.sin(rel)
    R[4] = np.hypot(GX, GY) <= 0.15 + 1e-9
    if face_robot is not None:
        # robot-frame normal -> world -> raster frame, by rotation of the vector
        cy, sy = math.cos(obs.robot.yaw), math.sin(obs.robot.yaw)
        nw = np.array([cy * face_robot[0] - sy * face_robot[1], sy * face_robot[0] + cy * face_robot[1]])
        nr = np.array([nw @ fr.ux, nw @ fr.uy])
        nr = nr / np.linalg.norm(nr)
        s = np.arange(51) * 0.01
        px, py = s * nr[0], s * nr[1]
        c = np.floor(px / CELL).astype(int) + 32
        r = np.floor(py / CELL).astype(int) + 32
        keep = (c >= 0) & (c < NC) & (r >= 0) & (r < NC)
        R[5][r[keep], c[keep]] = 1.0
    rx, ry = fr.local(obs.robot.x, obs.robot.y)
    R[6] = np.hypot(GX - rx, GY - ry) <= 0.3 + 1e-9
    return R


def union_hold(regions, wx, wy):
    wx = np.ravel(np.asarray(wx, float))
    wy = np.ravel(np.asarray(wy, float))
    out = np.zeros(wx.size, bool)
    for i in range(wx.size):
        p = (float(wx[i]), float(wy[i]))
        for g in regions:
            if point_in_polygon(p, g.polygon):
                out[i] = True
                break
    return out


def region_cells(regions, R, fr: RFrame):
    """Flat (row-major) indices of raster cells free in channel 1 whose centre is in the union of the regions,
    ascending. A bounding-box prefilter on the union is used only to skip far cells."""
    allv = np.concatenate([np.asarray(g.polygon, float) for g in regions])
    lo, hi = allv.min(0) - 0.01, allv.max(0) + 0.01
    wx, wy = fr.world(GX, GY)
    wx, wy = wx.ravel(), wy.ravel()
    free = R[0].ravel() > 0.5
    near = free & (wx >= lo[0]) & (wx <= hi[0]) & (wy >= lo[1]) & (wy <= hi[1])
    cand = np.flatnonzero(near)
    return cand[union_hold(regions, wx[cand], wy[cand])]


def first_region(regions, wx, wy):
    for g in regions:
        if point_in_polygon((float(wx), float(wy)), g.polygon):
            return g
    return None


def make_labels(key, cells, regions, fr: RFrame):
    rng = np.random.default_rng([20261018, crc(key)])
    k = rng.integers(0, len(cells), size=32)
    j = rng.uniform(-0.05, 0.05, size=(32, 2))
    sel = cells[k]
    cx, cy = GX.ravel()[sel], GY.ravel()[sel]
    x, y = cx + j[:, 0], cy + j[:, 1]
    wcx, wcy = fr.world(cx, cy)
    wpx, wpy = fr.world(x, y)
    lab = np.zeros((32, 4), np.float32)
    heads = np.zeros(32)
    for i in range(32):
        g = first_region(regions, wcx[i], wcy[i])
        h = float(g.heading.yaw) if g.heading.yaw is not None else bearing((float(wpx[i]), float(wpy[i])), g.heading.face_point)
        heads[i] = h
        rel = wrap_angle(h - fr.yaw)
        lab[i] = (x[i] / 3.2, y[i] / 3.2, math.cos(rel), math.sin(rel))
    return lab, np.stack([wpx, wpy], 1), heads


def ext_cells(regions, fl, fr: RFrame):
    """6(i) coverage lattice: 0.10 m lattice of the raster frame (centres (i - 31.5) * 0.1 for any integer i),
    extended beyond the raster, centre in the union of the regions and on the floor by channel 1's rule.
    Returns (M, 2) raster-frame centres, sorted by (y, x)."""
    allv = np.concatenate([np.asarray(g.polygon, float) for g in regions])
    lx, ly = fr.local(allv[:, 0], allv[:, 1])
    c0, c1 = int(math.floor(lx.min() / CELL + 31.5)) - 2, int(math.ceil(lx.max() / CELL + 31.5)) + 2
    r0, r1 = int(math.floor(ly.min() / CELL + 31.5)) - 2, int(math.ceil(ly.max() / CELL + 31.5)) + 2
    cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
    px, py = (cc.ravel() - 31.5) * CELL, (rr.ravel() - 31.5) * CELL
    wx, wy = fr.world(px, py)
    inside = union_hold(regions, wx, wy)
    onf = floor_lookup(fl, wx, wy)
    keep = inside & onf
    return np.stack([px[keep], py[keep]], 1)


def floor_cells_world(cells_robot, robot):
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    return np.stack([robot.x + c * cells_robot[:, 0] - s * cells_robot[:, 1],
                     robot.y + s * cells_robot[:, 0] + c * cells_robot[:, 1]], 1)
