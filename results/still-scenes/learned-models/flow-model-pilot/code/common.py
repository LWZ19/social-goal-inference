"""C5 flow pilot (FROZEN.md sha256 a60f1d62...767f): shared raster code for the project venv. Never imported by the
torch sidecar. No model, no network, no torch.

Raster (FROZEN.md section 2): 64 x 64 cells of 0.10 m; the origin o is the target point; x runs from the robot's start
to o, y is x turned +90 deg; cell [r, c] is centred at ((c - 31.5) * 0.1, (r - 31.5) * 0.1) m (rows along y, columns
along x). Channels (float32): 0 floor, 1 people, 2 facing cos, 3 facing sin, 4 target, 5 seen face, 6 robot.
"""
from __future__ import annotations

import math
import re
import sys
import zlib

import numpy as np

sys.path.insert(0, 'docs/model_notes/tools')
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1/i14_v0')
import research_harness as H  # noqa: E402,F401
from sgi import solver  # noqa: E402
from sgi.geometry import bearing, wrap_angle  # noqa: E402
from sgi.polygons import point_in_polygon  # noqa: E402

WT = '.-scenes'
ROOT = WT + '/results/2026-10-17-beyond-vlm'
OUT = ROOT + '/phase2/flow_pilot'
SEEDS_DIR = ROOT + '/phase2/seeds/placeholder'

N = 64
RES = 0.10
HALF = N // 2
SCALE = 3.2
EPS = 1e-9
PERSON_DISC_M = 0.25
TARGET_DISC_M = 0.15
ROBOT_DISC_M = 0.30
FACE_LEN_M = 0.5
FACE_PTS = 51
POSES = 32
LABEL_SEED = 20261018
JITTER_M = 0.05
CH = ('floor', 'people', 'face_cos', 'face_sin', 'target', 'seen_face', 'robot')

OFFS = (np.arange(N) - 31.5) * RES                 # centre coordinate of index i, metres
XC, YC = np.meshgrid(OFFS, OFFS)                   # XC[r, c] = OFFS[c] (x), YC[r, c] = OFFS[r] (y)

DEV = ('no_one', 'waiting_line', 'conversation_group', 'audience')
HELD = ('group_photo', 'lift_wait')
HELD_SCENE = re.compile(r'(?:^|_)eval_1[6-9](?![0-9])', re.I)


def assert_dev_scene(scene_id: str, situations) -> None:
    """FROZEN.md section 1: every scene is of development situations and is no held-out row."""
    assert not HELD_SCENE.search(scene_id), f"held-out scene {scene_id}"
    low = scene_id.lower()
    for h in HELD:
        assert h not in low, f"held-out situation in name {scene_id}"
    for s in situations:
        assert s in DEV, f"{scene_id}: situation {s} is not a development situation"


def crc(key: str) -> int:
    return zlib.crc32(key.encode())


class Frame:
    """The raster frame: origin o (world x, y), x axis from the robot's start to o."""

    def __init__(self, origin_xy, robot_xy):
        self.ox, self.oy = float(origin_xy[0]), float(origin_xy[1])
        dx, dy = self.ox - float(robot_xy[0]), self.oy - float(robot_xy[1])
        self.yaw = math.atan2(dy, dx)
        self.dist = math.hypot(dx, dy)
        self.c, self.s = math.cos(self.yaw), math.sin(self.yaw)

    def to_world(self, x, y):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        return self.ox + self.c * x - self.s * y, self.oy + self.s * x + self.c * y

    def to_raster(self, wx, wy):
        dx, dy = np.asarray(wx, dtype=np.float64) - self.ox, np.asarray(wy, dtype=np.float64) - self.oy
        return self.c * dx + self.s * dy, -self.s * dx + self.c * dy

    def as_dict(self):
        return dict(ox=self.ox, oy=self.oy, yaw=self.yaw, dist=self.dist)


def floor_at(fl, wx, wy) -> np.ndarray:
    """Channel 1's rule: selectable read from the 0.05 m solver cell that holds each point (Floor.index); 0 off grid."""
    out = np.zeros(np.shape(wx), dtype=bool)
    flat_x, flat_y, flat_o = np.ravel(wx), np.ravel(wy), out.reshape(-1)
    for i in range(flat_x.size):
        cell = fl.index(float(flat_x[i]), float(flat_y[i]))
        flat_o[i] = cell is not None and bool(fl.selectable[cell])
    return out


def paint(obs, people, fl, frame: Frame, face_normal_robot) -> np.ndarray:
    """The 7-channel raster. people: objects with .xy (world) and .yaw (world or None). fl: sgi.solver.Floor with the
    same people. face_normal_robot: _seen_face's output (robot frame) or None."""
    r = np.zeros((7, N, N), dtype=np.float32)
    wx, wy = frame.to_world(XC, YC)
    r[0] = floor_at(fl, wx, wy)
    best = np.full((N, N), np.inf)
    for p in people:
        px, py = frame.to_raster(p.xy[0], p.xy[1])
        d = np.hypot(XC - px, YC - py)
        inside = d <= PERSON_DISC_M + EPS
        r[1][inside] = 1.0
        take = inside & (d < best)                 # nearest person's values; an exact tie keeps the earlier person
        best = np.where(take, d, best)
        if p.yaw is None:
            r[2][take] = 0.0
            r[3][take] = 0.0
        else:
            rel = float(p.yaw) - frame.yaw
            r[2][take] = math.cos(rel)
            r[3][take] = math.sin(rel)
    r[4] = np.hypot(XC, YC) <= TARGET_DISC_M + EPS
    if face_normal_robot is not None:
        ang = math.atan2(float(face_normal_robot[1]), float(face_normal_robot[0])) + obs.robot.yaw - frame.yaw
        s = np.arange(FACE_PTS) * (FACE_LEN_M / (FACE_PTS - 1))
        x, y = s * math.cos(ang), s * math.sin(ang)
        c = np.floor(x / RES + HALF).astype(int)
        rr = np.floor(y / RES + HALF).astype(int)
        ok = (c >= 0) & (c < N) & (rr >= 0) & (rr < N)
        r[5][rr[ok], c[ok]] = 1.0
    rx, ry = frame.to_raster(obs.robot.x, obs.robot.y)
    r[6] = np.hypot(XC - rx, YC - ry) <= ROBOT_DISC_M + EPS
    return r


def in_union(regions, wx, wy) -> np.ndarray:
    """Whether each world point lies in the union of the regions' polygons (sgi.polygons.point_in_polygon)."""
    wx, wy = np.ravel(np.asarray(wx, dtype=np.float64)), np.ravel(np.asarray(wy, dtype=np.float64))
    out = np.zeros(wx.size, dtype=bool)
    for g in regions:
        v = np.array(g.polygon, dtype=np.float64)
        lo, hi = v.min(axis=0) - 1e-6, v.max(axis=0) + 1e-6
        box = np.flatnonzero((wx >= lo[0]) & (wx <= hi[0]) & (wy >= lo[1]) & (wy <= hi[1]) & ~out)
        for i in box:
            if point_in_polygon((float(wx[i]), float(wy[i])), g.polygon):
                out[i] = True
    return out


def first_holding(regions, wx, wy):
    """Index of the first region (list order) whose polygon holds the point, or None."""
    for i, g in enumerate(regions):
        if point_in_polygon((float(wx), float(wy)), g.polygon):
            return i
    return None


def region_cells_raster(regions, raster, frame: Frame) -> np.ndarray:
    """Flat indices (row-major) of raster cells free in channel 1 whose centre lies in the union of the regions."""
    free = raster[0].reshape(-1) > 0.5
    wx, wy = frame.to_world(XC.reshape(-1), YC.reshape(-1))
    cand = np.flatnonzero(free)
    hold = in_union(regions, wx[cand], wy[cand])
    return cand[hold]


def region_cells_extended(regions, fl, frame: Frame):
    """FROZEN.md 6(i) region coverage (choice): cells of the 0.10 m lattice of the raster frame, extended beyond the
    raster, whose centre lies in the region and on the floor by channel 1's rule. Returns (M, 2) raster-frame centres."""
    pts = []
    for g in regions:
        v = np.array(g.polygon, dtype=np.float64)
        rx, ry = frame.to_raster(v[:, 0], v[:, 1])
        c0, c1 = math.floor(rx.min() / RES + 31.5) - 1, math.ceil(rx.max() / RES + 31.5) + 1
        r0, r1 = math.floor(ry.min() / RES + 31.5) - 1, math.ceil(ry.max() / RES + 31.5) + 1
        cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
        pts.append(np.stack([(cc.ravel() - 31.5) * RES, (rr.ravel() - 31.5) * RES], axis=1))
    if not pts:
        return np.zeros((0, 2))
    allp = np.unique(np.round(np.concatenate(pts), 9), axis=0)
    wx, wy = frame.to_world(allp[:, 0], allp[:, 1])
    hold = in_union(regions, wx, wy)
    allp, wx, wy = allp[hold], wx[hold], wy[hold]
    on = floor_at(fl, wx, wy)
    return allp[on]


def labels_for(key: str, cells: np.ndarray, regions, frame: Frame):
    """FROZEN.md section 3: 32 poses, cells uniform with replacement, jitter U[-0.05, 0.05)^2, rng
    default_rng([20261018, crc32(key)]) (cells drawn first, then the jitter). Heading from the first region holding the
    cell centre: heading.yaw, else bearing(jittered pose, face_point); stored relative to the raster yaw.
    Returns (labels (32, 4) float32, pose world xy (32, 2), pose world heading (32,))."""
    rng = np.random.default_rng([LABEL_SEED, crc(key)])
    pick = cells[rng.integers(0, len(cells), size=POSES)]
    jit = rng.uniform(-JITTER_M, JITTER_M, size=(POSES, 2))
    cx, cy = XC.reshape(-1)[pick], YC.reshape(-1)[pick]
    x, y = cx + jit[:, 0], cy + jit[:, 1]
    wcx, wcy = frame.to_world(cx, cy)
    wx, wy = frame.to_world(x, y)
    heads = np.empty(POSES)
    for i in range(POSES):
        g = regions[first_holding(regions, wcx[i], wcy[i])]
        if g.heading.yaw is not None:
            h = float(g.heading.yaw)
        else:
            h = bearing((float(wx[i]), float(wy[i])), g.heading.face_point)
        heads[i] = h
    rel = np.array([wrap_angle(h - frame.yaw) for h in heads])
    lab = np.stack([x / SCALE, y / SCALE, np.cos(rel), np.sin(rel)], axis=1).astype(np.float32)
    return lab, np.stack([wx, wy], axis=1), heads


def floor_world_cells(cells_robot: np.ndarray, robot) -> np.ndarray:
    """_over_floor's cells (robot frame, grid order) in the world frame, as Floor.centres() computes them."""
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    return np.stack([robot.x + c * cells_robot[:, 0] - s * cells_robot[:, 1],
                     robot.y + s * cells_robot[:, 0] + c * cells_robot[:, 1]], axis=1)
