"""Independent verifier library for the I06 build (written from FROZEN.md, not from the builder's code).

Pre-checks c1-c8 as FROZEN.md section 3 words them, the family placements as section 4 and build/readings.txt word
the draws, and helpers. Imports only the sgi package (and numpy)."""
from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np

from sgi.geometry import wrap_angle
from sgi.polygons import distance_to_boundary, point_in_polygon
from sgi.scene import load_scene
from sgi.synth import recipes as R

HERE = Path(__file__).resolve().parent
I06 = HERE.parent
FAM_CODE = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
A_GAP_PHI = {1: (1.9, 180.0), 2: (1.6, 135.0), 3: (1.3, 90.0), 4: (1.0, 60.0), 5: (0.8, 30.0), 6: (0.6, 0.0)}
VIEW_INSIDE = math.atan(R.CAMERA_LENS["cx"] / R.CAMERA_LENS["fx"]) - math.radians(4.0)
TRI_R = R._GROUP_NEIGHBOUR_M / 2 / math.sin(math.pi / 3)   # circumradius of a 1.2 m equilateral triangle (0.693 m)


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def poly_dist(p, poly):
    """0 inside, else distance to the boundary (my own edge loop)."""
    if point_in_polygon(p, poly):
        return 0.0
    n = len(poly)
    return min(seg_dist(p[0], p[1], poly[i][0], poly[i][1], poly[(i + 1) % n][0], poly[(i + 1) % n][1])
               for i in range(n))


def sight(robot, x, y):
    """Ground line of sight from the robot's centre, sampled every <= 0.05 m (the package's own sampling)."""
    return np.array(R._sight_line(robot, x, y))


class Base:
    """One reproduced base (my own reproduction: out/bases/seed<N>/<id> and its Layout)."""

    def __init__(self, seed, index, row, scene_id, layout):
        self.seed, self.index, self.row, self.scene_id, self.layout = seed, index, row, scene_id, layout
        self.folder = HERE / "out" / "bases" / f"seed{seed}" / scene_id
        self.inputs, self.truth = load_scene(self.folder)
        obs = self.inputs.observation
        self.robot = obs.robot
        T = np.asarray(obs.camera.T_world_camera)
        self.cam = (float(T[0, 3]), float(T[1, 3]))
        self.people = list(self.truth.people)
        e0 = self.truth.episodes["e0"]
        self.gid = e0.group_at_target
        self.members = [p for p in self.people if p.group == self.gid]
        self.nonmembers = [p for p in self.people if p.group != self.gid]
        self.bounds = self.truth.bounds
        self.obstacles = list(self.truth.obstacles)
        self.targets = list(self.truth.targets.values())
        self.tfoot = [t.footprint for t in self.targets if t.footprint is not None]
        self.forbidden = [r.polygon for ep in self.truth.episodes.values() for r in (ep.forbidden_regions or [])]
        self.rep = tuple(e0.acceptable_regions[0].representative_point)
        self.anchor = tuple(self.targets[0].anchor[:2])
        self.taken = [f.color for f in layout.figures]
        self.groups = {p.group for p in self.people if p.group is not None}

    def off(self, x, y):
        r = self.robot
        return wrap_angle(math.atan2(y - r.y, x - r.x) - r.yaw)

    # ---------------------------------------------------------------- checks: True = passes
    def c1(self, p):
        xy = (p.x, p.y)
        if not point_in_polygon(xy, self.bounds) or distance_to_boundary(xy, self.bounds) < 0.5:
            return False
        polys = self.obstacles + self.tfoot
        return all(poly_dist(xy, q) >= 0.30 for q in polys)

    def c2(self, p, others):
        return all(math.hypot(p.x - o.x, p.y - o.y) >= 0.5 for o in others if o is not p)

    def c3(self, p):
        return math.hypot(p.x - self.robot.x, p.y - self.robot.y) >= 1.5

    def c4(self, p):
        cx, cy = self.cam
        return abs(wrap_angle(math.atan2(p.y - cy, p.x - cx) - self.robot.yaw)) <= VIEW_INSIDE

    def c5a(self, p, others):
        line = sight(self.robot, p.x, p.y)
        for o in others:
            if o is p:
                continue
            if np.min(np.hypot(line[:, 0] - o.x, line[:, 1] - o.y)) < 0.3:
                return False
        for fp in self.tfoot:
            if any(poly_dist((float(a), float(b)), fp) < 0.3 for a, b in line):
                return False
        return True

    def c5b(self, p, others):
        for o in others:
            if o is p:
                continue
            line = sight(self.robot, o.x, o.y)
            if np.min(np.hypot(line[:, 0] - p.x, line[:, 1] - p.y)) < 0.3:
                return False
        return True

    def c5c(self, p):
        wedges = []
        for t in self.targets:
            if t.footprint is not None and self.members:
                ax, ay = t.anchor[:2]
                first = min(self.members, key=lambda m: math.hypot(m.x - ax, m.y - ay))
                wedges.append(sorted((self.off(ax, ay), self.off(first.x, first.y))))
        o = self.off(p.x, p.y)
        w = 0.3 / math.hypot(p.x - self.robot.x, p.y - self.robot.y)
        return not any(lo - w < o < hi + w for lo, hi in wedges)

    def c6(self, p):
        cx, cy = self.cam
        return all(seg_dist(p.x, p.y, cx, cy, t.anchor[0], t.anchor[1]) >= 0.3 for t in self.targets)

    def c7(self, p):
        cx, cy = self.cam
        d = math.hypot(p.x - cx, p.y - cy)
        for t in self.targets:
            if t.footprint is None:
                continue
            ref = math.atan2(t.anchor[1] - cy, t.anchor[0] - cx)
            angs = [wrap_angle(math.atan2(y - cy, x - cx) - ref) for x, y in t.footprint]
            far = max(math.hypot(x - cx, y - cy) for x, y in t.footprint)
            a = wrap_angle(math.atan2(p.y - cy, p.x - cx) - ref)
            w = 0.3 / d
            if d < far and min(angs) - w < a < max(angs) + w:
                return False
        return True

    def c8(self, p):
        xy = (p.x, p.y)
        if any(poly_dist(xy, q) < 0.3 for q in self.forbidden):
            return False
        return all(poly_dist(xy, fp) >= 1.0 for fp in self.tfoot)

    def checks(self, family, p, others):
        """Which of the family's checks the person fails (others = every other person present, base and added)."""
        names = {"A": ["c1", "c2", "c3", "c5a", "c5b", "c5c", "c6"],
                 "E": ["c1", "c2", "c3", "c5a", "c6"]}.get(family,
                                                          ["c1", "c2", "c3", "c4", "c5a", "c5b", "c5c", "c6", "c7", "c8"])
        bad = []
        for n in names:
            f = getattr(self, n)
            ok = f(p, others) if n in ("c2", "c5a", "c5b") else f(p)
            if not ok:
                bad.append(n)
        return bad


def load_bases(seed):
    with open(HERE / "out" / "bases" / f"seed{seed}.pkl", "rb") as f:
        lay = pickle.load(f)
    return {index: Base(seed, index, row, sid, layout) for index, (row, sid, layout) in lay.items()}


# -------------------------------------------------------------------- family placements (replay from the stream)

class NoSpot(Exception):
    pass


def outer_root(f, step=0.02, far=30.0):
    ds = np.arange(0.0, far + step / 2, step)
    vals = np.array([f(d) for d in ds])
    neg = np.flatnonzero(vals < 0)
    if neg.size == 0 or neg[-1] + 1 >= ds.size:
        raise ValueError("no root")
    lo, hi = ds[neg[-1]], ds[neg[-1] + 1]
    while hi - lo > 1e-10:
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return hi


def cluster_centre(b, theta, psi, gap):
    mx = np.array([m.x for m in b.members])
    my = np.array([m.y for m in b.members])
    c0 = (mx.mean(), my.mean())
    corners = [(TRI_R * math.cos(psi + 2 * math.pi * k / 3), TRI_R * math.sin(psi + 2 * math.pi * k / 3)) for k in range(3)]

    def f(d):
        cx, cy = c0[0] + d * math.cos(theta), c0[1] + d * math.sin(theta)
        return min(float(np.min(np.hypot(cx + kx - mx, cy + ky - my))) for kx, ky in corners) - gap

    d = outer_root(f)
    return c0[0] + d * math.cos(theta), c0[1] + d * math.sin(theta), d


def place(rng, b, family, level):
    """One attempt's placement, replayed from the documented draw order. Returns (pairs, info, n_draws_checked)."""
    n0 = len(b.people)
    if family == "A":
        g, phi = A_GAP_PHI[level]
        line = sorted(b.members, key=lambda m: math.hypot(m.x - b.anchor[0], m.y - b.anchor[1]))
        L = np.array([line[-1].x, line[-1].y])
        B = np.array([line[-2].x, line[-2].y]) if len(line) > 1 else np.array(b.anchor)
        u = (L - B) / np.linalg.norm(L - B)
        face = math.atan2(-u[1], -u[0])
        shirts = R._shirts(rng, 1, b.taken)
        s = R._sign(rng)
        x, y = L + g * u
        for k in range(200):
            p, fig = R._person(rng, n0, float(x), float(y), face + s * math.radians(phi), shirts[0])
            if not b.checks("A", p, b.people):
                return [(p, fig)], {"nominal": (float(x), float(y)), "yaw0": face + s * math.radians(phi), "s": s}, k + 1
        raise NoSpot("A")
    if family == "E":
        cam, rep = np.array(b.cam), np.array(b.rep)
        v = (rep - cam) / np.linalg.norm(rep - cam)
        x, y = (rep - 1.5 * v) if level == 1 else (cam + 2.0 * v)
        shirts = R._shirts(rng, 1, b.taken)
        yaw = rng.uniform(-math.pi, math.pi)
        for k in range(200):
            p, fig = R._person(rng, n0, float(x), float(y), yaw, shirts[0])
            if not b.checks("E", p, b.people):
                return [(p, fig)], {"nominal": (float(x), float(y)), "yaw0": yaw}, k + 1
        raise NoSpot("E")
    if family in ("B", "C"):
        gap = 1.2 if family == "B" else 1.5
        group = None
        if family == "C":
            group = next(f"g{k}" for k in range(1, 100) if f"g{k}" not in b.groups)
        shirts = R._shirts(rng, 3, b.taken)
        n = 0
        for _ in range(20):
            for _ in range(200):
                theta = rng.uniform(-math.pi, math.pi)
                psi = rng.uniform(0.0, 2 * math.pi / 3)
                cx, cy, d = cluster_centre(b, theta, psi, gap)
                unit = []
                for k in range(3):
                    turn = psi + 2 * math.pi * k / 3
                    yaw = turn + math.pi if family == "C" else rng.uniform(-math.pi, math.pi)
                    unit.append(R._person(rng, n0 + k, cx + TRI_R * math.cos(turn), cy + TRI_R * math.sin(turn),
                                          yaw, shirts[k], group))
                n += 1
                ppl = [q for q, _ in unit]
                ok = all(all(math.hypot(q.x - o.x, q.y - o.y) >= 1.2 for o in b.nonmembers) for q in ppl)
                ok = ok and all(not b.checks(family, q, [*b.people, *ppl]) for q in ppl)
                if ok:
                    return unit, {"theta": theta, "psi": psi, "centre": (cx, cy), "d": d, "group": group}, n
        raise NoSpot(family)
    if family == "D":
        k_people = level
        mx = [m.x for m in b.members]
        my = [m.y for m in b.members]
        x0, x1, y0, y1 = min(mx) - 2.5, max(mx) + 2.5, min(my) - 2.5, max(my) + 2.5
        shirts = R._shirts(rng, k_people, b.taken)
        n = 0

        def band():
            for _ in range(10_000):
                x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
                near = min(math.hypot(x - a, y - c) for a, c in zip(mx, my))
                if 1.2 <= near <= 2.5:
                    return x, y
            return None

        for _ in range(20):
            placed = []
            for i in range(k_people):
                got = False
                for _ in range(200):
                    spot = band()
                    n += 1
                    if spot is None:
                        continue
                    yaw = rng.uniform(-math.pi, math.pi)
                    p, fig = R._person(rng, n0 + i, spot[0], spot[1], yaw, shirts[i])
                    earlier = [q for q, _ in placed]
                    if any(math.hypot(p.x - o.x, p.y - o.y) < 1.2 for o in [*b.people, *earlier]):
                        continue
                    if b.checks("D", p, [*b.people, *earlier]):
                        continue
                    placed.append((p, fig))
                    got = True
                    break
                if not got:
                    break
            if len(placed) == k_people:
                return placed, {"k": k_people}, n
        raise NoSpot("D")
    raise ValueError(family)


def planned():
    """Every planned (seed, row, family, level), from FROZEN.md sections 2-4."""
    lines = {6, 7, 8, 9, 13}
    out = []
    for seed in (0, 1, 2):
        for row in range(6, 16):
            for fam, levels in (("A", (1, 2, 3, 4, 5, 6)), ("B", (1,)), ("C", (1,)), ("D", (2, 5)), ("E", (1, 2))):
                if fam == "A" and row not in lines:
                    continue
                for lv in levels:
                    out.append((seed, row, fam, lv))
    return out
