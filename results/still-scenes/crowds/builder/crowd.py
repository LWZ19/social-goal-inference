"""I06 crowd builder (research code). It imports the sgi package and changes nothing in it.

Implements FROZEN.md (sha256 a83c0ffe...693e) sections 2-4:
- reproduce the 30 base layouts (rows eval_06..15 of seeds 0, 1, 2) with the package's own accept step and streams;
- add non-member people to a base layout by families A-E, each variant from its own stream
  default_rng([seed, 2, family code, row, level]);
- render and check each variant through the package's scene path (build_scene, save_scene, _rejection,
  _with_regions), with at most 6 attempts.

Nothing here reads a situation label inside a method; the builder reads the base row (recipe) only to decide where a
family applies, which is scene construction. No model, no network, no torch.

Builder readings of points the note leaves open (logged in build/readings.txt):
- Draw order inside one attempt: shirts (recipes._shirts, once per attempt, k = people added, taken = colours already
  worn), then the family's own draws in the order written in each place_* function.
- A and E draw their facing sign (A) or facing (E) once per attempt; each of the 200 inner draws is one `_person` call
  (its jitter).
- D: one "position draw" is one uniform sample of the band [1.2, 2.5] m round the members, made by rejection in the
  members' bounding box grown by 2.5 m (at most 10,000 tries; a band never found counts as a failed draw). The band is
  the sampling distribution, applied before the jitter; the spacing rule and c1-c8 are applied to the jittered person.
- B and C: the bisection is made before the jitter (as the note orders: bisect, then jitter and checks). The root
  taken is the outermost one along the bearing: the smallest distance beyond which the cluster stays at or past the
  gap.
- c6 uses the exact distance from a person's centre to the camera-to-anchor ground segment; c5 samples lines of sight
  every 0.05 m exactly as recipes._in_clear_view does (recipes._sight_line).
- c7 applies to targets with a footprint; a person target has none (c5(b) keeps the target person clear).
- "present" for c5(b) = the base's people plus the people of this variant placed earlier in the same attempt.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from sgi.config import Config, load_config
from sgi.geometry import RobotPose, camera_on_robot, wrap_angle
from sgi.gt.derive import _VISIBILITY_HEIGHTS_M, _sample_seen, prepare
from sgi.polygons import _segment_distance, distance_to_boundary, distance_to_polygon, point_in_polygon
from sgi.scene import Person, load_scene, save_scene
from sgi.synth import generate as G
from sgi.synth import recipes as R

I06 = Path(__file__).resolve().parents[1]          # phase2/i06
SCENES = I06 / "scenes"
BUILD = I06 / "build"

BASE_ROWS = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
FIRST_PASS_LAST = 13                               # eval_13 is the last first-pass row drawn
SECOND_PASS_ROWS = 2                               # eval_14, eval_15: drawing stops before eval_16 (group_photo)
HELD_OUT_RECIPES = ("group_photo", "lift_wait")
FAMILY_CODE = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
LEVELS = {"A": (1, 2, 3, 4, 5, 6), "B": (1,), "C": (1,), "D": (2, 5), "E": (1, 2)}
A_LEVELS = {1: (1.9, 180.0), 2: (1.6, 135.0), 3: (1.3, 90.0), 4: (1.0, 60.0), 5: (0.8, 30.0), 6: (0.6, 0.0)}
MAX_ATTEMPTS = 6                                   # 1 + at most 5 redraws
DRAWS = 200                                        # per person, cluster or group (as _on_free_spot)
RESTARTS = 20                                      # of the whole set (as _add_extras)
BAND_TRIES = 10_000                                # D: rejection tries for one band sample
SPACING_M = 1.2                                    # B, C: from other non-members; D: from every person
CLUSTER_SIDE_M = R._GROUP_NEIGHBOUR_M              # 1.2 m: B's triangle side and C's neighbour distance
CLUSTER_RADIUS_M = R._GROUP_NEIGHBOUR_M / 2 / math.sin(math.pi / 3)     # 0.693 m
B_GAP_M, C_GAP_M = 1.2, 1.5                        # nearest pair to a member
D_BAND_M = (1.2, 2.5)
E_NEAR_STRUCTURE_M, E_NEAR_ROBOT_M = 1.5, 2.0
BODY_SPACING_M = 0.5                               # c2: two person radii
ROOM_MARGIN_M = 0.5                                # c1
OBSTACLE_GAP_M = 0.30                              # c1: person radius + 0.05
ROBOT_GAP_M = 1.5                                  # c3
FORBIDDEN_GAP_M = 0.3                              # c8
FOOTPRINT_GAP_M = 1.0                              # c8
SIGHT_GAP_M = R._SIGHT_GAP_M                       # 0.3 m
CHECKS_ALL = ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")
CHECKS = {"A": ("c1", "c2", "c3", "c5", "c6"), "B": CHECKS_ALL, "C": CHECKS_ALL, "D": CHECKS_ALL,
          "E": ("c1", "c2", "c3", "c5a", "c6")}


def applies(family: str, row: tuple) -> bool:
    """Where a family applies (scene construction): A to waiting lines only, B-E to every base row."""
    return row[0] == "waiting_line" if family == "A" else True


def stream_key(seed: int, family: str, row_index: int, level: int) -> list[int]:
    return [seed, 2, FAMILY_CODE[family], row_index, level]


def scene_id_of(seed: int, family: str, level: int, row_index: int, name: str) -> str:
    return f"i06_s{seed}_{family}{level}_eval_{row_index:02d}_{name}"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def folder_hashes(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): sha256_file(p) for p in sorted(Path(folder).rglob("*")) if p.is_file()}


# ----------------------------------------------------------------------------------------------- base layouts

@dataclasses.dataclass
class BaseLayout:
    seed: int
    index: int                                     # the row's eval index, 6..15
    row: tuple                                     # the package's content row
    stream: str                                    # "default_rng(seed)" or "default_rng([seed, 1])"
    layout: R.Layout
    scene: object                                  # SceneFile with regions (no tour)
    rgb: np.ndarray
    depth: np.ndarray
    rejected_before: dict                          # rejections of this row's draws, by reason


def reproduce_bases(seed: int, cfg: Config, work_dir: Path):
    """The 10 base layouts of a seed, in generation order, each as a BaseLayout.

    First pass: default_rng(seed), rows eval_00..eval_13 through generate._accept (eval_00..05 only advance the stream).
    Second pass: default_rng([seed, generate._MORE_STREAM]), rows eval_14 and eval_15, then drawing stops, before the
    group_photo row eval_16. Folders are written under a new temporary folder in work_dir and removed."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"bases_s{seed}_", dir=work_dir))
    try:
        split_dir = run_dir / "eval"               # the split's name is the first part of a scene's id
        counts = dict.fromkeys(G._REASONS, 0)
        rng = np.random.default_rng(seed)
        for index, row in enumerate(G._EVAL[:FIRST_PASS_LAST + 1]):
            assert row[0] not in HELD_OUT_RECIPES
            before = dict(counts)
            layout, scene, rgb, depth = G._accept(split_dir, index, row, rng, cfg, counts)
            folder = split_dir / scene.scene_id
            if index in BASE_ROWS:
                scene_r = G._with_regions(folder, scene, cfg)
                yield BaseLayout(seed, index, row, "default_rng(seed)", layout, scene_r, rgb, depth,
                                 {k: counts[k] - before[k] for k in counts if counts[k] != before[k]})
            shutil.rmtree(folder)
        more = np.random.default_rng([seed, G._MORE_STREAM])
        start = len(G._EVAL)
        for index, row in enumerate(G._EVAL_MORE[:SECOND_PASS_ROWS], start=start):
            assert row[0] == "audience" and row[0] not in HELD_OUT_RECIPES, row
            before = dict(counts)
            layout, scene, rgb, depth = G._accept(split_dir, index, row, more, cfg, counts)
            folder = split_dir / scene.scene_id
            scene_r = G._with_regions(folder, scene, cfg)
            yield BaseLayout(seed, index, row, f"default_rng([seed, {G._MORE_STREAM}])", layout, scene_r, rgb, depth,
                             {k: counts[k] - before[k] for k in counts if counts[k] != before[k]})
            shutil.rmtree(folder)
        # nothing further is drawn from `more`: eval_16 (group_photo) and every later row are never instantiated
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


# ------------------------------------------------------------------------------------------------- pre-checks

class NoSpot(RuntimeError):
    def __init__(self, message: str, fails: Counter):
        super().__init__(message)
        self.fails = fails


class Context:
    """The base geometry the pre-checks read."""

    def __init__(self, base: BaseLayout):
        layout = base.layout
        self.base = base
        self.layout = layout
        self.robot: RobotPose = layout.robot
        T = camera_on_robot(layout.robot, **R.CAMERA_MOUNT)
        self.cam_xy = (float(T[0, 3]), float(T[1, 3]))
        gt = layout.scene_gt
        self.people: list[Person] = list(gt.people)
        self.bounds = gt.bounds.polygon
        self.obstacles = [o.polygon for o in gt.obstacles]
        self.footprints = [list(fp) for fp in layout.footprints]            # what the scene is about (c5a)
        self.targets = list(gt.targets)
        self.target_footprints = [t.footprint for t in gt.targets if t.footprint is not None]
        e0 = next(ep for ep in base.scene.episodes if ep.episode_id == "e0")
        self.group_id = e0.ground_truth.group_at_target
        assert isinstance(self.group_id, str), self.group_id
        self.members = [p for p in self.people if p.group == self.group_id]
        self.non_members = [p for p in self.people if p.group != self.group_id]
        self.forbidden = [r.polygon for ep in base.scene.episodes for r in (ep.ground_truth.forbidden_regions or [])]
        regions = e0.ground_truth.acceptable_regions
        self.rep_point = tuple(regions[0].representative_point)
        self.anchor_xy = tuple(self.targets[0].anchor[:2])
        self.taken = [figure.color for figure in layout.figures]
        self.groups_taken = {p.group for p in self.people if p.group is not None}
        # c5(c): the bearings between a place's anchor and the first in line, as recipes._in_clear_view finds them
        self.served = []
        waiting = [p for p in self.people if p.group == self.group_id]
        for target in self.targets:
            if target.footprint is not None and waiting:
                ax, ay = target.anchor[:2]
                first = min(waiting, key=lambda p: math.hypot(p.x - ax, p.y - ay))
                self.served.append(sorted((self._off_heading(ax, ay), self._off_heading(first.x, first.y))))
        self._lines = {}

    def _off_heading(self, x: float, y: float) -> float:
        robot = self.robot
        return wrap_angle(math.atan2(y - robot.y, x - robot.x) - robot.yaw)

    def sight_line(self, x: float, y: float):
        key = (x, y)
        if key not in self._lines:
            self._lines[key] = np.array(R._sight_line(self.robot, x, y))
        return self._lines[key]

    # each check: True when the person passes it
    def c1(self, p: Person, **_) -> bool:
        xy = (p.x, p.y)
        if not point_in_polygon(xy, self.bounds) or distance_to_boundary(xy, self.bounds) < ROOM_MARGIN_M:
            return False
        return all(distance_to_polygon(xy, poly) >= OBSTACLE_GAP_M for poly in self.obstacles)

    def c2(self, p: Person, others: Sequence[Person], **_) -> bool:
        return all(math.hypot(p.x - o.x, p.y - o.y) >= BODY_SPACING_M for o in others)

    def c3(self, p: Person, **_) -> bool:
        return math.hypot(p.x - self.robot.x, p.y - self.robot.y) >= ROBOT_GAP_M

    def c4(self, p: Person, **_) -> bool:
        cx, cy = self.cam_xy
        off = wrap_angle(math.atan2(p.y - cy, p.x - cx) - self.robot.yaw)
        return abs(off) <= R._VIEW_INSIDE_RAD

    def c5a(self, p: Person, others: Sequence[Person], **_) -> bool:
        line = self.sight_line(p.x, p.y)
        for o in others:
            if np.any(np.hypot(line[:, 0] - o.x, line[:, 1] - o.y) < SIGHT_GAP_M):
                return False
        for fp in self.footprints:
            if any(distance_to_polygon((float(x), float(y)), fp) < SIGHT_GAP_M for x, y in line):
                return False
        return True

    def c5b(self, p: Person, present: Sequence[Person], **_) -> bool:
        for o in present:
            line = self.sight_line(o.x, o.y)
            if np.any(np.hypot(line[:, 0] - p.x, line[:, 1] - p.y) < SIGHT_GAP_M):
                return False
        return True

    def c5c(self, p: Person, **_) -> bool:
        off = self._off_heading(p.x, p.y)
        wide = SIGHT_GAP_M / math.hypot(p.x - self.robot.x, p.y - self.robot.y)
        return not any(low - wide < off < high + wide for low, high in self.served)

    def c5(self, p: Person, others, present, **_) -> bool:
        return self.c5a(p, others=others) and self.c5b(p, present=present) and self.c5c(p)

    def c6(self, p: Person, **_) -> bool:
        cx, cy = self.cam_xy
        for target in self.targets:
            ax, ay = target.anchor[:2]
            if _segment_distance(p.x, p.y, cx, cy, ax, ay) < SIGHT_GAP_M:
                return False
        return True

    def c7(self, p: Person, **_) -> bool:
        cx, cy = self.cam_xy
        d = math.hypot(p.x - cx, p.y - cy)
        for target, fp in ((t, t.footprint) for t in self.targets if t.footprint is not None):
            ref = math.atan2(target.anchor[1] - cy, target.anchor[0] - cx)
            offs = [wrap_angle(math.atan2(y - cy, x - cx) - ref) for x, y in fp]
            far = max(math.hypot(x - cx, y - cy) for x, y in fp)
            wide = SIGHT_GAP_M / d
            off = wrap_angle(math.atan2(p.y - cy, p.x - cx) - ref)
            if d < far and min(offs) - wide < off < max(offs) + wide:
                return False
        return True

    def c8(self, p: Person, **_) -> bool:
        xy = (p.x, p.y)
        if any(distance_to_polygon(xy, poly) < FORBIDDEN_GAP_M for poly in self.forbidden):
            return False
        return all(distance_to_polygon(xy, fp) >= FOOTPRINT_GAP_M for fp in self.target_footprints)

    def first_failure(self, p: Person, checks: Sequence[str], unit: Sequence[Person] = (),
                      earlier: Sequence[Person] = ()) -> str | None:
        """The first check (in the order given) the person fails, or None.

        `unit` holds the other people placed together with this one (a cluster or a group); `earlier` the people of
        this variant placed before them. Others for c2 and c5(a) = base people + earlier + unit; present for c5(b) =
        base people + earlier."""
        present = [*self.people, *earlier]
        others = [*present, *(u for u in unit if u is not p)]
        for name in checks:
            if not getattr(self, name)(p, others=others, present=present):
                return name
        return None


# ------------------------------------------------------------------------------------------------- families

def _outermost_root(f, step: float = 0.02, far: float = 30.0) -> float:
    """The smallest d >= 0 beyond which f(d) >= 0 stays (f vectorised over an array of d), to 1e-10."""
    ds = np.arange(0.0, far + step / 2, step)
    values = f(ds)
    negative = np.flatnonzero(values < 0)
    if negative.size == 0:
        raise ValueError("the cluster already keeps the gap at the members' centroid")
    i = int(negative[-1])
    if i + 1 >= ds.size:
        raise ValueError("no root within reach")
    lo, hi = float(ds[i]), float(ds[i + 1])
    while hi - lo > 1e-10:
        mid = (lo + hi) / 2
        if f(np.array([mid]))[0] < 0:
            lo = mid
        else:
            hi = mid
    return hi


def _cluster_centre(ctx: Context, theta: float, psi: float, gap: float) -> tuple[float, float]:
    """The centre of a 3-person triangle (circumradius CLUSTER_RADIUS_M, turned by psi) on the bearing theta from the
    members' centroid, at the distance where its nearest corner is exactly `gap` from the nearest member."""
    mx = np.array([m.x for m in ctx.members])
    my = np.array([m.y for m in ctx.members])
    cx0, cy0 = float(mx.mean()), float(my.mean())
    ux, uy = math.cos(theta), math.sin(theta)
    corners = [(CLUSTER_RADIUS_M * math.cos(psi + 2 * math.pi * k / 3),
                CLUSTER_RADIUS_M * math.sin(psi + 2 * math.pi * k / 3)) for k in range(3)]

    def f(ds):
        best = np.full(ds.shape, np.inf)
        for kx, ky in corners:
            px = cx0 + ds * ux + kx
            py = cy0 + ds * uy + ky
            dist = np.hypot(px[:, None] - mx[None, :], py[:, None] - my[None, :]).min(axis=1)
            best = np.minimum(best, dist)
        return best - gap

    d = _outermost_root(f)
    return cx0 + d * ux, cy0 + d * uy


def place_A(rng, ctx: Context, level: int, fails: Counter):
    g, phi = A_LEVELS[level]
    anchor = np.array(ctx.anchor_xy)
    line = sorted(((m.x, m.y) for m in ctx.members), key=lambda q: math.dist(anchor, q))   # nearest first
    last = np.array(line[-1])
    before = np.array(line[-2]) if len(line) > 1 else anchor
    u = (last - before) / np.linalg.norm(last - before)
    facing = math.atan2(-u[1], -u[0])
    shirts = R._shirts(rng, 1, ctx.taken)
    s = R._sign(rng)
    x, y = last + g * u
    yaw = facing + s * math.radians(phi)
    for _ in range(DRAWS):
        person, figure = R._person(rng, len(ctx.people), float(x), float(y), yaw, shirts[0])
        why = ctx.first_failure(person, CHECKS["A"])
        if why is None:
            return [(person, figure)], {"s": s, "gap_m": g, "phi_deg": phi, "nominal": [float(x), float(y)]}
        fails[why] += 1
    raise NoSpot(f"A{level}: {DRAWS} jitter draws failed", fails)


def _spaced(p: Person, others: Sequence[Person], gap: float) -> bool:
    return all(math.hypot(p.x - o.x, p.y - o.y) >= gap for o in others)


def place_cluster(rng, ctx: Context, family: str, fails: Counter):
    """B: three non-members on an equilateral triangle (1.2 m sides), random yaws, nearest pair 1.2 m from a member.
    C: a second group of three, the package's conversation geometry, facing the centre, nearest pair 1.5 m."""
    gap = B_GAP_M if family == "B" else C_GAP_M
    group = None
    if family == "C":
        group = next(f"g{k}" for k in range(1, 100) if f"g{k}" not in ctx.groups_taken)
    shirts = R._shirts(rng, 3, ctx.taken)
    n0 = len(ctx.people)
    for _ in range(RESTARTS):
        for _ in range(DRAWS):
            theta = rng.uniform(-math.pi, math.pi)
            psi = rng.uniform(0.0, 2 * math.pi / 3)
            cx, cy = _cluster_centre(ctx, theta, psi, gap)
            unit = []
            for k in range(3):
                turn = psi + 2 * math.pi * k / 3
                x, y = cx + CLUSTER_RADIUS_M * math.cos(turn), cy + CLUSTER_RADIUS_M * math.sin(turn)
                yaw = turn + math.pi if family == "C" else rng.uniform(-math.pi, math.pi)
                unit.append(R._person(rng, n0 + k, x, y, yaw, shirts[k], group))
            people = [p for p, _ in unit]
            why = None
            for p in people:
                if not _spaced(p, ctx.non_members, SPACING_M):
                    why = "spacing"
                    break
                why = ctx.first_failure(p, CHECKS[family], unit=people)
                if why is not None:
                    break
            if why is None:
                return unit, {"theta": theta, "psi": psi, "centre": [cx, cy], "gap_m": gap, "group": group}
            fails[why] += 1
    raise NoSpot(f"{family}: {RESTARTS} x {DRAWS} draws failed", fails)


def place_D(rng, ctx: Context, k: int, fails: Counter):
    mx = [m.x for m in ctx.members]
    my = [m.y for m in ctx.members]
    x0, x1 = min(mx) - D_BAND_M[1], max(mx) + D_BAND_M[1]
    y0, y1 = min(my) - D_BAND_M[1], max(my) + D_BAND_M[1]
    shirts = R._shirts(rng, k, ctx.taken)
    n0 = len(ctx.people)

    def band():
        for _ in range(BAND_TRIES):
            x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
            near = min(math.hypot(x - a, y - b) for a, b in zip(mx, my))
            if D_BAND_M[0] <= near <= D_BAND_M[1]:
                return x, y
        return None

    for _ in range(RESTARTS):
        placed: list = []
        for i in range(k):
            earlier = [p for p, _ in placed]
            for _ in range(DRAWS):
                spot = band()
                if spot is None:
                    fails["band"] += 1
                    continue
                yaw = rng.uniform(-math.pi, math.pi)
                person, figure = R._person(rng, n0 + i, spot[0], spot[1], yaw, shirts[i])
                if not _spaced(person, [*ctx.people, *earlier], SPACING_M):
                    fails["spacing"] += 1
                    continue
                why = ctx.first_failure(person, CHECKS["D"], earlier=earlier)
                if why is not None:
                    fails[why] += 1
                    continue
                placed.append((person, figure))
                break
            else:
                break                              # this person found no spot: the whole set starts again
        if len(placed) == k:
            return placed, {"k": k}
        fails["restart"] += 1
    raise NoSpot(f"D{k}: {RESTARTS} restarts failed", fails)


def place_E(rng, ctx: Context, level: int, fails: Counter):
    cam = np.array(ctx.cam_xy)
    rep = np.array(ctx.rep_point)
    length = float(np.linalg.norm(rep - cam))
    v = (rep - cam) / length
    if level == 1:
        x, y = rep - E_NEAR_STRUCTURE_M * v
    else:
        x, y = cam + E_NEAR_ROBOT_M * v
    shirts = R._shirts(rng, 1, ctx.taken)
    yaw = rng.uniform(-math.pi, math.pi)
    for _ in range(DRAWS):
        person, figure = R._person(rng, len(ctx.people), float(x), float(y), yaw, shirts[0])
        why = ctx.first_failure(person, CHECKS["E"])
        if why is None:
            return [(person, figure)], {"nominal": [float(x), float(y)], "rep_point": list(ctx.rep_point),
                                        "camera_xy": list(ctx.cam_xy), "camera_to_rep_m": length}
        fails[why] += 1
    raise NoSpot(f"E{level}: {DRAWS} jitter draws failed", fails)


def place(rng, ctx: Context, family: str, level: int, fails: Counter):
    if family == "A":
        return place_A(rng, ctx, level, fails)
    if family in ("B", "C"):
        return place_cluster(rng, ctx, family, fails)
    if family == "D":
        return place_D(rng, ctx, level, fails)
    if family == "E":
        return place_E(rng, ctx, level, fails)
    raise ValueError(family)


# ------------------------------------------------------------------------------------------------- variants

def variant_layout(layout: R.Layout, added) -> R.Layout:
    """The base Layout with the added (Person, Figure) pairs appended: figures[i] still draws people[i]. The room,
    bounds, furnishings, targets, robot and camera are the base's."""
    gt = layout.scene_gt
    people = [*gt.people, *(person for person, _ in added)]
    figures = (*layout.figures, *(figure for _, figure in added))
    return dataclasses.replace(layout, figures=figures, scene_gt=gt.model_copy(update={"people": people}))


def build_variant(base: BaseLayout, family: str, level: int, cfg: Config, out_dir: Path) -> dict:
    """Generate one variant: at most MAX_ATTEMPTS attempts from its own stream. Returns its manifest entry (without the
    person tables, which label_checks adds); the scene is left in out_dir/<scene id> when accepted."""
    seed, index = base.seed, base.index
    key = stream_key(seed, family, index, level)
    rng = np.random.default_rng(key)
    ctx = Context(base)
    scene_id = scene_id_of(seed, family, level, index, base.layout.name)
    folder = Path(out_dir) / scene_id
    entry = {"kind": "variant", "seed": seed, "row": index, "row_spec": list(base.row), "base_scene_id": base.scene.scene_id,
             "family": family, "level": level, "stream": key, "scene_id": scene_id, "attempts": 0, "reasons": [],
             "accepted": False, "rejected_reason": None, "added": [], "placement": None, "draw_fails": [],
             "seconds": 0.0}
    started = time.perf_counter()
    for attempt in range(MAX_ATTEMPTS):
        entry["attempts"] = attempt + 1
        fails: Counter = Counter()
        try:
            added, info = place(rng, ctx, family, level, fails)
        except NoSpot as exc:
            entry["reasons"].append("no_spot")
            entry["draw_fails"].append(dict(exc.fails))
            entry["rejected_reason"] = "no_spot"
            break
        entry["draw_fails"].append(dict(fails))
        layout = variant_layout(base.layout, added)
        scene, rgb, depth = G.build_scene(layout, scene_id)
        if folder.exists():
            shutil.rmtree(folder)
        save_scene(folder, scene, rgb, depth)
        reason = G._rejection(folder, cfg)
        if reason is None:
            scene_r = G._with_regions(folder, scene, cfg)
            save_scene(folder, scene_r, rgb, depth)
            entry["reasons"].append("accepted")
            entry["accepted"] = True
            entry["placement"] = info
            entry["added"] = [{"id": p.id, "x": p.x, "y": p.y, "yaw": p.yaw, "group": p.group, "role": p.role,
                               "shirt": list(f.color)} for p, f in added]
            everyone = layout.scene_gt.people
            entry["added_geometry"] = [{
                "id": p.id,
                "nearest_member_m": min(math.hypot(p.x - m.x, p.y - m.y) for m in ctx.members),
                "nearest_other_person_m": min(math.hypot(p.x - o.x, p.y - o.y) for o in everyone if o.id != p.id),
                "robot_m": math.hypot(p.x - ctx.robot.x, p.y - ctx.robot.y),
                "off_heading_deg": math.degrees(wrap_angle(math.atan2(p.y - ctx.cam_xy[1], p.x - ctx.cam_xy[0])
                                                           - ctx.robot.yaw)),
                "rep_point_m": math.hypot(p.x - ctx.rep_point[0], p.y - ctx.rep_point[1])} for p, _ in added]
            entry["sha256"] = folder_hashes(folder)
            entry["folder"] = str(folder.relative_to(I06))
            share_for = None
            if family == "E":                      # E: every relevant person's pixel share, and the added person's
                inputs, truth = load_scene(folder)
                ets = prepare(inputs, truth, cfg)
                share_for = {*ets["e0"].relevant_people, *ets["e1"].relevant_people, *(p.id for p, _ in added)}
            entry["persons"] = person_table(folder, layout, cfg, share_for=share_for)
            break
        entry["reasons"].append(reason)
        entry["last_attempt_added"] = [{"id": p.id, "x": p.x, "y": p.y, "yaw": p.yaw, "group": p.group}
                                       for p, _ in added]
        shutil.rmtree(folder)
    else:
        entry["rejected_reason"] = "scene_check:" + ",".join(entry["reasons"])
    entry["seconds"] = time.perf_counter() - started
    return entry


# ------------------------------------------------------------------------------------------------- person tables

def visible_samples(observation, person: Person, th) -> int:
    return int(sum(_sample_seen(observation, (person.x, person.y, z), th) for z in _VISIBILITY_HEIGHTS_M))


def person_table(folder: Path, layout: R.Layout, cfg: Config, share_for=None) -> dict:
    """Per person of the scene in `folder`: relevant (e0 and e1, as sgi.gt.derive finds them), samples seen of 9,
    group, role; the pixel share shown (generate._share_shown on that person's figure) for the ids in share_for (a
    set) and for everyone under 9 of 9."""
    inputs, truth = load_scene(folder)
    th, obs = cfg.thresholds, inputs.observation
    ets = prepare(inputs, truth, cfg)
    figures = {person.id: figure for person, figure in zip(layout.scene_gt.people, layout.figures)}
    rows = {}
    for person in truth.people:
        seen = visible_samples(obs, person, th)
        row = {"group": person.group, "role": person.role, "x": person.x, "y": person.y, "yaw": person.yaw,
               "relevant_e0": person.id in ets["e0"].relevant_people,
               "relevant_e1": person.id in ets["e1"].relevant_people, "seen_of_9": seen}
        if seen < 9 or (share_for is not None and person.id in share_for):
            row["pixel_share"] = G._share_shown(obs.camera, obs.depth, [figures[person.id]])
        rows[person.id] = row
    return rows
