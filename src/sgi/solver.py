"""The solver of the constraint method: the floor from depth, and on it the place the constraints leave.

Design: the project's design document for the method, sections 6 and 7; docs/constraint_method.md describes it. The solver is
pure: it calls no model, reads no file and keeps nothing between calls. Everything is worked out in the robot's
frame, on a grid that lies in it, so the same picture gives the same place wherever the world frame lies. A cell
stands for its centre. "Within d" means a distance of d or less.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from sgi.config import Thresholds
from sgi.geometry import (Camera, RobotPose, columns_end_on_floor, lift_all, make_poses, surface_normal_xy, to_robot_xy,
                          to_world_xy)
from sgi.inputs import ObservationData
from sgi.types import (Candidate, Constraint, ConstraintUse, DetectedPerson, Flags, GoalResult, GroundedTarget, Keys,
                       Pose, SolverNote)

_EPS = 1e-9                                        # float-noise tolerance (6 * 0.05 is 0.30000000000000004); not a threshold
_TINY = 1e-12                                      # keeps a division from zero, and tells three points on one line
_MM = 3                                            # decimals of a metre: every value that is compared is rounded to them


@dataclass(frozen=True)
class Floor:
    """The floor of one observation, on a square grid in the robot's frame (the design, section 7, "Floor").

    The robot's position is the centre of the middle cell. Every mask is indexed [row, column]: rows run to the
    robot's left and columns along its heading, both in the order of `offsets`.
    """
    robot: RobotPose                               # the frame the grid lies in
    offsets: np.ndarray                            # (n,) where the cell centres lie along either axis, metres from the robot
    seen: np.ndarray                               # (n, n) bool: the picture shows the floor at the cell
    hidden: np.ndarray                             # something nearer hides the floor there, or the depth is not valid
    strip: np.ndarray                              # under the picture, between the robot and the nearest floor in view
    blocked: np.ndarray                            # too near something that stands over the floor, or a person
    own: np.ndarray                                # where the robot stands
    passable: np.ndarray                           # the robot's centre may cross the cell
    standable: np.ndarray                          # and may stop on it
    selectable: np.ndarray                         # standable, and joined to the robot through passable cells

    def centres(self) -> np.ndarray:
        """The cell centres in the world frame, (n, n, 2)."""
        ahead, left = np.meshgrid(self.offsets, self.offsets)
        c, s = math.cos(self.robot.yaw), math.sin(self.robot.yaw)
        return np.stack([self.robot.x + c * ahead - s * left, self.robot.y + s * ahead + c * left], axis=-1)

    def index(self, x: float, y: float) -> tuple[int, int] | None:
        """(row, column) of the cell that holds a world position, or None off the grid."""
        ahead, left = to_robot_xy(self.robot, x, y)
        step, half = self.offsets[1] - self.offsets[0], len(self.offsets) // 2
        row, col = math.floor(left / step + 0.5) + half, math.floor(ahead / step + 0.5) + half
        if 0 <= row < len(self.offsets) and 0 <= col < len(self.offsets):
            return row, col
        return None


@dataclass(frozen=True)
class Solved:
    """The place the solver chose, and what it made of the answer."""
    pose_world: Pose
    pose_robot: Pose
    uses: list[ConstraintUse]                      # one for each constraint of the answer, in its order
    candidates: list[Candidate]                    # the runners-up, the best first
    note: SolverNote
    flags: Flags                                   # the flags the solver sets; the rest are left as they are

    @property
    def heading(self) -> float:
        """Where the robot faces at the place, in the world frame."""
        return self.pose_world[2]


# --------------------------------------------------------------------------------------------------------- the floor
def _in_robot_frame(camera: Camera, robot: RobotPose) -> Camera:
    """The camera with its pose given in the robot's frame. Heights are the same in both frames."""
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    robot_from_world = np.array([[c, s, 0.0, -(c * robot.x + s * robot.y)], [-s, c, 0.0, s * robot.x - c * robot.y],
                                 [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    return replace(camera, T_world_camera=robot_from_world @ camera.T_world_camera)


def _project_ground(camera: Camera, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u, v, camera Z) of the ground points (x, y, 0), in the frame the camera's pose is given in. A point
    that is not before the camera has a Z of zero or less, and its u and v mean nothing."""
    to_camera = np.linalg.inv(camera.T_world_camera)
    xc, yc, zc = (to_camera[k, 0] * x + to_camera[k, 1] * y + to_camera[k, 3] for k in range(3))
    with np.errstate(divide="ignore", invalid="ignore"):
        return camera.fx * xc / zc + camera.cx, camera.fy * yc / zc + camera.cy, zc


def _ground_point(camera: Camera, u: float, v: float) -> np.ndarray | None:
    """Where the ray through a continuous pixel meets the ground plane; None when it does not come down."""
    ray = camera.T_world_camera[:3, :3] @ np.array([(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, 1.0])
    if ray[2] >= -_EPS:
        return None
    return (camera.T_world_camera[:3, 3] - camera.T_world_camera[2, 3] / ray[2] * ray)[:2]


def _in_triangle(x: np.ndarray, y: np.ndarray, a, b, c) -> np.ndarray:
    """Whether each point lies in the triangle a, b, c, its edges included."""
    def side(p, q):
        return (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0])
    ab, bc, ca = side(a, b), side(b, c), side(c, a)
    return ((ab >= 0) & (bc >= 0) & (ca >= 0)) | ((ab <= 0) & (bc <= 0) & (ca <= 0))


def _cells_within(mask: np.ndarray, radius_m: float, resolution: float) -> np.ndarray:
    """The cells whose centre is within `radius_m` of the centre of a cell of the mask."""
    reach = math.ceil(radius_m / resolution)
    padded = np.pad(mask, reach)
    rows, cols = mask.shape
    near = np.zeros_like(mask)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if math.hypot(dx, dy) * resolution <= radius_m + _EPS:
                near |= padded[reach + dy:reach + dy + rows, reach + dx:reach + dx + cols]
    return near


def _joined(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """The passable cells joined to `start` through cells that share an edge; none when `start` is not passable."""
    joined = np.zeros_like(passable)
    if not passable[start]:
        return joined
    joined[start] = True
    height, width = passable.shape
    front = np.array([start[0] * width + start[1]])            # the cells reached last, as flat indices
    while front.size:
        rows, cols = np.divmod(front, width)
        rows = np.concatenate([rows - 1, rows + 1, rows, rows])
        cols = np.concatenate([cols, cols, cols - 1, cols + 1])
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        rows, cols = rows[inside], cols[inside]
        new = passable[rows, cols] & ~joined[rows, cols]
        joined[rows[new], cols[new]] = True
        front = np.unique(rows[new] * width + cols[new])
    return joined


def floor(observation: ObservationData, people: Sequence[DetectedPerson], th: Thresholds) -> Floor:
    """The floor the picture shows, and where on it the robot may go and stand (the design, section 7, "Floor").

    A cell is seen when its centre projects inside the picture and the depth at that pixel is the centre's own, to
    within `seen_depth_ratio` of it plus `seen_depth_margin_m`. It is hidden when it projects inside and the depth
    is nearer than that, or not valid; where the depth is farther there is no ground at the cell. It is strip when
    its centre lies in the triangle between the robot and the ground points of the picture's two bottom corners,
    under the picture's lower edge, and the image column it projects to ends on floor: the rule the standard goal
    has for a position under the picture (`sgi.geometry.floor_in_view`). It is blocked within
    `standard_goal_clearance_m` of a cell that holds a depth point between `floor_tolerance_m` and the camera's
    height, and within `person_radius_m + robot_radius_m` of a person's centre. It is own within `robot_radius_m`
    of the robot. Seen is tested on depth, not on the height of the lifted point: a ray that grazes the floor
    meets the foot of a wall inside the floor tolerance a good way before the cell it was sent to.

    Hidden floor counts for getting somewhere, not for standing, and the robot's own cell is always a place.
    """
    resolution = th.grid_resolution_m
    half = math.floor(th.floor_range_m / resolution + _EPS)
    offsets = np.arange(-half, half + 1) * resolution
    x, y = np.meshgrid(offsets, offsets)           # the cell centres in the robot's frame: x ahead, y to its left
    from_robot = np.hypot(x, y)
    camera = _in_robot_frame(observation.camera, observation.robot)
    u, v, z = _project_ground(camera, x, y)
    in_front = (z > 0) & (from_robot <= th.floor_range_m + _EPS)
    between = in_front & (u >= 0) & (u < camera.width)         # between the picture's side edges
    in_picture = between & (v >= 0) & (v < camera.height)
    shown = observation.depth[np.where(in_picture, v, 0).astype(int), np.where(in_picture, u, 0).astype(int)]
    shown = shown.astype(np.float64)
    valid = np.isfinite(shown) & (shown > 0)
    slack = th.seen_depth_ratio * z + th.seen_depth_margin_m
    seen = in_picture & valid & (np.abs(shown - z) <= slack)
    hidden = in_picture & ~(valid & (shown >= z - slack))

    strip = np.zeros_like(seen)
    corners = [_ground_point(camera, edge, camera.height) for edge in (0.0, camera.width)]
    if corners[0] is not None and corners[1] is not None:
        under = between & (v >= camera.height) & _in_triangle(x, y, (0.0, 0.0), *corners)
        feet = columns_end_on_floor(observation.depth, camera, np.arange(camera.width), th.floor_tolerance_m,
                                    th.lift_window)
        strip = under & feet[np.where(under, u, 0).astype(int)]

    points = lift_all(observation.depth, camera)   # in the robot's frame
    standing = points[(points[:, 2] > th.floor_tolerance_m) & (points[:, 2] < camera.T_world_camera[2, 3])]
    cols = np.floor(standing[:, 0] / resolution + 0.5).astype(int) + half
    rows = np.floor(standing[:, 1] / resolution + 0.5).astype(int) + half
    on_grid = (rows >= 0) & (rows <= 2 * half) & (cols >= 0) & (cols <= 2 * half)
    occupied = np.zeros_like(seen)
    occupied[rows[on_grid], cols[on_grid]] = True
    blocked = _cells_within(occupied, th.standard_goal_clearance_m, resolution)
    for person in people:
        ahead, left = to_robot_xy(observation.robot, *person.xy)
        blocked |= np.hypot(x - ahead, y - left) <= th.person_radius_m + th.robot_radius_m + _EPS

    own = from_robot <= th.robot_radius_m + _EPS
    passable = own | ((seen | hidden | strip) & ~blocked)
    standable = own | ((seen | strip) & ~blocked)
    selectable = standable & _joined(passable, (half, half))
    return Floor(robot=observation.robot, offsets=offsets, seen=seen, hidden=hidden, strip=strip, blocked=blocked,
                 own=own, passable=passable, standable=standable, selectable=selectable)


# ------------------------------------------------------------------------------------------ people and what is asked
@dataclass(frozen=True)
class _Body:
    """A detected person in the frame the solver works in."""
    number: int
    xy: np.ndarray
    yaw: float | None


@dataclass(frozen=True)
class _Reading:
    """One constraint of the answer as the solver reads it (the design, section 6)."""
    status: str                                    # used, unusable, superseded or cut
    reason: str | None = None                      # why it is not used
    kind: str | None = None                        # of one that is used, and what it stands on:
    people: tuple[int, ...] = ()                   # the numbers that count
    target: bool = False
    range_m: tuple[float, float] | None = None     # at least, at most


def _turned(angle) -> np.ndarray:
    """An angle, or an array of them, wrapped into [-pi, pi)."""
    return (np.asarray(angle) + math.pi) % (2 * math.pi) - math.pi


def _bodies(people: Sequence[DetectedPerson], robot: RobotPose) -> list[_Body]:
    return [_Body(p.number, np.array(to_robot_xy(robot, *p.xy)), None if p.yaw is None else p.yaw - robot.yaw)
            for p in people]


def _faces(xy, yaw: float | None, other_xy, half_angle: float) -> bool:
    """Whether someone at `xy` who faces along `yaw` faces `other_xy`: its bearing from them is within
    `half_angle` of their heading (the design, section 7, "Faces"). Without a heading they face nothing."""
    if yaw is None:
        return False
    bearing = math.atan2(other_xy[1] - xy[1], other_xy[0] - xy[0])
    return abs(float(_turned(bearing - yaw))) <= half_angle + _EPS


def _target_number(bodies: Sequence[_Body], target_xy, camera_xy, th: Thresholds) -> int | None:
    """The number of the person who is the target, if one is: their centre is within `person_radius_m` of the
    target's ground point moved `person_radius_m` farther along the ground line from the camera through it, as
    a body point is moved. The nearest of several, and of two as near the lower number."""
    target_xy, camera_xy = np.asarray(target_xy, dtype=np.float64), np.asarray(camera_xy, dtype=np.float64)
    away = float(np.hypot(*(target_xy - camera_xy)))
    centre = target_xy + th.person_radius_m * (target_xy - camera_xy) / away if away > 0 else target_xy
    near = [(float(np.hypot(*(body.xy - centre))), body.number) for body in bodies]
    near = [(gap, number) for gap, number in near if gap <= th.person_radius_m + _EPS]
    return min(near)[1] if near else None


def target_person(observation: ObservationData, target: GroundedTarget, people: Sequence[DetectedPerson],
                  th: Thresholds) -> int | None:
    """The number of the detected person who is the target, or None when the target is no person (the design,
    section 5): the person whose centre is within `person_radius_m` of the target's ground point moved
    `person_radius_m` along its viewing ray, the nearest when several are."""
    robot = observation.robot
    camera_xy = to_robot_xy(robot, *observation.camera.T_world_camera[:2, 3])
    return _target_number(_bodies(people, robot), to_robot_xy(robot, *target.world[:2]), camera_xy, th)


def _read(constraints: Sequence[Constraint], bodies: dict[int, _Body], target_number: int | None,
          th: Thresholds) -> list[_Reading]:
    """What each constraint of the answer stands on, or why it is not used (the design, section 6)."""
    readings, faced = [], False
    for index, constraint in enumerate(constraints):
        if index >= th.max_constraints:
            readings.append(_Reading("cut", f"beyond the first {th.max_constraints}"))
            continue
        named = list(dict.fromkeys(constraint.people))         # a repeated number counts once
        listed = tuple(number for number in named if number in bodies)
        readings.append(_usable(constraint, named, listed, faced, bodies, target_number, th))
        faced = faced or (constraint.kind == "face" and readings[-1].status == "used")
    return readings


def _usable(constraint: Constraint, named: list[int], listed: tuple[int, ...], faced: bool, bodies: dict[int, _Body],
            target_number: int | None, th: Thresholds) -> _Reading:
    """One constraint within the first `max_constraints`. `listed` holds its numbers that are in the list of people."""
    kind = constraint.kind
    if named and not listed:                       # it was about people the solver does not have
        return _Reading("unusable", "none of its numbers is in the list of people")
    if kind == "keep_out_of_shared_space":
        if len(listed) < 2:
            return _Reading("unusable", "fewer than two people are listed")
        return _Reading("used", kind=kind, people=listed)
    if kind == "keep_out_of_view":                 # towards the target the target person gives no direction,
        giving = tuple(n for n in listed           # and straight ahead a person without a heading gives none
                       if (n != target_number if constraint.target else bodies[n].yaw is not None))
        if not giving:
            return _Reading("unusable", "no listed person gives a direction")
        return _Reading("used", kind=kind, people=giving, target=constraint.target)
    if kind == "come_after":
        waiting = tuple(n for n in listed if n != target_number)
        if not waiting:
            return _Reading("unusable", "nobody is listed but the target person")
        return _Reading("used", kind=kind, people=waiting)
    if kind == "within_distance":
        given = constraint.range_m
        if given is None or not all(math.isfinite(d) and d >= 0 for d in (given.at_least, given.at_most)):
            return _Reading("unusable", "no range of two distances, neither of them negative")
        range_m = (min(given.at_least, given.at_most), max(given.at_least, given.at_most))
        if range_m[0] >= th.max_keep_m - _EPS:
            return _Reading("unusable", "range too far")
        if not listed and not constraint.target:
            return _Reading("unusable", "neither a person nor the target")
        return _Reading("used", kind=kind, people=listed, target=not listed, range_m=range_m)
    if kind == "visible_to":
        seeing = tuple(n for n in listed if bodies[n].yaw is not None)
        if not seeing:
            return _Reading("unusable", "no listed person has a heading")
        return _Reading("used", kind=kind, people=seeing)
    if faced:                                      # the kind is face, and the first usable one stands
        return _Reading("superseded", "an earlier face stands")
    if not listed and not constraint.target:
        return _Reading("unusable", "neither a person nor the target")
    return _Reading("used", kind=kind, people=listed, target=not listed)


# ----------------------------------------------------------------------------------- distances and zones over the cells
def _distance(cells: np.ndarray, xy) -> np.ndarray:
    return np.hypot(cells[:, 0] - xy[0], cells[:, 1] - xy[1])


def _to_segment(cells: np.ndarray, a, b) -> np.ndarray:
    """The distance of each cell to the segment from a to b."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    along = b - a
    length2 = float(along @ along)
    if length2 == 0.0:
        return _distance(cells, a)
    t = np.clip(((cells[:, 0] - a[0]) * along[0] + (cells[:, 1] - a[1]) * along[1]) / length2, 0.0, 1.0)
    return np.hypot(cells[:, 0] - (a[0] + t * along[0]), cells[:, 1] - (a[1] + t * along[1]))


def _from_sight_lines(xy, eye, cells: np.ndarray) -> np.ndarray:
    """The distance from the point `xy` to each segment that runs from `eye` to a cell."""
    dx, dy = cells[:, 0] - eye[0], cells[:, 1] - eye[1]
    t = np.clip(((xy[0] - eye[0]) * dx + (xy[1] - eye[1]) * dy) / np.maximum(dx ** 2 + dy ** 2, _TINY), 0.0, 1.0)
    return np.hypot(eye[0] + t * dx - xy[0], eye[1] + t * dy - xy[1])


def _hull(points) -> list[tuple[float, float]]:
    """The corners of the convex hull of some points, anticlockwise; a point on a side is no corner."""
    points = sorted({(float(x), float(y)) for x, y in points})
    if len(points) <= 2:
        return points

    def turn(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def chain(ordered):
        kept: list[tuple[float, float]] = []
        for point in ordered:
            while len(kept) >= 2 and turn(kept[-2], kept[-1], point) <= _TINY:
                kept.pop()
            kept.append(point)
        return kept[:-1]

    return chain(points) + chain(reversed(points))


def _hull_signed(cells: np.ndarray, points) -> np.ndarray:
    """The signed distance of each cell to the convex hull of the points: negative inside, positive outside.
    Points on one line have a hull without an inside."""
    corners = _hull(points)
    if len(corners) == 1:
        return _distance(cells, corners[0])
    if len(corners) == 2:
        return _to_segment(cells, *corners)
    inside = np.ones(len(cells), dtype=bool)
    nearest = np.full(len(cells), np.inf)
    for a, b in zip(corners, corners[1:] + corners[:1]):
        inside &= (b[0] - a[0]) * (cells[:, 1] - a[1]) - (b[1] - a[1]) * (cells[:, 0] - a[0]) >= 0
        nearest = np.minimum(nearest, _to_segment(cells, a, b))
    return np.where(inside, -nearest, nearest)


def _rhombus_signed(cells: np.ndarray, a, b, half_angle: float) -> np.ndarray:
    """The signed distance to the rhombus with `a` and `b` as opposite corners whose sides leave each of the two
    at `half_angle` from the line between them: what each of two people sees of the other."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    along = b - a
    across = 0.5 * math.tan(half_angle) * np.array([-along[1], along[0]])
    middle = (a + b) / 2
    return _hull_signed(cells, [a, middle + across, b, middle - across])


def _shared_space_signed(cells: np.ndarray, points, half_angle: float) -> np.ndarray:
    """The signed distance to the space between people who attend to one another: the convex hull of their
    centres together with the rhombus of each side of that hull, the least of their signed distances. People
    on one line have a hull of one side, and their space is that side's rhombus."""
    corners = _hull(points)
    if len(corners) == 1:
        return _distance(cells, corners[0])
    if len(corners) == 2:
        return _rhombus_signed(cells, *corners, half_angle)
    signed = _hull_signed(cells, corners)
    for a, b in zip(corners, corners[1:] + corners[:1]):
        signed = np.minimum(signed, _rhombus_signed(cells, a, b, half_angle))
    return signed


def _sector_signed(cells: np.ndarray, apex, axis: float, half_angle: float, reach: float = math.inf) -> np.ndarray:
    """The signed distance to the sector with its point at `apex`, `half_angle` (under a right angle) either side
    of the direction `axis`, out to `reach` from the apex: negative inside, positive outside."""
    dx, dy = cells[:, 0] - apex[0], cells[:, 1] - apex[1]
    c, s = math.cos(axis), math.sin(axis)
    along, across = c * dx + s * dy, np.abs(c * dy - s * dx)   # the sector is the same either side of its axis
    away = np.hypot(along, across)
    edge = (math.cos(half_angle), math.sin(half_angle))        # the direction of the sector's side
    on_edge = along * edge[0] + across * edge[1]               # how far along that side a cell lies,
    off_edge = np.abs(along * edge[1] - across * edge[0])      # and how far from its line
    in_angle = np.arctan2(across, along) <= half_angle
    if math.isfinite(reach):
        inner = np.minimum(off_edge, reach - away)
        corner = np.hypot(along - reach * edge[0], across - reach * edge[1])     # from the end of the side
    else:
        inner, corner = off_edge, np.full(len(cells), np.inf)
    beside = np.where(on_edge <= 0, away, np.where(on_edge <= reach, off_edge, corner))
    return np.where(in_angle & (away <= reach), -inner, np.where(in_angle, away - reach, beside))


def _view_signed(cells: np.ndarray, reading: _Reading, bodies: dict[int, _Body], target_xy: np.ndarray,
                 half_view: float) -> np.ndarray:
    """The signed distance to the nearest zone of a `keep_out_of_view`: for each of its people the sector with
    its point at their centre, `half_view` either side of an axis. With `target` the axis is the direction to the
    target point and the sector reaches as far as it. Otherwise the axis is their heading, and the sector
    reaches as far as the nearest other person of the same constraint who stands inside it and faces them back;
    it has no far end when there is none."""
    nearest = np.full(len(cells), np.inf)
    for number in reading.people:
        body = bodies[number]
        if reading.target:
            to_target = target_xy - body.xy
            axis, reach = math.atan2(to_target[1], to_target[0]), float(np.hypot(*to_target))
        else:
            axis, reach = body.yaw, math.inf
            for other in (bodies[n] for n in reading.people if n != number):
                if _faces(body.xy, body.yaw, other.xy, half_view) and _faces(other.xy, other.yaw, body.xy, half_view):
                    reach = min(reach, float(np.hypot(*(other.xy - body.xy))))
        nearest = np.minimum(nearest, _sector_signed(cells, body.xy, axis, half_view, reach))
    return nearest


def _unseen(cells: np.ndarray, reading: _Reading, everyone: Sequence[_Body], bodies: dict[int, _Body],
            half_visible: float, th: Thresholds) -> np.ndarray:
    """The violation of a `visible_to`: for each of its people, `standoff_m` times the angle by which the bearing
    from them to the cell is more than `half_visible` off their heading, so that closing in from behind does not
    shrink it; plus, for every other detected person, `person_radius_m` less the distance from that person's
    centre to the segment between the two, where positive: a body in the line of sight."""
    violation = np.zeros(len(cells))
    for number in reading.people:
        body = bodies[number]
        off = np.abs(_turned(np.arctan2(cells[:, 1] - body.xy[1], cells[:, 0] - body.xy[0]) - body.yaw))
        violation += th.standoff_m * np.maximum(0.0, off - half_visible)
        for other in everyone:
            if other.number != number:
                violation += np.maximum(0.0, th.person_radius_m - _from_sight_lines(other.xy, body.xy, cells))
    return violation


# ---------------------------------------------------------------------------------------------------- the place in turn
def _line_end(waiting: Sequence[int], bodies: dict[int, _Body], target_xy: np.ndarray, robot_xy: np.ndarray,
              th: Thresholds) -> tuple[int, int | None]:
    """(the last of the waiting people, the one before them or None), from positions alone.

    The last is the farthest from the target point. Those within `person_radius_m` of that distance count as
    equally far, and of these the nearest the robot is taken. The one before is the waiting person nearest to
    the last, when that person is nearer to the target point than the last by more than `person_radius_m`:
    otherwise their nearest neighbour stands beside them. Of two equally near, to the millimetre, the lower
    number is taken.
    """
    from_target = {n: float(np.hypot(*(bodies[n].xy - target_xy))) for n in waiting}
    farthest = max(from_target.values())
    equally_far = [n for n in waiting if from_target[n] >= farthest - th.person_radius_m]
    last = min(equally_far, key=lambda n: (round(float(np.hypot(*(bodies[n].xy - robot_xy))), _MM), n))
    others = [n for n in waiting if n != last]
    if not others:
        return last, None
    neighbour = min(others, key=lambda n: (round(float(np.hypot(*(bodies[n].xy - bodies[last].xy))), _MM), n))
    return last, neighbour if from_target[neighbour] < from_target[last] - th.person_radius_m else None


def _behind(last: _Body, before: _Body | None, target_xy: np.ndarray, robot_xy: np.ndarray,
            th: Thresholds) -> tuple[np.ndarray, np.ndarray]:
    """(the place in turn, the direction of the line): `standoff_m` beyond the last on the line from the one
    before through the last. The line starts from the target point when nobody is before, and runs from the last
    towards the robot when the target point too is within `heading_degenerate_m` of the last."""
    for start in ((before.xy,) if before is not None else ()) + (target_xy,):
        length = float(np.hypot(*(last.xy - start)))
        if length >= th.heading_degenerate_m:
            along = (last.xy - start) / length
            return last.xy + th.standoff_m * along, along
    to_robot = robot_xy - last.xy
    along = to_robot / max(float(np.hypot(*to_robot)), _EPS)
    return last.xy + th.standoff_m * along, along


@dataclass(frozen=True)
class _Turn:
    """Where the robot's turn is, behind the people who wait (the design, section 7, "The place in turn")."""
    last: int                                      # the number of the last of them
    last_from_target: float                        # how far the last stands from the target point
    place: np.ndarray                              # the place in turn
    joined: list[int]                              # those taken as waiting whom the answer did not list, as they joined


def _turn(waiting: Sequence[int], apart: set[int], bodies: dict[int, _Body], target_number: int | None,
          target_xy: np.ndarray, robot_xy: np.ndarray, th: Thresholds) -> _Turn:
    """The place in turn for the waiting people of the answer, and who has taken it.

    A detected person who is neither the target, nor waiting, nor `apart` (listed among people who attend to one
    another) has taken the place in turn when their own space covers it and they stand behind the last: the
    direction from the last to them is within `view_half_angle_deg` of the line's. The nearest of those to the
    place, the lower number of two equally near, joins as the new last, the one who was last is the one before
    them, and the place is worked out again from the two; and so on until nobody has taken it.
    """
    waiting, joined = list(waiting), []
    last, before = _line_end(waiting, bodies, target_xy, robot_xy, th)
    while True:
        place, along = _behind(bodies[last], None if before is None else bodies[before], target_xy, robot_xy, th)
        taken = []
        for number, body in bodies.items():
            if number in waiting or number == target_number or number in apart:
                continue
            gap, back = float(np.hypot(*(body.xy - place))), body.xy - bodies[last].xy
            behind = float(back @ along) >= math.cos(math.radians(th.view_half_angle_deg)) * float(np.hypot(*back)) - _EPS
            if gap < th.standoff_m - _EPS and behind:
                taken.append((round(gap, _MM), number))
        if not taken:
            from_target = float(np.hypot(*(bodies[last].xy - target_xy)))
            return _Turn(last=last, last_from_target=from_target, place=place, joined=joined)
        last, before = min(taken)[1], last
        waiting.append(last)
        joined.append(last)


# -------------------------------------------------------------------------------------------------------- the seen side
def _seen_face(observation: ObservationData, target: GroundedTarget, th: Thresholds) -> np.ndarray | None:
    """The front of the face the target point lies on (the design, section 7, "The seen side"): the horizontal unit
    normal, on the camera's side, of the surface round the point, in the robot's frame. None where the depth image
    shows no near-vertical flat surface there. The fit and its thresholds are the standard goal's."""
    normal = surface_normal_xy(observation.depth, observation.camera, target.world, th.surface_radius_m,
                               th.surface_min_points, th.surface_max_tilt, th.surface_max_rms_m,
                               th.surface_min_spread_m, th.surface_max_rms_ratio)
    if normal is None:
        return None
    c, s = math.cos(observation.robot.yaw), math.sin(observation.robot.yaw)
    return np.array([c * normal[0] + s * normal[1], c * normal[1] - s * normal[0]])


def _ahead_of(cells: np.ndarray, least: np.ndarray, watching: Sequence[_Body], target_xy: np.ndarray,
              front: np.ndarray | None, standard_xy: np.ndarray,
              th: Thresholds) -> tuple[list[int], np.ndarray | None, str | None]:
    """(the people who watch the target from the side the robot was sent to, how far each cell is ahead of them,
    None), or ([], None, why) where the robot is not kept back with anyone. `least` marks the cells that break
    the other exclusions least.

    That side is before the seen face, across `front`; of a target without one, the side the standard goal stands
    on. Ahead is measured along the direction they watch in, from the middle of their centres to the target point:
    a cell is ahead by how much farther along it the cell lies than the hindmost of them, so everything beyond
    the target is ahead.

    Nobody is stood back with when nobody watches from that side; when the side or the direction cannot be told
    (the standard goal, or their middle, is nearer the target point than `heading_degenerate_m`); when none of the
    cells is level with or behind the hindmost of them, to the millimetre; and when none of the cells of `least`
    is. The rule is then not applied at all. In the first of these two it cannot be met, and in the second only by
    a cell that breaks another exclusion more than some cell ahead of them does. Applied there, how far ahead a
    place is was traded against the other exclusions, and the robot stood inside a person's own space on account
    of this rule."""
    side = front if front is not None else standard_xy - target_xy
    if float(np.hypot(*side)) < th.heading_degenerate_m:
        return [], None, "the target has no seen face and the standard goal stands on the target point"
    joined = [body for body in watching if float((body.xy - target_xy) @ side) > 0.0]
    if not joined:
        return [], None, "none of them stands on the side the robot was sent to"
    xy = np.array([body.xy for body in joined])
    to_target = target_xy - xy.mean(axis=0)
    gap = float(np.hypot(*to_target))
    if gap < th.heading_degenerate_m:
        return [], None, "the middle of them is on the target point"
    along = to_target / gap
    before = (target_xy - xy) @ along               # how far before the target each of them stands
    of_cells = (target_xy[0] - cells[:, 0]) * along[0] + (target_xy[1] - cells[:, 1]) * along[1]
    ahead = np.maximum(0.0, float(before.max()) - of_cells)
    level, hindmost = np.round(ahead, _MM) == 0.0, joined[int(before.argmax())].number
    if not level.any():                             # no cell meets it: the hindmost stands where no floor is in reach
        return [], None, f"no cell is level with or behind person {hindmost}"
    if not (level & least).any():                   # it can be met only by breaking another exclusion more
        return [], None, (f"every cell level with or behind person {hindmost} breaks the other exclusions more than "
                          "some cell ahead of them")
    return [body.number for body in joined], ahead, None


# ------------------------------------------------------------------------------------------------------------ the order
@dataclass(frozen=True)
class _Settled:
    """What the solver worked out over the cells it was given, in the frame they are in."""
    picks: list[int]                               # the cell of the place, then of each runner-up
    keys: dict[str, np.ndarray]                    # the values the cells are ordered by, each rounded to the millimetre
    readings: list[_Reading]
    violations: list[np.ndarray | None]            # for each constraint of the answer, its violation at every cell
    own_space: dict[int, np.ndarray]               # for each person, by number, the violation of their own space
    faced_xy: np.ndarray                           # what the robot faces
    target_xy: np.ndarray                          # the target point
    turn: _Turn | None                             # None without a come_after
    not_ahead_of: list[int]                        # the people who watch the target whom the place is not ahead of
    not_ahead_reason: str | None                   # with a view of the target used and nobody among them: why


_BANDED = ("turn_m", "preference_m")               # the keys at which the cells within one cell of the least stay in


def _first(keys: dict[str, np.ndarray], among: np.ndarray, band_m: float) -> int | None:
    """The first of the cells of a mask in the order of the design, section 7: key by key, the cells at the
    least value stay in, and at the banded keys those within `band_m` of it; then the first in the order the
    cells are given in, which is by row and then by column of the grid. None when the mask is empty."""
    if not among.any():
        return None
    left = among.copy()
    for name, values in keys.items():
        least = values[left].min()
        left &= values <= least + (band_m if name in _BANDED else 0.0) + _EPS
    return int(np.argmax(left))


def _settle(cells: np.ndarray, robot_xy, camera_xy, target_xy, bodies: Sequence[_Body],
            constraints: Sequence[Constraint], standard_xy, th: Thresholds, front=None) -> _Settled:
    """The place among `cells`, (N, 2) positions in the order of the grid, and the runners-up (the design,
    section 7). Every position is given in one frame, and so is every heading. `front` is the front of the seen
    face in that frame (`_seen_face`), or None where the target point lies on none. It needs at least one cell: with
    none, a usable view of the target makes it raise (`solve` always has cells, the robot's own)."""
    robot_xy, standard_xy = np.asarray(robot_xy, dtype=np.float64), np.asarray(standard_xy, dtype=np.float64)
    by_number = {body.number: body for body in bodies}
    target_number = _target_number(bodies, target_xy, camera_xy, th)
    target_xy = by_number[target_number].xy if target_number is not None else np.asarray(target_xy, dtype=np.float64)
    readings = _read(constraints, by_number, target_number, th)
    used = [reading for reading in readings if reading.status == "used"]
    half_view, half_visible = math.radians(th.view_half_angle_deg), math.radians(th.visible_half_angle_deg)
    from_target = _distance(cells, target_xy)
    nothing = np.zeros(len(cells))

    # Exclusions: each gives every cell a violation in metres, zero where it holds.
    own_space = {body.number: np.maximum(0.0, th.standoff_m - _distance(cells, body.xy)) for body in bodies}
    violations: list[np.ndarray | None] = [None] * len(readings)
    exclusions = list(own_space.values())
    for index, reading in enumerate(readings):
        if reading.status != "used":
            continue
        if reading.kind == "keep_out_of_shared_space":
            space = _shared_space_signed(cells, [by_number[n].xy for n in reading.people], half_view)
            violations[index] = np.maximum(0.0, th.robot_radius_m - space)
        elif reading.kind == "keep_out_of_view":
            violations[index] = np.maximum(0.0, th.robot_radius_m - _view_signed(cells, reading, by_number, target_xy,
                                                                                 half_view))
        else:
            continue
        exclusions.append(violations[index])
    waiting = sorted({n for reading in used if reading.kind == "come_after" for n in reading.people})
    turn = None
    if waiting:                                    # several come_after are one, about all their people
        apart = {n for reading in used if reading.kind == "keep_out_of_shared_space" for n in reading.people}
        turn = _turn(waiting, apart, by_number, target_number, target_xy, robot_xy, th)
        ahead = np.maximum(0.0, turn.last_from_target - from_target)       # nearer to the target than the last
        exclusions.append(ahead)
        for index, reading in enumerate(readings):
            if reading.status == "used" and reading.kind == "come_after":
                violations[index] = ahead
    towards = [index for index, reading in enumerate(readings)
               if reading.status == "used" and reading.kind == "keep_out_of_view" and reading.target]
    not_ahead_of: list[int] = []
    not_ahead_reason = None
    if towards and turn is not None:
        not_ahead_reason = "a place in turn is used"
    elif towards and target_number is not None:
        not_ahead_reason = "the target is a person"
    elif towards:                                              # several views of the target are one, about all their people
        watching = [by_number[n] for n in sorted({n for index in towards for n in readings[index].people})]
        worst, total = nothing, nothing                        # of the exclusions so far, as keys 1 and 2 have them
        for violation in exclusions:
            worst, total = np.maximum(worst, violation), total + violation
        worst, total = np.round(worst, _MM), np.round(total, _MM)
        least = worst <= worst.min() + _EPS                    # the cells that break them least: the largest violation,
        least &= total <= total[least].min() + _EPS            # then the sum
        not_ahead_of, before_them, not_ahead_reason = _ahead_of(cells, least, watching, target_xy, front, standard_xy,
                                                                th)
        if before_them is not None:
            exclusions.append(before_them)                     # once, however many constraints carry it
            for index in towards:
                violations[index] = np.maximum(violations[index], before_them)

    # Preferences.
    preferences = []
    for index, reading in enumerate(readings):
        if reading.status != "used":
            continue
        if reading.kind == "within_distance":
            at_least, at_most = reading.range_m
            outside = nothing.copy()
            for xy in [by_number[n].xy for n in reading.people] or [target_xy]:
                away = _distance(cells, xy)
                outside += np.maximum(0.0, np.maximum(at_least - away, away - at_most))
            violations[index] = outside
        elif reading.kind == "visible_to":
            violations[index] = _unseen(cells, reading, bodies, by_number, half_visible, th)
        else:
            continue
        preferences.append(violations[index])
    about_target = any(reading.kind == "within_distance" and (reading.target or target_number in reading.people)
                       for reading in used)
    if not about_target:                           # the approach distance, unless the answer gives its own
        preferences.append(np.abs(from_target - th.standoff_m))

    # The keys, in the order they are compared in.
    keys = {"excluded_m": nothing, "excluded_total_m": nothing}
    for violation in exclusions:
        keys["excluded_m"] = np.maximum(keys["excluded_m"], violation)
        keys["excluded_total_m"] = keys["excluded_total_m"] + violation
    # The seen face is not read where the answer says the robot came for a person: it names someone to face, and says
    # the place is to be visible to that same person. The place is then chosen by that person's front, and a face
    # fitted by depth must not rank above it: fitted on the body of a person the list of people has elsewhere, it would
    # hold the robot on the camera's side of them. A `visible_to` alone, or one about somebody the robot is not told
    # to face, leaves the face read: the robot came for the thing, and is not sent behind it to be seen by someone.
    face = next((reading for reading in used if reading.kind == "face"), None)
    faced = set(face.people) if face is not None else set()
    for_a_person = any(reading.kind == "visible_to" and faced & set(reading.people) for reading in used)
    if turn is not None:
        keys["turn_m"] = _distance(cells, turn.place)
    elif front is not None and target_number is None and not for_a_person:  # behind the plane of the face the robot saw
        keys["face_m"] = np.maximum(0.0, (target_xy[0] - cells[:, 0]) * front[0] + (target_xy[1] - cells[:, 1]) * front[1])
    keys["preference_m"] = sum(preferences, start=nothing)
    keys["standard_m"] = _distance(cells, standard_xy)
    keys["robot_m"] = _distance(cells, robot_xy)
    keys = {name: np.round(values, _MM) for name, values in keys.items()}

    picks: list[int] = []
    free = np.ones(len(cells), dtype=bool)
    while len(picks) <= th.candidate_count:        # the place, then each runner-up a robot's width from those before
        pick = _first(keys, free, th.grid_resolution_m)
        if pick is None:
            break
        picks.append(pick)
        free &= _distance(cells, cells[pick]) >= 2 * th.robot_radius_m - _EPS

    if face is not None and face.people:           # the listed person, or the middle of several
        faced_xy = np.mean([by_number[n].xy for n in face.people], axis=0)
    elif face is None and turn is not None:        # without one, the last of the waiting people
        faced_xy = by_number[turn.last].xy
    else:
        faced_xy = target_xy
    return _Settled(picks=picks, keys=keys, readings=readings, violations=violations, own_space=own_space,
                    faced_xy=faced_xy, target_xy=target_xy, turn=turn, not_ahead_of=not_ahead_of,
                    not_ahead_reason=not_ahead_reason)


def _over_floor(observation: ObservationData, target: GroundedTarget, people: Sequence[DetectedPerson],
                constraints: Sequence[Constraint], standard: GoalResult, th: Thresholds) -> tuple[np.ndarray, _Settled]:
    """The selectable cells of the observation's floor, (N, 2) in the robot's frame and in the order of the grid,
    and what the solver works out over them."""
    if standard.pose_world is None:
        raise ValueError("the solver needs the standard goal's pose, and this one has none")
    robot = observation.robot
    grid = floor(observation, people, th)
    ahead, left = np.meshgrid(grid.offsets, grid.offsets)
    cells = np.stack([ahead[grid.selectable], left[grid.selectable]], axis=-1)
    settled = _settle(cells, (0.0, 0.0), to_robot_xy(robot, *observation.camera.T_world_camera[:2, 3]),
                      to_robot_xy(robot, *target.world[:2]), _bodies(people, robot), constraints,
                      to_robot_xy(robot, *standard.pose_world[:2]), th, front=_seen_face(observation, target, th))
    return cells, settled


def solve(observation: ObservationData, target: GroundedTarget, people: Sequence[DetectedPerson],
          constraints: Sequence[Constraint], standard: GoalResult, th: Thresholds) -> Solved:
    """The place the constraints leave on the floor of the observation, the heading there and the runners-up
    (the design, section 7). `standard` is the standard goal for the same observation and target.

    Each exclusion and each preference gives every cell a violation in metres. Every value is rounded to the
    millimetre, and the selectable cells are compared by these keys in turn, the cells at the least value
    staying in: the largest violation of any one exclusion; the sum of the exclusions' violations; the distance
    to the place in turn, when there is a `come_after`; how far behind the seen face of the target, when it has
    one, is no person, no place in turn is used and the answer does not say the robot came for a person; the sum of
    the preferences' violations; the distance to the standard goal; the distance to the robot, then the lowest row
    and the lowest column of the grid. At the place in turn and at the preferences the cells within
    `grid_resolution_m` of the least stay in. So the place breaks no exclusion where some cell breaks none;
    otherwise each one it breaks is recorded with its violation.

    Two rules of the seen side (the design, section 7) are in this: the key for the seen face, and, with a
    `keep_out_of_view` towards the target, an exclusion of every cell ahead of those of its people who stand on
    the side the robot was sent to. Neither is applied where a place in turn is used or the target is a person.
    The key is not read either where the answer says the robot came for a person: a used `face` names someone and
    a usable `visible_to` lists that same person, whose front the place is then chosen by. And the exclusion is
    added only where, among the cells that break the other exclusions least, one is level with those people or
    behind them: with it the place breaks the other exclusions no more than without it, by the largest violation
    and by the sum. Where it is not added the note says why (`not_ahead_reason`).

    The robot faces what the `face` constraint names; without one the last of the waiting people; otherwise the
    target point. A runner-up is chosen in the same way among the cells at least a robot's width from every
    place already taken.
    """
    robot = observation.robot
    cells, settled = _over_floor(observation, target, people, constraints, standard, th)

    def keys_at(cell: int) -> Keys:
        values = {name: float(key[cell]) for name, key in settled.keys.items() if name != "robot_m"}
        return Keys(**values)

    def poses_at(cell: int) -> tuple[Pose, Pose, bool]:
        """The poses of the robot on a cell, and whether what it faces is too close there to give a heading."""
        toward = settled.faced_xy - cells[cell]
        degenerate = float(np.hypot(*toward)) < th.heading_degenerate_m
        theta = robot.yaw if degenerate else robot.yaw + math.atan2(toward[1], toward[0])
        return *make_poses(robot, to_world_xy(robot, *cells[cell]), theta), degenerate

    place = settled.picks[0]
    pose_world, pose_robot, degenerate = poses_at(place)
    statuses = [reading.status for reading in settled.readings]
    uses = [ConstraintUse(constraint=constraint, status=reading.status, reason=reading.reason,
                          broken_m=0.0 if violation is None else round(float(violation[place]), _MM))
            for constraint, reading, violation in zip(constraints, settled.readings, settled.violations)]
    own_space = {number: round(float(violation[place]), _MM) for number, violation in settled.own_space.items()}
    target_distance = float(np.hypot(*(cells[place] - settled.target_xy)))
    note = SolverNote(keys=keys_at(place), own_space_broken_m={n: v for n, v in own_space.items() if v > 0},
                      joined_waiting=[] if settled.turn is None else list(settled.turn.joined),
                      not_ahead_of=list(settled.not_ahead_of), not_ahead_reason=settled.not_ahead_reason,
                      target_distance_m=target_distance)
    flags = Flags(heading_degenerate=degenerate, no_usable_constraint="used" not in statuses,
                  constraint_unusable="unusable" in statuses or "superseded" in statuses,
                  constraints_cut="cut" in statuses, exclusion_broken=note.keys.excluded_m > 0,
                  far_from_target=(target_distance > th.max_keep_m + _EPS and settled.turn is None
                                   and not settled.not_ahead_of))
    candidates = [Candidate(pose_world=poses_at(cell)[0], keys=keys_at(cell)) for cell in settled.picks[1:]]
    return Solved(pose_world=pose_world, pose_robot=pose_robot, uses=uses, candidates=candidates, note=note, flags=flags)
