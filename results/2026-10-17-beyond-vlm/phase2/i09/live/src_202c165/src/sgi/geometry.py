"""Frames and geometry: world, robot and optical camera frames, in metres and radians."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sgi.types import Pose

_PLANE_POINTS = 3                                  # fewer points than this have no plane through them
_LEVEL = 1e-6                                      # a normal with a horizontal part shorter than this has no ground direction


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    T_world_camera: np.ndarray                     # (4, 4) float64, p_world = T @ p_camera


@dataclass(frozen=True)
class RobotPose:
    x: float
    y: float
    yaw: float


def wrap_angle(a: float) -> float:
    """Wrap an angle into (-pi, pi]."""
    return math.pi - ((math.pi - a) % (2 * math.pi))


def point_to_pixel(x: int, y: int, width: int, height: int) -> tuple[float, float]:
    """A normalised 0..1000 point as a continuous pixel inside the image."""
    u = min(max(x / 1000 * width, 0.0), width - 1e-3)
    v = min(max(y / 1000 * height, 0.0), height - 1e-3)
    return u, v


def pixel_to_point(u: float, v: float, width: int, height: int) -> tuple[int, int]:
    """A continuous pixel as the nearest normalised 0..1000 point."""
    x = round(float(u) / width * 1000)
    y = round(float(v) / height * 1000)
    return min(max(x, 0), 1000), min(max(y, 0), 1000)


def in_image(cam: Camera, u: float, v: float) -> bool:
    return bool(0 <= u < cam.width and 0 <= v < cam.height)


def _valid_depth(depth: np.ndarray) -> np.ndarray:
    """Mask of the usable depth values: finite and positive."""
    return np.isfinite(depth) & (depth > 0)


def _pixel_centres(height: int, width: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """u and v at the centre of every stride-th pixel, each of shape (rows, columns)."""
    v, u = np.meshgrid(np.arange(0, height, stride) + 0.5, np.arange(0, width, stride) + 0.5, indexing="ij")
    return u, v


def _depth_at(depth: np.ndarray, cam: Camera, u: float, v: float, window: int) -> float | None:
    """The median valid depth of the window around a continuous pixel; None outside the image or with none valid."""
    if not in_image(cam, u, v):
        return None
    row, col, half = math.floor(v), math.floor(u), window // 2
    patch = depth[max(row - half, 0):row + half + 1, max(col - half, 0):col + half + 1]
    valid = patch[_valid_depth(patch)].astype(np.float64)
    return float(np.median(valid)) if valid.size else None


def lift(depth: np.ndarray, cam: Camera, u: float, v: float, window: int = 5) -> np.ndarray | None:
    """The world point at a continuous pixel, using the median valid depth of the window around it.

    None when the pixel is outside the image or the window holds no valid depth.
    """
    Z = _depth_at(depth, cam, u, v, window)
    if Z is None:
        return None
    p_camera = np.array([(u - cam.cx) * Z / cam.fx, (v - cam.cy) * Z / cam.fy, Z, 1.0])
    return (cam.T_world_camera @ p_camera)[:3]


def lift_all(depth: np.ndarray, cam: Camera, stride: int = 1) -> np.ndarray:
    """World points (N, 3) of every stride-th pixel with valid depth, lifted at its centre, in row-major order."""
    Z = depth[::stride, ::stride].astype(np.float64)
    u, v = _pixel_centres(*depth.shape, stride)
    keep = _valid_depth(Z)
    u, v, Z = u[keep], v[keep], Z[keep]
    p_camera = np.stack([(u - cam.cx) * Z / cam.fx, (v - cam.cy) * Z / cam.fy, Z, np.ones_like(Z)], axis=1)
    return (p_camera @ cam.T_world_camera.T)[:, :3]


def surface_normal_xy(depth: np.ndarray, cam: Camera, p_world, radius_m: float, min_points: int, max_tilt: float,
                      max_rms_m: float, min_spread_m: float, max_rms_ratio: float) -> np.ndarray | None:
    """The horizontal unit normal, pointing to the camera's side, of the surface round a world point.

    A plane is fitted to the world points of the depth pixels within `radius_m` of the point. The spread of the
    points along a direction is their root-mean-square distance from their mean along it. The fit gives three
    directions with their spreads in descending order: two across the plane, and its normal, along which the
    spread is the root-mean-square distance of the points from the plane.

    None when there is no near-vertical flat surface there: fewer than `min_points` points; a second spread
    below `min_spread_m` (points along one line, as an edge or a thin post gives them: no one plane passes
    through a line); a normal whose vertical component is beyond `max_tilt` (a table top or a floor, not a
    face); a least spread beyond `max_rms_m` (a sharp corner, clutter), or beyond `max_rms_ratio` of the second
    spread (a square corner, a body seen in profile: points as thick as they are wide have no least direction
    that holds from one view to the next).
    """
    p = np.asarray(p_world, dtype=np.float64)
    points = lift_all(depth, cam)
    near = points[np.linalg.norm(points - p, axis=1) < radius_m]
    if len(near) < max(min_points, _PLANE_POINTS):
        return None
    _, values, axes = np.linalg.svd(near - near.mean(axis=0), full_matrices=False)
    _, across, through = values / math.sqrt(len(near))         # singular values descend: so do the three spreads
    normal = axes[2]                                           # the axis of least spread
    if across < min_spread_m or abs(normal[2]) > max_tilt or through > max_rms_m or through > max_rms_ratio * across:
        return None
    if np.dot(normal, cam.T_world_camera[:3, 3] - p) < 0:
        normal = -normal                           # a fitted normal has no sign of its own: the side that is seen
    length = float(np.hypot(normal[0], normal[1]))
    if length < _LEVEL:                            # only when max_tilt lets a level surface through
        return None
    return normal[:2] / length


def columns_end_on_floor(depth: np.ndarray, cam: Camera, cols, floor_tolerance_m: float, window: int = 5) -> np.ndarray:
    """Whether each of the image columns `cols` ends on floor: each of its lowest `window` rows (at least one)
    holds valid depth that lifts, at that pixel's centre, to within `floor_tolerance_m` of the ground plane.

    This is the floor a position under the picture's lower edge is given by (`floor_in_view`): one rule for the
    standard goal and for the strip of the solver's floor. Returns a bool array, one entry per column.
    """
    cols = np.asarray(cols, dtype=int)
    rows = np.arange(max(cam.height - max(window, 1), 0), cam.height)
    Z = depth[np.ix_(rows, cols)].astype(np.float64)           # (rows, columns)
    valid = _valid_depth(Z)
    Z = np.where(valid, Z, 0.0)
    u, v = np.broadcast_arrays(cols + 0.5, rows[:, None] + 0.5)
    p_camera = np.stack([(u - cam.cx) * Z / cam.fx, (v - cam.cy) * Z / cam.fy, Z, np.ones_like(Z)], axis=-1)
    height = p_camera @ cam.T_world_camera[2]      # the third row of the transform gives a point's world z
    return (valid & (np.abs(height) <= floor_tolerance_m)).all(axis=0)


def floor_in_view(depth: np.ndarray, cam: Camera, xy, floor_tolerance_m: float, depth_tolerance_m: float,
                  window: int = 5) -> bool:
    """Whether the depth image shows the floor at a ground position.

    The position projects inside the image, and the depth at that pixel, taken as `lift` takes it, is that of
    the floor there: within `depth_tolerance_m` of the position's own camera Z, and of a point within
    `floor_tolerance_m` of the ground plane. Either test alone lets something through. A ray that grazes the
    floor meets the foot of whatever hides the position at a height inside the floor tolerance; a ray that
    comes down steeply meets something low that stands on it at a depth inside the other.

    A position that projects under the picture's lower edge, between its side edges, is nearer the camera than
    any floor the picture shows. It counts as in view when the image column it projects to ends on floor: the
    lowest `window` rows of that column all hold valid depth that lifts to within `floor_tolerance_m` of the
    ground plane. The unseen strip from that floor back to the camera is then taken to be floor as well. What
    is lower than the picture's lowest ray and lies wholly under the picture is not seen, and is passed.

    False where the pixel has no valid depth, where the position projects above, left or right of the picture,
    and where it is behind the camera.
    """
    projected = project(cam, (float(xy[0]), float(xy[1]), 0.0))
    if projected is None:
        return False
    u, v, z = projected
    if v >= cam.height and 0 <= u < cam.width:     # under the lower edge: the foot of its column answers for it
        return bool(columns_end_on_floor(depth, cam, [math.floor(u)], floor_tolerance_m, window)[0])
    Z = _depth_at(depth, cam, u, v, window)        # None outside the image, too
    if Z is None or abs(Z - z) > depth_tolerance_m:
        return False
    height = cam.T_world_camera[2, 3] * (1.0 - Z / z)          # of what the ray shows: it falls evenly from the camera to z = 0
    return bool(abs(height) <= floor_tolerance_m)


def project(cam: Camera, p_world) -> tuple[float, float, float] | None:
    """(u, v, camera Z) of a world point, or None when it is not in front of the camera.

    The inverse of lift. Image bounds are not tested here; in_image does that.
    """
    x, y, z = p_world
    X, Y, Z = (np.linalg.inv(cam.T_world_camera) @ np.array([x, y, z, 1.0], dtype=np.float64))[:3]
    if Z <= 0:
        return None
    return float(cam.fx * X / Z + cam.cx), float(cam.fy * Y / Z + cam.cy), float(Z)


def ground_hits(cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Where the ray through each pixel centre meets the ground plane z = 0.

    Returns the world XY (H, W, 2) and the camera Z of that point (H, W); both are nan where the ray
    does not meet the ground in front of the camera.
    """
    R, t = cam.T_world_camera[:3, :3], cam.T_world_camera[:3, 3]
    u, v = _pixel_centres(cam.height, cam.width)
    rays = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u)], axis=-1)    # unit camera Z
    d = rays @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -t[2] / d[..., 2]
        s = np.where((d[..., 2] != 0) & (s > 0), s, np.nan)
        xy = t[:2] + s[..., None] * d[..., :2]
    return xy, s


def to_robot_xy(robot: RobotPose, x: float, y: float) -> tuple[float, float]:
    """A world-frame ground position in the robot frame."""
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    dx, dy = x - robot.x, y - robot.y
    return float(c * dx + s * dy), float(c * dy - s * dx)


def to_world_xy(robot: RobotPose, x: float, y: float) -> tuple[float, float]:
    """A robot-frame ground position in the world frame."""
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    return float(robot.x + c * x - s * y), float(robot.y + s * x + c * y)


def bearing(from_xy, to_xy) -> float:
    """The direction from one ground position to another; 0.0 when they coincide."""
    return wrap_angle(math.atan2(to_xy[1] - from_xy[1], to_xy[0] - from_xy[0]))


def make_poses(robot: RobotPose, goal_xy, theta_world: float) -> tuple[Pose, Pose]:
    """(pose_world, pose_robot) for a world-frame goal position and heading."""
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    pose_world = (gx, gy, float(wrap_angle(theta_world)))
    pose_robot = (*to_robot_xy(robot, gx, gy), float(wrap_angle(theta_world - robot.yaw)))
    return pose_world, pose_robot


def camera_on_robot(robot: RobotPose, forward_m: float, height_m: float, pitch_down_rad: float) -> np.ndarray:
    """T_world_camera of an optical-frame camera looking along robot +x, pitched down."""
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    T_world_robot = np.array([[c, -s, 0.0, robot.x], [s, c, 0.0, robot.y], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    cp, sp = math.cos(pitch_down_rad), math.sin(pitch_down_rad)
    T_robot_camera = np.array([[0.0, -sp, cp, forward_m],      # columns: right, down, forward, origin
                               [-1.0, 0.0, 0.0, 0.0],
                               [0.0, -cp, -sp, height_m],
                               [0.0, 0.0, 0.0, 1.0]])
    return T_world_robot @ T_robot_camera
