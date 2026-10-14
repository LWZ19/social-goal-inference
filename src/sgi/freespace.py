"""Free space on a grid: static cells, inflation by the robot radius and reachability (harness side).

A cell stands for its centre: every rule is evaluated at cell centres and every distance is
measured from a cell centre. "Within d" means distance <= d + _EPS."""
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from sgi.config import Thresholds
from sgi.geometry import RobotPose
from sgi.polygons import Polygon, polygon_bounds
from sgi.scene import GroundTruth

_EPS = 1e-9   # float-noise tolerance (6 * 0.05 is 0.30000000000000004); not a threshold
_FREE_PIXEL_MIN = 250   # scene format v0.1: an occupancy pixel is free only when it is this bright or brighter


@dataclass(frozen=True)
class Grid:
    x0: float
    y0: float
    resolution: float
    free: np.ndarray                               # bool (ny, nx); free[iy, ix], y grows with iy

    def index(self, x: float, y: float) -> tuple[int, int] | None:
        """(iy, ix) of the cell holding (x, y), or None outside the grid."""
        ix = math.floor((x - self.x0) / self.resolution)
        iy = math.floor((y - self.y0) / self.resolution)
        ny, nx = self.free.shape
        if 0 <= ix < nx and 0 <= iy < ny:
            return iy, ix
        return None

    def center(self, iy: int, ix: int) -> tuple[float, float]:
        """World (x, y) of a cell's centre."""
        return (self.x0 + (ix + 0.5) * self.resolution, self.y0 + (iy + 0.5) * self.resolution)

    def is_free(self, x: float, y: float) -> bool:
        """Whether the cell holding (x, y) is free; False outside the grid."""
        cell = self.index(x, y)
        return cell is not None and bool(self.free[cell])


def _cells(extent_m: float, resolution: float) -> int:
    """Cells needed to cover extent_m; the last one may overhang."""
    return max(1, math.ceil(extent_m / resolution - _EPS))


def _centres(origin: float, n: int, resolution: float) -> np.ndarray:
    return origin + (np.arange(n) + 0.5) * resolution


def _inside(xs: np.ndarray, ys: np.ndarray, poly: Polygon) -> np.ndarray:
    """Even-odd test of every (xs[ix], ys[iy]) against poly; shape (ny, nx). A point on an edge may fall either way."""
    x, y = xs[None, :], ys[:, None]
    inside = np.zeros((ys.size, xs.size), dtype=bool)
    vertices = [(float(vx), float(vy)) for vx, vy in poly]
    for (xi, yi), (xj, yj) in zip(vertices[-1:] + vertices[:-1], vertices):
        if yi == yj:                               # a horizontal edge crosses no horizontal ray
            continue
        x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
        inside ^= ((yi > y) != (yj > y)) & (x_cross > x)
    return inside


def _image_free(xs: np.ndarray, ys: np.ndarray, truth: GroundTruth) -> np.ndarray:
    """Whether each (xs[ix], ys[iy]) lies on a free pixel of the occupancy image; shape (ny, nx).

    Columns run along +X and rows along -Y from the origin, the outer corner of the bottom-left pixel."""
    image = truth.occupancy_image
    rows, cols = image.shape
    pixel_m = truth.occupancy.resolution
    ox, oy = truth.occupancy.origin[0], truth.occupancy.origin[1]    # a third element is ignored
    col = np.floor((xs - ox) / pixel_m).astype(int)
    up = np.floor((ys - oy) / pixel_m).astype(int)                   # rows counted from the bottom
    col_ok, up_ok = (col >= 0) & (col < cols), (up >= 0) & (up < rows)
    free = np.zeros((ys.size, xs.size), dtype=bool)                  # outside the image is not free
    free[np.ix_(up_ok, col_ok)] = image[np.ix_(rows - 1 - up[up_ok], col[col_ok])] >= _FREE_PIXEL_MIN
    return free


def static_grid(truth: GroundTruth, resolution: float) -> Grid:
    """The static free space on a grid over the bounding box of bounds, or over the occupancy image without bounds.

    A cell is free when its centre is inside bounds (if given), outside every obstacle polygon and on a
    free occupancy pixel (if an image is given)."""
    if truth.bounds is not None:
        x0, y0, x1, y1 = polygon_bounds(truth.bounds)
        width_m, height_m = x1 - x0, y1 - y0
    elif truth.occupancy is not None:
        rows, cols = truth.occupancy_image.shape
        x0, y0 = truth.occupancy.origin[0], truth.occupancy.origin[1]
        width_m, height_m = cols * truth.occupancy.resolution, rows * truth.occupancy.resolution
    else:
        raise ValueError(f"{truth.scene_id}: free space needs bounds, occupancy or both")
    xs = _centres(x0, _cells(width_m, resolution), resolution)
    ys = _centres(y0, _cells(height_m, resolution), resolution)
    free = np.ones((ys.size, xs.size), dtype=bool)
    if truth.bounds is not None:
        free &= _inside(xs, ys, truth.bounds)
    for obstacle in truth.obstacles:
        free &= ~_inside(xs, ys, obstacle)
    if truth.occupancy is not None:
        free &= _image_free(xs, ys, truth)
    return Grid(x0=x0, y0=y0, resolution=resolution, free=free)


def _near_blocked(blocked: np.ndarray, radius_m: float, resolution: float) -> np.ndarray:
    """Cells whose centre is within radius_m of a blocked cell's centre; everything outside the array is blocked."""
    ny, nx = blocked.shape
    pad = math.ceil(radius_m / resolution)
    padded = np.pad(blocked, pad, constant_values=True)
    near = np.zeros_like(blocked)
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            if math.hypot(dx, dy) * resolution <= radius_m + _EPS:
                near |= padded[pad + dy:pad + dy + ny, pad + dx:pad + dx + nx]
    return near


def robot_grid(truth: GroundTruth, robot: RobotPose, th: Thresholds) -> Grid:
    """Where the robot's centre may be: the static grid with obstacles, the grid edge and people inflated.

    Cells within the robot radius of the robot's own position are free whatever blocked them."""
    static = static_grid(truth, th.grid_resolution_m)
    ny, nx = static.free.shape
    xs = _centres(static.x0, nx, static.resolution)[None, :]
    ys = _centres(static.y0, ny, static.resolution)[:, None]
    blocked = _near_blocked(~static.free, th.robot_radius_m, static.resolution)
    for person in truth.people:
        blocked |= np.hypot(xs - person.x, ys - person.y) <= th.person_radius_m + th.robot_radius_m + _EPS
    blocked[np.hypot(xs - robot.x, ys - robot.y) <= th.robot_radius_m + _EPS] = False
    return Grid(x0=static.x0, y0=static.y0, resolution=static.resolution, free=~blocked)


def connected(grid: Grid, start_xy) -> np.ndarray:
    """The free cells joined to the start's cell through edge neighbours; all False when the start is not on a free cell."""
    ny, nx = grid.free.shape
    seen = np.zeros((ny, nx), dtype=bool)
    start = grid.index(*start_xy)
    if start is None or not grid.free[start]:
        return seen
    seen[start] = True
    queue = deque([start])
    while queue:
        iy, ix = queue.popleft()
        for cell in ((iy - 1, ix), (iy + 1, ix), (iy, ix - 1), (iy, ix + 1)):
            if 0 <= cell[0] < ny and 0 <= cell[1] < nx and grid.free[cell] and not seen[cell]:
                seen[cell] = True
                queue.append(cell)
    return seen


def reachable(grid: Grid, start_xy, goal_xy) -> bool:
    """Whether the goal's cell is free and joined to the start's cell."""
    goal = grid.index(*goal_xy)
    return goal is not None and bool(connected(grid, start_xy)[goal])
