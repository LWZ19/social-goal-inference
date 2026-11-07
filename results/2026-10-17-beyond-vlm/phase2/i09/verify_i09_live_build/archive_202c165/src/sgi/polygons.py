"""Polygon predicates, distances and region shape builders (metres, counter-clockwise)."""
import math

import numpy as np

Polygon = list[tuple[float, float]]

_EPS = 1e-9   # float-noise tolerance in metres; not a domain threshold


def _xy(p) -> tuple[float, float]:
    return float(p[0]), float(p[1])


def _vertices(poly) -> list[tuple[float, float]]:
    return [_xy(v) for v in poly]


def _segment_distance(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance_to_boundary(p, poly: Polygon) -> float:
    """Distance from p to the nearest edge, inside or outside (the closing edge included)."""
    px, py = _xy(p)
    v = _vertices(poly)
    return float(min(_segment_distance(px, py, *v[i - 1], *v[i]) for i in range(len(v))))


def point_in_polygon(p, poly: Polygon) -> bool:
    """Even-odd rule; the boundary (within _EPS) counts as inside."""
    if distance_to_boundary(p, poly) <= _EPS:
        return True
    px, py = _xy(p)
    v = _vertices(poly)
    inside = False
    for i in range(len(v)):
        xi, yi = v[i - 1]
        xj, yj = v[i]
        if (yi > py) != (yj > py):
            x_cross = xi + (py - yi) * (xj - xi) / (yj - yi)
            if x_cross > px:
                inside = not inside
    return inside


def distance_to_polygon(p, poly: Polygon) -> float:
    """0.0 when p is inside, otherwise the distance to the boundary."""
    if point_in_polygon(p, poly):
        return 0.0
    return distance_to_boundary(p, poly)


def polygon_bounds(poly: Polygon) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax)."""
    v = _vertices(poly)
    xs = [x for x, _ in v]
    ys = [y for _, y in v]
    return (min(xs), min(ys), max(xs), max(ys))


def grid_points_in(poly: Polygon, step: float) -> np.ndarray:
    """Lattice points (i * step, j * step) inside poly, ordered by ascending x then y; shape (N, 2)."""
    xmin, ymin, xmax, ymax = polygon_bounds(poly)
    i0, i1 = math.ceil((xmin - _EPS) / step), math.floor((xmax + _EPS) / step)
    j0, j1 = math.ceil((ymin - _EPS) / step), math.floor((ymax + _EPS) / step)
    pts = []
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            q = (i * step, j * step)
            if point_in_polygon(q, poly):
                pts.append(q)
    return np.array(pts, dtype=np.float64).reshape(-1, 2)


def convex_hull(points) -> Polygon:
    """The convex hull of the points, counter-clockwise from the leftmost of them (the lowest, of several).

    A point inside the hull or on one of its edges is left out, and so is a repeated one. Points that all lie
    on one line give the two ends of it, and points that are all one point give that point: fewer than three
    vertices, so no polygon with an inside."""
    unique = sorted(set(_vertices(points)))
    if len(unique) < 3:
        return unique

    def chain(ordered) -> Polygon:
        """Andrew's monotone chain along the points in order: every turn of it is to the left."""
        kept: Polygon = []
        for p in ordered:
            while len(kept) >= 2 and ((kept[-1][0] - kept[-2][0]) * (p[1] - kept[-2][1])
                                      - (kept[-1][1] - kept[-2][1]) * (p[0] - kept[-2][0])) <= 0.0:
                kept.pop()
            kept.append(p)
        return kept

    lower, upper = chain(unique), chain(reversed(unique))
    return lower[:-1] + upper[:-1]                 # each chain ends on the point the other begins with


def oriented_rectangle(origin, yaw: float, near: float, far: float, width: float) -> Polygon:
    """Rectangle spanning near..far along yaw from origin, width / 2 to each side.

    Corners: near-right, far-right, far-left, near-left (counter-clockwise)."""
    ox, oy = _xy(origin)
    dx, dy = math.cos(yaw), math.sin(yaw)
    lx, ly = -math.sin(yaw), math.cos(yaw)
    h = width / 2
    return [(float(ox + s * dx + t * lx), float(oy + s * dy + t * ly))
            for s, t in ((near, -h), (far, -h), (far, h), (near, h))]


def segment_rectangle(a, b, width: float) -> Polygon:
    """Rectangle centred on segment a-b with its ends flush with a and b."""
    ax, ay = _xy(a)
    bx, by = _xy(b)
    length = math.hypot(bx - ax, by - ay)
    if length <= _EPS:
        raise ValueError("segment_rectangle needs two distinct points")
    return oriented_rectangle((ax, ay), math.atan2(by - ay, bx - ax), 0.0, length, width)


def _circle_vertex(center, r: float, n: int, j: int) -> tuple[float, float]:
    cx, cy = _xy(center)
    t = 2 * math.pi * (j % n) / n
    return (float(cx + r * math.cos(t)), float(cy + r * math.sin(t)))


def disc(center, radius: float, n: int = 48) -> Polygon:
    """Regular polygon with vertices on the circle, from angle 0, counter-clockwise."""
    return [_circle_vertex(center, radius, n, j) for j in range(n)]


def _sector(center, r_in: float, r_out: float, n: int, start: int, m: int) -> Polygon:
    outer = [_circle_vertex(center, r_out, n, j) for j in range(start, start + m + 1)]
    inner = [_circle_vertex(center, r_in, n, j) for j in range(start + m, start - 1, -1)]
    return outer + inner


def ring(center, r_in: float, r_out: float, n: int = 72) -> Polygon:
    """One keyhole polygon: outer circle counter-clockwise, then the inner circle back, joined along angle 0."""
    return _sector(center, r_in, r_out, n, 0, n)


def ring_sectors(center, r_in: float, r_out: float, keep, step_deg: float = 5.0) -> list[Polygon]:
    """Split the ring into step_deg cells, keep those whose centre satisfies keep(x, y), one polygon per run.

    This is a cell approximation: a kept cell covers the whole band r_in..r_out, so a sector can
    contain points where keep is false. Cells cells - 1 and 0 are neighbours, so a run may wrap."""
    cells = round(360 / step_deg)
    cx, cy = _xy(center)
    r_mid = (r_in + r_out) / 2
    kept = []
    for i in range(cells):
        a = 2 * math.pi * (i + 0.5) / cells
        kept.append(bool(keep(float(cx + r_mid * math.cos(a)), float(cy + r_mid * math.sin(a)))))
    if all(kept):
        return [ring(center, r_in, r_out, n=cells)]
    if not any(kept):
        return []
    out = []
    for s in range(cells):
        if kept[s] and not kept[s - 1]:                        # first cell of a run
            m = 1
            while kept[(s + m) % cells]:
                m += 1
            out.append(_sector(center, r_in, r_out, cells, s, m))
    return out
