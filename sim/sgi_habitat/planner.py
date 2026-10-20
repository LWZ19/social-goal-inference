"""A local planner round moving people: A* on a grid of the robot's own navmesh, with each person stamped on it
as a disc, and the path then pulled straight wherever the grid allows.

The navmesh the grid is read from is cut for the robot's clearance (world.robot_pathfinder), so walls and
furniture are already inflated by it. A person blocks the cells within `person_m` of their centre: their own
radius, the robot's clearance and a margin. Only a window round the start and the goal is searched, so a plan
takes a few milliseconds and can be made again twice a second.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Sequence

import numpy as np

RES = 0.1                          # metres per cell
WINDOW_M = 3.0                     # searched beyond the box round the start and the goal
_STEPS = [(dr, dc, math.hypot(dr, dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


class GridPlanner:
    def __init__(self, pathfinder, floor_y: float):
        lo, hi = pathfinder.get_bounds()
        self.x0, self.z0, self.floor_y = float(lo[0]), float(lo[2]), floor_y
        self.free = np.asarray(pathfinder.get_topdown_view(RES, floor_y + 0.15), dtype=bool)   # rows z, columns x

    def cell(self, p) -> tuple:
        return int(round((p[2] - self.z0) / RES)), int(round((p[0] - self.x0) / RES))

    def point(self, row: int, col: int) -> np.ndarray:
        return np.array([self.x0 + col * RES, self.floor_y, self.z0 + row * RES])

    def plan(self, start, goal, people: Sequence[np.ndarray], person_m: float) -> Optional[List[np.ndarray]]:
        """Waypoints in Habitat's frame from `start` to `goal` (the start left out), keeping `person_m` from each
        person; None when the goal is out of reach that way. Cells round the start are freed, so a robot already
        closer to someone than `person_m` can still get away."""
        (r0, c0), (r1, c1) = self.cell(start), self.cell(goal)
        pad = int(WINDOW_M / RES)
        top, left = max(0, min(r0, r1) - pad), max(0, min(c0, c1) - pad)
        bottom, right = min(self.free.shape[0], max(r0, r1) + pad + 1), min(self.free.shape[1], max(c0, c1) + pad + 1)
        if not (top <= r0 < bottom and left <= c0 < right and top <= r1 < bottom and left <= c1 < right):
            return None
        free = self.free[top:bottom, left:right].copy()
        rows, cols = np.mgrid[top:bottom, left:right]
        xs, zs = self.x0 + cols * RES, self.z0 + rows * RES
        for person in people:
            free &= (xs - person[0]) ** 2 + (zs - person[2]) ** 2 > person_m ** 2
        free |= (xs - start[0]) ** 2 + (zs - start[2]) ** 2 <= 0.3 ** 2            # the robot may always leave
        s, g = (r0 - top, c0 - left), (r1 - top, c1 - left)
        if not free[g]:
            return None
        cells = self._astar(free, s, g)
        if cells is None:
            return None
        cells = self._pull(free, cells)
        return [self.point(r + top, c + left) for r, c in cells[1:-1]] + [np.array([goal[0], self.floor_y, goal[2]])]

    @staticmethod
    def _astar(free: np.ndarray, s: tuple, g: tuple) -> Optional[List[tuple]]:
        h = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])
        best = {s: 0.0}
        came = {}
        frontier = [(h(s), 0.0, s)]
        rows, cols = free.shape
        while frontier:
            _, cost, cur = heapq.heappop(frontier)
            if cur == g:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return path[::-1]
            if cost > best.get(cur, math.inf):
                continue
            for dr, dc, step in _STEPS:
                nxt = (cur[0] + dr, cur[1] + dc)
                if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or not free[nxt]:
                    continue
                if dr and dc and not (free[cur[0] + dr, cur[1]] and free[cur[0], cur[1] + dc]):
                    continue                       # no cutting a blocked corner
                new = cost + step
                if new < best.get(nxt, math.inf):
                    best[nxt], came[nxt] = new, cur
                    heapq.heappush(frontier, (new + h(nxt), new, nxt))
        return None

    @staticmethod
    def _clear(free: np.ndarray, a: tuple, b: tuple) -> bool:
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
        rr = np.rint(np.linspace(a[0], b[0], n)).astype(int)
        cc = np.rint(np.linspace(a[1], b[1], n)).astype(int)
        return bool(free[rr, cc].all())

    def _pull(self, free: np.ndarray, cells: List[tuple]) -> List[tuple]:
        """The grid path with every corner dropped that a straight free line can skip."""
        out, i = [cells[0]], 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not self._clear(free, cells[i], cells[j]):
                j -= 1
            out.append(cells[j])
            i = j
        return out
