"""Re-solving a stored answer of the constraint method on tracked people, with no further model call.

The answer names people by number ("come after 4, 5, 6"). Once the numbers are tied to tracks, the solver can be
run again on where those people are now, as often as the tracks are updated: the line moves up, a person who was
served walks away, others join. `Resolver` keeps that bookkeeping and gives the solver its people and
constraints; the solve itself (`sgi.solver.solve`) is the caller's, on a fresh observation.

The rules:

- **First match.** The model's people, as lifted when the answer was asked, are matched one to one to the tracks
  as they stood then, within `gate_m`. A matched person keeps their number; a number with no track is taken out
  of every constraint; a track with no number gets a new one, which no constraint names.
- **Left the line.** A person named in a `come_after` whose distance to the target has grown more than `leave_m`
  beyond their closest approach is taken out of the `come_after`, for good. Should they come back they are one
  more person whom no constraint names.
- **Off the line.** Where the waiting people stood along a line when the answer was asked (`line_direction`,
  from its front to its back), one of them who is more than `off_line_m` from that line has left it too.
- **Gone.** A person who has left the line is taken out of every constraint that names them, not only the
  `come_after`; so is a named person who is not waiting and has moved more than `gone_m` from where they stood
  when the answer was asked. The answer described the people as they were; who has walked off is no longer part
  of it.
- **Joining.** A person the solver has taken as waiting behind the last (`SolverNote.joined_waiting`) in
  `absorb_after` solves running is waiting from then on, and can leave as the others can. So the line is kept
  when everyone the model named has been served and gone.
- **Joined.** Once the robot has stood within `join_radius_m` of a place that was a place in turn for
  `join_dwell_s`, it is in the line: from then on, people farther back than the robot (along the line where its
  direction is known, else from the target) are left out.
- **The floor.** `paint_out` gives the first view with the people who stood in it painted out as floor, for the
  solver to stand the robot on where they stood: what the camera sees now is not needed, and a view that has
  turned away from the line loses nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sgi.inputs import ObservationData
from sgi.types import Constraint, DetectedPerson

_EPS = 1e-9
_ADVANCE_WITHIN_S = 3.0      # co-motion: a step towards the target this quick is an advance
_ADVANCE_EVERY_S = 5.0       # and one person's advances are taken at most this often
_STEP_UP_M = 0.55            # moved this much nearer the target lately: stepped up with the others
_ANCHOR_STILL_M, _ANCHOR_STILL_S = 0.10, 1.0     # anchor_rederive: still means this near over this long
_ANCHOR_SPACING_X, _ANCHOR_FRONT_TOL = 1.5, 0.5   # a step of the chain; the front gap's tolerance
_LOCK_LINE_M = 0.6            # hold_place: someone ahead this near the line when the robot joins is in it


@dataclass(frozen=True)
class Track:
    """One tracked person: an id that stays theirs, the centre of their body on the floor and their heading, in the
    world frame (None where it is not known)."""
    id: int
    xy: tuple[float, float]
    yaw: float | None = None


def match_one_to_one(cost: Sequence[Sequence[float]], gate: float) -> dict[int, int]:
    """The rows and columns paired so that as many pairs as possible cost at most `gate`, and of those pairings the
    one of least total cost (the Hungarian method): {row: column}, pairs over the gate left out."""
    rows = len(cost)
    cols = len(cost[0]) if rows else 0
    if not rows or not cols:
        return {}
    if rows > cols:                                # the method below wants no more rows than columns
        return {r: c for c, r in match_one_to_one([list(col) for col in zip(*cost)], gate).items()}
    big = 1.0 + sum(min(float(v), gate) for row in cost for v in row) * 2 + gate * rows
    a = [[float(v) if float(v) <= gate else big for v in row] for row in cost]
    inf = math.inf
    u, v, p, way = [0.0] * (rows + 1), [0.0] * (cols + 1), [0] * (cols + 1), [0] * (cols + 1)
    for i in range(1, rows + 1):
        p[0], j0 = i, 0
        least, used = [inf] * (cols + 1), [False] * (cols + 1)
        while True:
            used[j0], i0, delta, j1 = True, p[j0], inf, 0
            for j in range(1, cols + 1):
                if not used[j]:
                    here = a[i0 - 1][j - 1] - u[i0] - v[j]
                    if here < least[j]:
                        least[j], way[j] = here, j0
                    if least[j] < delta:
                        delta, j1 = least[j], j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    least[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return {p[j] - 1: j - 1 for j in range(1, cols + 1) if p[j] and a[p[j] - 1][j - 1] <= gate}


def paint_out(observation: ObservationData, people_xy: Sequence[Sequence[float]], radius_m: float = 0.4,
              floor_tolerance_m: float = 0.05) -> ObservationData:
    """The observation with every depth point above the floor within `radius_m` (on the floor plane) of one of
    `people_xy` replaced by where its pixel's ray meets the floor (0 where it never does): the floor as it would be
    seen with those people gone."""
    cam = observation.camera
    depth = observation.depth.astype(np.float64)
    rows, cols = depth.shape
    u, v = np.meshgrid(np.arange(cols) + 0.5, np.arange(rows) + 0.5)
    rays = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u)], -1) @ cam.T_world_camera[:3, :3].T
    origin = cam.T_world_camera[:3, 3]
    points = origin + depth[..., None] * rays          # camera Z along a ray whose camera z is 1
    near = np.zeros(depth.shape, dtype=bool)
    for x, y in people_xy:
        near |= np.hypot(points[..., 0] - x, points[..., 1] - y) <= radius_m
    body = near & np.isfinite(depth) & (depth > 0) & (points[..., 2] > floor_tolerance_m)
    down = rays[..., 2] < -1e-9
    on_floor = np.where(down, -origin[2] / np.where(down, rays[..., 2], -1.0), 0.0)
    painted = depth.copy()
    painted[body] = on_floor[body]
    return ObservationData(observation.rgb, painted.astype(observation.depth.dtype), cam, observation.robot)


def _apart(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Resolver:
    """The people and constraints to solve with at each update of the tracks, for one stored answer."""

    def __init__(self, people: Sequence[DetectedPerson], constraints: Sequence[Constraint],
                 target_xy: Sequence[float], at_answer: Sequence[Track], *, gate_m: float = 0.75,
                 leave_m: float = 0.75, join_radius_m: float = 0.35, join_dwell_s: float = 1.0,
                 line_span_m: float = 1.0, off_line_m: float = 1.0, absorb_after: int = 2, gone_m: float = 1.5,
                 readmit: bool = False, vacate_after_s: float | None = None, vacate_rejoined: bool = False,
                 path_memory: bool = False,
                 co_motion_s: float | None = None, co_motion_follow: bool = True, co_motion_rejoin: bool = False,
                 advance_m: float = 0.4,
                 follow_m: float = 0.25,
                 moved_m: float = 1.0, history_s: float = 20.0, group_follow: bool = False,
                 anchor_rederive: bool = False, hold_place: bool = False):
        """The options after `gone_m` are off by
        default. The first group re-admits and vacates: `readmit` lets a person who left wait again once the solver takes them as joining behind the last;
        `vacate_after_s` gives `vacated_pose`, the last still pose of the last of the waiting people to leave, once
        the robot has joined, nobody it came after still waits and that person has been away so long (`vacate_rejoined`
        also takes it when that person is back in the line but behind the robot, as one served who rejoined); `path_memory`
        remembers the line as the polyline through the waiting people's positions at the answer, not a straight
        axis, and points the solver along its back (`solve_direction`).

        `co_motion_s` asks the people to move together: when a
        waiting person advances (`advance_m` nearer the target within a few seconds), the waiting person next behind
        them must have come `follow_m` nearer within `co_motion_s`, or is taken as not waiting (`non_followers`);
        and nobody is taken as joining unless, within `history_s`, they walked `moved_m` or came a step nearer the
        target. The model's answer says who waits; how people move afterwards can overrule it. `exempt()` gives
        those who may not join, for the solver; `co_motion_follow` False keeps the gate and leaves out the follow
        check (the gate alone); `co_motion_rejoin` cancels a follow check made before the person was taken
        back into the line.

        Two more, using the same evidence: `group_follow` checks every
        waiting person behind an advance, not only the next; `anchor_rederive` finds the waiting people again from
        the target point once nobody the answer named still waits (the still person at the front gap the named
        people showed, then still people behind a member within 1.5 spacings), never taking someone found not
        following, and nobody unnamed who has not moved as a joiner would.

        `hold_place` keeps the robot's place once it has joined: the people ahead of it then
        are locked in (`locked`): those taken as waiting, and anyone else standing within 0.6 m of the line ahead
        of it (someone who has just come back to the line, before the rules have taken them back). From then on
        that set only shrinks. Nobody is added to it: no
        joiner, no one taken back after leaving, no one found again from the anchor, and everyone outside it is
        exempt from the solver's place-in-turn rule. People still leave it as before (served and gone, off the
        line, or not following an advance), so someone wrongly taken as waiting can still be dropped. Once it is
        empty, the robot is first, and with `vacate_after_s` set it goes at once (`vacated_pose`) to the front of
        the line: the still pose nearest the target that any of the waiting people stood in (where they were
        served), not the model's target point nor where the last of them happened to stand."""
        self._target = (float(target_xy[0]), float(target_xy[1]))
        self._readmit, self._vacate_after_s, self._vacate_rejoined = readmit, vacate_after_s, vacate_rejoined
        self._leave_m, self._join_radius_m, self._join_dwell_s = leave_m, join_radius_m, join_dwell_s
        cost = [[_apart(person.xy, track.xy) for track in at_answer] for person in people]
        pairs = match_one_to_one(cost, gate_m)
        self._number_of = {at_answer[j].id: people[i].number for i, j in pairs.items()}
        matched = set(self._number_of.values())
        self.match_rate = len(pairs) / len(people) if people else None
        self._next = max((person.number for person in people), default=0) + 1
        self._constraints = [c.model_copy(update={"people": [n for n in c.people if n in matched]}) for c in constraints]
        self._waiting = {n for c in self._constraints if c.kind == "come_after" for n in c.people}
        self._closest = {self._number_of[track.id]: _apart(track.xy, self._target)
                         for track in at_answer if self._number_of.get(track.id) in self._waiting}
        self._front: tuple[float, float] | None = None
        self.line_direction = self._line_direction(at_answer, line_span_m)
        self._path: list[tuple[float, float]] | None = None
        if path_memory and self.line_direction is not None:
            ordered = sorted((t.xy for t in at_answer if self._number_of.get(t.id) in self._waiting),
                             key=lambda xy: _apart(xy, self._target))
            self._path = [(float(x), float(y)) for x, y in ordered]
        self.solve_direction = self.line_direction
        self.vacated_pose: tuple[float, float, float] | None = None
        self._left_at: dict[int, float] = {}          # number: when a waiting person was taken as having left
        self._recent: dict[int, list] = {}            # track id: (t, xy) over the last second, for the still poses
        self._still: dict[int, tuple[float, float, float]] = {}
        self._off_line_m, self._absorb_after, self._gone_m = off_line_m, absorb_after, gone_m
        self._at_answer = {self._number_of[t.id]: t.xy for t in at_answer if t.id in self._number_of}
        self._named = {n for c in self._constraints for n in c.people}
        self.gone: set[int] = set()                   # named, not waiting, and walked off
        self._absorbing: dict[int, int] = {}          # number: solves running in which the solver took them as joining
        self._joiners: list[int] = []                 # those who have joined the waiting people since the answer
        self.dropped: set[int] = set()
        self.joined = False
        self._place: tuple[float, float] | None = None
        self._in_turn = False
        self._near_since: float | None = None
        self._co_motion_s, self._advance_m, self._follow_m = co_motion_s, advance_m, follow_m
        self._co_motion_follow, self._co_motion_rejoin = co_motion_follow, co_motion_rejoin
        self._readmitted_at: dict[int, float] = {}    # number: when they were last taken back into the line
        self._now: float | None = None                # the time of the last update of the tracks
        self.non_follower_t: dict[int, float] = {}    # number: when they were taken as not following
        self.events: list[tuple] = []                 # for the run logs: (t, "advance", q, p, before) and drops
        self._moved_m, self._history_s = moved_m, history_s
        self._history: dict[int, list] = {}           # number: (t, xy, distance to the target) over `history_s`
        self._advanced_at: dict[int, float] = {}      # number: when they last advanced
        self._due: list[tuple[float, int, float]] = []    # (when to look, who should follow, their distance before)
        self.non_followers: set[int] = set()
        self._group_follow, self._anchor_rederive = group_follow, anchor_rederive
        named = sorted((self._at_answer[n] for n in self._waiting if n in self._at_answer),
                       key=lambda xy: _apart(xy, self._target))
        self._front_gap = _apart(named[0], self._target) if named else None
        steps = [_apart(a, b) for a, b in zip(named, named[1:])]
        self._spacing = sum(steps) / len(steps) if steps else 0.8
        self.rederived = 0                             # times the waiting people were found again from the anchor
        self._hold_place = hold_place
        self.locked: set[int] | None = None            # with `hold_place`: those ahead of the robot when it joined
        self.locked_t: float | None = None

    def _line_direction(self, at_answer: Sequence[Track], span_m: float) -> tuple[float, float] | None:
        """From the front of the waiting people to their back as they stood when the answer was asked: the one
        nearest the target to the one farthest from it, when they are at least `span_m` apart."""
        waiting = [t.xy for t in at_answer if self._number_of.get(t.id) in self._waiting]
        if len(waiting) < 2:
            return None
        front = min(waiting, key=lambda xy: _apart(xy, self._target))
        back = max(waiting, key=lambda xy: _apart(xy, self._target))
        span = _apart(front, back)
        if span < span_m:
            return None
        self._front = (float(front[0]), float(front[1]))
        return ((back[0] - front[0]) / span, (back[1] - front[1]) / span)

    def _on_path(self, xy) -> tuple[float, float, tuple[float, float]]:
        """(arc length, distance, tangent) of a point on the remembered polyline, extended past both ends."""
        best = None
        n = len(self._path)
        start = 0.0
        for i in range(n - 1):
            (ax, ay), (bx, by) = self._path[i], self._path[i + 1]
            length = math.hypot(bx - ax, by - ay)
            if length < _EPS:
                continue
            ux, uy = (bx - ax) / length, (by - ay) / length
            s_ = (xy[0] - ax) * ux + (xy[1] - ay) * uy
            s_ = min(max(s_, -math.inf if i == 0 else 0.0), math.inf if i == n - 2 else length)
            d = math.hypot(xy[0] - (ax + s_ * ux), xy[1] - (ay + s_ * uy))
            if best is None or d <= best[1] + _EPS:            # of equals, the later segment
                best = (start + s_, d, (ux, uy))
            start += length
        return best

    def update_joined(self, robot_xy: Sequence[float], t: float) -> None:
        """The robot has joined once it has stood within `join_radius_m` of the last place, a place in turn, for
        `join_dwell_s`; joined for good."""
        if self.joined:
            return
        near = self._in_turn and self._place is not None and _apart(robot_xy, self._place) <= self._join_radius_m
        if not near:
            self._near_since = None
        else:
            self._near_since = t if self._near_since is None else self._near_since
            self.joined = t - self._near_since >= self._join_dwell_s - _EPS

    def behind(self, xy: Sequence[float], robot_xy: Sequence[float]) -> bool:
        """Farther back than the robot: along the line when there is one, else from the target point."""
        if self.line_direction is not None:
            return self._along(xy) > self._along(robot_xy) + _EPS
        return _apart(xy, self._target) > _apart(robot_xy, self._target) + _EPS

    def back_tangent(self, points) -> tuple[float, float] | None:
        """The remembered line's direction at the one of `points` farthest back along it (as `path_memory` points
        the solver); None without a remembered line or points."""
        if self._path is None or not points:
            return None
        return self._on_path(max(points, key=self._along))[2]

    def _along(self, xy) -> float | None:
        if self.line_direction is None:
            return None
        if self._path is not None:
            return self._on_path(xy)[0]
        return (xy[0] - self._front[0]) * self.line_direction[0] + (xy[1] - self._front[1]) * self.line_direction[1]

    def _off_line(self, xy) -> float:
        if self._path is not None:
            return self._on_path(xy)[1]
        dx, dy = self.line_direction
        return abs((xy[0] - self._front[0]) * dy - (xy[1] - self._front[1]) * dx)

    def settled(self, tracks: Sequence[Track], t: float, radius_m: float = 3.0, move_m: float = 0.10,
                window_s: float = 1.0) -> bool:
        """Whether every tracked person within `radius_m` of the waiting people (of the people the answer named,
        while none waits) has moved at most `move_m` over the last `window_s` seconds: for a caller that solves
        again only on settled tracks. Keeps the tracks' history."""
        history = getattr(self, "_history", {})
        self._history = history
        for track in tracks:
            seen = [(ts, xy) for ts, xy in history.get(track.id, []) if t - ts <= window_s + _EPS]
            history[track.id] = seen + [(t, (float(track.xy[0]), float(track.xy[1])))]
        left = self.dropped | self.gone
        anchors = [tr.xy for tr in tracks if self._number_of.get(tr.id) in self._waiting - left]
        if not anchors:
            anchors = [tr.xy for tr in tracks if self._number_of.get(tr.id) in self._named - left]
        for track in tracks:
            if not any(_apart(track.xy, a) <= radius_m + _EPS for a in anchors):
                continue
            if any(_apart(xy, track.xy) > move_m + _EPS for _, xy in history[track.id]):
                return False
        return True

    def record(self, place_xy: Sequence[float], in_turn: bool, absorbed: Sequence[int] = ()) -> None:
        """The place the last solve chose, whether it was a place in turn (a `come_after` was used), and whom the
        solver took as joining the waiting people behind the last (`SolverNote.joined_waiting`)."""
        self._place, self._in_turn = (float(place_xy[0]), float(place_xy[1])), in_turn
        if self.locked is not None:                    # hold_place: nobody joins ahead of the robot any more
            self._absorbing = {}
            return

        def eligible(n):
            ok = n not in self._waiting or (self._readmit and n in self.dropped)
            return ok and (self._co_motion_s is None or self._moved(n))

        self._absorbing = {n: self._absorbing.get(n, 0) + 1 for n in absorbed if eligible(n)}
        for number, count in self._absorbing.items():
            if count >= self._absorb_after:
                if number in self.dropped:                     # back in the line (v2's readmit)
                    self.dropped.discard(number)
                    if self._now is not None:
                        self._readmitted_at[number] = self._now
                    self._closest.pop(number, None)
                    self._left_at.pop(number, None)
                if number not in self._waiting:
                    self._waiting.add(number)
                    self._joiners.append(number)

    def inputs(self, tracks: Sequence[Track], robot_xy: Sequence[float],
               t: float) -> tuple[list[DetectedPerson], list[Constraint]]:
        """The people (numbered, where they are now) and the constraints to solve with, at time `t` (seconds)."""
        people, constraints = self._inputs(tracks, robot_xy, t)
        if self._anchor_rederive:
            constraints = self._rederive(people, constraints, t)
        return people, constraints

    def _rederive(self, people: list[DetectedPerson], constraints: list[Constraint], t: float) -> list[Constraint]:
        """With `anchor_rederive`: when nobody named still waits, the waiting people found again from the anchor."""
        if self.locked is not None:                    # hold_place: an empty locked set means the robot is first
            return constraints
        if self._front_gap is None or (self._waiting - self.dropped - self.gone):
            return constraints
        where = {p.number: p.xy for p in people}
        joiner = lambda n: n in self._named or self._moved(n)          # noqa: E731
        pool = [n for n in where if n not in self.non_followers and self._is_still(n, t)]
        base = [n for n in pool if joiner(n)
                and abs(_apart(where[n], self._target) - self._front_gap) <= _ANCHOR_FRONT_TOL + _EPS]
        if not base:
            return constraints
        chain = [min(base, key=lambda n: _apart(where[n], self._target))]
        while True:
            last = chain[-1]
            nxt = [n for n in pool if n not in chain
                   and _apart(where[n], where[last]) <= _ANCHOR_SPACING_X * self._spacing
                   and _apart(where[n], self._target) > _apart(where[last], self._target) + _EPS
                   and (self._co_motion_s is None or joiner(n))]
            if not nxt:
                break
            chain.append(min(nxt, key=lambda n: _apart(where[n], where[last])))
        for n in chain:
            self.dropped.discard(n)
            self.gone.discard(n)
            self._closest.pop(n, None)
            if n not in self._waiting:
                self._waiting.add(n)
                self._joiners.append(n)
        self.rederived += 1
        left = self.dropped | self.gone
        out, first = [], True
        for c in self._constraints:
            named = c.people + (self._joiners if c.kind == "come_after" and first else [])
            first = first and c.kind != "come_after"
            out.append(c.model_copy(update={"people": [n for n in named if n not in left]}))
        return out

    def _is_still(self, number: int, t: float) -> bool:
        """Within `_ANCHOR_STILL_M` of where they are now over the last `_ANCHOR_STILL_S` (from the history kept)."""
        h = [x for x in self._history.get(number, []) if t - x[0] <= _ANCHOR_STILL_S + _EPS]
        return len(h) >= 2 and all(_apart(x[1], h[-1][1]) <= _ANCHOR_STILL_M + _EPS for x in h)

    def _inputs(self, tracks: Sequence[Track], robot_xy: Sequence[float],
                t: float) -> tuple[list[DetectedPerson], list[Constraint]]:
        self._now = t
        self.update_joined(robot_xy, t)
        people = []
        for track in tracks:
            number = self._number_of.get(track.id)
            if number is None:
                number, self._next = self._next, self._next + 1
                self._number_of[track.id] = number
            gap = _apart(track.xy, self._target)
            if self._co_motion_s is not None:
                kept = [h for h in self._history.get(number, []) if t - h[0] <= self._history_s + _EPS]
                self._history[number] = kept + [(t, (float(track.xy[0]), float(track.xy[1])), gap)]
            if number in self._waiting and number not in self.dropped:
                closest = min(self._closest.get(number, gap), gap)
                self._closest[number] = closest
                if gap - closest > self._leave_m + _EPS or (
                        self.line_direction is not None and self._off_line(track.xy) > self._off_line_m + _EPS):
                    self.dropped.add(number)
                    self._left_at[number] = t
            if self._vacate_after_s is not None:
                recent = [(ts, xy) for ts, xy in self._recent.get(track.id, []) if t - ts <= 1.0 + _EPS]
                recent.append((t, (float(track.xy[0]), float(track.xy[1]))))
                self._recent[track.id] = recent
                if (number in self._waiting and number not in self.dropped and t - recent[0][0] >= 0.9
                        and all(_apart(xy, track.xy) <= 0.10 + _EPS for _, xy in recent)):
                    self._still[number] = (float(track.xy[0]), float(track.xy[1]),
                                           0.0 if track.yaw is None else float(track.yaw))
            elif (number in self._named and number not in self._waiting and number in self._at_answer
                  and _apart(track.xy, self._at_answer[number]) > self._gone_m + _EPS):
                self.gone.add(number)
            people.append(DetectedPerson(number=number, xy=(float(track.xy[0]), float(track.xy[1])), yaw=track.yaw,
                                         pixel=(0.0, 0.0)))
        if self._co_motion_s is not None and self._co_motion_follow:
            self._follow_up(t)
        if self._path is not None:                      # the solver looks along the back of the line
            active = [tr for tr in tracks if self._number_of.get(tr.id) in self._waiting - self.dropped - self.gone]
            if active:
                self.solve_direction = self._on_path(max(active, key=lambda tr: self._along(tr.xy)).xy)[2]
        self.vacated_pose = self._vacated(tracks, t, robot_xy)
        if self._hold_place and self.joined and self.locked is None:
            where = {p.number: p.xy for p in people}
            ahead = {n for n, xy in where.items() if not self.behind(xy, robot_xy)}
            listed = {n for n in self._waiting - self.dropped - self.gone if n in ahead}
            lined = set()                              # and whoever stands in the line ahead, listed or not
            if self.line_direction is not None:
                lined = {n for n in ahead if n not in self.non_followers
                         and self._off_line(where[n]) <= _LOCK_LINE_M + _EPS
                         and self._along(where[n]) >= -_LOCK_LINE_M - _EPS}       # not beyond the front
            for n in lined - listed:
                self.dropped.discard(n)
                self.gone.discard(n)
                if n not in self._waiting:
                    self._waiting.add(n)
                    self._joiners.append(n)
            self.locked = listed | lined
            self.locked_t = t
            self.events.append((round(t, 2), "locked", sorted(self.locked)))
        if self.locked is not None:
            self._waiting &= self.locked               # the set only shrinks: drops and leaving still apply
        if self.joined:
            people = [p for p in people if not self.behind(p.xy, robot_xy)]
        left = self.dropped | self.gone
        constraints, first = [], True
        for c in self._constraints:
            named = c.people + (self._joiners if c.kind == "come_after" and first else [])
            first = first and c.kind != "come_after"
            if self.locked is not None and c.kind == "come_after":
                named = [n for n in named if n in self.locked]
            constraints.append(c.model_copy(update={"people": [n for n in named if n not in left]}))
        return people, constraints

    def _gap_since(self, number: int, since: float) -> float:
        """Distance to the target at the first record at or after `since`."""
        return next(h[2] for h in self._history[number] if h[0] >= since - _EPS)

    def _follow_up(self, t: float) -> None:
        """Advances of the waiting people now, and who did not follow an advance in time."""
        active = {n: self._history[n][-1][2] for n in self._waiting - self.dropped - self.gone
                  if n in self._history and abs(self._history[n][-1][0] - t) <= _EPS}
        for q, gap in active.items():
            if (self._gap_since(q, t - _ADVANCE_WITHIN_S) - gap >= self._advance_m - _EPS
                    and t - self._advanced_at.get(q, -1e9) > _ADVANCE_EVERY_S):
                self._advanced_at[q] = t
                behind = [n for n in active if n != q and active[n] > gap + _EPS]
                if behind and self._group_follow:      # everyone behind the advance
                    for p in behind:
                        if not any(d[1] == p for d in self._due):
                            self._due.append((t + self._co_motion_s, p, self._gap_since(p, t - _ADVANCE_WITHIN_S)))
                            self.events.append((round(t, 2), "advance", q, p, round(self._due[-1][2], 3),
                                                round(self._gap_since(q, t - _ADVANCE_WITHIN_S), 3), round(gap, 3)))
                elif behind:
                    p = min(behind, key=active.get)
                    self._due.append((t + self._co_motion_s, p, self._gap_since(p, t - _ADVANCE_WITHIN_S)))
                    self.events.append((round(t, 2), "advance", q, p, round(self._due[-1][2], 3),
                                        round(self._gap_since(q, t - _ADVANCE_WITHIN_S), 3), round(gap, 3)))
        due, self._due = [d for d in self._due if d[0] <= t + _EPS], [d for d in self._due if d[0] > t + _EPS]
        for when, p, before in due:
            if self._co_motion_rejoin and self._readmitted_at.get(p, -1e9) > when - self._co_motion_s + _EPS:
                continue                                # taken back since the advance: that check is stale
            if p in active and before - active[p] < self._follow_m - _EPS:
                self.dropped.add(p)
                self._left_at[p] = t
                self.non_followers.add(p)
                self.non_follower_t[p] = t
                self.events.append((round(t, 2), "drop", p, round(before, 3), round(active[p], 3)))

    def exempt(self) -> set[int]:
        """With co-motion: everyone seen who is not waiting and has not moved as a joiner would, for the solver's
        place-in-turn rule to leave out. Without it: nobody."""
        waiting = self._waiting - self.dropped - self.gone
        if self.locked is not None:                    # hold_place: everyone outside the locked set
            return set(self._number_of.values()) - waiting
        if self._co_motion_s is None:
            return set()
        return {n for n in self._history if n not in waiting and not self._moved(n)}

    def waiting(self) -> set[int]:
        """The numbers of those taken as waiting now: named or joined, and not dropped or gone."""
        return self._waiting - self.dropped - self.gone

    def track_of(self, number: int) -> int | None:
        """The track id a person number stands for, or None."""
        return next((track for track, n in self._number_of.items() if n == number), None)

    def _moved(self, number: int) -> bool:
        """Walked `moved_m`, or came a step nearer the target, within the history kept."""
        history = self._history.get(number, [])
        if not history:
            return False
        _, xy, gap = history[-1]
        return any(_apart(xy, h[1]) >= self._moved_m - _EPS or h[2] - gap >= _STEP_UP_M - _EPS for h in history)

    def _vacated(self, tracks: Sequence[Track], t: float,
                 robot_xy: Sequence[float] | None = None) -> tuple[float, float, float] | None:
        if self._vacate_after_s is None or not self.joined or not self._left_at:
            return None
        if self._waiting - self.dropped - self.gone:
            return None
        if self.locked is not None:                    # hold_place, first in line: at once, to the front of the
            if not self._still:                        # line, where the people ahead stood still nearest the target
                return None                            # (where they were served), not where the last to leave stood
            x, y, _ = min(self._still.values(), key=lambda pose: _apart(pose[:2], self._target))
            if self._path is not None and len(self._path) >= 2:          # facing up the line, to its front
                up = (self._path[0][0] - self._path[1][0], self._path[0][1] - self._path[1][1])
            elif self.line_direction is not None:
                up = (-self.line_direction[0], -self.line_direction[1])
            else:
                up = (self._target[0] - x, self._target[1] - y)
            return (x, y, math.atan2(up[1], up[0]))
        last = max(self._left_at, key=self._left_at.get)
        if t - self._left_at[last] < self._vacate_after_s - _EPS:
            return None
        now = next((tr.xy for tr in tracks if self._number_of.get(tr.id) == last), None)
        if now is not None:                            # back within 0.75 m of the line (or of their place): not gone
            if self.line_direction is not None:
                back = self._off_line(now) <= self._leave_m + _EPS
            else:
                back = _apart(now, self._target) - self._closest.get(last, 0.0) <= self._leave_m + _EPS
            rejoined = self._vacate_rejoined and robot_xy is not None and self.behind(now, robot_xy)
            if back and not rejoined:                  # vacate_rejoined: back in the line behind the robot is gone
                return None
        return self._still.get(last)
