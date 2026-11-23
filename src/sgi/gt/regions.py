"""Reference region rules: where the robot may stand, and may not, per situation.

Pure functions of scene-format fields. Each returns (acceptable regions, forbidden regions) for one target."""
import math
from itertools import groupby
from typing import Callable, Sequence

from sgi.geometry import wrap_angle
from sgi.polygons import Polygon, convex_hull, disc, oriented_rectangle, ring, segment_rectangle
from sgi.scene import ForbiddenRegion, Heading, Person, Region, Target

# These numbers define the reference rules; they are not Config.thresholds.
# The first two say how near a person the robot may stand and how far off. They are the distances of "Where
# Should I Join? Robot Group Joining via Language-Guided Goal Prediction" (arXiv
# 2609.28467, September 2026), which counts a pose as valid 0.5 to 1.2 m from the relevant member: the last
# member of a queue, any member of a conversation.
_JOIN_NEAR_M = 0.5
_JOIN_FAR_M = 1.2
_NO_ONE_NEAR_M = 0.6                # a place or an object with nobody at it: this project's own numbers
_NO_ONE_FAR_M = 1.6
_NO_ONE_WIDTH_M = 1.2
_LINE_CORRIDOR_WIDTH_M = 1.2
# The rules for people who attend to something together and for a picture being taken are this project's own.
# They are built on the two distances of the paper above: the
# robot joins people at arm's length or a little more from the nearest of them. The three numbers below are
# theirs alone.
_VIEW_CLEAR_M = 0.3                 # how far the robot's centre keeps outside the space between people and what they
#                                     attend to: half a body's width, so that no part of a body stands in that space
_TAKER_SEES_DEG = 60.0              # whoever takes a picture has everyone else within this angle of the way they face,
#                                     and everyone else has them within it of theirs
_FRAME_MARGIN_M = 0.5               # how far to each side of the outermost of those in a picture its strip reaches
# Someone using a machine, a cash machine or a ticket machine, with nobody else waiting: this project's own rule.
# Banks commonly ask the next customer to
# wait about a metre back for privacy; 2 m keeps the robot clearly waiting for this machine; 60 degrees counts
# slightly-off-behind as polite and leaves out standing level beside the user.
_MACHINE_NEAR_M = 1.0               # no nearer the user than this: within it is over their shoulder, and forbidden
_MACHINE_NEAR_SENSITIVITY_M = 0.8   # the near limit of the variant reported beside the rule, fixed with it (no bar)
_MACHINE_FAR_M = 2.0                # and no further from them than this
_MACHINE_BEHIND_DEG = 60.0          # and within this of straight behind them, opposite the way they face
_MACHINE_STRIP_M = _JOIN_NEAR_M     # the strip before the machine reaches this past the user, and past each side of
#                                     the machine: arm's length, where a second person would stand at it with them

# How finely a band is sampled; these set the polygons' resolution, not the rules.
_BAND_RAYS = 180
_BAND_RADIAL_STEP_M = 0.02
_BAND_MIN_DEPTH_M = 0.1
_WALK_STEP_M = 0.05                 # along a straight stretch of a walked outline, between two normals, or less
_WALK_TURN_RAD = 2 * math.pi / _BAND_RAYS   # and round a corner of it, 2 degrees: the angle between two rays of a ring

_EPS_M = 1e-9   # float-noise tolerance, the one sgi.polygons uses: points this close are one point; not a rule number
_STRAIGHT_RAD = 1e-9    # float noise too, of an angle: a hull vertex at which the boundary turns by less lies on a
#                         straight stretch, and two normals of a walk that are nearer one another are one normal

_Point = tuple[float, float]
_Stretch = tuple[_Point, _Point]
_Regions = tuple[list[Region], list[ForbiddenRegion]]


class NoRule(ValueError):
    """A rule's own conditions are not met by what it was given: too few members, a target that has to be one
    of them and is not or must not be and is, a line with no direction, a front that is no direction, a picture
    with no one taking it.

    build_regions answers it with None. Any other error out of a rule is a fault, and passes through."""


def _ground(target: Target) -> _Point:
    """The anchor's ground projection."""
    return (target.anchor[0], target.anchor[1])


def _numbers(target: Target, *values: float) -> None:
    """Refuse a position or a facing that is no finite number. That is a fault of the scene and no condition of
    a rule, so it is a plain error, which build_regions lets through."""
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"target {target.id!r}: a position or a facing that its rule reads is not a finite number")


def _dist(p: _Point, q: _Point) -> float:
    return math.hypot(q[0] - p[0], q[1] - p[1])


def _region(target: Target, polygon: Polygon, heading: Heading) -> Region:
    return Region(polygon=polygon, heading=heading, target_id=target.id)


def _front(target: Target, people: Sequence[Person]) -> float | None:
    """World yaw of the side people approach, or None for a target approachable from any side."""
    if target.front_yaw is not None:
        return target.front_yaw
    if target.kind != "person":
        return None
    for person in people:
        if person.id == target.person_id:
            return person.yaw
    raise ValueError(f"target {target.id!r}: person {target.person_id!r} is not among the people given")


def region_no_one(target: Target, people: Sequence[Person]) -> _Regions:
    """A rectangle in front of the target, or the full ring round it when it has no front.

    Of `people` only the target person is looked at, for the way they face."""
    a = _ground(target)
    front = _front(target, people)
    if front is None:
        polygon = ring(a, _NO_ONE_NEAR_M, _NO_ONE_FAR_M)
    else:
        near, far = (_JOIN_NEAR_M, _JOIN_FAR_M) if target.kind == "person" else (_NO_ONE_NEAR_M, _NO_ONE_FAR_M)
        polygon = oriented_rectangle(a, front, near, far, _NO_ONE_WIDTH_M)
    return [_region(target, polygon, Heading(face_point=a))], []


def region_waiting_line(target: Target, members: Sequence[Person]) -> _Regions:
    """The half of the ring round the last person in the line that is not ahead of them, to be stood in facing
    the way the line faces; the corridor from that person to the target is forbidden."""
    if not members:
        raise NoRule(f"target {target.id!r}: a waiting line needs at least one member")
    a = _ground(target)
    line = sorted(((m.x, m.y) for m in members), key=lambda p: _dist(a, p))     # nearest the target first
    last = line[-1]
    before = line[-2] if len(line) > 1 else a
    if _dist(before, last) <= _EPS_M:
        raise NoRule(f"target {target.id!r}: the waiting line has no direction, "
                     f"its last member stands on {'the one before' if len(line) > 1 else 'the anchor'}")
    front = math.atan2(before[1] - last[1], before[0] - last[0])       # the way the line faces: from its last member on
    fx, fy = math.cos(front), math.sin(front)

    def not_ahead(x: float, y: float) -> bool:
        """Behind the last member or level with them: not on the target's side of the line across the line's
        direction through them."""
        return (x - last[0]) * fx + (y - last[1]) * fy <= _EPS_M

    # The first ray and the one opposite lie on that line, so the region ends on it exactly.
    (polygon,) = _ray_band(last, _JOIN_NEAR_M, _JOIN_FAR_M, not_ahead, turn=front + math.pi / 2)
    region = _region(target, polygon, Heading(yaw=front))
    polyline = line[::-1] + [a]
    forbidden = [ForbiddenRegion(kind="ahead_in_line", polygon=segment_rectangle(p, q, _LINE_CORRIDOR_WIDTH_M),
                                 target_id=target.id)
                 for p, q in zip(polyline, polyline[1:])
                 if _dist(p, q) > _EPS_M]     # a person target heading the line stands on the anchor
    return [region], forbidden


def _reaches(r_in: float, r_out: float) -> list[float]:
    """The distances at which a stretch from r_in to r_out is sampled, both ends included."""
    steps = max(1, round((r_out - r_in) / _BAND_RADIAL_STEP_M))
    return [r_in + (r_out - r_in) * k / steps for k in range(steps + 1)]


def _stretch(origin: _Point, direction: _Point, reaches: Sequence[float],
             ok: Callable[[float, float], bool]) -> _Stretch | None:
    """The points at the two ends, nearer one first, of the stretch of origin + r * direction on which ok
    holds, r being one of `reaches`; None when it holds nowhere, or on a stretch shallower than
    _BAND_MIN_DEPTH_M. ok must hold on at most one stretch of the line."""
    (ox, oy), (ux, uy) = origin, direction
    passing = [r for r in reaches if ok(ox + r * ux, oy + r * uy)]
    if not passing or passing[-1] - passing[0] < _BAND_MIN_DEPTH_M - _EPS_M:
        return None
    lo, hi = passing[0], passing[-1]
    return (ox + lo * ux, oy + lo * uy), (ox + hi * ux, oy + hi * uy)


def _bands(stretches: Sequence[_Stretch | None]) -> list[Polygon]:
    """The polygons of stretches that follow one another round a closed curve, None being an empty one: every
    run of two or more neighbouring stretches gives one polygon, the last stretch and the first being
    neighbours, so that a run may cross the start. A sequence without an empty stretch closes on itself. It is
    one polygon like the ring of sgi.polygons: round its outer edge, then back round its inner one, the two
    joined along the first stretch."""
    def outline(run: Sequence[_Stretch]) -> Polygon:
        """Out along the far ends of the stretches, and back along the near ones."""
        return [hi for _, hi in run] + [lo for lo, _ in reversed(run)]

    stretches = list(stretches)
    start = next((i for i, s in enumerate(stretches) if s is None), None)      # scan from an empty one: no run is split
    if start is None:
        return [outline(stretches + stretches[:1])]                            # right round, and on to the first
    polygons = []
    for filled, group in groupby(stretches[start:] + stretches[:start], key=lambda s: s is not None):
        run = list(group)
        if filled and len(run) >= 2:
            polygons.append(outline(run))
    return polygons


def _ray_band(center: _Point, r_in: float, r_out: float, ok: Callable[[float, float], bool],
              turn: float = 0.0) -> list[Polygon]:
    """The part of the ring r_in..r_out round center where ok(x, y) holds, as polygons.

    ok is sampled along _BAND_RAYS rays from center, the first at the angle `turn`, and must hold on at most
    one stretch of each. A ray whose stretch is shallower than _BAND_MIN_DEPTH_M is empty; every run of two
    or more neighbouring non-empty rays gives one polygon, the last ray and the first being neighbours.
    A band without an empty ray closes on itself. It is one polygon like the ring of sgi.polygons: round its
    outer edge, then back round its inner one, the two joined along the first ray."""
    radii = _reaches(r_in, r_out)
    angles = [turn + 2 * math.pi * i / _BAND_RAYS for i in range(_BAND_RAYS)]
    return _bands([_stretch(center, (math.cos(angle), math.sin(angle)), radii, ok) for angle in angles])


def _in_reach(people: Sequence[Person], x: float, y: float) -> bool:
    """Neither too near the nearest of the people nor too far from them."""
    nearest = min(_dist((p.x, p.y), (x, y)) for p in people)
    return _JOIN_NEAR_M - _EPS_M <= nearest <= _JOIN_FAR_M + _EPS_M


def region_conversation_group(target: Target, members: Sequence[Person]) -> _Regions:
    """The band round the group, outside its circle, that is neither too near its nearest member nor too far
    from them, to be stood in facing the target person; the group's circle is forbidden."""
    who = next((m for m in members if m.id == target.person_id), None)
    if who is None or len(members) < 2:
        raise NoRule(f"target {target.id!r}: a conversation group needs two or more members, "
                     f"one of them the target person {target.person_id!r}")
    c = (sum(m.x for m in members) / len(members), sum(m.y for m in members) / len(members))
    radius = max(_dist(c, (m.x, m.y)) for m in members)
    # The circle holds every member, so outside it no member comes nearer along a ray from its centre: one
    # stretch of the ray is in reach, and it ends no further beyond the circle than the far distance.
    polygons = _ray_band(c, radius, radius + _JOIN_FAR_M, lambda x, y: _in_reach(members, x, y))
    space = ForbiddenRegion(kind="group_space", polygon=disc(c, radius), target_id=target.id)
    return [_region(target, polygon, Heading(face_point=(who.x, who.y))) for polygon in polygons], [space]


def _space_between(points: Sequence[_Point]) -> tuple[Polygon, bool]:
    """The convex hull of the points, counter-clockwise, and whether it has an inside.

    A vertex at which the hull's boundary turns by less than _STRAIGHT_RAD is dropped, so that people in a
    straight row keep its two ends and no more; the turn is atan2(cross, dot) of the two edges that meet
    there, taken in 0..pi. The hull has an inside when its area is more than _EPS_M times its perimeter.
    Without one it comes back as what it then is: the two of its points furthest apart, a segment, or the one
    point they all are."""
    hull = convex_hull(points)

    def turn(k: int) -> float:
        (px, py), (qx, qy), (rx, ry) = hull[k - 1], hull[k], hull[(k + 1) % len(hull)]
        cross = (qx - px) * (ry - qy) - (qy - py) * (rx - qx)
        return math.atan2(abs(cross), (qx - px) * (rx - qx) + (qy - py) * (ry - qy))       # a hull turns one way: 0..pi

    ends = hull                                    # of the hull as convex_hull gave it: what a segment is taken from
    while len(hull) >= 3:
        straight = [k for k in range(len(hull)) if turn(k) < _STRAIGHT_RAD]
        if not straight:
            break
        hull = [v for k, v in enumerate(hull) if k != straight[0]]             # one at a time: its neighbours' turns change
    edges = list(zip(hull, hull[1:] + hull[:1]))
    area = sum(px * qy - qx * py for (px, py), (qx, qy) in edges) / 2
    if len(hull) >= 3 and area > _EPS_M * sum(_dist(p, q) for p, q in edges):
        return hull, True
    p, q = max(((p, q) for p in ends for q in ends), key=lambda pair: _dist(*pair))
    return ([p] if _dist(p, q) <= _EPS_M else [p, q]), False


def _walk(shape: Polygon, margin: float, also: Sequence[float] = ()) -> list[tuple[_Point, _Point]]:
    """The outline of a convex shape grown by `margin`, walked once counter-clockwise: each point of the walk
    with the unit normal that points out of the shape there.

    `shape` is a convex polygon given counter-clockwise, or two points, a segment, or one. The grown outline
    runs straight along every edge, where the points are no more than _WALK_STEP_M apart, and in an arc round
    every corner, where the normals are no more than _WALK_TURN_RAD apart. Each angle of `also` is the
    direction of one more normal at whichever corner's arc holds it, unless the arc has that normal already:
    no normal of the walk is there twice."""
    def arc(corner: _Point, first: float, turn: float) -> list[tuple[_Point, _Point]]:
        """Round a corner from the normal at the angle `first` on by `turn`, the last normal left out: the
        stretch that follows begins with it."""
        steps = max(1, math.ceil(turn / _WALK_TURN_RAD - _EPS_M))
        offs = [turn * k / steps for k in range(steps)]
        for off in ((angle - first) % (2 * math.pi) for angle in also):
            if off < turn - _STRAIGHT_RAD and all(abs(off - other) > _STRAIGHT_RAD for other in offs):
                offs.append(off)
        normals = [(math.cos(first + off), math.sin(first + off)) for off in sorted(offs)]
        return [((corner[0] + margin * nx, corner[1] + margin * ny), (nx, ny)) for nx, ny in normals]

    if len(shape) == 1:                            # a point: the circle round it
        return arc(shape[0], 0.0, 2 * math.pi)
    walk = []
    for k, (px, py) in enumerate(shape):           # the corner at each vertex, then the edge that leaves it
        (ox, oy), (qx, qy) = shape[k - 1], shape[(k + 1) % len(shape)]
        (ix, iy), (dx, dy) = (px - ox, py - oy), (qx - px, qy - py)            # the edge that arrives, and the one that leaves
        length = math.hypot(dx, dy)
        nx, ny = dy / length, -dx / length         # out of a shape that lies to the left of its edges
        turn = math.atan2(abs(ix * dy - iy * dx), ix * dx + iy * dy)           # in 0..pi; pi at the end of a segment
        walk += arc((px, py), math.atan2(ny, nx) - turn, turn)
        steps = max(1, math.ceil(length / _WALK_STEP_M - _EPS_M))
        walk += [((px + dx * j / steps + margin * nx, py + dy * j / steps + margin * ny), (nx, ny)) for j in range(steps)]
    return walk


def region_attending(target: Target, members: Sequence[Person], kind: str) -> _Regions:
    """For people who attend to the target together, watching it or waiting at it: the band behind and beside
    them that is neither too near their nearest member nor too far from them, keeps clear of the space between
    them and the target, and is no nearer the target than the nearest of them; to be stood in facing the
    target. The space between is forbidden, as `kind`. The target is not one of the members.

    The space between is the convex hull of the members' positions and the target's outline: its footprint,
    or without one a disc of the near distance round the anchor's ground point, so that a person or a small
    thing has room of its own. Nearer the target is measured along the target's front where it has one, from
    its anchor's ground point, and else as plain distance from that point. A front that is given and is no
    finite angle is no direction to measure along: there is no rule then.

    The region is built along the normals of that space grown by the clearance (see _walk and _bands). Where
    the target has a front, the normals across it are among them. They lie on the line level with the nearest
    member only at a corner of the space that is itself level with that member, as the nearest member's own
    place is when it is a corner of the space: there the region ends exactly on the line. Where the line cuts
    the arc of another corner or an edge of the space, and where nearer is plain distance, the region ends up
    to one sample short of the line or circle, and never past it.

    Each normal is taken to have one stretch on which the conditions hold. Out along a normal of a convex set
    the distance to every point of the set grows, so that is true of the distance to each member, and of which
    side of the level line a point is on. Of the plain distance to the anchor's ground point it is true when
    that point lies in the space between or within the clearance of it, which it does for a target without a
    footprint, whose disc is round it, and for a footprint that holds its own anchor. An anchor further
    outside the space than that is not provided for: the region could then reach nearer it than the nearest
    member."""
    if not members or any(m.id == target.person_id for m in members):
        raise NoRule(f"target {target.id!r}: the people attending to a target are one or more, "
                     f"and the target is none of them")
    a = _ground(target)
    outline = target.footprint or disc(a, _JOIN_NEAR_M)        # the disc of sgi.polygons at its own resolution: 48 vertices
    _numbers(target, *a, *(c for vertex in outline for c in vertex), *(c for m in members for c in (m.x, m.y)))
    space, solid = _space_between([*((m.x, m.y) for m in members), *outline])
    if target.front_yaw is not None:
        if not math.isfinite(target.front_yaw):
            raise NoRule(f"target {target.id!r}: its front_yaw is no direction, so nothing is nearer the target "
                         f"along it or further")
        ux, uy = math.cos(target.front_yaw), math.sin(target.front_yaw)
        level = min((m.x - a[0]) * ux + (m.y - a[1]) * uy for m in members)
        across = (target.front_yaw + math.pi / 2, target.front_yaw - math.pi / 2)

        def no_nearer(x: float, y: float) -> bool:
            return (x - a[0]) * ux + (y - a[1]) * uy >= level - _EPS_M
    else:
        least = min(_dist(a, (m.x, m.y)) for m in members)
        across = ()

        def no_nearer(x: float, y: float) -> bool:
            return _dist(a, (x, y)) >= least - _EPS_M

    def ok(x: float, y: float) -> bool:
        return _in_reach(members, x, y) and no_nearer(x, y)

    # A point t out along a normal of the grown space is the clearance and t outside the space, so it is clear
    # of it wherever it is taken, and within the far distance of a member up to t = far - clearance at most.
    # One stretch of every normal: see the docstring, which says of which anchors that holds.
    reaches = _reaches(0.0, _JOIN_FAR_M - _VIEW_CLEAR_M)
    polygons = _bands([_stretch(q, n, reaches, ok) for q, n in _walk(space, _VIEW_CLEAR_M, across)])
    forbidden = [ForbiddenRegion(kind=kind, polygon=space, target_id=target.id)] if solid else []
    return [_region(target, polygon, Heading(face_point=a)) for polygon in polygons], forbidden


def region_group_photo(target: Target, members: Sequence[Person]) -> _Regions:
    """Round a picture being taken, with the one taking it or one of those posing as the target: the half of
    the ring round the picture-taker that is not ahead of them, to be stood in facing the target person. The
    strip from the camera through those posing and on behind them is forbidden.

    The picture-taker is the one member who has every other within _TAKER_SEES_DEG of the way they face and
    whom every other has within it of the way they face. Ahead of them is towards the centroid of those posing;
    the way they face themselves is not read beyond that test. The strip is the convex hull of discs of
    _FRAME_MARGIN_M round the picture-taker, round every one posing and round the point the far distance behind
    each as the picture-taker sees them."""
    who = next((m for m in members if m.id == target.person_id), None)
    if who is None or len(members) < 2:
        raise NoRule(f"target {target.id!r}: a picture being taken needs two or more members, "
                     f"one of them the target person {target.person_id!r}")
    _numbers(target, *(value for m in members for value in (m.x, m.y, m.yaw)))

    def sees(one: Person, other: Person) -> bool:
        """Whether the other is within the angle of the way the one faces; nobody sees who stands where they do."""
        if _dist((one.x, one.y), (other.x, other.y)) <= _EPS_M:
            return False
        off = wrap_angle(math.atan2(other.y - one.y, other.x - one.x) - one.yaw)
        return abs(math.degrees(off)) <= _TAKER_SEES_DEG

    takers = [m for m in members if all(sees(m, other) and sees(other, m) for other in members if other is not m)]
    if len(takers) != 1:
        raise NoRule(f"target {target.id!r}: {len(takers)} of the members face all the others and are faced by "
                     f"them, so no one of them is the picture-taker")
    (taker,) = takers
    posing = [(m.x, m.y) for m in members if m is not taker]
    at = (taker.x, taker.y)
    middle = (sum(x for x, _ in posing) / len(posing), sum(y for _, y in posing) / len(posing))
    if _dist(at, middle) <= _EPS_M:                # a guard against dividing by zero below, which no layout reaches
        raise NoRule(f"target {target.id!r}: the picture-taker stands in the middle of those posing, "
                     f"so nothing is ahead of them")
    wx, wy = (middle[0] - at[0]) / _dist(at, middle), (middle[1] - at[1]) / _dist(at, middle)

    def not_ahead(x: float, y: float) -> bool:
        """Behind the picture-taker or level with them: not on the side of those posing."""
        return (x - at[0]) * wx + (y - at[1]) * wy <= _EPS_M

    # The first ray and the one opposite lie on the line across that way through the picture-taker, so the
    # region ends on it exactly, as the half ring behind a waiting line does.
    (polygon,) = _ray_band(at, _JOIN_NEAR_M, _JOIN_FAR_M, not_ahead, turn=math.atan2(wy, wx) + math.pi / 2)
    behind = [(x + _JOIN_FAR_M * (x - at[0]) / _dist(at, (x, y)), y + _JOIN_FAR_M * (y - at[1]) / _dist(at, (x, y)))
              for x, y in posing]                  # nobody posing stands on the picture-taker, who sees each of them
    strip = convex_hull([v for c in (at, *posing, *behind) for v in disc(c, _FRAME_MARGIN_M)])     # discs of 48 vertices
    picture = ForbiddenRegion(kind="photo_frame", polygon=strip, target_id=target.id)
    return [_region(target, polygon, Heading(face_point=(who.x, who.y)))], [picture]


def region_machine_use(target: Target, members: Sequence[Person], near: float = _MACHINE_NEAR_M) -> _Regions:
    """For one person using the target, a machine with a front, with nobody else waiting: the part of the ring
    `near` to _MACHINE_FAR_M round the user that is not ahead of them and lies within _MACHINE_BEHIND_DEG of
    straight behind them, to be stood in facing the machine. Ahead is the way the user faces, their `yaw`, and
    behind is opposite it.

    Forbidden: within `near` of the user, over their shoulder, as `over_shoulder`; and the strip before the
    machine beside the user, where a second person would stand at it with them, as `beside_user`. The strip runs
    from the machine's face, the line through the anchor across its front, out to _MACHINE_STRIP_M past the
    user, and across the machine's footprint and _MACHINE_STRIP_M past each side of it.

    `near` is the rule's near limit, _MACHINE_NEAR_M; _MACHINE_NEAR_SENSITIVITY_M gives the variant reported
    beside it. The rule needs exactly one member, the user, who is not the target, and a target whose front is
    a finite angle."""
    if len(members) != 1 or members[0].id == target.person_id:
        raise NoRule(f"target {target.id!r}: a machine in use needs exactly one member, the user, who is not the "
                     f"target; {len(members)} given")
    if target.front_yaw is None or not math.isfinite(target.front_yaw):
        raise NoRule(f"target {target.id!r}: a machine in use needs a front, and its front_yaw is no direction")
    (user,) = members
    a = _ground(target)
    _numbers(target, *a, user.x, user.y, user.yaw, *(c for vertex in target.footprint or [] for c in vertex))
    at = (user.x, user.y)
    behind = user.yaw + math.pi
    fx, fy = math.cos(user.yaw), math.sin(user.yaw)
    widest = math.radians(_MACHINE_BEHIND_DEG)

    def ok(x: float, y: float) -> bool:
        """Not ahead of the user, and within the angle of straight behind them."""
        dx, dy = x - at[0], y - at[1]
        return dx * fx + dy * fy <= _EPS_M and abs(wrap_angle(math.atan2(dy, dx) - behind)) <= widest + _STRAIGHT_RAD

    # The first ray lies on one side of the angle, and the ray as far round the other way on the other: with rays
    # every 2 degrees and 60 degrees each way, the region ends on both sides exactly.
    (polygon,) = _ray_band(at, near, _MACHINE_FAR_M, ok, turn=behind - widest)
    ux, uy = math.cos(target.front_yaw), math.sin(target.front_yaw)
    out = (user.x - a[0]) * ux + (user.y - a[1]) * uy                  # how far out of the machine's face the user stands
    across = [(x - a[0]) * -uy + (y - a[1]) * ux for x, y in target.footprint or [a]]
    left, right = max(across) + _MACHINE_STRIP_M, min(across) - _MACHINE_STRIP_M
    middle = ((left + right) / 2 * -uy + a[0], (left + right) / 2 * ux + a[1])
    strip = oriented_rectangle(middle, target.front_yaw, 0.0, out + _MACHINE_STRIP_M, left - right)
    forbidden = [ForbiddenRegion(kind="over_shoulder", polygon=disc(at, near), target_id=target.id),
                 ForbiddenRegion(kind="beside_user", polygon=strip, target_id=target.id)]
    return [_region(target, polygon, Heading(face_point=a))], forbidden


# The situations whose rule needs the group at the target, each with its rule as (target, members).
_GROUP_RULES: dict[str, Callable[[Target, Sequence[Person]], _Regions]] = {
    "waiting_line": region_waiting_line,
    "conversation_group": region_conversation_group,
    "audience": lambda target, members: region_attending(target, members, "audience_view"),
    "lift_wait": lambda target, members: region_attending(target, members, "ahead_in_line"),
    "group_photo": region_group_photo,
    "machine_use": region_machine_use,
}


def build_regions(situation: str, target: Target, people: Sequence[Person],
                  group_id: str | None) -> _Regions | None:
    """The rule for the situation applied to the target, or None when there is no rule, its input is missing
    or the rule's own conditions are not met (NoRule). Any other error of a rule passes through."""
    if situation == "no_one":
        return region_no_one(target, people)
    if situation not in _GROUP_RULES or group_id is None:
        return None
    members = [p for p in people if p.group == group_id]
    try:
        return _GROUP_RULES[situation](target, members)
    except NoRule:
        return None
