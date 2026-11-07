"""Placeholder layouts (spec section 12): a furnished room, people, a robot pose, a target and two requests for it.

A recipe lays its scene out in a canonical frame: the target's ground point at the origin, its front along
+x; where a picture is being taken, the middle of the row posing is there instead, whoever the target is.
transform_layout then moves a finished layout rigidly. A layout holds no regions; the generator fills
them from sgi.gt. `rng` is a numpy.random.Generator and the only source of randomness.

A target is built from several primitives, so that it looks like what its label says, and stands in a room
whose floor is the scene's bounds. The room is furnished along its walls only, and the walls keep their
margin from the robot, the people and the target, so no furnishing can stand between the camera and any of
them. Whatever stands in the room has its ground rectangle among the obstacles.

The metres and degrees here are layout constants, not thresholds: they were tuned so that almost every draw
passes the scene check. The camera is mounted 1.5 m up and pitched 10 degrees down, so its image ends
1.5 + 0.24 d metres above the floor d metres ahead: no target is higher than 2.25 m, and the robot keeps far
enough from what it looks at for that to fit. The lift is the one exception: its shaft reaches the ceiling and
the image cuts it off. What the image has to hold of it is the doorway and the number lit over it, up to
2.4 m, and the robot stands 4 m or more from the doors.

The furnishings are laid out for the picture the robot takes (see _furnish). None of them is partly hidden
behind what the scene is about, a place, a table or the person asked for: a window between the posts of a
desk reads as part of the desk. A poster or the second stall, which carry lettering, is whole in the picture,
turned towards the camera and hidden by nothing, or it is not in the picture at all: a name that is cut off
or half hidden reads as another name. Whatever else is too small to be known by half of it is likewise whole
in the picture, with nobody before it, or not in it.

A layout also says what its primitives are: its `items` name each run of them, a stall, a bench, a window,
with the frame it stands in and what its builder was called with, so that another generator can build the
same thing in the same spot by other means (realistic-scenes spec 5.1). Naming them draws no random number.
An item's rectangle is the ground its own primitives stand on, which is not always one obstacle: a deep
place is two items on one obstacle, and a parcel table with its crate is one item on two.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from sgi.geometry import RobotPose, wrap_angle
from sgi.polygons import Polygon, distance_to_polygon, oriented_rectangle
from sgi.scene import EpisodeGT, EpisodeModel, Person, PlanStep, PolygonModel, Rationale, SceneGT, Target
from sgi.synth.render import Box, Color, Cylinder, Figure, Room, Solid, Sphere, appearance

# The camera on the robot: 1.5 m up, so that it looks over a counter, and pitched little, so that a standing
# figure 2.5 to 7 m away is in the image from head to foot and less than half of the image is floor.
# sgi.synth.generate mounts it as given here, and the rules below that depend on what the image shows read
# it from here.
CAMERA_MOUNT = {"forward_m": 0.2, "height_m": 1.5, "pitch_down_rad": math.radians(10.0)}
CAMERA_LENS = {"fx": 554.3, "fy": 554.3, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
_VIEW_TAN = CAMERA_LENS["cx"] / CAMERA_LENS["fx"]  # of the angle between the optical axis and the side of the image
# The image's upper edge rises this many metres per metre ahead of the camera, and its lower edge drops this many.
_VIEW_RISE = math.tan(math.atan(CAMERA_LENS["cy"] / CAMERA_LENS["fy"]) - CAMERA_MOUNT["pitch_down_rad"])
_VIEW_DROP = math.tan(math.atan((CAMERA_LENS["height"] - CAMERA_LENS["cy"]) / CAMERA_LENS["fy"])
                      + CAMERA_MOUNT["pitch_down_rad"])

_TARGET_ID = "t0"
_GROUP_ID = "g0"                                   # the group at the target
_SECOND_GROUP_ID = "g1"
_BOUNDS_MARGIN_M = 3.0      # holds every reference region; a map no larger keeps the scene check near one second
_PLACE_WIDTH_M = 1.6
_PLACE_FRONT_M = 0.8                               # how deep the part of a place is that people are served at
_PLACE_DEPTH_M = {"place": _PLACE_FRONT_M, "deep_place": 2.4}
_PLACE_ANCHOR_Z_M = 1.0                            # on the front of a counter, a desk or a cabinet: all reach 1.1 m
_PLACE_TOP_M = 2.25                                # the top of a place: of its back wall and of its sign
_SIGN_Z_M = 1.9                                    # the underside of a place's sign, above every head
_SIGN_HEIGHT_M = _PLACE_TOP_M - _SIGN_Z_M
_PERSON_ANCHOR_Z_M = 1.1                           # about chest height
_PARCEL_ANCHOR = (0.0, 0.0, 0.825)                 # the centre of the parcel: 0.25 m high, on a table 0.7 m high
_PARCEL_RADIUS_M = 0.3                             # round the anchor: the parcel, and none of the things beside it
_JITTER_M = 0.05                                   # how far a person stands from the spot asked for, each way
_JITTER_RAD = 0.15                                 # and how far they turn from the yaw asked for
_TARGET_PERSON_COLOR = (210, 40, 40)
_OTHER_COLORS = ((60, 90, 200), (60, 160, 90), (230, 200, 60), (150, 80, 180), (240, 140, 40), (40, 170, 170))
_NO_ONE_KINDS = ("place", "deep_place", "object", "person", "lift", "screen")
_VIEW_YAW_JITTER_RAD = math.radians(5.0)           # how far a robot facing one target looks past it
_LINE_FIRST_M = 0.8                                # from the anchor to the first person in line
_LINE_SPACING_M = 0.6
_LINE_REGION_M = 1.4                               # how far behind the last person the view reaches, to frame the line
_LINE_MAX_PEOPLE = 6
_LINE_VIEW_MIN_M = 4.4                             # the robot stands this far to the side of a line, or further
_MAX_RANGE_M = 5.9                                 # the robot stays this near its target while the image holds the place
_LINE_EDGE_M = (0.05, 0.1)                         # inside the image's sides: a place's near corner, a region's middle
_GROUP_NEIGHBOUR_M = 1.2                           # between neighbours round a conversation group
# From these bearings, off the first member's facing, any two members are more than 5 degrees apart in the
# image. At 90 degrees a group of three has two in line. The reference region is a band round the group on every
# side, so the bearing no longer puts the plain stop on the wrong side of it. The plain stop (before the side of
# the target person that is seen where a plane fits it and a place there is clear, else on the robot's line of
# sight, slid by up to 0.5 m) still mostly lands among or beside the members and fails, but not in every
# picture: one group of three seen at 49 degrees has it inside the band.
_GROUP_BEARING_DEG = {2: 90.0, 3: 60.0, 4: 72.5}
# People watching a screen. Where each stands: a row for each tuple, the row nearest the screen first, and in it
# the places across, counted in neighbours' distances from the middle of the screen towards the robot's side.
# Three stand in one row. Of four and five the back row stands over to the side away from the robot, so that
# from the bearing below, off straight behind them, the robot sees each of them past the others and the middle
# of the screen past all of them. The places and the bearings are chosen for the picture alone: no body hides
# part of another in it. The reference region is one piece round each of these groups wherever they stand.
_AUDIENCE_PLACES = {3: ((-1.0, 0.0, 1.0),), 4: ((-0.5, 0.5), (-1.0, 0.0)), 5: ((-1.0, 0.0, 1.0), (-1.25, -0.25))}
_AUDIENCE_BEARING_DEG = {3: 50.0, 4: 52.0, 5: 55.0}
_AUDIENCE_NEIGHBOUR_M = 0.9                        # between neighbours in a row
_AUDIENCE_ROW_GAP_M = 0.9                          # from a row to the one behind it
_AUDIENCE_ROWS_M = (2.05, 3.15)                    # from the screen: no row stands nearer than the first, none further
# People waiting at a lift, not in single file. Where each stands: how far behind the one nearest the doors, and
# how far across from the middle of the doors towards the robot's side. Two stand side by side, one a step
# back. Four stand as people do who gather at a lift: three abreast in a shallow arc before the doors and the
# fourth behind a gap between two of them. The robot sees the four from nearer straight behind than the two, so
# that in its picture they are four people side by side before the doors, the middle of the doors between two of
# them, and not one person after another leading away from the doors, which is how a waiting line looks.
_LIFT_PLACES = {2: ((0.0, 0.35), (0.45, -0.45)), 4: ((0.0, -0.7), (0.0, 1.2), (0.35, 0.2), (1.35, -0.35))}
_LIFT_BEARING_DEG = {2: 45.0, 4: 30.0}
_LIFT_NEAREST_M = (1.3, 1.7)                       # from the doors to the nearest of those waiting
_VIEW_INSIDE_RAD = math.atan(_VIEW_TAN) - math.radians(4.0)      # what a picture has to hold whole is no further off the
#                                                    robot's heading: 4 degrees inside the image's side, room for a body's width
_ATTEND_VIEW_M = (4.0, 4.4)                        # from the middle of people watching or waiting to the robot
_ATTEND_FLOOR_M = 1.0                              # how far behind and beside those people the view reaches, on the side
#                                                    away from the robot: the floor there is where the region lies too
# A picture being taken: a row of people posing, and across from its middle someone behind a camera on a tripod.
_PHOTO_NEIGHBOUR_M = 0.6                           # between neighbours in the row
_PHOTO_TAKER_M = (2.5, 3.5)                        # from the row to whoever takes the picture
_PHOTO_TRIPOD_M = 0.5                              # from whoever takes the picture to the tripod before them
# Where the robot looks from, by who the target is. For the picture-taker it stands behind the row and to one
# side, and looks past the row at the camera and the one behind it: from behind the picture-taker the plain stop,
# which is the frontal one, before the side of a person that the robot sees, would be in the reference region
# behind them. For one of those posing it stands behind the picture-taker and to one side, and sees those posing
# from the front. The angle is the one at the far end of that view, at the picture-taker's place or at the middle
# of the row, between the line from the row to the camera and the way to the robot. It is chosen for the picture:
# those posing side by side and the camera on its tripod clear of them. Seen from behind the picture-taker, that
# person stands beside the camera; seen from behind the row, the camera stands before part of them and most of
# them shows past it.
_PHOTO_VIEW_DEG = {"taker": 20.0, "posing": 27.0}
_PHOTO_BEHIND_M = {"taker": (2.6, 3.0), "posing": (2.0, 2.4)}     # how far behind the row, or the picture-taker, it stands
_PHOTO_FLOOR_M = 1.0                               # how far beside and behind the picture-taker the view reaches where
#                                                    that fits: the region lies on the floor there
# A free spot, for someone who is no part of the situation: in the robot's view, outside every reference
# region and too far from the target to be a relevant person of a no_one scene.
_SPOT_RANGE_M = (3.0, 12.0)                        # from the robot
_SPOT_HALF_ANGLE_RAD = math.radians(22.0)          # to each side of the robot's heading
_SPOT_TARGET_GAP_M = 4.0
_SPOT_PERSON_GAP_M = 2.5
_SPOT_MAX_DRAWS = 200
_PAIR_HALF_GAP_M = 0.55                            # from the second group's spot to each of its two members
# In view also means not hidden: the camera, 1.5 m up, sees nothing behind a person and little behind a place.
# So nothing the scene is about and nobody else may be this near the ground line from the robot to someone who
# is added. A figure's radius is 0.22 m, and the camera, 0.2 m ahead of the robot's centre, looks along a line
# up to 0.08 m from that one. The gap is no wider than that asks, because behind a waiting line little is in
# view: at 0.5 m, a third of the lines of three with a second group have no spot left for their bystander.
_SIGHT_GAP_M = 0.3
_SIGHT_STEP_M = 0.05                               # between the points at which a line of sight is tested
_EXTRAS_MAX_TRIES = 20                             # at placing a layout's second group and bystanders together

# Furnishings stand against the walls. With the margin the walls keep, none reaches nearer than 2 m to the
# robot, and the gap below keeps them further than any reference region reaches from a person or a target,
# with room for the robot to stand in that region.
_FURNISH_GAP_M = 2.6                               # from a furnishing to every person and to what the scene is about
_FURNISH_APART_M = 0.3                             # between two furnishings
_WALL_ENDS_M = 0.9                                 # no furnishing begins this near a corner of the room
_WALL_STEP_M = 0.25                                # along a wall, from a place that is taken to the next one tried
_FURNISH_SPACING_M = (0.4, 1.2)                    # along a wall, from one furnishing to the next
_ROOM_HEIGHT_M = 3.0
_BENCH_LENGTH_M = 1.6                              # of every bench along a wall
# What the picture has to show of something lettered, and what counts as hiding it.
_IMAGE_EDGE_M = 0.15                               # what is whole in the image keeps this far inside its sides, and
#                                                    what is not in it this far outside them
_IMAGE_TOP_M = 0.05                                # and this far from its upper and its lower edge
_NEAR_CAMERA_M = 0.3                               # nothing nearer than this ahead of the camera counts as in the image
_READABLE_COS = math.cos(math.radians(62.0))       # lettering is read from no further off the square view of it
_HIDING_RAD = math.radians(2.5)                    # two things this near in bearing overlap in the image
_BROAD_SHARE = 0.5                                 # of a broad thing's width, the most that one person may stand before
_TALLEST_M = 1.85                                  # above every head: the tallest figure is 1.84 m

_WOOD, _DARK_WOOD, _STEEL, _CREAM = (156, 112, 74), (92, 66, 48), (168, 172, 178), (240, 234, 220)
_CHARCOAL, _PAPER, _GLASS, _SCREEN = (54, 56, 60), (244, 240, 228), (186, 216, 232), (206, 230, 244)
_LEAVES = ((52, 120, 62), (66, 140, 70), (44, 104, 56), (80, 150, 76))
_ROOMS = (   # the colours of a room, as fields of Room; the first is the renderer's own warm plaster
    {},
    {"wall": (226, 230, 232), "dado": (178, 190, 198), "floor": (190, 192, 192), "joint": (122, 126, 130)},
    {"wall": (238, 228, 208), "dado": (198, 174, 142), "floor": (206, 188, 164), "joint": (134, 118, 100)},
)
_PARCEL_LABEL = "blue parcel"
_PERSON_LABEL = "person in red"
# Grounded request, high-level request, and the context that goes with the high-level one.
_PARCEL_REQUESTS = ("go to the blue parcel", "bring me my parcel", "My parcel is the blue box on the table.")
_PERSON_REQUESTS = ("go to the person in red", "give this to Sam", "Sam is the person wearing red.")
_SCREEN_LABEL = "big screen"
_SCREEN_REQUESTS = ("go to the big screen", "note down the results for me", None)     # the slide on it is headed RESULTS
_LIFT_LABEL = "lift"
_LIFT_FLOOR_LIT = "3"                              # the floor the lit number over the doors shows: where the scene is
_LIFT_REQUESTS = ("go to the lift", "take this up to the fifth floor", None)          # another floor than that one
# The two held-out situations have no entry: their episodes carry no rationale, and none is ever a worked example.
_RATIONALE = {   # situation: (people, expectations, placement)
    "no_one": (
        "Nobody is waiting at the target or standing close to it.",
        "Nothing constrains an arrival except using the target from the side it is meant to be used from.",
        "Stand about a metre from the target on that side, facing it."),
    "waiting_line": (
        "People stand one behind another leading up to the target, each waiting for a turn there.",
        "Someone arriving is expected to join at the back and not to step in ahead of anyone.",
        "Stand just behind the last person, at arm's length or a little more, facing the way the line faces."),
    "conversation_group": (
        "The target person stands with others in a small circle, facing them in conversation.",
        "Someone arriving should come up to the group without entering the circle or crowding the others.",
        "Stand just outside the group, at arm's length or a little more from the nearest member, facing the "
        "person the robot came for."),
    "audience": (
        "Several people stand together facing the target and watching it.",
        "Someone arriving is expected to keep out of the space between them and what they are watching, and "
        "not to crowd them.",
        "Stand at the edge of the group, behind or beside the people, at arm's length or a little more from the "
        "nearest of them, facing the target."),
}

_Requests = tuple[str, str, str | None]
_Parts = list[Solid]


@dataclass(frozen=True)
class Item:
    """What a run of a layout's primitives is, and where it stands."""
    kind: str                                      # "stall" | "ticket_counter" | "desk" | "kiosk" | "back_room" |
    #                                                "parcel_table" | a name in _FURNISHINGS | "shop" | "screen" |
    #                                                "lift" | "tripod_camera"
    x: float                                       # its own frame in the world: for a place, the ground point under the
    y: float                                       # anchor with +x out of its front; for a furnishing, its point on the
    yaw: float                                     # wall with +x into the room. The yaw is wrapped
    footprint: Polygon                             # its ground rectangle, square to its frame: every primitive of it
    #                                                stands on it. A furnishing's is its obstacle. A place's is the
    #                                                target's footprint; of a deep place the front has the 0.8 m of
    #                                                that which people are served at and the back room the rest. A
    #                                                parcel table's holds the table, the crate beside it and the free
    #                                                floor between them. So not every footprint is an obstacle: what
    #                                                blocks the floor under `boxes` is Layout.footprints
    options: Mapping[str, Any]                     # what its builder was called with; a place's has its label as well
    source: str                                    # "boxes" or "props": the Layout tuple holding its primitives
    start: int                                     # they are getattr(layout, source)[start:stop]
    stop: int


@dataclass(frozen=True)
class Layout:
    name: str
    boxes: tuple[Solid, ...]                       # what the scene is about: the target's parts and what stands with it
    figures: tuple[Figure, ...]                    # figures[i] draws scene_gt.people[i]
    robot: RobotPose
    scene_gt: SceneGT                              # free space as bounds and obstacles, without an occupancy image
    episodes: tuple[EpisodeModel, ...]             # both region lists None
    room: Room | None = None                       # its floor is scene_gt.bounds
    props: tuple[Solid, ...] = ()                  # the furnishings along the walls
    footprints: tuple[Polygon, ...] = ()           # the ground rectangles of `boxes`; the first obstacles of scene_gt
    items: tuple[Item, ...] = ()                   # what `boxes` and then `props` are, run by run: every primitive of
    #                                                both is in one item's range; none for a layout built by hand


def _sign(rng) -> float:
    return 1.0 if rng.integers(2) else -1.0


def _moved(part: Solid, c: float, s: float, tx: float, ty: float, yaw: float) -> Solid:
    """The part turned by yaw about the origin, of which c and s are the cosine and the sine, then shifted."""
    if isinstance(part, Box):
        return replace(part, cx=c * part.cx - s * part.cy + tx, cy=s * part.cx + c * part.cy + ty,
                       yaw=wrap_angle(part.yaw + yaw))
    return replace(part, x=c * part.x - s * part.y + tx, y=s * part.x + c * part.y + ty)


def _placed(parts: Sequence[Solid], x: float, y: float, yaw: float) -> _Parts:
    """Parts laid out round the origin with their front along +x, turned by yaw and moved to (x, y)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [_moved(part, c, s, x, y, yaw) for part in parts]


def _extent(parts: Sequence[Solid]) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of the ground rectangle, square to the axes, that holds every part."""
    xs, ys = [], []
    for part in parts:
        if isinstance(part, Box):
            c, s = math.cos(part.yaw), math.sin(part.yaw)
            for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                xs.append(part.cx + c * i * part.sx / 2 - s * j * part.sy / 2)
                ys.append(part.cy + s * i * part.sx / 2 + c * j * part.sy / 2)
        else:
            reach = part.radius if isinstance(part, Sphere) else max(part.radius, part.top_radius or 0.0)
            xs += [part.x - reach, part.x + reach]
            ys += [part.y - reach, part.y + reach]
    return min(xs), min(ys), max(xs), max(ys)


def _rectangle(box: Box) -> Polygon:
    """The ground rectangle under a box."""
    return oriented_rectangle((box.cx, box.cy), box.yaw, -box.sx / 2, box.sx / 2, box.sy)


def _apart(a: Polygon, b: Polygon) -> float:
    """The gap between two rectangles; 0.0 when a corner of one lies in the other."""
    return min(*(distance_to_polygon(corner, b) for corner in a), *(distance_to_polygon(corner, a) for corner in b))


def _heights(parts: Sequence[Solid]) -> tuple[float, float]:
    """The height of the lowest underside and of the highest top among the parts."""
    low, high = [], []
    for part in parts:
        if isinstance(part, Sphere):
            low.append(part.z - part.radius)
            high.append(part.z + part.radius)
        else:
            low.append(part.z0)
            high.append(part.z0 + (part.sz if isinstance(part, Box) else part.height))
    return min(low), max(high)


class _View:
    """What the robot's camera shows of things that stand on the ground, from their ground polygons and heights.

    A span is how the camera sees a polygon: its least and greatest bearing off the heading and its least
    and greatest distance, or None when a corner of it is not ahead of the camera."""

    def __init__(self, robot: RobotPose):
        self.cos, self.sin = math.cos(robot.yaw), math.sin(robot.yaw)
        self.x = robot.x + CAMERA_MOUNT["forward_m"] * self.cos
        self.y = robot.y + CAMERA_MOUNT["forward_m"] * self.sin

    def _ahead_left(self, point) -> tuple[float, float]:
        dx, dy = point[0] - self.x, point[1] - self.y
        return dx * self.cos + dy * self.sin, dy * self.cos - dx * self.sin

    @staticmethod
    def half_width(ahead: float, z: float) -> float:
        """How far to each side of the heading the image reaches, `ahead` metres before the camera and z metres up."""
        pitch = CAMERA_MOUNT["pitch_down_rad"]
        return _VIEW_TAN * (ahead * math.cos(pitch) + (CAMERA_MOUNT["height_m"] - z) * math.sin(pitch))

    def span(self, polygon: Polygon) -> tuple[float, float, float, float] | None:
        bearings, distances = [], []
        for ahead, left in map(self._ahead_left, polygon):
            if ahead <= _NEAR_CAMERA_M:
                return None
            bearings.append(math.atan2(left, ahead))
            distances.append(math.hypot(ahead, left))
        return min(bearings), max(bearings), min(distances), max(distances)

    def holds(self, polygon: Polygon, low: float, high: float) -> bool:
        """Whether the image holds all of what stands on the polygon from `low` to `high`, with a margin."""
        eye = CAMERA_MOUNT["height_m"]
        for ahead, left in map(self._ahead_left, polygon):
            if ahead <= _NEAR_CAMERA_M or not (eye - _VIEW_DROP * ahead + _IMAGE_TOP_M <= low
                                               and high <= eye + _VIEW_RISE * ahead - _IMAGE_TOP_M):
                return False
            if abs(left) > min(self.half_width(ahead, low), self.half_width(ahead, high)) - _IMAGE_EDGE_M:
                return False
        return True

    def misses(self, polygon: Polygon, low: float, high: float) -> bool:
        """Whether nothing of what stands on the polygon is in the image, with a margin."""
        sides = set()
        for ahead, left in map(self._ahead_left, polygon):
            if ahead <= _NEAR_CAMERA_M:
                continue
            if abs(left) < max(self.half_width(ahead, low), self.half_width(ahead, high)) + _IMAGE_EDGE_M:
                return False
            sides.add(left > 0)
        return len(sides) < 2                        # with corners beyond both sides, its middle is in the image

    def faces(self, polygon: Polygon, normal: float) -> bool:
        """Whether what hangs or stands on the polygon, turned to the yaw `normal`, can be read from the camera."""
        x, y = (sum(c) / len(polygon) for c in zip(*polygon))
        dx, dy = self.x - x, self.y - y
        return dx * math.cos(normal) + dy * math.sin(normal) >= _READABLE_COS * math.hypot(dx, dy)

    @staticmethod
    def hides(front: tuple, top: float, behind: tuple, bottom: float) -> bool:
        """Whether something with the span `front`, reaching up to `top`, covers part of what has the span
        `behind` and begins at `bottom`: it is nearer, it overlaps it in bearing, and the camera does not
        look over it."""
        if front[0] > behind[1] + _HIDING_RAD or behind[0] > front[1] + _HIDING_RAD or front[2] >= behind[3]:
            return False
        eye = CAMERA_MOUNT["height_m"]
        over = max((top - eye) / front[2], (top - eye) / front[3])       # the slope of the highest ray it blocks
        return over > (bottom - eye) / (behind[2] if bottom < eye else behind[3])


# --- What a place is built from. Each lies in x from -0.8 to 0 and y within 0.8 of 0: its front on x = 0. ---

_POST = (78, 82, 90)                               # the frame that carries a place's sign


def _place_sign(text: str, color: Color, width: float = _PLACE_WIDTH_M, front_posts: bool = True,
                rear_posts: bool = False) -> _Parts:
    """The sign over a place: a box of boards 0.35 m high over its first 0.72 m, lettered on the front and
    on both sides, so that the name reads from the front and from either side, with a roof on it.

    The boards along the sides end flush with the board across the front, which then hides no letter of
    theirs from someone looking along the front. `front_posts` puts a post under each front corner and
    `rear_posts` one under each far corner, where no wall carries the sign. A post stands just inside the
    corner, so that below the top of a counter it is inside the counter and shares no face with it."""
    lettering = dict(z0=_SIGN_Z_M, color=color, label=text, label_color=_CREAM, label_height=0.62, label_width=0.86)
    reach = width / 2 - 0.015                        # from the middle of the place to the middle of a side board
    shade = tuple(int(0.7 * channel) for channel in color)
    parts: _Parts = [Box(-0.035, 0.0, 0.0, 0.07, width, _SIGN_HEIGHT_M, **lettering),
                     Box(-0.395, reach, 0.0, 0.65, 0.03, _SIGN_HEIGHT_M, label_face="left", **lettering),
                     Box(-0.395, -reach, 0.0, 0.65, 0.03, _SIGN_HEIGHT_M, label_face="right", **lettering),
                     Box(-0.705, 0.0, 0.0, 0.03, width - 0.06, _SIGN_HEIGHT_M, z0=_SIGN_Z_M, color=shade),    # its back
                     Box(-0.38, 0.0, 0.0, 0.62, width - 0.06, 0.04, z0=_PLACE_TOP_M - 0.04, color=shade)]     # its roof
    for at in (*((-0.04,) if front_posts else ()), *((-0.685,) if rear_posts else ())):
        parts += [Box(at, side * (width / 2 - 0.04), 0.0, 0.07, 0.07, _SIGN_Z_M, color=_POST) for side in (1, -1)]
    return parts


def _counter(color: Color, plinth: Color, panel: Color, top: Color) -> _Parts:
    """A counter 1.1 m high and 0.6 m deep across the place, with a panelled front and a top."""
    return [Box(-0.3, 0.0, 0.0, 0.6, _PLACE_WIDTH_M, 1.06, color=color, band=plinth, band_range=(0.0, 0.09),
                panel=panel, panel_rect=(0.05, 0.16, 0.95, 0.9)),
            Box(-0.3, 0.0, 0.0, 0.6, _PLACE_WIDTH_M, 0.04, z0=1.06, color=top)]


def _stall(text: str, color: Color) -> _Parts:
    """A counter with a machine, cups and a till on it, and a back wall with a board above them, which carries
    the sign overhead: a post before the board would cross its lettering from most sides."""
    return [*_counter(_WOOD, _DARK_WOOD, (176, 130, 88), (66, 58, 54)),
            Box(-0.76, 0.0, 0.0, 0.08, _PLACE_WIDTH_M, _PLACE_TOP_M, color=_DARK_WOOD, panel=(38, 44, 42),
                panel_rect=(0.08, 0.62, 0.92, 0.83), label="MENU", label_color=_CREAM, label_height=0.52),   # read over
            Box(-0.34, 0.47, 0.0, 0.3, 0.36, 0.3, z0=1.1, color=_CHARCOAL, panel=(150, 156, 164),
                panel_rect=(0.15, 0.12, 0.85, 0.6)),                                             # the machine
            Cylinder(-0.34, 0.47, 0.06, 0.06, z0=1.4, color=_STEEL),
            *(Cylinder(-0.14, -0.06 + 0.12 * k, 0.04, 0.1, z0=1.1, color=(250, 250, 246), top=(96, 60, 36))
              for k in range(3)),                                                                # cups
            Box(-0.34, -0.5, 0.0, 0.26, 0.3, 0.22, z0=1.1, color=(70, 72, 78), panel=(120, 200, 230),
                panel_rect=(0.12, 0.3, 0.88, 0.9)),                                              # the till
            *_place_sign(text, color, front_posts=False)]


def _ticket_counter(text: str, color: Color) -> _Parts:
    """A booth: a counter with a glass screen across the middle of it, and walls behind that carry the sign."""
    slate = (88, 96, 112)
    return [*_counter(slate, (52, 56, 66), (124, 134, 152), (212, 212, 206)),
            Box(-0.34, 0.0, 0.0, 0.04, _PLACE_WIDTH_M - 0.12, 0.76, z0=1.1, color=(60, 62, 70), panel=_GLASS,
                panel_rect=(0.02, 0.04, 0.98, 0.97)),                                            # the screen
            *(Box(-0.56, side * (_PLACE_WIDTH_M / 2 - 0.03), 0.0, 0.48, 0.06, 1.86, color=slate) for side in (1, -1)),
            Box(-0.76, 0.0, 0.0, 0.08, _PLACE_WIDTH_M, _PLACE_TOP_M, color=(60, 64, 78)),
            Box(-0.2, 0.0, 0.0, 0.2, 0.3, 0.03, z0=1.1, color=_STEEL),                           # the hatch's tray
            Box(-0.16, 0.5, 0.0, 0.12, 0.26, 0.2, z0=1.1, color=_PAPER, band=color, band_range=(0.6, 1.0)),   # leaflets
            Box(-0.14, -0.42, 0.0, 0.14, 0.1, 0.06, z0=1.1, color=(40, 42, 48)),                 # a card reader
            *_place_sign(text, color, front_posts=False)]


def _office_chair(x: float, y: float, cloth: Color) -> _Parts:
    """A swivel chair facing +x: a foot, a column, a seat with two arms, and a back with a headrest, each
    with a gap below it, so that from the side and from behind it is nothing but a chair."""
    return [Cylinder(x, y, 0.2, 0.04, color=_CHARCOAL),
            Cylinder(x, y, 0.03, 0.4, z0=0.04, color=_STEEL),
            Box(x, y, 0.0, 0.36, 0.42, 0.08, z0=0.44, color=cloth),
            *(Box(x, y + side * 0.2, 0.0, 0.05, 0.04, 0.12, z0=0.52, color=_CHARCOAL) for side in (1, -1)),
            *(Box(x, y + side * 0.2, 0.0, 0.26, 0.05, 0.03, z0=0.64, color=_CHARCOAL) for side in (1, -1)),
            Box(x - 0.155, y, 0.0, 0.05, 0.38, 0.48, z0=0.58, color=cloth),
            Box(x - 0.155, y, 0.0, 0.05, 0.05, 0.08, z0=1.06, color=_CHARCOAL),
            Box(x - 0.155, y, 0.0, 0.05, 0.24, 0.13, z0=1.12, color=cloth)]


def _desk(text: str, color: Color) -> _Parts:
    """A desk behind a raised front that carries its name, with a screen on it and leaflets on the ledge of
    the front, a chair at it, and the sign overhead on four posts."""
    pale = (230, 226, 214)
    return [Box(-0.04, 0.0, 0.0, 0.08, _PLACE_WIDTH_M, 1.1, color=pale, band=color, band_range=(0.0, 0.1), panel=color,
                panel_rect=(0.2, 0.32, 0.8, 0.7), label=f"{text} DESK", label_color=_CREAM, label_height=0.6,
                label_width=0.86),                                                               # the front
            Box(-0.1, 0.0, 0.0, 0.2, _PLACE_WIDTH_M, 0.04, z0=1.1, color=_DARK_WOOD),             # the ledge on it
            Box(-0.3, 0.0, 0.0, 0.44, _PLACE_WIDTH_M, 0.05, z0=0.7, color=_WOOD),                 # the desk top
            *(Box(-0.3, side * (_PLACE_WIDTH_M / 2 - 0.03), 0.0, 0.44, 0.06, 0.7, color=pale) for side in (1, -1)),
            Box(-0.34, 0.36, 0.0, 0.05, 0.52, 0.34, z0=0.94, color=(184, 188, 194), label_face="back",
                panel=(130, 190, 226), panel_rect=(0.05, 0.07, 0.95, 0.93)),                     # the screen, its back
            Cylinder(-0.34, 0.36, 0.03, 0.19, z0=0.75, color=_CHARCOAL),                         # to whoever comes
            Cylinder(-0.34, 0.36, 0.11, 0.015, z0=0.75, color=_CHARCOAL),
            *(Box(-0.1, -0.62 + 0.13 * k, 0.0, 0.1, 0.1, 0.17, z0=1.14, color=_PAPER, band=tint, band_range=(0.5, 1.0))
              for k, tint in enumerate((color, (200, 120, 50), (60, 100, 170)))),                # leaflets
            *_office_chair(-0.6, -0.22, (190, 96, 60)),
            *_place_sign(text, color, rear_posts=True)]


def _kiosk(text: str, color: Color) -> _Parts:
    """A cabinet with a screen, its lettered head on top, and a printer on each side with paper on one."""
    pale = (216, 216, 210)
    return [Box(-0.36, 0.0, 0.0, 0.72, 0.7, _SIGN_Z_M, color=color, band=(34, 38, 46), band_range=(0.0, 0.05),
                panel=_SCREEN, panel_rect=(0.1, 0.58, 0.9, 0.86), label="START", label_color=(30, 50, 90),
                label_height=0.3),                                                               # the cabinet
            *(Box(-0.3, side * 0.575, 0.0, 0.6, 0.45, 0.92, color=pale, band=(60, 62, 70), band_range=(0.0, 0.08),
                  panel=(48, 50, 58), panel_rect=(0.12, 0.52, 0.88, 0.7)) for side in (1, -1)),  # the printers
            *(Box(-0.3, side * 0.575, 0.0, 0.6, 0.45, 0.04, z0=0.92, color=(70, 72, 80)) for side in (1, -1)),
            Box(-0.22, 0.575, 0.0, 0.3, 0.22, 0.05, z0=0.96, color=(248, 248, 242)),             # printed sheets
            *_place_sign(text, color, width=0.7, front_posts=False)]


def _back_room(depth: float, text: str, color: Color) -> _Parts:
    """What a deep place has behind the part people are served at: a closed room with the name on both sides."""
    middle, length = -(_PLACE_FRONT_M + depth) / 2, depth - _PLACE_FRONT_M
    return [Box(middle, side * _PLACE_WIDTH_M / 4, 0.0, length, _PLACE_WIDTH_M / 2, _PLACE_TOP_M, color=(206, 200, 188),
                band=(96, 86, 76), band_range=(0.0, 0.05), panel=color, panel_rect=(0.12, 0.5, 0.88, 0.8), label=text,
                label_face=face, label_color=_CREAM, label_height=0.6) for side, face in ((1, "left"), (-1, "right"))]


_PLACES = (   # target label, text on its sign, the sign's colour, what builds it, grounded request, high-level request
    ("print kiosk", "PRINT", (44, 84, 150), _kiosk, "go to the print kiosk", "I need this printed"),
    ("help desk", "HELP", (30, 110, 100), _desk, "take me to the help desk", "I need to report a lost item"),
    ("ticket counter", "TICKETS", (140, 50, 70), _ticket_counter, "go to the ticket counter",
     "buy me a ticket for the next show"),
    ("coffee stall", "COFFEE", (126, 58, 40), _stall, "go to the coffee stall", "get me a coffee"),
)
_PLACE_KINDS = {_kiosk: "kiosk", _desk: "desk", _ticket_counter: "ticket_counter", _stall: "stall"}    # of its item


def _taped(box: Box, tape: Color) -> _Parts:
    """A strip of packing tape along the middle of a box: over its top and down both its ends. Its front
    stays whole: a strip down the front makes two things of it, two files side by side."""
    return [Box(box.cx, box.cy, box.yaw, 0.045, box.sy + 0.006, box.sz + 0.003, z0=box.z0, color=tape)]


def _parcel_table() -> _Parts:
    """A table with the parcel on it, towards +x of it a crate on the floor, and two more things on the table.

    The table's top comes first, then the parcel and the crate, then the legs and the two other things, a
    second parcel in brown paper and a mug, and last what is stuck on the two parcels. Each parcel is a
    box with tape along its top and an address label on the side turned to +x."""
    label = dict(panel=(244, 242, 234), panel_rect=(0.62, 0.4, 0.94, 0.86))        # clear of the anchor at its middle
    parcel = Box(0.0, 0.0, 0.0, 0.3, 0.42, 0.25, z0=0.7, color=(40, 80, 200), top=(52, 96, 216), **label)
    other = Box(-0.08, 0.5, 0.0, 0.2, 0.26, 0.14, z0=0.7, color=(176, 140, 96), **label)
    return [Box(0.0, 0.0, 0.0, 0.9, 1.3, 0.05, z0=0.65, color=(172, 128, 86)),
            parcel,
            Box(0.9, 0.35, 0.0, 0.3, 0.3, 0.3, color=(150, 118, 78), band=(110, 84, 54), band_range=(0.0, 0.15),
                top=(74, 58, 42)),                                                           # open: its inside is dark
            *(Cylinder(i * 0.38, j * 0.58, 0.03, 0.65, color=(72, 66, 62)) for i in (-1, 1) for j in (-1, 1)),
            other,
            Cylinder(0.05, -0.48, 0.05, 0.13, z0=0.7, color=(232, 228, 216), top=(96, 60, 36)),  # a mug
            *_taped(parcel, (226, 204, 140)), *_taped(other, (120, 92, 60))]


# --- What people watch, wait at and take a picture with. The first two have their front on x = 0, like a place. ---

_SCREEN_WIDTH_M = 2.0
_SCREEN_DEPTH_M = 0.5                              # of the cabinet the screen stands on
_SCREEN_ANCHOR_Z_M = 1.5                           # the middle of the picture it shows
_LIFT_WIDTH_M = 2.6                                # of the shaft the doors are let into
_LIFT_DEPTH_M = 1.8
_LIFT_ANCHOR_Z_M = 1.2                             # on the doors, where the two leaves meet


def _big_screen() -> _Parts:
    """A display on a low cabinet, showing a slide: the cabinet, two posts on it behind, and a dark frame round
    a bright picture with a heading, a chart of four bars and three lines of writing. The picture stands 3 mm
    proud of the frame, its face on x = 0.

    The cabinet is as wide and as deep as the whole, so no floor shows under the display: on a stand with a
    foot, the floor between its posts was floor in the picture and part of an obstacle on the map."""
    ink, bars = (36, 76, 140), ((226, 150, 40), (60, 140, 110), (196, 60, 50), (36, 76, 140))
    return [Box(-_SCREEN_DEPTH_M / 2, 0.0, 0.0, _SCREEN_DEPTH_M, _SCREEN_WIDTH_M, 0.86, color=_CHARCOAL,
                band=(30, 32, 36), band_range=(0.0, 0.08), panel=(78, 82, 88), panel_rect=(0.03, 0.16, 0.97, 0.92)),
            *(Box(-0.16, side * 0.6, 0.0, 0.08, 0.08, 0.64, z0=0.86, color=_STEEL) for side in (1, -1)),     # the posts
            Box(-0.0615, 0.0, 0.0, 0.117, _SCREEN_WIDTH_M, 1.2, z0=0.9, color=(30, 32, 36), panel=(236, 240, 244),
                panel_rect=(0.02, 0.035, 0.98, 0.965)),                                          # the frame and the slide
            Box(-0.06, 0.0, 0.0, 0.12, 1.84, 0.2, z0=1.83, color=ink, label="RESULTS", label_color=_CREAM,
                label_height=0.62),                                                              # its heading
            Box(-0.06, -0.3, 0.0, 0.12, 1.14, 0.02, z0=1.0, color=_CHARCOAL),                    # the chart's foot line
            *(Box(-0.06, -0.72 + 0.28 * k, 0.0, 0.12, 0.2, up, z0=1.02, color=tint)
              for k, (up, tint) in enumerate(zip((0.3, 0.5, 0.42, 0.66), bars))),
            *(Box(-0.06, 0.6, 0.0, 0.12, across, 0.045, z0=z, color=(120, 126, 134))
              for across, z in ((0.5, 1.6), (0.42, 1.46), (0.5, 1.32)))]                         # the writing


def _lift(plate: float) -> _Parts:
    """A lift in its shaft, which reaches the ceiling: two closed door leaves with the dark line between them
    where they meet, a frame round the doorway, the lit number of the floor over it, and beside it a call
    plate with two buttons, towards +y for a `plate` of 1.0 and towards -y for -1.0. The frame's face is on
    x = 0 and the doors stand back from it."""
    steel, frame, wide, tall = (178, 182, 188), (92, 96, 104), 1.1, 2.1               # the doorway is 1.1 m by 2.1 m
    return [Box(-(_LIFT_DEPTH_M + 0.04) / 2, 0.0, 0.0, _LIFT_DEPTH_M - 0.04, _LIFT_WIDTH_M, _ROOM_HEIGHT_M,
                color=(210, 206, 196), band=(96, 86, 76), band_range=(0.0, 0.04)),               # the shaft
            *(Box(-0.0275, side * 0.2825, 0.0, 0.025, 0.535, tall, color=steel, panel=(196, 200, 206),
                  panel_rect=(0.12, 0.04, 0.88, 0.96)) for side in (1, -1)),                     # the two leaves
            Box(-0.031, 0.0, 0.0, 0.018, 0.04, tall, color=(34, 36, 40)),                        # the line between them
            *(Box(-0.02, side * (wide / 2 + 0.04), 0.0, 0.04, 0.08, tall + 0.08, color=frame) for side in (1, -1)),
            Box(-0.02, 0.0, 0.0, 0.04, wide, 0.08, z0=tall, color=frame),
            Box(-0.02, 0.0, 0.0, 0.04, 0.42, 0.17, z0=tall + 0.13, color=(24, 26, 30), label=_LIFT_FLOOR_LIT,
                label_color=(255, 176, 60), label_height=0.78),                                  # the floor it is at
            Box(-0.03, plate * 0.86, 0.0, 0.02, 0.13, 0.28, z0=1.0, color=(150, 154, 160), panel=(58, 60, 66),
                panel_rect=(0.2, 0.1, 0.8, 0.9)),                                                # the call plate
            *(Sphere(-0.022, plate * 0.86, z, 0.02, color=tint)
              for z, tint in ((1.2, (255, 190, 80)), (1.08, (236, 236, 230))))]


def _screen_place() -> tuple[_Parts, list[Item], Target, _Requests]:
    """The big screen, its face on x = 0: its parts, the one item they are, the target on the picture it shows,
    and its requests."""
    parts = _big_screen()
    footprint = oriented_rectangle((0.0, 0.0), 0.0, -_SCREEN_DEPTH_M, 0.0, _SCREEN_WIDTH_M)
    target = Target(id=_TARGET_ID, kind="place", label=_SCREEN_LABEL, anchor=(0.0, 0.0, _SCREEN_ANCHOR_Z_M),
                    footprint=footprint, front_yaw=0.0)
    item = Item("screen", 0.0, 0.0, 0.0, footprint, {"label": _SCREEN_LABEL}, "boxes", 0, len(parts))
    return parts, [item], target, _SCREEN_REQUESTS


def _lift_place(plate: float) -> tuple[_Parts, list[Item], Target, _Requests]:
    """The lift, its doors on x = 0 and its call plate to the side `plate` of them: its parts, the one item they
    are, the target on the doors, and its requests."""
    parts = _lift(plate)
    footprint = oriented_rectangle((0.0, 0.0), 0.0, -_LIFT_DEPTH_M, 0.0, _LIFT_WIDTH_M)
    target = Target(id=_TARGET_ID, kind="place", label=_LIFT_LABEL, anchor=(0.0, 0.0, _LIFT_ANCHOR_Z_M),
                    footprint=footprint, front_yaw=0.0)
    item = Item("lift", 0.0, 0.0, 0.0, footprint, {"label": _LIFT_LABEL, "plate": plate}, "boxes", 0, len(parts))
    return parts, [item], target, _LIFT_REQUESTS


def _tripod_camera() -> _Parts:
    """A camera on a tripod over the origin, looking along +x: three legs, one of them forward, the column, and
    the camera's body with its lens before it and a viewfinder on it.

    A leg leans, and a primitive stands upright: so each leg is twelve short lengths, each nearer the column
    than the one below it by less than its own width, which is a leaning leg from a few metres off."""
    black, lengths, spread, top = (38, 38, 42), 12, 0.32, 1.05     # a leg's foot is 0.32 m out, and the legs meet 1.05 m up
    out = [spread * (1 - (k + 0.5) / lengths) for k in range(lengths)]             # of each length, from the floor up
    legs = [Cylinder(out[k] * math.cos(turn), out[k] * math.sin(turn), 0.018, top / lengths, z0=k * top / lengths,
                     color=(60, 60, 64)) for turn in (0.0, 2 * math.pi / 3, -2 * math.pi / 3) for k in range(lengths)]
    return [*legs,
            Cylinder(0.0, 0.0, 0.025, 0.3, z0=top - 0.03, color=_STEEL),
            Box(0.0, 0.0, 0.0, 0.16, 0.26, 0.17, z0=1.32, color=black),
            Box(0.13, 0.0, 0.0, 0.1, 0.13, 0.13, z0=1.34, color=(22, 22, 24), panel=(96, 128, 168),
                panel_rect=(0.14, 0.14, 0.86, 0.86)),                                            # the lens and its glass
            Box(-0.01, 0.0, 0.0, 0.07, 0.1, 0.05, z0=1.49, color=black)]


# --- Furnishings. Each stands with its back on x = 0, the wall, and reaches into the room along +x. ---

def _plant(scale: float) -> _Parts:
    blobs = ((0.0, 0.0, 0.98, 0.3), (0.16, 0.06, 0.8, 0.22), (-0.14, 0.1, 0.84, 0.22), (0.02, -0.17, 0.86, 0.2),
             (0.06, 0.1, 1.22, 0.2), (-0.1, -0.06, 1.2, 0.19))
    at = 0.42 * scale                                # from the wall to the middle of the pot
    return [Cylinder(at, 0.0, 0.17 * scale, 0.42 * scale, color=(168, 96, 66), top=(70, 50, 38),
                     top_radius=0.22 * scale),
            Cylinder(at, 0.0, 0.025 * scale, 0.5 * scale, z0=0.4 * scale, color=(96, 72, 50)),
            *(Sphere(at + x * scale, y * scale, z * scale, r * scale, color=_LEAVES[k % len(_LEAVES)])
              for k, (x, y, z, r) in enumerate(blobs))]


def _bench(length: float = _BENCH_LENGTH_M) -> _Parts:
    return [Box(0.3, 0.0, 0.0, 0.42, length, 0.06, z0=0.42, color=_WOOD),
            Box(0.1, 0.0, 0.0, 0.05, length, 0.4, z0=0.5, color=_WOOD),
            *(Box(0.3, side * (length / 2 - 0.12), 0.0, 0.4, 0.06, 0.42, color=(60, 60, 64)) for side in (1, -1))]


def _bin() -> _Parts:
    """A litter bin: a steel drum under a dark domed lid with its opening towards the room."""
    return [Cylinder(0.3, 0.0, 0.2, 0.66, color=(150, 156, 162)),
            Cylinder(0.3, 0.0, 0.205, 0.05, z0=0.66, color=(52, 56, 62)),
            Sphere(0.3, 0.0, 0.71, 0.2, color=(52, 56, 62)),
            Box(0.47, 0.0, 0.0, 0.07, 0.2, 0.09, z0=0.74, color=(16, 16, 18))]


def _pillar() -> _Parts:
    return [Cylinder(0.32, 0.0, 0.3, 0.12, color=(170, 164, 152)),
            Cylinder(0.32, 0.0, 0.24, _ROOM_HEIGHT_M, color=(214, 208, 196))]


def _poster(text: str, ink: Color, width: float) -> _Parts:
    return [Box(0.02, 0.0, 0.0, 0.04, width, 1.2, z0=1.0, color=(70, 64, 58), panel=_PAPER,
                panel_rect=(0.05, 0.04, 0.95, 0.96), label=text, label_color=ink, label_height=0.3)]


def _door() -> _Parts:
    return [Box(0.03, 0.0, 0.0, 0.06, 1.0, 2.1, color=(86, 70, 60), panel=(128, 98, 72),
                panel_rect=(0.07, 0.0, 0.93, 0.965)),
            Sphere(0.09, -0.36, 1.02, 0.035, color=_STEEL)]


def _window(width: float) -> _Parts:
    """A window with a sill, and through it the sky over roofs far off."""
    frame = (238, 236, 230)
    # Each roof: where across the pane, how wide, how high, how hazy.
    roofs = ((-0.33, 0.22, 0.3, (148, 164, 182)), (-0.1, 0.16, 0.5, (132, 150, 172)),
             (0.11, 0.2, 0.24, (158, 172, 188)), (0.33, 0.2, 0.4, (142, 158, 178)))
    return [Box(0.02, 0.0, 0.0, 0.04, width, 1.2, z0=1.05, color=frame, panel=(178, 210, 238),
                panel_rect=(0.04, 0.06, 0.96, 0.94)),
            *(Box(0.021, at * width, 0.0, 0.042, across * width, up, z0=1.122, color=haze)
              for at, across, up, haze in roofs),
            Box(0.03, 0.0, 0.0, 0.06, 0.05, 1.2, z0=1.05, color=frame),
            Box(0.05, 0.0, 0.0, 0.1, width + 0.1, 0.04, z0=1.01, color=frame)]


def _notice_board() -> _Parts:
    """A cork board with sheets of paper pinned to it."""
    sheets = ((-0.44, 1.5, 0.22, 0.3), (-0.15, 1.44, 0.2, 0.28), (0.14, 1.56, 0.24, 0.2), (0.44, 1.4, 0.2, 0.3),
              (0.14, 1.24, 0.22, 0.22), (-0.44, 1.18, 0.2, 0.22))     # where along the wall, how high up, width, height
    tints = ((244, 240, 228), (246, 226, 130), (240, 176, 176), (176, 208, 234), (204, 232, 184), (244, 240, 228))
    return [Box(0.02, 0.0, 0.0, 0.04, 1.3, 0.9, z0=1.1, color=(104, 84, 62), panel=(196, 158, 108),
                panel_rect=(0.03, 0.045, 0.97, 0.955)),
            *(Box(0.0225, y, 0.0, 0.045, across, up, z0=z, color=tint)
              for (y, z, across, up), tint in zip(sheets, tints))]


def _water_cooler() -> _Parts:
    """A cabinet with a tap niche and a blue bottle upside down on it."""
    water = (140, 186, 230)
    return [Box(0.19, 0.0, 0.0, 0.34, 0.34, 1.0, color=(236, 238, 240), band=(128, 134, 142), band_range=(0.0, 0.06),
                panel=(52, 58, 66), panel_rect=(0.2, 0.58, 0.8, 0.8)),
            Cylinder(0.19, 0.0, 0.07, 0.1, z0=1.0, color=water, top_radius=0.14),
            Cylinder(0.19, 0.0, 0.14, 0.3, z0=1.1, color=water, top=(168, 204, 238))]


def _lockers() -> _Parts:
    """Three lockers side by side, each with a framed door, a vent and a handle."""
    return [*(Box(0.2, y, 0.0, 0.4, 0.4, 1.8, color=(74, 98, 122), band=(52, 56, 62), band_range=(0.0, 0.045),
                  panel=(108, 142, 172), panel_rect=(0.08, 0.07, 0.92, 0.97)) for y in (-0.4, 0.0, 0.4)),
            *(Box(0.4025, y, 0.0, 0.005, 0.2, 0.06, z0=1.52, color=(52, 56, 62)) for y in (-0.4, 0.0, 0.4)),
            *(Box(0.405, y + 0.12, 0.0, 0.01, 0.03, 0.12, z0=0.92, color=_STEEL) for y in (-0.4, 0.0, 0.4))]


def _extinguisher() -> _Parts:
    """A fire extinguisher on the wall under its red plate."""
    red, black = (204, 36, 32), (36, 36, 40)
    return [Box(0.015, 0.0, 0.0, 0.03, 0.3, 0.3, z0=1.3, color=(238, 238, 232), panel=red,
                panel_rect=(0.14, 0.14, 0.86, 0.86)),
            Box(0.02, 0.0, 0.0, 0.04, 0.12, 0.3, z0=0.5, color=black),                           # its bracket
            Cylinder(0.12, 0.0, 0.075, 0.44, z0=0.42, color=red),
            Cylinder(0.12, 0.0, 0.03, 0.07, z0=0.86, color=black),
            Box(0.12, 0.05, 0.0, 0.03, 0.13, 0.03, z0=0.93, color=black)]                        # its handle


def _shop(text: str, color: Color, goods: Sequence[Color]) -> _Parts:
    """A second stall, 2 m wide, selling what no request of a scene asks for: a back board on the wall, a
    counter with goods on it, and a lettered sign on two posts under a canopy."""
    width = 2.0
    spots = [-0.72 + 1.44 * k / (len(goods) - 1) for k in range(len(goods))]
    return [Box(0.04, 0.0, 0.0, 0.08, width, _PLACE_TOP_M, color=(112, 104, 96)),
            Box(0.8, 0.0, 0.0, 0.5, width, 1.02, color=(214, 206, 190), band=color, band_range=(0.0, 0.12),
                panel=(196, 186, 168), panel_rect=(0.05, 0.2, 0.95, 0.9)),
            Box(0.8, 0.0, 0.0, 0.5, width, 0.04, z0=1.02, color=(84, 78, 74)),
            Box(1.015, 0.0, 0.0, 0.07, width, _SIGN_HEIGHT_M, z0=_SIGN_Z_M, color=color, label=text, label_color=_CREAM,
                label_height=0.62, label_width=0.86),
            *(Box(1.01, side * (width / 2 - 0.04), 0.0, 0.07, 0.07, _SIGN_Z_M, color=_POST) for side in (1, -1)),
            Box(0.565, 0.0, 0.0, 0.97, width, 0.05, z0=_PLACE_TOP_M, color=color),
            *(Cylinder(0.8, y, 0.07, 0.12, z0=1.06, color=(150, 110, 84)) for y in spots),
            *(Sphere(0.8, y, 1.27, 0.12, color=tint) for y, tint in zip(spots, goods))]


_SHOPS = (   # text on the sign, the sign's colour, the colours of what is on the counter
    ("FLOWERS", (48, 112, 74), ((226, 96, 150), (240, 200, 70), (236, 236, 228), (232, 130, 60), (190, 110, 200))),
    ("FRUIT", (60, 96, 44), ((226, 150, 40), (196, 60, 50), (236, 208, 70), (120, 170, 60), (226, 150, 40))),
    ("SWEETS", (120, 70, 140), ((240, 150, 190), (150, 210, 230), (246, 226, 130), (180, 230, 170), (240, 150, 190))),
)
_POSTERS = (("MAP", (40, 60, 120), 1.2), ("NOTICES", (40, 60, 120), 1.3), ("EVENTS", (150, 40, 40), 1.2),
            ("EXIT", (30, 120, 60), 0.9))
_FURNISHINGS = ("plant", "bench", "bin", "pillar", "poster", "door", "window", "board", "cooler", "lockers",
                "extinguisher")
_FURNISHING_ODDS = (0.2, 0.12, 0.08, 0.08, 0.14, 0.06, 0.12, 0.06, 0.05, 0.05, 0.04)
_FLAT = ("poster", "window", "door", "board")      # what hangs on a wall or is let into it
_LETTERED = ("poster", "shop")                     # what is read; the shop is the second stall, which no request names
_BROAD = ("window", "door", "bench", "lockers", "pillar")     # what is still itself with someone before part of it
_LONG = ("bench", "pillar")                        # and with the side of the picture cutting it off
_AT_MOST = {"plant": 4, "bench": 2, "bin": 2, "pillar": 2, "door": 2, "window": 3, "board": 1, "cooler": 1,
            "lockers": 1, "extinguisher": 1, "poster": len(_POSTERS)}
_FURNISHING_BUILDERS = {"plant": _plant, "bench": _bench, "bin": _bin, "pillar": _pillar, "poster": _poster,
                        "door": _door, "window": _window, "board": _notice_board, "cooler": _water_cooler,
                        "lockers": _lockers, "extinguisher": _extinguisher}


def _furnishing(rng, kind: str, text: int) -> tuple[_Parts, dict[str, Any]]:
    """The parts of one furnishing of the kind, and what its builder was called with to make them; a poster
    is the `text`-th of the posters."""
    if kind == "plant":
        options = {"scale": rng.uniform(0.95, 1.2)}
    elif kind == "poster":
        options = dict(zip(("text", "ink", "width"), _POSTERS[text]))
    elif kind == "window":
        options = {"width": rng.uniform(1.6, 2.2)}
    elif kind == "bench":
        options = {"length": _BENCH_LENGTH_M}
    else:
        options = {}
    return _FURNISHING_BUILDERS[kind](**options), options


def _body(person: Person) -> Polygon:
    """The ground square a standing person fills."""
    return oriented_rectangle((person.x, person.y), 0.0, -Figure.radius, Figure.radius, 2 * Figure.radius)


def _furnish(rng, floor: tuple[float, float, float, float], people: Sequence[Person], footprints: Sequence[Polygon],
             about: Sequence[tuple[Polygon, float]], robot: RobotPose) -> list[tuple[_Parts, Item]]:
    """What stands against the walls of the room on `floor`, (x0, y0, x1, y1): each thing's parts and the item
    that says what they are, with its ground rectangle and with its range among all the parts returned, in
    order. `about` holds what the scene is about, a place, a table or the person asked for, as ground
    polygons with the height of what stands on them.

    A second stall goes in first and then a poster, each where the robot's picture shows all of it, if there
    is such a place; the stall, failing that, where the picture shows none of it. Then every wall is walked
    from one corner to the other, and at each step the kinds of furnishing are tried, in an order drawn by
    their odds, until one keeps its gap from the people, from what the scene is about and from what is
    already there, and the picture allows it.

    The picture allows nothing that what the scene is about hides a part of: that stands out against bare
    wall. Unless a thing is long, the picture has to hold all of it or none of it, and only the broad things
    may have people before them, none of whom hides more than half its width: half a bin at the edge of a
    picture, a plant half behind someone or the edge of a window behind someone's back is a shape nobody
    can name. A poster or the stall, which are lettered, must also be turned towards the camera with nothing
    before them, and nothing may be put where it hides one of them."""
    x0, y0, x1, y1 = floor
    # A wall's first corner, the direction it runs in and its length; the room lies to its left.
    walls = (((x0, y0), 0.0, x1 - x0), ((x1, y0), math.pi / 2, y1 - y0),
             ((x1, y1), math.pi, x1 - x0), ((x0, y1), -math.pi / 2, y1 - y0))
    view = _View(robot)
    subject = [(span, reach) for polygon, reach in about if (span := view.span(polygon)) is not None]
    bodies = [(span, _TALLEST_M) for person in people if (span := view.span(_body(person))) is not None]
    placed: list[tuple[_Parts, Item]] = []
    standing: list[tuple[tuple, float]] = []          # the span and the top of what stands out from a wall
    lettering: list[tuple[tuple, float]] = []         # the span and the underside of the lettered things in the picture

    def stand(parts: _Parts, wall: tuple, s: float) -> tuple[_Parts, Polygon, float, tuple[float, float, float]]:
        """The parts moved to the wall so that they begin s along it, their ground rectangle, their width, and
        the frame they are then laid out in: its origin on the wall and its yaw, into the room."""
        (wx, wy), along, _ = wall
        _, right, _, left = _extent(parts)           # it faces the room: the wall runs on from its left to its right
        at = s + left
        frame = (wx + at * math.cos(along), wy + at * math.sin(along), along + math.pi / 2)
        moved = _placed(parts, *frame)
        ex0, ey0, ex1, ey1 = _extent(moved)
        return moved, [(ex0, ey0), (ex1, ey0), (ex1, ey1), (ex0, ey1)], left - right, frame

    def fits(footprint: Polygon) -> bool:
        if any(distance_to_polygon((person.x, person.y), footprint) < _FURNISH_GAP_M for person in people):
            return False
        if any(_apart(footprint, other) < _FURNISH_GAP_M for other in footprints):
            return False
        return all(_apart(footprint, other.footprint) >= _FURNISH_APART_M for _, other in placed)

    def shows(kind: str, parts: _Parts, footprint: Polygon, wall: tuple) -> bool:
        """Whether the picture allows the thing there."""
        (low, high), span = _heights(parts), view.span(footprint)
        if kind not in _LONG:
            if view.misses(footprint, low, high):
                return True
            if span is None or not view.holds(footprint, low, high):
                return False
        if span is None:
            return True
        before = [*subject, *(() if kind in _BROAD else bodies), *(standing if kind in _LETTERED else ())]
        if any(view.hides(other, reach, span, low) for other, reach in before):
            return False
        if kind in _BROAD and kind != "pillar":      # a pillar shows above every head
            most = _BROAD_SHARE * (span[1] - span[0])
            if any(min(other[1], span[1]) - max(other[0], span[0]) > most
                   for other, reach in bodies if view.hides(other, reach, span, low)):
                return False
        if kind in _LETTERED and not view.faces(footprint, wall[1] + math.pi / 2):
            return False
        return kind in _FLAT or not any(view.hides(span, high, other, underside) for other, underside in lettering)

    def put(kind: str, parts: _Parts, footprint: Polygon, frame: tuple[float, float, float],
            options: Mapping[str, Any]) -> None:
        (x, y, yaw), start = frame, placed[-1][1].stop if placed else 0      # its parts follow those already there
        item = Item(kind, x, y, wrap_angle(yaw), footprint, options, "props", start, start + len(parts))
        placed.append((parts, item))
        (low, high), span = _heights(parts), view.span(footprint)
        if span is not None and kind not in _FLAT:
            standing.append((span, high))
        if span is not None and kind in _LETTERED and view.holds(footprint, low, high):
            lettering.append((span, low))

    def spots(kind: str, parts: _Parts) -> list[tuple[bool, _Parts, Polygon, tuple[float, float, float]]]:
        """Where along the walls the thing can go: whether the picture then shows it, its parts there, its
        rectangle and its frame."""
        found = []
        for wall in walls:
            s = _WALL_ENDS_M
            while True:
                moved, footprint, width, frame = stand(parts, wall, s)
                if s + width > wall[2] - _WALL_ENDS_M:
                    break
                s += _WALL_STEP_M
                if fits(footprint) and shows(kind, moved, footprint, wall):
                    found.append((view.holds(footprint, *_heights(moved)), moved, footprint, frame))
        return found

    counts: dict[str, int] = {}
    texts = [int(k) for k in rng.permutation(len(_POSTERS))]     # the order the posters are hung in, none twice
    shop = dict(zip(("text", "color", "goods"), _SHOPS[int(rng.integers(len(_SHOPS)))]))
    first = (("shop", _shop(**shop), shop, True), ("poster", *_furnishing(rng, "poster", texts[0]), False))
    for kind, parts, options, anywhere in first:     # anywhere: also where the picture shows nothing of it
        found = spots(kind, parts)
        found = [spot for spot in found if spot[0]] or (found if anywhere else [])
        if found:
            _, moved, footprint, frame = found[int(rng.integers(len(found)))]
            put(kind, moved, footprint, frame, options)
            counts[kind] = 1
    last = None
    for wall in walls:
        s, end = _WALL_ENDS_M + rng.uniform(0.0, 1.0), wall[2] - _WALL_ENDS_M
        while s < end:
            order = rng.choice(len(_FURNISHINGS), size=len(_FURNISHINGS), replace=False, p=_FURNISHING_ODDS)
            for kind in (_FURNISHINGS[int(k)] for k in order):
                if kind == last or counts.get(kind, 0) >= _AT_MOST.get(kind, 99):
                    continue
                parts, options = _furnishing(rng, kind, texts[counts.get("poster", 0) % len(texts)])
                moved, footprint, width, frame = stand(parts, wall, s)
                if s + width <= end and fits(footprint) and shows(kind, moved, footprint, wall):
                    put(kind, moved, footprint, frame, options)
                    counts[kind], last = counts.get(kind, 0) + 1, kind
                    s += width + rng.uniform(*_FURNISH_SPACING_M)
                    break
            else:
                s += _WALL_STEP_M
    return placed


def _shirts(rng, n: int, taken: Sequence[Color] = ()) -> list[Color]:
    """Shirt colours for n people who are not the target: unlike one another and unlike those taken, while
    there are colours left."""
    free = [color for color in _OTHER_COLORS if color not in taken] or list(_OTHER_COLORS)
    order = [int(k) for k in rng.permutation(len(free))]
    return [free[order[k % len(order)]] for k in range(n)]


def _person(rng, index: int, x: float, y: float, yaw: float, shirt: Color,
            group: str | None = None) -> tuple[Person, Figure]:
    """Person p<index>, a little off (x, y) and off facing yaw, and the figure that draws them.

    The index gives the figure its height, skin, hair and trousers; the shirt is the colour asked for."""
    person = Person(id=f"p{index}", x=x + rng.uniform(-_JITTER_M, _JITTER_M), y=y + rng.uniform(-_JITTER_M, _JITTER_M),
                    yaw=wrap_angle(yaw + rng.uniform(-_JITTER_RAD, _JITTER_RAD)), group=group)
    return person, Figure(person.x, person.y, person.yaw, color=shirt, **appearance(index))


def _place_ground(near: float, far: float) -> Polygon:
    """The ground under a place from `near` to `far` metres behind its front, across its width, in its frame.

    It is measured from the front itself, not from its own middle: two stretches that meet then have the
    very same corners where they do, and corners in common with the target's footprint, which they divide."""
    return oriented_rectangle((0.0, 0.0), 0.0, -far, -near, _PLACE_WIDTH_M)


def _place(rng, depth: float) -> tuple[_Parts, list[Item], Target, _Requests]:
    """One of the places, its front on x = 0 and the rest of it behind: its parts, the items they are, the
    target on the front of its counter, desk or cabinet, and its requests.

    The part people are served at is one item and the room behind it, if it has one, another, both in the
    place's frame. Each has the ground it stands on itself: the first the 0.8 m behind the front, which is
    all of the target's footprint when there is no room, and the room the rest of that footprint. The two
    rectangles meet on one line, and between them they are the one obstacle under the place."""
    label, text, color, build, grounded, high_level = _PLACES[int(rng.integers(len(_PLACES)))]
    parts = build(text, color)
    front = len(parts)
    if depth > _PLACE_FRONT_M:
        parts += _back_room(depth, text, color)
    footprint = oriented_rectangle((-depth / 2, 0.0), 0.0, -depth / 2, depth / 2, _PLACE_WIDTH_M)
    target = Target(id=_TARGET_ID, kind="place", label=label, anchor=(0.0, 0.0, _PLACE_ANCHOR_Z_M),
                    footprint=footprint, front_yaw=0.0)
    items = [Item(_PLACE_KINDS[build], 0.0, 0.0, 0.0, _place_ground(0.0, _PLACE_FRONT_M),
                  {"label": label, "text": text, "color": color}, "boxes", 0, front)]
    if len(parts) > front:
        items.append(Item("back_room", 0.0, 0.0, 0.0, _place_ground(_PLACE_FRONT_M, depth),
                          {"depth": depth, "text": text, "color": color}, "boxes", front, len(parts)))
    return parts, items, target, (grounded, high_level, None)


def _person_target(person: Person) -> Target:
    return Target(id=_TARGET_ID, kind="person", label=_PERSON_LABEL, person_id=person.id,
                  anchor=(person.x, person.y, _PERSON_ANCHOR_Z_M))


def _episodes(situation: str, requests: _Requests, group: str | None = None) -> tuple[EpisodeModel, EpisodeModel]:
    """e0 asks for the target by name; e1 asks for what is done there, and its plan may end with the way back.

    Both carry the situation's rationale, and none where it has no entry."""
    grounded, high_level, context = requests
    rationale = None
    if situation in _RATIONALE:
        people, expectations, placement = _RATIONALE[situation]
        rationale = Rationale(people=people, expectations=expectations, placement=placement)

    def go_there() -> PlanStep:
        return PlanStep(type="navigate", target_ids=[_TARGET_ID])

    def episode(episode_id: str, request: str, request_type: str, plan: list[PlanStep],
                context: str | None = None) -> EpisodeModel:
        truth = EpisodeGT(request_type=request_type, target_ids=[_TARGET_ID], situation=situation,
                          group_at_target=group, reference_plan=plan, rationale=rationale)
        return EpisodeModel(episode_id=episode_id, request=request, context=context, ground_truth=truth)

    way_back = [PlanStep(type="navigate", optional=True), PlanStep(type="act", optional=True)]
    return (episode("e0", grounded, "grounded", [go_there()]),
            episode("e1", high_level, "high_level", [go_there(), PlanStep(type="act"), *way_back], context))


def _finish(rng, name: str, boxes: Sequence[Solid], footprints: Sequence[Polygon], figures: Sequence[Figure],
            people: Sequence[Person], robot: RobotPose, targets: Sequence[Target],
            episodes: Sequence[EpisodeModel], items: Sequence[Item] = ()) -> Layout:
    """The layout in its room: the floor a margin round everything in it, furnishings along the walls, and an
    obstacle under every footprint and every furnishing. `items` are those of `boxes`; the layout's are
    these and then one for each furnishing."""
    xs, ys = zip((robot.x, robot.y), *((person.x, person.y) for person in people),
                 *(corner for footprint in footprints for corner in footprint))
    x0, x1 = min(xs) - _BOUNDS_MARGIN_M, max(xs) + _BOUNDS_MARGIN_M
    y0, y1 = min(ys) - _BOUNDS_MARGIN_M, max(ys) + _BOUNDS_MARGIN_M
    floor = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    room = Room(polygon=tuple(floor), height=_ROOM_HEIGHT_M, **_ROOMS[int(rng.integers(len(_ROOMS)))])
    top = max((_heights([part])[1] for part in boxes), default=0.0)
    asked_for = {target.person_id for target in targets}
    about = [*((footprint, top) for footprint in footprints),
             *((_body(person), _TALLEST_M) for person in people if person.id in asked_for)]
    furnishings = _furnish(rng, (x0, y0, x1, y1), people, footprints, about, robot)
    obstacles = [*footprints, *(item.footprint for _, item in furnishings)]
    scene_gt = SceneGT(people=list(people), targets=list(targets), bounds=PolygonModel(polygon=floor),
                       obstacles=[PolygonModel(polygon=obstacle) for obstacle in obstacles])
    return Layout(name, tuple(boxes), tuple(figures), robot, scene_gt, tuple(episodes), room,
                  tuple(part for parts, _ in furnishings for part in parts), tuple(footprints),
                  (*items, *(item for _, item in furnishings)))


def recipe_no_one(rng, view_angle_deg: float, kind: str = "place") -> Layout:
    """Nobody at the target, seen from `view_angle_deg` off its front, to either side.

    kind: "place", a stall, a booth, a desk or a kiosk; "deep_place", one of those with a room behind it,
    2.4 m deep in all; "object", a parcel on a table, which has no front, so there the angle is only the
    robot's bearing; "person", someone standing alone; "lift" and "screen", the lift and the big screen that
    people wait at and watch in other scenes, here with nobody at them."""
    if kind not in _NO_ONE_KINDS:
        raise ValueError(f"unknown kind {kind!r}: expected one of {', '.join(_NO_ONE_KINDS)}")
    a = _sign(rng) * math.radians(view_angle_deg)
    d = rng.uniform(4.5, 5.5)
    robot = RobotPose(d * math.cos(a), d * math.sin(a),
                      wrap_angle(a + math.pi + rng.uniform(-1.0, 1.0) * _VIEW_YAW_JITTER_RAD))
    boxes, footprints, people, figures, items = [], [], [], [], []
    if kind == "object":
        # The table is turned to the robot. The crate, 0.9 m towards the robot and 0.35 m to the side, is the
        # low thing in its way (spec 12): its corner is 0.22 m from the standard goal's first candidate, and
        # the first candidate clear of it lies about 0.25 m further out, 1.40 m from the table's centre and so
        # inside the 1.6 m ring.
        parts = _parcel_table()
        boxes = _placed(parts, 0.0, 0.0, a)
        footprints = [_rectangle(boxes[0]), _rectangle(boxes[2])]
        # The item is the table with its crate, so its rectangle reaches from the one to the other. The two
        # obstacles above lie on it, and so does the floor between them, which stays free for the standard goal.
        x0, y0, x1, y1 = _extent(parts)              # in the table's own frame
        middle = (y0 + y1) / 2
        ground = oriented_rectangle((-middle * math.sin(a), middle * math.cos(a)), a, x0, x1, y1 - y0)
        items = [Item("parcel_table", 0.0, 0.0, wrap_angle(a), ground, {}, "boxes", 0, len(boxes))]
        target = Target(id=_TARGET_ID, kind="object", label=_PARCEL_LABEL, anchor=_PARCEL_ANCHOR,
                        match_radius=_PARCEL_RADIUS_M)
        requests = _PARCEL_REQUESTS
    elif kind == "person":
        person, figure = _person(rng, 0, 0.0, 0.0, 0.0, _TARGET_PERSON_COLOR)
        people, figures = [person], [figure]
        target, requests = _person_target(person), _PERSON_REQUESTS
    elif kind in ("lift", "screen"):                 # the call plate of the lift on the robot's side, as where people wait
        boxes, items, target, requests = _lift_place(math.copysign(1.0, a)) if kind == "lift" else _screen_place()
        footprints = [target.footprint]
    else:
        boxes, items, target, requests = _place(rng, _PLACE_DEPTH_M[kind])
        footprints = [target.footprint]
    return _finish(rng, f"no_one_{kind}", boxes, footprints, figures, people, robot, [target],
                   _episodes("no_one", requests), items)


def recipe_waiting_line(rng, n_people: int) -> Layout:
    """A line of 1 to 6 people leading straight out from a place, seen from one side.

    The robot looks at the line squarely from the side, so nobody hides anybody. It stands back until the
    place and the reference region behind the last person fit a 60 degree image, and stays within 5.9 m of
    the target while the image then holds the whole place. Behind five or six people it does not: the near
    end of the place, which the camera sees from nearer than the line, would be cut off, and its name with
    it. The robot then stands where the place's near corner and the middle of the region are each just
    inside the image, which for six people is 6.23 m from the target."""
    if not 1 <= n_people <= _LINE_MAX_PEOPLE:
        raise ValueError(f"a waiting line has 1 to {_LINE_MAX_PEOPLE} people, not {n_people}")
    boxes, items, target, requests = _place(rng, _PLACE_FRONT_M)
    shirts = _shirts(rng, n_people)
    people, figures = zip(*(_person(rng, i, _LINE_FIRST_M + _LINE_SPACING_M * i, 0.0, math.pi, shirts[i], _GROUP_ID)
                            for i in range(n_people)))
    span = _LINE_FIRST_M + _LINE_SPACING_M * (n_people - 1) + _LINE_REGION_M     # from the anchor to that region
    side = _sign(rng)
    x = (span - _PLACE_FRONT_M) / 2                  # opposite the middle of what the image has to hold
    h = min(max(_LINE_VIEW_MIN_M, 1.2 * (span + _PLACE_FRONT_M)), math.sqrt(_MAX_RANGE_M ** 2 - x ** 2))
    forward, near_side = CAMERA_MOUNT["forward_m"], _PLACE_WIDTH_M / 2
    left, right = -(_PLACE_FRONT_M + _LINE_EDGE_M[0]), span + _LINE_EDGE_M[1]    # what the image is to reach, each way
    if x - _View.half_width(h - forward - near_side, _PLACE_TOP_M) > left:
        # x - half_width(ahead - near_side, top) = left and x + half_width(ahead, 0) = right, solved for both.
        pitch, eye = CAMERA_MOUNT["pitch_down_rad"], CAMERA_MOUNT["height_m"]
        ahead = (((right - left) / _VIEW_TAN - (2 * eye - _PLACE_TOP_M) * math.sin(pitch)) / (2 * math.cos(pitch))
                 + near_side / 2)                    # from the camera to the line
        x, h = right - _View.half_width(ahead, 0.0), ahead + forward
    robot = RobotPose(x, side * h, -side * math.pi / 2)
    return _finish(rng, f"waiting_line_{n_people}", boxes, [target.footprint], figures, people, robot, [target],
                   _episodes("waiting_line", requests, _GROUP_ID), items)


def recipe_conversation_group(rng, n_people: int) -> Layout:
    """2, 3 or 4 people facing the centre of their circle; the first, at the origin and in red, is the target.

    The robot faces the centre from 4.2 to 4.8 m away, at a bearing that keeps the members apart in the image."""
    if n_people not in _GROUP_BEARING_DEG:
        raise ValueError(f"a conversation group has {', '.join(map(str, _GROUP_BEARING_DEG))} people, not {n_people}")
    radius = _GROUP_NEIGHBOUR_M / 2 / math.sin(math.pi / n_people)
    turns = [math.pi + 2 * math.pi * i / n_people for i in range(n_people)]      # of each member, round the centre
    shirts = [_TARGET_PERSON_COLOR, *_shirts(rng, n_people - 1)]
    people, figures = zip(*(_person(rng, i, radius + radius * math.cos(t), radius * math.sin(t), t + math.pi,
                                    shirts[i], _GROUP_ID) for i, t in enumerate(turns)))
    b = _sign(rng) * math.radians(_GROUP_BEARING_DEG[n_people])
    d = rng.uniform(4.2, 4.8)
    robot = RobotPose(radius + d * math.cos(b), d * math.sin(b), wrap_angle(b + math.pi))
    return _finish(rng, f"conversation_group_{n_people}", [], [], figures, people, robot, [_person_target(people[0])],
                   _episodes("conversation_group", _PERSON_REQUESTS, _GROUP_ID))


def _seeing(x: float, y: float, whole: Sequence[tuple[float, float]], more: Sequence[tuple[float, float]]) -> RobotPose:
    """The robot at (x, y), turned so that the leftmost and the rightmost of all the points are as far from its
    heading as one another, or as near that as leaves each point of `whole` well inside the picture's sides.

    `whole` is what the picture has to hold all of, the people and what they are at; `more` is floor that it
    shows as much of as then fits."""
    bearings = [math.atan2(py - y, px - x) for px, py in (*whole, *more)]
    about = math.atan2(sum(map(math.sin, bearings)), sum(map(math.cos, bearings)))       # their mean direction
    off = [wrap_angle(bearing - about) for bearing in bearings]
    turn, held = (min(off) + max(off)) / 2, off[:len(whole)]
    turn = min(max(turn, max(held) - _VIEW_INSIDE_RAD), min(held) + _VIEW_INSIDE_RAD)
    return RobotPose(x, y, wrap_angle(about + turn))


def _attending(rng, name: str, situation: str, place: tuple[_Parts, list[Item], Target, _Requests],
               spots: Sequence[tuple[float, float]], side: float, bearing_deg: float) -> Layout:
    """People standing on the spots before the place, whose face is on x = 0, each facing it, and the robot
    behind them and to `side` of them: `bearing_deg` off straight behind, seen from their middle.

    The picture holds the face of the target and the people, and as much as then fits of the floor behind and
    beside them on the side away from the robot: the region lies there, and what of it lies on the robot's own
    side is in the picture with the people."""
    parts, items, target, requests = place
    shirts = _shirts(rng, len(spots))
    people, figures = zip(*(_person(rng, i, x, y, math.atan2(-y, -x), shirts[i], _GROUP_ID)
                            for i, (x, y) in enumerate(spots)))
    cx, cy = (sum(c) / len(spots) for c in zip(*spots))
    bearing, distance = side * math.radians(bearing_deg), rng.uniform(*_ATTEND_VIEW_M)
    beyond = [point for x, y in spots for point in ((x + _ATTEND_FLOOR_M, y), (x, y - side * _ATTEND_FLOOR_M))]
    front = [corner for corner in target.footprint if corner[0] == 0.0]                  # the two ends of the target's face
    robot = _seeing(cx + distance * math.cos(bearing), cy + distance * math.sin(bearing), [*front, *spots], beyond)
    return _finish(rng, name, parts, [target.footprint], figures, people, robot, [target],
                   _episodes(situation, requests, _GROUP_ID), items)


def recipe_audience(rng, n_people: int) -> Layout:
    """3, 4 or 5 people standing before a screen and watching it, seen from behind them and to one side.

    Three stand in one row; four and five in two, the back row over to the side away from the robot (see
    _AUDIENCE_PLACES). The first row is 2.05 m from the screen or further and the last no further than 3.15 m."""
    if n_people not in _AUDIENCE_PLACES:
        raise ValueError(f"an audience has {', '.join(map(str, _AUDIENCE_PLACES))} people, not {n_people}")
    rows, side = _AUDIENCE_PLACES[n_people], _sign(rng)
    first = rng.uniform(_AUDIENCE_ROWS_M[0], _AUDIENCE_ROWS_M[1] - (len(rows) - 1) * _AUDIENCE_ROW_GAP_M)
    spots = [(first + k * _AUDIENCE_ROW_GAP_M, side * place * _AUDIENCE_NEIGHBOUR_M)
             for k, row in enumerate(rows) for place in row]
    return _attending(rng, f"audience_{n_people}", "audience", _screen_place(), spots, side,
                      _AUDIENCE_BEARING_DEG[n_people])


def recipe_lift_wait(rng, n_people: int) -> Layout:
    """2 or 4 people waiting before the closed doors of a lift, gathered there and not one behind another (see
    _LIFT_PLACES), seen from behind them and to one side. The nearest of them stands 1.3 to 1.7 m from the doors."""
    if n_people not in _LIFT_PLACES:
        raise ValueError(f"at a lift {', '.join(map(str, _LIFT_PLACES))} people wait, not {n_people}")
    side, nearest = _sign(rng), rng.uniform(*_LIFT_NEAREST_M)
    spots = [(nearest + behind, side * across) for behind, across in _LIFT_PLACES[n_people]]
    # The call plate is on the robot's side of the doors. The robot looks across the people at the doors, so in its
    # picture two of them stand before the far side of the doorway and leave the near side clear. Four fill the
    # width of the shaft, and one of them stands before the plate.
    return _attending(rng, f"lift_wait_{n_people}", "lift_wait", _lift_place(side), spots, side,
                      _LIFT_BEARING_DEG[n_people])


def recipe_group_photo(rng, n_posing: int, target: str = "posing") -> Layout:
    """Someone taking a picture of 2 or 3 people, who stand in a row at the origin and face them. The camera is
    on a tripod before the one who takes the picture.

    target: "posing", the one of the row at its end on the robot's side, or "taker", the one taking the
    picture; that person wears the target's colour. The robot looks on from behind the row and to one side
    when the target is the picture-taker, and from behind the picture-taker and to one side when it is one of
    those posing (see _PHOTO_VIEW_DEG).

    Everyone stands and turns as loosely as people do, the picture-taker too. The camera does not: the tripod
    stands on the line from the picture-taker to the middle of the row and the camera looks along it, however
    the one behind it has turned. The reference rule reads no more of their yaw than that they face the row."""
    if n_posing not in (2, 3) or target not in _PHOTO_VIEW_DEG:
        raise ValueError(f"a picture is taken of 2 or 3 people, with the target {' or '.join(map(repr, _PHOTO_VIEW_DEG))}: "
                         f"not of {n_posing}, with the target {target!r}")
    side = _sign(rng)
    asked = n_posing if target == "taker" else (n_posing - 1 if side > 0 else 0)         # the index of the target person
    others = iter(_shirts(rng, n_posing))
    shirts = [_TARGET_PERSON_COLOR if i == asked else next(others) for i in range(n_posing + 1)]
    away = rng.uniform(*_PHOTO_TAKER_M)
    spots = [(0.0, (i - (n_posing - 1) / 2) * _PHOTO_NEIGHBOUR_M, 0.0) for i in range(n_posing)] + [(away, 0.0, math.pi)]
    people, figures = zip(*(_person(rng, i, x, y, yaw, shirts[i], _GROUP_ID) for i, (x, y, yaw) in enumerate(spots)))
    taker = people[-1]
    aim = math.atan2(-taker.y, -taker.x)           # from where they stand to the row's middle: the way the camera looks
    stand = (taker.x + _PHOTO_TRIPOD_M * math.cos(aim), taker.y + _PHOTO_TRIPOD_M * math.sin(aim), aim)
    parts = _tripod_camera()
    x0, y0, x1, y1 = _extent(parts)
    ground = oriented_rectangle(stand[:2], aim, x0, x1, y1 - y0)
    back = rng.uniform(*_PHOTO_BEHIND_M[target])
    aside = side * (away + back) * math.tan(math.radians(_PHOTO_VIEW_DEG[target]))
    floor = [(away, _PHOTO_FLOOR_M), (away, -_PHOTO_FLOOR_M), (away + _PHOTO_FLOOR_M, 0.0)]
    robot = _seeing(-back if target == "taker" else away + back, aside, [*((x, y) for x, y, _ in spots), *ground], floor)
    item = Item("tripod_camera", stand[0], stand[1], wrap_angle(aim), ground, {}, "boxes", 0, len(parts))
    return _finish(rng, f"group_photo_{n_posing}_{target}", _placed(parts, *stand), [ground], figures, people, robot,
                   [_person_target(people[asked])], _episodes("group_photo", _PERSON_REQUESTS, _GROUP_ID), [item])


class _NoFreeSpot(RuntimeError):
    """The draws for a free spot ran out."""


def _sight_line(robot: RobotPose, x: float, y: float) -> list[tuple[float, float]]:
    """Points along the ground from the robot to (x, y), both ends included, no more than a step apart."""
    dx, dy = x - robot.x, y - robot.y
    steps = max(1, math.ceil(math.hypot(dx, dy) / _SIGHT_STEP_M))
    return [(robot.x + dx * i / steps, robot.y + dy * i / steps) for i in range(steps + 1)]


def _in_clear_view(layout: Layout, added: Sequence[Person]) -> bool:
    """Whether the robot's image still shows everyone once the people in `added` stand in the layout.

    Each of them has to stand inside the view cone with nothing on the robot's line of sight to them: nothing
    the scene is about and nobody else within the sight gap of it. Nor may one of them stand that near the
    line of sight to someone already there, or be seen between a place and the first of the people waiting
    at it, where the picture would show them as the one being served. The furnishings do not count: they
    are put against the walls afresh once the people stand, and no line of sight passes there."""
    robot, present = layout.robot, layout.scene_gt.people

    def stands_on(line: list[tuple[float, float]], person: Person) -> bool:
        return any(math.hypot(person.x - x, person.y - y) < _SIGHT_GAP_M for x, y in line)

    def off_heading(x: float, y: float) -> float:
        return wrap_angle(math.atan2(y - robot.y, x - robot.x) - robot.yaw)

    served = []                                      # the bearings between a place and the first in line at it
    waiting = [person for person in present if person.group == _GROUP_ID]
    for target in layout.scene_gt.targets:
        if target.footprint is not None and waiting:
            ax, ay = target.anchor[:2]
            first = min(waiting, key=lambda person: math.hypot(person.x - ax, person.y - ay))
            served.append(sorted((off_heading(ax, ay), off_heading(first.x, first.y))))
    for person in added:
        off, wide = off_heading(person.x, person.y), _SIGHT_GAP_M / math.hypot(person.x - robot.x, person.y - robot.y)
        if abs(off) > _SPOT_HALF_ANGLE_RAD or any(low - wide < off < high + wide for low, high in served):
            return False
        line = _sight_line(robot, person.x, person.y)
        if any(stands_on(line, other) for other in (*present, *added) if other is not person):
            return False
        if any(distance_to_polygon(point, footprint) < _SIGHT_GAP_M
               for footprint in layout.footprints for point in line):
            return False
    return not any(stands_on(_sight_line(robot, other.x, other.y), person) for other in present for person in added)


def _with_more_people(layout: Layout, rng, added: Sequence[tuple[Person, Figure]]) -> Layout:
    """The layout rebuilt with the people added: its room grows round them and is furnished afresh. What the
    scene is about stays, and its items with it."""
    gt = layout.scene_gt
    figures = (*layout.figures, *(figure for _, figure in added))
    people = (*gt.people, *(person for person, _ in added))
    return _finish(rng, layout.name, layout.boxes, layout.footprints, figures, people, layout.robot, gt.targets,
                   layout.episodes, [item for item in layout.items if item.source == "boxes"])


def _on_free_spot(layout: Layout, rng, stand: Callable[..., list[tuple[Person, Figure]]]) -> Layout:
    """The layout with the people that `stand(rng, index, x, y, shirts)` puts on a free spot (x, y), the first
    as p<index>, in shirts unlike those already worn.

    A spot is drawn in the robot's view, and drawn again until it keeps its gap from the target's anchor and
    from everyone in the layout and the people standing on it leave everyone in clear view."""
    robot, gt = layout.robot, layout.scene_gt
    keep_off = [(target.anchor[:2], _SPOT_TARGET_GAP_M) for target in gt.targets]
    keep_off += [((person.x, person.y), _SPOT_PERSON_GAP_M) for person in gt.people]
    shirts = _shirts(rng, 2, [figure.color for figure in layout.figures])
    for _ in range(_SPOT_MAX_DRAWS):
        rho = rng.uniform(*_SPOT_RANGE_M)
        direction = robot.yaw + rng.uniform(-1.0, 1.0) * _SPOT_HALF_ANGLE_RAD
        x, y = robot.x + rho * math.cos(direction), robot.y + rho * math.sin(direction)
        if any(math.hypot(x - px, y - py) < gap for (px, py), gap in keep_off):
            continue
        added = stand(rng, len(gt.people), x, y, shirts)
        if _in_clear_view(layout, [person for person, _ in added]):
            return _with_more_people(layout, rng, added)
    raise _NoFreeSpot(f"{layout.name}: no free spot in {_SPOT_MAX_DRAWS} draws")


def _bystander(rng, index: int, x: float, y: float, shirts: Sequence[Color]) -> list[tuple[Person, Figure]]:
    """Someone alone at (x, y), facing anywhere."""
    return [_person(rng, index, x, y, rng.uniform(-math.pi, math.pi), shirts[0])]


def _pair(rng, index: int, x: float, y: float, shirts: Sequence[Color]) -> list[tuple[Person, Figure]]:
    """Two people of group g1, facing each other across (x, y)."""
    along = rng.uniform(-math.pi, math.pi)         # from the first of the two to the second
    dx, dy = _PAIR_HALF_GAP_M * math.cos(along), _PAIR_HALF_GAP_M * math.sin(along)
    return [_person(rng, index, x - dx, y - dy, along, shirts[0], _SECOND_GROUP_ID),
            _person(rng, index + 1, x + dx, y + dy, along + math.pi, shirts[1], _SECOND_GROUP_ID)]


def _add_bystanders(layout: Layout, rng, n: int) -> Layout:
    """The layout with n more people, each alone on a free spot and facing anywhere."""
    for _ in range(n):
        layout = _on_free_spot(layout, rng, _bystander)
    return layout


def _add_second_group(layout: Layout, rng) -> Layout:
    """The layout with two more people, group g1, facing each other across a free spot."""
    return _on_free_spot(layout, rng, _pair)


def _add_extras(layout: Layout, rng, bystanders: int, second_group: bool) -> Layout:
    """The layout with its second group, if it is to have one, and then its bystanders.

    Whoever is placed first can leave the rest no spot: behind a waiting line seen from the side, all that
    is in clear view is a narrow wedge past the line's far end. They are then all placed afresh."""
    for _ in range(_EXTRAS_MAX_TRIES):
        try:
            return _add_bystanders(_add_second_group(layout, rng) if second_group else layout, rng, bystanders)
        except _NoFreeSpot:
            continue
    raise RuntimeError(f"{layout.name}: no free spots in {_EXTRAS_MAX_TRIES} tries")


def transform_layout(layout: Layout, yaw: float, tx: float, ty: float) -> Layout:
    """The layout turned by yaw about the origin, then shifted by (tx, ty).

    Every yaw is wrapped: a yaw of 1.5 pi in scene.json would be rejected as degrees. The name and the
    episodes hold no geometry and pass through unchanged. An item's frame and rectangle move with the rest."""
    c, s = math.cos(yaw), math.sin(yaw)

    def move(x: float, y: float) -> tuple[float, float]:
        return c * x - s * y + tx, s * x + c * y + ty

    def turn(angle: float) -> float:
        return wrap_angle(angle + yaw)

    def moved(polygon: PolygonModel) -> PolygonModel:
        return PolygonModel(polygon=[move(x, y) for x, y in polygon.polygon])

    def moved_target(target: Target) -> Target:
        update: dict = {"anchor": (*move(*target.anchor[:2]), target.anchor[2])}
        if target.footprint is not None:
            update["footprint"] = [move(x, y) for x, y in target.footprint]
        if target.front_yaw is not None:
            update["front_yaw"] = turn(target.front_yaw)
        return target.model_copy(update=update)

    def moved_figure(figure: Figure) -> Figure:
        x, y = move(figure.x, figure.y)
        return replace(figure, x=x, y=y, yaw=turn(figure.yaw))

    def moved_person(person: Person) -> Person:
        x, y = move(person.x, person.y)
        return person.model_copy(update={"x": x, "y": y, "yaw": turn(person.yaw)})

    def moved_item(item: Item) -> Item:              # its frame and its rectangle: the range and the options stay
        x, y = move(item.x, item.y)
        return replace(item, x=x, y=y, yaw=turn(item.yaw), footprint=[move(*corner) for corner in item.footprint])

    gt = layout.scene_gt
    scene_gt = gt.model_copy(update={"people": [moved_person(person) for person in gt.people],
                                     "targets": [moved_target(target) for target in gt.targets],
                                     "bounds": moved(gt.bounds),
                                     "obstacles": [moved(obstacle) for obstacle in gt.obstacles]})
    robot = RobotPose(*move(layout.robot.x, layout.robot.y), turn(layout.robot.yaw))
    room = layout.room and replace(layout.room, polygon=tuple(move(x, y) for x, y in layout.room.polygon))
    return Layout(layout.name, tuple(_moved(part, c, s, tx, ty, yaw) for part in layout.boxes),
                  tuple(moved_figure(figure) for figure in layout.figures), robot, scene_gt, layout.episodes, room,
                  tuple(_moved(part, c, s, tx, ty, yaw) for part in layout.props),
                  tuple([move(x, y) for x, y in footprint] for footprint in layout.footprints),
                  tuple(moved_item(item) for item in layout.items))
