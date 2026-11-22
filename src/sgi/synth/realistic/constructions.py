"""How each kind of item of a layout is built for the realistic set.

construct takes one item of a placeholder layout, with the primitives the layout draws it with, and gives
its construction: the parts and the library models the realistic scene draws in that spot, what of them
blocks the floor, where a staff member may stand, which rectangles carry lettering, which tops are free for
small things, and what is cut into the wall. A builder lays its item out in the item's own frame (a place:
the ground point under its anchor, +x out of its front; a furnishing: its point on the wall, +x into the
room); everything of a construction is in the world, as the primitives are.

The restyle builds every kind that has no builder of its own, and most of every kind that has one: each
primitive of the item becomes a part of the material its colour stands for, a band of a second colour a
piece of its own, a panel a thin plate on its face, lettering a sign. A scene so built has the geometry of
its twin, and it is built whatever the library holds. A part keeps its colour as a tint where its material
takes any colour and would by itself come out far from that one (_OWN): a pillar is pale stone, as its twin
is pale, and the plinth of a booth is darker than its walls.

Four kinds have someone behind them (STAFFED_KINDS): the three places a scene can be about, and the second
stall. The placeholder fills the ground of the three with a counter 0.6 m deep, a desk and a chair. Here the
front face with the anchor on it, the sign overhead and the ground rectangle stay, and behind the front the
counter is no deeper than 0.36 m and the fittings stand aside, so that a strip of floor is free for a body
0.2 m in radius: between the counter and the back wall of a stall or a booth, between the desk and the back
of its ground, where a deep place has its back room. The second stall (the kind "shop", a furnishing on a
wall that no request names) is built as the placeholder has it, which leaves such a strip between its back
board and its counter. A staffed kind lists what it draws as its obstacles, and its staff spot on the strip.
Every other kind blocks what its twin blocks.

Signage has two modes (SIGN_MODES). "labels" keeps the placeholder's words on the placeholder's boards, at
its size. "natural", which is what a real place has, spells nothing out in large letters: a stall keeps its
header board with a made-up logo, a booth gets a small plate, a kiosk a pictogram, a desk its pictogram on a
cube hung over it and on its front with a card on its ledge, and what the placeholder writes on a wall becomes
a menu, a map, an exit sign or sheets of notices. Construction.lettering lists what names the item, in either
mode; a menu, a map or a notice does not.

A plant, a bench, a water cooler and lockers are a model of the library where one goes into the item's place
at between 0.85 and 1.15 of its life size, and so are the screen, the chair, the machine and the cups of a
desk and a stall. Where no machine goes onto a stall's shallow counter, one is built of parts in the box the
placeholder gives it (_machine), and where no chair goes behind a desk, a swivel chair is built of parts on the
square the placeholder's stands on (_task_chair). A booth has a light fitting under its sign. The goods of the
second stall are built by what its sign says it sells (_GOODS), at the placeholder's spots along its counter
and in the placeholder's colours. A window and a door are openings for the room to cut, and draw nothing into
it.

construct takes one number from the generator it is given, whatever the kind, the signage and the library,
so one stream of draws gives every variant of a scene the same constructions.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, replace
from random import Random
from typing import Callable, Sequence

import numpy as np

from sgi.geometry import wrap_angle
from sgi.polygons import Polygon
from sgi.synth.realistic.library import Library
from sgi.synth.realistic.spec import PAINT, ModelRef, Opening, Part, Sign, SignArt
from sgi.synth.recipes import Item
from sgi.synth.render import _FACES, Box, Color, Cylinder, Solid, Sphere

STAFFED_KINDS = ("stall", "desk", "ticket_counter", "shop")
SIGN_MODES = ("natural", "labels")

# What a place is laid out on, as sgi.synth.recipes lays it out: its front on x = 0 of its frame.
_PLACE_FRONT_M = 0.8                               # how deep the ground of its front item is, behind the front
_OVERHEAD_M = 1.85                                 # above every head: the boards of the sign over a place begin higher
_COUNTER_M = 0.30                                  # how deep a counter is built here; the strip behind it is then
#                                                    0.42 m, which holds a body with a centimetre to spare each way
_COUNTER_TOP_M = 1.10                              # as the placeholder's: the anchor is 0.1 m under it
_DESK_M = 0.36                                     # from the front of a desk to the back of its top
_DESK_TOP_M = 0.75
_CEILING_M = 3.0                                   # how high sgi.synth.recipes makes every room: what hangs, hangs
#                                                    from there
_HUNG_M, _HUNG_Z_M = 0.35, 1.90                    # of a sign hung over a place: the side of the cube, and its
#                                                    underside, which is above every head. Its top is then at 2.25 m,
#                                                    as high as the placeholder builds a place and no higher: what
#                                                    is drawn of a target fits the view the robot has of its twin
_FRONT_SYMBOL_M = 0.24                             # the side of the pictogram on a desk's front: the symbol in it is
#                                                    then 0.2 m high, and is made out at 5 m
_LAMP = (255, 248, 232)                            # of a light fitting: the colour of what glows
_LAMP_GLOW = 12.0                                  # and how strongly: a lamp, where a lit screen glows with 1.0
_SCALE = (0.85, 1.15)                              # of its life size, the least and the most a model is used at
_PROUD_M = 0.003                                   # how far a plate stands out of the face it is on, and into it
_LABEL_FILLS = (0.86, 0.62)                        # of a label's rectangle, the width and the height that signs.draw
#                                                    letters; recipes._place_sign letters a board so too
_MARK_FILLS = 0.9                                  # of the face it is on, the most a small sign takes each way
_WHOLE = (0.0, 0.0, 1.0, 1.0)
_A3_M = (0.297, 0.42)                              # a sheet of a notice, upright

_WOOD, _DARK_WOOD, _STEEL, _CREAM = (156, 112, 74), (92, 66, 48), (168, 172, 178), (240, 234, 220)
_CHARCOAL, _PAPER, _SLATE = (54, 56, 60), (244, 240, 228), (88, 96, 112)
_INK, _PLATE = (40, 44, 52), (236, 234, 226)       # of a plate, a notice and a price list: dark type on pale ground
_BOARD = (38, 44, 42)                              # of a menu board, which is lettered in cream
_SCREEN_INK, _SCREEN = (226, 236, 244), (22, 30, 44)
_SHEETS = (_PAPER, (246, 226, 130), (176, 208, 234), (204, 232, 184), (240, 176, 176))
_LEAVES = ((52, 120, 62), (66, 140, 70), (44, 104, 56))        # of a bunch of flowers, as the placeholder's plants

# The small sign that names a kind in natural mode: its style, its width and its height. None is wider than 0.5 m.
_MARKS = {"stall": ("logo", 0.5, 0.25), "shop": ("logo", 0.5, 0.25), "ticket_counter": ("plate", 0.4, 0.16),
          "back_room": ("plate", 0.4, 0.16), "desk": ("plate", 0.32, 0.14), "kiosk": ("pictogram", 0.24, 0.24)}
_PICTOGRAMS = {"PRINT": "printer", "HELP": "information", "TICKETS": "ticket", "COFFEE": "cup"}    # by a place's words
# What a colour of a kind is made of, where the rule of _surface would take it for something else.
_ROLES: dict[str, dict[Color, str]] = {
    "pillar": {(170, 164, 152): "stone", (214, 208, 196): "stone"},
    "bench": {(60, 60, 64): "painted_metal"},
    "cooler": {(236, 238, 240): "plastic_light"},
    "extinguisher": {(238, 238, 232): "plastic_light"},
    "kiosk": {(216, 216, 210): "plastic_light", (248, 248, 242): "paper"},
    "back_room": {(206, 200, 188): PAINT},
    "parcel_table": {(176, 140, 96): "cardboard", (120, 92, 60): PAINT, (72, 66, 62): "painted_metal",
                     (232, 228, 216): "ceramic"},
    "board": {(196, 158, 108): "cork"},
    "shop": {(150, 110, 84): PAINT, (84, 78, 74): "stone"},
    "desk": {(190, 96, 60): "fabric", _PAPER: "paper"},
    "ticket_counter": {_PAPER: "paper"},
}
# What a material comes out in where a part gives it no tint, for the materials that a tint colours outright:
# those the library makes in code and those whose texture it calls neutral. It is the `colour` of each in the
# manifest, and a test holds the two equal. A part of one of them whose own colour is _FAR or more from that, on
# some channel of 255, keeps its colour as its tint: a pillar is pale and not the dark stone of a counter's
# top, and the frame of a booth is darker than the paint of its walls. Any other material has a colour of its
# own, which a tint could only darken: wood stays the library's wood.
_OWN: dict[str, Color] = {
    "painted_metal": (92, 104, 120), "plastic_dark": (48, 50, 54), "plastic_light": (214, 214, 210),
    "laminate_white": (236, 236, 232), "paper": (244, 242, 232), "ceramic": (240, 238, 232), "fabric": (88, 104, 132),
    "stone": (78, 78, 79), "wall_paint": (233, 227, 213)}
_FAR = 40


@dataclass(frozen=True)
class Lettered:
    """A rectangle that carries lettering, in the world: upright, `width` along `across` and `height` up."""
    centre: tuple[float, float, float]
    normal: tuple[float, float, float]             # out of the face it is on; level
    across: tuple[float, float, float]             # the way it reads, from its left to its right; level. Up is +z
    width: float
    height: float


@dataclass(frozen=True)
class Surface:
    """A free top where small things may stand: a level rectangle of a construction with nothing of that
    construction on it or within 0.3 m over it. Where the construction has a staff spot, no point of the
    rectangle is nearer than 0.45 m to it, which is as far as a pose reaches from the body's axis."""
    x: float                                       # its middle, in the world
    y: float
    yaw: float                                     # of the item it is on
    z: float                                       # its height
    length: float                                  # across that yaw
    depth: float                                   # along it


@dataclass(frozen=True)
class Construction:
    parts: tuple[Part, ...]
    models: tuple[ModelRef, ...]
    obstacles: tuple[Polygon, ...]                 # what blocks the floor, in the world: each a list of corners
    staff_spots: tuple[tuple[float, float, float], ...]        # x, y and the yaw to face, out of the front
    lettering: tuple[Lettered, ...]                # what names the item: a label, or a logo, a plate, a pictogram
    surfaces: tuple[Surface, ...]
    openings: tuple[Opening, ...] = ()


@dataclass(frozen=True)
class _Art:
    """Sign art over a box's panel, or over its label face without one: what a place shows, not what names it."""
    style: str
    text: str = ""
    ink: Color = _INK
    ground: Color | None = None                    # None: the colour of what it lies on
    emission: float = 0.0
    seed: int = 0
    rect: tuple[float, float, float, float] = _WHOLE           # the part of the panel or the face that it fills


@dataclass(frozen=True)
class _Mark:
    """A small sign that names its item, on a box's panel or on its label face without one."""
    style: str
    text: str
    width: float                                   # in metres; less on a face that is too small for it
    height: float
    ink: Color
    ground: Color | None = None                    # None: the colour of what it lies on
    seed: int = 0
    at: tuple[float, float] = (0.5, 0.5)           # its middle, as shares of the panel or the face
    emission: float = 0.0


def _moved(solid: Solid, tx: float, ty: float, yaw: float) -> Solid:
    """The solid turned by yaw about the origin, then shifted."""
    c, s = math.cos(yaw), math.sin(yaw)
    if isinstance(solid, Box):
        return replace(solid, cx=c * solid.cx - s * solid.cy + tx, cy=s * solid.cx + c * solid.cy + ty,
                       yaw=wrap_angle(solid.yaw + yaw))
    return replace(solid, x=c * solid.x - s * solid.y + tx, y=s * solid.x + c * solid.y + ty)


def _extent(solids: Sequence[Solid]) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of the ground rectangle, square to the axes, that holds every one of the solids."""
    xs, ys = [], []
    for solid in solids:
        if isinstance(solid, Box):
            c, s = math.cos(solid.yaw), math.sin(solid.yaw)
            for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                xs.append(solid.cx + c * i * solid.sx / 2 - s * j * solid.sy / 2)
                ys.append(solid.cy + s * i * solid.sx / 2 + c * j * solid.sy / 2)
        else:
            reach = solid.radius if isinstance(solid, Sphere) else max(solid.radius, solid.top_radius or 0.0)
            xs += [solid.x - reach, solid.x + reach]
            ys += [solid.y - reach, solid.y + reach]
    return min(xs), min(ys), max(xs), max(ys)


def _bottom(solid: Solid) -> float:
    return solid.z - solid.radius if isinstance(solid, Sphere) else solid.z0


def _top(solid: Solid) -> float:
    if isinstance(solid, Sphere):
        return solid.z + solid.radius
    return solid.z0 + (solid.sz if isinstance(solid, Box) else solid.height)


def _bevel(solid: Solid) -> float:
    """How far a part's edges are rounded off: 4 mm, and on a thin part a fifth of its least side."""
    if isinstance(solid, Sphere):
        return 0.0
    least = min(solid.sx, solid.sy, solid.sz) if isinstance(solid, Box) else min(2 * solid.radius, solid.height)
    return round(min(0.004, 0.2 * least), 4)


def _surface(colour: Color) -> str:
    """The material that one of the placeholder's flat colours stands for, or PAINT where the colour is the
    point: wood for the browns, metal for the blue-greys and the mid greys, a pale laminate or a dark plastic
    for what is nearly white or nearly black."""
    r, g, b = colour
    high, low = max(colour), min(colour)
    vivid = (high - low) / max(high, 1)
    if r > g > b and 0.22 <= vivid <= 0.62 and 60 <= high <= 200 and 0.6 <= g / r <= 0.9:
        return "wood_dark" if high < 120 else "wood"
    if vivid < 0.3 and high <= 130 and b - r >= 8:
        return "painted_metal"
    if vivid < 0.16:
        if high >= 200:
            return "laminate_white"
        if high >= 120 and b >= r:
            return "steel"
        if high < 100:
            return "plastic_dark"
    return PAINT


def _tint(colour: Color, material: str) -> Color | None:
    """The tint of a part of this colour and material: the colour, where the material takes any colour and
    would come out far from this one by itself; None where it would look the same without, and for a material
    that a tint does not colour (see _OWN)."""
    own = _OWN.get(material)
    if own is None or max(abs(int(level) - mine) for level, mine in zip(colour, own)) < _FAR:
        return None
    return tuple(int(level) for level in colour)


def _contrasting(ground: Color) -> Color:
    """The colour the placeholder letters in when a box names none: dark on a light ground, light on a dark one."""
    return (20, 20, 20) if sum(ground) / 3 >= 110 else (245, 245, 245)


def _upright(face: str) -> tuple[int, float, int, float]:
    """Of an upright face of a box: the box axis of its normal and the normal's sign, and the axis and the
    sign of the way the face reads. As sgi.synth.render._FACES, which the renderers share."""
    normal, sign, across, way, _, _ = _FACES[face]
    return normal, sign, across, way


def _in(area: tuple[float, float, float, float], rect: tuple[float, float, float, float]) -> tuple[float, ...]:
    """A rectangle given as shares of an area, as shares of what the area is given in."""
    left, bottom, right, top = area
    return (left + rect[0] * (right - left), bottom + rect[1] * (top - bottom),
            left + rect[2] * (right - left), bottom + rect[3] * (top - bottom))


def _about(at: tuple[float, float], across: float, up: float) -> tuple[float, float, float, float]:
    """The rectangle `across` wide and `up` high about `at`, moved no further than it takes to lie within 0 to 1."""
    x = min(max(at[0], across / 2), 1.0 - across / 2)
    y = min(max(at[1], up / 2), 1.0 - up / 2)
    return x - across / 2, y - up / 2, x + across / 2, y + up / 2


def _measures(box: Box, face: str) -> tuple[float, float]:
    """The width and the height of an upright face of a box."""
    return (box.sy if _upright(face)[2] == 1 else box.sx), box.sz


def _lettered(box: Box, face: str, rect: tuple[float, float, float, float]) -> Lettered:
    """The rectangle `rect` of an upright face of a box, given as Box.panel_rect gives one, in the world."""
    normal, sign, across, way = _upright(face)
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    axes, size = ((c, s), (-s, c)), (box.sx, box.sy)
    width, height = _measures(box, face)
    out = sign * size[normal] / 2                    # from the middle of the box to the face
    along = way * ((rect[0] + rect[2]) / 2 - 0.5) * width
    centre = (box.cx + out * axes[normal][0] + along * axes[across][0],
              box.cy + out * axes[normal][1] + along * axes[across][1],
              box.z0 + (rect[1] + rect[3]) / 2 * box.sz)
    return Lettered(centre, (sign * axes[normal][0], sign * axes[normal][1], 0.0),
                    (way * axes[across][0], way * axes[across][1], 0.0), (rect[2] - rect[0]) * width,
                    (rect[3] - rect[1]) * height)


def _plate(box: Box, face: str, rect: tuple[float, float, float, float], colour: Color) -> Box:
    """A thin box over the rectangle `rect` of an upright face of a box, half of it in the face and half out,
    so that no face of the two lies in one plane. Its own face of that name is the one that shows."""
    spot = _lettered(box, face, rect)
    thin = (2 * _PROUD_M, spot.width) if _upright(face)[0] == 0 else (spot.width, 2 * _PROUD_M)
    return Box(spot.centre[0], spot.centre[1], box.yaw, *thin, spot.height, z0=spot.centre[2] - spot.height / 2,
               color=colour)


def _pieces(box: Box) -> list[tuple[Box, bool, float, float]]:
    """A box without its lettering and its panel, cut level where its band begins and ends: each piece, whether
    it is the band, and the shares of the box's height it reaches from and to. A box without a band is one piece."""
    low, high = box.band_range if box.band is not None else (0.0, 0.0)
    cuts = [(0.0, low, False), (low, high, True), (high, 1.0, False)]
    found = []
    for begin, end, band in cuts:
        if end - begin <= 1e-9:
            continue
        found.append((Box(box.cx, box.cy, box.yaw, box.sx, box.sy, (end - begin) * box.sz, z0=box.z0 + begin * box.sz,
                          color=box.band if band else box.color, top=box.top if end >= 1.0 - 1e-9 else None),
                      band, begin, end))
    return found


def _fitted(size: tuple[float, float, float], box: tuple[float, float, float],
            roofed: bool = False) -> tuple[float, float, float] | None:
    """The scale, side by side, at which a model of this life size goes into a box (along its front, across,
    up), or None when it does not go in at 0.85 to 1.15 of life size.

    Each side is brought to the box's by 15 % at the most, either way. On the ground the model has to end
    inside the box, so one that would have to shrink by more does not go in. Up, the box is as a rule the
    height the placeholder gives the thing, which a real one need not have: the model may end 15 % over it.
    Under something, `roofed`, it has to end inside the box that way too."""
    need = [side / own for side, own in zip(box, size)]
    least, most = _SCALE
    if need[0] < least - 1e-9 or need[1] < least - 1e-9 or need[2] * (1.0 if roofed else most) < least - 1e-9:
        return None
    return tuple(min(max(side, least), most) for side in need)


class _Build:
    """A construction in the making: the item it is for, and what its builder has put so far."""

    def __init__(self, item: Item, natural: bool, library: Library, group: int, seed: int):
        self.item, self.natural, self.library, self.group, self.seed = item, natural, library, group, seed
        self.parts: list[Part] = []
        self.models: list[ModelRef] = []
        self.obstacles: list[Polygon] = []
        self.staff_spots: list[tuple[float, float, float]] = []
        self.lettering: list[Lettered] = []
        self.surfaces: list[Surface] = []
        self.openings: list[Opening] = []
        self._numbers = 0

    def number(self) -> int:
        """Another number for art whose content is drawn: the next of a run that the item's one draw began."""
        self._numbers += 1
        return (self.seed + 7919 * self._numbers) % 2 ** 31

    # --- The item's frame. ---

    def out(self, solid: Solid) -> Solid:
        """A solid laid out in the item's frame, where it stands in the world."""
        return _moved(solid, self.item.x, self.item.y, self.item.yaw)

    def local(self, solid: Solid) -> Solid:
        """A solid of the world, in the item's frame."""
        return _moved(_moved(solid, -self.item.x, -self.item.y, 0.0), 0.0, 0.0, -self.item.yaw)

    def point(self, x: float, y: float) -> tuple[float, float]:
        c, s = math.cos(self.item.yaw), math.sin(self.item.yaw)
        return self.item.x + c * x - s * y, self.item.y + s * x + c * y

    def ground(self) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) of the item's ground rectangle, in its frame."""
        c, s = math.cos(self.item.yaw), math.sin(self.item.yaw)
        own = [(c * (x - self.item.x) + s * (y - self.item.y), c * (y - self.item.y) - s * (x - self.item.x))
               for x, y in self.item.footprint]
        xs, ys = zip(*own)
        return min(xs), min(ys), max(xs), max(ys)

    # --- What is not drawn. ---

    def blocked(self, polygon: Sequence[tuple[float, float]]) -> None:
        """An obstacle, as a polygon of the world: its corners are kept as they are given."""
        self.obstacles.append([(float(x), float(y)) for x, y in polygon])

    def blocked_at(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """An obstacle, as a rectangle of the item's frame."""
        self.blocked([self.point(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])

    def staff(self, x: float, y: float) -> None:
        """Where a staff member stands, in the item's frame; they face out of its front."""
        self.staff_spots.append((*self.point(x, y), self.item.yaw))

    def free_top(self, x0: float, y0: float, x1: float, y1: float, z: float) -> None:
        """A surface, as a rectangle of the item's frame at the height z."""
        self.surfaces.append(Surface(*self.point((x0 + x1) / 2, (y0 + y1) / 2), self.item.yaw, z, y1 - y0, x1 - x0))

    # --- What is drawn. Solids are given in the world. ---

    def material(self, colour: Color) -> str:
        return _ROLES.get(self.item.kind, {}).get(colour) or _surface(colour)

    def part(self, solid: Solid, material: str, signs: Sequence[Sign] = (), glass: bool = False) -> None:
        """A part of the material, in the solid's own colour where the material would come out far from it."""
        self.parts.append(Part(solid, material, self.group, _bevel(solid), tuple(signs), glass,
                               _tint(solid.color, material)))

    def solid(self, solid: Solid, material: str | None = None) -> None:
        """One primitive restyled."""
        if isinstance(solid, Box):
            self.box(solid, material)
        else:
            self.part(solid, material or self.material(solid.color))

    def box(self, box: Box, material: str | None = None, *, art: _Art | None = None, mark: _Mark | None = None,
            panel: bool = True) -> None:
        """One box restyled: a part for each piece of it, a plate for its panel, and a sign.

        The sign is the box's own lettering in labels mode, as a label of the placeholder's size, which is then
        listed as lettering. Where the box has none, and in natural mode, it is `art` over the panel, or `mark`
        on it, which is listed as lettering too; with neither the panel is bare. A box without a panel takes
        the sign on its label face, and `panel` False builds a box that has one as if it had none.

        A sign's ground is its own picture's. Where that is the colour of what it lies on, a label or a mark
        on a bare board, the board is painted in that colour and takes no material, so that no edge shows."""
        face = box.label_face
        framed = box.panel is not None and panel
        area = box.panel_rect if framed else _WHOLE
        on = box.panel if framed else box.color
        width, height = _measures(box, face) if face != "top" else (box.sy, box.sx)
        width, height = (area[2] - area[0]) * width, (area[3] - area[1]) * height
        drawn: tuple[SignArt, tuple[float, float, float, float], float] | None = None    # art, where in the area, glow
        own_ground = False                             # whether the sign's ground is the colour it lies on
        if box.label and not self.natural:
            across, up = min(1.0, box.label_width / _LABEL_FILLS[0]), min(1.0, box.label_height / _LABEL_FILLS[1])
            ink = box.label_color if box.label_color is not None else _contrasting(on)
            picture = SignArt("label", box.label, ink, on, across * width, up * height)
            drawn, own_ground = (picture, _about((0.5, 0.5), across, up), 0.0), True
            if face != "top":                          # lettering that lies flat has no upright rectangle
                self.lettering.append(_lettered(box, face, _in(area, _about((0.5, 0.5), box.label_width,
                                                                              box.label_height))))
        elif art is not None:
            left, bottom, right, top = art.rect
            picture = SignArt(art.style, art.text, art.ink, art.ground if art.ground is not None else on,
                              (right - left) * width, (top - bottom) * height, art.seed)
            drawn, own_ground = (picture, art.rect, art.emission), art.ground is None
        elif mark is not None:
            fits = min(1.0, _MARK_FILLS * width / mark.width, _MARK_FILLS * height / mark.height)
            across, up = fits * mark.width, fits * mark.height
            where = _about(mark.at, across / width, up / height)
            picture = SignArt(mark.style, mark.text, mark.ink, mark.ground if mark.ground is not None else on,
                              across, up, mark.seed)
            drawn, own_ground = (picture, where, mark.emission), mark.ground is None
            if face != "top":
                self.lettering.append(_lettered(box, face, _in(area, where)))
        plated = framed and face != "top"
        pieces = _pieces(box)
        # The piece that takes a sign on the box itself: the topmost for a top, else the tallest that is no band.
        main = pieces[-1] if face == "top" else max(pieces, key=lambda piece: (not piece[1], piece[3] - piece[2]))
        bare = own_ground and drawn is not None and not plated      # a sign on the box itself, of the box's colour
        for piece in pieces:
            solid, band, begin, end = piece
            signs = ()
            if piece is main and drawn is not None and not plated:
                left, bottom, right, top = _in(area, drawn[1])
                if face != "top":                      # as shares of this piece's own height
                    bottom, top = ((share - begin) / (end - begin) for share in (bottom, top))
                    bottom, top = max(0.0, bottom), min(1.0, top)
                signs = (Sign(face, drawn[0], (left, bottom, right, top), drawn[2]),)
            if band:
                self.part(solid, self.material(solid.color))
            else:
                self.part(solid, PAINT if bare else material or self.material(solid.color), signs)
        if plated:
            plate = _plate(box, face, area, box.panel)
            if drawn is None:
                self.part(plate, self.material(box.panel))
            else:
                self.part(plate, PAINT, (Sign(face, drawn[0], drawn[1], drawn[2]),))

    def model(self, kind: str, x: float, y: float, z: float, box: tuple[float, float, float], turned: bool = False,
              against: float | None = None, roofed: bool = False) -> bool:
        """Stands a library model of the kind in a box of the item's frame, and says whether one went in.

        The box is `box` (along the item's +x, across, up) with its bottom centre at (x, y, z); `roofed`
        says that something is over it (see _fitted). The model faces +x, or -x when `turned`. It stands in
        the middle of the box, or with `against` its back at that x. Of several that go in, the item's draw
        picks one."""
        fitting = [(model, scale) for model in self.library.models_of(kind)
                   if (scale := _fitted(model.size, box, roofed)) is not None]
        if not fitting:
            return False
        model, scale = fitting[self.seed % len(fitting)]
        size = tuple(side * own for side, own in zip(scale, model.size))
        if against is not None:
            x = against + size[0] / 2
        yaw = wrap_angle(self.item.yaw + (math.pi if turned else 0.0))
        self.models.append(ModelRef(model.id, *self.point(x, y), z, yaw, scale, size, self.group))
        return True

    def mark(self) -> _Mark:
        """The small sign that names the item in natural mode: a logo, a plate or a pictogram, from
        what the placeholder writes on it.

        Its seed, which picks a logo's made-up name, comes from the words and the frame, not from the draw:
        the front of a deep place and the room behind it are two items in one frame, and show one name."""
        style, width, height = _MARKS[self.item.kind]
        text = str(self.item.options.get("text", ""))
        seed = zlib.crc32(f"{text}|{self.item.x:.3f}|{self.item.y:.3f}".encode()) % 2 ** 31
        if style == "pictogram" and text in _PICTOGRAMS:
            return _Mark("pictogram", _PICTOGRAMS[text], width, height, (255, 255, 255), seed=seed)
        if style == "logo":
            return _Mark("logo", text.lower(), width, height, _CREAM, seed=seed)
        if style == "pictogram":                       # no picture for these words: a plate says them small
            width, height = _MARKS["back_room"][1:]
        return _Mark("plate", text.lower(), width, height, _INK, _PLATE, seed)


# --- What a builder looks for among the placeholder's primitives of a place, each in the place's frame. ---

def _overhead(local: Solid) -> str | None:
    """What a primitive is of the sign over a place: "board" for one of its boards, which are above every
    head, "post" for a post that carries them from the floor, or None."""
    if not isinstance(local, Box):
        return None
    if local.z0 >= _OVERHEAD_M:
        return "board"
    return "post" if local.z0 == 0.0 and max(local.sx, local.sy) <= 0.1 and _top(local) >= _OVERHEAD_M else None


def _faces_front(local: Solid) -> bool:
    """Whether a primitive is a slab across the place with its face on the front, x = 0."""
    return isinstance(local, Box) and abs(local.cx + local.sx / 2) < 1e-6 and local.sy > 1.0


def _back_wall(local: Solid) -> bool:
    """Whether a primitive is a wall across the back of the place's ground."""
    return (isinstance(local, Box) and local.z0 == 0.0 and local.sy > 1.0
            and abs(local.cx - local.sx / 2 + _PLACE_FRONT_M) < 1e-6)


def _restyle(b: _Build, primitives: Sequence[Solid]) -> None:
    """The default: the layout's own primitives restyled, on the item's rectangle, which is the twin's obstacle."""
    for solid in primitives:
        b.solid(solid)
    b.blocked(b.item.footprint)


def _counter(b: _Build, width: float, colour: Color, plinth: Color, top: Color, top_material: str) -> None:
    """The counter of a stall or a booth: _COUNTER_M deep behind the front, as high as the placeholder's, with
    its front face on x = 0 from the floor up; and the obstacle under it."""
    body = Box(-_COUNTER_M / 2, 0.0, 0.0, _COUNTER_M, width, _COUNTER_TOP_M - 0.04, color=colour, band=plinth,
               band_range=(0.0, 0.09))
    b.box(b.out(body))
    b.part(b.out(Box(-_COUNTER_M / 2, 0.0, 0.0, _COUNTER_M, width, 0.04, z0=_COUNTER_TOP_M - 0.04, color=top)),
           top_material)
    b.blocked_at(-_COUNTER_M, -width / 2, 0.0, width / 2)


def _machine(b: _Build, y: float, z: float) -> None:
    """The espresso machine of a stall, built of parts where no model goes onto so shallow a counter. It stands
    on the counter at the height z with its middle at y across the place, and fills no more than the box the
    placeholder gives a machine: 0.28 m of the counter's depth, 0.36 m across and 0.36 m up.

    Its face is turned to the place's front, as the placeholder's is, so that whoever comes to the stall sees
    what it is: a drip tray, a body of brushed steel with dark cheeks and a dark front, and a head that hangs over
    the tray with two group heads under it, a handle in each, a steam wand at one side and a small screen on its
    face. Cups are kept warm on its top, rim down."""
    front, deep, across = -0.02, 0.26, 0.34            # its front, 2 cm behind the counter's; its depth, its width
    middle, back = front - deep / 2, front - deep
    column, tray, head = 0.14, 0.04, 0.07              # how deep its body is; how high its tray and its head are
    under = z + 0.23                                   # the underside of the head

    def box(x: float, at: float, sx: float, sy: float, sz: float, z0: float, material: str, colour: Color) -> None:
        b.part(b.out(Box(x, y + at, 0.0, sx, sy, sz, z0=z0, color=colour)), material)

    box(middle, 0.0, deep, across, tray, z, "steel", _STEEL)
    box(back + column / 2, 0.0, column, across - 0.04, under - z - tray, z + tray, "steel", _STEEL)
    for side in (1, -1):                               # the cheeks, flush with the tray and the head
        box(back + column / 2, side * (across / 2 - 0.01), column, 0.02, under - z - tray, z + tray, "plastic_dark",
            _CHARCOAL)
    box(middle, 0.0, deep, across, head, under, "steel", _STEEL)
    box(back + column + 0.005, 0.0, 0.01, across - 0.04, 0.15, z + tray + 0.02, "plastic_dark", _CHARCOAL)   # the front
    for side in (1, -1):                               # the group heads, and the handle of what hangs in each
        b.part(b.out(Cylinder(front - 0.055, y + side * 0.08, 0.026, 0.045, z0=under - 0.045, color=_STEEL)), "steel")
        box(front - 0.0225, side * 0.08, 0.055, 0.02, 0.02, under - 0.04, "plastic_dark", _CHARCOAL)
    b.part(b.out(Cylinder(front - 0.04, y + 0.15, 0.006, 0.12, z0=under - 0.12, color=_STEEL)), "steel")   # the wand
    box(front + 0.003, 0.0, 0.006, 0.07, 0.035, under + 0.0175, "screen", _SCREEN)
    # Cups on its top, in two rows of three with one of the six in use.
    spots = [(front - 0.065 - 0.09 * row, 0.1 * (column_ - 1)) for row in range(2) for column_ in range(3)]
    for number, (x, at) in enumerate(spots):
        if number != b.seed % len(spots):
            b.part(b.out(Cylinder(x, y + at, 0.03, 0.055, z0=under + head, color=(244, 242, 236), top_radius=0.02)),
                   "ceramic")


def _stall(b: _Build, primitives: Sequence[Solid]) -> None:
    """A counter with a machine and cups towards one end and a till towards the other, a back wall with the
    menu on it, which carries the sign overhead, and between the two the strip its staff stand on, in the
    middle of the width: beside the machine, not behind it."""
    theme, wall, width = str(b.item.options.get("text", "")).lower(), -_PLACE_FRONT_M, 1.6
    for solid, local in zip(primitives, map(b.local, primitives)):
        if _overhead(local):                           # the header: a logo in natural mode where its name was
            b.box(solid, mark=b.mark() if solid.label else None)
        elif _back_wall(local):
            b.box(solid, art=_Art("menu", theme, _CREAM, seed=b.number()))
            b.blocked_at(*_extent([local]))
            wall, width = local.cx + local.sx / 2, local.sy
        # The counter and what stands on it are built anew.
    _counter(b, width, _WOOD, _DARK_WOOD, (66, 58, 54), "stone")
    z, end = _COUNTER_TOP_M, width / 2
    if not b.model("espresso_machine", -_COUNTER_M / 2, end - 0.19, z, (_COUNTER_M - 0.02, 0.36, 0.36)):
        _machine(b, end - 0.19, z)
    for k in range(3):                                 # cups, between the machine and the middle
        y = end - 0.44 - 0.12 * k
        if not b.model("cup", -0.07, y, z, (0.105, 0.105, 0.12)):
            b.part(b.out(Cylinder(-0.07, y, 0.04, 0.1, z0=z, color=(250, 250, 246), top=(96, 60, 36))), PAINT)
    till = Box(-_COUNTER_M / 2, 0.15 - end, 0.0, _COUNTER_M - 0.04, 0.26, 0.22, z0=z, color=(70, 72, 78),
               panel=(120, 200, 230), panel_rect=(0.12, 0.3, 0.88, 0.9))
    b.box(b.out(till), "plastic_dark", art=_Art("screen", "", _SCREEN_INK, _SCREEN, 1.0))
    b.staff((wall - _COUNTER_M) / 2, 0.0)
    b.free_top(-0.12, 0.3 - end, -0.01, 0.57 - end, z)       # the front of the counter beside the till


def _ticket_counter(b: _Build, primitives: Sequence[Solid]) -> None:
    """A booth: a ledge with a glass screen along its back edge and a hatch in the screen at counter height,
    side walls that are solid up to the ledge and glazed above, a back wall that carries the sign overhead,
    and inside the strip its staff stand on, with a light over it."""
    theme, wall, width = str(b.item.options.get("text", "")).lower(), -_PLACE_FRONT_M, 1.6
    for solid, local in zip(primitives, map(b.local, primitives)):
        if _overhead(local):                           # the header: a small plate in natural mode, above the window
            b.box(solid, mark=b.mark() if solid.label else None)
        elif _back_wall(local):
            wall, width = local.cx + local.sx / 2, local.sy
            # In natural mode a price list hangs on it, behind the glass and beside whoever stands at the hatch.
            prices = (0.5 + 0.3 / width, 1.22 / local.sz, 0.5 + 0.66 / width, 1.78 / local.sz)
            b.box(solid, art=_Art("menu", theme, _INK, _PAPER, seed=b.number(), rect=prices) if b.natural else None)
            b.blocked_at(*_extent([local]))
        # The counter, the screen, the side walls and what lies on the ledge are built anew.
    _counter(b, width, _SLATE, (52, 56, 66), (212, 212, 206), "laminate_white")
    z, end, frame, head = _COUNTER_TOP_M, width / 2, (60, 62, 70), 1.82
    glass = (186, 216, 232)
    at = -_COUNTER_M + 0.015                           # the middle of the screen, which is 3 cm of the ledge's back
    bay = 0.26                                         # from the middle of the booth to the middle of a mullion

    def bar(x: float, y: float, sx: float, sy: float, sz: float, z0: float) -> None:
        b.part(b.out(Box(x, y, 0.0, sx, sy, sz, z0=z0, color=frame)), "painted_metal")

    def pane(x: float, y: float, sx: float, sy: float, sz: float, z0: float) -> None:
        b.part(b.out(Box(x, y, 0.0, sx, sy, sz, z0=z0, color=glass)), "glass", glass=True)

    bar(at, 0.0, 0.03, width - 0.08, 0.08, head)       # the head of the screen, which reaches up to the sign
    bar(at, 0.0, 0.03, 2 * bay - 0.03, 0.03, z + 0.2)  # over the hatch
    pane(at, 0.0, 0.01, 2 * bay - 0.03, head - z - 0.23, z + 0.23)
    for side in (1, -1):
        bar(at, side * (end - 0.02), 0.03, 0.04, head + 0.08 - z, z)
        bar(at, side * bay, 0.03, 0.03, head - z, z)
        pane(at, side * (bay + end - 0.04 + 0.015) / 2, 0.01, end - 0.04 - bay - 0.015, head - z, z)
        # A side wall, from the ledge to the back wall: solid up to the ledge's top and glazed above.
        length, middle = -_COUNTER_M - wall, (wall - _COUNTER_M) / 2
        b.part(b.out(Box(middle, side * (end - 0.03), 0.0, length, 0.06, z, color=_SLATE)), "painted_metal")
        bar(middle, side * (end - 0.03), length, 0.06, 0.08, head)
        pane(middle, side * (end - 0.03), length, 0.01, head - z, z)
        b.blocked_at(wall, min(side * end, side * (end - 0.06)), -_COUNTER_M, max(side * end, side * (end - 0.06)))
    # A light fitting inside, hung from the roof of the sign just behind the screen: a baffle with a strip along
    # its foot that glows towards whoever stands in the booth, so that they are seen through the glass from the
    # front and from the sides. It is above every head, and the sign's boards hide it from most of the room.
    baffle = Box(at - 0.035, 0.0, 0.0, 0.03, width - 0.5, 0.3, z0=_OVERHEAD_M + 0.06, color=frame)
    strip = (0.02, 0.03, 0.98, 0.25)
    glow = SignArt("paper", "", _LAMP, _LAMP, (strip[2] - strip[0]) * baffle.sy, (strip[3] - strip[1]) * baffle.sz)
    b.part(b.out(baffle), "painted_metal", (Sign("back", glow, strip, _LAMP_GLOW),))
    b.part(b.out(Box(-0.14, 0.0, 0.0, 0.24, 0.3, 0.02, z0=z, color=_STEEL)), "steel")       # the tray under the hatch
    b.box(b.out(Box(-0.13, 0.5, 0.0, 0.12, 0.26, 0.2, z0=z, color=_PAPER, band=b.item.options.get("color", _SLATE),
                    band_range=(0.6, 1.0))))                                                  # leaflets
    b.part(b.out(Box(-0.12, -0.42, 0.0, 0.14, 0.1, 0.06, z0=z, color=(40, 42, 48))), "plastic_dark")    # a card reader
    b.staff((wall - _COUNTER_M) / 2, 0.0)
    b.free_top(-0.24, 0.04 - end, -0.02, 0.3 - end, z)        # the ledge before the glass, at one end


def _symbol(b: _Build, side: float, at: tuple[float, float]) -> _Mark | None:
    """The pictogram of a place's words as a mark `side` metres square, white on the place's own colour and
    without a word; or None for words that no pictogram stands for."""
    name = _PICTOGRAMS.get(str(b.item.options.get("text", "")))
    if name is None:
        return None
    colour = tuple(int(level) for level in b.item.options.get("color", _SLATE))
    return _Mark("pictogram", name, side, side, (255, 255, 255), colour, at=at)


def _hanging_symbol(b: _Build, x: float, y: float) -> None:
    """A sign hung from the ceiling over (x, y) of the place's frame, where the place's words have a pictogram:
    a cube in the place's colour, above every head, with the pictogram on its four sides, so that it shows from
    the front and from along the place, which is how a waiting line is seen. It says nothing in letters. Two
    rods carry it, which reach the ceiling and are no part of what the place is (dress leaves them out of the
    target's block); nothing of it stands on the floor, and it blocks none."""
    mark = _symbol(b, _MARK_FILLS * _HUNG_M, (0.5, 0.5))
    if mark is None:
        return
    cube = b.out(Box(x, y, 0.0, _HUNG_M, _HUNG_M, _HUNG_M, z0=_HUNG_Z_M, color=mark.ground))
    art = SignArt("pictogram", mark.text, mark.ink, mark.ground, mark.width, mark.height)
    where = _about((0.5, 0.5), _MARK_FILLS, _MARK_FILLS)
    faces = ("front", "back", "left", "right")
    b.part(cube, PAINT, [Sign(face, art, where) for face in faces])
    b.lettering += [_lettered(cube, face, where) for face in faces]
    for side in (1, -1):
        b.part(b.out(Cylinder(x, y + side * 0.3 * _HUNG_M, 0.006, _CEILING_M - _HUNG_Z_M - _HUNG_M,
                              z0=_HUNG_Z_M + _HUNG_M, color=_STEEL)), "steel")


def _task_chair(b: _Build, ground: tuple[float, float, float, float], cloth: Color) -> None:
    """A swivel chair built of parts on the ground rectangle (x0, y0, x1, y1) of the place's frame, facing +x:
    five spokes on castors, a column, a seat and a low back in the cloth's colour. It has no arms and no head
    rest, and ends under the ledge of a desk, so that from the front nothing of it shows over the desk."""
    x0, y0, x1, y1 = ground
    x, y = (x0 + x1) / 2, (y0 + y1) / 2
    reach = min(x1 - x0, y1 - y0) / 2                  # of the foot, from the column
    for k in range(5):
        turn = (2 * k + 1) * math.pi / 5
        c, s = math.cos(turn), math.sin(turn)
        long = reach - 0.03
        b.part(b.out(Box(x + c * long / 2, y + s * long / 2, turn, long, 0.04, 0.025, z0=0.055, color=_CHARCOAL)),
               "plastic_dark")
        b.part(b.out(Sphere(x + c * (reach - 0.03), y + s * (reach - 0.03), 0.03, 0.03, color=(30, 30, 32))),
               "plastic_dark")
    b.part(b.out(Cylinder(x, y, 0.035, 0.07, z0=0.04, color=_CHARCOAL)), "plastic_dark")
    b.part(b.out(Cylinder(x, y, 0.02, 0.32, z0=0.11, color=_STEEL)), "steel")
    deep, across = min(0.37, 2 * reach - 0.03), min(0.4, y1 - y0 - 0.04)
    b.part(b.out(Box(x + 0.01, y, 0.0, deep, across, 0.03, z0=0.43, color=_CHARCOAL)), "plastic_dark")
    b.part(b.out(Box(x + 0.01, y, 0.0, deep - 0.01, across - 0.01, 0.06, z0=0.46, color=cloth)), "fabric")
    back = x + 0.01 - deep / 2                         # the seat's back edge
    b.part(b.out(Box(back + 0.02, y, 0.0, 0.025, 0.07, 0.3, z0=0.44, color=_CHARCOAL)), "plastic_dark")
    b.part(b.out(Box(back + 0.025, y, 0.0, 0.05, across - 0.02, 0.4, z0=0.62, color=cloth)), "fabric")


def _desk(b: _Build, primitives: Sequence[Solid]) -> None:
    """A desk behind a raised front, as the placeholder's but for its top, which ends _DESK_M behind the front
    and leaves the rest of the place's ground free: the chair stands there at one side, with the screen before
    it, and the staff at the other. In labels mode the sign stands over it on its four posts. In natural mode it
    has none, and says what it is as an information point does, without a word in large letters: its pictogram
    on a cube hung over it, which shows from the front and from both sides, the pictogram again on its front,
    large enough to be made out from across a hall, and a card with its name in small type on the ledge."""
    chair, front, ledge = [], None, None
    for solid, local in zip(primitives, map(b.local, primitives)):
        role = _overhead(local)
        if role:
            if not b.natural:
                b.box(solid)
                if role == "post" and local.cx + local.sx / 2 < -_DESK_M:      # one that stands behind the desk
                    b.blocked_at(*_extent([local]))
        elif _faces_front(local) and local.z0 == 0.0:  # the front, with the anchor on it: as it is
            front = local
            b.box(solid, mark=_symbol(b, _FRONT_SYMBOL_M, (0.5, 0.76)) or replace(b.mark(), at=(0.5, 0.84)),
                  panel=not b.natural)
        elif _faces_front(local):                      # the ledge on it
            ledge = local
            b.box(solid)
        elif (ledge is not None and isinstance(local, Box) and local.z0 >= _top(ledge) - 1e-6
              and local.cx > ledge.cx - ledge.sx / 2):
            b.box(solid)                               # leaflets on the ledge
        elif _top(local) < _OVERHEAD_M and _extent([local])[2] < -_DESK_M - 1e-6:
            chair.append(local)                        # what stands behind the desk is its chair
        # The top, the side panels and the screen are built anew.
    if front is None or ledge is None or not chair:
        raise ValueError("a desk is built from the placeholder's front, the ledge on it and the chair behind it")
    width, thick, pale = front.sy, front.sx, front.color
    deep, middle = _DESK_M - thick, -(thick + _DESK_M) / 2
    b.box(b.out(Box(middle, 0.0, 0.0, deep, width, 0.05, z0=_DESK_TOP_M - 0.05, color=_WOOD)))
    for side in (1, -1):
        b.box(b.out(Box(middle, side * (width / 2 - 0.03), 0.0, deep, 0.06, _DESK_TOP_M - 0.05, color=pale)))
    b.blocked_at(-_DESK_M, -width / 2, 0.0, width / 2)
    # The chair, in the middle of the floor that is left behind the desk.
    x0, y0, x1, y1 = _extent(chair)
    back = b.ground()[0]
    side = 1.0 if (y0 + y1) / 2 < 0 else -1.0          # the staff stand at the other side
    spot = ((back - _DESK_M) / 2, side * 0.33)
    room = (-_DESK_M - 0.02 - back - 0.02, y1 - y0, max(map(_top, chair)))
    if b.model("office_chair", spot[0], (y0 + y1) / 2, 0.0, room):
        placed = b.models[-1]
        b.blocked_at(spot[0] - placed.size[0] / 2, (y0 + y1 - placed.size[1]) / 2, spot[0] + placed.size[0] / 2,
                     (y0 + y1 + placed.size[1]) / 2)
    else:                                              # built of parts, on the square the placeholder's chair fills
        moved = [_moved(local, spot[0] - (x0 + x1) / 2, 0.0, 0.0) for local in chair]
        cloth = next((local.color for local in chair if b.material(local.color) == "fabric"), _CHARCOAL)
        _task_chair(b, _extent(moved), cloth)
        b.blocked_at(*_extent(moved))
    # The screen, turned to the chair and beside the staff, whose hands it would otherwise be in the way of. A
    # model low enough stands under the ledge, which hangs over the desk's top; a taller one, and the
    # placeholder's, on what of the top the ledge leaves clear.
    clear = ledge.cx - ledge.sx / 2
    at = ((clear - _DESK_M) / 2, -side * 0.14)
    under = ((-_DESK_M + 0.01 - thick - 0.03) / 2, at[1], _DESK_TOP_M)
    if not (b.model("monitor", *under, (_DESK_M - thick - 0.04, 0.52, ledge.z0 - _DESK_TOP_M - 0.01), turned=True,
                    roofed=True)
            or b.model("monitor", *at, _DESK_TOP_M, (clear + _DESK_M - 0.02, 0.52, 0.53), turned=True)):
        screen = Box(*at, 0.0, 0.05, 0.52, 0.34, z0=_DESK_TOP_M + 0.19, color=(184, 188, 194), label_face="back",
                     panel=(130, 190, 226), panel_rect=(0.05, 0.07, 0.95, 0.93))
        b.box(b.out(screen), "plastic_light", art=_Art("screen", "", _SCREEN_INK, _SCREEN, 1.0))
        b.part(b.out(Cylinder(*at, 0.03, 0.19, z0=_DESK_TOP_M, color=_CHARCOAL)), "plastic_dark")
        b.part(b.out(Cylinder(*at, 0.07, 0.015, z0=_DESK_TOP_M, color=_CHARCOAL)), "plastic_dark")
    rim = _top(ledge)
    if b.natural:                                      # a name card on the ledge, before the staff
        card = Box(ledge.cx, spot[1], 0.0, 0.02, 0.18, 0.07, z0=rim, color=_PLATE)
        b.box(b.out(card), "plastic_light", mark=replace(b.mark(), width=0.18, height=0.07))
        _hanging_symbol(b, -(_HUNG_M / 2 + 0.01), 0.0)
    b.staff(*spot)
    # Free tops: the ledge to either side of the card, clear of the leaflets, of the staff's reach and of the
    # post that a sign overhead comes up through at the corner; and the end of the desk's own top beyond the screen.
    for y_from, y_to in ((-0.26, 0.06), (0.59, 0.705)):
        b.free_top(clear + 0.01, min(side * y_from, side * y_to), clear + ledge.sx - 0.01,
                   max(side * y_from, side * y_to), rim)
    b.free_top(-_DESK_M + 0.02, min(-side * 0.72, -side * 0.44), -thick - 0.02, max(-side * 0.72, -side * 0.44),
               _DESK_TOP_M)


def _kiosk(b: _Build, primitives: Sequence[Solid]) -> None:
    """A self-service cabinet with nobody at it, restyled: in natural mode its screen shows its own interface
    and its head a small pictogram where the name was."""
    locals_ = [b.local(solid) for solid in primitives]
    for solid, local in zip(primitives, locals_):
        if isinstance(solid, Box) and solid.label and _overhead(local):
            b.box(solid, mark=b.mark())
        elif isinstance(solid, Box) and solid.label:   # the cabinet, lettered on its screen
            b.box(solid, art=_Art("screen", solid.label, _SCREEN_INK, _SCREEN, 1.0))
        else:
            b.solid(solid)
    b.blocked(b.item.footprint)
    # A free top: that of a printer beside the cabinet with nothing lying on it.
    for local in locals_:
        lid = isinstance(local, Box) and local.sz <= 0.05 and 0.5 < local.z0 < 1.3      # thin, at about table height
        if not lid or min(local.sx, local.sy) < 0.3:
            continue
        z, ground = _top(local), _extent([local])
        if not any(other is not local and _bottom(other) < z + 0.3 and _top(other) > z + 1e-6
                   and _overlap(_extent([other]), ground) for other in locals_):
            b.free_top(ground[0] + 0.03, ground[1] + 0.03, ground[2] - 0.03, ground[3] - 0.03, z)


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """Whether two ground rectangles, each (x0, y0, x1, y1), have more ground in common than an edge."""
    return a[0] < b[2] - 1e-6 and b[0] < a[2] - 1e-6 and a[1] < b[3] - 1e-6 and b[1] < a[3] - 1e-6


def _back_room(b: _Build, primitives: Sequence[Solid]) -> None:
    """The closed room behind a deep place, restyled: in natural mode the panel on each side keeps its colour
    and carries a small plate where the name was."""
    for solid in primitives:
        if isinstance(solid, Box) and solid.label:
            b.box(solid, mark=b.mark())
        else:
            b.solid(solid)
    b.blocked(b.item.footprint)


def _parcel_table(b: _Build, primitives: Sequence[Solid]) -> None:
    """A table with parcels on it and a crate beside it, restyled. It blocks what the twin blocks: the table's
    rectangle and the crate's, which are its first and its third primitive, and not the floor between them."""
    for solid in primitives:
        if isinstance(solid, Box) and solid.panel is not None and not solid.label:      # an address label, stuck on
            b.box(solid, art=_Art("paper", "", (60, 60, 70), solid.panel, seed=b.number(), rect=solid.panel_rect),
                  panel=False)
        else:
            b.solid(solid)
    table, crate = primitives[0], primitives[2]
    if not (isinstance(table, Box) and isinstance(crate, Box)):
        raise ValueError("a parcel table is drawn with its top first and its crate third, each a box")
    for box in (table, crate):
        c, s = math.cos(box.yaw), math.sin(box.yaw)
        b.blocked([(box.cx + c * i * box.sx / 2 - s * j * box.sy / 2, box.cy + s * i * box.sx / 2 + c * j * box.sy / 2)
                   for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    # A free top: the far corner of the table from the crate, clear of the parcel that the scene is about.
    x0, y0, _, _ = _extent([b.local(table)])
    b.free_top(x0 + 0.02, y0 + 0.02, x0 + 0.25, y0 + 0.35, _top(table))


def _modelled(b: _Build, primitives: Sequence[Solid]) -> None:
    """A furnishing that the library may have a model of: the model, where one goes into the item's rectangle
    under the placeholder's height, and otherwise the restyle. A plant stands in the middle of its rectangle
    and anything else with its back on the wall's side of it."""
    x0, y0, x1, y1 = b.ground()
    locals_ = [b.local(solid) for solid in primitives]
    box = (x1 - x0, y1 - y0, max(map(_top, locals_)))
    against = None if b.item.kind == "plant" else x0
    if not b.model(b.item.kind, (x0 + x1) / 2, (y0 + y1) / 2, 0.0, box, against=against):
        for solid in primitives:
            b.solid(solid)
        if b.item.kind == "bench":                     # a free top: its seat, the widest of its low boxes
            seat = max((local for local in locals_ if isinstance(local, Box) and _top(local) < 0.6),
                       key=lambda local: local.sx * local.sy)
            sx0, sy0, sx1, sy1 = _extent([seat])
            b.free_top(sx0 + 0.06, sy0 + 0.05, sx1 - 0.02, sy1 - 0.05, _top(seat))
    b.blocked(b.item.footprint)


def _poster(b: _Build, primitives: Sequence[Solid]) -> None:
    """What hangs lettered on a wall. In labels mode, the placeholder's poster with its word. In natural mode,
    by that word: a drawn plan for a map, a small lit sign where the head of the poster was for an exit, and
    for anything else a pin board with sheets of notices."""
    text = str(b.item.options.get("text", ""))
    for solid in primitives:
        if b.natural and isinstance(solid, Box) and solid.label and solid.label_face != "top":
            if text == "MAP":
                b.box(solid, art=_Art("map", "", b.item.options.get("ink", _INK), seed=b.number()))
            elif text == "EXIT":
                local = b.local(solid)
                sign = Box(local.cx, local.cy, local.yaw, local.sx, 0.4, 0.2, z0=_top(local) - 0.2, color=_PLATE)
                b.box(b.out(sign), "plastic_light",
                      mark=_Mark("exit", "", 0.4, 0.2, (255, 255, 255), (30, 140, 80), b.number(), emission=1.0))
            else:
                _notices(b, solid, text.lower())
        else:
            b.solid(solid)
    b.blocked(b.item.footprint)


def _notices(b: _Build, box: Box, theme: str) -> None:
    """A pin board in place of a lettered poster: its frame, a board of cork where its panel was, and on that
    as many upright sheets as it holds, one place left bare."""
    face = box.label_face
    area = box.panel_rect if box.panel is not None else _WHOLE
    b.box(replace(box, label=None, panel=None))
    board = _plate(box, face, area, (196, 158, 108))
    width, height = _measures(board, face)
    columns = max(1, int((width - 0.04) // (_A3_M[0] + 0.04)))
    rows = max(1, int((height - 0.04) // (_A3_M[1] + 0.04)))
    across, up = min(_A3_M[0], 0.9 * width) / width, min(_A3_M[1], 0.9 * height) / height
    bare = b.seed % (rows * columns) if rows * columns >= 4 else None
    sheets = []
    for k in range(rows * columns):
        row, column = divmod(k, columns)
        where = _about(((column + 0.5) / columns, (row + 0.5) / rows), across, up)
        art = SignArt("notice", theme, _INK, _SHEETS[k % len(_SHEETS)], across * width, up * height, b.number())
        if k != bare:
            sheets.append(Sign(face, art, where))
    b.part(board, "cork", sheets)


def _board(b: _Build, primitives: Sequence[Solid]) -> None:
    """A notice board, restyled: each sheet pinned to it is paper with a sheet's picture on it, in natural mode
    every second one a printed notice."""
    for k, solid in enumerate(primitives):
        if isinstance(solid, Box) and solid.panel is None and not solid.label:
            style = "notice" if b.natural and k % 2 else "paper"
            b.box(solid, "paper", art=_Art(style, "notices" if style == "notice" else "", _INK, solid.color,
                                           seed=b.number()))
        else:
            b.solid(solid)
    b.blocked(b.item.footprint)


def _crates(b: _Build, spots: Sequence[tuple[float, float, Color]], z: float, room: float, draws: Random) -> None:
    """Fruit: at each spot a shallow crate of wood, and in it pieces of fruit of the spot's colour, six lying
    side by side and two on top of them."""
    wall, high, floor = 0.012, 0.08, 0.03
    for x, y, colour in spots:
        r = draws.uniform(0.036, 0.042)                # an orange, an apple
        deep, across = 2 * (2 * r + 0.004), min(3 * (2 * r + 0.004), room - 2 * wall - 0.02)
        b.part(b.out(Box(x, y, 0.0, deep, across, floor, z0=z, color=_DARK_WOOD)), "wood_dark")
        for side in (1, -1):
            b.part(b.out(Box(x + side * (deep + wall) / 2, y, 0.0, wall, across + 2 * wall, high, z0=z, color=_WOOD)),
                   "wood")
            b.part(b.out(Box(x, y + side * (across + wall) / 2, 0.0, deep, wall, high, z0=z, color=_WOOD)), "wood")
        for i in (-0.5, 0.5):
            for j in (-1, 0, 1):
                dx, dy = draws.uniform(-0.002, 0.002), draws.uniform(-0.002, 0.002)
                b.part(b.out(Sphere(x + i * deep / 2 + dx, y + j * across / 3 + dy, z + floor + r, r, color=colour)),
                       PAINT)
        nested = math.sqrt(max(4 * r * r - (deep / 2) ** 2 / 4 - (across / 3) ** 2 / 4, 0.0))
        for j in (-0.5, 0.5):                          # the two on top lie in the hollows between four
            b.part(b.out(Sphere(x, y + j * across / 3, z + floor + r + nested, r, color=colour)), PAINT)


def _buckets(b: _Build, spots: Sequence[tuple[float, float, Color]], z: float, room: float, draws: Random) -> None:
    """Flowers: at each spot a florist's bucket of zinc, a mound of leaves in it and blooms of the spot's colour
    over that, one at the top and five round it."""
    rim, high, leaves = min(0.095, room / 2 - 0.02), 0.17, 0.085
    for k, (x, y, colour) in enumerate(spots):
        b.part(b.out(Cylinder(x, y, 0.075, high, z0=z, color=_STEEL, top_radius=rim)), "steel")
        heart = z + high + 0.03
        b.part(b.out(Sphere(x, y, heart, leaves, color=_LEAVES[k % len(_LEAVES)])), PAINT)
        turn = draws.uniform(0.0, 2 * math.pi)
        for n in range(6):
            lean = 0.0 if n == 0 else math.radians(draws.uniform(48.0, 62.0))     # off the upright
            round_ = turn + n * 2 * math.pi / 5
            out = (leaves + 0.008) * math.sin(lean)
            b.part(b.out(Sphere(x + out * math.cos(round_), y + out * math.sin(round_),
                                heart + (leaves + 0.008) * math.cos(lean), draws.uniform(0.03, 0.038), color=colour)),
                   PAINT)


def _jars(b: _Build, spots: Sequence[tuple[float, float, Color]], z: float, room: float, draws: Random) -> None:
    """Sweets: at each spot a tall jar full of sweets of the spot's colour, and before each gap between two a
    smaller one of another colour, every jar under a metal lid. A jar is glazed and takes its sweets' colour:
    the glass of a booth is the only glass a scene has."""
    def jar(x: float, y: float, radius: float, high: float, colour: Color) -> None:
        b.part(b.out(Cylinder(x, y, radius, high, z0=z, color=colour)), "ceramic")
        b.part(b.out(Cylinder(x, y, 0.86 * radius, 0.028, z0=z + high, color=_STEEL)), "steel")

    for x, y, colour in spots:
        jar(x - 0.03, y, min(0.075, room / 2 - 0.02), draws.uniform(0.19, 0.23), colour)
    for k, ((x, y, _), (_, beyond, _)) in enumerate(zip(spots, spots[1:])):
        jar(x + 0.1, (y + beyond) / 2, 0.05, draws.uniform(0.11, 0.14), spots[(k + 2) % len(spots)][2])


# What a second stall sells, by the word on its sign: what builds its goods, from the spots along the counter
# (each its x, its y and its colour), the height of the counter's top, how far apart the spots are, and
# something to draw small differences from.
_GOODS: dict[str, Callable[[_Build, Sequence[tuple[float, float, Color]], float, float, Random], None]] = {
    "fruit": _crates, "flowers": _buckets, "sweets": _jars}
_GOODS_APART_M = 0.3                               # the least from one spot to the next that they are built for


def _shop(b: _Build, primitives: Sequence[Solid]) -> None:
    """The second stall, restyled: in natural mode its sign board carries a logo where its name was, and a small
    price board hangs on its back board. Its goods are what the word on its sign says it sells (_GOODS), in the
    colours and at the spots along the counter of the placeholder's own, which are a ball on a stand each; a
    stall with any other word keeps those, restyled.

    It has a vendor (STAFFED_KINDS), and nothing of it is moved for that: the placeholder leaves floor between
    its back board on the wall and its counter, 0.47 m of it, where its twin has the one rectangle of all its
    ground. Here it blocks what it draws, the board and the counter with the posts that stand on the counter's
    ground, and the staff spot is on the strip between them: in the middle of its depth, and across it beside
    the middle of the goods, between two of them, so that whoever stands there shows from the front, at the
    side away from the price board."""
    theme = str(b.item.options.get("text", "")).lower()
    locals_ = [b.local(solid) for solid in primitives]
    stands = [local for local in locals_ if isinstance(local, Cylinder)]
    spots = sorted(((local.x, local.y, tuple(int(level) for level in local.color))
                    for local in locals_ if isinstance(local, Sphere)), key=lambda spot: spot[1])
    # What stands on the floor: the board on the wall, the counter before it, and posts.
    grounded = [local for local in locals_ if isinstance(local, Box) and local.z0 == 0.0]
    boards = [local for local in grounded if abs(local.cx - local.sx / 2) < 1e-6 and local.sy > 1.0]
    counters = [local for local in grounded if local.sy > 1.0 and local not in boards]
    if len(boards) != 1 or len(counters) != 1 or _extent(counters)[0] - _extent(boards)[2] < 0.4:
        raise ValueError("a shop is built from the placeholder's back board on the wall and its counter, with "
                         "floor between the two for whoever sells there")
    behind, before = _extent(boards), _extent(counters)
    b.blocked_at(*behind)
    b.blocked_at(*before)
    for post in grounded:                              # one that does not stand on the counter's ground
        x0, y0, x1, y1 = _extent([post])
        if (post not in boards and post not in counters
                and not (before[0] - 1e-9 <= x0 and x1 <= before[2] + 1e-9
                         and before[1] - 1e-9 <= y0 and y1 <= before[3] + 1e-9)):
            b.blocked_at(x0, y0, x1, y1)
    # Between two of the goods, those nearest the middle. Of two such gaps, the one at the side where the price
    # board does not hang (it hangs over the +y half of the counter): seen from that side, a shoulder would cover
    # the board's near edge. It is the same in either signage and however often the scene is dressed: a staff
    # spot is ground truth.
    gaps = sorted(((spot[1] + beyond[1]) / 2 for spot, beyond in zip(spots, spots[1:])), key=abs)
    nearest = [gap for gap in gaps if abs(gap) <= abs(gaps[0]) + 1e-9] or [0.0]
    b.staff((behind[2] + before[0]) / 2, min(nearest))
    apart = min((beyond[1] - spot[1] for spot, beyond in zip(spots, spots[1:])), default=_GOODS_APART_M)
    # Goods are built where the placeholder shows where they stand, with room for a crate or a bucket at each spot.
    stocked = _GOODS.get(theme) if spots and stands and apart >= _GOODS_APART_M - 1e-9 else None
    for solid, local in zip(primitives, map(b.local, primitives)):
        if isinstance(solid, Box) and solid.label:
            b.box(solid, mark=b.mark())
        elif b.natural and isinstance(local, Box) and local.cx - local.sx / 2 < 1e-6 and _top(local) >= 2.0:
            # The board on the wall: the prices at one side of it, over the counter.
            prices = (0.5 + 0.45 / local.sy, 1.25 / local.sz, 0.5 + 0.9 / local.sy, 1.85 / local.sz)
            b.box(solid, art=_Art("menu", theme, _CREAM, _BOARD, seed=b.number(), rect=prices))
        elif stocked is None or isinstance(solid, Box):
            b.solid(solid)
    if stocked is not None:
        # The goods are the same whatever the signage: what varies in them follows from the item's one number,
        # and not from the run of numbers that the sign art takes its seeds from. No generator of numpy's is made
        # for it: a dressing has five, one for each of its parts.
        stocked(b, spots, min(map(_bottom, stands)), apart, Random(b.seed))


def _opening(b: _Build, primitives: Sequence[Solid]) -> None:
    """A window or a door: the outline of the placeholder's box as an opening of the wall, which the room cuts
    and fills with a frame and glass or a leaf. Nothing of it is drawn into the room."""
    outline = next((b.local(solid) for solid in primitives if isinstance(solid, Box)), None)
    if outline is None:
        raise ValueError(f"a {b.item.kind} is drawn with a box, which gives the opening its outline")
    b.openings.append(Opening(b.item.kind, *b.point(0.0, outline.cy), b.item.yaw, outline.sy, outline.z0, outline.sz))
    b.blocked(b.item.footprint)


_BUILDERS: dict[str, Callable[[_Build, Sequence[Solid]], None]] = {
    "stall": _stall, "ticket_counter": _ticket_counter, "desk": _desk, "kiosk": _kiosk, "back_room": _back_room,
    "parcel_table": _parcel_table, "plant": _modelled, "bench": _modelled, "bin": _restyle, "pillar": _restyle,
    "poster": _poster, "door": _opening, "window": _opening, "board": _board, "cooler": _modelled,
    "lockers": _modelled, "extinguisher": _restyle, "shop": _shop}


def construct(item: Item, primitives: Sequence[Solid], signs: str, library: Library, rng: np.random.Generator,
              group: int = 0) -> Construction:
    """How the item is built in the realistic scene.

    `primitives` are the layout's own for the item, in the world: getattr(layout, item.source)[item.start:
    item.stop]. `signs` is one of SIGN_MODES. Every part and every model carries `group`. One number is
    taken from `rng`."""
    if signs not in SIGN_MODES:
        raise ValueError(f"unknown signage {signs!r}: expected one of {', '.join(SIGN_MODES)}")
    if item.kind not in _BUILDERS:
        raise ValueError(f"unknown kind {item.kind!r} of an item: expected one of {', '.join(_BUILDERS)}")
    build = _Build(item, signs == "natural", library, group, int(rng.integers(2 ** 31)))
    _BUILDERS[item.kind](build, tuple(primitives))
    return Construction(tuple(build.parts), tuple(build.models), tuple(build.obstacles), tuple(build.staff_spots),
                        tuple(build.lettering), tuple(build.surfaces), tuple(build.openings))
