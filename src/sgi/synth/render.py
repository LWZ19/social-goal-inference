"""Placeholder renderer: a numpy ray caster for a room, boxes, cylinders, spheres and standing figures.

Rays pass through pixel centres and are not normalised: their camera-frame Z is 1, so the ray parameter
of a hit is its camera Z, which is the depth written. Depth always comes from the ray through the pixel
centre. Colour does too, except in a pixel on an outline or on fine detail, whose colour is the mean of
nine rays spread over it.

The light comes from above, a little from behind the viewer and from their left, with a weaker one along
the line of sight, so a picture does not change when the scene and the camera are moved together. Floor,
ceiling, walls, boxes, props and figures are drawn in that order, and of two surfaces that coincide the
first drawn shows: give a poster on a wall a few centimetres of depth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sgi.geometry import Camera

Color = tuple[int, int, int]

# Colours a scene may give its figures, light to dark.
SKIN_TONES: tuple[Color, ...] = ((244, 208, 180), (229, 184, 150), (203, 150, 110), (160, 110, 76), (112, 74, 52))
HAIR_COLORS: tuple[Color, ...] = ((196, 160, 100), (150, 150, 148), (128, 88, 52), (58, 42, 32), (30, 26, 24))
TROUSER_COLORS: tuple[Color, ...] = ((150, 135, 110), (84, 100, 132), (92, 72, 58), (62, 62, 66), (44, 50, 74), (30, 30, 34))
HEIGHTS_M: tuple[float, ...] = (1.70, 1.62, 1.78, 1.66, 1.84, 1.58, 1.74)

_SKY = (225, 235, 245)
_GROUND = (205, 200, 185)
_AMBIENT = 0.58                                    # share of its colour a surface turned from both lights keeps
_KEY = 0.30                                        # added in full on a surface square to the light from above
_FILL = 0.12                                       # and on one square to the line of sight
_KEY_FROM = (0.92, 0.25, 0.30)                     # the light from above: its parts up, towards the viewer, to their left
_NEAR_M = 0.05                                     # a shape with a corner nearer than this is tested on every ray
_NEARER = 1.0 - 1e-9                               # a hit replaces an earlier one only below this share of its
#                                                    distance: two faces in one plane differ by rounding alone
_EDGE = 12                                         # a step in colour between neighbours, of 255, that marks an outline
_SAMPLES = tuple((du, dv) for dv in (-1 / 3, 0.0, 1 / 3) for du in (-1 / 3, 0.0, 1 / 3) if du or dv)   # round the centre
_NOTHING, _FLOOR, _SOLID = 0, 1, 2                 # what a ray has met
_ALL = slice(None)
_UP = np.array([0.0, 0.0, 1.0])
_CORNERS = np.array([(x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])

_LABEL_FONT_SIZE = 64                              # cells per em of the rasterised label
_LABEL_STROKE = 2                                  # cells added round every stroke, for a bolder letter
_INK_DARK = (20, 20, 20)
_INK_LIGHT = (245, 245, 245)
_INK_DARK_FROM = 110                               # mean of a background colour from which lettering on it is dark
_FRAME_DARKER = 0.55                               # the frame of a panel is the box's colour times this
_FRAME_MAX_M = 0.025
_FRAME_SHARE = 0.06                                # of the panel's shorter side, when that is less
# name: axis of the face's normal and its sign, axis the lettering runs along and its sign, axis that is up and its sign
_FACES = {"front": (0, 1.0, 1, 1.0, 2, 1.0), "back": (0, -1.0, 1, -1.0, 2, 1.0), "left": (1, 1.0, 0, -1.0, 2, 1.0),
          "right": (1, -1.0, 0, 1.0, 2, 1.0), "top": (2, 1.0, 1, 1.0, 0, -1.0)}

_TILE_JOINT_M = 0.014
_TILE_TONE = 0.045                                 # how far a tile's brightness strays from the floor colour
_CEILING_LIT = 0.93                                # the ceiling glows evenly; its light panels are brighter
_LIGHT_PANEL = (255, 254, 246)
_LIGHT_PANEL_M = (1.2, 0.3)
_WALL_FOOT_SHADE = (0.07, 0.5)                     # walls darken by this share towards the floor, over this many metres
_WALL_TOP_SHADE = (0.05, 0.4)
_WALL_END_SHADE = (0.06, 0.6)                      # and towards each corner
_DADO_RAIL_M = 0.05
_DADO_RAIL_DARKER = 0.8

_SHADOW = 0.45                                     # share of light lost on the floor where something stands
_SHADOW_MOST = 0.62                                # and the most lost where several shadows fall together
_SHADOW_REACH_M = 0.3                              # and how far from its foot the shadow fades out
_SHADOW_CLEAR_M = 1.2                              # nothing whose underside is this high darkens the floor
_FIGURE_SHADOW = (0.15, 0.5, 0.3)                  # radius, strength and reach of the shadow at a figure's feet
_WALL_SHADOW = (0.14, 0.5)                         # strength and reach of the shade along the foot of a wall

# A standing body 1.7 m tall in a footprint 0.22 m in radius, in metres: x forward, y to its left, z up.
_BODY_HEIGHT, _BODY_RADIUS = 1.7, 0.22
_HEAD_CENTRE, _HEAD_SEMI = (0.012, 0.0, 1.572), (0.103, 0.087, 0.116)
_HAIR_CENTRE, _HAIR_SEMI = (0.004, 0.0, 1.578), (0.113, 0.096, 0.122)   # a shell round the skull, cut away over the face
_HAIR_NAPE = -0.6                                  # and below this height, of 1, where the hair lies close on the neck
_EYE = (28, 24, 22)
_EYE_Y, _EYE_Z, _EYE_RY, _EYE_RZ = 0.034, 0.018, 0.015, 0.019      # each eye: its centre, off the middle of the head
_BROW_Z, _BROW_RY, _BROW_RZ = 0.051, 0.023, 0.0065
_LIPS = (150, 62, 56)
_MOUTH_Z, _MOUTH_RY, _MOUTH_RZ, _SMILE = -0.052, 0.031, 0.0085, 5.0  # the mouth curves up by _SMILE * y * y
_FACE_FROM = 0.2                                   # the features lie on the part of the head this far forward, of 1
_HAIR_TILT, _HAIRLINE = (-0.72, 0.69), -0.04       # hair where the forward and upward parts of a direction, weighted, pass this
_NECKLINE_Z, _NECKLINE_SLOPE = 1.33, 0.4          # the shirt opens in a V above this height


@dataclass(frozen=True)
class Box:
    cx: float                                      # centre on the ground plane; +x of the box is its front
    cy: float
    yaw: float
    sx: float                                      # extent along its front axis, across, and up
    sy: float
    sz: float
    z0: float = 0.0                                # height of its underside
    color: Color = (150, 150, 150)
    label: str | None = None                       # drawn on the front (+x) face, or on label_face
    label_face: str = "front"                      # the face with the label and the panel: front, back (-x), left (+y),
    #                                                right (-y) or top, where the lettering reads from the front
    label_color: Color | None = None               # of the lettering; None is dark or light, against what is behind it
    label_height: float = 0.5                      # the most of the face's height the lettering fills (the panel's, if any)
    label_width: float = 0.8                       # and of its width
    panel: Color | None = None                     # a framed rectangle on that face: a sign plate, a screen, a poster, a window
    panel_rect: tuple[float, float, float, float] = (0.1, 0.1, 0.9, 0.9)   # its left, bottom, right, top, as shares of the face
    band: Color | None = None                      # a second colour round the four upright faces: a plinth, a counter's edge
    band_range: tuple[float, float] = (0.0, 0.1)   # from and to, as shares of the height
    top: Color | None = None                       # the colour of the top face; None is `color`


@dataclass(frozen=True)
class Cylinder:
    x: float                                       # its upright axis
    y: float
    radius: float
    height: float
    z0: float = 0.0                                # height of its underside
    color: Color = (150, 150, 150)
    top: Color | None = None                       # the colour of its upper end; None is `color`
    top_radius: float | None = None                # the radius at its upper end, when it tapers or flares; None is `radius`


@dataclass(frozen=True)
class Sphere:
    x: float                                       # its centre
    y: float
    z: float
    radius: float
    color: Color = (150, 150, 150)


@dataclass(frozen=True)
class Figure:
    x: float
    y: float
    yaw: float
    color: Color = (60, 90, 200)                   # the shirt
    height: float = 1.7
    radius: float = 0.22                           # of the footprint the body, arms included, stays inside
    skin: Color = SKIN_TONES[1]
    hair: Color = HAIR_COLORS[3]
    trousers: Color = TROUSER_COLORS[4]
    shoes: Color = (34, 30, 28)


@dataclass(frozen=True)
class Room:
    """Walls on the edges of a floor polygon, a ceiling, and a tiled floor. The camera belongs inside."""
    polygon: Sequence[tuple[float, float]]         # the floor's outline in world XY, in order either way round
    height: float = 3.0                            # of the walls, and of the ceiling above the floor
    wall: Color = (233, 227, 213)
    dado: Color | None = (200, 190, 170)           # the lower part of the walls; None leaves them one colour
    dado_height: float = 0.95
    skirting: Color = (96, 86, 76)                 # the band at the foot of the walls
    skirting_height: float = 0.12
    floor: Color = (200, 192, 176)
    tile: float = 0.6                              # side of a floor tile, laid from the first corner along the first edge; 0.0 for none
    joint: Color = (128, 122, 110)
    ceiling: Color = (243, 242, 236)
    lights: float = 2.4                            # spacing of the ceiling's light panels each way; 0.0 for none

    @classmethod
    def rectangle(cls, cx: float, cy: float, yaw: float, sx: float, sy: float, **fields) -> Room:
        """A room sx long (along yaw) and sy across, centred on (cx, cy)."""
        c, s = math.cos(yaw), math.sin(yaw)
        corners = ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))
        return cls(tuple((cx + c * x - s * y, cy + s * x + c * y) for x, y in corners), **fields)


Solid = Box | Cylinder | Sphere


def appearance(index: int) -> dict:
    """A height, skin, hair and trousers for the index-th person of a scene, as fields of Figure.

    The same for the same index, and unlike for neighbours: Figure(x, y, yaw, color=shirt, **appearance(i))."""
    return dict(height=HEIGHTS_M[index % len(HEIGHTS_M)], skin=SKIN_TONES[(2 * index + 1) % len(SKIN_TONES)],
                hair=HAIR_COLORS[(3 * index + 3) % len(HAIR_COLORS)], trousers=TROUSER_COLORS[(5 * index + 4) % len(TROUSER_COLORS)])


class _Rays:
    """Rays from the camera's centre, and for each the nearest hit found so far: its distance and its lit colour.

    The rays are the centres of all the pixels, in row-major order, or with `pixels`, the indices of some of
    them, one ray per offset in each of those pixels, a block of len(pixels) rays for each offset.
    """

    def __init__(self, cam: Camera, pixels: np.ndarray | None = None, offsets: Sequence[tuple[float, float]] = ((0.0, 0.0),)):
        self.cam, self.blocks = cam, len(offsets)
        if pixels is None:
            rows, cols = np.divmod(np.arange(cam.width * cam.height), cam.width)
            self.rows = self.cols = None
        else:
            rows, cols = self.rows, self.cols = np.divmod(pixels, cam.width)
        du, dv = np.array(offsets).T[:, :, None]
        u, v = (cols + 0.5 + du).ravel(), (rows + 0.5 + dv).ravel()
        R = cam.T_world_camera[:3, :3].astype(np.float64)
        self.R, self.origin = R, cam.T_world_camera[:3, 3].astype(np.float64)
        self.d = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u)], axis=-1) @ R.T
        self.d_per_u, self.d_per_v = R[:, 0] / cam.fx, R[:, 1] / cam.fy       # how a ray turns from one pixel to the next
        back, left = _level(-R[:, 2]), _level(-R[:, 0])
        key = _KEY_FROM[0] * _UP + _KEY_FROM[1] * back + _KEY_FROM[2] * left
        self.key, self.fill = key / np.linalg.norm(key), -R[:, 2]
        self.t = np.full(u.size, np.inf)
        self.rgb = np.tile(np.array(_SKY, dtype=np.float64), (u.size, 1))
        self.kind = np.full(u.size, _NOTHING, dtype=np.uint8)
        self.detail = np.zeros(u.size, dtype=bool)                             # on something too fine for one ray a pixel

    def window(self, points) -> np.ndarray | slice | None:
        """The rays that can meet a convex shape with these world points as corners: their indices, or None for none."""
        cam, p = self.cam, (np.asarray(points, dtype=np.float64) - self.origin) @ self.R
        z = p[:, 2]
        if (z <= _NEAR_M).all():
            return None
        if (z <= _NEAR_M).any():
            return _ALL
        u, v = cam.fx * p[:, 0] / z + cam.cx, cam.fy * p[:, 1] / z + cam.cy
        u0, u1, v0, v1 = u.min() - 1.0, u.max() + 1.0, v.min() - 1.0, v.max() + 1.0   # a pixel to spare each way
        if self.rows is None:
            c0, c1 = max(math.floor(u0), 0), min(math.floor(u1) + 1, cam.width)
            r0, r1 = max(math.floor(v0), 0), min(math.floor(v1) + 1, cam.height)
            if c0 >= c1 or r0 >= r1:
                return None
            return (np.arange(r0, r1)[:, None] * cam.width + np.arange(c0, c1)).ravel()
        inside = np.nonzero((self.cols + 1 >= u0) & (self.cols <= u1) & (self.rows + 1 >= v0) & (self.rows <= v1))[0]
        if not inside.size:
            return None
        return (inside + self.rows.size * np.arange(self.blocks)[:, None]).ravel()

    def hit(self, sel, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Record t on the selected rays where it is positive and smaller than every hit so far by more than
        rounding, so of two coincident surfaces the first drawn stays.

        A ray the shape misses holds nan or inf in t. Returns the mask of those rays within the selection,
        and their indices.
        """
        mask = (t > 0) & (t < self.t[sel] * _NEARER)
        where = np.nonzero(mask)[0] if isinstance(sel, slice) else sel[mask]
        self.t[where] = t[mask]
        return mask, where

    def lit(self, normal: np.ndarray) -> np.ndarray:
        """The share of its colour a surface shows, from its outward normals (unit or not): one each, or one for all."""
        n = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
        return _AMBIENT + _KEY * np.maximum(n @ self.key, 0.0) + _FILL * np.maximum(n @ self.fill, 0.0)

    def paint(self, where: np.ndarray, base, lit, kind: int = _SOLID, detail=False) -> None:
        self.rgb[where] = np.asarray(base, dtype=np.float64) * np.asarray(lit)[..., None]
        self.kind[where] = kind
        self.detail[where] = detail

    def floor_xy(self, where: np.ndarray) -> np.ndarray:
        return self.origin[:2] + self.t[where, None] * self.d[where, :2]


def _level(v: np.ndarray) -> np.ndarray:
    """The horizontal part of a direction as a unit vector; zero for a vertical one."""
    flat = np.array([v[0], v[1], 0.0])
    length = np.linalg.norm(flat)
    return flat / length if length > 1e-9 else flat


def _room_frame(room: Room) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The first corner of the room, and unit vectors along its first edge and square to it."""
    first, second = np.asarray(room.polygon[0], dtype=np.float64), np.asarray(room.polygon[1], dtype=np.float64)
    along = (second - first) / np.linalg.norm(second - first)
    return first, along, np.array([-along[1], along[0]])


def _striped(x: np.ndarray, span: np.ndarray, period: float, width: float) -> np.ndarray:
    """The share of each interval [x - span / 2, x + span / 2] that lies on stripes `width` wide, centred on
    the multiples of `period`."""
    def upto(s: np.ndarray) -> np.ndarray:         # the length of stripe below s, less a constant
        s = s + width / 2
        whole = np.floor(s / period)
        return whole * width + np.minimum(s - whole * period, width)

    span = np.maximum(span, 1e-9)
    return (upto(x + span / 2) - upto(x - span / 2)) / span


def _tiles(rays: _Rays, where: np.ndarray, room: Room) -> np.ndarray:
    """The floor colour at each hit: tiles of slightly different brightness with a joint between them.

    A pixel takes the mean over the floor it covers, so that joints fade with distance and do not flicker."""
    first, along, across = _room_frame(room)
    d, t = rays.d[where], rays.t[where, None]
    p = rays.floor_xy(where) - first
    a, b = p @ along, p @ across
    # The floor a pixel covers: how far the hit moves along each tile axis from one pixel to the next.
    per_u = t * (rays.d_per_u[:2] - rays.d_per_u[2] / d[:, 2:] * d[:, :2])
    per_v = t * (rays.d_per_v[:2] - rays.d_per_v[2] / d[:, 2:] * d[:, :2])
    span_a = np.abs(per_u @ along) + np.abs(per_v @ along)
    span_b = np.abs(per_u @ across) + np.abs(per_v @ across)
    joint = 1.0 - (1.0 - _striped(a, span_a, room.tile, _TILE_JOINT_M)) * (1.0 - _striped(b, span_b, room.tile, _TILE_JOINT_M))
    i, j = np.floor(a / room.tile).astype(np.int64), np.floor(b / room.tile).astype(np.int64)
    tone = (((i * 73856093) ^ (j * 19349663)) & 255) / 127.5 - 1.0             # one of 256 values in [-1, 1] per tile
    tone = tone * np.clip(1.0 - np.maximum(span_a, span_b) / room.tile, 0.0, 1.0)
    tile = np.array(room.floor, dtype=np.float64) * (1.0 + _TILE_TONE * tone[:, None])
    return tile + (np.array(room.joint, dtype=np.float64) - tile) * joint[:, None]


def _draw_floor(rays: _Rays, room: Room | None) -> None:
    _, where = rays.hit(_ALL, -rays.origin[2] / rays.d[:, 2])
    if room is None:
        base = _GROUND
    elif room.tile > 0:
        base = _tiles(rays, where, room)
    else:
        base = room.floor
    rays.paint(where, base, rays.lit(_UP), _FLOOR)


def _draw_ceiling(rays: _Rays, room: Room) -> None:
    _, where = rays.hit(_ALL, (room.height - rays.origin[2]) / rays.d[:, 2])
    if not where.size:
        return
    base, lit = np.tile(np.array(room.ceiling, dtype=np.float64), (where.size, 1)), np.full(where.size, _CEILING_LIT)
    if room.lights > 0:
        first, along, across = _room_frame(room)
        p = rays.floor_xy(where) - first
        offset = (room.lights - np.array(_LIGHT_PANEL_M)) / 2                  # a panel in the middle of every square
        on = ((np.mod(p @ along - offset[0], room.lights) < _LIGHT_PANEL_M[0])
              & (np.mod(p @ across - offset[1], room.lights) < _LIGHT_PANEL_M[1]))
        base[on], lit[on] = _LIGHT_PANEL, 1.0
    rays.paint(where, base, lit)


def _draw_walls(rays: _Rays, room: Room) -> None:
    corners = np.asarray(room.polygon, dtype=np.float64)
    eye, slack = rays.origin[:2], 1e-6             # the slack closes the seam where two walls meet
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        length = np.linalg.norm(b - a)
        along = (b - a) / length
        normal = np.array([along[1], -along[0]])
        if normal @ (eye - a) < 0:                 # the side the camera sees
            normal = -normal
        sel = rays.window([(*a, 0.0), (*b, 0.0), (*a, room.height), (*b, room.height)])
        if sel is None:
            continue
        d = rays.d[sel]
        t = normal @ (a - eye) / (d[:, :2] @ normal)
        s, z = (eye - a) @ along + t * (d[:, :2] @ along), rays.origin[2] + t * d[:, 2]
        on = (s >= -slack) & (s <= length + slack) & (z >= -slack) & (z <= room.height + slack)
        mask, where = rays.hit(sel, np.where(on, t, np.nan))
        if not where.size:
            continue
        s, z = s[mask], z[mask]
        base = np.tile(np.array(room.wall, dtype=np.float64), (where.size, 1))
        if room.dado is not None:
            base[z < room.dado_height] = room.dado
            base[np.abs(z - room.dado_height) < _DADO_RAIL_M / 2] = np.array(room.dado) * _DADO_RAIL_DARKER
        base[z < room.skirting_height] = room.skirting
        end = np.minimum(s, length - s)
        shade = (1.0 - _WALL_FOOT_SHADE[0] * np.exp(-z / _WALL_FOOT_SHADE[1])
                 - _WALL_TOP_SHADE[0] * np.exp(-(room.height - z) / _WALL_TOP_SHADE[1])
                 - _WALL_END_SHADE[0] * np.exp(-end / _WALL_END_SHADE[1]))
        rays.paint(where, base, rays.lit(np.array([*normal, 0.0])) * shade)


@lru_cache(maxsize=256)
def _label_cells(label: str) -> np.ndarray:
    """The label as a mask of inked cells, cropped to the ink; rows run downwards."""
    font = ImageFont.load_default(size=_LABEL_FONT_SIZE)
    left, top, right, bottom = font.getbbox(label, stroke_width=_LABEL_STROKE)
    image = Image.new("L", (max(right - left, 1), max(bottom - top, 1)))
    ImageDraw.Draw(image).text((-left, -top), label, fill=255, font=font, stroke_width=_LABEL_STROKE, stroke_fill=255)
    cells = np.asarray(image) >= 128
    rows, cols = np.nonzero(cells.any(axis=1))[0], np.nonzero(cells.any(axis=0))[0]
    return cells[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] if rows.size else cells[:0, :0]


def _contrasting(background: Color) -> Color:
    return _INK_DARK if sum(background) / 3 >= _INK_DARK_FROM else _INK_LIGHT


def _box_colours(box: Box, axis: np.ndarray, outward: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The colour of each hit on a box, and whether it lies on lettering.

    axis and outward give the face met: the box axis of its normal, and the normal's sign. point is in the
    box frame. The label face is measured as its viewer sees it: a from its left edge, b up from its lower one.
    """
    base = np.tile(np.array(box.color, dtype=np.float64), (axis.size, 1))
    detail = np.zeros(axis.size, dtype=bool)
    if box.top is not None:
        base[(axis == 2) & (outward > 0)] = box.top
    if box.band is not None:
        share = (point[:, 2] - box.z0) / box.sz
        base[(axis != 2) & (share >= box.band_range[0]) & (share <= box.band_range[1])] = box.band
    if box.panel is None and not box.label:
        return base, detail
    if box.label_face not in _FACES:
        raise ValueError(f"unknown label_face {box.label_face!r}: expected one of {', '.join(_FACES)}")
    normal, sign, a_axis, a_sign, b_axis, b_sign = _FACES[box.label_face]
    face = np.nonzero((axis == normal) & (outward == sign))[0]
    size, middle = (box.sx, box.sy, box.sz), (0.0, 0.0, box.z0 + box.sz / 2)
    width, height = size[a_axis], size[b_axis]
    a = a_sign * (point[face, a_axis] - middle[a_axis]) + width / 2
    b = b_sign * (point[face, b_axis] - middle[b_axis]) + height / 2
    area, background = (0.0, 0.0, width, height), box.color        # where the lettering goes: left, bottom, right, top
    if box.panel is not None:
        left, bottom, right, top = box.panel_rect
        area, background = (left * width, bottom * height, right * width, top * height), box.panel
        inset = np.minimum(np.minimum(a - area[0], area[2] - a), np.minimum(b - area[1], area[3] - b))   # from the panel's edge
        frame = min(_FRAME_MAX_M, _FRAME_SHARE * min(area[2] - area[0], area[3] - area[1]))
        base[face[inset >= 0]] = np.array(box.color) * _FRAME_DARKER
        base[face[inset >= frame]] = box.panel
    cells = _label_cells(box.label) if box.label else np.zeros((0, 0), dtype=bool)
    if cells.size:
        n_rows, n_cols = cells.shape
        cell_m = min(box.label_width * (area[2] - area[0]) / n_cols, box.label_height * (area[3] - area[1]) / n_rows)
        col = np.floor((a - (area[0] + area[2]) / 2) / cell_m + n_cols / 2).astype(int)
        row = np.floor(((area[1] + area[3]) / 2 - b) / cell_m + n_rows / 2).astype(int)
        inside = (col >= 0) & (col < n_cols) & (row >= 0) & (row < n_rows)
        detail[face[inside]] = True
        inked = face[inside][cells[row[inside], col[inside]]]
        base[inked] = box.label_color if box.label_color is not None else _contrasting(background)
    return base, detail


def _draw_box(rays: _Rays, box: Box) -> None:
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    to_box = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])           # world to box: a turn of -yaw about Z
    centre = np.array([box.cx, box.cy, 0.0])
    low = np.array([-box.sx / 2, -box.sy / 2, box.z0])
    high = np.array([box.sx / 2, box.sy / 2, box.z0 + box.sz])
    sel = rays.window(centre + (low + _CORNERS * (high - low)) @ to_box)
    if sel is None:
        return
    o, d = to_box @ (rays.origin - centre), rays.d[sel] @ to_box.T
    t_low, t_high = (low - o) / d, (high - o) / d
    t_near = np.minimum(t_low, t_high)                                         # per axis: where the ray enters its slab
    t_enter, t_exit = t_near.max(axis=1), np.maximum(t_low, t_high).min(axis=1)
    mask, where = rays.hit(sel, np.where(t_enter <= t_exit, t_enter, np.nan))
    if not where.size:
        return
    d, each = d[mask], np.arange(where.size)
    axis = t_near[mask].argmax(axis=1)                                         # the axis that set t_enter is the face
    outward = -np.sign(d[each, axis])                                          # the face met is the one turned to the ray
    base, detail = _box_colours(box, axis, outward, o + rays.t[where, None] * d)
    normal = np.zeros_like(d)
    normal[each, axis] = outward
    rays.paint(where, base, rays.lit(normal @ to_box), detail=detail)


def _meet_ball(o: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Where rays o + t d enter the unit sphere; nan where they miss it."""
    a, half_b, c = (d * d).sum(axis=1), d @ o, o @ o - 1.0
    return (-half_b - np.sqrt(half_b * half_b - a * c)) / a


def _meet_limb(o: np.ndarray, d: np.ndarray, taper: float) -> tuple[np.ndarray, np.ndarray]:
    """Where rays o + t d enter the solid x * x + y * y <= (1 + taper * z) ** 2, 0 <= z <= 1; inf where they miss it.

    Also which part each ray meets: 0 the side, -1 the end at z = 0, 1 the end at z = 1."""
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    r = 1.0 + taper * o[2]                                                     # the radius at the height of the origin
    a = dx * dx + dy * dy - (taper * dz) ** 2
    half_b = o[0] * dx + o[1] * dy - taper * r * dz
    root = np.sqrt(half_b * half_b - a * (o[0] ** 2 + o[1] ** 2 - r * r))      # nan where the ray misses the side
    best, part = np.full(dx.shape, np.inf), np.zeros(dx.shape)
    for t in ((-half_b - root) / a, (-half_b + root) / a):
        z = o[2] + t * dz
        best = np.where((t > 0) & (z >= 0) & (z <= 1) & (t < best), t, best)
    for z, radius, sign in ((0.0, 1.0, -1.0), (1.0, 1.0 + taper, 1.0)):
        t = (z - o[2]) / dz
        x, y = o[0] + t * dx, o[1] + t * dy
        nearer = (t > 0) & (x * x + y * y <= radius * radius) & (t < best)
        best, part = np.where(nearer, t, best), np.where(nearer, sign, part)
    return best, part


def _draw_round(rays: _Rays, centre: np.ndarray, axes: np.ndarray, taper: float | None,
                paint: Color | Callable[[np.ndarray, np.ndarray | None], np.ndarray], detail: bool = False,
                keep: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
    """An ellipsoid, or a limb: a cylinder that may taper, slant, and be oval in section.

    Both are the world points centre + axes @ q: the ellipsoid for the q of the unit ball, the limb for the
    q with q_x ** 2 + q_y ** 2 <= (1 + taper * q_z) ** 2 and 0 <= q_z <= 1, so that a limb runs from `centre`
    along the third column of `axes`. paint is one colour, or gives the colours at the q of the hits, and is
    told for a limb which part each is on. keep, if given, says at which q the surface is there at all; a ray
    that enters where it is not passes through, so what is kept should hide the far side of the shape.
    """
    reach = 1.0 if taper is None else max(1.0, 1.0 + taper)
    corners = (_CORNERS * 2 - 1) * (reach, reach, 1.0) if taper is None else (_CORNERS * (2, 2, 1) - (1, 1, 0)) * (reach, reach, 1.0)
    sel = rays.window(centre + corners @ axes.T)
    if sel is None:
        return
    to_local = np.linalg.inv(axes)
    o, d = to_local @ (rays.origin - centre), rays.d[sel] @ to_local.T
    t, part = (_meet_ball(o, d), None) if taper is None else _meet_limb(o, d, taper)
    if keep is not None:
        t = np.where(keep(o + t[:, None] * d), t, np.nan)
    mask, where = rays.hit(sel, t)
    if not where.size:
        return
    q = o + t[mask, None] * d[mask]
    normal = q
    if taper is not None:
        part = part[mask]
        normal = np.stack([q[:, 0], q[:, 1], -taper * (1.0 + taper * q[:, 2])], axis=1)
        normal[part != 0] = 0.0
        normal[:, 2] += part
    base = paint(q, part) if callable(paint) else paint
    rays.paint(where, base, rays.lit(normal @ to_local), detail=detail)       # a normal turns by the inverse transpose


def _draw_cylinder(rays: _Rays, cyl: Cylinder) -> None:
    top = cyl.radius if cyl.top_radius is None else cyl.top_radius
    paint = cyl.color if cyl.top is None else lambda q, part: np.where((part > 0)[:, None], cyl.top, cyl.color)
    _draw_round(rays, np.array([cyl.x, cyl.y, cyl.z0]), np.diag([cyl.radius, cyl.radius, cyl.height]),
                top / cyl.radius - 1.0, paint)


def _draw_sphere(rays: _Rays, ball: Sphere) -> None:
    _draw_round(rays, np.array([ball.x, ball.y, ball.z]), np.eye(3) * ball.radius, None, ball.color)


def _face(fig: Figure, q: np.ndarray) -> np.ndarray:
    """The colours of points on the head, given as unit vectors from its centre: x forward, y to its left, z up.

    Hair covers the top and the back; the front has two eyes under brows, and a mouth."""
    y, z = q[:, 1] * _HEAD_SEMI[1], q[:, 2] * _HEAD_SEMI[2]                    # metres from the middle of the head
    front = q[:, 0] > _FACE_FROM
    base = np.tile(np.array(fig.skin, dtype=np.float64), (len(q), 1))
    base[front & (((y / _MOUTH_RY) ** 2 + ((z - _MOUTH_Z - _SMILE * y * y) / _MOUTH_RZ) ** 2) < 1.0)] = _LIPS
    beside = np.abs(y) - _EYE_Y                                                # from the middle of the nearer eye
    base[front & ((beside / _BROW_RY) ** 2 + ((z - _BROW_Z) / _BROW_RZ) ** 2 < 1.0)] = fig.hair
    base[front & ((beside / _EYE_RY) ** 2 + ((z - _EYE_Z) / _EYE_RZ) ** 2 < 1.0)] = _EYE
    base[_hairy(q)] = fig.hair
    return base


def _hairy(q: np.ndarray) -> np.ndarray:
    """Whether hair grows at each point of the head, given as for _face: on its top and its back."""
    return _HAIR_TILT[0] * q[:, 0] + _HAIR_TILT[1] * q[:, 2] > _HAIRLINE


def _figure_parts(fig: Figure) -> list[tuple]:
    """The body as the arguments of _draw_round: two shoes and legs, hips, a chest, two arms, a neck and a head.

    The parts are laid out in the body's own metres and scaled to the figure's height and footprint. From
    0.81 m up to the top of the head they enclose the vertical axis, which the visibility rule samples."""
    wide, tall = fig.radius / _BODY_RADIUS, fig.height / _BODY_HEIGHT
    c, s = math.cos(fig.yaw), math.sin(fig.yaw)
    frame = np.array([[c * wide, -s * wide, 0.0], [s * wide, c * wide, 0.0], [0.0, 0.0, tall]])   # body metres to world
    foot = np.array([fig.x, fig.y, 0.0])
    parts: list[tuple] = []

    def ball(centre, semi, paint, detail=False, keep=None) -> None:
        parts.append((foot + frame @ np.array(centre), frame * np.array(semi), None, paint, detail, keep))

    def limb(start, end, r0: float, r1: float, paint) -> None:                # round in section, r0 at start, r1 at end
        start, along = np.array(start), np.subtract(end, start)
        u = np.cross(along, (1.0, 0.0, 0.0))
        u, v = u / np.linalg.norm(u), np.cross(along, u)
        axes = np.column_stack([r0 * u, r0 * v / np.linalg.norm(v), along])
        parts.append((foot + frame @ start, frame @ axes, r1 / r0 - 1.0, paint))

    def shirt(centre, scale) -> Callable:          # the shirt's colour, with skin in the V of its neckline
        def paint(q: np.ndarray, part) -> np.ndarray:
            p = np.array(centre) + q * scale       # in body metres
            opening = (p[:, 0] > 0) & (np.abs(p[:, 1]) < _NECKLINE_SLOPE * (p[:, 2] - _NECKLINE_Z))
            return np.where(opening[:, None], fig.skin, fig.color)
        return paint

    for side in (1.0, -1.0):                       # left, right
        ball((0.05, side * 0.092, 0.042), (0.118, 0.05, 0.046), fig.shoes)
        limb((0.0, side * 0.092, 0.06), (0.0, side * 0.086, 0.84), 0.056, 0.082, fig.trousers)
    ball((0.0, 0.0, 0.87), (0.10, 0.164, 0.12), fig.trousers)                  # hips
    # The chest widens by a tenth from the hem of the shirt to the shoulders, which round it off at the same width.
    chest, shoulders = ((0.0, 0.0, 0.90), (0.098, 0.15, 0.47)), ((0.0, 0.0, 1.37), (0.1078, 0.165, 0.08))
    parts.append((foot + frame @ np.array(chest[0]), frame * np.array(chest[1]), 0.1, shirt(*chest)))
    ball(*shoulders, shirt(*shoulders))
    for side in (1.0, -1.0):
        shoulder, elbow, wrist = (0.0, side * 0.168, 1.335), (-0.012, side * 0.178, 1.10), (0.06, side * 0.168, 0.865)
        ball(shoulder, (0.047, 0.047, 0.047), fig.color)
        limb(shoulder, elbow, 0.046, 0.04, fig.color)
        ball(elbow, (0.04, 0.04, 0.04), fig.color)
        limb(elbow, wrist, 0.036, 0.029, fig.skin)
        ball((0.075, side * 0.165, 0.80), (0.036, 0.022, 0.06), fig.skin)     # hand
    limb((0.0, 0.0, 1.40), (0.008, 0.0, 1.52), 0.046, 0.044, fig.skin)         # neck
    ball(_HEAD_CENTRE, _HEAD_SEMI, lambda q, part: _face(fig, q), detail=True)
    ball(_HAIR_CENTRE, _HAIR_SEMI, fig.hair, detail=True, keep=lambda q: _hairy(q) & (q[:, 2] > _HAIR_NAPE))
    ball((_HEAD_CENTRE[0] + 0.098, 0.0, 1.56), (0.022, 0.014, 0.026), fig.skin)  # nose
    for side in (1.0, -1.0):
        ball((0.0, side * 0.09, 1.572), (0.014, 0.012, 0.027), fig.skin)      # ear
    return parts


def _draw_solid(rays: _Rays, solid: Solid) -> None:
    if isinstance(solid, Box):
        _draw_box(rays, solid)
    elif isinstance(solid, Cylinder):
        _draw_cylinder(rays, solid)
    elif isinstance(solid, Sphere):
        _draw_sphere(rays, solid)
    else:
        raise TypeError(f"cannot draw a {type(solid).__name__} as a box or a prop: expected a Box, a Cylinder or a Sphere")


def _shadows(room: Room | None, solids: Sequence[Solid], figures: Sequence[Figure]) -> list[tuple]:
    """What darkens the floor: (x, y, yaw, half length, half width, round, strength, reach) for each footprint.

    A shadow is strongest under a footprint and fades out over its reach. Something raised casts a weaker
    one, down to none at all from _SHADOW_CLEAR_M up."""
    out: list[tuple] = []

    def under(z0: float) -> float:
        return _SHADOW * max(0.0, 1.0 - z0 / _SHADOW_CLEAR_M)

    for solid in solids:
        if isinstance(solid, Box):
            out.append((solid.cx, solid.cy, solid.yaw, solid.sx / 2, solid.sy / 2, False, under(solid.z0), _SHADOW_REACH_M))
        elif isinstance(solid, Cylinder):
            out.append((solid.x, solid.y, 0.0, solid.radius, solid.radius, True, under(solid.z0), _SHADOW_REACH_M))
        elif isinstance(solid, Sphere):
            out.append((solid.x, solid.y, 0.0, 0.7 * solid.radius, 0.7 * solid.radius, True,
                        under(solid.z - solid.radius), _SHADOW_REACH_M))
    for fig in figures:
        radius, strength, reach = _FIGURE_SHADOW
        wide = fig.radius / _BODY_RADIUS
        out.append((fig.x, fig.y, 0.0, radius * wide, radius * wide, True, strength, reach * wide))
    if room is not None:
        corners = np.asarray(room.polygon, dtype=np.float64)
        for a, b in zip(corners, np.roll(corners, -1, axis=0)):
            (x, y), (dx, dy) = (a + b) / 2, b - a
            out.append((x, y, math.atan2(dy, dx), math.hypot(dx, dy) / 2, 0.0, False, *_WALL_SHADOW))
    return [shadow for shadow in out if shadow[6] > 0]


def _draw_shadows(rays: _Rays, shadows: Sequence[tuple]) -> None:
    light = np.ones(rays.t.size)                   # the share of its light each ray's hit keeps
    for x, y, yaw, half_x, half_y, is_round, strength, reach in shadows:
        c, s = math.cos(yaw), math.sin(yaw)
        ends = np.array([(i * (half_x + reach), j * (half_y + reach)) for i in (-1, 1) for j in (-1, 1)])
        sel = rays.window(np.column_stack([x + c * ends[:, 0] - s * ends[:, 1], y + s * ends[:, 0] + c * ends[:, 1], np.zeros(4)]))
        if sel is None:
            continue
        where = np.nonzero(rays.kind == _FLOOR)[0] if isinstance(sel, slice) else sel[rays.kind[sel] == _FLOOR]
        p = rays.floor_xy(where) - (x, y)
        along, across = c * p[:, 0] + s * p[:, 1], -s * p[:, 0] + c * p[:, 1]
        if is_round:
            gap = np.maximum(np.hypot(along, across) - half_x, 0.0)
        else:
            gap = np.hypot(np.maximum(np.abs(along) - half_x, 0.0), np.maximum(np.abs(across) - half_y, 0.0))
        light[where] *= 1.0 - strength * (1.0 - np.minimum(gap / reach, 1.0)) ** 2
    rays.rgb *= np.maximum(light, 1.0 - _SHADOW_MOST)[:, None]


def _trace(rays: _Rays, room: Room | None, solids: Sequence[Solid], figures: Sequence[Figure]) -> None:
    _draw_floor(rays, room)
    if room is not None:
        _draw_ceiling(rays, room)
        _draw_walls(rays, room)
    for solid in solids:
        _draw_solid(rays, solid)
    for fig in figures:
        for part in _figure_parts(fig):
            _draw_round(rays, *part)
    _draw_shadows(rays, _shadows(room, solids, figures))


def _outlines(rays: _Rays) -> np.ndarray:
    """The pixels to sample again: those on fine detail, and those whose colour steps from a neighbour's.

    A step between two floor pixels does not count: the floor's own pattern is already a mean over each pixel."""
    shape = (rays.cam.height, rays.cam.width)
    rgb, floor = rays.rgb.reshape(*shape, 3), (rays.kind == _FLOOR).reshape(shape)
    marked = rays.detail.reshape(shape).copy()
    for one, other in ((np.s_[1:], np.s_[:-1]), (np.s_[:, 1:], np.s_[:, :-1])):
        step = (np.abs(rgb[one] - rgb[other]).max(axis=2) > _EDGE) & ~(floor[one] & floor[other])
        marked[one] |= step
        marked[other] |= step
    return np.nonzero(marked.ravel())[0]


def render(cam: Camera, boxes: Sequence[Solid] = (), figures: Sequence[Figure] = (), room: Room | None = None,
           props: Sequence[Solid] = (), antialias: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Ray-cast the ground plane z = 0, the room if there is one, the boxes, the props and the figures.

    boxes and props are drawn alike and may each hold boxes, cylinders and spheres. Without a room the ground
    is one colour and a ray that meets nothing shows the sky; inside a room every ray meets something. With
    antialias off every pixel takes the colour of the one ray through its centre.

    Returns rgb (H, W, 3) uint8 and depth (H, W) float32: the camera Z of the first hit, 0.0 where a ray
    hits nothing.
    """
    if room is not None and len(room.polygon) < 3:
        raise ValueError(f"a room's polygon needs at least 3 corners, not {len(room.polygon)}")
    solids = (*boxes, *props)
    with np.errstate(divide="ignore", invalid="ignore"):
        centres = _Rays(cam)
        _trace(centres, room, solids, figures)
        rgb = centres.rgb
        pixels = _outlines(centres) if antialias else np.zeros(0, dtype=int)
        if pixels.size:
            more = _Rays(cam, pixels, _SAMPLES)
            _trace(more, room, solids, figures)
            rgb[pixels] = (rgb[pixels] + more.rgb.reshape(len(_SAMPLES), pixels.size, 3).sum(axis=0)) / (len(_SAMPLES) + 1)
    seen = np.isfinite(centres.t)
    shape = (cam.height, cam.width)
    return (np.clip(np.rint(rgb), 0, 255).astype(np.uint8).reshape(*shape, 3),
            np.where(seen, centres.t, 0.0).astype(np.float32).reshape(shape))
