"""The scene description of the realistic set: what a renderer is asked to draw, as plain data.

A SceneSpec names a room, parts (the placeholder renderer's boxes, cylinders and spheres, each with a
material), library models, people, and the light. It says nothing of how any of that is drawn: the Blender
side builds it with real materials, meshes and avatars, and the stub draws it with the numpy ray caster.
Everything is in world coordinates, in metres and radians, as a scene file is.

An object group is a positive number that parts, models and people carry. The things with one number draw
one target together, and a Frame can ask for some groups alone. Group 0 is no group.

to_json gives a description as JSON holds it and from_json reads it back. The Blender side, which cannot
import this module, reads that JSON; its layout is the classes below, field for field:

    {"room": {..., "openings": [{...}]},
     "parts": [{"solid": {"shape": "box", ...}, ..., "signs": [{..., "art": {...}}]}],
     "models": [{...}], "people": [{...}], "lighting": "ceiling", "hdri": null, "seed": 0}

A tuple is written as a list, None as null, and a solid as the fields of its Box, Cylinder or Sphere beside
"shape": "box", "cylinder" or "sphere".
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Protocol, Sequence

import numpy as np

from sgi.geometry import Camera
from sgi.synth.render import Box, Color, Cylinder, Solid, Sphere

# The names the scene description and the asset library share. A material is one of MATERIALS, which the asset library provides, or PAINT.
SIGN_STYLES = ("label", "logo", "pictogram", "plate", "menu", "map", "exit", "notice", "screen", "paper")
MATERIALS = ("floor_tile", "floor_terrazzo", "floor_laminate", "ceiling_tile", "wall_paint", "wood", "wood_dark",
             "laminate_white", "steel", "painted_metal", "plastic_dark", "plastic_light", "fabric", "stone", "cork",
             "paper", "cardboard", "ceramic", "glass", "screen")
PAINT = "paint"                                    # the pseudo-material: a plain coat of the solid's own colour


@dataclass(frozen=True)
class SignArt:
    """A picture to be drawn on a rectangle."""
    style: str                                     # one of SIGN_STYLES
    text: str = ""
    ink: Color = (245, 245, 245)
    ground: Color = (40, 40, 40)
    width_m: float = 1.0                           # of the rectangle it fills
    height_m: float = 0.3
    seed: int = 0                                  # for art with drawn content


@dataclass(frozen=True)
class Sign:
    """Sign art on one face of a box."""
    face: str                                      # as Box.label_face
    art: SignArt
    rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)   # left, bottom, right, top, as shares of the face
    emission: float = 0.0                          # how strongly it glows; 0.0 for a printed sign


@dataclass(frozen=True)
class Part:
    """One primitive of a construction."""
    solid: Solid                                   # in world coordinates
    material: str                                  # one of MATERIALS, or PAINT for the solid's own colour
    group: int = 0                                 # the object group it draws, 0 for none
    bevel: float = 0.004                           # how far its edges are rounded off, in metres; 0.0 leaves them sharp
    signs: tuple[Sign, ...] = ()
    glass: bool = False                            # seen through in the picture; a surface in the depth
    tint: Color | None = None                      # multiplies the colour of a material of MATERIALS: one that has no
    #                                                colour of its own comes out in the tint. None leaves the material
    #                                                as the library has it; PAINT is the solid's colour and ignores it


@dataclass(frozen=True)
class ModelRef:
    """A model of the asset library, placed."""
    asset: str
    x: float                                       # where its bottom centre stands; its front along yaw
    y: float
    z: float
    yaw: float
    scale: tuple[float, float, float]
    size: tuple[float, float, float]               # the box it fills once scaled: along its front, across, up
    group: int = 0


@dataclass(frozen=True)
class PersonRef:
    """A person: an avatar of the asset library in one of its poses."""
    person_id: str
    avatar: str
    pose: str
    x: float                                       # the ground point under the pelvis; the chest faces yaw
    y: float
    yaw: float
    height: float
    build: float = 1.0                             # how much wider than the avatar, across and in depth
    prop: str | None = None                        # "phone" | "cup"
    group: int = 0


@dataclass(frozen=True)
class Opening:
    """What is cut into a wall of the room for a window or a door."""
    kind: str                                      # "window" | "door"
    x: float                                       # on the wall, +x into the room
    y: float
    yaw: float
    width: float
    z0: float
    height: float


@dataclass(frozen=True)
class RoomSpec:
    polygon: tuple[tuple[float, float], ...]       # the floor's outline, as Room.polygon
    height: float
    floor: str                                     # material names
    ceiling: str
    wall: Color
    dado: Color | None
    dado_height: float
    lights: float                                  # spacing of the ceiling fittings, as Room.lights
    openings: tuple[Opening, ...] = ()


@dataclass(frozen=True)
class SceneSpec:
    room: RoomSpec
    parts: tuple[Part, ...] = ()
    models: tuple[ModelRef, ...] = ()
    people: tuple[PersonRef, ...] = ()
    lighting: str = "ceiling"                      # a preset
    hdri: str | None = None                        # an HDRI of the asset library, seen through the openings
    seed: int = 0


_SHAPES = {Box: "box", Cylinder: "cylinder", Sphere: "sphere"}
_SOLIDS = {name: cls for cls, name in _SHAPES.items()}


def _plain(value):
    """A value of a description as JSON holds it: an object for a dataclass, a list for a tuple, and Python's
    own number for one that numpy made, which json.dumps would refuse."""
    if is_dataclass(value):
        shape = {"shape": _SHAPES[type(value)]} if type(value) in _SHAPES else {}
        return {**shape, **{field.name: _plain(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tuples(value):
    """A value read from JSON with every list a tuple again, at any depth."""
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def _made(cls, data: dict, **nested):
    """The dataclass with the object's keys as its fields; `nested` replaces those that hold dataclasses.

    A key the object lacks takes the field's default, so a description written before a field was added
    still reads. A key that is no field is an error."""
    return cls(**{**{key: _tuples(value) for key, value in data.items()}, **nested})


def _solid(data: dict) -> Solid:
    shape = data.get("shape")
    if shape not in _SOLIDS:
        raise ValueError(f"unknown shape {shape!r} of a solid: expected one of {', '.join(_SOLIDS)}")
    return _made(_SOLIDS[shape], {key: value for key, value in data.items() if key != "shape"})


def _part(data: dict) -> Part:
    signs = tuple(_made(Sign, sign, art=_made(SignArt, sign["art"])) for sign in data.get("signs", ()))
    return _made(Part, data, solid=_solid(data["solid"]), signs=signs)


def to_json(spec: SceneSpec) -> dict:
    """The description as JSON holds it: json.dumps takes it, and from_json(to_json(spec)) == spec.

    It holds no tuple, so it equals what json.loads gives back of it: a stored description can be compared
    with one that is about to be rendered."""
    return _plain(spec)


def from_json(data: dict) -> SceneSpec:
    """The description that to_json wrote."""
    openings = tuple(_made(Opening, opening) for opening in data["room"].get("openings", ()))
    room = _made(RoomSpec, data["room"], openings=openings)
    return _made(SceneSpec, data, room=room, parts=tuple(_part(part) for part in data.get("parts", ())),
                 models=tuple(_made(ModelRef, model) for model in data.get("models", ())),
                 people=tuple(_made(PersonRef, person) for person in data.get("people", ())))


@dataclass(frozen=True)
class Frame:
    """One picture to take of a scene."""
    camera: Camera
    beauty: bool = True                            # False: depth only
    only: tuple[int, ...] | None = None            # draw only these object groups, without the room


class Renderer(Protocol):
    name: str                                      # written to source.simulator after "sgi-synth-"
    version: str | None

    def render(self, spec: SceneSpec, frames: Sequence[Frame]) -> list[tuple[np.ndarray | None, np.ndarray]]:
        """Per frame: rgb uint8 (None when beauty is off) and depth float32, camera Z through each pixel centre,
        0.0 where nothing is hit."""
        ...
