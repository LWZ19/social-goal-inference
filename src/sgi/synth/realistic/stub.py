"""The stub renderer: a scene description drawn with the placeholder's numpy ray caster, for tests without Blender.

It stands in for the Blender renderer wherever the look does not matter: it honours the same interface and
puts every surface where the description says, so what is derived from depth (the scene check, the regions,
what a tour frame shows) can be tested through it. The picture is a sketch:

- a part is its solid with the colour of its material in place of its own, or with its tint where it has one;
  with PAINT it is the solid as it is, whatever its tint;
- a model is the box it fills, every model in one colour;
- a person is the placeholder's figure at their height, widened by their build: the same figure for every
  avatar, pose and prop;
- the room is the placeholder's room with the floor and the ceiling in the colours of their materials.

Glass is drawn solid, as it is a surface in the depth. Bevels, signs, openings, the lighting preset and the
HDRI are not drawn: a wall with a window in it is a whole wall here.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from sgi.synth.realistic.spec import MATERIALS, PAINT, Frame, Part, SceneSpec
from sgi.synth.render import Box, Color, Figure, Room, Solid, render

_COLOURS: dict[str, Color] = dict(zip(MATERIALS, (             # one for each material, in the order of MATERIALS
    (200, 192, 176), (186, 182, 172), (176, 140, 100), (243, 242, 236), (233, 227, 213), (168, 124, 82), (96, 66, 44),
    (236, 236, 232), (170, 174, 180), (92, 104, 120), (48, 50, 54), (214, 214, 210), (88, 104, 132), (150, 148, 142),
    (186, 148, 104), (244, 242, 232), (178, 142, 98), (240, 238, 232), (176, 206, 214), (28, 34, 44)), strict=True))
_MODEL: Color = (120, 132, 112)                    # every model's box


def material_colour(name: str) -> Color:
    """The colour the stub paints a material in. A name that is none of MATERIALS is an error here, so that a
    description no library could serve fails in a test and not in the first real render."""
    if name not in _COLOURS:
        raise ValueError(f"unknown material {name!r}: expected one of {', '.join(MATERIALS)}"
                         + (f"; {PAINT!r} is for a part alone, which has a colour of its own" if name == PAINT else ""))
    return _COLOURS[name]


def _solid(part: Part) -> Solid:
    """The part's solid in the colour the sketch shows it in. A tint is drawn as the colour itself, although on a
    material with a colour of its own the library would only shift that colour by it. The material is asked for
    with a tint too: a name that is none of MATERIALS is an error either way."""
    if part.material == PAINT:
        return part.solid
    own = material_colour(part.material)
    return replace(part.solid, color=own if part.tint is None else tuple(int(level) for level in part.tint))


def _room(spec: SceneSpec) -> Room:
    room = spec.room
    return Room(room.polygon, height=room.height, wall=room.wall, dado=room.dado, dado_height=room.dado_height,
                floor=material_colour(room.floor), ceiling=material_colour(room.ceiling), lights=room.lights)


class StubRenderer:
    name = "stub"
    version: str | None = None

    def render(self, spec: SceneSpec, frames: Sequence[Frame]) -> list[tuple[np.ndarray | None, np.ndarray]]:
        """Per frame: rgb uint8 (None when beauty is off) and depth float32, camera Z through each pixel centre,
        0.0 where nothing is hit.

        A frame of some groups alone has depth 0.0 wherever none of them shows. Its picture, if it has one,
        shows them on the ray caster's bare ground under its sky."""
        return [self._draw(spec, frame) for frame in frames]

    def _draw(self, spec: SceneSpec, frame: Frame) -> tuple[np.ndarray | None, np.ndarray]:
        def asked_for(thing) -> bool:
            return frame.only is None or thing.group in frame.only

        solids = [_solid(part) for part in spec.parts if asked_for(part)]
        solids += [Box(model.x, model.y, model.yaw, *model.size, z0=model.z, color=_MODEL)
                   for model in spec.models if asked_for(model)]
        figures = [Figure(person.x, person.y, person.yaw, height=person.height, radius=Figure.radius * person.build)
                   for person in spec.people if asked_for(person)]
        room = _room(spec) if frame.only is None else None
        # Depth comes from the rays through the pixel centres either way; the other rays only smooth the picture.
        rgb, depth = render(frame.camera, solids, figures, room=room, antialias=frame.beauty)
        if frame.only is not None:
            # The ray caster always draws the ground, and nothing asked for lies in its plane: whatever is not
            # the bare ground's depth is theirs.
            depth = np.where(depth != render(frame.camera, antialias=False)[1], depth, np.float32(0.0))
        return (rgb if frame.beauty else None), depth
