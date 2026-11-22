"""PNG encoding and the marks drawn on an image: the target marker, plain points and the numbers on people."""
from __future__ import annotations

import io
import math
from functools import lru_cache
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sgi.types import DetectedPerson
from sgi.vlm.base import ImagePart

_MAGENTA = (255, 0, 255)
_WHITE = (255, 255, 255)
_WORD = "TARGET"
_WORD_GAP = 4                                      # pixels between the ring and the word's box
_WORD_PADDING = 2                                  # pixels of white around the word
_DISC = (255, 221, 0)                              # a person's number stands in a yellow disc,
_INK = (0, 0, 0)                                   # with its rim and the number in black: unlike the target's marker
_DISC_SHARE = 0.019                                # the disc's radius as a share of the image's short side,
_DISC_MIN = 8                                      # and the least it is, in pixels


def _to_image(rgb: np.ndarray) -> Image.Image:
    """A Pillow image holding its own copy of the pixels."""
    if not (isinstance(rgb, np.ndarray) and rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8):
        raise ValueError(f"expected an (H, W, 3) uint8 array, got shape {getattr(rgb, 'shape', None)} "
                         f"and dtype {getattr(rgb, 'dtype', None)}")
    return Image.fromarray(rgb)


@lru_cache(maxsize=None)
def _font(short_side: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=max(12, round(0.04 * short_side)))


@lru_cache(maxsize=None)
def _number_font(radius: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=round(1.3 * radius))      # a number of two digits is then narrower than its disc


def encode_png(rgb: np.ndarray, max_side: int | None = None) -> ImagePart:
    """Encode losslessly; `width` and `height` of the result describe the image as encoded.

    A larger image is scaled down until its longer side is `max_side`; None sends it as it is, and a limit
    below one pixel is a ValueError."""
    if max_side is not None and max_side < 1:
        raise ValueError(f"max_side is {max_side}: it is the longest side in pixels, 1 or more; None leaves the "
                         "image as it is")
    image = _to_image(rgb)
    longest = max(image.size)
    if max_side is not None and longest > max_side:
        scale = max_side / longest
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return ImagePart(buffer.getvalue(), image.width, image.height)


def draw_target_marker(rgb: np.ndarray, u: float, v: float) -> np.ndarray:
    """A copy with a magenta ring around (u, v) and the word TARGET beside it."""
    image = _to_image(rgb)
    draw = ImageDraw.Draw(image)
    short_side = min(image.size)
    radius = max(8, round(0.035 * short_side))                         # 17 at 640 x 480
    draw.ellipse([u - radius, v - radius, u + radius, v + radius], outline=_MAGENTA,
                 width=max(2, round(0.006 * short_side)))              # 3 at 640 x 480
    font = _font(short_side)
    left, top, right, bottom = font.getbbox(_WORD)
    box_w, box_h = right - left + 2 * _WORD_PADDING, bottom - top + 2 * _WORD_PADDING
    x0 = max(0, min(round(u - box_w / 2), image.width - box_w))        # centred on u, kept inside the image
    y0 = round(v - radius - _WORD_GAP - box_h)                         # above the ring
    if y0 < 0:
        y0 = round(v + radius + _WORD_GAP + 1)                         # no room above: below the ring
    draw.rectangle([x0, y0, x0 + box_w - 1, y0 + box_h - 1], fill=_WHITE)
    draw.text((x0 + _WORD_PADDING - left, y0 + _WORD_PADDING - top), _WORD, fill=_MAGENTA, font=font)
    return np.array(image)


def draw_point(rgb: np.ndarray, u: float, v: float, color=(255, 0, 0), label: str | None = None) -> np.ndarray:
    """A copy with a filled disc at (u, v) and, when given, the label to its right."""
    image = _to_image(rgb)
    draw = ImageDraw.Draw(image)
    short_side = min(image.size)
    radius = max(3, round(0.012 * short_side))
    draw.ellipse([u - radius, v - radius, u + radius, v + radius], fill=color, outline=_WHITE, width=1)
    if label is not None:
        draw.text((u + radius + 3, v), label, fill=color, font=_font(short_side), anchor="lm")
    return np.array(image)


def draw_people_numbers(rgb: np.ndarray, people: Sequence[DetectedPerson], camera_xy) -> np.ndarray:
    """A copy with each person's number in a filled disc on their body point, `pixel`.

    The disc's radius is 0.019 of the image's short side, at least 8 pixels. The discs are placed nearest the
    camera first: by the distance on the ground from `camera_xy`, the camera's ground position in the world
    frame, to the person's centre. One that would touch a disc placed before it is moved up the image by its own
    diameter until it is clear, and a line joins it to the body point. A disc is never moved out of the image's
    upper edge: where no room is left above, it stays and overlaps.
    """
    image = _to_image(rgb)
    draw = ImageDraw.Draw(image)
    radius = max(_DISC_MIN, round(_DISC_SHARE * min(image.size)))    # 9 at 640 x 480
    nearest_first = sorted(people, key=lambda person: (math.hypot(person.xy[0] - camera_xy[0],
                                                                  person.xy[1] - camera_xy[1]), person.number))
    placed: list[tuple[float, float]] = []                           # the middles of the discs, in that order
    for person in nearest_first:
        u, v = person.pixel
        while v - 2 * radius >= radius and any(math.hypot(u - a, v - b) < 2 * radius for a, b in placed):
            v -= 2 * radius
        placed.append((u, v))
    for person, (u, v) in zip(nearest_first, placed):                # the lines first: no line crosses a number
        if v != person.pixel[1]:
            draw.line([person.pixel, (u, v)], fill=_INK, width=3)
            draw.line([person.pixel, (u, v)], fill=_DISC, width=1)
    for person, (u, v) in zip(nearest_first, placed):
        draw.ellipse([u - radius, v - radius, u + radius, v + radius], fill=_DISC, outline=_INK, width=1)
        draw.text((u, v), str(person.number), fill=_INK, font=_number_font(radius), anchor="mm")
    return np.array(image)
