"""Point formats: the wire point type, its prompt instruction and the conversion to the internal Point."""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, create_model

from sgi.types import Point

_FULL_SCALE = 1000                                 # the internal Point runs from 0 to this on both axes


class PointFormat(Protocol):
    name: str

    def wire_model(self, width: int, height: int) -> type[BaseModel]: ...
    def instruction(self, width: int, height: int) -> str: ...
    def to_internal(self, wire: dict, width: int, height: int) -> Point: ...
    def from_internal(self, point: Point, width: int, height: int) -> dict: ...


@lru_cache(maxsize=None)
def _wire_model(class_name: str, x_max: int, y_max: int) -> type[BaseModel]:
    return create_model(
        class_name,
        __config__=ConfigDict(extra="forbid"),
        x=(int, Field(ge=0, le=x_max, description=f"0 = left edge, {x_max} = right edge")),
        y=(int, Field(ge=0, le=y_max, description=f"0 = top edge, {y_max} = bottom edge")),
    )


class _Normalised:
    name = "xy_1000"

    def wire_model(self, width: int, height: int) -> type[BaseModel]:
        return _wire_model("Point1000", _FULL_SCALE, _FULL_SCALE)

    def instruction(self, width: int, height: int) -> str:
        return ('Give each point as {"x": <integer>, "y": <integer>} in coordinates normalised to the image: '
                f"x runs from 0 at the left edge to {_FULL_SCALE} at the right edge and y from 0 at the top edge "
                f"to {_FULL_SCALE} at the bottom edge, whatever the image size.")

    def to_internal(self, wire: dict, width: int, height: int) -> Point:
        return Point(x=wire["x"], y=wire["y"])

    def from_internal(self, point: Point, width: int, height: int) -> dict:
        return {"x": point.x, "y": point.y}


class _Pixel:
    name = "xy_pixel"

    def wire_model(self, width: int, height: int) -> type[BaseModel]:
        return _wire_model(f"PixelPoint_{width}x{height}", width, height)

    def instruction(self, width: int, height: int) -> str:
        return ('Give each point as {"x": <integer>, "y": <integer>} in pixels of the image, '
                f"which is {width} pixels wide and {height} pixels high: x runs from 0 at the left edge to {width} "
                f"at the right edge and y from 0 at the top edge to {height} at the bottom edge.")

    def to_internal(self, wire: dict, width: int, height: int) -> Point:
        return Point(x=round(wire["x"] * _FULL_SCALE / width), y=round(wire["y"] * _FULL_SCALE / height))

    def from_internal(self, point: Point, width: int, height: int) -> dict:
        return {"x": round(point.x * width / _FULL_SCALE), "y": round(point.y * height / _FULL_SCALE)}


POINT_FORMATS: dict[str, PointFormat] = {fmt.name: fmt for fmt in (_Normalised(), _Pixel())}
