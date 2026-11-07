"""Locate: one model call that marks an already planned subtask's target in a given image."""
from __future__ import annotations

from pathlib import Path

from sgi.config import PROMPTS_DIR, Strategy
from sgi.prompts import build_locate_request
from sgi.schemas import location_from_wire, location_schema
from sgi.types import Location, Subtask
from sgi.vlm.base import ImagePart, VLMClient, VLMResponse


def run_locate(client: VLMClient, strategy: Strategy, request: str, context: str | None, subtask: Subtask,
               image: ImagePart, trial: int = 0,
               prompts_dir: Path = PROMPTS_DIR) -> tuple[Location | None, VLMResponse]:
    """The location and the response it came from; the location is None when the response is not ok.

    A target the model does not find is an ok answer: a Location whose `target_point` is None.
    """
    pf, size = client.entry.point_format, (image.width, image.height)
    system, parts = build_locate_request(strategy, request, context, subtask, image, pf, prompts_dir=prompts_dir)
    resp = client.generate(parts, location_schema(strategy, pf, size), system=system, trial=trial)
    if resp.status != "ok":
        return None, resp
    return location_from_wire(resp.parsed, pf, size), resp
