"""Stage 1: one model call that splits the request into subtasks and grounds each navigation target."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from sgi.config import PROMPTS_DIR, Strategy
from sgi.prompts import build_plan_request
from sgi.schemas import plan_from_wire, plan_schema
from sgi.types import Plan
from sgi.vlm.base import ImagePart, VLMClient, VLMResponse


def run_plan(client: VLMClient, strategy: Strategy, request: str, context: str | None, image: ImagePart | None,
             tour: Sequence[tuple[int, ImagePart]] | None, trial: int = 0,
             prompts_dir: Path = PROMPTS_DIR) -> tuple[Plan | None, VLMResponse]:
    """The plan and the response it came from; the plan is None when the response is not ok.

    View mode marks points on `image`. Tour mode shows `tour`, pairs of a tour index and its frame, and grounds
    a navigation subtask to one of those indices in place of a point. `image_sha256` is left None: only the
    caller has the array the image was encoded from.
    """
    if (image is None) == (tour is None):
        raise ValueError("run_plan takes exactly one of image and tour")
    pf = client.entry.point_format
    system, parts = build_plan_request(strategy, request, context, image, tour, pf, prompts_dir=prompts_dir)
    if image is not None:
        sent, tour_indices = image, None
    else:
        sent, tour_indices = tour[0][1], [index for index, _ in tour]      # a tour answer holds no point to scale
    size = (sent.width, sent.height)
    resp = client.generate(parts, plan_schema(strategy, pf, size, tour_indices), system=system, trial=trial)
    if resp.status != "ok":
        return None, resp
    return plan_from_wire(resp.parsed, pf, size), resp
