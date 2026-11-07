"""Stage 2: one model call that says which people matter and whether the standard goal would do."""
from __future__ import annotations

from pathlib import Path

from sgi.config import PROMPTS_DIR, Strategy
from sgi.prompts import build_context_request
from sgi.schemas import context_from_wire, context_schema
from sgi.types import ContextCheck, Subtask
from sgi.vlm.base import ImagePart, VLMClient, VLMResponse


def run_context(client: VLMClient, strategy: Strategy, request: str, subtask: Subtask, marked_image: ImagePart,
                standoff_m: float, trial: int = 0,
                prompts_dir: Path = PROMPTS_DIR) -> tuple[ContextCheck | None, VLMResponse]:
    """The context check and the response it came from; the check is None when the response is not ok.

    `marked_image` already carries the target marker.
    """
    system, parts = build_context_request(strategy, request, subtask, marked_image, standoff_m,
                                          prompts_dir=prompts_dir)
    resp = client.generate(parts, context_schema(strategy), system=system, trial=trial)
    if resp.status != "ok":
        return None, resp
    return context_from_wire(resp.parsed), resp
