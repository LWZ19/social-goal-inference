"""A scripted back end for tests, and the built-in responder that answers with the image centre."""
from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from sgi.config import ModelEntry
from sgi.types import Usage
from sgi.vlm.base import BackendResult, Part


class MockBackend:
    provider = "mock"

    def __init__(self, responder: Callable[[Sequence[Part], dict], dict | str | Exception | BackendResult]):
        self._responder = responder
        self.calls: list[tuple[str | None, Sequence[Part], dict]] = []     # (system, parts, json_schema) per call

    def call(self, entry: ModelEntry, system: str | None, parts: Sequence[Part], json_schema: dict) -> BackendResult:
        self.calls.append((system, list(parts), json_schema))              # first, so a call that raises is counted
        answer = self._responder(parts, json_schema)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, BackendResult):
            return answer
        return BackendResult(answer if isinstance(answer, str) else json.dumps(answer), "ok", Usage())


def centre_responder(parts: Sequence[Part], json_schema: dict) -> dict:
    """Answer any schema: every point at the image centre, the first allowed value of every choice."""
    return _fill(json_schema, json_schema.get("$defs", {}))


def _fill(node: dict, defs: dict) -> Any:
    """A value for one schema node. The first matching rule wins."""
    if "$ref" in node:
        return _fill(defs[node["$ref"].removeprefix("#/$defs/")], defs)
    if "const" in node:
        return node["const"]
    if "enum" in node:
        return node["enum"][0]
    if "anyOf" in node:
        return _fill(next(branch for branch in node["anyOf"] if branch.get("type") != "null"), defs)
    kind = node.get("type")
    if kind == "object":
        return {key: _fill(sub, defs) for key, sub in node.get("properties", {}).items()}
    if kind == "array":
        return [_fill(node.get("items", {}), defs) for _ in range(max(1, node.get("minItems", 0)))]
    if kind in ("integer", "number"):
        low = node.get("minimum", 0)
        value = (low + node["maximum"]) // 2 if "maximum" in node else low     # the midpoint: the image centre
        return float(value) if kind == "number" else value
    if kind == "boolean":
        return True
    if kind == "null":
        return None
    return "mock"
