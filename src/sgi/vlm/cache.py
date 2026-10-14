"""Disk cache of model answers and the key that identifies one request."""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Sequence

from sgi.config import ModelEntry
from sgi.vlm.base import ImagePart, Part, VLMResponse

# The fields of an entry that cannot change an answer. A field added to ModelEntry belongs here unless it can: the
# key of a stored answer covers every field that is not listed.
_NOT_IN_KEY = {"timeout_s", "max_requests_per_minute", "price_input_per_mtok", "price_output_per_mtok"}


def _part_payload(part: Part) -> dict:
    if isinstance(part, ImagePart):
        return {"image_sha256": hashlib.sha256(part.png).hexdigest(), "width": part.width, "height": part.height}
    return {"text": part.text}


def cache_key(entry: ModelEntry, system: str | None, parts: Sequence[Part], json_schema: dict, trial: int) -> str:
    """SHA-256 over everything that shapes the answer. The timeout, the request limit and the prices do not, so
    changing one of them keeps the cache."""
    payload = {"entry": entry.model_dump(exclude=_NOT_IN_KEY), "system": system,
               "parts": [_part_payload(part) for part in parts], "schema": json_schema, "trial": trial}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class Cache:
    """One JSON file per key. Responses are stored and returned as given; `cached` is the client's to set."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> VLMResponse | None:
        """The stored response, or None for a miss. An entry that cannot be read back is a miss, with one warning."""
        path = self._path(key)
        try:
            return VLMResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except ValueError as exc:                  # cut short, or not a VLMResponse: pydantic's error and a decode error
            warnings.warn(f"cache entry {path} cannot be read back ({type(exc).__name__}); it counts as a miss",
                          RuntimeWarning)          # the file is left as it is: the fresh answer overwrites it in put
            return None

    def put(self, key: str, response: VLMResponse) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        partial = self.directory / f"{key}.json.tmp"
        partial.write_text(response.model_dump_json(), encoding="utf-8")
        os.replace(partial, self._path(key))                 # atomic: an interrupted run leaves no half-written entry
