"""Gemini back end: one Interactions API call per request, mapped to a BackendResult or a classified error.

SDK names were read from google-genai 2.28.0 and run offline. What a server sends is settled by `python -m sgi.smoke`.
"""
from __future__ import annotations

import base64
from dataclasses import replace
from typing import Sequence

import httpx                                       # the HTTP library under the SDK: the listing raises its errors
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from sgi.config import ModelEntry
from sgi.types import Usage
from sgi.vlm.base import BackendError, BackendResult, ImagePart, Part, TransientError, strike_key

_BASE_URL = "https://generativelanguage.googleapis.com"         # where the key is sent: never left to the shell
_BLOCK_CODES = {"safety", "recitation", "language", "prohibited_content", "spii", "blocklist", "content_blocked"}
_NETWORK_ERRORS = ("APITimeoutError", "APIConnectionError")      # importable only from a private SDK module
_MAX_RETRY_AFTER_S = 120.0                         # a longer wait is left to VLMClient's backoff
_MODEL_PREFIX = "models/"                          # the listing names a model as a resource; an entry holds the id


class _Silent:
    """In place of the SDK's debug logger, which with GOOGLE_GENAI_DEBUG set in the shell prints the headers of
    every request, the key among them."""

    def debug(self, *args, **kwargs) -> None:
        pass


class GeminiBackend:
    provider = "gemini"

    def __init__(self, api_key: str, client=None):
        self._api_key = api_key                                  # kept to strike it from error messages
        if client is None:
            # Given whole, so that GOOGLE_GEMINI_BASE_URL and the Vertex switches of the shell decide nothing:
            # the SDK would send the key to whatever host they name.
            client = genai.Client(api_key=api_key, vertexai=False,
                                  http_options=genai_types.HttpOptions(base_url=_BASE_URL))
        self._client = client
        config = self._client.interactions.sdk_configuration
        config.retry_config = None                               # one attempt: VLMClient owns the retries
        config.debug_logger = _Silent()                          # keys are never logged

    def build_request(self, entry: ModelEntry, system: str | None, parts: Sequence[Part], json_schema: dict) -> dict:
        """The keyword arguments of `interactions.create`. A key is present only when its source is set."""
        generation = {"thinking_level": entry.reasoning_effort}
        if entry.temperature is not None:
            generation["temperature"] = entry.temperature
        if entry.max_output_tokens is not None:
            generation["max_output_tokens"] = entry.max_output_tokens
        request = {
            "model": entry.model,
            "input": [_content(part, entry.image_detail) for part in parts],
            "response_format": {"type": "text", "mime_type": "application/json", "schema": json_schema},
            "generation_config": generation,
            "store": False,
            "timeout": entry.timeout_s,
        }
        if system is not None:
            request["system_instruction"] = system
        return request

    def call(self, entry: ModelEntry, system: str | None, parts: Sequence[Part], json_schema: dict) -> BackendResult:
        request = self.build_request(entry, system, parts, json_schema)
        try:
            inter = self._client.interactions.create(**request)
        except Exception as exc:
            result = _refusal_or_error(exc, self._api_key)
            if result is None:
                raise                                            # not a provider error: unchanged
            if isinstance(result, Exception):
                raise result from None                           # unchained: the SDK's exception holds the key unstruck
        else:
            result = _result(inter, self._api_key)
        # A refusal can echo the key, and the text goes on into the cache and the run outputs: struck out here, once.
        return result if result.text is None else replace(result, text=strike_key(result.text, self._api_key))

    def list_models(self) -> list[str]:
        """The ids of the models the key may use, sorted: one request for each page of the listing.

        The listing is not part of the Interactions API. It is the SDK's older client, built on the same key and
        address, which never retries unless told to and raises errors of its own kind."""
        try:
            names = [model.name for model in self._client.models.list()]
        except genai_errors.APIError as exc:                     # a reply with an HTTP error status
            raise _status_error(exc.code, exc, self._api_key) from None
        except httpx.TransportError as exc:                      # no reply: a timeout, no connection
            raise TransientError(strike_key(f"gemini: {exc}", self._api_key)) from None
        return sorted(name.removeprefix(_MODEL_PREFIX) for name in names if name)


def _content(part: Part, resolution: str | None) -> dict:
    if isinstance(part, ImagePart):
        image = {"type": "image", "data": base64.b64encode(part.png).decode("ascii"), "mime_type": "image/png"}
        if resolution is not None:
            image["resolution"] = resolution
        return image
    return {"type": "text", "text": part.text}


def _result(inter, key: str) -> BackendResult:
    usage, text = _usage(inter.usage), inter.output_text or None
    if inter.status == "completed":
        return BackendResult(text, "ok", usage, inter.model)
    if inter.status == "incomplete":
        return BackendResult(text, "truncated", usage, inter.model)
    errors = inter.errors or []
    for error in errors:
        if error.code in _BLOCK_CODES:
            return BackendResult(error.message or None, "refused", usage, inter.model)
    named = "; ".join(f"{error.code}: {error.message}" for error in errors) or "no error given"
    raise TransientError(strike_key(f"gemini interaction {inter.status}: {named}", key))


def _usage(usage) -> Usage:
    if usage is None:
        return Usage()
    return Usage(input_tokens=usage.total_input_tokens, output_tokens=usage.total_output_tokens,
                 reasoning_tokens=usage.total_thought_tokens)


def _refusal_or_error(exc: Exception, key: str) -> BackendResult | Exception | None:
    """A failed call, classified by attribute: a refusal to return, an error to raise, or None for neither."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        network = type(exc).__name__ in _NETWORK_ERRORS
        return TransientError(strike_key(f"gemini: {exc}", key)) if network else None
    body = getattr(exc, "body", None)                            # parsed JSON, or plain text
    error = body.get("error") if isinstance(body, dict) else None
    code, said = (error.get("code"), error.get("message")) if isinstance(error, dict) else (None, None)
    if isinstance(code, str) and code in _BLOCK_CODES:
        return BackendResult(said if isinstance(said, str) and said else None, "refused", Usage())
    return _status_error(status, exc, key)


def _status_error(status: int, exc: Exception, key: str) -> Exception:
    """A reply with an HTTP error status as this layer's error, with the key struck out: a rate limit and a 5xx
    are transient, with the wait the server asked for; any other status stops the run."""
    message = strike_key(f"gemini {status}: {exc}", key)
    if status == 429 or status >= 500:
        return TransientError(message, _retry_after(getattr(getattr(exc, "response", None), "headers", None) or {}))
    return BackendError(message)


def _retry_after(headers) -> float | None:
    """The wait the server asked for, in seconds: None when absent, not a number, or outside 0 to 120 s."""
    try:
        seconds = float(headers.get("retry-after"))
    except (TypeError, ValueError):                              # absent, or an HTTP date
        return None
    return seconds if 0.0 <= seconds <= _MAX_RETRY_AFTER_S else None
