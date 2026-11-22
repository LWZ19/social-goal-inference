"""OpenAI back end: one Responses API call per request, mapped to a BackendResult or a classified error.

SDK names were read from openai 3.26.0 and run offline. What a server sends is settled by `python -m sgi.smoke`.
"""
from __future__ import annotations

import base64
from dataclasses import replace
from typing import Sequence

import openai                                      # the SDK: imports are absolute, so this is not this module

from sgi.config import ModelEntry
from sgi.types import Usage
from sgi.vlm.base import (BackendError, BackendResult, ImagePart, Part, QuotaExceededError, TransientError,
                          _allowance_spent, strike_key)

_BASE_URL = "https://api.openai.com/v1"            # where the key is sent: never left to the shell
_SCHEMA_NAME = "answer"                            # a constant: nothing sent names a task
_MAX_RETRY_AFTER_S = 120.0                         # a longer wait is left to VLMClient's backoff
_NO_CREDIT_CODES = frozenset({"credit_balance_exhausted", "organization_spend_limit_exceeded",
                              "project_spend_limit_exceeded", "organization_usage_limit_exceeded"})


class OpenAIBackend:
    provider = "openai"

    def __init__(self, api_key: str, client=None):
        self._api_key = api_key                                  # kept to strike it from error messages
        if client is None:
            # With the endpoint given, OPENAI_BASE_URL of the shell decides nothing: the SDK would send the key there.
            client = openai.OpenAI(api_key=api_key, base_url=_BASE_URL)
        self._client = client.with_options(max_retries=0)        # one attempt: VLMClient owns the retries

    def build_request(self, entry: ModelEntry, system: str | None, parts: Sequence[Part], json_schema: dict) -> dict:
        """The keyword arguments of `responses.create`. A key is present only when its source is set."""
        request = {
            "model": entry.model,
            "input": [{"role": "user", "content": [_content(part, entry.image_detail) for part in parts]}],
            "text": {"format": {"type": "json_schema", "name": _SCHEMA_NAME, "schema": json_schema, "strict": True}},
            "reasoning": {"effort": entry.reasoning_effort},
            "store": False,
            "timeout": entry.timeout_s,
        }
        if system is not None:
            request["instructions"] = system
        if entry.temperature is not None:
            request["temperature"] = entry.temperature
        if entry.max_output_tokens is not None:
            request["max_output_tokens"] = entry.max_output_tokens
        return request

    def call(self, entry: ModelEntry, system: str | None, parts: Sequence[Part], json_schema: dict) -> BackendResult:
        request = self.build_request(entry, system, parts, json_schema)
        try:                                                     # errors are raised unchained: the SDK's exception
            resp = self._client.responses.create(**request)      # holds the key unstruck
        except (openai.APIConnectionError, openai.APIStatusError) as exc:
            raise _error(exc, self._api_key) from None
        result = _result(resp, self._api_key)
        # A refusal can echo the key, and the text goes on into the cache and the run outputs: struck out here, once.
        return result if result.text is None else replace(result, text=strike_key(result.text, self._api_key))

    def list_models(self) -> list[str]:
        """The ids of the models the key may use, sorted: one request."""
        try:
            return sorted(model.id for model in self._client.models.list())
        except (openai.APIConnectionError, openai.APIStatusError) as exc:
            raise _error(exc, self._api_key) from None


def _content(part: Part, detail: str | None) -> dict:
    if isinstance(part, ImagePart):
        image = {"type": "input_image",
                 "image_url": "data:image/png;base64," + base64.b64encode(part.png).decode("ascii")}
        if detail is not None:
            image["detail"] = detail
        return image
    return {"type": "input_text", "text": part.text}


def _result(resp, key: str) -> BackendResult:
    if resp.status not in ("completed", "incomplete"):
        code, message = getattr(resp.error, "code", None), getattr(resp.error, "message", None)
        raise TransientError(strike_key(f"openai response {resp.status}: {code}: {message}", key))
    usage, text = _usage(resp.usage), resp.output_text or None
    refusals = [content.refusal for item in resp.output if item.type == "message"
                for content in item.content if content.type == "refusal"]
    if refusals:                                                 # the status is still "completed"
        return BackendResult(refusals[0] or None, "refused", usage, resp.model)
    if resp.status == "incomplete":
        filtered = getattr(resp.incomplete_details, "reason", None) == "content_filter"
        return BackendResult(text, "refused" if filtered else "truncated", usage, resp.model)
    return BackendResult(text, "ok", usage, resp.model)


def _usage(usage) -> Usage:
    if usage is None:
        return Usage()
    details = usage.output_tokens_details
    reasoning = details.reasoning_tokens if details is not None else None
    visible = usage.output_tokens                                # the API's figure includes reasoning
    if visible is not None and reasoning is not None:
        visible -= reasoning
    return Usage(input_tokens=usage.input_tokens, output_tokens=visible, reasoning_tokens=reasoning)


def _error(exc: openai.APIError, key: str) -> Exception:
    """The SDK's error as this layer's, with the key struck out.

    A request that got no reply (a timeout, no connection), a rate limit and a 5xx are transient. Any other
    status stops the run, and so does a 429 that tells of credit or an allowance used up."""
    if not isinstance(exc, openai.APIStatusError):               # APIConnectionError, which includes APITimeoutError
        return TransientError(strike_key(f"openai: {exc}", key))
    status, message = exc.status_code, strike_key(f"openai {exc.status_code}: {exc}", key)
    if status == 429 and _allowance_spent(str(exc)):             # insufficient_quota is worded so
        return QuotaExceededError(message)
    out_of_credit = status == 429 and exc.code in _NO_CREDIT_CODES       # waiting does not restore credit
    if (status == 429 or status >= 500) and not out_of_credit:
        return TransientError(message, _retry_after(exc.response.headers))
    return BackendError(message)


def _retry_after(headers) -> float | None:
    """The wait the server asked for, in seconds: None when absent, not a number, or outside 0 to 120 s."""
    for name, per_second in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        try:
            seconds = float(headers.get(name)) / per_second
        except (TypeError, ValueError):                          # absent, or an HTTP date
            continue
        return seconds if 0.0 <= seconds <= _MAX_RETRY_AFTER_S else None
    return None
