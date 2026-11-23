"""Anthropic back end: one Messages API call per request, mapped to a BackendResult or a classified error.

Thinking. An entry's `reasoning_effort` is the model's effort level and is sent as `output_config.effort`. No
`thinking` setting goes with it: the current models think by themselves, as much as the level allows, and refuse
a thinking budget. The level `none` is this back end's name for thinking switched off: `thinking: {type: disabled}`
is sent, and no effort level. A model that cannot switch thinking off refuses that request, and what it says
stops the run as a BackendError.

Sampling. No `temperature`, `top_p` or `top_k` is sent: the current models refuse them and the SDK's call takes
none. An entry of this provider that sets `temperature` is refused when the configuration is loaded.

Caching. No part of a request is marked for the provider's prompt cache: each stage sends its own system text,
and what two requests begin with in common is too short to be worth one.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator, Sequence

import anthropic                                   # the SDK: imports are absolute, so this is not this module

from sgi.config import ModelEntry
from sgi.types import Usage
from sgi.vlm.base import (BackendError, BackendResult, ImagePart, Part, QuotaExceededError, TransientError,
                          _allowance_spent, strike_key)

_BASE_URL = "https://api.anthropic.com"            # where the key is sent: never left to the shell
_MAX_TOKENS = 16000                                # for an entry that names none: thinking counts as output
_NO_THINKING = "none"                              # the level that switches thinking off; any other is an effort
_MAX_RETRY_AFTER_S = 120.0                         # a longer wait is left to VLMClient's backoff
_NO_CREDIT = "billing_error"                       # the API's error type for a billing or payment problem: a 402
# Keywords the API's structured outputs do not take: a bound on a number, and one on the length of a string. Of
# these the pipeline's schemas hold `minimum` and `maximum`, on the two coordinates of a point.
_UNSENT = frozenset({"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength",
                     "maxLength"})
_BY_NAME = ("properties", "$defs")                 # each holds schemas under names: a name there is no keyword
_LISTS = ("anyOf", "allOf")                        # each holds a list of schemas


class AnthropicBackend:
    provider = "anthropic"

    def __init__(self, api_key: str, client=None):
        self._api_key = api_key                                  # kept to strike it from error messages
        if client is None:
            # With the key given, the SDK reads no credential of the shell or of a profile on disk; with the
            # endpoint given, ANTHROPIC_BASE_URL decides nothing: the SDK would send the key there. The header
            # lines of ANTHROPIC_CUSTOM_HEADERS are not kept out, an x-api-key among them, which is then sent in
            # place of this key: the SDK has no argument that leaves that variable unread.
            client = anthropic.Anthropic(api_key=api_key, base_url=_BASE_URL)
        self._client = client.with_options(max_retries=0)        # one attempt: VLMClient owns the retries

    def build_request(self, entry: ModelEntry, system: str | None, parts: Sequence[Part], json_schema: dict) -> dict:
        """The keyword arguments of `messages.create`. A key is present only when its source is set."""
        output = {"format": {"type": "json_schema", "schema": _sendable(json_schema)}}
        request = {
            "model": entry.model,
            "max_tokens": _MAX_TOKENS if entry.max_output_tokens is None else entry.max_output_tokens,
            "messages": [{"role": "user", "content": [_content(part) for part in parts]}],
            "output_config": output,
            "timeout": entry.timeout_s,
        }
        if entry.reasoning_effort == _NO_THINKING:
            request["thinking"] = {"type": "disabled"}
        else:
            output["effort"] = entry.reasoning_effort
        if system is not None:
            request["system"] = system
        return request

    def call(self, entry: ModelEntry, system: str | None, parts: Sequence[Part], json_schema: dict) -> BackendResult:
        request = self.build_request(entry, system, parts, json_schema)
        with _classified(self._api_key):
            resp = self._client.messages.create(**request)
        result = _result(resp)
        # A refusal can echo the key, and the text goes on into the cache and the run outputs: struck out here, once.
        return result if result.text is None else replace(result, text=strike_key(result.text, self._api_key))

    def list_models(self) -> list[str]:
        """The ids of the models the key may use, sorted: one request for each page of the listing."""
        with _classified(self._api_key):
            return sorted(model.id for model in self._client.models.list())


def _content(part: Part) -> dict:
    if isinstance(part, ImagePart):
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                            "data": base64.b64encode(part.png).decode("ascii")}}
    return {"type": "text", "text": part.text}


def _sendable(schema: dict) -> dict:
    """A copy of a JSON schema without the keywords the API's structured outputs do not take, at every depth.

    Nothing else is changed: the keys keep their order and every description is sent as written. A bound that is
    not sent is still held to, because VLMClient validates each answer against the whole schema and asks again
    on a miss. The SDK's own `transform_schema` is not used: it writes each keyword it removes into the
    description, which would give this provider a prompt of its own, and it removes `const`, which the API takes."""
    kept = {}
    for keyword, value in schema.items():
        if keyword in _UNSENT:
            continue
        if keyword in _BY_NAME:
            value = {name: _sendable(sub) for name, sub in value.items()}
        elif keyword in _LISTS:
            value = [_sendable(sub) for sub in value]
        elif keyword == "items" and isinstance(value, dict):
            value = _sendable(value)
        kept[keyword] = value
    return kept


def _result(resp) -> BackendResult:
    usage = _usage(resp.usage)
    text = "".join(block.text for block in resp.content if block.type == "text") or None      # thinking is no text
    if resp.stop_reason == "end_turn":
        return BackendResult(text, "ok", usage, resp.model)
    if resp.stop_reason == "max_tokens":                         # cut short, or the whole budget went into thinking
        return BackendResult(text, "truncated", usage, resp.model)
    if resp.stop_reason == "refusal":
        # The API can be asked to give a refused request to another model, and is not: a run measures the model
        # its entry names, so an answer has to come from that model or be recorded as refused by it.
        return BackendResult(_refusal(resp.stop_details, text), "refused", usage, resp.model)
    # No other reason can end this request, which has no tools and no stop sequence: no answer, and none recorded.
    raise TransientError(f"anthropic: the reply ended with stop reason {resp.stop_reason!r}, which is no answer")


def _refusal(details, text: str | None) -> str:
    """What a refused reply is recorded with: the category and the explanation the API gives for the refusal,
    either of which it may leave out, and under them what the model had written, if anything."""
    category, explanation = getattr(details, "category", None), getattr(details, "explanation", None)
    said = f"refusal, category {category}" if category else "refusal, no category given"
    if explanation:
        said = f"{said}: {explanation}"
    return said if text is None else f"{said}\n{text}"


def _usage(usage) -> Usage:
    if usage is None:
        return Usage()
    details = usage.output_tokens_details
    reasoning = details.thinking_tokens if details is not None else None
    visible = usage.output_tokens                                # the API's figure includes thinking
    if visible is not None and reasoning is not None:
        visible -= reasoning
    return Usage(input_tokens=usage.input_tokens, output_tokens=visible, reasoning_tokens=reasoning)


@contextmanager
def _classified(key: str) -> Iterator[None]:
    """Round one request to the API: an error of the SDK is raised as this layer's, with the key struck out.

    The SDK's classes are tried from the most specific down. A request that got no reply (a timeout, no
    connection), a rate limit, an overload and every other status of 500 or more are transient, with the wait
    the server asked for. An allowance that is used up stops the run as a QuotaExceededError: a billing error,
    which is a 402, and a 429 that names a quota or a budget. Any other status stops the run as a BackendError.
    Each is raised unchained: the SDK's exception holds the key unstruck."""
    try:
        yield
    except anthropic.APIConnectionError as exc:                  # no reply; an APITimeoutError is one of these
        raise TransientError(strike_key(f"anthropic: {exc}", key)) from None
    except anthropic.RateLimitError as exc:                      # 429
        if _allowance_spent(str(exc)):
            raise QuotaExceededError(_said(exc, key)) from None
        raise TransientError(_said(exc, key), _retry_after(exc.response.headers)) from None
    except (anthropic.OverloadedError, anthropic.InternalServerError) as exc:       # 529, and 500 and above
        raise TransientError(_said(exc, key), _retry_after(exc.response.headers)) from None
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError,
            anthropic.BadRequestError) as exc:                   # 401, 403, 404, 400: asking again changes nothing
        raise BackendError(_said(exc, key)) from None
    except anthropic.APIStatusError as exc:                      # every other status, with a class of its own or not
        if exc.status_code == 402 or exc.type == _NO_CREDIT:     # waiting does not restore credit
            raise QuotaExceededError(_said(exc, key)) from None
        if exc.status_code >= 500:
            raise TransientError(_said(exc, key), _retry_after(exc.response.headers)) from None
        raise BackendError(_said(exc, key)) from None


def _said(exc: anthropic.APIStatusError, key: str) -> str:
    """The message of a reply with an HTTP error status: the status, then the server's words, without the key."""
    return strike_key(f"anthropic {exc.status_code}: {exc}", key)


def _retry_after(headers) -> float | None:
    """The wait the server asked for, in seconds: None when absent, not a number, or outside 0 to 120 s."""
    try:
        seconds = float(headers.get("retry-after"))
    except (TypeError, ValueError):                              # absent, or an HTTP date
        return None
    return seconds if 0.0 <= seconds <= _MAX_RETRY_AFTER_S else None
