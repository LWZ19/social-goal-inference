"""OpenAI-compatible back end: one chat-completions call per request, to the address a model entry names.

For a gateway that speaks the OpenAI chat-completions format without being OpenAI.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Sequence

import openai

from sgi.config import ConfigError, ModelEntry
from sgi.types import Usage
from sgi.vlm.base import (BackendError, BackendResult, ImagePart, Part, QuotaExceededError, TransientError,
                          _allowance_spent, strike_key)

_SCHEMA_NAME = "answer"                            # a constant: nothing sent names a task
_MAX_RETRY_AFTER_S = 120.0                         # a longer wait is left to VLMClient's backoff
_LISTING = "the model listing"                     # what a failed list_models was asking for
_ENGINE_MISMATCH = "only supports one backend per engine"   # a 400 from an engine set to another structured-output
_MISMATCH_TRIES = 10                                         # backend: asked again, this many times in all
_SHOWN_CHARS = 300                                 # of a reply that holds no answer: enough to tell what came back
_USUAL_FIELDS = frozenset({"id", "object", "created", "owned_by"})      # of a listed model: every service gives them
# Shell variables the SDK client turns into request headers, also on a copy of a client passed in: header lines of
# any name, among them an Authorization that is sent in place of the client's key, an organization and a project.
# The client has no argument that leaves them unread.
_SHELL_HEADERS = ("OPENAI_CUSTOM_HEADERS", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID")


class OpenAICompatibleBackend:
    provider = "openai_compatible"

    def __init__(self, api_key: str, base_url: str, client=None):
        if not (base_url and base_url.strip()):
            # Without an address the SDK uses its own, OpenAI's, and would send a gateway's key there.
            raise ValueError("an OpenAI-compatible back end needs the address of its service: base_url is empty")
        _refuse_shell_headers(base_url)                          # before a client exists that would send them
        self._api_key = api_key                                  # kept to strike it from error messages
        self._base_url = base_url
        # The key, the address and the retries are given whole, to a client passed in too: the key goes to the
        # entry's address, whatever OPENAI_BASE_URL of the shell says, in one attempt (VLMClient owns the retries).
        if client is None:
            self._client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        else:
            self._client = client.with_options(api_key=api_key, base_url=base_url, max_retries=0)

    def build_request(self, entry: ModelEntry, system: str | None, parts: Sequence[Part], json_schema: dict) -> dict:
        """The keyword arguments of `chat.completions.create`. A key is present only when its source is set."""
        messages = [] if system is None else [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": [_content(part, entry.image_detail) for part in parts]})
        request = {
            "model": entry.model,
            "messages": messages,
            # In the body, not as the SDK's own parameter: that one takes no level called xhigh or max.
            "extra_body": {"reasoning_effort": entry.reasoning_effort},
            "timeout": entry.timeout_s,
        }
        if entry.structured_output == "json_schema":
            request["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": _SCHEMA_NAME, "strict": True, "schema": json_schema}}
        elif entry.structured_output == "json_object":
            request["response_format"] = {"type": "json_object"}
        if entry.temperature is not None:
            request["temperature"] = entry.temperature
        if entry.max_output_tokens is not None:
            request["max_tokens"] = entry.max_output_tokens
        return request

    def call(self, entry: ModelEntry, system: str | None, parts: Sequence[Part], json_schema: dict) -> BackendResult:
        """One answer. A 400 saying the engine is set to another structured-output backend is asked again, up to
        _MISMATCH_TRIES times in all, with no wait: a gateway can serve one model name from several engines, each
        locked to the backend of the first request it took. An engine on 'guidance' is sent the schema in a form
        that backend takes (_for_guidance), any other the schema as given."""
        plain = self.build_request(entry, system, parts, json_schema)
        guided = self.build_request(entry, system, parts, _for_guidance(json_schema)) if json_schema else plain
        request = plain
        for attempt in range(1, _MISMATCH_TRIES + 1):
            try:                                                 # errors are raised unchained: the SDK's exception
                resp = self._client.chat.completions.create(**request)  # holds the key unstruck
            except openai.APIStatusError as exc:
                if exc.status_code == 400 and _ENGINE_MISMATCH in str(exc) and attempt < _MISMATCH_TRIES:
                    # the gateway's advice: use a schema the engine's backend takes. Same answers are valid either way.
                    request = guided if "using 'guidance'" in str(exc) else plain
                    continue
                raise self._error(exc, entry.model) from None
            except openai.APIConnectionError as exc:
                raise self._error(exc, entry.model) from None
            except json.JSONDecodeError as exc:                  # a 200 whose body is not the JSON it is sent as:
                said = _shown(exc.doc, self._api_key)            # the SDK's own error holds that body whole
                raise TransientError(f"{entry.model}: the reply is not JSON: {said}") from None
            return _result(resp, entry.model, self._api_key)

    def list_models(self) -> list[str]:
        """One line per model the gateway offers this key, sorted by id: one request.

        A line is the model's id. Where the gateway says more of a model than the usual fields (an alias, prices,
        capabilities), that follows the id as compact JSON, cut at 300 characters."""
        try:                                                     # read as it was sent: what a record holds is then
            reply = self._client.models.with_raw_response.list().text       # the gateway's alone, none of it the SDK's
        except (openai.APIConnectionError, openai.APIStatusError) as exc:
            raise self._error(exc, _LISTING) from None
        try:
            records = sorted(json.loads(reply)["data"], key=lambda record: record["id"])
            return [_model_line(record, self._api_key) for record in records]
        except (ValueError, LookupError, TypeError):             # not JSON, or JSON that is no list of models with ids
            said = _shown(reply, self._api_key)
            raise BackendError(f"{_LISTING} at {self._base_url} is no list of models: {said}") from None

    def _error(self, exc: openai.APIError, asked: str) -> Exception:
        """The SDK's error as this layer's, for a request that asked for `asked`, with the key struck out.

        A rate limit, a 5xx (the gateway answers 502 when the model behind it does not finish in time) and a
        request that got no reply (a timeout, no connection) are transient; any other status stops the run, and
        so does a 429 that says the allowance is used up, as a QuotaExceededError."""
        if not isinstance(exc, openai.APIStatusError):           # APIConnectionError, which includes APITimeoutError
            return TransientError(strike_key(f"{asked} at {self._base_url}: {exc}", self._api_key))
        status, message = exc.status_code, strike_key(f"{exc.status_code} from {asked}: {exc}", self._api_key)
        if status == 429 and _allowance_spent(str(exc)):         # not `message`: that names the model as well
            return QuotaExceededError(message)
        if status == 429 or status >= 500:
            return TransientError(message, _retry_after(exc.response.headers))
        return BackendError(message)


def _refuse_shell_headers(base_url: str) -> None:
    """Raise a ConfigError naming each variable of the shell that the SDK would send on to `base_url`.

    The address is not OpenAI's: what the shell holds for another service does not go there, and the key in use
    is the entry's. A variable that is set and empty counts too: an empty organization or project is still sent
    as a header. The message shows no value."""
    held = [name for name in _SHELL_HEADERS if name in os.environ]
    if held:
        raise ConfigError(
            f"the shell sets {', '.join(held)}, and the openai package would send what is set there to {base_url} "
            f"in request headers, beside the entry's key or in place of it: run the command with "
            f"env {' '.join(f'-u {name}' for name in held)} in front of it")


def _model_line(record: dict, key: str) -> str:
    """A listed model as one line: its id and, when its record holds more than the usual fields, the rest as
    compact JSON, with the key struck out before it is cut."""
    more = {name: value for name, value in record.items() if name not in _USUAL_FIELDS}
    if not more:
        return record["id"]
    return f"{record['id']}  {_shown(json.dumps(more, separators=(',', ':')), key)}"


def _content(part: Part, detail: str | None) -> dict:
    if isinstance(part, ImagePart):
        image = {"url": "data:image/png;base64," + base64.b64encode(part.png).decode("ascii")}
        if detail is not None:
            image["detail"] = detail
        return {"type": "image_url", "image_url": image}
    return {"type": "text", "text": part.text}


def _result(resp, model: str, key: str) -> BackendResult:
    """The reply as a BackendResult. Its text is returned with the key struck out: a refusal can echo the key,
    and the text goes on into the cache and the run outputs."""
    choices = getattr(resp, "choices", None)
    message = getattr(choices[0], "message", None) if choices else None
    if message is None:        # a 200 that holds an error, a page that is no JSON at all, or a choice with no message
        raise TransientError(f"{model}: the reply holds no answer: {_shown(resp, key)}") from None
    choice, usage = choices[0], _usage(resp.usage)
    text = _text(message.content)
    if message.refusal:
        text, finish = message.refusal, "refused"
    elif choice.finish_reason == "content_filter":
        finish = "refused"
    elif choice.finish_reason == "length" or text is None:       # no content: the budget went into thinking
        finish = "truncated"
    else:
        finish = "ok"
    return BackendResult(None if text is None else strike_key(text, key), finish, usage, resp.model)


def _text(content) -> str | None:
    """The answer text of a message: its content or, where that is a list of parts, the text parts joined. None
    when there is no text, or nothing but white space."""
    if isinstance(content, list):
        texts = [part.get("text") for part in content if isinstance(part, dict) and part.get("type") == "text"]
        content = "".join(text for text in texts if isinstance(text, str))
    return content if isinstance(content, str) and content.strip() else None


def _shown(reply, key: str) -> str:
    """The start of a reply, as an error message or a line of the listing shows it: enough to tell what came back.

    An object the SDK made of a reply is shown as its JSON. The key is struck out before the text is cut: a cut
    can halve a key."""
    if not isinstance(reply, str):
        as_json = getattr(reply, "to_json", None)
        reply = as_json(indent=None, warnings=False) if as_json else str(reply)
    return strike_key(reply, key)[:_SHOWN_CHARS]


def _usage(usage) -> Usage:
    if usage is None:
        return Usage()
    details = usage.completion_tokens_details
    reasoning = details.reasoning_tokens if details is not None else None
    visible = usage.completion_tokens                            # the API's figure includes reasoning
    if visible is not None and reasoning is not None:
        visible -= reasoning
    return Usage(input_tokens=usage.prompt_tokens, output_tokens=visible, reasoning_tokens=reasoning)


def _retry_after(headers) -> float | None:
    """The wait the server asked for, in seconds: None when absent, not a number, or outside 0 to 120 s."""
    try:
        seconds = float(headers.get("retry-after"))
    except (TypeError, ValueError):                              # absent, or an HTTP date
        return None
    return seconds if 0.0 <= seconds <= _MAX_RETRY_AFTER_S else None


_ANY_TEXT = "^[\\s\\S]*$"


def _for_guidance(schema: dict) -> dict:
    """The schema with keywords that make vLLM pick its 'guidance' backend and constrain nothing: multipleOf 1 on
    every integer, a match-anything pattern on every free string (one with no enum, const, pattern or format)."""
    def walk(node):
        if isinstance(node, dict):
            node = {k: walk(v) for k, v in node.items()}
            if node.get("type") == "integer":
                node.setdefault("multipleOf", 1)
            if node.get("type") == "string" and not {"enum", "const", "pattern", "format"} & set(node):
                node["pattern"] = _ANY_TEXT
            return node
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return walk(schema)

