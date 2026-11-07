"""Request parts, the response record, the back-end protocol and the client every model call goes through."""
from __future__ import annotations

import itertools
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, Sequence

from pydantic import BaseModel, ValidationError

from sgi.config import ModelEntry, StageName
from sgi.types import Call, Usage

if TYPE_CHECKING:                                  # cache.py imports this module: no import of it here at run time
    from sgi.vlm.cache import Cache

_ATTEMPTS = 3                                      # per request to the back end
_BACKOFF_S = (2.0, 10.0)                           # before attempt 2 and 3: three attempts leave room for two sleeps
_PACED_WINDOW_S = 60.0                             # a request limit counts the starts inside any window this long
_SPENT_WORDS = ("quota", "budget")                 # in a 429: the allowance is used up; any other 429 is a rate limit
_FENCE = "```"
_RETRY_NOTE = ("Your previous answer was rejected: {error}\n"
               "Answer again with one JSON object that matches the required schema exactly, and nothing else.")


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    png: bytes
    width: int
    height: int


Part = TextPart | ImagePart


class VLMResponse(BaseModel):
    status: Literal["ok", "parse_failed", "vlm_error"]
    parsed: dict | None = None                     # the validated wire answer as a plain dict
    text: str | None = None
    error: str | None = None
    cached: bool = False
    provider: str
    model: str
    model_version: str | None = None
    reasoning_effort: str
    point_format: str
    image_size: tuple[int, int] | None = None      # (width, height) as sent
    usage: Usage = Usage()
    latency_s: float = 0.0
    latency_total_s: float = 0.0
    validation_retries: int = 0
    transient_retries: int = 0


@dataclass(frozen=True)
class BackendResult:
    text: str | None
    finish: Literal["ok", "truncated", "refused"]
    usage: Usage
    model_version: str | None = None


class TransientError(Exception):
    """A rate limit, a 5xx or a timeout: retried by VLMClient."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class BackendError(Exception):
    """Any other provider error (bad key, rejected parameter): propagates and stops the run."""


class QuotaExceededError(BackendError):
    """The provider's allowance, a quota or a budget, is used up: no call is answered until it is renewed.

    VLMClient neither asks again nor records an answer. A run stops at once and keeps what it has done."""


def strike_key(text: str, key: str) -> str:
    """The text with every occurrence of the API key struck out: a provider's error or refusal can echo the key.

    Every back end passes what it raises or returns through here (spec 7)."""
    return text.replace(key, "<key>") if key else text


def _allowance_spent(said: str) -> bool:
    """Whether what a server said with a 429 names a quota or a budget, in any case: then the allowance is used
    up and no wait brings an answer, where any other 429 is a rate limit.

    The one rule of every back end that raises a QuotaExceededError. `said` is the server's own words, not the
    message a back end builds round them, which names the model."""
    said = said.lower()
    return any(word in said for word in _SPENT_WORDS)


class Backend(Protocol):
    provider: str

    def call(self, entry: ModelEntry, system: str | None, parts: Sequence[Part],
             json_schema: dict) -> BackendResult: ...


class Pacer:
    """Keeps the requests to one provider under its limit: at most `max_per_minute` start in any 60 seconds.

    The window slides over the start times `wait` has let through. `max_per_minute` may be set at any time; a
    lower value holds for the starts already counted."""

    def __init__(self, max_per_minute: int, sleep=time.sleep, clock=time.monotonic):
        self.max_per_minute = max_per_minute
        self._sleep, self._clock = sleep, clock
        self._starts: deque[float] = deque()       # of the last minute, oldest first

    def wait(self) -> None:
        """Return when a request may start, sleeping until then, and count it as started."""
        while True:
            now = self._clock()
            while self._starts and self._starts[0] + _PACED_WINDOW_S <= now:
                self._starts.popleft()
            if len(self._starts) < self.max_per_minute:
                self._starts.append(now)
                return
            # Full: there is room again once the limit-th most recent start is a minute old.
            self._sleep(self._starts[-self.max_per_minute] + _PACED_WINDOW_S - now)


@dataclass
class _Tally:
    latency_s: float = 0.0                         # of the last back-end call, also when it raised
    transient_retries: int = 0                     # sleeps so far


def _read_answer(result: BackendResult, schema: type[BaseModel]) -> tuple[dict | None, str | None]:
    """(parsed, None) for a valid answer, else (None, error). Only a surrounding code fence is removed."""
    if result.finish != "ok":
        return None, result.finish                 # cut off or refused: the model's own outcome, whatever the text
    text = (result.text or "").strip()
    if text.startswith(_FENCE):
        text = text.partition("\n")[2].removesuffix(_FENCE)       # the whole opening line and the closing fence
    try:
        payload = json.loads(text)
    except ValueError as exc:
        return None, f"invalid JSON: {exc}"
    try:
        return schema.model_validate(payload).model_dump(mode="json"), None
    except ValidationError as exc:
        return None, "; ".join(f"{'.'.join(str(step) for step in e['loc'])}: {e['msg']}" for e in exc.errors())


class VLMClient:
    """One model entry on one back end, with validation, retries, latency accounting, a disk cache and, when
    given a pacer, a wait for its turn before every request."""

    def __init__(self, entry: ModelEntry, backend: Backend, cache: "Cache | None" = None,
                 sleep=time.sleep, clock=time.monotonic, rng=random.random, pacer: Pacer | None = None):
        self.entry = entry
        self._backend = backend
        self._cache = cache
        self._sleep, self._clock, self._rng = sleep, clock, rng
        self._pacer = pacer

    def generate(self, parts: Sequence[Part], schema: type[BaseModel], system: str | None = None,
                 trial: int = 0) -> VLMResponse:
        from sgi.vlm.cache import cache_key        # here, not at the top: cache.py imports this module

        json_schema = schema.model_json_schema()
        provider = self._backend.provider          # not the entry's: a mock standing in for a real entry stays a mock
        key = cache_key(self.entry.model_copy(update={"provider": provider}), system, parts, json_schema, trial)
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                return hit.model_copy(update={"cached": True})

        images = [part for part in parts if isinstance(part, ImagePart)]
        tally = _Tally()
        validation_retries = 0
        started = self._clock()

        def response(status: str, result: BackendResult | None = None, parsed: dict | None = None,
                     error: str | None = None) -> VLMResponse:
            answered = {} if result is None else {"text": result.text, "usage": result.usage,
                                                  "model_version": result.model_version}
            return VLMResponse(
                status=status, parsed=parsed, error=error, provider=provider, model=self.entry.model,
                reasoning_effort=self.entry.reasoning_effort, point_format=self.entry.point_format,
                image_size=(images[-1].width, images[-1].height) if images else None,
                latency_s=tally.latency_s, latency_total_s=self._clock() - started,
                validation_retries=validation_retries, transient_retries=tally.transient_retries, **answered)

        try:
            result = self._request(system, list(parts), json_schema, tally)
            parsed, error = _read_answer(result, schema)
            if error is not None and result.finish == "ok":      # failed validation: ask once more, with the error
                validation_retries = 1
                note = TextPart(_RETRY_NOTE.format(error=error))
                result = self._request(system, [*parts, note], json_schema, tally)
                parsed, error = _read_answer(result, schema)
        except TransientError as exc:
            return response("vlm_error", error=str(exc))         # no model answer was obtained: never cached
        outcome = response("ok" if error is None else "parse_failed", result, parsed, error)
        if self._cache is not None:
            self._cache.put(key, outcome)          # ok and parse_failed are both the model's outcome for this key
        return outcome

    def _request(self, system: str | None, parts: list[Part], json_schema: dict, tally: _Tally) -> BackendResult:
        """One request with its own budget of attempts. Raises the last TransientError when all of them fail.
        Any other error of the back end, a QuotaExceededError among them, passes through from the attempt it met."""
        for attempt in itertools.count(1):
            if self._pacer is not None:
                self._pacer.wait()                 # ahead of `before`: a wait for a turn is not the provider's time
            before = self._clock()
            try:
                return self._backend.call(self.entry, system, parts, json_schema)
            except TransientError as exc:
                if attempt == _ATTEMPTS:
                    raise
                delay = exc.retry_after
                if delay is None:
                    delay = _BACKOFF_S[attempt - 1] * (0.8 + 0.4 * self._rng())
            finally:
                tally.latency_s = self._clock() - before
            self._sleep(delay)
            tally.transient_retries += 1


def to_call(stage: StageName, resp: VLMResponse) -> Call:
    """The record of one model call: the fields the two models share, with the raw text under `raw_text`."""
    shared = resp.model_dump(include=set(Call.model_fields) & set(VLMResponse.model_fields))
    return Call(stage=stage, raw_text=resp.text, **shared)
