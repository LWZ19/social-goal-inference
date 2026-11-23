import json
import random
from typing import Literal
import pytest
from pydantic import BaseModel
from sgi.config import ModelEntry
from sgi.types import Usage
from sgi.vlm.base import BackendError, BackendResult, ImagePart, QuotaExceededError, TextPart, TransientError
from sgi.vlm.base import _RETRY_NOTE, VLMClient, VLMResponse, _allowance_spent, strike_key, to_call
from sgi.vlm.cache import Cache, cache_key
from sgi.vlm.mock import MockBackend, centre_responder
from sgi.vlm.points import POINT_FORMATS

PixelPoint = POINT_FORMATS["xy_pixel"].wire_model(640, 480)

class Answer(BaseModel):
    label: str
    point: PixelPoint

ENTRY = ModelEntry(provider="mock", model="m1", point_format="xy_pixel")
PARTS = [ImagePart(b"\x89PNG example", 320, 240), TextPart("where?"), ImagePart(b"\x89PNG query", 640, 480)]
GOOD = {"label": "desk", "point": {"x": 400, "y": 300}}
OFF_IMAGE = {"label": "desk", "point": {"x": 700, "y": 300}}
PROSE = "Answer: " + json.dumps(GOOD)                                 # JSON wrapped in prose is a bad answer, not repaired
DOWN = TransientError("503 overloaded")

def test_cache_key_covers_everything_that_shapes_the_answer():
    def key(system="sys", parts=PARTS, schema=Answer.model_json_schema(), trial=0, **changed):
        return cache_key(ENTRY.model_copy(update=changed), system, parts, schema, trial)
    one_byte = PARTS[:2] + [ImagePart(b"\x89PNG querz", 640, 480)]
    split = [PARTS[0], TextPart("whe"), TextPart("re?"), PARTS[2]]
    keys = [key(), key(trial=1), key(parts=one_byte), key(parts=split), key(system=None),
            key(schema=PixelPoint.model_json_schema()), key(model="m2"), key(provider="gemini"),
            key(reasoning_effort="high"), key(temperature=0.0)]
    assert len(set(keys)) == 10 and len(keys[0]) == 64 and key(parts=tuple(PARTS)) == keys[0]
    assert key(timeout_s=5.0) == keys[0]                              # the timeout does not shape the answer

# One made-up request and the key it had before the first field was added to ModelEntry.
# A change that moves this digest orphans every answer in .cache/vlm: correct the change, never the digest.
PINNED_ENTRY = dict(provider="openai_compatible", model="made-up:1b", base_url="https://gateway.example/v1",
                    api_key_env="A_KEY", reasoning_effort="none")
PINNED_PARTS = [TextPart("where is it?"), ImagePart(b"\x89PNG pinned bytes", 64, 48)]
PINNED_SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"], "additionalProperties": False}
PINNED_KEY = "bdd93bf5c3ec6edb68c04b366993cc1561b60f1be9c10fb4cdd4fee07dfd6c58"

def test_the_cache_key_of_a_fixed_request_is_the_one_it_always_was():
    entry = ModelEntry(**PINNED_ENTRY)
    assert cache_key(entry, "You answer briefly.", PINNED_PARTS, PINNED_SCHEMA, 0) == PINNED_KEY

def test_a_request_limit_and_prices_leave_the_cache_key_as_it_was():
    paced = ModelEntry(**PINNED_ENTRY, max_requests_per_minute=80, price_input_per_mtok=0.408922, price_output_per_mtok=2.320054)
    assert cache_key(paced, "You answer briefly.", PINNED_PARTS, PINNED_SCHEMA, 0) == PINNED_KEY     # none shapes an answer
    assert cache_key(paced.model_copy(update={"model": "made-up:2b"}), "You answer briefly.", PINNED_PARTS, PINNED_SCHEMA, 0) != PINNED_KEY

def test_mock_backend_records_calls_and_maps_responder_results():
    script = [{"a": 1}, "raw text", BackendError("bad key"), BackendResult("cut", "truncated", Usage())]
    backend = MockBackend(lambda parts, json_schema: script.pop(0))
    first = backend.call(ENTRY, "sys", PARTS, {"type": "object"})
    assert (json.loads(first.text), first.finish, first.usage, backend.provider) == ({"a": 1}, "ok", Usage(), "mock")
    assert backend.call(ENTRY, None, PARTS, {}).text == "raw text"
    with pytest.raises(BackendError):
        backend.call(ENTRY, None, PARTS, {})
    assert backend.call(ENTRY, None, PARTS, {}) == BackendResult("cut", "truncated", Usage())      # passed through unchanged
    assert len(backend.calls) == 4 and backend.calls[0] == ("sys", PARTS, {"type": "object"})

def test_mock_backend_passes_a_backend_result_through_unchanged():
    cut_off = BackendResult(text='{"label": "de', finish="truncated", usage=Usage(output_tokens=7), model_version="m1-001")
    assert MockBackend(lambda parts, json_schema: cut_off).call(ENTRY, None, PARTS, {}) is cut_off

def test_centre_responder_fills_a_nested_schema():
    class Step(BaseModel): type: Literal["navigate", "act"]; target_point: PixelPoint | None
    class Nested(BaseModel): reasoning: str; people_matter: bool; goal_point: PixelPoint; subtasks: list[Step]
    out = centre_responder(PARTS, Nested.model_json_schema())
    assert Nested.model_validate(out).model_dump() == out == {
        "reasoning": "mock", "people_matter": True, "goal_point": {"x": 320, "y": 240},
        "subtasks": [{"type": "navigate", "target_point": {"x": 320, "y": 240}}]}
    normalised = POINT_FORMATS["xy_1000"].wire_model(640, 480).model_json_schema()
    assert centre_responder(PARTS, normalised) == {"x": 500, "y": 500}

def harness(items, costs=(), cache=None, rng=lambda: 0.5, entry=ENTRY):
    now, sleeps, items, costs = [50.0], [], list(items), list(costs)
    def responder(parts, json_schema):
        now[0] += costs.pop(0) if costs else 0.0                      # how long this backend call takes
        return items.pop(0)
    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
    backend = MockBackend(responder)
    return VLMClient(entry, backend, cache, sleep=sleep, clock=lambda: now[0], rng=rng), backend, sleeps

def test_fenced_json_is_parsed_and_the_response_is_labelled():
    fence = "`" * 3
    for raw in (f"{fence}json\n{json.dumps(GOOD)}\n{fence}", f"{fence}\n{json.dumps(GOOD)}\n{fence}\n"):
        resp = harness([raw])[0].generate(PARTS, Answer)
        assert (resp.status, resp.parsed, resp.text, resp.error, resp.cached) == ("ok", GOOD, raw, None, False)
    assert (resp.model, resp.reasoning_effort, resp.point_format) == ("m1", "low", "xy_pixel")
    assert resp.image_size == (640, 480)                              # the last image part, not the first

@pytest.mark.filterwarnings("error::RuntimeWarning")                  # neither a plain miss nor a hit warns
def test_one_validation_retry_then_the_answer_is_replayed_from_the_cache(tmp_path):
    stand_in = ENTRY.model_copy(update={"provider": "openai"})        # the mock answers for a real entry
    client, backend, _ = harness([PROSE, GOOD], costs=[0.5, 1.5], cache=Cache(tmp_path), entry=stand_in)
    first = client.generate(PARTS, Answer, system="sys", trial=2)
    second = client.generate(PARTS, Answer, system="sys", trial=2)    # the same request: no third backend call
    assert (first.status, first.parsed, first.validation_retries, first.cached) == ("ok", GOOD, 1, False)
    assert (first.provider, first.latency_s, first.latency_total_s) == ("mock", 1.5, 2.0)
    assert second.model_dump() == {**first.model_dump(), "cached": True} and len(backend.calls) == 2
    retry = backend.calls[1][1]
    assert retry[:-1] == PARTS == backend.calls[0][1] and len(PARTS) == 3
    assert isinstance(retry[-1], TextPart) and "invalid JSON" in retry[-1].text
    key = cache_key(ENTRY, "sys", PARTS, Answer.model_json_schema(), 2)      # keyed by the back end that answered
    assert [p.name for p in tmp_path.iterdir()] == [key + ".json"]

def test_two_bad_answers_give_parse_failed_which_is_cached(tmp_path):
    client, backend, _ = harness([OFF_IMAGE, OFF_IMAGE], cache=Cache(tmp_path))
    first, again = client.generate(PARTS, Answer), client.generate(PARTS, Answer)
    assert (first.status, first.parsed, first.validation_retries, len(backend.calls)) == ("parse_failed", None, 1, 2)
    assert "point.x" in first.error and backend.calls[1][1][-1] == TextPart(_RETRY_NOTE.format(error=first.error))
    assert (again.status, again.cached) == ("parse_failed", True)

@pytest.mark.parametrize("finish", ["truncated", "refused"])
def test_cut_off_and_refused_answers_are_parse_failed_without_a_retry(finish, tmp_path):
    class Fixed:                                                      # any object with provider and call is a back end
        provider, n = "mock", 0
        def call(self, entry, system, parts, json_schema):
            self.n += 1
            return BackendResult(text=json.dumps(GOOD), finish=finish, usage=Usage(output_tokens=7))
    backend = Fixed()
    resp = VLMClient(ENTRY, backend, Cache(tmp_path)).generate(PARTS, Answer)    # valid JSON, but finish wins
    assert (resp.status, resp.error, resp.parsed, backend.n) == ("parse_failed", finish, None, 1)
    assert (resp.text, resp.usage.output_tokens) == (json.dumps(GOOD), 7)
    assert len(list(tmp_path.iterdir())) == 1                         # the model's own outcome, so it is cached

def test_three_transient_errors_give_vlm_error_which_is_not_cached(tmp_path):
    client, backend, sleeps = harness([DOWN, DOWN, DOWN, GOOD], cache=Cache(tmp_path), rng=random.Random(0).random)
    first, again = client.generate(PARTS, Answer), client.generate(PARTS, Answer)
    assert (first.status, first.error, first.parsed, first.text) == ("vlm_error", "503 overloaded", None, None)
    assert first.transient_retries == len(sleeps) == 2 and 1.6 <= sleeps[0] <= 2.4 and 8.0 <= sleeps[1] <= 12.0
    assert sleeps != [2.0, 10.0]                                      # the jitter is really applied
    assert (again.status, again.cached, len(backend.calls)) == ("ok", False, 4)

def test_latencies_come_from_the_clock_and_each_request_has_its_own_retry_budget():
    script = [TransientError("429", retry_after=7.0), PROSE, DOWN, DOWN, OFF_IMAGE]
    client, _, sleeps = harness(script, costs=[0.25, 0.5, 0.25, 0.25, 1.5])
    resp = client.generate(PARTS, Answer)
    assert (resp.status, resp.validation_retries, resp.transient_retries) == ("parse_failed", 1, 3)
    assert sleeps == [7.0, 2.0, 10.0]                 # retry_after as given; rng() = 0.5 makes the jitter factor exactly 1
    assert (resp.latency_s, resp.latency_total_s) == (1.5, 21.75)     # 21.75 = the five costs plus the three sleeps
    call = to_call("goal", resp)
    assert (call.stage, call.raw_text, json.loads(call.raw_text)) == ("goal", resp.text, OFF_IMAGE)
    assert call.status == resp.status == "parse_failed"               # copied from the response, not left at the default "ok"
    assert "point.x" in call.error and call.model_dump(exclude={"stage", "raw_text"}).items() <= resp.model_dump().items()

def test_a_backend_error_propagates_and_nothing_is_stored(tmp_path):
    with pytest.raises(BackendError):
        harness([BackendError("bad key")], cache=Cache(tmp_path))[0].generate(PARTS, Answer)
    assert list(tmp_path.iterdir()) == []

def test_a_spent_allowance_is_let_through_after_one_call(tmp_path):
    spent = QuotaExceededError("429 from m1: Budget limit reached")
    client, backend, sleeps = harness([spent, GOOD], cache=Cache(tmp_path))
    with pytest.raises(QuotaExceededError) as err:                    # neither asked again nor turned into a vlm_error
        client.generate(PARTS, Answer)                                # answer: waiting does not bring the allowance back
    assert err.value is spent and (len(backend.calls), sleeps, list(tmp_path.iterdir())) == (1, [], [])
    assert isinstance(spent, BackendError) and not isinstance(spent, TransientError)      # it stops a run, as a bad key does

@pytest.mark.parametrize("stored", ["{", '{"status": "ok"}'])        # cut short; valid JSON that is not a VLMResponse
def test_unreadable_cache_entry_is_a_miss(stored, tmp_path):
    path = tmp_path / (cache_key(ENTRY, "sys", PARTS, Answer.model_json_schema(), 0) + ".json")
    path.write_text(stored)                                           # the entry of the request about to be sent
    client, backend, _ = harness([GOOD], cache=Cache(tmp_path))
    with pytest.warns(RuntimeWarning) as caught:
        resp = client.generate(PARTS, Answer, system="sys")
    assert len(caught) == 1 and path.name in str(caught[0].message)   # one warning, and it names the file
    assert (resp.status, resp.parsed, resp.cached, len(backend.calls)) == ("ok", GOOD, False, 1)
    assert VLMResponse.model_validate_json(path.read_text()) == resp  # the fresh answer has replaced the entry

@pytest.mark.parametrize("stored, error", [(b"{", "ValidationError"), (b'{"text": "\xe2\x82', "UnicodeDecodeError")])
def test_the_cache_warning_is_one_line_with_the_full_path_and_the_error_class(stored, error, tmp_path):
    path = tmp_path / "k.json"
    path.write_bytes(stored)                                          # the second is cut inside a three-byte character
    with pytest.warns(RuntimeWarning) as caught:
        assert Cache(tmp_path).get("k") is None
    assert [str(w.message) for w in caught] == [f"cache entry {path} cannot be read back ({error}); it counts as a miss"]
    assert path.read_bytes() == stored                                # get neither deletes nor repairs the entry

def test_strike_key_strikes_every_occurrence_and_leaves_the_text_alone_without_a_key():
    key = "k3y-9f2c7a41d8e6b05-never-shown"
    assert strike_key(f"Incorrect API key provided: {key}. Sent {key} to /v1?key={key}", key) == (
        "Incorrect API key provided: <key>. Sent <key> to /v1?key=<key>")
    assert strike_key("503: no answer", key) == "503: no answer"
    assert strike_key("503: no answer", "") == "503: no answer"       # str.replace("", ...) would fill it

@pytest.mark.parametrize("said, spent", [                             # what a server said with a 429
    ("quota exceeded", True), ("Budget limit reached", True),         # either word, in any case,
    ("Error code: 429 - {'error': {'code': 'insufficient_quota'}}", True),   # also inside a longer one
    ("rate limit exceeded", False), ("slow down", False), ("", False)])
def test_a_429_that_names_a_quota_or_a_budget_says_the_allowance_is_used_up(said, spent):
    assert _allowance_spent(said) is spent

def test_the_retry_note_has_one_placeholder_and_takes_an_error_with_braces_as_it_stands():
    assert _RETRY_NOTE.count("{") == _RETRY_NOTE.count("}") == 1 and "{error}" in _RETRY_NOTE
    assert '{"x": 1}' in _RETRY_NOTE.format(error='point: {"x": 1}')
