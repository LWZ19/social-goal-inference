import base64
import copy
import json
import logging
import os
import subprocess
import sys
import traceback
import anthropic
import httpx2
import pytest
from pydantic import BaseModel
import sgi.vlm.anthropic
from sgi.config import REPO_ROOT, ConfigError, ModelEntry, load_config
from sgi.schemas import goal_schema
from sgi.types import Usage
from sgi.vlm import factory
from sgi.vlm.anthropic import AnthropicBackend, _classified, _sendable
from sgi.vlm.base import (BackendError, BackendResult, ImagePart, QuotaExceededError, TextPart, TransientError, VLMClient,
                          to_call)
from sgi.vlm.cache import Cache
from sgi.vlm.env import MissingKeyError
from tests.test_key_order import SENT, objects

HOST, KEY = "https://api.anthropic.com", "sk-ant-test"                 # a made-up key: nothing is sent anywhere
SECRET = "k3y-9f2c7a41d8e6b05-never-shown"                             # unlike any path or source line of a traceback
PNG = b"\x89PNG\r\n\x1a\n"                                             # its base64 text is iVBORw0KGgo=
PARTS = [TextPart("example"), ImagePart(PNG, 4, 2), TextPart("where?"), ImagePart(PNG, 4, 2)]
SCHEMA = {"title": "Label", "type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"],
          "additionalProperties": False}
FORMAT = {"type": "json_schema", "schema": SCHEMA}
ENTRY = ModelEntry(provider="anthropic", model="claude-opus-5-5", reasoning_effort="medium")
VERSION = "the-model-as-the-server-names-it"                           # what the server echoes
SHELL = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_PROFILE")
# What a current model says to `thinking: {type: disabled}` when it cannot switch thinking off, with a 400.
NO_SWITCH = ('"thinking.type.disabled" is not supported for this model. Use "thinking.type.adaptive" and '
             '"output_config.effort" to control thinking behavior.')

class Label(BaseModel):
    label: str

@pytest.fixture(autouse=True)
def a_shell_without_settings_for_the_sdk(monkeypatch):                 # the tests run the same whatever the shell holds
    for name in SHELL:
        monkeypatch.delenv(name, raising=False)

def build(*request):                                                   # building a client touches no network
    return AnthropicBackend(KEY).build_request(*request)

def test_request_sends_only_what_the_entry_sets():
    text = lambda t: {"type": "text", "text": t}
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}
    request = build(ENTRY, "sys", PARTS, SCHEMA)
    assert request == {
        "model": "claude-opus-5-5", "max_tokens": 16000, "system": "sys", "timeout": 180.0,
        "messages": [{"role": "user", "content": [text("example"), image, text("where?"), image]}],     # one turn, the user's
        "output_config": {"format": FORMAT, "effort": "medium"}}
    # No thinking setting beside an effort level, no sampling parameter, no forced tool, no fallback to another model.
    assert not {"thinking", "temperature", "top_p", "top_k", "tools", "tool_choice", "fallbacks"} & set(request)
    tuned = ENTRY.model_copy(update=dict(reasoning_effort="xhigh", max_output_tokens=4096, timeout_s=30.0))
    assert build(tuned, None, PARTS[2:], SCHEMA) == {
        "model": "claude-opus-5-5", "max_tokens": 4096, "timeout": 30.0,                              # no system text
        "messages": [{"role": "user", "content": [text("where?"), image]}],
        "output_config": {"format": FORMAT, "effort": "xhigh"}}
    assert base64.b64decode(request["messages"][0]["content"][1]["source"]["data"]) == PNG

@pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max"])
def test_an_effort_level_is_sent_as_written_with_no_thinking_setting(level):
    request = build(ENTRY.model_copy(update={"reasoning_effort": level}), None, PARTS, SCHEMA)
    assert request["output_config"] == {"format": FORMAT, "effort": level} and "thinking" not in request

def test_the_level_none_switches_thinking_off_and_sends_no_effort_level():
    request = build(ENTRY.model_copy(update={"reasoning_effort": "none"}), "sys", PARTS, SCHEMA)
    assert request["thinking"] == {"type": "disabled"} and request["output_config"] == {"format": FORMAT}
    assert ModelEntry(provider="anthropic", model="m").reasoning_effort == "low"      # an entry that names no level
    assert build(ModelEntry(provider="anthropic", model="m"), None, PARTS, SCHEMA)["output_config"]["effort"] == "low"

def test_the_schema_of_a_stage_is_sent_without_the_bounds_of_its_points_and_is_otherwise_as_it_was():
    schema = goal_schema("cot", "xy_1000", (640, 480), "vlm_look_at").model_json_schema()
    point = {"description": "0 = left edge, 1000 = right edge", "title": "X", "type": "integer"}
    assert schema["$defs"]["Point1000"]["properties"]["x"] == {**point, "maximum": 1000, "minimum": 0}
    sent = build(ENTRY, "sys", PARTS, schema)["output_config"]["format"]["schema"]
    assert sent["$defs"]["Point1000"]["properties"]["x"] == point       # the description still says the range
    for name in ("x", "y"):                                             # with the four bounds put back, nothing differs
        sent["$defs"]["Point1000"]["properties"][name].update(minimum=0, maximum=1000)
    assert sent == schema and schema["$defs"]["Point1000"]["properties"]["x"]["maximum"] == 1000      # a copy was sent

def flat(node, path=()):                            # (path, value) of everything a schema holds that is no dict or list
    if isinstance(node, dict):
        for key, sub in node.items():
            yield from flat(sub, (*path, key))
    elif isinstance(node, list):
        for index, sub in enumerate(node):
            yield from flat(sub, (*path, index))
    else:
        yield path, node

@pytest.mark.parametrize("schema", [schema for _, schema in SENT], ids=[label for label, _ in SENT])
def test_every_schema_a_run_can_send_loses_minimum_and_maximum_alone_and_keeps_the_order_of_its_keys(schema):
    sent = _sendable(schema)
    kept = [(path, value) for path, value in flat(schema) if path[-1] not in ("minimum", "maximum")]
    assert list(flat(sent)) == kept                                     # every other keyword, value and description
    assert not {path[-1] for path, _ in flat(sent)} & sgi.vlm.anthropic._UNSENT
    assert list(objects(sent)) == list(objects(schema))                 # each object: its keys in their order, all required
    assert json.loads(json.dumps(sent)) == sent and list(json.loads(json.dumps(sent))["properties"]) == list(schema["properties"])

def test_only_keywords_are_removed_at_every_depth_and_the_schema_given_is_left_as_it_was():
    schema = {"type": "object", "additionalProperties": False, "required": ["minimum", "tags", "frame", "point"],
              "properties": {"minimum": {"type": "number", "minimum": 0, "exclusiveMaximum": 9, "multipleOf": 0.5},
                             "tags": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 9,
                                                                 "enum": ["a", "b"]}},
                             "frame": {"anyOf": [{"const": 3, "type": "integer", "maximum": 3}, {"type": "null"}]},
                             "point": {"$ref": "#/$defs/P"}},
              "$defs": {"P": {"type": "object", "properties": {"maximum": {"type": "integer", "exclusiveMinimum": 0}},
                              "required": ["maximum"], "additionalProperties": False}}}
    before = copy.deepcopy(schema)
    assert _sendable(schema) == {                                       # a key of an answer may be named as a keyword is
        "type": "object", "additionalProperties": False, "required": ["minimum", "tags", "frame", "point"],
        "properties": {"minimum": {"type": "number"},
                       "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}},
                       "frame": {"anyOf": [{"const": 3, "type": "integer"}, {"type": "null"}]},      # const is taken
                       "point": {"$ref": "#/$defs/P"}},
        "$defs": {"P": {"type": "object", "properties": {"maximum": {"type": "integer"}}, "required": ["maximum"],
                        "additionalProperties": False}}}
    assert schema == before

def wired(status=200, body=None, headers=None, error=None, says="no answer", key=KEY):
    # a back end on a real SDK client with default settings, whose HTTP layer is this handler: no network
    sent = []
    def handler(request):
        sent.append(request)
        if error:
            raise getattr(httpx2, error)(says, request=request)
        reply = {"text": body} if isinstance(body, str) else {"json": body}
        return httpx2.Response(status, headers=headers or {}, **reply)
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    return AnthropicBackend(key, client=anthropic.Anthropic(api_key=key, http_client=http)), sent

USAGE = {"input_tokens": 100, "output_tokens": 50, "output_tokens_details": {"thinking_tokens": 30}}
GOOD, CUT, text = '{"label": "desk"}', '{"lab', lambda t: {"type": "text", "text": t}
THOUGHT = {"type": "thinking", "thinking": "", "signature": "c2lnbmVk"}       # as sent by default: its text left out

def message(*content, stop="end_turn", usage=USAGE, **more):
    return {"id": "msg_01", "type": "message", "role": "assistant", "model": VERSION, "content": list(content),
            "stop_reason": stop, "stop_sequence": None, "usage": usage, **more}

def refused(category, explanation, *content):                         # a category of False: a reply without stop details
    details = None if category is False else {"type": "refusal", "category": category, "explanation": explanation}
    return message(*content, stop="refusal", stop_details=details)

def test_a_call_posts_the_request_to_the_messages_endpoint_with_the_key_in_its_own_header():
    backend, sent = wired(body=message(THOUGHT, text(GOOD)))
    entry = ENTRY.model_copy(update={"timeout_s": 30.0})
    assert backend.call(entry, "sys", PARTS, SCHEMA).text == GOOD
    (request,) = sent
    assert (request.method, str(request.url)) == ("POST", HOST + "/v1/messages")           # not the beta endpoint
    assert request.headers["x-api-key"] == KEY and request.extensions["timeout"]["read"] == 30.0
    assert "authorization" not in request.headers and "anthropic-beta" not in request.headers
    built = backend.build_request(entry, "sys", PARTS, SCHEMA)          # the SDK's call takes each keyword, and the body
    assert json.loads(request.content) == {name: value for name, value in built.items() if name != "timeout"}     # is those
    assert "stream" not in json.loads(request.content)

def test_the_back_end_builds_its_client_from_the_key_alone_and_sends_it_to_the_providers_own_host(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example")          # set for another tool, or by someone else
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-of-the-shell")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token-of-the-shell")
    monkeypatch.setenv("ANTHROPIC_PROFILE", "a-profile-there-is-not")            # read, it would be an error
    sent = []
    def handle_request(self, request):                                # in place of the network, under the client the back end builds
        sent.append(request)
        return httpx2.Response(401, json=error("authentication_error", "invalid x-api-key"), request=request)
    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", handle_request)
    backend = AnthropicBackend(KEY)
    assert (backend.provider, backend._client.api_key, backend._client.max_retries) == ("anthropic", KEY, 0)
    with pytest.raises(BackendError):
        backend.call(ENTRY, None, PARTS, SCHEMA)
    with pytest.raises(BackendError):
        backend.list_models()                                         # the listing carries the key too
    assert [str(request.url) for request in sent] == [HOST + "/v1/messages", HOST + "/v1/models"]
    for request in sent:                                              # the key of .env, and no credential of the shell
        assert request.headers["x-api-key"] == KEY and "authorization" not in request.headers

@pytest.mark.parametrize("body, said, finish", [
    (message(THOUGHT, text(GOOD)), GOOD, "ok"),                       # thinking is no part of the answer
    (message(text(GOOD)), GOOD, "ok"),
    (message(THOUGHT, text('{"label": '), text('"desk"}')), GOOD, "ok"),            # the text blocks, joined
    (message(THOUGHT), None, "ok"),                                   # a turn ended without text: the client finds no JSON
    (message(THOUGHT, text(CUT), stop="max_tokens"), CUT, "truncated"),
    (message(THOUGHT, stop="max_tokens"), None, "truncated"),         # the whole budget went into thinking
    (refused("cyber", "It could enable cyber harm."), "refusal, category cyber: It could enable cyber harm.", "refused"),
    (refused("bio", None), "refusal, category bio", "refused"),
    (refused(None, None), "refusal, no category given", "refused"),
    (refused(False, None), "refusal, no category given", "refused"),                # no stop details at all
    (refused(None, None, text("I can't help with that.")), "refusal, no category given\nI can't help with that.", "refused"),
    (refused("general_harms", "Why.", THOUGHT, text(CUT)), "refusal, category general_harms: Why.\n" + CUT, "refused")])
def test_response_maps_to_backend_result(body, said, finish):
    backend, sent = wired(body=body)
    usage = Usage(input_tokens=100, output_tokens=20, reasoning_tokens=30)           # the API reports 50 = 20 + 30
    assert backend.call(ENTRY, "sys", PARTS, SCHEMA) == BackendResult(said, finish, usage, VERSION) and len(sent) == 1

@pytest.mark.parametrize("reported, usage", [                         # null when the provider says nothing
    ({"input_tokens": 100, "output_tokens": 50}, Usage(input_tokens=100, output_tokens=50)),      # nothing is subtracted
    ({"input_tokens": 100, "output_tokens": 50, "output_tokens_details": None}, Usage(input_tokens=100, output_tokens=50)),
    ({"input_tokens": 100, "output_tokens": 50, "output_tokens_details": {"thinking_tokens": 0}},
     Usage(input_tokens=100, output_tokens=50, reasoning_tokens=0)),
    ({"input_tokens": 100, "output_tokens": 50, "output_tokens_details": {"thinking_tokens": 50}, "service_tier": "standard",
      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}, Usage(input_tokens=100, output_tokens=0, reasoning_tokens=50))])
def test_thinking_tokens_are_taken_out_of_the_output_tokens_and_are_none_when_the_api_does_not_report_them(reported, usage):
    backend, _ = wired(body=message(THOUGHT, text(GOOD), usage=reported))
    result = backend.call(ENTRY, None, PARTS, SCHEMA)
    assert result == BackendResult(GOOD, "ok", usage, VERSION)
    billed = result.usage.output_tokens + (result.usage.reasoning_tokens or 0)       # what a run prices as output
    assert billed == reported["output_tokens"]

@pytest.mark.parametrize("stop", ["pause_turn", "tool_use", "stop_sequence", None])
def test_a_reply_that_ends_for_a_reason_this_request_cannot_have_is_transient_and_names_the_reason(stop):
    backend, sent = wired(body=message(THOUGHT, text(GOOD), stop=stop))
    with pytest.raises(TransientError) as err:                        # no answer is recorded for it, and none cached
        backend.call(ENTRY, None, PARTS, SCHEMA)
    assert len(sent) == 1 and repr(stop) in str(err.value) and str(err.value).startswith("anthropic: ")

def error(kind, said):                                                # an error reply as the API sends it
    return {"type": "error", "error": {"type": kind, "message": said}, "request_id": "req_01"}

LIMIT = "This request would exceed your organization's rate limit of 50 requests per minute."
ERRORS = [                                                            # reply, raised, retry_after, what the server said
    (dict(status=429, body=error("rate_limit_error", LIMIT), headers={"retry-after": "7"}), TransientError, 7.0, "rate limit"),
    (dict(status=429, body=error("rate_limit_error", LIMIT)), TransientError, None, "rate limit"),
    (dict(status=429, body=error("rate_limit_error", LIMIT), headers={"retry-after": "3600"}), TransientError, None, "rate limit"),
    (dict(status=529, body=error("overloaded_error", "Overloaded"), headers={"retry-after": "7"}), TransientError, 7.0, "Overloaded"),
    (dict(status=529, body=error("overloaded_error", "Overloaded")), TransientError, None, "Overloaded"),
    (dict(status=500, body=error("api_error", "Internal server error")), TransientError, None, "Internal server error"),
    (dict(status=503, body="upstream connect error", headers={"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}),
     TransientError, None, "upstream connect error"),
    (dict(status=504, body=error("timeout_error", "took too long")), TransientError, None, "took too long"),
    (dict(error="ReadTimeout"), TransientError, None, "timed out"),
    (dict(error="ConnectError"), TransientError, None, "Connection error"),
    (dict(status=400, body=error("invalid_request_error", "max_tokens: must be a positive integer")), BackendError, None, "max_tokens"),
    (dict(status=401, body=error("authentication_error", "invalid x-api-key")), BackendError, None, "invalid x-api-key"),
    (dict(status=403, body=error("permission_error", "not allowed for this key")), BackendError, None, "not allowed"),
    (dict(status=404, body=error("not_found_error", "model: claude-opus-9")), BackendError, None, "model: claude-opus-9"),
    (dict(status=413, body=error("request_too_large", "Request exceeds the size limit")), BackendError, None, "size limit"),
    (dict(status=402, body=error("billing_error", "A billing problem."), headers={"retry-after": "7"}), QuotaExceededError, None, "billing"),
    (dict(status=402, body="Payment Required"), QuotaExceededError, None, "Payment Required"),
    (dict(status=429, body=error("rate_limit_error", "Your monthly budget is used up."), headers={"retry-after": "7"}),
     QuotaExceededError, None, "monthly budget")]

@pytest.mark.parametrize("reply, raised, retry_after, said", ERRORS)
def test_errors_are_classified_after_a_single_attempt(reply, raised, retry_after, said, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: pytest.fail("the SDK retried"))
    backend, sent = wired(**reply)
    with pytest.raises((TransientError, BackendError)) as err:
        backend.call(ENTRY, None, PARTS, SCHEMA)
    assert len(sent) == 1                                             # the SDK's own retries are off
    assert type(err.value) is raised and getattr(err.value, "retry_after", None) == retry_after
    assert said in str(err.value)                                     # what the server said reaches the user
    if "status" in reply:
        assert str(err.value).startswith(f"anthropic {reply['status']}: ")       # an HTTP status leads the message
    else:
        assert str(err.value).startswith("anthropic: ")

def test_a_model_that_cannot_switch_thinking_off_refuses_the_level_none_and_what_it_says_stops_the_run():
    backend, sent = wired(status=400, body=error("invalid_request_error", NO_SWITCH))
    entry = ENTRY.model_copy(update={"reasoning_effort": "none"})
    client = VLMClient(entry, backend, sleep=lambda seconds: pytest.fail("the client waited to ask again"))
    with pytest.raises(BackendError) as err:                          # no answer is recorded, and nothing is asked again
        client.generate(PARTS, Label)
    assert type(err.value) is BackendError and str(err.value).startswith("anthropic 400: ") and NO_SWITCH in str(err.value)
    assert len(sent) == 1 and json.loads(sent[0].content)["thinking"] == {"type": "disabled"}

def raised_by_the_sdk(cls, status, said="said the server", headers=None):
    """An error of the SDK's own class, as the SDK builds one from a reply with that status."""
    request = httpx2.Request("POST", HOST + "/v1/messages", headers={"x-api-key": SECRET})
    body = error("some_error", said)
    response = httpx2.Response(status, json=body, headers=headers or {}, request=request)
    return cls(f"Error code: {status} - {body}", response=response, body=body)

CLASSES = [                                                           # the SDK's class, its status, what this layer raises
    (anthropic.BadRequestError, 400, BackendError), (anthropic.AuthenticationError, 401, BackendError),
    (anthropic.PermissionDeniedError, 403, BackendError), (anthropic.NotFoundError, 404, BackendError),
    (anthropic.ConflictError, 409, BackendError), (anthropic.RequestTooLargeError, 413, BackendError),
    (anthropic.UnprocessableEntityError, 422, BackendError), (anthropic.APIStatusError, 418, BackendError),
    (anthropic.APIStatusError, 402, QuotaExceededError),
    (anthropic.RateLimitError, 429, TransientError), (anthropic.InternalServerError, 500, TransientError),
    (anthropic.InternalServerError, 502, TransientError), (anthropic.ServiceUnavailableError, 503, TransientError),
    (anthropic.DeadlineExceededError, 504, TransientError), (anthropic.OverloadedError, 529, TransientError),
    (anthropic.APIStatusError, 599, TransientError)]

@pytest.mark.parametrize("cls, status, raised", CLASSES, ids=[f"{row[0].__name__}_{row[1]}" for row in CLASSES])
def test_each_error_class_of_the_sdk_becomes_one_of_this_layers_with_the_wait_the_server_asked_for(cls, status, raised):
    with pytest.raises((TransientError, BackendError)) as err:
        with _classified(SECRET):
            raise raised_by_the_sdk(cls, status, f"said with {SECRET}", {"retry-after": "7"})
    assert type(err.value) is raised and str(err.value).startswith(f"anthropic {status}: ")
    assert getattr(err.value, "retry_after", None) == (7.0 if raised is TransientError else None)
    assert "said with <key>" in str(err.value) and SECRET not in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__     # raised "from None": nothing is chained
    assert SECRET not in "".join(traceback.format_exception(err.value))       # the SDK's error holds the request, key and all

def test_a_request_without_a_reply_is_transient_whichever_of_the_two_classes_the_sdk_raises():
    request = httpx2.Request("POST", HOST + "/v1/messages", headers={"x-api-key": SECRET})
    for unanswered in (anthropic.APITimeoutError(request=request), anthropic.APIConnectionError(request=request)):
        with pytest.raises(TransientError) as err:
            with _classified(SECRET):
                raise unanswered
        assert err.value.retry_after is None and str(err.value).startswith("anthropic: ")
        assert SECRET not in "".join(traceback.format_exception(err.value))
    with pytest.raises(ZeroDivisionError):                            # what is no error of the SDK passes as it is
        with _classified(SECRET):
            1 / 0

@pytest.mark.parametrize("said, raised", [("allowance used up", QuotaExceededError), ("quota exceeded", TransientError)])
def test_a_spent_allowance_in_a_429_is_read_by_the_one_rule_the_back_ends_share(said, raised, monkeypatch):
    monkeypatch.setattr("sgi.vlm.base._SPENT_WORDS", ("allowance",))  # the rule is kept once: changed there, it is
    backend, _ = wired(status=429, body=error("rate_limit_error", said))        # changed for this back end too
    with pytest.raises((TransientError, BackendError)) as err:
        backend.call(ENTRY, None, PARTS, SCHEMA)
    assert type(err.value) is raised

def test_a_rate_limit_on_a_model_named_with_one_of_the_words_is_still_transient():
    backend, _ = wired(status=429, body=error("rate_limit_error", LIMIT))       # the rule is given what the server said
    with pytest.raises((TransientError, BackendError)) as err:
        backend.call(ENTRY.model_copy(update={"model": "claude-budget-9"}), None, PARTS, SCHEMA)
    assert type(err.value) is TransientError

def test_through_the_client_a_spent_balance_is_asked_once_and_let_through():
    backend, sent = wired(status=402, body=error("billing_error", "A billing problem."))
    client = VLMClient(ENTRY, backend, sleep=lambda seconds: pytest.fail("the client waited to ask again"))
    with pytest.raises(QuotaExceededError, match="billing"):
        client.generate(PARTS, Label)                                 # no vlm_error answer either: the run has to stop
    assert len(sent) == 1

def test_through_the_client_an_overload_is_asked_three_times_and_an_answer_is_validated_and_recorded():
    backend, sent = wired(status=529, body=error("overloaded_error", "Overloaded"))
    response = VLMClient(ENTRY, backend, sleep=lambda seconds: None).generate(PARTS, Label)
    assert (response.status, response.provider, response.transient_retries, len(sent)) == ("vlm_error", "anthropic", 2, 3)
    assert response.error.startswith("anthropic 529: ") and "Overloaded" in response.error
    backend, _ = wired(body=message(THOUGHT, text(GOOD)))
    response = VLMClient(ENTRY, backend).generate(PARTS, Label)
    assert (response.status, response.parsed, response.model_version) == ("ok", {"label": "desk"}, VERSION)
    assert (response.provider, response.model, response.reasoning_effort, response.image_size) == (
        "anthropic", "claude-opus-5-5", "medium", (4, 2))
    assert response.usage == Usage(input_tokens=100, output_tokens=20, reasoning_tokens=30)
    backend, sent = wired(body=message(THOUGHT))                       # a turn that ended without text is asked once more,
    response = VLMClient(ENTRY, backend).generate(PARTS, Label)        # with the error, as any answer that is no JSON
    assert (response.status, response.validation_retries, len(sent)) == ("parse_failed", 1, 2)
    assert "invalid JSON" in response.error and "invalid JSON" in json.loads(sent[1].content)["messages"][0]["content"][-1]["text"]

ECHO = f"Incorrect API key provided: {SECRET}. Sent {SECRET} to /v1?key={SECRET}"
ECHOES = [                                                            # where the key comes back, raised, shown struck out
    ("a 400", dict(status=400, body=error("invalid_request_error", ECHO)), BackendError, True),
    ("a 401", dict(status=401, body=error("authentication_error", ECHO)), BackendError, True),
    ("a 402", dict(status=402, body=error("billing_error", ECHO)), QuotaExceededError, True),
    ("a 403", dict(status=403, body=error("permission_error", ECHO)), BackendError, True),
    ("a 404", dict(status=404, body=error("not_found_error", ECHO)), BackendError, True),
    ("a 429", dict(status=429, body=error("rate_limit_error", ECHO), headers={"retry-after": "7"}), TransientError, True),
    ("a 500", dict(status=500, body=error("api_error", ECHO)), TransientError, True),
    ("a 529 in plain text", dict(status=529, body=ECHO), TransientError, True),
    ("no connection", dict(error="ConnectError", says=ECHO), TransientError, False),      # the SDK keeps this text
    ("a timeout", dict(error="ReadTimeout", says=ECHO), TransientError, False)]            # in the exception it chains

@pytest.mark.parametrize("reply, raised, shown", [row[1:] for row in ECHOES], ids=[row[0] for row in ECHOES])
def test_an_error_that_echoes_the_key_is_raised_without_it_and_unchained(reply, raised, shown):
    backend, sent = wired(key=SECRET, **reply)                        # a key never reaches a log or a run output
    with pytest.raises((TransientError, BackendError)) as err:
        backend.call(ENTRY, None, PARTS, SCHEMA)
    assert sent[0].headers["x-api-key"] == SECRET                     # the request, which the SDK's error holds, had it
    assert type(err.value) is raised and SECRET not in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__     # raised "from None": nothing is chained
    assert SECRET not in "".join(traceback.format_exception(err.value))       # what a script or a notebook prints
    if shown:                                                         # each one struck out, and nothing else
        assert "Incorrect API key provided: <key>. Sent <key> to /v1?key=<key>" in str(err.value)

def test_a_back_end_without_a_key_leaves_an_error_message_as_it_is():
    assert sgi.vlm.anthropic.strike_key("503: no answer", "") == "503: no answer"         # str.replace("", ...) would fill it

SAID, CLEAN = f"I cannot act on the key {SECRET} you sent: {SECRET}.", "I cannot act on the key <key> you sent: <key>."
ECHOED = [                                                            # where the key comes back in a 200, the text, finish
    (refused("cyber", SAID), "refusal, category cyber: " + CLEAN, "refused"),
    (refused(None, None, text(SAID)), "refusal, no category given\n" + CLEAN, "refused"),
    (message(THOUGHT, text(SAID), stop="max_tokens"), CLEAN, "truncated")]

@pytest.mark.parametrize("body, clean, finish", ECHOED, ids=[f"{row[2]}_{i}" for i, row in enumerate(ECHOED)])
def test_text_that_echoes_the_key_is_returned_cached_and_recorded_without_it(body, clean, finish, tmp_path):
    backend, sent = wired(body=body, key=SECRET)
    result = backend.call(ENTRY, None, PARTS, SCHEMA)
    assert (result.finish, result.text) == (finish, clean)
    response = VLMClient(ENTRY, backend, Cache(tmp_path)).generate(PARTS, Label)
    assert (response.status, response.error, response.text) == ("parse_failed", finish, clean)
    assert (response.validation_retries, len(sent)) == (0, 2)          # a refusal and a cut are the model's outcome: not asked again
    (stored,) = tmp_path.glob("*.json")                                # a parse_failed answer is cached, and replayed in every later run
    assert SECRET not in stored.read_text() and to_call("goal", response).raw_text == clean
    backend, _ = wired(body=message(text(json.dumps({"label": SECRET}))), key=SECRET)      # an answer that parses is struck too
    assert backend.call(ENTRY, None, PARTS, SCHEMA).text == '{"label": "<key>"}'

@pytest.mark.parametrize("status", [200, 401])
def test_debug_logging_never_shows_the_key(status, caplog):
    caplog.set_level(logging.DEBUG)                                   # as ANTHROPIC_LOG=debug in the shell, or a script, would
    body = message(THOUGHT, text(GOOD)) if status == 200 else error("authentication_error", "invalid x-api-key")
    backend, sent = wired(status=status, body=body, key=SECRET)
    try:
        backend.call(ENTRY, None, PARTS, SCHEMA)
    except BackendError:
        assert status == 401
    assert len(sent) == 1 and sent[0].headers["x-api-key"] == SECRET  # the key went out in a header of the request,
    assert "anthropic" in {record.name.split(".")[0] for record in caplog.records}       # the SDK logged the request,
    assert SECRET not in caplog.text                                  # and no line of that holds the key

def model(name):                                                      # a model as the listing gives one
    return {"type": "model", "id": name, "display_name": name.title(), "created_at": "2026-09-01T00:00:00Z"}

def test_list_models_returns_the_sorted_ids_of_every_page_the_provider_reports():
    pages = {None: {"data": [model("claude-sonnet-5-5"), model("claude-opus-5-5")], "has_more": True,
                    "first_id": "claude-sonnet-5-5", "last_id": "claude-opus-5-5"},
             "claude-opus-5-5": {"data": [model("claude-haiku-5-5")], "has_more": False, "first_id": "claude-haiku-5-5",
                                 "last_id": "claude-haiku-5-5"}}
    sent = []
    def handler(request):
        sent.append(request)
        return httpx2.Response(200, json=pages[request.url.params.get("after_id")])
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    backend = AnthropicBackend(KEY, client=anthropic.Anthropic(api_key=KEY, http_client=http))
    assert backend.list_models() == ["claude-haiku-5-5", "claude-opus-5-5", "claude-sonnet-5-5"]      # as an entry writes them
    assert [(request.method, request.url.path, request.headers["x-api-key"]) for request in sent] == [("GET", "/v1/models", KEY)] * 2

@pytest.mark.parametrize("reply, raised", [(dict(status=401, body=error("authentication_error", ECHO)), BackendError),
                                           (dict(status=529, body=ECHO, headers={"retry-after": "7"}), TransientError),
                                           (dict(status=402, body=error("billing_error", ECHO)), QuotaExceededError),
                                           (dict(error="ConnectError", says=ECHO), TransientError)])
def test_a_failed_model_listing_is_classified_as_a_call_is_and_raised_without_the_key(reply, raised, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: pytest.fail("the SDK retried"))
    backend, sent = wired(key=SECRET, **reply)
    with pytest.raises((TransientError, BackendError)) as err:
        backend.list_models()
    assert type(err.value) is raised and len(sent) == 1 and str(err.value).startswith("anthropic")
    if "status" in reply:                                             # what the server said, with the key struck out,
        assert "Incorrect API key provided: <key>" in str(err.value)  # and the wait it asked for where one is made
        assert getattr(err.value, "retry_after", None) == (7.0 if raised is TransientError else None)
    assert err.value.__cause__ is None and err.value.__suppress_context__     # raised "from None": nothing is chained
    assert SECRET not in "".join(traceback.format_exception(err.value))

CLAUDE = {                                                            # entry: (model id, effort level, dollars per million tokens)
    "claude_opus55": ("claude-opus-5-5", "medium", 4.0, 20.0), "claude_opus55_low": ("claude-opus-5-5", "low", 4.0, 20.0),
    "claude_sonnet55": ("claude-sonnet-5-5", "medium", 2.0, 10.0), "claude_sonnet55_low": ("claude-sonnet-5-5", "low", 2.0, 10.0),
    "claude_haiku55": ("claude-haiku-5-5", "none", 0.1, 0.5), "claude_haiku55_low": ("claude-haiku-5-5", "low", 0.1, 0.5),
    # Older models, for how much the method depends on the model (6 Nov 2026); paced at 25 requests a minute.
    "claude_haiku45": ("claude-haiku-4-5", "none", 1.0, 5.0), "claude_sonnet46_low": ("claude-sonnet-4-6", "low", 3.0, 15.0),
    "claude_sonnet5_low": ("claude-sonnet-5", "low", 2.0, 10.0)}
OLDER_CLAUDE = {"claude_haiku45", "claude_sonnet46_low", "claude_sonnet5_low"}

def test_the_model_table_has_nine_claude_entries_each_with_its_model_its_level_and_its_prices():
    cfg = load_config()
    entries = {name: entry for name, entry in cfg.models.items() if entry.provider == "anthropic"}
    assert {name: (e.model, e.reasoning_effort, e.price_input_per_mtok, e.price_output_per_mtok) for name, e in entries.items()} == CLAUDE
    assert list(entries) == list(CLAUDE) and all(name.startswith("claude_") for name in entries)
    shared = {(e.base_url, e.api_key_env, e.temperature, e.image_detail, e.max_output_tokens) for e in entries.values()}
    assert shared == {(None, None, None, None, None)}
    assert {name: e.max_requests_per_minute for name, e in entries.items()} == {
        name: 25 if name in OLDER_CLAUDE else None for name in CLAUDE}
    # The format each model points best in, by `sgi.smoke --calibrate` (17 October; the older models 4 November);
    # Opus is not calibrated yet.
    assert {name: e.point_format for name, e in entries.items()} == {
        "claude_opus55": "xy_1000", "claude_opus55_low": "xy_1000", "claude_sonnet55": "xy_pixel",
        "claude_sonnet55_low": "xy_pixel", "claude_haiku55": "xy_pixel", "claude_haiku55_low": "xy_pixel",
        "claude_haiku45": "xy_1000", "claude_sonnet46_low": "xy_pixel", "claude_sonnet5_low": "xy_pixel"}
    build_all = AnthropicBackend(KEY).build_request                   # what each sends for its level
    sent = {name: build_all(entry, None, PARTS, SCHEMA) for name, entry in entries.items()}
    assert {name: (r["model"], r["output_config"].get("effort"), r.get("thinking"), r["max_tokens"]) for name, r in sent.items()} == {
        name: (model_id, None if level == "none" else level, {"type": "disabled"} if level == "none" else None, 16000)
        for name, (model_id, level, _, _) in CLAUDE.items()}

@pytest.mark.parametrize("setting, named", [({"temperature": 0.2}, "temperature"), ({"temperature": 1.0}, "temperature"),
                                            ({"image_detail": "high"}, "image_detail"),
                                            ({"temperature": 0.0, "image_detail": "low"}, "temperature and image_detail"),
                                            ({"base_url": "https://gateway.example"}, "base_url"),
                                            ({"api_key_env": "A_KEY"}, "api_key_env")])
def test_an_anthropic_entry_that_sets_what_its_back_end_cannot_send_is_refused_when_the_config_is_loaded(setting, named):
    entry = {"provider": "anthropic", "model": "claude-haiku-5-5", **setting}
    with pytest.raises(ConfigError) as refusal:                       # never dropped unsaid, and never sent to be refused
        load_config(overrides={"models": {"new_entry": entry}})
    assert "new_entry" in str(refusal.value) and f"sets {named}" in str(refusal.value)
    for name in CLAUDE:                                               # on an entry of the table too, as a copy of it would
        with pytest.raises(ConfigError, match=name):
            load_config(overrides={"models": {name: setting}})
    other = {"provider": "openai", "model": "m", "temperature": 0.2, "image_detail": "high"}      # another provider takes both
    assert load_config(overrides={"models": {"new_entry": other}}).models["new_entry"].temperature == 0.2

def test_make_client_builds_the_back_end_with_the_key_of_its_own_variable(monkeypatch):
    built = []
    monkeypatch.setattr(factory, "require_key", lambda name: f"value of {name}")
    monkeypatch.setattr(AnthropicBackend, "__init__", lambda self, api_key, client=None: built.append((self.provider, api_key)))
    cfg = load_config(overrides={"run": {"use_cache": False}})
    clients = [factory.make_client(name, cfg) for name in ("claude_haiku55", "claude_opus55_low")]
    assert [(c.entry.model, c.entry.reasoning_effort) for c in clients] == [("claude-haiku-5-5", "none"), ("claude-opus-5-5", "low")]
    assert built == [("anthropic", "value of ANTHROPIC_API_KEY")] * 2
    assert "key missing (ANTHROPIC_API_KEY)" in "\n".join(line for line in factory.entry_lines(cfg) if line.startswith("claude_"))

def test_a_missing_key_names_the_file_and_the_variable_and_nothing_is_built():
    cfg = load_config(overrides={"run": {"use_cache": False}})
    for name in CLAUDE:
        with pytest.raises(MissingKeyError, match="ANTHROPIC_API_KEY"):          # the safety net: the real .env is not read
            factory.make_client(name, cfg)
        with pytest.raises(MissingKeyError, match="ANTHROPIC_API_KEY"):
            factory.list_models(name, cfg)

LOADED = ("import sys; from sgi.config import load_config; from sgi.vlm import factory; cfg = load_config(overrides={'run': {'use_cache': False}});"
          "factory.require_key = lambda name: 'a-key'; factory.make_client(sys.argv[1], cfg);"
          "print(' '.join(sdk for sdk in ('anthropic', 'openai', 'google.genai') if sdk in sys.modules) or 'none')")

@pytest.mark.parametrize("entry, loaded", [("mock", "none"), ("claude_haiku55", "anthropic")])
def test_a_client_loads_the_sdk_of_its_own_provider_and_no_other(entry, loaded):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}         # a fresh interpreter: this one has every SDK loaded
    done = subprocess.run([sys.executable, "-c", LOADED, entry], env=env, capture_output=True, text=True, cwd=REPO_ROOT)
    assert (done.returncode, done.stdout.strip(), done.stderr) == (0, loaded, "")
