import base64
import json
import traceback
import httpx2
import openai
import pytest
from pydantic import BaseModel
import sgi.vlm.compatible
from sgi.config import ConfigError, ModelEntry, load_config
from sgi.types import Usage
from sgi.vlm import factory
from sgi.vlm.base import (BackendError, BackendResult, ImagePart, QuotaExceededError, TextPart, TransientError, VLMClient,
                          to_call)
from sgi.vlm.cache import Cache
from sgi.vlm.compatible import OpenAICompatibleBackend
from sgi.vlm.env import MissingKeyError, require_key

BASE, KEY = "https://gateway.example/v1", "gw-test"                    # a made-up gateway and key: nothing is sent anywhere
SECRET = "k3y-9f2c7a41d8e6b05-never-shown"                             # unlike any path or source line of a traceback
PNG = b"\x89PNG\r\n\x1a\n"                                             # its base64 text is iVBORw0KGgo=
PARTS = [TextPart("example"), ImagePart(PNG, 4, 2), TextPart("where?"), ImagePart(PNG, 4, 2)]
SCHEMA = {"title": "Label", "type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}
ENTRY = ModelEntry(provider="openai_compatible", model="qwen3.8:27b", base_url=BASE, api_key_env="GATEWAY_API_KEY",
                   reasoning_effort="none")
TUNED = dict(reasoning_effort="xhigh", temperature=0.2, image_detail="high", max_output_tokens=4096, timeout_s=30.0)
VERSION = "qwen3.8:27b-q4"                                             # what the server echoes
STRICT = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": SCHEMA}}
SHELL = ("OPENAI_CUSTOM_HEADERS", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID")       # what the SDK reads request headers from

class Label(BaseModel):
    label: str

@pytest.fixture(autouse=True)
def a_shell_without_headers_for_the_sdk(monkeypatch):                  # the tests run the same whatever the shell holds
    for name in SHELL:
        monkeypatch.delenv(name, raising=False)

def build(*request):                                                   # building a client touches no network
    return OpenAICompatibleBackend(KEY, BASE).build_request(*request)

def test_request_sends_only_what_the_entry_sets():
    text = lambda t: {"type": "text", "text": t}
    image = lambda **more: {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo=", **more}}
    assert build(ENTRY, "sys", PARTS, SCHEMA) == {
        "model": "qwen3.8:27b", "timeout": 180.0, "extra_body": {"reasoning_effort": "none"}, "response_format": STRICT,
        "messages": [{"role": "system", "content": "sys"},
                     {"role": "user", "content": [text("example"), image(), text("where?"), image()]}]}
    assert build(ENTRY.model_copy(update=TUNED), None, PARTS[2:], SCHEMA) == {
        "model": "qwen3.8:27b", "timeout": 30.0, "extra_body": {"reasoning_effort": "xhigh"}, "response_format": STRICT,
        "temperature": 0.2, "max_tokens": 4096,
        "messages": [{"role": "user", "content": [text("where?"), image(detail="high")]}]}     # no system message
    url = build(ENTRY, None, PARTS, SCHEMA)["messages"][0]["content"][1]["image_url"]["url"]
    assert base64.b64decode(url.removeprefix("data:image/png;base64,")) == PNG

@pytest.mark.parametrize("mode, sent", [("json_schema", STRICT), ("json_object", {"type": "json_object"}), ("none", None)])
def test_structured_output_decides_the_response_format(mode, sent):
    request = build(ENTRY.model_copy(update={"structured_output": mode}), "sys", PARTS, SCHEMA)
    assert request.get("response_format") == sent and ("response_format" in request) == (sent is not None)

@pytest.mark.parametrize("level", ["none", "low", "xhigh", "max"])     # the SDK's own parameter takes neither xhigh nor max
def test_reasoning_effort_is_always_sent_whatever_the_level_is_called(level):
    request = build(ENTRY.model_copy(update={"reasoning_effort": level}), None, PARTS, SCHEMA)
    assert request["extra_body"] == {"reasoning_effort": level} and "reasoning_effort" not in request

def wired(status=200, body=None, headers=None, error=None, says="no answer", key=KEY):
    # a back end on a real SDK client with default settings, whose HTTP layer is this handler: no network
    sent = []
    def handler(request):
        sent.append(request)
        if error:
            raise getattr(httpx2, error)(says, request=request)
        reply = {"text": body} if isinstance(body, str) else {"json": body}
        return httpx2.Response(status, headers=headers or {}, **reply)
    http = httpx2.Client(transport=httpx2.MockTransport(handler))      # the client is given no address: the back end's is used
    return OpenAICompatibleBackend(key, BASE, client=openai.OpenAI(api_key=key, http_client=http)), sent

USAGE = {"prompt_tokens": 100, "completion_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 30}}
GOOD, CUT = '{"label": "desk"}', '{"lab'
part, PICTURE = (lambda text: {"type": "text", "text": text}), {"type": "image_url", "image_url": {"url": "data:,"}}

def answer(content=None, finish="stop", usage=USAGE, **message):
    said = {"model": VERSION, "choices": [{"index": 0, "finish_reason": finish,
                                           "message": {"role": "assistant", "content": content, **message}}]}
    return said if usage is None else {**said, "usage": usage}

def test_a_call_posts_the_request_to_the_entrys_address_with_the_key_as_bearer(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://elsewhere.example/v1")     # set for another tool, or by someone else
    backend, sent = wired(body=answer(GOOD))
    entry = ENTRY.model_copy(update={"timeout_s": 30.0, "reasoning_effort": "max"})
    assert backend.call(entry, "sys", PARTS, SCHEMA).text == GOOD
    (request,) = sent
    assert (request.method, str(request.url)) == ("POST", BASE + "/chat/completions")
    assert request.headers["authorization"] == f"Bearer {KEY}" and request.extensions["timeout"]["read"] == 30.0
    built = backend.build_request(entry, "sys", PARTS, SCHEMA)         # the level is a field of the body itself
    assert json.loads(request.content) == {"model": built["model"], "messages": built["messages"],
                                           "response_format": STRICT, "reasoning_effort": "max"}

def test_the_back_end_builds_its_client_from_the_key_and_the_address_with_sdk_retries_off(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)               # the SDK's fallback; with no key it raises
    monkeypatch.setenv("OPENAI_BASE_URL", "https://elsewhere.example/v1")
    sent = []
    def handle_request(self, request):                                # in place of the network, under the client the back end builds
        sent.append(request)
        return httpx2.Response(401, json=error("Invalid API key"), request=request)
    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", handle_request)
    backend = OpenAICompatibleBackend(KEY, BASE)
    assert (backend.provider, backend._client.api_key, backend._client.max_retries) == ("openai_compatible", KEY, 0)
    with pytest.raises(BackendError):
        backend.call(ENTRY, None, PARTS, SCHEMA)
    assert [(str(request.url), request.headers["authorization"]) for request in sent] == [
        (BASE + "/chat/completions", f"Bearer {KEY}")]                # where the entry says, whatever address the shell names

OTHERS = "sk-of-another-service"                                      # what a shell holds for another tool
HELD = {"OPENAI_CUSTOM_HEADERS": f"Authorization: Bearer {OTHERS}\nX-Other: {OTHERS}",
        "OPENAI_ORG_ID": f"org-{OTHERS}", "OPENAI_PROJECT_ID": f"proj-{OTHERS}"}

def on_either_client(monkeypatch, injected):
    """How to build the back end, on a client of its own or on one passed in, and what reaches the HTTP layer."""
    sent = []
    def reply(request):
        sent.append(request)
        listing = {"object": "list", "data": [{"id": "m", "object": "model"}]}
        said = listing if request.url.path.endswith("/models") else answer(GOOD)
        return httpx2.Response(200, json=said, request=request)
    if not injected:                                                  # in place of the network, under the client it builds
        monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", lambda self, request: reply(request))
        return (lambda: OpenAICompatibleBackend(KEY, BASE)), sent
    http = httpx2.Client(transport=httpx2.MockTransport(reply))
    return (lambda: OpenAICompatibleBackend(KEY, BASE, client=openai.OpenAI(api_key=KEY, http_client=http))), sent

@pytest.mark.parametrize("injected", [False, True], ids=["a client it builds", "a client passed in"])
@pytest.mark.parametrize("names", [*[(name,) for name in SHELL], SHELL], ids=[*SHELL, "all three"])
def test_no_back_end_is_built_while_the_shell_holds_headers_for_the_sdk(names, injected, monkeypatch):
    for name in names:                                                # the SDK would send an organization, a project
        monkeypatch.setenv(name, HELD[name])                          # and, in place of the entry's key, another one
    backend_on, sent = on_either_client(monkeypatch, injected)
    with pytest.raises(ConfigError) as err:
        backend_on()
    said = str(err.value)                                             # one line: which variables, and bound for where
    assert [name for name in SHELL if name in said] == list(names) and BASE in said and "\n" not in said
    assert all(f"-u {name}" in said for name in names)                # and how to run the command without them
    assert OTHERS not in said and KEY not in said and sent == []      # no value is shown, and nothing was sent
    for name in names:
        monkeypatch.delenv(name)                                      # as `env -u NAME` in front of the command does
    backend = backend_on()
    assert backend.call(ENTRY, None, PARTS, SCHEMA).text == GOOD and backend.list_models() == ["m"]
    assert [str(request.url) for request in sent] == [BASE + "/chat/completions", BASE + "/models"]
    for request in sent:                                              # the entry's key, and nothing a shell could add
        assert request.headers["authorization"] == f"Bearer {KEY}"
        assert not [name for name in ("openai-organization", "openai-project", "x-other") if name in request.headers]

@pytest.mark.parametrize("name", SHELL)
def test_a_shell_variable_that_is_set_and_empty_refuses_the_back_end_too(name, monkeypatch):
    monkeypatch.setenv(name, "")                                      # an empty organization is a header as well
    with pytest.raises(ConfigError, match=name):
        OpenAICompatibleBackend(KEY, BASE)

@pytest.mark.parametrize("body, text, finish", [
    (answer(GOOD), GOOD, "ok"),
    (answer(GOOD, reasoning_content="The desk is on the left."), GOOD, "ok"),      # thinking is no part of the answer
    (answer(CUT, finish="length"), CUT, "truncated"),
    (answer(None, finish="length"), None, "truncated"),               # the whole budget went into thinking
    (answer(None), None, "truncated"),                                # no content, whatever the finish reason says
    (answer(""), None, "truncated"),
    (answer(" \n"), None, "truncated"),
    (answer(None, refusal="I can't help."), "I can't help.", "refused"),
    (answer(GOOD, finish="content_filter"), GOOD, "refused"),
    (answer(None, finish="content_filter"), None, "refused"),
    (answer([part('{"label": '), part('"desk"}')]), GOOD, "ok"),      # content as a list of parts: the text parts, joined
    (answer([PICTURE, part(GOOD)]), GOOD, "ok"),
    (answer([part(CUT)], finish="length"), CUT, "truncated"),
    (answer([]), None, "truncated"),                                  # no text part is no content
    (answer([PICTURE]), None, "truncated"),
    (answer([part(" \n")]), None, "truncated")])
def test_response_maps_to_backend_result(body, text, finish):
    backend, sent = wired(body=body)
    usage = Usage(input_tokens=100, output_tokens=20, reasoning_tokens=30)           # the gateway reports 50 = 20 + 30
    assert backend.call(ENTRY, "sys", PARTS, SCHEMA) == BackendResult(text, finish, usage, VERSION) and len(sent) == 1

@pytest.mark.parametrize("reported, usage", [                         # null when the provider says nothing
    ({"prompt_tokens": 100, "completion_tokens": 50}, Usage(input_tokens=100, output_tokens=50)),
    ({"prompt_tokens": 100, "completion_tokens": 50, "completion_tokens_details": {}},
     Usage(input_tokens=100, output_tokens=50)),                      # nothing is subtracted from 50
    ({"prompt_tokens": 100, "completion_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 0}},
     Usage(input_tokens=100, output_tokens=50, reasoning_tokens=0)),
    (None, Usage())])
def test_reasoning_tokens_are_none_when_the_gateway_does_not_report_them(reported, usage):
    backend, _ = wired(body=answer(GOOD, usage=reported))
    assert backend.call(ENTRY, None, PARTS, SCHEMA) == BackendResult(GOOD, "ok", usage, VERSION)

error = lambda message: {"error": {"message": message, "type": "invalid_request_error"}}
REJECTED = "Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low."
ERRORS = [                                                            # reply, raised, retry_after, what the server said
    (dict(status=429, body=error("slow down"), headers={"retry-after": "7"}), TransientError, 7.0, "slow down"),
    (dict(status=429, body=error("slow down")), TransientError, None, "slow down"),
    (dict(status=429, body=error("slow down"), headers={"retry-after": "3600"}), TransientError, None, "slow down"),
    (dict(status=429, body=error("rate limit exceeded"), headers={"retry-after": "7"}), TransientError, 7.0, "rate limit"),
    (dict(status=503, body=error("busy"), headers={"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}),
     TransientError, None, "busy"),
    (dict(status=500, body=error("boom")), TransientError, None, "boom"),
    (dict(status=502, body="provider request failed"), TransientError, None, "provider request failed"),
    (dict(error="ReadTimeout"), TransientError, None, "timed out"),
    (dict(error="ConnectError"), TransientError, None, "Connection error"),
    (dict(status=400, body=error(REJECTED)), BackendError, None, REJECTED),
    (dict(status=400, body=error("this model does not support image input")), BackendError, None, "image input"),
    (dict(status=401, body=error("Invalid API key")), BackendError, None, "Invalid API key"),
    (dict(status=403, body=error("model not allowed for this key")), BackendError, None, "not allowed"),
    (dict(status=404, body=error("model 'qwen9' not found")), BackendError, None, "model 'qwen9' not found")]

@pytest.mark.parametrize("reply, raised, retry_after, said", ERRORS)
def test_errors_are_classified_after_a_single_attempt(reply, raised, retry_after, said, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)           # an SDK that retried would sleep here
    backend, sent = wired(**reply)
    with pytest.raises((TransientError, BackendError)) as err:
        backend.call(ENTRY, None, PARTS, SCHEMA)
    assert len(sent) == 1                                             # the SDK's own retries are off
    assert type(err.value) is raised and getattr(err.value, "retry_after", None) == retry_after
    assert said in str(err.value) and "qwen3.8:27b" in str(err.value)  # what the server said, and of which model
    if "status" in reply:
        assert str(err.value).startswith(f"{reply['status']} ")       # an HTTP status leads the message
    else:
        assert BASE in str(err.value)                                 # no reply: where the request was going

QUOTA = "You exceeded your current quota, please check your plan and billing details."     # OpenAI's own sentence
SPENT = {                                                             # what a 429 says when the allowance is used up,
    "quota exceeded": f"quota exceeded for {SECRET}",                 # each with the key echoed: as plain text,
    "Budget limit reached": error(f"Budget limit reached for {SECRET}"),             # as a gateway words it,
    "insufficient_quota": {"error": {"message": f"{QUOTA} Key: {SECRET}", "type": "insufficient_quota", "param": None,
                                     "code": "insufficient_quota"}}}                # and as OpenAI does

@pytest.mark.parametrize("said", SPENT)
def test_a_429_that_says_the_allowance_is_used_up_is_a_quota_error_after_one_request(said, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: pytest.fail("the SDK retried"))
    backend, sent = wired(status=429, body=SPENT[said], headers={"retry-after": "7"}, key=SECRET)
    with pytest.raises(BackendError) as err:
        backend.call(ENTRY, None, PARTS, SCHEMA)
    assert type(err.value) is QuotaExceededError and len(sent) == 1    # no rate limit: waiting brings no answer
    assert str(err.value).startswith("429 from qwen3.8:27b: ") and "<key>" in str(err.value)
    assert said.split()[0].lower() in str(err.value).lower()           # the provider's own words are kept
    assert err.value.__cause__ is None and err.value.__suppress_context__     # raised "from None": nothing is chained
    assert SECRET not in "".join(traceback.format_exception(err.value))

@pytest.mark.parametrize("said, raised", [("allowance used up", QuotaExceededError), ("quota exceeded", TransientError)])
def test_a_spent_allowance_is_read_by_the_one_rule_the_back_ends_share(said, raised, monkeypatch):
    monkeypatch.setattr("sgi.vlm.base._SPENT_WORDS", ("allowance",))  # the rule is kept once: changed there, it is
    backend, _ = wired(status=429, body=error(said))                  # changed for this back end too
    with pytest.raises((TransientError, BackendError)) as err:
        backend.call(ENTRY, None, PARTS, SCHEMA)
    assert type(err.value) is raised

def test_a_rate_limit_on_a_model_named_with_one_of_the_words_is_still_transient():
    entry = ENTRY.model_copy(update={"model": "budget-qwen:27b"})     # the rule is given what the server said, not the
    backend, _ = wired(status=429, body=error("slow down"))           # message built round it, which names the model
    with pytest.raises((TransientError, BackendError)) as err:
        backend.call(entry, None, PARTS, SCHEMA)
    assert type(err.value) is TransientError and str(err.value).startswith("429 from budget-qwen:27b: ")

def test_through_the_client_a_spent_allowance_is_asked_once_and_let_through():
    backend, sent = wired(status=429, body=error("Budget limit reached"))
    client = VLMClient(ENTRY, backend, sleep=lambda seconds: pytest.fail("the client waited to ask again"))
    with pytest.raises(QuotaExceededError, match="Budget limit reached"):
        client.generate(PARTS, Label)                                 # no vlm_error answer either: the run has to stop
    assert len(sent) == 1

@pytest.mark.parametrize("body", [error("upstream failed"), "<html>Sign in to continue: upstream failed</html>",
                                  {"model": VERSION, "choices": []}])
def test_a_reply_that_holds_no_answer_is_transient(body):
    backend, sent = wired(body=body)                                  # a 200, and no model answer in it
    with pytest.raises(TransientError) as err:
        backend.call(ENTRY, None, PARTS, SCHEMA)
    assert len(sent) == 1 and "qwen3.8:27b" in str(err.value)
    assert isinstance(body, dict) and "choices" in body or "upstream failed" in str(err.value)

def test_a_long_reply_without_an_answer_is_shown_cut_short_and_never_with_part_of_the_key():
    backend, _ = wired(body="x" * 290 + SECRET + "y" * 5000, key=SECRET)       # the cut falls inside the key
    with pytest.raises(TransientError) as err:
        backend.call(ENTRY, None, PARTS, SCHEMA)
    assert len(str(err.value)) < 400 and "x" * 290 + "<key>" in str(err.value) and SECRET[:4] not in str(err.value)

@pytest.mark.parametrize("choice", [{"index": 0, "finish_reason": "stop"}, {"index": 0, "finish_reason": "stop", "message": None}],
                         ids=["no message", "a null message"])
def test_a_choice_without_a_message_is_transient_and_the_reply_is_shown(choice):
    backend, sent = wired(body={"model": VERSION, "choices": [choice], "note": f"sent with {SECRET}"}, key=SECRET)
    with pytest.raises(TransientError) as err:
        backend.call(ENTRY, None, PARTS, SCHEMA)
    said = str(err.value)                                             # what came back, as JSON, with the key struck out
    assert len(sent) == 1 and said.startswith("qwen3.8:27b: the reply holds no answer: ")
    assert '"finish_reason":"stop"' in said and '"note":"sent with <key>"' in said
    assert err.value.__cause__ is None and err.value.__suppress_context__
    assert SECRET not in "".join(traceback.format_exception(err.value))

def test_a_200_whose_body_is_not_the_json_it_says_it_is_is_transient_and_shown_cut_short_without_the_key():
    body = "{" + "x" * 289 + SECRET + "y" * 5000                      # broken off, or never JSON; the cut falls inside the key
    backend, sent = wired(body=body, headers={"content-type": "application/json"}, key=SECRET)
    with pytest.raises(TransientError) as err:
        backend.call(ENTRY, None, PARTS, SCHEMA)
    said = str(err.value)
    assert sent[0].headers["authorization"] == f"Bearer {SECRET}" and len(sent) == 1
    assert said.startswith("qwen3.8:27b: the reply is not JSON: ") and said.endswith("{" + "x" * 289 + "<key>" + "y" * 5)
    assert err.value.__cause__ is None and err.value.__suppress_context__     # the SDK's error holds the whole body
    assert SECRET[:8] not in "".join(traceback.format_exception(err.value))

def test_through_the_client_a_502_is_asked_three_times_and_recorded_as_no_answer():
    backend, sent = wired(status=502, body="provider request failed")
    response = VLMClient(ENTRY, backend, sleep=lambda seconds: None).generate(PARTS, Label)
    assert (response.status, response.provider, response.transient_retries, len(sent)) == (
        "vlm_error", "openai_compatible", 2, 3)                       # the layer's attempts, one request each
    assert response.error.startswith("502 ") and "provider request failed" in response.error
    backend, _ = wired(body=answer(GOOD))
    response = VLMClient(ENTRY, backend).generate(PARTS, Label)
    assert (response.status, response.parsed, response.model_version) == ("ok", {"label": "desk"}, VERSION)
    assert (response.provider, response.reasoning_effort, response.image_size) == ("openai_compatible", "none", (4, 2))

ECHO = f"Incorrect API key provided: {SECRET}. Sent {SECRET} to /v1?key={SECRET}"
ECHOES = [                                                            # where the key comes back, raised, shown struck out
    ("a 400", dict(status=400, body=error(ECHO)), BackendError, True),
    ("a 401", dict(status=401, body=error(ECHO)), BackendError, True),
    ("a 403", dict(status=403, body=error(ECHO)), BackendError, True),
    ("a 404", dict(status=404, body=error(ECHO)), BackendError, True),
    ("a 429", dict(status=429, body=error(ECHO), headers={"retry-after": "7"}), TransientError, True),
    ("a 500", dict(status=500, body=error(ECHO)), TransientError, True),
    ("a 502", dict(status=502, body=ECHO), TransientError, True),
    ("no answer in a 200", dict(body=error(ECHO)), TransientError, True),
    ("no connection", dict(error="ConnectError", says=ECHO), TransientError, False),      # the SDK keeps this text
    ("a timeout", dict(error="ReadTimeout", says=ECHO), TransientError, False)]            # in the exception it chains

@pytest.mark.parametrize("reply, raised, shown", [row[1:] for row in ECHOES], ids=[row[0] for row in ECHOES])
def test_an_error_that_echoes_the_key_is_raised_without_it_and_unchained(reply, raised, shown):
    backend, _ = wired(key=SECRET, **reply)                           # a key never reaches a log or a run output
    with pytest.raises((TransientError, BackendError)) as err:
        backend.call(ENTRY, None, PARTS, SCHEMA)
    assert type(err.value) is raised and SECRET not in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__     # raised "from None": nothing is chained
    assert SECRET not in "".join(traceback.format_exception(err.value))       # what a script or a notebook prints
    if shown:                                                         # each one struck out, and nothing else
        assert "Incorrect API key provided: <key>. Sent <key> to /v1?key=<key>" in str(err.value)

def test_a_back_end_without_a_key_leaves_an_error_message_as_it_is():
    assert sgi.vlm.compatible.strike_key("503: no answer", "") == "503: no answer"        # str.replace("", ...) would fill it

ECHOED = [                                                            # where the key comes back in a 200, finish
    (lambda said: answer(None, refusal=said), "refused"),
    (lambda said: answer(said, finish="content_filter"), "refused"),
    (lambda said: answer(said, finish="length"), "truncated"),
    (lambda said: answer([part(said)], finish="length"), "truncated")]      # as a text part of a list

@pytest.mark.parametrize("body, finish", ECHOED, ids=[f"{row[1]}_{i}" for i, row in enumerate(ECHOED)])
def test_text_that_echoes_the_key_is_returned_cached_and_recorded_without_it(body, finish, tmp_path):
    said, clean = f"I cannot act on the key {SECRET} you sent: {SECRET}.", "I cannot act on the key <key> you sent: <key>."
    backend, _ = wired(body=body(said), key=SECRET)
    result = backend.call(ENTRY, None, PARTS, SCHEMA)
    assert (result.finish, result.text) == (finish, clean)
    response = VLMClient(ENTRY, backend, Cache(tmp_path)).generate(PARTS, Label)
    assert (response.status, response.error, response.text) == ("parse_failed", finish, clean)
    (stored,) = tmp_path.glob("*.json")                                # a parse_failed answer is cached, and replayed in every later run
    assert SECRET not in stored.read_text() and to_call("goal", response).raw_text == clean
    backend, _ = wired(body=answer(json.dumps({"label": SECRET})), key=SECRET)       # an answer that parses is struck too
    assert backend.call(ENTRY, None, PARTS, SCHEMA).text == '{"label": "<key>"}'

def test_list_models_returns_the_sorted_ids_the_gateway_reports():
    listing = {"object": "list", "data": [{"id": name, "object": "model"} for name in ("qwen3.8:27b", "bge-m3", "x-test-1")]}
    backend, sent = wired(body=listing)
    assert backend.list_models() == ["bge-m3", "qwen3.8:27b", "x-test-1"]
    (request,) = sent
    assert (request.method, str(request.url), request.headers["authorization"]) == ("GET", BASE + "/models", f"Bearer {KEY}")

def test_a_line_of_the_listing_carries_what_else_the_gateway_reports_of_a_model():
    usual = {"object": "model", "created": 1759900000, "owned_by": "gateway"}        # said of every model: not shown
    listing = {"object": "list", "data": [
        {"id": "qwen3.8:27b", **usual},
        {"id": "x-test-1", **usual, "gateway": {"alias_of": "glm-5.3", "price_input": 408922}, "capabilities": ["vision"]},
        {"id": "bge-m3", "object": "model", "description": "d" * 400},
        {"id": "echo-1", "object": "model", "note": f"listed for {SECRET}"}]}
    backend, sent = wired(body=listing, key=SECRET)
    assert backend.list_models() == [
        'bge-m3  {"description":"' + "d" * 284,                       # compact JSON, cut at 300 characters
        'echo-1  {"note":"listed for <key>"}',                        # the key is struck here as everywhere
        "qwen3.8:27b",                                                # nothing but the usual fields: the id alone
        'x-test-1  {"gateway":{"alias_of":"glm-5.3","price_input":408922},"capabilities":["vision"]}']
    assert len(sent) == 1

@pytest.mark.parametrize("reply, raised", [(dict(status=401, body=error(ECHO)), BackendError),
                                           (dict(status=502, body=ECHO), TransientError),
                                           (dict(error="ConnectError", says=ECHO), TransientError)])
def test_a_failed_model_listing_is_classified_and_raised_without_the_key(reply, raised):
    backend, sent = wired(key=SECRET, **reply)
    with pytest.raises((TransientError, BackendError)) as err:
        backend.list_models()
    assert type(err.value) is raised and len(sent) == 1 and err.value.__cause__ is None and err.value.__suppress_context__
    assert SECRET not in "".join(traceback.format_exception(err.value)) and "model listing" in str(err.value)

JSON = {"content-type": "application/json"}
NO_LISTING = [                                                        # what came back with a 200, its headers, what is shown of it
    ("a page", "<" + "x" * 289 + SECRET + "y" * 5000, None, "<" + "x" * 289 + "<key>" + "y" * 5),
    ("JSON broken off", '{"object": "list", "data": [{"id": "' + SECRET, JSON, '{"object": "list", "data": [{"id": "<key>'),
    ("an error", error(f"upstream failed for {SECRET}"), None, "upstream failed for <key>"),
    ("no data", {"object": "list", "data": None}, None, '"data":null'),
    ("a model without an id", {"object": "list", "data": [{"object": "model", "owned_by": SECRET}]}, None, '"owned_by":"<key>"')]

@pytest.mark.parametrize("body, headers, shown", [row[1:] for row in NO_LISTING], ids=[row[0] for row in NO_LISTING])
def test_a_listing_reply_that_is_no_listing_is_a_backend_error_showing_its_start_without_the_key(body, headers, shown):
    backend, sent = wired(body=body, headers=headers, key=SECRET)
    with pytest.raises((TransientError, BackendError)) as err:
        backend.list_models()
    said = str(err.value)                                             # nothing asks a listing again: it is no transient error
    assert type(err.value) is BackendError and len(sent) == 1 and said.startswith(f"the model listing at {BASE} ")
    assert shown in said and len(said) < 400 and SECRET[:8] not in said
    assert err.value.__cause__ is None and err.value.__suppress_context__
    assert SECRET[:8] not in "".join(traceback.format_exception(err.value))

@pytest.mark.parametrize("address", ["", "  ", None], ids=["empty", "blank", "missing"])
def test_no_back_end_is_built_without_the_address_of_its_service(address, monkeypatch):
    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", lambda self, request: pytest.fail("a request was sent"))
    passed_in = openai.OpenAI(api_key=KEY)                            # its address is the SDK's own: OpenAI's
    for client in (None, passed_in):                                  # either would send the gateway's key there
        with pytest.raises(ValueError, match="base_url"):
            OpenAICompatibleBackend(KEY, address, client=client)

GATEWAYS = {"run": {"use_cache": False},
            "models": {"other_gateway": {"provider": "openai_compatible", "model": "m", "base_url": BASE,
                                         "api_key_env": "OTHER_API_KEY"}}}

def test_make_client_builds_the_back_end_with_the_key_variable_and_the_address_the_entry_names(monkeypatch):
    built = []
    monkeypatch.setattr(factory, "require_key", lambda name: f"value of {name}")
    monkeypatch.setattr(OpenAICompatibleBackend, "__init__",          # record the arguments; build no SDK client
                        lambda self, api_key, base_url, client=None: built.append((api_key, base_url)))
    cfg = load_config(overrides=GATEWAYS)
    clients = [factory.make_client(name, cfg) for name in ("soclaas_glm53_max", "other_gateway")]
    assert [(c.entry.model, c.entry.reasoning_effort) for c in clients] == [("x-test-1", "max"), ("m", "low")]
    assert built == [("value of SOCLAAS_API_KEY", "https://soclaas-api.comp.nus.edu.sg/v1"), ("value of OTHER_API_KEY", BASE)]

def test_a_missing_key_names_the_file_and_the_variable_of_the_entry(tmp_path, monkeypatch):
    cfg = load_config(overrides=GATEWAYS)
    with pytest.raises(MissingKeyError, match="SOCLAAS_API_KEY"):     # the safety net: the real .env is not read
        factory.make_client("soclaas_qwen38", cfg)
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=key-of-another-provider\nSOCLAAS_API_KEY=\n")     # .env.example copied, one key filled in
    monkeypatch.setattr(factory, "require_key", lambda name: require_key(name, env))
    for name, variable in (("soclaas_qwen38", "SOCLAAS_API_KEY"), ("other_gateway", "OTHER_API_KEY")):
        with pytest.raises(MissingKeyError) as err:
            factory.make_client(name, cfg)
        assert str(env) in str(err.value) and variable in str(err.value) and "key-of-another-provider" not in str(err.value)


MISMATCH = ("Structured output processing only supports one backend per engine. The engine is already using "
            "'guidance', but this request resolved to 'xgrammar'. Configure `structured_outputs_config.backend` "
            "explicitly or use schemas supported by the initialized backend.")     # what a gateway said


def in_turn(*replies):
    """A back end whose server gives these replies in turn, (status, body) each: no network."""
    sent = []
    def handler(request):
        status, body = replies[min(len(sent), len(replies) - 1)]
        sent.append(request)
        return httpx2.Response(status, json=body)
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    return OpenAICompatibleBackend(KEY, BASE, client=openai.OpenAI(api_key=KEY, http_client=http)), sent


def test_a_400_from_an_engine_set_to_another_structured_output_backend_is_asked_again():
    # several engines serve one model name, each locked to a backend: the same request can land on one that takes it
    backend, sent = in_turn((400, error(MISMATCH)), (400, error(MISMATCH)), (200, answer(GOOD)))
    result = backend.call(ENTRY, None, PARTS, SCHEMA)
    assert len(sent) == 3 and json.loads(result.text) == {"label": "desk"}


def test_the_engine_mismatch_is_asked_at_most_ten_times_and_any_other_400_once():
    backend, sent = in_turn((400, error(MISMATCH)))
    with pytest.raises(BackendError) as err:
        backend.call(ENTRY, None, PARTS, SCHEMA)
    assert len(sent) == 10 and "one backend per engine" in str(err.value)
    backend, sent = in_turn((400, error(REJECTED)), (200, answer(GOOD)))
    with pytest.raises(BackendError):
        backend.call(ENTRY, None, PARTS, SCHEMA)
    assert len(sent) == 1


def test_an_engine_on_guidance_is_sent_the_schema_in_a_form_that_backend_takes():
    # the gateway's own advice: "use schemas supported by the initialized backend"; the form keeps every answer valid
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body["response_format"]["json_schema"]["schema"])
        if len(seen) == 1:
            return httpx2.Response(400, json=error(MISMATCH))           # MISMATCH says the engine uses guidance
        return httpx2.Response(200, json=answer(GOOD))
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    backend = OpenAICompatibleBackend(KEY, BASE, client=openai.OpenAI(api_key=KEY, http_client=http))
    schema = {"type": "object", "properties": {"label": {"type": "string"}, "n": {"type": "integer"},
                                               "kind": {"type": "string", "enum": ["a", "b"]}}}
    backend.call(ENTRY, None, PARTS, schema)
    assert seen[0] == schema                                             # sent as given first
    assert seen[1]["properties"]["n"]["multipleOf"] == 1                 # then in guidance's form
    assert seen[1]["properties"]["label"]["pattern"] == "^[\\s\\S]*$"
    assert "pattern" not in seen[1]["properties"]["kind"]                # an enum is left as it is
    assert "multipleOf" not in schema["properties"]["n"]                 # the caller's schema is not changed
