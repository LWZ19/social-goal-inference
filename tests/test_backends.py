import json
import logging
import traceback
import httpx
import httpx2
import openai
import pytest
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel
import sgi.vlm.gemini
import sgi.vlm.openai
from sgi.config import ConfigError, ModelEntry, load_config
from sgi.types import Usage
from sgi.vlm import factory
from sgi.vlm.base import (BackendError, BackendResult, ImagePart, QuotaExceededError, TextPart, TransientError, VLMClient,
                          to_call)
from sgi.vlm.cache import Cache
from sgi.vlm.gemini import GeminiBackend
from sgi.vlm.mock import MockBackend, centre_responder
from sgi.vlm.openai import OpenAIBackend

PNG = b"\x89PNG\r\n\x1a\n"                                             # its base64 text is iVBORw0KGgo=
PARTS = [TextPart("example"), ImagePart(PNG, 4, 2), TextPart("where?"), ImagePart(PNG, 4, 2)]
SCHEMA = {"title": "Label", "type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}
ENTRY = {"openai": ModelEntry(provider="openai", model="gpt-6-luna"),
         "gemini": ModelEntry(provider="gemini", model="gemini-3.8-flash")}
TUNED = dict(reasoning_effort="medium", temperature=0.2, image_detail="high", max_output_tokens=4096, timeout_s=30.0)
VERSION = {"openai": "gpt-6-luna-2026-09-01", "gemini": "gemini-3.8-flash-001"}      # what the server echoes

def test_openai_request_sends_only_what_the_entry_sets():
    build, text = OpenAIBackend("sk-test").build_request, lambda t: {"type": "input_text", "text": t}
    image = {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}
    fixed = {"model": "gpt-6-luna", "store": False,
             "text": {"format": {"type": "json_schema", "name": "answer", "schema": SCHEMA, "strict": True}}}
    assert build(ENTRY["openai"], "sys", PARTS, SCHEMA) == {
        **fixed, "instructions": "sys", "reasoning": {"effort": "low"}, "timeout": 180.0,
        "input": [{"role": "user", "content": [text("example"), image, text("where?"), image]}]}
    assert build(ENTRY["openai"].model_copy(update=TUNED), None, PARTS[2:], SCHEMA) == {
        **fixed, "reasoning": {"effort": "medium"}, "timeout": 30.0, "temperature": 0.2, "max_output_tokens": 4096,
        "input": [{"role": "user", "content": [text("where?"), {**image, "detail": "high"}]}]}

def test_gemini_request_sends_only_what_the_entry_sets():
    build, text = GeminiBackend("g-test").build_request, lambda t: {"type": "text", "text": t}
    image = {"type": "image", "data": "iVBORw0KGgo=", "mime_type": "image/png"}      # base64 text, never bytes
    fixed = {"model": "gemini-3.8-flash", "store": False,
             "response_format": {"type": "text", "mime_type": "application/json", "schema": SCHEMA}}
    assert build(ENTRY["gemini"], "sys", PARTS, SCHEMA) == {
        **fixed, "system_instruction": "sys", "generation_config": {"thinking_level": "low"}, "timeout": 180.0,
        "input": [text("example"), image, text("where?"), image]}
    assert build(ENTRY["gemini"].model_copy(update=TUNED), None, PARTS[2:], SCHEMA) == {
        **fixed, "timeout": 30.0, "input": [text("where?"), {**image, "resolution": "high"}],
        "generation_config": {"thinking_level": "medium", "temperature": 0.2, "max_output_tokens": 4096}}

def test_a_back_end_builds_its_client_from_the_key_alone_with_sdk_retries_off(monkeypatch):
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)                       # the SDKs' fallback; with no key they raise
    oa, gm = OpenAIBackend("sk-test"), GeminiBackend("g-test")        # building a client touches no network
    assert (oa.provider, oa._client.api_key, oa._client.max_retries) == ("openai", "sk-test", 0)
    assert gm.provider == "gemini" and gm._client.interactions.sdk_configuration.retry_config is None

def wired(provider, status=200, body=None, headers=None, error=None, says="no answer", key=None):
    # a back end on a real SDK client with default settings, whose HTTP layer is this handler: no network
    lib, sent = (httpx2 if provider == "openai" else httpx), []
    key = key or KEY[provider]
    def handler(request):
        sent.append(request)
        if error:
            raise getattr(lib, error)(says, request=request)
        return lib.Response(status, json=body, headers=headers or {})
    if provider == "openai":
        http = httpx2.Client(transport=httpx2.MockTransport(handler))
        return OpenAIBackend(key, client=openai.OpenAI(api_key=key, http_client=http)), sent
    options = genai_types.HttpOptions(httpx_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return GeminiBackend(key, client=genai.Client(api_key=key, http_options=options)), sent

def openai_body(*content, status="completed", **extra):
    return {"status": status, "model": VERSION["openai"], **extra,
            "output": [{"type": "reasoning"}, {"type": "message", "content": list(content)}],
            "usage": {"input_tokens": 100, "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 30}}}

def gemini_body(text=None, status="completed", **extra):
    answer = [{"type": "model_output", "content": [{"type": "text", "text": text}]}] if text else []
    return {"status": status, "model": VERSION["gemini"], **extra, "steps": [{"type": "thought"}] + answer,
            "usage": {"total_input_tokens": 100, "total_output_tokens": 20, "total_thought_tokens": 30}}

GOOD, CUT, out = '{"label": "desk"}', '{"lab', lambda t: {"type": "output_text", "text": t}
error = lambda code: {"error": {"code": code, "message": "said the server"}}

@pytest.mark.parametrize("provider, body, text, finish", [
    ("openai", openai_body(out(GOOD)), GOOD, "ok"),
    ("openai", openai_body(out(CUT), status="incomplete", incomplete_details={"reason": "max_output_tokens"}),
     CUT, "truncated"),
    ("openai", openai_body(status="incomplete", incomplete_details={"reason": "content_filter"}), None, "refused"),
    ("openai", openai_body({"type": "refusal", "refusal": "I can't help."}), "I can't help.", "refused"),
    ("gemini", gemini_body(GOOD), GOOD, "ok"),
    ("gemini", gemini_body(CUT, status="incomplete"), CUT, "truncated"),
    ("gemini", gemini_body(status="failed", errors=[{"code": "safety", "message": "blocked"}]), "blocked", "refused")])
def test_response_maps_to_backend_result(provider, body, text, finish):
    backend, sent = wired(provider, body=body)
    usage = Usage(input_tokens=100, output_tokens=20, reasoning_tokens=30)           # OpenAI reports 50 = 20 + 30
    result = backend.call(ENTRY[provider].model_copy(update={"timeout_s": 30.0}), "sys", PARTS, SCHEMA)
    assert result == BackendResult(text, finish, usage, VERSION[provider]) and len(sent) == 1
    assert sent[0].extensions["timeout"]["read"] == 30.0 and json.loads(sent[0].content)["store"] is False

@pytest.mark.parametrize("provider, reported, usage", [               # null when the provider says nothing
    ("openai", {"input_tokens": 100, "output_tokens": 50},
     Usage(input_tokens=100, output_tokens=50, reasoning_tokens=None)),              # nothing is subtracted from 50
    ("gemini", {"total_input_tokens": 100, "total_output_tokens": 20},
     Usage(input_tokens=100, output_tokens=20, reasoning_tokens=None))])
def test_reasoning_tokens_are_none_when_the_provider_does_not_report_them(provider, reported, usage):
    body = {**(openai_body(out(GOOD)) if provider == "openai" else gemini_body(GOOD)), "usage": reported}
    backend, _ = wired(provider, body=body)
    assert backend.call(ENTRY[provider], None, PARTS, SCHEMA) == BackendResult(GOOD, "ok", usage, VERSION[provider])

BOTH = "openai gemini"
QUOTA = "You exceeded your current quota, please check your plan and billing details."     # OpenAI's own sentence
ERRORS = [                                                            # providers, reply, raised, retry_after
    (BOTH, dict(status=429, body=error("rate_limit_exceeded"), headers={"retry-after": "7"}), TransientError, 7.0),
    (BOTH, dict(status=503, body=error("unavailable"), headers={"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}),
     TransientError, None),
    (BOTH, dict(status=429, body=error("rate_limit_exceeded"), headers={"retry-after": "3600"}), TransientError, None),
    (BOTH, dict(error="ReadTimeout"), TransientError, None),
    (BOTH, dict(error="ConnectError"), TransientError, None),
    (BOTH, dict(status=401, body=error("invalid_api_key")), BackendError, None),
    (BOTH, dict(status=400, body=error("invalid_request")), BackendError, None),
    ("openai", dict(status=429, body=error("slow_down"), headers={"retry-after-ms": "1500", "retry-after": "7"}),
     TransientError, 1.5),
    ("openai", dict(status=429, body=error("credit_balance_exhausted")), BackendError, None),
    (BOTH, dict(status=429, body={"error": {"message": "rate limit exceeded"}}, headers={"retry-after": "7"}),
     TransientError, 7.0),
    ("gemini", dict(status=429, body={"error": {"code": "resource_exhausted", "message": QUOTA}}),
     TransientError, None),                                           # the quota stop is the two OpenAI back ends' alone
    ("openai", dict(body=openai_body(status="failed", error={"code": "server_error", "message": "boom"})),
     TransientError, None),
    ("gemini", dict(body=gemini_body(status="failed", errors=[{"code": "api_error", "message": "boom"}])),
     TransientError, None)]

@pytest.mark.parametrize("provider, reply, raised, retry_after",
                         [(p, *row[1:]) for row in ERRORS for p in row[0].split()])
def test_errors_are_classified_after_a_single_attempt(provider, reply, raised, retry_after, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)           # an SDK that retried would sleep here
    backend, sent = wired(provider, **reply)
    with pytest.raises((TransientError, BackendError)) as err:
        backend.call(ENTRY[provider], None, PARTS, SCHEMA)
    assert len(sent) == 1                                             # the SDK's own retries are off
    assert type(err.value) is raised and getattr(err.value, "retry_after", None) == retry_after
    assert str(reply.get("status", "")) in str(err.value)             # an HTTP status is named in the message

KEY = {"openai": "sk-test", "gemini": "g-test"}                       # the key wired() builds each back end with
ECHOES = [                                                            # providers, where the key comes back, raised
    (BOTH, lambda said: dict(status=401, body={"error": {"code": "invalid_api_key", "message": said}}), BackendError),
    (BOTH, lambda said: dict(status=503, body={"error": {"code": "unavailable", "message": said}}), TransientError),
    ("openai", lambda said: dict(body=openai_body(status="failed", error={"code": "server_error", "message": said})),
     TransientError),
    ("gemini", lambda said: dict(body=gemini_body(status="failed", errors=[{"code": "api_error", "message": said}])),
     TransientError),
    ("gemini", lambda said: dict(error="ConnectError", says=said), TransientError)]  # this SDK passes the text on

@pytest.mark.parametrize("provider, reply, raised", [(p, *row[1:]) for row in ECHOES for p in row[0].split()])
def test_an_error_that_echoes_the_key_is_raised_without_it(provider, reply, raised):
    key = KEY[provider]                                               # a key never reaches a log or a run output
    backend, _ = wired(provider, **reply(f"Incorrect API key provided: {key}. Sent {key} to /v1?key={key}"))
    with pytest.raises((TransientError, BackendError)) as err:
        backend.call(ENTRY[provider], None, PARTS, SCHEMA)
    assert type(err.value) is raised and key not in str(err.value)
    assert "Incorrect API key provided: <key>. Sent <key> to /v1?key=<key>" in str(err.value)    # each one, nothing else

@pytest.mark.parametrize("module", [sgi.vlm.openai, sgi.vlm.gemini])
def test_a_back_end_without_a_key_leaves_an_error_message_as_it_is(module):
    assert module.strike_key("503: no answer", "") == "503: no answer"        # str.replace("", ...) would fill it

def test_gemini_block_reported_as_an_http_error_is_a_refusal():
    backend, _ = wired("gemini", status=400, body=error("prohibited_content"))
    assert backend.call(ENTRY["gemini"], None, PARTS, SCHEMA) == BackendResult("said the server", "refused", Usage())

SECRET = "k3y-9f2c7a41d8e6b05-never-shown"                             # unlike any path or source line of a traceback
CHAINED = [                                                           # what the SDK raises with the key in its own text
    ("a 401", lambda said: dict(status=401, body={"error": {"code": "invalid_api_key", "message": said}})),
    ("a 503", lambda said: dict(status=503, body={"error": {"code": "unavailable", "message": said}})),
    ("no connection", lambda said: dict(error="ConnectError", says=said))]

@pytest.mark.parametrize("provider", ["openai", "gemini"])
@pytest.mark.parametrize("reply", [row[1] for row in CHAINED], ids=[row[0] for row in CHAINED])
def test_a_raised_error_leaves_the_sdk_exception_and_its_key_out_of_the_traceback(provider, reply):
    backend, _ = wired(provider, key=SECRET, **reply(f"Incorrect API key provided: {SECRET}"))
    with pytest.raises((TransientError, BackendError)) as err:
        backend.call(ENTRY[provider], None, PARTS, SCHEMA)
    assert err.value.__cause__ is None and err.value.__suppress_context__     # raised "from None": nothing is chained
    assert SECRET not in "".join(traceback.format_exception(err.value))       # what a script or a notebook prints

SPENT = {                                                             # what a 429 says when the allowance is used up,
    "quota exceeded": f"quota exceeded for {SECRET}",                 # each with the key echoed: as a bare string,
    "Budget limit reached": {"error": {"message": f"Budget limit reached for {SECRET}"}},      # as a gateway words it,
    "insufficient_quota": {"error": {"message": f"{QUOTA} Key: {SECRET}", "type": "insufficient_quota", "param": None,
                                     "code": "insufficient_quota"}}}                # and as OpenAI does

@pytest.mark.parametrize("said", SPENT)
def test_an_openai_429_that_says_the_allowance_is_used_up_is_a_quota_error_after_one_request(said, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: pytest.fail("the SDK retried"))
    backend, sent = wired("openai", status=429, body=SPENT[said], headers={"retry-after": "7"}, key=SECRET)
    with pytest.raises(BackendError) as err:
        backend.call(ENTRY["openai"], None, PARTS, SCHEMA)
    assert type(err.value) is QuotaExceededError and len(sent) == 1    # no rate limit: waiting brings no answer
    assert str(err.value).startswith("openai 429: ") and "<key>" in str(err.value)
    assert said.split()[0].lower() in str(err.value).lower()           # the provider's own words are kept
    assert err.value.__cause__ is None and err.value.__suppress_context__     # raised "from None": nothing is chained
    assert SECRET not in "".join(traceback.format_exception(err.value))

@pytest.mark.parametrize("said, raised", [("allowance used up", QuotaExceededError), ("quota exceeded", TransientError)])
def test_openai_reads_a_spent_allowance_by_the_one_rule_the_back_ends_share(said, raised, monkeypatch):
    monkeypatch.setattr("sgi.vlm.base._SPENT_WORDS", ("allowance",))  # the rule is kept once: changed there, it is
    backend, _ = wired("openai", status=429, body={"error": {"message": said}})      # changed for this back end too
    with pytest.raises((TransientError, BackendError)) as err:
        backend.call(ENTRY["openai"], None, PARTS, SCHEMA)
    assert type(err.value) is raised

LISTED = {"openai": {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "openai"}
                                                 for name in ("gpt-6.1-sol", "gpt-6-luna", "embed-4")]},
          "gemini": {"models": [{"name": f"models/{name}", "displayName": name.title()}       # named as a resource
                                for name in ("gemini-3.8-flash", "gemini-3.1-pro-preview", "aqa")]}}
KEY_HEADER = {"openai": ("authorization", "Bearer {}"), "gemini": ("x-goog-api-key", "{}")}

@pytest.mark.parametrize("provider, ids", [("openai", ["embed-4", "gpt-6-luna", "gpt-6.1-sol"]),
                                           ("gemini", ["aqa", "gemini-3.1-pro-preview", "gemini-3.8-flash"])])
def test_list_models_returns_the_sorted_ids_the_provider_reports(provider, ids):
    backend, sent = wired(provider, body=LISTED[provider])
    assert backend.list_models() == ids                               # as a model entry writes them
    (request,) = sent
    header, value = KEY_HEADER[provider]
    assert (request.method, request.url.path.rpartition("/")[2]) == ("GET", "models")
    assert request.headers[header] == value.format(KEY[provider])

FAILED_LISTINGS = [(*row, raised) for row, raised in zip(CHAINED, (BackendError, TransientError, TransientError))]

@pytest.mark.parametrize("provider", ["openai", "gemini"])
@pytest.mark.parametrize("reply, raised", [row[1:] for row in FAILED_LISTINGS], ids=[row[0] for row in FAILED_LISTINGS])
def test_a_failed_model_listing_is_classified_as_a_call_is_and_raised_without_the_key(provider, reply, raised, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)           # an SDK that retried would sleep here
    backend, sent = wired(provider, key=SECRET, headers={"retry-after": "7"}, **reply(f"Incorrect API key provided: {SECRET}"))
    with pytest.raises((TransientError, BackendError)) as err:
        backend.list_models()
    assert type(err.value) is raised and len(sent) == 1 and provider in str(err.value)
    if "status" in reply(""):                                         # what the server said, with the key struck out,
        assert "Incorrect API key provided: <key>" in str(err.value)  # and the wait it asked for where one is made
        assert getattr(err.value, "retry_after", None) == (7.0 if raised is TransientError else None)
    assert err.value.__cause__ is None and err.value.__suppress_context__     # raised "from None": nothing is chained
    assert SECRET not in "".join(traceback.format_exception(err.value))

ECHOED = [                                                            # provider, where the key comes back, finish
    ("openai", lambda said: openai_body({"type": "refusal", "refusal": said}), "refused"),
    ("gemini", lambda said: gemini_body(status="failed", errors=[{"code": "safety", "message": said}]), "refused"),
    ("gemini", lambda said: {"error": {"code": "prohibited_content", "message": said}}, "refused"),   # as an HTTP 400
    ("openai", lambda said: openai_body(out(said), status="incomplete", incomplete_details={"reason": "max_output_tokens"}),
     "truncated"),
    ("gemini", lambda said: gemini_body(said, status="incomplete"), "truncated")]

@pytest.mark.parametrize("provider, body, finish", ECHOED, ids=[f"{row[0]}_{row[2]}_{i}" for i, row in enumerate(ECHOED)])
def test_text_that_echoes_the_key_is_returned_cached_and_recorded_without_it(provider, body, finish, tmp_path):
    said, clean = f"I cannot act on the key {SECRET} you sent: {SECRET}.", "I cannot act on the key <key> you sent: <key>."
    reply = body(said)
    backend, _ = wired(provider, status=400 if "error" in reply else 200, body=reply, key=SECRET)
    result = backend.call(ENTRY[provider], None, PARTS, SCHEMA)
    assert (result.finish, result.text) == (finish, clean)
    response = VLMClient(ENTRY[provider], backend, Cache(tmp_path)).generate(PARTS, Label)
    assert (response.status, response.error, response.text) == ("parse_failed", finish, clean)
    (stored,) = tmp_path.glob("*.json")                                # a parse_failed answer is cached, and replayed in every later run
    assert SECRET not in stored.read_text() and to_call("goal", response).raw_text == clean

@pytest.mark.parametrize("status", [200, 401])
def test_gemini_debug_logging_never_shows_the_key(status, monkeypatch, caplog):
    monkeypatch.setenv("GOOGLE_GENAI_DEBUG", "1")                     # the SDK reads it when the back end is built
    caplog.set_level(logging.DEBUG)                                   # as a script that configured debug logging would
    backend, sent = wired("gemini", status=status, body=gemini_body(GOOD) if status == 200 else error("invalid_api_key"),
                          key=SECRET)
    try:
        backend.call(ENTRY["gemini"], None, PARTS, SCHEMA)
    except BackendError:
        assert status == 401
    assert len(sent) == 1 and SECRET in sent[0].headers.values()      # the key went out in a header of the request,
    assert SECRET not in caplog.text                                  # which the SDK's debug logger prints whole

def test_the_key_goes_to_the_providers_own_host_whatever_the_shell_points_at(monkeypatch):
    for name in ("OPENAI_BASE_URL", "GOOGLE_GEMINI_BASE_URL", "GOOGLE_VERTEX_BASE_URL"):
        monkeypatch.setenv(name, "https://gateway.example/v1")        # set for another tool, or by someone else
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    sent = []
    def answering(lib):                                               # in place of the network, under the clients the back ends build
        def handle_request(self, request):
            sent.append(request)
            return lib.Response(401, json=error("invalid_api_key"), request=request)
        return handle_request
    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", answering(httpx2))
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", answering(httpx))
    for backend in (OpenAIBackend("sk-test"), GeminiBackend("g-test")):
        with pytest.raises(BackendError):
            backend.call(ENTRY[backend.provider], None, PARTS, SCHEMA)
        with pytest.raises(BackendError):
            backend.list_models()                                     # the listing carries the key too
    assert [str(request.url) for request in sent] == [
        "https://api.openai.com/v1/responses", "https://api.openai.com/v1/models",
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        "https://generativelanguage.googleapis.com/v1beta/models"]

class Label(BaseModel):
    label: str

def test_make_client_mock_needs_no_key_and_use_cache_attaches_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(factory, "require_key", lambda name: pytest.fail(f"{name} was read"))
    ask = lambda cfg: factory.make_client("mock", cfg).generate([TextPart("q")], Label)
    on = load_config(overrides={"run": {"cache_dir": str(tmp_path / "on")}})
    off = load_config(overrides={"run": {"cache_dir": str(tmp_path / "off"), "use_cache": False}})
    first, again = ask(on), ask(on)                                   # two clients, one cache directory
    assert (first.status, first.parsed, first.cached, again.cached) == ("ok", {"label": "mock"}, False, True)
    assert len(list((tmp_path / "on").glob("*.json"))) == 1           # an absolute cache_dir is used as it is
    assert (ask(off).cached, ask(off).cached, (tmp_path / "off").exists()) == (False, False, False)
    stand_in = factory.make_client("openai_fast", off, backend=MockBackend(centre_responder))
    assert stand_in.entry == off.models["openai_fast"]                # a given back end is used as it is: no key
    with pytest.raises(ConfigError, match="no_such_entry"):
        factory.make_client("no_such_entry", off)

def test_relative_cache_dir_is_under_the_repo_root(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(factory, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(elsewhere)                                      # a run resumed from another folder
    cfg = load_config(overrides={"run": {"cache_dir": ".cache/vlm"}})
    assert factory.make_client("mock", cfg).generate([TextPart("q")], Label).status == "ok"
    assert len(list((tmp_path / ".cache" / "vlm").glob("*.json"))) == 1 and list(elsewhere.iterdir()) == []

def test_make_client_builds_the_back_end_of_the_entrys_provider_with_its_key(monkeypatch):
    built = []
    monkeypatch.setattr(factory, "require_key", lambda name: f"value of {name}")
    for cls in (GeminiBackend, OpenAIBackend):                        # record the key; build no SDK client
        monkeypatch.setattr(cls, "__init__", lambda self, api_key, client=None: built.append((self.provider, api_key)))
    cfg = load_config(overrides={"run": {"use_cache": False}})
    models = [factory.make_client(name, cfg).entry.model for name in ("gemini_fast", "openai_strong")]
    assert models == ["gemini-3.8-flash", "gpt-6.1-sol"]
    assert built == [("gemini", "value of GEMINI_API_KEY"), ("openai", "value of OPENAI_API_KEY")]
