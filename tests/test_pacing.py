import functools
import pytest
from pydantic import BaseModel
from sgi.config import ModelEntry, load_config
from sgi.vlm import factory
from sgi.vlm.base import ImagePart, Pacer, TextPart, TransientError, VLMClient
from sgi.vlm.cache import Cache
from sgi.vlm.mock import MockBackend

class Answer(BaseModel):
    label: str

ENTRY = ModelEntry(provider="mock", model="m1")
PARTS = [TextPart("where?"), ImagePart(b"\x89PNG query", 64, 48)]
GOOD, PROSE, DOWN = {"label": "desk"}, "Answer: desk", TransientError("429 too many requests")
GATEWAY = {"provider": "openai_compatible", "model": "m", "base_url": "https://gateway.example/v1", "api_key_env": "A_KEY"}
MODELS = {"wide": {**GATEWAY, "max_requests_per_minute": 5}, "narrow": {**GATEWAY, "model": "m2", "max_requests_per_minute": 2},
          "elsewhere": {**GATEWAY, "base_url": "https://other.example/v1", "max_requests_per_minute": 1},
          "unlimited": GATEWAY}                           # three entries of one address, and one of another

class Time:                                               # a clock that only a sleep or a back-end call moves
    def __init__(self):
        self.now, self.sleeps = 1000.0, []
    def clock(self):
        return self.now
    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
    def pacer(self, limit):
        return Pacer(limit, sleep=self.sleep, clock=self.clock)

def test_a_limit_of_three_lets_three_starts_through_and_holds_the_fourth_a_minute_after_the_first():
    time = Time()
    pacer = time.pacer(3)
    for _ in range(3):                                    # three at once
        pacer.wait()
    assert time.sleeps == [] and pacer.max_per_minute == 3
    time.now += 10.0
    pacer.wait()                                          # the fourth, ten seconds later
    assert (time.now, time.sleeps) == (1060.0, [50.0])

def test_the_window_slides_over_the_starts_let_through():
    time = Time()
    pacer = time.pacer(3)
    for _ in range(3):                                    # starts at 1000, 1020 and 1040
        pacer.wait()
        time.now += 20.0
    time.now = 1045.0
    pacer.wait()                                          # held until a minute after the first start
    assert time.now == 1060.0
    pacer.wait()                                          # and the next until a minute after the second
    assert (time.now, time.sleeps) == (1080.0, [15.0, 20.0])

def test_a_slow_caller_is_never_held():
    time = Time()
    pacer = time.pacer(3)
    for _ in range(12):                                   # a start every 20 s is three a minute, and no more
        pacer.wait()
        time.now += 20.0
    one = time.pacer(1)
    for _ in range(4):
        one.wait()
        time.now += 60.0
    assert time.sleeps == []
    time.now -= 1.0                                       # 59 s after its last start
    one.wait()
    assert time.sleeps == [1.0]

def test_a_lowered_limit_holds_for_the_starts_already_let_through():
    time = Time()
    pacer = time.pacer(5)
    for _ in range(4):                                    # starts at 1000, 1010, 1020 and 1030
        pacer.wait()
        time.now += 10.0
    time.now, pacer.max_per_minute = 1030.0, 2
    pacer.wait()                                          # two a minute: a minute after the start at 1020
    assert (time.now, time.sleeps) == (1080.0, [50.0])

def test_the_client_waits_its_turn_before_the_first_attempt_and_before_each_retry():
    time, script, called = Time(), [DOWN, PROSE, GOOD], []
    def responder(parts, json_schema):
        called.append(time.now)
        time.now += 0.5                                   # how long the provider takes
        return script.pop(0)
    pacer = time.pacer(1)
    pacer.wait()                                          # another client of the provider has just started a request
    client = VLMClient(ENTRY, MockBackend(responder), sleep=time.sleep, clock=time.clock, rng=lambda: 0.5, pacer=pacer)
    resp = client.generate(PARTS, Answer)
    assert (resp.status, resp.transient_retries, resp.validation_retries) == ("ok", 1, 1)
    assert called == [1060.0, 1120.0, 1180.0]             # the first attempt, the transient retry, the validation retry
    assert time.sleeps == [60.0, 2.0, 57.5, 59.5]         # the 2.0 is the backoff; the others are waits for a turn
    assert resp.latency_s == 0.5                          # the provider's time: no waiting in it
    assert resp.latency_total_s == 180.5                  # from the request to the result: every wait is in it

def test_a_cache_hit_does_not_wait(tmp_path):
    time = Time()
    backend = MockBackend(lambda parts, json_schema: GOOD)
    client = VLMClient(ENTRY, backend, Cache(tmp_path), sleep=time.sleep, clock=time.clock, pacer=time.pacer(1))
    first, again = client.generate(PARTS, Answer), client.generate(PARTS, Answer)
    assert (first.cached, again.cached, len(backend.calls), time.sleeps) == (False, True, 1, [])
    assert client.generate(PARTS, Answer, trial=1).cached is False and time.sleeps == [60.0]     # a request does

def test_a_client_without_a_pacer_never_waits():
    time = Time()
    client = VLMClient(ENTRY, MockBackend(lambda parts, json_schema: GOOD), sleep=time.sleep, clock=time.clock)
    assert [client.generate(PARTS, Answer, trial=trial).status for trial in range(200)] == ["ok"] * 200
    assert time.sleeps == []

@pytest.fixture
def paced(monkeypatch):                                   # make_client in a process that has paced nothing, on a fake clock
    time, registry = Time(), {}
    monkeypatch.setattr(factory, "_PACERS", registry)
    monkeypatch.setattr(factory, "Pacer", functools.partial(Pacer, sleep=time.sleep, clock=time.clock))
    cfg = load_config(overrides={"models": MODELS, "run": {"use_cache": False}})
    def client(entry_name, cfg=cfg):
        return factory.make_client(entry_name, cfg, backend=MockBackend(lambda parts, json_schema: GOOD))
    return time, registry, client

@pytest.mark.parametrize("order", [("wide", "narrow"), ("narrow", "wide")])
def test_clients_of_one_address_share_a_pacer_at_the_lowest_limit(paced, order):
    time, registry, client = paced
    first, second = (client(name) for name in order)
    assert list(registry) == [("openai_compatible", "https://gateway.example/v1")]
    first.generate(PARTS, Answer), second.generate(PARTS, Answer)
    assert time.sleeps == []
    first.generate(PARTS, Answer, trial=1)                # the third start at the address: two a minute, whoever asks
    assert time.sleeps == [60.0]
    client(order[0]).generate(PARTS, Answer)              # a client built later joins them
    second.generate(PARTS, Answer, trial=1)
    assert time.sleeps == [60.0, 60.0]

def test_another_address_is_paced_apart_and_an_entry_without_a_limit_not_at_all(paced):
    time, registry, client = paced
    unlimited, narrow, elsewhere = client("unlimited"), client("narrow"), client("elsewhere")
    assert sorted(base_url for _, base_url in registry) == ["https://gateway.example/v1", "https://other.example/v1"]
    for trial in range(2):
        narrow.generate(PARTS, Answer, trial=trial)       # the gateway's minute is used up
    elsewhere.generate(PARTS, Answer)
    for trial in range(10):
        unlimited.generate(PARTS, Answer, trial=trial)
    assert time.sleeps == []
    elsewhere.generate(PARTS, Answer, trial=1)            # its own limit: one a minute
    assert time.sleeps == [60.0]

def test_the_soclaas_entries_of_the_default_config_share_one_pacer_and_no_other_entry_has_one(paced):
    _, registry, client = paced
    cfg = load_config(overrides={"run": {"use_cache": False}})
    for name in cfg.models:
        client(name, cfg)
    assert {key: pacer.max_per_minute for key, pacer in registry.items()} == {
        ("openai_compatible", "https://soclaas-api.comp.nus.edu.sg/v1"): 80,
        ("anthropic", None): 25}                                       # the three older Claude models share one

def test_cache_only_answers_a_miss_with_an_error_and_never_calls(tmp_path, monkeypatch):
    time = Time()
    backend = MockBackend(lambda parts, json_schema: GOOD)
    client = VLMClient(ENTRY, backend, Cache(tmp_path), sleep=time.sleep, clock=time.clock)
    client.generate(PARTS, Answer)                        # stored
    monkeypatch.setenv("SGI_CACHE_ONLY", "1")             # replaying a paid model: a miss must not become a call
    hit, miss = client.generate(PARTS, Answer), client.generate(PARTS, Answer, trial=1)
    assert hit.cached and hit.status == "ok"
    assert (miss.status, len(backend.calls)) == ("vlm_error", 1) and "SGI_CACHE_ONLY" in miss.error
