from types import SimpleNamespace
import numpy as np
import pytest
from sgi.config import ModelEntry
from sgi.imaging import encode_png
from sgi.prompts import build_context_request, build_locate_request, build_plan_request
from sgi.stages.context import run_context
from sgi.stages.locate import run_locate
from sgi.stages.plan import run_plan
from sgi.types import Point, Subtask
from sgi.vlm.base import TransientError, VLMClient
from sgi.vlm.cache import Cache
from sgi.vlm.mock import MockBackend

REQUEST, HINT = "take me to the info desk", "the desk has a blue sign"
SUBTASK = Subtask(type="navigate", description="Go to the info desk", target="info desk")
IMAGE = encode_png(np.full((240, 320, 3), 128, np.uint8))               # sent size: 320 wide, 240 high
NAV = {"type": "navigate", "description": "Go to the info desk", "target": "info desk"}
PLAN = {"reasoning": "the desk is ahead", "subtasks": [
    {**NAV, "target_point": {"x": 80, "y": 180}},
    {"type": "act", "description": "Ask for directions", "target": None, "target_point": None}]}
STAGES = [lambda client, trial: run_plan(client, "zero_shot", REQUEST, None, IMAGE, None, trial=trial),
          lambda client, trial: run_locate(client, "zero_shot", REQUEST, None, SUBTASK, IMAGE, trial=trial),
          lambda client, trial: run_context(client, "zero_shot", REQUEST, SUBTASK, IMAGE, 1.0, trial=trial)]

def client_for(answer, cache=None):          # the model answers in pixels of the image sent
    backend = MockBackend(lambda parts, json_schema: answer)
    entry = ModelEntry(provider="mock", model="mock", point_format="xy_pixel")
    client = VLMClient(entry, backend, cache, sleep=lambda s: None)
    return SimpleNamespace(entry=entry, generate=client.generate), backend     # all a stage may use of a client

def test_run_plan_converts_the_answer_and_sends_the_built_request():
    client, backend = client_for(PLAN)
    plan, resp = run_plan(client, "cot", REQUEST, HINT, IMAGE, None)
    assert resp.status == "ok" and plan.reasoning == "the desk is ahead" and plan.image_sha256 is None
    assert [(s.type, s.target, s.target_point) for s in plan.subtasks] == [
        ("navigate", "info desk", Point(x=250, y=750)), ("act", None, None)]
    assert backend.calls[0][:2] == build_plan_request("cot", REQUEST, HINT, IMAGE, None, "xy_pixel")
    for image, tour in ((None, None), (IMAGE, [(0, IMAGE)])):
        with pytest.raises(ValueError):
            run_plan(client, "cot", REQUEST, HINT, image, tour)

@pytest.mark.parametrize("frame, status", [(7, "ok"), (3, "parse_failed")])                    # frame 3 was not shown
def test_run_plan_tour_mode_accepts_only_a_frame_that_was_shown(frame, status):
    client, _ = client_for({"subtasks": [{**NAV, "tour_frame": frame}]})
    plan, resp = run_plan(client, "zero_shot", REQUEST, None, None, [(0, IMAGE), (7, IMAGE)])
    assert resp.status == status and (plan is None) is (status != "ok")
    assert plan is None or (plan.subtasks[0].tour_frame, plan.subtasks[0].target_point) == (7, None)

@pytest.mark.parametrize("wire, status, point", [
    ({"x": 80, "y": 180}, "ok", Point(x=250, y=750)), (None, "ok", None),
    ({"x": 400, "y": 100}, "parse_failed", None)])                      # outside the 320-wide image sent
def test_run_locate_reads_the_point_in_the_format_and_size_sent(wire, status, point):
    client, backend = client_for({"target_point": wire})
    location, resp = run_locate(client, "zero_shot", REQUEST, HINT, SUBTASK, IMAGE)
    assert resp.status == status and (location is None) is (status != "ok")
    assert location is None or location.target_point == point
    assert backend.calls[0][:2] == build_locate_request("zero_shot", REQUEST, HINT, SUBTASK, IMAGE, "xy_pixel")

def test_run_context_converts_the_answer_and_sends_the_built_request():
    client, backend = client_for({"relevant_people": "two people at the desk", "stop_elsewhere": False})
    check, resp = run_context(client, "zero_shot", REQUEST, SUBTASK, IMAGE, 1.25)
    assert resp.status == "ok" and check.relevant_people == "two people at the desk" and check.people_matter is False
    assert backend.calls[0][:2] == build_context_request("zero_shot", REQUEST, SUBTASK, IMAGE, 1.25)

@pytest.mark.parametrize("stage", STAGES)
def test_not_ok_gives_none_with_the_response_and_the_trial_reaches_the_cache_key(stage, tmp_path):
    client, _ = client_for("not json", cache=Cache(tmp_path))
    outs = [stage(client, trial) for trial in (0, 0, 1)]
    assert [(out, resp.status, resp.cached) for out, resp in outs] == [
        (None, "parse_failed", False), (None, "parse_failed", True), (None, "parse_failed", False)]
    out, resp = stage(client_for(TransientError("down"))[0], 0)
    assert out is None and resp.status == "vlm_error"
