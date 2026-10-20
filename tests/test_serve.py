import base64
import io
import json
import numpy as np
import pytest
from PIL import Image
from sgi import serve
from sgi.inputs import ObservationData
from sgi.pipeline import build_pipeline
from tests.helpers import floor_observation

def body_for(obs: ObservationData, request: str = "go to the desk") -> dict:
    png, npy = io.BytesIO(), io.BytesIO()
    Image.fromarray(obs.rgb).save(png, format="PNG")
    np.save(npy, obs.depth)
    cam = obs.camera
    return {"request": request, "context": None,
            "rgb_png": base64.b64encode(png.getvalue()).decode(), "depth_npy": base64.b64encode(npy.getvalue()).decode(),
            "camera": {"fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy, "width": cam.width, "height": cam.height,
                       "T_world_camera": cam.T_world_camera.tolist()},
            "robot": {"x": obs.robot.x, "y": obs.robot.y, "yaw": obs.robot.yaw}}

def mock_pipeline(goal: str = "constraints"):
    cfg = serve.make_config("mock", goal=goal)
    return build_pipeline(cfg.model_copy(update={"run": cfg.run.model_copy(update={"use_cache": False})}))

def test_the_body_gives_back_the_observation_it_was_made_from():
    obs = floor_observation()
    got = serve.observation_from(body_for(obs))
    assert np.array_equal(got.rgb, obs.rgb) and np.array_equal(got.depth, obs.depth)
    assert np.allclose(got.camera.T_world_camera, obs.camera.T_world_camera) and got.robot == obs.robot

@pytest.mark.parametrize("change, message", [
    (lambda b: b.pop("rgb_png"), "no 'rgb_png'"),
    (lambda b: b["camera"].update(width=320), "differ in size"),
    (lambda b: b["camera"].update(T_world_camera=[[1, 0], [0, 1]]), "not 4 x 4"),
    (lambda b: b.update(rgb_png="not base64 at all"), "cannot be read"),
])
def test_a_broken_body_is_a_request_error(change, message):
    body = body_for(floor_observation())
    change(body)
    with pytest.raises(serve.RequestError, match=message):
        serve.observation_from(body)

def test_direct_marking_drops_a_strategy_that_needs_example_scenes():
    assert serve.make_config("mock", goal="direct_marking").stages["goal"].strategy == "cot"
    assert serve.make_config("mock", strategies=["goal=few_shot"], goal="direct_marking").stages["goal"].strategy == "few_shot"
    assert serve.make_config("mock").run.goal == "constraints"

@pytest.mark.parametrize("goal", ["constraints", "direct_marking"])
def test_the_mock_answers_with_a_goal_in_both_frames(goal):
    obs = floor_observation()
    answer = serve.infer(mock_pipeline(goal), body_for(obs))
    assert answer["status"] == "ok" and answer["subtasks"][0][0] == "navigate" and answer["subtask_index"] == 0
    x, y, theta = answer["pose_world"]
    rx, ry, rtheta = answer["pose_robot"]
    assert np.isclose(np.hypot(x - obs.robot.x, y - obs.robot.y), np.hypot(rx, ry), atol=1e-6)
    assert [stage for stage, _ in answer["calls"]][:2] == ["plan", "context"]
    shown = answer["inference"]
    assert shown["plan"]["subtasks"][0]["type"] == "navigate" and "image_sha256" not in shown["plan"]
    assert [call["stage"] for call in shown["calls"]] == [stage for stage, _ in answer["calls"]]
    assert all("raw_text" not in call for call in shown["calls"])
    assert shown["target"]["world"] == answer["target_world"] and isinstance(shown["context_check"]["people_matter"], bool)
    assert (shown["people"] is not None) == (goal == "constraints") and (shown["goal_mark"] is not None) == (goal == "direct_marking")
    json.dumps(answer)                                    # what is sent back has to be JSON as it stands

def call(handler, method: str, path: str, payload=None) -> tuple[int, dict]:
    """One request through the handler on in-memory streams: the suite opens no connection, not even to itself."""
    data = b"" if payload is None else json.dumps(payload).encode()
    h = handler.__new__(handler)
    h.rfile, h.wfile = io.BytesIO(data), io.BytesIO()
    h.command, h.path, h.request_version, h.requestline = method, path, "HTTP/1.1", f"{method} {path} HTTP/1.1"
    h.headers, h.client_address = {"Content-Length": str(len(data))}, ("127.0.0.1", 0)
    getattr(h, f"do_{method}")()
    head, _, content = h.wfile.getvalue().partition(b"\r\n\r\n")
    return int(head.split()[1]), json.loads(content)

def test_the_handler_answers_health_goal_and_bad_bodies():
    handler = serve.make_handler(mock_pipeline())
    assert call(handler, "GET", "/health") == (200, {"ok": True, "goal": "constraints", "models": {
        "plan": "mock", "locate": "mock", "context": "mock", "goal": "mock", "people": "mock", "constraints": "mock"}})
    code, answer = call(handler, "POST", "/goal", body_for(floor_observation()))
    assert code == 200 and answer["status"] == "ok"
    code, answer = call(handler, "POST", "/goal", {"request": "go"})
    assert code == 400 and "no 'rgb_png'" in answer["error"]
    assert call(handler, "POST", "/elsewhere", {})[0] == 404
