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


# ---- /resolve: a stored constraint answer solved again on tracked people

def tracks_of(people, shift=(0.0, 0.0)):
    return [{"id": 10 + i, "xy": [p["xy"][0] + shift[0], p["xy"][1] + shift[1]], "yaw": p["yaw"]}
            for i, p in enumerate(people)]

def answered(goal: str = "constraints"):
    pipeline, sessions = mock_pipeline(goal), serve.Sessions()
    obs = floor_observation()
    return pipeline, sessions, obs, serve.infer(pipeline, body_for(obs), sessions)

def test_a_constraint_answer_carries_a_session_and_a_marked_point_none():
    *_, answer = answered()
    assert answer["source"] == "constraints" and isinstance(answer["session"], str)
    *_, marked = answered("direct_marking")
    assert marked.get("session") is None

def test_resolving_on_the_same_view_gives_the_same_place():          # the mock finds nobody
    pipeline, sessions, obs, answer = answered()
    people = answer["inference"]["people"]
    body = {**body_for(obs), "session": answer["session"], "t": 3.0, "tracks": tracks_of(people),
            "tracks_at_answer": tracks_of(people)}
    got = serve.resolve(pipeline, sessions, body)
    assert got["status"] == "ok" and got["match_rate"] is None and got["joined"] is False and got["dropped"] == []
    assert np.allclose(got["pose_world"], answer["pose_world"], atol=1e-6)
    assert set(got) >= {"in_turn", "people", "timings_s"}
    json.dumps(got)

def test_resolving_an_unknown_session_is_a_request_error():
    pipeline, sessions, obs, _ = answered()
    with pytest.raises(serve.RequestError, match="no session"):
        serve.resolve(pipeline, sessions, {**body_for(obs), "session": "nope", "t": 0.0, "tracks": [], "tracks_at_answer": []})

def test_the_handler_answers_resolve():
    pipeline = mock_pipeline()
    handler = serve.make_handler(pipeline)
    obs = floor_observation()
    code, answer = call(handler, "POST", "/goal", body_for(obs))
    people = answer["inference"]["people"]
    code, got = call(handler, "POST", "/resolve", {**body_for(obs), "session": answer["session"], "t": 3.0,
                                                    "tracks": tracks_of(people), "tracks_at_answer": tracks_of(people)})
    assert code == 200 and got["status"] == "ok"
    code, got = call(handler, "POST", "/resolve", {**body_for(obs), "session": "nope", "t": 3.0, "tracks": []})
    assert code == 400


def test_the_place_in_turn_follows_the_line_as_it_moves_up():
    import math
    from sgi.stages.goal import standard_goal
    from sgi.types import Constraint, DetectedPerson, GroundedTarget, Point
    pipeline, sessions, obs = mock_pipeline(), serve.Sessions(), floor_observation()
    target = GroundedTarget(point=Point(x=500, y=300), pixel=(320.0, 150.0), world=(1.0, 9.0, 0.0))
    up = math.pi / 2                                       # they face the target, ahead of the robot
    people = [DetectedPerson(number=1, xy=(1.0, 8.0), yaw=up, pixel=(0.0, 0.0)),
              DetectedPerson(number=2, xy=(1.0, 7.0), yaw=up, pixel=(0.0, 0.0))]
    session = sessions.add({"target": target, "people": people, "resolver": None,
                            "standard": standard_goal(obs, target, pipeline.cfg.thresholds),
                            "constraints": [Constraint(because="they wait", kind="come_after", people=[1, 2])]})
    line = lambda front: [{"id": 7, "xy": [1.0, front], "yaw": up}, {"id": 8, "xy": [1.0, front - 1.0], "yaw": up}]
    first = serve.resolve(pipeline, sessions, {**body_for(obs), "session": session, "t": 3.0, "tracks": line(8.0),
                                               "tracks_at_answer": line(8.0)})
    later = serve.resolve(pipeline, sessions, {**body_for(obs), "session": session, "t": 4.0, "tracks": line(8.5)})
    assert first["in_turn"] and first["match_rate"] == 1.0
    assert first["pose_world"][:2] == pytest.approx([1.0, 6.0], abs=0.1)
    assert later["pose_world"][1] - first["pose_world"][1] == pytest.approx(0.5, abs=0.1)

def test_max_rpm_caps_every_entry_and_leaves_a_lower_limit():
    cfg = serve.make_config("mock", max_rpm=40)
    limits = {name: entry.max_requests_per_minute for name, entry in cfg.models.items()}
    assert limits["soclaas_qwen38_low"] == 40 and all(v is None or v <= 40 for v in limits.values())
    assert serve.make_config("mock").models["soclaas_qwen38_low"].max_requests_per_minute == 80

def test_a_resolve_with_a_floor_map_solves_on_it():
    pipeline, sessions, obs, answer = answered()
    mask = np.ones((200, 200), dtype=np.uint8) * 255
    png = io.BytesIO()
    Image.fromarray(mask).save(png, format="PNG")
    floor_map = {"origin": [-4.0, -4.0], "resolution": 0.05, "png": base64.b64encode(png.getvalue()).decode()}
    body = {**body_for(obs), "session": answer["session"], "t": 3.0, "tracks": [], "tracks_at_answer": [],
            "floor_map": floor_map}
    got = serve.resolve(pipeline, sessions, body)
    assert got["status"] == "ok" and got["floor"] == "map"
    again = serve.resolve(pipeline, sessions, {**body_for(obs), "session": answer["session"], "t": 3.5, "tracks": []})
    assert again["floor"] == "map"                          # kept from the first

def test_a_resolve_without_a_floor_map_uses_the_first_view():
    pipeline, sessions, obs, answer = answered()
    got = serve.resolve(pipeline, sessions, {**body_for(obs), "session": answer["session"], "t": 3.0, "tracks": [],
                                             "tracks_at_answer": []})
    assert got["floor"] == "first view"

def test_an_oracle_come_after_replaces_the_answer_for_a_ceiling():
    import math
    from sgi.stages.goal import standard_goal
    from sgi.types import GroundedTarget, Point
    pipeline, sessions, obs = mock_pipeline(), serve.Sessions(), floor_observation()
    target = GroundedTarget(point=Point(x=500, y=300), pixel=(320.0, 150.0), world=(1.0, 9.0, 0.0))
    session = sessions.add({"target": target, "people": [], "constraints": [], "resolver": None,
                            "standard": standard_goal(obs, target, pipeline.cfg.thresholds)})
    up = math.pi / 2
    line = [{"id": 7, "xy": [1.0, 8.0], "yaw": up}, {"id": 8, "xy": [1.0, 7.0], "yaw": up}]
    got = serve.resolve(pipeline, sessions, {**body_for(obs), "session": session, "t": 3.0, "tracks": line,
                                             "tracks_at_answer": line, "oracle": {"come_after": [7, 8]}})
    assert got["in_turn"] and got["pose_world"][:2] == pytest.approx([1.0, 6.0], abs=0.1)
    assert got["oracle"] == ["come_after"]

def test_an_oracle_target_moves_the_target_point():
    pipeline, sessions, obs, answer = answered()
    got = serve.resolve(pipeline, sessions, {**body_for(obs), "session": answer["session"], "t": 3.0, "tracks": [],
                                             "tracks_at_answer": [], "oracle": {"target_xy": [1.5, 6.0]}})
    assert got["oracle"] == ["target"] and sessions.get(answer["session"])["target"].world[:2] == (1.5, 6.0)

def test_a_settled_resolve_holds_the_last_place_while_people_step_up():
    import math
    from sgi.stages.goal import standard_goal
    from sgi.types import Constraint, DetectedPerson, GroundedTarget, Point
    pipeline, sessions, obs = mock_pipeline(), serve.Sessions(), floor_observation()
    target = GroundedTarget(point=Point(x=500, y=300), pixel=(320.0, 150.0), world=(1.0, 9.0, 0.0))
    up = math.pi / 2
    people = [DetectedPerson(number=1, xy=(1.0, 8.0), yaw=up, pixel=(0.0, 0.0)),
              DetectedPerson(number=2, xy=(1.0, 7.0), yaw=up, pixel=(0.0, 0.0))]
    session = sessions.add({"target": target, "people": people, "resolver": None,
                            "standard": standard_goal(obs, target, pipeline.cfg.thresholds),
                            "constraints": [Constraint(because="they wait", kind="come_after", people=[1, 2])]})
    line = lambda front: [{"id": 7, "xy": [1.0, front], "yaw": up}, {"id": 8, "xy": [1.0, front - 1.0], "yaw": up}]
    ask = lambda t, front, **extra: serve.resolve(pipeline, sessions, {**body_for(obs), "session": session, "t": t,
                                                                       "tracks": line(front), "settle": True, **extra})
    first = ask(3.0, 8.0, tracks_at_answer=line(8.0))
    stepping = ask(3.5, 8.3)
    later = ask(5.0, 8.3)
    assert first["status"] == "ok" and stepping["status"] == "held"
    assert stepping["pose_world"] == first["pose_world"]
    assert later["status"] == "ok" and later["pose_world"][1] - first["pose_world"][1] == pytest.approx(0.3, abs=0.1)

def test_a_variant_on_the_first_resolve_sets_the_resolver_options():
    import math
    from sgi.stages.goal import standard_goal
    from sgi.types import Constraint, DetectedPerson, GroundedTarget, Point
    pipeline, sessions, obs = mock_pipeline(), serve.Sessions(), floor_observation()
    target = GroundedTarget(point=Point(x=500, y=300), pixel=(320.0, 150.0), world=(1.0, 9.0, 0.0))
    up = math.pi / 2
    people = [DetectedPerson(number=1, xy=(1.0, 8.0), yaw=up, pixel=(0.0, 0.0)),
              DetectedPerson(number=2, xy=(1.0, 7.3), yaw=up, pixel=(0.0, 0.0))]
    item = lambda: {"target": target, "people": people, "resolver": None,
                    "standard": standard_goal(obs, target, pipeline.cfg.thresholds),
                    "constraints": [Constraint(because="they wait", kind="come_after", people=[1, 2])]}
    line = [{"id": 7, "xy": [1.0, 8.0], "yaw": up}, {"id": 8, "xy": [1.0, 7.3], "yaw": up}]
    body = lambda session, **extra: {**body_for(obs), "session": session, "t": 3.0, "tracks": line,
                                     "tracks_at_answer": line, **extra}
    v1 = serve.resolve(pipeline, sessions, body(sessions.add(item())))
    v2 = serve.resolve(pipeline, sessions, body(sessions.add(item()), variant={
        "readmit": True, "line_span_m": 0.5, "vacate_after_s": 5.0, "path_memory": True}))
    assert v1["line_direction"] is None and v2["line_direction"] == pytest.approx([0.0, -1.0])
    assert v2["solve_direction"] == pytest.approx([0.0, -1.0]) and v2["vacated"] is False
    with pytest.raises(serve.RequestError, match="variant"):
        serve.resolve(pipeline, sessions, body(sessions.add(item()), variant={"bogus": 1}))

def _line_session(pipeline, sessions, obs, people_xy, come_after):
    import math
    from sgi.stages.goal import standard_goal
    from sgi.types import Constraint, DetectedPerson, GroundedTarget, Point
    target = GroundedTarget(point=Point(x=500, y=300), pixel=(320.0, 150.0), world=(1.0, 9.0, 0.0))
    people = [DetectedPerson(number=i + 1, xy=xy, yaw=math.pi / 2, pixel=(0.0, 0.0)) for i, xy in enumerate(people_xy)]
    return sessions.add({"target": target, "people": people, "resolver": None,
                         "standard": standard_goal(obs, target, pipeline.cfg.thresholds),
                         "constraints": [Constraint(because="they wait", kind="come_after", people=come_after)]})


ASIDE_XY = [(1.0, 8.0), (1.0, 7.2), (1.4, 6.4)]          # two waiting, and a third just behind the last, to one side
V2_VARIANT = {"readmit": True, "line_span_m": 0.5, "vacate_after_s": 5.0, "path_memory": True}


def _tracks_at(xy):
    import math
    return [{"id": 7 + i, "xy": list(p), "yaw": math.pi / 2} for i, p in enumerate(xy)]


def test_co_motion_keeps_a_still_person_behind_the_last_out_of_the_place_in_turn():
    pipeline, obs = mock_pipeline(), floor_observation()
    got = {}
    for name, variant in (("v2", V2_VARIANT), ("ct3", {**V2_VARIANT, "co_motion_s": 5.0})):
        sessions = serve.Sessions()
        session = _line_session(pipeline, sessions, obs, ASIDE_XY, [1, 2])
        got[name] = serve.resolve(pipeline, sessions, {**body_for(obs), "session": session, "t": 3.0,
                                                       "tracks": _tracks_at(ASIDE_XY),
                                                       "tracks_at_answer": _tracks_at(ASIDE_XY), "variant": variant})
    assert got["v2"]["turn_joined"] == [9] and got["v2"]["last"] == 9 and got["v2"]["exempt"] == []
    assert got["ct3"]["turn_joined"] == [] and got["ct3"]["last"] == 8 and got["ct3"]["exempt"] == [9]
    assert got["ct3"]["pose_world"][1] == pytest.approx(6.2, abs=0.15)      # behind 8 (kept 1 m from 9, so to one side)
    assert got["v2"]["pose_world"][1] < 5.6                                  # behind 9
    assert got["ct3"]["answer_come_after"] == [[1, 7, 0.0], [2, 8, 0.0]] and got["ct3"]["non_followers"] == {}
    assert got["ct3"]["roster"] == [7, 8] and got["v2"]["roster"] == [7, 8]   # by track id: those taken as waiting


def _wait_history():
    """Tracks every 0.5 s from the photo (t = 0) to just before the first /resolve (t = 10): the front person (7)
    is served at 2 s and walks off, the next (8) steps up 0.8 m at 3-4 s, and a person behind them (9) stays put."""
    import math
    frames = []
    for k in range(20):
        t = 0.5 * k
        front = (1.0, 8.0) if t < 2.0 else (min(1.0 + 1.5 * (t - 2.0), 5.0), 8.0)
        second = (1.0, 7.2 + 0.8 * min(max(t - 3.0, 0.0), 1.0))
        frames.append({"t": t, "robot": [3.0, 3.0],
                       "tracks": [{"id": 7, "xy": list(front), "yaw": math.pi / 2},
                                  {"id": 8, "xy": list(second), "yaw": math.pi / 2},
                                  {"id": 9, "xy": [1.0, 6.1], "yaw": math.pi / 2}]})
    return frames


def test_a_history_from_the_photo_replays_the_wait_before_the_first_solve():
    import math
    pipeline, obs = mock_pipeline(), floor_observation()
    xy = [(1.0, 8.0), (1.0, 7.2), (1.0, 6.1)]             # the answer names all three as waiting
    now = [{"id": 7, "xy": [5.0, 8.0], "yaw": 0.0}, {"id": 8, "xy": [1.0, 8.0], "yaw": math.pi / 2},
           {"id": 9, "xy": [1.0, 6.1], "yaw": math.pi / 2}]
    got = {}
    for name, history in (("from delivery", None), ("from the photo", _wait_history())):
        sessions = serve.Sessions()
        session = _line_session(pipeline, sessions, obs, xy, [1, 2, 3])
        request = {**body_for(obs), "session": session, "t": 10.0, "tracks": now, "tracks_at_answer": _tracks_at(xy),
                   "variant": {**V2_VARIANT, "co_motion_s": 5.0}}
        if history is not None:
            request["history"] = history
        got[name] = serve.resolve(pipeline, sessions, request)
    assert got["from delivery"]["non_followers"] == {} and got["from delivery"]["last"] == 9
    assert set(got["from the photo"]["non_followers"]) == {"9"} and got["from the photo"]["last"] == 8
    assert got["from the photo"]["pose_world"][1] > 6.3      # behind 8, not behind the person who stayed put


def test_a_history_is_read_only_on_the_first_resolve():
    pipeline, sessions, obs = mock_pipeline(), serve.Sessions(), floor_observation()
    xy = [(1.0, 8.0), (1.0, 7.2), (1.0, 6.1)]
    session = _line_session(pipeline, sessions, obs, xy, [1, 2, 3])
    first = {**body_for(obs), "session": session, "t": 0.5, "tracks": _tracks_at(xy), "tracks_at_answer": _tracks_at(xy),
             "variant": {**V2_VARIANT, "co_motion_s": 5.0}}
    serve.resolve(pipeline, sessions, first)
    later = serve.resolve(pipeline, sessions, {**first, "t": 10.0, "history": _wait_history()})
    assert later["non_followers"] == {}                    # the history came too late to be replayed


def test_a_role_belief_gives_the_solver_its_roster_and_exempts_everyone_else():
    pipeline, sessions, obs = mock_pipeline(), serve.Sessions(), floor_observation()
    xy = [(1.0, 8.0), (1.0, 7.2), (2.6, 5.0)]             # two named as waiting; a third, unnamed, well aside
    session = _line_session(pipeline, sessions, obs, xy, [1, 2])
    got = serve.resolve(pipeline, sessions, {**body_for(obs), "session": session, "t": 0.5, "tracks": _tracks_at(xy),
                                             "tracks_at_answer": _tracks_at(xy), "variant": {"role_belief": True}})
    assert got["roster"] == [7, 8] and set(got["marginals"]) == {"7", "8", "9"}
    assert got["last"] == 8 and got["exempt"] == [9]


def test_a_role_decision_gives_the_solver_its_choice_and_logs_the_costs():
    pipeline, sessions, obs = mock_pipeline(), serve.Sessions(), floor_observation()
    xy = [(1.0, 8.0), (1.0, 7.2), (2.6, 5.0)]             # two named as waiting; a third, unnamed, well aside
    session = _line_session(pipeline, sessions, obs, xy, [1, 2])
    request = {**body_for(obs), "session": session, "t": 0.5, "tracks": _tracks_at(xy),
               "tracks_at_answer": _tracks_at(xy),
               "variant": {"role_belief": True, "static_win_s": 5.0, "role_decision": True}}
    got = serve.resolve(pipeline, sessions, request)
    assert sessions.get(session)["belief"].p["STATIC_WIN_S"] == 5.0
    assert got["choice"] == 8 and set(got["costs"]) == {"first", "7", "8", "9"}
    assert got["last"] == 8 and got["exempt"] == [9] and got["roster"] == [7, 8]
    assert sessions.get(session)["choice"] == 8                # kept for the next re-solve of the session
    again = serve.resolve(pipeline, sessions, {**request, "t": 1.0})
    assert again["choice"] == 8


def _role_resolve(xy, come_after, variant, now=None):
    pipeline, sessions, obs = mock_pipeline(), serve.Sessions(), floor_observation()
    session = _line_session(pipeline, sessions, obs, xy, come_after)
    return serve.resolve(pipeline, sessions, {**body_for(obs), "session": session, "t": 0.5,
                                              "tracks": _tracks_at(now or xy), "tracks_at_answer": _tracks_at(xy),
                                              "variant": variant})


def test_a_role_belief_on_the_path_lines_up_along_the_back_of_the_remembered_line():
    bent = [(1.0, 8.0), (1.0, 7.2), (1.6, 6.4)]           # all three named; the line bends at the second
    chord = _role_resolve(bent, [1, 2, 3], {"role_belief": True})
    path = _role_resolve(bent, [1, 2, 3], {"role_belief": True, "role_direction": "path"})
    assert chord["direction_source"] == "belief" and path["direction_source"] == "path"
    assert chord["solve_direction"] == pytest.approx([0.6 / np.hypot(0.6, 1.6), -1.6 / np.hypot(0.6, 1.6)])
    assert path["solve_direction"] == pytest.approx([0.6, -0.8])     # the last segment, where the back member is


def test_a_role_belief_on_the_path_changes_nothing_but_the_direction():
    xy = [(1.0, 8.0), (1.0, 7.2), (2.6, 5.0)]             # a straight line of two: the path and the chord agree
    chord = _role_resolve(xy, [1, 2], {"role_belief": True})
    path = _role_resolve(xy, [1, 2], {"role_belief": True, "role_direction": "path"})
    assert path["solve_direction"] == pytest.approx(chord["solve_direction"])
    assert {k: v for k, v in path.items() if k not in ("direction_source", "timings_s")} == \
        {k: v for k, v in chord.items() if k not in ("direction_source", "timings_s")}


def test_a_role_belief_on_the_path_falls_back_to_its_own_direction_without_a_path():
    xy = [(1.0, 8.0), (2.6, 5.0)]                         # one named: no remembered line
    chord = _role_resolve(xy, [1], {"role_belief": True})
    path = _role_resolve(xy, [1], {"role_belief": True, "role_direction": "path"})
    assert path["direction_source"] == "belief" and path["solve_direction"] == pytest.approx(chord["solve_direction"])


def test_the_role_direction_needs_a_role_belief_and_a_known_source():
    xy = [(1.0, 8.0), (1.0, 7.2)]
    for variant in ({"role_direction": "path"}, {"role_belief": True, "role_direction": "tangent"}):
        with pytest.raises(serve.RequestError):
            _role_resolve(xy, [1, 2], variant)


def _joined_run(variant):
    """Three named in line; the robot stands at its place in turn for 1.5 s, then the back person steps back past it."""
    pipeline, sessions, obs = mock_pipeline(), serve.Sessions(), floor_observation()
    xy = [(1.0, 8.0), (1.0, 7.2), (1.0, 6.4)]
    session = _line_session(pipeline, sessions, obs, xy, [1, 2, 3])
    request = {**body_for(obs), "session": session, "tracks": _tracks_at(xy), "tracks_at_answer": _tracks_at(xy),
               "variant": variant}
    got = serve.resolve(pipeline, sessions, {**request, "t": 0.5})
    assert got["in_turn"] and got["last"] == 9
    at = {"x": got["pose_world"][0], "y": got["pose_world"][1], "yaw": got["pose_world"][2]}
    for t in (1.0, 1.5, 2.0):
        held = serve.resolve(pipeline, sessions, {**request, "t": t, "robot": at})
    stepped_back = _tracks_at([(1.0, 8.0), (1.0, 7.2), (1.0, got["pose_world"][1] - 1.0)])
    return held, serve.resolve(pipeline, sessions, {**request, "t": 2.5, "robot": at, "tracks": stepped_back})


def test_a_joined_role_belief_leaves_out_those_behind_the_robot():
    held, after = _joined_run({"role_belief": True, "role_direction": "path", "role_joined": True})
    assert held["joined"] and after["joined"]
    assert after["behind"] == [9] and after["last"] == 8
    assert 9 not in after["exempt"] and 9 not in after["turn_joined"]


def test_a_role_belief_without_the_joined_rule_still_counts_those_behind():
    held, after = _joined_run({"role_belief": True, "role_direction": "path"})
    assert not held["joined"] and not after["joined"] and "behind" not in after
    assert 9 in after["roster"] or 9 in after["exempt"]


def _oracle_resolve(ahead_by_t, variant):
    """Three named in line (tracks 7, 8, 9); one /resolve per (t, the oracle's ids ahead of the robot)."""
    pipeline, sessions, obs = mock_pipeline(), serve.Sessions(), floor_observation()
    xy = [(1.0, 8.0), (1.0, 7.2), (1.0, 6.4)]
    session = _line_session(pipeline, sessions, obs, xy, [1, 2, 3])
    request = {**body_for(obs), "session": session, "tracks": _tracks_at(xy), "tracks_at_answer": _tracks_at(xy),
               "variant": variant}
    return [serve.resolve(pipeline, sessions, {**request, "t": t, **({} if ahead is None else {"oracle_ahead": ahead})})
            for t, ahead in ahead_by_t]


def test_an_oracle_roster_replaces_the_role_belief_s_roster_at_every_resolve():
    variant = {"role_belief": True, "role_direction": "path", "role_joined": True, "oracle_roster": True}
    first, second = _oracle_resolve([(0.5, [7, 8]), (1.0, [7])], variant)
    assert first["roster"] == [7, 8] and first["last"] == 8 and 9 in first["exempt"]
    assert second["roster"] == [7] and second["last"] == 7
    assert "roster" in first["oracle"]
    plain = _oracle_resolve([(0.5, None)], {"role_belief": True, "role_direction": "path", "role_joined": True})[0]
    assert plain["roster"] == [7, 8, 9] and plain["last"] == 9 and "roster" not in plain["oracle"]


def test_an_oracle_roster_needs_a_role_belief_and_ids_at_every_resolve():
    with pytest.raises(serve.RequestError):
        _oracle_resolve([(0.5, [7, 8])], {"oracle_roster": True})
    with pytest.raises(serve.RequestError):
        _oracle_resolve([(0.5, [7, 8]), (1.0, None)], {"role_belief": True, "oracle_roster": True})


def test_the_joined_rule_needs_the_role_belief_on_the_path():
    xy = [(1.0, 8.0), (1.0, 7.2)]
    for variant in ({"role_joined": True}, {"role_belief": True, "role_joined": True}):
        with pytest.raises(serve.RequestError):
            _role_resolve(xy, [1, 2], variant)


def test_the_role_decision_options_need_a_role_belief():
    pipeline, sessions, obs = mock_pipeline(), serve.Sessions(), floor_observation()
    xy = [(1.0, 8.0), (1.0, 7.2)]
    for variant in ({"role_decision": True}, {"static_win_s": 5.0}):
        session = _line_session(pipeline, sessions, obs, xy, [1, 2])
        with pytest.raises(serve.RequestError):
            serve.resolve(pipeline, sessions, {**body_for(obs), "session": session, "t": 0.5,
                                               "tracks": _tracks_at(xy), "tracks_at_answer": _tracks_at(xy),
                                               "variant": variant})


def test_a_role_belief_hears_the_wait_before_the_first_resolve():
    import math
    pipeline, obs = mock_pipeline(), floor_observation()
    xy = [(1.0, 8.0), (1.0, 7.2), (1.0, 6.1)]
    now = [{"id": 7, "xy": [5.0, 8.0], "yaw": 0.0}, {"id": 8, "xy": [1.0, 8.0], "yaw": math.pi / 2},
           {"id": 9, "xy": [1.0, 6.1], "yaw": math.pi / 2}]
    history = [{**f, "t": f["t"]} for f in _wait_history()]
    history += [{"t": 10.0 + 0.5 * k, "robot": [3.0, 3.0], "tracks": now} for k in range(12)]   # still to 15.5 s
    got = {}
    for name, h in (("without", None), ("with", history)):
        sessions = serve.Sessions()
        session = _line_session(pipeline, sessions, obs, xy, [1, 2, 3])
        request = {**body_for(obs), "session": session, "t": 16.0, "tracks": now, "tracks_at_answer": _tracks_at(xy),
                   "variant": {"role_belief": True}}
        if h is not None:
            request["history"] = h
        got[name] = serve.resolve(pipeline, sessions, request)
    assert got["with"]["marginals"]["9"] < got["without"]["marginals"]["9"]      # stayed put while the line moved up
    assert got["with"]["marginals"]["7"] < 0.5                                  # walked off after being served


def test_an_oracle_chain_takes_everyone_within_reach_of_the_one_nearest_the_target():
    pipeline, sessions, obs = mock_pipeline(), serve.Sessions(), floor_observation()
    xy = ASIDE_XY + [(3.5, 5.0)]                           # and someone well away
    session = _line_session(pipeline, sessions, obs, xy, [1, 2])
    got = serve.resolve(pipeline, sessions, {**body_for(obs), "session": session, "t": 3.0, "tracks": _tracks_at(xy),
                                             "tracks_at_answer": _tracks_at(xy), "oracle": {"chain_m": 1.0},
                                             "variant": V2_VARIANT})
    assert got["chain"] == [7, 8, 9] and got["oracle"] == ["chain"]
    assert got["last"] == 9


def test_session_always_stores_an_answer_that_left_people_out_for_a_geometry_arm(monkeypatch):
    pipeline, obs = mock_pipeline(), floor_observation()
    real = pipeline.infer_goal

    def standard_only(*args, **kwargs):                    # as when the context call says people do not matter
        result = real(*args, **kwargs)
        return result.model_copy(update={"source": "standard", "people": None, "constraints": None})
    monkeypatch.setattr(pipeline, "infer_goal", standard_only)
    plain = serve.infer(pipeline, body_for(obs), serve.Sessions())
    sessions = serve.Sessions()
    kept = serve.infer(pipeline, {**body_for(obs), "session_always": True}, sessions)
    assert plain["source"] == "standard" and plain["session"] is None
    assert kept["source"] == "standard" and isinstance(kept["session"], str)
    xy = ASIDE_XY
    got = serve.resolve(pipeline, sessions, {**body_for(obs), "session": kept["session"], "t": 3.0,
                                             "tracks": _tracks_at(xy), "tracks_at_answer": _tracks_at(xy),
                                             "oracle": {"chain_m": 1.0}, "variant": V2_VARIANT})
    assert got["status"] == "ok" and got["oracle"] == ["chain"] and got["chain"]


def test_plan_gives_the_target_the_full_answer_will_use():
    pipeline = mock_pipeline()
    obs = floor_observation()
    early = serve.plan_target(pipeline, body_for(obs))
    full = serve.infer(pipeline, body_for(obs))
    assert early["status"] == "ok" and early["target_world"] == pytest.approx(full["target_world"])
    assert set(early["timings_s"]) == {"plan"}
    handler = serve.make_handler(pipeline)
    code, got = call(handler, "POST", "/plan", body_for(obs))
    assert code == 200 and got["target_world"] == pytest.approx(full["target_world"])

def test_people_given_with_the_request_replace_the_people_call():
    import math
    obs = floor_observation()                                       # the robot at (1, 2) facing +y
    given = [{"xy": [1.0, 5.0], "yaw": math.pi / 2}, {"xy": [1.6, 6.0], "yaw": None}, {"xy": [1.0, -3.0], "yaw": 0.0}]
    answer = serve.infer(mock_pipeline(), {**body_for(obs), "people": given})
    stages = [stage for stage, _ in answer["calls"]]
    assert "people" not in stages and "constraints" in stages
    found = answer["inference"]["people"]
    assert sorted(tuple(p["xy"]) for p in found) == [(1.0, 5.0), (1.6, 6.0)]     # the one behind the robot is not shown
    assert all(p["label"] is None for p in found)

def test_a_broken_people_list_is_a_request_error():
    with pytest.raises(serve.RequestError, match="people"):
        serve.infer(mock_pipeline(), {**body_for(floor_observation()), "people": [{"xy": "here"}]})


def test_threshold_options_set_the_joining_gate_and_refuse_unknown_keys():
    import pytest as _pytest
    from sgi.config import ConfigError, load_config
    from sgi.serve import with_thresholds
    cfg = load_config()
    assert cfg.thresholds.join_facing_deg is None and with_thresholds(cfg, []) is cfg
    gated = with_thresholds(cfg, ["join_facing_deg=90", "join_spacing_ratio=1.5"])
    assert (gated.thresholds.join_facing_deg, gated.thresholds.join_spacing_ratio) == (90.0, 1.5)
    with _pytest.raises(ConfigError):
        with_thresholds(cfg, ["no_such_key=1"])
