import dataclasses
import itertools
import math
import pytest
from sgi.config import ModelEntry, Thresholds
from sgi.imaging import draw_target_marker, encode_png
from sgi.prompts import Example, build_goal_request
from sgi.schemas import goal_to_wire
from sgi.stages.goal import DirectMarkingInferrer
from sgi.synth.render import Box
from sgi.types import ContextCheck, GoalMark, GoalReasoning, GroundedTarget, Point, Subtask
from sgi.vlm.base import ImagePart, TextPart, TransientError, VLMClient
from sgi.vlm.cache import Cache
from sgi.vlm.mock import MockBackend
from tests.helpers import floor_observation, rendered_observation

REQUEST = "take me to the info desk"
SUBTASK = Subtask(type="navigate", description="Go to the info desk", target="info desk")
TARGET = GroundedTarget(point=Point(x=500, y=450), pixel=(320.0, 216.0), world=(1.0, 6.0, 0.0))
FLOOR = Point(x=400, y=700)                  # pixel (256.0, 336.0); lifts to (0.72597, 4.28963, 0.002) on the bare floor
SKY = Point(x=500, y=20)                     # above the horizon: no depth
WHY = GoalReasoning(people="a group stands at the desk", expectations="room to finish", placement="to their left")

def infer(mark, obs=None, target=TARGET, mode="face_target", check=None, point_format="xy_1000", max_side=None,
          answer=None, cache=None, trial=0, strategy="zero_shot", examples=()):
    def scripted(parts, json_schema):        # the mark in the wire form of the query image actually sent
        image = [p for p in parts if isinstance(p, ImagePart)][-1]
        return goal_to_wire(mark, strategy, point_format, (image.width, image.height), mode)
    backend = MockBackend(scripted if answer is None else lambda parts, json_schema: answer)
    entry = ModelEntry(provider="mock", model="mock", point_format=point_format)
    client = VLMClient(entry, backend, cache, sleep=lambda s: None, clock=itertools.count(0, 60).__next__)   # 60 s per read
    inferrer = DirectMarkingInferrer(client, strategy, mode, examples, Thresholds(), max_image_side=max_side, trial=trial)
    return inferrer.infer(obs or floor_observation(), REQUEST, SUBTASK, target, check), backend

@pytest.mark.parametrize("point_format, max_side, size", [("xy_1000", None, (640, 480)), ("xy_pixel", 320, (320, 240))])
def test_floor_point_gives_the_lifted_pose_facing_the_target(point_format, max_side, size):
    r, backend = infer(GoalMark(goal_point=FLOOR), point_format=point_format, max_side=max_side)
    assert r.status == "ok" and r.source == "direct_marking" and r.target == TARGET and r.goal_mark.goal_point == FLOOR
    assert r.pose_world == pytest.approx((0.72597, 4.28963, 1.41193), abs=1e-4)
    assert r.pose_robot == pytest.approx((2.28963, 0.27403, -0.15887), abs=1e-4)
    assert r.goal_pixel == pytest.approx((256.0, 336.0)) and not any(r.flags.model_dump().values())
    assert [(p.width, p.height) for p in backend.calls[0][1] if isinstance(p, ImagePart)] == [size]
    assert [c.stage for c in r.calls] == ["goal"] and set(r.timings_s) == {"goal", "geometry"}

@pytest.mark.parametrize("obs, point, xy", [
    (rendered_observation(boxes=[Box(cx=2.0, cy=4.5, yaw=0.0, sx=0.6, sy=0.6, sz=0.5)]), Point(x=861, y=438), (2.0, 4.5)),
    (dataclasses.replace(floor_observation(), depth=floor_observation().depth * 1.2), FLOOR, (0.67116, 4.70756))])
def test_point_off_the_floor_is_flagged_and_still_gives_a_pose(obs, point, xy):     # a box top (z = 0.5), and z = -0.24
    r, _ = infer(GoalMark(goal_point=point), obs=obs)
    assert r.status == "ok" and r.flags.marked_point_off_floor and r.pose_world[:2] == pytest.approx(xy, abs=0.01)

@pytest.mark.parametrize("mark, answer, status", [
    (GoalMark(goal_point=SKY), None, "lift_failed"),
    (None, "not json", "parse_failed"), (None, TransientError("down"), "vlm_error")])
def test_failures_are_reported_with_the_call_and_no_pose(mark, answer, status):
    r, _ = infer(mark, answer=answer)
    assert r.status == status and r.source == "direct_marking" and r.pose_robot is None and r.pose_world is None
    assert [c.stage for c in r.calls] == ["goal"] and set(r.timings_s) == {"goal", "geometry"}
    assert r.timings_s["goal"] >= r.calls[0].latency_total_s >= 60.0   # a failed answer keeps the model's time
    assert (r.goal_mark is not None) is (status == "lift_failed")

@pytest.mark.parametrize("world, mode, look_at, flag, theta_robot", [
    ((1.0, 6.0, 0.0), "vlm_look_at", Point(x=600, y=600), None, -0.89192),          # faces the lifted look-at point
    ((1.0, 6.0, 0.0), "vlm_look_at", None, "heading_fallback", -0.15887),
    ((1.0, 6.0, 0.0), "vlm_look_at", SKY, "heading_fallback", -0.15887),
    ((0.75, 4.29, 0.0), "face_target", None, "heading_degenerate", 0.0),            # target 0.024 m from the lifted goal
    ((0.80, 4.29, 0.0), "face_target", None, None, -1.56585)])                      # 0.074 m: a bearing exists
def test_heading_modes_fallback_and_degenerate_case(world, mode, look_at, flag, theta_robot):
    target = TARGET.model_copy(update={"world": world})
    r, _ = infer(GoalMark(goal_point=FLOOR, look_at_point=look_at), target=target, mode=mode, max_side=320)
    assert r.status == "ok" and {name for name, on in r.flags.model_dump().items() if on} == ({flag} if flag else set())
    assert r.pose_robot[2] == pytest.approx(theta_robot, abs=1e-4)
    assert r.pose_world[2] == pytest.approx(theta_robot + math.pi / 2, abs=1e-4)

def test_request_is_the_built_goal_request_with_people_only_from_a_context_check():
    marked = encode_png(draw_target_marker(floor_observation().rgb, *TARGET.pixel))
    said = ContextCheck(relevant_people="three people talking beside the desk", people_matter=True)
    shown, mark = [Example(marked, "hand this over", "front table", FLOOR, WHY)], GoalMark(reasoning=WHY, goal_point=FLOOR)
    for check, strategy, examples, mode in ((None, "zero_shot", (), "face_target"), (said, "cot", (), "face_target"),
                                            (said, "few_shot_cot", shown, "vlm_look_at")):
        r, backend = infer(mark, check=check, strategy=strategy, examples=examples, mode=mode)
        sent, people = backend.calls[0][:2], check.relevant_people if check else None
        assert sent == build_goal_request(strategy, REQUEST, SUBTASK, marked, people, examples, "xy_1000", mode)
        assert any(said.relevant_people in p.text for p in sent[1] if isinstance(p, TextPart)) is (check is not None)
        assert r.status == "ok" and r.goal_mark.reasoning == (None if strategy == "zero_shot" else WHY)   # the schema sent

def test_cached_rerun_reproduces_the_goal_time_and_another_trial_misses(tmp_path):
    runs = [infer(GoalMark(goal_point=FLOOR), cache=Cache(tmp_path), trial=trial)[0] for trial in (0, 0, 1)]
    assert [r.calls[0].cached for r in runs] == [False, True, False]
    stored = runs[0].calls[0].latency_total_s                           # fake-clock time: 60 s or more
    assert stored >= 60.0 and all(stored <= r.timings_s["goal"] < stored + 60.0 for r in runs[:2])
    assert all(0.0 <= r.timings_s["geometry"] < 60.0 for r in runs)
