import json
import math
import re
import statistics
import numpy as np
import pytest
from PIL import Image
from sgi import smoke
from sgi.config import ModelEntry, Thresholds
from sgi.geometry import pixel_to_point, project
from sgi.gt.derive import oracle_target
from sgi.prompts import people_lines
from sgi.run import OraclePeople
from sgi.scene import load_scene, read_scene_file
from sgi.stages.constraints import numbered_image
from sgi.stages.people import VlmPeople, number_people
from sgi.types import Usage
from sgi.vlm.base import BackendError, BackendResult, ImagePart, TextPart, TransientError, VLMClient
from sgi.vlm.env import require_key
from sgi.vlm.factory import make_client
from sgi.vlm.mock import MockBackend, centre_responder
from tests.helpers import floor_observation, scene_dict, write_scene
from tests.test_run import placeholder, scene_dirs  # noqa: F401

# Raw answers with their keys out of schema order: the report must show the order as returned, not as parsed, and warn of it.
PLAN = ('{"subtasks": [{"target_point": {"y": 600, "x": 500}, "target": "info desk", "type": "navigate", '
        '"description": "Go to the info desk"}], "reasoning": "The desk is straight ahead."}')
GOAL = '{"goal_point": {"x": 400, "y": 800}, "analysis": {"placement": "c", "arrangement": "a", "expectations": "b"}}'
# The same two answers as their schemas ask for them: the keys of every object in the order of the schema sent.
PLAN_AS_ASKED = ('{"reasoning": "The desk is straight ahead.", "subtasks": [{"description": "Go to the info desk", '
                 '"target": "info desk", "target_point": {"x": 500, "y": 600}, "type": "navigate"}]}')
GOAL_AS_ASKED = '{"analysis": {"arrangement": "a", "expectations": "b", "placement": "c"}, "goal_point": {"x": 400, "y": 800}}'
PLAN_KEYS = "subtasks[].description, subtasks[].target, subtasks[].target_point, subtasks[].target_point.x, subtasks[].target_point.y, subtasks[].type"
WARNING = "WARNING: [{}] the answer's keys are not in the order of the schema sent: {}"

def scripted(monkeypatch, responder):                     # every client smoke builds gets this mock back end
    def fake(entry_name, cfg, backend=None):
        assert cfg.run.use_cache is False                 # a smoke check must reach the provider
        return make_client(entry_name, cfg, MockBackend(responder))
    monkeypatch.setattr(smoke, "make_client", fake)

@pytest.mark.parametrize("plan, goal, code, orders, warnings", [
    (PLAN, GOAL, 1,                                       # both answers are valid, and neither is in the order asked for
     ["[plan] key order: subtasks, subtasks[].target_point, subtasks[].target_point.y, subtasks[].target_point.x, "
      "subtasks[].target, subtasks[].type, subtasks[].description, reasoning",
      "[goal] key order: goal_point, goal_point.x, goal_point.y, analysis, analysis.placement, analysis.arrangement, "
      "analysis.expectations"],
     [WARNING.format("plan", "subtasks, reasoning in place of reasoning, subtasks") + "\n",
      WARNING.format("goal", "goal_point, analysis in place of analysis, goal_point") + "\n"]),
    (PLAN_AS_ASKED, GOAL_AS_ASKED, 0,
     [f"[plan] key order: reasoning, subtasks, {PLAN_KEYS}",
      "[goal] key order: analysis, analysis.arrangement, analysis.expectations, analysis.placement, goal_point, goal_point.x, "
      "goal_point.y"], ["", ""]),
], ids=["out of order", "as asked"])
def test_smoke_run_reports_both_calls_and_writes_the_picture(tmp_path, monkeypatch, capsys, plan, goal, code, orders, warnings):
    monkeypatch.chdir(tmp_path)
    scripted(monkeypatch, lambda parts, schema: plan if "subtasks" in schema["properties"] else goal)
    scene = write_scene(tmp_path, scene_dict(), floor_observation())
    assert smoke.main(["--model", "mock", "--scene", str(scene)]) == code
    out = capsys.readouterr().out
    for line in ("[plan] mock mock-centre, effort low, xy_1000, image 640x480", "[goal] usage: input -, output -, reasoning -",
                 *orders, "picture: runs/smoke.png"):
        assert line + "\n" in out
    for stage, order, warning in zip(("plan", "goal"), orders, warnings):      # a warning is the line under the order it is about
        assert f"{order}\n{warning}[{stage}] answer: " in out
    assert out.count("WARNING") == sum(map(bool, warnings))
    assert len(re.findall(r"^\[(?:plan|goal)\] latency \d+\.\d\d s, total \d+\.\d\d s, retries 0\+0$", out, re.M)) == 2
    answers = dict(re.findall(r"^\[(plan|goal)\] answer: (.+)$", out, re.M))
    assert json.loads(answers["plan"])["subtasks"][0]["target_point"] == {"x": 500, "y": 600}
    assert json.loads(answers["goal"]) == {"reasoning": {"people": "a", "expectations": "b", "placement": "c"},
                                           "goal_point": {"x": 400, "y": 800}, "look_at_point": None}
    pose = re.search(r"^result: ok, pose_robot \((.+)\)$", out, re.M).group(1)
    assert [float(x) for x in pose.split(", ")] == pytest.approx([1.939, 0.236, -0.276], abs=0.002)
    picture = Image.open(tmp_path / "runs" / "smoke.png")     # the target mark at (320, 288), the goal mark at (256, 384)
    assert (picture.size, picture.getpixel((320, 288)), picture.getpixel((256, 384))) == ((640, 480), (255, 0, 255), (255, 0, 0))
    assert smoke.main(["--model", "mock", "--scene", str(tmp_path / "nowhere")]) == 2     # no such folder

ASK = '{"description": "Ask the way", "target": null, "target_point": null, "type": "act"}'       # a step with nothing under its keys
FENCE = "```"

@pytest.mark.parametrize("plan, goal, warnings", [
    # In order from top to bottom, also where a key holds null, in a list of two and inside a code fence: nothing to warn of.
    (f'{FENCE}json\n{PLAN_AS_ASKED[:-2]}, {ASK}]}}\n{FENCE}', GOAL_AS_ASKED, []),
    # The answer itself in order and an object under it not: the second step of the plan, and a point of the goal answer.
    (PLAN_AS_ASKED[:-2] + ', {"type": "act", "description": "Ask the way", "target": null, "target_point": null}]}',
     GOAL_AS_ASKED.replace('{"x": 400, "y": 800}', '{"y": 800, "x": 400}'),
     [WARNING.format("plan", "under subtasks[], type, description, target, target_point in place of description, target, "
                             "target_point, type"),
      WARNING.format("goal", "under goal_point, y, x in place of x, y")]),
    # One line for a call, whatever else is out of order below the first place found.
    (PLAN_AS_ASKED, '{"goal_point": {"y": 800, "x": 400}, "analysis": {"placement": "c", "arrangement": "a", "expectations": "b"}}',
     [WARNING.format("goal", "goal_point, analysis in place of analysis, goal_point")]),
], ids=["as asked", "an object under the answer", "the answer and an object under it"])
def test_the_order_is_held_against_the_schema_in_every_object_of_an_answer(tmp_path, monkeypatch, capsys, plan, goal, warnings):
    monkeypatch.chdir(tmp_path)
    scripted(monkeypatch, lambda parts, schema: plan if "subtasks" in schema["properties"] else goal)
    scene = write_scene(tmp_path, scene_dict(), floor_observation())
    assert smoke.main(["--model", "mock", "--scene", str(scene)]) == (1 if warnings else 0)
    out = capsys.readouterr().out
    assert [line for line in out.splitlines() if line.startswith("WARNING")] == warnings and "result: ok, pose_robot (" in out

def test_an_answer_that_is_no_answer_to_its_schema_gets_its_status_and_no_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)                           # the old names of the goal answer, in the order they were returned in
    old = '{"goal_point": {"x": 400, "y": 800}, "reasoning": {"placement": "c", "people": "a", "expectations": "b"}}'
    scripted(monkeypatch, lambda parts, schema: PLAN_AS_ASKED if "subtasks" in schema["properties"] else old)
    scene = write_scene(tmp_path, scene_dict(), floor_observation())
    assert smoke.main(["--model", "mock", "--scene", str(scene)]) == 1
    out = capsys.readouterr().out
    assert "[goal] key order: goal_point, goal_point.x, goal_point.y, reasoning, " in out and "[goal] answer: none (" in out
    assert "result: parse_failed, pose_robot none\n" in out and "WARNING" not in out

@pytest.mark.parametrize("goal, error", [                 # GOAL is a whole answer to the schema, with its keys out of order
    (f"Here is the answer: {GOAL} I hope it helps.", "invalid JSON: Expecting value: line 1 column 1 (char 0)"),
    (BackendResult(text=GOAL, finish="truncated", usage=Usage()), "truncated"),
], ids=["words round it", "truncated"])
def test_an_answer_the_client_rejected_draws_no_warning_whatever_its_text_holds(tmp_path, monkeypatch, capsys, goal, error):
    monkeypatch.chdir(tmp_path)                           # the order is a finding about an answer that was read, and this one was not
    scripted(monkeypatch, lambda parts, schema: PLAN_AS_ASKED if "subtasks" in schema["properties"] else goal)
    scene = write_scene(tmp_path, scene_dict(), floor_observation())
    assert smoke.main(["--model", "mock", "--scene", str(scene)]) == 1
    out = capsys.readouterr().out
    assert ("[goal] key order: goal_point, goal_point.x, goal_point.y, analysis, analysis.placement, analysis.arrangement, "
            f"analysis.expectations\n[goal] answer: none ({error})\nresult: parse_failed, pose_robot none\n") in out
    assert "WARNING" not in out

LOCATED = '{"reasoning": "The desk is the box ahead.", "target_point": {"x": 500, "y": 600}}'      # as the cot schema asks for it
UNMARKED = PLAN_AS_ASKED.replace('{"x": 500, "y": 600}', "null")       # a plan that leaves its target to the locate call

@pytest.mark.parametrize("located, warnings", [
    ('{"target_point": {"x": 500, "y": 600}, "reasoning": "The desk is the box ahead."}',
     [WARNING.format("locate", "target_point, reasoning in place of reasoning, target_point")]),
    (LOCATED, []),
], ids=["out of order", "as asked"])
def test_a_locate_answer_out_of_order_is_warned_of_like_any_other(tmp_path, monkeypatch, capsys, located, warnings):
    monkeypatch.chdir(tmp_path)
    answers = {"subtasks": UNMARKED, "target_point": located, "goal_point": GOAL_AS_ASKED}       # by a key of the schema sent
    scripted(monkeypatch, lambda parts, schema: next(text for key, text in answers.items() if key in schema["properties"]))
    scene = write_scene(tmp_path, scene_dict(), floor_observation())
    assert smoke.main(["--model", "mock", "--strategy", "locate=cot", "--scene", str(scene)]) == (1 if warnings else 0)
    out = capsys.readouterr().out
    assert [line for line in out.splitlines() if line.startswith("WARNING")] == warnings
    order = "[locate] key order: " + ("target_point, target_point.x, target_point.y, reasoning" if warnings else
                                      "reasoning, target_point, target_point.x, target_point.y")
    assert f"{order}\n{''.join(line + chr(10) for line in warnings)}[locate] answer: " in out      # under the order it is about
    assert [stage for stage in re.findall(r"^\[(\w+)\] answer: ", out, re.M)] == ["plan", "locate", "goal"]
    assert "result: ok, pose_robot (" in out

def boxes(root, scene_id, *anchors, person=None):         # scene_dict() with one labelled box per anchor
    d = scene_dict(scene_id=scene_id)
    gt = d["ground_truth"]
    gt["targets"] = [dict(gt["targets"][0], id=f"t{i}", label=f"box {i}", anchor=list(a)) for i, a in enumerate(anchors)]
    if person:
        gt["people"] = [{"id": "p0", "x": person[0], "y": person[1], "yaw": 0.0}]
        gt["targets"].append({"id": "tp", "kind": "person", "label": "person in red", "anchor": [*person, 1.3], "person_id": "p0"})
    return write_scene(root, d, floor_observation())

def spread(root):                                         # five scenes: seven boxes and a person; five boxes can be pointed at
    boxes(root, "a", (1.0, 6.0, 1.0))                     # projects to (320.0, 70.7)
    boxes(root, "b", (1.1, 5.0, 0.3))                     # (338.9, 218.9), 28.3 px from the image centre: the nearest to it
    boxes(root, "c", (2.0, 5.0, 0.5), (1.0, 0.0, 1.0))    # (513.1, 182.1); the second box is behind the camera
    boxes(root, "d", (0.0, 7.0, 1.0), person=(1.5, 5.0))  # (198.9, 64.0); a person is not a labelled box
    boxes(root, "e", (1.5, 4.0, 0.2), (9.0, 3.0, 1.0))    # (456.3, 328.3); the second box is outside the image

@pytest.mark.parametrize("answer, medians", [
    (None, {"xy_1000": 185.46, "xy_pixel": 185.46}),      # the mock entry answers the image centre in both formats
    ({"target_point": {"x": 400, "y": 300}}, {"xy_1000": 179.08, "xy_pixel": 203.13}),    # one answer, read two ways
])
def test_calibrate_prints_one_median_error_per_point_format(tmp_path, monkeypatch, capsys, answer, medians):
    monkeypatch.chdir(tmp_path)
    if answer:
        scripted(monkeypatch, lambda parts, schema: answer)
    spread(tmp_path / "scenes")
    assert smoke.main(["--model", "mock", "--calibrate", "--scene", str(tmp_path / "scenes")]) == 0
    out = capsys.readouterr().out
    lines = re.findall(r"^(xy_\w+): median error ([\d.]+) px \((\d+) of (\d+) points answered\)$", out, re.M)
    assert {name: float(median) for name, median, _, _ in lines} == pytest.approx(medians, abs=0.06)
    assert [line[2:] for line in lines] == [("4", "4"), ("4", "4")] and not (tmp_path / "runs" / "smoke.png").exists()
    # The four boxes asked for, in folder order: the one of scene b, nearest the image centre, is the one left out.
    assert out.startswith("points: 4 of 5 place and object targets in 5 scenes, those farthest from the image centre\n")
    assert re.findall(r'^  (\w+): "box 0" at \((\d+), (\d+)\), ([\d.]+) px from the centre$', out, re.M) == [
        ("a", "320", "71", "169.3"), ("c", "513", "182", "201.6"), ("d", "199", "64", "213.6"), ("e", "456", "328", "162.4")]
    # Beside each median, what the image centre scores on the same points: the median of those four distances.
    centre = re.findall(r"^(xy_\w+): image centre ([\d.]+) px \(the median error of the image centre over the "
                        r"points answered\)$", out, re.M)
    assert centre == [("xy_1000", "185.5"), ("xy_pixel", "185.5")]
    assert [line.split(":")[0] for line in out.splitlines()[5:]] == ["xy_1000", "xy_1000", "xy_pixel", "xy_pixel"]

def test_calibrate_scores_the_image_centre_on_the_points_a_format_answered(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    requests = []                                         # four in xy_1000, then four in xy_pixel: a, c, d, e each time
    def responder(parts, schema):                         # no point in xy_1000; in xy_pixel none for the box of scene a
        requests.append(schema)
        return {"target_point": None if len(requests) <= 5 else {"x": 400, "y": 300}}
    scripted(monkeypatch, responder)
    spread(tmp_path / "scenes")
    assert smoke.main(["--model", "mock", "--calibrate", "--scene", str(tmp_path / "scenes")]) == 1
    out = capsys.readouterr().out                         # (400, 300) is 163.4, 310.1 and 63.0 px from the boxes of c, d and e
    assert out.splitlines()[5:] == [
        "xy_1000: no answer (0 of 4 points answered)",    # and so no image-centre line: there is nothing to compare
        *(f'xy_1000: {scene}: "box 0" got no point: null answer' for scene in "acde"),
        "xy_pixel: median error 163.4 px (3 of 4 points answered)",
        "xy_pixel: image centre 201.6 px (the median error of the image centre over the points answered)",
        'xy_pixel: a: "box 0" got no point: null answer']
    assert len(requests) == 8                             # the centre is 201.6, 213.6 and 162.4 px from those three boxes

def test_calibrate_says_why_a_point_got_no_answer(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    answers = {"box 0": {"target_point": None},                       # the model says the target is not in the image
               "box 1": {"target_point": {"x": 700, "y": 300}},       # a point in xy_1000; outside a 640 x 480 image in xy_pixel
               "box 2": TransientError("gemini 503: unavailable", retry_after=0.0),      # no answer, however often it is asked
               "box 3": {"target_point": {"x": 400, "y": 300}}}
    def responder(parts, schema):
        text = " ".join(part.text for part in parts if isinstance(part, TextPart))
        return next(answer for label, answer in answers.items() if f"Go to the {label}" in text)
    scripted(monkeypatch, responder)
    boxes(tmp_path / "scenes", "s", (1.0, 6.0, 1.0), (2.0, 5.0, 0.5), (0.0, 7.0, 1.0), (1.5, 4.0, 0.2))     # four boxes in one image
    assert smoke.main(["--model", "mock", "--calibrate", "--scene", str(tmp_path / "scenes")]) == 0          # each format has an answer
    out = capsys.readouterr().out.splitlines()[5:]        # after the count and the four points: per format, its two lines,
    assert [line for line in out if "got no point" not in line] == [      # then one line for every point without an answer
        "xy_1000: median error 173.8 px (2 of 4 points answered)",    # (448, 144) and (256, 144): 75.4 and 272.2 px off
        "xy_1000: image centre 182.0 px (the median error of the image centre over the points answered)",
        "xy_pixel: median error 63.0 px (1 of 4 points answered)",
        "xy_pixel: image centre 162.4 px (the median error of the image centre over the points answered)"]
    assert [line for line in out if "got no point" in line] == [
        'xy_1000: s: "box 0" got no point: null answer',                          # nothing wrong with the format: ask again, or accept it
        'xy_1000: s: "box 2" got no point: vlm_error (gemini 503: unavailable)',  # an outage: run the command again
        'xy_pixel: s: "box 0" got no point: null answer',
        'xy_pixel: s: "box 1" got no point: parse_failed (target_point.x: Input should be less than or equal to 640)',
        'xy_pixel: s: "box 2" got no point: vlm_error (gemini 503: unavailable)'] # an answer outside the image: the wrong format for this model
    assert [line.split(":")[0] for line in out] == ["xy_1000"] * 4 + ["xy_pixel"] * 5

def test_missing_key_exits_2_naming_the_variable_and_no_key(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env, key = tmp_path / ".env", "sk-test-0123456789-do-not-print"
    env.write_text(f"OPENAI_API_KEY={key}\n")
    def keyless(entry_name, cfg, backend=None):           # what make_client does for a gemini entry, on this .env
        assert cfg.models[entry_name].provider == "gemini"
        require_key("GEMINI_API_KEY", env_path=env)
    monkeypatch.setattr(smoke, "make_client", keyless)
    scene = write_scene(tmp_path, scene_dict(), floor_observation())
    for extra in ([], ["--calibrate"]):
        assert smoke.main(["--model", "gemini_fast", "--scene", str(scene), *extra]) == 2
        out, err = capsys.readouterr()
        assert "GEMINI_API_KEY" in err and key not in out + err and "Traceback" not in err
        assert (out, len(err.splitlines())) == ("", 1)    # one line and nothing else, as sgi.run reports it
    assert smoke.main(["--model", "gpt-9", "--scene", str(scene)]) == 2 and "gpt-9" in capsys.readouterr().err

def toured(root, frames=3):                               # scene_dict() with a tour whose frames repeat the observation's view
    obs = floor_observation()
    pose = obs.camera.T_world_camera.tolist()
    tour = [{"rgb": f"tour/{k:03d}.png", "T_world_camera": pose} for k in range(frames)]
    folder = write_scene(root, scene_dict(tour=tour), obs)
    (folder / "tour").mkdir()
    for entry in tour:
        Image.fromarray(obs.rgb).save(folder / entry["rgb"])
    return folder

def test_a_scene_with_a_tour_adds_one_plan_call_on_a_single_frame(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    requests = []                                         # (parts, schema) of every request, in order
    def recording(parts, schema):
        requests.append((parts, schema))
        return centre_responder(parts, schema)
    scripted(monkeypatch, recording)
    assert smoke.main(["--model", "mock", "--scene", str(toured(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert out.endswith("picture: runs/smoke.png\n[tour] plan on tour frame 0 alone: ok, tour_frame 0\n")
    assert len(requests) == 3                             # the plan and the goal of the smoke run, then the plan on the tour
    parts, schema = requests[2]
    assert [type(part) for part in parts] == [TextPart, TextPart, ImagePart] and parts[1].text == "Frame 0"
    step = schema["$defs"]["SubtaskWire"]["properties"]   # one frame shown: its index is a "const", not an "enum"
    assert "target_point" not in step and step["tour_frame"]["anyOf"][0] == {"const": 0, "type": "integer"}
    assert len(re.findall(r"^\[tour\] ", out, re.M)) == 1 and len(re.findall(r"^\[plan\] answer: ", out, re.M)) == 1

@pytest.mark.parametrize("answer, line, code", [
    ({"reasoning": "r", "subtasks": [{"description": "d", "target": "t", "tour_frame": None, "type": "navigate"},
                                     {"description": "d", "target": None, "tour_frame": None, "type": "act"},
                                     {"description": "d", "target": "t", "tour_frame": 0, "type": "navigate"}]},
     "ok, tour_frame none, 0", 0),                        # one entry per navigate subtask; null: not in the frame shown
    ({"reasoning": "r", "subtasks": [{"description": "d", "target": None, "tour_frame": None, "type": "act"}]},
     "ok, no navigate subtask", 0),
    ({"reasoning": "r", "subtasks": [{"description": "d", "target": "t", "tour_frame": 1, "type": "navigate"}]},
     "parse_failed (subtasks.0.tour_frame: Input should be 0)", 1),     # frame 1 was not shown
    ({"reasoning": "r", "subtasks": [{"type": "navigate", "description": "d", "target": "t", "tour_frame": 0}]},
     "ok, tour_frame 0\n" + WARNING.format("tour", "under subtasks[], type, description, target, tour_frame in place of "
                                                   "description, target, tour_frame, type"), 1),     # a good plan, written out of order
    ('Here it is: {"reasoning": "r", "subtasks": [{"type": "navigate", "description": "d", "target": "t", "tour_frame": 0}]} Done.',
     "parse_failed (invalid JSON: Expecting value: line 1 column 1 (char 0))", 1),      # the same plan with words round it: not read
])
def test_the_tour_line_gives_the_outcome_of_the_tour_plan(tmp_path, monkeypatch, capsys, answer, line, code):
    monkeypatch.chdir(tmp_path)
    on_tour = lambda schema: "tour_frame" in schema["$defs"]["SubtaskWire"]["properties"]
    scripted(monkeypatch, lambda parts, schema: answer if "subtasks" in schema["properties"] and on_tour(schema)
             else centre_responder(parts, schema))
    assert smoke.main(["--model", "mock", "--scene", str(toured(tmp_path))]) == code
    out = capsys.readouterr().out
    assert "result: ok, pose_robot (" in out and out.endswith(f"[tour] plan on tour frame 0 alone: {line}\n")
    assert out.count("WARNING") == line.count("WARNING")  # none for a plan that failed: its status says what is wrong

def test_a_refused_request_exits_1_with_one_line_naming_the_call(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    refusal = BackendError("openai 400: Invalid schema for response_format 'answer':\n'const' is not permitted")
    on_tour = lambda schema: "tour_frame" in json.dumps(schema)
    scripted(monkeypatch, lambda parts, schema: refusal if on_tour(schema) else centre_responder(parts, schema))
    scene = str(toured(tmp_path))
    assert smoke.main(["--model", "mock", "--scene", scene]) == 1
    out, err = capsys.readouterr()                        # the smoke run is reported before the tour call is made
    assert out.endswith("picture: runs/smoke.png\n") and "result: ok" in out and "[tour]" not in out
    assert err == ("error: the plan call on tour frame 0 alone was refused: openai 400: Invalid schema for "
                   "response_format 'answer': 'const' is not permitted\n")
    scripted(monkeypatch, lambda parts, schema: BackendError("gemini 404: no such model"))
    for extra in ([], ["--calibrate"]):                   # refused at the first call: nothing to report
        assert smoke.main(["--model", "mock", "--scene", scene, *extra]) == 1
        assert capsys.readouterr() == ("", "error: gemini 404: no such model\n")

def test_on_the_generated_scenes_the_smoke_run_makes_three_calls_and_calibration_eight(placeholder, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    texts = []                                            # the text of every request, in order
    def recording(parts, schema):
        texts.append(" ".join(part.text for part in parts if isinstance(part, TextPart)))
        return centre_responder(parts, schema)
    scripted(monkeypatch, recording)
    scenes = placeholder / "eval"                         # what the default, data/placeholder/eval, holds
    first = read_scene_file(scene_dirs(scenes)[0])
    assert first.tour and smoke.main(["--model", "mock", "--scene", str(scenes)]) == 0
    out = capsys.readouterr().out                         # the plan, the goal and the plan on the tour's first frame
    assert out.endswith("picture: runs/smoke.png\n[tour] plan on tour frame 0 alone: ok, tour_frame 0\n")
    assert len(texts) == 3 and all(f"Request: {first.episodes[0].request}\n" in text for text in texts)
    del texts[:]
    assert smoke.main(["--model", "mock", "--calibrate", "--scene", str(scenes)]) == 0
    out = capsys.readouterr().out
    pool = []                                             # per place and object: (px from the image centre, px right of it, label)
    for inputs, truth in map(load_scene, scene_dirs(scenes)):
        camera = inputs.observation.camera
        for target in truth.targets.values():
            if target.kind != "person":
                right, down = (a - b for a, b in zip(project(camera, target.anchor), (camera.width / 2, camera.height / 2)))
                pool.append((math.hypot(right, down), right, target.label))
    asked = [point for point in pool if point[0] >= sorted(pool)[-4][0]]          # the four farthest, in folder order
    count = f"points: 4 of {len(pool)} place and object targets in {len(scene_dirs(scenes))} scenes, those farthest"
    assert len(asked) == 4 and out.startswith(count)
    assert len(texts) == 8 and all(f"Go to the {label}" in text for (_, _, label), text in zip(asked * 2, texts, strict=True))
    # What makes a format's two numbers worth comparing: the image centre is far from every point, on both sides of it.
    assert min(off for off, _, _ in asked) > 100 and {right > 0 for _, right, _ in asked} == {True, False}
    centre = f"{statistics.median(off for off, _, _ in asked):.1f}"                # the mock answers the image centre
    lines = re.findall(r"^(xy_\w+): (median error|image centre) ([\d.]+) px \((4 of 4 points answered|the median)", out, re.M)
    assert lines == [(name, what, centre, note) for name in ("xy_1000", "xy_pixel")
                     for what, note in (("median error", "4 of 4 points answered"), ("image centre", "the median"))]

# ------------------------------------------------------------------- --goal constraints: the two calls of the constraint method
TH = Thresholds()
STATED = [{"because": "They were there first.", "kind": "come_after", "people": [1, 2, 3], "range_m": None, "target": False},
          {"because": "It is handed over.", "kind": "within_distance", "people": [], "range_m": {"at_least": 0.5, "at_most": 1.5}, "target": True},
          {"because": "Someone watches.", "kind": "visible_to", "people": [99], "range_m": None, "target": False}]
REASONED = {"analysis": {"arrangement": "a", "expectations": "b", "placement": "c"}, "constraints": STATED}     # as the cot schema asks
PEOPLE_KEYS = ("people, people[].appearance, people[].body_point, people[].body_point.x, people[].body_point.y, "
               "people[].faced_point, people[].facing")
CONSTRAINT_KEYS = ("analysis, analysis.arrangement, analysis.expectations, analysis.placement, constraints, constraints[].because, "
                   "constraints[].kind, constraints[].people, constraints[].range_m, constraints[].target, "
                   "constraints[].range_m.at_least, constraints[].range_m.at_most")       # each key once, where it is first written

@pytest.fixture(scope="module")
def peopled(placeholder):                                 # a generated scene with three people or more in view: its folder and what it holds
    for folder in scene_dirs(placeholder / "eval"):
        inputs, truth = load_scene(folder)
        found = OraclePeople(truth, TH)(inputs.observation).found
        if len(found) >= 3:
            camera = inputs.observation.camera
            points = [pixel_to_point(*person.pixel, camera.width, camera.height) for person in found]
            bodies = {"people": [{"appearance": f"body {n}", "body_point": {"x": x, "y": y}, "faced_point": None, "facing": "left"}
                                 for n, (x, y) in enumerate(points)]}   # a model that points at each of them
            return folder, inputs, truth, bodies

def method(monkeypatch, bodies, constraints=REASONED):   # every client answers the two requests of the method, and logs them
    seen = []
    def responder(parts, schema):
        seen.append((next(key for key in ("people", "constraints", "subtasks", "goal_point") if key in schema["properties"]), parts))
        return bodies if seen[-1][0] == "people" else constraints
    scripted(monkeypatch, responder)
    return seen

def test_goal_constraints_makes_the_two_calls_of_the_method_prints_the_list_and_the_answer_and_writes_the_numbered_image(
        peopled, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    folder, inputs, truth, bodies = peopled
    seen = method(monkeypatch, bodies)
    assert smoke.main(["--model", "mock", "--goal", "constraints", "--scene", str(folder)]) == 0
    out = capsys.readouterr().out
    assert [stage for stage, _ in seen] == ["people", "constraints"]  # two requests: no plan, no context check, no goal mark
    observation, episode = inputs.observation, inputs.episodes[0]
    target = oracle_target(observation, truth.targets[truth.episodes[episode.episode_id].target_ids[0]], TH)     # the target, from ground truth
    asking = VLMClient(ModelEntry(provider="mock", model="mock-centre"), MockBackend(lambda parts, schema: bodies))
    people, _ = number_people(VlmPeople(asking, "zero_shot", TH)(observation).found, target, observation, TH, TH.max_people)
    assert len(people) == len(bodies["people"]) >= 3
    listed = people_lines(people, target, observation, TH).split("\n")
    for line in ("[people] mock mock-centre, effort low, xy_1000, image 640x480", f"[people] key order: {PEOPLE_KEYS}",
                 f"people: {len(people)} numbered", *(f"  {line}" for line in listed),
                 "[constraints] mock mock-centre, effort low, xy_1000, image 640x480", f"[constraints] key order: {CONSTRAINT_KEYS}",
                 "constraints: 3 stated", "  1. used: come_after, people [1, 2, 3]: They were there first.",
                 "  2. used: within_distance, people [], the target, 0.5 to 1.5 m: It is handed over.",
                 "  3. unusable (none of its numbers is in the list of people): visible_to, people [99]: Someone watches.",
                 "picture: runs/smoke_numbered.png"):
        assert line + "\n" in out, line
    answers = dict(re.findall(r"^\[(people|constraints)\] answer: (.+)$", out, re.M))       # each answer as it was written
    assert json.loads(answers["people"]) == bodies and json.loads(answers["constraints"]) == REASONED
    order = [out.index(mark) for mark in ("[people] answer: ", "people: ", "[constraints] mock", "[constraints] answer: ", "constraints: 3",
                                          "result: ok, pose_robot (", "picture: ")]
    assert order == sorted(order) and out.count("WARNING") == 0 and "[plan]" not in out and "[tour]" not in out
    text = "\n".join(part.text for part in seen[1][1] if isinstance(part, TextPart))        # what the model was shown: the same list,
    assert all(line in text.splitlines() for line in listed) and f"Request: {episode.request}" in text
    shown = [part for part in seen[1][1] if isinstance(part, ImagePart)]                     # and the image that was written
    picture = np.asarray(Image.open(tmp_path / "runs" / "smoke_numbered.png").convert("RGB"))
    assert np.array_equal(picture, numbered_image(observation, target, people)) and not np.array_equal(picture, observation.rgb)
    assert len(shown) == 1 and np.array_equal(np.asarray(Image.open(__import__("io").BytesIO(shown[0].png)).convert("RGB")), picture)
    assert not (tmp_path / "runs" / "smoke.png").exists()

def test_goal_constraints_warns_of_an_answer_out_of_order_and_gives_a_failed_call_its_status(peopled, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    folder, _, _, bodies = peopled
    swapped = json.dumps({"constraints": STATED, "analysis": REASONED["analysis"]})         # a valid answer, its reasoning written last
    method(monkeypatch, bodies, swapped)
    assert smoke.main(["--model", "mock", "--goal", "constraints", "--scene", str(folder)]) == 1
    out = capsys.readouterr().out
    assert WARNING.format("constraints", "constraints, analysis in place of analysis, constraints") + "\n[constraints] answer: " in out
    assert "result: ok, pose_robot (" in out and out.count("WARNING") == 1
    method(monkeypatch, bodies, "not json")
    assert smoke.main(["--model", "mock", "--goal", "constraints", "--scene", str(folder)]) == 1
    out = capsys.readouterr().out                         # the people are known, and the image the model was shown is written
    assert "[constraints] answer: none (invalid JSON: " in out and "result: parse_failed, pose_robot none\n" in out
    assert not re.search(r"^constraints: ", out, re.M) and f"people: {len(bodies['people'])} numbered\n" in out
    assert out.endswith("picture: runs/smoke_numbered.png\n")
    (tmp_path / "runs" / "smoke_numbered.png").unlink()
    seen = method(monkeypatch, TransientError("503", retry_after=0.0))
    assert smoke.main(["--model", "mock", "--goal", "constraints", "--scene", str(folder)]) == 1
    out = capsys.readouterr().out                         # no answer about the people: nothing to number, nothing more to ask
    assert "[people] answer: none (503)" in out and out.endswith("result: vlm_error, pose_robot none\n") and "people: " not in out
    assert {stage for stage, _ in seen} == {"people"} and not (tmp_path / "runs" / "smoke_numbered.png").exists()

def test_goal_constraints_takes_the_strategies_and_the_entries_of_its_two_stages_and_builds_no_other_client(peopled, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    folder, _, _, bodies = peopled
    built = []
    def recording(entry_name, cfg, backend=None):         # each client built, with the goal the run is set to
        built.append((entry_name, cfg.run.goal, cfg.run.people, cfg.run.constraints))
        def answer(parts, schema):                        # with the reasoning where the schema asks for it, and only there
            asked = schema["properties"]
            reasoned = {"analysis": "three of them" if "people" in asked else REASONED["analysis"]} if "analysis" in asked else {}
            return {**reasoned, **(bodies if "people" in asked else {"constraints": STATED})}
        return make_client(entry_name, cfg, MockBackend(answer))
    monkeypatch.setattr(smoke, "make_client", recording)
    options = ["--strategy", "people=cot", "--strategy", "constraints=zero_shot"]
    assert smoke.main(["--model", "mock", "--goal", "constraints", "--scene", str(folder), *options]) == 0
    out = capsys.readouterr().out
    assert "[people] key order: analysis, people, " in out and "[constraints] key order: constraints, constraints[].because, " in out
    assert built == [("mock", "constraints", "vlm", "vlm")]           # one entry for both stages: one client
    del built[:]
    assert smoke.main(["--model", "gemini_fast", "--stage-model", "people=mock", "--stage-model", "constraints=mock", "--goal", "constraints",
                       "--scene", str(folder)]) == 0                 # the entry of a stage that is not called needs no key
    assert [name for name, *_ in built] == ["mock"] and "result: ok" in capsys.readouterr().out

def test_goal_constraints_is_the_smoke_run_alone_and_names_a_target_that_is_not_in_view(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(smoke, "make_client", lambda *args, **kwargs: pytest.fail("a client was built"))
    scene = write_scene(tmp_path, scene_dict(), floor_observation())
    for other in ("--calibrate", "--list-models"):        # each stands in place of the smoke run: the goal would change nothing
        with pytest.raises(SystemExit) as stop:
            smoke.main(["--model", "mock", "--goal", "constraints", other, "--scene", str(scene)])
        assert stop.value.code == 2 and f"--goal constraints cannot be combined with {other}" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        smoke.main(["--model", "mock", "--goal", "oracle", "--scene", str(scene)])
    assert "invalid choice: 'oracle'" in capsys.readouterr().err
    scripted(monkeypatch, centre_responder)
    behind = scene_dict(scene_id="behind")
    behind["ground_truth"]["targets"][0]["anchor"] = [1.0, 0.0, 1.0]      # behind the camera
    hidden = write_scene(tmp_path, behind, floor_observation())
    assert smoke.main(["--model", "mock", "--goal", "constraints", "--scene", str(hidden)]) == 2
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("error: behind: episode e0: ") and "not in the image" in err and err.count("\n") == 1

def test_goal_constraints_on_the_mock_entry_runs_through_with_nobody_found(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)                           # the built-in mock points at the image centre, which is floor here
    scene = write_scene(tmp_path, scene_dict(), floor_observation())
    assert smoke.main(["--model", "mock", "--goal", "constraints", "--scene", str(scene)]) == 0
    out = capsys.readouterr().out
    assert "people: none found\n" in out and "constraints: 1 stated\n" in out and "result: ok, pose_robot (" in out
    assert "  1. unusable (none of its numbers is in the list of people): keep_out_of_shared_space, people [0], the target, 0 to 0 m: mock\n" in out
    assert (tmp_path / "runs" / "smoke_numbered.png").exists()
