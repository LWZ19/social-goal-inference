import json
import math
import re
import shutil

import pytest

from sgi import solver
from sgi.config import PROMPTS_DIR, RunConfig, Thresholds
from sgi.prompts import (Example, build_constraints_request, build_context_request, build_goal_request,
                         build_locate_request, build_people_request, build_plan_request, people_lines, stages_of)
from sgi.schemas import FACINGS
from sgi.types import DetectedPerson, GoalReasoning, Point, Subtask
from sgi.vlm.base import ImagePart, TextPart
from sgi.vlm.points import POINT_FORMATS
from tests.helpers import floor_observation
from tests.solver_layouts import HERE, TH, body_point, open_floor, target_at

IMG = ImagePart(b"query", 640, 480)
SUB = Subtask(type="navigate", description="go to the thing", target="the tall one")
QUERY = ["Request: do it", "Current subtask: go to the thing", "Target: the tall one"]     # whole lines of the text part
EXAMPLES = [Example(ImagePart(b"ex0", 640, 480), "request zero", "target zero", Point(x=250, y=750),
                    GoalReasoning(people="p0", expectations="e0", placement="s0"), Point(x=500, y=500)),
            Example(ImagePart(b"ex1", 320, 240), "request one", "target one", Point(x=500, y=500),
                    GoalReasoning(people="p1", expectations="e1", placement="s1"), None)]

def kinds(parts):
    return "".join("T" if isinstance(p, TextPart) else "I" for p in parts)

def goal_text(strategy="zero_shot", people=None, mode="face_target"):
    return build_goal_request(strategy, "do it", SUB, IMG, people, (), "xy_1000", mode)[1][0].text

def test_goal_request_emits_examples_then_the_query():
    system, parts = build_goal_request("few_shot", "do it", SUB, IMG, "PEOPLE", EXAMPLES, "xy_pixel", "face_target")
    assert system and kinds(parts) == "TITTITTTI"
    assert (parts[1], parts[4], parts[8]) == (EXAMPLES[0].image, EXAMPLES[1].image, IMG)
    assert parts[2].text.splitlines() == ["Request: request zero", "Target: target zero"] and "target one" in parts[5].text
    assert json.loads(parts[3].text) == {"goal_point": {"x": 160, "y": 360}}
    assert json.loads(parts[6].text) == {"goal_point": {"x": 160, "y": 120}}     # pixels of its own 320 x 240 image
    assert all(line in parts[7].text.splitlines() for line in QUERY) and "earlier step: PEOPLE" in parts[7].text
    assert POINT_FORMATS["xy_pixel"].instruction(640, 480) in parts[7].text      # the size of the query image
    _, parts = build_goal_request("few_shot_cot", "do it", SUB, IMG, None, EXAMPLES, "xy_1000", "vlm_look_at")
    answer = json.loads(parts[3].text)
    assert list(answer) == ["analysis", "goal_point", "look_at_point"]           # an example's answer, as it is to be written
    assert answer["analysis"] == {"arrangement": "p0", "expectations": "e0", "placement": "s0"}
    assert (answer["goal_point"], answer["look_at_point"]) == ({"x": 250, "y": 750}, {"x": 500, "y": 500})
    assert json.loads(parts[6].text)["look_at_point"] is None
    for strategy in ("zero_shot", "cot"):                                        # only the few-shot strategies show examples
        assert kinds(build_goal_request(strategy, "do it", SUB, IMG, None, EXAMPLES, "xy_1000", "face_target")[1]) == "TI"
    assert kinds(build_goal_request("few_shot", "do it", SUB, IMG, None, (), "xy_1000", "face_target")[1]) == "TI"

def test_goal_blocks_follow_strategy_heading_mode_and_relevant_people():
    assert "the point on the floor where the robot should stand" in goal_text()
    assert "earlier step: PEOPLE" in goal_text(people="PEOPLE") and "earlier step" not in goal_text(people=None)
    assert '"placement"' in goal_text("cot") and '"placement"' not in goal_text("zero_shot")
    for strategy in ("cot", "few_shot_cot"):                                     # the reasoning is asked for by its wire keys
        asked = goal_text(strategy)
        assert all(f'"{key}"' in asked for key in ("analysis", "arrangement", "expectations", "placement"))
        assert '"reasoning"' not in asked and '"people"' not in asked            # the names those two keys had before
    assert '"analysis"' not in goal_text("zero_shot") and '"arrangement"' not in goal_text("few_shot")
    assert '"look_at_point"' in goal_text(mode="vlm_look_at") and '"look_at_point"' not in goal_text("cot")

def test_context_request_describes_the_standard_goal():
    system, parts = build_context_request("zero_shot", "do it", SUB, IMG, 0.8)
    assert system and kinds(parts) == "TI" and parts[1] == IMG
    assert all(line in parts[0].text.splitlines() for line in QUERY)
    seen_side = "in front of the side of the target that the robot sees, squarely facing it."   # not the line of approach
    assert f"to stop about 0.8 m {seen_side}" in parts[0].text and '"reasoning"' not in parts[0].text
    cot = build_context_request("cot", "do it", SUB, IMG, 1.0)[1][0].text
    assert f"to stop about 1 m {seen_side}" in cot and '"reasoning"' in cot
    for text in (parts[0].text, cot):                                            # the answer is set by its wire key
        assert 'Set "stop_elsewhere" to false only when' in text and '"people_matter"' not in text
        assert "short of the target" not in text and "coming from" not in text   # the standard goal as it was before the normal

def test_locate_request_drops_empty_lines_and_inserts_data_verbatim():
    no_target = Subtask(type="navigate", description="go there")
    system, parts = build_locate_request("zero_shot", "put {target} in {x}", None, no_target, IMG, "xy_1000")
    lines = parts[0].text.splitlines()
    assert system and kinds(parts) == "TI" and lines[:2] == ["Request: put {target} in {x}", "Current subtask: go there"]
    assert not any(line.startswith(("Additional information", "Target:")) for line in lines)
    assert POINT_FORMATS["xy_1000"].instruction(640, 480) in parts[0].text
    text = build_locate_request("cot", "do it", "see {request}", SUB, IMG, "xy_1000")[1][0].text
    assert "Additional information from the requester: see {request}" in text.splitlines() and '"reasoning"' in text

def test_plan_request_view_and_tour():
    view = build_plan_request("cot", "do it", "NOTE", IMG, None, "xy_1000")[1]
    assert kinds(view) == "TI" and '"target_point"' in view[0].text and '"tour_frame"' not in view[0].text
    assert view[0].text.splitlines()[:2] == ["Request: do it", "Additional information from the requester: NOTE"]
    assert "the place, object or person the robot goes to in order to do something there" in view[0].text
    assert "never a position to stand in and never an intermediate waypoint" in view[0].text
    frames = [(0, ImagePart(b"f0", 640, 480)), (7, ImagePart(b"f7", 640, 480))]
    parts = build_plan_request("cot", "do it", None, None, frames, "xy_1000")[1]
    assert kinds(parts) == "TTITI" and (parts[1].text, parts[3].text) == ("Frame 0", "Frame 7") and parts[4] == frames[1][1]
    assert '"tour_frame"' in parts[0].text and '"target_point"' not in parts[0].text and "requester" not in parts[0].text
    for image, tour in ((None, None), (IMG, frames)):
        with pytest.raises(ValueError):
            build_plan_request("cot", "do it", None, image, tour, "xy_1000")

BASELINE = ("plan", "locate", "context", "goal")                                 # the stages of every run

@pytest.mark.parametrize("settings, more", [
    ({}, ()), ({"goal": "oracle"}, ()),
    ({"grounding": "oracle", "context_check": "always_no", "goal": "oracle"}, ()),          # whatever the oracle modes say
    ({"people": "oracle"}, ()), ({"constraints": "none"}, ()),                   # the method's own settings, without the method
    ({"goal": "constraints"}, ("people", "constraints")),
    ({"goal": "constraints", "people": "oracle"}, ("constraints",)),             # the people come from the harness
    ({"goal": "constraints", "constraints": "none"}, ("people", "constraints")), # the stage stays, as plan does under an oracle
    ({"goal": "constraints", "people": "oracle", "constraints": "none"}, ("constraints",)),
    ({"goal": "constraints", "grounding": "oracle", "context_check": "always_yes", "grounding_source": "view"},
     ("people", "constraints")),
])
def test_the_stages_of_a_run_follow_from_its_goal_and_people_settings_alone(settings, more):
    assert stages_of(RunConfig(**settings)) == (*BASELINE, *more)

# ------------------------------------------------------------------------- the people step of the constraint method
def test_people_request_is_the_image_and_nothing_of_an_episode():
    system, parts = build_people_request("zero_shot", IMG, "xy_1000", 40)
    assert system and kinds(parts) == "TI" and parts[1] == IMG        # the camera image, last: its size is the one recorded
    text = parts[0].text
    assert "List at most 40 people." in text and POINT_FORMATS["xy_1000"].instruction(640, 480) in text
    assert all(f'"{key}"' in text for key in ("appearance", "body_point", "faced_point", "facing"))
    assert all(f"- {facing}: " in text for facing in FACINGS)         # each of the nine values is said in words
    assert '"analysis"' not in text and text.endswith("Answer with one JSON object that matches the schema, and nothing else.")
    cot = build_people_request("cot", IMG, "xy_pixel", 12)[1][0].text
    assert '"analysis"' in cot and "List at most 12 people." in cot and POINT_FORMATS["xy_pixel"].instruction(640, 480) in cot
    assert cot.index('"analysis"') < cot.index('Set "body_point"')    # the reasoning is asked for before the points
    # No request, subtask or target goes in: every episode of a scene sends the same request, and shares one cached answer.
    assert not any(word in text + cot + system for word in ("Request", "subtask", "Target", "TARGET"))
    template = (PROMPTS_DIR / "people.md").read_text()               # the blocks the builder reads, and what it fills in
    assert re.findall(r"^## (\w+)$", template, re.M) == ["system", "task", "reasoning", "point_format", "output"]
    assert sorted(re.findall(r"\{[a-z_]+\}", template)) == ["{max_people}", "{point_instruction}"]

def test_people_request_reads_the_folder_it_is_given(tmp_path):
    folder = shutil.copytree(PROMPTS_DIR, tmp_path / "mine")
    own = (folder / "people.md").read_text().replace("List at most {max_people} people.", "Name no more than {max_people}.")
    (folder / "people.md").write_text(own)
    assert "Name no more than 7." in build_people_request("zero_shot", IMG, "xy_1000", 7, prompts_dir=folder)[1][0].text
    assert build_people_request("zero_shot", IMG, "xy_1000", 7) == build_people_request("zero_shot", IMG, "xy_1000", 7, prompts_dir=PROMPTS_DIR)

def seen(spots, us):                                                  # people numbered in the order given, each at a column of the image
    return [DetectedPerson(number=n, xy=spot[:2], yaw=spot[2] if len(spot) > 2 else None, pixel=(u, 240.0))
            for n, (spot, u) in enumerate(zip(spots, us), 1)]

def listed(people, target, observation=None, th=TH):                  # the list, line by line; seen from over the robot at the
    return people_lines(people, target, observation or open_floor(HERE), th).split("\n")     # origin, in a picture 501 wide

R = 1.2 / math.sqrt(3)                                                # three people 1.2 m from one another stand this far from their middle
CIRCLE = [(5.0 + R / 2, 0.6, math.radians(-120)), (5.0 - R, 0.0, 0.0), (5.0 + R / 2, -0.6, math.radians(120))]     # each faces the middle, (5, 0)
ROWS = [(4.0, 1.0, 0.0), (3.0, 0.5, 0.0), (4.0, 0.0, 0.0), (3.0, -0.5, 0.0), (4.0, -1.0, 0.0)]      # three in front, two behind, all facing +x

def test_three_in_a_circle_each_face_the_two_others_and_one_of_them_the_thing_beyond():
    people = seen(CIRCLE, (120.0, 250.0, 380.0))                      # the robot looks along +x: its left is the image's left
    assert listed(people, target_at(8.0, 0.0, 1.0)) == [
        "Person 1: 5.4 m from the robot, 2.7 m from the target, on the left of the image; faces person 2 (1.2 m), person 3 (1.2 m).",
        "Person 2: 4.3 m from the robot, 3.7 m from the target, in the centre of the image; "
        "faces person 1 (1.2 m), person 3 (1.2 m), the target (3.7 m).",
        "Person 3: 5.4 m from the robot, 2.7 m from the target, on the right of the image; faces person 1 (1.2 m), person 2 (1.2 m)."]

def test_a_person_the_target_point_was_marked_on_is_the_target_and_has_no_distance_to_it():
    people = seen(CIRCLE, (120.0, 250.0, 380.0))
    on_two = target_at(*body_point(CIRCLE[1][:2], (0.0, 0.0)))        # 0.2 m on the camera's side of person 2's centre, at chest height
    assert solver.target_person(open_floor(HERE), on_two, people, TH) == 2
    assert listed(people, on_two) == [                                # the others' distance is to that person, not to the point on them
        "Person 1: 5.4 m from the robot, 1.2 m from the target, on the left of the image; faces person 2 (the target) (1.2 m), person 3 (1.2 m).",
        "Person 2 (the target): 4.3 m from the robot, in the centre of the image; faces person 1 (1.2 m), person 3 (1.2 m).",
        "Person 3: 5.4 m from the robot, 1.2 m from the target, on the right of the image; faces person 1 (1.2 m), person 2 (the target) (1.2 m)."]
    beside = target_at(5.0 - R, 0.5, 1.0)                             # a thing half a metre beside person 2: nobody is the target
    assert solver.target_person(open_floor(HERE), beside, people, TH) is None
    lines = listed(people, beside)
    assert not any("(the target)" in line for line in lines) and lines[1].startswith("Person 2: 4.3 m from the robot, 0.5 m from the target, ")
    pair = [(4.15, 0.1), (4.15, -0.1)]                                # two within a body's radius of the moved point, (4.15, 0)
    lines = listed(seen(pair, (240.0, 260.0)), target_at(3.9, 0.0, 1.1))
    assert [line.split(":")[0] for line in lines] == ["Person 1 (the target)", "Person 2"]       # of two as near, the lower number

def test_two_rows_face_the_people_before_them_and_the_thing_they_watch():
    people = seen(ROWS, (60.0, 150.0, 250.0, 350.0, 440.0))
    assert listed(people, target_at(8.0, 0.0, 1.0)) == [
        "Person 1: 4.1 m from the robot, 4.1 m from the target, on the left of the image; faces the target (4.1 m).",
        "Person 2: 3.0 m from the robot, 5.0 m from the target, on the left of the image; "
        "faces person 1 (1.1 m), person 3 (1.1 m), the target (5.0 m).",
        "Person 3: 4.0 m from the robot, 4.0 m from the target, in the centre of the image; faces the target (4.0 m).",
        "Person 4: 3.0 m from the robot, 5.0 m from the target, on the right of the image; "
        "faces person 3 (1.1 m), person 5 (1.1 m), the target (5.0 m).",
        "Person 5: 4.1 m from the robot, 4.1 m from the target, on the right of the image; faces the target (4.1 m)."]
    narrow = listed(people, target_at(8.0, 0.0, 1.0), th=Thresholds(view_half_angle_deg=20.0))     # the angle is the threshold's:
    assert narrow[1].endswith("; faces the target (5.0 m).")          # the two before person 2 are 26.6 degrees off their heading

def test_what_is_faced_is_listed_nearest_first_with_the_robot_among_it():
    spots = [(4.0, 0.5, 0.0), (2.0, 0.5, 0.0), (3.0, 0.5, 0.0), (5.0, 0.5, math.pi)]     # one behind another, the last turned round
    lines = listed(seen(spots, (250.0,) * 4), target_at(6.0, 0.5, 1.0))
    assert [line.split("; ")[1] for line in lines] == [
        "faces person 4 (1.0 m), the target (2.0 m).",
        "faces person 3 (1.0 m), person 1 (2.0 m), person 4 (3.0 m), the target (4.0 m).",
        "faces person 1 (1.0 m), person 4 (2.0 m), the target (3.0 m).",
        "faces person 1 (1.0 m), person 3 (2.0 m), person 2 (3.0 m), the robot (5.0 m)."]       # and not the target, behind their back

def test_headings_that_are_not_known_and_people_who_face_nothing_listed():
    spots = [(4.0, 2.0, math.pi / 2), (3.0, 0.0, math.pi), (5.0, -1.0)]          # facing away, facing the robot, and not known
    assert listed(seen(spots, (100.0, 250.0, 400.0)), target_at(6.0, 0.0, 1.0)) == [
        "Person 1: 4.5 m from the robot, 2.8 m from the target, on the left of the image; faces none of these.",
        "Person 2: 3.0 m from the robot, 3.0 m from the target, in the centre of the image; faces the robot (3.0 m).",
        "Person 3: 5.1 m from the robot, 1.4 m from the target, on the right of the image; facing not known."]
    unknown = listed(seen([spot[:2] for spot in ROWS], (60.0, 150.0, 250.0, 350.0, 440.0)), target_at(8.0, 0.0, 1.0))
    assert len(unknown) == 5 and all(line.endswith(" of the image; facing not known.") for line in unknown)
    assert people_lines([], target_at(6.0, 0.0, 1.0), open_floor(HERE), TH) == ""        # nobody in view: no line

def test_the_third_of_the_image_is_read_from_the_width_of_the_camera_and_the_column_of_the_body_point():
    def third(u, observation=None):
        line, = listed(seen([(4.0, 0.0)], (u,)), target_at(8.0, 0.0), observation)
        return line.split(", ")[2].split(" of the image")[0]

    assert [third(u) for u in (0.0, 166.9, 167.0, 333.9, 334.0, 500.9)] == [                        # a picture 501 wide
        "on the left", "on the left", "in the centre", "in the centre", "on the right", "on the right"]
    wide = floor_observation()                                        # 640 wide: its thirds end at 213.3 and 426.7
    assert [third(u, wide) for u in (200.0, 213.4, 426.6, 426.7)] == ["on the left", "in the centre", "in the centre", "on the right"]

def test_distances_are_measured_from_the_robot_wherever_it_stands():
    wide = floor_observation()                                        # the robot at (1, 2), looking along +y
    line, = people_lines(seen([(1.0, 5.0, -math.pi / 2)], (320.0,)), target_at(4.0, 5.0, 1.0), wide, TH).split("\n")
    assert line == "Person 1: 3.0 m from the robot, 3.0 m from the target, in the centre of the image; faces the robot (3.0 m)."

# ---------------------------------------------------------------------- the constraints step of the constraint method
KINDS = ("keep_out_of_shared_space", "keep_out_of_view", "come_after", "within_distance", "visible_to", "face")
LIST = ("Person 1: 2.4 m from the robot, 1.1 m from the target, on the left of the image; faces the target (1.1 m).\n"
        "Person 2 (the target): 3.0 m from the robot, in the centre of the image; facing not known.")       # as people_lines gives it

def constraints_text(strategy="zero_shot", people=LIST, relevant=None, standoff=1.0, most=12, **given):
    return build_constraints_request(strategy, "do it", SUB, IMG, people, relevant, standoff, most, **given)[1][0].text

def test_constraints_request_is_the_case_the_list_of_people_the_six_kinds_and_the_numbered_image():
    system, parts = build_constraints_request("zero_shot", "do it", SUB, IMG, LIST, None, 1.0, 12)
    assert system and kinds(parts) == "TI" and parts[1] == IMG        # the image with the numbers, last: its size is the one recorded
    text = parts[0].text
    lines = text.splitlines()
    assert all(line in lines for line in QUERY) and all(line in lines for line in LIST.split("\n"))     # the list, line for line
    assert [line.split(":")[0] for line in lines if line.startswith("- ")] == [f"- {kind}" for kind in KINDS]     # each kind, in words
    assert all(f'"{key}"' in text for key in ("constraints", "because", "kind", "people", "range_m", "target", "at_least", "at_most"))
    assert "a ring and the word TARGET" in text and "a number, drawn in a yellow disc" in text and "by that number" in text
    assert "at most 12 constraints" in text and "at least 1 m from every person" in text and "about 1 m from the target" in text
    assert "give an empty list" in text and '"analysis"' not in text and "earlier step" not in text
    assert text.endswith("Answer with one JSON object that matches the schema, and nothing else.")
    template = (PROMPTS_DIR / "constraints.md").read_text()          # the blocks the builder reads, and what it fills in
    assert re.findall(r"^## (\w+)$", template, re.M) == ["system", "task", "people", "nobody", "relevant_people", "reasoning",
                                                          "constraints", "check", "output"]
    assert sorted(set(re.findall(r"\{[a-z_]+\}", template))) == ["{max_constraints}", "{people}", "{relevant_people}", "{request}",
                                                                 "{standoff_m}", "{subtask}", "{target}"]

def test_constraints_blocks_follow_strategy_who_was_found_and_relevant_people():
    cot = constraints_text("cot")
    assert all(f'"{key}"' in cot for key in ("analysis", "arrangement", "expectations", "placement"))
    assert '"arrangement"' not in constraints_text("zero_shot") and "each by number" in cot
    # The order of its instructions: describe what is in view, then decide, then check the decision once before answering.
    marks = [cot.index('fill in the three parts of "analysis"'), cot.index("- keep_out_of_shared_space: "),
             cot.index("check your constraints once against what the people are doing"), cot.index("Answer with one JSON object")]
    assert marks == sorted(marks) and cot.index(LIST) < marks[0]
    plain = constraints_text()                                       # without reasoning the same order, the description left out
    assert plain.index("- face: ") < plain.index("check your constraints once") < plain.index("Answer with one JSON object")
    assert "earlier step, before the numbers were drawn: PEOPLE" in constraints_text(relevant="PEOPLE")
    nobody = constraints_text("cot", people="")
    assert "No person was found in view" in nobody and "Person" not in nobody and "- come_after: " in nobody
    assert "No person was found" not in cot and "The people by number" in cot and "The people by number" not in nobody
    other = constraints_text(standoff=0.75, most=5)                  # the two numbers are the thresholds'
    assert "at least 0.75 m from every person" in other and "about 0.75 m from the target" in other and "at most 5 constraints" in other

def test_constraints_request_names_no_norm_and_inserts_what_it_is_given_verbatim():
    text = constraints_text("cot", people="Person 1: {request} {x}", relevant="see {people}")
    assert "Person 1: {request} {x}" in text.splitlines() and "drawn: see {people}" in text      # never scanned for placeholders
    assert "Various norms may apply, but state only the most essential ones" in text
    untargeted = build_constraints_request("zero_shot", "do it", Subtask(type="navigate", description="go there"), IMG, LIST, None, 1.0, 12)
    assert not any(line.startswith("Target:") for line in untargeted[1][0].text.splitlines())    # a line without its value is dropped

def test_constraints_request_reads_the_folder_it_is_given(tmp_path):
    folder = shutil.copytree(PROMPTS_DIR, tmp_path / "mine")
    own = (folder / "constraints.md").read_text().replace("at most {max_constraints} constraints", "no more than {max_constraints}")
    (folder / "constraints.md").write_text(own)
    assert "no more than 7." in constraints_text(most=7, prompts_dir=folder)
    assert constraints_text() == constraints_text(prompts_dir=PROMPTS_DIR) and "no more than" not in constraints_text()
