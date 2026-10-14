import ast
import json
import re
from pathlib import Path

from sgi import solver
from sgi.config import Thresholds
from sgi.prompts import (Example, build_constraints_request, build_context_request, build_goal_request,
                         build_locate_request, build_people_request, build_plan_request, people_lines)
from sgi.schemas import constraints_schema, context_schema, goal_schema, location_schema, people_schema, plan_schema
from sgi.stages import constraints as constraints_step
from sgi.stages import people as people_step
from sgi.types import DetectedPerson, GoalReasoning, GroundedTarget, Point, Subtask
from sgi.vlm.base import _RETRY_NOTE, ImagePart, TextPart
from sgi.vlm.points import POINT_FORMATS
from tests.helpers import floor_observation

DENY = ["coffee", "drink", "stall", "kiosk", "queue", "queueing", "queuing", "audience", "photo", "photograph",
        "elevator", "receptionist", "cashier", "shop", "restaurant", "cafe", "barista", "waiter", "counter", "desk",
        "reception", "lecture", "selfie", "conversation", "printer", "in line", "lift door"]
PATTERN = re.compile(r"\b(?:" + "|".join(w.replace(" ", r"\s+") for w in DENY) + r")(?:e?s)?\b", re.IGNORECASE)
SIZE = (640, 480)

def hits(text):
    return sorted({" ".join(m.group(0).lower().split()) for m in PATTERN.finditer(text)})

def requests():                                     # (label, system, parts, schema) for every combination a run can send
    img, sub = ImagePart(b"i", *SIZE), Subtask(type="navigate", description="do the step", target="the thing")
    blank = [Example(img, "", "", Point(x=500, y=500), GoalReasoning(people="", expectations="", placement=""), Point(x=500, y=250))]
    for pf in sorted(POINT_FORMATS):
        for s in ("zero_shot", "cot"):
            yield f"plan {s} {pf}", *build_plan_request(s, "do the thing", "a note", img, None, pf), plan_schema(s, pf, SIZE)
            yield f"plan tour {s}", *build_plan_request(s, "do the thing", None, None, [(0, img), (5, img)], pf), plan_schema(s, pf, SIZE, [0, 5])
            yield f"locate {s} {pf}", *build_locate_request(s, "do the thing", "a note", sub, img, pf), location_schema(s, pf, SIZE)
            yield f"context {s}", *build_context_request(s, "do the thing", sub, img, 1.0), context_schema(s)
            yield f"people {s} {pf}", *build_people_request(s, img, pf, 40), people_schema(s, pf, SIZE)
        for s in ("zero_shot", "cot", "few_shot", "few_shot_cot"):
            for mode in ("face_target", "vlm_look_at"):
                yield (f"goal {s} {mode} {pf}", *build_goal_request(s, "do the thing", sub, img, "some people", blank, pf, mode),
                       goal_schema(s, pf, SIZE, mode))
    for s in ("zero_shot", "cot"):                  # the constraints request holds no point: with people found, and with nobody
        for found in ("Person 1: facing not known.", ""):
            yield (f"constraints {s} {'people' if found else 'nobody'}",
                   *build_constraints_request(s, "do the thing", sub, img, found, "some people", 1.0, 12), constraints_schema(s))

def test_the_lint_matches_whole_words_plurals_and_phrases():
    assert hits("Wait in  line at the Coffee stalls; photos by the LIFT DOOR") == ["coffee", "in line", "lift door", "photos", "stalls"]
    assert hits("shopping for photons online, an uplift doorway, queued") == []

def test_nothing_sent_names_a_task_or_a_situation():
    cases = list(requests())
    assert len(cases) == 18 * len(POINT_FORMATS) + 4
    for label, system, parts, schema in cases:
        for text in [system, json.dumps(schema.model_json_schema())] + [p.text for p in parts if isinstance(p, TextPart)]:
            assert hits(text) == [], f"{label}: {hits(text)}"
            assert not re.search(r"\{[a-z_]+\}", text), f"{label}: unfilled placeholder"
    assert hits(_RETRY_NOTE.format(error="x")) == []                 # raises KeyError if the note has another placeholder

def test_the_solver_and_the_two_steps_of_the_method_name_no_task_and_no_situation():
    as_words = lambda text: text.replace("_", " ")                   # a word inside a name counts as well
    assert hits(as_words("def _queue_end(waiting): ...")) == ["queue"]
    for module, least in ((solver, 20000), (people_step, 4000), (constraints_step, 4000)):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert len(source) > least and hits(source) == [], module.__name__
        assert hits(as_words(source.replace("time.perf_counter", "time.clock"))) == [], module.__name__     # the standard library's name
    for name in ("test_solver.py", "test_solver_floor.py", "solver_layouts.py", "test_people.py", "test_constraints.py",
                 "test_constraint_runs.py"):                         # and what their tests are called
        tree = ast.parse((Path(__file__).parent / name).read_text(encoding="utf-8"))
        named = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        assert len(named) > 10 and hits(as_words(" ".join(named))) == [], name

def test_the_list_of_people_names_no_task_and_no_situation():
    people = [                                                       # the fixed words of the list reach the model with it
        DetectedPerson(number=1, xy=(1.0, 5.0), yaw=-1.57, pixel=(100.0, 200.0), label="a"),             # faces the robot, at (1, 2)
        DetectedPerson(number=2, xy=(1.0, 6.0), yaw=-1.57, pixel=(320.0, 200.0)),                        # faces person 1 and the robot
        DetectedPerson(number=3, xy=(3.0, 6.0), yaw=1.57, pixel=(600.0, 200.0)), DetectedPerson(number=4, xy=(0.0, 4.0), pixel=(5.0, 9.0))]
    on_person = GroundedTarget(point=Point(x=500, y=500), pixel=(320.0, 240.0), world=(1.0, 5.8, 1.1))
    a_thing = on_person.model_copy(update={"world": (3.0, 7.5, 1.0)})                                    # before person 3
    texts = [people_lines(people, target, floor_observation(), Thresholds()) for target in (on_person, a_thing)]
    said = " ".join(texts)
    assert all(phrase in said for phrase in ("(the target)", "from the target", "the robot (", "the target (", "on the left", "in the centre",
                                             "on the right", "faces none of these", "facing not known"))
    assert [hits(text) for text in texts] == [[], []]
