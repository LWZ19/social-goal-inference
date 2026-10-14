import json

import pytest

from sgi.schemas import constraints_schema, context_schema, goal_schema, location_schema, people_schema, plan_schema
from sgi.vlm.mock import centre_responder
from sgi.vlm.points import POINT_FORMATS

SIZE = (640, 480)                                   # (width, height) of the image as sent
PLAIN = ("zero_shot", "cot")                        # the strategies the plan, locate and context stages accept
ALL = (*PLAIN, "few_shot", "few_shot_cot")          # the goal stage takes these as well
MODES = ("face_target", "vlm_look_at")

def sent():                                         # (label, wire model) of every answer a run can ask for
    for pf in sorted(POINT_FORMATS):
        for s in PLAIN:
            yield f"plan {s} {pf}", plan_schema(s, pf, SIZE)
            yield f"plan on a tour {s} {pf}", plan_schema(s, pf, SIZE, [0, 5])
            yield f"locate {s} {pf}", location_schema(s, pf, SIZE)
            yield f"people {s} {pf}", people_schema(s, pf, SIZE)
        for s in ALL:
            for mode in MODES:
                yield f"goal {s} {mode} {pf}", goal_schema(s, pf, SIZE, mode)
    for s in PLAIN:
        yield f"context {s}", context_schema(s)
        yield f"constraints {s}", constraints_schema(s)

SENT = [(label, model.model_json_schema()) for label, model in sent()]

def objects(node, path="$"):                        # (path, keys, required) of every object under a node, $defs included
    if isinstance(node, dict):
        if isinstance(node.get("properties"), dict):
            yield path, list(node["properties"]), node.get("required")
        for key, sub in node.items():
            yield from objects(sub, f"{path}.{key}")
    elif isinstance(node, list):
        for index, sub in enumerate(node):
            yield from objects(sub, f"{path}[{index}]")

def keys(schema, *path):                            # the keys of the answer, or of the object under the keys named
    node = schema
    for name in path:
        node = node["properties"][name]
        node = node.get("items", node)                                                              # a list: its items
        node = next((branch for branch in node.get("anyOf", [node]) if "$ref" in branch), node)    # or null: the object
        node = schema["$defs"][node["$ref"].removeprefix("#/$defs/")]
    return list(node["properties"])

def test_the_walk_finds_nested_objects_and_the_cases_are_all_there():
    found = dict((path, held) for path, held, _ in objects({"properties": {"b": {}, "a": {"items": {"properties": {"z": {}}}}},
                                                           "$defs": {"D": {"properties": {"y": {}, "x": {}}}}}))
    assert found == {"$": ["b", "a"], "$.properties.a.items": ["z"], "$.$defs.D": ["y", "x"]}
    assert len(SENT) == len(POINT_FORMATS) * (4 * len(PLAIN) + len(ALL) * len(MODES)) + 2 * len(PLAIN) == 36

@pytest.mark.parametrize("schema", [schema for _, schema in SENT], ids=[label for label, _ in SENT])
def test_every_object_of_a_schema_sent_lists_its_keys_in_sorted_order(schema):
    # A provider that sorts an answer's keys and one that keeps the schema's order then write the same answer.
    found = list(objects(schema))
    assert found and found[0][0] == "$"
    for path, held, required in found:
        assert held == sorted(held), f"{path}: {held}"
        assert required == held, f"{path}: required {required}"         # every key is required, in the same order

@pytest.mark.parametrize("pf", sorted(POINT_FORMATS))
@pytest.mark.parametrize("plain, full, lead", [("zero_shot", "few_shot", []), ("cot", "few_shot_cot", ["reasoning"])])
def test_each_answer_has_the_keys_of_its_stage_in_the_order_they_are_written(plain, full, lead, pf):
    view, tour = (plan_schema(plain, pf, SIZE, shown).model_json_schema() for shown in (None, [0, 5]))
    assert keys(view) == keys(tour) == lead + ["subtasks"]
    assert keys(view, "subtasks") == ["description", "target", "target_point", "type"]
    assert keys(tour, "subtasks") == ["description", "target", "tour_frame", "type"]
    location = location_schema(plain, pf, SIZE).model_json_schema()
    assert keys(location) == lead + ["target_point"]
    assert keys(context_schema(plain).model_json_schema()) == lead + ["relevant_people", "stop_elsewhere"]
    for strategy in (plain, full):
        for mode, last in zip(MODES, ([], ["look_at_point"])):
            goal = goal_schema(strategy, pf, SIZE, mode).model_json_schema()
            assert keys(goal) == (["analysis"] if lead else []) + ["goal_point"] + last
            assert keys(goal, "goal_point") == ["x", "y"] and all(keys(goal, name) == ["x", "y"] for name in last)
            if lead:                                # the reasoning of the goal answer, in three parts
                assert keys(goal, "analysis") == ["arrangement", "expectations", "placement"]
    assert keys(view, "subtasks", "target_point") == keys(location, "target_point") == ["x", "y"]
    people = people_schema(plain, pf, SIZE).model_json_schema()                 # the people of the constraint method
    assert keys(people) == (["analysis"] if lead else []) + ["people"]          # its reasoning is one string, named as the goal's is
    assert keys(people, "people") == ["appearance", "body_point", "faced_point", "facing"]
    assert keys(people, "people", "body_point") == keys(people, "people", "faced_point") == ["x", "y"]
    stated = constraints_schema(plain).model_json_schema()                      # and its constraints, which hold no point
    assert keys(stated) == (["analysis"] if lead else []) + ["constraints"]
    assert keys(stated, "constraints") == ["because", "kind", "people", "range_m", "target"]
    assert keys(stated, "constraints", "range_m") == ["at_least", "at_most"]
    if lead:                                        # its reasoning has the three parts of the goal answer's
        assert keys(stated, "analysis") == ["arrangement", "expectations", "placement"]

def test_the_mock_writes_every_answer_in_the_order_of_its_schema():
    for label, model in sent():                     # the built-in responder fills a schema key by key, and names none
        answer = centre_responder((), model.model_json_schema())
        assert json.dumps(answer) == json.dumps(model.model_validate(answer).model_dump()), label
