import json

import pytest
from pydantic import ValidationError

from sgi.schemas import (FACINGS, ConstraintsAnswer, PeopleAnswer, PersonEntry, constraints_from_wire,
                         constraints_schema, context_from_wire, context_schema, goal_from_wire, goal_schema, goal_to_wire,
                         location_from_wire, location_schema, people_from_wire, people_schema, plan_from_wire, plan_schema)
from sgi.types import Constraint, ContextCheck, DistanceRange, GoalMark, GoalReasoning, Location, Plan, Point, Subtask
from sgi.vlm.points import POINT_FORMATS

SIZE = (640, 480)                                   # (width, height) of the image as sent
PT = {"x": 320, "y": 240}                           # valid in both point formats
R3 = {"arrangement": "a", "expectations": "b", "placement": "c"}     # the goal answer's reasoning as written: "analysis"
WHY = GoalReasoning(people="a", expectations="b", placement="c")      # and as the pipeline holds it
NAV = {"type": "navigate", "description": "d", "target": "t"}
FACED = {"appearance": "red coat", "body_point": PT, "faced_point": {"x": 160, "y": 360}, "facing": "left"}    # one entry of
ALONE = {"appearance": "tall, in grey", "body_point": {"x": 480, "y": 120}, "faced_point": None, "facing": "unclear"}    # the people answer

def props(model, definition=None):
    schema = model.model_json_schema()
    assert "minLength" not in json.dumps(schema)                     # a keyword in neither provider's schema subset
    node = schema["$defs"][definition] if definition else schema
    assert node["required"] == list(node["properties"]) and node["additionalProperties"] is False
    return list(node["properties"])

@pytest.mark.parametrize("pf", ["xy_1000", "xy_pixel"])
@pytest.mark.parametrize("strategy, lead", [("zero_shot", []), ("few_shot", []), ("cot", ["reasoning"]), ("few_shot_cot", ["reasoning"])])
def test_properties_are_ordered_and_all_required(strategy, lead, pf):
    assert props(plan_schema(strategy, pf, SIZE)) == lead + ["subtasks"]
    assert props(plan_schema(strategy, pf, SIZE), "SubtaskWire") == ["description", "target", "target_point", "type"]
    assert props(plan_schema(strategy, pf, SIZE, [0, 5]), "SubtaskWire") == ["description", "target", "tour_frame", "type"]
    assert props(location_schema(strategy, pf, SIZE)) == lead + ["target_point"]
    assert props(context_schema(strategy)) == lead + ["relevant_people", "stop_elsewhere"]
    analysis = ["analysis"] if lead else []                          # the goal answer's reasoning has a name of its own
    assert props(goal_schema(strategy, pf, SIZE, "face_target")) == analysis + ["goal_point"]
    look = goal_schema(strategy, pf, SIZE, "vlm_look_at")
    assert props(look) == analysis + ["goal_point", "look_at_point"]
    sent = look.model_json_schema()["properties"]
    assert list(sent["goal_point"]) == ["$ref"]                      # strict mode: a $ref carries no sibling key
    assert "reasoning" not in sent                                   # under no strategy: it would sort after goal_point
    if lead:
        assert list(sent["analysis"]) == ["$ref"]
        assert props(look, "GoalReasoningWire") == ["arrangement", "expectations", "placement"]
        assert context_schema(strategy).model_json_schema()["properties"]["reasoning"]["type"] == "string"

def test_empty_reasoning_and_extra_keys_fail():
    cases = [(plan_schema("cot", "xy_1000", SIZE), {"reasoning": "r", "subtasks": [{**NAV, "target_point": PT}]}),
             (location_schema("cot", "xy_1000", SIZE), {"reasoning": "r", "target_point": None}),
             (context_schema("cot"), {"reasoning": "r", "relevant_people": "", "stop_elsewhere": False}),
             (goal_schema("few_shot_cot", "xy_1000", SIZE, "vlm_look_at"), {"analysis": R3, "goal_point": PT, "look_at_point": None})]
    for model, good in cases:
        model.model_validate(good)
        lead = next(iter(good))                                      # "reasoning", and "analysis" in the goal answer
        empty = {**R3, "expectations": ""} if isinstance(good[lead], dict) else ""
        missing = {k: v for k, v in good.items() if k != lead}
        for bad in ({**good, lead: empty}, {**good, lead: None}, {**good, "extra": 1}, missing):
            with pytest.raises(ValidationError):
                model.model_validate(bad)

def test_only_the_empty_string_fails_and_any_other_reasoning_is_kept_as_written():
    location, goal = location_schema("cot", "xy_1000", SIZE), goal_schema("few_shot_cot", "xy_1000", SIZE, "face_target")
    for text in (" r\n", " ", "\n\t"):                                          # nothing is stripped, nothing else is judged
        kept = {"reasoning": text, "target_point": None}
        assert location.model_validate(kept).model_dump() == kept
    spaced = {"analysis": {**R3, "placement": "  "}, "goal_point": PT}
    assert goal.model_validate(spaced).model_dump() == spaced
    for model, bad, where in [(location, {"reasoning": "", "target_point": None}, ("reasoning",)),
                              (goal, {"analysis": {**R3, "placement": ""}, "goal_point": PT}, ("analysis", "placement"))]:
        with pytest.raises(ValidationError) as caught:
            model.model_validate(bad)
        # the text the retry note carries back to the model, so its wording is fixed (G2)
        assert [(e["loc"], e["msg"]) for e in caught.value.errors()] == [(where, "Value error, must not be empty")]

@pytest.mark.parametrize("strategy", ["cot", "few_shot_cot"])
def test_no_string_length_keyword_is_sent(strategy):
    # OpenAI strict mode supports no length keyword on a string, so only pydantic holds the rule that reasoning has text.
    for model, lead in ((plan_schema(strategy, "xy_1000", SIZE), "reasoning"), (plan_schema(strategy, "xy_1000", SIZE, [0, 5]), "reasoning"),
                        (location_schema(strategy, "xy_pixel", SIZE), "reasoning"), (context_schema(strategy), "reasoning"),
                        (goal_schema(strategy, "xy_1000", SIZE, "vlm_look_at"), "analysis")):
        sent = json.dumps(model.model_json_schema())
        assert f'"{lead}"' in sent and '"minLength"' not in sent and '"maxLength"' not in sent

def test_tour_frames_and_points_are_bounded():
    tour = plan_schema("zero_shot", "xy_1000", SIZE, [0, 5, 9])
    tour.model_validate({"subtasks": [{**NAV, "tour_frame": 5}, {**NAV, "tour_frame": None}]})
    pixel = location_schema("zero_shot", "xy_pixel", SIZE)
    pixel.model_validate({"target_point": {"x": 600, "y": 100}})
    for model, bad in [(tour, {"subtasks": [{**NAV, "tour_frame": 4}]}),            # not a shown index
                       (tour, {"subtasks": [{**NAV, "tour_frame": 5, "target_point": PT}]}),
                       (pixel, {"target_point": {"x": 100, "y": 600}})]:            # below a 480-row image
        with pytest.raises(ValidationError):
            model.model_validate(bad)
    with pytest.raises(ValueError):
        plan_schema("zero_shot", "xy_1000", SIZE, [])

def test_from_wire_converts_points_in_both_formats():
    wire = {"reasoning": "r", "subtasks": [{**NAV, "target_point": {"x": 160, "y": 360}},
                                            {"type": "act", "description": "a", "target": None, "target_point": None}]}
    assert plan_from_wire(wire, "xy_pixel", SIZE) == Plan(reasoning="r", image_sha256=None, subtasks=[
        Subtask(type="navigate", description="d", target="t", target_point=Point(x=250, y=750)),
        Subtask(type="act", description="a")])
    assert plan_from_wire(wire, "xy_1000", SIZE).subtasks[0].target_point == Point(x=160, y=360)
    tour = plan_from_wire({"subtasks": [{**NAV, "tour_frame": 5}]}, "xy_1000", SIZE)
    assert tour.reasoning is None and tour.subtasks[0].tour_frame == 5 and tour.subtasks[0].target_point is None
    assert location_from_wire({"target_point": {"x": 480, "y": 120}}, "xy_pixel", SIZE) == Location(target_point=Point(x=750, y=250))
    assert location_from_wire({"reasoning": "r", "target_point": None}, "xy_pixel", SIZE) == Location(reasoning="r")

@pytest.mark.parametrize("said", [True, False])
def test_stop_elsewhere_is_read_as_people_matter_with_the_same_polarity(said):
    assert context_from_wire({"relevant_people": "p", "stop_elsewhere": said}) == ContextCheck(relevant_people="p", people_matter=said)
    reasoned = context_from_wire({"reasoning": "r", "relevant_people": "p", "stop_elsewhere": said})
    assert reasoned == ContextCheck(reasoning="r", relevant_people="p", people_matter=said)

def test_the_names_an_answer_had_before_are_no_longer_valid():
    for model, old in [(context_schema("zero_shot"), {"relevant_people": "p", "people_matter": True}),
                       (goal_schema("cot", "xy_1000", SIZE, "face_target"), {"reasoning": R3, "goal_point": PT}),
                       (goal_schema("cot", "xy_1000", SIZE, "face_target"), {"analysis": WHY.model_dump(), "goal_point": PT})]:
        with pytest.raises(ValidationError):
            model.model_validate(old)

def test_goal_to_wire_inverts_goal_from_wire():
    wire = {"analysis": R3, "goal_point": {"x": 480, "y": 300}, "look_at_point": {"x": 320, "y": 240}}
    mark = goal_from_wire(wire, "xy_pixel", SIZE)
    assert mark == GoalMark(reasoning=WHY, goal_point=Point(x=750, y=625), look_at_point=Point(x=500, y=500))
    assert (mark.reasoning.people, mark.reasoning.expectations, mark.reasoning.placement) == ("a", "b", "c")
    back = goal_to_wire(mark, "few_shot_cot", "xy_pixel", SIZE, "vlm_look_at")
    assert back == wire and goal_from_wire(back, "xy_pixel", SIZE) == mark
    assert list(back) == ["analysis", "goal_point", "look_at_point"]            # the order they are written in,
    assert list(back["analysis"]) == ["arrangement", "expectations", "placement"]     # at every depth
    assert goal_schema("few_shot_cot", "xy_pixel", SIZE, "vlm_look_at").model_validate(back).model_dump() == back
    for strategy in ("zero_shot", "few_shot"):                                  # no reasoning asked for, none written
        assert goal_to_wire(mark, strategy, "xy_pixel", SIZE, "vlm_look_at") == {k: v for k, v in wire.items() if k != "analysis"}
        assert "analysis" not in goal_schema(strategy, "xy_pixel", SIZE, "vlm_look_at").model_json_schema()["properties"]
    assert goal_to_wire(mark, "few_shot", "xy_1000", SIZE, "face_target") == {"goal_point": {"x": 750, "y": 625}}
    bare = goal_from_wire({"goal_point": {"x": 750, "y": 625}}, "xy_1000", SIZE)
    assert goal_to_wire(bare, "zero_shot", "xy_1000", SIZE, "vlm_look_at") == {"goal_point": {"x": 750, "y": 625}, "look_at_point": None}
    with pytest.raises(ValueError):
        goal_to_wire(bare, "few_shot_cot", "xy_1000", SIZE, "face_target")       # an example without a rationale

# ---------------------------------------------------------------------------- the people answer of the constraint method
def baseline():                                     # every answer a run asked for before the constraint method
    for pf in sorted(POINT_FORMATS):
        for s in ("zero_shot", "cot", "few_shot", "few_shot_cot"):
            yield from (plan_schema(s, pf, SIZE), plan_schema(s, pf, SIZE, [0, 5]), location_schema(s, pf, SIZE), context_schema(s))
            yield from (goal_schema(s, pf, SIZE, mode) for mode in ("face_target", "vlm_look_at"))

def constructs(node, parent="", found=None):        # every JSON-schema keyword a schema uses, with the kind of what it holds
    found = set() if found is None else found
    kind = lambda branch: "$ref" if "$ref" in branch else str(branch.get("type"))
    if isinstance(node, dict):
        for key, sub in node.items():
            if parent not in ("properties", "$defs"):                           # there a key is a name, not a keyword
                held = {"type": lambda: f"={sub}", "items": lambda: f":{kind(sub)}", "anyOf": lambda: ":" + "|".join(sorted(map(kind, sub)))}
                found.add(key + held.get(key, lambda: "")())
            constructs(sub, key, found)
    elif isinstance(node, list):
        for sub in node:
            constructs(sub, parent, found)
    return found

def refs_with_a_sibling(node):                      # objects that hold a $ref beside another key: strict mode rejects them
    if isinstance(node, dict):
        yield from ([sorted(node)] if "$ref" in node and len(node) > 1 else [])
        for sub in node.values():
            yield from refs_with_a_sibling(sub)
    elif isinstance(node, list):
        for sub in node:
            yield from refs_with_a_sibling(sub)

def test_the_walks_see_keywords_and_not_names():
    schema = {"properties": {"type": {"anyOf": [{"$ref": "#/$defs/P"}, {"type": "null"}], "description": "d"},
                             "items": {"items": {"type": "integer"}, "type": "array", "maxItems": 3}},
              "$defs": {"P": {"properties": {"x": {"$ref": "#/$defs/Q", "description": "d"}}, "type": "object"}}, "type": "object"}
    assert constructs(schema) == {"properties", "$defs", "anyOf:$ref|null", "$ref", "type=null", "description", "items:integer",
                                  "type=integer", "type=array", "maxItems", "type=object"}
    assert list(refs_with_a_sibling(schema)) == [["$ref", "description"]]

@pytest.mark.parametrize("pf", ["xy_1000", "xy_pixel"])
@pytest.mark.parametrize("strategy, lead", [("zero_shot", []), ("cot", ["analysis"])])
def test_the_people_answer_has_its_keys_in_writing_order_and_the_shapes_a_strict_mode_takes(strategy, lead, pf):
    model = people_schema(strategy, pf, SIZE)
    assert props(model) == lead + ["people"] == sorted(props(model))
    assert props(model, "PersonWire") == ["appearance", "body_point", "faced_point", "facing"]
    sent = model.model_json_schema()
    shape = lambda field: {key: value for key, value in field.items() if key not in ("description", "title")}
    point = {"$ref": f"#/$defs/{POINT_FORMATS[pf].wire_model(*SIZE).__name__}"}
    assert shape(sent["properties"]["people"]) == {"items": {"$ref": "#/$defs/PersonWire"}, "type": "array"}     # no bound on the list
    person = sent["$defs"]["PersonWire"]["properties"]
    assert person["body_point"] == point                             # strict mode: a $ref carries no sibling key
    assert shape(person["faced_point"]) == {"anyOf": [point, {"type": "null"}]}
    assert shape(person["facing"]) == {"enum": list(FACINGS), "type": "string"} and shape(person["appearance"]) == {"type": "string"}
    assert [shape(sent["properties"][key]) for key in lead] == [{"type": "string"}] * len(lead)
    assert all("description" in field for name, field in {**sent["properties"], **person}.items() if name != "body_point")
    # Nothing in it that the answers of the baseline do not hold already, which both providers' strict modes take.
    assert constructs(sent) <= set().union(*(constructs(known.model_json_schema()) for known in baseline()))
    assert list(refs_with_a_sibling(sent)) == []

def test_the_facings_are_the_nine_of_the_design_in_its_order():
    assert FACINGS == ("towards_camera", "towards_right", "right", "away_right", "away_from_camera", "away_left", "left",
                       "towards_left", "unclear")

def test_a_people_answer_is_held_to_its_keys_its_facings_and_the_image_sent():
    model, plain = people_schema("cot", "xy_pixel", SIZE), people_schema("zero_shot", "xy_pixel", SIZE)
    good = {"analysis": "two people, one of them turned away", "people": [FACED, ALONE]}
    assert model.model_validate(good).model_dump() == good and list(model.model_validate(good).model_dump()) == ["analysis", "people"]
    assert model.model_validate({"analysis": "nobody in view", "people": []}).model_dump()["people"] == []
    assert plain.model_validate({"people": [ALONE]}).model_dump() == {"people": [ALONE]}
    entry = lambda **changed: {**good, "people": [{**FACED, **changed}]}
    for bad in ({**good, "analysis": ""}, {"people": [FACED]}, {**good, "extra": 1}, {**good, "people": None},
                entry(facing="north"), entry(facing=None), entry(body_point=None), entry(appearance=None), entry(number=1),
                entry(body_point={"x": 100, "y": 600}), entry(faced_point={"x": 700, "y": 100}),      # outside a 640 x 480 image
                {**good, "people": [{key: value for key, value in FACED.items() if key != "faced_point"}]}):     # null is written out
        with pytest.raises(ValidationError):
            model.model_validate(bad)
    with pytest.raises(ValidationError):
        plain.model_validate(good)                                   # no reasoning asked for, none taken

def test_people_from_wire_converts_the_points_and_gives_the_reasoning_its_own_name():
    answer = people_from_wire({"analysis": "two people", "people": [FACED, ALONE]}, "xy_pixel", SIZE)
    assert answer == PeopleAnswer(reasoning="two people", people=[
        PersonEntry(appearance="red coat", body_point=Point(x=500, y=500), faced_point=Point(x=250, y=750), facing="left"),
        PersonEntry(appearance="tall, in grey", body_point=Point(x=750, y=250), faced_point=None, facing="unclear")])
    plain = people_from_wire({"people": [FACED]}, "xy_1000", SIZE)
    assert plain.reasoning is None and (plain.people[0].body_point, plain.people[0].faced_point) == (Point(x=320, y=240), Point(x=160, y=360))
    assert people_from_wire({"people": []}, "xy_1000", SIZE) == PeopleAnswer(people=[])

def test_a_key_that_two_answers_share_is_described_by_the_answer_it_is_in():
    people, goal = (schema.model_json_schema()
                    for schema in (people_schema("cot", "xy_1000", SIZE), goal_schema("cot", "xy_1000", SIZE, "face_target")))
    assert people["properties"]["analysis"]["type"] == "string" and "person" in people["properties"]["people"]["description"]
    assert goal["properties"]["analysis"] == {"$ref": "#/$defs/GoalReasoningWire"}               # the same key, another answer
    described = {"plan": plan_schema("cot", "xy_1000", SIZE).model_json_schema()["$defs"]["SubtaskWire"]["properties"],
                 "locate": location_schema("cot", "xy_1000", SIZE).model_json_schema()["properties"]}
    assert described["plan"]["target"]["description"].startswith("For navigate: a few words that pick out the place")
    assert described["plan"]["target_point"]["description"] == described["locate"]["target_point"]["description"]     # as they were
    assert described["locate"]["reasoning"]["description"] == "What in the image could be the target, written before the point."

# ---------------------------------------------------------------------- the constraints answer of the constraint method
KINDS = ("keep_out_of_shared_space", "keep_out_of_view", "come_after", "within_distance", "visible_to", "face")
TURN = {"because": "They stand one behind another.", "kind": "come_after", "people": [3, 1], "range_m": None, "target": False}     # two
NEAR = {"because": "Something is handed over.", "kind": "within_distance", "people": [], "range_m": {"at_least": 0.5, "at_most": 1.25},
        "target": True}                                                 # constraints as an answer writes them

@pytest.mark.parametrize("strategy, lead", [("zero_shot", []), ("cot", ["analysis"])])
def test_the_constraints_answer_has_its_keys_in_writing_order_and_the_shapes_a_strict_mode_takes(strategy, lead):
    model = constraints_schema(strategy)
    assert props(model) == lead + ["constraints"] == sorted(props(model))
    assert props(model, "ConstraintWire") == ["because", "kind", "people", "range_m", "target"]  # the schema's order, which is sorted
    assert props(model, "RangeWire") == ["at_least", "at_most"]
    sent = model.model_json_schema()
    shape = lambda field: {key: value for key, value in field.items() if key not in ("description", "title")}
    assert shape(sent["properties"]["constraints"]) == {"items": {"$ref": "#/$defs/ConstraintWire"}, "type": "array"}     # no bound on the list
    one, sides = sent["$defs"]["ConstraintWire"]["properties"], sent["$defs"]["RangeWire"]["properties"]
    assert shape(one["kind"]) == {"enum": list(KINDS), "type": "string"} and shape(one["because"]) == {"type": "string"}
    assert shape(one["people"]) == {"items": {"type": "integer"}, "type": "array"} and shape(one["target"]) == {"type": "boolean"}
    assert shape(one["range_m"]) == {"anyOf": [{"$ref": "#/$defs/RangeWire"}, {"type": "null"}]}
    assert [shape(side) for side in sides.values()] == [{"type": "number"}] * 2
    described = {**sent["properties"], **one, **sides}
    if lead:                                                         # the reasoning: the three parts the goal answer has, by a lone $ref
        assert sent["properties"]["analysis"] == {"$ref": "#/$defs/ConstraintReasoningWire"}
        assert props(model, "ConstraintReasoningWire") == ["arrangement", "expectations", "placement"]
        described = {**described, **sent["$defs"]["ConstraintReasoningWire"]["properties"]}
    assert all("description" in field for name, field in described.items() if name != "analysis")
    # Two shapes the answers of the baseline do not hold, each a plain type: a list of whole numbers, and a number.
    known = set().union(*(constructs(answer.model_json_schema()) for answer in baseline()))
    assert constructs(sent) - known == {"items:integer", "type=number"} and list(refs_with_a_sibling(sent)) == []

def test_a_constraints_answer_is_held_to_its_keys_and_its_six_kinds():
    model, plain = constraints_schema("cot"), constraints_schema("zero_shot")
    good = {"analysis": R3, "constraints": [TURN, NEAR]}
    assert model.model_validate(good).model_dump() == good and list(model.model_validate(good).model_dump()) == ["analysis", "constraints"]
    assert plain.model_validate({"constraints": []}).model_dump() == {"constraints": []}        # nothing asked of the place
    for kind in KINDS:
        assert plain.model_validate({"constraints": [{**TURN, "kind": kind}]}).model_dump()["constraints"][0]["kind"] == kind
    loose = {**NEAR, "because": "", "people": [2, 2, 40, -1], "range_m": {"at_least": 3, "at_most": -0.5}}      # judged later, by the solver
    assert plain.model_validate({"constraints": [loose]}).model_dump() == {"constraints": [{**loose, "range_m": {"at_least": 3.0, "at_most": -0.5}}]}
    entry = lambda **changed: {**good, "constraints": [{**NEAR, **changed}]}
    for bad in ({**good, "analysis": {**R3, "placement": ""}}, {"constraints": [TURN]}, {**good, "extra": 1}, {**good, "constraints": None},
                entry(kind="keep_clear_of"), entry(kind=None), entry(because=None), entry(people=None), entry(people=["the tall one"]),
                entry(people=[1.5]), entry(target=None), entry(number=1), entry(range_m={"at_least": 0.5}),
                entry(range_m={"at_least": 0.5, "at_most": None}), entry(range_m={"at_least": 0.5, "at_most": 1.0, "unit": "m"}),
                {**good, "constraints": [{key: value for key, value in TURN.items() if key != "range_m"}]}):       # null is written out
        with pytest.raises(ValidationError):
            model.model_validate(bad)
    with pytest.raises(ValidationError):
        plain.model_validate(good)                                   # no reasoning asked for, none taken

def test_constraints_from_wire_keeps_each_constraint_as_it_was_written():
    answer = constraints_from_wire({"analysis": R3, "constraints": [TURN, NEAR]})
    assert answer == ConstraintsAnswer(reasoning=WHY, constraints=[                  # the reasoning under the names the goal mark's has
        Constraint(because="They stand one behind another.", kind="come_after", people=[3, 1], range_m=None, target=False),
        Constraint(because="Something is handed over.", kind="within_distance", people=[],
                   range_m=DistanceRange(at_least=0.5, at_most=1.25), target=True)])
    assert constraints_from_wire({"constraints": []}) == ConstraintsAnswer(constraints=[])
    odd = {**NEAR, "people": [2, 2, 40, -1], "range_m": {"at_least": 3.0, "at_most": -0.5}, "target": False}
    (kept,) = constraints_from_wire({"constraints": [odd]}).constraints             # a repeated or unknown number, a range the wrong way
    assert kept.model_dump() == odd and kept.people == [2, 2, 40, -1]               # round and below zero: all for the solver to judge

@pytest.mark.parametrize("side", [float("inf"), float("-inf"), float("nan"), None])
def test_a_range_with_a_side_that_is_no_finite_number_is_read_as_no_range(side):
    # JSON has no infinity: the cache gives such a side back as null. Read as no range either way, the answer and its replay agree.
    for given in ({"at_least": 0.5, "at_most": side}, {"at_least": side, "at_most": 2.0}, {"at_least": side, "at_most": side}):
        (read,) = constraints_from_wire({"constraints": [{**NEAR, "range_m": given}]}).constraints
        assert read == Constraint(because=NEAR["because"], kind="within_distance", people=[], range_m=None, target=True)
    written = json.loads('{"constraints": [{"because": "x", "kind": "within_distance", "people": [1], '
                         '"range_m": {"at_least": 1e999, "at_most": Infinity}, "target": false}]}')
    assert constraints_schema("zero_shot").model_validate(written).model_dump()["constraints"][0]["range_m"]["at_least"] == float("inf")
    assert constraints_from_wire(written).constraints[0].range_m is None            # a valid answer: the constraint will be unusable

def test_the_constraints_answer_describes_its_own_keys():
    sent = constraints_schema("cot").model_json_schema()
    one, parts = sent["$defs"]["ConstraintWire"]["properties"], sent["$defs"]["ConstraintReasoningWire"]["properties"]
    assert one["because"]["description"].startswith("One sentence") and "number" in one["people"]["description"]
    assert "within_distance" in one["range_m"]["description"] and "null" in one["range_m"]["description"]
    assert one["target"]["type"] == "boolean" and "the target" in one["target"]["description"]      # not the plan's `target`, a string
    assert "number" in parts["arrangement"]["description"]           # the people are referred to by the numbers drawn on them
    goal = goal_schema("cot", "xy_1000", SIZE, "face_target").model_json_schema()["$defs"]["GoalReasoningWire"]["properties"]
    assert goal["arrangement"]["description"] != parts["arrangement"]["description"] and "number" not in goal["arrangement"]["description"]
    assert goal["expectations"] == parts["expectations"]             # the same question in both answers
