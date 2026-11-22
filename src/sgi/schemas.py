"""Wire schemas per stage, strategy and point format, and the conversion of a wire answer to the stage outputs.

The order of the keys. In every wire schema, at every depth, the keys are defined in the order the answer is to
be written in, and that order is their sorted order (plain string sort). A model writes its answer key by key,
so what it writes first is all that a later key can lean on: reasoning has to come before the point it argues
for. One provider writes the keys in the order of the schema, another sorts them whatever the schema says, as
the school's gateway does. With the two orders the same, both write the same answer. A wire key is therefore
named to sort into its place, and where that gives it another name than the stage output has
(`stop_elsewhere`, `analysis`, `arrangement`) the *_from_wire functions map it back, so nothing after them sees
a wire name. tests/test_key_order.py holds every schema to the rule, and `python -m sgi.smoke` warns of an
answer whose keys came in another order.

The two answers of the constraint method have their stage outputs here, each beside its schema: `PeopleAnswer`
and `ConstraintsAnswer`.
"""
from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Mapping, Sequence

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, create_model

from sgi.config import Strategy
from sgi.types import (Constraint, ConstraintKind, ContextCheck, DistanceRange, GoalMark, GoalReasoning, Location, Plan,
                       Point, Subtask)
from sgi.vlm.points import POINT_FORMATS

_REASONING_STRATEGIES = ("cot", "few_shot_cot")    # their answers start with the reasoning fields
_LOOK_AT_MODE = "vlm_look_at"                      # the heading mode whose answer carries look_at_point
_ANALYSIS = "analysis"                             # the goal answer's reasoning: "reasoning" sorts after "goal_point"
# Its three parts, wire key to GoalReasoning field: "people" sorts after "expectations", and is written first. The
# constraints answer reasons in the same three parts.
_ANALYSIS_PARTS = {"arrangement": "people", "expectations": "expectations", "placement": "placement"}
# How a person faces as seen in the image, in a fixed order: the first eight go right round,
# a step of 45 degrees at a time, turning to the image's right. The order is part of the rule that reads them.
FACINGS = ("towards_camera", "towards_right", "right", "away_right", "away_from_camera", "away_left", "left",
           "towards_left", "unclear")


class PersonEntry(BaseModel):
    """One person of the people answer, as the model described them."""
    appearance: str                                # a few words that pick the person out
    body_point: Point                              # on the middle of the body
    faced_point: Point | None = None               # on what the person faces; None when that is not in view
    facing: Literal[FACINGS]


class PeopleAnswer(BaseModel):
    """The answer of the people stage of the constraint method."""
    reasoning: str | None = None
    people: list[PersonEntry]


class ConstraintsAnswer(BaseModel):
    """The answer of the constraints stage of the constraint method."""
    reasoning: GoalReasoning | None = None
    constraints: list[Constraint]                  # as the model stated them, in its order: the solver judges each


def _non_empty(value: str) -> str:
    """Reject the empty string. Any other reasoning is returned as written, never stripped and never judged."""
    if value == "":
        raise ValueError("must not be empty")                # this text reaches the model in the retry note (G2)
    return value


# A validator and not a length constraint, so the schema sent says only "string": OpenAI strict mode supports no
# "minLength" on a string, and every answer is validated with pydantic whatever the provider enforces.
_Reasoning = Annotated[str, AfterValidator(_non_empty)]

# Field descriptions are sent to the model, so they name no task and no situation (G2). The keys are wire keys.
# Each answer has its own: one key can stand in two answers and mean another thing in each, as `analysis` does,
# three parts in the goal answer and one string in the people answer.
# A reasoning field is described by what goes into it. Described as "your reasoning", it had Claude Sonnet 5.5 decline
# the request before writing anything (stop_details category reasoning_extraction), taken for a request for its
# internal reasoning.
_TARGET_POINT_DESC = "A point on the target if it is visible in the image, otherwise null."
_PLAN_DESC = {
    "reasoning": "What the request asks for and what in the scene can serve it, written before the subtasks.",
    "subtasks": "The ordered steps that carry out the request.",
    "description": "One short sentence saying what this step does.",
    "target": "For navigate: a few words that pick out the place, object or person to go to. For act: null.",
    "target_point": _TARGET_POINT_DESC,
    "tour_frame": "The index of the frame that shows the target best, or null.",
    "type": "navigate: go to a target. act: do something at the target of the preceding navigate step.",
}
_LOCATION_DESC = {"reasoning": "What in the image could be the target, written before the point.",
                  "target_point": _TARGET_POINT_DESC}
_CONTEXT_DESC = {
    "reasoning": ("What the people in view are doing and whether it concerns the target, written before the "
                  "answer."),
    "relevant_people": ("The people whose position, facing or activity affects where an arriving robot may stop "
                        "or what it would block."),
    "stop_elsewhere": "False only when stopping at the standard goal would be appropriate.",
}
_GOAL_DESC = {
    "arrangement": ("The people relevant to where the robot may stop, including the target if it is a person: "
                    "how they are arranged and what each person or group is doing."),
    "expectations": "What they would expect of someone arriving to do what the request and the subtask describe.",
    "placement": "Where to stand and face, and why.",
    "look_at_point": "The point the robot should face from where it stands, or null to face the target.",
}
_PEOPLE_DESC = {
    "analysis": ("How many people are in view, where each of them is and what shows the way each faces, written "
                 "before the list."),
    "people": "Every person visible in the image, one entry each.",
    "appearance": "A few words that pick this person out from the others in the image.",
    "faced_point": "A point on what this person faces if that is visible in the image, otherwise null.",
    "facing": "How this person faces as seen in the image; unclear when it cannot be told.",
}
_CONSTRAINTS_DESC = {
    "arrangement": ("The people relevant to where the robot may stop, each by the number drawn on them, including the "
                    "target if it is a person: how they are arranged and what each person or group is doing."),
    "expectations": _GOAL_DESC["expectations"],
    "placement": "Where the robot should stand and what it should face, in words, and why.",
    "constraints": "What the people in view require of the place where the robot stands, the most essential first.",
    "because": "One sentence: what about the people makes this apply.",
    "kind": "The kind of constraint.",
    "people": "The numbers of the people this is about, as drawn on them in the image; an empty list for nobody.",
    "range_m": "For within_distance: the least and the greatest distance, in metres. For every other kind: null.",
    "at_least": "The least distance, in metres.",
    "at_most": "The greatest distance, in metres.",
    "target": "True when this is about the target.",
}


def _wire_model(name: str, descriptions: Mapping[str, str], /, **annotations: Any) -> type[BaseModel]:
    """A wire model with its fields in the order given, all of them required, and no other key allowed.

    The order given is the order the answer is written in, and has to be the sorted one (the top of this module).
    `descriptions` holds what the answer the model belongs to says of its keys."""
    fields = {}
    for field, annotation in annotations.items():
        # A bare nested model is sent as a lone $ref, and OpenAI strict mode rejects a $ref with a sibling key.
        nested_model = isinstance(annotation, type) and issubclass(annotation, BaseModel)
        described = field in descriptions and not nested_model
        fields[field] = (annotation, Field(description=descriptions[field]) if described else ...)
    return create_model(name, __config__=ConfigDict(extra="forbid"), **fields)


_GoalReasoningWire = _wire_model("GoalReasoningWire", _GOAL_DESC, **dict.fromkeys(_ANALYSIS_PARTS, _Reasoning))
_ConstraintReasoningWire = _wire_model("ConstraintReasoningWire", _CONSTRAINTS_DESC,
                                       **dict.fromkeys(_ANALYSIS_PARTS, _Reasoning))


def _reasoning_field(strategy: Strategy, key: str = "reasoning", annotation: Any = _Reasoning) -> dict[str, Any]:
    """The leading reasoning field under a chain-of-thought strategy, nothing under the others.

    `key` has to sort before every other key of the answer it leads."""
    return {key: annotation} if strategy in _REASONING_STRATEGIES else {}


def _wire_point(point_format: str, size: tuple[int, int]) -> type[BaseModel]:
    return POINT_FORMATS[point_format].wire_model(size[0], size[1])


def plan_schema(strategy: Strategy, point_format: str, size: tuple[int, int],
                tour_indices: Sequence[int] | None = None) -> type[BaseModel]:
    """View mode grounds each subtask with `target_point`; tour mode (`tour_indices` given) with `tour_frame`."""
    if tour_indices is None:
        grounding = {"target_point": _wire_point(point_format, size) | None}
    elif len(tour_indices) == 0:
        raise ValueError("a tour-mode plan schema needs the index of at least one shown frame")
    else:
        grounding = {"tour_frame": Literal[tuple(tour_indices)] | None}
    subtask = _wire_model("SubtaskWire", _PLAN_DESC, description=str, target=str | None, **grounding,
                          type=Literal["navigate", "act"])
    return _wire_model("PlanWire", _PLAN_DESC, **_reasoning_field(strategy), subtasks=list[subtask])


def location_schema(strategy: Strategy, point_format: str, size: tuple[int, int]) -> type[BaseModel]:
    return _wire_model("LocationWire", _LOCATION_DESC, **_reasoning_field(strategy),
                       target_point=_wire_point(point_format, size) | None)


def context_schema(strategy: Strategy) -> type[BaseModel]:
    """`stop_elsewhere` is the stage output's `people_matter`, under a name that is written last."""
    return _wire_model("ContextWire", _CONTEXT_DESC, **_reasoning_field(strategy), relevant_people=str,
                       stop_elsewhere=bool)


def goal_schema(strategy: Strategy, point_format: str, size: tuple[int, int], heading_mode: str) -> type[BaseModel]:
    point = _wire_point(point_format, size)
    look_at = {"look_at_point": point | None} if heading_mode == _LOOK_AT_MODE else {}
    return _wire_model("GoalWire", _GOAL_DESC, **_reasoning_field(strategy, _ANALYSIS, _GoalReasoningWire),
                       goal_point=point, **look_at)


def people_schema(strategy: Strategy, point_format: str, size: tuple[int, int]) -> type[BaseModel]:
    """The people answer of the constraint method: every person in view, each with a few
    words that pick them out, a point on the body, a point on what they face or null, and how they face as seen
    in the image. Its reasoning is one string, `analysis`.

    The list has no bound, because a strict schema cannot hold one: the prompt asks for at most `max_people`."""
    point = _wire_point(point_format, size)
    person = _wire_model("PersonWire", _PEOPLE_DESC, appearance=str, body_point=point, faced_point=point | None,
                         facing=Literal[FACINGS])
    return _wire_model("PeopleWire", _PEOPLE_DESC, **_reasoning_field(strategy, _ANALYSIS), people=list[person])


def constraints_schema(strategy: Strategy) -> type[BaseModel]:
    """The constraints answer of the constraint method: a list of constraints, each with
    one sentence of why, one of the six kinds, the numbers of the people it is about, a range in metres or null,
    and whether it is about the target. Its reasoning, `analysis`, has the three parts of the goal answer's.

    Nothing here says what a kind reads or when a constraint can be used: the solver judges that, and a
    constraint it cannot use is recorded as such, not refused. The list has no bound, because a strict schema
    cannot hold one: the prompt asks for at most `max_constraints`, and the solver reads no more."""
    span = _wire_model("RangeWire", _CONSTRAINTS_DESC, at_least=float, at_most=float)
    constraint = _wire_model("ConstraintWire", _CONSTRAINTS_DESC, because=str, kind=ConstraintKind, people=list[int],
                             range_m=span | None, target=bool)
    return _wire_model("ConstraintsWire", _CONSTRAINTS_DESC,
                       **_reasoning_field(strategy, _ANALYSIS, _ConstraintReasoningWire), constraints=list[constraint])


def _to_point(wire_point: dict | None, point_format: str, size) -> Point | None:
    if wire_point is None:
        return None
    return POINT_FORMATS[point_format].to_internal(wire_point, size[0], size[1])


def _from_point(point: Point | None, point_format: str, size) -> dict | None:
    if point is None:
        return None
    return POINT_FORMATS[point_format].from_internal(point, size[0], size[1])


# The *_from_wire functions take an answer the client has already validated (VLMResponse.parsed).
# A key the schema did not have becomes None. A wire key named for its place in the order gets its own name back.

def plan_from_wire(wire: dict, point_format: str, size) -> Plan:
    subtasks = [Subtask(type=step["type"], description=step["description"], target=step["target"],
                        target_point=_to_point(step.get("target_point"), point_format, size),
                        tour_frame=step.get("tour_frame"))
                for step in wire["subtasks"]]
    return Plan(reasoning=wire.get("reasoning"), subtasks=subtasks)        # image_sha256 is the caller's to set


def location_from_wire(wire: dict, point_format: str, size) -> Location:
    return Location(reasoning=wire.get("reasoning"), target_point=_to_point(wire["target_point"], point_format, size))


def context_from_wire(wire: dict) -> ContextCheck:
    return ContextCheck(reasoning=wire.get("reasoning"), relevant_people=wire["relevant_people"],
                        people_matter=wire["stop_elsewhere"])


def _to_reasoning(analysis: dict | None) -> GoalReasoning | None:
    """The three parts of an `analysis`, each under its own name again; None for an answer without one."""
    if analysis is None:
        return None
    return GoalReasoning(**{field: analysis[key] for key, field in _ANALYSIS_PARTS.items()})


def goal_from_wire(wire: dict, point_format: str, size) -> GoalMark:
    return GoalMark(reasoning=_to_reasoning(wire.get(_ANALYSIS)),         # the three parts under their wire keys
                    goal_point=_to_point(wire["goal_point"], point_format, size),
                    look_at_point=_to_point(wire.get("look_at_point"), point_format, size))


def people_from_wire(wire: dict, point_format: str, size) -> PeopleAnswer:
    people = [PersonEntry(appearance=entry["appearance"], body_point=_to_point(entry["body_point"], point_format, size),
                          faced_point=_to_point(entry["faced_point"], point_format, size), facing=entry["facing"])
              for entry in wire["people"]]
    return PeopleAnswer(reasoning=wire.get(_ANALYSIS), people=people)


def _to_range(wire_range: dict | None) -> DistanceRange | None:
    """A range as written, or None for one that is no range: null, or with a side that is no finite number.

    JSON has no infinity, so the cache gives such a side back as null: read as no range either way, an answer
    and its replay from the cache are one. The solver calls a `within_distance` without a range unusable."""
    sides = None if wire_range is None else (wire_range["at_least"], wire_range["at_most"])
    if sides is None or not all(side is not None and math.isfinite(side) for side in sides):
        return None
    return DistanceRange(at_least=sides[0], at_most=sides[1])


def constraints_from_wire(wire: dict) -> ConstraintsAnswer:
    """Each constraint as the model wrote it, in its order. A number that no person has, a repeated one and a
    range the wrong way round are kept: what a constraint stands on is the solver's to judge."""
    constraints = [Constraint(because=stated["because"], kind=stated["kind"], people=stated["people"],
                              range_m=_to_range(stated["range_m"]), target=stated["target"])
                   for stated in wire["constraints"]]
    return ConstraintsAnswer(reasoning=_to_reasoning(wire.get(_ANALYSIS)), constraints=constraints)


def goal_to_wire(mark: GoalMark, strategy: Strategy, point_format: str, size, heading_mode: str) -> dict:
    """The answer `goal_schema` with the same arguments asks for: its keys in its order, anything else dropped."""
    wire: dict[str, Any] = {}
    if strategy in _REASONING_STRATEGIES:
        if mark.reasoning is None:
            raise ValueError(f"strategy {strategy!r} needs a goal mark with reasoning")
        wire[_ANALYSIS] = {key: getattr(mark.reasoning, field) for key, field in _ANALYSIS_PARTS.items()}
    wire["goal_point"] = _from_point(mark.goal_point, point_format, size)
    if heading_mode == _LOOK_AT_MODE:
        wire["look_at_point"] = _from_point(mark.look_at_point, point_format, size)
    return wire
