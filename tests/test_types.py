import json
import typing

import pytest
from pydantic import ValidationError

from sgi.types import (Call, Candidate, Constraint, ConstraintUse, ContextCheck, DetectedPerson, Flags, GoalMark,
                       GoalReasoning, GoalResult, GroundedTarget, Keys, Location, PlanOutcome, Point, SolverNote, Subtask,
                       Usage)

CALL = Call(stage="goal", provider="mock", model="mock-centre", reasoning_effort="low", point_format="xy_1000",
            image_size=(640, 480), latency_s=0.5, latency_total_s=0.75, validation_retries=0,
            transient_retries=1, cached=False, usage=Usage(input_tokens=10, output_tokens=4))
BASELINE_FLAGS = dict(marked_point_off_floor=False, standard_goal_blocked=False, standard_goal_frontal=False,
                      heading_fallback=False, heading_degenerate=False)
NO_FLAGS = dict(**BASELINE_FLAGS, no_usable_constraint=False, constraint_unusable=False, constraints_cut=False,
                exclusion_broken=False, people_cut=False, far_from_target=False)       # the six of the constraint method
METHOD_FIELDS = ("people", "constraints", "candidates", "solver")        # of a GoalResult: None outside the constraint method
KINDS = ("keep_out_of_shared_space", "keep_out_of_view", "come_after", "within_distance", "visible_to", "face")
KEYS = Keys(excluded_m=0.0, excluded_total_m=0.0, turn_m=0.25, preference_m=0.125, standard_m=1.5)
RANGED = Constraint(because="the two of them are talking", kind="within_distance", people=[2],
                    range_m={"at_least": 0.5, "at_most": 1.5}, target=True)


def test_point_is_an_integer_pair_in_0_to_1000():
    assert [Point(x=x, y=y).model_dump() for x, y in ((0, 1000), (1000, 0))] == [{"x": 0, "y": 1000}, {"x": 1000, "y": 0}]
    for bad in ({"x": -1, "y": 0}, {"x": 1001, "y": 0}, {"x": 0, "y": -1}, {"x": 0, "y": 1001}, {"x": 499.6, "y": 0}):
        with pytest.raises(ValidationError):
            Point(**bad)


def test_goal_result_defaults_literals_and_independent_instances():
    r = GoalResult(status="no_target")
    assert r.model_dump() == dict(
        status="no_target", source=None, pose_robot=None, pose_world=None, goal_pixel=None, target=None,
        flags=NO_FLAGS, calls=[], timings_s={}, location=None, context_check=None, goal_mark=None,
        **dict.fromkeys(METHOD_FIELDS))
    r.flags.heading_fallback = True
    r.calls.append(CALL)
    r.timings_s["goal"] = 0.75
    other = GoalResult(status="ok")
    assert (other.flags, other.calls, other.timings_s) == (Flags(), [], {})        # defaults are not shared
    assert CALL.status == "ok"                                     # CALL passes no status: this is the default
    assert PlanOutcome(status="ok").tour_indices is None           # only a tour-mode plan sets it
    for build in (lambda: GoalResult(status="failed"), lambda: PlanOutcome(status="no_target"),
                  lambda: Subtask(type="wait", description="x"), lambda: Call(**{**CALL.model_dump(), "stage": "tour"})):
        with pytest.raises(ValidationError):
            build()


def test_goal_result_json_round_trip():
    r = GoalResult(
        status="ok", source="direct_marking", pose_robot=(2.5, -0.5, 0.25), pose_world=(1.5, 4.5, 1.75),
        goal_pixel=(256.0, 384.0), flags=Flags(heading_fallback=True), calls=[CALL],
        target=GroundedTarget(point=Point(x=500, y=250), pixel=(320.0, 120.0), world=(1.0, 6.0, 1.0)),
        timings_s={"context": 0.5, "goal": 0.75, "geometry": 0.01}, location=Location(target_point=Point(x=500, y=250)),
        context_check=ContextCheck(relevant_people="two people at the desk", people_matter=True),
        goal_mark=GoalMark(reasoning=GoalReasoning(people="a", expectations="b", placement="c"),
                           goal_point=Point(x=400, y=800)))
    text = r.model_dump_json()
    back = GoalResult.model_validate_json(text)
    assert back == r
    assert back.pose_robot == (2.5, -0.5, 0.25) and back.calls[0].image_size == (640, 480)      # tuples again
    data = json.loads(text)
    assert data["pose_robot"] == [2.5, -0.5, 0.25] and data["flags"] == {**NO_FLAGS, "heading_fallback": True}
    assert data["calls"][0]["raw_text"] is None and data["goal_mark"]["look_at_point"] is None


def test_the_types_of_the_constraint_method_and_their_defaults():
    person = DetectedPerson(number=1, xy=(1.0, 4.5), pixel=(320.0, 200.0))
    assert person.model_dump() == dict(number=1, xy=(1.0, 4.5), yaw=None, pixel=(320.0, 200.0), label=None)
    bare = Constraint(because="they wait their turn", kind="come_after")
    assert bare.model_dump() == dict(because="they wait their turn", kind="come_after", people=[], range_m=None, target=False)
    assert (RANGED.range_m.at_least, RANGED.range_m.at_most) == (0.5, 1.5)
    backwards = Constraint(because="x", kind="within_distance", range_m={"at_least": 2.0, "at_most": -1.0})
    assert backwards.range_m.model_dump() == {"at_least": 2.0, "at_most": -1.0}      # kept as written: the solver judges it
    used = ConstraintUse(constraint=bare, status="used")
    assert used.model_dump(exclude={"constraint"}) == dict(status="used", reason=None, broken_m=0.0)
    assert Keys(excluded_m=0.0, excluded_total_m=0.0, preference_m=0.125, standard_m=1.5).turn_m is None     # no come_after
    assert Keys(excluded_m=0.0, excluded_total_m=0.0, preference_m=0.125, standard_m=1.5).face_m is None     # nor a seen face
    assert list(Keys.model_fields) == ["excluded_m", "excluded_total_m", "turn_m", "face_m", "preference_m", "standard_m"]
    note = SolverNote(keys=KEYS, target_distance_m=1.0)
    assert (note.own_space_broken_m, note.joined_waiting, note.not_ahead_of) == ({}, [], [])
    note.joined_waiting.append(3)
    note.not_ahead_of.append(2)
    assert SolverNote(keys=KEYS, target_distance_m=1.0).joined_waiting == [] and Constraint(because="x", kind="face").people == []
    assert SolverNote(keys=KEYS, target_distance_m=1.0).not_ahead_of == [] and note.not_ahead_reason is None
    assert list(SolverNote.model_fields) == ["keys", "own_space_broken_m", "joined_waiting", "not_ahead_of", "not_ahead_reason",
                                             "target_distance_m"]
    assert typing.get_args(Constraint.model_fields["kind"].annotation) == KINDS
    assert typing.get_args(ConstraintUse.model_fields["status"].annotation) == ("used", "unusable", "superseded", "cut")
    for stage in ("people", "constraints"):                            # a call of either new stage is a Call like any other
        assert Call(**{**CALL.model_dump(), "stage": stage}).stage == stage
    for build in (lambda: Constraint(because="x", kind="keep_clear_of"), lambda: Constraint(kind="face"),
                  lambda: ConstraintUse(constraint=bare, status="ignored"), lambda: DetectedPerson(number=1, xy=(1.0, 4.5)),
                  lambda: Constraint(because="x", kind="within_distance", range_m={"at_least": 0.5}),
                  lambda: Keys(excluded_m=0.0, excluded_total_m=0.0, preference_m=0.0),
                  lambda: Candidate(pose_world=(1.0, 2.0), keys=KEYS), lambda: SolverNote(keys=KEYS)):
        with pytest.raises(ValidationError):
            build()


def test_a_constraint_result_round_trips_through_json_and_an_old_record_still_loads():
    r = GoalResult(
        status="ok", source="constraints", pose_robot=(2.5, -0.5, 0.25), pose_world=(1.5, 4.5, 1.75),
        flags=Flags(exclusion_broken=True, people_cut=True), calls=[Call(**{**CALL.model_dump(), "stage": "people"})],
        people=[DetectedPerson(number=1, xy=(1.0, 4.5), yaw=-1.5, pixel=(300.0, 210.0), label="red coat"),
                DetectedPerson(number=2, xy=(2.0, 4.5), pixel=(400.5, 215.0))],
        constraints=[ConstraintUse(constraint=RANGED, status="used", broken_m=0.125),
                     ConstraintUse(constraint=Constraint(because="b", kind="face", people=[7]), status="unusable",
                                   reason="none of the people listed is known")],
        candidates=[Candidate(pose_world=(2.5, 3.5, 0.5), keys=KEYS)],
        solver=SolverNote(keys=KEYS, own_space_broken_m={2: 0.125}, joined_waiting=[3], target_distance_m=1.25))
    text = r.model_dump_json()
    back = GoalResult.model_validate_json(text)
    assert back == r == GoalResult.model_validate(json.loads(json.dumps(r.model_dump(mode="json"))))     # as results.jsonl is written
    assert back.people[0].xy == (1.0, 4.5) and back.candidates[0].pose_world == (2.5, 3.5, 0.5)         # tuples again
    assert back.solver.own_space_broken_m == {2: 0.125}                # and a person's number is an integer again
    data = json.loads(text)
    assert data["solver"]["own_space_broken_m"] == {"2": 0.125} and data["calls"][0]["stage"] == "people"
    said = data["constraints"][0]["constraint"]
    assert list(said) == ["because", "kind", "people", "range_m", "target"] == sorted(said)     # the order it is written in
    assert said["range_m"] == {"at_least": 0.5, "at_most": 1.5} and data["people"][1]["yaw"] is None
    assert data["flags"] == {**NO_FLAGS, "exclusion_broken": True, "people_cut": True}
    old = {**{k: v for k, v in data.items() if k not in METHOD_FIELDS}, "flags": BASELINE_FLAGS, "calls": []}
    loaded = GoalResult.model_validate(old)                            # a record written before the method existed
    assert [getattr(loaded, name) for name in METHOD_FIELDS] == [None] * 4 and loaded.flags == Flags()
    assert loaded.pose_world == r.pose_world and loaded.model_dump()["flags"] == NO_FLAGS
