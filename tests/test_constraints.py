import itertools
import json
import math
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from sgi import solver
from sgi.config import PROMPTS_DIR, ModelEntry, Thresholds
from sgi.geometry import in_image, pixel_to_point, project
from sgi.gt.derive import oracle_target
from sgi.imaging import draw_people_numbers, draw_target_marker, encode_png
from sgi.prompts import build_constraints_request, people_lines
from sgi.run import OraclePeople
from sgi.scene import load_scene, save_scene
from sgi.schemas import constraints_from_wire, constraints_schema
from sgi.stages.constraints import ConstraintInferrer, numbered_image
from sgi.stages.goal import standard_goal
from sgi.stages.people import FoundPerson, PeopleFound, VlmPeople, number_people
from sgi.synth.generate import build_scene
from sgi.synth.recipes import recipe_waiting_line
from sgi.synth.render import Box
from sgi.types import Call, Constraint, ContextCheck, Flags, GroundedTarget, Point, Subtask, Usage
from sgi.vlm.base import ImagePart, TextPart, TransientError, VLMClient
from sgi.vlm.cache import Cache
from sgi.vlm.mock import MockBackend
from tests.solver_layouts import HERE, TH, layout, mounted, open_floor

SHARED, VIEW, AFTER, WITHIN, VISIBLE, FACE = ("keep_out_of_shared_space", "keep_out_of_view", "come_after",
                                              "within_distance", "visible_to", "face")
USED, NOT = ("used", None), "unusable"
UNKNOWN = "none of its numbers is in the list of people"         # the solver's own words for what it cannot use
TOO_FEW = "fewer than two people are listed"
NO_DIRECTION = "no listed person gives a direction"
ONLY_TARGET = "nobody is listed but the target person"
NO_RANGE = "no range of two distances, neither of them negative"
TOO_FAR = "range too far"
NEITHER = "neither a person nor the target"
NO_HEADING = "no listed person has a heading"


def written(kind, *people, target=False, between=None, because="what the people do"):
    """One constraint as an answer writes it. `between` is (at least, at most) in metres."""
    range_m = None if between is None else {"at_least": between[0], "at_most": between[1]}
    return {"because": because, "kind": kind, "people": list(people), "range_m": range_m, "target": target}


def trio():
    """Person 1 with a heading, person 2 without, and person 3, who is the target."""
    return layout(HERE, [(4.0, 1.0, math.pi), (4.0, -1.0), (6.0, 0.0, math.pi)], needed=3)


def read(*constraints, th=TH):
    """What the solver makes of each constraint of an answer: through the schema the answer is asked with, as a
    client reads it, and through `constraints_from_wire`."""
    wire = constraints_schema("zero_shot").model_validate({"constraints": list(constraints)}).model_dump(mode="json")
    return [(use.status, use.reason) for use in trio().solve(*constraints_from_wire(wire).constraints, th=th).uses]


# -------------------------------------------------------------- the table of constraints, from the answer as written
TABLE = [
    # keep_out_of_shared_space reads people, two or more
    (written(SHARED, 1, 2), USED), (written(SHARED, 3, 1, target=True, between=(0.0, 0.1)), USED),      # the rest is ignored
    (written(SHARED, 1), (NOT, TOO_FEW)), (written(SHARED), (NOT, TOO_FEW)),
    # keep_out_of_view reads people and target, and needs one who gives a direction
    (written(VIEW, 1), USED), (written(VIEW, 2, target=True), USED), (written(VIEW, 1, between=(9.0, 9.0)), USED),
    (written(VIEW, 2), (NOT, NO_DIRECTION)),                         # straight ahead of someone whose heading is not known
    (written(VIEW, 3, target=True), (NOT, NO_DIRECTION)),            # towards the target, of the person who is the target
    # come_after reads people, in any order
    (written(AFTER, 2, 1), USED), (written(AFTER, 1, 3, target=False), USED),
    (written(AFTER, 3), (NOT, ONLY_TARGET)), (written(AFTER, target=True), (NOT, ONLY_TARGET)),
    # within_distance reads people, else the target, and a range
    (written(WITHIN, 1, between=(0.5, 1.5)), USED), (written(WITHIN, target=True, between=(0.5, 1.5)), USED),
    (written(WITHIN, 1, between=(1.5, 0.5)), USED),                  # at_least above at_most: the two are exchanged
    (written(WITHIN, 1, between=(0.0, 0.0)), USED), (written(WITHIN, 1, between=(2.9, 9.0)), USED),
    (written(WITHIN, 1), (NOT, NO_RANGE)),                           # the range is null,
    (written(WITHIN, 1, between=(-0.5, 1.0)), (NOT, NO_RANGE)),      # negative,
    (written(WITHIN, 1, between=(0.5, float("inf"))), (NOT, NO_RANGE)),      # or not finite
    (written(WITHIN, 1, between=(float("nan"), 1.0)), (NOT, NO_RANGE)),
    (written(WITHIN, 1, between=(3.0, 4.0)), (NOT, TOO_FAR)),        # it starts at max_keep_m,
    (written(WITHIN, target=True, between=(7.0, 5.0)), (NOT, TOO_FAR)),      # also once its sides are exchanged
    (written(WITHIN, between=(0.5, 1.5)), (NOT, NEITHER)),
    # visible_to reads people, and needs one with a heading
    (written(VISIBLE, 1), USED), (written(VISIBLE, 2, 1, target=True), USED), (written(VISIBLE, 2), (NOT, NO_HEADING)),
    (written(VISIBLE), (NOT, NO_HEADING)),
    # face reads people, else the target
    (written(FACE, 2), USED), (written(FACE, 1, 2, 3), USED), (written(FACE, target=True), USED),
    (written(FACE), (NOT, NEITHER)),
    # a number that is not in the list is left out, and a constraint that names no other is about nobody the solver has
    (written(VISIBLE, 1, 9), USED), (written(SHARED, 1, 9), (NOT, TOO_FEW)), (written(SHARED, 2, 2, 1), USED),
    (written(FACE, 8, 9, target=True), (NOT, UNKNOWN)), (written(WITHIN, 0, target=True, between=(0.5, 1.5)), (NOT, UNKNOWN)),
]


@pytest.mark.parametrize("constraint, becomes", TABLE, ids=[f"{row[0]['kind']} {n}" for n, row in enumerate(TABLE)])
def test_each_row_of_the_table_is_read_from_an_answer_as_the_design_has_it(constraint, becomes):
    assert read(constraint) == [becomes]


def test_the_first_usable_face_stands_and_what_is_beyond_the_bound_is_cut():
    assert read(written(FACE), written(FACE, 1), written(FACE, target=True), written(VISIBLE, 1), written(FACE, 2)) == [
        (NOT, NEITHER), USED, ("superseded", "an earlier face stands"), USED, ("superseded", "an earlier face stands")]
    few = Thresholds(max_constraints=2)
    assert read(written(AFTER, 1), written(AFTER, 3), written(AFTER, 2), written(FACE, 9), th=few) == [
        USED, (NOT, ONLY_TARGET), ("cut", "beyond the first 2"), ("cut", "beyond the first 2")]
    assert len(read(*[written(VISIBLE, 1)] * 13)) == 13 and read(*[written(VISIBLE, 1)] * 13)[11:] == [
        USED, ("cut", "beyond the first 12")]                        # twelve are read, as the threshold has it


# ------------------------------------------------------------------------------------------------------ the inferrer
REQUEST = "hand this over at the front"
SUBTASK = Subtask(type="navigate", description="Go to the front", target="the front")
FLOOR = open_floor(HERE)                           # a bare floor seen from over the robot, which looks along +x
A = (4.0, 2.5)                                     # where the thing the robot was sent to stands
ROW = [(4.0, -0.5), (4.0, 1.5, -math.pi / 2), (4.0, 0.5, -math.pi / 2)]      # three who lead away from it, as a source finds them
AFTER_ALL = {"constraints": [written(AFTER, 3, 1, 2, because="They stand one behind another before it.")]}
NOTHING = {"constraints": []}


def thing(x, y, z=1.0, observation=FLOOR):
    """A target whose point was marked on something at (x, y, z)."""
    u, v, _ = project(observation.camera, (x, y, z))
    return GroundedTarget(point=Point(x=500, y=500), pixel=(u, v), world=(x, y, z))


def someone(x, y, yaw=None, label=None, observation=FLOOR):
    """A person as a source finds them, marked at chest height."""
    return FoundPerson(xy=(x, y), yaw=yaw, pixel=project(observation.camera, (x, y, 1.1))[:2], label=label)


def given(*spots, calls=(), call_time=0.0, geometry_time=0.0):
    """A people source that asks no model: it finds these people, and reports these calls and times as its own."""
    found = PeopleFound(found=[someone(*spot) for spot in spots], calls=list(calls), call_time_s=call_time,
                        geometry_time_s=geometry_time)
    return lambda observation: found


def client_for(answer, cache=None, point_format="xy_1000"):
    """A client on a mock that gives `answer`, with a clock that jumps 60 s per read, and the back end behind it."""
    backend = MockBackend(answer if callable(answer) else lambda parts, json_schema: answer)
    entry = ModelEntry(provider="mock", model="mock", point_format=point_format)
    return VLMClient(entry, backend, cache, sleep=lambda s: None, clock=itertools.count(0, 60).__next__), backend


def infer(answer, source=None, target=None, observation=FLOOR, th=TH, strategy="zero_shot", check=None, **given_):
    """One inference on a mock that gives `answer` to the constraints request: the result and the back end."""
    client, backend = client_for(answer, given_.pop("cache", None))
    inferrer = ConstraintInferrer(source or given(*ROW), client, strategy, th, **given_)
    return inferrer.infer(observation, REQUEST, SUBTASK, target or thing(*A), check), backend


def numbered(spots, target=None, th=TH, observation=FLOOR):
    """The people of `spots` as the method numbers them."""
    found = [someone(*spot, observation=observation) for spot in spots]
    return number_people(found, target or thing(*A), observation, th, th.max_people)[0]


def solved(constraints, people, target=None, observation=FLOOR, th=TH):
    """What the solver gives for these constraints, written as an answer writes them, with the standard goal."""
    target = target or thing(*A)
    stated = constraints_from_wire({"constraints": list(constraints)}).constraints
    return solver.solve(observation, target, people, stated, standard_goal(observation, target, th), th)


def raised(result):
    return {name for name, on in result.flags.model_dump().items() if on}


def test_the_people_get_numbers_the_model_states_constraints_and_the_solver_finds_the_place():
    result, backend = infer(AFTER_ALL)
    people = numbered(ROW)
    assert [(person.number, person.xy) for person in people] == [(1, (4.0, 1.5)), (2, (4.0, 0.5)), (3, (4.0, -0.5))]     # from the image's left
    assert result.people == people and result.people[0].yaw == pytest.approx(-math.pi / 2) and result.people[2].yaw is None
    expected = solved(AFTER_ALL["constraints"], people)
    assert (result.status, result.source, result.target) == ("ok", "constraints", thing(*A))
    assert (result.pose_world, result.pose_robot) == (expected.pose_world, expected.pose_robot)
    assert math.dist(result.pose_world[:2], (4.0, -1.5)) <= 0.05 + 1e-9             # a metre beyond the last of them, to a cell,
    assert result.pose_world[2] == pytest.approx(math.atan2(1.0, 4.0 - result.pose_world[0]))      # and facing that person
    assert result.constraints == expected.uses and [(use.status, use.broken_m) for use in result.constraints] == [("used", 0.0)]
    assert result.constraints[0].constraint == Constraint(because="They stand one behind another before it.", kind="come_after",
                                                          people=[3, 1, 2])          # as it was written
    assert result.candidates == expected.candidates and len(result.candidates) == TH.candidate_count
    assert result.solver == expected.note and result.solver.keys.turn_m <= 0.05 and result.solver.joined_waiting == []
    assert result.flags == expected.flags == Flags() and result.goal_mark is None and result.location is None
    u, v, _ = project(FLOOR.camera, (*result.pose_world[:2], 0.0))                  # the place, where the picture shows it
    assert result.goal_pixel == (u, v) and in_image(FLOOR.camera, u, v)
    assert [(call.stage, call.status) for call in result.calls] == [("constraints", "ok")] and len(backend.calls) == 1
    assert set(result.timings_s) == {"goal", "geometry"}
    assert result.timings_s["goal"] >= result.calls[0].latency_total_s >= 60.0 and 0.0 < result.timings_s["geometry"] < 60.0


@pytest.mark.parametrize("strategy, check, max_side, size", [
    ("zero_shot", None, None, (501, 501)),
    ("cot", ContextCheck(relevant_people="three people one behind another", people_matter=True), 200, (200, 200))])
def test_the_request_is_the_built_one_on_the_image_with_the_target_ringed_and_the_numbers(strategy, check, max_side, size):
    answer = {**AFTER_ALL, **({"analysis": {"arrangement": "a", "expectations": "b", "placement": "c"}} if strategy == "cot" else {})}
    th = Thresholds(standoff_m=0.8, max_constraints=5)
    result, backend = infer(answer, th=th, strategy=strategy, check=check, max_image_side=max_side)
    people, target = numbered(ROW, th=th), thing(*A)
    system, parts, schema = backend.calls[0]
    drawn = numbered_image(FLOOR, target, people)
    assert np.array_equal(drawn, draw_people_numbers(draw_target_marker(FLOOR.rgb, *target.pixel), people, (0.0, 0.0)))
    assert not np.array_equal(drawn, draw_target_marker(FLOOR.rgb, *target.pixel)) and not np.array_equal(drawn, FLOOR.rgb)
    image = encode_png(drawn, max_side)
    said = None if check is None else check.relevant_people           # only what a model said of this image in the context stage
    listed = people_lines(people, target, FLOOR, th)
    assert (system, parts) == build_constraints_request(strategy, REQUEST, SUBTASK, image, listed, said, 0.8, 5)
    assert parts[-1] == image and (image.width, image.height) == size == result.calls[0].image_size
    assert schema == constraints_schema(strategy).model_json_schema()
    text = parts[0].text
    assert listed.count("\n") == 2 and listed in text and ("three people one behind another" in text) is (check is not None)
    assert "at most 5 constraints" in text and "at least 0.8 m from every person" in text
    assert result.status == "ok" and result.pose_world == solved(AFTER_ALL["constraints"], people, th=th).pose_world


def test_the_times_of_the_people_source_are_added_and_its_calls_come_first():
    asked = Call(stage="people", provider="mock", model="mock", reasoning_effort="low", point_format="xy_1000",
                 latency_s=7.0, latency_total_s=7.0, validation_retries=0, transient_retries=0, cached=False, usage=Usage())
    result, _ = infer(AFTER_ALL, source=given(*ROW, calls=[asked], call_time=1000.0, geometry_time=500.0))
    assert [call.stage for call in result.calls] == ["people", "constraints"] and result.calls[0] == asked
    spent = result.calls[1].latency_total_s                           # the constraints call, by the fake clock: 60 s or more
    assert 1000.0 + spent <= result.timings_s["goal"] < 1000.0 + spent + 60.0       # the two calls count under goal,
    assert 500.0 < result.timings_s["geometry"] < 560.0               # placing the people and the solver under geometry


@pytest.mark.parametrize("answer, status", [("not json", "parse_failed"), ({"constraints": [written("keep_clear_of", 1)]}, "parse_failed"),
                                            (TransientError("down"), "vlm_error")])
def test_a_constraints_call_without_a_valid_answer_ends_with_its_status_and_no_pose(answer, status):
    result, backend = infer(answer, source=given(*ROW[:2], (9.0, 0.0)), th=Thresholds(max_people=2))
    assert (result.status, result.source, result.pose_world, result.pose_robot, result.goal_pixel) == (status, "constraints", None, None, None)
    assert [(call.stage, call.status) for call in result.calls] == [("constraints", status)] and len(backend.calls) >= 1
    assert result.people == numbered([*ROW[:2], (9.0, 0.0)], th=Thresholds(max_people=2)) and len(result.people) == 2      # known by then,
    assert raised(result) == {"people_cut"}                           # and that one was cut
    assert (result.constraints, result.candidates, result.solver) == (None, None, None)
    assert result.timings_s["goal"] >= result.calls[0].latency_total_s >= 60.0 and 0.0 <= result.timings_s["geometry"] < 60.0


@pytest.mark.parametrize("status", ["parse_failed", "vlm_error"])
def test_a_people_call_without_a_valid_answer_ends_with_its_status_before_anything_is_asked(status):
    failed = Call(stage="people", provider="mock", model="mock", reasoning_effort="low", point_format="xy_1000", latency_s=7.0,
                  latency_total_s=7.0, validation_retries=0, transient_retries=0, cached=False, usage=Usage(), status=status)
    result, backend = infer(AFTER_ALL, source=given(calls=[failed], call_time=7.5))
    assert (result.status, result.source, result.pose_world, result.pose_robot) == (status, "constraints", None, None)
    assert result.calls == [failed] and backend.calls == []           # nobody found is not nobody in view: no constraints call
    assert (result.people, result.constraints, result.candidates, result.solver) == (None, None, None, None)
    assert result.timings_s == {"goal": 7.5, "geometry": 0.0} and raised(result) == set()


def test_an_answer_with_no_usable_constraint_is_no_failure_and_the_solver_runs_on_what_it_applies_by_itself():
    people = numbered(ROW)
    alone = solved([], people)
    assert math.dist(alone.pose_world[:2], A) == pytest.approx(1.0, abs=0.05) and raised(alone) == {"no_usable_constraint"}
    empty, _ = infer(NOTHING)
    assert (empty.status, empty.source, empty.pose_world) == ("ok", "constraints", alone.pose_world)
    assert empty.constraints == [] and raised(empty) == {"no_usable_constraint"} and empty.solver == alone.note
    useless = [written(AFTER, 7, 8), written(SHARED, 2), written(WITHIN, 1)]
    result, _ = infer({"constraints": useless})
    assert (result.status, result.pose_world) == ("ok", alone.pose_world)
    assert raised(result) == {"no_usable_constraint", "constraint_unusable"}
    assert [(use.status, use.reason) for use in result.constraints] == [(NOT, UNKNOWN), (NOT, TOO_FEW), (NOT, NO_RANGE)]
    assert [use.constraint.model_dump() for use in result.constraints] == useless      # each recorded as it was written
    mixed, _ = infer({"constraints": [written(SHARED, 2), *AFTER_ALL["constraints"]]})     # one that cannot be used beside one that can
    assert raised(mixed) == {"constraint_unusable"} and mixed.pose_world == solved(AFTER_ALL["constraints"], people).pose_world


def test_constraints_beyond_the_bound_are_cut_and_the_flag_says_so():
    stated = [written(FACE, 2), written(VISIBLE, 1), written(AFTER, 1, 2, 3), written(WITHIN, 3, between=(1.0, 1.2))]
    th = Thresholds(max_constraints=2)
    result, backend = infer({"constraints": stated}, th=th)
    assert [use.status for use in result.constraints] == ["used", "used", "cut", "cut"] and raised(result) == {"constraints_cut"}
    read, whole = solved(stated[:2], numbered(ROW)).pose_world, solved(stated, numbered(ROW)).pose_world
    assert result.pose_world == read and math.dist(read[:2], whole[:2]) > 2.0       # the turn that was cut is not taken
    assert "at most 2 constraints" in backend.calls[0][1][0].text
    assert raised(infer({"constraints": stated})[0]) == set()         # room for all four: none is cut


def test_people_beyond_the_bound_are_cut_before_they_are_numbered_and_the_flag_says_so():
    crowd = [*ROW, (9.0, -3.0), (2.0, 4.0)]                           # five found; the last two stand 7.4 and 2.5 m from the target
    th = Thresholds(max_people=4)
    result, backend = infer(NOTHING, source=given(*crowd), th=th)
    assert [person.xy for person in result.people] == [(2.0, 4.0), (4.0, 1.5), (4.0, 0.5), (4.0, -0.5)]     # the farthest is left out
    assert [person.number for person in result.people] == [1, 2, 3, 4] and raised(result) == {"people_cut", "no_usable_constraint"}
    text = backend.calls[0][1][0].text
    assert [line.split(":")[0] for line in text.splitlines() if line.startswith("Person ")] == [f"Person {n}" for n in (1, 2, 3, 4)]
    assert result.pose_world == solved([], result.people, th=th).pose_world
    assert raised(infer(NOTHING, source=given(*crowd))[0]) == {"no_usable_constraint"}       # room for all five


def test_nobody_in_view_is_still_asked_about_the_target_and_a_place_under_the_picture_has_no_pixel():
    box = Box(cx=3.3, cy=0.0, yaw=math.pi, sx=0.6, sy=1.0, sz=1.2)    # a thing 3 m ahead of the camera of the placeholder scenes
    seen = mounted(HERE, boxes=[box])
    target = thing(3.0, 0.0, 0.8, observation=seen)
    near = {"constraints": [written(WITHIN, target=True, between=(0.9, 1.1)), written(FACE, target=True)]}
    result, backend = infer(near, source=given(), target=target, observation=seen)
    text = backend.calls[0][1][0].text
    assert "No person was found in view" in text and "Person" not in text and result.people == []
    assert (result.status, [use.status for use in result.constraints], raised(result)) == ("ok", ["used", "used"], set())
    assert result.pose_world == pytest.approx((2.0, 0.0, 0.0), abs=0.051)           # a metre before it, on floor under the picture
    assert not in_image(seen.camera, *project(seen.camera, (*result.pose_world[:2], 0.0))[:2]) and result.goal_pixel is None


def test_without_a_constraints_call_the_solver_runs_alone_on_the_people_found():
    people = numbered(ROW)
    for client in (None, client_for(AFTER_ALL)[0]):                   # no model is asked, whether or not there is a client
        result = ConstraintInferrer(given(*ROW), client, "cot", TH, constraints_mode="none").infer(FLOOR, REQUEST, SUBTASK, thing(*A), None)
        assert (result.status, result.source, result.calls, result.people) == ("ok", "constraints", [], people)
        assert result.pose_world == solved([], people).pose_world and result.constraints == []
        assert raised(result) == {"no_usable_constraint"} and len(result.candidates) == TH.candidate_count
        assert result.timings_s["goal"] == 0.0 and result.timings_s["geometry"] > 0.0
    client, backend = client_for(AFTER_ALL)
    ConstraintInferrer(given(*ROW), client, "cot", TH, constraints_mode="none").infer(FLOOR, REQUEST, SUBTASK, thing(*A), None)
    assert backend.calls == []


def test_the_prompt_is_read_from_the_folder_the_inferrer_is_given(tmp_path):
    folder = shutil.copytree(PROMPTS_DIR, tmp_path / "mine")
    own = (folder / "constraints.md").read_text().replace("Refer to a person by that number", "Name a person by that number")
    (folder / "constraints.md").write_text(own)
    _, backend = infer(AFTER_ALL, prompts_dir=folder)
    assert "Name a person by that number" in backend.calls[0][1][0].text
    assert "Refer to a person by that number" in infer(AFTER_ALL)[1].calls[0][1][0].text     # the repository's, by default


def test_the_trial_reaches_the_constraints_request_and_a_cached_answer_gives_the_same_place(tmp_path):
    asked = []

    def generate(parts, schema, system=None, trial=0):
        asked.append(trial)
        return real.generate(parts, schema, system=system, trial=trial)

    real, _ = client_for(AFTER_ALL)
    watched = SimpleNamespace(entry=real.entry, generate=generate)    # all an inferrer may use of a client
    for trial in ({"trial": 3}, {}):
        ConstraintInferrer(given(*ROW), watched, "zero_shot", TH, **trial).infer(FLOOR, REQUEST, SUBTASK, thing(*A), None)
    assert asked == [3, 0]
    cache = Cache(tmp_path)
    runs = [infer(AFTER_ALL, cache=cache, trial=trial)[0] for trial in (0, 0, 1)]
    assert [run.calls[0].cached for run in runs] == [False, True, False]             # the trial is part of the cache key
    assert runs[0].pose_world == runs[1].pose_world == runs[2].pose_world and runs[0].constraints == runs[1].constraints
    stored = runs[0].calls[0].latency_total_s                                        # fake-clock time: 60 s or more
    assert stored >= 60.0 and all(stored <= run.timings_s["goal"] < stored + 60.0 for run in runs[:2])


def test_a_range_that_is_not_finite_gives_the_same_record_fresh_and_from_the_cache(tmp_path):
    text = ('{"constraints": [{"because": "x", "kind": "within_distance", "people": [3], '
            '"range_m": {"at_least": 0.5, "at_most": 1e999}, "target": false}, '
            '{"because": "y", "kind": "come_after", "people": [1, 2, 3], "range_m": null, "target": false}]}')
    cache = Cache(tmp_path)
    fresh, replayed = (infer(text, cache=cache)[0] for _ in range(2))
    assert (fresh.calls[0].cached, replayed.calls[0].cached) == (False, True) and fresh.calls[0].raw_text == text
    assert fresh.status == replayed.status == "ok" and fresh.constraints == replayed.constraints
    assert [(use.status, use.reason, use.constraint.range_m) for use in fresh.constraints] == [(NOT, NO_RANGE, None), ("used", None, None)]
    assert fresh.pose_world == replayed.pose_world and raised(fresh) == raised(replayed) == {"constraint_unusable"}
    line = json.dumps(fresh.model_dump(mode="json"))                  # as results.jsonl is written: plain JSON, no Infinity in it
    assert "Infinity" not in line.replace(json.dumps(text), "") and json.loads(line)["constraints"][0]["constraint"]["range_m"] is None


# ------------------------------------------------------------------------------- both calls, on a generated scene
@pytest.fixture(scope="module")
def generated(tmp_path_factory):                   # one generated scene with three people in view, loaded as a run loads it
    scene, rgb, depth = build_scene(recipe_waiting_line(np.random.default_rng(3), 3), "s")
    folder = tmp_path_factory.mktemp("constraints") / "s"
    save_scene(folder, scene, rgb, depth)
    return load_scene(folder)


def pointing_at(found, observation):
    """A people answer that points at the body of each of `found`, with the facing left unclear."""
    camera = observation.camera
    points = [pixel_to_point(*person.pixel, camera.width, camera.height) for person in found]
    return {"people": [{"appearance": f"person {n}", "body_point": {"x": x, "y": y}, "faced_point": None, "facing": "unclear"}
                       for n, (x, y) in enumerate(points)]}


def test_with_people_from_a_model_both_calls_are_made_and_the_place_is_the_one_ground_truth_gives(generated):
    inputs, truth = generated
    observation = inputs.observation
    target = oracle_target(observation, next(iter(truth.targets.values())), TH)
    oracle = OraclePeople(truth, TH)(observation)
    answers = {"people": pointing_at(oracle.found, observation), "constraints": AFTER_ALL}
    client, backend = client_for(lambda parts, schema: next(answer for key, answer in answers.items() if key in schema["properties"]))
    asking = ConstraintInferrer(VlmPeople(client, "zero_shot", TH), client, "zero_shot", TH)
    result = asking.infer(observation, REQUEST, SUBTASK, target, None)
    assert [(call.stage, call.status) for call in result.calls] == [("people", "ok"), ("constraints", "ok")] and len(backend.calls) == 2
    people_images, constraint_images = ([part for part in parts if isinstance(part, ImagePart)] for _, parts, _ in backend.calls)
    assert people_images == [encode_png(observation.rgb)]             # the people are asked for on the camera image as it is,
    assert constraint_images == [encode_png(numbered_image(observation, target, result.people))]     # the constraints on the numbered one
    assert [person.number for person in result.people] == [1, 2, 3] and [person.label for person in result.people] == [
        "person 0", "person 1", "person 2"]                           # in this scene the first found stands leftmost
    known = ConstraintInferrer(OraclePeople(truth, TH), client, "zero_shot", TH).infer(observation, REQUEST, SUBTASK, target, None)
    assert [call.stage for call in known.calls] == ["constraints"] and [person.label for person in known.people] == [None] * 3
    assert all(math.dist(a.xy, b.xy) < TH.person_radius_m for a, b in zip(result.people, known.people))
    assert result.status == known.status == "ok" and math.dist(result.pose_world[:2], known.pose_world[:2]) < 0.3
    last = max(known.people, key=lambda person: math.dist(person.xy, target.world[:2]))
    assert math.dist(known.pose_world[:2], last.xy) == pytest.approx(1.0, abs=0.06)  # a metre from the last of them, and beyond them
    assert math.dist(known.pose_world[:2], target.world[:2]) > math.dist(last.xy, target.world[:2]) + 0.9
    spent = sum(call.latency_total_s for call in result.calls)        # both calls count under goal
    assert spent >= 120.0 and spent <= result.timings_s["goal"] < spent + 60.0 and 0.0 < result.timings_s["geometry"] < 60.0
    lines = [line for _, parts, _ in backend.calls[1:2] for part in parts if isinstance(part, TextPart) for line in part.text.splitlines()
             if line.startswith("Person ")]
    assert len(lines) == 3 and all("facing not known." in line for line in lines)   # the list, from what the model found
