import math
import re

import pytest
import yaml

from sgi import run as harness
from sgi import solver
from sgi.config import PROMPTS_DIR, Thresholds
from sgi.geometry import pixel_to_point
from sgi.gt.derive import oracle_point, prepare
from sgi.imaging import encode_png
from sgi.scene import load_scene
from sgi.stages.constraints import ConstraintInferrer
from sgi.stages.goal import standard_goal
from sgi.types import GoalResult
from sgi.vlm.base import ImagePart, TextPart, TransientError
from tests.test_run import config, episodes, placeholder, records, scene_dirs, subset, summary, wire  # noqa: F401

TH = Thresholds()
BASELINE = ("plan", "locate", "context", "goal")    # the stages of every run, each with a prompt the run folder keeps
METHOD = {"grounding": "oracle", "context_check": "always_yes", "goal": "constraints"}       # the method's own calls alone
GIVEN = {**METHOD, "people": "oracle"}              # and of those the constraints call alone: the people come from ground truth
FIELD = {"subtasks": "plan", "target_point": "locate", "stop_elsewhere": "context", "goal_point": "goal",
         "people": "people", "constraints": "constraints"}
KEPT = ("people", "constraints", "candidates", "solver")     # what a result of the method holds beside a baseline's


def pointing_at(found, observation):
    """A people answer that points at the body of each of `found`, each seen from behind."""
    camera = observation.camera
    points = [pixel_to_point(*person.pixel, camera.width, camera.height) for person in found]
    return {"people": [{"appearance": f"body {n}", "body_point": {"x": x, "y": y}, "faced_point": None, "facing": "away_from_camera"}
                       for n, (x, y) in enumerate(points)]}


class Scripted:
    """A mock for runs under `goal: constraints` over the scenes of a folder.

    It marks the first target of a scene in a plan, says in a context check that the people matter, points at
    every person of ground truth that is in view, and states three constraints: to come after everyone the
    request lists, to face the target, and one about a number nobody has. `answers` replaces what a stage gets.
    `seen` holds the stage and the text of every request, in order."""

    def __init__(self, scenes, **answers):
        self.answers, self.seen, self.targets, self.bodies = answers, [], {}, {}
        for folder in scene_dirs(scenes):
            inputs, truth = load_scene(folder)
            observation = inputs.observation
            shown = encode_png(observation.rgb).png                    # a scene is known by the camera image a request shows
            first = truth.targets[next(iter(truth.episodes.values())).target_ids[0]]
            self.targets[shown] = oracle_point(observation, first.anchor)
            self.bodies[shown] = pointing_at(harness.OraclePeople(truth, TH)(observation).found, observation)

    def __call__(self, parts, json_schema):
        stage = next(name for field, name in FIELD.items() if field in json_schema["properties"])
        text = "\n".join(part.text for part in parts if isinstance(part, TextPart))
        self.seen.append((stage, text))
        if stage in self.answers:
            return self.answers[stage]
        image = [part for part in parts if isinstance(part, ImagePart)][-1].png
        if stage == "plan":
            point = self.targets[image]
            step = {"type": "navigate", "description": "Go there", "target": "it", "target_point": {"x": point.x, "y": point.y}}
            return {key: "reasons" if key == "reasoning" else [step] for key in json_schema["properties"]}
        if stage == "context":
            return {"relevant_people": "those who stand before it", "stop_elsewhere": True}
        if stage == "people":
            return self.bodies[image]
        listed = [int(number) for number in re.findall(r"^Person (\d+)", text, re.M)]
        stated = [{"because": "They were there first.", "kind": "come_after", "people": listed, "range_m": None, "target": False},
                  {"because": "It is what the robot came for.", "kind": "face", "people": [], "range_m": None, "target": True},
                  {"because": "Someone watches.", "kind": "visible_to", "people": [99], "range_m": None, "target": False}]
        analysis = {"analysis": {"arrangement": "a", "expectations": "b", "placement": "c"}}
        return {**(analysis if "analysis" in json_schema["properties"] else {}), "constraints": stated}

    def asked(self, stage):
        return [text for name, text in self.seen if name == stage]


def run_method(placeholder, tmp_path, monkeypatch, name, scenes=None, script=None, **run):
    """One run of the method on a scene of each kind: the run folder, its records, the mock and its back end."""
    scenes = scenes or subset(tmp_path, placeholder / "eval")
    script = script or Scripted(scenes)
    backend = wire(monkeypatch, script)
    out = harness.run(scenes, config(placeholder, run={**METHOD, **run}), name, tmp_path / "runs")
    return out, records(out), script, backend


def truths(scenes):
    """By scene id: the observation, the ground truth and the people of it that are in view."""
    loaded = {inputs.scene_id: (inputs.observation, truth) for inputs, truth in map(load_scene, scene_dirs(scenes))}
    return {scene_id: (observation, truth, harness.OraclePeople(truth, TH)(observation).found)
            for scene_id, (observation, truth) in loaded.items()}


def test_with_the_people_of_ground_truth_a_run_asks_for_the_constraints_alone_and_records_what_the_method_made(placeholder, tmp_path, monkeypatch):
    out, recs, script, backend = run_method(placeholder, tmp_path, monkeypatch, "given", people="oracle")
    scenes = tmp_path / "subset"
    known = truths(scenes)
    assert len(recs) == len(episodes(scenes)) == len(backend.calls) > 0 and {stage for stage, _ in script.seen} == {"constraints"}
    assert sum(len(found) for _, _, found in known.values()) > 4       # the scenes hold people to number
    for record in recs:
        result, (observation, truth, found) = record["result"], known[record["scene_id"]]
        assert (result["status"], result["source"], result["goal_mark"]) == ("ok", "constraints", None)
        assert [call["stage"] for call in result["calls"]] == ["constraints"] and set(result["timings_s"]) == {"goal", "geometry"}
        people = result["people"]                                      # the people of ground truth in view, numbered from the left
        assert [person["number"] for person in people] == list(range(1, len(found) + 1))
        assert sorted(tuple(person["xy"]) for person in people) == sorted(person.xy for person in found)
        assert [person["pixel"][0] for person in people] == sorted(person["pixel"][0] for person in people)
        assert all(person["label"] is None and person["yaw"] is not None for person in people)
        uses = result["constraints"]                                   # each constraint as it was stated, and what became of it
        assert [use["constraint"]["kind"] for use in uses] == ["come_after", "face", "visible_to"]
        assert uses[0]["constraint"]["people"] == [person["number"] for person in people]
        assert all(set(use) == {"constraint", "status", "reason", "broken_m"} for use in uses)
        assert [use["status"] for use in uses[1:]] == ["used", "unusable"] and uses[2]["reason"].startswith("none of its numbers")
        assert result["flags"]["constraint_unusable"] is True and result["flags"]["people_cut"] is False
        assert 1 <= len(result["candidates"]) <= TH.candidate_count    # the runners-up, each a pose with its keys
        assert all(len(candidate["pose_world"]) == 3 and set(candidate["keys"]) == set(result["solver"]["keys"])
                   for candidate in result["candidates"])
        assert set(result["solver"]) == {"keys", "own_space_broken_m", "joined_waiting", "not_ahead_of", "not_ahead_reason",
                                         "target_distance_m"}
        assert set(result["solver"]["keys"]) == {"excluded_m", "excluded_total_m", "turn_m", "face_m", "preference_m", "standard_m"}
        assert result["pose_robot"] is not None and record["score"]["target_correct"] is True
        assert (record["config"]["goal"], record["config"]["people"], record["config"]["constraints"]) == ("constraints", "oracle", "vlm")
        assert record["example_ids"] == []                             # the method is shown no worked example
    waited = [r for r in recs if r["result"]["constraints"][0]["status"] == "used"]       # where someone was there first
    assert len(waited) >= 2 and all(r["result"]["solver"]["keys"]["turn_m"] is not None for r in waited)
    doc = yaml.safe_load((out / "config.yaml").read_text())
    assert (doc["config"]["run"]["goal"], doc["config"]["run"]["people"], doc["config"]["run"]["constraints"]) == (
        "constraints", "oracle", "vlm") and (doc["incomplete"], doc["example_ids"]) == (False, [])
    kept = sorted(path.name for path in (out / "prompts").iterdir())   # the prompts of the run's stages: no model is asked for people
    assert kept == sorted(f"{stage}.md" for stage in (*BASELINE, "constraints"))
    assert (out / "prompts" / "constraints.md").read_text() == (PROMPTS_DIR / "constraints.md").read_text()
    assert sorted(path.name for path in (out / "viz").iterdir()) == sorted(f"{s}_{e}.png" for s, e, _ in episodes(scenes))
    overall = summary(out)[0]
    assert float(overall["flag_constraint_unusable_rate"]) == 1.0 and float(overall["status_parse_failed_rate"]) == 0.0
    assert (int(overall["calls"]), overall["latency_plan_s"]) == (len(recs), "")


def test_a_record_of_the_method_can_be_solved_again_without_a_model_call(placeholder, tmp_path, monkeypatch):
    _, recs, _, _ = run_method(placeholder, tmp_path, monkeypatch, "again", people="oracle")
    known = truths(tmp_path / "subset")
    for record in recs:
        result, observation = GoalResult.model_validate(record["result"]), known[record["scene_id"]][0]
        stated = [use.constraint for use in result.constraints]
        again = solver.solve(observation, result.target, result.people, stated, standard_goal(observation, result.target, TH), TH)
        assert (again.pose_world, again.pose_robot) == (result.pose_world, result.pose_robot)
        assert (again.uses, again.candidates, again.note) == (result.constraints, result.candidates, result.solver)


def test_with_people_from_a_model_both_calls_are_made_and_the_episodes_of_a_scene_share_the_people_answer(placeholder, tmp_path, monkeypatch):
    cached = {"use_cache": True, "cache_dir": str(tmp_path / "cache")}
    out, recs, script, backend = run_method(placeholder, tmp_path, monkeypatch, "asked", **cached)
    scenes = tmp_path / "subset"
    known = truths(scenes)
    assert [[call["stage"] for call in r["result"]["calls"]] for r in recs] == [["people", "constraints"]] * len(recs)
    assert all(r["result"]["status"] == "ok" and r["result"]["source"] == "constraints" for r in recs)
    assert (recs[0]["config"]["people"], recs[0]["config"]["constraints"]) == ("vlm", "vlm")
    by_scene = {}
    for record in recs:
        by_scene.setdefault(record["scene_id"], []).append(record["result"])
    for scene_id, results in by_scene.items():       # one people request per scene: nothing of an episode is in it
        assert [result["calls"][0]["cached"] for result in results] == [False] + [True] * (len(results) - 1)
        assert all(result["calls"][1]["cached"] is False for result in results)      # the constraints are asked per episode
        assert all(result["people"] == results[0]["people"] for result in results)
        found = known[scene_id][2]
        for person in results[0]["people"]:          # each stands where someone of ground truth stands, and is named as the answer named them
            assert min(math.dist(person["xy"], true.xy) for true in found) < 2 * TH.person_radius_m and person["label"].startswith("body ")
            assert person["yaw"] is not None         # seen from behind: the bearing to the camera, turned round
    assert len(script.asked("people")) == len(by_scene) and len(set(script.asked("people"))) == 1     # the same words for every scene
    assert len(script.asked("constraints")) == len(recs) and len(backend.calls) == len(by_scene) + len(recs)
    assert sorted(path.name for path in (out / "prompts").iterdir()) == sorted(f"{stage}.md" for stage in (*BASELINE, "people", "constraints"))
    overall = summary(out)[0]
    assert (int(overall["calls"]), int(overall["cached_calls"])) == (2 * len(recs), len(recs) - len(by_scene))
    assert float(overall["latency_goal_s"]) > 0.0 and overall["latency_context_s"] == ""


@pytest.mark.parametrize("people, stages, calls_per_record", [("oracle", (*BASELINE, "constraints"), []),
                                                                ("vlm", (*BASELINE, "people", "constraints"), ["people"])])
def test_under_constraints_none_the_constraints_are_not_asked_for_and_the_solver_runs_alone(placeholder, tmp_path, monkeypatch, people,
                                                                                            stages, calls_per_record):
    out, recs, script, backend = run_method(placeholder, tmp_path, monkeypatch, "alone", people=people, constraints="none")
    assert [[call["stage"] for call in r["result"]["calls"]] for r in recs] == [calls_per_record] * len(recs)
    assert script.asked("constraints") == [] and (backend.calls == []) is (people == "oracle")
    for record in recs:
        result = record["result"]
        assert (result["status"], result["source"], result["constraints"]) == ("ok", "constraints", [])
        assert result["flags"]["no_usable_constraint"] is True and result["flags"]["constraint_unusable"] is False
        assert result["people"] is not None and result["solver"]["keys"]["turn_m"] is None and result["pose_robot"] is not None
        assert (record["config"]["people"], record["config"]["constraints"]) == (people, "none")
    assert sorted(path.name for path in (out / "prompts").iterdir()) == sorted(f"{stage}.md" for stage in stages)     # the stage stays
    assert float(summary(out)[0]["flag_no_usable_constraint_rate"]) == 1.0


def test_two_trials_of_a_run_with_the_people_of_ground_truth_send_two_constraints_requests(placeholder, tmp_path, monkeypatch):
    cached = {"people": "oracle", "trials": 2, "use_cache": True, "cache_dir": str(tmp_path / "cache")}
    _, recs, script, backend = run_method(placeholder, tmp_path, monkeypatch, "twice", **cached)
    n = len(episodes(tmp_path / "subset"))
    assert [r["trial"] for r in recs] == [0, 1] * n and len(script.asked("constraints")) == len(backend.calls) == 2 * n
    assert not any(call["cached"] for r in recs for call in r["result"]["calls"])    # trial 1 is no cache hit of trial 0
    asked = script.asked("constraints")
    assert all(asked[2 * k] == asked[2 * k + 1] for k in range(n))     # the same words: the trial is in the cache key alone
    replay = Scripted(tmp_path / "subset")
    _, again, _, replayed = run_method(placeholder, tmp_path, monkeypatch, "replayed", scenes=tmp_path / "subset", script=replay, **cached)
    assert replayed.calls == [] and all(call["cached"] for r in again for call in r["result"]["calls"])
    assert [r["result"]["pose_robot"] for r in again] == [r["result"]["pose_robot"] for r in recs]


def test_the_harness_gives_the_people_of_ground_truth_with_the_trial_and_leaves_people_from_a_model_to_the_pipeline(placeholder):
    inputs, truth = load_scene(placeholder / "eval" / "eval_13_waiting_line_3")      # three people waiting in line at a place
    def given(**run):
        cfg = config(placeholder, run={**METHOD, **run})
        et = next(iter(prepare(inputs, truth, cfg).values()))
        chosen = et.gt.target_ids[0]
        return harness._given_inferrer(cfg.run, harness.build_pipeline(cfg), truth, et.gt, chosen, cfg.thresholds, 2), cfg
    inferrer, cfg = given(people="oracle")
    assert isinstance(inferrer, ConstraintInferrer) and isinstance(inferrer.people_source, harness.OraclePeople)
    assert (inferrer.trial, inferrer.constraints_mode, inferrer.th) == (2, "vlm", cfg.thresholds)       # the trial, or trial 1 hits trial 0
    assert inferrer.people_source.truth is truth and inferrer.people_source.th == cfg.thresholds
    assert given(people="oracle", constraints="none")[0].constraints_mode == "none"
    assert given(people="vlm")[0] is None and given(goal="direct_marking")[0] is None    # the pipeline builds its own, with its trial
    oracle, _ = given(goal="oracle")
    assert isinstance(oracle, harness.OracleGoalInferrer) and not isinstance(oracle, ConstraintInferrer)


def test_with_a_plan_and_a_context_check_from_the_model_the_method_runs_where_the_people_matter(placeholder, tmp_path, monkeypatch):
    run = {"grounding": "vlm", "context_check": "vlm"}
    _, recs, script, _ = run_method(placeholder, tmp_path, monkeypatch, "chain", **run)
    assert [[call["stage"] for call in r["result"]["calls"]] for r in recs] == [["plan", "context", "people", "constraints"]] * len(recs)
    assert all(r["result"]["status"] == "ok" and r["result"]["source"] == "constraints" and r["score"]["target_correct"] for r in recs)
    assert set(recs[0]["result"]["timings_s"]) == {"plan", "context", "goal", "geometry"}
    assert all("in an earlier step, before the numbers were drawn: those who stand before it" in text for text in script.asked("constraints"))
    assert all(r["measures"]["context_missed"] is False for r in recs) and script.asked("goal") == []
    unaware = Scripted(tmp_path / "subset", context={"relevant_people": "none", "stop_elsewhere": False})
    _, recs, script, _ = run_method(placeholder, tmp_path, monkeypatch, "unaware", scenes=tmp_path / "subset", script=unaware, **run)
    assert {r["result"]["source"] for r in recs} == {"standard"} and script.asked("people") == script.asked("constraints") == []
    assert all(r["result"][name] is None for r in recs for name in KEPT)    # the standard goal, as in a baseline run


@pytest.mark.parametrize("stage, answer, status, made", [
    ("people", "not json", "parse_failed", ["people"]), ("people", TransientError("503"), "vlm_error", ["people"]),
    ("constraints", "not json", "parse_failed", ["people", "constraints"]),
    ("constraints", TransientError("503"), "vlm_error", ["people", "constraints"])])
def test_a_call_of_the_method_without_a_valid_answer_is_recorded_with_its_status_and_scored_as_a_failure(placeholder, tmp_path, monkeypatch,
                                                                                                          stage, answer, status, made):
    scenes = subset(tmp_path, placeholder / "eval")
    out, recs, _, _ = run_method(placeholder, tmp_path, monkeypatch, "failed", scenes=scenes, script=Scripted(scenes, **{stage: answer}))
    for record in recs:
        result = record["result"]
        assert (result["status"], result["source"], result["pose_robot"], record["score"]["success"]) == (status, "constraints", None, False)
        assert [call["stage"] for call in result["calls"]] == made and result["calls"][-1]["status"] == status
        assert (result["people"] is None) is (stage == "people") and result["constraints"] is None and result["solver"] is None
    doc = yaml.safe_load((out / "config.yaml").read_text())
    assert doc["incomplete"] is (status == "vlm_error") and doc["vlm_errors"] == (len(recs) if status == "vlm_error" else 0)
    assert len(list((out / "viz").iterdir())) == len(recs)             # a picture all the same


def test_a_baseline_run_holds_nothing_of_the_method(placeholder, tmp_path, monkeypatch):
    scenes = subset(tmp_path, placeholder / "eval")
    backend = wire(monkeypatch)
    recs = records(harness.run(scenes, config(placeholder), "baseline", tmp_path / "runs"))
    assert all(r["result"][name] is None for r in recs for name in KEPT) and {r["result"]["source"] for r in recs} == {"direct_marking"}
    assert {FIELD[next(field for field in FIELD if field in schema["properties"])] for _, _, schema in backend.calls} == {
        "plan", "context", "goal"}
