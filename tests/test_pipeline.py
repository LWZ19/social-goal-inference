import itertools
import shutil
from types import SimpleNamespace
import pytest
from sgi.config import PROMPTS_DIR, Config, ConfigError, ModelEntry, RunConfig, StageConfig, Thresholds, load_config
from sgi.imaging import encode_png
from sgi.inputs import TourFrameData
from sgi.pipeline import Pipeline, build_pipeline
from sgi.prompts import Example
from sgi.stages.constraints import ConstraintInferrer, numbered_image
from sgi.stages.goal import standard_goal
from sgi.stages.people import PeopleFound, VlmPeople
from sgi.types import ContextCheck, GoalMark, GoalResult, Location, Plan, Point, Subtask
from sgi.vlm.base import ImagePart, TransientError, VLMClient, VLMResponse, to_call
from sgi.vlm.cache import Cache
from sgi.vlm.env import MissingKeyError
from sgi.vlm.factory import make_client
from sgi.vlm.mock import MockBackend, centre_responder
from tests.helpers import floor_observation

FIELD = {"subtasks": "plan", "target_point": "locate", "stop_elsewhere": "context", "goal_point": "goal",
         "people": "people", "constraints": "constraints"}       # the last two: the stages of goal: constraints
OBS, REQ, TICKS = floor_observation(), "take me to the info desk", itertools.count()
P55, P56 = Point(x=500, y=500), Point(x=500, y=600)
NAV = {"type": "navigate", "description": "Go to the info desk", "target": "info desk"}
ANSWERS = {"plan": {"subtasks": [{**NAV, "target_point": {"x": 500, "y": 500}}]},
           "locate": {"target_point": {"x": 500, "y": 600}},
           "context": {"relevant_people": "two people at the desk", "stop_elsewhere": True},
           "goal": {"goal_point": {"x": 400, "y": 800}},
           "people": {"people": []},                 # nobody on the bare floor
           "constraints": {"constraints": [{"because": "It is handed over there.", "kind": "within_distance", "people": [],
                                            "range_m": {"at_least": 1.5, "at_most": 2.0}, "target": True}]}}
SAID = ContextCheck(relevant_people="two people at the desk", people_matter=True)
EXAMPLE = Example(ImagePart(b"\x89PNG example", 64, 48), "bring this over", "front table", goal_point=P56)
FRAMES = [TourFrameData(index=10 + i, rgb=OBS.rgb, T_world_camera=OBS.camera.T_world_camera) for i in range(40)]

class Script:          # mock responder: reads the stage off the schema's top-level fields, logs it and its images
    def __init__(self, **answers):
        self.answers, self.seen, self.images = {**ANSWERS, **answers}, [], {}
    def __call__(self, parts, json_schema):
        self.seen.append(next(s for field, s in FIELD.items() if field in json_schema["properties"]))
        self.images[self.seen[-1]] = [p for p in parts if isinstance(p, ImagePart)]
        return self.answers[self.seen[-1]]

def make(script, cache=None, **run):         # the client's clock jumps 5 s per read: a fresh call "takes" 5 s or more
    entry = ModelEntry(provider="mock", model="mock-1")
    client = VLMClient(entry, MockBackend(script), cache, sleep=lambda _: None, clock=lambda: 5.0 * next(TICKS))
    stages = {s: StageConfig(model="m", strategy="few_shot" if s == "goal" else "zero_shot") for s in FIELD.values()}
    cfg = Config(models={"m": entry}, stages=stages, run=RunConfig(**run))
    return Pipeline(cfg, dict.fromkeys(stages, client), [EXAMPLE])

def test_plan_and_locate_on_one_view():
    script = Script()
    pipe = make(script, max_image_side=320)
    out = pipe.plan(REQ, None, observation=OBS)
    assert (out.status, out.call.stage, out.call.status, out.call.image_size) == ("ok", "plan", "ok", (320, 240))
    assert out.plan.image_sha256 == OBS.sha256()                       # of the frame as loaded, not as sent
    assert out.tour_indices is None                                    # view mode shows no tour frame
    location, call, time_s = pipe.locate(OBS, out.plan.subtasks[0], REQ, None)
    assert (location, call.stage, script.seen) == (Location(target_point=P56), "locate", ["plan", "locate"])
    assert script.images["locate"] == script.images["plan"]            # the same unmarked, resized frame
    assert 5.0 <= call.latency_total_s <= time_s < call.latency_total_s + 0.5
    bad = make(Script(plan="not json")).plan(REQ, None, observation=OBS)
    assert (bad.status, bad.plan, bad.call.validation_retries) == ("parse_failed", None, 1)
    assert (bad.call.status, bad.tour_indices) == ("parse_failed", None)   # to_call copies the status; still no tour
    for wrong in ({}, {"observation": OBS, "tour": FRAMES}):           # neither, and both
        with pytest.raises(ValueError):
            pipe.plan(REQ, None, **wrong)

def test_plan_tour_mode_subsamples_and_labels_frames(monkeypatch):
    got = {}
    def fake_run_plan(client, strategy, request, context, image, tour, trial=0, prompts_dir=None):
        got.update(image=image, tour=tour, prompts_dir=prompts_dir)
        resp = VLMResponse(status="ok", provider="mock", model="mock-1", reasoning_effort="low", point_format="xy_1000")
        return Plan(subtasks=[Subtask(type="navigate", description="go", target="desk", tour_frame=10)]), resp
    monkeypatch.setattr("sgi.pipeline.run_plan", fake_run_plan)
    out = make(Script(), max_tour_frames=16, max_image_side=320).plan(REQ, None, tour=FRAMES)
    assert got["image"] is None and all(isinstance(p, ImagePart) and p.width == 320 for _, p in got["tour"])
    assert got["prompts_dir"] == PROMPTS_DIR                           # the config's folder, handed down to the stage
    assert [i for i, _ in got["tour"]] == [10, 12, 15, 17, 20, 23, 25, 28, 30, 33, 36, 38, 41, 43, 46, 49]
    assert out.tour_indices == [i for i, _ in got["tour"]]             # the labels shown, not subsample positions
    assert (out.status, out.plan.image_sha256, out.plan.subtasks[0].tour_frame) == ("ok", None, 10)
    whole = make(Script(), max_tour_frames=64).plan(REQ, None, tour=FRAMES)
    assert [i for i, _ in got["tour"]] == list(range(10, 50))          # a tour shorter than the cap: every frame, once
    assert whole.tour_indices == list(range(10, 50))

@pytest.mark.parametrize("answer, status", [("not json", "parse_failed"), (TransientError("503"), "vlm_error")])
def test_tour_indices_are_set_when_the_plan_call_fails(answer, status):
    script = Script(plan=answer)                                       # the real run_plan, on 4 of the 40 frames
    out = make(script, max_tour_frames=4).plan(REQ, None, tour=FRAMES)
    assert (out.status, out.plan, out.call.status) == (status, None, status)
    assert out.tour_indices == [10, 23, 36, 49] and len(script.images["plan"]) == 4

class FakeInferrer:
    def infer(self, observation, request, subtask, target, context_check):
        self.got = (target.point, context_check)
        return GoalResult(status="ok", source="oracle", pose_robot=(1.0, 0.5, 0.25),
                          goal_mark=GoalMark(goal_point=P56), timings_s={"goal": 0.5, "geometry": 2.0})

def plan_of(point=P55, sha="same", kind="navigate"):
    step = Subtask(type=kind, description="Go to the info desk", target="info desk", target_point=point)
    return Plan(subtasks=[step], image_sha256=OBS.sha256() if sha == "same" else sha)

@pytest.mark.parametrize("point, sha, given, located", [
    (P56, "same", None, False),        # the single-view baseline: marked on this image, so no locate call
    (P55, "0" * 64, None, True),       # marked on a different image
    (P55, None, None, True),           # a tour-mode plan has no image hash
    (None, "same", None, True),        # a deferred subtask
    (P55, "same", P56, False),         # the argument wins over the plan's own point
    (None, "0" * 64, P56, False),      # the argument alone
])
def test_target_point_source_decides_whether_locate_runs(point, sha, given, located):
    script = Script()
    res = make(script, max_image_side=320).infer_goal(OBS, REQ, None, plan_of(point, sha), 0, target_point=given)
    stages = ["locate", "context", "goal"] if located else ["context", "goal"]
    assert script.seen == stages == [c.stage for c in res.calls] and set(res.timings_s) == {*stages, "geometry"}
    assert (res.status, res.source, res.context_check) == ("ok", "direct_marking", SAID)
    assert res.location == (Location(target_point=P56) if located else None)
    assert {c.image_size for c in res.calls} == {(320, 240)}           # every stage sends the resized image ...
    assert res.target.point == P56 and res.target.pixel == pytest.approx((320.0, 288.0))   # ... but lifts the full one
    assert res.target.world == pytest.approx((1.0, 4.774, 0.002), abs=0.01)
    assert res.pose_robot == pytest.approx((1.939, 0.236, -0.276), abs=0.01)
    shown, marked = script.images["goal"]                              # the example, then the frame with its marker
    assert shown == EXAMPLE.image and script.images["context"] == [marked]

@pytest.mark.parametrize("given", [None, ContextCheck(relevant_people="", people_matter=True)])
def test_custom_inferrer_and_context_check_argument(given):
    script, fake = Script(), FakeInferrer()
    res = make(script).infer_goal(OBS, REQ, None, plan_of(), 0, context_check=given, inferrer=fake)
    assert script.seen == (["context"] if given is None else [])
    assert fake.got == (P55, SAID if given is None else None)          # a check passed in is not forwarded
    assert (res.context_check, res.target.point) == (given or SAID, P55)
    assert (res.source, res.pose_robot, res.goal_mark) == ("oracle", (1.0, 0.5, 0.25), GoalMark(goal_point=P56))
    assert res.timings_s["goal"] == 0.5 and 2.0 <= res.timings_s["geometry"] < 2.5        # the lift time is added

@pytest.mark.parametrize("given", [None, ContextCheck(relevant_people="", people_matter=False)])
def test_people_matter_false_uses_the_standard_goal(given):
    script, fake = Script(context={"relevant_people": "nobody near the desk", "stop_elsewhere": False}), FakeInferrer()
    res = make(script).infer_goal(OBS, REQ, None, plan_of(), 0, context_check=given, inferrer=fake)
    assert script.seen == (["context"] if given is None else []) and not hasattr(fake, "got")
    assert (res.status, res.source, res.context_check.people_matter) == ("ok", "standard", False)
    assert res.pose_robot == pytest.approx((2.489, 0.0, 0.0), abs=0.01)    # one standoff short of the target

@pytest.mark.parametrize("plan, answers, status, stages, known", [
    (plan_of(kind="act"), {}, "no_target", [], set()),
    (plan_of(None), {"locate": {"target_point": None}}, "no_target", ["locate"], {"location"}),
    (plan_of(None), {"locate": {"target_point": {"x": 500, "y": 0}}}, "lift_failed", ["locate"], {"location"}),
    (plan_of(None), {"locate": "not json"}, "parse_failed", ["locate"], set()),
    (plan_of(None), {"context": TransientError("503")}, "vlm_error", ["locate", "context"], {"location", "target"}),
    (plan_of(), {"goal": "not json"}, "parse_failed", ["context", "goal"], {"target", "context_check"}),
])
def test_failure_statuses_keep_the_calls_and_outputs_so_far(plan, answers, status, stages, known):
    res = make(Script(**answers)).infer_goal(OBS, REQ, None, plan, 0)  # the top rows of the floor scene have no depth
    assert (res.status, res.pose_robot) == (status, None)
    assert [c.stage for c in res.calls] == stages and set(res.timings_s) == {*stages, "geometry"}
    failed = [status] if status in ("parse_failed", "vlm_error") else []           # a status that a model call gave
    assert [c.status for c in res.calls if c.status != "ok"] == failed
    assert {name for name in ("location", "target", "context_check") if getattr(res, name) is not None} == known

def test_infer_goal_locates_through_the_public_method_and_reads_the_call_status(monkeypatch):
    pipe, asked = make(Script()), []
    down = VLMResponse(status="vlm_error", provider="mock", model="mock-1", reasoning_effort="low", point_format="xy_1000")
    def locate(observation, subtask, request, context, trial=0):
        asked.append((subtask.target, request, context, trial))
        return None, to_call("locate", down), 1.5
    monkeypatch.setattr(pipe, "locate", locate)
    res = pipe.infer_goal(OBS, REQ, "a note", plan_of(None), 0, trial=3)
    assert asked == [("info desk", REQ, "a note", 3)]                  # the method the harness calls, with the trial
    assert (res.status, res.location, res.timings_s) == ("vlm_error", None, {"geometry": 0.0, "locate": 1.5})
    assert [(c.stage, c.status) for c in res.calls] == [("locate", "vlm_error")]

def test_cached_rerun_reproduces_model_time_and_a_new_trial_is_fresh(tmp_path):
    def run(trial):
        script = Script(plan={"subtasks": [{**NAV, "target_point": None}]})
        pipe = make(script, cache=Cache(tmp_path))
        out = pipe.plan(REQ, None, observation=OBS, trial=trial)
        res = pipe.infer_goal(OBS, REQ, None, out.plan, 0, trial=trial)
        return script.seen, [out.call, *res.calls], {"plan": out.time_s, **res.timings_s}
    (seen1, calls1, times1), (seen2, calls2, times2), (seen3, calls3, _) = run(0), run(0), run(1)
    assert seen1 == seen3 == ["plan", "locate", "context", "goal"] and seen2 == []
    assert [c.cached for c in calls1 + calls3] == [False] * 8 and [c.cached for c in calls2] == [True] * 4
    for first, second in zip(calls1, calls2):
        assert second.latency_total_s == first.latency_total_s >= 5.0
        assert [times1[first.stage], times2[first.stage]] == pytest.approx([first.latency_total_s] * 2, abs=0.5)

def test_build_pipeline_runs_the_default_config_on_the_mock_provider(tmp_path):
    pipe = build_pipeline(load_config(overrides={"run": {"cache_dir": str(tmp_path)}}))
    res = pipe.infer_goal(OBS, REQ, None, pipe.plan(REQ, None, observation=OBS).plan, 0)
    assert (res.status, res.source, [c.stage for c in res.calls]) == ("ok", "direct_marking", ["context", "goal"])
    assert {c.provider for c in res.calls} == {"mock"} and res.flags.heading_degenerate is True
    assert res.pose_robot == pytest.approx((3.489, 0.0, 0.0), abs=0.01)    # the mock marks the goal on the target

BASELINE, METHOD = ("plan", "locate", "context", "goal"), ("people", "constraints")

@pytest.fixture
def six(tmp_path, monkeypatch):       # a prompts folder with a stand-in for each of the method's two prompts, as the repository's own
    folder = tmp_path / "six"
    folder.mkdir()
    for stage in BASELINE:
        shutil.copy(PROMPTS_DIR / f"{stage}.md", folder)
    for stage in METHOD:
        (folder / f"{stage}.md").write_text(f"## system\nYou answer about the {stage}.\n\n## task\nRequest: {{request}}\n")
    monkeypatch.setattr("sgi.prompts.PROMPTS_DIR", folder)
    return folder

def test_build_pipeline_builds_a_client_for_each_stage_of_the_run_and_for_no_other(six, monkeypatch):
    models = {"keyed_a": {"provider": "gemini", "model": "a"}, "keyed_b": {"provider": "openai", "model": "b"}}
    stages = {"people": {"model": "keyed_a", "strategy": "zero_shot"}, "constraints": {"model": "keyed_b", "strategy": "cot"}}
    def cfg(**run):
        return load_config(overrides={"models": models, "stages": stages, "run": {"use_cache": False, **run}})
    pipe = build_pipeline(cfg())                                       # a baseline run: no key is asked for the two entries
    assert list(pipe.clients) == list(BASELINE) and {c.entry.model for c in pipe.clients.values()} == {"mock-centre"}
    assert list(build_pipeline(cfg(goal="oracle", people="oracle", constraints="none")).clients) == list(BASELINE)
    with pytest.raises(MissingKeyError, match="GEMINI_API_KEY"):       # goal: constraints: the people stage is one of the run's
        build_pipeline(cfg(goal="constraints", prompts_dir=str(six)))
    with pytest.raises(MissingKeyError, match="OPENAI_API_KEY"):       # with the people given, the constraints stage still is
        build_pipeline(cfg(goal="constraints", people="oracle", prompts_dir=str(six)))
    built, backend = [], MockBackend(centre_responder)
    def keyless(entry_name, config, given=None):
        built.append(entry_name)
        return make_client(entry_name, config, backend)
    monkeypatch.setattr("sgi.pipeline.make_client", keyless)
    pipe = build_pipeline(cfg(goal="constraints", prompts_dir=str(six)))
    assert list(pipe.clients) == [*BASELINE, *METHOD] and built == ["mock"] * 4 + ["keyed_a", "keyed_b"]
    assert (pipe.clients["people"].entry.model, pipe.clients["constraints"].entry.model) == ("a", "b")
    pipe = build_pipeline(cfg(goal="constraints", people="oracle", constraints="none", prompts_dir=str(six)))
    assert list(pipe.clients) == [*BASELINE, "constraints"] and built[6:] == ["mock"] * 4 + ["keyed_b"]
    (six / "constraints.md").unlink()                                  # a file of the run's set that is missing: refused here,
    with pytest.raises(ConfigError, match="constraints.md is missing"):    # before any call
        build_pipeline(cfg(goal="constraints", prompts_dir=str(six)))
    assert list(build_pipeline(cfg(prompts_dir=str(six))).clients) == list(BASELINE)       # a baseline run never looks for it
    assert backend.calls == []

def test_under_goal_constraints_infer_goal_finds_the_people_asks_for_the_constraints_and_solves():
    script = Script()
    res = make(script, goal="constraints", max_image_side=320).infer_goal(OBS, REQ, None, plan_of(), 0)
    assert script.seen == ["context", "people", "constraints"] == [c.stage for c in res.calls]     # never the baseline's goal call
    assert set(res.timings_s) == {"context", "goal", "geometry"} and {c.image_size for c in res.calls} == {(320, 240)}
    assert (res.status, res.source, res.context_check, res.goal_mark) == ("ok", "constraints", SAID, None)
    assert res.people == [] and [(use.status, use.constraint.kind) for use in res.constraints] == [("used", "within_distance")]
    away = ((res.pose_world[0] - res.target.world[0]) ** 2 + (res.pose_world[1] - res.target.world[1]) ** 2) ** 0.5
    assert 1.5 - 0.05 <= away <= 2.0 + 0.05 and res.solver.target_distance_m == pytest.approx(away)      # where the answer put it
    assert script.images["people"] == [encode_png(OBS.rgb, 320)]       # the people are asked for on the frame as it is,
    assert script.images["constraints"] == [encode_png(numbered_image(OBS, res.target, []), 320)]     # the constraints on the marked one
    assert script.images["constraints"] == script.images["context"]   # with nobody to number, the image the context stage saw
    nobody = make(Script(context={"relevant_people": "nobody near", "stop_elsewhere": False}), goal="constraints")
    assert nobody.infer_goal(OBS, REQ, None, plan_of(), 0).source == "standard"      # where people do not matter: the standard goal

def test_the_constraints_request_holds_what_the_model_said_in_the_context_stage_and_only_that(monkeypatch):
    sent = {}
    def recording(parts, json_schema):
        stage = next(s for field, s in FIELD.items() if field in json_schema["properties"])
        sent[stage] = "\n".join(p.text for p in parts if hasattr(p, "text"))
        return ANSWERS[stage]
    pipe = make(recording, goal="constraints")
    assert pipe.infer_goal(OBS, REQ, None, plan_of(), 0).status == "ok"
    assert "earlier step, before the numbers were drawn: two people at the desk" in sent["constraints"]
    assert "No person was found in view" in sent["constraints"] and f"Request: {REQ}" in sent["constraints"]
    assert "Request" not in sent["people"]                             # nothing of the episode goes into the people request
    sent.clear()
    given = ContextCheck(relevant_people="from the harness", people_matter=True)
    assert pipe.infer_goal(OBS, REQ, None, plan_of(), 0, context_check=given).status == "ok"
    assert set(sent) == {"people", "constraints"} and "earlier step" not in sent["constraints"]     # a check passed in is not forwarded

def test_the_trial_and_the_settings_of_the_run_reach_the_constraint_inferrer():
    th = Thresholds(max_people=7, max_constraints=3)
    script = Script()
    entry = ModelEntry(provider="mock", model="mock-1")
    asked = []
    def client(stage):                                                 # one client per stage, each recording the trial it is given
        real = VLMClient(entry, MockBackend(script), sleep=lambda _: None)
        def generate(parts, schema, system=None, trial=0):
            asked.append((stage, trial))
            return real.generate(parts, schema, system=system, trial=trial)
        return SimpleNamespace(entry=entry, generate=generate)
    stages = {s: StageConfig(model="m", strategy="cot" if s in ("people", "constraints") else "zero_shot") for s in FIELD.values()}
    cfg = Config(models={"m": entry}, stages=stages, thresholds=th, run=RunConfig(goal="constraints", max_image_side=320))
    pipe = Pipeline(cfg, {stage: client(stage) for stage in stages})
    script.answers.update(people={"analysis": "nobody", "people": []},
                          constraints={"analysis": {"arrangement": "a", "expectations": "b", "placement": "c"}, "constraints": []})
    res = pipe.infer_goal(OBS, REQ, None, plan_of(), 0, trial=3)
    assert asked == [("context", 3), ("people", 3), ("constraints", 3)] and res.status == "ok"     # each from its own stage's client
    built = pipe.constraint_inferrer(trial=2)
    assert isinstance(built, ConstraintInferrer) and isinstance(built.people_source, VlmPeople)
    assert (built.client, built.strategy, built.th, built.max_image_side, built.trial, built.constraints_mode, built.prompts_dir) == (
        pipe.clients["constraints"], "cot", th, 320, 2, "vlm", PROMPTS_DIR)
    source = built.people_source
    assert (source.client, source.strategy, source.th, source.max_image_side, source.trial, source.prompts_dir) == (
        pipe.clients["people"], "cot", th, 320, 2, PROMPTS_DIR)
    assert (pipe.constraint_inferrer().trial, pipe.constraint_inferrer().people_source.trial) == (0, 0)
    mine = lambda observation: PeopleFound(found=[], calls=[], call_time_s=0.0, geometry_time_s=0.0)
    assert pipe.constraint_inferrer(mine, trial=1).people_source is mine and pipe.constraint_inferrer(mine, trial=1).trial == 1

def test_under_constraints_none_no_model_is_asked_for_them_and_the_solver_runs_alone():
    script = Script()
    pipe = make(script, goal="constraints", constraints="none")
    res = pipe.infer_goal(OBS, REQ, None, plan_of(), 0)
    assert script.seen == ["context", "people"] == [c.stage for c in res.calls] and pipe.constraint_inferrer().constraints_mode == "none"
    assert (res.status, res.source, res.constraints) == ("ok", "constraints", []) and res.flags.no_usable_constraint is True
    alone = standard_goal(OBS, res.target, Thresholds())               # with nobody about, the solver stops where the standard goal does
    assert res.pose_world[:2] == pytest.approx(alone.pose_world[:2], abs=0.051)

def test_oracle_people_without_a_people_source_is_a_configuration_error_before_any_call():
    script, fake = Script(), FakeInferrer()
    pipe = make(script, goal="constraints", people="oracle")
    for build in (lambda: pipe.infer_goal(OBS, REQ, None, plan_of(None), 0), pipe.constraint_inferrer):
        with pytest.raises(ConfigError) as refusal:                    # ground truth is the harness's: a pipeline by itself has none
            build()
        assert "run.people: oracle" in str(refusal.value) and "people source" in str(refusal.value) and "\n" not in str(refusal.value)
    assert script.seen == []                                           # not the locate call either
    given = lambda observation: PeopleFound(found=[], calls=[], call_time_s=0.0, geometry_time_s=0.0)
    res = pipe.infer_goal(OBS, REQ, None, plan_of(), 0, inferrer=pipe.constraint_inferrer(given))
    assert script.seen == ["context", "constraints"] and (res.status, res.source, res.people) == ("ok", "constraints", [])
    assert pipe.infer_goal(OBS, REQ, None, plan_of(), 0, inferrer=fake).source == "oracle"     # any inferrer passed in is used as ever

@pytest.mark.parametrize("answers, status, stages, known", [
    ({"people": "not json"}, "parse_failed", ["context", "people"], set()),
    ({"people": TransientError("503")}, "vlm_error", ["context", "people"], set()),
    ({"constraints": "not json"}, "parse_failed", ["context", "people", "constraints"], {"people"}),
    ({"constraints": TransientError("503")}, "vlm_error", ["context", "people", "constraints"], {"people"}),
])
def test_a_failed_call_of_the_method_ends_the_result_with_its_status_and_what_was_known(answers, status, stages, known):
    res = make(Script(**answers), goal="constraints").infer_goal(OBS, REQ, None, plan_of(), 0)
    assert (res.status, res.source, res.pose_robot, res.pose_world) == (status, "constraints", None, None)
    assert [c.stage for c in res.calls] == stages and [c.status for c in res.calls] == ["ok"] * (len(stages) - 1) + [status]
    assert set(res.timings_s) == {"context", "goal", "geometry"} and res.target is not None and res.context_check == SAID
    assert {name for name in ("people", "constraints", "candidates", "solver") if getattr(res, name) is not None} == known
