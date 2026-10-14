import shutil
import pytest
import yaml
from sgi import run as harness
from sgi import smoke
from sgi.config import PROMPTS_DIR, ConfigError, RunConfig, load_config
from sgi.pipeline import Pipeline, build_pipeline
from sgi.prompts import (build_context_request, build_goal_request, build_locate_request, build_plan_request,
                         check_prompts_dir, prompt_files, stages_of)
from sgi.types import Subtask
from sgi.vlm import factory
from sgi.vlm.base import ImagePart, TextPart, VLMClient
from sgi.vlm.mock import MockBackend, centre_responder
from tests.helpers import floor_observation, scene_dict, write_scene

STAGES = ("plan", "locate", "context", "goal")
FIELD = {"subtasks": "plan", "target_point": "locate", "stop_elsewhere": "context", "goal_point": "goal",
         "people": "people", "constraints": "constraints"}
SYSTEM_OF = {path.stem: path.read_text().split("## system\n")[1].split("\n## ")[0] for path in PROMPTS_DIR.glob("*.md")}
SYSTEM = {stage: SYSTEM_OF[stage] for stage in STAGES}    # the system text of each prompt, and of the four of the baseline
OLD = "Choose the point on the floor where the robot should stand."     # a sentence of goal.md, and what a copy says instead
NEW = "Choose the spot on the floor where the robot should come to rest."
MARK = " Take your time."                                 # added to the system text of every file of a marked copy
METHOD = ("people", "constraints")                        # the two stages goal: constraints adds, each with a prompt of its own
SIX = ", ".join(f"{stage}.md" for stage in (*STAGES, *METHOD))
REQUEST = "take me to the info desk"
OWN = ("Refer to a person by that number", "Name a person by that number")     # a sentence of constraints.md, and a copy's

def copied(root, name):                                   # a copy of the repository's prompts/ under root
    return shutil.copytree(PROMPTS_DIR, root / name)

def edit(path, old, new):
    text = path.read_text()
    assert text.count(old) == 1
    path.write_text(text.replace(old, new))

def four(root, name):                                     # a folder with the four baseline files, whatever prompts/ holds by now
    (root / name).mkdir()
    for stage in STAGES:
        shutil.copy(PROMPTS_DIR / f"{stage}.md", root / name)
    return root / name

def six(root, name):                                      # and one with a stand-in for each of the method's two prompts as well
    folder = four(root, name)
    for stage in METHOD:
        (folder / f"{stage}.md").write_text(f"## system\nYou answer about the {stage}.\n\n## task\nRequest: {{request}}\n")
    return folder

@pytest.fixture
def repo(tmp_path, monkeypatch):                          # the repository's prompts/ as it is once the method's prompts exist
    folder = six(tmp_path, "repo_prompts")
    monkeypatch.setattr("sgi.prompts.PROMPTS_DIR", folder)
    return folder

def marked(root):                                         # a copy in which every system text ends in MARK
    folder = copied(root, "marked")
    for stage in STAGES:
        edit(folder / f"{stage}.md", SYSTEM[stage], SYSTEM[stage] + MARK)
    return folder

def sent(backend):                                        # by stage: the (system, text) of every request the back end got
    found = {}
    for system, parts, schema in backend.calls:
        stage = next(name for field, name in FIELD.items() if field in schema["properties"])
        found.setdefault(stage, []).append((system, "\n".join(p.text for p in parts if isinstance(p, TextPart))))
    return found

def systems(backend):
    return {stage: {system for system, _ in requests} for stage, requests in sent(backend).items()}

@pytest.fixture
def work(tmp_path, monkeypatch):                          # where a command runs: one small scene, a config without examples or cache
    monkeypatch.chdir(tmp_path)
    write_scene(tmp_path / "scenes", scene_dict(), floor_observation())
    cfg = load_config(overrides={"run": {"use_cache": False}, "stages": {"goal": {"strategy": "zero_shot"}}})
    (tmp_path / "table.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return tmp_path

@pytest.fixture
def backend(monkeypatch):                                 # the one mock back end behind every client either command builds
    backend = MockBackend(centre_responder)
    def client(entry_name, cfg):
        return factory.make_client(entry_name, cfg, backend)
    monkeypatch.setattr("sgi.pipeline.make_client", client)
    monkeypatch.setattr(smoke, "make_client", client)
    return backend

def run_main(*options):                                   # exit code of sgi.run on the scene, on the mock entry
    return harness.main(["--scenes", "scenes", "--config", "table.yaml", "--name", "prompts", "--skip-scene-check", *options])

def test_a_run_reads_its_prompts_from_the_folder_given_and_keeps_a_copy(work, backend):
    folder = copied(work, "my_prompts")
    edit(folder / "goal.md", OLD, NEW)
    assert run_main("--prompts", "my_prompts") == 0       # a relative folder is one of the working directory
    (out,) = (work / "runs").iterdir()
    assert yaml.safe_load((out / "config.yaml").read_text())["config"]["run"]["prompts_dir"] == str(folder.resolve())
    assert sorted(p.name for p in (out / "prompts").iterdir()) == sorted(f"{stage}.md" for stage in STAGES)
    assert all((out / "prompts" / f"{stage}.md").read_text() == (folder / f"{stage}.md").read_text() for stage in STAGES)
    asked = sent(backend)
    assert {stage: len(requests) for stage, requests in asked.items()} == {"plan": 1, "context": 1, "goal": 1}
    goal_text = asked["goal"][0][1]
    assert NEW in goal_text and OLD not in goal_text      # the changed sentence is the one that was sent
    assert systems(backend) == {stage: {SYSTEM[stage]} for stage in asked}       # the rest of the copy is the original
    assert OLD in (PROMPTS_DIR / "goal.md").read_text()   # and the repository's own file is as it was

def test_without_the_option_the_repository_prompts_are_used_as_before(work, backend):
    assert run_main() == 0
    (out,) = (work / "runs").iterdir()
    assert yaml.safe_load((out / "config.yaml").read_text())["config"]["run"]["prompts_dir"] == "prompts"
    assert not (work / "prompts").exists()                # relative to the repository root, not to the working directory
    assert all((out / "prompts" / f"{stage}.md").read_text() == (PROMPTS_DIR / f"{stage}.md").read_text() for stage in STAGES)
    assert systems(backend) == {stage: {SYSTEM[stage]} for stage in ("plan", "context", "goal")}
    assert OLD in sent(backend)["goal"][0][1]

def test_each_builder_reads_the_folder_it_is_given_and_the_repository_folder_by_default(tmp_path):
    folder = marked(tmp_path)
    image, subtask = ImagePart(b"query", 640, 480), Subtask(type="navigate", description="go to the thing", target="the tall one")
    builders = {
        "plan": lambda **given: build_plan_request("cot", REQUEST, None, image, None, "xy_1000", **given),
        "plan on a tour": lambda **given: build_plan_request("cot", REQUEST, None, None, [(0, image)], "xy_1000", **given),
        "locate": lambda **given: build_locate_request("zero_shot", REQUEST, None, subtask, image, "xy_1000", **given),
        "context": lambda **given: build_context_request("zero_shot", REQUEST, subtask, image, 1.0, **given),
        "goal": lambda **given: build_goal_request("zero_shot", REQUEST, subtask, image, None, (), "xy_1000", "face_target", **given)}
    for name, build in builders.items():
        system, parts = build()
        assert (system, parts) == build(prompts_dir=PROMPTS_DIR), name
        assert build(prompts_dir=folder) == (system + MARK, parts), name

def test_two_pipelines_in_one_process_each_send_the_text_of_their_own_folder(tmp_path):
    def pipeline(prompts_dir):
        cfg = load_config(overrides={"run": {"use_cache": False, "prompts_dir": str(prompts_dir)},
                                     "stages": {"goal": {"strategy": "zero_shot"}}})
        backend = MockBackend(centre_responder)
        return Pipeline(cfg, dict.fromkeys(STAGES, VLMClient(cfg.models["mock"], backend))), backend
    (one, one_got), (two, two_got) = pipeline("prompts"), pipeline(marked(tmp_path))
    obs = floor_observation()
    for pipe in (one, two, one):                          # in turn: neither changes what the other reads
        out = pipe.plan(REQUEST, None, observation=obs)
        pipe.locate(obs, out.plan.subtasks[0], REQUEST, None)
        assert pipe.infer_goal(obs, REQUEST, None, out.plan, 0).status == "ok"
    assert (len(one_got.calls), len(two_got.calls)) == (8, 4)        # plan, locate, context and goal each time
    assert systems(one_got) == {stage: {SYSTEM[stage]} for stage in STAGES}
    assert systems(two_got) == {stage: {SYSTEM[stage] + MARK} for stage in STAGES}

def test_smoke_takes_the_same_folder(work, backend, capsys):
    folder = copied(work, "my_prompts")
    edit(folder / "goal.md", OLD, NEW)
    assert smoke.main(["--model", "mock", "--scene", "scenes", "--prompts", "my_prompts"]) == 0
    goal_text = sent(backend)["goal"][0][1]
    assert NEW in goal_text and OLD not in goal_text and "result: ok" in capsys.readouterr().out

@pytest.mark.parametrize("breakage, named", [
    ("no folder", ["no such folder"]), ("no file", ["locate.md"]), ("no block", ["goal.md", "relevant_people"]),
    ("misspelt placeholder", ["goal.md", "block task", "{reqest}", "{request}"]),     # the one written, and the one it stands for
    ("placeholder left out", ["locate.md", "block point_format", "{point_instruction}"])])
def test_a_folder_without_a_file_or_a_block_stops_both_commands_before_anything_is_asked_or_written(work, backend, capsys, breakage, named):
    folder = work / "broken"
    if breakage != "no folder":
        copied(work, "broken")
    if breakage == "no file":
        (folder / "locate.md").unlink()
    if breakage == "no block":                            # the header gone, its line now belongs to the block above it
        edit(folder / "goal.md", "## relevant_people\n", "")
    if breakage == "misspelt placeholder":                # in the middle of a run this was a KeyError at the first goal request
        edit(folder / "goal.md", "The case to answer:\nRequest: {request}", "The case to answer:\nRequest: {reqest}")
    if breakage == "placeholder left out":                # and this a request that never said how to write a point
        edit(folder / "locate.md", " {point_instruction}", "")
    commands = [lambda: run_main("--prompts", "broken"),
                lambda: smoke.main(["--model", "mock", "--scene", "scenes", "--prompts", "broken"]),
                lambda: smoke.main(["--model", "mock", "--scene", "scenes", "--prompts", "broken", "--calibrate"])]
    for command in commands:
        assert command() == 2
        out, err = capsys.readouterr()
        line = err[err.index("error: "):]                 # sgi.run logs how many episodes it skipped before that
        assert out == "" and line.count("\n") == 1 and "Traceback" not in err
        assert str(folder.resolve()) in line and all(name in line for name in named)
    assert backend.calls == [] and not (work / "runs").exists()

def test_the_files_of_a_folder_are_those_of_the_stages_given(tmp_path):
    def names(**run):
        return [path.relative_to(tmp_path).as_posix() for path in prompt_files(tmp_path, stages_of(RunConfig(**run)))]
    assert names() == names(goal="oracle") == [f"{stage}.md" for stage in STAGES]
    assert names(goal="constraints") == [f"{stage}.md" for stage in (*STAGES, *METHOD)]
    assert names(goal="constraints", people="oracle") == [f"{stage}.md" for stage in (*STAGES, "constraints")]
    assert prompt_files(tmp_path, ("locate",)) == [tmp_path / "locate.md"]

def test_the_folder_check_asks_for_the_files_of_the_stages_of_the_run_and_no_other(tmp_path, repo):
    baseline, method = stages_of(RunConfig()), stages_of(RunConfig(goal="constraints"))
    given_people = stages_of(RunConfig(goal="constraints", people="oracle"))
    folder = four(tmp_path, "mine")
    check_prompts_dir(folder, baseline)                   # the four baseline files stay a valid folder for a baseline run
    for stages, absent in ((method, "people.md"), (given_people, "constraints.md")):
        with pytest.raises(ConfigError) as refusal:       # named, with the files that set of stages asks for
            check_prompts_dir(folder, stages)
        held = ", ".join(f"{stage}.md" for stage in stages)
        assert f"{folder}: {absent} is missing; the folder has to hold {held}" in str(refusal.value)
    shutil.copy(repo / "people.md", folder)
    with pytest.raises(ConfigError, match=f"constraints.md is missing; the folder has to hold {SIX}"):
        check_prompts_dir(folder, method)
    shutil.copy(repo / "constraints.md", folder)
    check_prompts_dir(folder, method)                     # all six
    (folder / "people.md").unlink()
    check_prompts_dir(folder, given_people)               # no model is asked for the people: their prompt is not needed
    edit(folder / "constraints.md", "## task\n", "")
    with pytest.raises(ConfigError, match="constraints.md has no block task"):       # a new file is held to its blocks too
        check_prompts_dir(folder, given_people)
    check_prompts_dir(folder, baseline)                   # and is not looked at by a run that does not use it
    (folder / "goal.md").unlink()
    shutil.copy(repo / "constraints.md", folder)
    for stages in (baseline, given_people):               # goal.md is asked for under goal: constraints as well
        with pytest.raises(ConfigError, match="goal.md is missing"):
            check_prompts_dir(folder, stages)

def test_a_stage_without_a_prompt_in_the_repository_is_a_config_error_not_a_traceback(tmp_path, monkeypatch):
    bare = four(tmp_path, "repo_prompts")                 # a repository that has no prompt yet for the method's two stages
    monkeypatch.setattr("sgi.prompts.PROMPTS_DIR", bare)
    folder = six(tmp_path, "mine")                        # a folder that brings its own: there is nothing to hold them against
    check_prompts_dir(folder, stages_of(RunConfig()))
    with pytest.raises(ConfigError) as refusal:
        check_prompts_dir(folder, stages_of(RunConfig(goal="constraints")))
    assert str(folder) in str(refusal.value) and str(bare / "people.md") in str(refusal.value)
    assert "\n" not in str(refusal.value)

@pytest.mark.parametrize("modes, stages, asked", [
    ({}, (*STAGES, *METHOD), {"plan": 1, "context": 1, "people": 1, "constraints": 1}),
    ({"people": "oracle"}, (*STAGES, "constraints"), {"plan": 1, "context": 1, "constraints": 1}),
    ({"people": "oracle", "constraints": "none"}, (*STAGES, "constraints"), {"plan": 1, "context": 1})])     # the stage stays
def test_a_goal_constraints_run_reads_the_prompts_of_its_stages_from_the_folder_given_and_keeps_a_copy(work, backend, modes, stages, asked):
    folder = copied(work, "my_prompts")
    edit(folder / "constraints.md", *OWN)
    edit(folder / "people.md", SYSTEM_OF["people"], SYSTEM_OF["people"] + MARK)
    cfg = load_config(work / "table.yaml", {"run": {"goal": "constraints", **modes}})
    (work / "method.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    assert harness.main(["--scenes", "scenes", "--config", "method.yaml", "--name", "method", "--skip-scene-check",
                         "--prompts", "my_prompts"]) == 0
    (out,) = (work / "runs").iterdir()
    assert sorted(p.name for p in (out / "prompts").iterdir()) == sorted(f"{stage}.md" for stage in stages)
    assert all((out / "prompts" / f"{stage}.md").read_text() == (folder / f"{stage}.md").read_text() for stage in stages)
    got = sent(backend)
    assert {stage: len(requests) for stage, requests in got.items()} == asked     # never the goal stage of the baseline
    if "constraints" in asked:
        assert OWN[1] in got["constraints"][0][1] and OWN[0] not in got["constraints"][0][1]
    if "people" in asked:
        assert got["people"][0][0] == SYSTEM_OF["people"] + MARK
    assert OWN[0] in (PROMPTS_DIR / "constraints.md").read_text()     # the repository's own files are as they were

@pytest.mark.parametrize("breakage, named", [
    ("no file", ["constraints.md is missing", SIX]), ("no block", ["constraints.md", "no block check"]),
    ("misspelt placeholder", ["constraints.md", "block people", "{persons}", "{people}"]),
    ("no block in the prompt of the people", ["people.md", "no block point_format"])])
def test_a_folder_that_lacks_a_part_of_a_prompt_of_the_method_stops_both_commands_before_anything_is_asked(work, backend, capsys, breakage,
                                                                                                           named):
    folder = copied(work, "broken")
    if breakage == "no file":
        (folder / "constraints.md").unlink()
    if breakage == "no block":
        edit(folder / "constraints.md", "## check\n", "")
    if breakage == "misspelt placeholder":
        edit(folder / "constraints.md", "\n{people}\n", "\n{persons}\n")
    if breakage == "no block in the prompt of the people":
        edit(folder / "people.md", "## point_format\n", "")
    cfg = load_config(work / "table.yaml", {"run": {"goal": "constraints"}})
    (work / "method.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    commands = [lambda: harness.main(["--scenes", "scenes", "--config", "method.yaml", "--name", "method", "--skip-scene-check",
                                      "--prompts", "broken"]),
                lambda: smoke.main(["--model", "mock", "--scene", "scenes", "--prompts", "broken", "--goal", "constraints"])]
    for command in commands:
        assert command() == 2
        out, err = capsys.readouterr()
        line = err[err.index("error: "):]
        assert out == "" and line.count("\n") == 1 and "Traceback" not in err
        assert str(folder.resolve()) in line and all(name in line for name in named)
    assert backend.calls == [] and not (work / "runs").exists()
    assert run_main("--prompts", "broken") == 0           # a baseline run does not look at either file

@pytest.mark.parametrize("people, absent", [("vlm", "people.md"), ("oracle", "constraints.md")])
def test_goal_constraints_without_a_prompt_of_its_stages_is_refused_where_the_pipeline_is_built(work, backend, monkeypatch, people, absent):
    folder = four(work, "four")                           # the repository before the method's prompts are written
    monkeypatch.setattr("sgi.prompts.PROMPTS_DIR", folder)
    cfg = load_config(work / "table.yaml", {"run": {"goal": "constraints", "people": people, "prompts_dir": str(folder)}})
    with pytest.raises(ConfigError) as refusal:           # the folder rule holds for the set of the method, before any call
        build_pipeline(cfg)
    held = ", ".join(f"{stage}.md" for stage in stages_of(cfg.run))
    assert str(refusal.value) == f"prompts folder {folder}: {absent} is missing; the folder has to hold {held}"
    assert backend.calls == []
