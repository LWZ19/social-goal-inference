import functools
import json
import anthropic
import httpx
import httpx2
import openai
import pytest
import yaml
from google import genai
from google.genai import types as genai_types
from sgi import run as harness
from sgi import smoke
from sgi.config import ConfigError, load_config, with_model_options
from sgi.vlm import factory
from sgi.vlm.anthropic import AnthropicBackend
from sgi.vlm.compatible import OpenAICompatibleBackend
from sgi.vlm.gemini import GeminiBackend
from sgi.vlm.openai import OpenAIBackend
from tests.helpers import floor_observation, scene_dict, write_scene

STAGES = ("plan", "locate", "context", "goal")
METHOD = ("people", "constraints")                        # the two stages of goal: constraints: no baseline run calls them
BASE, KEY = "https://gateway.example/v1", "k3y-9f2c7a41d8e6b05-never-shown"       # made up: nothing is sent anywhere
EXTRA = {"mock_b": {"provider": "mock", "model": "mock-b", "reasoning_effort": "high"},     # a second keyless entry
         "gateway": {"provider": "openai_compatible", "model": "m", "base_url": BASE, "api_key_env": "GATEWAY_API_KEY"}}
BAD = [("context", "STAGE=ENTRY"), ("=mock_b", "STAGE=ENTRY"), ("goal=", "STAGE=ENTRY"),   # value, what the error names
       ("judge=mock_b", "'judge'"), ("goal=gpt-9", "'gpt-9'")]
BAD_STRATEGY = [("goal", "STAGE=NAME"), ("=cot", "STAGE=NAME"), ("goal=", "STAGE=NAME"), ("judge=cot", "'judge'"),
                ("goal=tree_of_thought", "'tree_of_thought'"),                    # not a strategy, and two that only
                ("context=few_shot", "'context'"), ("plan=few_shot_cot", "'plan'"),      # the goal stage accepts
                ("people=few_shot", "'people'"), ("constraints=few_shot_cot", "'constraints'")]
SHELL = ("OPENAI_CUSTOM_HEADERS", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID")           # what the SDK reads request headers from

@pytest.fixture(autouse=True)
def a_shell_without_headers_for_the_sdk(monkeypatch):     # the tests run the same whatever the shell holds
    for name in SHELL:
        monkeypatch.delenv(name, raising=False)

@pytest.fixture
def table(tmp_path):                                      # a config file with the default model table and EXTRA; no examples needed
    cfg = load_config(overrides={"models": EXTRA, "run": {"use_cache": False}, "stages": {"goal": {"strategy": "zero_shot"}}})
    path = tmp_path / "table.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return path

@pytest.fixture
def scenes(tmp_path, monkeypatch):                        # one small scene, and the working directory a command writes under
    monkeypatch.chdir(tmp_path)
    write_scene(tmp_path / "scenes", scene_dict(), floor_observation())
    return tmp_path / "scenes"

def run_main(table, scenes, *options):                    # exit code of sgi.run on the scene, mock entries only
    return harness.main(["--scenes", str(scenes), "--config", str(table), "--name", "switch", "--skip-scene-check", *options])

def stages_of(doc):
    return [doc["stages"][stage]["model"] for stage in STAGES]

def strategies_of(doc):
    return [doc["stages"][stage]["strategy"] for stage in STAGES]

@pytest.mark.parametrize("options, used", [
    (["--stage-model", "context=mock_b", "--stage-model", "goal=mock_b"], ["mock", "mock", "mock_b", "mock_b"]),
    (["--model", "mock_b", "--stage-model", "plan=mock"], ["mock", "mock_b", "mock_b", "mock_b"]),      # after --model,
    (["--stage-model", "plan=mock", "--model", "mock_b"], ["mock", "mock_b", "mock_b", "mock_b"]),      # wherever it stands
    (["--model", "mock_b"], ["mock_b"] * 4), ([], ["mock"] * 4)])
def test_stage_models_are_applied_after_the_model_and_recorded_in_the_run_folder(table, scenes, tmp_path, options, used):
    assert run_main(table, scenes, *options) == 0
    (out,) = (tmp_path / "runs").iterdir()
    assert stages_of(yaml.safe_load((out / "config.yaml").read_text())["config"]) == used
    (record,) = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert stages_of(record["config"]) == used
    ids = {"mock": ("mock-centre", "low"), "mock_b": ("mock-b", "high")}       # each call went to its stage's entry
    assert [(c["stage"], c["model"], c["reasoning_effort"]) for c in record["result"]["calls"]] == [
        (stage, *ids[used[STAGES.index(stage)]]) for stage in ("plan", "context", "goal")]

@pytest.mark.parametrize("options, used", [                # (entry, strategy) of people and of constraints; the table's own first
    ([], [("mock", "zero_shot"), ("mock", "cot")]),
    (["--stage-model", "people=mock_b", "--strategy", "constraints=cot"], [("mock_b", "zero_shot"), ("mock", "cot")]),
    (["--strategy", "people=cot", "--strategy", "constraints=zero_shot", "--stage-model", "constraints=mock_b"],
     [("mock", "cot"), ("mock_b", "zero_shot")]),
    (["--stage-model", "people=mock", "--model", "mock_b"], [("mock", "zero_shot"), ("mock_b", "cot")]),     # --model: all six
    (["--stage-model", "goal=mock_b"], [("mock", "zero_shot"), ("mock", "cot")])])     # in a table that names them, they stay
def test_the_two_stages_of_the_constraint_method_take_an_entry_and_a_strategy_like_the_others(table, scenes, tmp_path, options, used):
    assert run_main(table, scenes, *options) == 0
    (out,) = (tmp_path / "runs").iterdir()
    doc = yaml.safe_load((out / "config.yaml").read_text())["config"]
    assert [(doc["stages"][stage]["model"], doc["stages"][stage]["strategy"]) for stage in METHOD] == used
    (record,) = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert record["config"]["stages"] == doc["stages"] and set(doc["stages"]) == {*STAGES, *METHOD}
    assert (record["config"]["goal"], record["config"]["people"], record["config"]["constraints"]) == ("direct_marking", "vlm", "vlm")
    assert [c["stage"] for c in record["result"]["calls"]] == ["plan", "context", "goal"]      # a baseline run calls neither

@pytest.mark.parametrize("value, named", BAD)
def test_a_bad_stage_model_stops_the_run_before_anything_is_written_or_asked(table, scenes, tmp_path, monkeypatch, capsys, value, named):
    monkeypatch.setattr(harness, "build_pipeline", lambda *args, **kwargs: pytest.fail("a client was built"))
    assert run_main(table, scenes, "--model", "mock_b", "--stage-model", value) == 2
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("error: ") and err.count("\n") == 1 and "Traceback" not in err
    assert named in err and not (tmp_path / "runs").exists()

def test_an_unknown_stage_or_entry_is_named_with_the_valid_ones():
    cfg = load_config(overrides={"models": EXTRA})
    assert with_model_options(cfg, None, []) == cfg
    mixed = with_model_options(cfg, "mock_b", [" context = mock ", "goal=gateway"])       # spaces round a name are dropped
    assert [mixed.stages[stage].model for stage in STAGES] == ["mock_b", "mock_b", "mock", "gateway"]
    assert [stage.strategy for stage in mixed.stages.values()] == [stage.strategy for stage in cfg.stages.values()]
    for value, named in BAD:
        with pytest.raises(ConfigError, match=named):
            with_model_options(cfg, None, ["plan=mock_b", value])
    with pytest.raises(ConfigError, match="plan, locate, context, goal, people, constraints"):
        with_model_options(cfg, None, ["judge=mock"])
    with pytest.raises(ConfigError, match="mock_b, gateway"):         # the entries there are, in table order
        with_model_options(cfg, None, ["goal=gpt-9"])
    with pytest.raises(ConfigError, match="'gpt-9'"):                 # --model is checked as before
        with_model_options(cfg, "gpt-9", ["goal=mock"])

@pytest.mark.parametrize("options, used", [                # the table's own: cot, zero_shot, zero_shot, zero_shot
    (["--strategy", "goal=cot"], ["cot", "zero_shot", "zero_shot", "cot"]),
    (["--strategy", "plan=zero_shot", "--model", "mock_b", "--strategy", "context=cot"], ["zero_shot", "zero_shot", "cot", "zero_shot"]),
    (["--strategy", "goal=cot", "--strategy", "goal=zero_shot"], ["cot", "zero_shot", "zero_shot", "zero_shot"]),     # the later one
    ([], ["cot", "zero_shot", "zero_shot", "zero_shot"])])
def test_strategies_are_applied_to_their_stages_and_recorded_in_the_run_folder(table, scenes, tmp_path, options, used):
    assert run_main(table, scenes, *options) == 0
    (out,) = (tmp_path / "runs").iterdir()
    assert strategies_of(yaml.safe_load((out / "config.yaml").read_text())["config"]) == used
    (record,) = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert strategies_of(record["config"]) == used
    reasoned = [record["plan"]["reasoning"] is not None, record["result"]["context_check"]["reasoning"] is not None,
                record["result"]["goal_mark"]["reasoning"] is not None]           # the answers were asked for that way
    assert reasoned == [used[0] == "cot", used[2] == "cot", used[3] == "cot"]
    assert stages_of(record["config"]) == [("mock_b" if "--model" in options else "mock")] * 4      # the entries are kept

@pytest.mark.parametrize("value, named", BAD_STRATEGY)
def test_a_bad_strategy_stops_the_run_before_anything_is_written_or_asked(table, scenes, tmp_path, monkeypatch, capsys, value, named):
    monkeypatch.setattr(harness, "build_pipeline", lambda *args, **kwargs: pytest.fail("a client was built"))
    assert run_main(table, scenes, "--strategy", "goal=cot", "--strategy", value) == 2
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("error: ") and err.count("\n") == 1 and "Traceback" not in err
    assert named in err and not (tmp_path / "runs").exists()

def test_strategies_are_applied_after_the_models_and_a_bad_one_is_named_with_the_valid_ones():
    cfg = load_config(overrides={"models": EXTRA})
    assert with_model_options(cfg, None, [], []) == cfg
    mixed = with_model_options(cfg, "mock_b", ["goal=mock"], [" goal = few_shot_cot ", "plan=zero_shot"])
    assert [(mixed.stages[stage].model, mixed.stages[stage].strategy) for stage in STAGES] == [
        ("mock_b", "zero_shot"), ("mock_b", "zero_shot"), ("mock_b", "zero_shot"), ("mock", "few_shot_cot")]
    for value, named in BAD_STRATEGY:
        with pytest.raises(ConfigError, match=named):
            with_model_options(cfg, None, [], ["goal=cot", value])
    with pytest.raises(ConfigError, match="--strategy 'goal' is not of the form STAGE=NAME"):
        with_model_options(cfg, None, [], ["goal"])
    with pytest.raises(ConfigError, match="zero_shot, cot, few_shot, few_shot_cot"):          # the strategies there are
        with_model_options(cfg, None, [], ["goal=tree_of_thought"])
    with pytest.raises(ConfigError, match="'gpt-9'"):                 # the entries are checked first
        with_model_options(cfg, None, ["goal=gpt-9"], ["goal=tree_of_thought"])

def listed(capsys, main, *argv):                          # the lines a command prints for --list-entries
    assert main([*argv, "--list-entries"]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    return out.splitlines()

def test_list_entries_prints_one_line_per_entry_and_runs_nothing(table, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(harness, "_run", lambda *args, **kwargs: pytest.fail("a run was started"))
    lines = listed(capsys, harness.main, "--config", str(table))      # neither --scenes nor --name is needed
    cfg = load_config(table)
    assert [line.split()[:3] for line in lines] == [[name, e.provider, e.model] for name, e in cfg.models.items()]
    assert len(lines) == 7 + 17 + 6 + 2 + 7 and not (tmp_path / "runs").exists()
    row = {line.split()[0]: line for line in lines}
    assert row["claude_haiku55"].split() == ["claude_haiku55", "anthropic", "claude-haiku-5-5", "effort", "none", "xy_pixel",
                                             "key", "missing", "(ANTHROPIC_API_KEY)"]
    assert row["mock"].split() == ["mock", "mock", "mock-centre", "effort", "low", "xy_1000", "no", "key", "needed"]
    assert row["soclaas_glm53_max"].split() == ["soclaas_glm53_max", "openai_compatible", "x-test-1", "effort", "max",
                                                "xy_1000", "key", "missing", "(SOCLAAS_API_KEY)"]
    assert row["gemini_fast"].endswith("key missing (GEMINI_API_KEY)")           # the safety net hides the real .env
    assert row["openai_strong"].endswith("key missing (OPENAI_API_KEY)") and row["mock_b"].endswith("no key needed")
    assert row["gateway"].endswith("key missing (GATEWAY_API_KEY)")
    assert len({line.index("effort ") for line in lines}) == 1        # the columns line up

def test_list_entries_says_which_keys_are_set_and_never_shows_one(table, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    env = tmp_path / ".env"
    env.write_text(f"GEMINI_API_KEY={KEY}\nOPENAI_API_KEY=\nGATEWAY_API_KEY='{KEY}-2'\nANTHROPIC_API_KEY={KEY}-3\n")
    monkeypatch.setattr(factory, "load_env", functools.partial(factory.load_env, env))
    lines = listed(capsys, harness.main, "--config", str(table))
    held = {line.split()[0]: line.split("  ")[-1].strip() for line in lines}
    assert (held["gemini_fast"], held["gemini_strong"]) == ("key set (GEMINI_API_KEY)",) * 2
    assert (held["claude_opus55"], held["claude_haiku55_low"]) == ("key set (ANTHROPIC_API_KEY)",) * 2
    assert (held["openai_fast"], held["gateway"]) == ("key missing (OPENAI_API_KEY)", "key set (GATEWAY_API_KEY)")
    assert held["soclaas_qwen38"] == "key missing (SOCLAAS_API_KEY)" and held["mock"] == "no key needed"
    assert KEY not in "\n".join(lines)

def smoke_on(monkeypatch, table):                         # sgi.smoke reads this table where it reads configs/default.yaml
    monkeypatch.setattr(smoke, "load_config", functools.partial(load_config, table))

def test_both_commands_list_the_same_entries(table, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    smoke_on(monkeypatch, table)
    monkeypatch.setattr(smoke, "make_client", lambda *args, **kwargs: pytest.fail("a client was built"))
    assert listed(capsys, smoke.main) == listed(capsys, harness.main, "--config", str(table))      # --model is not needed
    assert not (tmp_path / "runs").exists()
    for main in (smoke.main, harness.main):               # without it, what a command needs is still required
        with pytest.raises(SystemExit) as stop:
            main([])
        assert stop.value.code == 2 and "required" in capsys.readouterr().err

def test_smoke_takes_the_same_stage_models(table, scenes, tmp_path, monkeypatch, capsys):
    smoke_on(monkeypatch, table)
    scene = str(scenes / scene_dict()["scene_id"])
    assert smoke.main(["--model", "mock", "--stage-model", "goal=mock_b", "--scene", scene]) == 0
    out = capsys.readouterr().out
    assert "[plan] mock mock-centre, effort low, xy_1000, image 640x480\n" in out
    assert "[goal] mock mock-b, effort high, xy_1000, image 640x480\n" in out and (tmp_path / "runs" / "smoke.png").exists()
    asked = []                                            # the entry of each client built for --calibrate
    def recording(entry_name, cfg, backend=None):
        asked.append((entry_name, cfg.models[entry_name].point_format, cfg.stages["locate"].model))
        return factory.make_client(entry_name, cfg, backend)
    monkeypatch.setattr(smoke, "make_client", recording)
    assert smoke.main(["--model", "mock", "--stage-model", "locate=mock_b", "--calibrate", "--scene", scene]) == 0
    out = capsys.readouterr().out                         # the entry that locates is the one calibrated, in each format
    assert [line.split(":")[0] for line in out.splitlines() if ": median error " in line] == ["xy_1000", "xy_pixel"]
    assert asked == [("mock_b", "xy_1000", "mock_b"), ("mock_b", "xy_pixel", "mock_b")]       # and no other is given a client
    other = smoke._in_point_format(smoke._config("mock", ["locate=mock_b"]), "xy_pixel").models["mock"]
    assert other.point_format == "xy_1000"                # the other entry keeps its format

def test_smoke_builds_a_client_only_for_a_stage_it_calls(table, scenes, monkeypatch, capsys):
    smoke_on(monkeypatch, table)
    scene = str(scenes / scene_dict()["scene_id"])
    built = []                                            # the entry of each client built
    def recording(entry_name, cfg, backend=None):
        built.append(entry_name)
        return factory.make_client(entry_name, cfg, backend)
    monkeypatch.setattr(smoke, "make_client", recording)
    # --model leaves the context stage on the gateway entry, whose key is not held: the smoke run never calls that stage.
    called = [option for stage in ("plan", "locate", "goal") for option in ("--stage-model", f"{stage}=mock_b")]
    assert smoke.main(["--model", "gateway", *called, "--scene", scene]) == 0
    assert built == ["mock_b"] and "result: ok" in capsys.readouterr().out
    del built[:]                                          # a calibration calls the locate stage alone, in each point format
    assert smoke.main(["--model", "gateway", "--stage-model", "locate=mock_b", "--calibrate", "--scene", scene]) == 0
    assert built == ["mock_b", "mock_b"] and capsys.readouterr().err == ""

@pytest.mark.parametrize("option, value, stage", [
    ("--stage-model", "context=mock_b", "context"), ("--stage-model", " context = mock ", "context"),
    ("--strategy", "context=cot", "context")])
def test_smoke_refuses_an_option_for_a_stage_which_it_never_calls(table, scenes, tmp_path, monkeypatch, capsys, option, value, stage):
    smoke_on(monkeypatch, table)
    monkeypatch.setattr(smoke, "make_client", lambda *args, **kwargs: pytest.fail("a client was built"))
    for extra in ([], ["--calibrate"], ["--list-models"], ["--goal", "constraints"]):
        assert smoke.main(["--model", "mock", option, value, "--scene", str(scenes), *extra]) == 2
        out, err = capsys.readouterr()                    # the option would change nothing: said, not passed over
        assert out == "" and err.startswith(f"error: {option} {value.strip()}: ") and err.count("\n") == 1
        assert f"never calls the {stage} stage" in err and "python -m sgi.run" in err
    assert not (tmp_path / "runs").exists()

def test_smoke_takes_an_option_for_the_two_stages_of_the_constraint_method_which_one_of_its_runs_calls(table, scenes, monkeypatch, capsys):
    smoke_on(monkeypatch, table)
    scene = str(scenes / scene_dict()["scene_id"])
    options = ["--stage-model", "people=mock_b", "--strategy", "people=cot", "--strategy", "constraints=zero_shot"]
    assert smoke.main(["--model", "mock", "--goal", "constraints", *options, "--scene", scene]) == 0
    out = capsys.readouterr().out                         # the people call on its own entry, each of the two under its strategy
    assert "[people] mock mock-b, effort high, xy_1000, image 640x480\n" in out and "[people] key order: analysis, people" in out
    assert "[constraints] mock mock-centre, effort low, xy_1000, image 640x480\n" in out and "[constraints] key order: constraints, " in out
    assert smoke.main(["--model", "mock", *options, "--scene", scene]) == 0      # in a run that calls neither they are taken as well
    assert "[people]" not in capsys.readouterr().out

def test_smoke_takes_the_same_strategies(table, scenes, monkeypatch, capsys):
    smoke_on(monkeypatch, table)
    scene = str(scenes / scene_dict()["scene_id"])
    assert smoke.main(["--model", "mock", "--scene", scene]) == 0     # its own: the plan and the goal reason first
    out = capsys.readouterr().out
    assert "[plan] key order: reasoning, subtasks, " in out and "[goal] key order: analysis, analysis.arrangement, " in out
    assert smoke.main(["--model", "mock", "--strategy", "plan=zero_shot", "--strategy", "goal=zero_shot", "--scene", scene]) == 0
    out = capsys.readouterr().out
    assert "[plan] key order: subtasks, " in out and "[goal] key order: goal_point, goal_point.x, goal_point.y\n" in out

@pytest.mark.parametrize("value, named", BAD_STRATEGY)
def test_smoke_refuses_a_bad_strategy_before_anything_is_asked(table, scenes, tmp_path, monkeypatch, capsys, value, named):
    smoke_on(monkeypatch, table)
    monkeypatch.setattr(smoke, "make_client", lambda *args, **kwargs: pytest.fail("a client was built"))
    for extra in ([], ["--calibrate"], ["--list-models"]):
        assert smoke.main(["--model", "mock", "--strategy", value, "--scene", str(scenes), *extra]) == 2
        out, err = capsys.readouterr()
        assert out == "" and err.startswith("error: ") and err.count("\n") == 1 and named in err
    assert not (tmp_path / "runs").exists()

@pytest.mark.parametrize("value, named", BAD)
def test_smoke_refuses_a_bad_stage_model_before_anything_is_asked(table, scenes, tmp_path, monkeypatch, capsys, value, named):
    smoke_on(monkeypatch, table)
    monkeypatch.setattr(smoke, "make_client", lambda *args, **kwargs: pytest.fail("a client was built"))
    for extra in ([], ["--calibrate"], ["--list-models"]):
        assert smoke.main(["--model", "mock", "--stage-model", value, "--scene", str(scenes), *extra]) == 2
        out, err = capsys.readouterr()
        assert out == "" and err.startswith("error: ") and err.count("\n") == 1 and named in err
    assert not (tmp_path / "runs").exists()

def gateway(monkeypatch, status=200, body=None, error=None):
    """Give sgi.smoke, for --list-models, the real back end on an SDK client whose HTTP layer is this handler."""
    sent = []
    def handler(request):
        sent.append(request)
        if error:
            raise getattr(httpx2, error)(f"no route with {KEY}", request=request)
        return httpx2.Response(status, json=body)
    def injected(entry_name, cfg):
        entry = cfg.models[entry_name]
        client = openai.OpenAI(api_key=KEY, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
        return factory.list_models(entry_name, cfg, backend=OpenAICompatibleBackend(KEY, entry.base_url, client=client))
    monkeypatch.setattr(smoke, "list_models", injected)
    return sent

def test_list_models_prints_the_ids_the_provider_reports_one_per_line(table, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    smoke_on(monkeypatch, table)
    monkeypatch.setattr(smoke, "make_client", lambda *args, **kwargs: pytest.fail("a model was asked"))
    listing = {"object": "list", "data": [{"id": name, "object": "model"} for name in ("qwen3.8:27b", "bge-m3", "x-test-1")]}
    listing["data"][2]["gateway"] = {"alias_of": "glm-5.3"}      # said of one model beyond the usual: it follows the id
    sent = gateway(monkeypatch, body=listing)
    assert smoke.main(["--model", "gateway", "--list-models"]) == 0
    assert capsys.readouterr() == ('bge-m3\nqwen3.8:27b\nx-test-1  {"gateway":{"alias_of":"glm-5.3"}}\n', "")
    assert [(request.method, str(request.url)) for request in sent] == [("GET", BASE + "/models")]
    assert not (tmp_path / "runs").exists()

@pytest.mark.parametrize("reply, code, said", [
    (dict(status=401, body={"error": {"message": f"Incorrect API key provided: {KEY}"}}), 1, "401 from the model listing"),
    (dict(status=502, body={"error": {"message": "provider request failed"}}), 1, "502 from the model listing"),
    (dict(status=429, body={"error": {"message": "Budget limit reached"}}), 2, "429 from the model listing"),     # used up
    (dict(error="ConnectError"), 1, f"the model listing at {BASE}")])
def test_a_failed_model_listing_is_one_line_without_the_key(table, tmp_path, monkeypatch, capsys, reply, code, said):
    monkeypatch.chdir(tmp_path)
    smoke_on(monkeypatch, table)
    sent = gateway(monkeypatch, **reply)
    assert smoke.main(["--model", "gateway", "--list-models"]) == code
    out, err = capsys.readouterr()
    assert out == "" and err.startswith(f"error: {said}") and err.count("\n") == 1 and len(sent) == 1
    assert KEY not in err and "Traceback" not in err

def gemini_on(handler):                                   # the Gemini back end on an SDK client whose HTTP layer is the handler
    options = genai_types.HttpOptions(httpx_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return GeminiBackend(KEY, client=genai.Client(api_key=KEY, http_options=options))

def openai_on(handler):                                   # and the OpenAI back end on one
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    return OpenAIBackend(KEY, client=openai.OpenAI(api_key=KEY, http_client=http))

def anthropic_on(handler):                                # and the Anthropic back end on one
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    return AnthropicBackend(KEY, client=anthropic.Anthropic(api_key=KEY, http_client=http))

def claude(name):                                         # a model as Anthropic's listing gives one
    return {"type": "model", "id": name, "display_name": name, "created_at": "2026-09-01T00:00:00Z"}

LISTED = [                                                # entry, its back end, the HTTP library under it, the reply, what is printed
    ("gemini_fast", gemini_on, httpx, {"models": [{"name": "models/gemini-3.8-flash"}, {"name": "models/aqa"}]},
     "aqa\ngemini-3.8-flash\n"),
    ("openai_strong", openai_on, httpx2, {"object": "list", "data": [{"id": "gpt-6.1-sol", "object": "model"},
                                                                    {"id": "embed-4", "object": "model"}]},
     "embed-4\ngpt-6.1-sol\n"),
    ("claude_haiku55", anthropic_on, httpx2, {"data": [claude("claude-sonnet-5-5"), claude("claude-haiku-5-5")],
                                              "has_more": False, "first_id": "claude-sonnet-5-5", "last_id": "claude-haiku-5-5"},
     "claude-haiku-5-5\nclaude-sonnet-5-5\n")]

@pytest.mark.parametrize("entry, backend_on, lib, listing, printed", LISTED, ids=[row[0] for row in LISTED])
def test_list_models_works_on_an_entry_of_each_sdk_provider(table, tmp_path, monkeypatch, capsys, entry, backend_on, lib,
                                                            listing, printed):
    monkeypatch.chdir(tmp_path)
    smoke_on(monkeypatch, table)
    monkeypatch.setattr(smoke, "make_client", lambda *args, **kwargs: pytest.fail("a model was asked"))
    sent = []
    def handler(request):
        sent.append(request)
        return lib.Response(200, json=listing)
    def injected(entry_name, cfg):                        # as the gateway above: the real back end, and no key read
        return factory.list_models(entry_name, cfg, backend=backend_on(handler))
    monkeypatch.setattr(smoke, "list_models", injected)
    assert smoke.main(["--model", entry, "--list-models"]) == 0
    assert capsys.readouterr() == (printed, "")           # the ids as a model entry writes them, sorted
    assert [request.method for request in sent] == ["GET"] and not (tmp_path / "runs").exists()

@pytest.mark.parametrize("name", SHELL)
def test_a_gateway_entry_is_refused_while_the_shell_holds_headers_for_the_sdk(table, scenes, tmp_path, monkeypatch, capsys, name):
    smoke_on(monkeypatch, table)
    monkeypatch.setattr(factory, "require_key", lambda variable: KEY)             # .env holds the entry's key
    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", lambda self, request: pytest.fail("a request was sent"))
    monkeypatch.setenv(name, "Authorization: Bearer sk-of-another-service")       # set for another tool
    commands = [lambda: run_main(table, scenes, "--model", "mock", "--stage-model", "context=gateway"),
                lambda: smoke.main(["--model", "gateway", "--scene", str(scenes)]),
                lambda: smoke.main(["--model", "gateway", "--calibrate", "--scene", str(scenes)]),
                lambda: smoke.main(["--model", "gateway", "--list-models"])]
    for command in commands:                              # each stops before a client exists that would send it on
        assert command() == 2
        out, err = capsys.readouterr()
        line = err[err.index("error: "):]                 # sgi.run logs how many episodes it skipped before that
        assert out == "" and line.count("\n") == 1 and "Traceback" not in err and f"env -u {name} " in line
        assert BASE in line and "another-service" not in err and KEY not in err
    assert not (tmp_path / "runs").exists()               # nothing was written, by the run or by the smoke

def test_list_models_needs_the_key_and_a_back_end_that_lists(table, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    smoke_on(monkeypatch, table)
    for entry, named in (("gateway", "GATEWAY_API_KEY"),              # the safety net: no key is held
                         ("mock", "the mock back end has no model listing"), ("gpt-9", "'gpt-9'")):
        assert smoke.main(["--model", entry, "--list-models"]) == 2
        out, err = capsys.readouterr()
        assert out == "" and err.startswith("error: ") and err.count("\n") == 1 and named in err
