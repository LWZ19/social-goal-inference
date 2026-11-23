import importlib.util
import inspect
import re
import shlex
import subprocess
import typing
from pathlib import Path
from pydantic import BaseModel
from sgi import scene
from sgi.config import REPO_ROOT, ModelEntry, RunConfig, StageName, Strategy, load_config
from sgi.types import ConstraintKind, ConstraintUse, Flags, GoalResult
from sgi.vlm import compatible, factory

FRONT = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
README = (REPO_ROOT / "docs" / "usage.md").read_text(encoding="utf-8")      # the full reference
PAGE = (REPO_ROOT / "docs" / "scene_format.md").read_text(encoding="utf-8")
CFG = load_config()
PROVIDERS = typing.get_args(ModelEntry.model_fields["provider"].annotation)

def commands(text):                                       # (module, arguments) of each "<python> -m ..." line of a bash block
    lines = [line for block in re.findall(r"^```bash\n(.*?)^```", text, re.S | re.M) for line in block.splitlines()]
    return [(w[2], w[3:]) for w in (shlex.split(line, comments=True) for line in lines) if w[1:2] == ["-m"]]

def test_documented_commands_name_real_modules_flags_paths_and_model_entries():
    found = commands(FRONT) + commands(README) + commands(PAGE)
    assert {m for m, _ in commands(README)} >= {"venv", "sgi.synth.generate", "sgi.check_scene", "sgi.run", "sgi.smoke", "sgi.rescore"}
    assert ("sgi.check_scene", ["docs/example_scene"]) in commands(PAGE)
    for module, args in found:
        spec = importlib.util.find_spec(module)
        assert spec is not None, module
        source = Path(spec.origin).read_text(encoding="utf-8")
        assert [a for a in args if a.startswith("--") and not re.search(rf"""['"]{a}['"]""", source)] == [], module
        assert [a for a in args if a.startswith("docs/") and not (REPO_ROOT / a).exists()] == [], module
    models = {args[args.index("--model") + 1] for _, args in found if "--model" in args}
    per_stage = [args[i + 1].split("=") for _, args in found for i, a in enumerate(args) if a == "--stage-model"]
    assert per_stage and {stage for stage, _ in per_stage} <= set(CFG.stages)             # STAGE=ENTRY, each a real one
    models |= {entry for _, entry in per_stage}
    assert {CFG.models[m].provider for m in models} == set(PROVIDERS)                     # KeyError: no such entry
    per_strategy = [args[i + 1].split("=") for _, args in found for i, a in enumerate(args) if a == "--strategy"]
    assert per_strategy and {stage for stage, _ in per_strategy} <= set(CFG.stages)       # STAGE=NAME, each a real one
    assert {name for _, name in per_strategy} <= set(typing.get_args(Strategy))
    assert any(module == "sgi.run" and "--prompts" in args for module, args in found)
    assert any(module == "sgi.smoke" and "--calibrate" in args for module, args in found)
    for module, option in (("sgi.run", "--list-entries"), ("sgi.smoke", "--list-models")):
        assert any(m == module and option in args for m, args in found), option

def section(title):                                       # the text under one "## " heading of the usage page
    return README.split(f"\n## {title}\n")[1].split("\n## ")[0]

def test_usage_covers_the_run_modes_and_the_run_folder():
    words = [w for name in ("grounding", "context_check", "goal", "people", "constraints", "grounding_source")
             for w in (name, *typing.get_args(RunConfig.model_fields[name].annotation))]
    words += ["trials", "leave_out_situation", "--final-test", "config.yaml", "results.jsonl", "summary.csv"]
    assert len(words) == 27 and [w for w in words if f"`{w}`" not in section("Run modes") + section("Reading a run folder")] == []
    rows = {row.split("|")[1].strip(): row for row in section("Run modes").splitlines() if row.startswith("| `")}
    assert list(rows)[:5] == ["`grounding`", "`context_check`", "`goal`", "`people`", "`constraints`"]
    # Each of the constraint method's two own settings says when it is read.
    assert "constraint method" in rows["`goal`"]
    assert "`goal: constraints`" in rows["`people`"] and "`goal: constraints`" in rows["`constraints`"]
    assert "no model call" in rows["`people`"] and "the solver runs on its own rules alone" in rows["`constraints`"]
    keyless = ["--model mock"]                            # the keyless path
    assert [w for w in keyless if f"`{w}`" not in section("Commands")] == []
    report = ["latency_s", "latency_total_s", "transient_retry_share", "model", "reasoning_effort"]  # the latency columns
    assert [w for w in report if f"`{w}`" not in section("Reading a run folder")] == []

def test_readme_says_how_to_switch_models_and_what_state_each_provider_is_in():
    text = section("Switching models")
    words = ["--model", "--stage-model", "--list-entries", "--list-models", "--config", "base_url", "api_key_env",
             "structured_output", *PROVIDERS]
    assert len(words) == 13 and [w for w in words if not re.search(rf"`{w}[ `]", text)] == []
    table = {row.split("|")[1].strip(): row for row in text.splitlines() if row.startswith("| ")}
    assert {"Gemini", "SoCLaaS", "OpenAI", "Claude"} <= set(table)   # a row each, with its key variable and its state
    assert "`GEMINI_API_KEY`" in table["Gemini"] and "live calls" in table["Gemini"]
    assert "`SOCLAAS_API_KEY`" in table["SoCLaaS"] and "full runs" in table["SoCLaaS"]
    assert "prices" in table["SoCLaaS"]
    assert sum(name.startswith("soclaas_") for name in CFG.models) == 17 and "the seventeen `soclaas_*` entries" in table["SoCLaaS"]
    assert "`OPENAI_API_KEY`" in table["OpenAI"] and "Not used for the reported results" in table["OpenAI"]
    claude = [name for name, entry in CFG.models.items() if entry.provider == "anthropic"]
    assert "`ANTHROPIC_API_KEY`" in table["Claude"] and "live runs" in table["Claude"] and "`anthropic`" in table["Claude"]
    assert len(claude) == 9 and [name for name in claude if f"`{name}`" not in table["Claude"]] == []
    # What an entry of that provider sends for its level, and the two settings it may not have.
    (levels,) = [p for p in text.split("\n\n") if p.startswith("For an `anthropic` entry")]
    said = ["reasoning_effort", "none", "low", "medium", "high", "xhigh", "max", "temperature", "image_detail", "max_output_tokens"]
    assert [w for w in said if f"`{w}`" not in levels] == [] and "16000" in levels and "thinking switched off" in levels
    assert "(../results/README.md)" in text and (REPO_ROOT / "results" / "README.md").is_file()

def test_readme_says_how_to_choose_strategies_and_prompt_folders_and_what_pacing_and_cost_are():
    text = section("Switching models")
    stages = typing.get_args(StageName)                   # all six: the four of every run and the two of goal: constraints
    words = ["--strategy", "--prompts", *typing.get_args(Strategy), *stages, "prompts_dir", "max_requests_per_minute",
             "price_input_per_mtok", "price_output_per_mtok", "calls", "cached_calls", "cost_usd"]
    assert len(words) == 19 and [w for w in words if not re.search(rf"`{w}[ `]", text)] == []
    assert [f"{stage}.md" for stage in stages if f"`{stage}.md`" not in text] == []        # what a prompts folder holds
    limits = {entry.max_requests_per_minute for name, entry in CFG.models.items() if name.startswith("soclaas_")}
    assert limits == {80} and "80 a minute" in text and "one after another" in text       # the number and the pacing rule

def test_readme_says_that_every_real_provider_lists_its_models_and_what_a_line_of_a_gateway_carries():
    (listing,) = [p for p in section("Switching models").split("\n\n") if p.startswith("`--list-models` ")]
    assert [p for p in PROVIDERS if f"`{p}`" not in listing] == []
    assert "every provider" in listing and "300 characters" in listing      # and the one that has nothing to list
    assert re.search(r"`mock` entry has no(thing| models) to list", listing)

def test_readme_says_how_a_spent_allowance_stops_a_run_and_how_to_resume():
    text = section("Reading a run folder")
    (stop,) = [p for p in text.split("\n\n") if "allowance is used up" in p]
    words = ["openai", "openai_compatible", "results.jsonl", "summary.csv", "config.yaml", "incomplete: true", "stopped_by",
             "error:", "--no-cache"]
    assert [w for w in words if f"`{w}`" not in stop] == []
    assert all(word in stop for word in ("429", "quota", "budget", "rate limit", "code 2", "the same command again"))
    assert "`stopped_by`" in text.split("\n\n")[1]       # listed with what else config.yaml holds beside the configuration
    assert "allowance is used up" in section("Commands") and "(#reading-a-run-folder)" in section("Commands")

def test_readme_says_how_to_score_a_run_again_and_both_pages_give_the_distances_of_the_regions():
    part = README.split("\n### Score a finished run again\n")[1].split("\n#")[0]
    assert ("sgi.rescore", ["runs/20261009-101500_baseline_gemini"]) in commands(part)
    assert any(module == "sgi.rescore" and "--scenes" in args for module, args in commands(part))
    words = ["rescored/", "results.jsonl", "summary.csv", "note.txt", "config.yaml", "--scenes", "error:", "success", "score",
             "hidden_relevant_people"]
    assert [w for w in words if f"`{w}`" not in part] == []
    assert all(said in part for said in ("no model call", "code 2", "nothing is written", "generate the test scenes again"))
    assert "(#score-a-finished-run-again)" in section("Reading a run folder")
    rules = [line for line in PAGE.splitlines() if line.startswith(("- **`no_one`.**", "- **`waiting_line`.**", "- **`conversation_group`**"))]
    assert len(rules) == 3 and all("0.5 to 1.2 m" in rule for rule in rules) and "0.5 to 1.2 m" in part
    assert [old for old in ("0.9", "1.9", "0.8 m wide", "0.3 to", "60 degrees") if any(old in rule for rule in rules)] == []
    assert all("Where Should I Join?" in text for text in (part, rules[1]))      # whose distances they are
    assert "not ahead of" in rules[1] and "the way the line faces" in rules[1] and "on any side" in rules[2]
    # A region begins nearer a person than the robot can come: both pages say where the part that can be reached starts.
    assert all("0.55 m" in text and "0.25 m" in text and "0.30 m" in text for text in (part, rules[1]))
    assert "`in_region`" in part and "`reachable`" in part
    # The pictures stay and the worked examples' goal points follow the regions: what that does to the cache.
    assert all(said in part for said in ("`few_shot`", "`few_shot_cot`", "new one to the cache"))

def test_readme_says_that_smoke_leaves_the_context_stage_alone():
    (smoke,) = [p for p in section("Switching models").split("\n\n") if p.startswith("`sgi.smoke` takes `--model`")]
    assert "never calls the `context` stage" in smoke and "`error:`" in smoke and "code 2" in smoke
    assert "with `--goal constraints` it calls `people` and `constraints`" in smoke     # the two it calls in that run alone

def test_readme_says_how_to_run_the_constraint_method_and_what_a_record_of_it_holds():
    part = README.split("\n### Run the constraint method\n")[1].split("\n### ")[0]
    found = commands(part)                                # a run on a copy of the config, keyless and not, and the check by eye
    assert [module for module, _ in found] == ["sgi.run", "sgi.run", "sgi.smoke"] and all("--config" in args for _, args in found[:2])
    assert found[0][1][found[0][1].index("--model") + 1] == "mock" and found[2][1][2:4] == ["--goal", "constraints"]
    assert "--scene" in found[2][1] and "has nobody in view" in part           # on a scene with people: the default has none
    assert "set goal: constraints under run:" in part and "(#run-modes)" in part and "(#reading-a-run-folder)" in part
    words = ["goal: constraints", "people: vlm", "people: oracle", "constraints: none", "people", "constraints", "max_people",
             "max_constraints", "standoff_m", "candidate_count", "prompts/people.md", "prompts/constraints.md",
             "runs/smoke_numbered.png", "--scene", *typing.get_args(ConstraintKind),
             *typing.get_args(ConstraintUse.model_fields["status"].annotation)]
    assert len(words) == 24 and [w for w in words if f"`{w}`" not in part] == []
    assert all(said in part for said in ("by the numbers drawn on them", "share one cached answer", "makes no model call",
                                         "the difference between the two", "the solver runs alone in every episode"))
    assert "(#run-the-constraint-method)" in README.split("\n## Setup\n")[0]        # the introduction points to it
    (record,) = [line for line in section("Reading a run folder").splitlines() if line.startswith("- A result of the constraint method")]
    fields = ["source: constraints", *list(GoalResult.model_fields)[-4:], "number", "xy", "yaw", "pixel", "label", "constraint", "status",
              "reason", "broken_m", "pose_world", "keys", "own_space_broken_m", "joined_waiting", "not_ahead_of",
              "not_ahead_reason", "target_distance_m", *list(Flags.model_fields)[-6:], "goal_mark", "goal_pixel"]
    assert len(fields) == 29 and [w for w in fields if f"`{w}`" not in record] == []
    assert "solved again without a model call" in record and "no pose" in record
    (picture,) = [line for line in section("Reading a run folder").splitlines() if line.startswith("- `viz/")]
    assert all(said in picture for said in ("the number of each person found", "runners-up", "the darker a cell", "solved again from the record"))
    answers = [row for row in section("Commands").splitlines() if row.startswith(("| people |", "| constraints |"))]
    assert len(answers) == 2 and all(f"`{key}`" in answers[1] for key in ("because", "kind", "people", "range_m", "target", "at_least"))
    (folder,) = [line for line in section("Reading a run folder").splitlines() if line.startswith("- `prompts/`")]
    assert "`constraints.md`" in folder and "`people.md`" in folder and "`people: vlm`" in folder

def test_env_example_names_every_key_variable_and_holds_no_value():
    lines = (REPO_ROOT / ".env.example").read_text(encoding="utf-8").split()
    assert lines and all(line.endswith("=") and line.count("=") == 1 for line in lines)    # a name, and nothing after it
    named = {entry.api_key_env for entry in CFG.models.values() if entry.api_key_env}
    fixed = set(factory._PROVIDER_KEYS.values())          # of the providers whose key variable is theirs alone
    assert named and fixed == {"GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
    assert {line[:-1] for line in lines} == {*fixed, *named}
    assert [name for name in sorted(fixed | named) if f"`{name}`" not in section("Setup")] == []   # and the README says to fill it in
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert ".env" in ignored and "soclaas/" in ignored                # the keys, and local service settings

def test_setup_names_each_shell_variable_a_gateway_entry_is_not_built_under():
    held = compatible._SHELL_HEADERS                      # the back end refuses each: the README says which, and the way out
    assert len(held) == 3 and [name for name in held if f"`{name}`" not in section("Setup")] == []
    assert "`openai_compatible`" in section("Setup") and "`env -u`" in section("Setup")

def test_setup_creates_the_key_file_once_and_never_overwrites_it(tmp_path):
    copy = "[ -e .env ] || cp .env.example .env"          # the one Setup line that writes .env, which holds API keys
    (block,) = re.findall(r"^```bash\n(.*?)^```", section("Setup"), re.S | re.M)
    assert [line for line in block.splitlines() if ".env" in line] == [copy]
    assert "an existing `.env`" in section("Setup")       # and the text says that it is left alone
    def setup_again():                                    # the line, run where a clone would run it: silent, exit code 0
        done = subprocess.run(["sh", "-c", copy], cwd=tmp_path, capture_output=True, text=True)
        assert (done.returncode, done.stdout, done.stderr) == (0, "", "")
        return (tmp_path / ".env").read_text()
    (tmp_path / ".env.example").write_text("GEMINI_API_KEY=\nOPENAI_API_KEY=\n")
    assert setup_again() == "GEMINI_API_KEY=\nOPENAI_API_KEY=\n"         # a fresh clone: created from the example
    (tmp_path / ".env").write_text("GEMINI_API_KEY=a-real-key\n")
    assert setup_again() == "GEMINI_API_KEY=a-real-key\n"    # the block run a second time: kept

def test_scene_format_page_names_every_field_and_its_worked_example():
    names = {field for name, c in inspect.getmembers(scene, inspect.isclass) if issubclass(c, BaseModel)
             and c.__module__ == scene.__name__ and not name.startswith("_") for field in c.model_fields}
    assert len(names) == 70 and sorted(n for n in names if not re.search(rf'[`"]{n}[`"]', PAGE)) == []
    example = scene.read_scene_file(REPO_ROOT / "docs" / "example_scene")
    recipe = ["T = inverse(transpose(M))", "T[:3, 3] *= metersPerSceneUnit", "T_world_camera = T @ diag(1, -1, -1, 1)",
              "fx = fy = W * cameraFocalLength / cameraAperture[0]"]       # character for character: no paraphrase
    words = [*CFG.dev_situations, *CFG.held_out_situations, "ahead_in_line", "group_space", "cameraViewTransform",
             "metersPerSceneUnit", "distance_to_image_plane", "camera_params", "check.png", example.scene_id, *recipe,
             *(t.label for t in example.ground_truth.targets), *(e.request for e in example.episodes)]
    assert [w for w in words if w not in PAGE] == []

def test_front_page_is_short_and_points_to_the_reference_pages():
    assert len(FRONT.split()) < 1000
    links = re.findall(r"\]\((?!https?:)([^)#\s]+)", FRONT)
    assert {"docs/usage.md", "docs/constraint_method.md", "results/README.md", "sim/README.md"} <= set(links)
    assert [link for link in links if not (REPO_ROOT / link).exists()] == []
