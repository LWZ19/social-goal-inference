import typing

import pytest
import yaml

from sgi.config import (DEFAULT_CONFIG, PROMPTS_DIR, REPO_ROOT, Config, ConfigError, ModelEntry, RunConfig, StageConfig,
                        StageName, Thresholds, load_config, with_model, with_stage_models, with_stage_strategies)

STAGES = ("plan", "locate", "context", "goal")
METHOD_STAGES = ("people", "constraints")                          # of the constraint method: a config may leave them out
SITUATIONS = (["no_one", "waiting_line", "conversation_group", "audience"], ["group_photo", "lift_wait", "machine_use"])
RUN = dict(grounding="vlm", context_check="vlm", goal="direct_marking", people="vlm", constraints="vlm",
           grounding_source="view", heading_mode="face_target", trials=1, leave_out_situation=None, final_test=False, examples_dir=None,
           k=None, seed=0, max_tour_frames=16, max_image_side=None, prompts_dir="prompts", use_cache=True,
           cache_dir=".cache/vlm")


def test_defaults_in_code_and_in_the_yaml():
    assert Thresholds().model_dump() == dict(
        standoff_m=1.0, standard_goal_max_slide_m=0.5, standard_goal_step_m=0.05, standard_goal_clearance_m=0.40,
        surface_radius_m=0.30, surface_min_points=40, surface_min_spread_m=0.05, surface_max_tilt=0.5,
        surface_max_rms_m=0.06, surface_max_rms_ratio=0.5,
        floor_tolerance_m=0.15, lift_window=5, heading_degenerate_m=0.05, robot_radius_m=0.30,
        person_radius_m=0.25, grid_resolution_m=0.05, match_tolerance_m=0.3, default_match_radius_m=0.75,
        personal_gap_m=0.3, anchor_depth_tolerance_m=0.3, relevant_radius_m=2.0,
        visibility_depth_tolerance_m=0.3, visibility_min_samples=3, floor_depth_max_rel_error=0.02,
        floor_min_pixels=2000, floor_person_clearance_m=0.4, camera_height_min_m=0.2, camera_height_max_m=3.0,
        yaw_tolerance_rad=0.001, free_space_agreement_min=0.90,
        view_half_angle_deg=35.0, visible_half_angle_deg=60.0, floor_range_m=12.0, seen_depth_ratio=0.03,
        seen_depth_margin_m=0.02, max_keep_m=3.0, max_people=40, max_constraints=12, person_mark_height_m=1.1,
        candidate_count=4,                                         # the ten before: the constraint method's
        join_facing_deg=None, join_spacing_ratio=None)             # the joining gate, off by default
    assert RunConfig().model_dump() == RUN
    assert yaml.safe_load(DEFAULT_CONFIG.read_text())["thresholds"] == Thresholds().model_dump()   # all spelt out
    cfg = load_config()
    assert cfg.run.model_dump() == {**RUN, "examples_dir": "data/placeholder/examples"}
    assert {k: (e.provider, e.model) for k, e in cfg.models.items() if not k.startswith("soclaas_")} == {
        "mock": ("mock", "mock-centre"), "gemini_fast": ("gemini", "gemini-3.8-flash"),
        "gemini_strong": ("gemini", "gemini-3.1-pro-preview"), "gemini_35_flash": ("gemini", "gemini-3.5-flash"),
        "gemini_pointing": ("gemini", "gemini-robotics-er-2-preview"),
        "openai_fast": ("openai", "gpt-6-luna"), "openai_strong": ("openai", "gpt-6.1-sol"),
        "claude_opus55": ("anthropic", "claude-opus-5-5"), "claude_opus55_low": ("anthropic", "claude-opus-5-5"),
        "claude_sonnet55": ("anthropic", "claude-sonnet-5-5"), "claude_sonnet55_low": ("anthropic", "claude-sonnet-5-5"),
        "claude_haiku55": ("anthropic", "claude-haiku-5-5"), "claude_haiku55_low": ("anthropic", "claude-haiku-5-5"),
        # older models, for how much the method depends on the model
        "claude_haiku45": ("anthropic", "claude-haiku-4-5"), "claude_sonnet46_low": ("anthropic", "claude-sonnet-4-6"),
        "claude_sonnet5_low": ("anthropic", "claude-sonnet-5"),
        "gemini3_flash": ("gemini", "gemini-3-flash-preview"), "gemini31_flash_lite": ("gemini", "gemini-3.1-flash-lite"),
        "gemini35_flash_lite": ("gemini", "gemini-3.5-flash-lite"), "gemini36_flash": ("gemini", "gemini-3.6-flash")}
    assert cfg.models["mock"].model_dump(exclude={"provider", "model"}) == dict(
        base_url=None, api_key_env=None, structured_output="json_schema", max_requests_per_minute=None,
        price_input_per_mtok=None, price_output_per_mtok=None,
        reasoning_effort="low", point_format="xy_1000", temperature=None, image_detail=None,
        max_output_tokens=None, timeout_s=180.0)
    assert [(s, cfg.stages[s].model, cfg.stages[s].strategy) for s in STAGES] == [
        ("plan", "mock", "cot"), ("locate", "mock", "zero_shot"), ("context", "mock", "zero_shot"),
        ("goal", "mock", "few_shot")]
    assert (cfg.dev_situations, cfg.held_out_situations) == SITUATIONS
    assert (PROMPTS_DIR, DEFAULT_CONFIG) == (REPO_ROOT / "prompts", REPO_ROOT / "configs" / "default.yaml")
    assert (REPO_ROOT / "pyproject.toml").is_file()


@pytest.mark.parametrize("overrides, named", [
    ({"stages": {"goal": {"model": "absent"}}}, "absent"),
    ({"stages": {"plan": {"strategy": "few_shot"}}}, "plan"),
    ({"stages": {"locate": {"strategy": "few_shot_cot"}}}, "locate"),
    ({"stages": {"context": {"strategy": "few_shot"}}}, "context"),
    ({"stages": {"people": {"model": "mock", "strategy": "few_shot"}}}, "stage 'people' accepts only"),
    ({"stages": {"constraints": {"model": "mock", "strategy": "few_shot_cot"}}}, "stage 'constraints' accepts only"),
    ({"stages": {"people": {"model": "absent", "strategy": "zero_shot"}}}, "absent"),
    ({"stages": {"constraints": {"model": "mock"}}}, "strategy"),  # named, a stage is named whole
    ({"stages": {"judge": {"model": "mock", "strategy": "zero_shot"}}}, "judge"),
    ({"run": {"people": "detector"}}, "'vlm' or 'oracle'"),       # each of the two new settings takes its two values
    ({"run": {"constraints": "oracle"}}, "'vlm' or 'none'"),
    ({"thresholds": {"max_peple": 20}}, "max_peple"),
    ({"run": {"leave_out_situation": "group_photo"}}, "group_photo"),
    ({"run": {"grounding_source": "tour", "grounding": "oracle"}}, "tour"),
    ({"held_out_situations": ["audience", "group_photo", "lift_wait"]}, "audience"),
    ({"thresholds": {"standoff": 1.2}}, "standoff"),               # a misspelt key is an error, never ignored
    ({"run": {"max_image_side": 0}}, "max_image_side"),            # would send 1 x 1 images; "no resizing" is null
    ({"run": {"max_image_side": -1}}, "max_image_side"),
    ({"run": {"max_tour_frames": 0}}, "max_tour_frames"),          # a plan on a tour needs a frame to show
])
def test_each_rule_raises_config_error_naming_the_culprit(overrides, named):
    with pytest.raises(ConfigError, match=named):
        load_config(overrides=overrides)


def test_overrides_merge_key_by_key():
    cfg = load_config(overrides={
        "run": {"trials": 3, "grounding_source": "tour", "leave_out_situation": "audience"},
        "thresholds": {"standoff_m": 1.2}, "stages": {"goal": {"strategy": "few_shot_cot"}},
        "dev_situations": ["audience", "no_one"]})
    assert (cfg.run.trials, cfg.run.grounding_source, cfg.run.examples_dir) == (3, "tour", "data/placeholder/examples")
    assert (cfg.thresholds.standoff_m, cfg.thresholds.robot_radius_m) == (1.2, 0.30)
    assert cfg.stages["goal"] == StageConfig(model="mock", strategy="few_shot_cot")       # the model is kept
    assert cfg.dev_situations == ["audience", "no_one"]            # a list is replaced whole
    assert load_config().run.trials == 1                           # nothing is remembered between calls


def test_a_config_holds_six_stages_and_the_two_it_may_leave_out_follow_the_goal_stage():
    cfg = load_config()                                            # configs/default.yaml names the first four
    assert list(cfg.stages) == [*STAGES, *METHOD_STAGES] == list(typing.get_args(StageName))
    assert [cfg.stages[s] for s in METHOD_STAGES] == [StageConfig(model="mock", strategy="zero_shot"),
                                                      StageConfig(model="mock", strategy="cot")]
    moved = load_config(overrides={"stages": {"goal": {"model": "gemini_fast", "strategy": "few_shot_cot"}}})
    assert [(moved.stages[s].model, moved.stages[s].strategy) for s in METHOD_STAGES] == [
        ("gemini_fast", "zero_shot"), ("gemini_fast", "cot")]      # the goal stage's entry, never its strategy
    assert moved.stages["plan"].model == "mock"
    named = load_config(overrides={"stages": {"goal": {"model": "gemini_fast"},
                                              "people": {"model": "openai_fast", "strategy": "cot"}}})
    assert [(named.stages[s].model, named.stages[s].strategy) for s in METHOD_STAGES] == [
        ("openai_fast", "cot"), ("gemini_fast", "cot")]            # one that is named is taken as it is named
    entry = ModelEntry(provider="mock", model="x")
    four = {s: StageConfig(model="n" if s == "goal" else "m", strategy="zero_shot") for s in STAGES}
    built = Config(models={"m": entry, "n": entry}, stages=four)   # built in code, as loaded from a file
    assert list(built.stages) == [*STAGES, *METHOD_STAGES] and list(four) == list(STAGES)     # the argument is not changed
    assert [(built.stages[s].model, built.stages[s].strategy) for s in METHOD_STAGES] == [("n", "zero_shot"), ("n", "cot")]
    assert Config.model_validate(built.model_dump(mode="json")) == built      # written with six, as a run folder records it


def test_the_run_settings_of_the_constraint_method():
    assert typing.get_args(RunConfig.model_fields["goal"].annotation) == ("direct_marking", "oracle", "constraints")
    assert typing.get_args(RunConfig.model_fields["people"].annotation) == ("vlm", "oracle")
    assert typing.get_args(RunConfig.model_fields["constraints"].annotation) == ("vlm", "none")
    run = load_config(overrides={"run": {"goal": "constraints", "people": "oracle", "constraints": "none"}}).run
    assert (run.goal, run.people, run.constraints) == ("constraints", "oracle", "none")
    kept = load_config(overrides={"run": {"people": "oracle", "constraints": "none"}}).run    # read, and unused by the baseline
    assert (kept.goal, kept.people, kept.constraints) == ("direct_marking", "oracle", "none")
    th = load_config(overrides={"thresholds": {"max_people": 20, "view_half_angle_deg": 40}}).thresholds
    assert (th.max_people, th.view_half_angle_deg, th.max_constraints) == (20, 40.0, 12)


def test_with_model_points_every_stage_at_one_entry():
    cfg = load_config()
    new = with_model(cfg, "gemini_fast")
    assert {stage.model for stage in new.stages.values()} == {"gemini_fast"} and len(new.stages) == 6     # the two new ones too
    assert [new.stages[s].strategy for s in METHOD_STAGES] == ["zero_shot", "cot"]
    assert [(new.stages[s].model, new.stages[s].strategy) for s in STAGES] == [
        ("gemini_fast", "cot"), ("gemini_fast", "zero_shot"), ("gemini_fast", "zero_shot"), ("gemini_fast", "few_shot")]
    assert (new.run, new.thresholds, new.models) == (cfg.run, cfg.thresholds, cfg.models)
    assert cfg.stages["goal"].model == "mock"                      # the argument is not changed
    with pytest.raises(ConfigError, match="gpt-9"):
        with_model(cfg, "gpt-9")


def test_with_stage_models_points_only_the_named_stages_at_their_entries():
    cfg = load_config()
    new = with_stage_models(cfg, {"context": "gemini_fast", "goal": "openai_strong"})
    assert [(new.stages[s].model, new.stages[s].strategy) for s in STAGES] == [
        ("mock", "cot"), ("mock", "zero_shot"), ("gemini_fast", "zero_shot"), ("openai_strong", "few_shot")]
    assert (new.run, new.thresholds, new.models) == (cfg.run, cfg.thresholds, cfg.models)
    assert {cfg.stages[s].model for s in STAGES} == {"mock"}       # the argument is not changed
    assert with_stage_models(cfg, {}) == cfg                       # nothing named, nothing changed
    assert {new.stages[s].model for s in METHOD_STAGES} == {"mock"}        # the goal stage alone was named: they stay
    both = with_stage_models(cfg, {"people": "gemini_fast", "constraints": "openai_strong"})
    assert [both.stages[s].model for s in (*STAGES, *METHOD_STAGES)] == ["mock"] * 4 + ["gemini_fast", "openai_strong"]
    with pytest.raises(ConfigError) as refusal:                    # an unknown stage: named, with the six there are
        with_stage_models(cfg, {"goal": "gemini_fast", "judge": "mock"})
    assert "'judge'" in str(refusal.value) and "plan, locate, context, goal, people, constraints" in str(refusal.value)
    with pytest.raises(ConfigError) as refusal:                    # an unknown entry: named, with every entry there is
        with_stage_models(cfg, {"goal": "gpt-9"})
    assert "'gpt-9'" in str(refusal.value) and ", ".join(cfg.models) in str(refusal.value)


def test_with_stage_strategies_changes_only_the_named_stages():
    cfg = load_config()
    new = with_stage_strategies(cfg, {"context": "cot", "goal": "few_shot_cot"})
    assert [(new.stages[s].model, new.stages[s].strategy) for s in STAGES] == [
        ("mock", "cot"), ("mock", "zero_shot"), ("mock", "cot"), ("mock", "few_shot_cot")]
    assert (new.run, new.thresholds, new.models) == (cfg.run, cfg.thresholds, cfg.models)
    assert [cfg.stages[s].strategy for s in STAGES] == ["cot", "zero_shot", "zero_shot", "few_shot"]     # not changed
    assert with_stage_strategies(cfg, {}) == cfg                   # nothing named, nothing changed
    both = with_stage_strategies(cfg, {"people": "cot", "constraints": "zero_shot"})
    assert [(both.stages[s].model, both.stages[s].strategy) for s in METHOD_STAGES] == [("mock", "cot"), ("mock", "zero_shot")]
    assert [both.stages[s].strategy for s in STAGES] == ["cot", "zero_shot", "zero_shot", "few_shot"]
    with pytest.raises(ConfigError) as refusal:                    # an unknown stage: named, with the six there are
        with_stage_strategies(cfg, {"goal": "cot", "judge": "cot"})
    assert "'judge'" in str(refusal.value) and "plan, locate, context, goal, people, constraints" in str(refusal.value)
    with pytest.raises(ConfigError) as refusal:                    # an unknown strategy: named, with the four there are
        with_stage_strategies(cfg, {"goal": "tree_of_thought"})
    assert "'tree_of_thought'" in str(refusal.value) and "zero_shot, cot, few_shot, few_shot_cot" in str(refusal.value)


@pytest.mark.parametrize("stage", ["plan", "locate", "context", "people", "constraints"])
@pytest.mark.parametrize("strategy", ["few_shot", "few_shot_cot"])
def test_only_the_goal_stage_takes_a_strategy_with_examples(stage, strategy):
    cfg = load_config()
    with pytest.raises(ConfigError) as refusal:                    # one line that names the stage and the strategy
        with_stage_strategies(cfg, {stage: strategy})
    assert f"'{stage}'" in str(refusal.value) and f"'{strategy}'" in str(refusal.value) and "\n" not in str(refusal.value)
    assert with_stage_strategies(cfg, {"goal": strategy}).stages["goal"].strategy == strategy


def test_a_small_file_gets_the_code_defaults(tmp_path):
    raw = {"models": {"m": {"provider": "mock", "model": "x"}},
           "stages": {s: {"model": "m", "strategy": "zero_shot"} for s in STAGES}}
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(str(path))                                   # a str path is accepted
    assert (cfg.run, cfg.thresholds) == (RunConfig(), Thresholds())
    assert (cfg.dev_situations, cfg.held_out_situations) == SITUATIONS
    assert {s: (stage.model, stage.strategy) for s, stage in cfg.stages.items()} == {     # the four it names, and the two
        **{s: ("m", "zero_shot") for s in STAGES}, "people": ("m", "zero_shot"), "constraints": ("m", "cot")}
    del raw["stages"]["locate"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="locate"):               # every stage is required
        load_config(path)
    with pytest.raises(ConfigError, match="absent.yaml"):          # so is the file
        load_config(tmp_path / "absent.yaml")


GATEWAY = {"provider": "openai_compatible", "model": "m", "base_url": "https://gateway.example/v1", "api_key_env": "A_KEY"}
SOCLAAS = {                                                        # entry: (model id, reasoning effort)
    "soclaas_qwen38": ("qwen3.8:27b", "none"), "soclaas_qwen38_low": ("qwen3.8:27b", "low"),
    "soclaas_qwen38_medium": ("qwen3.8:27b", "medium"), "soclaas_qwen36_35": ("qwen3.6:35b", "none"),
    "soclaas_qwen3vl": ("qwen3-vl:32b", "none"), "soclaas_ornith15": ("ornith1.5:35b", "none"),
    "soclaas_ornith15_low": ("ornith1.5:35b", "low"),
    "soclaas_glm53": ("x-test-1", "none"), "soclaas_glm53_low": ("x-test-1", "low"),
    "soclaas_glm53_high": ("x-test-1", "high"),
    "soclaas_glm53_max": ("x-test-1", "max"), "soclaas_dsv41": ("x-test-2", "none"),
    "soclaas_dsv41_low": ("x-test-2", "low"), "soclaas_qwen38_next": ("x-test-3", "none"),
    "soclaas_qwen38_next_low": ("x-test-3", "low"), "soclaas_gemma4": ("gemma4:26b", "none"),
    "soclaas_qwen35_9": ("qwen3.5:9b", "none")}
PRICES = {                                                         # model id: dollars per million tokens (input, output)
    "qwen3.8:27b": (0.408922, 2.320054), "qwen3-vl:32b": (0.380025, 2.419278), "qwen3.6:35b": (0.450056, 1.5155),
    "x-test-1": (0.37, 1.25), "x-test-2": (0.3, 1.2), "ornith1.5:35b": (0.2, 1.7), "gemma4:26b": (0.102294, 0.379882),
    "qwen3.5:9b": (0.114438, 0.200438), "x-test-3": (None, None)}     # x-test-3: not read from the model list yet


@pytest.mark.parametrize("entry, named", [
    ({k: v for k, v in GATEWAY.items() if k != "base_url"}, "base_url"),
    ({k: v for k, v in GATEWAY.items() if k != "api_key_env"}, "api_key_env"),
    ({**GATEWAY, "base_url": ""}, "base_url"),                     # an empty value names no address
    ({"provider": "gemini", "model": "m", "base_url": "https://gateway.example/v1"}, "base_url"),
    ({"provider": "openai", "model": "m", "api_key_env": "A_KEY"}, "api_key_env"),
    ({"provider": "anthropic", "model": "m", "base_url": "https://gateway.example/v1"}, "base_url"),
    ({"provider": "mock", "model": "m", "api_key_env": "A_KEY"}, "api_key_env"),
    ({**GATEWAY, "structured_output": "xml"}, "structured_output"),
])
def test_only_an_openai_compatible_entry_names_an_address_and_a_key_variable(entry, named):
    with pytest.raises(ConfigError) as refusal:                    # the key of one provider is never sent to another
        load_config(overrides={"models": {"new_entry": entry}})
    assert "new_entry" in str(refusal.value) and named in str(refusal.value)


def test_an_openai_compatible_entry_loads_with_its_address_and_key_variable():
    entry = load_config(overrides={"models": {"new_entry": GATEWAY}}).models["new_entry"]
    assert (entry.provider, entry.base_url, entry.api_key_env) == ("openai_compatible", GATEWAY["base_url"], "A_KEY")
    assert (entry.structured_output, entry.reasoning_effort, entry.point_format) == ("json_schema", "low", "xy_1000")
    loose = load_config(overrides={"models": {"new_entry": {**GATEWAY, "structured_output": "none",
                                                            "reasoning_effort": "xhigh"}}}).models["new_entry"]
    assert (loose.structured_output, loose.reasoning_effort) == ("none", "xhigh")     # a level is any name a model takes


def test_soclaas_entries_share_one_gateway_and_each_has_its_model_and_level():
    cfg = load_config()
    entries = {name: entry for name, entry in cfg.models.items() if entry.provider == "openai_compatible"}
    assert {name: (entry.model, entry.reasoning_effort) for name, entry in entries.items()} == SOCLAAS
    shared = {(e.base_url, e.api_key_env, e.point_format, e.structured_output) for e in entries.values()}
    assert shared == {("https://soclaas-api.comp.nus.edu.sg/v1", "SOCLAAS_API_KEY", "xy_1000", "json_schema")}
    assert {(e.base_url, e.api_key_env) for name, e in cfg.models.items() if name not in SOCLAAS} == {(None, None)}
    assert not {"qwen3.6:27b", "ornith1.0:35b"} & {e.model for e in cfg.models.values()}      # second names of two others


def test_soclaas_entries_carry_the_request_limit_and_the_prices_of_their_model():
    cfg = load_config()
    assert {model for model, _ in SOCLAAS.values()} == set(PRICES)
    for name, (model, _) in SOCLAAS.items():
        entry = cfg.models[name]
        assert (entry.max_requests_per_minute, entry.price_input_per_mtok, entry.price_output_per_mtok) == (
            80, *PRICES[model]), name
    others = [e for name, e in cfg.models.items() if name not in SOCLAAS]
    # no other provider is paced, but for the three older Claude models, at 25 requests a minute
    assert {name: e.max_requests_per_minute for name, e in cfg.models.items() if name not in SOCLAAS
            and e.max_requests_per_minute is not None} == dict.fromkeys(
        ("claude_haiku45", "claude_sonnet46_low", "claude_sonnet5_low"), 25)
    priced = {e.provider for e in others if (e.price_input_per_mtok, e.price_output_per_mtok) != (None, None)}
    assert priced == {"anthropic"}                                 # and one other is priced: tests/test_anthropic.py has its prices


def test_an_entry_may_name_a_request_limit_and_prices():
    bare = ModelEntry(provider="mock", model="m")
    assert (bare.max_requests_per_minute, bare.price_input_per_mtok, bare.price_output_per_mtok) == (None, None, None)
    given = {"provider": "mock", "model": "m", "max_requests_per_minute": 1, "price_input_per_mtok": 0,
             "price_output_per_mtok": 2.5}                         # the lowest limit there is, and a free input
    entry = load_config(overrides={"models": {"new_entry": given}}).models["new_entry"]
    assert (entry.max_requests_per_minute, entry.price_input_per_mtok, entry.price_output_per_mtok) == (1, 0.0, 2.5)


@pytest.mark.parametrize("field, value", [
    ("max_requests_per_minute", 0), ("max_requests_per_minute", -5), ("max_requests_per_minute", 1.5),
    ("price_input_per_mtok", -0.01), ("price_output_per_mtok", -1)])
def test_a_limit_below_one_and_a_negative_price_are_refused(field, value):
    with pytest.raises(ConfigError) as refusal:
        load_config(overrides={"models": {"new_entry": {"provider": "mock", "model": "m", field: value}}})
    assert "new_entry" in str(refusal.value) and field in str(refusal.value)


@pytest.mark.parametrize("given, absent, price", [
    ("price_input_per_mtok", "price_output_per_mtok", 0.5), ("price_output_per_mtok", "price_input_per_mtok", 2.5),
    ("price_input_per_mtok", "price_output_per_mtok", 0)])         # a price of nothing is a price: the other is still due
def test_one_price_without_the_other_is_refused(given, absent, price):
    entry = {"provider": "mock", "model": "m", given: price}       # before, a run on it left every cost empty and said nothing
    for lone in (entry, {**entry, absent: None}):                  # the other left out, or written as null
        with pytest.raises(ConfigError) as refusal:
            load_config(overrides={"models": {"new_entry": lone}})
        assert "new_entry" in str(refusal.value) and f"sets {given} without {absent}" in str(refusal.value)
    both = load_config(overrides={"models": {"new_entry": {**entry, absent: 1.0}}}).models["new_entry"]
    assert (getattr(both, given), getattr(both, absent)) == (price, 1.0)       # with the other, it loads
