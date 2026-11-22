"""Configuration models and loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelEntry(_Strict):
    provider: Literal["gemini", "openai", "openai_compatible", "anthropic", "mock"]
    model: str
    base_url: str | None = None            # the address of an openai_compatible provider: set there, and only there
    api_key_env: str | None = None         # the .env variable holding that provider's key: likewise
    structured_output: Literal["json_schema", "json_object", "none"] = "json_schema"     # read by openai_compatible
    max_requests_per_minute: int | None = Field(default=None, ge=1)    # the provider's limit; None: no pacing
    price_input_per_mtok: float | None = Field(default=None, ge=0)     # US dollars per million tokens; None: not known
    price_output_per_mtok: float | None = Field(default=None, ge=0)    # thinking tokens are billed as output
    reasoning_effort: str = "low"          # any name: the levels a model takes differ from model to model
    point_format: Literal["xy_1000", "xy_pixel"] = "xy_1000"
    temperature: float | None = None       # not for an anthropic entry: see _NOT_FOR_ANTHROPIC
    image_detail: str | None = None        # OpenAI detail / Gemini media resolution, passed through; likewise
    max_output_tokens: int | None = None
    timeout_s: float = 180.0


# What an anthropic entry may not set, because its back end has nowhere to send it: the current Claude models
# refuse a sampling parameter, and an image goes to them without a detail setting. Left unsaid, the setting would
# be in the entry, in the run's config.yaml and in the cache key, and in no request.
_NOT_FOR_ANTHROPIC = ("temperature", "image_detail")


Strategy = Literal["zero_shot", "cot", "few_shot", "few_shot_cot"]
StageName = Literal["plan", "locate", "context", "goal", "people", "constraints"]
_NAMED_STAGES = ("plan", "locate", "context", "goal")     # the stages every config has to name
# The two stages of the constraint method, which a config may leave out: each then uses the model entry of the
# goal stage, with the strategy given here.
_FOLLOWING_STAGES = {"people": "zero_shot", "constraints": "cot"}
_WITHOUT_EXAMPLES = ("zero_shot", "cot")           # the strategies every stage accepts; the goal stage takes all four


class StageConfig(_Strict):
    model: str                              # key into Config.models
    strategy: Strategy


class Thresholds(_Strict):
    standoff_m: float = 1.0
    standard_goal_max_slide_m: float = 0.5
    standard_goal_step_m: float = 0.05
    standard_goal_clearance_m: float = 0.40
    surface_radius_m: float = 0.30
    surface_min_points: int = 40
    surface_min_spread_m: float = 0.05
    surface_max_tilt: float = 0.5
    surface_max_rms_m: float = 0.06
    surface_max_rms_ratio: float = 0.5
    floor_tolerance_m: float = 0.15
    lift_window: int = 5
    heading_degenerate_m: float = 0.05
    robot_radius_m: float = 0.30
    person_radius_m: float = 0.25
    grid_resolution_m: float = 0.05
    match_tolerance_m: float = 0.3
    default_match_radius_m: float = 0.75
    personal_gap_m: float = 0.3
    anchor_depth_tolerance_m: float = 0.3
    relevant_radius_m: float = 2.0
    visibility_depth_tolerance_m: float = 0.3
    visibility_min_samples: int = 3
    floor_depth_max_rel_error: float = 0.02
    floor_min_pixels: int = 2000
    floor_person_clearance_m: float = 0.4
    camera_height_min_m: float = 0.2
    camera_height_max_m: float = 3.0
    yaw_tolerance_rad: float = 0.001
    free_space_agreement_min: float = 0.90
    # The ten of the constraint method (its design, section 7), each with its reason.
    view_half_angle_deg: float = 35.0      # a person's view, who faces whom: as far as the eyes turn without the head
    visible_half_angle_deg: float = 60.0   # seen without turning round: half the field both eyes share, 120 degrees
    floor_range_m: float = 12.0            # the solver's floor: people are in frame from head to foot out to about 7 m
    seen_depth_ratio: float = 0.03         # seen floor: above depth's change across one pixel of floor, 0.7 % at 12 m
    seen_depth_margin_m: float = 0.02      # with the ratio 0.38 m at floor_range_m: inside standard_goal_clearance_m
    max_keep_m: float = 3.0                # three times standoff_m: farther than that is not being at the target
    max_people: int = 40                   # a guard: the proposal's largest episode, with room for a false find
    max_constraints: int = 12              # a guard: twice the number of kinds
    person_mark_height_m: float = 1.1      # chest height, inside the heights the visibility rule samples
    candidate_count: int = 4               # the runners-up kept beside the place
    # The joining gate of the place in turn (results/still-scenes/crowds/joining-gate/), off where None: a person the
    # answer did not list joins the waiting people only if they face within join_facing_deg of the line's front, and
    # stand no farther behind the last than join_spacing_ratio times the gap from the last to the one before.
    join_facing_deg: float | None = None
    join_spacing_ratio: float | None = None


class RunConfig(_Strict):
    grounding: Literal["vlm", "oracle"] = "vlm"
    context_check: Literal["vlm", "oracle", "always_yes", "always_no"] = "vlm"
    goal: Literal["direct_marking", "oracle", "constraints"] = "direct_marking"
    people: Literal["vlm", "oracle"] = "vlm"                   # under goal: constraints, who finds the people
    constraints: Literal["vlm", "none"] = "vlm"                # and who states the constraints; none: the solver alone
    grounding_source: Literal["view", "tour"] = "view"
    heading_mode: Literal["face_target", "vlm_look_at"] = "face_target"
    trials: int = 1
    leave_out_situation: str | None = None
    final_test: bool = False
    examples_dir: str | None = None
    k: int | None = None
    seed: int = 0
    max_tour_frames: int = Field(default=16, ge=1)             # a plan on a tour needs a frame to show
    max_image_side: int | None = Field(default=None, ge=1)     # None: no resizing; 0 would send 1 x 1 images
    prompts_dir: str = "prompts"                               # the folder of the stage prompts
    use_cache: bool = True
    cache_dir: str = ".cache/vlm"


class Config(_Strict):
    models: dict[str, ModelEntry]
    stages: dict[StageName, StageConfig]
    thresholds: Thresholds = Thresholds()
    run: RunConfig = RunConfig()
    dev_situations: list[str] = ["no_one", "waiting_line", "conversation_group", "audience"]
    held_out_situations: list[str] = ["group_photo", "lift_wait", "machine_use"]

    @model_validator(mode="after")
    def _check_rules(self) -> "Config":
        for name, entry in self.models.items():
            if entry.provider == "openai_compatible":
                unset = [field for field in ("base_url", "api_key_env") if not getattr(entry, field)]
                if unset:
                    raise ValueError(f"model entry '{name}' has provider openai_compatible and has to set "
                                     f"{' and '.join(unset)}")
            else:                                  # the address and the key of another provider are fixed in code
                given = [field for field in ("base_url", "api_key_env") if getattr(entry, field) is not None]
                if given:
                    raise ValueError(f"model entry '{name}' sets {' and '.join(given)}, which only an "
                                     f"openai_compatible entry may set, not one of provider {entry.provider}")
            if entry.provider == "anthropic":
                unsent = [field for field in _NOT_FOR_ANTHROPIC if getattr(entry, field) is not None]
                if unsent:
                    raise ValueError(f"model entry '{name}' sets {' and '.join(unsent)}, which an entry of provider "
                                     f"anthropic may not set: its back end sends no such setting, so remove it")
            prices = {field: getattr(entry, field) for field in ("price_input_per_mtok", "price_output_per_mtok")}
            priced = [field for field, price in prices.items() if price is not None]
            if len(priced) == 1:                   # a cost takes both: with one, every cost was left empty unsaid
                (absent,) = set(prices) - set(priced)
                raise ValueError(f"model entry '{name}' sets {priced[0]} without {absent}: a cost needs both "
                                 f"prices, so give both or neither")
        for name in _NAMED_STAGES:
            if name not in self.stages:
                raise ValueError(f"missing stage '{name}'")
        # A stage the config may leave out follows the goal stage, so the table always holds all six.
        entry = self.stages["goal"].model
        following = {name: StageConfig(model=entry, strategy=strategy)
                     for name, strategy in _FOLLOWING_STAGES.items() if name not in self.stages}
        self.stages = {**self.stages, **following}
        for name, stage in self.stages.items():
            if stage.model not in self.models:
                raise ValueError(f"stage '{name}' uses model '{stage.model}', which is not a key of models")
            if name != "goal" and stage.strategy not in _WITHOUT_EXAMPLES:
                raise ValueError(f"stage '{name}' accepts only zero_shot or cot, not '{stage.strategy}'")
        shared = sorted(set(self.dev_situations) & set(self.held_out_situations))
        if shared:
            raise ValueError(f"dev_situations and held_out_situations share {shared}")
        leave_out = self.run.leave_out_situation
        if leave_out is not None and leave_out not in self.dev_situations:
            raise ValueError(f"leave_out_situation '{leave_out}' is not in dev_situations")
        if self.run.grounding_source == "tour" and self.run.grounding == "oracle":
            raise ValueError("grounding_source 'tour' cannot be combined with grounding 'oracle'")
        return self


class ConfigError(ValueError):
    """A configuration file or override that cannot be used."""


def _merge(base: Any, over: Any) -> Any:
    if isinstance(base, dict) and isinstance(over, dict):
        merged = {k: v for k, v in base.items()}
        for key, value in over.items():
            merged[key] = _merge(base[key], value) if key in base else value
        return merged
    return over


def load_config(path: Path | str = DEFAULT_CONFIG, overrides: dict | None = None) -> Config:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"{path}: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: the top level must be a mapping")
    if overrides:
        raw = _merge(raw, overrides)
    try:
        return Config.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(f"{path}: {error}") from error


def _validated(data: dict) -> Config:
    try:
        return Config.model_validate(data)
    except ValidationError as error:
        raise ConfigError(str(error)) from error


def with_model(cfg: Config, entry_name: str) -> Config:
    if entry_name not in cfg.models:
        raise ConfigError(f"model entry '{entry_name}' is not a key of models")
    data = cfg.model_dump()
    for stage in data["stages"].values():
        stage["model"] = entry_name
    return _validated(data)


def with_stage_models(cfg: Config, assignments: Mapping[str, str]) -> Config:
    """A copy of `cfg` in which each stage named in `assignments` uses the entry named for it; the rest is kept.

    An unknown stage or entry is a ConfigError that names it and lists the valid ones."""
    stages = get_args(StageName)
    for stage, entry_name in assignments.items():
        if stage not in stages:
            raise ConfigError(f"'{stage}' is not a stage; the stages are {', '.join(stages)}")
        if entry_name not in cfg.models:
            raise ConfigError(f"model entry '{entry_name}' is not a key of models; the entries are "
                              f"{', '.join(cfg.models)}")
    data = cfg.model_dump()
    for stage, entry_name in assignments.items():
        data["stages"][stage]["model"] = entry_name
    return _validated(data)


def with_stage_strategies(cfg: Config, assignments: Mapping[str, str]) -> Config:
    """A copy of `cfg` in which each stage named in `assignments` uses the strategy named for it; the rest is kept.

    An unknown stage or strategy is a ConfigError that names it and lists the valid ones, and so is a strategy
    with examples for a stage other than goal."""
    stages, strategies = get_args(StageName), get_args(Strategy)
    for stage, strategy in assignments.items():
        if stage not in stages:
            raise ConfigError(f"'{stage}' is not a stage; the stages are {', '.join(stages)}")
        if strategy not in strategies:
            raise ConfigError(f"'{strategy}' is not a strategy; the strategies are {', '.join(strategies)}")
        if stage != "goal" and strategy not in _WITHOUT_EXAMPLES:
            raise ConfigError(f"stage '{stage}' accepts only {' or '.join(_WITHOUT_EXAMPLES)}, not '{strategy}': "
                              "only the goal stage shows examples")
    data = cfg.model_dump()
    for stage, strategy in assignments.items():
        data["stages"][stage]["strategy"] = strategy
    return _validated(data)


def _assignments(option: str, values: Sequence[str], form: str, example: str) -> dict[str, str]:
    """The STAGE=NAME values of one repeatable option as a mapping; a later value for a stage replaces an earlier."""
    assignments: dict[str, str] = {}
    for value in values:
        stage, equals, name = (piece.strip() for piece in value.partition("="))
        if not (stage and equals and name):
            raise ConfigError(f"{option} {value!r} is not of the form {form}, as in {example}")
        assignments[stage] = name
    return assignments


def with_model_options(cfg: Config, model: str | None, stage_models: Sequence[str],
                       stage_strategies: Sequence[str] = ()) -> Config:
    """`cfg` as the options --model, --stage-model and --strategy of a command change it, whichever command it is.

    `model`, when given, puts every stage on that entry; each STAGE=ENTRY of `stage_models` then puts one stage
    on an entry of its own, and each STAGE=NAME of `stage_strategies` gives one stage its strategy. A value of
    another form, an unknown stage, entry or strategy and a strategy the stage does not accept are ConfigErrors."""
    if model is not None:
        cfg = with_model(cfg, model)
    cfg = with_stage_models(cfg, _assignments("--stage-model", stage_models, "STAGE=ENTRY", "context=mock"))
    return with_stage_strategies(cfg, _assignments("--strategy", stage_strategies, "STAGE=NAME", "goal=cot"))
