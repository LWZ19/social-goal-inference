"""Stage outputs and results shared by inference code. Holds no ground truth."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from sgi.config import StageName


class Point(BaseModel):
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)


class Subtask(BaseModel):
    type: Literal["navigate", "act"]
    description: str
    target: str | None = None
    target_point: Point | None = None
    tour_frame: int | None = None


class Plan(BaseModel):
    reasoning: str | None = None
    subtasks: list[Subtask]
    image_sha256: str | None = None


class Location(BaseModel):
    reasoning: str | None = None
    target_point: Point | None = None


class ContextCheck(BaseModel):
    reasoning: str | None = None
    relevant_people: str
    people_matter: bool


class GoalReasoning(BaseModel):
    people: str
    expectations: str
    placement: str


class GoalMark(BaseModel):
    reasoning: GoalReasoning | None = None
    goal_point: Point
    look_at_point: Point | None = None


class GroundedTarget(BaseModel):
    point: Point
    pixel: tuple[float, float]
    world: tuple[float, float, float]


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None


class Call(BaseModel):
    stage: StageName
    provider: str
    model: str
    model_version: str | None = None
    reasoning_effort: str
    point_format: str
    image_size: tuple[int, int] | None = None      # (width, height) as sent
    latency_s: float
    latency_total_s: float
    validation_retries: int
    transient_retries: int
    cached: bool
    usage: Usage
    error: str | None = None
    raw_text: str | None = None
    status: Literal["ok", "parse_failed", "vlm_error"] = "ok"   # copied from the VLMResponse by to_call


class Flags(BaseModel):
    marked_point_off_floor: bool = False
    standard_goal_blocked: bool = False
    standard_goal_frontal: bool = False             # the standard goal stands before the surface the target point lies on
    heading_fallback: bool = False
    heading_degenerate: bool = False
    # The six of the constraint method.
    no_usable_constraint: bool = False             # the solver ran on what it applies by itself
    constraint_unusable: bool = False              # a constraint of the answer is unusable or superseded
    constraints_cut: bool = False                  # the answer held more constraints than are read
    exclusion_broken: bool = False                 # every place breaks an exclusion: the one taken breaks the least
    people_cut: bool = False                       # more people were found than are kept
    far_from_target: bool = False                  # beyond max_keep_m from the target, and the robot neither waited its
    #                                                turn nor stood back with people who watch


Status = Literal["ok", "no_target", "lift_failed", "parse_failed", "vlm_error"]
Pose = tuple[float, float, float]                  # x, y, theta


class DetectedPerson(BaseModel):
    """A person in view, as the constraint method knows them."""
    number: int                                    # drawn on them in the image the model is shown, from 1
    xy: tuple[float, float]                        # the centre of the body on the floor, world frame
    yaw: float | None = None                       # where they face, world frame; None where it cannot be told
    pixel: tuple[float, float]                     # the body point in the image
    label: str | None = None                       # a few words that pick the person out; None from ground truth


ConstraintKind = Literal["keep_out_of_shared_space", "keep_out_of_view", "come_after", "within_distance", "visible_to",
                         "face"]


class DistanceRange(BaseModel):
    at_least: float                                # metres, both as the model wrote them: the solver judges them
    at_most: float


class Constraint(BaseModel):
    """One constraint as the model stated it. What its kind does not read is ignored."""
    because: str                                   # one sentence: what about the people makes this apply
    kind: ConstraintKind
    people: list[int] = []                         # numbers of detected people; may be empty
    range_m: DistanceRange | None = None
    target: bool = False


class ConstraintUse(BaseModel):
    """What the solver made of one constraint."""
    constraint: Constraint
    status: Literal["used", "unusable", "superseded", "cut"]
    reason: str | None = None                      # why it is not used; None for one that is
    broken_m: float = 0.0                          # its violation at the place, in metres; 0.0 where it holds


class Keys(BaseModel):
    """The values a place is ordered by, in metres, in the order they are compared in."""
    excluded_m: float                              # the largest violation of any one exclusion
    excluded_total_m: float                        # the sum of the exclusions' violations
    turn_m: float | None = None                    # the distance to the place in turn; None without a come_after
    face_m: float | None = None                    # how far behind the plane of the seen face; None where there is none,
    #                                                the target is a person, a place in turn is used or the answer names
    #                                                a person to face and to be visible to
    preference_m: float                            # the sum of the preferences' violations
    standard_m: float                              # the distance to the standard goal


class Candidate(BaseModel):
    pose_world: Pose
    keys: Keys


class SolverNote(BaseModel):
    """What the solver records of the place it chose."""
    keys: Keys
    own_space_broken_m: dict[int, float] = {}      # by person number: how far inside their own space the place is
    joined_waiting: list[int] = []                 # numbers of the people taken as waiting whom the answer did not list
    not_ahead_of: list[int] = []                   # numbers of the people who watch the target whom the place is kept
    #                                                level with or behind
    not_ahead_reason: str | None = None            # why nobody is in that list, where a view of the target is used
    target_distance_m: float                       # from the place to the target point


class GoalResult(BaseModel):
    status: Status
    source: str | None = None                      # "standard", "direct_marking", "oracle", "constraints"
    pose_robot: Pose | None = None
    pose_world: Pose | None = None
    goal_pixel: tuple[float, float] | None = None
    target: GroundedTarget | None = None
    flags: Flags = Flags()
    calls: list[Call] = []
    timings_s: dict[str, float] = {}               # keys: plan, locate, context, goal, geometry
    location: Location | None = None               # stage outputs, kept for results.jsonl
    context_check: ContextCheck | None = None
    goal_mark: GoalMark | None = None
    people: list[DetectedPerson] | None = None     # the four of the constraint method: None outside it
    constraints: list[ConstraintUse] | None = None
    candidates: list[Candidate] | None = None      # the runners-up
    solver: SolverNote | None = None


class PlanOutcome(BaseModel):
    status: Literal["ok", "parse_failed", "vlm_error"]
    plan: Plan | None = None
    call: Call | None = None
    time_s: float = 0.0
    tour_indices: list[int] | None = None          # original indices of the tour frames shown; set in tour mode whatever the status
