"""Run harness: runs the pipeline over a folder of scenes and scores it.

    python -m sgi.run --scenes DIR [--config PATH] [--model ENTRY] [--stage-model STAGE=ENTRY ...]
                      [--strategy STAGE=NAME ...] [--prompts DIR] --name NAME
                      [--final-test] [--no-cache] [--skip-scene-check]
    python -m sgi.run [--config PATH] --list-entries

--model puts every stage on one entry of the config's model table; each --stage-model then puts one stage on an
entry of its own, and each --strategy then gives one stage its prompting strategy. --prompts names the folder
the stage prompts are read from, in place of the repository's prompts/. --list-entries prints the entries
of the model table, one per line, and runs nothing.

Writes runs/<timestamp>_<name>/ with config.yaml, results.jsonl (one record per episode and trial),
summary.csv, viz/<scene>_<episode>.png and prompts/, a copy of the prompts of the run's stages. The last
line printed gives the success rate, the model calls made, how many of them the cache answered and, where the
entries used give prices, about what the answers cost. Ground truth stays on this side: the harness selects
the in-context examples, puts a ground-truth-derived value in place of a stage's output where the run mode
says so, and scores every goal. Stage code never sees ground truth. With run.grounding_source: tour the plan
is made on the scene's tour and every navigation subtask is then located in the observation.
With run.goal: constraints the goal is placed by the constraint method: a record then
also holds the people with their numbers, what the solver made of each constraint, the runners-up and the
solver's note, so that it can be solved again without a model call. Under run.people: oracle the people of
that method are those of ground truth, which this side gives it. A record's measures then also hold, against
ground truth, how many of the people in view were found and whether those the answer lists as waiting are the
ones who wait, and how many constraints were unusable and exclusions broken; they are
the last measure columns of summary.csv, empty in a row whose records did not measure them.

Exit codes: 0 for a complete run, and for --list-entries; 1 when a record ended in vlm_error, where the same
command run again repeats only the calls that got no answer; 2 when the run was refused or stopped by a
configuration, scene, missing-key or provider error. A provider whose allowance is used up stops the run with 2
as well, at once and with what is done kept: results.jsonl holds the records finished before it, summary.csv is
written over them, config.yaml says `incomplete: true` and gives the provider's message as `stopped_by`, and the
same command run again resumes from the cache.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import yaml

from sgi.check_scene import check_scene
from sgi.config import (DEFAULT_CONFIG, REPO_ROOT, Config, ConfigError, RunConfig, Thresholds, load_config,
                        with_model_options)
from sgi.evaluate import Score, decomposition_accuracy, match_target, reduce_plan, score, tour_frame_correct
from sgi.geometry import bearing, lift, make_poses, point_to_pixel, project
from sgi.gt.derive import EpisodeTruth, is_visible, oracle_point, prepare, reference_people_matter
from sgi.imaging import draw_target_marker, encode_png
from sgi.inputs import EpisodeInputs, ObservationData, SceneInputs
from sgi.pipeline import Pipeline, build_pipeline
from sgi.prompts import Example, prompt_files, stages_of
from sgi.scene import EpisodeGT, GroundTruth, Person, Region, SceneError, load_scene, read_scene_file
from sgi.stages.goal import GoalInferrer
from sgi.stages.people import FoundPerson, PeopleFound
from sgi.types import (Call, ContextCheck, DetectedPerson, Flags, GoalMark, GoalReasoning, GoalResult, GroundedTarget,
                       Location, Plan, PlanOutcome, Point, Status, Subtask)
from sgi.viz import episode_figure
from sgi.vlm.base import BackendError, QuotaExceededError
from sgi.vlm.env import MissingKeyError
from sgi.vlm.factory import entry_lines

log = logging.getLogger("sgi.run")                 # by name: run as a command, this module is __main__

_Read = Callable[[dict], Any]                      # a record's value for one column; None where it was not measured

_UNLABELLED = "unlabelled"
_LABEL_KEYS = ("request_type", "situation")        # groupings read from a record's labels
_FACTOR_KEYS = ("density", "group_size", "bystanders", "other_groups", "view_angle_deg")     # and from its factors
_GROUP_SIZE_BINS = ((0, "0"), (2, "1-2"), (5, "3-5"), (10, "6-10"))     # the largest size in each bin
_GROUP_SIZE_BEYOND = "11+"
_VIEW_ANGLE_SPLIT_DEG = 30.0                       # up to 30 degrees, and above
_VIEW_ANGLE_DECIMALS = 3                           # a view of exactly 30 degrees keeps its bin through float noise
_SCORE_COLUMNS = (("success_rate", "success"), ("target_correct_rate", "target_correct"),
                  ("reachable_rate", "reachable"), ("in_region_rate", "in_region"), ("heading_ok_rate", "heading_ok"),
                  ("personal_space_rate", "personal_space_intrusion"),
                  ("distance_to_region_m", "distance_to_region_m"), ("min_body_gap_m", "min_body_gap_m"))
_MEASURE_COLUMNS = (("context_missed_rate", "context_missed"), ("context_unnecessary_rate", "context_unnecessary"),
                    ("plan_exact_rate", "plan_exact"), ("navigate_recall", "navigate_recall"),
                    ("later_subtask_hit_rate", "later_subtask_hit_target"),
                    ("tour_frame_accuracy", "tour_frame_correct"), ("latency_s", "latency_s"))
# The measures of the constraint method: the name each has in a record and as a column.
_METHOD_MEASURES = ("people_found_share", "people_unmatched", "members_precision", "members_recall",
                    "constraints_unusable", "exclusions_broken")
_TURN_SITUATION = "waiting_line"                   # in its reference the group at the target waits its turn there
_TURN_KIND = "come_after"                          # the kind of constraint that lists those who wait
_EXCLUSION_KINDS = ("keep_out_of_shared_space", "keep_out_of_view", _TURN_KIND)     # the kinds that exclude places
_NOT_USED = ("unusable", "superseded")             # the statuses the flag constraint_unusable is set for
_TIMED = ("plan", "locate", "context", "goal", "geometry")       # the keys of timings_s
_FAILURE_STATUSES = ("no_target", "lift_failed", "parse_failed")
_NO_ANSWER = "vlm_error"                           # no model answer was obtained: kept out of every rate
_REASONED_EXAMPLES = "few_shot_cot"                # the goal strategy whose examples carry their rationale
_LOOK_AT_MODE = "vlm_look_at"                      # the heading mode whose answers carry a look_at_point
_LOOK_AHEAD_M = 1.0                                # how far along a yaw heading an example looks
_UNGROUNDED, _UNMATCHED = "ungrounded", "unmatched"     # what a navigate subtask's target matched when it is no target
_RECORDED_SETTINGS = ("grounding", "context_check", "goal", "people", "constraints", "grounding_source", "heading_mode")
_EXAMPLE_STRATEGIES = ("few_shot", "few_shot_cot")      # the goal strategies that show in-context examples
_GENERATE_HINT = "the placeholder scenes are written by: python -m sgi.synth.generate --out data/placeholder"
_TOKENS_PER_PRICE = 1e6                            # an entry's prices are in US dollars per million tokens
# What main adds to the provider's message when its allowance stopped a run, and what a second run then does.
_SPENT = "the provider's allowance is used up, so the run stopped and kept what it had finished"
_RESUMES = "the same command run again resumes from the cache"
_STARTS_OVER = "this run did not use the cache, so the same command run again starts over"


def _look_at_xy(region: Region) -> tuple[float, float]:
    """The ground point an example's answer looks at: the region's face point or, for a yaw heading, the
    point `_LOOK_AHEAD_M` from the representative point along that yaw."""
    heading = region.heading
    if heading.face_point is not None:
        return heading.face_point
    x, y = region.representative_point
    return x + _LOOK_AHEAD_M * math.cos(heading.yaw), y + _LOOK_AHEAD_M * math.sin(heading.yaw)


def _worked_example(observation: ObservationData, truth: GroundTruth, et: EpisodeTruth, request: str,
                    cfg: Config) -> Example | None:
    """The episode as a worked example for the goal stage, or None when it is not eligible.

    The example answers for the episode's first target: the image is marked at its projected anchor and the
    goal is the projected representative point of its first acceptable region."""
    ep, run = et.gt, cfg.run
    reasoned, looks = cfg.stages["goal"].strategy == _REASONED_EXAMPLES, run.heading_mode == _LOOK_AT_MODE
    if ep.situation not in cfg.dev_situations or ep.situation == run.leave_out_situation:
        return None
    if et.problems:                                # a region or a point that fails the oracle path would be shown
        return None                                # as a wrong worked answer; an episode without a target is one too
    if run.context_check != "always_yes" and not reference_people_matter(et, None):
        return None
    if reasoned and ep.rationale is None:
        return None
    target = truth.targets[ep.target_ids[0]]
    region = next((r for r in ep.acceptable_regions if r.target_id in (None, target.id)), None)
    if region is None or region.representative_point is None:
        return None
    goal = oracle_point(observation, (*region.representative_point, 0.0))
    look_at = oracle_point(observation, (*_look_at_xy(region), 0.0)) if looks else None
    if oracle_point(observation, target.anchor) is None or goal is None or (looks and look_at is None):
        return None                                # every point of the example has to be in the image
    u, v, _ = project(observation.camera, target.anchor)
    return Example(image=encode_png(draw_target_marker(observation.rgb, u, v), run.max_image_side),
                   request=request, target_text=target.label, goal_point=goal,
                   reasoning=GoalReasoning(**ep.rationale.model_dump()) if reasoned else None, look_at_point=look_at)


def _select_examples(example_scenes: Sequence[Path], eval_scene_ids: set[str], needed: set[str],
                     cfg: Config) -> tuple[list[tuple[str, Example]], list[str]]:
    """What `select_examples` selects, and the situations that need an example and have no eligible one: those
    of `needed` that are development situations and not left out, in the order of the config. They are logged
    in one warning here, and the selection is made without them."""
    run = cfg.run
    candidates: dict[str, list[tuple[str, Example]]] = {}
    for folder in sorted(example_scenes):
        inputs, truth = load_scene(folder)
        if inputs.scene_id in eval_scene_ids:
            continue
        ets = prepare(inputs, truth, cfg)
        for episode in inputs.episodes:
            et = ets[episode.episode_id]
            example = _worked_example(inputs.observation, truth, et, episode.request, cfg)
            if example is not None:
                example_id = f"{inputs.scene_id}/{episode.episode_id}"
                candidates.setdefault(et.gt.situation, []).append((example_id, example))

    allowed = [situation for situation in cfg.dev_situations if situation != run.leave_out_situation]
    wanted = [situation for situation in allowed if situation in needed]
    order = [situation for situation in (*wanted, *(s for s in allowed if s not in needed)) if situation in candidates]
    rng = random.Random(run.seed)
    if run.k is None:
        queues, limit = [[rng.choice(candidates[situation])] for situation in order], len(order)
    else:
        for situation in order:
            rng.shuffle(candidates[situation])
        queues, limit = [candidates[situation] for situation in order], run.k

    selected: list[tuple[str, Example]] = []
    covered: set[str] = set()
    for one_each in itertools.zip_longest(*queues):          # a round: the next candidate of every situation
        for situation, candidate in zip(order, one_each):
            if candidate is not None and len(selected) < limit:
                selected.append(candidate)
                covered.add(situation)
    unreached = [situation for situation in wanted if situation in candidates and situation not in covered]
    if unreached:                                  # they have an eligible example: a larger run.k shows it
        raise ConfigError(f"no in-context example selected for {', '.join(unreached)}: run.k is {run.k}, too small "
                          "to reach every situation that needs an example")
    missing = [situation for situation in wanted if situation not in candidates]
    if missing:
        log.warning("no in-context example for %s: no scene of run.examples_dir that is not evaluated holds an "
                    "eligible episode; the goal stage is asked without an example of %s",
                    ", ".join(missing), "it" if len(missing) == 1 else "them")
    return selected, missing


def select_examples(example_scenes: Sequence[Path], eval_scene_ids: set[str], needed: set[str],
                    cfg: Config) -> list[tuple[str, Example]]:
    """The in-context examples of a run, as (example id, Example), chosen once so every episode sees the same.

    An example id is "<scene_id>/<episode_id>". No example comes from a scene in `eval_scene_ids` or from a
    held-out or left-out situation. Without `run.k` there is one example per situation that has an eligible
    episode; with it, `k` examples are taken round-robin over the situations, those in `needed` first.

    `needed` holds the situations of the evaluated episodes that can reach the goal stage. One of them that is
    a development situation, is not left out and has no eligible example is no reason to stop: the examples
    there are come back, none at all when no episode is eligible, and one warning names the situations left
    without. A ConfigError remains for a `run.k` too small to reach a needed situation that has one.
    """
    return _select_examples(example_scenes, eval_scene_ids, needed, cfg)[0]


def _lift_point(observation: ObservationData, point: Point,
                th: Thresholds) -> tuple[tuple[float, float], np.ndarray | None]:
    """The pixel a Point marks in the observation and the world point depth gives there, None where it gives
    none: the path a model's point takes."""
    camera = observation.camera
    pixel = point_to_pixel(point.x, point.y, camera.width, camera.height)
    return pixel, lift(observation.depth, camera, *pixel, th.lift_window)


class OracleGoalInferrer:
    """The goal a perfect Stage 3 would mark: the region's representative point, taken through the
    path a model's point takes, with the heading the region allows."""

    def __init__(self, region: Region, th: Thresholds):
        self.region = region
        self.th = th

    def infer(self, observation: ObservationData, request: str, subtask: Subtask, target: GroundedTarget,
              context_check: ContextCheck | None) -> GoalResult:
        started = time.perf_counter()
        xy, heading = self.region.representative_point, self.region.heading
        point = None if xy is None else oracle_point(observation, (xy[0], xy[1], 0.0))
        pixel, world = (None, None) if point is None else _lift_point(observation, point, self.th)
        if world is None:
            return GoalResult(status="lift_failed", source="oracle", target=target)
        goal = (float(world[0]), float(world[1]))
        theta = heading.yaw if heading.yaw is not None else bearing(goal, heading.face_point)
        pose_world, pose_robot = make_poses(observation.robot, goal, theta)
        flags = Flags(marked_point_off_floor=bool(abs(world[2]) > self.th.floor_tolerance_m))
        return GoalResult(status="ok", source="oracle", pose_world=pose_world, pose_robot=pose_robot,
                          goal_pixel=pixel, target=target, goal_mark=GoalMark(goal_point=point), flags=flags,
                          timings_s={"goal": 0.0, "geometry": time.perf_counter() - started})


class OraclePeople:
    """The `oracle` people source of the constraint method: the people of ground truth
    that pass the visibility rule in the order of ground truth, with their true
    position and heading. A person's `pixel` is the projection of the point `person_mark_height_m` above their
    position, and there is no label. No model is asked.

    It holds the visibility rule and the projection, and nothing else of ground truth. A person whose marked
    point does not project, because it is not in front of the camera, has no pixel and is left out."""

    def __init__(self, truth: GroundTruth, th: Thresholds):
        self.truth = truth
        self.th = th

    def __call__(self, observation: ObservationData) -> PeopleFound:
        started = time.perf_counter()
        found = []
        for person in self.truth.people:
            if not is_visible(person, observation, self.th):
                continue
            marked = project(observation.camera, (person.x, person.y, self.th.person_mark_height_m))
            if marked is not None:
                found.append(FoundPerson(xy=(person.x, person.y), yaw=person.yaw, pixel=marked[:2], label=None))
        return PeopleFound(found=found, calls=[], call_time_s=0.0, geometry_time_s=time.perf_counter() - started)


class _NoGoal:
    """The oracle where it has no region to mark: no goal is computed. The result is ok and carries no pose,
    because Status has no value of its own for that; the scorer fails a result without a pose."""

    def infer(self, observation: ObservationData, request: str, subtask: Subtask, target: GroundedTarget,
              context_check: ContextCheck | None) -> GoalResult:
        return GoalResult(status="ok", source="oracle")


def _matched(subtask: Subtask, observation: ObservationData, truth: GroundTruth, ep: EpisodeGT,
             th: Thresholds) -> str | None:
    """What a subtask's target is in the scene: None for an act step, "ungrounded" for a navigate step
    without a point, else the id of the episode's target its lifted point falls on, or "unmatched"."""
    if subtask.type != "navigate":
        return None
    if subtask.target_point is None:
        return _UNGROUNDED
    world = _lift_point(observation, subtask.target_point, th)[1]
    target_id = None if world is None else match_target(world, truth, ep, th)
    return _UNMATCHED if target_id is None else target_id


def _given_context_check(mode: str, et: EpisodeTruth, chosen: str | None) -> ContextCheck | None:
    """The context check the harness passes in place of the model's; None where the model is asked."""
    if mode == "vlm":
        return None
    people_matter = reference_people_matter(et, chosen) if mode == "oracle" else mode == "always_yes"
    return ContextCheck(relevant_people="", people_matter=people_matter)


def _given_inferrer(run: RunConfig, pipeline: Pipeline, truth: GroundTruth, ep: EpisodeGT, chosen: str | None,
                    th: Thresholds, trial: int) -> GoalInferrer | None:
    """The inferrer the harness passes in place of the pipeline's own; None where the pipeline builds
    the one `run.goal` names, with its own trial.

    Under `goal: constraints` with `people: oracle` it is the pipeline's constraint inferrer on the people of
    ground truth, for this trial: the trial is in the cache key of the constraints call, and without it every
    trial would be a cache hit of the first. Under `goal: oracle` it marks the first acceptable region of the
    chosen target, and computes no goal without one."""
    if run.goal == "constraints":
        return pipeline.constraint_inferrer(OraclePeople(truth, th), trial=trial) if run.people == "oracle" else None
    if run.goal != "oracle":
        return None
    region = None if chosen is None else next(
        (region for region in ep.acceptable_regions if region.target_id in (None, chosen)), None)
    return _NoGoal() if region is None else OracleGoalInferrer(region, th)


def _require_tour(inputs: SceneInputs, cfg: Config) -> None:
    """Raise a SceneError naming the scene when the run grounds on the tour and the scene has none."""
    if cfg.run.grounding_source == "tour" and not inputs.tour:
        raise SceneError(f"{inputs.scene_id}: tour: run.grounding_source is tour and the scene has no tour")


@dataclass(frozen=True)
class _TourGrounding:
    """Tour grounding of one episode: the plan made on the tour and what locate found for it."""
    outcome: PlanOutcome                           # of the plan call; it lists the frames shown whatever its status
    plan: Plan | None                              # the plan with every point locate found; None for a failed plan
    locations: dict[int, Location]                 # by subtask index: the valid answers of locate
    calls: tuple[Call, ...]                        # the locate calls, in plan order
    time_s: float                                  # the locate stage's time, summed over those calls
    stopped: Status | None                         # the status of a locate call that brought no valid answer


def _tour_grounding(pipeline: Pipeline, inputs: SceneInputs, episode: EpisodeInputs, trial: int) -> _TourGrounding:
    """Plan on the scene's tour, then locate the target of every navigation subtask in the observation, in plan order.

    A null answer of locate is a miss: its subtask keeps no point. A locate call that brought no valid answer
    ends the episode with the status of that call, and no later subtask is located: a bad answer is never
    counted as "not in view". The points found before it stay in the plan.
    """
    outcome = pipeline.plan(episode.request, episode.context, tour=inputs.tour, trial=trial)
    plan = outcome.plan
    locations: dict[int, Location] = {}
    calls: list[Call] = []
    time_s, stopped = 0.0, None
    if plan is not None:
        subtasks = list(plan.subtasks)
        for i, subtask in enumerate(subtasks):
            if subtask.type != "navigate":
                continue
            location, call, spent = pipeline.locate(inputs.observation, subtask, episode.request, episode.context,
                                                    trial=trial)
            calls.append(call)
            time_s += spent
            if location is None:                   # the call's status says why: a refusal without text leaves
                stopped = call.status              # raw_text as empty as an outage does, and is parse_failed
                break
            locations[i] = location
            subtasks[i] = subtask.model_copy(update={"target_point": location.target_point})
        plan = plan.model_copy(update={"subtasks": subtasks})
    return _TourGrounding(outcome, plan, locations, tuple(calls), time_s, stopped)


def _tour_record(grounding: _TourGrounding, inputs: SceneInputs, truth: GroundTruth, ep: EpisodeGT) -> dict:
    """The `tour` entry of a tour-mode record.

    `shown` holds the original indices of the frames Stage 1 saw, `chosen` the frame it gave the first
    navigation subtask, `reference_frames` the frames that show a target of the episode, and
    `coarse_goal_world` the chosen frame's camera pose on the ground: x, y and the yaw of the optical axis.
    """
    subtasks = [] if grounding.plan is None else grounding.plan.subtasks
    chosen = next((subtask.tour_frame for subtask in subtasks if subtask.type == "navigate"), None)
    reference = sorted({frame for target_id in ep.target_ids for frame in truth.targets[target_id].tour_frames or []})
    coarse_goal = None
    if chosen is not None:
        T = next(frame.T_world_camera for frame in inputs.tour if frame.index == chosen)
        coarse_goal = [float(T[0][3]), float(T[1][3]), math.atan2(T[1][2], T[0][2])]
    return {"shown": grounding.outcome.tour_indices, "chosen": chosen, "reference_frames": reference,
            "coarse_goal_world": coarse_goal}


def _cost_usd(calls: Sequence[Call], cfg: Config) -> float | None:
    """What the calls cost in US dollars, each at the prices of the entry its stage uses; None when one of those
    entries does not give both prices.

    A token count the provider did not report is taken as 0, and thinking tokens are billed as output. A call
    answered from the cache carries the counts of the request that was stored, so it counts at what that cost."""
    total = 0.0
    for call in calls:
        entry = cfg.models[cfg.stages[call.stage].model]
        if entry.price_input_per_mtok is None or entry.price_output_per_mtok is None:
            return None
        usage = call.usage
        output_tokens = (usage.output_tokens or 0) + (usage.reasoning_tokens or 0)
        total += ((usage.input_tokens or 0) * entry.price_input_per_mtok
                  + output_tokens * entry.price_output_per_mtok) / _TOKENS_PER_PRICE
    return total


def _matched_people(detected: Sequence[DetectedPerson], visible: Sequence[Person], th: Thresholds) -> dict[int, str]:
    """Who of ground truth each detected person is: by the number of a detected person, the id of the person of
    `visible` matched to them.

    The two can be one when their centres are within twice `person_radius_m`, a body's width. The pairs are
    taken nearest first, each detected person and each person of ground truth once; of two pairs as near, the
    lower number first and then the earlier of `visible`. A detected person left without a pair matches nobody.
    """
    width = 2 * th.person_radius_m
    pairs = sorted((math.dist(found.xy, (person.x, person.y)), found.number, index)
                   for found in detected for index, person in enumerate(visible))
    matched: dict[int, str] = {}
    taken: set[int] = set()
    for gap, number, index in pairs:
        if gap <= width and number not in matched and index not in taken:
            matched[number] = visible[index].id
            taken.add(index)
    return matched


def _exclusions_broken(result: GoalResult) -> int:
    """How many exclusions the place of a solved result breaks: the own space
    of each person it is inside, each exclusion of the answer recorded as broken, and coming after the people
    who wait once, however many constraints list them, because several are one."""
    broken = [use.constraint.kind for use in result.constraints
              if use.constraint.kind in _EXCLUSION_KINDS and use.broken_m > 0]
    stated = sum(kind != _TURN_KIND for kind in broken) + (_TURN_KIND in broken)
    return sum(inside > 0 for inside in result.solver.own_space_broken_m.values()) + stated


def _method_measures(result: GoalResult, visible: Sequence[Person], turn_group: str | None, run: RunConfig,
                     th: Thresholds) -> dict[str, float | None]:
    """The measures of the constraint method for one result, by name. Each is None where
    the result does not measure it, and all six are for a result that is not of the method.

    `visible` holds the people of ground truth that pass the visibility rule, and `turn_group` the id of the
    group whose members wait their turn at the target in the episode's reference, None where nobody does.

    - `people_found_share` and `people_unmatched`: of the people in view, the share that a detected person is
      matched to (`_matched_people`), and the number of detected people matched to nobody. Measured where a
      model found the people (`run.people: vlm`), whether or not the constraints call then brought an answer;
      with nobody in view there is no share.
    - `members_precision` and `members_recall`: of the numbers the answer lists under `come_after`, the share
      that are detected people matched to a member in view; and of the members in view, the share so listed.
      The numbers are those of every constraint of that kind, whatever became of it, and none that the solver
      took as waiting by itself. A number nobody has is listed, and wrong. Measured where a model stated the
      constraints (`run.constraints: vlm`) and the reference has a group that waits; there is no precision
      with nothing listed and no recall with no member in view.
    - `constraints_unusable`: how many constraints of the answer the solver could not use, unusable or
      superseded, as the flag `constraint_unusable` has it. `exclusions_broken`: how many exclusions the place
      breaks (`_exclusions_broken`). Both for every result the solver made, the solver alone included.
    """
    measures: dict[str, float | None] = dict.fromkeys(_METHOD_MEASURES)
    if result.people is None:                      # not of the method, or its people are not known
        return measures
    matched = _matched_people(result.people, visible, th)
    if run.people == "vlm":
        measures["people_found_share"] = len(matched) / len(visible) if visible else None
        measures["people_unmatched"] = len(result.people) - len(matched)
    if result.constraints is None or result.solver is None:      # the constraints call brought no valid answer
        return measures
    if run.constraints == "vlm" and turn_group is not None:
        listed = {number for use in result.constraints if use.constraint.kind == _TURN_KIND
                  for number in use.constraint.people}
        members = {person.id for person in visible if person.group == turn_group}
        right = sum(matched.get(number) in members for number in listed)
        measures["members_precision"] = right / len(listed) if listed else None
        measures["members_recall"] = right / len(members) if members else None
    measures["constraints_unusable"] = sum(use.status in _NOT_USED for use in result.constraints)
    measures["exclusions_broken"] = _exclusions_broken(result)
    return measures


def _turn_group(truth: GroundTruth, episode_id: str, ep: EpisodeGT, chosen: str | None) -> str | None:
    """The id of the group whose members wait their turn at the target in the episode's reference: where that
    reference is of people waiting, the group at the chosen target, or at the episode's first when none was
    chosen. None for another reference, and for one that names no group there."""
    if ep.situation != _TURN_SITUATION:
        return None
    return truth.group_for(episode_id, ep.target_ids[0] if chosen is None else chosen)


def run_episode(pipeline: Pipeline, inputs: SceneInputs, truth: GroundTruth, et: EpisodeTruth,
                episode: EpisodeInputs, cfg: Config, trial: int) -> dict:
    """Run one episode once under the run modes of the config and score it: the record for results.jsonl.

    The record is JSON-ready and holds everything about the episode but the ids of the examples shown. A
    tour-mode record has one more key, `tour`. A scene without a tour is refused in tour mode with a SceneError,
    before any model call. `measures.cost_usd` is what the record's calls cost at the prices of the entries
    used, None when one of them has none. The last six of `measures` are those of the constraint method
    (`_method_measures`), each None in a record that does not measure it.
    """
    obs, ep, th, run = inputs.observation, et.gt, cfg.thresholds, cfg.run
    _require_tour(inputs, cfg)
    first = truth.targets[ep.target_ids[0]]
    oracle_grounding = run.grounding == "oracle"

    outcome = tour = None
    if oracle_grounding:                           # the subtask a perfect Stage 1 would return: no call is made
        step = Subtask(type="navigate", description=f"Go to the {first.label}", target=first.label,
                       target_point=oracle_point(obs, first.anchor))
        plan = Plan(subtasks=[step], image_sha256=obs.sha256())
    elif run.grounding_source == "tour":           # the plan comes back with the points locate found
        tour = _tour_grounding(pipeline, inputs, episode, trial)
        outcome, plan = tour.outcome, tour.plan
    else:
        outcome = pipeline.plan(episode.request, episode.context, observation=obs, trial=trial)
        plan = outcome.plan

    subtasks = [] if plan is None else plan.subtasks
    matched = [_matched(subtask, obs, truth, ep, th) for subtask in subtasks]
    index = next((i for i, subtask in enumerate(subtasks)
                  if subtask.type == "navigate" and subtask.target_point is not None), None)
    chosen = None                                  # the target the reference context check and the oracle goal are for
    if plan is None:
        result = GoalResult(status=outcome.status)
    elif tour is not None and tour.stopped is not None:
        result = GoalResult(status=tour.stopped)
    elif index is None:
        result = GoalResult(status="no_target")
    else:
        if oracle_grounding:
            chosen = first.id
        elif matched[index] != _UNMATCHED:
            chosen = matched[index]
        # A tour plan has no image hash, so its point is passed in: infer_goal would locate a second time.
        point = None if tour is None else subtasks[index].target_point
        result = pipeline.infer_goal(obs, episode.request, episode.context, plan, index, target_point=point,
                                     context_check=_given_context_check(run.context_check, et, chosen),
                                     inferrer=_given_inferrer(run, pipeline, truth, ep, chosen, th, trial),
                                     trial=trial)
        if tour is not None:
            result = result.model_copy(update={"location": tour.locations[index]})
    if tour is not None and tour.calls:            # the locate calls follow the plan call
        result = result.model_copy(update={"calls": [*tour.calls, *result.calls],
                                           "timings_s": {**result.timings_s, "locate": tour.time_s}})
    if outcome is not None:                        # the plan call belongs to every result of a model-made plan
        result = result.model_copy(update={"calls": [outcome.call, *result.calls],
                                           "timings_s": {**result.timings_s, "plan": outcome.time_s}})
    sc = score(result, obs, truth, ep, th)

    # The reference is the chosen target's, not the scorer's match: the scorer matches nothing for a result
    # without a pose, while the context answer was given about the chosen target.
    reference = reference_people_matter(et, chosen)
    said = result.context_check if run.context_check == "vlm" else None
    own_plan = plan is not None and not oracle_grounding
    # A plan answer that could not be read is the model's own answer and counts as a wrong plan. An
    # outage is no answer: it stays unmeasured, like every other column of a vlm_error record.
    bad_plan = plan is None and outcome.status == "parse_failed"
    exact = recall = None
    if own_plan and ep.reference_plan is not None:
        exact, recall = decomposition_accuracy(reduce_plan(plan, matched), ep.reference_plan)
    elif bad_plan and ep.reference_plan is not None:
        exact, recall = False, 0.0
    tour_entry = None if tour is None else _tour_record(tour, inputs, truth, ep)
    frame_correct = None                           # measured in tour mode, where a frame shows a target of the episode
    if tour_entry is not None and tour_entry["reference_frames"]:      # a plan that chose no frame chose a wrong one
        frame_correct = tour_entry["chosen"] is not None and tour_frame_correct(
            tour_entry["chosen"], tour_entry["shown"], tour_entry["reference_frames"])
    measures = {
        "reference_people_matter": reference,
        "context_missed": None if said is None else (not said.people_matter and reference),
        "context_unnecessary": None if said is None else (said.people_matter and not reference),
        "later_subtask_hit_target": (False if bad_plan else None) if not own_plan else (
            index is not None and matched[index] == _UNMATCHED
            and any(later in ep.target_ids for later in matched[index + 1:])),
        "plan_exact": exact,
        "navigate_recall": recall,
        "tour_frame_correct": frame_correct,
        "latency_s": sum(result.timings_s.values()),
        "cost_usd": _cost_usd(result.calls, cfg),
    }
    visible = [person for person in truth.people if is_visible(person, obs, th)]
    measures.update(_method_measures(result, visible, _turn_group(truth, episode.episode_id, ep, chosen), run, th))
    record = {
        "scene_id": inputs.scene_id,
        "episode_id": episode.episode_id,
        "trial": trial,
        "request": episode.request,
        "config": {**{setting: getattr(run, setting) for setting in _RECORDED_SETTINGS},
                   "stages": {name: stage.model_dump() for name, stage in cfg.stages.items()}},
        "labels": {"request_type": ep.request_type, "situation": ep.situation, "factors": et.factors.model_dump()},
        "hidden_relevant_people": len(et.hidden_relevant_people),
        "plan": None if plan is None else plan.model_dump(mode="json"),
        "evaluated_subtask": index,
        "result": result.model_dump(mode="json"),
        "score": sc.model_dump(mode="json"),
        "measures": measures,
    }
    if tour_entry is not None:                     # a record of another mode keeps exactly the keys it had
        record["tour"] = tour_entry
    return record


def _reader(*path: str) -> _Read:
    """Reads the value at a path of keys in a record; None where a key is missing."""
    def read(record: dict) -> Any:
        value: Any = record
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        return value
    return read


def _measure_columns(records: Sequence[dict]) -> list[tuple[str, _Read]]:
    """The measure columns of the summary, in order, each with what reads its value from a record.

    There is one forbidden-region column per kind that any of the records has a region of. The measures of the
    constraint method come last."""
    kinds = sorted({kind for record in records for kind in record["score"].get("forbidden") or {}})
    columns = [(name, _reader("score", key)) for name, key in _SCORE_COLUMNS]
    columns += [(f"forbidden_{kind}_rate", _reader("score", "forbidden", kind)) for kind in kinds]
    columns += [(name, _reader("measures", key)) for name, key in _MEASURE_COLUMNS]
    columns += [(f"latency_{stage}_s", _reader("result", "timings_s", stage)) for stage in _TIMED]
    columns += [(f"status_{status}_rate", lambda record, status=status: record["result"]["status"] == status)
                for status in _FAILURE_STATUSES]
    columns += [(f"flag_{name}_rate", _reader("result", "flags", name)) for name in Flags.model_fields]
    columns.append(("hidden_relevant_people", _reader("hidden_relevant_people")))
    columns += [(name, _reader("measures", name)) for name in _METHOD_MEASURES]
    return columns


def _label(record: dict, key: str) -> str:
    """The group a record falls in under one grouping."""
    labels = record["labels"]
    value = labels[key] if key in _LABEL_KEYS else labels["factors"].get(key)
    if value is None:
        return _UNLABELLED
    if key == "group_size":
        return next((name for largest, name in _GROUP_SIZE_BINS if value <= largest), _GROUP_SIZE_BEYOND)
    if key == "view_angle_deg":
        return "<=30" if round(value, _VIEW_ANGLE_DECIMALS) <= _VIEW_ANGLE_SPLIT_DEG else ">30"
    return str(value)


def _mean(values: Sequence[float]) -> float | None:
    """None for no value: nothing in the summary is divided by zero."""
    return sum(values) / len(values) if values else None


def _trial_means(records: Sequence[dict], read: _Read) -> list[float]:
    """A column's mean within each trial, over the records that measured it; a trial where none did has no entry."""
    by_trial: dict[int, list[float]] = {}
    for record in records:
        value = read(record)
        if value is not None:
            by_trial.setdefault(record["trial"], []).append(float(value))      # a boolean counts 0 or 1
    return [sum(values) / len(values) for values in by_trial.values()]


def _row(group_by: str, group: str, records: Sequence[dict], columns: Sequence[tuple[str, _Read]],
         spread: bool) -> dict:
    """One row of the summary. `spread` adds each measure's minimum and maximum over the trials."""
    answered = [record for record in records if record["result"]["status"] != _NO_ANSWER]
    calls = [call for record in records for call in record["result"].get("calls") or []]     # of every record:
    row = {"group_by": group_by, "group": group,                                             # retries gather where
           "episodes": len({(record["scene_id"], record["episode_id"]) for record in records}),     # no answer came
           "records": len(records), "vlm_errors": len(records) - len(answered),
           "transient_retry_share": _mean([call["transient_retries"] > 0 for call in calls])}
    for name, read in columns:
        means = _trial_means(answered, read)
        row[name] = _mean(means)
        if spread:
            row[f"{name}_min"], row[f"{name}_max"] = min(means, default=None), max(means, default=None)
    costs = [record["measures"].get("cost_usd") for record in records]      # totals, over every record and trial
    row.update(calls=len(calls), cached_calls=sum(bool(call.get("cached")) for call in calls),
               cost_usd=None if None in costs else sum(costs, 0.0))
    return row


def summarise(records: Sequence[dict]) -> list[dict]:
    """The rows of summary.csv: the overall row, then one row per label of each grouping, every row with the same keys.

    A measure is the mean over the trials of its mean within a trial, taken over the records that got a model
    answer and measured it; None where there is none. With more than one trial among the records, every measure
    is followed by its minimum and maximum over the trials. `transient_retry_share` is the share of the row's
    model calls that needed a transient retry, over all its records and trials: the latency columns
    include the waits of those retries. The six measures of the constraint method follow
    every other measure, and are None in a row none of whose records measured them: every row of a baseline run.

    The last three columns are totals over all the row's records and trials: `calls`, the model calls made;
    `cached_calls`, those answered from the cache; and `cost_usd`, the sum of the records' costs, None when a
    record has none because a call of it used an entry without prices.
    """
    columns = _measure_columns(records)
    spread = len({record["trial"] for record in records}) > 1
    rows = [_row("overall", "all", records, columns, spread)]
    for key in (*_LABEL_KEYS, *_FACTOR_KEYS):
        groups: dict[str, list[dict]] = {}
        for record in records:
            groups.setdefault(_label(record, key), []).append(record)
        rows += [_row(key, label, groups[label], columns, spread) for label in sorted(groups)]
    return rows


@dataclass(frozen=True)
class _Scene:
    """One scene of a run: its folder, what was loaded from it and the episodes that are run."""
    folder: Path
    inputs: SceneInputs
    truth: GroundTruth
    episodes: tuple[tuple[EpisodeInputs, EpisodeTruth], ...]


def _scene_folders(folder: Path, setting: str) -> list[Path]:
    """The sub-folders of `folder` that hold a scene.json, by name. `setting` says where the folder was given.

    A folder that is missing or holds no scene is a ConfigError naming the setting, the folder looked for and
    the command that writes the placeholder scenes: data/ is git-ignored, so a fresh clone has none.
    """
    path = Path(folder).resolve()
    if not path.is_dir():
        problem = "no such folder"
    else:
        found = sorted((d for d in path.iterdir() if (d / "scene.json").is_file()), key=lambda d: d.name)
        if found:
            return found
        problem = "holds no scene (no sub-folder with a scene.json)"
    raise ConfigError(f"{setting}: {path}: {problem}; {_GENERATE_HINT}")


def _load_scenes(scenes_dir: Path, cfg: Config) -> list[tuple[Path, SceneInputs, GroundTruth]]:
    """Every scene of the folder as (folder, inputs, ground truth), in folder-name order.

    Refused with a SceneError: a scene without a tour when the run grounds on the tour; a scene_id that two
    folders carry, because records, pictures and example ids are keyed by it; an episode whose situation the
    config does not list; an episode without a target.
    """
    known = {*cfg.dev_situations, *cfg.held_out_situations}
    loaded = []
    folders: dict[str, Path] = {}
    for folder in _scene_folders(scenes_dir, "--scenes"):
        inputs, truth = load_scene(folder)
        _require_tour(inputs, cfg)
        scene_id = inputs.scene_id
        if scene_id in folders:
            raise SceneError(f"{scene_id}: scene_id is used by two folders: {folders[scene_id].name} and {folder.name}")
        folders[scene_id] = folder
        for episode_id, ep in truth.episodes.items():
            if ep.situation not in known:
                raise SceneError(f"{scene_id}: episode {episode_id}: unknown situation {ep.situation!r}; the known "
                                 "ones are the config's dev_situations and held_out_situations")
            if not ep.target_ids:
                raise SceneError(f"{scene_id}: episode {episode_id}: target_ids is empty; an episode is run "
                                 "towards its first target")
        loaded.append((folder, inputs, truth))
    return loaded


def _refuse_failed_check(folder: Path, scene_id: str, cfg: Config) -> None:
    """Raise a SceneError naming the scene and its failed checks when the scene check finds an error in the
    folder. A warning never stops a run: it is logged, because the warnings are the only sign of a flipped or
    shifted map, of a hidden person and of a hand-entered count that is off."""
    results = check_scene(folder, cfg, write_png=False)
    for result in results:
        if result.level == "warning":
            log.warning("%s: check %d (%s): %s", scene_id, result.number, result.name, result.message)
    failed = [result for result in results if result.level == "error"]
    if failed:
        checks = "; ".join(f"check {result.number} ({result.name}): {result.message}" for result in failed)
        raise SceneError(f"{scene_id}: the scene check fails for {folder}: {checks}")


def _examples(cfg: Config, scenes: Sequence[_Scene],
              skip_scene_check: bool) -> tuple[list[tuple[str, Example]], list[str]]:
    """The examples of the run as select_examples gives them, and the situations that need one and have none;
    neither, and `run.examples_dir` unread, unless the goal is marked by a model under a few-shot strategy.

    Unless the scene check is skipped, an example scene the run has not checked is checked like an evaluated
    one: a mis-exported example would reach the model as a worked answer with a wrongly projected point.
    """
    run = cfg.run
    if run.goal != "direct_marking" or cfg.stages["goal"].strategy not in _EXAMPLE_STRATEGIES:
        return [], []
    if run.examples_dir is None:
        raise ConfigError(f"run.examples_dir is not set: the goal strategy {cfg.stages['goal'].strategy} shows "
                          f"examples taken from a folder of scenes; {_GENERATE_HINT}")
    folders = _scene_folders(REPO_ROOT / run.examples_dir, "run.examples_dir")      # an absolute value is kept
    if not skip_scene_check:
        checked = {scene.folder.resolve() for scene in scenes}
        for folder in folders:
            if folder.resolve() not in checked:
                _refuse_failed_check(folder, read_scene_file(folder).scene_id, cfg)
    reaching = [et for scene in scenes for _, et in scene.episodes       # the episodes that can reach the goal stage
                if run.context_check != "always_no"
                and (run.context_check == "always_yes" or reference_people_matter(et, None))]
    needed = {et.gt.situation for et in reaching}
    return _select_examples(folders, {scene.inputs.scene_id for scene in scenes}, needed, cfg)


def _one_line(text: object) -> str:
    return " ".join(str(text).split())


def _run(scenes_dir: Path, cfg: Config, name: str, out_root: Path, skip_scene_check: bool = False) -> Path:
    """`run`, with the scene check optional. Everything that can refuse the run comes before its folder is made."""
    loaded = _load_scenes(scenes_dir, cfg)
    if not skip_scene_check:
        for folder, inputs, _ in loaded:
            _refuse_failed_check(folder, inputs.scene_id, cfg)

    scenes: list[_Scene] = []
    skipped = 0
    for folder, inputs, truth in loaded:
        ets = prepare(inputs, truth, cfg)
        kept = []
        for episode in inputs.episodes:
            et = ets[episode.episode_id]
            if et.gt.situation in cfg.held_out_situations and not cfg.run.final_test:
                skipped += 1
            else:
                kept.append((episode, et))
        for episode, et in kept:                   # no goal can be in a region there: the episode cannot be scored
            if not et.gt.acceptable_regions:       # the scene check only warns of it in a held-out situation
                why = "; ".join(et.problems) or f"{episode.episode_id}: no acceptable region"
                if not skip_scene_check:
                    raise SceneError(f"{inputs.scene_id}: episode {episode.episode_id} has no acceptable region and "
                                     f"cannot succeed: {why}")
                log.warning("%s: episode %s has no acceptable region and cannot succeed: %s",
                            inputs.scene_id, episode.episode_id, why)
        scenes.append(_Scene(folder, inputs, truth, tuple(kept)))
    log.info("skipped %d held-out episodes", skipped)
    if not any(scene.episodes for scene in scenes):
        log.warning("no episode to evaluate (%d held-out episodes skipped; --final-test includes them)", skipped)

    selected, examples_missing = _examples(cfg, scenes, skip_scene_check)
    example_ids = [example_id for example_id, _ in selected]
    pipeline = build_pipeline(cfg, [example for _, example in selected])       # reads the keys: the last refusal

    out = Path(out_root) / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
    out.mkdir(parents=True, exist_ok=False)
    doc = {"name": name, "scenes_dir": str(scenes_dir), "config": cfg.model_dump(mode="json"),
           "example_ids": example_ids, "examples_missing": examples_missing, "skipped_held_out": skipped,
           "vlm_errors": 0, "incomplete": True}
    config_path = out / "config.yaml"
    config_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")     # a run that dies stays marked
    (out / "prompts").mkdir()
    for path in prompt_files(pipeline.prompts_dir, stages_of(cfg.run)):      # the stage prompts as this run reads them
        shutil.copy(path, out / "prompts" / path.name)

    records = []
    spent: QuotaExceededError | None = None        # the provider's allowance ran out: no later call is made
    with open(out / "results.jsonl", "w", encoding="utf-8") as lines:
        try:
            for scene in scenes:
                scene_id, obs = scene.inputs.scene_id, scene.inputs.observation
                for episode, et in scene.episodes:
                    for trial in range(cfg.run.trials):
                        record = run_episode(pipeline, scene.inputs, scene.truth, et, episode, cfg, trial)
                        record["example_ids"] = example_ids
                        lines.write(json.dumps(record) + "\n")
                        lines.flush()              # what is done stays on disk when a later call stops the run
                        records.append(record)
                        if trial == 0:
                            episode_figure(out / "viz" / f"{scene_id}_{episode.episode_id}.png", obs, scene.truth,
                                           et.gt, GoalResult.model_validate(record["result"]),
                                           Score.model_validate(record["score"]),
                                           title=f"{scene_id}/{episode.episode_id}")
        except QuotaExceededError as error:        # the records finished before it are summarised like a whole run's
            spent = error

    rows = summarise(records)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as table:
        writer = csv.DictWriter(table, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)                     # None is written as an empty cell
    vlm_errors = rows[0]["vlm_errors"]
    doc.update(vlm_errors=vlm_errors, incomplete=vlm_errors > 0 or spent is not None)
    if spent is not None:                          # the reason, in the provider's words; the back end struck the key
        doc["stopped_by"] = _one_line(spent)
    config_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    if vlm_errors:                                 # the cache never holds a vlm_error: a rerun asks again
        log.warning("run incomplete: %d records ended in vlm_error", vlm_errors)
    if spent is not None:
        log.warning("run stopped, the provider's allowance is used up: %d records are kept in %s", len(records), out)
        raise spent
    return out


def run(scenes_dir: Path, cfg: Config, name: str, out_root: Path = Path("runs")) -> Path:
    """Run every episode of the scenes under `scenes_dir` and write the run folder, which is returned.

    The folder is `out_root/<timestamp>_<name>` with config.yaml, results.jsonl, summary.csv, viz/ and
    prompts/. A ConfigError, a SceneError or a MissingKeyError refuses the run before anything is written.
    A BackendError stops the run: the records written so far stay, config.yaml says `incomplete: true` and
    there is no summary.csv. A QuotaExceededError, the provider's allowance used up, stops it at once too, and
    is raised when the folder is as whole as it can be: summary.csv is written over the records finished before
    it, and config.yaml says `incomplete: true` and gives the provider's message as `stopped_by`.

    A situation that needs a worked example of the goal stage and has no eligible one refuses nothing: one
    warning names it, config.yaml lists it under `examples_missing` (an empty list when there is none), and
    its episodes are run with the examples there are, or with none.

    Every warning of the scene check is logged. An episode that would be run without an acceptable region is a
    SceneError naming the scene and the episode: no goal could succeed in it, and it would count in every
    rate. That includes a held-out episode under `final_test`, of which the scene check itself only warns.
    """
    return _run(scenes_dir, cfg, name, out_root, skip_scene_check=False)


def main(argv=None) -> int:
    """The command: run, then print the run folder with its success rate, its calls and what they cost.
    Returns the exit code."""
    parser = argparse.ArgumentParser(prog="python -m sgi.run", description="Run the pipeline over a folder of scenes.")
    parser.add_argument("--scenes", type=Path, help="folder whose sub-folders are scenes")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", help="entry of the config's model table to use for every stage")
    parser.add_argument("--stage-model", action="append", default=[], metavar="STAGE=ENTRY",
                        help="entry to use for one stage (plan, locate, context, goal, people or constraints), "
                             "applied after --model; may be given several times")
    parser.add_argument("--list-entries", action="store_true",
                        help="print the entries of the config's model table, and whether each one's key is set, "
                             "in place of a run")
    parser.add_argument("--strategy", action="append", default=[], metavar="STAGE=NAME",
                        help="prompting strategy for one stage (zero_shot or cot, and for goal also few_shot or "
                             "few_shot_cot), applied after --model and --stage-model; may be given several times")
    parser.add_argument("--prompts", type=Path, metavar="DIR",
                        help="folder to read the stage prompts from, in place of the repository's prompts/")
    parser.add_argument("--name", help="the run folder is runs/<timestamp>_<name>")
    parser.add_argument("--final-test", action="store_true", help="include the held-out situations")
    parser.add_argument("--no-cache", action="store_true", help="make every model call afresh")
    parser.add_argument("--skip-scene-check", action="store_true", help="run scenes that fail the scene check")
    args = parser.parse_args(argv)
    missing = [option for option, value in (("--scenes", args.scenes), ("--name", args.name)) if value is None]
    if missing and not args.list_entries:          # a run needs both; a listing needs neither
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    logging.basicConfig(level=logging.INFO)
    given: dict[str, Any] = {}                     # into the configuration, so that config.yaml records them
    if args.final_test:
        given["final_test"] = True
    if args.no_cache:
        given["use_cache"] = False
    if args.prompts is not None:                   # as given, it is a folder of the working directory
        given["prompts_dir"] = str(args.prompts.resolve())
    try:
        cfg = load_config(args.config, {"run": given} if given else None)
        if args.list_entries:
            print("\n".join(entry_lines(cfg)))
            return 0
        cfg = with_model_options(cfg, args.model, args.stage_model, args.strategy)     # recorded like the flags
        out = _run(args.scenes, cfg, args.name, Path("runs"), skip_scene_check=args.skip_scene_check)
    except QuotaExceededError as error:            # ahead of BackendError, of which it is one; _run alone raises it
        resume = _RESUMES if cfg.run.use_cache else _STARTS_OVER
        print(f"error: {_one_line(error)} ({_SPENT}: {resume})", file=sys.stderr)
        return 2
    except (ConfigError, SceneError, MissingKeyError, BackendError) as error:
        # The run was refused or stopped. Any other exception is a defect in this code and keeps its traceback.
        print(f"error: {_one_line(error)}", file=sys.stderr)                   # on one line, and nothing added
        return 2
    with open(out / "summary.csv", newline="", encoding="utf-8") as table:
        overall = next(csv.DictReader(table))
    incomplete = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))["incomplete"]
    rate = f"{float(overall['success_rate']):.3f}" if overall["success_rate"] else "n/a"     # no answered record
    cost = f", about ${float(overall['cost_usd']):.2f}" if overall["cost_usd"] else ""       # empty: no prices known
    print(f"{out}: success rate {rate} ({overall['records']} records, {overall['vlm_errors']} ended in vlm_error; "
          f"{overall['calls']} calls, {overall['cached_calls']} from the cache{cost})")
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
