"""Manual provider check and point-format calibration (spec section 15): real requests, reported as answered.

    python -m sgi.smoke --model ENTRY [--stage-model STAGE=ENTRY ...] [--strategy STAGE=NAME ...] [--prompts DIR]
                        [--calibrate | --goal constraints] [--scene DIR]
    python -m sgi.smoke --model ENTRY --list-models
    python -m sgi.smoke --list-entries

--model puts every stage on one entry of the model table in configs/default.yaml; each --stage-model then puts
one stage on an entry of its own, each --strategy gives one stage its prompting strategy, and --prompts names
the folder the stage prompts are read from, all as in `python -m sgi.run`. One stage is never called here, the
context stage: a --stage-model or a --strategy for it would change nothing and is refused. A client is built,
and its key read, only for the entry of a stage that is called: the plan, locate and goal stages in a smoke
run, the locate stage alone with --calibrate, the people and constraints stages with --goal constraints.
--list-entries prints the entries of that table, one per line with whether its key is set, and asks nothing.
--list-models prints the model ids the provider of --model reports for its key, one per line, each followed by
what else an openai_compatible service says of the model: one real request, and no model is asked.

The smoke run sends the first episode of one scene to the entry's provider, without the cache: the plan call,
then the goal call for the plan's first navigation subtask (and a locate call between them when the plan left
that target unmarked). The context check is passed in as "people matter", so the goal stage always runs. For
each call it prints the model and its settings, the latency, the token counts, the keys of the raw answer in
the order the model wrote them, and the parsed answer; then the goal pose, and it draws the target and the
goal on the image as runs/smoke.png. When the scene has a tour, one more plan call follows, in tour mode on
the tour's first frame alone, and its outcome is printed on a line of its own: the schema of a one-frame
tour is the only one of this build that holds the JSON-schema keyword "const".

The keys of every answer are also held against the schema the call was made with, which lists them in the
order they are to be written in (sgi.schemas): reasoning before the point it argues for. An answer that was
accepted (status ok) and has them in another order gets one line starting `WARNING:`, which names the call and
the first object that differs, and the exit code is then 1: under that provider or setting the written
reasoning is no chain of thought, and the strategies that ask for it measure nothing. An answer that was
rejected gets its status alone, whatever its text holds.

With --goal constraints the smoke run is that of the constraint method (`run.goal: constraints`), and it is the
check to make by eye before a model's numbers under that method are trusted: that the model refers to people
by the numbers drawn on them. Two requests are sent for the first episode of the scene: the people call on
the camera image, then the constraints call on the image with the target ringed and a number on every person
found. The target is the episode's first, taken from ground truth, so no plan is asked for, and the context
check is passed in as "people matter". Each call is reported as in a smoke run, with the answer as the model
wrote it. Under the people call comes the list of people as the model is then shown it, one line each; under
the constraints call each constraint with what the solver made of it; then the goal pose. The image the
constraints call was shown is written as runs/smoke_numbered.png. No plan is made on a tour frame.

With --calibrate it asks the model, once in each point format, to point at four place and object targets, and
prints each format's median distance in pixels between the answers and the projected anchors. The entry's
point_format in configs/default.yaml is set from these two numbers. The entry calibrated is the one of the
locate stage: --model, unless a --stage-model names another for it. The four targets are the ones, over every
scene of --scene, whose anchors project farthest from the image centre; they are listed before the results.
Under each median comes what the image centre scores on the same points, at no call: an entry whose median is
not clearly below that number is answering the middle of the image, not pointing at the targets. A point that
got no answer is listed under its format with the reason: a null answer, parse_failed with the validation
error (a point outside the image: the format is wrong for the model), or vlm_error (no answer came: run again).

Exit codes: 0 when the result is ok and so is the plan on the tour frame, with the keys of every answer in the
order of its schema, when every point format got an answer, or when a listing was printed; 1 for any other plan
or result status, for an answer with its keys in another order, for a format without an answer, for a request
the provider refused and for a model listing that did not come; 2 for a missing key, an unknown model entry,
stage or strategy, a --stage-model or --strategy for a stage that is never called here, --goal constraints
together with --calibrate or --list-models, a prompts folder that lacks a file or a block or has a block with
other placeholders than the repository's, a shell variable that an openai_compatible entry is not built under,
a provider whose back end has no model listing, a scene that is missing, cannot be loaded or has nothing to
point at, and a provider whose allowance is used up, which is one `error:` line with its message, as in
`python -m sgi.run`.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import NamedTuple, Sequence

from PIL import Image
from pydantic import BaseModel

from sgi.config import Config, ConfigError, StageName, load_config, with_model_options
from sgi.geometry import in_image, point_to_pixel, project
from sgi.gt.derive import oracle_point
from sgi.imaging import draw_point
from sgi.inputs import EpisodeInputs, ObservationData, SceneInputs, TourFrameData
from sgi.pipeline import Pipeline
from sgi.prompts import people_lines
from sgi.scene import GroundTruth, SceneError, load_scene
from sgi.schemas import constraints_schema, goal_schema, location_schema, people_schema, plan_schema
from sgi.stages.constraints import numbered_image
from sgi.types import Call, ConstraintUse, ContextCheck, Plan, Subtask
from sgi.vlm.base import BackendError, QuotaExceededError, TransientError
from sgi.vlm.env import MissingKeyError
from sgi.vlm.factory import entry_lines, list_models, make_client
from sgi.vlm.points import POINT_FORMATS

_SCENE_JSON = "scene.json"
_DEFAULT_SCENES = Path("data/placeholder/eval")
_PICTURE = Path("runs") / "smoke.png"              # under the current directory, where runs/ is git-ignored
_NUMBERED = Path("runs") / "smoke_numbered.png"    # with --goal constraints: the image the constraints call was shown
_TARGET_COLOR = (255, 0, 255)
_GOAL_COLOR = (255, 0, 0)
_GENERATE_HINT = "the placeholder scenes are written by: python -m sgi.synth.generate --out data/placeholder"
_SMOKE_STAGES: tuple[StageName, ...] = ("plan", "locate", "goal")       # the stages a smoke run can call,
_METHOD_STAGES: tuple[StageName, ...] = ("people", "constraints")       # the two it calls with --goal constraints,
_CALIBRATED_STAGES: tuple[StageName, ...] = ("locate",)                 # and the one a calibration calls
_UNCALLED_STAGES: tuple[StageName, ...] = ("context",)    # never called: every smoke run passes the context check in
_BASELINE, _CONSTRAINTS = "direct_marking", "constraints"               # the two goals a smoke run can be of
_CALIBRATION_POINTS = 4                            # --calibrate asks for this many targets, once in each point format
_POINTED_KINDS = ("place", "object")               # the targets --calibrate points at; a person target is skipped
_SPENT = "the provider's allowance is used up: it answers no call until that is renewed"     # added to its message


class _Pointing(NamedTuple):
    """One target --calibrate asks a model to point at."""
    order: int                                     # its place among the candidates, in folder and then target order
    scene_id: str
    label: str
    observation: ObservationData
    pixel: tuple[float, float]                     # where the target's anchor projects
    off_centre: float                              # pixels from there to the image centre: the error of that answer


def _config(entry: str, stage_models: Sequence[str] = (), strategies: Sequence[str] = (),
            prompts: Path | None = None, goal: str = _BASELINE) -> Config:
    """The default configuration with every stage on `entry`, then each STAGE=ENTRY of `stage_models` and each
    STAGE=NAME of `strategies` applied, and `run.goal` set to `goal`.

    No cache, because the command exists to reach the provider. The goal strategy is cot unless `strategies`
    names another, so the goal answer has nested reasoning keys to show; no strategy is shown examples here.
    `prompts`, when given, is the folder the stage prompts are read from: a relative one is a folder of the
    working directory. A value of `stage_models` or `strategies` for the context stage is a ConfigError: no
    mode of this command calls that stage, so the option would be taken and change nothing."""
    run = {"use_cache": False, "goal": goal}
    if prompts is not None:
        run["prompts_dir"] = str(prompts.resolve())
    overrides = {"run": run, "stages": {"goal": {"strategy": "cot"}}}
    cfg = with_model_options(load_config(overrides=overrides), entry, stage_models, strategies)
    for option, values in (("--stage-model", stage_models), ("--strategy", strategies)):
        for value in values:                       # of the form STAGE=NAME, each of them: the line above saw to that
            stage = value.partition("=")[0].strip()
            if stage in _UNCALLED_STAGES:
                raise ConfigError(f"{option} {value.strip()}: python -m sgi.smoke never calls the {stage} "
                                  f"stage, so the option would change nothing; leave it out, or try it with "
                                  f"python -m sgi.run")
    return cfg


def _in_point_format(cfg: Config, point_format: str) -> Config:
    """`cfg` with the entry of the locate stage, the one --calibrate asks, in the point format given."""
    name = cfg.stages["locate"].model
    models = {**cfg.models, name: cfg.models[name].model_copy(update={"point_format": point_format})}
    return cfg.model_copy(update={"models": models})


def _pipeline(cfg: Config, stages: Sequence[StageName]) -> Pipeline:
    """A pipeline for the stages about to be called, and no other: one client per model entry those stages name,
    shared by the stages on that entry. Building a client reads its key, so none is asked for an entry that
    only a stage left out is on."""
    clients = {name: make_client(name, cfg) for name in dict.fromkeys(cfg.stages[stage].model for stage in stages)}
    return Pipeline(cfg, {stage: clients[cfg.stages[stage].model] for stage in stages})


def _scene_folders(scene: Path) -> list[Path]:
    """`--scene` as scene folders: itself when it holds a scene.json, else its sub-folders that do, by name."""
    if (scene / _SCENE_JSON).is_file():
        return [scene]
    if not scene.is_dir():
        problem = "no such folder"
    else:
        found = sorted((d for d in scene.iterdir() if (d / _SCENE_JSON).is_file()), key=lambda d: d.name)
        if found:
            return found
        problem = "holds no scene (no scene.json in it or in a sub-folder)"
    raise ConfigError(f"--scene: {scene.resolve()}: {problem}; {_GENERATE_HINT}")


def _one_line(text: object) -> str:
    return " ".join(str(text).split())


def _raw_answer(raw_text: str | None) -> object:
    """A raw answer as JSON, its keys in the order the model wrote them.

    The text is parsed from its first { to its last }, so a code fence round it does not matter. ValueError when
    it has no brace at all, or no JSON between them."""
    text = raw_text or ""
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


def _key_order(raw_text: str | None) -> str:
    """The keys of a raw answer as dotted paths, each once, in the order the model wrote them.

    A list adds [] to the path of its items. "not JSON" when the text does not parse."""
    try:
        answer = _raw_answer(raw_text)
    except ValueError:
        return "not JSON"
    paths: dict[str, None] = {}                    # an ordered set

    def walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}" if path else key
                paths.setdefault(here)
                walk(value, here)
        elif isinstance(node, list):
            for item in node:
                walk(item, f"{path}[]")

    walk(answer, "")
    return ", ".join(paths)


def _schema_sent(cfg: Config, call: Call, tour_indices: Sequence[int] | None = None) -> type[BaseModel]:
    """The wire schema a call of the smoke run was made with, built again as its stage built it: from the
    stage's strategy and what the call records of its request. `tour_indices` are the frames a plan on a tour
    showed, None for a plan on a view."""
    strategy, point_format, size = cfg.stages[call.stage].strategy, call.point_format, call.image_size
    if call.stage == "plan":
        return plan_schema(strategy, point_format, size, tour_indices)
    if call.stage == "locate":
        return location_schema(strategy, point_format, size)
    if call.stage == "people":
        return people_schema(strategy, point_format, size)
    if call.stage == "constraints":
        return constraints_schema(strategy)
    return goal_schema(strategy, point_format, size, cfg.run.heading_mode)      # the context stage is not called


def _order_break(raw_text: str | None, schema: type[BaseModel]) -> str | None:
    """Where the keys of a raw answer leave the order of the schema it answers, None when they keep it.

    The answer is held, object by object, against itself with its keys in the schema's order. The first object
    that differs is given as "<its keys> in place of <the schema's>", after "under <path>" when it is not the
    answer itself. None also for a text that is no valid answer to the schema: it has no place for each key,
    and the status of its call says what is wrong with it."""
    try:
        written = _raw_answer(raw_text)
        asked = schema.model_validate(written).model_dump()        # the same answer as the schema orders it
    except ValueError:                             # not JSON, or not valid: a ValidationError is a ValueError
        return None

    def walk(wrote: object, ordered: object, path: str) -> str | None:
        if isinstance(wrote, dict) and isinstance(ordered, dict):
            if list(wrote) != list(ordered):       # the same keys, validation saw to that: the order differs
                return f"{f'under {path}, ' if path else ''}{', '.join(wrote)} in place of {', '.join(ordered)}"
            pairs = [(wrote[key], ordered[key], f"{path}.{key}" if path else key) for key in ordered]
        elif isinstance(wrote, list) and isinstance(ordered, list):
            pairs = [(item, as_asked, f"{path}[]") for item, as_asked in zip(wrote, ordered)]
        else:
            return None
        return next((found for found in (walk(*pair) for pair in pairs) if found), None)

    return walk(written, asked, "")


def _keeps_order(tag: str, call: Call, schema: type[BaseModel]) -> bool:
    """Whether the keys of the call's raw answer are in the order of `schema`, the schema the call was made
    with. When they are not, one line starting WARNING: is printed, with `tag` to name the call.

    Only an answer the client accepted is held against the schema. One it rejected (words round the JSON, an
    answer cut off) may still hold a whole object between its braces, and its status already says what is wrong."""
    if call.status != "ok":
        return True
    found = _order_break(call.raw_text, schema)
    if found is not None:
        print(f"WARNING: {tag} the answer's keys are not in the order of the schema sent: {found}")
    return found is None


def _count(tokens: int | None) -> str:
    return "-" if tokens is None else str(tokens)


def _report(call: Call, answer: BaseModel | None, schema: type[BaseModel]) -> bool:
    """The five lines of one call, and under the fourth a warning when the raw answer's keys are not in the order
    of `schema`, the schema the call was made with. `answer` is the stage output read from it, None when it gave
    none. Returns whether the keys were in that order."""
    tag, usage = f"[{call.stage}]", call.usage
    version = f" ({call.model_version})" if call.model_version else ""
    size = "none" if call.image_size is None else f"{call.image_size[0]}x{call.image_size[1]}"
    print(f"{tag} {call.provider} {call.model}{version}, effort {call.reasoning_effort}, {call.point_format}, "
          f"image {size}")
    print(f"{tag} latency {call.latency_s:.2f} s, total {call.latency_total_s:.2f} s, "
          f"retries {call.validation_retries}+{call.transient_retries}")
    print(f"{tag} usage: input {_count(usage.input_tokens)}, output {_count(usage.output_tokens)}, "
          f"reasoning {_count(usage.reasoning_tokens)}")
    print(f"{tag} key order: {_key_order(call.raw_text)}")      # of the raw text: `parsed` has the schema's order
    in_order = _keeps_order(tag, call, schema)
    print(f"{tag} answer: {f'none ({_one_line(call.error)})' if answer is None else answer.model_dump_json()}")
    return in_order


def _view_run(pipeline: Pipeline, obs: ObservationData, episode: EpisodeInputs) -> bool:
    """Plan on the observation, infer the goal of the first navigation subtask, report every call and the
    result, and write the picture. Returns whether the result is ok and every answer had its keys in the order
    of its schema."""
    cfg = pipeline.cfg
    out = pipeline.plan(episode.request, episode.context, observation=obs)
    in_order = [_report(out.call, out.plan, _schema_sent(cfg, out.call))]
    subtasks = [] if out.plan is None else out.plan.subtasks
    index = next((i for i, subtask in enumerate(subtasks) if subtask.type == "navigate"), None)
    if index is None:
        status = out.status if out.plan is None else "no_target"
        print(f"result: {status}, pose_robot none")
        return False

    # The check passed in forces the goal stage without a context call.
    result = pipeline.infer_goal(obs, episode.request, episode.context, out.plan, index,
                                 context_check=ContextCheck(relevant_people="", people_matter=True))
    answers = {"locate": result.location, "goal": result.goal_mark}
    for call in result.calls:
        in_order.append(_report(call, answers.get(call.stage), _schema_sent(cfg, call)))
    pose = "none" if result.pose_robot is None else "({:.3f}, {:.3f}, {:.3f})".format(*result.pose_robot)
    print(f"result: {result.status}, pose_robot {pose}")

    picture = obs.rgb
    if result.target is not None:
        picture = draw_point(picture, *result.target.pixel, _TARGET_COLOR, "target")
    if result.goal_pixel is not None:
        picture = draw_point(picture, *result.goal_pixel, _GOAL_COLOR, "goal")
    _PICTURE.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(picture).save(_PICTURE)
    print(f"picture: {_PICTURE}")
    return result.status == "ok" and all(in_order)


def _as_written(call: Call, schema: type[BaseModel]) -> BaseModel | None:
    """The answer of a call that brought a valid one, read again from its raw text; None for any other call."""
    return schema.model_validate(_raw_answer(call.raw_text)) if call.status == "ok" else None


def _stated(number: int, use: ConstraintUse) -> str:
    """One constraint of an answer with what the solver made of it, as one line."""
    constraint = use.constraint
    about = f"people {constraint.people}" + (", the target" if constraint.target else "")
    if constraint.range_m is not None:
        about += f", {constraint.range_m.at_least:g} to {constraint.range_m.at_most:g} m"
    reason = "" if use.reason is None else f" ({use.reason})"
    return f"  {number}. {use.status}{reason}: {constraint.kind}, {about}: {constraint.because}"


def _method_run(pipeline: Pipeline, inputs: SceneInputs, truth: GroundTruth, episode: EpisodeInputs) -> bool:
    """The two calls of the constraint method for the episode's first target: report each call, the list of
    people and what became of each constraint, then the result, and write the image the constraints call was
    shown. Returns whether the result is ok and every answer had its keys in the order of its schema.

    The target is marked as perfect grounding marks it, from ground truth, and the check passed in forces the
    goal step without a context call. An episode without a target, or with one whose anchor is not in the
    image, is a SceneError that names it."""
    cfg, th, obs = pipeline.cfg, pipeline.cfg.thresholds, inputs.observation
    targets = truth.episodes[episode.episode_id].target_ids
    point = oracle_point(obs, truth.targets[targets[0]].anchor) if targets else None
    if point is None:
        problem = "the anchor of its first target is not in the image" if targets else "target_ids is empty"
        raise SceneError(f"{inputs.scene_id}: episode {episode.episode_id}: {problem}: there is no target to ring")
    target = truth.targets[targets[0]]
    step = Subtask(type="navigate", description=f"Go to the {target.label}", target=target.label, target_point=point)
    plan = Plan(subtasks=[step], image_sha256=obs.sha256())
    result = pipeline.infer_goal(obs, episode.request, episode.context, plan, 0,
                                 context_check=ContextCheck(relevant_people="", people_matter=True))
    in_order = []
    for call in result.calls:
        schema = _schema_sent(cfg, call)
        in_order.append(_report(call, _as_written(call, schema), schema))
        if call.stage == "people" and result.people is not None:        # the list the constraints call is shown
            print(f"people: {len(result.people)} numbered" if result.people else "people: none found")
            for line in filter(None, people_lines(result.people, result.target, obs, th).split("\n")):
                print(f"  {line}")
        if call.stage == "constraints" and result.constraints is not None:
            print(f"constraints: {len(result.constraints)} stated")
            for number, use in enumerate(result.constraints, 1):
                print(_stated(number, use))
    pose = "none" if result.pose_robot is None else "({:.3f}, {:.3f}, {:.3f})".format(*result.pose_robot)
    print(f"result: {result.status}, pose_robot {pose}")
    if result.people is not None:                  # without the people there was no such image
        _NUMBERED.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(numbered_image(obs, result.target, result.people)).save(_NUMBERED)
        print(f"picture: {_NUMBERED}")
    return result.status == "ok" and all(in_order)


def _tour_plan(pipeline: Pipeline, frame: TourFrameData, episode: EpisodeInputs) -> bool:
    """Plan on one tour frame alone and print the outcome as one line, and under it a warning when the raw
    answer's keys are not in the order of the schema. Returns whether the plan call is ok with its keys in order.

    With a single frame shown, the schema sent holds that frame's index as a "const", which no other request
    of this build contains: this call is what shows whether the provider accepts the keyword. A refusal is a
    BackendError that names this call. An allowance that is used up is no refusal of this call, and passes."""
    shown = f"tour frame {frame.index} alone"
    try:
        out = pipeline.plan(episode.request, episode.context, tour=[frame])
    except QuotaExceededError:
        raise
    except BackendError as error:
        raise BackendError(f"the plan call on {shown} was refused: {error}") from error
    if out.plan is None:
        outcome = f"{out.status} ({_one_line(out.call.error)})"
    else:
        chosen = [subtask.tour_frame for subtask in out.plan.subtasks if subtask.type == "navigate"]
        frames = ", ".join("none" if index is None else str(index) for index in chosen)
        outcome = f"ok, tour_frame {frames}" if chosen else "ok, no navigate subtask"
    print(f"[tour] plan on {shown}: {outcome}")
    in_order = _keeps_order("[tour]", out.call, _schema_sent(pipeline.cfg, out.call, out.tour_indices))
    return out.status == "ok" and in_order


def _smoke(cfg: Config, folder: Path) -> int:
    """Send the first episode of the scene in `folder` and report it; then, when the scene has a tour, plan
    once more on its first frame alone. Under `run.goal: constraints` the two calls of the constraint method
    are sent in place of both, on the episode's first target. Returns the exit code."""
    method = cfg.run.goal == _CONSTRAINTS
    pipeline = _pipeline(cfg, _METHOD_STAGES if method else _SMOKE_STAGES)     # first: a missing key is found before
    inputs, truth = load_scene(folder)                                         # anything is printed
    if not inputs.episodes:
        raise SceneError(f"{inputs.scene_id}: episodes: the scene has no episode to send")
    episode = inputs.episodes[0]
    if method:
        return 0 if _method_run(pipeline, inputs, truth, episode) else 1
    passed = [_view_run(pipeline, inputs.observation, episode)]
    if inputs.tour:
        passed.append(_tour_plan(pipeline, inputs.tour[0], episode))
    return 0 if all(passed) else 1


def _pointings(folders: Sequence[Path]) -> tuple[list[_Pointing], int]:
    """What --calibrate asks for, in folder order, and the number of candidates it was chosen from.

    A candidate is a place or object target whose anchor is in the image of its scene. The ones kept are those
    farthest from the image centre, the earlier of two equally far. A model that answers near the middle of
    every image is then far from each of them, which it is not from a target the robot is facing."""
    kept: list[_Pointing] = []
    candidates = 0
    for folder in folders:
        inputs, truth = load_scene(folder)
        obs, camera = inputs.observation, inputs.observation.camera
        for target in truth.targets.values():
            projected = project(camera, target.anchor)
            if target.kind in _POINTED_KINDS and projected is not None and in_image(camera, *projected[:2]):
                u, v = projected[:2]
                off_centre = math.hypot(u - camera.width / 2, v - camera.height / 2)
                kept.append(_Pointing(candidates, inputs.scene_id, target.label, obs, (u, v), off_centre))
                candidates += 1
        # Cut down after every scene, so that no more observations are held than are asked for, plus one scene's.
        kept = sorted(kept, key=lambda pointing: (-pointing.off_centre, pointing.order))[:_CALIBRATION_POINTS]
    if not kept:
        where = (folders[0].name if len(folders) == 1 else
                 f"{len(folders)} scenes, {folders[0].name} to {folders[-1].name}")
        raise SceneError(f"--scene: no place or object target with its anchor in the image in {where}: "
                         f"nothing to point at")
    return sorted(kept, key=lambda pointing: pointing.order), candidates


def _print_points(asked: Sequence[_Pointing], candidates: int, scenes: int) -> None:
    """The targets asked for: a count line, then one line each with the anchor's pixel and how far that is from
    the image centre."""
    rule = ", those farthest from the image centre" if len(asked) < candidates else ""
    print(f"points: {len(asked)} of {candidates} place and object targets in "
          f"{scenes} scene{'' if scenes == 1 else 's'}{rule}")
    for pointing in asked:
        u, v = pointing.pixel
        print(f'  {pointing.scene_id}: "{pointing.label}" at ({u:.0f}, {v:.0f}), '
              f"{pointing.off_centre:.1f} px from the centre")


def _no_point(call: Call) -> str:
    """Why a locate call brought no point: the model answered null, its answer was not valid (an answer outside
    the image says the format is wrong for the model), or no answer came at all (ask again)."""
    return "null answer" if call.status == "ok" else f"{call.status} ({_one_line(call.error)})"


def _calibrate(cfg: Config, folders: Sequence[Path]) -> int:
    """Ask for each chosen target in each point format; print the targets, then each format's median pixel error
    and, under it, what the image centre scores on the points that format answered, and one line for every
    point it did not answer, with the reason. Returns the exit code.

    A format's configuration differs from the smoke run's in the point format of the locate stage's entry
    alone, and each format gets clients of its own."""
    asked, candidates = _pointings(folders)
    answered_all = True
    for number, name in enumerate(POINT_FORMATS):
        pipeline = _pipeline(_in_point_format(cfg, name), _CALIBRATED_STAGES)
        errors, centre_errors, unanswered = [], [], []
        for pointing in asked:
            obs, (u, v) = pointing.observation, pointing.pixel
            text = f"Go to the {pointing.label}"
            subtask = Subtask(type="navigate", description=text, target=pointing.label)
            location, call, _ = pipeline.locate(obs, subtask, text, None)
            if location is not None and location.target_point is not None:      # a failed call is not an answer
                point = location.target_point
                pu, pv = point_to_pixel(point.x, point.y, obs.camera.width, obs.camera.height)
                errors.append(math.hypot(pu - u, pv - v))
                centre_errors.append(pointing.off_centre)
            else:
                unanswered.append((pointing, _no_point(call)))
        if number == 0:                            # with the first result: a request refused at once prints nothing
            _print_points(asked, candidates, len(folders))
        if errors:
            print(f"{name}: median error {statistics.median(errors):.1f} px "
                  f"({len(errors)} of {len(asked)} points answered)")
            print(f"{name}: image centre {statistics.median(centre_errors):.1f} px "
                  f"(the median error of the image centre over the points answered)")
        else:
            print(f"{name}: no answer (0 of {len(asked)} points answered)")
            answered_all = False
        for pointing, why in unanswered:           # the count alone would cost eight more calls to explain
            print(f'{name}: {pointing.scene_id}: "{pointing.label}" got no point: {why}')
    return 0 if answered_all else 1


def main(argv=None) -> int:
    """The command. Returns the exit code; an error it expects is one line on stderr, without a traceback."""
    parser = argparse.ArgumentParser(prog="python -m sgi.smoke",
                                     description="Check one model entry against its provider with real requests.")
    parser.add_argument("--model", help="entry of the model table in configs/default.yaml, used for every stage")
    parser.add_argument("--stage-model", action="append", default=[], metavar="STAGE=ENTRY",
                        help="entry to use for one stage (plan, locate or goal, and people or constraints with --goal "
                             "constraints: the context stage is not called here), applied after --model; may be given "
                             "several times")
    parser.add_argument("--strategy", action="append", default=[], metavar="STAGE=NAME",
                        help="prompting strategy for one stage of those (zero_shot or cot, and for goal also few_shot "
                             "or few_shot_cot), applied after --model and --stage-model; may be given several times")
    parser.add_argument("--prompts", type=Path, metavar="DIR",
                        help="folder to read the stage prompts from, in place of the repository's prompts/")
    parser.add_argument("--list-entries", action="store_true",
                        help="print the entries of the model table, and whether each one's key is set, in place of "
                             "the smoke run")
    parser.add_argument("--list-models", action="store_true",
                        help="print the model ids the provider of --model reports for its key, with what else an "
                             "openai_compatible service says of each, in place of the smoke run: one real request")
    parser.add_argument("--calibrate", action="store_true",
                        help="measure the pointing error of each point format in place of the smoke run")
    parser.add_argument("--goal", choices=(_BASELINE, _CONSTRAINTS), default=_BASELINE,
                        help="constraints: the smoke run sends the two calls of the constraint method, people and "
                             "constraints, for the first episode's target, and writes the image with the numbers")
    parser.add_argument("--scene", type=Path, default=_DEFAULT_SCENES,
                        help="a scene folder, or a folder of scene folders: the smoke run sends the first by name, "
                             "--calibrate chooses its points among all of them")
    args = parser.parse_args(argv)
    if args.model is None and not args.list_entries:        # the listing of the table is all that needs no entry
        parser.error("the following arguments are required: --model")
    for option, given in (("--calibrate", args.calibrate), ("--list-models", args.list_models)):
        if given and args.goal != _BASELINE:       # either stands in place of the smoke run, whose goal this names
            parser.error(f"--goal {args.goal} cannot be combined with {option}")
    try:
        if args.list_entries:
            print("\n".join(entry_lines(load_config())))
            return 0
        # Whatever follows: a bad entry, stage or strategy gives one error.
        cfg = _config(args.model, args.stage_model, args.strategy, args.prompts, args.goal)
        if args.list_models:
            for line in list_models(args.model, cfg):
                print(line)
            return 0
        folders = _scene_folders(args.scene)
        if args.calibrate:
            return _calibrate(cfg, folders)
        return _smoke(cfg, folders[0])
    except QuotaExceededError as error:            # ahead of BackendError, of which it is one: no call can be checked
        print(f"error: {_one_line(error)} ({_SPENT})", file=sys.stderr)
        return 2
    except (BackendError, TransientError) as error:
        # The provider refused the request: the finding of a smoke run. A model call that got no reply never
        # raises here (it is a vlm_error in its report); a model listing that got none does.
        print(f"error: {_one_line(error)}", file=sys.stderr)
        return 1
    except (MissingKeyError, ConfigError, SceneError) as error:
        print(f"error: {_one_line(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
