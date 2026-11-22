"""Prompt templates and the assembly of each stage's request: the system text and the parts to send. Also the
list of people that the constraints step is shown, which is worked out from geometry."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from sgi.config import PROMPTS_DIR, ConfigError, RunConfig, StageName, Strategy, Thresholds
from sgi.inputs import ObservationData
from sgi.schemas import goal_to_wire
from sgi.solver import _faces, target_person       # the solver's own two rules: the list and the solver then agree
from sgi.types import DetectedPerson, GoalMark, GoalReasoning, GroundedTarget, Point, Subtask
from sgi.vlm.base import ImagePart, Part, TextPart
from sgi.vlm.points import POINT_FORMATS

_HEADER = re.compile(r"^## (\w+)\s*$")             # starts a block of a template
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
_REASONING_STRATEGIES = ("cot", "few_shot_cot")    # their answers start with the reasoning fields
_EXAMPLE_STRATEGIES = ("few_shot", "few_shot_cot")
_LOOK_AT_MODE = "vlm_look_at"                      # the heading mode whose answer carries look_at_point
_THIRDS = ("on the left", "in the centre", "on the right")      # of the image, by the column of a person's body point
_MM = 3                                            # decimals of a metre: equally near means equal when rounded to them


@dataclass(frozen=True)
class Example:
    """One worked example for the goal stage. It holds what the model is shown and no ground-truth field."""
    image: ImagePart                               # already marked at the example target
    request: str
    target_text: str
    goal_point: Point
    reasoning: GoalReasoning | None = None
    look_at_point: Point | None = None


def stages_of(run: RunConfig) -> tuple[StageName, ...]:
    """The stages of a run, which follow from its settings alone.

    plan, locate, context and goal always, whatever the oracle modes say; constraints as well when `run.goal` is
    constraints; people as well when `run.people` is then vlm. The set gives the prompt files a run's folder has
    to hold, the files the run copies and the clients it builds."""
    stages: list[StageName] = ["plan", "locate", "context", "goal"]
    if run.goal == "constraints":
        if run.people == "vlm":                    # with oracle people the harness brings them, and no model is asked
            stages.append("people")
        stages.append("constraints")
    return tuple(stages)


def prompt_files(prompts_dir: Path, stages: Sequence[StageName]) -> list[Path]:
    """The files a prompts folder holds for these stages, one each: <stage>.md."""
    return [Path(prompts_dir) / f"{stage}.md" for stage in stages]


def check_prompts_dir(prompts_dir: Path, stages: Sequence[StageName]) -> None:
    """Raise a ConfigError unless the folder can stand in for the repository's prompts/ for these stages.

    It has to hold the file of each of them, and each file every block that the repository's file of the same
    name has, each block with the {placeholders} of the repository's block and no other. One the builders do
    not fill would stop a run with a KeyError at the first request built from the block, and one left out
    would send every request without that part. The error names the folder and the file, and the blocks that
    are missing or the block and the placeholders at fault. The file of a stage that is not among `stages` is
    not looked at."""
    folder = Path(prompts_dir)
    if not folder.is_dir():
        raise ConfigError(f"prompts folder {folder}: no such folder")
    for path in prompt_files(folder, stages):
        if not path.is_file():
            names = ", ".join(file.name for file in prompt_files(folder, stages))
            raise ConfigError(f"prompts folder {folder}: {path.name} is missing; the folder has to hold {names}")
        reference = PROMPTS_DIR / path.name
        if not reference.is_file():                # a stage whose prompt the repository does not hold
            raise ConfigError(f"prompts folder {folder}: {path.name} cannot be checked: the repository has no "
                              f"{reference} to hold it against")
        held, ours = _blocks(path.stem, folder), _blocks(path.stem, PROMPTS_DIR)
        missing = [name for name in ours if name not in held]
        if missing:
            raise ConfigError(f"prompts folder {folder}: {path.name} has no block {', '.join(missing)}; it needs "
                              f"every block of the repository's prompts/{path.name}, each under a line '## <name>'")
        for name, block in ours.items():
            given, known = _placeholders(held[name]), _placeholders(block)
            faults = [f"unknown placeholder {{{placeholder}}}" for placeholder in sorted(given - known)]
            faults += [f"no placeholder {{{placeholder}}}" for placeholder in sorted(known - given)]
            if faults:
                raise ConfigError(f"prompts folder {folder}: {path.name}, block {name}: {'; '.join(faults)}; a block "
                                  f"holds the placeholders of that block of the repository's prompts/{path.name}, "
                                  f"and no other")


def _blocks(stage: str, prompts_dir: Path) -> dict[str, str]:
    """The named blocks of <stage>.md in the folder. A line `## <name>` starts a block, which runs to the next
    header."""
    found: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in (Path(prompts_dir) / f"{stage}.md").read_text(encoding="utf-8").splitlines():
        header = _HEADER.match(line)
        if header:
            current = found[header.group(1)] = []
        elif current is not None:
            current.append(line)
    return {name: _without_outer_blank_lines(lines) for name, lines in found.items()}


def _placeholders(block: str) -> set[str]:
    """The names a block holds in braces, each to be filled by the builder of its stage."""
    return set(_PLACEHOLDER.findall(block))


def _without_outer_blank_lines(lines: list[str]) -> str:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end])


def _fill(block: str, values: Mapping[str, str | None]) -> str:
    """A block with its placeholders filled, line by line.

    A line holding a placeholder whose value is None is dropped. Every other line is substituted in one pass,
    so inserted data is never scanned for placeholders. A placeholder that `values` does not hold raises KeyError.
    """
    kept = []
    for line in block.split("\n"):
        used = [values[name] for name in _PLACEHOLDER.findall(line)]
        if all(value is not None for value in used):
            kept.append(_PLACEHOLDER.sub(lambda match: values[match.group(1)], line))
    return "\n".join(kept)


def _text(blocks: Mapping[str, str], names: Sequence[str], values: Mapping[str, str | None]) -> TextPart:
    """The named blocks, filled and joined by a blank line. A block left empty is not emitted."""
    filled = [_fill(blocks[name], values) for name in names]
    return TextPart("\n\n".join(text for text in filled if text.strip()))


def _reasoning_block(strategy: Strategy) -> tuple[str, ...]:
    return ("reasoning",) if strategy in _REASONING_STRATEGIES else ()


def _point_instruction(point_format: str, image: ImagePart) -> str:
    return POINT_FORMATS[point_format].instruction(image.width, image.height)


def build_plan_request(strategy: Strategy, request: str, context: str | None, image: ImagePart | None,
                       tour: Sequence[tuple[int, ImagePart]] | None, point_format: str,
                       prompts_dir: Path = PROMPTS_DIR) -> tuple[str, list[Part]]:
    """View mode sends `image` to mark points on. Tour mode sends indexed frames and uses no point format.

    Like every builder here, it reads its template from `prompts_dir`, the repository's prompts/ by default."""
    if (image is None) == (tour is None):
        raise ValueError("a plan request takes exactly one of image and tour")
    blocks = _blocks("plan", prompts_dir)
    values = {"request": request, "context": context}
    if image is not None:
        values["point_instruction"] = _point_instruction(point_format, image)
        text = _text(blocks, ("task", *_reasoning_block(strategy), "point_format", "output"), values)
        return blocks["system"], [text, image]
    parts: list[Part] = [_text(blocks, ("task", *_reasoning_block(strategy), "tour", "output"), values)]
    for index, frame in tour:
        parts += [TextPart(f"Frame {index}"), frame]
    return blocks["system"], parts


def build_locate_request(strategy: Strategy, request: str, context: str | None, subtask: Subtask,
                         image: ImagePart, point_format: str,
                         prompts_dir: Path = PROMPTS_DIR) -> tuple[str, list[Part]]:
    blocks = _blocks("locate", prompts_dir)
    values = {"request": request, "context": context, "subtask": subtask.description, "target": subtask.target,
              "point_instruction": _point_instruction(point_format, image)}
    text = _text(blocks, ("task", *_reasoning_block(strategy), "point_format", "output"), values)
    return blocks["system"], [text, image]


def build_context_request(strategy: Strategy, request: str, subtask: Subtask, marked_image: ImagePart,
                          standoff_m: float, prompts_dir: Path = PROMPTS_DIR) -> tuple[str, list[Part]]:
    blocks = _blocks("context", prompts_dir)
    values = {"request": request, "subtask": subtask.description, "target": subtask.target,
              "standoff_m": f"{standoff_m:g}"}
    text = _text(blocks, ("task", *_reasoning_block(strategy), "output"), values)
    return blocks["system"], [text, marked_image]


def build_people_request(strategy: Strategy, image: ImagePart, point_format: str, max_people: int,
                         prompts_dir: Path = PROMPTS_DIR) -> tuple[str, list[Part]]:
    """The people step of the constraint method: the camera image as it is, and the most
    people the answer may list.

    Nothing of an episode goes in, neither request nor target, so every episode of a scene sends the same request
    and they share one cached answer."""
    blocks = _blocks("people", prompts_dir)
    values = {"max_people": str(max_people), "point_instruction": _point_instruction(point_format, image)}
    text = _text(blocks, ("task", *_reasoning_block(strategy), "point_format", "output"), values)
    return blocks["system"], [text, image]


def build_constraints_request(strategy: Strategy, request: str, subtask: Subtask, numbered_image: ImagePart,
                              people: str, relevant_people: str | None, standoff_m: float, max_constraints: int,
                              prompts_dir: Path = PROMPTS_DIR) -> tuple[str, list[Part]]:
    """The constraints step of the constraint method: the case, the list of people,
    the six kinds a constraint can be of, and the image with the target ringed and a number on every person.

    `people` is the list as `people_lines` gives it, an empty string when nobody was found: the request then says
    so in place of a list. `relevant_people` is the description of the context stage where a model gave one, as
    in the goal request. `standoff_m` is what the solver keeps without being told, and `max_constraints` the most
    constraints it reads."""
    blocks = _blocks("constraints", prompts_dir)
    names = ("task", "people" if people else "nobody", "relevant_people", *_reasoning_block(strategy), "constraints",
             "check", "output")
    values = {"request": request, "subtask": subtask.description, "target": subtask.target, "people": people,
              "relevant_people": relevant_people, "standoff_m": f"{standoff_m:g}",
              "max_constraints": str(max_constraints)}
    return blocks["system"], [_text(blocks, names, values), numbered_image]


def build_goal_request(strategy: Strategy, request: str, subtask: Subtask, marked_image: ImagePart,
                       relevant_people: str | None, examples: Sequence[Example],
                       point_format: str, heading_mode: str,
                       prompts_dir: Path = PROMPTS_DIR) -> tuple[str, list[Part]]:
    """Worked examples first, under a few-shot strategy that has any, then the case to answer.

    An example's answer is in the point format of the instruction and in the coordinates of its own image.
    """
    blocks = _blocks("goal", prompts_dir)
    parts: list[Part] = []
    if strategy in _EXAMPLE_STRATEGIES and len(examples) > 0:
        parts.append(_text(blocks, ("examples_intro",), {}))
        for example in examples:
            mark = GoalMark(reasoning=example.reasoning, goal_point=example.goal_point,
                            look_at_point=example.look_at_point)
            answer = goal_to_wire(mark, strategy, point_format, (example.image.width, example.image.height),
                                  heading_mode)
            parts += [example.image,
                      _text(blocks, ("example",), {"request": example.request, "target": example.target_text}),
                      TextPart(json.dumps(answer, ensure_ascii=False))]
    look_at_block = ("look_at",) if heading_mode == _LOOK_AT_MODE else ()
    names = ("task", "relevant_people", *_reasoning_block(strategy), *look_at_block, "point_format", "output")
    values = {"request": request, "subtask": subtask.description, "target": subtask.target,
              "relevant_people": relevant_people, "point_instruction": _point_instruction(point_format, marked_image)}
    parts += [_text(blocks, names, values), marked_image]
    return blocks["system"], parts


def _apart(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _faced(person: DetectedPerson, things: Sequence[tuple[str, tuple[float, float]]], half_angle: float) -> str:
    """The last clause of a person's line: of `things`, each a name and a ground position, those the person faces,
    nearest first. Of two as near, to the millimetre, the one given first comes first."""
    if person.yaw is None:
        return "facing not known"
    faced = []
    for name, xy in things:
        gap = _apart(person.xy, xy)
        if gap > 0 and _faces(person.xy, person.yaw, xy, half_angle):      # what stands where they stand has no bearing
            faced.append((gap, name))
    if not faced:
        return "faces none of these"
    faced.sort(key=lambda found: round(found[0], _MM))
    return "faces " + ", ".join(f"{name} ({gap:.1f} m)" for gap, name in faced)


def people_lines(people: Sequence[DetectedPerson], target: GroundedTarget, observation: ObservationData,
                 th: Thresholds) -> str:
    """The list of people that the constraints step is shown: one line each, in the order given, worked out from
    geometry alone. An empty string when there is nobody.

        Person 3: 2.4 m from the robot, 1.1 m from the target, on the left of the image; faces person 1 (0.9 m),
        the target (1.1 m).

    The third of the image is the one that the column of the person's `pixel` falls in, of the observation's own
    width. The person who is the target by the solver's rule (`sgi.solver.target_person`) is called
    `Person 3 (the target)`, in their own line and in the others', and has no distance to the target. For
    everyone else the target is then that person's centre; without such a person it is the ground under the
    target point. That is the point the solver measures to.

    The last clause lists every other person, the target unless a person is the target, and the robot, each when
    its bearing from this person is within `view_half_angle_deg` of their heading: the nearest first, each with
    its distance. It reads `faces none of these` when there is none, and `facing not known` without a heading.
    """
    robot_xy = (observation.robot.x, observation.robot.y)
    is_target = target_person(observation, target, people, th)
    target_xy = next((person.xy for person in people if person.number == is_target), target.world[:2])
    half_angle = math.radians(th.view_half_angle_deg)

    def name(person: DetectedPerson, word: str) -> str:
        return f"{word} {person.number}" + (" (the target)" if person.number == is_target else "")

    lines = []
    for person in people:
        said = [f"{_apart(person.xy, robot_xy):.1f} m from the robot"]
        if person.number != is_target:
            said.append(f"{_apart(person.xy, target_xy):.1f} m from the target")
        third = _THIRDS[min(max(int(3 * person.pixel[0] // observation.camera.width), 0), 2)]
        said.append(f"{third} of the image")
        things = [(name(other, "person"), other.xy) for other in people if other is not person]
        if is_target is None:
            things.append(("the target", target_xy))
        things.append(("the robot", robot_xy))
        lines.append(f"{name(person, 'Person')}: {', '.join(said)}; {_faced(person, things, half_angle)}.")
    return "\n".join(lines)
