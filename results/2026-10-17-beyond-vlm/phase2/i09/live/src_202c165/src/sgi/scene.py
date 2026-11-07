"""Scene format v0.1: the scene.json models, the ground-truth object, and the scene loader and writer."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from sgi.config import Thresholds
from sgi.geometry import Camera, RobotPose, wrap_angle
from sgi.inputs import EpisodeInputs, ObservationData, SceneInputs, TourFrameData
from sgi.polygons import Polygon

_SCENE_JSON = "scene.json"


class SceneError(ValueError):
    """A scene folder that breaks the format; the message reads "<scene id>: <path>: <problem>; ..."."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _four_by_four(rows: list[list[float]]) -> list[list[float]]:
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("must have 4 rows of 4 numbers")
    return rows


class Intrinsics(_Strict):
    fx: float
    fy: float
    cx: float
    cy: float


class RobotPoseModel(_Strict):
    x: float
    y: float
    yaw: float


class ObservationModel(_Strict):
    rgb: str
    depth: str
    width: int
    height: int
    intrinsics: Intrinsics
    T_world_camera: list[list[float]]
    robot_pose_world: RobotPoseModel

    _matrix_shape = field_validator("T_world_camera")(_four_by_four)


class TourFrameModel(_Strict):
    rgb: str
    T_world_camera: list[list[float]]

    _matrix_shape = field_validator("T_world_camera")(_four_by_four)


class Person(_Strict):
    id: str
    x: float
    y: float
    yaw: float
    group: str | None = None
    role: str | None = None                       # e.g. "staff": a person with a role is not a bystander (spec 4)


class Target(_Strict):
    id: str
    kind: Literal["place", "object", "person"]
    label: str
    anchor: tuple[float, float, float]
    footprint: Polygon | None = None
    match_radius: float | None = None
    front_yaw: float | None = None
    person_id: str | None = None
    tour_frames: list[int] | None = None


class Heading(_Strict):
    face_point: tuple[float, float] | None = None
    yaw: float | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Heading:
        if (self.face_point is None) == (self.yaw is None):
            raise ValueError("exactly one of face_point and yaw must be given")
        return self


class Region(_Strict):
    polygon: Polygon
    heading: Heading
    tolerance_deg: float = 45.0
    representative_point: tuple[float, float] | None = None
    target_id: str | None = None


class ForbiddenRegion(_Strict):
    kind: str
    polygon: Polygon
    target_id: str | None = None


class PlanStep(_Strict):
    type: Literal["navigate", "act"]
    target_ids: list[str] = []
    optional: bool = False


class Factors(_Strict):
    density: Literal["low", "medium", "high"] | None = None
    group_size: int | None = None
    bystanders: int | None = None
    other_groups: int | None = None
    view_angle_deg: float | None = None


class Rationale(_Strict):
    people: str
    expectations: str
    placement: str


class EpisodeGT(_Strict):
    request_type: Literal["grounded", "high_level"]
    target_ids: list[str]
    situation: str
    group_at_target: str | dict[str, str | None] | None = None
    acceptable_regions: list[Region] | None = None
    forbidden_regions: list[ForbiddenRegion] | None = None
    reference_plan: list[PlanStep] | None = None
    factors: Factors | None = None
    rationale: Rationale | None = None
    people_matter: bool | None = None


class EpisodeModel(_Strict):
    episode_id: str
    request: str
    context: str | None = None
    ground_truth: EpisodeGT


class Occupancy(_Strict):
    image: str
    resolution: float
    origin: list[float]

    @field_validator("origin")
    @classmethod
    def _no_rotation(cls, origin: list[float]) -> list[float]:
        if len(origin) == 2 or (len(origin) == 3 and origin[2] == 0):
            return origin[:2]
        raise ValueError("must be [x, y]; a third element is accepted only when it is 0")


class PolygonModel(_Strict):
    polygon: Polygon


class SceneGT(_Strict):
    people: list[Person] = []
    targets: list[Target]
    bounds: PolygonModel | None = None
    obstacles: list[PolygonModel] = []
    occupancy: Occupancy | None = None


class Source(_Strict):
    simulator: str
    version: str | None = None


class SceneFile(_Strict):
    format_version: Literal["0.1"]
    scene_id: str
    source: Source | None = None
    observation: ObservationModel
    tour: list[TourFrameModel] = []
    episodes: list[EpisodeModel]
    ground_truth: SceneGT


@dataclass(frozen=True)
class GroundTruth:
    scene_id: str
    people: tuple[Person, ...]
    targets: dict[str, Target]
    bounds: Polygon | None
    obstacles: tuple[Polygon, ...]
    occupancy: Occupancy | None
    occupancy_image: np.ndarray | None             # (rows, cols) uint8 grey
    episodes: dict[str, EpisodeGT]

    def person(self, person_id: str) -> Person:
        """The person with that id; KeyError when there is none."""
        for person in self.people:
            if person.id == person_id:
                return person
        raise KeyError(person_id)

    def group(self, group_id: str) -> list[Person]:
        """The members of a group in file order; empty when nobody carries the id."""
        return [person for person in self.people if person.group == group_id]

    def group_for(self, episode_id: str, target_id: str) -> str | None:
        """The group at that target in that episode, or None."""
        group = self.episodes[episode_id].group_at_target
        return group.get(target_id) if isinstance(group, dict) else group


def _error(scene_id: str, items: Sequence[str]) -> SceneError:
    return SceneError(f"{scene_id}: " + "; ".join(items))


def _why(exc: Exception) -> str:
    """A one-line reason for a file that could not be read."""
    return (exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)) or type(exc).__name__


def read_scene_file(scene_dir: Path) -> SceneFile:
    """Parse and schema-check scene.json. The load-time rules are not applied and no yaw is wrapped."""
    scene_dir = Path(scene_dir)
    scene_id = scene_dir.resolve().name
    try:
        data = json.loads((scene_dir / _SCENE_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _error(scene_id, [f"{_SCENE_JSON}: {_why(exc)}"]) from exc
    if isinstance(data, dict) and isinstance(data.get("scene_id"), str):
        scene_id = data["scene_id"]
    try:
        return SceneFile.model_validate(data)
    except ValidationError as exc:
        items = [f"{'.'.join(str(part) for part in err['loc']) or _SCENE_JSON}: "
                 f"{err['msg'].removeprefix('Value error, ')}" for err in exc.errors()]
        raise _error(scene_id, items) from exc


def _bad_yaw(path: str, yaw: float | None, limit: float) -> list[str]:
    if yaw is None or abs(yaw) <= limit:
        return []
    return [f"{path}: {yaw} is beyond pi in magnitude; yaws are in radians"]


def _duplicates(path: str, key: str, ids: Sequence[str]) -> list[str]:
    """One item per id that repeats an earlier one, reported at the later entry."""
    return [f"{path}.{i}.{key}: duplicate id {ident!r}" for i, ident in enumerate(ids) if ident in ids[:i]]


def _check_episode(at: str, episode: EpisodeGT, targets: set[str], groups: set[str], yaw_limit: float) -> list[str]:
    problems: list[str] = []
    several = len(episode.target_ids) > 1
    unknown = [t for t in episode.target_ids if t not in targets]
    if unknown:
        problems.append(f"{at}.target_ids: unknown target ids {unknown}")

    group = episode.group_at_target
    if isinstance(group, dict):
        for target_id, group_id in group.items():
            if target_id not in targets:
                problems.append(f"{at}.group_at_target: unknown target id {target_id!r}")
            if group_id is not None and group_id not in groups:
                problems.append(f"{at}.group_at_target: unknown group {group_id!r} for target {target_id!r}")
    elif group is not None:
        if group not in groups:
            problems.append(f"{at}.group_at_target: unknown group {group!r}")
        if several:
            problems.append(f"{at}.group_at_target: must be a map from target id to group "
                            "when target_ids has more than one entry")

    acceptable, forbidden = episode.acceptable_regions or [], episode.forbidden_regions or []
    regions = [(f"{at}.acceptable_regions.{j}", r) for j, r in enumerate(acceptable)]
    regions += [(f"{at}.forbidden_regions.{j}", r) for j, r in enumerate(forbidden)]
    for path, region in regions:
        if region.target_id is None:
            if several:
                problems.append(f"{path}.target_id: required when target_ids has more than one entry")
        elif region.target_id not in targets:
            problems.append(f"{path}.target_id: unknown target id {region.target_id!r}")
        elif episode.target_ids and region.target_id not in episode.target_ids:
            # A region of a target the episode does not name is never a candidate: no goal could succeed in it.
            # An episode that names no target at all is another fault, which the harness and the scene check report.
            problems.append(f"{path}.target_id: {region.target_id!r} is not one of this episode's target_ids "
                            f"{episode.target_ids}")
    for j, region in enumerate(acceptable):
        problems += _bad_yaw(f"{at}.acceptable_regions.{j}.heading.yaw", region.heading.yaw, yaw_limit)

    first = None                                                # the first navigation step that names targets
    for k, step in enumerate(episode.reference_plan or []):
        unknown = [t for t in step.target_ids if t not in targets]
        if unknown:
            problems.append(f"{at}.reference_plan.{k}.target_ids: unknown target ids {unknown}")
        if first is None and step.type == "navigate" and step.target_ids:
            first = step
    if first is not None and set(first.target_ids) != set(episode.target_ids):
        problems.append(f"{at}.reference_plan: the first navigation step with targets lists {first.target_ids}, "
                        f"not the episode's target_ids {episode.target_ids}")
    return problems


def _check(scene: SceneFile) -> list[str]:
    """The load-time rules that the schema cannot express, as "<path>: <problem>" items."""
    gt = scene.ground_truth
    yaw_limit = math.pi + Thresholds().yaw_tolerance_rad        # load_scene takes no config: the model's default
    people, targets = [p.id for p in gt.people], [t.id for t in gt.targets]
    groups = {p.group for p in gt.people if p.group is not None}

    problems = _bad_yaw("observation.robot_pose_world.yaw", scene.observation.robot_pose_world.yaw, yaw_limit)
    problems += _duplicates("ground_truth.people", "id", people)
    problems += _duplicates("ground_truth.targets", "id", targets)
    problems += _duplicates("episodes", "episode_id", [e.episode_id for e in scene.episodes])
    for i, person in enumerate(gt.people):
        problems += _bad_yaw(f"ground_truth.people.{i}.yaw", person.yaw, yaw_limit)
        if person.role == "":
            problems.append(f"ground_truth.people.{i}.role: an empty role; "
                            f"leave it out or use null for an ordinary person")
        if person.role and person.group is not None:
            # Staff given a line's group would become "the last in line".
            problems.append(f"ground_truth.people.{i}: person {person.id!r} has a role and a group; "
                            f"a person with a role belongs to no group")
    for i, target in enumerate(gt.targets):
        at = f"ground_truth.targets.{i}"
        problems += _bad_yaw(f"{at}.front_yaw", target.front_yaw, yaw_limit)
        if target.kind == "place" and target.footprint is None and target.match_radius is None:
            problems.append(f"{at}.footprint: a place target needs a footprint or a match_radius")
        if target.person_id is None:
            if target.kind == "person":
                problems.append(f"{at}.person_id: a person target needs a person_id")
        elif target.person_id not in people:
            problems.append(f"{at}.person_id: unknown person id {target.person_id!r}")
        if scene.tour:                                          # without a tour nothing reads the indices
            outside = [k for k in target.tour_frames or [] if not 0 <= k < len(scene.tour)]
            if outside:
                problems.append(f"{at}.tour_frames: {outside} not among the indices of the "
                                f"{len(scene.tour)}-frame tour; indices start at 0")
    if gt.bounds is None and gt.occupancy is None:
        problems.append("ground_truth.bounds: free space needs bounds, occupancy or both")
    for i, episode in enumerate(scene.episodes):
        problems += _check_episode(f"episodes.{i}.ground_truth", episode.ground_truth, set(targets), groups, yaw_limit)
    return problems


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))                   # drops an alpha channel


def _read_grey(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("L"))


def _read_depth(path: Path) -> np.ndarray:
    loaded = np.load(path)
    if not isinstance(loaded, np.ndarray):                      # an .npz archive holds named arrays, not one array
        loaded.close()
        raise ValueError("not a single .npy array")
    return loaded.astype(np.float32)


def _read_array(scene_dir: Path, path: str, file_name: str, reader: Callable[[Path], np.ndarray],
                shape: tuple[int, ...] | None, problems: list[str]) -> np.ndarray | None:
    """Read one file the scene names; a file that cannot be read or has the wrong shape is added to problems."""
    try:
        array = reader(scene_dir / file_name)
    except (OSError, ValueError, EOFError) as exc:
        problems.append(f"{path}: cannot read {file_name}: {_why(exc)}")
        return None
    if shape is not None and array.shape != shape:
        problems.append(f"{path}: {file_name} has shape {array.shape}, expected {shape}")
    return array


def _matrix(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


def _wrap_region(region: Region) -> Region:
    if region.heading.yaw is None:
        return region
    heading = region.heading.model_copy(update={"yaw": wrap_angle(region.heading.yaw)})
    return region.model_copy(update={"heading": heading})


def _wrap_episode(episode: EpisodeGT) -> EpisodeGT:
    if episode.acceptable_regions is None:
        return episode
    return episode.model_copy(update={"acceptable_regions": [_wrap_region(r) for r in episode.acceptable_regions]})


def _wrap_target(target: Target) -> Target:
    if target.front_yaw is None:
        return target
    return target.model_copy(update={"front_yaw": wrap_angle(target.front_yaw)})


def load_scene(scene_dir: Path) -> tuple[SceneInputs, GroundTruth]:
    """Load a scene folder as the inputs the pipeline may see and, separately, its ground truth."""
    scene_dir = Path(scene_dir)
    scene = read_scene_file(scene_dir)
    broken_rules = _check(scene)
    if broken_rules:
        raise _error(scene.scene_id, broken_rules)
    obs, gt = scene.observation, scene.ground_truth

    bad_arrays: list[str] = []
    rgb = _read_array(scene_dir, "observation.rgb", obs.rgb, _read_rgb, (obs.height, obs.width, 3), bad_arrays)
    depth = _read_array(scene_dir, "observation.depth", obs.depth, _read_depth, (obs.height, obs.width), bad_arrays)
    tour_rgb = [_read_array(scene_dir, f"tour.{i}.rgb", frame.rgb, _read_rgb, None, bad_arrays)
                for i, frame in enumerate(scene.tour)]          # a tour image is not held to width and height
    occupancy_image = None
    if gt.occupancy is not None:
        occupancy_image = _read_array(scene_dir, "ground_truth.occupancy.image", gt.occupancy.image, _read_grey,
                                      None, bad_arrays)
    if bad_arrays:
        raise _error(scene.scene_id, bad_arrays)

    k, pose = obs.intrinsics, obs.robot_pose_world
    camera = Camera(k.fx, k.fy, k.cx, k.cy, obs.width, obs.height, _matrix(obs.T_world_camera))
    observation = ObservationData(rgb, depth, camera, RobotPose(pose.x, pose.y, wrap_angle(pose.yaw)))
    inputs = SceneInputs(
        scene_id=scene.scene_id,
        observation=observation,
        tour=tuple(TourFrameData(i, image, _matrix(frame.T_world_camera))
                   for i, (frame, image) in enumerate(zip(scene.tour, tour_rgb))),
        episodes=tuple(EpisodeInputs(e.episode_id, e.request, e.context) for e in scene.episodes),
    )
    truth = GroundTruth(
        scene_id=scene.scene_id,
        people=tuple(p.model_copy(update={"yaw": wrap_angle(p.yaw)}) for p in gt.people),
        targets={t.id: _wrap_target(t) for t in gt.targets},
        bounds=gt.bounds.polygon if gt.bounds is not None else None,
        obstacles=tuple(o.polygon for o in gt.obstacles),
        occupancy=gt.occupancy,
        occupancy_image=occupancy_image,
        episodes={e.episode_id: _wrap_episode(e.ground_truth) for e in scene.episodes},
    )
    return inputs, truth


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def save_scene(scene_dir: Path, scene: SceneFile, rgb: np.ndarray, depth: np.ndarray,
               occupancy: np.ndarray | None = None, tour: Sequence[np.ndarray] = ()) -> None:
    """Write a scene folder: scene.json and the arrays under the file names the scene gives them.

    The load-time rules are not applied here; load_scene applies them when the folder is read back.
    """
    scene_dir, entry = Path(scene_dir), scene.ground_truth.occupancy
    problems = []
    if len(tour) != len(scene.tour):
        problems.append(f"tour: {len(tour)} images given for {len(scene.tour)} entries")
    if (occupancy is None) != (entry is None):
        problems.append("ground_truth.occupancy: the occupancy image and its entry must be given together")
    if problems:
        raise _error(scene.scene_id, problems)

    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / _SCENE_JSON).write_text(json.dumps(scene.model_dump(mode="json"), indent=2), encoding="utf-8")
    _write_image(scene_dir / scene.observation.rgb, rgb)
    depth_path = scene_dir / scene.observation.depth
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    with open(depth_path, "wb") as f:                           # a handle, so numpy cannot append ".npy" to the name
        np.save(f, np.asarray(depth, dtype=np.float32))
    if entry is not None:
        _write_image(scene_dir / entry.image, occupancy)
    for frame, image in zip(scene.tour, tour):
        _write_image(scene_dir / frame.rgb, image)
