"""Scene check: validates a v0.1 scene folder, from the generator or from an export.

    python -m sgi.check_scene <scene_dir> [--config PATH]

Twelve numbered checks, errors 1 to 7 and warnings 8 to 12, then each episode's derived people_matter and
factors, and the picture check.png in the scene folder. It reads ground truth, so it is harness code.

"The depth at a world point" is the depth image read at the pixel the point projects to, compared with the
point's own camera Z."""
from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from pydantic import BaseModel

from sgi.config import DEFAULT_CONFIG, Config, Thresholds, load_config
from sgi.freespace import Grid, static_grid
from sgi.geometry import Camera, RobotPose, ground_hits, in_image, lift_all, project
from sgi.gt.derive import EpisodeTruth, prepare, reference_people_matter
from sgi.inputs import ObservationData, SceneInputs
from sgi.scene import GroundTruth, Person, SceneError, load_scene
from sgi.viz import scene_figure

_Level = Literal["ok", "warning", "error"]

_NAMES = ("structure", "pose sanity", "floor depth", "landmarks", "anchors", "regions", "robot pose",
          "visibility", "standard goal", "map alignment", "free-space agreement", "factors")
_SEVERITY = {"ok": 0, "warning": 1, "error": 2}
_ROTATION_ATOL = 1e-3                              # check 2: how far T_world_camera may be from a rigid transform
_MAX_AXIS_Z = math.cos(math.radians(30.0))         # check 2: the optical axis keeps 30 degrees from vertical
_LANDMARK_HEIGHT_M = 1.0                           # check 4: a person's landmark is this far above their x, y
_COUNTED_FACTORS = ("group_size", "bystanders", "other_groups")     # check 12: the factors the harness derives


class CheckResult(BaseModel):
    number: int
    name: str
    level: _Level
    value: float | None = None
    message: str = ""


def has_errors(results: Sequence[CheckResult]) -> bool:
    return any(result.level == "error" for result in results)


def _result(number: int, findings: Sequence[tuple[_Level, str]] = (), value: float | None = None,
            summary: str = "") -> CheckResult:
    """One check's result from its findings, each a (level, text): the worst level, and the texts joined.

    `summary` says what the check measured; it leads the message whatever the level."""
    level = max((level for level, _ in findings), key=_SEVERITY.__getitem__, default="ok")
    message = "; ".join([*([summary] if summary else []), *(text for _, text in findings)])
    return CheckResult(number=number, name=_NAMES[number - 1], level=level, value=value, message=message)


def _errors(problems: Sequence[str]) -> list[tuple[_Level, str]]:
    return [("error", problem) for problem in problems]


def _warnings(problems: Sequence[str]) -> list[tuple[_Level, str]]:
    return [("warning", problem) for problem in problems]


def _structure(scene_dir: Path, cfg: Config) -> tuple[CheckResult, tuple[SceneInputs, GroundTruth] | None]:
    """Check 1: the scene loads and every situation label is one the config knows. Also returns the loaded scene."""
    try:
        inputs, truth = load_scene(scene_dir)
    except SceneError as exc:
        return _result(1, _errors([str(exc)])), None
    known = {*cfg.dev_situations, *cfg.held_out_situations}
    unknown = [f"{truth.scene_id}: episode {eid}: unknown situation {ep.situation!r}"
               for eid, ep in truth.episodes.items() if ep.situation not in known]
    return _result(1, _errors(unknown)), (inputs, truth)


def _pose_sanity(camera: Camera, th: Thresholds) -> CheckResult:
    """Check 2: T_world_camera is a rigid transform of an optical-frame camera at a plausible height and tilt."""
    T = camera.T_world_camera
    R, height = T[:3, :3], float(T[2, 3])
    problems = []
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=_ROTATION_ATOL):
        problems.append(f"the bottom row of T_world_camera is {np.round(T[3], 3).tolist()}, not [0, 0, 0, 1]")
    if not np.allclose(R.T @ R, np.eye(3), rtol=0.0, atol=_ROTATION_ATOL):
        problems.append("the rotation of T_world_camera is not orthonormal")
    determinant = float(np.linalg.det(R))
    if not abs(determinant - 1.0) <= _ROTATION_ATOL:
        problems.append(f"the rotation of T_world_camera has determinant {determinant:.3f}, not +1")
    if not R[2, 1] < 0:
        problems.append("camera +Y does not point down in the world: "
                        "USD camera axes (-Z forward, +Y up) or an inverted transform?")
    if not abs(R[2, 2]) <= _MAX_AXIS_Z:
        limit_deg = math.degrees(math.acos(_MAX_AXIS_Z))
        problems.append(f"the optical axis is nearer to vertical than {limit_deg:.0f} degrees")
    if not th.camera_height_min_m <= height <= th.camera_height_max_m:
        problems.append(f"the height is outside {th.camera_height_min_m} to {th.camera_height_max_m} m")
    return _result(2, _errors(problems), height, f"camera height {height:.2f} m")


def _cell_indices(grid: Grid, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid.index for arrays of positions: the mask of those on the grid, and the iy and ix of those."""
    ny, nx = grid.free.shape
    col, row = np.floor((x - grid.x0) / grid.resolution), np.floor((y - grid.y0) / grid.resolution)
    inside = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    return inside, row[inside].astype(np.intp), col[inside].astype(np.intp)


def _free_at(grid: Grid, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Grid.is_free for arrays of positions."""
    inside, iy, ix = _cell_indices(grid, x, y)
    free = np.zeros(x.shape, dtype=bool)
    free[inside] = grid.free[iy, ix]
    return free


def _floor_test_set(camera: Camera, people: Sequence[Person], grid: Grid,
                    th: Thresholds) -> tuple[np.ndarray, np.ndarray]:
    """Check 3's pixels, as a mask (H, W), and every pixel's ray-plane depth (H, W).

    A pixel is in the set when its ray meets the ground at a point P and the ground segment from under the
    camera to P keeps its clearance from every person's centre and stays on free cells, so that nothing the
    format describes stands between the camera and P. The segment is sampled every grid resolution and at P."""
    xy, plane_depth = ground_hits(camera)
    origin = camera.T_world_camera[:2, 3]
    hit = np.flatnonzero(np.isfinite(plane_depth))             # flat pixel indices; the arrays below are per hit
    step = xy.reshape(-1, 2)[hit] - origin
    length = np.hypot(step[:, 0], step[:, 1])
    direction = np.divide(step, length[:, None], out=np.zeros_like(step), where=length[:, None] > 0)
    clear = np.ones(hit.size, dtype=bool)
    for person in people:
        to_person = np.array([person.x, person.y]) - origin
        along = np.clip(direction @ to_person, 0.0, length)    # of the segment's point nearest the person
        gap = np.hypot(*(to_person - along[:, None] * direction).T)
        clear &= gap >= th.floor_person_clearance_m

    in_set = np.zeros(plane_depth.size, dtype=bool)
    marching = np.flatnonzero(clear)                           # all segments advance together, one sample per pass
    travelled = 0.0
    while marching.size:
        at = origin + np.minimum(travelled, length[marching])[:, None] * direction[marching]
        marching = marching[_free_at(grid, at[:, 0], at[:, 1])]    # a ray to the horizon ends at the grid's edge
        arrived = travelled >= length[marching]
        in_set[hit[marching[arrived]]] = True
        marching = marching[~arrived]
        travelled += th.grid_resolution_m
    return in_set.reshape(plane_depth.shape), plane_depth


def _floor_depth(observation: ObservationData, people: Sequence[Person], grid: Grid, th: Thresholds) -> CheckResult:
    """Check 3: on floor that nothing can hide, the depth image matches the depth of the ground plane."""
    in_set, plane_depth = _floor_test_set(observation.camera, people, grid, th)
    count = int(in_set.sum())
    if count < th.floor_min_pixels:
        return _result(3, _errors([f"too little visible floor: {count} pixels see free floor with nothing in "
                                   f"between, {th.floor_min_pixels} are needed"]))
    depth, plane = observation.depth[in_set].astype(np.float64), plane_depth[in_set]
    valid = np.isfinite(depth) & (depth > 0)
    relative_error = np.ones(count)                            # an invalid depth counts as wholly wrong
    relative_error[valid] = np.abs(depth[valid] - plane[valid]) / plane[valid]
    value = float(np.median(relative_error))
    problems = []
    if value > th.floor_depth_max_rel_error:
        problems.append(f"above the limit of {th.floor_depth_max_rel_error:.1%}: range depth, wrong units, wrong "
                        "tilt or height, a wrong axis convention or an inverted transform?")
    return _result(3, _errors(problems), value, f"median relative error {value:.2%} over {count} floor pixels")


def _depth_offset(observation: ObservationData, p_world) -> float | None:
    """The depth at a world point minus the point's camera Z: negative when something nearer hides the point.

    None when the point is out of view, and inf where the depth there is invalid."""
    camera = observation.camera
    projected = project(camera, p_world)
    if projected is None or not in_image(camera, projected[0], projected[1]):
        return None
    u, v, z = projected
    depth = float(observation.depth[math.floor(v), math.floor(u)])
    return depth - z if math.isfinite(depth) and depth > 0 else math.inf


def _landmarks(observation: ObservationData, truth: GroundTruth, tol: float) -> CheckResult:
    """Check 4: where a person or an anchor is in view, the depth image has something there or nearer."""
    landmarks = [(f"person {person.id}", (person.x, person.y, _LANDMARK_HEIGHT_M)) for person in truth.people]
    landmarks += [(f"target {target.id}", target.anchor) for target in truth.targets.values()]
    matched = occluded = skipped = 0
    problems = []
    for name, point in landmarks:
        offset = _depth_offset(observation, point)
        if offset is None:
            skipped += 1
        elif abs(offset) <= tol:
            matched += 1
        elif offset < 0:
            occluded += 1
        else:
            seen = "there is no depth there" if math.isinf(offset) else f"the depth there is {offset:.2f} m beyond it"
            problems.append(f"{name}: {seen}, so nothing stands there")
    if not matched:
        problems.append("no landmark matches the depth image")
    return _result(4, _errors(problems), matched, f"{matched} matched, {occluded} occluded, {skipped} skipped")


def _anchors(observation: ObservationData, truth: GroundTruth, tol: float) -> CheckResult:
    """Check 5: the anchor of every target an episode names is in view, on the surface the depth image shows."""
    named = dict.fromkeys(target_id for ep in truth.episodes.values() for target_id in ep.target_ids)
    problems = []
    for target_id in named:
        offset = _depth_offset(observation, truth.targets[target_id].anchor)
        if offset is None:
            problems.append(f"target {target_id}: the anchor is out of view")
        elif math.isinf(offset):
            problems.append(f"target {target_id}: there is no depth at the anchor")
        elif abs(offset) > tol:
            side = "beyond" if offset > 0 else "in front of"
            problems.append(f"target {target_id}: the depth at the anchor is {abs(offset):.2f} m {side} the anchor "
                            f"(tolerance {tol} m)")
    return _result(5, _errors(problems))


def _regions(ets: dict[str, EpisodeTruth], cfg: Config) -> CheckResult:
    """Check 6: every episode has an acceptable region, each with a representative point that passes the
    oracle path. An episode of a held-out situation that is left without a region is only a warning."""
    findings: list[tuple[_Level, str]] = []
    for eid, et in ets.items():
        regions = et.gt.acceptable_regions
        if not regions:
            level = "warning" if et.gt.situation in cfg.held_out_situations else "error"
            findings.append((level, "; ".join(et.problems) or f"{eid}: no acceptable region"))
        elif et.problems or any(region.representative_point is None for region in regions):
            findings.append(("error", "; ".join(et.problems) or f"{eid}: a region has no representative point"))
    return _result(6, findings)


def _robot_pose(robot: RobotPose, people: Sequence[Person], grid: Grid, th: Thresholds) -> CheckResult:
    """Check 7: the robot is not inside an obstacle or a person."""
    problems = []
    if not grid.is_free(robot.x, robot.y):
        problems.append(f"the robot at ({robot.x:.2f}, {robot.y:.2f}) is on a cell that is not free")
    for person in people:
        gap = math.hypot(robot.x - person.x, robot.y - person.y)
        if gap <= th.person_radius_m:
            problems.append(f"the robot is {gap:.2f} m from the centre of person {person.id}")
    return _result(7, _errors(problems))


def _visibility(ets: dict[str, EpisodeTruth]) -> CheckResult:
    """Check 8: every relevant person is visible in the image."""
    return _result(8, _warnings([f"{eid}: relevant people not visible: {', '.join(et.hidden_relevant_people)}"
                                 for eid, et in ets.items() if et.hidden_relevant_people]))


def _standard_goal(ets: dict[str, EpisodeTruth]) -> CheckResult:
    """Check 9: with nobody relevant, the standard goal passes; where it fails the scene or the view is why."""
    problems = []
    for eid, et in ets.items():
        failed = [target_id for target_id, ok in et.standard_goal_ok.items() if not ok]
        if failed and not et.relevant_people:
            problems.append(f"{eid}: the standard goal fails for {', '.join(failed)} with nobody relevant")
    return _result(9, _warnings(problems))


def _map_alignment(people: Sequence[Person], grid: Grid) -> CheckResult:
    """Check 10: nobody stands on a cell that is not statically free."""
    return _result(10, _warnings([f"person {person.id} stands on a cell that is not free: a shifted or flipped map?"
                                  for person in people if not grid.is_free(person.x, person.y)]))


def _beside_not_free(grid: Grid) -> np.ndarray:
    """The cells that are not free or touch one that is not, by an edge or a corner; beyond the grid nothing is free."""
    ny, nx = grid.free.shape
    padded = np.pad(~grid.free, 1, constant_values=True)
    beside = np.zeros((ny, nx), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            beside |= padded[dy:dy + ny, dx:dx + nx]
    return beside


def _free_space_agreement(observation: ObservationData, truth: GroundTruth, grid: Grid,
                          th: Thresholds) -> CheckResult:
    """Check 11: what the depth image shows agrees with the map: floor on free cells, anything standing up to
    the camera's height on cells that are not free. Points beyond the map, or on a person, are not counted."""
    x, y, z = lift_all(observation.depth, observation.camera).T
    counted, iy, ix = _cell_indices(grid, x, y)
    if truth.bounds is not None:                               # the grid is the bounds' bounding box, not the bounds
        bounds_only = dataclasses.replace(truth, obstacles=(), occupancy=None, occupancy_image=None)
        within = static_grid(bounds_only, th.grid_resolution_m).free[iy, ix]
        counted[counted] = within
        iy, ix = iy[within], ix[within]
    x, y, z = x[counted], y[counted], z[counted]

    floor = np.abs(z) <= th.floor_tolerance_m
    standing = (z > th.floor_tolerance_m) & (z < observation.camera.T_world_camera[2, 3])
    for person in truth.people:
        standing &= np.hypot(x - person.x, y - person.y) > th.floor_person_clearance_m
    # A wall's face lies exactly on a cell edge, so its points fall to either side: the neighbours count too.
    agreement = (("floor on free cells", grid.free[iy[floor], ix[floor]]),
                 ("obstacles on cells that are not free", _beside_not_free(grid)[iy[standing], ix[standing]]))
    fractions = [float(agrees.mean()) for _, agrees in agreement if agrees.size]
    summary = ", ".join(f"{label}: {agrees.mean():.1%} of {agrees.size} points" if agrees.size
                        else f"{label}: no points" for label, agrees in agreement)
    value = min(fractions, default=None)
    problems = []
    if value is not None and value < th.free_space_agreement_min:
        problems.append(f"below {th.free_space_agreement_min:.0%}: a shifted or flipped map?")
    return _result(11, _warnings(problems), value, summary)


def _factors(truth: GroundTruth, ets: dict[str, EpisodeTruth]) -> CheckResult:
    """Check 12: a hand-entered count agrees with the derived one."""
    problems = []
    for eid, et in ets.items():
        entered = truth.episodes[eid].factors      # the scene's own: et.factors has the derived values filled in
        if entered is None:
            continue
        for name in _COUNTED_FACTORS:
            hand, derived = getattr(entered, name), getattr(et.derived_factors, name)
            if hand is not None and hand != derived:
                problems.append(f"{eid}: {name} is {hand}, derived {derived}")
    return _result(12, _warnings(problems))


def _episode_line(eid: str, et: EpisodeTruth, hand_label: bool | None) -> str:
    """An episode's derived people_matter and factors, and the scene's own people_matter where it disagrees."""
    people_matter, factors = reference_people_matter(et, None), et.derived_factors
    view = None if factors.view_angle_deg is None else f"{factors.view_angle_deg:.1f}"
    line = (f"{eid}: people_matter={people_matter} group_size={factors.group_size} bystanders={factors.bystanders} "
            f"other_groups={factors.other_groups} view_angle_deg={view}")
    if hand_label is not None and hand_label != people_matter:
        line += f" (hand label people_matter={hand_label} disagrees)"
    return line


def _check(scene_dir: Path, cfg: Config, write_png: bool) -> tuple[list[CheckResult], list[str]]:
    """The results of check_scene, and one line per episode; no lines when the list stops early."""
    scene_dir = Path(scene_dir)
    structure, loaded = _structure(scene_dir, cfg)
    results = [structure]
    if has_errors(results):
        return results, []
    inputs, truth = loaded
    th, observation = cfg.thresholds, inputs.observation
    results.append(_pose_sanity(observation.camera, th))
    if has_errors(results):
        return results, []

    grid = static_grid(truth, th.grid_resolution_m)
    ets = prepare(inputs, truth, cfg)
    tol = th.anchor_depth_tolerance_m
    results += [
        _floor_depth(observation, truth.people, grid, th),
        _landmarks(observation, truth, tol),
        _anchors(observation, truth, tol),
        _regions(ets, cfg),
        _robot_pose(observation.robot, truth.people, grid, th),
        _visibility(ets),
        _standard_goal(ets),
        _map_alignment(truth.people, grid),
        _free_space_agreement(observation, truth, grid, th),
        _factors(truth, ets),
    ]
    if write_png:                                  # with the filled regions, so that their outlines are drawn
        filled = dataclasses.replace(truth, episodes={eid: et.gt for eid, et in ets.items()})
        scene_figure(scene_dir / "check.png", observation, filled)
    return results, [_episode_line(eid, et, truth.episodes[eid].people_matter) for eid, et in ets.items()]


def check_scene(scene_dir: Path, cfg: Config, write_png: bool = True) -> list[CheckResult]:
    """One result per check that ran, in number order; with write_png, check.png is written in the scene folder.

    The list stops after check 1 or check 2 when that check is an error: the later checks project, lift or
    derive ground truth, which behind an unknown label or a wrong pose would only add noise. No picture then."""
    return _check(scene_dir, cfg, write_png)[0]


def main(argv: Sequence[str] | None = None) -> int:
    """Print every check's verdict and the episode lines; 1 when a check is an error, else 0."""
    parser = argparse.ArgumentParser(prog="python -m sgi.check_scene", description="Validate a v0.1 scene folder.")
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    results, episode_lines = _check(args.scene_dir, load_config(args.config), write_png=True)
    for r in results:
        print(f"[{r.number:2d}] {r.level.upper():7s} {r.name}: {r.message}")
    if len(results) < len(_NAMES):
        print(f"checks {len(results) + 1} to {len(_NAMES)} skipped")
    for line in episode_lines:
        print(line)
    return 1 if has_errors(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
