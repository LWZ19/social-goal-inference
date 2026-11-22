"""Ground-truth derivation: what the scorer needs beyond the scene's facts.

Per episode: the regions filled from the reference rules, a representative point for each acceptable
region, the reference context check, the derived factors, and the relevant people the image hides.

The oracle path is what a perfect point-marking answer goes through: a world point is projected, rounded
to a Point and lifted back from depth, so what returns is the surface the depth image shows at that pixel."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sgi.config import Config, Thresholds
from sgi.evaluate import score
from sgi.freespace import Grid, connected, robot_grid
from sgi.geometry import RobotPose, bearing, in_image, lift, pixel_to_point, point_to_pixel, project, wrap_angle
from sgi.gt.regions import build_regions
from sgi.inputs import ObservationData, SceneInputs
from sgi.polygons import distance_to_boundary, distance_to_polygon, grid_points_in, point_in_polygon
from sgi.scene import EpisodeGT, Factors, ForbiddenRegion, GroundTruth, Person, Region, Target
from sgi.stages.goal import standard_goal
from sgi.types import GroundedTarget, Point

_VISIBILITY_HEIGHTS_M = (0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7)   # samples on a person's vertical axis
_GROUP_RULE_SITUATIONS = ("waiting_line", "conversation_group", "audience", "group_photo",
                          "lift_wait", "machine_use")                   # their rules need group_at_target
_RULE_SITUATIONS = ("no_one", *_GROUP_RULE_SITUATIONS)                  # the situations sgi.gt.regions has a rule for
_TIE_DECIMALS = 9     # distances to a region's boundary are compared to a nanometre: equal ones differ by float noise


@dataclass(frozen=True)
class EpisodeTruth:
    episode_id: str
    gt: EpisodeGT                                  # regions filled, representative points set
    standard_goal_ok: dict[str, bool]              # per target id
    factors: Factors                               # hand-entered values with derived ones filled in
    derived_factors: Factors                       # always the derived values
    relevant_people: tuple[str, ...]
    hidden_relevant_people: tuple[str, ...]
    problems: tuple[str, ...]                      # e.g. "e0: no region (waiting_line needs group_at_target)"


def oracle_point(observation: ObservationData, p_world) -> Point | None:
    """The Point a perfect answer would mark for a world point, or None when it is not in the image."""
    camera = observation.camera
    projected = project(camera, p_world)
    if projected is None or not in_image(camera, projected[0], projected[1]):
        return None
    x, y = pixel_to_point(projected[0], projected[1], camera.width, camera.height)
    return Point(x=x, y=y)


def _round_trip(observation: ObservationData, p_world,
                th: Thresholds) -> tuple[Point, tuple[float, float], np.ndarray] | None:
    """The oracle path of a world point: its Point, that Point's pixel and the world point depth gives there.

    None when the point is not in the image or its pixel has no depth."""
    point = oracle_point(observation, p_world)
    if point is None:
        return None
    camera = observation.camera
    pixel = point_to_pixel(point.x, point.y, camera.width, camera.height)
    world = lift(observation.depth, camera, *pixel, th.lift_window)
    if world is None:
        return None
    return point, pixel, world


def oracle_target(observation: ObservationData, target: Target, th: Thresholds) -> GroundedTarget | None:
    """The target as perfect grounding gives it: the anchor's Point, lifted from depth like a model's point."""
    found = _round_trip(observation, target.anchor, th)
    if found is None:
        return None
    point, pixel, world = found
    return GroundedTarget(point=point, pixel=pixel, world=(float(world[0]), float(world[1]), float(world[2])))


def oracle_path(observation: ObservationData, xy, th: Thresholds) -> np.ndarray | None:
    """The world point that comes back when the ground point xy is marked perfectly, or None when none does."""
    found = _round_trip(observation, (xy[0], xy[1], 0.0), th)
    return None if found is None else found[2]


def _reach(observation: ObservationData, truth: GroundTruth, th: Thresholds) -> tuple[Grid, np.ndarray]:
    """The grid of where the robot's centre may be, and the mask of its cells joined to the robot."""
    robot = observation.robot
    grid = robot_grid(truth, robot, th)
    return grid, connected(grid, (robot.x, robot.y))


def _passes(xy, region: Region, observation: ObservationData, grid: Grid, mask: np.ndarray, th: Thresholds) -> bool:
    """Whether xy passes the oracle path for the region: what comes back is on the floor, inside
    the region and in a free cell joined to the robot."""
    w = oracle_path(observation, xy, th)
    if w is None or abs(w[2]) > th.floor_tolerance_m or not point_in_polygon(w[:2], region.polygon):
        return False
    cell = grid.index(w[0], w[1])
    return cell is not None and bool(mask[cell])


def _reference(region: Region) -> tuple[float, float]:
    """The point that settles which of a region's equally deep points is taken: the one its heading faces,
    or, for a heading that is a direction, the centroid of its polygon."""
    if region.heading.face_point is not None:
        return region.heading.face_point
    x0, y0 = region.polygon[0]                     # measured from the first vertex, which keeps the sums small
    area = sx = sy = 0.0
    for (ax, ay), (bx, by) in zip(region.polygon, region.polygon[1:] + region.polygon[:1]):
        ax, ay, bx, by = ax - x0, ay - y0, bx - x0, by - y0
        cross = ax * by - bx * ay                  # twice the signed area of the triangle on this edge
        area, sx, sy = area + cross, sx + (ax + bx) * cross, sy + (ay + by) * cross
    return (x0 + sx / (3.0 * area), y0 + sy / (3.0 * area))


def _deepest_passing(region: Region, observation: ObservationData, grid: Grid, mask: np.ndarray,
                     th: Thresholds) -> tuple[float, float] | None:
    """The region's representative point, or None when none of its lattice points passes.

    Of the lattice points that pass, those no more than a grid step nearer the boundary than the farthest
    from it are equally deep. Where the heading faces a point, those of them no more than a grid step farther
    from that point than the nearest are equally near it, and the one of these nearest the robot is taken.
    Where the heading is a direction, the one least to the side of the line through the polygon's centroid
    along that direction is taken, then the nearest the centroid, then the nearest the robot. What is still
    tied goes to the smallest x, then y."""
    def passes(p: tuple[float, float]) -> bool:
        return _passes(p, region, observation, grid, mask, th)

    step = th.grid_resolution_m
    lattice = [(float(x), float(y)) for x, y in grid_points_in(region.polygon, step)]
    depth = {p: round(distance_to_boundary(p, region.polygon), _TIE_DECIMALS) for p in lattice}
    deepest = next((p for p in sorted(lattice, key=lambda p: -depth[p]) if passes(p)), None)
    if deepest is None:
        return None
    least = round(depth[deepest] - step, _TIE_DECIMALS)
    equals = [p for p in lattice if depth[p] >= least]                 # in the lattice's order, x then y, which sorted keeps
    (rx, ry), robot = _reference(region), (observation.robot.x, observation.robot.y)
    off = {p: round(math.dist(p, (rx, ry)), _TIE_DECIMALS) for p in equals}
    away = {p: round(math.dist(p, robot), _TIE_DECIMALS) for p in equals}
    if region.heading.face_point is not None:      # the reference is what the robot is to face
        nearest = next(p for p in sorted(equals, key=off.get) if passes(p))     # `deepest` is among them, so one passes
        most = round(off[nearest] + step, _TIE_DECIMALS)
        order = sorted((p for p in equals if off[p] <= most), key=away.get)
    else:                                          # the reference is on the region's axis, which runs along the heading
        ux, uy = math.cos(region.heading.yaw), math.sin(region.heading.yaw)
        aside = {p: round(abs((p[0] - rx) * uy - (p[1] - ry) * ux), _TIE_DECIMALS) for p in equals}
        order = sorted(equals, key=lambda p: (aside[p], off[p], away[p]))
    return next(p for p in order if passes(p))


def representative_point(region: Region, observation: ObservationData, truth: GroundTruth,
                         th: Thresholds) -> tuple[float, float] | None:
    """The point of the region's grid that passes the oracle path and is farthest from the region's boundary."""
    grid, mask = _reach(observation, truth, th)
    return _deepest_passing(region, observation, grid, mask, th)


def _distance_to_target(xy, target: Target) -> float:
    """From a ground point to the target's footprint or, without one, to its anchor's ground projection."""
    if target.footprint is not None:
        return distance_to_polygon(xy, target.footprint)
    return math.hypot(xy[0] - target.anchor[0], xy[1] - target.anchor[1])


def relevant_people(truth: GroundTruth, ep: EpisodeGT, th: Thresholds) -> list[Person]:
    """The people within the relevant radius of one of the episode's targets or regions, and everyone in a
    group with one of them, in the scene's order. A person with a role (staff) is left out unless one of
    the episode's targets refers to them."""
    targets = [truth.targets[target_id] for target_id in ep.target_ids]
    polygons = [region.polygon for region in [*(ep.acceptable_regions or []), *(ep.forbidden_regions or [])]]

    def near(person: Person) -> bool:
        xy = (person.x, person.y)
        distances = [_distance_to_target(xy, target) for target in targets]
        distances += [distance_to_polygon(xy, polygon) for polygon in polygons]
        return min(distances, default=math.inf) <= th.relevant_radius_m

    referred = {target.person_id for target in targets}
    near_ids = {person.id for person in truth.people if near(person) and (person.role is None or person.id in referred)}
    groups = {person.group for person in truth.people if person.id in near_ids and person.group is not None}
    return [person for person in truth.people if person.id in near_ids or person.group in groups]


def _sample_seen(observation: ObservationData, p_world, th: Thresholds) -> bool:
    """Whether the image shows the world point: it projects inside the image, onto a depth that matches its own."""
    camera = observation.camera
    projected = project(camera, p_world)
    if projected is None or not in_image(camera, projected[0], projected[1]):
        return False
    u, v, z = projected
    depth = float(observation.depth[math.floor(v), math.floor(u)])
    return math.isfinite(depth) and depth > 0 and abs(depth - z) <= th.visibility_depth_tolerance_m


def is_visible(person: Person, observation: ObservationData, th: Thresholds) -> bool:
    """The visibility rule: enough of the samples on the person's vertical axis are seen."""
    seen = sum(_sample_seen(observation, (person.x, person.y, z), th) for z in _VISIBILITY_HEIGHTS_M)
    return seen >= th.visibility_min_samples


def _why_no_region(situation: str, group_id: str | None, target_id: str) -> str:
    """What is missing when the reference rules give a target no acceptable region."""
    if situation not in _RULE_SITUATIONS:
        return f"no rule for {situation}"
    if group_id is None and situation in _GROUP_RULE_SITUATIONS:
        return f"{situation} needs group_at_target"
    return f"{situation} rule gives no region for {target_id}"


def _filled_regions(episode_id: str, ep: EpisodeGT,
                    truth: GroundTruth) -> tuple[list[Region], list[ForbiddenRegion], list[str]]:
    """The episode's acceptable and forbidden regions, built from the reference rules where the scene
    supplies no acceptable region, and a problem for every target the rules leave without one."""
    if ep.acceptable_regions:
        return list(ep.acceptable_regions), list(ep.forbidden_regions or []), []
    acceptable: list[Region] = []
    forbidden: list[ForbiddenRegion] = []
    problems: list[str] = []
    for target_id in ep.target_ids:
        group_id = truth.group_for(episode_id, target_id)
        built = build_regions(ep.situation, truth.targets[target_id], truth.people, group_id)
        regions, banned = ([], []) if built is None else built
        acceptable += [region.model_copy(update={"target_id": target_id}) for region in regions]
        forbidden += [region.model_copy(update={"target_id": target_id}) for region in banned]
        if not regions:
            problems.append(f"{episode_id}: no region ({_why_no_region(ep.situation, group_id, target_id)})")
    if ep.forbidden_regions is not None:           # the scene's own list stands, even an empty one
        forbidden = list(ep.forbidden_regions)
    return acceptable, forbidden, problems


def _view_angle_deg(target: Target, robot: RobotPose, truth: GroundTruth) -> float | None:
    """The angle between the target's front and the direction from its anchor to the robot's position;
    None for a target without a front."""
    front = target.front_yaw
    if front is None and target.kind == "person":
        front = truth.person(target.person_id).yaw
    if front is None:
        return None
    return abs(math.degrees(wrap_angle(bearing(target.anchor[:2], (robot.x, robot.y)) - front)))


def _derived_factors(episode_id: str, ep: EpisodeGT, robot: RobotPose, truth: GroundTruth) -> Factors:
    """The factors that follow from the people, the group at the first target and the robot's position.

    An episode without a target has no group at one and no view angle."""
    target_id = ep.target_ids[0] if ep.target_ids else None
    group_id = None if target_id is None else truth.group_for(episode_id, target_id)
    groups = {person.group for person in truth.people if person.group is not None}
    return Factors(
        group_size=0 if group_id is None else len(truth.group(group_id)),
        bystanders=sum(person.group is None and person.role is None for person in truth.people),
        other_groups=len(groups - {group_id}),
        view_angle_deg=None if target_id is None else _view_angle_deg(truth.targets[target_id], robot, truth),
    )


def _merged_factors(entered: Factors | None, derived: Factors) -> Factors:
    """The entered values with the derived ones where none was entered; a computed view angle always stands."""
    values = derived.model_dump()
    for name, value in (entered.model_dump() if entered is not None else {}).items():
        if value is not None and (name != "view_angle_deg" or derived.view_angle_deg is None):
            values[name] = value
    return Factors(**values)


def _episode_truth(episode_id: str, ep: EpisodeGT, observation: ObservationData, truth: GroundTruth,
                   grid: Grid, mask: np.ndarray, th: Thresholds) -> EpisodeTruth:
    """Everything derived for one episode. grid and mask are the scene's, as _reach gives them."""
    acceptable, forbidden, problems = _filled_regions(episode_id, ep, truth)
    if not ep.target_ids:                          # no goal can succeed there, whatever regions the scene supplies
        problems.append(f"{episode_id}: no target_ids")
    for i, region in enumerate(acceptable):
        if region.representative_point is None:
            point = _deepest_passing(region, observation, grid, mask, th)
            if point is None:
                problems.append(f"{episode_id}: region {i}: no representative point passes the oracle path")
            else:
                acceptable[i] = region.model_copy(update={"representative_point": point})
        elif not _passes(region.representative_point, region, observation, grid, mask, th):
            problems.append(f"{episode_id}: region {i}: supplied representative point fails the oracle path")
    gt = ep.model_copy(update={"acceptable_regions": acceptable, "forbidden_regions": forbidden})

    standard_goal_ok = {}
    for target_id in ep.target_ids:
        target = oracle_target(observation, truth.targets[target_id], th)
        if target is None:
            standard_goal_ok[target_id] = False
            continue
        result = standard_goal(observation, target, th).model_copy(update={"target": target})
        # Given every id, the scorer matches the first of them the point falls on, and beside another target
        # that can be the other one. Given this id alone, it tests the goal against this target's own regions,
        # whatever the order of target_ids (evaluated once per id).
        alone = gt.model_copy(update={"target_ids": [target_id]})
        standard_goal_ok[target_id] = score(result, observation, truth, alone, th).success

    derived = _derived_factors(episode_id, ep, observation.robot, truth)
    relevant = relevant_people(truth, gt, th)
    return EpisodeTruth(
        episode_id=episode_id,
        gt=gt,
        standard_goal_ok=standard_goal_ok,
        factors=_merged_factors(ep.factors, derived),
        derived_factors=derived,
        relevant_people=tuple(person.id for person in relevant),
        hidden_relevant_people=tuple(person.id for person in relevant if not is_visible(person, observation, th)),
        problems=tuple(problems),
    )


def prepare(inputs: SceneInputs, truth: GroundTruth, cfg: Config) -> dict[str, EpisodeTruth]:
    """The derived ground truth of every episode of the scene, by episode id. `truth` is left unchanged."""
    th, obs = cfg.thresholds, inputs.observation
    grid, mask = _reach(obs, truth, th)            # one grid and one mask serve every episode's regions
    return {episode_id: _episode_truth(episode_id, ep, obs, truth, grid, mask, th)
            for episode_id, ep in truth.episodes.items()}


def reference_people_matter(et: EpisodeTruth, matched_target_id: str | None) -> bool:
    """The reference context check: people matter when the standard goal fails for the matched
    target, or for the episode's first target when none matched.

    In an episode without a target no standard goal passes, so there they matter."""
    if matched_target_id is None and not et.gt.target_ids:
        return True
    target_id = et.gt.target_ids[0] if matched_target_id is None else matched_target_id
    return not et.standard_goal_ok[target_id]
