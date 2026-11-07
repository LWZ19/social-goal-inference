"""Placeholder scene generator (spec section 12): stands in for Isaac Sim until real scenes arrive.

    python -m sgi.synth.generate --out data/placeholder [--config PATH] [--seed N]
    python -m sgi.synth.generate --out DIR --rows NAME [--config PATH] [--seed N]

Writes <out>/eval and <out>/examples in scene format v0.1, with exact depth and ground truth, the same for
the same seed. With --rows it writes instead only the rows of the set NAME of _ON_REQUEST, straight into <out>
(see generate_rows): rows that the two splits never hold. A layout that fails the scene check, or whose
nobody-at-target standard goal cannot be reached, is rejected and drawn again. The reference regions of sgi.gt
are written into every scene kept, and every scene kept has a tour (spec 3.7): eight frames from a walk round
the targets, inside the room and clear of everything in it. A target's `tour_frames` are the frames it is
visible in: those that show its anchor, lettering on it, or half of it or more.

accepted_layouts gives the layouts kept for a seed without writing the set, for a generator that renders
the same layouts by other means (realistic-scenes spec 5.6). It and generate() take one accept step.
"""
from __future__ import annotations

import argparse
import logging
import math
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from sgi.check_scene import check_scene
from sgi.config import DEFAULT_CONFIG, Config, Thresholds, load_config
from sgi.freespace import reachable, robot_grid
from sgi.geometry import Camera, RobotPose, bearing, camera_on_robot, in_image, project, wrap_angle
from sgi.gt.derive import oracle_target, prepare
from sgi.polygons import distance_to_boundary, distance_to_polygon, point_in_polygon
from sgi.scene import (Factors, Intrinsics, ObservationModel, Person, RobotPoseModel, SceneFile, SceneGT, Source,
                       Target, TourFrameModel, load_scene, save_scene)
from sgi.stages.goal import standard_goal
from sgi.synth.recipes import (_READABLE_COS, CAMERA_LENS, CAMERA_MOUNT, Layout, _add_extras, recipe_audience,
                               recipe_conversation_group, recipe_group_photo, recipe_lift_wait, recipe_machine_use,
                               recipe_no_one, recipe_waiting_line, recipe_waiting_line_bent, transform_layout)
from sgi.synth.render import _FACES, Box, Cylinder, Figure, Solid, render

log = logging.getLogger(__name__)

# (recipe, view angle in degrees or number of people, kind, bystanders, second group)
# No waiting line has bystanders or a second group. Behind a line seen from the side the only free spots are past
# its far end, where the picture shows whoever stands there as more people in the line, over the very floor the
# robot should stop on. The extra people stand where the picture keeps them apart: beside a place with nobody at
# it, and beyond the conversation groups, about twice as far off as the group itself.
_EVAL = (("no_one", 0, "place", 0, False), ("no_one", 30, "place", 1, False), ("no_one", 60, "place", 0, False),
         ("no_one", 30, "deep_place", 0, False), ("no_one", 0, "object", 0, False), ("no_one", 0, "person", 0, False),
         ("waiting_line", 1, None, 0, False), ("waiting_line", 2, None, 0, False), ("waiting_line", 4, None, 0, False),
         ("waiting_line", 6, None, 0, False), ("conversation_group", 2, None, 2, False),
         ("conversation_group", 3, None, 0, True), ("conversation_group", 4, None, 1, True),
         ("waiting_line", 3, None, 0, False))
_EXAMPLES = (("no_one", 0, "place", 0, False), ("no_one", 60, "place", 0, False), ("waiting_line", 2, None, 0, False),
             ("waiting_line", 3, None, 0, False), ("conversation_group", 2, None, 0, False),
             ("conversation_group", 3, None, 0, False), ("no_one", 90, "person", 0, False))
_SPLITS = (("eval", _EVAL), ("examples", _EXAMPLES))
# The second pass: people watching a screen, a picture being taken (of that many posing, with the target among
# them or the one who takes it) and people waiting at a lift; then the lift and the screen with nobody at them,
# so that neither thing alone says that people matter. Its rows are written after all of the first pass,
# numbered on from it, and drawn with a random generator of their own, so the scenes of the first pass are what
# they were before there was a second. None has a bystander or a second group, and examples/ has no scene of a
# held-out situation.
_EVAL_MORE = (("audience", 3, None, 0, False), ("audience", 5, None, 0, False),
              ("group_photo", 3, "posing", 0, False), ("group_photo", 2, "taker", 0, False),
              ("lift_wait", 2, None, 0, False), ("lift_wait", 4, None, 0, False),
              ("no_one", 30, "lift", 0, False), ("no_one", 0, "screen", 0, False))
_EXAMPLES_MORE = (("audience", 4, None, 0, False),)
_MORE_SPLITS = (("eval", _EVAL_MORE), ("examples", _EXAMPLES_MORE))
_MORE_STREAM = 1                   # with the seed, what the second pass's random generator is made from
# Sets of rows written only on request, by generate_rows, and never into the two splits: `bend_and_machine` holds a
# waiting line that turns round a pillar, of four people and of six, and someone using a cash machine and a ticket
# machine with nobody else there (situation machine_use, held out). Each set is drawn with a random generator of its
# own, so writing it changes no scene of either pass, and none of its rows has a bystander or a second group.
_ON_REQUEST = {"bend_and_machine": (("waiting_line_bent", 4, None, 0, False), ("waiting_line_bent", 6, None, 0, False),
                                    ("machine_use", 1, "cash", 0, False), ("machine_use", 1, "ticket", 0, False))}
_ON_REQUEST_STREAM = 2             # with the seed, what the random generator of a set on request is made from
_PEOPLE_RECIPES = {"waiting_line": recipe_waiting_line, "conversation_group": recipe_conversation_group,
                   "audience": recipe_audience, "lift_wait": recipe_lift_wait,
                   "waiting_line_bent": recipe_waiting_line_bent}
_VIEW_ANGLE_SPREAD_DEG = 3.0       # a no_one scene is seen from up to this far inside the angle its row asks for
_SHIFT_M = 5.0                     # the random transform moves a layout up to this far along each axis
_MAX_ATTEMPTS = 20
_EXPECTED_WARNING = 9              # spec 6: the standard goal failing with nobody relevant. The standard goal stops
#                                    before the surface it sees, so a place seen obliquely keeps its stop and no
#                                    placeholder scene raises this; it stays allowed, as a layout that does is no fault
_UNREACHABLE = "standard_goal_unreachable"
_REASONS = (*(f"check_{number}" for number in range(1, 13)), _UNREACHABLE)
_SHADOW_M = 0.35                   # the renderer's shadows end nearer than this to the foot of what casts them
_TOUR_TURNS_DEG = (-90, -60, -30, 0, 30, 60, 90, 120)      # tour frame k: the observation's pose, turned this far,
#                                                            or as far the other way when that leads past the front
_TOUR_CLEARANCE_M = 0.8            # a tour pose keeps this far from the walls, from every obstacle and from every body
_TOUR_NEAREST_M = 1.5              # and comes no nearer than this to the point it walks round
_TOUR_STEP_M = 0.1                 # between the distances tried along one bearing
_TOUR_TURN_STEP_DEG = 5            # between the bearings tried for one frame
_TOUR_SHARE = 0.5                  # a tour frame shows a target of which it shows this share or more
_ON_TARGET_M = 1e-6                # a part stands on a target's ground when it is no further than this outside it
_ON_SURFACE_M = 0.05               # a frame shows a point of a surface where its depth is this near the point's: across
#                                    one pixel a face recedes by less, seen from as far off square as lettering is read
_SAME_DEPTH_M = 1e-3               # two pictures taken along the same rays show one surface where they differ by less
_LETTERING_STEP_M = 0.1            # between the points at which lettering is looked for, or less


def _corners(camera: Camera, x: float, y: float, yaw: float, half_x: float, half_y: float,
             top: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The eight corners of the upright block on the ground rectangle centred on (x, y), up to `top`, in the
    camera's frame: how far each lies to the right of its axis, below it, and ahead."""
    c, s = math.cos(yaw), math.sin(yaw)
    corners = np.array([(x + c * i * half_x - s * j * half_y, y + s * i * half_x + c * j * half_y, z, 1.0)
                        for i in (-1, 1) for j in (-1, 1) for z in (0.0, top)])
    return tuple((corners @ np.linalg.inv(camera.T_world_camera).T)[:, :3].T)


def _may_show(camera: Camera, x: float, y: float, yaw: float, half_x: float, half_y: float, top: float) -> bool:
    """Whether the camera can see into the upright block on the ground rectangle centred on (x, y), up to `top`.

    False only when the whole block lies behind the camera or beyond one side of its image."""
    u, v, z = _corners(camera, x, y, yaw, half_x, half_y, top)
    u, v = camera.fx * u, camera.fy * v              # a point is right of the image when u > (width - cx) z, and so on
    beyond = (z <= 0, u > (camera.width - camera.cx) * z, u < -camera.cx * z, v > (camera.height - camera.cy) * z,
              v < -camera.cy * z)
    return not any(side.all() for side in beyond)


def _block(thing: Solid | Figure) -> tuple[float, float, float, float, float, float]:
    """The upright block, standing on the ground, that holds the thing: the centre, the yaw and the half sides
    of its ground rectangle, and its top."""
    if isinstance(thing, Box):
        return thing.cx, thing.cy, thing.yaw, thing.sx / 2, thing.sy / 2, thing.z0 + thing.sz
    if isinstance(thing, Figure):
        reach, top = thing.radius, thing.height
    elif isinstance(thing, Cylinder):
        reach, top = max(thing.radius, thing.top_radius or 0.0), thing.z0 + thing.height
    else:
        reach, top = thing.radius, thing.z + thing.radius
    return thing.x, thing.y, 0.0, reach, reach, top


def _in_view(camera: Camera, thing: Solid | Figure) -> bool:
    """Whether the thing or the shadow at its foot can show in the camera's image."""
    x, y, yaw, half_x, half_y, top = _block(thing)
    # A figure's shadow grows with its girth, as measured by Figure.radius.
    shadow = _SHADOW_M * thing.radius / Figure.radius if isinstance(thing, Figure) else _SHADOW_M
    return _may_show(camera, x, y, yaw, half_x + shadow, half_y + shadow, top)


def _render(layout: Layout, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """The rgb and depth images of the layout: its room, what stands in it and its people.

    What cannot show in the image is left out first. The renderer would test it on every ray when part of
    it is beside the camera, and the picture is the same without it."""
    boxes, props, figures = ([thing for thing in things if _in_view(camera, thing)]
                             for things in (layout.boxes, layout.props, layout.figures))
    return render(camera, boxes, figures, room=layout.room, props=props)


def _density(n_people: int) -> str:
    """The scene author's label for how many people are in the scene."""
    return "low" if n_people <= 2 else "medium" if n_people <= 5 else "high"


def build_scene(layout: Layout, scene_id: str) -> tuple[SceneFile, np.ndarray, np.ndarray]:
    """The layout as a scene file, with the rgb and depth images of the camera mounted on its robot.

    The episodes are the layout's with the density label added; the harness derives the other factors."""
    camera = Camera(**CAMERA_LENS, T_world_camera=camera_on_robot(layout.robot, **CAMERA_MOUNT))
    rgb, depth = _render(layout, camera)
    density = _density(len(layout.scene_gt.people))
    observation = ObservationModel(
        rgb="rgb.png", depth="depth.npy", width=camera.width, height=camera.height,
        intrinsics=Intrinsics(fx=camera.fx, fy=camera.fy, cx=camera.cx, cy=camera.cy),
        T_world_camera=camera.T_world_camera.tolist(),
        robot_pose_world=RobotPoseModel(x=layout.robot.x, y=layout.robot.y, yaw=layout.robot.yaw))
    episodes = [episode.model_copy(update={"ground_truth": episode.ground_truth.model_copy(
        update={"factors": Factors(density=density)})}) for episode in layout.episodes]
    scene = SceneFile(format_version="0.1", scene_id=scene_id, source=Source(simulator="sgi-synth"),
                      observation=observation, episodes=episodes, ground_truth=layout.scene_gt)
    return scene, rgb, depth


def _turn_about(centre, angle: float) -> np.ndarray:
    """The 4 x 4 world transform of a turn by `angle` about the vertical axis through the ground point `centre`."""
    c, s = math.cos(angle), math.sin(angle)
    x, y = centre
    return np.array([[c, -s, 0.0, x - c * x + s * y],
                     [s, c, 0.0, y - s * x - c * y],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def _shows(camera: Camera, depth: np.ndarray, anchor, tol: float) -> bool:
    """The rule of scene check 5 for one frame: the anchor is in view, on the surface the frame's depth shows."""
    projected = project(camera, anchor)
    if projected is None or not in_image(camera, projected[0], projected[1]):
        return False
    u, v, z = projected
    seen = float(depth[math.floor(v), math.floor(u)])
    return math.isfinite(seen) and seen > 0 and abs(seen - z) <= tol


def _own(layout: Layout, people: Sequence[Person], target: Target, th: Thresholds) -> list[Solid | Figure]:
    """What the layout draws the target with: the figure of the person it is, or else every one of the
    layout's boxes that stands wholly on the target's ground, which is its footprint or, without one, its
    match radius round the anchor."""
    if target.kind == "person":                      # figures[i] draws people[i]
        return [figure for person, figure in zip(people, layout.figures) if person.id == target.person_id]
    radius = th.default_match_radius_m if target.match_radius is None else target.match_radius

    def on_it(x: float, y: float, reach: float) -> bool:
        """Whether the disc of radius `reach` round (x, y) lies on the target's ground."""
        if target.footprint is None:
            return math.hypot(x - target.anchor[0], y - target.anchor[1]) + reach <= radius + _ON_TARGET_M
        return (distance_to_polygon((x, y), target.footprint) <= _ON_TARGET_M
                and distance_to_boundary((x, y), target.footprint) >= reach - _ON_TARGET_M)

    def stands_on_it(part: Solid) -> bool:
        x, y, yaw, half_x, half_y, _ = _block(part)
        if not isinstance(part, Box):                # round: its ground is the disc about its axis
            return on_it(x, y, half_x)
        c, s = math.cos(yaw), math.sin(yaw)
        return all(on_it(x + c * i * half_x - s * j * half_y, y + s * i * half_x + c * j * half_y, 0.0)
                   for i in (-1, 1) for j in (-1, 1))

    return [part for part in layout.boxes if stands_on_it(part)]


def _reads(camera: Camera, depth: np.ndarray, box: Box) -> bool:
    """Whether the frame with this depth image shows the lettering of the box so that it can be read: all of
    the part of the label face that the lettering may fill, from no further off the square view of it than
    lettering is read from."""
    normal, sign, a_axis, a_sign, b_axis, b_sign = _FACES[box.label_face]
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    axes = np.array([(c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0)])               # the box's own, in the world
    size = np.array([box.sx, box.sy, box.sz])
    # The lettering lies about the middle of the panel, or of the face without one: in shares of the face, as read.
    left, bottom, right, top = box.panel_rect if box.panel is not None else (0.0, 0.0, 1.0, 1.0)
    across, up = box.label_width * (right - left), box.label_height * (top - bottom)
    middle = (np.array([box.cx, box.cy, box.z0 + box.sz / 2]) + sign * size[normal] / 2 * axes[normal]
              + a_sign * ((left + right) / 2 - 0.5) * size[a_axis] * axes[a_axis]
              + b_sign * ((bottom + top) / 2 - 0.5) * size[b_axis] * axes[b_axis])
    to_camera = camera.T_world_camera[:3, 3] - middle
    if to_camera @ (sign * axes[normal]) < _READABLE_COS * np.linalg.norm(to_camera):
        return False
    along = [np.linspace(-0.5, 0.5, math.ceil(share * size[axis] / _LETTERING_STEP_M) + 1) * share * size[axis]
             for share, axis in ((across, a_axis), (up, b_axis))]
    return all(_shows(camera, depth, middle + a * axes[a_axis] + b * axes[b_axis], _ON_SURFACE_M)
               for a in along[0] for b in along[1])


def _widened(camera: Camera, things: Sequence[Solid | Figure]) -> tuple[Camera, int, int]:
    """The camera with its image grown until it holds all of the things, and the column and the row that
    its first pixel has in the grown image.

    The image grows by whole pixels, so the rays through the pixels it had stay as they were, and by no more
    than its own width or height each way: further out a picture stretches what it shows without bound. A
    thing that reaches beside or behind the camera takes all of that."""
    most = np.array([camera.width, camera.height, camera.width, camera.height])     # to the left, above, right, below
    grow = np.zeros(4)
    for thing in things:
        u, v, z = _corners(camera, *_block(thing))
        if (z <= 0).any():
            grow = most
            break
        u, v = camera.fx * u / z + camera.cx, camera.fy * v / z + camera.cy
        grow = np.maximum(grow, [-u.min(), -v.min(), u.max() - camera.width, v.max() - camera.height])
    left, above, right, below = (int(n) for n in np.minimum(np.ceil(grow), most))
    return Camera(camera.fx, camera.fy, camera.cx + left, camera.cy + above, camera.width + left + right,
                  camera.height + above + below, camera.T_world_camera), left, above


def _share_shown(camera: Camera, depth: np.ndarray, things: Sequence[Solid | Figure]) -> float:
    """The share of the things that the frame with this depth image shows.

    The whole is the pixels they fill in a picture of them alone, taken by the same camera with an image
    wide enough to hold them (see _widened). The frame shows those of them that lie in its image and at
    which it has the same depth: nothing stands before them there. 0.0 for nothing at all."""
    if not things:
        return 0.0
    wide, left, above = _widened(camera, things)
    figures = [thing for thing in things if isinstance(thing, Figure)]
    alone = render(wide, [thing for thing in things if not isinstance(thing, Figure)], figures, antialias=False)[1]
    theirs = alone != render(wide, antialias=False)[1]          # where they stand before the bare ground or the sky
    if not theirs.any():
        return 0.0
    inside = np.s_[above:above + camera.height, left:left + camera.width]
    shown = theirs[inside] & (np.abs(depth - alone[inside]) < _SAME_DEPTH_M)
    return float(shown.sum() / theirs.sum())


def _visible(camera: Camera, depth: np.ndarray, target: Target, own: Sequence[Solid | Figure], th: Thresholds) -> bool:
    """Whether the target is visible in the frame with this depth image; `own` is what draws it (see _own).

    It is when the frame shows its anchor by the rule of scene check 5, when it shows lettering on the
    target so that it can be read, or when it shows half of the target or more. The anchor alone would leave
    out a place whose name stands readable over the heads of the people waiting before its counter, or
    along its side, and a person half hidden by another."""
    return (_shows(camera, depth, target.anchor, th.anchor_depth_tolerance_m)
            or any(_reads(camera, depth, part) for part in own if isinstance(part, Box) and part.label)
            or _share_shown(camera, depth, own) >= _TOUR_SHARE)


def _stands_clear(xy, truth: SceneGT, th: Thresholds) -> bool:
    """Whether the ground point is in the room with the tour's clearance from its walls, from every obstacle
    and from every person's body."""
    slack = _TOUR_CLEARANCE_M - 1e-9                # a pose exactly that far away stands clear
    if truth.bounds is not None:
        floor = truth.bounds.polygon
        if not point_in_polygon(xy, floor) or distance_to_boundary(xy, floor) < slack:
            return False
    if any(distance_to_polygon(xy, obstacle.polygon) < slack for obstacle in truth.obstacles):
        return False
    return all(math.hypot(xy[0] - person.x, xy[1] - person.y) >= th.person_radius_m + slack for person in truth.people)


def _tour_pose(T_obs: np.ndarray, robot: RobotPose, centre, turn_deg: int, truth: SceneGT,
               th: Thresholds) -> np.ndarray:
    """T_world_camera of the tour frame that asks for the observation's pose turned by `turn_deg` about `centre`.

    The robot keeps its camera mount and its facing relative to the centre. Of the poses on the bearing asked
    for, the one nearest the observation's distance from the centre is taken at which the robot and its
    camera stand clear, looking first nearer the centre, as someone walking up to it would. A bearing with no
    such pose gives way to the next one towards the observation's, and the observation's own pose stands when
    none is left."""
    out = np.array([robot.x, robot.y]) - centre
    distance = float(np.hypot(*out))
    if distance == 0.0:                              # a robot on the centre has no bearing to turn
        return T_obs
    nearest = min(_TOUR_NEAREST_M, distance)
    radii = [distance - k * _TOUR_STEP_M for k in range(int((distance - nearest) / _TOUR_STEP_M + 1e-9) + 1)]
    back = -1 if turn_deg > 0 else 1
    for degrees in range(turn_deg, 0, back * _TOUR_TURN_STEP_DEG):
        turn = _turn_about(centre, math.radians(degrees))
        along = turn[:2, :2] @ out / distance       # from the centre towards the turned robot
        for radius in radii:
            shift = (radius - distance) * along
            T = turn @ T_obs
            T[:2, 3] += shift
            stand = turn[:2, :2] @ np.array([robot.x, robot.y]) + turn[:2, 3] + shift
            if _stands_clear(stand, truth, th) and _stands_clear(T[:2, 3], truth, th):
                return T
    return T_obs


def _tour_side(robot: RobotPose, centre, truth: SceneGT) -> int:
    """1 or -1: the way round `centre` that the longer arm of the tour takes, anticlockwise or clockwise.

    It is the way that leads from the robot past the front of the first target that has one, the side its
    anchor is seen from, and anticlockwise when no target has a front or the robot stands straight before it."""
    for target in truth.targets:
        front = target.front_yaw
        if front is None and target.kind == "person":
            front = next(person.yaw for person in truth.people if person.id == target.person_id)
        if front is not None:
            off_front = wrap_angle(bearing(centre, (robot.x, robot.y)) - front)
            return -1 if off_front > 1e-6 else 1
    return 1


def _build_tour(layout: Layout, scene: SceneFile, th: Thresholds) -> tuple[SceneFile, list[np.ndarray]]:
    """A copy of the scene with a tour and every target's `tour_frames`, and the tour's images in index order.

    The tour walks round the mean ground position of the target anchors: frame k asks for the observation's
    robot turned about the vertical axis through that point by `_TOUR_TURNS_DEG[k]`, counted the way
    `_tour_side` gives, and `_tour_pose` moves it along that bearing, or failing that to a bearing nearby,
    until it stands in the room and clear of the walls, the obstacles and the people. The frame of the turn
    by 0 is the observation itself, so a target whose anchor passed scene check 5 has that frame. A target's
    `tour_frames` are the frames it is visible in (see _visible), and are empty when there is none. The scene
    passed in is not changed and no random number is drawn.
    """
    obs, targets = scene.observation, scene.ground_truth.targets
    lens, pose = obs.intrinsics, obs.robot_pose_world
    T_obs = np.array(obs.T_world_camera, dtype=np.float64)
    robot = RobotPose(pose.x, pose.y, pose.yaw)
    centre = np.mean([target.anchor[:2] for target in targets], axis=0)
    side = _tour_side(robot, centre, scene.ground_truth)
    entries, frames = [], []
    own = [_own(layout, scene.ground_truth.people, target, th) for target in targets]
    shown_in: list[list[int]] = [[] for _ in targets]
    for index, degrees in enumerate(_TOUR_TURNS_DEG):
        T = T_obs if degrees == 0 else _tour_pose(T_obs, robot, centre, side * degrees, scene.ground_truth, th)
        camera = Camera(lens.fx, lens.fy, lens.cx, lens.cy, obs.width, obs.height, T)
        rgb, depth = _render(layout, camera)
        entries.append(TourFrameModel(rgb=f"tour/{index:03d}.png", T_world_camera=T.tolist()))
        frames.append(rgb)
        for target, things, indices in zip(targets, own, shown_in):
            if _visible(camera, depth, target, things, th):
                indices.append(index)
    toured = [target.model_copy(update={"tour_frames": indices}) for target, indices in zip(targets, shown_in)]
    truth = scene.ground_truth.model_copy(update={"targets": toured})
    return scene.model_copy(update={"tour": entries, "ground_truth": truth}), frames


def _layout(rng, row: tuple) -> Layout:
    """A fresh layout for one content row, under its own random rigid transform."""
    recipe, amount, kind, bystanders, second_group = row
    if recipe == "no_one":                         # "about" the angle asked for, and never beyond it
        inside = rng.uniform(0.0, _VIEW_ANGLE_SPREAD_DEG)
        layout = recipe_no_one(rng, inside if amount == 0 else amount - inside, kind)
    elif recipe == "group_photo":                  # of `amount` people posing; the kind says who the target is
        layout = recipe_group_photo(rng, amount, kind)
    elif recipe == "machine_use":                  # one person, at the kind of machine
        layout = recipe_machine_use(rng, kind)
    else:
        layout = _PEOPLE_RECIPES[recipe](rng, amount)
    layout = _add_extras(layout, rng, bystanders, second_group)
    return transform_layout(layout, rng.uniform(-math.pi, math.pi), rng.uniform(-_SHIFT_M, _SHIFT_M),
                            rng.uniform(-_SHIFT_M, _SHIFT_M))


def _rejection(scene_dir: Path, cfg: Config) -> str | None:
    """Why the scene in the folder is not kept, or None when it is: the first check that is an error or a
    warning other than the expected one, else a nobody-at-target standard goal that cannot be reached."""
    for result in check_scene(scene_dir, cfg, write_png=False):
        if result.level == "error" or (result.level == "warning" and result.number != _EXPECTED_WARNING):
            return f"check_{result.number}"
    inputs, truth = load_scene(scene_dir)
    th, observation = cfg.thresholds, inputs.observation
    robot = observation.robot
    target_ids = dict.fromkeys(target_id for episode in truth.episodes.values() if episode.situation == "no_one"
                               for target_id in episode.target_ids)
    if not target_ids:
        return None
    grid = robot_grid(truth, robot, th)
    for target_id in target_ids:
        target = oracle_target(observation, truth.targets[target_id], th)
        if target is None:
            return _UNREACHABLE
        goal = standard_goal(observation, target, th)
        if not reachable(grid, (robot.x, robot.y), goal.pose_world[:2]):
            return _UNREACHABLE
    return None


def _with_regions(scene_dir: Path, scene: SceneFile, cfg: Config) -> SceneFile:
    """The scene with every episode's regions as sgi.gt derives them from its folder, representative points included."""
    ets = prepare(*load_scene(scene_dir), cfg)
    episodes = []
    for episode in scene.episodes:
        derived = ets[episode.episode_id].gt
        truth = episode.ground_truth.model_copy(update={"acceptable_regions": derived.acceptable_regions,
                                                        "forbidden_regions": derived.forbidden_regions})
        episodes.append(episode.model_copy(update={"ground_truth": truth}))
    return scene.model_copy(update={"episodes": episodes})


def _summary(counts: dict[str, int]) -> str:
    return ", ".join(f"{reason}={count}" for reason, count in counts.items() if count) or "none"


def _accept(split_dir: Path, index: int, row: tuple, rng, cfg: Config,
            counts: dict[str, int]) -> tuple[Layout, SceneFile, np.ndarray, np.ndarray]:
    """The accept step: the first of the row's layouts that is not rejected, with its scene and its rgb and
    depth images, as build_scene gives them. They are left written in the scene's folder under split_dir.

    A rejected layout is counted under its reason and its folder is removed. All the random numbers of a
    row are drawn here, so whoever takes this step row by row with one generator sees the layouts of
    generate()."""
    for _ in range(_MAX_ATTEMPTS):
        layout = _layout(rng, row)
        scene_id = f"{split_dir.name}_{index:02d}_{layout.name}"
        scene_dir = split_dir / scene_id
        scene, rgb, depth = build_scene(layout, scene_id)
        save_scene(scene_dir, scene, rgb, depth)
        reason = _rejection(scene_dir, cfg)
        if reason is None:
            return layout, scene, rgb, depth
        counts[reason] += 1
        shutil.rmtree(scene_dir)
    raise RuntimeError(f"{scene_id}: no layout accepted in {_MAX_ATTEMPTS} attempts; rejected: {_summary(counts)}")


def _write_scene(split_dir: Path, index: int, row: tuple, rng, cfg: Config, counts: dict[str, int]) -> None:
    """Write the row's scene: the first of its layouts that is not rejected, with its regions filled in and a tour.

    A rejected layout is counted under its reason and its folder is removed; no tour is rendered for it."""
    layout, scene, rgb, depth = _accept(split_dir, index, row, rng, cfg, counts)
    scene_dir = split_dir / scene.scene_id
    scene, frames = _build_tour(layout, _with_regions(scene_dir, scene, cfg), cfg.thresholds)
    save_scene(scene_dir, scene, rgb, depth, tour=frames)


def generate(out_dir: Path, cfg: Config, seed: int = 0) -> dict[str, int]:
    """Write the eval and examples splits under out_dir, replacing any that are there: the first pass, and then
    the second into the same two folders.

    Returns the number of rejected layouts by reason: check_1 to check_12, then standard_goal_unreachable."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    counts = dict.fromkeys(_REASONS, 0)
    for split, rows in _SPLITS:
        split_dir = out_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
        for index, row in enumerate(rows):
            _write_scene(split_dir, index, row, rng, cfg, counts)
    more = np.random.default_rng([seed, _MORE_STREAM])         # the second pass draws nothing from the first's generator
    for split, rows in _MORE_SPLITS:
        for index, row in enumerate(rows, start=len(dict(_SPLITS)[split])):
            _write_scene(out_dir / split, index, row, more, cfg, counts)
    log.info("rejected layouts: %s", _summary(counts))
    return counts


def generate_rows(out_dir: Path, cfg: Config, rows: Sequence[tuple], seed: int = 0) -> dict[str, int]:
    """Write the scenes of `rows` straight into out_dir, row k as <out_dir's name>_<k>_<layout name>, each the
    first of its layouts that is not rejected, with its regions and its tour, as generate() writes a scene. They are
    drawn with a random generator of their own, made from the seed and _ON_REQUEST_STREAM, so they are the same for
    the same seed and rows and no pass of generate() draws them.

    A scene folder already in out_dir under the same name prefix, one with a scene.json, is removed first; nothing
    else in out_dir is touched. Returns the number of rejected layouts by reason, as generate() does."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{out_dir.name}_*"):
        if (stale / "scene.json").is_file():
            shutil.rmtree(stale)
    rng = np.random.default_rng([seed, _ON_REQUEST_STREAM])
    counts = dict.fromkeys(_REASONS, 0)
    for index, row in enumerate(rows):
        _write_scene(out_dir, index, row, rng, cfg, counts)
    log.info("rejected layouts: %s", _summary(counts))
    return counts


def accepted_layouts(work_dir: Path, cfg: Config, seed: int = 0) -> Iterator[tuple[str, int, Layout, SceneFile]]:
    """Every accepted layout of both splits in generation order, as (split, index, layout, scene): the layouts
    generate() keeps for the seed, drawn with one random generator as it draws them. The scene has the
    placeholder's id and neither regions nor a tour. Folders are written under work_dir and removed again.

    These are the layouts of the first pass only. The second pass of generate(), the screens, the pictures
    being taken and the lifts, is left out: the generator that reads this renders the layouts by other means
    and has nothing to build a screen, a lift or a camera on a tripod from.

    The scene check reads a scene from its folder, so each layout tried is written to one. They all go into
    a folder of this run's own, new under work_dir, which is made if it is not there: nothing that is in
    work_dir already is read, replaced or removed. A layout's folder is gone before the layout is handed
    out, and the run's folder when the last one is, when the caller stops asking, or on an error."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="accepted_layouts_", dir=work_dir))
    try:
        rng = np.random.default_rng(seed)
        counts = dict.fromkeys(_REASONS, 0)
        for split, rows in _SPLITS:
            split_dir = run_dir / split              # the split's name is the first part of a scene's id
            for index, row in enumerate(rows):
                layout, scene, _, _ = _accept(split_dir, index, row, rng, cfg, counts)
                shutil.rmtree(split_dir / scene.scene_id)
                yield split, index, layout, scene
        log.info("rejected layouts: %s", _summary(counts))
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sgi.synth.generate", description="Write the placeholder scenes.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", choices=sorted(_ON_REQUEST),
                        help="write only this set of rows, straight into --out, in place of the two splits")
    args = parser.parse_args(argv)
    if args.rows is not None:
        rows = _ON_REQUEST[args.rows]
        counts = generate_rows(args.out, load_config(args.config), rows, args.seed)
        print(f"wrote {len(rows)} scenes to {args.out}; rejected layouts: {_summary(counts)}")
        return 0
    counts = generate(args.out, load_config(args.config), args.seed)
    n = sum(len(rows) for _, rows in (*_SPLITS, *_MORE_SPLITS))
    print(f"wrote {n} scenes to {args.out}; rejected layouts: {_summary(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
