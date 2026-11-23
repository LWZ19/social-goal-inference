"""What the solver's tests share: pictures of a floor, and the people on it.

Two kinds of picture. `mounted` is taken by the camera of the generated scenes on its robot and rendered, so
bodies and walls hide the floor behind them. `open_floor` is a bare floor seen from straight above the robot,
every cell within range in view: what is laid out on it is read from positions alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sgi import solver
from sgi.config import Thresholds
from sgi.geometry import Camera, RobotPose, camera_on_robot
from sgi.inputs import ObservationData
from sgi.synth.render import render
from sgi.types import Constraint, DetectedPerson, DistanceRange, GoalResult, GroundedTarget, Point

TH = Thresholds()
LENS = dict(fx=554.3, fy=554.3, cx=320.0, cy=240.0, width=640, height=480)
MOUNT = dict(forward_m=0.2, height_m=1.5, pitch_down_rad=math.radians(10.0))      # as the generated scenes have it
HERE = RobotPose(0.0, 0.0, 0.0)                    # at the origin, heading along +x: its frame is the world's

# Where the mounted camera's picture ends on a level floor, worked out from the mount alone. The lower edge is
# atan(240 / 554.3) under the optical axis, so its rays leave the camera 33.41 degrees under the horizon.
_EDGE_DROP = MOUNT["pitch_down_rad"] + math.atan((LENS["height"] - LENS["cy"]) / LENS["fy"])
EDGE_AHEAD_M = MOUNT["forward_m"] + MOUNT["height_m"] / math.tan(_EDGE_DROP)       # 2.474 m ahead of the robot
_EDGE_DEPTH = ((EDGE_AHEAD_M - MOUNT["forward_m"]) * math.cos(MOUNT["pitch_down_rad"])
               + MOUNT["height_m"] * math.sin(MOUNT["pitch_down_rad"]))             # camera Z of that floor: 2.500 m
EDGE_HALF_WIDTH_M = LENS["cx"] / LENS["fx"] * _EDGE_DEPTH                           # and 1.443 m to either side


def mounted_camera(robot: RobotPose = HERE) -> Camera:
    return Camera(**LENS, T_world_camera=camera_on_robot(robot, **MOUNT))


def mounted(robot: RobotPose = HERE, boxes=(), figures=()) -> ObservationData:
    """What the mounted camera shows of a level floor with these things and bodies on it."""
    camera = mounted_camera(robot)
    rgb, depth = render(camera, boxes, figures)
    return ObservationData(rgb=rgb, depth=depth, camera=camera, robot=robot)


def person(number: int, x: float, y: float, yaw: float | None = None) -> DetectedPerson:
    """A detected person. The solver does not read `pixel`."""
    return DetectedPerson(number=number, xy=(x, y), yaw=yaw, pixel=(0.0, 0.0))


def moved(pose: RobotPose, turn: float, shift) -> RobotPose:
    """A pose of the robot's frame of `HERE` as it lies after the world is turned by `turn` and then shifted."""
    c, s = math.cos(turn), math.sin(turn)
    return RobotPose(c * pose.x - s * pose.y + shift[0], s * pose.x + c * pose.y + shift[1], pose.yaw + turn)


def moved_xy(xy, turn: float, shift) -> tuple[float, float]:
    c, s = math.cos(turn), math.sin(turn)
    return (c * xy[0] - s * xy[1] + shift[0], s * xy[0] + c * xy[1] + shift[1])


def cells(mask: np.ndarray, floor, where) -> np.ndarray:
    """The values of a mask at the cells whose centres, in the frame of `HERE`, satisfy `where(x, y)`."""
    x, y = np.meshgrid(floor.offsets, floor.offsets)
    return mask[where(x, y)]


# ------------------------------------------------------------------------------------------------- the open floor
ABOVE_M = 20.0                                     # how high over the robot the camera of `open_floor` hangs
_ABOVE = dict(fx=400.0, fy=400.0, cx=250.5, cy=250.5, width=501, height=501)       # a pixel shows one cell of 0.05 m


def open_floor(robot: RobotPose = HERE, floor=None) -> ObservationData:
    """A level floor seen from straight above the robot, in view out to the solver's range and beyond.

    The camera looks down with the picture's upper edge ahead of the robot, and each cell centre of the solver's
    grid is the centre of a pixel. With `floor`, a function of positions in the robot's frame (metres ahead and
    to its left), there is floor only where it holds, and a drop of the camera's height elsewhere.
    """
    camera = Camera(**_ABOVE, T_world_camera=camera_on_robot(robot, 0.0, ABOVE_M, math.pi / 2))
    side = _ABOVE["width"]
    depth = np.full((side, side), ABOVE_M, dtype=np.float32)
    if floor is not None:
        ahead, left = np.meshgrid((side // 2 - np.arange(side)) * 0.05, (side // 2 - np.arange(side)) * 0.05, indexing="ij")
        depth[~floor(ahead, left)] = 2 * ABOVE_M
    return ObservationData(np.full((side, side, 3), 128, dtype=np.uint8), depth, camera, robot)


def looking(from_xy, at_xy) -> RobotPose:
    """The robot at one position, heading for another."""
    return RobotPose(from_xy[0], from_xy[1], math.atan2(at_xy[1] - from_xy[1], at_xy[0] - from_xy[0]))


def facing(xy, at_xy) -> float:
    """The heading of someone at `xy` who looks at `at_xy`."""
    return math.atan2(at_xy[1] - xy[1], at_xy[0] - xy[0])


def numbered(spots) -> list[DetectedPerson]:
    """People numbered from 1 in the order given, each (x, y) or (x, y, heading)."""
    return [person(n, *spot) for n, spot in enumerate(spots, 1)]


def target_at(x: float, y: float, z: float = 0.0) -> GroundedTarget:
    """A target whose point lifts to (x, y, z). The solver reads only `world`."""
    return GroundedTarget(point=Point(x=500, y=500), pixel=(0.0, 0.0), world=(x, y, z))


def standard_at(x: float, y: float) -> GoalResult:
    """A standard goal that stands at (x, y). The solver reads only where it stands."""
    return GoalResult(status="ok", source="standard", pose_world=(x, y, 0.0))


def body_point(xy, camera_xy) -> tuple[float, float, float]:
    """The point a model marks on someone who stands at `xy`: on the side of the body the camera sees."""
    away = math.hypot(camera_xy[0] - xy[0], camera_xy[1] - xy[1])
    return (xy[0] + 0.2 * (camera_xy[0] - xy[0]) / away, xy[1] + 0.2 * (camera_xy[1] - xy[1]) / away, TH.person_mark_height_m)


def ask(kind: str, *people: int, target: bool = False, between=None) -> Constraint:
    """One constraint of an answer. `between` is (at least, at most) in metres."""
    range_m = None if between is None else DistanceRange(at_least=between[0], at_most=between[1])
    return Constraint(because="what the people do", kind=kind, people=list(people), range_m=range_m, target=target)


@dataclass(frozen=True)
class Layout:
    """People on an open floor, what the robot was sent to, and where the standard goal stands."""
    observation: ObservationData
    people: list[DetectedPerson]
    target: GroundedTarget
    standard: GoalResult

    def solve(self, *constraints: Constraint, th: Thresholds = TH) -> solver.Solved:
        return solver.solve(self.observation, self.target, self.people, list(constraints), self.standard, th)

    def place(self, *constraints: Constraint, th: Thresholds = TH) -> tuple[float, float]:
        return self.solve(*constraints, th=th).pose_world[:2]

    def worked(self, *constraints: Constraint, th: Thresholds = TH):
        """The selectable cells in the robot's frame, and what the solver works out over them."""
        return solver._over_floor(self.observation, self.target, self.people, list(constraints), self.standard, th)


def layout(robot: RobotPose, spots, thing=None, needed: int | None = None, standard=None, floor=None) -> Layout:
    """People at `spots` on an open floor. The target is the thing at the ground position `thing`, or the person
    numbered `needed`. The standard goal stands where `standard` says, or `standoff_m` short of the target's
    point on the line from it to the robot, where the first spec has it when nothing is in the way."""
    people = numbered(spots)
    target = target_at(*thing) if needed is None else target_at(*body_point(people[needed - 1].xy, (robot.x, robot.y)))
    if standard is None:
        gap = math.hypot(robot.x - target.world[0], robot.y - target.world[1])
        standard = (target.world[0] + TH.standoff_m * (robot.x - target.world[0]) / gap,
                    target.world[1] + TH.standoff_m * (robot.y - target.world[1]) / gap)
    return Layout(open_floor(robot, floor), people, target, standard_at(*standard))


def at_points(points, spots, constraints, target, robot=(0.0, 0.0), standard=None, th: Thresholds = TH, front=None):
    """What the solver works out at hand-picked positions in place of a floor's cells: people at `spots`, the
    target's point at `target`, the robot and its camera at `robot`. The standard goal stands on the target's
    point unless `standard` says where, and the target point lies on no seen face unless `front` gives its front."""
    bodies = [solver._Body(p.number, np.array(p.xy), p.yaw) for p in numbered(spots)]
    return solver._settle(np.array(points, dtype=np.float64), robot, robot, target, bodies, list(constraints),
                          target if standard is None else standard, th,
                          front=None if front is None else np.array(front, dtype=np.float64))
