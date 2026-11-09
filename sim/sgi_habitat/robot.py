"""Spot, moved kinematically: it turns towards the next waypoint of a navmesh path, walks to it, and at the end
turns to the goal's heading. The base is placed directly, as in Habitat 3.0's social navigation task, and the
legs play a trot over it: diagonal pairs swing in turn, as far as the base moved or turned, and settle back
into the standing pose when it stops.

Paths are planned on the robot's own navmesh, cut back by its clearance from walls and furniture
(world.robot_pathfinder). Twice a second, while anyone is near, the path to the goal is planned again on a grid
with each person stamped on it as a disc (planner.GridPlanner), so the robot goes round people as they move.

Getting past people who stand still:
- The path keeps a comfortable distance from everyone (`person_m`). Where no such path is open, it is planned
  again keeping only a squeeze distance (SQUEEZE_M), so the robot can pass through a gap between two bodies.
- A way round people is taken only if it is not a long loop: at most DETOUR_RATIO times the straight distance,
  or DETOUR_EXTRA_M more. Otherwise the robot waits for them, as one does in a line, instead of circling a table.
- The robot stops only for a person in its way: ahead of it, within STOP_M, and within a body's width of the
  direction it walks. A person beside the path does not stop it.
- Stopped for BACKOFF_S with no way round, it backs off BACKOFF_M (never into someone behind it) and plans
  again from there; people who stand still no longer hold it in place for good.

A goal less than SHORT_M away on a clear straight line is stepped to directly, sideways or backwards as needed,
while the robot turns to the goal's heading: moving up a line is a shuffle, not a turn, a walk and a turn back."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

import habitat_sim
import magnum as mn
import numpy as np
from habitat.articulated_agents.robots.spot_robot import SpotRobot

from sgi_habitat.frames import heading_hab, to_hab, to_world, wrap, yaw_of
from sgi_habitat.planner import GridPlanner
from sgi_habitat.world import place_agent

ARRIVED_M = 0.10                   # a waypoint this near is reached
GOAL_RAD = 0.05                    # turned this near to the goal heading is done
TURN_FIRST_RAD = 0.35              # a waypoint further off the heading than this is turned to before walking
STOP_M = 0.65                      # a person this near, ahead, stops the robot
LOCAL_REPLAN_S = 0.5               # round people: plan again this often
NEAR_M = 3.0                       # people this near the robot or its path are planned round
PERSON_M = 0.25                    # a person's radius
BODY_HALF_M = 0.30                 # half the robot's width
IN_WAY_M = PERSON_M + BODY_HALF_M  # a person this near the line the robot walks along is in its way
SQUEEZE_M = IN_WAY_M + 0.05        # the least distance kept from a person's centre when no wider way is open
BACKOFF_S = 3.0                    # stopped this long with no way round: back off
BACKOFF_M = 0.5                    # this far
BACK_SPEED = 0.3                   # m/s
DETOUR_RATIO, DETOUR_EXTRA_M = 1.5, 1.0   # a way round people longer than both of these over straight is not taken
SHORT_M = 1.0                      # a goal nearer than this is stepped to straight, without turning to face the way
SHORT_SPEED = 0.4                  # m/s, stepping sideways or backwards
STRIDE_M = 0.45                    # base travel per gait cycle
TURN_STRIDE_RAD = 1.2              # turning in place per gait cycle
STAND = (0.0, 0.7, -1.5)           # hip roll, hip pitch, knee of each leg standing (SpotRobot's leg_init_params)
SWING_PITCH, LIFT_KNEE = 0.25, 0.55
SETTLE_S = 0.25                    # time to ease back into the standing pose


def _length(start: np.ndarray, path: Sequence[np.ndarray]) -> float:
    """The length of a path on the floor, from `start` through its waypoints."""
    points = [start] + list(path)
    return float(sum(np.linalg.norm((b - a)[[0, 2]]) for a, b in zip(points, points[1:])))


@dataclass
class Goal:
    x: float
    y: float
    yaw: float


class Spot:
    def __init__(self, sim: habitat_sim.Simulator, urdf: Path, floor_y: float, camera_forward_m: float, pathfinder,
                 clearance_m: float, lin_speed: float = 0.6, ang_speed: float = 1.2):
        self.sim, self.floor_y = sim, floor_y
        self.pathfinder = pathfinder               # the robot's own navmesh, cut back by its clearance
        self.planner = GridPlanner(pathfinder, floor_y)
        self.person_m = PERSON_M + clearance_m + 0.1          # how far the robot's centre keeps from a person's
        self.end: Optional[np.ndarray] = None      # the goal as the navmesh lets the robot stand there
        self.since_local = 0.0
        self.camera_forward_m = camera_forward_m
        self.lin_speed, self.ang_speed = lin_speed, ang_speed
        self.agent = SpotRobot(SimpleNamespace(articulated_agent_urdf=str(urdf)), sim, fixed_base=True)
        self.agent.reconfigure()
        self.agent.sim_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        self.agent.arm_joint_pos = self.agent.params.arm_init_params       # the arm folded down on the back
        self.agent.gripper_joint_pos = self.agent.params.gripper_init_params
        self.agent.update()
        self.goal: Optional[Goal] = None
        self.path: List[np.ndarray] = []
        self.stopped_s = 0.0
        self.backoff_m = 0.0                       # still to back off, after being stopped with no way round
        self.state = "idle"                        # idle | moving | waiting | arrived | unreachable
        self.phase = 0.0                           # of the trot, in radians
        self.stride = 0.0                          # 0 standing .. 1 walking: the gait's amplitude

    # --- pose -------------------------------------------------------------------------------------------------
    def place(self, x: float, y: float, yaw: float) -> None:
        p = self.pathfinder.snap_point(to_hab(x, y, self.floor_y))
        self._set(np.array([p[0], self.floor_y, p[2]]), yaw)

    def _set(self, position_hab: np.ndarray, yaw: float) -> None:
        self.agent.base_pos = mn.Vector3(*position_hab)
        self.agent.base_rot = yaw
        # The camera rides on agent 0: its body at the camera's foot, turned to look where the robot looks.
        mount = position_hab + self.camera_forward_m * heading_hab(yaw)
        place_agent(self.sim, 0, mount, yaw - math.pi / 2)

    @property
    def position_hab(self) -> np.ndarray:
        p = self.agent.base_pos
        return np.array([p[0], self.floor_y, p[2]])

    @property
    def yaw(self) -> float:
        return yaw_of(self.agent.base_transformation.transform_vector(mn.Vector3(1, 0, 0)))

    @property
    def world_pose(self) -> tuple:
        x, y, _ = to_world(self.position_hab, self.floor_y)
        return x, y, self.yaw

    # --- motion -----------------------------------------------------------------------------------------------
    def set_goal(self, goal: Goal) -> Optional[float]:
        """Plan to a goal; the distance from the goal to where the navmesh lets the robot stand, or None when no
        path leads there."""
        self.goal = goal
        end = self.pathfinder.snap_point(to_hab(goal.x, goal.y, self.floor_y))
        self.end = np.array([end[0], self.floor_y, end[2]])
        self.since_local = LOCAL_REPLAN_S          # plan round people on the next step
        if not self._plan(end):
            self.state = "unreachable"
            return None
        self.state = "moving"
        ex, ey, _ = to_world(end, self.floor_y)
        return math.hypot(ex - goal.x, ey - goal.y)

    def _plan(self, end: np.ndarray) -> bool:
        path = habitat_sim.ShortestPath()
        path.requested_start = self.pathfinder.snap_point(self.position_hab)
        path.requested_end = end
        if not self.pathfinder.find_path(path):
            return False
        self.path = [np.array([p[0], self.floor_y, p[2]]) for p in path.points[1:]]
        self.stopped_s = 0.0
        self.backoff_m = 0.0
        return True

    def cancel(self) -> None:
        self.goal, self.end, self.path, self.state = None, None, [], "idle"
        self.backoff_m = 0.0

    def _round_people(self, people: Sequence[np.ndarray], now: bool = False) -> bool:
        """Plan the rest of the way again round the people near the robot or its path, every LOCAL_REPLAN_S or
        `now`: keeping `person_m` from them where that is open, else only SQUEEZE_M. Whether a way round is open;
        where none is, the path is kept."""
        if self.end is None or not (now or self.since_local >= LOCAL_REPLAN_S):
            return True
        self.since_local = 0.0
        me = self.position_hab
        route = [me] + self.path
        near = [p for p in people if min(np.linalg.norm((p - q)[[0, 2]]) for q in route) < NEAR_M]
        if not near:
            return True
        straight = float(np.linalg.norm((self.end - me)[[0, 2]]))
        for margin in (self.person_m, SQUEEZE_M):
            path = self.planner.plan(me, self.end, near, margin)
            if path is not None and _length(me, path) <= max(DETOUR_RATIO * straight, straight + DETOUR_EXTRA_M):
                self.path = path
                return True
        return False                                   # no way round that is not a long loop: wait for them

    def _person_in_way(self, people: Sequence[np.ndarray], backwards: bool = False,
                       direction: Optional[np.ndarray] = None) -> bool:
        """Whether someone stands in the robot's way: ahead of it (behind it, `backwards`; along `direction`, a
        unit vector in Habitat's x-z plane, when given), within STOP_M, and within IN_WAY_M of the line it walks
        along."""
        me = self.position_hab
        forward = (np.asarray(direction, dtype=float) if direction is not None
                   else heading_hab(self.yaw)[[0, 2]] * (-1.0 if backwards else 1.0))
        for person in people:
            offset = (person - me)[[0, 2]]
            along = float(np.dot(offset, forward))
            across = abs(float(offset[0] * forward[1] - offset[1] * forward[0]))
            if 0.0 < along and float(np.linalg.norm(offset)) < STOP_M and across < IN_WAY_M:
                return True
        return False

    def _step_short(self, dt: float, people: Sequence[np.ndarray], me: np.ndarray, yaw: float) -> bool:
        """A goal less than SHORT_M away, on a clear straight line: step straight to it, sideways or backwards as
        needed, turning to the goal's heading on the way, as a legged robot shuffles up a line. Whether it did;
        within ARRIVED_M the path is dropped, and the robot only turns to the goal's heading."""
        if self.goal is None or self.end is None:
            return False
        offset = (self.end - me)[[0, 2]]
        distance = float(np.linalg.norm(offset))
        if distance < ARRIVED_M:
            self.path = []
            return False
        if distance >= SHORT_M:
            return False
        direction = offset / distance
        if self._person_in_way(people, direction=direction):
            return False                               # the usual way: plan round them, or wait
        step = min(distance, SHORT_SPEED * dt)
        target = me + step * np.array([direction[0], 0.0, direction[1]])
        allowed = np.array(self.pathfinder.try_step(mn.Vector3(*me), mn.Vector3(*target)))
        allowed = np.array([allowed[0], self.floor_y, allowed[2]])
        if float(np.linalg.norm((allowed - me)[[0, 2]])) < 1e-4:
            return False                               # a wall: the usual way
        error = wrap(self.goal.yaw - yaw)
        self._set(allowed, yaw + math.copysign(min(abs(error), self.ang_speed * dt), error))
        self.state, self.stopped_s = "moving", 0.0
        return True

    def _back_off(self, dt: float, people: Sequence[np.ndarray]) -> None:
        """One step straight back, facing the same way; at the end, or when it cannot go on, plan again."""
        me, yaw = self.position_hab, self.yaw
        step = min(self.backoff_m, BACK_SPEED * dt)
        moved = 0.0
        if not self._person_in_way(people, backwards=True):
            target = me - step * heading_hab(yaw)
            allowed = np.array(self.pathfinder.try_step(mn.Vector3(*me), mn.Vector3(*target)))
            allowed = np.array([allowed[0], self.floor_y, allowed[2]])
            moved = float(np.linalg.norm((allowed - me)[[0, 2]]))
            self._set(allowed, yaw)
        self.backoff_m = self.backoff_m - step if moved > 1e-4 else 0.0
        if self.backoff_m <= 1e-6:
            self.backoff_m = 0.0
            if not self._round_people(people, now=True) and self.end is not None:
                self._plan(self.end)               # no way round from here either: the navmesh path, to try again

    def _legs(self, dt: float, moved_m: float, turned_rad: float) -> None:
        """One frame of the trot. Legs are in SpotRobot's order (front left, front right, hind left, hind
        right); front left and hind right swing together, then the other pair."""
        walking = moved_m > 1e-4 or turned_rad > 1e-4
        self.phase += 2 * math.pi * (moved_m / STRIDE_M + turned_rad / TURN_STRIDE_RAD)
        self.stride = min(1.0, self.stride + dt / SETTLE_S) if walking else max(0.0, self.stride - dt / SETTLE_S)
        legs = []
        for offset in (0.0, math.pi, math.pi, 0.0):
            s = math.sin(self.phase + offset)
            hx, hy, kn = STAND
            legs += [hx, hy + self.stride * SWING_PITCH * math.cos(self.phase + offset),
                     kn - self.stride * LIFT_KNEE * max(0.0, s)]
        self.agent.leg_joint_pos = legs

    def step(self, dt: float, people: Sequence[np.ndarray]) -> None:
        before, before_yaw = self.position_hab, self.yaw
        self._move(dt, people)
        moved = float(np.linalg.norm((self.position_hab - before)[[0, 2]]))
        self._legs(dt, moved, abs(wrap(self.yaw - before_yaw)))

    def _move(self, dt: float, people: Sequence[np.ndarray]) -> None:
        if self.state not in ("moving", "waiting"):
            return
        self.since_local += dt
        if self.backoff_m > 0.0:
            self.state = "waiting"
            self._back_off(dt, people)
            return
        self._round_people(people)
        me, yaw = self.position_hab, self.yaw
        if self._step_short(dt, people, me, yaw):
            return
        while self.path and np.linalg.norm((self.path[0] - me)[[0, 2]]) < ARRIVED_M:
            self.path.pop(0)
        if self.path:
            offset = self.path[0] - me
            wanted = yaw_of(offset)
            error = wrap(wanted - yaw)
            if abs(error) > TURN_FIRST_RAD:
                self._set(me, yaw + math.copysign(min(abs(error), self.ang_speed * dt), error))
                return
            if self._person_in_way(people):
                self.state = "waiting"
                self.stopped_s += dt
                self._round_people(people, now=self.stopped_s > LOCAL_REPLAN_S)   # a way round: taken next step
                if self.stopped_s > BACKOFF_S:     # however the planning went, still stopped: back off
                    self.stopped_s, self.backoff_m = 0.0, BACKOFF_M
                return
            self.state, self.stopped_s = "moving", 0.0
            distance = float(np.linalg.norm(offset[[0, 2]]))
            step = min(distance, self.lin_speed * dt)
            target = me + step * heading_hab(yaw + error)
            allowed = np.array(self.pathfinder.try_step(mn.Vector3(*me), mn.Vector3(*target)))
            turn = math.copysign(min(abs(error), self.ang_speed * dt), error)
            self._set(np.array([allowed[0], self.floor_y, allowed[2]]), yaw + turn)
            return
        error = wrap(self.goal.yaw - yaw)
        if abs(error) > GOAL_RAD:
            self._set(me, yaw + math.copysign(min(abs(error), self.ang_speed * dt), error))
        else:
            self.state = "arrived"
