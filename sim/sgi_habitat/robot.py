"""Spot, moved kinematically: it turns towards the next waypoint of a navmesh path, walks to it, and at the end
turns to the goal's heading. The base is placed directly, as in Habitat 3.0's social navigation task, and the
legs play a trot over it: diagonal pairs swing in turn, as far as the base moved or turned, and settle back
into the standing pose when it stops.

Paths are planned on the robot's own navmesh, cut back by its clearance from walls and furniture
(world.robot_pathfinder). Twice a second, while anyone is near, the path to the goal is planned again on a grid
with each person stamped on it as a disc (planner.GridPlanner), so the robot goes round people as they move. A
person close ahead still makes it wait; when no way round them is open, it waits until they move."""
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
STOP_CONE = math.radians(40)
REPLAN_S = 3.0                     # stopped this long: plan again on the navmesh
LOCAL_REPLAN_S = 0.5               # round people: plan again this often
NEAR_M = 3.0                       # people this near the robot or its path are planned round
PERSON_M = 0.25                    # a person's radius
STRIDE_M = 0.45                    # base travel per gait cycle
TURN_STRIDE_RAD = 1.2              # turning in place per gait cycle
STAND = (0.0, 0.7, -1.5)           # hip roll, hip pitch, knee of each leg standing (SpotRobot's leg_init_params)
SWING_PITCH, LIFT_KNEE = 0.25, 0.55
SETTLE_S = 0.25                    # time to ease back into the standing pose


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
        return True

    def cancel(self) -> None:
        self.goal, self.end, self.path, self.state = None, None, [], "idle"

    def _round_people(self, people: Sequence[np.ndarray], now: bool = False) -> None:
        """Plan the rest of the way again round the people near the robot or its path, every LOCAL_REPLAN_S or
        `now`. Where no way round them is open, the path is kept, and the robot waits for them to move."""
        if self.end is None or not (now or self.since_local >= LOCAL_REPLAN_S):
            return
        self.since_local = 0.0
        me = self.position_hab
        route = [me] + self.path
        near = [p for p in people if min(np.linalg.norm((p - q)[[0, 2]]) for q in route) < NEAR_M]
        if not near:
            return
        path = self.planner.plan(me, self.end, near, self.person_m)
        if path is not None:
            self.path = path

    def _person_ahead(self, people: Sequence[np.ndarray]) -> bool:
        me, forward = self.position_hab, heading_hab(self.yaw)
        for person in people:
            offset = (person - me)[[0, 2]]
            distance = float(np.linalg.norm(offset))
            if distance < STOP_M and float(np.dot(offset, forward[[0, 2]])) > distance * math.cos(STOP_CONE):
                return True
        return False

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
        self._round_people(people)
        me, yaw = self.position_hab, self.yaw
        while self.path and np.linalg.norm((self.path[0] - me)[[0, 2]]) < ARRIVED_M:
            self.path.pop(0)
        if self.path:
            offset = self.path[0] - me
            wanted = yaw_of(offset)
            error = wrap(wanted - yaw)
            if abs(error) > TURN_FIRST_RAD:
                self._set(me, yaw + math.copysign(min(abs(error), self.ang_speed * dt), error))
                return
            if self._person_ahead(people):
                self.state = "waiting"
                self.stopped_s += dt
                self._round_people(people, now=self.stopped_s > LOCAL_REPLAN_S)
                if self.stopped_s > REPLAN_S and self.end is not None:
                    self._plan(self.end)
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
