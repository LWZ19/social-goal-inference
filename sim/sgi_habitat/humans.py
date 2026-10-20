"""Animated people: Habitat 3.0 humanoids walked by habitat-lab's walking controller along navmesh paths.

A person either wanders (walks to a random place, waits a little, picks another), stands at a place of their
own facing a given point (a barista behind a counter, say), or belongs to a queue. A queue
is a list of slots, the first at the counter; its members stand in the slots in order, the first facing the
counter and each other one the person ahead. Every `service_s` the person at the front is served: they leave,
wander for a while, and come back to the end of the line, and the others step up.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

import habitat_sim
import magnum as mn
import numpy as np
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.articulated_agents.humanoids.kinematic_humanoid import KinematicHumanoid

from sgi_habitat.frames import heading_hab, to_hab, to_world, yaw_of

ARRIVED_M = 0.15                   # a waypoint this near is reached
FACING_RAD = 0.15                  # turned this near to the wanted heading is facing it
YIELD_M = 0.7                      # someone this near, ahead, makes a walker wait
YIELD_CONE = math.radians(50)      # half-angle of "ahead"
MAX_YIELD_S = 2.0                  # a walker waits this long at most, then goes on (no deadlock)


@dataclass
class Slot:
    x: float
    y: float
    face: tuple                    # the world point the person in the slot faces


@dataclass
class Queue:
    slots: List[Slot]
    service_s: Optional[float] = 20.0              # None: nobody is ever served
    members: List["Human"] = field(default_factory=list)      # in line order: the first is served next
    since_served: float = 0.0

    def slot_of(self, human: "Human") -> Optional[Slot]:
        return self.slots[min(self.members.index(human), len(self.slots) - 1)] if human in self.members else None

    def join(self, human: "Human") -> None:
        if human not in self.members:
            self.members.append(human)

    def step(self, dt: float, rng: random.Random) -> None:
        if self.service_s is None:                 # nobody is ever served: the line stands still
            return
        front = self.members[0] if self.members else None
        if front is None or not front.settled:
            self.since_served = 0.0
            return
        self.since_served += dt
        if self.since_served >= self.service_s:
            self.since_served = 0.0
            self.members.pop(0)
            front.leave_queue(rng.uniform(8.0, 15.0))


class Human:
    def __init__(self, sim: habitat_sim.Simulator, name: str, avatar_dir: Path, floor_y: float,
                 lin_speed: float = 0.9, ang_speed: float = 3.0, fps: float = 30.0):
        self.sim, self.name, self.floor_y = sim, name, floor_y
        motion = avatar_dir / f"{avatar_dir.name}_motion_data_smplx.pkl"
        cfg = SimpleNamespace(articulated_agent_urdf=str(avatar_dir / f"{avatar_dir.name}.urdf"),
                              motion_data_path=str(motion), auto_update_sensor_transform=False)
        self.agent = KinematicHumanoid(cfg, sim)
        self.agent.reconfigure()
        self.controller = HumanoidRearrangeController(str(motion))
        self.controller.set_framerate_for_linspeed(lin_speed, ang_speed, fps)
        self.path: List[np.ndarray] = []
        self.face: Optional[tuple] = None          # a world point to turn to once the path is walked
        self.queue: Optional[Queue] = None
        self.home: Optional[Slot] = None           # a place of their own to stand at, facing its point
        self.wander_after_s = 0.0                  # left a queue: wander this long, then join it again
        self.wait_s = 0.0                          # standing before the next wander goal
        self.yielded_s = 0.0
        self.settled = False                       # at the end of its path and facing what it should
        self.wander_area: Optional[tuple] = None   # (x0, x1, y0, y1): where wander goals are taken; None: anywhere

    # --- where it is ------------------------------------------------------------------------------------------
    def place(self, x: float, y: float, yaw: float) -> None:
        p = self.sim.pathfinder.snap_point(to_hab(x, y, self.floor_y))
        self.agent.base_pos = mn.Vector3(p[0], self.floor_y, p[2])
        self.agent.base_rot = yaw
        self.controller.reset(self.agent.base_transformation)
        self.controller.calculate_stop_pose()
        self._apply()

    @property
    def position_hab(self) -> np.ndarray:
        return np.array(self.agent.base_pos)

    @property
    def world_xy(self) -> tuple:
        x, y, _ = to_world(self.position_hab, self.floor_y)
        return x, y

    @property
    def yaw(self) -> float:
        forward = self.agent.base_transformation.transform_vector(mn.Vector3(1, 0, 0))
        return yaw_of(forward)

    # --- what it does -----------------------------------------------------------------------------------------
    def go_to(self, x: float, y: float, face: Optional[tuple] = None) -> bool:
        """Walk to a world position, then turn to the world point `face`; False when no path leads there."""
        path = habitat_sim.ShortestPath()
        path.requested_start = self.position_hab
        path.requested_end = self.sim.pathfinder.snap_point(to_hab(x, y, self.floor_y))
        if not self.sim.pathfinder.find_path(path):
            return False
        self.path = [np.array(p) for p in path.points[1:]]
        self.face = face
        self.settled = False
        return True

    def leave_queue(self, wander_s: float) -> None:
        self.wander_after_s = wander_s
        self.wait_s = 0.0
        self.path, self.face, self.settled = [], None, False

    def step(self, dt: float, others: Sequence[np.ndarray], rng: random.Random, island: int) -> None:
        """Advance one frame: choose what to do next, then take one step of the walking motion."""
        if self.queue is not None and self.wander_after_s > 0:
            self.wander_after_s -= dt
            if self.wander_after_s <= 0:
                self.queue.join(self)
        slot = self.home if self.queue is None else self.queue.slot_of(self)
        if slot is not None:
            here = self.world_xy
            if math.hypot(here[0] - slot.x, here[1] - slot.y) > ARRIVED_M + 0.1 and not self.path:
                self.go_to(slot.x, slot.y, face=slot.face)
            elif self.path and self._goal_moved(slot):
                self.go_to(slot.x, slot.y, face=slot.face)
            elif not self.path:
                self.face = slot.face
        elif not self.path and self.face is None:
            self.wait_s -= dt
            if self.wait_s <= 0:
                self._new_wander_goal(rng, island)
        self._walk(dt, others)

    def _goal_moved(self, slot: Slot) -> bool:
        x, y, _ = to_world(self.path[-1], self.floor_y)
        return math.hypot(x - slot.x, y - slot.y) > 0.3

    def _new_wander_goal(self, rng: random.Random, island: int) -> None:
        for _ in range(200):                       # enough to land in a wander area of a large navmesh
            p = self.sim.pathfinder.get_random_navigable_point(island_index=island)
            x, y, _ = to_world(p, self.floor_y)
            if self.wander_area is not None:
                x0, x1, y0, y1 = self.wander_area
                if not (x0 <= x <= x1 and y0 <= y <= y1):
                    continue
            here = self.world_xy
            if 1.5 < math.hypot(x - here[0], y - here[1]) < 8.0 and self.go_to(x, y):
                self.wait_s = rng.uniform(1.0, 4.0)
                return

    def _blocked(self, others: Sequence[np.ndarray], direction: np.ndarray) -> bool:
        me = self.position_hab
        for other in others:
            offset = (other - me)[[0, 2]]
            distance = float(np.linalg.norm(offset))
            if 1e-3 < distance < YIELD_M:
                cos = float(np.dot(offset, direction[[0, 2]])) / (distance * float(np.linalg.norm(direction[[0, 2]])) + 1e-9)
                if cos > math.cos(YIELD_CONE):
                    return True
        return False

    def _walk(self, dt: float, others: Sequence[np.ndarray]) -> None:
        ctrl = self.controller
        ctrl.obj_transform_base = self.agent.base_transformation
        me = self.position_hab
        while self.path and np.linalg.norm((self.path[0] - me)[[0, 2]]) < ARRIVED_M:
            self.path.pop(0)
        if self.path:
            target = self.path[0] - me
            self.yielded_s = self.yielded_s + dt if self._blocked(others, target) else 0.0
            if 0.0 < self.yielded_s < MAX_YIELD_S:
                ctrl.calculate_stop_pose()
            else:
                ctrl.calculate_walk_pose(mn.Vector3(float(target[0]), 0.0, float(target[2])))
                self._keep_on_navmesh()
        elif self.face is not None:
            x, y, _ = to_world(me, self.floor_y)
            wanted = math.atan2(self.face[1] - y, self.face[0] - x)
            error = (wanted - self.yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(error) > FACING_RAD:
                ctrl.calculate_turn_pose(mn.Vector3(*heading_hab(wanted)))
            else:
                ctrl.calculate_stop_pose()
                self.settled = True
                if self.home is None and (self.queue is None or self.queue.slot_of(self) is None):
                    self.face = None
        else:
            ctrl.calculate_stop_pose()
        self._apply()

    def _keep_on_navmesh(self) -> None:
        ctrl = self.controller
        before = self.agent.base_pos
        wanted = ctrl.obj_transform_base.translation + self.agent.params.base_offset
        allowed = self.sim.pathfinder.try_step(before, wanted)
        ctrl.obj_transform_base.translation += mn.Vector3(allowed) - wanted
        # The navmesh lies a little above the floor: the feet are kept on the floor, which is taken to be flat.
        t = ctrl.obj_transform_base.translation
        ctrl.obj_transform_base.translation = mn.Vector3(t[0], self.floor_y - self.agent.params.base_offset[1], t[2])

    def _apply(self) -> None:
        ctrl = self.controller
        self.agent.set_joint_transform(ctrl.joint_pose, ctrl.obj_transform_offset, ctrl.obj_transform_base)


def avatar_dirs(root: Path) -> List[Path]:
    """The humanoid folders under `root`, each with its URDF and its motion data, in name order."""
    return sorted(p for p in root.iterdir() if (p / f"{p.name}.urdf").is_file()
                  and (p / f"{p.name}_motion_data_smplx.pkl").is_file())
