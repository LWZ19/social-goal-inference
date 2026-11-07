"""Spot among moving people in Habitat, sent where the social goal inference pipeline says.

    python -m sgi_habitat.run [--scenario FILE] [--sgi-url URL] [--request TEXT [--request-at S]]
                              [--context TEXT] [--goal X Y YAW] [--headless] [--record FILE] [--seconds S]
                              [--exit-on-arrival] [--out DIR]

The pipeline runs in its own environment as `python -m sgi.serve` (docs/habitat.md). Type a request in this
terminal at any time, such as "get me a cup from the kitchen counter": the robot's current RGB-D view is sent
with it, the people keep moving while the models answer, and the robot then walks to the goal pose that comes
back. In the window: q quits, r sends the last request again, c cancels the robot's goal, p pauses, and i shows
or hides what the pipeline made of the last request: the image it was sent with the target, the people it
found (numbered as the model saw them) and the goal drawn on it, beside each stage's reasoning as text.

A click also sends the robot somewhere, along a path planned on the navmesh: a click on the map goes to that
place (press and drag to choose the heading it ends in, else it faces the way it went), and a click on the
robot's image goes to the point the depth there shows, or the nearest place the robot can stand by it.

Every request is kept under --out (runs/habitat/<time>/NNN/): the image and depth that were sent, the camera,
the answer, and the watching camera's view when the answer came.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import habitat_sim
import magnum as mn
import numpy as np
import yaml

from sgi_habitat.client import Observation, SgiClient, SgiError, make_observation
from sgi_habitat.frames import heading_hab, to_hab
from sgi_habitat.humans import Human, Queue, Slot
from sgi_habitat.inference import annotate, describe, text_panel
from sgi_habitat.robot import Goal, Spot
from sgi_habitat.view import Map, compose
from sgi_habitat.world import (ROBOT_DEPTH, ROBOT_RGB, VIEW_RGB, CameraSpec, PlacedObject, WorldSpec, floor_height, make_sim,
                               place_agent, robot_pathfinder, sensor_transform)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO = REPO / "sim" / "scenarios" / "apt_0_kitchen.yaml"
ROBOT_CLEARANCE_M = 0.45                                # Spot's centre keeps this far from walls and furniture
HUMANOIDS = "data/habitat/humanoids/humanoid_data"      # where the download of docs/habitat.md puts them
SPOT_URDF = "data/habitat/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
FPS = 30.0
GOAL_MOVED_M, GOAL_TURNED_DEG = 0.3, 25.0      # an answer asked again replaces the goal past these
ARRIVAL_CHECKS = 2                             # asked again on arrival at most this often for one request
WINDOW = "sgi in habitat"
VIEW_PLACES = ((2.8, 2.3), (2.0, 3.0), (1.2, 3.6), (0.4, 4.2))    # (behind, above) the robot for the watching camera


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _largest_island(pathfinder) -> int:
    return max(range(pathfinder.num_islands), key=pathfinder.island_area)


class Demo:
    def __init__(self, scenario: dict, client: SgiClient, out: Path, context: Optional[str], seed: int = 0,
                 requery_s: float = 0.0):
        self.client, self.out, self.context = client, out, context
        self.rng = random.Random(seed)
        self.camera = CameraSpec(**scenario.get("camera", {}))
        if "stage" in scenario:                    # a .glb of its own, from Blender or elsewhere
            world = WorldSpec(None, str(_repo_path(scenario["stage"])), self.camera)
        else:                                      # a scene of a Habitat scene dataset, such as ReplicaCAD's
            world = WorldSpec(_repo_path(scenario["dataset"]), scenario["scene"], self.camera)
        objects = [PlacedObject(model=_repo_path(o["model"]), **{k: v for k, v in o.items() if k != "model"})
                   for o in scenario.get("objects", [])]
        self.sim = make_sim(world, objects)
        self.floor_y = floor_height(self.sim)
        self.island = _largest_island(self.sim.pathfinder)

        r = scenario["robot"]
        clearance = float(r.get("clearance_m", ROBOT_CLEARANCE_M))
        self.robot = Spot(self.sim, _repo_path(scenario.get("spot_urdf", SPOT_URDF)), self.floor_y, self.camera.forward_m,
                          robot_pathfinder(self.sim, clearance), clearance)
        face = r.get("face")
        self.robot.place(r["x"], r["y"], math.atan2(face[1] - r["y"], face[0] - r["x"]) if face else r.get("yaw", 0.0))

        self.queues = {}
        for name, q in scenario.get("queues", {}).items():
            slots = []
            for i, s in enumerate(q["slots"]):
                ahead = q["slots"][i - 1] if i else None
                face = s.get("face") or ([ahead["x"], ahead["y"]] if ahead else [s["x"] - 1.0, s["y"]])
                slots.append(Slot(s["x"], s["y"], tuple(face)))
            service = q.get("service_s", 20.0)
            if isinstance(service, (list, tuple)):      # [min_s, max_s]: every service takes a time drawn from it
                lo, hi = float(service[0]), float(service[1])
                self.queues[name] = Queue(slots, service_s=self.rng.uniform(lo, hi), service_range=(lo, hi))
            else:
                self.queues[name] = Queue(slots, service_s=None if service is None else float(service))
            if q.get("step_lag") is not None:          # [min_s, max_s]: members wait this long before stepping up
                self.queues[name].step_lag = (float(q["step_lag"][0]), float(q["step_lag"][1]))
            self.queues[name].lazy_share = float(q.get("lazy_share", 0.0))   # of those, wait for two free places

        root = _repo_path(scenario.get("humanoids", HUMANOIDS))
        self.humans: List[Human] = []
        for i, h in enumerate(scenario.get("humans", [])):
            # `z` raises a person standing still onto something, such as a stage
            human = Human(self.sim, f"{h['avatar']}#{i}", root / h["avatar"], self.floor_y + float(h.get("z", 0.0)))
            area = h.get("wander_area", scenario.get("wander_area"))
            human.wander_area = None if area is None else tuple(area)
            human.turn_rad = math.radians(float(h.get("turn_deg", 0.0)))   # faces this far off its point
            line = self.queues.get(h.get("queue")) if h.get("queue") else None
            if h.get("stand") is not None:     # stays at a place of their own, facing `face`
                x, y = h["stand"]
                face = h.get("face", [x, y - 1.0])
                human.home = human.anchor = Slot(x, y, tuple(face))
                human.place(x, y, math.atan2(face[1] - y, face[0] - x) + math.radians(float(h.get("turn_deg", 0.0))))
                if h.get("fidget") is not None:     # [min_s, max_s, radius_m]: shifts about the place now and then
                    human.fidget = tuple(float(v) for v in h["fidget"])
                    human.fidget_s = self.rng.uniform(human.fidget[0], human.fidget[1])
                if h.get("excursion") is not None:  # once: walks part way to `toward` and back (at, dist, hold drawn)
                    ex = h["excursion"]
                    human.excursion = {"at": self.rng.uniform(*ex["at"]), "dist": self.rng.uniform(*ex["dist"]),
                                       "hold": self.rng.uniform(*ex["hold"]), "toward": tuple(ex["toward"]),
                                       "phase": "before"}
            elif line is not None:
                human.queue = line
                if getattr(line, "lazy_share", 0.0) > 0.0:
                    human.lazy = self.rng.random() < line.lazy_share
                line.join(human)
                slot = line.slot_of(human)
                human.place(slot.x, slot.y, math.atan2(slot.face[1] - slot.y, slot.face[0] - slot.x))
            else:
                x, y = h.get("start", [0.0, 0.0])
                human.place(x, y, self.rng.uniform(-math.pi, math.pi))
                human.wait_s = self.rng.uniform(0.0, 2.0)
            self.humans.append(human)

        self.map = Map(self.sim.pathfinder, self.floor_y, scenario.get("map_area"))
        # Spot and the people, and each of their links: a ray that meets one of these is not met by the scene.
        self.movers = set()
        for body in self.sim.get_articulated_object_manager().get_objects_by_handle_substring("").values():
            self.movers |= {body.object_id, *body.link_object_ids.keys()}
        self.requests: "queue.Queue[str]" = queue.Queue()
        self.answers: "queue.Queue[tuple]" = queue.Queue()
        self.busy = False
        self.last_request: Optional[str] = None
        self.status = "idle: type a request in the terminal"
        self.answer: Optional[dict] = None
        self.discard_answer = False                # a click came after the request: its answer is not followed
        self.requery_s = requery_s                 # --requery: ask again this often while carrying a request out
        self.active_request: Optional[str] = None  # the request the robot is carrying out, if a request it is
        self.asking_again = False
        self.asked_on_arrival = False
        self.asked_at = 0.0
        self.checks = 0                            # times asked again on arrival, for the active request
        self.drag: Optional[tuple] = None          # ((x, y) pressed, (x, y) now) of a click on the map
        self.frames: Optional[dict] = None         # the last frames rendered, which a click in the RGB lands on
        self.inference_image: Optional[np.ndarray] = None    # the last answer drawn on the image it was asked about
        self.inference_lines: List[str] = []                 # and told as text
        self.show_inference = True
        self.count = 0
        self.t = 0.0

    # --- the world, one frame ---------------------------------------------------------------------------------
    def step(self, dt: float) -> None:
        for line in self.queues.values():
            line.step(dt, self.rng)
        robot_pos = self.robot.position_hab
        positions = [h.position_hab for h in self.humans]
        for i, human in enumerate(self.humans):
            others = positions[:i] + positions[i + 1:] + [robot_pos]
            human.step(dt, others, self.rng, self.island)
        before = self.robot.state
        self.robot.step(dt, [h.position_hab for h in self.humans])
        if self.robot.state == "arrived" and before != "arrived":
            x, y, yaw = self.robot.world_pose
            self.status = f"arrived at ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg)"
            print(f"[sim] {self.status}", flush=True)
            for line in self.arrival_report():
                print(f"[sim]   {line}", flush=True)
        self.t += dt

    def arrival_report(self) -> List[str]:
        """Where the robot stopped, against the true scene: each queue, the nearest person and the target."""
        x, y, yaw = self.robot.world_pose
        lines = []
        for name, line in self.queues.items():
            first, second = line.slots[0], line.slots[1]
            ax, ay = second.x - first.x, second.y - first.y           # along the line, from the counter backwards
            norm = math.hypot(ax, ay)
            ax, ay = ax / norm, ay / norm
            settled = list(line.members)           # in the line, whether standing or stepping up
            if not settled:
                lines.append(f"queue {name}: nobody in it")
                continue
            back = max(((hx - first.x) * ax + (hy - first.y) * ay) for hx, hy in (h.world_xy for h in settled))
            along = (x - first.x) * ax + (y - first.y) * ay
            side = abs(-(x - first.x) * ay + (y - first.y) * ax)
            facing = math.degrees(abs((yaw - math.atan2(-ay, -ax) + math.pi) % (2 * math.pi) - math.pi))
            where = (f"{along - back:.2f} m behind the last of {len(settled)}" if along > back
                     else f"{back - along:.2f} m AHEAD of the last of {len(settled)} (cut in)")
            lines.append(f"queue {name}: {where}, {side:.2f} m off the line, facing {facing:.0f} deg off the line's way")
        if self.humans:
            nearest = min(math.hypot(hx - x, hy - y) for hx, hy in (h.world_xy for h in self.humans))
            lines.append(f"nearest person: {nearest:.2f} m")
        target = (self.answer or {}).get("target_world")
        if target is not None:
            off = math.degrees(abs((yaw - math.atan2(target[1] - y, target[0] - x) + math.pi) % (2 * math.pi) - math.pi))
            lines.append(f"target: {math.hypot(target[0] - x, target[1] - y):.2f} m away, {off:.0f} deg off straight ahead")
        return lines

    def render(self) -> dict:
        _, _, yaw = self.robot.world_pose
        # Behind and above the robot, looking down at it: the first of the places, each higher and nearer than
        # the last, from which nothing hides it. ReplicaCAD's rooms have no ceiling to get in the way.
        body = self.robot.position_hab + np.array([0.0, 0.5, 0.0])
        movers = self.movers

        def blocked(sight):                        # only the scene and its furniture hide the robot
            hits = self.sim.cast_ray(habitat_sim.geo.Ray(mn.Vector3(*body), mn.Vector3(*sight)), 1.0).hits
            return [h for h in hits if h.object_id not in movers]

        for back_m, up_m in VIEW_PLACES:
            eye = self.robot.position_hab - back_m * heading_hab(yaw) + np.array([0.0, up_m, 0.0])
            sight = eye - body
            if not blocked(sight):
                break
        else:                                      # every place is hidden (a scene with a ceiling): come nearer
            back_m, up_m = VIEW_PLACES[0]
            sight = self.robot.position_hab - back_m * heading_hab(yaw) + np.array([0.0, up_m, 0.0]) - body
            hits = blocked(sight)
            eye = body + max(0.1, hits[0].ray_distance - 0.25 / float(np.linalg.norm(sight))) * sight
            back_m, up_m = max(0.3, back_m * (eye[1] - body[1]) / sight[1]), eye[1] - self.robot.position_hab[1]
        place_agent(self.sim, 1, eye, yaw - math.pi / 2, -math.atan2(up_m - 0.5, back_m))
        frames = self.sim.get_sensor_observations(agent_ids=[0, 1])
        self.frames = {"rgb": frames[0][ROBOT_RGB], "depth": frames[0][ROBOT_DEPTH], "view": frames[1][VIEW_RGB],
                       "T_hab_sensor": sensor_transform(self.sim, ROBOT_RGB), "robot": self.robot.world_pose}
        return self.frames

    def observation(self, frames: dict) -> Observation:
        return make_observation(frames["rgb"], frames["depth"], frames["T_hab_sensor"], self.camera.hfov_deg,
                                self.floor_y, frames["robot"])

    # --- clicks -----------------------------------------------------------------------------------------------
    def go_to(self, x: float, y: float, yaw: Optional[float], what: str) -> None:
        """Send the robot to a world position along a navmesh path; without a yaw it ends facing the way it
        travelled, from where it stands to the goal."""
        rx, ry, _ = self.robot.world_pose
        if yaw is None:
            yaw = math.atan2(y - ry, x - rx)
        if self.busy:
            self.discard_answer = True
        self.active_request = None                 # a click takes over: the pipeline is not asked again
        snapped = self.robot.set_goal(Goal(x, y, yaw))
        if snapped is None:
            self.status = f"{what}: no path to ({x:.2f}, {y:.2f})"
        else:
            self.status = f"{what}: going to ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg)"
            if snapped > 0.2:
                self.status += f", stopping {snapped:.2f} m off it (navmesh)"
        print(f"[sim] {self.status}", flush=True)

    def click_rgb(self, u: float, v: float) -> None:
        """A click on the robot's image at full-resolution pixel (u, v): the point the depth there shows."""
        frames = self.frames
        observation = self.observation(frames)
        cam = observation.camera
        row, col = min(int(v), cam["height"] - 1), min(int(u), cam["width"] - 1)
        Z = float(frames["depth"][row, col])
        if Z <= 0:
            self.status = "clicked on nothing the depth camera sees"
            return
        p = np.array([(col + 0.5 - cam["cx"]) * Z / cam["fx"], (row + 0.5 - cam["cy"]) * Z / cam["fy"], Z, 1.0])
        x, y, z = (np.array(cam["T_world_camera"]) @ p)[:3]
        self.go_to(float(x), float(y), None, "click on the image" + (f" ({z:.2f} m up)" if z > 0.15 else ""))

    def on_mouse(self, event: int, wx: int, wy: int, flags: int, layout: dict) -> None:
        mx, my, mw, mh, scale = layout["map"]
        rx, ry, rw, rh = layout["rgb"]
        on_map = mx <= wx < mx + mw and my <= wy < my + mh
        here = self.map.world_at((wx - mx) / scale, (wy - my) / scale) if on_map else None
        if event == cv2.EVENT_LBUTTONDOWN:
            if on_map:
                self.drag = (here, here)
            elif rx <= wx < rx + rw and ry <= wy < ry + rh:
                cam = self.camera
                self.click_rgb((wx - rx) * cam.width / rw, (wy - ry) * cam.height / rh)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None and here is not None:
            self.drag = (self.drag[0], here)
        elif event == cv2.EVENT_LBUTTONUP and self.drag is not None:
            (x, y), end = self.drag[0], (here or self.drag[1])
            self.drag = None
            dragged = math.hypot(end[0] - x, end[1] - y) > 0.15
            self.go_to(x, y, math.atan2(end[1] - y, end[0] - x) if dragged else None, "click on the map")

    # --- requests ---------------------------------------------------------------------------------------------
    def send(self, request: str, frames: dict, again: bool = False) -> None:
        """Ask the pipeline for the goal of a request on the current view. `again`: the request the robot is
        already carrying out, asked once more on a fresh view (--requery); the robot keeps going meanwhile."""
        if self.busy:
            if not again:
                print("[sim] still waiting for the last answer; request not sent", flush=True)
            return
        self.busy, self.asked_at, self.asking_again = True, self.t, again
        self.asked_on_arrival = again and self.robot.state == "arrived"
        if not again:
            self.last_request, self.active_request, self.discard_answer, self.checks = request, request, False, 0
            self.robot.cancel()
        self.count += 1
        folder = self.out / f"{self.count:03d}"
        observation = self.observation(frames)
        _save_observation(folder, observation, request, self.context)
        self.status = ("checking the goal again on a fresh view" if again else f"asking the pipeline: {request!r}")
        print(f"[sim] {self.status}", flush=True)

        def work():
            started = time.time()
            try:
                answer = self.client.goal(observation, request, self.context)
            except SgiError as error:
                answer = {"status": "error", "error": str(error)}
            answer["wall_s"] = time.time() - started
            self.answers.put((folder, observation, request, answer))

        threading.Thread(target=work, daemon=True).start()

    def receive(self, frames: dict) -> None:
        try:
            folder, observation, request, answer = self.answers.get_nowait()
        except queue.Empty:
            return
        self.busy, self.answer = False, answer
        (folder / "answer.json").write_text(json.dumps(answer, indent=2))
        cv2.imwrite(str(folder / "view.png"), cv2.cvtColor(frames["view"][..., :3], cv2.COLOR_RGB2BGR))
        self.inference_image = annotate(observation.rgb, observation.camera, answer)
        self.inference_lines = describe(answer, request)
        (folder / "inference.txt").write_text("\n".join(self.inference_lines) + "\n")
        h = self.inference_image.shape[0]
        cv2.imwrite(str(folder / "inference.png"),
                    np.hstack([self.inference_image, text_panel(self.inference_lines, 640, h)]))
        subtasks = "; ".join(f"{kind}: {text}" for kind, text in answer.get("subtasks", []))
        print(f"[sim] answer in {answer['wall_s']:.1f} s: {answer['status']}"
              + (f" ({answer['error']})" if "error" in answer else "")
              + (f"\n[sim]   plan: {subtasks}" if subtasks else "")
              + (f"\n[sim]   source: {answer.get('source')}, pose_world: {answer.get('pose_world')}"
                 if answer.get("pose_world") else ""), flush=True)
        pose = answer.get("pose_world")
        if self.discard_answer:
            self.status = "the answer came after a click: the click's goal is kept"
            print(f"[sim] {self.status}", flush=True)
            return
        if self.asking_again:
            self._update_goal(answer)
            return
        if answer.get("status") != "ok" or pose is None:
            self.status = f"no goal: {answer.get('status')}"
            return
        snapped = self.robot.set_goal(Goal(*pose))
        if snapped is None:
            self.status = "no path to the goal"
        else:
            self.status = f"going to ({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f} deg) [{answer.get('source')}]"
            if snapped > 0.2:
                self.status += f", stopping {snapped:.2f} m off it (navmesh)"
        print(f"[sim] {self.status}", flush=True)


    def _update_goal(self, answer: dict) -> None:
        """An answer to the request asked again: a goal that moved more than GOAL_MOVED_M or turned more than
        GOAL_TURNED_DEG replaces the one the robot has; any other answer leaves it as it is."""
        pose, goal = answer.get("pose_world"), self.robot.goal
        if answer.get("status") != "ok" or pose is None:
            self.status = f"asked again: no goal ({answer.get('status')}); keeping the one it has"
        elif goal is None:
            self.robot.set_goal(Goal(*pose))
            self.status = f"asked again: going to ({pose[0]:.2f}, {pose[1]:.2f})"
        else:
            moved = math.hypot(pose[0] - goal.x, pose[1] - goal.y)
            turned = abs(math.degrees((pose[2] - goal.yaw + math.pi) % (2 * math.pi) - math.pi))
            if moved > GOAL_MOVED_M or turned > GOAL_TURNED_DEG:
                self.robot.set_goal(Goal(*pose))
                self.status = f"asked again: the goal moved {moved:.2f} m and turned {turned:.0f} deg; going there"
            else:
                self.status = f"asked again: goal confirmed (within {moved:.2f} m, {turned:.0f} deg)"
                if self.asked_on_arrival:
                    self.checks = ARRIVAL_CHECKS   # confirmed on a view from where it stands: nothing more to ask
        print(f"[sim] {self.status}", flush=True)

    def requery(self, frames: dict) -> None:
        """Ask the pipeline again, every --requery seconds while the robot carries out a request, and once on
        arrival, at most ARRIVAL_CHECKS times for one request: the people may have moved since it last saw them."""
        if not self.requery_s or self.active_request is None or self.busy:
            return
        state = self.robot.state
        if state in ("moving", "waiting") and self.t - self.asked_at >= self.requery_s:
            self.send(self.active_request, frames, again=True)
        elif state == "arrived" and self.checks < ARRIVAL_CHECKS and self.t - self.asked_at >= 1.0:
            self.checks += 1
            self.send(self.active_request, frames, again=True)


def _save_observation(folder: Path, observation: Observation, request: str, context: Optional[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "rgb.png"), cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR))
    np.save(folder / "depth.npy", observation.depth)
    (folder / "request.json").write_text(json.dumps({"request": request, "context": context, "camera": observation.camera,
                                                     "robot": observation.robot}, indent=2))


def _read_stdin(requests: "queue.Queue[str]") -> None:
    for line in sys.stdin:
        line = line.strip()
        if line:
            requests.put(line)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sgi_habitat.run", description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--sgi-url", default="http://127.0.0.1:8765")
    parser.add_argument("--request", help="a request to send without typing it")
    parser.add_argument("--request-at", type=float, default=2.0, help="seconds of simulated time before --request")
    parser.add_argument("--context", help="context text sent with every request")
    parser.add_argument("--requery", type=float, default=0.0, metavar="S",
                        help="while carrying out a request, ask the pipeline again every S seconds on a fresh view, "
                             "and on arrival; a goal that moved replaces the old one (default 0: never)")
    parser.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "YAW"),
                        help="drive the robot to this world pose at the start, without the pipeline (a motion check)")
    parser.add_argument("--headless", action="store_true", help="no window (record with --record)")
    parser.add_argument("--window-width", type=int, default=1200,
                        help="width the window opens at, in pixels (default 1200); it can be resized by dragging")
    parser.add_argument("--record", type=Path, help="write the window's frames to this .mp4")
    parser.add_argument("--seconds", type=float, help="stop after this much simulated time")
    parser.add_argument("--exit-on-arrival", action="store_true", help="stop once the robot reaches its goal")
    parser.add_argument("--linger", type=float, default=0.0, metavar="S",
                        help="with --exit-on-arrival, keep going S seconds after arriving (for a recording)")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "habitat" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    scenario = yaml.safe_load(args.scenario.read_text())
    client = SgiClient(args.sgi_url)
    try:
        health = client.health()
        print(f"[sim] pipeline at {args.sgi_url}: goal {health['goal']}, models {health['models']}", flush=True)
    except SgiError as error:
        print(f"[sim] warning: {error}; requests will fail until it runs", flush=True)

    demo = Demo(scenario, client, args.out, args.context, args.seed, args.requery)
    print(f"[sim] {scenario.get('scene') or scenario['stage']}: {len(demo.humans)} people, floor at y = {demo.floor_y:.3f}; "
          f"requests are kept in {args.out}", flush=True)
    for example in scenario.get("requests", []):
        print(f"[sim]   try: {example}", flush=True)
    if not args.headless:
        print("[sim] type a request and press Enter, or click the map or the robot's image to send it there; "
              "in the window: q quit, r resend, c cancel, p pause, i inference panel", flush=True)
        threading.Thread(target=_read_stdin, args=(demo.requests,), daemon=True).start()

    if args.goal is not None:
        demo.robot.set_goal(Goal(*args.goal))
        demo.status = "going to --goal ({:.2f}, {:.2f}, {:.0f} deg)".format(args.goal[0], args.goal[1], math.degrees(args.goal[2]))
    layout: dict = {}
    if not args.headless:
        # Resizable, keeping the picture's shape: clicks still arrive in the picture's own pixels.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        sized = False
        cv2.setMouseCallback(WINDOW, lambda event, x, y, flags, _: demo.on_mouse(event, x, y, flags, layout))
    writer = None
    scripted = args.request
    paused = False
    arrived_at = None
    wall_start = time.time()
    try:
        while args.seconds is None or demo.t < args.seconds:
            tick = time.time()
            if not paused:
                demo.step(1.0 / FPS)
            frames = demo.render()
            if scripted and demo.t >= args.request_at:
                demo.send(scripted, frames)
                scripted = None
            try:
                demo.send(demo.requests.get_nowait(), frames)
            except queue.Empty:
                pass
            demo.receive(frames)
            demo.requery(frames)
            canvas, shown = compose(frames, demo)
            layout.update(shown)
            if args.record is not None:
                if writer is None:
                    args.record.parent.mkdir(parents=True, exist_ok=True)
                    size = (canvas.shape[1], canvas.shape[0])
                    writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
                if (canvas.shape[1], canvas.shape[0]) != size:       # a video keeps the size it started with
                    canvas = cv2.resize(canvas, size)
                writer.write(canvas)
            settled = not demo.busy and (not args.requery or demo.active_request is None or demo.checks >= ARRIVAL_CHECKS)
            if args.exit_on_arrival and demo.robot.state == "arrived" and settled:
                arrived_at = demo.t if arrived_at is None else arrived_at
                if demo.t - arrived_at >= args.linger:
                    break
            else:
                arrived_at = None
            if not args.headless:
                cv2.imshow(WINDOW, canvas)
                if not sized:                      # the first picture sets the shape the window opens with
                    width = min(args.window_width, canvas.shape[1])
                    cv2.resizeWindow(WINDOW, width, int(canvas.shape[0] * width / canvas.shape[1]))
                    sized = True
                key = cv2.waitKey(max(1, int(1000 / FPS - (time.time() - tick) * 1000))) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r") and demo.last_request:
                    demo.send(demo.last_request, frames)
                if key == ord("c"):
                    demo.robot.cancel()
                    demo.status = "goal cancelled"
                if key == ord("p"):
                    paused = not paused
                if key == ord("i"):
                    demo.show_inference = not demo.show_inference
            else:                              # in real time, so the people move on while the models answer
                time.sleep(max(0.0, 1.0 / FPS - (time.time() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        demo.sim.close()
    print(f"[sim] {demo.t:.1f} s simulated in {time.time() - wall_start:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
