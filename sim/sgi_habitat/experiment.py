"""Moving people: one episode of one arm, scored every second against the live line.

    python -m sgi_habitat.experiment --scenario FILE --arm {R1,Rq,C1,Cq,CT} --seed N --out DIR
                                     [--sgi-url URL] [--request TEXT] [--seconds 92] [--requery 10]
                                     [--resolve-every 0.5] [--record FILE]

The arms (research note moving/DESIGN.md): R1 and C1 ask the pipeline once; Rq and Cq ask again every --requery
seconds for the whole episode; CT asks once and then has the stored answer of the constraint method solved again
on the people's tracked positions (POST /resolve) every --resolve-every seconds. The R arms need a service started
with --goal direct_marking, the C arms one with the constraint method.

What every arm shares, so that only the arm differs:
- An answer is delivered at the simulated time it was asked plus the sum of its stages' reported times: a cached
  answer arrives as late as the call that was stored did. A re-solve is delivered after the time it took. The
  simulation never runs faster than real time.
- The robot counts as a member of the line once it has arrived at the line's end, so that people coming back to
  the line queue behind it, also when it has not moved up as the line did.
- The goal is replaced by a new answer only when it moves more than 0.3 m or turns more than 25 degrees.

The episode is written to DIR/episode.json: every answer and re-solve, a sample a second (the robot's pose, the
line, whether the robot is in line or has cut in, by moving/SCORING.md), and the summary.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml

from sgi_habitat import run as R
from sgi_habitat.client import SgiError, body
from sgi_habitat.humans import Queue
from sgi_habitat.robot import Goal

ARMS = {"R1": dict(asks_again=False, resolves=False), "Rq": dict(asks_again=True, resolves=False),
        "C1": dict(asks_again=False, resolves=False), "Cq": dict(asks_again=True, resolves=False),
        "CT": dict(asks_again=False, resolves=True)}
TRACK_RANGE_M = 12.0          # people farther from the robot than the solver's floor are not tracked
JOIN_GAP_M = (0.3, 1.5)       # the robot joins the line arrived this far behind the last person in it
CORRIDOR_M = 0.8              # and this near the line's axis; it leaves the line farther than this from it
BAND_M = (0.5, 1.2)           # in line: this far from the person ahead of it (SCORING.md)
FIRST_M = 1.6                 # first in line: this near slot 0
HEADING_DEG = 45.0
CLEAR_M = 0.55
CUT_CORRIDOR_M, CUT_AHEAD_M = 1.0, 0.25


class RobotInLine:
    """The robot's place among a queue's members, once it has joined: never served, so the line waits at the
    counter while it is first."""
    name = "robot"
    settled = False

    def __init__(self, robot):
        self.robot = robot
        self.slot_index = 0   # the slot nearest where the robot stands, kept up to date by the episode

    @property
    def world_xy(self) -> tuple:
        x, y, _ = self.robot.world_pose
        return x, y

    def leave_queue(self, wander_s: float) -> None:      # never called: an unsettled front is never served
        pass


def _slot_of(self: Queue, human) -> Optional[object]:
    """As Queue.slot_of, but those behind the robot in the line queue behind where the robot stands."""
    if human not in self.members:
        return None
    i = self.members.index(human)
    robot = next((m for m in self.members if isinstance(m, RobotInLine)), None)
    if robot is not None and i > self.members.index(robot):
        i = max(i, robot.slot_index + i - self.members.index(robot))
    return self.slots[min(i, len(self.slots) - 1)]


Queue.slot_of = _slot_of


class Line:
    """A queue's slots as a polyline from slot 0 backwards, extended straight past both ends."""

    def __init__(self, slots):
        self.p = np.array([[s.x, s.y] for s in slots], dtype=float)
        seg = np.diff(self.p, axis=0)
        self.length = np.hypot(seg[:, 0], seg[:, 1])
        self.unit = seg / self.length[:, None]
        self.start = np.concatenate([[0.0], np.cumsum(self.length)[:-1]])
        self.face0 = slots[0].face

    def project(self, x: float, y: float) -> tuple:
        """(along, side, the heading up the line there) of a point."""
        best = None
        n = len(self.length)
        for k in range(n):
            d = np.array([x, y]) - self.p[k]
            s = float(d @ self.unit[k])
            lo = -math.inf if k == 0 else 0.0
            hi = math.inf if k == n - 1 else float(self.length[k])
            s = min(max(s, lo), hi)
            foot = self.p[k] + s * self.unit[k]
            side = float(np.hypot(*(np.array([x, y]) - foot)))
            if best is None or side < best[1] - 1e-9:
                best = (float(self.start[k]) + s, side, math.atan2(-self.unit[k][1], -self.unit[k][0]))
        return best

    def slot_near(self, along: float) -> int:
        slots = np.concatenate([[0.0], np.cumsum(self.length)])
        return int(np.argmin(np.abs(slots - along)))


def _off(a: float, b: float) -> float:
    return abs(math.degrees((a - b + math.pi) % (2 * math.pi) - math.pi))


class Episode(R.Demo):
    def __init__(self, scenario: dict, client, out: Path, arm: str, seed: int, requery_s: float, resolve_s: float,
                 request: str):
        self.arm, self.spec = arm, ARMS[arm]
        super().__init__(scenario, client, out, None, seed, requery_s if self.spec["asks_again"] else 0.0)
        self.request, self.resolve_s = request, resolve_s
        (self.qname, self.queue), = self.queues.items()     # the moving-line scenarios have one queue
        self.line = Line(self.queue.slots)
        self.joined_at = {m: 0.0 for m in self.queue.members}
        join = self.queue.join

        def join_timed(human, _join=join):
            if human not in self.queue.members:
                self.joined_at[human] = self.t
            _join(human)
        self.queue.join = join_timed
        self.robot_in_line = RobotInLine(self.robot)
        self.robot_joined_t: Optional[float] = None
        self.held: list = []                       # (due, asked, tracks, item): answers waiting for their time
        self.held_resolves: list = []              # (due, sent, answer): re-solves waiting for theirs
        self.ready: "queue.Queue[tuple]" = queue.Queue()
        self.resolves: "queue.Queue[tuple]" = queue.Queue()
        self.resolving = False
        self.last_resolve_t = -math.inf
        self.session: Optional[str] = None
        self.tracks_at_answer: Optional[list] = None
        self.tracks_at_send: Optional[list] = None
        self.log: dict = {"answers": [], "resolves": [], "samples": [], "joins": []}
        self.next_sample: Optional[float] = None
        self.request_t: Optional[float] = None
        self.floor_map = self.navigation_map() if self.spec["resolves"] else None
        self.floor_map_sent = False
        self.history: list = []                    # arms with photo_time: the tracks of the wait, every 0.5 s
        self.next_history: Optional[float] = None
        self.history_sent = False

    # --- the robot's navigation map, for re-solving where the picture no longer shows the floor ---------------
    def navigation_map(self, resolution: float = 0.05) -> dict:
        """Where Spot's own navmesh (cut back by its clearance) lets it stand, as a world grid, rows along y."""
        import base64, io
        from PIL import Image
        pathfinder = self.robot.pathfinder
        lo, hi = pathfinder.get_bounds()
        xs = np.arange(lo[0], hi[0] + resolution, resolution)
        ys = np.arange(-hi[2], -lo[2] + resolution, resolution)          # the world's y is Habitat's -z
        mask = np.zeros((len(ys), len(xs)), np.uint8)
        for r, y in enumerate(ys):
            for c, x in enumerate(xs):
                if pathfinder.is_navigable(np.array([x, self.floor_y, -y], dtype=np.float32)):
                    mask[r, c] = 255
        png = io.BytesIO()
        Image.fromarray(mask).save(png, format="PNG")
        return {"origin": [float(xs[0]), float(ys[0])], "resolution": resolution,
                "png": base64.b64encode(png.getvalue()).decode()}

    # --- the people as a tracker gives them ----------------------------------------------------------------
    def tracks(self) -> list:
        x, y, _ = self.robot.world_pose
        return [{"id": i, "xy": [float(h.world_xy[0]), float(h.world_xy[1])], "yaw": float(h.yaw)}
                for i, h in enumerate(self.humans) if math.hypot(h.world_xy[0] - x, h.world_xy[1] - y) <= TRACK_RANGE_M]

    # --- asking, with answers delivered on simulated time ---------------------------------------------------
    def send(self, request: str, frames: dict, again: bool = False) -> None:
        if not self.busy:
            self.tracks_at_send = self.tracks()
            if self.request_t is None:
                self.request_t = self.t
                self.next_sample = self.t
                self.next_history = self.t if self.spec.get("photo_time") else None
        super().send(request, frames, again)

    def receive(self, frames: dict) -> None:
        while True:
            try:
                item = self.answers.get_nowait()
            except queue.Empty:
                break
            answer = item[3]
            timings = answer.get("timings_s") or {}
            # the stages' own times: a cached call counts the latency of the call that was stored ("total" is the
            # service's wall time, which a cached answer does not take)
            latency = sum(v for k, v in timings.items() if k != "total" and isinstance(v, (int, float)))
            self.held.append((self.asked_at + float(latency), self.asked_at, self.tracks_at_send, item))
        for entry in [h for h in self.held if h[0] <= self.t]:
            self.held.remove(entry)
            due, asked, tracks, item = entry
            answer = item[3]
            self.log["answers"].append({"asked_t": asked, "delivered_t": self.t, "latency_s": due - asked,
                                        "wall_s": answer.get("wall_s"), "status": answer.get("status"),
                                        "source": answer.get("source"), "pose_world": answer.get("pose_world"),
                                        "calls": answer.get("calls"), "session": answer.get("session"),
                                        "cached": [c.get("cached") for c in (answer.get("inference") or {}).get("calls", [])]})
            if self.spec["resolves"] and self.session is None and answer.get("session"):
                self.session, self.tracks_at_answer = answer["session"], tracks
            self.ready.put(item)
            answers, self.answers = self.answers, self.ready
            try:
                super().receive(frames)
            finally:
                self.answers = answers

    def requery(self, frames: dict) -> None:
        """Ask again every --requery seconds for the whole episode, whatever the robot is doing."""
        if not self.requery_s or self.active_request is None or self.busy:
            return
        if self.t - self.asked_at >= self.requery_s:
            # a fresh view: the loop decides whether to render before the last answer is delivered, so the frames
            # it holds can be the view asked about last time (the bug of DEVIATIONS 4)
            self.send(self.active_request, self.render(), again=True)

    # --- re-solving on tracks -----------------------------------------------------------------------------
    def resolve(self, frames: dict) -> None:
        while True:
            try:
                due, sent, answer = self.resolves.get_nowait()
            except queue.Empty:
                break
            self.held_resolves.append((due, sent, answer))
        for entry in [h for h in self.held_resolves if h[0] <= self.t]:
            self.held_resolves.remove(entry)
            due, sent, answer = entry
            self.resolving = False
            self.log["resolves"].append({"sent_t": sent, "delivered_t": self.t, "tracks": answer.get("_tracks"),
                                         "truth": answer.get("_truth"),
                                         **{k: answer.get(k) for k in (
                "status", "pose_world", "in_turn", "joined", "dropped", "match_rate", "people", "line_direction",
                "solve_direction", "direction_source", "floor", "error", "last", "turn_joined", "exempt", "non_followers", "co_motion_events", "roster",
                "choice", "costs", "marginals", "vacated", "behind") if k in answer},
                **({k: answer[k] for k in ("answer_come_after", "chain") if answer.get(k) is not None}
                   if not self.log["resolves"] else {})})
            self._replace_goal(answer)
        if (not self.spec["resolves"] or self.session is None or self.resolving or self.busy
                or self.t - self.last_resolve_t < self.resolve_s):
            return
        self.resolving, self.last_resolve_t = True, self.t
        observation, session, tracks, t0, sent = self.observation(frames), self.session, self.tracks(), self.tracks_at_answer, self.t
        truth = self.truth()
        floor_map = None if self.floor_map_sent else self.floor_map
        self.floor_map_sent = True
        history = None
        if self.next_history is not None and not self.history_sent:   # once, with the first re-solve
            history, self.history_sent = list(self.history), True
            self.log["history_t"] = [f["t"] for f in history if f["t"] < sent]   # the frames serve replays

        def work():
            started = time.time()
            payload = {**body(observation, ""), "session": session, "t": sent, "tracks": tracks, "tracks_at_answer": t0}
            if floor_map is not None:
                payload["floor_map"] = floor_map
            if history is not None:
                payload["history"] = history
            if self.spec.get("oracle_ahead"):              # a ceiling: who waits ahead of the robot, from the sim
                payload["oracle_ahead"] = truth["ahead"]
            try:
                answer = self.client._call("/resolve", payload)
            except SgiError as error:
                answer = {"status": "error", "error": str(error)}
            answer["_tracks"] = [[tr["id"], round(tr["xy"][0], 3), round(tr["xy"][1], 3)] for tr in tracks]   # for the log
            answer["_truth"] = truth
            self.resolves.put((sent + (time.time() - started), sent, answer))

        threading.Thread(target=work, daemon=True).start()

    def _replace_goal(self, answer: dict) -> None:
        pose, goal = answer.get("pose_world"), self.robot.goal
        if answer.get("status") != "ok" or pose is None:
            return
        if goal is None or (math.hypot(pose[0] - goal.x, pose[1] - goal.y) > R.GOAL_MOVED_M
                            or _off(pose[2], goal.yaw) > R.GOAL_TURNED_DEG):
            self.robot.set_goal(Goal(*pose))

    # --- the line and the score ----------------------------------------------------------------------------
    def step(self, dt: float) -> None:
        super().step(dt)
        self._robot_in_line()
        if self.next_history is not None and not self.history_sent and self.t >= self.next_history - 1e-9:
            x, y, _ = self.robot.world_pose
            self.history.append({"t": round(self.t, 3), "tracks": self.tracks(), "robot": [round(x, 3), round(y, 3)]})
            self.next_history += 0.5
        if self.next_sample is not None and self.t >= self.next_sample - 1e-9:
            self.log["samples"].append(self.sample())
            self.next_sample += 1.0

    def _robot_in_line(self) -> None:
        x, y, _ = self.robot.world_pose
        along, side, _ = self.line.project(x, y)
        me = self.robot_in_line
        me.slot_index = self.line.slot_near(along)
        members = self.queue.members
        if me in members:
            if side > CORRIDOR_M:
                members.remove(me)
                self.log["joins"].append({"t": self.t, "event": "left"})
            return
        if self.robot.state != "arrived" or side > CORRIDOR_M:
            return
        people = [m for m in members if not isinstance(m, RobotInLine)]
        if people:
            back = max(self.line.project(*m.world_xy)[0] for m in people)
            ok = JOIN_GAP_M[0] <= along - back <= JOIN_GAP_M[1]
        else:
            ok = math.hypot(x - self.queue.slots[0].x, y - self.queue.slots[0].y) <= FIRST_M
        if ok:
            members.append(me)
            self.robot_joined_t = self.t
            self.log["joins"].append({"t": self.t, "event": "joined", "behind": len(people)})

    def truth(self) -> dict:
        """Who is in the line now, as the sim has it, by track id: the queue's order (front first), those ahead of
        the robot in that order (everyone when the robot has not joined), the last of those (`last`: the right
        last for re-solving, CT4_FROZEN.md section 4), and the last of those by distance along the line (`last_along`,
        as `sample` scores the gap)."""
        order = [m for m in self.queue.members if not isinstance(m, RobotInLine)]
        mine = next((k for k, m in enumerate(self.queue.members) if m is self.robot_in_line), None)
        ahead = order if mine is None else [m for m in self.queue.members[:mine] if not isinstance(m, RobotInLine)]
        index = self.humans.index
        return {"order": [index(m) for m in order], "joined": mine is not None, "ahead": [index(m) for m in ahead],
                "last": index(ahead[-1]) if ahead else None,
                "last_along": index(max(ahead, key=lambda m: self.line.project(*m.world_xy)[0])) if ahead else None}

    def sample(self) -> dict:
        x, y, yaw = self.robot.world_pose
        along, side, up = self.line.project(x, y)
        me = self.robot_in_line
        joined = me in self.queue.members
        people = [m for m in self.queue.members if not isinstance(m, RobotInLine)]
        ahead = [m for m in people if not joined or self.joined_at.get(m, 0.0) < self.robot_joined_t]
        nearest = min((math.hypot(h.world_xy[0] - x, h.world_xy[1] - y) for h in self.humans), default=99.0)
        out = {"t": round(self.t, 2), "robot": [round(x, 3), round(y, 3), round(yaw, 4)], "along": round(along, 3),
               "side": round(side, 3), "state": self.robot.state, "joined": joined, "members": len(people),
               "ahead": len(ahead), "nearest_m": round(nearest, 3),
               "goal": None if self.robot.goal is None else [round(self.robot.goal.x, 3), round(self.robot.goal.y, 3)]}
        if ahead:
            last = max(ahead, key=lambda m: self.line.project(*m.world_xy)[0])
            lx, ly = last.world_xy
            gap = math.hypot(x - lx, y - ly)
            behind = along - self.line.project(lx, ly)[0]
            heading = _off(yaw, up)
            out.update(gap_m=round(gap, 3), behind_m=round(behind, 3), heading_off_deg=round(heading, 1))
            out["in_line"] = bool(behind >= 0 and BAND_M[0] <= gap <= BAND_M[1] and heading <= HEADING_DEG
                                  and nearest >= CLEAR_M)
        else:
            s0 = self.queue.slots[0]
            heading = _off(yaw, math.atan2(s0.face[1] - y, s0.face[0] - x))
            to0 = math.hypot(x - s0.x, y - s0.y)
            out.update(to_slot0_m=round(to0, 3), heading_off_deg=round(heading, 1))
            out["in_line"] = bool(to0 <= FIRST_M and heading <= HEADING_DEG and nearest >= CLEAR_M)
        out["cut_in"] = bool(side <= CUT_CORRIDOR_M and any(
            along < self.line.project(*m.world_xy)[0] - CUT_AHEAD_M for m in ahead))
        out["queue_ids"] = sorted(self.humans.index(m) for m in people)   # who is in the line now, by track id
        return out


def summary(log: dict) -> dict:
    samples = log["samples"]
    if not samples:
        return {"samples": 0}
    t0 = samples[0]["t"]
    first = next((s["t"] - t0 for s in samples if s["in_line"]), None)
    lost, run, seen = 0, 0, False
    for s in samples:
        if s["in_line"]:
            if seen and run >= 3:
                lost += 1
            seen, run = True, 0
        elif seen:
            run += 1
    if seen and run >= 3:
        lost += 1
    calls = sum(len(a.get("calls") or []) for a in log["answers"])
    return {"samples": len(samples), "in_line_at_end": samples[-1]["in_line"],
            "share_in_line": sum(s["in_line"] for s in samples) / len(samples), "first_in_line_s": first,
            "cut_in_samples": sum(s["cut_in"] for s in samples), "lost_place": lost, "answers": len(log["answers"]),
            "calls": calls, "resolves": len(log["resolves"]),
            "latencies_s": [round(a["latency_s"], 2) for a in log["answers"]],
            "match_rate": next((r["match_rate"] for r in log["resolves"] if r.get("match_rate") is not None), None),
            "dropped": log["resolves"][-1].get("dropped") if log["resolves"] else None,
            "resolver_joined": log["resolves"][-1].get("joined") if log["resolves"] else None}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sgi_habitat.experiment", description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sgi-url", default="http://127.0.0.1:8765")
    parser.add_argument("--request", help="default: the scenario's first example request")
    parser.add_argument("--request-at", type=float, default=2.0)
    parser.add_argument("--seconds", type=float, default=92.0)
    parser.add_argument("--requery", type=float, default=10.0)
    parser.add_argument("--resolve-every", type=float, default=0.5)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--label", default="", help="written into the episode, such as the model entry")
    args = parser.parse_args(argv)

    scenario = yaml.safe_load(args.scenario.read_text())
    request = args.request or scenario["requests"][0]
    client = R.SgiClient(args.sgi_url)
    health = client.health()
    episode = Episode(scenario, client, args.out, args.arm, args.seed, args.requery, args.resolve_every, request)
    args.out.mkdir(parents=True, exist_ok=True)
    writer, wall = None, time.time()
    try:
        while episode.t < args.seconds:
            tick = time.time()
            episode.step(1.0 / R.FPS)
            need = (args.record is not None or (episode.request_t is None and episode.t >= args.request_at)
                    or (episode.requery_s and not episode.busy and episode.t - episode.asked_at >= episode.requery_s)
                    or (episode.spec["resolves"] and episode.session is not None and not episode.resolving
                        and episode.t - episode.last_resolve_t >= episode.resolve_s))
            frames = episode.render() if need or episode.frames is None else episode.frames
            if episode.request_t is None and episode.t >= args.request_at:
                episode.send(request, frames)
            episode.receive(frames)
            episode.requery(frames)
            episode.resolve(frames)
            if args.record is not None:
                canvas, _ = R.compose(frames, episode)
                if writer is None:
                    args.record.parent.mkdir(parents=True, exist_ok=True)
                    size = (canvas.shape[1], canvas.shape[0])
                    writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"), R.FPS, size)
                writer.write(cv2.resize(canvas, size) if (canvas.shape[1], canvas.shape[0]) != size else canvas)
            time.sleep(max(0.0, 1.0 / R.FPS - (time.time() - tick)))     # never faster than real time
    finally:
        if writer is not None:
            writer.release()
        episode.sim.close()
    result = {"scenario": args.scenario.stem, "arm": args.arm, "seed": args.seed, "label": args.label,
              "models": health.get("models"), "goal": health.get("goal"), "request": request,
              "requery_s": args.requery, "resolve_s": args.resolve_every, "seconds": args.seconds,
              "wall_s": round(time.time() - wall, 1), "summary": summary(episode.log), **episode.log}
    (args.out / "episode.json").write_text(json.dumps(result, indent=1))
    s = result["summary"]
    print(f"[exp] {args.scenario.stem} {args.arm} seed {args.seed}: in line at end {s.get('in_line_at_end')}, "
          f"share {s.get('share_in_line', 0):.2f}, first {s.get('first_in_line_s')}, cut-in samples "
          f"{s.get('cut_in_samples')}, calls {s.get('calls')}, resolves {s.get('resolves')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
