"""Controls and break-it cases for the moving-people experiment (research note moving/QUESTIONS.md).

    python -m sgi_habitat.experiment_controls --scenario FILE --arm ARM --seed N --out DIR [as sgi_habitat.experiment]

Arms, besides those of sgi_habitat.experiment:
- RT: the marked point asked once, then moved with the tracked person nearest it ("R-track"): the point's offset
  and heading are kept in that person's frame and the goal follows them every --resolve-every seconds, under the
  same replace rule as CT. Does tracking alone give what CT gets?
- CTv: CT without the navigation map: re-solves on the first view with its people painted out.
- Rq0: the marked point asked again as soon as the last answer arrives (latency the only limit).
Every answer's staleness is logged: how far the queue's members moved between the frame sent and the answer.

A scenario may list `events`: {t: S, step_aside: K, offset: [dx, dy], for_s: S} sends the K-th person in line
(0: the front) that far aside for that long, still in the queue and keeping their place.
"""
from __future__ import annotations

import math
from typing import Optional

from sgi_habitat import experiment as E
from sgi_habitat.humans import Queue, Slot

E.ARMS.update({"RT": dict(asks_again=False, resolves=False), "CTv": dict(asks_again=False, resolves=True),
               "Rq0": dict(asks_again=True, resolves=False)})
_base_slot_of = Queue.slot_of


def _slot_of(self: Queue, human) -> Optional[object]:
    aside = getattr(self, "aside", {})
    if human in aside and human in self.members:
        return aside[human]
    return _base_slot_of(self, human)


Queue.slot_of = _slot_of


class ControlEpisode(E.Episode):
    def __init__(self, scenario: dict, client, out, arm: str, seed: int, requery_s: float, resolve_s: float,
                 request: str):
        super().__init__(scenario, client, out, arm, seed, requery_s, resolve_s, request)
        if arm == "CTv":
            self.floor_map = None
        if arm == "Rq0":
            self.requery_s = 0.1
        self.anchor: Optional[dict] = None
        self.last_track_t = -math.inf
        self.log["tracked"] = []
        self.events = sorted(scenario.get("events", []), key=lambda e: e["t"])
        self.queue.aside = {}
        self.ends: list = []

    # --- staleness of every answer, and the anchor of R-track ------------------------------------------------
    def receive(self, frames: dict) -> None:
        sent_tracks = self.tracks_at_send
        before = len(self.log["answers"])
        super().receive(frames)
        for entry in self.log["answers"][before:]:
            members = {i for i, h in enumerate(self.humans) if h in self.queue.members}
            then = {t["id"]: t["xy"] for t in (sent_tracks or [])}
            moved = [math.hypot(self.humans[i].world_xy[0] - then[i][0], self.humans[i].world_xy[1] - then[i][1])
                     for i in members if i in then]
            entry["line_moved_mean_m"] = round(sum(moved) / len(moved), 3) if moved else None
            entry["line_moved_max_m"] = round(max(moved), 3) if moved else None
            if self.arm == "RT" and self.anchor is None and entry.get("pose_world"):
                self._anchor(entry["pose_world"])

    def _anchor(self, pose) -> None:
        x, y, yaw = pose
        near = [(math.hypot(h.world_xy[0] - x, h.world_xy[1] - y), i) for i, h in enumerate(self.humans)]
        gap, i = min(near) if near else (math.inf, None)
        if i is None or gap > 3.0:
            self.log["tracked"].append({"t": self.t, "anchor": None, "gap_m": gap})
            return
        h = self.humans[i]
        hx, hy = h.world_xy
        c, s = math.cos(h.yaw), math.sin(h.yaw)
        dx, dy = x - hx, y - hy
        self.anchor = {"id": i, "rel": (c * dx + s * dy, -s * dx + c * dy), "turn": yaw - h.yaw}
        self.log["tracked"].append({"t": self.t, "anchor": i, "gap_m": round(gap, 3)})

    def resolve(self, frames: dict) -> None:
        super().resolve(frames)
        if self.arm != "RT" or self.anchor is None or self.t - self.last_track_t < self.resolve_s:
            return
        self.last_track_t = self.t
        h = self.humans[self.anchor["id"]]
        hx, hy = h.world_xy
        c, s = math.cos(h.yaw), math.sin(h.yaw)
        rx, ry = self.anchor["rel"]
        pose = [hx + c * rx - s * ry, hy + s * rx + c * ry, h.yaw + self.anchor["turn"]]
        self._replace_goal({"status": "ok", "pose_world": pose})

    # --- scripted events ------------------------------------------------------------------------------------
    def step(self, dt: float) -> None:
        while self.events and self.t >= self.events[0]["t"]:
            self._event(self.events.pop(0))
        for end_t, human in [e for e in self.ends if self.t >= e[0]]:
            self.queue.aside.pop(human, None)
            self.ends.remove((end_t, human))
            self.log.setdefault("events", []).append({"t": self.t, "event": "back", "who": human.name})
        super().step(dt)

    def _event(self, event: dict) -> None:
        if "step_aside" in event:
            people = [m for m in self.queue.members if not isinstance(m, E.RobotInLine)]
            k = int(event["step_aside"])
            if k >= len(people):
                return
            human = people[k]
            x, y = human.world_xy
            dx, dy = event.get("offset", [1.5, 0.0])
            slot = self.queue.slot_of(human)
            self.queue.aside[human] = Slot(x + dx, y + dy, slot.face if slot is not None else (x, y + 1.0))
            human.path = []
            self.ends.append((self.t + float(event.get("for_s", 7.0)), human))
            self.log.setdefault("events", []).append({"t": self.t, "event": "step_aside", "who": human.name,
                                                      "index": k, "offset": [dx, dy]})


def main(argv=None) -> int:
    E.Episode = ControlEpisode                     # main builds whichever class the module names
    return E.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
