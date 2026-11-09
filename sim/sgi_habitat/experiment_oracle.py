"""Oracle ceilings for the moving-people experiment (research note moving/QUESTIONS.md Q10): not methods.

    python -m sgi_habitat.experiment_oracle --scenario FILE --arm {CTo,CTt} --seed N --out DIR [as experiment]

- CTo: CT with the answer's people and constraints replaced, from the first re-solve, by the queue's true members
  at the moment the answer was asked (one `come_after` of them); the model's target point is kept.
- CTt: CT with the target point moved to the queue's true counter point (slot 0's face point); the model's people
  and constraints are kept.
- CT0: one re-solve on the tracked people, then the goal is frozen: CT0 against C1 measures the better people
  positions of tracking; CT against CT0 measures re-solving as people move.
- CT2, CT2-vac, CT2-rat: re-solving v2 (moving/CT2_FROZEN.md): readmit, a 0.5 m line span, the vacated place, path
  memory, and a forward ratchet in the replace rule; CT2-vac without the vacated place, CT2-rat without the ratchet.
- CT2m, Rm: CT2 and R1 that move while thinking (moving/CT2m_FROZEN.md): once the plan call answers (POST /plan, the
  call /goal then reuses from the cache), an interim goal on the ray to the plan's target, 2.5 m short of it or 1.5 m
  before the first tracked person within 0.6 m of the ray, whichever is farther from the target, facing it.
- CTd: CT2 on people tracked from the robot's own camera (E1D_FROZEN.md) instead of the sim's true positions:
  every 0.5 s from the request, and on the request frame itself, the frame goes to the tracker service
  (moving/tracker_service.py, $SGI_TRACKER_URL, default http://127.0.0.1:8897), whose tracks the re-solves use.
- CT2-nora: CT2 without readmit (post hoc ablation, CT2_NORA_POSTHOC.md); CT2-norv: without readmit and without the
  vacated place (CT2_NORV_POSTHOC.md); CT2-none: every CT2 part off through the CT2 code path, no ratchet or
  stood-on replace (a sanity arm that should reproduce CT; CT2_NONE_RAT_POSTHOC.md).
- CTe: CTd whose first answer uses the request frame's detected people (the tracker's live tracks) instead of the
  model's people call (e1/e1c/E1C_FROZEN.md): /goal is sent with `people`, so only the constraints call is new.
- CTs: CT that re-solves only on settled tracks and also replaces a goal someone stands on (frozen in
  moving/CTs_FROZEN.md; judged on fresh seeds only).
- CT4O (a ceiling, not a method): CT4PJ with the roster replaced at every re-solve by the queue's true members
  ahead of the robot (`oracle_roster`); nobody behind the robot is ever in it. The model's target point is kept.
- CT4OT (a ceiling, not a method): CT4O with the target point also moved to the queue's true counter point, as CTt.
- CT4PJT (a ceiling cell, not a method): CT4PJ with only the target point moved to the true counter point, as CTt.
- CT3FT (a ceiling cell, not a method): CT3F with only the target point moved to the true counter point, as CTt.
- OW (a ceiling, not a method): no solver. From the first answer on, every re-solve period the goal is the ideal
  place from the sim's own line: on the line's axis 0.85 m (the scored band's middle) farther back than the last
  person ahead of the robot, facing up the line, or slot 0 facing the counter when nobody is ahead.
- RTw: R-track with the point's offset from its anchor kept in world coordinates (RT keeps it in the anchor's
  frame, which swings when they turn), and the heading of the answer kept: the strongest role-free tracker. The
  anchor's leaving and joining of the queue are logged under `tracked`.
All share C1's first answer; only the re-solves differ.
"""
from __future__ import annotations

import math
import time

from sgi_habitat import experiment as E
from sgi_habitat import experiment_controls as X

E.ARMS.update({"CTo": dict(asks_again=False, resolves=True), "CTt": dict(asks_again=False, resolves=True),
               "CT0": dict(asks_again=False, resolves=True), "RTw": dict(asks_again=False, resolves=False),
               "CTs": dict(asks_again=False, resolves=True), "CT2": dict(asks_again=False, resolves=True),
               "CT2-vac": dict(asks_again=False, resolves=True), "CT2-rat": dict(asks_again=False, resolves=True),
               "CT2m": dict(asks_again=False, resolves=True), "Rm": dict(asks_again=False, resolves=False),
               "CTd": dict(asks_again=False, resolves=True), "CTe": dict(asks_again=False, resolves=True),
               "CT2-nora": dict(asks_again=False, resolves=True), "CT2-norv": dict(asks_again=False, resolves=True),
               "CT2-none": dict(asks_again=False, resolves=True),
               "Gl": dict(asks_again=False, resolves=True), "CT3": dict(asks_again=False, resolves=True),
               "CT3-j": dict(asks_again=False, resolves=True), "CT3b": dict(asks_again=False, resolves=True),
               # ideas round 2: CT3 with the wait's tracks replayed before the first re-solve (a ladder rung)
               "CT3p": dict(asks_again=False, resolves=True, photo_time=True),
               # moving/ideas2_1b/CT4_FROZEN.md: the best bolt-on (CT3F) against one estimate of who waits (CT4)
               "CT3F": dict(asks_again=False, resolves=True, photo_time=True),
               "CT4": dict(asks_again=False, resolves=True, photo_time=True),
               # moving/ideas2_1b/round3/ROUND3.md: CT4 with the 5 s window (W0), and with the cost decision (CT5)
               "W0": dict(asks_again=False, resolves=True, photo_time=True),
               "CT5": dict(asks_again=False, resolves=True, photo_time=True),
               # round3/CT4P.md: CT4 lined up along the remembered line, as CT3F is (a control)
               "CT4P": dict(asks_again=False, resolves=True, photo_time=True),
               # round4/CT4PJ.md: CT4P with the Resolver's joined rule, as CT3F has it (a control)
               "CT4PJ": dict(asks_again=False, resolves=True, photo_time=True),
               # ceilings for the confirmatory test: the true roster ahead (CT4O), the ideal place (OW)
               "CT4O": dict(asks_again=False, resolves=True, photo_time=True, oracle_ahead=True),
               "CT4OT": dict(asks_again=False, resolves=True, photo_time=True, oracle_ahead=True),
               "CT4PJT": dict(asks_again=False, resolves=True, photo_time=True),
               "CT3FT": dict(asks_again=False, resolves=True, photo_time=True),
               # CT3F that holds its place once in line: the people ahead are locked in and only leave (hold_place)
               "CT3FL": dict(asks_again=False, resolves=True, photo_time=True),
               "OW": dict(asks_again=False, resolves=True)})
# BYSTANDERS_FROZEN.md: CT2's options, and roles from geometry (Gl) or from the model checked by motion (CT3; CT3-j
# without the follow check); later arms add their own (see ARMS above)
_ROSTER = ("Gl", "CT3", "CT3-j", "CT3b", "CT3p", "CT3F", "CT4", "W0", "CT5", "CT4P", "CT4PJ", "CT4O", "CT4OT", "CT4PJT",
           "CT3FT", "CT3FL")
_CO_MOTION_S = 5.0
_TRACKED = ("CTd", "CTe")
_TRACK_EVERY_S = 0.5
_SHORT_M, _BEFORE_M, _RAY_M, _NEAR_M = 2.5, 1.5, 0.6, 0.3            # CT2m_FROZEN.md
_V2 = {"readmit": True, "line_span_m": 0.5, "vacate_after_s": 5.0, "path_memory": True}
_RATCHET_M, _CLEAR_M = 0.08, 0.55
_IDEAL_BACK_M, _IDEAL_MOVED_M, _IDEAL_TURNED_DEG = 0.85, 0.1, 15.0      # OW


class _Inject:
    """The episode's client, with the oracle added to the first /resolve."""

    def __init__(self, client, episode):
        self._client, self._episode, self._sent = client, episode, False

    def __getattr__(self, name):
        return getattr(self._client, name)

    def _call(self, path, payload=None):
        if path == "/resolve" and not self._sent:
            self._sent = True
            payload = {**payload, "oracle": self._episode.oracle()}
            if self._episode.arm.startswith("CT2") or self._episode.arm in _TRACKED + _ROSTER:    # the CT2 family
                arm = self._episode.arm
                payload["variant"] = {**_V2, "vacate_after_s": None if arm in ("CT2-vac", "CT2-norv") else 5.0,
                                      "readmit": arm not in ("CT2-nora", "CT2-norv")}
                if arm in ("CT3", "CT3-j", "CT3b", "CT3p", "CT3F", "CT3FT", "CT3FL"):
                    payload["variant"] = {**payload["variant"], "co_motion_s": _CO_MOTION_S,
                                          "co_motion_follow": arm != "CT3-j"}
                if arm == "CT3b":                           # post hoc: stale follow checks cancelled on re-admission
                    payload["variant"]["co_motion_rejoin"] = True
                if arm in ("CT3F", "CT3FT", "CT3FL"):       # CT4_FROZEN.md: the group test and the gated anchor
                    payload["variant"] = {**payload["variant"], "group_follow": True, "anchor_rederive": True}
                if arm == "CT3FL":                          # and the place held once in line
                    payload["variant"]["hold_place"] = True
                if arm == "CT4":                            # the roles from sgi.roles; CT2's ratchet stays (sim side)
                    payload["variant"] = {"role_belief": True}
                if arm in ("CT4P", "CT4PJ"):                # round3/CT4P.md, round4/CT4PJ.md
                    payload["variant"] = {"role_belief": True, "role_direction": "path",
                                          **({"role_joined": True} if arm == "CT4PJ" else {})}
                if arm == "CT4PJT":                         # a ceiling cell: CT4PJ, with the true target (oracle)
                    payload["variant"] = {"role_belief": True, "role_direction": "path", "role_joined": True}
                if arm in ("CT4O", "CT4OT"):                # ceilings: CT4PJ with the true roster ahead
                    payload["variant"] = {"role_belief": True, "role_direction": "path", "role_joined": True,
                                          "oracle_roster": True}
                if arm in ("W0", "CT5"):                    # ROUND3.md section 8
                    payload["variant"] = {"role_belief": True, "static_win_s": 5.0, "role_decision": arm == "CT5"}
                if arm == "CT2-none":                       # every CT2 part off, through the CT2 code path
                    payload["variant"] = {"readmit": False, "line_span_m": 1.0, "vacate_after_s": None,
                                          "path_memory": False}
        if path == "/resolve" and self._episode.arm == "CTs":
            payload = {**payload, "settle": True}
        return self._client._call(path, payload)

    def goal(self, observation, request, context=None):
        """CTe: the first answer is asked with the request frame's detected people (no people call)."""
        if self._episode.arm == "CTe" and getattr(self._episode, "_det_live", None) is not None:
            from sgi_habitat.client import body as _body
            return self._client._call("/goal", {**_body(observation, request, context), "people": self._episode._det_live})
        if self._episode.arm == "Gl":                       # keep a session even where the answer leaves people out
            from sgi_habitat.client import body as _body
            return self._client._call("/goal", {**_body(observation, request, context), "session_always": True})
        return self._client.goal(observation, request, context)


class OracleEpisode(X.ControlEpisode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = _Inject(self.client, self)
        self.members_at_send = None
        scenario = args[0] if args else kwargs["scenario"]
        self.log["bystanders"] = [i for i, h in enumerate(scenario.get("humans", [])) if h.get("bystander")]

    # --- CTd: people tracked from the robot's own camera -------------------------------------------------------
    def _track_frame(self, frames) -> None:
        """Send one frame to the tracker service and keep its tracks (and, for the log, the true positions)."""
        import json as _json
        import os
        import urllib.request
        from sgi_habitat.client import body as _body
        observation = self.observation(frames)
        payload = {**_body(observation, ""), "key": str(self.out), "t": self.t}
        truth = [[i, round(h.world_xy[0], 3), round(h.world_xy[1], 3), h in self.queue.members]
                 for i, h in enumerate(self.humans)]
        url = os.environ.get("SGI_TRACKER_URL", "http://127.0.0.1:8897") + "/track"
        t, started = self.t, time.time()
        try:
            req = urllib.request.Request(url, data=_json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as r:
                answer = _json.loads(r.read())
        except Exception as error:                          # no update for this frame
            self.log.setdefault("det", []).append({"t": round(t, 2), "error": str(error)[:200]})
            return
        self._det_tracks = [{"id": int(tr["id"]), "xy": [tr["x"], tr["y"]], "yaw": tr["yaw"]}
                            for tr in answer.get("tracks", [])]
        self._det_live_now = [{"xy": [tr["x"], tr["y"]], "yaw": tr["yaw"]} for tr in answer.get("tracks", [])
                              if not tr["stale"]]
        self.log.setdefault("det", []).append({
            "t": round(t, 2), "ms": answer.get("ms"), "wall_ms": round((time.time() - started) * 1000, 1),
            "detected": answer.get("detected"), "truth": truth,
            "tracks": [[tr["id"], round(tr["x"], 3), round(tr["y"], 3), bool(tr["stale"])] for tr in answer.get("tracks", [])]})

    def tracks(self) -> list:
        if self.arm in _TRACKED:
            return list(getattr(self, "_det_tracks", []))
        return super().tracks()

    def send(self, request, frames, again=False):
        if self.arm in _TRACKED and not again and not self.busy and getattr(self, "_det_tracks", None) is None:
            self._track_frame(frames)                       # the request frame: the tracks the answer is matched to
            self._last_track_t = self.t
            self._det_live = list(getattr(self, "_det_live_now", []))     # CTe: the people of the first answer
        if not self.busy and self.members_at_send is None:
            self.members_at_send = [i for i, h in enumerate(self.humans) if h in self.queue.members]
        if self.arm in ("CT2m", "Rm") and not again and not self.busy and not hasattr(self, "_plans"):
            import queue as _queue
            import threading
            from sgi_habitat.client import body as _body
            self._plans, sent = _queue.Queue(), self.t
            observation = self.observation(frames)

            def plan():
                try:
                    answer = self.client._call("/plan", _body(observation, request, self.context))
                except Exception as error:                  # an interim goal is optional
                    answer = {"status": "error", "error": str(error)}
                self._plans.put((sent, answer))

            threading.Thread(target=plan, daemon=True).start()
        super().send(request, frames, again)

    def _interim(self) -> None:
        """Deliver the plan's interim goal at its reported time, unless the full answer came first."""
        held = getattr(self, "_held_plan", None)
        if held is None and hasattr(self, "_plans") and not self._plans.empty():
            sent, answer = self._plans.get()
            held = self._held_plan = (sent + float((answer.get("timings_s") or {}).get("plan") or 0.0), answer)
        if held is None or self.t < held[0] or held[1] is None:
            return
        due, answer = held
        self._held_plan = (due, None)                       # once
        entry = {"t": round(self.t, 2), "event": "interim", "status": answer.get("status")}
        target = answer.get("target_world")
        if self.log["answers"] or answer.get("status") != "ok" or target is None:
            self.log.setdefault("events", []).append({**entry, "goal": None})
            return
        rx, ry, _ = self.robot.world_pose
        length = math.hypot(target[0] - rx, target[1] - ry)
        if length < 1e-6:
            return
        ux, uy = (target[0] - rx) / length, (target[1] - ry) / length
        stop = length - _SHORT_M                            # along the ray, from the robot
        on_ray = [((h.world_xy[0] - rx) * ux + (h.world_xy[1] - ry) * uy) for h in self.humans
                  if abs(-(h.world_xy[0] - rx) * uy + (h.world_xy[1] - ry) * ux) <= _RAY_M]
        on_ray = [s_ for s_ in on_ray if 0.0 < s_ < length]
        if on_ray:
            stop = min(stop, min(on_ray) - _BEFORE_M)       # farther from the target is nearer the robot
        if stop <= _NEAR_M:
            self.log.setdefault("events", []).append({**entry, "goal": None, "stop_m": round(stop, 3)})
            return
        goal = (rx + stop * ux, ry + stop * uy, math.atan2(uy, ux))
        self.robot.set_goal(E.Goal(*goal))
        self.log.setdefault("events", []).append({**entry, "goal": [round(v, 3) for v in goal], "stop_m": round(stop, 3)})

    def oracle(self) -> dict:
        if self.arm == "CTo":
            return {"come_after": list(self.members_at_send or [])}
        if self.arm == "Gl":                                # who waits, from geometry, on the answer's frame
            return {"chain_m": 1.0}
        if self.arm in ("CTt", "CT4OT", "CT4PJT", "CT3FT"):
            face = self.queue.slots[0].face
            return {"target_xy": [float(face[0]), float(face[1])]}
        return {}

    def _ideal(self) -> tuple:
        """OW: the ideal place now, from the sim's line (see the module docstring), and the truth it came from."""
        truth = self.truth()
        ahead = [self.humans[i] for i in truth["ahead"]]
        if not ahead:
            s0 = self.queue.slots[0]
            return (s0.x, s0.y, math.atan2(s0.face[1] - s0.y, s0.face[0] - s0.x)), truth
        line = self.line
        along = max(line.project(*m.world_xy)[0] for m in ahead) + _IDEAL_BACK_M
        k = 0 if along < 0 else max(i for i in range(len(line.length)) if line.start[i] <= along)
        x, y = line.p[k] + (along - line.start[k]) * line.unit[k]
        return (float(x), float(y), math.atan2(-line.unit[k][1], -line.unit[k][0])), truth

    def _ideal_step(self) -> None:
        if self.session is None or self.t - self.last_resolve_t < self.resolve_s:
            return
        self.last_resolve_t = self.t
        pose, truth = self._ideal()
        self.log["resolves"].append({"sent_t": self.t, "delivered_t": self.t, "truth": truth, "status": "ok",
                                     "pose_world": [round(v, 4) for v in pose], "last": truth["last"],
                                     "roster": list(truth["ahead"])})
        goal = self.robot.goal
        if (goal is None or math.hypot(pose[0] - goal.x, pose[1] - goal.y) > _IDEAL_MOVED_M
                or E._off(pose[2], goal.yaw) > _IDEAL_TURNED_DEG):
            self.robot.set_goal(E.Goal(*pose))

    def resolve(self, frames):
        if self.arm == "OW":
            self._ideal_step()
            return
        if self.arm in ("CT2m", "Rm"):
            self._interim()
        if (self.arm in _TRACKED and self.request_t is not None and not getattr(self, "_tracking", False)
                and self.t - getattr(self, "_last_track_t", -1e9) >= _TRACK_EVERY_S):
            import threading
            self._tracking, self._last_track_t = True, self.t
            snapshot = self.render()

            def work():
                try:
                    self._track_frame(snapshot)
                finally:
                    self._tracking = False

            threading.Thread(target=work, daemon=True).start()
        super().resolve(frames)
        if self.arm == "CT0" and self.log["resolves"] and self.spec["resolves"]:
            self.spec = {**self.spec, "resolves": False}          # one re-solve, then the goal stays
        if self.arm == "RTw" and self.anchor is not None and self.t - self.last_track_t >= self.resolve_s:
            self.last_track_t = self.t
            hx, hy = self.humans[self.anchor["id"]].world_xy
            dx, dy = self.anchor["offset"]
            self._replace_goal({"status": "ok", "pose_world": [hx + dx, hy + dy, self.anchor["yaw"]]})

    def _replace_goal(self, answer):
        """CTs also replaces a goal that someone now stands on (within the person and robot radii); CT2 and CT2-vac
        also take any new goal more than 0.08 m up the line (the forward ratchet)."""
        goal, pose = self.robot.goal, answer.get("pose_world")
        if (self.arm in ("CT2", "CT2-vac", "CT2m", "CT2-nora", "CT2-norv") + _TRACKED + _ROSTER and goal is not None
                and pose is not None
                and answer.get("status") == "ok"):
            d = answer.get("solve_direction")
            forward = (d is not None and -((pose[0] - goal.x) * d[0] + (pose[1] - goal.y) * d[1]) > _RATCHET_M)
            stood_on = any(math.hypot(h.world_xy[0] - goal.x, h.world_xy[1] - goal.y) < _CLEAR_M for h in self.humans)
            if forward or stood_on:
                self.robot.set_goal(E.Goal(*pose))
                return
        goal, pose = self.robot.goal, answer.get("pose_world")
        if (self.arm == "CTs" and goal is not None and pose is not None and answer.get("status") in ("ok", "held")
                and any(math.hypot(h.world_xy[0] - goal.x, h.world_xy[1] - goal.y) < 0.55 for h in self.humans)):
            self.robot.set_goal(E.Goal(*pose))
            return
        super()._replace_goal(answer)

    def receive(self, frames):
        before = len(self.log["answers"])
        super().receive(frames)
        if self.arm == "RTw" and self.anchor is None:
            for entry in self.log["answers"][before:]:
                if entry.get("pose_world"):
                    x, y, yaw = entry["pose_world"]
                    near = [(math.hypot(h.world_xy[0] - x, h.world_xy[1] - y), i) for i, h in enumerate(self.humans)]
                    gap, i = min(near) if near else (math.inf, None)
                    if i is not None and gap <= 3.0:
                        hx, hy = self.humans[i].world_xy
                        self.anchor = {"id": i, "offset": (x - hx, y - hy), "yaw": yaw}
                        self.in_queue = self.humans[i] in self.queue.members
                    self.log["tracked"].append({"t": self.t, "anchor": i if gap <= 3.0 else None, "gap_m": round(gap, 3)})
                    break

    def step(self, dt):
        super().step(dt)
        if self.arm in ("RT", "RTw") and self.anchor is not None:
            inside = self.humans[self.anchor["id"]] in self.queue.members
            if inside != getattr(self, "in_queue", inside):
                self.log["tracked"].append({"t": round(self.t, 2), "anchor_in_queue": inside})
            self.in_queue = inside


def main(argv=None) -> int:
    E.Episode = OracleEpisode
    return E.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
