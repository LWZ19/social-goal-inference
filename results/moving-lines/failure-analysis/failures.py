"""Failure taxonomy for the moving-line episodes (analysis, 2026-11-02; read-only, no model calls).

    .venv/bin/python failures.py                    # every table of FAILURES.md but section 2
    .venv/bin/python failures.py --matrix           # section 2: share of episode time by class, per model and arm
    .venv/bin/python failures.py --episode dsv41/cafe/seed0/CT [--t0 S --t1 S] [--quiet-resolves]

Every 1-second sample from the request to the end that is not in line is a failure sample. Each is given one
class, its most proximate cause, by the decision list in `classify` (see FAILURES.md). Seeds 10-12, the invalid
re-query batch and the repeat batch are never opened. Offline `sgi.solver.solve` replays only; no model calls.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = str(paths.MOVING_RUNS / "runs")
SCEN = str(paths.REPO / "sim" / "scenarios")
MODELS = ["dsv41", "x3", "sonnet55_low"]
EVAL = ["cafe", "room_coffee_cart", "cafe_line2_moving", "cafe_line6_moving", "sweet_spot_line2_moving"]
DEV = ["cafe_busy"]
BAND = (0.5, 1.2)
FIRST_M, HEAD, CLEAR = 1.6, 45.0, 0.55
MOVED_M, TURNED = 0.3, 25.0          # the goal replace rule
NEAR_GOAL = 0.35                     # the robot stands at its goal
LAG_S = 4.0                          # following lag allowed after the person ahead moved
CT_ARMS = ("CT", "CTv", "CT0", "CTo", "CTt")
ONCE_ARMS = ("R1", "C1")
REASK_ARMS = ("Rq", "Rq0", "Cq")
TRACK_R = ("RT", "RTw")


def off(a, b):
    return abs(math.degrees((a - b + math.pi) % (2 * math.pi) - math.pi))


class Line:
    """The queue's slots as a polyline from slot 0 back, extended straight past both ends (as experiment.Line)."""

    def __init__(self, slots):
        self.p = np.array([[s["x"], s["y"]] for s in slots], float)
        seg = np.diff(self.p, axis=0)
        self.length = np.hypot(seg[:, 0], seg[:, 1])
        self.unit = seg / self.length[:, None]
        self.start = np.concatenate([[0.0], np.cumsum(self.length)[:-1]])
        self.s0 = self.p[0]
        f = slots[0].get("face") or [slots[0]["x"], slots[0]["y"] + 1.0]
        self.face0 = np.array(f, float)

    def project(self, x, y):
        """(along, signed side (+ left of the way back), heading up the line)."""
        best = None
        n = len(self.length)
        for k in range(n):
            d = np.array([x, y]) - self.p[k]
            s = float(d @ self.unit[k])
            s = min(max(s, -math.inf if k == 0 else 0.0), math.inf if k == n - 1 else float(self.length[k]))
            foot = self.p[k] + s * self.unit[k]
            v = np.array([x, y]) - foot
            side = float(np.hypot(*v))
            sgn = 1.0 if (self.unit[k][0] * v[1] - self.unit[k][1] * v[0]) >= 0 else -1.0
            if best is None or side < abs(best[1]) - 1e-9:
                best = (float(self.start[k]) + s, sgn * side, math.atan2(-self.unit[k][1], -self.unit[k][0]), k)
        return best[:3]

    def point(self, along, side=0.0):
        k = int(np.searchsorted(self.start, along, side="right") - 1)
        k = min(max(k, 0), len(self.length) - 1)
        u = self.unit[k]
        base = self.p[k] + (along - self.start[k]) * u
        return base + side * np.array([-u[1], u[0]])


_scen = {}


def scenario(name):
    if name not in _scen:
        sc = yaml.safe_load(open(os.path.join(SCEN, name + ".yaml")))
        (q,) = sc["queues"].values()
        _scen[name] = {"line": Line(q["slots"]), "slots": q["slots"], "service": q.get("service_s"),
                       "members0": sum(1 for h in sc["humans"] if h.get("queue")), "robot": sc["robot"]}
    return _scen[name]


def episodes(scenarios=None, models=MODELS):
    scenarios = scenarios or EVAL + DEV
    out = []
    for m in models:
        for f in sorted(glob.glob(os.path.join(RUNS, m, "*", "seed*", "*", "episode.json"))):
            parts = f.split(os.sep)
            scen, seed, arm = parts[-4], parts[-3], parts[-2]
            if scen not in scenarios or re.fullmatch(r"seed1[012]", seed):
                continue
            d = json.load(open(f))
            d["_path"], d["_model"], d["_dir"] = os.path.relpath(os.path.dirname(f), RUNS), m, os.path.dirname(f)
            out.append(d)
    return out


def answer_json(e, i):
    f = os.path.join(e["_dir"], "%03d" % (i + 1), "answer.json")
    return json.load(open(f)) if os.path.exists(f) else None


# ---------------------------------------------------------------------------------------------- goal timeline
def goal_events(e):
    """(t, pose, source, index, accepted) for every answer / re-solve, with the sim's replace rule replayed."""
    evs = [(a["delivered_t"], a.get("pose_world"), "answer", i, a.get("status")) for i, a in enumerate(e["answers"])]
    evs += [(r["delivered_t"], r.get("pose_world"), "resolve", i, r.get("status")) for i, r in enumerate(e["resolves"])]
    evs.sort(key=lambda v: (v[0], 0 if v[2] == "answer" else 1))
    goal, out = None, []
    first = True
    for t, pose, src, i, status in evs:
        ok = status == "ok" and pose is not None
        acc = False
        if ok and (goal is None or (src == "answer" and first)):
            acc = True
        elif ok and (math.hypot(pose[0] - goal[0], pose[1] - goal[1]) > MOVED_M or off(pose[2], goal[2]) > TURNED):
            acc = True
        if src == "answer":
            first = False
        if acc:
            goal = pose
        out.append({"t": t, "pose": pose, "src": src, "i": i, "ok": ok, "acc": acc, "goal": goal})
    return out


def state_at(evs, t):
    """The goal held at t, the event that set it, and the latest delivered event of its kind."""
    held, last = None, None
    for ev in evs:
        if ev["t"] > t + 1e-6:
            break
        if ev["acc"]:
            held = ev
        if ev["ok"]:
            last = ev
    return held, last


# ---------------------------------------------------------------------------------------------- the line at a sample
def last_ahead(sc, x):
    """The estimated position of L (the member ahead of the robot farthest back) from a sample: along exactly,
    side up to a sign (the one nearer the line's axis is taken), and how far off the axis L stands."""
    line = sc["line"]
    a_r, s_r, _ = line.project(x["robot"][0], x["robot"][1])
    a_l = a_r - x["behind_m"]
    d = math.sqrt(max(0.0, x["gap_m"] ** 2 - x["behind_m"] ** 2))
    s_l = min((s_r - d, s_r + d), key=abs)
    return a_l, s_l, line.point(a_l, s_l)


def judge_goal(sc, x, pose, L=None):
    """Whether a pose would be in line against the line of sample x (position and heading separately)."""
    line = sc["line"]
    a_g, s_g, up = line.project(pose[0], pose[1])
    if x["ahead"] == 0:
        to0 = math.hypot(pose[0] - line.s0[0], pose[1] - line.s0[1])
        h = off(pose[2], math.atan2(line.face0[1] - pose[1], line.face0[0] - pose[0])) if len(pose) > 2 else None
        return {"pos": to0 <= FIRST_M, "head": h is None or h <= HEAD, "to0": to0, "side": abs(s_g)}
    a_l, s_l, pl = L or last_ahead(sc, x)
    gap = math.hypot(pose[0] - pl[0], pose[1] - pl[1])
    behind = a_g - a_l
    h = off(pose[2], up) if len(pose) > 2 else None
    geo = ("ahead" if behind < 0 else "close" if gap < BAND[0] else
           ("off" if abs(s_g) > 0.6 else "far") if gap > BAND[1] else
           "off" if abs(s_g) > 0.6 else "ok")      # a goal beside the line is not a place in it (the scorer allows it)
    return {"pos": geo == "ok", "head": h is None or h <= HEAD, "gap": gap, "behind": behind, "side": abs(s_g),
            "geo": geo, "head_off": h}


# ---------------------------------------------------------------------------------------------- per-episode context
def first_answer_facts(e):
    """What the first answer of a C arm says: come_after people against the queue as it stood when asked."""
    a = answer_json(e, 0)
    if a is None:
        return {}
    inf = a.get("inference") or {}
    sc = scenario(e["scenario"])
    n0 = sc["members0"]
    slots = [(s["x"], s["y"]) for s in sc["slots"][:n0]]           # the queue at t = 2 (no service before ~15 s)
    people = {p["number"]: p["xy"] for p in inf.get("people") or []}
    # which model numbers are queue members (nearest slot within 0.75 m, one to one, greedy by distance)
    pairs = sorted((math.hypot(xy[0] - sx, xy[1] - sy), n, k) for n, xy in people.items() for k, (sx, sy) in enumerate(slots))
    num_slot, used = {}, set()
    for dist, n, k in pairs:
        if dist <= 0.75 and n not in num_slot and k not in used:
            num_slot[n] = k
            used.add(k)
    out = {"people": len(people), "members_seen": len(used), "members": n0}
    cons = [c for c in inf.get("constraints") or []]
    ca = [c for c in cons if c["constraint"]["kind"] == "come_after"]
    ca_people = sorted({n for c in ca for n in c["constraint"]["people"]})
    ca_slots = sorted(num_slot[n] for n in ca_people if n in num_slot)
    out["come_after"] = ca_people
    out["come_after_status"] = [c["status"] for c in ca]
    out["ca_slots"] = ca_slots
    out["ca_has_last"] = (n0 - 1) in ca_slots
    out["ca_front_only"] = bool(ca_slots) and max(ca_slots) == 0 and n0 > 1
    out["ca_non_members"] = [n for n in ca_people if n not in num_slot]
    keys = ((inf.get("solver") or {}).get("keys") or {})
    out["turn_m"] = keys.get("turn_m")
    out["usable_turn"] = keys.get("turn_m") is not None and keys["turn_m"] <= 0.3
    out["face"] = [(c["constraint"].get("people"), c["constraint"].get("target"), c["status"]) for c in cons
                   if c["constraint"]["kind"] == "face"]
    out["kinds"] = [(c["constraint"]["kind"], c["status"]) for c in cons]
    tw = a.get("target_world")
    if tw:
        line = sc["line"]
        out["target_xy"] = tw[:2]
        out["target_to_face0"] = math.hypot(tw[0] - line.face0[0], tw[1] - line.face0[1])
        out["target_on_person"] = min((math.hypot(tw[0] - xy[0], tw[1] - xy[1]) for xy in people.values()),
                                      default=9) < 0.4
    return out


def ideal_first(e):
    """Would the first answer's pose be in line against the queue as it stood when the answer was asked?"""
    a = e["answers"][0]
    x = min(e["samples"], key=lambda s: abs(s["t"] - a["asked_t"]))
    pose = a.get("pose_world")
    return judge_goal(scenario(e["scenario"]), x, pose) if pose else None


def answer_valid_when_asked(e, i):
    a = e["answers"][i]
    pose = a.get("pose_world")
    if not pose:
        return None
    x = min(e["samples"], key=lambda s: abs(s["t"] - a["asked_t"]))
    j = judge_goal(scenario(e["scenario"]), x, pose)
    return j["pos"] and j["head"], j


# ---------------------------------------------------------------------------------------------- the classifier
GROUP = {
    "latency": "late start", "approach": "late start",
    # the model's answer (first or re-asked) was wrong when asked
    "fa_ahead": "model answer", "fa_counter": "model answer", "fa_far": "model answer", "fa_off": "model answer",
    "fa_close": "model answer", "fa_noturn": "model answer", "fa_front_goal": "model answer",
    "heading_goal": "model answer",
    # right when asked, but the line moved and the goal was not (or too slowly) updated: the arm's design
    "stale_once": "no update (arm)", "stale_reask": "no update (arm)",
    # the re-solve rules (CT family) and the R-track rule
    "hysteresis": "re-solve rules", "near_miss": "re-solve rules", "ct_ran_out": "re-solve rules",
    "ct_ran_out_front": "re-solve rules",
    "ct_wrong_front": "re-solve rules", "ct_wrong_back": "re-solve rules", "ct_off": "re-solve rules",
    "ct_front_turn": "re-solve rules",
    "rt_ahead": "R-track rule", "rt_counter": "R-track rule", "rt_far": "R-track rule", "rt_off": "R-track rule",
    "rt_close": "R-track rule", "rt_front": "R-track rule",
    # carrying out a valid goal
    "front_move": "execution", "heading_turn": "execution", "moving_valid": "execution", "blocked": "execution",
    "lag": "execution", "snap": "execution", "edge_stop": "execution",
    # the simulator or the scoring
    "clearance": "sim / scoring", "rejoiner_transit": "sim / scoring", "front_clear": "sim / scoring",
}
HYST_LOOKBACK_S = 2.0


def geo_of(j):
    if j["geo"] == "ahead":
        return "counter" if j["side"] > 0.6 else "ahead"
    return j["geo"]


def classify(e, detail=False):
    """A class for every sample not in line (None for those in line). With `detail`, (class, facts) pairs."""
    sc = scenario(e["scenario"])
    arm = e["arm"]
    evs = goal_events(e)
    samples = e["samples"]
    first_t = e["answers"][0]["delivered_t"] if e["answers"] else math.inf
    app_end = next((x["t"] for x in samples if x["t"] >= first_t - 1e-6 and (x["state"] == "arrived" or x["in_line"])),
                   math.inf)
    fa = first_answer_facts(e) if arm.startswith("C") else {}
    rtw_yaw = e["answers"][0]["pose_world"][2] if arm == "RTw" and e["answers"] and e["answers"][0].get("pose_world") else None
    valid_asked = {}

    def asked_ok(i):
        if i not in valid_asked:
            v = answer_valid_when_asked(e, i)
            valid_asked[i] = bool(v and v[0])
        return valid_asked[i]

    def turn_of_answer(i):
        aj = answer_json(e, i)
        return ((((aj or {}).get("inference") or {}).get("solver") or {}).get("keys") or {}).get("turn_m")

    out, facts = [], []
    phys = None                                  # the last estimate of L standing on the line's axis
    for k, x in enumerate(samples):
        f = {}
        if x["ahead"]:
            L = last_ahead(sc, x)
            if abs(L[1]) <= 0.5:
                phys = L
        if x["in_line"]:
            out.append(None)
            facts.append(f)
            continue
        t = x["t"]
        if t < first_t - 1e-6:
            out.append("latency"); facts.append(f); continue
        if t < app_end - 1e-6 or x["goal"] is None:
            out.append("approach"); facts.append(f); continue
        held, last = state_at(evs, t)
        g = x["goal"]
        if arm in TRACK_R:
            gpose = [g[0], g[1]] + ([rtw_yaw] if rtw_yaw is not None else [])
        else:
            gpose = list(held["goal"]) if held else [g[0], g[1]]
        at_goal = math.hypot(x["robot"][0] - g[0], x["robot"][1] - g[1]) <= NEAR_GOAL
        src_i = held["i"] if held else 0
        src = held["src"] if held else "answer"

        def wrong(jg, front=False):
            """The class of a goal that is not a place in line."""
            geo = "front" if front else geo_of(jg)
            f["geo"] = geo
            if arm == "CT0" and src == "resolve":   # one re-solve, then frozen: stale once it was right
                r0 = e["resolves"][src_i]
                x0 = min(samples, key=lambda s: abs(s["t"] - r0["sent_t"]))
                j0 = judge_goal(sc, x0, r0["pose_world"])
                if j0["pos"] and j0["head"]:
                    return "stale_once"
                return "fa_noturn" if not r0.get("in_turn") else "fa_" + geo
            if arm in ONCE_ARMS or arm in TRACK_R:
                if asked_ok(0):
                    return "rt_" + geo if arm in TRACK_R else "stale_once"
                if arm.startswith("C"):
                    tm = fa.get("turn_m")
                    if tm is None or tm > 0.3:
                        return "fa_noturn"
                return "fa_front_goal" if front else "fa_" + geo
            if arm in REASK_ARMS:
                if asked_ok(src_i):
                    return "stale_reask"
                if arm.startswith("C"):
                    tm = turn_of_answer(src_i)
                    if tm is None or tm > 0.3:
                        return "fa_noturn"
                return "fa_front_goal" if front else "fa_" + geo
            # the CT family
            if src == "answer":
                tm = fa.get("turn_m")
                return "fa_noturn" if (tm is None or tm > 0.3) else ("fa_front_goal" if front else "fa_" + geo)
            ref = last if (last is not None and last["src"] == "resolve") else held
            r = e["resolves"][ref["i"]]
            f["in_turn"] = r.get("in_turn")
            if front:
                return "ct_front_turn" if r.get("in_turn") else "fa_front_goal"
            if not r.get("in_turn"):
                had = any(rr.get("in_turn") for rr in e["resolves"][: ref["i"] + 1])
                if not had:
                    return "fa_noturn"
                was_first = any(y["ahead"] == 0 for y in samples if first_t < y["t"] < t)
                return "ct_ran_out_front" if was_first else "ct_ran_out"
            if jg["geo"] == "far" and jg["gap"] <= 1.5:
                return "near_miss"
            if geo in ("ahead", "close", "counter"):
                return "ct_wrong_front"
            if geo == "far":
                return "ct_wrong_back"
            return "ct_off"

        def hysteresis(Lx):
            """A goal offered within the lookback, not taken under the replace rule, that would be in line now."""
            for ev in evs:
                if ev["t"] > t + 1e-6:
                    break
                if t - ev["t"] <= HYST_LOOKBACK_S and ev["ok"] and not ev["acc"]:
                    jl = judge_goal(sc, x, ev["pose"], Lx)
                    if jl["pos"] and jl["head"]:
                        return True
            return False

        if x["ahead"] == 0:
            jg = judge_goal(sc, x, gpose)
            if x["to_slot0_m"] <= FIRST_M and x["heading_off_deg"] <= HEAD:
                c = "front_clear"
            elif not jg["pos"]:
                c = wrong(jg, front=True)
            elif x["to_slot0_m"] > FIRST_M:
                c = "front_move"
            elif x["state"] == "arrived":
                c = "fa_front_goal"          # at the front, facing the way its goal says, which is off the counter
            else:
                c = "front_move"
            out.append(c); facts.append(f); continue
        L = last_ahead(sc, x)
        pos_ok = x["behind_m"] >= 0 and BAND[0] <= x["gap_m"] <= BAND[1]
        if pos_ok and x["heading_off_deg"] > HEAD:
            out.append("heading_goal" if x["state"] == "arrived" else "heading_turn"); facts.append(f); continue
        if pos_ok:
            out.append("clearance"); facts.append(f); continue
        transit = abs(L[1]) > 0.5
        Lj = phys if (transit and phys is not None) else L
        jg = judge_goal(sc, x, gpose, Lj)
        f["gap_goal"], f["geo_goal"] = round(jg["gap"], 2), jg["geo"]
        if transit:
            rx = judge_goal(sc, x, list(x["robot"]), Lj)
            if rx["pos"] and rx["head"]:
                out.append("rejoiner_transit"); facts.append(f); continue
        if jg["pos"]:
            if x["state"] == "waiting":
                c = "blocked"
            elif x["state"] == "arrived" and not at_goal:
                c = "snap"
            elif x["state"] == "arrived":
                # stopped at a valid goal (within the 0.10 m arrival tolerance), but just outside the band
                c = "hysteresis" if hysteresis(Lj) else "edge_stop"
            else:
                c = "moving_valid"
            out.append(c); facts.append(f); continue
        if hysteresis(Lj):
            out.append("hysteresis"); facts.append(f); continue
        out.append(wrong(jg)); facts.append(f)
    # short dips after the line moved, recovered within LAG_S: the following lag
    n = len(samples)
    k = 0
    while k < n:
        if out[k] is None or out[k] in ("latency", "approach"):
            k += 1
            continue
        j = k
        while j < n and out[j] is not None:
            j += 1
        if k > 0 and out[k - 1] is None and j < n and (j - k) <= LAG_S:
            moved = False
            for m in range(k, min(j + 1, n)):
                a, b = samples[m - 1], samples[m]
                if a["ahead"] != b["ahead"] or a["members"] != b["members"]:
                    moved = True
                elif a["ahead"] and b["ahead"] and abs((b["along"] - b["behind_m"]) - (a["along"] - a["behind_m"])) > 0.15:
                    moved = True
            if moved:
                for m in range(k, j):
                    out[m] = "lag"
        k = j
    return list(zip(out, facts)) if detail else out


# ---------------------------------------------------------------------------------------------- offline solver replays
def _replay_inputs(folder):
    from PIL import Image
    from sgi.geometry import Camera, RobotPose
    from sgi.inputs import ObservationData
    from sgi.types import Constraint, DetectedPerson, GroundedTarget, Point
    req = json.load(open(os.path.join(folder, "request.json")))
    c = req["camera"]
    cam = Camera(c["fx"], c["fy"], c["cx"], c["cy"], c["width"], c["height"], np.array(c["T_world_camera"], float))
    r = req["robot"]
    obs = ObservationData(np.asarray(Image.open(os.path.join(folder, "rgb.png")).convert("RGB")),
                          np.load(os.path.join(folder, "depth.npy")), cam, RobotPose(r["x"], r["y"], r["yaw"]))
    inf = json.load(open(os.path.join(folder, "answer.json")))["inference"]
    t = inf["target"]
    target = GroundedTarget(point=Point(**t["point"]), pixel=tuple(t["pixel"]), world=tuple(t["world"]))
    people = [DetectedPerson(number=p["number"], xy=tuple(p["xy"]), yaw=p.get("yaw"), pixel=tuple(p["pixel"]),
                             label=p.get("label")) for p in inf["people"]]
    cons = [Constraint(**c["constraint"]) for c in inf["constraints"]]
    return obs, target, people, cons


def replay_first(e, standoff=None, oracle_come_after=False, no_face=False):
    """sgi.solver.solve on the logged first answer of a C arm, with one rule changed: the pose."""
    from sgi.config import load_config
    from sgi.solver import solve
    from sgi.stages.goal import standard_goal
    from sgi.types import Constraint, DetectedPerson
    th = load_config(str(paths.REPO / "configs" / "default.yaml")).thresholds
    if standoff is not None:
        th = th.model_copy(update={"standoff_m": standoff})
    obs, target, people, cons = _replay_inputs(os.path.join(e["_dir"], "001"))
    if oracle_come_after:          # come_after = everyone waiting when asked; the unseen added where they stood
        sc = scenario(e["scenario"])
        slots = [(s["x"], s["y"]) for s in sc["slots"][: sc["members0"]]]
        nums, nxt = [], max((p.number for p in people), default=0) + 1
        for sx, sy in slots:
            near = min(people, key=lambda p: math.hypot(p.xy[0] - sx, p.xy[1] - sy), default=None)
            if near is not None and math.hypot(near.xy[0] - sx, near.xy[1] - sy) <= 0.75 and near.number not in nums:
                nums.append(near.number)
            else:
                people.append(DetectedPerson(number=nxt, xy=(sx, sy), yaw=math.pi / 2, pixel=(0.0, 0.0)))
                nums.append(nxt)
                nxt += 1
        cons = [c for c in cons if c.kind != "come_after"] + [Constraint(because="oracle", kind="come_after", people=nums)]
    if no_face:
        cons = [c for c in cons if c.kind != "face"]
    s = solve(obs, target, people, cons, standard_goal(obs, target, th), th)
    return list(s.pose_world), s.note.keys.turn_m


def first_valid(e, pose):
    x = min(e["samples"], key=lambda s: abs(s["t"] - e["answers"][0]["asked_t"]))
    j = judge_goal(scenario(e["scenario"]), x, pose)
    return j["pos"] and j["head"], j


# ---------------------------------------------------------------------------------------------- goal-stream replays (CT family)
def goal_replay(e, moved=MOVED_M, turned=TURNED, shift_up=0.0, front_fix=False):
    """Replay the logged answer and re-solve stream under a changed rule and judge the HELD GOAL against the line of
    every sample after the first answer (the robot is assumed to stand at its goal: an upper bound that ignores
    travel). Returns (samples judged, samples whose goal would be in line)."""
    sc = scenario(e["scenario"])
    line = sc["line"]
    evs = [(a["delivered_t"], a.get("pose_world"), "answer", a.get("status"), None) for a in e["answers"][:1]]
    evs += [(r["delivered_t"], r.get("pose_world"), "resolve", r.get("status"), r) for r in e["resolves"]]
    evs.sort(key=lambda v: v[0])
    first_t = evs[0][0]
    goal, k, n, ok, was_first = None, 0, 0, 0, False
    for x in e["samples"]:
        while k < len(evs) and evs[k][0] <= x["t"] + 1e-6:
            t, pose, src, status, r = evs[k]
            k += 1
            if status != "ok" or pose is None:
                continue
            pose = list(pose)
            if r is not None and r.get("in_turn") and shift_up:
                a_g, s_g, up = line.project(pose[0], pose[1])
                pose[0] += shift_up * math.cos(up)
                pose[1] += shift_up * math.sin(up)
            if front_fix and r is not None and not r.get("in_turn") and was_first:
                # nobody left to come after and the robot was first: hold the front place, facing the counter
                pose = [line.s0[0], line.s0[1], math.atan2(line.face0[1] - line.s0[1], line.face0[0] - line.s0[0])]
            if goal is None or math.hypot(pose[0] - goal[0], pose[1] - goal[1]) > moved or off(pose[2], goal[2]) > turned:
                goal = pose
        if x["t"] < first_t or goal is None:
            continue
        was_first = was_first or x["ahead"] == 0
        n += 1
        j = judge_goal(sc, x, goal)
        if x["ahead"]:
            ok += j["pos"] and j["head"]
        else:
            ok += j["pos"] and j["head"]
    return n, ok


# ---------------------------------------------------------------------------------------------- the report
ARM_ORDER = ["CT", "CTv", "CT0", "C1", "Cq", "R1", "Rq", "Rq0", "RT", "RTw"]
GROUP_ORDER = ["late start", "model answer", "no update (arm)", "re-solve rules", "R-track rule", "execution",
               "sim / scoring"]


def run_all(eps):
    rows = []
    for e in eps:
        cl = classify(e)
        rows.append((e, cl))
    return rows


def periods(cl):
    """Runs of consecutive failure samples: (start, end, dominant class)."""
    out, k = [], 0
    while k < len(cl):
        if cl[k] is None:
            k += 1
            continue
        j = k
        while j < len(cl) and cl[j] is not None:
            j += 1
        out.append((k, j, collections.Counter(cl[k:j]).most_common(1)[0][0]))
        k = j
    return out


def report():
    eps = episodes()
    rows = run_all(eps)
    n_samples = sum(len(e["samples"]) for e, _ in rows)
    fail = collections.Counter(c for _, cl in rows for c in cl if c)
    tot = sum(fail.values())
    post = tot - fail["latency"]
    print(f"episodes {len(rows)}, samples {n_samples}, failure samples {tot} ({tot / n_samples:.2f}), "
          f"after the first answer {post}")
    print(collections.Counter((e["_model"], e["arm"]) for e, _ in rows))
    ep_with = collections.Counter()
    for e, cl in rows:
        for c in set(cl) - {None}:
            ep_with[c] += 1
    print("\n| Class | Group | Seconds | Share of failure time | Share after first answer | Episodes |")
    print("|---|---|---|---|---|---|")
    for c, v in fail.most_common():
        print(f"| {c} | {GROUP[c]} | {v} | {v / tot:.3f} | {'-' if c == 'latency' else f'{v / post:.3f}'} | {ep_with[c]} |")
    # per arm: share of the arm's samples by group, and top classes
    print("\n| Arm | Episodes | In line | " + " | ".join(GROUP_ORDER) + " | Top classes (share of arm time) |")
    print("|---|---|---|" + "---|" * len(GROUP_ORDER) + "---|")
    for arm in ARM_ORDER:
        sub = [(e, cl) for e, cl in rows if e["arm"] == arm]
        if not sub:
            continue
        ns = sum(len(cl) for _, cl in sub)
        g = collections.Counter(GROUP[c] for _, cl in sub for c in cl if c)
        cc = collections.Counter(c for _, cl in sub for c in cl if c)
        inl = sum(1 for _, cl in sub for c in cl if c is None)
        top = ", ".join(f"{c} {v / ns:.2f}" for c, v in cc.most_common(4))
        print(f"| {arm} | {len(sub)} | {inl / ns:.2f} | " + " | ".join(f"{g[k] / ns:.2f}" for k in GROUP_ORDER) + f" | {top} |")
    # per model x arm
    print("\n| Model | Arm | Episodes | In line | " + " | ".join(GROUP_ORDER) + " | Top classes |")
    print("|---|---|---|---|" + "---|" * len(GROUP_ORDER) + "---|")
    for m in MODELS:
        for arm in ARM_ORDER:
            sub = [(e, cl) for e, cl in rows if e["arm"] == arm and e["_model"] == m]
            if not sub:
                continue
            ns = sum(len(cl) for _, cl in sub)
            g = collections.Counter(GROUP[c] for _, cl in sub for c in cl if c)
            cc = collections.Counter(c for _, cl in sub for c in cl if c)
            inl = sum(1 for _, cl in sub for c in cl if c is None)
            top = ", ".join(f"{c} {v / ns:.2f}" for c, v in cc.most_common(3))
            print(f"| {m} | {arm} | {len(sub)} | {inl / ns:.2f} | " + " | ".join(f"{g[k] / ns:.2f}" for k in GROUP_ORDER) + f" | {top} |")
    # per scenario for CT
    print("\n| Scenario | CT episodes | In line | Top classes (CT) |")
    print("|---|---|---|---|")
    for s in EVAL + DEV:
        sub = [(e, cl) for e, cl in rows if e["arm"] == "CT" and e["scenario"] == s]
        ns = sum(len(cl) for _, cl in sub)
        cc = collections.Counter(c for _, cl in sub for c in cl if c)
        inl = sum(1 for _, cl in sub for c in cl if c is None)
        print(f"| {s} | {len(sub)} | {inl / ns:.2f} | " + ", ".join(f"{c} {v / ns:.2f}" for c, v in cc.most_common(4)) + " |")
    first_answer_table(eps)
    extras(eps, rows)
    counterfactuals(eps, rows)


def first_answer_table(eps):
    """The C arms' first answer, once per (model, scenario, seed), and R's."""
    print("\n| Model | First answers (C) | come_after has the last | names only the front | no come_after | usable place in turn | right when asked (C) | right when asked (R) | people matched to members | target on a person |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        seen, rseen = {}, {}
        for e in eps:
            if e["_model"] != m:
                continue
            key = (e["scenario"], e["seed"])
            if e["arm"] in ("C1", "Cq", "CT", "CTv") and key not in seen:
                seen[key] = (first_answer_facts(e), answer_valid_when_asked(e, 0)[0])
            if e["arm"] in ("R1", "Rq", "RT", "RTw", "Rq0") and key not in rseen:
                rseen[key] = answer_valid_when_asked(e, 0)[0]
        fa = [v[0] for v in seen.values()]
        n = len(fa)
        print(f"| {m} | {n} | {sum(f['ca_has_last'] for f in fa)} | {sum(f['ca_front_only'] for f in fa)} | "
              f"{sum(not f['come_after'] for f in fa)} | {sum(f['usable_turn'] for f in fa)} | "
              f"{sum(v[1] for v in seen.values())} | {sum(rseen.values())} of {len(rseen)} | "
              f"{sum(f['members_seen'] for f in fa)} of {sum(f['members'] for f in fa)} | "
              f"{sum(bool(f.get('target_on_person')) for f in fa)} |")


def quality(fa):
    if not fa.get("come_after"):
        return "no come_after"
    if fa["ca_non_members"]:
        return "names a non-member"
    if fa["ca_front_only"]:
        return "front only"
    return "has the last" if fa["ca_has_last"] else "misses the last"


def extras(eps, rows):
    # the largest class after the first answer, per episode
    dom = collections.Counter()
    for e, cl in rows:
        cc = collections.Counter(c for c in cl if c and c not in ("latency",))
        if cc:
            dom[cc.most_common(1)[0][0]] += 1
    print("\\nlargest post-answer class per episode:", dom.most_common())
    # failure periods (runs) by dominant class
    per = collections.Counter(p[2] for _, cl in rows for p in periods(cl))
    print("failure periods by dominant class:", per.most_common())
    # no usable place in turn, by the first answer's come_after
    print("\\n| Arm | fa_noturn seconds | " + " | ".join(["no come_after", "names a non-member", "front only", "misses the last", "has the last"]) + " |")
    print("|---|---|---|---|---|---|---|")
    for arm in ("CT", "CTv", "C1", "Cq"):
        cc = collections.Counter()
        for e, cl in rows:
            if e["arm"] == arm:
                q = quality(first_answer_facts(e))
                cc[q] += sum(c == "fa_noturn" for c in cl)
        print(f"| {arm} | {sum(cc.values())} | " + " | ".join(str(cc[k]) for k in ["no come_after", "names a non-member", "front only", "misses the last", "has the last"]) + " |")
    # failure time after the robot left the sim's queue (moved > 0.8 m off its axis), by arm
    print("\\n| Arm | Episodes the robot left the queue | Failure seconds after it left | of which it was first before |")
    print("|---|---|---|---|")
    for arm in ARM_ORDER:
        n = secs = first = 0
        for e, cl in rows:
            if e["arm"] != arm:
                continue
            left = [j["t"] for j in e["joins"] if j["event"] == "left"]
            if not left:
                continue
            n += 1
            t_left = left[0]
            was_first = any(x["ahead"] == 0 and x["t"] < t_left for x in e["samples"] if x["joined"])
            s = sum(1 for x, c in zip(e["samples"], cl) if c and x["t"] >= t_left)
            secs += s
            first += s if was_first else 0
        if n:
            print(f"| {arm} | {n} | {secs} | {first} |")
    # latency per model
    print("\\n| Model | first answer latency, median (C) | (R) | latency seconds share of all samples (C arms) |")
    print("|---|---|---|---|")
    for m in MODELS:
        lc = [e["answers"][0]["latency_s"] for e, _ in rows if e["_model"] == m and e["arm"] == "C1"]
        lr = [e["answers"][0]["latency_s"] for e, _ in rows if e["_model"] == m and e["arm"] == "R1"]
        sub = [(e, cl) for e, cl in rows if e["_model"] == m and e["arm"].startswith("C")]
        ns = sum(len(cl) for _, cl in sub)
        la = sum(c == "latency" for _, cl in sub for c in cl)
        print(f"| {m} | {np.median(lc):.1f} | {np.median(lr):.1f} | {la / ns:.2f} |")


def rescore(x, band_hi=1.2, head=45.0, first_m=1.6, clear=0.55):
    if x["ahead"]:
        return x["behind_m"] >= 0 and 0.5 <= x["gap_m"] <= band_hi and x["heading_off_deg"] <= head and x["nearest_m"] >= clear
    return x["to_slot0_m"] <= first_m and x["heading_off_deg"] <= head and x["nearest_m"] >= clear


def counterfactuals(eps, rows):
    print("\n## Counterfactuals")
    # scoring tolerances
    print("\n| Arm | In line (frozen) | band to 1.5 m | heading 60 deg | first within 0.6 m | on the axis (side <= 0.6 m) | both strict | in line while first (share of in-line time) | in line beside the line (share) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm in ARM_ORDER:
        xs = [x for e, _ in rows if e["arm"] == arm for x in e["samples"]]
        if not xs:
            continue
        base = sum(rescore(x) for x in xs)
        sc = {e["_path"]: scenario(e["scenario"]) for e, _ in rows if e["arm"] == arm}
        xs2 = [(x, scenario(e["scenario"])["line"]) for e, _ in rows if e["arm"] == arm for x in e["samples"]]
        on_axis = lambda x, line: (not x["ahead"]) or abs(line.project(*x["robot"][:2])[1]) <= 0.6
        ax = sum(rescore(x) and on_axis(x, ln) for x, ln in xs2)
        both = sum(rescore(x, first_m=0.6) and on_axis(x, ln) for x, ln in xs2)
        beside = sum(rescore(x) and not on_axis(x, ln) for x, ln in xs2)
        print(f"| {arm} | {base / len(xs):.3f} | {sum(rescore(x, band_hi=1.5) for x in xs) / len(xs):.3f} | "
              f"{sum(rescore(x, head=60) for x in xs) / len(xs):.3f} | {sum(rescore(x, first_m=0.6) for x in xs) / len(xs):.3f} | "
              f"{ax / len(xs):.3f} | {both / len(xs):.3f} | "
              f"{sum(1 for x in xs if x['in_line'] and not x['ahead']) / max(1, base):.2f} | {beside / max(1, base):.2f} |")
    # goal-stream replays for the CT family
    print("\n| Arm | Samples after the answer | goal in line, as run (0.3 m / 25 deg) | replace past 0.1 m / 10 deg | always replace | standoff 0.85 m (shift 0.15 m up the line) | hold the front once first | all three |")
    print("|---|---|---|---|---|---|---|---|")
    for arm in ("CT", "CTv"):
        tot = collections.Counter()
        for e, _ in rows:
            if e["arm"] != arm:
                continue
            for name, kw in (("base", {}), ("thr", dict(moved=0.1, turned=10.0)), ("always", dict(moved=-1, turned=-1)),
                             ("shift", dict(shift_up=0.15)), ("front", dict(front_fix=True)),
                             ("all", dict(moved=0.1, turned=10.0, shift_up=0.15, front_fix=True))):
                n, ok = goal_replay(e, **kw)
                tot[name] += ok
                tot["n"] = tot["n"] + n if name == "base" else tot["n"]
        n = tot["n"]
        print(f"| {arm} | {n} | {tot['base'] / n:.3f} | {tot['thr'] / n:.3f} | {tot['always'] / n:.3f} | {tot['shift'] / n:.3f} | {tot['front'] / n:.3f} | {tot['all'] / n:.3f} |")
    # solver replays of the first answer
    print("\n| Model | First answers | right when asked | standoff 0.85 m | come_after = true members | no face constraint | both (oracle come_after, no face) |")
    print("|---|---|---|---|---|---|---|")
    for m in MODELS:
        firsts = {}
        for e in eps:
            if e["_model"] == m and e["arm"] == "C1":
                firsts[(e["scenario"], e["seed"])] = e
        cnt = collections.Counter()
        for key, e in sorted(firsts.items()):
            cnt["n"] += 1
            cnt["base"] += first_valid(e, e["answers"][0]["pose_world"])[0]
            cnt["so"] += first_valid(e, replay_first(e, standoff=0.85)[0])[0]
            cnt["oc"] += first_valid(e, replay_first(e, oracle_come_after=True)[0])[0]
            cnt["nf"] += first_valid(e, replay_first(e, no_face=True)[0])[0]
            cnt["both"] += first_valid(e, replay_first(e, oracle_come_after=True, no_face=True)[0])[0]
        print(f"| {m} | {cnt['n']} | {cnt['base']} | {cnt['so']} | {cnt['oc']} | {cnt['nf']} | {cnt['both']} |")


# ---------------------------------------------------------------------------------------------- printing
def timeline(e, t0=0.0, t1=1e9, resolves=True):
    """One line a sample, with the goal-setting events between, for reading a case."""
    sc = scenario(e["scenario"])
    cl = classify(e)
    evs = goal_events(e)
    k = 0
    print(f"# {e['_path']}  {e['summary'].get('share_in_line'):.2f} in line; first answer at "
          f"{e['answers'][0]['delivered_t']:.1f} s")
    if e["arm"].startswith("C"):
        print("# first answer:", {k: v for k, v in first_answer_facts(e).items() if k != "kinds"})
    for x, c in zip(e["samples"], cl):
        while k < len(evs) and evs[k]["t"] <= x["t"] + 1e-6:
            ev = evs[k]
            k += 1
            if ev["t"] < t0 or ev["t"] > t1 or (ev["src"] == "resolve" and not resolves):
                continue
            if ev["src"] == "resolve":
                r = e["resolves"][ev["i"]]
                extra = f"turn={r.get('in_turn')} joined={r.get('joined')} dropped={r.get('dropped')} n={r.get('people')}"
            else:
                extra = f"answer {ev['i'] + 1} asked {e['answers'][ev['i']]['asked_t']:.1f} {e['answers'][ev['i']]['status']}"
            pose = ev["pose"] and [round(v, 2) for v in ev["pose"]]
            print(f"   {'*' if ev['acc'] else ' '} {ev['src'][:3]} {ev['t']:.1f} {pose} {extra}")
        if x["t"] < t0 or x["t"] > t1:
            continue
        if x["ahead"]:
            a_l, s_l, _ = last_ahead(sc, x)
            line = f"L@{a_l:.2f}/{s_l:+.2f} gap {x['gap_m']:.2f} beh {x['behind_m']:+.2f}"
        else:
            line = f"FIRST to0 {x['to_slot0_m']:.2f}"
        a_r, s_r, _ = sc["line"].project(*x["robot"][:2])
        print(f"{x['t']:5.1f} R({x['robot'][0]:.2f},{x['robot'][1]:.2f}) a{a_r:.2f}/{s_r:+.2f} {x['state'][:4]} "
              f"{'J' if x['joined'] else '-'} m{x['members']} a{x['ahead']} {line} h{x['heading_off_deg']:.0f} "
              f"n{x['nearest_m']:.2f} G{x['goal']} {'IN' if x['in_line'] else c}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--episode")
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t1", type=float, default=1e9)
    p.add_argument("--quiet-resolves", action="store_true")
    p.add_argument("--matrix", action="store_true")
    args = p.parse_args(argv)
    if args.matrix:
        matrix()
        return 0
    if args.episode:
        m, scen, seed, arm = args.episode.strip("/").split("/")[-4:]
        (e,) = [x for x in episodes([scen], [m]) if x["_path"] == f"{m}/{scen}/{seed}/{arm}"]
        timeline(e, args.t0, args.t1, not args.quiet_resolves)
        return 0
    report()
    return 0



MERGE = {"latency": "latency", "approach": "approach", "fa_noturn": "no place in turn", "fa_off": "beside line",
         "fa_counter": "at counter", "fa_ahead": "ahead (cut-in spot)", "fa_far": "far/close", "fa_close": "far/close",
         "fa_front_goal": "front: wrong goal", "heading_goal": "heading (goal)", "stale_once": "stale", "stale_reask": "stale",
         "ct_ran_out": "line ran out", "ct_ran_out_front": "line ran out", "hysteresis": "replace rule",
         "near_miss": "re-solve other", "ct_wrong_back": "re-solve other", "ct_wrong_front": "re-solve other",
         "ct_off": "re-solve other", "ct_front_turn": "re-solve other"}
COLS = ["latency", "approach", "no place in turn", "beside line", "at counter", "ahead (cut-in spot)", "far/close",
        "front: wrong goal", "heading (goal)", "stale", "line ran out", "replace rule", "re-solve other", "execution",
        "sim / scoring", "R-track"]


def col_of(c):
    if c in MERGE:
        return MERGE[c]
    if c.startswith("rt_"):
        return "R-track"
    return "execution" if GROUP[c] == "execution" else "sim / scoring"


def matrix():
    eps = episodes()
    rows = run_all(eps)
    print("| Model | Arm | Eps | In line | " + " | ".join(COLS) + " |")
    print("|---|---|---|---|" + "---|" * len(COLS))
    for m in MODELS + ["all"]:
        for arm in ARM_ORDER:
            sub = [(e, cl) for e, cl in rows if e["arm"] == arm and (m == "all" or e["_model"] == m)]
            if not sub:
                continue
            ns = sum(len(cl) for _, cl in sub)
            cc = collections.Counter(col_of(c) for _, cl in sub for c in cl if c)
            ep = collections.Counter(col for _, cl in sub for col in {col_of(c) for c in cl if c})
            inl = sum(1 for _, cl in sub for c in cl if c is None)
            cells = [f"{cc[k] / ns:.2f}" + (f" ({ep[k]})" if cc[k] else "") if cc[k] else "-" for k in COLS]
            print(f"| {m} | {arm} | {len(sub)} | {inl / ns:.2f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    sys.exit(main())
