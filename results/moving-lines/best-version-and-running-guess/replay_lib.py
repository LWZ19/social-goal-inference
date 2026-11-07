"""Phase 1b: offline replays of the role part of re-solving (zero calls; research only, nothing here enters src/).

The real `sgi.resolve.Resolver` is driven tick by tick as `sgi.serve.resolve` drives it, with the solver's
place-in-turn rule `sgi.solver._turn` standing in for the full solve (no floor, no depth: the solver's only effect on
roles is which people its turn rule takes as joining, and who it takes as last). Tracks come either from a run's own
logs (`logged`, from the first re-solve on; for checking that the replay reproduces the run) or from the people dumps
(`human_dump/`, 2 Hz from t = 0, robot idle; exact for every arm until its robot first moves).

    from replay_lib import cells, replay, CT3
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sgi import solver as S
from sgi.config import Thresholds
from sgi.resolve import Resolver, Track
from sgi.types import Constraint, DetectedPerson

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
MOVING = paths.MOVING_RUNS                # run folders are named relative to it, e.g. "runs/dsv41"
REPO = paths.REPO
TH = Thresholds()
V2 = dict(readmit=True, line_span_m=0.5, vacate_after_s=5.0, path_memory=True)           # experiment_oracle._V2
CT3 = dict(V2, co_motion_s=5.0, co_motion_follow=True)
CT2 = dict(V2)
BYSTANDER_SCENES = ("cafe_line2_bystanders", "room_coffee_cart_bystanders", "sweet_spot_line2_bystanders")
ORIGINAL_SCENES = ("cafe_line2_moving", "room_coffee_cart_slots", "sweet_spot_line2_moving")


@dataclass
class Cell:
    model: str
    scenario: str
    seed: int
    arm: str
    path: Path
    episode: dict
    people: list
    constraints: list
    target: tuple
    asked_t: float
    delivered_t: float
    bystanders: set
    dump: dict = field(repr=False, default=None)

    @property
    def key(self):
        return (self.model, self.scenario, self.seed)


def cells(roots=("runs/dsv41", "runs/sonnet55_low"), scenes=BYSTANDER_SCENES + ORIGINAL_SCENES, arm="CT3",
          seeds=range(10, 45)):
    """Every run cell with a recorded constraint answer (001/answer.json with a come_after or not)."""
    out = []
    for root in roots:
        for f in sorted((MOVING / root).glob(f"*/seed*/{arm}/episode.json")):
            e = json.loads(f.read_text())
            if e["scenario"] not in scenes or e["seed"] not in seeds:
                continue
            a = f.parent / "001" / "answer.json"
            answers = [x for x in e.get("answers", []) if x.get("delivered_t") is not None]
            if not a.exists() or not answers or not e.get("samples"):
                continue
            ans = json.loads(a.read_text())
            inf = ans.get("inference") or {}
            if ans.get("source") != "constraints" or not inf.get("people") or ans.get("target_world") is None:
                continue
            people = [DetectedPerson(number=p["number"], xy=tuple(p["xy"]), yaw=p.get("yaw"),
                                     pixel=tuple(p.get("pixel") or (0.0, 0.0)), label=p.get("label"))
                      for p in inf["people"]]
            cons = [Constraint(**c["constraint"]) for c in inf.get("constraints", [])]
            dump_f = HERE / "human_dump" / e["scenario"] / f"seed{e['seed']}.json"
            out.append(Cell(model=Path(root).name, scenario=e["scenario"], seed=e["seed"], arm=arm, path=f.parent,
                            episode=e, people=people, constraints=cons, target=tuple(ans["target_world"][:2]),
                            asked_t=answers[0]["asked_t"], delivered_t=answers[0]["delivered_t"],
                            bystanders=set(e.get("bystanders") or []),
                            dump=json.loads(dump_f.read_text()) if dump_f.exists() else None))
    return out


# ------------------------------------------------------------------------------------------------ the world over time
def dump_tracks(dump, t):
    """Tracks at time t from the 2 Hz dump, positions interpolated between frames (yaw from the nearer frame)."""
    fr = dump["frames"]
    dt = dump["every_s"]
    i = min(max(int(math.floor((t - fr[0]["t"]) / dt)), 0), len(fr) - 1)
    j = min(i + 1, len(fr) - 1)
    a, b = fr[i], fr[j]
    w = 0.0 if j == i else min(max((t - a["t"]) / (b["t"] - a["t"]), 0.0), 1.0)
    bj = {p["id"]: p for p in b["people"]}
    out = []
    for p in a["people"]:
        q = bj.get(p["id"], p)
        xy = (p["xy"][0] + w * (q["xy"][0] - p["xy"][0]), p["xy"][1] + w * (q["xy"][1] - p["xy"][1]))
        out.append(Track(id=int(p["id"]), xy=xy, yaw=(p if w < 0.5 else q).get("yaw")))
    return out


def dump_truth(dump, t):
    """(set of ids in the queue, {id: order}) at the frame nearest t."""
    fr = dump["frames"]
    i = min(max(int(round((t - fr[0]["t"]) / dump["every_s"])), 0), len(fr) - 1)
    q = {p["id"]: p["order"] for p in fr[i]["people"] if p.get("in_queue")}
    return set(q), q


def robot_at(episode, t):
    s = episode["samples"]
    prev = s[0]
    for x in s:
        if x["t"] > t:
            if x is s[0]:
                return tuple(x["robot"][:2])
            w = (t - prev["t"]) / max(x["t"] - prev["t"], 1e-9)
            return (prev["robot"][0] + w * (x["robot"][0] - prev["robot"][0]),
                    prev["robot"][1] + w * (x["robot"][1] - prev["robot"][1]))
        prev = x
    return tuple(prev["robot"][:2])


# ------------------------------------------------------------------------------------------------------- the replay
def solve_roles(resolver, people, constraints, robot_xy, target_xy, exempt):
    """What `solve` decides about roles: the come_after it can use, the turn rule's last and joiners."""
    numbers = {p.number for p in people}
    waiting = sorted({n for c in constraints if c.kind == "come_after" for n in c.people if n in numbers})
    if not waiting:
        return None, [], None
    apart = set()
    for c in constraints:
        if c.kind == "keep_out_of_shared_space":
            present = [n for n in c.people if n in numbers]
            if len(present) >= 2:
                apart |= set(present)
    apart |= set(exempt)
    bodies = {p.number: S._Body(p.number, np.array(p.xy, dtype=np.float64), p.yaw) for p in people}
    line = None if resolver.solve_direction is None else np.array(resolver.solve_direction, dtype=np.float64)
    turn = S._turn(waiting, apart, bodies, None, np.array(target_xy, dtype=np.float64),
                   np.array(robot_xy, dtype=np.float64), TH, line)
    return turn.last, list(turn.joined), turn.place


def replay(cell, options=CT3, source="dump", start="delivery", resolver_cls=Resolver, every_s=0.5, end_t=None,
           stress=None):
    """Drive the resolver over the cell's tracks. Returns a list of tick dicts (from delivery on; a photo-time start
    replays the wait first and logs from delivery). `source` 'logged' uses the run's own re-solve tracks and times."""
    e = cell.episode
    if source == "logged":
        rs = [r for r in e["resolves"] if r.get("delivered_t") is not None and r.get("tracks")]
        ticks = [(r["sent_t"], [Track(id=int(i), xy=(x, y)) for i, x, y in r["tracks"]], r) for r in rs]
    else:
        end = end_t if end_t is not None else e["samples"][-1]["t"]
        t0 = cell.asked_t if start == "photo" else cell.delivered_t
        n = int(math.floor((end - t0) / every_s)) + 1
        ticks = [(t0 + k * every_s, dump_tracks(cell.dump, t0 + k * every_s), None) for k in range(n)]
        if stress is not None:                      # a perturbation of the tracks (stress.py), seeded per cell
            ticks = stress(cell, ticks)
    at_answer = dump_tracks(cell.dump, cell.asked_t)
    r = resolver_cls(cell.people, cell.constraints, cell.target, at_answer, **options)
    out = []
    for t, tracks, logged in ticks:
        robot = robot_at(e, t) if t >= cell.delivered_t else tuple(e["samples"][0]["robot"][:2])
        people, cons = r.inputs(tracks, robot, t)
        if r.vacated_pose is not None:
            x, y, _ = r.vacated_pose
            r.record((x, y), True)
            last, joined = None, []
            vac = True
        else:
            vac = False
            ex = r.exempt()
            last_n, joined_n, place = solve_roles(r, people, cons, robot, cell.target, ex)
            if place is None:
                r.record(robot, False, [])
                last, joined = None, []
            else:
                r.record(tuple(place), True, joined_n)
                last, joined = r.track_of(last_n), [r.track_of(n) for n in joined_n]
        if t + 1e-9 < cell.delivered_t:
            continue
        roster = {r.track_of(n) for n in (r._waiting - r.dropped - r.gone)}
        out.append(dict(t=t, last=last, joined=joined, vacated=vac, roster={i for i in roster if i is not None},
                        nonfollowers={r.track_of(n) for n in r.non_followers}, logged=logged))
    return out


# ------------------------------------------------------------------------------------------------------- scoring
def score(cell, ticks, truth="dump"):
    """Role-tracking measures over ticks (seconds): right last, empty while a line stands, bystander as last
    (capture proxy), bystander in roster, member recall; and false drops (non-followers who were in the queue)."""
    tot = dict(ticks=0, right=0, empty=0, bys_last=0, other=0, bys_roster=0, rec_num=0, rec_den=0)
    false_drops, drop_t = set(), {}
    prev_nf = set()
    for k in ticks:
        if truth == "dump":
            q, order = dump_truth(cell.dump, k["t"])
        else:
            s = [x for x in cell.episode["samples"] if x["t"] <= k["t"] + 1e-9]
            q = set((s[-1] if s else cell.episode["samples"][0]).get("queue_ids") or [])
            order = None
        tot["ticks"] += 1
        true_last = max(q, key=lambda i: order[i]) if (q and order) else None
        if k["last"] is None and not k["vacated"]:
            tot["empty" if q else "other"] += 1
        elif k["last"] in cell.bystanders:
            tot["bys_last"] += 1
        elif true_last is not None and k["last"] == true_last:
            tot["right"] += 1
        else:
            tot["other"] += 1
        tot["bys_roster"] += bool(k["roster"] & cell.bystanders)
        tot["rec_num"] += len(k["roster"] & q)
        tot["rec_den"] += len(q)
        for i in k["nonfollowers"] - prev_nf:
            drop_t.setdefault(i, k["t"])
            if i in q:
                false_drops.add(i)
        prev_nf = k["nonfollowers"]
    return tot, false_drops, drop_t
