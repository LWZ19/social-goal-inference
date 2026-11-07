"""Round 3 shared helpers: design cells, conditions, and one replay loop for CT4 / CT5 with either decision."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import stress as ST  # noqa: E402
from a_flicker import s2m  # noqa: E402
from ct4 import CT4  # noqa: E402
from ct5 import CT5, UNSET, decide  # noqa: E402
from replay_lib import BYSTANDER_SCENES, ORIGINAL_SCENES, TH, cells, dump_tracks, dump_truth, robot_at  # noqa: E402
from sgi import solver as S  # noqa: E402

CONDITIONS = {"clean": None, "S2m": s2m, "S3": ST.s3, "S23m": ST.chain(s2m, ST.s3)}
_CELLS = None


def design_cells():
    global _CELLS
    if _CELLS is None:
        _CELLS = [c for c in cells(roots=("runs/dsv41", "runs/sonnet55_low"),
                                   scenes=BYSTANDER_SCENES + ORIGINAL_SCENES, arm="CT3", seeds=range(10, 45))
                  if c.dump is not None]
    return _CELLS


def ticks_of(cell, stress, every_s=0.5):
    end = cell.episode["samples"][-1]["t"]
    n = int(math.floor((end - cell.asked_t) / every_s)) + 1
    ticks = [(cell.asked_t + k * every_s, dump_tracks(cell.dump, cell.asked_t + k * every_s), None) for k in range(n)]
    cell._stale = {}
    return stress(cell, ticks) if stress is not None else ticks


def turn_last(waiting_ids, tracks, est, robot):
    if not waiting_ids:
        return None
    bodies = {tr.id + 1: S._Body(tr.id + 1, np.array(tr.xy, dtype=np.float64), tr.yaw) for tr in tracks}
    waiting = sorted(i + 1 for i in waiting_ids if i + 1 in bodies)
    if not waiting:
        return None
    turn = S._turn(waiting, set(bodies) - set(waiting), bodies, None, est.tgt, np.array(robot, dtype=np.float64),
                   TH, est.dir)
    return turn.last - 1


def run(cell, stress=None, est_cls=CT5, params=None, decision="threshold", costs=None, hook=None, every_s=0.5):
    """Rows from delivery: t, last, roster, marginals, truth. decision: "threshold" (roster = P >= 0.5, the solver's
    turn rule picks the last) or "cost" (ct5.decide picks whom to come after; the solver gets them and the roster
    members ahead of them)."""
    est = est_cls(cell, params)
    rows = []
    cur = UNSET
    for t, tracks, _ in ticks_of(cell, stress, every_s):
        if hook is not None:
            hook("before", est, t, tracks)
        est.step(t, tracks, every_s)
        if hook is not None:
            hook("after", est, t, tracks)
        if t + 1e-9 < cell.delivered_t:
            continue
        m = est.marginals()
        present = {tr.id for tr in tracks}
        roster = {i for i, v in m.items() if v >= 0.5 and i in present}
        robot = robot_at(cell.episode, t)
        if decision == "threshold":
            last = turn_last(roster, tracks, est, robot)
        else:
            cur, _ = decide(est, present, cur, **(costs or {}))
            if cur is None:
                last = None
            else:
                dj = est.dist_now[cur]
                last = turn_last({i for i in roster if est.dist_now.get(i, np.inf) < dj} | {cur}, tracks, est, robot)
        q, order = dump_truth(cell.dump, t)
        rows.append(dict(t=t, last=last, roster=roster, m=m, q=q, order=order))
    return rows


def classify(row):
    """right / cut (a waiter ahead of the true last: the robot would stand ahead of a waiter) / far (a non-waiter) /
    empty; None when nobody queues."""
    q, order = row["q"], row["order"]
    if not q:
        return None
    true_last = max(q, key=lambda i: order[i])
    last = row["last"]
    if last is None:
        return "empty"
    if last == true_last:
        return "right"
    if last in q:
        return "cut"
    return "far"
