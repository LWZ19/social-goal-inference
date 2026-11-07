"""Stress perturbations S1-S4 of the dumped tracks (second ideas round, idea 1; frozen with the arm). Each is a
function (cell, ticks) -> ticks, seeded per cell, applied before any resolver sees the tracks. Truth is untouched.

- S1 noise: Gaussian per tick, sigma 0.10 m ("S1") or 0.20 m ("S1b").
- S2 lagging members: while in the queue, each member's position is their own position L seconds earlier (never
  before they joined this stint); L ~ U(0, 6) s, redrawn at each service; with probability 0.2, L is the time to the
  next service plus U(0, 2) s (they wait for a second service).
- S3 restless bystanders: an Ornstein-Uhlenbeck drift (stationary sd 0.25 m, tau 8 s) plus one excursion of 0.8 m
  towards the target point (2 s there, 4 s stay, 2 s back) at a time drawn from U(asked, delivered + 30 s).
- S4 stale and switched: a person outside the robot's 90 deg view (or beyond 10 m) is held where last seen (never
  seen: where they stood at the photo); 2% of tracks a minute swap ids with the nearest other track within 1 m.
"""
from __future__ import annotations

import math
import random
import zlib

from replay_lib import dump_truth, robot_at
from sgi.resolve import Track


def _rng(cell, tag):
    return random.Random(zlib.crc32(f"{cell.model}/{cell.scenario}/{cell.seed}/{tag}".encode()))


def s1(sigma):
    def f(cell, ticks):
        rng = _rng(cell, f"S1{sigma}")
        return [(t, [Track(id=tr.id, xy=(tr.xy[0] + rng.gauss(0, sigma), tr.xy[1] + rng.gauss(0, sigma)), yaw=tr.yaw)
                     for tr in trs], lg) for t, trs, lg in ticks]
    return f


def _services(cell):
    fr = cell.dump["frames"]
    out, prev = [], None
    for f in fr:
        q = {p["id"] for p in f["people"] if p.get("in_queue")}
        if prev is not None and prev - q:
            out.append(f["t"])
        prev = q
    return out


def s2(cell, ticks):
    from replay_lib import dump_tracks
    rng = _rng(cell, "S2")
    sv = _services(cell)
    lag = {}                                       # (person, epoch) -> L
    joined_at = {}

    def epoch(t):
        return sum(1 for s in sv if s <= t)

    def lag_of(pid, t):
        k = (pid, epoch(t))
        if k not in lag:
            if rng.random() < 0.2:
                nxt = next((s for s in sv if s > t), t + 15.0)
                lag[k] = (nxt - t) + rng.uniform(0, 2)
            else:
                lag[k] = rng.uniform(0, 6)
        return lag[k]

    out = []
    for t, trs, lg in ticks:
        q, _ = dump_truth(cell.dump, t)
        for tr in trs:
            if tr.id in q:
                joined_at.setdefault(tr.id, t)
            else:
                joined_at.pop(tr.id, None)
        new = []
        then_cache = {}
        for tr in trs:
            if tr.id in q:
                t_then = max(t - lag_of(tr.id, t), joined_at[tr.id])
                if t_then not in then_cache:
                    then_cache[t_then] = {x.id: x for x in dump_tracks(cell.dump, t_then)}
                old = then_cache[t_then].get(tr.id, tr)
                new.append(Track(id=tr.id, xy=old.xy, yaw=old.yaw))
            else:
                new.append(tr)
        out.append((t, new, lg))
    return out


def s3(cell, ticks):
    rng = _rng(cell, "S3")
    tau, sd, dt = 8.0, 0.25, 0.5
    a = math.exp(-dt / tau)
    drift = {b: [0.0, 0.0] for b in cell.bystanders}
    t_ex = {b: rng.uniform(cell.asked_t, cell.delivered_t + 30.0) for b in cell.bystanders}
    out = []
    for t, trs, lg in ticks:
        new = []
        for tr in trs:
            if tr.id not in cell.bystanders:
                new.append(tr)
                continue
            d = drift[tr.id]
            for i in (0, 1):
                d[i] = a * d[i] + math.sqrt(1 - a * a) * sd * rng.gauss(0, 1)
            x, y = tr.xy[0] + d[0], tr.xy[1] + d[1]
            u = t - t_ex[tr.id]
            if 0.0 <= u <= 8.0:                    # 2 s out, 4 s stay, 2 s back
                w = min(u / 2.0, 1.0, (8.0 - u) / 2.0)
                gx, gy = cell.target[0] - tr.xy[0], cell.target[1] - tr.xy[1]
                g = math.hypot(gx, gy) or 1.0
                x, y = x + 0.8 * w * gx / g, y + 0.8 * w * gy / g
            new.append(Track(id=tr.id, xy=(x, y), yaw=tr.yaw))
        out.append((t, new, lg))
    return out


def s4(cell, ticks):
    rng = _rng(cell, "S4")
    seen = {}
    first = {tr.id: tr for tr in ticks[0][1]} if ticks else {}
    swap = {}                                      # true id -> reported id
    p_swap = 0.02 / 120.0                          # per track per 0.5 s tick
    out = []
    for t, trs, lg in ticks:
        s = [x for x in cell.episode["samples"] if x["t"] <= t + 1e-9]
        rob = (s[-1] if s else cell.episode["samples"][0])["robot"]
        rx, ry = robot_at(cell.episode, t) if t >= cell.delivered_t else tuple(cell.episode["samples"][0]["robot"][:2])
        yaw = rob[2]
        new = []
        held_ids = set()
        for tr in trs:
            dx, dy = tr.xy[0] - rx, tr.xy[1] - ry
            ang = abs((math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi)
            if ang <= math.radians(45) and math.hypot(dx, dy) <= 10.0:
                seen[tr.id] = tr
                new.append(tr)
            else:
                held = seen.get(tr.id, first.get(tr.id, tr))
                new.append(Track(id=tr.id, xy=held.xy, yaw=held.yaw))
                held_ids.add(tr.id)
        for tr in list(new):
            if rng.random() < p_swap:
                near = [o for o in new if o.id != tr.id and math.dist(o.xy, tr.xy) <= 1.0]
                if near:
                    o = min(near, key=lambda o: math.dist(o.xy, tr.xy))
                    a_, b_ = swap.get(tr.id, tr.id), swap.get(o.id, o.id)
                    swap[tr.id], swap[o.id] = b_, a_
        out.append((t, [Track(id=swap.get(tr.id, tr.id), xy=tr.xy, yaw=tr.yaw) for tr in new], lg))
        if hasattr(cell, "_stale"):                # what a tracker knows: which of its tracks are held, not seen
            cell._stale[round(t, 3)] = {swap.get(i, i) for i in held_ids}
    return out


def chain(*fs):
    def f(cell, ticks):
        for g in fs:
            ticks = g(cell, ticks)
        return ticks
    return f


STRESS = {"S1": s1(0.10), "S1b": s1(0.20), "S2": s2, "S3": s3, "S4": s4,
          "S23": chain(s2, s3), "S123": chain(s2, s3, s1(0.10)), "S1234": chain(s2, s3, s1(0.10), s4)}
