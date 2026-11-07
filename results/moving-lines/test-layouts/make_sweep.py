"""Candidate queue layouts for the fresh test and the ambiguity sweep (ideas round 2, SHORTLIST.md section 5.3).

Zero model calls. From two base scenes it draws candidate layouts that vary what the frozen bystander scenes held
fixed; a separate, frozen selection (`select_sweep.py`, from `sweep_report.py`'s strata) picks the test set and adds
the stress flags. Drawn per candidate:
- the line's length: 2, 3 or 4 members at the start;
- a lone non-member past the line's end: 0-0.9 m to the side of the axis (side drawn), 0.4-2.2 m past the last
  member along it, facing the counter, sideways or away (facing stratified over the candidates);
- a talking pair 0.75 m apart, its centre 1.0-2.5 m from the line's end at a drawn bearing, outside the line's
  corridor (more than 0.9 m from the axis, or more than 1.0 m beyond its end);
- the service clock: each service takes a time drawn from [12, 20] s (drawn by the sim per episode and service);
- who wears which avatar (the 8 dressed avatars, none repeated; walkers fill what is left).
Kept only if:
- every standing person is on navigable floor and at least 0.6 m from every other;
- the queue slots where the robot and returning people will stand are navigable, and the pair is 0.7 m from them;
- slot n (the robot's or the next person's place) is at least 0.55 m from the non-member;
- the pair is at least 1.8 m from the robot, so it does not fill the picture;
- the robot's first picture shows every member, the non-member and the pair (in the 80 deg view, each at least 40%
  unhidden by a nearer person, disc model);
- a place in line exists at the photo frame: a point 0.5-1.2 m behind the last member and at most 0.6 m off the
  axis that is at least 0.6 m from everyone (the in-line test's band, with a margin).
Slots from n + 1 on that lie within 0.7 m of the non-member, or behind him, jog 0.9 m to the axis's other side (the
pair's other side for one within 0.1 m of the axis): people coming back walk round him, as in a real line. Slot n
never jogs, because the scorer's line runs through it.

Room (room_coffee_cart) is left out: DeepSeek answered "people do not matter" in 6 of 6 room bystander cells, so
its cells test nothing about roles. cafe_line6 shares cafe_line2's stage, axis and robot start, so it adds no base.

Run with the sim's Python, from the repo root:
    PYTHONPATH=sim ~/miniforge3/envs/sgi-habitat/bin/python <this file> --per-base 150 --out DIR
Writes <base>_<kkk>.yaml per candidate and candidates.json (every drawn parameter).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import yaml

from sgi_habitat import run as R
from sgi_habitat.frames import to_hab
from sgi_habitat.world import CameraSpec, WorldSpec, floor_height, make_sim

BASES = {"cafe": "sim/scenarios/cafe_line2_bystanders.yaml", "sweet": "sim/scenarios/sweet_spot_line2_bystanders.yaml"}
AVATARS = ["male_0", "male_1", "male_2", "male_3", "female_0", "female_1", "female_2", "female_3"]
FACINGS = ["counter", "side", "away"]
OFFSET_MAX, PAST = 0.9, (0.4, 2.2)
FIDGET = [5.0, 10.0, 0.35]
SERVICE = [12.0, 20.0]
R_DISC, HALF_FOV, MIN_SEEN = 0.22, 40.0, 0.4


def _rng(base: str, k: int) -> random.Random:
    return random.Random(int(hashlib.sha256(f"sweep-{base}-{k}".encode()).hexdigest()[:12], 16))


def _bearing(o, p):
    return math.degrees(math.atan2(p[1] - o[1], p[0] - o[0]))


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _seen(o, pts, i):
    """Share of person i's bearing interval not behind a nearer person (discs of radius R_DISC)."""
    p, d = pts[i], math.dist(o, pts[i])
    b, w = _bearing(o, p), math.degrees(math.asin(min(1.0, R_DISC / d)))
    n, seen = 60, 0
    for k in range(n):
        a = b - w + 2 * w * (k + 0.5) / n
        hidden = False
        for j, q in enumerate(pts):
            dq = math.dist(o, q)
            if j != i and dq < d and abs(_wrap(a - _bearing(o, q))) <= math.degrees(math.asin(min(1.0, R_DISC / dq))):
                hidden = True
                break
        seen += not hidden
    return seen / n


def place_in_line(last, u, p, people, clear=0.6):
    """A point 0.5-1.2 m behind `last` along u and at most 0.6 m off the axis, at least `clear` from everyone."""
    best = None
    for i in range(15):
        a = 0.5 + 0.05 * i
        for j in range(-12, 13):
            s = 0.05 * j
            q = (last[0] + a * u[0] + s * p[0], last[1] + a * u[1] + s * p[1])
            c = min(math.dist(q, r) for r in people)
            if c >= clear and (best is None or abs(s) + abs(a - 1.0) < best[0]):
                best = (abs(s) + abs(a - 1.0), q, c)
    return best


class Floor:
    def __init__(self, base: dict):
        cam = CameraSpec(**base.get("camera", {}))
        self.sim = make_sim(WorldSpec(None, str(R._repo_path(base["stage"])), cam))
        self.floor_y = floor_height(self.sim)
        self.pf = self.sim.pathfinder

    def ok(self, x: float, y: float, clear: float = 0.2) -> bool:
        h = to_hab(x, y, self.floor_y)
        if not self.pf.is_navigable(h, 0.5):
            return False
        p = self.pf.snap_point(h)
        return math.hypot(float(p[0]) - x, -float(p[2]) - y) < 0.1 and self.pf.distance_to_closest_obstacle(p, 1.0) >= clear


def candidate(base_name: str, base: dict, floor: Floor, k: int, tries: int = 3000):
    rng = _rng(base_name, k)
    (qname, q), = base["queues"].items()
    slots = [(s["x"], s["y"]) for s in q["slots"]]
    counter = tuple(q["slots"][0]["face"])
    ux, uy = slots[1][0] - slots[0][0], slots[1][1] - slots[0][1]
    spacing = math.hypot(ux, uy)
    ux, uy = ux / spacing, uy / spacing                    # front -> back
    px, py = -uy, ux                                       # one side
    robot = (base["robot"]["x"], base["robot"]["y"])
    facing0 = _bearing(robot, base["robot"]["face"])
    barista = [h for h in base["humans"] if h.get("stand") is not None and not h.get("bystander")]
    walker_starts = [tuple(h["start"]) for h in base["humans"] if h.get("start") is not None]
    facing = FACINGS[k % 3]
    for _ in range(tries):
        n = rng.choice([2, 3, 4])
        side, off, d = rng.choice([-1, 1]), rng.uniform(0.0, OFFSET_MAX), rng.uniform(*PAST)
        last = slots[n - 1]
        lone = (last[0] + d * ux + side * off * px, last[1] + d * uy + side * off * py)
        if facing == "counter":
            lone_face = counter
        elif facing == "side":
            s2 = side if off >= 0.1 else rng.choice([-1, 1])
            lone_face = (lone[0] + s2 * px, lone[1] + s2 * py)
        else:
            lone_face = (lone[0] + ux, lone[1] + uy)
        r, psi, phi = rng.uniform(1.0, 2.5), rng.uniform(-math.pi, math.pi), rng.uniform(0, math.pi)
        end = (last[0] + 0.5 * spacing * ux, last[1] + 0.5 * spacing * uy)
        c = (end[0] + r * math.cos(psi), end[1] + r * math.sin(psi))
        c_along = (c[0] - slots[0][0]) * ux + (c[1] - slots[0][1]) * uy
        c_side = (c[0] - slots[0][0]) * px + (c[1] - slots[0][1]) * py
        if abs(c_side) <= 0.9 and c_along <= (n - 1) * spacing + 1.0 + 0.5 * spacing:
            continue                                       # in the line's corridor
        pair = [(c[0] + 0.375 * math.cos(phi), c[1] + 0.375 * math.sin(phi)),
                (c[0] - 0.375 * math.cos(phi), c[1] - 0.375 * math.sin(phi))]
        # slots: the base's; from n + 1 on, those near or behind the non-member jog to the other side
        jog_side = -side if off >= 0.1 else (-1 if c_side > 0 else 1)
        lone_along = (lone[0] - slots[0][0]) * ux + (lone[1] - slots[0][1]) * uy
        qslots, jogging = [dict(s) for s in q["slots"]], False
        for i in range(n + 1, len(qslots)):
            s_xy = (qslots[i]["x"], qslots[i]["y"])
            s_along = (s_xy[0] - slots[0][0]) * ux + (s_xy[1] - slots[0][1]) * uy
            jogging = jogging or math.dist(s_xy, lone) < 0.7 or s_along > lone_along - 0.35
            if jogging:
                qslots[i] = {"x": round(s_xy[0] + jog_side * 0.9 * px, 3), "y": round(s_xy[1] + jog_side * 0.9 * py, 3)}
        members = [(s["x"], s["y"]) for s in qslots[:n]]
        later = [(s["x"], s["y"]) for s in qslots[n:n + 3]]          # where the robot and people coming back stand
        others = [lone] + pair + [tuple(h["stand"]) for h in barista]
        standing = members + others
        if not all(floor.ok(*p) for p in [lone] + pair + later):
            continue
        apart = [math.dist(a, b) for i, a in enumerate(standing) for b in standing[i + 1:]]
        if min(apart) < 0.6 or min(math.dist(p, s) for p in pair for s in later) < 0.7:
            continue
        if math.dist(lone, later[0]) < 0.55 or min(math.dist(lone, s) for s in later[1:]) < 0.5:
            continue
        if min(math.dist(robot, p) for p in standing + later) < 1.0:
            continue
        if min(math.dist(robot, p) for p in pair) < 1.8:      # not filling the robot's picture
            continue
        place = place_in_line(last, (ux, uy), (px, py), standing)
        if place is None or not floor.ok(*place[1], clear=0.3):
            continue
        shown = members + [lone] + pair
        if any(abs(_wrap(_bearing(robot, p) - facing0)) > HALF_FOV for p in shown):
            continue
        seen = [_seen(robot, standing, i) for i in range(len(shown))]
        if min(seen) < MIN_SEEN:
            continue
        n_walkers = max(0, min(len(walker_starts), len(AVATARS) - (n + 3 + len(barista))))
        it = iter(rng.sample(AVATARS, n + 3 + len(barista) + n_walkers))
        humans = [{**{k2: v for k2, v in h.items() if k2 != "avatar"}, "avatar": next(it)} for h in barista]
        humans += [{"avatar": next(it), "queue": qname} for _ in range(n)]
        humans += [{"avatar": next(it), "start": list(walker_starts[i])} for i in range(n_walkers)]
        humans.append({"avatar": next(it), "bystander": True, "stand": [round(lone[0], 3), round(lone[1], 3)],
                       "face": [round(lone_face[0], 3), round(lone_face[1], 3)], "fidget": FIDGET})
        for i, p in enumerate(pair):
            o = pair[1 - i]
            humans.append({"avatar": next(it), "bystander": True, "stand": [round(p[0], 3), round(p[1], 3)],
                           "face": [round(o[0], 3), round(o[1], 3)], "fidget": FIDGET})
        scene = {k2: v for k2, v in base.items() if k2 not in ("humans", "queues")}
        scene["queues"] = {qname: {"service_s": SERVICE, "slots": qslots}}
        scene["humans"] = humans
        params = {"base": base_name, "k": k, "members": n, "offset_m": round(off, 3), "side": side,
                  "past_last_m": round(d, 3), "facing": facing, "pair_centre": [round(c[0], 3), round(c[1], 3)],
                  "pair_from_end_m": round(r, 3), "walkers": n_walkers,
                  "jogged_from": next((i for i in range(n + 1, len(qslots)) if qslots[i] != q["slots"][i]), None),
                  "place_in_line": [round(place[1][0], 3), round(place[1][1], 3)], "place_clear_m": round(place[2], 3),
                  "seen_min": round(min(seen), 2), "lone_avatar": humans[-3]["avatar"]}
        return scene, params
    return None, {"base": base_name, "k": k, "failed": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-base", type=int, default=150)
    ap.add_argument("--start", type=int, default=0, help="first candidate index (to draw more later)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    table = []
    for base_name, path in BASES.items():
        base = yaml.safe_load(Path(path).read_text())
        floor = Floor(base)
        try:
            for k in range(args.start, args.start + args.per_base):
                scene, params = candidate(base_name, base, floor, k)
                name = f"{base_name}_{k:03d}"
                table.append({"name": name, **params})
                if scene is None:
                    print(f"[sweep] {name}: no layout", flush=True)
                    continue
                header = (f"# Ideas round 2 candidate layout {name} (make_sweep.py, zero calls): {params['members']} in "
                          f"line; a non-member {params['offset_m']} m off the axis, {params['past_last_m']} m past the "
                          f"last member, facing {params['facing']}; a talking pair; service every 12-20 s.\n"
                          f"# `bystander: true` is read only by the research scripts.\n")
                (args.out / f"{name}.yaml").write_text(header + yaml.safe_dump(scene, sort_keys=False))
        finally:
            floor.sim.close()
    old = json.loads((args.out / "candidates.json").read_text()) if (args.out / "candidates.json").exists() else []
    names = {t["name"] for t in table}
    table = [t for t in old if t["name"] not in names] + table
    (args.out / "candidates.json").write_text(json.dumps(table, indent=1))
    ok = sum(1 for t in table if not t.get("failed"))
    print(f"{ok} of {len(table)} candidates -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
