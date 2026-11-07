"""Controller's check of ROUND3.md section 6 (draft blocked-place rule; ROUND3.md sha256 9a86fead). Zero model calls:
episodes are built with a client that refuses, only for their navmesh and line.

Part A (design dumps, robot idle at its start): per 1-s frame with someone in the line, is the band free, and if
not, why (people, floor, the corridor), and who blocks (a line member or a non-member).
Part B (closed loop, the spent DeepSeek sweep): per 1-s sample with someone ahead, after the first service that
follows delivery: band_free from the oracle tracks, the draft's `acceptable`, and whether a fallback-credited sample
ever has the robot ahead of someone waiting.

Run with the sim's Python, from the repo root:
    PYTHONPATH=sim ~/miniforge3/envs/sgi-habitat/bin/python <this file> A DUMP_ROOT SCEN_DIR [names...]
    PYTHONPATH=sim ~/miniforge3/envs/sgi-habitat/bin/python <this file> B RUN_ROOT SCEN_DIR [names...]
"""
from __future__ import annotations

import collections
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from sgi_habitat import experiment_oracle as O

BAND = (0.5, 1.2)
CORRIDOR = 0.6
CLEAR = 0.55
STEP = 0.1
HEADING = 45.0


class _NoClient:
    def __getattr__(self, name):
        raise RuntimeError(f"no model calls (client.{name} was used)")


def world(scen_path: Path):
    """(line, navigable(x, y)) of a scenario, from an episode built once and closed."""
    scenario = yaml.safe_load(scen_path.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        ep = O.OracleEpisode(scenario, _NoClient(), Path(tmp), "CT2", 0, 10.0, 0.5, scenario["requests"][0])
        try:
            fm = ep.floor_map
            line = ep.line
        finally:
            ep.sim.close()
    import base64, io
    from PIL import Image
    mask = np.array(Image.open(io.BytesIO(base64.b64decode(fm["png"])))) > 0
    ox, oy, res = fm["origin"][0], fm["origin"][1], fm["resolution"]

    def nav(x, y):
        c, r = int(round((x - ox) / res)), int(round((y - oy) / res))
        return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])
    return line, nav


def back_unit(line, x, y):
    _, _, up = line.project(x, y)
    return np.array([-math.cos(up), -math.sin(up)])        # from the front towards the back


GRID = [(dx, dy) for dx in np.arange(-1.2, 1.2001, STEP) for dy in np.arange(-1.2, 1.2001, STEP)
        if BAND[0] <= math.hypot(dx, dy) <= BAND[1]]


def band(line, nav, last_xy, humans, last_id):
    """(free, reason, blockers): `humans` is [(id, xy, in_line_member)], the last among them; blockers are the
    people other than the last within 0.55 m of a band point."""
    lx, ly = last_xy
    la = line.project(lx, ly)[0]
    u = back_unit(line, lx, ly)
    geo, geo_floor = [], []
    for dx, dy in GRID:
        x, y = lx + dx, ly + dy
        if line.project(x, y)[0] < la:                       # behind the last along the line
            continue
        if abs(dx * u[1] - dy * u[0]) > CORRIDOR:            # within 0.6 m of the line through the last
            continue
        geo.append((x, y))
        if nav(x, y):
            geo_floor.append((x, y))
    free_pts = [p for p in geo_floor if all(math.dist(p, h[1]) >= CLEAR for h in humans)]
    if free_pts:
        return True, "free", []
    if not geo_floor:
        return False, "floor" if geo else "geometry", []
    blockers = sorted({h[0] for p in geo_floor for h in humans if h[0] != last_id and math.dist(p, h[1]) < CLEAR})
    if not blockers:
        return False, "last_only", []                      # only the last's own 0.55 m clearance rules points out
    return False, "people", blockers


def part_a(dump_root: Path, scen_dir: Path, names):
    rows = collections.defaultdict(collections.Counter)
    for scen in names:
        files = sorted((dump_root / scen).glob("seed*.json"))
        if not files:
            continue
        line, nav = world(scen_dir / f"{scen}.yaml")
        for f in files:
            d = json.loads(f.read_text())
            first_service = None
            prev_front = None
            for fr in d["frames"]:
                if abs(fr["t"] - round(fr["t"])) > 1e-6:
                    continue
                q = [p for p in fr["people"] if p["in_queue"]]
                if not q:
                    continue
                front = min(q, key=lambda p: p["order"])["id"]
                if prev_front is not None and front != prev_front and first_service is None:
                    first_service = fr["t"]
                prev_front = front
                last = max(q, key=lambda p: line.project(*p["xy"])[0])
                last_order = max(q, key=lambda p: p["order"])
                humans = [(p["id"], tuple(p["xy"]), p["in_queue"]) for p in fr["people"]]
                free, why, blockers = band(line, nav, tuple(last["xy"]), humans, last["id"])
                c = rows[scen]
                for part in (("all",) + (("after",) if first_service is not None else ())):
                    c[part, "n"] += 1
                    c[part, why] += 1
                    if why == "people":
                        member = any(p["in_queue"] for p in fr["people"] if p["id"] in blockers)
                        nonmember = any(not p["in_queue"] for p in fr["people"] if p["id"] in blockers)
                        c[part, "blk_member"] += member
                        c[part, "blk_nonmember"] += nonmember
                    c[part, "last_order_ne_along"] += last["id"] != last_order["id"]
    print("Part A: design dumps (robot idle). Per scene: frames with someone in line, share with the band not free,")
    print("by reason (people / floor / geometry), and blockers; 'after' = from the first service on.\n")
    print("| scene | part | frames | not free | people | last only | floor | geom | blocker member | blocker non-member | "
          "last by order != by along |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    tot = collections.Counter()
    for scen, c in rows.items():
        for part in ("all", "after"):
            n = c[part, "n"]
            if not n:
                continue
            nf = n - c[part, "free"]
            print(f"| {scen} | {part} | {n} | {nf / n:.3f} | {c[part, 'people'] / n:.3f} | {c[part, 'last_only'] / n:.3f} | {c[part, 'floor'] / n:.3f} | "
                  f"{c[part, 'geometry'] / n:.3f} | {c[part, 'blk_member']} | {c[part, 'blk_nonmember']} | "
                  f"{c[part, 'last_order_ne_along']} |")
            for k, v in c.items():
                if k[0] == part:
                    tot[k] += v
    for part in ("all", "after"):
        n = tot[part, "n"]
        if n:
            print(f"| ALL | {part} | {n} | {(n - tot[part, 'free']) / n:.3f} | {tot[part, 'people'] / n:.3f} | {tot[part, 'last_only'] / n:.3f} | "
                  f"{tot[part, 'floor'] / n:.3f} | {tot[part, 'geometry'] / n:.3f} | {tot[part, 'blk_member']} | "
                  f"{tot[part, 'blk_nonmember']} | {tot[part, 'last_order_ne_along']} |")


def part_b(run_root: Path, scen_dir: Path, names, arms=("CT2", "CT3", "Gl", "CT3F", "CT4")):
    stats = collections.defaultdict(collections.Counter)
    ahead_cases = []
    for scen in names:
        eps = sorted((run_root / "sweep" / scen).glob("seed*/*/episode.json"))
        eps = [e for e in eps if e.parent.name in arms]
        if not eps:
            continue
        line, nav = world(scen_dir / f"{scen}.yaml")
        for f in eps:
            arm = f.parent.name
            e = json.loads(f.read_text())
            rs = [r for r in e.get("resolves", []) if r.get("truth") is not None and r.get("tracks")]
            if not rs or not e.get("answers"):
                continue
            delivered = e["answers"][0].get("delivered_t") or 0.0
            first_service = None
            for a, b in zip(rs, rs[1:]):
                if b["sent_t"] >= delivered and a["truth"]["order"] and (
                        not b["truth"]["order"] or b["truth"]["order"][0] != a["truth"]["order"][0]):
                    first_service = b["sent_t"]
                    break
            if first_service is None:
                continue
            times = np.array([r["sent_t"] for r in rs])
            for s in e["samples"]:
                if s["t"] < first_service or not s.get("ahead"):
                    continue
                k = int(np.argmin(np.abs(times - s["t"])))
                if abs(times[k] - s["t"]) > 0.5:
                    continue
                r = rs[k]
                tr = r["truth"]
                if not tr["ahead"]:
                    continue
                pos = {int(t[0]): (float(t[1]), float(t[2])) for t in r["tracks"]}
                if tr["last_along"] not in pos:
                    stats[arm]["last_untracked"] += 1
                    continue
                members = set(tr["order"])
                humans = [(i, xy, i in members) for i, xy in pos.items()]
                free, why, blockers = band(line, nav, pos[tr["last_along"]], humans, tr["last_along"])
                st = stats[arm]
                st["n"] += 1
                st["free"] += free
                st["in_line"] += bool(s["in_line"])
                strict = bool(s["in_line"]) and s.get("side", 9) <= 0.6
                st["strict"] += strict
                if free:
                    st["acc"] += bool(s["in_line"])
                    st["acc_strict"] += strict
                    continue
                # fallback: behind the last, no cut-in, heading, clearance, 0.5-1.2 m behind the blocker nearest the last
                rx, ry = s["robot"][0], s["robot"][1]
                ra = line.project(rx, ry)[0]
                la = line.project(*pos[tr["last_along"]])[0]
                bl = [b for b in blockers if b in pos]
                ok = False
                if bl and ra >= la and not s["cut_in"] and s.get("heading_off_deg", 180) <= HEADING \
                        and s.get("nearest_m", 0) >= CLEAR:
                    b = min(bl, key=lambda i: math.dist(pos[i], pos[tr["last_along"]]))
                    gap = math.dist((rx, ry), pos[b])
                    ok = ra >= line.project(*pos[b])[0] and BAND[0] <= gap <= BAND[1]
                st["fallback"] += ok
                st["acc"] += ok or bool(s["in_line"])         # in_line kept, fallback added
                st["acc_strict"] += ok or strict
                st["acc_draft"] += ok                         # the draft as written: in_line not counted when blocked
                if ok:
                    ahead_of = [m for m in tr["ahead"] if m in pos and line.project(*pos[m])[0] > ra]
                    if ahead_of:
                        st["fallback_ahead_of_waiter"] += 1
                        ahead_cases.append((scen, arm, s["t"], ahead_of))
                    if any(b in members for b in bl):
                        st["fallback_behind_member_blocker"] += 1
                st[f"blocked_{why}"] += 1
                if s["in_line"]:
                    st["blocked_but_in_line"] += 1
                    st["blocked_but_strict"] += strict
            stats[arm]["episodes"] += 1
    print("\nPart B: closed loop, spent DeepSeek sweep, samples with someone ahead from the first service after "
          "delivery.\n")
    print("| arm | episodes | samples | band free | in_line | strict | fallback credited | acceptable (in_line or fb) | "
          "acceptable (strict or fb) | blocked but in_line | blocked but strict | fb ahead of a waiter |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        st = stats[arm]
        n = st["n"] or 1
        print(f"| {arm} | {st['episodes']} | {st['n']} | {st['free'] / n:.3f} | {st['in_line'] / n:.3f} | "
              f"{st['strict'] / n:.3f} | {st['fallback'] / n:.3f} | {st['acc'] / n:.3f} | {st['acc_strict'] / n:.3f} | "
              f"{st['blocked_but_in_line']} | {st['blocked_but_strict']} | {st['fallback_ahead_of_waiter']} |")
    print("\nBlocked reasons and other counts by arm:")
    for arm in arms:
        st = stats[arm]
        print(f"  {arm}: " + ", ".join(f"{k} {v}" for k, v in sorted(st.items()) if k.startswith("blocked_")
                                       or k in ("last_untracked", "fallback_behind_member_blocker")))
    if ahead_cases:
        print("\nFallback samples with the robot ahead of a waiter (scene, arm, t, waiters):")
        for c in ahead_cases[:20]:
            print("  ", c)


def part_c(dump_root: Path, scen_dir: Path, names):
    """How much room the band has (robot idle dumps): per 1-s frame with someone in line, from the first service on,
    the best clearance (to the nearest human) over band points on the floor, and the area of the free band."""
    rows = collections.defaultdict(list)
    for scen in names:
        files = sorted((dump_root / scen).glob("seed*.json"))
        if not files:
            continue
        line, nav = world(scen_dir / f"{scen}.yaml")
        for f in files:
            d = json.loads(f.read_text())
            prev_front, served = None, False
            for fr in d["frames"]:
                if abs(fr["t"] - round(fr["t"])) > 1e-6:
                    continue
                q = [p for p in fr["people"] if p["in_queue"]]
                if not q:
                    continue
                front = min(q, key=lambda p: p["order"])["id"]
                served = served or (prev_front is not None and front != prev_front)
                prev_front = front
                if not served:
                    continue
                last = max(q, key=lambda p: line.project(*p["xy"])[0])
                lx, ly = last["xy"]
                la = line.project(lx, ly)[0]
                u = back_unit(line, lx, ly)
                best, free = 0.0, 0
                for dx, dy in GRID:
                    x, y = lx + dx, ly + dy
                    if line.project(x, y)[0] < la or abs(dx * u[1] - dy * u[0]) > CORRIDOR or not nav(x, y):
                        continue
                    c = min(math.dist((x, y), p["xy"]) for p in fr["people"])
                    best = max(best, c)
                    free += c >= CLEAR
                rows[scen].append((best, free * STEP * STEP))
    print("Part C: room in the band from the first service on (robot idle). best = the largest clearance to the")
    print("nearest human over band points; area = free band area (m^2).\n")
    print("| scene | frames | best < 0.55 | best 0.55-0.70 | best 0.70-0.85 | best >= 0.85 | median area |")
    print("|---|---|---|---|---|---|---|")
    allr = []
    for scen, r in rows.items():
        allr += r
    for scen, r in list(rows.items()) + [("ALL", allr)]:
        n = len(r)
        b = [x[0] for x in r]
        a = sorted(x[1] for x in r)
        print(f"| {scen} | {n} | {sum(x < 0.55 for x in b) / n:.3f} | {sum(0.55 <= x < 0.70 for x in b) / n:.3f} | "
              f"{sum(0.70 <= x < 0.85 for x in b) / n:.3f} | {sum(x >= 0.85 for x in b) / n:.3f} | {a[n // 2]:.2f} |")


if __name__ == "__main__":
    mode, root, scen_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    names = sys.argv[4:] or sorted(p.name for p in root.iterdir() if p.is_dir())
    {"A": part_a, "B": part_b, "C": part_c}[mode](root, scen_dir, names)
