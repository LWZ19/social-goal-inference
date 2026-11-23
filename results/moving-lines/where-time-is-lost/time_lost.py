"""Descriptive (the ceiling check): where strict time in line is lost, per arm. Zero calls; read-only.

    .venv/bin/python results/moving-lines/where-time-is-lost/time_lost.py SELECTION_JSON SCENARIO_DIR RUN_ROOT \
        ARM [ARM ...] [--csv OUT.csv]
    (SELECTION_JSON and OUT.csv relative to results/moving-lines/, RUN_ROOT relative to results/data/moving/,
    SCENARIO_DIR relative to the repo.)

Window: the 1 Hz samples from the first service after delivery to the end, exactly as score_sweep.episode_scores'
after_service_strict (the script checks it gets the same share).

1. Seconds. Each sample in the window is strict in line or not. A failing sample gets one primary reason, the first
   of these that holds:
   - someone ahead (the sample's `ahead` >= 1): "ahead of last" (behind_m < 0), "off axis" (side > 0.6),
     "too far" (gap_m > 1.2), "too close" (gap_m < 0.5), "heading" (> 45 deg), "crowded" (nearest_m < 0.55);
   - nobody ahead: "far from slot 0" (to_slot0_m > 0.6), "heading", "crowded".
   Each is split by the robot's state: moving (on its way to a goal) or arrived (parked).
   Also: the share of the window before its first strict sample ("arrival") and after it ("keeping").
2. Goals, at each ok re-solve in the window that logs tracks (OW logs none, so it has no row here):
   - the goal is the re-solve's pose_world; the person to stand behind is truth.last_along, at the re-solve's
     tracked position; nobody ahead means slot 0;
   - "goal right": the goal passes the sample's geometry if the robot stood on it with the goal's heading
     (behind >= 0, gap 0.5-1.2, side <= 0.6, heading <= 45; or within 0.6 m of slot 0 facing the counter);
     clearance is left out, as it depends on who walks past;
   - "robot there": the first sample at or after the re-solve's delivery is strict in line;
   - also split by right last (the solver's `last` equals truth.last and the tick is not vacated);
   - the goal's error from the ideal place (0.85 m behind the person on the axis): along (+ = too far back) and side.
"""
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
MOVING = paths.MOVING_RUNS                # run folders are named relative to it, e.g. "runs_confirm4/dsv41"
REPO = paths.REPO
sys.path.insert(0, str(paths.DESIGN))
sys.path.insert(0, str(paths.MOVING / "test-layouts"))
from score_sweep import episode_scores  # noqa: E402
from summarise_fresh import in_line  # noqa: E402

BAND, SIDE, HEAD, CLEAR, SLOT0, IDEAL = (0.5, 1.2), 0.6, 45.0, 0.55, 0.6, 0.85
REASONS = ("ahead of last", "off axis", "too far", "too close", "heading", "crowded", "far from slot 0")


class Line:
    """Copied from sim/sgi_habitat/experiment.py (Line), so this script needs no simulator imports."""

    def __init__(self, slots):
        self.p = np.array([[s["x"], s["y"]] for s in slots], dtype=float)
        seg = np.diff(self.p, axis=0)
        self.length = np.hypot(seg[:, 0], seg[:, 1])
        self.unit = seg / self.length[:, None]
        self.start = np.concatenate([[0.0], np.cumsum(self.length)[:-1]])
        self.slot0, self.face0 = slots[0], slots[0]["face"]

    def project(self, x, y):
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


def off(a, b):
    return abs(math.degrees((a - b + math.pi) % (2 * math.pi) - math.pi))


def window(e):
    """(start time, samples) as episode_scores' after_service_strict, or (None, [])."""
    rs = [r for r in e.get("resolves", []) if r.get("truth") is not None]
    for a, b in zip(rs, rs[1:]):
        if a["truth"]["order"] and a["truth"]["order"][0] not in b["truth"]["order"]:
            later = [x for x in e["samples"] if x["t"] >= b["sent_t"] - 1e-9]
            return (b["sent_t"], later) if later else (None, [])
    return None, []


def reason(x):
    if x["ahead"] >= 1:
        checks = (("ahead of last", x.get("behind_m", 0) < 0), ("off axis", x["side"] > SIDE),
                  ("too far", x.get("gap_m", 0) > BAND[1]), ("too close", x.get("gap_m", 99) < BAND[0]),
                  ("heading", x.get("heading_off_deg", 0) > HEAD), ("crowded", x["nearest_m"] < CLEAR))
    else:
        checks = (("far from slot 0", (x.get("to_slot0_m") or 99) > SLOT0),
                  ("heading", x.get("heading_off_deg", 0) > HEAD), ("crowded", x["nearest_m"] < CLEAR))
    return next((k for k, ok in checks if ok), "other")


def goal_right(line, pose, person):
    gx, gy, gyaw = pose
    ga, gs, up = line.project(gx, gy)
    if person is None:
        s0 = line.slot0
        face = math.atan2(line.face0[1] - gy, line.face0[0] - gx)
        return math.hypot(gx - s0["x"], gy - s0["y"]) <= SLOT0 and off(gyaw, face) <= HEAD, None, gs
    pa = line.project(*person)[0]
    gap = math.hypot(gx - person[0], gy - person[1])
    ok = ga - pa >= 0 and BAND[0] <= gap <= BAND[1] and gs <= SIDE and off(gyaw, up) <= HEAD
    return ok, ga - (pa + IDEAL), gs


def episode(e, line):
    start, win = window(e)
    out = {"strict": None, "n": len(win)}
    if not win:
        return out
    ok = [in_line(x, "strict") for x in win]
    out["strict"] = sum(ok) / len(ok)
    ref = episode_scores(e)["after_service_strict"]
    assert ref is None or abs(ref - out["strict"]) < 1e-9, (ref, out["strict"])
    first = next((i for i, v in enumerate(ok) if v), len(ok))
    out["arrival"] = first / len(ok)
    out["keeping_lost"] = sum(1 for v in ok[first:] if not v) / len(ok)
    out["front"] = (sum(1 for x in win if x["ahead"] == 0), sum(v for x, v in zip(win, ok) if x["ahead"] == 0))
    out["fails"] = {}
    for x, v in zip(win, ok):
        if not v:
            k = (reason(x), "moving" if x["state"] != "arrived" else "arrived")
            out["fails"][k] = out["fails"].get(k, 0) + 1
    st = [x["t"] for x in win]
    goals = []
    for r in e["resolves"]:
        if (r.get("status") != "ok" or r.get("truth") is None or not r.get("tracks") or r.get("pose_world") is None
                or r["sent_t"] < start - 1e-9):
            continue
        tr = r["truth"]
        pos = {int(i): (x, y) for i, x, y in r["tracks"]}
        la = tr.get("last_along")
        if la is not None and la not in pos:
            continue
        g, ea, gs = goal_right(line, r["pose_world"], None if la is None else pos[la])
        j = next((i for i, t in enumerate(st) if t >= r["delivered_t"] - 1e-9), None)
        if j is None:
            continue
        right = bool(tr["ahead"]) and not r.get("vacated") and r.get("last") == tr["last"]
        goals.append({"goal": g, "there": ok[j], "right_last": right, "someone": bool(tr["ahead"]),
                      "e_along": ea, "side": gs})
    out["goals"] = goals
    return out


def pct(a, b):
    return f"{a / b:.3f}" if b else "-"


def report(arm, eps):
    have = [x for x in eps if x["strict"] is not None]
    secs = sum(x["n"] for x in have)
    print(f"\n### {arm}: {len(have)} episodes with a window, {secs} window seconds\n")
    print(f"- strict share, mean over episodes: {statistics.mean(x['strict'] for x in have):.3f}"
          if have else "- no windows")
    if not have:
        return
    print(f"- lost before the first strict second (arrival), mean: {statistics.mean(x['arrival'] for x in have):.3f};"
          f" lost after it (keeping): {statistics.mean(x['keeping_lost'] for x in have):.3f}")
    fs, fp = sum(x["front"][0] for x in have), sum(x["front"][1] for x in have)
    bp = sum(round(x["strict"] * x["n"]) for x in have) - fp
    print(f"- window seconds with nobody ahead (the robot first; the sim never serves it): {pct(fs, secs)}, strict "
          f"there {pct(fp, fs)}; with someone ahead: strict {pct(bp, secs - fs)}")
    tot = {}
    for x in have:
        for k, v in x["fails"].items():
            tot[k] = tot.get(k, 0) + v
    print("\n| primary reason | moving | arrived | share of all window seconds |")
    print("|---|---|---|---|")
    for rsn in REASONS + ("other",):
        m, a = tot.get((rsn, "moving"), 0), tot.get((rsn, "arrived"), 0)
        if m or a:
            print(f"| {rsn} | {m} | {a} | {pct(m + a, secs)} |")
    gs = [g for x in have for g in x.get("goals", [])]
    if not gs:
        print("\n(no re-solves with tracks: goals not checked)")
        return
    print(f"\nGoals at {len(gs)} re-solves in the window:\n")
    print("| | re-solves | goal right | robot there | there, goal right | there, goal wrong |")
    print("|---|---|---|---|---|---|")
    for name, sub in (("all", gs), ("someone ahead, right last", [g for g in gs if g["someone"] and g["right_last"]]),
                      ("someone ahead, wrong last", [g for g in gs if g["someone"] and not g["right_last"]]),
                      ("nobody ahead", [g for g in gs if not g["someone"]])):
        n = len(sub)
        gr = [g for g in sub if g["goal"]]
        gw = [g for g in sub if not g["goal"]]
        print(f"| {name} | {n} | {pct(len(gr), n)} | {pct(sum(g['there'] for g in sub), n)} | "
              f"{pct(sum(g['there'] for g in gr), len(gr))} | {pct(sum(g['there'] for g in gw), len(gw))} |")
    ea = [g["e_along"] for g in gs if g["someone"] and g["right_last"]]
    sd = [g["side"] for g in gs if g["someone"] and g["right_last"]]
    if ea:
        q = statistics.quantiles(ea, n=4)
        print(f"\nRight last, someone ahead: goal along error from the ideal place (+ = farther back) median "
              f"{statistics.median(ea):+.2f} m, quartiles {q[0]:+.2f} / {q[2]:+.2f}; |error| > 0.35 m "
              f"{pct(sum(abs(v) > 0.35 for v in ea), len(ea))}; goal side > 0.6 m {pct(sum(v > SIDE for v in sd), len(sd))}")


def main():
    args = sys.argv[1:]
    csv = None
    if "--csv" in args:
        i = args.index("--csv")
        csv = args[i + 1]
        args = args[:i] + args[i + 2:]
    sel_path, scen_dir, root, arms = args[0], args[1], args[2], args[3:]
    sel = json.loads((paths.MOVING / sel_path).read_text())["layouts"]
    print(f"# Where strict time in line is lost ({sel_path}, {root})")
    rows = []
    for arm in arms:
        eps = []
        for lay in sel:
            hits = sorted((MOVING / root).glob(f"**/{lay['name']}/seed{lay['seed']}/{arm}/episode.json"))
            if not hits:
                continue
            e = json.loads(hits[0].read_text())
            scen = yaml.safe_load((REPO / scen_dir / f"{lay['name']}.yaml").read_text())
            (q,) = scen["queues"].values()
            x = episode(e, Line(q["slots"]))
            eps.append(x)
            gs = x.get("goals") or []
            rows.append((lay["name"], arm, x["strict"], x.get("arrival"), x.get("keeping_lost"), len(gs),
                         sum(g["goal"] for g in gs) / len(gs) if gs else None))
        report(arm, eps)
    if csv:
        with open(paths.MOVING / csv, "w") as f:
            f.write("layout,arm,strict,arrival_lost,keeping_lost,goal_resolves,goal_right\n")
            for r in rows:
                f.write(",".join("" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r) + "\n")


if __name__ == "__main__":
    main()
