"""How close the robot came to people, per arm, from the per-second samples of every fresh-seed episode (seed >= 10).

nearest_m is the floor distance from the robot's centre to the nearest person's centre (sim/sgi_habitat/experiment.py).
Below 0.55 m is SCORING.md's clearance limit; below 0.3 m the bodies would overlap (a bump). Zero calls.
    .venv/bin/python results/moving-lines/bumps/bumps.py   (reads the run folders in results/data/moving/)"""
import collections, glob, json, sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

ARMS = {"R1": "decide once, pointing", "C1": "decide once, ours", "CT": "updating v1", "CT2": "updating v2",
        "CT3F": "best version", "RTw": "follow marked spot", "Rq0": "re-ask each answer", "Cq": "re-ask every 10 s"}
agg = collections.defaultdict(collections.Counter)
seen = set()
for f in sorted(glob.glob("runs*/*/*/seed*/*/episode.json", root_dir=paths.MOVING_RUNS)):
    p = Path(f); arm, seed = p.parent.name, int(p.parent.parent.name[4:])
    key = (p.parts[1], p.parts[2], seed, arm)
    if seed < 10 or arm not in ARMS or key in seen:
        continue
    seen.add(key)
    s = json.load(open(paths.MOVING_RUNS / f))["samples"]
    c = agg[arm]; c["runs"] += 1; c["bumped_runs"] += any((x.get("nearest_m") or 9) < 0.3 for x in s)
    for i, x in enumerate(s):
        d = x.get("nearest_m")
        if d is None:
            continue
        c["secs"] += 1; c["lt55"] += d < 0.55
        if d < 0.3:
            prev = s[i - 1]["robot"] if i else x["robot"]
            moving = ((x["robot"][0] - prev[0]) ** 2 + (x["robot"][1] - prev[1]) ** 2) ** 0.5 > 0.05
            c["lt30"] += 1; c["lt30_moving"] += moving
print("| version | runs | time within 0.55 m | time within 0.3 m | runs with a moment within 0.3 m | of those seconds, robot moving |")
print("|---|---|---|---|---|---|")
for arm, name in ARMS.items():
    c = agg[arm]
    if c["runs"]:
        print(f"| {name} ({arm}) | {c['runs']} | {c['lt55']/c['secs']:.1%} | {c['lt30']/c['secs']:.2%} | "
              f"{c['bumped_runs']} ({c['bumped_runs']/c['runs']:.0%}) | {c['lt30_moving']/max(c['lt30'],1):.0%} |")
