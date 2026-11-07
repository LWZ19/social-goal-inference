"""Round 7, POST HOC DESCRIPTION (after e_ct3ft_out.txt): CT3F and CT3FT at re-solves with nobody ahead (the robot
first): the share where the arm knows nobody is ahead (`last` None), and its goal's distance to slot 0. Zero calls.

    .venv/bin/python round7/f_front.py
"""
import json
import math
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from a_loss import MOVING, REPO, Line, window  # noqa: E402

sel = json.loads((paths.MOVING / "test-layouts/confirm_selection.json").read_text())["layouts"]
for arm, root in (("CT3FT", "runs_ceiling4/dsv41"), ("CT3F", "runs_confirm4/dsv41")):
    d0, none, n = [], 0, 0
    for lay in sel:
        hits = sorted((MOVING / root).glob(f"**/{lay['name']}/seed{lay['seed']}/{arm}/episode.json"))
        if not hits:
            continue
        e = json.loads(hits[0].read_text())
        (q,) = yaml.safe_load((REPO / "sim/scenarios/confirm4" / f"{lay['name']}.yaml").read_text())["queues"].values()
        line = Line(q["slots"])
        start, _ = window(e)
        if start is None:
            continue
        for r in e["resolves"]:
            if (r.get("status") != "ok" or r.get("truth") is None or r.get("pose_world") is None
                    or r["sent_t"] < start - 1e-9 or r["truth"]["ahead"]):
                continue
            n += 1
            none += r.get("last") is None
            gx, gy, _ = r["pose_world"]
            d0.append(math.hypot(gx - line.slot0["x"], gy - line.slot0["y"]))
    q = statistics.quantiles(d0, n=4)
    print(f"{arm}: {n} re-solves with nobody ahead; knows nobody is ahead {none / n:.3f}; goal to slot 0 median "
          f"{statistics.median(d0):.2f} m, quartiles {q[0]:.2f} / {q[2]:.2f}; within 0.6 m "
          f"{sum(v <= 0.6 for v in d0) / len(d0):.3f}")
