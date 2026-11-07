"""CT4-min against the best bolt-ons, clean and under stress, on role-tracking measures. Zero calls.
    .venv/bin/python ladder_ct4.py [STRESS]"""
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ct4 import replay_ct4  # noqa: E402
from ladder import named_tracks, report  # noqa: E402
from replay_lib import BYSTANDER_SCENES, CT3, cells, replay, score  # noqa: E402
from sgi.resolve import Resolver  # noqa: E402
from stress import STRESS  # noqa: E402
from variants import CT3FixGated, CT3Group  # noqa: E402

def run(stress_name=None, params=None, only=None):
    stress = STRESS[stress_name] if stress_name else None
    agg = defaultdict(lambda: defaultdict(float)); drops = defaultdict(list)
    for c in cells():
        group = (c.model, "bystander" if c.scenario in BYSTANDER_SCENES else "original")
        nb = named_tracks(c) & c.bystanders
        arms = [("CT3", lambda: replay(c, CT3, stress=stress)),
                ("+group+photo", lambda: replay(c, CT3, start="photo", resolver_cls=CT3Group, stress=stress)),
                ("CT3-fix gated", lambda: replay(c, CT3, start="photo", resolver_cls=CT3FixGated, stress=stress)),
                ("CT4-min", lambda: replay_ct4(c, params=params, stress=stress))]
        for name, fn in arms:
            if only and name not in only:
                continue
            c._stale = {}
            tot, fd, dt = score(c, fn())
            a = agg[(group, name)]
            for k, v in tot.items(): a[k] += v
            a["false_drops"] += len(fd); a["cells"] += 1; a["captured_cells"] += tot["bys_last"] >= 2
            for b in nb: drops[(group, name)].append(None if b not in dt else dt[b] - c.delivered_t)
    return agg, drops

if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"# stress: {s or 'none (clean tracks)'}")
    agg, drops = run(s)
    report(agg, drops, ["CT3", "+group+photo", "CT3-fix gated", "CT4-min"])
