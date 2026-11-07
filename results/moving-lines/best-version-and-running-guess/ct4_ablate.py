"""CT4-min: the round-2 changes one at a time against the interim version (right last, capture, recall, empty)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ladder_ct4 import run  # noqa: E402
V1 = dict(STAND_S=0, WALKERS_STATIC=False, LR_BACK=1.0, STALE_DECIDE_S=1e9)
VARIANTS = {"v1 (interim)": V1, "+walkers in static": dict(V1, WALKERS_STATIC=True),
            "+standing place 10 s": dict(V1, STAND_S=10.0), "+step back": dict(V1, LR_BACK=3.0),
            "+stale not last": dict(V1, STALE_DECIDE_S=2.0)}
for s in [None, "S3", "S4", "S1234"]:
    for name, params in VARIANTS.items():
        if s != "S4" and name == "+stale not last":
            continue
        agg, _ = run(s, params=params, only={"CT4-min"})
        row = []
        for g in sorted({g for g, _ in agg}):
            a = agg[(g, "CT4-min")]; n = a["ticks"]
            row.append(f"{g[0][:3]}-{g[1][:3]} R{a['right']/n:.2f} C{a['bys_last']*0.5:.0f} E{a['empty']/n:.2f}")
        print(f"{s or 'clean':6} {name:22} " + " | ".join(row), flush=True)
