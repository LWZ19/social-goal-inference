"""CT4-min parameter variants against the bolt-on, on 'right last', capture, member recall and empty; clean and stressed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ladder_ct4 import run  # noqa: E402
VARIANTS = {"CT4-min": {}, "no own-step credit": {"LR_STEP": 1.0}}
for s in (None, "S3", "S1234"):
    for name, params in VARIANTS.items():
        agg, drops = run(s, params=params, only={"CT4-min", "CT3-fix gated"} if name == "CT4-min" else {"CT4-min"})
        for (g, arm), a in sorted(agg.items()):
            label = arm if arm != "CT4-min" else name
            n = a["ticks"]
            print(f"{s or 'clean':6} {g[0]:13} {g[1]:9} {label:20} right {a['right']/n:.2f} empty {a['empty']/n:.2f} "
                  f"capture {a['bys_last']*0.5:5.0f} s recall {a['rec_num']/max(a['rec_den'],1):.2f}")
