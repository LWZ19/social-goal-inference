"""The ladder's last rungs with the gated anchor (only named people skip the join gate). Zero calls."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ladder import report, run  # noqa: E402
from replay_lib import CT3  # noqa: E402
from variants import CT3Fix, CT3FixGated  # noqa: E402

RUNGS = [("CT3-fix", CT3Fix, CT3, "photo"), ("CT3-fix gated", CT3FixGated, CT3, "photo")]
if __name__ == "__main__":
    agg, drops = run(rungs=RUNGS, stress=sys.argv[1] if len(sys.argv) > 1 else None)
    report(agg, drops, [r[0] for r in RUNGS])
