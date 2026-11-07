"""The builder's dryrun.py parts c and b2 (hashed in BUILD.sha256 beside driver 84bab2e4), run unchanged except that
their output folder OUT is redirected into review/dryrun_rerun/. Shows which of the first build's checks still run
against the current driver. Zero calls: dryrun.py's own fakes, its _forbid_keys, plus the reviewer's audit hook."""
import json
import sys
import traceback
from pathlib import Path

REVIEW = Path(__file__).resolve().parent
sys.path.insert(0, str(REVIEW))
import audit  # noqa: E402

LIVE = REVIEW.parents[1] / "live"
sys.path.insert(0, str(LIVE))
import dryrun as DR  # noqa: E402

DR.OUT = REVIEW / "dryrun_rerun"
DR.OUT.mkdir(exist_ok=True)
summary = {}
for name in ("c", "b2"):
    DR.RESULTS.clear()
    try:
        getattr(DR, f"part_{name}")()
        crashed = None
    except BaseException as exc:                    # noqa: BLE001  (report whatever stops the part)
        crashed = f"{type(exc).__name__}: {' '.join(str(exc).split())[:220]}"
        traceback.print_exc()
    summary[name] = {"checks": [[n, ok] for n, ok, _ in DR.RESULTS], "crashed": crashed}
summary["audit"] = {"env_hits": len(audit.HITS["env"]), "main_cache_hits": len(audit.HITS["main_cache"])}
(REVIEW / "dryrun_rerun.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
