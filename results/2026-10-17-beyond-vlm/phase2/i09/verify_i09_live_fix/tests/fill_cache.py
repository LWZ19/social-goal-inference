"""Fill a cache of my own with plan/locate/context/constraints answers under the FAKE gateway's provider keys (the live
keys), using the FIRST build's code (before_fix_e6e73181/driver.py), which has no upstream guard.

    fill_cache.py OUT_ROOT CACHE_DIR [model ...]

How: the first build's source is read and exec'd in a fresh module whose __file__ is LIVE/driver.py (so its
HERE-relative paths resolve; nothing is written into LIVE except __pycache__ by imports). Its run_arm is called in
live mode for every arm with backend_factory = a fake gateway (provider = the entry's own) and a stand-in ledger that
counts but never refuses (the first build's 60-per-model cap would stop a 84-record fill). _forbid_keys is applied, so
no key can be read and no real back end built. The audit hook of audit.py is on.
"""
import collections
import json
import sys
import types
from pathlib import Path

V = Path(__file__).resolve().parent
sys.path.insert(0, str(V))
import audit  # noqa: E402

AUD = audit.install("fill")
LIVE = Path("results/2026-10-17-beyond-vlm/phase2/i09/live")
OLD = LIVE / "before_fix_e6e73181" / "driver.py"

out_root, cache_dir = Path(sys.argv[1]), Path(sys.argv[2])
models = sys.argv[3:] or ["glm53", "dsv41"]
out_root.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

src = OLD.read_text(encoding="utf-8")
M = types.ModuleType("i09_first_build")
M.__file__ = str(LIVE / "driver.py")
sys.modules["i09_first_build"] = M
exec(compile(src, str(OLD), "exec"), M.__dict__)
assert M.HERE == LIVE, M.HERE

import json as _json  # noqa: E402
from sgi.types import Usage  # noqa: E402
from sgi.vlm.base import BackendResult  # noqa: E402
from sgi.vlm.mock import centre_responder  # noqa: E402

CALLS = collections.Counter()


class FakeGateway:
    def __init__(self, entry):
        self.provider = entry.provider

    def call(self, entry, system, parts, json_schema):
        CALLS[(entry.model, json_schema.get("title"))] += 1
        return BackendResult(_json.dumps(centre_responder(parts, json_schema)), "ok", Usage())


class CountingLedger:
    def __init__(self):
        self.reserved, self.notes = 0, collections.Counter()

    def reserve(self, info):
        self.reserved += 1

    def note(self, event, info):
        self.notes[event] += 1


M._forbid_keys("verifier fill")
M.sgi_run._given_inferrer = M._given_inferrer_detector
table = M.detector_table()
summary = {}
for model in models:
    for arm in M.ARMS:
        tally, ledger = M.Tally(), CountingLedger()
        entry = M.run_arm(model, arm, "live", out_root, cache_dir, ledger, table, tally, prefix="fill_",
                          backend_factory=lambda e: FakeGateway(e))
        summary[f"{model}:{arm}"] = {"records": entry["records"], "cache": entry["cache"],
                                     "reserved": ledger.reserved, "vlm_errors": entry["vlm_errors"],
                                     "run_folder": entry["run_folder"]}
        print(model, arm, summary[f"{model}:{arm}"], flush=True)
summary["fake_calls_by_model_title"] = {f"{m}|{t}": n for (m, t), n in CALLS.items()}
summary["cache_files"] = len(list(cache_dir.glob("*.json")))
summary["audit"] = audit.summary()
(V / f"fill_summary_{'_'.join(models)}.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
