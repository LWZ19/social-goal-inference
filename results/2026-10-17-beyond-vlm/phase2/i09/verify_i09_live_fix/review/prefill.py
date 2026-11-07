"""Fill a cache of the reviewer's own with fake-gateway answers under the live keys, for 3 real-arm scenes and both
models, then remove the constraints entries: plan and context hit, constraints miss, as count-only expects on the
main cache. Zero model calls: the back end is an in-process fake that claims the entry's provider.

The driver's own run_arm is used with mode "fill", which no command line can give: build_clients then takes the
backend_factory branch and DriverClient neither counts-only nor refuses upstream sends. Only for building the cache.
"""
import json
import shutil
import sys
from pathlib import Path

REVIEW = Path(__file__).resolve().parent
sys.path.insert(0, str(REVIEW))
import audit  # noqa: E402,F401

LIVE = REVIEW.parents[1] / "live"
sys.path.insert(0, str(LIVE))
import driver as D  # noqa: E402
from sgi.types import Usage  # noqa: E402
from sgi.vlm.base import BackendResult  # noqa: E402
from sgi.vlm.mock import centre_responder  # noqa: E402

ONLY = ["real_eval_08_waiting_line_4", "real_eval_11_conversation_group_3", "real_eval_01_no_one_place"]
FILL = REVIEW / "fill"
if FILL.exists():
    shutil.rmtree(FILL)
cache = FILL / "cache_full"
cache.mkdir(parents=True)


class Fake:
    def __init__(self, provider):
        self.provider = provider
        self.calls = []

    def call(self, entry, system, parts, json_schema):
        self.calls.append(json_schema.get("title"))
        return BackendResult(json.dumps(centre_responder(parts, json_schema)), "ok", Usage())


D.sgi_run._given_inferrer = D._given_inferrer_detector          # as main() does
table = D.detector_table()
summary = {}
removed = {}
for model in ("glm53", "dsv41"):
    fakes = []

    def factory(entry):
        fakes.append(Fake(entry.provider))
        return fakes[-1]

    ledger = D.Ledger(FILL / "ledger_fill.jsonl", model, 120, 60)
    tally = D.Tally()
    entry = D.run_arm(model, "real", "fill", FILL / "runs", cache, ledger, table, tally, only=ONLY,
                      backend_factory=factory)
    keys = sorted({r["key"] for r in tally.rows if r["stage"] == "constraints" and r["outcome"] == "sent"})
    removed[model] = keys
    summary[model] = {"cache": entry["cache"], "records": entry["records"], "fake_calls": sum(len(f.calls) for f in fakes),
                      "providers": sorted({f.provider for f in fakes}), "constraints_keys": len(keys)}

cache_cut = FILL / "cache_cut"
shutil.copytree(cache, cache_cut)
for model, keys in removed.items():
    for key in keys:
        (cache_cut / f"{key}.json").unlink()
(FILL / "removed_keys.json").write_text(json.dumps(removed, indent=1))
(FILL / "prefill_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
print("cache_full", len(list(cache.glob("*.json"))), "cache_cut", len(list(cache_cut.glob("*.json"))))
