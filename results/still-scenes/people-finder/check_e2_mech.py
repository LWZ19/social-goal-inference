"""E2 mechanical checks of one model's run folder (controller; no score, pose or success field is read).

Units (160 expected, no duplicates), vlm_error, plan status, people-matter count, arm statuses, calls per arm and per
stage, call statuses, retries, cached calls, the rotation of orders, and the run meta (git HEAD, empty diff).

Usage: python check_e2_mech.py RUN_FOLDER
"""
import collections
import json
import pathlib
import sys

folder = pathlib.Path(sys.argv[1])
meta = json.loads((folder / "run_meta.json").read_text())
print(f"meta: model {meta['model']}, git {meta['git_head'][:7]}, diff {meta['diff_sha256'][:8]}, "
      f"people {meta['people_sha256'][:8]}, dry {meta['dry']}")
units, keys = [], collections.Counter()
for f in sorted(folder.glob("records_*.jsonl")):
    for line in f.read_text().splitlines():
        u = json.loads(line)
        units.append(u)
        keys[(u["scene"], u["episode"], u["t"])] += 1
dup = [k for k, n in keys.items() if n > 1]
print(f"units {len(units)}, distinct {len(keys)}, duplicates {len(dup)}")
print("vlm_error:", sum(bool(u["vlm_error"]) for u in units), "| plan status:", dict(collections.Counter(u["plan_status"] for u in units)))
pm = [u for u in units if u["people_matter"]]
print("people-matter units:", len(pm))
arm_status, arm_calls, call_status, stages = (collections.Counter() for _ in range(4))
retries, cached = collections.Counter(), collections.Counter()
for u in pm:
    for arm, d in u["arms"].items():
        arm_status[(arm, d.get("status"))] += 1
        calls = d.get("calls") or []
        arm_calls[(arm, len(calls))] += 1
        for c in calls:
            call_status[c.get("status")] += 1
            stages[(arm, c.get("stage"))] += 1
            retries["transient"] += c.get("transient_retries") or 0
            retries["validation"] += c.get("validation_retries") or 0
            cached[bool(c.get("cached"))] += 1
print("arm status:", dict(sorted(arm_status.items())))
print("calls listed per arm (incl. shared plan and context):", dict(sorted(arm_calls.items())))
print("calls by arm and stage:", dict(sorted(stages.items())))
print("call status:", dict(call_status), "| retries:", dict(retries), "| cached:", dict(cached))
print("orders:", dict(collections.Counter(tuple(u["order"]) for u in pm)))
print("n_calls per unit:", dict(collections.Counter(u["n_calls"] for u in units)))
