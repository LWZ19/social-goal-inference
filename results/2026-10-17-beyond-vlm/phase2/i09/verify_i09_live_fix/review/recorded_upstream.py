"""Through the step-0 harness only: the recorded t0 records of the six D-live baseline runs, by the stages their calls
reached and each call's status. Answers: can UpstreamMiss fire in a legitimate run because a recorded upstream answer
was a vlm_error (never cached)? Zero model calls; reads nothing but what research_harness exposes."""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit  # noqa: E402,F401  (hook first)

WT = Path(".-scenes")
sys.path.insert(0, str(WT / "results/2026-10-17-beyond-vlm/phase2/i09/live/src_202c165/src"))
sys.path.insert(1, str(WT / "docs/model_notes/tools"))
import research_harness as H  # noqa: E402

MODELS = {"glm53": "s19", "dsv41": "s20"}
out = {}
for model, stage in MODELS.items():
    per = {"records": 0, "result_status": collections.Counter(), "call_status": collections.Counter(),
           "upstream_vlm_error": [], "plan_none": 0, "goal_steps": 0, "constraints_retries": collections.Counter(),
           "upstream_retries": collections.Counter(), "locate_calls": 0, "stage_sequences": collections.Counter()}
    for arm in ("real", "staffed", "plain"):
        refs = H.runs(stage=stage, arm=arm, model=model, method="con")
        assert len(refs) == 1, refs
        for rec in H.records(refs[0]):
            if rec.trial != 0:
                continue
            per["records"] += 1
            per["result_status"][rec.result.get("status")] += 1
            calls = rec.result.get("calls") or []
            per["stage_sequences"]["/".join(c["stage"] for c in calls)] += 1
            if rec.raw.get("plan") is None:
                per["plan_none"] += 1
            for c in calls:
                per["call_status"][f"{c['stage']}:{c['status']}"] += 1
                if c["stage"] == "locate":
                    per["locate_calls"] += 1
                if c["stage"] in ("plan", "locate", "context"):
                    if c["status"] == "vlm_error":
                        per["upstream_vlm_error"].append([arm, rec.scene_id, rec.episode_id, c["stage"]])
                    per["upstream_retries"][f"{c['stage']}:v{c.get('validation_retries', 0)}"
                                            f"t{c.get('transient_retries', 0)}"] += 1
                if c["stage"] == "constraints":
                    per["constraints_retries"][f"v{c.get('validation_retries', 0)}t{c.get('transient_retries', 0)}"] += 1
            if any(c["stage"] == "people" for c in calls):
                per["goal_steps"] += 1
    out[model] = {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in per.items()}

path = Path(__file__).resolve().parent / "recorded_upstream.json"
path.write_text(json.dumps(out, indent=1))
for model, per in out.items():
    print(model, "records", per["records"], "goal_steps", per["goal_steps"], "plan_none", per["plan_none"],
          "upstream_vlm_error", len(per["upstream_vlm_error"]), "locate_calls", per["locate_calls"])
    print("  result_status", per["result_status"])
    print("  call_status", per["call_status"])
    print("  upstream_retries", per["upstream_retries"])
    print("  constraints_retries", per["constraints_retries"])
    print("  sequences", per["stage_sequences"])
