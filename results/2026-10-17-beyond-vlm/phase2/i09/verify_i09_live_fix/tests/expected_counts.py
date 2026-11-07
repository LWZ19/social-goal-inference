"""REQUEST.md's expected count-only table, re-derived from the recorded t0 records through the step-0 harness only
(as the driver reads them): per model and arm, records, records with a plan call, a context call, a people call
(= goal steps = expected constraints misses), a locate call, and constraints-call retries."""
import collections
import json
import sys
from pathlib import Path

V = Path(__file__).resolve().parent
sys.path.insert(0, str(V))
import audit  # noqa: E402

AUD = audit.install("expected_counts")
LIVE = Path("results/2026-10-17-beyond-vlm/phase2/i09/live")
sys.path.insert(0, str(LIVE / "src_202c165" / "src"))
sys.path.insert(1, "docs/model_notes/tools")
import research_harness as H  # noqa: E402

out = {}
for model, (stage, mkey) in {"glm53": ("s19", "glm53"), "dsv41": ("s20", "dsv41")}.items():
    tot = collections.Counter()
    for arm in ("real", "staffed", "plain"):
        refs = H.runs(stage=stage, arm=arm, model=mkey, method="con")
        assert len(refs) == 1, refs
        recs = [r for r in H.records(refs[0]) if r.trial == 0]
        c = collections.Counter()
        c["records"] = len(recs)
        for r in recs:
            st = [x["stage"] for x in r.result.get("calls") or []]
            for s in ("plan", "locate", "context", "people", "constraints"):
                c[f"with_{s}"] += s in st
            c["status_" + str(r.result.get("status"))] += 1
            c["constraints_retries"] += sum((x.get("validation_retries") or 0) + (x.get("transient_retries") or 0)
                                            for x in r.result.get("calls") or [] if x["stage"] == "constraints")
        out[f"{model}:{arm}"] = dict(c)
        tot.update(c)
    out[f"{model}:total"] = dict(tot)
out["audit"] = audit.summary()
(V / "expected_counts.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, v)
