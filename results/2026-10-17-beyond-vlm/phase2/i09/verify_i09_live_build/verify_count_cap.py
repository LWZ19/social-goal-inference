"""Verifier: summarise my count-only runs (empty cache of my own) and my cap runs (fake gateway, default caps)."""
import collections
import json
from pathlib import Path

V = Path(__file__).resolve().parent
home = str(Path.home())
res = {}


def allowed(writes):
    return [w for w in writes if not (w.startswith(str(V)) or w.startswith(home + "/.cache/")
                                      or w.startswith(home + "/.config/matplotlib") or w.startswith("/tmp/"))]


# (b) count-only
for m in ("glm53", "dsv41"):
    w = json.loads((V / f"wrap_b_count_{m}.json").read_text())
    rep = sorted((V / "b_count").glob(f"count_only_{m}_*.json"))[-1]
    r = json.loads(rep.read_text())
    rows = [json.loads(l) for l in rep.with_suffix(".jsonl").read_text().splitlines()]
    res[f"b_{m}"] = {"exit": w["exit"], "records": sum(a["records"] for a in r["arms"]),
                     "cache_total": r["cache_total"], "bounds": r["bounds"], "rows": len(rows),
                     "outcomes": dict(collections.Counter(x["outcome"] for x in rows)),
                     "fake_calls": len(w["fake_calls"]), "blocked": w["audit"]["blocked"],
                     "writes_outside": allowed(w["audit"]["writes"]), "subprocess": w["audit"]["subprocess"]}
res["b_cache_files"] = len(list((V / "b_empty_cache").iterdir()))
res["b_run_folders"] = [p.name for p in (V / "b_count").iterdir() if p.is_dir()]

# (c) cap
ledger = [json.loads(l) for l in (V / "c_cap" / "ledger.jsonl").read_text().splitlines()]
sent = [x for x in ledger if x["event"] == "sent"]
res["c_ledger"] = {"sent": len(sent), "sent_by_model": dict(collections.Counter(x["model"] for x in sent)),
                   "refused": [(x["model"], x["stage"], x["total"], x["model_total"]) for x in ledger
                               if x["event"] == "refused"],
                   "events": dict(collections.Counter(x["event"] for x in ledger)),
                   "errors": [(x["model"], x.get("error")) for x in ledger if x["event"] == "error"],
                   "sent_by_stage": dict(collections.Counter(x["model"] + ":" + x["stage"] for x in sent))}
for tag in ("c1_glm_cap", "c2_dsv_cap", "c3_glm_again", "c4_raise_cap"):
    p = V / f"wrap_{tag}.json"
    if not p.exists():
        continue
    w = json.loads(p.read_text())
    res[tag] = {"exit": w["exit"], "fake_calls": len(w["fake_calls"]),
                "fake_models": dict(collections.Counter(" ".join(map(str, (c["model"], c["provider"], c["base_url"])))
                                                        for c in w["fake_calls"])),
                "blocked": w["audit"]["blocked"], "writes_outside": allowed(w["audit"]["writes"])}
folders = sorted(p for p in (V / "c_cap").iterdir() if p.is_dir())
import yaml  # noqa: E402
res["c_folders"] = {}
for f in folders:
    doc = yaml.safe_load((f / "config.yaml").read_text())
    n = len((f / "results.jsonl").read_text().splitlines()) if (f / "results.jsonl").exists() else None
    res["c_folders"][f.name] = {"incomplete": doc.get("incomplete"), "stopped_by": doc.get("stopped_by"),
                                "records": n, "driver_sha256": doc["i09_live"]["driver_sha256"][:12]}
res["c_lock_left"] = (V / "c_cap" / "ledger.lock").exists()
res["c_cache_files"] = len(list((V / "c_cache").iterdir()))
(V / "verify_count_cap.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
