import sys, json, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
for st in ("s16b","s19","s21","s23"):
    rs = H.runs(stage=st)
    print(st, len(rs), sorted({(r.arm, r.model, r.method) for r in rs})[:60])
r = H.runs(stage="s19", arm="real", model="qwen38_low", method="con")
print(r)
recs = list(H.records(r[0]))
rec = recs[0]
print(rec.scene_id, rec.episode_id, rec.trial, rec.situation)
print("result keys", list(rec.result.keys()))
print("raw keys", list(rec.raw.keys()))
print("measures", rec.measures)
print("timings_s", rec.timings_s)
for c in rec.result.get("calls") or []:
    print({k: (v if k not in ("raw_text","prompt","messages") else str(v)[:60]) for k, v in c.items()})
print(json.dumps(rec.timings, indent=0)[:1500])
