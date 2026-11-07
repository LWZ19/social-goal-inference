import sys, json, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
r = H.runs(stage="s23", arm="real", model="haiku55", method="con")[0]
recs = list(H.records(r))
print(len(recs))
c = collections.Counter((x.situation, x.result.get("status"), x.result.get("source"), x.people is None, x.constraints is None, x.target is None) for x in recs)
for k, v in sorted(c.items(), key=str): print(k, v)
x = next(x for x in recs if x.situation == "waiting_line")
print(json.dumps({k: v for k, v in x.result.items() if k not in ("calls",)}, indent=1, default=str)[:4000])
print(x.score)
