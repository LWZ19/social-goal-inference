import sys, collections, json
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
for st in ('s19', 's20', 's21', 's23'):
    for r in H.runs(stage=st, method='con'):
        if r.arm in ('real', 'staffed', 'plain'):
            print(st, r.arm, r.model, r.folder)
r = [x for x in H.runs(stage='s19', method='con', model='glm53', arm='real')][0]
recs = list(H.records(r))
print(len(recs))
rec = next(x for x in recs if x.solver_step and x.situation == 'waiting_line')
print(json.dumps(rec.score, indent=1, default=str)[:3000])
print(rec.people[:2]); print([c for c in rec.constraints][:2])
print(rec.result.keys())
