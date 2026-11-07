import sys, collections, json
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
r = H.runs(stage='s19', arm='real', model='glm53', method='con')[0]
recs = [x for x in H.records(r) if x.solver is not None]
x = recs[0]
print(json.dumps(x.constraints, indent=0)[:1200])
print(json.dumps(x.candidates[:2], indent=0)[:1500])
print(json.dumps(x.target, indent=0)[:800])
print(json.dumps(x.solver, indent=0)[:600])
print('people', json.dumps(x.people[:1], indent=0)[:600])
for st in ('s19','s21','s23'):
    for rr in H.runs(stage=st, method='con'):
        if rr.arm in ('real','staffed','plain'):
            n = list(H.records(rr))
            print(st, rr.arm, rr.model, len(n), sum(bool(y.score.get('success')) for y in n), sum(y.solver is not None for y in n))
