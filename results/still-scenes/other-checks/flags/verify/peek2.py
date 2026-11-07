import sys, json
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
r = H.runs(stage='s20', model='dsv41', method='con', arm='real')[0]
for rec in H.records(r):
    if rec.solver_step:
        c = rec.candidates
        print(type(c), len(c) if c else None)
        print(json.dumps(c[:2] if isinstance(c, list) else c, default=str)[:1500])
        print(json.dumps(rec.solver, default=str)[:800])
        break
for s, m in [('s16b','qwen38_low'),('s16b','glm53'),('s20','dsv41')]:
    for meth in ('cot','fsc','fsc3'):
        print(s, m, meth, [x.folder for x in H.runs(stage=s, model=m, method=meth) if x.arm in ('real','staffed','plain')])
