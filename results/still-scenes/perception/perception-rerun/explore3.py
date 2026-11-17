import sys, json
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
rs = H.runs(stage='s23', model='glm53', method='con')
print([ (r.folder, r.arm) for r in rs])
r = rs[0]
recs = list(H.records(r))
x = recs[0]
sc = H.scene(x.ref, x.scene_id)
obs = sc.observation
print(type(obs), [a for a in dir(obs) if not a.startswith('_')])
print(type(obs.camera), [a for a in dir(obs.camera) if not a.startswith('_')])
print(obs.robot)
print(type(sc.truth.bounds), str(sc.truth.bounds)[:300])
print(type(sc.truth.obstacles), str(sc.truth.obstacles)[:300])
print([ (p.id, p.group, p.role) for p in sc.truth.people])
for k in ['s19','s20','s23']:
    print(k, sorted(set((r.arm, r.model) for r in H.runs(stage=k, method='con'))))
