import sys, json
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
rs = H.runs(stage='s19', arm='real', model='glm53', method='con')
recs = list(H.records(rs[0]))
r = next(x for x in recs if x.situation=='waiting_line' and x.solver_step)
for c in r.result['calls']:
    d = {k:(v if k!='raw_text' else None) for k,v in c.items()}
    print(json.dumps(d)[:800])
    print('RAW', repr(c.get('raw_text'))[:1500])
print('PEOPLE', json.dumps(r.people)[:1500])
print('CONS', json.dumps(r.constraints)[:2000])
print('SOLVER', json.dumps(r.solver))
g = next(x for x in recs if x.situation=='conversation_group' and x.solver_step)
print('GROUP CONS', json.dumps(g.constraints)[:2000])
sc = H.scene(r.run, r.scene_id)
print(type(sc.truth), [a for a in dir(sc.truth) if not a.startswith('_')])
print(sc.truth.people[0])
print(sc.truth.episodes[r.episode_id])
print(sc.truth.episodes[g.episode_id].group_at_target if g.scene_id==r.scene_id else None)
sg = H.scene(g.run, g.scene_id)
print('G ep', sg.truth.episodes[g.episode_id])
print(H.recorded_config(r.ref)[1].thresholds)
