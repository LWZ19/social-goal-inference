import sys, json
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
rs = H.runs(stage='s19', model='glm53', method='con')
for r in rs: print(r.name, r.arm, r.stage)
r = [x for x in rs if x.arm=='real'][0]
for rec in H.records(r):
    if rec.situation=='waiting_line' and rec.solver_step:
        print(rec.scene_id, rec.episode_id, rec.trial, rec.labels)
        print('measures', json.dumps(rec.measures, default=str)[:3000])
        print('score', json.dumps(rec.score, default=str)[:3000])
        res = rec.result
        print('result keys', list(res.keys()))
        print('people', json.dumps(res['people'], default=str)[:1500])
        print('constraints', json.dumps(res['constraints'], default=str)[:1500])
        print('target', json.dumps(res['target'], default=str)[:800])
        for c in res.get('calls') or []:
            print('call', c.get('stage'), list(c.keys()), c.get('cached'), (c.get('usage')))
            if c.get('stage')=='constraints': print('RAW', c.get('raw_text')[:2000])
        break
