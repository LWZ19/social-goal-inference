import sys, json, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
run = H.runs(stage='s20', arm='real', model='dsv41', method='con')[0]
for r in H.records(run):
    if r.result.get('source')=='constraints' and (r.result['flags'].get('constraint_unusable')):
        print(json.dumps(r.result['solver'], indent=1)[:3000])
        print(json.dumps(r.result['constraints'], indent=1)[:1500])
        print(json.dumps(r.result['candidates'][:2], indent=1)[:1500])
        break
