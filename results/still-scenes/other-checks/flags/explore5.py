import sys, json, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
for run in H.runs(stage='s20', model='dsv41', method='con'):
    c=collections.Counter()
    for r in H.records(run):
        for f,v in r.result['flags'].items():
            if v and f!='standard_goal_frontal': c[(f,bool(r.score['success']))]+=1
        if r.result.get('source')=='standard' and not r.result['flags'].get('standard_goal_frontal'):
            c[('std_fallback',bool(r.score['success']))]+=1
    print(run.arm, dict(c))
