import sys, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
for st in ['s10','s16b','s19']:
    for r in H.runs(stage=st):
        if 'qwen' not in r.model or r.arm not in ('ph','real','staffed','plain') and st!='s10': continue
        if st!='s10' and r.arm!='ph': continue
        recs = list(H.records(r))
        succ = sum(1 for x in recs if x.score.get('success'))
        trials = collections.Counter(x.trial for x in recs)
        eff = collections.Counter((c['stage'], c.get('reasoning_effort'), c.get('model')) for x in recs for c in x.result.get('calls') or [])
        print(r.folder, r.model, r.method, len(recs), 'succ', succ, dict(trials), dict(eff))
for r in H.runs(stage='s8'):
    if r.model not in ('qwen36_35','qwen38'): continue
    recs = list(H.records(r))
    eff = collections.Counter((c['stage'], c.get('reasoning_effort'), c.get('model')) for x in recs for c in x.result.get('calls') or [])
    print(r.folder, dict(eff))
    eps = collections.Counter(x.episode_id for x in recs); print(len(eps), list(eps)[:3])
