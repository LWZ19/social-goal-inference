import sys, collections
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
for st in ['s8','s10']:
    for r in H.runs(stage=st):
        recs=list(H.records(r))
        n=len(recs); s=sum(1 for x in recs if x.score.get('success'))
        tr=sorted(set(x.trial for x in recs), key=lambda t:(t is None, t))
        print(r.folder, r.stage, r.arm, r.model, r.method, n, s, 'trials', tr)
for st,arm in [('s16b','ph'),('s19','ph')]:
    for r in H.runs(stage=st, arm=arm):
        recs=list(H.records(r)); n=len(recs); s=sum(1 for x in recs if x.score.get('success'))
        print(r.folder, r.stage, r.arm, r.model, r.method, n, s)
