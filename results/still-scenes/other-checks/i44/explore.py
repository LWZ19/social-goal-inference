import sys, collections
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
for r in H.runs(stage='s8'):
    recs = list(H.records(r))
    succ = sum(1 for x in recs if x.score.get('success'))
    trials = collections.Counter(x.trial for x in recs)
    stages = collections.Counter(); cached = collections.Counter()
    for x in recs:
        for c in x.result.get('calls') or []:
            stages[c['stage']] += 1; cached[c['stage']] += bool(c.get('cached'))
    print(r.name, r.model, r.method, len(recs), 'succ', succ, dict(trials), dict(stages), 'cached', dict(cached))
x = recs[0]
print(sorted(x.raw.keys())); print(sorted(x.result.keys())); print(x.score)
print((x.result.get('calls') or [{}])[0].keys())
print((x.result.get('calls') or [{}])[0])
