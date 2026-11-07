import sys, collections
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
for st in ('s16b','s19','s21','s22','s23'):
    for ref in H.runs(stage=st):
        if ref.arm not in ('real','staffed','plain'): continue
        tally = collections.Counter()
        recs = list(H.records(ref, tally=tally))
        stages = collections.Counter(); cached = collections.Counter(); src = collections.Counter()
        trials = collections.Counter(r.trial for r in recs)
        for r in recs:
            src[(r.result.get('source'), tuple(sorted(r.timings)))] += 1
            for s, t in r.timings.items():
                stages[s] += t['calls']; cached[s] += t['cached']
        print(ref.name, dict(tally), 'trials', dict(trials))
        print('   calls', dict(stages), 'cached', dict(cached))
        print('   src', dict(src))
