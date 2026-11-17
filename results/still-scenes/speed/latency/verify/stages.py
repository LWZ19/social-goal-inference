import sys, collections
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
for st in ("s16b","s19","s21","s22","s23","s20"):
    for ref in H.runs(stage=st):
        if ref.arm not in ("real","staffed","plain"): continue
        n=0; stages=collections.Counter(); cached=collections.Counter(); sources=collections.Counter(); tk=collections.Counter()
        for rec in H.records(ref):
            n+=1
            sources[rec.result.get("source")]+=1
            for k in rec.timings_s: tk[k]+=1
            for c in rec.result.get("calls") or []:
                stages[c["stage"]]+=1; cached[c["stage"]]+=bool(c.get("cached"))
        print(ref.folder, n, dict(stages), "cached", dict(cached), dict(sources), dict(tk))
