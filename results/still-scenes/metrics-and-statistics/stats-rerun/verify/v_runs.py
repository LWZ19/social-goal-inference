import sys, collections
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
names = collections.Counter()
for st in ('s16b','s19','s20','s21','s23'):
    for r in H.runs(stage=st):
        if r.arm in ('real','staffed','plain'):
            names[r.name]+=1
            print(st, r.folder, r.arm, r.model, r.method)
print('dup names:', [n for n,c in names.items() if c>1])
