import sys
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
for st in ('s16b','s19','s20','s21','s23'):
    for r in H.runs(stage=st):
        print(st, r.folder, r.arm, r.model, r.method)
