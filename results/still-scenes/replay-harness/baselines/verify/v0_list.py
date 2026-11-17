import sys
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
for st in ("s19", "s20", "s21", "s23", "s16b"):
    for r in H.runs(stage=st):
        print(st, r.arm, r.model, r.method, r.folder)
