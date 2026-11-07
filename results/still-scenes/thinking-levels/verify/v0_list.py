import sys
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
for st in ['s8','s10','s19','s21']:
    for r in H.runs(stage=st):
        if 'qwen' in r.model or st in ('s8','s10'):
            print(st, r.folder, r.arm, r.model, r.method)
