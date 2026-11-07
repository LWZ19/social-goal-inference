import sys
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
for st in ('s16b','s19','s20','s21','s23'):
    for r in H.runs(stage=st):
        print(st, r.folder, r.arm, r.model, r.method)
